# 實驗 Protocol（凍結版 v1，2026-08-14）

> 本文件凍結主實驗的全部定義。**8/15 之後只允許修 bug，不允許改實驗定義**；
> 任何變更需在本文件留下版本記錄與理由。
> 執行細節（指令、RunPod SOP）見 `RUNPOD_SOP.md`；方法定義見 `docs/architecture.md`。

## 1. 資料集與任務序列

### 1.1 主實驗：can_dataset（4 任務序列）

來源：`/Users/aaron/research/can_dataset/tcga_{esca,lung,rcc,brca}`（CONCH l1-s256 patch 特徵，無座標）。

| 任務 | cohort | 類別 | slide 數（有特徵者） |
|---|---|---|---|
| T1 | ESCA | ESCC vs ESAD | 150 |
| T2 | LUNG | LUAD vs LUSC | 965 |
| T3 | RCC | CCRCC vs PRCC vs CHRCC（3 類） | 888 |
| T4 | BRCA | IDC vs ILC | 952 |

- **主順序**：ESCA → LUNG → RCC → BRCA（小任務起頭，與 01 討論一致）。
- **穩健性順序**：BRCA → RCC → LUNG → ESCA（反向）。兩個順序都跑。
- **Split**：使用 can_dataset 內建 `datasplit/fold_1.npz` 的 patient-level train/val/test；
  slide 隨其 patient 歸屬 split。fold 固定為 fold_1（split 不隨 seed 變，避免 split 變異與方法變異混淆）。
- **Pseudo-region 環境**（無座標的誠實替代，論文明確聲明）：
  - 每張 slide 的 patch 特徵以 k-means 聚成 R=32 個 cluster（N<32 時 R=N），
    `MiniBatchKMeans(n_clusters=R, random_state=0)`，離線一次、cache 到 `data/can_cache/`。
  - **高倍證據** `z_r`：cluster r 內全部 patch 特徵的 mean（512 維），zoom 後才可見。
  - **低倍摘要** `F_low[r]`：`z_r` 經固定 Gaussian random projection（512→64，seed=0，全域共用矩陣）
    的有損壓縮，全程可見。資訊不對稱：64 維摘要 vs 512 維證據（類比 thumbnail 是高倍影像的降採樣）。
  - cache 內容：`{assign, low(R,64), high(R,512)}`，projection 矩陣另存 `data/can_cache/proj_512to64.npz`。

### 1.2 第二資料集：pilot40（層級環境，真空間 region）

- 沿用現行設定：低倍 HIPT region（192）、高倍 CONCH 48-patch mean（512）、
  T1 renal（KIRC vs KIRP）→ T2 breast（IDC vs ILC），任務內 stratified 70/30（隨 seed 重抽）。
- **8/18 checkpoint**：若 KL 交付 80 張（各類 20），pilot80 直接替換 pilot40（loader 不變）；
  未到位則 pilot40 維持現狀，只作為第二資料集與 Gate 1 場域。

## 2. 觀察預算與 seeds

- **K ∈ {1, 2, 4}**（Gate 1 顯示小預算差距最大；K=8 只在 Gate 1 加固中保留）。
- **Seeds = {0, 1, 2, 3, 4}**（5 顆），控制：模型初始化、teacher 展開的 tie-break、
  訓練資料順序、random policy 重抽。split 不隨 seed 變（見 1.1）。
- Random baseline 每個設定內部再取 5 次抽樣平均。

## 3. 方法（main table 的列）

| 代號 | 定義 | 超參數 |
|---|---|---|
| `seqft` | 新任務直接 fine-tune 共享 navigator | — |
| `ewc` | + Fisher 正則（Fisher 由模仿 loss 梯度平方估計，任務結束時計算） | λ_ewc 由 T1→T2 val 掃 {10,100,1000} 選定後凍結 |
| `lwf` | + KL(π_new‖π_old)，計算於**新任務** states（無 buffer） | λ 同 ours |
| `replay` | + 舊任務 teacher states 重放（對舊 gain target 的模仿 loss） | buffer 同 ours |
| `distill` | + KL(π_new‖π_old) 於 buffer states，**uniform 權重** | λ、buffer 同 ours |
| `ours` | + utility-weighted distillation + replay target（完整方法） | λ=1.0、τ=0.05 |
| `joint` | 全部已見任務混合重訓（上界，非 CL 方法） | — |

- **Replay buffer**：每張舊任務訓練 slide 存前 2 個 teacher steps；每個舊任務 cap 512 states
  （超出時取 utility 最高者）。ablation 掃 buffer size ∈ {128, 512, 2048}。
- 訓練排程（can_dataset）：evaluator 30 epochs（Adam 1e-3, wd 1e-4, batch 64）、
  navigator 10 epochs（Adam 1e-3）；pilot40 沿用 60/30。
  （epochs 為運行成本參數，8/15 smoke 只校準運行時間，不回頭調結果。）

## 4. 指標與量測時點

每完成一個任務 t，對所有 j ≤ t 在 T_j **test** 上量測（evaluator f_j 永遠是 T_j 訓練完當下的凍結版）：

| 指標 | 定義 |
|---|---|
| `bal_acc(j, t)` | balanced accuracy（類別不平衡：BRCA 761/191、RCC 3 類） |
| `AA` | (1/n)·Σ_j bal_acc(j, n)（序列結束） |
| `forgetting(j)` | max_{t'≥j} bal_acc(j, t') − bal_acc(j, n) |
| `BWT` | (1/(n−1))·Σ_{j<n} [bal_acc(j, n) − bal_acc(j, j)] |
| `jaccard(j)` | T_j test 每張 slide：t=j 當下所選 region 集合 vs 序列結束所選集合的 Jaccard，取平均 |
| `action_kl(j)` | 固定 probe states（t=j 當下 T_j test 的 teacher states）上 KL(π_final‖π_{t=j}) |
| `sel_utility(j)` | 序列結束時所選 region 在 f_j 下的平均 counterfactual gain |
| `random_ref(j)` | random policy 在 f_j 下的 bal_acc（5 抽樣平均，遺忘的地板參考） |

## 5. 統計分析（凍結）

- 主表報 5 seeds 的 mean ± std；關鍵比較（ours vs seqft、ours vs distill）另報
  **paired bootstrap 95% CI**（以 seed × task 為配對單位，10,000 次重抽）。
- Gate 1 加固：5 seeds × 5 folds，learned−random 與 learned−uniform 的
  per-fold paired difference + bootstrap 95% CI + Wilcoxon signed-rank test。
- 顯著性聲明一律基於 CI 不跨 0；std 陰影不作顯著性宣稱（pilot 報告的教訓）。

## 6. 輸出格式（凍結）

- 主實驗逐列 CSV：`results/cl_main_{dataset}_{order}_{tag}.csv`
  欄位：`dataset, order, method, seed, K, stage, task, bal_acc, acc, jaccard, action_kl, sel_utility, random_ref, n_test, wall_s`
- Gate 1：`results/gate1_{task}_{tag}.csv`（沿用現行欄位 + 新增 `paired_diff` 分析輸出
  `results/gate1_significance.md`）。
- 聚合腳本：`scripts/aggregate_results.py` → `results/main_table.md`（論文主表直接複製用）。
- 圖：`figures/cl_budget_forgetting.png`（K × forgetting 交互）、`figures/cl_jaccard_bars.png`、
  `figures/gate1_ci.png`。

## 7. Gate（Go/No-Go 判準）

| Gate | 內容 | 判準 |
|---|---|---|
| G1'（8/18） | Gate 1 加固（pilot40 + can_dataset 單任務） | learned−random 的 95% CI > 0（至少 2 個 K） |
| G2'（8/22） | 4 任務 seqft 的遺忘 | K=1 或 K=2 下 forgetting(1) 的 CI > 0，或 jaccard(1) < 0.2 |
| G3'（8/24） | ours vs seqft | AA 與 forgetting 顯著改善；jaccard 顯著提高 |
| 備援 | G2' 若 accuracy 層仍不顯形 | 論文主 finding 轉為「redundancy 掩蓋 policy forgetting 的量化分析」（budget × forgetting 交互 + 行為層指標體系），方法與其餘實驗不變 |

## 8. 版本記錄

- v1（2026-08-14）：初版凍結。
