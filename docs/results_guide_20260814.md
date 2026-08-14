# Results Guide — 目前實驗與論文重新評估導讀

日期：2026-08-14  
Repository：`AaronHung/mllm_hwsi_ah`  
目的：讓新的 paper project 可以快速理解目前所有實驗、數字、圖、程式與限制，
再決定論文主 claim、下一輪實驗與稿件重寫方式。

這不是單純的 result dump。閱讀順序應該是：

```text
先讀本導讀
  → 再讀 project_handoff_20260814.md
  → 看 Gate 1 / main / reverse / ablation aggregate tables
  → 最後才看逐 seed CSV 與 implementation
```

---

## 1. 一頁結論

### 1.1 目前最可靠的研究結論

1. **Budgeted evidence-acquisition policy 是可學的。**
   Gate 1 在 pilot40 的 K=1、K=2 下 learned navigator 顯著優於 random。
2. **Navigation forgetting 是獨立於 prediction forgetting 的現象。**
   在 frozen evaluator protocol 下，seqft 的 selection Jaccard 明顯崩壞，
   即使 accuracy 沒有同步崩壞。
3. **Replay / policy distillation 類方法能降低 navigation forgetting。**
   Main order 與 reverse order 都顯示 ours 相對 seqft 的 Jaccard 改善。
4. **`ours` 不是所有 metric、所有 budget 的全面最佳方法。**
   `distill` 在多個設定下有更高 Jaccard、較低 action-KL；
   `ours` 的優勢較集中在 forgetting，以及部分 K=4 / memory / utility setting。
5. **Utility weighting 是 budget-dependent refinement，不是已證明的 universal winner。**
   K=2 相對 uniform weighting 有明顯 descriptive gain；K=1 差異很小。
6. **Memory size 與 λ 有 stability–plasticity trade-off。**
   memory 128 明顯不足；memory 2048 在 K=2 表現較好；
   λ=3 改善 Jaccard/KL，但不一定改善 forgetting。

### 1.2 目前不能宣稱

- 不能宣稱 `ours` 全面優於所有 CL baselines。
- 不能宣稱 utility weighting 在每個 K 都有效。
- 不能宣稱 navigation forgetting 已必然造成 accuracy forgetting；
  pilot Gate 2/3 反而顯示冗餘會遮蔽 accuracy 層遺忘。
- 不能把 can_dataset pseudo-region 說成完整的 spatial WSI navigation。
- 不能只用 main order 宣稱 task-order robustness；reverse order 雖已完成，
  仍需在 paper 中正式做 cross-order analysis。

---

## 2. 研究與系統背景

### 2.1 研究問題

研究的是：

> 在固定 observation budget 下，逐步選擇「下一個要觀察的 region」的
> evidence-acquisition policy，在 sequential task learning 後是否會忘記原本的取證行為？

WSI 只是 benchmark；一般 CV 抽象如下：

| WSI 現象 | CV 抽象 |
|---|---|
| gigapixel slide 無法全部讀取 | budgeted observation / active token selection |
| 只有 slide-level label | weakly supervised policy learning |
| 低倍→高倍逐步揭露 | coarse-to-fine decision under partial observability |
| task 依序到來 | continual learning of a selection policy |

### 2.2 實際架構

```text
coarse low feature F_low[r]：全程可見
        ↓
shared Navigator：對未 zoom candidates 打分
        ↓
選擇 region → 揭露 high evidence z_r
        ↓
evidence state E_t 更新
        ↓
固定 K 步後交給 frozen evaluator f_task
```

CL mechanism：

```text
old-task compressed states
        ↓
replay target + policy distillation
        ↓
utility-weighted regularization
```

關鍵程式：

- `nav/models.py`：`Evaluator`、`Navigator`
- `nav/engine.py`：environment、teacher rollout、training primitives
- `nav/cl.py`：seqft/EWC/LwF/replay/distill/ours/joint
- `scripts/cl_main.py`：主 CL grid
- `scripts/gate1_significance.py`：Gate 1 加固
- `scripts/aggregate_results.py`：main/reverse aggregation
- `scripts/aggregate_ablation.py`：ablation aggregation

完整 architecture 說明：

- `docs/architecture.md`
- `figures/fig1_architecture.png`

---

## 3. 實驗總覽與檔案對照

| 實驗 | 問題 | 主要輸入 | aggregate 文件 | 主要 figures |
|---|---|---|---|---|
| Exp A | feature 是否含有 task signal？ | pilot40 feature hierarchy | `results/expA_summary.md` | `figures/expA_main_bar.png` |
| Exp B | budgeted selection 是否有意義？ | pilot40 region mean | `results/expB_summary.md` | `figures/expB_budget_curve_4class.png` |
| Gate 1 | learned navigator 是否優於 random/uniform？ | pilot40 + can single-task | `results/gate1_significance.md` | `figures/gate1_ci.png` |
| Gate 2/3 pilot | navigation forgetting 是否出現？ | pilot40 renal→breast | `results/cl_navigation_summary.md` | `figures/...` 舊版 pilot 圖 |
| Main order | 4-task CL 是否遺忘？ | can ESCA→LUNG→RCC→BRCA | `results/main_table_can_main.md` | `figures/cl_*_can_main.png` |
| Reverse order | task order 是否影響結論？ | can BRCA→RCC→LUNG→ESCA | `results/main_table_can_reverse.md` | `figures/cl_*_can_reverse.png` |
| Ablation | utility/memory/λ 機制 | can main order | `results/ablation_table.md` | `figures/ablation_mechanism.png` |

---

## 4. Exp A：feature signal sanity check

### 問題

在投入 navigation 前，先確認 frozen feature 本身是否含有足夠的診斷 signal。
否則 navigator 失敗可能只是 feature 沒有 signal。

### 結果

來自 `n=40 pilot`：

- 二分類最佳 probe：cell/logreg，LOO accuracy = **97.5%**
- frozen MLLM fixed prompt：**55.0%**
- 四分類最佳 probe：region/logreg，LOO accuracy = **92.5%**
- frozen MLLM fixed prompt：**20.0%**
- 四分類 chance：**25.0%**

### 可支持 claim

> Feature asset contains substantial task signal; the original failure is not
> evidence that the WSI feature representation is useless.

### 限制

- WSI feature 是 region mean 的 hierarchy，部分層級不是獨立 evidence。
- cell feature 使用未經病理微調的 DINOv2。
- 這是 probe sanity check，不是 clinical performance。

### 對應檔案

- `results/expA_summary.md`
- `results/expA_probe_results.csv`
- `results/expA_inventory.md`
- `figures/expA_main_bar.png`

---

## 5. Exp B：budget 與手工 navigation sanity check

### 問題

若少量 region 已包含大部分 signal，budgeted navigation 才有研究空間；
同時測試手工 diversity / atypicality heuristic 是否足夠。

### 結果

- 四分類 random K=4：約 **78%**
- 四分類 K=32：約 **90%**
- full-information 上限：約 **95%**
- 手工 heuristic 多數沒有穩定勝過 random。
- spatial-uniform 與 random 大致打平。

四分類各 budget：

| K | 最佳策略 | 最佳 accuracy | random |
|---:|---|---:|---:|
| 1 | spatial_uniform | 60.0% | 57.0% |
| 2 | spatial_uniform | 62.5% | 67.0% |
| 4 | spatial_uniform | 77.5% | 78.0% |
| 8 | spatial_uniform | 65.0% | 76.0% |

### 可支持 claim

> Signal is redundant and spatial heuristics are insufficient; the selection
> policy must learn task-aligned diagnostic utility.

### 對應檔案

- `results/expB_summary.md`
- `results/expB_budget_curves.csv`
- `figures/expB_budget_curve_4class.png`

---

## 6. Gate 1：learnability 與統計加固

### 問題

在談 CL forgetting 前，先證明 learned navigator 在單任務內確實學得到
比 random 更好的 selection。

### Protocol

- pilot40：5 seeds × 5 folds
- K={1,2,4}
- learned vs random / uniform
- paired bootstrap 95% CI
- Wilcoxon signed-rank
- can single-task：ESCA、RCC 作第二環境檢查

### Full 結果

Pilot40：

| K | learned−random | 95% CI | Wilcoxon p |
|---:|---:|---|---:|
| 1 | +0.234 | [+0.188,+0.283] | <0.001 |
| 2 | +0.085 | [+0.023,+0.146] | 0.0156 |
| 4 | +0.042 | [−0.010,+0.094] | 0.1301 |

can ESCA：

- K=1：+0.053，CI [−0.005,+0.107]
- K=2：+0.035，CI [−0.016,+0.077]
- K=4：+0.051，CI [−0.003,+0.104]

can RCC：

- K=1：+0.059，CI [+0.033,+0.085]
- K=2：+0.034，CI [+0.012,+0.054]
- K=4：+0.019，CI [−0.001,+0.038]

### 結論

G1' = **PASS**。最穩健的論述是 learned > random at small budgets，
不是 learned 在全部 K、全部 dataset 都優於 uniform。

### 對應檔案

- `results/gate1_hardened_full.csv`
- `results/gate1_hardened_stats.csv`
- `results/gate1_significance.md`
- `figures/gate1_ci.png`
- `scripts/gate1_significance.py`

---

## 7. 舊版 pilot Gate 2/3：為什麼要擴大主實驗

### 結果

pilot40、K=4、2-task renal→breast：

| method | T1 before | T1 after | forgetting | action-KL | Jaccard |
|---|---:|---:|---:|---:|---:|
| seqft | 0.722 | 0.722 | 0.000 | 0.021 | 0.061 |
| distill | 0.722 | 0.778 | −0.056 | 0.009 | 0.211 |

### 正確解讀

- 行為層遺忘明確：seqft Jaccard = **0.061**。
- accuracy 層遺忘在 pilot 沒有顯形。
- 原因：slice-internal signal redundancy + test set 太小。
- 不能寫成「navigation forgetting 已造成 prediction accuracy drop」。

這個 pilot 直接導向目前主實驗：

1. K 壓到 1–2；
2. task sequence 擴到 3–4；
3. can_dataset 增加 test slides；
4. seeds 增加到 5。

### 對應檔案

- `results/cl_navigation_summary.md`
- `results/cl_navigation_full.csv`
- `figures/gate1_4class.png`
- 舊版相關 pilot figures

---

## 8. Main order：四任務 CL

### Protocol

```text
ESCA → LUNG → RCC → BRCA
```

- 5 seeds
- 7 methods
- K={1,2,4}
- fixed patient-level fold_1
- frozen evaluator

### Aggregate 檔案

- `results/main_table_can_main.md`
- `results/cl_main_can_main_full_s0.csv` … `s4.csv`
- `figures/cl_budget_forgetting_can_main.png`
- `figures/cl_jaccard_bars_can_main.png`

### Main-order 重要數字

`ours` vs `seqft`：

| K | AA difference | Forgetting difference | Jaccard difference |
|---:|---|---|---|
| 1 | +0.030，CI crosses 0 | −0.059，CI excludes 0 | +0.208，CI excludes 0 |
| 2 | +0.014，CI crosses 0 | −0.028，CI excludes 0 | +0.045，CI excludes 0 |
| 4 | +0.040，CI excludes 0 | −0.047，CI excludes 0 | +0.063，CI excludes 0 |

### Main-order interpretation

- `seqft` Jaccard：K=1 **0.051**、K=2 **0.099**、K=4 **0.079**。
- `ours` forgetting：K=1 **0.011**、K=2 **0.021**、K=4 **0.013**。
- `distill` 在 Jaccard/action-KL 上很有競爭力：
  - K=1 Jaccard 0.296、KL 0.059
  - K=2 Jaccard 0.181、KL 0.020
  - K=4 Jaccard 0.166、KL 0.004
- 因此 `ours` 的主 claim 應集中在 forgetting / selected utility / 特定 budget，
  不能寫成所有 behavior metrics 都最佳。

---

## 9. Reverse order：task-order robustness

### Protocol

```text
BRCA → RCC → LUNG → ESCA
```

同樣使用 5 seeds × 7 methods × K={1,2,4}。

### Aggregate 檔案

- `results/main_table_can_reverse.md`
- `results/cl_main_can_reverse_full_s0.csv` … `s4.csv`
- `figures/cl_budget_forgetting_can_reverse.png`
- `figures/cl_jaccard_bars_can_reverse.png`

### Reverse-order 重要數字

`ours` vs `seqft`：

| K | AA difference | Forgetting difference | Jaccard difference |
|---:|---|---|---|
| 1 | +0.064，CI excludes 0 | −0.064，CI excludes 0 | +0.235，CI excludes 0 |
| 2 | +0.030，CI crosses 0 | −0.024，CI crosses 0 | +0.089，CI excludes 0 |
| 4 | +0.038，CI crosses 0 | −0.034，CI crosses 0 | +0.089，CI excludes 0 |

### Reverse-order interpretation

- Jaccard improvement remains consistent。
- AA / forgetting improvement is strongest at K=1。
- `distill` 仍然很強：
  - K=1 Jaccard 0.351、forgetting 0.014
  - K=2 Jaccard 0.207、forgetting 0.024
  - K=4 Jaccard 0.192、forgetting 0.032
- 反向 order 支持 navigation forgetting 與 CL mitigation 的方向，
  但也再次提醒：`ours` 不是所有 metric 的 Pareto winner。

---

## 10. Ablation / mechanism

### 實驗設定

- 5 seeds
- main order
- K={1,2}
- utility weighting
- memory cap 128 / 512 / 2048
- λ=0.3 / 1.0 / 3.0

### Aggregate 檔案

- `results/ablation_table.md`
- `results/ablation_per_run.csv`
- `figures/ablation_mechanism.png`
- `scripts/aggregate_ablation.py`

### Utility weighting

比較 `ours`（utility）與 `ours_uniform`：

| K | ours forgetting | uniform forgetting | ours utility | uniform utility |
|---:|---:|---:|---:|---:|
| 1 | 0.011 | 0.012 | 0.411 | 0.387 |
| 2 | 0.021 | 0.044 | 0.136 | 0.092 |

解讀：

- K=2 支持 utility weighting 的方向。
- K=1 差異很小。
- 需要 paired seed-level CI 才能宣稱統計顯著。

### Memory size

K=2：

| memory | AA | forgetting | Jaccard | action-KL |
|---:|---:|---:|---:|---:|
| 128 | 0.891 | 0.031 | 0.121 | 0.047 |
| 512 | 0.897 | 0.021 | 0.144 | 0.028 |
| 2048 | 0.907 | 0.010 | 0.189 | 0.020 |

解讀：

- memory 128 明顯不足。
- memory 2048 在 K=2 顯示較好的 preservation。
- 目前還不能宣稱 2048 是最佳容量，因為尚未做 cost-normalized analysis。

### λ

K=2：

| λ | AA | forgetting | Jaccard | action-KL |
|---:|---:|---:|---:|---:|
| 0.3 | 0.905 | 0.021 | 0.124 | 0.040 |
| 1.0 | 0.897 | 0.021 | 0.144 | 0.028 |
| 3.0 | 0.900 | 0.026 | 0.201 | 0.016 |

解讀：

- λ 越大，behavior preservation（Jaccard/KL）傾向較強。
- λ=3 的 forgetting 反而較高，顯示 stability–plasticity trade-off。
- 不能只用 Jaccard 選 λ，也不能只用 forgetting 選 λ。

---

## 11. 目前所有 figures 的閱讀順序

### 第一層：系統與方法

- `figures/fig1_architecture.png`
  - causal access rule
  - shared navigator
  - frozen evaluator
  - replay/distillation loop

### 第二層：learnability

- `figures/expA_main_bar.png`
  - feature signal sanity check
- `figures/expB_budget_curve_4class.png`
  - budget 與 random / heuristic baseline
- `figures/gate1_4class.png`
  - 舊版 Gate 1 visualization
- `figures/gate1_ci.png`
  - 新版 5-seed / 5-fold CI visualization

### 第三層：continual forgetting

- `figures/cl_budget_forgetting_can_main.png`
  - main order 的 budget × forgetting / Jaccard
- `figures/cl_jaccard_bars_can_main.png`
  - main order 的 behavior preservation
- `figures/cl_budget_forgetting_can_reverse.png`
  - reverse order 的 budget × forgetting / Jaccard
- `figures/cl_jaccard_bars_can_reverse.png`
  - reverse order 的 behavior preservation

### 第四層：mechanism

- `figures/ablation_mechanism.png`
  - AA / forgetting / Jaccard across distill, ours, memory and λ settings

---

## 12. 重新評估時的建議 paper decision

### 推薦主故事

```text
navigation forgetting 是一個被現有 prediction-only CL metrics 忽略的
behavior-level failure；frozen evaluator protocol 可以把它量出來，
而 replay/distillation 可以降低它。
```

### `ours` 的保守定位

```text
Utility-weighted policy distillation with replay is a budget-dependent
refinement. It reduces forgetting relative to seqft, but uniform distillation
is a strong baseline for Jaccard and action-KL.
```

### 重新評估的三個必要問題

1. `ours` 是否要作主要 method，還是作為 replay/distillation family 中的
   utility-weighted variant？
2. paper primary metric 是：
   - forgetting？
   - Jaccard？
   - action-KL？
   - AA？
   必須事先定義，而不能看完結果後選最有利的 metric。
3. pilot40 是否值得跑 full CL？
   如果主 claim 是 benchmark / behavior-level protocol，pilot40 可只跑
   seqft/distill/ours/joint；不需要複製全部 7 methods。

---

## 13. 下一步執行建議

目前不要直接再擴張 experiment matrix。建議：

1. 對 main/reverse/ablation 的關鍵差異做 paired seed-level CI。
2. 審計 `distill-only`、`ours_uniform`、`ours` 的 loss / update / buffer fairness。
3. 決定 `ours` 是否保留為 main method。
4. 若需要外部 generalization，再跑 pilot40 的最小四方法：

```text
seqft / distill / ours / joint
K={1,2,4}
seeds={0,1,2,3,4}
```

5. 最後才更新 `paper/main.tex`、abstract、main table 與 slides。

目前不要把未完成的 pilot40 當成 blocker；main/reverse/ablation 已足以先做
一次完整 paper story review。

---

## 14. 相關核心文件

- `docs/project_handoff_20260814.md`
  - 長版 research handoff、claim ledger、copy-ready prompt
- `docs/architecture.md`
  - P0 architecture / what-why-how / training / loss
- `docs/protocol.md`
  - frozen experiment protocol
- `docs/vln_survey.md`
  - 31 篇 VLN survey 與 WSI 適用性分析
- `docs/slides_monthly.md`
  - 月會投影片 skeleton
- `paper/main.tex`
  - ICASSP 四頁 draft skeleton，仍有 placeholder
- `RUNPOD_SOP.md`
  - RunPod / tmux / data transfer / result sync

---

## 15. 給新的 Paper Project 的交接指令

請先閱讀本導讀與 `docs/project_handoff_20260814.md`，然後輸出一份
**paper project adjustment memo**，不要直接重跑：

1. 列出 supported / provisional / unsupported claims。
2. 比較 main order 與 reverse order 的 effect consistency。
3. 審計 `ours`、`distill`、`ours_uniform` 的公平性。
4. 決定 primary metric 與 main table rows。
5. 決定 pilot40 是否需要，以及最小必要 matrix。
6. 指出 paper draft 哪些句子必須改寫。
7. 只有 adjustment memo 確認後，才啟動下一輪實驗。
