# Paper Project Handoff — 2026-08-14

> 這份文件是交給下一個 paper project / research planning session 的完整交接包。
> 目的不是要求新 project 盲目照舊計畫跑，而是讓它先理解：
> **目前真正做了什麼、目前結果支持什麼、哪些故事已經不能直接宣稱、下一步該如何調整。**
>
> 建議把本文件全文貼給新的 paper project，並要求它先輸出「計畫調整 memo」，
> 經確認後才執行新的實驗。不要只把原始 CSV 或原始 sprint plan 單獨丟過去。

---

## 0. 給新 Paper Project 的第一條指令

請先不要重跑實驗，也不要先改論文文字。先完成以下四件事：

1. 讀本文件與列出的關鍵檔案。
2. 將所有論文 claim 分成：
   - **已由結果支持**
   - **目前有方向但證據不足**
   - **目前結果不支持或需要改寫**
   - **尚未完成**
3. 審計 `scripts/aggregate_results.py`、`scripts/cl_main.py`、`nav/cl.py` 的指標與方法實作，
   確認目前表格不是由 aggregation 或 implementation bug 造成。
4. 提出一份新的 project adjustment memo，至少回答：
   - 論文主 claim 要保留哪一版？
   - `ours` 是否真的要作為主要方法，而不是只作其中一個 CL baseline？
   - 哪些實驗必須補跑，哪些可以刪除？
   - 主表、ablation、mechanism figure、paper text 應如何重排？

只有 adjustment memo 確定後，才開始下一輪實驗。

---

## 1. Project identity 與研究主線

### 1.1 Repository

- Local repo：`/Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah`
- GitHub repo：`AaronHung/mllm_hwsi_ah`
- Upstream：`nsysu-aivisual/MLLM-HWSI`
- Legacy navigation repo：`/Users/aaron/research/01_navipath`

### 1.2 暫定論文題目

> **Remembering Where to Look: Continual Learning of Budgeted Visual Evidence Acquisition**

這個題目目前仍可使用，但不能把它視為已定稿。新 project 要先根據主實驗與反向 task order
確認「navigation forgetting」是否是最穩健的主軸。

### 1.3 研究問題

我們研究的不是一般 WSI classifier forgetting，而是：

> 一個在固定 observation budget 下，逐步選擇「下一個要觀察的 region」的
> evidence-acquisition policy，在 sequential task learning 後是否會忘記原本的取證行為？

這個問題可抽象成一般 CV 問題：

| WSI 實例 | 一般 CV 抽象 |
|---|---|
| gigapixel slide 無法全部讀取 | budgeted observation / active token selection |
| 只有 slide-level label | weakly supervised policy learning |
| 低倍到高倍逐步揭露 | coarse-to-fine decision under partial observability |
| 癌種任務依序到來 | continual learning of a selection policy |

### 1.4 最重要的概念區分

- **Prediction forgetting**：classifier / evaluator 的預測能力下降。
- **Navigation forgetting**：policy 選擇 evidence 的行為改變。
- 本 project 的方法性價值在於：利用 frozen per-task evaluator，把兩者分離。

---

## 2. 目前實際完成狀態（截至 2026-08-14 約 17:18）

| 項目 | 狀態 | 證據 / 位置 |
|---|---|---|
| P0 architecture package | 完成 | `docs/architecture.md`、`figures/fig1_architecture.png` |
| what/why/how | 完成 | `docs/architecture.md` |
| training flow / loss | 完成 | `docs/architecture.md` |
| VLN survey | 完成 | `docs/vln_survey.md`，301 行、31 篇參考文獻 |
| protocol freeze | 完成 | `docs/protocol.md` |
| can_dataset inventory | 完成 | 4 cohorts、pseudo-region cache |
| can cache | 完成 | `data/can_cache/`，約 197MB |
| Gate 1 hardening | 完成，PASS | `results/gate1_significance.md` |
| 主順序 main CL grid | 完成 | 5 個 `cl_main_can_main_full_s*.csv`，每個 211 行 |
| 主順序 aggregation | 已完成 | `results/main_table_can_main.md`、兩張 figures |
| 反向 task order | 完成 | `results/main_table_can_reverse.md`、兩張 figures |
| pilot40 full CL grid | 尚未開始，queue 已暫停 | 待 paper plan 調整後決定 |
| utility / memory / lambda ablation | 完成 | `results/ablation_table.md`、`figures/ablation_mechanism.png` |
| paper draft | 有四頁 skeleton，但數字仍有 placeholder | `paper/main.tex` |
| monthly slides | 有 skeleton | `docs/slides_monthly.md` |

### 2.1 目前的 process 狀態

主順序與反向順序都已完成；完整 ablation 也已完成。
pilot40 queue 目前刻意暫停，等待 paper project 根據全部結果重新調整 claim 與實驗優先序。

重要：這些 background jobs 不是 `tmux`。它們依賴目前 Mac / Cursor session；
若需要關機或合蓋長時間執行，應先把下一批移到 RunPod `tmux`，不要把這些 process
誤當成可跨機器接管的 checkpoint job。

---

## 3. 已實作系統：不要把未實作構想當成結果

### 3.1 Causal feature pyramid

每張 slide 有 `R` 個候選 region：

1. `F_low[r]`：低倍 / coarse 特徵，全程可見。
2. `z_r`：高倍 / fine evidence，只有 region 被 zoom 後才能取得。
3. 每一步選一個未 zoom region，更新 evidence state。
4. 固定跑 `K` 步，沒有 learned stop。
5. frozen evaluator 只讀最後累積的 high-level evidence。

這個 causal access rule 排除了「先讀完全部 high feature，再事後排序」的假導航。

### 3.2 Shared Navigator

目前是單一共享 MLP：

```text
input = candidate low feature
      ⊕ current evidence state
      ⊕ remaining budget
output = one score per unzoomed candidate
policy = softmax over candidates
```

預設 pilot40 維度是 HIPT 192 + CONCH 512；can_dataset 使用：

- high evidence：512 維 cluster mean
- low summary：512→64 fixed Gaussian random projection

`Navigator` 位於 `nav/models.py`；training loop 主要位於 `nav/cl.py` 與
`nav/engine.py`。

### 3.3 Frozen Evaluator

每個 task 各自訓練一個 evaluator，task 結束後 freeze。

這個 protocol 的目的不是提升 classifier，而是 attribution：

> 舊 task 的結果在後續 task 下降時，evaluator 沒有更新，因此下降只能歸因於
> shared navigator 的 evidence-selection 行為改變。

### 3.4 Counterfactual teacher

對 candidate region `r`：

```text
gain(r | E)
  = CE(f(E), y) - CE(f(E ∪ {r}), y)
```

使用 frozen evaluator 產生 teacher gain distribution，再以：

```text
q(r | E) ∝ exp(gain(r | E) / tau)
```

監督 navigator。

這是因為 TCGA 沒有大規模 pathologist viewing trajectories；目前不是 imitation
human trajectory，而是 label-derived counterfactual teacher。

### 3.5 CL methods

目前主表的 7 列：

| 方法 | 實作定義 |
|---|---|
| `seqft` | 新 task 直接 fine-tune shared navigator |
| `ewc` | Fisher-weighted parameter regularization |
| `lwf` | 新 task state 上的 old-policy KL |
| `replay` | 舊 task teacher states replay |
| `distill` | 舊 task replay states 上 uniform policy distillation |
| `ours` | replay target + utility-weighted policy distillation |
| `joint` | 已見 task 混合重訓，上界 |

`ours` 的 utility weighting：

```text
u(s) = max_r gain(r | E_s)
w(s) = normalized u(s), clipped at 5
```

其目標是把保存能力集中在真正有診斷價值的 decision states。

---

## 4. Frozen protocol

### 4.1 Main dataset

`can_dataset` 原始資料只有 patch feature，沒有座標：

| Task | Cohort | Classes | 有特徵 slide 約數 |
|---|---|---|---:|
| T1 | ESCA | ESCC / ESAD | 150 |
| T2 | LUNG | LUAD / LUSC | 965 |
| T3 | RCC | CCRCC / PRCC / CHRCC | 888 |
| T4 | BRCA | IDC / ILC | 952 |

主順序：

```text
ESCA → LUNG → RCC → BRCA
```

反向 robustness order：

```text
BRCA → RCC → LUNG → ESCA
```

### 4.2 Pseudo-region construction

因為 can_dataset 沒有 patch coordinates，不能假裝有真實 spatial region。
目前採取明確標示為 feature-space pseudo-region 的設計：

1. 每張 slide 的 patch feature 用 `MiniBatchKMeans` 聚成 `R=32`。
2. 每一群的 mean 是 high evidence `z_r ∈ R^512`。
3. 用固定 `512→64` Gaussian projection 產生 low summary。
4. projection matrix seed=0，全 dataset 共用。
5. cache 放在 `data/can_cache/`。

這個設計可以支持方法研究，但不能在論文中宣稱是完整的 spatial WSI navigation。
論文必須把它定位成 feature-space causal pyramid benchmark。

### 4.3 Secondary dataset

`pilot40` 有真實 region coordinates：

- renal：KIRC / KIRP
- breast：IDC / ILC
- HIPT region 192 維
- CONCH patch mean 512 維
- task sequence：renal → breast
- task 內 70/30 stratified split

pilot40 的角色：

1. Gate 1 learnability sanity check。
2. 第二環境 generalization。
3. 真空間 region 與 pseudo-region can_dataset 的對照。

### 4.4 Budget / seeds / metrics

- `K ∈ {1,2,4}`
- seeds `{0,1,2,3,4}`
- can split 固定 `fold_1.npz`，patient-level train/val/test
- random baseline 每個設定 5 次抽樣

主要 metrics：

- balanced accuracy
- AA
- Forgetting
- BWT
- selection Jaccard
- action KL drift
- selected-region counterfactual utility

統計：

- main table：5 seeds mean ± std
- 關鍵差異：paired bootstrap 95% CI
- Gate 1：paired difference + bootstrap CI + Wilcoxon
- CI 是否跨 0 才可作顯著性說明

完整定義在 `docs/protocol.md`，不要在新 project 中無記錄地改動。

---

## 5. Gate 1 已知結果

Gate 1 已跑完，產物：

- `results/gate1_hardened_full.csv`
- `results/gate1_hardened_stats.csv`
- `results/gate1_significance.md`
- `figures/gate1_ci.png`

### 5.1 Pilot40

| K | learned − random | 95% CI | Wilcoxon p |
|---:|---:|---|---:|
| 1 | +0.234 | [+0.188, +0.283] | <0.001 |
| 2 | +0.085 | [+0.023, +0.146] | 0.0156 |
| 4 | +0.042 | [−0.010, +0.094] | 0.1301 |

learned vs uniform：

- K=1：+0.055，CI crosses 0
- K=2：+0.075，CI [+0.020,+0.130]
- K=4：+0.020，CI crosses 0

### 5.2 can single-task checks

`can_esca`：

- K=1：+0.053，CI [−0.005,+0.107]
- K=2：+0.035，CI [−0.016,+0.077]
- K=4：+0.051，CI [−0.003,+0.104]

`can_rcc`：

- K=1：+0.059，CI [+0.033,+0.085]
- K=2：+0.034，CI [+0.012,+0.054]
- K=4：+0.019，CI [−0.001,+0.038]

### 5.3 Gate 1 結論

修正 aggregation bug 後，G1' = **PASS**：

- pilot40 K=1、K=2 的 learned−random CI > 0
- can_rcc K=1、K=2 的 CI > 0
- learned navigator 在小 budget 下不是 random baseline

不要宣稱 learned 在每個 dataset、每個 K 都顯著優於 uniform；
目前結果只支持小 budget 的 learnability。

---

## 6. Main-order 實際結果

主順序 5 seeds 已完整聚合：

- 原始：`results/cl_main_can_main_full_s0.csv` 到 `s4.csv`
- 每個 CSV：211 行（header + 210 data rows）
- 聚合表：`results/main_table_can_main.md`
- budget/forgetting figure：
  `figures/cl_budget_forgetting_can_main.png`
- behavior figure：
  `figures/cl_jaccard_bars_can_main.png`

以下為 mean ± std over 5 seeds 的重點。

### 6.1 K=1

| method | AA | Forgetting | Jaccard | action-KL |
|---|---:|---:|---:|---:|
| seqft | 0.858 ± 0.028 | 0.070 ± 0.036 | 0.051 ± 0.029 | 0.322 |
| ewc | 0.864 ± 0.020 | 0.069 ± 0.028 | 0.118 ± 0.070 | 0.198 |
| lwf | 0.896 ± 0.027 | 0.018 ± 0.016 | 0.136 ± 0.033 | 0.085 |
| replay | 0.893 ± 0.033 | 0.016 ± 0.010 | 0.179 ± 0.032 | 0.104 |
| distill | 0.892 ± 0.025 | 0.015 ± 0.013 | **0.296 ± 0.050** | **0.058** |
| ours | 0.888 ± 0.031 | **0.011 ± 0.012** | 0.259 ± 0.030 | 0.069 |
| joint | 0.905 ± 0.034 | 0.017 ± 0.015 | 0.242 ± 0.027 | 0.076 |

ours vs seqft paired CI：

- AA：+0.030，CI [−0.002,+0.056]，不顯著
- Forgetting：−0.059，CI [−0.093,−0.027]，顯著改善
- Jaccard：+0.208，CI [+0.189,+0.229]，顯著改善

### 6.2 K=2

| method | AA | Forgetting | Jaccard | action-KL |
|---|---:|---:|---:|---:|
| seqft | 0.883 ± 0.027 | 0.048 ± 0.038 | 0.099 ± 0.027 | 0.111 |
| ewc | 0.903 ± 0.005 | 0.021 ± 0.009 | 0.111 ± 0.032 | 0.087 |
| lwf | 0.898 ± 0.016 | 0.031 ± 0.017 | 0.171 ± 0.043 | 0.032 |
| replay | 0.898 ± 0.029 | 0.025 ± 0.012 | 0.128 ± 0.023 | 0.055 |
| distill | 0.902 ± 0.025 | 0.021 ± 0.022 | **0.181 ± 0.029** | **0.020** |
| ours | 0.897 ± 0.020 | **0.021 ± 0.021** | 0.144 ± 0.011 | 0.028 |
| joint | 0.914 ± 0.014 | 0.017 ± 0.019 | 0.169 ± 0.046 | 0.028 |

ours vs seqft paired CI：

- AA：+0.014，CI [−0.008,+0.039]，不顯著
- Forgetting：−0.028，CI [−0.059,−0.005]，顯著改善
- Jaccard：+0.045，CI [+0.031,+0.058]，顯著改善

### 6.3 K=4

| method | AA | Forgetting | Jaccard | action-KL |
|---|---:|---:|---:|---:|
| seqft | 0.858 ± 0.014 | 0.060 ± 0.023 | 0.079 ± 0.019 | 0.020 |
| ewc | 0.860 ± 0.033 | 0.049 ± 0.042 | 0.102 ± 0.012 | 0.018 |
| lwf | 0.888 ± 0.034 | 0.032 ± 0.017 | 0.128 ± 0.048 | 0.007 |
| replay | 0.886 ± 0.012 | 0.029 ± 0.012 | 0.135 ± 0.043 | 0.028 |
| distill | 0.896 ± 0.018 | 0.024 ± 0.027 | **0.166 ± 0.037** | **0.004** |
| ours | **0.898 ± 0.011** | **0.013 ± 0.011** | 0.143 ± 0.013 | 0.010 |
| joint | 0.890 ± 0.022 | 0.019 ± 0.015 | 0.131 ± 0.029 | 0.006 |

ours vs seqft paired CI：

- AA：+0.040，CI [+0.024,+0.061]，顯著改善
- Forgetting：−0.047，CI [−0.065,−0.024]，顯著改善
- Jaccard：+0.063，CI [+0.052,+0.075]，顯著改善

### 6.4 Ablation / mechanism results

完整 ablation 產物：

- `results/ablation_table.md`
- `results/ablation_per_run.csv`
- `figures/ablation_mechanism.png`
- aggregation script：`scripts/aggregate_ablation.py`

設定：

- `ablA1`：uniform utility weight，等同 `ours_uniform`
- `ablA2a` / `ablA2b`：memory cap 128 / 2048
- `ablA3a` / `ablA3b`：λ=0.3 / 3.0
- 每個 setting：5 seeds、K={1,2}

關鍵數字（mean over 5 seeds）：

| setting | K | AA | forgetting | Jaccard | action-KL | selected utility |
|---|---:|---:|---:|---:|---:|---:|
| ours, memory 512, λ=1 | 1 | 0.888 | 0.011 | 0.259 | 0.069 | 0.411 |
| uniform weight | 1 | 0.904 | 0.012 | 0.253 | 0.068 | 0.387 |
| ours, memory 512, λ=1 | 2 | 0.897 | 0.021 | 0.144 | 0.028 | 0.136 |
| uniform weight | 2 | 0.887 | 0.044 | 0.146 | 0.030 | 0.092 |
| memory 128 | 2 | 0.891 | 0.031 | 0.121 | 0.047 | 0.085 |
| memory 2048 | 2 | 0.907 | 0.010 | 0.189 | 0.020 | 0.190 |
| λ=0.3 | 2 | 0.905 | 0.021 | 0.124 | 0.040 | 0.115 |
| λ=3.0 | 2 | 0.900 | 0.026 | 0.201 | 0.016 | 0.141 |

目前可支持的 interpretation：

1. **Utility weighting 有 budget-dependent effect。** K=2 時相對 uniform
   將 forgetting 從 0.044 降至 0.021，selected utility 從 0.092 升至 0.136；
   K=1 時差異很小，不能宣稱 universal improvement。
2. **Memory 128 明顯不足。** K=2 時 memory 128 的 forgetting/Jaccard/KL
   都劣於 memory 512；memory 2048 在 K=2 進一步改善 AA、forgetting、Jaccard。
3. **λ 是 stability-plasticity trade-off。** λ=3 在 K=2 提高 Jaccard、
   降低 action-KL，但 forgetting 反而高於 λ=1；不能只用單一 metric 選 λ。
4. 這些是 5-seed descriptive results；若要寫「顯著」，仍需對重要 ablation
   做 paired seed-level CI。

---

## 7. 目前結果真正支持的 claim

### 7.1 已支持

1. **Navigation forgetting 是可量測且不同於 prediction forgetting 的現象。**
   seqft 的 selection Jaccard 在主順序中非常低：
   K=1 約 0.051、K=2 約 0.099、K=4 約 0.079。
2. **Frozen evaluator protocol 可以把舊 task 的變化歸因於 navigator。**
3. **Learned navigator 在小 budget 下優於 random。**
   Gate 1 pilot40 的 K=1、K=2 CI 都不跨 0。
4. **CL regularization / replay 顯著減少 forgetting。**
   `ours` vs `seqft` 的 forgetting CI 在 K=1、2、4 都不跨 0。
5. **Budget 與 forgetting 有 interaction 的候選證據。**
   K=1/2/4 的 behavior-level Jaccard 都低，且 seqft forgetting 仍存在；
   需要反向 order 與 mechanism analysis 後才能寫成最終 claim。
6. **WSI 可以作為一般 CV evidence-acquisition 的 benchmark。**
   這個方法論 framing 已由 architecture、VLN survey、causal access rule 支持。

### 7.2 目前不能直接宣稱

1. 不能宣稱 `ours` 在所有 metric、所有 K 都是最佳。
   `distill` 在 K=1、2、4 都有更高 Jaccard，且 action-KL 也常更低。
2. 不能宣稱 utility weighting 在所有 budget 都有效。
   Ablation 顯示 K=2 相對 uniform 有明顯 descriptive gain，但 K=1 差異很小；
   重要 ablation 差異仍需 paired seed-level CI，且 `distill-only` 與
   `uniform+replay` 不是同一個 baseline。
3. 不能宣稱 AA 在所有 budget 顯著改善。
   目前 ours vs seqft 的 AA：
   - K=1：CI 跨 0
   - K=2：CI 跨 0
   - K=4：CI 不跨 0
4. 不能把 can_dataset pseudo-region 寫成真實 spatial navigation。
5. 不能只把主順序結果當成 task-order robust；反向 order 已完成，
   但仍需將兩個 order 的 ranking 與 effect size 合併分析。
6. 不能使用 paper draft 中尚未填數字的 placeholder 作為結果。

---

## 8. 論文故事需要如何調整

### 8.1 建議主故事（目前最穩健）

建議採用：

> **A policy can forget where to look even when prediction accuracy remains
> apparently stable. A frozen-evaluator protocol reveals this behavioral
> forgetting, and replay/distillation-based CL substantially reduces it.**

這個版本把主要 novelty 放在：

1. problem definition：navigation forgetting；
2. attribution protocol：frozen evaluator；
3. behavior-level metrics；
4. weak-supervision setting；
5. CL mitigation。

### 8.2 `ours` 的定位要保守重寫

目前較誠實的說法是：

> Utility-weighted policy distillation with replay is a strong forgetting
> mitigation, with a budget-dependent advantage over uniform weighting;
> however, uniform policy distillation remains a competitive baseline for
> behavioral overlap. Memory size and lambda expose a stability--plasticity
> trade-off rather than a single universally optimal setting.

中文意思：

- `ours` 不是已證明的全面 Pareto winner。
- `ours` 目前最強的證據是 forgetting 與 K=4 AA。
- `distill` 是很強的 competitive baseline，甚至在 Jaccard/KL 上更好。
- 這個差異本身可以變成 mechanism finding，而不是藏起來。

### 8.3 備援故事

如果 paired ablation CI 無法支持 utility weighting 的穩定增益，應把方法 claim 降級成：

> Replay-based policy preservation is sufficient to prevent much of the
> navigation forgetting; utility weighting is a targeted variant whose
> benefit depends on budget, memory, and task order.

論文仍可成立，因為 navigation forgetting + behavior-level evaluation
本身仍是主要方法分析 contribution。

---

## 9. 新 Paper Project 必須做的 audit

### 9.1 Aggregation audit

逐項確認：

- `results/main_table_can_main.md` 的 per-run grouping 是否正確。
- `forgetting` 是否使用 task-own stage 到 final stage。
- `jaccard` 是否對同一 slide 的 own-time selection 與 final selection 配對。
- action KL 的方向是否全篇一致。
- `joint` 是否真的是 all-seen-task mixed training。
- EWC 的 λ=1000 是否只由 validation 選一次且未看 test。

### 9.2 Method audit

特別比較：

```text
replay-only
distill-only
uniform replay + distill
utility-weighted replay + distill
```

確認 `ours` 與 `distill` 的差異到底來自：

- replay target
- policy KL
- utility weight
- random state sampling
- buffer content
- training update count

如果每個 method 的 gradient update 數不等，需補一個 compute-matched comparison。

### 9.3 Statistical audit

主表目前用 5 seeds；新 project 應增加：

- seed-level paired bootstrap：ours vs seqft
- seed-level paired bootstrap：ours vs distill
- task-level paired bootstrap 作為 sensitivity analysis
- 不要把每個 slide 當成完全獨立樣本後宣稱泛化顯著
- 若多個 K / 多個 method 都做 hypothesis test，說明 multiple-comparison policy

### 9.4 Data / environment audit

- can_dataset 的 R=32 cluster 是否對不同 seed 固定？
- low projection 是否只 fit 一次且沒有 label leakage？
- high evidence cluster mean 是否在 zoom 前真的沒有被 navigator 使用？
- evaluator 的 random K-subset training 是否與 random baseline 公平？
- pilot40 與 can_dataset 的 dimension / training epoch 差異是否需要 normalization？

---

## 10. 建議下一輪實驗順序

### Phase 0：不要重跑已完成的東西

保留：

- Gate 1 full artifacts
- can main-order 5-seed full grid
- P0 docs
- VLN survey

### Phase 1：完成現有計畫中的必要結果

1. 完成 can reverse order。
2. 聚合 reverse-order table。
3. 比較 main vs reverse：
   - forgetting
   - Jaccard
   - action KL
   - ours vs distill ranking
4. 確認主 claim 是否依 task order 改變。

### Phase 2：ablation（已完成，待統計整理）

已完成的設定：

1. uniform utility weight vs utility weighting；
2. replay buffer size 128 / 512 / 2048；
3. λ=0.3 / 1.0 / 3.0。

結果已寫入 `results/ablation_table.md`。下一步不是盲目擴大矩陣，
而是：

```text
對關鍵差異做 paired seed-level CI
確認 utility weighting 是否值得保留為主要 method claim
確認 memory 2048 是否只是 descriptive trend
```

若需要補跑，優先只補最能改變 paper claim 的設定。

### Phase 3：pilot40 generalization

至少跑：

- seqft
- distill
- ours
- joint

不要在時間不足時先跑所有 7 methods；pilot40 的角色是確認環境 generalization，
不是再複製完整 main table。

### Phase 4：mechanism figures

至少需要：

1. budget × forgetting curve
2. budget × selection Jaccard
3. action KL drift
4. selected-region utility
5. ours vs distill / uniform 的 utility weighting ablation

如果要借 VLN survey 的 insight，再加：

- teacher top-K coverage（OSR-like metric）
- accuracy-per-budget curve

---

## 11. RunPod / Mac 執行決策

### Mac

優點：

- cache 已經在 local；
- 不需要重新抽 feature；
- Gate 1 已證明可以跑；
- 可做小規模 audit / smoke / 3 seeds ablation。

限制：

- 目前 local background jobs 不是 `tmux`；
- 合蓋、sleep、關機都可能停止 process；
- 不能把 Mac 正在跑的 process 無縫搬到 RunPod；
- 目前 script 沒有 model checkpoint，只有 CSV output。

### RunPod

建議用於：

- 反向 order（若 Mac 還未完成且時間緊）
- full ablation
- pilot40 full grid
- 任何需要重跑 5 seeds 的設定

RunPod 每次重建的原則：

1. 程式從 `AaronHung/mllm_hwsi_ah` clone/pull。
2. `can_cache` 用 tar/scp 或 persistent Volume 傳入。
3. table 與 fold split 保持在 `CAN_ROOT`。
4. 每次開機重新執行 bootstrap：
   - `requirements-nav.txt`
   - PATH / Python environment
   - `CAN_ROOT`
   - git identity / remote
5. `tmux` 內執行。
6. 每一個 experiment 用新的 `tag` 和輸出檔，避免與 Mac partial CSV 混寫。

不要把 RunPod 的 ephemeral root dotfiles 當成永久設定；所有設定應寫進 repo 內的
bootstrap / run script。

---

## 12. Paper draft 目前需要修改的地方

`paper/main.tex` 目前是四頁 skeleton，仍有 `XX` placeholder。
新 project 需要先修改以下敘述，再填數字：

1. 將「ours improves final average accuracy over all baselines」改成條件式：
   - forgetting consistently improves over seqft；
   - AA improvement strongest at K=4；
   - distill is competitive on Jaccard / action KL。
2. 不要寫「ours best at every budget」。
3. 把 `distill` 的競爭力放進 main analysis，而不是只放 ablation。
4. 把 pseudo-region limitation 明確寫入 problem setting。
5. 等 reverse order 後再寫 task-order robustness。
6. Gate 1 的 claim 寫成：
   - learned > random at small budgets；
   - 不宣稱 learned > uniform 所有 K。
7. Figure 3 / budget curve 要用實際 aggregation 產出的 figures。

---

## 13. 可直接貼給新 Paper Project 的 Prompt

以下內容可直接複製：

```text
你現在接手一個 ICASSP 2027 paper project。請先不要重跑實驗或直接改 paper。
請完整讀取：

1. docs/project_handoff_20260814.md
2. docs/architecture.md
3. docs/protocol.md
4. docs/vln_survey.md
5. results/gate1_significance.md
6. results/main_table_can_main.md
7. paper/main.tex
8. nav/engine.py
9. nav/models.py
10. nav/cl.py
11. scripts/cl_main.py
12. scripts/aggregate_results.py

研究主題是：
Continual Learning of Budgeted Visual Evidence Acquisition。
核心問題是 navigation forgetting：shared evidence-selection policy 在
sequential task learning 後會忘記去哪裡取證；這與 prediction forgetting 分離。
WSI 是 benchmark，CV methodology 是主要定位。

目前已知實驗事實：
- Gate 1 learned navigator 在 pilot40 K=1/2 顯著優於 random。
- can_dataset main order 的 5 seeds × 7 methods × K={1,2,4} 已完成。
- seqft 的 selection Jaccard 很低，且 forgetting 明顯。
- ours 的 forgetting 顯著低於 seqft。
- 但是 distill-only 在部分 Jaccard / action-KL 指標優於 ours。
- 因此不能直接宣稱 ours 在所有 metric 都是最佳。
- reverse order 與 ablation 已完成；pilot40 full CL 尚未開始，
  且 ablation 的 paired CI / method audit 尚待完成。

請先輸出一份「paper project adjustment memo」，內容必須包括：

A. Evidence ledger：
   每一個 planned claim 對應哪個結果、哪個 CSV、哪個 metric；
   標記 supported / provisional / unsupported / missing。

B. Story decision：
   比較以下三個故事：
   1. method-centric：utility-weighted distillation + replay；
   2. analysis-centric：navigation forgetting 被 redundancy masking；
   3. protocol-centric：frozen evaluator + behavior-level metrics。
   推薦一個主故事與一個 fallback，並說明理由。

C. Method audit：
   審計 replay、distill、ours 的實作差異、update count、buffer sampling、
   utility weighting、loss scale 與 compute fairness。

D. Experiment adjustment：
   列出 reverse order、pilot40、utility ablation、memory ablation、
   lambda sweep、teacher top-K coverage 各自的必要性；
   指出哪些可以刪除，哪些必須 5 seeds，哪些只需 3 seeds。

E. Main table redesign：
   建議 final table 應保留哪些 method；
   是否把 distill 做成主要 baseline；
   primary metrics 應是 AA/forgetting/Jaccard/action-KL 中哪些；
   禁止 cherry-pick 結果。

F. Paper rewrite plan：
   重新安排 abstract、introduction、method、experiments、mechanism analysis；
   明確列出 paper/main.tex 中所有必須刪改的 unsupported claims。

G. Execution plan：
   依「最少新增計算換最大論文證據」排序，給出下一步 command-level plan。
   不得修改 frozen protocol；若需要改，建立 protocol v2 並記錄理由。

最後只在 adjustment memo 被確認後，才開始修改程式或啟動新實驗。
```

---

## 14. 最終交接判斷

目前不應把 project 描述成：

> 我們已經證明 utility-weighted ours 全面優於所有 CL baseline。

目前更準確的描述是：

> 我們已經建立並驗證一個可學習的 budgeted evidence-acquisition environment，
> 證明 shared navigator 在 sequential tasks 下產生明顯 navigation forgetting，
> 並觀察到 replay/distillation 類方法能顯著減少 forgetting。
> Ablation 顯示 utility weighting 的增益依 budget 而變化：K=2 時相對
> uniform weighting 明顯降低 forgetting，但 K=1 時差異很小；memory size
> 與 lambda 也呈現 stability--plasticity trade-off。

這個較保守的版本有三個優點：

1. 不會被主表中的 `distill` 結果反駁。
2. 把 navigation forgetting 與評估 protocol 保留為穩健核心。
3. 讓 paired ablation CI 與 reverse-order 結果決定 `ours` 是否升級為主要
   method claim，而不是事先假設答案。
