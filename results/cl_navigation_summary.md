# Gate 2/3 — navigation forgetting 與緩解（K=4）

```
         t1_acc_before  t1_acc_after  forgetting  t2_acc  action_kl_drift  selection_jaccard
method                                                                                      
distill          0.722         0.778      -0.056   0.722            0.009              0.211
seqft            0.722         0.722       0.000   0.722            0.021              0.061
```

- `seqft` = Gate 2 sequential fine-tuning baseline
- `distill` = Gate 3 utility-weighted policy distillation + state replay
- evaluator 凍結（navigation-only protocol），T1 掉分只可能來自導航行為改變

**Gate 2（觀察到 navigation forgetting）：seqft forgetting = +0.000 → WEAK/FAIL**
**Gate 3（forgetting 明顯下降）：distill forgetting = -0.056（Δ = +0.056）→ PASS**

## 正確判讀（pilot 規模的結論）

- **行為層遺忘明確**：seqft 後舊任務 selection overlap 崩到 0.061（幾乎完全換地方看）；
  distill 拉回平均 0.211（per-seed 0.10/0.26/0.27，恢復本身有 seed 變異），
  action KL drift 減半（0.021 → 0.009）。
- **accuracy 層遺忘未顯形（= 0.000）**，原因有二：
  1. 訊號冗餘（實驗 B：random K=4 已 78%，亂看也看得到證據）；
  2. 每任務 test 僅 6 張、一張 = 16.7 pp，統計功效不足
     （seed 2 的 T1 起始 acc 僅 0.333，即此雜訊水準）。
- 不可宣稱「觀察到 forgetting 導致效能下降」；正確說法是
  「行為層遺忘劇烈、在 pilot 規模尚未轉化為 accuracy 下降」。

## 完整版實驗設計（目標：讓遺忘在 accuracy 層顯形）

1. 預算壓到 K=1–2（Gate 1 顯示 K=1 策略差距最大、冗餘救不了）。
2. cohort 擴到 80 張或 can_dataset（每任務 test 拉到有統計意義）。
3. 任務序列從 2 個擴到 3–4 個（遺忘會累積）。
4. 每組 5+ seeds，報 confidence interval。