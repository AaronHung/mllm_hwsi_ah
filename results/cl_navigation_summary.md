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