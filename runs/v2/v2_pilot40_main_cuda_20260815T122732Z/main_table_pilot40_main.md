# Main table — pilot40 / main (mean ± std over 5 seeds)


## K = 1

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| `seqft` — sequential fine-tuning | 0.717 ± 0.112 | 0.100 ± 0.149 | -0.067 ± 0.190 | 0.067 ± 0.091 | 0.164 ± 0.079 | -0.280 ± 0.502 |
| `ewc` — EWC parameter regularization | 0.750 ± 0.102 | 0.067 ± 0.091 | -0.033 ± 0.139 | 0.100 ± 0.149 | 0.151 ± 0.081 | -0.095 ± 0.529 |
| `lwf` — new-state old-policy distillation | 0.750 ± 0.083 | 0.033 ± 0.075 | -0.033 ± 0.075 | 0.267 ± 0.253 | 0.066 ± 0.041 | -0.205 ± 0.378 |
| `replay` — counterfactual-teacher replay | 0.800 ± 0.126 | 0.067 ± 0.091 | -0.033 ± 0.139 | 0.200 ± 0.139 | 0.073 ± 0.035 | -0.436 ± 0.104 |
| `distill` — old-policy / policy-fidelity distillation | 0.783 ± 0.095 | 0.033 ± 0.075 | 0.033 ± 0.139 | 0.367 ± 0.075 | 0.042 ± 0.011 | -0.334 ± 0.594 |
| `ours` — Utility-Weighted Replay Distillation | 0.750 ± 0.118 | 0.067 ± 0.149 | -0.067 ± 0.149 | 0.300 ± 0.139 | 0.038 ± 0.013 | -0.595 ± 0.476 |
| `joint` — joint-training reference | 0.800 ± 0.095 | 0.033 ± 0.075 | 0.000 ± 0.118 | 0.233 ± 0.091 | 0.065 ± 0.025 | -0.431 ± 0.620 |

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（AA，higher better）：+0.033，95% CI [+0.000, +0.100]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（forgetting，lower better）：-0.033，95% CI [-0.100, +0.000]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（jaccard，higher better）：+0.233，95% CI [+0.167, +0.300]

## K = 2

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| `seqft` — sequential fine-tuning | 0.733 ± 0.160 | 0.100 ± 0.224 | -0.033 ± 0.274 | 0.033 ± 0.030 | 0.053 ± 0.013 | -0.195 ± 0.324 |
| `ewc` — EWC parameter regularization | 0.750 ± 0.102 | 0.100 ± 0.224 | -0.000 ± 0.312 | 0.033 ± 0.030 | 0.050 ± 0.012 | -0.197 ± 0.210 |
| `lwf` — new-state old-policy distillation | 0.717 ± 0.095 | 0.000 ± 0.000 | 0.067 ± 0.091 | 0.100 ± 0.082 | 0.019 ± 0.004 | -0.205 ± 0.373 |
| `replay` — counterfactual-teacher replay | 0.667 ± 0.186 | 0.167 ± 0.204 | -0.100 ± 0.279 | 0.111 ± 0.088 | 0.033 ± 0.017 | -0.353 ± 0.268 |
| `distill` — old-policy / policy-fidelity distillation | 0.767 ± 0.070 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.056 | 0.012 ± 0.003 | -0.346 ± 0.494 |
| `ours` — Utility-Weighted Replay Distillation | 0.733 ± 0.137 | 0.100 ± 0.149 | -0.067 ± 0.190 | 0.156 ± 0.061 | 0.011 ± 0.002 | -0.406 ± 0.568 |
| `joint` — joint-training reference | 0.800 ± 0.095 | 0.033 ± 0.075 | 0.067 ± 0.149 | 0.167 ± 0.088 | 0.031 ± 0.016 | -0.162 ± 0.339 |

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（AA，higher better）：-0.000，95% CI [-0.100, +0.100]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（forgetting，lower better）：+0.000，95% CI [-0.100, +0.100]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（jaccard，higher better）：+0.122，95% CI [+0.089, +0.156]

## K = 4

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| `seqft` — sequential fine-tuning | 0.750 ± 0.102 | 0.067 ± 0.091 | -0.033 ± 0.139 | 0.029 ± 0.011 | 0.037 ± 0.011 | -0.054 ± 0.156 |
| `ewc` — EWC parameter regularization | 0.850 ± 0.037 | 0.033 ± 0.075 | 0.067 ± 0.190 | 0.076 ± 0.055 | 0.033 ± 0.014 | -0.045 ± 0.179 |
| `lwf` — new-state old-policy distillation | 0.767 ± 0.091 | 0.033 ± 0.075 | 0.067 ± 0.253 | 0.110 ± 0.067 | 0.010 ± 0.003 | -0.082 ± 0.197 |
| `replay` — counterfactual-teacher replay | 0.767 ± 0.109 | 0.067 ± 0.091 | -0.033 ± 0.139 | 0.087 ± 0.014 | 0.019 ± 0.004 | -0.370 ± 0.242 |
| `distill` — old-policy / policy-fidelity distillation | 0.700 ± 0.162 | 0.067 ± 0.091 | -0.033 ± 0.139 | 0.126 ± 0.073 | 0.007 ± 0.004 | -0.184 ± 0.176 |
| `ours` — Utility-Weighted Replay Distillation | 0.733 ± 0.037 | 0.033 ± 0.075 | 0.000 ± 0.118 | 0.149 ± 0.052 | 0.006 ± 0.002 | -0.265 ± 0.260 |
| `joint` — joint-training reference | 0.800 ± 0.095 | 0.000 ± 0.000 | 0.100 ± 0.149 | 0.079 ± 0.029 | 0.012 ± 0.005 | 0.038 ± 0.164 |

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（AA，higher better）：-0.017，95% CI [-0.067, +0.033]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（forgetting，lower better）：-0.033，95% CI [-0.100, +0.000]

- `Utility-Weighted Replay Distillation` − `sequential fine-tuning`（jaccard，higher better）：+0.120，95% CI [+0.084, +0.169]