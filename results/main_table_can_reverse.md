# Main table — can / reverse (mean ± std over 5 seeds)


## K = 1

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.832 ± 0.032 | 0.099 ± 0.038 | -0.098 ± 0.039 | 0.093 ± 0.021 | 0.291 ± 0.028 | 0.388 ± 0.071 |
| ewc | 0.871 ± 0.016 | 0.052 ± 0.020 | -0.046 ± 0.022 | 0.173 ± 0.029 | 0.170 ± 0.026 | 0.375 ± 0.058 |
| lwf | 0.886 ± 0.011 | 0.062 ± 0.010 | -0.056 ± 0.019 | 0.230 ± 0.044 | 0.117 ± 0.019 | 0.401 ± 0.057 |
| counterfactual-teacher replay | 0.889 ± 0.048 | 0.038 ± 0.053 | -0.030 ± 0.058 | 0.261 ± 0.040 | 0.114 ± 0.060 | 0.378 ± 0.046 |
| old-policy / policy-fidelity distillation | 0.908 ± 0.016 | 0.014 ± 0.004 | -0.007 ± 0.005 | 0.351 ± 0.030 | 0.058 ± 0.010 | 0.404 ± 0.036 |
| Utility-Weighted Replay Distillation (variant) | 0.896 ± 0.042 | 0.035 ± 0.039 | -0.025 ± 0.045 | 0.328 ± 0.018 | 0.069 ± 0.025 | 0.391 ± 0.038 |
| joint-training reference | 0.918 ± 0.020 | 0.022 ± 0.009 | -0.012 ± 0.016 | 0.292 ± 0.029 | 0.058 ± 0.005 | 0.395 ± 0.069 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.064，95% CI [+0.025, +0.103]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.064，95% CI [-0.113, -0.009]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.235，95% CI [+0.222, +0.251]

## K = 2

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.860 ± 0.028 | 0.063 ± 0.017 | -0.053 ± 0.034 | 0.088 ± 0.021 | 0.117 ± 0.012 | 0.197 ± 0.052 |
| ewc | 0.878 ± 0.029 | 0.039 ± 0.026 | -0.024 ± 0.031 | 0.145 ± 0.045 | 0.080 ± 0.016 | 0.187 ± 0.046 |
| lwf | 0.904 ± 0.019 | 0.032 ± 0.016 | -0.018 ± 0.029 | 0.229 ± 0.072 | 0.048 ± 0.008 | 0.199 ± 0.041 |
| counterfactual-teacher replay | 0.881 ± 0.033 | 0.044 ± 0.050 | -0.017 ± 0.066 | 0.142 ± 0.017 | 0.077 ± 0.042 | 0.192 ± 0.018 |
| old-policy / policy-fidelity distillation | 0.903 ± 0.017 | 0.024 ± 0.007 | 0.000 ± 0.024 | 0.207 ± 0.031 | 0.024 ± 0.008 | 0.195 ± 0.043 |
| Utility-Weighted Replay Distillation (variant) | 0.891 ± 0.031 | 0.040 ± 0.037 | -0.010 ± 0.062 | 0.177 ± 0.048 | 0.048 ± 0.022 | 0.208 ± 0.028 |
| joint-training reference | 0.911 ± 0.025 | 0.020 ± 0.029 | 0.005 ± 0.045 | 0.204 ± 0.024 | 0.025 ± 0.005 | 0.199 ± 0.038 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.030，95% CI [-0.006, +0.067]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.024，95% CI [-0.047, +0.010]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.089，95% CI [+0.051, +0.138]

## K = 4

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.854 ± 0.033 | 0.062 ± 0.054 | -0.041 ± 0.070 | 0.096 ± 0.016 | 0.027 ± 0.008 | 0.089 ± 0.023 |
| ewc | 0.876 ± 0.034 | 0.035 ± 0.021 | -0.020 ± 0.034 | 0.140 ± 0.046 | 0.020 ± 0.006 | 0.100 ± 0.009 |
| lwf | 0.881 ± 0.021 | 0.035 ± 0.023 | -0.016 ± 0.036 | 0.199 ± 0.040 | 0.013 ± 0.006 | 0.076 ± 0.026 |
| counterfactual-teacher replay | 0.873 ± 0.031 | 0.038 ± 0.029 | -0.032 ± 0.024 | 0.193 ± 0.102 | 0.041 ± 0.014 | 0.073 ± 0.015 |
| old-policy / policy-fidelity distillation | 0.897 ± 0.023 | 0.032 ± 0.024 | -0.003 ± 0.017 | 0.192 ± 0.034 | 0.007 ± 0.005 | 0.084 ± 0.016 |
| Utility-Weighted Replay Distillation (variant) | 0.892 ± 0.038 | 0.028 ± 0.023 | -0.011 ± 0.031 | 0.185 ± 0.053 | 0.016 ± 0.004 | 0.074 ± 0.020 |
| joint-training reference | 0.893 ± 0.022 | 0.024 ± 0.024 | 0.003 ± 0.041 | 0.153 ± 0.048 | 0.008 ± 0.005 | 0.096 ± 0.015 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.038，95% CI [-0.007, +0.069]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.034，95% CI [-0.079, +0.007]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.089，95% CI [+0.047, +0.130]