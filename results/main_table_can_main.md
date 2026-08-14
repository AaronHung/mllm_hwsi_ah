# Main table — can / main (mean ± std over 5 seeds)


## K = 1

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.858 ± 0.028 | 0.070 ± 0.036 | -0.055 ± 0.033 | 0.051 ± 0.029 | 0.322 ± 0.038 | 0.282 ± 0.091 |
| ewc | 0.864 ± 0.020 | 0.069 ± 0.028 | -0.045 ± 0.030 | 0.118 ± 0.070 | 0.198 ± 0.018 | 0.426 ± 0.047 |
| lwf | 0.896 ± 0.027 | 0.018 ± 0.016 | 0.001 ± 0.018 | 0.136 ± 0.033 | 0.085 ± 0.011 | 0.472 ± 0.063 |
| counterfactual-teacher replay | 0.893 ± 0.033 | 0.016 ± 0.010 | 0.006 ± 0.008 | 0.179 ± 0.032 | 0.104 ± 0.013 | 0.387 ± 0.093 |
| old-policy / policy-fidelity distillation | 0.892 ± 0.025 | 0.015 ± 0.013 | 0.009 ± 0.019 | 0.296 ± 0.050 | 0.058 ± 0.009 | 0.375 ± 0.160 |
| Utility-Weighted Replay Distillation (variant) | 0.888 ± 0.031 | 0.011 ± 0.012 | 0.004 ± 0.021 | 0.259 ± 0.030 | 0.069 ± 0.012 | 0.411 ± 0.084 |
| joint-training reference | 0.905 ± 0.034 | 0.017 ± 0.015 | 0.010 ± 0.014 | 0.242 ± 0.027 | 0.076 ± 0.015 | 0.506 ± 0.022 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.030，95% CI [-0.002, +0.056]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.059，95% CI [-0.093, -0.027]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.208，95% CI [+0.189, +0.229]

## K = 2

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.883 ± 0.027 | 0.048 ± 0.038 | -0.040 ± 0.040 | 0.099 ± 0.027 | 0.111 ± 0.013 | 0.208 ± 0.066 |
| ewc | 0.903 ± 0.005 | 0.021 ± 0.009 | -0.010 ± 0.021 | 0.111 ± 0.032 | 0.087 ± 0.007 | 0.196 ± 0.088 |
| lwf | 0.898 ± 0.016 | 0.031 ± 0.017 | -0.024 ± 0.015 | 0.171 ± 0.043 | 0.032 ± 0.002 | 0.173 ± 0.057 |
| counterfactual-teacher replay | 0.898 ± 0.029 | 0.025 ± 0.012 | -0.022 ± 0.010 | 0.128 ± 0.023 | 0.055 ± 0.010 | 0.158 ± 0.034 |
| old-policy / policy-fidelity distillation | 0.902 ± 0.025 | 0.021 ± 0.022 | -0.015 ± 0.027 | 0.181 ± 0.029 | 0.020 ± 0.003 | 0.181 ± 0.060 |
| Utility-Weighted Replay Distillation (variant) | 0.897 ± 0.020 | 0.021 ± 0.021 | -0.017 ± 0.020 | 0.144 ± 0.011 | 0.028 ± 0.006 | 0.136 ± 0.034 |
| joint-training reference | 0.914 ± 0.014 | 0.017 ± 0.019 | 0.004 ± 0.026 | 0.169 ± 0.046 | 0.028 ± 0.005 | 0.198 ± 0.054 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.014，95% CI [-0.008, +0.039]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.028，95% CI [-0.059, -0.005]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.045，95% CI [+0.031, +0.058]

## K = 4

| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |
|---|---|---|---|---|---|---|
| seqft | 0.858 ± 0.014 | 0.060 ± 0.023 | -0.046 ± 0.022 | 0.079 ± 0.019 | 0.020 ± 0.002 | 0.052 ± 0.045 |
| ewc | 0.860 ± 0.033 | 0.049 ± 0.042 | -0.022 ± 0.039 | 0.102 ± 0.012 | 0.018 ± 0.003 | 0.052 ± 0.022 |
| lwf | 0.888 ± 0.034 | 0.032 ± 0.017 | -0.005 ± 0.018 | 0.128 ± 0.048 | 0.007 ± 0.001 | 0.063 ± 0.039 |
| counterfactual-teacher replay | 0.886 ± 0.012 | 0.029 ± 0.012 | -0.005 ± 0.024 | 0.135 ± 0.043 | 0.028 ± 0.004 | 0.047 ± 0.026 |
| old-policy / policy-fidelity distillation | 0.896 ± 0.018 | 0.024 ± 0.027 | -0.004 ± 0.035 | 0.166 ± 0.037 | 0.004 ± 0.001 | 0.061 ± 0.013 |
| Utility-Weighted Replay Distillation (variant) | 0.898 ± 0.011 | 0.013 ± 0.011 | 0.006 ± 0.020 | 0.143 ± 0.013 | 0.010 ± 0.001 | 0.049 ± 0.049 |
| joint-training reference | 0.890 ± 0.022 | 0.019 ± 0.015 | 0.015 ± 0.030 | 0.131 ± 0.029 | 0.006 ± 0.002 | 0.097 ± 0.026 |

- Utility-Weighted Replay Distillation − seqft（AA，higher better）：+0.040，95% CI [+0.024, +0.061]

- Utility-Weighted Replay Distillation − seqft（forgetting，lower better）：-0.047，95% CI [-0.065, -0.024]

- Utility-Weighted Replay Distillation − seqft（jaccard，higher better）：+0.063，95% CI [+0.052, +0.075]