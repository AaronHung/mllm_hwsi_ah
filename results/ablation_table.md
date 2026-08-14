# Ablation / mechanism analysis

Main-order can_dataset, mean ± std over 5 seeds.

## Aggregate metrics

| setting | K | AA | forgetting ↓ | Jaccard ↑ | action-KL ↓ | utility ↑ |
|---|---:|---:|---:|---:|---:|---:|
| distill baseline | 1 | 0.892 ± 0.025 | 0.015 ± 0.013 | 0.296 ± 0.050 | 0.059 ± 0.009 | 0.375 ± 0.160 |
| distill baseline | 2 | 0.902 ± 0.025 | 0.021 ± 0.022 | 0.181 ± 0.029 | 0.020 ± 0.003 | 0.181 ± 0.060 |
| distill baseline | 4 | 0.896 ± 0.018 | 0.024 ± 0.027 | 0.166 ± 0.037 | 0.004 ± 0.001 | 0.061 ± 0.013 |
| ours (memory 512, λ=1) | 1 | 0.888 ± 0.031 | 0.011 ± 0.012 | 0.259 ± 0.030 | 0.069 ± 0.012 | 0.411 ± 0.084 |
| ours (memory 512, λ=1) | 2 | 0.897 ± 0.020 | 0.021 ± 0.021 | 0.144 ± 0.011 | 0.028 ± 0.006 | 0.136 ± 0.034 |
| ours (memory 512, λ=1) | 4 | 0.898 ± 0.011 | 0.013 ± 0.011 | 0.143 ± 0.013 | 0.010 ± 0.001 | 0.049 ± 0.049 |
| uniform utility weight (ours_uniform) | 1 | 0.904 ± 0.030 | 0.012 ± 0.015 | 0.253 ± 0.026 | 0.068 ± 0.010 | 0.387 ± 0.172 |
| uniform utility weight (ours_uniform) | 2 | 0.887 ± 0.007 | 0.044 ± 0.016 | 0.146 ± 0.021 | 0.030 ± 0.005 | 0.092 ± 0.063 |
| memory cap 128 | 1 | 0.885 ± 0.029 | 0.032 ± 0.025 | 0.192 ± 0.029 | 0.097 ± 0.009 | 0.233 ± 0.189 |
| memory cap 128 | 2 | 0.891 ± 0.024 | 0.031 ± 0.022 | 0.121 ± 0.011 | 0.047 ± 0.008 | 0.085 ± 0.069 |
| memory cap 2048 | 1 | 0.898 ± 0.030 | 0.012 ± 0.011 | 0.279 ± 0.031 | 0.066 ± 0.013 | 0.447 ± 0.090 |
| memory cap 2048 | 2 | 0.907 ± 0.012 | 0.010 ± 0.014 | 0.189 ± 0.021 | 0.020 ± 0.005 | 0.190 ± 0.037 |
| lambda 0.3 | 1 | 0.896 ± 0.027 | 0.014 ± 0.010 | 0.202 ± 0.031 | 0.089 ± 0.013 | 0.390 ± 0.110 |
| lambda 0.3 | 2 | 0.905 ± 0.014 | 0.021 ± 0.017 | 0.124 ± 0.011 | 0.040 ± 0.006 | 0.115 ± 0.070 |
| lambda 3.0 | 1 | 0.903 ± 0.020 | 0.014 ± 0.007 | 0.347 ± 0.033 | 0.047 ± 0.011 | 0.413 ± 0.154 |
| lambda 3.0 | 2 | 0.900 ± 0.020 | 0.026 ± 0.014 | 0.201 ± 0.026 | 0.016 ± 0.005 | 0.141 ± 0.058 |

## Interpretation guide

- `ours_main` is the reference configuration: utility weighting, replay, memory cap 512, λ=1.
- `ablA1` tests uniform weighting with the same replay/distillation structure.
- `ablA2a` / `ablA2b` test memory caps 128 / 2048.
- `ablA3a` / `ablA3b` test λ=0.3 / 3.0.
- `distill baseline` is the main-grid distill-only baseline; it is not identical to `ablA1`.
