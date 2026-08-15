# WP3 plasticity report

Zero-compute report from frozen per-seed CSVs. `A[t,t]` is the own-time new-task **balanced accuracy** at the stage in which task t first arrives; `std` is the sample standard deviation across seeds. Main and reverse rows are never pooled.

## Interpretation

No consistent own-time plasticity degradation is observed: λ=3.0 is lower than λ=1.0 in only 2/8 matched cells. Use the phrase behavior-fidelity / capability-retention trade-off.

Use the phrase **behavior-fidelity / capability-retention trade-off**; do not upgrade it to a stability–plasticity claim.

## Own-time accuracy

### main, K=1
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.828 ± 0.055 | 0.901 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.898 ± 0.036 | 0.893 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.857 ± 0.019 | 0.855 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.957 ± 0.010 | 0.953 | 5 |
| EWC parameter regularization | brca | 0.876 ± 0.085 | 0.914 | 5 |
| EWC parameter regularization | esca | 0.898 ± 0.036 | 0.893 | 5 |
| EWC parameter regularization | lung | 0.875 ± 0.015 | 0.874 | 5 |
| EWC parameter regularization | rcc | 0.943 ± 0.021 | 0.942 | 5 |
| joint-training reference | brca | 0.847 ± 0.097 | 0.905 | 5 |
| joint-training reference | esca | 0.898 ± 0.036 | 0.893 | 5 |
| joint-training reference | lung | 0.868 ± 0.004 | 0.865 | 5 |
| joint-training reference | rcc | 0.976 ± 0.014 | 0.968 | 5 |
| new-state old-policy distillation | brca | 0.863 ± 0.028 | 0.918 | 5 |
| new-state old-policy distillation | esca | 0.898 ± 0.036 | 0.893 | 5 |
| new-state old-policy distillation | lung | 0.868 ± 0.012 | 0.865 | 5 |
| new-state old-policy distillation | rcc | 0.951 ± 0.010 | 0.945 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | brca | 0.843 ± 0.053 | 0.905 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | esca | 0.898 ± 0.036 | 0.893 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | lung | 0.874 ± 0.008 | 0.872 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | rcc | 0.949 ± 0.013 | 0.945 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.822 ± 0.086 | 0.897 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.898 ± 0.036 | 0.893 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.868 ± 0.009 | 0.865 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.953 ± 0.015 | 0.947 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | brca | 0.847 ± 0.057 | 0.905 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | esca | 0.898 ± 0.036 | 0.893 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | lung | 0.874 ± 0.015 | 0.872 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | rcc | 0.960 ± 0.009 | 0.953 | 5 |
| counterfactual-teacher replay | brca | 0.835 ± 0.079 | 0.905 | 5 |
| counterfactual-teacher replay | esca | 0.898 ± 0.036 | 0.893 | 5 |
| counterfactual-teacher replay | lung | 0.868 ± 0.018 | 0.865 | 5 |
| counterfactual-teacher replay | rcc | 0.953 ± 0.013 | 0.953 | 5 |
| sequential fine-tuning | brca | 0.872 ± 0.057 | 0.914 | 5 |
| sequential fine-tuning | esca | 0.898 ± 0.036 | 0.893 | 5 |
| sequential fine-tuning | lung | 0.874 ± 0.008 | 0.872 | 5 |
| sequential fine-tuning | rcc | 0.955 ± 0.022 | 0.950 | 5 |

### main, K=2
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.882 ± 0.047 | 0.925 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.950 ± 0.028 | 0.947 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.861 ± 0.020 | 0.859 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.959 ± 0.016 | 0.955 | 5 |
| EWC parameter regularization | brca | 0.886 ± 0.048 | 0.925 | 5 |
| EWC parameter regularization | esca | 0.950 ± 0.028 | 0.947 | 5 |
| EWC parameter regularization | lung | 0.863 ± 0.012 | 0.861 | 5 |
| EWC parameter regularization | rcc | 0.942 ± 0.036 | 0.945 | 5 |
| joint-training reference | brca | 0.886 ± 0.030 | 0.918 | 5 |
| joint-training reference | esca | 0.950 ± 0.028 | 0.947 | 5 |
| joint-training reference | lung | 0.856 ± 0.012 | 0.855 | 5 |
| joint-training reference | rcc | 0.955 ± 0.012 | 0.950 | 5 |
| new-state old-policy distillation | brca | 0.877 ± 0.055 | 0.923 | 5 |
| new-state old-policy distillation | esca | 0.950 ± 0.028 | 0.947 | 5 |
| new-state old-policy distillation | lung | 0.876 ± 0.024 | 0.874 | 5 |
| new-state old-policy distillation | rcc | 0.961 ± 0.010 | 0.961 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | brca | 0.877 ± 0.066 | 0.923 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | esca | 0.950 ± 0.028 | 0.947 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | lung | 0.853 ± 0.020 | 0.851 | 5 |
| Utility-Weighted Replay Distillation (λ=0.3) | rcc | 0.957 ± 0.011 | 0.958 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.881 ± 0.068 | 0.923 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.950 ± 0.028 | 0.947 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.851 ± 0.006 | 0.848 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.959 ± 0.011 | 0.958 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | brca | 0.893 ± 0.045 | 0.929 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | esca | 0.950 ± 0.028 | 0.947 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | lung | 0.847 ± 0.025 | 0.844 | 5 |
| Utility-Weighted Replay Distillation (λ=3.0) | rcc | 0.959 ± 0.011 | 0.955 | 5 |
| counterfactual-teacher replay | brca | 0.888 ± 0.078 | 0.927 | 5 |
| counterfactual-teacher replay | esca | 0.950 ± 0.028 | 0.947 | 5 |
| counterfactual-teacher replay | lung | 0.849 ± 0.014 | 0.846 | 5 |
| counterfactual-teacher replay | rcc | 0.973 ± 0.011 | 0.971 | 5 |
| sequential fine-tuning | brca | 0.861 ± 0.064 | 0.916 | 5 |
| sequential fine-tuning | esca | 0.950 ± 0.028 | 0.947 | 5 |
| sequential fine-tuning | lung | 0.878 ± 0.017 | 0.876 | 5 |
| sequential fine-tuning | rcc | 0.965 ± 0.009 | 0.963 | 5 |

### main, K=4
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.892 ± 0.008 | 0.910 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.896 ± 0.038 | 0.893 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.857 ± 0.023 | 0.855 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.951 ± 0.012 | 0.950 | 5 |
| EWC parameter regularization | brca | 0.815 ± 0.128 | 0.886 | 5 |
| EWC parameter regularization | esca | 0.896 ± 0.038 | 0.893 | 5 |
| EWC parameter regularization | lung | 0.863 ± 0.011 | 0.861 | 5 |
| EWC parameter regularization | rcc | 0.933 ± 0.020 | 0.939 | 5 |
| joint-training reference | brca | 0.828 ± 0.065 | 0.908 | 5 |
| joint-training reference | esca | 0.896 ± 0.038 | 0.893 | 5 |
| joint-training reference | lung | 0.843 ± 0.020 | 0.842 | 5 |
| joint-training reference | rcc | 0.945 ± 0.024 | 0.955 | 5 |
| new-state old-policy distillation | brca | 0.854 ± 0.113 | 0.899 | 5 |
| new-state old-policy distillation | esca | 0.896 ± 0.038 | 0.893 | 5 |
| new-state old-policy distillation | lung | 0.858 ± 0.028 | 0.857 | 5 |
| new-state old-policy distillation | rcc | 0.958 ± 0.014 | 0.961 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.887 ± 0.015 | 0.920 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.896 ± 0.038 | 0.893 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.853 ± 0.015 | 0.851 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.937 ± 0.031 | 0.945 | 5 |
| counterfactual-teacher replay | brca | 0.883 ± 0.046 | 0.914 | 5 |
| counterfactual-teacher replay | esca | 0.896 ± 0.038 | 0.893 | 5 |
| counterfactual-teacher replay | lung | 0.843 ± 0.042 | 0.842 | 5 |
| counterfactual-teacher replay | rcc | 0.937 ± 0.043 | 0.945 | 5 |
| sequential fine-tuning | brca | 0.864 ± 0.086 | 0.901 | 5 |
| sequential fine-tuning | esca | 0.896 ± 0.038 | 0.893 | 5 |
| sequential fine-tuning | lung | 0.862 ± 0.011 | 0.859 | 5 |
| sequential fine-tuning | rcc | 0.950 ± 0.011 | 0.950 | 5 |

### reverse, K=1
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.895 ± 0.053 | 0.927 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.936 ± 0.047 | 0.933 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.861 ± 0.026 | 0.859 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.959 ± 0.011 | 0.955 | 5 |
| EWC parameter regularization | brca | 0.895 ± 0.053 | 0.927 | 5 |
| EWC parameter regularization | esca | 0.909 ± 0.039 | 0.907 | 5 |
| EWC parameter regularization | lung | 0.865 ± 0.016 | 0.863 | 5 |
| EWC parameter regularization | rcc | 0.954 ± 0.029 | 0.947 | 5 |
| joint-training reference | brca | 0.895 ± 0.053 | 0.927 | 5 |
| joint-training reference | esca | 0.975 ± 0.034 | 0.973 | 5 |
| joint-training reference | lung | 0.878 ± 0.009 | 0.876 | 5 |
| joint-training reference | rcc | 0.960 ± 0.014 | 0.953 | 5 |
| new-state old-policy distillation | brca | 0.895 ± 0.053 | 0.927 | 5 |
| new-state old-policy distillation | esca | 0.963 ± 0.034 | 0.960 | 5 |
| new-state old-policy distillation | lung | 0.882 ± 0.009 | 0.880 | 5 |
| new-state old-policy distillation | rcc | 0.972 ± 0.013 | 0.968 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.895 ± 0.053 | 0.927 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.948 ± 0.056 | 0.947 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.854 ± 0.021 | 0.853 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.962 ± 0.024 | 0.958 | 5 |
| counterfactual-teacher replay | brca | 0.895 ± 0.053 | 0.927 | 5 |
| counterfactual-teacher replay | esca | 0.936 ± 0.047 | 0.933 | 5 |
| counterfactual-teacher replay | lung | 0.859 ± 0.024 | 0.857 | 5 |
| counterfactual-teacher replay | rcc | 0.957 ± 0.013 | 0.953 | 5 |
| sequential fine-tuning | brca | 0.895 ± 0.053 | 0.927 | 5 |
| sequential fine-tuning | esca | 0.896 ± 0.038 | 0.893 | 5 |
| sequential fine-tuning | lung | 0.867 ± 0.022 | 0.865 | 5 |
| sequential fine-tuning | rcc | 0.962 ± 0.015 | 0.955 | 5 |

### reverse, K=2
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.856 ± 0.071 | 0.914 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.923 ± 0.032 | 0.920 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.863 ± 0.025 | 0.861 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.969 ± 0.007 | 0.968 | 5 |
| EWC parameter regularization | brca | 0.856 ± 0.071 | 0.914 | 5 |
| EWC parameter regularization | esca | 0.912 ± 0.034 | 0.907 | 5 |
| EWC parameter regularization | lung | 0.852 ± 0.019 | 0.851 | 5 |
| EWC parameter regularization | rcc | 0.961 ± 0.009 | 0.961 | 5 |
| joint-training reference | brca | 0.856 ± 0.071 | 0.914 | 5 |
| joint-training reference | esca | 0.950 ± 0.028 | 0.947 | 5 |
| joint-training reference | lung | 0.858 ± 0.019 | 0.857 | 5 |
| joint-training reference | rcc | 0.965 ± 0.009 | 0.966 | 5 |
| new-state old-policy distillation | brca | 0.856 ± 0.071 | 0.914 | 5 |
| new-state old-policy distillation | esca | 0.975 ± 0.034 | 0.973 | 5 |
| new-state old-policy distillation | lung | 0.873 ± 0.024 | 0.872 | 5 |
| new-state old-policy distillation | rcc | 0.967 ± 0.008 | 0.966 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.856 ± 0.071 | 0.914 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.923 ± 0.032 | 0.920 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.853 ± 0.028 | 0.851 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.959 ± 0.005 | 0.958 | 5 |
| counterfactual-teacher replay | brca | 0.856 ± 0.071 | 0.914 | 5 |
| counterfactual-teacher replay | esca | 0.900 ± 0.034 | 0.893 | 5 |
| counterfactual-teacher replay | lung | 0.852 ± 0.032 | 0.851 | 5 |
| counterfactual-teacher replay | rcc | 0.969 ± 0.013 | 0.968 | 5 |
| sequential fine-tuning | brca | 0.856 ± 0.071 | 0.914 | 5 |
| sequential fine-tuning | esca | 0.911 ± 0.037 | 0.907 | 5 |
| sequential fine-tuning | lung | 0.865 ± 0.011 | 0.863 | 5 |
| sequential fine-tuning | rcc | 0.969 ± 0.017 | 0.968 | 5 |

### reverse, K=4
| setting | task | A[t,t] | raw accuracy | n |
|---|---|---:|---:|---:|
| old-policy / policy-fidelity distillation | brca | 0.865 ± 0.082 | 0.903 | 5 |
| old-policy / policy-fidelity distillation | esca | 0.950 ± 0.028 | 0.947 | 5 |
| old-policy / policy-fidelity distillation | lung | 0.853 ± 0.011 | 0.851 | 5 |
| old-policy / policy-fidelity distillation | rcc | 0.928 ± 0.048 | 0.937 | 5 |
| EWC parameter regularization | brca | 0.865 ± 0.082 | 0.903 | 5 |
| EWC parameter regularization | esca | 0.921 ± 0.076 | 0.920 | 5 |
| EWC parameter regularization | lung | 0.836 ± 0.025 | 0.834 | 5 |
| EWC parameter regularization | rcc | 0.942 ± 0.024 | 0.945 | 5 |
| joint-training reference | brca | 0.865 ± 0.082 | 0.903 | 5 |
| joint-training reference | esca | 0.909 ± 0.039 | 0.907 | 5 |
| joint-training reference | lung | 0.846 ± 0.024 | 0.844 | 5 |
| joint-training reference | rcc | 0.945 ± 0.034 | 0.950 | 5 |
| new-state old-policy distillation | brca | 0.865 ± 0.082 | 0.903 | 5 |
| new-state old-policy distillation | esca | 0.909 ± 0.035 | 0.907 | 5 |
| new-state old-policy distillation | lung | 0.841 ± 0.027 | 0.838 | 5 |
| new-state old-policy distillation | rcc | 0.958 ± 0.022 | 0.963 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | brca | 0.865 ± 0.082 | 0.903 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | esca | 0.963 ± 0.034 | 0.960 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | lung | 0.848 ± 0.012 | 0.846 | 5 |
| Utility-Weighted Replay Distillation (λ=1.0) | rcc | 0.927 ± 0.015 | 0.929 | 5 |
| counterfactual-teacher replay | brca | 0.865 ± 0.082 | 0.903 | 5 |
| counterfactual-teacher replay | esca | 0.923 ± 0.032 | 0.920 | 5 |
| counterfactual-teacher replay | lung | 0.848 ± 0.017 | 0.846 | 5 |
| counterfactual-teacher replay | rcc | 0.953 ± 0.024 | 0.961 | 5 |
| sequential fine-tuning | brca | 0.865 ± 0.082 | 0.903 | 5 |
| sequential fine-tuning | esca | 0.896 ± 0.038 | 0.893 | 5 |
| sequential fine-tuning | lung | 0.828 ± 0.034 | 0.825 | 5 |
| sequential fine-tuning | rcc | 0.950 ± 0.021 | 0.955 | 5 |

