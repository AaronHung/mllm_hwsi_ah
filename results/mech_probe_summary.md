# WP4 mechanism probe summary

Seed 0, `K=1`, `can_dataset`; separate fresh sequences for main and reverse orders. Methods are `seqft`, old-policy / policy-fidelity distillation (`distill`), and Utility-Weighted Replay Distillation (`ours`).

Teacher gains are min–max normalized **per state before any derived utility aggregation**. ε is fixed at 0.05, so the ε-equivalent set contains candidates with normalized teacher gain ≥ 0.95. Utility regret is `1 - normalized teacher gain(selected action)`; ε-optimal mass is the learned policy probability assigned to that set. Action drift is `KL(policy at source-task boundary || current policy)`.

Resolved device: `mps`; torch `2.12.0`.

## Final-stage state aggregates

| order | method | source task | n | ε-set size | ε-optimal mass | normalized utility regret | action drift | drift–regret r |
|---|---|---|---:|---:|---:|---:|---:|---:|
| main | `distill` | brca | 93 | 14.312 | 0.739 | 0.095 | 0.000 | nan |
| main | `distill` | esca | 15 | 18.600 | 0.741 | 0.093 | 0.059 | 0.209 |
| main | `distill` | lung | 95 | 13.979 | 0.712 | 0.118 | 0.052 | 0.008 |
| main | `distill` | rcc | 76 | 22.197 | 0.816 | 0.119 | 0.036 | 0.114 |
| main | `ours` | brca | 93 | 14.312 | 0.736 | 0.129 | 0.000 | nan |
| main | `ours` | esca | 15 | 18.600 | 0.766 | 0.049 | 0.074 | 0.208 |
| main | `ours` | lung | 95 | 13.979 | 0.722 | 0.142 | 0.066 | 0.225 |
| main | `ours` | rcc | 76 | 22.197 | 0.822 | 0.104 | 0.032 | -0.089 |
| main | `seqft` | brca | 93 | 14.312 | 0.744 | 0.070 | 0.000 | nan |
| main | `seqft` | esca | 15 | 18.600 | 0.671 | 0.044 | 0.200 | 0.630 |
| main | `seqft` | lung | 95 | 13.979 | 0.501 | 0.107 | 0.437 | 0.178 |
| main | `seqft` | rcc | 76 | 22.197 | 0.750 | 0.063 | 0.336 | 0.064 |
| reverse | `distill` | brca | 93 | 13.237 | 0.700 | 0.077 | 0.074 | -0.003 |
| reverse | `distill` | esca | 15 | 18.933 | 0.785 | 0.075 | 0.000 | nan |
| reverse | `distill` | lung | 95 | 13.400 | 0.692 | 0.127 | 0.031 | -0.033 |
| reverse | `distill` | rcc | 76 | 22.355 | 0.807 | 0.075 | 0.040 | 0.080 |
| reverse | `ours` | brca | 93 | 13.237 | 0.708 | 0.075 | 0.089 | 0.168 |
| reverse | `ours` | esca | 15 | 18.933 | 0.770 | 0.057 | 0.000 | nan |
| reverse | `ours` | lung | 95 | 13.400 | 0.674 | 0.103 | 0.039 | 0.045 |
| reverse | `ours` | rcc | 76 | 22.355 | 0.817 | 0.104 | 0.034 | -0.059 |
| reverse | `seqft` | brca | 93 | 13.237 | 0.594 | 0.109 | 0.380 | 0.457 |
| reverse | `seqft` | esca | 15 | 18.933 | 0.766 | 0.053 | 0.000 | nan |
| reverse | `seqft` | lung | 95 | 13.400 | 0.522 | 0.133 | 0.384 | 0.221 |
| reverse | `seqft` | rcc | 76 | 22.355 | 0.781 | 0.075 | 0.225 | 0.158 |

## Observed pattern

Across both orders, `seqft` has visibly larger final-stage action drift on earlier tasks than `distill` or `ours`, while replay and distillation retain substantial ε-optimal probability mass. The ε-equivalent sets are broad (many candidates are near-optimal), and the drift–regret relationship is not uniformly strong for the preservation variants. This supports a behavior-preservation mechanism, but not a claim that utility weighting is universally necessary or sufficient.

## Interpretation gate

This is mechanism analysis, not an EUP gate. The K=1 result is stable enough to support the existing analysis-centric behavior-preservation claim, but the broad ε-equivalent sets and weak/variable drift–regret correlations do not justify extending to `K=2` before the science freeze. Do not add a new method.

Raw per-state records and stage-boundary checkpoints are under the fresh run directory passed to `mechanism_probe.py`.
