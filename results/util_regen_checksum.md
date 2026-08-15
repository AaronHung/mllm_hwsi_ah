# util_regen checksum (v0.33.2 item 3)

Tolerance: `|Δ| <= 0.001` on AA, forgetting, jaccard, action_kl, per (method, seed, K), frozen Protocol-v1 rows vs. probe-only regeneration `cl_main_can_main_util_regen_20260815_mps.csv`. The frozen `can` grid was run on Mac MPS (verified against `logs/cl_main_s*.log` / `logs/abl_ablA1_s*.log`); the regen run must use the same backend for this checksum to be meaningful.

| method | seed | K | ΔAA | Δforgetting | Δjaccard | Δaction_kl | pass |
|---|---|---|---|---|---|---|---|
| `distill` | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |

**Overall: ALL PASS** (12/12 (method,seed,K) rows admitted as g2 utility comparators).

## Admitted utility-axis comparator values (final stage, old tasks only, macro-average across old tasks)

| method | seed | K | eps_optimal_mass (higher better) | normalized_regret (lower better) |
|---|---|---|---|---|
| `distill` | 0 | 1 | 0.7562 | 0.1032 |
| `distill` | 0 | 2 | 0.7174 | 0.1088 |
| `distill` | 1 | 1 | 0.7421 | 0.1054 |
| `distill` | 1 | 2 | 0.6846 | 0.1185 |
| `distill` | 2 | 1 | 0.7442 | 0.1014 |
| `distill` | 2 | 2 | 0.7305 | 0.1074 |
| `ours_uniform` | 0 | 1 | 0.7718 | 0.1017 |
| `ours_uniform` | 0 | 2 | 0.7218 | 0.1092 |
| `ours_uniform` | 1 | 1 | 0.7513 | 0.1041 |
| `ours_uniform` | 1 | 2 | 0.6959 | 0.1149 |
| `ours_uniform` | 2 | 1 | 0.7486 | 0.1045 |
| `ours_uniform` | 2 | 2 | 0.7276 | 0.1085 |
