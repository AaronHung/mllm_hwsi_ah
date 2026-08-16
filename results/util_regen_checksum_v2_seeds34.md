### Checksum: seeds{3,4} utility-axis backfill vs frozen rows (§10.2 item 3, same method as v0.33.2 §5.2)

Tolerance: `|Δ| <= 0.001` on AA, forgetting, jaccard, action_kl, per (method, seed, K).

| method | seed | K | ΔAA | Δforgetting | Δjaccard | Δaction_kl | pass |
|---|---|---|---|---|---|---|---|
| `distill` | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 4 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `distill` | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 4 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| `ours_uniform` | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |

**8/8 (method,seed,K) rows admitted** as new2 utility comparators; any FAIL row is excluded from g2 below, not silently passed.

