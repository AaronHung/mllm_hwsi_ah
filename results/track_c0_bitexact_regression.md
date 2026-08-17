# Track C0 pre-launch bit-exactness regression (docs/track_c0.md 5)

Source (new run): `runs/v2/c0_bitexact_20260817T013843Z/cl_main_can_main_c0_bitexact_20260817T013843Z.csv`  
Seeds: [0] | K: [1]  
Rule: **exact equality** on `bal_acc`, `acc`, `jaccard`, `action_kl`, `sel_utility`, `eps_optimal_mass`, `normalized_regret`, `random_ref`, `n_test`, keyed on `seed`, `K`, `stage`, `task_idx`. This is stricter than the `util_regen` `|Δ| <= 0.001` checksum by design (docs/method_gate_v034.md §2.5/§2.6).

| new method | frozen method | rows compared | identical | mismatched | verdict |
|---|---|---|---|---|---|
| `eq_pres` | `eq_pres` | 10 | 10 | 0 | **PASS (bit-identical)** |
| `ours_uniform` | `ours_uniform` | 10 | 10 | 0 | **PASS (bit-identical)** |

## Verdict

**PASS — 20/20 rows bit-identical.**
