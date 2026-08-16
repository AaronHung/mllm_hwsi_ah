# v0.34 pre-launch bit-exactness regression (docs/method_gate_v034.md 2.6)

Source (new run): `runs/v2/v034_bitexact_20260816T153845Z/cl_main_can_main_v034_bitexact_20260816T153845Z.csv`  
Seeds: [0] | K: [1]  
Rule: **exact equality** on `bal_acc`, `acc`, `jaccard`, `action_kl`, `sel_utility`, `eps_optimal_mass`, `normalized_regret`, `random_ref`, `n_test`, keyed on `seed`, `K`, `stage`, `task_idx`. This is stricter than the `util_regen` `|Δ| <= 0.001` checksum by design (docs/method_gate_v034.md §2.5/§2.6).

| new method | frozen method | rows compared | identical | mismatched | verdict |
|---|---|---|---|---|---|
| `eq_pres` | `eq_pres` | 10 | 10 | 0 | **PASS (bit-identical)** |
| `ours_uniform` | `ours_uniform` | 10 | 10 | 0 | **PASS (bit-identical)** |

## Verdict

**PASS — 20/20 rows bit-identical.**

## Provenance (appended by hand, 2026-08-16 CST)

- The run was launched against commit `f8514ae` (the v0.34 implementation
  commit); `runs/v2/v034_bitexact_20260816T153845Z/metadata.json` records
  that commit, the backend (`mps`), `torch 2.12.0`, the host, and the exact
  command, per `docs/compute_policy.md` rule 5.
- Two later commits touch code before the development gate launches:
  `2e38fa3` (verdict/checker reporting only, no training code) and `8b55ba8`
  (an entry-guard in `train_navigator_cl` that raises when `use_arbiter` is
  combined with `ewc_terms`). Neither can execute on the frozen path — the
  guard's condition is false whenever `use_arbiter=False`, which is the case
  for both configs above — so this result carries to the launched commit.
  Stated rather than assumed, because "the frozen path is untouched" is
  exactly the claim under test.
- **This regression is the fast tripwire, not the whole proof.** The
  §2.5 instrumentation-only batch queued behind the development gate
  re-checks bit-exactness for `eq_pres` across all 3 seeds x K∈{1,2,4}
  against the launched commit, and its result is reported separately in
  `results/eq_pres_diag_bitexact_check.md`.
- Two configs were chosen deliberately: `eq_pres` exercises the `L_eq`
  branch of the memory objective and `ours_uniform` the old-policy-KL
  branch, so both code paths the arbiter had to be threaded around are
  covered.
- The `ours_uniform` comparison uses the checksum-passed `util_regen`
  MPS backfill for `eps_optimal_mass` / `normalized_regret`: the frozen
  `ablA1_s0.csv` predates those two columns entirely (they are absent, not
  merely stale), which is the same data gap that motivated v0.33.2 §5.2.
  All other columns come from the frozen Protocol-v1 CSV directly.
