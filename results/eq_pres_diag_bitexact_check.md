# `eq_pres_diag` bit-exactness check (docs/method_gate_v034.md §2.5 condition 3)

Rule: exact equality on `bal_acc`, `acc`, `jaccard`, `action_kl`, `sel_utility`, `eps_optimal_mass`, `normalized_regret`, `random_ref`, `n_test`. Any mismatch discards the entire batch — no partial admission.

- rows compared: **90**
- identical: **90**
- mismatched: **0**
- not comparable: none — every metric column had a frozen value

**ADMITTED — bit-identical to the frozen `eq_pres` rows.** The conflict statistics from this batch are a direct measurement of the diagnosed config itself.

