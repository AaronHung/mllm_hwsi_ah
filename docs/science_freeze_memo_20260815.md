# Science-freeze memo after WP1–WP4

**Parent direction:** v0.32 — Continual Budgeted Evidence Acquisition  
**Execution contract:** v0.292 (method track extended in v0.33.1, see
`docs/method_gate_v033.md`)  
**Decision date:** 2026-08-15

> **v0.33.1 update:** the statistics policy below changed from "one global
> Benjamini–Hochberg FDR policy" to **purely descriptive** (paired mean diff +
> 95% bootstrap CI + per-seed sign agreement, no p/q-value language) — with
> `n=5` seeds the exact sign-flip floor is `p_min=0.0625 > 0.05`, so no
> comparison family, however small, can ever be BH-significant here; shrinking
> the family does not help (see `docs/handoff_fable_sol_20260815.md` §2.6).
> `results/paired_stats_pack.md` also gained three `*-seqft` rows (carry-over
> c1) — these have the largest, most consistent effects in the whole table
> (e.g. main K=1 `distill-seqft` action-KL: mean +0.2637, 95% CI
> [+0.2386,+0.2872], 5/5 seeds agree), directly supporting C2/C4.

## Decision

The analysis-centric direction remains supported. No new CL loss, EUP, query
conditioning, or completed-grid rerun is justified before the paper freeze.
The first K=1 mechanism probe supports the existing behavior-preservation
analysis, but does not justify a K=2 extension or a new method claim.

## Claim status

| Claim | Status after WP1–WP4 | Paper action |
|---|---|---|
| C1: budgeted evidence acquisition is learnable | Supported by frozen Gate 1 | Keep |
| C2: sequential learning causes acquisition drift in both orders/budgets | Supported by frozen main/reverse tables and K=1 probe | Keep |
| C3: frozen evaluators attribute old-task change to the shared policy | Supported by protocol and audit | Keep |
| C4: replay/distillation reduces acquisition drift and often forgetting | Supported, but method ranking is metric/order dependent | Keep with “substantially reduces drift”; no universal winner |
| C5: behavior fidelity and capability retention are different axes | Provisional, not strengthened by WP3: after fixing an order-pooling bug in `plasticity_report.py`, λ=3 own-time degradation appears in only 2/8 matched task×K cells (main order only) | No stability–plasticity wording; use behavior-fidelity / capability-retention framing only |
| C6: utility weighting is a budget-dependent refinement | Descriptive/provisional: K=2 `ours−ours_uniform` forgetting CI is positive but not consistent across K (sign flips at K=1); buffer composition is shared | Do not claim universal utility benefit |
| C7: task order changes method behavior | Descriptive/provisional across paired main/reverse outputs; no ranking-reversal claim | Say “task-order dependence” only if qualified/descriptive |

## Evidence produced

- `results/paired_stats_pack.md`: 210 paired seed-level bootstrap rows (six
  confirmatory comparison families plus three `*-seqft` exploratory rows added
  in v0.33), both orders where available, purely descriptive (mean diff + 95%
  CI + per-seed sign agreement; see the v0.33.1 update above).
- `docs/audit_20260815.md`: metric directions, update/loss/buffer fairness,
  joint reference semantics, and the reverse `K=1` `distill` std check.
- `results/plasticity_report.md`: own-time `A[t,t]` by method/λ, both orders.
- `results/mech_probe_summary.md`: seed 0, `K=1`, main/reverse,
  `seqft`/`distill`/`ours`; raw JSONL state dumps and stage checkpoints are
  under `runs/v2/mech_probe_v0292_seed0_k1_r2/`; figures are
  `figures/mech_*.png`.

## Required claim downgrades before 8/21

1. Replace “`ours` is best” with “Utility-Weighted Replay Distillation is a
   targeted variant whose benefit depends on budget, order, and metric.”
2. Replace “joint upper bound” with “joint-training reference.”
3. State that `ours` and `ours_uniform` share utility-prioritized buffer
   truncation; the utility ablation isolates distillation-loss weighting.
4. Do not turn the K=1 mechanism into an EUP motivation or add K=2 before a
   separate review.
5. Keep `pilot40` unclaimed until WP5 is run; its absence does not invalidate
   the can_dataset analysis-centric paper, but it limits spatial-generalization
   language.

## WP5 status

`scripts/run_pilot40_minimal.sh` is prepared for RunPod CUDA inside `tmux`,
with a fresh tag, `--device auto`, `--resume`, and checkpoints. It was not
started under the v0.292 stop condition.
