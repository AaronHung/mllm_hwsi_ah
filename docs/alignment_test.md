# E1 — Utility-Capability Alignment Test (pre-registration)

**Status:** committed BEFORE any E1 run, per the same r1/r3 discipline that
governs every gate document in this repo. Part of **v0.35-closeout** (joint
three-way sign-off): after E1 and E2, **all compute for this ICASSP paper
ends permanently, regardless of outcome**. E1 creates no method, reopens no
verdict, and feeds no promotion path.

**Date convention:** document dates are local CST calendar dates; run tags
are UTC timestamps.

---

## 1. Required disclosure (verbatim from the signed-off closeout text)

> "The oracle selects argmax raw counterfactual gain at each step. Gains are
> computed against ground-truth labels through the frozen evaluator, so the
> oracle is LABEL-INFORMED: it is a one-step greedy ceiling diagnostic, not a
> deployable policy, and its result is an upper bound on what one-step
> utility can deliver."

**Phrase-lock note.** `scripts/check_forbidden_phrases.py` reserves the two
maximum-attainable-value metaphors used in the quotation above for statements
about `joint` (`docs/research_contract_v0.292.md` forbidden changes). The
quotation is a **mandated verbatim disclosure about a label-informed oracle**,
not about `joint`, and the closeout authorization requires it word for word —
so it is kept exactly as written and the checker's report on this file is
disclosed rather than worked around. The checker script is not in this
authorization's allowed-to-change list, so no exemption was added to it, and
this document is expected to report exactly one uncompensated hit, on the
quoted line. Every other new closeout document passes cleanly.

## 2. What E1 asks

Every Track-B and Track-C result rests on the same one-step counterfactual
utility signal: the counterfactual-teacher target, `eps_optimal_mass`,
`normalized_regret`, `trajectory_utility`, and `L_eq` are all defined against
it. Across five loss-level interventions and one architecture-level one, that
signal was preserved or improved without capability following. E1 asks the
prior question:

> **Does the one-step counterfactual utility target itself convert into final
> capability?**

If even a policy that maximizes the target perfectly, with label information,
does not beat a plainly-trained navigator's own-time accuracy, then the
non-conversion observed all along is a property of **the target**, not of any
method that tried to preserve it. That is a limitation of the measurement
apparatus this project has been using, and it belongs in the paper as such.

## 3. Oracle definition

The oracle is exactly the existing `nav.engine.teacher_rollout` selection
sequence, evaluated on **test** slides: at each step it computes, for every
un-zoomed candidate, the counterfactual loss reduction under the frozen
per-task evaluator against the slide's ground-truth label, and zooms
`argmax(gain)`. No navigator is trained or loaded. `nav/engine.py` is not
modified — the oracle trajectory is read out of the returned `TeacherStep`
list.

Final balanced accuracy is then computed exactly as `eval_policy_balanced`
does: mean of the evaluator's prediction on `evidence_of(slide, selection)`,
macro-averaged recall over classes.

**Determinism.** The oracle path draws no random numbers (`@torch.no_grad()`,
greedy argmax). E1 nevertheless runs the whole oracle pass **twice** and
asserts bit-identical results; non-determinism is a STOP condition.

## 4. The evaluator, and an honest limitation

E1 needs the per-task frozen evaluator. Evaluators are not checkpointed, so
they are retrained with the same code, data, `epochs=30`, and `seed`. Two
facts about `nav.engine.train_evaluator`:

- its batching/augmentation uses a **local** `torch.Generator` seeded by
  `seed`, so that part reproduces exactly;
- the `Evaluator` module's **weight initialization draws from the global RNG
  stream**, whose state at task `t >= 1` inside `run_sequence` depends on
  everything trained before it.

E1 therefore calls `torch.manual_seed(seed)` immediately before each
`train_evaluator` call. The resulting evaluator is **equivalent in
construction but not bit-identical** to the one a frozen run used. This is
disclosed, not hidden, and two guards are pre-registered against it:

1. **Equivalence check.** E1 measures the random policy on its own evaluator
   using the same convention `scripts/cl_main.py` uses (5 rollouts, seeds
   `seed*100 + r`) and compares it to the frozen `random_ref` column for the
   same `(task, seed, K)`. The per-cell difference is reported. A small
   difference is evidence the evaluators are interchangeable for this
   purpose; a large one is a caveat on the whole comparison.
2. **Random-anchored read.** Alongside the pre-registered rule, E1 reports
   `oracle - random` (both measured on E1's own evaluator, so no
   cross-evaluator term) next to `seqft_own_time - random_ref` (both from the
   frozen CSVs, likewise internally consistent). This contrast is robust to
   evaluator differences and is reported as descriptive supporting evidence.

## 5. Comparators (frozen; never rerun)

| quantity | source |
|---|---|
| `seqft` own-time `A[t,t]` | `results/cl_main_can_main_full_s{0,1,2}.csv`, rows with `stage == task_idx` |
| random reference | the `random_ref` column of the same frozen rows |
| per-task quantum `1/n_test` | the `n_test` column of the same frozen rows |

## 6. Grid

Seeds {0,1,2}; K in {1,2} for the read-out rule; **K=4 additionally reported
as descriptive context only** because it costs minutes — it is explicitly
**not** part of the rule below. Main order, `can_dataset` test split, Mac MPS
(`docs/compute_policy.md`). Forward passes and evaluator training only; **no
navigator is trained anywhere in E1**.

## 7. Deterministic read-out rule (pre-registered tiers, descriptive)

Applied to K in {1,2} only:

- **MISALIGNMENT SUPPORTED** — oracle final balanced accuracy `<=` `seqft`
  own-time mean, averaged across tasks, at **both** K.
- **ALIGNMENT SUPPORTED** — oracle `>=` `seqft` own-time + one per-task
  quantum (`1/n_test`) on the cross-task mean **AND** better on a majority of
  tasks, at **both** K.
- **Otherwise INDETERMINATE** — written up as a limitation. **No further runs
  under any tier.**

Per-task values are reported with their quanta alongside. These tiers are
**descriptive**: no p/q language, consistent with the repo statistical policy.

## 8. Internal-consistency checks (reported, not gating)

- The oracle should sit at or near the top of the `eps_optimal_mass` scale and
  near zero `normalized_regret` by construction — it is the argmax of the very
  quantity those metrics score. Reported as a sanity check that the read-out
  measures what it claims to.
- Oracle determinism across two identical passes (§3).
- Evaluator equivalence against `random_ref` (§4.1).

## 9. Zero-compute correlation half

From existing CSVs only, no runs: for every `L_eq`-bearing config and every
control, tabulate per-seed `(delta eps_optimal_mass, delta AA, delta
Forgetting)` against `ours_uniform` and `distill`, and report sign patterns
and Spearman correlations **descriptively**. If the utility axis moved
consistently while `AA` did not track it, that is the same non-conversion seen
from a different angle, measured across every configuration this project ever
ran.

## 10. Scope limits

E1 is a **diagnostic**. It does not create a method, does not reopen the
v0.33.3, Gate v2, v0.34 or C0 verdicts, and does not feed any promotion path.
Wording locks from the ratified v0.34 verdict apply. **STOP after the report
is written**, for three-way review. After E1 and E2, compute for this paper is
over.

## 11. Changelog

- **E1 pre-registered (2026-08-17, joint three-way closeout sign-off;
  document written by Cursor, no E1 numbers observed):** oracle definition,
  the label-informed disclosure verbatim, the evaluator-reconstruction
  limitation with its two pre-registered guards, the frozen comparator list,
  and the three-tier read-out rule — all fixed before any E1 run.
