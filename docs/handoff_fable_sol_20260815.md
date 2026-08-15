# Handoff to Fable (Claude.ai) & Sol (ChatGPT) — 2026-08-15

**Purpose:** status report + list of problems found/fixed during v0.292 execution, for
independent evaluation and next-step instructions before the 8/21 science freeze.
Read this alongside `docs/research_contract_v0.292.md` (the execution contract) and the
[v0.32 HackMD direction freeze](https://hackmd.io/@aaronh/rkqRxshLMl) (parent decision).

**Repo:** `AaronHung/mllm_hwsi_ah`, branch `main`, HEAD = `3080763` (pushed and in sync
with `origin/main` as of this writing).

## 1. What was executed since the v0.292 freeze

All five work packages from `docs/research_contract_v0.292.md` were run at "zero/limited
compute" as specified:

- **WP1** `scripts/paired_stats_pack.py` → `results/paired_stats_pack.md` /
  `.csv`: paired seed-level bootstrap CIs for 6 comparison families × both orders
  (where data exists) × K ∈ {1,2,4} × 5 metrics (AA, Forgetting, BWT, Jaccard,
  action-KL). One global Benjamini–Hochberg FDR policy at q=0.05, 120 rows total.
- **WP2** `docs/audit_20260815.md`: implementation/fairness audit — metric formulas,
  update counts, loss scales, buffer composition, `joint` schedule, and the
  reverse `K=1` `distill` std=0.004 sanity check.
- **WP3** `scripts/plasticity_report.py` → `results/plasticity_report.md`: own-time
  new-task accuracy `A[t,t]` per method/λ, both orders, all K.
- **WP4** `scripts/mechanism_probe.py` → `results/mech_probe_summary.md` +
  `figures/mech_*.png`: seed-0, K=1, main+reverse, `seqft`/`distill`/`ours`,
  stage-boundary checkpoints, per-state JSONL dumps under `runs/v2/mech_probe_v0292_seed0_k1_r2/`.
- **WP5** `scripts/run_pilot40_minimal.sh` is written and gated (not run), per the
  contract's stop condition.

Two commits landed this content: `10b23e8` (WP1–WP3 + relabeling) and `a9992b0`
(WP4 + `docs/science_freeze_memo_20260815.md`). A third commit `3080763`
regenerates `figures/fig1_architecture.png` (see §2.2).

## 2. Problems found during execution (fix these before trusting any diagram/formula)

These are the substantive issues an evaluator needs to know about — several would make
an architecture diagram or a written formula **wrong relative to the actual code**, not
just cosmetically stale.

### 2.1 `action-KL` direction was written backwards in `docs/protocol.md`

The implementation (`nav.cl.kl_drift`, called from `scripts/cl_main.py` as
`kl_drift(pi_at_own[j], policy_on_probes(...))`) computes
`KL(π_{t=j} ‖ π_final)` — reference is the policy **at the task's own stage**, current
is the **final** policy. `docs/protocol.md` had this reversed
(`KL(π_final ‖ π_{t=j})`) before this pass. **Fixed** in `docs/protocol.md` and confirmed
consistent with `mechanism_probe.py`'s `kl_ref_now`. Any paper text or slide that copied
the old direction needs re-checking — I found and fixed the copies in
`docs/protocol.md`, but I did not exhaustively grep every historical slide/doc for the
old formula string; a second pass by whichever of you can grep is worth doing before the
figure/equation goes into `paper/main.tex`.

### 2.2 Fig. 1 architecture diagram was stale relative to the terminology fix

`scripts/make_fig1.py`'s CL-mechanism box literally said "Utility-Weighted Replay
Distillation" with `w(s) ∝ u(s)` as if that were *the* mechanism, but the
v0.292 contract says `ours` is a *variant* (uniform-weight `ours_uniform` and
utility-weight `ours` share the same distillation term, differing only in `w(s)`).
The rendered PNG was also generated **before** the Aug-15 label edits, so image and
contract text had already diverged. **Fixed**: box now reads "Replay + policy-fidelity
distillation (family)" with `w(s)=1` (uniform) vs. `w(s) ∝ u(s)` (utility-weighted variant)
shown side by side, and the PNG was regenerated (commit `3080763`). If either of you
render your own version of Fig. 1 for the draft, use this framing, not the old one.

### 2.3 `joint` is not compute-matched — must not appear as an upper bound

Confirmed in `docs/audit_20260815.md`: at stage `t`, `joint` trains on the concatenation
of all tasks seen so far for the same 10 epochs, so its optimizer-update count **grows**
with `t` while every sequential method's update count is fixed. This was already a
stated risk in earlier docs, but the audit is the first place it's checked against the
actual training loop line-by-line. Treat any `joint`-vs-`X` gap as "reference gap", never
as "headroom" or "ceiling" in the paper.

### 2.4 `ours` and `ours_uniform` share utility-prioritized buffer truncation

Both variants use the same buffer composition rule (2 states/slide floor, then
globally-highest-utility fill to capacity). The utility ablation therefore isolates
**only the loss-weighting term**, not "utility-aware memory" as a whole. This constrains
what WP1's `ours-ours_uniform` row can claim — see §3 below, it is not significant
anyway.

### 2.5 `mechanism_probe.py`: flat-teacher-gain states were silently zeroing the
ε-equivalent set

Original `normalized_gain` returned `np.zeros_like(raw)` when `hi - lo <= 1e-12` (all
candidates tied). That made a genuinely ε-equivalent state (all candidates equally
useful) look like an empty ε-optimal set, which would bias ε-set-size and
ε-optimal-mass downward exactly where the teacher signal is flattest. **Fixed** to
return `np.ones_like(raw)` (all candidates are within ε of each other when the gain
spread is ~0) before the final WP4 run that produced `results/mech_probe_summary.md`.
If either of you re-derive these formulas from the raw JSONL dumps, use the corrected
convention.

### 2.6 Statistical power problem in WP1 — needs a decision, not just a note

With `n=5` seeds, the exact paired sign-flip test's smallest achievable two-sided
p-value is `2/2^5 = 0.0625`. Over the 120-hypothesis global BH-FDR family, **zero rows
survive q ≤ 0.05** (`results/paired_stats_pack.md` — grep confirms no
`FDR-significant` tag anywhere in the file). This is expected given the sample size and
family size, and `docs/science_freeze_memo_20260815.md` already treats every
comparison-based claim (C4, C6, C7) as "descriptive/provisional" rather than
"significant" for this reason — but it means **the paper cannot make any FDR-controlled
significance claim from the existing 5-seed grids as currently pooled**. Options, for
your evaluation:
  (a) keep the paper purely descriptive (CI direction + magnitude, no p/q-value
      language at all — this is what the current memo effectively already does);
  (b) shrink the pre-registered hypothesis family (e.g. report only the `ours` vs.
      `distill`/`replay` primary comparisons, not all 6 families × both orders) so BH-FDR
      has a chance to detect something real;
  (c) accept that 5 seeds is underpowered and flag it as a stated limitation rather than
      running more seeds this late in the timeline.
  My default until told otherwise: (a), because it is already what the memo does and
  needs no new compute.

### 2.7 `plasticity_report.py` silently pooled main+reverse order in the λ comparison (found and fixed while writing this handoff)

`lambda_conclusion()`'s pivot grouped by `["K", "task", "setting_key"]`, **omitting
`order`**. The λ ablations (`ablA3a`/`ablA3b`) only exist for `order=main`, but the
`"ours_lambda1"` key is produced by the base full grid for **both** `order=main` and
`order=reverse`. So the λ=1.0 baseline used in the headline conclusion was actually the
mean of main-order and reverse-order own-time accuracy pooled together, silently
violating the file's own stated rule ("Main and reverse rows are never pooled") and
comparing it against a main-only λ=3.0 ablation. The original (wrong) run reported
**"degradation in 4/8 matched task×K cells"**, which is what fed
`docs/audit_20260815.md`'s naming section and `docs/science_freeze_memo_20260815.md`'s
C5 row before this fix.

**Fixed**: `lambda_conclusion()` now restricts to `order == "main"` before pivoting.
Re-running `scripts/plasticity_report.py` gives a materially different, weaker result:
**only 2/8 matched cells degrade** (`main K=1`: brca, esca degrade; `main K=2`: lung, rcc
degrade — note this is a different pair of tasks than the original wrong run reported,
because the pooled numbers were simply incorrect, not just off by a rounding margin).
`2/8 < majority`, so the report's own decision rule now selects the **"no consistent
degradation"** branch. I have already propagated this correction to
`docs/audit_20260815.md` and `docs/science_freeze_memo_20260815.md`'s C5 row (both
now say: use only "behavior-fidelity / capability-retention trade-off", the qualified
stability–plasticity phrase is not licensed at all). **This is the single most
consequential fix in this pass** — please double-check my re-derivation
(`results/plasticity_lambda_differences.csv`) independently if either of you can, since
it flips a claim that was going into the paper.

## 3. Key numeric results (for cross-checking, not just the summary prose)

### WP1 highlights (`results/paired_stats_pack.md`, full 120 rows there)

- No comparison reaches FDR-significance anywhere (see §2.6).
- Consistently signed, largest-magnitude effects (still only descriptive):
  `distill`/`ours`/`replay` all show **positive** Jaccard and action-KL gaps vs. `seqft`-adjacent
  baselines are not directly tested here (WP1 only compares the preservation family
  against each other), but within the family: `ours-distill` and `ours-replay` action-KL
  differences are small and sign-flip depending on order/K; `distill-replay` action-KL is
  consistently **positive** (`distill` shows more drift than `replay`) across every
  order/K row available.
- `ours-ours_uniform` AA gap at `main K=1` is `-0.0158` (CI [-0.0203,-0.0114], not
  FDR-sig) — i.e. in this one cell, uniform weighting has slightly *higher* AA than
  utility weighting; at `K=2` the sign flips (`+0.0099`, wide CI). No consistent
  direction — supports "budget-dependent, not universal" framing already in the memo.

### WP2 verdict (`docs/audit_20260815.md`)

No fairness bug found in update counts / loss scale / buffer composition among the
sequential methods. `joint`'s growing update count and the `ours`/`ours_uniform` shared
buffer truncation are the two documented interpretation limits (§2.3, §2.4). Reverse
`K=1` `distill` Forgetting std = 0.004021 is a normal 5-seed sample std, not degenerate
(other metrics on the same rows have std 0.010–0.030).

### WP3 (`results/plasticity_report.md`, corrected — see §2.7)

Own-time accuracy `A[t,t]` degradation from λ=1.0→3.0, **main order only** (the λ
ablation does not exist for reverse order, and must not be pooled with it): degradation
in only 2 of 8 matched task×K cells (`results/plasticity_lambda_differences.csv`:
`K=1` brca −0.0119, esca −0.0250 degrade; `K=1` lung +0.0132, rcc +0.0025 improve;
`K=2` brca +0.0244, esca +0.0134 improve; `K=2` lung −0.0048, rcc −0.0004 degrade). This
is a minority of cells, so the report's own majority rule now selects **"no consistent
degradation"** — the qualified stability–plasticity phrasing is not licensed anywhere;
use "behavior-fidelity / capability-retention trade-off" as the only framing.

### WP4 (`results/mech_probe_summary.md`)

Seed 0, K=1, main+reverse: `seqft` shows visibly larger final-stage action drift on
earlier tasks (e.g. reverse `brca`: drift 0.380 vs. `distill` 0.074 vs. `ours` 0.089) and
lower ε-optimal probability mass than `distill`/`ours`. ε-equivalent sets are broad
(13–22 candidates near-optimal out of the region set), and drift–regret correlation `r`
is weak/inconsistent across tasks (ranges roughly -0.09 to 0.63). This supports the
qualitative behavior-preservation story but explicitly does **not** justify extending to
K=2 or motivating a new EUP-style loss (contract forbids both).

## 4. Claim status (from `docs/science_freeze_memo_20260815.md`)

| Claim | Status | Paper action |
|---|---|---|
| C1 budgeted evidence acquisition is learnable | Supported (Gate 1) | Keep |
| C2 sequential learning causes acquisition drift, both orders/budgets | Supported | Keep |
| C3 frozen evaluators attribute old-task change to the shared policy | Supported | Keep |
| C4 replay/distillation reduces drift and often forgetting | Supported, ranking is metric/order-dependent | Keep, no universal winner |
| C5 behavior fidelity vs. capability retention are different axes | Provisional, **not** strengthened by WP3 after the order-pooling fix (2/8 cells, main order only, §2.7) | No stability–plasticity wording anywhere; behavior-fidelity / capability-retention only |
| C6 utility weighting is a budget-dependent refinement | Descriptive/provisional, no FDR significance | No universal-benefit claim |
| C7 task order changes method behavior | Descriptive/provisional | "Task-order dependence", qualified only |

## 5. Open questions for your evaluation

1. **Statistics policy (§2.6)** — confirm option (a)/(b)/(c), or propose another. This
   determines whether `paired_stats_pack.md`'s p/q columns appear in the paper at all.
2. **Is WP5 (pilot40) required before 8/21**, or does the paper ship as a `can_dataset`-only
   analysis paper with pilot40 explicitly out of scope (current default per the memo)?
3. **Fig. 1 framing (§2.2)** — is "family with two weight choices" the right way to draw
   the CL mechanism, or should the figure drop the formula entirely and just name the
   three preservation methods (`replay`, `distill`, `ours`) as boxes?
4. Any remaining place in `paper/main.tex` that still implies `ours` wins universally or
   `joint` is an upper bound — please flag by line number if either of you spot one; I
   did a manual pass but have not run an automated string check against the forbidden
   phrase list in `docs/research_contract_v0.292.md`.
5. Sign-off on `docs/audit_20260815.md` §"Reverse K=1 distill standard deviation" — the
   contract explicitly asked this to be verified; I consider it closed, but a second
   read is welcome given it's a "is this a bug or just noise" judgment call.

## 6. Files to read, in order

1. `docs/research_contract_v0.292.md` — the contract itself.
2. This file.
3. `docs/audit_20260815.md` (WP2).
4. `results/paired_stats_pack.md` (WP1, full 120 rows).
5. `results/plasticity_report.md` (WP3).
6. `results/mech_probe_summary.md` + `figures/mech_*.png` (WP4).
7. `docs/science_freeze_memo_20260815.md` (claim table, same as §4 here).
8. `figures/fig1_architecture.png` (regenerated).

## 7. Infra note (not science, but relevant to reproducing any of the above)

Dual-backend infra (`nav.resolve_device`, `scripts/smoke_test.sh`,
`scripts/runpod_bootstrap.sh`, `docs/RUNPOD_SOP.md` v2) landed in commit `116b0ad` and
was rehearsed end-to-end on a live RunPod pod (bootstrap + data-integrity check against
`SHA256SUMS.txt` passed). All WP1–WP4 numbers above were produced on Mac
(CPU for WP1–WP3 pandas jobs, MPS for the WP4 mechanism probe); WP5 is the only package
that requires RunPod CUDA per the backend-consistency rule (5 seeds must share one
backend).
