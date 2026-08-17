# Method gate v0.34 — pre-registration (conflict-aware equivalence
preservation)

**Status:** committed BEFORE any v0.34 training run, per the same r1/r3
discipline that governs `docs/method_gate_v033.md`. This document opens a
**new, hypothesis-driven method track**. It does **not** reopen, re-score,
or reinterpret anything decided under v0.33: the v0.33.3 gate verdict and
the Gate v2 verdict (`results/method_gate_v0333_verdict.md`,
`results/method_gate_v2_verdict.md`) remain frozen, and `eq_pres` remains
**unpromoted**. v0.34 exists because Gate v2 produced a specific,
mechanistic, fixable diagnosis, not because its answer was unwelcome.

**Date convention (stated once, applies to this whole document):** dates in
this document are **local CST calendar dates**; run tags are **UTC
timestamps**. Where sign-off text written by the red team carries its own
date, that date is quoted verbatim and annotated editorially rather than
rewritten.

**Parent documents:** `docs/method_gate_v033.md` (v0.33.3 gate + Gate v2,
frozen), `docs/research_contract_v0.292.md` (foundation contract,
statistical policy), `docs/protocol.md` (frozen main-experiment
definitions), `docs/compute_policy.md` (backend rules).

---

## 0. Supersession amendment (S0)

Quoted verbatim from the joint Aaron+Sol+Fable sign-off text:

> "v0.33 §10.5 barred further method compute. That clause is superseded
> by PI decision on 2026-08-17: Gate v2 produced a specific, mechanistic,
> fixable diagnosis (EqPres preserves evidence utility 5/5 but its
> preservation gradients conflict with new-task acquisition). v0.34 is a
> new pre-registered protocol, not a rescue: the v2 FAIL verdict is
> unchanged; eq_pres remains unpromoted. v0.34 carries its own one-shot
> config set and hard calendar stops; no v0.35 will be opened for this
> ICASSP submission regardless of outcome."

*[date as written in the sign-off text; committed 2026-08-16 CST]*

**What §10.5 actually says** (quoted so the supersession is checkable, not
paraphrased): "Seeds 3 and 4 are fixed now, run together in one command, no
seed-by-seed peeking or early stopping. **This is the last main-order
method-rescue compute for this paper cycle**, regardless of outcome: no
lambda tuning, no epsilon change, no fail-cell-only reruns, no new
modules/losses." §10.6's FAIL branch closes with "No further method-rescue
compute."

**Scope of the supersession.** The superseded reading is the one that treats
§10.5/§10.6 as a ban on *any* subsequent method work. What remains fully in
force, and is re-asserted here:

1. The Gate v2 verdict is **FAIL** and is never reinterpreted.
2. `eq_pres` is **not promoted** and is not a v0.34 candidate. It appears in
   v0.34 only as the **diagnostic precursor**, using frozen data.
3. There is **no rescue of `eq_pres`**: no lambda tuning, no epsilon change,
   no fail-cell-only reruns, no re-scoring of any v0.33 cell.
4. v0.34 is one-shot: **three fixed configs, no fourth**, hard calendar
   stops, and **no v0.35 for this ICASSP submission regardless of
   outcome**.

## 1. Sign-off chain and precedence

Four texts govern this protocol. Later entries win where they conflict.

| # | Text | Role |
|---|---|---|
| 1 | Fable's v0.34 prompt (S0-S4) | base protocol |
| 2 | Sol focused red-team amendment (items 1-6) | clarifications before launch |
| 3 | **Fable FINAL RIDER (r1-r4)** | **resolves conflicts; wins where texts differ** |
| 4 | PI (Aaron) rulings, 2026-08-16 | mechanism measurement, launch, date convention |

Explicitly superseded by the rider:

- **S2 criterion B** -> replaced by Sol item 1 (targeted repair **plus**
  no-new-failure over all `task x K` cells, plus the K=2/4 forgetting
  no-regression check). See §3.2.
- **S2 "MECHANISM LINK ... if ~0, ABORT early"** -> replaced by Sol item 2:
  instrumentation is mandatory, the conflict link is a **pre-registered
  hypothesis reported post-hoc**, and there is **no numeric early-abort**.
  See §3.3.
- **S3 "5 fresh seeds"** -> replaced by Sol item 5: reverse order, **exact
  seeds {0,1,2,3,4}**, K={1,2,4}, described as "previously unseen v0.34
  reverse-order confirmation runs". See §4.

## 2. Method

Implemented in `nav/cl.py` behind opt-in kwargs whose defaults reproduce the
existing behavior exactly, per the convention already stated at
`nav/cl.py`'s `train_navigator_cl` docstring. **No frozen code path is
restructured**; see §2.6 for the bit-exactness proof required before launch.

### 2.1 Memory objective

```
L_mem = L_replay + lambda * E_s[ sg[v(s)] * L_eq(s) ]

v(s)     = 1 - sum_{a in A_eps(s)} pi_theta(a|s)        (stop-gradient)
L_eq(s)  = -log( clamp( sum_{a in A_eps(s)} pi_theta(a|s), 1e-8, 1 ) )
A_eps(s) = { a : normalized_gain(a) >= 1 - eps },  eps = 0.05
```

- `eps = 0.05` is **unchanged** from v0.33 §3, and `lambda` stays at its
  existing value (`--lam`, default 1.0). **Neither is tuned in v0.34.**
- `v(s)` is the equivalence **violation**: how much probability mass the
  navigator has already moved outside the task-useful evidence set. It is
  applied with stop-gradient so it acts as a per-state constant weight and
  cannot introduce second-order gradient terms.
- `A_eps(s)`, `normalized_gain`, and the `clamp` guard reuse the existing
  `nav.cl.epsilon_optimal_mask` / `nav.cl.normalized_gain` implementations
  verbatim — including the canonical convention that a flat-gain state makes
  every candidate epsilon-equivalent (all-ones, never all-zeros).
- The counterfactual replay-target term (`use_replay`) is unchanged and
  keeps its existing weight of 1.0.

### 2.2 Gradient arbiter

Per update, with `L_new` the current-task term and `L_mem` the replay-side
term of §2.1:

```
g_n = grad(L_new),  g_m = grad(L_mem)          (flattened over all params)

if g_n . g_m < 0:
    g_m <- g_m - (g_m . g_n) / (|g_n|^2 + 1e-12) * g_n

final grad = g_n + lambda_arb * g_m,           lambda_arb = 1
```

- The projection applies to the **whole** `g_m` (replay + preservation term
  combined), so the arbiter is **identical for all projected configs** and
  the only isolated variable across §2.3 is the memory objective itself.
- **Claim precision (r4 / Sol item 3).** What this guarantees is, in **raw
  gradient space**, removal of the component of the memory gradient that is
  first-order anti-aligned with the current-task gradient: if
  `g_n . g_m < 0` then after projection `g_n . g_m_proj = 0`. Because the
  optimizer is Adam (`nav/cl.py`, `torch.optim.Adam`), this does **not**
  imply that the realized parameter update cannot worsen the current-task
  objective. New-task learning capacity remains an **empirical gate**, never
  a theoretical claim.
- The training loop is single-state (batch size 1); "current-task batch" and
  "replay batch" in the sign-off text refer to the current-task state and
  the sampled replay state of each update, which is the existing loop
  structure, not a new batching scheme.

### 2.3 Configs (fixed, one-shot — no fourth config)

All three build on the `ours_uniform` base (`use_replay=True,
use_distill=True, utility_weight=False`), exactly as the v0.33 gate configs
do, so that the comparison isolates the memory objective and the arbiter and
never buffer membership or the utility-weight ablation axis.

| config | `L_mem` | arbiter | violation weight | role |
|---|---|---|---|---|
| `proj_distill` | `L_replay` + old-policy KL | ON | no | generic control |
| `proj_eq_pres` | `L_replay` + `L_eq` | ON | no | proposed |
| `conflict_eq_pres` | `L_replay` + `sg[v(s)]`·`L_eq` | ON | yes | proposed (full) |
| `eq_pres` | `L_replay` + `L_eq` | OFF | no | **frozen precursor, not rerun** |

**Naming caveat, stated so nobody misreads the table:** `proj_distill`
carries the name used in the sign-off text, but its `L_mem` is
`L_replay + old-policy KL`, i.e. the **`ours_uniform`** composition plus the
arbiter — **not** the frozen `distill` method, which has no replay term.
The frozen `distill` remains a comparator only.

The three questions the configs answer, in the order they isolate:

- `proj_distill` vs `ours_uniform` — does the arbiter alone do the work?
- `proj_eq_pres` vs `proj_distill` — does the equivalence objective beat
  generic old-policy distillation once both are arbitrated?
- `conflict_eq_pres` vs `proj_eq_pres` — does violation-aware selective
  preservation add anything on top?

`eq_pres` rows come from the frozen Gate-v2 / v0.33.3 data as the diagnostic
precursor. **No rerun as a candidate, no re-scoring.**

### 2.4 Required instrumentation

Logged **every update** for every arbitrated config, and for the §2.5
replication batch:

- `cos(g_n, g_m)`
- conflict flag (`g_n . g_m < 0`)
- `|projection| / |g_m|`
- per-stage conflict fraction (derived)

Raw per-update records are written to
`runs/v2/<tag>/arbiter/<method>_seed<S>_K<k>.jsonl` (excluded from version
control by the existing `runs/v2/**/*.jsonl` rule, same convention as the
mechanism-probe raw dumps); the committed artifact is the per-stage summary
`runs/v2/<tag>/arbiter/arbiter_summary_<method>_seed<S>_K<k>.json`.

### 2.5 Instrumentation-only replication of frozen `eq_pres`

**PI decision, 2026-08-16 (Aaron), pre-registered here before any run.**
Rider r2 requires the dev verdict to carry a conflict-fraction table for the
diagnosed target cells (`brca` K=2, `lung` K=4). Those cells are `eq_pres`'s
failures, but `eq_pres` is not rerun as a candidate, and the arbitrated
configs' pre-projection statistics are only an approximation of `eq_pres`'s
own conflict rate (the trajectories diverge after the first projection).
This batch closes that gap under four binding conditions:

1. **Label and standing.** The batch is an "instrumentation-only
   replication of frozen `eq_pres`". It runs under the separate method key
   `eq_pres_diag` so that no metric row from it can be pooled with, or
   substituted for, any `eq_pres` row. **No metric row from this batch is
   admitted anywhere, and it cannot alter any verdict** — its only output is
   the conflict diagnostic.
2. **Implementation.** The diagnostic `g_n` / `g_m` are obtained with
   `torch.autograd.grad(..., retain_graph=True)` on the **same forward
   graph** already built by the frozen code path: no extra forward passes,
   no RNG consumption, no optimizer-state mutation. The update path stays
   `opt.zero_grad(); loss.backward(); opt.step()`, byte-identical to frozen
   `eq_pres`.
3. **Validity condition (pre-declared).** The resulting CSV must be
   **bit-identical** to the frozen `eq_pres` rows, verified in the same
   spirit as the `util_regen` checksum convention (§5.2 / §10.2 of
   `docs/method_gate_v033.md`). **Any mismatch discards the ENTIRE
   instrumentation batch**, and the verdict falls back to reporting the
   arbitrated configs' pre-projection statistics with an explicit limitation
   note (trajectories diverge after the first projection; approximation, not
   counterfactual measurement).
4. **Scheduling.** Queued **after** the 27-unit development gate in the same
   runner. The development gate owns the critical path.

### 2.6 Bit-exactness precondition (before the gate launches)

Because the arbiter changes how gradients are assembled, the frozen loss
accumulation order in `train_navigator_cl` is left untouched and the new
behavior is added behind flags. This is **verified, not asserted**: before
the development gate starts, `eq_pres` seed 0 K=1 and `ours_uniform` seed 0
K=1 are rerun with the new code into a scratch tag and must reproduce the
frozen rows exactly (these two cover both memory-objective branches,
`L_eq` and old-policy KL). Any mismatch blocks the launch. Result recorded
in `results/method_gate_v034_bitexact_regression.md`.

## 3. Development gate

**Environment labelling.** The development gate runs on the **main order**
(`esca -> lung -> rcc -> brca`), which is **already unblinded**. It is
therefore explicitly a **development environment**, not a confirmation.
Nothing observed here can be reported as confirmatory evidence; that is what
§4 is for.

### 3.1 Runs

Mac MPS (per `docs/compute_policy.md`), `tmux` + `caffeinate`, atomic
resume, fresh tag, checkpoints ON.

| # | units | what | role tag |
|---|---|---|---|
| 1 | 27 | `proj_distill` / `proj_eq_pres` / `conflict_eq_pres` x seeds {0,1,2} x K∈{1,2,4}, main order | `dev_candidate` |
| 2 | 9 | `eq_pres_diag` x seeds {0,1,2} x K∈{1,2,4}, main order (§2.5) | `instrumentation_only_frozen_replication` |

Total **36 units**, all Mac MPS, in that order. Runner:
`scripts/run_gate_v034.py` (atomic resume keyed on
`seed=..|method=..|K=..|role=..`, same pattern as `scripts/run_gate_v2.py`).

**Instrumentation self-check (code check, not a scientific rule).** After
the first 2 units complete, the runner verifies that the conflict
diagnostics were actually written and are non-degenerate (not all-NaN, not
all exactly zero, update count > 0) and aborts if they were not. This
guards against a silently broken logger. It is **not** a numeric threshold
on the conflict statistic and must never be read as one — the numeric
early-abort rule was removed by Sol item 2 / rider r2.

### 3.2 Criteria (3-seed paired means vs frozen comparators)

Seeds {0,1,2}. All criteria are **mean-only** (paired seed-level mean
differences), matching the sign-off text's "3-seed paired means". Per-seed
signs and 95% bootstrap CIs are reported alongside every number as
**descriptive disclosure only** — never as a pass/fail test, per the
statistical policy in `docs/research_contract_v0.292.md` (no p/q language).

**A — Utility retained.**
`delta eps_optimal_mass > 0` vs `distill` **AND** vs `ours_uniform`, at
K ∈ {1,2}. Aggregation is the frozen §5.1 convention: final stage, old tasks
only, mean within task then macro-average across old tasks.
`normalized_regret` is reported alongside as secondary descriptive evidence
but is **not** a pass condition (criterion A names `eps_optimal_mass`).

**B — Targeted repair + no new failure** (per rider r1 = Sol item 1).

- **B-a (no new failure, plasticity).** For **every** `(task, K)` cell —
  4 tasks x 3 budgets = 12 cells — the 3-seed paired mean
  `delta A[t,t]` vs `ours_uniform` must be `>= -0.01`.
- **B-b (no new failure, forgetting).** At K=2 and K=4,
  `delta Forgetting <= +0.005` vs `ours_uniform` **AND** `distill`.
- **B-c (diagnosed target repairs, explicitly required).** `brca` K=2 and
  `lung` K=4 must satisfy `delta A[t,t] >= -0.01` vs `ours_uniform`; and at
  K=1, `delta Forgetting <= 0` vs `ours_uniform` **AND** `distill`. These
  cells are named separately in the report even though B-a already covers
  the plasticity half — the point of the amendment is that fixing only the
  already-known cells is not enough, not that the known cells stop
  mattering.

**C — No behavior-fidelity requirement; replay safety unchanged.**
Selection Jaccard carries **no requirement** in v0.34 and is reported for
description only. This is by design: the method deliberately does not
require the navigator to re-walk its exact historical route, so a lower
Jaccard than `distill` is an expected consequence, not a defect. Replay
safety is unchanged from v0.33: `delta Forgetting` vs `replay` `<= +0.005`
at all K.

**A config PASSES the development gate iff A and B (a, b, c) and C all
hold.**

### 3.3 Mechanism link (pre-registered hypothesis, reported post-hoc)

The hypothesis under test is: *the plasticity cost observed for `eq_pres` in
the Gate-v2 failure cells arises because preservation gradients conflict
with new-task acquisition gradients.*

- Instrumentation (§2.4) is **mandatory**.
- The dev verdict **must** include a conflict-fraction table for the
  diagnosed target cells (`brca` K=2, `lung` K=4), plus the full
  stage x K table, for three-way review.
- There is **no numeric early-abort** and **no threshold is chosen after
  observing the values**. If conflict turns out to be empirically near
  absent in those cells, the verdict states plainly that the
  gradient-conflict explanation **lacks support** — that statement is the
  falsification outcome, not a re-tuned criterion.

### 3.4 Winner selection (pre-registered before any training)

The sign-off text specifies a "winner config" for §4 but does not define the
tie-break. The rule below is authored by the coding agent (Cursor) and
committed **before** training so it cannot be chosen after seeing results.
It is flagged as such in the verdict, and the three-way review may object to
it **before** the numbers are read.

1. `proj_distill` is a **control, not a promotion candidate**. The winner is
   drawn only from `{proj_eq_pres, conflict_eq_pres}`.
2. If both pass: the winner is the one with the larger criterion-A effect
   (mean over K ∈ {1,2} of `delta eps_optimal_mass` against both
   comparators).
3. If still tied: the one whose worst B-cell margin is larger.
4. If still tied: `proj_eq_pres` (the simpler mechanism).
5. **Disclosure rule.** If `proj_distill` also satisfies criterion A, that
   must be stated explicitly and prominently in the verdict: it would mean
   the utility advantage is not attributable to the equivalence objective,
   which is direct evidence against the novelty positioning of §6. The
   three-way review, not the coding agent, decides what follows.

### 3.5 Hard stops

- Development verdict by **EOD 2026-08-19** or automatic abandonment.
- **No config additions.** No `lambda` tuning. No `eps` change.
- No re-scoring, re-running, or reinterpretation of any v0.33 cell.
- **No v0.35** for this ICASSP submission, whatever the outcome.

## 4. Confirmation branch (only after three-way review)

**IF the development gate PASSES** and the three-way review approves:
confirmation = winner config, **reverse order**
(`brca -> rcc -> lung -> esca`), **exact seeds {0,1,2,3,4}**, K ∈ {1,2,4},
Mac MPS. These are described as **"previously unseen v0.34 reverse-order
confirmation runs"** — never as "new" or "fresh" seed IDs, because the seed
IDs themselves are reused deliberately to preserve pairing with the existing
frozen reverse-order comparators. Reported in the old/new/pooled style with
per-seed signs. Then one winner row on `pilot40` (RunPod CUDA, matched to
the existing pilot40 table). Deadline 2026-08-21.

**IF FAIL or abort:** archive honestly. The paper ships framework-first with
`eq_pres` as the diagnostic precursor and the v0.34 configs reported as
tested-negative with full per-seed tables, exactly as `ia_samp` / `ia_ep`
are. Science freeze 2026-08-22 (absolute 2026-08-23).

**Compute policy is unchanged** (Sol item 6): development and confirmation
on MPS; the pilot40 winner row on RunPod CUDA matched to the existing
pilot40 table; **no formal CPU training**.

## 5. Reporting and STOP

- Every v0.34 report opens with the two-line Track A / Track B status
  header.
- The development verdict is written to
  `results/method_gate_v034_dev_verdict.md`, computed by
  `scripts/method_gate_v034_verdict.py`, which imports the existing metric
  implementations (`scripts/aggregate_results.py`,
  `scripts/method_gate_v2_verdict.py`) rather than reimplementing them, so
  the definitions cannot silently drift.
- **STOP after the development verdict is written**, for three-way
  (Aaron + Sol + Fable) review. **No confirmation run, no promotion, no
  paper-claim change before that review**, regardless of which way the
  numbers point.

## 6. Paper-claim locks (rider r4)

- **Never** state that new-task learning "cannot be blocked" under Adam, or
  any equivalent. The only claim licensed by §2.2 is removal of the
  first-order anti-aligned component in raw gradient space; new-task
  learning capacity stays an empirical gate.
- Novelty is stated as **equivalence-aware continual evidence acquisition +
  violation-aware preservation + conflict-aware arbitration of preservation
  updates**. Gradient projection is the **mechanism**, not the claim.
- **Do not** describe the contribution as "reverse A-GEM" or as inventing
  gradient projection. Prior work (GEM/A-GEM, PCGrad, GPM/SGP/TRGP and
  successors) is acknowledged in related work as the established technique
  this method applies.
- The v0.33 archival statement is unchanged: `eq_pres` improves evidence
  utility but incurs costs on old-task retention and new-task acquisition;
  `ia_samp` and `ia_ep` are RETIRED / tested-negative.

## 7. Comparators and data provenance (all same-backend MPS, none rerun)

| comparator | source |
|---|---|
| `ours_uniform`, K ∈ {1,2} | `results/cl_main_can_main_ablA1_s{0,1,2}.csv` (frozen Protocol-v1) |
| `ours_uniform`, K = 4 | `runs/v2/gate_v2_20260816T073200Z/` rows with `gate_v2_role=comparator_completion_k4` |
| `distill`, `replay`, K ∈ {1,2,4} | `results/cl_main_can_main_full_s{0,1,2}.csv` (frozen Protocol-v1) |
| utility columns for `ours_uniform`/`distill`, seeds 0-2, K ∈ {1,2} | `runs/v2/util_regen_20260815_mps/` (already checksum-passed, `results/util_regen_checksum.md`) |
| `eq_pres` diagnostic precursor | `runs/v2/method_gate_v0333_run1/` (frozen, seeds 0-2) |

No comparator is retrained. Every arm of every v0.34 comparison is Mac MPS,
per `docs/compute_policy.md` rule 4.

## 8. Changelog

- **v0.34 opened (2026-08-16, joint Aaron+Sol+Fable sign-off; document
  written by Cursor, criteria transcribed from the sign-off texts, no
  numbers observed):** supersession amendment for `docs/method_gate_v033.md`
  §10.5/§10.6 (§0 above), three fixed configs (§2.3), development gate
  criteria as amended by Sol item 1 and rider r1/r2 (§3.2, §3.3),
  confirmation seeds fixed to {0,1,2,3,4} per rider r3 (§4), paper-claim
  locks per rider r4 (§6). PI additions of 2026-08-16: the
  instrumentation-only replication of frozen `eq_pres` with its four binding
  conditions (§2.5), the runner's instrumentation self-check (§3.1), and the
  date convention stated in the header. Cursor-authored operational rule
  requiring three-way sign-off or objection before results are read: the
  winner-selection tie-break (§3.4). Nothing in `docs/method_gate_v033.md`
  is edited by this document except one dated provenance footnote at the end
  of its §10 and one changelog entry pointing here.

- **Development-gate verdict computed (2026-08-17, by Cursor — analysis only,
  NOT a red-team-reviewed decision, NOT a protocol amendment):** the
  pre-registered 36-unit grid (`runs/v2/gate_v034_dev_20260816T154548Z/`,
  5h03m wall-clock, Mac MPS) finished cleanly — exit 0, 36/36 units, 10 rows
  each, no missing cells. The pre-launch bit-exactness regression passed
  20/20 rows before launch (`results/method_gate_v034_bitexact_regression.md`)
  and the §2.5 instrumentation batch was **ADMITTED** at 90/90 bit-identical
  (`results/eq_pres_diag_bitexact_check.md`), so the conflict statistics are a
  direct measurement of the config that failed Gate v2 rather than a proxy.
  `scripts/method_gate_v034_verdict.py` applied the §3.2 criteria exactly as
  frozen, with zero threshold or formula choices made after seeing results.
  Full table: `results/method_gate_v034_dev_verdict.md`.
  - **DEV GATE FAIL for all three configs.** `proj_distill` fails A/B-a/B-b/
    B-c; `proj_eq_pres` passes A and B-c but fails B-a, B-b and C;
    `conflict_eq_pres` passes A, B-b and C but fails B-a and B-c. No winner,
    no confirmation run, no promotion.
  - **Criterion A replicates the one Track-B effect that holds.**
    `proj_eq_pres` improves `eps_optimal_mass` on 3/3 seeds at both K
    (+0.0411/+0.0509 at K=1 vs `ours_uniform`/`distill`), `conflict_eq_pres`
    likewise, and the generic control `proj_distill` — arbiter but no `L_eq` —
    does **not** (K=2 vs `ours_uniform`, −0.0028). The §3.4 item-5 disclosure
    is therefore **not triggered**: the effect tracks `L_eq`, not the arbiter.
  - **Cell-level plasticity is not resolvable at this sample size.** With 3
    seeds and per-task `n = 76-95`, the cell means are comparable to the
    per-item quantum `1/n_test` and far smaller than the per-seed spread, so
    no cell-level movement is reported as a repair or as a newly introduced
    failure. The pre-registered statement is simply that **no configuration
    met the targets**. This under-power is a disclosed limitation of the gate
    itself, not a property of any config.
  - **Mechanism: measured NULL.** Conflict fraction is 0.458-0.504 across
    every config, stage and budget, with mean cosine about 0; the diagnosed
    target cells are indistinguishable from healthy ones (`rcc` K=1, which
    never had a problem, is the highest at 0.5039); and projection does not
    change the conflict rate (`brca` K=2: `eq_pres_diag` 0.4913 vs
    `proj_eq_pres` 0.4876). This is what unrelated high-dimensional gradients
    look like. Per rider r2 no threshold was invented after the fact.
  - **This is an analysis artifact awaiting joint Aaron+Sol+Fable review** —
    Cursor started no confirmation run, altered no threshold, added no config,
    and re-scored no v0.33 cell.

- **v0.34 verdict RATIFIED (2026-08-17, Fable, external red-team, not
  Cursor; PI Aaron concurring):** the DEV FAIL verdict and the bit-exactness
  checks were re-verified cell by cell. **Method compute is formally closed
  per §0/§4; no v0.35 for this ICASSP submission.** Three rulings modify how
  the result is *described*, not what it is:
  1. **Wording lock.** Cursor's first report narrated the diagnosed-cell
     movements as repairs, and the `brca` K=4 movement as a newly opened
     failure. Both framings are **withdrawn**: at 3 seeds the
     means are smaller than or comparable to the per-item quantum and to seed
     spread. Mandated phrasing: *cell-level plasticity differences fall within
     seed variability; no configuration met the pre-registered targets*, with
     the gate's under-power stated explicitly. Corrected in
     `docs/research_timeline_20260817.md` under a dated erratum, and made
     machine-checked in `scripts/check_forbidden_phrases.py`.
  2. **Mechanism is a NULL RESULT, not an unconfirmed hypothesis.** Fable
     added the internal-consistency argument Cursor had missed: projection
     does not change the conflict rate, so the conflict is a structural
     random phenomenon rather than a removable lesion. Gradient conflict is
     the field's default explanation for this kind of interference; a
     pre-registered instrumented measurement showing it does not hold for
     continual evidence acquisition is a publishable negative result.
  3. **The no-new-failure clause is the most valuable process fact of the
     round.** Under the original S2-B (repair the two diagnosed cells plus
     K=1 forgetting), `proj_eq_pres` would have PASSED and gone straight to a
     confirmation run. Sol's amendment item 1 prevented that. This belongs in
     the paper's protocol section.

  Follow-up assigned to Cursor and carried out under this entry: the two
  changelog entries above, the wording corrections, and three **zero-compute**
  descriptive analyses derived from existing CSVs (utility-axis replication
  across every `L_eq`-bearing config, a resolution/power note, and the
  conflict-null table). None of them reopens any gate. **Track B is closed;
  all remaining Track-B work is writing.**
