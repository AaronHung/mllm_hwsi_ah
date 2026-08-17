# Track C0 — Minimal Stable-Plastic Navigator (pre-registration)

**Status:** committed BEFORE any C0 training run, per the same r1/r3
discipline that governs `docs/method_gate_v033.md` and
`docs/method_gate_v034.md`. C0 is a **minimal screening experiment, not a
promotion gate**. Track A is frozen and continues in parallel (writing);
Track B is closed and archived (`docs/method_gate_v034.md` §0/§4 — no v0.35).

**Date convention:** document dates are local CST calendar dates; run tags
are UTC timestamps.

**Sign-off chain:** Sol architecture proposal -> PI (Aaron) approval,
2026-08-17 -> Fable C0 specification and sign-off, 2026-08-17. This document
transcribes the signed-off `TRACK C0` prompt (S0-S5) and records, separately
and explicitly, every implementation decision the prompt left open (§6).

---

## 0. Framing and hypothesis

Track B established two things across five pre-registered interventions
(utility weighting, interference sampling, gradient arbitration, violation
weighting, and their combinations):

1. A loss-level objective **can** stably preserve evidence-access utility.
   `L_eq` improved `eps_optimal_mass` / `normalized_regret` consistently
   across four configs, two gates, and eleven seeds, and the effect tracks
   `L_eq` rather than the arbiter (`proj_distill`, which has the arbiter but
   no `L_eq`, does not show it).
2. That preservation **does not convert** into capability or new-task
   acquisition gains, and the default field explanation for the
   interference — gradient conflict — was **measured and not supported**
   (conflict fraction 0.458-0.504 with mean cosine about 0 across every
   config, stage and budget; target cells indistinguishable from healthy
   ones; projection does not change the conflict rate).

**C0 hypothesis (architectural, not loss-level):** a *single shared*
navigator forces stability and plasticity into the same parameters, and that
sharing — not the choice of preservation objective — is what caps new-task
acquisition. Giving each task a small private modulation of a slowly-updated
shared core should let new-task acquisition proceed at unconstrained-baseline
speed without paying for it in retention.

**What C0 is not:** it is not a promotion gate, not a claim of a new method,
and not evidence about a router. It screens one architectural hypothesis on
the smallest configuration that can answer it.

## 1. ORACLE TASK IDENTITY — required disclosure

**C0 assumes oracle task identity at BOTH train and test time.** The task
index selects which adapter is active, during training and during every
evaluation, including the evaluation of old tasks at later stages.

This is a **best-case assumption**. A task-incremental setting with known
task identity is strictly easier than the class-incremental or task-agnostic
settings, and a reviewer is entitled to say so. C0 makes no claim that
survives without it. (The signed-off prompt words this as an "upper-bound
assumption"; the repo's machine-checked phrase lock reserves that phrase for
statements about `joint`, so C0 says "best-case" instead. Same meaning, no
weakening — `scripts/check_forbidden_phrases.py` flagged the collision before
this document was committed.) Inferring task identity (a router) is **future work and
is explicitly not part of any C0 claim**. Every C0 report must carry this
disclosure; the verdict document repeats it verbatim.

## 2. Architecture (S1)

Implemented in `nav/models.py` and `nav/cl.py` behind opt-in flags whose
defaults reproduce the pre-C0 behavior exactly. **Frozen paths must be
byte-identical when the flags are off**, verified by the bit-exactness
regression (§5) before any C0 unit runs.

### 2.1 Core

Unchanged from Protocol-v1:

```
Linear(low_dim + high_dim + 1 -> 256) -> ReLU
  -> Linear(256 -> 64) -> ReLU
  -> Linear(64 -> 1)
```

For `can_dataset`, `low_dim = 64` and `high_dim = 512`, so the core holds
`577*256 + 256 + 256*64 + 64 + 64 + 1 = 164,481` parameters.

### 2.2 Per-task FiLM adapters

For each task `t`, a per-layer diagonal scale and shift applied to the two
hidden activations (after their ReLUs):

```
h1 = ReLU(W1 x + b1);   h1 <- gamma_t^(1) * h1 + beta_t^(1)     (256-dim)
h2 = ReLU(W2 h1 + b2);  h2 <- gamma_t^(2) * h2 + beta_t^(2)     (64-dim)
score = W3 h2 + b3
```

`gamma` is initialized to 1 and `beta` to 0, so **the adapter is exactly the
identity at task onset** and the first update of a new task is undisturbed.
Adapter size is `2*(256 + 64) = 640` parameters per task, i.e. **0.39% of
the shared core per task** and **1.56% for all four tasks** — well inside the
5% disclosure bound of c4.

FiLM rather than LoRA is a deliberate choice at this width: for 256/64 hidden
layers a low-rank factorization saves little, while FiLM is smaller, is the
identity at initialization, and is a dozen lines of code.

### 2.3 Slow shared core

The core is **slowly updated, not frozen**: `lr_core = 0.1 * lr_adapter`.
Freezing the core outright would leave it permanently in the state it reached
after task 1 (`esca`), which is indefensible under review; slow update is the
literal reading of "stable-plastic".

At stage `t`: adapter `t` trains at the full learning rate, the core trains at
one tenth of it, and **adapters of earlier tasks are not updated** once their
stage has ended (their parameters have `requires_grad=False`, so no gradient
is computed for them at all). Adapters of future tasks are untouched and
remain at identity until their own stage.

### 2.4 Inference

Evaluation of task `j` at any stage uses **adapter `j`** (oracle task ID,
§1). This applies to every measurement: own-time accuracy, old-task
accuracy, selection Jaccard, action-KL, trajectory utility, and the
utility-axis probe metrics.

## 3. Configs (S2)

| config | composition | what it isolates |
|---|---|---|
| `sp_nav` | core + per-task adapters, shared imitation objective only (no replay, no distillation, no `L_eq`) | the architecture alone |
| `sp_nav_eq` | `sp_nav` + `L_eq` consolidation on replayed states (`eps = 0.05`, same lambda slot as `distill`) | architecture + the one objective Track B showed to work |

Comparators are the **frozen** rows for `seqft`, `distill` and `eq_pres`.
**None of them is rerun.**

## 4. Runs (S3)

Seeds {0,1,2} x K in {1,2}, main order, Mac MPS (per
`docs/compute_policy.md`), `tmux` + `caffeinate`, atomic resume, fresh tag,
checkpoints ON. **12 units**, estimated 2-3h. Per-seed values and 95%
bootstrap CIs are reported as **descriptive disclosure**, never as a
pass/fail test (repo statistical policy: no p/q language).

## 5. Bit-exactness precondition

Before any C0 unit runs, `eq_pres` and `ours_uniform` seed 0 K=1 are replayed
through the C0 code into a scratch tag and must reproduce the frozen rows
exactly. This is the same check v0.34 §2.6 used and for the same reason: the
adapters change how the forward pass is assembled, and "the frozen path is
untouched" is a claim that must be verified rather than asserted. Recorded in
`results/track_c0_bitexact_regression.md`. **A mismatch blocks the launch.**

## 6. Implementation decisions the prompt left open

Recorded here, before training, so none of them can be chosen after seeing a
number. Each is open to red-team objection **before** the verdict is read.

1. **`sp_nav_eq` does not include the replay imitation term.** The prompt
   defines it as "`sp_nav` + `L_eq` consolidation on replayed states", and
   `sp_nav` is "no CL loss beyond the shared imitation objective". The
   literal composition is therefore `imitation + L_eq`, isolating exactly one
   addition. The alternative reading — matching `eq_pres`'s composition,
   `imitation + L_replay + L_eq` — was not chosen because it would change two
   things at once relative to `sp_nav`. Note that c3 compares against
   `distill`, not against `eq_pres`, so nothing in the criteria requires
   composition-matching with `eq_pres`.
2. **Replayed states use their source task's adapter.** A replayed state from
   task `j` is scored with adapter `j` (frozen at that point) while gradients
   flow into the shared core. Using the current task's adapter would evaluate
   old evidence through a modulation that task never had, which contradicts
   the oracle-task-identity design.
3. **No weight decay on FiLM parameters.** The core keeps its existing
   `weight_decay=1e-4`; the adapter parameter group uses `weight_decay=0`.
   Applying L2 to a scale parameter initialized at 1 pulls it toward 0, i.e.
   away from the identity the design depends on. Excluding scale/shift
   parameters from weight decay is standard practice.
4. **c1 is judged per task, pooled over K; c2 per K, pooled over tasks.**
   Criterion c1 says "for every task" and c2 says "at every K", and the
   preamble forbids per-cell vetoes. So c1 is four checks (one per task, the
   3-seed paired mean over K in {1,2}) and c2 is two checks (one per K). The
   per-`(task, K)` breakdown is reported alongside as **descriptive
   disclosure only** and has no veto power.
5. **Adapter learning rate equals the pre-C0 navigator learning rate**
   (`lr = 1e-3`), so `lr_core = 1e-4`. No learning-rate tuning is performed
   in C0.

## 7. C0 criteria (S4)

3-seed paired means, **aggregate-level, no per-cell veto** — three seeds
cannot resolve cell-level plasticity differences, which is the direct lesson
of the v0.34 result (cell means smaller than the per-item quantum
`1/n_test`, with per-seed spread of several quanta).

- **c1 PLASTICITY.** Own-time `A[t,t]` `>= seqft - 0.01` for **every task**
  (3-seed paired mean, pooled over K in {1,2}).
- **c2 STABILITY.** `Forgetting <= distill` at **every K** (3-seed paired
  mean).
- **c3 UTILITY** (`sp_nav_eq` only). `delta eps_optimal_mass > 0` vs
  `distill`, aggregated by the frozen §5.1 convention (final stage, old tasks
  only, mean within task then macro-average across old tasks).
- **c4 DISCLOSURE.** Report parameter counts and update counts; adapter
  parameters must be **< 5% of the shared core**.

A config passes C0 iff its applicable criteria all hold. **Passing C0 is not
a promotion**: it licenses proposing C1 to the three-way review, nothing more.

## 8. Reporting and STOP

The verdict is written to `results/track_c0_verdict.md` by
`scripts/track_c0_verdict.py`, opens with the two-line Track A / Track B/C
status header, repeats the §1 oracle-task-identity disclosure verbatim, and
applies the Track-B wording locks (below). **STOP after the verdict** for
three-way review. **No C1, no router, no confirmation runs.**

**Wording locks inherited from the ratified v0.34 verdict** (Fable,
2026-08-17), which apply to this document and every C0 report:

- Do not describe a cell-level plasticity difference at 3 seeds as a repair
  or a newly introduced failure. At per-task `n = 76-95` these means are
  comparable to the per-item quantum and to seed spread. Correct phrasing:
  *cell-level plasticity differences fall within seed variability*.
- State the under-power limitation explicitly wherever cell-level plasticity
  is discussed.
- Report the v0.34 gradient-conflict measurement as a **null result**, not as
  "unconfirmed".
- Claim precision inherited from rider r4 remains in force.

## 9. Hard stop (S5)

**C0 verdict by EOD 2026-08-19**, or automatic abort back to Track A. No
C0.1. Track A writing proceeds in parallel and is **not blocked by C0 in any
way**.

## 10. Changelog

- **C0 verdict computed (2026-08-17, by Cursor — analysis only, NOT a
  red-team-reviewed decision, NOT a protocol amendment):** the pre-registered
  12-unit grid (`runs/v2/track_c0_20260817T014543Z/`, 12/12 units, Mac MPS)
  finished cleanly, after the §5 bit-exactness regression passed 20/20 rows
  (`results/track_c0_bitexact_regression.md`).
  `scripts/track_c0_verdict.py` applied the §7 criteria exactly as frozen.
  Full table: `results/track_c0_verdict.md`.
  - **C0 FAIL for both configs.** `sp_nav` fails c1 and c2; `sp_nav_eq` fails
    c1 while passing c2, c3 and c4. Branch per §9: return to Track A. No
    C0.1, no router, no confirmation runs.
  - **c1 (the C0 hypothesis) is not met.** Parameter isolation did not
    deliver unconstrained-baseline plasticity; `brca`, the last and hardest
    task, is negative for both configs.
  - **c2/c3 reproduce the Track-B shape.** `sp_nav_eq` is stability-safe
    (`-0.0106`/`-0.0053` vs `distill`) and shows the largest utility effect
    measured in this project (`+0.1429`/`+0.1807` vs `distill`, 3/3 seeds),
    while `sp_nav` — architecture with no consolidation term — fails
    stability and is negative on utility. Architecture alone buys neither.
  - Three structural caveats are recorded separately in
    `results/track_c0_observations.md`, written after the verdict and
    changing nothing in it: `esca` is stage 1 and therefore cannot test the
    hypothesis at all (both configs are the same algorithm there, verified);
    c1's `-0.01` threshold is finer than one test item on every task, so the
    criterion is under-powered in both directions; and part of c3's margin is
    plausibly attributable to the §6.1 composition decision rather than to
    the architecture.
  - **This is an analysis artifact awaiting joint Aaron+Sol+Fable review.**

- **C0 opened (2026-08-17, Sol proposal + PI approval + Fable specification
  and sign-off; document written by Cursor, criteria transcribed from the
  signed-off prompt, no C0 numbers observed):** pre-registration of the
  minimal stable-plastic navigator screening experiment. Track B remains
  closed; nothing in `docs/method_gate_v033.md` or
  `docs/method_gate_v034.md` is reopened, re-scored, or reinterpreted by this
  document. Implementation decisions the signed-off prompt left open are
  listed in §6 and are open to objection before the verdict is read.
