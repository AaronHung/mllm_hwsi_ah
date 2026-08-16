# Method gate v0.33.3 — pre-registration

**Status:** §0-§8 below (the v0.33.3 gate) was committed BEFORE any gate
training run, per r1/r3, and is immutable — its result has been read: **all
three main configs GATE FAIL** (`results/method_gate_v0333_verdict.md`,
2026-08-16, Cursor analysis). That verdict **stands and is never
reinterpreted.** Per joint Sol+Fable unblinding review (2026-08-16/17):
`ia_samp` (M1) and `ia_ep` (M3) are **RETIRED** (no further compute); `eq_pres`
(M2) got its one pre-registered **Gate v2 extension** (§10 below). **Gate v2
result: GATE FAIL** (`results/method_gate_v2_verdict.md`, 2026-08-16, Cursor
analysis) — g2 (utility) PASSES cleanly at 5/5 seeds (stronger than the
original 3/3), g3 (replay safety) PASSES, but g1 (capability) FAILS at K=1
and g4 (plasticity) newly FAILS at `lung` K=4 and `brca` K=2, in both cases
because the 2 new seeds **reinforce** rather than wash out the concern (not a
seed-3/4-only reversal — see changelog for detail). Per §10.5, this was the
last main-order method-rescue compute regardless of outcome: `eq_pres` is
retired from promotion and will be reported as an intervention/mechanism
result. Track A status: 8/21 freeze conditions all met 2026-08-16 (pilot40
R6 closed). Pre-registration discipline unchanged: each gate version is
immutable once its result is read (r3); amendments live in new, dated,
clearly-versioned sections/changelog entries, never silent edits to a frozen
section.

**v0.33.3 in one line:** a backend-reproduction audit (§5.2) found the
method-gate's originally planned "CUDA new methods vs. MPS frozen
baselines" comparison would confound method effect with backend effect;
§5.4/§5.5 below redesign the gate as **MPS-first, same-backend
throughout** — no comparator retraining needed. No formula, threshold,
seed, K, or order changed from v0.33.2.

**Parent direction:** v0.32 — Continual Budgeted Evidence Acquisition
(analysis-centric, frozen).
**Foundation contract:** v0.292 (`docs/research_contract_v0.292.md`, frozen
evidence, paper-facing names, statistical policy — updated in v0.33.1, see
`docs/handoff_fable_sol_20260815.md` §2.6).
**This document:** Track B of v0.33 two-track execution — a bounded, gated
method-exploration track on top of the frozen v0.292 foundation. It does
**not** reopen the v0.32 direction, add EUP, add query conditioning, or
introduce any loss/method beyond M1/M2/M3 below.

## 0. Motivation (for framing only, not a gate criterion)

WP4's mechanism probe (`results/mech_probe_summary.md`) shows three
partially-decoupled layers: action drift, per-state utility regret, and
task-level capability forgetting can all move independently (e.g. `seqft`
on main `rcc`: drift 0.336 but utility regret 0.063, lower than `distill`'s
0.119, while `seqft`'s capability forgetting is the worst of the three).
M1/M2 are motivated by this: **sample what is being forgotten and matters,
and constrain the policy to keep probability mass on near-optimal evidence
rather than to reproduce its past distribution.** If no config passes the
gate, the paper stays analysis-centric and this attempt is reported honestly
in an appendix — that outcome is a valid, pre-accepted result of this
protocol, not a failure that requires a rerun.

## 1. Buffer membership is unchanged

M1 and M2 never change which states are in the replay buffer. Buffer
construction (`nav.cl.build_buffer`: 2 states/slide floor, then
within-chunk-highest-utility fill to capacity) is byte-identical to every
frozen Protocol-v1 method. This is intentional: it prevents any gate config
from being confounded with the buffer-truncation ablation audited in
`docs/audit_20260815.md` §"Naming and scope corrections" / §2.4 of the
handoff.

**Buffer-cap semantics (v0.33.2 doc-fix item 7):** `build_buffer(..., cap=
buffer_cap)` is called once **per source task**, on that task's own steps
only, right after the task finishes (`scripts/cl_main.py`'s `run_sequence`:
`buffer_by_task.append(build_buffer(steps, ..., cap=buffer_cap))`). So
`buffer_cap=512` truncates **within each task's own chunk**, keeping that
chunk's highest-utility states up to 512 — it is **not** a single global
512-state memory budget. The training-time `buffer` is the **concatenation**
of all completed tasks' capped chunks
(`buffer = [st for bb in buffer_by_task for st in bb]`), so with e.g. 3 old
tasks the replay pool can hold up to `3 * 512` states. This does not affect
gate fairness (every arm, including all frozen baselines, uses the identical
per-chunk-cap-then-concatenate code path) but must be described correctly in
the paper and in `docs/audit_20260815.md` (both corrected in this revision —
see Changelog). The existing 128/512/2048 memory ablation is therefore a
**per-task-cap** ablation, not a total-continual-memory-budget ablation.

## 2. M1 — interference-aware replay sampling (v0.33.1, corrected)

**Buffer sampling probability** for a candidate replay state `s` in the
current buffer:

```
p(s) ∝ u_state(s)^alpha · d_t(s)^beta,   alpha = beta = 1
p(s) >= 0.1 / |buffer|                    (floor, then renormalized)
```

### 2.1 SUPERSEDED definition (v0.33, deleted before any gate run — r1)

> `u_bar(s)` = per-state min-max-normalized utility of the state's top-gain
> action.

**Why this was deleted:** within-state min-max normalization makes the
top-gain value identically `1` for every non-flat state, and flat states are
also all-ones by the epsilon-equivalence convention (§2.5 of the handoff).
So `u_bar(s) ≡ 1` for the entire buffer, and `p(s) ∝ d_t(s)` collapses to
pure drift-only sampling — "important × being forgotten" degenerates to just
"being forgotten." This was caught by Sol's red-team review before any gate
training started (see `Sol-v0.33-...` HackMD note) and independently
confirmed by Fable.

### 2.2 SUPERSEDED definition (v0.33.1, deleted before any gate run — Sol's
formulation-risk finding)

> `u_state(s) = percentile_rank( max_a u_s(a) )` computed **across the whole
> buffer** (all source tasks pooled together), using the raw counterfactual
> gain.

**Why this was deleted:** each source task's counterfactual gain comes from
that task's own independently-trained frozen evaluator, and the tasks are
not homogeneous (ESCA/LUNG/BRCA are 2-class, RCC is 3-class), so evaluator
calibration, base cross-entropy, and class count all differ across tasks. A
percentile rank computed over the pooled, cross-task buffer does not correct
for this: if one task's raw-gain distribution is systematically higher than
another's, its states get systematically higher `u_state` regardless of
whether they are actually more important *within their own task*. That would
let evaluator-scale differences masquerade as state importance — a
formulation risk caught by Sol's second-round focused red-team (joint
Sol+Fable review, `docs/handoff...` HackMD thread) before any gate training
started, independently of the v0.33 -> v0.33.1 fix in §2.1 above.

### 2.3 Current definition (v0.33.2)

```
u_state(s) = percentile_rank( max_a u_s(a) )  computed WITHIN s's own
             source-task buffer chunk (buffer_by_task[i], BEFORE
             flattening), using the RAW (un-normalized) counterfactual
             gain — i.e. TeacherStep.utility, which is already
             max_a gain(a) clamped at 0.
```

- Ranks are computed **per source-task chunk** in `buffer_by_task`, then
  concatenated in the exact same order used to flatten `buffer` for training
  (`chunkwise_percentile_rank`) — this guarantees index alignment with
  `eps_masks` (built from the same flattened `buffer` inside
  `train_navigator_cl`) without any re-sorting step, and without any new
  hyperparameter.
- Cross-**task** replay pressure is now carried entirely by `d_t(s)` (§2.4):
  whichever old task has drifted more gets more total sampling mass; within
  each task, `u_state` still ranks that task's own states by their own
  relative importance. M1's contribution is intentionally scoped to
  **state-level** prioritization within a task, not task-level
  prioritization across tasks.
- Cross-state importance (`u_state`) must **never** use within-state
  min-max-normalized values — that normalization is correct and unchanged
  for a different question (§3, M2's `A_eps`, and all mechanism metrics).
  `u_state` answers "which state matters more than other states in the same
  task"; the within-state normalization in M2 answers "which actions are
  near-optimal inside this one state." Conflating the two is exactly the
  v0.33 bug (§2.1).
- Implementation: `nav.cl.chunkwise_percentile_rank` (per-chunk
  `nav.cl.percentile_rank`, i.e. `scipy.stats.rankdata`, `method="average"`,
  divided by chunk size, then `np.concatenate`), passed into
  `nav.cl.ImportanceReplaySampler` as `u_state=...` by
  `scripts/cl_main.py`'s `run_sequence`.

### 2.4 `d_t(s)` — drift factor

```
d_t(s) = KL( pi_boundary(s) || pi_theta(s) )
```

over the state's own candidate set, where:

- **i1. `pi_boundary(s)`** is the per-state **source-task-boundary policy
  snapshot** — the navigator's policy at the end of the stage in which that
  state's task was learned — stored with the state at the moment it is
  admitted into the buffer (`nav.cl.attach_boundary_policy`, called right
  after `build_buffer`, using the navigator as it stood at that stage's end,
  before it becomes `old_nav` for the next stage). It is never
  re-snapshotted.
- Refreshed (recomputed against the *current* `pi_theta`) every **N=50**
  optimizer updates via a no-grad forward pass over the whole buffer
  (`ImportanceReplaySampler.refresh`, called every `drift_refresh_every`
  update via `maybe_refresh`).

### 2.5 Numerical/attribution details

- Both `u_state` and `d_t` are clipped to `>= 1e-8` before exponentiation so
  `alpha=0` or `beta=0` (mini-arms, §5) yields a clean constant factor
  (`x**0 == 1`) rather than a `0**0` edge case.
- **i3. Diagnostics logged at every refresh:** sampling-distribution entropy
  (`-sum(p log p)`), the Pearson correlation between `u_state` and `d_t`
  across the buffer (redundancy check for whether the two factors carry the
  same information), and — **added in v0.33.2, item 5** — total sampling
  probability mass grouped by source task (`{task_name: p[task==name].sum()}`
  via `nav.cl.flatten_task_labels`), a cheap way to see whether a stage's
  replay is pathologically concentrated on one old task. Written to
  `runs/v2/<tag>/diag_<method>_seed<S>_K<K>.json` by `scripts/cl_main.py`.

## 3. M2 — ε-equivalence preservation loss

On replayed states only, **replace** the old-policy KL term (the `distill`
slot) with:

```
L_eq(s) = -log( clamp( sum_{a in A_eps(s)} pi_theta(a|s), 1e-8, 1 ) )     (i2)

A_eps(s) = { a : normalized_gain(a) >= 1 - eps },   eps = 0.05
```

- `normalized_gain` is the **within-state** min-max normalization
  (`nav.cl.normalized_gain`) — identical convention to
  `scripts/mechanism_probe.py`: a flat-gain state makes every candidate
  epsilon-equivalent (all-ones), never all-zeros.
- The counterfactual replay-target term (`use_replay`) is unchanged.
- Same `lambda * w(s)` slot as `distill` (i.e. the utility-weight ablation
  axis, `utility_weight=False` for every gate config — see §1).
- Implementation: `nav.cl.epsilon_optimal_mask`, applied inside
  `train_navigator_cl`'s `use_eq_pres` branch.

## 4. M3 = M1 + M2

Both changes applied together: importance sampling for *which* buffer state
is drawn, ε-equivalence loss for *what* is optimized on it.

## 5. Configs (`nav.cl.METHOD_GATE_KWARGS`)

All build on the `ours_uniform` base (`use_replay=True, use_distill=True,
utility_weight=False`) so the gate isolates exactly M1 and/or M2.

| config | M1 (sampling) | M2 (loss) | seeds | K | order |
|---|---|---|---|---|---|
| `ia_samp` | importance (α=β=1) | KL (unchanged) | {0,1,2} | {1,2,4} | main |
| `eq_pres` | uniform (unchanged) | L_eq | {0,1,2} | {1,2,4} | main |
| `ia_ep` (M3) | importance (α=β=1) | L_eq | {0,1,2} | {1,2,4} | main |
| `samp_util_only` (mini-arm) | importance, β=0 (util-only) | KL (unchanged) | {0,1,2} | {1} | main |
| `samp_drift_only` (mini-arm) | importance, α=0 (drift-only) | KL (unchanged) | {0,1,2} | {1} | main |

Baselines are the **frozen Protocol-v1 rows** for `ours_uniform` and
`distill` (`results/cl_main_can_main_full_s*.csv` /
`results/cl_main_can_main_ablA1_s*.csv`) and `replay` — never rerun; new
runs are paired seed-by-seed against these frozen CSVs. **Under the v0.33.3
MPS-first backend design (§5.4), these frozen rows plus the already-passing
`util_regen` MPS backfill (§5.2) ARE the same-backend comparators — no
comparator retraining, CUDA or otherwise, is needed for the gate verdict.**

Launch command (Mac MPS, inside `tmux`, `caffeinate`, fresh run tag,
checkpoints ON, per v0.33.3 §5.4/§5.6; writes real per-stage navigator
weights under `runs/v2/<tag>/ckpt/`, §5.3):

```bash
tmux new -s gate
caffeinate -i python scripts/run_method_gate.py --device mps \
  --tag method_gate_v0333_<run_tag> --resume \
  2>&1 | tee logs/method_gate_v0333_<run_tag>.log
# Ctrl-b d to detach; tmux attach -t gate to reattach.
```

`scripts/run_method_gate.py` (v0.33.3 item D) pins the exact grid above —
27 main-config units (`ia_samp`/`eq_pres`/`ia_ep` x seeds x `K∈{1,2,4}`) plus
6 mini-arm units (`samp_util_only`/`samp_drift_only` x seeds, `K=1` only) —
into one resumable invocation, instead of the two hand-split `cl_main.py`
calls used in earlier drafts of this document.

### 5.1 Utility-axis metrics (v0.33.2 items 2–3)

`scripts/cl_main.py`'s `run_sequence` now writes two extra columns to every
row of the results CSV, computed inline (reusing the `policy_on_probes`
forward pass already done for `action_kl`, zero extra training cost):

```
eps_optimal_mass(s)   = sum_{a in A_eps(s)} pi_theta(a|s)     (higher better)
normalized_regret(s)  = 1 - sum_a pi_theta(a|s) * normalized_gain_s(a)  (lower better)
```

using the same `A_eps`/`normalized_gain` (eps=0.05) as M2 (§3). Each row's
value is the mean over that task's probe states, at that stage — identical
aggregation granularity to the existing `jaccard`/`action_kl` columns.

**Pre-registered gate-verdict aggregation** (fixes Sol's rider — do not
choose an aggregation after seeing results): for the g2 utility axis, use
only the **final stage**, **old tasks only** (i.e. exclude the task that was
just learned in that final stage — it is not "preserved old evidence" yet).
Mean over probe states within each old task (already done at the row level),
then macro-average across old tasks (unweighted mean of the per-task row
values). `eps_optimal_mass` is higher-better; `normalized_regret` is
lower-better.

These two columns exist for **every** method run through `run_sequence`
(gate configs and comparators alike) — nothing method-specific about the
column, only about which runs are re-executed to backfill it (§5.2).

### 5.2 Comparator utility metrics: probe-only regeneration (v0.33.2 item 3)

The frozen Protocol-v1 rows for `ours_uniform` and `distill` predate
`eps_optimal_mass`/`normalized_regret` and cannot be retrofitted (the frozen
CSVs are immutable). To give the gate's g2 utility axis a comparator to beat,
regenerate — **do not re-declare as new baseline performance numbers** —
`ours_uniform` and `distill` under a separate tag:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/cl_main.py --dataset can --order main \
  --methods ours_uniform distill --seeds 0 1 2 --budgets 1 2 \
  --device mps --tag util_regen_<run_tag> --resume \
  2>&1 | tee logs/util_regen_<run_tag>.log
```

- **Backend correction (found during implementation, not in the original
  patch text):** the joint red-team's patch said "ON MAC CPU (same backend as
  the frozen rows)". That assumption was checked against
  `logs/cl_main_s*.log` / `logs/abl_ablA1_s*.log` (the actual launch logs for
  the frozen `can` grid, main+reverse+all ablations) and found to be
  **incorrect** — every one of those logs prints `device = mps`. The frozen
  Protocol-v1 rows were produced on Mac **MPS**, not CPU. A first regen
  attempt on CPU failed the checksum below on **all 12/12** (method, seed, K)
  rows with deltas 2-150x the tolerance (e.g. `distill` seed1 K1 jaccard
  Δ=0.148); rerunning the identical command on MPS instead reproduces the
  frozen rows within tolerance (see the checksum report). This is a same-
  backend-rerun precondition failure, not evidence that the simulation is
  generally non-deterministic: three independent CPU-only reruns of the same
  (method, seed, K) matched each other exactly, so CPU vs. MPS is a real
  floating-point-cascade effect (this project's argmax-driven sequential
  rollout is sensitive to tiny per-step numeric differences), not a bug.
- **Acceptance checksum (pre-declared, checked before any utility metric from
  this rerun is used):** per seed, per `K∈{1,2}`, `|Δ| <= 0.001` on `bal_acc`
  (AA), the derived Forgetting metric, `jaccard`, and `action_kl` versus the
  matching frozen Protocol-v1 row (`results/cl_main_can_main_full_s*.csv` for
  `distill`; `results/cl_main_can_main_ablA1_s*.csv` for `ours_uniform` — it
  was only ever run as the utility-ablation comparator, never as part of the
  7-method `full` grid). Only rows that pass the checksum have their
  `eps_optimal_mass`/`normalized_regret` admitted as g2 comparator values.
- **Frozen tables are never replaced** — `runs/v2/util_regen_<run_tag>/` is a
  separate artifact, used only to extract the two new columns.
- **Superseded by v0.33.3 (see §5.4/§5.5 below):** the sentence that used to
  stand here described a "gate-arm utility on CUDA vs. comparator utility on
  MPS" cross-backend pairing. That plan is retracted. It is exactly the
  Δ=0.148 CPU-vs-MPS discrepancy discovered while producing *this section's
  own* MPS regen (immediately above) that triggered the redesign: if CPU vs.
  MPS diverges by 150x tolerance, an untested MPS-vs-CUDA gap cannot be
  assumed negligible either. The gate now runs entirely on MPS — see §5.4 —
  so both gate-arm and comparator utility values are same-backend (MPS) by
  construction; the CUDA/MPS pairing convention below no longer applies to
  this section (it may still describe how `pilot40`'s independent table
  relates to `can`'s, which is a different, already-legitimate case of
  "different dataset, different self-contained table").

### 5.3 Real model checkpoints (v0.33.2 item 4)

`checkpoints.json` remains `--resume` completion bookkeeping only. In
addition, for every gate-config run (`method in
nav.cl.METHOD_GATE_KWARGS`), `run_sequence` now calls `torch.save` on the
navigator's `state_dict()` at every stage boundary (the last stage's save is
by definition the final navigator), under
`runs/v2/<tag>/ckpt/<method>_seed<S>_K<K>_stage<T>.pt`. The navigator is
<1M parameters, so this is negligible storage/compute cost; it exists so
that whichever config passes the gate can be probed with the same mechanism
analysis already used in WP4 (`scripts/mechanism_probe.py`) without needing
to retrain.

### 5.4 Backend design (v0.33.3 item B — supersedes the CUDA-matched plan)

Track-B gate runs **entirely on Mac MPS**, not RunPod CUDA. Under
same-backend matching, the comparators are:

- the **frozen Protocol-v1 MPS rows** (capability/behavior axis — AA,
  Forgetting, Jaccard, action-KL — at all `K∈{1,2,4}`), and
- the **bit-exact MPS `util_regen`** backfill from §5.2 (utility axis —
  `eps_optimal_mass`/`normalized_regret` — at `K∈{1,2}`, which is exactly
  what g2 requires: "at `K=1` or `K=2`").

**No comparator retraining of any kind is required** — this is strictly
cheaper and cleaner than the CUDA-matched plan that was floated between the
first Cursor report and this revision (which would have required rerunning
`ours_uniform`/`distill`/`replay` on CUDA as well: 60 total runs vs. this
design's 33).

Gate compute is **only the 33 new runs**: `ia_samp`/`eq_pres`/`ia_ep` x
seeds `{0,1,2}` x `K∈{1,2,4}` (27) + the two mini-arms x seeds `{0,1,2}` x
`K=1` (6), main order, inline utility metrics (§5.1), stage-boundary
`torch.save` (§5.3), fresh `runs/v2/<tag>/` tag. Run via
`scripts/run_method_gate.py` (§5, item D).

**WP5 `pilot40` runs on RunPod CUDA in parallel, unaffected by this
redesign.** It is a self-contained table (its own dataset, never previously
run on any backend, so there is no frozen-row comparator to backend-match
against) — the compute-policy rule that applies is "one table, one
backend," which a from-scratch CUDA `pilot40` table satisfies on its own.
Note the backend in `pilot40`'s table header.

### 5.5 Timing decision rule (v0.33.3 item C — executed, recorded)

Pre-registered rule: run one representative unit (`eq_pres`, seed 0, `K=2`)
on MPS with `caffeinate`; `T_total = 33 * t_unit * 1.15`. If `T_total <=
48h` active compute, proceed all-MPS immediately, no further approval
needed; if `> 48h`, stop and fall back to the CUDA-matched plan (rerun
`ours_uniform`/`distill`/`replay` as CUDA comparators after all).

**Result (2026-08-15, `runs/v2/gate_v0333_main_timing/`, commit
`a4bce9c`):** `t_unit = 351s` (`eq_pres` seed 0, `K=2`, Mac MPS, `torch
2.12.0`). `T_total = 33 * 351s * 1.15 = 3.70h`. **3.70h <= 48h -> proceed
all-MPS, per the pre-registered rule.** No CUDA fallback triggered. This
decision was reached automatically per the rule above, before any gate
result was read, and is not subject to further approval.

### 5.6 Runner (v0.33.3 item D)

`scripts/run_method_gate.py` fixes the 33-unit grid above into one
resumable invocation (reusing `scripts.cl_main.run_sequence`, so all
per-row behavior — inline utility columns, `torch.save` checkpoints,
per-refresh diagnostics — is identical to `cl_main.py`). After each
(method, seed, K) unit: CSV rows are appended, the navigator checkpoint is
on disk, and `checkpoints.json` is rewritten with a per-unit manifest entry
(`method`, `seed`, `K`, `status`, `git_commit`, `backend`, `torch_version`,
wall-clock seconds — via `nav.device.run_provenance()`) *before* the next
unit starts. Safe to `Ctrl-C` between units (never mid-unit) and continue
later with `--resume`. Launch pattern: `tmux new -s gate; caffeinate -i
python scripts/run_method_gate.py --device mps --tag <tag> --resume`.
`tmux` is session persistence only, **not** sleep prevention — keep the
lid open and the machine on power for the duration of a run; safe-stop
(finish the current unit, do not kill mid-unit) before moving the machine.

## 6. Pre-registered PASS criteria

Unchanged from v0.33; the g2/g4 aggregation definitions below are written
down precisely as of v0.33.2 (item 6) to close off post-hoc interpretation.

A config passes **iff ALL** of:

- **g1.** Forgetting: paired mean diff vs. `ours_uniform` AND vs. `distill`
  is `<= 0` (parity or better) at every `K`, with 3/3 seeds non-worse at
  `K in {1,2}`.
- **g2.** At least one axis improves vs. BOTH comparators with 3/3 seed sign
  consistency at `K=1` or `K=2`: capability (Forgetting/AA) OR behavior
  (Jaccard/action-KL) OR utility (ε-optimal mass higher, or normalized regret
  lower, aggregated exactly as in §5.1: final stage, old tasks only, mean
  within task then macro-average across old tasks; comparator values must
  pass the §5.2 checksum before being used).
- **g3.** Not worse than `replay` on Forgetting at any `K` (paired mean
  within +0.005).
- **g4.** New-task plasticity `A[t,t]` not degraded vs. `ours_uniform` by
  more than 0.01 in any cell. **Precise definition (v0.33.2 item 6):** for
  each `(task, K)` cell, compute the 3-seed **paired mean** own-time `A[t,t]`
  difference vs. `ours_uniform`; the cell fails iff that mean difference
  `< -0.01`. This is evaluated on the seed-mean, **not** per-single-seed —
  a single noisy seed cannot fail a cell on its own.

Report the gate verdict per config in `results/method_gate_verdict.md` with
the paired tables (mean diff, 95% CI, per-seed sign agreement — descriptive
only, per the v0.33.1 statistics policy, no p/q language). **STOP after the
verdict and wait for external review** before any promotion run.

## 7. Promotion (only after external OK)

Passing config(s) -> 5 seeds x `{main, reverse}` x `K={1,2,4}`, appended as
**new rows** to fresh copies of the main tables (Protocol-v1 files remain
immutable — new rows live in `runs/v2/<tag>/` and fresh result files, never
overwrite `cl_main_can_*_full_s*.csv`). Extend `paired_stats_pack.py` and
`plasticity_report.py` with the new rows. Deadline for promoted rows: 8/23.

Method public name is decided externally on 8/18; use the `ia_ep`-family
code labels above until then. Framing rule: if a config passes, the
mechanism analysis (§0) is the *motivation* for the method; if none passes,
the paper stays analysis-centric and this section is reported honestly in an
appendix. Never claim universal superiority; every forbidden phrase in
`docs/research_contract_v0.292.md` still applies (`scripts/check_forbidden_phrases.py`
is the automated check).

## 8. Run-handling rules

- **r1.** If gate runs have NOT started under any superseded (§2.1, §2.2)
  definition, or under any superseded backend design: apply the pending
  patch, bump this document with a dated changelog entry stating the
  amendment precedes any result readout, then launch. **This is the branch
  that applies here** — no gate training has been run under any definition
  of M1 (neither §2.1 nor §2.2), and no gate training has been run under the
  originally-floated CUDA-matched backend design (superseded by §5.4/§5.5),
  before this v0.33.3 revision was committed.
- **r2.** (Not applicable — kept for completeness.) If gate runs had started
  under the degenerate §2.1 definition, `samp_drift_only`'s results would
  remain valid (the buggy `ia_samp` is mathematically identical to
  `samp_drift_only` when `u_bar(s) ≡ 1`), and buggy `ia_samp`/`ia_ep` runs
  would be discarded or relabeled as drift-only duplicates, then rerun under
  the corrected definition with fresh tags.
- **r3.** After any gate result has been read by anyone (Aaron, Fable, or
  Sol), this file is immutable; further changes require a new gate version
  (e.g. `v0.34`) and must be reported as such in a changelog entry, not a
  silent edit.

## 9. Changelog

- **v0.33 (2026-08-15, pre-red-team):** initial M1/M2/M3 spec drafted from
  `Fab-v0.33-兩軌制-1`. M1 used the within-state min-max-normalized top-gain
  value as `u_bar(s)` — never implemented in code, caught by Sol's red-team
  before any training run.
- **v0.33.1 (2026-08-15, this revision, before any gate result is read):**
  M1's cross-state importance factor redefined from the degenerate
  within-state-normalized `u_bar(s)` to `u_state(s) = percentile_rank(raw
  max gain)` computed across the buffer (§2.1–2.2). Added implementation
  clarifications i1 (boundary-policy snapshot semantics), i2 (`L_eq`
  numerical guard), i3 (per-refresh diagnostics). Gate criteria g1–g4,
  seeds, K, orders, and baselines are unchanged from v0.33. Implemented in
  `nav/cl.py` (`ImportanceReplaySampler`, `normalized_gain`,
  `epsilon_optimal_mask`, `percentile_rank`, `attach_boundary_policy`,
  `METHOD_GATE_KWARGS`) and `scripts/cl_main.py` (boundary-policy wiring,
  per-run diagnostics dump). Verified on Mac CPU and MPS smoke tests (2
  cohorts, 1 seed, `K=1`, 5/2 epochs) for all five configs — finite losses,
  CSVs written, existing frozen methods (`seqft`/`distill`/`ours`/
  `ours_uniform`) numerically unaffected by the refactor. **No gate training
  run (§6) has started as of this commit** — this document and the
  implementation diff are being sent for focused red-team (Sol, Fable) before
  launch, per r1.
- **v0.33.2 (2026-08-15, this revision, joint Sol+Fable focused red-team,
  both GO after this patch; before any gate result is read):** Fable's
  round-1 GO surfaced two rider items which Sol's round-2 focused red-team
  turned into concrete, pre-launch-mandatory patches. Amendments 1–7 below
  all precede any gate training run under any definition of M1 (§2.1, §2.2,
  or §2.3) — r1 still applies.
  1. **M1 utility calibration (§2.2 -> §2.3):** `u_state(s)` changed from a
     global-buffer percentile rank (v0.33.1, now itself superseded — §2.2)
     to a **source-task-chunk-wise** percentile rank of raw max gain,
     computed per chunk in `buffer_by_task` before flattening
     (`nav.cl.chunkwise_percentile_rank`). Rationale: different tasks' raw
     gains come from different frozen evaluators with different class
     counts/base entropy, so a pooled cross-task percentile could let
     evaluator-scale differences masquerade as state importance
     (formulation risk, not a coding bug). Cross-task replay pressure is now
     carried entirely by `d_t(s)`; no new hyperparameter.
  2. **Inline utility-axis metrics (§5.1):** `run_sequence` now computes
     `eps_optimal_mass`/`normalized_regret` per probe state at every stage
     (not just gate configs) and writes them as two new CSV columns,
     reusing the `policy_on_probes` forward pass already done for
     `action_kl`. Pre-registered g2 aggregation: final stage, old tasks
     only, mean within task then macro-average across tasks.
  3. **Comparator utility metrics (§5.2):** frozen `ours_uniform`/`distill`
     rows predate these two columns and are never retrofitted; a probe-only
     regeneration (seeds {0,1,2}, `K∈{1,2}`, main order) backfills them
     under a separate `runs/v2/util_regen_<tag>/` artifact, gated by a
     pre-declared `|Δ|<=0.001` checksum against the frozen AA/Forgetting/
     Jaccard/action-KL values before its utility columns are trusted as g2
     comparators. **Backend correction:** the patch text said Mac CPU; the
     actual frozen-grid logs (`logs/cl_main_s*.log`, `logs/abl_ablA1_s*.log`)
     show `device = mps` for every seed of the `can` grid (main, reverse,
     all ablations). A CPU regen failed the checksum on 12/12 rows; the
     identical command on Mac **MPS** passed. `docs/method_gate_v033.md`
     §5.2 and `scripts/verify_util_regen_checksum.py` use MPS.
  4. **Real checkpoints (§5.3):** gate-config runs now `torch.save` the
     navigator at every stage boundary under `runs/v2/<tag>/ckpt/`;
     `checkpoints.json` remains `--resume` bookkeeping only, unchanged.
  5. **Diagnostics (§2.5):** added per-refresh sampling probability mass by
     source task, alongside the existing entropy/`corr(u_state, d_t)` log.
  6. **g4 precision (§6):** "not degraded by >0.01 in any cell" now
     explicitly means the 3-seed **paired mean** own-time `A[t,t]`
     difference per `(task, K)` cell versus `ours_uniform`; not evaluated
     per-single-seed.
  7. **Buffer-cap doc fix (§1):** corrected "globally-highest-utility fill
     to capacity" -> "within-chunk"; `buffer_cap=512` is per source-task
     chunk, chunks concatenate (so the true replay pool can exceed 512 with
     multiple old tasks) — also corrected in `docs/audit_20260815.md` and
     the memory-ablation description in the paper.
  Implemented in `nav/cl.py` (`chunkwise_percentile_rank`,
  `flatten_task_labels`, `ImportanceReplaySampler.u_state`/`task_labels`
  params) and `scripts/cl_main.py` (chunk-wise ranks wired into
  `run_sequence`, inline utility metrics, `torch.save` checkpointing).
  Verified on Mac CPU/MPS smoke tests — finite losses, new CSV columns
  populated and in `[0,1]`-ish range, frozen methods numerically unaffected.
  **No gate training run (§6) has started as of this commit** — both Sol
  and Fable are joint-GO conditional on this patch; launch (gate + pilot40 +
  util_regen) follows immediately after this commit, per r1.
- **v0.33.3 (2026-08-15, this revision, final pre-launch amendment, joint
  Sol+Fable, pre-unblinding; no formula/threshold changes):** triggered
  solely by the backend-reproduction audit surfaced while executing v0.33.2
  item 3 (§5.2) — a CPU regen of `ours_uniform`/`distill` failed the
  checksum on 12/12 rows (`Δ` up to 0.148) while the identical command on
  MPS reproduced the frozen Protocol-v1 rows bit-exactly. This established
  backend/execution-path as a material protocol variable, so the originally
  planned "CUDA new methods vs. MPS frozen baselines" gate comparison was
  retracted before any gate training ran. **No gate result has been read
  under this or any prior amendment.** M1/M2 formulas, epsilon, lambda,
  seeds, K, order, and the g1–g4 thresholds are unchanged from v0.33.2;
  this amendment is scoped entirely to experimental-control/backend design:
  1. **CPU→MPS provenance correction committed (item A):** the v0.33.2
     commit (including its CPU→MPS correction, §5.2) is now landed. The
     failed CPU regen's launch log and a written explanation are archived
     at `runs/v2/util_regen_20260815_cpu_FAILED/` (the CPU run's own output
     CSV was lost to working-directory cleanup before archiving; the launch
     log's per-unit summary and the previously-computed `Δ≈0.148` finding
     are preserved as evidence, not used in any table). Protocol-v1 is now
     documented everywhere as historically produced on Apple MPS
     (`docs/audit_20260815.md`, `docs/compute_policy.md`, this file).
  2. **Backend design changed to MPS-first (§5.4):** the gate now runs
     entirely on Mac MPS. Frozen Protocol-v1 MPS rows + the bit-exact MPS
     `util_regen` (§5.2) serve as same-backend comparators; **no comparator
     retraining is needed** (cheaper and cleaner than the CUDA-matched
     alternative, which would have required 60 total runs instead of 33).
     Gate compute = exactly the 33 new-method runs. WP5 `pilot40` remains on
     RunPod CUDA in parallel — a self-contained table with no frozen-row
     backend to match, unaffected by this change.
  3. **Timing decision rule executed (§5.5):** the pre-registered
     representative-unit timing check (`eq_pres`, seed 0, `K=2`, MPS) ran
     for `t_unit=351s`; `T_total = 33*351s*1.15 = 3.70h <= 48h`, so the rule
     resolved to "proceed all-MPS" automatically, with no CUDA fallback
     triggered.
  4. **Runner added (§5.6):** `scripts/run_method_gate.py` pins the 33-unit
     gate grid into one atomic-resume invocation and writes a per-unit
     manifest (`git_commit`/`backend`/`torch_version`/wall-clock seconds via
     the new `nav.device.run_provenance()` helper, also wired into
     `scripts/cl_main.py`'s metadata for every run in this project, not
     just the gate).
  5. **Compute policy established (`docs/compute_policy.md`):** a
     repo-level contract — CPU never for formal training/regeneration/
     evaluation (post-processing only); MPS is the default backend for all
     formal Track-A/Track-B runs; CUDA only inside an explicitly opened
     backend-matched protocol; never present a cross-backend delta as a
     method effect; torch version pinned until submission; every
     result-producing run logs device/torch/host/commit/seed/tag/command.
  Implemented in `nav/device.py` (`run_provenance`), `scripts/cl_main.py`
  (provenance wired into `metadata.json`), `scripts/run_method_gate.py`
  (new), `docs/compute_policy.md` (new),
  `runs/v2/util_regen_20260815_cpu_FAILED/` (new, archived diagnostic
  evidence), and this document (§5.4–§5.6, r1, title/status). **After this
  commit the pre-registration freezes per r3** — gate launch (Mac MPS,
  `tmux`+`caffeinate`, per §5/§5.6) and WP5 `pilot40` launch (RunPod CUDA)
  proceed immediately and in parallel; the process stops at the gate
  verdict for joint unblinding review, with no promotion runs before that
  review.
- **Gate verdict computed (2026-08-16, by Cursor — analysis only, NOT a
  red-team-reviewed decision, NOT a protocol amendment):** the 33-unit Mac
  MPS gate (`runs/v2/method_gate_v0333_run1/`) finished; `scripts/
  method_gate_verdict.py` applied the g1–g4 thresholds exactly as frozen
  above, against the frozen Protocol-v1 `distill`/`replay`/`ours_uniform`
  rows and the checksum-verified MPS `util_regen` backfill (§5.2), with
  **zero threshold or formula choices made after seeing the results**.
  Full table: `results/method_gate_v0333_verdict.md`.
  - **All three main configs (`ia_samp`/M1, `eq_pres`/M2, `ia_ep`/M3) GATE
    FAIL** as pre-registered. `eq_pres` is the closest: **g2 PASS** (clean,
    sign-consistent 3/3-seed utility-axis win — higher `eps_optimal_mass`,
    lower `normalized_regret` — vs both comparators at K=1 and K=2) and
    **g3 PASS** (not worse than `replay` at any K), but **g1 FAIL** (only
    at K=1) and **g4 FAIL** (own-time accuracy on `brca`, the last task, at
    both K=1 and K=2).
  - **Disclosed data gap:** frozen `ours_uniform` was never run at K=4 in
    Protocol-v1 (all 5 archived seeds are K∈{1,2} only) — every g1/g4
    sub-check against `ours_uniform` at K=4 is reported **N/A**, not
    silently passed, per r1 (no new comparator training at verdict time).
  - **Seed-sensitivity diagnostic (informational, does not overturn the
    pre-registered verdict):** of the 18 individual FAIL sub-checks across
    g1/g4, **15 are single- or partial-seed-driven** (1 of 3 seeds flips an
    otherwise-passing mean past threshold), not a consistent 3/3 signal —
    e.g. `eq_pres` K=1 `brca` A[t,t] is bit-identical to `ours_uniform` for
    seeds 0,1 and only diverges on seed 2. This is disclosed as a candid
    read on statistical power (3 seeds x `n_test=15`), not as grounds to
    override g1/g4 as written.
  - **This is an analysis artifact awaiting joint Sol+Fable unblinding
    review, per the r1/r3 process** — Cursor has not chosen a remediation
    path, retrained anything, or altered any threshold. Candidate next
    steps (listed in `results/method_gate_v0333_verdict.md`, all
    conditional on red-team sign-off before any new compute): widen the
    seed set on the FAIL cells to check for noise-washout; backfill
    `ours_uniform` at K=4 to close the N/A gap; investigate whether the
    `brca`/last-task plasticity dip is training-order noise rather than an
    M1/M2-specific effect; or, if the verdict holds, report `eq_pres` as an
    honest negative/mixed appendix result (real utility-axis gain traded
    against a borderline, seed-sensitive forgetting/plasticity cost).
- **Red-team unblinding review, round 1 (2026-08-16, Fable + Sol +
  Fable-final, external, not Cursor):** independently re-verified the
  verdict computation (no arithmetic errors found), then reframed the
  result. Two flagged process items, both resolved same-day: (1) a Fable
  flag that WP5 `pilot40` "had not started" was checked against the repo
  and found to be **a stale-information false alarm** — `pilot40`'s main
  grid had completed and been pushed on 8/15 (commit `45fb7cf`), one day
  before the flag; (2) the existing pilot40 Gate-1' significance result
  predated that same run by one day, so it was rerun end-to-end
  (`results/gate1_significance.md`, commit `05ab259`) and reproduced
  **bit-identically**, confirmed structurally impossible to have been
  affected (independent code path, never touches `nav/cl.py`). Substantive
  ruling: `ia_samp` (M1) and `ia_ep` (M3) **RETIRED** — no axis improvement,
  multiple 3/3-seed-consistent FAILs, RETIRE without further compute.
  `eq_pres` (M2) reframed as **intervention evidence, not just an analysis
  artifact**: it shows a fully sign-consistent utility-axis improvement
  (3/3 seeds, both K, both comparators) traded against reduced behavior
  fidelity (Jaccard down, action-KL up — expected by design, since the loss
  explicitly permits non-exact-imitation actions), with capability/
  plasticity margins decided by a handful of high-variance cells rather
  than a consistent signal. Sol's focused correction (accepted by Fable):
  do **not** justify a rescue run by claiming the g4 threshold sits below
  measurement resolution (g4 already averages over 3 seeds, so its
  effective resolution is finer than any single-seed accuracy quantum —
  Fable's own §-heading claiming otherwise is **retracted**); the correct,
  defensible justification is that v0.33.3 was an intentionally small
  3-seed **screening** gate and `eq_pres`'s effect-size estimate warrants a
  pre-registered, seed-fixed extension for stability, not a redesign of the
  gate. Also ruled: pilot40 capability findings must be worded "unresolved
  in this small two-task validation setting; consistent with masking, not
  validated" (not "masking confirmed" — that overclaims from a
  wide-CI-crossing-zero result); `ours_uniform` K=4 backfill is legitimate
  as **post-unblinding comparator completion**, never mislabeled as frozen
  Protocol-v1.   Full transcripts of both rounds are the authority for any
  wording dispute; §10 below is the operational translation into a runnable
  spec.
- **Gate v2 verdict computed (2026-08-16, by Cursor — analysis only, NOT a
  red-team-reviewed decision, NOT a protocol amendment):** the pre-registered
  19-unit Mac MPS extension (`runs/v2/gate_v2_20260816T073200Z/`, 2h19m
  wall-clock, matching the §10.2 estimate) finished cleanly (exit 0, all 19
  (method,seed,K) cells present, 10 rows each as expected). The seeds{3,4}
  utility-axis backfill checksummed **8/8 bit-exact** against the frozen
  `full_s{3,4}.csv`/`ablA1_s{3,4}.csv` rows (`results/
  util_regen_checksum_v2_seeds34.md`) before being admitted as g2
  comparators — same reproducibility pattern as every other MPS regen in
  this project. `scripts/method_gate_v2_verdict.py` applied the §10.1
  thresholds exactly as frozen, with zero threshold/formula choices made
  after seeing the results. Full table: `results/method_gate_v2_verdict.md`.
  - **GATE v2 FAIL for `eq_pres`.** g1=FAIL, g2=**PASS**, g3=PASS, g4=FAIL;
    §10.3 replication rider **not triggered** on any cell (no PASS cell was
    carried by a pooled-mean rescue against both new seeds' individual
    direction — every PASS cell has at least one of seeds 3/4 on the passing
    side too).
  - **g2 (utility) came back stronger, not weaker, with the extension:**
    both `eps_optimal_mass`-higher and `normalized_regret`-lower now hold
    5/5 seeds (not just the original 3/3) vs BOTH `ours_uniform` and
    `distill`, at both K=1 and K=2 — the utility-axis effect that motivated
    Gate v2 in the first place replicated cleanly on the two held-out new
    seeds.
  - **g1 (capability) FAILs at K=1 only** (K=2, K=4 PASS) — consistent with
    the original v0.33.3 finding, not a new problem; new seeds (pooled mean
    `+0.0056` vs `ours_uniform`, `+0.0026` vs `distill`) confirm rather than
    reverse the K=1 direction.
  - **g4 (plasticity) newly FAILs at 2 cells that the original 3-seed
    screening gate had passed or not clearly failed:** `lung` K=4 (old3
    mean `-0.0079`, just inside the old -0.01 tolerance; pooled5 with the 2
    new seeds `-0.0129`, now over) and, more substantially, `brca` K=2 (old3
    mean `-0.0175`; new2 mean `-0.0590`, with **both** new seeds
    individually well past the threshold — seed3 `-0.0195`, seed4 `-0.0986`,
    the latter roughly 9x the cell's own accuracy quantum, not a
    single-item-flip artifact). **This is the extension doing exactly what
    it was pre-registered to do:** more seeds sharpened a real signal that a
    thin 3-seed screen could not resolve, rather than averaging noise away.
    It does not indicate the new seeds are anomalous — the direction is the
    same sign as the old seeds at both cells, just larger.
  - **Per §10.5/§10.6 (pre-committed before this result was seen):** this
    was the last main-order method-rescue compute for `eq_pres` regardless
    of outcome. No lambda tuning, no epsilon change, no fail-cell-only
    reruns were performed or will be performed. `eq_pres` is **not
    promoted** and is reported as an **intervention/mechanism result** in
    the main analysis paper (§10.6 FAIL branch): a real, 5/5-seed
    utility-axis improvement traded against a real, extension-confirmed
    capability/plasticity cost concentrated at `brca` (the last, hardest
    task) and `lung` K=4.
  - **This is an analysis artifact awaiting joint Sol+Fable+Aaron
    unblinding review** — Cursor has not chosen any further remediation
    path, retrained anything, or altered any threshold. Per the §10.5
    scope-discipline clause, there is no pending "next step" compute
    proposal this time (unlike the v0.33.3 verdict): the extension gate
    is now closed, and the paper-writing question is how to frame `eq_pres`
    in the mechanism-analysis section, not whether to run more seeds.

- **v0.34 opened; §10.5/§10.6 method-compute clause superseded (2026-08-16,
  joint Aaron+Sol+Fable sign-off, external decision — this entry is a
  pointer, not a re-scoring):** the joint review ratified the Gate v2 FAIL
  verdict in full and then, on PI decision, opened a **new** pre-registered
  method track, `docs/method_gate_v034.md`, on the grounds that Gate v2
  produced a specific, mechanistic, fixable diagnosis (utility preserved
  5/5 seeds; preservation gradients conflicting with new-task acquisition)
  rather than a random failure. **Nothing in §0-§10 of this document
  changes.** The Gate v2 verdict remains **FAIL** and is never
  reinterpreted; `eq_pres` remains **unpromoted** and enters v0.34 only as a
  frozen diagnostic precursor (no rerun as a candidate, no re-scoring, no
  lambda/epsilon tuning, no fail-cell-only reruns). What is superseded is
  strictly the reading of §10.5's "last main-order method-rescue compute"
  and §10.6's "No further method-rescue compute" as a ban on any subsequent
  hypothesis-driven method work; v0.34 carries its own one-shot config set
  (three configs, no fourth), its own hard calendar stops, and an explicit
  "no v0.35 for this ICASSP submission regardless of outcome" clause. Full
  amendment text, criteria, and sign-off chain: `docs/method_gate_v034.md`
  §0-§1.

## 10. Gate v2 — Extension (`eq_pres` only, joint Sol+Fable sign-off,
2026-08-17)

**Non-retroactivity clause.** The v0.33.3 gate (§0-§8) and its FAIL verdict
for all three configs are **frozen and never reinterpreted**. Gate v2 is a
**newly declared extension gate**, scoped to `eq_pres` only. It does not
reopen, re-score, or retroactively pass/fail anything decided under §6.
Framing (per Sol's correction, replacing an earlier, now-retracted "gate
threshold sits below measurement resolution" justification): v0.33.3 was an
intentionally small 3-seed **screening** gate; `eq_pres` showed a fully
sign-consistent utility-axis improvement while its capability/plasticity
margins were decided by a few high-variance cells; v2 raises effect-estimate
stability with 2 more, pre-committed seeds. This is **not** a claim that
v0.33.3 was mis-designed, and **not** a claim that any threshold sat below
instrument resolution.

(Provenance note, added 2026-08-16 CST while opening v0.34: this section's
header date follows the sign-off text as written, whereas the Gate v2 runs
themselves executed on 2026-08-16 CST under the UTC run tag
`gate_v2_20260816T073200Z`. Recorded here for the audit trail only — no
criterion, threshold, number, or verdict in §10 is changed by this note.
Repo-wide convention going forward, stated in `docs/method_gate_v034.md`:
document dates are local CST calendar dates, run tags are UTC timestamps.)

### 10.1 Old -> v2 criteria mapping (explicit, nothing silently changed)

| axis | v0.33.3 (§6, frozen, unchanged) | Gate v2 (new, this section) |
|---|---|---|
| capability (g1) | 3-seed paired mean `dForgetting <= 0` vs `ours_uniform` AND `distill`, per K, **plus** 3/3-seed non-worse at K∈{1,2} | **Pooled 5-seed** paired mean `dForgetting <= 0` vs both comparators, per K — **mean-only**; the old 3/3-seed clause is replaced by the §10.3 replication rider, not by a looser mean rule |
| utility (g2) | >= 1 axis (capability/behavior/utility), 3/3-seed sign-consistent vs both comparators, K=1 or K=2 | **Utility axis only** (`eps_optimal_mass`/`normalized_regret`), **>=4/5 seeds** improvement direction vs BOTH comparators, K=1 or K=2. *This is a new extension criterion defined after the v0.33.3 gate was unblinded; it does not retroactively modify the original 3/3 criterion.* |
| replay safety (g3) | pooled/3-seed mean `dForgetting` vs `replay` `<= +0.005` | **Unchanged**: 5-seed pooled mean `dForgetting` vs `replay` `<= +0.005` |
| plasticity (g4) | per `(task,K)` 3-seed paired mean `dA[t,t] >= -0.01` vs `ours_uniform` | **Unchanged threshold**, extended to **5-seed** paired mean per `(task,K)` cell. Each cell footnoted with its per-task test-set accuracy quantum (`1/n_test`) — **descriptive only**, not used to argue the threshold is wrong (Sol's correction, §9 changelog above) |

`eq_pres` seeds 0-2 are **not rerun** — they are the same frozen v0.33.3
rows. Seeds 3,4 are new. All four criteria are reported three ways (§10.4).

### 10.2 Runs (Mac MPS, `tmux` + `caffeinate`, atomic resume, fresh tags,
checkpoints ON)

| # | units | what | status |
|---|---|---|---|
| 1 | 6 | `eq_pres`, seeds **{3,4}** x K∈{1,2,4}, main order | genuinely new candidate-arm training |
| 2 | 5 | `ours_uniform`, seeds **{0..4}** x **K=4**, main order | genuinely new — Protocol-v1 never ran this method at K=4 for any seed. Labeled **"post-unblinding comparator completion for Gate-v2"** everywhere in tables/audit text; never labeled frozen Protocol-v1 (Sol §8 ruling) |
| 3 | 8 | `distill` + `ours_uniform`, seeds **{3,4}** x K∈{1,2}, main order | **not new experiments** — `bal_acc`/Forgetting/Jaccard/action-KL for these (method,seed,K) cells are already frozen (`..._full_s{3,4}.csv`, `..._ablA1_s{3,4}.csv`); this rerun only exists to backfill the `eps_optimal_mass`/`normalized_regret` columns those frozen rows predate (same v0.33.2 §5.2 technique already checksum-verified for seeds 0-2, extended verbatim to seeds 3-4). **Gated by the same `|Δ|<=0.001` checksum** against those frozen rows before any utility value from this rerun is trusted as a g2 comparator. Disclosed here, not in the original red-team prompt text, because without it g2's "new2"/"pooled5" columns would be structurally N/A for `eq_pres` vs these two comparators — it is data-completeness infrastructure, not a new analysis choice, and follows the same legitimacy test Sol applied to run #2 (missingness is pre-existing coverage, not motivated by an observed bad number, and checksum-gated). |

Total: **19 units**, all Mac MPS. Estimated from the actual v0.33.3
per-unit wall-clock (`eq_pres` K∈{1,2,4}: 184/348/693s; `ours_uniform`/
`distill` K∈{1,2}: ~240-500s): **~2.5h** wall-clock, within the ~3-4h Sol/
Fable estimated.

### 10.3 Replication rider (disclosure rule, hard for promotion)

If the pooled 5-seed result PASSES a criterion but **both** new seeds (3
and 4) individually move toward failure on that same critical metric/cell,
the result must **not** be called a robust confirmation and `eq_pres` is
**NOT promoted** on the strength of that criterion, regardless of the
pooled mean. This is checked and reported per g1/g4 cell in
`results/method_gate_v2_verdict.md`, not silently absorbed into the pooled
number.

### 10.4 Reporting

Every criterion (g1, g2, g3, g4) is reported **three ways**: **old3**
(seeds 0-2, i.e. the already-unblinded v0.33.3 numbers, unchanged), **new2**
(seeds 3-4 alone), **pooled5** (all 5 seeds) — so a reader can see whether
seeds 3-4 confirm or merely average-rescue the old direction, per Sol's
explicit anti-cherry-pick requirement (§9). **Pooled5 is primary** for the
pass/fail call; **old3/new2 are secondary evidence**, always shown
alongside, never omitted. In addition: 95% bootstrap CI (`paired_stats_pack.
bootstrap_ci`, `n_boot=10000`, same convention as elsewhere in this repo) is
reported for every pooled mean as an **uncertainty rider only** — explicitly
**not** a pass/fail test, per Sol's ruling against growing a second
hypothesis-testing regime alongside the estimation-first policy. Each g4
cell is footnoted with its per-task test-set accuracy quantum
(`1/n_test`, descriptive only).

### 10.5 Seed fixation / scope discipline

Seeds 3 and 4 are fixed now, run together in one command, no seed-by-seed
peeking or early stopping. **This is the last main-order method-rescue
compute for this paper cycle**, regardless of outcome: no lambda tuning, no
epsilon change, no fail-cell-only reruns, no new modules/losses.

### 10.6 Branches

- **PASS** (per §10.1 thresholds, with the §10.3 rider not triggered) ->
  `eq_pres` reverse-order promotion run: 5 seeds x K∈{1,2,4}, Mac MPS,
  labeled "Gate-v2 promotion." Method line named toward
  "equivalence-preserving continual evidence navigation"; the naming
  collision check (originally 8/18, deferred by Fable to trigger on v2
  PASS) runs at this point.
- **FAIL** (or PASS with the §10.3 rider triggered) -> `eq_pres` is reported
  as an **intervention/mechanism result** in the main analysis paper (not a
  promoted method): a real, disclosed utility-axis improvement traded
  against a capability/plasticity cost that a 5-seed extension could not
  resolve as parity. No further method-rescue compute.

STOP after `results/method_gate_v2_verdict.md` is written, for joint
Sol+Fable+Aaron unblinding review. **No promotion runs before that review**,
regardless of which branch the numbers point to.

### 10.7 Carry-over (zero compute, same commit train)

1. **Pilot40 sel-utility audit:** `trajectory_utility()` (`nav/cl.py`) is
   called identically for `can` and `pilot40` — same function, same
   `run_sequence` code path (`scripts/cl_main.py`), zero dataset-specific
   branching; sign convention (`loss_now - loss_next`, positive = the
   zoomed-in evidence reduced cross-entropy loss) is defined once and
   applies to both datasets by construction, not by convention duplicated
   in two places. **Conclusion: definition/sign/aggregation-identical.**
   pilot40's `sel_utility` column is legitimately negative in several
   (method, K) cells (evaluator/case-specific evidence choices that raise
   rather than lower loss are a real, non-buggy possible outcome of the
   same formula) — kept as-is, explained honestly in any paper table, never
   re-signed or re-normalized for cosmetics, per Sol/Fable's explicit
   instruction.
2. **Paper wording locks** (apply when paper prose reaches these claims;
   `paper/main.tex` does not yet contain pilot40-capability or M1/M3 prose
   to retrofit as of this commit — locked here so the first draft is
   correct on arrival): pilot40 capability = *"unresolved in this small
   two-task validation setting; consistent with masking, not validated"*
   (never *"masking confirmed"* or *"validated"*). `ia_samp` (M1) and
   `ia_ep` (M3) = **RETIRED**, reported in the appendix as tested-negative
   with their full per-seed tables (never silently dropped). Two new
   patterns added to `scripts/check_forbidden_phrases.py` to make the first
   lock machine-checked (`masking confirmed`, bare `pilot40.*validated`
   without a negation cue nearby).
3. **Milestone report format:** every future Track A/B milestone report
   (chat or doc) opens with a two-line status header — one line each for
   Track A and Track B current state — so a reader never has to infer
   "not mentioned" as "not done" (the direct process fix for the 8/16
   pilot40 false-alarm flag). Applied starting with this document's Status
   line above and the report accompanying this commit.
