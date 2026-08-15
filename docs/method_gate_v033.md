# Method gate v0.33.3 — pre-registration

**Status:** committed BEFORE any gate training run (v0.33.3 revision r1
applies — see Changelog; no gate training has been run under any definition
of M1, and under no backend design, as of this commit). Pre-registration
discipline: this file is immutable once any gate result below has been read
by anyone (r3); further changes require a new gate version, reported as
such.

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
