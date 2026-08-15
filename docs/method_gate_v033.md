# Method gate v0.33.1 — pre-registration

**Status:** committed BEFORE any gate training run (v0.33.1 revision r1
applies — see Changelog). Pre-registration discipline: this file is
immutable once any gate result below has been read by anyone (r3); further
changes require a new gate version, reported as such.

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
globally-highest-utility fill to capacity) is byte-identical to every frozen
Protocol-v1 method. This is intentional: it prevents any gate config from
being confounded with the buffer-truncation ablation audited in
`docs/audit_20260815.md` §"Naming and scope corrections" / §2.4 of the
handoff.

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

### 2.2 Current definition

```
u_state(s) = percentile_rank( max_a u_s(a) )  computed ACROSS all states
             currently in the buffer, using the RAW (un-normalized)
             counterfactual gain — i.e. TeacherStep.utility, which is
             already max_a gain(a) clamped at 0.
```

- Ranks are recomputed whenever the buffer is (re)built — once per stage,
  since buffer composition is fixed within a stage.
- Cross-state importance (`u_state`) must **never** use within-state
  min-max-normalized values — that normalization is correct and unchanged
  for a different question (§3, M2's `A_eps`, and all mechanism metrics).
  `u_state` answers "which state matters more than other states"; the
  within-state normalization in M2 answers "which actions are near-optimal
  inside this one state." Conflating the two is exactly the v0.33 bug.
- Implementation: `nav.cl.percentile_rank` (`scipy.stats.rankdata`,
  `method="average"`, divided by buffer size), consumed by
  `nav.cl.ImportanceReplaySampler`.

### 2.3 `d_t(s)` — drift factor

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

### 2.4 Numerical/attribution details

- Both `u_state` and `d_t` are clipped to `>= 1e-8` before exponentiation so
  `alpha=0` or `beta=0` (mini-arms, §5) yields a clean constant factor
  (`x**0 == 1`) rather than a `0**0` edge case.
- **i3. Diagnostics logged at every refresh:** sampling-distribution entropy
  (`-sum(p log p)`), and the Pearson correlation between `u_state` and `d_t`
  across the buffer (redundancy check for whether the two factors carry the
  same information). Written to `runs/v2/<tag>/diag_<method>_seed<S>_K<K>.json`
  by `scripts/cl_main.py`.

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
`distill` (`results/cl_main_can_main_full_s*.csv`) and `replay` — never
rerun; new runs are paired seed-by-seed against these frozen CSVs.

Launch command (RunPod CUDA, inside `tmux`, fresh run tag, checkpoints ON,
per `docs/RUNPOD_SOP.md`):

```bash
python scripts/cl_main.py --dataset can --order main \
  --methods ia_samp eq_pres ia_ep samp_util_only samp_drift_only \
  --seeds 0 1 2 --budgets 1 2 4 --device auto \
  --tag method_gate_v0331_<run_tag> --resume \
  2>&1 | tee logs/method_gate_v0331_<run_tag>.log
```

(`samp_util_only`/`samp_drift_only` only need `--budgets 1`; run them in a
separate invocation to avoid wasting compute on `K∈{2,4}` for the mini-arms.)

## 6. Pre-registered PASS criteria (unchanged from v0.33)

A config passes **iff ALL** of:

- **g1.** Forgetting: paired mean diff vs. `ours_uniform` AND vs. `distill`
  is `<= 0` (parity or better) at every `K`, with 3/3 seeds non-worse at
  `K in {1,2}`.
- **g2.** At least one axis improves vs. BOTH comparators with 3/3 seed sign
  consistency at `K=1` or `K=2`: capability (Forgetting/AA) OR behavior
  (Jaccard/action-KL) OR utility (ε-optimal mass / regret from a
  probe-style eval pass on the gate checkpoints).
- **g3.** Not worse than `replay` on Forgetting at any `K` (paired mean
  within +0.005).
- **g4.** New-task plasticity `A[t,t]` not degraded vs. `ours_uniform` by
  more than 0.01 in any cell.

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

- **r1.** If gate runs have NOT started under the superseded (§2.1)
  definition: apply the v0.33.1 patch (done — see Changelog), bump this
  document to v0.33.1 with a dated changelog entry stating the amendment
  precedes any result readout, then launch. **This is the branch that
  applies here** — no gate training has been run under any definition of
  M1 before this document was committed.
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
