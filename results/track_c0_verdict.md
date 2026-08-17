# Track C0 — Verdict (minimal stable-plastic navigator)

**Track A:** frozen; writing proceeds in parallel and was never blocked by C0. **Track B/C:** Track B is closed and archived (`docs/method_gate_v034.md` §0/§4, no v0.35); this document reports the Track C0 screening experiment only.

**C0 assumes ORACLE TASK IDENTITY at both train and test time** (`docs/track_c0.md` §1): the task index selects which adapter is active during training and during every evaluation, including evaluation of old tasks at later stages. This is a best-case assumption — a task-incremental setting with known task identity is strictly easier than the class-incremental or task-agnostic settings. **C0 makes no claim that survives without it.** Inferring task identity (a router) is future work and is explicitly not part of any C0 claim.

Computed by `scripts/track_c0_verdict.py` against `runs/v2/track_c0_20260817T014543Z/cl_main_can_main_track_c0_20260817T014543Z.csv` (12 units). `seqft`, `distill` and `eq_pres` are frozen, same-backend (Mac MPS) comparators and were not rerun. Thresholds are transcribed from `docs/track_c0.md` §7, committed before any C0 training run.

**C0 is a screening experiment, not a promotion gate.** Passing it licenses proposing C1 to the three-way review and nothing more. Criteria are aggregate-level with **no per-cell veto** (§6.4/§7): three seeds cannot resolve cell-level plasticity differences, which is the direct lesson of v0.34, where cell means were smaller than the per-item quantum `1/n_test` with per-seed spread of several quanta. Per-seed values and 95% bootstrap CIs are **descriptive disclosure**, never a pass/fail test.

## c1 — Plasticity: own-time `A[t,t]` ≥ `seqft` − 0.01 for every task (3-seed paired mean, pooled over K ∈ {1,2})

| config | task | mean ΔA[t,t] vs `seqft` [seeds 0,1,2] | 95% CI | c1 (this task) |
|---|---|---|---|---|
| `sp_nav` | esca | -0.0238 [-0.0357, -0.0670, +0.0312] | [-0.0670, +0.0312] | FAIL |
| `sp_nav` | lung | +0.0053 [-0.0102, +0.0152, +0.0109] | [-0.0102, +0.0152] | PASS |
| `sp_nav` | rcc | -0.0113 [-0.0297, +0.0102, -0.0143] | [-0.0297, +0.0102] | FAIL |
| `sp_nav` | brca | -0.0163 [-0.0131, +0.0034, -0.0391] | [-0.0391, +0.0034] | FAIL |
| `sp_nav_eq` | esca | -0.0238 [-0.0357, -0.0670, +0.0312] | [-0.0670, +0.0312] | FAIL |
| `sp_nav_eq` | lung | -0.0045 [-0.0153, -0.0103, +0.0119] | [-0.0153, +0.0119] | PASS |
| `sp_nav_eq` | rcc | -0.0014 [+0.0102, +0.0000, -0.0143] | [-0.0143, +0.0102] | PASS |
| `sp_nav_eq` | brca | -0.0515 [-0.0492, +0.0000, -0.1053] | [-0.1053, +0.0000] | FAIL |

Per-`(task, K)` breakdown, **descriptive only — no veto power** (§6.4):

| config | task | K | mean ΔA[t,t] vs `seqft` [0,1,2] |
|---|---|---|---|
| `sp_nav` | esca | 1 | +0.0208 [+0.0000, +0.0000, +0.0625] |
| `sp_nav` | esca | 2 | -0.0684 [-0.0714, -0.1339, +0.0000] |
| `sp_nav` | lung | 1 | -0.0000 [-0.0109, +0.0210, -0.0102] |
| `sp_nav` | lung | 2 | +0.0107 [-0.0095, +0.0095, +0.0320] |
| `sp_nav` | rcc | 1 | -0.0130 [-0.0287, +0.0102, -0.0204] |
| `sp_nav` | rcc | 2 | -0.0096 [-0.0306, +0.0102, -0.0083] |
| `sp_nav` | brca | 1 | -0.0305 [-0.0526, -0.0330, -0.0060] |
| `sp_nav` | brca | 2 | -0.0020 [+0.0263, +0.0398, -0.0722] |
| `sp_nav_eq` | esca | 1 | +0.0208 [+0.0000, +0.0000, +0.0625] |
| `sp_nav_eq` | esca | 2 | -0.0684 [-0.0714, -0.1339, +0.0000] |
| `sp_nav_eq` | lung | 1 | +0.0002 [+0.0000, +0.0006, +0.0000] |
| `sp_nav_eq` | lung | 2 | -0.0093 [-0.0306, -0.0211, +0.0238] |
| `sp_nav_eq` | rcc | 1 | -0.0096 [+0.0019, -0.0102, -0.0204] |
| `sp_nav_eq` | rcc | 2 | +0.0068 [+0.0185, +0.0102, -0.0083] |
| `sp_nav_eq` | brca | 1 | -0.0679 [-0.0722, -0.0330, -0.0985] |
| `sp_nav_eq` | brca | 2 | -0.0351 [-0.0263, +0.0331, -0.1120] |

## c2 — Stability: `Forgetting` ≤ `distill` at every K (3-seed paired mean)

| config | K | mean ΔForgetting vs `distill` [seeds 0,1,2] | 95% CI | c2 (this K) |
|---|---|---|---|---|
| `sp_nav` | 1 | +0.0728 [+0.0566, +0.0653, +0.0964] | [+0.0566, +0.0964] | FAIL |
| `sp_nav` | 2 | +0.0460 [+0.0467, +0.0332, +0.0583] | [+0.0332, +0.0583] | FAIL |
| `sp_nav_eq` | 1 | -0.0106 [-0.0200, -0.0270, +0.0153] | [-0.0270, +0.0153] | PASS |
| `sp_nav_eq` | 2 | -0.0053 [+0.0195, -0.0285, -0.0068] | [-0.0285, +0.0195] | PASS |

## c3 — Utility (`sp_nav_eq` only): Δ`eps_optimal_mass` > 0 vs `distill`

Aggregated by the frozen §5.1 convention (final stage, old tasks only, mean within task then macro-average across old tasks). `normalized_regret` is secondary descriptive evidence.

| config | K | mean Δε-mass vs `distill` [0,1,2] | 95% CI | mean Δregret (descriptive) | c3 (this K) |
|---|---|---|---|---|---|
| `sp_nav` | 1 | -0.1163 [-0.1157, -0.1215, -0.1118] | [-0.1215, -0.1118] | +0.0389 [+0.0370, +0.0401, +0.0395] | FAIL |
| `sp_nav` | 2 | -0.0414 [-0.0504, -0.0360, -0.0378] | [-0.0504, -0.0360] | +0.0116 [+0.0160, +0.0082, +0.0105] | FAIL |
| `sp_nav_eq` | 1 | +0.1429 [+0.1403, +0.1451, +0.1433] | [+0.1403, +0.1451] | -0.0445 [-0.0480, -0.0446, -0.0410] | PASS |
| `sp_nav_eq` | 2 | +0.1807 [+0.1628, +0.2065, +0.1728] | [+0.1628, +0.2065] | -0.0567 [-0.0513, -0.0695, -0.0493] | PASS |

`sp_nav` has no `L_eq` term, so its row above is **contrast, not a criterion** — c3 applies to `sp_nav_eq` only (§7).

## c4 — Disclosure: parameter and update counts

| quantity | value |
|---|---|
| shared core parameters | 164,481 |
| FiLM adapter parameters per task | 640 (**0.3891%** of core) |
| FiLM adapter parameters, all 4 tasks | 2,560 (**1.5564%** of core) |
| c4 bound | adapters < 5.0% of shared core — **PASS** |
| training units completed | 12 / 12 |
| per-unit wall clock (last unit) | 356.5 s |

Update counts: every config sees the same number of optimizer steps as any other method in this repo — `nav_epochs` (10) x |steps| per stage — because the adapters change *which* parameters move, not how often. The shared core additionally moves at one tenth of the adapter learning rate at every one of those steps (§2.3).

## Overall C0 verdict

| config | c1 (plasticity) | c2 (stability) | c3 (utility) | c4 (disclosure) | C0 |
|---|---|---|---|---|---|
| `sp_nav` | FAIL | FAIL | n/a | PASS | **C0 FAIL** |
| `sp_nav_eq` | FAIL | PASS | PASS | PASS | **C0 FAIL** |

## Branch and STOP

**C0 does not pass.** Per §9 the branch is to return to Track A. No C0.1, no router, no confirmation runs.

**STOP for three-way review** (§8). The architectural hypothesis was screened at the smallest configuration that can answer it, under the oracle-task-identity assumption disclosed above.

Wording locks inherited from the ratified v0.34 verdict apply to this document: cell-level plasticity differences at three seeds are reported as falling within seed variability and are never described as a repair or a newly introduced failure; the v0.34 gradient-conflict measurement is a **null result**, not an unconfirmed one.

This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable review**. No threshold was altered and no comparator was rerun in producing it.

