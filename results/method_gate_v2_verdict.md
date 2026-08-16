# Method Gate v2 — `eq_pres`-only Extension Verdict

**Track A:** 8/21 freeze conditions all met 2026-08-16 (pilot40 R6 closed). **Track B:** v0.33.3 verdict (all 3 configs FAIL) is **frozen and unchanged** — this document evaluates ONLY the pre-registered Gate v2 extension (`docs/method_gate_v033.md` §10) for `eq_pres`. `ia_samp`/`ia_ep` are RETIRED, not evaluated here.

Computed by `scripts/method_gate_v2_verdict.py` against `runs/v2/gate_v2_20260816T073200Z/cl_main_can_main_gate_v2_20260816T073200Z.csv` (19 new units), the frozen v0.33.3 `eq_pres` seeds{0,1,2} rows (NOT rerun), the frozen Protocol-v1 comparator CSVs (`full_s{0..4}`, `ablA1_s{0..4}`), and the checksum-verified utility-axis backfill (old: v0.33.2 §5.2 seeds{0,1,2}; new: this run's seeds{3,4}, checksummed above). No thresholds or formulas were chosen after seeing this data — all definitions are copied from §10.1 (frozen before this script ran).

Every criterion is reported **old3** (seeds 0-2, unchanged from v0.33.3) / **new2** (seeds 3-4 alone) / **pooled5** (all 5) — pooled5 is primary for the pass/fail call, old3/new2 are secondary evidence, always shown (§10.4). 95% bootstrap CIs (`n_boot=10000`) are reported as **uncertainty riders only**, never as a pass/fail test.

## g1 — Capability: pooled 5-seed paired mean Forgetting ≤ 0 vs `ours_uniform` AND `distill`

Rule (§10.1): mean-only, no 3/3 seed-consistency clause (replaced by the §10.3 replication rider, checked separately below).

| K | comparator | old3 mean [seeds0-2] | new2 mean [seeds3-4] | pooled5 mean [all] | pooled5 95% CI | g1 (pooled) |
|---|---|---|---|---|---|---|
| 1 | ours_uniform | +0.0048 [-0.0040, -0.0007, +0.0191] | +0.0068 [-0.0062, +0.0198] | +0.0056 | [-0.0040, +0.0154] | FAIL |
| 1 | distill | -0.0006 [+0.0034, -0.0175, +0.0123] | +0.0073 [-0.0045, +0.0191] | +0.0026 | [-0.0089, +0.0133] | FAIL |
| 2 | ours_uniform | -0.0298 [-0.0410, -0.0208, -0.0276] | -0.0355 [-0.0564, -0.0146] | -0.0321 | [-0.0450, -0.0198] | PASS |
| 2 | distill | -0.0028 [+0.0000, -0.0020, -0.0064] | -0.0182 [+0.0057, -0.0421] | -0.0089 | [-0.0256, +0.0019] | PASS |
| 4 | ours_uniform | -0.0015 [-0.0028, +0.0121, -0.0138] | -0.0290 [-0.0247, -0.0334] | -0.0125 | [-0.0260, +0.0018] | PASS |
| 4 | distill | -0.0038 [-0.0064, +0.0330, -0.0381] | -0.0167 [+0.0034, -0.0368] | -0.0090 | [-0.0312, +0.0133] | PASS |

## g2 — Utility axis: ≥4/5 seeds improvement direction vs BOTH comparators, K=1 or K=2

Rule (§10.1, new extension criterion, does not retroactively modify the original v0.33.3 3/3 criterion): `eps_optimal_mass` higher AND `normalized_regret` lower, simultaneously vs `ours_uniform` AND `distill`, for the same seed, at K=1 or K=2. A K passes if ≥4 of the 5 seeds show this simultaneous improvement.

| K | axis | per-seed Δ vs ours_uniform [0,1,2,3,4] | per-seed Δ vs distill [0,1,2,3,4] | n/5 seeds improving both | g2 (this K,axis) |
|---|---|---|---|---|---|
| 1 | eps_optimal_mass | [+0.0195, +0.0397, +0.0332, +0.0437, +0.0269] | [+0.0351, +0.0489, +0.0376, +0.0762, +0.0488] | 5/5 | PASS |
| 1 | normalized_regret | [-0.0048, -0.0095, -0.0065, -0.0133, -0.0048] | [-0.0063, -0.0108, -0.0034, -0.0246, -0.0107] | 5/5 | PASS |
| 2 | eps_optimal_mass | [+0.0355, +0.0280, +0.0292, +0.0260, +0.0298] | [+0.0399, +0.0394, +0.0263, +0.0378, +0.0367] | 5/5 | PASS |
| 2 | normalized_regret | [-0.0133, -0.0070, -0.0088, -0.0077, -0.0096] | [-0.0129, -0.0105, -0.0078, -0.0122, -0.0103] | 5/5 | PASS |

## g3 — Replay safety: pooled 5-seed mean Forgetting vs `replay` ≤ +0.005 (unchanged)

| K | old3 mean | new2 mean | pooled5 mean | pooled5 95% CI | g3 |
|---|---|---|---|---|---|
| 1 | -0.0008 | +0.0042 | +0.0012 | [-0.0053, +0.0085] | PASS |
| 2 | -0.0105 | -0.0170 | -0.0131 | [-0.0264, +0.0009] | PASS |
| 4 | -0.0083 | -0.0209 | -0.0133 | [-0.0223, -0.0035] | PASS |

## g4 — Plasticity: per (task,K) pooled 5-seed paired mean ΔA[t,t] vs `ours_uniform` ≥ -0.01 (unchanged)

Each cell footnoted with its per-task test-set accuracy quantum (1/n_test) — descriptive only, not used to argue the threshold is wrong (Sol's correction, §9 changelog).

| task | K | quantum (1/n_test) | old3 mean [0,1,2] | new2 mean [3,4] | pooled5 mean [all] | pooled5 95% CI | g4 |
|---|---|---|---|---|---|---|---|
| esca | 1 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | PASS |
| esca | 2 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | PASS |
| esca | 4 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | PASS |
| lung | 1 | 0.0105 (n=95) | +0.0034 [+0.0211, -0.0319, +0.0211] | -0.0047 [+0.0109, -0.0204] | +0.0002 | [-0.0190, +0.0191] | PASS |
| lung | 2 | 0.0105 (n=95) | +0.0032 [-0.0102, +0.0000, +0.0198] | -0.0058 [-0.0320, +0.0204] | -0.0004 | [-0.0172, +0.0161] | PASS |
| lung | 4 | 0.0105 (n=95) | -0.0079 [+0.0204, -0.0333, -0.0109] | -0.0204 [-0.0115, -0.0293] | -0.0129 | [-0.0273, +0.0042] | FAIL |
| rcc | 1 | 0.0132 (n=76) | -0.0034 [-0.0204, +0.0000, +0.0102] | -0.0194 [-0.0574, +0.0185] | -0.0098 | [-0.0348, +0.0115] | PASS |
| rcc | 2 | 0.0132 (n=76) | +0.0068 [+0.0102, +0.0000, +0.0102] | -0.0051 [-0.0102, +0.0000] | +0.0020 | [-0.0041, +0.0082] | PASS |
| rcc | 4 | 0.0132 (n=76) | +0.0000 [+0.0185, -0.0083, -0.0102] | -0.0042 [+0.0102, -0.0186] | -0.0017 | [-0.0132, +0.0111] | PASS |
| brca | 1 | 0.0108 (n=93) | -0.0176 [+0.0000, +0.0000, -0.0527] | +0.0034 [+0.0068, +0.0000] | -0.0092 | [-0.0316, +0.0041] | PASS |
| brca | 2 | 0.0108 (n=93) | -0.0175 [-0.0661, +0.0000, +0.0135] | -0.0590 [-0.0195, -0.0986] | -0.0341 | [-0.0724, +0.0015] | FAIL |
| brca | 4 | 0.0108 (n=93) | +0.0218 [+0.1113, -0.0128, -0.0331] | -0.0297 [-0.0790, +0.0195] | +0.0012 | [-0.0501, +0.0617] | PASS |

## §10.3 Replication rider check

**Not triggered on any g1/g4 cell** — for every PASS cell, at least one of the two new seeds (3, 4) is not individually on the failing side.

## Overall Gate v2 verdict for `eq_pres`

| g1 (capability) | g2 (utility) | g3 (replay safety) | g4 (plasticity) | §10.3 rider | GATE v2 |
|---|---|---|---|---|---|
| FAIL | PASS | PASS | FAIL | clear | **GATE v2 FAIL** |

**Branch (§10.6): FAIL.** `eq_pres` is reported as an intervention/mechanism result in the main analysis paper, not a promoted method. This is the last main-order method-rescue compute (§10.5) — no further rescue regardless of this outcome. **STOP for joint Sol+Fable+Aaron unblinding review.**
