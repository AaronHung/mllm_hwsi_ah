# Method Gate v0.33.3 — g1-g4 Verdict

Computed by `scripts/method_gate_verdict.py` against the completed 33-unit Mac MPS gate run (`runs/v2/method_gate_v0333_run1/cl_main_can_main_method_gate_v0333_run1.csv`), the frozen Protocol-v1 comparator CSVs, and the checksum-verified MPS `util_regen` backfill. No thresholds or formulas were chosen after seeing this data — all definitions are copied from `docs/method_gate_v033.md` §5-§6 (v0.33.3, frozen before this script ran).

**Known data-availability gap (disclosed, not worked around):** the frozen Protocol-v1 `ours_uniform` rows (`results/cl_main_can_main_ablA1_s*.csv`) only cover K∈{1,2} across all 5 archived seeds — `ours_uniform` was never run at K=4 in Protocol-v1. Per `docs/compute_policy.md` / gate rule r1, no new comparator training is launched at verdict time (that would be a post-hoc protocol change). Consequently, every g1/g4 sub-check **against `ours_uniform` specifically** is evaluated at K∈{1,2} only; K=4 for that comparator is reported as **N/A** below, not silently passed or failed. The `distill`/`replay` comparators (from `..._full_s*.csv`) have full K∈{1,2,4} coverage, so their sub-checks are unaffected.

## g1 — Forgetting: parity or better vs `ours_uniform` AND `distill`

Rule: paired mean diff (config − comparator) on `Forgetting` ≤ 0 at every available K; additionally 3/3 seeds non-worse (diff ≤ 0) at K∈{1,2}.

| config | K | vs ours_uniform mean Δ [per-seed 0,1,2] | 3/3 ≤0? | vs distill mean Δ [per-seed 0,1,2] | 3/3 ≤0? | g1 (this K) |
|---|---|---|---|---|---|---|
| ia_samp | 1 | +0.0162 [-0.0117, +0.0374, +0.0229] | no | +0.0108 [-0.0043, +0.0206, +0.0161] | no | FAIL |
| ia_samp | 2 | +0.0044 [+0.0064, +0.0074, -0.0006] | no | +0.0314 [+0.0474, +0.0263, +0.0206] | no | FAIL |
| ia_samp | 4 | N/A | N/A | -0.0093 [+0.0108, +0.0066, -0.0453] | no | PASS |
| eq_pres | 1 | +0.0048 [-0.0040, -0.0007, +0.0191] | no | -0.0006 [+0.0034, -0.0175, +0.0123] | no | FAIL |
| eq_pres | 2 | -0.0298 [-0.0410, -0.0208, -0.0276] | yes | -0.0028 [+0.0000, -0.0020, -0.0064] | yes | PASS |
| eq_pres | 4 | N/A | N/A | -0.0038 [-0.0064, +0.0330, -0.0381] | no | PASS |
| ia_ep | 1 | +0.0169 [+0.0134, +0.0174, +0.0198] | no | +0.0115 [+0.0208, +0.0006, +0.0130] | no | FAIL |
| ia_ep | 2 | -0.0005 [+0.0342, -0.0265, -0.0093] | no | +0.0265 [+0.0753, -0.0077, +0.0119] | no | FAIL |
| ia_ep | 4 | N/A | N/A | -0.0059 [+0.0028, +0.0327, -0.0532] | no | PASS |

Per-seed Δ shown as `[seed0, seed1, seed2]` so a single-outlier-seed failure is visible directly, not hidden behind the mean/3-of-3 summary. See the seed-sensitivity note at the end of this report.

## g2 — At least one axis improves vs BOTH comparators (3/3 seeds), at K=1 or K=2

Axes: capability = Forgetting(lower) or AA(higher); behavior = Jaccard(higher) or action-KL(lower); utility = eps_optimal_mass(higher) or normalized_regret(lower), final stage / old tasks only (§5.1). "Improves" = strict sign-consistent 3/3-seed diff vs ours_uniform AND vs distill simultaneously.

| config | K | axis | Δ vs ours_uniform (3 seeds) | Δ vs distill (3 seeds) | 3/3 improve both? |
|---|---|---|---|---|---|
| ia_samp | 1 | capability: Forgetting | -0.0117, +0.0374, +0.0229 | -0.0043, +0.0206, +0.0161 | no |
| ia_samp | 1 | capability: AA | +0.0112, +0.0031, -0.0053 | +0.0102, +0.0279, -0.0075 | no |
| ia_samp | 1 | behavior: Jaccard | +0.0404, -0.0050, -0.0032 | -0.0146, -0.0748, -0.0716 | no |
| ia_samp | 1 | behavior: action-KL | +0.0055, -0.0011, +0.0016 | +0.0137, -0.0000, +0.0158 | no |
| ia_samp | 1 | utility: eps_optimal_mass | -0.0156, +0.0042, -0.0026 | +0.0000, +0.0135, +0.0018 | no |
| ia_samp | 1 | utility: normalized_regret | +0.0060, +0.0024, +0.0041 | +0.0045, +0.0011, +0.0072 | no |
| ia_samp | 2 | capability: Forgetting | +0.0064, +0.0074, -0.0006 | +0.0474, +0.0263, +0.0206 | no |
| ia_samp | 2 | capability: AA | -0.0017, -0.0246, +0.0280 | -0.0212, -0.0526, -0.0070 | no |
| ia_samp | 2 | behavior: Jaccard | -0.0117, -0.0384, -0.0333 | -0.0436, -0.0330, -0.0941 | no |
| ia_samp | 2 | behavior: action-KL | +0.0135, +0.0039, +0.0049 | +0.0230, +0.0200, +0.0110 | no |
| ia_samp | 2 | utility: eps_optimal_mass | +0.0019, -0.0032, -0.0064 | +0.0063, +0.0082, -0.0093 | no |
| ia_samp | 2 | utility: normalized_regret | +0.0006, +0.0037, +0.0023 | +0.0010, +0.0002, +0.0034 | no |
| eq_pres | 1 | capability: Forgetting | -0.0040, -0.0007, +0.0191 | +0.0034, -0.0175, +0.0123 | no |
| eq_pres | 1 | capability: AA | +0.0032, -0.0104, -0.0304 | +0.0022, +0.0144, -0.0325 | no |
| eq_pres | 1 | behavior: Jaccard | -0.0535, -0.0483, -0.0696 | -0.1085, -0.1181, -0.1380 | no |
| eq_pres | 1 | behavior: action-KL | +0.0692, +0.0654, +0.0863 | +0.0774, +0.0665, +0.1005 | no |
| eq_pres | 1 | utility: eps_optimal_mass | +0.0195, +0.0397, +0.0332 | +0.0351, +0.0489, +0.0376 | YES |
| eq_pres | 1 | utility: normalized_regret | -0.0048, -0.0095, -0.0065 | -0.0063, -0.0108, -0.0034 | YES |
| eq_pres | 2 | capability: Forgetting | -0.0410, -0.0208, -0.0276 | +0.0000, -0.0020, -0.0064 | no |
| eq_pres | 2 | capability: AA | +0.0379, +0.0156, +0.0362 | +0.0184, -0.0124, +0.0012 | no |
| eq_pres | 2 | behavior: Jaccard | -0.0335, -0.0341, -0.0210 | -0.0654, -0.0287, -0.0817 | no |
| eq_pres | 2 | behavior: action-KL | +0.0568, +0.0587, +0.0761 | +0.0663, +0.0748, +0.0822 | no |
| eq_pres | 2 | utility: eps_optimal_mass | +0.0355, +0.0280, +0.0292 | +0.0399, +0.0394, +0.0263 | YES |
| eq_pres | 2 | utility: normalized_regret | -0.0133, -0.0070, -0.0088 | -0.0129, -0.0105, -0.0078 | YES |
| ia_ep | 1 | capability: Forgetting | +0.0134, +0.0174, +0.0198 | +0.0208, +0.0006, +0.0130 | no |
| ia_ep | 1 | capability: AA | -0.0004, +0.0159, +0.0050 | -0.0014, +0.0406, +0.0029 | no |
| ia_ep | 1 | behavior: Jaccard | -0.0643, -0.0260, -0.0863 | -0.1193, -0.0958, -0.1547 | no |
| ia_ep | 1 | behavior: action-KL | +0.1434, +0.1284, +0.1245 | +0.1516, +0.1295, +0.1387 | no |
| ia_ep | 1 | utility: eps_optimal_mass | +0.0144, +0.0310, +0.0335 | +0.0300, +0.0402, +0.0379 | YES |
| ia_ep | 1 | utility: normalized_regret | -0.0019, -0.0018, -0.0082 | -0.0034, -0.0030, -0.0051 | YES |
| ia_ep | 2 | capability: Forgetting | +0.0342, -0.0265, -0.0093 | +0.0753, -0.0077, +0.0119 | no |
| ia_ep | 2 | capability: AA | -0.0061, +0.0034, +0.0264 | -0.0256, -0.0246, -0.0087 | no |
| ia_ep | 2 | behavior: Jaccard | -0.0288, -0.0409, -0.0230 | -0.0608, -0.0356, -0.0837 | no |
| ia_ep | 2 | behavior: action-KL | +0.0959, +0.1084, +0.0757 | +0.1054, +0.1245, +0.0818 | no |
| ia_ep | 2 | utility: eps_optimal_mass | +0.0101, +0.0197, +0.0181 | +0.0145, +0.0311, +0.0152 | YES |
| ia_ep | 2 | utility: normalized_regret | +0.0001, -0.0018, -0.0046 | +0.0005, -0.0053, -0.0035 | no |

## g3 — Not worse than `replay` on Forgetting (paired mean within +0.005)

| config | K | vs replay (mean Δ forgetting) | g3 (this K) |
|---|---|---|---|
| ia_samp | 1 | +0.0106 | FAIL |
| ia_samp | 2 | +0.0237 | FAIL |
| ia_samp | 4 | -0.0137 | PASS |
| eq_pres | 1 | -0.0008 | PASS |
| eq_pres | 2 | -0.0105 | PASS |
| eq_pres | 4 | -0.0083 | PASS |
| ia_ep | 1 | +0.0113 | FAIL |
| ia_ep | 2 | +0.0188 | FAIL |
| ia_ep | 4 | -0.0104 | PASS |

## g4 — New-task plasticity: A[t,t] vs `ours_uniform` not degraded by >0.01 in any (task,K) cell

Rule: per (task,K) cell, 3-seed paired mean own-time diff vs ours_uniform; fail iff mean diff < -0.01. K=4 is N/A (no `ours_uniform` data at K=4, see disclosure above).

| config | task | K | mean ΔA[t,t] [per-seed 0,1,2] | g4 (this cell) |
|---|---|---|---|---|
| ia_samp | esca | 1 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| ia_samp | esca | 2 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| ia_samp | esca | 4 | N/A | N/A |
| ia_samp | lung | 1 | -0.0064 [+0.0102, -0.0197, -0.0096] | PASS |
| ia_samp | lung | 2 | +0.0168 [+0.0096, +0.0122, +0.0287] | PASS |
| ia_samp | lung | 4 | N/A | N/A |
| ia_samp | rcc | 1 | +0.0062 [-0.0102, +0.0185, +0.0102] | PASS |
| ia_samp | rcc | 2 | -0.0096 [-0.0204, +0.0000, -0.0083] | PASS |
| ia_samp | rcc | 4 | N/A | N/A |
| ia_samp | brca | 1 | +0.0962 [+0.0722, +0.1376, +0.0789] | PASS |
| ia_samp | brca | 2 | -0.0110 [+0.0128, -0.0985, +0.0526] | FAIL |
| ia_samp | brca | 4 | N/A | N/A |
| eq_pres | esca | 1 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| eq_pres | esca | 2 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| eq_pres | esca | 4 | N/A | N/A |
| eq_pres | lung | 1 | +0.0034 [+0.0211, -0.0319, +0.0211] | PASS |
| eq_pres | lung | 2 | +0.0032 [-0.0102, +0.0000, +0.0198] | PASS |
| eq_pres | lung | 4 | N/A | N/A |
| eq_pres | rcc | 1 | -0.0034 [-0.0204, +0.0000, +0.0102] | PASS |
| eq_pres | rcc | 2 | +0.0068 [+0.0102, +0.0000, +0.0102] | PASS |
| eq_pres | rcc | 4 | N/A | N/A |
| eq_pres | brca | 1 | -0.0176 [+0.0000, +0.0000, -0.0527] | FAIL |
| eq_pres | brca | 2 | -0.0175 [-0.0661, +0.0000, +0.0135] | FAIL |
| eq_pres | brca | 4 | N/A | N/A |
| ia_ep | esca | 1 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| ia_ep | esca | 2 | +0.0000 [+0.0000, +0.0000, +0.0000] | PASS |
| ia_ep | esca | 4 | N/A | N/A |
| ia_ep | lung | 1 | +0.0068 [+0.0300, -0.0306, +0.0211] | PASS |
| ia_ep | lung | 2 | +0.0207 [+0.0198, +0.0231, +0.0191] | PASS |
| ia_ep | lung | 4 | N/A | N/A |
| ia_ep | rcc | 1 | -0.0123 [-0.0185, +0.0000, -0.0185] | FAIL |
| ia_ep | rcc | 2 | +0.0006 [+0.0102, -0.0287, +0.0204] | PASS |
| ia_ep | rcc | 4 | N/A | N/A |
| ia_ep | brca | 1 | +0.1118 [+0.0985, +0.1579, +0.0789] | PASS |
| ia_ep | brca | 2 | -0.0110 [+0.0060, -0.0789, +0.0398] | FAIL |
| ia_ep | brca | 4 | N/A | N/A |

Per-seed Δ shown as `[seed0, seed1, seed2]`.

## Overall verdict per config

| config | g1 | g2 | g3 | g4 | OVERALL |
|---|---|---|---|---|---|
| ia_samp (M1 (importance sampling only)) | FAIL | FAIL | FAIL | FAIL | **GATE FAIL** |
| eq_pres (M2 (eps-equivalence loss only)) | FAIL | PASS | PASS | FAIL | **GATE FAIL** |
| ia_ep (M3 (M1+M2 combined)) | FAIL | PASS | FAIL | FAIL | **GATE FAIL** |

A config needs g1 AND g2 AND g3 AND g4 all PASS (across all evaluated K) to pass the gate.

## Mini-arm diagnostics (K=1 only, informational — not gate candidates)

| method | K | AA (mean, 3 seeds) | Forgetting (mean) | Jaccard (mean) | action-KL (mean) |
|---|---|---|---|---|---|
| samp_util_only | 1 | 0.8973 | 0.0166 | 0.1796 | 0.0759 |
| samp_drift_only | 1 | 0.8969 | 0.0168 | 0.2267 | 0.0678 |

## Seed-sensitivity diagnostic (not a pass criterion — read before deciding remediation)

Only 3 seeds and `n_test=15` per task give a thin base for a hard ±0.01 / sign-consistency rule. For every FAIL cell above, this checks whether the failure is driven by ALL 3 seeds pointing the same direction ("consistent") or by a single seed dragging an otherwise-passing mean past the threshold ("single-seed-driven", i.e. the other 2 seeds are individually within tolerance or even improving).

| check | config | K | task/comparator | per-seed Δ | verdict |
|---|---|---|---|---|---|
| g4 | ia_samp | 2 | brca | [+0.0128, -0.0985, +0.0526] | single/partial-seed-driven (1/3 below tol) |
| g4 | eq_pres | 1 | brca | [+0.0000, +0.0000, -0.0527] | single/partial-seed-driven (1/3 below tol) |
| g4 | eq_pres | 2 | brca | [-0.0661, +0.0000, +0.0135] | single/partial-seed-driven (1/3 below tol) |
| g4 | ia_ep | 1 | rcc | [-0.0185, +0.0000, -0.0185] | single/partial-seed-driven (2/3 below tol) |
| g4 | ia_ep | 2 | brca | [+0.0060, -0.0789, +0.0398] | single/partial-seed-driven (1/3 below tol) |
| g1 | ia_samp | 1 | vs ours_uniform | [-0.0117, +0.0374, +0.0229] | single/partial-seed-driven (2/3 worse) |
| g1 | ia_samp | 1 | vs distill | [-0.0043, +0.0206, +0.0161] | single/partial-seed-driven (2/3 worse) |
| g1 | ia_samp | 2 | vs ours_uniform | [+0.0064, +0.0074, -0.0006] | single/partial-seed-driven (2/3 worse) |
| g1 | ia_samp | 2 | vs distill | [+0.0474, +0.0263, +0.0206] | consistent (3/3 worse) |
| g1 | ia_samp | 4 | vs distill | [+0.0108, +0.0066, -0.0453] | single/partial-seed-driven (2/3 worse) |
| g1 | eq_pres | 1 | vs ours_uniform | [-0.0040, -0.0007, +0.0191] | single/partial-seed-driven (1/3 worse) |
| g1 | eq_pres | 1 | vs distill | [+0.0034, -0.0175, +0.0123] | single/partial-seed-driven (2/3 worse) |
| g1 | eq_pres | 4 | vs distill | [-0.0064, +0.0330, -0.0381] | single/partial-seed-driven (1/3 worse) |
| g1 | ia_ep | 1 | vs ours_uniform | [+0.0134, +0.0174, +0.0198] | consistent (3/3 worse) |
| g1 | ia_ep | 1 | vs distill | [+0.0208, +0.0006, +0.0130] | consistent (3/3 worse) |
| g1 | ia_ep | 2 | vs ours_uniform | [+0.0342, -0.0265, -0.0093] | single/partial-seed-driven (1/3 worse) |
| g1 | ia_ep | 2 | vs distill | [+0.0753, -0.0077, +0.0119] | single/partial-seed-driven (2/3 worse) |
| g1 | ia_ep | 4 | vs distill | [+0.0028, +0.0327, -0.0532] | single/partial-seed-driven (2/3 worse) |

**15/18** flagged FAIL sub-checks above are single/partial-seed-driven rather than a consistent 3/3 signal. This does not overturn any g1/g4 verdict (the pre-registered rule is the rule), but it is directly relevant to how much weight to put on "gate fail" as evidence the mechanism doesn't work, versus evidence that 3 seeds x n_test=15 is too thin to resolve small effects at this sample size. Candidate remediation directions (require red-team sign-off before any new compute; NOT started by Cursor unilaterally):
1. Re-seed a wider seed set (e.g. seeds 3-7) for exactly the FAIL cells to see if the single-seed outliers wash out — cheap, reuses the existing pipeline, no formula change.
2. Backfill `ours_uniform` at K=4 (probe-only regen, MPS, same as §5.2) so g1/g4 at K=4 are no longer N/A — closes the disclosed data gap rather than working around it.
3. Investigate whether the `brca` (task 4) plasticity dip is a training-order artifact (last task, least remaining budget) rather than an M1/M2-specific effect, e.g. by checking whether `ia_samp`'s own large K=1 brca *improvement* (+0.10, all 3 seeds positive) and its K=2 dip (driven by seed 1 alone, other two seeds positive) can both be true of a genuinely noisy cell.
4. If neither of the above changes the verdict, report `eq_pres` (M2) as the primary honest finding in the paper appendix: clear utility-axis win (g2 PASS, sign-consistent 3/3 at both K=1 and K=2 vs both comparators) traded against a borderline, seed-sensitive regression on forgetting-at-K=1 and new-task plasticity on the last task — a real, reportable trade-off, not a broken method.
