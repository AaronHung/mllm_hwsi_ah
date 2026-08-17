# E1 — Utility-Capability Alignment Test (report)

**Track A:** frozen; writing is the only remaining work. **Track B/C:** Track B closed and archived, Track C0 screened and FAILed; this is one of the two v0.35-closeout artifacts, after which all compute for this paper ends permanently.

Pre-registered in `docs/alignment_test.md`, committed before this script ran. **No navigator is trained anywhere in E1** and no comparator was rerun. The oracle is **LABEL-INFORMED** (§1 of the pre-registration): it is a diagnostic, never a deployable policy.

**Determinism check (§3): PASS** — two identical oracle passes over 36 cells produced identical tables.

## Internal consistency (§8)

- oracle mean `eps_optimal_mass` = **1.0000** (a policy that is the argmax of the gain should be inside the epsilon-optimal set essentially always)
- oracle mean `normalized_regret` = **0.0000** (should be near zero by construction)

## Evaluator equivalence check (§4.1)

E1 retrains each per-task evaluator; the check is whether the random policy scores the same on it as on the frozen run's. Frozen values come from the `random_ref` column.

| K | task | E1 random (mean over seeds) | frozen `random_ref` | difference |
|---|---|---|---|---|
| 1 | esca | 0.8339 | 0.8339 | -0.0000 |
| 1 | lung | 0.7595 | 0.7685 | -0.0090 |
| 1 | rcc | 0.8712 | 0.8756 | -0.0044 |
| 1 | brca | 0.6475 | 0.6787 | -0.0311 |
| 2 | esca | 0.8524 | 0.8524 | +0.0000 |
| 2 | lung | 0.7943 | 0.7943 | +0.0000 |
| 2 | rcc | 0.9058 | 0.9025 | +0.0033 |
| 2 | brca | 0.7087 | 0.7123 | -0.0035 |
| 4 | esca | 0.8583 | 0.8583 | +0.0000 |
| 4 | lung | 0.8372 | 0.8329 | +0.0043 |
| 4 | rcc | 0.9267 | 0.9215 | +0.0052 |
| 4 | brca | 0.7656 | 0.7782 | -0.0126 |

Largest absolute difference: **0.0311**. Read this as the size of the cross-evaluator term that the primary comparison below carries; the random-anchored columns are free of it.

## Oracle vs `seqft` own-time accuracy

| K | task | quantum `1/n_test` | oracle bal-acc | `seqft` own-time | oracle − seqft | oracle − random (E1) | seqft − random_ref (frozen) |
|---|---|---|---|---|---|---|---|
| 1 | esca | 0.0667 (n=15) | 0.9375 | 0.8720 | +0.0655 | +0.1036 | +0.0381 |
| 1 | lung | 0.0105 (n=95) | 0.9966 | 0.8760 | +0.1206 | +0.2371 | +0.1075 |
| 1 | rcc | 0.0132 (n=76) | 1.0000 | 0.9666 | +0.0334 | +0.1288 | +0.0910 |
| 1 | brca | 0.0108 (n=93) | 0.9825 | 0.8436 | +0.1388 | +0.3349 | +0.1650 |
| 2 | esca | 0.0667 (n=15) | 0.9792 | 0.9583 | +0.0208 | +0.1268 | +0.1060 |
| 2 | lung | 0.0105 (n=95) | 1.0000 | 0.8758 | +0.1242 | +0.2057 | +0.0815 |
| 2 | rcc | 0.0132 (n=76) | 1.0000 | 0.9605 | +0.0395 | +0.0942 | +0.0580 |
| 2 | brca | 0.0108 (n=93) | 0.9912 | 0.8852 | +0.1060 | +0.2825 | +0.1730 |
| 4 * | esca | 0.0667 (n=15) | 1.0000 | 0.9167 | +0.0833 | +0.1417 | +0.0583 |
| 4 * | lung | 0.0105 (n=95) | 1.0000 | 0.8690 | +0.1310 | +0.1628 | +0.0361 |
| 4 * | rcc | 0.0132 (n=76) | 1.0000 | 0.9481 | +0.0519 | +0.0733 | +0.0266 |
| 4 * | brca | 0.0108 (n=93) | 0.9912 | 0.8499 | +0.1413 | +0.2256 | +0.0717 |

`*` K=4 is descriptive context only and is **not** part of the read-out rule (§6).

## Pre-registered read-out (§7)

| K | oracle cross-task mean | `seqft` cross-task mean | difference | mean quantum | tasks where oracle is better |
|---|---|---|---|---|---|
| 1 | 0.9791 | 0.8896 | +0.0896 | 0.0253 | 4/4 |
| 2 | 0.9926 | 0.9200 | +0.0726 | 0.0253 | 4/4 |

### Tier: **ALIGNMENT SUPPORTED**

The one-step target does convert into capability when it is maximized with label information. The non-conversion seen in Tracks B and C is then a property of the methods or of the optimization, not of the target. Descriptive; no further runs are authorized either way.

## Zero-compute correlation half (§9)

Across every configuration this project ran, does the utility axis move together with capability? Per-seed deltas against each comparator; Spearman is **descriptive**, with no p/q language, per the repo statistical policy.

| config | `L_eq` | K | comparator | Δε-mass | ΔAA | ΔForgetting | signs |
|---|---|---|---|---|---|---|---|
| `conflict_eq_pres` | yes | 1 | `ours_uniform` (seed 0) | +0.0080 | +0.0175 | -0.0074 | ++ |
| `conflict_eq_pres` | yes | 1 | `ours_uniform` (seed 1) | +0.0143 | -0.0028 | +0.0029 | +− |
| `conflict_eq_pres` | yes | 1 | `ours_uniform` (seed 2) | +0.0111 | -0.0118 | +0.0195 | +− |
| `conflict_eq_pres` | yes | 1 | `distill` (seed 0) | +0.0236 | +0.0165 | +0.0001 | ++ |
| `conflict_eq_pres` | yes | 1 | `distill` (seed 1) | +0.0235 | +0.0220 | -0.0139 | ++ |
| `conflict_eq_pres` | yes | 1 | `distill` (seed 2) | +0.0155 | -0.0139 | +0.0127 | +− |
| `conflict_eq_pres` | yes | 2 | `ours_uniform` (seed 0) | +0.0138 | +0.0046 | -0.0410 | ++ |
| `conflict_eq_pres` | yes | 2 | `ours_uniform` (seed 1) | +0.0155 | +0.0091 | -0.0144 | ++ |
| `conflict_eq_pres` | yes | 2 | `ours_uniform` (seed 2) | +0.0106 | +0.0370 | -0.0174 | ++ |
| `conflict_eq_pres` | yes | 2 | `distill` (seed 0) | +0.0182 | -0.0148 | +0.0000 | +− |
| `conflict_eq_pres` | yes | 2 | `distill` (seed 1) | +0.0268 | -0.0190 | +0.0044 | +− |
| `conflict_eq_pres` | yes | 2 | `distill` (seed 2) | +0.0077 | +0.0020 | +0.0038 | ++ |
| `eq_pres` | yes | 1 | `ours_uniform` (seed 0) | +0.0195 | +0.0032 | -0.0040 | ++ |
| `eq_pres` | yes | 1 | `ours_uniform` (seed 1) | +0.0397 | -0.0104 | -0.0007 | +− |
| `eq_pres` | yes | 1 | `ours_uniform` (seed 2) | +0.0332 | -0.0304 | +0.0191 | +− |
| `eq_pres` | yes | 1 | `ours_uniform` (seed 3) | +0.0437 | +0.0014 | -0.0062 | ++ |
| `eq_pres` | yes | 1 | `ours_uniform` (seed 4) | +0.0269 | -0.0174 | +0.0198 | +− |
| `eq_pres` | yes | 1 | `distill` (seed 0) | +0.0351 | +0.0022 | +0.0034 | ++ |
| `eq_pres` | yes | 1 | `distill` (seed 1) | +0.0489 | +0.0144 | -0.0175 | ++ |
| `eq_pres` | yes | 1 | `distill` (seed 2) | +0.0376 | -0.0325 | +0.0123 | +− |
| `eq_pres` | yes | 2 | `ours_uniform` (seed 0) | +0.0355 | +0.0379 | -0.0410 | ++ |
| `eq_pres` | yes | 2 | `ours_uniform` (seed 1) | +0.0280 | +0.0156 | -0.0208 | ++ |
| `eq_pres` | yes | 2 | `ours_uniform` (seed 2) | +0.0292 | +0.0362 | -0.0276 | ++ |
| `eq_pres` | yes | 2 | `ours_uniform` (seed 3) | +0.0260 | +0.0112 | -0.0564 | ++ |
| `eq_pres` | yes | 2 | `ours_uniform` (seed 4) | +0.0298 | +0.0074 | -0.0146 | ++ |
| `eq_pres` | yes | 2 | `distill` (seed 0) | +0.0399 | +0.0184 | +0.0000 | ++ |
| `eq_pres` | yes | 2 | `distill` (seed 1) | +0.0394 | -0.0124 | -0.0020 | +− |
| `eq_pres` | yes | 2 | `distill` (seed 2) | +0.0263 | +0.0012 | -0.0064 | ++ |
| `ia_ep` | yes | 1 | `ours_uniform` (seed 0) | +0.0144 | -0.0004 | +0.0134 | +− |
| `ia_ep` | yes | 1 | `ours_uniform` (seed 1) | +0.0310 | +0.0159 | +0.0174 | ++ |
| `ia_ep` | yes | 1 | `ours_uniform` (seed 2) | +0.0335 | +0.0050 | +0.0198 | ++ |
| `ia_ep` | yes | 1 | `distill` (seed 0) | +0.0300 | -0.0014 | +0.0208 | +− |
| `ia_ep` | yes | 1 | `distill` (seed 1) | +0.0402 | +0.0406 | +0.0006 | ++ |
| `ia_ep` | yes | 1 | `distill` (seed 2) | +0.0379 | +0.0029 | +0.0130 | ++ |
| `ia_ep` | yes | 2 | `ours_uniform` (seed 0) | +0.0101 | -0.0061 | +0.0342 | +− |
| `ia_ep` | yes | 2 | `ours_uniform` (seed 1) | +0.0197 | +0.0034 | -0.0265 | ++ |
| `ia_ep` | yes | 2 | `ours_uniform` (seed 2) | +0.0181 | +0.0264 | -0.0093 | ++ |
| `ia_ep` | yes | 2 | `distill` (seed 0) | +0.0145 | -0.0256 | +0.0753 | +− |
| `ia_ep` | yes | 2 | `distill` (seed 1) | +0.0311 | -0.0246 | -0.0077 | +− |
| `ia_ep` | yes | 2 | `distill` (seed 2) | +0.0152 | -0.0087 | +0.0119 | +− |
| `ia_samp` | no | 1 | `ours_uniform` (seed 0) | -0.0156 | +0.0112 | -0.0117 | −+ |
| `ia_samp` | no | 1 | `ours_uniform` (seed 1) | +0.0042 | +0.0031 | +0.0374 | ++ |
| `ia_samp` | no | 1 | `ours_uniform` (seed 2) | -0.0026 | -0.0053 | +0.0229 | −− |
| `ia_samp` | no | 1 | `distill` (seed 0) | +0.0000 | +0.0102 | -0.0043 | −+ |
| `ia_samp` | no | 1 | `distill` (seed 1) | +0.0135 | +0.0279 | +0.0206 | ++ |
| `ia_samp` | no | 1 | `distill` (seed 2) | +0.0018 | -0.0075 | +0.0161 | +− |
| `ia_samp` | no | 2 | `ours_uniform` (seed 0) | +0.0019 | -0.0017 | +0.0064 | +− |
| `ia_samp` | no | 2 | `ours_uniform` (seed 1) | -0.0032 | -0.0246 | +0.0074 | −− |
| `ia_samp` | no | 2 | `ours_uniform` (seed 2) | -0.0064 | +0.0280 | -0.0006 | −+ |
| `ia_samp` | no | 2 | `distill` (seed 0) | +0.0063 | -0.0212 | +0.0474 | +− |
| `ia_samp` | no | 2 | `distill` (seed 1) | +0.0082 | -0.0526 | +0.0263 | +− |
| `ia_samp` | no | 2 | `distill` (seed 2) | -0.0093 | -0.0070 | +0.0206 | −− |
| `proj_distill` | no | 1 | `ours_uniform` (seed 0) | -0.0025 | +0.0106 | -0.0244 | −+ |
| `proj_distill` | no | 1 | `ours_uniform` (seed 1) | +0.0009 | +0.0024 | +0.0098 | ++ |
| `proj_distill` | no | 1 | `ours_uniform` (seed 2) | +0.0050 | -0.0204 | +0.0089 | +− |
| `proj_distill` | no | 1 | `distill` (seed 0) | +0.0132 | +0.0096 | -0.0170 | ++ |
| `proj_distill` | no | 1 | `distill` (seed 1) | +0.0101 | +0.0272 | -0.0070 | ++ |
| `proj_distill` | no | 1 | `distill` (seed 2) | +0.0094 | -0.0225 | +0.0021 | +− |
| `proj_distill` | no | 2 | `ours_uniform` (seed 0) | -0.0029 | +0.0127 | -0.0376 | −+ |
| `proj_distill` | no | 2 | `ours_uniform` (seed 1) | -0.0041 | +0.0151 | -0.0166 | −+ |
| `proj_distill` | no | 2 | `ours_uniform` (seed 2) | -0.0014 | +0.0120 | -0.0072 | −+ |
| `proj_distill` | no | 2 | `distill` (seed 0) | +0.0015 | -0.0068 | +0.0034 | +− |
| `proj_distill` | no | 2 | `distill` (seed 1) | +0.0072 | -0.0129 | +0.0023 | +− |
| `proj_distill` | no | 2 | `distill` (seed 2) | -0.0043 | -0.0230 | +0.0140 | −− |
| `proj_eq_pres` | yes | 1 | `ours_uniform` (seed 0) | +0.0293 | +0.0152 | -0.0302 | ++ |
| `proj_eq_pres` | yes | 1 | `ours_uniform` (seed 1) | +0.0432 | -0.0032 | +0.0066 | +− |
| `proj_eq_pres` | yes | 1 | `ours_uniform` (seed 2) | +0.0508 | -0.0256 | +0.0000 | +− |
| `proj_eq_pres` | yes | 1 | `distill` (seed 0) | +0.0449 | +0.0142 | -0.0227 | ++ |
| `proj_eq_pres` | yes | 1 | `distill` (seed 1) | +0.0524 | +0.0216 | -0.0102 | ++ |
| `proj_eq_pres` | yes | 1 | `distill` (seed 2) | +0.0552 | -0.0278 | -0.0068 | +− |
| `proj_eq_pres` | yes | 2 | `ours_uniform` (seed 0) | +0.0374 | +0.0227 | -0.0202 | ++ |
| `proj_eq_pres` | yes | 2 | `ours_uniform` (seed 1) | +0.0280 | +0.0071 | -0.0068 | ++ |
| `proj_eq_pres` | yes | 2 | `ours_uniform` (seed 2) | +0.0224 | +0.0294 | -0.0064 | ++ |
| `proj_eq_pres` | yes | 2 | `distill` (seed 0) | +0.0418 | +0.0032 | +0.0208 | ++ |
| `proj_eq_pres` | yes | 2 | `distill` (seed 1) | +0.0393 | -0.0209 | +0.0121 | +− |
| `proj_eq_pres` | yes | 2 | `distill` (seed 2) | +0.0195 | -0.0056 | +0.0149 | +− |
| `sp_nav` | no | 1 | `ours_uniform` (seed 0) | -0.1313 | -0.0351 | +0.0492 | −− |
| `sp_nav` | no | 1 | `ours_uniform` (seed 1) | -0.1308 | -0.0685 | +0.0821 | −− |
| `sp_nav` | no | 1 | `ours_uniform` (seed 2) | -0.1162 | -0.0737 | +0.1032 | −− |
| `sp_nav` | no | 1 | `distill` (seed 0) | -0.1157 | -0.0361 | +0.0566 | −− |
| `sp_nav` | no | 1 | `distill` (seed 1) | -0.1215 | -0.0437 | +0.0653 | −− |
| `sp_nav` | no | 1 | `distill` (seed 2) | -0.1118 | -0.0758 | +0.0964 | −− |
| `sp_nav` | no | 2 | `ours_uniform` (seed 0) | -0.0548 | -0.0134 | +0.0056 | −− |
| `sp_nav` | no | 2 | `ours_uniform` (seed 1) | -0.0473 | -0.0171 | +0.0143 | −− |
| `sp_nav` | no | 2 | `ours_uniform` (seed 2) | -0.0349 | -0.0298 | +0.0370 | −− |
| `sp_nav` | no | 2 | `distill` (seed 0) | -0.0504 | -0.0329 | +0.0467 | −− |
| `sp_nav` | no | 2 | `distill` (seed 1) | -0.0360 | -0.0451 | +0.0332 | −− |
| `sp_nav` | no | 2 | `distill` (seed 2) | -0.0378 | -0.0648 | +0.0583 | −− |
| `sp_nav_eq` | yes | 1 | `ours_uniform` (seed 0) | +0.1247 | +0.0278 | -0.0274 | ++ |
| `sp_nav_eq` | yes | 1 | `ours_uniform` (seed 1) | +0.1358 | -0.0019 | -0.0102 | +− |
| `sp_nav_eq` | yes | 1 | `ours_uniform` (seed 2) | +0.1389 | -0.0335 | +0.0221 | +− |
| `sp_nav_eq` | yes | 1 | `distill` (seed 0) | +0.1403 | +0.0268 | -0.0200 | ++ |
| `sp_nav_eq` | yes | 1 | `distill` (seed 1) | +0.1451 | +0.0228 | -0.0270 | ++ |
| `sp_nav_eq` | yes | 1 | `distill` (seed 2) | +0.1433 | -0.0356 | +0.0153 | +− |
| `sp_nav_eq` | yes | 2 | `ours_uniform` (seed 0) | +0.1584 | +0.0164 | -0.0215 | ++ |
| `sp_nav_eq` | yes | 2 | `ours_uniform` (seed 1) | +0.1952 | +0.0403 | -0.0474 | ++ |
| `sp_nav_eq` | yes | 2 | `ours_uniform` (seed 2) | +0.1757 | +0.0070 | -0.0280 | ++ |
| `sp_nav_eq` | yes | 2 | `distill` (seed 0) | +0.1628 | -0.0030 | +0.0195 | +− |
| `sp_nav_eq` | yes | 2 | `distill` (seed 1) | +0.2065 | +0.0123 | -0.0285 | ++ |
| `sp_nav_eq` | yes | 2 | `distill` (seed 2) | +0.1728 | -0.0280 | -0.0068 | +− |

**Spearman(Δε-mass, ΔAA) = +0.339** over 100 config x K x comparator x seed observations — **descriptive**, no significance claim. Of the 77 observations where the utility axis improved, 43 (56%) also improved `AA`.

**Read:** if the utility axis moved consistently while `AA` did not track it, this is the same non-conversion the gates found, measured across every configuration this project ever ran rather than one at a time.

