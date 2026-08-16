# Method Gate v0.34 — Development Verdict

**Track A:** 8/21 freeze conditions all met 2026-08-16 (pilot40 R6 closed); writing is the remaining work. **Track B:** v0.33 is archived — the v0.33.3 and Gate v2 verdicts are frozen and `eq_pres` stays unpromoted; this document evaluates ONLY the v0.34 development gate (`docs/method_gate_v034.md` §3) for `proj_distill`, `proj_eq_pres`, `conflict_eq_pres`.

Computed by `scripts/method_gate_v034_verdict.py` against `runs/v2/gate_v034_dev_20260816T154548Z/cl_main_can_main_gate_v034_dev_20260816T154548Z.csv`. All comparators are frozen, same-backend (Mac MPS), and were not rerun (§7). Thresholds and formulas are transcribed from §3.2, which was committed before any v0.34 training run — nothing was chosen after seeing this data.

**This is a development environment.** The main task order is already unblinded, so nothing below is confirmatory evidence; confirmation is reverse order with seeds {0,1,2,3,4} and only after three-way review (§4).

Criteria are **mean-only** 3-seed paired means (§3.2). Per-seed values and 95% bootstrap CIs (`n_boot=10000`) are shown throughout as **descriptive disclosure**, never as a pass/fail test.

## A — Utility retained: Δ`eps_optimal_mass` > 0 vs `distill` AND `ours_uniform`, at K ∈ {1,2}

Aggregation is the frozen §5.1 convention (final stage, old tasks only, mean within task then macro-average). `normalized_regret` is shown as secondary descriptive evidence and is **not** a pass condition.

| config | K | Δε-mass vs `ours_uniform` [seeds 0,1,2] | Δε-mass vs `distill` [0,1,2] | Δregret vs `ours_uniform` (descriptive) | A (this K) |
|---|---|---|---|---|---|
| `proj_distill` | 1 | +0.0011 [-0.0025, +0.0009, +0.0050] | +0.0109 [+0.0132, +0.0101, +0.0094] | -0.0006 [+0.0006, +0.0002, -0.0025] | PASS |
| `proj_distill` | 2 | -0.0028 [-0.0029, -0.0041, -0.0014] | +0.0015 [+0.0015, +0.0072, -0.0043] | +0.0019 [+0.0013, +0.0029, +0.0014] | FAIL |
| `proj_eq_pres` | 1 | +0.0411 [+0.0293, +0.0432, +0.0508] | +0.0509 [+0.0449, +0.0524, +0.0552] | -0.0133 [-0.0100, -0.0152, -0.0146] | PASS |
| `proj_eq_pres` | 2 | +0.0293 [+0.0374, +0.0280, +0.0224] | +0.0335 [+0.0418, +0.0393, +0.0195] | -0.0099 [-0.0133, -0.0081, -0.0082] | PASS |
| `conflict_eq_pres` | 1 | +0.0111 [+0.0080, +0.0143, +0.0111] | +0.0209 [+0.0236, +0.0235, +0.0155] | -0.0035 [-0.0019, -0.0066, -0.0020] | PASS |
| `conflict_eq_pres` | 2 | +0.0133 [+0.0138, +0.0155, +0.0106] | +0.0176 [+0.0182, +0.0268, +0.0077] | -0.0037 [-0.0050, -0.0034, -0.0026] | PASS |

## B — Targeted repair + no new failure (rider r1 = Sol item 1)

### B-a — every `(task, K)` cell: 3-seed paired mean ΔA[t,t] vs `ours_uniform` ≥ −0.01

| config | task | K | quantum (1/n_test) | mean ΔA[t,t] [seeds 0,1,2] | 95% CI | B-a | diagnosed target |
|---|---|---|---|---|---|---|---|
| `proj_distill` | esca | 1 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_distill` | esca | 2 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_distill` | esca | 4 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_distill` | lung | 1 | 0.0105 (n=95) | -0.0032 [-0.0102, -0.0102, +0.0109] | [-0.0102, +0.0109] | PASS |  |
| `proj_distill` | lung | 2 | 0.0105 (n=95) | -0.0034 [-0.0306, +0.0109, +0.0096] | [-0.0306, +0.0109] | PASS |  |
| `proj_distill` | lung | 4 | 0.0105 (n=95) | -0.0100 [+0.0204, +0.0089, -0.0592] | [-0.0592, +0.0204] | PASS | **yes** |
| `proj_distill` | rcc | 1 | 0.0132 (n=76) | +0.0028 [-0.0102, +0.0083, +0.0102] | [-0.0102, +0.0102] | PASS |  |
| `proj_distill` | rcc | 2 | 0.0132 (n=76) | +0.0034 [+0.0000, +0.0000, +0.0102] | [+0.0000, +0.0102] | PASS |  |
| `proj_distill` | rcc | 4 | 0.0132 (n=76) | +0.0034 [+0.0000, +0.0019, +0.0083] | [+0.0000, +0.0083] | PASS |  |
| `proj_distill` | brca | 1 | 0.0108 (n=93) | +0.0198 [+0.0398, +0.0526, -0.0331] | [-0.0331, +0.0526] | PASS |  |
| `proj_distill` | brca | 2 | 0.0108 (n=93) | -0.0153 [-0.0526, +0.0000, +0.0067] | [-0.0526, +0.0067] | FAIL | **yes** |
| `proj_distill` | brca | 4 | 0.0108 (n=93) | +0.0175 [+0.0331, +0.0067, +0.0128] | [+0.0067, +0.0331] | PASS |  |
| `proj_eq_pres` | esca | 1 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_eq_pres` | esca | 2 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_eq_pres` | esca | 4 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `proj_eq_pres` | lung | 1 | 0.0105 (n=95) | -0.0034 [-0.0102, +0.0102, -0.0102] | [-0.0102, +0.0102] | PASS |  |
| `proj_eq_pres` | lung | 2 | 0.0105 (n=95) | +0.0068 [-0.0204, +0.0218, +0.0191] | [-0.0204, +0.0218] | PASS |  |
| `proj_eq_pres` | lung | 4 | 0.0105 (n=95) | +0.0177 [+0.0204, +0.0415, -0.0089] | [-0.0089, +0.0415] | PASS | **yes** |
| `proj_eq_pres` | rcc | 1 | 0.0132 (n=76) | -0.0096 [-0.0185, +0.0083, -0.0185] | [-0.0185, +0.0083] | PASS |  |
| `proj_eq_pres` | rcc | 2 | 0.0132 (n=76) | +0.0096 [+0.0204, -0.0204, +0.0287] | [-0.0204, +0.0287] | PASS |  |
| `proj_eq_pres` | rcc | 4 | 0.0132 (n=76) | +0.0034 [+0.0287, -0.0371, +0.0185] | [-0.0371, +0.0287] | PASS |  |
| `proj_eq_pres` | brca | 1 | 0.0108 (n=93) | +0.0022 [+0.0594, +0.0000, -0.0527] | [-0.0527, +0.0594] | PASS |  |
| `proj_eq_pres` | brca | 2 | 0.0108 (n=93) | +0.0088 [-0.0331, +0.0068, +0.0526] | [-0.0331, +0.0526] | PASS | **yes** |
| `proj_eq_pres` | brca | 4 | 0.0108 (n=93) | -0.0331 [+0.0000, -0.0729, -0.0263] | [-0.0729, +0.0000] | FAIL |  |
| `conflict_eq_pres` | esca | 1 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `conflict_eq_pres` | esca | 2 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `conflict_eq_pres` | esca | 4 | 0.0667 (n=15) | +0.0000 [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000] | PASS |  |
| `conflict_eq_pres` | lung | 1 | 0.0105 (n=95) | +0.0138 [+0.0211, +0.0000, +0.0204] | [+0.0000, +0.0211] | PASS |  |
| `conflict_eq_pres` | lung | 2 | 0.0105 (n=95) | +0.0241 [+0.0000, +0.0211, +0.0511] | [+0.0000, +0.0511] | PASS |  |
| `conflict_eq_pres` | lung | 4 | 0.0105 (n=95) | +0.0000 [+0.0095, +0.0109, -0.0204] | [-0.0204, +0.0109] | PASS | **yes** |
| `conflict_eq_pres` | rcc | 1 | 0.0132 (n=76) | +0.0000 [-0.0102, -0.0102, +0.0204] | [-0.0102, +0.0204] | PASS |  |
| `conflict_eq_pres` | rcc | 2 | 0.0132 (n=76) | +0.0062 [+0.0102, -0.0019, +0.0102] | [-0.0019, +0.0102] | PASS |  |
| `conflict_eq_pres` | rcc | 4 | 0.0132 (n=76) | -0.0096 [+0.0000, -0.0185, -0.0102] | [-0.0185, +0.0000] | PASS |  |
| `conflict_eq_pres` | brca | 1 | 0.0108 (n=93) | +0.0130 [+0.0263, +0.0195, -0.0068] | [-0.0068, +0.0263] | PASS |  |
| `conflict_eq_pres` | brca | 2 | 0.0108 (n=93) | -0.0351 [-0.1053, -0.0263, +0.0263] | [-0.1053, +0.0263] | FAIL | **yes** |
| `conflict_eq_pres` | brca | 4 | 0.0108 (n=93) | -0.0306 [+0.0331, -0.0654, -0.0594] | [-0.0654, +0.0331] | FAIL |  |

### B-b — no new forgetting regression at K=2 and K=4: ΔForgetting ≤ +0.005 vs `ours_uniform` AND `distill`

### B-c — diagnosed target: at K=1, ΔForgetting ≤ 0 vs `ours_uniform` AND `distill`

| config | K | rule | vs `ours_uniform` mean [0,1,2] | vs `distill` mean [0,1,2] | 95% CI (vs `ours_uniform`) | verdict |
|---|---|---|---|---|---|---|
| `proj_distill` | 1 | ≤ 0 (B-c target) | -0.0019 [-0.0244, +0.0098, +0.0089] | -0.0073 [-0.0170, -0.0070, +0.0021] | [-0.0244, +0.0098] | PASS |
| `proj_distill` | 2 | ≤ +0.005 (B-b) | -0.0205 [-0.0376, -0.0166, -0.0072] | +0.0066 [+0.0034, +0.0023, +0.0140] | [-0.0376, -0.0072] | FAIL |
| `proj_distill` | 4 | ≤ +0.005 (B-b) | -0.0001 [-0.0031, +0.0130, -0.0102] | -0.0024 [-0.0066, +0.0339, -0.0345] | [-0.0102, +0.0130] | PASS |
| `proj_eq_pres` | 1 | ≤ 0 (B-c target) | -0.0079 [-0.0302, +0.0066, +0.0000] | -0.0133 [-0.0227, -0.0102, -0.0068] | [-0.0302, +0.0066] | PASS |
| `proj_eq_pres` | 2 | ≤ +0.005 (B-b) | -0.0111 [-0.0202, -0.0068, -0.0064] | +0.0159 [+0.0208, +0.0121, +0.0149] | [-0.0202, -0.0064] | FAIL |
| `proj_eq_pres` | 4 | ≤ +0.005 (B-b) | +0.0080 [+0.0304, -0.0002, -0.0061] | +0.0057 [+0.0268, +0.0206, -0.0304] | [-0.0061, +0.0304] | FAIL |
| `conflict_eq_pres` | 1 | ≤ 0 (B-c target) | +0.0050 [-0.0074, +0.0029, +0.0195] | -0.0004 [+0.0001, -0.0139, +0.0127] | [-0.0074, +0.0195] | FAIL |
| `conflict_eq_pres` | 2 | ≤ +0.005 (B-b) | -0.0243 [-0.0410, -0.0144, -0.0174] | +0.0028 [+0.0000, +0.0044, +0.0038] | [-0.0410, -0.0144] | PASS |
| `conflict_eq_pres` | 4 | ≤ +0.005 (B-b) | -0.0174 [-0.0035, -0.0208, -0.0279] | -0.0197 [-0.0070, +0.0000, -0.0522] | [-0.0279, -0.0035] | PASS |

## C — Replay safety: ΔForgetting vs `replay` ≤ +0.005 at all K (selection Jaccard carries **no** requirement, by design)

| config | K | mean ΔForgetting vs `replay` [0,1,2] | 95% CI | mean ΔJaccard vs `ours_uniform` (descriptive only) | C (this K) |
|---|---|---|---|---|---|
| `proj_distill` | 1 | -0.0075 [-0.0183, +0.0034, -0.0076] | [-0.0183, +0.0034] | -0.0300 [+0.0237, -0.0696, -0.0441] | PASS |
| `proj_distill` | 2 | -0.0011 [-0.0208, +0.0140, +0.0034] | [-0.0208, +0.0140] | -0.0082 [-0.0218, -0.0108, +0.0079] | PASS |
| `proj_distill` | 4 | -0.0068 [-0.0172, -0.0106, +0.0073] | [-0.0172, +0.0073] | +0.0450 [+0.0908, +0.0153, +0.0287] | PASS |
| `proj_eq_pres` | 1 | -0.0134 [-0.0240, +0.0002, -0.0165] | [-0.0240, +0.0002] | -0.0900 [-0.0638, -0.1111, -0.0950] | PASS |
| `proj_eq_pres` | 2 | +0.0082 [-0.0034, +0.0238, +0.0043] | [-0.0034, +0.0238] | -0.0373 [-0.0222, -0.0585, -0.0313] | FAIL |
| `proj_eq_pres` | 4 | +0.0012 [+0.0162, -0.0238, +0.0113] | [-0.0238, +0.0162] | +0.0585 [+0.0420, +0.0187, +0.1149] | PASS |
| `conflict_eq_pres` | 1 | -0.0005 [-0.0012, -0.0034, +0.0030] | [-0.0034, +0.0030] | -0.1086 [-0.0871, -0.1471, -0.0915] | PASS |
| `conflict_eq_pres` | 2 | -0.0049 [-0.0242, +0.0162, -0.0068] | [-0.0242, +0.0162] | -0.0134 [-0.0043, -0.0145, -0.0213] | PASS |
| `conflict_eq_pres` | 4 | -0.0242 [-0.0176, -0.0445, -0.0104] | [-0.0445, -0.0104] | +0.0410 [+0.0663, -0.0078, +0.0645] | PASS |

## Mechanism link — gradient conflict (pre-registered hypothesis, reported post-hoc)

Per Sol amendment item 2 / rider r2 there is **no numeric early-abort and no threshold chosen after observing these values**. The table exists so the three-way review can judge whether the gradient-conflict explanation of the Gate-v2 failure is supported.

### Diagnosed target cells (`brca` K=2, `lung` K=4)

| config | task (stage) | K | updates | conflict fraction [per-seed 0,1,2] | mean cos | mean \|proj\|/\|g_m\| (conflicting updates) |
|---|---|---|---|---|---|---|
| `eq_pres_diag` | brca (stage 4) | 2 | 45780 | 0.4913 [0.4904, 0.5074, 0.4761] | +0.0006 | 0.0898 |
| `eq_pres_diag` | lung (stage 2) | 4 | 92880 | 0.4988 [0.4897, 0.4943, 0.5124] | -0.0006 | 0.0903 |
| `proj_distill` | brca (stage 4) | 2 | 45780 | 0.4678 [0.4667, 0.4716, 0.4653] | +0.0154 | 0.0847 |
| `proj_distill` | lung (stage 2) | 4 | 92880 | 0.4577 [0.4620, 0.4588, 0.4522] | +0.0617 | 0.0979 |
| `proj_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4876 [0.4857, 0.4903, 0.4868] | +0.0018 | 0.0651 |
| `proj_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4904 [0.4788, 0.4993, 0.4930] | +0.0018 | 0.0548 |
| `conflict_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4773 [0.4757, 0.4863, 0.4701] | +0.0066 | 0.1090 |
| `conflict_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4806 [0.4696, 0.4967, 0.4755] | +0.0062 | 0.1245 |

`eq_pres_diag` is the instrumentation-only replication of the frozen `eq_pres` run (§2.5): **admitted**, so its rows are a direct measurement of the config that actually failed Gate v2.

### All stages and budgets

| config | task (stage) | K | updates | conflict fraction [per-seed 0,1,2] | mean cos | mean \|proj\|/\|g_m\| (conflicting updates) |
|---|---|---|---|---|---|---|
| `eq_pres_diag` | lung (stage 2) | 1 | 23220 | 0.4580 [0.4540, 0.4544, 0.4655] | +0.0115 | 0.0928 |
| `eq_pres_diag` | lung (stage 2) | 2 | 46440 | 0.4958 [0.4964, 0.5038, 0.4873] | -0.0009 | 0.1319 |
| `eq_pres_diag` | lung (stage 2) | 4 | 92880 | 0.4988 [0.4897, 0.4943, 0.5124] | -0.0006 | 0.0903 |
| `eq_pres_diag` | rcc (stage 3) | 1 | 18480 | 0.5039 [0.5141, 0.4945, 0.5031] | +0.0005 | 0.0728 |
| `eq_pres_diag` | rcc (stage 3) | 2 | 36960 | 0.5028 [0.4959, 0.4938, 0.5188] | +0.0001 | 0.0585 |
| `eq_pres_diag` | rcc (stage 3) | 4 | 73920 | 0.4787 [0.4798, 0.4765, 0.4798] | +0.0012 | 0.0320 |
| `eq_pres_diag` | brca (stage 4) | 1 | 22890 | 0.4768 [0.4744, 0.4813, 0.4746] | +0.0053 | 0.0728 |
| `eq_pres_diag` | brca (stage 4) | 2 | 45780 | 0.4913 [0.4904, 0.5074, 0.4761] | +0.0006 | 0.0898 |
| `eq_pres_diag` | brca (stage 4) | 4 | 91560 | 0.4861 [0.4853, 0.4884, 0.4847] | +0.0003 | 0.0526 |
| `proj_distill` | lung (stage 2) | 1 | 23220 | 0.4889 [0.4828, 0.4929, 0.4910] | +0.0033 | 0.1207 |
| `proj_distill` | lung (stage 2) | 2 | 46440 | 0.4446 [0.4467, 0.4573, 0.4297] | +0.0496 | 0.1507 |
| `proj_distill` | lung (stage 2) | 4 | 92880 | 0.4577 [0.4620, 0.4588, 0.4522] | +0.0617 | 0.0979 |
| `proj_distill` | rcc (stage 3) | 1 | 18480 | 0.4940 [0.5117, 0.4797, 0.4907] | +0.0026 | 0.0934 |
| `proj_distill` | rcc (stage 3) | 2 | 36960 | 0.4890 [0.4905, 0.4814, 0.4952] | +0.0102 | 0.0715 |
| `proj_distill` | rcc (stage 3) | 4 | 73920 | 0.4699 [0.4759, 0.4708, 0.4630] | +0.0077 | 0.0394 |
| `proj_distill` | brca (stage 4) | 1 | 22890 | 0.4730 [0.4657, 0.4814, 0.4721] | +0.0087 | 0.0947 |
| `proj_distill` | brca (stage 4) | 2 | 45780 | 0.4678 [0.4667, 0.4716, 0.4653] | +0.0154 | 0.0847 |
| `proj_distill` | brca (stage 4) | 4 | 91560 | 0.4710 [0.4747, 0.4690, 0.4695] | +0.0202 | 0.0584 |
| `proj_eq_pres` | lung (stage 2) | 1 | 23220 | 0.4593 [0.4572, 0.4552, 0.4656] | +0.0112 | 0.0947 |
| `proj_eq_pres` | lung (stage 2) | 2 | 46440 | 0.4894 [0.4938, 0.4835, 0.4910] | +0.0001 | 0.1376 |
| `proj_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4904 [0.4788, 0.4993, 0.4930] | +0.0018 | 0.0548 |
| `proj_eq_pres` | rcc (stage 3) | 1 | 18480 | 0.4971 [0.5143, 0.4872, 0.4898] | +0.0012 | 0.0777 |
| `proj_eq_pres` | rcc (stage 3) | 2 | 36960 | 0.4970 [0.5015, 0.4890, 0.5003] | +0.0004 | 0.0549 |
| `proj_eq_pres` | rcc (stage 3) | 4 | 73920 | 0.4798 [0.4867, 0.4819, 0.4708] | +0.0009 | 0.0276 |
| `proj_eq_pres` | brca (stage 4) | 1 | 22890 | 0.4762 [0.4716, 0.4814, 0.4756] | +0.0056 | 0.0767 |
| `proj_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4876 [0.4857, 0.4903, 0.4868] | +0.0018 | 0.0651 |
| `proj_eq_pres` | brca (stage 4) | 4 | 91560 | 0.4869 [0.4893, 0.4865, 0.4851] | +0.0016 | 0.0316 |
| `conflict_eq_pres` | lung (stage 2) | 1 | 23220 | 0.4702 [0.4651, 0.4655, 0.4800] | +0.0069 | 0.1069 |
| `conflict_eq_pres` | lung (stage 2) | 2 | 46440 | 0.4742 [0.4696, 0.4789, 0.4743] | +0.0156 | 0.1457 |
| `conflict_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4806 [0.4696, 0.4967, 0.4755] | +0.0062 | 0.1245 |
| `conflict_eq_pres` | rcc (stage 3) | 1 | 18480 | 0.4884 [0.4989, 0.4744, 0.4920] | +0.0037 | 0.0847 |
| `conflict_eq_pres` | rcc (stage 3) | 2 | 36960 | 0.4980 [0.5043, 0.4847, 0.5050] | +0.0012 | 0.0713 |
| `conflict_eq_pres` | rcc (stage 3) | 4 | 73920 | 0.4644 [0.4682, 0.4624, 0.4627] | +0.0015 | 0.0330 |
| `conflict_eq_pres` | brca (stage 4) | 1 | 22890 | 0.4689 [0.4654, 0.4763, 0.4649] | +0.0066 | 0.0870 |
| `conflict_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4773 [0.4757, 0.4863, 0.4701] | +0.0066 | 0.1090 |
| `conflict_eq_pres` | brca (stage 4) | 4 | 91560 | 0.4766 [0.4729, 0.4886, 0.4684] | +0.0009 | 0.0475 |

Stage 1 never appears: the first task has no buffer and no `old_nav`, so there is no memory gradient to conflict with (§2.2).

## Overall development-gate verdict

| config | A (utility) | B-a (no new plasticity failure) | B-b (no new forgetting regression) | B-c (diagnosed targets) | C (replay safety) | DEV GATE |
|---|---|---|---|---|---|---|
| `proj_distill` | FAIL | FAIL | FAIL | FAIL | PASS | **DEV FAIL** |
| `proj_eq_pres` | PASS | FAIL | FAIL | PASS | FAIL | **DEV FAIL** |
| `conflict_eq_pres` | PASS | FAIL | PASS | FAIL | PASS | **DEV FAIL** |

### Winner selection (§3.4, pre-registered before training)

No candidate config passes, so there is no winner and no confirmation run. Branch: §4 FAIL — archive honestly, ship framework-first with `eq_pres` as the diagnostic precursor.

## Branch and STOP

**Development gate FAILS.** Per §4 the honest branch is to archive: the paper ships framework-first with `eq_pres` as the diagnostic precursor, and the three v0.34 configs are reported as tested-negative with their full per-seed tables, exactly as `ia_samp`/`ia_ep` are. **STOP for three-way review** — no further method compute is opened, and per §0 there is no v0.35 for this ICASSP submission.

This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable review**. No threshold was altered, no config was added, and no v0.33 cell was re-scored in producing it.

