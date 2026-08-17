# Track B close-out — three descriptive analyses (ZERO COMPUTE)

**Track A:** frozen; writing continues. **Track B:** closed and archived (`docs/method_gate_v034.md` §0/§4 — no v0.35).

Assigned in Fable's ratification of the v0.34 DEV FAIL verdict (2026-08-17). Every number below is derived from CSVs that already existed — **no run was executed and no gate is reopened**. All three sections are **descriptive / exploratory**: they apply no threshold and change no verdict. The v0.33.3, Gate v2 and v0.34 verdicts stand exactly as written.

## (a) Utility-axis replication across every `L_eq`-bearing config

This is the one Track-B effect that replicates. `eq_pres` (5 seeds), `proj_eq_pres` (3) and `conflict_eq_pres` (3) all carry `L_eq`; `proj_distill` carries the gradient arbiter but **no** `L_eq` and is shown as the contrast. Aggregation is the frozen §5.1 convention (final stage, old tasks only, mean within task then macro-average). Comparator utility columns come from the checksum-passed MPS backfills.

| config | has `L_eq` | K | metric | vs `ours_uniform` mean [per-seed] | vs `distill` mean [per-seed] | seeds improving both |
|---|---|---|---|---|---|---|
| `eq_pres` | **yes** | 1 | `eps_optimal_mass` | +0.0326 [+0.0195, +0.0397, +0.0332, +0.0437, +0.0269] | +0.0493 [+0.0351, +0.0489, +0.0376, +0.0762, +0.0488] | 5/5 |
| `eq_pres` | **yes** | 1 | `normalized_regret` | -0.0078 [-0.0048, -0.0095, -0.0065, -0.0133, -0.0048] | -0.0111 [-0.0063, -0.0108, -0.0034, -0.0246, -0.0107] | 5/5 |
| `eq_pres` | **yes** | 2 | `eps_optimal_mass` | +0.0297 [+0.0355, +0.0280, +0.0292, +0.0260, +0.0298] | +0.0360 [+0.0399, +0.0394, +0.0263, +0.0378, +0.0367] | 5/5 |
| `eq_pres` | **yes** | 2 | `normalized_regret` | -0.0093 [-0.0133, -0.0070, -0.0088, -0.0077, -0.0096] | -0.0107 [-0.0129, -0.0105, -0.0078, -0.0122, -0.0103] | 5/5 |
| `proj_eq_pres` | **yes** | 1 | `eps_optimal_mass` | +0.0411 [+0.0293, +0.0432, +0.0508] | +0.0509 [+0.0449, +0.0524, +0.0552] | 3/3 |
| `proj_eq_pres` | **yes** | 1 | `normalized_regret` | -0.0133 [-0.0100, -0.0152, -0.0146] | -0.0131 [-0.0115, -0.0165, -0.0115] | 3/3 |
| `proj_eq_pres` | **yes** | 2 | `eps_optimal_mass` | +0.0293 [+0.0374, +0.0280, +0.0224] | +0.0335 [+0.0418, +0.0393, +0.0195] | 3/3 |
| `proj_eq_pres` | **yes** | 2 | `normalized_regret` | -0.0099 [-0.0133, -0.0081, -0.0082] | -0.0105 [-0.0129, -0.0116, -0.0071] | 3/3 |
| `conflict_eq_pres` | **yes** | 1 | `eps_optimal_mass` | +0.0111 [+0.0080, +0.0143, +0.0111] | +0.0209 [+0.0236, +0.0235, +0.0155] | 3/3 |
| `conflict_eq_pres` | **yes** | 1 | `normalized_regret` | -0.0035 [-0.0019, -0.0066, -0.0020] | -0.0034 [-0.0034, -0.0079, +0.0011] | 2/3 |
| `conflict_eq_pres` | **yes** | 2 | `eps_optimal_mass` | +0.0133 [+0.0138, +0.0155, +0.0106] | +0.0176 [+0.0182, +0.0268, +0.0077] | 3/3 |
| `conflict_eq_pres` | **yes** | 2 | `normalized_regret` | -0.0037 [-0.0050, -0.0034, -0.0026] | -0.0043 [-0.0046, -0.0069, -0.0016] | 3/3 |
| `proj_distill` | no | 1 | `eps_optimal_mass` | +0.0011 [-0.0025, +0.0009, +0.0050] | +0.0109 [+0.0132, +0.0101, +0.0094] | 2/3 |
| `proj_distill` | no | 1 | `normalized_regret` | -0.0006 [+0.0006, +0.0002, -0.0025] | -0.0004 [-0.0009, -0.0011, +0.0007] | 0/3 |
| `proj_distill` | no | 2 | `eps_optimal_mass` | -0.0028 [-0.0029, -0.0041, -0.0014] | +0.0015 [+0.0015, +0.0072, -0.0043] | 0/3 |
| `proj_distill` | no | 2 | `normalized_regret` | +0.0019 [+0.0013, +0.0029, +0.0014] | +0.0012 [+0.0017, -0.0007, +0.0024] | 0/3 |

Both-metric, both-comparator agreement per config and K (the strictest reading):

| config | K=1 | K=2 |
|---|---|---|
| `eq_pres` | 5/5 | 5/5 |
| `proj_eq_pres` | 3/3 | 3/3 |
| `conflict_eq_pres` | 2/3 | 3/3 |
| `proj_distill` | 0/3 | 0/3 |

**Read:** the utility effect tracks `L_eq`, not the arbiter. Every config carrying `L_eq` improves both utility metrics against both comparators on essentially every seed; `proj_distill`, which has the same arbiter but no `L_eq`, does not. Descriptive only — no threshold is applied here, and none of these numbers re-scores any gate.

## (b) Resolution and power note for the plasticity cells

Every cell-level plasticity number in Track B rests on a per-task test set of 15-95 slides and three seeds. The table gives the measurement quantum `1/n_test` — the accuracy change caused by a **single** test item flipping — next to the observed per-seed spread. Where the spread is several quanta wide and the mean is around one quantum, the cell cannot resolve the pre-registered ±0.01 threshold, which is why cell-level differences are reported as falling within seed variability.

| config | task | K | quantum `1/n_test` | mean ΔA[t,t] vs `ours_uniform` | per-seed | spread | spread / quantum |
|---|---|---|---|---|---|---|---|
| `proj_eq_pres` | esca | 1 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_eq_pres` | esca | 2 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_eq_pres` | esca | 4 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_eq_pres` | lung | 1 | 0.0105 (n=95) | -0.0034 | [-0.0102, +0.0102, -0.0102] | 0.0204 | 1.9x |
| `proj_eq_pres` | lung | 2 | 0.0105 (n=95) | +0.0068 | [-0.0204, +0.0218, +0.0191] | 0.0422 | 4.0x |
| `proj_eq_pres` | lung **(diagnosed)** | 4 | 0.0105 (n=95) | +0.0177 | [+0.0204, +0.0415, -0.0089] | 0.0504 | 4.8x |
| `proj_eq_pres` | rcc | 1 | 0.0132 (n=76) | -0.0096 | [-0.0185, +0.0083, -0.0185] | 0.0268 | 2.0x |
| `proj_eq_pres` | rcc | 2 | 0.0132 (n=76) | +0.0096 | [+0.0204, -0.0204, +0.0287] | 0.0491 | 3.7x |
| `proj_eq_pres` | rcc | 4 | 0.0132 (n=76) | +0.0034 | [+0.0287, -0.0371, +0.0185] | 0.0658 | 5.0x |
| `proj_eq_pres` | brca | 1 | 0.0108 (n=93) | +0.0022 | [+0.0594, +0.0000, -0.0527] | 0.1121 | 10.4x |
| `proj_eq_pres` | brca **(diagnosed)** | 2 | 0.0108 (n=93) | +0.0088 | [-0.0331, +0.0068, +0.0526] | 0.0857 | 8.0x |
| `proj_eq_pres` | brca | 4 | 0.0108 (n=93) | -0.0331 | [+0.0000, -0.0729, -0.0263] | 0.0729 | 6.8x |
| `conflict_eq_pres` | esca | 1 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `conflict_eq_pres` | esca | 2 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `conflict_eq_pres` | esca | 4 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `conflict_eq_pres` | lung | 1 | 0.0105 (n=95) | +0.0138 | [+0.0211, +0.0000, +0.0204] | 0.0211 | 2.0x |
| `conflict_eq_pres` | lung | 2 | 0.0105 (n=95) | +0.0241 | [+0.0000, +0.0211, +0.0511] | 0.0511 | 4.9x |
| `conflict_eq_pres` | lung **(diagnosed)** | 4 | 0.0105 (n=95) | +0.0000 | [+0.0095, +0.0109, -0.0204] | 0.0313 | 3.0x |
| `conflict_eq_pres` | rcc | 1 | 0.0132 (n=76) | +0.0000 | [-0.0102, -0.0102, +0.0204] | 0.0306 | 2.3x |
| `conflict_eq_pres` | rcc | 2 | 0.0132 (n=76) | +0.0062 | [+0.0102, -0.0019, +0.0102] | 0.0121 | 0.9x |
| `conflict_eq_pres` | rcc | 4 | 0.0132 (n=76) | -0.0096 | [+0.0000, -0.0185, -0.0102] | 0.0185 | 1.4x |
| `conflict_eq_pres` | brca | 1 | 0.0108 (n=93) | +0.0130 | [+0.0263, +0.0195, -0.0068] | 0.0331 | 3.1x |
| `conflict_eq_pres` | brca **(diagnosed)** | 2 | 0.0108 (n=93) | -0.0351 | [-0.1053, -0.0263, +0.0263] | 0.1316 | 12.2x |
| `conflict_eq_pres` | brca | 4 | 0.0108 (n=93) | -0.0306 | [+0.0331, -0.0654, -0.0594] | 0.0985 | 9.2x |
| `proj_distill` | esca | 1 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_distill` | esca | 2 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_distill` | esca | 4 | 0.0667 (n=15) | +0.0000 | [+0.0000, +0.0000, +0.0000] | 0.0000 | 0.0x |
| `proj_distill` | lung | 1 | 0.0105 (n=95) | -0.0032 | [-0.0102, -0.0102, +0.0109] | 0.0211 | 2.0x |
| `proj_distill` | lung | 2 | 0.0105 (n=95) | -0.0034 | [-0.0306, +0.0109, +0.0096] | 0.0415 | 3.9x |
| `proj_distill` | lung **(diagnosed)** | 4 | 0.0105 (n=95) | -0.0100 | [+0.0204, +0.0089, -0.0592] | 0.0796 | 7.6x |
| `proj_distill` | rcc | 1 | 0.0132 (n=76) | +0.0028 | [-0.0102, +0.0083, +0.0102] | 0.0204 | 1.6x |
| `proj_distill` | rcc | 2 | 0.0132 (n=76) | +0.0034 | [+0.0000, +0.0000, +0.0102] | 0.0102 | 0.8x |
| `proj_distill` | rcc | 4 | 0.0132 (n=76) | +0.0034 | [+0.0000, +0.0019, +0.0083] | 0.0083 | 0.6x |
| `proj_distill` | brca | 1 | 0.0108 (n=93) | +0.0198 | [+0.0398, +0.0526, -0.0331] | 0.0857 | 8.0x |
| `proj_distill` | brca **(diagnosed)** | 2 | 0.0108 (n=93) | -0.0153 | [-0.0526, +0.0000, +0.0067] | 0.0593 | 5.5x |
| `proj_distill` | brca | 4 | 0.0108 (n=93) | +0.0175 | [+0.0331, +0.0067, +0.0128] | 0.0264 | 2.5x |

**Read:** across the cells that decided the v0.34 verdict the per-seed spread is several times the quantum, and the cell means are comparable to a single test item flipping. The pre-registered cell-level plasticity criteria are **under-powered at this sample size** — a limitation of the gate design, disclosed rather than argued away, and not a property of any configuration.

## (c) Gradient-conflict null

Per-update instrumentation from the v0.34 run. `eq_pres_diag` is the instrumentation-only replication of the frozen `eq_pres` run, admitted at 90/90 bit-identical — so for that row this is a **direct measurement of the configuration that actually failed Gate v2**, not a proxy.

| config | task (stage) | K | updates | conflict fraction | mean cos | cell |
|---|---|---|---|---|---|---|
| `eq_pres_diag` | lung (stage 2) | 1 | 23220 | 0.4580 | +0.0115 | healthy |
| `eq_pres_diag` | lung (stage 2) | 2 | 46440 | 0.4958 | -0.0009 | healthy |
| `eq_pres_diag` | lung (stage 2) | 4 | 92880 | 0.4988 | -0.0006 | **diagnosed target** |
| `eq_pres_diag` | rcc (stage 3) | 1 | 18480 | 0.5039 | +0.0005 | healthy |
| `eq_pres_diag` | rcc (stage 3) | 2 | 36960 | 0.5028 | +0.0001 | healthy |
| `eq_pres_diag` | rcc (stage 3) | 4 | 73920 | 0.4787 | +0.0012 | healthy |
| `eq_pres_diag` | brca (stage 4) | 1 | 22890 | 0.4768 | +0.0053 | healthy |
| `eq_pres_diag` | brca (stage 4) | 2 | 45780 | 0.4913 | +0.0006 | **diagnosed target** |
| `eq_pres_diag` | brca (stage 4) | 4 | 91560 | 0.4861 | +0.0003 | healthy |
| `proj_eq_pres` | lung (stage 2) | 1 | 23220 | 0.4593 | +0.0112 | healthy |
| `proj_eq_pres` | lung (stage 2) | 2 | 46440 | 0.4894 | +0.0001 | healthy |
| `proj_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4904 | +0.0018 | **diagnosed target** |
| `proj_eq_pres` | rcc (stage 3) | 1 | 18480 | 0.4971 | +0.0012 | healthy |
| `proj_eq_pres` | rcc (stage 3) | 2 | 36960 | 0.4970 | +0.0004 | healthy |
| `proj_eq_pres` | rcc (stage 3) | 4 | 73920 | 0.4798 | +0.0009 | healthy |
| `proj_eq_pres` | brca (stage 4) | 1 | 22890 | 0.4762 | +0.0056 | healthy |
| `proj_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4876 | +0.0018 | **diagnosed target** |
| `proj_eq_pres` | brca (stage 4) | 4 | 91560 | 0.4869 | +0.0016 | healthy |
| `conflict_eq_pres` | lung (stage 2) | 1 | 23220 | 0.4702 | +0.0069 | healthy |
| `conflict_eq_pres` | lung (stage 2) | 2 | 46440 | 0.4742 | +0.0156 | healthy |
| `conflict_eq_pres` | lung (stage 2) | 4 | 92880 | 0.4806 | +0.0062 | **diagnosed target** |
| `conflict_eq_pres` | rcc (stage 3) | 1 | 18480 | 0.4884 | +0.0037 | healthy |
| `conflict_eq_pres` | rcc (stage 3) | 2 | 36960 | 0.4980 | +0.0012 | healthy |
| `conflict_eq_pres` | rcc (stage 3) | 4 | 73920 | 0.4644 | +0.0015 | healthy |
| `conflict_eq_pres` | brca (stage 4) | 1 | 22890 | 0.4689 | +0.0066 | healthy |
| `conflict_eq_pres` | brca (stage 4) | 2 | 45780 | 0.4773 | +0.0066 | **diagnosed target** |
| `conflict_eq_pres` | brca (stage 4) | 4 | 91560 | 0.4766 | +0.0009 | healthy |
| `proj_distill` | lung (stage 2) | 1 | 23220 | 0.4889 | +0.0033 | healthy |
| `proj_distill` | lung (stage 2) | 2 | 46440 | 0.4446 | +0.0496 | healthy |
| `proj_distill` | lung (stage 2) | 4 | 92880 | 0.4577 | +0.0617 | **diagnosed target** |
| `proj_distill` | rcc (stage 3) | 1 | 18480 | 0.4940 | +0.0026 | healthy |
| `proj_distill` | rcc (stage 3) | 2 | 36960 | 0.4890 | +0.0102 | healthy |
| `proj_distill` | rcc (stage 3) | 4 | 73920 | 0.4699 | +0.0077 | healthy |
| `proj_distill` | brca (stage 4) | 1 | 22890 | 0.4730 | +0.0087 | healthy |
| `proj_distill` | brca (stage 4) | 2 | 45780 | 0.4678 | +0.0154 | **diagnosed target** |
| `proj_distill` | brca (stage 4) | 4 | 91560 | 0.4710 | +0.0202 | healthy |

**Read:** conflict fraction spans 0.4446-0.5039 over seed-averaged cells (0.4297-0.5188 over the individual per-stage-per-seed summaries), with mean cosine about zero everywhere. The diagnosed target cells are indistinguishable from healthy ones — the highest cell of all is `eq_pres_diag` rcc K=1 at 0.5039, a cell that never had a plasticity problem. Projection does not change the conflict rate either (`brca` K=2: `eq_pres_diag` 0.4913 vs `proj_eq_pres` 0.4876), so the conflict is a structural property of two unrelated high-dimensional gradients rather than a removable lesion. Gradient conflict is the field's default explanation for this kind of interference; a pre-registered instrumented measurement showing it does not hold for continual evidence acquisition is a **null result**, reported as such.

