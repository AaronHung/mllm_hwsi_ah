# WP1 paired statistics pack

v0.292 zero-compute analysis of frozen per-seed CSVs, extended in v0.33 (carry-over c1) with `*-seqft` rows. Positive `mean` means the first method is better for the metric direction.

## Statistics policy (v0.33.1, decided — see docs/handoff_fable_sol_20260815.md §2.6)

This report is **descriptive by design, not a significance test**. With `n=5` paired seeds, the exact two-sided sign-flip test's smallest achievable p-value is `p_min = 2/2^5 = 0.0625 > 0.05`; Benjamini–Hochberg's adjusted q-value for the single best-ranked hypothesis in any family of size `m` is `q_(1) = p_(1)`, independent of `m`, so **no comparison family size can ever clear `q <= 0.05` here** — shrinking the family does not help. `p_value`/`q_value` are kept in the CSV for audit only; paper-facing text must use only `mean`, the 95% bootstrap CI, and `sign_agree` (how many of the `n` seeds agree with `mean`'s sign, e.g. `5/5`). Each row is also tagged `confirmatory` (the six comparisons pre-registered in docs/research_contract_v0.292.md) or `exploratory` (the `*-seqft` rows added afterward in v0.33) — do not present exploratory rows as pre-registered.

## Coverage

- `main` full grid: available
- `main:ours_uniform`: available
- `main:mem2048`: available
- `main:lambda3`: available
- `reverse` full grid: available
- `reverse:ours_uniform`: missing
- `reverse:mem2048`: missing
- `reverse:lambda3`: missing

## Results

### main, K=1, `distill-replay` (confirmatory)
- AA: mean=-0.0009, 95% CI [-0.0118, +0.0120], sign_agree=3/5
- Forgetting: mean=+0.0014, 95% CI [-0.0051, +0.0067], sign_agree=4/5
- BWT: mean=+0.0033, 95% CI [-0.0090, +0.0156], sign_agree=3/5
- Jaccard: mean=+0.1162, 95% CI [+0.0844, +0.1479], sign_agree=5/5
- action-KL: mean=+0.0453, 95% CI [+0.0370, +0.0530], sign_agree=5/5

### main, K=1, `distill-seqft` (exploratory)
- AA: mean=+0.0337, 95% CI [+0.0025, +0.0611], sign_agree=4/5
- Forgetting: mean=+0.0553, 95% CI [+0.0192, +0.0895], sign_agree=4/5
- BWT: mean=+0.0641, 95% CI [+0.0342, +0.0961], sign_agree=5/5
- Jaccard: mean=+0.2450, 95% CI [+0.2019, +0.2814], sign_agree=5/5
- action-KL: mean=+0.2637, 95% CI [+0.2386, +0.2872], sign_agree=5/5

### main, K=1, `lambda3-ours` (confirmatory)
- AA: mean=+0.0151, 95% CI [+0.0056, +0.0246], sign_agree=4/5
- Forgetting: mean=-0.0023, 95% CI [-0.0081, +0.0062], sign_agree=4/5
- BWT: mean=+0.0072, 95% CI [-0.0084, +0.0199], sign_agree=3/5
- Jaccard: mean=+0.0878, 95% CI [+0.0476, +0.1106], sign_agree=5/5
- action-KL: mean=+0.0220, 95% CI [+0.0162, +0.0294], sign_agree=5/5

### main, K=1, `mem2048-ours` (confirmatory)
- AA: mean=+0.0102, 95% CI [+0.0047, +0.0166], sign_agree=5/5
- Forgetting: mean=-0.0008, 95% CI [-0.0102, +0.0136], sign_agree=4/5
- BWT: mean=+0.0041, 95% CI [-0.0106, +0.0188], sign_agree=3/5
- Jaccard: mean=+0.0200, 95% CI [-0.0235, +0.0577], sign_agree=4/5
- action-KL: mean=+0.0036, 95% CI [-0.0021, +0.0098], sign_agree=3/5

### main, K=1, `ours-distill` (confirmatory)
- AA: mean=-0.0039, 95% CI [-0.0134, +0.0041], sign_agree=3/5
- Forgetting: mean=+0.0038, 95% CI [-0.0038, +0.0122], sign_agree=3/5
- BWT: mean=-0.0051, 95% CI [-0.0100, +0.0009], sign_agree=4/5
- Jaccard: mean=-0.0368, 95% CI [-0.0876, +0.0269], sign_agree=4/5
- action-KL: mean=-0.0109, 95% CI [-0.0188, -0.0043], sign_agree=5/5

### main, K=1, `ours-ours_uniform` (confirmatory)
- AA: mean=-0.0158, 95% CI [-0.0203, -0.0114], sign_agree=5/5
- Forgetting: mean=+0.0008, 95% CI [-0.0101, +0.0152], sign_agree=2/5
- BWT: mean=-0.0096, 95% CI [-0.0214, +0.0009], sign_agree=4/5
- Jaccard: mean=+0.0057, 95% CI [-0.0196, +0.0259], sign_agree=3/5
- action-KL: mean=-0.0013, 95% CI [-0.0038, +0.0004], sign_agree=4/5

### main, K=1, `ours-replay` (confirmatory)
- AA: mean=-0.0048, 95% CI [-0.0109, +0.0004], sign_agree=4/5
- Forgetting: mean=+0.0052, 95% CI [-0.0057, +0.0160], sign_agree=3/5
- BWT: mean=-0.0018, 95% CI [-0.0170, +0.0149], sign_agree=3/5
- Jaccard: mean=+0.0794, 95% CI [+0.0520, +0.1148], sign_agree=5/5
- action-KL: mean=+0.0344, 95% CI [+0.0275, +0.0414], sign_agree=5/5

### main, K=1, `ours-seqft` (exploratory)
- AA: mean=+0.0298, 95% CI [-0.0020, +0.0557], sign_agree=4/5
- Forgetting: mean=+0.0590, 95% CI [+0.0273, +0.0928], sign_agree=5/5
- BWT: mean=+0.0591, 95% CI [+0.0273, +0.0970], sign_agree=5/5
- Jaccard: mean=+0.2082, 95% CI [+0.1894, +0.2298], sign_agree=5/5
- action-KL: mean=+0.2528, 95% CI [+0.2209, +0.2802], sign_agree=5/5

### main, K=1, `replay-seqft` (exploratory)
- AA: mean=+0.0346, 95% CI [+0.0030, +0.0591], sign_agree=4/5
- Forgetting: mean=+0.0539, 95% CI [+0.0181, +0.0842], sign_agree=4/5
- BWT: mean=+0.0608, 95% CI [+0.0334, +0.0882], sign_agree=5/5
- Jaccard: mean=+0.1288, 95% CI [+0.1125, +0.1459], sign_agree=5/5
- action-KL: mean=+0.2184, 95% CI [+0.1896, +0.2448], sign_agree=5/5

### main, K=2, `distill-replay` (confirmatory)
- AA: mean=+0.0040, 95% CI [-0.0037, +0.0116], sign_agree=3/5
- Forgetting: mean=+0.0041, 95% CI [-0.0069, +0.0152], sign_agree=3/5
- BWT: mean=+0.0075, 95% CI [-0.0084, +0.0234], sign_agree=3/5
- Jaccard: mean=+0.0522, 95% CI [+0.0345, +0.0712], sign_agree=5/5
- action-KL: mean=+0.0349, 95% CI [+0.0288, +0.0423], sign_agree=5/5

### main, K=2, `distill-seqft` (exploratory)
- AA: mean=+0.0186, 95% CI [-0.0086, +0.0465], sign_agree=3/5
- Forgetting: mean=+0.0279, 95% CI [-0.0005, +0.0570], sign_agree=4/5
- BWT: mean=+0.0252, 95% CI [-0.0157, +0.0621], sign_agree=4/5
- Jaccard: mean=+0.0814, 95% CI [+0.0648, +0.0984], sign_agree=5/5
- action-KL: mean=+0.0908, 95% CI [+0.0814, +0.1004], sign_agree=5/5

### main, K=2, `lambda3-ours` (confirmatory)
- AA: mean=+0.0022, 95% CI [-0.0115, +0.0162], sign_agree=3/5
- Forgetting: mean=-0.0056, 95% CI [-0.0204, +0.0108], sign_agree=3/5
- BWT: mean=+0.0004, 95% CI [-0.0212, +0.0221], sign_agree=3/5
- Jaccard: mean=+0.0572, 95% CI [+0.0410, +0.0705], sign_agree=5/5
- action-KL: mean=+0.0122, 95% CI [+0.0107, +0.0138], sign_agree=5/5

### main, K=2, `mem2048-ours` (confirmatory)
- AA: mean=+0.0098, 95% CI [+0.0018, +0.0188], sign_agree=5/5
- Forgetting: mean=+0.0104, 95% CI [+0.0026, +0.0183], sign_agree=4/5
- BWT: mean=+0.0138, 95% CI [+0.0071, +0.0207], sign_agree=5/5
- Jaccard: mean=+0.0453, 95% CI [+0.0318, +0.0611], sign_agree=5/5
- action-KL: mean=+0.0074, 95% CI [+0.0063, +0.0089], sign_agree=5/5

### main, K=2, `ours-distill` (confirmatory)
- AA: mean=-0.0047, 95% CI [-0.0196, +0.0092], sign_agree=3/5
- Forgetting: mean=-0.0003, 95% CI [-0.0118, +0.0120], sign_agree=3/5
- BWT: mean=-0.0024, 95% CI [-0.0162, +0.0169], sign_agree=4/5
- Jaccard: mean=-0.0368, 95% CI [-0.0528, -0.0204], sign_agree=5/5
- action-KL: mean=-0.0080, 95% CI [-0.0113, -0.0053], sign_agree=5/5

### main, K=2, `ours-ours_uniform` (confirmatory)
- AA: mean=+0.0099, 95% CI [-0.0054, +0.0252], sign_agree=3/5
- Forgetting: mean=+0.0229, 95% CI [+0.0042, +0.0416], sign_agree=3/5
- BWT: mean=+0.0201, 95% CI [+0.0088, +0.0315], sign_agree=5/5
- Jaccard: mean=-0.0021, 95% CI [-0.0175, +0.0106], sign_agree=3/5
- action-KL: mean=+0.0024, 95% CI [+0.0014, +0.0034], sign_agree=5/5

### main, K=2, `ours-replay` (confirmatory)
- AA: mean=-0.0007, 95% CI [-0.0202, +0.0142], sign_agree=3/5
- Forgetting: mean=+0.0039, 95% CI [-0.0151, +0.0173], sign_agree=3/5
- BWT: mean=+0.0051, 95% CI [-0.0122, +0.0191], sign_agree=4/5
- Jaccard: mean=+0.0154, 95% CI [-0.0011, +0.0348], sign_agree=4/5
- action-KL: mean=+0.0269, 95% CI [+0.0215, +0.0322], sign_agree=5/5

### main, K=2, `ours-seqft` (exploratory)
- AA: mean=+0.0139, 95% CI [-0.0073, +0.0399], sign_agree=3/5
- Forgetting: mean=+0.0276, 95% CI [+0.0051, +0.0589], sign_agree=5/5
- BWT: mean=+0.0229, 95% CI [-0.0015, +0.0530], sign_agree=4/5
- Jaccard: mean=+0.0446, 95% CI [+0.0309, +0.0583], sign_agree=5/5
- action-KL: mean=+0.0828, 95% CI [+0.0753, +0.0905], sign_agree=5/5

### main, K=2, `replay-seqft` (exploratory)
- AA: mean=+0.0146, 95% CI [-0.0153, +0.0445], sign_agree=3/5
- Forgetting: mean=+0.0238, 95% CI [-0.0080, +0.0555], sign_agree=3/5
- BWT: mean=+0.0177, 95% CI [-0.0145, +0.0500], sign_agree=3/5
- Jaccard: mean=+0.0292, 95% CI [+0.0113, +0.0467], sign_agree=4/5
- action-KL: mean=+0.0559, 95% CI [+0.0508, +0.0607], sign_agree=5/5

### main, K=4, `distill-replay` (confirmatory)
- AA: mean=+0.0101, 95% CI [-0.0099, +0.0297], sign_agree=3/5
- Forgetting: mean=+0.0044, 95% CI [-0.0213, +0.0274], sign_agree=3/5
- BWT: mean=+0.0013, 95% CI [-0.0270, +0.0295], sign_agree=3/5
- Jaccard: mean=+0.0315, 95% CI [-0.0190, +0.0847], sign_agree=4/5
- action-KL: mean=+0.0238, 95% CI [+0.0197, +0.0276], sign_agree=5/5

### main, K=4, `distill-seqft` (exploratory)
- AA: mean=+0.0383, 95% CI [+0.0219, +0.0563], sign_agree=5/5
- Forgetting: mean=+0.0357, 95% CI [-0.0060, +0.0718], sign_agree=4/5
- BWT: mean=+0.0426, 95% CI [-0.0021, +0.0828], sign_agree=4/5
- Jaccard: mean=+0.0868, 95% CI [+0.0673, +0.1051], sign_agree=5/5
- action-KL: mean=+0.0157, 95% CI [+0.0138, +0.0176], sign_agree=5/5

### main, K=4, `ours-distill` (confirmatory)
- AA: mean=+0.0018, 95% CI [-0.0113, +0.0141], sign_agree=3/5
- Forgetting: mean=+0.0112, 95% CI [-0.0122, +0.0347], sign_agree=3/5
- BWT: mean=+0.0101, 95% CI [-0.0130, +0.0338], sign_agree=3/5
- Jaccard: mean=-0.0234, 95% CI [-0.0474, -0.0002], sign_agree=4/5
- action-KL: mean=-0.0060, 95% CI [-0.0073, -0.0047], sign_agree=5/5

### main, K=4, `ours-replay` (confirmatory)
- AA: mean=+0.0119, 95% CI [-0.0024, +0.0252], sign_agree=4/5
- Forgetting: mean=+0.0155, 95% CI [+0.0061, +0.0267], sign_agree=5/5
- BWT: mean=+0.0113, 95% CI [-0.0115, +0.0324], sign_agree=3/5
- Jaccard: mean=+0.0081, 95% CI [-0.0290, +0.0391], sign_agree=3/5
- action-KL: mean=+0.0178, 95% CI [+0.0146, +0.0209], sign_agree=5/5

### main, K=4, `ours-seqft` (exploratory)
- AA: mean=+0.0401, 95% CI [+0.0238, +0.0606], sign_agree=5/5
- Forgetting: mean=+0.0468, 95% CI [+0.0239, +0.0639], sign_agree=5/5
- BWT: mean=+0.0526, 95% CI [+0.0270, +0.0721], sign_agree=5/5
- Jaccard: mean=+0.0634, 95% CI [+0.0525, +0.0749], sign_agree=5/5
- action-KL: mean=+0.0097, 95% CI [+0.0077, +0.0117], sign_agree=5/5

### main, K=4, `replay-seqft` (exploratory)
- AA: mean=+0.0282, 95% CI [+0.0155, +0.0394], sign_agree=5/5
- Forgetting: mean=+0.0313, 95% CI [+0.0137, +0.0490], sign_agree=5/5
- BWT: mean=+0.0413, 95% CI [+0.0199, +0.0627], sign_agree=5/5
- Jaccard: mean=+0.0553, 95% CI [+0.0176, +0.0938], sign_agree=4/5
- action-KL: mean=-0.0081, 95% CI [-0.0129, -0.0045], sign_agree=5/5

### reverse, K=1, `distill-replay` (confirmatory)
- AA: mean=+0.0187, 95% CI [-0.0027, +0.0536], sign_agree=3/5
- Forgetting: mean=+0.0239, 95% CI [-0.0076, +0.0730], sign_agree=3/5
- BWT: mean=+0.0236, 95% CI [-0.0081, +0.0727], sign_agree=3/5
- Jaccard: mean=+0.0901, 95% CI [+0.0664, +0.1138], sign_agree=5/5
- action-KL: mean=+0.0559, 95% CI [+0.0285, +0.1017], sign_agree=5/5

### reverse, K=1, `distill-seqft` (exploratory)
- AA: mean=+0.0756, 95% CI [+0.0534, +0.0997], sign_agree=5/5
- Forgetting: mean=+0.0850, 95% CI [+0.0555, +0.1144], sign_agree=5/5
- BWT: mean=+0.0910, 95% CI [+0.0601, +0.1218], sign_agree=5/5
- Jaccard: mean=+0.2578, 95% CI [+0.2329, +0.2768], sign_agree=5/5
- action-KL: mean=+0.2330, 95% CI [+0.2091, +0.2594], sign_agree=5/5

### reverse, K=1, `ours-distill` (confirmatory)
- AA: mean=-0.0113, 95% CI [-0.0387, +0.0057], sign_agree=3/5
- Forgetting: mean=-0.0207, 95% CI [-0.0559, +0.0002], sign_agree=4/5
- BWT: mean=-0.0183, 95% CI [-0.0549, +0.0038], sign_agree=3/5
- Jaccard: mean=-0.0228, 95% CI [-0.0519, +0.0093], sign_agree=3/5
- action-KL: mean=-0.0106, 95% CI [-0.0242, -0.0008], sign_agree=4/5

### reverse, K=1, `ours-replay` (confirmatory)
- AA: mean=+0.0074, 95% CI [-0.0024, +0.0171], sign_agree=3/5
- Forgetting: mean=+0.0032, 95% CI [-0.0073, +0.0163], sign_agree=3/5
- BWT: mean=+0.0053, 95% CI [-0.0070, +0.0183], sign_agree=4/5
- Jaccard: mean=+0.0673, 95% CI [+0.0302, +0.1045], sign_agree=5/5
- action-KL: mean=+0.0453, 95% CI [+0.0242, +0.0779], sign_agree=5/5

### reverse, K=1, `ours-seqft` (exploratory)
- AA: mean=+0.0643, 95% CI [+0.0254, +0.1016], sign_agree=4/5
- Forgetting: mean=+0.0643, 95% CI [+0.0092, +0.1129], sign_agree=4/5
- BWT: mean=+0.0727, 95% CI [+0.0134, +0.1197], sign_agree=4/5
- Jaccard: mean=+0.2350, 95% CI [+0.2216, +0.2521], sign_agree=5/5
- action-KL: mean=+0.2224, 95% CI [+0.1952, +0.2507], sign_agree=5/5

### reverse, K=1, `replay-seqft` (exploratory)
- AA: mean=+0.0569, 95% CI [+0.0133, +0.0952], sign_agree=4/5
- Forgetting: mean=+0.0611, 95% CI [-0.0075, +0.1153], sign_agree=4/5
- BWT: mean=+0.0674, 95% CI [-0.0057, +0.1249], sign_agree=4/5
- Jaccard: mean=+0.1677, 95% CI [+0.1441, +0.1920], sign_agree=5/5
- action-KL: mean=+0.1771, 95% CI [+0.1273, +0.2234], sign_agree=5/5

### reverse, K=2, `distill-replay` (confirmatory)
- AA: mean=+0.0215, 95% CI [-0.0015, +0.0573], sign_agree=3/5
- Forgetting: mean=+0.0202, 95% CI [-0.0134, +0.0677], sign_agree=3/5
- BWT: mean=+0.0174, 95% CI [-0.0126, +0.0653], sign_agree=3/5
- Jaccard: mean=+0.0648, 95% CI [+0.0353, +0.0943], sign_agree=5/5
- action-KL: mean=+0.0527, 95% CI [+0.0315, +0.0871], sign_agree=5/5

### reverse, K=2, `distill-seqft` (exploratory)
- AA: mean=+0.0426, 95% CI [+0.0183, +0.0670], sign_agree=5/5
- Forgetting: mean=+0.0392, 95% CI [+0.0194, +0.0532], sign_agree=5/5
- BWT: mean=+0.0533, 95% CI [+0.0398, +0.0668], sign_agree=5/5
- Jaccard: mean=+0.1189, 95% CI [+0.0831, +0.1547], sign_agree=5/5
- action-KL: mean=+0.0926, 95% CI [+0.0826, +0.1038], sign_agree=5/5

### reverse, K=2, `ours-distill` (confirmatory)
- AA: mean=-0.0123, 95% CI [-0.0399, +0.0082], sign_agree=3/5
- Forgetting: mean=-0.0154, 95% CI [-0.0526, +0.0124], sign_agree=3/5
- BWT: mean=-0.0097, 95% CI [-0.0459, +0.0194], sign_agree=3/5
- Jaccard: mean=-0.0294, 95% CI [-0.0814, +0.0194], sign_agree=4/5
- action-KL: mean=-0.0241, 95% CI [-0.0387, -0.0125], sign_agree=5/5

### reverse, K=2, `ours-replay` (confirmatory)
- AA: mean=+0.0092, 95% CI [-0.0067, +0.0242], sign_agree=3/5
- Forgetting: mean=+0.0049, 95% CI [-0.0112, +0.0221], sign_agree=3/5
- BWT: mean=+0.0077, 95% CI [-0.0107, +0.0261], sign_agree=3/5
- Jaccard: mean=+0.0354, 95% CI [+0.0021, +0.0690], sign_agree=4/5
- action-KL: mean=+0.0286, 95% CI [+0.0166, +0.0491], sign_agree=5/5

### reverse, K=2, `ours-seqft` (exploratory)
- AA: mean=+0.0303, 95% CI [-0.0065, +0.0672], sign_agree=3/5
- Forgetting: mean=+0.0239, 95% CI [-0.0096, +0.0473], sign_agree=4/5
- BWT: mean=+0.0436, 95% CI [+0.0024, +0.0764], sign_agree=4/5
- Jaccard: mean=+0.0895, 95% CI [+0.0511, +0.1382], sign_agree=5/5
- action-KL: mean=+0.0685, 95% CI [+0.0479, +0.0890], sign_agree=5/5

### reverse, K=2, `replay-seqft` (exploratory)
- AA: mean=+0.0211, 95% CI [-0.0184, +0.0605], sign_agree=4/5
- Forgetting: mean=+0.0190, 95% CI [-0.0290, +0.0530], sign_agree=4/5
- BWT: mean=+0.0359, 95% CI [-0.0172, +0.0699], sign_agree=4/5
- Jaccard: mean=+0.0541, 95% CI [+0.0297, +0.0773], sign_agree=5/5
- action-KL: mean=+0.0399, 95% CI [-0.0001, +0.0696], sign_agree=4/5

### reverse, K=4, `distill-replay` (confirmatory)
- AA: mean=+0.0235, 95% CI [+0.0001, +0.0464], sign_agree=3/5
- Forgetting: mean=+0.0060, 95% CI [-0.0160, +0.0333], sign_agree=3/5
- BWT: mean=+0.0291, 95% CI [+0.0178, +0.0389], sign_agree=5/5
- Jaccard: mean=-0.0012, 95% CI [-0.1017, +0.0932], sign_agree=3/5
- action-KL: mean=+0.0338, 95% CI [+0.0244, +0.0433], sign_agree=5/5

### reverse, K=4, `distill-seqft` (exploratory)
- AA: mean=+0.0428, 95% CI [+0.0041, +0.0815], sign_agree=4/5
- Forgetting: mean=+0.0299, 95% CI [-0.0220, +0.0818], sign_agree=3/5
- BWT: mean=+0.0384, 95% CI [-0.0134, +0.0862], sign_agree=4/5
- Jaccard: mean=+0.0956, 95% CI [+0.0683, +0.1200], sign_agree=5/5
- action-KL: mean=+0.0193, 95% CI [+0.0165, +0.0217], sign_agree=5/5

### reverse, K=4, `ours-distill` (confirmatory)
- AA: mean=-0.0048, 95% CI [-0.0276, +0.0187], sign_agree=3/5
- Forgetting: mean=+0.0045, 95% CI [-0.0145, +0.0199], sign_agree=3/5
- BWT: mean=-0.0086, 95% CI [-0.0232, +0.0056], sign_agree=3/5
- Jaccard: mean=-0.0071, 95% CI [-0.0536, +0.0596], sign_agree=3/5
- action-KL: mean=-0.0090, 95% CI [-0.0124, -0.0056], sign_agree=5/5

### reverse, K=4, `ours-replay` (confirmatory)
- AA: mean=+0.0187, 95% CI [+0.0048, +0.0326], sign_agree=4/5
- Forgetting: mean=+0.0105, 95% CI [+0.0039, +0.0187], sign_agree=5/5
- BWT: mean=+0.0205, 95% CI [+0.0095, +0.0315], sign_agree=5/5
- Jaccard: mean=-0.0082, 95% CI [-0.0633, +0.0413], sign_agree=3/5
- action-KL: mean=+0.0248, 95% CI [+0.0151, +0.0345], sign_agree=5/5

### reverse, K=4, `ours-seqft` (exploratory)
- AA: mean=+0.0380, 95% CI [-0.0068, +0.0686], sign_agree=4/5
- Forgetting: mean=+0.0344, 95% CI [-0.0069, +0.0785], sign_agree=3/5
- BWT: mean=+0.0299, 95% CI [-0.0350, +0.0819], sign_agree=4/5
- Jaccard: mean=+0.0885, 95% CI [+0.0466, +0.1304], sign_agree=5/5
- action-KL: mean=+0.0103, 95% CI [+0.0038, +0.0161], sign_agree=4/5

### reverse, K=4, `replay-seqft` (exploratory)
- AA: mean=+0.0193, 95% CI [-0.0191, +0.0563], sign_agree=4/5
- Forgetting: mean=+0.0239, 95% CI [-0.0129, +0.0697], sign_agree=3/5
- BWT: mean=+0.0094, 95% CI [-0.0533, +0.0669], sign_agree=3/5
- Jaccard: mean=+0.0967, 95% CI [+0.0119, +0.1816], sign_agree=4/5
- action-KL: mean=-0.0145, 95% CI [-0.0246, -0.0045], sign_agree=4/5

