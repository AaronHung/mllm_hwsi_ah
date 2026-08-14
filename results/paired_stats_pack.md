# WP1 paired statistics pack

v0.292 zero-compute analysis of frozen per-seed CSVs. Positive `mean` means the first method is better for the metric direction.

## Multiple-comparison policy

One policy is used for the entire report: Benjamini–Hochberg FDR control at q=0.05 over every available comparison × order × K × metric hypothesis. Bootstrap intervals are descriptive 95% CIs; claim-level significance requires the adjusted q-value and a CI that points in the same direction. No metric is selected after seeing the results.

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

### main, K=1, `distill-replay`
- AA: mean=-0.0009, 95% CI [-0.0118, +0.0120], p=0.8750, q=0.9292, n=5
- Forgetting: mean=+0.0014, 95% CI [-0.0051, +0.0067], p=0.6250, q=0.7979, n=5
- BWT: mean=+0.0033, 95% CI [-0.0090, +0.0156], p=0.6875, q=0.8418, n=5
- Jaccard: mean=+0.1162, 95% CI [+0.0844, +0.1479], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0453, 95% CI [+0.0370, +0.0530], p=0.0625, q=0.1875, n=5

### main, K=1, `lambda3-ours`
- AA: mean=+0.0151, 95% CI [+0.0056, +0.0246], p=0.1250, q=0.3409, n=5
- Forgetting: mean=-0.0023, 95% CI [-0.0081, +0.0062], p=0.6250, q=0.7979, n=5
- BWT: mean=+0.0072, 95% CI [-0.0084, +0.0199], p=0.5000, q=0.7595, n=5
- Jaccard: mean=+0.0878, 95% CI [+0.0476, +0.1106], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0220, 95% CI [+0.0162, +0.0294], p=0.0625, q=0.1875, n=5

### main, K=1, `mem2048-ours`
- AA: mean=+0.0102, 95% CI [+0.0047, +0.0166], p=0.0625, q=0.1875, n=5
- Forgetting: mean=-0.0008, 95% CI [-0.0102, +0.0136], p=0.8750, q=0.9292, n=5
- BWT: mean=+0.0041, 95% CI [-0.0106, +0.0188], p=0.6250, q=0.7979, n=5
- Jaccard: mean=+0.0200, 95% CI [-0.0235, +0.0577], p=0.3750, q=0.6923, n=5
- action-KL: mean=+0.0036, 95% CI [-0.0021, +0.0098], p=0.3750, q=0.6923, n=5

### main, K=1, `ours-distill`
- AA: mean=-0.0039, 95% CI [-0.0134, +0.0041], p=0.5625, q=0.7979, n=5
- Forgetting: mean=+0.0038, 95% CI [-0.0038, +0.0122], p=0.6250, q=0.7979, n=5
- BWT: mean=-0.0051, 95% CI [-0.0100, +0.0009], p=0.1875, q=0.4688, n=5
- Jaccard: mean=-0.0368, 95% CI [-0.0876, +0.0269], p=0.3125, q=0.6250, n=5
- action-KL: mean=-0.0109, 95% CI [-0.0188, -0.0043], p=0.0625, q=0.1875, n=5

### main, K=1, `ours-ours_uniform`
- AA: mean=-0.0158, 95% CI [-0.0203, -0.0114], p=0.0625, q=0.1875, n=5
- Forgetting: mean=+0.0008, 95% CI [-0.0101, +0.0152], p=1.0000, q=1.0000, n=5
- BWT: mean=-0.0096, 95% CI [-0.0214, +0.0009], p=0.3125, q=0.6250, n=5
- Jaccard: mean=+0.0057, 95% CI [-0.0196, +0.0259], p=0.7500, q=0.8824, n=5
- action-KL: mean=-0.0013, 95% CI [-0.0038, +0.0004], p=0.3125, q=0.6250, n=5

### main, K=1, `ours-replay`
- AA: mean=-0.0048, 95% CI [-0.0109, +0.0004], p=0.2500, q=0.5556, n=5
- Forgetting: mean=+0.0052, 95% CI [-0.0057, +0.0160], p=0.4375, q=0.7394, n=5
- BWT: mean=-0.0018, 95% CI [-0.0170, +0.0149], p=0.8750, q=0.9292, n=5
- Jaccard: mean=+0.0794, 95% CI [+0.0520, +0.1148], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0344, 95% CI [+0.0275, +0.0414], p=0.0625, q=0.1875, n=5

### main, K=2, `distill-replay`
- AA: mean=+0.0040, 95% CI [-0.0037, +0.0116], p=0.4375, q=0.7394, n=5
- Forgetting: mean=+0.0041, 95% CI [-0.0069, +0.0152], p=0.6250, q=0.7979, n=5
- BWT: mean=+0.0075, 95% CI [-0.0084, +0.0234], p=0.5000, q=0.7595, n=5
- Jaccard: mean=+0.0522, 95% CI [+0.0345, +0.0712], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0349, 95% CI [+0.0288, +0.0423], p=0.0625, q=0.1875, n=5

### main, K=2, `lambda3-ours`
- AA: mean=+0.0022, 95% CI [-0.0115, +0.0162], p=0.6250, q=0.7979, n=5
- Forgetting: mean=-0.0056, 95% CI [-0.0204, +0.0108], p=0.6250, q=0.7979, n=5
- BWT: mean=+0.0004, 95% CI [-0.0212, +0.0221], p=1.0000, q=1.0000, n=5
- Jaccard: mean=+0.0572, 95% CI [+0.0410, +0.0705], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0122, 95% CI [+0.0107, +0.0138], p=0.0625, q=0.1875, n=5

### main, K=2, `mem2048-ours`
- AA: mean=+0.0098, 95% CI [+0.0018, +0.0188], p=0.0625, q=0.1875, n=5
- Forgetting: mean=+0.0104, 95% CI [+0.0026, +0.0183], p=0.1250, q=0.3409, n=5
- BWT: mean=+0.0138, 95% CI [+0.0071, +0.0207], p=0.0625, q=0.1875, n=5
- Jaccard: mean=+0.0453, 95% CI [+0.0318, +0.0611], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0074, 95% CI [+0.0063, +0.0089], p=0.0625, q=0.1875, n=5

### main, K=2, `ours-distill`
- AA: mean=-0.0047, 95% CI [-0.0196, +0.0092], p=0.6250, q=0.7979, n=5
- Forgetting: mean=-0.0003, 95% CI [-0.0118, +0.0120], p=1.0000, q=1.0000, n=5
- BWT: mean=-0.0024, 95% CI [-0.0162, +0.0169], p=0.8125, q=0.9112, n=5
- Jaccard: mean=-0.0368, 95% CI [-0.0528, -0.0204], p=0.0625, q=0.1875, n=5
- action-KL: mean=-0.0080, 95% CI [-0.0113, -0.0053], p=0.0625, q=0.1875, n=5

### main, K=2, `ours-ours_uniform`
- AA: mean=+0.0099, 95% CI [-0.0054, +0.0252], p=0.4375, q=0.7394, n=5
- Forgetting: mean=+0.0229, 95% CI [+0.0042, +0.0416], p=0.2500, q=0.5556, n=5
- BWT: mean=+0.0201, 95% CI [+0.0088, +0.0315], p=0.0625, q=0.1875, n=5
- Jaccard: mean=-0.0021, 95% CI [-0.0175, +0.0106], p=0.8125, q=0.9112, n=5
- action-KL: mean=+0.0024, 95% CI [+0.0014, +0.0034], p=0.0625, q=0.1875, n=5

### main, K=2, `ours-replay`
- AA: mean=-0.0007, 95% CI [-0.0202, +0.0142], p=0.8750, q=0.9292, n=5
- Forgetting: mean=+0.0039, 95% CI [-0.0151, +0.0173], p=0.6250, q=0.7979, n=5
- BWT: mean=+0.0051, 95% CI [-0.0122, +0.0191], p=0.5625, q=0.7979, n=5
- Jaccard: mean=+0.0154, 95% CI [-0.0011, +0.0348], p=0.3125, q=0.6250, n=5
- action-KL: mean=+0.0269, 95% CI [+0.0215, +0.0322], p=0.0625, q=0.1875, n=5

### main, K=4, `distill-replay`
- AA: mean=+0.0101, 95% CI [-0.0099, +0.0297], p=0.4375, q=0.7394, n=5
- Forgetting: mean=+0.0044, 95% CI [-0.0213, +0.0274], p=0.6875, q=0.8418, n=5
- BWT: mean=+0.0013, 95% CI [-0.0270, +0.0295], p=0.9375, q=0.9783, n=5
- Jaccard: mean=+0.0315, 95% CI [-0.0190, +0.0847], p=0.3125, q=0.6250, n=5
- action-KL: mean=+0.0238, 95% CI [+0.0197, +0.0276], p=0.0625, q=0.1875, n=5

### main, K=4, `ours-distill`
- AA: mean=+0.0018, 95% CI [-0.0113, +0.0141], p=0.8125, q=0.9112, n=5
- Forgetting: mean=+0.0112, 95% CI [-0.0122, +0.0347], p=0.5000, q=0.7595, n=5
- BWT: mean=+0.0101, 95% CI [-0.0130, +0.0338], p=0.5000, q=0.7595, n=5
- Jaccard: mean=-0.0234, 95% CI [-0.0474, -0.0002], p=0.1875, q=0.4688, n=5
- action-KL: mean=-0.0060, 95% CI [-0.0073, -0.0047], p=0.0625, q=0.1875, n=5

### main, K=4, `ours-replay`
- AA: mean=+0.0119, 95% CI [-0.0024, +0.0252], p=0.2500, q=0.5556, n=5
- Forgetting: mean=+0.0155, 95% CI [+0.0061, +0.0267], p=0.0625, q=0.1875, n=5
- BWT: mean=+0.0113, 95% CI [-0.0115, +0.0324], p=0.4375, q=0.7394, n=5
- Jaccard: mean=+0.0081, 95% CI [-0.0290, +0.0391], p=0.7500, q=0.8824, n=5
- action-KL: mean=+0.0178, 95% CI [+0.0146, +0.0209], p=0.0625, q=0.1875, n=5

### reverse, K=1, `distill-replay`
- AA: mean=+0.0187, 95% CI [-0.0027, +0.0525], p=0.3750, q=0.6923, n=5
- Forgetting: mean=+0.0239, 95% CI [-0.0076, +0.0710], p=0.5000, q=0.7595, n=5
- BWT: mean=+0.0236, 95% CI [-0.0081, +0.0720], p=0.5000, q=0.7595, n=5
- Jaccard: mean=+0.0901, 95% CI [+0.0664, +0.1138], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0559, 95% CI [+0.0285, +0.1017], p=0.0625, q=0.1875, n=5

### reverse, K=1, `ours-distill`
- AA: mean=-0.0113, 95% CI [-0.0387, +0.0057], p=0.7500, q=0.8824, n=5
- Forgetting: mean=-0.0207, 95% CI [-0.0559, +0.0012], p=0.1875, q=0.4688, n=5
- BWT: mean=-0.0183, 95% CI [-0.0549, +0.0038], p=0.5000, q=0.7595, n=5
- Jaccard: mean=-0.0228, 95% CI [-0.0519, +0.0093], p=0.2500, q=0.5556, n=5
- action-KL: mean=-0.0106, 95% CI [-0.0242, -0.0010], p=0.1250, q=0.3409, n=5

### reverse, K=1, `ours-replay`
- AA: mean=+0.0074, 95% CI [-0.0024, +0.0171], p=0.3750, q=0.6923, n=5
- Forgetting: mean=+0.0032, 95% CI [-0.0073, +0.0163], p=0.8125, q=0.9112, n=5
- BWT: mean=+0.0053, 95% CI [-0.0073, +0.0183], p=0.5625, q=0.7979, n=5
- Jaccard: mean=+0.0673, 95% CI [+0.0302, +0.1076], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0453, 95% CI [+0.0242, +0.0779], p=0.0625, q=0.1875, n=5

### reverse, K=2, `distill-replay`
- AA: mean=+0.0215, 95% CI [-0.0015, +0.0582], p=0.2500, q=0.5556, n=5
- Forgetting: mean=+0.0202, 95% CI [-0.0134, +0.0696], p=0.6875, q=0.8418, n=5
- BWT: mean=+0.0174, 95% CI [-0.0126, +0.0663], p=0.8750, q=0.9292, n=5
- Jaccard: mean=+0.0648, 95% CI [+0.0353, +0.0937], p=0.0625, q=0.1875, n=5
- action-KL: mean=+0.0527, 95% CI [+0.0310, +0.0871], p=0.0625, q=0.1875, n=5

### reverse, K=2, `ours-distill`
- AA: mean=-0.0123, 95% CI [-0.0399, +0.0075], p=0.5625, q=0.7979, n=5
- Forgetting: mean=-0.0154, 95% CI [-0.0526, +0.0128], p=0.6250, q=0.7979, n=5
- BWT: mean=-0.0097, 95% CI [-0.0484, +0.0194], p=0.8125, q=0.9112, n=5
- Jaccard: mean=-0.0294, 95% CI [-0.0814, +0.0194], p=0.3125, q=0.6250, n=5
- action-KL: mean=-0.0241, 95% CI [-0.0387, -0.0125], p=0.0625, q=0.1875, n=5

### reverse, K=2, `ours-replay`
- AA: mean=+0.0092, 95% CI [-0.0067, +0.0242], p=0.3750, q=0.6923, n=5
- Forgetting: mean=+0.0049, 95% CI [-0.0123, +0.0221], p=0.6875, q=0.8418, n=5
- BWT: mean=+0.0077, 95% CI [-0.0107, +0.0261], p=0.5000, q=0.7595, n=5
- Jaccard: mean=+0.0354, 95% CI [+0.0021, +0.0690], p=0.1875, q=0.4688, n=5
- action-KL: mean=+0.0286, 95% CI [+0.0164, +0.0488], p=0.0625, q=0.1875, n=5

### reverse, K=4, `distill-replay`
- AA: mean=+0.0235, 95% CI [+0.0001, +0.0464], p=0.2500, q=0.5556, n=5
- Forgetting: mean=+0.0060, 95% CI [-0.0160, +0.0333], p=0.7500, q=0.8824, n=5
- BWT: mean=+0.0291, 95% CI [+0.0178, +0.0389], p=0.0625, q=0.1875, n=5
- Jaccard: mean=-0.0012, 95% CI [-0.1017, +0.0932], p=1.0000, q=1.0000, n=5
- action-KL: mean=+0.0338, 95% CI [+0.0244, +0.0433], p=0.0625, q=0.1875, n=5

### reverse, K=4, `ours-distill`
- AA: mean=-0.0048, 95% CI [-0.0292, +0.0187], p=0.9375, q=0.9783, n=5
- Forgetting: mean=+0.0045, 95% CI [-0.0145, +0.0199], p=0.6250, q=0.7979, n=5
- BWT: mean=-0.0086, 95% CI [-0.0240, +0.0056], p=0.4375, q=0.7394, n=5
- Jaccard: mean=-0.0071, 95% CI [-0.0536, +0.0596], p=1.0000, q=1.0000, n=5
- action-KL: mean=-0.0090, 95% CI [-0.0125, -0.0056], p=0.0625, q=0.1875, n=5

### reverse, K=4, `ours-replay`
- AA: mean=+0.0187, 95% CI [+0.0048, +0.0326], p=0.1250, q=0.3409, n=5
- Forgetting: mean=+0.0105, 95% CI [+0.0039, +0.0187], p=0.0625, q=0.1875, n=5
- BWT: mean=+0.0205, 95% CI [+0.0095, +0.0315], p=0.0625, q=0.1875, n=5
- Jaccard: mean=-0.0082, 95% CI [-0.0626, +0.0413], p=0.8750, q=0.9292, n=5
- action-KL: mean=+0.0248, 95% CI [+0.0150, +0.0345], p=0.0625, q=0.1875, n=5

