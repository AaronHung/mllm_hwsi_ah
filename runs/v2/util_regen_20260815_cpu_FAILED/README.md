# util_regen 20260815 — Mac CPU attempt (FAILED checksum, archived as diagnostic evidence)

**Status: FAILED. Do not use for any table or comparator. Kept only as
backend-sensitivity evidence per v0.33.3 item A.**

## What this is

The v0.33.2 patch text (item 3) instructed a probe-only regeneration of
`ours_uniform` / `distill` (seeds {0,1,2}, K in {1,2}, main order) "ON MAC
CPU (same backend as the frozen rows)". This directory preserves the launch
log (`launch.log`, copied from `/tmp/util_regen_20260815.log`) of that CPU
run.

The full per-stage output CSV
(`runs/v2/util_regen_20260815/cl_main_can_main_util_regen_20260815.csv`,
see the `done ->` line in `launch.log`) was deleted during working-directory
cleanup before this archive was created and could not be recovered. Only
the per-unit summary printed to the launch log survives.

## Why it failed

The "same backend as the frozen rows" premise in the original patch text
was itself wrong: `logs/cl_main_s*.log` and `logs/abl_ablA1_s*.log` (the
actual Protocol-v1 launch logs) show `device = mps` on every line, not
`cpu`. Running the checksum script
(`scripts/verify_util_regen_checksum.py`) against this CPU regen's full CSV
found **12/12 (method, seed, K) rows failed** the pre-declared
`|Δ| <= 0.001` tolerance on AA/Forgetting/Jaccard/action-KL, with the
largest single-metric discrepancy reaching **Δ≈0.148** — roughly 150x the
tolerance.

The identical command, run on Mac **MPS** instead
(`runs/v2/util_regen_20260815_mps/`), reproduced all 12/12 rows
**bit-exactly** (`Δ=0.0000`; see `results/util_regen_checksum.md`).

## Eyeball comparison (final-task T1 row, from the two launch logs)

Full precision deltas are not recoverable (this CPU CSV is gone), but even
the 3-decimal numbers printed inline show the two backends diverging from
the very first unit:

| method | seed | K | CPU AA | MPS AA | CPU T1 kl | MPS T1 kl |
|---|---|---|---|---|---|---|
| ours_uniform | 0 | 1 | 0.907 | 0.866 | 0.070 | 0.077 |
| ours_uniform | 0 | 2 | 0.874 | 0.895 | 0.036 | 0.033 |
| distill | 0 | 1 | 0.896 | 0.867 | 0.095 | 0.059 |
| distill | 0 | 2 | 0.908 | 0.914 | 0.024 | 0.021 |
| ours_uniform | 1 | 1 | 0.862 | 0.895 | 0.088 | 0.103 |
| distill | 2 | 1 | 0.878 | 0.892 | 0.092 | 0.096 |

(Full table: compare `launch.log` in this directory against
`../util_regen_20260815_mps/`'s log; the surviving `Δ=0.148` figure was
computed by `verify_util_regen_checksum.py` on the full per-stage CSVs,
not on these rounded summary lines.)

## Conclusion drawn from this evidence

Backend / execution-path (CPU vs. MPS vs., presumably, CUDA) is a
**material protocol variable** for this argmax-driven sequential
navigator simulation — not a rounding-noise nuisance. This finding,
surfaced *before* gate unblinding, triggered the v0.33.2 CPU→MPS
provenance correction and the v0.33.3 MPS-first backend design
(`docs/method_gate_v033.md` §5.2 / changelog; `docs/compute_policy.md`).
