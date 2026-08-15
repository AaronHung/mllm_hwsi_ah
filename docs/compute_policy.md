# Compute policy — repo-level contract

**Status:** adopted 2026-08-15, triggered by the v0.33.3 backend-reproduction
audit (`docs/method_gate_v033.md` §5.2 / changelog). Applies to every
run-producing script in this repo from this date forward, not just the
method gate.

## Why this exists

While regenerating comparator utility metrics for the method gate
(`docs/method_gate_v033.md` §5.2), a probe-only rerun of `ours_uniform`/
`distill` on Mac **CPU** failed the pre-declared `|Δ| <= 0.001` checksum on
**12/12** (method, seed, K) rows against the frozen Protocol-v1 rows, with
discrepancies up to **Δ≈0.148** — roughly 150x tolerance (archived in
`runs/v2/util_regen_20260815_cpu_FAILED/`). The identical command on Mac
**MPS** reproduced all 12/12 rows **bit-exactly** (`results/util_regen_checksum.md`).

Conclusion: for this project's small, argmax-driven sequential
navigator simulation, **backend / execution path is a material
experimental-control variable**, not a rounding-noise nuisance. Three
independent CPU-only reruns matched each other exactly, so this is a
genuine CPU-vs-MPS floating-point-cascade effect, not general
non-determinism — but it means a result produced on one backend must never
be read as directly comparable to a result produced on another.

## Rules (MANDATORY)

1. **CPU is never used for formal model training, model regeneration, gate
   experiments, baseline reproduction, or any result-producing model
   evaluation.** CPU is permitted only for non-model post-processing:
   CSV aggregation, statistics (`scripts/paired_stats_pack.py`,
   `scripts/aggregate_results.py`), plotting, static code checks
   (`scripts/check_forbidden_phrases.py`), and file conversion.
2. **MPS is the default backend for all formal Track-A/Track-B runs.**
   Protocol-v1 was historically produced on Apple MPS (verified against
   `logs/cl_main_s*.log` / `logs/abl_ablA1_s*.log`, every line reads
   `device = mps`); any run whose purpose is to reproduce or backfill
   metrics for a frozen Protocol-v1 model **must** use MPS.
3. **CUDA is used only within an explicitly opened, backend-matched
   protocol** — e.g. the WP5 `pilot40` table (its own self-contained
   CUDA table; a dataset never previously run on MPS, so there is no
   cross-backend confound to introduce), or a future Protocol-v2 if the
   MPS timing decision rule (`docs/method_gate_v033.md` §C) ever forces a
   fallback.
4. **Never present a cross-backend delta as a method effect.** Every direct
   comparison used for a gate/pass/fail decision or a paper table row must
   have all of its compared arms produced on the *same* backend. (Frozen
   Protocol-v1 MPS rows may be reused as comparators for a new MPS run
   without rerunning them — that is same-backend, not cross-backend.)
5. **Every result-producing run must record:** device/backend, torch
   version, machine (hostname), git commit, seed, run tag, and the exact
   launch command. Implemented via `nav.device.run_provenance()`, merged
   into every `metadata.json`/`checkpoints.json` written by
   `scripts/cl_main.py` and `scripts/run_method_gate.py`.
6. **Pin the torch version until submission.** Bit-exact reproducibility
   (item 5.2's checksum, and this policy's whole premise) depends on a
   fixed `torch.__version__`; do not upgrade PyTorch mid-protocol. Current
   pinned version: see any recent `metadata.json`'s `torch_version` field
   (`requirements-nav.txt`).

## Quick reference

| Backend | Permitted use |
|---|---|
| CPU | aggregation / statistics / plotting / static checks / bookkeeping only |
| **MPS** | **default backend for all formal Track-A/Track-B model runs** |
| CUDA | only inside an explicitly opened backend-matched protocol (e.g. `pilot40`, or a declared Protocol-v2 fallback) |

## Cross-reference

- `docs/method_gate_v033.md` §5.2 — the checksum finding that triggered this
  policy, and the resulting MPS-first gate design (§B/§C of the v0.33.3
  amendment).
- `runs/v2/util_regen_20260815_cpu_FAILED/` — archived diagnostic evidence
  (failed CPU regen launch log + README).
- `results/util_regen_checksum.md` — the passing MPS regeneration checksum.
