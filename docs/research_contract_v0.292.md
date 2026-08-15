# Research contract v0.292 under direction freeze v0.32

**Fable × Sol sign-off:** 2026-08-14  
**Repository:** `AaronHung/mllm_hwsi_ah`  
**Parent direction:** [HackMD v0.32](https://hackmd.io/@aaronh/rkqRxshLMl)

## Consensus

v0.32 freezes the paper as an analysis/protocol-centric CV methodology paper:
**Continual Budgeted Evidence Acquisition**. The contribution is the problem
formulation, causal attribution protocol, and three-level evaluation of
capability, acquisition behavior, and evidence utility. WSI is the benchmark
environment; `can_dataset` is the feature-space causal-pyramid benchmark and
`pilot40` is the minimal real-spatial validation.

v0.292 is the execution contract below. It does not reopen the direction, add
EUP, add query conditioning, or introduce a new CL loss.

## Immutable evidence

The following are read-only evidence and must not be recomputed or overwritten:

- Exp A/B and Gate 1;
- completed `can_dataset` main and reverse five-seed grids;
- completed utility, memory, and lambda ablations;
- frozen evaluators, counterfactual teacher, and causal access protocol.

New reports may read these files. New runs use fresh `runs/v2/<run_tag>/`
directories and never overwrite Protocol-v1 files.

## Paper-facing method names

Internal CSV keys remain unchanged for reproducibility:

- `distill`: old-policy / policy-fidelity distillation;
- `replay`: counterfactual-teacher replay;
- `ours_uniform`: replay + policy distillation, uniform loss weight;
- `ours`: **Utility-Weighted Replay Distillation**, a variant rather than a
  universal winner;
- `joint`: joint-training reference, not an upper bound.

The utility ablation is limited: `ours` and `ours_uniform` share the
utility-prioritized buffer truncation. It isolates utility weighting in the
distillation loss, not the full effect of utility-aware memory.

## Statistical policy

> **Superseded 2026-08-15 (v0.33.1):** the Benjamini–Hochberg policy below was
> the original v0.292 plan, but with `n=5` seeds the exact sign-flip floor is
> `p_min=2/2^5=0.0625 > 0.05`, so no comparison family can ever clear
> `q<=0.05` — this is a mathematical fact, not an artifact of family size.
> Current policy is purely descriptive: paired mean diff + 95% bootstrap CI +
> per-seed sign agreement, tagged `confirmatory`/`exploratory`. See
> `docs/handoff_fable_sol_20260815.md` §2.6 and `docs/method_gate_v033.md`.

WP1 uses paired seed-level bootstrap 95% CIs for every available comparison,
order, budget, and metric. The primary axes are capability (`AA`, `Forgetting`,
`BWT`), behavior (`Jaccard`, `action-KL`), and utility when available.

The single multiplicity policy is Benjamini–Hochberg FDR control at
`q = 0.05` over all reported comparison × order × budget × metric hypotheses.
Bootstrap intervals remain descriptive 95% intervals; claims use the adjusted
`q` value and the direction of the CI. No post-hoc selection of the most
favorable metric is allowed.

## Work packages and stop condition

1. **WP1, zero compute:** paired stats pack, both orders where available.
2. **WP2, zero compute:** implementation/fairness audit and reverse
   `K=1` distill standard-deviation check.
3. **WP3, zero compute:** own-time new-task plasticity `A[t,t]`; use
   “behavior-fidelity / capability-retention trade-off” unless degradation is
   observed.
4. **WP4, limited compute:** seed-0, `K=1`, main and reverse mechanism probe
   for `seqft`, `distill`, and `ours`; extend to `K=2` only if the `K=1`
   mechanism is stable.
5. **WP5, prepared but gated:** minimal pilot40 spatial validation with
   `seqft`, `distill`, `ours`, and joint-training reference, five seeds and
   `K={1,2,4}` on RunPod CUDA.

Stop after WP1, WP2, WP3, and the first WP4 mechanism report. Before the
8/21 science freeze, summarize whether any existing claim must be downgraded.
WP5 is not a reason to reopen the paper direction.

## Forbidden changes

- no EUP implementation or new CL loss;
- no query conditioning;
- no rerun of completed Protocol-v1 grids;
- no mixed backends within one results table;
- no use of `joint` as an “upper bound” claim;
- no statement that `ours` wins universally;
- no claim of a stability–plasticity trade-off before WP3 observes new-task
  plasticity degradation.
