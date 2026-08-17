# E2 — `eq_pres_norep`: completing the C0 attribution square

**Track A:** frozen; writing is the only remaining work. **Track B/C:** Track B closed and archived, Track C0 screened and FAILed; this is the second of the two v0.35-closeout artifacts, after which all compute for this paper ends permanently.

Computed from `runs/v2/e2_norep_20260817T045648Z/cl_main_can_main_e2_norep_20260817T045648Z.csv` (6 units: `eq_pres_norep`, seeds {0,1,2} x K{1,2}, main order, Mac MPS). `distill` and the utility backfill are frozen comparators and were not rerun. **This is a descriptive attribution, not a gate**: no verdict is reopened and nothing here promotes anything.

## Read-out

| criterion | K | mean [seeds 0,1,2] | holds |
|---|---|---|---|
| Forgetting ≤ `distill` | 1 | -0.0148 [-0.0268, -0.0208, +0.0032] | yes |
| Forgetting ≤ `distill` | 2 | +0.0008 [+0.0208, -0.0049, -0.0136] | no |
| Δε-mass > 0 vs `distill` | 1 | +0.1191 [+0.1426, +0.1119, +0.1027] | yes |
| Δε-mass > 0 vs `distill` | 2 | +0.1365 [+0.1243, +0.1350, +0.1501] | yes |

### **`L_eq` alone NOT sufficient (descriptive)**

## The attribution square

| corner | replay imitation | `L_eq` | per-task adapter | status |
|---|---|---|---|---|
| `eq_pres` | yes | yes | no | frozen (Gate v2) |
| `eq_pres_norep` | no | yes | no | **this experiment** |
| `sp_nav_eq` | no | yes | yes | frozen (Track C0) |
| *(not run)* | yes | yes | yes | **never run** |

**The fourth corner was not run, so the two main effects are described from three corners rather than estimated from a complete 2x2.** Any statement about an interaction between replay composition and adapter architecture is therefore unavailable, and none is made here. This is a stated limit of the design, not an oversight: the closeout authorization fixes the config set and no additional arm is authorized under any outcome.

Reading the three corners that exist: comparing `eq_pres_norep` with `eq_pres` isolates the **replay imitation term** at fixed architecture; comparing `eq_pres_norep` with `sp_nav_eq` isolates the **adapter architecture** at fixed loss composition.

## Wording locks

Locks from the ratified v0.34 verdict apply: no cell-level movement is described as a repair or as a newly introduced failure, and per-task quanta are disclosed wherever plasticity is discussed. This report deliberately confines itself to `Forgetting` and the utility axis — the two quantities the read-out rule names — precisely because cell-level plasticity is not resolvable at three seeds.

This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable review**. **STOP** — after E1 and E2 all compute for this paper is over; the only remaining work is writing.

