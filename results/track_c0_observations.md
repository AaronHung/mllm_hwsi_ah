# Track C0 — post-verdict observations (descriptive; changes no verdict)

**Written 2026-08-17 10:40 CST, after `results/track_c0_verdict.md` was
computed and read.** That verdict file is the mechanical application of the
criteria frozen in `docs/track_c0.md` §7 and is **not modified by this
document**. Nothing here re-scores anything, applies any threshold, or argues
any FAIL away. Its purpose is to put three structural facts in front of the
three-way review, because they bear on *what the C0 result licenses one to
conclude* — which is the review's call, not this document's.

**C0 verdict, unchanged:** `sp_nav` **C0 FAIL** (c1, c2), `sp_nav_eq`
**C0 FAIL** (c1). Branch per §9: return to Track A. No C0.1, no router, no
confirmation runs.

**Oracle task identity** (`docs/track_c0.md` §1) applies to everything below:
the task index selects the active adapter at train and test time. This is a
best-case assumption and no C0 claim survives without it.

---

## 1. `esca` cannot test the C0 hypothesis, and it is one of the four c1 cells

`esca` is the **first** task in main order. At stage 1 there is no buffer and
no `old_nav`, so `sp_nav` and `sp_nav_eq` are the *same algorithm* — verified
directly: all 6 stage-1 rows are identical between the two configs, and the
c1 table shows the identical value `-0.0238 [-0.0357, -0.0670, +0.0312]` for
both.

What stage 1 actually measures is therefore **not** whether parameter
isolation protects plasticity — there is nothing yet to protect against — but
the cost of C0's own learning-rate split: on task 1 the C0 navigator moves
its 164,481-parameter core at `0.1 * lr` and puts full `lr` only on a
640-parameter adapter, while `seqft` moves the whole core at full `lr`. A
deficit there is a **design handicap of the slow-core choice**
(`docs/track_c0.md` §2.3), not evidence about the architectural hypothesis.

This matters for reading c1: one of its four cells is structurally incapable
of answering the question c1 exists to ask. The other three carry the
continual-learning content.

## 2. c1's threshold is finer than the measurement quantum on every task

The per-task measurement quantum is `1/n_test` — the accuracy change caused
by a single test slide flipping:

| task | `n_test` | quantum `1/n_test` | c1 threshold (0.01) expressed in quanta |
|---|---|---|---|
| `esca` | 15 | 0.0667 | 0.15 |
| `lung` | 95 | 0.0105 | 0.95 |
| `rcc` | 76 | 0.0132 | 0.76 |
| `brca` | 93 | 0.0108 | 0.93 |

On every task the `-0.01` threshold is **smaller than one test item**, and on
`esca` it is smaller than a *sixth* of one. A cell mean of `-0.0238` on
`esca` is about one third of a single slide changing its label.

This is the same under-power that Fable's ratification of the v0.34 verdict
required to be disclosed, recurring in C0 with the same cause: three seeds
and per-task test sets of 15-95 slides. The consequence for C0 specifically:
**c1 is an under-powered criterion, in both directions.** It cannot reliably
confirm a plasticity gain and it cannot reliably attribute a plasticity loss.
Per the wording locks, no individual cell movement in the C0 tables is
described as a repair or as a newly introduced failure.

Stating this does **not** weaken the verdict. The criterion was frozen before
training, it was applied as written, and C0 did not meet it. The
disclosure is that a *passing* c1 would have been weak evidence too — which
is a limitation of the screening design, and belongs in the paper's
limitations rather than in a post-hoc defence.

## 3. What the three later tasks show, and what `sp_nav_eq` bought

Setting `esca` aside for the reason in §1, the tasks that do carry continual
content:

| task | `sp_nav` ΔA[t,t] vs `seqft` | `sp_nav_eq` ΔA[t,t] vs `seqft` |
|---|---|---|
| `lung` | +0.0053 (PASS) | -0.0045 (PASS) |
| `rcc` | -0.0113 (FAIL) | -0.0014 (PASS) |
| `brca` | -0.0163 (FAIL) | -0.0515 (FAIL) |

`brca` is the last and hardest task, and it is negative for both configs —
most so for `sp_nav_eq`. Parameter isolation did not deliver
unconstrained-baseline plasticity on the task where the pressure is highest.
That is the honest reading of the C0 screening question, subject to §2.

On the other two axes `sp_nav_eq` behaves as Track B would predict:

- **c2 stability PASSES** (`-0.0106`, `-0.0053` vs `distill`), while `sp_nav`
  — which carries no CL loss at all — fails badly (`+0.0728`, `+0.0460`).
  Architecture alone does not buy retention; the consolidation term does.
- **c3 utility PASSES by the largest margin measured anywhere in this
  project**: `+0.1429` at K=1 and `+0.1807` at K=2 vs `distill`, 3/3 seeds,
  with tight bootstrap intervals — roughly 4x the effect `eq_pres` showed.
  `sp_nav`, which has no `L_eq`, is *negative* on the same axis
  (`-0.1163`, `-0.0414`).

**Interpretation caveat on c3, tied to a decision I made.** Per
`docs/track_c0.md` §6.1, `sp_nav_eq` carries `L_eq` and **no replay
imitation term**, so `L_eq` is its only memory signal and it optimizes
`eps_optimal_mass` more directly than `eq_pres` did (which also carried
`L_replay`). Part of the larger margin is plausibly that difference in
composition rather than the architecture. This was pre-registered before
training and is flagged here because it changes how much the c3 number
should be leaned on, not whether it passed.

## 4. Net reading offered to the review

The pattern that has now reproduced across a loss-level track and an
architecture-level track, with different mechanisms each time:

> Evidence-access utility can be preserved, and preserving it does not
> deliver capability or new-task acquisition.

C0 adds one thing Track B could not: the non-conversion survives **giving each
task its own parameters**, under the most favourable task-identity assumption
available. That is a stronger version of the same negative result, not a new
failure to explain.

What C0 does **not** establish, and must not be written as establishing:

- that a router-based or larger-capacity architecture would also fail — C0
  screened one minimal configuration under oracle task identity;
- that any individual cell was repaired or broken — see §2;
- that the slow-core learning-rate split is the reason for the c1 outcome —
  §1 shows it confounds `esca` specifically, and says nothing about `brca`.

**STOP.** This document is a Cursor analysis artifact awaiting joint
Aaron+Sol+Fable review, alongside `results/track_c0_verdict.md`. No C1 has
been designed, no router exists, and no further compute has been proposed.
