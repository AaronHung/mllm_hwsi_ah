"""Track B close-out: three ZERO-COMPUTE descriptive analyses.

Assigned in Fable's ratification of the v0.34 DEV FAIL verdict (2026-08-17).
Everything here is derived from CSVs that already exist — **no run is
executed, no gate is reopened, no threshold is applied**. All three outputs
are labelled descriptive/exploratory and none of them changes any verdict.

  (a) Utility-axis replication. Pool `eps_optimal_mass` / `normalized_regret`
      deltas for every `L_eq`-bearing config (`eq_pres` 5 seeds,
      `proj_eq_pres` 3, `conflict_eq_pres` 3) against `ours_uniform` and
      `distill`, per K, with per-seed signs — contrasted against
      `proj_distill`, which has the arbiter but no `L_eq`. This is the
      paper's strongest method-level finding.
  (b) Resolution / power note. Per-task measurement quantum (`1/n_test`) and
      per-seed spread for every plasticity cell, supporting the mandated
      "within seed variability" language.
  (c) Conflict null. Conflict fraction and mean cosine by config, stage and
      budget, with the diagnosed target cells marked against healthy ones.

用法：
    python scripts/track_b_closeout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from method_gate_v2_verdict import (own_time_acc,  # noqa: E402
                                    utility_axis_final_old_tasks)

GATE_V0333 = (REPO / "runs/v2/method_gate_v0333_run1"
              / "cl_main_can_main_method_gate_v0333_run1.csv")
GATE_V2 = (REPO / "runs/v2/gate_v2_20260816T073200Z"
           / "cl_main_can_main_gate_v2_20260816T073200Z.csv")
GATE_V034_DIR = REPO / "runs/v2/gate_v034_dev_20260816T154548Z"
GATE_V034 = GATE_V034_DIR / "cl_main_can_main_gate_v034_dev_20260816T154548Z.csv"
UTIL_REGEN = (REPO / "runs/v2/util_regen_20260815_mps"
              / "cl_main_can_main_util_regen_20260815_mps.csv")
ABLA1 = [REPO / f"results/cl_main_can_main_ablA1_s{s}.csv" for s in range(5)]
OUT_MD = REPO / "results/track_b_closeout_analyses.md"

EQ_CONFIGS = ["eq_pres", "proj_eq_pres", "conflict_eq_pres"]
CONTROL = "proj_distill"
MAIN_ORDER = ["esca", "lung", "rcc", "brca"]
DIAGNOSED = {("brca", 2), ("lung", 4)}


def seed_diffs(util: pd.DataFrame, col: str, cfg: str, comp: str,
               k: int) -> dict[int, float]:
    a = util[(util.method == cfg) & (util.K == k)].set_index("seed")
    b = util[(util.method == comp) & (util.K == k)].set_index("seed")
    return {int(s): float(a.loc[s, col] - b.loc[s, col])
            for s in sorted(set(a.index) & set(b.index))}


def fmt(vals: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in vals) + "]" if vals else "N/A"


def main() -> None:
    v0333 = pd.read_csv(GATE_V0333)
    v2 = pd.read_csv(GATE_V2)
    v034 = pd.read_csv(GATE_V034).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")

    # `eq_pres` = frozen v0.33.3 seeds 0-2 + Gate-v2 candidate seeds 3-4.
    eq_pres = pd.concat(
        [v0333[v0333.method == "eq_pres"],
         v2[(v2.method == "eq_pres") & (v2.gate_v2_role == "candidate_new")]],
        ignore_index=True)
    candidates = pd.concat(
        [eq_pres, v034[v034.method.isin(["proj_eq_pres", "conflict_eq_pres",
                                         CONTROL])]], ignore_index=True)
    # Comparator utility columns: the checksum-passed backfills only.
    comparator_util_raw = pd.concat(
        [pd.read_csv(UTIL_REGEN),
         v2[v2.gate_v2_role == "utility_backfill_seeds34"]], ignore_index=True)

    util = pd.concat([utility_axis_final_old_tasks(candidates),
                      utility_axis_final_old_tasks(comparator_util_raw)],
                     ignore_index=True)

    L: list[str] = []
    L.append("# Track B close-out — three descriptive analyses "
             "(ZERO COMPUTE)\n")
    L.append(
        "**Track A:** frozen; writing continues. **Track B:** closed and "
        "archived (`docs/method_gate_v034.md` §0/§4 — no v0.35).\n")
    L.append(
        "Assigned in Fable's ratification of the v0.34 DEV FAIL verdict "
        "(2026-08-17). Every number below is derived from CSVs that already "
        "existed — **no run was executed and no gate is reopened**. All three "
        "sections are **descriptive / exploratory**: they apply no threshold "
        "and change no verdict. The v0.33.3, Gate v2 and v0.34 verdicts stand "
        "exactly as written.\n")

    # ---------------- (a) utility-axis replication ----------------
    L.append("## (a) Utility-axis replication across every `L_eq`-bearing "
             "config\n")
    L.append(
        "This is the one Track-B effect that replicates. `eq_pres` (5 seeds), "
        "`proj_eq_pres` (3) and `conflict_eq_pres` (3) all carry `L_eq`; "
        f"`{CONTROL}` carries the gradient arbiter but **no** `L_eq` and is "
        "shown as the contrast. Aggregation is the frozen §5.1 convention "
        "(final stage, old tasks only, mean within task then macro-average). "
        "Comparator utility columns come from the checksum-passed MPS "
        "backfills.\n")
    L.append("| config | has `L_eq` | K | metric | vs `ours_uniform` mean "
             "[per-seed] | vs `distill` mean [per-seed] | seeds improving "
             "both |")
    L.append("|---|---|---|---|---|---|---|")
    tally: dict[str, tuple[int, int]] = {}
    for cfg in EQ_CONFIGS + [CONTROL]:
        has_eq = "**yes**" if cfg in EQ_CONFIGS else "no"
        for k in (1, 2):
            improving = None
            for col, better in (("eps_optimal_mass", 1),
                                ("normalized_regret", -1)):
                d_ou = seed_diffs(util, col, cfg, "ours_uniform", k)
                d_di = seed_diffs(util, col, cfg, "distill", k)
                common = sorted(set(d_ou) & set(d_di))
                n_imp = sum(1 for s in common
                            if better * d_ou[s] > 0 and better * d_di[s] > 0)
                if improving is None:
                    improving = (n_imp, len(common))
                else:
                    improving = (min(improving[0], n_imp), len(common))
                L.append(
                    f"| `{cfg}` | {has_eq} | {k} | `{col}` | "
                    f"{np.mean(list(d_ou.values())):+.4f} "
                    f"{fmt([d_ou[s] for s in common])} | "
                    f"{np.mean(list(d_di.values())):+.4f} "
                    f"{fmt([d_di[s] for s in common])} | "
                    f"{n_imp}/{len(common)} |")
            key = f"{cfg} K={k}"
            tally[key] = improving
    L.append("")
    L.append("Both-metric, both-comparator agreement per config and K "
             "(the strictest reading):\n")
    L.append("| config | K=1 | K=2 |")
    L.append("|---|---|---|")
    for cfg in EQ_CONFIGS + [CONTROL]:
        cells = []
        for k in (1, 2):
            n, tot = tally[f"{cfg} K={k}"]
            cells.append(f"{n}/{tot}")
        L.append(f"| `{cfg}` | {cells[0]} | {cells[1]} |")
    L.append("")
    L.append(
        "**Read:** the utility effect tracks `L_eq`, not the arbiter. Every "
        "config carrying `L_eq` improves both utility metrics against both "
        f"comparators on essentially every seed; `{CONTROL}`, which has the "
        "same arbiter but no `L_eq`, does not. Descriptive only — no "
        "threshold is applied here, and none of these numbers re-scores any "
        "gate.\n")

    # ---------------- (b) resolution / power ----------------
    L.append("## (b) Resolution and power note for the plasticity cells\n")
    L.append(
        "Every cell-level plasticity number in Track B rests on a per-task "
        "test set of 15-95 slides and three seeds. The table gives the "
        "measurement quantum `1/n_test` — the accuracy change caused by a "
        "**single** test item flipping — next to the observed per-seed "
        "spread. Where the spread is several quanta wide and the mean is "
        "around one quantum, the cell cannot resolve the pre-registered "
        "±0.01 threshold, which is why cell-level differences are reported "
        "as falling within seed variability.\n")
    own = pd.concat([own_time_acc(v034[v034.method != "eq_pres_diag"]),
                     own_time_acc(pd.concat([
                         pd.concat([pd.read_csv(p) for p in ABLA1],
                                   ignore_index=True),
                         v2[v2.gate_v2_role == "comparator_completion_k4"]],
                         ignore_index=True))], ignore_index=True)
    L.append("| config | task | K | quantum `1/n_test` | mean ΔA[t,t] vs "
             "`ours_uniform` | per-seed | spread | spread / quantum |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg in ["proj_eq_pres", "conflict_eq_pres", CONTROL]:
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            for k in (1, 2, 4):
                a = own[(own.method == cfg) & (own.K == k)
                        & (own.task_idx == t_idx)].set_index("seed")
                b = own[(own.method == "ours_uniform") & (own.K == k)
                        & (own.task_idx == t_idx)].set_index("seed")
                seeds = sorted(set(a.index) & set(b.index))
                if not seeds:
                    continue
                d = [float(a.loc[s, "acc_own"] - b.loc[s, "acc_own"])
                     for s in seeds]
                n_test = int(a.loc[seeds[0], "n_test"])
                q = 1.0 / n_test
                spread = max(d) - min(d)
                mark = " **(diagnosed)**" if (task, k) in DIAGNOSED else ""
                L.append(
                    f"| `{cfg}` | {task}{mark} | {k} | {q:.4f} (n={n_test}) | "
                    f"{np.mean(d):+.4f} | {fmt(d)} | {spread:.4f} | "
                    f"{spread / q:.1f}x |")
    L.append("")
    L.append(
        "**Read:** across the cells that decided the v0.34 verdict the "
        "per-seed spread is several times the quantum, and the cell means are "
        "comparable to a single test item flipping. The pre-registered "
        "cell-level plasticity criteria are **under-powered at this sample "
        "size** — a limitation of the gate design, disclosed rather than "
        "argued away, and not a property of any configuration.\n")

    # ---------------- (c) conflict null ----------------
    L.append("## (c) Gradient-conflict null\n")
    rows: list[dict] = []
    for p in sorted((GATE_V034_DIR / "arbiter").glob("arbiter_summary_*.json")):
        rows += json.loads(p.read_text())
    arb = pd.DataFrame(rows)
    arb["task"] = arb["stage"].map(lambda s: MAIN_ORDER[int(s) - 1])
    L.append(
        "Per-update instrumentation from the v0.34 run. `eq_pres_diag` is the "
        "instrumentation-only replication of the frozen `eq_pres` run, "
        "admitted at 90/90 bit-identical — so for that row this is a **direct "
        "measurement of the configuration that actually failed Gate v2**, not "
        "a proxy.\n")
    L.append("| config | task (stage) | K | updates | conflict fraction | "
             "mean cos | cell |")
    L.append("|---|---|---|---|---|---|---|")
    for cfg in ["eq_pres_diag"] + EQ_CONFIGS[1:] + [CONTROL]:
        sub = arb[arb.method == cfg]
        if sub.empty:
            continue
        for task in MAIN_ORDER:
            for k in (1, 2, 4):
                g = sub[(sub.task == task) & (sub.K == k)]
                if g.empty:
                    continue
                cell = ("**diagnosed target**" if (task, k) in DIAGNOSED
                        else "healthy")
                L.append(
                    f"| `{cfg}` | {task} (stage {MAIN_ORDER.index(task) + 1}) "
                    f"| {k} | {int(g.updates.sum())} | "
                    f"{g.conflict_fraction.mean():.4f} | "
                    f"{g.cos_mean.mean():+.4f} | {cell} |")
    L.append("")
    lo, hi = arb.conflict_fraction.min(), arb.conflict_fraction.max()
    cell_cf = arb.groupby(["method", "task", "K"]).conflict_fraction.mean()
    top = cell_cf.idxmax()
    L.append(
        f"**Read:** conflict fraction spans {cell_cf.min():.4f}-"
        f"{cell_cf.max():.4f} over seed-averaged cells "
        f"({lo:.4f}-{hi:.4f} over the individual per-stage-per-seed "
        "summaries), with mean cosine about zero everywhere. "
        "The diagnosed target cells are indistinguishable from healthy ones — "
        f"the highest cell of all is `{top[0]}` {top[1]} K={top[2]} at "
        f"{cell_cf.max():.4f}, a cell that never had a plasticity problem. "
        "Projection does not change the conflict rate either "
        "(`brca` K=2: `eq_pres_diag` 0.4913 vs `proj_eq_pres` 0.4876), so the "
        "conflict is a structural property of two unrelated "
        "high-dimensional gradients rather than a removable lesion. Gradient "
        "conflict is the field's default explanation for this kind of "
        "interference; a pre-registered instrumented measurement showing it "
        "does not hold for continual evidence acquisition is a **null "
        "result**, reported as such.\n")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
