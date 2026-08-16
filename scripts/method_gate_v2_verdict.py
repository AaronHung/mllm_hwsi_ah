"""Compute the Gate v2 (`docs/method_gate_v033.md` §10) verdict for `eq_pres`
only, pre-registered BEFORE this script was run against the completed v2 CSV.

Inputs (all frozen / already on disk except the v2 run, nothing re-executed
here beyond the checksum re-derivation for the utility-axis backfill):
  - runs/v2/method_gate_v0333_run1/...csv   (frozen v0.33.3 gate: eq_pres
    seeds {0,1,2} x K{1,2,4} — NOT rerun, used as-is, §10 non-retroactivity)
  - runs/v2/<v2-tag>/cl_main_can_main_<v2-tag>.csv   (this script's only new
    input: 19 units per §10.2 — eq_pres seeds{3,4}xK{1,2,4}; ours_uniform
    K=4 seeds{0..4}; distill/ours_uniform seeds{3,4}xK{1,2} utility backfill)
  - results/cl_main_can_main_full_s{0..4}.csv    (frozen; distill, replay,
    already 5-seed x K{1,2,4} complete, no v2 compute needed)
  - results/cl_main_can_main_ablA1_s{0..4}.csv   (frozen; ours_uniform,
    K{1,2} only, already 5-seed complete for those two K)
  - runs/v2/util_regen_20260815_mps/...csv   (v0.33.2 §5.2 checksum-passed
    utility-axis backfill for distill/ours_uniform, seeds {0,1,2}, K{1,2})

Utility-axis backfill discipline (§10.2 item 3): the v2 CSV's
`distill`/`ours_uniform` seeds{3,4} rows are NOT trusted as g2 comparators
until they pass the same `|Δ|<=0.001` checksum against the frozen
`full_s{3,4}.csv`/`ablA1_s{3,4}.csv` rows used everywhere else in this
project (`verify_util_regen_checksum.load_frozen`/checksum logic, reused
verbatim here, not reimplemented).

Metric definitions reuse `per_run_metrics`/`boot_ci_paired` from
`scripts/aggregate_results.py` verbatim, exactly like the v0.33.3 verdict
script, so this cannot silently drift from the rest of the analysis
pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from aggregate_results import boot_ci_paired, per_run_metrics  # noqa: E402
from verify_util_regen_checksum import (CHECK_COLS, DEFAULT_TOL,  # noqa: E402
                                        METHODS as UTIL_METHODS,
                                        load_frozen as util_load_frozen)

GATE_V0333_CSV = REPO / "runs/v2/method_gate_v0333_run1/cl_main_can_main_method_gate_v0333_run1.csv"
FULL_CSVS = [REPO / f"results/cl_main_can_main_full_s{s}.csv" for s in range(5)]
ABLA1_CSVS = [REPO / f"results/cl_main_can_main_ablA1_s{s}.csv" for s in range(5)]
UTIL_REGEN_OLD_CSV = REPO / "runs/v2/util_regen_20260815_mps/cl_main_can_main_util_regen_20260815_mps.csv"
OUT_MD = REPO / "results/method_gate_v2_verdict.md"
OUT_CHECKSUM_MD = REPO / "results/util_regen_checksum_v2_seeds34.md"

CFG = "eq_pres"
OLD_SEEDS = [0, 1, 2]
NEW_SEEDS = [3, 4]
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
KS = [1, 2, 4]
EPS_TOL_G3 = 0.005
EPS_TOL_G4 = -0.01


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_v2(v2_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(v2_csv)
    dup_cols = ["method", "seed", "K", "stage", "task_idx"]
    return df.drop_duplicates(subset=dup_cols, keep="last")


def utility_axis_final_old_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """final stage, old tasks only, mean-within-task (row-level already) then
    macro-average across old tasks -> one (method,seed,K) row. Doc §5.1/§10.1."""
    if df.empty:
        return pd.DataFrame(columns=["method", "seed", "K", "eps_optimal_mass",
                                      "normalized_regret"])
    rows = []
    for (m, s, k), g in df.groupby(["method", "seed", "K"]):
        n_stage = g.stage.max()
        old = g[(g.stage == n_stage) & (g.task_idx < n_stage)]
        if old.empty:
            continue
        rows.append(dict(method=m, seed=s, K=k,
                         eps_optimal_mass=old.eps_optimal_mass.mean(),
                         normalized_regret=old.normalized_regret.mean()))
    return pd.DataFrame(rows)


def own_time_acc(df: pd.DataFrame) -> pd.DataFrame:
    """A[t,t] = bal_acc at stage==task_idx, per (method,seed,K,task)."""
    if df.empty:
        return pd.DataFrame(columns=["method", "seed", "K", "task_idx", "task",
                                      "acc_own", "n_test"])
    rows = []
    for (m, s, k), g in df.groupby(["method", "seed", "K"]):
        for j, gj in g.groupby("task_idx"):
            own = gj[gj.stage == j]
            if own.empty:
                continue
            rows.append(dict(method=m, seed=s, K=k, task_idx=j,
                             task=own.task.iloc[0], acc_own=own.bal_acc.iloc[0],
                             n_test=int(own.n_test.iloc[0])))
    return pd.DataFrame(rows)


def paired_diffs(df: pd.DataFrame, col: str, method_a: str, method_b: str,
                 k: int, seeds: list[int]) -> dict[int, float]:
    """seed -> (a - b) for col, at K=k, only for seeds present in both."""
    a = df[(df.method == method_a) & (df.K == k)].set_index("seed")
    b = df[(df.method == method_b) & (df.K == k)].set_index("seed")
    out = {}
    for s in seeds:
        if s in a.index and s in b.index:
            out[s] = float(a.loc[s, col] - b.loc[s, col])
    return out


def three_way(diffs: dict[int, float]) -> tuple[list[float], list[float], list[float]]:
    """Split a {seed: diff} dict into (old3, new2, pooled5) diff lists,
    each possibly shorter if a seed is missing."""
    old = [diffs[s] for s in OLD_SEEDS if s in diffs]
    new = [diffs[s] for s in NEW_SEEDS if s in diffs]
    pooled = old + new
    return old, new, pooled


def fmt_list(vals: list[float]) -> str:
    if not vals:
        return "N/A"
    return "[" + ", ".join(f"{v:+.4f}" for v in vals) + "]"


def fmt_mean(vals: list[float]) -> str:
    if not vals:
        return "N/A"
    return f"{np.mean(vals):+.4f}"


def run_utility_checksum(v2: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Checksum the v2 CSV's distill/ours_uniform seeds{3,4} K{1,2} backfill
    rows against frozen full_s{3,4}.csv / ablA1_s{3,4}.csv, exactly like
    v0.33.2 §5.2 / verify_util_regen_checksum.py. Returns (admitted utility
    rows, report lines)."""
    regen = v2[v2.method.isin(UTIL_METHODS) & v2.K.isin([1, 2])
              & v2.seed.isin(NEW_SEEDS)]
    merge_cols = ["method", "seed", "K"]
    empty_cols = merge_cols + [f"{c}_frozen" for c in CHECK_COLS] + \
        [f"{c}_regen" for c in CHECK_COLS]
    if regen.empty:
        merged = pd.DataFrame(columns=empty_cols)
    else:
        frozen = util_load_frozen("can", "main")
        frozen_m = per_run_metrics(frozen)
        regen_m = per_run_metrics(regen)
        if frozen_m.empty or regen_m.empty:
            merged = pd.DataFrame(columns=empty_cols)
        else:
            merged = frozen_m.merge(regen_m, on=merge_cols,
                                    suffixes=("_frozen", "_regen"))

    lines = ["### Checksum: seeds{3,4} utility-axis backfill vs frozen rows "
            "(§10.2 item 3, same method as v0.33.2 §5.2)\n",
            f"Tolerance: `|Δ| <= {DEFAULT_TOL}` on {', '.join(CHECK_COLS)}, "
            "per (method, seed, K).\n",
            "| method | seed | K | " + " | ".join(f"Δ{c}" for c in CHECK_COLS)
            + " | pass |",
            "|---|---|---|" + "---|" * (len(CHECK_COLS) + 1)]
    admitted: list[tuple] = []
    for _, row in merged.sort_values(["method", "seed", "K"]).iterrows():
        deltas = [abs(row[f"{c}_frozen"] - row[f"{c}_regen"]) for c in CHECK_COLS]
        ok = all(d <= DEFAULT_TOL for d in deltas)
        if ok:
            admitted.append((row.method, row.seed, row.K))
        lines.append(
            f"| `{row.method}` | {row.seed} | {row.K} | "
            + " | ".join(f"{d:.4f}" for d in deltas)
            + f" | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append(f"**{len(admitted)}/{len(merged)} (method,seed,K) rows admitted** "
                 "as new2 utility comparators; any FAIL row is excluded from "
                 "g2 below, not silently passed.\n")

    util = utility_axis_final_old_tasks(regen)
    if not util.empty:
        util = util[util.apply(
            lambda r: (r.method, r.seed, r.K) in admitted, axis=1)]
    return util, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-csv", type=Path, required=True)
    args = ap.parse_args()
    args.v2_csv = args.v2_csv.resolve()

    gate_v0333 = pd.read_csv(GATE_V0333_CSV)
    gate_v0333 = gate_v0333[gate_v0333.method == CFG]  # only eq_pres needed
    v2 = load_v2(args.v2_csv)
    full = load_concat(FULL_CSVS)
    abla1 = load_concat(ABLA1_CSVS)
    util_regen_old = pd.read_csv(UTIL_REGEN_OLD_CSV)

    # ---- checksum the seeds{3,4} utility backfill first (gates g2 below) ----
    util_new_admitted, checksum_lines = run_utility_checksum(v2)
    OUT_CHECKSUM_MD.write_text("\n".join(checksum_lines) + "\n")
    print(f"wrote {OUT_CHECKSUM_MD.relative_to(REPO)}")

    # ---- assemble pooled capability tables ----
    eq_pres_raw = pd.concat(
        [gate_v0333, v2[(v2.method == CFG) & (v2.seed.isin(NEW_SEEDS))]],
        ignore_index=True)
    ours_u_raw = pd.concat(
        [abla1[abla1.method == "ours_uniform"],
         v2[(v2.method == "ours_uniform") & (v2.K == 4)
            & (v2.gate_v2_role == "comparator_completion_k4")]],
        ignore_index=True)
    distill_raw = full[full.method == "distill"]
    replay_raw = full[full.method == "replay"]

    metrics_all = pd.concat(
        [per_run_metrics(eq_pres_raw), per_run_metrics(ours_u_raw),
         per_run_metrics(distill_raw), per_run_metrics(replay_raw)],
        ignore_index=True)

    # utility axis: eq_pres always available (inline for every method);
    # distill/ours_uniform only at K in {1,2}, old (checksum-passed already
    # in v0.33.2) + new (checksum-passed just above)
    eq_pres_util = utility_axis_final_old_tasks(eq_pres_raw)
    comparator_util_old = utility_axis_final_old_tasks(util_regen_old)
    comparator_util = pd.concat([comparator_util_old, util_new_admitted],
                                ignore_index=True)
    util_all = pd.concat([eq_pres_util, comparator_util], ignore_index=True)

    eq_pres_own = own_time_acc(eq_pres_raw)
    ours_u_own = own_time_acc(ours_u_raw)

    lines: list[str] = []
    lines.append("# Method Gate v2 — `eq_pres`-only Extension Verdict\n")
    lines.append(
        "**Track A:** 8/21 freeze conditions all met 2026-08-16 (pilot40 R6 "
        "closed). **Track B:** v0.33.3 verdict (all 3 configs FAIL) is "
        "**frozen and unchanged** — this document evaluates ONLY the "
        "pre-registered Gate v2 extension (`docs/method_gate_v033.md` §10) "
        "for `eq_pres`. `ia_samp`/`ia_ep` are RETIRED, not evaluated here.\n")
    lines.append(
        f"Computed by `scripts/method_gate_v2_verdict.py` against "
        f"`{args.v2_csv.relative_to(REPO)}` (19 new units), the frozen "
        "v0.33.3 `eq_pres` seeds{0,1,2} rows (NOT rerun), the frozen "
        "Protocol-v1 comparator CSVs (`full_s{0..4}`, `ablA1_s{0..4}`), and "
        "the checksum-verified utility-axis backfill (old: v0.33.2 §5.2 "
        "seeds{0,1,2}; new: this run's seeds{3,4}, checksummed above). No "
        "thresholds or formulas were chosen after seeing this data — all "
        "definitions are copied from §10.1 (frozen before this script "
        "ran).\n")
    lines.append(
        "Every criterion is reported **old3** (seeds 0-2, unchanged from "
        "v0.33.3) / **new2** (seeds 3-4 alone) / **pooled5** (all 5) — "
        "pooled5 is primary for the pass/fail call, old3/new2 are secondary "
        "evidence, always shown (§10.4). 95% bootstrap CIs (`n_boot=10000`) "
        "are reported as **uncertainty riders only**, never as a pass/fail "
        "test.\n")

    # ================= g1: capability (pooled mean-only) =================
    lines.append("## g1 — Capability: pooled 5-seed paired mean Forgetting ≤ 0 vs `ours_uniform` AND `distill`\n")
    lines.append(
        "Rule (§10.1): mean-only, no 3/3 seed-consistency clause (replaced "
        "by the §10.3 replication rider, checked separately below).\n")
    lines.append("| K | comparator | old3 mean [seeds0-2] | new2 mean [seeds3-4] | pooled5 mean [all] | pooled5 95% CI | g1 (pooled) |")
    lines.append("|---|---|---|---|---|---|---|")
    g1_pass_by_k: dict[tuple[int, str], bool] = {}
    g1_rider_flags: list[str] = []
    for k in KS:
        for comparator in ("ours_uniform", "distill"):
            diffs = paired_diffs(metrics_all, "forgetting", CFG, comparator, k, ALL_SEEDS)
            old, new, pooled = three_way(diffs)
            if not pooled:
                lines.append(f"| {k} | {comparator} | N/A | N/A | N/A | N/A | N/A |")
                continue
            pooled_mean = float(np.mean(pooled))
            ok = pooled_mean <= 0
            g1_pass_by_k[(k, comparator)] = ok
            ci_str = "N/A"
            if len(pooled) >= 2:
                _, lo, hi = boot_ci_paired(np.array(pooled), np.zeros(len(pooled)))
                ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            lines.append(
                f"| {k} | {comparator} | {fmt_mean(old)} {fmt_list(old)} | "
                f"{fmt_mean(new)} {fmt_list(new)} | {pooled_mean:+.4f} | "
                f"{ci_str} | {'PASS' if ok else 'FAIL'} |")
            # replication rider: pooled PASS but both new seeds individually worse
            if ok and len(new) == 2 and all(d > 0 for d in new):
                g1_rider_flags.append(
                    f"g1 K={k} vs {comparator}: pooled PASSES ({pooled_mean:+.4f}) "
                    f"but BOTH new seeds individually worse ({fmt_list(new)}) — "
                    "§10.3 rider TRIGGERED, not a robust confirmation.")
    lines.append("")
    g1_overall = all(g1_pass_by_k.get((k, c), False)
                     for k in KS for c in ("ours_uniform", "distill"))

    # ================= g2: utility (>=4/5 seeds), K=1 or K=2 =================
    lines.append("## g2 — Utility axis: ≥4/5 seeds improvement direction vs BOTH comparators, K=1 or K=2\n")
    lines.append(
        "Rule (§10.1, new extension criterion, does not retroactively "
        "modify the original v0.33.3 3/3 criterion): `eps_optimal_mass` "
        "higher AND `normalized_regret` lower, simultaneously vs "
        "`ours_uniform` AND `distill`, for the same seed, at K=1 or K=2. "
        "A K passes if ≥4 of the 5 seeds show this simultaneous "
        "improvement.\n")
    lines.append("| K | axis | per-seed Δ vs ours_uniform [0,1,2,3,4] | per-seed Δ vs distill [0,1,2,3,4] | n/5 seeds improving both | g2 (this K,axis) |")
    lines.append("|---|---|---|---|---|---|")
    g2_pass_by_k = {1: False, 2: False}
    for k in (1, 2):
        for axis_name, col, higher_better in (
                ("eps_optimal_mass", "eps_optimal_mass", True),
                ("normalized_regret", "normalized_regret", False)):
            ou = paired_diffs(util_all, col, CFG, "ours_uniform", k, ALL_SEEDS)
            di = paired_diffs(util_all, col, CFG, "distill", k, ALL_SEEDS)
            common = [s for s in ALL_SEEDS if s in ou and s in di]
            n_improve = 0
            for s in common:
                if higher_better:
                    if ou[s] > 0 and di[s] > 0:
                        n_improve += 1
                else:
                    if ou[s] < 0 and di[s] < 0:
                        n_improve += 1
            ok = len(common) >= 4 and n_improve >= 4
            if ok:
                g2_pass_by_k[k] = True
            ou_str = "[" + ", ".join(f"{ou.get(s, float('nan')):+.4f}" for s in ALL_SEEDS) + "]"
            di_str = "[" + ", ".join(f"{di.get(s, float('nan')):+.4f}" for s in ALL_SEEDS) + "]"
            lines.append(
                f"| {k} | {axis_name} | {ou_str} | {di_str} | "
                f"{n_improve}/{len(common)} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    g2_overall = g2_pass_by_k[1] or g2_pass_by_k[2]

    # ================= g3: replay safety (unchanged, pooled) =================
    lines.append(f"## g3 — Replay safety: pooled 5-seed mean Forgetting vs `replay` ≤ +{EPS_TOL_G3} (unchanged)\n")
    lines.append("| K | old3 mean | new2 mean | pooled5 mean | pooled5 95% CI | g3 |")
    lines.append("|---|---|---|---|---|---|")
    g3_pass_by_k = {}
    for k in KS:
        diffs = paired_diffs(metrics_all, "forgetting", CFG, "replay", k, ALL_SEEDS)
        old, new, pooled = three_way(diffs)
        if not pooled:
            lines.append(f"| {k} | N/A | N/A | N/A | N/A | N/A |")
            continue
        pooled_mean = float(np.mean(pooled))
        ok = pooled_mean <= EPS_TOL_G3
        g3_pass_by_k[k] = ok
        ci_str = "N/A"
        if len(pooled) >= 2:
            _, lo, hi = boot_ci_paired(np.array(pooled), np.zeros(len(pooled)))
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"| {k} | {fmt_mean(old)} | {fmt_mean(new)} | "
                    f"{pooled_mean:+.4f} | {ci_str} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    g3_overall = all(g3_pass_by_k.get(k, False) for k in KS)

    # ================= g4: plasticity (unchanged threshold, 5-seed) =================
    lines.append(f"## g4 — Plasticity: per (task,K) pooled 5-seed paired mean ΔA[t,t] vs `ours_uniform` ≥ {EPS_TOL_G4} (unchanged)\n")
    lines.append(
        "Each cell footnoted with its per-task test-set accuracy quantum "
        "(1/n_test) — descriptive only, not used to argue the threshold is "
        "wrong (Sol's correction, §9 changelog).\n")
    lines.append("| task | K | quantum (1/n_test) | old3 mean [0,1,2] | new2 mean [3,4] | pooled5 mean [all] | pooled5 95% CI | g4 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    g4_pass_by_cell: dict[tuple[str, int], bool] = {}
    g4_rider_flags: list[str] = []
    tasks_seen = sorted(set(zip(eq_pres_own.task_idx, eq_pres_own.task))) if not eq_pres_own.empty else []
    for task_idx, task_name in tasks_seen:
        for k in KS:
            a = eq_pres_own[(eq_pres_own.task_idx == task_idx) & (eq_pres_own.K == k)].set_index("seed")
            b = ours_u_own[(ours_u_own.task_idx == task_idx) & (ours_u_own.K == k)].set_index("seed")
            common = [s for s in ALL_SEEDS if s in a.index and s in b.index]
            if not common:
                lines.append(f"| {task_name} | {k} | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            n_test = int(a.loc[common[0], "n_test"])
            quantum = 1.0 / n_test if n_test else float("nan")
            diffs = {s: float(a.loc[s, "acc_own"] - b.loc[s, "acc_own"]) for s in common}
            old, new, pooled = three_way(diffs)
            pooled_mean = float(np.mean(pooled))
            ok = pooled_mean >= EPS_TOL_G4
            g4_pass_by_cell[(task_name, k)] = ok
            ci_str = "N/A"
            if len(pooled) >= 2:
                _, lo, hi = boot_ci_paired(np.array(pooled), np.zeros(len(pooled)))
                ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            lines.append(
                f"| {task_name} | {k} | {quantum:.4f} (n={n_test}) | "
                f"{fmt_mean(old)} {fmt_list(old)} | {fmt_mean(new)} {fmt_list(new)} | "
                f"{pooled_mean:+.4f} | {ci_str} | {'PASS' if ok else 'FAIL'} |")
            if ok and len(new) == 2 and all(d < 0 for d in new):
                g4_rider_flags.append(
                    f"g4 {task_name} K={k}: pooled PASSES ({pooled_mean:+.4f}) "
                    f"but BOTH new seeds individually negative ({fmt_list(new)}) "
                    "— §10.3 rider TRIGGERED, not a robust confirmation.")
    lines.append("")
    g4_overall = all(g4_pass_by_cell.values()) if g4_pass_by_cell else False

    # ================= replication rider summary =================
    lines.append("## §10.3 Replication rider check\n")
    rider_flags = g1_rider_flags + g4_rider_flags
    if rider_flags:
        lines.append("**TRIGGERED on the following cells** — even if the pooled "
                     "verdict below says PASS, these specific cells are NOT a "
                     "robust confirmation and `eq_pres` is NOT promoted on their "
                     "strength:\n")
        for f in rider_flags:
            lines.append(f"- {f}")
    else:
        lines.append("**Not triggered on any g1/g4 cell** — for every PASS cell, "
                     "at least one of the two new seeds (3, 4) is not "
                     "individually on the failing side.")
    lines.append("")
    rider_triggered = len(rider_flags) > 0

    # ================= overall verdict =================
    lines.append("## Overall Gate v2 verdict for `eq_pres`\n")
    lines.append("| g1 (capability) | g2 (utility) | g3 (replay safety) | g4 (plasticity) | §10.3 rider | GATE v2 |")
    lines.append("|---|---|---|---|---|---|")
    gate_v2_pass = g1_overall and g2_overall and g3_overall and g4_overall
    branch = "PASS" if (gate_v2_pass and not rider_triggered) else "FAIL"
    lines.append(
        f"| {'PASS' if g1_overall else 'FAIL'} | {'PASS' if g2_overall else 'FAIL'} | "
        f"{'PASS' if g3_overall else 'FAIL'} | {'PASS' if g4_overall else 'FAIL'} | "
        f"{'triggered' if rider_triggered else 'clear'} | **GATE v2 {branch}** |")
    lines.append("")
    if branch == "PASS":
        lines.append(
            "**Branch (§10.6): PASS.** Next: `eq_pres` reverse-order promotion "
            "run (5 seeds x K∈{1,2,4}, Mac MPS), naming collision check "
            "triggered. **STOP for joint Sol+Fable+Aaron unblinding review "
            "before launching that promotion run** — no promotion compute "
            "has been started by this script.")
    else:
        lines.append(
            "**Branch (§10.6): FAIL"
            + (" (gate criteria pass on aggregate, but the §10.3 replication "
               "rider is triggered, so this cannot be called a robust "
               "confirmation)" if gate_v2_pass and rider_triggered else "")
            + ".** `eq_pres` is reported as an intervention/mechanism result "
            "in the main analysis paper, not a promoted method. This is the "
            "last main-order method-rescue compute (§10.5) — no further "
            "rescue regardless of this outcome. **STOP for joint Sol+Fable+"
            "Aaron unblinding review.**")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    print(f"\nGATE v2 verdict for eq_pres: {branch}")
    print(f"  g1={g1_overall} g2={g2_overall} g3={g3_overall} g4={g4_overall} "
          f"rider_triggered={rider_triggered}")


if __name__ == "__main__":
    main()
