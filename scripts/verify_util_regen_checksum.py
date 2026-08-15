"""v0.33.2 item 3 — checksum a probe-only regeneration of `ours_uniform` /
`distill` against the frozen Protocol-v1 rows, then extract the newly
archived utility-axis comparator metrics (only for rows that pass).

The frozen rows (`results/cl_main_can_main_full_s*.csv` for `distill`;
`results/cl_main_can_main_ablA1_s*.csv` for `ours_uniform` — it was only ever
run as the utility-ablation comparator, never part of the 7-method `full`
grid) predate `eps_optimal_mass`/`normalized_regret` and are never modified.
A probe-only rerun (seeds {0,1,2}, K in {1,2}, main order, same backend as
the frozen rows — see `docs/method_gate_v033.md` §5.2) reproduces those same
runs to backfill the two new columns. Before those columns are trusted as g2
comparators, this script checks the pre-declared acceptance checksum:
|delta| <= tol on AA, Forgetting, Jaccard, action-KL per (method, seed, K),
computed with the same `per_run_metrics` aggregation used everywhere else in
this project.

**Backend note:** the frozen `can` grid (`logs/cl_main_s*.log`,
`logs/abl_ablA1_s*.log`) was run on Mac **MPS**, not CPU — verify this
against those logs before trusting a checksum PASS; a same-seed CPU rerun of
this tiny-MLP sequential simulation does NOT reproduce the MPS numbers
within 0.001 (confirmed empirically: 12/12 rows failed on CPU, 12/12 passed
on MPS for the same command).

用法：
    python scripts/verify_util_regen_checksum.py \
        --regen-csv runs/v2/util_regen_<tag>/cl_main_can_main_util_regen_<tag>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.aggregate_results import per_run_metrics  # noqa: E402

RESULTS = REPO / "results"
DEFAULT_TOL = 0.001
CHECK_COLS = ["AA", "forgetting", "jaccard", "action_kl"]
METHODS = ("ours_uniform", "distill")


def load_frozen(dataset: str, order: str) -> pd.DataFrame:
    """`distill` lives in the main Protocol-v1 grid (`..._full_s*.csv`);
    `ours_uniform` was only ever run as the utility-ablation comparator
    (`..._ablA1_s*.csv`, K in {1,2}) — it was never part of the 7-method
    `full` grid. Both are frozen evidence; search both tags.
    """
    csvs = sorted(RESULTS.glob(f"cl_main_{dataset}_{order}_full_s*.csv")) + \
        sorted(RESULTS.glob(f"cl_main_{dataset}_{order}_ablA1_s*.csv"))
    if not csvs:
        raise SystemExit(f"frozen Protocol-v1 CSVs not found under {RESULTS}")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    return df[df.method.isin(METHODS)]


def utility_axis_g2(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered g2 aggregation (docs/method_gate_v033.md §5.1): final
    stage, OLD tasks only (task_idx < n_stage); each row is already a
    within-task mean over probe states, so this step is the macro-average
    across old tasks.
    """
    n_stage = df.stage.max()
    rows = []
    for (m, s, k), g in df.groupby(["method", "seed", "K"]):
        old = g[(g.stage == n_stage) & (g.task_idx < n_stage)]
        if old.empty:
            continue
        rows.append(dict(method=m, seed=s, K=k,
                         eps_optimal_mass=old.eps_optimal_mass.mean(),
                         normalized_regret=old.normalized_regret.mean()))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-csv", type=Path, required=True)
    ap.add_argument("--dataset", default="can")
    ap.add_argument("--order", default="main")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--out", type=Path, default=RESULTS / "util_regen_checksum.md")
    args = ap.parse_args()

    frozen = load_frozen(args.dataset, args.order)
    regen_raw = pd.read_csv(args.regen_csv)
    regen = regen_raw[regen_raw.method.isin(METHODS)]
    if regen.empty:
        raise SystemExit(f"{args.regen_csv} has no ours_uniform/distill rows")

    frozen_m = per_run_metrics(frozen)
    regen_m = per_run_metrics(regen)
    merged = frozen_m.merge(regen_m, on=["method", "seed", "K"],
                            suffixes=("_frozen", "_regen"))
    if merged.empty:
        raise SystemExit("no overlapping (method, seed, K) rows — did the "
                         "regen run finish for the expected seeds/K?")

    lines = ["# util_regen checksum (v0.33.2 item 3)", "",
            f"Tolerance: `|Δ| <= {args.tol}` on {', '.join(CHECK_COLS)}, per "
            "(method, seed, K), frozen Protocol-v1 rows vs. probe-only "
            f"regeneration `{args.regen_csv.name}`. The frozen `can` grid "
            "was run on Mac MPS (verified against `logs/cl_main_s*.log` / "
            "`logs/abl_ablA1_s*.log`); the regen run must use the same "
            "backend for this checksum to be meaningful.",
            "",
            "| method | seed | K | " + " | ".join(f"Δ{c}" for c in CHECK_COLS)
            + " | pass |",
            "|---|---|---|" + "---|" * (len(CHECK_COLS) + 1)]
    all_pass = True
    admitted: list[tuple] = []
    for _, row in merged.sort_values(["method", "seed", "K"]).iterrows():
        deltas = [abs(row[f"{c}_frozen"] - row[f"{c}_regen"]) for c in CHECK_COLS]
        ok = all(d <= args.tol for d in deltas)
        all_pass = all_pass and ok
        if ok:
            admitted.append((row.method, row.seed, row.K))
        lines.append(
            f"| `{row.method}` | {row.seed} | {row.K} | "
            + " | ".join(f"{d:.4f}" for d in deltas)
            + f" | {'PASS' if ok else 'FAIL'} |")

    lines += ["", f"**Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}** "
             f"({len(admitted)}/{len(merged)} (method,seed,K) rows admitted "
             "as g2 utility comparators).", ""]

    if admitted:
        util = utility_axis_g2(regen)
        util = util[util.apply(
            lambda r: (r.method, r.seed, r.K) in admitted, axis=1)]
        lines += ["## Admitted utility-axis comparator values (final stage, "
                 "old tasks only, macro-average across old tasks)", "",
                 "| method | seed | K | eps_optimal_mass (higher better) | "
                 "normalized_regret (lower better) |",
                 "|---|---|---|---|---|"]
        for _, r in util.sort_values(["method", "seed", "K"]).iterrows():
            lines.append(f"| `{r.method}` | {r.seed} | {r.K} | "
                         f"{r.eps_optimal_mass:.4f} | {r.normalized_regret:.4f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote -> {args.out}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
