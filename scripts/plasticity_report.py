"""WP3: zero-compute own-time new-task accuracy report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nav import add_device_argument  # noqa: E402
from nav.method_labels import method_label  # noqa: E402


def load_raw(paths: list[Path], setting: str, order: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.drop_duplicates(
            subset=["method", "seed", "K", "stage", "task_idx"],
            keep="last",
        ).copy()
        df["setting"] = setting
        df["order"] = order
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all(input_dir: Path) -> pd.DataFrame:
    frames = []
    for order in ("main", "reverse"):
        paths = sorted(input_dir.glob(f"cl_main_can_{order}_full_s*.csv"))
        frames.append(load_raw(paths, "method", order))

    # The lambda ablations are main-order only.  Keep them as explicit
    # settings rather than pretending they are reverse-order observations.
    for tag, setting in (("ablA3a", "ours_lambda0.3"),
                         ("ablA3b", "ours_lambda3")):
        paths = sorted(input_dir.glob(f"cl_main_can_main_{tag}_s*.csv"))
        df = load_raw(paths, setting, "main")
        if not df.empty:
            df = df[df["method"].eq("ours")]
            frames.append(df)
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def own_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["stage"].eq(df["task_idx"])].copy()
    out["setting_label"] = out.apply(
        lambda row: (
            ("Utility-Weighted Replay Distillation (λ=1.0)"
             if row["method"] == "ours" else method_label(str(row["method"])))
            if row["setting"] == "method"
            else {
                "ours_lambda0.3": "Utility-Weighted Replay Distillation (λ=0.3)",
                "ours_lambda3": "Utility-Weighted Replay Distillation (λ=3.0)",
            }[str(row["setting"])]
        ),
        axis=1,
    )
    out["setting_key"] = out.apply(
        lambda row: ("ours_lambda1"
                     if row["setting"] == "method" and row["method"] == "ours"
                     else str(row["method"]))
        if row["setting"] == "method" else str(row["setting"]),
        axis=1,
    )
    return out


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["order", "setting_key", "setting_label", "K", "task"],
                   as_index=False)
        .agg(A_t_t=("bal_acc", "mean"),
             std=("bal_acc", "std"),
             n=("bal_acc", "count"),
             raw_acc=("acc", "mean"))
        .sort_values(["order", "K", "setting_key", "task"])
    )


def lambda_conclusion(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    # The lambda ablation (ablA3a/ablA3b) only exists for order="main"; the
    # "ours_lambda1" key also matches order="reverse" rows from the full grid.
    # Restrict to main here so the diff never silently pools the two orders
    # (main/reverse own-time accuracy must never be pooled, per the report
    # header and the v0.292 backend-consistency rule).
    lam = df[df["setting_key"].isin(
        ["ours_lambda0.3", "ours_lambda1", "ours_lambda3"])
        & df["order"].eq("main")].copy()
    if lam.empty:
        return ("No lambda ablation rows were available.", pd.DataFrame())
    pivot = (
        lam.groupby(["K", "task", "setting_key"])["bal_acc"].mean()
        .unstack("setting_key")
        .dropna()
    )
    if pivot.empty or "ours_lambda3" not in pivot or "ours_lambda1" not in pivot:
        return ("Lambda=3 and lambda=1 settings could not be paired at a "
                "common task/K cell.",
                pivot)
    diff = pivot["ours_lambda3"] - pivot["ours_lambda1"]
    lower = int((diff < 0).sum())
    n = len(diff)
    if lower >= (n + 1) // 2:
        conclusion = (
            f"Own-time plasticity degradation is observed in {lower}/{n} "
            "matched task×K cells when increasing λ from 1.0 to 3.0. "
            "The paper may use stability–plasticity trade-off only with this "
            "qualified scope; do not generalize beyond these cells."
        )
    else:
        conclusion = (
            f"No consistent own-time plasticity degradation is observed: "
            f"λ=3.0 is lower than λ=1.0 in only {lower}/{n} matched cells. "
            "Use the phrase behavior-fidelity / capability-retention "
            "trade-off."
        )
    return conclusion, diff.rename("lambda3_minus_lambda1").reset_index()


def render(s: pd.DataFrame, conclusion: str) -> str:
    if "No consistent" in conclusion:
        wording = (
            "Use the phrase **behavior-fidelity / capability-retention "
            "trade-off**; do not upgrade it to a stability–plasticity claim.")
    else:
        wording = (
            "A qualified **stability–plasticity trade-off** is permissible "
            "for the matched λ cells above, but the paper must retain the "
            "broader behavior-fidelity / capability-retention framing and "
            "must not claim a universal monotonic effect.")
    lines = [
        "# WP3 plasticity report",
        "",
        "Zero-compute report from frozen per-seed CSVs. `A[t,t]` is the "
        "own-time new-task **balanced accuracy** at the stage in which task "
        "t first arrives; `std` is the sample standard deviation across "
        "seeds. Main and reverse rows are never pooled.",
        "",
        "## Interpretation",
        "",
        conclusion,
        "",
        wording,
        "",
        "## Own-time accuracy",
        "",
    ]
    if s.empty:
        lines.append("No own-time rows were found.")
        return "\n".join(lines) + "\n"
    for (order, k), g in s.groupby(["order", "K"], sort=True):
        lines.append(f"### {order}, K={k}")
        lines.append("| setting | task | A[t,t] | raw accuracy | n |")
        lines.append("|---|---|---:|---:|---:|")
        for row in g.itertuples():
            sd = "nan" if pd.isna(row.std) else f"{row.std:.3f}"
            lines.append(
                f"| {row.setting_label} | {row.task} | "
                f"{row.A_t_t:.3f} ± {sd} | {row.raw_acc:.3f} | {row.n} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=REPO / "results")
    ap.add_argument("--output-md", type=Path,
                    default=REPO / "results" / "plasticity_report.md")
    ap.add_argument("--output-csv", type=Path,
                    default=REPO / "results" / "plasticity_report.csv")
    add_device_argument(ap)
    args = ap.parse_args()

    raw = load_all(args.input_dir)
    own = own_time(raw)
    report = summary(own)
    conclusion, lambda_diff = lambda_conclusion(own)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(report, conclusion))
    report.to_csv(args.output_csv, index=False)
    if not lambda_diff.empty:
        lambda_diff.to_csv(args.output_md.with_name(
            "plasticity_lambda_differences.csv"), index=False)
    print(f"WP3 -> {args.output_md}")
    print(f"rows={len(report)}")
    print(conclusion)


if __name__ == "__main__":
    main()
