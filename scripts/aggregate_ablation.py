"""Aggregate the completed ablation grid and generate mechanism outputs.

Inputs:
    results/cl_main_can_main_ablA*_s*.csv
    results/cl_main_can_main_full_s*.csv

Outputs:
    results/ablation_table.md
    results/ablation_per_run.csv
    figures/ablation_mechanism.png

The aggregation uses the same per-run definitions as aggregate_results.py:
AA, forgetting, BWT, selection Jaccard, action KL, and selected utility.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from nav import add_device_argument  # noqa: E402
from aggregate_results import per_run_metrics  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"

ABLATION_LABELS = {
    "ablA1": "uniform utility weight (ours_uniform)",
    "ablA2a": "memory cap 128",
    "ablA2b": "memory cap 2048",
    "ablA3a": "lambda 0.3",
    "ablA3b": "lambda 3.0",
}


def load_runs(input_dir: Path = RESULTS) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.rglob("*ablA*_s*.csv")):
        m = re.search(r"_(ablA[123][ab]? )_s", path.name.replace(" ", ""))
        if not m:
            m = re.search(r"_(ablA[123][ab]?)_s", path.name)
        if not m:
            raise ValueError(f"cannot parse ablation tag from {path.name}")
        tag = m.group(1)
        df = pd.read_csv(path)
        run = per_run_metrics(df)
        run["tag"] = tag
        rows.append(run)

    # Include main-order baselines for direct comparison.
    main = pd.concat(
        [pd.read_csv(p) for p in sorted(
            input_dir.rglob("cl_main_can_main_full_s*.csv"))],
        ignore_index=True,
    )
    main = main[main.method.isin(["ours", "distill"])]
    main_run = per_run_metrics(main)
    main_run["tag"] = main_run.method.map({"ours": "ours_main", "distill": "distill_main"})
    rows.append(main_run)
    return pd.concat(rows, ignore_index=True)


def summary_table(run: pd.DataFrame) -> pd.DataFrame:
    cols = ["AA", "forgetting", "jaccard", "action_kl", "sel_utility"]
    s = run.groupby(["tag", "K"])[cols].agg(["mean", "std"]).reset_index()
    return s


def fmt(row, col: str) -> str:
    return f"{row[(col, 'mean')]:.3f} ± {row[(col, 'std')]:.3f}"


def write_markdown(run: pd.DataFrame, output_dir: Path = RESULTS) -> None:
    s = summary_table(run).set_index(["tag", "K"])
    lines = [
        "# Ablation / mechanism analysis",
        "",
        "Main-order can_dataset, mean ± std over 5 seeds.",
        "",
        "## Aggregate metrics",
        "",
        "| setting | K | AA | forgetting ↓ | Jaccard ↑ | action-KL ↓ | utility ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["distill_main", "ours_main", "ablA1", "ablA2a", "ablA2b", "ablA3a", "ablA3b"]
    for tag in order:
        for k in sorted(run.K.unique()):
            key = (tag, k)
            if key not in s.index:
                continue
            row = s.loc[key]
            label = {
                "distill_main": "distill baseline",
                "ours_main": "ours (memory 512, λ=1)",
                **ABLATION_LABELS,
            }[tag]
            lines.append(
                f"| {label} | {k} | {fmt(row, 'AA')} | "
                f"{fmt(row, 'forgetting')} | {fmt(row, 'jaccard')} | "
                f"{fmt(row, 'action_kl')} | {fmt(row, 'sel_utility')} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- `ours_main` is the reference configuration: utility weighting, replay, memory cap 512, λ=1.",
            "- `ablA1` tests uniform weighting with the same replay/distillation structure.",
            "- `ablA2a` / `ablA2b` test memory caps 128 / 2048.",
            "- `ablA3a` / `ablA3b` test λ=0.3 / 3.0.",
            "- `distill baseline` is the main-grid distill-only baseline; it is not identical to `ablA1`.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_table.md").write_text("\n".join(lines))


def plot(run: pd.DataFrame, output_dir: Path = FIGURES) -> None:
    s = summary_table(run).set_index(["tag", "K"])
    settings = ["distill_main", "ours_main", "ablA1", "ablA2a", "ablA2b", "ablA3a", "ablA3b"]
    labels = ["distill", "ours", "uniform", "mem128", "mem2048", "λ=.3", "λ=3"]
    colors = ["#64748b", "#dc2626", "#f59e0b", "#2563eb", "#7c3aed", "#0891b2", "#16a34a"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(settings))
    for ax, col, ylabel in [
        (axes[0], "AA", "final average balanced accuracy ↑"),
        (axes[1], "forgetting", "mean forgetting ↓"),
        (axes[2], "jaccard", "selection Jaccard ↑"),
    ]:
        width = 0.22
        for i, k in enumerate(sorted(run.K.unique())):
            vals, errs = [], []
            for tag in settings:
                if (tag, k) not in s.index:
                    vals.append(np.nan)
                    errs.append(0)
                else:
                    row = s.loc[(tag, k)]
                    vals.append(row[(col, "mean")])
                    errs.append(row[(col, "std")])
            ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2,
                   label=f"K={k}", alpha=0.82)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend()
    fig.suptitle("Ablation and mechanism analysis — can_dataset main order")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ablation_mechanism.png", dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--input-dir", type=Path, default=RESULTS)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new-run output; defaults to runs/v2/ablation_<tag>")
    add_device_argument(ap)
    args = ap.parse_args()
    output_dir = (args.output_dir if args.output_dir is not None else
                  REPO / "runs" / "v2" / f"ablation_{args.tag}")
    figure_dir = output_dir / "figures"
    run = load_runs(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run.to_csv(output_dir / "ablation_per_run.csv", index=False)
    write_markdown(run, output_dir)
    plot(run, figure_dir)
    print(f"loaded {len(run)} per-run rows")
    print(output_dir / "ablation_table.md")
    print(figure_dir / "ablation_mechanism.png")


if __name__ == "__main__":
    main()
