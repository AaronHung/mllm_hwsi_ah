"""WP1: CPU-only paired statistics over completed navigation CSVs.

This is deliberately independent of torch/MPS/CUDA.  It reads frozen
Protocol-v1 result files and writes a new v2 analysis artifact; it never
modifies the frozen files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_results import per_run_metrics


def bootstrap_ci(d: np.ndarray, seed: int = 0, n_boot: int = 10_000):
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def load_metrics(input_dir: Path, dataset: str, order: str) -> pd.DataFrame:
    csvs = sorted(input_dir.glob(f"cl_main_{dataset}_{order}_*.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    required = ["method", "seed", "K", "stage", "task_idx"]
    return per_run_metrics(df.drop_duplicates(subset=required, keep="last"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("runs/v2/wp1_paired_stats"))
    ap.add_argument("--dataset", default="can")
    ap.add_argument("--orders", nargs="+", default=["main", "reverse"])
    ap.add_argument("--device", default="cpu",
                    help="accepted for common CLI; WP1 always executes on CPU")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    markdown = [
        "# WP1 paired statistics (CPU pandas)",
        "",
        "Frozen Protocol-v1 CSVs are read-only inputs; this artifact is v2.",
        "",
    ]
    for order in args.orders:
        metrics = load_metrics(args.input_dir, args.dataset, order)
        if metrics.empty:
            markdown.append(f"- `{order}`: no CSVs found; skipped.")
            continue
        for k in sorted(metrics["K"].unique()):
            sub = metrics[metrics["K"] == k]
            for baseline in ("seqft", "distill"):
                pair = sub[sub["method"].isin(["ours", baseline])].pivot(
                    index="seed", columns="method",
                    values=["AA", "forgetting", "jaccard"])
                if "ours" not in pair.columns.get_level_values(1) or \
                        baseline not in pair.columns.get_level_values(1):
                    continue
                for metric in ("AA", "forgetting", "jaccard"):
                    ours = pair[(metric, "ours")].dropna()
                    base = pair[(metric, baseline)].reindex(ours.index).dropna()
                    common = ours.index.intersection(base.index)
                    if len(common) < 2:
                        continue
                    ours_v = ours.loc[common].to_numpy()
                    base_v = base.loc[common].to_numpy()
                    # Positive means ours is better for all three reportable
                    # directions: higher AA/Jaccard, lower forgetting.
                    diff = base_v - ours_v if metric == "forgetting" \
                        else ours_v - base_v
                    mean, lo, hi = bootstrap_ci(diff)
                    rows.append(dict(
                        dataset=args.dataset, order=order, K=k,
                        comparison=f"ours-{baseline}", metric=metric,
                        n=len(diff), mean=mean, ci_lo=lo, ci_hi=hi,
                        resolved_device="cpu"))
                    markdown.append(
                        f"- `{order}` K={k} ours vs {baseline} {metric}: "
                        f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(diff)}")

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "paired_stats.csv", index=False)
    (args.output_dir / "paired_stats.md").write_text(
        "\n".join(markdown) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(
        {"resolved_device": "cpu", "input_dir": str(args.input_dir),
         "analysis": "paired bootstrap 95% CI"}, indent=2))
    print(f"WP1 done -> {args.output_dir}")


if __name__ == "__main__":
    main()
