"""WP1: paired seed-level statistics for the v0.292 contract.

Reads completed Protocol-v1 CSVs only.  No model or dataset loading occurs.
The six requested comparison families are reported wherever the source files
provide the paired settings.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from aggregate_results import per_run_metrics  # noqa: E402
from nav import add_device_argument  # noqa: E402

METRICS = ("AA", "Forgetting", "BWT", "Jaccard", "action-KL")
RAW_METRIC = {
    "AA": "AA",
    "Forgetting": "forgetting",
    "BWT": "BWT",
    "Jaccard": "jaccard",
    "action-KL": "action_kl",
}
HIGHER_IS_BETTER = {"AA", "BWT", "Jaccard"}


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        required = ["method", "seed", "K", "stage", "task_idx"]
        raw = raw.drop_duplicates(subset=required, keep="last")
        frames.append(per_run_metrics(raw))
    return pd.concat(frames, ignore_index=True)


def load_order(input_dir: Path, order: str) -> pd.DataFrame:
    return load_csvs(sorted(
        input_dir.glob(f"cl_main_can_{order}_full_s*.csv")))


def load_ablation(input_dir: Path, tag: str, alias: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(f"cl_main_can_main_{tag}_s*.csv"))
    df = load_csvs(paths)
    if df.empty:
        return df
    df = df[df["method"].eq("ours") | df["method"].eq("ours_uniform")].copy()
    df["method"] = alias
    return df


def bootstrap_ci(diff: np.ndarray, seed: int) -> tuple[float, float, float]:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(10_000, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def sign_flip_p(diff: np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value for the small seed sample."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(diff))))
    null = (signs * diff[None, :]).mean(axis=1)
    return float(np.mean(np.abs(null) >= abs(float(diff.mean())) - 1e-12))


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    """Benjamini–Hochberg q-values over the one pre-registered family."""
    q = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return q
    ordered = valid.sort_values()
    m = len(ordered)
    raw = ordered.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
    q.loc[ordered.index] = adjusted
    return q


def add_setting(
    settings: dict[str, pd.DataFrame],
    name: str,
    df: pd.DataFrame,
) -> None:
    if not df.empty:
        settings[name] = df


def compute(settings: dict[str, pd.DataFrame], orders: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("ours", "distill"),
        ("ours", "replay"),
        ("distill", "replay"),
        ("ours", "ours_uniform"),
        ("mem2048", "ours"),
        ("lambda3", "ours"),
    ]
    for order in orders:
        if order not in settings:
            continue
        base = settings[order]
        for focal, baseline in comparisons:
            focal_df = (base[base["method"].eq(focal)]
                        if focal in {"ours", "distill", "replay"}
                        else settings.get(f"{order}:{focal}",
                                          pd.DataFrame()))
            base_df = (base[base["method"].eq(baseline)]
                       if baseline in {"ours", "distill", "replay"}
                       else settings.get(f"{order}:{baseline}",
                                         pd.DataFrame()))
            if focal_df.empty or base_df.empty:
                continue
            for k in sorted(set(focal_df["K"]) & set(base_df["K"])):
                left = focal_df[focal_df["K"] == k].set_index("seed")
                right = base_df[base_df["K"] == k].set_index("seed")
                common = left.index.intersection(right.index)
                if len(common) < 2:
                    continue
                for metric in METRICS:
                    col = RAW_METRIC[metric]
                    lv = left.loc[common, col].astype(float)
                    rv = right.loc[common, col].astype(float)
                    diff = (lv - rv if metric in HIGHER_IS_BETTER
                            else rv - lv).to_numpy()
                    mean, lo, hi = bootstrap_ci(
                        diff, seed=17 + len(rows))
                    rows.append(dict(
                        order=order, K=int(k),
                        comparison=f"{focal}-{baseline}",
                        metric=metric, n=len(diff), mean=mean,
                        ci_lo=lo, ci_hi=hi, p_value=sign_flip_p(diff)))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = bh_qvalues(result["p_value"])
        result["fdr_significant"] = result["q_value"] <= 0.05
    return result


def render(result: pd.DataFrame, settings: dict[str, pd.DataFrame],
           orders: list[str]) -> str:
    lines = [
        "# WP1 paired statistics pack",
        "",
        "v0.292 zero-compute analysis of frozen per-seed CSVs. Positive "
        "`mean` means the first method is better for the metric direction.",
        "",
        "## Multiple-comparison policy",
        "",
        "One policy is used for the entire report: Benjamini–Hochberg FDR "
        "control at q=0.05 over every available comparison × order × K × "
        "metric hypothesis. Bootstrap intervals are descriptive 95% CIs; "
        "claim-level significance requires the adjusted q-value and a CI "
        "that points in the same direction. No metric is selected after "
        "seeing the results.",
        "",
        "## Coverage",
        "",
    ]
    for order in orders:
        df = settings.get(order, pd.DataFrame())
        lines.append(f"- `{order}` full grid: "
                     f"{'available' if not df.empty else 'missing'}")
        for tag in ("ours_uniform", "mem2048", "lambda3"):
            lines.append(f"- `{order}:{tag}`: "
                         f"{'available' if f'{order}:{tag}' in settings else 'missing'}")
    lines.extend(["", "## Results", ""])
    if result.empty:
        lines.append("No requested paired comparison had at least two common seeds.")
        return "\n".join(lines) + "\n"
    for (order, k, comp), group in result.groupby(
            ["order", "K", "comparison"], sort=True):
        lines.append(f"### {order}, K={k}, `{comp}`")
        for row in group.itertuples():
            sig = " **FDR-significant**" if row.fdr_significant else ""
            lines.append(
                f"- {row.metric}: mean={row.mean:+.4f}, "
                f"95% CI [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}], "
                f"p={row.p_value:.4f}, q={row.q_value:.4f}, n={row.n}{sig}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=REPO / "results")
    ap.add_argument("--output-md", type=Path,
                    default=REPO / "results" / "paired_stats_pack.md")
    ap.add_argument("--output-csv", type=Path,
                    default=REPO / "results" / "paired_stats_pack.csv")
    ap.add_argument("--orders", nargs="+", default=["main", "reverse"])
    add_device_argument(ap)
    args = ap.parse_args()

    settings: dict[str, pd.DataFrame] = {}
    for order in args.orders:
        add_setting(settings, order, load_order(args.input_dir, order))
    # Ablations currently exist for main order only.  Keep this explicit so
    # absence on reverse is reported rather than silently pooled.
    add_setting(settings, "main:ours_uniform",
                load_ablation(args.input_dir, "ablA1", "ours_uniform"))
    add_setting(settings, "main:mem2048",
                load_ablation(args.input_dir, "ablA2b", "mem2048"))
    add_setting(settings, "main:lambda3",
                load_ablation(args.input_dir, "ablA3b", "lambda3"))
    # Aliases for the base comparison resolver.
    if "main:ours_uniform" in settings:
        settings["main:ours_uniform"] = settings["main:ours_uniform"]
    result = compute(settings, args.orders)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(result, settings, args.orders))
    result.to_csv(args.output_csv, index=False)
    print(f"WP1 -> {args.output_md}")
    print(f"rows={len(result)}")


if __name__ == "__main__":
    main()
