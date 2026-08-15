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


BASE_GRID_METHODS = {"ours", "distill", "replay", "seqft"}

# v0.292 pre-registered the first six; the three seqft rows were added in the
# v0.33 carry-over (c1) after the fact, so they are exploratory by
# definition even though they test the paper's central C2/C4 claims.
CONFIRMATORY_COMPARISONS = {
    "ours-distill", "ours-replay", "distill-replay",
    "ours-ours_uniform", "mem2048-ours", "lambda3-ours",
}


def compute(settings: dict[str, pd.DataFrame], orders: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("ours", "distill"),
        ("ours", "replay"),
        ("distill", "replay"),
        ("ours", "ours_uniform"),
        ("mem2048", "ours"),
        ("lambda3", "ours"),
        # v0.33 carry-over c1: CL-vs-no-preservation rows were missing even
        # though C2/C4 are the paper's central claims.
        ("distill", "seqft"),
        ("ours", "seqft"),
        ("replay", "seqft"),
    ]
    for order in orders:
        if order not in settings:
            continue
        base = settings[order]
        for focal, baseline in comparisons:
            focal_df = (base[base["method"].eq(focal)]
                        if focal in BASE_GRID_METHODS
                        else settings.get(f"{order}:{focal}",
                                          pd.DataFrame()))
            base_df = (base[base["method"].eq(baseline)]
                       if baseline in BASE_GRID_METHODS
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
                    comparison = f"{focal}-{baseline}"
                    sign_agree = int(max((diff > 0).sum(), (diff < 0).sum()))
                    rows.append(dict(
                        order=order, K=int(k), comparison=comparison,
                        family=("confirmatory"
                               if comparison in CONFIRMATORY_COMPARISONS
                               else "exploratory"),
                        metric=metric, n=len(diff), mean=mean,
                        ci_lo=lo, ci_hi=hi, sign_agree=sign_agree,
                        p_value=sign_flip_p(diff)))
    result = pd.DataFrame(rows)
    if not result.empty:
        # Kept for transparency/audit only — v0.33.1 stats policy (§2.6 of
        # docs/handoff_fable_sol_20260815.md) is descriptive-first: the
        # n=5 exact sign-flip floor (p_min=0.0625) means no comparison here
        # can ever clear q<=0.05, so paper-facing text must not use these
        # two columns as a significance claim.
        result["q_value"] = bh_qvalues(result["p_value"])
        result["fdr_significant"] = result["q_value"] <= 0.05
    return result


def render(result: pd.DataFrame, settings: dict[str, pd.DataFrame],
           orders: list[str]) -> str:
    lines = [
        "# WP1 paired statistics pack",
        "",
        "v0.292 zero-compute analysis of frozen per-seed CSVs, extended in v0.33 "
        "(carry-over c1) with `*-seqft` rows. Positive `mean` means the first "
        "method is better for the metric direction.",
        "",
        "## Statistics policy (v0.33.1, decided — see "
        "docs/handoff_fable_sol_20260815.md §2.6)",
        "",
        "This report is **descriptive by design, not a significance test**. With "
        "`n=5` paired seeds, the exact two-sided sign-flip test's smallest "
        "achievable p-value is `p_min = 2/2^5 = 0.0625 > 0.05`; Benjamini–Hochberg's "
        "adjusted q-value for the single best-ranked hypothesis in any family of "
        "size `m` is `q_(1) = p_(1)`, independent of `m`, so **no comparison family "
        "size can ever clear `q <= 0.05` here** — shrinking the family does not "
        "help. `p_value`/`q_value` are kept in the CSV for audit only; paper-facing "
        "text must use only `mean`, the 95% bootstrap CI, and `sign_agree` (how "
        "many of the `n` seeds agree with `mean`'s sign, e.g. `5/5`). Each row is "
        "also tagged `confirmatory` (the six comparisons pre-registered in "
        "docs/research_contract_v0.292.md) or `exploratory` (the `*-seqft` rows "
        "added afterward in v0.33) — do not present exploratory rows as "
        "pre-registered.",
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
        family = group["family"].iloc[0]
        lines.append(f"### {order}, K={k}, `{comp}` ({family})")
        for row in group.itertuples():
            lines.append(
                f"- {row.metric}: mean={row.mean:+.4f}, "
                f"95% CI [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}], "
                f"sign_agree={row.sign_agree}/{row.n}")
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
