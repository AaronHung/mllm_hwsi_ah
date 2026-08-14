"""聚合 cl_main 結果 → 論文 main table（markdown）與主圖。

輸入：results/cl_main_{dataset}_{order}_{tag}.csv（scripts/cl_main.py 產出）
輸出：
    results/main_table_{dataset}_{order}.md
    figures/cl_budget_forgetting_{dataset}_{order}.png   （K × forgetting 交互，核心假說圖）
    figures/cl_jaccard_bars_{dataset}_{order}.png        （行為層保存）

用法：
    python scripts/aggregate_results.py --dataset can --order main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nav import add_device_argument  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
FIGURES = REPO / "figures"

METHOD_ORDER = ["seqft", "ewc", "lwf", "replay", "distill", "ours_uniform",
                "ours", "joint"]


def per_run_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """每個 (method, seed, K) 一列：AA / forgetting / BWT / jaccard / action_kl。"""
    rows = []
    n_stage = df.stage.max()
    for (m, s, k), g in df.groupby(["method", "seed", "K"]):
        final = g[g.stage == n_stage]
        aa = final.bal_acc.mean()
        forg, bwt, jac, akl, util = [], [], [], [], []
        for j, gj in g.groupby("task_idx"):
            acc_final = gj[gj.stage == n_stage].bal_acc.iloc[0]
            acc_own = gj[gj.stage == j].bal_acc.iloc[0]
            if j < n_stage:
                forg.append(gj.bal_acc.max() - acc_final)
                bwt.append(acc_final - acc_own)
                jac.append(gj[gj.stage == n_stage].jaccard.iloc[0])
                akl.append(gj[gj.stage == n_stage].action_kl.iloc[0])
                util.append(gj[gj.stage == n_stage].sel_utility.iloc[0])
        rows.append(dict(method=m, seed=s, K=k, AA=aa,
                         forgetting=np.mean(forg), BWT=np.mean(bwt),
                         jaccard=np.mean(jac), action_kl=np.mean(akl),
                         sel_utility=np.mean(util)))
    return pd.DataFrame(rows)


def boot_ci_paired(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="can")
    ap.add_argument("--order", default="main")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--input-dir", type=Path, default=RESULTS,
                    help="directory to search recursively for run CSVs")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="directory for the table and figures; defaults to runs/v2")
    add_device_argument(ap)
    args = ap.parse_args()
    output_dir = (args.output_dir if args.output_dir is not None else
                  REPO / "runs" / "v2" /
                  f"aggregate_{args.dataset}_{args.order}_{args.tag}")

    csvs = sorted(args.input_dir.rglob(
        f"cl_main_{args.dataset}_{args.order}_{args.tag}*.csv"))
    if not csvs:
        sys.exit(f"找不到 {args.input_dir}/cl_main_{args.dataset}_"
                 f"{args.order}_{args.tag}*.csv")
    df = pd.concat([pd.read_csv(c) for c in csvs]).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")
    print(f"loaded {len(csvs)} csv(s), {len(df)} rows")
    pr = per_run_metrics(df)

    lines = [f"# Main table — {args.dataset} / {args.order} "
             f"(mean ± std over {pr.seed.nunique()} seeds)", ""]
    for k in sorted(pr.K.unique()):
        sub = pr[pr.K == k]
        lines.append(f"\n## K = {k}\n")
        lines.append("| method | AA ↑ | Forgetting ↓ | BWT ↑ | Jaccard ↑ | action-KL ↓ | sel-utility ↑ |")
        lines.append("|---|---|---|---|---|---|---|")
        for m in [m for m in METHOD_ORDER if m in sub.method.unique()]:
            g = sub[sub.method == m]
            def ms(col):
                return f"{g[col].mean():.3f} ± {g[col].std():.3f}"
            lines.append(f"| {m} | {ms('AA')} | {ms('forgetting')} | {ms('BWT')} "
                         f"| {ms('jaccard')} | {ms('action_kl')} | {ms('sel_utility')} |")
        # paired CI：ours vs seqft
        if {"ours", "seqft"} <= set(sub.method.unique()):
            a = sub[sub.method == "ours"].sort_values("seed")
            b = sub[sub.method == "seqft"].sort_values("seed")
            if len(a) == len(b) and len(a) > 1:
                for col, better in [("AA", "higher"), ("forgetting", "lower"),
                                    ("jaccard", "higher")]:
                    m_, lo, hi = boot_ci_paired(a[col].values, b[col].values)
                    lines.append(f"\n- ours − seqft（{col}，{better} better）："
                                 f"{m_:+.3f}，95% CI [{lo:+.3f}, {hi:+.3f}]")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_md = output_dir / f"main_table_{args.dataset}_{args.order}.md"
    out_md.write_text("\n".join(lines))
    print(f"table -> {out_md}")

    # ---- 圖 1：budget × forgetting 交互 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = (FIGURES if output_dir.resolve() == RESULTS.resolve()
                  else output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {"seqft": "tab:gray", "ewc": "tab:olive", "lwf": "tab:cyan",
              "replay": "tab:blue", "distill": "tab:purple",
              "ours_uniform": "tab:orange", "ours": "tab:red",
              "joint": "black"}
    for m in [m for m in METHOD_ORDER if m in pr.method.unique()]:
        g = pr[pr.method == m].groupby("K")
        for ax, col in zip(axes, ["forgetting", "jaccard"]):
            mean, std = g[col].mean(), g[col].std()
            ls = "--" if m == "joint" else "-o"
            ax.plot(mean.index, mean.values, ls, color=colors.get(m, None),
                    label=m, lw=1.8, ms=4)
            ax.fill_between(mean.index, mean - std, mean + std,
                            color=colors.get(m, None), alpha=0.10)
    for ax, (col, ylab) in zip(axes, [("forgetting", "forgetting (bal. acc) ↓"),
                                      ("jaccard", "selection Jaccard ↑")]):
        ax.set_xscale("log", base=2)
        ks = sorted(pr.K.unique())
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("observation budget K")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Budget × forgetting interaction — {args.dataset}/{args.order}")
    fig.tight_layout()
    p1 = figure_dir / f"cl_budget_forgetting_{args.dataset}_{args.order}.png"
    fig.savefig(p1, dpi=160)
    print(f"fig -> {p1}")

    # ---- 圖 2：行為層保存（Jaccard / action-KL bars @ 最小 K） ----
    kmin = min(pr.K.unique())
    sub = pr[pr.K == kmin]
    ms = [m for m in METHOD_ORDER if m in sub.method.unique() and m != "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(ms))
    for ax, col, ylab in [(axes[0], "jaccard", "selection Jaccard ↑"),
                          (axes[1], "action_kl", "action-KL drift ↓")]:
        mean = [sub[sub.method == m][col].mean() for m in ms]
        std = [sub[sub.method == m][col].std() for m in ms]
        ax.bar(x, mean, yerr=std, capsize=4,
               color=[colors.get(m) for m in ms], alpha=0.8)
        ax.set_xticks(x, ms, rotation=20)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Behavior-level preservation @ K={kmin} — "
                 f"{args.dataset}/{args.order}")
    fig.tight_layout()
    p2 = figure_dir / f"cl_jaccard_bars_{args.dataset}_{args.order}.png"
    fig.savefig(p2, dpi=160)
    print(f"fig -> {p2}")


if __name__ == "__main__":
    main()
