"""Gate 1 顯著性加固：learned navigator 優勢的統計檢定（protocol §5、G1'）。

Part A  pilot40（4class）：5 seeds × 5 stratified folds × K∈{1,2,4}
        baselines：random（5 draws）、spatial-uniform、counterfactual oracle
        配對單位 =（seed, fold）；paired bootstrap 95% CI + Wilcoxon signed-rank
Part B  can_dataset 單任務（esca 與 rcc）：5 seeds、固定 fold_1 split
        baseline：random（5 draws；can 無座標 → 無 uniform）
        配對單位 =（seed, slide）的 per-slide correctness

輸出：results/gate1_hardened.csv、results/gate1_significance.md、figures/gate1_ci.png

用法：
    python scripts/gate1_significance.py --smoke
    python scripts/gate1_significance.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import get_device  # noqa: E402
from nav.candata import CanTask  # noqa: E402
from nav.engine import (build_bank, eval_policy, evidence_of,  # noqa: E402
                        rollout_policy, teacher_rollout, train_evaluator,
                        train_navigator)
from nav.pilot40 import CLASSES, Pilot40  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"
CACHE_ROOT = REPO / "data" / "can_cache"


@torch.no_grad()
def per_slide_correct(bank, evaluator, k, device, policy, navigator=None,
                      seed=0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for s in bank:
        sel = rollout_policy(s, k, device, policy, navigator=navigator, rng=rng)
        pred = evaluator(evidence_of(s, sel, device)[None]).argmax().item()
        out[s.sid] = float(pred == s.y)
    return out


def boot_ci(diffs: np.ndarray, n_boot=10000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def part_a_pilot(budgets, seeds, device, ev_epochs, nav_epochs, n_folds):
    from sklearn.model_selection import StratifiedKFold
    ds = Pilot40(REPO / "data")
    sids = ds.slide_ids
    label_of = {s: CLASSES.index(ds.label4(s)) for s in sids}
    bank_all = build_bank(ds, sids, label_of)
    by_sid = {b.sid: b for b in bank_all}
    coords_of = {s: ds.coords(s) for s in sids}
    y_all = np.array([label_of[s] for s in sids])

    rows = []
    t0 = time.time()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_i, (tr, te) in enumerate(skf.split(sids, y_all)):
            torch.manual_seed(seed * 100 + fold_i)
            tr_bank = [by_sid[sids[i]] for i in tr]
            te_bank = [by_sid[sids[i]] for i in te]
            for k in budgets:
                ev = train_evaluator(tr_bank, len(CLASSES), k, device,
                                     epochs=ev_epochs, seed=seed)
                steps = []
                for s in tr_bank:
                    steps += teacher_rollout(s, ev, k, device)
                nav = train_navigator(steps, device, epochs=nav_epochs,
                                      seed=seed)
                acc_l, _ = eval_policy(te_bank, ev, k, device, "learned",
                                       navigator=nav)
                acc_u, _ = eval_policy(te_bank, ev, k, device, "uniform",
                                       coords_of=coords_of)
                acc_r = float(np.mean([eval_policy(te_bank, ev, k, device,
                                                   "random", seed=rs)[0]
                                       for rs in range(5)]))
                rows.append(dict(dataset="pilot40", seed=seed, fold=fold_i,
                                 K=k, learned=acc_l, random=acc_r,
                                 uniform=acc_u))
                print(f"[pilot40 seed{seed} fold{fold_i} K={k}] "
                      f"learned={acc_l:.3f} random={acc_r:.3f} "
                      f"uniform={acc_u:.3f} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def part_b_can(cohorts, budgets, seeds, device, ev_epochs, nav_epochs, smoke):
    rows = []
    t0 = time.time()
    for cohort in cohorts:
        task = CanTask(cohort, CACHE_ROOT)
        tr_bank = task.load_bank("train")
        te_bank = task.load_bank("test")
        if smoke:
            tr_bank, te_bank = tr_bank[:24], te_bank[:16]
        n_cls = len(task.classes)
        for seed in seeds:
            torch.manual_seed(seed)
            for k in budgets:
                ev = train_evaluator(tr_bank, n_cls, k, device,
                                     epochs=ev_epochs, seed=seed)
                steps = []
                for s in tr_bank:
                    steps += teacher_rollout(s, ev, k, device)
                nav = train_navigator(steps, device, epochs=nav_epochs,
                                      seed=seed)
                corr_l = per_slide_correct(te_bank, ev, k, device, "learned",
                                           navigator=nav)
                corr_r = {sid: 0.0 for sid in corr_l}
                for rs in range(5):
                    cr = per_slide_correct(te_bank, ev, k, device, "random",
                                           seed=seed * 100 + rs)
                    for sid in cr:
                        corr_r[sid] += cr[sid] / 5
                for sid in corr_l:
                    rows.append(dict(dataset=f"can_{cohort}", seed=seed, K=k,
                                     sid=sid, learned=corr_l[sid],
                                     random=corr_r[sid]))
                print(f"[can_{cohort} seed{seed} K={k}] "
                      f"learned={np.mean(list(corr_l.values())):.3f} "
                      f"random={np.mean(list(corr_r.values())):.3f} "
                      f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def analyze(df_a: pd.DataFrame, df_b: pd.DataFrame, budgets) -> str:
    from scipy.stats import wilcoxon
    lines = ["# Gate 1 顯著性分析（protocol G1'）", ""]
    stats_rows = []

    lines.append("## pilot40（配對單位 = seed × fold）\n")
    lines.append("| K | learned−random | 95% CI | Wilcoxon p | learned−uniform | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in budgets:
        d = df_a[df_a.K == k]
        dr = (d.learned - d.random).values
        du = (d.learned - d.uniform).values
        cir, ciu = boot_ci(dr), boot_ci(du)
        pr = wilcoxon(dr).pvalue if np.abs(dr).sum() > 0 else 1.0
        pu = wilcoxon(du).pvalue if np.abs(du).sum() > 0 else 1.0
        lines.append(f"| {k} | {dr.mean():+.3f} | [{cir[0]:+.3f}, {cir[1]:+.3f}] "
                     f"| {pr:.4f} | {du.mean():+.3f} "
                     f"| [{ciu[0]:+.3f}, {ciu[1]:+.3f}] | {pu:.4f} |")
        stats_rows.append(dict(dataset="pilot40", K=k, diff="learned-random",
                               mean=dr.mean(), lo=cir[0], hi=cir[1], p=pr))
        stats_rows.append(dict(dataset="pilot40", K=k, diff="learned-uniform",
                               mean=du.mean(), lo=ciu[0], hi=ciu[1], p=pu))

    for name in sorted(df_b.dataset.unique()):
        lines.append(f"\n## {name}（配對單位 = seed × slide）\n")
        lines.append("| K | learned−random | 95% CI | Wilcoxon p |")
        lines.append("|---|---|---|---|")
        for k in budgets:
            d = df_b[(df_b.dataset == name) & (df_b.K == k)]
            dr = (d.learned - d.random).values
            ci = boot_ci(dr)
            p = wilcoxon(dr).pvalue if np.abs(dr).sum() > 0 else 1.0
            lines.append(f"| {k} | {dr.mean():+.3f} "
                         f"| [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.4f} |")
            stats_rows.append(dict(dataset=name, K=k, diff="learned-random",
                                   mean=dr.mean(), lo=ci[0], hi=ci[1], p=p))

    st = pd.DataFrame(stats_rows)
    n_pos = int(((st["diff"] == "learned-random") & (st["lo"] > 0)).sum())
    lines.append(f"\n**G1' 判準（learned−random 的 95% CI > 0，至少 2 個 K）："
                 f"CI>0 的設定數 = {n_pos} → "
                 f"{'PASS' if n_pos >= 2 else 'FAIL'}**\n")

    # 圖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = ["pilot40"] + sorted(df_b.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, datasets):
        sub = st[(st.dataset == name) & (st["diff"] == "learned-random")]
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"], width=0.55, color="tab:red", alpha=0.75)
        ax.errorbar(x, sub["mean"],
                    yerr=[sub["mean"] - sub.lo, sub.hi - sub["mean"]],
                    fmt="none", ecolor="black", capsize=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, [f"K={k}" for k in sub.K])
        ax.set_title(f"{name}\nlearned − random (95% CI)")
        ax.set_ylabel("accuracy difference")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "gate1_ci.png", dpi=160)
    st.to_csv(RESULTS / "gate1_hardened_stats.csv", index=False)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--cohorts", nargs="+", default=["esca", "rcc"])
    args = ap.parse_args()

    device = get_device()
    print(f"device = {device}")
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    budgets = [2] if args.smoke else args.budgets
    ev_p, nav_p = (15, 8) if args.smoke else (60, 30)      # pilot40
    ev_c, nav_c = (5, 2) if args.smoke else (30, 10)       # can
    n_folds = 5

    df_a = part_a_pilot(budgets, seeds, device, ev_p, nav_p, n_folds)
    df_b = part_b_can(args.cohorts, budgets, seeds, device, ev_c, nav_c,
                      args.smoke)
    RESULTS.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    pd.concat([df_a.assign(kind="pilot_folds"),
               df_b.assign(kind="can_slides")]).to_csv(
        RESULTS / f"gate1_hardened_{tag}.csv", index=False)

    if not args.smoke:
        md = analyze(df_a, df_b, budgets)
        (RESULTS / "gate1_significance.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
