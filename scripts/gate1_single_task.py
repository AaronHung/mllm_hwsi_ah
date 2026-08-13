"""Gate 1：causal pyramid environment 上的單任務 learned navigator。

判準（pre-registered）：learned navigator 在小預算 K 下優於 random / spatial-uniform。

流程（每折）：
  1. training slides 上以 random K-子集訓練 evaluator（診斷頭）。
  2. 凍結 evaluator，對 training slides 做 counterfactual-gain greedy 展開 → 教師軌跡。
  3. navigator 以 KL 對齊 softmax(gain/τ)（弱監督，無 region 標註）。
  4. test slides：learned / random×5 / spatial-uniform 各自選 K region → 同一 evaluator。

用法：
    python scripts/gate1_single_task.py --smoke     # Mac 快速煙測（1 fold, 1 seed）
    python scripts/gate1_single_task.py             # 完整（5-fold × 3 seeds × K∈{1,2,4,8}）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import get_device  # noqa: E402
from nav.engine import (build_bank, eval_policy, teacher_rollout,  # noqa: E402
                        train_evaluator, train_navigator)
from nav.pilot40 import CLASSES, Pilot40  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", choices=["4class", "2class"], default="4class")
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    args = ap.parse_args()

    device = get_device()
    print(f"device = {device}")
    ds = Pilot40(REPO / "data")
    sids = ds.slide_ids

    if args.task == "4class":
        classes = CLASSES
        label_of = {s: classes.index(ds.label4(s)) for s in sids}
    else:
        classes = ["Renal", "Breast"]
        label_of = {s: classes.index(ds.label2(s)) for s in sids}

    print("載入特徵 bank …")
    bank_all = build_bank(ds, sids, label_of)
    coords_of = {s: ds.coords(s) for s in sids}
    by_sid = {b.sid: b for b in bank_all}

    budgets = args.budgets or ([4] if args.smoke else [1, 2, 4, 8])
    seeds = [0] if args.smoke else [0, 1, 2]
    ev_epochs = 15 if args.smoke else 60
    nav_epochs = 8 if args.smoke else 30

    from sklearn.model_selection import StratifiedKFold

    rows = []
    t0 = time.time()
    y_all = np.array([label_of[s] for s in sids])

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(skf.split(sids, y_all))
        if args.smoke:
            folds = folds[:1]
        for fold_i, (tr, te) in enumerate(folds):
            torch.manual_seed(seed * 100 + fold_i)
            tr_bank = [by_sid[sids[i]] for i in tr]
            te_bank = [by_sid[sids[i]] for i in te]
            for k in budgets:
                evaluator = train_evaluator(tr_bank, len(classes), k, device,
                                            epochs=ev_epochs, seed=seed)
                steps = []
                for s in tr_bank:
                    steps += teacher_rollout(s, evaluator, k, device)
                navigator = train_navigator(steps, device, epochs=nav_epochs, seed=seed)

                acc_l, _ = eval_policy(te_bank, evaluator, k, device, "learned",
                                       navigator=navigator)
                acc_u, _ = eval_policy(te_bank, evaluator, k, device, "uniform",
                                       coords_of=coords_of)
                accs_r = [eval_policy(te_bank, evaluator, k, device, "random",
                                      seed=rs)[0] for rs in range(5)]
                # 教師上限：test slide 也用 counterfactual greedy（需要 label＝作弊，僅供參考）
                acc_t = 0.0
                for s in te_bank:
                    t_steps = teacher_rollout(s, evaluator, k, device)
                    sel = [int(st.candidates[st.gain.argmax()]) for st in t_steps]
                    pred = evaluator(s.high[sel].to(device).mean(0)[None]).argmax().item()
                    acc_t += int(pred == s.y)
                acc_t /= len(te_bank)

                rows.append(dict(task=args.task, seed=seed, fold=fold_i, K=k,
                                 learned=acc_l, uniform=acc_u,
                                 random_mean=float(np.mean(accs_r)),
                                 random_std=float(np.std(accs_r)),
                                 teacher_oracle=float(acc_t)))
                print(f"[seed{seed} fold{fold_i} K={k}] learned={acc_l:.3f} "
                      f"random={np.mean(accs_r):.3f}±{np.std(accs_r):.3f} "
                      f"uniform={acc_u:.3f} oracle={acc_t:.3f} "
                      f"({time.time() - t0:.0f}s)")

    import pandas as pd
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    df.to_csv(RESULTS / f"gate1_{args.task}_{tag}.csv", index=False)

    agg = df.groupby("K")[["learned", "random_mean", "uniform",
                           "teacher_oracle"]].agg(["mean", "std"])
    print(agg.round(3))

    if not args.smoke:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for col, color, label in [("learned", "tab:red", "learned navigator"),
                                  ("random_mean", "tab:gray", "random"),
                                  ("uniform", "tab:green", "spatial uniform"),
                                  ("teacher_oracle", "black", "counterfactual oracle")]:
            m = df.groupby("K")[col].mean()
            s = df.groupby("K")[col].std()
            ls = "--" if col == "teacher_oracle" else "-o"
            ax.plot(m.index, m.values, ls, color=color, label=label)
            if col != "teacher_oracle":
                ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("budget K")
        ax.set_ylabel("test accuracy (5-fold × 3 seeds)")
        ax.set_title(f"Gate 1: learned navigator vs baselines — {args.task}")
        ax.legend()
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"gate1_{args.task}.png", dpi=150)

        gate = (df.groupby("K")["learned"].mean()
                > df.groupby("K")["random_mean"].mean())
        verdict = "PASS" if gate.any() else "FAIL"
        (RESULTS / f"gate1_{args.task}_summary.md").write_text(
            f"# Gate 1 — {args.task}\n\n"
            f"{agg.round(3).to_markdown()}\n\n"
            f"**Gate 1（learned > random 至少一個 K）：{verdict}**\n"
            f"（判準細節與逐折數據見 gate1_{args.task}_{tag}.csv）\n")
        print(f"Gate 1 verdict: {verdict}")


if __name__ == "__main__":
    main()
