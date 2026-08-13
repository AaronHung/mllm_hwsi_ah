"""實驗 B：預算限制下的區域選擇模擬（導航可行性）。

規格來源：background_prompt/09_任務B_給ClaudeCode.md
- 每張切片在預算 K 下選 K 個 region，slide 表示 = 被選 region 特徵 mean (192,)
- 策略：random(5 seeds) / spatial-uniform / feature-diversity(FPS, HPS 風格) /
        atypicality(離 training-fold 全域中心最遠) / full-information 上限
- 評估：logistic regression + LOO CV；K ∈ {1,2,4,8,16,32,all}
- 任務：二分類（Renal vs Breast）與四分類（KIRC/KIRP/IDC/ILC）

用法：
    python scripts/expB_budget_navigation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav.pilot40 import Pilot40  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"
BUDGETS = [1, 2, 4, 8, 16, 32, "all"]
CHANCE = {"2class": 0.50, "4class": 0.25}


# ---------------------------------------------------------------- policies
def select_random(feats, coords, k, rng):
    n = feats.shape[0]
    return rng.choice(n, size=min(k, n), replace=False)


def select_spatial_uniform(feats, coords, k, rng=None):
    """把切片 bounding box 均勻分格，每格取離格心最近的 region，湊滿 k。"""
    n = coords.shape[0]
    if k >= n:
        return np.arange(n)
    g = int(np.ceil(np.sqrt(k)))
    xy = coords.astype(np.float64)
    mn, mx = xy.min(0), xy.max(0)
    span = np.maximum(mx - mn, 1e-9)
    cell = np.minimum(((xy - mn) / span * g).astype(int), g - 1)
    chosen: list[int] = []
    for cx in range(g):
        for cy in range(g):
            if len(chosen) >= k:
                break
            mask = (cell[:, 0] == cx) & (cell[:, 1] == cy)
            idx = np.where(mask)[0]
            idx = idx[~np.isin(idx, chosen)]
            if len(idx) == 0:
                continue
            center = mn + (np.array([cx, cy]) + 0.5) / g * span
            d = np.linalg.norm(xy[idx] - center, axis=1)
            chosen.append(int(idx[np.argmin(d)]))
    # 空格不足 k 時：用最大最小距離（空間 FPS）補滿
    while len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), chosen)
        dmin = np.linalg.norm(xy[rest, None] - xy[chosen][None], axis=2).min(1)
        chosen.append(int(rest[np.argmax(dmin)]))
    return np.array(chosen[:k])


def select_diversity(feats, coords, k, rng):
    """HPS 風格 farthest-point sampling（cosine 空間，去冗餘留異質）。"""
    n = feats.shape[0]
    if k >= n:
        return np.arange(n)
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    chosen = [int(rng.integers(n))]
    max_sim = f @ f[chosen[0]]
    for _ in range(k - 1):
        max_sim[chosen] = np.inf
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, f @ f[nxt])
    return np.array(chosen)


def select_atypical(feats, coords, k, center):
    """離全域中心（僅由 training fold 計算）最遠的 k 個 region。"""
    n = feats.shape[0]
    if k >= n:
        return np.arange(n)
    d = np.linalg.norm(feats - center[None], axis=1)
    return np.argsort(-d)[:k]


# ---------------------------------------------------------------- evaluation
def loo_accuracy(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    preds = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneOut().split(X):
        m = Pipeline([("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=5000, C=1.0))])
        m.fit(X[tr], y[tr])
        preds[te[0]] = m.predict(X[te])[0]
    return accuracy_score(y, preds), balanced_accuracy_score(y, preds)


def loo_accuracy_atypical(region_feats, y, k):
    """atypicality 的中心必須只用 training fold 計算 → selection 在每折內重做。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    preds = np.empty(n, dtype=object)
    for tr, te in LeaveOneOut().split(np.arange(n)):
        center = np.concatenate([region_feats[i] for i in tr]).mean(0)
        X = np.stack([
            region_feats[i][select_atypical(region_feats[i], None, k, center)].mean(0)
            for i in range(n)])
        m = Pipeline([("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=5000, C=1.0))])
        m.fit(X[tr], y[tr])
        preds[te[0]] = m.predict(X[te])[0]
    return accuracy_score(y, preds), balanced_accuracy_score(y, preds)


# ---------------------------------------------------------------- main
def main():
    ds = Pilot40(REPO / "data")
    sids = ds.slide_ids
    region_feats = [ds.region(s).float().numpy() for s in sids]
    coords = [ds.coords(s) for s in sids]
    y4 = np.array([ds.label4(s) for s in sids])
    y2 = np.array([ds.label2(s) for s in sids])
    tasks = {"2class": y2, "4class": y4}

    # 盤點：coords 與 region 列數一致性
    mismatch = [(s, region_feats[i].shape[0], coords[i].shape[0])
                for i, s in enumerate(sids)
                if region_feats[i].shape[0] != coords[i].shape[0]]
    if mismatch:
        print("座標/特徵列數不一致，停止：", mismatch)
        sys.exit(1)
    print(f"盤點 OK：40 slides，R 範圍 "
          f"{min(f.shape[0] for f in region_feats)}–{max(f.shape[0] for f in region_feats)}")

    rows = []

    def record(task, policy, k, seed, acc, bacc):
        rows.append(dict(task=task, policy=policy, K=k, seed=seed,
                         loo_accuracy=round(acc, 4), balanced_accuracy=round(bacc, 4)))
        print(f"  {task} {policy:16s} K={str(k):>4s} seed={seed} acc={acc:.3f}")

    for task, y in tasks.items():
        print(f"== task {task} ==")
        # full-information 上限（= 全 region mean）
        X_full = np.stack([f.mean(0) for f in region_feats])
        acc, bacc = loo_accuracy(X_full, y)
        record(task, "full_info", "all", 0, acc, bacc)

        for k in [b for b in BUDGETS if b != "all"]:
            # random × 5 seeds
            for seed in range(5):
                rng = np.random.default_rng(seed)
                X = np.stack([f[select_random(f, c, k, rng)].mean(0)
                              for f, c in zip(region_feats, coords)])
                acc, bacc = loo_accuracy(X, y)
                record(task, "random", k, seed, acc, bacc)
            # spatial-uniform（決定性）
            X = np.stack([f[select_spatial_uniform(f, c, k)].mean(0)
                          for f, c in zip(region_feats, coords)])
            acc, bacc = loo_accuracy(X, y)
            record(task, "spatial_uniform", k, 0, acc, bacc)
            # feature-diversity FPS（3 個起始 seed）
            for seed in range(3):
                rng = np.random.default_rng(seed)
                X = np.stack([f[select_diversity(f, c, k, rng)].mean(0)
                              for f, c in zip(region_feats, coords)])
                acc, bacc = loo_accuracy(X, y)
                record(task, "diversity_fps", k, seed, acc, bacc)
            # atypicality（fold 內算中心）
            acc, bacc = loo_accuracy_atypical(region_feats, y, k)
            record(task, "atypicality", k, 0, acc, bacc)

    import pandas as pd
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "expB_budget_curves.csv", index=False)

    # ---------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [b for b in BUDGETS if b != "all"]
    style = {"random": ("tab:gray", "random"),
             "spatial_uniform": ("tab:green", "spatial uniform"),
             "diversity_fps": ("tab:blue", "diversity (HPS-style FPS)"),
             "atypicality": ("tab:red", "atypicality-guided")}
    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = df[df.task == task]
        for pol, (color, label) in style.items():
            m = sub[sub.policy == pol].groupby("K").loo_accuracy.agg(["mean", "std"])
            m = m.reindex(ks)
            ax.plot(ks, m["mean"], "-o", color=color, label=label)
            if pol in ("random", "diversity_fps"):
                ax.fill_between(ks, m["mean"] - m["std"].fillna(0),
                                m["mean"] + m["std"].fillna(0), color=color, alpha=0.15)
        full = sub[sub.policy == "full_info"].loo_accuracy.iloc[0]
        ax.axhline(full, color="black", ls="--", lw=1,
                   label=f"full information ({full:.2f})")
        ax.axhline(CHANCE[task], color="lightgray", ls=":", lw=1,
                   label=f"chance ({CHANCE[task]:.2f})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("observation budget K (regions)")
        ax.set_ylabel("LOO accuracy")
        ax.set_title(f"Exp B: accuracy vs budget — {task} (pilot-40, exploratory)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"expB_budget_curve_{task}.png", dpi=150)

    # ---- navigation map 範例：KIRC 與 IDC 各一張，diversity K=8 ----
    examples = [next(s for s in sids if ds.label4(s) == "KIRC"),
                next(s for s in sids if ds.label4(s) == "IDC")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, sid in zip(axes, examples):
        i = sids.index(sid)
        f, c = region_feats[i], coords[i]
        sel = select_diversity(f, c, 8, np.random.default_rng(0))
        ax.scatter(c[:, 0], c[:, 1], s=12, c="lightgray", label="all regions")
        ax.scatter(c[sel, 0], c[sel, 1], s=60, c="red", marker="o",
                   label="selected (diversity, K=8)")
        ax.invert_yaxis()
        ax.set_title(f"{sid}\n({ds.label4(sid)})", fontsize=9)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    fig.suptitle("Exp B: where does the navigator go? (region coordinates)")
    fig.tight_layout()
    fig.savefig(FIGURES / "expB_navigation_map_examples.png", dpi=150)

    # ---------------------------------------------------------------- summary
    lines = ["# 實驗 B 結論（探索性結果；n=40 pilot，seed 已固定）", ""]
    for task in tasks:
        sub = df[df.task == task]
        small = sub[sub.K.isin([1, 2, 4, 8])]
        rnd = small[small.policy == "random"].groupby("K").loo_accuracy.mean()
        best_rows = []
        for k in [1, 2, 4, 8]:
            cand = small[(small.K == k) & (small.policy != "random")]
            pol_mean = cand.groupby("policy").loo_accuracy.mean()
            bp = pol_mean.idxmax()
            gain = pol_mean.max() - rnd[k]
            best_rows.append((k, bp, pol_mean.max(), rnd[k], gain))
        lines.append(f"## {task}")
        for k, bp, b, r, g in best_rows:
            lines.append(f"- K={k}: 最佳策略 {bp} = {b:.1%}，random = {r:.1%}，"
                         f"領先 {g * 100:+.1f} pp")
        avg_gain = np.mean([g for *_, g in best_rows])
        lines.append(f"- 小預算（K≤8）平均領先 random：{avg_gain * 100:+.1f} pp")
        lines.append("")
    (RESULTS / "expB_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
