"""實驗 A：四層病理特徵的 linear probe（問題定位）。

規格來源：background_prompt/08_任務A_給ClaudeCode.md
- 四層 slide-level 表示（wsi / region-mean / patch-mean / cell-mean）
- 任務：二分類（Renal vs Breast）與四分類（KIRC/KIRP/IDC/ILC）
- 分類器：Logistic regression（C 掃 {0.01,0.1,1,10}，inner 3-fold 選）、kNN（k=3,5）
- 驗證：Leave-one-out CV（主要）；Stratified 5-fold × 3 seeds（mean±std）
- 對照：frozen MLLM（二分類固定 55%、四分類固定 20%）

用法：
    python scripts/expA_linear_probe.py --inventory-only   # 第 0 步盤點
    python scripts/expA_linear_probe.py                    # 全部跑
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import add_device_argument, device_info, resolve_device  # noqa: E402
from nav.pilot40 import CLASSES, Pilot40  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"
FROZEN_MLLM_REF = {"2class": 0.55, "4class": 0.20}
CHANCE = {"2class": 0.50, "4class": 0.25}


# ---------------------------------------------------------------- inventory
def run_inventory(ds: Pilot40, results_dir: Path = RESULTS) -> bool:
    lines = ["# 實驗 A 第 0 步：資料盤點", ""]
    ok = True

    counts = Counter(ds.labels.values())
    lines += [f"- slide 總數：{len(ds.slide_ids)}",
              f"- 類別分佈：{dict(counts)}", ""]
    if len(ds.slide_ids) != 40 or any(counts[c] != 10 for c in CLASSES):
        ok = False
        lines.append("**異常：不是每類 10 張 / 共 40 張**")

    rng = np.random.default_rng(0)
    sample = list(rng.choice(ds.slide_ids, 2, replace=False))
    lines.append("## 抽樣檢查（每層 2 檔）")
    for sid in sample:
        w, r, p, c = ds.wsi(sid), ds.region(sid), ds.patch(sid), ds.cell(sid)
        xy = ds.coords(sid)
        zero_cell = float((c.abs().sum(-1) == 0).float().mean())
        lines += [f"### {sid}（{ds.label4(sid)}）",
                  f"- wsi {tuple(w.shape)} {w.dtype} | region {tuple(r.shape)} | "
                  f"patch {tuple(p.shape)} | cell {tuple(c.shape)} | coords {xy.shape}",
                  f"- zero-cell slots 比例：{zero_cell:.1%}",
                  f"- wsi == region.mean 最大差：{(w - r.mean(0)).abs().max().item():.2e}", ""]
        if r.shape[0] != p.shape[0] or r.shape[0] != c.shape[0] or r.shape[0] != xy.shape[0]:
            ok = False
            lines.append(f"**異常：{sid} 各層 R 不一致**")

    lines.append("## 交叉比對（40 張在四層 + coords 是否齊）")
    missing = []
    total_regions = 0
    for sid in ds.slide_ids:
        for name, fn in [("wsi", ds.wsi), ("region", ds.region),
                         ("patch", ds.patch), ("cell", ds.cell)]:
            try:
                t = fn(sid)
                if name == "region":
                    total_regions += t.shape[0]
            except Exception as e:  # noqa: BLE001
                missing.append(f"{sid}/{name}: {e}")
        try:
            ds.coords(sid)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{sid}/coords: {e}")
    lines.append(f"- 總 region 數：{total_regions}（預期 7656）")
    if missing:
        ok = False
        lines += ["- 缺漏："] + [f"  - {m}" for m in missing]
    else:
        lines.append("- 缺漏：無，40/40 五層齊全")

    lines += ["", f"**盤點結論：{'PASS' if ok else 'FAIL'}**"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "expA_inventory.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return ok


# ---------------------------------------------------------------- features
def build_slide_features(ds: Pilot40) -> dict[str, np.ndarray]:
    feats = {"wsi": [], "region": [], "patch": [], "cell": []}
    for sid in ds.slide_ids:
        feats["wsi"].append(ds.wsi(sid).float().numpy())
        feats["region"].append(ds.region(sid).float().mean(0).numpy())
        feats["patch"].append(ds.patch(sid).float().mean(dim=(0, 1)).numpy())
        feats["cell"].append(ds.cell(sid).float().mean(dim=(0, 1)).numpy())
    return {k: np.stack(v) for k, v in feats.items()}


# ---------------------------------------------------------------- models
def make_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def logreg():
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000))])
        return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=3, n_jobs=1)

    def knn(k):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=k))])

    return {"logreg": logreg, "knn3": lambda: knn(3), "knn5": lambda: knn(5)}


def loo_eval(make_model, X, y):
    from sklearn.model_selection import LeaveOneOut
    preds = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneOut().split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[te[0]] = m.predict(X[te])[0]
    return preds


def kfold_eval(make_model, X, y, seeds=(0, 1, 2)):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    accs, baccs = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            m = make_model()
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        accs.append(accuracy_score(y, p))
        baccs.append(balanced_accuracy_score(y, p))
    return np.mean(accs), np.std(accs), np.mean(baccs), np.std(baccs)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "data"))
    ap.add_argument("--inventory-only", action="store_true")
    add_device_argument(ap)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new-run directory; defaults to runs/v2/expA")
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    print(f"resolved device = {device} (sklearn probe executes on CPU)")
    ds = Pilot40(args.data_root)
    out_dir = (Path(args.output_dir) if args.output_dir is not None
               else REPO / "runs" / "v2" / "expA")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    ok = run_inventory(ds, out_dir)
    if args.inventory_only:
        sys.exit(0 if ok else 1)
    if not ok:
        print("盤點 FAIL，停止。", file=sys.stderr)
        sys.exit(1)

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, recall_score)

    X_by_level = build_slide_features(ds)
    y4 = np.array([ds.label4(s) for s in ds.slide_ids])
    y2 = np.array([ds.label2(s) for s in ds.slide_ids])
    tasks = {"2class": y2, "4class": y4}
    clfs = make_classifiers()

    rows = []
    loo_preds_store = {}
    for level, X in X_by_level.items():
        for task, y in tasks.items():
            labels_order = sorted(set(y))
            for clf_name, make_model in clfs.items():
                preds = loo_eval(make_model, X, y)
                loo_preds_store[(level, task, clf_name)] = preds
                acc = accuracy_score(y, preds)
                bacc = balanced_accuracy_score(y, preds)
                rec = recall_score(y, preds, labels=labels_order, average=None)
                rows.append(dict(feature_level=level, task=task, classifier=clf_name,
                                 cv_type="loo", accuracy=round(acc, 4),
                                 balanced_accuracy=round(bacc, 4),
                                 per_class_recall="|".join(
                                     f"{c}:{r:.2f}" for c, r in zip(labels_order, rec)),
                                 resolved_device=str(device),
                                 torch_version=meta["torch_version"]))
                am, asd, bm, bsd = kfold_eval(make_model, X, y)
                rows.append(dict(feature_level=level, task=task, classifier=clf_name,
                                 cv_type="5fold_x3seed", accuracy=round(am, 4),
                                 balanced_accuracy=round(bm, 4),
                                 per_class_recall=f"acc_std:{asd:.3f}|bacc_std:{bsd:.3f}",
                                 resolved_device=str(device),
                                 torch_version=meta["torch_version"]))
                print(f"[{level:6s}] {task} {clf_name:6s} "
                      f"LOO acc={acc:.3f} bacc={bacc:.3f} | 5f={am:.3f}±{asd:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "expA_probe_results.csv", index=False)

    # ---- 主圖：各層 LOO accuracy（取該層最佳分類器） ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loo = df[df.cv_type == "loo"]
    best = (loo.sort_values("accuracy", ascending=False)
            .groupby(["feature_level", "task"]).first().reset_index())
    levels = ["wsi", "region", "patch", "cell"]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    xs = np.arange(len(levels))
    for i, task in enumerate(["2class", "4class"]):
        vals = [best[(best.feature_level == lv) & (best.task == task)].accuracy.iloc[0]
                for lv in levels]
        bars = ax.bar(xs + (i - 0.5) * w, vals, w,
                      label=f"{task} probe (best clf)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9)
    ax.axhline(FROZEN_MLLM_REF["2class"], color="tab:blue", ls="--", lw=1,
               label="frozen MLLM 2class (0.55)")
    ax.axhline(FROZEN_MLLM_REF["4class"], color="tab:orange", ls="--", lw=1,
               label="frozen MLLM 4class (0.20)")
    ax.axhline(CHANCE["2class"], color="gray", ls=":", lw=1, label="chance 2c (0.50)")
    ax.axhline(CHANCE["4class"], color="lightgray", ls=":", lw=1, label="chance 4c (0.25)")
    ax.set_xticks(xs, levels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("LOO accuracy")
    ax.set_title("Exp A: linear probe by feature level (pilot-40, exploratory)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "expA_main_bar.png", dpi=150)

    # ---- 四分類最佳層的 confusion matrix ----
    best4 = best[best.task == "4class"].sort_values("accuracy", ascending=False).iloc[0]
    preds = loo_preds_store[(best4.feature_level, "4class", best4.classifier)]
    cm = confusion_matrix(y4, preds, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Exp A confusion: 4class, {best4.feature_level}/"
                 f"{best4.classifier} (LOO acc={best4.accuracy:.2f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figure_dir / f"expA_confusion_{best4.feature_level}.png", dpi=150)

    # ---- summary ----
    b2 = best[best.task == "2class"].sort_values("accuracy", ascending=False).iloc[0]
    verdict = ("特徵含有子型訊號，問題定位在 projector/LLM 對齊層（Stage-I 缺口）"
               if best4.accuracy > 0.45 else
               "probe 也接近 chance → 特徵層本身訊號不足")
    summary = f"""# 實驗 A 結論（探索性結果，非臨床效能宣稱；n=40 pilot）

- 二分類最佳 probe：{b2.feature_level}/{b2.classifier}，LOO accuracy = {b2.accuracy:.1%}
  （frozen MLLM 固定 prompt 為 55.0%，chance 50%）。
- 四分類最佳 probe：{best4.feature_level}/{best4.classifier}，LOO accuracy = {best4.accuracy:.1%}
  （frozen MLLM 固定 prompt 為 20.0%，chance 25%）。
- 判讀：四分類 probe 最佳 = {best4.accuracy:.0%}，對照 frozen MLLM 的 20% → **{verdict}**。
- 注意：wsi 層 = region 層特徵的算術平均，兩者非獨立證據；cell 層 CCAF 使用
  未經病理微調的 DINOv2 權重，解讀需保留。
- 所有隨機過程 seed 固定（inventory seed=0；5-fold seeds=0,1,2）。
"""
    (out_dir / "expA_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
