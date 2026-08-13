"""
A.1 定性分析:四個尺度各一張圖,每張圖用 2x2 子圖分別呈現 KIRC/KIRP/IDC/ILC
四個亞型「固定亞型內、依 source site 上色」的 UMAP。

⚠️ 小樣本警告:每個亞型僅 10 張 WSI,UMAP random_state 固定為 42,
n_neighbors 依樣本數自動縮小(min(5, n-1)),結果僅供 proof-of-concept 參考,
不具統計檢定力,換種子/換參數可能改變圖形結構。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import plot_config  # noqa: F401  (設定 CJK 字型)
import matplotlib.pyplot as plt
from umap import UMAP

from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]   # <repo>/analysis
TABLES = f"{ANALYSIS}/tables"
FIGS = f"{ANALYSIS}/figures"
RANDOM_STATE = 42

labels = pd.read_csv(f"{TABLES}/wsi_labels.csv")
npz = np.load(f"{TABLES}/wsi_level_features.npz", allow_pickle=True)
assert list(npz["slide_id"]) == list(labels["slide_id"]), "順序需與標籤表一致"

SCALES = ["cell", "patch", "region", "wsi"]
SUBTYPES = ["KIRC", "KIRP", "IDC", "ILC"]

for scale in SCALES:
    X_all = npz[scale]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(
        f"尺度: {scale} — 固定亞型內依 source site 上色的 UMAP "
        f"(random_state={RANDOM_STATE}, n=10/亞型, proof-of-concept)",
        fontsize=12,
    )

    for ax, cls in zip(axes.ravel(), SUBTYPES):
        mask = (labels["class"] == cls).values
        X = X_all[mask]
        sites = labels.loc[mask, "tss"].values
        n = X.shape[0]
        n_neighbors = max(2, min(5, n - 1))

        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
                        random_state=RANDOM_STATE, metric="euclidean")
        emb = reducer.fit_transform(X)

        uniq_sites = sorted(set(sites))
        cmap = plt.get_cmap("tab10")
        for i, s in enumerate(uniq_sites):
            m = sites == s
            ax.scatter(emb[m, 0], emb[m, 1], label=f"{s} (n={m.sum()})",
                       color=cmap(i % 10), s=80, edgecolor="k", linewidth=0.5)

        ax.set_title(f"{cls} (n_neighbors={n_neighbors})")
        ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{FIGS}/A1_umap_{scale}_by_site.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已存: {out_path}")
