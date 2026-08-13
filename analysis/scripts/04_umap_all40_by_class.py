"""
把 40 張 WSI 全部放進同一個 UMAP 空間(不再依亞型切開),依「亞型」上色,
檢查四個尺度是否都能把 KIRC/KIRP/IDC/ILC 分成四群。
這是類別可分性的 sanity check(不是路線 A 的域偏移分析,是額外確認)。

固定 random_state=42、n_neighbors=10,並標註於圖上。
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
N_NEIGHBORS = 10

labels = pd.read_csv(f"{TABLES}/wsi_labels.csv")
npz = np.load(f"{TABLES}/wsi_level_features.npz", allow_pickle=True)
assert list(npz["slide_id"]) == list(labels["slide_id"])

SCALES = ["cell", "patch", "region", "wsi"]
CLASS_COLORS = {"KIRC": "tab:red", "KIRP": "tab:orange", "IDC": "tab:blue", "ILC": "tab:cyan"}
ORGAN_MARKERS = {"kidney": "o", "breast": "^"}

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
fig.suptitle(
    f"全部 40 張 WSI 同一 UMAP 空間,依亞型上色 (random_state={RANDOM_STATE}, "
    f"n_neighbors={N_NEIGHBORS}, WSI-level mean-pooled, n=40)",
    fontsize=12,
)

for ax, scale in zip(axes.ravel(), SCALES):
    X = npz[scale]
    reducer = UMAP(n_components=2, n_neighbors=N_NEIGHBORS, min_dist=0.3,
                    random_state=RANDOM_STATE, metric="euclidean")
    emb = reducer.fit_transform(X)

    for cls in ["KIRC", "KIRP", "IDC", "ILC"]:
        mask = (labels["class"] == cls).values
        organ = "kidney" if cls in ("KIRC", "KIRP") else "breast"
        ax.scatter(emb[mask, 0], emb[mask, 1], label=f"{cls} (n={mask.sum()})",
                   color=CLASS_COLORS[cls], marker=ORGAN_MARKERS[organ],
                   s=90, edgecolor="k", linewidth=0.5)

    title = scale if scale != "wsi" else "wsi (= mean(region), 冗餘)"
    ax.set_title(f"尺度: {title}")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{FIGS}/class_sanitycheck_umap_all40_by_class.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"已存: {out_path}")
