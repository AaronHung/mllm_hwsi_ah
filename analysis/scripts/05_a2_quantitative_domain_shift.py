"""
A.2 定量分析:對每個亞型、每個尺度,計算 site(域)之間的
- proxy A-distance(域分類器準確率轉換而來,越大代表越容易被分類器認出是哪個 site → 偏移越大)
- MMD^2(兩組分布的核方法距離,越大代表兩組分布差越多)

同時算 WSI-level(n=10/亞型,mean-pooled)跟 instance-level(每個 4096px region 一筆,
group-aware CV 避免同一張 WSI 的 region 同時出現在 train/test)兩種顆粒度。
KIRP 額外算一組樣本數對稱的 2Z(n=3) vs BQ(n=3) 版本。

⚠️ 小樣本 proof-of-concept:WSI-level 每組只有個位數樣本,統計檢定力非常有限。
"""
import csv
import sys

import numpy as np
import pandas as pd

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ANALYSIS = SCRIPTS.parent                         # <repo>/analysis
sys.path.insert(0, str(SCRIPTS))
from domain_shift_utils import mmd_rbf, proxy_a_distance_instance_level, proxy_a_distance_wsi_level

TABLES = f"{ANALYSIS}/tables"

wsi_labels = pd.read_csv(f"{TABLES}/wsi_labels.csv")
wsi_npz = np.load(f"{TABLES}/wsi_level_features.npz", allow_pickle=True)
assert list(wsi_npz["slide_id"]) == list(wsi_labels["slide_id"])

inst_meta = pd.read_csv(f"{TABLES}/instance_level_meta.csv")
inst_npz = np.load(f"{TABLES}/instance_level_features.npz", allow_pickle=True)

SCALES_WSI = ["cell", "patch", "region", "wsi"]
SCALES_INSTANCE = ["cell", "patch", "region"]
SUBTYPES = ["KIRC", "KIRP", "IDC", "ILC"]
MMD_SUBSAMPLE_CAP = 800
RNG_SEED = 0

rows = []


def subsample(X, cap, seed):
    if X.shape[0] <= cap:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], cap, replace=False)
    return X[idx]


def run_one(class_name, scale, level, X, y, groups, note=""):
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if level == "wsi":
        res = proxy_a_distance_wsi_level(X, y, n_splits=3, n_repeats=30, random_state=RNG_SEED)
    else:
        res = proxy_a_distance_instance_level(X, y, groups, n_splits=3, n_repeats=30, random_state=RNG_SEED)

    Xa = subsample(X[y == 1], MMD_SUBSAMPLE_CAP, RNG_SEED)
    Xb = subsample(X[y == 0], MMD_SUBSAMPLE_CAP, RNG_SEED)
    mmd2 = mmd_rbf(Xa, Xb)

    row = {
        "class": class_name, "scale": scale, "level": level,
        "n_group_majority_or_A": n1, "n_group_rest_or_B": n0,
        "acc_mean": round(res["acc_mean"], 4), "acc_std": round(res["acc_std"], 4),
        "a_distance_mean": round(res["a_distance_mean"], 4),
        "a_distance_std": round(res["a_distance_std"], 4),
        "mmd2": round(mmd2, 6),
        "note": note,
    }
    rows.append(row)
    print(f"[{class_name:5s}][{scale:6s}][{level:8s}] n={n1}v{n0}  "
          f"acc={res['acc_mean']:.3f}±{res['acc_std']:.3f}  "
          f"A-dist={res['a_distance_mean']:.3f}±{res['a_distance_std']:.3f}  "
          f"MMD^2={mmd2:.4f}  {note}")


print("=== WSI-level (mean-pooled, n=10/亞型, 多數site vs 其餘) ===")
for cls in SUBTYPES:
    mask = (wsi_labels["class"] == cls).values
    y = (wsi_labels.loc[mask, "site_group"] == "majority").astype(int).values
    for scale in SCALES_WSI:
        X = wsi_npz[scale][mask]
        note = "wsi=mean(region),冗餘" if scale == "wsi" else ""
        run_one(cls, scale, "wsi", X, y, None, note)

print("\n=== instance-level (每個 4096px region 一筆, group-CV by WSI, 多數site vs 其餘) ===")
for cls in SUBTYPES:
    mask = (inst_meta["class"] == cls).values
    y = (inst_meta.loc[mask, "site_group"] == "majority").astype(int).values
    groups = inst_meta.loc[mask, "slide_id"].values
    for scale in SCALES_INSTANCE:
        X = inst_npz[scale][mask]
        run_one(cls, scale, "instance", X, y, groups)

print("\n=== KIRP 平衡版 sanity check: 2Z(n=3) vs BQ(n=3) ===")
kirp_mask_wsi = (wsi_labels["class"] == "KIRP") & (wsi_labels["tss"].isin(["2Z", "BQ"]))
kirp_mask_wsi = kirp_mask_wsi.values
y_bal_wsi = (wsi_labels.loc[kirp_mask_wsi, "tss"] == "2Z").astype(int).values
for scale in SCALES_WSI:
    X = wsi_npz[scale][kirp_mask_wsi]
    note = "KIRP平衡版(2Z vs BQ)" + (",wsi=mean(region)冗餘" if scale == "wsi" else "")
    run_one("KIRP_balanced", scale, "wsi", X, y_bal_wsi, None, note)

kirp_mask_inst = (inst_meta["class"] == "KIRP") & (inst_meta["tss"].isin(["2Z", "BQ"]))
kirp_mask_inst = kirp_mask_inst.values
y_bal_inst = (inst_meta.loc[kirp_mask_inst, "tss"] == "2Z").astype(int).values
groups_bal_inst = inst_meta.loc[kirp_mask_inst, "slide_id"].values
for scale in SCALES_INSTANCE:
    X = inst_npz[scale][kirp_mask_inst]
    run_one("KIRP_balanced", scale, "instance", X, y_bal_inst, groups_bal_inst, "KIRP平衡版(2Z vs BQ)")

out_csv = f"{TABLES}/A2_domain_shift_quantitative.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已存: {out_csv}")
