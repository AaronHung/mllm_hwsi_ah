"""
第一步:讀取 40 張 WSI 的四尺度特徵,做 mean pooling 得到「一張 WSI 一個代表向量」,
並整理 class / TSS(site)標籤,存成 npz + csv 供後續 UMAP / A-distance / MMD 使用。

Pooling 方式(依 CLAUDE.md 第4節要求,需明確標註):
- cell:   encoded_cell_features [num_region, 48, 384] -> 對 (region, cell) 兩維取 mean -> [384]
- patch:  patches_filtered.selected_features [num_region, 48, 512] -> 對 (region, patch) 兩維取 mean -> [512]
- region: region_4k [num_region, 192] -> 對 region 取 mean -> [192]
- wsi:    wsi [192] -> 本身就是單一向量,不需 pooling

這是 WSI 層級(mean pooling)版本。instance-level(不做 pooling,保留全部 region/patch/cell)
版本另外在 02 腳本處理。
"""
import csv
import json
import os

import numpy as np
import torch

from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]   # <repo>/analysis
REPO = ANALYSIS.parent                            # <repo> (MLLM-HWSI)
BASE = f"{REPO}/outputs/brca_pilot40_k48"
SELECTION_CSV = f"{ANALYSIS}/pilot40_selection.csv"
OUT_DIR = f"{ANALYSIS}/tables"

os.makedirs(OUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(SELECTION_CSV)))
print(f"讀取 pilot40_selection.csv: {len(rows)} 張 WSI")

records = []
feats = {"cell": [], "patch": [], "region": [], "wsi": []}

for r in rows:
    sid = r["slide_id"]
    cls = r["class"]
    tss = r["patient_id"].split("-")[1]

    cell = torch.load(f"{BASE}/cells_k48/{sid}/encoded_cell_features.pt",
                       map_location="cpu", weights_only=False)["encoded_cell_features"]
    patch = torch.load(f"{BASE}/features/patches_filtered/{sid}.pt",
                        map_location="cpu", weights_only=False)["selected_features"]
    region = torch.load(f"{BASE}/features/region_4k/{sid}.pt",
                         map_location="cpu", weights_only=False)
    wsi = torch.load(f"{BASE}/features/wsi/{sid}.pt",
                      map_location="cpu", weights_only=False)

    feats["cell"].append(cell.mean(dim=(0, 1)).numpy())
    feats["patch"].append(patch.mean(dim=(0, 1)).numpy())
    feats["region"].append(region.mean(dim=0).numpy())
    feats["wsi"].append(wsi.numpy())

    records.append({
        "slide_id": sid,
        "class": cls,
        "organ": "kidney" if cls in ("KIRC", "KIRP") else "breast",
        "tss": tss,
        "num_regions": cell.shape[0],
    })

# 標註每個亞型內的「多數 site」vs「其餘 site」二元域標籤(供 A-distance / MMD 使用)
from collections import Counter
by_class_tss = {}
for rec in records:
    by_class_tss.setdefault(rec["class"], Counter())[rec["tss"]] += 1

for rec in records:
    majority_tss, _ = by_class_tss[rec["class"]].most_common(1)[0]
    rec["majority_site_of_class"] = majority_tss
    rec["site_group"] = "majority" if rec["tss"] == majority_tss else "rest"

# 存標籤表
label_csv = f"{OUT_DIR}/wsi_labels.csv"
with open(label_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    w.writerows(records)
print(f"已存標籤表: {label_csv}")

# 存特徵 npz(WSI 層級, mean-pooled)
npz_path = f"{OUT_DIR}/wsi_level_features.npz"
save_dict = {f"{k}": np.stack(v) for k, v in feats.items()}
save_dict["slide_id"] = np.array([r["slide_id"] for r in records])
np.savez(npz_path, **save_dict)
print(f"已存 WSI 層級特徵: {npz_path}")

print("\n=== 各尺度 WSI 層級特徵矩陣 shape ===")
for k, v in feats.items():
    arr = np.stack(v)
    print(f"  {k:8s}: {arr.shape}")

print("\n=== 每亞型多數 site vs 其餘 拆分(A-distance/MMD 用的二元域標籤) ===")
for cls in sorted(by_class_tss.keys()):
    maj = by_class_tss[cls].most_common(1)[0]
    total = sum(by_class_tss[cls].values())
    print(f"  {cls}: 多數 site={maj[0]} (n={maj[1]}) vs 其餘 (n={total - maj[1]})")

print("\n=== class x organ x tss 樣本數健檢 ===")
for cls in sorted(by_class_tss.keys()):
    n = sum(1 for r in records if r["class"] == cls)
    print(f"  {cls}: {n} 張")
