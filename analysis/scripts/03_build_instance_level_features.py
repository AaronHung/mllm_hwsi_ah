"""
建立 instance-level 特徵:每個 instance = 一個 4096px region,
cell/patch 尺度在該 region 內對 48 個 token 取 mean,得到「每個 region 一個向量」,
與 region 尺度(HIPT ViT_r 本身就是一個 region 一個向量)在 instance 顆粒度上對齊,
方便跨尺度比較「同一個空間位置,不同尺度的向量對 site 有多敏感」。

wsi 尺度沒有 instance(一張 WSI 一個向量),不在此腳本產生。
"""
import csv
import os

import numpy as np
import torch

from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]   # <repo>/analysis
REPO = ANALYSIS.parent                            # <repo> (MLLM-HWSI)
BASE = f"{REPO}/outputs/brca_pilot40_k48"
OUT_DIR = f"{ANALYSIS}/tables"

labels_by_slide = {r["slide_id"]: r for r in csv.DictReader(open(f"{OUT_DIR}/wsi_labels.csv"))}

cell_rows, patch_rows, region_rows = [], [], []
meta_rows = []

for sid, rec in labels_by_slide.items():
    cell = torch.load(f"{BASE}/cells_k48/{sid}/encoded_cell_features.pt",
                       map_location="cpu", weights_only=False)["encoded_cell_features"]
    patch = torch.load(f"{BASE}/features/patches_filtered/{sid}.pt",
                        map_location="cpu", weights_only=False)["selected_features"]
    region = torch.load(f"{BASE}/features/region_4k/{sid}.pt",
                         map_location="cpu", weights_only=False)

    cell_region_mean = cell.mean(dim=1).numpy()      # [num_region, 384]
    patch_region_mean = patch.mean(dim=1).numpy()    # [num_region, 512]
    region_np = region.numpy()                       # [num_region, 192]

    n = region_np.shape[0]
    assert cell_region_mean.shape[0] == n and patch_region_mean.shape[0] == n

    cell_rows.append(cell_region_mean)
    patch_rows.append(patch_region_mean)
    region_rows.append(region_np)

    for _ in range(n):
        meta_rows.append({
            "slide_id": sid,
            "class": rec["class"],
            "tss": rec["tss"],
            "site_group": rec["site_group"],
        })

cell_mat = np.concatenate(cell_rows, axis=0)
patch_mat = np.concatenate(patch_rows, axis=0)
region_mat = np.concatenate(region_rows, axis=0)

np.savez(f"{OUT_DIR}/instance_level_features.npz",
         cell=cell_mat, patch=patch_mat, region=region_mat)

with open(f"{OUT_DIR}/instance_level_meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["slide_id", "class", "tss", "site_group"])
    w.writeheader()
    w.writerows(meta_rows)

print("=== instance-level 特徵矩陣 shape ===")
print("cell  :", cell_mat.shape)
print("patch :", patch_mat.shape)
print("region:", region_mat.shape)
print("meta rows:", len(meta_rows))

print("\n=== 每亞型 instance 總數(=各張 WSI region 數總和) ===")
import collections
cnt = collections.Counter((r["class"]) for r in meta_rows)
for cls, n in sorted(cnt.items()):
    print(f"  {cls}: {n} instances")
