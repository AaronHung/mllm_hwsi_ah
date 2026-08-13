"""Pilot-40（brca_pilot40_k48）特徵資產的載入工具。

資料來源：MLLM-HWSI pipeline 抽取的四層特徵（見 data/brca_pilot40_k48/）。
格式依 README_FEATURES.md：
    cell   : cells_k48/<slide>/encoded_cell_features.pt -> dict["encoded_cell_features"]  [R,48,384]
    patch  : features/patches_filtered/<slide>.pt       -> dict["selected_features"]      [R,48,512]
    region : features/region_4k/<slide>.pt              -> tensor                          [R,192]
    wsi    : features/wsi/<slide>.pt                    -> tensor                          [192]
    coords : features/coords_region4096_valid/<slide>.h5 -> dataset "coords"               (R,2)

注意：wsi 特徵 == region.mean(dim=0)（逐元素相等），不是獨立訊號。
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import torch

CLASSES = ["KIRC", "KIRP", "IDC", "ILC"]
ORGAN_OF = {"KIRC": "Renal", "KIRP": "Renal", "IDC": "Breast", "ILC": "Breast"}


def _as_tensor(obj, key_candidates=()):
    """torch.load 的結果可能是 tensor 或 dict，統一取出 tensor。"""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in key_candidates:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        tensors = [v for v in obj.values() if torch.is_tensor(v)]
        if len(tensors) == 1:
            return tensors[0]
        raise ValueError(f"dict 內找不到唯一 tensor，keys={list(obj.keys())}")
    raise TypeError(f"不支援的型別：{type(obj)}")


class Pilot40:
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root)
        self.base = self.root / "brca_pilot40_k48"
        self.labels: dict[str, str] = {}
        csv_path = self.root / "pilot40_selection.csv"
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.labels[row["slide_id"]] = row["class"]
        self.slide_ids = sorted(self.labels.keys())

    # ---- 各層特徵 ----
    def wsi(self, sid: str) -> torch.Tensor:
        return _as_tensor(torch.load(self.base / "features/wsi" / f"{sid}.pt",
                                     map_location="cpu", weights_only=False))

    def region(self, sid: str) -> torch.Tensor:
        return _as_tensor(torch.load(self.base / "features/region_4k" / f"{sid}.pt",
                                     map_location="cpu", weights_only=False))

    def patch(self, sid: str) -> torch.Tensor:
        obj = torch.load(self.base / "features/patches_filtered" / f"{sid}.pt",
                         map_location="cpu", weights_only=False)
        return _as_tensor(obj, key_candidates=("selected_features",))

    def cell(self, sid: str) -> torch.Tensor:
        obj = torch.load(self.base / "cells_k48" / sid / "encoded_cell_features.pt",
                         map_location="cpu", weights_only=False)
        return _as_tensor(obj, key_candidates=("encoded_cell_features",))

    def coords(self, sid: str) -> np.ndarray:
        with h5py.File(self.base / "features/coords_region4096_valid" / f"{sid}.h5") as f:
            return f["coords"][:]

    # ---- 標籤 ----
    def label4(self, sid: str) -> str:
        return self.labels[sid]

    def label2(self, sid: str) -> str:
        return ORGAN_OF[self.labels[sid]]
