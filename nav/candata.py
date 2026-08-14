"""can_dataset（TCGA ESCA/LUNG/RCC/BRCA，CONCH l1-s256 patch 特徵）的載入工具。

原始資料只有裸 patch 特徵 [N,512]、無座標，因此 region 以 pseudo-region 建立
（見 docs/protocol.md §1.1）：

    高倍證據 z_r  = k-means cluster r 內全部 patch 特徵的 mean（512 維，zoom 後可見）
    低倍摘要 F_low[r] = z_r 經固定 Gaussian random projection（512→64）的有損壓縮（全程可見）

cache 由 scripts/prep_can_cache.py 離線建立：
    data/can_cache/<cohort>/<slide_id>.npz : {low(R,64) f32, high(R,512) f32, n_patches}
    data/can_cache/proj_512to64.npz        : 全域共用 projection 矩陣（seed=0）
    data/can_cache/<cohort>_manifest.csv   : slide_id, patient_id, subtype, label, n_patches, R
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CAN_ROOT = Path(os.environ.get("CAN_ROOT", "/Users/aaron/research/can_dataset"))

COHORTS = {
    "esca": ["ESCC", "ESAD"],
    "lung": ["LUAD", "LUSC"],
    "rcc": ["CCRCC", "PRCC", "CHRCC"],
    "brca": ["IDC", "ILC"],
}

R_CLUSTERS = 32   # pseudo-region 數（N < 32 時取 N）
LOW_DIM = 64      # 低倍摘要維度
PROJ_SEED = 0


def projection_matrix(cache_root: Path, high_dim: int = 512) -> np.ndarray:
    """固定 Gaussian random projection（LOW_DIM×high_dim），存檔以保證可重現。"""
    path = cache_root / "proj_512to64.npz"
    if path.exists():
        return np.load(path)["P"]
    rng = np.random.default_rng(PROJ_SEED)
    P = rng.standard_normal((LOW_DIM, high_dim)).astype(np.float32) / np.sqrt(LOW_DIM)
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, P=P)
    return P


def cohort_table(cohort: str, can_root: Path = CAN_ROOT) -> pd.DataFrame:
    """回傳該 cohort 的 slide 表：slide_id（=pt 檔名 stem）、patient_id、subtype、label。

    table CSV 的 pathology_id 與 pt 檔名的 UUID 大小寫不一致，以 lowercase 比對。
    只保留 (a) 特徵檔存在且 (b) subtype 在 COHORTS 類別清單內的列。
    """
    croot = can_root / f"tcga_{cohort}"
    csv = list((croot / "table").glob("*.csv"))[0]
    t = pd.read_csv(csv)
    pt_dir = croot / "feats-l1-s256_CONCH/pt_files"
    stem_of = {p.stem.lower(): p.stem for p in pt_dir.glob("*.pt")}
    t["slide_id"] = t["pathology_id"].str.lower().map(stem_of)
    t = t[t["slide_id"].notna() & t["subtype"].isin(COHORTS[cohort])]
    t = t.drop_duplicates("slide_id").reset_index(drop=True)
    return t[["slide_id", "patient_id", "subtype", "label"]]


def fold_split(cohort: str, fold: int = 1, can_root: Path = CAN_ROOT
               ) -> dict[str, set[str]]:
    """內建 patient-level split（protocol 凍結 fold_1）。"""
    d = np.load(can_root / f"tcga_{cohort}/datasplit/fold_{fold}.npz",
                allow_pickle=True)
    return {k.split("_")[0]: set(d[k].tolist())
            for k in ["train_patients", "val_patients", "test_patients"]}


class CanTask:
    """單一 cohort = 單一任務。介面對齊 nav.engine 的使用方式。"""

    def __init__(self, cohort: str, cache_root: Path, fold: int = 1,
                 can_root: Path = CAN_ROOT):
        self.cohort = cohort
        self.classes = COHORTS[cohort]
        self.cache = cache_root / cohort
        t = cohort_table(cohort, can_root)
        have = {p.stem for p in self.cache.glob("*.npz")}
        missing = set(t["slide_id"]) - have
        if missing:
            raise FileNotFoundError(
                f"{cohort}: {len(missing)} 張缺 cache，先跑 scripts/prep_can_cache.py")
        self.table = t
        split = fold_split(cohort, fold, can_root)
        self.sids = {name: t.loc[t["patient_id"].isin(pats), "slide_id"].tolist()
                     for name, pats in split.items()}
        self.label_of = {r.slide_id: self.classes.index(r.subtype)
                         for r in t.itertuples()}

    def load_bank(self, split: str) -> list:
        """回傳 nav.engine.Slide 清單（low=64 維摘要、high=512 維證據）。"""
        from .engine import Slide
        bank = []
        for sid in self.sids[split]:
            d = np.load(self.cache / f"{sid}.npz")
            bank.append(Slide(sid=sid, y=self.label_of[sid],
                              low=torch.from_numpy(d["low"]),
                              high=torch.from_numpy(d["high"])))
        return bank
