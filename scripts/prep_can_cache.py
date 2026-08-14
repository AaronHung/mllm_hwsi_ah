"""can_dataset pseudo-region cache 一次性建置（可中斷續跑、可多進程）。

每張 slide：
    1. 讀 CONCH patch 特徵 [N,512]
    2. MiniBatchKMeans(R=32, random_state=0) 聚成 pseudo-region
    3. high[r] = cluster r 的 mean（512）
    4. low[r]  = P @ high[r]（固定 random projection 512→64）
    5. 存 data/can_cache/<cohort>/<slide_id>.npz

用法：
    python scripts/prep_can_cache.py --cohorts esca            # 先做最小 cohort 驗證
    python scripts/prep_can_cache.py --cohorts esca lung rcc brca --jobs 6
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import add_device_argument, resolve_device  # noqa: E402
from nav.candata import (CAN_ROOT, R_CLUSTERS, cohort_table,  # noqa: E402
                         projection_matrix)

CACHE_ROOT = REPO / "data" / "can_cache"


def process_slide(args: tuple[str, str, str]) -> tuple[str, int, int]:
    cohort, sid, can_root = args
    import torch
    from sklearn.cluster import MiniBatchKMeans

    out = CACHE_ROOT / cohort / f"{sid}.npz"
    pt = Path(can_root) / f"tcga_{cohort}/feats-l1-s256_CONCH/pt_files/{sid}.pt"
    feats = torch.load(pt, map_location="cpu", weights_only=False)
    if not torch.is_tensor(feats):  # 防禦：部分檔可能是 dict
        feats = next(v for v in feats.values() if torch.is_tensor(v))
    X = feats.float().numpy()
    n = X.shape[0]
    r = min(R_CLUSTERS, n)
    if r < 2:
        high = X.mean(0, keepdims=True)
    else:
        km = MiniBatchKMeans(n_clusters=r, random_state=0, n_init=3,
                             batch_size=1024)
        assign = km.fit_predict(X)
        high = np.stack([X[assign == c].mean(0) if (assign == c).any()
                         else km.cluster_centers_[c] for c in range(r)])
    P = projection_matrix(CACHE_ROOT, high_dim=X.shape[1])
    low = high @ P.T
    np.savez_compressed(out, low=low.astype(np.float32),
                        high=high.astype(np.float32), n_patches=n)
    return sid, n, high.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+",
                    default=["esca", "lung", "rcc", "brca"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--can-root", default=str(CAN_ROOT))
    add_device_argument(ap)
    args = ap.parse_args()
    device = resolve_device(args.device)
    print(f"resolved device = {device} (cache preparation is CPU-bound)")

    projection_matrix(CACHE_ROOT)  # 先確保矩陣存在（避免多進程競態）

    for cohort in args.cohorts:
        t = cohort_table(cohort, Path(args.can_root))
        out_dir = CACHE_ROOT / cohort
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [(cohort, r.slide_id, args.can_root) for r in t.itertuples()
                if not (out_dir / f"{r.slide_id}.npz").exists()]
        print(f"[{cohort}] total={len(t)}  todo={len(todo)}")
        t0 = time.time()
        rows = []
        if todo:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(process_slide, a) for a in todo]
                for i, f in enumerate(as_completed(futs), 1):
                    sid, n, r = f.result()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  [{cohort}] {i}/{len(todo)} "
                              f"({time.time() - t0:.0f}s)  last: {sid} N={n} R={r}")
        # manifest（每次重建，涵蓋已有 cache 的全部 slide）
        for r in t.itertuples():
            d = np.load(out_dir / f"{r.slide_id}.npz")
            rows.append(dict(slide_id=r.slide_id, patient_id=r.patient_id,
                             subtype=r.subtype, label=r.label,
                             n_patches=int(d["n_patches"]),
                             R=d["high"].shape[0]))
        import pandas as pd
        pd.DataFrame(rows).to_csv(CACHE_ROOT / f"{cohort}_manifest.csv",
                                  index=False)
        print(f"[{cohort}] done in {time.time() - t0:.0f}s "
              f"-> {CACHE_ROOT / (cohort + '_manifest.csv')}")


if __name__ == "__main__":
    main()
