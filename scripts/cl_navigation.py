"""Gate 2 + Gate 3：共享 navigator 的持續學習（navigation forgetting 與其緩解）。

任務序列（pilot40）：
    T1 = renal subtyping（KIRC vs KIRP，20 張）
    T2 = breast subtyping（IDC vs ILC，20 張）

Navigation-only protocol（把遺忘歸因到「導航」而非「分類」）：
    - 每個任務有自己的 evaluator（診斷頭），任務訓練結束即凍結、永不再動。
    - 只有 navigator 是共享且持續更新的。
    - 之後 T1 的效能變化，只可能來自 navigator 行為改變。

方法：
    seqft   : Gate 2 baseline，T2 直接 fine-tune 同一個 navigator。
    distill : old-policy / policy-fidelity distillation on compressed
              navigation-state replay（legacy two-task probe）。

量測（都在 T1 test 上）：
    - acc_before / acc_after / forgetting = before - after
    - action distribution drift：固定 probe states 上 KL(π_after || π_before)
    - selection overlap：T1 test 每張切片前後所選 region 的 Jaccard

用法：
    python scripts/cl_navigation.py --smoke
    python scripts/cl_navigation.py            # 3 seeds 完整版
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import add_device_argument, device_info, resolve_device  # noqa: E402
from nav.engine import (build_bank, eval_policy, teacher_rollout,  # noqa: E402
                        train_evaluator, train_navigator)
from nav.pilot40 import Pilot40  # noqa: E402

RESULTS = REPO / "results"
FIGURES = REPO / "figures"

TASKS = {"T1_renal": ["KIRC", "KIRP"], "T2_breast": ["IDC", "ILC"]}


def split_task(ds, task_classes, seed):
    """任務內 stratified 70/30 split。"""
    from sklearn.model_selection import train_test_split
    sids = [s for s in ds.slide_ids if ds.label4(s) in task_classes]
    ys = [task_classes.index(ds.label4(s)) for s in sids]
    tr, te = train_test_split(sids, test_size=0.3, stratify=ys, random_state=seed)
    label_of = {s: task_classes.index(ds.label4(s)) for s in sids}
    return tr, te, label_of


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def run_method(method: str, ds, k: int, seed: int, device,
               ev_epochs: int, nav_epochs: int,
               replay_per_slide: int = 2, lam: float = 1.0) -> dict:
    torch.manual_seed(seed)
    t1_cls, t2_cls = TASKS["T1_renal"], TASKS["T2_breast"]
    tr1, te1, lab1 = split_task(ds, t1_cls, seed)
    tr2, te2, lab2 = split_task(ds, t2_cls, seed)

    bank_tr1 = build_bank(ds, tr1, lab1)
    bank_te1 = build_bank(ds, te1, lab1)
    bank_tr2 = build_bank(ds, tr2, lab2)
    bank_te2 = build_bank(ds, te2, lab2)

    # ---- Phase 1：T1 訓練（evaluator1 + navigator），完成後凍結 evaluator1 ----
    ev1 = train_evaluator(bank_tr1, 2, k, device, epochs=ev_epochs, seed=seed)
    steps1 = []
    for s in bank_tr1:
        steps1 += teacher_rollout(s, ev1, k, device)
    nav = train_navigator(steps1, device, epochs=nav_epochs, seed=seed)

    acc1_before, sel_before = eval_policy(bank_te1, ev1, k, device, "learned",
                                          navigator=nav)
    nav_before = copy.deepcopy(nav)

    # probe states：T1 test 切片的教師展開 state（固定，用於 drift 量測）
    probe_states = []
    for s in bank_te1:
        probe_states += teacher_rollout(s, ev1, k, device)
    with torch.no_grad():
        pi_before = [F.softmax(nav_before(st.low[st.candidates].to(device),
                                          st.context.to(device)), dim=0).cpu()
                     for st in probe_states]

    # ---- Phase 2：T2 訓練（sequential FT 或 distill+replay） ----
    ev2 = train_evaluator(bank_tr2, 2, k, device, epochs=ev_epochs, seed=seed)
    steps2 = []
    for s in bank_tr2:
        steps2 += teacher_rollout(s, ev2, k, device)

    if method == "seqft":
        nav = train_navigator(steps2, device, navigator=nav,
                              epochs=nav_epochs, seed=seed)
    elif method == "distill":
        # compressed navigation-state replay：每張 T1 訓練切片存前 replay_per_slide 步
        by_sid: dict[str, list] = {}
        for st in steps1:
            by_sid.setdefault(st.sid, []).append(st)
        replay = [st for sid in by_sid for st in by_sid[sid][:replay_per_slide]]
        nav = train_navigator(steps2, device, navigator=nav,
                              epochs=nav_epochs, seed=seed,
                              distill=(nav_before, replay, lam))
    else:
        raise ValueError(method)

    # ---- 量測 T1 遺忘（evaluator1 凍結 → 全部歸因 navigator） ----
    acc1_after, sel_after = eval_policy(bank_te1, ev1, k, device, "learned",
                                        navigator=nav)
    acc2, _ = eval_policy(bank_te2, ev2, k, device, "learned", navigator=nav)
    with torch.no_grad():
        pi_after = [F.softmax(nav(st.low[st.candidates].to(device),
                                  st.context.to(device)), dim=0).cpu()
                    for st in probe_states]
    drift = float(np.mean([
        F.kl_div(pa.log(), pb, reduction="sum").item()
        for pa, pb in zip(pi_after, pi_before)]))
    overlap = float(np.mean([jaccard(sel_before[s], sel_after[s])
                             for s in sel_before]))
    # random 對照（同一凍結 evaluator1）
    acc1_rand = float(np.mean([eval_policy(bank_te1, ev1, k, device, "random",
                                           seed=rs)[0] for rs in range(5)]))

    return dict(method=method, seed=seed, K=k,
                t1_acc_before=acc1_before, t1_acc_after=acc1_after,
                forgetting=acc1_before - acc1_after,
                t2_acc=acc2, t1_random_ref=acc1_rand,
                action_kl_drift=round(drift, 4),
                selection_jaccard=round(overlap, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    add_device_argument(ap)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new-run directory; defaults to runs/v2/<tag>")
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    print(f"device = {device} | torch = {meta['torch_version']} | "
          f"MPS fallback = {meta['mps_fallback']}")
    ds = Pilot40(REPO / "data")

    seeds = [0] if args.smoke else [0, 1, 2]
    ev_epochs = 15 if args.smoke else 60
    nav_epochs = 8 if args.smoke else 30

    rows = []
    t0 = time.time()
    for seed in seeds:
        for method in ["seqft", "distill"]:
            r = run_method(method, ds, args.k, seed, device,
                           ev_epochs, nav_epochs, lam=args.lam)
            rows.append(r)
            r.update(resolved_device=str(device),
                     torch_version=meta["torch_version"],
                     mps_fallback=meta["mps_fallback"])
            print(f"[{method} seed{seed}] T1 {r['t1_acc_before']:.3f} -> "
                  f"{r['t1_acc_after']:.3f} (forget {r['forgetting']:+.3f}) | "
                  f"T2 {r['t2_acc']:.3f} | drift KL {r['action_kl_drift']:.3f} | "
                  f"overlap {r['selection_jaccard']:.2f} ({time.time() - t0:.0f}s)")

    import pandas as pd
    df = pd.DataFrame(rows)
    tag = "smoke" if args.smoke else "full"
    out_dir = (Path(args.output_dir) if args.output_dir is not None
               else REPO / "runs" / "v2" / f"cl_navigation_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    df.to_csv(out_dir / f"cl_navigation_{tag}.csv", index=False)

    agg = df.groupby("method")[["t1_acc_before", "t1_acc_after", "forgetting",
                                "t2_acc", "action_kl_drift",
                                "selection_jaccard"]].mean()
    print(agg.round(3))

    if not args.smoke:
        summary = ["# Gate 2/3 — navigation forgetting 與緩解（K=%d）" % args.k, "",
                   "```", agg.round(3).to_string(), "```", "",
                   "- `seqft` = Gate 2 sequential fine-tuning baseline",
                   "- `distill` = old-policy / policy-fidelity distillation on compressed state replay",
                   "- evaluator 凍結（navigation-only protocol），T1 掉分只可能來自導航行為改變",
                   ""]
        f_seq = agg.loc["seqft", "forgetting"]
        f_dis = agg.loc["distill", "forgetting"]
        summary.append(f"**Gate 2（觀察到 navigation forgetting）：seqft forgetting = "
                       f"{f_seq:+.3f} → {'PASS' if f_seq > 0.02 else 'WEAK/FAIL'}**")
        summary.append(f"**Gate 3（forgetting 明顯下降）：distill forgetting = "
                       f"{f_dis:+.3f}（Δ = {f_seq - f_dis:+.3f}）→ "
                       f"{'PASS' if f_dis < f_seq else 'FAIL'}**")
        (out_dir / "cl_navigation_summary.md").write_text("\n".join(summary))
        print("\n".join(summary[-2:]))


if __name__ == "__main__":
    main()
