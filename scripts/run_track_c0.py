"""Track C0 — atomic-resume runner (`docs/track_c0.md` §4), pre-registered
BEFORE this script's first training run.

Pins the exact 12-unit C0 grid into one command, mirroring
`scripts/run_gate_v034.py`:

    sp_nav      seeds {0,1,2} x K{1,2}   =  6   (architecture alone)
    sp_nav_eq   seeds {0,1,2} x K{1,2}   =  6   (architecture + L_eq)
                                         ----
                                          12 units

`seqft`, `distill` and `eq_pres` are frozen comparators and are NOT rerun.

C0 is a **screening experiment, not a promotion gate**, and it assumes
**oracle task identity at both train and test time** (`docs/track_c0.md` §1)
— a best-case assumption that a router would have to remove. Every C0 report
carries that disclosure.

用法（Mac MPS, tmux + caffeinate, per docs/track_c0.md §4）：
    tmux new -s track_c0
    caffeinate -i python scripts/run_track_c0.py --device mps \
        --tag track_c0_<UTC>Z --resume 2>&1 | tee logs/track_c0_<UTC>Z.log
    # 離開請打 `tmux detach-client`；**不要按 Ctrl-C**，那會殺掉跑批
    # (docs/RUNPOD_SOP.md)。監看：bash scripts/watch_run.sh <tag>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import add_device_argument, device_info, resolve_device, run_provenance  # noqa: E402
from nav.cl import TRACK_C0_KWARGS  # noqa: E402
from nav.models import Navigator  # noqa: E402
from scripts.cl_main import load_can_tasks, run_sequence  # noqa: E402

SEEDS = (0, 1, 2)
BUDGETS = (1, 2)
CONFIGS = ("sp_nav", "sp_nav_eq")
ROLE_C0 = "c0_screening"


def c0_units() -> list[tuple[str, int, int, str]]:
    """Returns (method, seed, K, role) tuples, §4 order."""
    return [(m, s, k, ROLE_C0)
            for m in CONFIGS for s in SEEDS for k in BUDGETS]


def parameter_budget() -> dict[str, object]:
    """c4 disclosure: adapter parameters must be < 5% of the shared core.

    Computed from the real module rather than from the numbers written in the
    pre-registration, so a future architecture change cannot leave a stale
    figure in the verdict.
    """
    n_tasks = TRACK_C0_KWARGS["sp_nav"]["film_tasks"]
    nav = Navigator(low_dim=64, high_dim=512, film_tasks=n_tasks)
    core = sum(p.numel() for p in nav.core_parameters())
    per_task = sum(p.numel() for p in nav.film_parameters(0))
    return {"shared_core_params": core,
            "adapter_params_per_task": per_task,
            "adapter_params_all_tasks": per_task * n_tasks,
            "n_tasks": n_tasks,
            "adapter_pct_of_core_per_task": round(100 * per_task / core, 4),
            "adapter_pct_of_core_all_tasks": round(
                100 * per_task * n_tasks / core, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replay-per-slide", type=int, default=2)
    ap.add_argument("--buffer-cap", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="skip (method,seed,K,role) units already in "
                         "checkpoints.json")
    add_device_argument(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    meta["run_tag"] = args.tag
    meta.update(run_provenance())
    if str(device) != "mps":
        print(f"WARNING: docs/track_c0.md §4 pre-registers C0 on Mac MPS; "
              f"running on device={device} instead.", file=sys.stderr)

    out_dir = REPO / "runs" / "v2" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = parameter_budget()
    meta["c0_parameter_budget"] = budget
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"device = {device} | torch = {meta['torch_version']} | "
          f"commit = {meta['git_commit'][:12]} | host = {meta['hostname']}")
    print(f"[c4] shared core {budget['shared_core_params']} params | "
          f"adapter {budget['adapter_params_per_task']}/task "
          f"({budget['adapter_pct_of_core_per_task']}% of core) | "
          f"all {budget['n_tasks']} adapters "
          f"{budget['adapter_pct_of_core_all_tasks']}%")
    print("[c0] ORACLE TASK IDENTITY assumed at train and test time "
          "(docs/track_c0.md §1) — a best-case assumption; a router is future "
          "work and is not part of any C0 claim.")

    ev_epochs, nav_epochs = 30, 10
    out_csv = out_dir / f"cl_main_can_main_{out_dir.name}.csv"
    checkpoint_path = out_dir / "checkpoints.json"
    completed: set[str] = set()
    if args.resume and checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text())
                        .get("completed", []))

    tasks = load_can_tasks("main", smoke=False)
    units = c0_units()
    print(f"[c0] {len(units)} units pinned per §4 "
          f"({len(CONFIGS)} configs x {len(SEEDS)} seeds x {len(BUDGETS)} K)")

    t0 = time.time()
    for method, seed, k, role in units:
        key = f"seed={seed}|method={method}|K={k}|role={role}"
        if key in completed:
            print(f"[resume] skip {key}")
            continue
        unit_t0 = time.time()
        rows = run_sequence(
            tasks, method, seed, k, device, ev_epochs, nav_epochs,
            args.lam, 100.0, args.replay_per_slide, args.buffer_cap,
            run_meta=meta, ckpt_dir=out_dir / "ckpt")
        for r in rows:
            r.update(dataset="can", order="main", c0_role=role)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        completed.add(key)
        unit_s = time.time() - unit_t0
        checkpoint_path.write_text(json.dumps({
            "completed": sorted(completed),
            "metadata": meta,
            "last_unit": {"method": method, "seed": seed, "K": k, "role": role,
                          "status": "done", "seconds": round(unit_s, 1),
                          "git_commit": meta["git_commit"],
                          "backend": str(device),
                          "torch_version": meta["torch_version"]},
        }, indent=2, ensure_ascii=False))
        fin = [r for r in rows if r["stage"] == len(tasks)]
        aa = np.mean([r["bal_acc"] for r in fin])
        print(f"[{method} seed{seed} K={k}] AA={aa:.3f} unit={unit_s:.0f}s "
              f"total={time.time() - t0:.0f}s ({len(completed)}/{len(units)})")

    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
