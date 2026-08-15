"""v0.33.3 item D — atomic-resume runner for the Track-B method gate.

Pins the exact pre-registered gate grid (`docs/method_gate_v033.md` v0.33.3
§B) into one command, instead of the two hand-assembled `cl_main.py`
invocations (main configs at K in {1,2,4}; mini-arms restricted to K=1 only)
that were error-prone to keep in sync by hand:

    main configs   : ia_samp, eq_pres, ia_ep   x seeds {0,1,2} x K {1,2,4}  = 27
    mini-arms      : samp_util_only, samp_drift_only x seeds {0,1,2} x K=1 =  6
                                                                    total  = 33

Reuses `scripts.cl_main.run_sequence` directly, so every per-unit behavior
(inline eps_optimal_mass/normalized_regret columns, stage-boundary
torch.save under `<out_dir>/ckpt/`, per-refresh diagnostics) is identical to
running `cl_main.py` — this script only fixes *which* units run and adds
atomic resume + a manifest with per-unit provenance (v0.33.3 item D/E5):
after each (method, seed, K) unit finishes, its CSV rows are appended, its
checkpoint is on disk, and `checkpoints.json` is updated *before* the next
unit starts — safe to Ctrl-C between units and continue later with
`--resume`.

Also implements the v0.33.3 item C timing decision rule: `--timing-only`
runs exactly the one pre-registered representative unit (`eq_pres`, seed 0,
K=2) and prints `T_total = 33 * t_unit * 1.15` plus the pre-declared
<=48h / >48h decision, without touching the real gate output directory.

用法（per docs/method_gate_v033.md §5, MPS-first per v0.33.3 §B）：
    # 1) timing decision rule (§C) — run once before the real gate:
    caffeinate -i python scripts/run_method_gate.py --device mps \
        --tag gate_v0333_<run_tag> --timing-only

    # 2) real gate, inside tmux, resumable:
    tmux new -s gate
    caffeinate -i python scripts/run_method_gate.py --device mps \
        --tag gate_v0333_<run_tag> --resume
    # Ctrl-b d to detach; `tmux attach -t gate` to reattach.
    # Safe-stop: Ctrl-C between units (never mid-unit) before closing the lid.
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
from scripts.cl_main import load_can_tasks, run_sequence  # noqa: E402

MAIN_CONFIGS = ["ia_samp", "eq_pres", "ia_ep"]
MINI_ARMS = ["samp_util_only", "samp_drift_only"]
SEEDS = [0, 1, 2]
MAIN_BUDGETS = [1, 2, 4]
MINI_BUDGET = [1]
TIMING_UNIT = ("eq_pres", 0, 2)  # pre-registered representative unit (§C)
N_UNITS = len(MAIN_CONFIGS) * len(SEEDS) * len(MAIN_BUDGETS) \
    + len(MINI_ARMS) * len(SEEDS) * len(MINI_BUDGET)
SAFETY_FACTOR = 1.15
TIMING_LIMIT_H = 48.0


def gate_units() -> list[tuple[str, int, int]]:
    return ([(m, s, k) for m in MAIN_CONFIGS for s in SEEDS for k in MAIN_BUDGETS]
            + [(m, s, k) for m in MINI_ARMS for s in SEEDS for k in MINI_BUDGET])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replay-per-slide", type=int, default=2)
    ap.add_argument("--buffer-cap", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="skip (method,seed,K) units already in checkpoints.json")
    ap.add_argument("--timing-only", action="store_true",
                    help="run exactly the pre-registered representative unit "
                    "(eq_pres, seed 0, K=2) and print the v0.33.3 item-C "
                    "timing decision (T_total = 33 * t_unit * 1.15 vs 48h), "
                    "then exit without touching the real gate output dir")
    add_device_argument(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    meta["run_tag"] = args.tag
    meta.update(run_provenance())
    if str(device) != "mps":
        print(f"WARNING: v0.33.3 §B pre-registers the gate on Mac MPS; "
              f"running on device={device} instead. This is only expected "
              f"if §C's timing rule already triggered the CUDA-matched "
              f"fallback (see docs/method_gate_v033.md changelog).",
              file=sys.stderr)

    out_dir = REPO / "runs" / "v2" / (f"{args.tag}_timing" if args.timing_only
                                      else args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"device = {device} | torch = {meta['torch_version']} | "
          f"commit = {meta['git_commit'][:12]} | host = {meta['hostname']}")

    ev_epochs, nav_epochs = 30, 10
    out_csv = out_dir / f"cl_main_can_main_{out_dir.name}.csv"
    checkpoint_path = out_dir / "checkpoints.json"
    completed: set[str] = set()
    if args.resume and checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text())
                        .get("completed", []))

    tasks = load_can_tasks("main", smoke=False)
    units = [TIMING_UNIT] if args.timing_only else gate_units()

    t0 = time.time()
    for method, seed, k in units:
        key = f"seed={seed}|method={method}|K={k}"
        if key in completed:
            print(f"[resume] skip {key}")
            continue
        unit_t0 = time.time()
        diag_log: list[dict] = []
        ckpt_dir = out_dir / "ckpt"
        rows = run_sequence(
            tasks, method, seed, k, device, ev_epochs, nav_epochs,
            args.lam, 100.0, args.replay_per_slide, args.buffer_cap,
            run_meta=meta, diag_log=diag_log, ckpt_dir=ckpt_dir)
        if diag_log:
            (out_dir / f"diag_{method}_seed{seed}_K{k}.json").write_text(
                json.dumps(diag_log, indent=2))
        for r in rows:
            r.update(dataset="can", order="main")
        df = pd.DataFrame(rows)
        header = not out_csv.exists()
        df.to_csv(out_csv, mode="a", header=header, index=False)
        completed.add(key)
        unit_s = time.time() - unit_t0
        manifest = {
            "completed": sorted(completed),
            "metadata": meta,
            "last_unit": {"method": method, "seed": seed, "K": k,
                         "status": "done", "seconds": round(unit_s, 1),
                         "git_commit": meta["git_commit"],
                         "backend": str(device),
                         "torch_version": meta["torch_version"]},
        }
        checkpoint_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False))
        fin = [r for r in rows if r["stage"] == len(tasks)]
        aa = np.mean([r["bal_acc"] for r in fin])
        print(f"[{method} seed{seed} K={k}] AA={aa:.3f} unit={unit_s:.0f}s "
              f"total={time.time() - t0:.0f}s")

        if args.timing_only:
            t_total_h = unit_s * N_UNITS * SAFETY_FACTOR / 3600
            decision = ("<=48h -> proceed all-MPS immediately, per v0.33.3 §C"
                       if t_total_h <= TIMING_LIMIT_H else
                       ">48h -> STOP; fall back to the CUDA-matched plan "
                       "(rerun ours_uniform/distill/replay as CUDA "
                       "comparators, per v0.33.3 §C)")
            print(f"\n[timing-decision] t_unit={unit_s:.0f}s "
                  f"({method} seed{seed} K={k}), n_units={N_UNITS}, "
                  f"T_total = {N_UNITS} * {unit_s:.0f}s * {SAFETY_FACTOR} "
                  f"= {t_total_h:.2f}h")
            print(f"[timing-decision] {decision}")
            print(f"[timing-decision] record this in "
                  f"docs/method_gate_v033.md's changelog before launching "
                  f"the real gate.")
            return

    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
