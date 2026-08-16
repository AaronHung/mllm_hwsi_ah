"""Gate v2 — atomic-resume runner for the `eq_pres`-only extension
(`docs/method_gate_v033.md` §10), pre-registered BEFORE this script's first
training run per §10.5/r1.

Pins the exact 19-unit v2 grid (§10.2) into one command, mirroring
`scripts/run_method_gate.py`'s atomic-resume pattern:

    1. eq_pres            seeds {3,4} x K{1,2,4}          =  6  (new candidate arm)
    2. ours_uniform        seeds {0..4} x K=4              =  5  (new — Protocol-v1
                                                                    never ran this
                                                                    method at K=4)
    3. distill             seeds {3,4} x K{1,2}            =  4  (utility-axis
       ours_uniform        seeds {3,4} x K{1,2}            =  4   backfill — NOT new
                                                                    experiments; the
                                                                    frozen bal_acc/
                                                                    Forgetting/Jaccard/
                                                                    action_kl for these
                                                                    cells already exist
                                                                    in `..._full_s{3,4}
                                                                    .csv` / `..._ablA1_
                                                                    s{3,4}.csv` — this
                                                                    rerun only backfills
                                                                    eps_optimal_mass/
                                                                    normalized_regret,
                                                                    checksum-gated
                                                                    against those frozen
                                                                    rows exactly like
                                                                    v0.33.2 §5.2)
                                                          -----
                                                          19 units total

Reuses `scripts.cl_main.run_sequence` directly (identical per-row behavior:
inline eps_optimal_mass/normalized_regret, stage-boundary torch.save under
`<out_dir>/ckpt/`, per-refresh diagnostics for methods that need them) — only
fixes *which* units run, adds atomic resume, and stamps every row with a
`gate_v2_role` column so the verdict script / audit trail can tell "new
candidate-arm training" apart from "comparator completion" apart from
"utility-axis backfill of already-frozen numbers" without re-deriving it.

用法（per docs/method_gate_v033.md §10.2, Mac MPS, tmux + caffeinate）：
    tmux new -s gate_v2
    caffeinate -i python scripts/run_gate_v2.py --device mps \
        --tag gate_v2_<run_tag> --resume
    # Ctrl-b d to detach; tmux attach -t gate_v2 to reattach.
    # Safe-stop: Ctrl-C between units (never mid-unit).
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

NEW_SEEDS = (3, 4)
ALL_SEEDS = (0, 1, 2, 3, 4)
ROLE_CANDIDATE = "candidate_new"
ROLE_COMPARATOR_K4 = "comparator_completion_k4"
ROLE_UTILITY_BACKFILL = "utility_backfill_seeds34"


def gate_v2_units() -> list[tuple[str, int, int, str]]:
    """Returns (method, seed, K, role) tuples, §10.2 order."""
    units: list[tuple[str, int, int, str]] = []
    units += [("eq_pres", s, k, ROLE_CANDIDATE)
              for s in NEW_SEEDS for k in (1, 2, 4)]
    units += [("ours_uniform", s, 4, ROLE_COMPARATOR_K4) for s in ALL_SEEDS]
    units += [(m, s, k, ROLE_UTILITY_BACKFILL)
              for m in ("distill", "ours_uniform")
              for s in NEW_SEEDS for k in (1, 2)]
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replay-per-slide", type=int, default=2)
    ap.add_argument("--buffer-cap", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="skip (method,seed,K) units already in checkpoints.json")
    add_device_argument(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    meta["run_tag"] = args.tag
    meta.update(run_provenance())
    if str(device) != "mps":
        print(f"WARNING: docs/method_gate_v033.md §10.2 pre-registers Gate v2 "
              f"on Mac MPS; running on device={device} instead.",
              file=sys.stderr)

    out_dir = REPO / "runs" / "v2" / args.tag
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
    units = gate_v2_units()
    print(f"[gate_v2] {len(units)} units pinned per §10.2 "
          f"({sum(1 for u in units if u[3] == ROLE_CANDIDATE)} candidate-new, "
          f"{sum(1 for u in units if u[3] == ROLE_COMPARATOR_K4)} comparator-K4, "
          f"{sum(1 for u in units if u[3] == ROLE_UTILITY_BACKFILL)} utility-backfill)")

    t0 = time.time()
    for method, seed, k, role in units:
        key = f"seed={seed}|method={method}|K={k}|role={role}"
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
            r.update(dataset="can", order="main", gate_v2_role=role)
        df = pd.DataFrame(rows)
        header = not out_csv.exists()
        df.to_csv(out_csv, mode="a", header=header, index=False)
        completed.add(key)
        unit_s = time.time() - unit_t0
        manifest = {
            "completed": sorted(completed),
            "metadata": meta,
            "last_unit": {"method": method, "seed": seed, "K": k, "role": role,
                         "status": "done", "seconds": round(unit_s, 1),
                         "git_commit": meta["git_commit"],
                         "backend": str(device),
                         "torch_version": meta["torch_version"]},
        }
        checkpoint_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False))
        fin = [r for r in rows if r["stage"] == len(tasks)]
        aa = np.mean([r["bal_acc"] for r in fin])
        print(f"[{method} seed{seed} K={k} role={role}] AA={aa:.3f} "
              f"unit={unit_s:.0f}s total={time.time() - t0:.0f}s "
              f"({len(completed)}/{len(units)})")

    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
