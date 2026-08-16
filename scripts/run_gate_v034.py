"""v0.34 development gate — atomic-resume runner
(`docs/method_gate_v034.md` §3.1), pre-registered BEFORE this script's first
training run.

Pins the exact 36-unit grid into one command, mirroring
`scripts/run_gate_v2.py`'s atomic-resume pattern:

    1. proj_distill      seeds {0,1,2} x K{1,2,4}   =  9  (generic control)
       proj_eq_pres      seeds {0,1,2} x K{1,2,4}   =  9  (proposed)
       conflict_eq_pres  seeds {0,1,2} x K{1,2,4}   =  9  (proposed, full)
                                                    ----
                                                     27  development gate

    2. eq_pres_diag      seeds {0,1,2} x K{1,2,4}   =  9  (instrumentation-
                                                          only replication of
                                                          frozen eq_pres,
                                                          §2.5 — NO metric
                                                          row from this batch
                                                          is admitted
                                                          anywhere; it must
                                                          reproduce the
                                                          frozen eq_pres rows
                                                          bit-identically or
                                                          the whole batch is
                                                          discarded)
                                                    ----
                                                     36 units total

Order matters: the development gate owns the critical path and runs first
(§2.5 condition 4).

Main order only, and that order is **already unblinded** — every row this
script produces is a **development environment** observation, never
confirmation evidence (§3). Confirmation is reverse order, seeds {0,1,2,3,4},
and only after the three-way review of the development verdict.

Reuses `scripts.cl_main.run_sequence` directly (identical per-row behavior),
adds the `gate_v034_role` provenance column, and routes the mandatory
per-update gradient-conflict instrumentation (§2.4) to
`runs/v2/<tag>/arbiter/`.

用法（Mac MPS, tmux + caffeinate, per docs/method_gate_v034.md §3.1）：
    tmux new -s gate_v034
    caffeinate -i python scripts/run_gate_v034.py --device mps \
        --tag gate_v034_dev_<UTC>Z --resume 2>&1 \
        | tee logs/gate_v034_dev_<UTC>Z.log
    # Ctrl-b d to detach; tmux attach -t gate_v034 to reattach.
    # Safe-stop: Ctrl-C between units (never mid-unit).
    # Monitor: bash scripts/watch_run.sh gate_v034_dev_<UTC>Z
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

SEEDS = (0, 1, 2)
BUDGETS = (1, 2, 4)
DEV_CONFIGS = ("proj_distill", "proj_eq_pres", "conflict_eq_pres")
ROLE_DEV = "dev_candidate"
ROLE_INSTRUMENTATION = "instrumentation_only_frozen_replication"


def gate_v034_units() -> list[tuple[str, int, int, str]]:
    """Returns (method, seed, K, role) tuples, §3.1 order."""
    units: list[tuple[str, int, int, str]] = []
    units += [(m, s, k, ROLE_DEV)
              for m in DEV_CONFIGS for s in SEEDS for k in BUDGETS]
    units += [("eq_pres_diag", s, k, ROLE_INSTRUMENTATION)
              for s in SEEDS for k in BUDGETS]
    return units


def check_arbiter_health(arbiter_dir: Path) -> None:
    """Code-level self-check that the §2.4 instrumentation actually ran.

    This is NOT a scientific criterion.  Sol amendment item 2 / rider r2
    removed the "conflict ~= 0 -> abort" rule precisely because no defensible
    numeric threshold exists, and nothing here reinstates one: the check only
    fails when the logger is broken (no summaries, zero updates, all-NaN or
    all-exactly-zero cosines), which would be a software fault, not a result.
    """
    summaries = sorted(arbiter_dir.glob("arbiter_summary_*.json"))
    if not summaries:
        raise SystemExit(
            f"[self-check] FAILED: no arbiter summaries under {arbiter_dir}. "
            "The §2.4 instrumentation is not being written — aborting before "
            "burning the rest of the grid.")
    rows: list[dict] = []
    for path in summaries:
        rows += json.loads(path.read_text())
    updates = sum(r["updates"] for r in rows)
    cos = [r["cos_mean"] for r in rows if r["cos_mean"] is not None]
    if updates == 0 or not cos:
        raise SystemExit(
            f"[self-check] FAILED: {len(rows)} stage summaries but "
            f"{updates} logged updates — instrumentation is degenerate.")
    if all(np.isnan(c) for c in cos):
        raise SystemExit("[self-check] FAILED: every cos(g_n,g_m) is NaN.")
    if all(c == 0.0 for c in cos):
        raise SystemExit(
            "[self-check] FAILED: every cos(g_n,g_m) is exactly 0.0, which "
            "means one of the two gradients is not being computed.")
    print(f"[self-check] OK: {len(rows)} stage summaries, {updates} logged "
          f"updates, cos range [{min(cos):+.4f}, {max(cos):+.4f}], "
          f"conflict fraction range "
          f"[{min(r['conflict_fraction'] for r in rows):.4f}, "
          f"{max(r['conflict_fraction'] for r in rows):.4f}] "
          "(descriptive only — no threshold is applied here).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replay-per-slide", type=int, default=2)
    ap.add_argument("--buffer-cap", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="skip (method,seed,K,role) units already in "
                         "checkpoints.json")
    ap.add_argument("--only-role", choices=[ROLE_DEV, ROLE_INSTRUMENTATION],
                    help="restrict to one batch (used by the pre-launch "
                         "bit-exactness check, not by the gate itself)")
    add_device_argument(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    meta["run_tag"] = args.tag
    meta.update(run_provenance())
    if str(device) != "mps":
        print(f"WARNING: docs/method_gate_v034.md §3.1 pre-registers the v0.34 "
              f"development gate on Mac MPS; running on device={device} "
              f"instead.", file=sys.stderr)

    out_dir = REPO / "runs" / "v2" / args.tag
    arbiter_dir = out_dir / "arbiter"
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
    units = gate_v034_units()
    if args.only_role:
        units = [u for u in units if u[3] == args.only_role]
    print(f"[gate_v034] {len(units)} units pinned per §3.1 "
          f"({sum(1 for u in units if u[3] == ROLE_DEV)} dev-candidate, "
          f"{sum(1 for u in units if u[3] == ROLE_INSTRUMENTATION)} "
          f"instrumentation-only)")

    t0 = time.time()
    self_checked = False
    for method, seed, k, role in units:
        key = f"seed={seed}|method={method}|K={k}|role={role}"
        if key in completed:
            print(f"[resume] skip {key}")
            continue
        unit_t0 = time.time()
        diag_log: list[dict] = []
        rows = run_sequence(
            tasks, method, seed, k, device, ev_epochs, nav_epochs,
            args.lam, 100.0, args.replay_per_slide, args.buffer_cap,
            run_meta=meta, diag_log=diag_log, ckpt_dir=out_dir / "ckpt",
            arbiter_dir=arbiter_dir)
        if diag_log:
            (out_dir / f"diag_{method}_seed{seed}_K{k}.json").write_text(
                json.dumps(diag_log, indent=2))
        for r in rows:
            r.update(dataset="can", order="main", gate_v034_role=role)
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

        if not self_checked and len(completed) >= 2:
            # §3.1: verify the instrumentation once, early, then never again.
            check_arbiter_health(arbiter_dir)
            self_checked = True

    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
