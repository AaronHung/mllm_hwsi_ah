#!/usr/bin/env bash
# Tiny cross-backend gate.  Run from the repo root or from any directory.
#
# Examples:
#   bash scripts/smoke_test.sh --device cpu
#   bash scripts/smoke_test.sh --device mps
#   bash scripts/smoke_test.sh --device auto
#   bash scripts/smoke_test.sh --all       # CPU + MPS; intended for Mac

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif [[ -x "${NAVIPATH_VENV:-$HOME/research/01_navipath/.venv/bin/python}" ]]; then
    # Local compatibility with the already-tested NaviPath environment.
    PYTHON_BIN="${NAVIPATH_VENV:-$HOME/research/01_navipath/.venv/bin/python}"
fi

mode="auto"
if [[ "${1:-}" == "--all" ]]; then
    mode="all"
elif [[ "${1:-}" == "--device" ]]; then
    mode="${2:?usage: $0 [--device auto|cpu|mps|cuda|--all]}"
elif [[ $# -gt 0 ]]; then
    mode="$1"
fi

case "$mode" in
    auto|cpu|mps|cuda) devices=("$mode") ;;
    all)
        devices=(cpu)
        has_mps="$("$PYTHON_BIN" - <<'PY'
import torch
print(int(torch.backends.mps.is_available()))
PY
)"
        [[ "$has_mps" == "1" ]] || {
            echo "[smoke] ERROR: --all requires an available MPS backend" >&2
            exit 1
        }
        devices+=(mps)
        ;;
    *) echo "usage: $0 [--device auto|cpu|mps|cuda|--all]" >&2; exit 2 ;;
esac

for device in "${devices[@]}"; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    out_dir="$ROOT/runs/v2/smoke_${device}_${stamp}"
    mkdir -p "$out_dir"
    log="$out_dir/smoke.log"
    echo "[smoke] device=$device output=$out_dir"
    set -o pipefail
    "$PYTHON_BIN" "$ROOT/scripts/cl_main.py" \
        --dataset can \
        --order main \
        --smoke \
        --device "$device" \
        --output-dir "$out_dir" \
        --tag "smoke_${device}_${stamp}" \
        --resume \
        2>&1 | tee "$log"

    expected_device="$device"
    if [[ "$device" == "auto" ]]; then
        expected_device="$("$PYTHON_BIN" - <<'PY'
from nav import resolve_device
print(resolve_device("auto"))
PY
)"
    fi
    "$PYTHON_BIN" - "$out_dir" "$expected_device" <<'PY'
import math
import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1])
expected = sys.argv[2]
csvs = sorted(out_dir.glob("cl_main_*_smoke_*.csv"))
if len(csvs) != 1:
    raise SystemExit(f"smoke expected one CSV, found {csvs}")
df = pd.read_csv(csvs[0])
if df.empty or len(df["task"].unique()) != 2 or set(df["K"]) != {1}:
    raise SystemExit(f"smoke shape/config check failed: rows={len(df)}")
for col in ("bal_acc", "acc", "jaccard", "action_kl", "sel_utility"):
    values = pd.to_numeric(df[col], errors="coerce")
    if not values.map(math.isfinite).all():
        raise SystemExit(f"non-finite values in {col}")
if set(df["resolved_device"]) != {expected}:
    raise SystemExit(
        f"resolved device mismatch: expected={expected}, "
        f"got={sorted(df['resolved_device'].unique())}"
    )
print(
    f"[smoke] PASS device={expected} rows={len(df)} "
    f"csv={csvs[0].relative_to(out_dir.parent.parent.parent)}"
)
PY
done

echo "[smoke] PASS: finite losses/metrics and CSV output on ${devices[*]}"
