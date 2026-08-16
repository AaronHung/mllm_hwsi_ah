#!/usr/bin/env bash
# One-shot status check for a background tmux-wrapped run (gate, pilot40, etc.).
# Usage: bash scripts/watch_run.sh <run_tag> [checkpoints.json path]
# Example: bash scripts/watch_run.sh method_gate_v0333_run1
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: watch_run.sh <run_tag>}"
CKPT="${2:-runs/v2/${TAG}/checkpoints.json}"
LOG="logs/${TAG}.log"

echo "=== tmux sessions ==="
tmux ls 2>&1

echo
echo "=== checkpoint progress ($CKPT) ==="
if [[ -f "$CKPT" ]]; then
    python3 - "$CKPT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
completed = d.get("completed", {})
n = len(completed) if isinstance(completed, dict) else len(completed)
print(f"completed: {n}")
last = d.get("last_unit")
if last:
    print(f"last_unit: {last}")
PY
else
    echo "(no checkpoint file at $CKPT yet)"
fi

echo
echo "=== last 8 lines of $LOG ==="
if [[ -f "$LOG" ]]; then
    tail -n 8 "$LOG"
else
    echo "(no log file at $LOG yet)"
fi
