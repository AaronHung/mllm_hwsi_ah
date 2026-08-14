#!/usr/bin/env bash
set -euo pipefail

# WP5 is intentionally gated after the first WP4 report.
# Run from a RunPod tmux session after sourcing runpod_bootstrap.sh.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_TAG="${RUN_TAG:-pilot40_min_v0292_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs/v2}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$RUN_ROOT/$RUN_TAG"

if [[ "$RUN_TAG" == *full* || "$RUN_TAG" == *protocol_v1* ]]; then
  echo "refusing unsafe run tag: $RUN_TAG" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "[pilot40] tag=$RUN_TAG"
echo "[pilot40] output=$OUT_DIR"
echo "[pilot40] device=$DEVICE"
echo "[pilot40] checkpoints are enabled by cl_main.py"

"$PYTHON_BIN" "$REPO_DIR/scripts/cl_main.py" \
  --dataset pilot40 \
  --order main \
  --methods seqft distill ours joint \
  --budgets 1 2 4 \
  --seeds 0 1 2 3 4 \
  --device "$DEVICE" \
  --tag "$RUN_TAG" \
  --output-dir "$OUT_DIR" \
  --resume \
  2>&1 | tee "$OUT_DIR/pilot40.log"
