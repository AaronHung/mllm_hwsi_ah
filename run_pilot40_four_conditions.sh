#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/alan0804/CL/Navigator/mllm_hwsi_demo/MLLM-HWSI"
OUTPUT_DIR="$PROJECT_ROOT/outputs/brca_pilot40_k48/inference_four_conditions"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

python -m Ours.run_pilot40_four_conditions \
  --model-dir checkpoints/mllm_hwsi \
  --feature-root outputs/brca_pilot40_k48/features \
  --cell-root outputs/brca_pilot40_k48/cells_k48 \
  --selection-csv /home/alan0804/CL/Navigator/pilot40_selection.csv \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "$@"
