#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/alan0804/CL/Navigator/mllm_hwsi_demo/MLLM-HWSI"
OUTPUT_DIR="$PROJECT_ROOT/outputs/rcc_pilot20_40x/inference_k16_three_prompt"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

python -m Ours.run_rcc_pilot20_k16 \
  --model-dir checkpoints/mllm_hwsi \
  --feature-root outputs/rcc_pilot20_40x/features \
  --cell-root outputs/rcc_pilot20_40x/cells \
  --selection-csv /home/alan0804/CL/Navigator/pilot20_actual_selection.csv \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "$@"
