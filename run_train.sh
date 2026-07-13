#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_train.sh 2|3|2 3
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 2|3|2 3"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1 # Set this to the GPU device you want to use

MODEL_DIR="../MLLM_HWSI_ckpts3" # Change this to your model checkpoint save path
BASE_DIR="/mnt/Data_Partition/TCGA" # Change this to your base data path
TRAIN_SUB_FOLDER="train_25samples" # Change this to your training data subfolder

DATA_DIR="$BASE_DIR/$TRAIN_SUB_FOLDER"
WSI_DIR="$DATA_DIR/WSIs"
REPORT_JSON="$BASE_DIR/WSI-Bench-train-Report-only.jsonl"
CONVERSATION_JSON="$BASE_DIR/WSI-Bench-train.json"
MAG_LEVEL="mag20x"
REGIONS_DIR="$DATA_DIR/regions_${MAG_LEVEL}"
FEATURES_DIR="$DATA_DIR/features_${MAG_LEVEL}"

COMMON_ARGS=(
  --wsi_dir "$WSI_DIR"
  --wsi_feat_dir "$FEATURES_DIR/wsi"
  --region_feat_dir "$FEATURES_DIR/region_4k"
  --patch_feat_dir "$FEATURES_DIR/patches_filtered"
  --cell_feat_dir "$FEATURES_DIR/cells"
  --cell_pt_filename encoded_cell_features.pt
  --save_dir "$MODEL_DIR"
  --model_name Qwen/Qwen2.5-7B-Instruct
  --batch_size 1
  --val_ratio 0.1
  --num_slide_tokens 64
)

for STAGE in "$@"; do
  if [[ "$STAGE" != "2" && "$STAGE" != "3" ]]; then
    echo "Invalid stage: '$STAGE'. Use '2' or '3'."
    exit 1
  fi

  STAGE_COMMON=(--training_stage "$STAGE")

  if [[ "$STAGE" == "2" ]]; then
    echo "===== Stage 2 Training: V-L projectors ====="
    STAGE_ARGS=(
      --report_json "$REPORT_JSON"
      --lambda_lm 0.7
      --lambda_contrastive 2.0
      --lambda_consistency 0.25
      --save_name vl_projector.pt
      --epochs 30
      --lr 1e-5
    )
    python main_train.py "${STAGE_COMMON[@]}" "${COMMON_ARGS[@]}" "${STAGE_ARGS[@]}"
    echo "===== Projector Training Finished Successfully ====="
  else
    echo "===== Stage 3 Training: LLM Fine-tuning ====="
    STAGE_ARGS=(
      --conversation_json "$CONVERSATION_JSON"
      --stage2_ckpt "$MODEL_DIR/best_vl_projector.pt"
      --lambda_lm 0.7
      --lambda_contrastive 0.2
      --lambda_consistency 0.07
      --epochs 50
      --lr 2e-5
      --llm_lr 2e-5
      --projector_lr 5e-5
      --autocast_enabled
      --save_name stage3_model.pt
    )
    python main_train.py "${STAGE_COMMON[@]}" "${COMMON_ARGS[@]}" "${STAGE_ARGS[@]}"
    echo "===== Stage 3 Training Finished Successfully ====="
  fi
done