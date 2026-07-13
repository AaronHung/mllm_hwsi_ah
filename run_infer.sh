#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/mnt/Data_Partition/TCGA" # Change this to your base data path
DATA_DIR="$BASE_DIR/test_2samples" # Change this to your test data subfolder

WSI_DIR="$DATA_DIR/WSIs"
#JSON_PATH="$BASE_DIR/WSI-Bench-test-Report-only.jsonl"
JSON_PATH="$BASE_DIR/WSI-Bench-test.jsonl"
MAG_LEVEL="mag20x"
REGIONS_DIR="$DATA_DIR/regions_${MAG_LEVEL}"
FEATURES_DIR="$DATA_DIR/features_${MAG_LEVEL}"

#MODEL_DIR="../MLLM_HWSI_ckpts3/best_hf_stage3_model/"
MODEL_DIR="Bastech/MLLM-HWSI"

TEST_TYPE="Classification"  # Set to "Caption", "Morphology", "Classification", or "Report" 
TASK="eval"  # Set to "eval" or "retrieval"

OUTPUT_JSON="$DATA_DIR/result3.jsonl"

echo "===== Running Inference ====="
python main_test.py \
  --wsi_dir "$WSI_DIR" \
  --gt_json_path "$JSON_PATH" \
  --test_type "$TEST_TYPE" \
  --task "$TASK" \
  --wsi_feat_dir "$FEATURES_DIR/wsi" \
  --region_feat_dir "$FEATURES_DIR/region_4k" \
  --patch_feat_dir "$FEATURES_DIR/patches_filtered" \
  --cell_feat_dir "$FEATURES_DIR/cells" \
  --cell_pt_filename encoded_cell_features.pt \
  --output_json "$OUTPUT_JSON" \
  --model_name "$MODEL_DIR" \
  --device cuda \
  --num_slide_tokens 64 \
  --max_new_tokens 256 \
  --do_sample \
  --num_beams 1 \
  --temperature 0.75 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --no_repeat_ngram_size 4 

  

echo "===== Inference Finished Successfully ====="