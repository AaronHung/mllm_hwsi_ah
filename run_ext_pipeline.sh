#!/usr/bin/env bash
set -euo pipefail

echo "===== [1/4] Extracting regions ====="
python ext_regions.py \
  --source ../TCGA_data/WSIs_vqa \
  --save_dir ../TCGA_data/regions_mag20x_vqa \
  --mag_level 20 \
  --patch_size 4096 \
  --seg \
  --patch \
  --stitch
echo "===== Completed: Region Extraction ====="

echo "===== [2/4] Extracting hierarchical CONCH + HIPT features ====="
python ext_feats_conch_hierar_par.py \
  --wsi_dir ../TCGA_data/WSIs_vqa \
  --reports_path ../TCGA_data/WSI-Bench-train-Report-only.jsonl \
  --h5_dir_4096 ../TCGA_data/regions_mag20x_vqa/patches \
  --hipt_repo ./HIPT_4K \
  --checkpoint256 ../HIPT_ckpts/vit_256_small_dino.pth \
  --checkpoint4k ../HIPT_ckpts/vit_4096_xs_dino.pth \
  --trident_repo ./trident \
  --conch_ckpt_path ../CONCH_ckpt/pytorch_model.bin \
  --conch_batch_size 128 \
  --encoder_name conch_v1 \
  --conch_model_cfg conch_ViT-B-16 \
  --out_dir ../TCGA_data/features_mag20x_vqa \
  --patch_size 256 \
  --extract_patch_features \
  --n_diss_features 32 \
  --top_k 16 \
  --all_gpus \
  --workers_per_gpu 1
echo "===== Completed: Feature Extraction ====="

echo "===== [3/4] Saving sample images ====="
python save_patch_images.py \
  --wsi_dir ../TCGA_data/WSIs_vqa \
  --coords_dir ../TCGA_data/features_mag20x_vqa/coords_region4096_valid \
  --indices_dir ../TCGA_data/features_mag20x_vqa/patches_filtered \
  --output_dir ../TCGA_data/features_mag20x_vqa/sample_images \
  --region_size 4096 \
  --patch_size 256 \
  --mag_level 20 \
  --max_wsi 2 \
  --max_regions_per_wsi 20 \
  --selection_mode random
echo "===== Completed: Image Saving ====="

echo "===== [4/4] Extracting cell features ====="
python ext_cell_feat_par.py \
  --wsi_dir ../TCGA_data/WSIs_vqa \
  --region_coords_dir ../TCGA_data/features_mag20x_vqa/coords_region4096_valid \
  --selected_indices_dir ../TCGA_data/features_mag20x_vqa/patches_filtered \
  --checkpoint ../CELLVITpp_ckpts/CellViT-256-x40-AMP.pth \
  --output_dir ../TCGA_data/features_mag20x_vqa/cells \
  --batch_size 16 \
  --magnification 40 \
  --all_gpus \
  --workers_per_gpu 1 \
  --feature_mode mask_mean \
  --enforce_amp \
  --save_full_outputs_regions_per_wsi 10 \
  --full_output_region_selection random
echo "===== Completed: Cell Feature Extraction ====="

echo "===== Extraction Pipeline Finished Successfully ====="