# 實驗室工作流程說明(NSYSU AI Visual Lab)

> 本文件記錄本實驗室在 MLLM-HWSI 上的**實際執行流程與本地新增內容**。
> 上游原始專案說明請見 [README.md](README.md)。

---

## 0. 這個 repo 是什麼

- **上游來源**:[BasitAlawode/MLLM-HWSI](https://github.com/BasitAlawode/MLLM-HWSI)(CVPR 2026)
- **本 fork 的用途**:在此基礎上做多尺度特徵抽取,並進行 WSI 持續學習(CL)/ 測試時自適應(CTTA)的前期診斷實驗
- **本地新增**:`Ours/`(推論協定與探針腳本)、`utils/`、批次執行腳本、本文件

### ⚠️ 未包含在 repo 中的內容

以下項目體積過大或屬資料集,**不上傳**,請依下方說明自行準備:

| 項目 | 路徑 | 取得方式 |
|---|---|---|
| 模型權重 | `checkpoints/` | 見 §2 |
| 特徵與推論輸出 | `outputs/` | 執行 §3 pipeline 產生 |
| WSI 原始影像 | `data/`、`*.svs` | 從 [GDC Portal](https://portal.gdc.cancer.gov/) 下載 |

---

## 1. 環境建置

```bash
conda create -y --name mllm_hwsi python==3.10.0
conda activate mllm_hwsi
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

實測環境:Python 3.10.20、torch 2.11.0+cu128、numpy 2.2.6、sklearn 1.7.2。

### 額外套件(前期診斷分析用)

```bash
pip install umap-learn
```

### 字型注意事項(繪圖含中文時)

matplotlib 解析 `.ttc` 字型集合檔時只認得 `Noto Sans CJK JP` 這個名稱。即使系統已安裝 Noto Sans CJK TC/SC,指定 TC/SC 仍會導致中文顯示為方框,須指定 JP 變體(繁體字仍正常顯示):

```python
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
```

---

## 2. 模型權重下載

建立 `checkpoints/` 並依下表放置:

| 子目錄 | 檔案 | 模型 | 來源 |
|---|---|---|---|
| `checkpoints/hipt/` | `vit_256_small_dino.pth` | HIPT ViT-S/16(384 維) | [HIPT](https://github.com/mahmoodlab/HIPT) |
| `checkpoints/hipt/` | `vit_4096_xs_dino.pth` | HIPT ViT-XS(192 維) | [HIPT](https://github.com/mahmoodlab/HIPT) |
| `checkpoints/conch/` | `pytorch_model.bin` | CONCH v1(`conch_ViT-B-16`) | [CONCH](https://github.com/mahmoodlab/CONCH) |
| `checkpoints/cellvit/` | `CellViT-256-x40-AMP.pth` | CellViT256 | [CellViT](https://github.com/TIO-IKIM/CellViT) |
| `checkpoints/mllm_hwsi/` | `model.safetensors`, `vl_projector.pt`, ... | MLLM-HWSI 微調權重 | HF `Bastech/MLLM-HWSI` |

---

## 3. 特徵抽取 Pipeline(四階段,須依序執行)

### Stage 1 — Region 切割

```bash
python ext_regions.py \
  --source   <WSI 目錄> \
  --save_dir <regions 輸出目錄> \
  --seg --patch --stitch
```

產出:`regions/patches/*.h5`(4096px region 座標)、`masks/`、`stitches/`

### Stage 2 — Patch / Region / WSI 特徵

```bash
python ext_feats_conch_hierar_par.py \
  --wsi_dir       <WSI 目錄> \
  --reports_path  <報告 json;推論資料填 None> \
  --h5_dir_4096   <regions 輸出目錄>/patches \
  --hipt_repo     ./HIPT_4K \
  --checkpoint256 checkpoints/hipt/vit_256_small_dino.pth \
  --checkpoint4k  checkpoints/hipt/vit_4096_xs_dino.pth \
  --trident_repo  ./trident \
  --conch_ckpt_path checkpoints/conch/pytorch_model.bin \
  --encoder_name  conch_v1 \
  --data_mode     <train 或 inference> \
  --out_dir       <features 輸出目錄> \
  --extract_patch_features \
  --all_gpus --workers_per_gpu 1
```

產出:

| 子目錄 | 內容 | Shape |
|---|---|---|
| `features/patches/` | CONCH 全部 patch 特徵 | `[num_region, 256, 512]` |
| `features/patches_filtered/` | 語意篩選後的 top-k patch | `[num_region, k, 512]` |
| `features/region_4k/` | HIPT region 特徵 | `[num_region, 192]` |
| `features/wsi/` | WSI 特徵 | `[192]` |
| `features/coords_region4096_valid/` | 有效 region 座標 | `.h5` |
| `features/metadata/` | 每張 slide 的抽取參數與 shape | `.json` |

### Stage 3 — Cell 特徵

```bash
python ext_cell_feat_par.py \
  --wsi_dir               <WSI 目錄> \
  --region_coords_dir     <features 目錄>/coords_region4096_valid \
  --selected_indices_dir  <features 目錄>/patches_filtered \
  --checkpoint            checkpoints/cellvit/CellViT-256-x40-AMP.pth \
  --output_dir            <features 目錄>/cells \
  --all_gpus --workers_per_gpu 1 --enforce_amp
```

產出:`cells/<slide_id>/encoded_cell_features.pt`
- `encoded_cell_features`:`[num_region, k, 384]`
- `selected_patch_cell_level_indices`、`region_coords_level0`、`region_has_detail_dump`、`feature_dim`

### Stage 4(選配)— 視覺化用的 patch 影像

```bash
python save_patch_images.py \
  --wsi_dir     <WSI 目錄> \
  --coords_dir  <features 目錄>/coords_region4096_valid \
  --indices_dir <features 目錄>/patches_filtered \
  --output_dir  <features 目錄>/sample_images \
  --region_size 4096 --patch_size 256 \
  --max_wsi 2 --max_regions_per_wsi 20 --selection_mode random
```

---

## 4. 四個尺度的 Encoder 對照表

| 尺度 | Encoder | 維度 |
|---|---|---|
| **Cell** | CellViT256(骨幹 HIPT ViT-S/16)偵測細胞 → CCAF 融合 | 384 |
| **Patch** | CONCH v1(`conch_ViT-B-16`,768 維 trunk → `attn_pool_contrast` 投影) | 512 |
| **Region** | HIPT 兩層:ViT-S/16(384)編碼 patch → ViT-XS(192)聚合 16×16 網格 | 192 |
| **WSI** | **無獨立模型**,見下方註記 | 192 |
| **LLM** | Qwen2.5-7B-Instruct(`Qwen2ForCausalLM`,hidden 3584,28 層) | 3584 |

### ⚠️ 註記一:WSI 特徵是 region 的算術平均

`ext_feats_conch_hierar_par.py:501`:

```python
wsi_feat = region_cls.mean(dim=0)
```

實測驗證 `region.mean(dim=0)` 與存檔的 `wsi.pt` **逐元素完全相等(最大絕對差 = 0.0)**。
亦即 WSI 尺度不是獨立學習的表徵,而是 region 的 mean pooling,**在 WSI 層級分析中不提供額外資訊**。架構上目前沒有可學習的 WSI 層級聚合器。

### ⚠️ 註記二:CCAF 使用未經訓練的現成權重

`ext_cell_feat_par.py` 的 `CellToCellAttentionFusionViTS` 直接載入 **DINOv2 ViT-S/14**(`torch.hub facebookresearch/dinov2`),借用其 CLS token + blocks + norm 作為 attention 融合,**未經病理資料訓練或微調**。ViT-H 版本 docstring 亦自述:

> "No reducer to 768 is included, because I could not verify a generic already-trained adapter module for arbitrary 1280-d ViT features."

解讀 cell 尺度的實驗結果時須考慮此領域落差。

---

## 5. 訓練

```bash
# 編輯 run_train.sh 中的 MODEL_DIR / BASE_DIR / TRAIN_SUB_FOLDER
./run_train.sh 2      # Stage 2:訓練 V-L Projector
./run_train.sh 3      # Stage 3:指令微調
./run_train.sh 2 3    # 連續執行
```

Stage 1(編碼器預訓練)請參考 HIPT / CONCH / CellViT 各自的專案。

---

## 6. 推論

### 上游標準推論

```bash
# 編輯 run_infer.sh 中的 BASE_DIR / DATA_DIR / MODEL_DIR / TEST_TYPE
./run_infer.sh
```

`TEST_TYPE` 可選:`Caption`、`Morphology`、`Classification`、`Report`

### 本地新增:`Ours/` 實驗協定

| 腳本 | 用途 |
|---|---|
| `Ours/run_rcc_pilot20_k16.py` | RCC pilot20(k=16)三提示分類實驗 |
| `Ours/run_pilot40_four_conditions.py` | BRCA/RCC pilot40(k=48)四條件實驗 |
| `Ours/rcc_k16_protocol.py` | RCC k16 協定定義(凍結 prompt、評分邏輯) |
| `Ours/pilot40_protocol.py` | pilot40 協定定義 |
| `Ours/probe_features_vs_projector.py` | 探針:確認 WSI 特徵是否真的影響 projector 輸出(只載 `vl_projector.pt`,不載 7B LLM) |
| `Ours/probe_feature_swap.py` | 探針:固定 prompt、交換兩張 slide 的四尺度特徵,檢查輸出是否隨之改變 |
| `Ours/ablation.py` | 消融實驗 |
| `Ours/tests/` | 單元測試 |

執行範例:

```bash
# 先 preflight 檢查
bash run_rcc_pilot20_k16.sh --preflight-only

# 單張 slide smoke test
bash run_rcc_pilot20_k16.sh --limit 1

# 完整執行
bash run_rcc_pilot20_k16.sh
```

```bash
bash run_pilot40_four_conditions.sh
```

> ⚠️ 兩個 `.sh` 腳本內含**絕對路徑**(`PROJECT_ROOT`、`--selection-csv`),在別台機器上執行前須先修改。

單元測試:

```bash
python -m unittest discover -s Ours/tests -v
```

詳細的 RCC k16 實驗說明(含凍結的三個 prompt 原文)見 [RCC_K16_INFERENCE_README.md](RCC_K16_INFERENCE_README.md)。

---

## 7. 已知問題與注意事項

1. **`.sh` 腳本有絕對路徑**,換機器須改(見 §6)。
2. **WSI 尺度與 region 尺度數值完全相同**,做多尺度比較時不應視為兩個獨立證據(見 §4 註記一)。
3. **CCAF 未經訓練**,cell 尺度結果的解讀須保留(見 §4 註記二)。
4. **每張 WSI 的 region 數量差異極大**(實測 17 ~ 563),做 WSI 層級比較時必須明確說明聚合方式(mean pooling 或保留 instance)。
5. `data/inference_placeholder.jsonl` 是空的佔位檔,不含任何病患資料。

---

## 8. 授權與引用

本 repo 為 [BasitAlawode/MLLM-HWSI](https://github.com/BasitAlawode/MLLM-HWSI) 的 fork。上游專案未附 LICENSE 檔,**使用與再散布前請先向原作者確認授權條件**。

本專案亦間接使用以下模型,各自有獨立授權條款(部分為非商業用途限定),請分別確認:

- [HIPT](https://github.com/mahmoodlab/HIPT)
- [CONCH](https://github.com/mahmoodlab/CONCH)
- [CellViT](https://github.com/TIO-IKIM/CellViT)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

資料來源為 [TCGA](https://portal.gdc.cancer.gov/),使用時請遵守 NIH/GDC 的資料使用規範。
