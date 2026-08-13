# WSI 多尺度特徵 — 前期診斷實驗報告

- **日期**:2026-08-13
- **性質**:確認性實驗(confirmatory pilot),不是論文成果
- **目的**:確認 (a) 域偏移問題是否真實存在、(b) 若存在,集中在哪個尺度、(c) cell 尺度是否有獨立價值
- **範圍**:純特徵層面分析,**未訓練任何 CTTA/CL 模型**

---

## 0. 一句話結論

在 40 張 TCGA WSI 上,**機構(source site)造成的特徵偏移確實存在,且在三個亞型中都集中在 region 尺度**(cell/patch 相對較輕),與原先「cell/patch 偏移最大」的假設方向相反。但此結論**尚有一個未排除的混淆**(見 §7.1),且樣本數極小,僅為 proof-of-concept。

---

## 1. 資料來源與盤點

### 1.1 特徵來源

```
mllm_hwsi_demo/MLLM-HWSI/outputs/brca_pilot40_k48/
```

40 張 WSI,KIRC / KIRP / IDC / ILC 各 10 張,清單見 `pilot40_selection.csv`。
已驗證:**40 張的四個尺度特徵檔全部齊全,無缺檔**。

### 1.2 四個尺度的實際 shape(已驗證)

| 尺度 | 檔案路徑 | Shape | 維度 |
|---|---|---|---|
| Cell | `cells_k48/<slide>/encoded_cell_features.pt` → `encoded_cell_features` | `[num_region, 48, 384]` | 384 |
| Patch | `features/patches_filtered/<slide>.pt` → `selected_features` | `[num_region, 48, 512]` | 512 |
| Region | `features/region_4k/<slide>.pt` | `[num_region, 192]` | 192 |
| WSI | `features/wsi/<slide>.pt` | `[192]` | 192 |

- 每張 WSI 的 region 數量差異極大:**17 ~ 563**(總計 7656 個 region)
- 每個 region 固定選 48 個 patch / 48 個 cell token(k=48)

### 1.3 各亞型的 instance(region)總數

| 亞型 | instance 數 |
|---|---|
| KIRC | 2150 |
| KIRP | 2881 |
| IDC | 1758 |
| ILC | 867 |

---

## 2. 特徵抽取模型的確認

### 2.1 MLLM(語言端)

| 項目 | 內容 |
|---|---|
| LLM backbone | **Qwen2.5-7B-Instruct**(`Qwen2ForCausalLM`, hidden_size=3584, 28 層) |
| 微調 checkpoint | `Bastech/MLLM-HWSI`(HF)/ 本地 `checkpoints/mllm_hwsi/` |
| 投影器 | 每尺度一個 MLP:LayerNorm → Linear(in,1024) → GELU → Dropout → Linear(1024,3584),`num_query_tokens=64` |

### 2.2 四個尺度的視覺 Encoder

**Cell(384 維)** — 兩段式
1. **CellViT256**(`checkpoints/cellvit/CellViT-256-x40-AMP.pth`,log 確認 `arch=CellViT256`),骨幹為 HIPT ViT-S/16,輸出每顆細胞 384 維
2. **CCAF 融合**:`CellToCellAttentionFusionViTS` — 直接載入 **DINOv2 ViT-S/14**(`torch.hub facebookresearch/dinov2 dinov2_vits14`),借用其 CLS token + transformer blocks + norm,對一個 patch 內多顆細胞做 attention 融合

**Patch(512 維)** — **CONCH v1**(`conch_ViT-B-16`, Mahmoodlab)
- 已開啟 checkpoint 驗證:trunk 為 768 維 ViT-B/16,經 `attn_pool_contrast` 投影至 512 維
- 512 維與資料吻合 → 確定是 v1 而非 v1.5(v1.5 為 768 維)

**Region(192 維)** — **HIPT 兩層**
- `vit_256_small_dino.pth`(ViT-S/16, 384 維)編碼每個 256px patch
- `vit_4096_xs_dino.pth`(ViT-XS, **192 維**)聚合 16×16 patch token 網格 → region 向量

**WSI(192 維)** — **無獨立模型**(見 §4)

### 2.3 ⚠️ 重要限制:CCAF 未經訓練

此 repo 的 CCAF **不是論文中訓練出來的模組**,而是直接套用凍結的 DINOv2 blocks。證據為 ViT-H 版本 docstring 自述:

> "No reducer to 768 is included, because I could not verify a generic already-trained adapter module for arbitrary 1280-d ViT features."

亦即 cell 尺度特徵是「用**自然影像**預訓練的 DINOv2 blocks,去融合 CellViT 抽出的細胞 embedding」,兩者間存在領域落差,且未經任何病理資料訓練或微調。此點對 §6 的結論解讀有實質影響(見 §7.2)。

---

## 3. 實驗 1:隱藏域標籤檢查(決定走哪條路線)

### 3.1 方法

解析 TCGA barcode `TCGA-XX-YYYY-...` 的第二段 `XX` = Tissue Source Site(TSS,機構代碼),建立「亞型 × site」交叉表。

### 3.2 結果

**四個亞型全部橫跨多個 site,並非綁死單一機構:**

| 亞型 | 橫跨 site 數 | 分布 |
|---|---|---|
| IDC | 6 | A1:1, A2:1, A8:2, AO:2, B6:1, BH:3 |
| ILC | 5 | A2:1, A7:2, A8:1, **AC:5**, AR:1 |
| KIRC | 7 | A3:1, AK:1, B2:1, B8:1, **BP:4**, CJ:1, CZ:1 |
| KIRP | 6 | **2Z:3**, B9:1, **BQ:3**, G7:1, O9:1, UZ:1 |

**判定:符合走路線 A(同亞型內域偏移診斷)的條件。**
完整表格:`analysis/tables/subtype_x_site_crosstab.csv`

### 3.3 域標籤定義

- 主要切分:**「該亞型的多數 site」vs「其餘 site」**(KIRC=BP, KIRP=2Z, IDC=BH, ILC=AC)
- 額外對照:KIRP 有天然平衡的 **2Z(n=3) vs BQ(n=3)**,另算一組

---

## 4. 關鍵發現:WSI 尺度 = region 的平均池化(冗餘)

### 4.1 證據

抽特徵原始碼 `ext_feats_conch_hierar_par.py:501`:

```python
wsi_feat = region_cls.mean(dim=0)   # WSI 特徵 = 所有 region 向量的算術平均
```

實測驗證:`region.mean(dim=0)` 與存檔的 `wsi.pt` **逐元素完全相等,最大絕對差 = 0.0**(非近似,是位元對位元相同)。

### 4.2 影響

- 「四尺度」實際上只有 **cell / patch / region 三個獨立尺度**
- 在 WSI-level mean-pooling 的分析方式下,wsi 尺度**不提供任何新資訊**
- 所有 region 與 wsi 的 UMAP 圖、定量數字必然完全相同
- 後續所有表格中 wsi 一列均標註「= mean(region),冗餘」,不應視為獨立證據
- **架構意涵**:現行架構在 WSI 層級沒有任何可學習的聚合器。若希望 CTTA 能在整張片子層級調整,需另外引入(如 attention pooling),但那會偏離原始 MLLM-HWSI 流程

---

## 5. 實驗 2(A.1):定性分析 — UMAP

### 5.1 方法

- **Pooling**:WSI-level,每張 WSI 一個代表向量
  - cell:`[num_region, 48, 384]` → 對 (region, cell) 兩維取 mean → `[384]`
  - patch:`[num_region, 48, 512]` → 對 (region, patch) 兩維取 mean → `[512]`
  - region:`[num_region, 192]` → 對 region 取 mean → `[192]`
  - wsi:本身即單一向量
- **UMAP 參數**:`n_components=2`, `min_dist=0.3`, `metric=euclidean`, **`random_state=42`(固定)**
  - `n_neighbors = min(5, n-1)`:因每亞型僅 n=10,遠低於 UMAP 預設的 15,被迫縮小
- **未做標準化**:直接以原始向量計算歐氏距離(尚未加 StandardScaler)

### 5.2 產出

| 圖 | 內容 |
|---|---|
| `A1_umap_cell_by_site.png` | cell 尺度,2×2 子圖(四亞型),依 site 上色 |
| `A1_umap_patch_by_site.png` | patch 尺度,同上 |
| `A1_umap_region_by_site.png` | region 尺度,同上 |
| `A1_umap_wsi_by_site.png` | wsi 尺度(與 region 圖完全相同) |
| `class_sanitycheck_umap_all40_by_class.png` | 全 40 張同一 UMAP 空間,依**亞型**上色(額外 sanity check) |

### 5.3 觀察

**依 site 上色(A.1)**:三個獨立尺度中,同色點大致與異色點混雜,未形成乾淨分群。

**依亞型上色(sanity check)**:
- **器官層級(腎 vs 乳)分界清楚**,尤其在 region/wsi 尺度 — sanity check 通過
- **同器官內的細微亞型(KIRC vs KIRP、IDC vs ILC)分得不乾淨**,KIRC/KIRP 在四個尺度都高度混雜
- cell 尺度是四者中器官分界最模糊的

### 5.4 ⚠️ 此步驟的可信度限制

`n_neighbors=5` 而總點數僅 10(扣除自身剩 9 個候選)→ 超過一半的其他點都被算作「鄰居」,局部與全局結構的界線幾乎消失。**UMAP 在此樣本量下極不穩定,視覺上「看不出分群」很可能是假陰性**,不能作為結論依據,必須由 §6 的定量指標交叉驗證。

### 5.5 UMAP 座標的解讀限制

- X/Y 軸(UMAP-1/2)**無物理意義**,不對應任何原始特徵維度
- 僅「點與點的相對靠近程度」有意義;UMAP 只保真局部結構,**不保真全局距離**
- **不同子圖之間的座標不可互相比較** — 每張圖是各自獨立的隨機優化,原點/旋轉/鏡像/縮放都由該次優化自行決定。因此無法從這些圖回答「同一張 WSI 在三個尺度是否落在相同位置」

---

## 6. 實驗 3(A.2):定量分析 — A-distance 與 MMD

### 6.1 指標定義

**proxy A-distance**
訓練一個域分類器去猜「這筆資料來自哪個 site」,猜得越準代表該尺度混入越多機構痕跡。

```
A-distance = 4 × accuracy − 2
```
範圍 0~2(acc=0.5 → 0 等於亂猜;acc=1.0 → 2 完美分開)。acc < 0.5 時會出現小幅負值,屬正常估計波動。

**MMD²(Maximum Mean Discrepancy)**
不訓練分類器,直接比較兩組樣本的整體分布距離。
- 實作:自行以 **RBF kernel** + **median heuristic** 決定 bandwidth,**unbiased 估計量**
- 計算前先對合併資料做 StandardScaler
- 兩組樣本各上限 800 筆(超過則隨機抽樣,seed 固定)
- unbiased 估計量在真實 MMD≈0 時可能出現小幅負值(如 ILC patch = −0.0026)

### 6.2 分類器設定(嚴格遵守小樣本規範)

- 模型:**logistic regression**(L2 正則,C=1.0,max_iter=2000)+ StandardScaler
- **未使用任何深層網路或大模型**
- 交叉驗證:
  - **WSI-level**:`RepeatedStratifiedKFold`,3-fold × 30 repeats
  - **instance-level**:`StratifiedGroupKFold`,3-fold × 30 repeats,**以 slide_id 分組**,確保同一張 WSI 的所有 region 同進同出,避免資訊洩漏
- 報告 30 次重複的**平均值 ± 標準差**

### 6.3 已修正的 bug

初次執行時,KIRP 的 instance-level 分析在某些切分下,少數 site 的 3 張 WSI 全部落入同一 fold,導致訓練集只剩單一類別而崩潰。已修正為**跳過此類異常切分且不計入平均**(`domain_shift_utils.py`)。

### 6.4 結果 — instance-level(**主要判讀依據**)

每個 4096px region 一筆,group-CV by WSI。

| 亞型 | 尺度 | 猜對率 | A-distance | MMD² |
|---|---|---|---|---|
| KIRC | cell | 0.837 ± 0.069 | 1.348 ± 0.278 | 0.2080 |
| KIRC | patch | 0.807 ± 0.082 | 1.226 ± 0.328 | 0.0962 |
| KIRC | **region** | **0.945 ± 0.046** | **1.779 ± 0.183** | 0.2780 |
| KIRP | cell | 0.638 ± 0.086 | 0.553 ± 0.345 | 0.1504 |
| KIRP | patch | 0.710 ± 0.095 | 0.839 ± 0.378 | 0.1266 |
| KIRP | **region** | **0.795 ± 0.133** | **1.180 ± 0.531** | 0.1615 |
| IDC | cell | 0.623 ± 0.158 | 0.492 ± 0.634 | 0.1464 |
| IDC | patch | 0.557 ± 0.133 | 0.229 ± 0.532 | 0.1587 |
| IDC | region | 0.512 ± 0.168 | 0.047 ± 0.673 | 0.1450 |
| ILC | cell | 0.666 ± 0.125 | 0.664 ± 0.500 | 0.1607 |
| ILC | patch | 0.566 ± 0.156 | 0.262 ± 0.622 | 0.1113 |
| ILC | **region** | **0.700 ± 0.139** | **0.800 ± 0.556** | 0.1730 |

### 6.5 結果 — WSI-level(n=10/亞型,僅供方向參考)

| 亞型 | 尺度 | 猜對率 | A-distance | MMD² |
|---|---|---|---|---|
| KIRC | cell | 0.839 ± 0.076 | 1.356 ± 0.304 | 0.1794 |
| KIRC | patch | 0.580 ± 0.149 | 0.319 ± 0.595 | 0.0576 |
| KIRC | region (=wsi) | 0.994 ± 0.021 | 1.978 ± 0.083 | 0.2558 |
| KIRP | cell | 0.764 ± 0.109 | 1.056 ± 0.435 | 0.0815 |
| KIRP | patch | 0.826 ± 0.110 | 1.304 ± 0.442 | 0.1474 |
| KIRP | region (=wsi) | 0.888 ± 0.061 | 1.552 ± 0.244 | 0.1101 |
| IDC | cell | 0.900 ± 0.045 | 1.600 ± 0.181 | 0.1638 |
| IDC | patch | 0.779 ± 0.076 | 1.115 ± 0.304 | 0.0893 |
| IDC | region (=wsi) | 0.680 ± 0.073 | 0.719 ± 0.294 | 0.0931 |
| ILC | cell | 0.569 ± 0.092 | 0.274 ± 0.369 | 0.1021 |
| ILC | patch | 0.484 ± 0.111 | −0.063 ± 0.445 | −0.0026 |
| ILC | region (=wsi) | 0.667 ± 0.135 | 0.667 ± 0.539 | 0.0327 |

### 6.6 KIRP 平衡版對照(2Z n=3 vs BQ n=3)

| 層級 | 尺度 | 猜對率 | A-distance |
|---|---|---|---|
| WSI-level | cell | **1.000 ± 0.000** | 2.000 ± 0.000 |
| WSI-level | patch | 0.789 ± 0.074 | 1.156 ± 0.295 |
| WSI-level | region (=wsi) | **1.000 ± 0.000** | 2.000 ± 0.000 |
| instance | cell | 0.680 ± 0.192 | 0.721 ± 0.769 |
| instance | patch | 0.650 ± 0.293 | 0.598 ± 1.173 |
| instance | region | 0.989 ± 0.018 | 1.955 ± 0.070 |

### 6.7 ⚠️ 為何 WSI-level 數字不可信

WSI-level 是「**10 個樣本 vs 192~512 維特徵**」——樣本數遠少於特徵維度。此情況下即使做交叉驗證,logistic regression 也極易在小測試集上巧合分對。

最明顯的紅色警訊是 **KIRP 平衡版在 cell 與 region 尺度出現 100.0% ± 0.000 的零變異滿分**(n=3 vs 3)。這在統計上更像過擬合訊號,而非穩健的域偏移證據。

**因此 §6.4 的 instance-level 結果(n=343~2204,樣本數遠大於維度)才是建議引用的版本**;WSI-level 僅供確認「方向是否一致」。

完整表格:`analysis/tables/A2_domain_shift_quantitative.csv`

---

## 7. 結論

### 7.1 主要結論

**(1) 機構造成的偏移確實存在,但嚴重程度依亞型而異**

| 亞型 | 偏移程度 | instance-level 猜對率範圍 |
|---|---|---|
| KIRC | 最明顯 | 81 ~ 94% |
| KIRP | 中等 | 64 ~ 80% |
| ILC | 中等 | 57 ~ 70% |
| IDC | **幾乎測不到** | 51 ~ 62%(region 僅 51.2%,等同亂猜) |

**(2) 偏移集中在 region(=wsi)尺度,不是 cell**

四個亞型中有三個(KIRC、KIRP、ILC),region 都是最容易被認出 site 的尺度。
**這與原先假設(「cell/patch 偏移 > region/WSI,因染色/掃描差異直接體現在細胞紋理」)方向相反。**

**(3) cell 尺度並未表現出「乾淨、免疫於機構干擾」的優勢**
cell 普遍低於 region 但未接近亂猜水準(IDC 除外),屬「中等」而非「安全」。

### 7.2 ⚠️ 尚未排除的混淆(重要)

A-distance 高只代表「分類器容易從該尺度認出 site」,可能來自兩種原因:

- **(a)** region 真的混入較多掃描/染色痕跡 → 現行結論成立
- **(b)** region 特徵本身表達力較強,導致**任何**標籤都較好預測 → 結論須打折

**目前尚未做對照實驗排除 (b)。** 建議的驗證方式:同一套流程,把標籤從 site 換成**亞型**(或隨機標籤)重算一次。
- 若 region 對亞型的預測力也同比例偏高 → (b) 成立
- 若 region 僅對 site 特別敏感 → (a) 成立

此對照同時能檢驗 §2.3 的疑慮:若 cell 尺度**同時**在「猜 site」與「猜亞型」都明顯偏弱,則傾向是未訓練的 CCAF 把訊號糊掉了,而非「cell 特徵天生乾淨」。

### 7.3 其他限制

- 所有數字來自**同一批 40 張 WSI**,site 分組是「多數 vs 其餘」湊出來的二分類,非嚴謹多中心對照設計
- IDC 測不到偏移,不代表 IDC 真的沒問題,也可能是這批樣本各機構染色風格恰好接近
- 本階段為**小樣本 proof-of-concept,統計檢定力有限**,結論方向可參考,數字大小不宜看死

---

## 8. 對 CTTA 設計的初步啟示

> 以下建議建立在 §7.2 的 (a) 成立之前提上,該前提**尚未驗證**。

1. **適應作用點放在 region 表徵**,而非 cell。注意 HIPT 為 ViT 架構、使用 **LayerNorm 而非 BatchNorm**,經典 TENT 的「更新 BN 統計量」做法無法直接套用;可行替代為更新 LN 仿射參數,或 prompt/adapter 路線。

2. **cell 當「錨」而非主要適應對象**:region 允許較大幅度適應,cell 凍結或加強正則,並用 cell 輸出作一致性約束(若適應後 region 與 cell 判斷嚴重衝突,可作為煞車訊號)。

3. **適應強度需可關閉**:IDC 幾乎無偏移(region 僅 51.2%),永遠開啟的無條件適應在此類情況下純屬無用功,且會累積誤差(CTTA 已知失效模式)。建議加入偏移偵測閘門,可用線上版 MMD 作為訊號。

4. **wsi 尺度無需獨立處理**(適應 region 即自動改變 wsi),但這也暴露架構缺口:WSI 層級無可學習聚合器,若需該層級的適應能力須另行引入。

---

## 9. 產出物清單

### 表格 `analysis/tables/`
| 檔案 | 內容 |
|---|---|
| `subtype_x_site_crosstab.csv` | 亞型 × source site 交叉表 |
| `wsi_labels.csv` | 40 張 WSI 的 class/organ/tss/site_group/num_regions 標籤 |
| `wsi_level_features.npz` | WSI 層級 mean-pooled 特徵(cell/patch/region/wsi) |
| `instance_level_features.npz` | instance 層級特徵(7656 × cell/patch/region) |
| `instance_level_meta.csv` | instance 層級對應標籤 |
| `A2_domain_shift_quantitative.csv` | A.2 完整定量結果(A-distance + MMD) |

### 圖 `analysis/figures/`
| 檔案 | 內容 |
|---|---|
| `A1_umap_cell_by_site.png` | cell 尺度依 site 上色 UMAP |
| `A1_umap_patch_by_site.png` | patch 尺度依 site 上色 UMAP |
| `A1_umap_region_by_site.png` | region 尺度依 site 上色 UMAP |
| `A1_umap_wsi_by_site.png` | wsi 尺度(與 region 相同) |
| `class_sanitycheck_umap_all40_by_class.png` | 全 40 張依亞型上色 sanity check |

### 程式 `analysis/scripts/`
| 檔案 | 用途 |
|---|---|
| `plot_config.py` | matplotlib CJK 字型設定(須用 `Noto Sans CJK JP`,見備註) |
| `domain_shift_utils.py` | A-distance / MMD 共用函式 |
| `01_build_wsi_level_features.py` | 建立 WSI 層級 mean-pooled 特徵與標籤 |
| `02_umap_by_site_within_subtype.py` | A.1 定性 UMAP |
| `03_build_instance_level_features.py` | 建立 instance 層級特徵 |
| `04_umap_all40_by_class.py` | 全 40 張依亞型 sanity check UMAP |
| `05_a2_quantitative_domain_shift.py` | A.2 定量分析主腳本 |

---

## 10. 環境備註

- 使用 conda env:`/home/alan0804/miniconda3/envs/mllm_hwsi`(Python 3.10.20)
- 本次**新裝**:`umap-learn 0.5.12`(+ 相依 `pynndescent 0.6.0`)
- 既有:torch 2.11.0+cu128, numpy 2.2.6, sklearn 1.7.2, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.8
- **字型踩雷**:系統雖裝有 Noto Sans CJK TC/SC,但 matplotlib 解析 `.ttc` 字型集合檔時只認得 `Noto Sans CJK JP` 這個名稱。指定 TC/SC 會導致中文顯示為方框,須指定 JP 變體(繁體字仍正常顯示)。

---

## 11. 待辦

- [ ] **(優先)** §7.2 的對照實驗:把標籤從 site 換成亞型重算,排除「region 表達力較強」的混淆
- [ ] 四尺度亞型判別力分析(原 CLAUDE.md 路線 B:KIRC vs KIRP、IDC vs ILC 的分類器準確率 + 信心校正)
- [ ] 考慮在 UMAP 前加入 StandardScaler,確認結果是否穩健
- [ ] (選配)跨尺度鄰居一致性分析:k-NN 鄰居重疊率 / 距離矩陣相關性(Mantel test),回答「三個尺度是否對『誰跟誰像』有一致判斷」
- [ ] (選配)視覺–文字對齊視覺化(需確認是否有對應診斷文字)
