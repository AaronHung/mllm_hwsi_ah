# 實驗 A 結論（探索性結果，非臨床效能宣稱；n=40 pilot）

- 二分類最佳 probe：cell/logreg，LOO accuracy = 97.5%
  （frozen MLLM 固定 prompt 為 55.0%，chance 50%）。
- 四分類最佳 probe：region/logreg，LOO accuracy = 92.5%
  （frozen MLLM 固定 prompt 為 20.0%，chance 25%）。
- 判讀：四分類 probe 最佳 = 92%，對照 frozen MLLM 的 20% → **特徵含有子型訊號，問題定位在 projector/LLM 對齊層（Stage-I 缺口）**。
- 注意：wsi 層 = region 層特徵的算術平均，兩者非獨立證據；cell 層 CCAF 使用
  未經病理微調的 DINOv2 權重，解讀需保留。
- 所有隨機過程 seed 固定（inventory seed=0；5-fold seeds=0,1,2）。
