# 實驗 B 結論（探索性結果；n=40 pilot，seed 已固定）

## 判讀（給開會用）

1. **小預算就有大部分訊號**：四分類 random K=4（每張只看約 1–2% 的 region）
   已達 78%，K=32 約 90%，接近 full-information 上限 95%（固定 C=1 的 LR；
   實驗 A 的 grid-C 版本為 92.5%）。→ 預算式導航的「省觀察」空間確實存在。
2. **手工啟發式選區並沒有贏過 random**：diversity-FPS 與 atypicality 在多數 K
   下低於 random——它們專挑特徵空間的離群 region（邊緣、脂肪、壞死等
   「怪但無診斷資訊」的區域）。spatial-uniform 與 random 大致打平。
   → 病理子型訊號在切片內高度冗餘、空間分佈廣，「挑異常」不等於「挑證據」。
3. **對 Gate 1 的含義**：導航策略必須從診斷訊號「學」出來
   （counterfactual diagnostic gain），手寫規則不夠；且 random 是很強的
   baseline，Gate 1 判準（優於 random/uniform）誠實而有挑戰性。
   若把每 region 的 48 個 patch 特徵納入 observation（實驗 B 只用 region mean），
   learned navigator 有更大的資訊面可以利用。

## 各策略數據

## 2class
- K=1: 最佳策略 diversity_fps = 94.2%，random = 93.5%，領先 +0.7 pp
- K=2: 最佳策略 diversity_fps = 87.5%，random = 91.0%，領先 -3.5 pp
- K=4: 最佳策略 spatial_uniform = 95.0%，random = 95.5%，領先 -0.5 pp
- K=8: 最佳策略 spatial_uniform = 100.0%，random = 98.0%，領先 +2.0 pp
- 小預算（K≤8）平均領先 random：-0.3 pp

## 4class
- K=1: 最佳策略 spatial_uniform = 60.0%，random = 57.0%，領先 +3.0 pp
- K=2: 最佳策略 spatial_uniform = 62.5%，random = 67.0%，領先 -4.5 pp
- K=4: 最佳策略 spatial_uniform = 77.5%，random = 78.0%，領先 -0.5 pp
- K=8: 最佳策略 spatial_uniform = 65.0%，random = 76.0%，領先 -11.0 pp
- 小預算（K≤8）平均領先 random：-3.2 pp
