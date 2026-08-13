# 實驗 A 第 0 步：資料盤點

- slide 總數：40
- 類別分佈：{'KIRC': 10, 'KIRP': 10, 'IDC': 10, 'ILC': 10}

## 抽樣檢查（每層 2 檔）
### TCGA-BQ-5878-01Z-00-DX1（KIRP）
- wsi (192,) torch.float32 | region (282, 192) | patch (282, 48, 512) | cell (282, 48, 384) | coords (282, 2)
- zero-cell slots 比例：16.0%
- wsi == region.mean 最大差：0.00e+00

### TCGA-BH-A0BQ-01Z-00-DX1（IDC）
- wsi (192,) torch.float32 | region (359, 192) | patch (359, 48, 512) | cell (359, 48, 384) | coords (359, 2)
- zero-cell slots 比例：17.6%
- wsi == region.mean 最大差：0.00e+00

## 交叉比對（40 張在四層 + coords 是否齊）
- 總 region 數：7656（預期 7656）
- 缺漏：無，40/40 五層齊全

**盤點結論：PASS**