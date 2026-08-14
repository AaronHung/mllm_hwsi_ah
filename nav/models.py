"""Gate 1–3 的兩個小模型：診斷 evaluator 與 navigation scorer。

尺寸刻意小（<1M 參數），Mac CPU/MPS 可訓練；RunPod 只是跑更多 seed/fold。
"""

from __future__ import annotations

import torch
import torch.nn as nn

PATCH_DIM = 512   # CONCH patch 特徵（zoom 後才可見的高倍證據）
REGION_DIM = 192  # HIPT region 特徵（低倍全可見）


class Evaluator(nn.Module):
    """診斷頭：聚合後的高倍證據 (in_dim,) -> n_classes。"""

    def __init__(self, n_classes: int, hidden: int = 256, in_dim: int = PATCH_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, n_classes))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class Navigator(nn.Module):
    """共享導航策略：對每個候選 region 打分。

    輸入 = 候選 region 的低倍特徵 (low_dim) ⊕ context（目前證據 mean high_dim ⊕ 剩餘 budget 1）。
    輸出 = 每個候選一個 logit；policy = softmax over 未 zoom 的候選。
    預設維度 = pilot40（HIPT 192 / CONCH 512）；can_dataset 用 low_dim=64。
    """

    def __init__(self, hidden: int = 256, low_dim: int = REGION_DIM,
                 high_dim: int = PATCH_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(low_dim + high_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1))

    def forward(self, region_low: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """region_low: [R,192]；context: [512+1]（同一 state 下所有候選共用）。"""
        ctx = context.expand(region_low.shape[0], -1)
        return self.net(torch.cat([region_low, ctx], dim=1)).squeeze(-1)
