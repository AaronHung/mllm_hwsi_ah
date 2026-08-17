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
                 high_dim: int = PATCH_DIM, film_tasks: int | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(low_dim + high_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1))
        # Track C0 (docs/track_c0.md §2.2): per-task diagonal FiLM adapters,
        # off by default.  Built AFTER self.net on purpose — the core's
        # initialization must consume the RNG stream in exactly the pre-C0
        # order, and ones/zeros draw no random numbers at all, so a C0
        # navigator's core starts from the same weights a pre-C0 one would.
        self.film_tasks = film_tasks
        self.active_task: int | None = None
        if film_tasks is not None:
            for name, dim, fill in (("film_gamma1", hidden, 1.0),
                                    ("film_beta1", hidden, 0.0),
                                    ("film_gamma2", hidden // 4, 1.0),
                                    ("film_beta2", hidden // 4, 0.0)):
                setattr(self, name, nn.ParameterList(
                    [nn.Parameter(torch.full((dim,), fill))
                     for _ in range(film_tasks)]))

    def film_parameters(self, task: int | None = None) -> list[nn.Parameter]:
        """FiLM parameters, all tasks or one task. Empty when the flag is off."""
        if self.film_tasks is None:
            return []
        tasks = range(self.film_tasks) if task is None else [task]
        return [getattr(self, name)[t] for t in tasks
                for name in ("film_gamma1", "film_beta1",
                             "film_gamma2", "film_beta2")]

    def core_parameters(self) -> list[nn.Parameter]:
        """The shared core, i.e. everything that is not a FiLM adapter."""
        return list(self.net.parameters())

    def forward(self, region_low: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """region_low: [R,192]；context: [512+1]（同一 state 下所有候選共用）。"""
        ctx = context.expand(region_low.shape[0], -1)
        x = torch.cat([region_low, ctx], dim=1)
        if self.film_tasks is None:
            return self.net(x).squeeze(-1)
        # Oracle task identity (docs/track_c0.md §1/§2.4): the caller states
        # which task's adapter is active; there is no inference of it here,
        # and forgetting to set it is an error rather than a silent default.
        t = self.active_task
        if t is None:
            raise RuntimeError(
                "FiLM navigator called without active_task; C0 assumes oracle "
                "task identity (docs/track_c0.md §1)")
        h = self.net[1](self.net[0](x))
        h = self.film_gamma1[t] * h + self.film_beta1[t]
        h = self.net[3](self.net[2](h))
        h = self.film_gamma2[t] * h + self.film_beta2[t]
        return self.net[4](h).squeeze(-1)
