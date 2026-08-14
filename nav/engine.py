"""Causal pyramid environment + counterfactual-gain 教師 + 訓練/評估迴圈。

環境定義（離散 causal pyramid，特徵空間版）：
- 低倍層（永遠可見）：每張切片 R 個 region 的 HIPT 特徵 [R,192]。
- 高倍層（**選了才可見**）：region r 的 48 個 CONCH patch 特徵的 mean（512,）。
  未 zoom 的 region 的高倍證據對 agent 不存在——這就是 access rule。
- 行動：從未 zoom 的 region 中選一個 zoom-in；固定 budget K 步，不學 stop。
- 證據聚合：已 zoom regions 的高倍向量取 mean → evaluator 診斷。

弱監督（counterfactual diagnostic gain）：
  gain(r | E) = CE(evaluator(E), y) - CE(evaluator(E ∪ {r}), y)
教師 = 凍結 evaluator 下的 greedy 展開；navigator 以 KL 對齊 softmax(gain/τ)。
不需要任何 region-level 人工標註。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .models import Evaluator, Navigator, PATCH_DIM


# ---------------------------------------------------------------- data bank
@dataclass
class Slide:
    sid: str
    y: int
    low: torch.Tensor    # [R,192] 低倍（全可見）
    high: torch.Tensor   # [R,512] 高倍（zoom 才可見；已由 48 patch mean 壓成一向量）


def build_bank(ds, sids: list[str], label_of: dict[str, int]) -> list[Slide]:
    bank = []
    for sid in sids:
        low = ds.region(sid).float()
        high = ds.patch(sid).float().mean(dim=1)  # [R,48,512] -> [R,512]
        bank.append(Slide(sid=sid, y=label_of[sid], low=low, high=high))
    return bank


# ---------------------------------------------------------------- env helpers
def evidence_of(slide: Slide, zoomed: list[int], device) -> torch.Tensor:
    if not zoomed:
        return torch.zeros(slide.high.shape[1], device=device,
                           dtype=slide.high.dtype)
    return slide.high[zoomed].to(device).mean(0)


def context_of(slide: Slide, zoomed: list[int], k_total: int, device) -> torch.Tensor:
    remain = torch.tensor(
        [(k_total - len(zoomed)) / k_total],
        device=device,
        dtype=slide.high.dtype,
    )
    return torch.cat([evidence_of(slide, zoomed, device), remain])


# ---------------------------------------------------------------- teacher
@dataclass
class TeacherStep:
    sid: str
    low: torch.Tensor        # [R,192]
    candidates: torch.Tensor  # [C] 未 zoom 的 region 索引
    context: torch.Tensor     # [513]
    gain: torch.Tensor        # [C] counterfactual gain
    utility: float = field(default=0.0)


@torch.no_grad()
def teacher_rollout(slide: Slide, evaluator: Evaluator, k: int, device,
                    tau: float = 0.05) -> list[TeacherStep]:
    """凍結 evaluator 的 greedy counterfactual 展開，回傳每步的 (state, gain)。"""
    evaluator.eval()
    R = slide.low.shape[0]
    y = torch.tensor([slide.y], device=device)
    zoomed: list[int] = []
    steps = []
    high = slide.high.to(device)
    for _ in range(min(k, R)):
        cand = torch.tensor([i for i in range(R) if i not in zoomed], device=device)
        ev_now = evidence_of(slide, zoomed, device)
        loss_now = F.cross_entropy(evaluator(ev_now[None]), y)
        # 向量化：每個候選加入後的證據 = (sum + high_r) / (n+1)
        n = len(zoomed)
        ev_next = (ev_now * n + high[cand]) / (n + 1)          # [C,512]
        loss_next = F.cross_entropy(evaluator(ev_next), y.expand(len(cand)),
                                    reduction="none")           # [C]
        gain = loss_now - loss_next
        steps.append(TeacherStep(
            sid=slide.sid, low=slide.low, candidates=cand.cpu(),
            context=context_of(slide, zoomed, k, device).cpu(),
            gain=gain.cpu(), utility=float(gain.max().clamp(min=0))))
        zoomed.append(int(cand[gain.argmax()].item()))
    return steps


# ---------------------------------------------------------------- training
def train_evaluator(bank: list[Slide], n_classes: int, k: int, device,
                    epochs: int = 60, samples_per_slide: int = 8,
                    lr: float = 1e-3, seed: int = 0) -> Evaluator:
    """random K-子集當 augmentation，訓練診斷頭。"""
    g = torch.Generator().manual_seed(seed)
    model = Evaluator(n_classes, in_dim=bank[0].high.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        evs, ys = [], []
        for s in bank:
            R = s.low.shape[0]
            for _ in range(samples_per_slide):
                idx = torch.randperm(R, generator=g)[:min(k, R)]
                evs.append(s.high[idx].mean(0))
                ys.append(s.y)
        perm = torch.randperm(len(ys), generator=g)
        evs_t = torch.stack(evs)[perm].to(device)
        ys_t = torch.tensor(ys)[perm].to(device)
        for i in range(0, len(ys_t), 64):
            loss = F.cross_entropy(model(evs_t[i:i + 64]), ys_t[i:i + 64])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_navigator(steps: list[TeacherStep], device, navigator: Navigator | None = None,
                    epochs: int = 30, lr: float = 1e-3, tau: float = 0.05,
                    seed: int = 0,
                    distill: tuple[Navigator, list[TeacherStep], float] | None = None
                    ) -> Navigator:
    """以 KL(policy || softmax(gain/τ)) 模仿教師。

    distill=(old_navigator, replay_steps, λ) 時，加上 utility-weighted policy
    distillation：在舊任務的 replay states 上，KL(π_new || π_old)，
    以該 state 的 diagnostic utility 加權（Gate 3 機制）。
    """
    rng = np.random.default_rng(seed)
    if navigator is None:
        low_dim = steps[0].low.shape[1]
        high_dim = steps[0].context.shape[0] - 1
        navigator = Navigator(low_dim=low_dim, high_dim=high_dim).to(device)
    nav = navigator
    opt = torch.optim.Adam(nav.parameters(), lr=lr, weight_decay=1e-4)
    nav.train()

    old_nav, replay, lam = (None, [], 0.0) if distill is None else distill
    if old_nav is not None:
        old_nav.eval()
        u = torch.tensor([st.utility for st in replay])
        u_w = (u / (u.mean() + 1e-8)).clamp(max=5.0)  # utility 正規化權重

    for _ in range(epochs):
        order = rng.permutation(len(steps))
        for i in order:
            st = steps[i]
            low_c = st.low[st.candidates].to(device)
            scores = nav(low_c, st.context.to(device))
            target = F.softmax(st.gain.to(device) / tau, dim=0)
            loss = F.kl_div(F.log_softmax(scores, dim=0), target, reduction="sum")

            if old_nav is not None and len(replay) > 0:
                j = int(rng.integers(len(replay)))
                rst = replay[j]
                rlow = rst.low[rst.candidates].to(device)
                rctx = rst.context.to(device)
                with torch.no_grad():
                    old_p = F.softmax(old_nav(rlow, rctx), dim=0)
                new_lp = F.log_softmax(nav(rlow, rctx), dim=0)
                loss = loss + lam * u_w[j].to(device) * F.kl_div(
                    new_lp, old_p, reduction="sum")

            opt.zero_grad()
            loss.backward()
            opt.step()
    return nav


# ---------------------------------------------------------------- policies
def spatial_uniform(coords: np.ndarray, k: int) -> list[int]:
    """bounding box 均勻分格，每格取離格心最近的 region；空格用空間 FPS 補滿。"""
    n = coords.shape[0]
    if k >= n:
        return list(range(n))
    g = int(np.ceil(np.sqrt(k)))
    # MPS does not support float64 tensors; keep geometry preprocessing
    # float32 as well so backend-specific dtype promotion cannot leak in.
    xy = coords.astype(np.float32, copy=False)
    mn, mx = xy.min(0), xy.max(0)
    span = np.maximum(mx - mn, 1e-9)
    cell = np.minimum(((xy - mn) / span * g).astype(int), g - 1)
    chosen: list[int] = []
    for cx in range(g):
        for cy in range(g):
            if len(chosen) >= k:
                break
            idx = np.where((cell[:, 0] == cx) & (cell[:, 1] == cy))[0]
            idx = idx[~np.isin(idx, chosen)]
            if len(idx) == 0:
                continue
            center = mn + (np.array([cx, cy]) + 0.5) / g * span
            chosen.append(int(idx[np.argmin(np.linalg.norm(xy[idx] - center, axis=1))]))
    while len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), chosen)
        dmin = np.linalg.norm(xy[rest, None] - xy[chosen][None], axis=2).min(1)
        chosen.append(int(rest[np.argmax(dmin)]))
    return chosen[:k]


@torch.no_grad()
def rollout_policy(slide: Slide, k: int, device, policy: str,
                   navigator: Navigator | None = None,
                   rng: np.random.Generator | None = None,
                   coords: np.ndarray | None = None) -> list[int]:
    """回傳被 zoom 的 region 索引（依 policy）。"""
    R = slide.low.shape[0]
    k = min(k, R)
    if policy == "random":
        return list(rng.choice(R, size=k, replace=False))
    if policy == "uniform":
        return spatial_uniform(coords, k)
    if policy == "learned":
        navigator.eval()
        zoomed: list[int] = []
        for _ in range(k):
            cand = [i for i in range(R) if i not in zoomed]
            scores = navigator(slide.low[cand].to(device),
                               context_of(slide, zoomed, k, device))
            zoomed.append(cand[int(scores.argmax())])
        return zoomed
    raise ValueError(policy)


@torch.no_grad()
def eval_policy(bank: list[Slide], evaluator: Evaluator, k: int, device,
                policy: str, navigator: Navigator | None = None,
                seed: int = 0,
                coords_of: dict[str, np.ndarray] | None = None
                ) -> tuple[float, dict[str, list[int]]]:
    """回傳 (accuracy, 每張切片被選 region)。"""
    evaluator.eval()
    rng = np.random.default_rng(seed)
    correct, selections = 0, {}
    for s in bank:
        sel = rollout_policy(s, k, device, policy, navigator=navigator, rng=rng,
                             coords=None if coords_of is None else coords_of[s.sid])
        selections[s.sid] = sel
        pred = evaluator(evidence_of(s, sel, device)[None]).argmax().item()
        correct += int(pred == s.y)
    return correct / len(bank), selections


@torch.no_grad()
def action_distributions(bank: list[Slide], navigator: Navigator, k: int, device,
                         tau_states: list[TeacherStep] | None = None) -> list[torch.Tensor]:
    """在固定的 probe states（教師展開的 state）上取 policy 分佈，供 drift 量測。"""
    navigator.eval()
    out = []
    for st in tau_states:
        low_c = st.low[st.candidates].to(device)
        out.append(F.softmax(navigator(low_c, st.context.to(device)), dim=0).cpu())
    return out
