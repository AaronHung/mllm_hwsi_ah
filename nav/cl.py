"""持續學習方法（main table 的全部方法）與 CL 專用量測。

方法定義見 docs/protocol.md §3。paper-facing labels：
    replay = counterfactual-teacher replay
    distill = old-policy / policy-fidelity distillation
    ours_uniform = replay + policy distillation (uniform loss weight)
    ours = Utility-Weighted Replay Distillation (a variant)
    joint = joint-training reference

engine.train_navigator 保留給舊腳本；本模組的 train_navigator_cl 是一般化版本：
    - imitation：新任務 teacher 分佈的 KL 模仿（所有方法共同）
    - lwf     ：新任務 state 上 KL(π_new ‖ π_old)（無 buffer）
    - replay  ：buffer state 上模仿 counterfactual-teacher gain target
    - distill ：buffer state 上 KL(π_new ‖ π_old)，uniform 權重
    - ewc     ：Fisher 加權的參數距離懲罰
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .engine import TeacherStep, context_of, evidence_of, rollout_policy
from .models import Evaluator, Navigator


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def eval_policy_balanced(bank, evaluator: Evaluator, k: int, device, policy: str,
                         navigator: Navigator | None = None, seed: int = 0,
                         n_classes: int | None = None
                         ) -> tuple[float, float, dict[str, list[int]]]:
    """回傳 (balanced_acc, acc, selections)。policy 同 engine.rollout_policy。"""
    evaluator.eval()
    rng = np.random.default_rng(seed)
    ys, preds, selections = [], [], {}
    for s in bank:
        sel = rollout_policy(s, k, device, policy, navigator=navigator, rng=rng)
        selections[s.sid] = sel
        pred = evaluator(evidence_of(s, sel, device)[None]).argmax().item()
        ys.append(s.y)
        preds.append(pred)
    ys, preds = np.array(ys), np.array(preds)
    acc = float((ys == preds).mean())
    ncls = n_classes or (ys.max() + 1)
    recalls = [float((preds[ys == c] == c).mean()) for c in range(ncls)
               if (ys == c).any()]
    return float(np.mean(recalls)), acc, selections


@torch.no_grad()
def trajectory_utility(bank, navigator: Navigator, evaluator: Evaluator,
                       k: int, device) -> float:
    """序列結束時：navigator 實際軌跡上每步所選 region 的 counterfactual gain 平均。"""
    evaluator.eval()
    navigator.eval()
    gains = []
    for s in bank:
        y = torch.tensor([s.y], device=device)
        zoomed: list[int] = []
        high = s.high.to(device)
        for _ in range(min(k, s.low.shape[0])):
            cand = [i for i in range(s.low.shape[0]) if i not in zoomed]
            ctx = context_of(s, zoomed, k, device)
            scores = navigator(s.low[cand].to(device), ctx)
            choice = cand[int(scores.argmax())]
            ev_now = evidence_of(s, zoomed, device)
            loss_now = F.cross_entropy(evaluator(ev_now[None]), y)
            n = len(zoomed)
            ev_next = (ev_now * n + high[choice]) / (n + 1)
            loss_next = F.cross_entropy(evaluator(ev_next[None]), y)
            gains.append(float(loss_now - loss_next))
            zoomed.append(choice)
    return float(np.mean(gains))


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


@torch.no_grad()
def policy_on_probes(navigator: Navigator, probes: list[TeacherStep],
                     device) -> list[torch.Tensor]:
    navigator.eval()
    return [F.softmax(navigator(st.low[st.candidates].to(device),
                                st.context.to(device)), dim=0).cpu()
            for st in probes]


def kl_drift(pi_ref: list[torch.Tensor], pi_now: list[torch.Tensor]) -> float:
    """KL(π_ref ‖ π_now)，與 pilot 報告同一慣例。"""
    return float(np.mean([F.kl_div(pn.log(), pr, reduction="sum").item()
                          for pr, pn in zip(pi_ref, pi_now)]))


# ------------------------------------------------------------------ buffer / fisher
def build_buffer(steps: list[TeacherStep], per_slide: int = 2,
                 cap: int = 512) -> list[TeacherStep]:
    """每張 slide 取前 per_slide 步；超過 cap 時保留 utility 最高者。

    This utility-prioritized truncation is shared by `ours` and
    `ours_uniform`; their ablation isolates loss weighting, not memory
    composition.
    """
    by_sid: dict[str, list[TeacherStep]] = {}
    for st in steps:
        by_sid.setdefault(st.sid, []).append(st)
    picked = [st for sid in by_sid for st in by_sid[sid][:per_slide]]
    if len(picked) > cap:
        picked = sorted(picked, key=lambda st: -st.utility)[:cap]
    return picked


@dataclass
class EwcTerm:
    fisher: dict[str, torch.Tensor]
    anchor: dict[str, torch.Tensor]


def fisher_of(nav: Navigator, steps: list[TeacherStep], device,
              tau: float = 0.05, n_max: int = 512, seed: int = 0) -> EwcTerm:
    """以模仿 loss 的梯度平方估計 Fisher（對角）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(steps))[:min(n_max, len(steps))]
    fisher = {n: torch.zeros_like(p) for n, p in nav.named_parameters()}
    nav.eval()
    for i in idx:
        st = steps[int(i)]
        nav.zero_grad()
        scores = nav(st.low[st.candidates].to(device), st.context.to(device))
        target = F.softmax(st.gain.to(device) / tau, dim=0)
        loss = F.kl_div(F.log_softmax(scores, dim=0), target, reduction="sum")
        loss.backward()
        for n, p in nav.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= max(len(idx), 1)
    anchor = {n: p.detach().clone() for n, p in nav.named_parameters()}
    nav.zero_grad()
    return EwcTerm(fisher=fisher, anchor=anchor)


# ------------------------------------------------------------------ trainer
def train_navigator_cl(steps: list[TeacherStep], device,
                       navigator: Navigator | None = None,
                       epochs: int = 10, lr: float = 1e-3, tau: float = 0.05,
                       seed: int = 0,
                       old_nav: Navigator | None = None,
                       buffer: list[TeacherStep] | None = None,
                       use_replay: bool = False,
                       use_distill: bool = False,
                       utility_weight: bool = False,
                       use_lwf: bool = False,
                       lam: float = 1.0,
                       ewc_terms: list[EwcTerm] | None = None,
                       lam_ewc: float = 100.0) -> Navigator:
    rng = np.random.default_rng(seed)
    if navigator is None:
        navigator = Navigator(low_dim=steps[0].low.shape[1],
                              high_dim=steps[0].context.shape[0] - 1).to(device)
    nav = navigator
    opt = torch.optim.Adam(nav.parameters(), lr=lr, weight_decay=1e-4)
    nav.train()
    if old_nav is not None:
        old_nav.eval()

    buffer = buffer or []
    if buffer and utility_weight:
        u = torch.tensor([st.utility for st in buffer])
        u_w = (u / (u.mean() + 1e-8)).clamp(max=5.0)
    else:
        u_w = torch.ones(max(len(buffer), 1))

    for _ in range(epochs):
        order = rng.permutation(len(steps))
        for i in order:
            st = steps[i]
            low_c = st.low[st.candidates].to(device)
            ctx = st.context.to(device)
            scores = nav(low_c, ctx)
            target = F.softmax(st.gain.to(device) / tau, dim=0)
            loss = F.kl_div(F.log_softmax(scores, dim=0), target, reduction="sum")

            if use_lwf and old_nav is not None:
                with torch.no_grad():
                    old_p = F.softmax(old_nav(low_c, ctx), dim=0)
                loss = loss + lam * F.kl_div(F.log_softmax(scores, dim=0),
                                             old_p, reduction="sum")

            if buffer and (use_replay or use_distill):
                j = int(rng.integers(len(buffer)))
                rst = buffer[j]
                rlow = rst.low[rst.candidates].to(device)
                rctx = rst.context.to(device)
                r_scores = nav(rlow, rctx)
                r_logp = F.log_softmax(r_scores, dim=0)
                if use_replay:
                    r_target = F.softmax(rst.gain.to(device) / tau, dim=0)
                    loss = loss + F.kl_div(r_logp, r_target, reduction="sum")
                if use_distill and old_nav is not None:
                    with torch.no_grad():
                        old_p = F.softmax(old_nav(rlow, rctx), dim=0)
                    loss = loss + lam * u_w[j].to(device) * F.kl_div(
                        r_logp, old_p, reduction="sum")

            if ewc_terms:
                pen = torch.zeros((), device=device)
                for term in ewc_terms:
                    for n, p in nav.named_parameters():
                        pen = pen + (term.fisher[n].to(device)
                                     * (p - term.anchor[n].to(device)) ** 2).sum()
                loss = loss + lam_ewc * pen

            opt.zero_grad()
            loss.backward()
            opt.step()
    return nav


METHOD_KWARGS = {
    "seqft": dict(),
    "ewc": dict(),                                  # ewc_terms 由 caller 傳入
    "lwf": dict(use_lwf=True),
    "replay": dict(use_replay=True),
    "distill": dict(use_distill=True, utility_weight=False),
    "ours_uniform": dict(use_replay=True, use_distill=True, utility_weight=False),
    "ours": dict(use_replay=True, use_distill=True, utility_weight=True),
}
