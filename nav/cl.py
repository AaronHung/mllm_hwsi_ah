"""持續學習方法（main table 的全部方法）與 CL 專用量測。

方法定義見 docs/protocol.md §3。paper-facing labels：
    replay = counterfactual-teacher replay
    distill = old-policy / policy-fidelity distillation
    ours_uniform = replay + policy distillation (uniform loss weight)
    ours = Utility-Weighted Replay Distillation (a variant)
    joint = joint-training reference

engine.train_navigator 保留給舊腳本；本模組的 train_navigator_cl 是一般化版本：
    - imitation：新任務 teacher 分佈的 KL 模仿（所有方法共同）
    - lwf     ：新任務 state 上 KL(π_old ‖ π_new)（無 buffer；F.kl_div(log π_new, π_old)
                的慣例是 target=π_old 在左，見 §2.1 的 action-KL 方向澄清同一慣例）
    - replay  ：buffer state 上模仿 counterfactual-teacher gain target
    - distill ：buffer state 上 KL(π_old ‖ π_new)，uniform 權重
    - ewc     ：Fisher 加權的參數距離懲罰

v0.33.1 方法軌（docs/method_gate_v033.md）：
    - M1 (replay_sampling="importance")：buffer 抽樣機率改為
      p(s) ∝ u_state(s)^alpha · d_t(s)^beta，buffer 成員本身不變。
    - M2 (use_eq_pres=True)：distill 槽的 old-policy KL 換成 ε-equivalence
      preservation loss L_eq，只作用在有抽到的 replay state 上。

v0.33.2（Sol+Fable joint red-team patch）：u_state(s) 改為逐 source-task
chunk 內算 percentile rank（見 chunkwise_percentile_rank），而非整個 buffer
一起排名——各 task 的 raw counterfactual gain 出自各自獨立的 frozen
evaluator（class 數、base entropy 不同），全域排名會把 evaluator-scale
差異誤讀成 state importance。跨 task 的 replay 壓力完全交給 d_t(s)。

v0.34 method track (docs/method_gate_v034.md), all opt-in and defaulting to
the pre-v0.34 behavior:
    - use_arbiter     : per-update gradient arbitration between the
                        current-task gradient g_n and the memory-side
                        gradient g_m (§2.2). Projects the whole g_m, so the
                        arbiter is identical across projected configs.
    - violation_weight: weights L_eq by the stop-gradient equivalence
                        violation sg[v(s)] = sg[1 - mass] (§2.1), i.e.
                        preserve only where access is actually being lost.
    - conflict_diag   : instrumentation-only mode. Records the same
                        conflict statistics WITHOUT touching the update
                        path, so a frozen config replays bit-identically
                        (§2.5). Never combine with use_arbiter.
The frozen `loss = loss + ...` accumulation order below is deliberately left
untouched by all three (§2.6): a different summation order would break
bit-exactness against the frozen Protocol-v1 / v0.33 rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

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


def normalized_gain(gain: torch.Tensor) -> np.ndarray:
    """Within-state min-max normalization of counterfactual gain.

    Canonical definition shared with scripts/mechanism_probe.py — a flat
    teacher state (no gain spread) makes every candidate epsilon-equivalent,
    so it returns all-ones rather than all-zeros.
    """
    raw = gain.detach().cpu().numpy().astype(float)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo <= 1e-12:
        return np.ones_like(raw)
    return (raw - lo) / (hi - lo)


def epsilon_optimal_mask(gain: torch.Tensor, eps: float = 0.05) -> np.ndarray:
    """Boolean mask over `gain`'s candidates for A_eps(s) (M2 preregistration)."""
    return normalized_gain(gain) >= (1.0 - eps)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1], ties get the average rank.

    Used for M1's cross-state importance u_state(s): raw (un-normalized) gain
    is NOT comparable across states after within-state min-max normalization
    (every non-flat state's top action would be exactly 1) — see the v0.33.1
    patch changelog in docs/method_gate_v033.md.
    """
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values)


def chunkwise_percentile_rank(chunks: list[list[TeacherStep]]) -> np.ndarray:
    """v0.33.2 M1 fix: percentile rank of raw max-gain utility computed
    WITHIN each source-task buffer chunk, then concatenated in the same
    order as ``[st for chunk in chunks for st in chunk]`` — the flattened
    buffer order used everywhere else in `train_navigator_cl` (including
    `eps_masks`). Each task's counterfactual gain is produced by that
    task's own frozen evaluator (different class counts, different base
    entropy), so a single global-buffer percentile would let evaluator-scale
    differences masquerade as cross-task state importance. All cross-task
    replay pressure is left to `d_t(s)`.
    """
    parts = [percentile_rank(np.array([st.utility for st in chunk], dtype=float))
             for chunk in chunks if chunk]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def flatten_task_labels(chunks: list[list[TeacherStep]],
                        labels: list[str]) -> np.ndarray:
    """Concatenate one label per state, repeated per chunk, in the exact
    same flattened order as `chunkwise_percentile_rank` / the training
    buffer. Used only for the per-source-task sampling-mass diagnostic
    (v0.33.2 item 5); never consumed by the loss or the sampler's math.
    """
    parts = [np.full(len(chunk), label, dtype=object)
             for chunk, label in zip(chunks, labels) if chunk]
    return np.concatenate(parts) if parts else np.array([], dtype=object)


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


@torch.no_grad()
def attach_boundary_policy(buffer_chunk: list[TeacherStep],
                          navigator: Navigator, device) -> None:
    """Store the stage-boundary policy snapshot on each newly buffered step.

    Called once, right after `build_buffer`, with the navigator as it stood
    at the end of the stage in which that step's task was learned (i.e.
    before it becomes `old_nav` for the next stage). Only needed by M1's
    drift term d_t(s); harmless no-op cost for methods that never read it.
    """
    navigator.eval()
    for st in buffer_chunk:
        logits = navigator(st.low[st.candidates].to(device),
                           st.context.to(device))
        st.boundary_policy = F.softmax(logits, dim=0).detach().cpu()


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


# ------------------------------------------------------------------ v0.33.1 M1
class ImportanceReplaySampler:
    """M1: cross-state importance replay sampling (docs/method_gate_v033.md).

    Buffer MEMBERSHIP is untouched — this only changes which buffer index is
    drawn each update. `u_state(s)` is the percentile rank of `s`'s raw
    (un-normalized) max-gain utility computed ACROSS the whole buffer, fixed
    once per call (buffer composition does not change within a stage).
    `d_t(s) = KL(pi_boundary(s) ‖ pi_theta(s))` is refreshed every
    `refresh_every` optimizer updates via a no-grad forward pass. Exponents
    `alpha`/`beta` let a factor be switched off (`x ** 0 == 1`) for the
    samp_util_only / samp_drift_only attribution mini-arms without a
    separate code path.
    """

    def __init__(self, buffer: list[TeacherStep], device, alpha: float = 1.0,
                beta: float = 1.0, floor: float = 0.1,
                refresh_every: int = 50,
                diag_log: list[dict] | None = None,
                u_state: np.ndarray | None = None,
                task_labels: np.ndarray | None = None) -> None:
        self.buffer = buffer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.floor = floor / max(len(buffer), 1)
        self.refresh_every = max(int(refresh_every), 1)
        self.diag_log = diag_log
        if u_state is not None:
            u_state = np.asarray(u_state, dtype=float)
            if len(u_state) != len(buffer):
                raise ValueError(
                    f"u_state length {len(u_state)} != buffer length "
                    f"{len(buffer)} — chunk order must match the flattened "
                    "buffer order (see chunkwise_percentile_rank)")
            self.u_state = u_state
        else:
            # Fallback for standalone callers that never pass chunk-aware
            # ranks (e.g. ad-hoc scripts): global-buffer percentile. Every
            # v0.33.2 gate config always passes `u_state` explicitly.
            raw_u = np.array([st.utility for st in buffer], dtype=float)
            self.u_state = percentile_rank(raw_u)
        if task_labels is not None and len(task_labels) != len(buffer):
            raise ValueError(
                f"task_labels length {len(task_labels)} != buffer length "
                f"{len(buffer)}")
        self.task_labels = (np.asarray(task_labels)
                           if task_labels is not None else None)
        self.dt_state = np.zeros(len(buffer))
        self.p = np.full(len(buffer), 1.0 / max(len(buffer), 1))

    @torch.no_grad()
    def refresh(self, nav: Navigator, step: int) -> None:
        was_training = nav.training
        nav.eval()
        dt = np.zeros(len(self.buffer))
        for idx, st in enumerate(self.buffer):
            if st.boundary_policy is None:
                continue
            logits = nav(st.low[st.candidates].to(self.device),
                        st.context.to(self.device))
            now_p = F.softmax(logits, dim=0)
            ref_p = st.boundary_policy.to(self.device)
            dt[idx] = float(F.kl_div(now_p.clamp_min(1e-12).log(), ref_p,
                                     reduction="sum").item())
        if was_training:
            nav.train()
        self.dt_state = dt
        importance = (np.clip(self.u_state, 1e-8, None) ** self.alpha
                     * np.clip(dt, 1e-8, None) ** self.beta)
        p = importance / importance.sum()
        p = np.maximum(p, self.floor)
        self.p = p / p.sum()
        if self.diag_log is not None:
            entropy = float(-(self.p * np.log(self.p + 1e-12)).sum())
            if self.u_state.std() > 1e-12 and dt.std() > 1e-12:
                corr = float(np.corrcoef(self.u_state, dt)[0, 1])
            else:
                corr = None  # undefined when one side hasn't drifted yet
            entry = dict(step=int(step), entropy=entropy, corr_u_dt=corr)
            if self.task_labels is not None:
                # v0.33.2 item 5: cheap per-refresh diagnostic — reveals if
                # a stage's replay mass suddenly concentrates on one old
                # task (e.g. the freshest chunk, whose d_t starts near 0).
                entry["sampling_mass_by_task"] = {
                    str(label): float(self.p[self.task_labels == label].sum())
                    for label in np.unique(self.task_labels)
                }
            self.diag_log.append(entry)

    def maybe_refresh(self, nav: Navigator, step: int) -> None:
        if step % self.refresh_every == 0:
            self.refresh(nav, step)

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.buffer), p=self.p))


# ----------------------------------------------- v0.34 gradient arbiter
def flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter],
              retain_graph: bool) -> torch.Tensor:
    """Flatten d(loss)/d(params) into one vector, in `params` order.

    Uses torch.autograd.grad rather than .backward() so nothing is written
    into p.grad — that matters for conflict_diag mode (docs/
    method_gate_v034.md §2.5), where the real update must stay byte-identical
    to the frozen path. Unused parameters materialize as zeros so the vector
    layout is stable across updates.
    """
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def conflict_scalars(g_n: torch.Tensor, g_m: torch.Tensor
                     ) -> tuple[float, float, float]:
    """(g_n.g_m, |g_n|^2, |g_m|^2) in a SINGLE device sync.

    Every scalar the arbiter and the §2.4 log need comes from here. Reading
    them one at a time would cost four MPS->CPU syncs per update, which at
    ~10^4 updates per stage is pure wall-clock for no information.
    """
    packed = torch.stack([torch.dot(g_n, g_m), torch.dot(g_n, g_n),
                          torch.dot(g_m, g_m)])
    dot, nn2, mm2 = packed.tolist()
    return dot, nn2, mm2


def conflict_stats(dot: float, nn2: float, mm2: float,
                   step: int) -> dict[str, float]:
    """The four quantities §2.4 requires, for one update.

    `proj_ratio` is the magnitude of the component that the arbiter actually
    removes, relative to |g_m| — hence 0.0 whenever the update is not in
    conflict, and |cos| when it is. In conflict_diag mode nothing is removed,
    so the same number is the counterfactual "would-be" projection ratio.
    """
    cos = dot / ((nn2 ** 0.5) * (mm2 ** 0.5) + 1e-12)
    conflict = dot < 0.0
    return {"step": step,
            "cos": round(cos, 6),
            "conflict": int(conflict),
            "proj_ratio": round(abs(cos) if conflict else 0.0, 6),
            "gn_norm": round(nn2 ** 0.5, 6),
            "gm_norm": round(mm2 ** 0.5, 6)}


def arbiter_grad(g_n: torch.Tensor, g_m: torch.Tensor, dot: float,
                 nn2: float, lambda_arb: float) -> torch.Tensor:
    """§2.2: drop the component of g_m that is first-order anti-aligned with
    g_n, then combine. Guarantees g_n . g_m_proj == 0 in RAW gradient space
    when the two conflict — it does NOT guarantee anything about the realized
    Adam update (docs/method_gate_v034.md §2.2, rider r4).

    `dot` and `nn2` come from conflict_scalars(), so the coefficient is a
    plain float and the projection costs no additional sync."""
    if dot < 0.0:
        g_m = g_m - (dot / (nn2 + 1e-12)) * g_n
    return g_n + lambda_arb * g_m


def assign_flat_grad(params: list[torch.nn.Parameter],
                     flat: torch.Tensor) -> None:
    """Write a flat gradient vector back into p.grad, in `params` order."""
    off = 0
    for p in params:
        n = p.numel()
        p.grad = flat[off:off + n].view_as(p).clone()
        off += n


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
                       lam_ewc: float = 100.0,
                       replay_sampling: str = "uniform",
                       sampling_alpha: float = 1.0,
                       sampling_beta: float = 1.0,
                       sampling_floor: float = 0.1,
                       drift_refresh_every: int = 50,
                       use_eq_pres: bool = False,
                       eps: float = 0.05,
                       use_arbiter: bool = False,
                       violation_weight: bool = False,
                       conflict_diag: bool = False,
                       lambda_arb: float = 1.0,
                       arbiter_log: list[dict] | None = None,
                       diag_log: list[dict] | None = None,
                       u_state_override: np.ndarray | None = None,
                       task_labels: np.ndarray | None = None) -> Navigator:
    """v0.33.1 adds `replay_sampling="importance"` (M1) and `use_eq_pres` (M2)
    behind opt-in kwargs; every Protocol-v1 method keeps its old code path
    (`replay_sampling="uniform"`, `use_eq_pres=False` are the defaults).

    v0.34 (docs/method_gate_v034.md) adds `use_arbiter` / `violation_weight`
    / `conflict_diag`, all default-off for the same reason. `conflict_diag`
    is instrumentation only: it computes the diagnostic gradients on the SAME
    forward graph and leaves opt.zero_grad()/backward()/step() untouched, so a
    frozen config replays bit-identically (§2.5 validity condition).

    v0.33.2: `u_state_override` should be the caller-precomputed, chunk-wise
    percentile rank (`chunkwise_percentile_rank`) aligned to `buffer`'s
    flattened order; `task_labels` (optional) enables the per-source-task
    sampling-mass diagnostic. If `u_state_override` is omitted while
    `replay_sampling="importance"`, the sampler falls back to a (less
    correct, pre-v0.33.2) global-buffer percentile — callers driving a gate
    config should always pass it.
    """
    if replay_sampling not in ("uniform", "importance"):
        raise ValueError(f"unknown replay_sampling: {replay_sampling!r}")
    if use_arbiter and conflict_diag:
        # conflict_diag exists to leave the update path alone; combining the
        # two would silently make a "bit-identical replication" run apply a
        # projected gradient (docs/method_gate_v034.md §2.5 condition 2).
        raise ValueError("use_arbiter and conflict_diag are mutually exclusive")
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

    eps_masks = None
    if buffer and use_eq_pres:
        eps_masks = [torch.tensor(epsilon_optimal_mask(st.gain, eps),
                                  device=device)
                    for st in buffer]

    sampler = None
    if buffer and replay_sampling == "importance":
        sampler = ImportanceReplaySampler(
            buffer, device, alpha=sampling_alpha, beta=sampling_beta,
            floor=sampling_floor, refresh_every=drift_refresh_every,
            diag_log=diag_log, u_state=u_state_override,
            task_labels=task_labels)
        sampler.refresh(nav, 0)

    # v0.34 §2.2: the arbiter needs the current-task and memory-side terms
    # separately, but the `loss = loss + ...` chain below must keep its exact
    # summation order (§2.6), so the split is done by holding extra
    # references, never by regrouping the sum.
    split_terms = use_arbiter or conflict_diag
    params = [p for p in nav.parameters() if p.requires_grad]

    global_step = 0
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

            loss_new = loss if split_terms else None
            mem_terms: list[torch.Tensor] = []

            if buffer and (use_replay or use_distill):
                j = sampler.sample(rng) if sampler is not None \
                    else int(rng.integers(len(buffer)))
                rst = buffer[j]
                rlow = rst.low[rst.candidates].to(device)
                rctx = rst.context.to(device)
                r_scores = nav(rlow, rctx)
                r_logp = F.log_softmax(r_scores, dim=0)
                if use_replay:
                    r_target = F.softmax(rst.gain.to(device) / tau, dim=0)
                    r_term = F.kl_div(r_logp, r_target, reduction="sum")
                    loss = loss + r_term
                    if split_terms:
                        mem_terms.append(r_term)
                if use_distill and old_nav is not None:
                    if use_eq_pres:
                        # M2: replace the old-policy KL term with the
                        # epsilon-equivalence preservation loss (i2 guard).
                        mask = eps_masks[j]
                        pi_theta = F.softmax(r_scores, dim=0)
                        mass = pi_theta[mask].sum().clamp(1e-8, 1.0)
                        if violation_weight:
                            # v0.34 §2.1: preserve only where the policy has
                            # actually started to lose access to A_eps(s).
                            # sg[] keeps v(s) a constant weight, so no
                            # second-order term enters the gradient.
                            v_s = (1.0 - mass).detach()
                            m_term = lam * u_w[j].to(device) * v_s \
                                * (-mass.log())
                        else:
                            m_term = lam * u_w[j].to(device) * (-mass.log())
                        loss = loss + m_term
                    else:
                        with torch.no_grad():
                            old_p = F.softmax(old_nav(rlow, rctx), dim=0)
                        m_term = lam * u_w[j].to(device) * F.kl_div(
                            r_logp, old_p, reduction="sum")
                        loss = loss + m_term
                    if split_terms:
                        mem_terms.append(m_term)

            if ewc_terms:
                pen = torch.zeros((), device=device)
                for term in ewc_terms:
                    for n, p in nav.named_parameters():
                        pen = pen + (term.fisher[n].to(device)
                                     * (p - term.anchor[n].to(device)) ** 2).sum()
                loss = loss + lam_ewc * pen

            # v0.34 §2.2/§2.5. `mem_terms` is empty on the first task (no
            # buffer, no old_nav), where both new modes degrade to the frozen
            # single-backward update by construction.
            if split_terms and mem_terms:
                loss_mem = mem_terms[0]
                for extra in mem_terms[1:]:
                    loss_mem = loss_mem + extra
                g_n = flat_grad(loss_new, params, retain_graph=True)
                g_m = flat_grad(loss_mem, params, retain_graph=conflict_diag)
                dot, nn2, mm2 = conflict_scalars(g_n, g_m)
                if arbiter_log is not None:
                    arbiter_log.append(
                        conflict_stats(dot, nn2, mm2, global_step))
                if use_arbiter:
                    opt.zero_grad()
                    assign_flat_grad(
                        params, arbiter_grad(g_n, g_m, dot, nn2, lambda_arb))
                    opt.step()
                else:
                    # conflict_diag: the frozen update path, untouched.
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()

            global_step += 1
            if sampler is not None:
                sampler.maybe_refresh(nav, global_step)
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

# v0.33.1 method-gate configs (docs/method_gate_v033.md).  All build on the
# ours_uniform base (use_replay + use_distill, utility_weight=False) so the
# gate isolates exactly M1 (sampling) and/or M2 (loss), never buffer
# membership or the pre-existing utility_weight ablation axis.
METHOD_GATE_KWARGS = {
    "ia_samp": dict(use_replay=True, use_distill=True, utility_weight=False,
                    replay_sampling="importance"),
    "eq_pres": dict(use_replay=True, use_distill=True, utility_weight=False,
                    use_eq_pres=True),
    "ia_ep": dict(use_replay=True, use_distill=True, utility_weight=False,
                 replay_sampling="importance", use_eq_pres=True),
    # Attribution mini-arms (K=1, 3 seeds only): zero one factor via its
    # exponent rather than a separate sampler code path.
    "samp_util_only": dict(use_replay=True, use_distill=True,
                           utility_weight=False, replay_sampling="importance",
                           sampling_beta=0.0),
    "samp_drift_only": dict(use_replay=True, use_distill=True,
                            utility_weight=False, replay_sampling="importance",
                            sampling_alpha=0.0),
    # v0.34 method track (docs/method_gate_v034.md §2.3).  Same ours_uniform
    # base as above, so the only isolated variables are the memory objective
    # and the arbiter.  NOTE: `proj_distill` keeps the name used in the
    # sign-off text, but its L_mem is L_replay + old-policy KL — the
    # ours_uniform composition plus the arbiter, NOT the frozen `distill`
    # method (which has no replay term).  Fixed set of three, no fourth.
    "proj_distill": dict(use_replay=True, use_distill=True,
                         utility_weight=False, use_arbiter=True),
    "proj_eq_pres": dict(use_replay=True, use_distill=True,
                         utility_weight=False, use_eq_pres=True,
                         use_arbiter=True),
    "conflict_eq_pres": dict(use_replay=True, use_distill=True,
                             utility_weight=False, use_eq_pres=True,
                             use_arbiter=True, violation_weight=True),
    # Instrumentation-only replication of frozen `eq_pres` (§2.5): identical
    # update path, conflict statistics recorded on the same forward graph.
    # Its metric rows are never admitted anywhere — the separate key exists
    # so they cannot be pooled with real `eq_pres` rows by accident.
    "eq_pres_diag": dict(use_replay=True, use_distill=True,
                         utility_weight=False, use_eq_pres=True,
                         conflict_diag=True),
}

# Configs whose per-update gradient-conflict instrumentation must exist
# (docs/method_gate_v034.md §2.4); read by scripts/cl_main.py and the runner.
ARBITER_CONFIGS = ("proj_distill", "proj_eq_pres", "conflict_eq_pres",
                   "eq_pres_diag")
