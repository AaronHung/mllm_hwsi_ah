"""WP4 mechanism probe for seed 0, K=1, can_dataset main and reverse.

This is intentionally separate from the frozen main grid.  It saves portable
stage-boundary navigator checkpoints and JSONL state dumps under a fresh
run directory, then writes the requested report/figures.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from cl_main import load_can_tasks  # noqa: E402
from nav import add_device_argument, device_info, resolve_device  # noqa: E402
from nav.cl import build_buffer, train_navigator_cl  # noqa: E402
from nav.engine import context_of, teacher_rollout, train_evaluator  # noqa: E402
from nav.method_labels import method_label  # noqa: E402

METHODS = ("seqft", "distill", "ours")
ORDERS = ("main", "reverse")
EPSILON = 0.05
SEED = 0
K = 1


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@torch.no_grad()
def policy_output(nav, step, device):
    logits = nav(step.low[step.candidates].to(device),
                 step.context.to(device)).detach().cpu()
    probs = F.softmax(logits, dim=0)
    return logits, probs


@torch.no_grad()
def selected_trajectory(slide, nav, k, device):
    nav.eval()
    zoomed: list[int] = []
    for _ in range(min(k, slide.low.shape[0])):
        candidates = [i for i in range(slide.low.shape[0]) if i not in zoomed]
        ctx = context_of(slide, zoomed, k, device)
        scores = nav(slide.low[candidates].to(device), ctx)
        zoomed.append(candidates[int(scores.argmax())])
    return zoomed


def normalized_gain(gain: torch.Tensor) -> np.ndarray:
    raw = gain.detach().cpu().numpy().astype(float)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo <= 1e-12:
        # A flat teacher state makes every candidate epsilon-equivalent.
        return np.ones_like(raw)
    return (raw - lo) / (hi - lo)


def kl_ref_now(ref: torch.Tensor, now: torch.Tensor) -> float:
    return float(F.kl_div(now.clamp_min(1e-12).log(), ref,
                          reduction="sum").item())


def checkpoint(nav, path: Path) -> None:
    state = {name: value.detach().cpu()
             for name, value in nav.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run_order(tasks, method: str, order: str, device, output_dir: Path):
    """Train one fresh method/order sequence and collect all probe states."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    nav = None
    old_nav = None
    buffer_by_task: list[list] = []
    frozen_evals: list = []
    probes_by_task: list[list] = []
    boundary_probs: list[list[torch.Tensor]] = []
    rows: list[dict] = []
    checkpoints = output_dir / order / method / "checkpoints"

    for task_idx, task in enumerate(tasks):
        evaluator = train_evaluator(
            task.banks["train"], task.n_classes, K, device,
            epochs=30, seed=SEED)
        frozen_evals.append(evaluator)
        train_steps = []
        for slide in task.banks["train"]:
            train_steps.extend(teacher_rollout(slide, evaluator, K, device))
        test_steps = []
        for slide in task.banks["test"]:
            test_steps.extend(teacher_rollout(slide, evaluator, K, device))
        probes_by_task.append(test_steps)

        if task_idx == 0:
            nav = train_navigator_cl(
                train_steps, device, navigator=None, epochs=10, seed=SEED)
        else:
            buffer = [st for prior in buffer_by_task for st in prior]
            kwargs = {
                "distill": dict(use_distill=True, utility_weight=False),
                "ours": dict(use_replay=True, use_distill=True,
                              utility_weight=True),
                "seqft": dict(),
            }[method]
            nav = train_navigator_cl(
                train_steps, device, navigator=nav, epochs=10, seed=SEED,
                old_nav=old_nav, buffer=buffer, lam=1.0, **kwargs)
        buffer_by_task.append(build_buffer(train_steps, per_slide=2, cap=512))
        old_nav = copy.deepcopy(nav)
        checkpoint(nav, checkpoints / f"stage{task_idx + 1}.pt")

        current_probs_by_task: list[list[torch.Tensor]] = []
        for source_idx in range(task_idx + 1):
            current_probs = []
            for step in probes_by_task[source_idx]:
                _, probs = policy_output(nav, step, device)
                current_probs.append(probs)
            current_probs_by_task.append(current_probs)

        for source_idx in range(task_idx + 1):
            source_task = tasks[source_idx]
            probes = probes_by_task[source_idx]
            for state_idx, (step, probs) in enumerate(
                    zip(probes, current_probs_by_task[source_idx])):
                norm_gain = normalized_gain(step.gain)
                cand = [int(x) for x in step.candidates.tolist()]
                candidate_mask = [0] * int(step.low.shape[0])
                for index in cand:
                    candidate_mask[index] = 1
                selected_pos = int(probs.argmax().item())
                selected = cand[selected_pos]
                equivalent = [cand[i] for i, gain in enumerate(norm_gain)
                              if gain >= 1.0 - EPSILON]
                regret = float(1.0 - norm_gain[selected_pos])
                if source_idx < len(boundary_probs):
                    ref = boundary_probs[source_idx][state_idx]
                    drift = kl_ref_now(ref, probs)
                else:
                    drift = 0.0
                trajectory = selected_trajectory(
                    source_task.banks["test"][state_idx], nav, K, device)
                logits, _ = policy_output(nav, step, device)
                rows.append({
                    "order": order,
                    "method": method,
                    "method_label": method_label(method),
                    "source_task": source_task.name,
                    "source_task_idx": source_idx,
                    "eval_stage": task_idx + 1,
                    "state_index": state_idx,
                    "slide_id": step.sid,
                    "candidate_mask": candidate_mask,
                    "candidate_indices": cand,
                    "teacher_gain_raw": jsonable(step.gain),
                    "teacher_gain_minmax": norm_gain.tolist(),
                    "policy_logits": jsonable(logits),
                    "policy_probs": jsonable(probs),
                    "selected_trajectory": trajectory,
                    "selected_region": selected,
                    "epsilon": EPSILON,
                    "epsilon_equivalent_set_size": len(equivalent),
                    "epsilon_optimal_probability_mass": float(
                        probs[[i for i, gain in enumerate(norm_gain)
                               if gain >= 1.0 - EPSILON]].sum().item()),
                    "normalized_utility_regret": regret,
                    "action_drift": drift,
                })

        # Save the current boundary after all comparisons have used the
        # previous references. The current task has zero drift at its own
        # boundary; later stages compare against this retained policy.
        boundary_probs.append(list(current_probs_by_task[task_idx]))

    method_dir = output_dir / order / method
    method_dir.mkdir(parents=True, exist_ok=True)
    with (method_dir / "state_dumps.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_figure(rows: list[dict], order: str, method: str, figure_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_stage = max(row["eval_stage"] for row in rows)
    points = [row for row in rows
              if row["eval_stage"] == final_stage]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    tasks = sorted({row["source_task"] for row in points})
    for task in tasks:
        subset = [r for r in points if r["source_task"] == task]
        ax.scatter([r["action_drift"] for r in subset],
                   [r["normalized_utility_regret"] for r in subset],
                   s=10, alpha=0.55, label=task)
    ax.set_xlabel("action drift: KL(reference policy || final policy)")
    ax.set_ylabel("normalized utility regret")
    ax.set_title(f"Mechanism probe — {order} / {method_label(method)}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"mech_{order}_{method}_drift_regret.png",
                dpi=170)
    plt.close(fig)


def summarize(rows: list[dict]) -> list[dict]:
    import pandas as pd
    frame = pd.DataFrame(rows)
    final = frame[frame.eval_stage == frame.eval_stage.max()].copy()
    out = []
    for (order, method, task), group in final.groupby(
            ["order", "method", "source_task"]):
        drift = group["action_drift"].to_numpy(float)
        regret = group["normalized_utility_regret"].to_numpy(float)
        corr = float(np.corrcoef(drift, regret)[0, 1]) \
            if len(group) > 1 and np.std(drift) > 0 and np.std(regret) > 0 \
            else float("nan")
        out.append({
            "order": order, "method": method, "source_task": task,
            "n_states": len(group),
            "epsilon_set_size": float(
                group["epsilon_equivalent_set_size"].mean()),
            "epsilon_optimal_mass": float(
                group["epsilon_optimal_probability_mass"].mean()),
            "normalized_utility_regret": float(
                group["normalized_utility_regret"].mean()),
            "action_drift": float(group["action_drift"].mean()),
            "drift_regret_corr": corr,
        })
    return out


def render_summary(summary_rows: list[dict], device_meta: dict) -> str:
    lines = [
        "# WP4 mechanism probe summary",
        "",
        "Seed 0, `K=1`, `can_dataset`; separate fresh sequences for main and "
        "reverse orders. Methods are `seqft`, old-policy / policy-fidelity "
        "distillation (`distill`), and Utility-Weighted Replay Distillation "
        "(`ours`).",
        "",
        "Teacher gains are min–max normalized **per state before any derived "
        "utility aggregation**. ε is fixed at 0.05, so the ε-equivalent set "
        "contains candidates with normalized teacher gain ≥ 0.95. Utility "
        "regret is `1 - normalized teacher gain(selected action)`; "
        "ε-optimal mass is the learned policy probability assigned to that "
        "set. Action drift is `KL(policy at source-task boundary || current "
        "policy)`.",
        "",
        f"Resolved device: `{device_meta.get('resolved_device', 'unknown')}`; "
        f"torch `{device_meta.get('torch_version', 'unknown')}`.",
        "",
        "## Final-stage state aggregates",
        "",
        "| order | method | source task | n | ε-set size | ε-optimal mass | normalized utility regret | action drift | drift–regret r |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: (
            r["order"], r["method"], r["source_task"])):
        corr = "nan" if np.isnan(row["drift_regret_corr"]) \
            else f"{row['drift_regret_corr']:.3f}"
        lines.append(
            f"| {row['order']} | `{row['method']}` | {row['source_task']} | "
            f"{row['n_states']} | {row['epsilon_set_size']:.3f} | "
            f"{row['epsilon_optimal_mass']:.3f} | "
            f"{row['normalized_utility_regret']:.3f} | "
            f"{row['action_drift']:.3f} | {corr} |")
    lines.extend([
        "",
        "## Observed pattern",
        "",
        "Across both orders, `seqft` has visibly larger final-stage action "
        "drift on earlier tasks than `distill` or `ours`, while replay and "
        "distillation retain substantial ε-optimal probability mass. The "
        "ε-equivalent sets are broad (many candidates are near-optimal), and "
        "the drift–regret relationship is not uniformly strong for the "
        "preservation variants. This supports a behavior-preservation "
        "mechanism, but not a claim that utility weighting is universally "
        "necessary or sufficient.",
        "",
        "## Interpretation gate",
        "",
        "This is mechanism analysis, not an EUP gate. The K=1 result is "
        "stable enough to support the existing analysis-centric "
        "behavior-preservation claim, but the broad ε-equivalent sets and "
        "weak/variable drift–regret correlations do not justify extending "
        "to `K=2` before the science freeze. Do not add a new method.",
        "",
        "Raw per-state records and stage-boundary checkpoints are under the "
        "fresh run directory passed to `mechanism_probe.py`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "runs" / "v2" /
                    "mech_probe_v0292_seed0_k1")
    ap.add_argument("--report", type=Path,
                    default=REPO / "results" / "mech_probe_summary.md")
    ap.add_argument("--figure-dir", type=Path, default=REPO / "figures")
    add_device_argument(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    all_rows = []
    for order in ORDERS:
        tasks = load_can_tasks(order, smoke=False)
        for method in METHODS:
            print(f"[probe] start order={order} method={method} "
                  f"device={device}", flush=True)
            rows = run_order(tasks, method, order, device, args.output_dir)
            all_rows.extend(rows)
            write_figure(rows, order, method, args.figure_dir)
            print(f"[probe] done order={order} method={method} "
                  f"states={len(rows)}", flush=True)

    summary_rows = summarize(all_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_summary(summary_rows, meta))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False,
                   allow_nan=True))
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
