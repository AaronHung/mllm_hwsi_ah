"""主實驗：任務序列上的 navigation forgetting 與緩解（main table 資料來源）。

Protocol（凍結）見 docs/protocol.md：
    - dataset=can    ：ESCA → LUNG → RCC → BRCA（fold_1 patient split，pseudo-region 環境）
    - dataset=pilot40：renal → breast（stratified 70/30 per seed，HIPT/CONCH 層級環境）
    - 方法：seqft / ewc / lwf / replay / distill / ours_uniform / ours / joint
      （paper-facing labels: counterfactual-teacher replay; old-policy /
      policy-fidelity distillation; Utility-Weighted Replay Distillation;
      joint-training reference）
    - K ∈ {1,2,4}；seeds {0..4}；指標見 protocol §4

v0.33.2 method-gate configs (docs/method_gate_v033.md; NOT part of the frozen
Protocol-v1 table above): ia_samp / eq_pres / ia_ep / samp_util_only /
samp_drift_only. Pass them via --methods like any other method key; they
read nav.cl.METHOD_GATE_KWARGS instead of METHOD_KWARGS. Gate rows also get
two utility-axis columns (eps_optimal_mass, normalized_regret) and, for gate
methods only, real per-stage navigator checkpoints under
runs/v2/<tag>/ckpt/ (checkpoints.json stays --resume bookkeeping only).

用法：
    python scripts/cl_main.py --dataset can --smoke                  # Mac 煙測
    python scripts/cl_main.py --dataset can --order main             # 完整（可用 --methods/--seeds/--budgets 分片）
    python scripts/cl_main.py --dataset can --select-ewc-lambda      # EWC λ 的 val 選擇（跑一次）
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nav import (add_device_argument, device_info,  # noqa: E402
                 resolve_device, run_provenance)
from nav.candata import COHORTS, CanTask  # noqa: E402
from nav.cl import (METHOD_GATE_KWARGS, METHOD_KWARGS,  # noqa: E402
                    attach_boundary_policy, build_buffer,
                    chunkwise_percentile_rank, epsilon_optimal_mask,
                    eval_policy_balanced, fisher_of, flatten_task_labels,
                    jaccard, kl_drift, normalized_gain, policy_on_probes,
                    train_navigator_cl, trajectory_utility)
from nav.engine import teacher_rollout, train_evaluator  # noqa: E402
from nav.pilot40 import Pilot40  # noqa: E402

RESULTS = REPO / "results"
CACHE_ROOT = REPO / "data" / "can_cache"

CAN_ORDERS = {"main": ["esca", "lung", "rcc", "brca"],
              "reverse": ["brca", "rcc", "lung", "esca"]}
PILOT_TASKS = {"T1_renal": ["KIRC", "KIRP"], "T2_breast": ["IDC", "ILC"]}


@dataclass
class TaskSpec:
    name: str
    n_classes: int
    banks: dict  # split -> list[Slide]（pilot40 的 split 依 seed 重抽，見 get_banks）


# ------------------------------------------------------------------ data
def load_can_tasks(order: str, smoke: bool) -> list[TaskSpec]:
    tasks = []
    cohorts = CAN_ORDERS[order][:2] if smoke else CAN_ORDERS[order]
    for cohort in cohorts:
        t = CanTask(cohort, CACHE_ROOT)
        banks = {sp: t.load_bank(sp) for sp in ["train", "val", "test"]}
        if smoke:
            banks = {sp: _cap_per_class(b, {"train": 8, "val": 4, "test": 6}[sp],
                                        len(t.classes))
                     for sp, b in banks.items()}
        tasks.append(TaskSpec(name=cohort, n_classes=len(t.classes), banks=banks))
    return tasks


def _cap_per_class(bank, n_per_class, n_classes):
    out, cnt = [], {c: 0 for c in range(n_classes)}
    for s in bank:
        if cnt[s.y] < n_per_class:
            out.append(s)
            cnt[s.y] += 1
    return out


def load_pilot_tasks(seed: int, smoke: bool) -> list[TaskSpec]:
    from sklearn.model_selection import train_test_split
    from nav.engine import build_bank
    ds = Pilot40(REPO / "data")
    tasks = []
    for name, classes in PILOT_TASKS.items():
        sids = [s for s in ds.slide_ids if ds.label4(s) in classes]
        ys = [classes.index(ds.label4(s)) for s in sids]
        tr, te = train_test_split(sids, test_size=0.3, stratify=ys,
                                  random_state=seed)
        lab = {s: classes.index(ds.label4(s)) for s in sids}
        banks = {"train": build_bank(ds, tr, lab), "val": [],
                 "test": build_bank(ds, te, lab)}
        tasks.append(TaskSpec(name=name, n_classes=2, banks=banks))
    return tasks


# ------------------------------------------------------------------ sequence
def run_sequence(tasks: list[TaskSpec], method: str, seed: int, k: int,
                 device, ev_epochs: int, nav_epochs: int,
                 lam: float, lam_ewc: float,
                 replay_per_slide: int = 2, buffer_cap: int = 512,
                 eval_split: str = "test",
                 run_meta: dict[str, object] | None = None,
                 diag_log: list[dict] | None = None,
                 ckpt_dir: Path | None = None) -> list[dict]:
    torch.manual_seed(seed)
    t_start = time.time()

    # v0.33.1 method-gate configs (docs/method_gate_v033.md) live in
    # METHOD_GATE_KWARGS, never in the frozen METHOD_KWARGS table.
    method_kwargs = dict(METHOD_KWARGS.get(method)
                         or METHOD_GATE_KWARGS.get(method, {}))
    needs_boundary_policy = method_kwargs.get("replay_sampling") == "importance"

    nav = None
    old_nav = None
    frozen_evals: list = []
    steps_by_task: list[list] = []
    buffer_by_task: list[list] = []
    ewc_terms: list = []
    sel_at_own: list[dict] = []
    probes: list[list] = []
    pi_at_own: list[list] = []
    random_ref: list[float] = []
    rows = []

    for t, task in enumerate(tasks):
        # ---- Stage A：evaluator（訓練後凍結） ----
        ev = train_evaluator(task.banks["train"], task.n_classes, k, device,
                             epochs=ev_epochs, seed=seed)
        frozen_evals.append(ev)
        # ---- Stage B：teacher rollout ----
        steps = []
        for s in task.banks["train"]:
            steps += teacher_rollout(s, ev, k, device)
        steps_by_task.append(steps)
        # ---- Stage C：navigator 更新（依方法） ----
        if method == "joint":
            all_steps = [st for ss in steps_by_task for st in ss]
            nav = train_navigator_cl(all_steps, device, navigator=None,
                                     epochs=nav_epochs, seed=seed)
        elif t == 0:
            nav = train_navigator_cl(steps, device, navigator=nav,
                                     epochs=nav_epochs, seed=seed)
        else:
            buffer = [st for bb in buffer_by_task for st in bb]
            gate_extra: dict[str, object] = {}
            if needs_boundary_policy:
                # v0.33.2 M1 fix: ranks computed per source-task chunk
                # BEFORE flattening, then concatenated in the exact same
                # order as `buffer` above — index alignment with eps_masks
                # (built from this same `buffer` inside train_navigator_cl)
                # is guaranteed by construction, not by re-sorting.
                gate_extra["u_state_override"] = chunkwise_percentile_rank(
                    buffer_by_task)
                gate_extra["task_labels"] = flatten_task_labels(
                    buffer_by_task,
                    [tasks[i].name for i in range(len(buffer_by_task))])
            nav = train_navigator_cl(
                steps, device, navigator=nav, epochs=nav_epochs, seed=seed,
                old_nav=old_nav, buffer=buffer, lam=lam,
                ewc_terms=ewc_terms if method == "ewc" else None,
                lam_ewc=lam_ewc, diag_log=diag_log, **gate_extra,
                **method_kwargs)

        # ---- 任務結束的 bookkeeping ----
        if method == "ewc":
            ewc_terms.append(fisher_of(nav, steps, device, seed=seed))
        buffer_by_task.append(build_buffer(steps, per_slide=replay_per_slide,
                                           cap=buffer_cap))
        if needs_boundary_policy:
            # M1 reads d_t(s) against this snapshot in every later stage.
            attach_boundary_policy(buffer_by_task[-1], nav, device)
        old_nav = copy.deepcopy(nav)
        if ckpt_dir is not None:
            # v0.33.2 item 4: real weights (stage-boundary == final on the
            # last stage) so a passing config can be probed post-gate;
            # checkpoints.json remains --resume bookkeeping, unrelated.
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(nav.state_dict(),
                      ckpt_dir / f"{method}_seed{seed}_K{k}_stage{t + 1}.pt")

        te_bank = task.banks[eval_split]
        _, _, sel = eval_policy_balanced(te_bank, ev, k, device, "learned",
                                         navigator=nav,
                                         n_classes=task.n_classes)
        sel_at_own.append(sel)
        pstates = []
        for s in te_bank:
            pstates += teacher_rollout(s, ev, k, device)
        probes.append(pstates)
        pi_at_own.append(policy_on_probes(nav, pstates, device))
        r_accs = [eval_policy_balanced(te_bank, ev, k, device, "random",
                                       seed=seed * 100 + rs,
                                       n_classes=task.n_classes)[0]
                  for rs in range(5)]
        random_ref.append(float(np.mean(r_accs)))

        # ---- 對所有 j ≤ t 量測 ----
        for j in range(t + 1):
            bank_j = tasks[j].banks[eval_split]
            bal, acc, sel_now = eval_policy_balanced(
                bank_j, frozen_evals[j], k, device, "learned", navigator=nav,
                n_classes=tasks[j].n_classes)
            jac = float(np.mean([jaccard(sel_at_own[j][sid], sel_now[sid])
                                 for sid in sel_now]))
            pi_now_j = policy_on_probes(nav, probes[j], device)
            drift = kl_drift(pi_at_own[j], pi_now_j)
            util = trajectory_utility(bank_j, nav, frozen_evals[j], k, device)
            # v0.33.2 item 2: utility-axis probe metrics, reusing the same
            # policy_on_probes forward pass already computed for `drift`
            # above (no extra forward cost). Aggregation for the gate
            # verdict (final stage, old tasks only, per-task mean then
            # macro-average) is pre-registered in docs/method_gate_v033.md;
            # here we just write the per-stage per-task mean over probe
            # states, like every other row-level metric.
            eps_mass_vals, regret_vals = [], []
            for st, probs in zip(probes[j], pi_now_j):
                mask = epsilon_optimal_mask(st.gain, eps=0.05)
                p = probs.numpy()
                eps_mass_vals.append(float(p[mask].sum()))
                regret_vals.append(
                    float(1.0 - (p * normalized_gain(st.gain)).sum()))
            rows.append(dict(
                method=method, seed=seed, K=k, stage=t + 1,
                task=tasks[j].name, task_idx=j + 1,
                bal_acc=round(bal, 4), acc=round(acc, 4),
                jaccard=round(jac, 4), action_kl=round(drift, 4),
                sel_utility=round(util, 4),
                eps_optimal_mass=round(float(np.mean(eps_mass_vals)), 4),
                normalized_regret=round(float(np.mean(regret_vals)), 4),
                random_ref=round(random_ref[j], 4),
                n_test=len(bank_j), wall_s=round(time.time() - t_start, 1),
                **(run_meta or {})))
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["can", "pilot40"], default="can")
    ap.add_argument("--order", choices=["main", "reverse"], default="main")
    ap.add_argument("--methods", nargs="+",
                    default=["seqft", "ewc", "lwf", "replay", "distill",
                             "ours", "joint"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replay-per-slide", type=int, default=2)
    ap.add_argument("--buffer-cap", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    add_device_argument(ap)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new-run directory; defaults to runs/v2/<tag>")
    ap.add_argument("--resume", action="store_true",
                    help="skip completed seed/method/K checkpoints in output-dir")
    ap.add_argument("--select-ewc-lambda", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    meta = device_info(args.device, device).as_dict()
    meta["resolved_device"] = meta.pop("resolved")
    meta["run_tag"] = args.tag or ("smoke" if args.smoke else "full")
    meta.update(run_provenance())  # v0.33.3 item E5: commit/hostname/command
    tag = meta["run_tag"]
    out_dir = (Path(args.output_dir) if args.output_dir is not None
               else REPO / "runs" / "v2" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(
        meta, indent=2, ensure_ascii=False))
    print(f"device = {device} | torch = {meta['torch_version']} | "
          f"MPS fallback = {meta['mps_fallback']}")

    if args.smoke:
        args.methods = ["seqft", "ours"] if args.methods == ap.get_default(
            "methods") else args.methods
        args.budgets = [1]
        args.seeds = [0]
        ev_epochs, nav_epochs = 5, 2
    elif args.dataset == "can":
        ev_epochs, nav_epochs = 30, 10
    else:
        ev_epochs, nav_epochs = 60, 30

    # EWC λ：val 選擇模式 or 讀取既有選擇
    lam_ewc_path = out_dir / "ewc_lambda.json"
    if args.select_ewc_lambda:
        assert args.dataset == "can", "λ 選擇依 protocol 只在 can 的 val 上做"
        tasks = load_can_tasks("main", smoke=False)[:2]  # T1→T2
        best = None
        for lam_ewc in [10.0, 100.0, 1000.0]:
            rows = run_sequence(tasks, "ewc", seed=0, k=2, device=device,
                                ev_epochs=ev_epochs, nav_epochs=nav_epochs,
                                lam=args.lam, lam_ewc=lam_ewc, eval_split="val",
                                run_meta=meta)
            aa = np.mean([r["bal_acc"] for r in rows if r["stage"] == 2])
            print(f"[ewc-select] λ={lam_ewc}: val AA(after T2) = {aa:.4f}")
            if best is None or aa > best[1]:
                best = (lam_ewc, aa)
        lam_ewc_path.write_text(json.dumps(
            {"lam_ewc": best[0], "val_aa": round(float(best[1]), 4)}))
        print(f"selected λ_ewc = {best[0]} -> {lam_ewc_path}")
        return

    lam_ewc = 100.0
    if lam_ewc_path.exists():
        lam_ewc = json.loads(lam_ewc_path.read_text())["lam_ewc"]
    print(f"λ_ewc = {lam_ewc}")

    out_csv = out_dir / f"cl_main_{args.dataset}_{args.order}_{tag}.csv"
    checkpoint_path = out_dir / "checkpoints.json"
    completed: set[str] = set()
    if args.resume and checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text()).get("completed", []))

    if args.dataset == "can":
        tasks_fixed = load_can_tasks(args.order, args.smoke)

    t0 = time.time()
    for seed in args.seeds:
        tasks = tasks_fixed if args.dataset == "can" else \
            load_pilot_tasks(seed, args.smoke)
        for method in args.methods:
            for k in args.budgets:
                key = f"seed={seed}|method={method}|K={k}"
                if key in completed:
                    print(f"[resume] skip {key}")
                    continue
                diag_log: list[dict] = [] if method in METHOD_GATE_KWARGS else None
                ckpt_dir = (out_dir / "ckpt"
                           if method in METHOD_GATE_KWARGS else None)
                rows = run_sequence(tasks, method, seed, k, device,
                                    ev_epochs, nav_epochs,
                                    args.lam, lam_ewc,
                                    args.replay_per_slide, args.buffer_cap,
                                    run_meta=meta, diag_log=diag_log,
                                    ckpt_dir=ckpt_dir)
                if diag_log:
                    diag_path = out_dir / f"diag_{method}_seed{seed}_K{k}.json"
                    diag_path.write_text(json.dumps(diag_log, indent=2))
                for r in rows:
                    r.update(dataset=args.dataset, order=args.order)
                df = pd.DataFrame(rows)
                header = not out_csv.exists()
                df.to_csv(out_csv, mode="a", header=header, index=False)
                completed.add(key)
                checkpoint_path.write_text(json.dumps(
                    {"completed": sorted(completed), "metadata": meta},
                    indent=2, ensure_ascii=False))
                fin = [r for r in rows if r["stage"] == len(tasks)]
                aa = np.mean([r["bal_acc"] for r in fin])
                jac1 = fin[0]["jaccard"]
                print(f"[{method} seed{seed} K={k}] AA={aa:.3f} "
                      f"T1(bal={fin[0]['bal_acc']:.3f}, jac={jac1:.2f}, "
                      f"kl={fin[0]['action_kl']:.3f}) "
                      f"({time.time() - t0:.0f}s)")
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
