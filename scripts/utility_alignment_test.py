"""v0.35-closeout analysis script — E1 (utility-capability alignment) and E2
(eq_pres_norep attribution read-out).

E1 is pre-registered in `docs/alignment_test.md`, committed before this script
was run. E2's composition and read-out rule are pre-registered in the closeout
authorization and recorded in `docs/track_c0.md`'s changelog.

**Why one script for both.** The closeout authorization lists exactly one new
script among the files allowed to change. Keeping the E2 read-out here rather
than in an uncommitted scratch file is the choice that keeps both artifacts
reproducible from the repo, which matters more than the file name being a
perfect fit. `e1` and `e2` are separate subcommands and share nothing but
loaders.

Neither subcommand trains a navigator, reruns a comparator, reopens a verdict,
or feeds a promotion path.

用法：
    python scripts/utility_alignment_test.py e1
    python scripts/utility_alignment_test.py e2 --c0-csv runs/v2/<tag>/....csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from aggregate_results import per_run_metrics  # noqa: E402
from cl_main import load_can_tasks  # noqa: E402
from method_gate_v2_verdict import utility_axis_final_old_tasks  # noqa: E402
from nav import resolve_device  # noqa: E402
from nav.cl import epsilon_optimal_mask, eval_policy_balanced, normalized_gain  # noqa: E402
from nav.engine import evidence_of, teacher_rollout, train_evaluator  # noqa: E402

FULL_CSVS = [REPO / f"results/cl_main_can_main_full_s{s}.csv" for s in (0, 1, 2)]
GATE_V0333 = (REPO / "runs/v2/method_gate_v0333_run1"
              / "cl_main_can_main_method_gate_v0333_run1.csv")
GATE_V2 = (REPO / "runs/v2/gate_v2_20260816T073200Z"
           / "cl_main_can_main_gate_v2_20260816T073200Z.csv")
GATE_V034 = (REPO / "runs/v2/gate_v034_dev_20260816T154548Z"
             / "cl_main_can_main_gate_v034_dev_20260816T154548Z.csv")
TRACK_C0 = (REPO / "runs/v2/track_c0_20260817T014543Z"
            / "cl_main_can_main_track_c0_20260817T014543Z.csv")
UTIL_REGEN = (REPO / "runs/v2/util_regen_20260815_mps"
              / "cl_main_can_main_util_regen_20260815_mps.csv")
OUT_E1 = REPO / "results/utility_alignment_report.md"
OUT_E2 = REPO / "results/track_c0_ablation_norep.md"

SEEDS = [0, 1, 2]
RULE_KS = [1, 2]          # the pre-registered read-out rule (docs/alignment_test.md 7)
EXTRA_KS = [4]            # descriptive context only, explicitly outside the rule
MAIN_ORDER = ["esca", "lung", "rcc", "brca"]
EV_EPOCHS = 30


def load_concat(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths if p.exists()],
                     ignore_index=True)


def fmt(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]" if v else "N/A"


# ============================================================== E1
@torch.no_grad()
def oracle_pass(bank, evaluator, k: int, device) -> tuple[float, float, float,
                                                          float]:
    """Greedy label-informed oracle on one task's test bank.

    The trajectory is exactly `teacher_rollout`'s: at each step it zooms the
    argmax of the counterfactual gain computed against the slide's true label
    through the frozen evaluator (docs/alignment_test.md 3). Nothing is
    trained here and no random number is drawn.

    Returns (balanced_acc, acc, mean eps_optimal_mass, mean normalized_regret)
    where the last two are the internal-consistency check of 8: a policy that
    is the argmax of the quantity those metrics score should sit near the top
    of one and near zero on the other.
    """
    evaluator.eval()
    ys, preds, eps_mass, regret = [], [], [], []
    for s in bank:
        steps = teacher_rollout(s, evaluator, k, device)
        sel = [int(st.candidates[int(st.gain.argmax())]) for st in steps]
        preds.append(int(evaluator(evidence_of(s, sel, device)[None]).argmax()))
        ys.append(s.y)
        for st in steps:
            # One-hot at the argmax: the oracle's own "policy" on this state.
            p = np.zeros(len(st.candidates))
            p[int(st.gain.argmax())] = 1.0
            eps_mass.append(float(p[epsilon_optimal_mask(st.gain, 0.05)].sum()))
            regret.append(float(1.0 - (p * normalized_gain(st.gain)).sum()))
    ys, preds = np.array(ys), np.array(preds)
    ncls = int(ys.max()) + 1
    recalls = [float((preds[ys == c] == c).mean()) for c in range(ncls)
               if (ys == c).any()]
    return (float(np.mean(recalls)), float((ys == preds).mean()),
            float(np.mean(eps_mass)), float(np.mean(regret)))


def run_e1(device) -> None:
    tasks = load_can_tasks("main", smoke=False)
    frozen = load_concat(FULL_CSVS)
    seqft = frozen[(frozen.method == "seqft") & (frozen.stage == frozen.task_idx)]

    rows: list[dict] = []
    for pass_id in (1, 2):                        # 3: determinism assertion
        for seed in SEEDS:
            for k in RULE_KS + EXTRA_KS:
                for t_idx, task in enumerate(tasks, start=1):
                    torch.manual_seed(seed)       # 4: stated, deterministic
                    ev = train_evaluator(task.banks["train"], task.n_classes,
                                         k, device, epochs=EV_EPOCHS, seed=seed)
                    bal, acc, em, reg = oracle_pass(task.banks["test"], ev, k,
                                                    device)
                    rnd = [eval_policy_balanced(task.banks["test"], ev, k,
                                                device, "random",
                                                seed=seed * 100 + r,
                                                n_classes=task.n_classes)[0]
                           for r in range(5)]
                    rows.append(dict(pass_id=pass_id, seed=seed, K=k,
                                     task=task.name, task_idx=t_idx,
                                     oracle_bal=bal, oracle_acc=acc,
                                     oracle_eps_mass=em, oracle_regret=reg,
                                     random_bal=float(np.mean(rnd)),
                                     n_test=len(task.banks["test"])))
                print(f"  [E1] pass{pass_id} seed{seed} K={k} {task.name}: "
                      f"oracle bal={bal:.4f}")
    df = pd.DataFrame(rows)

    p1 = df[df.pass_id == 1].drop(columns="pass_id").reset_index(drop=True)
    p2 = df[df.pass_id == 2].drop(columns="pass_id").reset_index(drop=True)
    deterministic = p1.equals(p2)
    if not deterministic:
        raise SystemExit("STOP: oracle is non-deterministic across two "
                         "identical passes (docs/alignment_test.md 3)")

    L: list[str] = []
    L.append("# E1 — Utility-Capability Alignment Test (report)\n")
    L.append(
        "**Track A:** frozen; writing is the only remaining work. "
        "**Track B/C:** Track B closed and archived, Track C0 screened and "
        "FAILed; this is one of the two v0.35-closeout artifacts, after which "
        "all compute for this paper ends permanently.\n")
    L.append(
        "Pre-registered in `docs/alignment_test.md`, committed before this "
        "script ran. **No navigator is trained anywhere in E1** and no "
        "comparator was rerun. The oracle is **LABEL-INFORMED** (§1 of the "
        "pre-registration): it is a diagnostic, never a deployable policy.\n")
    L.append(f"**Determinism check (§3): PASS** — two identical oracle passes "
             f"over {len(p1)} cells produced identical tables.\n")

    # -- internal consistency (8)
    L.append("## Internal consistency (§8)\n")
    L.append(f"- oracle mean `eps_optimal_mass` = "
             f"**{p1.oracle_eps_mass.mean():.4f}** (a policy that is the "
             "argmax of the gain should be inside the epsilon-optimal set "
             "essentially always)")
    L.append(f"- oracle mean `normalized_regret` = "
             f"**{p1.oracle_regret.mean():.4f}** (should be near zero by "
             "construction)\n")

    # -- evaluator equivalence (4.1)
    L.append("## Evaluator equivalence check (§4.1)\n")
    L.append("E1 retrains each per-task evaluator; the check is whether the "
             "random policy scores the same on it as on the frozen run's. "
             "Frozen values come from the `random_ref` column.\n")
    L.append("| K | task | E1 random (mean over seeds) | frozen `random_ref` | "
             "difference |")
    L.append("|---|---|---|---|---|")
    eq_diffs = []
    for k in RULE_KS + EXTRA_KS:
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            mine = p1[(p1.K == k) & (p1.task_idx == t_idx)].random_bal.mean()
            fz = seqft[(seqft.K == k) & (seqft.task_idx == t_idx)
                       & (seqft.seed.isin(SEEDS))].random_ref.mean()
            eq_diffs.append(abs(mine - fz))
            L.append(f"| {k} | {task} | {mine:.4f} | {fz:.4f} | "
                     f"{mine - fz:+.4f} |")
    L.append("")
    L.append(f"Largest absolute difference: **{max(eq_diffs):.4f}**. Read this "
             "as the size of the cross-evaluator term that the primary "
             "comparison below carries; the random-anchored columns are free "
             "of it.\n")

    # -- main table + rule
    L.append("## Oracle vs `seqft` own-time accuracy\n")
    L.append("| K | task | quantum `1/n_test` | oracle bal-acc | `seqft` "
             "own-time | oracle − seqft | oracle − random (E1) | seqft − "
             "random_ref (frozen) |")
    L.append("|---|---|---|---|---|---|---|---|")
    tier_inputs: dict[int, dict] = {}
    for k in RULE_KS + EXTRA_KS:
        per_task = []
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            o = p1[(p1.K == k) & (p1.task_idx == t_idx)]
            s = seqft[(seqft.K == k) & (seqft.task_idx == t_idx)
                      & (seqft.seed.isin(SEEDS))]
            n_test = int(o.n_test.iloc[0])
            q = 1.0 / n_test
            om, sm = o.oracle_bal.mean(), s.bal_acc.mean()
            per_task.append((task, om, sm, q))
            L.append(
                f"| {k}{' *' if k in EXTRA_KS else ''} | {task} | "
                f"{q:.4f} (n={n_test}) | {om:.4f} | {sm:.4f} | {om - sm:+.4f} "
                f"| {om - o.random_bal.mean():+.4f} | "
                f"{sm - s.random_ref.mean():+.4f} |")
        tier_inputs[k] = {
            "oracle_mean": float(np.mean([p[1] for p in per_task])),
            "seqft_mean": float(np.mean([p[2] for p in per_task])),
            "quantum_mean": float(np.mean([p[3] for p in per_task])),
            "n_better": sum(1 for p in per_task if p[1] > p[2]),
            "n_tasks": len(per_task)}
    L.append("")
    L.append("`*` K=4 is descriptive context only and is **not** part of the "
             "read-out rule (§6).\n")

    L.append("## Pre-registered read-out (§7)\n")
    L.append("| K | oracle cross-task mean | `seqft` cross-task mean | "
             "difference | mean quantum | tasks where oracle is better |")
    L.append("|---|---|---|---|---|---|")
    for k in RULE_KS:
        ti = tier_inputs[k]
        L.append(f"| {k} | {ti['oracle_mean']:.4f} | {ti['seqft_mean']:.4f} | "
                 f"{ti['oracle_mean'] - ti['seqft_mean']:+.4f} | "
                 f"{ti['quantum_mean']:.4f} | "
                 f"{ti['n_better']}/{ti['n_tasks']} |")
    L.append("")
    mis = all(tier_inputs[k]["oracle_mean"] <= tier_inputs[k]["seqft_mean"]
              for k in RULE_KS)
    align = all(
        tier_inputs[k]["oracle_mean"]
        >= tier_inputs[k]["seqft_mean"] + tier_inputs[k]["quantum_mean"]
        and tier_inputs[k]["n_better"] * 2 > tier_inputs[k]["n_tasks"]
        for k in RULE_KS)
    tier = ("MISALIGNMENT SUPPORTED" if mis else
            "ALIGNMENT SUPPORTED" if align else "INDETERMINATE")
    L.append(f"### Tier: **{tier}**\n")
    if tier == "MISALIGNMENT SUPPORTED":
        L.append(
            "A label-informed policy that maximizes the one-step "
            "counterfactual gain at every step does **not** exceed a plainly "
            "sequential-fine-tuned navigator's own-time accuracy. The "
            "non-conversion this project observed across five loss-level "
            "interventions and one architecture-level one is therefore a "
            "property of **the one-step utility target itself**, not of the "
            "methods that preserved it. This is a limitation of the "
            "measurement apparatus and belongs in the paper as one.\n")
    elif tier == "ALIGNMENT SUPPORTED":
        L.append(
            "The one-step target does convert into capability when it is "
            "maximized with label information. The non-conversion seen in "
            "Tracks B and C is then a property of the methods or of the "
            "optimization, not of the target. Descriptive; no further runs "
            "are authorized either way.\n")
    else:
        L.append(
            "Neither tier is met. Written up as a limitation: at this sample "
            "size the comparison cannot separate the two readings. **No "
            "further runs**, per §7.\n")

    # -- zero-compute correlation half (9)
    L.append("## Zero-compute correlation half (§9)\n")
    L.append(
        "Across every configuration this project ran, does the utility axis "
        "move together with capability? Per-seed deltas against each "
        "comparator; Spearman is **descriptive**, with no p/q language, per "
        "the repo statistical policy.\n")
    corr = correlation_half()
    L += corr["lines"]

    OUT_E1.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_E1.relative_to(REPO)}  -> tier: {tier}")


def correlation_half() -> dict:
    """delta eps-mass vs delta AA / delta Forgetting, every config, from CSVs."""
    v0333 = pd.read_csv(GATE_V0333)
    v2 = pd.read_csv(GATE_V2)
    v034 = pd.read_csv(GATE_V034).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")
    c0 = pd.read_csv(TRACK_C0)
    eq_pres = pd.concat(
        [v0333[v0333.method == "eq_pres"],
         v2[(v2.method == "eq_pres") & (v2.gate_v2_role == "candidate_new")]],
        ignore_index=True)
    cand = pd.concat([eq_pres, v0333[v0333.method.isin(["ia_samp", "ia_ep"])],
                      v034[v034.method != "eq_pres_diag"], c0],
                     ignore_index=True)
    comp_util = pd.concat([pd.read_csv(UTIL_REGEN),
                           v2[v2.gate_v2_role == "utility_backfill_seeds34"]],
                          ignore_index=True)
    util = pd.concat([utility_axis_final_old_tasks(cand),
                      utility_axis_final_old_tasks(comp_util)],
                     ignore_index=True)
    frozen = load_concat(FULL_CSVS)
    abl = load_concat([REPO / f"results/cl_main_can_main_ablA1_s{s}.csv"
                       for s in range(5)])
    metrics = pd.concat(
        [per_run_metrics(cand[cand.method == m]) for m in cand.method.unique()]
        + [per_run_metrics(frozen[frozen.method == "distill"]),
           per_run_metrics(abl[abl.method == "ours_uniform"])],
        ignore_index=True)

    has_eq = {"eq_pres", "ia_ep", "proj_eq_pres", "conflict_eq_pres",
              "sp_nav_eq"}
    lines = ["| config | `L_eq` | K | comparator | Δε-mass | ΔAA | "
             "ΔForgetting | signs |", "|---|---|---|---|---|---|---|---|"]
    pts: list[tuple[float, float]] = []
    for cfg in sorted(set(util.method) & set(metrics.method) - {"distill",
                                                               "ours_uniform"}):
        for k in (1, 2):
            for comp in ("ours_uniform", "distill"):
                for s in sorted(SEEDS + [3, 4]):
                    def get(df, m, col):
                        r = df[(df.method == m) & (df.K == k) & (df.seed == s)]
                        return float(r[col].iloc[0]) if len(r) else None
                    du = (None if get(util, cfg, "eps_optimal_mass") is None
                          or get(util, comp, "eps_optimal_mass") is None else
                          get(util, cfg, "eps_optimal_mass")
                          - get(util, comp, "eps_optimal_mass"))
                    da = (None if get(metrics, cfg, "AA") is None
                          or get(metrics, comp, "AA") is None else
                          get(metrics, cfg, "AA") - get(metrics, comp, "AA"))
                    df_ = (None if get(metrics, cfg, "forgetting") is None
                           or get(metrics, comp, "forgetting") is None else
                           get(metrics, cfg, "forgetting")
                           - get(metrics, comp, "forgetting"))
                    if du is None or da is None:
                        continue
                    pts.append((du, da))
                    lines.append(
                        f"| `{cfg}` | {'yes' if cfg in has_eq else 'no'} | {k} "
                        f"| `{comp}` (seed {s}) | {du:+.4f} | {da:+.4f} | "
                        f"{'N/A' if df_ is None else f'{df_:+.4f}'} | "
                        f"{'+' if du > 0 else '−'}{'+' if da > 0 else '−'} |")
    lines.append("")
    if len(pts) >= 3:
        rho, _ = spearmanr([p[0] for p in pts], [p[1] for p in pts])
        n_up_u = sum(1 for p in pts if p[0] > 0)
        n_up_both = sum(1 for p in pts if p[0] > 0 and p[1] > 0)
        lines.append(
            f"**Spearman(Δε-mass, ΔAA) = {rho:+.3f}** over {len(pts)} "
            "config x K x comparator x seed observations — **descriptive**, "
            "no significance claim. Of the "
            f"{n_up_u} observations where the utility axis improved, "
            f"{n_up_both} ({100 * n_up_both / max(n_up_u, 1):.0f}%) also "
            "improved `AA`.\n")
        lines.append(
            "**Read:** if the utility axis moved consistently while `AA` did "
            "not track it, this is the same non-conversion the gates found, "
            "measured across every configuration this project ever ran rather "
            "than one at a time.\n")
    return {"lines": lines}


# ============================================================== E2
def run_e2(c0_csv: Path) -> None:
    norep = pd.read_csv(c0_csv).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")
    norep = norep[norep.method == "eq_pres_norep"]
    frozen = load_concat(FULL_CSVS)
    distill = frozen[frozen.method == "distill"]
    comp_util = pd.read_csv(UTIL_REGEN)

    metrics = pd.concat([per_run_metrics(norep), per_run_metrics(distill)],
                        ignore_index=True)
    util = pd.concat([utility_axis_final_old_tasks(norep),
                      utility_axis_final_old_tasks(comp_util)],
                     ignore_index=True)

    def paired(df, col, a, b, k):
        ta = df[(df.method == a) & (df.K == k)].set_index("seed")
        tb = df[(df.method == b) & (df.K == k)].set_index("seed")
        return [float(ta.loc[s, col] - tb.loc[s, col]) for s in SEEDS
                if s in ta.index and s in tb.index]

    L: list[str] = []
    L.append("# E2 — `eq_pres_norep`: completing the C0 attribution square\n")
    L.append(
        "**Track A:** frozen; writing is the only remaining work. "
        "**Track B/C:** Track B closed and archived, Track C0 screened and "
        "FAILed; this is the second of the two v0.35-closeout artifacts, "
        "after which all compute for this paper ends permanently.\n")
    L.append(
        f"Computed from `{c0_csv.relative_to(REPO)}` (6 units: "
        "`eq_pres_norep`, seeds {0,1,2} x K{1,2}, main order, Mac MPS). "
        "`distill` and the utility backfill are frozen comparators and were "
        "not rerun. **This is a descriptive attribution, not a gate**: no "
        "verdict is reopened and nothing here promotes anything.\n")

    L.append("## Read-out\n")
    L.append("| criterion | K | mean [seeds 0,1,2] | holds |")
    L.append("|---|---|---|---|")
    forget_ok, util_ok = True, True
    for k in RULE_KS:
        d = paired(metrics, "forgetting", "eq_pres_norep", "distill", k)
        ok = bool(d) and float(np.mean(d)) <= 0
        forget_ok = forget_ok and ok
        L.append(f"| Forgetting ≤ `distill` | {k} | "
                 f"{np.mean(d):+.4f} {fmt(d)} | {'yes' if ok else 'no'} |")
    for k in RULE_KS:
        d = paired(util, "eps_optimal_mass", "eq_pres_norep", "distill", k)
        ok = bool(d) and float(np.mean(d)) > 0
        util_ok = util_ok and ok
        L.append(f"| Δε-mass > 0 vs `distill` | {k} | "
                 f"{np.mean(d):+.4f} {fmt(d)} | {'yes' if ok else 'no'} |")
    L.append("")
    sufficient = forget_ok and util_ok
    word = "sufficient" if sufficient else "NOT sufficient"
    L.append(f"### **`L_eq` alone {word} (descriptive)**\n")

    L.append("## The attribution square\n")
    L.append("| corner | replay imitation | `L_eq` | per-task adapter | status |")
    L.append("|---|---|---|---|---|")
    L.append("| `eq_pres` | yes | yes | no | frozen (Gate v2) |")
    L.append("| `eq_pres_norep` | no | yes | no | **this experiment** |")
    L.append("| `sp_nav_eq` | no | yes | yes | frozen (Track C0) |")
    L.append("| *(not run)* | yes | yes | yes | **never run** |")
    L.append("")
    L.append(
        "**The fourth corner was not run, so the two main effects are "
        "described from three corners rather than estimated from a complete "
        "2x2.** Any statement about an interaction between replay composition "
        "and adapter architecture is therefore unavailable, and none is made "
        "here. This is a stated limit of the design, not an oversight: the "
        "closeout authorization fixes the config set and no additional arm is "
        "authorized under any outcome.\n")
    L.append(
        "Reading the three corners that exist: comparing `eq_pres_norep` with "
        "`eq_pres` isolates the **replay imitation term** at fixed "
        "architecture; comparing `eq_pres_norep` with `sp_nav_eq` isolates "
        "the **adapter architecture** at fixed loss composition.\n")

    L.append("## Wording locks\n")
    L.append(
        "Locks from the ratified v0.34 verdict apply: no cell-level movement "
        "is described as a repair or as a newly introduced failure, and "
        "per-task quanta are disclosed wherever plasticity is discussed. This "
        "report deliberately confines itself to `Forgetting` and the utility "
        "axis — the two quantities the read-out rule names — precisely "
        "because cell-level plasticity is not resolvable at three seeds.\n")
    L.append(
        "This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable "
        "review**. **STOP** — after E1 and E2 all compute for this paper is "
        "over; the only remaining work is writing.\n")

    OUT_E2.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_E2.relative_to(REPO)}  -> L_eq alone sufficient: "
          f"{sufficient}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e1 = sub.add_parser("e1", help="utility-capability alignment test")
    e1.add_argument("--device", default="mps")
    e2 = sub.add_parser("e2", help="eq_pres_norep attribution read-out")
    e2.add_argument("--c0-csv", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "e1":
        run_e1(resolve_device(args.device))
    else:
        run_e2(args.c0_csv.resolve())


if __name__ == "__main__":
    main()
