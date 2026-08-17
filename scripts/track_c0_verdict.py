"""Compute the Track C0 verdict (`docs/track_c0.md` §7), pre-registered
BEFORE this script was run against any C0 data.

Inputs (only the C0 CSV is new; every comparator is frozen, same-backend MPS,
and is not rerun — §3):
  - runs/v2/<tag>/cl_main_can_main_<tag>.csv   (12 units: sp_nav, sp_nav_eq)
  - results/cl_main_can_main_full_s{0,1,2}.csv (frozen `seqft`, `distill`)
  - runs/v2/util_regen_20260815_mps/...csv     (checksum-passed utility-axis
    columns for `distill`, seeds{0,1,2}, K{1,2})
  - runs/v2/method_gate_v0333_run1/...csv      (frozen `eq_pres`, context only)

Criteria are transcribed from §7 and the reading fixed in §6.4: c1 is judged
per task pooled over K, c2 per K pooled over tasks, and there is NO per-cell
veto — three seeds cannot resolve cell-level plasticity, which is the direct
lesson of v0.34. Metric definitions are imported from
`scripts/aggregate_results.py` and `scripts/method_gate_v2_verdict.py` rather
than reimplemented.

用法：
    python scripts/track_c0_verdict.py \
        --c0-csv runs/v2/<tag>/cl_main_can_main_<tag>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from aggregate_results import boot_ci_paired, per_run_metrics  # noqa: E402
from method_gate_v2_verdict import (own_time_acc,  # noqa: E402
                                    utility_axis_final_old_tasks)
from run_track_c0 import parameter_budget  # noqa: E402

FULL_CSVS = [REPO / f"results/cl_main_can_main_full_s{s}.csv" for s in (0, 1, 2)]
UTIL_REGEN_CSV = (REPO / "runs/v2/util_regen_20260815_mps"
                  / "cl_main_can_main_util_regen_20260815_mps.csv")
OUT_MD = REPO / "results/track_c0_verdict.md"

CONFIGS = ["sp_nav", "sp_nav_eq"]
SEEDS = [0, 1, 2]
KS = [1, 2]
MAIN_ORDER = ["esca", "lung", "rcc", "brca"]
TOL_C1 = -0.01          # own-time A[t,t] vs seqft
TOL_C2 = 0.0            # Forgetting vs distill
ADAPTER_PCT_MAX = 5.0   # c4

ORACLE_DISCLOSURE = (
    "**C0 assumes ORACLE TASK IDENTITY at both train and test time** "
    "(`docs/track_c0.md` §1): the task index selects which adapter is active "
    "during training and during every evaluation, including evaluation of old "
    "tasks at later stages. This is a best-case assumption — a "
    "task-incremental setting with known task identity is strictly easier "
    "than the class-incremental or task-agnostic settings. **C0 makes no "
    "claim that survives without it.** Inferring task identity (a router) is "
    "future work and is explicitly not part of any C0 claim.")


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired(df: pd.DataFrame, col: str, a: str, b: str,
           k: int | None = None) -> dict[int, float]:
    """seed -> (a - b) for `col`, optionally restricted to one K."""
    def sel(m: str) -> pd.DataFrame:
        s = df[df.method == m]
        return s[s.K == k] if k is not None else s
    ta, tb = sel(a), sel(b)
    out: dict[int, float] = {}
    for s in SEEDS:
        va, vb = ta[ta.seed == s][col], tb[tb.seed == s][col]
        if len(va) and len(vb):
            out[s] = float(va.mean() - vb.mean())
    return out


def fmt(vals: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in vals) + "]" if vals else "N/A"


def ci(vals: list[float]) -> str:
    if len(vals) < 2:
        return "N/A"
    _, lo, hi = boot_ci_paired(np.array(vals), np.zeros(len(vals)))
    return f"[{lo:+.4f}, {hi:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0-csv", type=Path, required=True)
    args = ap.parse_args()
    c0_csv = args.c0_csv.resolve()

    c0 = pd.read_csv(c0_csv).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")
    full = load_concat(FULL_CSVS)
    seqft_raw = full[full.method == "seqft"]
    distill_raw = full[full.method == "distill"]

    metrics = pd.concat([per_run_metrics(c0), per_run_metrics(seqft_raw),
                         per_run_metrics(distill_raw)], ignore_index=True)
    own = pd.concat([own_time_acc(c0), own_time_acc(seqft_raw)],
                    ignore_index=True)
    util = pd.concat([utility_axis_final_old_tasks(c0),
                      utility_axis_final_old_tasks(pd.read_csv(UTIL_REGEN_CSV))],
                     ignore_index=True)
    budget = parameter_budget()
    n_updates = json.loads((c0_csv.parent / "checkpoints.json").read_text())

    L: list[str] = []
    L.append("# Track C0 — Verdict (minimal stable-plastic navigator)\n")
    L.append(
        "**Track A:** frozen; writing proceeds in parallel and was never "
        "blocked by C0. **Track B/C:** Track B is closed and archived "
        "(`docs/method_gate_v034.md` §0/§4, no v0.35); this document reports "
        "the Track C0 screening experiment only.\n")
    L.append(ORACLE_DISCLOSURE + "\n")
    L.append(
        f"Computed by `scripts/track_c0_verdict.py` against "
        f"`{c0_csv.relative_to(REPO)}` (12 units). `seqft`, `distill` and "
        "`eq_pres` are frozen, same-backend (Mac MPS) comparators and were "
        "not rerun. Thresholds are transcribed from `docs/track_c0.md` §7, "
        "committed before any C0 training run.\n")
    L.append(
        "**C0 is a screening experiment, not a promotion gate.** Passing it "
        "licenses proposing C1 to the three-way review and nothing more. "
        "Criteria are aggregate-level with **no per-cell veto** (§6.4/§7): "
        "three seeds cannot resolve cell-level plasticity differences, which "
        "is the direct lesson of v0.34, where cell means were smaller than "
        "the per-item quantum `1/n_test` with per-seed spread of several "
        "quanta. Per-seed values and 95% bootstrap CIs are **descriptive "
        "disclosure**, never a pass/fail test.\n")

    verdicts: dict[str, dict[str, bool]] = {c: {} for c in CONFIGS}

    # ---------------- c1 plasticity ----------------
    L.append("## c1 — Plasticity: own-time `A[t,t]` ≥ `seqft` − 0.01 for every "
             "task (3-seed paired mean, pooled over K ∈ {1,2})\n")
    L.append("| config | task | mean ΔA[t,t] vs `seqft` [seeds 0,1,2] | "
             "95% CI | c1 (this task) |")
    L.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        ok_all = True
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            sub = own[(own.task_idx == t_idx) & (own.K.isin(KS))]
            d = list(paired(sub, "acc_own", cfg, "seqft").values())
            m = float(np.mean(d)) if d else None
            ok = m is not None and m >= TOL_C1
            ok_all = ok_all and ok
            L.append(f"| `{cfg}` | {task} | "
                     f"{'N/A' if m is None else f'{m:+.4f}'} {fmt(d)} | "
                     f"{ci(d)} | {'PASS' if ok else 'FAIL'} |")
        verdicts[cfg]["c1"] = ok_all
    L.append("")
    L.append("Per-`(task, K)` breakdown, **descriptive only — no veto power** "
             "(§6.4):\n")
    L.append("| config | task | K | mean ΔA[t,t] vs `seqft` [0,1,2] |")
    L.append("|---|---|---|---|")
    for cfg in CONFIGS:
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            for k in KS:
                sub = own[own.task_idx == t_idx]
                d = list(paired(sub, "acc_own", cfg, "seqft", k).values())
                m = float(np.mean(d)) if d else None
                L.append(f"| `{cfg}` | {task} | {k} | "
                         f"{'N/A' if m is None else f'{m:+.4f}'} {fmt(d)} |")
    L.append("")

    # ---------------- c2 stability ----------------
    L.append("## c2 — Stability: `Forgetting` ≤ `distill` at every K "
             "(3-seed paired mean)\n")
    L.append("| config | K | mean ΔForgetting vs `distill` [seeds 0,1,2] | "
             "95% CI | c2 (this K) |")
    L.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        ok_all = True
        for k in KS:
            d = list(paired(metrics, "forgetting", cfg, "distill", k).values())
            m = float(np.mean(d)) if d else None
            ok = m is not None and m <= TOL_C2
            ok_all = ok_all and ok
            L.append(f"| `{cfg}` | {k} | "
                     f"{'N/A' if m is None else f'{m:+.4f}'} {fmt(d)} | "
                     f"{ci(d)} | {'PASS' if ok else 'FAIL'} |")
        verdicts[cfg]["c2"] = ok_all
    L.append("")

    # ---------------- c3 utility (sp_nav_eq only) ----------------
    L.append("## c3 — Utility (`sp_nav_eq` only): Δ`eps_optimal_mass` > 0 vs "
             "`distill`\n")
    L.append("Aggregated by the frozen §5.1 convention (final stage, old tasks "
             "only, mean within task then macro-average across old tasks). "
             "`normalized_regret` is secondary descriptive evidence.\n")
    L.append("| config | K | mean Δε-mass vs `distill` [0,1,2] | 95% CI | "
             "mean Δregret (descriptive) | c3 (this K) |")
    L.append("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        ok_all = True
        for k in KS:
            d = list(paired(util, "eps_optimal_mass", cfg, "distill", k).values())
            r = list(paired(util, "normalized_regret", cfg, "distill", k).values())
            m = float(np.mean(d)) if d else None
            ok = m is not None and m > 0
            ok_all = ok_all and ok
            L.append(f"| `{cfg}` | {k} | "
                     f"{'N/A' if m is None else f'{m:+.4f}'} {fmt(d)} | "
                     f"{ci(d)} | "
                     f"{'N/A' if not r else f'{np.mean(r):+.4f}'} {fmt(r)} | "
                     f"{'PASS' if ok else 'FAIL'} |")
        # c3 applies to sp_nav_eq only; sp_nav's row is reported for contrast.
        verdicts[cfg]["c3"] = ok_all if cfg == "sp_nav_eq" else None
    L.append("")
    L.append("`sp_nav` has no `L_eq` term, so its row above is **contrast, not "
             "a criterion** — c3 applies to `sp_nav_eq` only (§7).\n")

    # ---------------- c4 disclosure ----------------
    pct = budget["adapter_pct_of_core_per_task"]
    pct_all = budget["adapter_pct_of_core_all_tasks"]
    c4_ok = pct_all < ADAPTER_PCT_MAX
    L.append("## c4 — Disclosure: parameter and update counts\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| shared core parameters | {budget['shared_core_params']:,} |")
    L.append(f"| FiLM adapter parameters per task | "
             f"{budget['adapter_params_per_task']:,} (**{pct}%** of core) |")
    L.append(f"| FiLM adapter parameters, all {budget['n_tasks']} tasks | "
             f"{budget['adapter_params_all_tasks']:,} (**{pct_all}%** of core) |")
    L.append(f"| c4 bound | adapters < {ADAPTER_PCT_MAX}% of shared core — "
             f"**{'PASS' if c4_ok else 'FAIL'}** |")
    L.append(f"| training units completed | "
             f"{len(n_updates.get('completed', []))} / 12 |")
    L.append(f"| per-unit wall clock (last unit) | "
             f"{n_updates.get('last_unit', {}).get('seconds', 'N/A')} s |")
    L.append("")
    L.append("Update counts: every config sees the same number of optimizer "
             "steps as any other method in this repo — `nav_epochs` (10) x "
             "|steps| per stage — because the adapters change *which* "
             "parameters move, not how often. The shared core additionally "
             "moves at one tenth of the adapter learning rate at every one of "
             "those steps (§2.3).\n")
    for cfg in CONFIGS:
        verdicts[cfg]["c4"] = c4_ok

    # ---------------- overall ----------------
    L.append("## Overall C0 verdict\n")
    L.append("| config | c1 (plasticity) | c2 (stability) | c3 (utility) | "
             "c4 (disclosure) | C0 |")
    L.append("|---|---|---|---|---|---|")
    overall: dict[str, bool] = {}
    for cfg in CONFIGS:
        v = verdicts[cfg]
        applicable = [v["c1"], v["c2"], v["c4"]]
        if v["c3"] is not None:
            applicable.append(v["c3"])
        passed = all(applicable)
        overall[cfg] = passed
        c3_cell = "n/a" if v["c3"] is None else ("PASS" if v["c3"] else "FAIL")
        L.append(f"| `{cfg}` | {'PASS' if v['c1'] else 'FAIL'} | "
                 f"{'PASS' if v['c2'] else 'FAIL'} | {c3_cell} | "
                 f"{'PASS' if v['c4'] else 'FAIL'} | "
                 f"**{'C0 PASS' if passed else 'C0 FAIL'}** |")
    L.append("")

    L.append("## Branch and STOP\n")
    if any(overall.values()):
        winners = ", ".join(f"`{c}`" for c in CONFIGS if overall[c])
        L.append(
            f"**C0 passes for {winners}.** Per `docs/track_c0.md` §7 this is "
            "**not a promotion**: it licenses *proposing* C1 to the three-way "
            "review and nothing more. No C1 has been designed, no router "
            "exists, and no confirmation run has been started.\n")
    else:
        L.append(
            "**C0 does not pass.** Per §9 the branch is to return to Track A. "
            "No C0.1, no router, no confirmation runs.\n")
    L.append(
        "**STOP for three-way review** (§8). The architectural hypothesis was "
        "screened at the smallest configuration that can answer it, under the "
        "oracle-task-identity assumption disclosed above.\n")
    L.append(
        "Wording locks inherited from the ratified v0.34 verdict apply to this "
        "document: cell-level plasticity differences at three seeds are "
        "reported as falling within seed variability and are never described "
        "as a repair or a newly introduced failure; the v0.34 "
        "gradient-conflict measurement is a **null result**, not an "
        "unconfirmed one.\n")
    L.append(
        "This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable "
        "review**. No threshold was altered and no comparator was rerun in "
        "producing it.\n")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    for cfg in CONFIGS:
        print(f"  {cfg}: {'C0 PASS' if overall[cfg] else 'C0 FAIL'}")


if __name__ == "__main__":
    main()
