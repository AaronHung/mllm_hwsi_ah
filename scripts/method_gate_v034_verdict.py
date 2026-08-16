"""Compute the v0.34 **development gate** verdict
(`docs/method_gate_v034.md` §3), pre-registered BEFORE this script was run
against any v0.34 data.

Inputs (only the dev CSV is new; every comparator is frozen and same-backend
MPS, none re-executed here — §7):
  - runs/v2/<tag>/cl_main_can_main_<tag>.csv  (27 dev units + 9
    instrumentation-only units)
  - runs/v2/<tag>/arbiter/arbiter_summary_*.json  (§2.4 conflict statistics)
  - results/cl_main_can_main_ablA1_s{0,1,2}.csv  (frozen `ours_uniform`, K{1,2})
  - runs/v2/gate_v2_20260816T073200Z/...csv      (frozen `ours_uniform` K=4,
    Gate-v2 comparator completion)
  - results/cl_main_can_main_full_s{0,1,2}.csv   (frozen `distill`, `replay`)
  - runs/v2/util_regen_20260815_mps/...csv       (checksum-passed utility-axis
    columns for `distill`/`ours_uniform`, seeds{0,1,2}, K{1,2})
  - runs/v2/method_gate_v0333_run1/...csv        (frozen `eq_pres`, diagnostic
    precursor — displayed for context, never re-scored)

Criteria are transcribed from §3.2 (A / B-a / B-b / B-c / C) as amended by
Sol item 1 and rider r1. The mechanism link is reported post-hoc with no
numeric early-abort (Sol item 2 / rider r2). Metric definitions are imported
from `scripts/aggregate_results.py` and `scripts/method_gate_v2_verdict.py`
rather than reimplemented, so they cannot silently drift.

用法：
    python scripts/method_gate_v034_verdict.py \
        --dev-csv runs/v2/<tag>/cl_main_can_main_<tag>.csv
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
from verify_v034_bitexact import METRIC_COLS  # noqa: E402
from verify_v034_bitexact import compare as bitexact_compare  # noqa: E402
from verify_v034_bitexact import load_frozen as bitexact_load_frozen  # noqa: E402

GATE_V0333_CSV = (REPO / "runs/v2/method_gate_v0333_run1"
                  / "cl_main_can_main_method_gate_v0333_run1.csv")
GATE_V2_CSV = (REPO / "runs/v2/gate_v2_20260816T073200Z"
               / "cl_main_can_main_gate_v2_20260816T073200Z.csv")
FULL_CSVS = [REPO / f"results/cl_main_can_main_full_s{s}.csv" for s in (0, 1, 2)]
ABLA1_CSVS = [REPO / f"results/cl_main_can_main_ablA1_s{s}.csv" for s in (0, 1, 2)]
UTIL_REGEN_CSV = (REPO / "runs/v2/util_regen_20260815_mps"
                  / "cl_main_can_main_util_regen_20260815_mps.csv")
OUT_MD = REPO / "results/method_gate_v034_dev_verdict.md"
OUT_BITEXACT_MD = REPO / "results/eq_pres_diag_bitexact_check.md"

DEV_CONFIGS = ["proj_distill", "proj_eq_pres", "conflict_eq_pres"]
CANDIDATES = ["proj_eq_pres", "conflict_eq_pres"]     # §3.4: control excluded
SEEDS = [0, 1, 2]
KS = [1, 2, 4]
MAIN_ORDER = ["esca", "lung", "rcc", "brca"]          # stage 1..4
DIAGNOSED_CELLS = [("brca", 2), ("lung", 4)]          # §3.2 B-c

TOL_PLASTICITY = -0.01     # B-a / B-c: dA[t,t] vs ours_uniform
TOL_FORGET_NEW = 0.005     # B-b: dForgetting at K in {2,4}
TOL_REPLAY = 0.005         # C: dForgetting vs replay, all K


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_diffs(df: pd.DataFrame, col: str, a: str, b: str, k: int,
                 extra_key: str | None = None,
                 extra_val: object = None) -> dict[int, float]:
    """seed -> (a - b) for `col` at K=k, restricted to seeds present in both."""
    def sel(method: str) -> pd.DataFrame:
        m = df[(df.method == method) & (df.K == k)]
        if extra_key is not None:
            m = m[m[extra_key] == extra_val]
        return m.set_index("seed")
    ta, tb = sel(a), sel(b)
    return {s: float(ta.loc[s, col] - tb.loc[s, col])
            for s in SEEDS if s in ta.index and s in tb.index}


def fmt_list(vals: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in vals) + "]" if vals else "N/A"


def mean_or_none(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def ci_str(vals: list[float]) -> str:
    if len(vals) < 2:
        return "N/A"
    _, lo, hi = boot_ci_paired(np.array(vals), np.zeros(len(vals)))
    return f"[{lo:+.4f}, {hi:+.4f}]"


# ------------------------------------------------------------- conflict table
def load_arbiter_summaries(arbiter_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(arbiter_dir.glob("arbiter_summary_*.json")):
        rows += json.loads(path.read_text())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Stage t trains on task MAIN_ORDER[t-1]; stage 1 has no memory term and
    # therefore no arbiter record at all, by construction (§2.2).
    df["task"] = df["stage"].map(lambda t: MAIN_ORDER[int(t) - 1])
    return df


def conflict_table(arb: pd.DataFrame, methods: list[str],
                   cells: list[tuple[str, int]] | None) -> list[str]:
    lines = ["| config | task (stage) | K | updates | conflict fraction "
             "[per-seed 0,1,2] | mean cos | mean \\|proj\\|/\\|g_m\\| "
             "(conflicting updates) |",
             "|---|---|---|---|---|---|---|"]
    for m in methods:
        sub_m = arb[arb.method == m]
        if sub_m.empty:
            continue
        keys = cells if cells is not None else sorted(
            {(r.task, int(r.K)) for r in sub_m.itertuples()},
            key=lambda tk: (MAIN_ORDER.index(tk[0]), tk[1]))
        for task, k in keys:
            g = sub_m[(sub_m.task == task) & (sub_m.K == k)]
            if g.empty:
                lines.append(f"| `{m}` | {task} (stage "
                             f"{MAIN_ORDER.index(task) + 1}) | {k} | 0 | N/A "
                             "| N/A | N/A |")
                continue
            per_seed = [float(g[g.seed == s].conflict_fraction.mean())
                        for s in SEEDS if not g[g.seed == s].empty]
            lines.append(
                f"| `{m}` | {task} (stage {MAIN_ORDER.index(task) + 1}) | {k} "
                f"| {int(g.updates.sum())} | {np.mean(per_seed):.4f} "
                f"[{', '.join(f'{v:.4f}' for v in per_seed)}] "
                f"| {g.cos_mean.mean():+.4f} "
                f"| {g.proj_ratio_mean_conflict.mean():.4f} |")
    return lines


# ------------------------------------------------------------- §2.5 gate
def check_instrumentation_batch(dev: pd.DataFrame) -> tuple[bool, list[str]]:
    """§2.5 condition 3: `eq_pres_diag` must reproduce the frozen `eq_pres`
    rows bit-identically, or the ENTIRE batch is discarded."""
    new = dev[dev.method == "eq_pres_diag"]
    lines = ["# `eq_pres_diag` bit-exactness check "
             "(docs/method_gate_v034.md §2.5 condition 3)", ""]
    if new.empty:
        lines += ["**No `eq_pres_diag` rows present** — the "
                  "instrumentation-only batch did not run.", ""]
        return False, lines
    frozen = bitexact_load_frozen("eq_pres", SEEDS)
    frozen = frozen[frozen.seed.isin(SEEDS) & frozen.K.isin(KS)]
    findings, n_bad = bitexact_compare(new, frozen, "eq_pres_diag", "eq_pres")
    ok = n_bad == 0
    lines += [
        f"Rule: exact equality on `{'`, `'.join(METRIC_COLS)}`. "
        "Any mismatch discards the entire batch — no partial admission.", "",
        f"- rows compared: **{len(findings)}**",
        f"- identical: **{len(findings) - n_bad}**",
        f"- mismatched: **{n_bad}**", "",
        ("**ADMITTED — bit-identical to the frozen `eq_pres` rows.** The "
         "conflict statistics from this batch are a direct measurement of the "
         "diagnosed config itself." if ok else
         "**DISCARDED — the batch is not bit-identical, so per §2.5 condition "
         "3 none of it is used.** The verdict falls back to the arbitrated "
         "configs' pre-projection statistics, which approximate but do not "
         "measure `eq_pres`'s own conflict rate (trajectories diverge after "
         "the first projection)."), ""]
    if not ok:
        lines += ["## Mismatched rows", ""]
        lines += [f"- seed={f['seed']} K={f['K']} stage={f['stage']} "
                  f"task_idx={f['task_idx']}: {f['status']} {f['cols']}"
                  for f in findings if f["status"] != "identical"]
        lines.append("")
    return ok, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-csv", type=Path, required=True)
    args = ap.parse_args()
    dev_csv = args.dev_csv.resolve()

    dev = pd.read_csv(dev_csv).drop_duplicates(
        subset=["method", "seed", "K", "stage", "task_idx"], keep="last")
    arbiter_dir = dev_csv.parent / "arbiter"
    arb = load_arbiter_summaries(arbiter_dir)

    # ---- §2.5 validity gate for the instrumentation batch (runs first) ----
    diag_ok, bitexact_lines = check_instrumentation_batch(dev)
    OUT_BITEXACT_MD.write_text("\n".join(bitexact_lines) + "\n")
    print(f"wrote {OUT_BITEXACT_MD.relative_to(REPO)} "
          f"({'ADMITTED' if diag_ok else 'DISCARDED'})")

    # ---- frozen comparators (§7), none rerun ----
    full = load_concat(FULL_CSVS)
    abla1 = load_concat(ABLA1_CSVS)
    gate_v2 = pd.read_csv(GATE_V2_CSV)
    ours_u_raw = pd.concat(
        [abla1[abla1.method == "ours_uniform"],
         gate_v2[(gate_v2.method == "ours_uniform")
                 & (gate_v2.gate_v2_role == "comparator_completion_k4")
                 & (gate_v2.seed.isin(SEEDS))]], ignore_index=True)
    distill_raw = full[full.method == "distill"]
    replay_raw = full[full.method == "replay"]
    eq_pres_raw = pd.read_csv(GATE_V0333_CSV)
    eq_pres_raw = eq_pres_raw[eq_pres_raw.method == "eq_pres"]
    cand_raw = dev[dev.method.isin(DEV_CONFIGS)]

    metrics = pd.concat([per_run_metrics(cand_raw), per_run_metrics(ours_u_raw),
                         per_run_metrics(distill_raw),
                         per_run_metrics(replay_raw),
                         per_run_metrics(eq_pres_raw)], ignore_index=True)
    util = pd.concat([utility_axis_final_old_tasks(cand_raw),
                      utility_axis_final_old_tasks(
                          pd.read_csv(UTIL_REGEN_CSV)),
                      utility_axis_final_old_tasks(eq_pres_raw)],
                     ignore_index=True)
    own = pd.concat([own_time_acc(cand_raw), own_time_acc(ours_u_raw),
                     own_time_acc(eq_pres_raw)], ignore_index=True)

    L: list[str] = []
    L.append("# Method Gate v0.34 — Development Verdict\n")
    L.append(
        "**Track A:** 8/21 freeze conditions all met 2026-08-16 (pilot40 R6 "
        "closed); writing is the remaining work. **Track B:** v0.33 is "
        "archived — the v0.33.3 and Gate v2 verdicts are frozen and `eq_pres` "
        "stays unpromoted; this document evaluates ONLY the v0.34 development "
        "gate (`docs/method_gate_v034.md` §3) for `proj_distill`, "
        "`proj_eq_pres`, `conflict_eq_pres`.\n")
    L.append(
        f"Computed by `scripts/method_gate_v034_verdict.py` against "
        f"`{dev_csv.relative_to(REPO)}`. All comparators are frozen, "
        "same-backend (Mac MPS), and were not rerun (§7). Thresholds and "
        "formulas are transcribed from §3.2, which was committed before any "
        "v0.34 training run — nothing was chosen after seeing this data.\n")
    L.append(
        "**This is a development environment.** The main task order is "
        "already unblinded, so nothing below is confirmatory evidence; "
        "confirmation is reverse order with seeds {0,1,2,3,4} and only after "
        "three-way review (§4).\n")
    L.append(
        "Criteria are **mean-only** 3-seed paired means (§3.2). Per-seed "
        "values and 95% bootstrap CIs (`n_boot=10000`) are shown throughout "
        "as **descriptive disclosure**, never as a pass/fail test.\n")

    verdicts: dict[str, dict[str, bool]] = {c: {} for c in DEV_CONFIGS}

    # =================== A — utility retained ===================
    L.append("## A — Utility retained: Δ`eps_optimal_mass` > 0 vs `distill` "
             "AND `ours_uniform`, at K ∈ {1,2}\n")
    L.append("Aggregation is the frozen §5.1 convention (final stage, old "
             "tasks only, mean within task then macro-average). "
             "`normalized_regret` is shown as secondary descriptive evidence "
             "and is **not** a pass condition.\n")
    L.append("| config | K | Δε-mass vs `ours_uniform` [seeds 0,1,2] | "
             "Δε-mass vs `distill` [0,1,2] | Δregret vs `ours_uniform` "
             "(descriptive) | A (this K) |")
    L.append("|---|---|---|---|---|---|")
    for cfg in DEV_CONFIGS:
        a_ok = True
        for k in (1, 2):
            d_ou = list(paired_diffs(util, "eps_optimal_mass", cfg,
                                     "ours_uniform", k).values())
            d_di = list(paired_diffs(util, "eps_optimal_mass", cfg,
                                     "distill", k).values())
            r_ou = list(paired_diffs(util, "normalized_regret", cfg,
                                     "ours_uniform", k).values())
            m_ou, m_di = mean_or_none(d_ou), mean_or_none(d_di)
            ok = (m_ou is not None and m_di is not None
                  and m_ou > 0 and m_di > 0)
            a_ok = a_ok and ok
            L.append(
                f"| `{cfg}` | {k} | "
                f"{'N/A' if m_ou is None else f'{m_ou:+.4f}'} {fmt_list(d_ou)} "
                f"| {'N/A' if m_di is None else f'{m_di:+.4f}'} "
                f"{fmt_list(d_di)} | "
                f"{'N/A' if not r_ou else f'{np.mean(r_ou):+.4f}'} "
                f"{fmt_list(r_ou)} | {'PASS' if ok else 'FAIL'} |")
        verdicts[cfg]["A"] = a_ok
    L.append("")

    # =================== B — targeted repair + no new failure ===================
    L.append("## B — Targeted repair + no new failure (rider r1 = Sol item 1)\n")
    L.append("### B-a — every `(task, K)` cell: 3-seed paired mean "
             "ΔA[t,t] vs `ours_uniform` ≥ −0.01\n")
    L.append("| config | task | K | quantum (1/n_test) | mean ΔA[t,t] "
             "[seeds 0,1,2] | 95% CI | B-a | diagnosed target |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg in DEV_CONFIGS:
        b_a_ok, b_c_plast_ok = True, True
        for t_idx, task in enumerate(MAIN_ORDER, start=1):
            for k in KS:
                sub = own[own.task_idx == t_idx]
                d = list(paired_diffs(sub, "acc_own", cfg, "ours_uniform",
                                      k).values())
                m = mean_or_none(d)
                ok = m is not None and m >= TOL_PLASTICITY
                b_a_ok = b_a_ok and ok
                is_target = (task, k) in DIAGNOSED_CELLS
                if is_target:
                    b_c_plast_ok = b_c_plast_ok and ok
                n_test = sub[(sub.method == cfg) & (sub.K == k)]
                quantum = (f"{1 / int(n_test.n_test.iloc[0]):.4f} "
                           f"(n={int(n_test.n_test.iloc[0])})"
                           if not n_test.empty else "N/A")
                L.append(
                    f"| `{cfg}` | {task} | {k} | {quantum} | "
                    f"{'N/A' if m is None else f'{m:+.4f}'} {fmt_list(d)} | "
                    f"{ci_str(d)} | {'PASS' if ok else 'FAIL'} | "
                    f"{'**yes**' if is_target else ''} |")
        verdicts[cfg]["B-a"] = b_a_ok
        verdicts[cfg]["B-c-plasticity"] = b_c_plast_ok
    L.append("")

    L.append("### B-b — no new forgetting regression at K=2 and K=4: "
             "ΔForgetting ≤ +0.005 vs `ours_uniform` AND `distill`\n")
    L.append("### B-c — diagnosed target: at K=1, ΔForgetting ≤ 0 vs "
             "`ours_uniform` AND `distill`\n")
    L.append("| config | K | rule | vs `ours_uniform` mean [0,1,2] | "
             "vs `distill` mean [0,1,2] | 95% CI (vs `ours_uniform`) | "
             "verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for cfg in DEV_CONFIGS:
        b_b_ok, b_c_forget_ok = True, True
        for k in KS:
            tol, rule = ((0.0, "≤ 0 (B-c target)") if k == 1
                         else (TOL_FORGET_NEW, "≤ +0.005 (B-b)"))
            d_ou = list(paired_diffs(metrics, "forgetting", cfg,
                                     "ours_uniform", k).values())
            d_di = list(paired_diffs(metrics, "forgetting", cfg, "distill",
                                     k).values())
            m_ou, m_di = mean_or_none(d_ou), mean_or_none(d_di)
            ok = (m_ou is not None and m_di is not None
                  and m_ou <= tol and m_di <= tol)
            if k == 1:
                b_c_forget_ok = ok
            else:
                b_b_ok = b_b_ok and ok
            L.append(
                f"| `{cfg}` | {k} | {rule} | "
                f"{'N/A' if m_ou is None else f'{m_ou:+.4f}'} {fmt_list(d_ou)} "
                f"| {'N/A' if m_di is None else f'{m_di:+.4f}'} "
                f"{fmt_list(d_di)} | {ci_str(d_ou)} | "
                f"{'PASS' if ok else 'FAIL'} |")
        verdicts[cfg]["B-b"] = b_b_ok
        verdicts[cfg]["B-c-forgetting"] = b_c_forget_ok
    L.append("")

    # =================== C — replay safety (Jaccard unconstrained) ==========
    L.append("## C — Replay safety: ΔForgetting vs `replay` ≤ +0.005 at all K "
             "(selection Jaccard carries **no** requirement, by design)\n")
    L.append("| config | K | mean ΔForgetting vs `replay` [0,1,2] | 95% CI | "
             "mean ΔJaccard vs `ours_uniform` (descriptive only) | C (this K) |")
    L.append("|---|---|---|---|---|---|")
    for cfg in DEV_CONFIGS:
        c_ok = True
        for k in KS:
            d = list(paired_diffs(metrics, "forgetting", cfg, "replay",
                                  k).values())
            j = list(paired_diffs(metrics, "jaccard", cfg, "ours_uniform",
                                  k).values())
            m = mean_or_none(d)
            ok = m is not None and m <= TOL_REPLAY
            c_ok = c_ok and ok
            L.append(f"| `{cfg}` | {k} | "
                     f"{'N/A' if m is None else f'{m:+.4f}'} {fmt_list(d)} | "
                     f"{ci_str(d)} | "
                     f"{'N/A' if not j else f'{np.mean(j):+.4f}'} "
                     f"{fmt_list(j)} | {'PASS' if ok else 'FAIL'} |")
        verdicts[cfg]["C"] = c_ok
    L.append("")

    # =================== mechanism link ===================
    L.append("## Mechanism link — gradient conflict (pre-registered "
             "hypothesis, reported post-hoc)\n")
    L.append(
        "Per Sol amendment item 2 / rider r2 there is **no numeric "
        "early-abort and no threshold chosen after observing these values**. "
        "The table exists so the three-way review can judge whether the "
        "gradient-conflict explanation of the Gate-v2 failure is supported.\n")
    if arb.empty:
        L.append("**No arbiter summaries found** — the §2.4 instrumentation "
                 "did not produce output. The mechanism hypothesis cannot be "
                 "evaluated from this run.\n")
    else:
        L.append("### Diagnosed target cells (`brca` K=2, `lung` K=4)\n")
        diag_methods = (["eq_pres_diag"] if diag_ok else []) + DEV_CONFIGS
        L += conflict_table(arb, diag_methods, DIAGNOSED_CELLS)
        L.append("")
        L.append(
            "`eq_pres_diag` is the instrumentation-only replication of the "
            "frozen `eq_pres` run (§2.5): "
            + ("**admitted**, so its rows are a direct measurement of the "
               "config that actually failed Gate v2."
               if diag_ok else
               "**discarded** by the bit-exactness gate, so only the "
               "arbitrated configs' pre-projection statistics are available "
               "— they share `eq_pres`'s memory objective but diverge in "
               "trajectory after the first projection, and are therefore an "
               "approximation, not a counterfactual measurement of "
               "`eq_pres`.") + "\n")
        L.append("### All stages and budgets\n")
        L += conflict_table(arb, diag_methods, None)
        L.append("")
        L.append("Stage 1 never appears: the first task has no buffer and no "
                 "`old_nav`, so there is no memory gradient to conflict with "
                 "(§2.2).\n")

    # =================== overall ===================
    L.append("## Overall development-gate verdict\n")
    L.append("| config | A (utility) | B-a (no new plasticity failure) | "
             "B-b (no new forgetting regression) | B-c (diagnosed targets) | "
             "C (replay safety) | DEV GATE |")
    L.append("|---|---|---|---|---|---|---|")
    overall: dict[str, bool] = {}
    for cfg in DEV_CONFIGS:
        v = verdicts[cfg]
        b_c = v["B-c-plasticity"] and v["B-c-forgetting"]
        passed = all([v["A"], v["B-a"], v["B-b"], b_c, v["C"]])
        overall[cfg] = passed
        L.append(
            f"| `{cfg}` | {'PASS' if v['A'] else 'FAIL'} | "
            f"{'PASS' if v['B-a'] else 'FAIL'} | "
            f"{'PASS' if v['B-b'] else 'FAIL'} | {'PASS' if b_c else 'FAIL'} | "
            f"{'PASS' if v['C'] else 'FAIL'} | "
            f"**{'DEV PASS' if passed else 'DEV FAIL'}** |")
    L.append("")

    # §3.4 winner selection, applied exactly as pre-registered
    L.append("### Winner selection (§3.4, pre-registered before training)\n")
    if overall.get("proj_distill"):
        L.append(
            "> **Disclosure (§3.4 item 5): the generic control "
            "`proj_distill` also passes.** Read together with criterion A "
            "below, this is direct evidence that the utility advantage may "
            "not be attributable to the equivalence objective, which bears "
            "on the novelty positioning in §6. The three-way review decides "
            "what follows; this script does not.\n")
    passing = [c for c in CANDIDATES if overall.get(c)]
    if not passing:
        L.append("No candidate config passes, so there is no winner and no "
                 "confirmation run. Branch: §4 FAIL — archive honestly, ship "
                 "framework-first with `eq_pres` as the diagnostic "
                 "precursor.\n")
        winner = None
    elif len(passing) == 1:
        winner = passing[0]
        L.append(f"Exactly one candidate passes: **`{winner}`**.\n")
    else:
        scores = {}
        for c in passing:
            vals = []
            for k in (1, 2):
                for comp in ("ours_uniform", "distill"):
                    vals += list(paired_diffs(util, "eps_optimal_mass", c,
                                              comp, k).values())
            scores[c] = float(np.mean(vals)) if vals else float("-inf")
        winner = max(scores, key=lambda c: (scores[c], c == "proj_eq_pres"))
        L.append("Both candidates pass; tie-break rule 2 (larger criterion-A "
                 "effect, mean over K ∈ {1,2} against both comparators): "
                 + ", ".join(f"`{c}` {scores[c]:+.4f}" for c in passing)
                 + f" -> winner **`{winner}`**.\n")

    L.append("## Branch and STOP\n")
    if winner:
        L.append(
            f"**Development gate PASSES with `{winner}`.** Per §5 this "
            "document is where v0.34 stops: the confirmation run (reverse "
            "order, exact seeds {0,1,2,3,4}, K ∈ {1,2,4}) and the pilot40 "
            "winner row require **three-way Aaron+Sol+Fable review of this "
            "verdict first**. No confirmation run, no promotion, and no "
            "paper-claim change has been made.\n")
    else:
        L.append(
            "**Development gate FAILS.** Per §4 the honest branch is to "
            "archive: the paper ships framework-first with `eq_pres` as the "
            "diagnostic precursor, and the three v0.34 configs are reported "
            "as tested-negative with their full per-seed tables, exactly as "
            "`ia_samp`/`ia_ep` are. **STOP for three-way review** — no "
            "further method compute is opened, and per §0 there is no v0.35 "
            "for this ICASSP submission.\n")
    L.append(
        "This is a **Cursor analysis artifact awaiting joint Aaron+Sol+Fable "
        "review**. No threshold was altered, no config was added, and no "
        "v0.33 cell was re-scored in producing it.\n")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    for cfg in DEV_CONFIGS:
        print(f"  {cfg}: {'DEV PASS' if overall[cfg] else 'DEV FAIL'}")


if __name__ == "__main__":
    main()
