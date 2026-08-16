"""Bit-exactness checker for the v0.34 method track.

Two uses, one rule:

1. **Pre-launch regression** (`docs/method_gate_v034.md` §2.6). The arbiter
   changes how gradients are assembled, so before the development gate runs,
   frozen configs are replayed through the NEW code and must reproduce the
   frozen rows exactly. `eq_pres` covers the `L_eq` branch and `ours_uniform`
   covers the old-policy-KL branch.
2. **Instrumentation-batch validity condition** (§2.5 condition 3). The
   `eq_pres_diag` batch is admitted as a conflict diagnostic ONLY if it
   reproduces the frozen `eq_pres` rows bit-identically. Any mismatch
   discards the entire batch.

Stricter than the `util_regen` `|Δ| <= 0.001` checksum on purpose: here the
claim is "the update path was not touched at all", and the honest test of
that claim is exact equality, not a tolerance.

用法：
    python scripts/verify_v034_bitexact.py \
        --new-csv runs/v2/<tag>/cl_main_can_main_<tag>.csv \
        --map eq_pres=eq_pres --map ours_uniform=ours_uniform \
        --seeds 0 --budgets 1 \
        --title "v0.34 pre-launch bit-exactness regression" \
        --out results/method_gate_v034_bitexact_regression.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Every metric column a result row carries.  wall_s and the provenance
# columns are excluded: they are recording metadata, not model output.
METRIC_COLS = ["bal_acc", "acc", "jaccard", "action_kl", "sel_utility",
               "eps_optimal_mass", "normalized_regret", "random_ref",
               "n_test"]
KEY_COLS = ["seed", "K", "stage", "task_idx"]

GATE_V0333_CSV = (REPO / "runs/v2/method_gate_v0333_run1"
                  / "cl_main_can_main_method_gate_v0333_run1.csv")
GATE_V2_CSV = (REPO / "runs/v2/gate_v2_20260816T073200Z"
               / "cl_main_can_main_gate_v2_20260816T073200Z.csv")
UTIL_REGEN_CSV = (REPO / "runs/v2/util_regen_20260815_mps"
                  / "cl_main_can_main_util_regen_20260815_mps.csv")
UTIL_COLS = ["eps_optimal_mass", "normalized_regret"]


def backfill_utility(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Fill the two utility columns from the checksum-passed `util_regen` run.

    The frozen Protocol-v1 CSVs predate `eps_optimal_mass`/`normalized_regret`
    entirely — for `ours_uniform`/`distill` those cells are literally absent,
    which is the whole reason the v0.33.2 §5.2 backfill exists. Comparing
    against NaN would report a spurious mismatch; comparing against the
    already-checksum-verified MPS regeneration is the same source every other
    analysis in this repo uses.
    """
    if not UTIL_REGEN_CSV.exists():
        return df
    reg = pd.read_csv(UTIL_REGEN_CSV)
    reg = reg[reg["method"] == method][KEY_COLS + UTIL_COLS]
    if reg.empty:
        return df
    for col in UTIL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    merged = df.merge(reg, on=KEY_COLS, how="left", suffixes=("", "_reg"))
    for col in UTIL_COLS:
        merged[col] = merged[col].fillna(merged[f"{col}_reg"])
    return merged.drop(columns=[f"{c}_reg" for c in UTIL_COLS])


def load_frozen(method: str, seeds: list[int]) -> pd.DataFrame:
    """Frozen, same-backend (MPS) source of truth for one method.

    Provenance is spelled out in docs/method_gate_v034.md §7; the two
    subtleties are `ours_uniform` at K=4 (Protocol-v1 never ran it, so it
    comes from the Gate-v2 comparator-completion batch) and the utility
    columns (see backfill_utility).
    """
    if method in ("eq_pres", "ia_samp", "ia_ep"):
        df = pd.read_csv(GATE_V0333_CSV)
    elif method in ("distill", "replay", "seqft", "ewc", "lwf", "ours",
                    "joint"):
        df = pd.concat([pd.read_csv(REPO / "results"
                                    / f"cl_main_can_main_full_s{s}.csv")
                        for s in seeds], ignore_index=True)
    elif method == "ours_uniform":
        abl = pd.concat([pd.read_csv(REPO / "results"
                                     / f"cl_main_can_main_ablA1_s{s}.csv")
                         for s in seeds], ignore_index=True)
        v2 = pd.read_csv(GATE_V2_CSV)
        v2 = v2[v2["gate_v2_role"] == "comparator_completion_k4"]
        df = pd.concat([abl, v2], ignore_index=True)
    else:
        raise ValueError(f"no frozen source registered for method {method!r}")
    return backfill_utility(df[df["method"] == method].copy(), method)


def compare(new: pd.DataFrame, frozen: pd.DataFrame,
            new_method: str, frozen_method: str
            ) -> tuple[list[dict], int, dict[str, int]]:
    """Row-by-row exact comparison, keyed on (seed, K, stage, task_idx).

    A column whose frozen value is still NaN after `backfill_utility` is
    **not comparable** — the frozen row predates that column and no
    checksum-verified backfill exists for it. Those are counted and disclosed
    separately rather than being silently scored as passes.
    """
    merged = new.merge(frozen, on=KEY_COLS, suffixes=("_new", "_frozen"),
                       how="left", indicator=True)
    findings: list[dict] = []
    not_comparable: dict[str, int] = {}
    for _, row in merged.iterrows():
        entry = {"method_new": new_method, "method_frozen": frozen_method,
                 **{c: row[c] for c in KEY_COLS}}
        if row["_merge"] != "both":
            entry["status"] = "MISSING IN FROZEN SOURCE"
            entry["cols"] = ""
            findings.append(entry)
            continue
        bad = []
        for c in METRIC_COLS:
            frozen_val = row[f"{c}_frozen"]
            if pd.isna(frozen_val):
                not_comparable[c] = not_comparable.get(c, 0) + 1
                continue
            if float(row[f"{c}_new"]) != float(frozen_val):
                bad.append(c)
        entry["status"] = "identical" if not bad else "MISMATCH"
        entry["cols"] = ", ".join(
            f"{c}: {row[f'{c}_new']} vs {row[f'{c}_frozen']}" for c in bad)
        findings.append(entry)
    n_bad = sum(1 for f in findings if f["status"] != "identical")
    return findings, n_bad, not_comparable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-csv", required=True, type=Path)
    ap.add_argument("--map", action="append", required=True,
                    metavar="NEW=FROZEN",
                    help="method key in --new-csv = frozen method key to "
                         "compare it against (repeatable)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1])
    ap.add_argument("--title", default="v0.34 bit-exactness check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.new_csv = args.new_csv.resolve()

    new_all = pd.read_csv(args.new_csv)
    lines = [f"# {args.title}", "",
             f"Source (new run): `{args.new_csv.relative_to(REPO)}`  ",
             f"Seeds: {args.seeds} | K: {args.budgets}  ",
             "Rule: **exact equality** on "
             f"`{'`, `'.join(METRIC_COLS)}`, keyed on "
             f"`{'`, `'.join(KEY_COLS)}`. This is stricter than the "
             "`util_regen` `|Δ| <= 0.001` checksum by design "
             "(docs/method_gate_v034.md §2.5/§2.6).", ""]

    total_rows = 0
    total_bad = 0
    table = ["| new method | frozen method | rows compared | identical | "
             "mismatched | verdict |", "|---|---|---|---|---|---|"]
    details: list[str] = []
    gap_notes: list[str] = []
    for pair in args.map:
        new_method, frozen_method = pair.split("=", 1)
        new = new_all[(new_all["method"] == new_method)
                      & (new_all["seed"].isin(args.seeds))
                      & (new_all["K"].isin(args.budgets))].copy()
        if new.empty:
            table.append(f"| `{new_method}` | `{frozen_method}` | 0 | 0 | 0 | "
                         "**NO ROWS FOUND** |")
            total_bad += 1
            continue
        frozen = load_frozen(frozen_method, args.seeds)
        frozen = frozen[frozen["K"].isin(args.budgets)
                        & frozen["seed"].isin(args.seeds)]
        findings, n_bad, gaps = compare(new, frozen, new_method,
                                        frozen_method)
        total_rows += len(findings)
        total_bad += n_bad
        verdict = "**PASS (bit-identical)**" if n_bad == 0 else "**FAIL**"
        table.append(f"| `{new_method}` | `{frozen_method}` | {len(findings)} "
                     f"| {len(findings) - n_bad} | {n_bad} | {verdict} |")
        if gaps:
            gap_notes.append(
                f"- `{new_method}` vs `{frozen_method}`: "
                + "; ".join(f"`{c}` not comparable on {n} row(s)"
                            for c, n in sorted(gaps.items()))
                + " — the frozen row predates that column and no "
                  "checksum-verified backfill covers this cell. Excluded "
                  "from the pass/fail count, disclosed here.")
        for f in findings:
            if f["status"] != "identical":
                details.append(
                    f"- `{f['method_new']}` seed={f['seed']} K={f['K']} "
                    f"stage={f['stage']} task_idx={f['task_idx']}: "
                    f"{f['status']} {f['cols']}")

    lines += table + [""]
    if gap_notes:
        lines += ["## Columns not comparable", ""] + gap_notes + [""]
    if details:
        lines += ["## Mismatches", ""] + details + [""]
    ok = total_bad == 0
    lines += [
        "## Verdict", "",
        (f"**PASS — {total_rows}/{total_rows} rows bit-identical.**"
         if ok else
         f"**FAIL — {total_bad} of {total_rows} rows differ.**"), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(table))
    print(f"-> {args.out}")
    if not ok:
        print("BIT-EXACTNESS CHECK FAILED", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
