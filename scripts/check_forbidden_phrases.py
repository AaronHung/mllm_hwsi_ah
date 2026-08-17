"""v0.33 carry-over c6: pre-freeze forbidden-phrase gate.

Scans paper-facing files for the claim-level violations listed in
`docs/research_contract_v0.292.md` ("Forbidden changes") and
`docs/method_gate_v033.md` (no universal-superiority claim for any gate
config). A match is only a FAIL if there is no negation cue nearby (e.g.
"not treated as an upper bound" is compliant); every match is still printed
for a human audit trail. Exit code is 1 iff at least one uncompensated match
is found in a scanned file.

Usage:
    python scripts/check_forbidden_phrases.py                  # default files
    python scripts/check_forbidden_phrases.py path/to/other.tex
"""

from __future__ import annotations

import argparse
import bisect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_FILES = [
    REPO / "paper" / "main.tex",
    REPO / "docs" / "slides_monthly.md",
    REPO / "docs" / "vln_survey.md",
]

# (pattern, reason) — patterns are case-insensitive, matched per line.
FORBIDDEN = [
    (r"upper[\s-]?bound", "joint must never be called an upper bound "
     "(research_contract_v0.292.md forbidden changes)"),
    (r"\bceiling\b", "joint must not be framed as a ceiling/headroom"),
    (r"\bheadroom\b", "joint must not be framed as a ceiling/headroom"),
    (r"universal(ly)?\s+(winner|best|superior)",
     "no method may be claimed to win universally"),
    (r"\bwins?\s+universally\b",
     "no method may be claimed to win universally"),
    (r"stability[\s\u2013\u2014-]plasticity",
     "science_freeze_memo_20260815.md: WP3 (main-order, 2/8 cells) does not "
     "license this phrase anywhere; use behavior-fidelity / "
     "capability-retention trade-off instead"),
    (r"\bexact[\s-]action\s+preservation\b",
     "distill is old-policy/policy-fidelity distillation, not exact-action "
     "preservation (docs/audit_20260815.md)"),
    (r"masking\s+(?:is\s+)?confirmed",
     "pilot40 capability wording lock (docs/method_gate_v033.md §10.7, "
     "Gate v2 red-team ruling): report as 'unresolved in this small "
     "two-task validation setting; consistent with masking, not validated' "
     "— never 'masking confirmed' (wide-CI-crossing-zero result overclaims "
     "confirmation)"),
    # v0.34 wording locks (Fable's ratification of the DEV FAIL verdict,
    # 2026-08-17). At 3 seeds with per-task n=76-95, cell-level plasticity
    # means are comparable to the per-item quantum and to seed spread, so
    # narrating them as repairs or new failures is over-reading noise.
    (r"\brepair(?:ed|s|ing)?\s+(?:the\s+)?(?:`?brca`?|`?lung`?|"
     r"(?:diagnosed|target(?:ed)?)\s+cells?)",
     "v0.34 wording lock: cell-level plasticity differences at 3 seeds fall "
     "within seed variability and must not be described as a repair; correct "
     "phrasing is 'cell-level plasticity differences fall within seed "
     "variability; no configuration met the pre-registered targets'"),
    (r"introduc(?:ed|es|ing)\s+a\s+new\s+(?:plasticity\s+)?"
     r"(?:failure|regression|hole)",
     "v0.34 wording lock: same reason in the other direction — a cell that "
     "crosses a threshold at 3 seeds has not been shown to be a newly "
     "introduced failure"),
    (r"gradient[\s-]conflict\s+(?:explanation\s+)?(?:is\s+)?unconfirmed",
     "v0.34 mechanism lock: the conflict measurement is a NULL RESULT "
     "(conflict fraction 0.458-0.504, mean cos ~0, target cells "
     "indistinguishable from healthy ones, projection does not change the "
     "conflict rate), not an 'unconfirmed' hypothesis"),
]

# A match preceded by one of these cues (within NEGATION_WINDOW chars on the
# same line) is treated as a compliant negation rather than a violation.
NEGATION_CUES = (
    "not ", "n't ", "never ", "no ", "without ", "isn't ", "aren't ",
    "cannot ", "can't ", "does not ", "do not ", "not an ", "not treated",
)
NEGATION_WINDOW = 60


def has_negation_cue(flat: str, match_start: int) -> bool:
    # Negation is checked across the flattened (line-joined) text so a
    # wrapped sentence like "is not treated as an\nupper bound." is not
    # flagged as a false positive.
    window = flat[max(0, match_start - NEGATION_WINDOW):match_start].lower()
    return any(cue in window for cue in NEGATION_CUES)


def scan_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (ok_lines, fail_lines) report strings for one file."""
    ok, fail = [], []
    if not path.exists():
        return ok, fail
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    # Join with a single space (not newline) so negation lookback can cross
    # a line-wrapped sentence; track each joined-line's start offset so a
    # match can still be attributed to the right source line number.
    line_starts: list[int] = []
    flat_parts: list[str] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        flat_parts.append(line)
        pos += len(line) + 1
    flat = " ".join(flat_parts)

    for pattern, reason in FORBIDDEN:
        for m in re.finditer(pattern, flat, flags=re.IGNORECASE):
            lineno = bisect.bisect_right(line_starts, m.start())
            source_line = lines[lineno - 1].strip()
            entry = (f"{path.relative_to(REPO)}:{lineno}: `{m.group(0)}` — "
                     f"{reason}\n    > {source_line}")
            if has_negation_cue(flat, m.start()):
                ok.append(entry)
            else:
                fail.append(entry)
    return ok, fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path,
                    help="files to scan (default: paper + paper-facing docs)")
    args = ap.parse_args()

    targets = args.files or DEFAULT_FILES
    all_ok: list[str] = []
    all_fail: list[str] = []
    for path in targets:
        ok, fail = scan_file(Path(path))
        all_ok.extend(ok)
        all_fail.extend(fail)

    if all_ok:
        print(f"[forbidden-phrases] {len(all_ok)} compliant negated mention(s):")
        for entry in all_ok:
            print(f"  OK   {entry}")
    if all_fail:
        print(f"[forbidden-phrases] {len(all_fail)} FAIL(s):")
        for entry in all_fail:
            print(f"  FAIL {entry}")
        print(f"\n[forbidden-phrases] GATE FAILED: {len(all_fail)} "
              f"uncompensated forbidden phrase(s).")
        return 1
    print(f"[forbidden-phrases] GATE PASSED: 0 uncompensated forbidden "
          f"phrases across {len(targets)} file(s) "
          f"({len(all_ok)} compliant negated mentions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
