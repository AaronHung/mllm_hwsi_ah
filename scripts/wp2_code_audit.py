"""WP2: fast CPU static audit for device/dtype portability.

The audit is intentionally text/AST based and does not import the experiment
modules or require a dataset.  It reports findings instead of editing code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def source_without_comments(text: str) -> str:
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path,
                    default=Path("runs/v2/wp2_code_audit"))
    ap.add_argument("--device", default="cpu",
                    help="accepted for common CLI; static audit is CPU-only")
    args = ap.parse_args()

    nav_dir = args.repo / "nav"
    scripts_dir = args.repo / "scripts"
    rows: list[dict[str, object]] = []
    hardcoded = re.compile(
        r"""(?:\.to|torch\.device)\(\s*["'](?:cpu|cuda|mps)["']""")
    for path in sorted(list(nav_dir.glob("*.py")) +
                       list(scripts_dir.glob("*.py"))):
        text = path.read_text()
        code = source_without_comments(text)
        findings = []
        if path.name != "wp2_code_audit.py" and \
                re.search(r"\bfloat64\b|\.double\s*\(", code):
            findings.append("float64/double")
        if hardcoded.search(code):
            findings.append("hardcoded torch device")
        if path.parent == scripts_dir and "ArgumentParser" in text and \
                "--device" not in text and "add_device_argument" not in text:
            findings.append("entry point lacks --device")
        rows.append(dict(
            path=str(path.relative_to(args.repo)),
            status="PASS" if not findings else "REVIEW",
            findings="; ".join(findings),
            resolved_device="cpu"))

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "audit.csv", index=False)
    lines = ["# WP2 code audit (CPU)", ""]
    for row in rows:
        suffix = row["findings"] or "none"
        lines.append(f"- `{row['path']}`: **{row['status']}** — {suffix}")
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(
        {"resolved_device": "cpu", "scope": "nav/*.py + scripts/*.py"},
        indent=2))
    review_count = int((result["status"] == "REVIEW").sum())
    print(f"WP2 done -> {args.output_dir}; review_items={review_count}")


if __name__ == "__main__":
    main()
