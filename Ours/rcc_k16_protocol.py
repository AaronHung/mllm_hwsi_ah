from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


PROMPTS = (
    (
        "P0_open",
        "This whole-slide image is from a renal tumor. What is the most likely "
        "histologic subtype? Answer with one diagnosis only.",
    ),
    (
        "P1_clear_first",
        "This whole-slide image is from a renal tumor. Classify it as either "
        "clear cell renal cell carcinoma or papillary renal cell carcinoma. "
        "Answer with one diagnosis only.",
    ),
    (
        "P2_papillary_first",
        "This whole-slide image is from a renal tumor. Classify it as either "
        "papillary renal cell carcinoma or clear cell renal cell carcinoma. "
        "Answer with one diagnosis only.",
    ),
)

PROTOCOL_VERSION = "rcc-k16-three-prompt-v1"

TARGET_CLASSES = ("KIRC", "KIRP")
PARSED_CLASSES = ("KIRC", "KIRP", "OTHER", "AMBIGUOUS")

_KIRC_PATTERN = re.compile(r"\b(?:clear[-\s]+cell|conventional|kirc)\b", re.I)
_KIRP_PATTERN = re.compile(r"\b(?:papillary|kirp)\b", re.I)


def parse_answer(answer: str) -> str:
    text = answer.strip()
    has_kirc = bool(_KIRC_PATTERN.search(text))
    has_kirp = bool(_KIRP_PATTERN.search(text))
    if has_kirc and has_kirp:
        return "AMBIGUOUS"
    if has_kirc:
        return "KIRC"
    if has_kirp:
        return "KIRP"
    return "OTHER"


def slide_id_from_filename(filename: str) -> str:
    return Path(filename).name.split(".", maxsplit=1)[0]


def load_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"class", "filename"}.issubset(fields):
            raise ValueError("selection CSV requires class and filename columns")

        selection: list[dict[str, str]] = []
        seen_slide_ids: set[str] = set()
        counts = {"KIRC": 0, "KIRP": 0}
        for row in reader:
            true_class = row["class"]
            if true_class not in TARGET_CLASSES:
                raise ValueError(f"unsupported class: {true_class}")
            filename = row["filename"]
            slide_id = slide_id_from_filename(filename)
            if slide_id in seen_slide_ids:
                raise ValueError(f"duplicate normalized slide ID: {slide_id}")
            seen_slide_ids.add(slide_id)
            counts[true_class] += 1
            selection.append(
                {
                    "slide_id": slide_id,
                    "true_class": true_class,
                    "filename": filename,
                }
            )

    if len(selection) != 20 or counts != {"KIRC": 10, "KIRP": 10}:
        raise ValueError(
            "expected 10 KIRC and 10 KIRP; "
            f"observed {counts['KIRC']} KIRC and {counts['KIRP']} KIRP"
        )
    return sorted(selection, key=lambda row: row["slide_id"])


def _torn_tail_offset(data: bytes) -> int | None:
    physical_lines = data.split(b"\n")
    last_nonempty_line = max(
        (
            number
            for number, line in enumerate(physical_lines, start=1)
            if line.strip()
        ),
        default=0,
    )
    offset = 0
    for number, line in enumerate(physical_lines, start=1):
        if line.strip():
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if number == last_nonempty_line:
                    return offset
                raise ValueError(f"malformed JSON at line {number}") from exc
        offset += len(line)
        if number < len(physical_lines):
            offset += 1
    return None


def append_record(path: Path, record: dict[str, object]) -> None:
    serialized = json.dumps(
        record, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("ab") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        return

    with path.open("r+b") as handle:
        data = handle.read()
        repair_offset = _torn_tail_offset(data)
        repaired = False
        if repair_offset is not None:
            handle.seek(repair_offset)
            handle.truncate()
            data = data[:repair_offset]
            repaired = True
        if data and data[-1:] != b"\n":
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            repaired = True
        if repaired:
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0, os.SEEK_END)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def read_success_records(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}

    lines = path.read_bytes().split(b"\n")
    last_nonempty_line = max(
        (number for number, line in enumerate(lines, start=1) if line.strip()), default=0
    )
    successful: dict[tuple[str, str], dict[str, object]] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if number == last_nonempty_line:
                continue
            raise ValueError(f"malformed JSON at line {number}") from exc
        if record.get("status") == "ok":
            key = (str(record["slide_id"]), str(record["prompt_id"]))
            successful[key] = record
    return successful


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _empty_confusion_counts() -> dict[str, dict[str, int]]:
    return {
        true_class: {parsed_class: 0 for parsed_class in PARSED_CLASSES}
        for true_class in TARGET_CLASSES
    }


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    successful = [record for record in records if record.get("status") == "ok"]
    per_prompt: dict[str, object] = {}
    records_by_prompt: dict[str, list[dict[str, object]]] = {
        prompt_id: [] for prompt_id, _ in PROMPTS
    }
    for record in successful:
        prompt_id = str(record["prompt_id"])
        if prompt_id in records_by_prompt:
            records_by_prompt[prompt_id].append(record)

    for prompt_id, _ in PROMPTS:
        prompt_records = records_by_prompt[prompt_id]
        prediction_counts = {parsed_class: 0 for parsed_class in PARSED_CLASSES}
        confusion_counts = _empty_confusion_counts()
        correct = 0
        class_totals = {true_class: 0 for true_class in TARGET_CLASSES}
        class_correct = {true_class: 0 for true_class in TARGET_CLASSES}
        for record in prompt_records:
            true_class = str(record["true_class"])
            parsed_class = str(record["parsed_class"])
            if true_class not in TARGET_CLASSES:
                raise ValueError(f"unsupported true class: {true_class}")
            if parsed_class not in PARSED_CLASSES:
                raise ValueError(f"unsupported parsed class: {parsed_class}")
            class_totals[true_class] += 1
            prediction_counts[parsed_class] += 1
            confusion_counts[true_class][parsed_class] += 1
            if parsed_class == true_class:
                correct += 1
                class_correct[true_class] += 1

        n = len(prompt_records)
        kirc_recall = _rate(class_correct["KIRC"], class_totals["KIRC"])
        kirp_recall = _rate(class_correct["KIRP"], class_totals["KIRP"])
        per_prompt[prompt_id] = {
            "n": n,
            "accuracy": _rate(correct, n),
            "kirc_recall": kirc_recall,
            "kirp_recall": kirp_recall,
            "balanced_accuracy": (kirc_recall + kirp_recall) / 2,
            "other_rate": _rate(prediction_counts["OTHER"], n),
            "ambiguous_rate": _rate(prediction_counts["AMBIGUOUS"], n),
            "prediction_counts": prediction_counts,
            "confusion_counts": confusion_counts,
        }

    p1_by_slide = {
        str(record["slide_id"]): record
        for record in records_by_prompt["P1_clear_first"]
    }
    p2_by_slide = {
        str(record["slide_id"]): record
        for record in records_by_prompt["P2_papillary_first"]
    }
    paired_slide_ids = sorted(set(p1_by_slide) & set(p2_by_slide))
    agreement = 0
    stable_correct = 0
    stable_wrong = 0
    for slide_id in paired_slide_ids:
        p1_record = p1_by_slide[slide_id]
        p2_record = p2_by_slide[slide_id]
        p1_parsed = str(p1_record["parsed_class"])
        p2_parsed = str(p2_record["parsed_class"])
        if p1_parsed == p2_parsed:
            agreement += 1
            if p1_parsed == str(p1_record["true_class"]):
                stable_correct += 1
            else:
                stable_wrong += 1

    n_paired = len(paired_slide_ids)
    closed_set_stability = {
        "prompt_ids": ["P1_clear_first", "P2_papillary_first"],
        "n_paired": n_paired,
        "agreement_rate": _rate(agreement, n_paired),
        "prompt_sensitive_rate": _rate(n_paired - agreement, n_paired),
        "stable_correct_rate": _rate(stable_correct, n_paired),
        "stable_wrong_rate": _rate(stable_wrong, n_paired),
    }
    return {
        "total_successful_records": len(successful),
        "unique_slides": len({str(record["slide_id"]) for record in successful}),
        "per_prompt": per_prompt,
        "closed_set_stability": closed_set_stability,
    }
