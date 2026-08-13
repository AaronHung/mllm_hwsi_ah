from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


PROMPTS = (
    (
        "B_open",
        "What is the most likely broad cancer type in this whole-slide image? "
        "Answer with the organ-level cancer type only, without predicting a histologic subtype.",
    ),
    (
        "B_fixed",
        "Classify this whole-slide image as Renal cancer or Breast cancer. "
        "Answer with exactly one class only: Renal or Breast.",
    ),
    (
        "F_open",
        "What is the most likely histopathologic diagnosis in this whole-slide image? "
        "Answer with the most specific diagnosis only.",
    ),
    (
        "F_fixed",
        "Classify this whole-slide image as one of the following: "
        "clear cell renal cell carcinoma, papillary renal cell carcinoma, "
        "invasive ductal carcinoma, or invasive lobular carcinoma. "
        "Answer with exactly one diagnosis only.",
    ),
)

PROTOCOL_VERSION = "pilot40-four-conditions-k48-v1"
TARGET_CLASSES = ("KIRC", "KIRP", "IDC", "ILC")
BINARY_CLASSES = ("Renal", "Breast")
PROMPT_TASK = {
    "B_open": "binary",
    "B_fixed": "binary",
    "F_open": "four",
    "F_fixed": "four",
}

_KIRC_PATTERN = re.compile(
    r"\b(?:clear[\s-]+cell|conventional|kirc)\b", re.IGNORECASE
)
_KIRP_PATTERN = re.compile(r"\b(?:papillary|kirp)\b", re.IGNORECASE)
_IDC_PATTERN = re.compile(
    r"\b(?:invasive[\s-]+ductal|ductal[\s-]+carcinoma|idc)\b",
    re.IGNORECASE,
)
_ILC_PATTERN = re.compile(
    r"\b(?:invasive[\s-]+lobular|lobular[\s-]+carcinoma|ilc)\b",
    re.IGNORECASE,
)
_RENAL_GENERIC_PATTERN = re.compile(
    r"\b(?:renal|kidney|rcc|renal[\s-]+cell[\s-]+carcinoma)\b",
    re.IGNORECASE,
)
_BREAST_GENERIC_PATTERN = re.compile(
    r"\b(?:breast|mammary)\b",
    re.IGNORECASE,
)


def parse_answer(answer: str, task: str) -> str:
    if task not in {"binary", "four"}:
        raise ValueError(f"unsupported task: {task}")

    text = answer.strip()
    matches = {
        "KIRC": bool(_KIRC_PATTERN.search(text)),
        "KIRP": bool(_KIRP_PATTERN.search(text)),
        "IDC": bool(_IDC_PATTERN.search(text)),
        "ILC": bool(_ILC_PATTERN.search(text)),
    }

    if task == "binary":
        has_renal = (
            matches["KIRC"]
            or matches["KIRP"]
            or bool(_RENAL_GENERIC_PATTERN.search(text))
        )
        has_breast = (
            matches["IDC"]
            or matches["ILC"]
            or bool(_BREAST_GENERIC_PATTERN.search(text))
        )
        if has_renal and has_breast:
            return "AMBIGUOUS"
        if has_renal:
            return "Renal"
        if has_breast:
            return "Breast"
        return "OTHER"

    hits = [label for label, matched in matches.items() if matched]
    if len(hits) > 1:
        return "AMBIGUOUS"
    return hits[0] if hits else "OTHER"


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
        counts = {label: 0 for label in TARGET_CLASSES}
        for row in reader:
            true_class = (row.get("class") or "").strip()
            if true_class not in TARGET_CLASSES:
                raise ValueError(f"unsupported class: {true_class}")
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError("selection CSV contains an empty filename")
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

    if len(selection) != 40 or any(counts[label] != 10 for label in TARGET_CLASSES):
        observed = ", ".join(f"{label}={counts[label]}" for label in TARGET_CLASSES)
        raise ValueError(f"expected 10 each across 40 slides; observed {observed}")
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
    serialized = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
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
        (number for number, line in enumerate(lines, start=1) if line.strip()),
        default=0,
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


def _task_for_prompt(prompt_id: str) -> str:
    try:
        return PROMPT_TASK[prompt_id]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt ID: {prompt_id}") from exc


def _truth_for_task(true_class: str, task: str) -> str:
    if task == "binary":
        return "Renal" if true_class in {"KIRC", "KIRP"} else "Breast"
    if task == "four":
        return true_class
    raise ValueError(f"unsupported task: {task}")


def _labels_for_task(task: str) -> tuple[str, ...]:
    if task == "binary":
        return (*BINARY_CLASSES, "OTHER", "AMBIGUOUS")
    if task == "four":
        return (*TARGET_CLASSES, "OTHER", "AMBIGUOUS")
    raise ValueError(f"unsupported task: {task}")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    successful = [record for record in records if record.get("status") == "ok"]
    per_condition: dict[str, object] = {}

    for prompt_id, _prompt in PROMPTS:
        task = _task_for_prompt(prompt_id)
        labels = _labels_for_task(task)
        targets = labels[:-2]
        condition_records = [
            record
            for record in successful
            if str(record.get("prompt_id")) == prompt_id
        ]
        prediction_counts = {label: 0 for label in labels}
        confusion_counts = {
            true_label: {predicted: 0 for predicted in labels}
            for true_label in targets
        }
        class_totals = {true_label: 0 for true_label in targets}
        class_correct = {true_label: 0 for true_label in targets}

        for record in condition_records:
            truth = _truth_for_task(str(record["true_class"]), task)
            parsed = str(record.get("parsed_class", "OTHER"))
            if parsed not in labels:
                parsed = "OTHER"
            class_totals[truth] += 1
            prediction_counts[parsed] += 1
            confusion_counts[truth][parsed] += 1
            if parsed == truth:
                class_correct[truth] += 1

        per_class: dict[str, dict[str, float]] = {}
        f1_values: list[float] = []
        recalls: list[float] = []
        for label in targets:
            true_positive = confusion_counts[label][label]
            predicted_total = sum(confusion_counts[truth][label] for truth in targets)
            recall = _rate(true_positive, class_totals[label])
            precision = _rate(true_positive, predicted_total)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": class_totals[label],
            }
            f1_values.append(f1)
            recalls.append(recall)

        correct = sum(class_correct.values())
        n = len(condition_records)
        per_condition[prompt_id] = {
            "task": task,
            "n": n,
            "correct": correct,
            "accuracy": _rate(correct, n),
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
            "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "per_class": per_class,
            "prediction_counts": prediction_counts,
            "confusion_counts": confusion_counts,
            "other_rate": _rate(prediction_counts["OTHER"], n),
            "ambiguous_rate": _rate(prediction_counts["AMBIGUOUS"], n),
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "total_successful": len(successful),
        "per_condition": per_condition,
    }
