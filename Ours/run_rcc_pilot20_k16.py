from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Ours.rcc_k16_protocol import (  # noqa: E402
    PROMPTS,
    PROTOCOL_VERSION,
    append_record,
    build_summary,
    load_selection,
    parse_answer,
    read_success_records,
)


CSV_COLUMNS = (
    "slide_id",
    "true_class",
    "prompt_id",
    "prompt",
    "raw_answer",
    "parsed_class",
    "correct",
    "generation_time_sec",
    "started_at",
)

_INDEX_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _load_tensor_payload(path: Path, slide_id: str, level: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"{slide_id}: failed to load {level}: {exc}") from exc


def _payload_value(
    payload: object, key: str, slide_id: str, level: str
) -> torch.Tensor:
    try:
        value = payload[key]  # type: ignore[index]
    except Exception as exc:
        raise ValueError(f"{slide_id}: {level} payload missing key {key!r}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{slide_id}: {level} payload key {key!r} is not a tensor")
    return value


def validate_features(
    slide_id: str,
    wsi: torch.Tensor,
    region: torch.Tensor,
    patch: torch.Tensor,
    cell: torch.Tensor,
    patch_indices: torch.Tensor,
    cell_indices: torch.Tensor,
) -> None:
    if tuple(wsi.shape) != (192,):
        raise ValueError(f"{slide_id}: wsi shape must be (192,), got {tuple(wsi.shape)}")

    if region.ndim != 2 or region.shape[1] != 192:
        raise ValueError(
            f"{slide_id}: region shape must be (R, 192), got {tuple(region.shape)}"
        )
    region_count = region.shape[0]
    if region_count <= 0:
        raise ValueError(f"{slide_id}: region count R must be greater than zero")

    expected_patch_shape = (region_count, 16, 512)
    if tuple(patch.shape) != expected_patch_shape:
        raise ValueError(
            f"{slide_id}: patch shape must be {expected_patch_shape}, "
            f"got {tuple(patch.shape)}"
        )

    expected_cell_shape = (region_count, 16, 384)
    if tuple(cell.shape) != expected_cell_shape:
        raise ValueError(
            f"{slide_id}: cell shape must be {expected_cell_shape}, "
            f"got {tuple(cell.shape)}"
        )

    expected_index_shape = (region_count, 16)
    if (
        tuple(patch_indices.shape) != expected_index_shape
        or tuple(cell_indices.shape) != expected_index_shape
    ):
        raise ValueError(
            f"{slide_id}: patch/cell indices shapes must both be "
            f"{expected_index_shape}, got {tuple(patch_indices.shape)} and "
            f"{tuple(cell_indices.shape)}"
        )
    if patch_indices.dtype not in _INDEX_DTYPES or cell_indices.dtype not in _INDEX_DTYPES:
        raise TypeError(
            f"{slide_id}: patch/cell indices must use an integer dtype, got "
            f"{patch_indices.dtype} and {cell_indices.dtype}"
        )
    if patch_indices.dtype != cell_indices.dtype:
        raise TypeError(
            f"{slide_id}: patch/cell indices dtypes must match exactly, got "
            f"{patch_indices.dtype} and {cell_indices.dtype}"
        )
    if not torch.equal(patch_indices, cell_indices):
        raise ValueError(f"{slide_id}: patch/cell indices are not exactly equal")

    for level, tensor in (
        ("wsi", wsi),
        ("region", region),
        ("patch", patch),
        ("cell", cell),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{slide_id}: {level} contains NaN or Inf values")


def load_features(
    slide_id: str, feature_root: Path, cell_root: Path
) -> dict[str, torch.Tensor]:
    wsi_payload = _load_tensor_payload(
        feature_root / "wsi" / f"{slide_id}.pt",
        slide_id,
        "wsi",
    )
    region_payload = _load_tensor_payload(
        feature_root / "region_4k" / f"{slide_id}.pt",
        slide_id,
        "region",
    )
    patch_payload = _load_tensor_payload(
        feature_root / "patches_filtered" / f"{slide_id}.pt",
        slide_id,
        "patch",
    )
    cell_payload = _load_tensor_payload(
        cell_root / slide_id / "encoded_cell_features.pt",
        slide_id,
        "cell",
    )
    if not isinstance(wsi_payload, torch.Tensor):
        raise TypeError(f"{slide_id}: wsi payload is not a tensor")
    if not isinstance(region_payload, torch.Tensor):
        raise TypeError(f"{slide_id}: region payload is not a tensor")

    wsi = wsi_payload.float()
    region = region_payload.float()
    patch = _payload_value(
        patch_payload, "selected_features", slide_id, "patch"
    ).float()
    cell = _payload_value(
        cell_payload, "encoded_cell_features", slide_id, "cell"
    ).float()
    patch_indices = _payload_value(
        patch_payload, "selected_indices", slide_id, "patch/cell indices"
    )
    cell_indices = _payload_value(
        cell_payload,
        "selected_patch_cell_level_indices",
        slide_id,
        "patch/cell indices",
    )
    validate_features(
        slide_id,
        wsi,
        region,
        patch,
        cell,
        patch_indices,
        cell_indices,
    )
    return {
        "wsi": wsi.float().unsqueeze(0),
        "region": region.float().unsqueeze(0),
        "patch": patch.float().unsqueeze(0),
        "cell": cell.float().unsqueeze(0),
    }


def load_frozen_model(model_dir: Path, device: str) -> tuple[object, dict[str, object]]:
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), f"CUDA device requested but unavailable: {device}"

    from mllm_hwsi import MLLMHWSI

    model, loading_info = MLLMHWSI.from_pretrained(
        str(model_dir.resolve()),
        device=device,
        dtype=torch.float16,
        local_files_only=True,
        strict_projector=True,
        return_loading_info=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Loading info: {loading_info}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}")
    print(f"LLM dtype: {next(model.llm.parameters()).dtype}")
    print(f"Projector dtype: {next(model.vl_projector.parameters()).dtype}")
    return model, loading_info


def _build_expected_metadata(
    selection: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["slide_id"], prompt_id): {
            "protocol_version": PROTOCOL_VERSION,
            "prompt": prompt,
            "true_class": row["true_class"],
        }
        for row in selection
        for prompt_id, prompt in PROMPTS
    }


def _is_current_success(
    record: dict[str, object], expected: dict[str, str]
) -> bool:
    if record.get("protocol_version") != expected["protocol_version"]:
        return False
    if record.get("prompt") != expected["prompt"]:
        return False
    if record.get("true_class") != expected["true_class"]:
        return False
    raw_answer = record.get("raw_answer")
    if not isinstance(raw_answer, str):
        return False
    parsed_class = parse_answer(raw_answer)
    if record.get("parsed_class") != parsed_class:
        return False
    correct = record.get("correct")
    return type(correct) is bool and correct == (parsed_class == expected["true_class"])


def _read_current_success_records(
    jsonl_path: Path,
    expected_metadata: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str], dict[str, object]]:
    current: dict[tuple[str, str], dict[str, object]] = {}
    for key, record in read_success_records(jsonl_path).items():
        expected = expected_metadata.get(key)
        if expected is not None and _is_current_success(record, expected):
            current[key] = record
    return current


def _write_artifacts(
    output_dir: Path,
    jsonl_path: Path,
    expected_metadata: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = list(
        _read_current_success_records(jsonl_path, expected_metadata).values()
    )
    successful.sort(key=lambda record: (str(record["slide_id"]), str(record["prompt_id"])))

    csv_path = output_dir / "predictions.csv"
    csv_temporary_path = output_dir / ".predictions.csv.tmp"
    with csv_temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    csv_temporary_path.replace(csv_path)

    summary_path = output_dir / "summary.json"
    summary_temporary_path = output_dir / ".summary.json.tmp"
    with summary_temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            build_summary(successful),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    summary_temporary_path.replace(summary_path)
    return successful


def run(
    args: argparse.Namespace,
    model_loader: Callable[[Path, str], tuple[object, dict[str, object]]] = load_frozen_model,
) -> int:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    selection = load_selection(args.selection_csv)
    expected_metadata = _build_expected_metadata(selection)
    expected_keys = set(expected_metadata)
    for row in selection:
        features = load_features(row["slide_id"], args.feature_root, args.cell_root)
        region_count = int(features["region"].shape[1])
        zero_cell_slots = int((features["cell"] == 0).all(dim=-1).sum().item())
        print(
            f"{row['slide_id']} true_class={row['true_class']} "
            f"R={region_count} zero_cell_slots={zero_cell_slots}"
        )
        del features

    class_counts = Counter(row["true_class"] for row in selection)
    print(
        f"Total slides: {len(selection)}; "
        f"class counts: KIRC={class_counts['KIRC']}, KIRP={class_counts['KIRP']}"
    )
    if args.preflight_only:
        return 0

    selected = selection[: args.limit] if args.limit is not None else selection
    output_dir = args.output_dir
    jsonl_path = output_dir / "predictions.jsonl"
    successful_by_key = _read_current_success_records(jsonl_path, expected_metadata)

    model, loading_info = model_loader(args.model_dir, args.device)
    print(f"Model loading info: {loading_info}")

    for row in selected:
        pending_prompts = [
            (prompt_id, prompt)
            for prompt_id, prompt in PROMPTS
            if (row["slide_id"], prompt_id) not in successful_by_key
        ]
        if not pending_prompts:
            continue

        features = load_features(row["slide_id"], args.feature_root, args.cell_root)
        for prompt_id, prompt in pending_prompts:
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                with torch.inference_mode():
                    answer = model.generate_from_features(
                        **features,
                        prompt=prompt,
                        max_new_tokens=32,
                        do_sample=False,
                        num_beams=1,
                    )
                parsed_class = parse_answer(answer)
                record: dict[str, object] = {
                    "status": "ok",
                    "protocol_version": PROTOCOL_VERSION,
                    "slide_id": row["slide_id"],
                    "true_class": row["true_class"],
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "raw_answer": answer,
                    "parsed_class": parsed_class,
                    "correct": parsed_class == row["true_class"],
                    "generation_time_sec": time.perf_counter() - started,
                    "started_at": started_at,
                }
                append_record(jsonl_path, record)
                successful_by_key[(row["slide_id"], prompt_id)] = record
            except Exception as exc:
                record = {
                    "status": "error",
                    "protocol_version": PROTOCOL_VERSION,
                    "slide_id": row["slide_id"],
                    "true_class": row["true_class"],
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "generation_time_sec": time.perf_counter() - started,
                    "started_at": started_at,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                append_record(jsonl_path, record)
                print(
                    f"Generation failed for {row['slide_id']} / {prompt_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    successful = _write_artifacts(output_dir, jsonl_path, expected_metadata)
    successful_keys = {
        (str(record["slide_id"]), str(record["prompt_id"]))
        for record in successful
    }
    required_keys = (
        {
            (row["slide_id"], prompt_id)
            for row in selected
            for prompt_id, _prompt in PROMPTS
        }
        if args.limit is not None
        else expected_keys
    )
    missing_keys = sorted(required_keys - successful_keys)
    for slide_id, prompt_id in missing_keys:
        print(f"Missing successful key: {slide_id} / {prompt_id}")
    if args.limit is not None:
        return 0 if not missing_keys else 1
    return 0 if successful_keys == expected_keys else 1


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RCC K16 three-prompt frozen-model experiment."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--cell-root", required=True, type=Path)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=_positive_integer)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
