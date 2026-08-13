#!/usr/bin/env python3
"""Diagnose whether WSI features vary and whether the released projector uses them.

This intentionally loads only ``vl_projector.pt`` rather than the 7B LLM.  It is
therefore safe to run on the 4090 while another process is not using the GPU.
The forward call mirrors our inference path: four feature tensors, no masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mllm_hwsi import MultiLevelProjectorEncoder


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _value(payload: Any, key: str, path: Path) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get(key), torch.Tensor):
        return payload[key]
    raise TypeError(f"{path}: expected tensor or key {key!r}")


def load_features(
    slide_id: str,
    feature_root: Path,
    cell_root: Path,
) -> dict[str, torch.Tensor]:
    wsi_path = feature_root / "wsi" / f"{slide_id}.pt"
    region_path = feature_root / "region_4k" / f"{slide_id}.pt"
    patch_path = feature_root / "patches_filtered" / f"{slide_id}.pt"
    cell_path = cell_root / slide_id / "encoded_cell_features.pt"

    wsi = _value(_load(wsi_path), "wsi_features", wsi_path).float()
    region = _value(_load(region_path), "region_features", region_path).float()
    patch_payload = _load(patch_path)
    cell_payload = _load(cell_path)
    patch = _value(patch_payload, "selected_features", patch_path).float()
    cell = _value(cell_payload, "encoded_cell_features", cell_path).float()

    if wsi.ndim == 2 and wsi.shape[0] == 1:
        wsi = wsi.squeeze(0)
    expected = {
        "wsi": (1, 192),
        "region": (2, 192),
        "patch": (3, 512),
        "cell": (3, 384),
    }
    if wsi.ndim != 1 or wsi.shape[0] != 192:
        raise ValueError(f"{slide_id}: bad WSI shape {tuple(wsi.shape)}")
    if region.ndim != 2 or region.shape[1] != 192:
        raise ValueError(f"{slide_id}: bad region shape {tuple(region.shape)}")
    if patch.ndim != 3 or patch.shape[2] != 512:
        raise ValueError(f"{slide_id}: bad patch shape {tuple(patch.shape)}")
    if cell.ndim != 3 or cell.shape[2] != 384:
        raise ValueError(f"{slide_id}: bad cell shape {tuple(cell.shape)}")
    if region.shape[0] != patch.shape[0] or region.shape[0] != cell.shape[0]:
        raise ValueError(
            f"{slide_id}: R mismatch region={region.shape[0]} "
            f"patch={patch.shape[0]} cell={cell.shape[0]}"
        )
    if patch.shape[1] != cell.shape[1]:
        raise ValueError(
            f"{slide_id}: C mismatch patch={patch.shape[1]} cell={cell.shape[1]}"
        )
    for name, tensor in {"wsi": wsi, "region": region, "patch": patch, "cell": cell}.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{slide_id}: {name} contains NaN/Inf")

    patch_indices = patch_payload.get("selected_indices") if isinstance(patch_payload, dict) else None
    cell_indices = (
        cell_payload.get("selected_patch_cell_level_indices")
        if isinstance(cell_payload, dict)
        else None
    )
    if isinstance(patch_indices, torch.Tensor) and isinstance(cell_indices, torch.Tensor):
        if not torch.equal(patch_indices.long(), cell_indices.long()):
            raise ValueError(f"{slide_id}: patch/cell selected indices differ")

    return {"wsi": wsi, "region": region, "patch": patch, "cell": cell}


def _summary_vector(features: dict[str, torch.Tensor]) -> torch.Tensor:
    wsi = features["wsi"]
    region = features["region"].mean(dim=0)
    patch = features["patch"].mean(dim=(0, 1))
    cell = features["cell"]
    valid = cell.norm(dim=-1) > 0
    cell_mean = cell.mean(dim=(0, 1))
    valid_cell_mean = cell[valid].mean(dim=0) if valid.any() else torch.zeros(384)
    return torch.cat((wsi, region, patch, cell_mean, valid_cell_mean))


def _level_summary_vectors(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cell = features["cell"]
    valid = cell.norm(dim=-1) > 0
    return {
        "wsi": features["wsi"],
        "region": features["region"].mean(dim=0),
        "patch": features["patch"].mean(dim=(0, 1)),
        "cell": cell[valid].mean(dim=0) if valid.any() else torch.zeros(384),
    }


def _pairwise_cosine(values: torch.Tensor) -> dict[str, float]:
    flat = values.reshape(values.shape[0], -1).float()
    normalized = F.normalize(flat, dim=-1)
    matrix = normalized @ normalized.T
    n = matrix.shape[0]
    if n < 2:
        return {"mean": 1.0, "min": 1.0, "max": 1.0}
    off_diag = matrix[~torch.eye(n, dtype=torch.bool)]
    return {
        "mean": float(off_diag.mean()),
        "min": float(off_diag.min()),
        "max": float(off_diag.max()),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.float().norm().clamp_min(1e-12)
    return float((a.float() - b.float()).norm() / denom)


def _project(
    projector: torch.nn.Module,
    features: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        output = projector(
            wsi=features["wsi"].unsqueeze(0).to(device),
            region=features["region"].unsqueeze(0).to(device),
            patch=features["patch"].unsqueeze(0).to(device),
            cell=features["cell"].unsqueeze(0).to(device),
        )
    output = output.detach().float().cpu()
    if not torch.isfinite(output).all():
        raise ValueError("Projector output contains NaN/Inf")
    return output


def _noise_like(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in features.items():
        scale = tensor.float().std().clamp_min(1e-6)
        result[name] = torch.randn_like(tensor) * scale
    return result


def _zero_level(features: dict[str, torch.Tensor], level: str) -> dict[str, torch.Tensor]:
    return {
        name: (torch.zeros_like(tensor) if name == level else tensor)
        for name, tensor in features.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.999)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)

    patch_paths = sorted((args.feature_root / "patches_filtered").glob("*.pt"))
    slide_ids = [path.stem for path in patch_paths]
    if args.limit is not None:
        slide_ids = slide_ids[: args.limit]
    if len(slide_ids) < 2:
        raise RuntimeError("Need at least two slides")

    device = torch.device(args.device)
    projector, loading_info = MultiLevelProjectorEncoder.from_pretrained(
        str(args.model_dir.resolve()),
        map_location="cpu",
        device=device,
        dtype=torch.float32,
        strict=True,
        local_files_only=True,
        return_loading_info=True,
    )
    projector.eval()
    print(f"Projector loading: {loading_info}")
    print(f"Projector device: {device}")
    print(f"Slides: {len(slide_ids)}")

    input_summaries = []
    input_level_summaries = {name: [] for name in ("wsi", "region", "patch", "cell")}
    real_outputs = []
    per_slide = []
    for index, slide_id in enumerate(slide_ids, start=1):
        features = load_features(slide_id, args.feature_root, args.cell_root)
        input_summaries.append(_summary_vector(features))
        for name, vector in _level_summary_vectors(features).items():
            input_level_summaries[name].append(vector)
        real = _project(projector, features, device)
        zero = _project(
            projector,
            {name: torch.zeros_like(tensor) for name, tensor in features.items()},
            device,
        )
        noise = _project(projector, _noise_like(features), device)
        real_outputs.append(real)

        row = {
            "slide_id": slide_id,
            "regions": int(features["region"].shape[0]),
            "patches_per_region": int(features["patch"].shape[1]),
            "zero_cell_slots": int((features["cell"] == 0).all(dim=-1).sum()),
            "real_vs_zero_cosine": _cosine(real, zero),
            "real_vs_zero_relative_l2": _relative_l2(real, zero),
            "real_vs_noise_cosine": _cosine(real, noise),
            "real_vs_noise_relative_l2": _relative_l2(real, noise),
        }
        for level in ("wsi", "region", "patch", "cell"):
            ablated = _project(projector, _zero_level(features, level), device)
            row[f"ablate_{level}_cosine"] = _cosine(real, ablated)
            row[f"ablate_{level}_relative_l2"] = _relative_l2(real, ablated)
        per_slide.append(row)
        print(
            f"[{index:02d}/{len(slide_ids)}] {slide_id}: "
            f"R={row['regions']} C={row['patches_per_region']} "
            f"real/zero={row['real_vs_zero_cosine']:.6f} "
            f"real/noise={row['real_vs_noise_cosine']:.6f}"
        )
        del features, real, zero, noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    input_stats = _pairwise_cosine(torch.stack(input_summaries))
    input_level_stats = {
        name: _pairwise_cosine(torch.stack(vectors))
        for name, vectors in input_level_summaries.items()
    }
    output_stats = _pairwise_cosine(torch.stack(real_outputs))
    real_zero = [row["real_vs_zero_cosine"] for row in per_slide]
    real_noise = [row["real_vs_noise_cosine"] for row in per_slide]
    result = {
        "slides": slide_ids,
        "n_slides": len(slide_ids),
        "projector_loading": loading_info,
        "input_summary_pairwise_cosine": input_stats,
        "input_level_pairwise_cosine": input_level_stats,
        "projector_output_pairwise_cosine": output_stats,
        "mean_real_vs_zero_cosine": sum(real_zero) / len(real_zero),
        "mean_real_vs_noise_cosine": sum(real_noise) / len(real_noise),
        "per_slide": per_slide,
    }

    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Input summary pairwise cosine: {input_stats}")
    for name in ("wsi", "region", "patch", "cell"):
        print(f"Input {name} pairwise cosine: {input_level_stats[name]}")
    print(f"Projector output pairwise cosine: {output_stats}")
    print(f"Mean real vs zero cosine: {result['mean_real_vs_zero_cosine']:.6f}")
    print(f"Mean real vs noise cosine: {result['mean_real_vs_noise_cosine']:.6f}")
    threshold = args.threshold
    input_diverse = any(
        stats["mean"] < threshold for stats in input_level_stats.values()
    )
    if (
        input_diverse
        and output_stats["mean"] >= threshold
        and result["mean_real_vs_noise_cosine"] >= threshold
    ):
        verdict = "PROJECTOR_INPUT_INVARIANT: consistent with GitHub Issue #5"
    elif output_stats["mean"] < threshold:
        verdict = "PROJECTOR_OUTPUT_VARIES: Issue #5 not reproduced by this probe"
    else:
        verdict = "INCONCLUSIVE: inspect per-level ablations and input statistics"
    result["verdict"] = verdict
    print(f"Verdict: {verdict}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "features_vs_projector_probe.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
