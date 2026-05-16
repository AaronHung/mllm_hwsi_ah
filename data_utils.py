#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_slide_ids(wsi_dir: str) -> List[str]:
    exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}
    slide_ids = []
    for p in sorted(Path(wsi_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            slide_ids.append(p.stem)
    return slide_ids


def read_slide_report(reports_path: Path, slide_id: str) -> Tuple[str, str]:
    """Read JSONL report rows where question_id contains slide_id."""
    matches: List[Dict[str, Any]] = []
    with reports_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_id = str(row.get("question_id", ""))
            if slide_id in question_id:
                row["_line_num"] = line_num
                matches.append(row)

    if not matches:
        raise KeyError(f"No report rows found for slide_id: {slide_id}")

    question = matches[0].get("question", "")
    t_answer = matches[0].get("T-answer", "")

    return str(question), str(t_answer)


def load_tensor_or_keyed_tensor(path: str, allowed_keys=None) -> torch.Tensor:
    if allowed_keys is None:
        allowed_keys = [
            "features", "feature", "feats", "x", "tensor", "embeddings",
            "selected_features", "encoded_cell_features", "wsi_features", "region_features"
        ]

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict):
        for key in allowed_keys:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float()

    raise ValueError(f"Could not load tensor from {path}. Checked keys: {allowed_keys}")


def build_prompt(question_text: str) -> str:
    return (
        "You are an expert histopathology assistant.\n"
        "You are given compressed multimodal pathology tokens extracted from a whole-slide image.\n"
        "Answer the user's histopathology question carefully and in detail.\n\n"
        f"Question:\n{question_text}\n"
    )


def ensure_batched_shapes(
    wsi: torch.Tensor,      # (192,)
    region: torch.Tensor,   # (R, 192)
    patch: torch.Tensor,    # (R, 16, 512)
    cell: torch.Tensor,     # (R, 16, 384)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if wsi.ndim != 1:
        raise ValueError(f"wsi must be (192,), got {tuple(wsi.shape)}")
    if region.ndim != 2:
        raise ValueError(f"region must be (R, 192), got {tuple(region.shape)}")
    if patch.ndim != 3:
        raise ValueError(f"patch must be (R, 16, 512), got {tuple(patch.shape)}")
    if cell.ndim != 3:
        raise ValueError(f"cell must be (R, 16, 384), got {tuple(cell.shape)}")

    if wsi.shape[0] != 192:
        raise ValueError(f"Expected wsi shape [192], got {tuple(wsi.shape)}")
    if region.shape[1] != 192:
        raise ValueError(f"Expected region shape [R, 192], got {tuple(region.shape)}")
    if patch.shape[2] != 512:
        raise ValueError(f"Expected patch shape [R, 16, 512], got {tuple(patch.shape)}")
    if cell.shape[2] != 384:
        raise ValueError(f"Expected cell shape [R, 16, 384], got {tuple(cell.shape)}")

    if region.shape[0] != patch.shape[0]:
        raise ValueError("Region R and patch R do not match")
    if region.shape[0] != cell.shape[0]:
        raise ValueError("Region R and cell R do not match")
    if patch.shape[1] != cell.shape[1]:
        raise ValueError("Patch selected count and cell selected count do not match")

    return (
        wsi.unsqueeze(0),     # (1, 192)
        region.unsqueeze(0),  # (1, R, 192)
        patch.unsqueeze(0),   # (1, R, 16, 512)
        cell.unsqueeze(0),    # (1, R, 16, 384)
    )


def load_slide_feature_quadruplet(
    slide_id: str,
    wsi_feat_dir: str,
    region_feat_dir: str,
    patch_feat_dir: str,
    cell_feat_dir: str,
    cell_pt_filename: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wsi_path = os.path.join(wsi_feat_dir, f"{slide_id}.pt")
    region_path = os.path.join(region_feat_dir, f"{slide_id}.pt")
    patch_path = os.path.join(patch_feat_dir, f"{slide_id}.pt")
    cell_path = os.path.join(cell_feat_dir, slide_id, cell_pt_filename)

    if not os.path.isfile(wsi_path):
        raise FileNotFoundError(wsi_path)
    if not os.path.isfile(region_path):
        raise FileNotFoundError(region_path)
    if not os.path.isfile(patch_path):
        raise FileNotFoundError(patch_path)
    if not os.path.isfile(cell_path):
        raise FileNotFoundError(cell_path)

    wsi = load_tensor_or_keyed_tensor(wsi_path, allowed_keys=[
        "features", "feature", "feats", "x", "tensor", "embeddings", "wsi_features"
    ])
    region = load_tensor_or_keyed_tensor(region_path, allowed_keys=[
        "features", "feature", "feats", "x", "tensor", "embeddings", "region_features"
    ])
    patch = load_tensor_or_keyed_tensor(patch_path, allowed_keys=[
        "selected_features"
    ])
    cell = load_tensor_or_keyed_tensor(cell_path, allowed_keys=[
        "encoded_cell_features"
    ])

    return ensure_batched_shapes(wsi, region, patch, cell)
