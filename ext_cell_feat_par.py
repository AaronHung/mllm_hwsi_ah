#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel CellViT++ cell detection + per-cell feature extraction on selected patches.

Main outputs
------------
1) Always saves ONE slide-level tensor file:
   output_dir/
     <slide_id>/
       encoded_cell_features.pt

   Contents:
     - encoded_cell_features: [R, P, 384]
     - selected_patch_cell_level_indices: [R, P]
     - region_coords_level0:   [R, 2]
     - region_has_detail_dump: [R]
     - feature_dim: 384

   where:
     R = number of processed regions
     P = number of selected patches per region

2) Optionally saves detailed per-patch dumps only for a subset of regions:
   output_dir/
     <slide_id>/
       region_000123/
         patch_000007/
           patch.png
           overlay.png              # optional
           features.pt              # full cell features for that patch
           metadata.json
           patch_info.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import multiprocessing as mp
from transformers import ViTModel
from torchvision import transforms as T

# CellViT imports
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
try:
    from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
except Exception:
    CellViTUNI = None

from cellvit.utils.tools import unflatten_dict


# ----------------------------
# Utility
# ----------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def find_matching_wsi(wsi_dir: Path, stem: str, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        p = wsi_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = []
    for ext in exts:
        matches.extend(wsi_dir.glob(f"{stem}{ext}"))
    return matches[0] if matches else None


# ----------------------------
# Load inputs
# ----------------------------

def load_region_coords(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["coords", "region_coords", "locations", "xy", "data"]:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            arr = obj.cpu().numpy()
        else:
            arr = np.asarray(obj)

    elif suffix == ".npy":
        arr = np.load(path)

    elif suffix == ".json":
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k in ["coords", "region_coords", "locations", "xy", "data"]:
                if k in obj:
                    obj = obj[k]
                    break
        arr = np.asarray(obj)

    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if {"x", "y"}.issubset(df.columns):
            arr = df[["x", "y"]].values
        else:
            arr = df.iloc[:, :2].values

    elif suffix == ".h5":
        import h5py
        with h5py.File(path, "r") as f:
            arr = None

            for k in ["coords", "region_coords", "locations", "xy", "data"]:
                if k in f:
                    arr = f[k][:]
                    break

            if arr is None:
                def find_dataset(group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            if item.ndim == 2 and item.shape[1] == 2:
                                return item[:]
                        elif isinstance(item, h5py.Group):
                            out = find_dataset(item)
                            if out is not None:
                                return out
                    return None

                arr = find_dataset(f)

            if arr is None:
                raise ValueError(f"No suitable [N,2] coordinate dataset found in h5 file: {path}")

    else:
        raise ValueError(f"Unsupported region coordinate format: {path}")

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected region coords [N,2], got {arr.shape} from {path}")
    return arr.astype(np.int64)


def load_selected_indices(path: Path) -> List[np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "selected_indices" not in obj:
        raise ValueError(f"{path} must be a .pt dict with key 'selected_indices'")
    data = obj["selected_indices"]

    if isinstance(data, torch.Tensor):
        if data.ndim == 1:
            return [data.cpu().numpy().astype(np.int64)]
        return [row.cpu().numpy().astype(np.int64) for row in data]

    if isinstance(data, (list, tuple)):
        out = []
        for x in data:
            if isinstance(x, torch.Tensor):
                out.append(x.cpu().numpy().astype(np.int64))
            else:
                out.append(np.asarray(x, dtype=np.int64))
        return out

    arr = np.asarray(data)
    if arr.ndim == 1:
        return [arr.astype(np.int64)]
    return [np.asarray(row, dtype=np.int64) for row in arr]


# ----------------------------
# Model loading
# ----------------------------

def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    run_conf = unflatten_dict(ckpt["config"], ".")
    arch = ckpt["arch"]

    if arch == "CellViT":
        model = CellViT(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            embed_dim=run_conf["model"]["embed_dim"],
            input_channels=run_conf["model"].get("input_channels", 3),
            depth=run_conf["model"]["depth"],
            num_heads=run_conf["model"]["num_heads"],
            extract_layers=run_conf["model"]["extract_layers"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif arch == "CellViT256":
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif arch == "CellViTSAM":
        model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            vit_structure=run_conf["model"]["backbone"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif arch == "CellViTUNI":
        if CellViTUNI is None:
            raise ImportError("Checkpoint arch is CellViTUNI, but CellViTUNI import is unavailable.")
        model = CellViTUNI(
            model_uni_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
        )
    else:
        raise NotImplementedError(f"Unsupported CellViT arch: {arch}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().to(device)

    transform_settings = run_conf.get("transformations", {})
    if "normalize" in transform_settings:
        mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
        std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    inference_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    patch_token_size = int(getattr(model, "patch_size", 16))
    return model, run_conf, inference_transform, patch_token_size, arch


class CellToCellAttentionFusionViTS(nn.Module):
    """
    Aggregates variable number of cell features within ONE patch.

    Input:
      - (K, 384)    or
      - (B, K, 384)

    Output:
      - (384,)      or
      - (B, 384)
    """

    def __init__(self, device=None):
        super().__init__()

        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.embed_dim = 384
        self.cls_token = nn.Parameter(backbone.cls_token.detach().clone())
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        if device is not None:
            self.to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        squeeze_back = False

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)   # (1, K, 384)
            squeeze_back = True

        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected (K, 384) or (B, K, 384), got {tuple(tokens.shape)}")

        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1).to(tokens.device, tokens.dtype)
        x = torch.cat([cls, tokens], dim=1)  # (B, 1+K, 384)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        out = x[:, 0]  # (B, 384)

        if squeeze_back:
            out = out[0]
        return out


class CellToCellAttentionFusionViTH(nn.Module):
    """
    Aggregates a variable number of 1280-d cell features within one patch
    using a pretrained ViT-H backbone.

    Input:
      - (K, 1280)    or
      - (B, K, 1280)

    Output:
      - (1280,)      or
      - (B, 1280)

    Notes:
      - Uses pretrained google/vit-huge-patch14-224-in21k
      - No positional encoding added here
      - No reducer to 768 is included, because I could not verify a generic
        already-trained adapter module for arbitrary 1280-d ViT features.
    """

    def __init__(self, device=None):
        super().__init__()

        backbone = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")

        self.embed_dim = backbone.config.hidden_size  # should be 1280

        if self.embed_dim != 1280:
            raise ValueError(f"Expected hidden size 1280, got {self.embed_dim}")

        # Reuse pretrained CLS token and encoder blocks
        self.cls_token = nn.Parameter(
            backbone.embeddings.cls_token.detach().clone()
        )  # (1, 1, 1280)

        self.encoder = backbone.encoder
        self.layernorm = backbone.layernorm

        if device is not None:
            self.to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        squeeze_back = False

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)   # (1, K, 1280)
            squeeze_back = True

        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected (K, 1280) or (B, K, 1280), got {tuple(tokens.shape)}"
            )

        B = tokens.shape[0]

        cls = self.cls_token.expand(B, -1, -1).to(tokens.device, tokens.dtype)
        x = torch.cat([cls, tokens], dim=1)  # (B, 1+K, 1280)

        # Hugging Face ViT encoder expects embeddings directly here
        encoder_outputs = self.encoder(
            x,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        x = encoder_outputs.last_hidden_state
        x = self.layernorm(x)

        out = x[:, 0]  # CLS -> (B, 1280)

        if squeeze_back:
            out = out[0]

        return out

# ----------------------------
# Geometry / IO
# ----------------------------

def patch_index_to_xy(region_x: int, region_y: int, patch_index: int, region_size: int, patch_size: int):
    n_cols = region_size // patch_size
    row = int(patch_index) // n_cols
    col = int(patch_index) % n_cols
    x = int(region_x + col * patch_size)
    y = int(region_y + row * patch_size)
    return x, y, row, col


def read_patch_rgb_from_slide(slide: openslide.OpenSlide, x: int, y: int, patch_size: int) -> np.ndarray:
    img = slide.read_region((int(x), int(y)), 0, (int(patch_size), int(patch_size))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# ----------------------------
# Feature extraction
# ----------------------------

def apply_softmax_reorder(predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    out["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1).contiguous()
    out["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1).contiguous()
    out["hv_map"] = predictions["hv_map"].contiguous()

    if "tissue_types" in predictions:
        out["tissue_types"] = predictions["tissue_types"]
    if "tokens" in predictions:
        out["tokens"] = predictions["tokens"]
    return out


def extract_single_cell_feature(
    token_map: torch.Tensor,     # [D, Ht, Wt]
    instance_mask: np.ndarray,   # [H, W]
    centroid_xy: Tuple[float, float],
    token_patch_size: int,
    mode: str = "mask_mean",
) -> torch.Tensor:
    D, Ht, Wt = token_map.shape
    cx, cy = centroid_xy
    tx = int(np.clip(cx // token_patch_size, 0, Wt - 1))
    ty = int(np.clip(cy // token_patch_size, 0, Ht - 1))

    if mode == "centroid":
        return token_map[:, ty, tx]

    if mode != "mask_mean":
        raise ValueError(f"Unknown feature mode: {mode}")

    mask_t = torch.from_numpy(instance_mask.astype(np.float32))[None, None, :, :].to(token_map.device)
    pooled = F.avg_pool2d(mask_t, kernel_size=token_patch_size, stride=token_patch_size)
    pooled = pooled[0, 0] > 0

    coords = pooled.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return token_map[:, ty, tx]

    feats = token_map[:, coords[:, 0], coords[:, 1]]
    return feats.mean(dim=1)


def draw_overlay(rgb_patch: np.ndarray, cells_meta: List[Dict[str, Any]], alpha: float = 0.55) -> Image.Image:
    base = Image.fromarray(rgb_patch)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay, "RGBA")

    for cell in cells_meta:
        contour = cell.get("contour", None)
        if contour is not None and len(contour) >= 3:
            contour_xy = [(float(p[0]), float(p[1])) for p in contour]
            drawer.polygon(
                contour_xy,
                fill=(255, 0, 0, int(255 * alpha)),
                outline=(255, 255, 0, 255),
            )
        centroid = cell.get("centroid", None)
        if centroid is not None and len(centroid) == 2:
            x, y = float(centroid[0]), float(centroid[1])
            r = 2
            drawer.ellipse((x-r, y-r, x+r, y+r), fill=(0, 255, 255, 255))

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


@torch.no_grad()
def run_model_on_batch(
    model,
    batch_rgb: List[np.ndarray],
    transform,
    device: torch.device,
    mixed_precision: bool,
    magnification: int,
):
    batch_tensor = torch.stack([transform(img) for img in batch_rgb], dim=0).to(device)

    if mixed_precision and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = model.forward(batch_tensor, retrieve_tokens=True)
    else:
        preds = model.forward(batch_tensor, retrieve_tokens=True)

    token_maps = preds["tokens"]  # [B, D, Ht, Wt]
    preds_pp = apply_softmax_reorder(preds)
    instance_maps, type_preds = model.calculate_instance_map(preds_pp, magnification=magnification)
    return instance_maps.cpu().numpy().astype(np.int32), type_preds, token_maps.detach()


def extract_patch_outputs(
    rgb_patch: np.ndarray,
    instance_map: np.ndarray,
    type_pred_dict: Dict[int, Dict[str, Any]],
    token_map: torch.Tensor,
    token_patch_size: int,
    feature_mode: str,
    cell_to_cell_encoder: nn.Module,
) -> Dict[str, Any]:
    cell_features = []
    cell_metadata = []

    for inst_id, cell_info in type_pred_dict.items():
        inst_id_int = int(inst_id)
        mask = instance_map == inst_id_int
        if not np.any(mask):
            continue

        centroid = cell_info.get("centroid", None)
        if centroid is None or len(centroid) != 2:
            ys, xs = np.where(mask)
            centroid = [float(xs.mean()), float(ys.mean())]

        feat = extract_single_cell_feature(
            token_map=token_map,
            instance_mask=mask,
            centroid_xy=(float(centroid[0]), float(centroid[1])),
            token_patch_size=token_patch_size,
            mode=feature_mode,
        )
        cell_features.append(feat)

        md = {
            "instance_id": inst_id_int,
            "centroid": to_builtin(cell_info.get("centroid")),
            "bbox": to_builtin(cell_info.get("bbox")),
            "contour": to_builtin(cell_info.get("contour")),
            "type": int(cell_info.get("type", -1)) if cell_info.get("type", None) is not None else -1,
            "type_prob": float(cell_info.get("type_prob", -1.0)) if cell_info.get("type_prob", None) is not None else -1.0,
            "feature_mode": feature_mode,
        }
        cell_metadata.append(md)

    encoder_device = next(cell_to_cell_encoder.parameters()).device

    if len(cell_features) == 0:
        feat_tensor = torch.empty((0, token_map.shape[0]), dtype=torch.float32, device=encoder_device)
        encoded_patch_feature = torch.zeros((cell_to_cell_encoder.embed_dim,), dtype=torch.float32, device=encoder_device)
    else:
        feat_tensor = torch.stack(cell_features, dim=0).to(encoder_device).float()   # [K, 384]
        with torch.no_grad():
            encoded_patch_feature = cell_to_cell_encoder(feat_tensor).detach().cpu().float()

    encoded_patch_feature = encoded_patch_feature.detach().cpu().float()

    return {
        "cell_features": feat_tensor.detach().cpu(),
        "encoded_patch_feature": encoded_patch_feature,
        "cell_metadata": cell_metadata,
        "num_cells": int(feat_tensor.shape[0]),
    }


def save_patch_output(
    out_dir: Path,
    rgb_patch: np.ndarray,
    instance_map: np.ndarray,
    token_map: torch.Tensor,
    token_patch_size: int,
    patch_outputs: Dict[str, Any],
    save_overlay: bool,
):
    mkdir(out_dir)

    feat_tensor = patch_outputs["cell_features"]
    cell_metadata = patch_outputs["cell_metadata"]

    torch.save(
        {
            "cell_features": feat_tensor,
            "encoded_patch_feature": patch_outputs["encoded_patch_feature"],
            "token_grid_shape": tuple(token_map.shape[-2:]),
            "token_patch_size": token_patch_size,
            "num_cells": int(feat_tensor.shape[0]),
        },
        out_dir / "features.pt",
    )

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"num_cells": int(feat_tensor.shape[0]), "cells": cell_metadata}, f, indent=2)

    Image.fromarray(rgb_patch).save(out_dir / "patch.png")

    if save_overlay:
        overlay = draw_overlay(rgb_patch, cell_metadata)
        overlay.save(out_dir / "overlay.png")


# ----------------------------
# WSI task preparation
# ----------------------------

def collect_triplets(
    wsi_dir: Path,
    region_coords_dir: Path,
    selected_indices_dir: Path,
    wsi_exts: Sequence[str],
):
    region_files = []
    for ext in ["*.pt", "*.pth", "*.npy", "*.json", "*.csv", "*.h5"]:
        region_files.extend(region_coords_dir.glob(ext))

    triplets = []
    for rc in sorted(region_files):
        stem = rc.stem
        wsi_path = find_matching_wsi(wsi_dir, stem, wsi_exts)
        if wsi_path is None:
            continue

        sel_path = None
        for ext in [".pt", ".pth"]:
            cand = selected_indices_dir / f"{stem}{ext}"
            if cand.exists():
                sel_path = cand
                break

        if sel_path is None:
            continue

        triplets.append((wsi_path, rc, sel_path))

    return triplets


# ----------------------------
# Per-WSI processing
# ----------------------------

def process_single_wsi(
    wsi_path: Path,
    region_coords_path: Path,
    selected_indices_path: Path,
    args,
    model,
    transform,
    token_patch_size: int,
    cell_to_cell_encoder: nn.Module,
    device: torch.device,
    worker_rank: int,
):
    slide_id = wsi_path.stem
    print(f"[Worker {worker_rank}] Processing {slide_id} on {device}")

    region_coords = load_region_coords(region_coords_path)
    selected_per_region = load_selected_indices(selected_indices_path)

    if len(region_coords) != len(selected_per_region):
        raise ValueError(
            f"Mismatch for {slide_id}: {len(region_coords)} region coords vs "
            f"{len(selected_per_region)} selected-index entries"
        )

    slide_out_dir = mkdir(Path(args.output_dir) / slide_id)
    slide = openslide.OpenSlide(str(wsi_path))

    processed_region_ids = list(range(len(region_coords)))
    if args.max_regions_per_wsi is not None and args.max_regions_per_wsi > 0:
        processed_region_ids = processed_region_ids[:min(args.max_regions_per_wsi, len(region_coords))]

    if args.save_full_outputs_regions_per_wsi > 0:
        n_detail = min(args.save_full_outputs_regions_per_wsi, len(processed_region_ids))
        if args.full_output_region_selection == "random":
            detailed_region_ids = set(random.sample(processed_region_ids, n_detail))
        else:
            detailed_region_ids = set(processed_region_ids[:n_detail])
    else:
        detailed_region_ids = set()

    per_region_patch_indices: List[np.ndarray] = []
    per_region_coords_scaled: List[Tuple[int, int]] = []

    for region_idx in processed_region_ids:
        rx, ry = region_coords[region_idx]
        rx = int(round(rx * args.coord_scale))
        ry = int(round(ry * args.coord_scale))

        sel_indices = np.asarray(selected_per_region[region_idx], dtype=np.int64)
        if sel_indices.size == 0:
            raise ValueError(
                f"{slide_id}: region {region_idx} has zero selected patches. "
                "This no-padding version requires all processed regions to have the same number of patches."
            )

        if args.max_patches_per_region is not None and args.max_patches_per_region > 0:
            sel_indices = sel_indices[:args.max_patches_per_region]

        per_region_patch_indices.append(sel_indices)
        per_region_coords_scaled.append((rx, ry))

    if len(per_region_patch_indices) == 0:
        slide.close()
        torch.save(
            {
                "encoded_cell_features": torch.empty((0, 0, cell_to_cell_encoder.embed_dim), dtype=torch.float32),
                "selected_patch_cell_level_indices": torch.empty((0, 0), dtype=torch.long),
                "region_coords_level0": torch.empty((0, 2), dtype=torch.long),
                "region_has_detail_dump": torch.empty((0,), dtype=torch.bool),
                "feature_dim": int(cell_to_cell_encoder.embed_dim),
            },
            slide_out_dir / "slide_encoded_features.pt",
        )
        print(f"[Worker {worker_rank}] No processed regions for {slide_id}")
        return

    num_regions = len(per_region_patch_indices)
    num_patches_per_region = len(per_region_patch_indices[0])

    for i, arr in enumerate(per_region_patch_indices):
        if len(arr) != num_patches_per_region:
            raise ValueError(
                f"{slide_id}: inconsistent selected patch counts across regions in no-padding mode. "
                f"Region compact 0 has {num_patches_per_region}, but region compact {i} has {len(arr)}."
            )

    encoded_cell_features = torch.zeros(
        (num_regions, num_patches_per_region, cell_to_cell_encoder.embed_dim),
        dtype=torch.float32,
    )
    selected_patch_cell_level_indices_tensor = torch.zeros(
        (num_regions, num_patches_per_region),
        dtype=torch.long,
    )
    region_coords_tensor = torch.zeros((num_regions, 2), dtype=torch.long)
    region_has_detail_dump = torch.zeros((num_regions,), dtype=torch.bool)

    work_items = []
    for out_region_pos, (region_idx, (rx, ry), sel_indices) in enumerate(
        zip(processed_region_ids, per_region_coords_scaled, per_region_patch_indices)
    ):
        region_coords_tensor[out_region_pos] = torch.tensor([rx, ry], dtype=torch.long)
        region_has_detail_dump[out_region_pos] = region_idx in detailed_region_ids

        for patch_pos_in_region, patch_idx in enumerate(sel_indices.tolist()):
            patch_x, patch_y, row, col = patch_index_to_xy(
                region_x=rx,
                region_y=ry,
                patch_index=int(patch_idx),
                region_size=args.region_size,
                patch_size=args.patch_size,
            )

            selected_patch_cell_level_indices_tensor[out_region_pos, patch_pos_in_region] = int(patch_idx)

            work_items.append(
                {
                    "region_idx": region_idx,
                    "out_region_pos": out_region_pos,
                    "patch_pos_in_region": patch_pos_in_region,
                    "patch_idx": int(patch_idx),
                    "patch_x": patch_x,
                    "patch_y": patch_y,
                    "row": row,
                    "col": col,
                    "save_detail": region_idx in detailed_region_ids,
                }
            )

    if args.save_overlay_p > 0:
        candidate_overlay_ids = [i for i, it in enumerate(work_items) if it["save_detail"]]
        overlay_ids = set(random.sample(candidate_overlay_ids, min(args.save_overlay_p, len(candidate_overlay_ids))))
    else:
        overlay_ids = set()

    for batch_start in tqdm(
        range(0, len(work_items), args.batch_size),
        desc=f"{slide_id}",
        position=worker_rank,
        leave=True,
    ):
        batch_items = work_items[batch_start: batch_start + args.batch_size]
        batch_rgb = [
            read_patch_rgb_from_slide(slide, it["patch_x"], it["patch_y"], args.patch_size)
            for it in batch_items
        ]

        instance_maps, type_preds, token_maps = run_model_on_batch(
            model=model,
            batch_rgb=batch_rgb,
            transform=transform,
            device=device,
            mixed_precision=args.enforce_amp,
            magnification=args.magnification,
        )

        for j, it in enumerate(batch_items):
            patch_outputs = extract_patch_outputs(
                rgb_patch=batch_rgb[j],
                instance_map=instance_maps[j],
                type_pred_dict=type_preds[j],
                token_map=token_maps[j],
                token_patch_size=token_patch_size,
                feature_mode=args.feature_mode,
                cell_to_cell_encoder=cell_to_cell_encoder,
            )

            encoded_cell_features[it["out_region_pos"], it["patch_pos_in_region"]] = patch_outputs["encoded_patch_feature"]

            if it["save_detail"]:
                patch_out_dir = mkdir(
                    slide_out_dir /
                    f"region_{it['region_idx']:06d}" /
                    f"patch_{it['patch_idx']:06d}"
                )

                save_patch_output(
                    out_dir=patch_out_dir,
                    rgb_patch=batch_rgb[j],
                    instance_map=instance_maps[j],
                    token_map=token_maps[j],
                    token_patch_size=token_patch_size,
                    patch_outputs=patch_outputs,
                    save_overlay=(batch_start + j) in overlay_ids,
                )

                with open(patch_out_dir / "patch_info.json", "w") as f:
                    json.dump(
                        {
                            "wsi": slide_id,
                            "region_idx": int(it["region_idx"]),
                            "region_compact_pos": int(it["out_region_pos"]),
                            "patch_pos_in_region": int(it["patch_pos_in_region"]),
                            "patch_idx": int(it["patch_idx"]),
                            "patch_x_level0": int(it["patch_x"]),
                            "patch_y_level0": int(it["patch_y"]),
                            "region_grid_row": int(it["row"]),
                            "region_grid_col": int(it["col"]),
                            "patch_size": int(args.patch_size),
                        },
                        f,
                        indent=2,
                    )

    slide.close()

    torch.save(
        {
            "encoded_cell_features": encoded_cell_features,   # [R, P, 384]
            "selected_patch_cell_level_indices": selected_patch_cell_level_indices_tensor,  # [R, P]
            "region_coords_level0": region_coords_tensor,       # [R, 2]
            "region_has_detail_dump": region_has_detail_dump,   # [R]
            "feature_dim": int(cell_to_cell_encoder.embed_dim),
        },
        slide_out_dir / "encoded_cell_features.pt",
    )

    print(
        f"[Worker {worker_rank}] Finished {slide_id} | "
        f"saved slide tensor: {tuple(encoded_cell_features.shape)}"
    )


# ----------------------------
# Parallel workers
# ----------------------------

def split_round_robin(items: List[Any], n_buckets: int) -> List[List[Any]]:
    buckets = [[] for _ in range(n_buckets)]
    for i, item in enumerate(items):
        buckets[i % n_buckets].append(item)
    return buckets


def worker_main(worker_rank: int, gpu_id: int, assigned_triplets: List[Tuple[Path, Path, Path]], args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed_everything(args.seed + worker_rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model, run_conf, transform, token_patch_size, arch = build_model_from_checkpoint(
        Path(args.checkpoint), device
    )

    if "ViT-256" in args.checkpoint:
        cell_to_cell_encoder = CellToCellAttentionFusionViTS(device=device).eval()
    elif "SAM-H" in args.checkpoint or "Virchow" in args.checkpoint:
        cell_to_cell_encoder = CellToCellAttentionFusionViTH(device=device).eval()
    else:
        raise NotImplementedError(
            "Unknown checkpoint type for cell-to-cell encoder selection. "
            "Expected checkpoint name to contain 'ViT-256' or 'SAM-H'/'Virchow'."
        )

    print(
        f"[Worker {worker_rank}] GPU logical=0 physical={gpu_id}, "
        f"arch={arch}, assigned_wsis={len(assigned_triplets)}"
    )

    for wsi_path, region_coords_path, selected_indices_path in assigned_triplets:
        process_single_wsi(
            wsi_path=wsi_path,
            region_coords_path=region_coords_path,
            selected_indices_path=selected_indices_path,
            args=args,
            model=model,
            transform=transform,
            token_patch_size=token_patch_size,
            cell_to_cell_encoder=cell_to_cell_encoder,
            device=device,
            worker_rank=worker_rank,
        )


# ----------------------------
# Args / main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--wsi_dir", type=str, required=True)
    p.add_argument("--region_coords_dir", type=str, required=True)
    p.add_argument("--selected_indices_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument(
        "--wsi_exts",
        type=str,
        nargs="+",
        default=[".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu"]
    )

    p.add_argument("--region_size", type=int, default=4096)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--coord_scale", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--magnification", type=int, choices=[20, 40], default=20)
    p.add_argument("--enforce_amp", action="store_true")

    p.add_argument("--feature_mode", type=str, choices=["centroid", "mask_mean"], default="mask_mean")
    p.add_argument("--save_overlay_p", type=int, default=10000000)

    p.add_argument("--max_wsis", type=int, default=None)
    p.add_argument("--max_regions_per_wsi", type=int, default=None)
    p.add_argument("--max_patches_per_region", type=int, default=None)

    p.add_argument("--save_full_outputs_regions_per_wsi", type=int, default=20)
    p.add_argument("--full_output_region_selection", type=str, choices=["first", "random"], default="first")

    p.add_argument("--all_gpus", action="store_true")
    p.add_argument("--workers_per_gpu", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.region_size % args.patch_size != 0:
        raise ValueError("region_size must be divisible by patch_size")
    if args.patch_size % 16 != 0:
        raise ValueError("patch_size must be divisible by 16 for CellViT tokenization")

    triplets = collect_triplets(
        wsi_dir=Path(args.wsi_dir),
        region_coords_dir=Path(args.region_coords_dir),
        selected_indices_dir=Path(args.selected_indices_dir),
        wsi_exts=args.wsi_exts,
    )

    if len(triplets) == 0:
        raise RuntimeError("No matching (WSI, region_coords, selected_indices) triplets found.")

    if args.max_wsis is not None and args.max_wsis > 0:
        triplets = triplets[:args.max_wsis]

    gpu_count = torch.cuda.device_count() if args.all_gpus else 1
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPUs detected. Disable --all_gpus or make GPUs visible.")

    total_workers = int(gpu_count) * int(args.workers_per_gpu)
    if total_workers < 1:
        raise ValueError("total workers must be >= 1")

    print(f"Found {len(triplets)} WSI(s)")
    print(f"Launching {total_workers} workers = {gpu_count} GPU(s) x {args.workers_per_gpu} worker(s)/GPU")

    buckets = split_round_robin(triplets, total_workers)

    processes = []
    for worker_rank in range(total_workers):
        gpu_id = worker_rank % gpu_count
        assigned = buckets[worker_rank]
        if len(assigned) == 0:
            continue

        p = mp.Process(
            target=worker_main,
            args=(worker_rank, gpu_id, assigned, args),
        )
        p.start()
        processes.append(p)

    exit_code = 0
    for p in processes:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode

    if exit_code != 0:
        raise RuntimeError(f"One or more workers failed. Last non-zero exit code: {exit_code}")

    print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()