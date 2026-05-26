#!/usr/bin/env python3
"""
Extract HIPT features from CLAM coordinate H5 files.

Inputs
- WSIs in a directory (.svs, .tif, .tiff, .ndpi, .mrxs, .scn, .vms, .vmu, .bif)
- CLAM 256-patch H5 files with dataset '([github.com](https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_model_utils.py))h dataset 'coords'
- Local HIPT repository and pretrained checkpoints

Outputs per slide
- patch256/<slide_id>.pt        : [M, 384] patch-level CLS embeddings
- region_tokens/<slide_id>.pt   : [R, 256, 384] region token embeddings (optional)
- region_4k/<slide_id>.pt       : [R, 192] region-level CLS embeddings
- wsi/<slide_id>.pt             : [192] mean pooled WSI embedding
- metadata/<slide_id>.json      : run metadata and counts

Example
python ext_feats_conch_hierar_par.py \
  --wsi_dir ../TCGA_data/WSIs \
  --h5_dir_4096 ../TCGA_data/regions_mag20x/patches \
  --hipt_repo ./HIPT_4K \
  --checkpoint256 ../HIPT_ckpts/vit_256_small_dino.pth \
  --checkpoint4k ../HIPT_ckpts/vit_4096_xs_dino.pth \
  --trident_repo ./trident \
  --conch_ckpt_path ../CONCH_ckpt/pytorch_model.bin \
  --conch_batch_size 128 \
  --encoder_name conch_v1 \
  --out_dir ../TCGA_data/features_mag20x_conch_v1_par \
  --encoder_name conch_v1 \
  --device cuda:1 \
  --patch_size 256 \
  --extract_patch_features
  --all_gpus
  --workers_per_gpu 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import Tensor
from tqdm import tqdm

from semantic_patch_filtering import semantic_patch_filtering2
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

try:
    import openslide
except ImportError as exc:
    raise ImportError(
        "OpenSlide is required. Install openslide-python and the system OpenSlide library."
    ) from exc


SUPPORTED_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HIPT features from CLAM coordinate files")
    parser.add_argument("--wsi_dir", type=str, required=True)
    parser.add_argument("--reports_path", type=str, required=False, help="Optional directory with slide-level reports for semantic filtering")
    parser.add_argument("--h5_dir_4096", type=str, required=True)
    parser.add_argument("--hipt_repo", type=str, required=True, help="Path to local HIPT repo root")
    parser.add_argument("--checkpoint256", type=str, required=True)
    parser.add_argument("--checkpoint4k", type=str, required=True)
    parser.add_argument("--trident_repo", type=str, required=True, help="Path to local TRIDENT repository root")
    parser.add_argument("--conch_ckpt_path", type=str, default=None, help="Optional local CONCH 1.5 checkpoint path")
    parser.add_argument("--conch_model_cfg", type=str, default="conch_ViT-B-16", help="CONCH model configuration")
    parser.add_argument("--encoder_name", type=str, default="conch_v15", choices=["conch_v1", "conch_v15"], help="TRIDENT encoder name")
    parser.add_argument("--conch_batch_size", type=int, default=128, help="Batch size for forward pass")
    parser.add_argument("--data_mode", type=str, default="train", choices=["train", "inference"], help="Data mode for feature extraction")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size_256", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--region_size", type=int, default=4096)
    parser.add_argument("--read_level_256", type=int, default=None, help="Override OpenSlide level for 256 crops")
    parser.add_argument("--read_level_4096", type=int, default=None, help="Override OpenSlide level for 4096 crops")
    parser.add_argument("--process_single_slide", type=str, default=None)
    parser.add_argument("--slide_ext", type=str, default=None, help="Optional fixed slide extension, e.g. .svs")
    parser.add_argument("--extract_patch_features", action="store_true")
    parser.add_argument("--skip_existing", default=True, help="Skip existing feature files")
    parser.add_argument("--allow_edge_padding", action="store_true", help="Pad partial edge crops instead of skipping them")
    parser.add_argument("--fp16", action="store_true", help="Use autocast for model forward on CUDA")
    parser.add_argument("--all_gpus", action="store_true", help="Use all visible CUDA GPUs")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of multiprocessing workers per GPU")
    parser.add_argument("--log_every", type=int, default=1, help="Log worker progress every N slides")
    parser.add_argument("--filtering_window_size", type=int, default=7, help="Window size for local similarity in semantic patch filtering")
    parser.add_argument("--n_diss_features", type=int, default=32, help="Number of dissimilar features to consider")
    parser.add_argument("--top_k", type=int, default=16, help="Number of patches to keep per region after semantic filtering")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_coords(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5_file:
        if "coords" not in h5_file:
            raise KeyError(f"Missing 'coords' dataset in {h5_path}")
        coords = np.array(h5_file["coords"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Unexpected coords shape {coords.shape} in {h5_path}")
    return coords[:, :2].astype(np.int64)


def build_wsi_index(wsi_dir: Path, fixed_ext: Optional[str] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if fixed_ext is not None:
        fixed_ext = fixed_ext if fixed_ext.startswith(".") else f".{fixed_ext}"
        for path in sorted(wsi_dir.glob(f"*{fixed_ext}")):
            index[path.stem] = path
        return index

    for path in sorted(wsi_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_EXTS:
            index[path.stem] = path
    return index


def add_hipt_to_syspath(hipt_repo: Path) -> None:
    repo_root = str(hipt_repo.resolve())
    hipt_4k = str((hipt_repo / "HIPT_4K").resolve())
    for candidate in [repo_root, hipt_4k]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def load_hipt_models(
    hipt_repo: Path,
    checkpoint256: Path,
    checkpoint4k: Path,
    device: torch.device,
):
    add_hipt_to_syspath(hipt_repo)

    from HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms

    model256 = get_vit256(str(checkpoint256)).to(device)
    model4k = get_vit4k(str(checkpoint4k)).to(device)
    model256.eval()
    model4k.eval()
    transform = eval_transforms()
    return model256, model4k, transform

'''
def add_trident_to_syspath(trident_repo: Path) -> None:
    repo_root = str(trident_repo.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_trident_conch_model(
    trident_repo: Path,
    encoder_name: str,
    checkpoint_path: Optional[Path],
    device: torch.device,
):
    add_trident_to_syspath(trident_repo)
    from trident.patch_encoder_models.load import encoder_factory

    kwargs = {}
    if checkpoint_path is not None:
        kwargs["weights_path"] = str(checkpoint_path)

    encoder = encoder_factory(encoder_name, **kwargs)
    encoder = encoder.to(device)
    encoder.eval()

    preprocess = encoder.eval_transforms
    precision = getattr(encoder, "precision", torch.float32)
    return encoder, preprocess, precision
'''

def get_slide_dimensions(slide: openslide.OpenSlide, level: int) -> Tuple[int, int]:
    width, height = slide.level_dimensions[level]
    return int(width), int(height)


def choose_level(slide: openslide.OpenSlide, requested: Optional[int]) -> int:
    if requested is None:
        return 0
    if requested < 0 or requested >= slide.level_count:
        raise ValueError(f"Requested level {requested} out of range [0, {slide.level_count - 1}]")
    return int(requested)


def read_square_region(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    size: int,
    level: int,
    allow_edge_padding: bool,
) -> Optional[Image.Image]:
    level_w, level_h = get_slide_dimensions(slide, level)
    downsample = float(slide.level_downsamples[level])
    size_l = int(round(size / downsample)) if level != 0 else size

    if size_l <= 0:
        return None

    if not allow_edge_padding:
        if x < 0 or y < 0 or x + size_l > level_w or y + size_l > level_h:
            return None
        img = slide.read_region((int(round(x * downsample)), int(round(y * downsample))), level, (size_l, size_l)).convert("RGB")
        if size_l != size:
            img = img.resize((size, size), resample=Image.BILINEAR)
        return img

    read_x = max(0, x)
    read_y = max(0, y)
    read_w = max(0, min(size_l, level_w - read_x))
    read_h = max(0, min(size_l, level_h - read_y))
    if read_w <= 0 or read_h <= 0:
        return None

    tile = slide.read_region((int(round(read_x * downsample)), int(round(read_y * downsample))), level, (read_w, read_h)).convert("RGB")
    canvas = Image.new("RGB", (size_l, size_l), (255, 255, 255))
    paste_x = max(0, -x)
    paste_y = max(0, -y)
    canvas.paste(tile, (paste_x, paste_y))
    if size_l != size:
        canvas = canvas.resize((size, size), resample=Image.BILINEAR)
    return canvas


def pil_to_tensor(img: Image.Image, transform) -> Tensor:
    return transform(img)


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def extract_region_patch_features(
    slide: openslide.OpenSlide,
    coords_4096: np.ndarray,
    model256,
    model4k,
    conch_model,
    conch_preprocess,
    batch_size: int,
    transform,
    device: torch.device,
    patch_size: int,
    region_size: int,
    level: int,
    allow_edge_padding: bool,
    extract_patch_features: bool,
    use_fp16: bool,
) -> Tuple[Optional[Tensor], Tensor, np.ndarray]:
    grid_size = region_size // patch_size
    if grid_size * patch_size != region_size:
        raise ValueError("region_size must be divisible by patch_size")

    patch_features: List[Tensor] = []
    region_cls: List[Tensor] = []
    valid_coords: List[Tuple[int, int]] = []
    autocast_enabled = use_fp16 and device.type == "cuda"

    with torch.inference_mode():
        for xy in tqdm(coords_4096, desc="4096 regions", leave=False):
            x0, y0 = int(xy[0]), int(xy[1])
            region_img = read_square_region(slide, x0, y0, region_size, level, allow_edge_padding)
            if region_img is None:
                continue

            patch_batch: List[Tensor] = []
            patch_tensors: List[Tensor] = []

            for gy in range(grid_size):
                for gx in range(grid_size):
                    left = gx * patch_size
                    upper = gy * patch_size
                    patch = region_img.crop((left, upper, left + patch_size, upper + patch_size))
                    patch_batch.append(pil_to_tensor(patch, transform))
                    patch_tensors.append(conch_preprocess(patch))

            xb = torch.stack(patch_batch, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                feat256 = model256(xb)  # [256, 384]
                feat_grid = feat256.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2).contiguous()
                cls4k = model4k(feat_grid)  # [1, 192]

            # Extract patch level features with conch
            feats: List[Tensor] = []
            for batch in batched(patch_tensors, batch_size):
                xb = torch.stack(list(batch), dim=0).to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    #out = conch_model(xb)
                    out = conch_model.encode_image(xb)
                feats.append(out.detach().float().cpu())
            
            feat256_cpu = torch.cat(feats, dim=0).detach().float().cpu()

            cls4k_cpu = cls4k.detach().float().cpu()

            if extract_patch_features:
                patch_features.append(feat256_cpu)
            region_cls.append(cls4k_cpu.squeeze(0))
            valid_coords.append((x0, y0))

    patches_tensor: Optional[Tensor]
    if extract_patch_features:
        if patch_features:
            patches_tensor = torch.stack(patch_features, dim=0)
        else:
            patches_tensor = torch.empty((0, grid_size * grid_size, 384), dtype=torch.float32)
    else:
        patches_tensor = None

    if region_cls:
        cls_tensor = torch.stack(region_cls, dim=0)
    else:
        cls_tensor = torch.empty((0, 192), dtype=torch.float32)

    return patches_tensor, cls_tensor, np.asarray(valid_coords, dtype=np.int64)


def save_tensor(path: Path, tensor: Tensor) -> None:
    ensure_dir(path.parent)
    torch.save(tensor, path)


def save_coords_h5(path: Path, coords: np.ndarray) -> None:
    ensure_dir(path.parent)
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=coords, compression="gzip")


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_gpu_ids(args: argparse.Namespace, device: torch.device) -> List[int]:
    if device.type != "cuda":
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    if args.all_gpus:
        n_gpu = torch.cuda.device_count()
        if n_gpu < 1:
            raise RuntimeError("--all_gpus was set but no CUDA GPUs are visible")
        return list(range(n_gpu))

    if device.index is not None:
        return [int(device.index)]
    return [0]


def _outputs_exist_for_slide(
    slide_id: str,
    patches_out: Path,
    region_4k_out: Path,
    wsi_out: Path,
    coords_region_out: Path,
    metadata_out: Path,
    args: argparse.Namespace,
) -> bool:
    region4k_path = region_4k_out / f"{slide_id}.pt"
    wsi_path = wsi_out / f"{slide_id}.pt"
    coords_path = coords_region_out / f"{slide_id}.h5"
    patches_path = patches_out / f"{slide_id}.pt"
    filtered_patches_path = Path(str(patches_out) + "_filtered") / f"{slide_id}.pt"
    meta_path = metadata_out / f"{slide_id}.json"

    base_ok = region4k_path.exists() and wsi_path.exists() and coords_path.exists() and meta_path.exists()
    if not base_ok:
        return False

    if args.extract_patch_features and (not patches_path.exists()):
        return False

    if args.reports_path is not None and (not filtered_patches_path.exists()):
        return False

    return True


def _process_slides_for_worker(
    worker_rank: int,
    world_size: int,
    device: torch.device,
    assigned_slide_ids: Sequence[str],
    args: argparse.Namespace,
    wsi_index: Dict[str, Path],
    h5_4096_paths: Dict[str, Path],
    patches_out: Path,
    region_4k_out: Path,
    wsi_out: Path,
    coords_region_out: Path,
    metadata_out: Path,
) -> Tuple[int, int, int]:
    worker_name = f"worker-{worker_rank:02d}/{world_size:02d}"
    print(f"[{worker_name}] starting on device={device} with {len(assigned_slide_ids)} slides")

    model256, model4k, transform = load_hipt_models(
        Path(args.hipt_repo),
        Path(args.checkpoint256),
        Path(args.checkpoint4k),
        device,
    )

    '''
    conch_model, conch_preprocess, _ = load_trident_conch_model(
        trident_repo=Path(args.trident_repo),
        encoder_name=args.encoder_name,
        checkpoint_path=Path(args.conch_ckpt_path) if args.conch_ckpt_path is not None else None,
        device=device,
    )
    '''

    conch_model_cfg = args.conch_model_cfg
    conch_checkpoint_path = args.conch_ckpt_path
    conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, 
                                                           conch_checkpoint_path)
    conch_model = conch_model.to(device)
    conch_model.eval()


    processed_slides = 0
    skipped_slides = 0
    failed_slides = 0

    for slide_idx, slide_id in enumerate(assigned_slide_ids, start=1):
        slide_start = time.perf_counter()
        region4k_path = region_4k_out / f"{slide_id}.pt"
        wsi_path = wsi_out / f"{slide_id}.pt"
        patches_path = patches_out / f"{slide_id}.pt"
        filtered_patches_path = Path(str(patches_out)+'_filtered') / f"{slide_id}.pt"
        meta_path = metadata_out / f"{slide_id}.json"

        outputs_exist = (
            region4k_path.exists()
            and wsi_path.exists()
            and meta_path.exists()
            and (not args.extract_patch_features or patches_path.exists())
        )
        if args.skip_existing and outputs_exist:
            skipped_slides += 1
            if slide_idx % max(1, args.log_every) == 0 or slide_idx == len(assigned_slide_ids):
                print(
                    f"[{worker_name}] progress {slide_idx}/{len(assigned_slide_ids)} "
                    f"processed={processed_slides} skipped={skipped_slides} failed={failed_slides}"
                )
            continue

        try:
            slide_path = wsi_index[slide_id]
            coords_4096 = load_coords(h5_4096_paths[slide_id])

            slide = openslide.OpenSlide(str(slide_path))
            try:
                level4096 = choose_level(slide, args.read_level_4096)

                patches_tensor, region_cls, valid_region_coords = extract_region_patch_features(
                    slide=slide,
                    coords_4096=coords_4096,
                    model256=model256,
                    model4k=model4k,
                    conch_model=conch_model,
                    conch_preprocess=preprocess,
                    batch_size=args.conch_batch_size,
                    transform=transform,
                    device=device,
                    patch_size=args.patch_size,
                    region_size=args.region_size,
                    level=level4096,
                    allow_edge_padding=args.allow_edge_padding,
                    extract_patch_features=args.extract_patch_features,
                    use_fp16=args.fp16,
                )
            finally:
                slide.close()

            if patches_tensor is not None:
                patches_feats = patches_tensor
            else:
                patches_feats = torch.empty((0, 384), dtype=torch.float32)

            if region_cls.numel() > 0:
                wsi_feat = region_cls.mean(dim=0)
            else:
                wsi_feat = torch.empty((192,), dtype=torch.float32)

            if args.extract_patch_features and patches_tensor is not None:
                save_tensor(patches_path, patches_feats)
            
            # Perform semantic patch filtering if reports_path is provided
            selected_features: Optional[Tensor] = None
            if args.reports_path is not None and patches_tensor is not None and patches_tensor.shape[0] > 0:
                #selected_features, selected_indices = semantic_patch_filtering(
                #                                                patches_tensor, args.reports_path, slide_id, 
                #                                                conch_model, device, args.filtering_window_size, 
                #                                                args.top_k)
                
                selected_features, selected_indices = semantic_patch_filtering2(
                                                                patches_tensor, args.reports_path, slide_id, 
                                                                conch_model, device, args.n_diss_features, 
                                                                args.top_k, mode=args.data_mode)
                
                save_tensor(filtered_patches_path, {"selected_features": selected_features, 
                                                    "selected_indices": selected_indices})

            save_tensor(region4k_path, region_cls)
            save_coords_h5(coords_region_out / f"{slide_id}.h5", valid_region_coords)
            save_tensor(wsi_path, wsi_feat)

            metadata = {
                "slide_id": slide_id,
                "slide_path": str(slide_path),
                "orig_num_region_coords_4096": int(len(coords_4096)),
                "num_valid_patch_feats": int(patches_feats.shape[0]),
                "num_valid_region_feats": int(region_cls.shape[0]),
                "patch_feat_dim": int(patches_feats.shape[1]) if patches_feats.ndim > 1 else 0,
                "region_feat_dim": int(region_cls.shape[1]) if region_cls.ndim > 1 else 0,
                "wsi_feat_dim": int(wsi_feat.shape[0]),
                "read_level_4096": int(level4096),
                "patch_size": int(args.patch_size),
                "region_size": int(args.region_size),
                "save_patches_features": bool(args.extract_patch_features),
                "patches_shape": list(patches_tensor.shape) if patches_tensor is not None else [],
                "filtered_patches_shape": list(selected_features.shape) if selected_features is not None else [],
                "patch_tensor_shape": list(patches_feats.shape),
                "region_tensor_shape": list(region_cls.shape),
                "wsi_tensor_shape": list(wsi_feat.shape),
                "extraction_time_sec": float(time.perf_counter() - slide_start),
            }
            save_json(meta_path, metadata)

            processed_slides += 1
            print(f"[{worker_name}] done {slide_id} in {time.perf_counter() - slide_start:.2f}s")
        except Exception as exc:
            failed_slides += 1
            print(f"[{worker_name}] failed {slide_id}: {exc}")

        if slide_idx % max(1, args.log_every) == 0 or slide_idx == len(assigned_slide_ids):
            print(
                f"[{worker_name}] progress {slide_idx}/{len(assigned_slide_ids)} "
                f"processed={processed_slides} skipped={skipped_slides} failed={failed_slides}"
            )

    print(
        f"[{worker_name}] finished processed={processed_slides}, "
        f"skipped={skipped_slides}, failed={failed_slides}"
    )
    return processed_slides, skipped_slides, failed_slides


def _worker_entrypoint(
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int],
    all_slide_ids: Sequence[str],
    args: argparse.Namespace,
    wsi_index: Dict[str, Path],
    h5_4096_paths: Dict[str, Path],
    patches_out: Path,
    region_4k_out: Path,
    wsi_out: Path,
    coords_region_out: Path,
    metadata_out: Path,
    result_queue,
) -> None:
    if gpu_ids:
        assigned_gpu = int(gpu_ids[rank % len(gpu_ids)])
        torch.cuda.set_device(assigned_gpu)
        device = torch.device(f"cuda:{assigned_gpu}")
    else:
        device = torch.device("cpu")

    assigned_slide_ids = list(all_slide_ids)[rank::world_size]
    processed, skipped, failed = _process_slides_for_worker(
        worker_rank=rank,
        world_size=world_size,
        device=device,
        assigned_slide_ids=assigned_slide_ids,
        args=args,
        wsi_index=wsi_index,
        h5_4096_paths=h5_4096_paths,
        patches_out=patches_out,
        region_4k_out=region_4k_out,
        wsi_out=wsi_out,
        coords_region_out=coords_region_out,
        metadata_out=metadata_out,
    )
    result_queue.put((rank, processed, skipped, failed))




def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    wsi_dir = Path(args.wsi_dir)
    h5_dir_4096 = Path(args.h5_dir_4096)
    hipt_repo = Path(args.hipt_repo)
    ckpt256 = Path(args.checkpoint256)
    ckpt4k = Path(args.checkpoint4k)

    trident_repo = Path(args.trident_repo)

    for required in [wsi_dir, h5_dir_4096, hipt_repo, ckpt256, ckpt4k, trident_repo]:
        if not required.exists():
            raise FileNotFoundError(f"Missing path: {required}")

    patches_out = out_dir / "patches"
    region_4k_out = out_dir / "region_4k"
    wsi_out = out_dir / "wsi"
    coords_region_out = out_dir / "coords_region4096_valid"
    metadata_out = out_dir / "metadata"

    for directory in [patches_out, region_4k_out, wsi_out, coords_region_out, metadata_out]:
        ensure_dir(directory)
    if args.extract_patch_features:
        ensure_dir(patches_out)
    
    wsi_index = build_wsi_index(wsi_dir, args.slide_ext)
    if not wsi_index:
        raise RuntimeError(f"No supported WSI files found in {wsi_dir}")

    h5_4096_paths = {p.stem: p for p in sorted(h5_dir_4096.glob("*.h5"))}

    common_ids = sorted(set(h5_4096_paths) & set(wsi_index))
    if args.process_single_slide is not None:
        common_ids = [sid for sid in common_ids if sid == args.process_single_slide]

    pre_skipped_slides = 0
    if args.skip_existing:
        remaining_ids: List[str] = []
        for sid in common_ids:
            if _outputs_exist_for_slide(
                slide_id=sid,
                patches_out=patches_out,
                region_4k_out=region_4k_out,
                wsi_out=wsi_out,
                coords_region_out=coords_region_out,
                metadata_out=metadata_out,
                args=args,
            ):
                pre_skipped_slides += 1
            else:
                remaining_ids.append(sid)
        common_ids = remaining_ids

        if pre_skipped_slides > 0:
            print(f"[Pre-filter] skipped already-processed slides before GPU assignment: {pre_skipped_slides}")

    if not common_ids:
        print("No pending slides to process after skip-existing pre-filter.")
        print("================ Run Summary ==================================")
        print("Processed slides: 0")
        print(f"Skipped slides: {pre_skipped_slides}")
        print("Failed slides: 0")
        print(f"Overall feature extraction time: {time.perf_counter() - overall_start:.2f} sec")
        print("===============================================================")
        print("Done.")
        return

    gpu_ids = _resolve_gpu_ids(args, device)
    n_devices = max(1, len(gpu_ids))
    workers_per_gpu = max(1, int(args.workers_per_gpu))
    world_size = n_devices * workers_per_gpu
    world_size = min(world_size, len(common_ids))

    if world_size < 1:
        world_size = 1

    print("================ Multiprocessing Configuration ================")
    print(f"Slides to process: {len(common_ids)}")
    print(f"Device mode: {'CUDA' if gpu_ids else 'CPU'}")
    if gpu_ids:
        print(f"GPUs in use: {gpu_ids}")
    print(f"Workers per GPU: {workers_per_gpu}")
    print(f"Total workers: {world_size}")
    print("===============================================================")

    if world_size == 1:
        single_device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
        processed_slides, skipped_slides, failed_slides = _process_slides_for_worker(
            worker_rank=0,
            world_size=1,
            device=single_device,
            assigned_slide_ids=common_ids,
            args=args,
            wsi_index=wsi_index,
            h5_4096_paths=h5_4096_paths,
            patches_out=patches_out,
            region_4k_out=region_4k_out,
            wsi_out=wsi_out,
            coords_region_out=coords_region_out,
            metadata_out=metadata_out,
        )
        skipped_slides += pre_skipped_slides
    else:
        mp.set_start_method("spawn", force=True)
        result_queue = mp.SimpleQueue()
        mp.spawn(
            _worker_entrypoint,
            args=(
                world_size,
                gpu_ids,
                common_ids,
                args,
                wsi_index,
                h5_4096_paths,
                patches_out,
                region_4k_out,
                wsi_out,
                coords_region_out,
                metadata_out,
                result_queue,
            ),
            nprocs=world_size,
            join=True,
        )

        processed_slides = 0
        skipped_slides = 0
        failed_slides = 0
        worker_summaries: List[Tuple[int, int, int, int]] = []
        for _ in range(world_size):
            rank, processed, skipped, failed = result_queue.get()
            worker_summaries.append((int(rank), int(processed), int(skipped), int(failed)))
            processed_slides += int(processed)
            skipped_slides += int(skipped)
            failed_slides += int(failed)

        skipped_slides += pre_skipped_slides

        worker_summaries.sort(key=lambda x: x[0])
        print("================ Worker Summary ===============================")
        for rank, processed, skipped, failed in worker_summaries:
            print(f"worker-{rank:02d}: processed={processed}, skipped={skipped}, failed={failed}")
        print("===============================================================")

    overall_elapsed = time.perf_counter() - overall_start
    print("================ Run Summary ==================================")
    print(f"Processed slides: {processed_slides}")
    print(f"Skipped slides: {skipped_slides}")
    print(f"Failed slides: {failed_slides}")
    print(f"Overall feature extraction time: {overall_elapsed:.2f} sec")
    print("===============================================================")
    print("Done.")


if __name__ == "__main__":
    main()
