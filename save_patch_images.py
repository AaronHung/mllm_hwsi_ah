import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    raise ImportError("Please install openslide-python: pip install openslide-python")

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
except ImportError:
    torch = None


WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}


def find_wsi_files(wsi_dir):
    wsi_dir = Path(wsi_dir)
    return sorted([p for p in wsi_dir.iterdir() if p.is_file() and p.suffix.lower() in WSI_EXTENSIONS])


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_region_coords(coord_path):
    """
    Supported formats:
      - .json
      - .npy
      - .csv
      - .h5 / .hdf5

    Returns:
      coords: list of dicts
             each dict has:
               {
                 "region_id": int or str,
                 "x": int,
                 "y": int
               }
    """
    coord_path = Path(coord_path)
    suffix = coord_path.suffix.lower()

    if suffix == ".json":
        with open(coord_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            coords = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    coords.append({
                        "region_id": item.get("region_id", i),
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                    })
                else:
                    x, y = item
                    coords.append({"region_id": i, "x": int(x), "y": int(y)})
            return coords

        if isinstance(data, dict):
            if "regions" in data:
                coords = []
                for i, item in enumerate(data["regions"]):
                    coords.append({
                        "region_id": item.get("region_id", i),
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                    })
                return coords

            if "coords" in data:
                coords = []
                for i, xy in enumerate(data["coords"]):
                    x, y = xy
                    coords.append({"region_id": i, "x": int(x), "y": int(y)})
                return coords

        raise ValueError(f"Unsupported JSON format in {coord_path}")

    elif suffix == ".npy":
        arr = np.load(coord_path, allow_pickle=True)

        if arr.dtype == object:
            coords = []
            for i, item in enumerate(arr):
                if isinstance(item, dict):
                    coords.append({
                        "region_id": item.get("region_id", i),
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                    })
                else:
                    x, y = item
                    coords.append({"region_id": i, "x": int(x), "y": int(y)})
            return coords

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [
                {"region_id": i, "x": int(arr[i, 0]), "y": int(arr[i, 1])}
                for i in range(arr.shape[0])
            ]

        raise ValueError(f"Unsupported NPY format in {coord_path}")

    elif suffix == ".csv":
        import csv
        coords = []
        with open(coord_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                region_id = row.get("region_id", i)
                coords.append({
                    "region_id": region_id,
                    "x": int(float(row["x"])),
                    "y": int(float(row["y"])),
                })
        return coords

    elif suffix in {".h5", ".hdf5"}:
        if h5py is None:
            raise ImportError("h5py is required to read .h5 coordinate files")

        with h5py.File(coord_path, "r") as f:
            if "coords" not in f:
                raise ValueError(f"Expected dataset 'coords' in {coord_path}")

            arr = f["coords"][:]
            region_ids = f["region_ids"][:] if "region_ids" in f else None

            coords = []
            for i in range(arr.shape[0]):
                rid = region_ids[i] if region_ids is not None else i
                if isinstance(rid, bytes):
                    rid = rid.decode("utf-8")
                coords.append({
                    "region_id": rid,
                    "x": int(arr[i, 0]),
                    "y": int(arr[i, 1]),
                })
            return coords

    else:
        raise ValueError(f"Unsupported coordinate file format: {coord_path}")


def _to_int_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().flatten().tolist()
    elif isinstance(x, np.ndarray):
        x = x.flatten().tolist()
    return [int(v) for v in x]


def load_selected_patch_indices(indices_path):
    """
    Supported formats:
      - .pt   (torch.load)
      - .json
      - .npy
      - .h5 / .hdf5

    For .pt:
      expects a dictionary containing key "selected_indices"

    Returns:
      dict mapping region_id (as str) -> list[int]
    """
    indices_path = Path(indices_path)
    suffix = indices_path.suffix.lower()

    if suffix == ".pt":
        if torch is None:
            raise ImportError("torch is required to read .pt selected-index files")

        obj = torch.load(indices_path, map_location="cpu")

        if not isinstance(obj, dict):
            raise ValueError(f"Expected .pt file to contain a dict, got {type(obj)}")

        if "selected_indices" not in obj:
            raise ValueError(f"Expected key 'selected_indices' in {indices_path}")

        selected = obj["selected_indices"]

        # Case 1: dict[region_id] -> indices
        if isinstance(selected, dict):
            out = {}
            for k, v in selected.items():
                out[str(k)] = _to_int_list(v)
            return out

        # Case 2: list/tuple where position = region_id/order
        if isinstance(selected, (list, tuple)):
            out = {}
            for i, v in enumerate(selected):
                out[str(i)] = _to_int_list(v)
            return out

        # Case 3: tensor/ndarray
        if (torch is not None and isinstance(selected, torch.Tensor)) or isinstance(selected, np.ndarray):
            if len(selected.shape) == 1:
                return {"0": _to_int_list(selected)}
            elif len(selected.shape) == 2:
                out = {}
                for i in range(selected.shape[0]):
                    out[str(i)] = _to_int_list(selected[i])
                return out

        raise ValueError(
            f"Unsupported format for 'selected_indices' in {indices_path}. "
            f"Supported: dict, list/tuple, tensor, ndarray."
        )

    elif suffix == ".json":
        with open(indices_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return {str(k): [int(vv) for vv in v] for k, v in data.items()}

        if isinstance(data, list):
            out = {}
            for item in data:
                rid = str(item["region_id"])
                out[rid] = [int(v) for v in item["indices"]]
            return out

        raise ValueError(f"Unsupported JSON format in {indices_path}")

    elif suffix == ".npy":
        arr = np.load(indices_path, allow_pickle=True).item()
        return {str(k): [int(vv) for vv in v] for k, v in arr.items()}

    elif suffix in {".h5", ".hdf5"}:
        if h5py is None:
            raise ImportError("h5py is required to read .h5 index files")

        out = {}
        with h5py.File(indices_path, "r") as f:
            for key in f.keys():
                out[str(key)] = [int(v) for v in f[key][:]]
        return out

    else:
        raise ValueError(f"Unsupported selected-indices file format: {indices_path}")


def crop_patch_from_region(region_img, patch_idx, patch_size):
    width, height = region_img.size
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size
    total_patches = patches_per_row * patches_per_col

    if patch_idx < 0 or patch_idx >= total_patches:
        raise ValueError(
            f"patch_idx={patch_idx} is out of range for region of size "
            f"{width}x{height} with patch_size={patch_size}. "
            f"Valid range is [0, {total_patches - 1}]."
        )

    row = patch_idx // patches_per_row
    col = patch_idx % patches_per_row

    left = col * patch_size
    upper = row * patch_size
    right = left + patch_size
    lower = upper + patch_size

    return region_img.crop((left, upper, right, lower))


def choose_regions(region_coords, max_regions_per_wsi, selection_mode="first", seed=123):
    if max_regions_per_wsi is None or max_regions_per_wsi >= len(region_coords):
        return region_coords

    if selection_mode == "first":
        return region_coords[:max_regions_per_wsi]

    if selection_mode == "random":
        rng = random.Random(seed)
        return rng.sample(region_coords, max_regions_per_wsi)

    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def read_region_image(slide, x, y, region_size, level):
    return slide.read_region((x, y), level, (region_size, region_size)).convert("RGB")


def get_matching_file(base_dir, stem, allowed_suffixes=None):
    base_dir = Path(base_dir)
    allowed_suffixes = allowed_suffixes or {".json", ".npy", ".csv", ".h5", ".hdf5", ".pt"}

    for ext in allowed_suffixes:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_single_wsi(
    wsi_path,
    coord_path,
    indices_path,
    output_root,
    region_size,
    patch_size,
    mag_level,
    max_regions_per_wsi,
    selection_mode,
    seed,
):
    
    mag_to_patch_level = {40: 0, 20: 1, 10: 2, 5: 3}
    level = mag_to_patch_level.get(mag_level, 0)

    slide_id = wsi_path.stem
    print(f"\nProcessing WSI: {slide_id}")

    region_coords = load_region_coords(coord_path)
    selected_indices = load_selected_patch_indices(indices_path)

    selected_regions = choose_regions(
        region_coords=region_coords,
        max_regions_per_wsi=max_regions_per_wsi,
        selection_mode=selection_mode,
        seed=seed,
    )

    slide_out_dir = Path(output_root) / slide_id
    ensure_dir(slide_out_dir)

    slide = openslide.OpenSlide(str(wsi_path))

    saved_region_count = 0

    for region_info in selected_regions:
        region_id = str(region_info["region_id"])
        x = int(region_info["x"])
        y = int(region_info["y"])

        if region_id not in selected_indices:
            print(f"  Skipping region_id={region_id}: no selected patch indices found")
            continue

        region_img = read_region_image(slide, x, y, region_size, level)

        region_dir = slide_out_dir / f"region_{region_id}"
        patches_dir = region_dir / "patches"
        ensure_dir(region_dir)
        ensure_dir(patches_dir)

        region_img.save(region_dir / "region.png")

        patch_ids = selected_indices[region_id]
        for patch_idx in patch_ids:
            try:
                patch_img = crop_patch_from_region(region_img, patch_idx, patch_size)
                patch_img.save(patches_dir / f"{patch_idx}.png")
            except Exception as e:
                print(f"  Failed patch {patch_idx} in region {region_id}: {e}")

        meta = {
            "slide_id": slide_id,
            "region_id": region_id,
            "x": x,
            "y": y,
            "region_size": region_size,
            "patch_size": patch_size,
            "level": level,
            "selected_patch_indices": patch_ids,
        }
        with open(region_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        saved_region_count += 1

    slide.close()
    print(f"Saved {saved_region_count} regions for {slide_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Save region images and selected patch images for WSIs."
    )
    parser.add_argument("--wsi_dir", type=str, required=True, help="Folder containing WSI files")
    parser.add_argument("--coords_dir", type=str, required=True,
                        help="Folder containing per-WSI region coordinate files")
    parser.add_argument("--indices_dir", type=str, required=True,
                        help="Folder containing per-WSI selected patch index files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root folder")

    parser.add_argument("--region_size", type=int, required=True,
                        help="Region size in pixels, e.g. 4096")
    parser.add_argument("--patch_size", type=int, required=True,
                        help="Patch size in pixels, e.g. 256")
    parser.add_argument("--mag_level", type=int, default=20,
                        help="OpenSlide magnification level to read from")
    parser.add_argument("--max_wsi", type=int, default=None,
                        help="Maximum number of WSIs to process")
    parser.add_argument("--max_regions_per_wsi", type=int, default=None,
                        help="Maximum number of regions to save per WSI")
    parser.add_argument("--selection_mode", type=str, default="first", choices=["first", "random"],
                        help="How to select regions when max_regions_per_wsi is set")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()

    if args.region_size % args.patch_size != 0:
        raise ValueError(
            f"region_size ({args.region_size}) must be divisible by patch_size ({args.patch_size})"
        )

    ensure_dir(args.output_dir)

    wsi_files = find_wsi_files(args.wsi_dir)
    if len(wsi_files) == 0:
        raise FileNotFoundError(f"No WSI files found in {args.wsi_dir}")

    if args.max_wsi is not None:
        wsi_files = wsi_files[:args.max_wsi]

    print(f"Found {len(wsi_files)} WSI(s) to process")

    for wsi_path in wsi_files:
        slide_id = wsi_path.stem

        coord_path = get_matching_file(args.coords_dir, slide_id)
        indices_path = get_matching_file(
            args.indices_dir,
            slide_id,
            allowed_suffixes={".pt", ".json", ".npy", ".h5", ".hdf5"},
        )

        if coord_path is None:
            print(f"Skipping {slide_id}: coordinate file not found")
            continue

        if indices_path is None:
            print(f"Skipping {slide_id}: selected patch index file not found")
            continue

        process_single_wsi(
            wsi_path=wsi_path,
            coord_path=coord_path,
            indices_path=indices_path,
            output_root=args.output_dir,
            region_size=args.region_size,
            patch_size=args.patch_size,
            mag_level=args.mag_level,
            max_regions_per_wsi=args.max_regions_per_wsi,
            selection_mode=args.selection_mode,
            seed=args.seed,
        )
    
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()