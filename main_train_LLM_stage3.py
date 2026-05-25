#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model 
from mllm_hwsi import VLProjectorConfig, MLLMHWSI
import main_utils as _mu
from data_utils import (
    find_slide_ids,
    load_slide_feature_quadruplet,
    set_seed,
)

# =========================================================
# Dataset
# =========================================================

class WSITrainDataset(Dataset):
    """
    One training sample per slide_id.

    Uses:
      - read_slide_report(...) from main_inference.py
      - load_slide_feature_quadruplet(...) from main_inference.py

    Returns:
      {
        "slide_id": str,
        "question": str,
        "answer": str,
        "wsi": Tensor[192],
        "region": Tensor[R, 192],
        "patch": Tensor[R, C, 512],   where C can vary per region before collation
        "cell": Tensor[R, C, 384],    where C can vary per region before collation
      }

    Notes
    -----
    - Raw loaded shapes:
        wsi    (1, 192)
        region (1, R, 192)
        patch  (1, R, 16, 512)
        cell   (1, R, 16, 384)

    - For each region, cell vectors that are empty/all-zero are removed.
    - If a region has no valid cell vectors left, that entire region is dropped
      from region/patch/cell so all three stay aligned in R.
    """
    def __init__(
        self,
        wsi_dir: str,
        conversation_json: str,
        wsi_feat_dir: str,
        region_feat_dir: str,
        patch_feat_dir: str,
        cell_feat_dir: str,
        cell_pt_filename: str,
        strict_report_match: bool = False,
    ):
        self.wsi_dir = wsi_dir
        self.reports_path = Path(conversation_json)
        self.wsi_feat_dir = wsi_feat_dir
        self.region_feat_dir = region_feat_dir
        self.patch_feat_dir = patch_feat_dir
        self.cell_feat_dir = cell_feat_dir
        self.cell_pt_filename = cell_pt_filename
        self.strict_report_match = strict_report_match

        slide_ids = find_slide_ids(wsi_dir)
        slide_id_set = set(slide_ids)

        with self.reports_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"conversation_json must contain a list, got {type(records)}")

        samples: List[Dict[str, str]] = []
        skipped: List[str] = []

        for rec in records:
            try:
                if not isinstance(rec, dict):
                    raise ValueError("Each conversation record must be a dict")

                slide_id = self._resolve_slide_id(rec, slide_id_set)
                if slide_id is None:
                    raise KeyError("Could not resolve slide_id from conversation record")

                question, t_answer = self._extract_qa_pair(rec)
                if not question or not t_answer:
                    raise KeyError(f"Empty question/answer for {slide_id}")

                # Keep only slides that have all required extracted features.
                wsi_feat_path = os.path.join(self.wsi_feat_dir, f"{slide_id}.pt")
                region_feat_path = os.path.join(self.region_feat_dir, f"{slide_id}.pt")
                patch_feat_path = os.path.join(self.patch_feat_dir, f"{slide_id}.pt")
                cell_feat_path = os.path.join(self.cell_feat_dir, slide_id, self.cell_pt_filename)

                missing = [
                    p for p in [wsi_feat_path, region_feat_path, patch_feat_path, cell_feat_path]
                    if not os.path.isfile(p)
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Missing extracted features for {slide_id}: {', '.join(missing)}"
                    )

                samples.append({
                    "slide_id": slide_id,
                    "question": question,
                    "answer": t_answer,
                })
            except Exception:
                sid = None
                if isinstance(rec, dict):
                    sid = rec.get("id") or rec.get("image") or rec.get("question_id")
                skipped.append(str(sid))
                if strict_report_match:
                    raise

        self.samples = samples
        self.skipped = skipped

        if len(self.samples) == 0:
            raise RuntimeError("No valid training samples found.")

        print(f"[Dataset] usable slides: {len(self.samples)}")
        print(f"[Dataset] skipped slides: {len(self.skipped)}")

    @staticmethod
    def _strip_image_token(text: str) -> str:
        text = str(text).replace("<image>", " ").replace("<Image>", " ")
        return " ".join(text.split()).strip()

    @classmethod
    def _extract_qa_pair(cls, rec: Dict[str, Any]) -> Tuple[str, str]:
        conv = rec.get("conversations", [])
        if not isinstance(conv, list) or len(conv) == 0:
            raise ValueError("Missing or empty 'conversations' list")

        human_text: Optional[str] = None
        gpt_text: Optional[str] = None

        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).strip().lower()
            value = cls._strip_image_token(str(turn.get("value", "")))
            if role == "human" and value and human_text is None:
                human_text = value
            if role in {"gpt", "assistant"} and value and gpt_text is None:
                gpt_text = value
            if human_text is not None and gpt_text is not None:
                break

        if human_text is None or gpt_text is None:
            raise ValueError("Could not extract human/gpt pair from 'conversations'")

        return human_text, gpt_text

    @staticmethod
    def _resolve_slide_id(rec: Dict[str, Any], slide_id_set: set) -> Optional[str]:
        image_field = rec.get("image")
        if image_field:
            stem = Path(str(image_field)).stem
            if stem in slide_id_set:
                return stem

        # Fallback: match by containment in id/question_id/image strings.
        haystacks = [
            str(rec.get("id", "")),
            str(rec.get("question_id", "")),
            str(rec.get("image", "")),
        ]
        for sid in slide_id_set:
            if any(sid in h for h in haystacks):
                return sid
        return None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _valid_cell_mask(cell_region: torch.Tensor) -> torch.Tensor:
        """
        cell_region: (16, 384)
        Returns:
          valid_mask: (16,) bool

        A cell token is valid if it is not empty/all-zero.
        """
        if cell_region.ndim != 2 or cell_region.shape[-1] != 384:
            raise ValueError(f"Expected cell_region shape (K, 384), got {tuple(cell_region.shape)}")

        # Treat exactly-zero rows as invalid.
        # abs().sum(-1) > 0 is robust and simple.
        valid_mask = cell_region.abs().sum(dim=-1) > 0
        return valid_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        slide_id = item["slide_id"]

        wsi, region, patch, cell = load_slide_feature_quadruplet(
            slide_id=slide_id,
            wsi_feat_dir=self.wsi_feat_dir,
            region_feat_dir=self.region_feat_dir,
            patch_feat_dir=self.patch_feat_dir,
            cell_feat_dir=self.cell_feat_dir,
            cell_pt_filename=self.cell_pt_filename,
        )

        # Raw loaded shapes:
        #   wsi    (1, 192)
        #   region (1, R, 192)
        #   patch  (1, R, 16, 512)
        #   cell   (1, R, 16, 384)
        wsi = wsi.squeeze(0)         # (192,)
        region = region.squeeze(0)   # (R, 192)
        patch = patch.squeeze(0)     # (R, 16, 512)
        cell = cell.squeeze(0)       # (R, 16, 384)

        if region.ndim != 2 or patch.ndim != 3 or cell.ndim != 3:
            raise ValueError(
                f"Unexpected shapes after squeeze for slide {slide_id}: "
                f"region={tuple(region.shape)}, patch={tuple(patch.shape)}, cell={tuple(cell.shape)}"
            )

        if not (region.shape[0] == patch.shape[0] == cell.shape[0]):
            raise ValueError(
                f"Mismatched region counts for slide {slide_id}: "
                f"region={region.shape[0]}, patch={patch.shape[0]}, cell={cell.shape[0]}"
            )

        kept_region_feats: List[torch.Tensor] = []
        kept_patch_feats: List[torch.Tensor] = []
        kept_cell_feats: List[torch.Tensor] = []

        num_regions = region.shape[0]
        for r_idx in range(num_regions):
            region_feat = region[r_idx]          # (192,)
            patch_region = patch[r_idx]          # (16, 512)
            cell_region = cell[r_idx]            # (16, 384)

            valid_cells = self._valid_cell_mask(cell_region)   # (16,)
            num_valid = int(valid_cells.sum().item())

            # Drop this entire region if no valid cells remain
            if num_valid == 0:
                continue

            filtered_patch_region = patch_region[valid_cells]  # (C, 512)
            filtered_cell_region = cell_region[valid_cells]    # (C, 384)

            kept_region_feats.append(region_feat)
            kept_patch_feats.append(filtered_patch_region)
            kept_cell_feats.append(filtered_cell_region)

        # If all regions are invalid for this slide, fail fast.
        # This preserves the invariant that region/patch/cell have the same R.
        if len(kept_region_feats) == 0:
            raise RuntimeError(
                f"Slide {slide_id} has no valid regions after filtering zero/empty cell features."
            )

        region_out = torch.stack(kept_region_feats, dim=0)  # (R_keep, 192)

        # Note:
        # patch/cell now have variable C per region, so keep them as Python lists here.
        # They will be padded inside collate_wsis.
        return {
            "slide_id": slide_id,
            "question": item["question"],
            "answer": item["answer"],
            "wsi": wsi,                       # (192,)
            "region": region_out,            # (R_keep, 192)
            "patch_list": kept_patch_feats,  # list of (C_r, 512)
            "cell_list": kept_cell_feats,    # list of (C_r, 384)
        }


collate_wsis = _mu.collate_wsis


# =========================================================
# Losses
# =========================================================

masked_mean = _mu.masked_mean


sentence_split = _mu.sentence_split


make_phrase_chunks = _mu.make_phrase_chunks


pool_text_embedding_from_string = _mu.pool_text_embedding_from_string


build_hierarchical_text_reps = _mu.build_hierarchical_text_reps


info_nce_bidirectional = _mu.info_nce_bidirectional


hierarchical_contrastive_loss = _mu.hierarchical_contrastive_loss


cross_scale_consistency_loss = _mu.cross_scale_consistency_loss


# =========================================================
# Training helpers
# =========================================================

def freeze_non_llm(model: MLLMHWSI) -> None:
    for p in model.vl_projector.parameters():
        p.requires_grad = False


get_visual_scale_reps = _mu.get_visual_scale_reps


build_training_batch_inputs = _mu.build_training_batch_inputs


def save_llm_checkpoint(
    save_path: str,
    model: MLLMHWSI,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_val_loss: float,
    best_epoch: int,
    cfg: VLProjectorConfig,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "llm_state_dict": model.llm.state_dict(),
        "vl_projector_state_dict": model.vl_projector.state_dict(),
        "pathology_encoder_state_dict": model.vl_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "projector_cfg": asdict(cfg),
    }
    torch.save(payload, save_path)


save_best_model_metadata = _mu.save_best_model_metadata


format_hms = _mu.format_hms


# =========================================================
# Train / Eval
# =========================================================

def run_epoch(
    model: MLLMHWSI,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    device: torch.device,
    grad_accum_steps: int,
    temperature: float,
    lambda_lm: float,
    lambda_contrastive: float,
    lambda_consistency: float,
    max_grad_norm: float,
    autocast_enabled: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.vl_projector.train(is_train)
    model.llm.train(is_train)

    autocast_enabled = bool(autocast_enabled and device.type == "cuda")
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    scaler_enabled = bool(autocast_enabled and (torch.float16 not in trainable_dtypes))
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    total_loss = 0.0
    total_lm = 0.0
    n_steps = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        wsi = batch["wsi"].to(device)
        region = batch["region"].to(device)
        patch = batch["patch"].to(device)
        cell = batch["cell"].to(device)
        region_mask = batch["region_mask"].to(device)
        cell_mask = batch["cell_mask"].to(device)

        dtype = next(model.llm.parameters()).dtype

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            slide_tokens, _ = get_visual_scale_reps(
                model=model,
                wsi=wsi.to(dtype=dtype),
                region=region.to(dtype=dtype),
                patch=patch.to(dtype=dtype),
                cell=cell.to(dtype=dtype),
                region_mask=region_mask,
                cell_mask=cell_mask,
                device=device
            )

            inputs_embeds, attention_mask, labels = build_training_batch_inputs(
                model=model,
                slide_tokens=slide_tokens,
                questions=batch["questions"],
                answers=batch["answers"],
                device=device,
                dtype=dtype,
                prompt_builder=build_prompt,
            )

            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            lm_loss = outputs.loss
            loss = lambda_lm * lm_loss

        if is_train:
            loss_for_backward = loss / grad_accum_steps
            if scaler_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (step + 1) % grad_accum_steps == 0:
                if scaler_enabled:
                    scaler.unscale_(optimizer)

                trainable = [
                    p for p in model.parameters() if p.requires_grad and p.grad is not None
                ]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                did_optimizer_step = False
                if scaler_enabled:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_optimizer_step = scaler.get_scale() >= prev_scale
                else:
                    optimizer.step()
                    did_optimizer_step = True
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and did_optimizer_step:
                    scheduler.step()

        total_loss += float(loss.detach().item())
        total_lm += float(lm_loss.detach().item())
        n_steps += 1

    if is_train and (n_steps % grad_accum_steps) != 0:
        if scaler_enabled:
            scaler.unscale_(optimizer)
        trainable = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        did_optimizer_step = False
        if scaler_enabled:
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_optimizer_step = scaler.get_scale() >= prev_scale
        else:
            optimizer.step()
            did_optimizer_step = True
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and did_optimizer_step:
            scheduler.step()

    return {
        "loss": total_loss / max(n_steps, 1),
        "lm_loss": total_lm / max(n_steps, 1),
        "contrastive_loss": 0.0,
        "consistency_loss": 0.0,
    }


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--wsi_dir", type=str, required=True)
    parser.add_argument("--conversation_json", type=str, required=True)
    parser.add_argument("--wsi_feat_dir", type=str, required=True)
    parser.add_argument("--region_feat_dir", type=str, required=True)
    parser.add_argument("--patch_feat_dir", type=str, required=True)
    parser.add_argument("--cell_feat_dir", type=str, required=True)
    parser.add_argument("--cell_pt_filename", type=str, required=True)
    parser.add_argument(
        "--strict_report_match",
        action="store_true",
        help="Fail fast if any slide has missing conversation fields or required feature files.",
    )
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to stage-2 projector checkpoint")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Projector config
    parser.add_argument("--llm_dim", type=int, default=3584)
    parser.add_argument("--proj_hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_slide_tokens", type=int, default=64)
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--no_change",
        type=int,
        default=5,
        help="Stop training after this many consecutive epochs without a new best val loss. Set <= 0 to disable.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--llm_lr",
        type=float,
        default=None,
        help="Optional LLM learning rate. If unset, falls back to --lr.",
    )
    parser.add_argument(
        "--projector_lr",
        type=float,
        default=None,
        help="Optional VL projector learning rate. If unset, falls back to --lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--autocast_enabled",
        "--use_amp",
        action="store_true",
        help="Enable torch.autocast mixed precision (legacy alias: --use_amp)",
    )

    # Loss weights
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_lm", type=float, default=1.0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.0)

    # Split
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Save
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="stage3_model.pt")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float16 if (device.type == "cuda") else torch.float32

    cfg = VLProjectorConfig(
        llm_dim=args.llm_dim,
        hidden_dim=args.proj_hidden_dim,
        dropout=args.dropout,
        use_layernorm=True,
        num_query_tokens=args.num_slide_tokens,
        projector_type=args.projector_type,
    )

    dataset = WSITrainDataset(
        wsi_dir=args.wsi_dir,
        conversation_json=args.conversation_json,
        wsi_feat_dir=args.wsi_feat_dir,
        region_feat_dir=args.region_feat_dir,
        patch_feat_dir=args.patch_feat_dir,
        cell_feat_dir=args.cell_feat_dir,
        cell_pt_filename=args.cell_pt_filename,
        strict_report_match=args.strict_report_match,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio)) if n_total > 1 else 0
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError("Validation split too large; no training samples remain.")

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_wsis,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_wsis,
    ) if n_val > 0 else None

    model = MLLMHWSI(
        model_name=args.model_name,
        projector_cfg=cfg,
        device=str(device),
        torch_dtype=dtype,
    )
    lora_targets = _mu.apply_lora_to_llm(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        print_trainable=True,
    )

    stage2_ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
    if isinstance(stage2_ckpt, dict) and "vl_projector_state_dict" in stage2_ckpt:
        stage2_state = stage2_ckpt["vl_projector_state_dict"]
    else:
        stage2_state = stage2_ckpt.get("state_dict", stage2_ckpt)
    missing, unexpected = model.vl_projector.load_state_dict(stage2_state, strict=False)
    print(f"[Stage2 VL Projector Load] missing keys: {missing}")
    print(f"[Stage2 VL Projector Load] unexpected keys: {unexpected}")

    llm_lr = args.llm_lr if args.llm_lr is not None else args.lr
    projector_lr = args.projector_lr if args.projector_lr is not None else args.lr

    llm_params = [p for p in model.llm.parameters() if p.requires_grad]
    projector_params = [p for p in model.vl_projector.parameters() if p.requires_grad]

    param_groups = []
    if llm_params:
        param_groups.append({"params": llm_params, "lr": llm_lr, "name": "llm"})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": projector_lr, "name": "vl_projector"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_history: List[Dict[str, float]] = []

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state_dict = ckpt["llm_state_dict"] if "llm_state_dict" in ckpt else ckpt
        missing, unexpected = model.llm.load_state_dict(state_dict, strict=False)
        print(f"[Resume] missing keys: {missing}")
        print(f"[Resume] unexpected keys: {unexpected}")

        projector_state = None
        if "vl_projector_state_dict" in ckpt:
            projector_state = ckpt["vl_projector_state_dict"]
        elif "pathology_encoder_state_dict" in ckpt:
            projector_state = ckpt["pathology_encoder_state_dict"]

        if projector_state is not None:
            pe_missing, pe_unexpected = model.vl_projector.load_state_dict(
                projector_state,
                strict=False,
            )
            print(f"[Resume VL Projector] missing keys: {pe_missing}")
            print(f"[Resume VL Projector] unexpected keys: {pe_unexpected}")

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))

    os.makedirs(args.save_dir, exist_ok=True)
    merged_best_hf_dir = os.path.join(args.save_dir, f"best_hf_{Path(args.save_name).stem}")
    merged_last_hf_dir = os.path.join(args.save_dir, f"last_hf_{Path(args.save_name).stem}")
    best_ckpt_path = os.path.join(args.save_dir, f"best_{Path(args.save_name).name}")
    history_json_path = os.path.join(args.save_dir, f"train_history_{Path(args.save_name).stem}.json")
    epochs_without_improvement = max(start_epoch - best_epoch, 0) if best_epoch > 0 else 0

    def _export_current_merged_hf(out_dir: str, tag: str) -> None:
        if not hasattr(model.llm, "merge_and_unload"):
            print(f"[Merge] Skipped {tag} export: model.llm is not LoRA-wrapped.")
            return

        export_model = MLLMHWSI(
            model_name=args.model_name,
            projector_cfg=cfg,
            device=str(device),
            dtype=dtype,
        )
        _mu.apply_lora_to_llm(
            model=export_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            print_trainable=True,
        )
        export_model.llm.load_state_dict(model.llm.state_dict(), strict=False)
        export_model.vl_projector.load_state_dict(model.vl_projector.state_dict(), strict=False)

        print(f"[Merge] Merging LoRA adapters for {tag} export...")
        export_model.llm = export_model.llm.merge_and_unload()
        export_model.save_pretrained(out_dir)
        print(f"[Merge] saved {tag} merged HF model to {out_dir}")

    if os.path.isfile(history_json_path):
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                hist_payload = json.load(f)
            if isinstance(hist_payload.get("epoch_history"), list):
                epoch_history = hist_payload["epoch_history"]
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    print(f"[Train] train samples: {n_train}")
    print(f"[Train] val samples  : {n_val}")
    print(f"[Train] device       : {device}")
    print(f"[Train] dtype        : {dtype}")
    print(f"[Train] LoRA         : enabled")
    print(f"[Train] lora_r       : {args.lora_r}")
    print(f"[Train] lora_alpha   : {args.lora_alpha}")
    print(f"[Train] lora_dropout : {args.lora_dropout}")
    print(f"[Train] lora_targets : {lora_targets}")
    print(f"[Train] llm_lr       : {llm_lr}")
    print(f"[Train] projector_lr : {projector_lr}")
    print(f"[Train] total steps  : {total_update_steps}")
    print(f"[Train] warmup steps : {warmup_steps}")

    total_train_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            temperature=args.temperature,
            lambda_lm=args.lambda_lm,
            lambda_contrastive=args.lambda_contrastive,
            lambda_consistency=args.lambda_consistency,
            max_grad_norm=args.max_grad_norm,
            autocast_enabled=args.autocast_enabled,
        )

        if val_loader is not None:
            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scheduler=None,
                    device=device,
                    grad_accum_steps=1,
                    temperature=args.temperature,
                    lambda_lm=args.lambda_lm,
                    lambda_contrastive=args.lambda_contrastive,
                    lambda_consistency=args.lambda_consistency,
                    max_grad_norm=args.max_grad_norm,
                    autocast_enabled=args.autocast_enabled,
                )
        else:
            val_stats = {"loss": train_stats["loss"]}

        global_step += math.ceil(len(train_loader) / args.grad_accum_steps)
        epoch_elapsed = time.perf_counter() - epoch_start

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_lm={train_stats['lm_loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"epoch_time={format_hms(epoch_elapsed)}"
        )

        epoch_history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_stats["loss"]),
                "train_lm_loss": float(train_stats["lm_loss"]),
                "val_loss": float(val_stats["loss"]),
            }
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_llm_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                cfg=cfg,
            )
            _export_current_merged_hf(merged_best_hf_dir, "best")
            print(
                f"[Checkpoint] updated best model on disk at epoch {best_epoch} "
                f"(resume checkpoint: {best_ckpt_path})"
            )

            save_best_model_metadata(
                json_path=history_json_path,
                model_path=merged_best_hf_dir,
                checkpoint_path=best_ckpt_path,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                epoch_history=epoch_history,
            )

            total_train_elapsed = time.perf_counter() - total_train_start
            print(f"[Time] total_training_time={format_hms(total_train_elapsed)}")
        else:
            epochs_without_improvement += 1
            save_best_model_metadata(
                json_path=history_json_path,
                model_path=merged_best_hf_dir,
                checkpoint_path=best_ckpt_path if os.path.isfile(best_ckpt_path) else None,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                epoch_history=epoch_history,
            )
            if args.no_change > 0:
                print(
                    f"[EarlyStop] no improvement for {epochs_without_improvement}/{args.no_change} epoch(s)"
                )
                if epochs_without_improvement >= args.no_change:
                    print(
                        f"[EarlyStop] stopping after epoch {epoch + 1} "
                        f"with best epoch {best_epoch} and val_loss {best_val_loss:.4f}"
                    )
                    break

    _export_current_merged_hf(merged_last_hf_dir, "last")

    if best_epoch == 0 and not os.path.isdir(merged_best_hf_dir):
        save_llm_checkpoint(
            save_path=best_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=max(start_epoch, 0),
            step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            cfg=cfg,
        )
        _export_current_merged_hf(merged_best_hf_dir, "best")
        best_epoch = min(max(start_epoch, 0) + 1, len(epoch_history)) if epoch_history else 0
        if epoch_history:
            best_val_loss = float(epoch_history[-1]["val_loss"])

    save_best_model_metadata(
        json_path=history_json_path,
        model_path=merged_best_hf_dir,
        checkpoint_path=best_ckpt_path if os.path.isfile(best_ckpt_path) else None,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epoch_history=epoch_history,
    )

    print(f"[Done] merged best HF model dir: {merged_best_hf_dir}")
    print(f"[Done] merged last HF model dir: {merged_last_hf_dir}")
    print(f"[Done] training history json: {history_json_path}")


if __name__ == "__main__":
    main()