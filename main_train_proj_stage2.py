#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from mllm_hwsi import VLProjectorConfig, MLLMHWSIQWEN
from data_utils import (
    find_slide_ids,
    read_slide_report,
    load_slide_feature_quadruplet,
    build_prompt,
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
        report_json: str,
        wsi_feat_dir: str,
        region_feat_dir: str,
        patch_feat_dir: str,
        cell_feat_dir: str,
        cell_pt_filename: str,
        strict_report_match: bool = False,
    ):
        self.wsi_dir = wsi_dir
        self.reports_path = Path(report_json)
        self.wsi_feat_dir = wsi_feat_dir
        self.region_feat_dir = region_feat_dir
        self.patch_feat_dir = patch_feat_dir
        self.cell_feat_dir = cell_feat_dir
        self.cell_pt_filename = cell_pt_filename
        self.strict_report_match = strict_report_match

        slide_ids = find_slide_ids(wsi_dir)

        samples: List[Dict[str, str]] = []
        skipped: List[str] = []

        for slide_id in slide_ids:
            try:
                question, t_answer = read_slide_report(self.reports_path, slide_id)
                if not question or not t_answer:
                    raise KeyError(f"Empty question/T-answer for {slide_id}")

                # Cheap existence check for feature files
                _ = os.path.join(self.wsi_feat_dir, f"{slide_id}.pt")
                _ = os.path.join(self.region_feat_dir, f"{slide_id}.pt")
                _ = os.path.join(self.patch_feat_dir, f"{slide_id}.pt")
                _ = os.path.join(self.cell_feat_dir, slide_id, self.cell_pt_filename)

                samples.append({
                    "slide_id": slide_id,
                    "question": question,
                    "answer": t_answer,
                })
            except Exception:
                skipped.append(slide_id)
                if strict_report_match:
                    raise

        self.samples = samples
        self.skipped = skipped

        if len(self.samples) == 0:
            raise RuntimeError("No valid training samples found.")

        print(f"[Dataset] usable slides: {len(self.samples)}")
        print(f"[Dataset] skipped slides: {len(self.skipped)}")

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


def collate_wsis(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable number of regions R and variable valid-cell count C across batch.

    Output shapes:
      wsi         : (B, 192)
      region      : (B, Rmax, 192)
      patch       : (B, Rmax, Cmax, 512)
      cell        : (B, Rmax, Cmax, 384)
      region_mask : (B, Rmax)          1=valid region, 0=pad
      cell_mask   : (B, Rmax, Cmax)    1=valid cell slot, 0=pad
    """
    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_wsis.")

    bsz = len(batch)
    r_max = max(item["region"].shape[0] for item in batch)

    # max valid-cell count across all kept regions in the batch
    c_max = 1
    for item in batch:
        for cell_region in item["cell_list"]:
            c_max = max(c_max, int(cell_region.shape[0]))

    wsi = torch.stack([item["wsi"] for item in batch], dim=0)  # (B, 192)

    region = torch.zeros(bsz, r_max, 192, dtype=torch.float32)
    patch = torch.zeros(bsz, r_max, c_max, 512, dtype=torch.float32)
    cell = torch.zeros(bsz, r_max, c_max, 384, dtype=torch.float32)

    region_mask = torch.zeros(bsz, r_max, dtype=torch.bool)
    cell_mask = torch.zeros(bsz, r_max, c_max, dtype=torch.bool)

    slide_ids: List[str] = []
    questions: List[str] = []
    answers: List[str] = []

    for i, item in enumerate(batch):
        r = item["region"].shape[0]
        region[i, :r] = item["region"]
        region_mask[i, :r] = True

        for rr in range(r):
            patch_rr = item["patch_list"][rr]   # (C_rr, 512)
            cell_rr = item["cell_list"][rr]     # (C_rr, 384)

            c = patch_rr.shape[0]
            if c == 0:
                # Should never happen because empty-cell regions were already dropped,
                # but keep this guard anyway.
                continue

            patch[i, rr, :c] = patch_rr
            cell[i, rr, :c] = cell_rr
            cell_mask[i, rr, :c] = True

        slide_ids.append(item["slide_id"])
        questions.append(item["question"])
        answers.append(item["answer"])

    return {
        "slide_ids": slide_ids,
        "questions": questions,
        "answers": answers,
        "wsi": wsi,
        "region": region,
        "patch": patch,
        "cell": cell,
        "region_mask": region_mask,
        "cell_mask": cell_mask,
    }


# =========================================================
# Losses
# =========================================================

def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dims,
) -> torch.Tensor:
    """
    Mean over one or more dims with boolean mask support.

    Examples
    --------
    x:    (B, R, C, D)
    mask: (B, R, C)
    dims=(1,2) -> (B, D)

    x:    (B, R, D)
    mask: (B, R)
    dims=1 -> (B, D)
    """
    if mask is None:
        return x.mean(dim=dims)

    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(sorted(dims))

    mask = mask.to(x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    num = x * mask
    den = mask

    # reduce dims in reverse order to preserve indexing
    for d in sorted(dims, reverse=True):
        num = num.sum(dim=d)
        den = den.sum(dim=d)

    den = den.clamp_min(1e-6)
    return num / den


def sentence_split(text: str) -> List[str]:
    # Lightweight sentence split to avoid extra deps.
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    parts: List[str] = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?;":
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
    if cur:
        s = "".join(cur).strip()
        if s:
            parts.append(s)
    return parts if parts else [text]


def make_phrase_chunks(tokens: List[str], chunk_size: int = 4) -> List[str]:
    if not tokens:
        return []
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def pool_text_embedding_from_string(
    tokenizer,
    embed_tokens: nn.Module,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Frozen embedding-space text representation via input embeddings mean-pool.
    Returns: (D,)
    """
    toks = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    )
    input_ids = toks["input_ids"].to(device)
    if input_ids.numel() == 0:
        # fallback to EOS token embedding if empty
        eos_id = tokenizer.eos_token_id
        input_ids = torch.tensor([[eos_id]], device=device)

    embeds = embed_tokens(input_ids).to(dtype=dtype)  # (1, T, D)
    return embeds.mean(dim=1).squeeze(0)  # (D,)


def build_hierarchical_text_reps(
    tokenizer,
    embed_tokens: nn.Module,
    answers: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Creates one text representation per scale:
      - cell   ~ word-level
      - patch  ~ phrase-level
      - region ~ sentence-level
      - wsi    ~ paragraph-level
    """
    cell_reps = []
    patch_reps = []
    region_reps = []
    wsi_reps = []

    for ans in answers:
        ans = ans.strip()
        words = ans.split()
        phrases = make_phrase_chunks(words, chunk_size=4)
        sentences = sentence_split(ans)

        word_text = " ".join(words) if words else ans
        phrase_text = " ".join(phrases) if phrases else ans
        sentence_text = " ".join(sentences) if sentences else ans
        para_text = ans

        cell_reps.append(pool_text_embedding_from_string(tokenizer, embed_tokens, word_text, device, dtype))
        patch_reps.append(pool_text_embedding_from_string(tokenizer, embed_tokens, phrase_text, device, dtype))
        region_reps.append(pool_text_embedding_from_string(tokenizer, embed_tokens, sentence_text, device, dtype))
        wsi_reps.append(pool_text_embedding_from_string(tokenizer, embed_tokens, para_text, device, dtype))

    return {
        "cell": torch.stack(cell_reps, dim=0),
        "patch": torch.stack(patch_reps, dim=0),
        "region": torch.stack(region_reps, dim=0),
        "wsi": torch.stack(wsi_reps, dim=0),
    }


def info_nce_bidirectional(
    image_feats: torch.Tensor,   # (B, D)
    text_feats: torch.Tensor,    # (B, D)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric CLIP-style InfoNCE with a stable single-sample fallback.

    Notes
    -----
    - For B >= 2: standard bidirectional cross-entropy over in-batch negatives.
    - For B == 1: InfoNCE has no negatives; we fall back to cosine distance
      so the projector still receives a useful alignment signal.
    """
    if image_feats.ndim != 2 or text_feats.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors (B, D), got image={tuple(image_feats.shape)}, "
            f"text={tuple(text_feats.shape)}"
        )
    if image_feats.shape != text_feats.shape:
        raise ValueError(
            f"image_feats and text_feats must have the same shape, got "
            f"{tuple(image_feats.shape)} vs {tuple(text_feats.shape)}"
        )

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    batch_size = image_feats.size(0)

    # Keep cosine similarities and logits numerically stable under bf16/fp16.
    image_feats = F.normalize(image_feats.float(), dim=-1, eps=1e-6)
    text_feats = F.normalize(text_feats.float(), dim=-1, eps=1e-6)

    if batch_size == 1:
        return 1.0 - (image_feats * text_feats).sum(dim=-1).mean()

    logits = (image_feats @ text_feats.t()) / temperature
    targets = torch.arange(batch_size, device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def hierarchical_contrastive_loss(
    visual_reps: Dict[str, torch.Tensor],
    text_reps: Dict[str, torch.Tensor],
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Contrastive alignment at four scales:
      cell <-> word
      patch <-> phrase
      region <-> sentence
      wsi <-> paragraph
    """
    loss_cell = info_nce_bidirectional(visual_reps["cell"], text_reps["cell"], temperature)
    loss_patch = info_nce_bidirectional(visual_reps["patch"], text_reps["patch"], temperature)
    loss_region = info_nce_bidirectional(visual_reps["region"], text_reps["region"], temperature)
    loss_wsi = info_nce_bidirectional(visual_reps["wsi"], text_reps["wsi"], temperature)

    loss = (loss_cell + loss_patch + loss_region + loss_wsi) / 4.0
    stats = {
        "contrast_cell": float(loss_cell.detach().item()),
        "contrast_patch": float(loss_patch.detach().item()),
        "contrast_region": float(loss_region.detach().item()),
        "contrast_wsi": float(loss_wsi.detach().item()),
    }
    return loss, stats


def cross_scale_consistency_loss(visual_reps: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Adjacent-scale semantic consistency:
      cell ~ patch
      patch ~ region
      region ~ wsi
    """
    v_cell = F.normalize(visual_reps["cell"], dim=-1)
    v_patch = F.normalize(visual_reps["patch"], dim=-1)
    v_region = F.normalize(visual_reps["region"], dim=-1)
    v_wsi = F.normalize(visual_reps["wsi"], dim=-1)

    loss_cp = 1.0 - (v_cell * v_patch).sum(dim=-1).mean()
    loss_pr = 1.0 - (v_patch * v_region).sum(dim=-1).mean()
    loss_rw = 1.0 - (v_region * v_wsi).sum(dim=-1).mean()

    loss = (loss_cp + loss_pr + loss_rw) / 3.0
    stats = {
        "cons_cell_patch": float(loss_cp.detach().item()),
        "cons_patch_region": float(loss_pr.detach().item()),
        "cons_region_wsi": float(loss_rw.detach().item()),
    }
    return loss, stats


# =========================================================
# Training helpers
# =========================================================

def freeze_llm(model: MLLMHWSIQWEN) -> None:
    for p in model.llm.parameters():
        p.requires_grad = False
    for p in model.embed_tokens.parameters():
        p.requires_grad = False


def get_visual_scale_reps(
    model: MLLMHWSIQWEN,
    wsi: torch.Tensor,
    region: torch.Tensor,
    patch: torch.Tensor,
    cell: torch.Tensor,
    region_mask: Optional[torch.Tensor] = None,
    cell_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cuda")
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Runs the pathology encoder submodules explicitly to get:
      - slide_tokens: (B, M, D)
      - scale reps  : (B, D) for cell/patch/region/wsi

    patch/cell are padded on C dimension. cell_mask specifies which
    (B, R, C) entries are real.
    """
    pe = model.pathology_encoder
    encoder_dtype = next(pe.parameters()).dtype

    wsi_token = pe.wsi_proj(wsi.to(dtype=encoder_dtype))                # (B, 1, D)
    region_tokens = pe.region_proj(region.to(dtype=encoder_dtype))      # (B, R, D)
    patch_tokens = pe.patch_proj(patch.to(dtype=encoder_dtype))         # (B, R, C, D)
    cell_tokens = pe.cell_proj(cell.to(dtype=encoder_dtype))            # (B, R, C, D)

    # Zero-out padded slots explicitly so downstream ops do not see junk
    if cell_mask is not None:
        mask4 = cell_mask.unsqueeze(-1).to(patch_tokens.dtype)   # (B, R, C, 1)
        patch_tokens = patch_tokens * mask4
        cell_tokens = cell_tokens * mask4

    fused_region_tokens = pe.fusion(
        wsi_token=wsi_token,
        region_tokens=region_tokens,
        patch_tokens=patch_tokens,
        cell_tokens=cell_tokens,
    )                                           # (B, R, D)

    slide_tokens = pe.compressor(fused_region_tokens, mask=region_mask)  # (B, M, D)

    # Scale-level pooled visual representations using masks
    cell_repr = masked_mean(cell_tokens, cell_mask, dims=(1, 2))         # (B, D)
    patch_repr = masked_mean(patch_tokens, cell_mask, dims=(1, 2))       # (B, D)
    region_repr = masked_mean(region_tokens, region_mask, dims=1)        # (B, D)
    wsi_repr = wsi_token.squeeze(1)                                      # (B, D)

    visual_reps = {
        "cell": cell_repr,
        "patch": patch_repr,
        "region": region_repr,
        "wsi": wsi_repr,
    }
    return slide_tokens, visual_reps


def build_training_batch_inputs(
    model: MLLMHWSIQWEN,
    slide_tokens: torch.Tensor,       # (B, M, D)
    questions: List[str],
    answers: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds:
      inputs_embeds: (B, L, D)
      attention_mask: (B, L)
      labels: (B, L) with answer-only supervision

    Sequence layout:
      [ pathology_prefix | prompt_chat_tokens | answer_tokens ]
    """
    tokenizer = model.tokenizer
    embed_tokens = model.embed_tokens

    all_embeds = []
    all_attn = []
    all_labels = []

    bsz = len(questions)

    for i in range(bsz):
        prompt = build_prompt(questions[i])
        prompt_chat = model.format_chat(prompt)

        prompt_tok = tokenizer(
            prompt_chat,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )
        answer_tok = tokenizer(
            answers[i] + (tokenizer.eos_token or ""),
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )

        prompt_ids = prompt_tok["input_ids"].to(device)
        answer_ids = answer_tok["input_ids"].to(device)

        prompt_embeds = embed_tokens(prompt_ids).to(dtype=dtype)   # (1, Tp, D)
        answer_embeds = embed_tokens(answer_ids).to(dtype=dtype)   # (1, Ta, D)

        vis = slide_tokens[i:i+1]                                  # (1, M, D)
        vis = vis.to(dtype=dtype)
        inp = torch.cat([vis, prompt_embeds, answer_embeds], dim=1)

        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=device)

        labels = torch.full((1, inp.shape[1]), -100, dtype=torch.long, device=device)
        ans_start = vis.shape[1] + prompt_embeds.shape[1]
        labels[:, ans_start:ans_start + answer_ids.shape[1]] = answer_ids

        all_embeds.append(inp)
        all_attn.append(attn)
        all_labels.append(labels)

    max_len = max(x.shape[1] for x in all_embeds)
    d = all_embeds[0].shape[-1]

    batch_embeds = torch.zeros(bsz, max_len, d, dtype=dtype, device=device)
    batch_attn = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
    batch_labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)

    for i in range(bsz):
        cur_len = all_embeds[i].shape[1]
        batch_embeds[i, :cur_len] = all_embeds[i][0]
        batch_attn[i, :cur_len] = all_attn[i][0]
        batch_labels[i, :cur_len] = all_labels[i][0]

    return batch_embeds, batch_attn, batch_labels


def save_pathology_encoder_checkpoint(
    save_path: str,
    model: MLLMHWSIQWEN,
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
        "state_dict": model.pathology_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "projector_cfg": asdict(cfg),
    }
    torch.save(payload, save_path)


def save_best_model_metadata(
    json_path: str,
    model_path: str,
    best_epoch: int,
    best_val_loss: float,
    epoch_history: List[Dict[str, float]],
):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    payload = {
        "model_path": model_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epoch_history": epoch_history,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# Train / Eval
# =========================================================

def run_epoch(
    model: MLLMHWSIQWEN,
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
    model.pathology_encoder.train(is_train)
    model.llm.eval()  # always frozen

    autocast_enabled = bool(autocast_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=autocast_enabled)

    total_loss = 0.0
    total_lm = 0.0
    total_ctr = 0.0
    total_cons = 0.0
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
            slide_tokens, visual_reps = get_visual_scale_reps(
                model=model,
                wsi=wsi.to(dtype=dtype),
                region=region.to(dtype=dtype),
                patch=patch.to(dtype=dtype),
                cell=cell.to(dtype=dtype),
                region_mask=region_mask,
                cell_mask=cell_mask,
                device=device
            )

            text_reps = build_hierarchical_text_reps(
                tokenizer=model.tokenizer,
                embed_tokens=model.embed_tokens,
                answers=batch["answers"],
                device=device,
                dtype=slide_tokens.dtype,
            )

            contrastive_loss, _ = hierarchical_contrastive_loss(
                visual_reps=visual_reps,
                text_reps=text_reps,
                temperature=temperature,
            )

            consistency_loss, _ = cross_scale_consistency_loss(visual_reps)

            inputs_embeds, attention_mask, labels = build_training_batch_inputs(
                model=model,
                slide_tokens=slide_tokens,
                questions=batch["questions"],
                answers=batch["answers"],
                device=device,
                dtype=dtype,
            )

            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            lm_loss = outputs.loss

            loss = (
                lambda_lm * lm_loss
                + lambda_contrastive * contrastive_loss
                + lambda_consistency * consistency_loss
            )

        if is_train:
            loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.pathology_encoder.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.detach().item())
        total_lm += float(lm_loss.detach().item())
        total_ctr += float(contrastive_loss.detach().item())
        total_cons += float(consistency_loss.detach().item())
        n_steps += 1

    if is_train and (n_steps % grad_accum_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.pathology_encoder.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return {
        "loss": total_loss / max(n_steps, 1),
        "lm_loss": total_lm / max(n_steps, 1),
        "contrastive_loss": total_ctr / max(n_steps, 1),
        "consistency_loss": total_cons / max(n_steps, 1),
    }


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--wsi_dir", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--wsi_feat_dir", type=str, required=True)
    parser.add_argument("--region_feat_dir", type=str, required=True)
    parser.add_argument("--patch_feat_dir", type=str, required=True)
    parser.add_argument("--cell_feat_dir", type=str, required=True)
    parser.add_argument("--cell_pt_filename", type=str, required=True)

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

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.25)

    # Split
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Save
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="pathology_encoder.pt")
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
        report_json=args.report_json,
        wsi_feat_dir=args.wsi_feat_dir,
        region_feat_dir=args.region_feat_dir,
        patch_feat_dir=args.patch_feat_dir,
        cell_feat_dir=args.cell_feat_dir,
        cell_pt_filename=args.cell_pt_filename,
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

    model = MLLMHWSIQWEN(
        model_name=args.model_name,
        projector_cfg=cfg,
        device=str(device),
        torch_dtype=dtype,
    )

    freeze_llm(model)

    trainable_params = [p for p in model.pathology_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
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
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.pathology_encoder.load_state_dict(state_dict, strict=False)
        print(f"[Resume] missing keys: {missing}")
        print(f"[Resume] unexpected keys: {unexpected}")

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f"best_{args.save_name}")
    last_path = os.path.join(args.save_dir, f"last_{args.save_name}")
    best_meta_path = os.path.splitext(best_path)[0] + ".json"

    if os.path.isfile(best_meta_path):
        try:
            with open(best_meta_path, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
            if isinstance(best_meta.get("epoch_history"), list):
                epoch_history = best_meta["epoch_history"]
        except (OSError, json.JSONDecodeError, TypeError):
            epoch_history = []

    print(f"[Train] train samples: {n_train}")
    print(f"[Train] val samples  : {n_val}")
    print(f"[Train] device       : {device}")
    print(f"[Train] dtype        : {dtype}")
    print(f"[Train] total steps  : {total_update_steps}")
    print(f"[Train] warmup steps : {warmup_steps}")

    for epoch in range(start_epoch, args.epochs):
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

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_lm={train_stats['lm_loss']:.4f} "
            f"train_ctr={train_stats['contrastive_loss']:.4f} "
            f"train_cons={train_stats['consistency_loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f}"
        )

        epoch_history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_stats["loss"]),
                "train_lm_loss": float(train_stats["lm_loss"]),
                "train_contrastive_loss": float(train_stats["contrastive_loss"]),
                "train_consistency_loss": float(train_stats["consistency_loss"]),
                "val_loss": float(val_stats["loss"]),
            }
        )

        save_pathology_encoder_checkpoint(
            save_path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            cfg=cfg,
        )

        save_best_model_metadata(
            json_path=best_meta_path,
            model_path=best_path,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            epoch_history=epoch_history,
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch + 1
            save_pathology_encoder_checkpoint(
                save_path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                cfg=cfg,
            )
            save_best_model_metadata(
                json_path=best_meta_path,
                model_path=best_path,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                epoch_history=epoch_history,
            )
            print(f"[Checkpoint] saved best to {best_path}")

    print(f"[Done] last checkpoint: {last_path}")
    print(f"[Done] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()