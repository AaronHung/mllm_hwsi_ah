#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_text(text: str) -> str:
	return " ".join(str(text).strip().split())


def normalize_model_source(source: str) -> str:
	"""Normalize model source to avoid invalid dynamic module names."""
	normalized = str(source).strip().rstrip("/\\")
	if not normalized:
		raise ValueError("Model source cannot be empty.")
	return normalized


def apply_lora_to_llm(
	model: Any,
	lora_r: int,
	lora_alpha: int,
	lora_dropout: float,
	print_trainable: bool = False,
) -> List[str]:
	"""Attach LoRA adapters to the LLM and return selected target modules."""
	try:
		from peft import LoraConfig, TaskType, get_peft_model
	except ImportError as exc:
		raise ImportError(
			"peft is required to load LoRA checkpoints. Install with: pip install peft"
		) from exc

	candidate_targets = [
		"q_proj",
		"k_proj",
		"v_proj",
		"o_proj",
		"gate_proj",
		"up_proj",
		"down_proj",
	]
	linear_leaf_names = {
		name.split(".")[-1]
		for name, module in model.llm.named_modules()
		if isinstance(module, nn.Linear)
	}
	target_modules = [name for name in candidate_targets if name in linear_leaf_names]
	if not target_modules:
		raise RuntimeError("Could not find LoRA target modules in the loaded LLM.")

	lora_cfg = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=lora_r,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout,
		bias="none",
		target_modules=target_modules,
	)
	model.llm = get_peft_model(model.llm, lora_cfg)
	if print_trainable and hasattr(model.llm, "print_trainable_parameters"):
		model.llm.print_trainable_parameters()
	return target_modules


def collate_wsis(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	if len(batch) == 0:
		raise ValueError("Empty batch passed to collate_wsis.")

	bsz = len(batch)
	r_max = max(item["region"].shape[0] for item in batch)

	c_max = 1
	for item in batch:
		for cell_region in item["cell_list"]:
			c_max = max(c_max, int(cell_region.shape[0]))

	wsi = torch.stack([item["wsi"] for item in batch], dim=0)

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
			patch_rr = item["patch_list"][rr]
			cell_rr = item["cell_list"][rr]

			c = patch_rr.shape[0]
			if c == 0:
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


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dims) -> torch.Tensor:
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

	for d in sorted(dims, reverse=True):
		num = num.sum(dim=d)
		den = den.sum(dim=d)

	den = den.clamp_min(1e-6)
	return num / den


def sentence_split(text: str) -> List[str]:
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
	toks = tokenizer(
		text,
		return_tensors="pt",
		add_special_tokens=False,
		truncation=True,
	)
	input_ids = toks["input_ids"].to(device)
	if input_ids.numel() == 0:
		eos_id = tokenizer.eos_token_id
		input_ids = torch.tensor([[eos_id]], device=device)

	embeds = embed_tokens(input_ids).to(dtype=dtype)
	return embeds.mean(dim=1).squeeze(0)


def build_hierarchical_text_reps(
	tokenizer,
	embed_tokens: nn.Module,
	answers: List[str],
	device: torch.device,
	dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
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
	image_feats: torch.Tensor,
	text_feats: torch.Tensor,
	temperature: float = 0.07,
) -> torch.Tensor:
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


def get_visual_scale_reps(
	model: Any,
	wsi: torch.Tensor,
	region: torch.Tensor,
	patch: torch.Tensor,
	cell: torch.Tensor,
	region_mask: Optional[torch.Tensor] = None,
	cell_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
	pe = model.vl_projector
	encoder_dtype = next(pe.parameters()).dtype

	wsi_token = pe.wsi_proj(wsi.to(dtype=encoder_dtype))
	region_tokens = pe.region_proj(region.to(dtype=encoder_dtype))
	patch_tokens = pe.patch_proj(patch.to(dtype=encoder_dtype))
	cell_tokens = pe.cell_proj(cell.to(dtype=encoder_dtype))

	if cell_mask is not None:
		mask4 = cell_mask.unsqueeze(-1).to(patch_tokens.dtype)
		patch_tokens = patch_tokens * mask4
		cell_tokens = cell_tokens * mask4

	fused_region_tokens = pe.fusion(
		wsi_token=wsi_token,
		region_tokens=region_tokens,
		patch_tokens=patch_tokens,
		cell_tokens=cell_tokens,
	)
	slide_tokens = pe.compressor(fused_region_tokens, mask=region_mask)

	cell_repr = masked_mean(cell_tokens, cell_mask, dims=(1, 2))
	patch_repr = masked_mean(patch_tokens, cell_mask, dims=(1, 2))
	region_repr = masked_mean(region_tokens, region_mask, dims=1)
	wsi_repr = wsi_token.squeeze(1)

	visual_reps = {
		"cell": cell_repr,
		"patch": patch_repr,
		"region": region_repr,
		"wsi": wsi_repr,
	}
	return slide_tokens, visual_reps


def build_training_batch_inputs(
	model: Any,
	slide_tokens: torch.Tensor,
	questions: List[str],
	answers: List[str],
	device: torch.device,
	dtype: torch.dtype,
	prompt_builder: Optional[Callable[[str], str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	tokenizer = model.tokenizer
	embed_tokens = model.embed_tokens

	all_embeds = []
	all_attn = []
	all_labels = []

	bsz = len(questions)

	for i in range(bsz):
		prompt = prompt_builder(questions[i]) if prompt_builder is not None else questions[i]
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

		prompt_embeds = embed_tokens(prompt_ids).to(dtype=dtype)
		answer_embeds = embed_tokens(answer_ids).to(dtype=dtype)

		vis = slide_tokens[i:i + 1].to(dtype=dtype)
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


def save_best_model_metadata(
	json_path: str,
	model_path: str,
	best_epoch: int,
	best_val_loss: float,
	epoch_history: List[Dict[str, float]],
	checkpoint_path: Optional[str] = None,
):
	os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
	payload = {
		"model_path": model_path,
		"checkpoint_path": checkpoint_path,
		"best_epoch": best_epoch,
		"best_val_loss": best_val_loss,
		"epoch_history": epoch_history,
	}
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def format_hms(total_seconds: float) -> str:
	total_seconds = max(0.0, float(total_seconds))
	hours = int(total_seconds // 3600)
	minutes = int((total_seconds % 3600) // 60)
	seconds = total_seconds % 60
	return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
