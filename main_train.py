#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

import main_utils as _mu
from data_utils import (build_prompt, find_slide_ids, 
						load_slide_feature_quadruplet, 
						read_slide_report, set_seed)
from mllm_hwsi import MLLMHWSI, VLProjectorConfig


STAGES = ("stage2", "stage3")
CLI_STAGES = ("2", "3")


def parse_training_stage(value: str) -> str:
	key = (value or "").strip().lower()
	if key == "2":
		return "stage2"
	if key == "3":
		return "stage3"
	raise argparse.ArgumentTypeError(f"Invalid --training_stage '{value}'. Allowed: {', '.join(CLI_STAGES)}")


def load_json_records(path: Path) -> List[Dict[str, Any]]:
	text = path.read_text(encoding="utf-8").strip()
	if not text:
		return []

	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			return [parsed]
		if isinstance(parsed, list):
			return [row for row in parsed if isinstance(row, dict)]
	except json.JSONDecodeError:
		pass

	records: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(row, dict):
				records.append(row)
	return records


def strip_image_token(text: str) -> str:
	return " ".join(str(text).replace("<image>", " ").replace("<Image>", " ").split()).strip()


def resolve_stage3_slide_id(record: Dict[str, Any], available_slide_ids: set) -> Optional[str]:
	image_field = record.get("image")
	if image_field:
		stem = Path(str(image_field)).stem
		if stem in available_slide_ids:
			return stem

	haystacks = [
		str(record.get("id", "")),
		str(record.get("question_id", "")),
		str(record.get("image", "")),
	]
	for slide_id in available_slide_ids:
		if any(slide_id in h for h in haystacks):
			return slide_id
	return None


def extract_stage3_qa(record: Dict[str, Any]) -> Tuple[str, str]:
	conv = record.get("conversations", [])
	if not isinstance(conv, list) or len(conv) == 0:
		raise ValueError("Missing or empty 'conversations' list")

	question = None
	answer = None
	for turn in conv:
		if not isinstance(turn, dict):
			continue
		role = str(turn.get("from", turn.get("role", ""))).strip().lower()
		value = strip_image_token(str(turn.get("value", "")))
		if role in {"human", "user"} and value and question is None:
			question = value
		if role in {"gpt", "assistant"} and value and answer is None:
			answer = value
		if question is not None and answer is not None:
			break

	if not question or not answer:
		raise ValueError("Could not extract human/gpt pair from 'conversations'")
	return question, answer


class WSITrainDataset(Dataset):
	def __init__(
		self,
		training_stage: str,
		wsi_dir: str,
		records_json: str,
		wsi_feat_dir: str,
		region_feat_dir: str,
		patch_feat_dir: str,
		cell_feat_dir: str,
		cell_pt_filename: str,
		strict_report_match: bool = False,
	):
		self.training_stage = training_stage
		self.wsi_dir = wsi_dir
		self.records_path = Path(records_json)
		self.wsi_feat_dir = wsi_feat_dir
		self.region_feat_dir = region_feat_dir
		self.patch_feat_dir = patch_feat_dir
		self.cell_feat_dir = cell_feat_dir
		self.cell_pt_filename = cell_pt_filename
		self.strict_report_match = strict_report_match

		samples: List[Dict[str, str]] = []
		skipped: List[str] = []

		if training_stage == "stage2":
			for slide_id in find_slide_ids(wsi_dir):
				try:
					question, answer = read_slide_report(self.records_path, slide_id)
					if not question or not answer:
						raise KeyError(f"Empty question/answer for {slide_id}")
					self._assert_features_exist(slide_id)
					samples.append({"slide_id": slide_id, "question": question, "answer": answer})
				except Exception:
					skipped.append(slide_id)
					if strict_report_match:
						raise
		else:
			available_slide_ids = set(find_slide_ids(wsi_dir))
			records = load_json_records(self.records_path)
			if not records:
				raise RuntimeError(f"No records found in {self.records_path}")

			for record in records:
				try:
					if not isinstance(record, dict):
						raise ValueError("Each conversation record must be a dict")
					slide_id = resolve_stage3_slide_id(record, available_slide_ids)
					if slide_id is None:
						raise KeyError("Could not resolve slide_id from conversation record")
					question, answer = extract_stage3_qa(record)
					if not question or not answer:
						raise KeyError(f"Empty question/answer for {slide_id}")
					self._assert_features_exist(slide_id)
					samples.append({"slide_id": slide_id, "question": question, "answer": answer})
				except Exception:
					sid = None
					if isinstance(record, dict):
						sid = record.get("id") or record.get("image") or record.get("question_id")
					skipped.append(str(sid))
					if strict_report_match:
						raise

		self.samples = samples
		self.skipped = skipped
		if not self.samples:
			raise RuntimeError("No valid training samples found.")

		print(f"[Dataset] usable slides: {len(self.samples)}")
		print(f"[Dataset] skipped slides: {len(self.skipped)}")

	def _assert_features_exist(self, slide_id: str) -> None:
		feature_paths = [
			os.path.join(self.wsi_feat_dir, f"{slide_id}.pt"),
			os.path.join(self.region_feat_dir, f"{slide_id}.pt"),
			os.path.join(self.patch_feat_dir, f"{slide_id}.pt"),
			os.path.join(self.cell_feat_dir, slide_id, self.cell_pt_filename),
		]
		missing = [path for path in feature_paths if not os.path.isfile(path)]
		if missing:
			raise FileNotFoundError(f"Missing extracted features for {slide_id}: {', '.join(missing)}")

	@staticmethod
	def _valid_cell_mask(cell_region: torch.Tensor) -> torch.Tensor:
		if cell_region.ndim != 2 or cell_region.shape[-1] != 384:
			raise ValueError(f"Expected cell_region shape (K, 384), got {tuple(cell_region.shape)}")
		return cell_region.abs().sum(dim=-1) > 0

	def __len__(self) -> int:
		return len(self.samples)

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

		wsi = wsi.squeeze(0)
		region = region.squeeze(0)
		patch = patch.squeeze(0)
		cell = cell.squeeze(0)

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
		for r_idx in range(region.shape[0]):
			region_feat = region[r_idx]
			patch_region = patch[r_idx]
			cell_region = cell[r_idx]
			valid_cells = self._valid_cell_mask(cell_region)
			if int(valid_cells.sum().item()) == 0:
				continue
			kept_region_feats.append(region_feat)
			kept_patch_feats.append(patch_region[valid_cells])
			kept_cell_feats.append(cell_region[valid_cells])

		if not kept_region_feats:
			raise RuntimeError(f"Slide {slide_id} has no valid regions after filtering zero/empty cell features.")

		return {
			"slide_id": slide_id,
			"question": item["question"],
			"answer": item["answer"],
			"wsi": wsi,
			"region": torch.stack(kept_region_feats, dim=0),
			"patch_list": kept_patch_feats,
			"cell_list": kept_cell_feats,
		}


collate_wsis = _mu.collate_wsis
masked_mean = _mu.masked_mean
sentence_split = _mu.sentence_split
make_phrase_chunks = _mu.make_phrase_chunks
pool_text_embedding_from_string = _mu.pool_text_embedding_from_string
build_hierarchical_text_reps = _mu.build_hierarchical_text_reps
info_nce_bidirectional = _mu.info_nce_bidirectional
hierarchical_contrastive_loss = _mu.hierarchical_contrastive_loss
cross_scale_consistency_loss = _mu.cross_scale_consistency_loss
get_visual_scale_reps = _mu.get_visual_scale_reps
build_training_batch_inputs = _mu.build_training_batch_inputs
save_best_model_metadata = _mu.save_best_model_metadata
format_hms = _mu.format_hms


def freeze_llm(model: MLLMHWSI) -> None:
	for p in model.llm.parameters():
		p.requires_grad = False
	for p in model.embed_tokens.parameters():
		p.requires_grad = False


def freeze_projector(model: MLLMHWSI) -> None:
	for p in model.vl_projector.parameters():
		p.requires_grad = False


def save_stage2_checkpoint(
	save_path: str,
	model: MLLMHWSI,
	optimizer: torch.optim.Optimizer,
	scheduler,
	epoch: int,
	step: int,
	best_val_loss: float,
	best_epoch: int,
	cfg: VLProjectorConfig,
) -> None:
	os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
	payload = {
		"state_dict": model.vl_projector.state_dict(),
		"vl_projector_state_dict": model.vl_projector.state_dict(),
		"optimizer": optimizer.state_dict(),
		"scheduler": None if scheduler is None else scheduler.state_dict(),
		"epoch": epoch,
		"step": step,
		"best_val_loss": best_val_loss,
		"best_epoch": best_epoch,
		"projector_cfg": asdict(cfg),
	}
	torch.save(payload, save_path)


def save_stage3_checkpoint(
	save_path: str,
	model: MLLMHWSI,
	optimizer: torch.optim.Optimizer,
	scheduler,
	epoch: int,
	step: int,
	best_val_loss: float,
	best_epoch: int,
	cfg: VLProjectorConfig,
) -> None:
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


def build_optimizer(args, model: MLLMHWSI, training_stage: str) -> Tuple[torch.optim.Optimizer, float, float]:
	if training_stage == "stage2":
		lr = args.projector_lr if args.projector_lr is not None else args.lr
		params = [p for p in model.vl_projector.parameters() if p.requires_grad]
		if not params:
			raise RuntimeError("No trainable projector parameters found.")
		optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
		return optimizer, lr, lr

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
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
	return optimizer, llm_lr, projector_lr


def load_resume_checkpoint(training_stage: str, model: MLLMHWSI, optimizer, scheduler, resume_ckpt: str) -> Tuple[int, int, float, int]:
	ckpt = torch.load(resume_ckpt, map_location="cpu")
	if training_stage == "stage2":
		state_dict = ckpt["vl_projector_state_dict"] if "vl_projector_state_dict" in ckpt else ckpt.get("state_dict", ckpt)
		missing, unexpected = model.vl_projector.load_state_dict(state_dict, strict=False)
		print(f"[Resume] missing keys: {missing}")
		print(f"[Resume] unexpected keys: {unexpected}")
	else:
		llm_state = ckpt["llm_state_dict"] if "llm_state_dict" in ckpt else ckpt
		missing, unexpected = model.llm.load_state_dict(llm_state, strict=False)
		print(f"[Resume] missing keys: {missing}")
		print(f"[Resume] unexpected keys: {unexpected}")

		projector_state = ckpt.get("vl_projector_state_dict") or ckpt.get("pathology_encoder_state_dict")
		if projector_state is not None:
			pe_missing, pe_unexpected = model.vl_projector.load_state_dict(projector_state, strict=False)
			print(f"[Resume VL Projector] missing keys: {pe_missing}")
			print(f"[Resume VL Projector] unexpected keys: {pe_unexpected}")

	if "optimizer" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer"])
	if "scheduler" in ckpt and ckpt["scheduler"] is not None:
		scheduler.load_state_dict(ckpt["scheduler"])

	start_epoch = int(ckpt.get("epoch", 0))
	global_step = int(ckpt.get("step", 0))
	best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
	best_epoch = int(ckpt.get("best_epoch", 0))
	return start_epoch, global_step, best_val_loss, best_epoch


def export_merged_hf_model(
	args,
	model: MLLMHWSI,
	cfg: VLProjectorConfig,
	device: torch.device,
	dtype: torch.dtype,
	out_dir: str,
	tag: str,
) -> None:
	if not hasattr(model.llm, "merge_and_unload"):
		print(f"[Merge] Skipped {tag} export: model.llm is not LoRA-wrapped.")
		return

	export_model = MLLMHWSI(
		model_name=args.model_name,
		projector_cfg=cfg,
		device=str(device),
		torch_dtype=dtype,
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
	training_stage: str,
) -> Dict[str, float]:
	is_train = optimizer is not None
	model.vl_projector.train(is_train)
	model.llm.train(is_train) if training_stage == "stage3" else model.llm.eval()

	autocast_enabled = bool(autocast_enabled and device.type == "cuda")
	trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
	scaler_enabled = bool(autocast_enabled and (torch.float16 not in trainable_dtypes))
	scaler = torch.amp.GradScaler(enabled=scaler_enabled)

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
			)

			if training_stage == "stage2":
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
				prompt_builder = lambda question: build_prompt(question, style="report")
			else:
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
				prompt_builder = lambda question: build_prompt(question, style="conversation")

			inputs_embeds, attention_mask, labels = build_training_batch_inputs(
				model=model,
				slide_tokens=slide_tokens,
				questions=batch["questions"],
				answers=batch["answers"],
				device=device,
				dtype=dtype,
				prompt_builder=prompt_builder,
			)

			outputs = model.llm(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				labels=labels,
				return_dict=True,
			)
			lm_loss = outputs.loss
			loss = lambda_lm * lm_loss + lambda_contrastive * contrastive_loss + lambda_consistency * consistency_loss

		if is_train:
			loss_for_backward = loss / grad_accum_steps
			if scaler_enabled:
				scaler.scale(loss_for_backward).backward()
			else:
				loss_for_backward.backward()

			if (step + 1) % grad_accum_steps == 0:
				if scaler_enabled:
					scaler.unscale_(optimizer)
				trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
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
		total_ctr += float(contrastive_loss.detach().item())
		total_cons += float(consistency_loss.detach().item())
		n_steps += 1

	if is_train and (n_steps % grad_accum_steps) != 0:
		if scaler_enabled:
			scaler.unscale_(optimizer)
		trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
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
		"contrastive_loss": total_ctr / max(n_steps, 1),
		"consistency_loss": total_cons / max(n_steps, 1),
	}


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--training_stage",
		type=parse_training_stage,
		required=True,
		metavar="{2,3}",
		help="Training stage: 2 (projector training) or 3 (LLM fine-tuning).",
	)

	parser.add_argument("--wsi_dir", type=str, required=True)
	parser.add_argument("--report_json", type=str, default=None)
	parser.add_argument("--conversation_json", type=str, default=None)
	parser.add_argument("--wsi_feat_dir", type=str, required=True)
	parser.add_argument("--region_feat_dir", type=str, required=True)
	parser.add_argument("--patch_feat_dir", type=str, required=True)
	parser.add_argument("--cell_feat_dir", type=str, required=True)
	parser.add_argument("--cell_pt_filename", type=str, required=True)
	parser.add_argument(
		"--strict_report_match",
		action="store_true",
		help="Fail fast if any slide has missing conversation/report fields or required feature files.",
	)

	parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--llm_dim", type=int, default=3584)
	parser.add_argument("--proj_hidden_dim", type=int, default=1024)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--num_slide_tokens", type=int, default=64)
	parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])
	parser.add_argument("--lora_r", type=int, default=64)
	parser.add_argument("--lora_alpha", type=int, default=256)
	parser.add_argument("--lora_dropout", type=float, default=0.1)

	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument(
		"--no_change",
		type=int,
		default=5,
		help="Stop after this many consecutive epochs without a new best val loss. Set <= 0 to disable.",
	)
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--llm_lr", type=float, default=None)
	parser.add_argument("--projector_lr", type=float, default=None)
	parser.add_argument("--weight_decay", type=float, default=1e-2)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--grad_accum_steps", type=int, default=4)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument(
		"--autocast_enabled",
		"--use_amp",
		action="store_true",
		help="Enable torch.autocast mixed precision (legacy alias: --use_amp)",
	)

	parser.add_argument("--temperature", type=float, default=0.75)
	parser.add_argument("--lambda_lm", type=float, default=0.7)
	parser.add_argument("--lambda_contrastive", type=float, default=2.0)
	parser.add_argument("--lambda_consistency", type=float, default=0.25)
	parser.add_argument("--val_ratio", type=float, default=0.1)

	parser.add_argument("--save_dir", type=str, required=True)
	parser.add_argument("--save_name", type=str, default=None)
	parser.add_argument("--resume_ckpt", type=str, default=None)
	parser.add_argument("--stage2_ckpt", type=str, default=None)

	return parser.parse_args()


def main():
	args = parse_args()
	set_seed(args.seed)

	if args.training_stage == "stage2":
		if not args.report_json:
			raise ValueError("--report_json is required for --training_stage stage2")
		records_json = args.report_json
		save_name = args.save_name or "vl_projector.pt"
		lambda_lm = 0.7 if args.lambda_lm is None else args.lambda_lm
		lambda_contrastive = 2.0 if args.lambda_contrastive is None else args.lambda_contrastive
		lambda_consistency = 0.25 if args.lambda_consistency is None else args.lambda_consistency
	else:
		if not args.conversation_json:
			raise ValueError("--conversation_json is required for --training_stage stage3")
		if not args.stage2_ckpt:
			raise ValueError("--stage2_ckpt is required for --training_stage stage3")
		records_json = args.conversation_json
		save_name = args.save_name or "stage3_model.pt"
		lambda_lm = 1.0 if args.lambda_lm is None else args.lambda_lm
		lambda_contrastive = 0.1 if args.lambda_contrastive is None else args.lambda_contrastive
		lambda_consistency = 0.05 if args.lambda_consistency is None else args.lambda_consistency

	device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
	dtype = torch.float16 if device.type == "cuda" else torch.float32

	cfg = VLProjectorConfig(
		llm_dim=args.llm_dim,
		hidden_dim=args.proj_hidden_dim,
		dropout=args.dropout,
		use_layernorm=True,
		num_query_tokens=args.num_slide_tokens,
		projector_type=args.projector_type,
	)

	dataset = WSITrainDataset(
		training_stage=args.training_stage,
		wsi_dir=args.wsi_dir,
		records_json=records_json,
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
	val_loader = (
		DataLoader(
			val_set,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=(device.type == "cuda"),
			collate_fn=collate_wsis,
		)
		if n_val > 0
		else None
	)

	model = MLLMHWSI(
		model_name=args.model_name,
		projector_cfg=cfg,
		device=str(device),
		torch_dtype=dtype,
	)

	if args.training_stage == "stage2":
		freeze_llm(model)
	else:
		_mu.apply_lora_to_llm(
			model=model,
			lora_r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			print_trainable=True,
		)
		freeze_projector(model)

	optimizer, llm_lr, projector_lr = build_optimizer(args, model, args.training_stage)

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
		start_epoch, global_step, best_val_loss, best_epoch = load_resume_checkpoint(
			training_stage=args.training_stage,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			resume_ckpt=args.resume_ckpt,
		)

	os.makedirs(args.save_dir, exist_ok=True)
	if args.training_stage == "stage2":
		best_path = os.path.join(args.save_dir, f"best_{save_name}")
		last_path = os.path.join(args.save_dir, f"last_{save_name}")
		best_meta_path = os.path.splitext(best_path)[0] + ".json"
		history_path = best_meta_path
		export_best_dir = None
		export_last_dir = None
	else:
		export_best_dir = os.path.join(args.save_dir, f"best_hf_{Path(save_name).stem}")
		export_last_dir = os.path.join(args.save_dir, f"last_hf_{Path(save_name).stem}")
		best_path = os.path.join(args.save_dir, f"best_{Path(save_name).name}")
		last_path = os.path.join(args.save_dir, f"last_{Path(save_name).name}")
		history_path = os.path.join(args.save_dir, f"train_history_{Path(save_name).stem}.json")
		best_meta_path = history_path

	if os.path.isfile(history_path):
		try:
			with open(history_path, "r", encoding="utf-8") as f:
				hist_payload = json.load(f)
			if isinstance(hist_payload.get("epoch_history"), list):
				epoch_history = hist_payload["epoch_history"]
		except (OSError, json.JSONDecodeError, TypeError):
			pass

	print(f"[Train] stage        : {args.training_stage}")
	print(f"[Train] train samples : {n_train}")
	print(f"[Train] val samples   : {n_val}")
	print(f"[Train] device        : {device}")
	print(f"[Train] dtype         : {dtype}")
	print(f"[Train] total steps   : {total_update_steps}")
	print(f"[Train] warmup steps  : {warmup_steps}")
	print(f"[Train] lambda_lm     : {lambda_lm}")
	print(f"[Train] lambda_ctr    : {lambda_contrastive}")
	print(f"[Train] lambda_cons   : {lambda_consistency}")

	total_train_start = time.perf_counter()
	epochs_without_improvement = max(start_epoch - best_epoch, 0) if best_epoch > 0 else 0

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
			lambda_lm=lambda_lm,
			lambda_contrastive=lambda_contrastive,
			lambda_consistency=lambda_consistency,
			max_grad_norm=args.max_grad_norm,
			autocast_enabled=args.autocast_enabled,
			training_stage=args.training_stage,
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
					lambda_lm=lambda_lm,
					lambda_contrastive=lambda_contrastive,
					lambda_consistency=lambda_consistency,
					max_grad_norm=args.max_grad_norm,
					autocast_enabled=args.autocast_enabled,
					training_stage=args.training_stage,
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
				"train_contrastive_loss": float(train_stats["contrastive_loss"]),
				"train_consistency_loss": float(train_stats["consistency_loss"]),
				"val_loss": float(val_stats["loss"]),
			}
		)

		if args.training_stage == "stage2":
			save_stage2_checkpoint(
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
				json_path=history_path,
				model_path=best_path,
				best_epoch=best_epoch,
				best_val_loss=best_val_loss,
				epoch_history=epoch_history,
			)
		else:
			save_stage3_checkpoint(
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
				json_path=history_path,
				model_path=export_best_dir,
				checkpoint_path=best_path,
				best_epoch=best_epoch,
				best_val_loss=best_val_loss,
				epoch_history=epoch_history,
			)

		if val_stats["loss"] < best_val_loss:
			best_val_loss = val_stats["loss"]
			best_epoch = epoch + 1
			epochs_without_improvement = 0

			if args.training_stage == "stage2":
				save_stage2_checkpoint(
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
					json_path=history_path,
					model_path=best_path,
					best_epoch=best_epoch,
					best_val_loss=best_val_loss,
					epoch_history=epoch_history,
				)
				print(f"[Checkpoint] saved best to {best_path}")
			else:
				save_stage3_checkpoint(
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
				export_merged_hf_model(args, model, cfg, device, dtype, export_best_dir, "best")
				save_best_model_metadata(
					json_path=history_path,
					model_path=export_best_dir,
					checkpoint_path=best_path,
					best_epoch=best_epoch,
					best_val_loss=best_val_loss,
					epoch_history=epoch_history,
				)
				total_train_elapsed = time.perf_counter() - total_train_start
				print(f"[Time] total_training_time={format_hms(total_train_elapsed)}")
		else:
			epochs_without_improvement += 1
			if args.training_stage == "stage2":
				save_best_model_metadata(
					json_path=history_path,
					model_path=best_path,
					best_epoch=best_epoch,
					best_val_loss=best_val_loss,
					epoch_history=epoch_history,
				)
				if args.no_change > 0:
					print(f"[EarlyStop] no improvement for {epochs_without_improvement}/{args.no_change} epoch(s)")
					if epochs_without_improvement >= args.no_change:
						print(
							f"[EarlyStop] stopping after epoch {epoch + 1} with best epoch {best_epoch} and val_loss {best_val_loss:.4f}"
						)
						break
			else:
				save_best_model_metadata(
					json_path=history_path,
					model_path=export_best_dir,
					checkpoint_path=best_path if os.path.isfile(best_path) else None,
					best_epoch=best_epoch,
					best_val_loss=best_val_loss,
					epoch_history=epoch_history,
				)
				if args.no_change > 0:
					print(f"[EarlyStop] no improvement for {epochs_without_improvement}/{args.no_change} epoch(s)")
					if epochs_without_improvement >= args.no_change:
						print(
							f"[EarlyStop] stopping after epoch {epoch + 1} with best epoch {best_epoch} and val_loss {best_val_loss:.4f}"
						)
						break

	if args.training_stage == "stage3":
		export_merged_hf_model(args, model, cfg, device, dtype, export_last_dir, "last")

	if best_epoch == 0:
		best_epoch = min(max(start_epoch, 0) + 1, len(epoch_history)) if epoch_history else 0
		if epoch_history:
			best_val_loss = float(epoch_history[-1]["val_loss"])

	save_best_model_metadata(
		json_path=history_path,
		model_path=best_path if args.training_stage == "stage2" else export_best_dir,
		checkpoint_path=best_path if os.path.isfile(best_path) else None,
		best_epoch=best_epoch,
		best_val_loss=best_val_loss,
		epoch_history=epoch_history,
	)

	if args.training_stage == "stage2":
		print(f"[Done] last checkpoint: {last_path}")
		print(f"[Done] best checkpoint: {best_path}")
	else:
		print(f"[Done] merged best HF model dir: {export_best_dir}")
		print(f"[Done] merged last HF model dir: {export_last_dir}")
		print(f"[Done] training history json: {history_path}")


if __name__ == "__main__":
	main()
