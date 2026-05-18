#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from mllm_hwsi import VLProjectorConfig, MLLMHWSIQWEN
from data_utils import (
    set_seed,
    find_slide_ids,
    read_slide_report,
    load_slide_feature_quadruplet,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


# =========================================================
# Metrics
# =========================================================

def compute_text_metrics(prediction: str, reference: str) -> Dict[str, float]:
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference, prediction)

    smooth = SmoothingFunction().method1
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()

    bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smooth)
    meteor = meteor_score([reference_tokens], prediction_tokens)

    return {
        "rouge1_fmeasure": float(rouge_scores["rouge1"].fmeasure),
        "rouge2_fmeasure": float(rouge_scores["rouge2"].fmeasure),
        "rougeL_fmeasure": float(rouge_scores["rougeL"].fmeasure),
        "bleu": float(bleu),
        "meteor": float(meteor),
    }


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wsi_dir", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)

    parser.add_argument("--wsi_feat_dir", type=str, required=True)
    parser.add_argument("--region_feat_dir", type=str, required=True)
    parser.add_argument("--patch_feat_dir", type=str, required=True)
    parser.add_argument("--cell_feat_dir", type=str, required=True)
    parser.add_argument(
        "--cell_pt_filename",
        type=str,
        required=True,
        help="Filename of the cell .pt file inside cell_feat_dir/<slide_id>/"
    )

    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm_dim", type=int, default=3584)
    parser.add_argument("--proj_hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_slide_tokens", type=int, default=64)
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    parser.add_argument("--pathology_encoder_ckpt", type=str, default=None)
    parser.add_argument("--llm_lora_ckpt", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def apply_lora_to_llm(model: MLLMHWSIQWEN, lora_r: int, lora_alpha: int, lora_dropout: float) -> None:
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


def main():
    args = parse_args()
    set_seed(args.seed)

    reports_path = Path(args.report_json)
    slide_ids = find_slide_ids(args.wsi_dir)

    cfg = VLProjectorConfig(
        llm_dim=args.llm_dim,
        hidden_dim=args.proj_hidden_dim,
        dropout=args.dropout,
        use_layernorm=True,
        num_query_tokens=args.num_slide_tokens,
        projector_type=args.projector_type,
    )

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    model = MLLMHWSIQWEN(
        model_name=args.model_name,
        projector_cfg=cfg,
        device=args.device,
        torch_dtype=dtype,
    )

    if args.llm_lora_ckpt is not None:
        apply_lora_to_llm(
            model=model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        lora_ckpt = torch.load(args.llm_lora_ckpt, map_location="cpu")
        lora_state = lora_ckpt.get("llm_state_dict", lora_ckpt)
        missing, unexpected = model.llm.load_state_dict(lora_state, strict=False)
        print("Loaded LLM LoRA checkpoint.")
        print("LoRA missing keys:", missing)
        print("LoRA unexpected keys:", unexpected)

    if args.pathology_encoder_ckpt is not None:
        ckpt = torch.load(args.pathology_encoder_ckpt, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.pathology_encoder.load_state_dict(ckpt, strict=False)
        print("Loaded pathology encoder checkpoint.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    outputs: Dict[str, Dict[str, Any]] = {}

    for slide_id in slide_ids:
        try:
            question, t_answer = read_slide_report(reports_path, slide_id)

            if not question:
                raise KeyError(f"'question' not found or empty for slide_id='{slide_id}'")
            if not t_answer:
                raise KeyError(f"'T-answer' not found or empty for slide_id='{slide_id}'")

            wsi, region, patch, cell = load_slide_feature_quadruplet(
                slide_id=slide_id,
                wsi_feat_dir=args.wsi_feat_dir,
                region_feat_dir=args.region_feat_dir,
                patch_feat_dir=args.patch_feat_dir,
                cell_feat_dir=args.cell_feat_dir,
                cell_pt_filename=args.cell_pt_filename,
            )

            prompt = question

            response = model.generate_from_features(
                wsi=wsi,
                region=region,
                patch=patch,
                cell=cell,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            metrics = compute_text_metrics(response, t_answer)

            print("=" * 80)
            print(f"SLIDE ID: {slide_id}")
            print("-" * 80)
            print("QUESTION:")
            print(question)
            print("-" * 80)
            print("QWEN RESPONSE:")
            print(response)
            print("-" * 80)
            print("T-ANSWER:")
            print(t_answer)
            print("-" * 80)
            print("METRICS:")
            for k, v in metrics.items():
                print(f"{k}: {v:.6f}")
            print("=" * 80)

            outputs[slide_id] = {
                "question_id": slide_id,
                "question": question,
                "generated_response": response,
                "T-answer": t_answer,
                "metrics": metrics,
            }

        except Exception as e:
            outputs[slide_id] = {
                "question_id": slide_id,
                "error": str(e),
            }
            print(f"[ERR] {slide_id}: {e}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        for _, item in outputs.items():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved outputs to: {args.output_json}")


if __name__ == "__main__":
    main()