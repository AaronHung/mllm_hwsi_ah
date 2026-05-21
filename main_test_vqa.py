#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from bert_score import score
BERTSCORE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BERTSCORE_NUM_LAYERS = 9  # optimal layer for BERT-base architecture (per bert_score paper)
BERTSCORE_MAX_WORDS = 256  # pre-truncate to ~512 tokens (1 word ≈ 2 tokens)
from mllm_hwsi import VLProjectorConfig, MLLMHWSIQWEN
from data_utils import (
    set_seed,
    find_slide_ids,
    load_slide_feature_quadruplet,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


# =========================================================
# Metrics
# =========================================================

def normalize_vqa_answer(text: str) -> str:
    """
    Normalizes VQA answers for exact-match accuracy.

    Handles common multiple-choice outputs such as:
      - "A", "(C)", "Option D", "Answer: B"
    Falls back to normalized free text for non-MCQ answers.
    """
    t = normalize_text(text)
    if not t:
        return ""

    t_upper = t.upper()

    if re.fullmatch(r"[A-E]", t_upper):
        return t_upper

    m = re.search(r"\b(?:OPTION\s*|ANSWER\s*[:\-]?\s*)?\(?([A-E])\)?\b", t_upper)
    if m:
        return m.group(1)

    m = re.match(r"^\(?([A-E])\)?[\s\.:\)\-].*", t_upper)
    if m:
        return m.group(1)

    return t.lower()


def compute_accuracy(prediction: str, reference: str) -> Dict[str, float]:
    pred_norm = normalize_vqa_answer(prediction)
    ref_norm = normalize_vqa_answer(reference)
    acc = float(pred_norm == ref_norm)
    return {
        "accuracy": acc,
        "pred_norm": pred_norm,
        "ref_norm": ref_norm,
    }

def clean_text_list(texts):
    cleaned = []
    for x in texts:
        if x is None:
            x = ""
        x = str(x).strip()
        if len(x) == 0:
            x = "empty"
        cleaned.append(x)
    return cleaned

def compute_bertscore_accuracy(
    predictions,
    references,
    model_type=BERTSCORE_MODEL,
    num_layers=BERTSCORE_NUM_LAYERS,
    lang="en",
    device=None,
    batch_size=16,
    max_length=512,
    return_individual=False,
):
    """
    Compute BERTScore precision, recall, and F1 for
    pathology / WSI / biomedical text evaluation.

    Parameters
    ----------
    predictions : List[str]
        Generated LLM outputs.

    references : List[str]
        Ground-truth reference texts.

    model_type : str
        HuggingFace model name.
        Default:
        microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

    lang : str
        Language code.

    device : str or None
        "cuda", "cpu", or None for auto-detect.

    batch_size : int
        Batch size for BERTScore computation.

    return_individual : bool
        If True, returns per-sample scores.

    Returns
    -------
    dict
        Mean precision, recall, and F1 scores.
    """

    assert len(predictions) == len(references), (
        "predictions and references must have same length"
    )

    predictions = clean_text_list(predictions)
    references = clean_text_list(references)

    # Pre-truncate to avoid tokenizer OverflowError.
    # bert_score doesn't forward max_length to the tokenizer, so we truncate
    # by whitespace words as a safe approximation (1 word ≈ 1-2 subword tokens).
    predictions = [" ".join(t.split()[:BERTSCORE_MAX_WORDS]) for t in predictions]
    references = [" ".join(t.split()[:BERTSCORE_MAX_WORDS]) for t in references]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = score(
        cands=predictions,
        refs=references,
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        device=device,
        batch_size=batch_size,
        verbose=True,
    )

    results = {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }

    if return_individual:
        results.update({
            "precision_per_sample": [round(x, 4) for x in P.tolist()],
            "recall_per_sample": [round(x, 4) for x in R.tolist()],
            "f1_per_sample": [round(x, 4) for x in F1.tolist()],
        })

    return results

def _load_records_from_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
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
                rows.append(row)
    return rows


def _resolve_available_slide_id(candidate_ids: List[str], available_slide_ids: set) -> str:
    for candidate in candidate_ids:
        if not candidate:
            continue
        if candidate in available_slide_ids:
            return candidate
        for available_slide_id in available_slide_ids:
            if candidate in available_slide_id or available_slide_id in candidate:
                return available_slide_id
    raise ValueError(f"Could not match any candidate slide id {candidate_ids} to available slides")


def _extract_vqa_qa(rec: Dict[str, Any], available_slide_ids: set | None = None) -> Tuple[str, str, str, str]:
    """
    Extract (sample_id, slide_id, question, answer) from VQA records.
    
    Supports two formats:
    1. Conversation format: has 'conversations' array with human/gpt turns
    2. Report format: has 'question' and 'T-answer' fields directly
    """
    sample_id = str(rec.get("id") or rec.get("question_id") or "")
    question = None
    answer = None
    slide_id = None
    candidate_slide_ids: List[str] = []
    
    # ===== Try conversation format first =====
    conv = rec.get("conversations", [])
    if isinstance(conv, list) and len(conv) > 0:
        # Conversation format detected
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).strip().lower()
            value = normalize_text(str(turn.get("value", "")).replace("<image>", " ").replace("<Image>", " "))
            if role in {"human", "user"} and value and question is None:
                question = value
            if role in {"gpt", "assistant"} and value and answer is None:
                answer = value
            if question is not None and answer is not None:
                break
        
        # Extract slide_id from image field (conversation format)
        image = rec.get("image")
        if image is not None:
            image_stem = Path(str(image)).stem
            candidate_slide_ids.extend([image_stem, str(image)])
        else:
            # Fallback: try to extract from sample_id or generate
            if sample_id:
                candidate_slide_ids.append(sample_id)
    
    # ===== If conversation format failed or incomplete, try report format =====
    if question is None or answer is None:
        q_field = rec.get("question")
        a_field = rec.get("T-answer")
        
        if q_field is not None and a_field is not None:
            # Report format detected
            question = normalize_text(str(q_field))
            answer = normalize_text(str(a_field))
            
            if sample_id:
                candidate_slide_ids.append(sample_id)
        else:
            # Both formats failed
            raise ValueError(
                "Could not extract question/answer. "
                "Expected 'conversations' array (conversation format) or "
                "'question'+'T-answer' fields (report format)"
            )
    
    if available_slide_ids:
        if sample_id:
            candidate_slide_ids.append(sample_id)
        question_id = str(rec.get("question_id") or "")
        if question_id:
            candidate_slide_ids.append(question_id)
        if rec.get("image") is not None:
            candidate_slide_ids.append(str(rec.get("image")))
        slide_id = _resolve_available_slide_id(candidate_slide_ids, available_slide_ids)
    elif candidate_slide_ids:
        slide_id = candidate_slide_ids[0]

    if not slide_id:
        raise ValueError("Could not extract slide_id from record")
    
    if not sample_id:
        sample_id = f"{slide_id}_{abs(hash(question)) % 100000000}"
    
    return sample_id, slide_id, question, answer

def _load_records_for_available_slides(path: Path, available_slides: set) -> List[Dict[str, Any]]:
    """
    Load records from JSONL/JSON that match available slides (memory-efficient).
    Only records with slide_id containing any available slide name are loaded.
    """
    filtered_rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try parsing as JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict):
                    try:
                        _, slide_id, _, _ = _extract_vqa_qa(row, available_slides)
                        if slide_id in available_slides:
                            filtered_rows.append(row)
                    except Exception:
                        pass
            return filtered_rows
        elif isinstance(parsed, dict):
            try:
                _, slide_id, _, _ = _extract_vqa_qa(parsed, available_slides)
                if slide_id in available_slides:
                    filtered_rows.append(parsed)
            except Exception:
                pass
            return filtered_rows
    except json.JSONDecodeError:
        pass

    # Parse as JSONL
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    try:
                        _, slide_id, _, _ = _extract_vqa_qa(row, available_slides)
                        if slide_id in available_slides:
                            filtered_rows.append(row)
                    except Exception:
                        pass
            except json.JSONDecodeError:
                continue

    return filtered_rows




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

    parser.add_argument("--max_new_tokens", type=int, default=512)
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

    parser.add_argument("--use_accuracy", type=float, default=False)

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

    # Load available slide IDs from WSI directory
    available_slides_ids = find_slide_ids(args.wsi_dir)
    available_slides_set = set(available_slides_ids)
    print(f"[VQA] found {len(available_slides_ids)} slides in {args.wsi_dir}")

    reports_path = Path(args.report_json)
    # Load only records matching available slides (memory-efficient)
    records = _load_records_for_available_slides(reports_path, available_slides_set)
    if not records:
        raise RuntimeError(f"No records found matching available slides in {reports_path}")

    print(f"[VQA] loaded {len(records)} records matching {len(available_slides_ids)} available slides")

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
    outputs: List[Dict[str, Any]] = []
    
    slide_accuracies: Dict[str, Dict[str, Any]] = {}  # slide_id -> {correct, total, accuracy}
    total_scored = 0
    total_correct = 0
    total_skipped = 0
    total_unavailable = 0

    responses, references = [], []
    for i, rec in enumerate(records, start=1):
        slide_id = "unknown"
        sample_id = "unknown"
        try:
            sample_id, slide_id, question, t_answer = _extract_vqa_qa(rec, available_slides_set)
        except (KeyError, ValueError) as e:
            total_skipped += 1
            rec_id = str(rec.get("id") or rec.get("question_id") or f"row_{i}")
            print(f"[SKIP] record {i}/{len(records)} (id={rec_id}): {type(e).__name__}: {e}")
            continue
        except Exception as e:
            total_skipped += 1
            rec_id = str(rec.get("id") or rec.get("question_id") or f"row_{i}")
            print(f"[SKIP] record {i}/{len(records)} (id={rec_id}): unexpected error: {e}")
            continue

        try:
            wsi, region, patch, cell = load_slide_feature_quadruplet(
                slide_id=slide_id,
                wsi_feat_dir=args.wsi_feat_dir,
                region_feat_dir=args.region_feat_dir,
                patch_feat_dir=args.patch_feat_dir,
                cell_feat_dir=args.cell_feat_dir,
                cell_pt_filename=args.cell_pt_filename,
            )
        except FileNotFoundError as e:
            # Features not available for this slide
            total_unavailable += 1
            rec_id = str(rec.get("id") or rec.get("question_id") or f"row_{i}")
            print(f"[UNAVAIL] record {i}/{len(records)} (id={rec_id}, slide={slide_id}): features not found")
            continue
        except Exception as e:
            # Other errors during feature loading
            total_skipped += 1
            print(f"[ERR] record {i}/{len(records)} (sample={sample_id}, slide={slide_id}): {type(e).__name__}: {e}")
            outputs.append({
                "sample_id": sample_id,
                "question_id": sample_id,
                "error": str(e),
            })
            continue

        try:
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

            
            responses.append(response)
            references.append(t_answer)

            if args.use_accuracy:
                metrics = compute_accuracy(response, t_answer)
                total_scored += 1
                total_correct += int(metrics["accuracy"])
                running_acc = total_correct / max(total_scored, 1)

                # Track per-slide accuracy
                if slide_id not in slide_accuracies:
                    slide_accuracies[slide_id] = {"correct": 0, "total": 0}
                slide_accuracies[slide_id]["total"] += 1
                slide_accuracies[slide_id]["correct"] += int(metrics["accuracy"])
                slide_accuracies[slide_id]["accuracy"] = slide_accuracies[slide_id]["correct"] / slide_accuracies[slide_id]["total"]

            print("=" * 80)
            print(f"[{i}/{len(records)}] SAMPLE ID: {sample_id}")
            print(f"SLIDE ID: {slide_id}")
            print("-" * 80)
            print("QUESTION:")
            print(question)
            print("-" * 80)
            print("MLLM-HWSI RESPONSE:")
            print(response)
            print("-" * 80)
            print("T-ANSWER:")
            print(t_answer)
            print("-" * 80)

            if args.use_accuracy:
                print(f"ACCURACY: {metrics['accuracy']:.6f} | RUNNING_ACC: {running_acc:.6f}")
                print("=" * 80)

                outputs.append({
                    "sample_id": sample_id,
                    "question_id": slide_id,
                    "question": question,
                    "generated_response": response,
                    "T-answer": t_answer,
                    "metrics": metrics,
                })

            outputs.append({
                "sample_id": sample_id,
                "question_id": slide_id,
                "question": question,
                "generated_response": response,
                "T-answer": t_answer,
            })

        except Exception as e:
            total_skipped += 1
            print(f"[ERR] record {i}/{len(records)} (sample={sample_id}, slide={slide_id}): {type(e).__name__}: {e}")
            outputs.append({
                "sample_id": sample_id,
                "question_id": sample_id,
                "error": str(e),
            })

    if args.use_accuracy:
        final_acc = total_correct / max(total_scored, 1)
        avg_slide_acc = sum(s["accuracy"] for s in slide_accuracies.values()) / max(len(slide_accuracies), 1) if slide_accuracies else 0.0
    else:    
        print(f"[VQA] Computing BERTScore accuracy for {len(responses)} scored samples...")
        metrics = compute_bertscore_accuracy(responses, references, device=args.device, return_individual=True)

    if args.use_accuracy:
        summary = {
            "type": "summary",
            "total_records": len(records),
            "total_skipped": total_skipped,
            "total_unavailable": total_unavailable,
            "total_scored": total_scored,
            "total_correct": total_correct,
            "accuracy": final_acc,
            "num_slides": len(slide_accuracies),
            "avg_slide_accuracy": avg_slide_acc,
            "per_slide_accuracy": slide_accuracies,
        }
    else:
        summary = {
            "type": "summary",
            "total_records": len(records),
            "total_skipped": total_skipped,
            "total_unavailable": total_unavailable,
            "total_scored": total_scored,
            "total_correct": total_correct,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        }

    outputs.append(summary)

    print("=" * 80)
    if args.use_accuracy:
        print(f"overall_accuracy : {final_acc:.6f}")
        print(f"num_slides       : {len(slide_accuracies)}")
        print(f"avg_slide_acc    : {avg_slide_acc:.6f}")
        print("\n[PER-SLIDE ACCURACY]")
        for sid in sorted(slide_accuracies.keys()):
            acc_data = slide_accuracies[sid]
            print(f"  {sid}: {acc_data['accuracy']:.6f} ({acc_data['correct']}/{acc_data['total']})")
    else:
        print("[VQA SUMMARY]")
        print(f"total_records    : {len(records)}")
        print(f"total_skipped    : {total_skipped} (extraction/inference errors)")
        print(f"total_unavailable: {total_unavailable} (features not found)")
        print(f"total_scored     : {total_scored}")
        print(f"total_correct    : {total_correct}")
        print(f"Precision        : {metrics['precision']:.6f}")
        print(f"Recall           : {metrics['recall']:.6f}")
        print(f"F1               : {metrics['f1']:.6f}")

    print("=" * 80)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved outputs to: {args.output_json}")


if __name__ == "__main__":
    main()