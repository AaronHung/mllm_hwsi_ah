#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
BERTSCORE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BERTSCORE_NUM_LAYERS = 9  # optimal layer for BERT-base architecture (per bert_score paper)
BERTSCORE_MAX_WORDS = 256  # pre-truncate to ~512 tokens (1 word ≈ 2 tokens)
TEST_TYPES = ("Morphology", "Caption", "Classification", "Report")
QUERY_DIRECTIONS = ("report_to_slide", "slide_to_report", "both")

from mllm_hwsi import VLProjectorConfig, MLLMHWSI
from main_utils import (
    build_hierarchical_text_reps,
    get_visual_scale_reps,
    normalize_text,
    apply_lora_to_llm,
    normalize_model_source,
)
from data_utils import (
    set_seed,
    find_slide_ids,
    load_slide_feature_quadruplet,
)


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


def extract_classification_label(text: str) -> str:
    """Extract the primary diagnosis/class label from free-text classification answers."""
    normalized = normalize_text(text)
    if not normalized:
        return ""

    # Prefer the first clause/sentence because many answers include an explanation after the label.
    primary_clause = re.split(r"[\.;\n]", normalized, maxsplit=1)[0].strip()

    cue_patterns = [
        r"(?:classification|diagnosis|answer|label|result|finding|impression)\s*(?:is|are|was|were|:)\s*(.+)$",
        r"(?:is|are|was|were|shows|suggests|indicates|reveals|points to|consistent with|represents|demonstrates|supports)\s+(.+)$",
    ]

    candidate = primary_clause
    for pattern in cue_patterns:
        match = re.search(pattern, primary_clause, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            break

    candidate = re.split(r"[,;:\\(\)]", candidate, maxsplit=1)[0].strip()
    candidate = re.sub(r"^\b(?:the|a|an)\b\s+", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate


def classification_label_match(prediction: str, reference: str) -> Dict[str, Any]:
    pred_label = extract_classification_label(prediction)
    ref_label = extract_classification_label(reference)

    if not pred_label or not ref_label:
        pred_norm = normalize_vqa_answer(prediction)
        ref_norm = normalize_vqa_answer(reference)
        return {
            "accuracy": float(pred_norm == ref_norm),
            "pred_norm": pred_norm,
            "ref_norm": ref_norm,
            "match_mode": "fallback_exact",
        }

    pred_norm = normalize_text(pred_label)
    ref_norm = normalize_text(ref_label)

    if pred_norm == ref_norm:
        return {
            "accuracy": 1.0,
            "pred_norm": pred_norm,
            "ref_norm": ref_norm,
            "match_mode": "exact_label",
        }

    if re.search(rf"\b{re.escape(pred_norm)}\b", ref_norm):
        return {
            "accuracy": 1.0,
            "pred_norm": pred_norm,
            "ref_norm": ref_norm,
            "match_mode": "prediction_in_reference",
        }

    if re.search(rf"\b{re.escape(ref_norm)}\b", pred_norm):
        return {
            "accuracy": 1.0,
            "pred_norm": pred_norm,
            "ref_norm": ref_norm,
            "match_mode": "reference_in_prediction",
        }

    return {
        "accuracy": 0.0,
        "pred_norm": pred_norm,
        "ref_norm": ref_norm,
        "match_mode": "no_match",
    }


def compute_accuracy(prediction: str, reference: str) -> Dict[str, Any]:
    return classification_label_match(prediction, reference)

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

def compute_text_metrics(prediction: str, reference: str) -> Dict[str, float]:
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference, prediction)

    smooth = SmoothingFunction().method1
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()

    weights_list = [
        (1.0, 0.0, 0.0, 0.0),       # BLEU-1
        (0.5, 0.5, 0.0, 0.0),       # BLEU-2
        (1/3, 1/3, 1/3, 0.0),       # BLEU-3
        (0.25, 0.25, 0.25, 0.25)    # BLEU-4
    ]

    bleu1 = sentence_bleu([reference_tokens], prediction_tokens, 
                          smoothing_function=smooth, weights=weights_list[0])
    bleu2 = sentence_bleu([reference_tokens], prediction_tokens, 
                          smoothing_function=smooth, weights=weights_list[1])
    bleu3 = sentence_bleu([reference_tokens], prediction_tokens, 
                          smoothing_function=smooth, weights=weights_list[2])
    bleu4 = sentence_bleu([reference_tokens], prediction_tokens, 
                          smoothing_function=smooth, weights=weights_list[3])
    meteor = meteor_score([reference_tokens], prediction_tokens)

    return {
        "ROUGE-1": float(rouge_scores["rouge1"].fmeasure),
        "ROUGE-2": float(rouge_scores["rouge2"].fmeasure),
        "ROUGE-L": float(rouge_scores["rougeL"].fmeasure),
        "BLEU-1": float(bleu1),
        "BLEU-2": float(bleu2),
        "BLEU-3": float(bleu3),
        "BLEU-4": float(bleu4),
        "METEOR": float(meteor),
    }


def aggregate_metric_statistics(outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_values: Dict[str, List[float]] = defaultdict(list)

    for item in outputs:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_values[key].append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
        }

    return summary


def parse_direction(value: str) -> str:
    key = (value or "").strip().lower()
    mapping = {name.lower(): name for name in QUERY_DIRECTIONS}
    if key not in mapping:
        allowed = ", ".join(QUERY_DIRECTIONS)
        raise argparse.ArgumentTypeError(f"Invalid --direction '{value}'. Allowed: {allowed}")
    return mapping[key]


def append_mode_to_output_path(output_path: str, mode: str) -> str:
    path = Path(output_path)
    suffixes = "".join(path.suffixes)
    stem = path.name[:-len(suffixes)] if suffixes else path.name
    mode_suffix = f"_{mode.lower()}"
    if stem.lower().endswith(mode_suffix):
        return str(path)
    return str(path.with_name(f"{stem}{mode_suffix}{suffixes}"))


def extract_report_text(record: Dict[str, Any]) -> str:
    for key in ("report", "caption", "text"):
        value = record.get(key)
        if value is not None:
            text = normalize_text(str(value))
            if text:
                return text

    question = normalize_text(str(record.get("question", "")))
    answer = normalize_text(str(record.get("T-answer", record.get("answer", ""))))
    if question and answer:
        return f"{question} {answer}".strip()
    return question or answer


def build_slide_documents(records: List[Dict[str, Any]], test_type: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        row_type = str(record.get("metadata", "")).strip()
        if row_type and row_type.lower() != test_type.lower():
            continue
        slide_id = str(record.get("slide_id") or record.get("_slide_id") or "").strip()
        if not slide_id:
            continue
        grouped[slide_id].append(record)

    documents: List[Dict[str, Any]] = []
    for slide_id in sorted(grouped):
        texts: List[str] = []
        seen_texts = set()
        for row in grouped[slide_id]:
            report_text = extract_report_text(row)
            if report_text and report_text not in seen_texts:
                texts.append(report_text)
                seen_texts.add(report_text)

        if not texts:
            continue

        documents.append(
            {
                "slide_id": slide_id,
                "records": grouped[slide_id],
                "report_text": " \n".join(texts),
            }
        )

    return documents


def combine_scale_reps(scale_reps: Dict[str, torch.Tensor]) -> torch.Tensor:
    ordered = [scale_reps[key].squeeze(0) for key in ("cell", "patch", "region", "wsi")]
    stacked = torch.stack(ordered, dim=0)
    return F.normalize(stacked.mean(dim=0), dim=-1)


def embed_slide(model: MLLMHWSI, slide_id: str, feat_dirs: Dict[str, str]) -> torch.Tensor:
    wsi, region, patch, cell = load_slide_feature_quadruplet(
        slide_id=slide_id,
        wsi_feat_dir=feat_dirs["wsi_feat_dir"],
        region_feat_dir=feat_dirs["region_feat_dir"],
        patch_feat_dir=feat_dirs["patch_feat_dir"],
        cell_feat_dir=feat_dirs["cell_feat_dir"],
        cell_pt_filename=feat_dirs["cell_pt_filename"],
    )

    encoder_device = next(model.vl_projector.parameters()).device
    wsi = wsi.to(encoder_device)
    region = region.to(encoder_device)
    patch = patch.to(encoder_device)
    cell = cell.to(encoder_device)

    with torch.no_grad():
        _, visual_reps = get_visual_scale_reps(model, wsi, region, patch, cell)
        return combine_scale_reps(visual_reps).detach().cpu()


def embed_text(model: MLLMHWSI, text: str) -> torch.Tensor:
    with torch.no_grad():
        text_reps = build_hierarchical_text_reps(
            tokenizer=model.tokenizer,
            embed_tokens=model.embed_tokens,
            answers=[text],
            device=torch.device(model.device),
            dtype=next(model.embed_tokens.parameters()).dtype,
        )
        return combine_scale_reps(text_reps).detach().cpu()


def rank_indices(query_vec: torch.Tensor, gallery_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    query_vec = F.normalize(query_vec.float(), dim=-1)
    gallery_vecs = F.normalize(gallery_vecs.float(), dim=-1)
    scores = gallery_vecs @ query_vec
    return torch.argsort(scores, descending=True), scores


def retrieval_metrics(ranks: List[int]) -> Dict[str, float]:
    total = max(1, len(ranks))
    recall_1 = sum(rank <= 1 for rank in ranks) / total
    recall_3 = sum(rank <= 3 for rank in ranks) / total
    recall_5 = sum(rank <= 5 for rank in ranks) / total
    recall_10 = sum(rank <= 10 for rank in ranks) / total
    return {
        "Recall@1": recall_1,
        "Recall@3": recall_3,
        "Recall@5": recall_5,
        "Recall@10": recall_10,
        "Recall@avg": (recall_1 + recall_3 + recall_5 + recall_10) / 4.0,
        "MRR": sum(1.0 / rank for rank in ranks) / total,
    }


def run_slide_report_retrieval(
    model: MLLMHWSI,
    records: List[Dict[str, Any]],
    feat_dirs: Dict[str, str],
    test_type: str,
    direction: str,
    top_k: int,
    output_path: str,
):
    docs = build_slide_documents(records, test_type)
    if not docs:
        raise ValueError(f"No {test_type} records were available for retrieval.")

    print(f"Building retrieval index for {len(docs)} slides")
    slide_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    for doc in docs:
        slide_embeddings.append(embed_slide(model, doc["slide_id"], feat_dirs))
        text_embeddings.append(embed_text(model, doc["report_text"]))

    slide_matrix = torch.stack(slide_embeddings, dim=0)
    text_matrix = torch.stack(text_embeddings, dim=0)
    outputs: List[Dict[str, Any]] = []

    if direction in {"report_to_slide", "both"}:
        ranks: List[int] = []
        for idx, doc in enumerate(docs):
            top_indices, scores = rank_indices(text_matrix[idx], slide_matrix)
            rank = int((top_indices == idx).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            outputs.append(
                {
                    "direction": "report_to_slide",
                    "slide_id": doc["slide_id"],
                    "report_text": doc["report_text"],
                    "rank": rank,
                    "results": [
                        {
                            "slide_id": docs[j]["slide_id"],
                            "score": float(scores[j].item()),
                            "report_text": docs[j]["report_text"],
                        }
                        for j in top_indices[:top_k].tolist()
                    ],
                }
            )

        metrics = retrieval_metrics(ranks)
        print("REPORT -> SLIDE")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        outputs.append({"direction": "report_to_slide_metrics", **metrics})

    if direction in {"slide_to_report", "both"}:
        ranks = []
        for idx, doc in enumerate(docs):
            top_indices, scores = rank_indices(slide_matrix[idx], text_matrix)
            rank = int((top_indices == idx).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            outputs.append(
                {
                    "direction": "slide_to_report",
                    "slide_id": doc["slide_id"],
                    "rank": rank,
                    "results": [
                        {
                            "slide_id": docs[j]["slide_id"],
                            "score": float(scores[j].item()),
                            "report_text": docs[j]["report_text"],
                        }
                        for j in top_indices[:top_k].tolist()
                    ],
                }
            )

        metrics = retrieval_metrics(ranks)
        print("SLIDE -> REPORT")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        outputs.append({"direction": "slide_to_report_metrics", **metrics})

    final_output = append_mode_to_output_path(output_path, "retrieval")
    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved retrieval results to: {final_output}")


def parse_test_type(value: str) -> str:
    key = (value or "").strip().lower()
    mapping = {name.lower(): name for name in TEST_TYPES}
    if key not in mapping:
        allowed = ", ".join(TEST_TYPES)
        raise argparse.ArgumentTypeError(f"Invalid --test_type '{value}'. Allowed: {allowed}")
    return mapping[key]


def append_test_type_to_output_path(output_path: str, test_type: str) -> str:
    path = Path(output_path)
    suffixes = "".join(path.suffixes)
    stem = path.name[:-len(suffixes)] if suffixes else path.name
    test_suffix = f"_{test_type.lower()}"
    if stem.lower().endswith(test_suffix):
        return str(path)
    new_name = f"{stem}{test_suffix}{suffixes}"
    return str(path.with_name(new_name))


def find_slide_id_for_record(record: Dict[str, Any], slide_ids_sorted: List[str], slide_id_set: set) -> Optional[str]:
    slide_id = str(record.get("slide_id", "")).strip()
    if slide_id and slide_id in slide_id_set:
        return slide_id

    question_id = str(record.get("question_id", ""))
    for sid in slide_ids_sorted:
        if sid in question_id:
            return sid
    return None


def load_records_for_test_type(
    reports_path: Path,
    test_type: str,
    slide_ids: List[str],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    slide_id_set = set(slide_ids)
    # Longest-first avoids partial matches when one slide_id is a substring of another.
    slide_ids_sorted = sorted(slide_ids, key=len, reverse=True)

    with reports_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_test_type = str(row.get("metadata", "")).strip()
            if row_test_type.lower() != test_type.lower():
                continue

            resolved_slide_id = find_slide_id_for_record(row, slide_ids_sorted, slide_id_set)
            if not resolved_slide_id:
                continue

            row["_line_num"] = line_num
            row["_slide_id"] = resolved_slide_id
            selected.append(row)

    return selected


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wsi_dir", type=str, required=True)
    parser.add_argument("--gt_json_path", type=str, required=True)
    parser.add_argument(
        "--test_type",
        type=parse_test_type,
        default="Report",
        help="Test subset in metadata: Morphology, Caption, Classification, or Report",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="eval",
        choices=["eval", "retrieval"],
        help="Run the generation evaluation or slide/report retrieval",
    )
    parser.add_argument(
        "--direction",
        type=parse_direction,
        default="both",
        help="Retrieval direction: report_to_slide, slide_to_report, or both",
    )
    parser.add_argument("--top_k", type=int, default=20, 
                        help="Number of top results to save for retrieval")

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

    parser.add_argument("--vl_projector_ckpt", type=str, default=None)
    parser.add_argument("--pathology_encoder_ckpt", type=str, default=None)
    parser.add_argument("--llm_lora_ckpt", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gt_json_path = Path(args.gt_json_path)
    slide_ids = find_slide_ids(args.wsi_dir)
    selected_records = load_records_for_test_type(gt_json_path, args.test_type, slide_ids)
    if not selected_records:
        raise ValueError(
            f"No records found in {gt_json_path} with metadata='{args.test_type}' that map to available slide IDs."
        )
    if args.task == "retrieval":
        final_output_json = append_mode_to_output_path(args.output_json, "retrieval")
        print(f"Running slide/report retrieval on {len(selected_records)} records.")
    else:
        final_output_json = append_test_type_to_output_path(args.output_json, args.test_type)
        print(f"Running {args.test_type} test on {len(selected_records)} records.")

    cfg = VLProjectorConfig(
        llm_dim=args.llm_dim,
        hidden_dim=args.proj_hidden_dim,
        dropout=args.dropout,
        use_layernorm=True,
        num_query_tokens=args.num_slide_tokens,
        projector_type=args.projector_type,
    )

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    model_source = normalize_model_source(args.model_name)
    model = MLLMHWSI.from_pretrained(
        source=model_source,
        projector_cfg=cfg,
        device=args.device,
        dtype=dtype,
        load_projector=True,
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

    projector_ckpt = args.vl_projector_ckpt or args.pathology_encoder_ckpt
    if projector_ckpt is not None:
        ckpt = torch.load(projector_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "vl_projector_state_dict" in ckpt:
            ckpt = ckpt["vl_projector_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.vl_projector.load_state_dict(ckpt, strict=False)
        print("Loaded VL projector checkpoint override.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()

    if args.task == "retrieval":
        feat_dirs = {
            "wsi_feat_dir": args.wsi_feat_dir,
            "region_feat_dir": args.region_feat_dir,
            "patch_feat_dir": args.patch_feat_dir,
            "cell_feat_dir": args.cell_feat_dir,
            "cell_pt_filename": args.cell_pt_filename,
        }
        run_slide_report_retrieval(
            model=model,
            records=selected_records,
            feat_dirs=feat_dirs,
            test_type=args.test_type,
            direction=args.direction,
            top_k=args.top_k,
            output_path=final_output_json,
        )
        return

    outputs: List[Dict[str, Any]] = []
    cls_predictions: List[str] = []
    cls_references: List[str] = []
    cls_output_indices: List[int] = []

    for row in selected_records:
        slide_id = str(row.get("_slide_id", "")).strip()
        question_id = str(row.get("question_id", "")).strip() or slide_id

        try:
            question = str(row.get("question", "")).strip()
            t_answer = str(row.get("T-answer", "")).strip()

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

            if args.test_type == "Classification":
                metrics = compute_accuracy(response, t_answer)
            else:
                metrics = compute_text_metrics(response, t_answer)

            print("=" * 80)
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
            print("METRICS:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.6f}")
                else:
                    print(f"{k}: {v}")
            print("=" * 80)

            item = {
                "question_id": question_id,
                "slide_id": slide_id,
                "metadata": args.test_type,
                "question": question,
                "generated_response": response,
                "T-answer": t_answer,
                "metrics": metrics,
            }
            outputs.append(item)

            if args.test_type == "Classification":
                cls_predictions.append(response)
                cls_references.append(t_answer)
                cls_output_indices.append(len(outputs) - 1)

        except Exception as e:
            outputs.append({
                "question_id": question_id,
                "slide_id": slide_id,
                "metadata": args.test_type,
                "error": str(e),
            })
            print(f"[ERR] {slide_id}: {e}")

    if args.test_type == "Classification" and cls_predictions:
        bert_results = compute_bertscore_accuracy(
            predictions=cls_predictions,
            references=cls_references,
            device=args.device,
            return_individual=True,
        )
        p_list = bert_results.get("precision_per_sample", [])
        r_list = bert_results.get("recall_per_sample", [])
        f1_list = bert_results.get("f1_per_sample", [])

        for i, out_idx in enumerate(cls_output_indices):
            item_metrics = outputs[out_idx].setdefault("metrics", {})
            if i < len(p_list):
                item_metrics["BERTScore-Precision"] = p_list[i]
            if i < len(r_list):
                item_metrics["BERTScore-Recall"] = r_list[i]
            if i < len(f1_list):
                item_metrics["BERTScore-F1"] = f1_list[i]

        mean_acc = sum(
            outputs[i].get("metrics", {}).get("accuracy", 0.0)
            for i in cls_output_indices
        ) / max(1, len(cls_output_indices))

        print("=" * 80)
        print("CLASSIFICATION AGGREGATE METRICS")
        print(f"Accuracy (mean): {mean_acc:.6f}")
        print(f"BERTScore Precision (mean): {bert_results.get('precision', 0.0):.6f}")
        print(f"BERTScore Recall (mean): {bert_results.get('recall', 0.0):.6f}")
        print(f"BERTScore F1 (mean): {bert_results.get('f1', 0.0):.6f}")
        print("=" * 80)

    aggregate_stats = aggregate_metric_statistics(outputs)
    if aggregate_stats:
        print("=" * 80)
        print("AGGREGATE METRICS (MIN / MAX / AVG)")
        for metric_name in sorted(aggregate_stats):
            stat = aggregate_stats[metric_name]
            print(
                f"{metric_name}: "
                f"min={stat['min']:.6f}, "
                f"max={stat['max']:.6f}, "
                f"avg={stat['avg']:.6f}, "
                f"count={int(stat['count'])}"
            )
        print("=" * 80)

    os.makedirs(os.path.dirname(final_output_json) or ".", exist_ok=True)
    with open(final_output_json, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_output_json = str(Path(final_output_json).with_name(f"{Path(final_output_json).stem}_summary.json"))
    summary_payload = {
        "task": args.task,
        "test_type": args.test_type,
        "num_samples": len(outputs),
        "num_success": sum(1 for item in outputs if "metrics" in item),
        "num_errors": sum(1 for item in outputs if "error" in item),
        "aggregate_metrics": aggregate_stats,
    }
    with open(summary_output_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {final_output_json}")
    print(f"Saved summary to: {summary_output_json}")


if __name__ == "__main__":
    main()