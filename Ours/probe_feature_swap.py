#!/usr/bin/env python3
"""Feature-swap test for MLLM-HWSI.

The prompt is held constant while the four-level feature quadruplet is swapped
between two slides.  If the projector tokens change but the generated answer
does not, the remaining bottleneck is downstream LLM conditioning rather than
the feature extractor or a dead projector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ours.run_pilot40_four_conditions import load_features, load_frozen_model


PROMPTS = {
    "binary": (
        "Classify this whole-slide image as Renal cancer or Breast cancer. "
        "Answer with exactly one class only: Renal or Breast."
    ),
    "four": (
        "Classify this whole-slide image as one of the following: "
        "clear cell renal cell carcinoma, papillary renal cell carcinoma, "
        "invasive ductal carcinoma, or invasive lobular carcinoma. "
        "Answer with exactly one diagnosis only."
    ),
    "open": (
        "What is the most likely cancer type in this whole-slide image? "
        "Answer with the most specific diagnosis you can infer from the visual features."
    ),
}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _visual_tokens(model, features: dict[str, torch.Tensor]) -> torch.Tensor:
    projector_dtype = next(model.vl_projector.parameters()).dtype
    with torch.inference_mode():
        return model.vl_projector(
            wsi=features["wsi"].to(dtype=projector_dtype),
            region=features["region"].to(dtype=projector_dtype),
            patch=features["patch"].to(dtype=projector_dtype),
            cell=features["cell"].to(dtype=projector_dtype),
        ).float().cpu()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--slide-a", required=True)
    parser.add_argument("--slide-b", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", default="binary,four,open")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.slide_a == args.slide_b:
        raise ValueError("--slide-a and --slide-b must be different")

    features_a = load_features(args.slide_a, args.feature_root, args.cell_root)
    features_b = load_features(args.slide_b, args.feature_root, args.cell_root)
    model, loading_info = load_frozen_model(args.model_dir, args.device)
    model.eval()

    device = torch.device(args.device)
    tokens_a = _visual_tokens(model, {k: v.to(device) for k, v in features_a.items()})
    tokens_b = _visual_tokens(model, {k: v.to(device) for k, v in features_b.items()})
    token_cosine = _cosine(tokens_a, tokens_b)

    print(f"Projector loading: {loading_info}")
    print(f"Slide A: {args.slide_a} R={features_a['region'].shape[1]}")
    print(f"Slide B: {args.slide_b} R={features_b['region'].shape[1]}")
    print(f"Visual-token cosine (A vs B): {token_cosine:.6f}")

    results = {
        "slide_a": args.slide_a,
        "slide_b": args.slide_b,
        "visual_token_cosine": token_cosine,
        "projector_loading": loading_info,
        "prompts": {},
    }

    for prompt_id in [p.strip() for p in args.prompts.split(",") if p.strip()]:
        if prompt_id not in PROMPTS:
            raise ValueError(f"Unknown prompt id {prompt_id!r}; choose from {sorted(PROMPTS)}")
        prompt = PROMPTS[prompt_id]
        print(f"\n[{prompt_id}] {prompt}")
        answers = {}
        for slide_id, features in ((args.slide_a, features_a), (args.slide_b, features_b)):
            with torch.inference_mode():
                answer = model.generate_from_features(
                    **features,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                )
            answers[slide_id] = str(answer)
            print(f"{slide_id}: {answer}")
        same_answer = answers[args.slide_a].strip() == answers[args.slide_b].strip()
        print(f"Exact answer match: {same_answer}")
        results["prompts"][prompt_id] = {
            "prompt": prompt,
            "answers": answers,
            "exact_answer_match": same_answer,
        }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
