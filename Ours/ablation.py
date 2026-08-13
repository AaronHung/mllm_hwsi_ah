from pathlib import Path
import json

import torch

from mllm_hwsi import MLLMHWSI


SLIDE_ID = "TCGA-B3-3925-01Z-00-DX1"
DEVICE = "cuda:0"
MODEL_DIR = Path("checkpoints/mllm_hwsi").resolve()
FEATURE_ROOT = Path("outputs/tcga_b3_3925/features")
CELL_ROOT = Path("outputs/tcga_b3_3925/cells")
OUTPUT_PATH = Path("outputs/tcga_b3_3925/prompt_ablation.json")


def load_tensor(path):
    assert path.is_file(), f"Missing file: {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


def predicted_class(answer):
    text = answer.lower()
    if "papillary renal" in text:
        return "KIRP"
    if "clear cell renal" in text:
        return "KIRC"
    if "chromophobe renal" in text:
        return "KICH"
    return "OTHER"


assert torch.cuda.is_available(), "CUDA is unavailable"

wsi = load_tensor(
    FEATURE_ROOT / "wsi" / f"{SLIDE_ID}.pt"
).float()
region = load_tensor(
    FEATURE_ROOT / "region_4k" / f"{SLIDE_ID}.pt"
).float()

patch_payload = load_tensor(
    FEATURE_ROOT / "patches_filtered" / f"{SLIDE_ID}.pt"
)
cell_payload = load_tensor(
    CELL_ROOT / SLIDE_ID / "encoded_cell_features.pt"
)

patch = patch_payload["selected_features"].float()
cell = cell_payload["encoded_cell_features"].float()

assert torch.equal(
    patch_payload["selected_indices"].long(),
    cell_payload["selected_patch_cell_level_indices"].long(),
), "Patch/cell indices do not match"

features = {
    "wsi": wsi.unsqueeze(0),
    "region": region.unsqueeze(0),
    "patch": patch.unsqueeze(0),
    "cell": cell.unsqueeze(0),
}

for name, tensor in features.items():
    assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"

prompts = [
    (
        "P0_open_diagnosis",
        "What is the most likely cancer type in this whole-slide image? "
        "Answer with the most specific diagnosis only.",
    ),
    (
        "P1_renal_context",
        "This whole-slide image is from a renal tumor. "
        "What is the most likely diagnosis? Answer with one diagnosis only.",
    ),
    (
        "P2_rcc_subtype_task",
        "What is the histologic subtype of this renal cell carcinoma? "
        "Answer with one subtype only.",
    ),
    (
        "P3_closed_order_middle",
        "Classify this renal tumor as clear cell renal cell carcinoma, "
        "papillary renal cell carcinoma, or chromophobe renal cell carcinoma. "
        "Answer with one class only.",
    ),
    (
        "P4_closed_order_last",
        "Classify this renal tumor as chromophobe renal cell carcinoma, "
        "clear cell renal cell carcinoma, or papillary renal cell carcinoma. "
        "Answer with one class only.",
    ),
    (
        "P5_closed_order_first",
        "Classify this renal tumor as papillary renal cell carcinoma, "
        "chromophobe renal cell carcinoma, or clear cell renal cell carcinoma. "
        "Answer with one class only.",
    ),
]

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model from:", MODEL_DIR)

model, loading_info = MLLMHWSI.from_pretrained(
    str(MODEL_DIR),
    device=DEVICE,
    dtype=torch.float16,
    local_files_only=True,
    strict_projector=True,
    return_loading_info=True,
)
model.eval()
print("Projector loading:", loading_info)

results = []
for prompt_id, prompt in prompts:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    answer = model.generate_from_features(
        **features,
        prompt=prompt,
        max_new_tokens=32,
        do_sample=False,
        num_beams=1,
    ).strip()

    label = predicted_class(answer)
    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "answer": answer,
        "predicted_class": label,
        "ground_truth": "KIRP",
        "correct": label == "KIRP",
    }
    results.append(result)

    print("\n" + "=" * 72)
    print(prompt_id)
    print("Prompt:", prompt)
    print("Answer:", answer)
    print("Parsed class:", label)
    print("Correct:", label == "KIRP")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

closed_set = [row for row in results if row["prompt_id"].startswith(("P3", "P4", "P5"))]
closed_labels = [row["predicted_class"] for row in closed_set]

print("\n" + "=" * 72)
print("PROMPT ABLATION SUMMARY")
for row in results:
    print(
        f'{row["prompt_id"]:24s} -> '
        f'{row["predicted_class"]:5s} | {row["answer"]}'
    )
print("Closed-set predictions:", closed_labels)
print("Closed-set order stable:", len(set(closed_labels)) == 1)
print("Saved:", OUTPUT_PATH)
