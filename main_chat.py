#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA Chatbot — Gradio front-end backed by the MLLMHWSI model.

Users select:
  • A single WSI file  (used to derive the slide_id)
  • A base features directory  (must contain wsi/, region/, patch/, cells/ sub-dirs)
    • A model folder/repo for MLLM-HWSI (includes VL projector artifacts)

If the expected .pt feature files for the chosen slide are absent the app
launches feature extraction in a background thread, streams live log output
into the chat, and blocks further questions until extraction finishes.

Once features are ready the user can type questions and receive answers from
the model in the same chat window.
"""

import os
import re
import sys
import json
import queue
import shlex
import threading
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import gradio as gr
import base64
from main_utils import normalize_model_source as _normalize_model_source

# ---------------------------------------------------------------------------
# Lazy model container — loaded once per session
# ---------------------------------------------------------------------------

class _ModelHandle:
    def __init__(self):
        self.model = None
        self.slide_id: Optional[str] = None
        self.features: Optional[Tuple] = None   # (wsi, region, patch, cell) tensors
        self._lock = threading.Lock()

    def reset(self):
        self.model = None
        self.slide_id = None
        self.features = None


_handle = _ModelHandle()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEAT_SUBDIRS = {
    "wsi":    "wsi",
    "regions": "region_4k",
    "patches":  "patches_filtered",
    "cells":   "cells",
}
CELL_PT_FILENAME = "encoded_cell_features.pt"
WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}
DEFAULT_JSON_PATH = "/TCGA/WSI-Bench-train-Report-only.jsonl"
DEFAULT_MAG_LEVEL = "mag20x"
DEFAULT_MAG_NUM = 20
DEFAULT_HIPT_256_CKPT = "../HIPT_ckpts/vit_256_small_dino.pth"
DEFAULT_HIPT_4096_CKPT = "../HIPT_ckpts/vit_4096_xs_dino.pth"
DEFAULT_CONCH_CKPT = "../CONCH_ckpt/pytorch_model.bin"
DEFAULT_CELLVIT_CKPT = "../CELLVITpp_ckpts/CellViT-256-x40-AMP.pth"


def _slide_id_from_path(wsi_path: str) -> str:
    return Path(wsi_path).stem


def _normalize_selector_path(value) -> str:
    """Normalize UI-provided value to a single path string."""
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _initial_dialog_dir(current_path: str) -> str:
    p = Path(current_path).expanduser() if current_path else Path.cwd()
    if p.exists():
        return str(p if p.is_dir() else p.parent)
    if p.parent.exists():
        return str(p.parent)
    return str(Path.cwd())


def _pick_file_dialog(current_path: str, title: str, filetypes=None) -> str:
    """Open a native file picker dialog and return selected path or current value."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=_initial_dialog_dir(current_path),
            filetypes=filetypes or [("All files", "*.*")],
        )
        root.destroy()
        return selected or current_path
    except Exception:
        return current_path


def _pick_folder_dialog(current_path: str, title: str) -> str:
    """Open a native folder picker dialog and return selected path or current value."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title=title,
            initialdir=_initial_dialog_dir(current_path),
        )
        root.destroy()
        return selected or current_path
    except Exception:
        return current_path


def _feat_dirs(base_dir: str):
    b = Path(base_dir)
    return {
        "wsi_feat_dir":    str(b / FEAT_SUBDIRS["wsi"]),
        "region_feat_dir": str(b / FEAT_SUBDIRS["regions"]),
        "patch_feat_dir":  str(b / FEAT_SUBDIRS["patches"]),
        "cell_feat_dir":   str(b / FEAT_SUBDIRS["cells"]),
    }


def _features_exist(slide_id: str, base_dir: str) -> Tuple[bool, List[str]]:
    """Return (all_present, missing_description)."""
    dirs = _feat_dirs(base_dir)
    checks = [
        Path(dirs["wsi_feat_dir"])    / f"{slide_id}.pt",
        Path(dirs["region_feat_dir"]) / f"{slide_id}.pt",
        Path(dirs["patch_feat_dir"])  / f"{slide_id}.pt",
        Path(dirs["cell_feat_dir"])   / slide_id / CELL_PT_FILENAME,
    ]
    missing = [str(p) for p in checks if not p.exists()]
    return (len(missing) == 0), missing


def _resolve_ckpt_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent / path).resolve()


def _available_device_choices() -> Tuple[list, str]:
    choices = []
    default_value = "cpu"

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            dev = f"cuda:{idx}"
            try:
                free_mem, _ = torch.cuda.mem_get_info(idx)
                mem_gb = free_mem / (1024 ** 3)
                label = f"{dev.upper()} ({mem_gb:.1f}GB free)"
            except Exception:
                total_mem = torch.cuda.get_device_properties(idx).total_memory
                mem_gb = total_mem / (1024 ** 3)
                label = f"{dev.upper()} ({mem_gb:.1f}GB)"
            choices.append((label, dev))
        if choices:
            default_value = choices[0][1]

    choices.append(("CPU", "cpu"))
    return choices, default_value


def _load_model(
    model_name: str,
    device: str,
    encoder_ckpt: Optional[str],
    llm_dim: int,
    proj_hidden_dim: int,
    dropout: float,
    num_slide_tokens: int,
    projector_type: str,
):
    """Load and return a ready-to-use MLLMHWSI model."""
    from mllm_hwsi import VLProjectorConfig, MLLMHWSI

    cfg = VLProjectorConfig(
        llm_dim=llm_dim,
        hidden_dim=proj_hidden_dim,
        dropout=dropout,
        use_layernorm=True,
        num_query_tokens=num_slide_tokens,
        projector_type=projector_type,
    )
    model_source = _normalize_model_source(model_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = MLLMHWSI.from_pretrained(
        source=model_source,
        projector_cfg=cfg,
        device=device,
        dtype=dtype,
        load_projector=True,
    )

    if encoder_ckpt:
        ckpt = torch.load(encoder_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "vl_projector_state_dict" in ckpt:
            ckpt = ckpt["vl_projector_state_dict"]
        else:
            ckpt = ckpt.get("state_dict", ckpt)
        model.vl_projector.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


def _load_features(slide_id: str, base_dir: str) -> Tuple:
    from data_utils import load_slide_feature_quadruplet
    dirs = _feat_dirs(base_dir)
    return load_slide_feature_quadruplet(
        slide_id=slide_id,
        cell_pt_filename=CELL_PT_FILENAME,
        **dirs,
    )


# ---------------------------------------------------------------------------
# Feature extraction in a background thread (streams stdout to a queue)
# ---------------------------------------------------------------------------

def _run_cmd_and_stream(cmd: List[str], log_queue: queue.Queue, cwd: Optional[Path] = None) -> int:
    log_queue.put(f"[EXTRACTION] Running: {' '.join(shlex.quote(c) for c in cmd)}\n")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_queue.put(line)
    proc.wait()
    return int(proc.returncode)


def _run_extraction_pipeline(
    wsi_path: str,
    base_dir: str,
    reports_path: str,
    hipt_4096_ckpt: str,
    hipt_256_ckpt: str,
    conch_ckpt: str,
    cellvit_ckpt: str,
    log_queue: queue.Queue,
):
    """Run extraction for a single slide using defaults from run_ext_pipeline.sh."""
    wsi_file = Path(wsi_path).expanduser().resolve()
    slide_id = wsi_file.stem
    project_root = Path(__file__).parent.resolve()
    base_features = Path(base_dir).expanduser().resolve()
    regions_dir = base_features.parent / f"regions_{DEFAULT_MAG_LEVEL}"

    reports = Path(reports_path).expanduser().resolve()
    hipt_256_path = _resolve_ckpt_path(hipt_256_ckpt)
    hipt_4096_path = _resolve_ckpt_path(hipt_4096_ckpt)
    conch_path = _resolve_ckpt_path(conch_ckpt)
    cellvit_path = _resolve_ckpt_path(cellvit_ckpt)

    try:
        if not reports.exists():
            log_queue.put(f"[EXTRACTION] ✗ reports_path not found: {reports}\n")
            return

        for label, ckpt in [
            ("HIPT 4096", hipt_4096_path),
            ("HIPT 256", hipt_256_path),
            ("CONCH", conch_path),
            ("CellViT", cellvit_path),
        ]:
            if not ckpt.is_file():
                log_queue.put(f"[EXTRACTION] ✗ {label} checkpoint not found: {ckpt}\n")
                return

        with tempfile.TemporaryDirectory(prefix="vqa_extract_") as tmp_root:
            tmp_root_p = Path(tmp_root)
            tmp_wsi_dir = tmp_root_p / "wsi"
            tmp_wsi_dir.mkdir(parents=True, exist_ok=True)
            (tmp_wsi_dir / wsi_file.name).symlink_to(wsi_file)

            regions_dir.mkdir(parents=True, exist_ok=True)
            base_features.mkdir(parents=True, exist_ok=True)

            step1 = [
                sys.executable,
                str(project_root / "ext_regions.py"),
                "--source", str(tmp_wsi_dir),
                "--save_dir", str(regions_dir),
                "--mag_level", str(DEFAULT_MAG_NUM),
                "--patch_size", "4096",
                "--step_size", "4096",
                "--seg",
                "--patch",
                "--stitch",
            ]
            rc = _run_cmd_and_stream(step1, log_queue, cwd=project_root)
            if rc != 0:
                log_queue.put(f"[EXTRACTION] ✗ ext_regions.py failed with code {rc}.\n")
                return

            step2 = [
                sys.executable,
                str(project_root / "ext_feats_conch_hierar_par.py"),
                "--wsi_dir", str(tmp_wsi_dir),
                "--reports_path", str(reports),
                "--h5_dir_4096", str(regions_dir / "patches"),
                "--hipt_repo", str(project_root / "HIPT_4K"),
                "--checkpoint256", str(hipt_256_path),
                "--checkpoint4k", str(hipt_4096_path),
                "--trident_repo", str(project_root / "trident"),
                "--conch_ckpt_path", str(conch_path),
                "--conch_batch_size", "128",
                "--encoder_name", "conch_v1",
                "--conch_model_cfg", "conch_ViT-B-16",
                "--out_dir", str(base_features),
                "--patch_size", "256",
                "--extract_patch_features",
                "--n_diss_features", "32",
                "--top_k", "16",
                "--process_single_slide", slide_id,
                "--all_gpus",
                "--workers_per_gpu", "1",
            ]
            rc = _run_cmd_and_stream(step2, log_queue, cwd=project_root)
            if rc != 0:
                log_queue.put(f"[EXTRACTION] ✗ ext_feats_conch_hierar_par.py failed with code {rc}.\n")
                return

            coords_path = base_features / "coords_region4096_valid" / f"{slide_id}.h5"
            selected_path = base_features / "patches_filtered" / f"{slide_id}.pt"
            if not coords_path.exists() or not selected_path.exists():
                log_queue.put(
                    "[EXTRACTION] ✗ Missing intermediate files for cell extraction: "
                    f"{coords_path} and/or {selected_path}.\n"
                )
                return

            tmp_region_coords = tmp_root_p / "coords_region4096_valid"
            tmp_selected = tmp_root_p / "patches_filtered"
            tmp_region_coords.mkdir(parents=True, exist_ok=True)
            tmp_selected.mkdir(parents=True, exist_ok=True)
            (tmp_region_coords / coords_path.name).symlink_to(coords_path)
            (tmp_selected / selected_path.name).symlink_to(selected_path)

            step3 = [
                sys.executable,
                str(project_root / "ext_cell_feat_par.py"),
                "--wsi_dir", str(tmp_wsi_dir),
                "--region_coords_dir", str(tmp_region_coords),
                "--selected_indices_dir", str(tmp_selected),
                "--checkpoint", str(cellvit_path),
                "--output_dir", str(base_features / "cells"),
                "--batch_size", "16",
                "--magnification", "40",
                "--all_gpus",
                "--workers_per_gpu", "1",
                "--feature_mode", "mask_mean",
                "--enforce_amp",
                "--save_full_outputs_regions_per_wsi", "10",
                "--full_output_region_selection", "random",
            ]
            rc = _run_cmd_and_stream(step3, log_queue, cwd=project_root)
            if rc != 0:
                log_queue.put(f"[EXTRACTION] ✗ ext_cell_feat_par.py failed with code {rc}.\n")
                return

            log_queue.put("[EXTRACTION] ✓ Feature extraction completed successfully.\n")
    except Exception as exc:
        log_queue.put(f"[EXTRACTION] ✗ Exception: {exc}\n")
    finally:
        log_queue.put(None)


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def _answer_question(
    question: str,
    model,
    features: Tuple,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_beams: int,
    repetition_penalty: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    wsi, region, patch, cell = features
    with torch.no_grad():
        response = model.generate_from_features(
            wsi=wsi,
            region=region,
            patch=patch,
            cell=cell,
            prompt=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return response


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
#log_box textarea { font-family: monospace; font-size: 12px; }
"""

# Convert image to base64
with open("assets/Logo.PNG", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

title_html = f"""
<div style="display:flex; align-items:center; gap:15px;">
    <img src="data:image/png;base64,{encoded}" width="80">
    <h1 style="margin:0;">
        MLLM-HWSI Whole-Slide Image VQA Chatbot
    </h1>
</div>
"""

def build_app() -> gr.Blocks:

    # Per-session mutable state
    state_keys = [
        "model", "slide_id", "features", "base_dir", "device",
        "extracting", "extract_thread", "log_queue",
        "extract_confirmed", "confirm_target",
        "max_new_tokens", "temperature", "top_p", "do_sample",
        "num_beams", "repetition_penalty", "length_penalty",
        "no_repeat_ngram_size",
    ]

    device_choices, default_device = _available_device_choices()

    with gr.Blocks(title="MLLM-HWSI Whole-Slide Image VQA Chatbot") as demo:
        # ---- session state ----
        sess = gr.State({k: None for k in state_keys})

        #gr.Markdown("# MLLM-HWSI Whole-Slide Image VQA Chatbot")
        gr.HTML(title_html)
        gr.Markdown(
            "Select a WSI file and the directory where features are stored "
            "(or should be extracted to). Then ask any histopathology question."
        )

        # ---- Setup panel ----
        with gr.Accordion("⚙️ Setup", open=True) as setup_accordion:
            gr.Markdown("Use Browse to open a native OS dialog, or type paths manually in each text box.")

            with gr.Row():
                wsi_path_box = gr.Textbox(
                    label="WSI Path (.svs / .tif / .ndpi ...)",
                    placeholder="/path/to/slide.svs",
                    scale=8,
                )
                browse_wsi_btn = gr.Button("Browse File", scale=2)

            with gr.Row():
                base_feat_dir_box = gr.Textbox(
                    label="Base Features Directory",
                    placeholder="/path/to/features_base",
                    scale=8,
                )
                browse_base_feat_btn = gr.Button("Browse Folder", scale=2)

            gr.Markdown("Base features directory must contain: wsi/, region_4k/, patches_filtered/, cells/")

            with gr.Group(visible=False) as extract_confirm_group:
                extract_prompt_md = gr.Markdown("")
                with gr.Row():
                    extract_yes_btn = gr.Button("Yes, extract features", variant="primary")
                    extract_no_btn = gr.Button("No", variant="secondary")

            with gr.Column(visible=False) as extraction_inputs_col:
                with gr.Row():
                    reports_path_box = gr.Textbox(
                        label="Reports JSON/JSONL Path",
                        value=DEFAULT_JSON_PATH,
                        scale=8,
                    )
                    browse_reports_btn = gr.Button("Browse File", scale=2)

                with gr.Row():
                    hipt_4096_ckpt_box = gr.Textbox(
                        label="HIPT 4096 Checkpoint",
                        value=DEFAULT_HIPT_4096_CKPT,
                        scale=8,
                    )
                    browse_hipt_4096_btn = gr.Button("Browse File", scale=2)

                with gr.Row():
                    hipt_256_ckpt_box = gr.Textbox(
                        label="HIPT 256 Checkpoint",
                        value=DEFAULT_HIPT_256_CKPT,
                        scale=8,
                    )
                    browse_hipt_256_btn = gr.Button("Browse File", scale=2)

                with gr.Row():
                    conch_ckpt_box = gr.Textbox(
                        label="CONCH Checkpoint",
                        value=DEFAULT_CONCH_CKPT,
                        scale=8,
                    )
                    browse_conch_btn = gr.Button("Browse File", scale=2)

                with gr.Row():
                    cellvit_ckpt_box = gr.Textbox(
                        label="CellViT Checkpoint",
                        value=DEFAULT_CELLVIT_CKPT,
                        scale=8,
                    )
                    browse_cellvit_btn = gr.Button("Browse File", scale=2)

            with gr.Row():
                encoder_ckpt_box = gr.Textbox(
                    label="VL Projector Checkpoint Override (Optional)",
                    placeholder="Optional: /path/to/VL_projector_checkpoint.pt",
                    scale=8,
                )
                browse_encoder_btn = gr.Button("Browse File", scale=2)

            with gr.Row():
                model_name_box = gr.Textbox(
                    label="MLLM-HWSI model folder/ Huggingface repo",
                    placeholder="Qwen/Qwen2.5-7B-Instruct or /path/to/model_dir",
                    scale=8,
                )
                browse_model_btn = gr.Button("Browse Folder", scale=2)
            
            gr.Markdown("Enter Qwen/Qwen2.5-7B-Instruct if trained model is absent.")

            with gr.Row():
                device_box = gr.Dropdown(
                    label="Device",
                    choices=device_choices,
                    value=default_device,
                )

            with gr.Row():
                load_btn = gr.Button("Load Model & Slide", variant="primary")
                reload_btn = gr.Button("Reload", variant="secondary")
            with gr.Column(visible=False) as readiness_col:
                readiness_box = gr.Markdown(
                    value=(
                        "<span style='color:#b91c1c; font-weight:600'>Extraction readiness: Not ready</span>"
                    )
                )
            status_box = gr.Textbox(label="Status", interactive=False, lines=2)

        # ---- Extraction log ----
        with gr.Accordion("📋 Extraction Log", open=False) as log_accordion:
            log_box = gr.Textbox(
                label="Live extraction output",
                interactive=False,
                lines=18,
                max_lines=18,
                elem_id="log_box",
            )
            refresh_btn = gr.Button("↻ Refresh Log")

        # ---- Chat ----
        chatbot = gr.Chatbot(label="VQA Chat", height=500)
        with gr.Row():
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Ask a histopathology question about this slide…",
                scale=8,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear chat")

        # ------------------------------------------------------------------ #
        #  Callbacks
        # ------------------------------------------------------------------ #

        def on_load(
            wsi_path, base_dir, enc_ckpt,
            mdl_name, dev,
            reports_path,
            hipt_4096_ckpt,
            hipt_256_ckpt,
            conch_ckpt,
            cellvit_ckpt,
            state,
            force_reload: bool = False,
        ):
            """Validate inputs, check features, prompt/launch extraction, or load model."""

            wsi_path = _normalize_selector_path(wsi_path)
            base_dir = _normalize_selector_path(base_dir)
            enc_ckpt = _normalize_selector_path(enc_ckpt)
            mdl_name = _normalize_selector_path(mdl_name)
            reports_path = _normalize_selector_path(reports_path)
            hipt_4096_ckpt = _normalize_selector_path(hipt_4096_ckpt)
            hipt_256_ckpt = _normalize_selector_path(hipt_256_ckpt)
            conch_ckpt = _normalize_selector_path(conch_ckpt)
            cellvit_ckpt = _normalize_selector_path(cellvit_ckpt)

            def _ret(
                state_out,
                status_text,
                log_text="",
                prompt_text="",
                show_prompt=False,
                show_extract_inputs=False,
            ):
                return (
                    state_out,
                    status_text,
                    log_text,
                    gr.update(value=prompt_text),
                    gr.update(visible=show_prompt),
                    gr.update(visible=show_extract_inputs),
                    gr.update(visible=show_extract_inputs),
                )

            if not wsi_path:
                return _ret(state, "⚠️ Please select a WSI file.")

            if Path(wsi_path).suffix.lower() not in WSI_EXTENSIONS:
                return _ret(state, f"⚠️ Selected WSI path is not a supported slide file: '{wsi_path}'")

            if not base_dir or not Path(base_dir).is_dir():
                return _ret(state, (
                    f"⚠️ Base features directory does not exist: '{base_dir}'. "
                    "Please create it or choose an existing one."
                ))

            slide_id = _slide_id_from_path(wsi_path)
            ok, missing = _features_exist(slide_id, base_dir)
            confirm_target = f"{slide_id}|{Path(base_dir).resolve()}"
            was_confirmed = bool(state.get("extract_confirmed")) and state.get("confirm_target") == confirm_target

            # Guard: if extraction is already running, don't reset state or double-start
            _active_thread = state.get("extract_thread")
            if _active_thread is not None and _active_thread.is_alive():
                return _ret(
                    state,
                    "⏳ Feature extraction is still running. "
                    "Check the Extraction Log panel and try again when it finishes.",
                )

            new_state = dict(state)
            new_state.update({
                "model": None,
                "features": None,
                "slide_id": slide_id,
                "base_dir": base_dir,
                "device": dev,
                "extracting": False,
                "extract_thread": None,
                "log_queue": None,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": False,
                "num_beams": 3,
                "repetition_penalty": 1.15,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "confirm_target": confirm_target,
                "extract_confirmed": was_confirmed,
            })

            if not ok:
                missing_feature_lines = "\n".join(f"  - {m}" for m in missing)

                if not was_confirmed:
                    prompt = (
                        f"⚠️ Selected slide features for '{slide_id}' do not exist in the selected folder.\n\n"
                        f"Missing files:\n{missing_feature_lines}\n\n"
                        "Do you want to extract features?"
                    )
                    new_state["extract_confirmed"] = False
                    return _ret(
                        new_state,
                        "⚠️ Slide features are missing. Please choose Yes/No below.",
                        prompt_text=prompt,
                        show_prompt=True,
                        show_extract_inputs=False,
                    )

                missing_ckpt_lines = []

                reports_resolved = Path(reports_path).expanduser() if reports_path else None
                if reports_resolved is None or not reports_resolved.exists():
                    missing_ckpt_lines.append(
                        f"reports_path: {reports_resolved if reports_resolved is not None else '(empty)'}"
                    )

                for label, raw_path in [
                    ("HIPT 4096", hipt_4096_ckpt),
                    ("HIPT 256", hipt_256_ckpt),
                    ("CONCH", conch_ckpt),
                    ("CellViT", cellvit_ckpt),
                ]:
                    resolved = _resolve_ckpt_path(raw_path) if raw_path else None
                    if resolved is None or not resolved.is_file():
                        missing_ckpt_lines.append(
                            f"{label}: {resolved if resolved is not None else '(empty)'}"
                        )

                if missing_ckpt_lines:
                    missing_ckpt_text = "\n".join(f"  - {m}" for m in missing_ckpt_lines)
                    return _ret(new_state, (
                        f"⚠️ Features are missing for slide '{slide_id}':\n{missing_feature_lines}\n\n"
                        "Please provide valid checkpoint/report paths:\n"
                        f"{missing_ckpt_text}"
                    ), show_extract_inputs=True)

                # Kick off extraction in background
                log_q = queue.Queue()
                t = threading.Thread(
                    target=_run_extraction_pipeline,
                    args=(
                        wsi_path,
                        base_dir,
                        reports_path,
                        hipt_4096_ckpt,
                        hipt_256_ckpt,
                        conch_ckpt,
                        cellvit_ckpt,
                        log_q,
                    ),
                    daemon=True,
                )
                t.start()
                new_state["extracting"] = True
                new_state["extract_thread"] = t
                new_state["log_queue"] = log_q
                status_msg = (
                    f"⏳ Feature files not found for slide '{slide_id}'.\n"
                    f"Missing files:\n{missing_feature_lines}\n"
                    "Extraction is running in the background — check the "
                    "Extraction Log panel for progress. "
                    "Click 'Load Model & Slide' again once extraction finishes."
                )
                return _ret(new_state, status_msg, show_extract_inputs=True)

            # Features are present → load model
            new_state["extract_confirmed"] = False
            new_state["confirm_target"] = None
            if force_reload:
                status_lines = [f"🔄 Reload requested for '{slide_id}'. Loading model…"]
            else:
                status_lines = [f"✅ Features found for '{slide_id}'. Loading model…"]
            try:
                effective_model_name = mdl_name or "Qwen/Qwen2.5-7B-Instruct"
                if mdl_name and Path(mdl_name).is_file():
                    # If a file is selected, use its parent directory as model root.
                    effective_model_name = str(Path(mdl_name).parent)
                model = _load_model(
                    model_name=effective_model_name,
                    device=dev,
                    encoder_ckpt=enc_ckpt or None,
                    llm_dim=3584,
                    proj_hidden_dim=1024,
                    dropout=0.1,
                    num_slide_tokens=64,
                    projector_type="mlp",
                )
                features = _load_features(slide_id, base_dir)
            except Exception as exc:
                return _ret(new_state, f"❌ Failed to load model/features:\n{exc}")

            new_state["model"]    = model
            new_state["features"] = features
            status_lines.append(
                f"✅ Model loaded. Slide '{slide_id}' is ready. Ask your question below."
            )
            return _ret(new_state, "\n".join(status_lines), show_extract_inputs=False)

        def on_refresh_log(state):
            """Drain the extraction log queue and append to the log box."""
            log_q = state.get("log_queue")
            if log_q is None:
                return state, ""
            lines = []
            try:
                while True:
                    item = log_q.get_nowait()
                    if item is None:
                        # Extraction finished
                        new_state = dict(state)
                        new_state["extracting"] = False
                        new_state["log_queue"] = None
                        lines.append("\n[LOG] Extraction process has finished.")
                        return new_state, "".join(lines)
                    lines.append(item)
            except queue.Empty:
                pass
            return state, "".join(lines)

        def on_send(question, chat_history, state):
            """Answer a question using the loaded model."""
            if not question or not question.strip():
                return chat_history, state, ""

            model    = state.get("model")
            features = state.get("features")

            if model is None or features is None:
                chat_history = list(chat_history) + [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content":
                        "⚠️ Model not loaded yet. Please click 'Load Model & Slide' first."},
                ]
                return chat_history, state, ""

            if state.get("extracting"):
                chat_history = list(chat_history) + [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content":
                        "⏳ Feature extraction is still running. "
                        "Please wait until it finishes, then click "
                        "'Load Model & Slide' again before asking questions."},
                ]
                return chat_history, state, ""

            try:
                answer = _answer_question(
                    question=question,
                    model=model,
                    features=features,
                    device=state["device"],
                    max_new_tokens=state["max_new_tokens"],
                    temperature=state["temperature"],
                    top_p=state["top_p"],
                    do_sample=state["do_sample"],
                    num_beams=state["num_beams"],
                    repetition_penalty=state["repetition_penalty"],
                    length_penalty=state["length_penalty"],
                    no_repeat_ngram_size=state["no_repeat_ngram_size"],
                )
            except Exception as exc:
                answer = f"❌ Inference error: {exc}"

            chat_history = list(chat_history) + [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ]
            return chat_history, state, ""

        def on_clear(state):
            return [], state

        def on_browse_wsi(current_path):
            return _pick_file_dialog(
                current_path=current_path,
                title="Select WSI File",
                filetypes=[
                    ("Whole-slide images", "*.svs *.tif *.tiff *.ndpi *.mrxs *.scn *.vms *.vmu *.bif"),
                    ("All files", "*.*"),
                ],
            )

        def on_browse_base_features(current_path):
            return _pick_folder_dialog(current_path=current_path, title="Select Base Features Directory")

        def on_browse_encoder(current_path):
            return _pick_file_dialog(
                current_path=current_path,
                title="Select VL Projector Checkpoint",
                filetypes=[
                    ("PyTorch checkpoints", "*.pt *.pth *.bin"),
                    ("All files", "*.*"),
                ],
            )

        def on_browse_reports(current_path):
            return _pick_file_dialog(
                current_path=current_path,
                title="Select reports JSON/JSONL",
                filetypes=[
                    ("JSON/JSONL", "*.json *.jsonl"),
                    ("All files", "*.*"),
                ],
            )

        def on_browse_checkpoint(current_path):
            return _pick_file_dialog(
                current_path=current_path,
                title="Select checkpoint file",
                filetypes=[
                    ("Model checkpoints", "*.pt *.pth *.bin"),
                    ("All files", "*.*"),
                ],
            )

        def on_browse_model(current_path):
            return _pick_folder_dialog(current_path=current_path, title="Select MLLM-HWSI Model Directory")

        def on_check_features(wsi_path, base_dir, state):
            """Immediately check feature existence when wsi or base dir change."""
            wsi_path = _normalize_selector_path(wsi_path)
            base_dir = _normalize_selector_path(base_dir)

            # Not enough info — leave UI untouched
            if (not wsi_path or not base_dir
                    or Path(wsi_path).suffix.lower() not in WSI_EXTENSIONS
                    or not Path(base_dir).is_dir()):
                return state, gr.update(), gr.update(value=""), gr.update(visible=False)

            slide_id = _slide_id_from_path(wsi_path)
            ok, missing = _features_exist(slide_id, base_dir)
            confirm_target = f"{slide_id}|{Path(base_dir).resolve()}"

            new_state = dict(state) if state else {}
            new_state["slide_id"] = slide_id
            new_state["base_dir"] = base_dir

            if ok:
                # Features present — hide any stale confirm dialog
                new_state["extract_confirmed"] = False
                new_state["confirm_target"] = None
                return (
                    new_state,
                    f"✅ Features found for slide '{slide_id}'.",
                    gr.update(value=""),
                    gr.update(visible=False),
                )

            # Already confirmed for this exact slide+dir — don't re-prompt
            was_confirmed = (
                bool(state.get("extract_confirmed"))
                and state.get("confirm_target") == confirm_target
            ) if state else False

            if was_confirmed:
                return state, gr.update(), gr.update(), gr.update()

            missing_lines = "\n".join(f"  - {m}" for m in missing)
            prompt = (
                f"⚠️ Features for slide **'{slide_id}'** do not exist in the selected directory.\n\n"
                f"Missing files:\n{missing_lines}\n\n"
                "Do you want to extract features into that directory?"
            )
            new_state["confirm_target"] = confirm_target
            new_state["extract_confirmed"] = False
            return (
                new_state,
                "⚠️ Slide features are missing. Please choose Yes/No below.",
                gr.update(value=prompt),
                gr.update(visible=True),
            )

        def on_readiness_change(
            wsi_path,
            base_dir,
            reports_path,
            hipt_4096_ckpt,
            hipt_256_ckpt,
            conch_ckpt,
            cellvit_ckpt,
        ):
            wsi_path = _normalize_selector_path(wsi_path)
            base_dir = _normalize_selector_path(base_dir)
            reports_path = _normalize_selector_path(reports_path)
            hipt_4096_ckpt = _normalize_selector_path(hipt_4096_ckpt)
            hipt_256_ckpt = _normalize_selector_path(hipt_256_ckpt)
            conch_ckpt = _normalize_selector_path(conch_ckpt)
            cellvit_ckpt = _normalize_selector_path(cellvit_ckpt)

            checks = []

            wsi_ok = bool(wsi_path) and Path(wsi_path).is_file() and Path(wsi_path).suffix.lower() in WSI_EXTENSIONS
            checks.append(("WSI", wsi_ok))

            base_ok = bool(base_dir) and Path(base_dir).is_dir()
            checks.append(("Features dir", base_ok))

            reports_ok = bool(reports_path) and Path(reports_path).expanduser().exists()
            checks.append(("Reports", reports_ok))

            h4096_ok = bool(hipt_4096_ckpt) and _resolve_ckpt_path(hipt_4096_ckpt).is_file()
            h256_ok = bool(hipt_256_ckpt) and _resolve_ckpt_path(hipt_256_ckpt).is_file()
            conch_ok = bool(conch_ckpt) and _resolve_ckpt_path(conch_ckpt).is_file()
            cellvit_ok = bool(cellvit_ckpt) and _resolve_ckpt_path(cellvit_ckpt).is_file()
            checks.extend([
                ("HIPT-4096", h4096_ok),
                ("HIPT-256", h256_ok),
                ("CONCH", conch_ok),
                ("CellViT", cellvit_ok),
            ])

            all_ok = all(ok for _, ok in checks)
            parts = [f"{name}: {'OK' if ok else 'Missing'}" for name, ok in checks]

            if all_ok:
                return (
                    "<span style='color:#15803d; font-weight:600'>"
                    "Extraction readiness: Ready"
                    "</span>"
                    f"<br><span>{' | '.join(parts)}</span>"
                )

            return (
                "<span style='color:#b91c1c; font-weight:600'>"
                "Extraction readiness: Not ready"
                "</span>"
                f"<br><span>{' | '.join(parts)}</span>"
            )

        def on_reload_click(
            wsi_path,
            base_dir,
            enc_ckpt,
            mdl_name,
            dev,
            reports_path,
            hipt_4096_ckpt,
            hipt_256_ckpt,
            conch_ckpt,
            cellvit_ckpt,
            state,
        ):
            return on_load(
                wsi_path=wsi_path,
                base_dir=base_dir,
                enc_ckpt=enc_ckpt,
                mdl_name=mdl_name,
                dev=dev,
                reports_path=reports_path,
                hipt_4096_ckpt=hipt_4096_ckpt,
                hipt_256_ckpt=hipt_256_ckpt,
                conch_ckpt=conch_ckpt,
                cellvit_ckpt=cellvit_ckpt,
                state=state,
                force_reload=True,
            )

        def on_extract_confirm_yes(
            wsi_path, base_dir,
            reports_path, hipt_4096_ckpt, hipt_256_ckpt, conch_ckpt, cellvit_ckpt,
            state,
        ):
            wsi_path = _normalize_selector_path(wsi_path)
            base_dir = _normalize_selector_path(base_dir)
            reports_path = _normalize_selector_path(reports_path)
            hipt_4096_ckpt = _normalize_selector_path(hipt_4096_ckpt)
            hipt_256_ckpt = _normalize_selector_path(hipt_256_ckpt)
            conch_ckpt = _normalize_selector_path(conch_ckpt)
            cellvit_ckpt = _normalize_selector_path(cellvit_ckpt)

            new_state = dict(state)
            new_state["extract_confirmed"] = True

            slide_id = state.get("slide_id") or (
                _slide_id_from_path(wsi_path) if wsi_path else None
            )

            # Validate all checkpoints before committing
            missing_ckpt_lines = []
            reports_resolved = Path(reports_path).expanduser() if reports_path else None
            if reports_resolved is None or not reports_resolved.exists():
                missing_ckpt_lines.append(
                    f"reports_path: {reports_resolved if reports_resolved is not None else '(empty)'}"
                )
            for label, raw_path in [
                ("HIPT 4096", hipt_4096_ckpt),
                ("HIPT 256", hipt_256_ckpt),
                ("CONCH", conch_ckpt),
                ("CellViT", cellvit_ckpt),
            ]:
                resolved = _resolve_ckpt_path(raw_path) if raw_path else None
                if resolved is None or not resolved.is_file():
                    missing_ckpt_lines.append(
                        f"{label}: {resolved if resolved is not None else '(empty)'}"
                    )

            if missing_ckpt_lines:
                missing_text = "\n".join(f"  - {m}" for m in missing_ckpt_lines)
                return (
                    new_state,
                    f"⚠️ Cannot start extraction — provide valid paths:\n{missing_text}",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            # All checkpoints present — check again in case features appeared
            ok, missing = (
                _features_exist(slide_id, base_dir)
                if (slide_id and base_dir)
                else (False, [])
            )
            if ok:
                return (
                    new_state,
                    f"✅ Features already present for '{slide_id}'. Click 'Load Model & Slide'.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            # Launch extraction in background thread
            log_q = queue.Queue()
            t = threading.Thread(
                target=_run_extraction_pipeline,
                args=(
                    wsi_path, base_dir,
                    reports_path, hipt_4096_ckpt, hipt_256_ckpt, conch_ckpt, cellvit_ckpt,
                    log_q,
                ),
                daemon=True,
            )
            t.start()
            new_state["extracting"] = True
            new_state["extract_thread"] = t
            new_state["log_queue"] = log_q

            missing_lines = "\n".join(f"  - {m}" for m in missing)
            status = (
                f"⏳ Extraction started for slide '{slide_id}'.\n"
                f"Missing files:\n{missing_lines}\n"
                "Check the Extraction Log panel for progress (↻ Refresh Log). "
                "Click 'Load Model & Slide' once extraction finishes."
            )
            return (
                new_state,
                status,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        def on_extract_confirm_no(state):
            new_state = dict(state)
            new_state["extract_confirmed"] = False
            new_state["confirm_target"] = None
            return (
                new_state,
                "Extraction cancelled. Select another slide/folder or click Load again when ready.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # ---- Wire up ----
        browse_wsi_btn.click(fn=on_browse_wsi, inputs=[wsi_path_box], outputs=[wsi_path_box])
        browse_base_feat_btn.click(
            fn=on_browse_base_features,
            inputs=[base_feat_dir_box],
            outputs=[base_feat_dir_box],
        )
        browse_encoder_btn.click(fn=on_browse_encoder, inputs=[encoder_ckpt_box], outputs=[encoder_ckpt_box])
        browse_model_btn.click(fn=on_browse_model, inputs=[model_name_box], outputs=[model_name_box])
        browse_reports_btn.click(fn=on_browse_reports, inputs=[reports_path_box], outputs=[reports_path_box])
        browse_hipt_4096_btn.click(fn=on_browse_checkpoint, inputs=[hipt_4096_ckpt_box], outputs=[hipt_4096_ckpt_box])
        browse_hipt_256_btn.click(fn=on_browse_checkpoint, inputs=[hipt_256_ckpt_box], outputs=[hipt_256_ckpt_box])
        browse_conch_btn.click(fn=on_browse_checkpoint, inputs=[conch_ckpt_box], outputs=[conch_ckpt_box])
        browse_cellvit_btn.click(fn=on_browse_checkpoint, inputs=[cellvit_ckpt_box], outputs=[cellvit_ckpt_box])
        extract_yes_btn.click(
            fn=on_extract_confirm_yes,
            inputs=[
                wsi_path_box, base_feat_dir_box,
                reports_path_box, hipt_4096_ckpt_box, hipt_256_ckpt_box,
                conch_ckpt_box, cellvit_ckpt_box,
                sess,
            ],
            outputs=[sess, status_box, extract_confirm_group, extraction_inputs_col, readiness_col],
        )
        extract_no_btn.click(
            fn=on_extract_confirm_no,
            inputs=[sess],
            outputs=[sess, status_box, extract_confirm_group, extraction_inputs_col, readiness_col],
        )

        readiness_inputs = [
            wsi_path_box,
            base_feat_dir_box,
            reports_path_box,
            hipt_4096_ckpt_box,
            hipt_256_ckpt_box,
            conch_ckpt_box,
            cellvit_ckpt_box,
        ]

        for comp in readiness_inputs:
            comp.change(fn=on_readiness_change, inputs=readiness_inputs, outputs=[readiness_box])

        _check_feat_inputs = [wsi_path_box, base_feat_dir_box, sess]
        _check_feat_outputs = [sess, status_box, extract_prompt_md, extract_confirm_group]
        wsi_path_box.change(fn=on_check_features, inputs=_check_feat_inputs, outputs=_check_feat_outputs)
        base_feat_dir_box.change(fn=on_check_features, inputs=_check_feat_inputs, outputs=_check_feat_outputs)

        load_btn.click(
            fn=on_load,
            inputs=[
                wsi_path_box, base_feat_dir_box, encoder_ckpt_box,
                model_name_box, device_box,
                reports_path_box,
                hipt_4096_ckpt_box,
                hipt_256_ckpt_box,
                conch_ckpt_box,
                cellvit_ckpt_box,
                sess,
            ],
            outputs=[
                sess,
                status_box,
                log_box,
                extract_prompt_md,
                extract_confirm_group,
                extraction_inputs_col,
                readiness_col,
            ],
        )

        reload_btn.click(
            fn=on_reload_click,
            inputs=[
                wsi_path_box, base_feat_dir_box, encoder_ckpt_box,
                model_name_box, device_box,
                reports_path_box,
                hipt_4096_ckpt_box,
                hipt_256_ckpt_box,
                conch_ckpt_box,
                cellvit_ckpt_box,
                sess,
            ],
            outputs=[
                sess,
                status_box,
                log_box,
                extract_prompt_md,
                extract_confirm_group,
                extraction_inputs_col,
                readiness_col,
            ],
        )

        refresh_btn.click(
            fn=on_refresh_log,
            inputs=[sess],
            outputs=[sess, log_box],
        )

        send_btn.click(
            fn=on_send,
            inputs=[question_box, chatbot, sess],
            outputs=[chatbot, sess, question_box],
        )
        question_box.submit(
            fn=on_send,
            inputs=[question_box, chatbot, sess],
            outputs=[chatbot, sess, question_box],
        )
        clear_btn.click(fn=on_clear, inputs=[sess], outputs=[chatbot, sess])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WSI VQA Chatbot")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
