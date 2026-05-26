#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA Chatbot — Gradio front-end backed by the MLLMHWSIQWEN model.

Users select:
  • A single WSI file  (used to derive the slide_id)
  • A base features directory  (must contain wsi/, region/, patch/, cells/ sub-dirs)
  • A checkpoint for the MLLM-HWSI V-L encoder

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
from pathlib import Path
from typing import Optional, Tuple

import torch
import gradio as gr
import base64

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


def _features_exist(slide_id: str, base_dir: str) -> Tuple[bool, str]:
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
    """Load and return a ready-to-use MLLMHWSIQWEN model."""
    from mllm_hwsi import VLProjectorConfig, MLLMHWSIQWEN

    cfg = VLProjectorConfig(
        llm_dim=llm_dim,
        hidden_dim=proj_hidden_dim,
        dropout=dropout,
        use_layernorm=True,
        num_query_tokens=num_slide_tokens,
        projector_type=projector_type,
    )
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = MLLMHWSIQWEN(
        model_name=model_name,
        projector_cfg=cfg,
        device=device,
        torch_dtype=dtype,
    )

    if encoder_ckpt:
        ckpt = torch.load(encoder_ckpt, map_location="cpu")
        ckpt = ckpt.get("state_dict", ckpt)
        model.pathology_encoder.load_state_dict(ckpt, strict=False)

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

def _run_extraction_pipeline(wsi_path: str, base_dir: str, log_queue: queue.Queue):
    """
    Runs run_ext_pipeline.sh for a single WSI.
    Streams lines into log_queue. Puts None when done.

    NOTE: The extraction script is designed to process a whole directory.
    We pass the parent of the WSI as --wsi_dir so only that one slide is
    processed (the script skips slides that already have outputs).
    """
    wsi_file = Path(wsi_path)
    script = Path(__file__).parent / "run_ext_pipeline.sh"

    if not script.exists():
        log_queue.put(f"[ERROR] Extraction script not found: {script}\n")
        log_queue.put(None)
        return

    env = os.environ.copy()
    cmd = [
        "bash", str(script),
        "--wsi_dir",   str(wsi_file.parent),
        "--out_dir",   str(base_dir),
        "--single_slide", wsi_file.stem,   # script must support this flag, else omit
    ]

    log_queue.put(f"[EXTRACTION] Running: {' '.join(shlex.quote(c) for c in cmd)}\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            log_queue.put(line)
        proc.wait()
        if proc.returncode == 0:
            log_queue.put("[EXTRACTION] ✓ Feature extraction completed successfully.\n")
        else:
            log_queue.put(f"[EXTRACTION] ✗ Extraction exited with code {proc.returncode}.\n")
    except Exception as exc:
        log_queue.put(f"[EXTRACTION] ✗ Exception: {exc}\n")
    finally:
        log_queue.put(None)   # sentinel


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

            with gr.Row():
                encoder_ckpt_box = gr.Textbox(
                    label="MLLM-HWSI V-L Encoder Checkpoint",
                    placeholder="/path/to/encoder_checkpoint.pt",
                    scale=8,
                )
                browse_encoder_btn = gr.Button("Browse File", scale=2)

            with gr.Row():
                model_name_box = gr.Textbox(
                    label="MLLM-HWSI model folder",
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
            state,
            force_reload: bool = False,
        ):
            """Validate inputs, check features, kick off extraction or load model."""

            wsi_path = _normalize_selector_path(wsi_path)
            base_dir = _normalize_selector_path(base_dir)
            enc_ckpt = _normalize_selector_path(enc_ckpt)
            mdl_name = _normalize_selector_path(mdl_name)

            if not wsi_path:
                return state, "⚠️ Please select a WSI file.", ""

            if Path(wsi_path).suffix.lower() not in WSI_EXTENSIONS:
                return state, f"⚠️ Selected WSI path is not a supported slide file: '{wsi_path}'", ""

            if not base_dir or not Path(base_dir).is_dir():
                return state, (
                    f"⚠️ Base features directory does not exist: '{base_dir}'. "
                    "Please create it or choose an existing one."
                ), ""

            slide_id = _slide_id_from_path(wsi_path)
            ok, missing = _features_exist(slide_id, base_dir)

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
            })

            if not ok:
                # Kick off extraction in background
                log_q = queue.Queue()
                t = threading.Thread(
                    target=_run_extraction_pipeline,
                    args=(wsi_path, base_dir, log_q),
                    daemon=True,
                )
                t.start()
                new_state["extracting"] = True
                new_state["extract_thread"] = t
                new_state["log_queue"] = log_q
                status_msg = (
                    f"⏳ Feature files not found for slide '{slide_id}'.\n"
                    "Extraction is running in the background — check the "
                    "Extraction Log panel for progress. "
                    "Click 'Load Model & Slide' again once extraction finishes."
                )
                return new_state, status_msg, ""

            # Features are present → load model
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
                return new_state, f"❌ Failed to load model/features:\n{exc}", ""

            new_state["model"]    = model
            new_state["features"] = features
            status_lines.append(
                f"✅ Model loaded. Slide '{slide_id}' is ready. Ask your question below."
            )
            return new_state, "\n".join(status_lines), ""

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
                title="Select Pathology Encoders Checkpoint",
                filetypes=[
                    ("PyTorch checkpoints", "*.pt *.pth *.bin"),
                    ("All files", "*.*"),
                ],
            )

        def on_browse_model(current_path):
            return _pick_folder_dialog(current_path=current_path, title="Select MLLM-HWSI Model Directory")

        def on_reload_click(wsi_path, base_dir, enc_ckpt, mdl_name, dev, state):
            return on_load(
                wsi_path=wsi_path,
                base_dir=base_dir,
                enc_ckpt=enc_ckpt,
                mdl_name=mdl_name,
                dev=dev,
                state=state,
                force_reload=True,
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

        load_btn.click(
            fn=on_load,
            inputs=[
                wsi_path_box, base_feat_dir_box, encoder_ckpt_box,
                model_name_box, device_box,
                sess,
            ],
            outputs=[sess, status_box, log_box],
        )

        reload_btn.click(
            fn=on_reload_click,
            inputs=[
                wsi_path_box, base_feat_dir_box, encoder_ckpt_box,
                model_name_box, device_box,
                sess,
            ],
            outputs=[sess, status_box, log_box],
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
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
