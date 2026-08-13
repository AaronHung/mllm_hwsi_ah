#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


@dataclass
class VLProjectorConfig:
    llm_dim: int = 3584
    hidden_dim: int = 1024
    dropout: float = 0.1
    use_layernorm: bool = True
    num_query_tokens: int = 64
    projector_type: str = "mlp"   # "linear" or "mlp"


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Union[int, Tuple[int, ...]],
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Masked mean over one or multiple dimensions.

    Examples
    --------
    x:    (B, R, C, D)
    mask: (B, R, C)
    dim=2 or dim=(1, 2)

    If mask is None, falls back to ordinary mean.
    """
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = (dim,)
    dim = tuple(sorted(dim))

    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    num = x * mask
    den = mask

    # reduce in reverse order so axis numbers remain valid
    for d in sorted(dim, reverse=True):
        num = num.sum(dim=d, keepdim=keepdim)
        den = den.sum(dim=d, keepdim=keepdim)

    den = den.clamp_min(eps)
    return num / den


def _make_projector(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    dropout: float,
    use_layernorm: bool,
    projector_type: str = "mlp",
) -> nn.Module:
    layers = []

    if use_layernorm:
        layers.append(nn.LayerNorm(in_dim))

    if projector_type == "linear":
        layers.append(nn.Linear(in_dim, out_dim))
    elif projector_type == "mlp":
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ])
    else:
        raise ValueError(f"Unsupported projector_type: {projector_type}")

    return nn.Sequential(*layers)


class BaseVLProjector(nn.Module):
    def __init__(self, in_dim: int, cfg: VLProjectorConfig):
        super().__init__()
        self.in_dim = in_dim
        self.projector = _make_projector(
            in_dim=in_dim,
            out_dim=cfg.llm_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
            projector_type=cfg.projector_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")
        return self.projector(x)


class WSILevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 192):
        super().__init__()
        self.proj = BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"WSI input must be (B, 192), got {tuple(x.shape)}")
        return self.proj(x).unsqueeze(1)  # (B, 1, llm_dim)


class RegionLevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 192):
        super().__init__()
        self.proj = BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Region input must be (B, R, 192), got {tuple(x.shape)}")
        return self.proj(x)


class PatchLevelProjector(nn.Module):
    """
    selected patch features shape: [B, R, C, 512]
    where C can vary across samples/regions after padding in collate.
    """
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 512):
        super().__init__()
        self.proj = BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Patch input must be (B, R, C, 512), got {tuple(x.shape)}")
        return self.proj(x)


class CellLevelProjector(nn.Module):
    """
    encoded cell features shape: [B, R, C, 384]
    where C can vary across samples/regions after padding in collate.
    """
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 384):
        super().__init__()
        self.proj = BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Cell input must be (B, R, C, 384), got {tuple(x.shape)}")
        return self.proj(x)


class RegionPatchCellFusion(nn.Module):
    """
    Fuses:
      - global WSI token
      - region token
      - masked pooled patch tokens
      - masked pooled cell tokens

    Output:
      fused region tokens: (B, R, D)
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        wsi_token: torch.Tensor,       # (B, 1, D)
        region_tokens: torch.Tensor,   # (B, R, D)
        patch_tokens: torch.Tensor,    # (B, R, C, D)
        cell_tokens: torch.Tensor,     # (B, R, C, D)
        patch_mask: Optional[torch.Tensor] = None,   # (B, R, C)
        cell_mask: Optional[torch.Tensor] = None,    # (B, R, C)
    ) -> torch.Tensor:
        if region_tokens.ndim != 3:
            raise ValueError(f"region_tokens must be (B, R, D), got {tuple(region_tokens.shape)}")
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be (B, R, C, D), got {tuple(patch_tokens.shape)}")
        if cell_tokens.ndim != 4:
            raise ValueError(f"cell_tokens must be (B, R, C, D), got {tuple(cell_tokens.shape)}")

        # Masked pooling over C
        pooled_patch = masked_mean(patch_tokens, patch_mask, dim=2)  # (B, R, D)
        pooled_cell = masked_mean(cell_tokens, cell_mask, dim=2)     # (B, R, D)

        wsi_expand = wsi_token.expand(-1, region_tokens.shape[1], -1)  # (B, R, D)

        fused = torch.cat([wsi_expand, region_tokens, pooled_patch, pooled_cell], dim=-1)
        fused = self.fuse(fused)

        # residual with region tokens for stability
        return self.out_norm(fused + region_tokens)


class LearnedQueryResampler(nn.Module):
    def __init__(self, dim: int, num_query_tokens: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_query_tokens, dim) / math.sqrt(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, N, D), got {tuple(x.shape)}")

        bsz = x.shape[0]
        q = self.queries.expand(bsz, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        attn_out, _ = self.attn(
            query=self.norm1(q),
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        z = q + attn_out
        z = z + self.mlp(self.norm2(z))
        return z


class MultiLevelProjectorEncoder(nn.Module):
    def __init__(self, cfg: VLProjectorConfig):
        super().__init__()
        self.wsi_proj = WSILevelProjector(cfg, in_dim=192)
        self.region_proj = RegionLevelProjector(cfg, in_dim=192)
        self.patch_proj = PatchLevelProjector(cfg, in_dim=512)
        self.cell_proj = CellLevelProjector(cfg, in_dim=384)

        self.fusion = RegionPatchCellFusion(dim=cfg.llm_dim, dropout=cfg.dropout)
        self.compressor = LearnedQueryResampler(
            dim=cfg.llm_dim,
            num_query_tokens=cfg.num_query_tokens,
            num_heads=8,
            dropout=cfg.dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        cfg: Optional[VLProjectorConfig] = None,
        *,
        filename: str = "vl_projector.pt",
        revision: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
        token: Optional[str] = None,
        return_loading_info: bool = False,
    ):
        def _extract_state_dict(payload):
            if isinstance(payload, dict):
                if "vl_projector_state_dict" in payload:
                    return payload["vl_projector_state_dict"]
                if "pathology_encoder_state_dict" in payload:
                    return payload["pathology_encoder_state_dict"]
                if "state_dict" in payload:
                    return payload["state_dict"]
            return payload

        def _try_read_cfg_from_payload(payload):
            if isinstance(payload, dict) and isinstance(payload.get("projector_cfg"), dict):
                return VLProjectorConfig(**payload["projector_cfg"])
            return None

        def _load_cfg_from_json(path: Path) -> Optional[VLProjectorConfig]:
            if not path.is_file():
                return None
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return VLProjectorConfig(**data)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return None
            return None

        source_path = Path(source).expanduser()
        payload = None
        resolved_cfg = cfg

        if source_path.exists():
            if source_path.is_file():
                ckpt_path = source_path
            else:
                candidates = [
                    source_path / filename,
                    source_path / "vl_projector.pt",
                    source_path / "best_vl_projector.pt",
                    source_path / "pytorch_model.bin",
                ]
                ckpt_path = next((p for p in candidates if p.is_file()), None)
                if ckpt_path is None:
                    raise FileNotFoundError(
                        f"No projector checkpoint found in {source_path}. "
                        f"Tried: {[str(p.name) for p in candidates]}"
                    )

            payload = torch.load(ckpt_path, map_location=map_location)
            if resolved_cfg is None:
                resolved_cfg = _try_read_cfg_from_payload(payload)
            if resolved_cfg is None and source_path.is_dir():
                resolved_cfg = _load_cfg_from_json(source_path / "projector_config.json")
        else:
            repo_id = source
            cfg_path = None
            if resolved_cfg is None:
                try:
                    cfg_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="projector_config.json",
                        revision=revision,
                        token=token,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    cfg_path = None
                if cfg_path is not None:
                    resolved_cfg = _load_cfg_from_json(Path(cfg_path))

            candidate_files = [filename, "vl_projector.pt", "best_vl_projector.pt", "pytorch_model.bin"]
            last_err = None
            ckpt_path = None
            for candidate in candidate_files:
                try:
                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=candidate,
                        revision=revision,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    break
                except Exception as exc:
                    last_err = exc
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Could not download projector checkpoint from repo '{repo_id}'. "
                    f"Tried files: {candidate_files}. Last error: {last_err}"
                )
            payload = torch.load(ckpt_path, map_location=map_location)
            if resolved_cfg is None:
                resolved_cfg = _try_read_cfg_from_payload(payload)

        if resolved_cfg is None:
            raise ValueError(
                "Projector config not found. Pass cfg=VLProjectorConfig(...) explicitly, "
                "or include projector_cfg/projector_config.json in checkpoint artifacts."
            )

        model = cls(resolved_cfg)
        state_dict = _extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if device is not None and dtype is not None:
            model = model.to(device=device, dtype=dtype)
        elif device is not None:
            model = model.to(device=device)
        elif dtype is not None:
            model = model.to(dtype=dtype)

        if return_loading_info:
            return model, {
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
        return model

    def forward(
        self,
        wsi: torch.Tensor,      # (B, 192)
        region: torch.Tensor,   # (B, R, 192)
        patch: torch.Tensor,    # (B, R, C, 512)
        cell: torch.Tensor,     # (B, R, C, 384)
        region_mask: Optional[torch.Tensor] = None,  # (B, R)
        patch_mask: Optional[torch.Tensor] = None,   # (B, R, C)
        cell_mask: Optional[torch.Tensor] = None,    # (B, R, C)
    ) -> torch.Tensor:
        wsi_token = self.wsi_proj(wsi)           # (B, 1, D)
        region_tokens = self.region_proj(region) # (B, R, D)
        patch_tokens = self.patch_proj(patch)    # (B, R, C, D)
        cell_tokens = self.cell_proj(cell)       # (B, R, C, D)

        # Explicitly zero padded slots before fusion.
        # In your training pipeline, patch validity is aligned to valid cells,
        # so patch_mask and cell_mask can often be identical.
        if patch_mask is not None:
            patch_tokens = patch_tokens * patch_mask.unsqueeze(-1).to(dtype=patch_tokens.dtype)

        if cell_mask is not None:
            cell_tokens = cell_tokens * cell_mask.unsqueeze(-1).to(dtype=cell_tokens.dtype)

        fused_region_tokens = self.fusion(
            wsi_token=wsi_token,
            region_tokens=region_tokens,
            patch_tokens=patch_tokens,
            cell_tokens=cell_tokens,
            patch_mask=patch_mask,
            cell_mask=cell_mask,
        )

        slide_tokens = self.compressor(fused_region_tokens, mask=region_mask)
        return slide_tokens


class MLLMHWSI(nn.Module):
    def __init__(
        self,
        model_name: str,
        projector_cfg: VLProjectorConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.projector_cfg = projector_cfg
        if dtype is not None:
            torch_dtype = dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            # Some checkpoints only provide a slow tokenizer unless optional
            # converter deps are available (e.g., sentencepiece/tiktoken).
            err = str(exc).lower()
            needs_slow_fallback = (
                "backend tokenizer" in err
                or "sentencepiece" in err
                or "tiktoken" in err
                or "convert a slow tokenizer" in err
            )
            if not needs_slow_fallback:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
            )

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

        # Keep VL projector in fp32 for stability; load projector weights explicitly via checkpoint.
        self.vl_projector = MultiLevelProjectorEncoder(projector_cfg).to(dtype=torch.float32)
        self.embed_tokens = self.llm.get_input_embeddings()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pathology_encoder(self):
        return self.vl_projector

    @pathology_encoder.setter
    def pathology_encoder(self, value):
        self.vl_projector = value

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        projector_cfg: Optional[VLProjectorConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        strict_projector: bool = False,
        load_projector: bool = True,
        projector_filename: str = "vl_projector.pt",
        return_loading_info: bool = False,
    ):
        source_path = Path(source).expanduser()

        resolved_cfg = projector_cfg
        cfg_path: Optional[Path] = None

        if resolved_cfg is None:
            if source_path.is_dir():
                local_cfg = source_path / "projector_config.json"
                if local_cfg.is_file():
                    cfg_path = local_cfg
            elif not source_path.exists():
                try:
                    downloaded_cfg = hf_hub_download(
                        repo_id=source,
                        filename="projector_config.json",
                        revision=revision,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    cfg_path = Path(downloaded_cfg)
                except Exception:
                    cfg_path = None

            if cfg_path is not None:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                resolved_cfg = VLProjectorConfig(**cfg_data)

        if resolved_cfg is None:
            raise ValueError(
                "projector_cfg is required when projector_config.json is unavailable."
            )

        model = cls(
            model_name=source,
            projector_cfg=resolved_cfg,
            device=device,
            dtype=dtype,
        )

        load_info = None
        if load_projector:
            projector, load_info = MultiLevelProjectorEncoder.from_pretrained(
                source=source,
                cfg=resolved_cfg,
                filename=projector_filename,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
                strict=strict_projector,
                map_location="cpu",
                device=device,
                dtype=torch.float32,
                return_loading_info=True,
            )
            model.vl_projector = projector

        if return_loading_info:
            return model, {
                "vl_projector": load_info,
            }
        return model

    def format_chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert histopathology assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "<|system|>\nYou are an expert histopathology assistant.\n"
            "<|user|>\n" + prompt + "\n"
            "<|assistant|>\n"
        )

    def save_pretrained(self, save_directory: str) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LLM and tokenizer in standard HF layout.
        self.llm.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        # Save VL projector weights and config alongside HF artifacts.
        projector_payload = {
            "vl_projector_state_dict": self.vl_projector.state_dict(),
        }
        torch.save(projector_payload, save_dir / "vl_projector.pt")

        projector_cfg = {
            "llm_dim": int(self.projector_cfg.llm_dim),
            "hidden_dim": int(self.projector_cfg.hidden_dim),
            "dropout": float(self.projector_cfg.dropout),
            "use_layernorm": bool(self.projector_cfg.use_layernorm),
            "num_query_tokens": int(self.projector_cfg.num_query_tokens),
            "projector_type": str(self.projector_cfg.projector_type),
        }
        with (save_dir / "projector_config.json").open("w", encoding="utf-8") as f:
            json.dump(projector_cfg, f, indent=2)

    @torch.no_grad()
    def generate_from_features(
        self,
        wsi: torch.Tensor,       # (1, 192)
        region: torch.Tensor,    # (1, R, 192)
        patch: torch.Tensor,     # (1, R, C, 512)
        cell: torch.Tensor,      # (1, R, C, 384)
        prompt: str,
        region_mask: Optional[torch.Tensor] = None,  # (1, R)
        patch_mask: Optional[torch.Tensor] = None,   # (1, R, C)
        cell_mask: Optional[torch.Tensor] = None,    # (1, R, C)
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        llm_dtype = next(self.llm.parameters()).dtype
        encoder_dtype = next(self.vl_projector.parameters()).dtype

        wsi = wsi.to(self.device, dtype=encoder_dtype)
        region = region.to(self.device, dtype=encoder_dtype)
        patch = patch.to(self.device, dtype=encoder_dtype)
        cell = cell.to(self.device, dtype=encoder_dtype)

        if region_mask is not None:
            region_mask = region_mask.to(self.device)
        if patch_mask is not None:
            patch_mask = patch_mask.to(self.device)
        if cell_mask is not None:
            cell_mask = cell_mask.to(self.device)

        slide_tokens = self.vl_projector(
            wsi=wsi,
            region=region,
            patch=patch,
            cell=cell,
            region_mask=region_mask,
            patch_mask=patch_mask,
            cell_mask=cell_mask,
        )
        slide_tokens = slide_tokens.to(dtype=llm_dtype)

        formatted_prompt = self.format_chat(prompt)
        text_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        text_attention_mask = text_inputs["attention_mask"].to(self.device)

        text_embeds = self.embed_tokens(input_ids)
        if text_embeds.dtype != llm_dtype:
            text_embeds = text_embeds.to(dtype=llm_dtype)
        inputs_embeds = torch.cat([slide_tokens, text_embeds], dim=1)

        prefix_attention_mask = torch.ones(
            (slide_tokens.shape[0], slide_tokens.shape[1]),
            dtype=text_attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([prefix_attention_mask, text_attention_mask], dim=1)

        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        if no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        generated_ids = self.llm.generate(**generate_kwargs)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)