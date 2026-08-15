"""Single device/backend layer for Mac MPS, RunPod CUDA, and CPU.

Import this module before constructing models.  MPS fallback is enabled by
default for unsupported operators, while an explicit user value is respected.
All tensors in the navigation package are expected to remain float32.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# PyTorch reads this variable when MPS is initialized.  Set it before import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    torch_version: str
    cuda_available: bool
    mps_available: bool
    mps_fallback: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def setup_mps() -> None:
    """Enable CPU fallback for MPS operators not implemented by PyTorch."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _available(name: str) -> bool:
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "mps":
        return bool(torch.backends.mps.is_available())
    if name == "cpu":
        return True
    return False


def resolve_device(preference: DevicePreference | str = "auto") -> torch.device:
    """Resolve one backend; auto uses CUDA > MPS > CPU.

    Explicit unavailable backends fail loudly so a RunPod/MPS smoke test cannot
    silently run on CPU.
    """
    setup_mps()
    pref = str(preference).lower()
    if pref not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unknown device preference: {preference}")
    if pref == "auto":
        for name in ("cuda", "mps", "cpu"):
            if _available(name):
                return torch.device(name)
        raise RuntimeError("no usable torch backend found")
    if not _available(pref):
        raise RuntimeError(
            f"requested device={pref!r} is unavailable; "
            f"cuda={torch.cuda.is_available()} "
            f"mps={torch.backends.mps.is_available()}"
        )
    return torch.device(pref)


def get_device(preference: DevicePreference | str = "auto") -> torch.device:
    """Backward-compatible alias for older navigation scripts."""
    return resolve_device(preference)


def device_info(
    preference: DevicePreference | str = "auto",
    resolved: torch.device | None = None,
) -> DeviceInfo:
    """Return serializable run metadata for CSV/log/run manifests."""
    setup_mps()
    dev = resolved or resolve_device(preference)
    return DeviceInfo(
        requested=str(preference),
        resolved=str(dev),
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=bool(torch.backends.mps.is_available()),
        mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
    )


def run_provenance() -> dict[str, str]:
    """Process-level provenance for run manifests (v0.33.3 item E5 /
    ``docs/compute_policy.md`` rule 5): git commit, machine hostname, and the
    exact command line that launched this process. Merge the result into any
    ``metadata.json`` / ``checkpoints.json`` dict alongside ``device_info``'s
    backend/torch fields. Best-effort — never raises, since provenance
    logging must not be able to crash a training run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return {
        "git_commit": commit,
        "hostname": host,
        "command": " ".join(shlex.quote(a) for a in sys.argv),
    }


def add_device_argument(parser):
    """Add the common CLI argument to an argparse parser."""
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="backend preference; auto resolves cuda > mps > cpu",
    )
    return parser
