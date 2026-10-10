"""Capability probing for neural video generation.

Every probe is best-effort and never raises: unknown hardware is data,
not an error. The pipeline uses these to route honestly — neural when
the VRAM is there, motion studio when it isn't, and it always says
which path it took.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from ...core.logging_setup import get_logger
from ...core.profiles import get_profile_kind

_log = get_logger(__name__)

__all__ = [
    "VideogenError",
    "cuda_vram_gb",
    "diffusers_available",
    "torch_cuda_available",
    "neural_capability",
    "NeuralCapability",
    "LTX_MIN_VRAM_GB",
    "WAN_MIN_VRAM_GB",
]

#: practical VRAM floors (GB) from the feasibility research — these are
#: measured/documented numbers, not guesses (see research/video-generation.md)
LTX_MIN_VRAM_GB = 8.0    # LTX-Video 2B distilled — the only laptop-viable model
WAN_MIN_VRAM_GB = 16.0   # Wan 2.2 TI2V-5B needs 16GB+ (24GB comfortable)


class VideogenError(Exception):
    """Neural video generation failure — always plain-language."""


def cuda_vram_gb() -> float | None:
    """Total VRAM of the first CUDA device, or None when unknowable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.get_device_properties(0).total_memory) / 1e9
    except Exception:  # noqa: BLE001 - probe, None is a valid answer
        return None


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def diffusers_available() -> bool:
    return importlib.util.find_spec("diffusers") is not None


@dataclass
class NeuralCapability:
    available: bool
    backend: str          # "ltx" | "wan" | "none"
    vram_gb: float | None
    profile: str
    diffusers: bool
    reason: str           # human-readable: why this backend (or none)

    def explain(self) -> str:
        if self.available:
            vram = f"{self.vram_gb:.0f}GB VRAM" if self.vram_gb else "CUDA"
            return (f"neural video available via {self.backend} ({vram}) — "
                    f"{self.reason}")
        return f"neural video unavailable: {self.reason}"


def neural_capability(prefer: str = "auto") -> NeuralCapability:
    """Probe what neural video this machine can actually run.

    ``prefer``: "auto" | "ltx" | "wan". Auto picks the best backend the
    VRAM supports (wan for quality when ≥16GB, else ltx).
    """
    profile = get_profile_kind()
    vram = cuda_vram_gb()
    has_diffusers = diffusers_available()
    prefer = (prefer or "auto").lower()

    if profile == "termux":
        return NeuralCapability(False, "none", vram, profile, has_diffusers,
            "phone/termux has no CUDA GPU — neural video is physically "
            "impossible here (60–100x CPU slowdown); motion studio instead")
    if not torch_cuda_available():
        return NeuralCapability(False, "none", vram, profile, has_diffusers,
            "no CUDA GPU detected on this machine; motion studio instead")
    if not has_diffusers:
        return NeuralCapability(False, "none", vram, profile, False,
            "the `diffusers` package is not installed — install it "
            "(`pip install diffusers accelerate`) to unlock neural video; "
            "motion studio instead")
    vram = vram or 0.0
    if prefer == "wan":
        if vram >= WAN_MIN_VRAM_GB:
            return NeuralCapability(True, "wan", vram, profile, True,
                f"{vram:.0f}GB VRAM clears the 16GB Wan 2.2 TI2V-5B floor")
        return NeuralCapability(False, "none", vram, profile, True,
            f"Wan 2.2 needs ≥16GB VRAM, this machine has {vram:.0f}GB; "
            "motion studio instead (or pick backend=ltx)")
    if prefer == "ltx":
        if vram >= LTX_MIN_VRAM_GB:
            return NeuralCapability(True, "ltx", vram, profile, True,
                f"{vram:.0f}GB VRAM clears the 8GB LTX-Video 2B floor")
        return NeuralCapability(False, "none", vram, profile, True,
            f"LTX-Video 2B needs ≥8GB VRAM, this machine has {vram:.0f}GB; "
            "motion studio instead")
    # auto: best quality the VRAM supports
    if vram >= WAN_MIN_VRAM_GB:
        return NeuralCapability(True, "wan", vram, profile, True,
            f"{vram:.0f}GB VRAM — Wan 2.2 TI2V-5B for quality finals")
    if vram >= LTX_MIN_VRAM_GB:
        return NeuralCapability(True, "ltx", vram, profile, True,
            f"{vram:.0f}GB VRAM — LTX-Video 2B distilled for fast iteration")
    return NeuralCapability(False, "none", vram, profile, True,
        f"only {vram:.0f}GB VRAM — below the 8GB neural floor; "
        "motion studio instead")
