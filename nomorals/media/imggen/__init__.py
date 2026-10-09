"""Devon's own native image generation organ (``nomorals.media.imggen``).

Hand-built diffusion: DDPM + DDIM schedulers, a UNet, LoRA fine-tuning,
a training loop, and a text-to-image pipeline — Devon's own code, not a
fan-out to someone else's API.

Layout:
- :mod:`.diffusion` — DDPM/DDIM schedulers (numpy + torch, no torch needed
  for the math)
- :mod:`.unet` — hand-built UNet (requires torch)
- :mod:`.lora` — LoRA injection/training (requires torch)
- :mod:`.train` — real training loop (requires torch)
- :mod:`.pipeline` — text-to-image inference (requires torch)
- :mod:`.data` — dataset tooling (captioning, manifests)

Torch-gated modules expose ``TORCH_AVAILABLE`` and raise a clear
:class:`ImgGenError` (never a bare ImportError) when torch is missing.
"""

from __future__ import annotations

__all__ = [
    "ImgGenError",
    "TORCH_AVAILABLE",
    "checkpoint_dir",
]

try:  # optional: the whole organ degrades honestly without torch
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without torch
    TORCH_AVAILABLE = False


class ImgGenError(Exception):
    """Raised for every imggen failure with a human-usable message."""


def checkpoint_dir() -> str:
    """Directory where Devon-native checkpoints live."""
    import os

    return os.environ.get(
        "DEVON_IMGGEN_DIR",
        os.path.expanduser("~/.nomorals/imggen"),
    )
