"""Tier 1 — Regional prompting, inpaint/outpaint, deterministic
adjustment stack.

Regional prompting: a prompt bound to a mask, so different regions
get different instructions. Inpaint/outpaint: deterministic spec
builders — inpaint replaces a masked region (fill prompt + mask +
strength), outpaint extends the canvas (direction + extra pixels +
blend). The adjustment stack is the deterministic finish chain
(brightness/contrast/saturation/temperature/sharpen): pure arithmetic
in a fixed order, so the same stack + same pixels = same pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .masks import Mask

ADJUST_ORDER = ("brightness", "contrast", "saturation", "temperature", "sharpen")


@dataclass
class RegionalPrompt:
    prompt: str
    mask: Mask
    strength: float = 0.8  # 0..1, how strongly the region follows the prompt

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("regional prompt needs text")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be 0..1")

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "mask": self.mask.to_dict(),
                "strength": self.strength}


def regional_prompt(prompt: str, mask: Mask, *, strength: float = 0.8) -> RegionalPrompt:
    return RegionalPrompt(prompt=prompt, mask=mask, strength=strength)


def inpaint_spec(image_path: str, mask: Mask, prompt: str, *,
                 strength: float = 0.9, seed: int = 0) -> dict[str, Any]:
    """Deterministic inpaint job spec: fill ``mask`` per ``prompt``."""
    if not prompt.strip():
        raise ValueError("inpaint needs a fill prompt")
    return {"op": "inpaint", "image": image_path, "mask": mask.to_dict(),
            "prompt": prompt, "strength": strength, "seed": seed}


def outpaint_spec(image_path: str, *, direction: str = "all",
                  extra_px: int = 256, blend_px: int = 32,
                  seed: int = 0) -> dict[str, Any]:
    """Deterministic outpaint job spec: extend canvas, blend the seam."""
    if direction not in ("all", "left", "right", "top", "bottom"):
        raise ValueError(f"unknown outpaint direction {direction!r}")
    if extra_px <= 0 or blend_px < 0:
        raise ValueError("extra_px must be > 0, blend_px >= 0")
    return {"op": "outpaint", "image": image_path, "direction": direction,
            "extra_px": extra_px, "blend_px": blend_px, "seed": seed}


class AdjustmentStack:
    """Fixed-order deterministic adjustments. Same stack = same pixels."""

    def __init__(self, adjustments: dict[str, float] | None = None) -> None:
        self.adjustments: dict[str, float] = dict(adjustments or {})
        for k in self.adjustments:
            if k not in ADJUST_ORDER:
                raise ValueError(f"unknown adjustment {k!r}")

    def set(self, name: str, value: float) -> "AdjustmentStack":
        if name not in ADJUST_ORDER:
            raise ValueError(f"unknown adjustment {name!r}")
        self.adjustments[name] = float(value)
        return self

    def apply(self, image: Any) -> Any:
        """Apply in ADJUST_ORDER with PIL ImageEnhance (pure arithmetic)."""
        try:
            from PIL import Image, ImageEnhance
        except ImportError as exc:
            raise RuntimeError("AdjustmentStack needs Pillow") from exc
        out = image.convert("RGB")
        a = self.adjustments
        if "brightness" in a:
            out = ImageEnhance.Brightness(out).enhance(1.0 + a["brightness"])
        if "contrast" in a:
            out = ImageEnhance.Contrast(out).enhance(1.0 + a["contrast"])
        if "saturation" in a:
            out = ImageEnhance.Color(out).enhance(1.0 + a["saturation"])
        if "temperature" in a:
            t = a["temperature"]
            r, g, b = out.split()
            r = r.point(lambda v: max(0, min(255, int(v * (1.0 + t)))))
            b = b.point(lambda v: max(0, min(255, int(v * (1.0 - t)))))
            out = Image.merge("RGB", (r, g, b))
        if "sharpen" in a and a["sharpen"]:
            out = ImageEnhance.Sharpness(out).enhance(1.0 + a["sharpen"])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"order": list(ADJUST_ORDER),
                "adjustments": {k: self.adjustments[k] for k in ADJUST_ORDER
                                if k in self.adjustments}}
