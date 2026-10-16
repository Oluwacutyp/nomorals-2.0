"""Tier 0 — Masks / selections.

Three mask families, all deterministic and serialisable to dict:
- geometric: rect / ellipse / polygon, normalised 0..1 coordinates.
- text-promptable: a prompt string resolved against a caller-supplied
  resolver (real segmentation model lives outside this package; without
  one the mask raises a named error instead of faking a selection).
- derived: subject / depth / edges / skin / sky, computed with real
  image features (skin = YCbCr skin-tone range, sky = top-band blue
  dominance, edges = gradient magnitude, depth = vertical-gradient
  proxy, subject = centre-weighted saliency proxy). Derived masks are
  documented proxies, not model segmentations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

DERIVED_KINDS = ("subject", "depth", "edges", "skin", "sky")
GEOMETRIC_KINDS = ("rect", "ellipse", "polygon")


@dataclass
class Mask:
    kind: str  # geometric:<shape> | prompt | derived:<kind>
    params: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    inverted: bool = False
    feather: float = 0.0  # 0..1 edge softness

    def __post_init__(self) -> None:
        if self.kind.startswith("derived:"):
            name = self.kind.split(":", 1)[1]
            if name not in DERIVED_KINDS:
                raise ValueError(f"unknown derived mask {name!r}")
        elif self.kind.startswith("geometric:"):
            shape = self.kind.split(":", 1)[1]
            if shape not in GEOMETRIC_KINDS:
                raise ValueError(f"unknown geometric mask {shape!r}")
        elif self.kind != "prompt":
            raise ValueError(f"unknown mask kind {self.kind!r}")
        if not 0.0 <= self.feather <= 1.0:
            raise ValueError("feather must be 0..1")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params),
                "prompt": self.prompt, "inverted": self.inverted,
                "feather": self.feather}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mask":
        return cls(kind=data["kind"], params=dict(data.get("params", {})),
                   prompt=data.get("prompt", ""),
                   inverted=bool(data.get("inverted", False)),
                   feather=float(data.get("feather", 0.0)))

    def contains(self, x: float, y: float) -> bool:
        """Point-in-mask test for geometric masks (normalised coords)."""
        if not self.kind.startswith("geometric:"):
            raise TypeError("contains() only applies to geometric masks")
        shape = self.kind.split(":", 1)[1]
        p = self.params
        inside = False
        if shape == "rect":
            inside = (p.get("x0", 0.0) <= x <= p.get("x1", 1.0)
                      and p.get("y0", 0.0) <= y <= p.get("y1", 1.0))
        elif shape == "ellipse":
            cx, cy = p.get("cx", 0.5), p.get("cy", 0.5)
            rx, ry = p.get("rx", 0.5), p.get("ry", 0.5)
            inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0 if rx and ry else False
        elif shape == "polygon":
            pts = p.get("points", [])
            inside = _point_in_polygon(x, y, pts)
        return (not inside) if self.inverted else inside


def _point_in_polygon(x: float, y: float, pts: list) -> bool:
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def geometric_mask(shape: str, **params: Any) -> Mask:
    """Rect/ellipse/polygon mask, e.g. geometric_mask('rect', x0=0, y0=0, x1=0.5, y1=1)."""
    return Mask(kind=f"geometric:{shape}", params=dict(params))


def mask_from_prompt(prompt: str, *,
                     resolver: Callable[[str], Mask] | None = None) -> Mask:
    """Text-promptable mask.

    With a ``resolver`` (a real segmentation model adapter) the prompt
    resolves to a concrete mask. Without one, raises — a prompt mask
    must never silently become a whole-frame mask.
    """
    if not prompt.strip():
        raise ValueError("prompt mask needs a non-empty prompt")
    if resolver is not None:
        resolved = resolver(prompt)
        if not isinstance(resolved, Mask):
            raise TypeError("resolver must return a Mask")
        return resolved
    return Mask(kind="prompt", prompt=prompt.strip())


def derive_mask(image: Any, kind: str, *, threshold: float = 0.5) -> list[list[float]]:
    """Derived mask as a 2-D alpha grid (values 0..1).

    Documented feature proxies, not model segmentation:
    skin = YCbCr skin-tone box; sky = top-half blue-dominant pixels;
    edges = gradient magnitude; depth = vertical position gradient
    (near = bottom, far = top); subject = centre-weighted saliency
    (high local contrast near centre). ``threshold`` binarises when >0.
    """
    if kind not in DERIVED_KINDS:
        raise ValueError(f"unknown derived mask {kind!r}")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("derive_mask needs Pillow + numpy") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("derive_mask expects a PIL Image")
    small = image.convert("RGB").resize((64, 64))
    arr = np.asarray(small, dtype=np.float64)
    h, w, _ = arr.shape
    if kind == "skin":
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) * 0.564 + 128
        cr = (r - y) * 0.713 + 128
        score = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)).astype(float)
    elif kind == "sky":
        score = ((arr[:, :, 2] > arr[:, :, 0] + 10) & (arr[:, :, 2] > 80)).astype(float)
        yy = np.linspace(0, 1, h)[:, None]
        score = score * (1.0 - yy)  # sky lives at the top
    elif kind == "edges":
        grey = arr.mean(axis=2)
        gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        score = np.clip((gx + gy) / 48.0, 0.0, 1.0)
    elif kind == "depth":
        yy = np.linspace(1, 0, h)[:, None]
        score = np.repeat(yy, w, axis=1)
    else:  # subject: centre-weighted local contrast
        grey = arr.mean(axis=2)
        gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        contrast = np.clip((gx + gy) / 48.0, 0.0, 1.0)
        ys, xs = np.mgrid[0:h, 0:w]
        centre = 1.0 - np.sqrt(((xs - w / 2) / w) ** 2 + ((ys - h / 2) / h) ** 2) * 1.4
        score = np.clip(contrast * np.clip(centre, 0.15, 1.0), 0.0, 1.0)
    if threshold and threshold > 0:
        score = (score >= threshold).astype(float)
    return [[float(v) for v in row] for row in score]
