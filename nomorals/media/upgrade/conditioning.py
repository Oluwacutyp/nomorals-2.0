"""Tier 0 — Reference conditioning.

A reference (image, palette, pose, depth map, edge map, style note) is
a typed, weighted conditioning input. ``conditioning_pack`` turns a
list of references into the deterministic conditioning payload a
generation call consumes: per-type weights normalised to 1, palette
hexes extracted from a PIL image when the caller passes one, and a
stable fingerprint so the same refs + same seed reproduce the same
pack. No generation happens here — this is the conditioning contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

REFERENCE_TYPES = ("style", "character", "pose", "depth", "edge", "palette", "composition")
DEFAULT_WEIGHTS = {
    "style": 0.30, "character": 0.30, "pose": 0.15,
    "depth": 0.10, "edge": 0.05, "palette": 0.05, "composition": 0.05,
}


@dataclass
class Reference:
    kind: str  # one of REFERENCE_TYPES
    source: str  # path, URL, or inline payload id
    weight: float = 0.0  # 0 = use type default; else explicit
    note: str = ""
    image: Any = None  # optional PIL Image, only used for palette extraction

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_TYPES:
            raise ValueError(f"unknown reference kind {self.kind!r}")
        if not self.source:
            raise ValueError("reference needs a source")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("reference weight must be 0..1")

    def effective_weight(self) -> float:
        return self.weight if self.weight > 0 else DEFAULT_WEIGHTS[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source,
                "weight": self.weight, "note": self.note}


@dataclass
class ReferenceConditioning:
    refs: list[Reference] = field(default_factory=list)
    fingerprint: str = ""
    palette: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Reference]:
        return [r for r in self.refs if r.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return {"refs": [r.to_dict() for r in self.refs],
                "fingerprint": self.fingerprint, "palette": list(self.palette)}


def _extract_palette(image: Any, n: int = 5) -> list[str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("palette extraction needs Pillow") from exc
    if not isinstance(image, Image.Image):
        return []
    small = image.convert("RGB").resize((32, 32))
    colours = small.getcolors(1024)
    if not colours:
        return []
    colours.sort(reverse=True)
    return [f"#{r:02x}{g:02x}{b:02x}" for _count, (r, g, b) in colours[:n]]


def conditioning_pack(refs: list[Reference]) -> ReferenceConditioning:
    """Normalise weights, extract palettes, fingerprint the pack."""
    if not refs:
        raise ValueError("conditioning_pack needs at least one reference")
    total = sum(r.effective_weight() for r in refs)
    if total <= 0:
        raise ValueError("reference weights sum to zero")
    normalised = []
    for r in refs:
        normalised.append(Reference(kind=r.kind, source=r.source,
                                    weight=r.effective_weight() / total,
                                    note=r.note, image=r.image))
    palette: list[str] = []
    for r in normalised:
        if r.kind == "palette" and r.image is not None:
            palette.extend(_extract_palette(r.image))
    palette = list(dict.fromkeys(palette))
    payload = json.dumps([{"k": r.kind, "s": r.source, "w": round(r.weight, 6),
                           "n": r.note} for r in normalised], sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return ReferenceConditioning(refs=normalised, fingerprint=fingerprint, palette=palette)
