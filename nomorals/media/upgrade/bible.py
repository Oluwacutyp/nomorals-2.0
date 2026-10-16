"""Tier 1 — Scene bible + measured consistency + segment re-render.

The bible is the single source of truth for a scene: characters (with
reference descriptors), palette, style, and per-segment notes. Nothing
about consistency is assumed: :func:`face_similarity` computes cosine
similarity over normalised colour-histogram + edge features (a
documented proxy — a real face-embedding model plugs in via
``embed``), :func:`palette_drift` measures palette distance between
two frames, and :func:`drift_report` compares a clip's segments
against the bible. Target is 0.85, NOT 1.0 — a render at 1.0 is a
copy, not a consistent render.

Segment-level re-render: :func:`segment_needs_rerender` flags only
the bad 2-second segments (optical-flow jump, identity drift, colour
drift) so the caller re-renders 2s, not the whole clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

CONSISTENCY_TARGET = 0.85
#: Below this a segment is flagged for re-render.
DRIFT_THRESHOLD = 0.70


@dataclass
class SceneBible:
    title: str
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)
    palette: list[str] = field(default_factory=list)
    style: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "characters": self.characters,
                "palette": list(self.palette), "style": self.style,
                "segments": self.segments, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneBible":
        return cls(title=data.get("title", ""), characters=dict(data.get("characters", {})),
                   palette=list(data.get("palette", [])), style=data.get("style", ""),
                   segments=list(data.get("segments", [])), notes=data.get("notes", ""))


def _features(image: Any) -> list[float]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("consistency features need Pillow + numpy") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("expected a PIL Image")
    small = image.convert("RGB").resize((32, 32))
    arr = np.asarray(small, dtype=np.float64) / 255.0
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 1))
        hist.extend((h / h.sum()).tolist())
    grey = arr.mean(axis=2)
    gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1])).mean()
    gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :])).mean()
    return hist + [float(gx), float(gy), float(arr.mean())]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return max(0.0, min(1.0, num / (da * db)))


def face_similarity(a: Any, b: Any, *,
                    embed: Callable[[Any], list[float]] | None = None) -> float:
    """Similarity 0..1 between two face/reference frames.

    With ``embed`` (a real face-embedding model) this is cosine over
    embeddings. Without one it is cosine over colour-histogram + edge
    features — a documented proxy, labelled as such in the report.
    """
    if embed is not None:
        return _cosine(embed(a), embed(b))
    return _cosine(_features(a), _features(b))


def _hex_to_rgb(hexs: str) -> tuple[float, float, float]:
    h = hexs.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def palette_drift(palette_a: list[str], palette_b: list[str]) -> float:
    """0 = identical palettes, 1 = maximally drifted.

    Mean nearest-neighbour RGB distance between the two palettes.
    """
    if not palette_a or not palette_b:
        return 1.0
    total = 0.0
    for ha in palette_a:
        ra = _hex_to_rgb(ha)
        best = min(math.sqrt(sum((x - y) ** 2 for x, y in zip(ra, _hex_to_rgb(hb))))
                   for hb in palette_b)
        total += best
    return max(0.0, min(1.0, (total / len(palette_a)) / math.sqrt(3.0)))


@dataclass
class ConsistencyReport:
    face_similarity: float
    palette_drift: float
    overall: float
    meets_target: bool
    method: str  # "embedding" | "feature-proxy"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"face_similarity": self.face_similarity,
                "palette_drift": self.palette_drift, "overall": self.overall,
                "meets_target": self.meets_target, "method": self.method,
                "notes": list(self.notes)}


def drift_report(bible: SceneBible, frame_a: Any, frame_b: Any, *,
                 embed: Callable[[Any], list[float]] | None = None) -> ConsistencyReport:
    """Compare two frames against the bible's target (0.85, not 1.0)."""
    sim = face_similarity(frame_a, frame_b, embed=embed)
    drift = 1.0  # no palette evidence from frame_b → worst case, never assumed
    # Real drift: palette of frame_b vs bible palette — measured from the
    # frame's dominant colours when a PIL image is supplied.
    try:
        from PIL import Image as _Img
        if isinstance(frame_b, _Img.Image):
            from .conditioning import _extract_palette
            drift = palette_drift(bible.palette, _extract_palette(frame_b))
    except Exception:
        pass
    overall = 0.6 * sim + 0.4 * (1.0 - drift)
    notes = []
    if sim >= 0.99:
        notes.append("similarity ~1.0: likely a copy, not a consistent render")
    if overall < CONSISTENCY_TARGET:
        notes.append(f"below consistency target {CONSISTENCY_TARGET}")
    return ConsistencyReport(face_similarity=sim, palette_drift=drift,
                             overall=overall, meets_target=overall >= CONSISTENCY_TARGET,
                             method="embedding" if embed else "feature-proxy",
                             notes=notes)


def segment_needs_rerender(segment: dict[str, Any], *,
                           flow_jump: float = 0.0,
                           identity_drift: float = 0.0,
                           color_drift: float = 0.0) -> dict[str, Any]:
    """Flag one ~2s segment for re-render instead of the whole clip.

    Caller supplies measured drifts (from optical flow / identity /
    colour checks); this function applies the threshold and names the
    reason. A segment is flagged when ANY drift exceeds the threshold.
    """
    reasons = []
    if flow_jump > DRIFT_THRESHOLD:
        reasons.append(f"optical-flow jump {flow_jump:.2f}")
    if identity_drift > DRIFT_THRESHOLD:
        reasons.append(f"identity drift {identity_drift:.2f}")
    if color_drift > DRIFT_THRESHOLD:
        reasons.append(f"colour drift {color_drift:.2f}")
    return {"segment": segment.get("id", segment.get("start", "?")),
            "needs_rerender": bool(reasons), "reasons": reasons,
            "duration_s": segment.get("duration_s", 2.0)}
