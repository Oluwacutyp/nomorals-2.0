"""Scene segmentation for movie intelligence.

PySceneDetect (AdaptiveDetector) when installed — the mined best practice.
Falls back to the existing ffmpeg select-filter detection in
studio_automation (no duplication). Then sub-segments long scenes into
highlight windows (the Cadrivyn coverage-first principle from mining).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Segment:
    index: int
    start: float      # seconds
    end: float        # seconds
    kind: str = "scene"   # "scene" | "window"
    parent: int = -1      # parent scene index for windows
    thumbnail: str = ""


@dataclass
class Segmentation:
    source: str
    segments: list[Segment] = field(default_factory=list)
    backend: str = ""     # "pyscenedetect" | "ffmpeg"
    duration: float = 0.0


def _pyscenedetect_segments(src: str, min_length: float = 1.0) -> list[tuple[float, float]] | None:
    """Try PySceneDetect AdaptiveDetector. Returns None if unavailable."""
    try:
        from scenedetect import detect, AdaptiveDetector
    except ImportError:
        return None
    try:
        cuts = detect(src, AdaptiveDetector())
    except Exception:
        return None
    bounds: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(cuts):
        s, e = start.get_seconds(), end.get_seconds()
        if e - s >= min_length:
            bounds.append((round(s, 3), round(e, 3)))
    return bounds


def _ffmpeg_segments(src: str, min_length: float = 1.0) -> list[tuple[float, float]]:
    """Fallback: reuse studio_automation's ffmpeg scene detection."""
    from ..studio_automation import detect_scenes
    scenes = detect_scenes(src, min_length=min_length)
    return [(s.start, s.end) for s in scenes]


def segment_movie(src: str, *, min_length: float = 1.0,
                  sub_window_s: float = 8.0) -> Segmentation:
    """Segment a movie into scenes, then sub-segment long scenes into windows.

    Long scenes get split into overlapping highlight windows (coverage-first:
    a 20s scene can hold multiple cool moments). Windows overlap 50% so no
    moment is cut at a boundary.
    """
    src = str(src)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    bounds = _pyscenedetect_segments(src, min_length)
    backend = "pyscenedetect"
    if bounds is None:
        bounds = _ffmpeg_segments(src, min_length)
        backend = "ffmpeg"

    segs: list[Segment] = []
    for i, (s, e) in enumerate(bounds):
        segs.append(Segment(index=len(segs), start=s, end=e, kind="scene"))
        dur = e - s
        # Sub-segment scenes longer than 2x the window.
        if dur > sub_window_s * 2:
            step = sub_window_s / 2  # 50% overlap
            t = s
            while t + sub_window_s <= e:
                segs.append(Segment(index=len(segs), start=round(t, 3),
                                    end=round(t + sub_window_s, 3),
                                    kind="window", parent=i))
                t += step
    duration = bounds[-1][1] if bounds else 0.0
    return Segmentation(source=src, segments=segs, backend=backend,
                        duration=duration)
