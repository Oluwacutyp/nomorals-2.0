"""Color grading + format exports — data-driven ffmpeg filter chains.

``GRADE_PRESETS`` maps a name to a real ffmpeg ``-vf`` chain (the LUT-ish
look, done with curves/colorbalance/eq so no LUT files are needed).
``FORMAT_SIZES`` maps aspect names to profile-aware canvases.

    from nomorals.media.motion_studio.grading import grade, export
    grade("clip.mp4", "graded.mp4", preset="cinematic")
    export("graded.mp4", "short.mp4", format="9:16")
"""

from __future__ import annotations

import os
from pathlib import Path

from ._core import (
    MotionStudioError,
    new_render_path,
    profile_defaults,
    record_ledger,
)
from ...media_edit.videos import run_ffmpeg, video_probe
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "GRADE_PRESETS", "FORMAT_SIZES", "list_grades", "grade",
    "apply_grades", "export",
]

#: name → {"label", "filter"} — every filter is a real ffmpeg -vf chain
GRADE_PRESETS: dict[str, dict[str, str]] = {
    "cinematic": {
        "label": "teal-shadow / warm-highlight blockbuster",
        "filter": ("colorbalance=gs=0.08:gm=-0.04:gh=-0.10:"
                   "bs=0.10:bm=0.02:bh=-0.06,"
                   "eq=contrast=1.08:saturation=1.12,vignette=PI/4.2"),
    },
    "noir": {
        "label": "black & white, crushed blacks",
        "filter": "hue=s=0,eq=contrast=1.25:brightness=-0.03,vignette=PI/3.5",
    },
    "vintage": {
        "label": "faded film, warm cast",
        "filter": ("curves=master='0/0 0.5/0.42 1/0.92',"
                   "colorbalance=rs=0.10:gs=0.03:bs=-0.12,"
                   "eq=saturation=0.85"),
    },
    "vibrant": {
        "label": "punchy saturated pop",
        "filter": "eq=contrast=1.12:saturation=1.45:brightness=0.02",
    },
    "phonk": {
        "label": "hard contrast, aggressive — phonk edits",
        "filter": "eq=contrast=1.35:saturation=1.30,unsharp=5:5:0.8",
    },
    "faded": {
        "label": "washed pastel fade",
        "filter": "eq=contrast=0.92:brightness=0.06:saturation=0.75",
    },
    "cold": {
        "label": "icy blue push",
        "filter": "colorbalance=bs=0.15:bm=0.06,eq=saturation=1.05",
    },
    "warm": {
        "label": "golden-hour warmth",
        "filter": "colorbalance=rs=0.15:rm=0.08,eq=saturation=1.10",
    },
    "none": {"label": "no grade (passthrough)", "filter": ""},
}


def list_grades() -> list[dict[str, str]]:
    return [{"name": n, "label": v["label"]} for n, v in GRADE_PRESETS.items()]


def grade(video: str | os.PathLike, out: str | os.PathLike | None = None, *,
          preset: str = "cinematic", crf: int | None = None) -> str:
    """Apply a color grade preset. Returns the output path."""
    spec = GRADE_PRESETS.get(preset)
    if spec is None:
        raise MotionStudioError(
            f"unknown grade {preset!r} — pick from: {', '.join(sorted(GRADE_PRESETS))}")
    src = Path(video)
    if not src.exists():
        raise MotionStudioError(f"no such video: {video}")
    out_path = Path(out) if out else new_render_path(f"graded-{preset}")
    args = ["-i", str(src)]
    if spec["filter"]:
        args += ["-vf", spec["filter"]]
    args += ["-c:v", "libx264", "-preset", "veryfast",
             "-crf", str(crf if crf is not None else 20),
             "-c:a", "copy", "-movflags", "+faststart", str(out_path)]
    try:
        run_ffmpeg(args, timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"grading failed: {exc}") from exc
    record_ledger({"kind": "grade", "path": str(out_path), "preset": preset})
    return str(out_path)


def apply_grades(video: str | os.PathLike,
                 presets: list[str]) -> str:
    """Chain several grades in one pass (single re-encode)."""
    filters = []
    for name in presets:
        spec = GRADE_PRESETS.get(name)
        if spec is None:
            raise MotionStudioError(f"unknown grade {name!r}")
        if spec["filter"]:
            filters.append(spec["filter"])
    src = Path(video)
    if not src.exists():
        raise MotionStudioError(f"no such video: {video}")
    out_path = new_render_path("graded-chain")
    args = ["-i", str(src)]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "copy", "-movflags", "+faststart", str(out_path)]
    try:
        run_ffmpeg(args, timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"grading failed: {exc}") from exc
    record_ledger({"kind": "grade", "path": str(out_path),
                   "preset": "+".join(presets)})
    return str(out_path)


#: aspect name → which profile canvas to use
FORMAT_SIZES: dict[str, str] = {
    "9:16": "portrait", "16:9": "landscape", "1:1": "square",
    "4:5": "portrait45",
}


def export(video: str | os.PathLike, out: str | os.PathLike | None = None, *,
           format: str = "9:16", pad_color: str = "black") -> str:
    """Reframe a video to a delivery aspect (crop-first, pad never stretches).

    ``format``: 9:16 | 16:9 | 1:1 | 4:5.
    """
    key = (format or "9:16").strip()
    if key not in FORMAT_SIZES:
        raise MotionStudioError(
            f"unknown format {format!r} — pick from: {', '.join(sorted(FORMAT_SIZES))}")
    src = Path(video)
    if not src.exists():
        raise MotionStudioError(f"no such video: {video}")
    defaults = profile_defaults()
    pw, ph = defaults["size"]
    lw, lh = defaults["landscape"]
    targets = {
        "portrait": (pw, ph),
        "landscape": (lw, lh),
        "square": (min(pw, ph), min(pw, ph)),
        "portrait45": (pw, int(pw * 1.25)),
    }
    tw, th = targets[FORMAT_SIZES[key]]
    # scale to cover, then center-crop — never stretches, never pads bars
    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
          f"crop={tw}:{th}")
    out_path = Path(out) if out else new_render_path(f"export-{key.replace(':', 'x')}")
    try:
        run_ffmpeg(["-i", str(src), "-vf", vf, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-c:a", "copy",
                    "-movflags", "+faststart", str(out_path)], timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"export failed: {exc}") from exc
    info = {}
    try:
        info = video_probe(out_path)
    except Exception:  # noqa: BLE001
        pass
    record_ledger({"kind": "export", "path": str(out_path), "format": key})
    return str(out_path)
