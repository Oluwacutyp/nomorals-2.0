"""Timeline assembler — clips + images + text + audio → final video.

A timeline is a list of segment dicts (data, not code):

    {"kind": "video", "src": "a.mp4", "duration": 4.0, "transition": "crossfade"}
    {"kind": "image", "src": "cover.png", "duration": 3.0, "move": "zoom_in"}
    {"kind": "text",  "text": "THE DROP", "duration": 2.0, "preset": "bold_statement"}

Segments are normalized to one canvas (image/text segments are rendered
through the Ken Burns / typography engines), joined with real transitions
(reused from :mod:`nomorals.media_edit.videos`), and the audio bed is
mixed underneath.

    from nomorals.media.motion_studio.montage import assemble
    assemble(timeline, "final.mp4", audio="song.mp3")
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._core import (
    MotionStudioError,
    new_render_path,
    profile_defaults,
    record_ledger,
    workdir,
)
from .kenburns import kenburns
from .typography import render_quote_card
from ...media_edit.videos import (
    concat,
    mix_audio,
    run_ffmpeg,
    transition,
    video_probe,
)
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["Segment", "normalize_segment", "assemble", "TRANSITIONS"]

#: transition name → (media_edit kind, xfade name)
TRANSITIONS: dict[str, tuple[str, str]] = {
    "cut": ("cut", ""),
    "crossfade": ("xfade", "fade"),
    "dissolve": ("xfade", "dissolve"),
    "fadeblack": ("fadeblack", ""),
    "slideleft": ("xfade", "slideleft"),
    "slideright": ("xfade", "slideright"),
    "wipeleft": ("xfade", "wipeleft"),
    "circleopen": ("xfade", "circleopen"),
}


@dataclass
class Segment:
    kind: str                      # video | image | text
    src: str = ""                  # path for video/image
    text: str = ""                 # for text segments
    duration: float = 3.0
    transition: str = "crossfade"  # transition INTO this segment
    transition_duration: float = 0.6
    move: str = "auto"             # kenburns move for image segments
    preset: str = "bold_statement"  # typography preset for text segments


def _as_segment(spec: dict[str, Any] | Segment) -> Segment:
    if isinstance(spec, Segment):
        return spec
    s = dict(spec)
    kind = str(s.pop("kind", "video"))
    return Segment(kind=kind, **{k: v for k, v in s.items()
                                 if k in Segment.__dataclass_fields__})


def _normalize_video(src: str, duration: float, size: tuple[int, int],
                     fps: float, tmp: Path) -> Path:
    """Scale/pad + trim/loop a video segment to the timeline canvas."""
    out = tmp / f"seg-{abs(hash(src + str(duration))) % 10**8}.mp4"
    if out.exists():
        return out
    vf = (f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
          f"crop={size[0]}:{size[1]},setsar=1,fps={fps:.2f}")
    # loop short inputs up to duration, trim long ones
    args = ["-y", "-stream_loop", "8", "-i", src,
            "-vf", vf, "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an", str(out)]
    try:
        run_ffmpeg(args, timeout=600.0)
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"could not normalize segment {src}: {exc}") from exc
    return out


def normalize_segment(seg: Segment, *, size: tuple[int, int] | None = None,
                      fps: float | None = None,
                      tmpdir: str | os.PathLike | None = None) -> str:
    """Render one segment to a canvas-normalized silent mp4. Returns path."""
    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    tmp = Path(tmpdir) if tmpdir else Path(tempfile.mkdtemp(prefix="montage-"))
    tmp.mkdir(parents=True, exist_ok=True)

    if seg.kind == "video":
        if not seg.src or not Path(seg.src).exists():
            raise MotionStudioError(f"segment video not found: {seg.src!r}")
        return str(_normalize_video(seg.src, seg.duration, size, fps, tmp))
    if seg.kind == "image":
        if not seg.src or not Path(seg.src).exists():
            raise MotionStudioError(f"segment image not found: {seg.src!r}")
        out = tmp / f"img-{abs(hash(seg.src)) % 10**8}.mp4"
        if not out.exists():
            kenburns(seg.src, out, duration=seg.duration, move=seg.move,
                     size=size, fps=fps)
        return str(out)
    if seg.kind == "text":
        out = tmp / f"txt-{abs(hash(seg.text)) % 10**8}.mp4"
        if not out.exists():
            render_quote_card(seg.text or "…", out, duration=seg.duration,
                              preset=seg.preset, size=size, fps=fps)
        return str(out)
    raise MotionStudioError(
        f"unknown segment kind {seg.kind!r} — video | image | text")


def _join_two(a: str, b: str, transition_name: str,
              trans_dur: float, tmp: Path) -> str:
    kind, xfade = TRANSITIONS.get(transition_name, ("xfade", "fade"))
    if kind == "cut":
        res = concat([a, b], out_dir=str(tmp))
        return str(res["output"])
    # transitions need both clips longer than the transition itself
    try:
        da = float(video_probe(a).get("duration") or 0)
        db = float(video_probe(b).get("duration") or 0)
    except Exception:  # noqa: BLE001
        da = db = 0
    dur = min(trans_dur, max(0.2, da - 0.3), max(0.2, db - 0.3))
    res = transition(a, b, kind=kind, duration=max(0.2, dur),
                     transition=xfade or "fade", out_dir=str(tmp))
    return str(res["output"])


def assemble(timeline: list[dict[str, Any] | Segment],
             out: str | os.PathLike | None = None, *,
             audio: str | os.PathLike | None = None,
             audio_duck: bool = False,
             size: tuple[int, int] | None = None,
             fps: float | None = None,
             grade_preset: str = "") -> str:
    """Assemble a timeline into one video. Returns the output path.

    ``audio`` is mixed under the full cut (``audio_duck`` ducks it under
    any segment audio — currently a straight mix; ducking hooks into
    :func:`media_edit.videos.ducking` when segments carry audio).
    ``grade_preset`` applies a motion-studio grade to the final cut.
    """
    if not timeline:
        raise MotionStudioError("empty timeline — nothing to assemble")
    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    tmp = Path(tempfile.mkdtemp(prefix="montage-"))

    segments = [_as_segment(s) for s in timeline]
    clips = [normalize_segment(s, size=size, fps=fps, tmpdir=tmp)
             for s in segments]

    joined = clips[0]
    for seg, nxt in zip(segments[1:], clips[1:]):
        joined = _join_two(joined, nxt, seg.transition,
                           seg.transition_duration, tmp)

    final = joined
    if audio:
        if not Path(audio).exists():
            raise MotionStudioError(f"audio not found: {audio}")
        mixed = tmp / "mixed.mp4"
        res = mix_audio(joined, audio, out_dir=str(tmp))
        final = str(res.get("output", joined))

    if grade_preset:
        from .grading import grade as _grade
        final = _grade(final, grade_preset)

    out_path = Path(out) if out else new_render_path("montage")
    # final container normalization (faststart etc.)
    try:
        run_ffmpeg(["-y", "-i", final, "-c", "copy",
                    "-movflags", "+faststart", str(out_path)], timeout=300.0)
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"final mux failed: {exc}") from exc

    total = sum(s.duration for s in segments)
    record_ledger({"kind": "montage", "path": str(out_path),
                   "segments": len(segments), "duration": round(total, 2)})
    return str(out_path)
