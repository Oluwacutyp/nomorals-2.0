"""Motion Studio — the high-level product API.

One call per product: lyric videos, music visualizers, slideshows,
trailers. Every function picks profile-aware defaults, routes through
the engines, grades, exports to the requested format, and records the
ledger. Nothing here raises a raw traceback — failures come back as
:class:`MotionStudioError` with a plain-language message.

    from nomorals.media.motion_studio import studio
    studio.make_lyric_video("song.mp3", lyrics_text)
    studio.make_music_visualizer("song.mp3", images=["cover.png"])
    studio.make_slideshow(["a.png", "b.png"], audio="song.mp3")
    studio.make_trailer(["c1.mp4", "c2.mp4"], title="MIDNIGHT RUN")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ._core import (
    MotionStudioError,
    new_render_path,
    profile_defaults,
    probe_duration,
    record_ledger,
)
from .grading import export, grade
from .kenburns import kenburns
from .montage import Segment, assemble
from .typography import render_lyrics
from .visualizer import VISUAL_PRESETS, render_visualizer
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "make_lyric_video",
    "make_music_visualizer",
    "make_slideshow",
    "make_trailer",
    "quick_montage",
    "FORMATS",
]

FORMATS = ("9:16", "16:9", "1:1")


def _need(path: str | os.PathLike | None, what: str) -> Path:
    p = Path(path) if path else None
    if p is None or not p.exists():
        raise MotionStudioError(f"{what} not found: {path}")
    return p


def _finish(path: str, out: str | os.PathLike | None = None, *,
            grade_preset: str = "", format: str = "") -> str:
    """Optional grade + format export pass shared by all products.

    ``out`` (when given) is the FINAL destination: intermediate renders
    go to the workspace and the finished file is moved to ``out``.
    """
    if grade_preset:
        path = grade(path, preset=grade_preset)
    if format:
        if format not in FORMATS:
            raise MotionStudioError(
                f"unknown format {format!r} — pick from: {', '.join(FORMATS)}")
        path = export(path, format=format)
    if out:
        import shutil
        dest = Path(out)
        if Path(path).resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, dest)
            path = str(dest)
    return path


def make_lyric_video(audio: str | os.PathLike,
                     lyrics: str | Sequence,
                     out: str | os.PathLike | None = None, *,
                     preset: str = "neon_pop",
                     image: str | os.PathLike | None = None,
                     grade_preset: str = "",
                     format: str = "9:16",
                     wpm: float = 150.0,
                     size: tuple[int, int] | None = None,
                     fps: float | None = None) -> str:
    """Full lyric video: beat-snapped kinetic type over the song.

    ``lyrics`` is plain text or a timed word list. The audio is muxed;
    the video runs the song's full length. ``size``/``fps`` override the
    profile defaults (used by tests and low-power callers).
    """
    audio_p = _need(audio, "audio")
    path = render_lyrics(lyrics, None, audio=audio_p, preset=preset,
                         image=image, wpm=wpm, size=size, fps=fps)
    path = _finish(path, out, grade_preset=grade_preset, format=format)
    record_ledger({"kind": "studio.lyric_video", "path": path,
                   "preset": preset, "format": format})
    return path


def make_music_visualizer(audio: str | os.PathLike,
                          out: str | os.PathLike | None = None, *,
                          images: Sequence[str | os.PathLike] = (),
                          preset: str = "phonk",
                          duration: float | None = None,
                          grade_preset: str = "",
                          format: str = "",
                          size: tuple[int, int] | None = None,
                          fps: float | None = None) -> str:
    """Audio-reactive visualizer video with the song muxed in.

    ``preset``: lofi | phonk | shorts | ambient | club | minimal.
    ``duration`` caps the render (default: full song).
    """
    audio_p = _need(audio, "audio")
    if preset not in VISUAL_PRESETS:
        raise MotionStudioError(
            f"unknown visualizer preset {preset!r} — pick from: "
            f"{', '.join(sorted(VISUAL_PRESETS))}")
    vertical = bool(VISUAL_PRESETS[preset].get("vertical"))
    defaults = profile_defaults()
    size = size or (defaults["size"] if vertical else defaults["landscape"])
    path = render_visualizer(audio_p, None, preset=preset,
                             images=[str(i) for i in images],
                             duration=duration, size=size, fps=fps)
    path = _finish(path, out, grade_preset=grade_preset, format=format)
    record_ledger({"kind": "studio.visualizer", "path": path,
                   "preset": preset, "format": format})
    return path


def make_slideshow(images: Sequence[str | os.PathLike],
                   out: str | os.PathLike | None = None, *,
                   audio: str | os.PathLike | None = None,
                   per_image: float = 4.0,
                   move: str = "auto",
                   transition: str = "crossfade",
                   grade_preset: str = "cinematic",
                   format: str = "9:16",
                   size: tuple[int, int] | None = None,
                   fps: float | None = None) -> str:
    """Cinematic slideshow: eased Ken Burns moves + crossfades + music bed."""
    if not images:
        raise MotionStudioError("no images — nothing to slide")
    for img in images:
        _need(img, "image")
    if audio:
        _need(audio, "audio")
        total = len(images) * per_image
        song_len = probe_duration(audio)
        if song_len > 0:
            per_image = min(per_image, song_len / len(images))
    timeline = [
        Segment(kind="image", src=str(img), duration=per_image,
                move=move, transition=transition)
        for img in images
    ]
    path = assemble(timeline, None, audio=audio, size=size, fps=fps)
    path = _finish(path, out, grade_preset=grade_preset, format=format)
    record_ledger({"kind": "studio.slideshow", "path": path,
                   "images": len(images), "format": format})
    return path


def make_trailer(clips: Sequence[str | os.PathLike],
                 out: str | os.PathLike | None = None, *,
                 title: str = "", tagline: str = "",
                 audio: str | os.PathLike | None = None,
                 clip_len: float = 2.2,
                 transition: str = "cut",
                 grade_preset: str = "cinematic",
                 format: str = "16:9",
                 size: tuple[int, int] | None = None,
                 fps: float | None = None) -> str:
    """Trailer cut: title card → rapid clips → tagline card, graded.

    Clips are trimmed to ``clip_len`` each and hard-cut (or ``transition``)
    together — trailers want impact, not dissolves.
    """
    if not clips:
        raise MotionStudioError("no clips — nothing to cut")
    for c in clips:
        _need(c, "clip")
    if audio:
        _need(audio, "audio")
    timeline: list[Segment] = []
    if title:
        timeline.append(Segment(kind="text", text=title, duration=2.4,
                                preset="bold_statement", transition="fadeblack",
                                transition_duration=0.5))
    for c in clips:
        timeline.append(Segment(kind="video", src=str(c), duration=clip_len,
                                transition=transition, transition_duration=0.35))
    if tagline:
        timeline.append(Segment(kind="text", text=tagline, duration=2.6,
                                preset="neon_pop", transition="fadeblack",
                                transition_duration=0.5))
    path = assemble(timeline, None, audio=audio, size=size, fps=fps)
    path = _finish(path, out, grade_preset=grade_preset, format=format)
    record_ledger({"kind": "studio.trailer", "path": path,
                   "clips": len(clips), "format": format})
    return path


def quick_montage(clips: Sequence[str | os.PathLike],
                  audio: str | os.PathLike,
                  out: str | os.PathLike | None = None, *,
                  beats_per_clip: int = 8,
                  transition: str = "crossfade",
                  transition_duration: float = 0.4,
                  grade_preset: str = "cinematic",
                  format: str = "9:16",
                  size: tuple[int, int] | None = None,
                  fps: float | None = None,
                  seed: int = 0) -> str:
    """Beat-cut montage: the missing "make me a montage" verb.

    Detects the audio's beat grid, then cuts each clip to
    ``beats_per_clip`` beats (cycling clips when the song outlasts
    them), joined with crossfades on the beat. Falls back to even
    2.5 s cuts when beat detection is unavailable — honestly noted
    in the ledger.
    """
    if not clips:
        raise MotionStudioError("no clips — nothing to montage")
    for c in clips:
        _need(c, "clip")
    audio_p = _need(audio, "audio")
    beat_times: list[float] = []
    beat_note = ""
    try:
        from ..contentops.beats import detect_beats
        info = detect_beats(audio_p)
        beat_times = [float(t) for t in (info.times or [])]
        beat_note = f"{len(beat_times)} beats @ {info.bpm:.0f} BPM"
    except Exception as exc:  # noqa: BLE001
        beat_note = f"beat detection unavailable ({exc}) — even cuts"
    song_len = probe_duration(audio_p) or 60.0
    timeline: list[Segment] = []
    t = 0.0
    ci = 0
    import random as _random
    rng = _random.Random(seed)
    clip_list = [str(c) for c in clips]
    rng.shuffle(clip_list)
    if beat_times:
        # cut every Nth beat
        marks = beat_times[::max(1, beats_per_clip)]
        bounds = [0.0] + [m for m in marks if m > 0.5] + [song_len]
        for a, b in zip(bounds, bounds[1:]):
            if b - a < 0.4:
                continue
            src = clip_list[ci % len(clip_list)]
            ci += 1
            # random in-point so cycling clips don't repeat frames
            dur_src = probe_duration(src) or (b - a + 1.0)
            start_at = rng.uniform(0, max(0.0, dur_src - (b - a))) \
                if dur_src > (b - a) else 0.0
            timeline.append(Segment(
                kind="video", src=src, duration=b - a,
                start=start_at, transition=transition,
                transition_duration=transition_duration))
            t = b
            if t >= song_len:
                break
    else:
        per = 2.5
        n = max(1, int(song_len / per))
        for i in range(n):
            src = clip_list[i % len(clip_list)]
            timeline.append(Segment(
                kind="video", src=src, duration=per,
                transition=transition,
                transition_duration=transition_duration))
    if not timeline:
        raise MotionStudioError("montage came out empty — check the audio")
    path = assemble(timeline, None, audio=audio_p, size=size, fps=fps)
    path = _finish(path, out, grade_preset=grade_preset, format=format)
    record_ledger({"kind": "studio.quick_montage", "path": path,
                   "clips": len(clips), "beats": beat_note,
                   "format": format})
    return path
