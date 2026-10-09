"""Compat shim — DEPRECATED, do not extend.

The style-baked edit engine lived here. It has been refactored:

* :mod:`nomorals.media.edit_engine` — style-agnostic primitives
  (timeline, effects, transitions, text, audio, render, spec).
* :mod:`nomorals.media.contentops.styles` — style presets
  (``phonk``/``documentary``/``vlog``/``minimal``), each composing
  engine primitives.

This module re-exports the old surface so existing callers (pipeline,
tests, chat wiring) keep working unchanged. ``render()`` here keeps
the legacy 4-stage behavior (beats → assembly → mix → captions) via
the ``phonk`` preset; pass ``style=`` to pick another preset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..edit_engine import (
    WIDTH,
    HEIGHT,
    Clip,
    Effect,
    EditSpec as _EngineEditSpec,
    EFFECTS,
    TextLayer,
    _effect_filter,
    apply_effect,
    assemble_segments,
    render_vertical,
)
from ..edit_engine import render_report as _engine_render_report
from ..edit_engine.text import (
    as_words as _as_words,
    build_captions,
    estimate_word_timings,
)
from ..edit_engine.audio import make_music_bed
from ...media_edit import captions as _captions
from ...media_edit.videos import MediaEditError
from . import styles
from .styles import phonk as _phonk

__all__ = [
    "Clip",
    "Effect",
    "CaptionSpec",
    "AudioSpec",
    "EditSpec",
    "EFFECTS",
    "apply_effect",
    "assemble_cuts",
    "render_vertical",
    "beat_sync_words",
    "words_to_beat_ass",
    "burn_beat_captions",
    "estimate_word_timings",
    "build_captions",
    "make_music_bed",
    "render",
    "render_report",
    "WIDTH",
    "HEIGHT",
    "_effect_filter",
]

# phonk-flavored caption helpers live in the preset now
beat_sync_words = _phonk.beat_sync_words
words_to_beat_ass = _phonk.words_to_beat_ass
burn_beat_captions = _phonk.burn_beat_captions


@dataclass
class CaptionSpec:
    """Legacy caption plan → converted to engine TextLayers at render.

    Provide exactly one of ``words`` / ``text`` / ``transcribe``.
    ``sync``: "auto" (snap estimated words to the cut grid),
    "beats", "voiceover".
    """
    words: list[dict[str, Any]] | None = None
    text: str | None = None
    transcribe: bool = False
    language: str = "en"
    style: str = "karaoke"
    sync: str = "auto"
    wpm: float = 150.0
    start: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"words": self.words, "text": self.text,
                "transcribe": self.transcribe, "language": self.language,
                "style": self.style, "sync": self.sync, "wpm": self.wpm,
                "start": self.start}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionSpec":
        return cls(**{k: data[k] for k in
                      ("words", "text", "transcribe", "language", "style",
                       "sync", "wpm", "start") if k in data})


@dataclass
class AudioSpec:
    """Legacy audio plan → converted to engine AudioMix at render."""
    music: str | None = None
    voiceover: str | None = None
    ducking: str = "sidechain"
    duck_level: float = 0.30
    target_lufs: float = -14.0
    loop_music: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"music": self.music, "voiceover": self.voiceover,
                "ducking": self.ducking, "duck_level": self.duck_level,
                "target_lufs": self.target_lufs,
                "loop_music": self.loop_music}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioSpec":
        return cls(**{k: data[k] for k in
                      ("music", "voiceover", "ducking", "duck_level",
                       "target_lufs", "loop_music") if k in data})


def _coerce_caption_spec(captions: Any) -> CaptionSpec | None:
    if captions is None or isinstance(captions, CaptionSpec):
        return captions
    if isinstance(captions, TextLayer):
        return CaptionSpec(
            words=[dict(w) for w in (captions.words or [])],
            text=captions.text, style=captions.style)
    if isinstance(captions, str):
        return CaptionSpec(text=captions) if captions.strip() else None
    if isinstance(captions, dict):
        return CaptionSpec.from_dict(captions)
    raise MediaEditError(
        f"cannot coerce {type(captions).__name__} to CaptionSpec")


def _coerce_audio_spec(audio: Any, music: str | None = None) -> AudioSpec | None:
    spec: AudioSpec | None = None
    if audio is None:
        pass
    elif isinstance(audio, AudioSpec):
        spec = audio
    elif isinstance(audio, str):
        spec = AudioSpec(voiceover=audio) if audio.strip() else None
    elif isinstance(audio, dict):
        spec = AudioSpec.from_dict(audio)
    else:
        raise MediaEditError(
            f"cannot coerce {type(audio).__name__} to AudioSpec")
    if music:
        if spec is None:
            spec = AudioSpec()
        spec = AudioSpec(
            music=music, voiceover=spec.voiceover, ducking=spec.ducking,
            duck_level=spec.duck_level, target_lufs=spec.target_lufs,
            loop_music=spec.loop_music)
    return spec


class EditSpec(_EngineEditSpec):
    """Legacy spec shape → translated to a style preset at render.

    Accepts every legacy kwarg (beats, beat_audio, captions as
    CaptionSpec/str, audio as AudioSpec/str, scenes, transition,
    music, output…). ``style`` selects the preset (default "phonk",
    which reproduces the original behavior exactly).
    """

    def __init__(self, *args: Any, clips: Any = (),
                 beats: list[float] | None = None,
                 beat_audio: str | os.PathLike[str] | None = None,
                 captions: Any = None,
                 audio: Any = None,
                 fps: int = 30,
                 width: int = WIDTH, height: int = HEIGHT,
                 seed: int = 0,
                 transition: Any = "cut",
                 crf: int | None = None,
                 preset: str | None = None,
                 profile: str | None = None,
                 out: str | None = None,
                 keep_clip_audio: bool = False,
                 style: str = "phonk",
                 # legacy aliases:
                 scenes: Any = None,
                 beat_times: list[float] | None = None,
                 output: str | None = None,
                 music: str | None = None,
                 **_ignored: Any) -> None:
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                self.__init__(**args[0])  # type: ignore[arg-type]
                return
            raise MediaEditError(
                f"EditSpec takes kwargs (or one mapping), got {len(args)} "
                "positional args")
        if beat_times is not None and beats is None:
            beats = list(beat_times)
        if output is not None and out is None:
            out = output
        # engine-level translation (cut grid, clips); the preset
        # handles captions/audio/beat_audio at render time.
        super().__init__(
            clips=clips, cut_points=beats, scenes=scenes,
            transition=transition, fps=fps, width=width, height=height,
            seed=seed, crf=crf, preset=preset, profile=profile, out=out,
            keep_clip_audio=keep_clip_audio)
        self.beat_audio = str(beat_audio) if beat_audio else None
        self.caption_spec = _coerce_caption_spec(captions)
        self.audio_spec = _coerce_audio_spec(audio, music)
        self.style = str(style or "phonk")
        # legacy attribute surface (plain attributes — the engine's
        # dataclass fields are set by super().__init__ first)
        self.beats = self.cut_points
        self.captions = self.caption_spec
        self.audio = self.audio_spec

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["beats"] = d.pop("cut_points")
        d.update({
            "beat_audio": self.beat_audio,
            "captions": (self.caption_spec.to_dict()
                         if self.caption_spec else None),
            "audio": (self.audio_spec.to_dict()
                      if self.audio_spec else None),
            "style": self.style,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditSpec":
        data = dict(data)
        # engine shape uses cut_points; legacy shape uses beats
        beats = data.pop("cut_points", None) or data.pop("beats", None)
        captions = data.pop("captions", None)
        audio = data.pop("audio", None)
        style = data.pop("style", "phonk")
        spec = cls(beats=beats, captions=captions, audio=audio,
                   style=style, **data)
        return spec


def assemble_cuts(clips: list[Clip], beats: list[float] | None, *,
                  out: str | os.PathLike[str] | None = None,
                  fps: int = 30,
                  width: int = WIDTH, height: int = HEIGHT,
                  seed: int = 0,
                  transition: str = "cut",
                  crf: int | None = None,
                  preset: str | None = None,
                  profile: str | None = None,
                  keep_audio: bool = False,
                  suffix: str = "edit",
                  workdir: str | os.PathLike[str] | None = None,
                  progress_cb: Any = None) -> dict[str, Any]:
    """Cut ``clips`` on ``beats`` → one video (single ffmpeg run).

    Legacy entry point → engine :func:`assemble_segments`.
    """
    clip_objs = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                 for c in clips]
    report = assemble_segments(
        clip_objs, cut_points=list(beats) if beats else None,
        transitions=transition, out=out, fps=fps, width=width,
        height=height, seed=seed, crf=crf, preset=preset,
        profile=profile, keep_audio=keep_audio, suffix=suffix,
        workdir=workdir, progress_cb=progress_cb)
    report["beats"] = len(beats) if beats else 0
    return report


def _resolve_caption_words(spec: EditSpec) -> tuple[Any, dict[str, Any]]:
    """caption_spec → (captions payload, preset caption kwargs)."""
    cap = spec.caption_spec
    if cap is None:
        return None, {}
    if cap.transcribe:
        au = spec.audio_spec
        vo = au.voiceover if au and au.voiceover else None
        if not vo:
            raise MediaEditError(
                "CaptionSpec(transcribe=True) needs a voiceover to "
                "transcribe")
        words = _captions.transcribe_words(vo, language=cap.language)
        if not words:
            raise MediaEditError(f"transcription produced no words for {vo}")
        payload: Any = [{"start": w.start, "end": w.end, "text": w.text}
                        for w in words]
    elif cap.words:
        payload = cap.words
    elif cap.text:
        payload = cap.text
    else:
        raise MediaEditError(
            "CaptionSpec needs words=, text=, or transcribe=True")
    kwargs = {"caption_style": cap.style, "caption_sync": cap.sync,
              "caption_wpm": cap.wpm}
    return payload, kwargs


def render(spec: EditSpec, *,
           workdir: str | os.PathLike[str] | None = None,
           progress_cb: Any = None) -> str:
    """Render a legacy :class:`EditSpec` → finished mp4.

    Dispatches to the spec's style preset (default "phonk", which
    reproduces the original beats → assembly → mix → captions pipeline
    on engine primitives). Returns the output path.
    """
    return render_report(spec, workdir=workdir,
                         progress_cb=progress_cb)["output"]


def render_report(spec: EditSpec, *,
                  workdir: str | os.PathLike[str] | None = None,
                  progress_cb: Any = None) -> dict[str, Any]:
    """Like :func:`render` but returns the full run report dict."""
    if not isinstance(spec, EditSpec):
        # tolerate a raw engine spec or mapping
        if isinstance(spec, dict):
            spec = EditSpec(spec)
        elif isinstance(spec, _EngineEditSpec):
            s = EditSpec(clips=spec.clips)
            s.__dict__.update(spec.__dict__)
            spec = s
        else:
            raise MediaEditError(
                f"cannot render {type(spec).__name__}")
    if not spec.clips:
        raise MediaEditError("EditSpec needs at least one clip")

    captions_payload, cap_kwargs = _resolve_caption_words(spec)
    au = spec.audio_spec
    preset_mod = styles.get_style(spec.style)
    # legacy beat source: explicit grid > beat_audio > music file
    beat_audio = spec.beat_audio or (au.music if au and au.music else None)
    engine_spec = preset_mod.build_spec(
        clips=spec.clips,
        beats=spec.cut_points,
        beat_audio=beat_audio,
        voiceover=au.voiceover if au else None,
        music=au.music if au else None,
        captions=captions_payload,
        transition=spec.transition,
        seed=spec.seed,
        width=spec.width, height=spec.height, fps=spec.fps,
        out=spec.out, keep_clip_audio=spec.keep_clip_audio,
        crf=spec.crf, preset=spec.preset, profile=spec.profile,
        ducking=au.ducking if au else "sidechain",
        duck_level=au.duck_level if au else 0.30,
        target_lufs=au.target_lufs if au else -14.0,
        loop_music=au.loop_music if au else True,
        **cap_kwargs)
    report = _engine_render_report(
        engine_spec, workdir=workdir, progress_cb=progress_cb)
    # legacy report keys (superset — old and new callers both work)
    grid = report.get("cut_points") or []
    report["beats"] = grid
    report["n_beats"] = report.get("n_cuts", 0)
    cap = spec.caption_spec
    synced = False
    if cap is not None:
        estimated = bool(cap.text and not cap.words)
        synced = bool((cap.sync == "beats" or
                       (cap.sync == "auto" and estimated and grid)))
    report["captions"] = {"layers": len(engine_spec.text_layers),
                          "synced_to_beats": synced,
                          "style": cap.style if cap else None}
    report["preset"] = engine_spec.preset
    report["crf"] = engine_spec.crf
    report["style"] = spec.style
    return report
