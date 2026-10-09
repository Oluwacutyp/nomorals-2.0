"""Declarative edit spec: JSON-serializable → Timeline → render.

An :class:`EditSpec` describes *any* edit without touching code:
clips, a cut grid or explicit segment plan, transitions, text layers,
an audio mix, and output parameters. Styles build these; the engine
renders them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError
from .audio import AudioLayer, AudioMix
from .render import (
    HEIGHT,
    WIDTH,
    _materialize_stills,
    assemble_segments,
    render_timeline,
)
from .text import TextLayer
from .timeline import Clip, Effect, Timeline, Track, Transition


def _scene_to_clip(scene: Any) -> Clip:
    """Loose scene mapping → Clip.

    Scene shape: {"image", "duration", "effect", "zoom_direction"} —
    the pipeline's scene vocabulary. Stills are materialised at render.
    """
    if isinstance(scene, dict):
        image = scene.get("image", "")
        duration = scene.get("duration")
        effect = (scene.get("effect") or "").strip().lower()
        zoom_dir = (scene.get("zoom_direction") or "in").strip().lower()
    else:
        image = str(getattr(scene, "image", ""))
        duration = getattr(scene, "duration", None)
        effect = str(getattr(scene, "effect", "") or "").strip().lower()
        zoom_dir = str(getattr(scene, "zoom_direction", "in") or "").strip().lower()
    effects: tuple[Effect, ...] = ()
    if effect:
        params: dict[str, Any] = {}
        if effect in ("kenburns", "ken_burns"):
            params["direction"] = "out" if zoom_dir == "out" else "in"
        effects = (Effect(name=effect, params=params),)
    return Clip(path=image,
                end=float(duration) if duration else None,
                effects=effects)


def _coerce_text_layers(layers: Any) -> list[TextLayer]:
    if layers is None:
        return []
    if isinstance(layers, (TextLayer, str, dict)):
        layers = [layers]
    out: list[TextLayer] = []
    for l in layers:
        if isinstance(l, TextLayer):
            out.append(l)
        elif isinstance(l, str):
            if l.strip():
                out.append(TextLayer(text=l))
        elif isinstance(l, dict):
            out.append(TextLayer.from_dict(l))
        else:
            raise MediaEditError(
                f"cannot coerce {type(l).__name__} to TextLayer")
    return out


def _coerce_audio_mix(audio: Any) -> AudioMix | None:
    if audio is None:
        return None
    if isinstance(audio, AudioMix):
        return audio
    if isinstance(audio, dict):
        return AudioMix.from_dict(audio)
    raise MediaEditError(
        f"cannot coerce {type(audio).__name__} to AudioMix")


@dataclass(init=False)
class EditSpec:
    """Full declarative edit plan → :func:`render`.

    ``cut_points`` = time grid for round-robin assembly (beats, silence
    gaps, chapter marks — any grid). ``segments`` = explicit
    ``[(clip_idx, play_seconds), …]`` plan (overrides ``cut_points``).
    ``transition``: one kind/Transition for every boundary, or a
    per-boundary list. Aliases tolerated: ``beats`` → ``cut_points``,
    ``scenes`` → ``clips``.
    """
    clips: list[Clip]
    cut_points: list[float] | None
    segments: list[tuple[int, float]] | None
    transition: Any
    text_layers: list[TextLayer]
    audio: AudioMix | None
    fps: int
    width: int
    height: int
    seed: int
    crf: int | None
    preset: str | None
    profile: str | None
    out: str | None
    keep_clip_audio: bool

    def __init__(self, *args: Any, clips: Any = (),
                 cut_points: list[float] | None = None,
                 segments: Any = None,
                 transition: Any = "cut",
                 text_layers: Any = None,
                 audio: Any = None,
                 fps: int = 30,
                 width: int = WIDTH, height: int = HEIGHT,
                 seed: int = 0,
                 crf: int | None = None,
                 preset: str | None = None,
                 profile: str | None = None,
                 out: str | None = None,
                 keep_clip_audio: bool = False,
                 # tolerated aliases:
                 beats: list[float] | None = None,
                 scenes: Any = None,
                 **_ignored: Any) -> None:
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                self.__init__(**args[0])  # type: ignore[arg-type]
                return
            raise MediaEditError(
                f"EditSpec takes kwargs (or one mapping), got {len(args)} "
                "positional args")
        if scenes is not None and not clips:
            clips = [_scene_to_clip(s) for s in scenes]
        clips = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                 for c in clips]
        if beats is not None and cut_points is None:
            cut_points = list(beats)
        if segments is not None:
            segments = [(int(ci), float(d)) for ci, d in segments]
        self.clips = list(clips)
        self.cut_points = ([float(b) for b in cut_points]
                           if cut_points is not None else None)
        self.segments = segments
        self.transition = transition
        self.text_layers = _coerce_text_layers(text_layers)
        self.audio = _coerce_audio_mix(audio)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.crf = crf
        self.preset = preset
        self.profile = profile
        self.out = out
        self.keep_clip_audio = bool(keep_clip_audio)

    def to_dict(self) -> dict[str, Any]:
        def _tr(t: Any) -> Any:
            return t.to_dict() if isinstance(t, Transition) else t
        transition = ([_tr(t) for t in self.transition]
                      if isinstance(self.transition, (list, tuple))
                      else _tr(self.transition))
        return {
            "clips": [c.to_dict() for c in self.clips],
            "cut_points": list(self.cut_points) if self.cut_points else None,
            "segments": ([[ci, d] for ci, d in self.segments]
                         if self.segments else None),
            "transition": transition,
            "text_layers": [t.to_dict() for t in self.text_layers],
            "audio": self.audio.to_dict() if self.audio else None,
            "fps": self.fps, "width": self.width, "height": self.height,
            "seed": self.seed, "crf": self.crf, "preset": self.preset,
            "profile": self.profile, "out": self.out,
            "keep_clip_audio": self.keep_clip_audio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditSpec":
        return cls(
            clips=[Clip.from_dict(c) for c in data.get("clips", [])],
            cut_points=(list(data["cut_points"])
                        if data.get("cut_points") else None),
            segments=([tuple(s) for s in data["segments"]]
                      if data.get("segments") else None),
            transition=data.get("transition", "cut"),
            text_layers=[TextLayer.from_dict(t)
                         for t in data.get("text_layers", [])],
            audio=(AudioMix.from_dict(data["audio"])
                   if data.get("audio") else None),
            fps=int(data.get("fps", 30)),
            width=int(data.get("width", WIDTH)),
            height=int(data.get("height", HEIGHT)),
            seed=int(data.get("seed", 0)),
            crf=data.get("crf"), preset=data.get("preset"),
            profile=data.get("profile"), out=data.get("out"),
            keep_clip_audio=bool(data.get("keep_clip_audio", False)),
        )

    def to_timeline(self) -> Timeline:
        """Spec → Timeline model (segments resolved, not yet rendered)."""
        return Timeline(
            video=Track(kind="video", clips=list(self.clips)),
            width=self.width, height=self.height, fps=self.fps,
            seed=self.seed, text_layers=list(self.text_layers),
            audio_mix=self.audio)


def render(spec: EditSpec, *,
           workdir: str | Path | None = None,
           progress_cb: Any = None) -> str:
    """Render an :class:`EditSpec` → finished mp4. Returns the path."""
    return render_report(spec, workdir=workdir,
                         progress_cb=progress_cb)["output"]


def render_report(spec: EditSpec, *,
                  workdir: str | Path | None = None,
                  progress_cb: Any = None) -> dict[str, Any]:
    """Like :func:`render` but returns the full run report dict."""
    if not spec.clips:
        raise MediaEditError("EditSpec needs at least one clip")
    timeline = spec.to_timeline()
    report = render_timeline(
        timeline, segments=spec.segments, cut_points=spec.cut_points,
        transitions=spec.transition,
        out=spec.out, crf=spec.crf, preset=spec.preset,
        profile=spec.profile, keep_clip_audio=spec.keep_clip_audio,
        workdir=workdir, progress_cb=progress_cb)
    # cut grid provenance for the report
    if spec.cut_points:
        report["cut_points"] = list(spec.cut_points)
        report["n_cuts"] = len(spec.cut_points)
    report["seed"] = spec.seed
    return report
