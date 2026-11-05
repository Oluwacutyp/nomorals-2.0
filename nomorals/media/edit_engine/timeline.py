"""Style-agnostic timeline model — clips, tracks, transitions.

No aesthetic opinions here: a :class:`Timeline` is pure structure —
what plays when, at what speed, with which parameterized effects and
transitions. Styles (phonk, documentary, vlog, …) *compose* these
primitives; they never live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .failures import record_failure
from .frametime import FrameGrid
from .keyframes import KeyframeTrack


#: pipeline/scene effect spellings → canonical effect names.
#: Naming normalization is engine business (not style).
_EFFECT_ALIASES = {
    "kenburns": "ken_burns",
    "punchzoom": "punch_zoom",
    "punchin": "punch_zoom",
    "rgbsplit": "rgb_split",
    "glitch": "rgb_split",
    "zoom": "punch_zoom",
}


def canonical_effect(name: str) -> str:
    """Canonical effect name (case/alias-insensitive)."""
    return _EFFECT_ALIASES.get(str(name or "").strip().lower(),
                               str(name or "").strip().lower())


@dataclass
class Effect:
    """One parameterized effect on a clip.

    ``at`` = (start, end) window in seconds, clip-relative;
    ``None`` = whole clip. ``params`` are effect-specific and validated
    by the effect's builder in :mod:`effects` (never here).
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    at: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.name = canonical_effect(self.name)
        if self.at is not None:
            a, b = float(self.at[0]), float(self.at[1])
            if not a < b:
                raise ValueError(f"effect window must satisfy start < end, "
                                 f"got {self.at!r}")
            self.at = (a, b)

    def at_frames(self, fps: Any) -> tuple[int, int] | None:
        """The effect window as frame counts on the ``fps`` grid."""
        if self.at is None:
            return None
        grid = FrameGrid(fps)
        return grid.frames(self.at[0]), grid.frames(self.at[1])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params),
                "at": list(self.at) if self.at else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Effect":
        at = data.get("at")
        return cls(name=str(data["name"]),
                   params=dict(data.get("params") or {}),
                   at=(float(at[0]), float(at[1])) if at else None)


@dataclass
class Clip:
    """One source clip on a track.

    ``start``/``end`` = source window in seconds (``end=None`` = to end
    of file). ``at`` = timeline position in seconds. ``duration`` =
    timeline duration (``None`` = derived from the source window and
    ``speed``). ``speed`` = playback speed. ``gain`` = audio gain for
    the clip's own audio (when kept).
    """
    path: str
    start: float = 0.0
    end: float | None = None
    at: float = 0.0
    duration: float | None = None
    speed: float = 1.0
    effects: tuple[Effect, ...] = ()
    gain: float = 1.0
    #: stable identity keyframe tracks bind to (see :mod:`keyframes`).
    #: Empty = unaddressed; renderers also match by per-track index
    #: and by source path, in that order. Serialised only when set,
    #: so legacy clip dicts stay byte-identical.
    id: str = ""

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.at = max(0.0, float(self.at))
        self.speed = float(self.speed)
        if self.speed <= 0:
            raise ValueError("clip speed must be > 0")
        if self.end is not None:
            self.end = float(self.end)
            if self.end <= self.start:
                raise ValueError(
                    f"clip end ({self.end}) must exceed start ({self.start})")
        if self.duration is not None:
            self.duration = float(self.duration)
            if self.duration <= 0:
                raise ValueError("clip duration must be > 0")
        self.effects = tuple(
            e if isinstance(e, Effect) else Effect.from_dict(dict(e))
            for e in self.effects)

    def source_duration(self, probe_duration: float | None = None) -> float:
        """Source window length in seconds (pre-speed)."""
        end = self.end if self.end is not None else probe_duration
        if end is None:
            raise ValueError(f"clip {self.path}: end unknown and no probe "
                             "duration given")
        return max(0.0, end - self.start)

    def play_duration(self, probe_duration: float | None = None) -> float:
        """Timeline duration in seconds (post-speed)."""
        if self.duration is not None:
            return self.duration
        return self.source_duration(probe_duration) / self.speed

    # -- rational frame time (Wave 1.2a): integer views of the same --
    # -- seconds the model stores; the grid is the timeline's fps.   --
    def start_frame(self, fps: Any) -> int:
        return FrameGrid(fps).frames(self.start)

    def at_frame(self, fps: Any) -> int:
        return FrameGrid(fps).frames(self.at)

    def end_frame(self, fps: Any) -> int | None:
        if self.end is None:
            return None
        return FrameGrid(fps).frames(self.end)

    def duration_frames(self, fps: Any,
                        probe_duration: float | None = None) -> int:
        """Play duration as a whole frame count on the ``fps`` grid."""
        return FrameGrid(fps).frames(self.play_duration(probe_duration))

    def to_dict(self) -> dict[str, Any]:
        data = {"path": self.path, "start": self.start, "end": self.end,
                "at": self.at, "duration": self.duration,
                "speed": self.speed, "gain": self.gain,
                "effects": [e.to_dict() for e in self.effects]}
        if self.id:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        return cls(path=str(data["path"]),
                   start=float(data.get("start", 0.0)),
                   end=(float(data["end"]) if data.get("end") is not None
                        else None),
                   at=float(data.get("at", 0.0)),
                   duration=(float(data["duration"])
                             if data.get("duration") is not None else None),
                   speed=float(data.get("speed", 1.0)),
                   gain=float(data.get("gain", 1.0)),
                   id=str(data.get("id", "") or ""),
                   effects=tuple(Effect.from_dict(e)
                                 for e in data.get("effects", ())))


@dataclass
class Transition:
    """A boundary between two consecutive clips.

    ``kind``: cut | fade | dissolve | flash | dip | whip_pan | glitch_cut
    (see :mod:`transitions`). ``duration`` only matters for the
    overlapping kinds (fade/dissolve/whip_pan); the blink kinds
    (flash/dip) use fixed 0.06 s edge fades unless overridden via
    ``params``.
    """
    kind: str = "cut"
    duration: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .transitions import validate_transition  # lazy: no cycle
        self.kind = validate_transition(self.kind)
        self.duration = max(0.0, float(self.duration))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "duration": self.duration,
                "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "Transition":
        if isinstance(data, str):
            return cls(kind=data)
        return cls(kind=str(data.get("kind", "cut")),
                   duration=float(data.get("duration", 0.5)),
                   params=dict(data.get("params") or {}))


@dataclass
class Track:
    """One track of clips. ``kind``: "video" | "audio"."""
    kind: str = "video"
    clips: list[Clip] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ("video", "audio"):
            raise ValueError(f"track kind must be video|audio, "
                             f"got {self.kind!r}")
        self.clips = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                      for c in self.clips]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "clips": [c.to_dict() for c in self.clips]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(kind=str(data.get("kind", "video")),
                   clips=[Clip.from_dict(c)
                          for c in data.get("clips", [])])


#: blend modes an overlay track may use — the Document track
#: vocabulary (``DocTrack.blend``), shared by stills and video (1.2b).
OVERLAY_BLENDS = ("normal", "screen", "multiply", "overlay", "add")


@dataclass
class OverlayTrack:
    """One picture track composited OVER the base track (Wave 1.2b).

    The timeline-level form of a :class:`~.document.DocTrack` beyond
    the first: same blend/opacity/muted vocabulary, plus optional
    placement. ``x``/``y``/``width``/``height`` place the composited
    frame (picture-in-picture); all ``None`` = full-frame cover.
    Placement applies to ``blend="normal"`` only — the blend modes
    are full-frame by definition. Clips carry ``at`` (timeline
    position) and play windows; gaps are transparent.
    """
    kind: str = "video"  # "video" | "image"
    clips: list[Clip] = field(default_factory=list)
    blend: str = "normal"
    opacity: float = 1.0
    muted: bool = False
    label: str = ""
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("video", "image"):
            raise ValueError(f"overlay track kind must be video|image, "
                             f"got {self.kind!r}")
        if self.blend not in OVERLAY_BLENDS:
            raise ValueError(f"unknown overlay blend {self.blend!r}; "
                             f"use {list(OVERLAY_BLENDS)}")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError("overlay opacity must be in [0, 1]")
        self.opacity = float(self.opacity)
        self.clips = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                      for c in self.clips]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "clips": [c.to_dict() for c in self.clips],
                "blend": self.blend, "opacity": self.opacity,
                "muted": self.muted, "label": self.label,
                "x": self.x, "y": self.y,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlayTrack":
        return cls(kind=str(data.get("kind", "video")),
                   clips=[Clip.from_dict(c) for c in data.get("clips", [])],
                   blend=str(data.get("blend", "normal")),
                   opacity=float(data.get("opacity", 1.0)),
                   muted=bool(data.get("muted", False)),
                   label=str(data.get("label", "")),
                   x=data.get("x"), y=data.get("y"),
                   width=data.get("width"), height=data.get("height"))


@dataclass
class Timeline:
    """The full edit structure.

    ``video`` holds the picture track (clips in play order; ``at`` is
    informational — play order is list order). Audio lives in
    :class:`AudioMix` (see :mod:`audio`), text in ``text_layers``
    (see :mod:`text`) — both kept out of the track model because they
    mix/overlay rather than sequence.

    Wave 1.2: ``overlays`` carries the Document's remaining picture
    tracks (composited by :func:`render_timeline`), and ``keyframes``
    carries :class:`KeyframeTrack` animation consumed by the render.
    Both serialise only when non-empty, so legacy timeline dicts stay
    byte-identical.
    """
    video: Track
    width: int = 1080
    height: int = 1920
    fps: int = 30
    seed: int = 0
    text_layers: list = field(default_factory=list)
    audio_mix: Any = None
    overlays: list[OverlayTrack] = field(default_factory=list)
    keyframes: list[KeyframeTrack] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.video, dict):
            self.video = Track.from_dict(self.video)
        self.width = int(self.width)
        self.height = int(self.height)
        self.fps = int(self.fps)
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("timeline needs positive width/height/fps")
        self.overlays = [
            o if isinstance(o, OverlayTrack)
            else OverlayTrack.from_dict(dict(o)) for o in self.overlays]
        self.keyframes = [
            k if isinstance(k, KeyframeTrack)
            else KeyframeTrack.from_dict(dict(k)) for k in self.keyframes]

    @property
    def duration(self) -> float:
        """Sum of clip play durations (probe-free estimate)."""
        total = 0.0
        for c in self.video.clips:
            try:
                total += c.play_duration()
            except ValueError as e:
                record_failure("timeline.duration", e)
                pass
        return total

    def frame_counts(self) -> list[int]:
        """Base-track play durations as integer frame counts (1.2a).

        The exact assembly arithmetic: the timeline's total frame
        count is ``sum(frame_counts())`` with no float anywhere, so a
        10,000-cut assembly's duration is the frame sum *exactly*.
        Raises ValueError (from :meth:`Clip.play_duration`) when a
        clip's duration cannot be known without probing.
        """
        grid = FrameGrid(self.fps)
        return [grid.frames(c.play_duration()) for c in self.video.clips]

    def total_frames(self) -> int:
        """Total base-track duration in whole frames (exact)."""
        return sum(self.frame_counts())

    def quantized(self) -> "Timeline":
        """A copy with every edit time snapped to the frame grid.

        Clip windows/placements/durations, effect windows, overlay
        clips and keyframe times all become exact frame multiples at
        this timeline's fps (Wave 1.2a: assemble/disassemble at
        24/25/30/60 fps round-trips exactly through the grid).
        """
        grid = FrameGrid(self.fps)
        q = grid.quantize_float
        out = Timeline.from_dict(self.to_dict())

        def snap_clip(c: Clip) -> None:
            c.start = q(c.start)
            c.at = q(c.at)
            if c.end is not None:
                c.end = q(c.end)
            if c.duration is not None:
                c.duration = q(c.duration)
            for fx in c.effects:
                if fx.at is not None:
                    fx.at = (q(fx.at[0]), q(fx.at[1]))

        for c in out.video.clips:
            snap_clip(c)
        for ov in out.overlays:
            for c in ov.clips:
                snap_clip(c)
        for track in out.keyframes:
            for k in track.keys:
                k.time = q(k.time)
        return out

    def to_dict(self) -> dict[str, Any]:
        from .text import TextLayer  # lazy: no cycle
        layers = [l.to_dict() if isinstance(l, TextLayer) else dict(l)
                  for l in self.text_layers]
        data = {"video": self.video.to_dict(),
                "width": self.width, "height": self.height,
                "fps": self.fps, "seed": self.seed,
                "text_layers": layers,
                "audio_mix": (self.audio_mix.to_dict()
                              if self.audio_mix is not None else None)}
        if self.overlays:
            data["overlays"] = [o.to_dict() for o in self.overlays]
        if self.keyframes:
            data["keyframes"] = [k.to_dict() for k in self.keyframes]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timeline":
        from .text import TextLayer
        from .audio import AudioMix
        layers = [TextLayer.from_dict(l)
                  for l in data.get("text_layers", [])]
        am = data.get("audio_mix")
        return cls(video=Track.from_dict(data["video"]),
                   width=int(data.get("width", 1080)),
                   height=int(data.get("height", 1920)),
                   fps=int(data.get("fps", 30)),
                   seed=int(data.get("seed", 0)),
                   text_layers=layers,
                   audio_mix=AudioMix.from_dict(am) if am else None,
                   overlays=[OverlayTrack.from_dict(o)
                             for o in data.get("overlays", [])],
                   keyframes=[KeyframeTrack.from_dict(k)
                              for k in data.get("keyframes", [])])
