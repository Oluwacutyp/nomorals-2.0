"""Short-form edit engine — the "movie edit" grammar, programmatic.

Reproduces the reference style (beat-synced cuts, shake/zoom/velocity,
bold word-highlight captions, phonk-style bed under voiceover, 9:16)
with ORIGINAL or AI-generated visuals — never ripped clips.

Pipeline::

    spec = EditSpec(
        clips=[Clip("a.mp4", effects=(Effect("shake"),)),
               Clip("b.mp4", effects=(Effect("punch_zoom"),))],
        beat_audio="phonk.mp3",
        audio=AudioSpec(music="phonk.mp3", voiceover="vo.wav"),
        captions=CaptionSpec(text="they counted him out ..."),
    )
    out = render(spec)["output"]          # 1080x1920 mp4, captioned + mixed

Building blocks are usable standalone: :func:`detect_beats`,
:func:`assemble_cuts`, :func:`apply_effect`, :func:`render_vertical`,
:func:`mix_audio` (in :mod:`audio`), caption helpers below.

Determinism: every stochastic choice (shake phases, glitch phases)
derives from ``seed``; ffmpeg itself is deterministic for fixed inputs,
so ``render(spec)`` is bit-stable for a fixed seed.

Profile gating (never feature gating): :func:`detect_profile` picks
encode speed knobs — termux/mobile/embedded get ``ultrafast``+higher CRF,
workstations get ``medium``. Resolution, fps and effects are identical
everywhere.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...core.profile import detect_profile
from ...media_edit import captions as _captions
from ...media_edit.videos import (
    MediaEditError,
    run_ffmpeg,
    video_probe,
)
from .audio import mix_audio
from .beats import detect_beats, detect_beats_full

_log = get_logger(__name__)

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
]

WIDTH = 1080
HEIGHT = 1920

#: profile kind → encode knobs (speed only — never features)
_PROFILE_PERF: dict[str, dict[str, Any]] = {
    "termux": {"preset": "ultrafast", "crf": 23, "threads": 2},
    "mobile": {"preset": "ultrafast", "crf": 23, "threads": 2},
    "embedded": {"preset": "ultrafast", "crf": 24, "threads": 1},
    "pc": {"preset": "veryfast", "crf": 21, "threads": 0},
    "vps": {"preset": "veryfast", "crf": 21, "threads": 0},
    "workstation": {"preset": "medium", "crf": 19, "threads": 0},
}


def _perf_for(profile: str | None) -> dict[str, Any]:
    kind = (profile or "").strip().lower()
    if not kind:
        try:
            kind = detect_profile().kind
        except Exception:  # noqa: BLE001 — detection must never break a render
            kind = "pc"
    perf = _PROFILE_PERF.get(kind, _PROFILE_PERF["pc"])
    return {"profile": kind, **perf}


def _seed_phases(seed: int) -> tuple[float, float]:
    """Deterministic pseudo-random phases from ``seed`` (for shake etc.)."""
    s = int(seed) & 0xFFFFFFFF
    p1 = ((s * 2654435761) % 100000) / 100000.0 * 2 * math.pi
    p2 = ((s * 40503 + 17) % 100000) / 100000.0 * 2 * math.pi
    return p1, p2


# ---------------------------------------------------------------------------
# spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    """One effect on a clip. ``at`` = (start, end) window in seconds,
    clip-relative; None = whole clip."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    at: tuple[float, float] | None = None

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
    """One source clip. ``start``/``end`` = source window (seconds);
    ``end=None`` = to end of file. ``speed`` = playback speed."""
    path: str
    start: float = 0.0
    end: float | None = None
    effects: tuple[Effect, ...] = ()
    speed: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "start": self.start, "end": self.end,
                "effects": [e.to_dict() for e in self.effects],
                "speed": self.speed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        return cls(path=str(data["path"]),
                   start=float(data.get("start", 0.0)),
                   end=(float(data["end"]) if data.get("end") is not None
                        else None),
                   effects=tuple(Effect.from_dict(e)
                                 for e in data.get("effects", [])),
                   speed=float(data.get("speed", 1.0)))


@dataclass
class CaptionSpec:
    """Caption plan. Provide exactly one of ``words`` / ``text`` /
    ``transcribe``. ``sync``: "auto" (snap estimated words to the beat
    grid, keep transcribed/aligned words voiceover-timed), "beats",
    "voiceover"."""
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
    """Audio plan. ``music`` is the trending-audio slot (user-supplied)."""
    music: str | None = None
    voiceover: str | None = None
    ducking: str = "sidechain"
    duck_level: float = 0.30
    target_lufs: float = -14.0
    loop_music: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"music": self.music, "voiceover": self.voiceover,
                "ducking": self.ducking, "duck_level": self.duck_level,
                "target_lufs": self.target_lufs, "loop_music": self.loop_music}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioSpec":
        return cls(**{k: data[k] for k in
                      ("music", "voiceover", "ducking", "duck_level",
                       "target_lufs", "loop_music") if k in data})


#: pipeline/scene effect spellings → canonical effect names
_EFFECT_ALIASES = {
    "kenburns": "ken_burns",
    "punchzoom": "punch_zoom",
    "punchin": "punch_zoom",
    "rgbsplit": "rgb_split",
    "glitch": "rgb_split",
    "zoom": "punch_zoom",
}


def _scene_to_clip(scene: Any) -> Clip:
    """Pipeline scene mapping → Clip.

    Scene shape: {"image", "duration", "effect", "zoom_direction"}.
    Stills are materialised to video inside :func:`render`.
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
        name = _EFFECT_ALIASES.get(effect, effect)
        params: dict[str, Any] = {}
        if name == "ken_burns":
            params["direction"] = "out" if zoom_dir == "out" else "in"
        effects = (Effect(name=name, params=params),)
    return Clip(path=image,
                end=float(duration) if duration else None,
                effects=effects)


def _coerce_captions(captions: Any) -> CaptionSpec | None:
    if captions is None or isinstance(captions, CaptionSpec):
        return captions
    if isinstance(captions, str):
        return CaptionSpec(text=captions) if captions.strip() else None
    if isinstance(captions, dict):
        return CaptionSpec.from_dict(captions)
    raise MediaEditError(f"cannot coerce {type(captions).__name__} to CaptionSpec")


def _coerce_audio(audio: Any, music: str | None = None) -> AudioSpec | None:
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
        raise MediaEditError(f"cannot coerce {type(audio).__name__} to AudioSpec")
    if music:
        if spec is None:
            spec = AudioSpec()
        spec = AudioSpec(
            music=music, voiceover=spec.voiceover, ducking=spec.ducking,
            duck_level=spec.duck_level, target_lufs=spec.target_lufs,
            loop_music=spec.loop_music)
    return spec


@dataclass(init=False)
class EditSpec:
    """Full declarative edit plan → :func:`render`.

    Canonical kwargs: ``clips`` (list[Clip]), ``beats``, ``beat_audio``,
    ``captions`` (CaptionSpec), ``audio`` (AudioSpec), ``fps``,
    ``width``/``height``, ``seed``, ``transition``, ``crf``/``preset``/
    ``profile``, ``out``, ``keep_clip_audio``.

    Also accepts the pipeline's looser payload shape (tolerated, never
    rejected): ``scenes`` (list of {image, duration, effect,
    zoom_direction} mappings), ``beat_times``, ``output``, ``audio`` as a
    voiceover path string, ``music`` as a path string, ``captions`` as a
    text string, or a single positional mapping.
    """
    clips: list[Clip]
    beats: list[float] | None
    beat_audio: str | None
    captions: CaptionSpec | None
    audio: AudioSpec | None
    fps: int
    width: int
    height: int
    seed: int
    transition: str  # cut | flash | dip
    crf: int | None  # None = profile default
    preset: str | None  # None = profile default
    profile: str | None  # None = auto-detect
    out: str | None
    keep_clip_audio: bool

    def __init__(self, *args: Any, clips: Any = (),
                 beats: list[float] | None = None,
                 beat_audio: str | None = None,
                 captions: Any = None,
                 audio: Any = None,
                 fps: int = 30,
                 width: int = WIDTH, height: int = HEIGHT,
                 seed: int = 0,
                 transition: str = "cut",
                 crf: int | None = None,
                 preset: str | None = None,
                 profile: str | None = None,
                 out: str | None = None,
                 keep_clip_audio: bool = False,
                 # pipeline-payload aliases (tolerated):
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
        if scenes is not None and not clips:
            clips = [_scene_to_clip(s) for s in scenes]
        clips = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                 for c in clips]
        if beat_times is not None and beats is None:
            beats = list(beat_times)
        if output is not None and out is None:
            out = output
        captions = _coerce_captions(captions)
        audio = _coerce_audio(audio, music)
        self.clips = list(clips)
        self.beats = ([float(b) for b in beats]
                      if beats is not None else None)
        self.beat_audio = beat_audio
        self.captions = captions
        self.audio = audio
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.transition = str(transition)
        self.crf = crf
        self.preset = preset
        self.profile = profile
        self.out = out
        self.keep_clip_audio = bool(keep_clip_audio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clips": [c.to_dict() for c in self.clips],
            "beats": list(self.beats) if self.beats else None,
            "beat_audio": self.beat_audio,
            "captions": self.captions.to_dict() if self.captions else None,
            "audio": self.audio.to_dict() if self.audio else None,
            "fps": self.fps, "width": self.width, "height": self.height,
            "seed": self.seed, "transition": self.transition,
            "crf": self.crf, "preset": self.preset, "profile": self.profile,
            "out": self.out, "keep_clip_audio": self.keep_clip_audio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditSpec":
        return cls(
            clips=[Clip.from_dict(c) for c in data.get("clips", [])],
            beats=(list(data["beats"]) if data.get("beats") else None),
            beat_audio=data.get("beat_audio"),
            captions=(CaptionSpec.from_dict(data["captions"])
                      if data.get("captions") else None),
            audio=(AudioSpec.from_dict(data["audio"])
                   if data.get("audio") else None),
            fps=int(data.get("fps", 30)),
            width=int(data.get("width", WIDTH)),
            height=int(data.get("height", HEIGHT)),
            seed=int(data.get("seed", 0)),
            transition=str(data.get("transition", "cut")),
            crf=data.get("crf"), preset=data.get("preset"),
            profile=data.get("profile"), out=data.get("out"),
            keep_clip_audio=bool(data.get("keep_clip_audio", False)),
        )


# ---------------------------------------------------------------------------
# effect filter builders
#
# Each builder: (params, ctx) -> ffmpeg filter-chain string.
# ctx: {"w","h","fps","duration","seed","beats"}.
# Verified against ffmpeg 8.1.2. Three corrections this build needed:
# - no rand() in crop exprs → sin-hash pseudo-random offsets instead
#   (also deterministic per seed — a feature, not a workaround);
# - rgbashift instead of mergeplanes for channel splits (mergeplanes
#   rejects packed rgb24; gbrp mapping is fragile — rgbashift is exact);
# - crop w/h are init-evaluated (t fails there) → all zoom motion uses
#   scale=eval=frame (truly per-frame) + static center crop.
# ---------------------------------------------------------------------------

def _fx_shake(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    amp = float(params.get("amplitude", 14.0))
    w, h = ctx["w"], ctx["h"]
    margin = int(math.ceil(amp)) + 4
    s1, s2 = _seed_phases(int(ctx["seed"]) + int(params.get("seed_offset", 0)))
    return (
        f"crop={w - 2 * margin}:{h - 2 * margin}:"
        f"x='(iw-ow)/2+{amp:.1f}*sin(n*12.9898+{s1:.4f})':"
        f"y='(ih-oh)/2+{amp:.1f}*sin(n*78.233+{s2:.4f})',"
        f"scale={w}:{h}:flags=bilinear,setsar=1"
    )


def _fx_punch_zoom(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    # zoom 1 → zmax across the segment. scale=eval=frame is the robust
    # primitive here: this ffmpeg build evaluates crop w/h once at init
    # (t fails there), while scale with eval=frame is truly per-frame.
    zmax = float(params.get("zmax", 1.18))
    w, h, dur = ctx["w"], ctx["h"], max(ctx["duration"], 0.04)
    k = zmax - 1.0
    z = f"(1+{k:.4f}*min(t/{dur:.4f}\\,1))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
    )


def _fx_flash(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    dur = ctx["duration"]
    at = params.get("time", params.get("at", dur / 2.0))
    d = float(params.get("duration", 0.12))
    level = float(params.get("level", 0.85))
    t0 = max(0.0, float(at) - d / 2.0)
    # triangular brightness pulse — a white blink without concat surgery
    return (
        f"eq=brightness='if(between(t\\,{t0:.3f}\\,{t0 + d:.3f})\\,"
        f"{level:.2f}*(1-abs(t-{t0 + d / 2:.3f})/({d / 2:.3f}))\\,0)'"
    )


def _fx_grain(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    strength = float(params.get("strength", 9.0))
    return f"noise=alls={strength:.1f}:allf=t+u"


def _fx_vignette(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    angle = params.get("angle", "PI/4.2")
    return f"vignette={angle}"


def _fx_rgb_split(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    shift = float(params.get("shift", 10.0))
    if params.get("animate"):
        period = float(params.get("period", 0.5))
        s1, _ = _seed_phases(int(ctx["seed"]) + 7)
        env = f"{shift:.1f}*sin(2*PI*t/{period:.3f}+{s1:.4f})"
        return f"rgbashift=rh='{env}':bh='-({env})'"
    return f"rgbashift=rh={shift:.1f}:bh={-shift:.1f}"


def _fx_ken_burns(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    zoom = float(params.get("zoom", 1.25))
    direction = str(params.get("direction", "in")).lower()
    pan_x = float(params.get("pan_x", 0.0))  # px/sec drift
    pan_y = float(params.get("pan_y", 0.0))
    w, h, dur = ctx["w"], ctx["h"], max(ctx["duration"], 0.04)
    k = zoom - 1.0
    if direction == "out":
        z = f"({zoom:.4f}-{k:.4f}*min(t/{dur:.4f}\\,1))"
    else:
        z = f"(1+{k:.4f}*min(t/{dur:.4f}\\,1))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2+{pan_x:.1f}*t':"
        f"y='(ih-oh)/2+{pan_y:.1f}*t',setsar=1"
    )


def _fx_beat_pulse(params: dict[str, Any], ctx: dict[str, Any]) -> str:
    beats = [float(b) for b in (params.get("beats") or ctx.get("beats") or [])]
    beats = beats[:10]  # keep the expression small
    if not beats:
        raise MediaEditError("beat_pulse needs beats= (or ctx beats)")
    strength = float(params.get("strength", 0.10))
    width = float(params.get("width", 0.14))
    w, h = ctx["w"], ctx["h"]
    terms = "+".join(
        f"max(0\\,1-abs(t-{b:.3f})/{width:.3f})" for b in beats)
    z = f"(1+{strength:.3f}*({terms}))"
    return (
        f"scale=eval=frame:w='iw*{z}':h='ih*{z}':flags=bilinear,"
        f"crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
    )


#: effect name → filter builder. velocity_ramp is structural (rewrites the
#: chain via trim/setpts/concat) and handled separately.
EFFECTS: dict[str, Any] = {
    "shake": _fx_shake,
    "punch_zoom": _fx_punch_zoom,
    "flash": _fx_flash,
    "grain": _fx_grain,
    "vignette": _fx_vignette,
    "rgb_split": _fx_rgb_split,
    "ken_burns": _fx_ken_burns,
    "beat_pulse": _fx_beat_pulse,
}


def _effect_filter(name: str, params: dict[str, Any],
                   ctx: dict[str, Any]) -> str:
    name = _EFFECT_ALIASES.get(name, name)
    builder = EFFECTS.get(name)
    if builder is None:
        raise MediaEditError(
            f"unknown effect {name!r}; use: {sorted(EFFECTS)}")
    return builder(params or {}, ctx)


def _ctx(w: int, h: int, fps: int, duration: float, seed: int,
         beats: list[float] | None = None) -> dict[str, Any]:
    return {"w": w, "h": h, "fps": fps, "duration": duration,
            "seed": seed, "beats": beats or []}


# ---------------------------------------------------------------------------
# chain machinery — apply effects (optionally windowed) to a labelled stream
# ---------------------------------------------------------------------------

def _chain_effect(parts: list[str], cur: str, fx: Effect,
                  ctx: dict[str, Any], tag: str) -> str:
    """Append filters applying ``fx`` to stream ``cur``; return new label.

    Windowed effects (``fx.at``) split the stream into head/window/tail,
    filter the window, and concat — works for every filter, including the
    ones without timeline support (noise, vignette, ...).
    """
    if fx.name == "velocity_ramp":
        return _chain_velocity(parts, cur, fx, ctx, tag)
    filt = _effect_filter(fx.name, fx.params, ctx)
    at = fx.at
    if not at:
        parts.append(f"[{cur}]{filt}[{tag}]")
        return tag
    try:
        a, b = float(at[0]), float(at[1])
    except (TypeError, IndexError, ValueError):
        raise MediaEditError(
            f"effect window at={at!r} must be a (start, end) pair")
    dur = ctx["duration"]
    a, b = max(0.0, a), min(dur, b)
    if not a < b:
        raise MediaEditError(
            f"effect window at={(at[0], at[1])} is empty for a {dur:.2f}s clip")
    eps = 0.001
    segs: list[tuple[str, str]] = []  # (trim-args, piece-tag)
    if a > eps:
        segs.append((f"start=0:end={a:.4f}", f"{tag}p0"))
    segs.append((f"start={a:.4f}:end={b:.4f}", f"{tag}p1"))
    if b < dur - eps:
        segs.append((f"start={b:.4f}", f"{tag}p2"))
    if len(segs) == 1:
        parts.append(f"[{cur}]trim={segs[0][0]},setpts=PTS-STARTPTS,"
                     f"{filt}[{tag}]")
        return tag
    srcs: list[str] = []
    if len(segs) > 1:
        outs = "".join(f"[{tag}s{i}]" for i in range(len(segs)))
        parts.append(f"[{cur}]split={len(segs)}{outs}")
        srcs = [f"{tag}s{i}" for i in range(len(segs))]
    else:
        srcs = [cur]
    concat_in = ""
    for (trim_args, ptag), s in zip(segs, srcs):
        extra = f",{filt}" if ptag == f"{tag}p1" else ""
        parts.append(f"[{s}]trim={trim_args},setpts=PTS-STARTPTS{extra}"
                     f"[{ptag}]")
        concat_in += f"[{ptag}]"
    parts.append(f"{concat_in}concat=n={len(segs)}:v=1:a=0[{tag}]")
    return tag


def _chain_velocity(parts: list[str], cur: str, fx: Effect,
                    ctx: dict[str, Any], tag: str) -> str:
    """velocity_ramp: piecewise setpts — the one structural effect."""
    dur = ctx["duration"]
    params = fx.params or {}
    if "ramps" in params:
        ramps = [(float(t0), float(t1), float(s))
                 for t0, t1, s in params["ramps"]]
    else:
        speed = float(params.get("speed", 2.0))
        at = fx.at or (0.0, dur)
        ramps = [(float(at[0]), float(at[1]), speed)]
    if not ramps:
        raise MediaEditError("velocity_ramp needs speed= or ramps=")
    # full-coverage timeline: identity outside ramp windows
    bounds = sorted({0.0, dur} | {t for r in ramps for t in (r[0], r[1])})
    pieces: list[tuple[float, float, float]] = []
    for x0, x1 in zip(bounds, bounds[1:]):
        if x1 - x0 < 0.001:
            continue
        speed = 1.0
        for t0, t1, s in ramps:
            if t0 <= x0 + 1e-6 and x1 <= t1 + 1e-6:
                speed = s
                break
        if speed <= 0:
            raise MediaEditError(f"velocity_ramp speed must be > 0")
        pieces.append((x0, x1, speed))
    if len(pieces) == 1 and pieces[0][2] == 1.0:
        parts.append(f"[{cur}]null[{tag}]")
        return tag
    outs = "".join(f"[{tag}v{i}]" for i in range(len(pieces)))
    if len(pieces) > 1:
        parts.append(f"[{cur}]split={len(pieces)}{outs}")
        srcs = [f"{tag}v{i}" for i in range(len(pieces))]
    else:
        srcs = [cur]
    concat_in = ""
    for i, ((x0, x1, s), s_lbl) in enumerate(zip(pieces, srcs)):
        parts.append(f"[{s_lbl}]trim=start={x0:.4f}:end={x1:.4f},"
                     f"setpts=(PTS-STARTPTS)/{s:.4f},settb=AVTB[{tag}w{i}]")
        concat_in += f"[{tag}w{i}]"
    parts.append(f"{concat_in}concat=n={len(pieces)}:v=1:a=0[{tag}]")
    return tag


def _atempo_chain(speed: float) -> str:
    """atempo filter chain for ``speed`` (atempo only does 0.5–2.0)."""
    if speed <= 0:
        raise MediaEditError("speed must be > 0")
    chain: list[str] = []
    s = speed
    while s > 2.0:
        chain.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        chain.append("atempo=0.5")
        s /= 0.5
    chain.append(f"atempo={s:.4f}")
    return ",".join(chain)


# ---------------------------------------------------------------------------
# apply_effect
# ---------------------------------------------------------------------------

def apply_effect(src: str | os.PathLike[str], effect: str, *,
                 out: str | os.PathLike[str] | None = None,
                 seed: int = 0,
                 at: tuple[float, float] | None = None,
                 still_duration: float | None = None,
                 fps: int = 30,
                 suffix: str | None = None,
                 progress_cb: Any = None,
                 **params: Any) -> dict[str, Any]:
    """Apply one named effect to ``src`` → new video file.

    Effects: shake, punch_zoom, velocity_ramp, flash, grain, vignette,
    rgb_split, ken_burns, beat_pulse. ``at`` windows the effect to
    (start, end) seconds. ``still_duration`` treats an image input as a
    still of that many seconds (needed for ken_burns on photos).

    Raises MediaEditError on unknown effects — never silently ignores.
    """
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    fx = Effect(name=effect, params=dict(params), at=at)
    canon = _EFFECT_ALIASES.get(fx.name, fx.name)
    if canon not in EFFECTS and canon != "velocity_ramp":
        raise MediaEditError(
            f"unknown effect {effect!r}; use: {sorted(EFFECTS)}")
    info = video_probe(p)
    w, h = info.get("width") or 0, info.get("height") or 0
    dur = info.get("duration") or 0.0
    is_still = still_duration is not None
    if is_still:
        if not (w and h):
            # probe the image directly (ffprobe reports streams for images)
            w, h = w or 1080, h or 1080
        dur = float(still_duration)
    if dur <= 0:
        raise MediaEditError(f"could not determine duration of {src}")
    ctx = _ctx(w, h, fps, dur, seed)
    parts: list[str] = []
    _chain_effect(parts, "0:v", fx, ctx, "fx")
    out_p = Path(out) if out else p.with_name(
        f"{p.stem}.{suffix or effect}.mp4")
    cmd: list[str] = []
    if is_still:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}"]
    cmd += ["-i", str(p), "-filter_complex", ";".join(parts),
            "-map", "[fx]", "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(out_p)]
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "effect": effect}


# ---------------------------------------------------------------------------
# assemble_cuts — beat-synced multi-clip assembly, one ffmpeg run
# ---------------------------------------------------------------------------

def _clip_window(clip: Clip) -> tuple[float, float]:
    """Source window (start, end) in seconds; probes duration when needed."""
    p = Path(clip.path)
    if not p.exists():
        raise MediaEditError(f"no such clip: {clip.path}")
    info = video_probe(p)
    dur = info.get("duration") or 0.0
    start = max(0.0, float(clip.start))
    end = float(clip.end) if clip.end is not None else dur
    if end <= start:
        raise MediaEditError(
            f"clip {clip.path}: empty source window [{start}, {end}]")
    if start >= dur and dur > 0:
        raise MediaEditError(
            f"clip {clip.path}: start {start}s past duration {dur:.1f}s")
    return start, min(end, dur) if dur > 0 else end


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
                  progress_cb: Any = None) -> dict[str, Any]:
    """Cut ``clips`` on ``beats`` → one vertical video (single ffmpeg run).

    Clips are dealt round-robin across beat intervals, walking each clip's
    source window continuously (wrapping = honest looping). Segment
    boundaries are quantized to the frame grid (frame-accurate cuts).
    ``beats=None`` degrades to plain sequential assembly (no beat sync).

    ``transition``: "cut" (hard), "flash" (white blink on the cut),
    "dip" (black). ``keep_audio=True`` also cuts + speed-matches audio
    (every clip must have an audio stream, else silence is filled).
    """
    if not clips:
        raise MediaEditError("assemble_cuts needs at least one clip")
    if transition not in ("cut", "flash", "dip"):
        raise MediaEditError(
            f"unknown transition {transition!r}; use cut/flash/dip")
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset

    windows = [_clip_window(c) for c in clips]
    infos = [video_probe(c.path) for c in clips]
    has_audio = [any(s.get("type") == "audio" for s in i.get("streams", []))
                 for i in infos]

    # --- segment plan: (clip_idx, play_duration) -----------------------
    if beats:
        qb = sorted({round(float(b) * fps) / fps for b in beats if b >= 0})
        if len(qb) < 2:
            raise MediaEditError("need at least 2 beat times to cut on")
        seg_durs = [b - a for a, b in zip(qb, qb[1:]) if b - a >= 1.0 / fps]
        if not seg_durs:
            raise MediaEditError("beat grid produced no usable segments")
    else:
        seg_durs = None  # sequential: one segment per clip, full window

    plan: list[tuple[int, float]] = []  # (clip_idx, play_duration)
    if seg_durs is None:
        for i, c in enumerate(clips):
            s0, s1 = windows[i]
            d = (s1 - s0) / max(c.speed, 1e-6)
            plan.append((i, d))
    else:
        for j, d in enumerate(seg_durs):
            plan.append((j % len(clips), d))

    # --- one ffmpeg invocation ----------------------------------------
    parts: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []
    cursors = [w0 for w0, _ in windows]  # continuous walk per clip
    total_dur = 0.0
    for n, (ci, play_d) in enumerate(plan):
        clip = clips[ci]
        s0, s1 = windows[ci]
        src_dur = s1 - s0
        speed = max(float(clip.speed), 1e-6)
        need = play_d * speed  # source seconds to cover play_d
        # take source footage at the cursor, wrapping (honest loop)
        takes: list[tuple[float, float]] = []
        remaining, cur = need, cursors[ci]
        guard = 0
        while remaining > 1e-6 and guard < 64:
            take = min(remaining, s1 - cur)
            takes.append((cur, cur + take))
            remaining -= take
            cur = s0 if cur + take >= s1 - 1e-9 else cur + take
            guard += 1
        cursors[ci] = cur
        tag = f"sg{n}"
        if len(takes) == 1:
            a, b = takes[0]
            parts.append(f"[{ci}:v]trim=start={a:.4f}:end={b:.4f},"
                         f"setpts=PTS-STARTPTS[{tag}raw]")
            vcur = f"{tag}raw"
        else:
            outs = "".join(f"[{tag}r{i}]" for i in range(len(takes)))
            parts.append(f"[{ci}:v]split={len(takes)}{outs}")
            cat = ""
            for i, (a, b) in enumerate(takes):
                parts.append(f"[{tag}r{i}]trim=start={a:.4f}:end={b:.4f},"
                             f"setpts=PTS-STARTPTS[{tag}w{i}]")
                cat += f"[{tag}w{i}]"
            parts.append(f"{cat}concat=n={len(takes)}:v=1:a=0[{tag}raw]")
            vcur = f"{tag}raw"
        if abs(speed - 1.0) > 1e-6:
            parts.append(f"[{vcur}]setpts=PTS/{speed:.4f},settb=AVTB[{tag}spd]")
            vcur = f"{tag}spd"
        # normalize to the vertical canvas BEFORE effects (they assume w/h)
        parts.append(
            f"[{vcur}]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}[{tag}norm]")
        vcur = f"{tag}norm"
        ctx = _ctx(width, height, fps, play_d, seed + n * 131,
                   beats=[0.0, play_d])
        for k, fx in enumerate(clip.effects):
            vcur = _chain_effect(parts, vcur, fx, ctx, f"{tag}e{k}")
        # transition flashes at the cut
        if transition in ("flash", "dip"):
            color = "white" if transition == "flash" else "black"
            flashes = []
            if n > 0:
                flashes.append(f"fade=t=in:st=0:d=0.06:c={color}")
            if n < len(plan) - 1:
                flashes.append(
                    f"fade=t=out:st={play_d - 0.06:.3f}:d=0.06:c={color}")
            if flashes:
                parts.append(f"[{vcur}]{','.join(flashes)}[{tag}tr]")
                vcur = f"{tag}tr"
        parts.append(f"[{vcur}]format=yuv420p[{tag}v]")
        v_labels.append(f"[{tag}v]")
        total_dur += play_d
        # audio twin (optional)
        if keep_audio:
            if has_audio[ci]:
                atakes = takes  # same source takes, speed-matched
                if len(atakes) == 1:
                    a, b = atakes[0]
                    parts.append(
                        f"[{ci}:a]atrim=start={a:.4f}:end={b:.4f},"
                        f"asetpts=PTS-STARTPTS[{tag}araw]")
                    acur = f"{tag}araw"
                else:
                    outs = "".join(f"[{tag}ar{i}]" for i in range(len(atakes)))
                    parts.append(f"[{ci}:a]asplit={len(atakes)}{outs}")
                    cat = ""
                    for i, (a, b) in enumerate(atakes):
                        parts.append(
                            f"[{tag}ar{i}]atrim=start={a:.4f}:end={b:.4f},"
                            f"asetpts=PTS-STARTPTS[{tag}aw{i}]")
                        cat += f"[{tag}aw{i}]"
                    parts.append(f"{cat}concat=n={len(atakes)}:v=0:a=1"
                                 f"[{tag}araw]")
                    acur = f"{tag}araw"
                if abs(speed - 1.0) > 1e-6:
                    parts.append(f"[{acur}]{_atempo_chain(speed)}[{tag}aspd]")
                    acur = f"{tag}aspd"
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo:d={play_d:.4f}"
                             f"[{tag}araw]")
                acur = f"{tag}araw"
            parts.append(f"[{acur}]aresample=48000,aformat=channel_layouts=stereo"
                         f"[{tag}a]")
            a_labels.append(f"[{tag}a]")

    vcat = "".join(v_labels)
    if keep_audio:
        acat = "".join(a_labels)
        parts.append(f"{vcat}concat=n={len(plan)}:v=1:a=0[vcat];"
                     f"{acat}concat=n={len(plan)}:v=0:a=1[acat]")
        maps = ["-map", "[vcat]", "-map", "[acat]"]
    else:
        parts.append(f"{vcat}concat=n={len(plan)}:v=1:a=0[vcat]")
        maps = ["-map", "[vcat]"]

    out_p = Path(out) if out else Path(
        f"{Path(clips[0].path).stem}.{suffix}.mp4")
    cmd = []
    for c in clips:
        cmd += ["-i", c.path]
    cmd += ["-filter_complex", ";".join(parts), *maps,
            "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    cmd.append(str(out_p))
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=total_dur or None)
    return {"input_clips": [c.path for c in clips], "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "segments": len(plan), "duration": round(total_dur, 3),
            "beats": len(qb) if beats else 0, "fps": fps,
            "width": width, "height": height, "profile": perf["profile"]}


# ---------------------------------------------------------------------------
# render_vertical
# ---------------------------------------------------------------------------

def render_vertical(src: str | os.PathLike[str], *,
                    out: str | os.PathLike[str] | None = None,
                    fps: int = 30,
                    width: int = WIDTH, height: int = HEIGHT,
                    crf: int | None = None,
                    preset: str | None = None,
                    profile: str | None = None,
                    still_duration: float | None = None,
                    suffix: str = "vertical",
                    progress_cb: Any = None) -> dict[str, Any]:
    """Any video (or still image with ``still_duration``) → 1080x1920
    H.264 + AAC. Centre-crop fill, CFR, yuv420p."""
    p = Path(src)
    if not p.exists():
        raise MediaEditError(f"no such video: {src}")
    perf = _perf_for(profile)
    crf = perf["crf"] if crf is None else crf
    preset = perf["preset"] if preset is None else preset
    info = video_probe(p)
    dur = info.get("duration") or 0.0
    has_a = any(s.get("type") == "audio" for s in info.get("streams", []))
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p")
    out_p = Path(out) if out else p.with_name(f"{p.stem}.{suffix}.mp4")
    cmd: list[str] = []
    if still_duration is not None:
        cmd += ["-loop", "1", "-framerate", str(fps),
                "-t", f"{still_duration:.3f}"]
        dur = still_duration
    cmd += ["-i", str(p), "-vf", vf, "-fps_mode", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if has_a and still_duration is None:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd.append(str(out_p))
    run = run_ffmpeg(cmd, progress_cb=progress_cb, duration=dur or None)
    return {"input": str(p), "output": str(out_p),
            "bytes": out_p.stat().st_size, "seconds": run["seconds"],
            "fps": fps, "width": width, "height": height,
            "profile": perf["profile"]}


# ---------------------------------------------------------------------------
# captions — wired to Devon's engine (nomorals.media_edit.captions),
# extended with beat/voiceover timing
# ---------------------------------------------------------------------------

def _as_words(words: list[Any]) -> list[_captions.Word]:
    out: list[_captions.Word] = []
    for w in words:
        if isinstance(w, _captions.Word):
            out.append(w)
        else:
            out.append(_captions.Word.from_dict(dict(w)))
    return out


def beat_sync_words(words: list[Any], beats: list[float] | None, *,
                    mode: str = "both") -> list[_captions.Word]:
    """Snap word timings to the nearest beat (the edit-grid caption look).

    ``mode``: "both" (snap start+end), "start" (starts only — ends follow
    the next word). Monotonicity is enforced: no zero-length or
    overlapping words. ``beats=None``/empty → words unchanged.
    """
    ws = _as_words(words)
    if mode not in ("both", "start"):
        raise MediaEditError(f"unknown sync mode {mode!r}; use both/start")
    if not beats:
        return ws
    grid = sorted(float(b) for b in beats)

    def snap(t: float) -> float:
        return min(grid, key=lambda g: abs(g - t))

    out: list[_captions.Word] = []
    for w in ws:
        s = snap(w.start)
        e = snap(w.end) if mode == "both" else w.end
        e = max(e, s + 0.08)
        if out:
            s = max(s, out[-1].end + 0.01)
            e = max(e, s + 0.08)
        out.append(_captions.Word(start=round(s, 3), end=round(e, 3),
                                  text=w.text))
    return out


def words_to_beat_ass(words: list[Any], beats: list[float] | None = None, *,
                      style: str = "karaoke",
                      width: int = WIDTH, height: int = HEIGHT,
                      sync: bool = True) -> str:
    """Word timings → word-highlight .ass, timed to voiceover or beat grid.

    Uses Devon's caption engine (styles hormozi/karaoke/minimal/mrbeast)
    and rewrites the script resolution to the vertical canvas so libass
    renders 1:1 on 1080x1920. ``sync=False`` keeps voiceover timings
    untouched; ``sync=True`` (default) snaps words to ``beats``.
    """
    ws = _as_words(words)
    if sync and beats:
        ws = beat_sync_words(ws, beats)
    ass = _captions.words_to_ass(ws, style=style)
    ass = re.sub(r"^PlayResX:.*$", f"PlayResX: {width}", ass,
                 flags=re.M)
    ass = re.sub(r"^PlayResY:.*$", f"PlayResY: {height}", ass,
                 flags=re.M)
    return ass


def burn_beat_captions(video: str | os.PathLike[str],
                       words: list[Any],
                       beats: list[float] | None = None, *,
                       style: str = "karaoke",
                       out_dir: str | os.PathLike[str] | None = None,
                       suffix: str = "captioned",
                       sync: bool = True) -> dict[str, Any]:
    """Burn beat/voiceover-timed word-highlight captions into ``video``.

    Keeps the engine's decoupled-transcript pattern: ``{video}.words.json``
    + ``{video}.srt`` sidecars are written next to the video, so
    :func:`recaption` can fix typos without re-running anything.
    """
    p = Path(video)
    if not p.exists():
        raise MediaEditError(f"no such video: {video}")
    ws = _as_words(words)
    if not ws:
        raise MediaEditError("no words — nothing to caption")
    ass_text = words_to_beat_ass(ws, beats, style=style, sync=sync)
    ass_path = p.with_name(f"{p.stem}.{style}.ass")
    ass_path.write_text(ass_text, encoding="utf-8")
    _captions.save_transcript(p, ws)
    srt_path = p.with_suffix(".srt")
    srt_path.write_text(_captions.words_to_srt(ws), encoding="utf-8")
    run = _captions.burn_captions(p, ass_path, out_dir=out_dir, suffix=suffix)
    run["words"] = len(ws)
    run["style"] = style
    run["synced_to_beats"] = bool(sync and beats)
    return run


def estimate_word_timings(text: str, *, start: float = 0.0,
                          wpm: float = 150.0,
                          beats: list[float] | None = None
                          ) -> list[_captions.Word]:
    """APPROXIMATE word timings from plain text (even spacing at ``wpm``).

    This is an estimate for when no TTS alignment or transcription exists
    — it does not claim to be a real alignment. Pair with ``beats`` to
    snap each word's start to the beat grid (the edit-caption look).
    """
    tokens = [t for t in (text or "").split() if t]
    if not tokens:
        raise MediaEditError("no text — nothing to time")
    if wpm <= 0:
        raise MediaEditError("wpm must be > 0")
    per = 60.0 / float(wpm)
    grid = sorted(float(b) for b in beats) if beats else []
    words: list[_captions.Word] = []
    t = max(0.0, float(start))
    for tok in tokens:
        if grid:
            t = min(grid, key=lambda g: abs(g - t))
            if words:
                t = max(t, words[-1].end + 0.01)
        words.append(_captions.Word(start=round(t, 3),
                                    end=round(t + per * 0.92, 3),
                                    text=tok))
        t += per
    return words


# ---------------------------------------------------------------------------
# render(spec) — the full declarative pipeline
# ---------------------------------------------------------------------------

def _is_still(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is a still image (not a video)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 — not an image, or PIL disagrees
        return False


def render(spec: EditSpec, *, workdir: str | os.PathLike[str] | None = None,
           progress_cb: Any = None) -> str:
    """Render an :class:`EditSpec` → finished 1080x1920 mp4.

    Returns the output path (string). For the full metadata dict, see
    :func:`render_report`.

    1. beats ← spec.beats | detect(spec.beat_audio | audio.music)
    2. video ← assemble_cuts (beat-synced, effects, transitions)
    3. audio ← mix_audio (trending bed ducked under voiceover, loudnormed)
    4. captions ← burn_beat_captions (beat grid or voiceover timed)
    Deterministic for a fixed ``seed``.
    """
    return render_report(spec, workdir=workdir,
                         progress_cb=progress_cb)["output"]


def render_report(spec: EditSpec, *,
                  workdir: str | os.PathLike[str] | None = None,
                  progress_cb: Any = None) -> dict[str, Any]:
    """Like :func:`render` but returns the full run report dict."""
    if not spec.clips:
        raise MediaEditError("EditSpec needs at least one clip")
    perf = _perf_for(spec.profile)
    crf = spec.crf if spec.crf is not None else perf["crf"]
    preset = spec.preset or perf["preset"]

    # 1. beats -------------------------------------------------------
    beats = (list(spec.beats) if spec.beats else None)
    beat_src = spec.beat_audio or (spec.audio.music if spec.audio else None)
    if beats is None and beat_src:
        info = detect_beats_full(beat_src)
        beats = info.beats
        _log.info("render: %d beats (%.1f bpm, %s engine)",
                  len(beats), info.bpm, info.backend)
        if not beats:
            raise MediaEditError(
                f"no beats detected in {beat_src} — supply spec.beats "
                "explicitly or pick a track with a clear beat")

    import tempfile
    tmp = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="edit-"))
    tmp.mkdir(parents=True, exist_ok=True)

    # 1b. materialise still images → video segments (ken_burns needs
    # real frames; assemble_cuts only takes video)
    clips: list[Clip] = []
    for i, clip in enumerate(spec.clips):
        if _is_still(clip.path):
            dur = ((clip.end or 0.0) - clip.start) if clip.end else 0.0
            still = render_vertical(
                clip.path, out=tmp / f"still{i}.mp4", fps=spec.fps,
                width=spec.width, height=spec.height,
                still_duration=max(dur, 3.0), crf=crf, preset=preset,
                profile=perf["profile"])
            clips.append(Clip(path=still["output"], start=0.0,
                              end=max(dur, 3.0) if dur else None,
                              effects=clip.effects, speed=clip.speed))
        else:
            clips.append(clip)

    # 2. video -------------------------------------------------------
    video = assemble_cuts(
        clips, beats, out=tmp / "edit.mp4", fps=spec.fps,
        width=spec.width, height=spec.height, seed=spec.seed,
        transition=spec.transition, crf=crf, preset=preset,
        profile=perf["profile"], keep_audio=spec.keep_clip_audio,
        progress_cb=progress_cb)
    final_video = Path(video["output"])
    video_dur = video["duration"]

    # 3. audio -------------------------------------------------------
    audio_out: dict[str, Any] | None = None
    if spec.audio and (spec.audio.music or spec.audio.voiceover):
        a = spec.audio
        if a.music:
            audio_out = mix_audio(
                a.music, a.voiceover, out=tmp / "mix.m4a",
                ducking=a.ducking, duck_level=a.duck_level,
                target_lufs=a.target_lufs, duration=video_dur,
                loop_music=a.loop_music)
        else:
            from .audio import normalize_loudness
            audio_out = normalize_loudness(
                a.voiceover, out=tmp / "vo.m4a", target_lufs=a.target_lufs)
        muxed = tmp / "muxed.mp4"
        run_ffmpeg(["-i", str(final_video), "-i", audio_out["output"],
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(muxed)], duration=video_dur)
        final_video = muxed

    # 4. captions ----------------------------------------------------
    caption_out: dict[str, Any] | None = None
    if spec.captions:
        cs = spec.captions
        estimated = False
        if cs.words:
            words = _as_words(cs.words)
        elif cs.text:
            words = estimate_word_timings(cs.text, start=cs.start,
                                          wpm=cs.wpm)
            estimated = True
        elif cs.transcribe:
            words = _captions.transcribe_words(final_video,
                                               language=cs.language)
            if not words:
                raise MediaEditError(
                    f"transcription produced no words for {final_video}")
        else:
            raise MediaEditError(
                "CaptionSpec needs words=, text=, or transcribe=True")
        sync_mode = cs.sync
        snap = (sync_mode == "beats"
                or (sync_mode == "auto" and estimated and beats))
        if sync_mode not in ("auto", "beats", "voiceover"):
            raise MediaEditError(
                f"unknown caption sync {sync_mode!r}; "
                "use auto/beats/voiceover")
        caption_out = burn_beat_captions(
            final_video, words, beats if snap else None,
            style=cs.style, out_dir=tmp, suffix="captioned", sync=snap)
        final_video = Path(caption_out["output"])

    # 5. deliver -----------------------------------------------------
    if spec.out:
        out_p = Path(spec.out)
    else:
        out_p = Path(spec.clips[0].path).parent / (
            f"{Path(spec.clips[0].path).stem}.short.mp4")
    if final_video.resolve() != out_p.resolve():
        run_ffmpeg(["-i", str(final_video), "-c", "copy", str(out_p)],
                   duration=video_dur)
    return {"output": str(out_p), "bytes": out_p.stat().st_size,
            "duration": video_dur, "beats": beats or [],
            "n_beats": len(beats or []), "profile": perf["profile"],
            "preset": preset, "crf": crf, "seed": spec.seed,
            "video": video, "audio": audio_out, "captions": caption_out,
            "workdir": str(tmp)}


# ---------------------------------------------------------------------------
# pipeline-contract helpers (used by ShortPipeline's stages)
# ---------------------------------------------------------------------------

def build_captions(phrases: list[tuple[str, float, float]],
                   out_path: str | os.PathLike[str]) -> str:
    """Write an SRT sidecar from ``[(text, start_s, end_s), …]``.

    Returns the path. (The word-highlight .ass path lives in
    :func:`burn_beat_captions`; this is the pipeline stage's SRT contract.)
    """
    words = [_captions.Word(start=float(a), end=float(b), text=str(t))
             for t, a, b in phrases]
    if not words:
        raise MediaEditError("no caption phrases — nothing to write")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_captions.words_to_srt(words), encoding="utf-8")
    return str(p)


def make_music_bed(duration_s: float, out_path: str | os.PathLike[str],
                   seed: int = 7) -> str:
    """Deterministic ambient pad bed (numpy → wav). Returns the path.

    A quiet (-18 dB) minor pad with a slow filter sweep and fade in/out —
    a *bed* to sit under a voiceover, not a track. The real trending
    audio comes through ``AudioSpec.music``; this is the fallback the
    pipeline uses when no track is supplied.
    """
    import numpy as np

    sr = 22050
    n = max(1, int(float(duration_s) * sr))
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    t = np.arange(n) / sr
    # Am – F – C – G roots, 8 s each, detuned saw-ish stack
    roots = [110.0, 87.31, 130.81, 98.0]
    seg = 8.0
    y = np.zeros(n)
    for i in range(int(math.ceil(float(duration_s) / seg))):
        f0 = roots[i % len(roots)]
        a, b = int(i * seg * sr), min(n, int((i + 1) * seg * sr))
        if b <= a:
            continue
        tt = t[a:b] - t[a]
        env = np.minimum(1.0, tt / 2.0) * np.minimum(
            1.0, np.maximum((b - a) / sr - tt, 0.0) / 2.0)
        env = np.clip(env, 0, 1)
        det = 1.0 + (rng.rand() - 0.5) * 0.004
        tone = (np.sin(2 * np.pi * f0 * tt)
                + 0.6 * np.sin(2 * np.pi * f0 * det * tt)
                + 0.35 * np.sin(2 * np.pi * f0 * 2.0 * tt + 0.7)
                + 0.2 * np.sin(2 * np.pi * f0 * 3.0 * tt + 1.9))
        # one-pole lowpass with a slow sweep (breathing feel)
        cutoff = 600 + 500 * np.sin(2 * np.pi * tt / seg + i)
        alpha = np.clip(2 * np.pi * cutoff / sr, 0.001, 1.0)
        lp = np.zeros_like(tone)
        acc = 0.0
        for j in range(len(tone)):
            acc += alpha[j] * (tone[j] - acc)
            lp[j] = acc
        y[a:b] += lp * env
    # gentle fade in/out + bed level
    f = int(min(n, sr * 1.5))
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 10 ** (-18 / 20)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import wave as _wave
    with _wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(p)
