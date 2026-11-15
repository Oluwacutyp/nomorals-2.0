"""Trend formats — one trend = one composable edit spec.

A trend format is DATA, not code: audio, pacing/ramp curve, caption
system, transitions and structure, stored as JSON in the format
library (:mod:`nomorals.media.trendops.library`) with an expiry date.
Trends are presets with expiry dates, per the 2026-10-09 architecture
correction — the edit engine stays style-agnostic and a trend composes
its primitives (never a hardcoded look in the engine).

A format builds an :class:`~nomorals.media.edit_engine.EditSpec` from
footage through :meth:`TrendFormat.build_spec`: the footage is split
into atomic pieces at the cut grid (beats, or a fixed interval), each
piece plays at the speed the ramp curve prescribes at its midpoint,
each piece carries one motion effect from the cycle, the base style
preset composes captions + the audio mix + transitions, and the
explicit segment plan pins the piece order. Nothing is faked: a
format whose cut rule needs beats and has no beat source fails closed
naming what is missing, and an expired format never applies.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...media_edit.videos import MediaEditError, video_probe
from ..contentops.styles import build_spec as build_style_spec
from ..contentops.styles import get_style as get_base_style
from ..edit_engine import Clip, Effect, list_effects, validate_transition

__all__ = ["TrendFormat", "ROADSTER_SEED", "CAPTION_STYLES", "CUT_RULES",
           "DEFAULT_EXPIRY_DAYS"]

#: Caption rendering choices (the engine's caption styles).
CAPTION_STYLES = ("karaoke", "hormozi", "minimal", "mrbeast")

#: How the cut grid is derived.
CUT_RULES = ("beats", "interval", "none")

#: A captured trend lives this long by default — trends die.
DEFAULT_EXPIRY_DAYS = 14

#: Seed formats live this long from the day the library is first built.
SEED_EXPIRY_DAYS = 30

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")

#: The roadster ramp: slow cinematic roll-in, then snap fast on the
#: drop — the GTA-edit grammar from the owner's reference clip.
ROADSTER_RAMP: list[list[Any]] = [[0.0, 0.35, 0.5], [0.35, 0.7, 1.0],
                                  [0.7, 1.0, 2.0]]

#: The reference format, from the owner's TikTok clip
#: ("Me and Bro in 2030 after he got the new Tesla Roadster",
#: @smq.lul): velocity ramps, beat-synced cinematic angle switches,
#: meme caption. Seeded into a fresh library; everything else about
#: it is data a later capture can replace.
ROADSTER_SEED: dict[str, Any] = {
    "slug": "roadster",
    "name": "Roadster GTA edit",
    "source": "tiktok",
    "niche": "gta",
    "description": ("GTA edit: slow cinematic roll-in, snap fast on "
                    "the drop, beat-synced angle switches, meme caption."),
    "reference_url": "",
    "base_style": "phonk",
    "ramp_curve": [list(p) for p in ROADSTER_RAMP],
    "caption": {"style": "hormozi", "text": "",
                "position": "center"},
    "transitions": "cut",
    "effect_cycle": ["shake", "punch_zoom", "beat_pulse"],
    "cut_rule": "beats",
    "interval_s": 1.0,
    "canvas": {"width": 1080, "height": 1920, "fps": 30},
}


def _slug(value: Any) -> str:
    slug = str(value or "").strip().lower()
    if not _SLUG_RE.match(slug):
        raise MediaEditError(
            f"trend slug {value!r} must be a lowercase slug (a-z0-9_)")
    return slug


def _ramp_curve(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MediaEditError("ramp_curve needs at least one piece")
    out: list[list[float]] = []
    for piece in value:
        if not isinstance(piece, (list, tuple)) or len(piece) != 3:
            raise MediaEditError(
                f"ramp piece {piece!r} must be [start, end, speed]")
        start, end, speed = float(piece[0]), float(piece[1]), float(piece[2])
        if not 0.0 <= start < end <= 1.0:
            raise MediaEditError(
                f"ramp piece {piece!r}: fractions must satisfy "
                "0 <= start < end <= 1")
        if not 0.0 < speed <= 16.0:
            raise MediaEditError(
                f"ramp piece {piece!r}: speed must be in (0, 16]")
        out.append([start, end, speed])
    out.sort(key=lambda p: p[0])
    for a, b in zip(out, out[1:], strict=False):
        if b[0] < a[1]:
            raise MediaEditError(
                f"ramp pieces overlap: {a} and {b}")
    return out


@dataclass
class TrendFormat:
    """One trend as data: everything an edit needs, nothing more.

    ``ramp_curve`` = [[start_frac, end_frac, speed], …] fractions of
    the footage duration (0..1), resolved to seconds against the
    probed footage at build time. ``expires_at`` is an epoch — a
    format past it is dead and never applies. ``uses`` counts
    successful applies (the library bumps it, never the caller).
    """

    slug: str
    name: str = ""
    source: str = ""
    niche: str = ""
    description: str = ""
    reference_url: str = ""
    base_style: str = "phonk"
    ramp_curve: list[list[float]] = field(
        default_factory=lambda: [list(p) for p in ROADSTER_RAMP])
    caption: dict[str, Any] = field(default_factory=lambda: {
        "style": "hormozi", "text": "", "position": "center"})
    transitions: Any = "cut"
    effect_cycle: list[str] = field(
        default_factory=lambda: ["shake", "punch_zoom", "beat_pulse"])
    cut_rule: str = "beats"
    interval_s: float = 1.0
    canvas: dict[str, int] = field(
        default_factory=lambda: {"width": 1080, "height": 1920, "fps": 30})
    expires_at: float = 0.0
    created_at: float = 0.0
    uses: int = 0
    measured: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slug = _slug(self.slug)
        self.name = str(self.name or self.slug)
        self.source = str(self.source or "")
        self.niche = str(self.niche or "")
        self.description = str(self.description or "")
        self.reference_url = str(self.reference_url or "")
        self.base_style = str(self.base_style or "phonk").strip().lower()
        get_base_style(self.base_style)  # raises MediaEditError when unknown
        self.ramp_curve = _ramp_curve(self.ramp_curve)
        cap = dict(self.caption or {})
        style = str(cap.get("style", "hormozi")).strip().lower()
        if style not in CAPTION_STYLES:
            raise MediaEditError(
                f"caption style {style!r} unknown; use: {CAPTION_STYLES}")
        cap["style"] = style
        cap["text"] = str(cap.get("text", "") or "")
        cap["position"] = str(cap.get("position", "center") or "center")
        self.caption = cap
        try:
            validate_transition(
                self.transitions if isinstance(self.transitions, str)
                else "cut")
        except MediaEditError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raise as edit error
            raise MediaEditError(str(exc)) from exc
        if isinstance(self.transitions, (list, tuple)):
            for t in self.transitions:
                validate_transition(t if isinstance(t, str) else "cut")
        known = set(list_effects()) | {"velocity_ramp"}
        for name in (self.effect_cycle or []):
            if str(name) not in known:
                raise MediaEditError(
                    f"effect {name!r} is not an engine effect; "
                    f"use: {sorted(known)}")
        if not self.effect_cycle:
            raise MediaEditError("effect_cycle needs at least one effect")
        if self.cut_rule not in CUT_RULES:
            raise MediaEditError(
                f"cut_rule {self.cut_rule!r} unknown; use: {CUT_RULES}")
        if self.cut_rule == "interval" and float(self.interval_s) <= 0:
            raise MediaEditError("interval_s must be > 0 for cut_rule interval")
        self.interval_s = float(self.interval_s)
        canvas = dict(self.canvas or {})
        self.canvas = {
            "width": int(canvas.get("width", 1080)),
            "height": int(canvas.get("height", 1920)),
            "fps": int(canvas.get("fps", 30)),
        }
        if self.canvas["width"] <= 0 or self.canvas["height"] <= 0 \
                or self.canvas["fps"] <= 0:
            raise MediaEditError("canvas width/height/fps must be > 0")
        self.expires_at = float(self.expires_at or 0.0)
        self.created_at = float(self.created_at or time.time())
        self.uses = int(self.uses or 0)
        self.measured = dict(self.measured or {})

    def is_live(self, now: float | None = None) -> bool:
        """True while the trend has not expired (0 = never expires)."""
        if not self.expires_at:
            return True
        return float(now if now is not None else time.time()) < self.expires_at

    def days_left(self, now: float | None = None) -> float | None:
        """Days until expiry; None when the format never expires."""
        if not self.expires_at:
            return None
        now = float(now if now is not None else time.time())
        return max(0.0, (self.expires_at - now) / 86400.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "name": self.name, "source": self.source,
            "niche": self.niche, "description": self.description,
            "reference_url": self.reference_url,
            "base_style": self.base_style,
            "ramp_curve": [list(p) for p in self.ramp_curve],
            "caption": dict(self.caption),
            "transitions": self.transitions,
            "effect_cycle": list(self.effect_cycle),
            "cut_rule": self.cut_rule, "interval_s": self.interval_s,
            "canvas": dict(self.canvas),
            "expires_at": self.expires_at, "created_at": self.created_at,
            "uses": self.uses, "measured": dict(self.measured),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendFormat:
        return cls(**{k: v for k, v in dict(data or {}).items()})

    # -- spec building -------------------------------------------------

    def _speed_at(self, frac: float) -> float:
        for start, end, speed in self.ramp_curve:
            if start <= frac < end or (frac >= 1.0 and end == 1.0):
                return speed
        return self.ramp_curve[-1][2]

    def build_spec(self, footage: list[str], *, beats: list[float] | None = None,
                   music: str | None = None, caption: str | None = None,
                   out: str | None = None, seed: int = 0,
                   crf: int | None = None) -> Any:
        """Footage + beats → engine EditSpec in this format's grammar.

        The footage is split into atomic pieces at the cut grid; each
        piece plays at the ramp speed for its midpoint and carries one
        motion effect from the cycle; the base style preset composes
        captions + the audio mix + transitions; the explicit segment
        plan pins the piece order. ``beats`` must be supplied (or
        detected by the caller) when ``cut_rule`` is ``beats`` — a
        missing grid fails closed here, never a silent uncut edit.
        """
        paths = [str(p) for p in (footage or [])]
        if not paths:
            raise MediaEditError("build_spec needs at least one footage file")
        for p in paths:
            if not Path(p).exists():
                raise MediaEditError(f"no such footage: {p}")
        probe = video_probe(paths[0])
        duration = float(probe.get("duration") or 0.0)
        if duration <= 0:
            raise MediaEditError(
                f"could not probe a duration for {paths[0]} — "
                "the footage is unreadable")
        grid = self._cut_grid(duration, beats)
        pieces = self._pieces(paths, duration, grid)
        caption_text = caption if caption is not None \
            else str(self.caption.get("text", "") or "")
        spec = build_style_spec(
            self.base_style,
            clips=pieces,
            beats=grid if grid else None,
            music=music,
            captions=caption_text or None,
            transition=self.transitions,
            caption_style=str(self.caption.get("style", "hormozi")),
            effect_cycle=tuple(self.effect_cycle),
            width=int(self.canvas["width"]), height=int(self.canvas["height"]),
            fps=int(self.canvas["fps"]), out=out, seed=seed, crf=crf)
        spec.segments = [(i, piece.play_duration())
                         for i, piece in enumerate(pieces)]
        spec.cut_points = list(grid) if grid else None
        return spec

    def _cut_grid(self, duration: float,
                  beats: list[float] | None) -> list[float]:
        if self.cut_rule == "none":
            return []
        if self.cut_rule == "interval":
            step = float(self.interval_s)
            grid = []
            t = step
            while t < duration - 0.05:
                grid.append(round(t, 3))
                t += step
            return grid
        # beats
        grid = sorted({round(float(b), 3) for b in (beats or [])
                       if 0.0 < float(b) < duration})
        if not grid:
            raise MediaEditError(
                "cut_rule is beats but no beat grid was supplied — "
                "pass beats= explicitly or a music file to detect them from")
        return grid

    def _pieces(self, paths: list[str], duration: float,
                grid: list[float]) -> list[Clip]:
        """Atomic pieces: split at every grid boundary, speed-stepped.

        Boundaries live in timeline time; each piece consumes
        ``interval * speed`` seconds of source, so the piece's play
        duration is exactly its grid interval. The source cursor
        advances through the first footage file (extra files append
        after it, same grammar).
        """
        bounds = [0.0] + list(grid) + [duration]
        bounds = sorted({round(b, 3) for b in bounds})
        clips: list[Clip] = []
        src_cursor = 0.0
        cycle = list(self.effect_cycle)
        for i in range(len(bounds) - 1):
            t0, t1 = bounds[i], bounds[i + 1]
            play = t1 - t0
            if play < 0.05:
                continue
            mid = (t0 + t1) / 2.0 / duration
            speed = self._speed_at(mid)
            src_len = play * speed
            clips.append(Clip(
                path=paths[0],
                start=round(src_cursor, 3),
                end=round(min(src_cursor + src_len, duration), 3),
                speed=speed,
                effects=(Effect(name=cycle[i % len(cycle)]),)))
            src_cursor += src_len
            if src_cursor >= duration - 0.01:
                break
        for extra in paths[1:]:
            clips.append(Clip(path=extra, effects=(
                Effect(name=cycle[len(clips) % len(cycle)]),)))
        if not clips:
            raise MediaEditError("no pieces — the cut grid ate the footage")
        return clips
