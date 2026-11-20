"""A VTuber act — a scripted performance, timed to a clock.

An act is data: a name, the character that performs it, the fps
its beats are timed against, and beats sorted by time. A beat is
one cue at one moment:

* ``say`` — a line of dialogue; the rig's text→viseme path speaks
  it (uniform timing, as the driver documents).
* ``expression`` — a named expression from the character, held for
  ``duration_s`` (default 2.0s).
* ``gesture`` — a named gesture from the character, held for
  ``duration_s`` (default 2.0s).
* ``camera`` — a shot name; the active shot until the next camera
  beat.
* ``take`` — a body take path (.bvh/.json) loaded into the show.
* ``face_take`` — a face take path (manifest/.morph_anim.json/.csv)
  loaded into the show.

A beat with nothing to do is refused at load — a beat that fires
and does nothing is a silent no-op, and this organ does not do
silent. Times are non-negative seconds; beats arrive sorted by
``at`` (stable) whatever order the file lists them in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import VtuberError

__all__ = [
    "Act",
    "Beat",
]

#: The fields a beat may carry. Anything else in a beat is refused
#: by name — a typo in a beat must not silently become nothing.
BEAT_FIELDS = frozenset({
    "at", "say", "expression", "gesture", "camera", "take",
    "face_take", "duration_s",
})

#: The fields that do something on a beat. A beat with none of
#: these is refused.
BEAT_ACTIONS = frozenset({
    "say", "expression", "gesture", "camera", "take", "face_take",
})

DEFAULT_HOLD_S = 2.0


@dataclass(frozen=True)
class Beat:
    """One cue at one moment of the act."""

    at: float
    say: str | None = None
    expression: str | None = None
    gesture: str | None = None
    camera: str | None = None
    take: str | None = None
    face_take: str | None = None
    duration_s: float = DEFAULT_HOLD_S

    def to_dict(self) -> dict[str, Any]:
        """The beat as plain JSON-able data (empty fields out)."""
        data: dict[str, Any] = {"at": self.at}
        if self.say is not None:
            data["say"] = self.say
        if self.expression is not None:
            data["expression"] = self.expression
        if self.gesture is not None:
            data["gesture"] = self.gesture
        if self.camera is not None:
            data["camera"] = self.camera
        if self.take is not None:
            data["take"] = self.take
        if self.face_take is not None:
            data["face_take"] = self.face_take
        if (self.expression is not None or self.gesture is not None
                or self.say is not None):
            data["duration_s"] = self.duration_s
        return data


@dataclass
class Act:
    """A scripted performance: beats on a clock."""

    name: str
    character: str
    fps: float = 15.0
    beats: list[Beat] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        if not self.name:
            raise VtuberError("act name must not be empty",
                              kind="bad_act")
        self.character = (self.character or "").strip()
        if not self.character:
            raise VtuberError(
                f"act {self.name!r} names no character",
                kind="bad_act")
        if self.fps <= 0:
            raise VtuberError(
                f"act {self.name!r} fps must be positive, got "
                f"{self.fps}",
                kind="bad_act")
        self.beats = sorted(self.beats, key=lambda b: b.at)

    # -- serialisation ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The act as plain JSON-able data."""
        return {
            "name": self.name,
            "character": self.character,
            "fps": self.fps,
            "beats": [beat.to_dict() for beat in self.beats],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Act:
        """Build an act from data, validated."""
        if not isinstance(data, dict):
            raise VtuberError(
                f"act must be a mapping, got {type(data).__name__}",
                kind="bad_act")
        raw_beats = data.get("beats")
        if not isinstance(raw_beats, list):
            raise VtuberError(
                f"act {data.get('name')!r} needs a 'beats' list",
                kind="bad_act")
        beats: list[Beat] = []
        for index, raw in enumerate(raw_beats):
            beats.append(_beat_from_dict(index, raw))
        try:
            fps = float(data.get("fps") or 15.0)
        except (TypeError, ValueError) as exc:
            raise VtuberError(
                f"act {data.get('name')!r} fps must be a number: "
                f"{exc}",
                kind="bad_act") from exc
        return cls(
            name=str(data.get("name") or ""),
            character=str(data.get("character") or ""),
            fps=fps,
            beats=beats)

    @classmethod
    def from_file(cls, path: str | Path) -> Act:
        """Load an act from a JSON file (``.vtact.json``)."""
        target = Path(path).expanduser()
        if not target.exists():
            raise VtuberError(f"no act file at {target}",
                              kind="missing_act")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise VtuberError(
                f"act file at {target} is unreadable: {exc}",
                kind="bad_act") from exc
        return cls.from_dict(data)


def _beat_from_dict(index: int, raw: Any) -> Beat:
    """One beat, validated."""
    if not isinstance(raw, dict):
        raise VtuberError(
            f"beat {index} must be a mapping, got "
            f"{type(raw).__name__}",
            kind="bad_beat")
    unknown = set(raw) - BEAT_FIELDS
    if unknown:
        raise VtuberError(
            f"beat {index} has unknown fields "
            f"{', '.join(sorted(unknown))} — a beat carries "
            f"{', '.join(sorted(BEAT_FIELDS))}",
            kind="bad_beat")
    if not (set(raw) & BEAT_ACTIONS):
        raise VtuberError(
            f"beat {index} does nothing — a beat needs one of "
            f"{', '.join(sorted(BEAT_ACTIONS))}",
            kind="bad_beat")
    try:
        at = float(raw.get("at", 0.0))
    except (TypeError, ValueError) as exc:
        raise VtuberError(
            f"beat {index} 'at' must be a number: {exc}",
            kind="bad_beat") from exc
    if at < 0:
        raise VtuberError(
            f"beat {index} 'at' must be >= 0, got {at}",
            kind="bad_beat")
    duration = DEFAULT_HOLD_S
    if raw.get("duration_s") is not None:
        try:
            duration = float(raw["duration_s"])
        except (TypeError, ValueError) as exc:
            raise VtuberError(
                f"beat {index} 'duration_s' must be a number: "
                f"{exc}",
                kind="bad_beat") from exc
        if duration <= 0:
            raise VtuberError(
                f"beat {index} 'duration_s' must be positive, "
                f"got {duration}",
                kind="bad_beat")
    say = raw.get("say")
    if say is not None and not str(say).strip():
        raise VtuberError(f"beat {index} 'say' must not be empty",
                          kind="bad_beat")
    return Beat(
        at=at,
        say=(str(say).strip() if say is not None else None),
        expression=(str(raw["expression"]).strip()
                    if raw.get("expression") is not None else None),
        gesture=(str(raw["gesture"]).strip()
                 if raw.get("gesture") is not None else None),
        camera=(str(raw["camera"]).strip()
                if raw.get("camera") is not None else None),
        take=(str(raw["take"]).strip()
              if raw.get("take") is not None else None),
        face_take=(str(raw["face_take"]).strip()
                   if raw.get("face_take") is not None else None),
        duration_s=duration)
