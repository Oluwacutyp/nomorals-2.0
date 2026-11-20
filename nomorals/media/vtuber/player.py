"""The act player — a scripted performance, stepped by the show.

An ActPlayer binds an Act + a Character to one Show. The show's
own clock drives it: every live tick the plugin calls
:meth:`ActPlayer.tick`, which fires the beats whose time has come
and holds the active overlays. A beat that fails (a take file that
is not there, a live capture holding the channel) is named in the
show's journal and the act goes on — one bad beat never kills a
performance.

Overlays (what the plugin applies each tick):

* ``say`` — the active line, held for its duration; the plugin
  writes it into the face channel, where the rig's text→viseme
  path speaks it.
* ``expression`` — the active expression's blendshape weights,
  held for its duration; the plugin merges them over the posed
  frame (the act wins on the keys it names, the take's other
  blendshapes survive).
* ``gesture`` — the active gesture's bone rotations, held for its
  duration; merged the same way.
* ``camera`` — the active shot, persistent until the next camera
  beat or cue.

Audience cues (:meth:`ActPlayer.cue`) are immediate overlay beats
from outside the script: ``say`` / ``expression`` / ``gesture`` /
``camera``. A cue with an unknown kind or value is refused by
name — the audience never silently does nothing.
"""

from __future__ import annotations

from typing import Any

from ..rig.rig import RigDefinition
from .act import Act
from .camera import SHOTS
from .character import Character
from .errors import VtuberError

__all__ = [
    "ActPlayer",
]

CUE_KINDS = ("say", "expression", "gesture", "camera")


def _say_duration(text: str) -> float:
    """How long a line holds: at least a second, longer for more
    words. The rig speaks on uniform timing, so this is the honest
    clock for it — not forced alignment."""
    words = len(text.split())
    return max(1.0, 0.15 * words)


class ActPlayer:
    """One act, performed on one show."""

    def __init__(self, act: Act, character: Character,
                 rig: RigDefinition) -> None:
        if act.character != character.name:
            raise VtuberError(
                f"act {act.name!r} is for character "
                f"{act.character!r}, not {character.name!r}",
                kind="bad_character")
        # Every beat name resolves against the character now —
        # a bad name is refused at attach, never mid-performance.
        # And every preset the character carries is checked against
        # the rig this show actually drives — a character loaded
        # against one rig must not silently mis-drive another.
        blendshapes = set(rig.blendshapes)
        bones = {bone.name for bone in rig.bones}
        for expr_name, weights in character.expressions.items():
            for blend in weights:
                if blend not in blendshapes:
                    raise VtuberError(
                        f"character {character.name!r} expression "
                        f"{expr_name!r} names blendshape {blend!r}, "
                        f"which is not in rig {rig.name!r}",
                        kind="bad_blendshape")
        for gest_name, rotations in character.gestures.items():
            for bone in rotations:
                if bone not in bones:
                    raise VtuberError(
                        f"character {character.name!r} gesture "
                        f"{gest_name!r} names bone {bone!r}, which "
                        f"is not in rig {rig.name!r}",
                        kind="bad_bone")
        for beat in act.beats:
            if beat.expression is not None:
                character.expression(beat.expression)
            if beat.gesture is not None:
                character.gesture(beat.gesture)
            if beat.camera is not None and beat.camera not in SHOTS:
                raise VtuberError(
                    f"act {act.name!r} beat at {beat.at}s names "
                    f"shot {beat.camera!r} — shots are "
                    f"{', '.join(sorted(SHOTS))}",
                    kind="unknown_shot")
        self.act = act
        self.character = character
        self.rig = rig
        self._ticks = 0
        self._next_beat = 0
        self._beats_fired = 0
        self._cues_applied = 0
        self._attached = True
        self._now_s = 0.0
        self._shot = "wide"
        self._say: dict[str, Any] | None = None
        self._expression: dict[str, Any] | None = None
        self._gesture: dict[str, Any] | None = None

    # -- the clock ---------------------------------------------------

    def tick(self, show: Any) -> dict[str, Any]:
        """One live tick: fire due beats, expire old overlays."""
        if not self._attached:
            return {"fired": [], "now_s": self._now_s}
        # The clock is the show's: beats are timed in seconds.
        fps = float(getattr(show, "fps", 15.0) or 15.0)
        now_s = self._ticks / fps
        self._now_s = now_s
        fired: list[dict[str, Any]] = []
        while (self._next_beat < len(self.act.beats)
               and self.act.beats[self._next_beat].at <= now_s):
            beat = self.act.beats[self._next_beat]
            self._next_beat += 1
            self._beats_fired += 1
            fired.append(self._fire(show, beat, now_s))
        self._expire(now_s)
        self._ticks += 1
        return {"fired": fired, "now_s": now_s}

    @property
    def now_s(self) -> float:
        """Seconds into the act, on the show's clock."""
        return self._now_s

    def _expire(self, now_s: float) -> None:
        """Drop overlays whose duration is over."""
        for slot in ("_say", "_expression", "_gesture"):
            overlay = getattr(self, slot)
            if overlay is not None and now_s >= overlay["until"]:
                setattr(self, slot, None)

    # -- beats ---------------------------------------------------------

    def _fire(self, show: Any, beat: Any,
              now_s: float) -> dict[str, Any]:
        """Fire one beat: apply it, name it once in the journal.
        One beat is one journal event — the kind is the beat's
        first action (say, expression, gesture, camera, take,
        face_take) — so the journal reads as the act, beat by
        beat. A beat whose take cannot load is named as failed
        and the act goes on."""
        fired: dict[str, Any] = {"at": beat.at}
        failed: list[str] = []
        if beat.say is not None:
            self._say = {"text": beat.say,
                         "until": now_s + _say_duration(beat.say)}
            fired["say"] = beat.say
        if beat.expression is not None:
            weights = self.character.expression(beat.expression)
            self._expression = {
                "name": beat.expression, "weights": weights,
                "until": now_s + beat.duration_s}
            fired["expression"] = beat.expression
        if beat.gesture is not None:
            rotations = self.character.gesture(beat.gesture)
            self._gesture = {
                "name": beat.gesture, "rotations": rotations,
                "until": now_s + beat.duration_s}
            fired["gesture"] = beat.gesture
        if beat.camera is not None:
            self._shot = beat.camera
            fired["camera"] = beat.camera
        if beat.take is not None:
            if getattr(show, "_hold_take", False):
                failed.append(
                    f"live capture holds the take ({beat.take})")
                fired["take_failed"] = beat.take
            else:
                try:
                    show.load_take(beat.take)
                    fired["take"] = beat.take
                except Exception as exc:  # noqa: BLE001 - beat fails named
                    failed.append(str(exc))
                    fired["take_failed"] = beat.take
        if beat.face_take is not None:
            try:
                show.load_face_take(beat.face_take)
                fired["face_take"] = beat.face_take
            except Exception as exc:  # noqa: BLE001 - beat fails named
                failed.append(str(exc))
                fired["face_take_failed"] = beat.face_take
        if failed:
            show._record("vtuber_beat_failed",
                         {"at": beat.at, "beat": fired,
                          "failed": failed})
        else:
            primary = next(
                field for field in
                ("say", "expression", "gesture", "camera", "take",
                 "face_take")
                if fired.get(field) is not None)
            show._record(f"vtuber_beat_{primary}",
                         {"at": beat.at, "beat": fired})
        return fired

    # -- audience cues ---------------------------------------------------

    def cue(self, kind: str, value: str,
            duration_s: float | None = None) -> dict[str, Any]:
        """An audience cue — an immediate overlay beat. Refused
        by name when the kind or the value is unknown."""
        kind = (kind or "").strip().lower()
        value = (value or "").strip()
        if kind not in CUE_KINDS:
            raise VtuberError(
                f"unknown cue kind {kind!r} — cues are "
                f"{', '.join(CUE_KINDS)}",
                kind="bad_cue")
        if not value:
            raise VtuberError(f"cue {kind!r} needs a value",
                              kind="bad_cue")
        now_s = self.now_s
        if kind == "say":
            self._say = {"text": value,
                         "until": now_s + _say_duration(value)}
        elif kind == "expression":
            weights = self.character.expression(value)
            hold = duration_s if duration_s else 2.0
            self._expression = {"name": value, "weights": weights,
                                "until": now_s + hold}
        elif kind == "gesture":
            rotations = self.character.gesture(value)
            hold = duration_s if duration_s else 2.0
            self._gesture = {"name": value, "rotations": rotations,
                             "until": now_s + hold}
        elif kind == "camera":
            if value not in SHOTS:
                raise VtuberError(
                    f"unknown shot {value!r} — shots are "
                    f"{', '.join(sorted(SHOTS))}",
                    kind="unknown_shot")
            self._shot = value
        self._cues_applied += 1
        return {"cue": kind, "value": value,
                "cues_applied": self._cues_applied}

    # -- what the plugin applies ------------------------------------------

    def active_face(self) -> tuple[dict[str, float], str | None]:
        """The face overlay now: (blendshapes, spoken text)."""
        blend: dict[str, float] = {}
        if self._expression is not None:
            blend.update(self._expression["weights"])
        text = self._say["text"] if self._say is not None else None
        return blend, text

    def active_pose(self) -> tuple[dict[str, float],
                                   dict[str, tuple[float, float, float]]]:
        """The pose overlay now: (blendshapes, bone rotations)."""
        blend: dict[str, float] = {}
        if self._expression is not None:
            blend.update(self._expression["weights"])
        rotations: dict[str, tuple[float, float, float]] = {}
        if self._gesture is not None:
            rotations.update(self._gesture["rotations"])
        return blend, rotations

    @property
    def current_shot(self) -> str:
        """The active shot."""
        return self._shot

    def detach(self) -> None:
        """Stop performing — later ticks are no-ops."""
        self._attached = False
        self._say = None
        self._expression = None
        self._gesture = None

    def state(self) -> dict[str, Any]:
        """The performance, as data."""
        return {
            "act": self.act.name,
            "character": self.character.name,
            "attached": self._attached,
            "now_s": round(self.now_s, 3),
            "beats_fired": self._beats_fired,
            "beats_total": len(self.act.beats),
            "next_beat_at": (self.act.beats[self._next_beat].at
                             if self._next_beat < len(self.act.beats)
                             else None),
            "current_shot": self._shot,
            "saying": self._say["text"] if self._say else None,
            "expression": (self._expression["name"]
                           if self._expression else None),
            "gesture": (self._gesture["name"]
                        if self._gesture else None),
            "cues_applied": self._cues_applied,
        }
