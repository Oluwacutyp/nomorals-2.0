"""Breakdown — scenes into addressable shots.

The deterministic core follows how an AD and a DP actually break a
scene down:

- every new location opens on an **establishing** shot (a scene that
  continues the previous location/time does not re-establish);
- each action block is a shot — long blocks split at sentence
  boundaries into beats a camera can actually hold;
- each dialogue line is a shot of its speaker; in a sustained
  exchange, a **reaction** shot of the listener lands after every
  second reply, because coverage is shot in pairs, not singles;
- parentheticals ride their dialogue shot as performance notes;
- transitions close the scene as their own beat.

Prose (no sluglines) takes the brain-assisted path: the brain
segments the text into scenes and beats against a JSON contract,
and the same shot grammar is built from its answer. No brain, no
segmentation — an honest :class:`DirectorError`, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.logging_setup import get_logger
from .design import DesignSpec
from .errors import DirectorError
from .screenplay import Element, Scene, Screenplay

_log = get_logger(__name__)

__all__ = ["Shot", "break_down", "break_down_prose", "SHOT_KINDS"]

SHOT_KINDS = ("establishing", "action", "dialogue", "reaction",
              "detail", "transition")


@dataclass
class Shot:
    id: str                       # "1.1" — scene.shot, addressable
    scene_number: int
    index: int                    # within the scene
    kind: str                     # one of SHOT_KINDS
    description: str              # what the camera sees
    characters: list[str] = field(default_factory=list)
    speaker: str = ""             # dialogue shots: who speaks
    dialogue: str = ""            # dialogue shots: the line
    performance_note: str = ""    # parenthetical, when present
    duration_s: float = 4.0
    design: DesignSpec | None = None
    # production state (filled by the shoot/QA stages)
    status: str = "planned"       # planned|generated|qa_pass|qa_fail|
                                  # qa_unavailable|failed|regenerated
    clip_path: str = ""
    engine: str = ""
    qa: dict[str, Any] = field(default_factory=dict)
    regen_count: int = 0

    @property
    def subject(self) -> str:
        """The frame's expected subject, for QA and prompt assembly."""
        if self.characters:
            return ", ".join(self.characters)
        return self.description

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "scene_number": self.scene_number,
            "index": self.index, "kind": self.kind,
            "description": self.description,
            "characters": list(self.characters), "speaker": self.speaker,
            "dialogue": self.dialogue,
            "performance_note": self.performance_note,
            "duration_s": self.duration_s,
            "design": self.design.to_dict() if self.design else None,
            "status": self.status, "clip_path": self.clip_path,
            "engine": self.engine, "qa": dict(self.qa),
            "regen_count": self.regen_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Shot":
        design = d.get("design")
        return cls(
            id=str(d.get("id", "")), scene_number=int(d.get("scene_number", 0)),
            index=int(d.get("index", 0)), kind=str(d.get("kind", "action")),
            description=str(d.get("description", "")),
            characters=[str(c) for c in d.get("characters", [])],
            speaker=str(d.get("speaker", "")),
            dialogue=str(d.get("dialogue", "")),
            performance_note=str(d.get("performance_note", "")),
            duration_s=float(d.get("duration_s", 4.0)),
            design=DesignSpec.from_dict(design)
            if isinstance(design, dict) else None,
            status=str(d.get("status", "planned")),
            clip_path=str(d.get("clip_path", "")),
            engine=str(d.get("engine", "")),
            qa=dict(d.get("qa") or {}),
            regen_count=int(d.get("regen_count", 0)),
        )


def _words(text: str) -> int:
    return len((text or "").split())


def _dialogue_duration(line: str) -> float:
    # screen dialogue plays at roughly 2.2 words/second with breath
    return min(9.0, max(2.5, _words(line) / 2.2 + 0.6))


def _action_duration(text: str) -> float:
    return min(8.0, max(3.0, 2.5 + _words(text) / 18.0))


def _split_action(text: str) -> list[str]:
    """Split a long action block into camera-holdable beats."""
    import re
    if _words(text) <= 42:
        return [text.strip()]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    beats: list[str] = []
    buf = ""
    for s in sentences:
        if buf and _words(buf) + _words(s) > 42:
            beats.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf.strip():
        beats.append(buf.strip())
    return beats or [text.strip()]


def break_down(play: Screenplay) -> list[Shot]:
    """Deterministic breakdown of a parsed screenplay into shots."""
    shots: list[Shot] = []
    prev_key: tuple[str, str] | None = None
    for scene in play.scenes:
        scene_shots: list[Shot] = []

        def add(kind: str, description: str, **kw: Any) -> Shot:
            shot = Shot(id=f"{scene.number}.{len(scene_shots) + 1}",
                        scene_number=scene.number,
                        index=len(scene_shots) + 1, kind=kind,
                        description=description, **kw)
            scene_shots.append(shot)
            return shot

        key = (scene.location.upper(), scene.time_of_day.upper())
        continues = prev_key is not None and key == prev_key
        prev_key = key
        if not continues:
            add("establishing",
                f"Establish {scene.location or 'the location'}"
                + (f" ({scene.time_of_day})" if scene.time_of_day else ""),
                characters=[], duration_s=4.0)

        exchange = 0            # consecutive cross-speaker replies
        last_speaker = ""
        pending_paren = ""
        for el in scene.elements:
            if el.kind == "heading":
                continue
            if el.kind == "parenthetical":
                pending_paren = el.text
                continue
            if el.kind == "transition":
                add("transition", el.text, duration_s=1.5)
                continue
            if el.kind == "dialogue":
                if last_speaker and el.character != last_speaker:
                    exchange += 1
                else:
                    exchange = 0
                add("dialogue",
                    f"{el.character} speaks"
                    + (f" — {pending_paren}" if pending_paren else ""),
                    characters=[el.character], speaker=el.character,
                    dialogue=el.text, performance_note=pending_paren,
                    duration_s=_dialogue_duration(el.text))
                pending_paren = ""
                if exchange >= 2 and last_speaker:
                    add("reaction",
                        f"{last_speaker} reacts, listening",
                        characters=[last_speaker], duration_s=2.0)
                    exchange = 0
                last_speaker = el.character
                continue
            # action
            for beat in _split_action(el.text):
                add("action", beat,
                    characters=list(scene.characters),
                    duration_s=_action_duration(beat))
        shots.extend(scene_shots)
    return shots


# ---------------------------------------------------------------------------
# Brain-assisted path for prose (no sluglines)
# ---------------------------------------------------------------------------

def break_down_prose(text: str, *, brain: Any,
                     title: str = "") -> tuple[Screenplay, list[Shot]]:
    """Segment prose into scenes/beats with the brain, then break down.

    The brain's answer must satisfy the segmentation contract; the
    scenes it returns are rebuilt as a real :class:`Screenplay` and
    pass through the same deterministic shot grammar as a parsed
    script, so both paths produce one kind of production.
    """
    from ...llm.base import Message

    if brain is None:
        raise DirectorError(
            "this text has no screenplay sluglines — segmenting prose "
            "into scenes needs the brain, and no brain is configured")
    if not (text or "").strip():
        raise DirectorError("nothing to break down — the text is empty")
    prompt = (
        "Segment this prose story/script into scenes for a film "
        "breakdown. Reply with ONLY JSON of exactly this shape:\n"
        '{"scenes": [{"location": str, "time_of_day": str, '
        '"interior": "INT"|"EXT", "beats": [{"kind": '
        '"action"|"dialogue", "text": str, "speaker": str}]}]}\n'
        "One scene per distinct location/time. Keep beat text faithful "
        "to the source — segment, do not rewrite.\n\n" + text[:30000])
    try:
        data, resp = brain.chat_json([Message.user(prompt)],
                                     task_kind="breakdown")
    except Exception as exc:  # noqa: BLE001
        raise DirectorError(f"prose breakdown failed: {exc}") from exc
    if not isinstance(data, dict) or not data.get("scenes"):
        err = getattr(resp, "error", "") or "no scenes in reply"
        raise DirectorError(
            f"prose breakdown failed: the brain returned no usable "
            f"segmentation ({err})")
    play = Screenplay(title=title, source_chars=len(text))
    for i, sdata in enumerate(data["scenes"], 1):
        if not isinstance(sdata, dict):
            continue
        location = str(sdata.get("location") or "UNSPECIFIED")
        tod = str(sdata.get("time_of_day") or "")
        interior = str(sdata.get("interior") or "INT").upper()
        heading = f"{interior}. {location}" + (f" - {tod}" if tod else "")
        scene = Scene(number=i, heading=heading, interior=interior,
                      location=location, time_of_day=tod)
        scene.elements.append(Element("heading", heading))
        for beat in sdata.get("beats") or []:
            if not isinstance(beat, dict):
                continue
            btext = str(beat.get("text") or "").strip()
            if not btext:
                continue
            if str(beat.get("kind")) == "dialogue":
                scene.elements.append(Element(
                    "dialogue", btext,
                    character=str(beat.get("speaker") or "UNKNOWN")))
            else:
                scene.elements.append(Element("action", btext))
        play.scenes.append(scene)
    if not play.scenes:
        raise DirectorError(
            "prose breakdown failed: segmentation produced no scenes")
    return play, break_down(play)
