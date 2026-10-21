"""Screenplay parsing — standard format in, structured scenes out.

A real parser for the format working screenplays are written in:

- sluglines: ``INT.`` / ``EXT.`` / ``INT./EXT.`` / ``EST.`` headings
  (location + time of day split on the last `` - ``);
- action blocks (description prose);
- character cues (an ALL-CAPS line) followed by dialogue, with
  parentheticals attached to the line they precede;
- transitions (``CUT TO:``, ``FADE OUT.`` …) kept as their own elements.

Nothing here invents structure: text with no sluglines parses to a
:class:`Screenplay` with zero scenes and ``is_standard == False`` —
the caller decides whether to take the brain-assisted prose path
(:mod:`nomorals.media.director.breakdown`) or refuse honestly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Element",
    "Scene",
    "Screenplay",
    "parse_screenplay",
    "looks_like_screenplay",
]

_SLUG_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|INT\.\\EXT\.|EST\.)\s+(.+)$")
_TRANSITION_RE = re.compile(
    r"^([A-Z][A-Z ]{2,}:|FADE (IN|OUT)\.?|DISSOLVE TO:?)$")
# a character cue: short, mostly caps, may carry (V.O.) / (O.S.) / (CONT'D)
_CUE_RE = re.compile(
    r"^[A-Z0-9][A-Z0-9 .'\-&]{0,38}(\s*\((V\.O\.|O\.S\.|CONT'D|CONT’D)\))?$")


@dataclass
class Element:
    """One block of a scene.

    ``kind``: heading | action | dialogue | parenthetical | transition.
    Dialogue elements carry ``character`` (the cue, extension stripped).
    """

    kind: str
    text: str
    character: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text,
                "character": self.character}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Element":
        return cls(kind=str(d.get("kind", "action")),
                   text=str(d.get("text", "")),
                   character=str(d.get("character", "")))


@dataclass
class Scene:
    number: int
    heading: str
    interior: str = ""          # INT | EXT | INT/EXT | EST | ""
    location: str = ""
    time_of_day: str = ""
    elements: list[Element] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"SC{self.number:02d}"

    @property
    def characters(self) -> list[str]:
        """Speaking characters, in first-appearance order."""
        out: list[str] = []
        for el in self.elements:
            if el.kind == "dialogue" and el.character \
                    and el.character not in out:
                out.append(el.character)
        return out

    @property
    def action_text(self) -> str:
        return "\n".join(el.text for el in self.elements
                         if el.kind == "action")

    @property
    def dialogue(self) -> list[Element]:
        return [el for el in self.elements if el.kind == "dialogue"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "heading": self.heading,
            "interior": self.interior, "location": self.location,
            "time_of_day": self.time_of_day,
            "elements": [e.to_dict() for e in self.elements],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        return cls(
            number=int(d.get("number", 0)),
            heading=str(d.get("heading", "")),
            interior=str(d.get("interior", "")),
            location=str(d.get("location", "")),
            time_of_day=str(d.get("time_of_day", "")),
            elements=[Element.from_dict(e)
                      for e in d.get("elements", [])],
        )


@dataclass
class Screenplay:
    title: str = ""
    scenes: list[Scene] = field(default_factory=list)
    source_chars: int = 0

    @property
    def is_standard(self) -> bool:
        return bool(self.scenes)

    @property
    def characters(self) -> list[str]:
        out: list[str] = []
        for sc in self.scenes:
            for name in sc.characters:
                if name not in out:
                    out.append(name)
        return out

    def outline(self) -> str:
        """One line per scene — the map a coverage reader starts from."""
        lines = []
        for sc in self.scenes:
            chars = ", ".join(sc.characters) or "—"
            lines.append(f"{sc.id}  {sc.heading}  [{chars}]")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "source_chars": self.source_chars,
                "scenes": [s.to_dict() for s in self.scenes]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Screenplay":
        return cls(title=str(d.get("title", "")),
                   source_chars=int(d.get("source_chars", 0)),
                   scenes=[Scene.from_dict(s)
                           for s in d.get("scenes", [])])


def looks_like_screenplay(text: str) -> bool:
    """True when the text carries at least one slugline."""
    return any(_SLUG_RE.match(line.strip())
               for line in (text or "").splitlines())


def _split_heading(heading_body: str) -> tuple[str, str]:
    """``COFFEE SHOP - DAY`` → (``COFFEE SHOP``, ``DAY``)."""
    if " - " in heading_body:
        loc, tod = heading_body.rsplit(" - ", 1)
        return loc.strip(), tod.strip()
    return heading_body.strip(), ""


def _cue_character(line: str) -> str | None:
    """The character name if ``line`` is a dialogue cue, else None."""
    if len(line) > 42 or not line:
        return None
    if line != line.upper():
        return None
    if not _CUE_RE.match(line):
        return None
    # ALL-CAPS action lines ("THE DOOR SLAMS SHUT.") are not cues; a cue
    # is short and name-like (no sentence punctuation inside).
    if line.endswith(".") and " " in line.strip():
        return None
    return re.sub(r"\s*\((V\.O\.|O\.S\.|CONT'D|CONT’D)\)\s*$", "",
                  line).strip()


def parse_screenplay(text: str, *, title: str = "") -> Screenplay:
    """Parse screenplay text into scenes and elements.

    Blocks are separated by blank lines (the screenplay convention).
    A dialogue cue binds to the block that follows it; a parenthetical
    block between cue and dialogue attaches to that dialogue.
    """
    raw = text or ""
    play = Screenplay(title=title, source_chars=len(raw))
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", raw):
        chunk = chunk.strip("\n")
        if chunk.strip():
            blocks.append(chunk)

    scene: Scene | None = None

    def _flush_scene() -> None:
        nonlocal scene
        if scene is not None:
            play.scenes.append(scene)
            scene = None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            i += 1
            continue
        first = lines[0].strip()
        m = _SLUG_RE.match(first)
        if m:
            _flush_scene()
            prefix = m.group(1).replace("\\", "")
            interior = {"INT.": "INT", "EXT.": "EXT",
                        "INT/EXT.": "INT/EXT", "INT.EXT.": "INT/EXT",
                        "EST.": "EST"}.get(prefix, prefix.rstrip("."))
            location, tod = _split_heading(m.group(2).strip())
            heading = f"{prefix} {m.group(2).strip()}".strip()
            scene = Scene(number=len(play.scenes) + 1, heading=heading,
                          interior=interior, location=location,
                          time_of_day=tod)
            scene.elements.append(Element("heading", heading))
            rest = "\n".join(lines[1:]).strip()
            if rest:
                scene.elements.append(Element("action", rest))
            i += 1
            continue
        if scene is None:
            # front matter / title page: only a plausible title is kept
            if not play.title and len(first) <= 80:
                play.title = first.title() if first.isupper() else first
            i += 1
            continue
        whole = "\n".join(lines).strip()
        if _TRANSITION_RE.match(first) and len(lines) == 1:
            scene.elements.append(Element("transition", first))
            i += 1
            continue
        cue = _cue_character(first) if len(lines) >= 1 else None
        if cue is not None and len(lines) >= 2:
            # cue block: optional parenthetical lines, then dialogue
            paren = ""
            spoken: list[str] = []
            for ln in lines[1:]:
                s = ln.strip()
                if s.startswith("(") and s.endswith(")") and not spoken:
                    paren = s
                else:
                    spoken.append(s)
            if paren:
                scene.elements.append(Element("parenthetical", paren,
                                              character=cue))
            if spoken:
                scene.elements.append(Element(
                    "dialogue", " ".join(spoken), character=cue))
                i += 1
                continue
        if cue is not None and i + 1 < len(blocks):
            # cue alone in its block; the dialogue is the next block
            nxt = [ln.strip() for ln in blocks[i + 1].splitlines()
                   if ln.strip()]
            if nxt and not _SLUG_RE.match(nxt[0]):
                paren = ""
                spoken = []
                for s in nxt:
                    if s.startswith("(") and s.endswith(")") \
                            and not spoken:
                        paren = s
                    else:
                        spoken.append(s)
                if spoken:
                    if paren:
                        scene.elements.append(Element(
                            "parenthetical", paren, character=cue))
                    scene.elements.append(Element(
                        "dialogue", " ".join(spoken), character=cue))
                    i += 2
                    continue
        scene.elements.append(Element("action", whole))
        i += 1
    _flush_scene()
    return play
