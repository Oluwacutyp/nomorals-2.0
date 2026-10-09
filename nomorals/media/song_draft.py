"""The artist's notebook — song drafting before performance.

An artist doesn't sing straight from a topic. They draft: lyrics with real
cadence, a melody contour that fits the words, a structure with an emotional
arc, and notes to themselves about what the song is *about*. This module is
that notebook.

- :func:`draft_song` — the LLM writes a full song draft as one creative act
  (lyrics + cadence + melody contour + emotional arc + artist notes).
- :func:`revise_draft` — natural-language revision ("make the chorus hit
  harder") against the active draft.
- :func:`render_notebook` — the draft as a readable notebook page.
- :func:`save_draft` / :func:`load_draft` — draft persistence (JSON).

The draft is a real creative artifact, not metadata. The perform phase
(:mod:`nomorals.media.vocal_lite`) reads the draft's melody contours and
energy arc so beat and vocals are shaped together from the draft.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

DRAFTS_DIR = "drafts"


# ── data ────────────────────────────────────────────────────────────────

@dataclass
class DraftLine:
    """One lyric line with its musical DNA."""
    text: str = ""
    syllables: int = 0
    # melody contour: per-syllable-ish moves, e.g. ["up","up","down","hold"]
    contour: list[str] = field(default_factory=list)
    rhyme: str = ""  # rhyme group label, e.g. "A", "B"


@dataclass
class DraftSection:
    """A song section: verse, pre-chorus, chorus, bridge, ..."""
    name: str = ""
    bars: int = 4
    energy: float = 0.5  # 0.0 whisper → 1.0 full belt
    purpose: str = ""    # what this section DOES in the song
    lines: list[DraftLine] = field(default_factory=list)


@dataclass
class SongDraft:
    """The artist's notebook page for one song."""
    title: str = ""
    topic: str = ""
    style: str = "pop"
    tempo: int = 100
    key: str = "C"
    mode: str = "major"
    about: str = ""          # what the song is really about (1-2 sentences)
    sections: list[DraftSection] = field(default_factory=list)
    rhyme_scheme: str = ""
    # the artist showing their work:
    why_melody_fits: str = ""   # why this melody fits these words
    energy_arc: str = ""        # where the energy peaks and why
    performance_notes: str = "" # how it should be sung/played
    revisions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SongDraft":
        secs = []
        for s in d.get("sections", []) or []:
            lines = [DraftLine(**ln) for ln in (s.get("lines") or [])]
            secs.append(DraftSection(
                name=s.get("name", ""), bars=int(s.get("bars", 4) or 4),
                energy=float(s.get("energy", 0.5) or 0.5),
                purpose=s.get("purpose", ""), lines=lines))
        return cls(
            title=d.get("title", ""), topic=d.get("topic", ""),
            style=d.get("style", "pop"), tempo=int(d.get("tempo", 100) or 100),
            key=d.get("key", "C"), mode=d.get("mode", "major"),
            about=d.get("about", ""), sections=secs,
            rhyme_scheme=d.get("rhyme_scheme", ""),
            why_melody_fits=d.get("why_melody_fits", ""),
            energy_arc=d.get("energy_arc", ""),
            performance_notes=d.get("performance_notes", ""),
            revisions=list(d.get("revisions", "") or []),
            created_at=float(d.get("created_at", 0) or time.time()))


# ── drafting ────────────────────────────────────────────────────────────

_DRAFT_PROMPT = """You are a songwriter drafting a song in your notebook — the way \
artists actually draft before ever singing. This is the private draft: raw, \
musical, honest.

Write a {style} song about: {topic!r}
Tempo: {tempo} BPM. Key: {key} {mode}.

Respond with ONLY valid JSON (no markdown, no commentary) in this shape:
{{
  "title": "song title",
  "about": "what this song is REALLY about, 1-2 sentences — the feeling underneath the topic",
  "rhyme_scheme": "e.g. verse ABAB, chorus AAAA hook",
  "sections": [
    {{
      "name": "verse 1",
      "bars": 8,
      "energy": 0.4,
      "purpose": "what this section does — sets the scene, raises stakes, releases, etc.",
      "lines": [
        {{"text": "the lyric line", "syllables": 9,
          "contour": ["hold","up","up","down","down","hold","down","up","hold"],
          "rhyme": "A"}}
      ]
    }}
  ],
  "why_melody_fits": "2-3 sentences: why THIS melody contour fits THESE words — where the rises land on emotional peaks, where it falls on confessions",
  "energy_arc": "2-3 sentences: how energy moves through the song — where it peaks, where it breathes, and why",
  "performance_notes": "how this should be sung and played — vocal delivery, dynamics, feel"
}}

Rules for the draft:
- Structure: verse 1, pre-chorus, chorus, verse 2, pre-chorus, chorus, bridge, final chorus, outro. Adjust bars to fit the style.
- Lyrics must SCAN: syllable counts should feel singable at {tempo} BPM. Short punchy lines for high energy, longer flowing lines for intimacy.
- Contour moves are per-phrase: "up" (rise), "down" (fall), "hold" (stay), "leap" (big jump), "drop" (big fall). The contour must MATCH the lyric's emotion — rises on hope/defiance, falls on loss/confession.
- Rhyme groups must be real rhymes, not near-misses. Label them consistently.
- Energy 0.0-1.0 per section: verses lower, pre-chorus climbing, chorus peaking, bridge the emotional pivot.
- Write like a human artist, not a greeting card. Specific images, real feeling, no clichés ("dancing in the moonlight" is banned).
- The "about", "why_melody_fits", "energy_arc", and "performance_notes" are you showing your work — be specific, not generic.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model output, defensively."""
    text = (text or "").strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in draft response")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # try to salvage: find the largest balanced brace span
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        raise


def draft_song(topic: str, *, style: str = "pop", tempo: int = 0,
               key: str = "", context: Any = None,
               taste_hint: str = "") -> SongDraft:
    """Write a full song draft — the artist's notebook page.

    One LLM call, one creative act: lyrics with cadence, melody contours,
    structure, emotional arc, and the artist's own notes showing the work.
    Raises on failure — a draft is never faked.
    """
    from ..llm.base import Message, SamplingParams
    from ..llm.brain import brain_for

    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is required")
    if context is None:
        raise ValueError("an LLM context is required — drafts are written "
                         "by the model, never templated")

    # sensible musical defaults per style when not specified
    style_key = (style or "pop").lower()
    tempo = tempo or {"afrobeats": 102, "hiphop": 92, "rnb": 88,
                      "pop": 104, "dancehall": 96, "soul": 84,
                      "rock": 128, "edm": 126}.get(style_key, 100)
    key = key or "C"

    prompt = _DRAFT_PROMPT.format(style=style_key, topic=topic,
                                  tempo=tempo, key=key, mode="major")
    if (taste_hint or "").strip():
        prompt += ("\n\nThe person you're writing for: "
                   f"{taste_hint.strip()} — let it shape your choices, "
                   "don't announce it.")
    resp = brain_for(context).chat(
        [Message.user(prompt)],
        SamplingParams(temperature=0.95, max_tokens=4096),
        task_kind="creative")
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise ValueError("the model returned an empty draft")
    data = _extract_json(text)
    data.setdefault("topic", topic)
    data.setdefault("style", style_key)
    data.setdefault("tempo", tempo)
    data.setdefault("key", key)
    draft = SongDraft.from_dict(data)
    if not draft.sections or not any(s.lines for s in draft.sections):
        raise ValueError("the draft came back without lyric sections")
    _log.info("drafted '%s' (%d sections)", draft.title, len(draft.sections))
    return draft


_REVISE_PROMPT = """You are revising a song draft in your notebook. The artist \
(the owner) gave this note:

"{instruction}"

Here is the current draft (JSON):
{draft_json}

Apply the note faithfully: change lyrics, contours, energy, or structure \
— whatever the note calls for. Keep everything the note does NOT touch \
intact. Respond with ONLY the full revised draft as valid JSON in the same \
shape (no markdown, no commentary).
"""


def revise_draft(draft: SongDraft, instruction: str,
                 context: Any = None) -> SongDraft:
    """Revise a draft from a natural-language note.

    e.g. "make the chorus hit harder", "the bridge feels flat — give it
    teeth". Returns a new draft; the old one keeps its revision history.
    """
    from ..llm.base import Message, SamplingParams
    from ..llm.brain import brain_for

    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("a revision note is required")
    if context is None:
        raise ValueError("an LLM context is required for revision")
    prompt = _REVISE_PROMPT.format(
        instruction=instruction,
        draft_json=json.dumps(draft.to_dict(), indent=1)[:12000])
    resp = brain_for(context).chat(
        [Message.user(prompt)],
        SamplingParams(temperature=0.9, max_tokens=4096),
        task_kind="creative")
    text = (getattr(resp, "text", "") or "").strip()
    data = _extract_json(text)
    new = SongDraft.from_dict(data)
    new.revisions = list(draft.revisions) + [instruction]
    new.topic = new.topic or draft.topic
    new.style = new.style or draft.style
    _log.info("revised '%s': %s", new.title, instruction[:60])
    return new


# ── notebook rendering + persistence ─────────────────────────────────────

def render_notebook(draft: SongDraft) -> str:
    """The draft as a readable notebook page (markdown)."""
    out = [f"# 🎵 {draft.title or '(untitled)'}",
           f"*{draft.style} · {draft.tempo} BPM · {draft.key} {draft.mode}*",
           "",
           f"> {draft.about}" if draft.about else "",
           ""]
    for sec in draft.sections:
        bar = "▰" * max(1, int(sec.energy * 10)) + "▱" * (10 - max(1, int(sec.energy * 10)))
        out.append(f"## {sec.name} — {sec.bars} bars  `{bar}` {sec.energy:.1f}")
        if sec.purpose:
            out.append(f"*{sec.purpose}*")
        for ln in sec.lines:
            contour = "".join({"up": "↗", "down": "↘", "hold": "→",
                               "leap": "⤴", "drop": "⤵"}.get(c, "·")
                              for c in (ln.contour or []))
            out.append(f"- {ln.text}  `[{ln.syllables} syl · {contour} · {ln.rhyme}]`")
        out.append("")
    if draft.rhyme_scheme:
        out.append(f"**rhyme:** {draft.rhyme_scheme}")
    if draft.why_melody_fits:
        out.append(f"\n**why the melody fits:** {draft.why_melody_fits}")
    if draft.energy_arc:
        out.append(f"\n**energy arc:** {draft.energy_arc}")
    if draft.performance_notes:
        out.append(f"\n**performance:** {draft.performance_notes}")
    if draft.revisions:
        out.append("\n*revisions:* " + " → ".join(draft.revisions))
    return "\n".join(out).strip() + "\n"


def _draft_path(workdir: str, title: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "untitled").lower()).strip("-")
    return Path(workdir) / DRAFTS_DIR / f"{slug or 'untitled'}.json"


def save_draft(draft: SongDraft, workdir: str = "music") -> str:
    """Persist a draft as JSON. Returns the path."""
    p = _draft_path(workdir, draft.title)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(draft.to_dict(), indent=1), encoding="utf-8")
    return str(p)


def load_draft(path: str) -> SongDraft:
    """Load a draft from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SongDraft.from_dict(data)


def list_drafts(workdir: str = "music") -> list[str]:
    """Paths of saved drafts."""
    d = Path(workdir) / DRAFTS_DIR
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.json"))
