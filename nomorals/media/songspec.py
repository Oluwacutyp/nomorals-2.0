"""SongSpec — the composition contract between the LLM and the producer.

The LLM (brain) writes the song: lyrics, structure, chords, groove feel,
melodic ideas, arrangement.  The producer engine RENDERS that vision into
audio.  The algorithm doesn't guess; it executes the model's composition.

When no brain is available, :mod:`nomorals.media.composer_llm` builds a
SongSpec algorithmically — but it still flows through this same schema, so
the render pipeline is unified.

A SongSpec is serializable (to_dict/from_dict) and validatable.
``validate()`` returns a list of error strings; empty means valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GrooveSpec",
    "SectionSpec",
    "SongSpec",
    "SONGSPEC_JSON_SCHEMA",
    "normalize_roman",
]


# ── groove ────────────────────────────────────────────────────────────────

@dataclass
class GrooveSpec:
    """How the rhythm section should feel — in the model's own words,
    plus machine-readable parameters the rhythm engine uses."""

    feel: str = ""
    """Free-text groove description, e.g. "laid-back afrobeats bounce"."""

    swing: float = 0.0
    """0.0 = straight, up to 0.6 = heavy shuffle. Applied to offbeat 16ths."""

    kick_style: str = ""
    """E.g. "four-on-floor driving", "syncopated sparse", "half-time boom-bap"."""

    snare_style: str = ""
    """E.g. "backbeat on 2 and 4", "half-time rim clicks in verse"."""

    hat_style: str = ""
    """E.g. "16th shaker with swing", "sparse 8ths", "trap triplets"."""

    percussion: str = ""
    """Extra percussion, e.g. "congas and shaker", "none", "808 cowbell"."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feel": self.feel, "swing": self.swing,
            "kick_style": self.kick_style, "snare_style": self.snare_style,
            "hat_style": self.hat_style, "percussion": self.percussion,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GrooveSpec":
        d = d or {}
        try:
            swing = float(d.get("swing", 0.0))
        except (TypeError, ValueError):
            swing = 0.0
        return cls(
            feel=str(d.get("feel", "") or ""),
            swing=max(0.0, min(0.6, swing)),
            kick_style=str(d.get("kick_style", "") or ""),
            snare_style=str(d.get("snare_style", "") or ""),
            hat_style=str(d.get("hat_style", "") or ""),
            percussion=str(d.get("percussion", "") or ""),
        )


# ── section ───────────────────────────────────────────────────────────────

_VALID_SECTIONS = frozenset({
    "intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge",
    "drop", "buildup", "break", "interlude", "outro", "hook", "tag",
})


@dataclass
class SectionSpec:
    """One structural block of the song."""

    name: str = "verse"
    bars: int = 8
    chords: list[str] = field(default_factory=list)
    """Roman numerals, one per bar (cycles if fewer than bars)."""

    energy: float = 0.5
    """0.0 = whisper, 1.0 = full drop."""

    lyrics: list[str] = field(default_factory=list)
    melody_idea: str = ""
    """E.g. "stepwise ascent peaking on the hook"."""

    melody_abc: str = ""
    """The LLM's actual melody in ABC notation (research: ChatMusician,
    NotaGen, ComposerX all use ABC as the LLM-music format).  When present,
    the producer renders THESE notes instead of developing a motif."""

    groove_note: str = ""
    """Section-specific rhythm direction, e.g. "strip to kick+shaker"."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "bars": self.bars,
            "chords": list(self.chords), "energy": self.energy,
            "lyrics": list(self.lyrics), "melody_idea": self.melody_idea,
            "melody_abc": self.melody_abc, "groove_note": self.groove_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionSpec":
        d = d or {}
        try:
            bars = int(d.get("bars", 8))
        except (TypeError, ValueError):
            bars = 8
        try:
            energy = float(d.get("energy", 0.5))
        except (TypeError, ValueError):
            energy = 0.5
        chords = d.get("chords") or []
        lyrics = d.get("lyrics") or []
        return cls(
            name=str(d.get("name", "verse") or "verse").lower(),
            bars=max(1, min(64, bars)),
            chords=[str(c) for c in chords if c],
            energy=max(0.0, min(1.0, energy)),
            lyrics=[str(l) for l in lyrics if l],
            melody_idea=str(d.get("melody_idea", "") or ""),
            melody_abc=str(d.get("melody_abc", "") or ""),
            groove_note=str(d.get("groove_note", "") or ""),
        )


def normalize_roman(roman: str, mode: str) -> str:
    """Convert LLM-style roman numerals to the code's ROMAN_DEGREES keys.

    The LLM writes "VI" / "VII" / "III" (common in pop notation); the
    chord engine expects "bVI" / "bVII" / "bIII" in minor keys.
    """
    r = (roman or "").strip()
    if not r:
        return "i"
    if (mode or "").lower() == "minor":
        # uppercase VI/VII/III in a minor context → flat variants
        if r == "VI":
            return "bVI"
        if r == "VII":
            return "bVII"
        if r == "III":
            return "bIII"
    return r


# ── song ──────────────────────────────────────────────────────────────────

@dataclass
class SongSpec:
    """The full composition, as written by the LLM (or the algorithmic
    fallback).  The render pipeline consumes this — never raw prompts."""

    title: str = "Untitled"
    style: str = ""
    topic: str = ""
    key: str = "C"
    mode: str = "minor"
    tempo: int = 100
    time_signature: tuple[int, int] = (4, 4)
    mood: str = ""
    groove: GrooveSpec = field(default_factory=GrooveSpec)
    sections: list[SectionSpec] = field(default_factory=list)
    bass_approach: str = ""
    instrumentation: list[str] = field(default_factory=list)
    arrangement_notes: str = ""
    #: where this spec came from: "llm" or "algorithmic"
    source: str = "algorithmic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "style": self.style, "topic": self.topic,
            "key": self.key, "mode": self.mode, "tempo": self.tempo,
            "time_signature": list(self.time_signature), "mood": self.mood,
            "groove": self.groove.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "bass_approach": self.bass_approach,
            "instrumentation": list(self.instrumentation),
            "arrangement_notes": self.arrangement_notes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SongSpec":
        d = d or {}
        ts = d.get("time_signature") or [4, 4]
        try:
            ts_t = (int(ts[0]), int(ts[1]))
        except (TypeError, ValueError, IndexError):
            ts_t = (4, 4)
        try:
            tempo = int(d.get("tempo", 100))
        except (TypeError, ValueError):
            tempo = 100
        secs = d.get("sections") or []
        inst = d.get("instrumentation") or []
        return cls(
            title=str(d.get("title", "Untitled") or "Untitled"),
            style=str(d.get("style", "") or ""),
            topic=str(d.get("topic", "") or ""),
            key=str(d.get("key", "C") or "C"),
            mode=str(d.get("mode", "minor") or "minor").lower(),
            tempo=max(40, min(220, tempo)),
            time_signature=ts_t,
            mood=str(d.get("mood", "") or ""),
            groove=GrooveSpec.from_dict(d.get("groove") or {}),
            sections=[SectionSpec.from_dict(s) for s in secs
                      if isinstance(s, dict)],
            bass_approach=str(d.get("bass_approach", "") or ""),
            instrumentation=[str(i) for i in inst if i],
            arrangement_notes=str(d.get("arrangement_notes", "") or ""),
            source=str(d.get("source", "algorithmic") or "algorithmic"),
        )

    def validate(self) -> list[str]:
        """Human-readable problems.  Empty list = valid."""
        errors: list[str] = []
        if not self.title.strip():
            errors.append("title is empty")
        if not self.sections:
            errors.append("no sections defined")
        if self.mode not in ("major", "minor"):
            errors.append(f"mode must be major/minor, got {self.mode!r}")
        for i, s in enumerate(self.sections):
            if s.name not in _VALID_SECTIONS:
                errors.append(
                    f"section {i}: unknown name {s.name!r} "
                    f"(expected one of {sorted(_VALID_SECTIONS)})")
            if not s.chords:
                errors.append(f"section {i} ({s.name}): no chords")
        # energy arc sanity: the song should breathe, not flatline
        if len(self.sections) >= 3:
            energies = [s.energy for s in self.sections]
            if max(energies) - min(energies) < 0.15:
                errors.append(
                    "energy arc is flat — sections should rise and fall")
        return errors

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)


# ── JSON schema for the LLM prompt ────────────────────────────────────────

SONGSPEC_JSON_SCHEMA = """{
  "title": "song title",
  "style": "genre, e.g. afrobeats, trap, lo-fi, alt-rnb",
  "key": "musical key root, e.g. A, F#, Eb",
  "mode": "major or minor",
  "tempo": 98,
  "time_signature": [4, 4],
  "mood": "one-line mood description",
  "groove": {
    "feel": "rhythmic feel in words, e.g. 'laid-back afrobeats bounce, lazy behind the beat'",
    "swing": 0.0,
    "kick_style": "e.g. 'syncopated, sparse in verse / driving in chorus'",
    "snare_style": "e.g. 'backbeat on 2 and 4, rim clicks in the verse'",
    "hat_style": "e.g. '16th shaker pattern with swing, open hats on offbeats'",
    "percussion": "e.g. 'congas and shaker' or 'none'"
  },
  "sections": [
    {
      "name": "intro",
      "bars": 4,
      "chords": ["i", "VI", "III", "VII"],
      "energy": 0.25,
      "lyrics": [],
      "melody_idea": "sparse pluck motif, lots of space",
      "melody_abc": "C2 D2 | E2 z2 | G2 E2 | C4 |",
      "groove_note": "kick and shaker only, no snare yet"
    }
  ],
  "bass_approach": "how the bass moves, e.g. 'root-fifth pump, octave jumps into the chorus, locks with the kick'",
  "instrumentation": ["808 bass", "pluck synth", "pad", "shaker"],
  "arrangement_notes": "overall arc — where it builds, where it breathes, where it drops"
}"""
