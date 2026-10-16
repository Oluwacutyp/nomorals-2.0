"""Publishing extras — cover briefs, blurbs, metadata, ARC, series,
release notifications.

Extends (does not replace) books/publish.py's SerialPublication:
that module owns the release schedule; this one owns everything around
a release — the cover, the blurb, the catalogue metadata, the ARC
(advance-reader-copy) program, series ordering, and the announce
payload a gateway turns into a message.

Cover generation: this module produces a *brief + prompt* (culture-pack
aware palettes, typography notes, element lists). When the caller
supplies an image generator callable, :func:`generate_cover` calls it
with the prompt and returns the bytes/path it produced; when none is
supplied it returns the brief and prompt honestly instead of a fake
image. Attribution is a *label*, not a gate: metadata carries
``attribution`` (e.g. ``"cover: AI-generated"``) wherever it goes.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "CoverBrief", "BookMetadata", "ARCProgram", "SeriesManager",
    "build_cover_brief", "build_cover_prompt", "generate_cover",
    "blurb", "release_notification", "culture_palettes",
    "attribution_label",
]

#: Culture packs: palette + motif notes a cover brief draws on. Packs are
#: *presets* (composable descriptors), never a claim about a whole culture.
_CULTURE_PACKS: dict[str, dict[str, Any]] = {
    "nigeria": {
        "palette": ["#0B6E4F", "#F2C230", "#8A3324", "#F5EFE0", "#1A1A1A"],
        "motifs": ["gele headwrap geometry", "adire pattern",
                   "Lagos skyline at dusk", "yam festival", "bronze plaque"],
        "typography": "bold condensed sans, high contrast, Yoruba/Igbo/"
                      "Hausa diacritics preserved",
    },
    "yoruba": {
        "palette": ["#1F4E79", "#E8B93C", "#7A1F2B", "#F5EFE0"],
        "motifs": ["odu Ifa divination tray", "orisha colours (no sacred "
                   "imagery without permission)", "talking drum"],
        "typography": "serif display, diacritics preserved (ọ, ẹ, ṣ)",
    },
    "generic-fantasy": {
        "palette": ["#1B2430", "#C9A227", "#6B7C93", "#EDEDED"],
        "motifs": ["sigil", "mountain pass", "crowned figure"],
        "typography": "trajan-style serif caps",
    },
    "generic-scifi": {
        "palette": ["#0A0E1A", "#35D0FF", "#FF5A3C", "#E8EDF2"],
        "motifs": ["orbital ring", "derelict station", "signal waveform"],
        "typography": "geometric sans, tight tracking",
    },
    "generic-romance": {
        "palette": ["#7A1F3D", "#F2A65A", "#F5EFE0", "#2B2B2B"],
        "motifs": ["two figures, back to back", "market at golden hour"],
        "typography": "elegant serif italic",
    },
}


def culture_palettes() -> dict[str, dict[str, Any]]:
    """Available cover culture packs (name → palette/motifs/typography)."""
    return json.loads(json.dumps(_CULTURE_PACKS))


def attribution_label(*, cover_ai: bool = False,
                      text_ai: bool = False) -> str:
    """Attribution is a label, not a gate: plain text for metadata."""
    parts = []
    if text_ai:
        parts.append("text: AI-assisted")
    if cover_ai:
        parts.append("cover: AI-generated")
    return "; ".join(parts) if parts else "human-authored"


@dataclass
class CoverBrief:
    title: str
    author: str
    genre: str = ""
    mood: str = ""
    culture_pack: str = ""
    elements: list[str] = field(default_factory=list)
    palette: list[str] = field(default_factory=list)
    typography: str = ""
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cover_brief(title: str, author: str, *, genre: str = "",
                      mood: str = "", culture_pack: str = "",
                      elements: list[str] | None = None) -> CoverBrief:
    """Cover brief from metadata + a culture pack (palette/motifs)."""
    pack = _CULTURE_PACKS.get((culture_pack or "").lower(), {})
    palette = list(pack.get("palette", ["#1A1A1A", "#EDEDED"]))
    motifs = list(pack.get("motifs", []))
    merged = list(elements or []) or motifs[:2]
    brief = CoverBrief(
        title=title, author=author, genre=genre, mood=mood,
        culture_pack=culture_pack, elements=merged, palette=palette,
        typography=str(pack.get("typography", "clean sans, high contrast")))
    brief.prompt = build_cover_prompt(brief)
    return brief


def build_cover_prompt(brief: CoverBrief) -> str:
    """Deterministic image prompt from a brief (no model needed)."""
    parts = [f"Book cover for {brief.title!r} by {brief.author}"]
    if brief.genre:
        parts.append(f"genre: {brief.genre}")
    if brief.mood:
        parts.append(f"mood: {brief.mood}")
    if brief.elements:
        parts.append("elements: " + ", ".join(brief.elements))
    if brief.palette:
        parts.append("palette: " + ", ".join(brief.palette))
    if brief.typography:
        parts.append(f"typography: {brief.typography}")
    parts.append("no text errors, no watermark, print-ready composition, "
                 "title legible at thumbnail size")
    return "; ".join(parts)


def generate_cover(brief: CoverBrief, *,
                   image_generator: Callable[[str], Any] | None = None
                   ) -> dict[str, Any]:
    """Cover via imggen when a generator is supplied, else brief+prompt.

    ``image_generator`` is any callable ``prompt -> path | bytes`` (the
    caller's imggen wiring). Without one, returns
    ``{"ok": False, "reason": "no image generator supplied", "brief", "prompt"}``
    — never a fake image.
    """
    if image_generator is None:
        return {"ok": False, "reason": "no image generator supplied",
                "brief": brief.to_dict(), "prompt": brief.prompt}
    result = image_generator(brief.prompt)
    return {"ok": True, "brief": brief.to_dict(), "prompt": brief.prompt,
            "result": result,
            "attribution": attribution_label(cover_ai=True)}


_SENT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"“])")


def blurb(synopsis_or_text: str, *, max_chars: int = 600,
          hook_sentence: str = "") -> str:
    """Extractive blurb: first sentences + one hook, trimmed to budget.

    No model invention — the blurb is cut from the author's own synopsis
    or opening, never a hallucinated pitch.
    """
    text = " ".join((synopsis_or_text or "").split())
    if not text:
        return ""
    sentences = [s.strip() for s in _SENT.split(text) if s.strip()]
    picked: list[str] = []
    budget = max_chars
    for s in sentences:
        if len(s) > budget - 40 and picked:
            break
        picked.append(s)
        budget -= len(s) + 1
        if len(picked) >= 3:
            break
    out = " ".join(picked)
    if hook_sentence and hook_sentence.strip() not in out:
        out = (out + " " + hook_sentence.strip()).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


@dataclass
class BookMetadata:
    """Catalogue metadata for one book/serial (ONIX-lite, JSON-shaped)."""

    title: str
    author: str
    genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    language: str = "en"
    blurb_text: str = ""
    series: str = ""
    volume: int = 0
    attribution: str = ""
    rights: str = ""
    isbn: str = ""
    age_rating: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.title:
            problems.append("title is required")
        if not self.author:
            problems.append("author is required")
        if self.volume < 0:
            problems.append("volume must be >= 0")
        if self.isbn and not re.match(r"^(97[89][- ]?)?\d{1,5}[- ]?\d+"
                                      r"[- ]?\d+[- ]?[\dX]$",
                                      self.isbn.strip()):
            problems.append(f"isbn {self.isbn!r} does not look like an ISBN")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ARCProgram:
    """Advance-reader-copy program: readers, copies, feedback, stats.

    Persists to ``path`` when given (JSON); otherwise in-memory only.
    Feedback is stored verbatim with reader id + chapter — the author
    decides what to act on.
    """

    def __init__(self, book_slug: str, *,
                 path: Path | str | None = None) -> None:
        self.book_slug = book_slug
        self._path = Path(path) if path else None
        self._data: dict[str, Any] = {"readers": {}, "feedback": []}
        if self._path and self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(
                    encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {"readers": {}, "feedback": []}

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2,
                                      ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self._path)

    def add_reader(self, reader_id: str, *, contact: str = "",
                   chapters: list[int] | None = None) -> dict[str, Any]:
        if not reader_id:
            raise ValueError("add_reader needs a reader_id")
        self._data["readers"][reader_id] = {
            "contact": contact, "chapters": chapters or [],
            "joined": True}
        self._save()
        return {"ok": True, "reader": reader_id,
                "readers": len(self._data["readers"])}

    def record_feedback(self, reader_id: str, chapter: int,
                        feedback: str, *, rating: int = 0) -> dict[str, Any]:
        if reader_id not in self._data["readers"]:
            return {"ok": False,
                    "reason": f"unknown ARC reader {reader_id!r}"}
        if rating and not 1 <= rating <= 5:
            return {"ok": False, "reason": "rating must be 1..5"}
        self._data["feedback"].append(
            {"reader": reader_id, "chapter": chapter,
             "feedback": feedback, "rating": rating})
        self._save()
        return {"ok": True, "feedback_count": len(self._data["feedback"])}

    def stats(self) -> dict[str, Any]:
        fb = self._data["feedback"]
        ratings = [f["rating"] for f in fb if f.get("rating")]
        return {"book": self.book_slug,
                "readers": len(self._data["readers"]),
                "feedback": len(fb),
                "avg_rating": round(sum(ratings) / len(ratings), 2)
                if ratings else None}


class SeriesManager:
    """Series ordering: volumes in order, next unreleased, catch-up."""

    def __init__(self, *, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._series: dict[str, dict[str, Any]] = {}
        if self._path and self._path.exists():
            try:
                self._series = json.loads(self._path.read_text(
                    encoding="utf-8"))
            except (OSError, ValueError):
                self._series = {}

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._series, indent=2,
                                      ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self._path)

    def add_volume(self, series: str, volume: int, book_slug: str, *,
                   title: str = "") -> dict[str, Any]:
        entry = self._series.setdefault(series, {"volumes": []})
        if any(v["volume"] == volume for v in entry["volumes"]):
            return {"ok": False,
                    "reason": f"volume {volume} already in {series!r}"}
        entry["volumes"].append({"volume": volume, "book": book_slug,
                                 "title": title, "released": False})
        entry["volumes"].sort(key=lambda v: v["volume"])
        self._save()
        return {"ok": True, "series": series, "volumes": entry["volumes"]}

    def mark_released(self, series: str, volume: int) -> dict[str, Any]:
        for v in self._series.get(series, {}).get("volumes", []):
            if v["volume"] == volume:
                v["released"] = True
                self._save()
                return {"ok": True, "series": series, "volume": volume}
        return {"ok": False, "reason": "volume not found"}

    def order(self, series: str) -> list[dict[str, Any]]:
        return list(self._series.get(series, {}).get("volumes", []))

    def next_unreleased(self, series: str) -> dict[str, Any] | None:
        for v in self.order(series):
            if not v["released"]:
                return v
        return None


def release_notification(publication: dict[str, Any], *,
                         chapter: int, title: str = "",
                         blurb_text: str = "") -> dict[str, Any]:
    """Announce payload for a chapter release.

    Returns channel-agnostic payloads (announcement text + follower list
    + chapter) — the gateway decides the channel. No send happens here.
    """
    story = publication.get("title") or publication.get("story") or title
    message = (f"New chapter! {story} — chapter {chapter} is live."
               + (f" {blurb_text}" if blurb_text else "")
               + " Read it now.")
    return {"story": story, "chapter": chapter, "message": message,
            "followers": list(publication.get("followers", [])),
            "announce_chat": publication.get("announce_chat", ""),
            "send_to": len(publication.get("followers", []))}
