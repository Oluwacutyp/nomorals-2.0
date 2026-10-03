"""Book model: chapters, status, and on-disk persistence.

A book is a durable, resumable object. Everything lives under
``workspace/books/<slug>/`` so an interrupted run (phone off, wifi dropped,
kernel restarted) can be picked up exactly where it stopped — the owner's
real constraint, since a whole book takes real model time.

Layout::

    books/<slug>/
        book.json          # full book (meta + chapters + status)
        notes.md           # research notes gathered during create/plan
        chapters/ch01.md   # one file per chapter (readable, editable)
        manuscript.md      # compiled book (rebuilt on build)
        <slug>.pdf         # final PDF (rebuilt on build)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Book", "Chapter", "BookError"]

_SLUG = re.compile(r"[^A-Za-z0-9]+")

STATUS_PLAN = "planned"
STATUS_WRITING = "writing"
STATUS_WRITTEN = "written"
STATUS_BUILT = "built"
STATUS_SENT = "sent"


class BookError(ValueError):
    """Raised for invalid book operations."""


def slugify(text: str, fallback: str = "book") -> str:
    """Filesystem-safe slug from a title/topic."""
    text = (text or "").strip().lower()
    slug = _SLUG.sub("-", text).strip("-")
    slug = slug[:60].strip("-")
    if not slug:
        slug = fallback
    return slug


def count_words(text: str) -> int:
    return len((text or "").split())


@dataclass
class Chapter:
    number: int
    title: str = ""
    beats: list[str] = field(default_factory=list)
    text: str = ""
    status: str = STATUS_PLAN
    words: int = 0
    #: which coverage item this chapter addresses (organic books).  The
    #: seed chapters use "__intro__" / "__foundations__"; the closing arc
    #: uses "__synthesis__" / "__mastery__".  Lets the continuation
    #: assessment know what is already covered without fuzzy matching.
    coverage: str = ""

    def __post_init__(self) -> None:
        self.words = count_words(self.text)

    def mark_written(self) -> None:
        self.status = STATUS_WRITTEN
        self.words = count_words(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "beats": list(self.beats),
            "text": self.text,
            "status": self.status,
            "words": self.words,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chapter":
        return cls(
            number=int(d.get("number", 0)),
            title=str(d.get("title", "")),
            beats=[str(b) for b in d.get("beats", [])],
            text=str(d.get("text", "")),
            status=str(d.get("status", STATUS_PLAN)),
            words=int(d.get("words", 0)),
            coverage=str(d.get("coverage", "")),
        )


@dataclass
class Book:
    topic: str
    slug: str
    title: str = ""
    subtitle: str = ""
    author: str = ""
    genre: str = ""
    description: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    target_words: int = 1200  # per chapter — a soft guide, never a quota
    status: str = STATUS_PLAN
    notes: str = ""
    #: organic mode: the book grows as it is written.  No chapter count is
    #: ever decided up front — chapters emerge from continuation
    #: assessments until the topic is genuinely covered.  Set when the book
    #: is created without an explicit chapter count.
    organic: bool = False
    #: the continuation assessment decided the book is complete.  Only
    #: meaningful in organic mode.
    concluded: bool = False
    #: ordered coverage map for the heuristic continuation path
    #: (content key terms + aspects).  The book is done when every item is
    #: covered and the closing arc is written.
    coverage: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── derived ─────────────────────────────────────────────────────────
    @property
    def display_title(self) -> str:
        return self.title or self.topic or "Untitled"

    @property
    def total_words(self) -> int:
        return sum(c.words for c in self.chapters)

    @property
    def chapters_written(self) -> int:
        return sum(1 for c in self.chapters if c.status == STATUS_WRITTEN)

    @property
    def complete(self) -> bool:
        if not self.chapters:
            return False
        all_written = all(c.status == STATUS_WRITTEN for c in self.chapters)
        if self.organic:
            # organic books are done when the author (continuation
            # assessment) says the topic is covered AND everything planned
            # is written — the outline keeps growing until then.
            return self.concluded and all_written
        return all_written

    def chapter(self, number: int) -> Chapter:
        for c in self.chapters:
            if c.number == number:
                return c
        raise BookError(f"no chapter {number} in {self.slug!r}")

    def next_unwritten(self) -> Chapter | None:
        for c in self.chapters:
            if c.status != STATUS_WRITTEN:
                return c
        return None

    def touch(self) -> None:
        self.updated_at = time.time()

    # ── persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "slug": self.slug,
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "genre": self.genre,
            "description": self.description,
            "chapters": [c.to_dict() for c in self.chapters],
            "target_words": self.target_words,
            "status": self.status,
            "notes": self.notes,
            "organic": self.organic,
            "concluded": self.concluded,
            "coverage": list(self.coverage),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Book":
        return cls(
            topic=str(d.get("topic", "")),
            slug=str(d.get("slug", "book")),
            title=str(d.get("title", "")),
            subtitle=str(d.get("subtitle", "")),
            author=str(d.get("author", "")),
            genre=str(d.get("genre", "")),
            description=str(d.get("description", "")),
            chapters=[Chapter.from_dict(c) for c in d.get("chapters", [])],
            target_words=int(d.get("target_words", 1200)),
            status=str(d.get("status", STATUS_PLAN)),
            notes=str(d.get("notes", "")),
            organic=bool(d.get("organic", False)),
            concluded=bool(d.get("concluded", False)),
            coverage=[str(x) for x in d.get("coverage", [])],
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    # ── manuscript assembly ─────────────────────────────────────────────
    def manuscript(self) -> str:
        """The whole book as markdown, front matter + chapters.

        This is what the PDF builder and the .md export consume.  Chapters
        are joined with a page break sentinel so the PDF can start each
        chapter on a fresh page.
        """
        parts: list[str] = []
        head = f"# {self.display_title}\n"
        if self.subtitle:
            head += f"*{self.subtitle}*\n"
        if self.author:
            head += f"by {self.author}\n"
        parts.append(head.strip())
        for c in self.chapters:
            if c.status != STATUS_WRITTEN or not c.text.strip():
                continue
            parts.append(f"# {c.title or f'Chapter {c.number}'}\n\n{c.text.strip()}")
        return "\n\n".join(p for p in parts if p).strip() + "\n"
