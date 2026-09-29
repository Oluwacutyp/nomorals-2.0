"""Outline generation: model-first, template fallback.

A book needs a real spine before a word is written.  When a live model is
answering we ask it for a JSON outline (titles + 4-6 beats per chapter).
When it is not, a deterministic template builds a genuine arc — foundations
→ deep dives (spun from the topic's own key terms) → mastery — so a book
still gets written, never stubbed, even fully offline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .model import Book, Chapter

__all__ = ["make_outline", "template_outline", "key_terms"]

_WORD = re.compile(r"[a-z][a-z\-]{2,}")

#: stop words for key-term extraction (topic words that carry no structure)
_STOP = frozenset(
    ("the and for with from that this have has are was were been being does do "
     "can could will would should may might must about into over under out up "
     "down off by or of in on a an how what why when where who which your "
     "you we they it his her its our their not but also more most very much "
     "much many some any each all other others such than then so as if at to "
     "into upon across during before after between through per via without "
     "within among using use used uses new best top real true full guide book")
    .split()
)

_ASPECTS = (
    "core concepts and vocabulary",
    "how it actually works under the hood",
    "common mistakes and how to avoid them",
    "tools, techniques, and practical workflows",
    "real-world examples and case studies",
    "advanced patterns and edge cases",
    "performance, scaling, and failure",
    "trade-offs, risks, and what nobody tells you",
    "building it yourself, step by step",
    "where the field is heading",
)


def key_terms(topic: str, *, limit: int = 8) -> list[str]:
    """The structural words of a topic — what the deep-dive chapters spin on."""
    seen: set[str] = set()
    out: list[str] = []
    for w in _WORD.findall((topic or "").lower()):
        if w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
    if not out:
        out = [(topic or "the subject").strip().lower() or "the subject"]
    return out[:limit]


def template_outline(book: Book, *, n_chapters: int) -> list[Chapter]:
    """Deterministic arc: intro → foundations → N deep dives → mastery.

    Every title and beat is specific to the topic's own key terms — this is
    a real outline, not placeholder text.
    """
    terms = key_terms(book.topic)
    subject = book.topic.strip() or book.display_title
    # anchor chapters are fixed; the middle deep dives absorb the rest.
    # n=3 drops the synthesis chapter (intro + foundations + mastery).
    n = max(3, min(int(n_chapters), 16))
    chapters: list[Chapter] = []

    def add(title: str, beats: list[str]) -> None:
        chapters.append(Chapter(number=len(chapters) + 1, title=title, beats=list(beats)))

    # 1 — what this book is and why it matters
    add(
        f"What {subject} Is — and Why It Matters",
        [
            f"open with the problem {subject} solves and who it is for",
            "define the territory: scope, assumptions, and what the book will cover",
            "why this matters now — the state of the field in one honest picture",
            "how to read this book: structure, prerequisites, and the payoff",
        ],
    )
    # 2 — foundations
    add(
        "Foundations: The Core Ideas",
        [
            f"the essential vocabulary of {subject}, defined plainly",
            "the 3-5 principles everything else is built on",
            "a first, complete mental model you can hold in your head",
            "worked example: the simplest real case, traced end to end",
        ],
    )
    # deep dives — one per key term (capped), then aspect chapters.
    # n=3 has no room for them (nor for synthesis): intro + foundations + mastery.
    middle = max(0, n - 4)
    deep_topics: list[str] = []
    for t in terms[:middle]:
        deep_topics.append(t)
    for aspect in _ASPECTS:
        if len(deep_topics) >= middle:
            break
        deep_topics.append(aspect)
    for topic in deep_topics:
        add(
            f"Deep Dive: {topic.title()}",
            [
                f"what {topic} means in practice, with concrete examples",
                "how it fits with the foundations — the model updated",
                "where it breaks: failure modes, limits, and gotchas",
                "a hands-on walkthrough you can reproduce",
                "key takeaways and how this chapter feeds the next",
            ],
        )
    # penultimate — putting it together (dropped only at n=3)
    if n >= 4:
        add(
            "Putting It All Together",
            [
                "the complete picture: every idea from the book, connected",
                "a real end-to-end project that uses the whole stack",
                "how to decide what applies to your situation (and what doesn't)",
                "the mistakes that waste the most time, with fixes",
            ],
        )
    # final — mastery
    add(
        f"Mastery: Beyond the Basics",
        [
            f"where {subject} is heading — trends, open problems, frontiers",
            "the advanced toolkit: techniques for when the basics are routine",
            "how to keep learning: sources, communities, and self-testing",
            "your first project as an authority: ship something real",
        ],
    )
    return chapters[:n]


def _model_outline(book: Book, *, n_chapters: int) -> list[Chapter] | None:
    """Ask the live model for a JSON outline.  None on any failure — the
    caller falls back to the template."""
    context = getattr(book, "_context", None)
    if context is None:
        return None
    router = getattr(context, "router", None)
    if router is None:
        return None
    try:
        from ..llm.base import Message, SamplingParams

        notes = (book.notes or "")[:6000]
        prompt = (
            f"Write a book outline on: {book.topic}\n"
            f"Genre/tone: {book.genre or 'practical non-fiction'}\n"
            f"Chapters: exactly {n_chapters}\n\n"
            f"Research notes (may be empty):\n{notes or '(none)'}\n\n"
            "Reply with ONLY a JSON array of "
            f"{n_chapters} objects, each exactly: "
            '{"title": "<chapter title>", "beats": ["<4-6 section beats>"]}. '
            "Titles must be specific to the topic (no 'Introduction'/'Conclusion' "
            "as the only structure). No prose, no markdown fences."
        )
        response = router.chat(
            [
                Message.system(
                    "You are a sharp non-fiction editor. You outline books that "
                    "teach real things, in a specific and practical order."
                ),
                Message.user(prompt),
            ],
            SamplingParams(temperature=0.4, max_tokens=2500),
        )
        text = (getattr(response, "text", "") or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        raw = json.loads(text[start : end + 1])
        if not isinstance(raw, list) or not raw:
            return None
        chapters: list[Chapter] = []
        for i, item in enumerate(raw[:n_chapters], 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            beats = [str(b).strip() for b in (item.get("beats") or []) if str(b).strip()]
            if not title:
                continue
            chapters.append(Chapter(number=i, title=title, beats=beats[:8]))
        return chapters or None
    except Exception:  # noqa: BLE001 - outline is best-effort; template is the floor
        return None


def make_outline(book: Book, *, n_chapters: int, context: Any = None) -> list[Chapter]:
    """Fill ``book.chapters`` with a real outline (model first, template floor)."""
    book._context = context  # type: ignore[attr-defined]
    chapters = _model_outline(book, n_chapters=n_chapters)
    if not chapters or len(chapters) < max(3, n_chapters // 2):
        chapters = template_outline(book, n_chapters=n_chapters)
    book.chapters = chapters
    book.status = "planned"
    book.touch()
    return chapters
