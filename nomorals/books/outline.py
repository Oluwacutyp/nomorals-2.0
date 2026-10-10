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

from ..llm.brain import brain_for
from .model import Book, Chapter

__all__ = ["make_outline", "template_outline", "key_terms",
           "seed_outline", "assess_continuation", "coverage_map",
           "MAX_ORGANIC_CHAPTERS", "MAX_ORGANIC_WORDS",
           "beat_sheet_outline", "three_act_map"]

#: Safety backstops for organic books — high enough to never constrain a
#: legitimate book, only to stop a runaway (model glitch, pathological
#: topic).  A human author is never told "exactly N chapters"; these are
#: the fire exits, not the floor plan.
MAX_ORGANIC_CHAPTERS = 60
MAX_ORGANIC_WORDS = 200_000

#: words that describe the *form* of the book or the request itself, not
#: its content — excluded from the coverage map so a "novel about mars"
#: grows chapters about mars, not about "novel" or "write".
_FORM_WORDS = frozenset(
    "write writes writing make makes making create creates creating give "
    "novel book guide story essay memoir poem poetry collection tale "
    "fable handbook manual course textbook pamphlet volume series".split()
)

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


#: adjectives / generic filler that make embarrassing chapter topics
#: ("Deep Dive: Practical") — dropped from the coverage map
_WEAK_TERMS = frozenset(
    "practical complete ultimate essential basic simple easy quick good "
    "great big small short long whole old young high low".split()
)

#: hints that the topic is instructional — a human writing a how-to book
#: naturally covers the standard angles (mistakes, tools, examples), so
#: the checklist includes a few; other topics grow only their own threads
_INSTRUCTIONAL_HINTS = (
    "guide", "handbook", "manual", "course", "learn", "tutorial",
    "how to", "textbook", "mastering", "masterclass",
)


def coverage_map(topic: str) -> list[str]:
    """What the book has to cover before it can be called done.

    The topic's own content terms (form words like "novel"/"guide" and
    weak adjectives excluded — they describe the container, not the
    content).  Instructional topics additionally get up to three of the
    standard how-to angles, the way a human author would.  This is a
    checklist, not a chapter count: chapters emerge one decision at a
    time as items get covered.
    """
    terms = [t for t in key_terms(topic, limit=10)
             if t not in _FORM_WORDS and t not in _WEAK_TERMS]
    if not terms:
        terms = [t for t in key_terms(topic, limit=6)
                 if t not in _FORM_WORDS] or ["the subject"]
    items = list(terms)
    lowered = (topic or "").lower()
    if any(h in lowered for h in _INSTRUCTIONAL_HINTS):
        for aspect in _ASPECTS:
            if len(items) >= len(terms) + 3:
                break
            if aspect not in items:
                items.append(aspect)
    return items


def _intro_beats(subject: str) -> list[str]:
    return [
        f"open with the problem {subject} solves and who it is for",
        "define the territory: scope, assumptions, and what the book will cover",
        "why this matters now — the state of the field in one honest picture",
        "how to read this book: structure, prerequisites, and the payoff",
    ]


def _foundations_beats(subject: str) -> list[str]:
    return [
        f"the essential vocabulary of {subject}, defined plainly",
        "the 3-5 principles everything else is built on",
        "a first, complete mental model you can hold in your head",
        "worked example: the simplest real case, traced end to end",
    ]


def _deep_dive_beats(topic: str) -> list[str]:
    return [
        f"what {topic} means in practice, with concrete examples",
        "how it fits with the foundations — the model updated",
        "where it breaks: failure modes, limits, and gotchas",
        "a hands-on walkthrough you can reproduce",
        "key takeaways and how this chapter feeds the next",
    ]


def _synthesis_beats() -> list[str]:
    return [
        "the complete picture: every idea from the book, connected",
        "a real end-to-end project that uses the whole stack",
        "how to decide what applies to your situation (and what doesn't)",
        "the mistakes that waste the most time, with fixes",
    ]


def _mastery_beats(subject: str) -> list[str]:
    return [
        f"where {subject} is heading — trends, open problems, frontiers",
        "the advanced toolkit: techniques for when the basics are routine",
        "how to keep learning: sources, communities, and self-testing",
        "your first project as an authority: ship something real",
    ]


def _next_number(book: Book) -> int:
    return len(book.chapters) + 1


def template_outline(book: Book, *, n_chapters: int) -> list[Chapter]:
    """Deterministic arc: intro → foundations → N deep dives → mastery.

    Every title and beat is specific to the topic's own key terms — this is
    a real outline, not placeholder text.  Used for the count-based path
    (explicit chapter count); the organic path uses seed_outline +
    assess_continuation instead.
    """
    # key terms from the cleaned title when available — never "write"/"me"
    # from a pasted raw request
    _clean = (book.display_title or "").strip()
    if _clean and not _clean.lower().startswith(("write me", "make me")):
        terms = key_terms(_clean)
    else:
        terms = key_terms(book.topic)
    # use the cleaned title as the subject when available — never the raw
    # pasted request ("Write me a book about...")
    subject = _clean or book.topic.strip()
    if len(subject) > 80:
        subject = key_terms(book.topic)[0:1]
        subject = subject[0].title() if subject else book.topic.strip()[:60]
    # anchor chapters are fixed; the middle deep dives absorb the rest.
    # n=3 drops the synthesis chapter (intro + foundations + mastery).
    n = max(3, min(int(n_chapters), 24))
    chapters: list[Chapter] = []

    def add(title: str, beats: list[str], coverage: str = "") -> None:
        chapters.append(Chapter(number=len(chapters) + 1, title=title,
                                beats=list(beats), coverage=coverage))

    # 1 — what this book is and why it matters
    add(f"What {subject} Is — and Why It Matters",
        _intro_beats(subject), "__intro__")
    # 2 — foundations
    add("Foundations: The Core Ideas",
        _foundations_beats(subject), "__foundations__")
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
        add(f"Deep Dive: {topic.title()}", _deep_dive_beats(topic), topic)
    # penultimate — putting it together (dropped only at n=3)
    if n >= 4:
        add("Putting It All Together", _synthesis_beats(), "__synthesis__")
    # final — mastery
    add("Mastery: Beyond the Basics", _mastery_beats(subject), "__mastery__")
    return chapters[:n]


def _model_outline(book: Book, *, n_chapters: int) -> list[Chapter] | None:
    """Ask the live model for a JSON outline.  None on any failure — the
    caller falls back to the template."""
    context = getattr(book, "_context", None)
    if context is None:
        return None
    try:
        from ..llm.power import model_usable
        if not model_usable(context):
            return None
    except Exception:  # noqa: BLE001
        pass
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
        response = brain_for(context).chat(
            [
                Message.system(
                    "You are a sharp non-fiction editor. You outline books that "
                    "teach real things, in a specific and practical order."
                ),
                Message.user(prompt),
            ],
            SamplingParams(temperature=0.4, max_tokens=2500),
        task_kind="creative")
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
    """Fill ``book.chapters`` with a real outline (model first, template floor).

    Count-based path: used when the user explicitly asked for N chapters.
    The organic path (no count given) uses seed_outline + assess_continuation.
    """
    book._context = context  # type: ignore[attr-defined]
    chapters = _model_outline(book, n_chapters=n_chapters)
    if not chapters or len(chapters) < max(3, n_chapters // 2):
        chapters = template_outline(book, n_chapters=n_chapters)
    book.chapters = chapters
    book.status = "planned"
    book.touch()
    return chapters


# ── beat-sheet outlines (Save the Cat, novel-adapted) ───────────────────────
#
# Mined from Blake Snyder's 15-beat structure (see fiction.BeatSheet):
# chapters are annotated with the story beat they serve, so the fiction
# writer's chapter_brief can steer by arc position, not just local beats.


def three_act_map(n_chapters: int) -> dict[str, tuple[int, int]]:
    """STC-aligned act boundaries: I (setup→break into two), II (fun and
    games→dark night), III (break into three→final image)."""
    n = max(3, int(n_chapters))
    a1 = max(1, round(n * 0.20))
    a2 = max(a1 + 1, round(n * 0.80))
    return {"act_1": (1, a1), "act_2": (a1 + 1, a2), "act_3": (a2 + 1, n)}


def beat_sheet_outline(book: Book, *, n_chapters: int,
                       genre: str = "") -> list[Chapter]:
    """Chapters annotated with Save-the-Cat beats for a fiction book.

    Each chapter gets ``coverage`` = the beat name and a beat instruction
    in its beats list.  Genre flavor comes from the engine's
    :meth:`beat_note`.  Beats sharing a chapter merge (Brody's
    multi-scene beats); skipped chapters inherit their position's beat.
    """
    from .fiction import BeatSheet, engine_for
    n = max(3, int(n_chapters))
    sheet = BeatSheet(n)
    engine = engine_for(genre or "fantasy")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in sheet.full():
        grouped.setdefault(entry["chapter"], []).append(entry)
    chapters: list[Chapter] = []
    for ch_no in range(1, n + 1):
        entries = grouped.get(ch_no)
        if entries:
            names = [e["beat"] for e in entries]
            notes = [e["note"] for e in entries]
            coverage = f"__beat__:{names[-1]}"
        else:
            beat = sheet.beat_for(ch_no)
            names, notes = [beat["beat"]], [beat["note"]]
            coverage = f"__beat__:{beat['beat']}"
        title = " / ".join(names)
        beats = list(notes)
        try:
            beats.append(engine.beat_note(ch_no, n))
        except Exception:  # noqa: BLE001
            pass
        chapters.append(Chapter(number=ch_no, title=title, beats=beats,
                                coverage=coverage))
    return chapters


# ── organic path: the outline is a living document ──────────────────────────
#
# A human author never decides "exactly 12 chapters" and then pads or
# truncates.  They write a section, feel what is covered and what is
# missing, and keep going until the book is done.  The organic path works
# the same way: create() plants a seed (the opening arc), and after every
# chapter assess_continuation() decides — from the actual content — whether
# the book needs another chapter and what it should cover, or whether the
# topic is genuinely complete.


def _model_seed(book: Book) -> list[Chapter] | None:
    """Ask the live model for the book's opening arc: the first 3 chapters."""
    context = getattr(book, "_context", None)
    router = getattr(context, "router", None) if context is not None else None
    if router is None:
        return None
    try:
        from ..llm.base import Message, SamplingParams

        notes = (book.notes or "")[:4000]
        prompt = (
            f"You are starting a book, not planning the whole thing. Outline "
            f"only the OPENING ARC — the first 3 chapters — of a book on:\n"
            f"{book.topic}\n"
            f"Genre/tone: {book.genre or 'practical non-fiction'}\n\n"
            f"Research notes (may be empty):\n{notes or '(none)'}\n\n"
            "Reply with ONLY a JSON array of 3 objects, each exactly: "
            '{"title": "<specific chapter title>", '
            '"beats": ["<4-6 section beats>"]}. '
            "Chapter 1 opens the book and frames the subject. Chapter 2 lays "
            "the foundations. Chapter 3 goes deep on the first real thread. "
            "No 'Introduction'/'Conclusion' filler titles. No prose, no "
            "markdown fences."
        )
        response = brain_for(context).chat(
            [Message.system(
                "You are a sharp non-fiction editor. You open books with "
                "momentum — every chapter earns the next one."),
             Message.user(prompt)],
            SamplingParams(temperature=0.4, max_tokens=1500),
        task_kind="creative")
        return _parse_chapters_json(
            getattr(response, "text", "") or "", want=3,
            ok=getattr(response, "ok", False))
    except Exception:  # noqa: BLE001 - template seed is the floor
        return None


def _parse_chapters_json(text: str, *, want: int,
                         ok: bool = True) -> list[Chapter] | None:
    """Parse a model JSON chapter array into Chapter objects (1-based)."""
    if not ok:
        return None
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        raw = json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, list) or not raw:
        return None
    chapters: list[Chapter] = []
    for i, item in enumerate(raw[:want], 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        beats = [str(b).strip() for b in (item.get("beats") or [])
                 if str(b).strip()][:8]
        if not title:
            continue
        chapters.append(Chapter(number=i, title=title, beats=beats))
    return chapters or None


def seed_outline(book: Book, context: Any = None) -> list[Chapter]:
    """Plant the opening arc: the first chapters of an organic book.

    Model-first (the opening 3 chapters), template floor (intro →
    foundations → first deep dive).  The rest of the book grows from
    assess_continuation() as chapters are written — nothing beyond the
    seed is decided here.
    """
    book._context = context  # type: ignore[attr-defined]
    chapters = _model_seed(book)
    if not chapters or len(chapters) < 2:
        subject = book.display_title or book.topic.strip()
        terms = coverage_map(book.topic)
        first = terms[0] if terms else subject
        chapters = [
            Chapter(number=1,
                    title=f"What {subject} Is — and Why It Matters",
                    beats=_intro_beats(subject), coverage="__intro__"),
            Chapter(number=2, title="Foundations: The Core Ideas",
                    beats=_foundations_beats(subject),
                    coverage="__foundations__"),
            Chapter(number=3, title=f"Deep Dive: {first.title()}",
                    beats=_deep_dive_beats(first), coverage=first),
        ]
    else:
        # tag the seed so the heuristic path knows what is covered
        seed_tags = ("__intro__", "__foundations__", None)
        for ch, tag in zip(chapters, seed_tags):
            if tag:
                ch.coverage = tag
            elif book.coverage:
                ch.coverage = book.coverage[0]
    book.chapters = chapters
    book.status = "planned"
    book.touch()
    return chapters


def _model_continuation(book: Book) -> dict[str, Any] | None:
    """Ask the live model: is this book done, or what comes next?

    Returns {"complete": bool, "reason": str, "next": [Chapter, ...]} or
    None when the model is unavailable / its answer is unusable — the
    caller falls back to the heuristic coverage path.
    """
    context = getattr(book, "_context", None)
    router = getattr(context, "router", None)
    if router is None:
        return None
    try:
        from ..llm.base import Message, SamplingParams

        written = [c for c in book.chapters if c.status == "written"]
        lines = []
        for c in written:
            head = " ".join((c.text or "").split()[:40])
            lines.append(f"- Ch{c.number} “{c.title}”: {head}…")
        notes = (book.notes or "")[:3000]
        prompt = (
            f"You are the author of this book, deciding what to write next.\n"
            f"Book: {book.display_title}\nTopic: {book.topic}\n"
            f"Chapters written so far ({len(written)}):\n"
            + ("\n".join(lines) if lines else "(none yet)")
            + f"\n\nResearch notes (may be empty):\n{notes or '(none)'}\n\n"
            "Is the book COMPLETE — has the topic been genuinely covered, "
            "with nothing important missing? Or does it need more chapters? "
            "Be honest: do not pad a finished book, do not starve an "
            "unfinished one.\n\n"
            "Reply with ONLY JSON: "
            '{"complete": true/false, "reason": "<one honest sentence>", '
            '"next": [{"title": "<specific chapter title>", '
            '"beats": ["<4-6 beats>"]}]} — at most 2 chapters, each with '
            "4-6 beats. Empty \"next\" when complete. No prose, no fences."
        )
        response = brain_for(context).chat(
            [Message.system(
                "You are the author, not an outliner. You feel when a book "
                "is done. You never pad and never truncate."),
             Message.user(prompt)],
            SamplingParams(temperature=0.4, max_tokens=1500),
        task_kind="creative")
        text = (getattr(response, "text", "") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start or not getattr(response, "ok", False):
            return None
        decision = json.loads(text[start:end + 1])
        if not isinstance(decision, dict):
            return None
        complete = bool(decision.get("complete"))
        reason = str(decision.get("reason", ""))[:200]
        nxt: list[Chapter] = []
        if not complete:
            for item in (decision.get("next") or [])[:2]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                beats = [str(b).strip() for b in (item.get("beats") or [])
                         if str(b).strip()][:8]
                if title and beats:
                    nxt.append(Chapter(number=0, title=title, beats=beats))
        return {"complete": complete, "reason": reason, "next": nxt}
    except Exception:  # noqa: BLE001 - heuristic path is the floor
        return None


def _heuristic_continuation(book: Book) -> dict[str, Any]:
    """Coverage-driven continuation without a model.

    The book's coverage map is the checklist: every item gets a chapter,
    then the closing arc (synthesis → mastery), then the book is done.
    One decision at a time — the count is never fixed up front.
    """
    covered = {c.coverage for c in book.chapters if c.coverage}
    for item in (book.coverage or []):
        if item not in covered:
            ch = Chapter(number=0, title=f"Deep Dive: {item.title()}",
                         beats=_deep_dive_beats(item), coverage=item)
            return {"complete": False,
                    "reason": f"still to cover: {item}",
                    "next": [ch]}
    subject = book.topic.strip() or book.display_title
    if "__synthesis__" not in covered:
        ch = Chapter(number=0, title="Putting It All Together",
                     beats=_synthesis_beats(), coverage="__synthesis__")
        return {"complete": False,
                "reason": "body covered — time to connect it all",
                "next": [ch]}
    if "__mastery__" not in covered:
        ch = Chapter(number=0, title="Mastery: Beyond the Basics",
                     beats=_mastery_beats(subject), coverage="__mastery__")
        return {"complete": False,
                "reason": "closing the book with the road ahead",
                "next": [ch]}
    return {"complete": True,
            "reason": "every thread covered and the closing arc written",
            "next": []}


def assess_continuation(book: Book, context: Any = None) -> dict[str, Any]:
    """The organic heartbeat: after a chapter is written, decide what — if
    anything — comes next.

    Model-first (the author judges its own book), heuristic floor
    (coverage checklist).  Appends 0-2 new chapters or marks the book
    concluded.  Safety backstops conclude runaway books.  Returns the
    decision: {"complete": bool, "reason": str, "added": [titles]}.
    """
    if context is not None:
        book._context = context  # type: ignore[attr-defined]
    # safety backstops — fire exits, not a floor plan
    if (len(book.chapters) >= MAX_ORGANIC_CHAPTERS
            or book.total_words >= MAX_ORGANIC_WORDS):
        book.concluded = True
        book.touch()
        return {"complete": True,
                "reason": "safety backstop reached — concluding the book",
                "added": []}
    decision = _model_continuation(book)
    if decision is None:
        decision = _heuristic_continuation(book)
    added: list[str] = []
    if decision["complete"]:
        book.concluded = True
    else:
        for ch in decision["next"][:2]:
            ch.number = _next_number(book)
            book.chapters.append(ch)
            added.append(ch.title)
        # a model that keeps saying "not complete" with nothing concrete
        # is stalling — the heuristic checklist still knows what is missing
        if not added:
            fallback = _heuristic_continuation(book)
            if fallback["complete"]:
                book.concluded = True
                decision = fallback
            else:
                for ch in fallback["next"][:2]:
                    ch.number = _next_number(book)
                    book.chapters.append(ch)
                    added.append(ch.title)
                decision = fallback
    book.touch()
    return {"complete": bool(decision["complete"]),
            "reason": str(decision.get("reason", "")),
            "added": added}
