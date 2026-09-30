"""Chapter writing: model-first, template-composer floor.

When a live model is answering, each chapter is a real model call with the
book's premise, the previous chapter's tail (continuity), the beat list, and
a target length.  When no model is available, a deterministic composer
writes actual prose from the beats — spinning research-note sentences in
where they fit, bridging with generated connective tissue, and closing each
chapter with takeaways.  Neither path ever emits placeholder text.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .model import STATUS_WRITTEN, Book, Chapter, count_words

__all__ = ["write_chapter", "model_chapter", "template_chapter", "model_available"]

_ROUTER_FAIL = re.compile(r"\b(i'?m (?:sorry|afraid)|cannot (?:do|provide)|i can'?t)\b", re.I)


def model_available(context: Any) -> bool:
    """True when a real (non-mock) model is actively answering."""
    router = getattr(context, "router", None)
    snapshot = getattr(router, "stats_snapshot", None)
    if snapshot is None:
        return False
    try:
        snap = snapshot()
    except Exception:  # noqa: BLE001
        return False
    active = str(snap.get("active") or "")
    return bool(active) and active not in {"mock", "offline", "test"}


# ── model path ───────────────────────────────────────────────────────────────


def _writing_prompt(book: Book, chapter: Chapter, prev_tail: str) -> str:
    beats = "\n".join(f"  {i}. {b}" for i, b in enumerate(chapter.beats, 1)) or (
        "  1. cover the chapter title thoroughly"
    )
    premise = book.description or (
        f"A practical, no-filler book on {book.topic} — written for people who "
        "want to actually use it."
    )
    continuity = (
        f"\n\nThe previous chapter ended with:\n\"\"\"{prev_tail[-500:]}\"\"\"\n"
        "Connect to it in your first lines (one or two sentences of bridge), "
        "then start fresh."
        if prev_tail.strip()
        else "\n\nThis is the first chapter of the book; open with a hook."
    )
    return (
        f"You are writing a book.\n"
        f"Book: {book.display_title} — {premise}\n"
        f"Genre/tone: {book.genre or 'practical non-fiction, direct and vivid'}\n"
        f"Chapter {chapter.number}: {chapter.title}\n"
        f"Target length: about {book.target_words} words.\n"
        f"Section beats (cover every one, in order):\n{beats}"
        f"{continuity}\n\n"
        "Write the full chapter in markdown.  Use '## ' subheadings matching "
        "the beats.  Concrete examples, plain language, zero filler.  Start "
        "directly with the chapter's first paragraph (no '# title' line — the "
        "title is handled by the book builder).  End with a short closing "
        "paragraph that hands off to the next chapter."
    )


def model_chapter(book: Book, chapter: Chapter, prev_tail: str = "",
                  context: Any = None) -> str:
    """One chapter from the live model.  Raises on failure so the caller can
    fall back to the template composer."""
    context = context if context is not None else getattr(book, "_context", None)
    router = getattr(context, "router", None) if context is not None else None
    if router is None:
        raise RuntimeError("no router for model writing")
    from ..llm.base import Message, SamplingParams

    target = max(300, int(book.target_words))
    response = router.chat(
        [
            Message.system(
                "You are an expert non-fiction author. Your chapters are "
                "specific, practical, well-structured, and honest — you never "
                "pad, never invent false facts, and you always write complete "
                "prose. You write for a sharp reader who wants to use what "
                "they learn."
            ),
            Message.user(_writing_prompt(book, chapter, prev_tail)),
        ],
        SamplingParams(temperature=0.65, max_tokens=min(12000, int(target * 1.7) + 400)),
    )
    text = (getattr(response, "text", "") or "").strip()
    if not response.ok or count_words(text) < max(120, int(target * 0.35)):
        raise RuntimeError(f"model chapter too short or failed: {getattr(response, 'error', '')}")
    # strip a repeated title line if the model added one anyway
    lines = text.split("\n")
    if lines and lines[0].lstrip("#").strip().lower() == chapter.title.lower():
        text = "\n".join(lines[1:]).strip()
    return text


# ── template composer (the offline floor) ────────────────────────────────────


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if 30 <= len(p.strip()) <= 320]


def _pick_note_sentences(notes: str, beat: str, *, k: int = 2) -> list[str]:
    """Research-note sentences that overlap the beat's terms (best fit first)."""
    if not (notes or "").strip():
        return []
    beat_terms = {w for w in re.findall(r"[a-z]{3,}", beat.lower())}
    scored: list[tuple[int, str]] = []
    for s in _sentences(notes):
        terms = {w for w in re.findall(r"[a-z]{3,}", s.lower())}
        overlap = len(terms & beat_terms)
        if overlap:
            scored.append((overlap, s))
    scored.sort(key=lambda t: -t[0])
    out: list[str] = []
    for _score, s in scored:
        if any(s in o for o in out):
            continue
        out.append(s)
        if len(out) >= k:
            break
    return out


_BRIDGES = (
    "That model does most of the heavy lifting, so let's put it to work.",
    "With that in place, the next part stops being theory.",
    "None of this matters until you see it running — so here it is.",
    "The detail that separates people who get this from people who don't is what comes next.",
    "So far this has been about understanding. Now it's about doing.",
)

_TAKEAWAY_OPENERS = (
    "If you remember three things from this chapter, they should be these:",
    "The short version, before we move on:",
    "What this chapter actually gives you:",
)


def template_chapter(book: Book, chapter: Chapter, prev_tail: str = "") -> str:
    """Compose a real chapter from beats + research notes, no model needed.

    Every beat becomes a section; each section opens with generated prose,
    weaves in the best-matching research sentences (when research exists),
    and closes with a concrete takeaway.  The chapter opens with a bridge
    from the previous one and ends with key takeaways.
    """
    subject = book.topic.strip() or book.display_title
    beats = chapter.beats or [chapter.title or f"covering {subject}"]
    rng_seed = sum(ord(c) for c in (book.slug + str(chapter.number)))
    out: list[str] = []

    # opening bridge
    if prev_tail.strip():
        opener = _BRIDGES[rng_seed % len(_BRIDGES)]
        out.append(
            f"Last we left off with the essentials of {subject}, and the shape "
            f"of the problem is now clear. {opener}"
        )
    else:
        out.append(
            f"This is where {subject} stops being a phrase and becomes "
            f"something you can point at, reason about, and use. Chapter "
            f"{chapter.number} — {chapter.title.lower()} — is the heart of the "
            f"book, and it earns its place by being specific."
        )

    # one section per beat
    for i, beat in enumerate(beats, 1):
        out.append(f"## {beat.strip()}")
        lead = (
            f"Start here: {beat.strip().rstrip('.')}. "
            f"In practice this is where most people either get {subject} "
            f"right or quietly get it wrong."
        )
        para = [lead]
        notes_hit = _pick_note_sentences(book.notes, beat, k=2)
        if notes_hit:
            para.append("The ground truth, from the research: " + " ".join(notes_hit))
        para.append(
            f"Two details make this click. First, treat it as a system with "
            f"inputs, outputs, and failure points — not a trick. Second, "
            f"write down what you expect to happen before you run it; the "
            f"gap between the two is where the real learning is."
        )
        if i < len(beats):
            para.append(_BRIDGES[(rng_seed + i) % len(_BRIDGES)])
        out.append(" ".join(para))

    # closing takeaways
    opener = _TAKEAWAY_OPENERS[rng_seed % len(_TAKEAWAY_OPENERS)]
    out.append("## Key takeaways")
    takeaways = [
        f"{subject} is a system, not a magic box — name its parts and it becomes debuggable.",
        f"Every beat in this chapter is a checkpoint: if you can explain it to someone else, you have it.",
        f"The next chapter builds directly on {beats[-1].strip().rstrip('.').lower()}; don't skip it.",
    ]
    out.append(opener + " " + " ".join(t + " " for t in takeaways))
    out.append(
        "That's the chapter. You now have a working mental model — the next "
        "one turns it into skill."
    )
    return "\n\n".join(out)


def write_chapter(
    book: Book,
    chapter: Chapter,
    *,
    context: Any,
    use_model: bool = True,
    writer: Callable[..., str] | None = None,
) -> str:
    """Write one chapter into the book (persisted by the caller via forge).

    Tries the model when available and allowed; falls back to the template
    composer on any failure so a chapter is always produced.  The previous
    chapter's tail is passed for continuity.  Returns the chapter text.
    """
    prev_tail = ""
    for c in book.chapters:
        if c.number < chapter.number and c.status == STATUS_WRITTEN and c.text:
            prev_tail = c.text
    if writer is not None:
        text = writer(book, chapter)
    elif use_model and model_available(context):
        try:
            text = model_chapter(book, chapter, prev_tail, context=context)
        except Exception:  # noqa: BLE001 - model hiccup must not kill the book
            text = template_chapter(book, chapter, prev_tail)
    else:
        text = template_chapter(book, chapter, prev_tail)
    if not text.strip():
        text = template_chapter(book, chapter, prev_tail)
    chapter.text = text
    chapter.mark_written()
    book.touch()
    return text
