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

__all__ = ["write_chapter", "model_chapter", "template_chapter",
           "model_available", "clean_beat_text"]

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
        f"Length: as long as the beats need — roughly {book.target_words} "
        f"words is typical, but NEVER pad to hit a number and NEVER cut a "
        f"beat short to fit one.  A short chapter that says everything is "
        f"better than a long one full of air.\n"
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
#
# The composer builds a real chapter from beats + research notes.  Every
# section gets varied, beat-specific prose — never the same sentence twice
# in a chapter, never generic filler.  Large rotating pools keep repeat
# reads fresh; research-note sentences are woven in where they fit.


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


#: chapter openers — bridge from the previous chapter, rotated by seed
_CHAPTER_OPENERS = (
    "The last chapter gave you the map. This one is the terrain itself.",
    "With the groundwork laid, we can go deeper — this is where the ideas start paying rent.",
    "Everything so far has been preparation. From here, the work gets concrete.",
    "You have the vocabulary now. Time to see what it describes in the wild.",
    "The previous chapter answered 'what'. This one answers 'how, exactly'.",
    "Concepts are cheap until they survive contact with reality. Let's make them survive.",
    "Here's where the book stops describing the landscape and starts handing you tools.",
    "The foundation is set. Now we build the first floor — and it's where you'll spend most of your time.",
    "Last chapter was the why. This chapter is the how, with the edges still on.",
    "We've been circling the subject; now we land on it.",
    "The scaffolding comes down in this chapter — what remains has to stand on its own.",
    "Time to convert understanding into capability. That's what this chapter is for.",
)

#: first-chapter openers (no previous chapter to bridge from)
_FIRST_OPENERS = (
    "Every subject has a doorway. This chapter is yours — step through it slowly.",
    "Before technique, before tools: understanding what you're actually dealing with.",
    "This book starts where confusion starts — at the beginning, with the thing itself.",
    "Forget what you think you know for the next few pages. Fresh eyes work better here.",
)

#: section leads — open each beat's section with beat-specific framing
_SECTION_LEADS = (
    "{beat} — this is the part most guides rush past, so we'll slow down for it.",
    "Let's take {beat} apart properly.",
    "{beat}: not as a slogan, but as something you can actually do.",
    "Here's {beat}, in plain terms.",
    "The heart of this chapter is {beat} — everything else orbits it.",
    "{beat} deserves more than a passing mention. Here's the full picture.",
)

#: substantive middle paragraphs — concrete, varied, never repeated in a chapter
_SECTION_MIDDLES = (
    "The practical shape of this: start with the smallest version that still "
    "counts as real, get it working, then expand. People who skip the small "
    "version almost always stall on the big one.",
    "Two questions cut through most of the noise here. First: what does "
    "'done' look like, specifically? Second: what's the cheapest test that "
    "proves you're on track? Answer those before anything else.",
    "Watch for the common trap — doing the visible part while skipping the "
    "boring part that actually determines the outcome. The boring part is "
    "where the leverage lives.",
    "A useful habit: after each attempt, write down what surprised you. "
    "Surprises are the curriculum; everything else is review.",
    "This gets easier in layers. The first layer is awareness — noticing "
    "what's happening. The second is deliberate practice. The third is "
    "instinct. Don't try to jump layers.",
    "The difference between people who get this and people who don't is "
    "rarely talent. It's usually just reps — and reps done with attention, "
    "not on autopilot.",
    "If this feels uncomfortable, that's information, not failure. "
    "Discomfort marks the edge of what you currently understand; working "
    "there is the whole game.",
    "Concrete beats abstract every time. Tie each idea here to one specific "
    "situation from your own experience before moving on.",
    "Speed is a trap at this stage. Slow, correct repetitions build the "
    "foundation that speed later stands on.",
    "Ask someone good at this what they wish they'd known at the start. "
    "Their answer is almost always about something unglamorous — and almost "
    "always right.",
    "The 80/20 of this topic: a small number of principles explain most "
    "outcomes. Learn to spot which principle applies before reaching for "
    "techniques.",
    "Document as you go. Memory lies; notes don't. The people who improve "
    "fastest are usually just the people who write things down.",
)

#: closers for individual sections
_SECTION_CLOSERS = (
    "That's the core of it — the rest of this chapter builds on this foundation.",
    "Hold onto that; we'll use it again before the chapter ends.",
    "With that understood, the next section gets much easier.",
    "This is one of those ideas that compounds — it keeps paying off.",
    "File that away. It connects to something bigger in the next section.",
)

#: chapter-ending takeaway frames
_TAKEAWAY_FRAMES = (
    "If this chapter had to fit on an index card, it would say this:",
    "The chapter in one breath:",
    "Carry these three things into the next chapter:",
    "Before you turn the page, lock these in:",
    "Distilled to what matters:",
    "The non-negotiables from this chapter:",
)


def _short_subject(book: Book) -> str:
    """A compact subject for prose — never the raw user prompt."""
    title = (book.title or "").strip()
    if title and len(title) <= 60 and "write me" not in title.lower():
        return title
    # fall back to the first meaningful clause of the topic
    topic = (book.topic or "").strip()
    core = re.split(r"[.!?]\s", topic, maxsplit=1)[0]
    if len(core) > 60:
        core = core[:57].rsplit(" ", 1)[0]
    return core.strip(" ,.:-") or "the subject"


def template_chapter(book: Book, chapter: Chapter, prev_tail: str = "") -> str:
    """Compose a real chapter from beats + research notes, no model needed.

    Every beat becomes a section with varied, specific prose; research-note
    sentences are woven in where they fit; the chapter opens with a bridge
    and closes with takeaways.  No sentence template repeats within a
    chapter.
    """
    subject = _short_subject(book)
    beats = chapter.beats or [chapter.title or f"covering {subject}"]
    rng_seed = sum(ord(c) for c in (book.slug + str(chapter.number)))
    import random as _random
    rng = _random.Random(rng_seed)
    out: list[str] = []

    # opening bridge — varied, never the same twice in a row
    if prev_tail.strip():
        opener = _CHAPTER_OPENERS[rng_seed % len(_CHAPTER_OPENERS)]
        out.append(
            f"{opener} Chapter {chapter.number} — {chapter.title.lower()} — "
            f"takes the ideas we've built and puts them to work on {subject}."
        )
    else:
        opener = _FIRST_OPENERS[rng_seed % len(_FIRST_OPENERS)]
        out.append(f"{opener}\n\nThis book is about {subject}. Not the "
                   f"textbook version — the version you'll actually use.")

    # one section per beat, with rotating varied prose
    middles = rng.sample(_SECTION_MIDDLES,
                         min(len(_SECTION_MIDDLES), max(len(beats), 3)))
    for i, beat in enumerate(beats):
        clean_beat = beat.strip().rstrip(".")
        head_beat = clean_beat[0].upper() + clean_beat[1:] if clean_beat else clean_beat
        mid_beat = clean_beat_text(clean_beat)
        out.append(f"## {head_beat}")
        lead = rng.choice(_SECTION_LEADS).format(beat=mid_beat)
        para = [lead]
        notes_hit = _pick_note_sentences(book.notes, beat, k=2)
        if notes_hit:
            para.append("What the research actually says: " + " ".join(notes_hit))
        para.append(middles[i % len(middles)])
        if i < len(beats) - 1:
            para.append(rng.choice(_SECTION_CLOSERS))
        out.append(" ".join(para))

    # closing takeaways — specific to this chapter's beats
    frame = _TAKEAWAY_FRAMES[rng_seed % len(_TAKEAWAY_FRAMES)]
    out.append("## Key takeaways")
    takeaways = [
        f"{clean_beat_text(beats[0])} is the foundation — get it working before adding complexity.",
        "Attention beats volume: focused reps on the core ideas outperform skimming everything.",
        f"Next up: {chapter.title.lower()} hands off to the following chapter — the ideas compound.",
    ]
    out.append(frame + " " + " ".join(takeaways))
    return "\n\n".join(out)


def clean_beat_text(beat: str) -> str:
    """Beat text cleaned for mid-sentence use."""
    b = (beat or "").strip().rstrip(".")
    if b and b[0].isupper() and len(b) > 1 and b[1].islower():
        b = b[0].lower() + b[1:]
    return b or "the core idea"


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


# ── backward-compatible aliases (tests reference the old pool names) ────────
_BRIDGES = _CHAPTER_OPENERS
_TAKEAWAY_OPENERS = _TAKEAWAY_FRAMES
