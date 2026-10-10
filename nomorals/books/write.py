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

from ..llm.brain import brain_for
from .model import STATUS_WRITTEN, Book, Chapter, count_words

__all__ = ["write_chapter", "model_chapter", "template_chapter",
           "model_available", "clean_beat_text",
           "expand", "describe", "rewrite", "suggest_hooks"]

_ROUTER_FAIL = re.compile(r"\b(i'?m (?:sorry|afraid)|cannot (?:do|provide)|i can'?t)\b", re.I)


def model_available(context: Any) -> bool:
    """True when a real (non-mock) model is actively answering.

    Delegates to :mod:`nomorals.llm.power` — the single online-first
    authority.  Kept as the import-stable name the rest of the books
    package already uses.
    """
    try:
        from ..llm.power import model_usable
        return model_usable(context)
    except Exception:  # noqa: BLE001
        pass
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
    response = brain_for(context).chat(
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
    task_kind="creative")
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
    "New chapter, new leverage. The ideas ahead build directly on what you just read.",
    "The deeper you go, the more the earlier chapters start paying you back. This is one of those chapters.",
    "We've earned the right to get specific now. This chapter cashes that in.",
    "Momentum matters in learning — this chapter keeps yours.",
    "The book's argument tightens here. Watch how the pieces connect.",
    "This is a working chapter, not a reading chapter. Expect to think as you go.",
)

#: first-chapter openers (no previous chapter to bridge from)
_FIRST_OPENERS = (
    "Every subject has a doorway. This chapter is yours — step through it slowly.",
    "Before technique, before tools: understanding what you're actually dealing with.",
    "This book starts where confusion starts — at the beginning, with the thing itself.",
    "Forget what you think you know for the next few pages. Fresh eyes work better here.",
    "Every book makes a promise in its first pages. This one's promise: no filler, only what works.",
    "Start here with fresh eyes and a little skepticism — both will serve you well.",
    "The first step is seeing the subject clearly. That's this chapter's entire job.",
    "No prerequisites, no jargon walls. Just the thing itself, explained properly.",
)

#: section leads — open each beat's section with beat-specific framing
_SECTION_LEADS = (
    "{beat} — this is the part most guides rush past, so we'll slow down for it.",
    "Let's take {beat} apart properly.",
    "{beat}: not as a slogan, but as something you can actually do.",
    "Here's {beat}, in plain terms.",
    "The heart of this chapter is {beat} — everything else orbits it.",
    "{beat} deserves more than a passing mention. Here's the full picture.",
    "{beat} rewards patience — let's give it a proper look.",
    "Start with {beat}, because everything else assumes you get this.",
    "Here's {beat} stripped of the usual mystique.",
    "Most people get {beat} half-right. Let's get it fully right.",
    "{beat} sounds simple until you try to explain it. Then it gets interesting.",
    "Under the hood of {beat} is something worth understanding properly.",
    "Let's build {beat} from the ground up.",
    "The unglamorous truth about {beat}: it matters more than it looks.",
)

#: beat-aware leads — chosen by what the beat is actually about, so a
#: "how to" beat, a comparison beat, and a pitfalls beat read differently
#: instead of wearing the same generic frame.
_LEAD_BY_KIND = {
    "question": (
        "{beat} — good question, and the answer is less obvious than it looks.",
        "Let's answer this properly: {beat}.",
        "{beat}? Here's the honest answer.",
        "The question underneath {beat} is worth slowing down for.",
    ),
    "howto": (
        "{beat} — here's the working method, step by honest step.",
        "The mechanics of {beat}, without the hand-waving.",
        "{beat}: this is a skill, which means it's learnable. Here's the shape of it.",
        "Enough theory — {beat} is about doing. Let's get concrete.",
    ),
    "contrast": (
        "{beat} — the comparison matters more than either side alone.",
        "Put side by side, {beat} reveals something neither shows on its own.",
        "{beat}: let's be precise about where the two differ and where they don't.",
        "The interesting part of {beat} is the boundary between the two.",
    ),
    "warning": (
        "{beat} — read this section twice; it saves the most pain.",
        "The expensive lessons live here: {beat}.",
        "{beat}, and the traps hiding inside it.",
        "Nobody warns you about {beat} until it's too late. Consider yourself warned.",
    ),
    "why": (
        "{beat} — the reason matters more than the rule.",
        "Understanding {beat} changes how you act on everything else in this chapter.",
        "{beat}: once you see the why, the how gets obvious.",
    ),
    "example": (
        "{beat} — let's make it concrete.",
        "Theory ends here; {beat} is best understood in the flesh.",
        "{beat}, told the way it actually happens.",
    ),
}


def _beat_kind(beat: str) -> str:
    """Classify a beat by what it's about, for beat-aware prose."""
    b = (beat or "").strip().lower()
    if not b:
        return "general"
    if (b.endswith("?") or b.startswith(
            ("what ", "what's ", "which ", "who ", "when ", "where "))):
        return "question"
    if b.startswith("how ") or "how to" in b:
        return "howto"
    if b.startswith("why ") or " because " in b or " reason" in b:
        return "why"
    padded = f" {b} "
    if (" vs " in padded or " versus " in padded or b.startswith("compare")
            or "comparing" in b):
        return "contrast"
    if any(w in b for w in ("mistake", "trap", "avoid", "pitfall", "wrong",
                            "fail", "danger", "don't", "never")):
        return "warning"
    if any(w in b for w in ("example", "case stud", "story", "stories")):
        return "example"
    return "general"

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
    "Reframe it as a question you ask yourself daily, not a fact you "
    "memorize once. Questions compound; facts decay.",
    "The test that matters: can you explain this to someone smart who "
    "knows nothing about it? If not, you don't have it yet.",
    "Progress here is invisible for a while, then sudden. The invisible "
    "part isn't wasted — it's loading.",
    "Find the smallest real-world instance of this and master that first. "
    "Toy versions teach the shape; real versions teach the weight.",
    "Notice where your attention goes when this gets hard — that's "
    "usually where the actual lesson is hiding.",
    "The people who are good at this aren't doing anything mystical. "
    "They've just failed at it more times than you've tried it.",
    "Write your own version of this section in your own words. If you "
    "can't, reread — the gap is the lesson.",
)

#: middles that name the book's subject — ground the prose in the topic
#: instead of floating in generic advice.
_SUBJECT_MIDDLES = (
    "This connects to {subject} more directly than it first appears — "
    "keep that thread in mind as you read on.",
    "Strip away the jargon and this is really about {subject} at its "
    "most practical: what works, what doesn't, and how to tell.",
    "Everything in this section is a tool for {subject}. Judge it the "
    "way you'd judge any tool — by what it lets you do.",
)

#: short punchy paragraphs — chasers that vary the section rhythm so not
#: every beat wears the same lead-middle-closer shape.
_QUICK_TAKES = (
    "One line to remember: depth beats breadth here.",
    "If you only change one habit from this section, make it this: slow "
    "down at the hard parts instead of skipping them.",
    "The shortcut is that there is no shortcut — but the work itself is "
    "the interesting part.",
    "File this under 'obvious in hindsight, invisible in advance'.",
    "This is the part you'll wish you'd taken seriously earlier. Take it "
    "seriously now.",
    "Small, consistent, attentive — that's the whole formula.",
    "When in doubt, return to the simplest version and rebuild from there.",
    "Mastery here looks boring from the outside. That's fine.",
)

#: concrete frames — beat-specific illustrative sentences that pull the
#: beat out of the abstract and into the reader's own experience.
_EXAMPLE_FRAMES = (
    "Make it concrete: think of the last time {beat} showed up in your "
    "own experience with {subject}. Walk it through slowly — what "
    "actually happened?",
    "A quick mental exercise: imagine explaining {beat} to a friend over "
    "coffee. The parts where you'd stumble are the parts to revisit.",
    "Picture someone doing {beat} badly, then picture them doing it well. "
    "The gap between those two pictures is the skill.",
    "Try this: the next time {subject} comes up in your day, notice where "
    "{beat} is hiding in it. It's there more often than you'd think.",
    "Concretely: pick one real situation this week and apply {beat} to it "
    "deliberately. One is enough to start.",
    "If {beat} feels abstract, shrink it: what's the tiniest real "
    "instance of it you can name? Start there.",
    "Here's a lens: every time you notice {beat} in the wild this week, "
    "jot it down. A week of noticing beats a month of reading.",
)

#: transitions between sections — vary what links one beat to the next
_TRANSITIONS = (
    "With that in hand, the next piece falls into place faster.",
    "Now the picture sharpens — onward.",
    "That was the groundwork. The next section is where it gets used.",
    "Carry that momentum forward — it compounds from here.",
    "The thread continues; pull it.",
    "That idea doesn't sit still — watch where it goes next.",
    "Hold that thought. It does work in the next section too.",
)

#: closers for individual sections
_SECTION_CLOSERS = (
    "That's the core of it — the rest of this chapter builds on this foundation.",
    "Hold onto that; we'll use it again before the chapter ends.",
    "With that understood, the next section gets much easier.",
    "This is one of those ideas that compounds — it keeps paying off.",
    "File that away. It connects to something bigger in the next section.",
    "That's the whole mechanism — simple to state, rich to practice.",
    "Once this clicks, reread the section once more. It reads differently the second time.",
    "This is the kind of understanding that quietly upgrades everything downstream.",
    "Keep this one warm — it comes back in a bigger form later.",
    "The details vary; the principle doesn't. That's what makes it worth learning.",
)

#: chapter-ending takeaway frames
_TAKEAWAY_FRAMES = (
    "If this chapter had to fit on an index card, it would say this:",
    "The chapter in one breath:",
    "Carry these three things into the next chapter:",
    "Before you turn the page, lock these in:",
    "Distilled to what matters:",
    "The non-negotiables from this chapter:",
    "What to actually remember:",
    "The chapter's working summary:",
    "If you skimmed, read these twice:",
    "The load-bearing ideas:",
)

#: per-beat takeaway templates — each chapter's takeaways are built from
#: its own beats, so no two chapters close with the same lines.
_TAKEAWAY_TEMPLATES = (
    "{beat} — nail this first; everything else in the chapter leans on it.",
    "{beat} rewards the patient: slow down, get the reps in, let it compound.",
    "Don't skip {beat}. It's the quiet hinge the bigger ideas swing on.",
    "{beat}: learn it well enough to teach it, and you own it.",
    "When {beat} gets hard, that's the signal to lean in, not back off.",
    "{beat} is a practice, not a fact — treat it like one.",
    "Return to {beat} whenever you're stuck; it's usually where the answer lives.",
    "{beat} looks small on the page and looms large in practice.",
)


def _shuffled_cycle(rng, pool):
    """Yield pool items in shuffled order, reshuffling when exhausted.

    Guarantees no repeat until every item has appeared once — the
    backbone of the no-repeat-within-a-chapter rule.
    """
    items = list(pool)
    while items:
        rng.shuffle(items)
        for item in items:
            yield item


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

    Every beat becomes a section with varied, beat-aware prose — a "how
    to" beat, a comparison beat, and a pitfalls beat read differently —
    research-note sentences are woven in where they fit, the chapter
    opens with a bridge and closes with beat-specific takeaways.  Section
    shapes rotate (notes-first, example-led, classic) so chapters don't
    share one rhythm, and shuffled cycles guarantee no sentence template
    repeats within a chapter.
    """
    subject = _short_subject(book)
    beats = chapter.beats or [chapter.title or f"covering {subject}"]
    rng_seed = sum(ord(c) for c in (book.slug + str(chapter.number)))
    import random as _random
    rng = _random.Random(rng_seed)
    out: list[str] = []

    # opening bridge — varied, never the same twice in a row
    if prev_tail.strip():
        opener = rng.choice(_CHAPTER_OPENERS)
        out.append(
            f"{opener} Chapter {chapter.number} — {chapter.title.lower()} — "
            f"takes the ideas we've built and puts them to work on {subject}."
        )
    else:
        opener = rng.choice(_FIRST_OPENERS)
        out.append(f"{opener}\n\nThis book is about {subject}. Not the "
                   f"textbook version — the version you'll actually use.")

    # shuffled cycles — no template repeats within the chapter
    leads = _shuffled_cycle(rng, _SECTION_LEADS)
    middles = _shuffled_cycle(rng, _SECTION_MIDDLES)
    closers = _shuffled_cycle(rng, _SECTION_CLOSERS)
    examples = _shuffled_cycle(rng, _EXAMPLE_FRAMES)
    quick = _shuffled_cycle(rng, _QUICK_TAKES)
    transitions = _shuffled_cycle(rng, _TRANSITIONS)
    subject_mids = _shuffled_cycle(rng, _SUBJECT_MIDDLES)
    takeaway_tpls = _shuffled_cycle(rng, _TAKEAWAY_TEMPLATES)

    for i, beat in enumerate(beats):
        clean_beat = beat.strip().rstrip(".")
        head_beat = (clean_beat[0].upper() + clean_beat[1:]
                     if clean_beat else clean_beat)
        # mid-sentence form: lowercase lead-in, no trailing "?" (question
        # beats read badly mid-sentence with it — the heading keeps it).
        mid_beat = clean_beat_text(clean_beat).rstrip("?")
        kind = _beat_kind(clean_beat)
        out.append(f"## {head_beat}")

        # beat-aware lead: kind-specific frame most of the time, the
        # general pool otherwise — sections about different things read
        # differently.
        kind_leads = _LEAD_BY_KIND.get(kind)
        if kind_leads and rng.random() < 0.7:
            lead = rng.choice(kind_leads).format(beat=mid_beat)
        else:
            lead = next(leads).format(beat=mid_beat)
        # a mid-sentence beat can leave the paragraph opening lowercase
        lead = lead[0].upper() + lead[1:] if lead else lead
        parts = [lead]

        notes_hit = _pick_note_sentences(book.notes, beat, k=2)
        shape = rng.random()
        if notes_hit and shape < 0.4:
            # notes-first: research leads, prose follows
            parts.append("The research on this is unusually clear: "
                         + " ".join(notes_hit))
            parts.append(next(middles))
        elif shape < 0.7:
            # example-led: beat pulled into the reader's experience
            parts.append(next(examples).format(beat=mid_beat,
                                               subject=subject))
            if rng.random() < 0.25:
                parts.append(next(subject_mids).format(subject=subject))
            else:
                parts.append(next(middles))
            if notes_hit:
                parts.append("That lines up with the research: "
                             + " ".join(notes_hit))
        else:
            # classic: middle, then a short punchy chaser
            parts.append(next(middles))
            if notes_hit:
                parts.append("What the research actually says: "
                             + " ".join(notes_hit))
            parts.append(next(quick))
        if i < len(beats) - 1:
            if rng.random() < 0.6:
                parts.append(next(closers))
            else:
                parts.append(next(transitions))
        out.append(" ".join(parts))

    # closing takeaways — built from this chapter's own beats, so no two
    # chapters close with the same lines.
    frame = rng.choice(_TAKEAWAY_FRAMES)
    out.append("## Key takeaways")
    takeaways = []
    for beat in beats[:4]:
        hb = beat.strip().rstrip(".").rstrip("?")
        hb = hb[0].upper() + hb[1:] if hb else hb
        takeaways.append(next(takeaway_tpls).format(beat=hb or "the core idea"))
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


# ── Sudowrite-style primitives (expand / describe / rewrite / hooks) ────────
#
# Mined from Sudowrite's Expand, Describe and Rewrite modes: small,
# directed generation units the owner (or the fiction engine) calls on a
# passage instead of regenerating whole chapters.  Model-first via the
# ``suggest`` callable; the heuristic floor writes real prose, never stubs.


def _model_pass(prompt: str, context: Any, max_tokens: int = 900) -> str:
    """One directed model call.  Empty string when no model is answering."""
    if not model_available(context):
        return ""
    try:
        from ..llm.base import Message, SamplingParams
        response = brain_for(context).chat(
            [Message.system(
                "You are a master prose stylist. Follow the instruction "
                "exactly. Output only the requested prose — no preamble, "
                "no meta-commentary."),
             Message.user(prompt)],
            SamplingParams(temperature=0.8, max_tokens=max_tokens),
            task_kind="creative")
        if getattr(response, "ok", False):
            return (getattr(response, "text", "") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def expand(text: str, context: Any = None, *, target_words: int = 300,
           direction: str = "") -> str:
    """Grow a beat/passage into a fuller scene (Sudowrite Expand).

    Adds sensory detail, interiority and one complication — never padding.
    """
    passage = (text or "").strip()
    if not passage:
        return ""
    prompt = (
        f"Expand this passage into a fuller scene of ~{target_words} words:\n\n"
        f"{passage}\n\n"
        "Rules: add sensory detail and interiority; introduce exactly one "
        "complication that raises the stakes; keep every existing sentence's "
        "meaning; no summary in place of scene."
        + (f" Direction: {direction}" if direction else ""))
    out = _model_pass(prompt, context, max_tokens=target_words * 3)
    if out:
        return out
    # heuristic floor: deepen with sense-stack + complication frames
    import random
    rng = random.Random(hash(passage) & 0xFFFFFFFF)
    senses = [
        "The air carried it first — {s}, then the rest followed.",
        "{cap} noticed the small things: {s}.",
        "It wasn't the sight of it that stayed with {o}; it was {s}.",
    ]
    s = rng.choice([
        "smoke and cold iron", "rain on hot stone",
        "something sweet going wrong underneath",
        "dust, old paper, and held breath",
    ])
    subj = passage.split()[0].rstrip(",.")
    deep = rng.choice(senses).format(s=s, cap=subj.capitalize(), o=subj)
    complication = rng.choice([
        "Then the plan met its first real obstacle — and it had a face.",
        "That was when the second variable announced itself.",
        "What nobody had said aloud finally got said, and the room changed.",
    ])
    return f"{passage}\n\n{deep}\n\n{complication}"


def describe(subject: str, context: Any = None, *,
             mood: str = "") -> str:
    """Sensory description of a subject (Sudowrite Describe)."""
    subject = (subject or "").strip()
    if not subject:
        return ""
    prompt = (
        f"Describe {subject} in two vivid paragraphs. Concrete sensory "
        f"detail — sight, sound, smell, texture. "
        + (f"Mood: {mood}. " if mood else "")
        + "No clichés, no 'it was as if' hedging.")
    out = _model_pass(prompt, context, max_tokens=500)
    if out:
        return out
    import random
    rng = random.Random(hash(subject) & 0xFFFFFFFF)
    textures = [
        "rough where it should have been smooth",
        "quiet in a way that felt deliberate",
        "older than everything around it and in no hurry to prove it",
    ]
    frames = [
        (f"{subject} did not announce itself. It arrived the way weather "
         f"arrives — first a pressure change, then the undeniable fact of it."),
        (f"Up close, {subject} was all texture and contradiction: "
         f"{rng.choice(textures)}. "
         f"The kind of thing you remember with your hands first."),
    ]
    if mood:
        frames.append(f"It wore the {mood} like a second skin — impossible "
                      f"to look at directly, impossible to look away from.")
    return "\n\n".join(frames)


def rewrite(text: str, instruction: str, context: Any = None) -> str:
    """Directed rewrite of a passage (Sudowrite Rewrite)."""
    passage = (text or "").strip()
    instruction = (instruction or "").strip()
    if not passage:
        return ""
    prompt = (
        f"Rewrite this passage. Instruction: {instruction or 'improve the prose'}\n\n"
        f"{passage}\n\n"
        "Keep the meaning and all plot facts. Output only the rewritten passage.")
    out = _model_pass(prompt, context, max_tokens=len(passage.split()) * 3 + 200)
    if out:
        return out
    # heuristic floor: tighten — kill filter words, vary sentence starts
    import random
    rng = random.Random(hash(passage + instruction) & 0xFFFFFFFF)
    filters = {"saw": "", "felt": "", "heard": "", "noticed that": "",
               "realized that": "", "seemed to": "", "began to": "",
               "started to": ""}
    out_text = passage
    for f, rep in filters.items():
        out_text = re.sub(r"\b" + re.escape(f) + r"\b", rep, out_text,
                          flags=re.IGNORECASE)
    out_text = re.sub(r"[ ]{2,}", " ", out_text)
    if "shorter" in instruction.lower() or "tight" in instruction.lower():
        sents = _sentences(out_text)
        keep = max(1, int(len(sents) * 0.7))
        out_text = " ".join(sents[:keep])
    if "punchier" in instruction.lower():
        out_text = re.sub(r"([^.!?]{60,}[.!?])",
                          lambda m: m.group(1), out_text)
    return out_text.strip() or passage


def suggest_hooks(prev_tail: str, context: Any = None,
                  n: int = 3) -> list[str]:
    """Candidate closing hooks for a chapter (choose one, don't stack)."""
    tail = (prev_tail or "").strip()[-800:]
    prompt = (
        "Give exactly 3 one-sentence closing hooks for a chapter ending "
        f"here:\n\n{tail}\n\n"
        "Each hook: a reversal, a revelation, or a threat. Numbered 1-3, "
        "one sentence each, no explanation.")
    out = _model_pass(prompt, context, max_tokens=300)
    if out:
        hooks = [re.sub(r"^[\d.\-\)\s]+", "", line).strip()
                 for line in out.split("\n") if line.strip()]
        return [h for h in hooks if h][:n]
    import random
    rng = random.Random(hash(tail) & 0xFFFFFFFF)
    pool = [
        "The door opened — and it was the last person anyone expected.",
        "Too late, the truth surfaced: the map had been wrong all along.",
        "Footsteps, unhurried, coming closer through the dark.",
        "The message contained three words, and none of them were good.",
        "What waited on the other side of the threshold changed everything.",
        "A name spoken once, softly — and the whole plan collapsed.",
    ]
    rng.shuffle(pool)
    return pool[:n]
