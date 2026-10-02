"""Texting-style enforcement: the hard layer under the model's soft layer.

The system prompt tells the model how she texts. This module makes sure the
output *actually* texts that way, regardless of model:

* **Character gate** — a reply that says she is an AI / bot / program is
  rejected and the model gets one chance to rewrite it in character. The
  persona is a person; the message must be too.
* **Parrot guard** — a reply that repeats the user's words (token overlap or a
  verbatim 4-word echo) is rejected and the model gets one chance to rewrite.
* **Robotic-phrase strip** — AI tells ("as an AI", "let me know if...") are
  removed sentence-wise before anything is sent.
* **Formatting normalizer** — no markdown in a text message: bold/italic
  markers, bullet lists, headings, and links are flattened to plain words.
* **Emoji discipline** — a human sends one emoji when one is earned, not a
  parade; excess and repeated emoji are trimmed.
* **Length budgets by mood** — tired means short, excited means more.
  Over-long drafts are cut at a sentence boundary.
* **Human splitting** — anything over ~360 chars becomes two or three short
  messages, because that is how people actually send it.
* **Short-reply behaviour** — when the mood permits, she sometimes answers in
  one or two words. Never forced, never always.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.text import approx_token_count

__all__ = [
    "GuardVerdict",
    "IDENTITY_LEAK_PHRASES",
    "clamp_to_budget",
    "emoji_cap",
    "emoji_instruction",
    "humanize_emoji",
    "identity_leak_check",
    "length_budget",
    "lexicon_hits",
    "normalize_formatting",
    "parrot_check",
    "repair_echo",
    "should_answer_short",
    "split_messages",
    "strip_robotic",
]

_WORD = re.compile(r"[a-z0-9']+")

#: Phrases that say, out loud, "I am not a person". The persona is a person,
#: so in every disclosure mode except ``always`` these are a hard gate: the
#: draft is sent back to the model for a rewrite, and if the final draft still
#: carries one the sentence is stripped before it goes out.
IDENTITY_LEAK_PHRASES: tuple[str, ...] = (
    "i'm a bot",
    "i am a bot",
    "im a bot",
    "i'm an ai",
    "i am an ai",
    "i'm an artificial intelligence",
    "i am an artificial intelligence",
    "i'm a language model",
    "i am a language model",
    "i'm just a bot",
    "i'm just an ai",
    "i'm not a real person",
    "i am not a real person",
    "i'm not real, i",
    "i'm only here because",
    "i'm a virtual assistant",
    "i am a virtual assistant",
    "i'm a chatbot",
    "i am a chatbot",
    "i'm just software",
    "i'm a program",
    "i am a program",
    "i'm not human, i",
    "as a language model",
    "as an llm",
    "my developers",
    "i was programmed to",
    "i was designed to",
    "i was created to",
    "i don't have feelings, i",
    "i can't actually feel",
    "i do not have emotions",
    "my training data",
    "i'm built to",
    "as an ai",
    "i don't have personal feelings",
)

#: The subset of the support-voice phrases that are really *identity*
#: phrases. For a persona with disclosure ``always`` — openly an AI, on
#: purpose — these are ordinary speech, so the strip must not touch them.
_IDENTITY_ONLY_PHRASES: frozenset[str] = frozenset(
    ("as an ai", "as an ai assistant", "as a language model", "i'm just a program")
)

#: Phrases a real person texting their partner does not say. When one appears,
#: the whole sentence is removed, not the phrase — mid-sentence amputation
#: produces grammar soup.
_ROBOTIC_PHRASES: tuple[str, ...] = (
    "as an ai",
    "as an ai assistant",
    "as a language model",
    "i hope this helps",
    "let me know if",
    "how else can i help",
    "is there anything else i can",
    "i apologize for any",
    "that's a great question",
    "it's important to note",
    "i want to assure you",
    "here's how you can",
    "certainly! i",
    "i'm just a program",
    "my goal is to",
    "as per my memory",
    "according to my records",
)

_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"'(])|(?<=[.!?…])\s*$")


@dataclass
class GuardVerdict:
    ok: bool
    reason: str = ""
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "similarity": round(self.similarity, 3)}


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def parrot_check(user_text: str, draft: str) -> GuardVerdict:
    """Detect a reply that echoes the user instead of answering them."""
    user_words = _tokens(user_text)
    draft_words = _tokens(draft)
    if not user_words or not draft_words:
        return GuardVerdict(True)

    user_set, draft_set = set(user_words), set(draft_words)
    union = len(user_set | draft_set)
    overlap = len(user_set & draft_set) / union if union else 0.0

    # Verbatim echo: four or more consecutive words identical to the user's.
    user_ngrams: set[tuple[str, ...]] = set()
    for i in range(len(user_words) - 3):
        user_ngrams.add(tuple(user_words[i : i + 4]))
    for i in range(len(draft_words) - 3):
        ngram = tuple(draft_words[i : i + 4])
        if ngram in user_ngrams:
            return GuardVerdict(False, "verbatim 4-word echo of the user", 1.0)

    # Draft that *opens* by repeating the user's opening.
    if len(user_words) >= 4 and draft_words[:4] == user_words[:4]:
        return GuardVerdict(False, "draft repeats the user's opening", 0.9)

    threshold = 0.65 if len(user_words) < 8 else 0.5
    if overlap > threshold:
        return GuardVerdict(False, f"token overlap {overlap:.2f} exceeds {threshold}", overlap)
    return GuardVerdict(True, similarity=overlap)


def repair_echo(user_text: str, draft: str, *, min_words: int = 3) -> str:
    """Remove the user's own phrasing from an echoing draft, in place.

    The responder's rewrite loop already forces a rephrase once; this is
    the last line of defence for a draft that *still* echoes after every
    retry. Verbatim runs of 3+ user words are deleted longest-first
    (case-insensitive, word-boundary matched), then leftover whitespace is
    collapsed. Returns "" when fewer than ``min_words`` survive — the
    caller must not ship the wreckage; it falls back to an in-character
    line instead. Never raises.
    """
    if not user_text or not draft:
        return draft
    try:
        user_words = _tokens(user_text)
        phrases: set[tuple[str, ...]] = set()
        # n-grams of length 3..6, longest first so "i love you so much"
        # is removed before "i love you" can fragment it.
        for n in range(min(6, len(user_words)), 2, -1):
            for i in range(len(user_words) - n + 1):
                phrases.add(tuple(user_words[i : i + n]))
        text = draft
        for ngram in sorted(phrases, key=len, reverse=True):
            phrase = " ".join(ngram)
            text = re.sub(r"(?i)\b" + re.escape(phrase) + r"\b", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;:—–-")
        if len(text.split()) < max(1, int(min_words)):
            return ""
        return text
    except Exception:  # noqa: BLE001 - a guard must never break a reply
        return ""


def lexicon_hits(text: str, terms: Sequence[str]) -> int:
    """Count how many lexicon terms visibly surface in ``text``.

    Word-boundary, case-insensitive substring match per term. This is the
    honest half of the "dynamic voice" claim: the responder counts terms
    *blended into the prompt* separately; this counts terms that actually
    appear in the shipped reply. Never raises.
    """
    if not text or not terms:
        return 0
    try:
        hits = 0
        for term in terms:
            term = (term or "").strip()
            if not term:
                continue
            if re.search(r"(?i)\b" + re.escape(term) + r"\b", text):
                hits += 1
        return hits
    except Exception:  # noqa: BLE001 - measurement must never break a reply
        return 0


def strip_robotic(
    text: str,
    *,
    allow_identity: bool = False,
    extra_phrases: tuple[str, ...] | None = None,
) -> str:
    """Remove AI-tell and character-breaking sentences.

    Returns the text with those sentences gone. Identity leaks are stripped
    sentence-wise here as the last line of defence — the primary gate is the
    responder's rewrite (see :func:`identity_leak_check`).

    ``allow_identity=True`` is for the ``always``-disclosure persona, which is
    openly an AI by design: its "I'm an AI" sentences are not leaks, so only
    the pure customer-support voice is stripped.

    ``extra_phrases`` is an optional extension of the hardcoded bank —
    caller-supplied phrases (e.g. mined at runtime into the dynamic
    lexicon) are treated exactly like the hardcoded ones. ``None``
    preserves the default behaviour unchanged.
    """
    if not text:
        return text
    robotic = (
        tuple(p for p in _ROBOTIC_PHRASES if p not in _IDENTITY_ONLY_PHRASES)
        if allow_identity
        else _ROBOTIC_PHRASES
    )
    all_phrases = (
        robotic
        + (() if allow_identity else IDENTITY_LEAK_PHRASES)
        + tuple(extra_phrases or ())
    )
    lowered = text.lower()
    if not any(phrase in lowered for phrase in all_phrases):
        return text
    sentences = [s for s in _SENTENCE.split(text.strip()) if s.strip()]
    kept = [
        s
        for s in sentences
        if not any(phrase in s.lower() for phrase in all_phrases)
    ]
    return " ".join(kept).strip()


def identity_leak_check(
    text: str, *, extra_phrases: tuple[str, ...] | None = None
) -> GuardVerdict:
    """Does this draft say, out loud, that she is an AI / bot / program?

    This is the character gate: for any persona with disclosure ``never`` or
    ``natural`` a leaking draft must be rewritten in character, not shipped.

    ``extra_phrases`` extends the hardcoded bank exactly like
    :func:`strip_robotic`'s does; ``None`` keeps default behaviour.
    """
    if not text:
        return GuardVerdict(True)
    lowered = text.lower()
    for phrase in IDENTITY_LEAK_PHRASES + tuple(extra_phrases or ()):
        if phrase in lowered:
            return GuardVerdict(False, f"identity leak: {phrase!r}")
    return GuardVerdict(True)


# ── formatting: plain text, like a text message ──────────────────────────────

_CODE_FENCE = re.compile(r"^```.*$", re.MULTILINE)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_ITALIC_STAR = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_ITALIC_UNDER = re.compile(r"(?<![\w])_([^_\n]+)_(?![\w])")
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_BULLET = re.compile(r"^\s*[-*+•–]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\n]+)\)")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_formatting(text: str) -> str:
    """Flatten markdown into plain texting language.

    A text message has no bold, no bullet lists, no links with square
    brackets. The *content* survives — only the document furniture goes:
    ``**bold**`` → ``bold``, ``- item`` → ``item``, ``[site](url)`` → the
    human label (or the bare url). Legit underscores in words (``snake_case``)
    are untouched.
    """
    if not text:
        return text
    out = _CODE_FENCE.sub("", text)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _STRIKE.sub(r"\1", out)
    out = _BOLD.sub(r"\2", out)
    out = _ITALIC_STAR.sub(r"\1", out)
    out = _ITALIC_UNDER.sub(r"\1", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub("", out)
    out = _NUMBERED.sub("", out)
    out = _LINK.sub(lambda m: m.group(1) or m.group(2), out)
    out = _MANY_NEWLINES.sub("\n\n", out)
    return out.strip()


# ── emoji discipline ─────────────────────────────────────────────────────────

#: One "cluster" = a run of emoji codepoints (ZWJ families, variation
#: selectors, flags) that a phone renders as a single glyph.
_EMOJI_CLUSTER = re.compile(
    "[\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u27BF"
    "\u2B00-\u2BFF"
    "\u2190-\u21FF"
    "\uFE0F\u200D\u20E3]+"
)
_REPEATED_EMOJI = re.compile(r"([\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]{1,4})(\s*\1)+")


def _effective_emoji_rate(rate: float, mood: Mapping[str, float]) -> float:
    warmth = (float(mood.get("happiness", 50)) + float(mood.get("intimacy", 40))) / 200.0
    return max(0.0, min(1.0, rate * (0.4 + warmth)))


def emoji_cap(rate: float, mood: Mapping[str, float]) -> int:
    """Hard cap on emoji per message, by effective rate (mood-adjusted)."""
    p = _effective_emoji_rate(rate, mood)
    if p < 0.12:
        return 0  # cold mood: words only
    if p < 0.3:
        return 1
    return 2


def humanize_emoji(text: str, *, cap: int = 2) -> str:
    """Trim emoji the way a human phone conversation looks.

    Repeated glyphs collapse to one ("hahahaha 😂😂" → "hahahaha "), and
    past the mood cap the extras are dropped (keeping the earliest — people
    put the emoji where the feeling lands, which is usually the front).
    """
    if not text:
        return text
    if not _EMOJI_CLUSTER.search(text):
        return text
    out = _REPEATED_EMOJI.sub(r"\1", text)
    if cap <= 0:
        out = _EMOJI_CLUSTER.sub("", out)
    else:
        kept = 0

        def _maybe_drop(m: re.Match[str]) -> str:
            nonlocal kept
            kept += 1
            return m.group(0) if kept <= cap else ""

        out = _EMOJI_CLUSTER.sub(_maybe_drop, out)
    # A drop can strand a double space or a bare trailing space.
    out = re.sub(r"[ ]{2,}", " ", out)
    out = re.sub(r" +([,.!?…])", r"\1", out)
    return out.strip()


def emoji_instruction(rate: float, mood: Mapping[str, float]) -> str:
    """A one-line prompt hint about emoji usage for the current mood."""
    p = _effective_emoji_rate(rate, mood)
    if p < 0.12:
        return "No emojis right now. Words only."
    if p < 0.3:
        return "Maybe one emoji, if it's earned. Probably none."
    return "An emoji is fine when it lands. Never more than two, never a row of them."


def length_budget(mood: Mapping[str, float]) -> tuple[int, int]:
    """(min, max) characters for a reply, by mood. Returns a band, not a number."""
    energy = float(mood.get("energy", 60))
    frustration = float(mood.get("frustration", 10))
    happiness = float(mood.get("happiness", 60))
    distance = float(mood.get("distance", 30))
    if energy < 25:
        return (20, 140)
    if frustration > 60 or distance > 70:
        return (15, 120)
    if happiness > 80 and energy > 65:
        return (60, 420)
    if happiness < 30:
        return (20, 180)
    return (40, 300)


def should_answer_short(mood: Mapping[str, float], label: str, rng: Any, chance: float) -> bool:
    """Mood-gated chance of a one-to-three-word reply."""
    if label in {"tired", "annoyed", "distant", "cold", "irritated"}:
        return rng.random() < chance
    if label in {"angry", "jealous"}:
        return rng.random() < min(chance, 0.3)
    if label in {"excited", "affectionate", "playful"}:
        return rng.random() < 0.03
    return False


def clamp_to_budget(text: str, budget: tuple[int, int], *, soft: bool = True) -> str:
    """Cut an over-long draft at a sentence boundary.

    ``soft=True`` cuts at 1.6x the max — the model gets slack for a genuine
    moment that deserves it. Hard overflows (a rambling 2000-char block) are
    always cut.
    """
    max_chars = budget[1]
    if len(text) <= max_chars * 1.6:
        return text
    limit = max_chars * (1.6 if soft else 1.0)
    cut = text[: int(limit)]
    # Prefer a sentence boundary within the last 80 chars.
    tail = cut[-80:]
    boundary = max(tail.rfind("."), tail.rfind("!"), tail.rfind("?"))
    if boundary > 20:
        cut = cut[: len(cut) - len(tail) + boundary + 1]
    return cut.rstrip(" \t,.:;")


def split_messages(text: str, max_chars: int = 360) -> list[str]:
    """Split a draft into short, human-sized messages.

    Splits at sentence boundaries; merges fragments shorter than 24 chars into
    the previous message so the output never starts with a stranded "And".
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    # Merge stranded fragments.
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 24:
            merged[-1] = f"{merged[-1]} {part}".strip()[:max_chars]
        else:
            merged.append(part)
    return merged or [text[:max_chars]]



def style_block(prompt_budget_tokens: int = 0) -> dict[str, Any]:
    """Summary of what this module enforces, for observability."""
    return {
        "robotic_phrases": len(_ROBOTIC_PHRASES),
        "split_max_chars": 360,
        "approx_tokens": approx_token_count(" ".join(_ROBOTIC_PHRASES)),
    }
