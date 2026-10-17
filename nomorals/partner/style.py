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
    "MINED_SECTION_MARKERS",
    "StyleMeasurement",
    "StyleProfile",
    "VOICE_THEMES",
    "apply_texting_voice",
    "apply_voice_theme",
    "clamp_to_budget",
    "emoji_cap",
    "emoji_instruction",
    "format_reply",
    "humanize_emoji",
    "identity_leak_check",
    "length_budget",
    "lexicon_hits",
    "measure_style",
    "normalize_formatting",
    "parrot_check",
    "platform_hard_cap",
    "purposeful_emoji",
    "repair_echo",
    "should_answer_short",
    "split_messages",
    "strip_assistant_framing",
    "strip_robotic",
    "style_for_chat",
    "style_prompt_lines",
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

#: Sentences that frame the reply as an assistant completing a task
#: instead of a person talking. In personal (restricted) chats Devon is a
#: person — "Devon Personal" — not support. These patterns break human
#: presence, so the whole sentence is removed (same rule as
#: _ROBOTIC_PHRASES: mid-sentence amputation produces grammar soup).
#: Distinct from _ROBOTIC_PHRASES (stock support phrases): these are
#: task-framing patterns the model generates fresh — "gotcha, what do
#: you need?", "drop it here and i'll take a look", "if you want me to
#: ID it, could you upload the audio file here?". Also covers FALSE
#: limitations: Devon CAN send voice notes (TTS), so a claim that it
#: can only reply in text must never ship.
ASSISTANT_FRAMING_PATTERNS: tuple[str, ...] = (
    # Task-framed openers / offers — "how can I help" in sentence form.
    "what do you need",
    "how can i help",
    "how can i assist",
    "what can i do for you",
    "how may i help",
    "is there anything i can do",
    "let me know what you need",
    # File/clip task framing — "upload it here and I'll handle it".
    "drop it here",
    "upload the audio",
    "upload it here",
    "send it here",
    "drop it in here",
    "if you've got a file",
    "if you have a file",
    "if you want me to id",
    "want me to id it",
    "if you want me to identify",
    # Assistant conditionals — "if you'd like me to [do a task]".
    "if you want me to",
    "if you'd like me to",
    "would you like me to",
    "i'll take a look",
    # False limitations — Devon CAN send voice notes (TTS). Never claim
    # otherwise; a person doesn't announce what they can't do, and a
    # false "I can't" is worse than silence.
    "i can only reply in text",
    "i can only send text",
    "i can't send voice notes",
    "i can't reply with a voice note",
    "i can't send a voice note",
    "only reply in text",
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


def _history_ngrams(history_texts: Sequence[str] | None, n: int) -> set[tuple[str, ...]]:
    """Verbatim n-grams across earlier user turns, as one deduped set.

    ``None`` / empty means "no history was given" — the caller keeps its
    latest-turn-only behaviour byte-for-byte.
    """
    ngrams: set[tuple[str, ...]] = set()
    for text in history_texts or ():
        words = _tokens(text)
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i : i + n]))
    return ngrams


def parrot_check(
    user_text: str, draft: str, *, history_texts: Sequence[str] | None = None
) -> GuardVerdict:
    """Detect a reply that echoes the user instead of answering them.

    ``history_texts`` is the conversation's earlier user turns (oldest to
    newest). A draft that repeats a verbatim 4-word run from ANY earlier
    turn — not just the latest message — is still an echo: without this the
    guard degrades as the thread grows, because a model that lifts a phrase
    from turn 3 of a 60-turn thread passes a latest-turn-only check. The
    token-overlap ratios stay computed against the latest turn only, so a
    long thread cannot dilute the tuned thresholds.
    """
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
    if history_texts:
        # History-length-independent: the bank covers every earlier user
        # turn the responder was given, not just the recent window.
        user_ngrams |= _history_ngrams(history_texts, 4)
    for i in range(len(draft_words) - 3):
        ngram = tuple(draft_words[i : i + 4])
        if ngram in user_ngrams:
            source = "an earlier user turn" if history_texts else "the user"
            return GuardVerdict(False, f"verbatim 4-word echo of {source}", 1.0)

    # Draft that *opens* by repeating the user's opening.
    if len(user_words) >= 4 and draft_words[:4] == user_words[:4]:
        return GuardVerdict(False, "draft repeats the user's opening", 0.9)

    threshold = 0.65 if len(user_words) < 8 else 0.5
    if overlap > threshold:
        return GuardVerdict(False, f"token overlap {overlap:.2f} exceeds {threshold}", overlap)
    return GuardVerdict(True, similarity=overlap)


def repair_echo(
    user_text: str,
    draft: str,
    *,
    min_words: int = 3,
    history_texts: Sequence[str] | None = None,
) -> str:
    """Remove the user's own phrasing from an echoing draft, in place.

    The responder's rewrite loop already forces a rephrase once; this is
    the last line of defence for a draft that *still* echoes after every
    retry. Verbatim runs of 3+ user words are deleted longest-first
    (case-insensitive, word-boundary matched), then leftover whitespace is
    collapsed. ``history_texts`` extends the strip bank to the
    conversation's earlier user turns — a draft echoing a phrase from turn
    3 of a 60-turn thread is repaired exactly like one echoing the latest
    message, so the repair is history-length-independent. Returns "" when
    fewer than ``min_words`` survive — the caller must not ship the
    wreckage; it falls back to an in-character line instead. Never raises.
    """
    if not user_text or not draft:
        return draft
    try:
        corpora = [user_text, *(t for t in (history_texts or ()) if t)]
        phrases: set[tuple[str, ...]] = set()
        # n-grams of length 3..6, longest first so "i love you so much"
        # is removed before "i love you" can fragment it.
        for corpus in corpora:
            corpus_words = _tokens(corpus)
            for n in range(min(6, len(corpus_words)), 2, -1):
                for i in range(len(corpus_words) - n + 1):
                    phrases.add(tuple(corpus_words[i : i + n]))
        text = draft
        # Sort longest-first AND alphabetically within a length: set iteration
        # order is hash-seed dependent, and removal order within one length
        # changes which fragments survive. A guard must be deterministic.
        for ngram in sorted(phrases, key=lambda ng: (-len(ng), ng)):
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


def strip_assistant_framing(text: str) -> str:
    """Remove assistant/task-framed sentences (presence guard).

    Applied to personal (restricted) chats, where Devon is a person —
    "Devon Personal" — not an assistant. Sentences matching
    :data:`ASSISTANT_FRAMING_PATTERNS` ("gotcha, what do you need?",
    "drop it here and i'll take a look", "if you want me to ID it…",
    "i can only reply in text…") are removed whole; the rest of the
    reply survives. Returns ``""`` when every sentence was stripped —
    the caller falls back to an in-character line, never ships an
    assistant sentence.

    Owner chats are NOT stripped: there Devon is the owner's assistant
    by design, and task framing is correct there.
    """
    if not text:
        return text
    lowered = text.lower()
    if not any(p in lowered for p in ASSISTANT_FRAMING_PATTERNS):
        return text
    sentences = [s for s in _SENTENCE.split(text.strip()) if s.strip()]
    kept = [
        s
        for s in sentences
        if not any(p in s.lower() for p in ASSISTANT_FRAMING_PATTERNS)
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


# ── mined style system: dynamic, per-chat, no tone presets ─────────────────
# Replaces the old VOICE_THEMES preset table (removed 2026-10-17: named
# tone presets are invented voice, not mined voice). Everything below is
# derived from three real sources — see STYLE_MINING.md at the repo root
# for the full mining log with counts and citations:
#
# 1. Devon's OWN code (nomorals/social/chat/style.py, tgbot_buttons.py,
#    action_footers.py, telegram_menu.py): emoji section headers, <code>
#    commands, the 🎯 CTA, ━━━ dividers, progress bars, one job per
#    screen. The marker vocabulary in MINED_SECTION_MARKERS is counted
#    from actual usage in that code, not invented.
# 2. Platform facts (nomorals/social/chat/platform_limits.py + the
#    Telegram Bot API formatting docs, core.telegram.org/bots/api):
#    per-message hard caps (TG 4096, WA 4000, Discord 2000, SMS 160),
#    HTML on Telegram, *bold* / _italic_ / ```mono``` on WhatsApp,
#    plain text on SMS. No inline buttons / HTML on WhatsApp.
# 3. The chat's OWN observed messages (measure_style): median length,
#    emoji rate, question rate, caps ratio, list rate. When a chat has
#    history, the profile follows that chat — no preset can do that.
#
# What was NOT mined, honestly: no real owner messages exist anywhere
# in this repo (fixtures and tests are synthetic), and the first-edition
# repo (Oluwacutyp/No-morals-ai) is unreachable read-only — GitHub API
# returns 404 unauthenticated and git prompts for credentials — so no
# owner voice and no first-edition conventions were copied. Nothing here
# pretends otherwise: with no observed sample, profiles fall back to the
# audience + platform defaults below, which come from gating.py's
# documented per-audience rules (owner = full, private = short +
# guarded, group = shortest, no lists, no paragraphs).

#: Section-kind -> emoji marker, counted from Devon's own output code.
#: Only these markers may open a section header: an emoji anywhere else
#: is decoration, not structure. Counts are uses across
#: nomorals/social/chat + nomorals/partner (Oct 2026 mining pass):
#: 🎮 games 19, ✅ success/done 11, 📊 stats 11, 📋 lists/tasks 11,
#: ❌ failure/cancel 8, 🎯 next-action CTA (style.cta), 🔍 search 4,
#: 💡 tip (style.card footer), ✨ header (style.header/menu),
#: 🎵 music/track 4, 🎬 video 3, 🎙️ voice 5, 📖 books/reading 2,
#: ⚠️ warning 15, 📣 announcement 4, 📌 pinned 2, 🏆 results 2.
#: Greeting/closing patterns and owner phrasing are deliberately NOT in
#: this table — no real owner messages were available to mine them from.
MINED_SECTION_MARKERS: dict[str, str] = {
    "games": "🎮",
    "done": "✅",
    "stats": "📊",
    "list": "📋",
    "failed": "❌",
    "next": "🎯",
    "search": "🔍",
    "tip": "💡",
    "header": "✨",
    "music": "🎵",
    "video": "🎬",
    "voice": "🎙️",
    "reading": "📖",
    "warning": "⚠️",
    "announce": "📣",
    "pinned": "📌",
    "results": "🏆",
}

#: Hardcoded tone presets: REMOVED. Kept as an empty dict so existing
#: imports (``from .style import VOICE_THEMES``) do not crash; it carries
#: no entries and :func:`apply_voice_theme` is a passthrough. Style now
#: comes from :func:`style_for_chat`, computed per chat from audience +
#: platform + mood + the chat's own observed messages.
VOICE_THEMES: dict[str, dict[str, Any]] = {}


def apply_voice_theme(speech: Any, theme: str) -> Any:
    """Deprecated passthrough (tone presets removed — see above).

    Returns ``speech`` unchanged for any ``theme``. Kept so old callers
    do not crash; new code uses :func:`style_for_chat`.
    """
    return speech


#: Fallback hard caps per platform, mirroring
#: nomorals/social/chat/platform_limits.py (single source of truth — this
#: table is only the fallback when that module cannot be imported).
_PLATFORM_HARD_CAPS: dict[str, int] = {
    "telegram": 4096,
    "telegram-bot": 4096,
    "telegram-personal": 4096,
    "whatsapp": 4000,
    "discord": 2000,
    "sms": 160,
    "web": 8000,
    "local": 8000,
}


def platform_hard_cap(platform: str) -> int:
    """Hard per-message cap for a platform (never raises)."""
    try:
        from ..social.chat.platform_limits import for_platform

        return int(for_platform(platform or "telegram").max_text)
    except Exception:  # noqa: BLE001 - fallback table is the same numbers
        return _PLATFORM_HARD_CAPS.get(
            (platform or "telegram").strip().lower(), 4096
        )


@dataclass(frozen=True)
class StyleMeasurement:
    """What a chat's OWN messages actually look like — measured, not assumed.

    Computed by :func:`measure_style` over observed messages from that
    chat. All rates are 0..1; ``n`` is the sample size (0 = no data).
    """

    n: int = 0
    median_chars: float = 0.0
    mean_chars: float = 0.0
    emoji_per_msg: float = 0.0
    caps_ratio: float = 0.0
    question_rate: float = 0.0
    exclaim_rate: float = 0.0
    list_rate: float = 0.0
    lowercase_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "median_chars": round(self.median_chars, 1),
            "mean_chars": round(self.mean_chars, 1),
            "emoji_per_msg": round(self.emoji_per_msg, 3),
            "caps_ratio": round(self.caps_ratio, 3),
            "question_rate": round(self.question_rate, 3),
            "exclaim_rate": round(self.exclaim_rate, 3),
            "list_rate": round(self.list_rate, 3),
            "lowercase_ratio": round(self.lowercase_ratio, 3),
        }


def measure_style(texts: Sequence[str] | None) -> StyleMeasurement:
    """Measure a chat's style from its own observed messages.

    No presets, no dictionaries of "how people talk": median length,
    emoji density, caps/question/exclamation/list rates straight from
    the sample. Empty / tiny samples return ``n=0`` measurements — the
    caller falls back to audience + platform defaults, never to an
    invented voice. Never raises.
    """
    try:
        sample = [str(t or "") for t in (texts or []) if str(t or "").strip()]
        if not sample:
            return StyleMeasurement()
        lengths = sorted(len(t) for t in sample)
        mid = len(lengths) // 2
        median = (
            float(lengths[mid])
            if len(lengths) % 2
            else (lengths[mid - 1] + lengths[mid]) / 2.0
        )
        emoji_counts = [len(_EMOJI_CLUSTER.findall(t)) for t in sample]
        caps = 0
        total_alpha = 0
        questions = sum(1 for t in sample if "?" in t)
        exclaims = sum(1 for t in sample if "!" in t)
        listed = sum(
            1
            for t in sample
            if re.search(r"(?m)^\s*([-*+•–]|\d+[.)])\s+\S", t)
        )
        lowered = sum(1 for t in sample if t == t.lower() and any(c.isalpha() for c in t))
        for t in sample:
            letters = [c for c in t if c.isalpha()]
            total_alpha += len(letters)
            caps += sum(1 for c in letters if c.isupper())
        return StyleMeasurement(
            n=len(sample),
            median_chars=median,
            mean_chars=sum(lengths) / len(lengths),
            emoji_per_msg=sum(emoji_counts) / len(sample),
            caps_ratio=(caps / total_alpha) if total_alpha else 0.0,
            question_rate=questions / len(sample),
            exclaim_rate=exclaims / len(sample),
            list_rate=listed / len(sample),
            lowercase_ratio=lowered / len(sample),
        )
    except Exception:  # noqa: BLE001 - measurement never breaks a reply
        return StyleMeasurement()


@dataclass(frozen=True)
class StyleProfile:
    """The computed style for ONE reply in ONE chat.

    Fields are numbers and booleans, never a tone name: the responder
    formats with these, and :meth:`prompt_lines` tells the model the
    same numbers so the soft layer and the hard layer agree.
    """

    audience: str  # "owner" | "private" | "group"
    platform: str
    max_chars: int  # hard cap: over this, format_reply truncates
    target_chars: int  # soft target: aim here or shorter
    emoji_budget: int  # max emoji clusters kept per message (0..3)
    allow_sections: bool
    allow_lists: bool
    allow_markup: bool
    measured: StyleMeasurement = StyleMeasurement()

    def prompt_lines(self) -> list[str]:
        """Style instructions for the model, in numbers, not tone words."""
        lines = [
            f"Reply length: aim at or under {self.target_chars} characters "
            f"(hard cap {self.max_chars}). Short by default; longer only "
            f"when the content needs it.",
        ]
        if self.emoji_budget <= 0:
            lines.append(
                "Emoji: none in this reply. Use words, not symbols."
            )
        else:
            lines.append(
                f"Emoji: at most {self.emoji_budget} purposeful marker(s), "
                f"each opening a section or marking a result "
                f"(✅ done, ❌ failed, 🎮 games, 🎵 music, 📊 stats). "
                f"Never a row of emoji, never decoration."
            )
        if not self.allow_lists:
            lines.append("Lists: none. Say it in flowing sentences.")
        else:
            lines.append("Lists: short ones only, when the content is a list.")
        if not self.allow_sections:
            lines.append("Sections: none. One short message, no headers.")
        if not self.allow_markup:
            lines.append("Formatting: plain text only, no markdown or HTML.")
        if self.measured.n >= 5:
            lines.append(
                f"This chat's own messages run about "
                f"{int(self.measured.median_chars)} characters "
                f"(median of {self.measured.n}); match that pace."
            )
        return lines


def style_for_chat(
    *,
    audience: str = "owner",
    platform: str = "telegram",
    mood_values: Mapping[str, float] | None = None,
    observed: Sequence[str] | None = None,
    content_kind: str = "chat",
) -> StyleProfile:
    """Compute the style profile for one reply — dynamically.

    Inputs, in priority order:
    1. **audience** (from gating: owner / private / group) — sets the
       structural rules: groups get no sections/lists and the shortest
       cap (gating.py: "no paragraphs, no lists, no explanations");
       private chats get short + guarded; owner DMs get the full range.
    2. **platform** — hard cap from platform_limits; markup off where
       the platform cannot render it (SMS plain, WhatsApp no HTML).
    3. **mood** — the existing :func:`length_budget` band + emoji cap
       math; low energy shortens, high frustration zeroes emoji.
    4. **observed** — the chat's own messages (measured). With >= 5
       samples the target length bends toward that chat's median and
       the emoji budget toward its emoji rate. This is the dynamic
       part: two chats with different histories get different profiles
       from identical audience/platform/mood.
    5. **content_kind** — ``"structured"`` (menus, reports, results)
       allows sections/lists and a larger target for the owner; ``"chat"``
       (default) keeps texting discipline.

    No tone names anywhere in this function: there is nothing to
    misspell, nothing to rotate, nothing the owner cannot trace to a
    measurement. Never raises.
    """
    try:
        aud = (audience or "owner").strip().lower()
        if aud not in ("owner", "private", "group"):
            aud = "owner"
        plat = (platform or "telegram").strip().lower()
        mood = dict(mood_values or {})
        measured = measure_style(observed)
        hard = platform_hard_cap(plat)
        structured = (content_kind or "chat").strip().lower() == "structured"

        # Base from mood (the same band the rest of the pipeline uses).
        _mn, mood_max = length_budget(mood)
        base_target = mood_max
        # Emoji budget from mood: reuse emoji_cap with the persona-neutral
        # rate 0.5 so mood alone decides; observed data refines it below.
        base_emoji = emoji_cap(0.5, mood)

        # Audience structural rules (from gating.py's documented modes).
        if aud == "group":
            allow_sections, allow_lists = False, False
            target = min(base_target, 280)
            # Groups stay one short burst: hard cap 360 (one split_messages
            # chunk), not the mood slack the owner gets.
            max_chars = min(360, hard)
            emoji_budget = min(base_emoji, 1)
            allow_markup = plat in ("telegram", "telegram-bot", "discord", "web")
        elif aud == "private":
            allow_sections, allow_lists = False, True
            target = min(base_target, 400)
            max_chars = min(max(int(mood_max * 1.6), 400), hard)
            emoji_budget = min(base_emoji, 2)
            allow_markup = plat in ("telegram", "telegram-bot", "discord", "web")
        else:  # owner
            allow_sections = structured
            allow_lists = True
            target = base_target * (3 if structured else 1)
            max_chars = min(max(int(mood_max * 1.6), target), hard)
            emoji_budget = base_emoji
            allow_markup = plat in ("telegram", "telegram-bot", "discord", "web")

        # Observed chat history bends the numbers (needs >= 5 samples so
        # one long message cannot yank the profile).
        if measured.n >= 5:
            # Target: halfway between the audience default and the chat's
            # own median, clamped to [80, max_chars].
            target = int(
                max(80, min(max_chars, (target + measured.median_chars) / 2))
            )
            # Emoji budget: follow the chat's own density, clamped 0..3.
            if measured.emoji_per_msg < 0.1:
                emoji_budget = 0
            elif measured.emoji_per_msg < 0.6:
                emoji_budget = min(emoji_budget, 1)
            else:
                emoji_budget = max(emoji_budget, min(3, round(measured.emoji_per_msg)))
            emoji_budget = max(0, min(3, emoji_budget))

        # High frustration kills emoji outright (matches apply_texting_voice).
        if float(mood.get("frustration", 10)) > 60:
            emoji_budget = 0

        # SMS is plain text, always; WhatsApp never gets HTML.
        if plat in ("sms", "twilio"):
            allow_markup = False
            allow_sections = False
            max_chars = min(max_chars, 160)
            target = min(target, 160)

        return StyleProfile(
            audience=aud,
            platform=plat,
            max_chars=int(max(80, max_chars)),
            target_chars=int(max(40, min(target, max_chars))),
            emoji_budget=int(max(0, min(3, emoji_budget))),
            allow_sections=bool(allow_sections),
            allow_lists=bool(allow_lists),
            allow_markup=bool(allow_markup),
            measured=measured,
        )
    except Exception:  # noqa: BLE001 - style never breaks a reply
        return StyleProfile(
            audience="owner",
            platform="telegram",
            max_chars=640,
            target_chars=300,
            emoji_budget=1,
            allow_sections=False,
            allow_lists=True,
            allow_markup=False,
        )


def style_prompt_lines(profile: StyleProfile) -> list[str]:
    """Prompt lines for a computed profile (soft layer matches hard layer)."""
    try:
        return list(profile.prompt_lines())
    except Exception:  # noqa: BLE001
        return []


_SECTION_HEADER = re.compile(
    r"^\s*([" + "".join(re.escape(m) for m in MINED_SECTION_MARKERS.values()) + r"])"
    r"\s*(?:<b>)?([^<\n]{1,60})(?:</b>)?\s*$"
)
_DIVIDER_LINE = re.compile(r"^\s*[─━═\-_=*·•]{4,}\s*$")
_BULLET_LINE = re.compile(r"^\s*[-*+•–]\s+")
_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+")


def purposeful_emoji(text: str, budget: int) -> str:
    """Keep emoji purposeful: markers and earned reactions only.

    Rules (all mined — see module header):
    * repeated runs collapse to one (``😂😂😂`` → ``😂``);
    * past ``budget``, the extras are dropped, earliest kept (people put
      the emoji where the feeling lands);
    * a cluster of 3+ emoji with no words between them is decoration —
      keep the first, drop the rest.
    Never raises.
    """
    if not text:
        return text
    try:
        # Collapse runs of the SAME single-codepoint emoji first: the
        # inherited _REPEATED_EMOJI pair-regex only halves a run
        # ("😂😂😂😂" -> "😂😂"), it does not collapse it to one.
        out = re.sub(
            r"([\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF])\1+",
            r"\1",
            text,
        )
        out = _REPEATED_EMOJI.sub(r"\1", out)
        # Decorative run: 3+ adjacent clusters (allow one space between).
        out = re.sub(
            r"((?:[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]\s*){3,})",
            lambda m: _EMOJI_CLUSTER.findall(m.group(0))[0]
            if _EMOJI_CLUSTER.findall(m.group(0))
            else "",
            out,
        )
        if budget <= 0:
            out = _EMOJI_CLUSTER.sub("", out)
        else:
            kept = 0

            def _maybe_drop(m: re.Match[str]) -> str:
                nonlocal kept
                kept += 1
                return m.group(0) if kept <= budget else ""

            out = _EMOJI_CLUSTER.sub(_maybe_drop, out)
        out = re.sub(r"[ ]{2,}", " ", out)
        out = re.sub(r" +([,.!?…])", r"\1", out)
        return out.strip()
    except Exception:  # noqa: BLE001 - a guard must never break a reply
        return text


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cut at a sentence boundary at or under ``max_chars`` + ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if boundary > max_chars // 3:
        return cut[: boundary + 1].rstrip()
    # No sentence boundary worth using: cut at the last space.
    space = cut.rfind(" ")
    if space > max_chars // 2:
        return cut[:space].rstrip() + "…"
    return cut + "…"


def format_reply(text: str, profile: StyleProfile) -> str:
    """Apply a computed :class:`StyleProfile` to a finished draft.

    The hard layer: dividers normalized (or dropped when sections are
    off), lists flattened when lists are off, section headers kept only
    when sections are on (and only with a mined marker), emoji held to
    the profile budget, over-long output truncated — structured output
    drops whole trailing sections first, then sentence-cuts, so a reply
    never ends mid-section. Returns "" only for empty input. Never raises.
    """
    if not text:
        return text
    try:
        lines = [ln.rstrip() for ln in str(text).splitlines()]
        # Collapse 3+ blank lines to one blank line.
        squashed: list[str] = []
        blanks = 0
        for ln in lines:
            if not ln.strip():
                blanks += 1
                if blanks <= 1:
                    squashed.append("")
                continue
            blanks = 0
            squashed.append(ln)
        lines = squashed

        out_lines: list[str] = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                out_lines.append("")
                continue
            if _DIVIDER_LINE.match(stripped):
                if profile.allow_sections:
                    out_lines.append("━━━━━━━━━━━━━━━")
                # else: drop divider-only lines entirely
                continue
            if _SECTION_HEADER.match(stripped) and not profile.allow_sections:
                # Header line without sections: keep the words, drop the
                # marker + any bold tags.
                m = _SECTION_HEADER.match(stripped)
                words = re.sub(r"</?b>", "", m.group(2)) if m else stripped
                out_lines.append(words.strip())
                continue
            if not profile.allow_lists and (
                _BULLET_LINE.match(ln) or _NUMBERED_LINE.match(ln)
            ):
                # Flatten list items into flowing text (group chats).
                item = _BULLET_LINE.sub("", ln)
                item = _NUMBERED_LINE.sub("", item)
                out_lines.append(item.strip())
                continue
            out_lines.append(ln)
        # When lists are off, flowing lines stay separate lines — the
        # caller (split_messages) handles chunking; joining here would
        # merge unrelated sentences into one wall.
        out = "\n".join(out_lines).strip()
        out = re.sub(r"\n{3,}", "\n\n", out)

        # Emoji discipline after structure: markers kept within budget.
        out = purposeful_emoji(out, profile.emoji_budget)

        # Truncation: structured output drops whole trailing sections
        # first (a reply that ends mid-section reads broken).
        if len(out) > profile.max_chars:
            if profile.allow_sections:
                sections = re.split(r"(?m)(?=^━━━━━━━━━━━━━━━$)", out)
                kept: list[str] = []
                total = 0
                for sec in sections:
                    if kept and total + len(sec) > profile.max_chars:
                        break
                    kept.append(sec)
                    total += len(sec)
                candidate = "".join(kept).strip()
                if candidate and len(candidate) >= profile.max_chars // 2:
                    out = candidate
            if len(out) > profile.max_chars:
                out = _truncate_at_sentence(out, profile.max_chars)
        return out
    except Exception:  # noqa: BLE001 - formatting never breaks a reply
        return text


def apply_texting_voice(
    draft: str,
    speech: Any,
    mood_values: Mapping[str, float],
    label: str = "",
    rng: Any = None,
) -> str:
    """Apply the persona's texting profile as deterministic post-processing.

    The prompt *asks* the model to text like her; this *makes* the output
    text like her regardless of model:

    * **lowercase runs** — with ``lowercase_bias`` probability (raised when
      tired/annoyed/distant, lowered when excited) the draft goes lowercase,
      the way thumbs actually type;
    * **text-speak** — each configured substitution applies with small
      probability, never all of them (a substitution in every message is a
      costume);
    * **tired punctuation** — low energy drifts sentence ends toward "…";
    * **high-frustration flatness** — high frustration strips exclamation
      marks (she doesn't exclaim when she's mad, she goes flat).

    Seeded RNG in, same text out. Never raises.
    """
    if not draft:
        return draft
    try:
        import random as _random
        rng = rng or _random
        energy = float(mood_values.get("energy", 60))
        frustration = float(mood_values.get("frustration", 10))
        distance = float(mood_values.get("distance", 30))
        out = draft

        # Lowercase run.
        bias = float(getattr(speech, "lowercase_bias", 0.0) or 0.0)
        if label in {"tired", "annoyed", "distant", "cold", "irritated"}:
            bias *= 1.3
        elif label in {"excited", "playful"}:
            bias *= 0.5
        if bias > 0 and rng.random() < min(0.95, bias):
            out = out.lower()

        # Text-speak substitutions, each with its own small roll.
        text_speak = getattr(speech, "text_speak", None) or {}
        for full, short in text_speak.items():
            if not full or not short:
                continue
            if rng.random() < 0.35:
                out = re.sub(r"(?i)\b" + re.escape(full) + r"\b", short, out)

        # Tired punctuation drift: trailing periods sag into ellipses.
        if energy < 30 and rng.random() < 0.4:
            out = re.sub(r"\.(?=\s*$)", "…", out)
            out = re.sub(r"\!\s*$", "…", out)

        # High-frustration flatness: high frustration kills exclamation.
        if frustration > 60:
            out = out.replace("!", ".")

        # Collapse the damage a substitution can leave behind.
        out = re.sub(r"[ ]{2,}", " ", out).strip()
        return out or draft
    except Exception:  # noqa: BLE001 - a guard must never break a reply
        return draft
