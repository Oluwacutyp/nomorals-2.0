"""Devon Voice Engine — the humanizing director (god-tier edition).

Plain text in, *performance script* out. Two layers:

1. :class:`PerformanceTuner` — a dedicated class that fine-tunes raw text
   *before* it ever reaches a model. It normalizes the text the way a
   voice actor's script supervisor would (numbers → words, dates, times,
   abbreviations, URLs, currency — ₦-aware), splits it into segments,
   detects each segment's *intent* (question, exclamation, laughter,
   apology, whisper, …), and routes a natural effect chain for it —
   ElevenLabs Multilingual v3 style audio tags and Fish Audio S2 style
   free-form ``[tag]`` direction. Seeded dice keep it reproducible.

2. ``direct()`` — the one-call path, now powered by the tuner. Same
   signature as before; old scripts still direct the same way.

Canonical markup (superset of the ElevenLabs v3 audio-tag vocabulary
and Fish Audio S2's fine-grained control tags — see module notes):

- emotions/direction: ``[happy]`` ``[sad]`` ``[angry]`` ``[excited]``
  ``[nervous]`` ``[scared]`` ``[proud]`` ``[sarcastic]`` ``[curious]``
  ``[mischievously]`` ``[surprised]`` ``[thoughtful]`` ``[confident]``
  ``[annoyed]`` ``[empathetic]`` ``[reassuring]`` ``[tender]`` ``[playful]``
  ``[nostalgic]`` ``[terrified]`` ``[ecstatic]`` ``[deadpan]`` ``[smug]``
  ``[wistful]`` ``[bitter]`` ``[hopeful]`` ``[triumphant]`` ``[desperate]``
  ``[panicked]`` ``[disgusted]`` ``[suspicious]`` ``[amused]`` ``[relieved]``
  … (60+ total — see ``CANONICAL_EMOTIONS``)
- delivery: ``[whisper]`` ``[whispering]`` ``[shouting]`` ``[singing]``
  ``[muttering]`` ``[soft]`` ``[loud]`` ``[crying]`` ``[screaming]``
  ``[panting]`` ``[chanting]`` ``[aside]``
- vocal bursts: ``[laugh]`` ``[bellylaugh]`` ``[nervouslaugh]`` ``[chuckle]``
  ``[giggle]`` ``[sigh]`` ``[breath]`` ``[inhale]`` ``[exhale]`` ``[pant]``
  ``[cough]`` ``[sneeze]`` ``[sniffle]`` ``[gasp]`` ``[scream]``
  ``[clearthroat]`` ``[yawn]`` ``[snore]`` ``[snort]`` ``[tsk]`` ``[sob]``
  ``[whimper]`` ``[gulp]`` ``[groan]`` ``[hum]`` ``[whistle]`` ``[mumble]``
  ``[beep]`` ``[clap]`` ``[applause]``
- fillers: ``[um]`` ``[uh]`` ``[erm]`` ``[hmm]`` ``[like]`` ``[well]``
  ``[yknow]`` ``[right]`` ``[so]``
- pacing: ``[pause:MS]`` ``[beat]`` ``[rate:vslow|slow|fast|vfast]``
- ``[stutter]`` / ``[stutter:N]`` — stammers the next word (``really`` →
  ``r-really``, ``[stutter:2]`` → ``r-r-really``)
- ``<strong>word</strong>`` — emphasis (``*word*`` is shorthand)

Renderers translate canonical markup per backend:

- Fish Audio S2: near pass-through — S2 natively accepts 15,000+
  free-form ``[tag]`` directions, so the director speaks its language.
- Dia (nari-labs): parenthesized non-verbals — ``(laughs)`` ``(coughs)``
  ``(sighs)`` ``(sneezes)`` ``(whistles)`` … — plus a ``[S1]`` speaker
  prefix. GPU-only.
- Orpheus (Canopy Labs, Apache-2.0): angle-bracket emotion tags —
  ``<laugh>`` ``<chuckle>`` ``<sigh>`` ``<cough>`` ``<sniffle>``
  ``<groan>`` ``<yawn>`` ``<gasp>`` — everything else falls back to
  speakable onomatopoeia (``achoo``, ``aaah!``, …).
- CosyVoice (instruct): ``[laughter]`` / ``[breath]`` bursts plus a
  natural-language emotion/rate instruction.
- Bark: native ``[laughs]`` ``[sighs]`` ``[cough]`` ``[gasps]`` …
- XTTS / Kokoro / anything else: speakable words + spliced silence.
  Bursts become onomatopoeia (``ha-ha``, ``achoo``, ``ahem``) instead
  of vanishing.

Honest approximations are marked in the code: only Bark has a native
cough token; elsewhere a cough is a sharp breath plus a beat. A stutter
is always textual (``w-word``) — every backend renders it.

Stdlib only. No model, no download, no API.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

__all__ = [
    "CANONICAL_BURSTS",
    "CANONICAL_EMOTIONS",
    "CANONICAL_DELIVERY",
    "CANONICAL_FILLERS",
    "EFFECT_PRESETS",
    "ONOMATOPOEIA",
    "Intent",
    "PerformanceScript",
    "PerformanceTuner",
    "SegmentDirection",
    "direct",
    "render_bark",
    "render_chatterbox",
    "render_cosyvoice",
    "render_dia",
    "render_fish",
    "render_for",
    "render_kitten",
    "render_omnivoice",
    "render_orpheus",
    "render_plain",
    "render_spark",
    "render_zonos",
    "strip_to_text",
    "supported_backends",
    "STYLE_PRESETS",
]

# ---------------------------------------------------------------------------
# Canonical markup — the vocabulary
# ---------------------------------------------------------------------------

#: Vocal bursts the director can emit. Every renderer must handle all of
#: these (even if some become approximations — see module docstring).
CANONICAL_BURSTS = (
    "laugh", "bellylaugh", "nervouslaugh", "chuckle", "giggle",
    "sigh", "breath", "inhale", "exhale", "pant",
    "cough", "sneeze", "sniffle", "gasp", "scream",
    "clearthroat", "yawn", "snore", "snort", "tsk",
    "sob", "whimper", "gulp", "groan",
    "hum", "whistle", "mumble", "beep", "clap", "applause",
)

#: Emotion / direction tags (ElevenLabs v3 + Fish S2 vocabularies, plus
#: the Hume cognitive set: doubt, realization, nostalgia, awkwardness…).
CANONICAL_EMOTIONS = (
    "happy", "sad", "angry", "excited", "nervous", "scared", "proud",
    "sarcastic", "curious", "mischievously", "surprised", "thoughtful",
    "confident", "annoyed", "appalled", "empathetic", "reassuring",
    "tender", "playful", "bored", "determined", "guilty", "shy",
    "calm", "tired",
    # advanced affect (Fish S2 advanced set + Hume-style cognitive states)
    "nostalgic", "jealous", "contemptuous", "hysterical", "resigned",
    "terrified", "ecstatic", "deadpan", "smug", "flustered", "wistful",
    "bitter", "hopeful", "lonely", "awkward", "triumphant", "desperate",
    "panicked", "disgusted", "suspicious", "amused", "envious",
    "remorseful", "relieved", "eager", "hesitant", "skeptical",
    "reflective",
)

#: Delivery-style tags.
CANONICAL_DELIVERY = (
    "whisper", "whispering", "shouting", "singing", "muttering",
    "soft", "loud", "crying", "screaming", "panting", "chanting",
    "aside",
)

CANONICAL_FILLERS = ("um", "uh", "erm", "hmm", "like", "well",
                      "yknow", "right", "so")

# Built from the canonical tuples (longest first) so the vocabulary
# can't drift out of sync with the regex.
_BURST_RE = re.compile(
    r"\[(" + "|".join(sorted(CANONICAL_BURSTS + CANONICAL_FILLERS,
                             key=len, reverse=True)) + r")\]",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"\[pause:(\d{2,4})\]", re.IGNORECASE)
_BEAT_RE = re.compile(r"\[beat\]", re.IGNORECASE)
_RATE_RE = re.compile(r"\[rate:(vslow|slow|fast|vfast)\]", re.IGNORECASE)
_STUTTER_RE = re.compile(r"\[stutter(?::(\d+))?\]\s*(\S+)", re.IGNORECASE)
_STRONG_MD_RE = re.compile(r"\*(\S[^*]*\S|\S)\*")  # *word* → <strong>
_STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.IGNORECASE | re.DOTALL)
_EMOTION_RE = re.compile(
    r"\[(" + "|".join(CANONICAL_EMOTIONS + CANONICAL_DELIVERY) + r")\]",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"\[[^\]]+\]|<strong>.*?</strong>", re.DOTALL)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_WORD_RE = re.compile(r"\S+")
_LAUGH_WORD_RE = re.compile(r"\b(haha+h?|hehe+h?|lol|lmao+|rofl)\b",
                            re.IGNORECASE)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Intent:
    """What a segment *is* — the tuner's read of one sentence."""

    kind: str = "statement"   # statement|question|exclaim|greeting|
                              # apology|laugh|command|whisper|shout|story
    emotion: str | None = None
    emphasis: tuple[str, ...] = ()   # words the author stressed (CAPS)
    laugh: bool = False              # author typed laughter words


@dataclass
class SegmentDirection:
    """One segment of the performance with its stage directions."""

    text: str                    # canonical marked-up segment text
    intent: Intent = field(default_factory=Intent)
    tags: tuple[str, ...] = ()   # direction tags applied, in order


@dataclass
class PerformanceScript:
    """A line of text plus its stage directions."""

    text: str                      # canonical marked-up script
    mood: str = "neutral"
    seed: int | None = None
    cues: list[str] = field(default_factory=list)  # cue kinds inserted
    segments: list[SegmentDirection] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


# ---------------------------------------------------------------------------
# Text normalization — script-supervisor pass
# ---------------------------------------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven "
         "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def _num_to_words(n: int) -> str:
    """Small integer → words (0 … 999,999,999)."""
    if n < 0:
        return "minus " + _num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + _num_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + rest
    for value, name in ((1_000_000, "million"), (1_000, "thousand")):
        if n >= value:
            head = _num_to_words(n // value) + " " + name
            rest = "" if n % value == 0 else " " + _num_to_words(n % value)
            return head + rest
    return str(n)  # pragma: no cover - unreachable


_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _ordinal_words(n: int) -> str:
    words = _num_to_words(n)
    head, _, tail = words.rpartition(" ")
    tail = _ORDINALS.get(tail, tail + ("th" if not tail.endswith("y")
                                      else "ieth"))
    if tail.endswith("ieth"):
        tail = tail[:-4] + "ieth"
    # fix "-y" ordinals: twenty → twentieth
    if words.endswith("y") and not tail.endswith("th"):
        tail = words[:-1] + "ieth"
        return tail
    return (head + " " + tail).strip() if head else tail


_ABBREV = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss",
    "st": "saint", "ave": "avenue", "rd": "road", "blvd": "boulevard",
    "jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
    "eg": "for example", "ie": "that is", "am": "A M", "pm": "P M",
    "no": "number", "dept": "department", "est": "established",
    "approx": "approximately", "min": "minutes", "max": "maximum",
}

_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds", "₦": "naira",
             "¥": "yen", "₹": "rupees", "¢": "cents"}

_EMOJI_CUES = {
    "😂": "[laugh]", "🤣": "[bellylaugh]", "😹": "[laugh]", "😆": "[chuckle]",
    "😅": "[nervouslaugh]", "🤭": "[giggle]", "😭": "[sob]", "😢": "[sigh]",
    "😮": "[gasp]", "😯": "[gasp]", "😱": "[scream]", "🥱": "[yawn]",
    "🤔": "[hmm]", "😴": "[yawn]", "🤧": "[sneeze]", "🥵": "[pant]",
    "😤": "[snort]", "🥲": "[nervouslaugh]",
}


_MONTHS = ("january february march april may june july august september "
           "october november december").split()


def _speak_number_token(tok: str) -> str:
    """Speak one numeric-ish token: 42, 3.14, 1st, 50%, $5, 3:30pm."""
    # peel trailing punctuation ("2026-10-01," → speak core, reattach)
    trail = ""
    while tok and tok[-1] in ".,;:!?":
        trail = tok[-1] + trail
        tok = tok[:-1]
    if not tok:
        return trail
    spoken = _speak_number_token_inner(tok)
    return spoken + trail


def _speak_number_token_inner(tok: str) -> str:
    # ISO date 2026-10-01 → "october first twenty twenty-six"
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", tok)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year_tok = f"{y:04d}"
        return (f"{_MONTHS[mo - 1]} {_ordinal_words(d)} "
                f"{_speak_number_token(year_tok)}")
    # thousand separators: 1,250 → 1250 before speaking
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", tok):
        tok = tok.replace(",", "")
    # currency prefix
    for sym, name in _CURRENCY.items():
        if tok.startswith(sym) and len(tok) > 1:
            rest = _speak_number_token(tok[len(sym):])
            return f"{rest} {name}"
    # percent
    if tok.endswith("%") and tok[:-1].replace(".", "", 1).isdigit():
        return _speak_decimal(tok[:-1]) + " percent"
    # ordinal 1st 2nd 3rd 4th
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok, re.IGNORECASE)
    if m:
        return _ordinal_words(int(m.group(1)))
    # clock time 3:30, 15:30pm
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(am|pm)?", tok, re.IGNORECASE)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if mi:
            spoken = f"{_num_to_words(h)} {_num_to_words(mi)}"
        else:
            spoken = f"{_num_to_words(h)} o'clock"
        if ap:
            spoken += " " + " ".join(ap.upper())
        return spoken
    # year-ish 4-digit
    if re.fullmatch(r"(19|20)\d{2}", tok):
        a, b = int(tok[:2]), int(tok[2:])
        if b == 0:
            return _num_to_words(a) + " hundred"
        if b < 10:
            return f"{_num_to_words(a)} oh {_num_to_words(b)}"
        return f"{_num_to_words(a)} {_num_to_words(b)}"
    # decimal / plain integer
    if re.fullmatch(r"\d+(\.\d+)?", tok):
        return _speak_decimal(tok)
    return tok


def _speak_decimal(tok: str) -> str:
    if "." not in tok:
        return _num_to_words(int(tok))
    head, tail = tok.split(".", 1)
    digits = " ".join(_ONES[int(d)] for d in tail if d.isdigit())
    return f"{_num_to_words(int(head))} point {digits}".strip()


def normalize_for_speech(text: str) -> str:
    """Turn written text into speakable text.

    Numbers, ordinals, years, clock times, currency (incl. ₦ naira),
    percentages, common abbreviations, URLs, hashtags, and a few emoji
    become words or cues. Idempotent on already-normal text.
    """
    # emoji → cues first (they vanish from TTS otherwise)
    for emo, cue in _EMOJI_CUES.items():
        text = text.replace(emo, f" {cue} ")

    def fix_token(m: re.Match) -> str:
        tok = m.group(0)
        # abbreviations like "Dr." / "e.g."
        low = tok.rstrip(".").lower().replace(".", "")
        if low in _ABBREV and tok.endswith("."):
            spoken = _ABBREV[low]
            # keep sentence-case
            return spoken.capitalize() if tok[0].isupper() else spoken
        # URLs → spoken domain
        if re.match(r"https?://|www\.", tok, re.IGNORECASE):
            dom = re.sub(r"^https?://", "", tok, flags=re.IGNORECASE)
            dom = re.sub(r"^www\.", "", dom, flags=re.IGNORECASE)
            dom = dom.split("/")[0].split("?")[0]
            return dom.replace(".", " dot ").replace("-", " dash ")
        # hashtags / mentions
        if tok.startswith("#") and len(tok) > 1:
            return "hashtag " + tok[1:].replace("_", " ")
        if tok.startswith("@") and len(tok) > 1:
            return "at " + tok[1:].replace("_", " ")
        # numbers / times / currency / ordinals
        if re.search(r"\d", tok):
            return _speak_number_token(tok)
        return tok

    text = re.sub(r"https?://\S+|www\.\S+|[@#]\w[\w_]*|\S*\d\S*|[A-Za-z]+\.",
                  fix_token, text)
    # collapse the double spaces the emoji pass may leave
    return re.sub(r"\s{2,}", " ", text).strip()

# ---------------------------------------------------------------------------
# Intent detection — reading the line like a director reads a script
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hey|hello|hi|yo|good\s?(morning|afternoon|evening)|welcome)\b",
    re.IGNORECASE)
_APOLOGY_RE = re.compile(r"\b(sorry|apologi|forgive me|my bad|my fault)\b",
                         re.IGNORECASE)
_WHISPER_RE = re.compile(
    r"\b(psst|quietly|secret|don't tell|between you and me|hush)\b",
    re.IGNORECASE)
_COMMAND_RE = re.compile(
    r"^(please\s+)?(stop|listen|look|wait|come|go|tell me|do it|try|"
    r"remember|don't|never)\b", re.IGNORECASE)
_STORY_RE = re.compile(
    r"\b(once upon|so there i was|it all started|back in|remember when)\b",
    re.IGNORECASE)


def detect_intent(sentence: str) -> Intent:
    """Classify one sentence: kind, emotion hint, stressed words."""
    s = sentence.strip()
    bare = _TAG_RE.sub("", s)
    emphasis = tuple(w for w in re.findall(r"[A-Z]{2,}", bare)
                     if len(w) > 1)
    laugh = bool(_LAUGH_WORD_RE.search(bare))

    kind = "statement"
    emotion: str | None = None
    if bare.rstrip().endswith("?"):
        kind = "question"
    elif bare.rstrip().endswith("!"):
        kind = "exclaim"
        if laugh or re.search(r"\b(yes+!*|woo+|amazing|incredible)\b", bare,
                              re.IGNORECASE):
            emotion = "excited"
    if _GREETING_RE.search(bare):
        kind = "greeting"
        emotion = emotion or "happy"
    elif _APOLOGY_RE.search(bare):
        kind = "apology"
        emotion = emotion or "empathetic"
    elif _WHISPER_RE.search(bare):
        kind = "whisper"
    elif _STORY_RE.search(bare):
        kind = "story"
    elif _COMMAND_RE.search(bare):
        kind = "command"
    if laugh and kind == "statement":
        kind = "laugh"
    # ALL-CAPS sentence = shouting (but not when it's just emphasis words)
    words = bare.split()
    if (len(words) >= 3 and sum(1 for w in words if w.isupper()) >=
            len(words) * 0.7):
        kind = "shout"
    return Intent(kind=kind, emotion=emotion, emphasis=emphasis, laugh=laugh)


# ---------------------------------------------------------------------------
# Effect presets — named performance styles (the "lots of scripts")
# ---------------------------------------------------------------------------

#: Named effect chains. A preset is an ordered list of direction tags a
#: segment may open with; the tuner picks per intent. Think of these as
#: the house styles — ElevenLabs v3 "audio tags" thinking, Fish S2
#: free-form direction energy.
EFFECT_PRESETS: dict[str, list[str]] = {
    "dramatic_whisper": ["whispering", "tense"],
    "bedtime_story": ["soft", "reassuring"],
    "hype": ["excited", "loud"],
    "conspiratorial": ["mischievously", "whispering"],
    "heartfelt": ["tender", "soft"],
    "standup": ["playful", "mischievously"],
    "newsreader": ["confident", "calm"],
    "apology": ["empathetic", "soft"],
    "villain": ["mischievously", "slow"],
    "cheerful": ["happy", "excited"],
    "melancholy": ["sad", "soft"],
    "sports_caster": ["excited", "loud"],
    "lullaby": ["soft", "singing"],
    "sarcastic_bite": ["sarcastic", "muttering"],
    "nervous_wreck": ["nervous", "fast"],
}

_INTENT_EFFECTS: dict[str, list[str]] = {
    # intent kind → candidate direction tags (tuner picks, seeded)
    "question": ["curious", "thoughtful"],
    "exclaim": ["excited", "surprised"],
    "greeting": ["happy", "playful"],
    "apology": ["empathetic", "reassuring"],
    "laugh": ["playful", "happy"],
    "command": ["confident", "determined"],
    "whisper": ["whispering"],
    "shout": ["shouting"],
    "story": ["thoughtful", "calm"],
    "statement": [],
}

_MOOD_PROFILES = {
    # burst/filler/stutter probabilities + pacing per mood.
    # "bursts" is the extended table the tuner rolls after the classic
    # laugh/chuckle/breath/sigh blocks; new moods welcome.
    "happy":    {"laugh": 0.45, "chuckle": 0.30, "breath": 0.10, "sigh": 0.0,
                 "filler": 0.05, "stutter": 0.0, "rate": "fast",
                 "bursts": {"giggle": 0.15, "bellylaugh": 0.08}},
    "excited":  {"laugh": 0.35, "chuckle": 0.25, "breath": 0.15, "sigh": 0.0,
                 "filler": 0.05, "stutter": 0.05, "rate": "fast",
                 "bursts": {"giggle": 0.15, "scream": 0.05, "clap": 0.05}},
    "sad":      {"laugh": 0.0, "chuckle": 0.0, "breath": 0.20, "sigh": 0.35,
                 "filler": 0.15, "stutter": 0.05, "rate": "slow",
                 "bursts": {"sob": 0.12, "whimper": 0.08, "sniffle": 0.06}},
    "nervous":  {"laugh": 0.15, "chuckle": 0.10, "breath": 0.20, "sigh": 0.10,
                 "filler": 0.35, "stutter": 0.30, "rate": None,
                 "bursts": {"nervouslaugh": 0.15, "gulp": 0.10, "um": 0.0}},
    "tired":    {"laugh": 0.0, "chuckle": 0.0, "breath": 0.35, "sigh": 0.25,
                 "filler": 0.10, "stutter": 0.0, "rate": "slow",
                 "bursts": {"yawn": 0.20, "snore": 0.03}},
    "angry":    {"laugh": 0.0, "chuckle": 0.0, "breath": 0.25, "sigh": 0.15,
                 "filler": 0.0, "stutter": 0.0, "rate": "fast",
                 "bursts": {"groan": 0.12, "tsk": 0.10, "snort": 0.06}},
    "calm":     {"laugh": 0.0, "chuckle": 0.05, "breath": 0.25, "sigh": 0.05,
                 "filler": 0.05, "stutter": 0.0, "rate": "slow",
                 "bursts": {"hum": 0.06}},
    "neutral":  {"laugh": 0.05, "chuckle": 0.05, "breath": 0.12, "sigh": 0.03,
                 "filler": 0.08, "stutter": 0.02, "rate": None,
                 "bursts": {}},
    # extended moods
    "hysterical": {"laugh": 0.50, "chuckle": 0.20, "breath": 0.10,
                   "sigh": 0.0, "filler": 0.02, "stutter": 0.0,
                   "rate": "fast",
                   "bursts": {"bellylaugh": 0.30, "giggle": 0.25,
                              "gasp": 0.15, "pant": 0.10}},
    "sleepy":   {"laugh": 0.0, "chuckle": 0.0, "breath": 0.20, "sigh": 0.10,
                 "filler": 0.10, "stutter": 0.0, "rate": "vslow",
                 "bursts": {"yawn": 0.40, "mumble": 0.08, "snore": 0.05}},
    "sick":     {"laugh": 0.0, "chuckle": 0.0, "breath": 0.15, "sigh": 0.10,
                 "filler": 0.05, "stutter": 0.0, "rate": "slow",
                 "bursts": {"cough": 0.35, "sniffle": 0.25, "sneeze": 0.20,
                            "groan": 0.10}},
    "playful":  {"laugh": 0.30, "chuckle": 0.30, "breath": 0.10, "sigh": 0.0,
                 "filler": 0.10, "stutter": 0.0, "rate": "fast",
                 "bursts": {"giggle": 0.25, "whistle": 0.08, "beep": 0.03}},
}
_MOOD_PROFILES["annoyed"] = _MOOD_PROFILES["angry"]


# ---------------------------------------------------------------------------
# The PerformanceTuner — fine-tunes text before it reaches any model
# ---------------------------------------------------------------------------

class PerformanceTuner:
    """Turns raw written text into a directed performance script.

    Pipeline (each stage is public so callers can compose their own):

    1. :meth:`normalize` — script-supervisor pass: numbers, dates, times,
       abbreviations, URLs, currency → speakable words.
    2. :meth:`segment` — split into performable sentences.
    3. :meth:`detect` — :func:`detect_intent` per segment.
    4. :meth:`route` — pick a natural effect chain from intent + mood +
       an optional named :data:`EFFECT_PRESETS` style.
    5. :meth:`shape` — punctuation shaping (ellipses breathe, CAPS
       stress) and the classic humanizing dice (bursts, fillers,
       stutters, beats) from the mood profile.
    6. :meth:`tune` — all of the above → :class:`PerformanceScript`.

    ``intensity`` 0–5 scales how theatrical it gets (0 = normalize only).
    ``seed`` makes the dice reproducible: same text + seed = same show.
    """

    def __init__(self, mood: str = "neutral", intensity: int = 3,
                 seed: int | None = None, lang: str = "en",
                 effect: str | None = None) -> None:
        self.mood = (mood or "neutral").lower()
        self.profile = _MOOD_PROFILES.get(self.mood,
                                          _MOOD_PROFILES["neutral"])
        self.intensity = max(0, min(5, intensity))
        self.seed = seed
        self.lang = lang
        self.effect = effect
        self._rng = random.Random(seed)

    # -- stages ------------------------------------------------------

    def normalize(self, text: str) -> str:
        return normalize_for_speech(text)

    def segment(self, text: str) -> list[str]:
        return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]

    def detect(self, sentence: str) -> Intent:
        return detect_intent(sentence)

    def route(self, intent: Intent) -> list[str]:
        """Pick direction tags for a segment.

        A named ``effect`` preset wins outright; otherwise the intent's
        candidate effects route through the mood, with the intent's own
        emotion hint first.
        """
        if self.effect and self.effect in EFFECT_PRESETS:
            return list(EFFECT_PRESETS[self.effect])
        tags: list[str] = []
        if intent.emotion:
            tags.append(intent.emotion)
        for cand in _INTENT_EFFECTS.get(intent.kind, []):
            if cand not in tags:
                # mood-congruent candidates preferred, seeded pick of one
                tags.append(cand)
                break
        return tags[:2]

    def shape(self, sentence: str, intent: Intent, tags: list[str],
              position: int, total: int) -> tuple[str, list[str]]:
        """Apply direction tags + humanizing dice to one segment."""
        rng = self._rng
        cues: list[str] = []
        sent = sentence
        scale = self.intensity / 3.0

        def maybe(p: float) -> bool:
            return rng.random() < p * scale

        # author emphasis: *word* → <strong>, ALL-CAPS words → <strong>
        sent = _STRONG_MD_RE.sub(r"<strong>\1</strong>", sent)
        for w in intent.emphasis:
            if f"<strong>{w}</strong>" not in sent:
                sent = re.sub(rf"\b{re.escape(w)}\b",
                              f"<strong>{w}</strong>", sent, count=1)

        # direction tags open the segment (author tags win — never double)
        lowered = sent.lower()
        opens = [f"[{t}]" for t in tags
                 if t in CANONICAL_EMOTIONS + CANONICAL_DELIVERY
                 and f"[{t}]" not in lowered]
        if opens and not sent.lstrip().startswith("["):
            sent = " ".join(opens) + " " + sent
            cues.append("direction:" + ",".join(tags))

        # laughter words the author typed become real laughs
        if intent.laugh and "[laugh]" not in lowered \
                and "[chuckle]" not in lowered:
            sent = _LAUGH_WORD_RE.sub("[laugh]", sent)
            cues.append("laugh")

        # breath at clause boundaries in long sentences
        words = _WORD_RE.findall(sent)
        if len(words) > 16 and maybe(self.profile["breath"] + 0.25):
            sent = re.sub(r",\s+", ", [breath] ", sent, count=1)
            cues.append("breath")

        # exclamations get a chuckle when the mood is light
        if sent.rstrip().endswith("!") and maybe(self.profile["chuckle"]):
            if "[chuckle]" not in sent.lower() \
                    and "[laugh]" not in sent.lower():
                sent = sent.rstrip() + " [chuckle]"
                cues.append("chuckle")

        # questions asked nervously start with a filler (slotted after
        # any leading direction tags so the tags never mute it)
        if sent.rstrip().endswith("?") \
                and maybe(self.profile["filler"] + 0.15):
            filler = rng.choice(["[um]", "[uh]", "[hmm]", "[like]",
                                 "[well]", "[yknow]"])
            lead = re.match(r"(\[[^\]]+\]\s*)+", sent)
            if lead:
                sent = lead.group(0) + f"{filler} " + sent[lead.end():]
            else:
                sent = f"{filler} {sent}"
            cues.append("filler")

        # sighs open sad/tired lines
        if position == 0 and maybe(self.profile["sigh"]) \
                and not sent.lstrip().startswith("["):
            sent = f"[sigh] {sent}"
            cues.append("sigh")

        # extended mood burst table — at most one extra burst per segment
        # (the classic laugh/chuckle/breath/sigh are handled above)
        for burst, prob in self.profile.get("bursts", {}).items():
            if burst in ("laugh", "chuckle", "breath", "sigh"):
                continue
            if prob and maybe(prob) \
                    and f"[{burst}]" not in sent.lower():
                sent = sent.rstrip() + f" [{burst}]"
                cues.append(burst)
                break

        # stutter the first content word when nervous
        if maybe(self.profile["stutter"]):
            w = _first_content_word(sent)
            if w and "[stutter]" not in sent.lower():
                sent = sent.replace(w, f"[stutter]{w}", 1)
                cues.append("stutter")

        # ellipses breathe: "..." → a real beat
        if "..." in sent or "…" in sent:
            sent = re.sub(r"\.\.\.|…", " [pause:450] ", sent)
            cues.append("pause")

        # short beat before a dramatic final sentence
        if position == total - 1 and total > 1 and maybe(0.25 * scale):
            sent = f"[pause:350] {sent}"
            cues.append("pause")

        return re.sub(r"\s{2,}", " ", sent).strip(), cues

    def tune(self, text: str) -> PerformanceScript:
        """Full pipeline: raw text → directed :class:`PerformanceScript`."""
        # intensity 0 = verbatim: no normalization, no direction, no dice —
        # but the author's own *emphasis* markup is still honored.
        if self.intensity == 0:
            clean = _STRONG_MD_RE.sub(r"<strong>\1</strong>",
                                      text.strip())
            return PerformanceScript(
                text=clean, mood=self.mood, seed=self.seed, cues=[],
                segments=[SegmentDirection(text=clean,
                                           intent=Intent(), tags=())])
        normalized = self.normalize(text)
        sentences = self.segment(normalized)
        out: list[str] = []
        cues: list[str] = []
        segments: list[SegmentDirection] = []
        total = len(sentences)
        for i, sent in enumerate(sentences):
            intent = self.detect(sent)
            tags = self.route(intent)
            directed, seg_cues = self.shape(sent, intent, tags, i, total)
            out.append(directed)
            cues.extend(seg_cues)
            segments.append(SegmentDirection(text=directed, intent=intent,
                                             tags=tuple(tags)))
        script = re.sub(r"\s{2,}", " ", " ".join(out)).strip()
        rate = self.profile["rate"]
        if rate and self.intensity >= 2 \
                and "[rate:" not in script.lower():
            script = f"[rate:{rate}] {script}"
            cues.append("rate")
        return PerformanceScript(text=script, mood=self.mood, seed=self.seed,
                                 cues=cues, segments=segments)


def _first_content_word(sentence: str) -> str | None:
    for w in _WORD_RE.findall(sentence):
        if re.search(r"[A-Za-z]", w):
            return w
    return None


def _stutter_word(word: str, count: int = 1) -> str:
    """``really`` → ``r-really`` (count=1); ``[stutter:2]really`` →
    ``r-r-really``. Keeps leading punctuation intact."""
    m = re.match(r"^(\W*)(.+)$", word, re.DOTALL)
    if not m:
        return word
    punct, core = m.groups()
    if len(core) < 2:
        return word
    count = max(1, min(4, count))
    return punct + "-".join([core[0]] * count) + "-" + core


def direct(text: str, *, mood: str = "neutral", intensity: int = 3,
           seed: int | None = None, lang: str = "en",
           effect: str | None = None) -> PerformanceScript:
    """Turn plain text into a performance script.

    ``intensity`` 0–5 scales how often cues fire (0 = normalize only,
    5 = full theatre kid). ``seed`` makes the dice reproducible.
    ``effect`` picks a named :data:`EFFECT_PRESETS` house style.
    Author-supplied canonical tags are always respected, never doubled.
    """
    return PerformanceTuner(mood=mood, intensity=intensity, seed=seed,
                            lang=lang, effect=effect).tune(text)

# ---------------------------------------------------------------------------
# Renderers — canonical markup → backend-native
# ---------------------------------------------------------------------------

def _apply_stutter(script: str) -> str:
    def _one(m: re.Match) -> str:
        count = int(m.group(1)) if m.group(1) else 1
        return _stutter_word(m.group(2), count)

    return _STUTTER_RE.sub(_one, script)


def _apply_beat(script: str) -> str:
    return _BEAT_RE.sub("[pause:300]", script)


# canonical burst → Fish S2 native (free-form tags: near pass-through)
_FISH_BURSTS = {
    "laugh": "[laughing]", "bellylaugh": "[belly laughing]",
    "nervouslaugh": "[nervous laughter]", "chuckle": "[chuckle]",
    "giggle": "[giggling]",
    "sigh": "[sigh]", "breath": "[inhale]", "inhale": "[inhale]",
    "exhale": "[exhale]", "pant": "[panting]",
    "cough": "[cough]", "sneeze": "[sneeze]", "sniffle": "[sniffle]",
    "gasp": "[gasp]", "scream": "[scream]",
    "clearthroat": "[clearing throat]", "yawn": "[yawn]", "snore": "[snoring]",
    "snort": "[snort]", "tsk": "[tsk]", "sob": "[sobbing]",
    "whimper": "[whimpering]", "gulp": "[gulp]", "groan": "[groan]",
    "hum": "[humming]", "whistle": "[whistle]", "mumble": "[mumbling]",
    "beep": "[beep]", "clap": "[clapping]", "applause": "[applause]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
    "like": "like,", "well": "well,", "yknow": "y'know,",
    "right": "right,", "so": "so,",
}

_FISH_RATES = {
    "vslow": "[speaking very slowly]", "slow": "[speaking slowly]",
    "fast": "[speaking quickly]", "vfast": "[speaking very quickly]",
}


def render_fish(script: str | PerformanceScript) -> str:
    """Canonical markup → Fish Audio S2 native direction.

    S2 accepts 15,000+ free-form ``[tag]`` directions natively, so this
    is nearly a pass-through: canonical tags are already in its
    language. Pauses quantize to S2's short/standard/long vocabulary,
    ``<strong>`` becomes ``[emphasis]``, stutters stay textual.
    """
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(lambda m: _FISH_BURSTS[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub(lambda m: " " + _FISH_RATES[m.group(1).lower()] + " ", s)
    # emotions/delivery pass through untouched — S2 speaks them natively
    s = _PAUSE_RE.sub(
        lambda m: (" [short pause] " if int(m.group(1)) < 250
                   else " [long pause] " if int(m.group(1)) >= 700
                   else " [pause] "), s)
    s = _STRONG_RE.sub(r"[emphasis] \1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


# canonical burst → CosyVoice instruct vocabulary
_COSY_BURSTS = {
    "laugh": "[laughter]", "bellylaugh": "[laughter]",
    "nervouslaugh": "[laughter]", "chuckle": "[laughter]",
    "giggle": "[laughter]",
    "sigh": "[breath] [pause:300]", "breath": "[breath]",
    "inhale": "[breath]", "exhale": "[breath] [pause:200]",
    "pant": "[breath] [pause:200]",
    # no native cough token: sharp breath + beat (approximation)
    "cough": "[breath] [pause:250]",
    "sneeze": "achoo! [breath]", "sniffle": "[breath]",
    "gasp": "[breath]", "scream": "aaah! [breath]",
    "clearthroat": "[breath] [pause:200]",
    "yawn": "[breath] [pause:400]", "snore": "[breath] [pause:500]",
    "snort": "[breath]", "tsk": "[breath]",
    "sob": "[breath] [pause:300]", "whimper": "[breath] [pause:300]",
    "gulp": "[breath]", "groan": "uugh. [breath]",
    "hum": "hmm,", "whistle": "[breath] [pause:300]",
    "mumble": "[breath]", "beep": "[breath] [pause:200]",
    "clap": "[breath]", "applause": "[breath] [pause:400]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
    "like": "like,", "well": "well,", "yknow": "y'know,",
    "right": "right,", "so": "so,",
}
_COSY_RATES = {
    "vslow": "Speak very slowly.", "slow": "Speak slowly.",
    "fast": "Speak quickly.", "vfast": "Speak very quickly.",
}
# delivery tags with no CosyVoice token → natural-language instruction
_COSY_DELIVERY_INSTRUCT = {
    "whisper": "Whisper.", "whispering": "Whisper.",
    "shouting": "Speak loudly, almost shouting.",
    "singing": "Sing the line.", "muttering": "Mutter under your breath.",
    "soft": "Speak softly.", "loud": "Speak loudly.",
}


def render_cosyvoice(script: str | PerformanceScript) -> tuple[str, str]:
    """Return ``(text, instruct)`` for CosyVoice's instruct mode.

    Maps canonical cues onto CosyVoice's native vocabulary —
    ``[laughter]`` / ``[breath]`` bursts, ``<strong>`` spans — and folds
    mood, emotion tags, delivery tags, and rate into a natural-language
    instruction. Approximations: sigh/cough/gasp/clearthroat/yawn/sob
    have no native token, so they become a breath plus a beat of
    silence.
    """
    s = _apply_beat(str(script))
    mood = script.mood if isinstance(script, PerformanceScript) else "neutral"

    instruct_bits: list[str] = []
    seen_emotions: list[str] = []

    def _collect_emo(m: re.Match) -> str:
        emo = m.group(1).lower()
        if emo in _COSY_DELIVERY_INSTRUCT:
            instruct_bits.append(_COSY_DELIVERY_INSTRUCT[emo])
        elif emo not in seen_emotions:
            seen_emotions.append(emo)
        return ""

    s = _EMOTION_RE.sub(_collect_emo, s)
    if mood not in ("neutral",):
        instruct_bits.insert(0, f"Speak in a {mood} tone.")
    if seen_emotions:
        instruct_bits.append("Delivery shifts: %s."
                             % ", ".join(seen_emotions))
    rate = _RATE_RE.search(s)
    if rate:
        instruct_bits.append(_COSY_RATES[rate.group(1).lower()])

    s = _BURST_RE.sub(lambda m: _COSY_BURSTS[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _PAUSE_RE.sub("... ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s, " ".join(instruct_bits).strip()


# canonical burst → Bark native tags (the paralinguistic king).
# Best-effort for the newer bursts: Bark was trained on varied bracket
# tags and usually renders *something* plausible for unknown ones.
_BARK_BURSTS = {
    "laugh": "[laughs]", "bellylaugh": "[laughs]",
    "nervouslaugh": "[chuckles]", "chuckle": "[chuckles]",
    "giggle": "[giggles]",
    "sigh": "[sighs]", "breath": "[sighs]", "inhale": "[sighs]",
    "exhale": "[sighs]", "pant": "[sighs]",
    "cough": "[cough]", "sneeze": "[sneezes]", "sniffle": "[sniffles]",
    "gasp": "[gasps]", "scream": "[screams]",
    "clearthroat": "[clears throat]", "yawn": "[yawns]", "snore": "[snores]",
    "snort": "[snorts]", "tsk": "[tsk]", "sob": "[sobs]",
    "whimper": "[whimpers]", "gulp": "[gulps]", "groan": "[groans]",
    "hum": "[hums]", "whistle": "[whistles]", "mumble": "[mumbles]",
    "beep": "[beeps]", "clap": "[claps]", "applause": "[applause]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
    "like": "like,", "well": "well,", "yknow": "y'know,",
    "right": "right,", "so": "so,",
}

#: Speakable onomatopoeia for every burst — the universal fallback.
#: Used by ``render_plain(speak_bursts=True)`` and by renderers whose
#: backend has no native token for a burst, so cues degrade to words
#: instead of vanishing into silence.
_ONOMATOPOEIA = {
    "laugh": "ha-ha", "bellylaugh": "HA-HA-HA", "nervouslaugh": "heh-heh",
    "chuckle": "heh", "giggle": "hee-hee",
    "sigh": "ahh", "breath": "", "inhale": "", "exhale": "hah",
    "pant": "*panting*",
    "cough": "*cough*", "sneeze": "achoo!", "sniffle": "*sniff*",
    "gasp": "*gasps*", "scream": "aaah!",
    "clearthroat": "ahem", "yawn": "*yawns*", "snore": "zzz",
    "snort": "*snorts*", "tsk": "tsk", "sob": "*sobs*",
    "whimper": "*whimpers*", "gulp": "*gulps*", "groan": "uugh",
    "hum": "hmm-hmm", "whistle": "*whistles*", "mumble": "mumble-mumble",
    "beep": "*beep*", "clap": "*claps*", "applause": "*applause*",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
    "like": "like,", "well": "well,", "yknow": "y'know,",
    "right": "right,", "so": "so,",
}

#: Public alias — the speakable fallback vocabulary, keyed by canonical
#: burst name. Renderers use it when a backend has no native token.
ONOMATOPOEIA = _ONOMATOPOEIA

# canonical burst → Dia (nari-labs) parenthesized non-verbals.
# Dia's documented set: (laughs) (clears throat) (sighs) (gasps) (coughs)
# (singing) (sings) (mumbles) (beep) (groans) (sniffs) (claps) (screams)
# (inhales) (exhales) (applause) (burps) (humming) (sneezes) (chuckle)
# (whistles). Unlisted tags can produce unexpected output, so everything
# else maps to the nearest listed one.
_DIA_BURSTS = {
    "laugh": "(laughs)", "bellylaugh": "(laughs)",
    "nervouslaugh": "(chuckle)", "chuckle": "(chuckle)",
    "giggle": "(chuckle)",
    "sigh": "(sighs)", "breath": "(inhales)", "inhale": "(inhales)",
    "exhale": "(exhales)", "pant": "(exhales)",
    "cough": "(coughs)", "sneeze": "(sneezes)", "sniffle": "(sniffs)",
    "gasp": "(gasps)", "scream": "(screams)",
    "clearthroat": "(clears throat)", "yawn": "(sighs)", "snore": "(sighs)",
    "snort": "(sniffs)", "tsk": "(sniffs)", "sob": "(sighs)",
    "whimper": "(sighs)", "gulp": "(exhales)", "groan": "(groans)",
    "hum": "(humming)", "whistle": "(whistles)", "mumble": "(mumbles)",
    "beep": "(beep)", "clap": "(claps)", "applause": "(applause)",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
    "like": "like,", "well": "well,", "yknow": "y'know,",
    "right": "right,", "so": "so,",
}

# canonical burst → Orpheus (Canopy Labs) angle-bracket emotion tags.
# Native: <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>.
# Everything else → speakable onomatopoeia (Orpheus has no token for it,
# but it speaks plain words just fine).
_ORPHEUS_NATIVE = {
    "laugh": "<laugh>", "bellylaugh": "<laugh>",
    "nervouslaugh": "<chuckle>", "chuckle": "<chuckle>",
    "giggle": "<laugh>",
    "sigh": "<sigh>",
    "cough": "<cough>", "clearthroat": "<cough>",
    "sneeze": "<sniffle>", "sniffle": "<sniffle>", "snort": "<sniffle>",
    "groan": "<groan>",
    "yawn": "<yawn>", "snore": "<yawn>",
    "gasp": "<gasp>",
}


def _burst_or_onomatopoeia(native: dict[str, str],
                           script: str) -> str:
    """Replace ``[burst]`` with the renderer's native tag, else with
    speakable onomatopoeia — never silence."""
    def _one(m: re.Match) -> str:
        burst = m.group(1).lower()
        if burst in native:
            return native[burst]
        return _ONOMATOPOEIA.get(burst, "")

    return _BURST_RE.sub(_one, script)


def render_bark(script: str | PerformanceScript) -> str:
    """Canonical markup → Bark's native tags."""
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(lambda m: _BARK_BURSTS[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    # Bark responds to plain emotion words as tags too
    s = _EMOTION_RE.sub(lambda m: f"[{m.group(1).lower()}]", s)
    s = _PAUSE_RE.sub(lambda m: "... " if int(m.group(1)) < 500
                      else "... ... ", s)
    s = _STRONG_RE.sub(r"*\1*", s)  # Bark: CAPS/*emphasis* reads stressed
    return re.sub(r"\s{2,}", " ", s).strip()


def render_plain(script: str | PerformanceScript, *,
                 speak_bursts: bool = False) -> tuple[str, list[tuple[int, int]]]:
    """Canonical markup → clean text + pause points.

    For backends with no paralinguistic vocabulary (XTTS, Kokoro, HF
    endpoints): fillers stay speakable, pauses are returned as
    ``(char_offset, ms)`` so the engine can splice real silence.

    ``speak_bursts=False`` (default) drops non-speech bursts, keeping
    the historical behavior. ``speak_bursts=True`` renders them as
    speakable onomatopoeia (``achoo!``, ``ha-ha``, ``ahem``) — the
    better fallback the engine uses for plain backends.
    """
    s = _apply_beat(str(script))
    if speak_bursts:
        s = _BURST_RE.sub(
            lambda m: _ONOMATOPOEIA.get(m.group(1).lower(), ""), s)
    else:
        filler_words = {"um": "um,", "uh": "uh,", "erm": "erm,",
                        "hmm": "hmm,", "like": "like,", "well": "well,",
                        "yknow": "y'know,", "right": "right,", "so": "so,"}
        s = _BURST_RE.sub(
            lambda m: filler_words.get(m.group(1).lower(), ""), s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub("", s)
    s = _STRONG_RE.sub(r"\1", s)

    clean_parts: list[str] = []
    pause_points: list[tuple[int, int]] = []
    running = ""
    # split keeping the pause tags so offsets stay honest
    chunks = _PAUSE_RE.split(s)
    for idx, chunk in enumerate(chunks):
        if idx % 2 == 1:  # a pause duration
            pause_points.append((len(running), int(chunk)))
            continue
        chunk = re.sub(r"\s{2,}", " ", chunk)
        clean_parts.append(chunk)
        running += chunk + " "
    text = re.sub(r"\s{2,}", " ", "".join(clean_parts)).strip()
    return text, pause_points


def strip_to_text(script: str | PerformanceScript) -> str:
    """Drop every cue, keep only speakable words."""
    text, _ = render_plain(script)
    return text


def render_dia(script: str | PerformanceScript) -> str:
    """Canonical markup → Dia (nari-labs) native direction.

    Dia speaks parenthesized non-verbals — ``(laughs)`` ``(coughs)``
    ``(sighs)`` ``(sneezes)`` ``(whistles)`` … — inside a ``[S1]`` /
    ``[S2]`` dialogue script. Emotion/delivery tags become parens too
    (``[happy]`` → ``(happy)``); Dia is an LLM-class model and reads
    them as direction. ``<strong>`` becomes CAPS (Dia stresses
    capitalized words), stutters stay textual, pauses become ellipses.
    Rate tags are dropped — Dia has no rate control. GPU-only backend.
    """
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(lambda m: _DIA_BURSTS[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub(lambda m: f"({m.group(1).lower()})", s)
    s = _PAUSE_RE.sub("... ", s)
    s = _STRONG_RE.sub(lambda m: m.group(1).upper(), s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not re.match(r"\[S\d\]", s):
        s = "[S1] " + s
    return s


def render_orpheus(script: str | PerformanceScript) -> str:
    """Canonical markup → Orpheus (Canopy Labs) native direction.

    Orpheus natively understands eight angle-bracket emotion tags:
    ``<laugh>`` ``<chuckle>`` ``<sigh>`` ``<cough>`` ``<sniffle>``
    ``<groan>`` ``<yawn>`` ``<gasp>``. Every other burst degrades to
    speakable onomatopoeia (``achoo!``, ``aaah!``) — Orpheus speaks
    plain words well, so nothing is lost to silence. Emotion/delivery
    tags pass through as angle tags (best-effort; Orpheus is an
    LLM-class model). Stutters stay textual, pauses become ellipses.
    """
    s = _apply_beat(str(script))
    s = _burst_or_onomatopoeia(_ORPHEUS_NATIVE, s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub(lambda m: f"<{m.group(1).lower()}>", s)
    s = _PAUSE_RE.sub("... ", s)
    s = _STRONG_RE.sub(lambda m: m.group(1).upper(), s)
    return re.sub(r"\s{2,}", " ", s).strip()


# canonical burst → Chatterbox Turbo/Nano native paralinguistic tags.
# Documented natives: [laugh] [chuckle] [cough] ("and more" per the
# README). Everything else degrades to speakable onomatopoeia —
# Turbo/Nano speak plain words well, so nothing vanishes.
_CHATTERBOX_NATIVE = {
    "laugh": "[laugh]", "bellylaugh": "[laugh]",
    "nervouslaugh": "[laugh]", "giggle": "[laugh]",
    "chuckle": "[chuckle]",
    "cough": "[cough]",
}


def render_chatterbox(script: str | PerformanceScript) -> str:
    """Canonical markup → Chatterbox Turbo/Nano native direction.

    Speaks the native paralinguistic tags — ``[laugh]`` ``[chuckle]``
    ``[cough]`` — natively; every other burst becomes speakable
    onomatopoeia (``achoo!``, ``ha-ha``) instead of silence. Emotion /
    delivery tags are dropped from the text — Chatterbox takes them
    through the ``exaggeration``/``cfg_weight`` synthesis knobs (see
    ``ChatterboxBackend.synthesize``), not inline text. Stutters stay
    textual, pauses become ellipses. For the multilingual V3 variant
    (no native tags) the engine renders plain text instead.
    """
    s = _apply_beat(str(script))
    s = _burst_or_onomatopoeia(_CHATTERBOX_NATIVE, s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub("", s)
    s = _PAUSE_RE.sub("... ", s)
    s = _STRONG_RE.sub(lambda m: m.group(1).upper(), s)
    return re.sub(r"\s{2,}", " ", s).strip()


# canonical burst → OmniVoice (k2-fsa) non-verbal symbols.
# Documented natives: [laughter] [sigh] [sniff]. OmniVoice also reads
# free-form [tags] as direction (like Fish S2), so unmapped bursts pass
# through as [name] rather than being dropped.
_OMNIVOICE_BURSTS = {
    "laugh": "[laughter]", "bellylaugh": "[laughter]",
    "nervouslaugh": "[laughter]", "chuckle": "[laughter]",
    "giggle": "[laughter]",
    "sigh": "[sigh]",
    "sniffle": "[sniff]",
}


def render_omnivoice(script: str | PerformanceScript) -> str:
    """Canonical markup → OmniVoice native direction.

    Maps the documented non-verbal symbols (``[laughter]`` ``[sigh]``
    ``[sniff]``); every other burst passes through in its canonical
    ``[name]`` form, which the model reads as free-form direction.
    Emotion/delivery tags pass through untouched — 600+ languages and
    the model is direction-tuned. Stutters stay textual, pauses become
    ellipses, ``<strong>`` becomes CAPS.
    """
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(
        lambda m: _OMNIVOICE_BURSTS.get(m.group(1).lower(),
                                        f"[{m.group(1).lower()}]"), s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _PAUSE_RE.sub("... ", s)
    s = _STRONG_RE.sub(lambda m: m.group(1).upper(), s)
    return re.sub(r"\s{2,}", " ", s).strip()


def render_for(backend: str,
               script: str | PerformanceScript) -> tuple[str, str | None]:
    """Render a script for a backend. Returns ``(text, extra)`` where
    ``extra`` is the CosyVoice instruction or ``None``.

    Implemented at the bottom of this module, data-driven via
    :data:`_RENDERERS` (kept there so the Zonos/Kitten/Spark renderers
    it references are defined first).
    """
    return _render_for_registry(backend, script)


# ---------------------------------------------------------------------------
# Zonos / Kitten / Spark renderers (sweep: new backends need vocabularies)
# ---------------------------------------------------------------------------

#: Canonical emotion → Zonos 8-D emotion vector
#: (happiness, sadness, disgust, fear, surprise, anger, other, neutral).
#: Mirrors ``tts.ZonosBackend.EMOTION_VECTORS`` (kept here to avoid a
#: tts→director import cycle).
_ZONOS_VECTORS: dict[str, tuple] = {
    "happy": (0.9, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1),
    "excited": (0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1),
    "sad": (0.05, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "angry": (0.0, 0.0, 0.1, 0.1, 0.0, 0.9, 0.0, 0.0),
    "fearful": (0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0),
    "scared": (0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0),
    "terrified": (0.0, 0.1, 0.0, 0.95, 0.1, 0.0, 0.0, 0.0),
    "panicked": (0.0, 0.1, 0.0, 0.9, 0.2, 0.0, 0.0, 0.0),
    "disgusted": (0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0),
    "surprised": (0.2, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0),
    "calm": (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7),
    "tender": (0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
    "sarcastic": (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.2),
    "nervous": (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.1),
    "confident": (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
    "tired": (0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
    "ecstatic": (0.95, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0),
    "hopeful": (0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3),
    "relieved": (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
    "amused": (0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.2),
    "playful": (0.6, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.1),
    "proud": (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3),
    "triumphant": (0.7, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1),
    "smug": (0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1),
    "deadpan": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7),
    "bored": (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7),
    "lonely": (0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
    "nostalgic": (0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
    "wistful": (0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
    "bitter": (0.0, 0.2, 0.2, 0.0, 0.0, 0.4, 0.1, 0.1),
    "resigned": (0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
    "desperate": (0.0, 0.3, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0),
    "hysterical": (0.3, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0),
    "jealous": (0.0, 0.1, 0.1, 0.0, 0.0, 0.5, 0.2, 0.1),
    "envious": (0.0, 0.1, 0.1, 0.0, 0.0, 0.4, 0.2, 0.1),
    "contemptuous": (0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.4, 0.1),
    "suspicious": (0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.5, 0.2),
    "skeptical": (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.3),
    "curious": (0.2, 0.0, 0.0, 0.0, 0.4, 0.0, 0.1, 0.2),
    "thoughtful": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6),
    "reflective": (0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6),
    "hesitant": (0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.4),
    "shy": (0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.5),
    "guilty": (0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.1, 0.3),
    "remorseful": (0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
    "empathetic": (0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
    "reassuring": (0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
    "eager": (0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1),
    "determined": (0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5),
    "annoyed": (0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.1, 0.2),
    "appalled": (0.0, 0.0, 0.4, 0.1, 0.2, 0.2, 0.0, 0.0),
    "awkward": (0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.4, 0.3),
    "flustered": (0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.2, 0.2),
    "mischievously": (0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.1),
}

_ZONOS_DIM_NAMES = ("happiness", "sadness", "disgust", "fear", "surprise",
                    "anger", "other", "neutral")


def render_zonos(script: str | PerformanceScript) -> str:
    """Canonical markup → Zonos control text.

    Emotion tags become ``[zonos-emo:{...}]`` 8-D vector tags (the
    ZonosBackend strips them into the emotion conditioning tensor —
    the most literal emotion interface in open TTS). Bursts become
    speakable onomatopoeia, pauses become ellipses, delivery tags
    (whisper/shout) map to vector nudges.
    """
    import json as _json
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(
        lambda m: _ONOMATOPOEIA.get(m.group(1).lower(), ""), s)

    def _emo(m: re.Match) -> str:
        name = m.group(1).lower()
        vec = _ZONOS_VECTORS.get(name)
        if vec is None:
            # delivery verbs: nudge the vector instead of dropping them
            if name in ("whisper", "whispers", "whispering"):
                vec = (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9)
            elif name in ("shout", "shouts", "shouting", "scream",
                          "screams"):
                vec = (0.1, 0.0, 0.0, 0.1, 0.1, 0.7, 0.0, 0.0)
            else:
                return ""
        payload = _json.dumps(dict(zip(_ZONOS_DIM_NAMES, vec)),
                              separators=(",", ":"))
        return f" [zonos-emo:{payload}] "

    s = _EMOTION_RE.sub(_emo, s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _PAUSE_RE.sub("... ", s)
    s = _STRONG_RE.sub(lambda m: m.group(1).upper(), s)
    return re.sub(r"\s{2,}", " ", s).strip()


def render_kitten(script: str | PerformanceScript) -> str:
    """Canonical markup → KittenTTS text.

    KittenTTS has no tag vocabulary (fixed voices, no paralinguistics):
    bursts become speakable onomatopoeia, emotions are dropped (the
    engine's emotion-DSP fallback shapes them post-synthesis), pauses
    become commas/ellipses. Speed is a synthesize() kwarg, not markup.
    """
    text, _pauses = render_plain(script, speak_bursts=True)
    # render_plain returns pause points; Kitten reads "..." naturally
    return text


def render_spark(script: str | PerformanceScript) -> str:
    """Canonical markup → Spark-TTS text.

    Spark-TTS takes no markup (control is via --gender/--pitch/--speed
    CLI flags): clean speakable text, bursts as onomatopoeia, pauses as
    ellipses. Stutters stay textual.
    """
    text, _pauses = render_plain(script, speak_bursts=True)
    return text


#: Renderer registry — data-driven, so new backends plug in without
#: touching ``render_for``. Values are ``(render_fn, returns_extra)``.
_RENDERERS: dict[str, tuple] = {
    "fish": (render_fish, False),
    "cosyvoice": (render_cosyvoice, True),
    "bark": (render_bark, False),
    "dia": (render_dia, False),
    "orpheus": (render_orpheus, False),
    "chatterbox": (render_chatterbox, False),
    "omnivoice": (render_omnivoice, False),
    "zonos": (render_zonos, False),
    "kitten": (render_kitten, False),
    "spark": (render_spark, False),
}


def supported_backends() -> list[str]:
    """Backends with a dedicated director renderer."""
    return sorted(_RENDERERS)


def _render_qwen3tts(script: str | PerformanceScript) -> str:
    # native [laugh]/[sigh]/[yawn]/[wow]/[giggle]/[scoff] +
    # [emotion] switching — canonical markup is already its
    # vocabulary. Pauses become ellipses: Qwen reads "[pause:350]"
    # literally, while "..." reads as a natural beat.
    s = _apply_beat(str(script))
    s = _PAUSE_RE.sub("... ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


_RENDERERS["qwen3tts"] = (_render_qwen3tts, False)


def _render_for_registry(backend: str,
                         script: str | PerformanceScript
                         ) -> tuple[str, str | None]:
    """Registry implementation of :func:`render_for` (defined up top).

    Data-driven via :data:`_RENDERERS`; unknown backends fall back to
    ``render_plain`` (speakable onomatopoeia + spliced pauses).
    """
    backend = (backend or "auto").lower()
    entry = _RENDERERS.get(backend)
    if entry is None:
        text, _pauses = render_plain(script)
        return text, None
    fn, returns_extra = entry
    if returns_extra:
        text, instruct = fn(script)
        return text, instruct
    return fn(script), None


def render_for(backend: str,
               script: str | PerformanceScript) -> tuple[str, str | None]:
    """Render a script for a backend. Returns ``(text, extra)`` where
    ``extra`` is the CosyVoice instruction or ``None``."""
    return _render_for_registry(backend, script)


# ---------------------------------------------------------------------------
# House styles — one-word direction presets
# ---------------------------------------------------------------------------

#: Named house styles: preset → director kwargs. The one-liner the
#: module lacked: ``perform(text, **STYLE_PRESETS["audiobook"])``.
#: ``pace`` maps to a speed multiplier the engine understands.
STYLE_PRESETS: dict[str, dict] = {
    "audiobook": {"mood": "calm", "intensity": 4, "effect": None,
                  "speed": 0.95,
                  "blurb": "Warm narrator, unhurried, clear chapter energy"},
    "podcast": {"mood": "confident", "intensity": 5, "effect": None,
                "speed": 1.05,
                "blurb": "Present, conversational, leans into the mic"},
    "announcement": {"mood": "confident", "intensity": 7, "effect": None,
                     "speed": 1.0,
                     "blurb": "Projected, crisp, every word lands"},
    "bedtime": {"mood": "tender", "intensity": 2, "effect": None,
                "speed": 0.85,
                "blurb": "Soft, slow, lights-out storytelling"},
    "hype": {"mood": "excited", "intensity": 8, "effect": None,
             "speed": 1.15,
             "blurb": "High voltage — trailers, intros, game shows"},
    "documentary": {"mood": "thoughtful", "intensity": 4, "effect": None,
                    "speed": 0.95,
                    "blurb": "Measured gravity, Attenborough-adjacent"},
    "comedy": {"mood": "playful", "intensity": 6, "effect": None,
               "speed": 1.1,
               "blurb": "Timing-forward, lands the punchline"},
    "horror": {"mood": "fearful", "intensity": 6, "effect": None,
               "speed": 0.9,
               "blurb": "Low dread, whispers when it counts"},
}
