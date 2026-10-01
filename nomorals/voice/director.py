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
- delivery: ``[whisper]`` ``[whispering]`` ``[shouting]`` ``[singing]``
  ``[muttering]`` ``[soft]`` ``[loud]``
- vocal bursts: ``[laugh]`` ``[chuckle]`` ``[giggle]`` ``[sigh]``
  ``[breath]`` ``[inhale]`` ``[exhale]`` ``[cough]`` ``[gasp]``
  ``[clearthroat]`` ``[yawn]`` ``[snort]`` ``[tsk]`` ``[sob]`` ``[gulp]``
- fillers: ``[um]`` ``[uh]`` ``[erm]`` ``[hmm]``
- pacing: ``[pause:MS]`` ``[beat]`` ``[rate:vslow|slow|fast|vfast]``
- ``[stutter]`` — stammers the next word (``really`` → ``r-really``)
- ``<strong>word</strong>`` — emphasis (``*word*`` is shorthand)

Renderers translate canonical markup per backend:

- Fish Audio S2: near pass-through — S2 natively accepts 15,000+
  free-form ``[tag]`` directions, so the director speaks its language.
- CosyVoice (instruct): ``[laughter]`` / ``[breath]`` bursts plus a
  natural-language emotion/rate instruction.
- Bark: native ``[laughs]`` ``[sighs]`` ``[cough]`` ``[gasps]`` …
- XTTS / Kokoro / anything else: clean speakable text + pause points
  the engine splices as real silence.

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
    "Intent",
    "PerformanceScript",
    "PerformanceTuner",
    "SegmentDirection",
    "direct",
    "render_bark",
    "render_cosyvoice",
    "render_fish",
    "render_for",
    "render_plain",
    "strip_to_text",
]

# ---------------------------------------------------------------------------
# Canonical markup — the vocabulary
# ---------------------------------------------------------------------------

#: Vocal bursts the director can emit. Every renderer must handle all of
#: these (even if some become approximations — see module docstring).
CANONICAL_BURSTS = (
    "laugh", "chuckle", "giggle", "sigh", "breath", "inhale", "exhale",
    "cough", "gasp", "clearthroat", "yawn", "snort", "tsk", "sob", "gulp",
)

#: Emotion / direction tags (ElevenLabs v3 + Fish S2 vocabularies).
CANONICAL_EMOTIONS = (
    "happy", "sad", "angry", "excited", "nervous", "scared", "proud",
    "sarcastic", "curious", "mischievously", "surprised", "thoughtful",
    "confident", "annoyed", "appalled", "empathetic", "reassuring",
    "tender", "playful", "bored", "determined", "guilty", "shy",
    "calm", "tired",
)

#: Delivery-style tags.
CANONICAL_DELIVERY = (
    "whisper", "whispering", "shouting", "singing", "muttering",
    "soft", "loud",
)

CANONICAL_FILLERS = ("um", "uh", "erm", "hmm")

_BURST_RE = re.compile(
    r"\[(laugh|chuckle|giggle|sigh|breath|inhale|exhale|cough|gasp|"
    r"clearthroat|yawn|snort|tsk|sob|gulp|um|uh|erm|hmm)\]",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"\[pause:(\d{2,4})\]", re.IGNORECASE)
_BEAT_RE = re.compile(r"\[beat\]", re.IGNORECASE)
_RATE_RE = re.compile(r"\[rate:(vslow|slow|fast|vfast)\]", re.IGNORECASE)
_STUTTER_RE = re.compile(r"\[stutter\]\s*(\S+)", re.IGNORECASE)
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
    "😂": "[laugh]", "🤣": "[laugh]", "😹": "[laugh]", "😆": "[chuckle]",
    "😅": "[chuckle]", "😭": "[sob]", "😢": "[sigh]", "😮": "[gasp]",
    "😯": "[gasp]", "🥱": "[yawn]", "🤔": "[hmm]", "😴": "[yawn]",
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
    # burst/filler/stutter probabilities + pacing per mood
    "happy":    {"laugh": 0.45, "chuckle": 0.30, "breath": 0.10, "sigh": 0.0,
                 "filler": 0.05, "stutter": 0.0, "rate": "fast"},
    "excited":  {"laugh": 0.35, "chuckle": 0.25, "breath": 0.15, "sigh": 0.0,
                 "filler": 0.05, "stutter": 0.05, "rate": "fast"},
    "sad":      {"laugh": 0.0, "chuckle": 0.0, "breath": 0.20, "sigh": 0.35,
                 "filler": 0.15, "stutter": 0.05, "rate": "slow"},
    "nervous":  {"laugh": 0.15, "chuckle": 0.10, "breath": 0.20, "sigh": 0.10,
                 "filler": 0.35, "stutter": 0.30, "rate": None},
    "tired":    {"laugh": 0.0, "chuckle": 0.0, "breath": 0.35, "sigh": 0.25,
                 "filler": 0.10, "stutter": 0.0, "rate": "slow"},
    "angry":    {"laugh": 0.0, "chuckle": 0.0, "breath": 0.25, "sigh": 0.15,
                 "filler": 0.0, "stutter": 0.0, "rate": "fast"},
    "calm":     {"laugh": 0.0, "chuckle": 0.05, "breath": 0.25, "sigh": 0.05,
                 "filler": 0.05, "stutter": 0.0, "rate": "slow"},
    "neutral":  {"laugh": 0.05, "chuckle": 0.05, "breath": 0.12, "sigh": 0.03,
                 "filler": 0.08, "stutter": 0.02, "rate": None},
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

        # questions asked nervously start with a filler
        if sent.rstrip().endswith("?") \
                and maybe(self.profile["filler"] + 0.15):
            filler = rng.choice(["[um]", "[uh]", "[hmm]"])
            if not sent.lstrip().startswith("["):
                sent = f"{filler} {sent}"
                cues.append("filler")

        # sighs open sad/tired lines
        if position == 0 and maybe(self.profile["sigh"]) \
                and not sent.lstrip().startswith("["):
            sent = f"[sigh] {sent}"
            cues.append("sigh")

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


def _stutter_word(word: str) -> str:
    """``really`` → ``r-really``. Keeps leading punctuation intact."""
    m = re.match(r"^(\W*)(.+)$", word, re.DOTALL)
    if not m:
        return word
    punct, core = m.groups()
    if len(core) < 2:
        return word
    return f"{punct}{core[0]}-{core}"


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
    return _STUTTER_RE.sub(lambda m: _stutter_word(m.group(1)), script)


def _apply_beat(script: str) -> str:
    return _BEAT_RE.sub("[pause:300]", script)


# canonical burst → Fish S2 native (free-form tags: near pass-through)
_FISH_BURSTS = {
    "laugh": "[laughing]", "chuckle": "[chuckle]", "giggle": "[giggling]",
    "sigh": "[sigh]", "breath": "[inhale]", "inhale": "[inhale]",
    "exhale": "[exhale]", "cough": "[cough]", "gasp": "[gasp]",
    "clearthroat": "[clearing throat]", "yawn": "[yawn]", "snort": "[snort]",
    "tsk": "[tsk]", "sob": "[sobbing]", "gulp": "[gulp]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
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
    "laugh": "[laughter]", "chuckle": "[laughter]", "giggle": "[laughter]",
    "sigh": "[breath] [pause:300]", "breath": "[breath]",
    "inhale": "[breath]", "exhale": "[breath] [pause:200]",
    # no native cough token: sharp breath + beat (approximation)
    "cough": "[breath] [pause:250]",
    "gasp": "[breath]", "clearthroat": "[breath] [pause:200]",
    "yawn": "[breath] [pause:400]", "snort": "[breath]",
    "tsk": "[breath]", "sob": "[breath] [pause:300]",
    "gulp": "[breath]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
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


# canonical burst → Bark native tags (the paralinguistic king)
_BARK_BURSTS = {
    "laugh": "[laughs]", "chuckle": "[chuckles]", "giggle": "[giggles]",
    "sigh": "[sighs]", "breath": "[sighs]", "inhale": "[sighs]",
    "exhale": "[sighs]", "cough": "[cough]", "gasp": "[gasps]",
    "clearthroat": "[clears throat]", "yawn": "[yawns]",
    "snort": "[snorts]", "tsk": "[tsk]", "sob": "[sobs]", "gulp": "[gulps]",
    "um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,",
}


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


def render_plain(script: str | PerformanceScript) -> tuple[str, list[tuple[int, int]]]:
    """Canonical markup → clean text + pause points.

    For backends with no paralinguistic vocabulary (XTTS, Kokoro, HF
    endpoints): fillers stay speakable, bursts are dropped, pauses are
    returned as ``(char_offset, ms)`` so the engine can splice real
    silence.
    """
    s = _apply_beat(str(script))
    s = _BURST_RE.sub(
        lambda m: {"um": "um,", "uh": "uh,", "erm": "erm,", "hmm": "hmm,"}
        .get(m.group(1).lower(), ""), s)
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


def render_for(backend: str,
               script: str | PerformanceScript) -> tuple[str, str | None]:
    """Render a script for a backend. Returns ``(text, extra)`` where
    ``extra`` is the CosyVoice instruction or ``None``."""
    backend = (backend or "auto").lower()
    if backend == "fish":
        return render_fish(script), None
    if backend == "cosyvoice":
        text, instruct = render_cosyvoice(script)
        return text, instruct
    if backend == "bark":
        return render_bark(script), None
    text, _pauses = render_plain(script)
    return text, None
