"""Devon Voice Engine — the humanizing director.

Plain text in, *performance script* out. The director is the layer that
makes speech sound like a person talking instead of a robot reading:

- vocal bursts: ``[laugh]`` ``[chuckle]`` ``[sigh]`` ``[breath]``
  ``[cough]`` ``[gasp]`` ``[clearthroat]``
- fillers: ``[um]`` ``[uh]`` ``[erm]``
- ``[pause:MS]`` — real silence, in milliseconds
- ``[stutter]`` — stammers the next word (``really`` → ``r-really``)
- ``<strong>word</strong>`` — emphasis
- ``[rate:slow|fast]`` — pacing hint for the whole line

``direct()`` inserts these cues from punctuation, wording, mood, and a
seeded dice roll, so the same line with the same seed always performs
the same way. The renderers below translate the canonical markup into
whatever each backend natively understands:

- CosyVoice (instruct): ``[laughter]`` / ``[breath]`` / ``<laughter>``
  / ``<strong>`` plus a natural-language emotion/rate instruction.
- Bark: ``[laughs]`` ``[sighs]`` ``[cough]`` ``[gasps]`` …
- XTTS / Kokoro / anything else: cues become speakable words
  (``um,`` ``uh,``) or pause points the engine splices as silence.

Honest approximations are marked in the code: no open model we know
of has a native ``[cough]`` token except Bark, so elsewhere a cough is
rendered as a sharp breath plus a beat of silence. A stutter is always
textual (``w-word``) — every backend renders it.

Stdlib only. No model, no download, no API.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

__all__ = [
    "CANONICAL_BURSTS",
    "PerformanceScript",
    "direct",
    "render_cosyvoice",
    "render_bark",
    "render_plain",
    "strip_to_text",
]

# ---------------------------------------------------------------------------
# Canonical markup
# ---------------------------------------------------------------------------

#: Vocal bursts the director can emit. Every renderer must handle all of
#: these (even if some become approximations — see module docstring).
CANONICAL_BURSTS = (
    "laugh", "chuckle", "sigh", "breath", "cough", "gasp", "clearthroat",
)

CANONICAL_FILLERS = ("um", "uh", "erm")

_BURST_RE = re.compile(
    r"\[(laugh|chuckle|sigh|breath|cough|gasp|clearthroat|um|uh|erm)\]",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"\[pause:(\d{2,4})\]", re.IGNORECASE)
_RATE_RE = re.compile(r"\[rate:(vslow|slow|fast|vfast)\]", re.IGNORECASE)
_STUTTER_RE = re.compile(r"\[stutter\]\s*(\S+)", re.IGNORECASE)
_STRONG_MD_RE = re.compile(r"\*(\S[^*]*\S|\S)\*")  # *word* → <strong>
_EMOTION_RE = re.compile(
    r"\[(happy|excited|sad|angry|annoyed|tired|nervous|whisper|calm)\]",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_WORD_RE = re.compile(r"\S+")
_LAUGH_WORD_RE = re.compile(r"\b(haha+h?|hehe+h?|lol|lmao+|rofl)\b",
                            re.IGNORECASE)


@dataclass
class PerformanceScript:
    """A line of text plus its stage directions."""

    text: str                      # canonical marked-up script
    mood: str = "neutral"
    seed: int | None = None
    cues: list[str] = field(default_factory=list)  # cue kinds inserted

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


# ---------------------------------------------------------------------------
# The director
# ---------------------------------------------------------------------------

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


def _stutter_word(word: str) -> str:
    """``really`` → ``r-really``. Keeps leading punctuation intact."""
    m = re.match(r"^(\W*)(.+)$", word, re.DOTALL)
    if not m:
        return word
    punct, core = m.groups()
    if len(core) < 2:
        return word
    return f"{punct}{core[0]}-{core}"


def _first_content_word(sentence: str) -> str | None:
    for w in _WORD_RE.findall(sentence):
        if re.search(r"[A-Za-z]", w):
            return w
    return None


def direct(text: str, *, mood: str = "neutral", intensity: int = 3,
           seed: int | None = None, lang: str = "en") -> PerformanceScript:
    """Turn plain text into a performance script.

    ``intensity`` 0–5 scales how often cues fire (0 = leave the text
    alone, 5 = full theatre kid). ``seed`` makes the dice reproducible.
    Author-supplied canonical tags (``[laugh]`` …) are always respected
    and never doubled.
    """
    mood = (mood or "neutral").lower()
    profile = _MOOD_PROFILES.get(mood, _MOOD_PROFILES["neutral"])
    intensity = max(0, min(5, intensity))
    rng = random.Random(seed)
    cues: list[str] = []

    # *word* → <strong>word</strong> (author emphasis)
    text = _STRONG_MD_RE.sub(r"<strong>\1</strong>", text)

    scale = intensity / 3.0

    def maybe(p: float) -> bool:
        return rng.random() < p * scale

    sentences = _SENT_SPLIT_RE.split(text.strip())
    out: list[str] = []
    for i, sent in enumerate(sentences):
        words = _WORD_RE.findall(sent)
        n = len(words)

        # laughter words the author typed become real laughs
        if _LAUGH_WORD_RE.search(sent):
            if "[laugh]" not in sent.lower() and "[chuckle]" not in sent.lower():
                sent = _LAUGH_WORD_RE.sub("[laugh]", sent)
                cues.append("laugh")

        # breath at clause boundaries in long sentences
        if n > 16 and maybe(profile["breath"] + 0.25):
            sent = re.sub(r",\s+", ", [breath] ", sent, count=1)
            cues.append("breath")

        # exclamations get a chuckle when the mood is light
        if sent.rstrip().endswith("!") and maybe(profile["chuckle"]):
            if "[chuckle]" not in sent.lower() and "[laugh]" not in sent.lower():
                sent = sent.rstrip() + " [chuckle]"
                cues.append("chuckle")

        # questions asked nervously start with a filler
        if sent.rstrip().endswith("?") and maybe(profile["filler"] + 0.15):
            filler = rng.choice(["[um]", "[uh]"])
            if not sent.lstrip().startswith("["):
                sent = f"{filler} {sent}"
                cues.append("filler")

        # sighs open sad/tired lines
        if i == 0 and maybe(profile["sigh"]) and not sent.lstrip().startswith("["):
            sent = f"[sigh] {sent}"
            cues.append("sigh")

        # stutter the first content word when nervous
        if maybe(profile["stutter"]):
            w = _first_content_word(sent)
            if w and "[stutter]" not in sent.lower():
                sent = sent.replace(w, f"[stutter]{w}", 1)
                cues.append("stutter")

        # short beat before a dramatic final sentence
        if i == len(sentences) - 1 and len(sentences) > 1 and maybe(0.25 * scale):
            sent = f"[pause:350] {sent}"
            cues.append("pause")

        out.append(sent)

    script = " ".join(out).strip()
    rate = profile["rate"]
    if rate and intensity >= 2 and "[rate:" not in script.lower():
        script = f"[rate:{rate}] {script}"
        cues.append("rate")

    return PerformanceScript(text=re.sub(r"\s{2,}", " ", script),
                             mood=mood, seed=seed, cues=cues)


# ---------------------------------------------------------------------------
# Renderers — canonical markup → backend-native
# ---------------------------------------------------------------------------

def _apply_stutter(script: str) -> str:
    return _STUTTER_RE.sub(lambda m: _stutter_word(m.group(1)), script)


def render_cosyvoice(script: str | PerformanceScript) -> tuple[str, str]:
    """Return ``(text, instruct)`` for CosyVoice's instruct mode.

    Maps canonical cues onto CosyVoice's native vocabulary —
    ``[laughter]`` / ``[breath]`` bursts, ``<laughter>`` /
    ``<strong>`` spans — and folds mood + rate into a natural-language
    instruction. Approximations: sigh/cough/gasp/clearthroat have no
    native token, so they become a breath plus a beat of silence.
    """
    s = str(script)
    mood = script.mood if isinstance(script, PerformanceScript) else "neutral"

    instruct_bits: list[str] = []
    emo = _EMOTION_RE.search(s)
    if emo:
        instruct_bits.append(f"Speak in a {emo.group(1).lower()} tone.")
    elif mood not in ("neutral",):
        instruct_bits.append(f"Speak in a {mood} tone.")
    rate = _RATE_RE.search(s)
    if rate:
        instruct_bits.append(
            {"vslow": "Speak very slowly.", "slow": "Speak slowly.",
             "fast": "Speak quickly.", "vfast": "Speak very quickly."}
            [rate.group(1).lower()])

    s = _BURST_RE.sub(
        lambda m: {
            "laugh": "[laughter]", "chuckle": "[laughter]",
            "sigh": "[breath] [pause:300]", "breath": "[breath]",
            # no native cough token: sharp breath + beat (approximation)
            "cough": "[breath] [pause:250]",
            "gasp": "[breath]", "clearthroat": "[breath] [pause:200]",
            "um": "um,", "uh": "uh,", "erm": "erm,",
        }[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub("", s)
    s = _PAUSE_RE.sub("... ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s, " ".join(instruct_bits).strip()


def render_bark(script: str | PerformanceScript) -> str:
    """Canonical markup → Bark's native tags (the paralinguistic king:
    laughs, sighs, coughs, gasps are all first-class here)."""
    s = str(script)
    s = _BURST_RE.sub(
        lambda m: {
            "laugh": "[laughs]", "chuckle": "[chuckles]",
            "sigh": "[sighs]", "breath": "[sighs]",
            "cough": "[cough]", "gasp": "[gasps]",
            "clearthroat": "[clears throat]",
            "um": "um,", "uh": "uh,", "erm": "erm,",
        }[m.group(1).lower()], s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub(lambda m: f"[{m.group(1).lower()}]", s)
    s = _PAUSE_RE.sub(lambda m: "... " if int(m.group(1)) < 500 else "... ... ",
                      s)
    return re.sub(r"\s{2,}", " ", s).strip()


def render_plain(script: str | PerformanceScript) -> tuple[str, list[tuple[int, int]]]:
    """Canonical markup → clean text + pause points.

    For backends with no paralinguistic vocabulary (XTTS, Kokoro):
    fillers stay speakable, bursts are dropped, pauses are returned as
    ``(char_offset, ms)`` so the engine can splice real silence.
    """
    s = str(script)
    s = _BURST_RE.sub(
        lambda m: {"um": "um,", "uh": "uh,", "erm": "erm,"}
        .get(m.group(1).lower(), ""), s)
    s = _apply_stutter(s)
    s = _RATE_RE.sub("", s)
    s = _EMOTION_RE.sub("", s)

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
