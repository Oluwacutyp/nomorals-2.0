"""English G2P rung — arpabet phonemes in the CMUdict pattern.

The English counterpart to :mod:`nomorals.media.tonal_g2p`: that module
solves tone languages (phoneme + tone + voice quality per syllable);
this one solves English, whose problem is a deep orthography — the
same word-shape can sound several ways, so the rung is a *lexicon*
first and rules second, exactly the CMUdict architecture (the
``arpabet.rs`` donor pattern from the storyteller-era mining):

1. **Lexicon** — word → arpabet phonemes with lexical stress
   (``0`` none, ``1`` primary, ``2`` secondary on vowels). A curated
   core of CMUdict entries is bundled below; a full CMUdict-format
   file can be layered on top with :func:`load_cmudict` (or the
   ``NM_CMUDICT_PATH`` env var) — coverage is data, reported by
   :func:`lexicon_stats`, never implied.
2. **Letter-to-sound rules** — for out-of-lexicon words, an ordered
   grapheme rule table (digraphs before single letters). This path is
   *approximate by construction* and says so: every result carries
   its ``source`` (``"lexicon"`` / ``"lexicon_user"`` / ``"rules"``)
   and its ``rung`` (``"g2p_en"``), so a consumer can tell a
   dictionary pronunciation from a rule guess at a glance.

Heteronyms (``read``, ``live``, ``use``) resolve to the lexicon's
first entry, as in CMUdict; alternates in a user file ``WORD(1)``
style are parsed but only the first is kept.

Consumers: TTS front-ends — ``UniversalTTS.pronounce()`` exposes the
rung at engine level, and the DiffSinger path in
``nomorals.voice.singing`` prefers it over its naive letter chunker.

Stdlib only. Never raises on text input: an unpronounceable token
yields an empty phoneme list with its source recorded, and callers
fall back honestly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "ARPABET_VOWELS", "ARPABET_CONSONANTS", "G2PWord",
    "g2p_en", "phonemize", "phonemize_flat",
    "load_cmudict", "lexicon_stats", "strip_stress", "base_phoneme",
    "RUNG",
]

#: The rung name recorded on every result — provenance, not prose.
RUNG = "g2p_en"

ARPABET_VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})

ARPABET_CONSONANTS = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
    "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
})

_STRESS_DIGITS = "012"


def base_phoneme(phoneme: str) -> str:
    """``"AH1"`` → ``"AH"``; consonants pass through unchanged."""
    return phoneme.rstrip(_STRESS_DIGITS)


def strip_stress(phoneme: str) -> str:
    """Alias of :func:`base_phoneme`, named for call-site readability."""
    return base_phoneme(phoneme)


# ---------------------------------------------------------------------
# Bundled core lexicon — CMUdict-derived entries (CMUdict 0.7b shapes).
# A curated core, NOT the full dictionary: layer a complete CMUdict
# file on top via load_cmudict() when full coverage matters.
# ---------------------------------------------------------------------

_CORE_LEXICON: dict[str, str] = {
    "the": "DH AH0", "and": "AE1 N D", "of": "AH1 V", "to": "T UW1",
    "in": "IH0 N", "is": "IH1 Z", "you": "Y UW1", "that": "DH AE1 T",
    "it": "IH1 T", "he": "HH IY1", "was": "W AH1 Z", "for": "F AO1 R",
    "on": "AA1 N", "are": "AA1 R", "with": "W IH1 DH", "as": "AE1 Z",
    "i": "AY1", "his": "HH IH1 Z", "they": "DH EY1", "be": "B IY1",
    "at": "AE1 T", "this": "DH IH1 S", "have": "HH AE1 V",
    "from": "F R AH1 M", "or": "AO1 R", "had": "HH AE1 D",
    "by": "B AY1", "word": "W ER1 D", "what": "W AH1 T",
    "some": "S AH1 M", "we": "W IY1", "can": "K AE1 N",
    "out": "AW1 T", "other": "AH1 DH ER0", "were": "W ER1",
    "all": "AO1 L", "there": "DH EH1 R", "when": "W EH1 N",
    "up": "AH1 P", "use": "Y UW1 S", "your": "Y AO1 R",
    "how": "HH AW1", "said": "S EH1 D", "each": "IY1 CH",
    "she": "SH IY1", "which": "W IH1 CH", "their": "DH EH1 R",
    "will": "W IH1 L", "about": "AH0 B AW1 T", "if": "IH1 F",
    "would": "W UH1 D", "them": "DH EH1 M", "so": "S OW1",
    "her": "HH ER1", "than": "DH AE1 N", "then": "DH EH1 N",
    "many": "M EH1 N IY0", "very": "V EH1 R IY0",
    "after": "AE1 F T ER0", "most": "M OW1 S T", "our": "AW1 ER0",
    "just": "JH AH1 S T", "where": "W EH1 R", "much": "M AH1 CH",
    "before": "B IH0 F AO1 R", "here": "HH IY1 R",
    "over": "OW1 V ER0", "think": "TH IH1 NG K",
    "also": "AO1 L S OW0", "around": "AH0 R AW1 N D",
    "form": "F AO1 R M", "again": "AH0 G EH1 N",
    "hello": "HH AH0 L OW1", "world": "W ER1 L D",
    "voice": "V OY1 S", "speech": "S P IY1 CH", "sound": "S AW1 N D",
    "computer": "K AH0 M P Y UW1 T ER0",
    "language": "L AE1 NG G W AH0 JH", "english": "IH1 NG G L IH0 SH",
    "music": "M Y UW1 Z IH0 K", "sing": "S IH1 NG", "song": "S AO1 NG",
    "story": "S T AO1 R IY0", "tell": "T EH1 L", "time": "T AY1 M",
    "people": "P IY1 P AH0 L", "water": "W AO1 T ER0",
    "family": "F AE1 M AH0 L IY0", "friend": "F R EH1 N D",
    "through": "TH R UW1", "thought": "TH AO1 T",
    "right": "R AY1 T", "night": "N AY1 T", "light": "L AY1 T",
    "work": "W ER1 K", "home": "HH OW1 M", "love": "L AH1 V",
    "heart": "HH AA1 R T", "day": "D EY1", "way": "W EY1",
    "make": "M EY1 K", "take": "T EY1 K", "know": "N OW1",
    "go": "G OW1", "see": "S IY1", "good": "G UH1 D",
    "book": "B UH1 K", "look": "L UH1 K", "school": "S K UW1 L",
    "child": "CH AY1 L D", "woman": "W UH1 M AH0 N",
    "women": "W IH1 M AH0 N", "man": "M AE1 N", "men": "M EH1 N",
    "life": "L AY1 F", "live": "L IH1 V", "read": "R EH1 D",
    "write": "R AY1 T", "speak": "S P IY1 K", "talk": "T AO1 K",
    "walk": "W AO1 K", "happy": "HH AE1 P IY0",
    "beautiful": "B Y UW1 T IH0 F AH0 L",
    "morning": "M AO1 R N IH0 NG", "evening": "IY1 V N IH0 NG",
    "question": "K W EH1 S CH AH0 N", "answer": "AE1 N S ER0",
    "problem": "P R AA1 B L AH0 M", "system": "S IH1 S T AH0 M",
    "program": "P R OW1 G R AE2 M", "data": "D EY1 T AH0",
    "model": "M AA1 D AH0 L",
    "one": "W AH1 N", "two": "T UW1", "three": "TH R IY1",
    "first": "F ER1 S T", "new": "N UW1", "old": "OW1 L D",
    "big": "B IH1 G", "small": "S M AO1 L", "long": "L AO1 NG",
    "little": "L IH1 T AH0 L",
    "don't": "D OW1 N T", "can't": "K AE1 N T",
}

_user_lexicon: dict[str, list[str]] = {}
_user_loaded_from: list[str] = []


def _parse_entry(phones: str) -> list[str]:
    return [p for p in phones.split() if p]


_BUNDLED: dict[str, list[str]] = {
    w: _parse_entry(p) for w, p in _CORE_LEXICON.items()}


def load_cmudict(path: str) -> int:
    """Layer a CMUdict-format file over the bundled core lexicon.

    Format per line: ``WORD  PH PH PH`` (comment lines start with
    ``;;;``); ``WORD(1)`` alternates are parsed but only each word's
    first pronunciation is kept, matching the first-entry rule.
    Returns the number of *new* words added. Raises ``OSError`` /
    ``ValueError`` for an unreadable or empty file — a dictionary
    that fails to load must say so, never silently shrink coverage.
    """
    added = 0
    seen: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
            if word in seen:
                continue
            seen.add(word)
            phones = [p for p in parts[1:]
                      if base_phoneme(p) in ARPABET_VOWELS
                      or base_phoneme(p) in ARPABET_CONSONANTS]
            if not phones:
                continue
            if word not in _user_lexicon and word not in _BUNDLED:
                added += 1
            _user_lexicon[word] = phones
    if not seen:
        raise ValueError(f"no CMUdict entries parsed from {path!r}")
    _user_loaded_from.append(path)
    return added


def _env_cmudict() -> None:
    path = os.environ.get("NM_CMUDICT_PATH", "").strip()
    if path and path not in _user_loaded_from:
        try:
            load_cmudict(path)
        except (OSError, ValueError):
            # an env-named dictionary that cannot load must not break
            # synthesis; coverage reporting shows it as absent.
            _user_loaded_from.append(path)


def lexicon_stats() -> dict:
    """Coverage as data: how many words each lexicon layer holds."""
    _env_cmudict()
    return {
        "bundled": len(_BUNDLED),
        "user": len(_user_lexicon),
        "total": len(set(_BUNDLED) | set(_user_lexicon)),
        "user_files": list(_user_loaded_from),
        "rung": RUNG,
    }


# ---------------------------------------------------------------------
# Letter-to-sound fallback — ordered rules, longest match first at
# each position. Approximate by construction; results say "rules".
# ---------------------------------------------------------------------

_DIGRAPHS: list[tuple[str, list[str]]] = [
    ("tch", ["CH"]), ("dge", ["JH"]),
    ("tion", ["SH", "AH", "N"]), ("sion", ["ZH", "AH", "N"]),
    ("cious", ["SH", "AH", "S"]), ("tious", ["SH", "AH", "S"]),
    ("eigh", ["EY"]), ("igh", ["AY"]),
    ("ph", ["F"]), ("ch", ["CH"]), ("sh", ["SH"]), ("th", ["TH"]),
    ("wh", ["W"]), ("ck", ["K"]), ("ng", ["NG"]), ("nk", ["NG", "K"]),
    ("qu", ["K", "W"]), ("wr", ["R"]), ("kn", ["N"]),
    ("ee", ["IY"]), ("ea", ["IY"]), ("ai", ["EY"]), ("ay", ["EY"]),
    ("oa", ["OW"]), ("ow", ["AW"]), ("oo", ["UW"]), ("ou", ["AW"]),
    ("oi", ["OY"]), ("oy", ["OY"]), ("au", ["AO"]), ("aw", ["AO"]),
    ("er", ["ER"]), ("ir", ["ER"]), ("ur", ["ER"]),
    ("ar", ["AA", "R"]), ("or", ["AO", "R"]),
]

_SINGLES: dict[str, list[str]] = {
    "a": ["AE"], "e": ["EH"], "i": ["IH"], "o": ["AA"], "u": ["AH"],
    "b": ["B"], "d": ["D"], "f": ["F"], "g": ["G"], "h": ["HH"],
    "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"],
    "p": ["P"], "r": ["R"], "s": ["S"], "t": ["T"], "v": ["V"],
    "w": ["W"], "x": ["K", "S"], "z": ["Z"],
}


def _lts(word: str) -> list[str]:
    """Letter-to-sound approximation for an out-of-lexicon word."""
    w = word
    # consonant + final silent e (make, stone) — drop the e, the
    # preceding vowel is handled by the plain-letter table as-is;
    # this rung does not fake vowel lengthening it cannot know.
    if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]
    out: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for pat, phones in _DIGRAPHS:
            if w.startswith(pat, i):
                out.extend(phones)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        c = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if c == "c":
            out.append("S" if nxt in "eiy" else "K")
        elif c == "g":
            out.append("JH" if nxt in "ei" and i + 1 == len(w) - 1
                       else "G")
        elif c == "y":
            out.append("IY" if i == len(w) - 1 and i > 0 else "Y")
        elif c in _SINGLES:
            out.extend(_SINGLES[c])
        # unknown characters contribute nothing
        i += 1
    # stress: primary on the first vowel nucleus, none elsewhere —
    # a stated convention of the rules path, not a measurement.
    stressed = False
    result: list[str] = []
    for p in out:
        if p in ARPABET_VOWELS:
            result.append(p + ("1" if not stressed else "0"))
            stressed = True
        else:
            result.append(p)
    return result


# ---------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class G2PWord:
    """One word's pronunciation + where it came from."""

    word: str
    phonemes: list[str] = field(default_factory=list)
    source: str = "rules"          # "lexicon" | "lexicon_user" | "rules"
    rung: str = RUNG

    def to_dict(self) -> dict:
        return {"word": self.word, "phonemes": list(self.phonemes),
                "source": self.source, "rung": self.rung}


_TOKEN_EDGE = "\"'.,;:!?()[]{}“”‘’—–-"


def _normalize(word: str) -> str:
    return word.strip(_TOKEN_EDGE).lower()


def g2p_en(word: str) -> G2PWord:
    """Pronounce one English word: ``(phonemes, source)`` as data.

    Lexicon first (bundled core, then the user layer), letter-to-
    sound rules for the rest. Internal apostrophes are kept
    (``don't`` is a lexicon word). Never raises.
    """
    _env_cmudict()
    raw = (word or "").strip()
    w = _normalize(raw)
    if not w or not any(c.isalpha() for c in w):
        return G2PWord(word=raw, phonemes=[], source="rules")
    if w in _user_lexicon:
        return G2PWord(word=raw, phonemes=list(_user_lexicon[w]),
                       source="lexicon_user")
    if w in _BUNDLED:
        return G2PWord(word=raw, phonemes=list(_BUNDLED[w]),
                       source="lexicon")
    return G2PWord(word=raw, phonemes=_lts(w), source="rules")


def phonemize(text: str) -> list[G2PWord]:
    """Pronounce every word in ``text``, in order."""
    return [g2p_en(tok) for tok in (text or "").split()
            if _normalize(tok)]


def phonemize_flat(text: str, *, with_stress: bool = True) -> list[str]:
    """The flat phoneme stream for ``text`` (word boundaries dropped)."""
    out: list[str] = []
    for gw in phonemize(text):
        for p in gw.phonemes:
            out.append(p if with_stress else base_phoneme(p))
    return out
