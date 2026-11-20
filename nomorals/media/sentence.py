"""Sentence tokenization + segmentation for TTS input.

Design pattern from echelon/sentence.rs (BSD-4-Clause; Copyright (c)
2016, 2020, Brandon Thomas — full licence text at the end of this
module). The Rust crate lexes English sentences into semantic tokens
for TTS systems: a ``SentenceTokenizer`` splits on whitespace and
then runs three repair passes — end-punctuation separation, integer /
real-number materialization, word classification — into a 12-kind
``Token`` enum. All code here is written from scratch for Devon; the
design, the token kinds, and the test vectors follow the donor.

What this module is
-------------------
The sentence engine Devon's voice stack was missing. Three places
each carried their own copy of one naive regex —
``re.compile(r"(?<=[.!?…])\\s+")`` — in ``voice/director.py``
(``PerformanceTuner.segment``), ``voice/tts.py``
(``_split_sentences``) and ``voice/livekit_call.py`` (``_speak``).
All three split ``"Dr. Smith went home."`` into ``["Dr.", "Smith
went home."]`` — a wrong TTS pause after "Dr.". This module is the
one engine those three now share:

* :class:`SentenceTokenizer` — ``tokenize(text)`` and the
  iterator-first ``tokenize_iter(text)`` (the census line calls the
  donor iterator-first; here the iterator is real, not a list).
  A single left-to-right scanner, so every token carries ``start``
  / ``end`` offsets into the original text. It reproduces every
  donor test vector exactly.
* :func:`segment` / :func:`segment_iter` / :func:`segment_text` —
  sentence segmentation over the token stream. A terminal
  Period/Question/Exclamation/Ellipsis ends a sentence unless the
  period belongs to a known abbreviation, an acronym (``U.S.A.``),
  a dotted abbreviation (``e.g.``), a decimal, or is followed by a
  lowercase word. A closing quote after terminal punctuation
  belongs to the sentence it closes. A single-capital-letter
  initial (``J. Smith``) is treated as no boundary AND flagged
  :attr:`Sentence.ambiguous` — the one shape this segmenter cannot
  decide, named instead of silently guessed.

Deliberate divergences from the donor (pinned by tests)
-------------------------------------------------------
* **One scanner, not three repair passes.** The donor splits on
  whitespace and then repairs tokens; the repairs cannot produce
  offsets, and the end-punctuation pass only handles a *single*
  punctuation character, so ``"yes!!!!!"`` falls through every
  match arm and stays one ``Unknown`` token (the donor's own test
  only asserts it does not loop forever). The scanner emits one
  ``Punctuation`` token per mark.
* **Ellipsis and em-dash are real tokens.** The donor TODOs both.
  ``"..."`` / ``"…"`` → ``Ellipsis``; ``"—"`` / ``"--"`` → ``EmDash``.
* **The roadmap items are built.** The donor lists abbreviations,
  dates, times, currency/prices, emoji, ordinals and ratios as not
  yet implemented. This module classifies ``Currency`` (``$12.50``,
  ``£3``, ``€1,000.00``, ``₦1,500``), ``Percentage`` (``45%``),
  ``Ordinal`` (``1st``, ``21st``), ``Date`` (``2026-11-20``,
  ``20/11/2026``), ``Time`` (``10:30``, ``10:30am``) and ``Emoji``
  as their own kinds instead of ``Unknown``.
* **Quotes are tokens.** The donor has no quote handling (``"good"``
  stays ``Unknown``). Double quotes (``"“”``) are ``Punctuation``
  ``Quote`` tokens, so the segmenter can attach a closing quote to
  the sentence it closes. Single quotes stay in the apostrophe
  family, exactly like the donor (``'em``, ``O'Grady's'``).
* **Segmentation exists at all.** The donor tokenizes one sentence;
  it never finds where sentences end. The segmenter is the new
  work, built on the token stream.
* **Python errors, not a Rust enum.** There is no error type here:
  tokenizing and segmenting never raise on any string input —
  unclassifiable input is an ``Unknown`` token, never a crash.
  Callers that must never raise (``voice/tts.py``) keep their own
  try/except around the call.

Abbreviation rules (the segmenter's knowledge)
----------------------------------------------
``TITLES`` never end a sentence (``Dr.``, ``Mr.``, ``Prof.`` ...).
``ENDABLE`` abbreviations end one only when the next token is a
capitalized word — ``"No. 5"`` and ``"etc. and so on"`` stay whole,
``"Fig. 3 shows"`` stays whole, ``"etc. He left"`` splits. ``am``
and ``pm`` are deliberately NOT abbreviations here: ``"10 am."``
ends a sentence, and the donor's own normalization table only
knows them as time words. ``St.`` is a title (``St. Louis``) even
though it can also be a street — the merge error is the lesser
evil for TTS, and it is documented, not hidden.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Punctuation",
    "Token",
    "TokenKind",
    "Sentence",
    "SentenceTokenizer",
    "tokenize",
    "tokenize_iter",
    "segment",
    "segment_iter",
    "segment_text",
    "ABBREVIATIONS",
    "TITLES",
    "ENDABLE",
]

# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------


class Punctuation(Enum):
    """Punctuation marks (the donor's seven, plus the donor's TODOs)."""

    Colon = ":"
    Comma = ","
    Dash = "-"
    Ellipsis = "..."
    EmDash = "—"
    Exclamation = "!"
    Period = "."
    Question = "?"
    Quote = '"'
    Semicolon = ";"


#: Single-char punctuation → Punctuation member.
_CHAR_PUNCT = {
    ":": Punctuation.Colon,
    ",": Punctuation.Comma,
    "-": Punctuation.Dash,
    "!": Punctuation.Exclamation,
    ".": Punctuation.Period,
    "?": Punctuation.Question,
    ";": Punctuation.Semicolon,
    '"': Punctuation.Quote,
    "“": Punctuation.Quote,
    "”": Punctuation.Quote,
    "‘": Punctuation.Quote,
}

#: Token kinds. The donor's twelve, plus the donor roadmap kinds.
TokenKind = str

KINDS = (
    "ApostrophenatedWord",
    "CommaFormattedInteger",
    "CommaFormattedRealNumber",
    "Currency",
    "Date",
    "Emoji",
    "Hashtag",
    "HyphenatedWord",
    "Integer",
    "Ordinal",
    "Percentage",
    "Punctuation",
    "RealNumber",
    "Time",
    "Url",
    "UsernameMention",
    "Word",
    "Unknown",
)


@dataclass(frozen=True)
class Token:
    """One lexed token, with offsets into the original text.

    ``kind`` is one of :data:`KINDS`; ``text`` is the exact slice
    ``source[start:end]``. ``punct`` is set only for
    ``Punctuation`` tokens. ``kind`` is a plain string (not an enum)
    so callers can compare without importing an enum type, the way
    the G2P rung names sources as strings.
    """

    kind: TokenKind
    text: str
    start: int
    end: int
    punct: Punctuation | None = None

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return f"{self.kind}({self.text!r})"


@dataclass(frozen=True)
class Sentence:
    """One segmented sentence.

    ``text`` is the exact slice ``source[start:end]`` (leading and
    trailing whitespace stripped, interior kept verbatim).
    ``terminator`` is the punctuation kind that ended it
    (``Period`` / ``Question`` / ``Exclamation`` / ``Ellipsis``),
    or ``None`` when the text simply ran out. ``ambiguous`` is True
    only when a boundary decision inside this sentence was a
    judgement call — a single-capital-letter initial (``J. Smith``)
    or a period after an unknown word followed by a lowercase word
    — rather than knowledge from the abbreviation tables.
    """

    text: str
    start: int
    end: int
    tokens: tuple[Token, ...] = ()
    terminator: Punctuation | None = None
    ambiguous: bool = False


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------

#: Titles: a period after one of these never ends a sentence.
TITLES = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "gen", "col", "capt", "lt",
    "sgt", "rev", "hon", "pres", "gov", "sen", "rep", "adm",
    "jr", "sr", "st", "vs",
})

#: Abbreviations that CAN end a sentence: they suppress a boundary
#: only when the next token is a lowercase word or a number
#: ("No. 5", "etc. and so on", "Fig. 3 shows" stay whole;
#: "etc. He left" splits). "am"/"pm" are deliberately absent:
#: "10 am." ends a sentence.
ENDABLE = frozenset({
    "etc", "approx", "dept", "est", "fig", "vol", "pp", "ed",
    "al", "no", "inc", "ltd", "corp", "co", "mt", "ave", "rd",
    "blvd", "min", "max",
})

#: The full abbreviation set (titles + endable).
ABBREVIATIONS = TITLES | ENDABLE

# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = "$£€₦¥₹"
_QUOTE_CHARS = set(_CHAR_PUNCT)  # double-quote family handled inline
_APOS = "'’"  # straight + curly apostrophes inside words
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # emoticons, pictographs, transport, etc.
    (0x2600, 0x27BF),    # misc symbols, dingbats
    (0x2B00, 0x2BFF),    # misc symbols and arrows
    (0xFE00, 0xFE0F),    # variation selectors
    (0x1F1E6, 0x1F1FF),  # regional indicators (flags)
)


def _is_emoji_char(ch: str) -> bool:
    """True for a character that belongs in an Emoji token."""
    cp = ord(ch)
    return (unicodedata.category(ch) == "So"
            or any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES))


def _is_word_char(ch: str) -> bool:
    """A character that belongs inside a word token (donor WORD is
    ``^\\w+$`` — letters, digits, underscore — plus apostrophes)."""
    return ch.isalnum() or ch == "_" or ch in _APOS


_NUMBER_RE = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")
_ORDINAL_RE = re.compile(r"\d+(st|nd|rd|th)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?\s?[ap]\.?m",
                      re.IGNORECASE)
_TIME_BARE_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?")


class SentenceTokenizer:
    """Lex English text into semantic tokens for TTS input.

    The donor's ``SentenceTokenizer`` (``sentence.rs``), written from
    scratch: same token kinds, same test vectors, but a single
    left-to-right scanner instead of three repair passes — so every
    token carries offsets, multi-punctuation runs tokenize
    properly, and the donor roadmap kinds exist.
    """

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize ``text`` into a list of :class:`Token`."""
        return list(self.tokenize_iter(text))

    def tokenize_iter(self, text: str) -> Iterator[Token]:
        """Yield tokens lazily, in source order. Never raises on any
        string input; unclassifiable characters become ``Unknown``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            start = i
            # Ellipsis: "..." or the single char "…".
            if text.startswith("...", i):
                yield Token("Punctuation", "...", start, start + 3,
                            Punctuation.Ellipsis)
                i += 3
                continue
            if ch == "…":
                yield Token("Punctuation", "…", start, start + 1,
                            Punctuation.Ellipsis)
                i += 1
                continue
            # Em-dash: "—" or "--".
            if ch == "—" or text.startswith("--", i):
                width = 1 if ch == "—" else 2
                yield Token("Punctuation", text[i:i + width], start,
                            start + width, Punctuation.EmDash)
                i += width
                continue
            # Emoji run (ZWJ / variation selector / skin tone join).
            if _is_emoji_char(ch):
                j = i + 1
                while j < n and (_is_emoji_char(text[j])
                                 or text[j] in ("\u200d", "\ufe0f")
                                 or 0x1F3FB <= ord(text[j]) <= 0x1F3FF):
                    j += 1
                yield Token("Emoji", text[i:j], start, j)
                i = j
                continue
            # Double-quote family (single quotes stay apostrophes).
            if ch in _CHAR_PUNCT and ch not in _APOS:
                yield Token("Punctuation", ch, start, start + 1,
                            _CHAR_PUNCT[ch])
                i += 1
                continue
            # Currency: symbol + number ("$12.50", "₦1,500").
            if ch in _CURRENCY_SYMBOLS and i + 1 < n \
                    and text[i + 1].isdigit():
                m = _NUMBER_RE.match(text, i + 1)
                assert m is not None  # guarded by isdigit above
                yield Token("Currency", text[i:m.end()], start, m.end())
                i = m.end()
                continue
            # Date / time (checked before the bare-number scan, since
            # they contain "-" / "/" / ":").
            m = _DATE_RE.match(text, i)
            if m and (m.end() == n or not text[m.end()].isalnum()):
                yield Token("Date", m.group(0), start, m.end())
                i = m.end()
                continue
            m = _TIME_RE.match(text, i) or _TIME_BARE_RE.match(text, i)
            if m and (m.end() == n or not text[m.end()].isalnum()):
                yield Token("Time", m.group(0), start, m.end())
                i = m.end()
                continue
            # Numbers: ordinals, percentages, reals, integers.
            if ch.isdigit():
                m = _ORDINAL_RE.match(text, i)
                if m:
                    yield Token("Ordinal", m.group(0), start, m.end())
                    i = m.end()
                    continue
                m = _NUMBER_RE.match(text, i)
                assert m is not None
                num = m.group(0)
                end = m.end()
                if end < n and text[end] == "%":
                    yield Token("Percentage", text[i:end + 1], start,
                                end + 1)
                    i = end + 1
                    continue
                if "," in num and "." in num:
                    kind = "CommaFormattedRealNumber"
                elif "," in num:
                    kind = "CommaFormattedInteger"
                elif "." in num:
                    kind = "RealNumber"
                else:
                    kind = "Integer"
                yield Token(kind, num, start, end)
                i = end
                continue
            # URL: http(s):// or www. — trailing sentence punctuation
            # peels off ("https://www.google.com." → Url + Period).
            if text.startswith(("http://", "https://", "www."), i):
                j = i
                while j < n and not text[j].isspace():
                    j += 1
                raw = text[i:j]
                # Peel trailing sentence punctuation, unless the URL
                # carries a query (then the tail is part of the URL).
                if "?" not in raw and "=" not in raw:
                    k = len(raw)
                    while k > 0 and raw[k - 1] in ".,;:!?":
                        k -= 1
                    if k < len(raw):
                        yield Token("Url", raw[:k], start, start + k)
                        for pch in raw[k:]:
                            yield Token(
                                "Punctuation", pch, start + k,
                                start + k + 1, _CHAR_PUNCT[pch])
                            k += 1
                        i = j
                        continue
                yield Token("Url", raw, start, j)
                i = j
                continue
            # Hashtag / mention.
            if ch in "#@" and i + 1 < n and (
                    text[i + 1].isalnum() or text[i + 1] == "_"):
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                kind = "Hashtag" if ch == "#" else "UsernameMention"
                yield Token(kind, text[i:j], start, j)
                i = j
                continue
            # Word-ish run: letters/digits/underscore/apostrophe, with
            # hyphens that join letters ("double-check"). A trailing
            # hyphen not followed by a letter is a Dash, not word
            # material ("but- no" → Word + Dash).
            if ch.isalpha() or ch == "_" or (
                    ch in _APOS and i + 1 < n and text[i + 1].isalpha()):
                j = i
                while j < n:
                    c = text[j]
                    if _is_word_char(c) or (
                            c == "-" and j + 1 < n
                            and text[j + 1].isalpha()):
                        j += 1
                    else:
                        break
                word = text[i:j]
                # "version2.0": a dotted number glued to a word is not
                # any token kind — the donor leaves it Unknown. Keep
                # the dots and digits with the word, as one Unknown.
                if j < n and text[j] == "." and j + 1 < n \
                        and text[j + 1].isdigit():
                    k = j
                    while k < n and (text[k].isdigit()
                                     or text[k] == "."):
                        k += 1
                    word = text[i:k]
                    yield Token("Unknown", word, start, k)
                    i = k
                    continue
                if "-" in word:
                    kind = "HyphenatedWord"
                elif any(a in word for a in _APOS):
                    kind = "ApostrophenatedWord"
                else:
                    kind = "Word"
                yield Token(kind, word, start, j)
                i = j
                continue
            # Anything else: one Unknown character (guarantees progress).
            yield Token("Unknown", ch, start, start + 1)
            i += 1


_DEFAULT_TOKENIZER = SentenceTokenizer()


def tokenize(text: str) -> list[Token]:
    """Tokenize ``text`` with the default tokenizer."""
    return _DEFAULT_TOKENIZER.tokenize(text)


def tokenize_iter(text: str) -> Iterator[Token]:
    """Yield tokens for ``text`` lazily (iterator-first)."""
    yield from _DEFAULT_TOKENIZER.tokenize_iter(text)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

#: Punctuation kinds that can end a sentence.
_TERMINATORS = frozenset({
    Punctuation.Period,
    Punctuation.Question,
    Punctuation.Exclamation,
    Punctuation.Ellipsis,
})


def _next_word(tokens: list[Token], idx: int) -> Token | None:
    """The next word-ish token after index ``idx`` (skipping
    punctuation), or None."""
    for tok in tokens[idx + 1:]:
        if tok.kind == "Punctuation":
            continue
        return tok
    return None


def _prev_word(tokens: list[Token], idx: int) -> Token | None:
    """The nearest word-ish token before index ``idx``."""
    for tok in reversed(tokens[:idx]):
        if tok.kind == "Punctuation":
            continue
        return tok
    return None


def _is_boundary(tokens: list[Token], idx: int, text: str,
                 ) -> tuple[bool, bool]:
    """Decide whether the terminal punctuation at ``idx`` ends a
    sentence. Returns ``(is_boundary, ambiguous)``.

    The punctuation token itself is never a word, so the word before
    it is the last word token; the word after it is the next
    word-ish token. Rules, in order:

    1. ``?`` / ``!`` always end a sentence.
    2. An ellipsis ends one only when the next word is capitalized
       (or there is no next word) — ``"Wait... what?"`` splits,
       ``"He trailed off... and left"`` does not.
    3. A period after a single letter: a following single letter
       continues a dotted run (``U.S.A.``); a following lowercase
       word continues it too (``e.g. this``); a following
       capitalized word after an UPPERCASE letter is an initial
       (``J. Smith``) — no boundary, flagged ambiguous; after a
       lowercase letter it is a dotted abbreviation at sentence end
       (``10 a.m. He left``) — a boundary.
    4. A period after a title (``Dr.``, ``Mr.`` ...) never ends one.
    5. A period after an endable abbreviation (``etc.``, ``No.`` ...)
       ends one unless the next token is a lowercase word or a
       number (``No. 5``, ``etc. and so on`` stay whole).
    6. A period after any other word ends one unless the next word
       starts lowercase (an unknown abbreviation mid-sentence) —
       that guess is flagged ambiguous.
    """
    tok = tokens[idx]
    punct = tok.punct
    if punct in (Punctuation.Question, Punctuation.Exclamation):
        return True, False
    if punct is Punctuation.Ellipsis:
        nxt = _next_word(tokens, idx)
        if nxt is None:
            return True, False
        return (nxt.text[:1].isupper(), False)
    # From here: a Period.
    prev = _prev_word(tokens, idx)
    nxt = _next_word(tokens, idx)
    if prev is None:
        return True, False
    word = prev.text
    # Single-letter word before the period.
    if len(word) == 1 and word.isalpha():
        if nxt is not None and len(nxt.text) == 1 \
                and nxt.text.isalpha():
            return False, False  # dotted run continues: U.S.A.
        if nxt is not None and nxt.text[:1].islower():
            return False, False  # e.g. this / U.S.A. is
        if nxt is not None and nxt.text[:1].isupper():
            if word.isupper():
                return False, True  # initial: J. Smith
            return True, False  # dotted abbrev at sentence end: a.m.
        return True, False
    low = word.lower()
    if low in TITLES:
        return False, False
    if low in ENDABLE:
        if nxt is None:
            return True, False
        if nxt.text[:1].islower() or nxt.kind in (
                "Integer", "RealNumber", "CommaFormattedInteger",
                "CommaFormattedRealNumber", "Ordinal", "Date", "Time"):
            return False, False
        return True, False
    # Any other word: a lowercase next word means the period did not
    # end a sentence — a guess on an unknown abbreviation, flagged.
    if nxt is not None and nxt.text[:1].islower():
        return False, True
    return True, False


def _gap_has_blank_line(text: str, start: int, end: int) -> bool:
    """True when the raw text between two offsets holds a blank
    line (two newlines with only whitespace between them)."""
    return "\n" in text[start:end] and \
        bool(re.search(r"\n\s*\n", text[start:end]))


def segment(text: str) -> list[Sentence]:
    """Split ``text`` into :class:`Sentence` objects."""
    return list(segment_iter(text))


def segment_iter(text: str) -> Iterator[Sentence]:
    """Yield sentences lazily. Never raises on any string input."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    tokens = tokenize(text)
    if not tokens:
        return
    current: list[Token] = []
    ambiguous = False
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        # A blank line in the raw gap before this token forces a
        # boundary, even without terminal punctuation.
        if current and _gap_has_blank_line(
                text, current[-1].end, tok.start):
            yield _finish_sentence(text, current, None, ambiguous)
            current, ambiguous = [], False
        current.append(tok)
        if tok.kind == "Punctuation" and tok.punct in _TERMINATORS:
            boundary, amb = _is_boundary(tokens, idx, text)
            ambiguous = ambiguous or amb
            if boundary:
                # A closing quote right after the terminator belongs
                # to this sentence: 'He said "go." She left.'
                j = idx + 1
                while j < len(tokens) \
                        and tokens[j].kind == "Punctuation" \
                        and tokens[j].punct is Punctuation.Quote:
                    current.append(tokens[j])
                    j += 1
                yield _finish_sentence(text, current, tok.punct,
                                       ambiguous)
                current, ambiguous = [], False
                idx = j
                continue
        idx += 1
    if current:
        yield _finish_sentence(text, current, None, ambiguous)


def _finish_sentence(text: str, tokens: list[Token],
                     terminator: Punctuation | None,
                     ambiguous: bool) -> Sentence:
    """Build a Sentence from its tokens (offsets into ``text``)."""
    start = tokens[0].start
    end = tokens[-1].end
    return Sentence(
        text=text[start:end],
        start=start,
        end=end,
        tokens=tuple(tokens),
        terminator=terminator,
        ambiguous=ambiguous,
    )


def segment_text(text: str) -> list[str]:
    """Split ``text`` into plain sentence strings.

    The one call the voice stack shares: ``PerformanceTuner.segment``,
    ``tts._split_sentences`` and ``livekit_call._speak`` all route
    through here. Never raises on any string input.
    """
    return [s.text for s in segment(text)]


# ---------------------------------------------------------------------------
# BSD 4-Clause licence (donor: echelon/sentence.rs)
# ---------------------------------------------------------------------------
# Copyright (c) 2016, 2020, Brandon Thomas. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. All advertising materials mentioning features or use of this
#    software must display the following acknowledgement:
#
#    This product includes software developed by Brandon Thomas
#    (bt@brand.io, echelon@gmail.com).
#
# 4. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
