"""Arpabet dictionary tiers — the CMUdict-pattern lexicon layer.

Design pattern from echelon/arpabet.rs (BSD-4-Clause; Copyright (c)
2016-2018, Brandon Thomas — full licence text at the end of this
module). The Rust crate is a library for speech synthesis over
Carnegie Mellon University's CMUdict: a dictionary maps lowercase
words to Arpabet polyphones (phoneme lists with lexical stress), a
strict parser reads CMUdict-format text, and dictionaries combine
with later layers winning. All code here is written from scratch
for Devon; the design, the phoneme inventory, and the error shapes
follow the donor.

What this module is
-------------------
The tiered-dictionary layer the English G2P rung was missing. Before
it, :mod:`nomorals.media.g2p_en` kept its lexicon as two bare
module-level dicts (bundled core + user layer) with the lookup order
hard-coded in one function, no way to inspect which layer answered,
no validation that a loaded file's phonemes are even Arpabet, and
malformed lines silently skipped. This module gives the lexicon a
real shape:

* :class:`Phoneme` — a vowel or consonant with a stress value, as
  data instead of a bare string with a digit suffix. The full
  Arpabet inventory: 31 consonants, 19 vowel bases x 4 stresses
  (unknown / none / primary / secondary) = 76 vowel phonemes,
  107 entries in :data:`PHONEME_MAP`.
* :class:`Arpabet` — one dictionary: word -> polyphone, with the
  donor's ``get_polyphone`` / ``combine`` (second wins) /
  ``merge_from`` (in place, other wins) / ``insert`` (returns the
  old entry) / ``remove`` / ``keys`` / ``len``.
* :func:`load_from_str` / :func:`load_from_file` — the strict
  CMUdict parser: ``WORD  PH PH PH`` lines, ``;;;`` comments,
  words lowercased, phonemes uppercased, and named failures
  (:class:`ArpabetError`) — ``empty_file`` for an empty file,
  ``invalid_format`` naming the line number and the offending
  text, never a silent skip.
* :class:`Tier` / :class:`TieredLexicon` — named layers in
  precedence order (overrides > user > core): lookup names the
  tier that answered, stats report per-tier counts, and a file
  loads into exactly one tier.
* :class:`SentenceToken` / :func:`tokenize_sentence` — the donor's
  sentence-token extension: phoneme tokens plus punctuation
  tokens (``[start]`` / ``[space]`` / ``[comma]`` / ... /
  ``[end]``), with the donor's u8 encoding for ML consumers
  (consonants 1-31, vowels 101-176, punctuation 201-209,
  ``[end]`` 254).

Deliberate divergences from the donor (pinned by tests)
-------------------------------------------------------
* **First entry wins within a file.** The Rust parser inserts
  every line, so the *last* entry for a word wins; Devon's
  loader keeps the *first* (``WORD(1)`` alternates are parsed
  but only the first pronunciation is kept), matching the
  first-entry rule the existing G2P tests pin.
* **Later files win within a tier.** Loading a second file into
  the same tier overwrites words it redefines (the donor's
  ``merge_from`` semantic), so a corrected dictionary takes
  effect without editing files.
* **Python errors, not a Rust enum.** :class:`ArpabetError`
  subclasses :class:`ValueError` so the existing "unreadable or
  empty file raises ``OSError`` / ``ValueError``" contract holds;
  unreadable files raise the raw :class:`OSError` (the donor's
  ``Io`` case, in Python idiom).

Stdlib only. Never invents a pronunciation: a word the tiers do
not know is simply absent, and the G2P rung falls through to the
learned model and the rules — the tiers never fake a lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "VOWEL_STRESS_UNKNOWN", "VOWEL_STRESS_NONE",
    "VOWEL_STRESS_PRIMARY", "VOWEL_STRESS_SECONDARY",
    "VOWEL_BASES", "ALL_CONSONANTS", "ALL_VOWELS", "ALL_PUNCTUATION",
    "PHONEME_MAP", "PUNCTUATION_CODES",
    "Phoneme", "ArpabetError", "Arpabet",
    "load_from_str", "load_from_file",
    "Tier", "TieredLexicon",
    "Punctuation", "SentenceToken", "tokenize_sentence",
    "sentence_to_u8", "token_to_u8",
]

#: Stress values, as the donor's ``VowelStress`` (unknown is -1).
VOWEL_STRESS_UNKNOWN = -1
VOWEL_STRESS_NONE = 0
VOWEL_STRESS_PRIMARY = 1
VOWEL_STRESS_SECONDARY = 2

#: The 19 Arpabet vowel bases, in the donor's constants order.
VOWEL_BASES: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER",
    "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX",
)

#: The 31 Arpabet consonants, in the donor's constants order.
ALL_CONSONANTS: tuple[str, ...] = (
    "B", "CH", "D", "DH", "DX", "EL", "EM", "EN", "F", "G", "HH",
    "JH", "K", "L", "M", "N", "NG", "NX", "P", "Q", "R", "S",
    "SH", "T", "TH", "V", "W", "WH", "Y", "Z", "ZH",
)

#: The 76 vowel phonemes: every base in all four stresses, in the
#: donor's order (bare, 0, 1, 2 per base).
ALL_VOWELS: tuple[str, ...] = tuple(
    base + suffix
    for base in VOWEL_BASES
    for suffix in ("", "0", "1", "2")
)

#: The 10 punctuation tokens, in the donor's constants order.
ALL_PUNCTUATION: tuple[str, ...] = (
    "[start]", "[space]", "[comma]", "[period]", "[question]",
    "[exclamation]", "[interjection]", "[quote]", "[ellipsis]",
    "[end]",
)

#: u8 codes for the punctuation tokens (the donor's extension
#: mapping; ``[end]`` is deliberately 254, not 210).
PUNCTUATION_CODES: dict[str, int] = {
    "[start]": 201, "[space]": 202, "[comma]": 203,
    "[period]": 204, "[question]": 205, "[exclamation]": 206,
    "[interjection]": 207, "[quote]": 208, "[ellipsis]": 209,
    "[end]": 254,
}

#: Punctuation characters in running text, mapped to tokens.
_CHAR_PUNCTUATION: dict[str, str] = {
    ",": "[comma]", ".": "[period]", "?": "[question]",
    "!": "[exclamation]", "-": "[interjection]",
    "\u2014": "[interjection]", "\u2013": "[interjection]",
    '"': "[quote]", "\u201c": "[quote]", "\u201d": "[quote]",
}


@dataclass(frozen=True)
class Phoneme:
    """One Arpabet phoneme: a base plus a stress value.

    Consonants carry no stress (``stress`` is always
    :data:`VOWEL_STRESS_UNKNOWN` for them); vowels carry -1
    (unknown, written bare: ``"AA"``), 0, 1, or 2.
    """

    base: str
    stress: int = VOWEL_STRESS_UNKNOWN
    kind: str = "consonant"  # "consonant" | "vowel"

    def to_str(self) -> str:
        """``"EH1"`` for a primary-stress vowel, ``"B"`` for a
        consonant, ``"AA"`` for an unstressed-marked vowel."""
        if self.kind != "vowel" or self.stress < 0:
            return self.base
        return f"{self.base}{self.stress}"

    def to_stressless(self) -> str:
        """The base with the stress digit dropped (``"EH1"`` →
        ``"EH"``)."""
        return self.base

    @property
    def is_vowel(self) -> bool:
        return self.kind == "vowel"

    @property
    def is_consonant(self) -> bool:
        return self.kind == "consonant"

    @classmethod
    def from_str(cls, text: str) -> Phoneme:
        """Parse ``"EH1"`` / ``"B"`` into a :class:`Phoneme`.

        Raises :class:`ArpabetError` (``string_parse``) for text
        that is not an Arpabet phoneme — the donor's
        ``TryFrom<&str>``.
        """
        phoneme = PHONEME_MAP.get((text or "").strip().upper())
        if phoneme is None:
            raise ArpabetError(
                "string_parse",
                description=f"Not a phoneme: '{text}'")
        return phoneme


def _build_phoneme_map() -> dict[str, Phoneme]:
    mapping: dict[str, Phoneme] = {}
    for consonant in ALL_CONSONANTS:
        mapping[consonant] = Phoneme(base=consonant, kind="consonant")
    for base in VOWEL_BASES:
        for stress, suffix in (
                (VOWEL_STRESS_UNKNOWN, ""),
                (VOWEL_STRESS_NONE, "0"),
                (VOWEL_STRESS_PRIMARY, "1"),
                (VOWEL_STRESS_SECONDARY, "2")):
            mapping[base + suffix] = Phoneme(
                base=base, stress=stress, kind="vowel")
    return mapping


#: All 107 Arpabet phonemes by their string form.
PHONEME_MAP: dict[str, Phoneme] = _build_phoneme_map()


class ArpabetError(ValueError):
    """A named failure from the Arpabet layer.

    Mirrors the donor's ``ArpabetError`` enum: ``empty_file``
    ("The file was empty."), ``invalid_format`` (names the line
    number and the offending text), ``string_parse`` (names what
    failed to parse), ``io`` (wraps an :class:`OSError`).
    Devon's own layer adds two kinds the donor has no occasion
    for: ``invalid_phoneme`` (a phoneme handed to
    :meth:`Arpabet.insert` that is not in the Arpabet inventory)
    and ``unknown_tier`` (a load into a tier that does not
    exist). A :class:`ValueError` subclass, so callers that
    already catch ``ValueError`` for bad dictionary files keep
    working.
    """

    def __init__(self, kind: str, *, line_number: int = 0,
                 text: str = "", description: str = "",
                 cause: BaseException | None = None) -> None:
        self.kind = kind
        self.line_number = line_number
        self.text = text
        self.description = description
        self.cause = cause
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.kind == "empty_file":
            return "The file was empty."
        if self.kind == "invalid_format":
            return (f"Invalid format on line {self.line_number}: "
                    f"{self.text}")
        if self.kind == "string_parse":
            return f"Parse error: {self.description}"
        if self.kind == "invalid_phoneme":
            return f"Invalid phoneme: {self.description}"
        if self.kind == "unknown_tier":
            return f"Unknown tier: {self.description}"
        if self.kind == "io":
            return str(self.cause) if self.cause is not None else "I/O error"
        return self.description or "Arpabet error"


# ---------------------------------------------------------------------
# One dictionary
# ---------------------------------------------------------------------

class Arpabet:
    """A dictionary of lowercase words to Arpabet polyphones.

    The donor's ``Arpabet`` struct: one map, looked up by word,
    combined with later layers winning. Phoneme lists are stored
    as plain ``list[str]`` (``["T", "EH1", "S", "T"]``) so the
    existing G2P consumers need no conversion; every lookup
    validates through :data:`PHONEME_MAP` on the way in.
    """

    def __init__(self, mapping: dict[str, list[str]] | None = None
                 ) -> None:
        # The passed mapping is owned, not copied: callers that
        # keep a reference (the G2P rung's module-level dicts)
        # keep seeing the tier's contents.
        self._dictionary: dict[str, list[str]] = (
            mapping if mapping is not None else {})

    @classmethod
    def new(cls) -> Arpabet:
        """An empty dictionary."""
        return cls({})

    @classmethod
    def from_map(cls, mapping: dict[str, list[str]]) -> Arpabet:
        """A dictionary over an existing word -> polyphone map."""
        return cls(mapping)

    def get_polyphone(self, word: str) -> list[str] | None:
        """A copy of ``word``'s polyphone, or ``None`` when the
        word is not in the dictionary. Never fakes an entry."""
        found = self._dictionary.get((word or "").lower())
        return list(found) if found is not None else None

    def get_polyphone_ref(self, word: str) -> list[str] | None:
        """The stored polyphone itself (no copy), or ``None``."""
        return self._dictionary.get((word or "").lower())

    def combine(self, other: Arpabet) -> Arpabet:
        """A new dictionary of this one plus ``other`` — entries
        in ``other`` take precedence, as the donor's ``combine``."""
        merged = dict(self._dictionary)
        for key, value in other._dictionary.items():
            merged[key] = list(value)
        return Arpabet(merged)

    def merge_from(self, other: Arpabet) -> None:
        """Merge ``other`` into this dictionary in place — entries
        in ``other`` override existing ones, as the donor's
        ``merge_from``."""
        for key, value in other._dictionary.items():
            self._dictionary[key] = list(value)

    def insert(self, word: str, phones: list[str]) -> list[str] | None:
        """Insert (or replace) ``word``'s entry, returning the old
        polyphone, or ``None`` when the word was not present."""
        key = (word or "").lower()
        self._validate(phones, key)
        return self._insert(key, phones)

    def _insert(self, key: str, phones: list[str]) -> list[str] | None:
        old = self._dictionary.get(key)
        self._dictionary[key] = list(phones)
        return list(old) if old is not None else None

    def remove(self, word: str) -> list[str] | None:
        """Remove ``word``'s entry, returning it, or ``None``."""
        removed = self._dictionary.pop((word or "").lower(), None)
        return list(removed) if removed is not None else None

    def __iter__(self):
        """Iterate the words, in sorted order like keys()."""
        return iter(self.keys())

    def keys(self) -> list[str]:
        """The words in the dictionary, in sorted order (the
        donor's keys walk in random order; sorted is the
        deterministic improvement)."""
        return sorted(self._dictionary)

    def __len__(self) -> int:
        return len(self._dictionary)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self._dictionary

    @staticmethod
    def _validate(phones: list[str], word: str) -> None:
        for phone in phones:
            if phone not in PHONEME_MAP:
                raise ArpabetError(
                    "invalid_phoneme",
                    description=f"Not a phoneme: '{phone}' (word '{word}')")


# ---------------------------------------------------------------------
# The strict CMUdict parser
# ---------------------------------------------------------------------

_ALTERNATE_SUFFIX = re.compile(r"\(\d+\)$")


def load_from_str(text: str) -> Arpabet:
    """Parse CMUdict-format text into an :class:`Arpabet`.

    One entry per line: ``WORD  PH PH PH``. Comment lines start
    with ``;;;``; blank lines are skipped; words are lowercased;
    phonemes are uppercased. ``WORD(1)`` alternates are parsed
    but only each word's first pronunciation is kept.

    Raises :class:`ArpabetError`: ``empty_file`` when nothing
    parses, ``invalid_format`` (line number + offending text)
    for a line with no phonemes or an unknown phoneme — a bad
    line fails the load loudly instead of shrinking coverage
    silently.
    """
    mapping: dict[str, list[str]] = {}
    for line_number, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ArpabetError(
                "invalid_format", line_number=line_number, text=raw)
        word = _ALTERNATE_SUFFIX.sub("", parts[0]).lower()
        if word in mapping:
            continue  # first entry wins; alternates are not kept
        phones = [p.upper() for p in parts[1:]]
        for phone in phones:
            if phone not in PHONEME_MAP:
                raise ArpabetError(
                    "invalid_format", line_number=line_number, text=raw)
        mapping[word] = phones
    if not mapping:
        raise ArpabetError("empty_file")
    return Arpabet(mapping)


def load_from_file(path: str) -> Arpabet:
    """Parse a CMUdict-format file into an :class:`Arpabet`.

    Raises :class:`ArpabetError` (``io``) for an unreadable file
    and :class:`ArpabetError` for an empty or malformed one.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise ArpabetError("io", description=str(exc),
                           cause=exc) from exc
    return load_from_str(text)


# ---------------------------------------------------------------------
# Named tiers, in precedence order
# ---------------------------------------------------------------------

@dataclass
class Tier:
    """One named layer of the lexicon: a dictionary plus the
    source label its words answer with and the files it was
    loaded from."""

    name: str
    source: str
    arpabet: Arpabet = field(default_factory=Arpabet)
    files: list[str] = field(default_factory=list)


class TieredLexicon:
    """The lexicon as ordered tiers — overrides > user > core.

    Lookup walks the tiers in order and names the tier that
    answered, so a pronunciation's provenance is data, not a
    guess. Files load into exactly one tier; within a tier,
    first entry wins per file and later files win across files.
    """

    def __init__(self, tiers: list[Tier] | None = None) -> None:
        self._tiers: list[Tier] = list(tiers or [])

    def add_tier(self, tier: Tier, *, before: str | None = None
                 ) -> None:
        """Add a tier; ``before`` inserts it ahead of the named
        tier (higher precedence), otherwise it goes last."""
        if before is None:
            self._tiers.append(tier)
            return
        for index, existing in enumerate(self._tiers):
            if existing.name == before:
                self._tiers.insert(index, tier)
                return
        self._tiers.append(tier)

    def tier(self, name: str) -> Tier | None:
        """The tier called ``name``, or ``None``."""
        for tier in self._tiers:
            if tier.name == name:
                return tier
        return None

    def tier_names(self) -> list[str]:
        """Tier names in precedence order (highest first)."""
        return [tier.name for tier in self._tiers]

    def lookup(self, word: str
               ) -> tuple[list[str], str, str] | None:
        """``(phonemes, tier_name, source)`` for ``word``, or
        ``None`` when no tier knows it."""
        key = (word or "").lower()
        if not key:
            return None
        for tier in self._tiers:
            found = tier.arpabet.get_polyphone(key)
            if found is not None:
                return found, tier.name, tier.source
        return None

    def load_into_tier(self, path: str, tier_name: str) -> int:
        """Load a CMUdict-format file into the named tier.

        Returns the number of words that are *new coverage* for
        the whole lexicon (not already in any tier). Words the file
        redefines in the same tier take the new pronunciation
        (later files win); words already answered by another tier
        are still recorded in the tier but do not count as new
        coverage.
        Raises :class:`ArpabetError` for an unreadable (``io``),
        empty, or malformed file.
        """
        tier = self.tier(tier_name)
        if tier is None:
            raise ArpabetError(
                "unknown_tier",
                description=f"unknown tier: '{tier_name}'")
        parsed = load_from_file(path)
        before: set[str] = set()
        for other in self._tiers:
            before.update(other.arpabet.keys())
        added = 0
        for word in parsed:
            if word not in before:
                added += 1
            tier.arpabet.insert(word, parsed.get_polyphone_ref(word)
                                or [])
        if path not in tier.files:
            tier.files.append(path)
        return added

    def stats(self) -> dict:
        """Per-tier counts, total distinct words, and files."""
        counts = {tier.name: len(tier.arpabet) for tier in self._tiers}
        total = len(set().union(
            *(set(tier.arpabet.keys()) for tier in self._tiers))
            ) if self._tiers else 0
        return {
            "tiers": counts,
            "total": total,
            "files": {tier.name: list(tier.files)
                      for tier in self._tiers},
        }

    def combine(self) -> Arpabet:
        """One flattened dictionary: lowest tier first, each
        higher tier overriding, as chained ``combine`` calls."""
        merged = Arpabet.new()
        for tier in reversed(self._tiers):
            merged = merged.combine(tier.arpabet)
        return merged

    def __len__(self) -> int:
        return self.stats()["total"]


# ---------------------------------------------------------------------
# Sentence tokens (the donor's extension module)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SentenceToken:
    """One token of a sentence: a phoneme or a punctuation
    token, as the donor's ``SentenceToken``."""

    kind: str  # "phoneme" | "punctuation"
    value: str

    def to_str(self) -> str:
        return self.value

    def to_u8(self) -> int:
        """The donor's numeric encoding: consonants 1-31,
        vowels 101-176, punctuation 201-209, ``[end]`` 254."""
        return token_to_u8(self)


def token_to_u8(token: SentenceToken) -> int:
    """Encode one :class:`SentenceToken` as a u8, per the donor's
    ``From<SentenceToken> for u8``."""
    if token.kind == "punctuation":
        return PUNCTUATION_CODES[token.value]
    phoneme = PHONEME_MAP.get(token.value)
    if phoneme is None:
        raise ArpabetError(
            "string_parse",
            description=f"Not a phoneme: '{token.value}'")
    if phoneme.is_consonant:
        return ALL_CONSONANTS.index(phoneme.base) + 1
    base_index = VOWEL_BASES.index(phoneme.base)
    return 101 + base_index * 4 + (phoneme.stress + 1)


class Punctuation:
    """The punctuation tokens and their string forms."""

    START = "[start]"
    SPACE = "[space]"
    COMMA = "[comma]"
    PERIOD = "[period]"
    QUESTION = "[question]"
    EXCLAMATION = "[exclamation]"
    INTERJECTION = "[interjection]"
    QUOTE = "[quote]"
    ELLIPSIS = "[ellipsis]"
    END = "[end]"

    @staticmethod
    def to_str(token: str) -> str:
        """The token's string form (it already is one)."""
        return token

    @staticmethod
    def to_u8(token: str) -> int:
        """The donor's numeric code for a punctuation token."""
        return PUNCTUATION_CODES[token]


@dataclass
class SentenceTokens:
    """A tokenized sentence: the tokens plus the words no tier
    knew, so callers can say what was skipped instead of
    guessing."""

    tokens: list[SentenceToken] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def tokenize_sentence(text: str, lookup) -> SentenceTokens:
    """Tokenize ``text`` into phoneme + punctuation tokens.

    ``lookup`` is a callable taking a word and returning a
    phoneme list or ``None`` (e.g. ``TieredLexicon.lookup``
    wrapped to its first element, or ``Arpabet.get_polyphone``).
    Words are lowercased for lookup; punctuation characters
    become punctuation tokens; the stream is framed with
    ``[start]`` / ``[end]`` and words are separated by
    ``[space]``. A word no tier knows contributes no phoneme
    tokens and is listed in ``unknown`` — never faked.
    """
    result = SentenceTokens()
    result.tokens.append(SentenceToken("punctuation", "[start]"))
    first_word = True
    for raw_word in (text or "").split():
        word, trail = _split_trailing_punct(raw_word)
        if not word:
            if trail:
                for token in trail:
                    result.tokens.append(
                        SentenceToken("punctuation", token))
            continue
        if not first_word:
            result.tokens.append(SentenceToken("punctuation", "[space]"))
        first_word = False
        phones = lookup(word.lower())
        if phones is None:
            result.unknown.append(word)
        else:
            for phone in phones:
                result.tokens.append(SentenceToken("phoneme", phone))
        for token in trail:
            result.tokens.append(SentenceToken("punctuation", token))
    result.tokens.append(SentenceToken("punctuation", "[end]"))
    return result


def _split_trailing_punct(raw: str) -> tuple[str, list[str]]:
    """Split trailing punctuation off a word: ``"hello..."`` →
    ``("hello", ["[ellipsis]"])``; ``"wait,"`` → ``("wait",
    ["[comma]"])``. Internal apostrophes stay (``don't``)."""
    word = raw
    trail: list[str] = []
    while word and word[-1] in _CHAR_PUNCTUATION:
        if word.endswith("..."):
            trail.insert(0, "[ellipsis]")
            word = word[:-3]
        else:
            trail.insert(0, _CHAR_PUNCTUATION[word[-1]])
            word = word[:-1]
    return word, trail


def sentence_to_u8(text: str, lookup) -> tuple[list[int], list[str]]:
    """The u8 encoding of ``text``'s tokens, plus the unknown
    words, for ML consumers (the donor's token-code shape)."""
    tokenized = tokenize_sentence(text, lookup)
    return ([token.to_u8() for token in tokenized.tokens],
            tokenized.unknown)


# ---------------------------------------------------------------------
# Licence (BSD-4-Clause, echelon/arpabet.rs)
# ---------------------------------------------------------------------
# Copyright (c) 2016-2018, Brandon Thomas. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
