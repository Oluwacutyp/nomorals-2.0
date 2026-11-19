"""Case- and accent-insensitive matching for search fields, with
a map back to the original text for highlighting — the craft-libs
``craft-i18n`` crate's ``fold.rs``, ported.

:func:`Folder.fold` turns text into a search key: compatibility
forms decomposed (``ﬁ`` → ``fi``, full-width ``Ａ`` → ``a``,
``㎏`` → ``kg``), lower case, ``ß`` → ``ss``, final ``ς`` → ``σ``,
and with accents ignored (the default) combining accents, Hebrew
points and Arabic vowel marks dropped and the few letters with
built-in strokes mapped to their base (``ø``, ``ł``, ``đ``, ``ı``,
``æ`` → ``ae``), and Persian letter forms mapped to the Arabic
ones (``ک`` → ``ك``, ``ی`` → ``ي``, ``ة`` → ``ه``).  Decimal
digits of every script fold to ASCII (``١٢`` → ``12``).
Invisible format characters (soft hyphens, zero-width joiners,
bidi isolates) are dropped too.

Dotted and dotless i follow the locale when accents count.  In
Turkish and Azerbaijani, ``I`` folds to ``ı`` and ``İ`` to ``i``;
elsewhere both fold to ``i`` and ``ı`` stays a letter of its own.
Ignoring accents, all four fold to ``i``, so ``istanbul`` finds
``İSTANBUL`` and ``irmak`` finds ``IRMAK`` and ``ırmak``.  Plain
``str.lower`` follows no locale and turns ``İ`` into ``i`` plus a
combining dot, which nothing typed would match.

The Turkish i handling and sigma folding come from PdfCraft's
``fold_case`` (snipeTR, nevicia2010-hub).

Two notes on the port.  First, offsets are character offsets
into Python strings, where upstream's are byte offsets into Rust
strings; the meaning is identical (a range into the source text).
Second, Unicode normalisation is Python's :mod:`unicodedata`
(NFKD — Unicode 15.0 here) where upstream's ICU data is Unicode
16.0; newly-added characters may fold differently, and the
grapheme-extend test is the combining class plus the small
explicit set below, where upstream reads the ICU property.
"""

from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass, field

__all__ = ["Folded", "Folder"]

# Hangul syllable types, by code point (the Hangul_Syllable_Type
# property, arithmetically): L = leading jamo, V = vowel jamo,
# T = trailing jamo, LV/LVT = precomposed syllables.
_L_Lo, _L_HI = 0x1100, 0x115F
_L2_LO, _L2_HI = 0xA960, 0xA97C
_V_LO, _V_HI = 0x1160, 0x11A7
_V2_LO, _V2_HI = 0xD7B0, 0xD7C6
_T_LO, _T_HI = 0x11A8, 0x11FF
_T2_LO, _T2_HI = 0xD7FB, 0xD7FF
_SYL_LO, _SYL_HI = 0xAC00, 0xD7A3


def _hangul_type(char: str) -> str:
    """``L``/``V``/``T``/``LV``/``LVT`` for ``char``, else ``""``."""
    code = ord(char)
    if _L_Lo <= code <= _L_HI or _L2_LO <= code <= _L2_HI:
        return "L"
    if _V_LO <= code <= _V_HI or _V2_LO <= code <= _V2_HI:
        return "V"
    if _T_LO <= code <= _T_HI or _T2_LO <= code <= _T2_HI:
        return "T"
    if _SYL_LO <= code <= _SYL_HI:
        return "LV" if (code - _SYL_LO) % 28 == 0 else "LVT"
    return ""


def _continues_syllable(prev: str, nxt: str) -> bool:
    """Whether a conjoining jamo of kind ``nxt`` continues the
    syllable a character of kind ``prev`` ends: a vowel after a
    leading consonant, a final consonant after a vowel or an open
    syllable (``가`` and ``ᆨ`` spell ``각``)."""
    return ((prev, nxt) in (("L", "V"), ("V", "T"), ("LV", "T")))


def _is_grapheme_extend(char: str) -> bool:
    """Whether ``char`` extends a grapheme cluster: a combining
    mark (Unicode combining class), a joiner, or a variation
    selector.  Upstream reads the ICU Grapheme_Extend property;
    this is its stdlib shape."""
    code = ord(char)
    if unicodedata.combining(char):
        return True
    return (code in (0x200C, 0x200D, 0xFF9E, 0xFF9F)
            or 0xFE00 <= code <= 0xFE0F
            or 0xE0100 <= code <= 0xE01EF)


def _base_letter(char: str) -> str | None:
    """The base of a lower-case letter whose diacritic is built in
    rather than combining, and the Arabic form of a letter that
    Persian and Urdu write differently: keheh (ک) is kaf (ك),
    Farsi yeh (ی) and alef maksura (ى) are yeh (ي), teh marbuta
    (ة) is heh (ه), alef wasla (ٱ) is alef."""
    return {
        "ک": "ك", "ی": "ي", "ى": "ي", "ة": "ه", "ٱ": "ا",
        "ı": "i", "ø": "o", "ł": "l", "đ": "d", "ħ": "h",
        "ŧ": "t", "æ": "ae", "œ": "oe",
    }.get(char)


def _is_ignorable(char: str) -> bool:
    """Invisible characters that only steer layout: soft hyphen,
    Arabic letter mark, the kashida that stretches Arabic words,
    Mongolian variation selectors, zero-width space and joiners,
    direction marks, embeddings and isolates, word joiner,
    variation selectors, byte order mark and tags."""
    code = ord(char)
    return (code in (0x00AD, 0x061C, 0x0640, 0xFEFF)
            or 0x180B <= code <= 0x180F
            or 0x200B <= code <= 0x200F
            or 0x202A <= code <= 0x202E
            or 0x2060 <= code <= 0x206F
            or 0xFE00 <= code <= 0xFE0F
            or 0xE0000 <= code <= 0xE0FFF)


def _is_accent(char: str) -> bool:
    """Marks that searching without accents ignores: Latin, Greek
    and Cyrillic combining diacritics, Hebrew points and
    cantillation, Arabic vowel marks.  Kana voicing marks, Indic
    and Thai vowel signs are letters' parts, not accents, and
    stay."""
    code = ord(char)
    return (0x0300 <= code <= 0x036F
            or 0x1AB0 <= code <= 0x1AFF
            or 0x1DC0 <= code <= 0x1DFF
            or 0x20D0 <= code <= 0x20FF
            or 0xFE20 <= code <= 0xFE2F
            or 0x0591 <= code <= 0x05BD or code in (
                0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7)
            or 0x0610 <= code <= 0x061A
            or 0x064B <= code <= 0x065F or code == 0x0670
            or 0x06D6 <= code <= 0x06DC
            or 0x06DF <= code <= 0x06E4
            or code in (0x06E7, 0x06E8)
            or 0x06EA <= code <= 0x06ED)


#: The first code point of each run of ten Unicode decimal digits
#: (general category Nd), Unicode 16.0.
_ZEROS = (
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66,
    0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0, 0xF20, 0x1040,
    0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50,
    0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0,
    0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x10D40, 0x11066,
    0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x116D0, 0x116DA, 0x11730, 0x118E0, 0x11950, 0x11BF0,
    0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16130, 0x16A60, 0x16AC0,
    0x16B50, 0x16D70, 0x1CCF0, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC,
    0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E5F1, 0x1E950, 0x1FBF0,
)


def _ascii_digit(char: str) -> str | None:
    """A decimal digit of any script as its ASCII digit: ``١٢``
    and ``१२`` search as ``12``."""
    code = ord(char)
    idx = bisect.bisect_right(_ZEROS, code) - 1
    if idx < 0:
        return None
    value = code - _ZEROS[idx]
    return str(value) if value < 10 else None


@dataclass
class Folded:
    """Folded text and where each part came from."""

    text: str
    #: One (folded offset, source offset) per source cluster, in
    #: order, then both lengths.  A cluster folds to the text up
    #: to the next entry's folded offset, possibly none.
    starts: list[tuple[int, int]] = field(default_factory=list)

    def source_range(self, rng: slice) -> slice:
        """The source range that folded range ``rng`` came from,
        widened to whole source characters with their accents:
        ``ss`` in the folding of ``Straße`` maps back to ``ß``,
        ``e`` in that of ``é`` to the whole ``é``.  An empty range
        maps to an empty one."""
        def cluster(at: int) -> int:
            # The last cluster starting at or before `at`.
            idx = bisect.bisect_right(
                [f for f, _ in self.starts], at) - 1
            return max(0, idx)

        if rng.start == rng.stop:
            start = self.starts[cluster(rng.start)][1]
            return slice(start, start)
        first = cluster(rng.start)
        last = cluster(rng.stop - 1)
        end_idx = min(last + 1, len(self.starts) - 1)
        return slice(self.starts[first][1], self.starts[end_idx][1])

    def find(self, needle: Folded) -> slice | None:
        """The first match of ``needle`` (folded the same way), as
        a range of the source text.  An empty needle matches at
        the start."""
        found = next(iter(self.matches(needle)), None)
        return found

    def matches(self, needle: Folded) -> list[slice]:
        """Every non-overlapping match of ``needle`` (folded the
        same way), as ranges of the source text.  A match may start
        or end inside an expansion (``ss`` of ``ß``, ``fi`` of
        ``ﬁ``) but not leave out a mark that follows a letter, so
        ``re`` doesn't match ``ré`` when accents count."""
        out: list[slice] = []
        n = needle.text
        if n == "":
            return [slice(0, 0)]

        def starts_mark(at: int) -> bool:
            if at >= len(self.text):
                return False
            char = self.text[at]
            if _is_grapheme_extend(char):
                return True
            # Inside one source cluster (not at a cluster start),
            # a jamo that continues the previous syllable is a
            # mark too; standalone jamo are letters.
            starts = [f for f, _ in self.starts]
            if at in starts:
                return False
            if at == 0:
                return False
            prev = self.text[at - 1]
            return _continues_syllable(_hangul_type(prev),
                                       _hangul_type(char))

        start = 0
        while True:
            idx = self.text.find(n, start)
            if idx < 0:
                return out
            end = idx + len(n)
            if not starts_mark(idx) and not starts_mark(end):
                out.append(self.source_range(slice(idx, end)))
                start = end
            else:
                start = idx + 1


class Folder:
    """Folds text for one locale.  Cheap to build and copy."""

    def __init__(self, locale: str = "en") -> None:
        language = locale.split("-", 1)[0].split("_", 1)[0] \
            .split(".", 1)[0].split("@", 1)[0].lower()
        self.turkic = language in ("tr", "az")
        self.accents = False

    def keep_accents(self) -> Folder:
        """Accents count: ``é`` no longer matches ``e``, and a
        match can't end on a letter whose accent it leaves out."""
        other = Folder.__new__(Folder)
        other.turkic = self.turkic
        other.accents = True
        return other

    def fold(self, text: str) -> Folded:
        """``text`` folded, with the map back to ``text``."""
        folded: list[str] = []
        starts: list[tuple[int, int]] = []
        # A cluster is a character and the combining marks after
        # it, folded and mapped as one; conjoining jamo that spell
        # one syllable are one cluster too.
        if text == "":
            return Folded("", [(0, 0)])
        chars = list(text)
        cluster_start = 0
        prev = _hangul_type(chars[0])
        for i in range(1, len(chars)):
            kind = _hangul_type(chars[i])
            if (not _is_grapheme_extend(chars[i])
                    and not _continues_syllable(prev, kind)):
                starts.append((len("".join(folded)), cluster_start))
                folded.append(self._fold_cluster(
                    unicodedata.normalize(
                        "NFKD", text[cluster_start:i])))
                cluster_start = i
            prev = kind
        starts.append((len("".join(folded)), cluster_start))
        folded.append(self._fold_cluster(
            unicodedata.normalize("NFKD", text[cluster_start:])))
        starts.append((len("".join(folded)), len(text)))
        return Folded("".join(folded), starts)

    def key(self, text: str) -> str:
        """``text`` folded, without the map: a key to compare or
        store."""
        return self.fold(text).text

    def contains(self, haystack: str, needle: str) -> bool:
        """Whether ``haystack`` contains ``needle``, both folded."""
        return self.fold(haystack).find(self.fold(needle)) is not None

    def _fold_cluster(self, cluster: str) -> str:
        """One cluster, already in NFKD."""
        out: list[str] = []
        idx = 0
        while idx < len(cluster):
            char = cluster[idx]
            peek = cluster[idx + 1] if idx + 1 < len(cluster) else None
            # `İ` decomposes to `I` and a combining dot above: the
            # dot of a lower-case `i`.
            if char == "I" and peek == "\u0307":
                idx += 2
                out.append("i")
                continue
            if char == "I" and self.turkic and self.accents:
                out.append("ı")
                idx += 1
                continue
            # Persian heh with hamza above (ۀ) decomposes to ae (ە)
            # and hamza.
            if (char == "\u06d5" and not self.accents
                    and peek == "\u0654"):
                idx += 2
                out.append("ه")
                continue
            if char in ("ß", "ẞ"):
                out.append("ss")
                idx += 1
                continue
            digit = _ascii_digit(char)
            if digit is not None:
                out.append(digit)
                idx += 1
                continue
            if char in ("Σ", "σ", "ς"):
                out.append("σ")
                idx += 1
                continue
            if _is_ignorable(char) or (
                    not self.accents and _is_accent(char)):
                idx += 1
                continue
            for lowered in char.lower():
                base = _base_letter(lowered)
                if base is not None and not self.accents:
                    out.append(base)
                else:
                    out.append(lowered)
            idx += 1
        return "".join(out)
