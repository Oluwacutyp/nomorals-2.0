"""The i18n formatting rung (Craft-Libs port of the ``i18n``
crate).

The formatting rung Devon's own i18n runtime names as missing:
numbers, percentages and file sizes written for a region, numbers
read back in any digits, and the region itself — the territory's
paper and measurement, the system's separators, the clock it
chose. The message catalogs live where they always have
(``nomorals/i18n/``); this module writes the numbers inside
them.

Separators and digits come from CLDR: the tables here carry the
locales the organs use, with the root (``1,234.5``) for anything
else. The parse is the upstream reader: any Unicode decimal
digits from one set, the format's marks, a lone mark that cannot
be grouping, groups of three or Indian 2-2-3.
"""

from __future__ import annotations

import math
import os
from enum import Enum

__all__ = [
    "Digits",
    "FileSize",
    "Format",
    "HourCycle",
    "Measurement",
    "NO_GROUPING",
    "Paper",
    "Region",
    "SizeUnit",
    "file_size",
    "parse_number",
]


class _NoGrouping:
    """The sentinel for "no grouping at all" (``1234567``),
    distinct from ``None`` ("follow the locale")."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NO_GROUPING"


NO_GROUPING = _NoGrouping()


class Digits(Enum):
    """Which digits numbers are written in."""
    LOCALE = "locale"
    LATIN = "latin"


class Measurement(Enum):
    """How lengths, distances and weights are measured."""
    METRIC = "metric"
    US = "us"


class Paper(Enum):
    """The paper a new document starts on."""
    A4 = "a4"
    LETTER = "letter"


class HourCycle(Enum):
    """A clock the user chose, where the system says so."""
    H12 = "h12"
    H23 = "h23"


class SizeUnit(Enum):
    """A file-size unit, smallest first."""
    BYTE = 0
    KILO = 1
    MEGA = 2
    GIGA = 3
    TERA = 4


# ---------------------------------------------------------------- regions

#: Territories whose default paper is US Letter (CLDR paperSize).
_LETTER = {"BZ", "CA", "CL", "CO", "CR", "GT", "MX", "NI", "PA", "PH",
           "PR", "SV", "US", "VE"}
#: Territories that measure in inches (CLDR measurementSystem US).
_INCHES = {"LR", "MM", "US"}

# Digit systems by their Unicode extension name, as the first code
# point of each run of ten decimal digits.
_DIGIT_ZEROS = {
    "latn": 0x30, "arab": 0x660, "arabext": 0x6F0, "deva": 0x966,
    "beng": 0x9E6, "guru": 0xA66, "gujr": 0xAE6, "orya": 0xB66,
    "taml": 0xBE6, "telu": 0xC66, "knda": 0xCE6, "mlym": 0xD66,
    "thai": 0xE50, "laoo": 0xED0, "tibt": 0xF20, "mymr": 0x1040,
    "fullwide": 0xFF10,
}


def _is_territory(part: str) -> bool:
    return ((len(part) == 2 and part.isalpha() and part.isascii())
            or (len(part) == 3 and part.isdigit()))


def _title_case(part: str) -> str:
    lower = part.lower()
    return lower[:1].upper() + lower[1:] if lower else ""


def _is_c_locale(tag: str) -> bool:
    first = tag.replace("@", ".").split(".", 1)[0].strip()
    return first.lower() in ("c", "posix")


class _Tag:
    """The parts of a locale tag the region uses."""

    def __init__(self, language="en", script="", territory="",
                 measurement=None, hour_cycle=None, digits=""):
        self.language = language
        self.script = script
        self.territory = territory
        self.measurement = measurement
        self.hour_cycle = hour_cycle
        self.digits = digits

    @classmethod
    def parse(cls, tag: str):
        """Parse a tag, or ``None`` when it is not one. Locale
        tags are short; anything long is junk and only its start
        is read."""
        tag = "".join(list(tag)[:64])
        if "@" in tag:
            tag, modifier = tag.split("@", 1)
        else:
            modifier = ""
        tag = tag.split(".", 1)[0].strip()
        if _is_c_locale(tag):
            return None
        parts = tag.replace("_", "-").split("-")
        language = parts[0] if parts else ""
        if not 2 <= len(language) <= 3 \
                or not language.isalpha() or not language.isascii():
            return None
        out = cls(language=language.lower())
        rest = parts[1:]
        extension = None
        index = 0
        while index < len(rest):
            part = rest[index]
            index += 1
            if len(part) == 1:
                extension = part.lower()
                break
            if (not out.territory and not out.script and len(part) == 4
                    and part.isalpha() and part.isascii()):
                out.script = _title_case(part)
            elif not out.territory and _is_territory(part):
                out.territory = part.upper()
        keywords: list[tuple[str, str]] = []
        if extension == "u":
            key = ""
            while index < len(rest):
                part = rest[index]
                index += 1
                if len(part) == 1:
                    break
                if len(part) == 2:
                    key = part.lower()
                elif key:
                    keywords.append((key, part.lower()))
                    key = ""
        for item in modifier.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                keywords.append((key.strip().lower(),
                                 value.strip().lower()))
            elif item.lower() == "latin":
                out.script = "Latn"
            elif item.lower() == "cyrillic":
                out.script = "Cyrl"
        for key, value in keywords:
            if key == "ms" and value in ("metric", "uksystem"):
                out.measurement = Measurement.METRIC
            elif key == "ms" and value == "ussystem":
                out.measurement = Measurement.US
            elif key == "hc" and value in ("h11", "h12"):
                out.hour_cycle = HourCycle.H12
            elif key == "hc" and value in ("h23", "h24"):
                out.hour_cycle = HourCycle.H23
            elif key in ("nu", "numbers") and 3 <= len(value) <= 8 \
                    and value.isalnum() and value.isascii():
                out.digits = value
            elif (key == "rg" and len(value) == 6
                  and value.endswith("zzzz")
                  and _is_territory(value[:2])):
                out.territory = value[:2].upper()
        if not out.territory and out.language == "en":
            out.territory = "US"
        return out

    def bcp47(self) -> str:
        tag = "-".join(p for p in (self.language, self.script,
                                   self.territory) if p)
        if self.digits:
            tag += f"-u-nu-{self.digits}"
        return tag


def _single_char(text: str | None):
    """A separator that is one character; longer ones cannot be
    carried."""
    if text is None or len(text) != 1:
        return None
    return text


def _grouping_value(text: str | None):
    """A grouping separator the system sets: none at all when
    empty, else one character."""
    if text is None:
        return None
    if text == "":
        return NO_GROUPING
    return _single_char(text)


def _hour_cycle_of(pattern: str):
    """The clock a time pattern (``HH:mm``, ``h:mm tt``) uses;
    quoted literals are skipped."""
    quoted = False
    for char in pattern:
        if char == "'":
            quoted = not quoted
        elif char in ("H", "k") and not quoted:
            return HourCycle.H23
        elif char in ("h", "K") and not quoted:
            return HourCycle.H12
    return None


class Region:
    """The conventions numbers, times and measurements follow.
    Read it with :meth:`Region.system` or, for a saved choice,
    :meth:`Region.from_setting`; the fields are public so an app
    can keep a separate choice of, say, measurement."""

    def __init__(self, numbers: str = "en-US", time: str = "en-US",
                 measurement: Measurement = Measurement.US,
                 paper: Paper = Paper.LETTER, decimal=None,
                 grouping=None, hour_cycle=None) -> None:
        self.numbers = numbers
        self.time = time
        self.measurement = measurement
        self.paper = paper
        #: The decimal separator the system sets; ``None`` follows
        #: the locale.
        self.decimal = decimal
        #: The grouping separator the system sets: ``NO_GROUPING``
        #: for none at all; ``None`` follows the locale.
        self.grouping = grouping
        #: The clock the system sets; ``None`` follows the locale.
        self.hour_cycle = hour_cycle

    def __eq__(self, other) -> bool:
        return (isinstance(other, Region)
                and self.numbers == other.numbers
                and self.time == other.time
                and self.measurement == other.measurement
                and self.paper == other.paper
                and self.decimal == other.decimal
                and self.grouping == other.grouping
                and self.hour_cycle == other.hour_cycle)

    def __repr__(self) -> str:
        return (f"Region({self.numbers!r}, time={self.time!r}, "
                f"measurement={self.measurement.value!r}, "
                f"paper={self.paper.value!r})")

    @classmethod
    def default(cls) -> Region:
        """``en-US``: Letter, inches."""
        return cls()

    @classmethod
    def _from_parts(cls, tag: _Tag) -> Region:
        territory = tag.territory
        measurement = tag.measurement or (
            Measurement.US if territory in _INCHES
            else Measurement.METRIC)
        return cls(numbers=tag.bcp47(), time=tag.bcp47(),
                   measurement=measurement,
                   paper=Paper.LETTER if territory in _LETTER
                   else Paper.A4,
                   hour_cycle=tag.hour_cycle)

    @classmethod
    def from_tag(cls, tag: str) -> Region:
        """The region a BCP 47 or POSIX locale tag names. ``C``
        and ``POSIX`` are the C locale: English, A4, metric, a
        ``.`` decimal and no grouping. An empty or unreadable tag
        is ``en-US``; a bare ``en`` is the United States, any
        other bare language has no territory."""
        parsed = _Tag.parse(tag)
        if parsed is not None:
            return cls._from_parts(parsed)
        if _is_c_locale(tag):
            region = cls.default()
            region.measurement = Measurement.METRIC
            region.paper = Paper.A4
            region.decimal = "."
            region.grouping = NO_GROUPING
            return region
        return cls.default()

    @classmethod
    def from_setting(cls, setting: str) -> Region:
        """The region for a saved setting: ``""`` or ``auto`` (any
        case) is the system, anything else a locale tag."""
        setting = setting.strip()
        if not setting or setting.lower() == "auto":
            return cls.system()
        return cls.from_tag(setting)

    @classmethod
    def system(cls) -> Region:
        """The system's region, from the POSIX locale variables
        on this platform."""
        return cls.from_posix(os.environ.get, platform=lambda: None)

    @classmethod
    def from_posix(cls, var, platform=None) -> Region:
        """The region from the POSIX locale variables ``var``
        reads. An empty variable is unset; ``C`` in ``LC_ALL``
        or an ``LC_*`` variable is the C locale's conventions.
        ``LANG`` set to ``C`` or unset gives way to the
        platform's preferred locale."""
        def read(name: str):
            value = var(name)
            if value is None or not value.strip():
                return None
            return value

        all_value = read("LC_ALL")
        if all_value is not None:
            return cls.from_tag(all_value)
        lang = read("LANG")
        if lang is not None and _is_c_locale(lang):
            lang = None
        if lang is None and platform is not None:
            lang = platform()
        if lang is None:
            lang = ""

        def tag(name: str) -> str:
            return read(name) or lang

        time = cls.from_tag(tag("LC_TIME"))
        region = cls.from_tag(tag("LC_NUMERIC"))
        region.time = time.time
        region.hour_cycle = time.hour_cycle
        region.measurement = cls.from_tag(
            tag("LC_MEASUREMENT")).measurement
        region.paper = cls.from_tag(tag("LC_PAPER")).paper
        return region

    @classmethod
    def from_windows(cls, value, platform=None) -> Region:
        """The region from the values under Windows'
        ``Control Panel\\International`` that ``value`` reads.
        ``LocaleName`` is the format, else the platform's
        preferred locale."""
        name = value("LocaleName")
        if name is None and platform is not None:
            name = platform()
        region = cls.from_tag(name or "")
        region.decimal = _single_char(value("sDecimal"))
        region.grouping = _grouping_value(value("sThousand"))
        measure = value("iMeasure")
        if measure is not None:
            if measure.strip() == "0":
                region.measurement = Measurement.METRIC
            elif measure.strip() == "1":
                region.measurement = Measurement.US
        paper_size = value("iPaperSize")
        if paper_size is not None:
            if paper_size.strip() == "1":
                region.paper = Paper.LETTER
            elif paper_size.strip() == "9":
                region.paper = Paper.A4
        cycle = _hour_cycle_of(value("sShortTime") or "")
        if cycle is not None:
            region.hour_cycle = cycle
        return region


# ---------------------------------------------------------------- symbols

def _base_row(language: str, territory: str):
    """A locale's own (decimal point, grouping separator,
    Indian grouping?) from CLDR, before any digit system is
    applied."""
    if language == "de":
        return (".", "'", False) if territory == "CH" else (",", ".", False)
    if language == "fr":
        return (",", "\u202f", False)
    if language == "hi":
        return (".", ",", True)
    if language == "bn":
        return (".", ",", True)
    if language == "te":
        return (".", ",", True)
    if language in ("ru", "sv", "nb", "sk"):
        return (",", "\u00a0", False)
    if language == "es":
        return (".", ",", False) if territory == "419" else (",", ".", False)
    if language in ("pt", "tr", "lo"):
        return (",", ".", False)
    if language == "en":
        if territory in ("PL", "DE"):
            return (",", ".", False)
        return (".", ",", False)
    if language == "ar":
        if territory == "MA":
            return (",", ".", False)
        if territory:
            return ("\u066b", "\u066c", False)
        return (".", ",", False)
    if language == "fa":
        return ("\u066b", "\u066c", False)
    return (".", ",", False)


def _resolve(numbers: str, digits: Digits):
    """(point, group, digit zero, Indian grouping) for a number
    locale and a digit choice."""
    tag = _Tag.parse(numbers) or _Tag()
    language, territory = tag.language, tag.territory
    if digits == Digits.LATIN:
        zero = 0x30
    elif tag.digits and tag.digits in _DIGIT_ZEROS:
        zero = _DIGIT_ZEROS[tag.digits]
    elif language == "ar" and territory and territory != "MA":
        zero = 0x660
    elif language == "fa":
        zero = 0x6F0
    elif language == "bn":
        zero = 0x9E6
    else:
        zero = 0x30
    point, group, indian = _base_row(language, territory)
    if zero == 0x30:
        # Latin digits take the locale's Latin symbols: Arabic and
        # Persian write 1,234.5, everything else keeps its own.
        if language in ("ar", "fa") and territory != "MA":
            point, group = ".", ","
        elif language == "ar" and territory == "MA":
            point, group = ",", "."
    return point, group, zero, indian


# The percent affixes (CLDR 48.2.1): (locale, Latin digits?,
# prefix, suffix). A locale without a row falls back to its
# parent, then to ("", "%").
_PERCENT = [
    ("ar", True, "", "\u200e%\u200e"),
    ("ar-BH", False, "", "\u066a\u061c"),
    ("ar-DJ", False, "", "\u066a\u061c"),
    ("ar-EG", False, "", "\u066a\u061c"),
    ("ar-ER", False, "", "\u066a\u061c"),
    ("ar-IL", False, "", "\u066a\u061c"),
    ("ar-IQ", False, "", "\u066a\u061c"),
    ("ar-JO", False, "", "\u066a\u061c"),
    ("ar-KM", False, "", "\u066a\u061c"),
    ("ar-KW", False, "", "\u066a\u061c"),
    ("ar-LB", False, "", "\u066a\u061c"),
    ("ar-MR", False, "", "\u066a\u061c"),
    ("ar-OM", False, "", "\u066a\u061c"),
    ("ar-PS", False, "", "\u066a\u061c"),
    ("ar-QA", False, "", "\u066a\u061c"),
    ("ar-SA", False, "", "\u066a\u061c"),
    ("ar-SA", True, "", "\u066a"),
    ("ar-SD", False, "", "\u066a\u061c"),
    ("ar-SO", False, "", "\u066a\u061c"),
    ("ar-SO", True, "", "\u066a"),
    ("ar-SS", False, "", "\u066a\u061c"),
    ("ar-SY", False, "", "\u066a\u061c"),
    ("ar-TD", False, "", "\u066a\u061c"),
    ("ar-YE", False, "", "\u066a\u061c"),
    ("az-Arab", False, "", "\u066a"),
    ("bas", True, "", "\u00a0%"),
    ("be", True, "", "\u00a0%"),
    ("bgn", False, "", "\u066a"),
    ("bgn", True, "", "\u066a"),
    ("blo", True, "%\u00a0", ""),
    ("br", True, "", "\u00a0%"),
    ("bua", True, "", "\u00a0%"),
    ("ca", True, "", "\u00a0%"),
    ("ce", True, "", "\u00a0%"),
    ("ckb", False, "", "\u00a0\u066a"),
    ("cs", True, "", "\u00a0%"),
    ("cv", True, "", "\u00a0%"),
    ("da", True, "", "\u00a0%"),
    ("de", True, "", "\u00a0%"),
    ("de-CH", True, "", "%"),
    ("de-LI", True, "", "%"),
    ("es", True, "", "\u00a0%"),
    ("es-419", True, "", "%"),
    ("es-AR", True, "", "%"),
    ("es-BO", True, "", "%"),
    ("es-BR", True, "", "%"),
    ("fa", False, "", "\u066a"),
    ("fr", True, "", "\u00a0%"),
    ("ru", True, "", "\u00a0%"),
    ("tr", True, "%", ""),
]


def _percent_affixes(tag: str, latin: bool):
    """The percent affixes for a number locale, found by
    dropping trailing subtags."""
    while True:
        for locale, is_latin, before, after in _PERCENT:
            if is_latin == latin and locale.lower() == tag.lower():
                return before, after
        if "-" in tag:
            tag = tag.rsplit("-", 1)[0]
        else:
            return "", "%"


# ---------------------------------------------------------------- format

def _to_decimal(value: float, decimals: int):
    """``value`` rounded to ``decimals`` digits after the point,
    as (sign, integer digits, fraction digits); ``None`` for
    infinities and NaN. A value that rounds to zero has no
    sign."""
    if not math.isfinite(value):
        return None
    text = f"{value:.{decimals}f}"
    sign = ""
    if text.startswith("-"):
        sign = "-"
        text = text[1:]
    if "." in text:
        integer, fraction = text.split(".", 1)
    else:
        integer, fraction = text, ""
    if int(integer or "0") == 0 and not fraction.strip("0"):
        sign = ""
    return sign, integer, fraction


def _non_finite(value: float) -> str:
    if value == math.inf:
        return "∞"
    if value == -math.inf:
        return "-∞"
    return "NaN"


def _map_digits(text: str, zero: int) -> str:
    if zero == 0x30:
        return text
    return "".join(chr(zero + ord(c) - 0x30) if c.isdigit() else c
                   for c in text)


def _group_integer(integer: str, group: str, indian: bool) -> str:
    """Group an integer part, from the right: threes, or Indian
    3 then 2s (``12,34,567``)."""
    if not group or len(integer) <= 3:
        return integer
    parts: list[str] = []
    rest = integer
    first = 3
    while len(rest) > first:
        parts.append(rest[-first:])
        rest = rest[:-first]
        first = 2 if indian else 3
    parts.append(rest)
    parts.reverse()
    return group.join(parts)


class Format:
    """How numbers are written in a formatting region: digits,
    decimal and grouping separators, grouping sizes and the
    percent sign. Build it once per region and keep it."""

    def __init__(self, region: Region, digits: Digits = Digits.LOCALE
                 ) -> None:
        self.region = region
        self.digits = digits
        point, group, zero, indian = _resolve(region.numbers, digits)
        latin = zero == 0x30
        self._point = point
        self._group = group
        if latin:
            # A separator the system sets replaces the locale's,
            # where it differs.
            if region.decimal is not None and region.decimal != point:
                self._point = region.decimal
            if region.grouping is not None:
                replacement = "" if region.grouping is NO_GROUPING \
                    else region.grouping
                if replacement != group:
                    self._group = replacement
            # A grouping mark equal to the decimal mark would not
            # read back. With a decimal comma and the locale's
            # grouping, 1,234.5 becomes 1.234,5; a grouping comma
            # the system sets as well leaves no grouping.
            if self._group == self._point:
                swapped = region.grouping is None \
                    and point != self._point
                self._group = point if swapped else ""
        self._zero = zero
        self._indian = indian
        self._percent = _percent_affixes(region.numbers, latin)
        # The marks parse() reads: whether '.' is the decimal
        # mark, and the format's own marks when they are neither
        # '.' nor ','.
        if latin:
            comma = self._point == "," or self._group == "."
        elif digits == Digits.LOCALE:
            latin_point, latin_group, _, _ = _resolve(
                region.numbers, Digits.LATIN)
            comma = latin_point == "," or latin_group == "."
        else:
            comma = False
        self._marks_point = not comma
        self._marks_decimal = self._own(self._point)
        self._marks_group = self._own(self._group)

    @staticmethod
    def _own(separator: str):
        if len(separator) == 1 and separator not in (".", ","):
            return separator
        return None

    @classmethod
    def new(cls, region: Region, digits: Digits) -> Format:
        """The format for a region's number locale. With Latin
        digits, a decimal or grouping separator the region
        carries replaces the locale's."""
        return cls(region, digits)

    @classmethod
    def from_tag(cls, tag: str) -> Format:
        """The format for a locale tag, with the locale's own
        digits."""
        return cls(Region.from_tag(tag), Digits.LOCALE)

    def _write(self, sign: str, integer: str, fraction: str | None,
               grouped: bool = True) -> str:
        group = self._group if grouped else ""
        integer = _group_integer(integer, group, self._indian)
        text = _map_digits(integer, self._zero)
        if fraction is not None:
            text += self._point + _map_digits(fraction, self._zero)
        return sign + text

    def integer(self, number: int) -> str:
        """A whole number: ``1,234,567``, ``١٬٢٣٤٬٥٦٧``."""
        sign = "-" if number < 0 else ""
        return self._write(sign, str(abs(number)), None)

    def number(self, value: float, decimals: int) -> str:
        """``value`` rounded to ``decimals`` digits after the
        decimal separator; infinities and NaN are ``∞``, ``-∞``
        and ``NaN``."""
        rounded = _to_decimal(value, decimals)
        if rounded is None:
            return _non_finite(value)
        sign, integer, fraction = rounded
        return self._write(sign, integer,
                           fraction if decimals else None)

    def number_ungrouped(self, value: float, decimals: int) -> str:
        """Like :meth:`number` without grouping separators, for
        values such as pixel sizes and years."""
        rounded = _to_decimal(value, decimals)
        if rounded is None:
            return _non_finite(value)
        sign, integer, fraction = rounded
        return self._write(sign, integer,
                           fraction if decimals else None,
                           grouped=False)

    def percent(self, ratio: float, decimals: int) -> str:
        """``ratio`` as a percentage: ``0.5`` is ``50%``,
        ``50 %`` (French, German), ``%50`` (Turkish),
        ``٥٠٪؜`` (Egyptian Arabic)."""
        rounded = _to_decimal(ratio * 100.0, decimals)
        before, after = self._percent
        if rounded is None:
            return f"{before}{_non_finite(ratio)}{after}"
        sign, integer, fraction = rounded
        text = self._write("", integer,
                           fraction if decimals else None)
        return f"{sign}{before}{text}{after}"

    def parse(self, text: str) -> float:
        """The number in ``text``: what this format writes, and
        what people type or paste."""
        return _parse_number(text, self._marks_point,
                             self._marks_decimal, self._marks_group)


def parse_number(text: str, locale: str) -> float:
    """The number in ``text`` as ``locale`` writes it. A locale
    CLDR doesn't know reads as CLDR's root does (``1,234.5``)."""
    return Format.from_tag(locale).parse(text)


# ---------------------------------------------------------------- reading

#: The first code point of each run of ten Unicode decimal digits
#: (general category Nd), Unicode 16.0.
_ZEROS = [
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
]


def _digit(char: str):
    """``char``'s digit set (its zero) and value, or ``None``."""
    code = ord(char)
    import bisect
    index = bisect.bisect_right(_ZEROS, code) - 1
    if index < 0:
        return None
    zero = _ZEROS[index]
    value = code - zero
    if value > 9:
        return None
    return zero, value


def _is_bidi_mark(char: str) -> bool:
    code = ord(char)
    return code in (0x200E, 0x200F, 0x061C) \
        or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069


# Tokens: ("digit", n), ("point",), ("comma",), ("decimal",),
# ("group", kind).
def _lone_decimal(tokens, point: bool) -> bool:
    """Whether the locale's grouping mark of ``.`` and ``,`` is
    the decimal mark here because it cannot be grouping: it is
    the only mark of the two, there is one of it, and the digits
    before it are none, start with 0 or are more than three, or
    the digits after it are not three."""
    group = ("comma",) if point else ("point",)
    decimal = ("point",) if point else ("comma",)
    if decimal in tokens or ("decimal",) in tokens:
        return False
    if tokens.count(group) != 1:
        return False
    at = tokens.index(group)
    before, after = tokens[:at], tokens[at + 1:]
    return (not before or before[0] == ("digit", 0)
            or len(before) > 3 or len(after) != 3)


def _read(tokens, point: bool):
    """The ASCII number ``tokens`` spell when the decimal mark is
    ``.`` (``point``) or ``,``; ``None`` when the marks do not fit
    that reading."""
    mapped = []
    for token in tokens:
        if token == ("point",):
            mapped.append(("decimal",) if point else ("group", "."))
        elif token == ("comma",):
            mapped.append(("group", ",") if point else ("decimal",))
        else:
            mapped.append(token)
    if ("decimal",) in mapped:
        at = mapped.index(("decimal",))
        integer, fraction = mapped[:at], mapped[at + 1:]
    else:
        integer, fraction = mapped, []
    ascii_frac = "".join(str(token[1]) for token in fraction
                         if token[0] == "digit")
    if len(ascii_frac) != len(fraction):
        return None
    groups: list[str] = []
    current: list[str] = []
    for token in integer:
        if token[0] == "group":
            groups.append("".join(current))
            current = []
        elif token[0] == "digit":
            current.append(str(token[1]))
        else:
            return None
    groups.append("".join(current))
    kind = None
    for token in integer:
        if token[0] == "group":
            if kind is None:
                kind = token[1]
            elif kind != token[1]:
                return None
    if len(groups) >= 2 and kind is not None:
        first, *middle, last = groups
        width = len(middle[0]) if middle else 3
        first_max = 2 if width == 2 else 3
        fits = (1 <= len(first) <= first_max
                and not first.startswith("0")
                and len(last) == 3
                and all(len(part) == width for part in middle)
                and 2 <= width <= 3)
        if not fits:
            return None
    integer_text = "".join(groups)
    if not integer_text and not ascii_frac:
        return None
    return f"{integer_text or '0'}.{ascii_frac or '0'}"


def _parse_number(text: str, marks_point: bool, marks_decimal,
                  marks_group) -> float:
    """The number in ``text`` for one format's marks."""
    cleaned = "".join(c for c in text if not _is_bidi_mark(c))
    rest = cleaned.strip()
    negative = False
    if rest and rest[0] in ("-", "−", "－", "+", "＋"):
        negative = rest[0] in ("-", "−", "－")
        rest = rest[1:]
    tokens: list = []
    zero = None
    for char in rest:
        if marks_decimal is not None and char == marks_decimal:
            tokens.append(("decimal",))
        elif marks_group is not None and char == marks_group:
            tokens.append(("group", char))
        elif char in (".", "．"):
            tokens.append(("point",))
        elif char in (",", "，"):
            tokens.append(("comma",))
        elif char == "٫":
            tokens.append(("decimal",))
        elif char == "٬":
            tokens.append(("group", "٬"))
        elif char in ("'", "’"):
            tokens.append(("group", "'"))
        elif char in (" ", "\u00a0", "\u202f", "\u2009"):
            tokens.append(("group", " "))
        else:
            found = _digit(char)
            if found is None:
                raise ValueError(f"unexpected character {char!r}")
            found_zero, value = found
            if zero is None:
                zero = found_zero
            elif zero != found_zero:
                raise ValueError("digits from more than one script")
            tokens.append(("digit", value))
    if zero is None:
        if not tokens:
            raise ValueError("no digits")
        raise ValueError("misplaced or ambiguous decimal or grouping "
                         "mark")
    point = (not marks_point) if _lone_decimal(tokens, marks_point) \
        else marks_point
    spelled = _read(tokens, point)
    if spelled is None:
        raise ValueError("misplaced or ambiguous decimal or grouping "
                         "mark")
    value = float(spelled)
    if not math.isfinite(value):
        raise ValueError("number too large")
    return -value if negative else value


# ---------------------------------------------------------------- sizes

class FileSize:
    """How :func:`file_size` writes a byte count: the step
    between units, their names, and the digits after the decimal
    point. Start from :data:`FileSize.BINARY`,
    :data:`FileSize.IEC` or :data:`FileSize.DECIMAL` and adjust.
    """

    _UNITS = (SizeUnit.BYTE, SizeUnit.KILO, SizeUnit.MEGA, SizeUnit.GIGA,
              SizeUnit.TERA)

    def __init__(self, step: int = 1024, iec: bool = False,
                 decimals=(0, 1, 1, 1, 1), whole_from: int = 0,
                 smallest: SizeUnit = SizeUnit.BYTE,
                 largest: SizeUnit = SizeUnit.TERA,
                 byte_symbol: bool = False) -> None:
        self._step = step
        self._iec = iec
        self._decimals = tuple(decimals)
        self._whole_from = whole_from
        self._smallest = smallest
        self._largest = largest
        self._byte_symbol = byte_symbol

    def __eq__(self, other) -> bool:
        return (isinstance(other, FileSize)
                and (self._step, self._iec, self._decimals,
                     self._whole_from, self._smallest, self._largest,
                     self._byte_symbol)
                == (other._step, other._iec, other._decimals,
                    other._whole_from, other._smallest, other._largest,
                    other._byte_symbol))

    def decimals(self, n: int) -> FileSize:
        """``n`` digits after the decimal point in every unit
        above bytes."""
        return FileSize(self._step, self._iec, (0, n, n, n, n),
                        self._whole_from, self._smallest, self._largest,
                        self._byte_symbol)

    def unit_decimals(self, unit: SizeUnit, n: int) -> FileSize:
        """``n`` digits after the decimal point in ``unit``."""
        decimals = list(self._decimals)
        if unit is not SizeUnit.BYTE:
            decimals[unit.value] = n
        return FileSize(self._step, self._iec, tuple(decimals),
                        self._whole_from, self._smallest, self._largest,
                        self._byte_symbol)

    def whole_from(self, n: int) -> FileSize:
        """No decimals from ``n`` of a unit up: ``whole_from(10)``
        writes ``9.5 MB`` and ``12 MB``."""
        return FileSize(self._step, self._iec, self._decimals, n,
                        self._smallest, self._largest, self._byte_symbol)

    def smallest(self, unit: SizeUnit) -> FileSize:
        """The smallest unit shown. Sizes below it are written in
        it, rounded up, so a file that is not empty never reads
        ``0 KB``."""
        return FileSize(self._step, self._iec, self._decimals,
                        self._whole_from, unit, self._largest,
                        self._byte_symbol)

    def largest(self, unit: SizeUnit) -> FileSize:
        """The largest unit shown; larger sizes are counted in it
        (``2048 GB``)."""
        return FileSize(self._step, self._iec, self._decimals,
                        self._whole_from, self._smallest, unit,
                        self._byte_symbol)

    def byte_symbol(self) -> FileSize:
        """Bytes as the symbol ``B`` (``512 B``) instead of words
        (``512 bytes``)."""
        return FileSize(self._step, self._iec, self._decimals,
                        self._whole_from, self._smallest, self._largest,
                        True)

    def _scale(self, unit: SizeUnit) -> float:
        return float(self._step) ** unit.value

    def _parts(self, size: int):
        """The unit ``size`` is written in, the number in it and
        its digits after the point."""
        smallest = min(self._smallest, self._largest,
                       key=lambda u: u.value)
        largest = self._largest
        unit = smallest
        for candidate in self._UNITS:
            if smallest.value <= candidate.value <= largest.value \
                    and float(size) >= self._scale(candidate):
                unit = candidate
        while True:
            value = float(size) / self._scale(unit)

            def rounded(decimals: int, unit=unit,
                        value=value) -> float:
                factor = 10.0 ** decimals
                if unit == smallest and size > 0 and value < 1.0:
                    return math.ceil(value * factor) / factor
                return math.floor(value * factor + 0.5) / factor

            decimals = 0 if unit is SizeUnit.BYTE \
                else self._decimals[unit.value]
            number = rounded(decimals)
            # After rounding, so 9.96 MB, which rounds to 10.0,
            # is written 10 MB.
            if self._whole_from > 0 and number >= self._whole_from:
                decimals = 0
                number = rounded(0)
            # 1023.96 KB rounds to 1024.0 KB: the next unit writes
            # it as 1.0 MB.
            if number >= self._step and unit.value + 1 < len(self._UNITS):
                nxt = self._UNITS[unit.value + 1]
                if nxt.value <= largest.value:
                    unit = nxt
                    continue
            return unit, number, decimals

    def _template(self, unit: SizeUnit) -> str:
        if unit is SizeUnit.BYTE:
            return "{n} B"
        names = {SizeUnit.KILO: "KB", SizeUnit.MEGA: "MB",
                 SizeUnit.GIGA: "GB", SizeUnit.TERA: "TB"}
        if self._iec:
            names = {SizeUnit.KILO: "KiB", SizeUnit.MEGA: "MiB",
                     SizeUnit.GIGA: "GiB", SizeUnit.TERA: "TiB"}
        return "{n} " + names[unit]


#: Steps of 1024 named KB, MB, GB and TB, as Windows Explorer
#: shows sizes; one decimal, bytes in words.
FileSize.BINARY = FileSize(1024, False, (0, 1, 1, 1, 1), 0,
                           SizeUnit.BYTE, SizeUnit.TERA, False)
#: Steps of 1024 named KiB, MiB, GiB and TiB.
FileSize.IEC = FileSize(1024, True, (0, 1, 1, 1, 1), 0,
                        SizeUnit.BYTE, SizeUnit.TERA, False)
#: Steps of 1000 named KB, MB, GB and TB, as macOS shows sizes.
FileSize.DECIMAL = FileSize(1000, False, (0, 1, 1, 1, 1), 0,
                            SizeUnit.BYTE, SizeUnit.TERA, False)


def file_size(fmt: Format, size: int, style: FileSize) -> str:
    """``size`` bytes in the interface language, written the way
    ``style`` says (``1.5 MB``, ``512 bytes``), the number as the
    format writes it."""
    unit, number, decimals = style._parts(size)
    if unit is SizeUnit.BYTE and not style._byte_symbol:
        text = fmt.integer(int(number))
        word = "byte" if number == 1 else "bytes"
        return f"{text} {word}"
    text = fmt.number(number, decimals)
    return style._template(unit).replace("{n}", text)
