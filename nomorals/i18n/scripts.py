"""The scripts a language is written in, its direction, and which
regional forms its Han characters take — the craft-libs
``craft-i18n`` crate's ``scripts.rs``, ported.

Used for picking fonts and laying out the interface (which script
a string is in), for the runtime's right-to-left decision, and for
locale matching (a region can imply a script, a script subtag can
win over the language's usual one).  Pure tables; no ICU, no data
files — the same rows upstream ships, including the ones it calls
out as the languages a Craft app could plausibly ship.
"""

from __future__ import annotations

import unicodedata
from enum import Enum

__all__ = [
    "Direction",
    "Han",
    "Script",
    "base_direction",
    "direction",
    "han",
    "region_script",
    "scripts",
    "usual_script",
]


class Script(Enum):
    """A writing system, by its ISO 15924 code."""

    ARAB = "Arab"
    ARMN = "Armn"
    BENG = "Beng"
    CYRL = "Cyrl"
    DEVA = "Deva"
    ETHI = "Ethi"
    GEOR = "Geor"
    GREK = "Grek"
    GUJR = "Gujr"
    GURU = "Guru"
    HANG = "Hang"
    HANI = "Hani"
    HEBR = "Hebr"
    HIRA = "Hira"
    KANA = "Kana"
    KHMR = "Khmr"
    KNDA = "Knda"
    LAOO = "Laoo"
    LATN = "Latn"
    MLYM = "Mlym"
    MONG = "Mong"
    MYMR = "Mymr"
    ORYA = "Orya"
    SINH = "Sinh"
    SYRC = "Syrc"
    TAML = "Taml"
    TELU = "Telu"
    THAA = "Thaa"
    THAI = "Thai"
    TIBT = "Tibt"

    @property
    def code(self) -> str:
        """The ISO 15924 code, ``Latn``."""
        return self.value

    @property
    def rtl(self) -> bool:
        """Whether text in this script runs right to left."""
        return self in (Script.ARAB, Script.HEBR,
                        Script.SYRC, Script.THAA)


class Direction(Enum):
    """Which way a language's text, and an interface in it, runs."""

    LTR = "ltr"
    RTL = "rtl"


class Han(Enum):
    """The regional glyph forms Han characters take, as in the Noto
    CJK face names (SC, TC, HK, JP, KR).  The same code point is
    drawn differently in each, so a font stack picks its Han face
    by this, not by script."""

    SIMPLIFIED = "Simplified"
    TRADITIONAL = "Traditional"
    HONG_KONG = "HongKong"
    JAPANESE = "Japanese"
    KOREAN = "Korean"


def _is_region(subtag: str) -> bool:
    """A BCP 47 region subtag: two letters or three digits."""
    return ((len(subtag) == 2 and subtag.isalpha()
             and subtag.isascii())
            or (len(subtag) == 3 and subtag.isdigit()))


class _Subtags:
    """A language tag's language, script and region subtags."""

    def __init__(self, tag: str) -> None:
        lowered = tag.strip().lower()
        base = lowered.split(".", 1)[0].split("@", 1)[0].replace("_", "-")
        parts = [p for p in base.split("-") if p]
        self.language = parts[0] if parts else ""
        script = None
        if "@" in lowered:
            modifier = lowered.split("@", 1)[1]
            if modifier == "latin":
                script = "latn"
            elif modifier == "cyrillic":
                script = "cyrl"
        self.script = script
        self.region = None
        for part in parts[1:]:
            if (len(part) == 4 and part.isalpha() and part.isascii()
                    and self.script is None):
                self.script = part
            elif _is_region(part):
                self.region = part
                break


_SUBTAG_SCRIPTS: dict[str, tuple[Script, ...]] = {
    "hans": (Script.HANI,), "hant": (Script.HANI,),
    "hani": (Script.HANI,),
    "jpan": (Script.HANI, Script.HIRA, Script.KANA),
    "kore": (Script.HANG, Script.HANI),
    "arab": (Script.ARAB,), "armn": (Script.ARMN,),
    "beng": (Script.BENG,), "cyrl": (Script.CYRL,),
    "deva": (Script.DEVA,), "ethi": (Script.ETHI,),
    "geor": (Script.GEOR,), "grek": (Script.GREK,),
    "gujr": (Script.GUJR,), "guru": (Script.GURU,),
    "hang": (Script.HANG,), "hebr": (Script.HEBR,),
    "khmr": (Script.KHMR,), "knda": (Script.KNDA,),
    "laoo": (Script.LAOO,), "latn": (Script.LATN,),
    "mlym": (Script.MLYM,), "mong": (Script.MONG,),
    "mymr": (Script.MYMR,), "orya": (Script.ORYA,),
    "sinh": (Script.SINH,), "syrc": (Script.SYRC,),
    "taml": (Script.TAML,), "telu": (Script.TELU,),
    "thaa": (Script.THAA,), "thai": (Script.THAI,),
    "tibt": (Script.TIBT,),
}

_LANGUAGE_SCRIPTS: dict[str, tuple[Script, ...]] = {
    "ja": (Script.HANI, Script.HIRA, Script.KANA),
    "ko": (Script.HANG, Script.HANI),
    "zh": (Script.HANI,), "yue": (Script.HANI,), "wuu": (Script.HANI,),
    "be": (Script.CYRL,), "bg": (Script.CYRL,), "ba": (Script.CYRL,),
    "ce": (Script.CYRL,), "cv": (Script.CYRL,), "kk": (Script.CYRL,),
    "ky": (Script.CYRL,), "mk": (Script.CYRL,), "mn": (Script.CYRL,),
    "ru": (Script.CYRL,), "sah": (Script.CYRL,), "sr": (Script.CYRL,),
    "tg": (Script.CYRL,), "tt": (Script.CYRL,), "uk": (Script.CYRL,),
    "el": (Script.GREK,),
    "ar": (Script.ARAB,), "ckb": (Script.ARAB,), "fa": (Script.ARAB,),
    "ks": (Script.ARAB,), "ps": (Script.ARAB,), "sd": (Script.ARAB,),
    "ug": (Script.ARAB,), "ur": (Script.ARAB,),
    "he": (Script.HEBR,), "yi": (Script.HEBR,),
    "syr": (Script.SYRC,),
    "dv": (Script.THAA,),
    "ka": (Script.GEOR,),
    "hy": (Script.ARMN,),
    "hi": (Script.DEVA,), "mr": (Script.DEVA,), "ne": (Script.DEVA,),
    "sa": (Script.DEVA,), "kok": (Script.DEVA,), "mai": (Script.DEVA,),
    "bho": (Script.DEVA,),
    "bn": (Script.BENG,), "as": (Script.BENG,), "mni": (Script.BENG,),
    "pa": (Script.GURU,),
    "gu": (Script.GUJR,),
    "or": (Script.ORYA,),
    "ta": (Script.TAML,),
    "te": (Script.TELU,),
    "kn": (Script.KNDA,),
    "ml": (Script.MLYM,),
    "si": (Script.SINH,),
    "th": (Script.THAI,),
    "lo": (Script.LAOO,),
    "km": (Script.KHMR,),
    "my": (Script.MYMR,),
    "am": (Script.ETHI,), "ti": (Script.ETHI,),
    "bo": (Script.TIBT,), "dz": (Script.TIBT,),
}


def region_script(language: str, region: str | None) -> str | None:
    """The script subtag of a language in a region where CLDR's
    likely subtags give it a script other than its usual one:
    ``pa-PK`` → ``arab`` (``pa-Arab-PK``), ``zh-TW`` → ``hant``."""
    if region is None:
        return None
    language = language.lower()
    region = region.lower()
    if ((language, region) in {
            ("az", "iq"), ("az", "ir"), ("ha", "cm"), ("ha", "sd"),
            ("kk", "af"), ("kk", "cn"), ("kk", "ir"), ("kk", "mn"),
            ("ku", "iq"), ("ku", "ir"), ("ku", "lb"),
            ("ky", "cn"), ("ms", "cc"), ("pa", "pk"), ("tg", "pk"),
            ("uz", "af")}):
        return "arab"
    if ((language, region) in {
            ("az", "ru"), ("ku", "am"), ("ku", "az"), ("ku", "ge"),
            ("ku", "tm"), ("ug", "kz"), ("ug", "mn"), ("uz", "cn")}):
        return "cyrl"
    if ((language, region) in {
            ("ky", "tr"), ("sr", "me"), ("sr", "ro"), ("sr", "tr")}):
        return "latn"
    if (language, region) == ("sd", "in"):
        return "deva"
    if language == "zh" and region in {
            "au", "bn", "gb", "gf", "hk", "id", "mo", "pa", "pf",
            "ph", "sr", "th", "tw", "us", "vn"}:
        return "hant"
    if (language, region) == ("yue", "cn"):
        return "hans"
    return None


def usual_script(language: str) -> str | None:
    """The script a code without one means, for languages written
    in more than one (CLDR likely subtags).  Any other language is
    taken to have one script."""
    return {
        "be": "cyrl", "kk": "cyrl", "ky": "cyrl", "mn": "cyrl",
        "sr": "cyrl", "tg": "cyrl",
        "zh": "hans",
        "az": "latn", "bs": "latn", "ha": "latn", "ku": "latn",
        "ms": "latn", "uz": "latn",
        "ks": "arab", "sd": "arab", "ug": "arab",
        "pa": "guru",
    }.get(language.lower())


def scripts(tag: str) -> tuple[Script, ...]:
    """The scripts a language tag's interface text is written in
    (``ja`` → Hani, Hira, Kana; ``sr`` → Cyrl; ``sr-latn`` → Latn),
    the main one first.  Any case and glibc form
    (``sr_RS.UTF-8@latin``).  A script subtag wins, then a region
    that implies a script (``pa-PK`` → Arab), then the language's
    usual script.  Latin product names and units appear in every
    language, so Latn is listed only where it is the language's
    own script.  A language not in the table is taken to be
    written in Latin."""
    parsed = _Subtags(tag)
    if parsed.script is not None:
        found = _SUBTAG_SCRIPTS.get(parsed.script)
        if found is not None:
            return found
    if parsed.region is not None:
        implied = region_script(parsed.language, parsed.region)
        if implied is not None:
            found = _SUBTAG_SCRIPTS.get(implied)
            if found is not None:
                return found
    return _LANGUAGE_SCRIPTS.get(parsed.language, (Script.LATN,))


def direction(tag: str) -> Direction:
    """The direction of a language tag's text: right to left when
    its main script is."""
    first = scripts(tag)[0]
    return Direction.RTL if first.rtl else Direction.LTR


def base_direction(text: str) -> Direction | None:
    """The direction of a paragraph of text, from its first strong
    character (UAX #9 rules P2 and P3): the contents of isolates
    (LRI, RLI or FSI up to the matching PDI) are skipped, and the
    search stops at the first paragraph separator.  ``None`` when
    the paragraph has no strong character; P3 then reads it as
    left to right, and an app may use the interface direction."""
    depth = 0
    for char in text:
        if char in ("\n", "\r", "\u2029"):
            break
        if char in ("\u2066", "\u2067", "\u2068"):
            depth += 1
            continue
        if char == "\u2069":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        bidi_class = unicodedata.bidirectional(char)
        if bidi_class == "L":
            return Direction.LTR
        if bidi_class in ("R", "AL"):
            return Direction.RTL
    return None


def han(tag: str) -> Han | None:
    """The forms a language tag's Han characters take: ``zh-CN``,
    ``zh-SG``, ``zh-Hans`` and plain ``zh`` → Simplified; ``zh-TW``,
    ``zh-Hant`` and the other regions CLDR writes in Traditional
    characters (``zh-US``, ``zh-AU``) → Traditional; ``zh-HK``,
    ``zh-MO``, ``zh-Hant-HK`` and ``yue`` → HongKong; ``ja`` →
    Japanese; ``ko`` → Korean.  ``None`` for a language not written
    with Han, which leaves the choice to the font stack."""
    parsed = _Subtags(tag)
    hong_kong = parsed.region in ("hk", "mo")
    script = parsed.script if parsed.script != "hani" else None
    if script is None:
        script = region_script(parsed.language, parsed.region)
    language = parsed.language
    if language == "ja":
        return Han.JAPANESE
    if language == "ko":
        return Han.KOREAN
    if language in ("zh", "yue", "wuu") and script == "hans":
        return Han.SIMPLIFIED
    if language == "yue" and script == "hant":
        return (Han.TRADITIONAL if parsed.region == "tw"
                else Han.HONG_KONG)
    if language in ("zh", "wuu") and script == "hant":
        return Han.HONG_KONG if hong_kong else Han.TRADITIONAL
    if language == "yue":
        return Han.HONG_KONG
    if language in ("zh", "wuu"):
        return Han.SIMPLIFIED
    return None
