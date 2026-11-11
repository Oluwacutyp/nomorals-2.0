"""The fancy-text style engine — every Unicode text style she
knows, as DATA over a generated canonical core (FontLibrary F3,
2026-11-11; census ~/workspace/font-census/CENSUS.md §4 + §7-U).

Before this module she had 18 styles hand-built inside
``emoji_art`` (12 character maps + 6 decorations) plus a SECOND,
duplicate small-caps table in ``style.py``.  The two public
generators (Namecheap's fancy generator, Picsart's) expose an
89-name union over a small canonical core — 13 Mathematical
Alphanumeric alphabets, the enclosed sets, fullwidth, super/
subscript, combining overlays, reversal transforms and
cross-script lookalike tables.  This module ships that core:

* **Generated, never hand-typed, where Unicode has a block.**
  The math alphabets and enclosed sets are built from their
  block bases at import, and a candidate codepoint is only used
  when ``unicodedata`` says it is assigned — so the tables are
  provably complete against the Unicode data itself, reserved
  holes (the math-italic ``h`` slot U+1D455, the fraktur
  capitals living in Letterlike Symbols) included, via the
  recorded exception glyphs.
* **Hand tables only where Unicode has no block** — small caps,
  super/subscript, and the lookalike families (Cyrillic, Greek,
  Cherokee, Gothic, kana/bopomofo, Arabic-script, Armenian/Thai,
  Lisu, stroke/acute/umlaut sets, upside-down, mirrored, emoji
  substitution).  Each is finite (a–z/A–Z/0–9), ships as data,
  and is marked ``lookalike`` in the catalogue with its fallback
  characters recorded — coverage honestly differs per letter.
* **Overlays compose.**  Combining-mark sets (strike, slash,
  underline family, stacked glitch marks…) are layers that stack
  onto ANY base map, the way both generators structure them —
  not 18 disconnected named styles.
* **Wrappers are data** — affix / interleave / per-character
  decorations in one table.
* **The named catalogue is one data table** (:data:`CATALOGUE`):
  each of the 89 union names resolves to a composition (base map
  + overlays + optional wrapper, or a banner variant).  Adding a
  style is a data row, never new code.

Text-art banner styles are FIGlet-class glyph composition — a
different transform kind: the composer and her own block/dots/
slant glyph fonts live here too.  Seven Picsart banner brand
names (Modern, Retro Vibes, Ogre, Small Letters, Stick Letters,
Sub-Zero, 3D Text Art) are NOT claimed: their glyph data is not
derivable from Unicode and was never substantiated — the
composer is hers to extend as data when real glyph tables exist.

Presentation-only rule (inherited from ``style.small_caps``):
stylized text is presentation, never applied to anything a
reader must type back — a stylized command is an untypable
command.  Surfaces decide where styling is allowed; this module
only knows how.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

__all__ = [
    "StyleSpec",
    "CATALOGUE",
    "STYLE_NAMES",
    "style_names",
    "resolve_style",
    "suggest_styles",
    "apply_style",
    "apply_spec",
    "base_map",
    "BASE_MAPS",
    "OVERLAYS",
    "WRAPPERS",
    "SMALLCAPS_GLYPHS",
    "small_caps_glyph",
    "compose_banner",
]


# ── the generated canonical core ─────────────────────────────

def _assigned(cp: int) -> bool:
    return bool(unicodedata.name(chr(cp), ""))


def _math_alphabet(upper: int, lower: int, digits: int,
                   exceptions: dict[str, str] | None = None
                   ) -> dict[str, str]:
    """One Mathematical Alphanumeric alphabet from its block
    bases.  A letter whose slot is a reserved hole in the block
    is skipped (it passes through unchanged) unless an exception
    glyph — the Letterlike Symbols residents Unicode assigned
    instead — is recorded for it."""
    table = dict(exceptions or {})
    for i in range(26):
        for src, base in (("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i], upper),
                          ("abcdefghijklmnopqrstuvwxyz"[i], lower)):
            if src not in table and _assigned(base + i):
                table[src] = chr(base + i)
    if digits:
        for i in range(10):
            table[str(i)] = chr(digits + i)
    return table


def _enclosed(upper: int, lower: int | None,
              digits: dict[str, str] | None = None
              ) -> dict[str, str]:
    """One enclosed-alphanumeric set.  ``lower=None`` folds
    lowercase onto the capital glyphs (the negative sets and the
    squared set have no lowercase forms in Unicode)."""
    table: dict[str, str] = {}
    for i in range(26):
        table[chr(0x41 + i)] = chr(upper + i)
        table[chr(0x61 + i)] = chr((lower if lower is not None
                                    else upper) + i)
    table.update(digits or {})
    return table


def _circled_digits(zero_cp: int, one_cp: int) -> dict[str, str]:
    out = {"0": chr(zero_cp)}
    for i in range(1, 10):
        out[str(i)] = chr(one_cp + i - 1)
    return out


#: The 13 Mathematical Alphanumeric alphabets + enclosed sets +
#: fullwidth + regional indicators — all generated from their
#: Unicode blocks, exception glyphs recorded inline.
_GENERATED_MAPS: dict[str, dict[str, str]] = {
    "serif_bold": _math_alphabet(0x1D400, 0x1D41A, 0x1D7CE),
    "serif_italic": _math_alphabet(0x1D434, 0x1D44E, 0,
                                   {"h": "ℎ"}),
    "serif_bold_italic": _math_alphabet(0x1D468, 0x1D482, 0),
    "script": _math_alphabet(0x1D49C, 0x1D4B6, 0, {
        "B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ",
        "L": "ℒ", "M": "ℳ", "R": "ℛ", "e": "ℯ", "g": "ℊ",
        "o": "ℴ"}),
    "script_bold": _math_alphabet(0x1D4D0, 0x1D4EA, 0),
    "fraktur": _math_alphabet(0x1D504, 0x1D51E, 0, {
        "C": "ℭ", "H": "ℌ", "I": "ℑ", "R": "ℜ", "Z": "ℨ"}),
    "fraktur_bold": _math_alphabet(0x1D56C, 0x1D586, 0),
    "double_struck": _math_alphabet(0x1D538, 0x1D552, 0x1D7D8, {
        "C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ",
        "R": "ℝ", "Z": "ℤ"}),
    "sans": _math_alphabet(0x1D5A0, 0x1D5BA, 0x1D7E2),
    "sans_bold": _math_alphabet(0x1D5D4, 0x1D5EE, 0x1D7EC),
    "sans_italic": _math_alphabet(0x1D608, 0x1D622, 0),
    "sans_bold_italic": _math_alphabet(0x1D63C, 0x1D656, 0),
    "monospace": _math_alphabet(0x1D670, 0x1D68A, 0x1D7F6),
    "circled": _enclosed(0x24B6, 0x24D0,
                         _circled_digits(0x24EA, 0x2460)),
    "negative_circled": _enclosed(0x1F150, None,
                                  _circled_digits(0x24FF, 0x2776)),
    "squared": _enclosed(0x1F130, None),
    "negative_squared": _enclosed(0x1F170, None),
    "parenthesized": _enclosed(
        0x1F110, 0x249C,
        {str(i): chr(0x2474 + i - 1) for i in range(1, 10)}),
    "regional": _enclosed(0x1F1E6, 0x1F1E6),
    "fullwidth": {**{chr(c): chr(c + 0xFEE0)
                     for c in range(0x21, 0x7F)},
                  " ": "　"},
    "gothic": {**{chr(0x41 + i): chr(0x10330 + i)
                  for i in range(26)},
               **{chr(0x61 + i): chr(0x10330 + i)
                  for i in range(26)}},
}


def _lisu() -> dict[str, str]:
    """The Lisu (Fraser) lookalike alphabet, generated from the
    Unicode names themselves: each Lisu letter is named for its
    sound (LISU LETTER BA, PA, DA…), so the table is a sound map
    read straight out of the Unicode data."""
    by_sound: dict[str, str] = {}
    for cp in range(0xA4D0, 0xA4F0):
        name = unicodedata.name(chr(cp), "")
        if name.startswith("LISU LETTER "):
            by_sound[name[len("LISU LETTER "):]] = chr(cp)
    sounds = {"a": "A", "b": "BA", "c": "CA", "d": "DA",
              "e": "E", "f": "FA", "g": "GA", "h": "HA",
              "i": "I", "j": "JA", "k": "KA", "l": "LA",
              "m": "MA", "n": "NA", "o": "O", "p": "PA",
              "s": "SA", "t": "TA", "u": "U", "w": "WA",
              "x": "XA", "y": "YA", "z": "ZA"}
    table: dict[str, str] = {}
    for letter, sound in sounds.items():
        if sound in by_sound:
            table[letter] = by_sound[sound]
            table[letter.upper()] = by_sound[sound]
    return table


# ── hand tables: where Unicode has no block ──────────────────
# Cross-script and partial sets are written as codepoint escapes
# wherever the glyph could be mistyped for its Latin lookalike —
# a Cyrillic А typed as a Latin A would silently no-op the style.

#: Unicode small capitals (no block of their own — a lookalike
#: set).  ``s`` and ``x`` have no small-capital codepoint in the
#: set the generators use and fall back to themselves; ``q``
#: takes ǫ.  This ONE table now backs both ``emoji_art``'s
#: smallcaps style and ``style.small_caps`` — the duplicate is
#: gone.
SMALLCAPS_GLYPHS = "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ"


def small_caps_glyph(ch: str) -> str:
    """The small-capital glyph for one lowercase a–z letter, or
    the character itself when the set has no glyph for it."""
    if len(ch) == 1 and "a" <= ch <= "z":
        return SMALLCAPS_GLYPHS[ord(ch) - 0x61]
    return ch


_SMALLCAPS_LOWER = {chr(0x61 + i): SMALLCAPS_GLYPHS[i]
                    for i in range(26)}
_SMALLCAPS_FULL = {**_SMALLCAPS_LOWER,
                   **{chr(0x41 + i): SMALLCAPS_GLYPHS[i]
                      for i in range(26)}}

_SUPERSCRIPT = {
    "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ",
    "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ",
    "M": "ᴹ", "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "R": "ᴿ",
    "T": "ᵀ", "U": "ᵁ", "V": "ⱽ", "W": "ᵂ",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
    "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ",
    "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
    "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
    "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    **{str(i): g for i, g in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
}

_SUBSCRIPT = {
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
    **{str(i): g for i, g in enumerate("₀₁₂₃₄₅₆₇₈₉")},
}

_CYRILLIC = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "G": "Г",
    "H": "Н", "I": "І", "J": "Ј", "K": "К", "M": "М",
    "N": "И", "O": "О", "P": "Р", "R": "Я", "S": "Ѕ",
    "T": "Т", "W": "Ш", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "h": "һ", "i": "і",
    "j": "ј", "m": "м", "n": "п", "o": "о", "p": "р",
    "r": "я", "s": "ѕ", "t": "т", "x": "х", "y": "у",
}

_GREEK = {
    "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "I": "Ι",
    "K": "Κ", "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ",
    "T": "Τ", "X": "Χ", "Y": "Υ", "Z": "Ζ",
    "a": "α", "e": "ε", "i": "ι", "k": "κ", "m": "μ",
    "n": "ν", "o": "ο", "p": "ρ", "s": "σ", "t": "τ",
    "u": "υ", "w": "ω", "x": "χ", "y": "γ", "z": "ζ",
}

#: The Greek-led "Delta" variant: the delta letters themselves
#: ride on the Greek lookalikes.
_DELTA = {**_GREEK, "A": "Δ", "a": "δ", "D": "Δ", "d": "δ"}

_CHEROKEE = {
    "a": "Ꮧ", "b": "Ꮟ", "c": "Ꮯ", "d": "Ꮄ", "e": "Ꮛ",
    "f": "Ꭶ", "g": "Ꮆ", "h": "Ꮒ", "i": "Ꭵ", "j": "Ꮰ",
    "k": "Ꮶ", "l": "Ꮭ", "m": "Ꮇ", "n": "Ꮑ", "o": "Ꭷ",
    "p": "Ꭾ", "q": "Ꭴ", "r": "Ꮢ", "s": "Ꮥ", "t": "Ꮦ",
    "u": "Ꮼ", "v": "Ꮙ", "w": "Ꮿ", "y": "Ꭹ", "z": "Ꮓ",
}
_CHEROKEE.update({k.upper(): v for k, v in _CHEROKEE.items()})

_JAPANESE = {
    "a": "卂", "b": "乃", "c": "匚", "d": "刀", "e": "乇",
    "f": "下", "g": "厶", "h": "卄", "i": "工", "j": "丁",
    "k": "长", "l": "乚", "m": "从", "n": "几", "o": "口",
    "p": "尸", "r": "尺", "s": "丂", "t": "丄", "u": "凵",
    "v": "リ", "w": "山", "x": "乂", "y": "丫", "z": "乙",
}
_JAPANESE.update({k.upper(): v for k, v in _JAPANESE.items()})

_ARABIC = {
    "a": "ا", "b": "ب", "c": "چ", "d": "د", "e": "ع",
    "f": "ف", "g": "گ", "h": "ح", "i": "ي", "j": "ج",
    "k": "ك", "l": "ل", "m": "م", "n": "ن", "o": "و",
    "p": "پ", "q": "ق", "r": "ر", "s": "س", "t": "ت",
    "u": "ۇ", "v": "ڤ", "y": "ى", "z": "ز",
}
_ARABIC.update({k.upper(): v for k, v in _ARABIC.items()})

_WIZARD = {  # Armenian sound-map lookalikes
    "a": "ա", "b": "բ", "d": "դ", "e": "ե", "f": "ֆ",
    "g": "գ", "h": "հ", "i": "ի", "j": "ջ", "k": "կ",
    "l": "լ", "m": "մ", "n": "ն", "o": "օ", "p": "պ",
    "r": "ր", "s": "ս", "t": "տ", "u": "ւ", "v": "վ",
    "w": "ո", "x": "խ", "y": "յ", "z": "զ",
}
_WIZARD.update({k.upper(): v for k, v in _WIZARD.items()})

_LEFTHANDED = {  # Thai-led lookalikes, Armenian patches
    "a": "ค", "b": "บ", "d": "ด", "e": "Ե", "f": "ฟ",
    "h": "ห", "j": "จ", "k": "ก", "l": "ล", "m": "ม",
    "n": "น", "o": "อ", "p": "ป", "r": "ร", "s": "ส",
    "t": "ท", "u": "ս", "v": "ว", "w": "พ", "y": "ย",
    "z": "ซ",
}
_LEFTHANDED.update({k.upper(): v for k, v in _LEFTHANDED.items()})

_MANUSCRIPT = {  # Cyrillic-led manuscript capitals
    "A": "А", "B": "В", "C": "С", "D": "Д", "E": "Е",
    "F": "Ф", "G": "Г", "H": "Н", "I": "І", "J": "Ј",
    "K": "К", "L": "Л", "M": "М", "N": "И", "O": "О",
    "P": "Р", "R": "Я", "S": "Ѕ", "T": "Т", "U": "Υ",
    "V": "Ѵ", "W": "Ш", "X": "Х", "Y": "У", "Z": "З",
}
_MANUSCRIPT.update({k.lower(): v for k, v in _MANUSCRIPT.items()})

_STROKED = {
    "a": "Ⱥ", "b": "ƀ", "c": "ȼ", "d": "đ", "e": "ɇ",
    "h": "ħ", "i": "ɨ", "j": "ɉ", "l": "ł", "o": "ø",
    "r": "ɍ", "t": "ŧ", "u": "ʉ", "y": "ɏ", "z": "ƶ",
    "A": "Ⱥ", "D": "Đ", "H": "Ħ", "L": "Ł", "O": "Ø",
    "T": "Ŧ", "Y": "Ɏ", "Z": "Ƶ",
}

_ACUTE = {
    "a": "á", "c": "ć", "e": "é", "g": "ǵ", "i": "í",
    "k": "ḱ", "l": "ĺ", "m": "ḿ", "n": "ń", "o": "ó",
    "p": "ṕ", "r": "ŕ", "s": "ś", "u": "ú", "w": "ẃ",
    "y": "ý", "z": "ź",
    "A": "Á", "C": "Ć", "E": "É", "G": "Ǵ", "I": "Í",
    "K": "Ḱ", "L": "Ĺ", "M": "Ḿ", "N": "Ń", "O": "Ó",
    "P": "Ṕ", "R": "Ŕ", "S": "Ś", "U": "Ú", "W": "Ẃ",
    "Y": "Ý", "Z": "Ź",
}

_UMLAUT = {
    "a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü",
    "x": "ẍ", "y": "ÿ",
    "A": "Ä", "E": "Ë", "I": "Ï", "O": "Ö", "U": "Ü",
    "Y": "Ÿ",
}

_EMOJI_ALPHABET = {
    "a": "🍎", "b": "🍌", "c": "🍒", "d": "🍩", "e": "🥚",
    "f": "🍟", "g": "🍇", "h": "🍯", "i": "🍦", "j": "🧃",
    "k": "🥝", "l": "🍋", "m": "🍉", "n": "🥜", "o": "🍊",
    "p": "🍑", "q": "👑", "r": "🌹", "s": "🍓", "t": "🍅",
    "u": "🦄", "v": "🎻", "w": "🌊", "x": "🎄", "y": "🪀",
    "z": "🦓",
}
_EMOJI_ALPHABET.update({k.upper(): v
                        for k, v in _EMOJI_ALPHABET.items()})

_INVERTED = {  # upside-down: rotation lookalikes + reversal
    "a": "ɐ", "b": "q", "c": "ɔ", "d": "p", "e": "ǝ",
    "f": "ɟ", "g": "ƃ", "h": "ɥ", "i": "ı", "j": "ɾ",
    "k": "ʞ", "l": "ʅ", "m": "ɯ", "n": "u", "o": "o",
    "p": "d", "q": "b", "r": "ɹ", "s": "s", "t": "ʇ",
    "u": "n", "v": "ʌ", "w": "ʍ", "x": "x", "y": "ʎ",
    "z": "z",
    "A": "∀", "C": "Ɔ", "E": "Ǝ", "F": "Ⅎ", "G": "⅁",
    "H": "H", "I": "I", "J": "ſ", "L": "˥", "M": "Ɯ",
    "N": "N", "O": "O", "P": "Ԁ", "R": "ᴚ", "T": "⊥",
    "U": "∩", "V": "Λ", "W": "M", "X": "X", "Y": "⅄",
    "Z": "Z",
    "0": "0", "1": "Ɩ", "6": "9", "9": "6",
    "?": "¿", "!": "¡",
}

_MIRRORED = {  # horizontal-flip lookalikes + reversal
    "a": "ɒ", "b": "d", "c": "ɔ", "d": "b", "e": "ɘ",
    "f": "ꟻ", "i": "i", "o": "o", "p": "q", "q": "p",
    "r": "ɿ", "u": "u", "x": "x", "y": "ʏ",
}

_BLACKOUT = {chr(c): "█" for c in range(0x21, 0x7F)}

#: Every base map the engine can apply, by key.  Catalogue rows
#: name these; ``cycle:a,b`` and the ``upanddown`` policy are
#: composition shorthands resolved in :func:`_apply_base`.
BASE_MAPS: dict[str, dict[str, str]] = {
    **_GENERATED_MAPS,
    "lisu": _lisu(),
    "smallcaps_lower": _SMALLCAPS_LOWER,
    "smallcaps": _SMALLCAPS_FULL,
    "superscript": _SUPERSCRIPT,
    "subscript": _SUBSCRIPT,
    "cyrillic": _CYRILLIC,
    "greek": _GREEK,
    "delta": _DELTA,
    "cherokee": _CHEROKEE,
    "japanese": _JAPANESE,
    "arabic": _ARABIC,
    "wizard": _WIZARD,
    "lefthanded": _LEFTHANDED,
    "manuscript": _MANUSCRIPT,
    "stroked": _STROKED,
    "acute": _ACUTE,
    "umlaut": _UMLAUT,
    "emoji_alphabet": _EMOJI_ALPHABET,
    "inverted": _INVERTED,
    "mirrored": _MIRRORED,
    "blackout": _BLACKOUT,
}


# ── combining overlays (compose with ANY base map) ──────────

#: Overlay name → the combining-mark sequence laid over each
#: character. Namecheap ships its strikethrough (7) and
#: underline (11) generators as separate tools; here they are
#: one composable family, exactly as the census prescribed.
OVERLAYS: dict[str, str] = {
"strike": "̶", # long stroke overlay
"slash": "̷", # short solidus overlay
"long_slash": "̸", # long solidus overlay
"x_cross": "ͯ", # combining latin small letter x
"underline": "̲", # combining low line
"double_underline": "̳", # combining double low line
"dash_underline": "̱", # combining macron below
"squiggle_underline": "̰", # combining tilde below
"overline": "̅", # combining overline
"double_overline": "̿", # combining double overline
"macron": "̄", # combining macron
"ring": "⃝", # combining enclosing circle
"square_frame": "⃞", # combining enclosing square
"diamond_frame": "⃟", # combining enclosing diamond
"bandaid": "͓͝", # stacked breve above + x below
}

#: The glitch overlay's deterministic mark cycle — position in
#: the text picks the stack, so a glitch render is stable and
#: pinnable (no randomness in a style).
_GLITCH_STACKS = ("̈", "́", "̵", "̀", "̶",
"͗", "͏", "̴")


# ── wrappers / decorations (data) ────────────────────────────
# kind: "affix" (pre/post around the whole text) ·
# "interleave" (a separator between characters, optionally at
# both ends too) · "per_char" (a template around every char).

WRAPPERS: dict[str, tuple[str, str, str]] = {
# the legacy six — byte-identical to emoji_art's originals
"sparkle": ("affix", "✨ ", " ✨"),
"stars": ("affix", "⋆★ ", " ★⋆"),
"hearts": ("affix", "♡ ", " ♡"),
"frame": ("affix", "【 ", " 】"),
"quotes": ("affix", "『 ", " 』"),
"waves": ("affix", "˗ˏˋ ", " ˎˊ˗"),
# the Picsart symbol family + kin
"sparkles": ("affix", "✧˖° ", " °˖✧"),
"brackets": ("affix", "[ ", "]"),
"parenthesis": ("affix", "( ", ")"),
"air_quotes": ("affix", "❝ ", " ❞"),
"arrows": ("affix", "➸ ", " ➸"),
"music": ("affix", "♪ ", " ♫"),
"princess": ("affix", "👑 ", " 👑"),
"cupido": ("affix", "💘 ", " 💘"),
"volume": ("affix", "🔊 ", " 🔊"),
"finish": ("affix", "🏁 ", " 🏁"),
"bubbly": ("affix", "🫧 ", " 🫧"),
"sweets": ("affix", "🍭 ", " 🍬"),
"veggies": ("affix", "🥦 ", " 🥕"),
"food": ("affix", "🍔 ", " 🍟"),
"drinks": ("affix", "🥤 ", " 🧋"),
"love": ("affix", "❤️ ", " 💕"),
"fading": ("affix", "", " ▒░·"),
"hearts_interleave": ("interleave", "♥", "ends"),
"notes_interleave": ("interleave", "♪", ""),
"arrow_flow": ("interleave", "→", ""),
"barcode": ("per_char", "|", "|"),
}


# ── text-art banners (FIGlet-class glyph composition) ───────
# Her own glyph font: 5-row block letters, drawn here.  Glyphs
# are encoded with '#' for a filled cell and '.' for an empty
# one, so the data never depends on runs of spaces surviving;
# the composer translates to real block characters, folds case,
# normalizes ragged rows, and renders the same grids through
# the block / rectangles / dots fills, a shadow pass, or an
# italic shear ("slant").

_BANNER_SRC: dict[str, tuple[str, ...]] = {
    "A": (".##.", "#..#", "####", "#..#", "#..#"),
    "B": ("###.", "#..#", "###.", "#..#", "###."),
    "C": (".###", "#...", "#...", "#...", ".###"),
    "D": ("###.", "#..#", "#..#", "#..#", "###."),
    "E": ("####", "#...", "###.", "#...", "####"),
    "F": ("####", "#...", "###.", "#...", "#..."),
    "G": (".###", "#...", "#.##", "#..#", ".###"),
    "H": ("#..#", "#..#", "####", "#..#", "#..#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": (".###", "...#", "...#", "#..#", ".##."),
    "K": ("#..#", "#.#.", "##..", "#.#.", "#..#"),
    "L": ("#...", "#...", "#...", "#...", "####"),
    "M": ("#..#", "####", "#..#", "#..#", "#..#"),
    "N": ("#..#", "##.#", "#.##", "#..#", "#..#"),
    "O": (".##.", "#..#", "#..#", "#..#", ".##."),
    "P": ("###.", "#..#", "###.", "#...", "#..."),
    "Q": (".##.", "#..#", "#..#", "#.#.", ".##."),
    "R": ("###.", "#..#", "###.", "#.#.", "#..#"),
    "S": (".###", "#...", ".##.", "...#", "###."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#..#", "#..#", "#..#", "#..#", ".##."),
    "V": ("#..#", "#..#", "#..#", "#.#.", ".#.."),
    "W": ("#..#", "#..#", "#..#", "####", "#..#"),
    "X": ("#..#", ".##.", "..#..", ".##.", "#..#"),
    "Y": ("#..#", "#..#", ".##.", ".#..", ".#.."),
    "Z": ("####", "..#.", ".#..", "#...", "####"),
    "0": (".##.", "#..#", "#..#", "#..#", ".##."),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###.", "...#", ".###", "#...", "####"),
    "3": ("###.", "...#", ".##.", "...#", "###."),
    "4": ("#..#", "#..#", "####", "...#", "...#"),
    "5": ("####", "#...", "###.", "...#", "###."),
    "6": (".##.", "#...", "###.", "#..#", ".##."),
    "7": ("####", "...#", "..#.", ".#..", ".#.."),
    "8": (".##.", "#..#", ".##.", "#..#", ".##."),
    "9": (".##.", "#..#", ".###", "...#", ".##."),
    " ": ("...", "...", "...", "...", "..."),
    "!": ("#", "#", "#", ".", "#"),
    "?": ("###.", "...#", ".##.", "....", ".#.."),
    ".": (".", ".", ".", ".", "#"),
    "-": ("....", "....", "####", "....", "...."),
    "'": ("#", "#", ".", ".", "."),
}

_BANNER_GLYPHS: dict[str, tuple[str, ...]] = {
    ch: tuple(row.replace("#", "█").replace(".", " ")
              for row in rows)
    for ch, rows in _BANNER_SRC.items()
}

_BANNER_FILLS = {"block": "█", "rectangles": "▮", "dots": "●"}


def compose_banner(text: str, variant: str = "block") -> str:
    """Compose text as a multi-line banner in her glyph font.
    Variants: ``block`` · ``rectangles`` · ``dots`` · ``shadow``
    (block + a ▒ shadow pass) · ``slant`` (block, italic shear).
    Characters with no glyph are skipped; case folds up."""
    variant = str(variant or "block").lower()
    fill = _BANNER_FILLS.get(variant, "█")
    glyphs = []
    for ch in str(text or "").upper():
        glyph = _BANNER_GLYPHS.get(ch)
        if glyph is None:
            continue
        width = max(len(row) for row in glyph)
        glyphs.append([row.ljust(width) for row in glyph])
    if not glyphs:
        return ""
    rows = [" ".join(g[r] for g in glyphs) for r in range(5)]
    if fill != "█":
        rows = [row.replace("█", fill) for row in rows]
    if variant == "shadow":
        grid = [list(row) for row in rows]
        for r in range(5):
            for c, cell in enumerate(list(rows[r])):
                if cell == "█" and r + 1 < 5:
                    if c + 1 >= len(grid[r + 1]):
                        grid[r + 1].extend(
                            " " * (c + 2 - len(grid[r + 1])))
                    if grid[r + 1][c + 1] == " ":
                        grid[r + 1][c + 1] = "▒"
        rows = ["".join(row).rstrip() for row in grid]
    elif variant == "slant":
        rows = [" " * (4 - r) + row for r, row in enumerate(rows)]
    return "\n".join(row.rstrip() for row in rows)


# ── the named catalogue — one data table ─────────────────────

@dataclass(frozen=True)
class StyleSpec:
    """One named style = a composition.  ``kind`` is ``map``
    (base map ± overlays ± wrapper), ``overlay`` (identity base
    + overlays), ``wrapper`` (identity base + a wrapper),
    ``mixed`` (a cycling base), or ``banner`` (glyph composition
    in the named variant).  ``lookalike`` marks styles whose
    glyphs come from lookalike tables rather than true Unicode
    alphabets; ``fallbacks`` records the characters that pass
    through unchanged for partial tables."""

    name: str
    slug: str
    kind: str = "map"
    base: str = ""
    overlays: tuple[str, ...] = ()
    wrapper: str = ""
    reverse: bool = False
    lookalike: bool = False
    fallbacks: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    note: str = ""


def _fallbacks(table: dict[str, str]) -> tuple[str, ...]:
    """The a–z/A–Z/0–9 characters a partial table does NOT
    carry — computed, never hand-listed, so the catalogue's
    honesty about coverage can't drift from the table."""
    alpha = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz0123456789")
    return tuple(ch for ch in alpha if ch not in table)


def _row(name, slug, kind="map", base="", overlays=(), wrapper="",
         reverse=False, lookalike=False, table=None, aliases=(),
         note=""):
    return StyleSpec(
        name=name, slug=slug, kind=kind, base=base,
        overlays=tuple(overlays), wrapper=wrapper,
        reverse=reverse, lookalike=lookalike,
        fallbacks=_fallbacks(table) if table is not None else (),
        aliases=tuple(aliases), note=note)


#: The catalogue.  Display names are hers; the generators' brand
#: names ride as aliases (the outputs are Unicode text — nothing
#: is copied from either site, §5 of the census).  Order is the
#: browse order.
CATALOGUE: tuple[StyleSpec, ...] = (
    # — the 13 Mathematical Alphanumeric alphabets + kin —
    _row("Bold", "bold", base="serif_bold",
         aliases=("serif bold", "math bold")),
    _row("Italic", "italic", base="serif_italic",
         aliases=("math italic",)),
    _row("Bold Italic", "bolditalic", base="serif_bold_italic",
         aliases=("serif bold italic", "math bold italic")),
    _row("Script", "script", base="script",
         aliases=("cursive", "handwriting")),
    _row("Script Bold", "script_bold", base="script_bold",
         aliases=("bold script",)),
    _row("Script Bold Italic", "script_bold_italic",
         base="script_bold",
         note="Unicode has no script bold-italic alphabet; the "
              "generators render script bold under this name."),
    _row("Fraktur", "fraktur", base="fraktur",
         aliases=("gothic", "blackletter")),
    _row("Fraktur Bold", "fraktur_bold", base="fraktur_bold",
         aliases=("gothic bold", "blackletter bold")),
    _row("Double Struck", "doublestruck", base="double_struck",
         aliases=("double struck", "blackboard bold", "outlined")),
    _row("Monospace", "monospace", base="monospace",
         aliases=("monospace text", "typewriter")),
    _row("Sans", "sans", base="sans",
         aliases=("sans serif", "math sans")),
    _row("Sans Bold", "sansbold", base="sans_bold",
         aliases=("sans serif bold", "math sans bold")),
    _row("Sans Italic", "sans_italic", base="sans_italic",
         aliases=("sans serif italic", "math sans italic")),
    _row("Sans Bold Italic", "sans_bold_italic",
         base="sans_bold_italic",
         aliases=("sans serif bold italic", "math sans bold italic")),
    _row("Math Style", "math_style", base="serif_italic",
         note="Picsart's Small-page name for the math italic set."),
    # — enclosed sets + fullwidth + regional —
    _row("Circled", "circled", base="circled",
         aliases=("light text bubbles", "bubbles")),
    _row("Negative Circled", "negative_circled",
         base="negative_circled",
         aliases=("dark text bubbles",)),
    _row("Squared", "squared", base="squared",
         aliases=("light text squares",)),
    _row("Negative Squared", "negative_squared",
         base="negative_squared",
         aliases=("dark text squares",)),
    _row("Parenthesized", "parenthesized", base="parenthesized"),
    _row("Fullwidth", "fullwidth", base="fullwidth",
         aliases=("full width",)),
    _row("Regional", "regional", base="regional",
         aliases=("flags", "regional indicators"),
         note="Letters as regional indicators — pairs render "
              "as flags on most clients."),
    # — partial Unicode sets (lookalike class) —
    _row("Small Caps", "smallcaps", base="smallcaps",
         lookalike=True, table=BASE_MAPS["smallcaps"],
         aliases=("mono upper",),
         note="s/x have no small-capital glyph and stay plain."),
    _row("Small Caps (lowercase only)", "smallcaps_lower",
         base="smallcaps_lower", lookalike=True,
         table=BASE_MAPS["smallcaps_lower"],
         note="The legacy form: capitals pass through untouched."),
    _row("Superscript", "superscript", base="superscript",
         lookalike=True, table=BASE_MAPS["superscript"],
         note="Partial by Unicode design — unmapped letters "
              "pass through."),
    _row("Subscript", "subscript", base="subscript",
         lookalike=True, table=BASE_MAPS["subscript"],
         note="Partial by Unicode design — unmapped letters "
              "pass through."),
    # — cross-script lookalikes —
    _row("Cyrillic", "cyrillic", base="cyrillic", lookalike=True,
         table=BASE_MAPS["cyrillic"]),
    _row("Greek", "greek", base="greek", lookalike=True,
         table=BASE_MAPS["greek"], aliases=("modern greek",)),
    _row("Delta", "delta", base="delta", lookalike=True,
         table=BASE_MAPS["delta"]),
    _row("Cherokee", "cherokee", base="cherokee", lookalike=True,
         table=BASE_MAPS["cherokee"], aliases=("fairytale",)),
    _row("Gothic Runes", "gothic_runes", base="gothic",
         lookalike=True, table=BASE_MAPS["gothic"],
         aliases=("cryptic italic",),
         note="The Gothic block alphabet, letters in block "
              "order — a lookalike set, not a transliteration."),
    _row("Japanese", "japanese", base="japanese", lookalike=True,
         table=BASE_MAPS["japanese"]),
    _row("Arabic", "arabic", base="arabic", lookalike=True,
         table=BASE_MAPS["arabic"]),
    _row("Wizard", "wizard", base="wizard", lookalike=True,
         table=BASE_MAPS["wizard"],
         note="Armenian sound-map lookalikes."),
    _row("Lefthanded", "lefthanded", base="lefthanded",
         lookalike=True, table=BASE_MAPS["lefthanded"],
         note="Thai-led lookalikes with Armenian patches."),
    _row("Ancient Lisu", "lisu", base="lisu", lookalike=True,
         table=BASE_MAPS["lisu"], aliases=("ancient style 1",),
         note="Lisu (Fraser) letters, mapped by the sound "
              "Unicode names them for."),
    _row("Manuscript", "manuscript", base="manuscript",
         lookalike=True, table=BASE_MAPS["manuscript"]),
    _row("Stroked", "stroked", base="stroked", lookalike=True,
         table=BASE_MAPS["stroked"]),
    _row("Acute Accents", "acute", base="acute", lookalike=True,
         table=BASE_MAPS["acute"]),
    _row("Metal Umlauts", "umlaut", base="umlaut", lookalike=True,
         table=BASE_MAPS["umlaut"]),
    _row("Emoji Alphabet", "emoji_alphabet", base="emoji_alphabet",
         lookalike=True, aliases=("emojis text",),
         note="Per-letter emoji substitution — a code, not a "
              "font; digits and punctuation pass through."),
    # — reversals —
    _row("Inverted", "inverted", base="inverted", reverse=True,
         lookalike=True, table=BASE_MAPS["inverted"],
         aliases=("upside down", "aussie")),
    _row("Mirrored", "mirrored", base="mirrored", reverse=True,
         lookalike=True, table=BASE_MAPS["mirrored"],
         aliases=("flipped",)),
    _row("Reversed", "reversed", reverse=True,
         note="Plain string reversal, no glyph change."),
    # — mixed / cycling policies —
    _row("Mixed Style 1", "mixed1", kind="mixed",
         base="cycle:script,sans_bold"),
    _row("Mixed Style 2", "mixed2", kind="mixed",
         base="cycle:fraktur_bold,double_struck"),
    _row("Mixed Style 3", "mixed3", kind="mixed",
         base="cycle:sans_italic,monospace,script_bold"),
    _row("Funky", "funky", kind="mixed",
         base="cycle:circled,squared,negative_squared"),
    _row("Up and Down", "upanddown", kind="mixed", base="upanddown",
         lookalike=True,
         note="Alternating small-cap and capital letters."),
    # — overlays (identity base; they compose onto any map too) —
    _row("Strikethrough", "strikethrough", kind="overlay",
         overlays=("strike",)),
    _row("Slashed", "slashed", kind="overlay",
         overlays=("slash",)),
    _row("Diagonal Strikes", "diagonal_strikes", kind="overlay",
         overlays=("slash",)),
    _row("Long Slash", "long_slash", kind="overlay",
         overlays=("long_slash",)),
    _row("X Cross", "x_cross", kind="overlay",
         overlays=("x_cross",)),
    _row("Bandaid", "bandaid", kind="overlay",
         overlays=("bandaid",)),
    _row("Glitch", "glitch", kind="overlay",
         overlays=("glitch",)),
    _row("Underlined", "underlined", kind="overlay",
         overlays=("underline",)),
    _row("Double Underlined", "double_underlined", kind="overlay",
         overlays=("double_underline",)),
    _row("Dash Underlined", "dash_underlined", kind="overlay",
         overlays=("dash_underline",),
         aliases=("underlined by dashes",)),
    _row("Squiggle Underlined", "squiggle_underlined",
         kind="overlay", overlays=("squiggle_underline",)),
    _row("Overlined", "overlined", kind="overlay",
         overlays=("overline",)),
    _row("Double Overlined", "double_overlined", kind="overlay",
         overlays=("double_overline",)),
    _row("Ringed", "ringed", kind="overlay", overlays=("ring",)),
    _row("Blackout", "blackout", base="blackout",
         note="Every character redacted to a block — the "
              "generators' block-out strike variation."),
    # — wrappers / decorations —
    _row("Sparkle", "sparkle", kind="wrapper", wrapper="sparkle"),
    _row("Stars", "stars", kind="wrapper", wrapper="stars"),
    _row("Hearts Frame", "hearts_frame", kind="wrapper",
         wrapper="hearts",
         note="The legacy hearts decoration (affix form)."),
    _row("Frame", "frame", kind="wrapper", wrapper="frame"),
    _row("Quotes Frame", "quotes_frame", kind="wrapper",
         wrapper="quotes"),
    _row("Waves", "waves", kind="wrapper", wrapper="waves"),
    _row("Sparkles", "sparkles", kind="wrapper", wrapper="sparkles"),
    _row("Brackets", "brackets", kind="wrapper", wrapper="brackets"),
    _row("Parenthesis", "parenthesis", kind="wrapper",
         wrapper="parenthesis"),
    _row("Air Quotes", "air_quotes", kind="wrapper",
         wrapper="air_quotes"),
    _row("Arrows", "arrows", kind="wrapper", wrapper="arrows"),
    _row("Arrow Flow", "arrow_flow", kind="wrapper",
         wrapper="arrow_flow"),
    _row("Music Notes", "music_notes", kind="wrapper",
         wrapper="music"),
    _row("Notes Between", "notes_between", kind="wrapper",
         wrapper="notes_interleave"),
    _row("Princess", "princess", kind="wrapper", wrapper="princess",
         aliases=("princess style",)),
    _row("Cupido", "cupido", kind="wrapper", wrapper="cupido"),
    _row("Volume", "volume", kind="wrapper", wrapper="volume"),
    _row("Finish", "finish", kind="wrapper", wrapper="finish"),
    _row("Bubbly Bubbles", "bubbly", kind="wrapper",
         wrapper="bubbly"),
    _row("Sweets", "sweets", kind="wrapper", wrapper="sweets",
         aliases=("sweets emojis",)),
    _row("Veggies", "veggies", kind="wrapper", wrapper="veggies",
         aliases=("veggies emojis",)),
    _row("Food", "food", kind="wrapper", wrapper="food",
         aliases=("food emojis",)),
    _row("Drinks", "drinks", kind="wrapper", wrapper="drinks",
         aliases=("drinks emojis",)),
    _row("Love", "love", kind="wrapper", wrapper="love",
         aliases=("love emojis",)),
    _row("Fading", "fading", kind="wrapper", wrapper="fading",
         aliases=("fading effect",)),
    _row("Hearts Between", "hearts", kind="wrapper",
         wrapper="hearts_interleave",
         note="Picsart's Hearts: a ♥ between every character."),
    _row("Barcode", "barcode", kind="wrapper", wrapper="barcode"),
    # — banners (glyph composition; a different transform kind) —
    _row("Block Letters", "block_letters", kind="banner",
         base="block", aliases=("standard text art",)),
    _row("Rectangles", "rectangles", kind="banner",
         base="rectangles"),
    _row("Lines and Dots", "lines_and_dots", kind="banner",
         base="dots"),
    _row("Shadows", "shadows", kind="banner", base="shadow"),
    _row("Slant", "slant", kind="banner", base="slant",
         aliases=("italic letters",)),
    _row("Squared Letters", "squared_letters", base="squared",
         note="Picsart files this under Text Art; its mechanism "
              "is the squared enclosed map, so it resolves "
              "there."),
)

STYLE_NAMES: tuple[str, ...] = tuple(s.name for s in CATALOGUE)


def style_names() -> list[str]:
    """Every catalogue display name, in browse order."""
    return [spec.name for spec in CATALOGUE]


# ── resolution + application ─────────────────────────────────

def _key(name: str) -> str:
    out = []
    for ch in str(name or "").lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_")


_INDEX: dict[str, StyleSpec] = {}
for _spec in CATALOGUE:
    for _name in (_spec.slug, _spec.name, *_spec.aliases):
        _INDEX.setdefault(_key(_name), _spec)


def resolve_style(name: str) -> StyleSpec | None:
    """The catalogue row for a style name (slug, display name or
    alias, case/punctuation-insensitive), or None."""
    return _INDEX.get(_key(name))


def suggest_styles(name: str, limit: int = 4) -> list[str]:
    """The nearest real style names for a miss — a failed lookup
    answers with these, never with silent plain text."""
    import difflib

    probe = _key(name)
    hits = difflib.get_close_matches(
        probe, list(_INDEX), n=limit, cutoff=0.45)
    seen: list[str] = []
    for hit in hits:
        display = _INDEX[hit].name
        if display not in seen:
            seen.append(display)
    if not seen:
        for key, spec in _INDEX.items():
            if probe and (probe in key or key in probe):
                if spec.name not in seen:
                    seen.append(spec.name)
                if len(seen) >= limit:
                    break
    return seen


def base_map(key: str) -> dict[str, str] | None:
    """One base map by key, or None."""
    return BASE_MAPS.get(key)


def _apply_base(text: str, base: str) -> str:
    if not base:
        return text
    if base.startswith("cycle:"):
        maps = [BASE_MAPS[k] for k in base[6:].split(",")
                if k in BASE_MAPS]
        if not maps:
            return text
        out = []
        idx = 0
        for ch in text:
            if ch.isalpha():
                out.append(maps[idx % len(maps)].get(ch, ch))
                idx += 1
            else:
                out.append(ch)
        return "".join(out)
    if base == "upanddown":
        out = []
        idx = 0
        for ch in text:
            if ch.isalpha():
                if idx % 2 == 0:
                    out.append(small_caps_glyph(ch.lower()))
                else:
                    out.append(ch.upper())
                idx += 1
            else:
                out.append(ch)
        return "".join(out)
    table = BASE_MAPS.get(base)
    if table is None:
        return text
    return "".join(table.get(ch, ch) for ch in text)


def _apply_overlay(text: str, overlay: str) -> str:
    if overlay == "glitch":
        out = []
        idx = 0
        for ch in text:
            if ch == " ":
                out.append(ch)
                continue
            stack = _GLITCH_STACKS[idx % len(_GLITCH_STACKS)]
            if idx % 3 == 2:
                stack += _GLITCH_STACKS[
                    (idx * 3 + 1) % len(_GLITCH_STACKS)]
            out.append(ch + stack)
            idx += 1
        return "".join(out)
    marks = OVERLAYS.get(overlay)
    if not marks:
        return text
    return "".join(ch if ch == " " else ch + marks
                   for ch in text)


def _apply_wrapper(text: str, wrapper: str) -> str:
    entry = WRAPPERS.get(wrapper)
    if entry is None:
        return text
    kind, main, extra = entry
    if kind == "affix":
        return f"{main}{text}{extra}"
    if kind == "interleave":
        joined = main.join(text)
        return f"{main}{joined}{main}" if extra == "ends" else joined
    if kind == "per_char":
        return "".join(f"{main}{ch}" for ch in text) + extra
    return text


def apply_spec(text: str, spec: StyleSpec) -> str:
    """Apply one catalogue row (or any hand-built composition)
    to text: base map → reversal → overlays → wrapper."""
    text = str(text or "")
    if spec.kind == "banner":
        return compose_banner(text, spec.base or "block")
    out = _apply_base(text, spec.base)
    if spec.reverse:
        out = out[::-1]
    for overlay in spec.overlays:
        out = _apply_overlay(out, overlay)
    if spec.wrapper:
        out = _apply_wrapper(out, spec.wrapper)
    return out


def apply_style(text: str, style: str) -> str | None:
    """Restyle text in a named catalogue style.  Characters a
    style doesn't carry pass through unchanged — the text is
    never eaten.  An unknown style returns None (the caller
    answers with :func:`suggest_styles`, never plain text)."""
    spec = resolve_style(style)
    if spec is None:
        return None
    return apply_spec(text, spec)
