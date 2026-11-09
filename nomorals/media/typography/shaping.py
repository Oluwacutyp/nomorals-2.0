"""Text engineering for rendered assets: shaping, bidi-aware
measurement, grapheme-safe line breaking.

Shaping runs through Pillow's complex-text engine — when the
host Pillow carries libraqm (HarfBuzz + FriBidi), fonts are
loaded with the RAQM layout engine and Arabic joining, Indic
conjuncts and bidirectional runs are shaped by the real stack.
When raqm is absent the module says so (``shaping_engine()``)
and measurement falls back to the basic engine; it never claims
shaping it did not perform.

Line breaking is grapheme-first: text is segmented into
user-perceived characters (a UAX #29 subset covering combining
marks, spacing marks, Hangul jamo sequences, emoji ZWJ
sequences, regional-indicator pairs, keycaps and skin tones) and
a break is only ever placed between clusters — a virama, a ZWJ
sequence or a flag can never be split across lines.
"""

from __future__ import annotations

import unicodedata
from typing import Any

__all__ = [
    "graphemes", "shaping_engine", "load_font", "shaped_width",
    "unshaped_width", "wrap_text", "dominant_direction",
]

_PREPEND = frozenset(
    list(range(0x0600, 0x0606)) + [0x06DD, 0x070F, 0x0890, 0x0891,
                                   0x08E2, 0x110BD, 0x110CD,
                                   0x111C0, 0x111C2, 0x111C3,
                                   0x111C4, 0x111C5, 0x111D0, 0x111D1,
                                   0x111D2, 0x111D3, 0x111D4,
                                   0x111D5, 0x11660, 0x11661, 0x11662,
                                   0x11663, 0x11664, 0x11665, 0x11666,
                                   0x11667, 0x11668, 0x11669, 0x1166A,
                                   0x1166B, 0x1166C, 0x10A02, 0x10A03,
                                   0x10A05, 0x10A06, 0x10A0C, 0x10A0D,
                                   0x10A0F])


def _is_extend(ch: str) -> bool:
    cp = ord(ch)
    if 0x1F3FB <= cp <= 0x1F3FF:      # emoji skin-tone modifiers
        return True
    if 0xE0020 <= cp <= 0xE007F:      # tag characters
        return True
    return unicodedata.category(ch) in ("Mn", "Me")


def _is_spacing_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mc"


def _is_control(ch: str) -> bool:
    if ch in ("\r", "\n", "\u200d", "\u200c"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp")


def _jamo(ch: str) -> str:
    cp = ord(ch)
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11FF or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7FB <= cp <= 0xD7FF:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LV" if (cp - 0xAC00) % 28 == 0 else "LVT"
    return ""


def _is_ri(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _is_indic_consonant(ch: str) -> bool:
    cp = ord(ch)
    if unicodedata.category(ch) != "Lo":
        return False
    return (0x0900 <= cp <= 0x0DFF or 0x1000 <= cp <= 0x109F
            or 0x1780 <= cp <= 0x17FF or 0x1880 <= cp <= 0x18AA)


def _is_virama(ch: str) -> bool:
    name = unicodedata.name(ch, "")
    return "VIRAMA" in name or "HALANT" in name or name.endswith(" PULLI")


def _is_pictographic(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF
            or 0x2300 <= cp <= 0x23FF
            or cp in (0x203C, 0x2049, 0x2122, 0x2139, 0x3030,
                      0x303D, 0x3297, 0x3299))


def graphemes(text: str) -> list[str]:
    """Segment ``text`` into grapheme clusters (UAX #29 subset)."""
    if not text:
        return []
    clusters: list[str] = []
    current = [text[0]]
    ri_run = 1 if _is_ri(text[0]) else 0
    for i in range(1, len(text)):
        cur = text[i]
        if not _break_before(text, i, ri_run):
            current.append(cur)
            ri_run = ri_run + 1 if _is_ri(cur) else 0
            continue
        clusters.append("".join(current))
        current = [cur]
        ri_run = 1 if _is_ri(cur) else 0
    clusters.append("".join(current))
    return clusters


def _break_before(text: str, i: int, ri_run: int) -> bool:
    prev, cur = text[i - 1], text[i]
    if prev == "\r" and cur == "\n":
        return False
    if prev in "\r\n" or cur in "\r\n":
        return True
    if _is_control(prev) or _is_control(cur):
        return True
    pj, cj = _jamo(prev), _jamo(cur)
    if pj and cj:
        if pj == "L" and cj in ("L", "V", "LV", "LVT"):
            return False
        if pj in ("LV", "V") and cj in ("V", "T"):
            return False
        if pj in ("LVT", "T") and cj == "T":
            return False
    if _is_extend(cur) or cur in ("\u200d", "\u200c"):
        return False                      # GB9 (+ ZWJ glue)
    if prev == "\u200d":
        # ZWJ glues forward, pictographic or not — Indic uses it
        # inside clusters (क्‍य) as surely as emoji do.
        return False
    if _is_spacing_mark(cur):
        return False                      # GB9a
    if ord(prev) in _PREPEND:
        return False                      # GB9b
    if _is_indic_consonant(cur):
        # GB9c: Consonant [Extend|ZWJ]* Virama [Extend|ZWJ]* x
        # Consonant — a virama-linked conjunct (क्ष) is one
        # cluster. Walk back over the whole mark run (the virama
        # is itself a mark, so it sits inside the run) and
        # require: a virama in the run, an Indic consonant base.
        j = i - 1
        saw_virama = False
        while j >= 0 and (_is_extend(text[j]) or text[j] == "\u200d"):
            saw_virama = saw_virama or _is_virama(text[j])
            j -= 1
        if saw_virama and j >= 0 and _is_indic_consonant(text[j]):
            return False
    if _is_pictographic(cur):
        # GB11: extPict Extend* ZWJ x extPict
        j = i - 1
        while j >= 0 and _is_extend(text[j]):
            j -= 1
        if j >= 0 and text[j] == "\u200d":
            k = j - 1
            while k >= 0 and _is_extend(text[k]):
                k -= 1
            if k >= 0 and _is_pictographic(text[k]):
                return False
    # GB12/13: regional indicators break only in pairs (flags)
    return not (_is_ri(prev) and _is_ri(cur) and ri_run % 2 == 1)


# ── Pillow shaping / measurement ──────────────────────────────────

def shaping_engine() -> str:
    """'raqm' when the host Pillow shapes complex text, else
    'basic'. Honest capability reporting, used in cache identity."""
    try:
        from PIL import features
        return "raqm" if features.check("raqm") else "basic"
    except Exception:  # noqa: BLE001 - no Pillow at all
        return "none"


def load_font(path: str, size: int, face_index: int = 0):
    """Load a face the way the service renders: RAQM engine when
    available so shaping and bidi actually happen."""
    from PIL import ImageFont
    engine = None
    try:
        from PIL import features
        if features.check("raqm"):
            engine = ImageFont.Layout.RAQM
    except Exception:  # noqa: BLE001 - fall back to default engine
        engine = None
    if engine is not None:
        return ImageFont.truetype(path, size, index=face_index,
                                  layout_engine=engine)
    return ImageFont.truetype(path, size, index=face_index)


_DRAW: Any = None


def _draw():
    global _DRAW
    if _DRAW is None:
        from PIL import Image, ImageDraw
        _DRAW = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    return _DRAW


def shaped_width(text: str, font: Any) -> float:
    """Advance width of ``text`` under the font's own layout
    engine (raqm-shaped when the font was loaded by load_font)."""
    if not text:
        return 0.0
    return float(_draw().textlength(text, font=font))


def unshaped_width(text: str, face: Any, size: int) -> float:
    """The same text under the BASIC engine — the control
    measurement shaping tests compare against."""
    from PIL import ImageFont
    font = ImageFont.truetype(face.path, size, index=face.face_index,
                              layout_engine=ImageFont.Layout.BASIC)
    return shaped_width(text, font)


def dominant_direction(text: str) -> str | None:
    """'rtl'/'ltr' from the first strong character, else None."""
    for ch in text:
        cp = ord(ch)
        if (0x590 <= cp <= 0x8FF or 0xFB1D <= cp <= 0xFDFF
                or 0xFE70 <= cp <= 0xFEFF):
            return "rtl"
        if ch.isalpha():
            return "ltr"
    return None


def wrap_text(text: str, font: Any, max_width: float) -> list[str]:
    """Greedy word wrap that only ever breaks between grapheme
    clusters (and at spaces). A word longer than the line is split
    at cluster boundaries, never inside one."""
    out: list[str] = []
    for paragraph in str(text).split("\n"):
        if not paragraph:
            out.append("")
            continue
        lines: list[str] = []
        current = ""
        for word in paragraph.split(" "):
            trial = word if not current else current + " " + word
            if shaped_width(trial, font) <= max_width or not current:
                if shaped_width(word, font) > max_width and not current:
                    chunks = _split_word(word, font, max_width)
                    lines.extend(chunks[:-1])
                    current = chunks[-1] if chunks else ""
                else:
                    current = trial
                continue
            lines.append(current)
            if shaped_width(word, font) > max_width:
                chunks = _split_word(word, font, max_width)
                lines.extend(chunks[:-1])
                current = chunks[-1] if chunks else ""
            else:
                current = word
        lines.append(current)
        out.extend(lines)
    return out


def _split_word(word: str, font: Any, max_width: float) -> list[str]:
    chunks: list[str] = []
    current = ""
    for cluster in graphemes(word):
        trial = current + cluster
        if shaped_width(trial, font) <= max_width or not current:
            current = trial
        else:
            chunks.append(current)
            current = cluster
    if current:
        chunks.append(current)
    return chunks
