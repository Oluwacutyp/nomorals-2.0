"""Pure-Python sfnt parsing — the catalogue's evidence layer.

Reads the tables that carry design facts out of real font files:
``name`` (family, subfamily, full/PostScript names, copyright and
licence strings), ``OS/2`` (weight class, width class, italic/bold
selection bits, IBM family class), ``head`` (macStyle, units per
em), ``post`` (fixed-pitch, italic angle), ``cmap`` (the actual
codepoint ranges the face can render — formats 12/4/6/0), ``fvar``
(variable axes, when the face is variable) and the colour-table
tags (COLR/CBDT/sbix/SVG) that mark an emoji-capable face.

Formats: plain TTF/OTF, TrueType Collections (every face indexed),
and WOFF (zlib-reassembled). WOFF2 needs a brotli transform this
parser does not fake: when fontTools + brotli are importable the
file is read through them, otherwise it is reported unparsed by
the caller — never guessed at.

Script coverage is derived from the cmap, not from the filename:
for each script a *core set* (its primary letters and digits, the
codepoints ordinary text cannot do without) is intersected with
the cmap ranges and the covered fraction recorded. A fraction is
a heuristic over that core set — honest evidence a face can set
text in the script, never a full-repertoire claim.

Nothing here raises on a malformed file: unparseable input yields
an empty list and the catalogue counts the failure.
"""

from __future__ import annotations

import hashlib
import struct
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FontAxis", "FontFace", "SCRIPTS", "COVER_THRESHOLD",
    "parse_font_file", "script_fraction",
]


#: A face "covers" a script when at least this fraction of the
#: script's core set is present in its cmap.
COVER_THRESHOLD = 0.55


def _ranges(*pairs: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(pairs)


#: Core codepoint sets per script — primary letters + digits.
#: Han and Hangul are sample sets (their repertoires are open /
#: combinatorial); the fraction is over the sample.
_HAN_SAMPLE = (
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之"
    "过家对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如"
    "前所本见经头面公同三已老从动两长知回么点水电把被给重正机更九新"
)
_HANGUL_SAMPLE = (
    "가나다라마바사아자차카타파하한글국어사람생각하늘바다산들꽃비눈봄여름가을겨울오늘내일어제"
    "지금여기저기우리저희것들수있없같다해요입니다합니다그리고그러나그래서하지만때문"
)

SCRIPTS: dict[str, tuple[tuple[int, int], ...]] = {
    "Latin": _ranges((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0xFF)),
    "Greek": _ranges((0x391, 0x3A1), (0x3A3, 0x3AB), (0x3B1, 0x3C9)),
    "Cyrillic": _ranges((0x401, 0x401), (0x410, 0x42F),
                        (0x430, 0x44F), (0x451, 0x451)),
    "Hebrew": _ranges((0x5D0, 0x5EA)),
    "Arabic": _ranges((0x620, 0x64A), (0x660, 0x669)),
    "Devanagari": _ranges((0x904, 0x914), (0x915, 0x939),
                          (0x93E, 0x94D), (0x966, 0x96F)),
    "Bengali": _ranges((0x985, 0x98C), (0x98F, 0x990), (0x993, 0x9A8),
                       (0x9AA, 0x9B0), (0x9B6, 0x9B9), (0x9E6, 0x9FE)),
    "Gurmukhi": _ranges((0xA01, 0xA02), (0xA05, 0xA0A), (0xA0F, 0xA10),
                        (0xA13, 0xA28), (0xA2A, 0xA30), (0xA66, 0xA75)),
    "Gujarati": _ranges((0xA85, 0xA8D), (0xA8F, 0xA91), (0xA93, 0xAA8),
                        (0xAAA, 0xAB0), (0xAB6, 0xAB9), (0xAE6, 0xAEF)),
    "Oriya": _ranges((0xB05, 0xB0C), (0xB0F, 0xB10), (0xB13, 0xB28),
                     (0xB2A, 0xB30), (0xB66, 0xB6F)),
    "Tamil": _ranges((0xB85, 0xB8A), (0xB8E, 0xB90), (0xB92, 0xB95),
                     (0xB99, 0xB9A), (0xB9C, 0xB9C), (0xB9E, 0xB9F),
                     (0xBE6, 0xBEF)),
    "Telugu": _ranges((0xC05, 0xC0C), (0xC0E, 0xC10), (0xC12, 0xC28),
                      (0xC2A, 0xC33), (0xC66, 0xC6F)),
    "Kannada": _ranges((0xC85, 0xC8C), (0xC8E, 0xC90), (0xC92, 0xCA8),
                       (0xCAA, 0xCB3), (0xCE6, 0xCEF)),
    "Malayalam": _ranges((0xD05, 0xD0C), (0xD0E, 0xD10), (0xD12, 0xD3A),
                         (0xD66, 0xD6F)),
    "Thai": _ranges((0xE01, 0xE30), (0xE32, 0xE33), (0xE38, 0xE3A),
                    (0xE47, 0xE4E), (0xE50, 0xE59)),
    "Lao": _ranges((0xE81, 0xE82), (0xE84, 0xE84), (0xE87, 0xE88),
                   (0xE8A, 0xE8A), (0xE8D, 0xE8D), (0xE94, 0xE97),
                   (0xE99, 0xE9F), (0xEA1, 0xEA3), (0xEA5, 0xEA5),
                   (0xEA7, 0xEA7), (0xEAA, 0xEAB), (0xEAD, 0xEB0),
                   (0xEB2, 0xEB3), (0xEBD, 0xEBD), (0xEC0, 0xEC4),
                   (0xEC6, 0xEC6), (0xED0, 0xED9)),
    "Myanmar": _ranges((0x1000, 0x1021), (0x1022, 0x1027),
                       (0x1029, 0x102A), (0x1040, 0x1049)),
    "Georgian": _ranges((0x10D0, 0x10FA)),
    "Armenian": _ranges((0x531, 0x556), (0x561, 0x586)),
    "Ethiopic": _ranges((0x1200, 0x1248), (0x124A, 0x124D),
                        (0x1250, 0x1256), (0x1258, 0x125A),
                        (0x125D, 0x125D), (0x1260, 0x1288)),
    "Khmer": _ranges((0x1780, 0x17A2), (0x17D0, 0x17D3),
                     (0x17E0, 0x17E9)),
    "Mongolian": _ranges((0x1820, 0x182A), (0x182C, 0x1842),
                         (0x1844, 0x1848), (0x1850, 0x1852),
                         (0x1857, 0x1857), (0x1880, 0x1884)),
    "Tibetan": _ranges((0xF40, 0xF6C), (0xF90, 0xFBC)),
    "Sinhala": _ranges((0xD81, 0xD81), (0xD85, 0xD8A), (0xD91, 0xD96),
                       (0xD9A, 0xDA5), (0xDA7, 0xDB1)),
    "Hiragana": _ranges((0x3041, 0x3096)),
    "Katakana": _ranges((0x30A1, 0x30FA)),
}

_SAMPLE_SCRIPTS: dict[str, str] = {
    "Han": _HAN_SAMPLE, "Hangul": _HANGUL_SAMPLE,
}


@dataclass(frozen=True)
class FontAxis:
    tag: str
    minimum: float
    default: float
    maximum: float
    name: str = ""


@dataclass
class FontFace:
    """One face (one index inside one font file) as design data."""

    family: str
    subfamily: str = "Regular"
    full_name: str = ""
    postscript_name: str = ""
    weight: int = 400
    width_class: int = 5
    italic: bool = False
    fixed_pitch: bool = False
    family_class: int = 0        # OS/2 sFamilyClass high byte (IBM)
    family_subclass: int = 0
    axes: tuple[FontAxis, ...] = ()
    scripts: dict[str, float] = field(default_factory=dict)
    coverage_known: bool = False
    ranges: tuple[tuple[int, int], ...] = ()
    color: bool = False
    units_per_em: int = 1000
    path: str = ""
    face_index: int = 0
    file_id: str = ""
    licence: str = ""
    licence_url: str = ""
    copyright: str = ""

    def covers(self, script: str) -> bool:
        return self.scripts.get(script, 0.0) >= COVER_THRESHOLD

    def covers_codepoint(self, cp: int) -> bool:
        ranges = self.ranges
        lo, hi = 0, len(ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = ranges[mid]
            if cp < start:
                hi = mid - 1
            elif cp > end:
                lo = mid + 1
            else:
                return True
        return False

    def covers_text(self, text: str) -> bool:
        """True when every significant character has a glyph.

        Whitespace, format controls and ASCII punctuation are
        neutral (they survive via fallback); letters, digits and
        marks must be in the cmap itself.
        """
        if not self.coverage_known:
            return False
        for ch in text:
            if ch.isspace() or unicodedata.category(ch) == "Cf":
                continue
            cp = ord(ch)
            if cp < 0x80 and not ch.isalnum():
                continue
            if not self.covers_codepoint(cp):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "subfamily": self.subfamily,
            "full_name": self.full_name,
            "postscript_name": self.postscript_name,
            "weight": self.weight, "width_class": self.width_class,
            "italic": self.italic, "fixed_pitch": self.fixed_pitch,
            "family_class": self.family_class,
            "family_subclass": self.family_subclass,
            "axes": [vars(a) for a in self.axes],
            "scripts": dict(self.scripts),
            "coverage_known": self.coverage_known,
            "ranges": [list(r) for r in self.ranges],
            "color": self.color, "units_per_em": self.units_per_em,
            "path": self.path, "face_index": self.face_index,
            "file_id": self.file_id, "licence": self.licence,
            "licence_url": self.licence_url, "copyright": self.copyright,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FontFace:
        data = dict(data)
        data["axes"] = tuple(FontAxis(**a) for a in data.get("axes", ()))
        data["ranges"] = tuple(tuple(r) for r in data.get("ranges", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# ── coverage ──────────────────────────────────────────────────────

def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def _range_cover(ranges: tuple[tuple[int, int], ...], lo: int, hi: int) -> int:
    """How many codepoints of [lo, hi] the merged ranges cover."""
    total = 0
    for start, end in ranges:
        if end < lo:
            continue
        if start > hi:
            break
        total += min(end, hi) - max(start, lo) + 1
    return total


def script_fraction(ranges: tuple[tuple[int, int], ...],
                    script: str) -> float:
    if script in _SAMPLE_SCRIPTS:
        sample = _SAMPLE_SCRIPTS[script]
        hits = sum(1 for ch in sample if _range_cover(ranges, ord(ch),
                                                      ord(ch)))
        return hits / len(sample) if sample else 0.0
    core = SCRIPTS.get(script)
    if not core:
        return 0.0
    total = sum(hi - lo + 1 for lo, hi in core)
    covered = sum(_range_cover(ranges, lo, hi) for lo, hi in core)
    return covered / total if total else 0.0


def _all_scripts(ranges) -> dict[str, float]:
    out: dict[str, float] = {}
    for script in list(SCRIPTS) + list(_SAMPLE_SCRIPTS):
        frac = script_fraction(ranges, script)
        if frac > 0.0:
            out[script] = round(frac, 4)
    return out


# ── table extraction ──────────────────────────────────────────────

def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from(">H", data, off)[0]


def _i16(data: bytes, off: int) -> int:
    return struct.unpack_from(">h", data, off)[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def _sfnt_tables(data: bytes, face_offset: int) -> dict[str, bytes]:
    num = _u16(data, face_offset + 4)
    tables: dict[str, bytes] = {}
    for i in range(num):
        rec = face_offset + 12 + i * 16
        if rec + 16 > len(data):
            break
        tag = data[rec:rec + 4].decode("latin-1")
        off, length = _u32(data, rec + 8), _u32(data, rec + 12)
        if off + length <= len(data):
            tables[tag] = data[off:off + length]
    return tables


def _woff_tables(data: bytes) -> dict[str, bytes]:
    num = _u16(data, 12)
    tables: dict[str, bytes] = {}
    for i in range(num):
        rec = 44 + i * 20
        if rec + 20 > len(data):
            break
        tag = data[rec:rec + 4].decode("latin-1")
        off, comp_len, orig_len = (_u32(data, rec + 4), _u32(data, rec + 8),
                                   _u32(data, rec + 12))
        raw = data[off:off + comp_len]
        if comp_len < orig_len:
            raw = zlib.decompress(raw)
        if len(raw) == orig_len:
            tables[tag] = raw
    return tables


def _woff2_faces(path: Path, file_id: str) -> list[FontFace]:
    """WOFF2 via fontTools, when the host provides it + brotli."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return []
    faces: list[FontFace] = []
    try:
        font = TTFont(str(path))
    except Exception:  # noqa: BLE001 - unparseable here, counted by caller
        return []
    try:
        tables: dict[str, bytes] = {}
        for tag in font:
            if tag == "GlyphOrder":
                continue
            try:
                tables[tag] = font.getTableData(tag)
            except Exception:  # noqa: BLE001 - skip exotic tables
                continue
        face = _face_from_tables(tables, str(path), 0, file_id)
        if face is not None:
            faces.append(face)
    finally:
        font.close()
    return faces


# ── table parsing ─────────────────────────────────────────────────

def _decode_name(raw: bytes, platform: int) -> str:
    if platform in (0, 3):
        return raw.decode("utf-16-be", "replace")
    return raw.decode("mac_roman", "replace")


def _parse_names(table: bytes) -> dict[int, str]:
    if len(table) < 6:
        return {}
    count, storage = _u16(table, 2), _u16(table, 4)
    best: dict[int, tuple[int, str]] = {}
    for i in range(count):
        rec = 6 + i * 12
        if rec + 12 > len(table):
            break
        platform, _enc, lang, name_id, length, offset = struct.unpack_from(
            ">HHHHHH", table, rec)
        raw = table[storage + offset: storage + offset + length]
        if len(raw) < length:
            continue
        # preference: Windows en-US, then any Unicode, then Mac
        rank = {(3, 0x409): 0, (0, 0): 1, (3, 0): 2}.get(
            (platform, lang), 3 if platform in (0, 3) else 4)
        text = _decode_name(raw, platform).strip("\x00").strip()
        if not text:
            continue
        if name_id not in best or rank < best[name_id][0]:
            best[name_id] = (rank, text)
    return {nid: text for nid, (_rank, text) in best.items()}


def _cmap_ranges(table: bytes) -> tuple[tuple[int, int], ...]:
    if len(table) < 4:
        return ()
    num = _u16(table, 2)
    subs: list[tuple[int, int, int]] = []  # (format, pref, offset)
    for i in range(num):
        rec = 4 + i * 8
        if rec + 8 > len(table):
            break
        platform, encoding, offset = struct.unpack_from(">HHI", table, rec)
        if offset + 2 > len(table):
            continue
        fmt = _u16(table, offset)
        pref = {12: 0, 4: 1, 6: 2, 0: 3}.get(fmt, 9)
        if platform in (0, 3) or pref <= 1:
            subs.append((fmt, pref, offset))
    if not subs:
        return ()
    subs.sort(key=lambda s: s[1])
    fmt, _pref, offset = subs[0]
    sub = table[offset:]
    ranges: list[tuple[int, int]] = []
    if fmt == 12 and len(sub) >= 16:
        groups = _u32(sub, 12)
        for i in range(groups):
            rec = 16 + i * 12
            if rec + 12 > len(sub):
                break
            start, end = _u32(sub, rec), _u32(sub, rec + 4)
            ranges.append((start, end))
    elif fmt == 4 and len(sub) >= 14:
        seg_x2 = _u16(sub, 6)
        segs = seg_x2 // 2
        end_off = 14
        start_off = end_off + seg_x2 + 2
        delta_off = start_off + seg_x2
        ro_off = delta_off + seg_x2
        if ro_off + seg_x2 <= len(sub):
            for i in range(segs):
                end = _u16(sub, end_off + i * 2)
                start = _u16(sub, start_off + i * 2)
                delta = _i16(sub, delta_off + i * 2)
                ro = _u16(sub, ro_off + i * 2)
                if start == 0xFFFF and end == 0xFFFF:
                    continue
                if ro == 0:
                    if (start + delta) % 65536 != 0 or \
                            (end + delta) % 65536 != 0:
                        ranges.append((start, end))
                    else:  # sparse .notdef-only segment is not coverage
                        continue
                else:
                    run_start = None
                    for cp in range(start, end + 1):
                        gi_pos = (ro_off + i * 2 + ro
                                  + (cp - start) * 2)
                        glyph = _u16(sub, gi_pos) if gi_pos + 2 <= len(sub) \
                            else 0
                        mapped = glyph != 0 and \
                            (glyph + delta) % 65536 != 0
                        if mapped and run_start is None:
                            run_start = cp
                        if not mapped and run_start is not None:
                            ranges.append((run_start, cp - 1))
                            run_start = None
                    if run_start is not None:
                        ranges.append((run_start, end))
    elif fmt == 6 and len(sub) >= 10:
        first, count = _u16(sub, 6), _u16(sub, 8)
        run_start = None
        for i in range(count):
            glyph = _u16(sub, 10 + i * 2) if 10 + i * 2 + 2 <= len(sub) else 0
            if glyph and run_start is None:
                run_start = first + i
            if not glyph and run_start is not None:
                ranges.append((run_start, first + i - 1))
                run_start = None
        if run_start is not None:
            ranges.append((run_start, first + count - 1))
    elif fmt == 0:
        for cp in range(min(256, len(sub) - 6)):
            if sub[6 + cp]:
                ranges.append((cp, cp))
    return _merge_ranges(ranges)


def _parse_fvar(table: bytes, names: dict[int, str]) -> tuple[FontAxis, ...]:
    if len(table) < 16:
        return ()
    axes_off = _u16(table, 4)
    axis_count, axis_size = _u16(table, 8), _u16(table, 10)
    axes: list[FontAxis] = []
    for i in range(axis_count):
        rec = axes_off + i * axis_size
        if rec + 20 > len(table):
            break
        tag = table[rec:rec + 4].decode("latin-1")
        mn, df, mx = struct.unpack_from(">iii", table, rec + 4)
        name_id = _u16(table, rec + 18)
        axes.append(FontAxis(tag=tag, minimum=mn / 65536.0,
                             default=df / 65536.0, maximum=mx / 65536.0,
                             name=names.get(name_id, "")))
    return tuple(axes)


def _face_from_tables(tables: dict[str, bytes], path: str,
                      face_index: int, file_id: str) -> FontFace | None:
    names = _parse_names(tables.get("name", b""))
    family = names.get(16) or names.get(1) or ""
    if not family:
        return None
    subfamily = names.get(17) or names.get(2) or "Regular"
    os2 = tables.get("OS/2", b"")
    weight, width_class, family_class, family_sub = 400, 5, 0, 0
    fs_selection = 0
    if len(os2) >= 66:
        weight = _u16(os2, 4) or 400
        width_class = _u16(os2, 6) or 5
        fs_selection = _u16(os2, 62)
        fam = _i16(os2, 30)  # sFamilyClass: class high, subclass low
        family_class, family_sub = (fam >> 8) & 0xFF, fam & 0xFF
    head = tables.get("head", b"")
    mac_style = _u16(head, 44) if len(head) >= 46 else 0
    units_per_em = _u16(head, 18) if len(head) >= 20 else 1000
    post = tables.get("post", b"")
    fixed_pitch = bool(_u32(post, 12)) if len(post) >= 16 else False
    italic = bool(fs_selection & 0x01) or bool(mac_style & 0x02)
    if not (fs_selection & 0x20) and (mac_style & 0x01) and weight < 600 \
            and subfamily.lower().startswith(("bold",)):
        weight = 700
    ranges = _cmap_ranges(tables["cmap"]) if "cmap" in tables else ()
    return FontFace(
        family=family, subfamily=subfamily,
        full_name=names.get(4, f"{family} {subfamily}"),
        postscript_name=names.get(6, ""),
        weight=weight, width_class=width_class, italic=italic,
        fixed_pitch=fixed_pitch, family_class=family_class,
        family_subclass=family_sub,
        axes=_parse_fvar(tables["fvar"], names) if "fvar" in tables else (),
        scripts=_all_scripts(ranges) if ranges else {},
        coverage_known=bool(ranges), ranges=ranges,
        color=any(t in tables for t in ("COLR", "CBDT", "sbix", "SVG ")),
        units_per_em=units_per_em or 1000,
        path=path, face_index=face_index, file_id=file_id,
        licence=names.get(13, ""), licence_url=names.get(14, ""),
        copyright=names.get(0, ""),
    )


def parse_font_file(path: str | Path) -> list[FontFace]:
    """Parse every face in one font file; [] when unparseable."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return []
    if len(data) < 12:
        return []
    file_id = hashlib.sha256(data).hexdigest()
    signature = data[:4]
    try:
        if signature == b"ttcf":
            count = _u32(data, 8)
            faces: list[FontFace] = []
            for i in range(min(count, 64)):
                offset = _u32(data, 12 + i * 4)
                tables = _sfnt_tables(data, offset)
                face = _face_from_tables(tables, str(p), i, file_id)
                if face is not None:
                    faces.append(face)
            return faces
        if signature == b"wOFF":
            tables = _woff_tables(data)
            face = _face_from_tables(tables, str(p), 0, file_id)
            return [face] if face is not None else []
        if signature == b"wOF2":
            return _woff2_faces(p, file_id)
        if signature in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"):
            tables = _sfnt_tables(data, 0)
            face = _face_from_tables(tables, str(p), 0, file_id)
            return [face] if face is not None else []
    except (struct.error, zlib.error, IndexError, OverflowError):
        return []
    return []
