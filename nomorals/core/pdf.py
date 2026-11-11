"""Pure-Python PDF writer — no dependencies, works on a phone.

Generates a standards-compliant, multi-page PDF (Helvetica,
A4 or Letter) from markdown-ish text: plain paragraphs by default, and
with ``headings=True`` real book layout — bold h1/h2/h3, chapters on fresh
pages, and a two-pass table of contents with true page numbers.  Enough for real deliverables: research
reports, OSINT summaries, datasets manifests — anything that needs to
leave the bot as a file.  Binary-safe input, WinAnsi-safe output (non
-Latin-1 characters are transliterated or replaced, never corrupting the
file), word wrapping, per-page headers/footers, and a correct xref table.
"""
from __future__ import annotations

import math
import re
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Iterable

from .branding import BrandMark

__all__ = ["PdfError", "PdfPage", "render_pdf", "pdf_write", "read_pdf_text",
           "pdf_read", "render_pdf_embedded"]


class PdfError(ValueError):
    """Raised for structurally invalid PDF input."""


# WinAnsi is what /WinAnsiEncoding maps to; keep a small table for the
# characters people actually use that differ from plain Latin-1.
_WINANSI_SPECIAL = {
    "\u2018": 0x91, "\u2019": 0x92, "\u201C": 0x93, "\u201D": 0x94,
    "\u2013": 0x96, "\u2014": 0x97, "\u2026": 0x85, "\u00B7": 0xB7,
    "\u02DC": 0x88, "\u2022": 0x95, "\u20AC": 0x80, "\u0192": 0x99,
}

# Simple ASCII fallbacks for the most common Unicode that WinAnsi lacks.
_ASCII_FALLBACK = {
    "\u2018": "'", "\u2019": "'", "\u2018": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u00B0": " deg",
    "\u00D7": "x", "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
    "\u2265": ">=", "\u2192": "->", "\u00B1": "+/-", "\u2020": "*",
    "\u2021": "*", "\u00A9": "(c)", "\u00AE": "(r)", "\u2122": "(tm)",
}

# Rough Helvetica glyph widths in thousandths of an em — close enough for
# wrapping (real metrics would need a font file; this keeps us dependency-free).
_AvgWidth = 500
_MONO_HINT = re.compile(r"^[\s\dA-Za-z0-9_./:=|\\-]*$")


def _encode_char(ch: str) -> bytes | None:
    """One character to WinAnsi bytes, or None when it cannot be represented."""
    if ch in _WINANSI_SPECIAL:
        return bytes([_WINANSI_SPECIAL[ch]])
    code = ord(ch)
    if code < 0x100:
        return bytes([code])
    return None


def _to_pdf_text(text: str) -> str:
    """Best-effort WinAnsi encoding; unknown characters become '?'.

    We never raise here — a report must render, even if a glyph is lost.
    """
    out: list[str] = []
    for ch in text:
        encoded = _encode_char(ch)
        if encoded is not None:
            out.append(encoded.decode("latin-1"))
        elif ch in _ASCII_FALLBACK:
            out.append(_ASCII_FALLBACK[ch])
        elif ch == "\u00A0":
            out.append(" ")
        else:
            out.append("?")
    return "".join(out)


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _wrap_line(line: str, budget: int, mono: bool) -> list[str]:
    """Word-wrap ``line`` to roughly ``budget`` characters.

    Monospaced lines are capped tighter because fixed-width glyphs are
    wider per character in Helvetica.
    """
    if not line:
        return [""]
    cap = max(20, budget) if not mono else max(20, int(budget * 0.72))
    if len(line) <= cap:
        return [line]
    words = line.split(" ")
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or len(candidate) <= cap:
            current = candidate
        else:
            parts.append(current)
            # a single word longer than the budget gets hard-broken
            while len(word) > cap:
                parts.append(word[:cap])
                word = word[cap:]
            current = word
    if current:
        parts.append(current)
    return parts


@dataclass
class PdfPage:
    """One logical page: a list of paragraphs (blank line separated)."""

    lines: list[str] = field(default_factory=list)


def _paginate(paragraphs: Iterable[str], *, width_chars: int) -> list[list[str]]:
    """Lay paragraphs into pages of wrapped lines with page budget."""
    pages: list[list[str]] = []
    current: list[str] = []
    max_lines = 54  # A4 at 10pt with margins
    for paragraph in paragraphs:
        for line in paragraph.split("\n"):
            mono = bool(_MONO_HINT.match(line))
            for wrapped in _wrap_line(line, width_chars, mono):
                if len(current) >= max_lines:
                    pages.append(current)
                    current = []
                current.append(wrapped)
        if len(current) >= max_lines:
            pages.append(current)
            current = []
    if current or not pages:
        pages.append(current)
    return pages


# ── the designed watermark ──────────────────────────────────────────
#
# A real on-page watermark, drawn FIRST in every page's content
# stream (so it sits UNDER the content) as page furniture: the
# brand text tiled diagonally across the page at 32 degrees, in a
# mid gray under an ExtGState alpha — visible as a deliberate mark,
# subordinate to everything printed over it.  It is part of the
# page, not the text flow: extractors meet it as furniture ops,
# printers print it, and cropping one instance still leaves two.

_WATERMARK_ANGLE_DEG = 32.0
_WATERMARK_SIZE = 34
_WATERMARK_GRAY = 0.30
_WATERMARK_ROWS = (0.27, 0.50, 0.73)


def _coerce_watermark(watermark: object) -> BrandMark | None:
    """Normalize a render_pdf watermark argument: None → no mark;
    a BrandMark → itself; a non-empty string → a mark with that
    text; anything else is a caller error worth naming."""
    if watermark is None:
        return None
    if isinstance(watermark, BrandMark):
        return watermark
    if isinstance(watermark, str):
        text = watermark.strip()
        return BrandMark(text=text) if text else None
    raise PdfError(
        f"watermark must be a BrandMark, a string, or None — "
        f"got {type(watermark).__name__}")


def _num(value: float) -> str:
    """Compact PDF number: 4 decimals, no trailing zeros."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _watermark_placements(mark: BrandMark, *, page_width: int,
                          page_height: int
                          ) -> tuple[float, float, list]:
    """The tiled diagonal geometry BOTH PDF writers share:
    (cos, sin, [(x, y), ...]) — one placement per watermark
    row, the text's start point after the 32° rotation about
    the page-centre anchor.  ``render_pdf`` shows the text as
    a WinAnsi literal; ``render_pdf_embedded`` shows the same
    text as embedded-font glyphs — the placement math is one
    home so the two paths can never drift apart."""
    angle = math.radians(_WATERMARK_ANGLE_DEG)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Helvetica advances average ~0.52 em for mixed-case brand text
    text_w = len(mark.text) * _WATERMARK_SIZE * 0.52
    points = []
    for frac in _WATERMARK_ROWS:
        cx, cy = page_width / 2, page_height * frac
        points.append((cx - (text_w / 2) * cos_a,
                       cy - (text_w / 2) * sin_a))
    return cos_a, sin_a, points


def _watermark_ops(mark: BrandMark, *, page_width: int,
                   page_height: int) -> str:
    """The watermark op block for one page (empty for no mark)."""
    text = _pdf_escape(_to_pdf_text(mark.text))
    if not text:
        return ""
    cos_a, sin_a, points = _watermark_placements(
        mark, page_width=page_width, page_height=page_height)
    ops = ["q /GSW gs"]
    for x, y in points:
        ops.append(
            f"BT /F1 {_WATERMARK_SIZE} Tf {_WATERMARK_GRAY:.2f} g "
            f"{_num(cos_a)} {_num(sin_a)} {_num(-sin_a)} {_num(cos_a)} "
            f"{x:.1f} {y:.1f} Tm ({text}) Tj ET"
        )
    ops.append("Q")
    return "\n".join(ops)


def _watermark_ops_embedded(mark: BrandMark, glyphs: bytes, *,
                            page_width: int,
                            page_height: int) -> bytes:
    """The same watermark block for the embedded-font writer:
    identical geometry, alpha and tiling as :func:`_watermark_ops`,
    but the text is shown as Identity-H glyph ids (a hex string)
    because the page's F1 is the subsetted TrueType face, not
    Helvetica.  Empty when the font carries none of the mark's
    glyphs — the caller then omits the ExtGState entirely,
    exactly as the WinAnsi path omits an unshowable mark."""
    if not glyphs:
        return b""
    cos_a, sin_a, points = _watermark_placements(
        mark, page_width=page_width, page_height=page_height)
    hex_text = glyphs.hex().encode("ascii")
    ops = [b"q /GSW gs"]
    for x, y in points:
        ops.append(
            f"BT /F1 {_WATERMARK_SIZE} Tf {_WATERMARK_GRAY:.2f} g "
            f"{_num(cos_a)} {_num(sin_a)} {_num(-sin_a)} "
            f"{_num(cos_a)} {x:.1f} {y:.1f} Tm <".encode("ascii")
            + hex_text + b"> Tj ET")
    ops.append(b"Q")
    return b"\n".join(ops)


def _extgstate_body(mark: BrandMark) -> bytes:
    alpha = _num(mark.opacity)
    return (f"<< /Type /ExtGState /CA {alpha} /ca {alpha} >>"
            ).encode("latin-1")


def _content_stream(lines: list[str], *, page_width: int, page_height: int,
                    font_size: int = 10, header: str = "", footer: str = "",
                    watermark_ops: str = "") -> bytes:
    """Build one page's content stream (BT/Tf/Tm/Tj/ET)."""
    left = 56  # ~1cm margin
    top = page_height - 56
    leading = font_size + 4
    ops: list[str] = []
    if watermark_ops:
        ops.append(watermark_ops)
    if header:
        ops.append(
            f"BT /F2 {font_size - 2} Tf 1 0 0 1 {left} {top - 14} Tm "
            f"({_pdf_escape(_to_pdf_text(header))}) Tj ET"
        )
    y = top
    for line in lines:
        text = _pdf_escape(_to_pdf_text(line))
        if text:
            ops.append(f"BT /F1 {font_size} Tf 1 0 0 1 {left} {y} Tm ({text}) Tj ET")
        y -= leading
    if footer:
        ops.append(
            f"BT /F2 {font_size - 2} Tf 1 0 0 1 {left} 40 Tm "
            f"({_pdf_escape(_to_pdf_text(footer))}) Tj ET"
        )
    return ("\n".join(ops)).encode("latin-1", "replace")



# ── structured (book) rendering ─────────────────────────────────────────────
#
# A styled block is a (kind, text) pair.  The legacy path below renders
# plain strings and is byte-stable for pre-book documents; the structured
# path reuses the same object-assembly rules with an optional bold font.

_BLOCK_BODY = "body"
_BLOCK_BOLD = "bold"
_BLOCK_H1 = "h1"
_BLOCK_H2 = "h2"
_BLOCK_H3 = "h3"
_BLOCK_SPACER = "spacer"
_BLOCK_TABLE = "table"  #: one markdown table row, pre-aligned for Courier

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_LINE = re.compile(r"^\|.*\|\s*$")
_TABLE_SEP = re.compile(r":?-+:?")


def _table_row_blocks(lines: list[str]) -> list[str]:
    """Parse markdown pipe-table lines into pre-aligned Courier row strings.

    Returns one string per rendered row: the header, a dashed rule, then
    data rows. Column alignment markers (``:---`` / ``---:``) are honored.
    """
    raw: list[list[str]] = []
    aligns: list[str] = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(_TABLE_SEP.fullmatch(c or "-") for c in cells):
            aligns = ["r" if c.endswith(":") and not c.startswith(":")
                      else "c" if c.startswith(":") and c.endswith(":")
                      else "l" for c in cells]
            continue
        raw.append(cells)
    if not raw:
        return []
    ncols = max(len(r) for r in raw)
    rows = [r + [""] * (ncols - len(r)) for r in raw]
    widths = [max(len(r[c]) for r in rows) for c in range(ncols)]
    while len(aligns) < ncols:
        aligns.append("l")

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for c, (cell, w, a) in enumerate(zip(cells, widths, aligns)):
            parts.append(cell.rjust(w) if a == "r"
                         else cell.center(w) if a == "c"
                         else cell.ljust(w))
        return "  ".join(parts).rstrip()

    out = [fmt_row(rows[0])]
    out.append("  ".join("-" * w for w in widths))
    out.extend(fmt_row(r) for r in rows[1:])
    return out


def _parse_blocks(text: str, *, strip_title_line: bool) -> tuple[str, list[tuple[str, str]]]:
    """Markdown-ish lines to styled blocks.

    ``# ``/``## ``/``### `` become h1/h2/h3; a whole-line ``**...**`` wrap
    becomes bold; other ``**`` pairs are stripped (words kept).  Pipe
    tables (``| a | b |`` runs) become one ``table`` block per row,
    pre-aligned for the monospaced font.  When ``strip_title_line`` is
    set, the FIRST h1 is promoted to the document title (legacy
    behaviour) and removed from the body.
    """
    blocks: list[tuple[str, str]] = []
    title = ""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            continue  # fence markers; inner lines pass through as body
        if _TABLE_LINE.match(stripped):
            j = i
            while j < len(lines) and _TABLE_LINE.match(lines[j].strip()):
                j += 1
            for row_text in _table_row_blocks(lines[i:j]):
                blocks.append((_BLOCK_TABLE, row_text))
            i = j
            continue
        m = _HEADING_LINE.match(stripped)
        if m:
            level = len(m.group(1))
            htext = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2)).strip()
            if not htext:
                i += 1
                continue
            if level == 1:
                if not title and strip_title_line:
                    title = htext
                    i += 1
                    continue
                blocks.append((_BLOCK_H1, htext))
            elif level == 2:
                blocks.append((_BLOCK_H2, htext))
            elif level == 3:
                blocks.append((_BLOCK_H3, htext))
            else:
                blocks.append((_BLOCK_BODY, htext))
            i += 1
            continue
        bold = (stripped.startswith("**") and stripped.endswith("**")
                and len(stripped) > 4)
        plain = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped).rstrip()
        if not plain:
            i += 1
            continue
        blocks.append((_BLOCK_BOLD if bold else _BLOCK_BODY, plain))
        i += 1
    return title, blocks


def _paginate_blocks(
    blocks: list[tuple[str, str]],
    *,
    width_chars: int,
    chapter_break: bool = False,
    max_lines: int = 54,
) -> list[list[tuple[str, str]]]:
    """Lay styled blocks into pages.  h1 forces a fresh page when
    ``chapter_break``; headings get one spacer line of breathing room."""
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    def push(block: tuple[str, str]) -> None:
        nonlocal current
        if len(current) >= max_lines:
            pages.append(current)
            current = []
        current.append(block)

    for kind, text in blocks:
        if kind == _BLOCK_H1 and chapter_break:
            if current:
                pages.append(current)
                current = []
            elif not pages:
                current = []
            else:
                pages.append([])
                current = []
            current.append((_BLOCK_SPACER, ""))
            current.append((kind, text))
            current.append((_BLOCK_SPACER, ""))
            continue
        if kind in (_BLOCK_H2, _BLOCK_H3) and current:
            push((_BLOCK_SPACER, ""))
        push((kind, text))
    if current or not pages:
        pages.append(current)
    return pages


def _content_stream_styled(
    lines: list[tuple[str, str]],
    *,
    page_width: int,
    page_height: int,
    font_size: int = 10,
    header: str = "",
    footer: str = "",
    watermark_ops: str = "",
) -> bytes:
    """One page of styled lines: F3 (bold) for headings, F1 for body,
    F4 (Courier) for table rows."""
    left = 56
    top = page_height - 56
    leading = font_size + 4
    ops: list[str] = []
    if watermark_ops:
        ops.append(watermark_ops)
    if header:
        ops.append(
            f"BT /F2 {font_size - 2} Tf 1 0 0 1 {left} {top - 14} Tm "
            f"({_pdf_escape(_to_pdf_text(header))}) Tj ET"
        )
    y = top
    for kind, text in lines:
        out = _to_pdf_text(text)
        if kind == _BLOCK_SPACER or not out.strip():
            y -= leading
            continue
        if kind == _BLOCK_H1:
            font, size = "/F3", font_size + 6
        elif kind == _BLOCK_H2:
            font, size = "/F3", font_size + 3
        elif kind in (_BLOCK_H3, _BLOCK_BOLD):
            font, size = "/F3", font_size
        elif kind == _BLOCK_TABLE:
            font, size = "/F4", font_size
        else:
            font, size = "/F1", font_size
        ops.append(f"BT {font} {size} Tf 1 0 0 1 {left} {y} Tm ({_pdf_escape(out)}) Tj ET")
        y -= leading
    if footer:
        ops.append(
            f"BT /F2 {font_size - 2} Tf 1 0 0 1 {left} 40 Tm "
            f"({_pdf_escape(_to_pdf_text(footer))}) Tj ET"
        )
    return ("\n".join(ops)).encode("latin-1", "replace")


def _info_dict(metadata: dict) -> bytes:
    """Build a PDF /Info dictionary body from a metadata mapping.

    Recognized keys: title, author, subject, keywords, creator.
    """
    parts: list[str] = []
    for key, pdf_key in (("title", "/Title"), ("author", "/Author"),
                         ("subject", "/Subject"), ("keywords", "/Keywords"),
                         ("creator", "/Creator")):
        val = metadata.get(key)
        if val:
            parts.append(
                f"{pdf_key} ({_pdf_escape(_to_pdf_text(str(val)))})")
    parts.append("/Producer (nomorals core.pdf)")
    parts.append(f"/CreationDate (D:{time.strftime('%Y%m%d%H%M%S', time.gmtime())}Z)")
    return ("<< " + " ".join(parts) + " >>").encode("latin-1", "replace")


def _assemble(
    pages: list[list[tuple[str, str]]],
    *,
    page_width: int,
    page_height: int,
    font_size: int,
    header: str,
    footer: str,
    has_bold: bool,
    metadata: dict | None = None,
    mark: BrandMark | None = None,
) -> bytes:
    """Object assembly for styled pages (mirrors the legacy numbering)."""
    n_pages = len(pages)
    has_table = any(kind == _BLOCK_TABLE for page in pages for kind, _ in page)
    # font objects: 3 = F1 Helvetica, 4 = F2 Oblique, then optional
    # 5 = F3 Bold and F4 Courier; page objects start after them.
    next_id = 5
    font_entries = ["/F1 3 0 R", "/F2 4 0 R"]
    slots: dict[int, bytes] = {}
    if has_bold:
        slots[5] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                    b"/Encoding /WinAnsiEncoding >>")
        font_entries.append("/F3 5 0 R")
        next_id = 6
    if has_table:
        slots[next_id] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier "
                          b"/Encoding /WinAnsiEncoding >>")
        font_entries.append(f"/F4 {next_id} 0 R")
        next_id += 1
    first = next_id
    page_obj_ids = [first + 2 * i for i in range(n_pages)]
    content_obj_ids = [first + 1 + 2 * i for i in range(n_pages)]
    font_map = " ".join(font_entries)

    children = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    catalog = f"<< /Type /Catalog /Pages 2 0 R"
    info_id = first + 2 * n_pages
    if metadata:
        catalog += f" /Info {info_id} 0 R"
        slots[info_id] = _info_dict(metadata)
    extgs_id = info_id + 1 if metadata else info_id
    extgs_entry = ""
    wm_ops = ""
    if mark is not None:
        wm_ops = _watermark_ops(mark, page_width=page_width,
                                page_height=page_height)
        if wm_ops:
            slots[extgs_id] = _extgstate_body(mark)
            extgs_entry = f" /ExtGState << /GSW {extgs_id} 0 R >>"
        else:
            wm_ops = ""
    slots[1] = (catalog + " >>").encode()
    slots[2] = f"<< /Type /Pages /Kids [{children}] /Count {n_pages} >>".encode()
    slots[3] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                b"/Encoding /WinAnsiEncoding >>")
    slots[4] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique "
                b"/Encoding /WinAnsiEncoding >>")

    for i in range(n_pages):
        stream = _content_stream_styled(
            pages[i], page_width=page_width, page_height=page_height,
            font_size=font_size, header=header,
            footer=footer.replace("N", str(i + 1)).replace("{total}", str(n_pages)),
            watermark_ops=wm_ops,
        )
        compressed = zlib.compress(stream)
        slots[content_obj_ids[i]] = (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n"
            f"stream\n".encode() + compressed + b"\nendstream"
        )
        slots[page_obj_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << {font_map} >>{extgs_entry} >> "
            f"/Contents {content_obj_ids[i]} 0 R >>".encode()
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(slots):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += slots[obj_id]
        out += b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(offsets)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


def _render_structured(
    text: str,
    *,
    title: str,
    page_size: str,
    font_size: int,
    header: str,
    footer: str,
    chapter_break: bool,
    toc: bool,
    metadata: dict | None = None,
    mark: BrandMark | None = None,
) -> bytes:
    """Book layout: styled blocks, optional chapter page breaks, and a
    two-pass table of contents whose page numbers converge on the true
    pagination (pagination is deterministic, so this always settles)."""
    page_width, page_height = (595, 842) if page_size == "A4" else (612, 792)
    text = (text or "").strip()
    if not text:
        raise PdfError("refusing to render an empty PDF")

    title, blocks = _parse_blocks(text, strip_title_line=not title)
    chars_per_line = 96 if page_size == "A4" else 92
    has_h1 = any(k == _BLOCK_H1 for k, _ in blocks)
    # the cover: an explicit title that ALSO appears as an h1 line is a title
    # page, not a chapter — it renders as a heading but never makes the
    # table of contents.  (Position heuristics can't tell a cover title from
    # a first chapter; the title text can.)
    cover_block = -1
    if title:
        for i, (k, t) in enumerate(blocks):
            if k == _BLOCK_H1 and t.strip() == title.strip():
                cover_block = i
                break
    toc_block_ids = {i for i, (k, _) in enumerate(blocks)
                     if k == _BLOCK_H1 and i != cover_block}

    def layout(candidate: list[tuple[str, str]]) -> tuple[list, list]:
        """(pages, h1 locations) — locations are (page_index, block index in
        ``candidate``) so the TOC can tell front matter from chapters."""
        pages = _paginate_blocks(candidate, width_chars=chars_per_line,
                                 chapter_break=chapter_break)
        h1_iter = iter(i for i, (k, _) in enumerate(candidate) if k == _BLOCK_H1)
        locs: list[tuple[int, int]] = []
        for pi, page in enumerate(pages):
            for kind, _txt in page:
                if kind == _BLOCK_H1:
                    locs.append((pi, next(h1_iter)))
        return pages, locs

    pages, locs = layout(blocks)
    if toc and toc_block_ids:
        for _ in range(4):  # page numbers shift by the TOC's own size; settles fast
            toc_lines = [(pi, bi) for pi, bi in locs if bi in toc_block_ids]
            toc_blocks: list[tuple[str, str]] = [(_BLOCK_H2, "Contents")]
            for n, (pi, bi) in enumerate(toc_lines, 1):
                toc_blocks.append((_BLOCK_BODY, f"{n}. {blocks[bi][1]}  (p. {pi + 1})"))
            offset = len(toc_blocks)
            cpages, clocs = layout(toc_blocks + blocks)
            # back to original-block coordinates for the next round
            clocs = [(pi, bi - offset) for pi, bi in clocs if bi >= offset]
            if clocs == locs:
                pages = cpages
                break
            pages, locs = cpages, clocs

    has_bold = any(k != _BLOCK_BODY for page in pages for k, _ in page)
    if title and not header:
        header = title
    now = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    if not footer:
        footer = f"{title} — page N — {now}"
    return _assemble(pages, page_width=page_width, page_height=page_height,
                     font_size=font_size, header=header, footer=footer,
                     has_bold=has_bold, metadata=metadata, mark=mark)


def render_pdf(
    text: str,
    *,
    title: str = "Report",
    page_size: str = "A4",
    font_size: int = 10,
    header: str = "",
    footer: str = "",
    headings: bool = False,
    chapter_break: bool = False,
    toc: bool = False,
    metadata: dict | None = None,
    watermark: BrandMark | str | None = None,
) -> bytes:
    """Render ``text`` (markdown-ish plain text) into a complete PDF file.

    Blank lines separate paragraphs; a leading ``# `` line becomes the
    document title if no explicit title is given.  Returns the full
    ``%PDF`` byte string, ready to write to disk or send as a file.

    ``headings=True`` switches to book layout: ``#``/``##``/``###`` render
    as bold headings (F3), whole-line ``**bold**`` renders bold,
    ``chapter_break=True`` starts every h1 on a fresh page, and
    ``toc=True`` prepends a table of contents with true page numbers
    (two-pass; pagination is deterministic so the numbers always match).

    Markdown pipe tables (``| a | b |``) render as monospaced tables with
    a dashed header rule; alignment markers (``:---``/``---:``) are
    honored.  ``metadata`` sets the document info dictionary (title,
    author, subject, keywords, creator).  Footers may use ``N`` for the
    page number and ``{total}`` for the page count.

    ``watermark`` draws the designed brand watermark on EVERY page,
    under the content: a :class:`~nomorals.core.branding.BrandMark`
    (resolve one from a policy with
    :func:`~nomorals.core.branding.resolve_brand_mark`), or a plain
    string for its text.  ``None`` (the default) draws no mark —
    producers of public-bound PDFs pass the resolved mark; personal
    documents simply don't.
    """
    if page_size not in {"A4", "Letter"}:
        raise PdfError(f"unknown page size {page_size!r}; use A4 or Letter")
    mark = _coerce_watermark(watermark)
    if headings:
        return _render_structured(
            text, title=title, page_size=page_size, font_size=font_size,
            header=header, footer=footer,
            chapter_break=chapter_break and headings, toc=toc,
            metadata=metadata, mark=mark,
        )
    page_width, page_height = (595, 842) if page_size == "A4" else (612, 792)

    text = (text or "").strip()
    if not text:
        raise PdfError("refusing to render an empty PDF")

    if title and not header:
        header = title
    # markdown title promotion: "# Heading" on the first line
    first = text.split("\n", 1)[0]
    if not title and first.startswith("# "):
        title = first[2:].strip()
    # strip markdown chrome we do not render natively (keep the words)
    cleaned = [re.sub(r"\*\*(.+?)\*\*", r"\1", ln) for ln in text.split("\n")]
    cleaned = [
        "" if ln.strip().startswith("```") else ln
        for ln in cleaned
    ]
    cleaned = ["" if ln.strip().startswith("# ") else ln for ln in cleaned]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in cleaned:
        if not line.strip():
            if buffer:
                paragraphs.append("\n".join(buffer))
                buffer = []
        else:
            buffer.append(line.rstrip())
    if buffer:
        paragraphs.append("\n".join(buffer))

    chars_per_line = 96 if page_size == "A4" else 92
    pages = _paginate(paragraphs, width_chars=chars_per_line)
    now = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    if not footer:
        footer = f"{title} — page N — {now}"

    objects: list[bytes] = []  # 1-indexed by insertion; object #i = objects[i-1]

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # Object numbering plan:
    # 1 catalog, 2 pages tree, 3 font F1, 4 font F2, then per page:
    # page object + content object.
    n_pages = len(pages)
    page_obj_ids = [5 + 2 * i for i in range(n_pages)]
    content_obj_ids = [6 + 2 * i for i in range(n_pages)]

    # We must emit objects in id order; build them by index.
    slots: dict[int, bytes] = {}
    catalog = "<< /Type /Catalog /Pages 2 0 R"
    info_id = 5 + 2 * n_pages
    if metadata:
        catalog += f" /Info {info_id} 0 R"
        slots[info_id] = _info_dict(metadata)
    extgs_id = info_id + 1 if metadata else info_id
    extgs_entry = ""
    wm_ops = ""
    if mark is not None:
        wm_ops = _watermark_ops(mark, page_width=page_width,
                                page_height=page_height)
        if wm_ops:
            slots[extgs_id] = _extgstate_body(mark)
            extgs_entry = f" /ExtGState << /GSW {extgs_id} 0 R >>"
        else:
            wm_ops = ""
    slots[1] = (catalog + " >>").encode()
    children = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    slots[2] = (
        f"<< /Type /Pages /Kids [{children}] /Count {n_pages} >>".encode()
    )
    slots[3] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    slots[4] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique "
        b"/Encoding /WinAnsiEncoding >>"
    )
    for i in range(n_pages):
        stream = _content_stream(
            pages[i], page_width=page_width, page_height=page_height,
            font_size=font_size, header=header,
            footer=footer.replace("N", str(i + 1)).replace("{total}", str(n_pages)),
            watermark_ops=wm_ops,
        )
        compressed = zlib.compress(stream)
        slots[content_obj_ids[i]] = (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n"
            f"stream\n".encode()
            + compressed
            + b"\nendstream"
        )
        slots[page_obj_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >>{extgs_entry} >> "
            f"/Contents {content_obj_ids[i]} 0 R >>".encode()
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(slots):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += slots[obj_id]
        out += b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(offsets)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


def pdf_write(path: str, text: str, **kwargs) -> tuple[str, int]:
    """Render to ``path``; returns (path, bytes)."""
    data = render_pdf(text, **kwargs)
    with open(path, "wb") as handle:
        handle.write(data)
    return str(path), len(data)


# ── reading ─────────────────────────────────────────────────────────────────


def _parse_objects(data: bytes) -> dict[int, bytes]:
    """Map object number → raw object body.  Works even when the xref table
    is broken or the file was truncated: it scans for 'N 0 obj' markers."""
    objects: dict[int, bytes] = {}
    for match in re.finditer(rb"(\d+)\s+0\s+obj\b", data):
        num = int(match.group(1))
        start = match.end()
        end = data.find(b"endobj", start)
        if end < 0:
            continue
        objects[num] = data[start:end].strip()
    return objects


def _resolve_ref(body: bytes, name: bytes, objects: dict[int, bytes]) -> bytes | None:
    """Pull a named entry out of a dictionary body; follows one indirection."""
    m = re.search(name + rb"\s+(\d+)\s+0\s+R", body)
    if m:
        return objects.get(int(m.group(1)))
    m = re.search(name + rb"\s+(<<.*?>>)", body, re.S)
    if m:
        return m.group(1)
    m = re.search(name + rb"\s+(?!/)([^\s\[\]<][^\n]*)", body)
    if m:
        return m.group(1).strip()
    return None


def _collect_pages(objects: dict[int, bytes], root: bytes) -> list[bytes]:
    """Walk the (possibly nested) page tree in document order.

    ``root`` is the catalog body; page trees can be arbitrarily deep
    (Chromium exports nest them), so the walk is recursive on /Kids.
    """
    pages_ref = _resolve_ref(root, b"/Pages", objects)
    if pages_ref is None:
        return []
    out: list[bytes] = []
    _walk_pages(objects, pages_ref, out, set())
    return out


def _walk_pages(objects: dict[int, bytes], node_body: bytes,
                out: list[bytes], seen: set[int]) -> None:
    m = re.search(rb"/Kids\s*\[(.*?)\]", node_body, re.S)
    if not m:
        return
    for ref in re.findall(rb"(\d+)\s+0\s+R", m.group(1)):
        num = int(ref)
        if num in seen or num not in objects:
            continue
        seen.add(num)
        body = objects[num]
        # a leaf page has /Contents; an intermediate node has its own /Kids
        if re.search(rb"/Kids\s*\[", body):
            _walk_pages(objects, body, out, seen)
        elif re.search(rb"/Contents\b", body):
            out.append(body)


def _decode_pdf_string(raw: bytes) -> str:
    """A PDF literal string body (between the parens, escapes decoded)."""
    out = bytearray()
    i = 0
    while i < len(raw):
        ch = raw[i:i + 1]
        if ch == b"\\" and i + 1 < len(raw):
            nxt = raw[i + 1:i + 2]
            if nxt.isdigit():
                # octal escape, up to 3 digits
                j = i + 1
                digits = b""
                while j < len(raw) and raw[j:j + 1].isdigit() and len(digits) < 3:
                    digits += raw[j:j + 1]
                    j += 1
                out.append(int(digits, 8) & 0xFF if digits else 0)
                i = j
                continue
            simple = {b"n": 0x0A, b"r": 0x0D, b"t": 0x09, b"b": 0x08,
                      b"f": 0x0C, b"(": 0x28, b")": 0x29, b"\\": 0x5C}
            out.append(simple.get(nxt, 0x00))
            i += 2
            continue
        out += ch
        i += 1
    return out.decode("latin-1", "replace")


def _cmap_to_text(codepoints: bytes) -> str:
    return "".join(
        chr(int.from_bytes(codepoints[i:i + 2], "big"))
        for i in range(0, len(codepoints) - 1, 2)
    )


def _parse_cmap(data: bytes) -> tuple[dict[int, str], int]:
    """Parse a ToUnicode CMap → (glyph-id → unicode, glyph byte width)."""
    cmap: dict[int, str] = {}
    width = 2
    m = re.search(rb"begincodespacerange\s*<([0-9A-Fa-f]+)>", data)
    if m:
        width = max(1, len(m.group(1)) // 2)
    for block in re.finditer(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for gid_raw, uni_raw in re.findall(
                rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block.group(1)):
            try:
                uni = bytes.fromhex(uni_raw.decode())
                cmap[int(gid_raw, 16)] = _cmap_to_text(uni)
            except (ValueError, TypeError):
                continue
    for block in re.finditer(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for start_raw, end_raw, target in re.findall(
                rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<(?:[0-9A-Fa-f]+)>|\[.*?\])",
                block.group(1), re.S):
            try:
                s, e = int(start_raw, 16), int(end_raw, 16)
            except ValueError:
                continue
            if target.startswith(b"["):
                arr = re.findall(rb"<([0-9A-Fa-f]+)>", target)
                for i, gid in enumerate(range(s, e + 1)):
                    if i >= len(arr):
                        break
                    try:
                        cmap[gid] = _cmap_to_text(bytes.fromhex(arr[i].decode()))
                    except (ValueError, TypeError):  # noqa: E103 - malformed cmap entry skipped
                        pass
            else:
                try:
                    base = int(target.strip(b"<>"), 16)
                except ValueError:
                    continue
                for i, gid in enumerate(range(s, e + 1)):
                    if base + i < 0x110000:
                        cmap[gid] = chr(base + i)
    return cmap, width


_FONT_MAP = tuple[dict[int, str], int] | None


def _decode_hex(hexstr: bytes, font: _FONT_MAP) -> str:
    """A hex string body to text, via the active font's ToUnicode map."""
    hexstr = re.sub(rb"\s+", b"", hexstr)
    if not hexstr:
        return ""
    if font is not None:
        cmap, width = font
        out = []
        for i in range(0, len(hexstr) - width + 1, width):
            try:
                gid = int.from_bytes(bytes.fromhex(hexstr[i:i + width].decode()),
                                     "big")
            except (ValueError, TypeError):
                continue
            out.append(cmap.get(gid, ""))
        return "".join(out)
    # no CMap: 1-byte latin-1 is the common simple-font case
    try:
        raw = bytes.fromhex(hexstr.decode())
        return raw.decode("latin-1", "replace")
    except ValueError:
        return ""


def _draws_in_stream(stream: bytes,
                     font_maps: dict[str, _FONT_MAP] | None = None
                     ) -> list[tuple[float, float, str]]:
    """All (x, y, text) draws in a content stream, in order.

    Tracks the text matrix and the active font (Tf) so visual line breaks
    and CID glyph strings can be reconstructed.
    """
    draws: list[tuple[float, float, str]] = []
    x = y = 0.0
    active: _FONT_MAP = None
    font_maps = font_maps or {}
    pattern = re.compile(
        rb"/([A-Za-z]+\w*)\s+[\d.]+\s+Tf\b"
        rb"|([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+Tm\b"
        rb"|([-\d.]+)\s+([-\d.]+)\s+T[dD]\b"
        rb"|\bT\*\b"
        rb"|(\[(?:\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>|[^\]])*\])\s*TJ\b"
        rb"|(\((?:\\.|[^\\()])*\))\s*Tj\b"
        rb"|(<[0-9A-Fa-f\s]*>)\s*Tj\b"
    )
    for m in pattern.finditer(stream):
        if m.group(1) is not None:
            active = font_maps.get(m.group(1).decode())
        elif m.group(7) is not None:
            # Tm: a b c d e f — e, f are the new text origin
            try:
                x, y = float(m.group(6)), float(m.group(7))
            except (TypeError, ValueError):  # noqa: E103 - malformed Tm operands keep last position
                pass
        elif m.group(9) is not None:
            # Td/TD: move the line start
            try:
                x += float(m.group(8))
                y += float(m.group(9))
            except (TypeError, ValueError):  # noqa: E103 - malformed Td operands keep last position
                pass
        elif m.group(0).strip() == b"T*":
            # T*: next line down (leading unknown; any negative works)
            y -= 12
        elif m.group(10) is not None:
            # TJ array: literal/hex strings interleaved with kerning numbers
            text = ""
            for part in re.finditer(rb"\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>"
                                    rb"|([-\d.]+)", m.group(10)):
                token = part.group(0)
                if token.startswith(b"("):
                    text += _decode_pdf_string(token[1:-1])
                elif token.startswith(b"<"):
                    text += _decode_hex(token[1:-1], active)
                else:
                    try:
                        if float(part.group(1)) < -80:
                            text += " "
                    except (TypeError, ValueError):  # noqa: E103 - malformed kerning number ignored
                        pass
            if text:
                draws.append((x, y, text))
        elif m.group(11) is not None:
            text = _decode_pdf_string(m.group(11)[1:-1])
            if text:
                draws.append((x, y, text))
        elif m.group(12) is not None:
            text = _decode_hex(m.group(12)[1:-1], active)
            if text:
                draws.append((x, y, text))
    return draws


def _stream_bytes(body: bytes) -> bytes | None:
    """The raw stream payload of an object body, or None."""
    m = re.search(rb"stream\r?\n", body)
    if not m:
        return None
    end = body.rfind(b"endstream")
    return body[m.end():end if end > 0 else len(body)]


def _decompress(payload: bytes) -> bytes | None:
    if payload[:1] == b"\x78" or b"FlateDecode" in payload[:200]:
        for candidate in (payload, payload.rstrip(b"\r\n")):
            try:
                return zlib.decompress(candidate)
            except zlib.error:
                continue
        return None
    return payload


def _page_font_maps(page_body: bytes,
                    objects: dict[int, bytes]) -> dict[str, _FONT_MAP]:
    """Font name → ToUnicode map for every font this page declares."""
    maps: dict[str, _FONT_MAP] = {}
    fonts = re.search(rb"/Font\s*<<(.*?)>>", page_body, re.S)
    if not fonts:
        return maps
    for name, ref in re.findall(rb"/([A-Za-z]+\w*)\s+(\d+)\s+0\s+R",
                                fonts.group(1)):
        body = objects.get(int(ref), b"")
        tu = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", body)
        if not tu:
            continue
        payload = _stream_bytes(objects.get(int(tu.group(1)), b""))
        if payload is None:
            continue
        decoded = _decompress(payload) or payload
        cmap, width = _parse_cmap(decoded)
        if cmap:
            maps[name.decode()] = (cmap, width)
    return maps


def _page_text(page_body: bytes, objects: dict[int, bytes]) -> str:
    font_maps = _page_font_maps(page_body, objects)
    streams: list[bytes] = []
    contents = re.search(rb"/Contents\s+(\d+)\s+0\s+R", page_body)
    if contents:
        payload = _stream_bytes(objects.get(int(contents.group(1)), b""))
        if payload is not None:
            streams.append(payload)
    else:
        arr = re.search(rb"/Contents\s*\[(.*?)\]", page_body, re.S)
        if arr:
            for ref in re.findall(rb"(\d+)\s+0\s+R", arr.group(1)):
                payload = _stream_bytes(objects.get(int(ref), b""))
                if payload is not None:
                    streams.append(payload)
    if not streams:
        # content stream carried inline in the page object itself
        payload = _stream_bytes(page_body)
        if payload is not None:
            streams.append(payload)

    lines: list[str] = []
    current = ""
    prev_y: float | None = None
    for payload in streams:
        stream = _decompress(payload)
        if stream is None:
            continue
        for _dx, dy, text in _draws_in_stream(stream, font_maps):
            if not text:
                continue
            if prev_y is not None and abs(dy - prev_y) > 2:
                if current:
                    lines.append(current)
                current = ""
            current += text
            prev_y = dy
    if current:
        lines.append(current)
    return "\n".join(lines)


def read_pdf_text(data: bytes, *, max_pages: int = 100) -> str:
    """Extract readable text from a PDF.  Pure Python, no dependencies.

    Handles FlateDecode streams, TJ/Tj text operators, Tm/Td/TD/T* line
    positioning, escaped strings, and broken xref tables (falls back to
    scanning object markers).  Scanned/image-only PDFs return little or
    nothing — they need OCR, which is a separate capability.
    """
    if not data[:5].startswith(b"%PDF-"):
        raise PdfError("not a PDF file")
    objects = _parse_objects(data)
    if not objects:
        return ""
    root_match = re.search(rb"/Root\s+(\d+)\s+0\s+R", data)
    root = objects.get(int(root_match.group(1)), b"") if root_match else b""
    if not root:
        # last resort: any object that looks like a catalog
        root = next((b for b in objects.values()
                     if b"/Catalog" in b), b"")
    pages = _collect_pages(objects, root)
    if not pages:
        # last resort: every object with a /Contents
        pages = [b for b in objects.values()
                 if re.search(rb"/Contents\b", b)
                 and not re.search(rb"/Kids\s*\[", b)]
    texts: list[str] = []
    for page in pages[:max_pages]:
        t = _page_text(page, objects)
        if t.strip():
            texts.append(t)
    return "\n\n".join(texts).strip()


def pdf_read(path: str, *, max_pages: int = 100) -> str:
    """Read the text of the PDF at ``path``."""
    with open(path, "rb") as handle:
        return read_pdf_text(handle.read(), max_pages=max_pages)


# ── embedded TrueType (Wave 2: PDF write fidelity) ──────────────────────────
#
# The legacy writer above is WinAnsi-only: any character outside
# Latin-1 + the special table is transliterated or replaced with '?'.
# ``render_pdf_embedded`` closes that gap the way the PDF spec intends:
# a TrueType font is subsetted to the glyphs the document actually
# uses, embedded as a /FontFile2 stream, and text is drawn as Identity-H
# CIDs with a /ToUnicode CMap — so Greek, Cyrillic, CJK, math, emoji-free
# BMP text (and anything else the font carries) renders *and* extracts
# exactly.  Everything is stdlib: the TTF is parsed and rebuilt with
# ``struct``; no fontTools, no external engine.  CFF-flavored OpenType
# (.otf) is refused with a named error — the subsetter is TrueType-only.

_TTF_SCALER_TRUETYPE = (0x00010000, 0x74727565)  # 1.0 / 'true'


def _ttf_u16(data: bytes, off: int) -> int:
    return struct.unpack_from(">H", data, off)[0]


def _ttf_i16(data: bytes, off: int) -> int:
    return struct.unpack_from(">h", data, off)[0]


def _ttf_u32(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def _ttf_tables(data: bytes) -> dict[str, tuple[int, int]]:
    """tag -> (offset, length) for a TrueType font, or PdfError."""
    if len(data) < 12:
        raise PdfError("font file too small to be a TrueType font")
    scaler = _ttf_u32(data, 0)
    if scaler == 0x4F54544F:  # 'OTTO'
        raise PdfError(
            "CFF-flavored OpenType (.otf) cannot be embedded by the "
            "stdlib subsetter; supply a TrueType (.ttf) font")
    if scaler not in _TTF_SCALER_TRUETYPE:
        if data[:4] == b"ttcf":
            raise PdfError(
                "TrueType collections (.ttc) are not supported; supply "
                "a single .ttf font")
        raise PdfError("not a TrueType font (bad scaler type)")
    num = _ttf_u16(data, 4)
    tables: dict[str, tuple[int, int]] = {}
    for i in range(num):
        rec = 12 + i * 16
        tag = data[rec:rec + 4].decode("latin-1")
        tables[tag] = (_ttf_u32(data, rec + 8), _ttf_u32(data, rec + 12))
    for required in ("head", "hhea", "maxp", "hmtx", "cmap", "glyf", "loca"):
        if required not in tables:
            raise PdfError(f"TrueType font is missing its {required!r} table")
    return tables


def _ttf_cmap(data: bytes, tables: dict[str, tuple[int, int]]) -> dict[int, int]:
    """Unicode codepoint -> glyph id, best subtable (12 beats 4)."""
    off, _length = tables["cmap"]
    n = _ttf_u16(data, off + 2)
    fmt12: int | None = None
    fmt4: int | None = None
    for i in range(n):
        rec = off + 4 + i * 8
        platform, encoding = _ttf_u16(data, rec), _ttf_u16(data, rec + 2)
        sub = off + _ttf_u32(data, rec + 4)
        fmt = _ttf_u16(data, sub)
        if fmt == 12 and (platform in (0, 3)):
            fmt12 = sub
        elif fmt == 4 and (platform, encoding) in ((3, 1), (0, 3), (0, 2),
                                                   (0, 1), (0, 0)):
            if fmt4 is None:
                fmt4 = sub
    cmap: dict[int, int] = {}
    if fmt12 is not None:
        groups = _ttf_u32(data, fmt12 + 12)
        for g in range(groups):
            grec = fmt12 + 16 + g * 12
            start, end, base = (_ttf_u32(data, grec), _ttf_u32(data, grec + 4),
                                _ttf_u32(data, grec + 8))
            for cp in range(start, min(end, start + 0x110000) + 1):
                cmap[cp] = base + (cp - start)
        return cmap
    if fmt4 is None:
        raise PdfError("TrueType font has no format-4/12 cmap subtable")
    seg_count = _ttf_u16(data, fmt4 + 6) // 2
    end_off = fmt4 + 14
    start_off = end_off + 2 * seg_count + 2
    delta_off = start_off + 2 * seg_count
    range_off = delta_off + 2 * seg_count
    for s in range(seg_count):
        end = _ttf_u16(data, end_off + 2 * s)
        start = _ttf_u16(data, start_off + 2 * s)
        delta = _ttf_i16(data, delta_off + 2 * s)
        ro = _ttf_u16(data, range_off + 2 * s)
        for cp in range(start, end + 1):
            if cp == 0xFFFF:
                continue
            if ro == 0:
                gid = (cp + delta) & 0xFFFF
            else:
                gi_off = range_off + 2 * s + ro + 2 * (cp - start)
                gid = _ttf_u16(data, gi_off)
                if gid:
                    gid = (gid + delta) & 0xFFFF
            if gid:
                cmap[cp] = gid
    return cmap


class _TtfFont:
    """Parsed TrueType facts the embedder needs."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.tables = _ttf_tables(data)
        self.cmap = _ttf_cmap(data, self.tables)
        head_off = self.tables["head"][0]
        self.units_per_em = _ttf_u16(data, head_off + 18)
        self.bbox = tuple(_ttf_i16(data, head_off + 36 + 2 * i)
                          for i in range(4))
        self.loc_format = _ttf_i16(data, head_off + 50)
        hhea_off = self.tables["hhea"][0]
        self.ascent = _ttf_i16(data, hhea_off + 4)
        self.descent = _ttf_i16(data, hhea_off + 6)
        self.num_h_metrics = _ttf_u16(data, hhea_off + 34)
        maxp_off = self.tables["maxp"][0]
        self.num_glyphs = _ttf_u16(data, maxp_off + 4)
        self.cap_height = int(self.ascent * 0.7)
        if "OS/2" in self.tables:
            os2_off, os2_len = self.tables["OS/2"]
            if os2_len >= 90:
                cap = _ttf_i16(data, os2_off + 88)
                if cap:
                    self.cap_height = cap
        self.name = self._postscript_name()

    def _postscript_name(self) -> str:
        if "name" not in self.tables:
            return "EmbeddedFont"
        off, _length = self.tables["name"]
        count = _ttf_u16(self.data, off + 2)
        storage = off + _ttf_u16(self.data, off + 4)
        fallback = ""
        for i in range(count):
            rec = off + 6 + i * 12
            platform = _ttf_u16(self.data, rec)
            name_id = _ttf_u16(self.data, rec + 6)
            if name_id != 6:
                continue
            length = _ttf_u16(self.data, rec + 8)
            noff = storage + _ttf_u16(self.data, rec + 10)
            raw = self.data[noff:noff + length]
            try:
                text = raw.decode("utf-16-be") if platform in (0, 3) \
                    else raw.decode("latin-1")
            except UnicodeDecodeError:
                continue
            text = re.sub(r"[^A-Za-z0-9-]", "-", text).strip("-")
            if platform in (0, 3) and text:
                return text
            fallback = fallback or text
        return fallback or "EmbeddedFont"

    def glyph_offset(self, gid: int) -> tuple[int, int]:
        """(offset, length) of a glyph's glyf data; (0, 0) when empty."""
        loca_off, _ = self.tables["loca"]
        glyf_off, _ = self.tables["glyf"]
        if self.loc_format == 0:
            start = _ttf_u16(self.data, loca_off + 2 * gid) * 2
            end = _ttf_u16(self.data, loca_off + 2 * (gid + 1)) * 2
        else:
            start = _ttf_u32(self.data, loca_off + 4 * gid)
            end = _ttf_u32(self.data, loca_off + 4 * (gid + 1))
        return glyf_off + start, end - start

    def advance(self, gid: int) -> int:
        """Advance width of a glyph, in font units."""
        hmtx_off, _ = self.tables["hmtx"]
        if gid < self.num_h_metrics:
            return _ttf_u16(self.data, hmtx_off + 4 * gid)
        return _ttf_u16(
            self.data, hmtx_off + 4 * (self.num_h_metrics - 1))

    def width1000(self, gid: int) -> int:
        """Advance width normalized to a 1000-unit em (PDF /W units)."""
        return round(self.advance(gid) * 1000 / self.units_per_em)


def _ttf_component_gids(font: _TtfFont, gid: int) -> list[int]:
    """Glyph ids a composite glyph references (empty for simple glyphs)."""
    off, length = font.glyph_offset(gid)
    if length < 10:
        return []
    data = font.data
    if _ttf_i16(data, off) >= 0:
        return []
    out: list[int] = []
    pos = off + 10
    while True:
        flags = _ttf_u16(data, pos)
        out.append(_ttf_u16(data, pos + 2))
        pos += 4
        pos += 4 if flags & 0x0001 else 2       # args
        if flags & 0x0008:
            pos += 2                            # scale
        elif flags & 0x0040:
            pos += 4                            # x/y scale
        elif flags & 0x0080:
            pos += 8                            # 2x2
        if not flags & 0x0020:                  # MORE_COMPONENTS
            return out


def _ttf_rewrite_glyph(font: _TtfFont, gid: int,
                       gid_map: dict[int, int]) -> bytes:
    """A glyph's glyf bytes with composite references remapped."""
    off, length = font.glyph_offset(gid)
    raw = bytearray(font.data[off:off + length])
    if length < 10 or _ttf_i16(font.data, off) >= 0:
        return bytes(raw)
    pos = 10
    while True:
        flags = _ttf_u16(font.data, off + pos)
        old = _ttf_u16(font.data, off + pos + 2)
        struct.pack_into(">H", raw, pos + 2, gid_map[old])
        pos += 4
        pos += 4 if flags & 0x0001 else 2
        if flags & 0x0008:
            pos += 2
        elif flags & 0x0040:
            pos += 4
        elif flags & 0x0080:
            pos += 8
        if not flags & 0x0020:
            return bytes(raw)


def _ttf_table_checksum(blob: bytes) -> int:
    padded = blob + b"\0" * (-len(blob) % 4)
    total = 0
    for i in range(0, len(padded), 4):
        total = (total + int.from_bytes(padded[i:i + 4], "big")) & 0xFFFFFFFF
    return total


def _ttf_subset_cmap(cmap: dict[int, int]) -> bytes:
    """Format-4 cmap covering the used BMP codepoints -> NEW glyph ids.

    ``cmap`` here maps codepoint -> new glyph id (already subset-space).
    """
    cps = sorted(cp for cp in cmap if cp <= 0xFFFE)
    segments: list[tuple[int, int, list[int]]] = []
    for cp in cps:
        if segments and cp == segments[-1][1] + 1:
            segments[-1][2].append(cmap[cp])
            segments[-1] = (segments[-1][0], cp, segments[-1][2])
        else:
            segments.append((cp, cp, [cmap[cp]]))
    # (start, end, delta|None, glyph-array|None)
    built: list[tuple[int, int, int | None, list[int] | None]] = []
    glyph_array: list[int] = []
    for start, end, gids in segments:
        if all(gids[i] == gids[0] + i for i in range(len(gids))):
            built.append((start, end, (gids[0] - start) & 0xFFFF, None))
        else:
            built.append((start, end, None, gids))
            glyph_array.extend(gids)
    built.append((0xFFFF, 0xFFFF, 1, None))
    seg_count = len(built)
    range_offsets: list[int] = []
    positions: dict[int, int] = {}
    cursor = 0
    for idx, (_s, _e, _d, gids) in enumerate(built):
        if gids is not None:
            positions[idx] = cursor
            cursor += len(gids)
    for idx, (_s, _e, _d, gids) in enumerate(built):
        if gids is None:
            range_offsets.append(0)
        else:
            # idRangeOffset is measured from its own slot to the first
            # glyphId for this segment.
            slot_distance = 2 * (seg_count - idx)
            range_offsets.append(slot_distance + 2 * positions[idx])
    length = 16 + 8 * seg_count + 2 * len(glyph_array)
    pow2 = 1 << int(math.log2(seg_count))
    entry_selector = int(math.log2(pow2)) if pow2 else 0
    out = bytearray()
    out += struct.pack(">HHHHHHH", 4, length, 0, seg_count * 2,
                       pow2 * 2, entry_selector,
                       seg_count * 2 - pow2 * 2)
    for _s, end, _d, _g in built:
        out += struct.pack(">H", end)
    out += struct.pack(">H", 0)
    for start, _e, _d, _g in built:
        out += struct.pack(">H", start)
    for _s, _e, delta, _g in built:
        out += struct.pack(">H", delta if delta is not None else 0)
    for ro in range_offsets:
        out += struct.pack(">H", ro)
    for gid in glyph_array:
        out += struct.pack(">H", gid)
    header = struct.pack(">HHHHI", 0, 1, 3, 1, 12)
    return header + bytes(out)


def _ttf_subset(font: _TtfFont, used: dict[int, int]) -> tuple[bytes, dict[int, int]]:
    """Subset ``font`` to ``used`` (codepoint -> old gid).

    Returns (subset TTF bytes, codepoint -> new glyph id).  Glyph 0 is
    always kept; composite components are pulled in by closure.
    """
    keep = {0}
    keep.update(used.values())
    queue = list(keep)
    while queue:
        gid = queue.pop()
        for comp in _ttf_component_gids(font, gid):
            if comp not in keep:
                keep.add(comp)
                queue.append(comp)
    ordered = sorted(keep)
    gid_map = {old: new for new, old in enumerate(ordered)}

    src = font.data
    tables = font.tables

    def table_bytes(tag: str) -> bytes:
        off, length = tables[tag]
        return src[off:off + length]

    # glyf + loca
    glyf_parts: list[bytes] = []
    loca: list[int] = [0]
    for old in ordered:
        blob = _ttf_rewrite_glyph(font, old, gid_map)
        blob += b"\0" * (-len(blob) % 4)
        glyf_parts.append(blob)
        loca.append(loca[-1] + len(blob))
    glyf = b"".join(glyf_parts)
    use_long_loca = loca[-1] > 0x1FFFC
    if use_long_loca:
        loca_blob = b"".join(struct.pack(">I", v) for v in loca)
    else:
        loca_blob = b"".join(struct.pack(">H", v // 2) for v in loca)

    head = bytearray(table_bytes("head"))
    struct.pack_into(">I", head, 8, 0)  # checkSumAdjustment, fixed last
    struct.pack_into(">h", head, 50, 1 if use_long_loca else 0)
    maxp = bytearray(table_bytes("maxp"))
    struct.pack_into(">H", maxp, 4, len(ordered))
    hhea = bytearray(table_bytes("hhea"))
    struct.pack_into(">H", hhea, 34, len(ordered))
    hmtx = bytearray()
    for old in ordered:
        goff, glen = font.glyph_offset(old)
        # hmtx is (advance, lsb) pairs; a glyph's lsb is its glyf xMin.
        lsb = _ttf_i16(src, goff + 2) if glen >= 10 else 0
        hmtx += struct.pack(">Hh", font.advance(old), lsb)
    cmap = _ttf_subset_cmap({cp: gid_map[gid] for cp, gid in used.items()})
    post = struct.pack(">IihhIIIII", 0x00030000, 0, 0, 0, 0, 0, 0, 0, 0)
    out_tables: dict[str, bytes] = {
        "cmap": cmap,
        "glyf": glyf,
        "head": bytes(head),
        "hhea": bytes(hhea),
        "hmtx": bytes(hmtx),
        "loca": loca_blob,
        "maxp": bytes(maxp),
        "post": post,
    }
    for tag in ("name", "OS/2", "cvt ", "fpgm", "prep", "gasp"):
        if tag in tables:
            blob = table_bytes(tag)
            if tag == "OS/2" and len(blob) >= 68:
                blob = bytearray(blob)
                cps = sorted(used)
                if cps:
                    struct.pack_into(">H", blob, 64, min(cps[0], 0xFFFF))
                    struct.pack_into(">H", blob, 66, min(cps[-1], 0xFFFF))
                blob = bytes(blob)
            out_tables[tag] = blob

    tags = sorted(out_tables)
    num = len(tags)
    pow2 = 1 << int(math.log2(num))
    header = struct.pack(">IHHHH", 0x00010000, num, pow2 * 16,
                         int(math.log2(pow2)), num * 16 - pow2 * 16)
    offset = 12 + num * 16
    records = bytearray()
    body = bytearray()
    head_offset = 0
    for tag in tags:
        blob = out_tables[tag]
        if tag == "head":
            head_offset = offset
        records += tag.encode("latin-1")
        records += struct.pack(">III", _ttf_table_checksum(blob), offset,
                               len(blob))
        body += blob
        pad = -len(blob) % 4
        body += b"\0" * pad
        offset += len(blob) + pad
    font_bytes = bytearray(header + bytes(records) + bytes(body))
    adjustment = (0xB1B0AFBA - _ttf_table_checksum(bytes(font_bytes))) \
        & 0xFFFFFFFF
    struct.pack_into(">I", font_bytes, head_offset + 8, adjustment)
    return bytes(font_bytes), {cp: gid_map[gid] for cp, gid in used.items()}


def _tounicode_cmap(mapping: dict[int, int]) -> bytes:
    """A /ToUnicode CMap: new glyph id (CID) -> unicode, from codepoint
    -> new gid."""
    entries = sorted((gid, cp) for cp, gid in mapping.items())
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) "
        "/Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    for i in range(0, len(entries), 100):
        chunk = entries[i:i + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for gid, cp in chunk:
            if cp <= 0xFFFF:
                uni = f"{cp:04X}"
            else:  # surrogate pair
                v = cp - 0x10000
                uni = f"{0xD800 + (v >> 10):04X}{0xDC00 + (v & 0x3FF):04X}"
            lines.append(f"<{gid:04X}> <{uni}>")
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop",
              "end", "end"]
    return "\n".join(lines).encode("ascii")


def _embed_wrap(lines_text: list[tuple[str, str]], font: _TtfFont,
                *, size: int, max_width: float,
                cmap: dict[int, int]) -> list[list[tuple[str, str]]]:
    """Wrap (kind, text) blocks into pages of (kind, line) using real
    glyph advances (points), not character counts."""
    max_lines = 54

    def text_width(text: str, size: int) -> float:
        total = 0.0
        for ch in text:
            gid = cmap.get(ord(ch))
            if gid is not None:
                total += font.advance(gid)
        return total * size / font.units_per_em

    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for kind, text in lines_text:
        eff_size = size + 6 if kind == _BLOCK_H1 else \
            size + 3 if kind == _BLOCK_H2 else \
            size if kind != _BLOCK_H3 else size
        wrapped: list[str] = []
        for raw_line in text.split("\n"):
            words = raw_line.split(" ")
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if not line or text_width(candidate, eff_size) <= max_width:
                    line = candidate
                else:
                    wrapped.append(line)
                    while text_width(word, eff_size) > max_width \
                            and len(word) > 1:
                        cut = len(word) // 2
                        while cut > 1 and text_width(
                                word[:cut], eff_size) > max_width:
                            cut //= 2
                        wrapped.append(word[:cut])
                        word = word[cut:]
                    line = word
            wrapped.append(line)
        for line in wrapped:
            if len(current) >= max_lines:
                pages.append(current)
                current = []
            current.append((kind, line))
    if current or not pages:
        pages.append(current)
    return pages


def render_pdf_embedded(
    text: str,
    *,
    font_data: bytes,
    title: str = "Report",
    page_size: str = "A4",
    font_size: int = 10,
    header: str = "",
    footer: str = "",
    headings: bool = False,
    metadata: dict | None = None,
    watermark: BrandMark | str | None = None,
) -> bytes:
    """Render ``text`` to PDF with a subsetted TrueType font embedded.

    The non-WinAnsi path: any character the font carries renders and
    extracts exactly (Identity-H CIDs + /ToUnicode).  Characters the
    font lacks fall back to ``?`` when the font has one, else are
    dropped — the same never-fail posture as the WinAnsi writer, but
    the fallback set is now the font's real coverage, not Latin-1.

    Layout is the plain/structured block model of :func:`render_pdf`
    with one face: headings scale the same font (no synthetic bold).

    ``watermark`` draws the designed brand watermark on EVERY page,
    under the content, with the same geometry, alpha and tiling as
    :func:`render_pdf` — a :class:`~nomorals.core.branding.BrandMark`
    (resolve one from a policy with
    :func:`~nomorals.core.branding.resolve_brand_mark`), or a plain
    string for its text.  The mark's glyphs join the subset, so the
    mark renders in the embedded face and extracts through the
    /ToUnicode CMap like any other text.  ``None`` (the default)
    draws no mark — producers of public-bound PDFs pass the
    resolved mark; personal documents simply don't.
    Raises :class:`PdfError` for a bad page size, empty text, or a
    font that is not subsettable TrueType.
    """
    if page_size not in {"A4", "Letter"}:
        raise PdfError(f"unknown page size {page_size!r}; use A4 or Letter")
    mark = _coerce_watermark(watermark)
    text = (text or "").strip()
    if not text:
        raise PdfError("refusing to render an empty PDF")
    font = _TtfFont(font_data)
    page_width, page_height = (595, 842) if page_size == "A4" else (612, 792)

    if headings:
        _title, blocks = _parse_blocks(text, strip_title_line=False)
    else:
        cleaned = [re.sub(r"\*\*(.+?)\*\*", r"\1", ln)
                   for ln in text.split("\n")]
        blocks = [(_BLOCK_BODY, ln) for ln in cleaned if ln.strip()]
    if not blocks:
        raise PdfError("refusing to render an empty PDF")

    # Header/footer defaults are settled BEFORE subsetting: the subset
    # must contain every glyph the finished pages will draw, including
    # the timestamp and page-number digits that only exist at render
    # time (hence the explicit digit run in the coverage text).
    if title and not header:
        header = title
    now = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    if not footer:
        footer = f"{title} — page N — {now}"

    # Which codepoints does the document actually use?  The
    # watermark's text joins the coverage set: its glyphs must
    # be in the subset or the mark could not be drawn (the
    # subset is built from exactly this set).
    used: dict[int, int] = {}
    all_text = ("\n".join(t for _k, t in blocks) + header + footer + title
                + "0123456789"
                + (mark.text if mark is not None else ""))
    for ch in all_text:
        cp = ord(ch)
        if cp in font.cmap:
            used.setdefault(cp, font.cmap[cp])
    subset, new_cmap = _ttf_subset(font, used)

    max_width = page_width - 112
    pages = _embed_wrap(blocks, font, size=font_size,
                        max_width=max_width, cmap=new_cmap)

    def glyphs_of(line: str) -> bytes:
        out = bytearray()
        for ch in line:
            gid = new_cmap.get(ord(ch))
            if gid is None and ch != " ":
                gid = new_cmap.get(ord("?"))
            if gid is None:
                continue
            out += struct.pack(">H", gid)
        return bytes(out)

    # /W widths keyed by NEW glyph id: recompute the subset's glyph
    # order the same way _ttf_subset built it (sorted kept set, with
    # composite components pulled in by closure), so new id -> the old
    # glyph whose advance belongs in the /W array.
    keep = {0} | set(used.values())
    queue = list(keep)
    while queue:
        g = queue.pop()
        for comp in _ttf_component_gids(font, g):
            if comp not in keep:
                keep.add(comp)
                queue.append(comp)
    widths = {new: font.width1000(old)
              for new, old in enumerate(sorted(keep))}

    # The watermark block (identical on every page): drawn FIRST
    # in each page's content stream, under the content, when a
    # mark is owed and the embedded face carries its glyphs.
    wm_ops = b""
    if mark is not None:
        wm_ops = _watermark_ops_embedded(
            mark, glyphs_of(mark.text), page_width=page_width,
            page_height=page_height)

    slots: dict[int, bytes] = {}
    n_pages = len(pages)
    # Object plan: 1 catalog, 2 pages, 3 Type0 font, 4 CIDFont, 5
    # FontDescriptor, 6 FontFile2, 7 ToUnicode, then per page:
    # page + content, then the info object.  A drawn watermark's
    # ExtGState takes the first post-font slot and the pages
    # shift by one — an unmarked document's numbering is exactly
    # what it has always been.
    base = 8
    extgs_id = base
    page_base = base + 1 if wm_ops else base
    page_obj_ids = [page_base + 2 * i for i in range(n_pages)]
    content_obj_ids = [page_base + 1 + 2 * i
                       for i in range(n_pages)]
    info_id = page_base + 2 * n_pages
    extgs_entry = (f" /ExtGState << /GSW {extgs_id} 0 R >>"
                   if wm_ops else "")

    left = 56
    top = page_height - 56
    for i, page in enumerate(pages):
        ops: list[bytes] = []
        if wm_ops:
            ops.append(wm_ops)
        y = top
        if header:
            ops.append(
                b"BT /F1 8 Tf 1 0 0 1 %d %d Tm <%s> Tj ET"
                % (left, top + 16, glyphs_of(header).hex().encode()))
            y = top
        for kind, line in page:
            size = font_size + 6 if kind == _BLOCK_H1 else \
                font_size + 3 if kind == _BLOCK_H2 else font_size
            if line.strip():
                ops.append(
                    b"BT /F1 %d Tf 1 0 0 1 %d %d Tm <%s> Tj ET"
                    % (size, left, y, glyphs_of(line).hex().encode()))
            y -= size + 4
        if footer:
            foot = footer.replace("N", str(i + 1)).replace(
                "{total}", str(n_pages))
            ops.append(
                b"BT /F1 8 Tf 1 0 0 1 %d 40 Tm <%s> Tj ET"
                % (left, glyphs_of(foot).hex().encode()))
        stream = b"\n".join(ops)
        compressed = zlib.compress(stream)
        slots[content_obj_ids[i]] = (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n"
            "stream\n".encode() + compressed + b"\nendstream")
        slots[page_obj_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox "
            f"[0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >>{extgs_entry} >> "
            f"/Contents {content_obj_ids[i]} 0 R >>".encode())

    w_runs: list[str] = []
    for gid in sorted(widths):
        w_runs.append(f"{gid} [{widths[gid]}]")
    x_min, y_min, x_max, y_max = [
        round(v * 1000 / font.units_per_em) for v in font.bbox]
    slots[3] = (
        f"<< /Type /Font /Subtype /Type0 /BaseFont /{font.name} "
        f"/Encoding /Identity-H /DescendantFonts [4 0 R] "
        f"/ToUnicode 7 0 R >>").encode()
    slots[4] = (
        f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{font.name} "
        f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) "
        f"/Supplement 0 >> /FontDescriptor 5 0 R "
        f"/DW {widths.get(0, 500)} /W [{' '.join(w_runs)}] >>").encode()
    slots[5] = (
        f"<< /Type /FontDescriptor /FontName /{font.name} /Flags 4 "
        f"/FontBBox [{x_min} {y_min} {x_max} {y_max}] /ItalicAngle 0 "
        f"/Ascent {round(font.ascent * 1000 / font.units_per_em)} "
        f"/Descent {round(font.descent * 1000 / font.units_per_em)} "
        f"/CapHeight {round(font.cap_height * 1000 / font.units_per_em)} "
        f"/StemV 80 /FontFile2 6 0 R >>").encode()
    compressed_font = zlib.compress(subset)
    slots[6] = (
        f"<< /Length {len(compressed_font)} /Length1 {len(subset)} "
        f"/Filter /FlateDecode >>\nstream\n".encode()
        + compressed_font + b"\nendstream")
    touni = _tounicode_cmap(new_cmap)
    slots[7] = (
        f"<< /Length {len(touni)} >>\nstream\n".encode()
        + touni + b"\nendstream")
    if wm_ops:
        slots[extgs_id] = _extgstate_body(mark)
    catalog = "<< /Type /Catalog /Pages 2 0 R"
    if metadata:
        catalog += f" /Info {info_id} 0 R"
        slots[info_id] = _info_dict(metadata)
    slots[1] = (catalog + " >>").encode()
    children = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    slots[2] = f"<< /Type /Pages /Kids [{children}] /Count {n_pages} >>".encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(slots):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += slots[obj_id]
        out += b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(offsets)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(out)
