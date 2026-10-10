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

import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["PdfError", "PdfPage", "render_pdf", "pdf_write", "read_pdf_text",
           "pdf_read"]


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


def _content_stream(lines: list[str], *, page_width: int, page_height: int,
                    font_size: int = 10, header: str = "", footer: str = "") -> bytes:
    """Build one page's content stream (BT/Tf/Tm/Tj/ET)."""
    left = 56  # ~1cm margin
    top = page_height - 56
    leading = font_size + 4
    ops: list[str] = []
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
) -> bytes:
    """One page of styled lines: F3 (bold) for headings, F1 for body,
    F4 (Courier) for table rows."""
    left = 56
    top = page_height - 56
    leading = font_size + 4
    ops: list[str] = []
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
        )
        compressed = zlib.compress(stream)
        slots[content_obj_ids[i]] = (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n"
            f"stream\n".encode() + compressed + b"\nendstream"
        )
        slots[page_obj_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << {font_map} >> >> "
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
                     has_bold=has_bold, metadata=metadata)


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
    """
    if page_size not in {"A4", "Letter"}:
        raise PdfError(f"unknown page size {page_size!r}; use A4 or Letter")
    if headings:
        return _render_structured(
            text, title=title, page_size=page_size, font_size=font_size,
            header=header, footer=footer,
            chapter_break=chapter_break and headings, toc=toc,
            metadata=metadata,
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
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
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
