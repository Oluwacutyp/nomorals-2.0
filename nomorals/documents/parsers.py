"""Parsers: bytes/path in any supported format -> :class:`Document`.

Format detection order: file extension first, then magic bytes
(PDF ``%PDF``, RTF ``{\\rtf``, ZIP ``PK`` for the OOXML/ODF/EPUB family
with content sniffing), then the explicit ``mime`` hint, then a plain-text
fallback.  Anything else raises :class:`DocumentError` — the engine never
returns a silently-empty Document.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..core.pdf import PdfError, read_pdf_text
from ..tools.browser import node_to_markdown, parse_html
from .errors import DocumentError
from .model import Document, Section, Table, new_document
from .pdf_tables import extract_text_tables

__all__ = ["parse_bytes", "parse_path"]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break parsing (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"

_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "txt",
    ".rtf": "rtf",
    ".epub": "epub",
    ".odt": "odt",
    ".ods": "ods",
}

_MIMES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "application/epub+zip": "epub",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "text/plain": "txt",
}

# OOXML content-type markers used to tell docx/xlsx/pptx apart inside a ZIP.
_OOXML_MARKERS = (
    ("wordprocessingml", "docx"),
    ("spreadsheetml", "xlsx"),
    ("presentationml", "pptx"),
)

# Fallback part probing when [Content_Types].xml is missing or unhelpful.
_OOXML_PARTS = (
    ("word/document.xml", "docx"),
    ("xl/workbook.xml", "xlsx"),
    ("ppt/presentation.xml", "pptx"),
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_HTML_START_RE = re.compile(r"(?is)^\s*(<!doctype\s+html|<html[\s>])")


def _stem(filename: str) -> str:
    name = Path(filename).name
    return name.rsplit(".", 1)[0] if "." in name else name


def _promote_title(doc: Document) -> None:
    """A leading h1 beats a filename-stem placeholder title."""
    if not doc.sections:
        return
    first = doc.sections[0]
    if first.level == 1 and first.heading:
        stem = _stem(doc.source)
        if not doc.title or doc.title == stem:
            doc.title = first.heading


def _sniff_zip(data: bytes) -> str:
    """Identify the ZIP-based flavour inside an archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            # OpenDocument (ODT/ODS): the spec mandates a leading "mimetype" entry.
            if "mimetype" in names:
                try:
                    mime = archive.read("mimetype").decode("ascii", "replace").strip()
                except Exception:  # noqa: BLE001 - fall through to other probes
                    mime = ""
                if mime == "application/vnd.oasis.opendocument.text":
                    return "odt"
                if mime == "application/vnd.oasis.opendocument.spreadsheet":
                    return "ods"
            # EPUB: META-INF/container.xml points at the OPF package file.
            if "META-INF/container.xml" in names:
                try:
                    container = archive.read("META-INF/container.xml").decode(
                        "utf-8", "replace")
                    if ".opf" in container:
                        return "epub"
                except Exception:  # noqa: BLE001 - fall through to other probes
                    pass
            try:
                types_xml = archive.read("[Content_Types].xml").decode("utf-8", "replace")
            except KeyError:
                types_xml = ""
            for marker, fmt in _OOXML_MARKERS:
                if marker in types_xml:
                    return fmt
            for part, fmt in _OOXML_PARTS:
                if part in names:
                    return fmt
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"not a readable ZIP archive: {exc}") from exc
    raise DocumentError(
        "ZIP archive is not a recognized document "
        "(docx/xlsx/pptx/odt/ods/epub)")


def _sniff(data: bytes, filename: str, mime: str) -> str:
    """Return the format key for ``data`` or raise DocumentError."""
    if not data:
        raise DocumentError("cannot parse empty input")
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in _EXTENSIONS:
        return _EXTENSIONS[ext]
    if ext:
        raise DocumentError(f"unsupported file extension {ext!r} for {filename!r}")
    if data[:4] == _PDF_MAGIC:
        return "pdf"
    if data[:4] == _ZIP_MAGIC:
        return _sniff_zip(data)
    if data.lstrip()[:5] == b"{\\rtf":
        return "rtf"
    if mime:
        fmt = _MIMES.get(mime.split(";")[0].strip().lower())
        if fmt:
            return fmt
        raise DocumentError(f"unsupported MIME type {mime!r}")
    text_start = data.lstrip()[:4096]
    if _HTML_START_RE.match(text_start.decode("utf-8", "replace")):
        return "html"
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DocumentError(
            "unrecognized binary input: not PDF, not an Office document, "
            "not decodable text") from None
    if "\x00" in decoded:
        raise DocumentError("unrecognized binary input: NUL bytes in text stream")
    return "txt"


# ── shared markdown sectioning ──────────────────────────────────────────────


def _parse_markdown_sections(text: str) -> list[Section]:
    """Split markdown-ish text into sections on ``#``..``######`` headings.

    Fenced code blocks pass through as body text — a ``#`` inside a fence
    is not a heading.
    """
    sections: list[Section] = []
    current = Section(level=1, heading="", text="")
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        text_out = "\n".join(body).strip("\n")
        if current.heading or text_out.strip():
            sections.append(Section(level=current.level, heading=current.heading,
                                    text=text_out.strip()))
        body.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        match = _HEADING_RE.match(stripped) if not in_fence else None
        if match:
            flush()
            current = Section(level=len(match.group(1)),
                              heading=match.group(2), text="")
        else:
            body.append(line)
    flush()
    if not sections:
        raise DocumentError("markdown document has no content")
    return sections


# ── per-format parsers ──────────────────────────────────────────────────────


_PDF_INFO_KEY_RE = re.compile(rb"/(Title|Author|Subject|Keywords|Creator|Producer|CreationDate|ModDate)")
_PDF_LITERAL_RE = re.compile(rb"\((?:\\.|[^()\\])*\)")
_PDF_HEX_RE = re.compile(rb"<([0-9A-Fa-f\s]+)>")


def _pdf_unescape_string(raw: bytes) -> str:
    """Decode a PDF literal string ``( ... )`` (escapes + octal codes)."""
    body = raw[1:-1]
    out: list[str] = []
    i, n = 0, len(body)
    simple = {ord("n"): "\n", ord("r"): "\r", ord("t"): "\t", ord("b"): "\b",
              ord("f"): "\f", ord("\\"): "\\", ord("("): "(", ord(")"): ")"}
    while i < n:
        byte = body[i]
        if byte != 0x5C or i + 1 >= n:  # not a backslash escape
            out.append(chr(byte))
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt in (0x0A, 0x0D):  # line continuation: drop
            i += 2
            if nxt == 0x0D and i < n and body[i] == 0x0A:
                i += 1
        elif 0x30 <= nxt <= 0x37:  # octal \ddd
            digits = body[i + 1:i + 4]
            count = 0
            for digit in digits:
                if 0x30 <= digit <= 0x37:
                    count += 1
                else:
                    break
            out.append(chr(int(digits[:count], 8) & 0xFF))
            i += 1 + count
        else:
            out.append(chr(nxt))
            i += 2
    return "".join(out)


def _pdf_info(data: bytes) -> dict[str, str]:
    """Extract the PDF /Info dictionary (title/author/dates), best effort.

    Returns {} when the trailer has no /Info entry or it cannot be read —
    metadata is a bonus, never a failure.
    """
    info_ref = re.search(rb"/Info\s+(\d+)\s+\d+\s+R", data)
    if not info_ref:
        return {}
    obj = re.search(rb"\b" + info_ref.group(1) + rb"\s+\d+\s+obj(.*?)endobj",
                    data, re.DOTALL)
    if not obj:
        return {}
    body = obj.group(1)
    out: dict[str, str] = {}
    for key_match in _PDF_INFO_KEY_RE.finditer(body):
        key = key_match.group(1).decode("ascii")
        rest = body[key_match.end():]
        literal = _PDF_LITERAL_RE.match(rest.lstrip())
        if literal is not None:
            out[key] = _pdf_unescape_string(literal.group(0))
            continue
        hexed = _PDF_HEX_RE.match(rest.lstrip())
        if hexed is not None:
            try:
                raw_hex = re.sub(rb"\s", b"", hexed.group(1))
                decoded = bytes.fromhex(raw_hex.decode("ascii"))
                if decoded[:2] == b"\xfe\xff":
                    out[key] = decoded[2:].decode("utf-16-be", "replace")
                else:
                    out[key] = decoded.decode("latin-1", "replace")
            except (ValueError, UnicodeDecodeError):
                # A corrupt hex Info value is best-effort metadata, not a
                # parse failure: skip the key and keep the document.
                _log.debug("unreadable hex PDF Info value for %s", key)
    for date_key in ("CreationDate", "ModDate"):
        raw = out.get(date_key, "")
        parsed = _pdf_date_to_iso(raw)
        if parsed:
            out[date_key] = parsed
    return out


def _pdf_date_to_iso(raw: str) -> str:
    """``D:20261003120000+02'00'`` -> ``2026-10-03T12:00:00``; '' on failure."""
    match = re.match(
        r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", raw.strip())
    if not match:
        return ""
    year, mon, day, hour, minute, second = match.groups()
    try:
        return (f"{year}-{mon or '01'}-{day or '01'}"
                f"T{hour or '00'}:{minute or '00'}:{second or '00'}")
    except (TypeError, ValueError):
        return ""


def _parse_pdf(data: bytes, doc: Document) -> Document:
    try:
        text = read_pdf_text(data)
    except PdfError as exc:
        raise DocumentError(f"invalid PDF: {exc}") from exc
    if not text.strip():
        raise DocumentError("PDF contains no extractable text (scanned/image-only?)")
    # read_pdf_text joins pages with "\n\n" and never emits a blank line
    # inside a page, so splitting there recovers the page boundaries.
    pages = [p for p in text.split("\n\n") if p.strip()]
    doc.sections = [Section(level=1, heading=f"Page {i + 1}", text=page)
                    for i, page in enumerate(pages)]
    # Recover whitespace-aligned tables from the page text (conservative:
    # only consistent columnar blocks become tables).
    for i, page in enumerate(pages):
        for table in extract_text_tables(page):
            table.name = f"Page {i + 1} {table.name}"
            doc.tables.append(table)
    doc.metadata["pages"] = len(pages)
    if doc.tables:
        doc.metadata["tables"] = len(doc.tables)
    # PDF document-information dictionary: title/author/dates.
    info = _pdf_info(data)
    if info.get("Title"):
        doc.title = info["Title"]
    if info.get("Author"):
        doc.author = info["Author"]
    for key in ("Subject", "Keywords", "Creator", "Producer",
                "CreationDate", "ModDate"):
        if info.get(key):
            doc.metadata[key.lower()] = info[key]
    return doc


def _parse_docx(data: bytes, doc: Document) -> Document:
    try:
        from docx import Document as DocxDocument
        from docx.table import Table as DocxTable
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise DocumentError(
            "parsing .docx requires the optional 'python-docx' package "
            "(pip install python-docx)") from exc

    try:
        oxml_doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"invalid .docx file: {exc}") from exc
    props = oxml_doc.core_properties
    if props.title:
        doc.title = props.title
    if props.author:
        doc.author = props.author
    for attr, key in (("keywords", "keywords"), ("comments", "comments"),
                      ("description", "description"),
                      ("last_modified_by", "last_modified_by")):
        value = getattr(props, attr, None)
        if value:
            doc.metadata[key] = str(value)
    for attr, key in (("created", "created"), ("modified", "modified")):
        value = getattr(props, attr, None)
        if value is not None and hasattr(value, "isoformat"):
            doc.metadata[key] = value.isoformat()

    word_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    sections: list[Section] = []
    current = Section(level=1, heading="", text="")
    body: list[str] = []
    table_count = 0

    def flush_section() -> None:
        text_out = "\n".join(body).strip()
        if current.heading or text_out:
            sections.append(Section(level=current.level, heading=current.heading,
                                    text=text_out))
        body.clear()

    for child in oxml_doc.element.body.iterchildren():
        if child.tag == f"{word_ns}p":
            para = Paragraph(child, oxml_doc)
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style is not None else ""
            if style.startswith("Heading"):
                flush_section()
                try:
                    level = int(style.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    level = 1
                current = Section(level=min(max(level, 1), 6), heading=text, text="")
            else:
                body.append(text)
        elif child.tag == f"{word_ns}tbl":
            table_count += 1
            dtable = DocxTable(child, oxml_doc)
            cells = [[c.text.strip() for c in row.cells] for row in dtable.rows]
            cells = [r for r in cells if any(r)]
            if not cells:
                continue
            name = f"Table {table_count}"
            doc.tables.append(Table(name=name, headers=cells[0], rows=cells[1:]))
            body.append(f"[{name}: {len(cells) - 1} data rows]")
    flush_section()
    if not sections and not doc.tables:
        raise DocumentError(".docx contains no paragraphs or tables")
    doc.sections = sections
    doc.metadata["tables"] = len(doc.tables)
    return doc


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - fall back to str() for odd date-likes
            pass
    return str(value).strip()


def _parse_xlsx(data: bytes, doc: Document) -> Document:
    try:
        import openpyxl
    except ImportError as exc:
        raise DocumentError(
            "parsing .xlsx requires the optional 'openpyxl' package "
            "(pip install openpyxl)") from exc

    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True,
                                          data_only=True)
    except Exception as exc:
        raise DocumentError(f"invalid .xlsx file: {exc}") from exc
    props = workbook.properties
    if getattr(props, "title", None):
        doc.title = props.title
    if getattr(props, "creator", None):
        doc.author = props.creator
    for attr, key in (("keywords", "keywords"), ("description", "description"),
                      ("lastModifiedBy", "last_modified_by")):
        value = getattr(props, attr, None)
        if value:
            doc.metadata[key] = str(value)
    for attr, key in (("created", "created"), ("modified", "modified")):
        value = getattr(props, attr, None)
        if value is not None and hasattr(value, "isoformat"):
            doc.metadata[key] = value.isoformat()

    for sheet in workbook.worksheets:
        grid = [[_cell_text(cell.value) for cell in row]
                for row in sheet.iter_rows()]
        grid = [row for row in grid if any(cell for cell in row)]
        if not grid:
            continue
        width = max(len(row) for row in grid)
        padded = [row + [""] * (width - len(row)) for row in grid]
        headers, rows = padded[0], padded[1:]
        name = sheet.title or f"Sheet {len(doc.tables) + 1}"
        doc.tables.append(Table(name=name, headers=headers, rows=rows))
        doc.sections.append(Section(
            level=1, heading=name,
            text=(f"Worksheet {name!r}: {len(rows)} data rows × "
                  f"{len(headers)} columns.")))
    workbook.close()
    if not doc.tables:
        raise DocumentError(".xlsx workbook has no non-empty worksheets")
    doc.metadata["sheets"] = [t.name for t in doc.tables]
    return doc


_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_A = f"{{{_DRAWING_NS}}}"


def _parse_pptx(data: bytes, doc: Document) -> Document:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"invalid .pptx file: {exc}") from exc
    with archive:
        numbered: list[tuple[int, str]] = []
        for entry in archive.namelist():
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", entry)
            if match:
                numbered.append((int(match.group(1)), entry))
        if not numbered:
            raise DocumentError(".pptx contains no slides")
        numbered.sort()

        for index, (_num, name) in enumerate(numbered, 1):
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError as exc:
                raise DocumentError(f"malformed slide XML in {name}: {exc}") from exc
            # a:t runs that live inside a:tbl cells are rendered as table
            # rows instead, so collect their element ids to skip them below.
            table_text_ids: set[int] = set()
            table_lines: list[str] = []
            for tbl in root.iter(f"{_A}tbl"):
                for t_elem in tbl.iter(f"{_A}t"):
                    table_text_ids.add(id(t_elem))
                for tr in tbl.findall(f"{_A}tr"):
                    cells = []
                    for tc in tr.findall(f"{_A}tc"):
                        cell = " ".join(t.text or "" for t in tc.iter(f"{_A}t"))
                        cells.append(cell.strip())
                    if any(cells):
                        table_lines.append(" | ".join(cells))
            lines = [(elem.text or "").strip()
                     for elem in root.iter(f"{_A}t")
                     if id(elem) not in table_text_ids]
            lines = [line for line in lines if line]
            lines.extend(table_lines)
            if not lines:
                continue
            doc.sections.append(Section(level=1, heading=f"Slide {index}",
                                        text="\n".join(lines)))
    if not doc.sections:
        raise DocumentError(".pptx slides contain no text")
    doc.metadata["slides"] = len(doc.sections)
    return doc


def _parse_html(data: bytes, doc: Document) -> Document:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"HTML is not valid UTF-8: {exc}") from exc
    root = parse_html(text)
    titles = root.find_all("title")
    if titles and titles[0].inner_text().strip():
        doc.title = titles[0].inner_text().strip()
    # <head> is metadata: drop it so its text can't glue itself to the
    # first body heading (node_to_markdown strips inner block edges).
    for head in root.find_all("head"):
        parent = head.parent
        if parent is not None and head in parent.children:
            parent.children.remove(head)
    bodies = root.find_all("body")
    markdown = node_to_markdown(bodies[0] if bodies else root)
    if not markdown.strip():
        raise DocumentError("HTML document has no readable content")
    doc.sections = _parse_markdown_sections(markdown)
    _promote_title(doc)
    for i, table_node in enumerate(root.find_all("table"), 1):
        grid: list[list[str]] = []
        for tr in table_node.find_all("tr"):
            cells = [c.inner_text().strip()
                     for c in tr.find_all(("th", "td"))]
            if any(cells):
                grid.append(cells)
        if not grid:
            continue
        width = max(len(row) for row in grid)
        grid = [row + [""] * (width - len(row)) for row in grid]
        doc.tables.append(Table(name=f"Table {i}", headers=grid[0], rows=grid[1:]))
    return doc


def _parse_markdown(data: bytes, doc: Document) -> Document:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"markdown is not valid UTF-8: {exc}") from exc
    if not text.strip():
        raise DocumentError("markdown document is empty")
    doc.sections = _parse_markdown_sections(text)
    _promote_title(doc)
    return doc


def _parse_delimited(data: bytes, doc: Document, delimiter: str, label: str) -> Document:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"{label} is not valid UTF-8: {exc}") from exc
    if not text.strip():
        raise DocumentError(f"{label} file is empty")
    if delimiter == ",":
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    grid = [[cell.strip() for cell in row] for row in reader]
    grid = [row for row in grid if any(cell for cell in row)]
    if not grid:
        raise DocumentError(f"{label} file has no data rows")
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    name = _stem(doc.source) or "data"
    doc.tables.append(Table(name=name, headers=grid[0], rows=grid[1:]))
    doc.sections.append(Section(
        level=1, heading=name,
        text=f"{label.upper()} table {name!r}: {len(grid) - 1} data rows × "
             f"{len(grid[0])} columns."))
    doc.metadata["delimiter"] = delimiter
    return doc


def _parse_txt(data: bytes, doc: Document) -> Document:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"text is not valid UTF-8: {exc}") from exc
    if not text.strip():
        raise DocumentError("text document is empty")
    doc.sections = [Section(level=1, heading="", text=text.strip())]
    return doc


# ── RTF (stdlib-only) ───────────────────────────────────────────────────────

#: Destination groups whose content is formatting, not prose.
_RTF_SKIP_DESTINATIONS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "info", "themedata", "listtable",
    "listoverridetable", "rsidtbl", "generator", "colorschememapping",
    "datastore", "header", "footer", "footnote", "annotation", "pict",
    "shppict", "pnseclvl", "themedata", "latentstyles",
})

#: One-shot control words that map to literal characters.
_RTF_SPECIAL = {
    "par": "\n", "line": "\n", "tab": "\t", "emdash": "\u2014",
    "endash": "\u2013", "bullet": "\u2022", "lquote": "\u2018",
    "rquote": "\u2019", "ldblquote": "\u201c", "rdblquote": "\u201d",
}

_RTF_WORD_RE = re.compile(r"[a-z]+")
_RTF_NUM_RE = re.compile(r"-?\d+")


def _rtf_to_text(data: bytes) -> str:
    """Extract plain text from RTF markup.

    Handles groups (with ignorable ``{\\*...}`` destinations and known
    formatting destinations skipped), control words, ``\\uN`` Unicode
    escapes with their fallback character, and ``\\'hh`` hex escapes.
    """
    src = data.decode("latin-1", "replace")  # 1:1 bytes; \'hh handled below
    out: list[str] = []
    skip_stack: list[bool] = []  # skip-state of each open group
    skip = False
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == "{":
            skip_stack.append(skip)
            i += 1
            continue
        if ch == "}":
            skip = skip_stack.pop() if skip_stack else False
            i += 1
            continue
        if ch != "\\":
            # Raw newlines in RTF source are formatting, not content.
            if not skip and ch not in "\r\n":
                out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        esc = src[i]
        if esc in "\\{}":
            if not skip:
                out.append(esc)
            i += 1
            continue
        if esc == "'":
            hexpair = src[i + 1:i + 3]
            try:
                code = int(hexpair, 16)
            except ValueError:
                code = 0x3F  # '?'
            if not skip:
                out.append(chr(code))
            i += 3
            continue
        if esc == "*":  # ignorable destination group
            skip = True
            i += 1
            continue
        if esc == "~":
            if not skip:
                out.append("\u00a0")
            i += 1
            continue
        if esc in "-_":
            # \- optional hyphen (dropped); \_ non-breaking hyphen
            if not skip and esc == "_":
                out.append("-")
            i += 1
            continue
        word_match = _RTF_WORD_RE.match(src, i)
        if not word_match:
            i += 1
            continue
        word = word_match.group(0)
        i = word_match.end()
        num: int | None = None
        num_match = _RTF_NUM_RE.match(src, i)
        if num_match:
            num = int(num_match.group(0))
            i = num_match.end()
        if i < n and src[i] == " ":  # control-word delimiter space
            i += 1
        if skip:
            continue
        if word in _RTF_SKIP_DESTINATIONS:
            skip = True
            continue
        if word == "u" and num is not None:
            out.append(chr(num + 65536 if num < 0 else num))
            # Skip the single fallback character (a char or \'hh).
            if i < n:
                if src[i] == "\\" and i + 3 < n and src[i + 1] == "'":
                    i += 4
                else:
                    i += 1
            continue
        special = _RTF_SPECIAL.get(word)
        if special is not None:
            out.append(special)
    return "".join(out)


def _parse_rtf(data: bytes, doc: Document) -> Document:
    text = _rtf_to_text(data)
    blocks = [b.strip() for b in re.split(r"\n{2,}|\r\n{2,}", text) if b.strip()]
    # Single-newline paragraphs collapse into blocks; stray lone lines that
    # look like headings are not promoted — RTF carries no heading info.
    if not blocks:
        raise DocumentError("RTF document contains no readable text")
    doc.sections = [Section(level=1, heading="", text=block)
                    for block in blocks]
    return doc


# ── EPUB (stdlib-only zip + XML) ────────────────────────────────────────────

_OPF_NS = "{http://www.idpf.org/2007/opf}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_EPUB_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xhtml_text(elem: Any) -> str:
    return "".join(elem.itertext()).strip()


def _parse_epub(data: bytes, doc: Document) -> Document:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"invalid .epub file: {exc}") from exc
    with archive:
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
        except (KeyError, ET.ParseError) as exc:
            raise DocumentError(f".epub has no readable container.xml: {exc}") from exc
        rootfile = container.find(f".//{_CONTAINER_NS}rootfile")
        opf_path = (rootfile.get("full-path", "")
                    if rootfile is not None else "")
        if not opf_path or opf_path not in archive.namelist():
            raise DocumentError(".epub container points to no OPF package file")
        try:
            opf = ET.fromstring(archive.read(opf_path))
        except ET.ParseError as exc:
            raise DocumentError(f"malformed OPF package file {opf_path}: {exc}") from exc
        title_node = opf.find(f".//{_OPF_NS}metadata/{_DC_NS}title")
        if title_node is not None and title_node.text and title_node.text.strip():
            doc.title = title_node.text.strip()
        creator_node = opf.find(f".//{_OPF_NS}metadata/{_DC_NS}creator")
        if (creator_node is not None and creator_node.text
                and creator_node.text.strip()):
            doc.author = creator_node.text.strip()
        manifest = {item.get("id", ""): item.get("href", "")
                    for item in opf.iter(f"{_OPF_NS}item")}
        opf_dir = str(Path(opf_path).parent)
        spine_hrefs: list[str] = []
        for ref in opf.iter(f"{_OPF_NS}itemref"):
            href = manifest.get(ref.get("idref", ""), "")
            if not href:
                continue
            full = href if opf_dir in ("", ".") else f"{opf_dir}/{href}"
            if full in archive.namelist() and \
                    full.lower().endswith((".xhtml", ".html", ".htm")):
                spine_hrefs.append(full)

        sections: list[Section] = []
        current = Section(level=1, heading="", text="")
        body: list[str] = []

        def flush() -> None:
            text_out = "\n".join(body).strip()
            if current.heading or text_out:
                sections.append(Section(level=current.level,
                                        heading=current.heading, text=text_out))
            body.clear()

        def flush_table(table_elem: Any) -> None:
            grid: list[list[str]] = []
            for tr in table_elem.iter():
                if _localname(tr.tag) != "tr":
                    continue
                cells = [_xhtml_text(c) for c in tr
                         if _localname(c.tag) in ("th", "td")]
                if any(cells):
                    grid.append(cells)
            if not grid:
                return
            width = max(len(row) for row in grid)
            grid = [row + [""] * (width - len(row)) for row in grid]
            doc.tables.append(Table(name=f"Table {len(doc.tables) + 1}",
                                    headers=grid[0], rows=grid[1:]))

        for href in spine_hrefs:
            try:
                chapter = ET.fromstring(archive.read(href))
            except ET.ParseError as exc:
                raise DocumentError(
                    f"malformed chapter {href} in .epub: {exc}") from exc
            bodies = [e for e in chapter.iter()
                      if _localname(e.tag) == "body"]
            scope = bodies[0] if bodies else chapter
            for elem in scope.iter():
                tag = _localname(elem.tag)
                if tag in _EPUB_HEADINGS:
                    flush()
                    current = Section(level=_EPUB_HEADINGS[tag],
                                      heading=_xhtml_text(elem), text="")
                elif tag == "p":
                    text = _xhtml_text(elem)
                    if text:
                        body.append(text)
                elif tag == "table":
                    flush_table(elem)
    flush()
    if not sections and not doc.tables:
        raise DocumentError(".epub contains no readable chapters")
    doc.sections = sections
    doc.metadata["chapters"] = len(spine_hrefs)
    if doc.tables:
        doc.metadata["tables"] = len(doc.tables)
    return doc


# ── ODT / ODS (stdlib-only zip + XML) ───────────────────────────────────────

_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DC_NS_URI = "http://purl.org/dc/elements/1.1/"

_ODT_OFFICE = f"{{{_OFFICE_NS}}}"
_ODT_TEXT = f"{{{_TEXT_NS}}}"
_ODT_TABLE = f"{{{_TABLE_NS}}}"
_ODT_DC = f"{{{_DC_NS_URI}}}"


def _odt_cell_text(cell: Any) -> str:
    return " ".join(cell.itertext()).strip()


def _odt_table_grid(table_elem: Any) -> list[list[str]]:
    grid: list[list[str]] = []
    for row in table_elem.iter(f"{_ODT_TABLE}table-row"):
        cells = [_odt_cell_text(c) for c in row
                 if _localname(c.tag) in ("table-cell", "covered-table-cell")]
        if any(cells):
            grid.append(cells)
    if not grid:
        return []
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


class _OdtFlow:
    """Accumulates ODT flow content (headings/paragraphs/tables) in order."""

    def __init__(self, doc: Document) -> None:
        self._doc = doc
        self.sections: list[Section] = []
        self._current = Section(level=1, heading="", text="")
        self._body: list[str] = []

    def flush(self) -> None:
        text_out = "\n".join(self._body).strip()
        if self._current.heading or text_out:
            self.sections.append(Section(level=self._current.level,
                                         heading=self._current.heading,
                                         text=text_out))
        self._body.clear()

    def heading(self, level: int, text: str) -> None:
        if not text:
            return
        self.flush()
        self._current = Section(level=min(max(level, 1), 6),
                                heading=text, text="")

    def para(self, text: str) -> None:
        if text:
            self._body.append(text)

    def table(self, grid: list[list[str]], name: str) -> None:
        if not grid:
            return
        self._doc.tables.append(
            Table(name=name, headers=grid[0], rows=grid[1:]))
        self._body.append(f"[{name}: {len(grid) - 1} data rows]")

    def walk(self, elem: Any) -> None:
        for child in elem:
            tag = _localname(child.tag)
            if tag == "h":
                try:
                    level = int(child.get(f"{_ODT_TEXT}outline-level", "1"))
                except (TypeError, ValueError):
                    level = 1
                self.heading(level, _odt_cell_text(child))
            elif tag == "p":
                self.para(_odt_cell_text(child))
            elif tag == "table":
                name = child.get(f"{_ODT_TABLE}name", "") or \
                    f"Table {len(self._doc.tables) + 1}"
                grid = _odt_table_grid(child)
                self.table(grid, name)
            elif tag in ("list", "list-item", "section", "table-of-content",
                         "index-body", "alphabetical-index", "toc"):
                self.walk(child)
            # Other elements (draw:frame, text:soft-page-break, …) carry no
            # flow text and are skipped on purpose.


def _parse_odt(data: bytes, doc: Document) -> Document:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"invalid .odt file: {exc}") from exc
    with archive:
        try:
            root = ET.fromstring(archive.read("content.xml"))
        except (KeyError, ET.ParseError) as exc:
            raise DocumentError(f".odt has no readable content.xml: {exc}") from exc
        try:
            meta_root = ET.fromstring(archive.read("meta.xml"))
            title_node = meta_root.find(f".//{_ODT_DC}title")
            if (title_node is not None and title_node.text
                    and title_node.text.strip()):
                doc.title = title_node.text.strip()
            creator_node = meta_root.find(f".//{_ODT_DC}creator")
            if (creator_node is not None and creator_node.text
                    and creator_node.text.strip()):
                doc.author = creator_node.text.strip()
        except (KeyError, ET.ParseError):
            # meta.xml is optional metadata; a missing/corrupt one must
            # never fail the document parse.
            _log.debug("odt meta.xml unreadable; continuing without it")
        body = root.find(f"{_ODT_OFFICE}body")
        if body is None:
            raise DocumentError(".odt content.xml has no office:body")
        text_root = body.find(f"{_ODT_OFFICE}text")
        if text_root is None:
            raise DocumentError(".odt has no office:text content")
        flow = _OdtFlow(doc)
        flow.walk(text_root)
        flow.flush()
    if not flow.sections and not doc.tables:
        raise DocumentError(".odt contains no paragraphs or tables")
    doc.sections = flow.sections
    if doc.tables:
        doc.metadata["tables"] = len(doc.tables)
    return doc


def _parse_ods(data: bytes, doc: Document) -> Document:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"invalid .ods file: {exc}") from exc
    with archive:
        try:
            root = ET.fromstring(archive.read("content.xml"))
        except (KeyError, ET.ParseError) as exc:
            raise DocumentError(f".ods has no readable content.xml: {exc}") from exc
        body = root.find(f"{_ODT_OFFICE}body")
        sheet_root = (body.find(f"{_ODT_OFFICE}spreadsheet")
                      if body is not None else None)
        if sheet_root is None:
            raise DocumentError(".ods has no office:spreadsheet content")
        for table_elem in sheet_root.iter(f"{_ODT_TABLE}table"):
            name = table_elem.get(f"{_ODT_TABLE}name", "") or \
                f"Sheet {len(doc.tables) + 1}"
            grid = _odt_table_grid(table_elem)
            if not grid:
                continue
            doc.tables.append(Table(name=name, headers=grid[0], rows=grid[1:]))
            doc.sections.append(Section(
                level=1, heading=name,
                text=(f"Worksheet {name!r}: {len(grid) - 1} data rows × "
                      f"{len(grid[0])} columns.")))
    if not doc.tables:
        raise DocumentError(".ods spreadsheet has no non-empty tables")
    doc.metadata["sheets"] = [t.name for t in doc.tables]
    return doc


_PARSERS: dict[str, Callable[..., Document]] = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "xlsx": _parse_xlsx,
    "pptx": _parse_pptx,
    "html": _parse_html,
    "markdown": _parse_markdown,
    "txt": _parse_txt,
    "rtf": _parse_rtf,
    "epub": _parse_epub,
    "odt": _parse_odt,
    "ods": _parse_ods,
}


def parse_bytes(data: bytes, *, filename: str = "", mime: str = "") -> Document:
    """Parse raw ``data`` into a Document.

    The format is detected from ``filename``'s extension first, then magic
    bytes, then ``mime``.  Raises :class:`DocumentError` on unknown or
    unparseable input — never returns an empty Document.
    """
    fmt = _sniff(data, filename, mime)
    doc = new_document(format=fmt, title=_stem(filename),
                       source=filename or mime or "")
    if fmt in ("csv", "tsv"):
        doc = _parse_delimited(data, doc, "\t" if fmt == "tsv" else ",", fmt)
    else:
        doc = _PARSERS[fmt](data, doc)
    _emit("document.parsed", {
        "doc_id": doc.id,
        "format": doc.format,
        "title": doc.title,
        "sections": len(doc.sections),
        "tables": len(doc.tables),
        "source": doc.source,
    })
    return doc


def parse_path(path: str | Path) -> Document:
    """Parse the file at ``path``.  Raises DocumentError when missing or bad."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DocumentError(f"no such file: {file_path}")
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise DocumentError(f"cannot read {file_path}: {exc}") from exc
    doc = parse_bytes(data, filename=file_path.name)
    doc.source = str(file_path)
    return doc
