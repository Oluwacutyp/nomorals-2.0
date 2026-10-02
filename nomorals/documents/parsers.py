"""Parsers: bytes/path in any supported format -> :class:`Document`.

Format detection order: file extension first, then magic bytes
(PDF ``%PDF``, ZIP ``PK`` for the OOXML family with ``[Content_Types].xml``
sniffing), then the explicit ``mime`` hint, then a plain-text fallback.
Anything else raises :class:`DocumentError` — the engine never returns a
silently-empty Document.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

from ..core.pdf import PdfError, read_pdf_text
from ..tools.browser import node_to_markdown, parse_html
from .errors import DocumentError
from .model import Document, Section, Table, new_document

__all__ = ["parse_bytes", "parse_path"]

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
}

_MIMES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
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
    """Identify the OOXML flavour inside a ZIP archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
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
    raise DocumentError("ZIP archive is not a recognized Office document (docx/xlsx/pptx)")


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
    doc.metadata["pages"] = len(pages)
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


_PARSERS: dict[str, Callable[..., Document]] = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "xlsx": _parse_xlsx,
    "pptx": _parse_pptx,
    "html": _parse_html,
    "markdown": _parse_markdown,
    "txt": _parse_txt,
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
        return _parse_delimited(data, doc, "\t" if fmt == "tsv" else ",", fmt)
    parser = _PARSERS[fmt]
    return parser(data, doc)


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
