"""Converters: :class:`Document` -> markdown / text / HTML / PDF / CSV /
JSON / EPUB / DOCX.

``to_html`` renders a *designed* page, not a tag dump: four embedded
themes (light/dark/print/minimal, WeasyPrint-grade typography), an
auto-generated table of contents with anchor links, section numbering,
and responsive tables.  ``to_epub`` builds a real EPUB 3 file with
stdlib ``zipfile`` only; ``to_docx`` needs the optional ``python-docx``
package and fails fast with an install hint when it is missing.
``to_xlsx`` / ``to_pptx`` (Wave 2) live in
:mod:`nomorals.documents.writers` — stdlib ``zipfile`` like the EPUB
writer — and are re-exported here so the converter surface is one
place.
"""

from __future__ import annotations

import csv
import html as html_module
import io
import re
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from ..core.pdf import PdfError, render_pdf, render_pdf_embedded
from .errors import DocumentError
from .model import Document, Table
from .writers import to_pptx, to_xlsx

__all__ = ["to_csv", "to_csv_all", "to_docx", "to_epub", "to_html",
           "to_json", "to_markdown", "to_pdf", "to_pdf_embedded",
           "to_pptx", "to_text", "to_xlsx"]


def _heading_line(level: int, heading: str) -> str:
    level = min(max(int(level), 1), 6)
    clean = re.sub(r"\s+", " ", heading).strip()
    return f"{'#' * level} {clean}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _table_as_markdown(table: Table) -> str:
    def esc(cell: str) -> str:
        return re.sub(r"\s+", " ", cell).replace("|", "\\|").strip()

    # The grid is as wide as the widest row — a row with more cells
    # than the header must not lose its trailing cells, and a shorter
    # header row is padded.
    width = max([len(table.headers)] + [len(r) for r in table.rows])
    headers = list(table.headers) + [""] * (width - len(table.headers))
    lines = []
    if table.headers:
        lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table.rows:
        cells = [esc(c) for c in row]
        while len(cells) < width:
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _front_matter(doc: Document) -> str:
    lines = ["---"]
    if doc.title:
        lines.append(f'title: "{doc.title}"')
    if doc.author:
        lines.append(f'author: "{doc.author}"')
    if doc.format:
        lines.append(f"format: {doc.format}")
    if doc.source:
        lines.append(f'source: "{doc.source}"')
    if doc.created_at:
        lines.append("date: " + datetime.fromtimestamp(
            doc.created_at, tz=timezone.utc).strftime("%Y-%m-%d"))
    lines.append("---")
    return "\n".join(lines)


def _markdown_toc(doc: Document) -> str:
    lines = []
    for entry in doc.outline():
        indent = "  " * (entry["level"] - 1)
        anchor = _slugify(entry["heading"])
        lines.append(f"{indent}- [{entry['heading']}](#{anchor})")
    return "\n".join(lines)


def to_markdown(doc: Document, *, front_matter: bool = False,
                toc: bool = False) -> str:
    """Render a Document as markdown.

    ``front_matter=True`` prepends a YAML metadata block (pandoc-style);
    ``toc=True`` inserts a table of contents after the title.

    Round-trip guarantee: ``parse_bytes(to_markdown(doc).encode(),
    filename="x.md")`` preserves every section heading.
    """
    parts: list[str] = []
    if front_matter:
        parts.append(_front_matter(doc))
        parts.append("")
    if doc.title:
        clean_title = re.sub(r"\s+", " ", doc.title).strip()
        parts.append(f"# {clean_title}")
        parts.append("")
    if doc.author:
        parts.append(f"*Author: {doc.author}*")
        parts.append("")
    if toc and doc.outline():
        parts.append("## Contents")
        parts.append("")
        parts.append(_markdown_toc(doc))
        parts.append("")
    for section in doc.sections:
        if section.heading:
            parts.append(_heading_line(section.level, section.heading))
            parts.append("")
        if section.text and section.text.strip():
            parts.append(section.text.strip())
            parts.append("")
    for table in doc.tables:
        if table.name:
            parts.append(f"**{table.name}**")
            parts.append("")
        md_table = _table_as_markdown(table)
        if md_table:
            parts.append(md_table)
            parts.append("")
    out = "\n".join(parts).strip()
    if not out:
        raise DocumentError("document has no content to render as markdown",
                        kind="empty")
    return out + "\n"


def to_text(doc: Document) -> str:
    """Render a Document as plain text."""
    parts: list[str] = []
    if doc.title:
        parts.append(doc.title)
        parts.append("=" * min(len(doc.title), 72))
        parts.append("")
    if doc.author:
        parts.append(f"Author: {doc.author}")
        parts.append("")
    for section in doc.sections:
        if section.heading:
            parts.append(section.heading)
            parts.append("-" * min(len(section.heading), 72))
        if section.text and section.text.strip():
            parts.append(section.text.strip())
        parts.append("")
    for table in doc.tables:
        if table.name:
            parts.append(table.name)
        text = table.as_text()
        if text:
            parts.append(text)
        parts.append("")
    out = "\n".join(parts).strip()
    if not out:
        raise DocumentError("document has no content to render as text",
                        kind="empty")
    return out + "\n"


# ── HTML editions ───────────────────────────────────────────────────────────
#
# Since Doc 16 P3 the HTML producer is the shared branded edition
# renderer (core.edition) over the document's manifest — this module
# keeps the historical theme vocabulary and anchor scheme as the
# compatibility surface.  The per-theme CSS that used to live here
# moved into the renderer's shared tokens.

#: The themes to_html accepts (the historical vocabulary; the
#: renderer adds "sepia" for its own callers).
_HTML_THEMES = ("light", "dark", "print", "minimal")


def _edition_resolver(doc: Document):
    """Resolve manifest segments to edition content: a section's
    raw text (its heading rides the segment title, exactly as the
    historical renderer emitted it) and a table's pipe grid."""
    sections = {f"sec-{i + 1}": s for i, s in enumerate(doc.sections)}
    tables = {f"tbl-{i + 1}": t for i, t in enumerate(doc.tables)}

    def resolve(segment) -> str | None:
        if not segment.present:
            return None
        section = sections.get(segment.segment_id)
        if section is not None:
            return section.text
        table = tables.get(segment.segment_id)
        if table is not None:
            return table.as_text()
        return None

    return resolve


def to_html(doc: Document, *, theme: str = "light",
            toc: bool = True, brand_policy: str = "",
            brand_text: str = "") -> str:
    """Render a Document as a branded, self-contained HTML edition.

    ``theme`` is one of ``light`` (default), ``dark``, ``print``
    (serif, print CSS), or ``minimal`` — the historical vocabulary,
    validated here; the rendering itself is the shared edition
    renderer (core.edition) over the document's manifest, so a
    Document edition carries the Peacethefirst brand and the
    declared completeness/revision front matter like every other
    edition.  ``toc=False`` skips the contents list (and the
    heading anchors, as before).  Everything is escaped; headings
    keep their historical slug anchors.
    """
    if theme not in _HTML_THEMES:
        raise DocumentError(
            f"unknown HTML theme {theme!r}: choose from "
            f"{sorted(_HTML_THEMES)}", kind="invalid_argument")
    from ..core.edition import render_edition
    from .manifest import manifest_from_document

    manifest = manifest_from_document(doc)
    anchors: dict[str, str] = {}
    if toc:
        used: list[str] = []
        for i, section in enumerate(doc.sections):
            base = _slugify(section.heading) if section.heading \
                else "section"
            anchor, counter = base, 2
            while anchor in used:
                anchor = f"{base}-{counter}"
                counter += 1
            used.append(anchor)
            anchors[f"sec-{i + 1}"] = anchor
    return render_edition(
        manifest, _edition_resolver(doc), theme=theme, toc=toc,
        anchors=anchors, anchor_mode="compat",
        brand_policy=brand_policy, brand_text=brand_text)


def to_pdf(doc: Document) -> bytes:
    """Render a Document to PDF bytes via the pure-Python PDF writer.

    The PDF is a stamped derivative of the document's CURRENT
    revision (Doc 16 P5): the revision + declared completeness ride
    the /Info metadata (the owner-words line as subject, the full
    manifest revision_ref in keywords) and a front-matter line under
    the title, so a saved copy can never be mistaken for another
    revision of the document.  ``to_markdown`` itself is untouched —
    the stamp lands only in the copy rendered here."""
    markdown = to_markdown(doc)  # raises DocumentError when there is no content
    from ..core.branding import resolve_brand_mark
    from ..core.edition import completeness_line
    from .manifest import manifest_from_document

    manifest = manifest_from_document(doc)
    line = f"Revision {doc.revision} · {completeness_line(manifest)}"
    if "\n\n" in markdown:
        head, rest = markdown.split("\n\n", 1)
        markdown = f"{head}\n\n{line}\n\n{rest}"
    else:
        markdown = f"{line}\n\n{markdown}"
    try:
        return render_pdf(markdown, title=doc.title or "Document",
                          headings=True,
                          metadata={"title": doc.title or "Document",
                                    "author": doc.author or "",
                                    "subject": line.replace("—", "-"),
                                    "keywords": "nomorals-longdoc "
                                                + manifest.revision_ref},
                          # a document export is public-bound: the
                          # designed watermark rides every page,
                          # resolved from the manifest's brand policy.
                          watermark=resolve_brand_mark(
                              manifest.presentation.brand_policy))
    except PdfError as exc:
        raise DocumentError(f"PDF rendering failed: {exc}", kind="write") from exc


def to_pdf_embedded(doc: Document, font_path: str) -> bytes:
    """Render a Document to PDF with a TrueType font embedded (Wave 2).

    The WinAnsi writer transliterates or drops characters outside
    Latin-1; this path subsets and embeds the TTF at ``font_path`` so
    non-WinAnsi text (Greek, Cyrillic, CJK, symbols — whatever the
    font carries) renders and extracts exactly.  Raises
    :class:`DocumentError` (``kind="io"``) when the font file cannot
    be read, ``kind="write"`` when it is not subsettable TrueType.
    """
    from pathlib import Path

    from ..core.branding import resolve_brand_mark
    from .manifest import manifest_from_document

    markdown = to_markdown(doc)  # raises DocumentError when there is no content
    manifest = manifest_from_document(doc)
    try:
        font_data = Path(font_path).expanduser().read_bytes()
    except OSError as exc:
        raise DocumentError(
            f"cannot read font file {font_path}: {exc}", kind="io") from exc
    try:
        return render_pdf_embedded(
            markdown, font_data=font_data,
            title=doc.title or "Document", headings=True,
            # a document export is public-bound, exactly as in
            # to_pdf: the designed watermark rides every page,
            # resolved from the manifest's brand policy.
            watermark=resolve_brand_mark(
                manifest.presentation.brand_policy))
    except PdfError as exc:
        raise DocumentError(f"PDF rendering failed: {exc}", kind="write") from exc


def _resolve_table(doc: Document, table: int | str) -> Table:
    """Pick a table by index or (case-insensitive) name."""
    if not doc.tables:
        raise DocumentError("document has no tables to export as CSV",
                        kind="empty")
    if isinstance(table, str):
        for index, candidate in enumerate(doc.tables):
            if candidate.name.lower() == table.lower():
                return doc.tables[index]
        names = [t.name for t in doc.tables]
        raise DocumentError(
            f"no table named {table!r}: document has {names}",
            kind="invalid_argument")
    if table < 0 or table >= len(doc.tables):
        raise DocumentError(
            f"table index {table} out of range: document has "
            f"{len(doc.tables)} table(s)", kind="invalid_argument")
    return doc.tables[table]


def to_csv(doc: Document, table: int | str = 0) -> str:
    """Render one of the document's tables as CSV text.

    ``table`` is a 0-based index or a table name (case-insensitive).
    """
    chosen = _resolve_table(doc, table)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if chosen.headers:
        writer.writerow(chosen.headers)
    writer.writerows(chosen.rows)
    return buffer.getvalue()


def to_csv_all(doc: Document) -> dict[str, str]:
    """Render every table as CSV text, keyed by table name."""
    if not doc.tables:
        raise DocumentError("document has no tables to export as CSV",
                        kind="empty")
    out: dict[str, str] = {}
    for index, table in enumerate(doc.tables):
        name = table.name or f"Table {index + 1}"
        key = name
        counter = 2
        while key in out:
            key = f"{name} ({counter})"
            counter += 1
        out[key] = to_csv(doc, index)
    return out


def to_json(doc: Document, *, indent: int = 2) -> str:
    """Render a Document as JSON text (round-trips via
    ``Document.from_dict``)."""
    if indent < 0:
        raise DocumentError(f"indent must be >= 0, got {indent}",
                        kind="invalid_argument")
    return doc.to_json(indent=indent) + "\n"


def to_docx(doc: Document) -> bytes:
    """Render a Document as .docx bytes (optional ``python-docx``)."""
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise DocumentError(
            "rendering .docx requires the optional 'python-docx' package "
            "(pip install python-docx)", kind="missing_dependency",
            code="document.missing_dependency") from exc
    if not doc.title and not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render as docx",
                        kind="empty")
    oxml = DocxDocument()
    if doc.title:
        oxml.core_properties.title = doc.title
    if doc.author:
        oxml.core_properties.author = doc.author
    if doc.title:
        oxml.add_heading(doc.title, level=0)
    for section in doc.sections:
        if section.heading:
            oxml.add_heading(section.heading,
                             level=min(max(section.level, 1), 6))
        for para in section.text.split("\n"):
            para = para.strip()
            if not para:
                continue
            list_match = re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)(.*)$", para)
            if list_match and section.kind == "list":
                oxml.add_paragraph(list_match.group(1),
                                   style="List Bullet")
            else:
                oxml.add_paragraph(para)
    for table in doc.tables:
        if table.name:
            oxml.add_heading(table.name, level=2)
        grid = ([table.headers] if table.headers else []) + table.rows
        if not grid:
            continue
        width = max(len(row) for row in grid)
        oxml_table = oxml.add_table(rows=len(grid), cols=width)
        for i, row in enumerate(grid):
            for j in range(width):
                oxml_table.cell(i, j).text = row[j] if j < len(row) else ""
    buffer = io.BytesIO()
    oxml.save(buffer)
    return buffer.getvalue()


# ── EPUB (stdlib zip) ───────────────────────────────────────────────────────

_EPUB_CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0"
 xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
 <rootfiles><rootfile full-path="OEBPS/content.opf"
  media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""


def _xhtml_page(title: str, body_parts: list[str]) -> bytes:
    esc = html_module.escape
    doc = ET.Element("html", xmlns="http://www.w3.org/1999/xhtml")
    head = ET.SubElement(doc, "head")
    ET.SubElement(head, "title").text = title
    ET.SubElement(head, "meta", charset="utf-8")
    body = ET.SubElement(doc, "body")
    for kind, payload in body_parts:  # type: ignore[misc]
        if kind == "h":
            level, text = payload
            ET.SubElement(body, f"h{level}").text = text
        elif kind == "p":
            ET.SubElement(body, "p").text = payload
        elif kind == "table":
            table_elem = ET.SubElement(body, "table", border="1")
            headers, rows = payload
            if headers:
                tr = ET.SubElement(table_elem, "tr")
                for header in headers:
                    ET.SubElement(tr, "th").text = header
            for row in rows:
                tr = ET.SubElement(table_elem, "tr")
                for cell in row:
                    ET.SubElement(tr, "td").text = cell
    return ET.tostring(doc, encoding="utf-8", xml_declaration=True)


def to_epub(doc: Document, *, epub_title: str = "") -> bytes:
    """Render a Document as EPUB 3 bytes (stdlib ``zipfile`` only).

    One chapter per section, tables appended after the prose.
    """
    if not doc.title and not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render as epub",
                        kind="empty")
    title = epub_title or doc.title or "Document"
    chapters: list[tuple[str, bytes]] = []
    for i, section in enumerate(doc.sections):
        parts: list[tuple[str, object]] = []
        if section.heading:
            parts.append(("h", (min(max(section.level, 1), 6),
                                section.heading)))
        for para in section.text.split("\n"):
            para = para.strip()
            if para:
                parts.append(("p", para))
        chapters.append((f"chapter{i + 1}.xhtml",
                         _xhtml_page(section.heading or title, parts)))
    if doc.tables:
        table_parts: list[tuple[str, object]] = [("h", (1, "Tables"))]
        for table in doc.tables:
            if table.name:
                table_parts.append(("h", (2, table.name)))
            table_parts.append(("table", (table.headers, table.rows)))
        chapters.append(("tables.xhtml", _xhtml_page("Tables", table_parts)))
    if not chapters:
        chapters.append(("chapter1.xhtml", _xhtml_page(title, [])))

    # the derivative's truth (Doc 16 P5): the OPF records the
    # manifest revision_ref this EPUB was built from plus the
    # declared completeness, so a saved copy is provably tied to one
    # revision of the document.
    from .manifest import manifest_from_document

    manifest = manifest_from_document(doc)
    comp = manifest.completeness
    comp_total = (comp.declared_total if comp.declared_total is not None
                  else comp.present_count)
    stamp_meta = (
        '  <meta property="nomorals:revision_ref">'
        f'{html_module.escape(manifest.revision_ref)}</meta>\n'
        '  <meta property="nomorals:completeness">'
        f'{comp.status} {comp.present_count} of {comp_total}</meta>\n')

    opf_items = "\n".join(
        f'    <item id="ch{i}" href="{name}" '
        f'media-type="application/xhtml+xml"/>'
        for i, (name, _data) in enumerate(chapters))
    opf_spine = "\n".join(
        f'    <itemref idref="ch{i}"/>' for i in range(len(chapters)))
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" unique-identifier="uid"
 xmlns="http://www.idpf.org/2007/opf">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:identifier id="uid">urn:uuid:{doc.id}</dc:identifier>
  <dc:title>{html_module.escape(title)}</dc:title>
  <dc:creator>{html_module.escape(doc.author)}</dc:creator>
  <dc:language>en</dc:language>
{stamp_meta} </metadata>
 <manifest>
{opf_items}
 </manifest>
 <spine>
{opf_spine}
 </spine>
</package>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # The mimetype entry must be first and uncompressed (EPUB spec).
        archive.writestr("mimetype", "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", _EPUB_CONTAINER)
        archive.writestr("OEBPS/content.opf", opf)
        for name, data in chapters:
            archive.writestr(f"OEBPS/{name}", data)
    return buffer.getvalue()
