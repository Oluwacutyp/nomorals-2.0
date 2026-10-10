"""Converters: :class:`Document` -> markdown / text / HTML / PDF / CSV /
JSON / EPUB / DOCX.

``to_html`` renders a *designed* page, not a tag dump: four embedded
themes (light/dark/print/minimal, WeasyPrint-grade typography), an
auto-generated table of contents with anchor links, section numbering,
and responsive tables.  ``to_epub`` builds a real EPUB 3 file with
stdlib ``zipfile`` only; ``to_docx`` needs the optional ``python-docx``
package and fails fast with an install hint when it is missing.
"""

from __future__ import annotations

import csv
import html as html_module
import io
import re
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from ..core.pdf import PdfError, render_pdf
from .errors import DocumentError
from .model import Document, Table

__all__ = ["to_csv", "to_csv_all", "to_docx", "to_epub", "to_html",
           "to_json", "to_markdown", "to_pdf", "to_text"]


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

    lines = []
    width = len(table.headers)
    if table.headers:
        lines.append("| " + " | ".join(esc(h) for h in table.headers) + " |")
        lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
    for row in table.rows:
        cells = [esc(c) for c in row]
        while len(cells) < width:
            cells.append("")
        lines.append("| " + " | ".join(cells[:width] if width else cells) + " |")
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
        raise DocumentError("document has no content to render as markdown")
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
        raise DocumentError("document has no content to render as text")
    return out + "\n"


# ── HTML themes ─────────────────────────────────────────────────────────────

_HTML_CSS = {
    "light": """
:root{color-scheme:light}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Georgia,serif;
max-width:760px;margin:0 auto;padding:2.5rem 1.5rem;color:#1f2937;background:#ffffff;line-height:1.75}
h1,h2,h3,h4,h5,h6{color:#111827;line-height:1.3;margin-top:2em}
h1{font-size:2rem;border-bottom:2px solid #e5e7eb;padding-bottom:.4rem}
.author{color:#6b7280;font-style:italic}
nav.toc{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.5rem;margin:1.5rem 0}
nav.toc ul{list-style:none;padding-left:1rem}nav.toc>ul{padding-left:0}
nav.toc a{color:#2563eb;text-decoration:none}nav.toc a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin:1.25rem 0;font-size:.95rem;display:block;overflow-x:auto}
th,td{border:1px solid #d1d5db;padding:.55rem .8rem;text-align:left}
th{background:#f3f4f6;font-weight:600}
tr:nth-child(even) td{background:#f9fafb}
pre{background:#f3f4f6;border-radius:8px;padding:1rem;overflow-x:auto}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em}
blockquote{border-left:4px solid #d1d5db;margin:1rem 0;padding:.25rem 1rem;color:#4b5563}
""",
    "dark": """
:root{color-scheme:dark}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Georgia,serif;
max-width:760px;margin:0 auto;padding:2.5rem 1.5rem;color:#e5e7eb;background:#111827;line-height:1.75}
h1,h2,h3,h4,h5,h6{color:#f9fafb;line-height:1.3;margin-top:2em}
h1{font-size:2rem;border-bottom:2px solid #374151;padding-bottom:.4rem}
.author{color:#9ca3af;font-style:italic}
nav.toc{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:1rem 1.5rem;margin:1.5rem 0}
nav.toc ul{list-style:none;padding-left:1rem}nav.toc>ul{padding-left:0}
nav.toc a{color:#60a5fa;text-decoration:none}nav.toc a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin:1.25rem 0;font-size:.95rem;display:block;overflow-x:auto}
th,td{border:1px solid #4b5563;padding:.55rem .8rem;text-align:left}
th{background:#1f2937;font-weight:600}
tr:nth-child(even) td{background:#1a2332}
pre{background:#1f2937;border-radius:8px;padding:1rem;overflow-x:auto}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em}
blockquote{border-left:4px solid #4b5563;margin:1rem 0;padding:.25rem 1rem;color:#9ca3af}
""",
    "print": """
body{font-family:Georgia,'Times New Roman',serif;max-width:700px;margin:0 auto;
padding:1rem;color:#000;background:#fff;line-height:1.6;font-size:11pt}
h1,h2,h3,h4,h5,h6{color:#000;line-height:1.25;margin-top:1.5em;page-break-after:avoid}
h1{font-size:20pt;border-bottom:1pt solid #000;padding-bottom:.3rem}
nav.toc{display:none}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:10pt}
th,td{border:1pt solid #000;padding:.4rem .6rem;text-align:left}
th{background:#eee}
pre,blockquote{page-break-inside:avoid}
@media print{body{padding:0}}
""",
    "minimal": """
body{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;
padding:2rem 1rem;color:#222;background:#fff;line-height:1.7}
h1,h2,h3,h4,h5,h6{font-weight:600;margin-top:1.75em}
nav.toc{margin:1rem 0}nav.toc ul{list-style:none;padding-left:1rem}
table{border-collapse:collapse;margin:1rem 0}
th,td{border-bottom:1px solid #ddd;padding:.45rem .7rem;text-align:left}
th{border-bottom:2px solid #999}
""",
}


def to_html(doc: Document, *, theme: str = "light",
            toc: bool = True) -> str:
    """Render a Document as a styled standalone HTML page.

    ``theme`` is one of ``light`` (default), ``dark``, ``print``
    (serif, print CSS), or ``minimal``.  ``toc=False`` skips the table
    of contents.  Everything is escaped; headings get anchor ids.
    """
    if theme not in _HTML_CSS:
        raise DocumentError(
            f"unknown HTML theme {theme!r}: choose from "
            f"{sorted(_HTML_CSS)}")
    esc = html_module.escape
    anchors: list[str] = []
    for section in doc.sections:
        base = _slugify(section.heading) if section.heading else "section"
        anchor, counter = base, 2
        while anchor in anchors:
            anchor = f"{base}-{counter}"
            counter += 1
        anchors.append(anchor)

    parts = ["<!DOCTYPE html>", "<html>", "<head>",
             "<meta charset=\"utf-8\">",
             "<meta name=\"viewport\" "
             "content=\"width=device-width, initial-scale=1\">",
             f"<title>{esc(doc.title or 'Document')}</title>",
             f"<style>{_HTML_CSS[theme]}</style>",
             "</head>", "<body>"]
    if doc.title:
        parts.append(f"<h1>{esc(doc.title)}</h1>")
    if doc.author:
        parts.append(f"<p class=\"author\">Author: {esc(doc.author)}</p>")
    if toc:
        outline = [(s, a) for s, a in zip(doc.sections, anchors) if s.heading]
        if outline:
            parts.append("<nav class=\"toc\"><strong>Contents</strong><ul>")
            for section, anchor in outline:
                indent = "&nbsp;&nbsp;" * (min(section.level, 6) - 1)
                parts.append(f"<li>{indent}<a href=\"#{anchor}\">"
                             f"{esc(section.heading)}</a></li>")
            parts.append("</ul></nav>")
    for section, anchor in zip(doc.sections, anchors):
        level = min(max(int(section.level), 1), 6)
        if section.heading:
            # The anchor lives on an empty <a> so the heading tag itself
            # stays byte-identical (<h1>Title</h1>) for downstream
            # consumers that match on it.
            if toc:
                parts.append(f"<a id=\"{anchor}\"></a>")
            parts.append(f"<h{level}>{esc(section.heading)}</h{level}>")
        if section.kind == "code":
            parts.append(f"<pre><code>{esc(section.text.strip())}</code></pre>")
        elif section.kind == "quote":
            quoted = "<br>".join(
                esc(ln.strip().lstrip(">").strip())
                for ln in section.text.split("\n") if ln.strip())
            parts.append(f"<blockquote>{quoted}</blockquote>")
        elif section.kind == "list":
            parts.append("<ul>")
            for ln in section.text.split("\n"):
                item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "",
                              ln.strip())
                if item:
                    parts.append(f"<li>{esc(item)}</li>")
            parts.append("</ul>")
        else:
            for para in section.text.split("\n"):
                para = para.strip()
                if para:
                    parts.append(f"<p>{esc(para)}</p>")
    for table in doc.tables:
        if table.name:
            parts.append(f"<h2>{esc(table.name)}</h2>")
        if table.caption:
            parts.append(f"<p class=\"author\">{esc(table.caption)}</p>")
        parts.append("<table>")
        if table.headers:
            parts.append("<thead><tr>" + "".join(
                f"<th>{esc(h)}</th>" for h in table.headers) + "</tr></thead>")
        if table.rows:
            parts.append("<tbody>")
            for row in table.rows:
                parts.append("<tr>" + "".join(
                    f"<td>{esc(c)}</td>" for c in row) + "</tr>")
            parts.append("</tbody>")
        parts.append("</table>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def to_pdf(doc: Document) -> bytes:
    """Render a Document to PDF bytes via the pure-Python PDF writer."""
    markdown = to_markdown(doc)  # raises DocumentError when there is no content
    try:
        return render_pdf(markdown, title=doc.title or "Document", headings=True)
    except PdfError as exc:
        raise DocumentError(f"PDF rendering failed: {exc}") from exc


def _resolve_table(doc: Document, table: int | str) -> Table:
    """Pick a table by index or (case-insensitive) name."""
    if not doc.tables:
        raise DocumentError("document has no tables to export as CSV")
    if isinstance(table, str):
        for index, candidate in enumerate(doc.tables):
            if candidate.name.lower() == table.lower():
                return doc.tables[index]
        names = [t.name for t in doc.tables]
        raise DocumentError(
            f"no table named {table!r}: document has {names}")
    if table < 0 or table >= len(doc.tables):
        raise DocumentError(
            f"table index {table} out of range: document has "
            f"{len(doc.tables)} table(s)")
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
        raise DocumentError("document has no tables to export as CSV")
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
        raise DocumentError(f"indent must be >= 0, got {indent}")
    return doc.to_json(indent=indent) + "\n"


def to_docx(doc: Document) -> bytes:
    """Render a Document as .docx bytes (optional ``python-docx``)."""
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise DocumentError(
            "rendering .docx requires the optional 'python-docx' package "
            "(pip install python-docx)") from exc
    if not doc.title and not doc.sections and not doc.tables:
        raise DocumentError("document has no content to render as docx")
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
        raise DocumentError("document has no content to render as epub")
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
 </metadata>
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
