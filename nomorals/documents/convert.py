"""Converters: :class:`Document` -> markdown / text / HTML / PDF / CSV."""

from __future__ import annotations

import csv
import html as html_module
import io
import re

from ..core.pdf import PdfError, render_pdf
from .errors import DocumentError
from .model import Document, Table

__all__ = ["to_csv", "to_html", "to_markdown", "to_pdf", "to_text"]


def _heading_line(level: int, heading: str) -> str:
    level = min(max(int(level), 1), 6)
    clean = re.sub(r"\s+", " ", heading).strip()
    return f"{'#' * level} {clean}"


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


def to_markdown(doc: Document) -> str:
    """Render a Document as markdown.

    Round-trip guarantee: ``parse_bytes(to_markdown(doc).encode(),
    filename="x.md")`` preserves every section heading.
    """
    parts: list[str] = []
    if doc.title:
        clean_title = re.sub(r"\s+", " ", doc.title).strip()
        parts.append(f"# {clean_title}")
        parts.append("")
    if doc.author:
        parts.append(f"*Author: {doc.author}*")
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


def to_html(doc: Document) -> str:
    """Render a Document as a standalone HTML page (escaped properly)."""
    esc = html_module.escape
    parts = ["<!DOCTYPE html>", "<html>", "<head>",
             "<meta charset=\"utf-8\">",
             f"<title>{esc(doc.title)}</title>", "</head>", "<body>"]
    if doc.title:
        parts.append(f"<h1>{esc(doc.title)}</h1>")
    if doc.author:
        parts.append(f"<p><em>Author: {esc(doc.author)}</em></p>")
    for section in doc.sections:
        level = min(max(int(section.level), 1), 6)
        if section.heading:
            parts.append(f"<h{level}>{esc(section.heading)}</h{level}>")
        for para in section.text.split("\n"):
            para = para.strip()
            if para:
                parts.append(f"<p>{esc(para)}</p>")
    for table in doc.tables:
        if table.name:
            parts.append(f"<h2>{esc(table.name)}</h2>")
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


def to_csv(doc: Document, table: int = 0) -> str:
    """Render one of the document's tables as CSV text."""
    if not doc.tables:
        raise DocumentError("document has no tables to export as CSV")
    if table < 0 or table >= len(doc.tables):
        raise DocumentError(
            f"table index {table} out of range: document has "
            f"{len(doc.tables)} table(s)")
    chosen = doc.tables[table]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if chosen.headers:
        writer.writerow(chosen.headers)
    writer.writerows(chosen.rows)
    return buffer.getvalue()
