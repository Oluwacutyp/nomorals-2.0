"""Universal document engine (L4): parse any document format into one model.

Supported formats: pdf, docx, xlsx, pptx (stdlib zipfile XML — no
python-pptx needed), html, markdown, csv/tsv, txt.  Parse with
:func:`parse_bytes` / :func:`parse_path`, convert with the ``to_*``
helpers, search with :class:`DocumentIndex`.
"""

from __future__ import annotations

from .convert import to_csv, to_html, to_markdown, to_pdf, to_text
from .errors import DocumentError
from .index import DocumentIndex
from .model import Document, Section, Table, full_text
from .parsers import parse_bytes, parse_path

__all__ = [
    "Document",
    "Section",
    "Table",
    "DocumentError",
    "DocumentIndex",
    "parse_bytes",
    "parse_path",
    "full_text",
    "to_markdown",
    "to_text",
    "to_html",
    "to_pdf",
    "to_csv",
]
