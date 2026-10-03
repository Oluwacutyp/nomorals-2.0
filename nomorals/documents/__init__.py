"""Universal document engine (L4): parse any document format into one model.

Supported formats: pdf (with /Info metadata + whitespace table recovery),
docx, xlsx, pptx (stdlib zipfile XML — no python-pptx needed), html,
markdown, csv/tsv, txt, rtf (stdlib), epub (stdlib), odt/ods (stdlib).
Parse with :func:`parse_bytes` / :func:`parse_path`, convert with the
``to_*`` helpers, search with :class:`DocumentIndex`, OCR scanned PDFs
with :func:`ocr_pdf`, diff documents with :func:`compare_documents`,
and summarize with :func:`summarize`.
"""

from __future__ import annotations

from .compare import DocumentComparison, compare_documents, diff_documents
from .convert import to_csv, to_html, to_markdown, to_pdf, to_text
from .errors import DocumentError
from .index import DocumentIndex
from .model import Document, Section, Table, full_text
from .ocr import ocr_available, ocr_image, ocr_pdf
from .parsers import parse_bytes, parse_path
from .pdf_tables import extract_text_tables
from .summarize import keywords, summarize, summarize_text

__all__ = [
    "Document",
    "DocumentComparison",
    "Section",
    "Table",
    "DocumentError",
    "DocumentIndex",
    "compare_documents",
    "diff_documents",
    "extract_text_tables",
    "full_text",
    "keywords",
    "ocr_available",
    "ocr_image",
    "ocr_pdf",
    "parse_bytes",
    "parse_path",
    "summarize",
    "summarize_text",
    "to_markdown",
    "to_text",
    "to_html",
    "to_pdf",
    "to_csv",
]
