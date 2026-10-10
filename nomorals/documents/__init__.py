"""Universal document engine (L4): parse any document format into one model.

Supported formats: pdf (with /Info metadata + whitespace table recovery),
docx (lists, hyperlinks, footnotes, tables), xlsx/xls, pptx (stdlib
zipfile XML — no python-pptx needed), html (with meta harvesting),
markdown, csv/tsv (encoding fallback), txt, rtf (stdlib, incl. tables),
epub (stdlib), odt/ods (stdlib), json, xml.
Parse with :func:`parse_bytes` / :func:`parse_path`, sniff with
:func:`detect_format`, convert with the ``to_*`` helpers, search with
:class:`DocumentIndex`, OCR scanned PDFs with :func:`ocr_pdf`, diff
documents with :func:`compare_documents`, and summarize with
:func:`summarize`.
"""

from __future__ import annotations

from .compare import (DocumentComparison, compare_documents, diff_documents,
                      render_html, render_markdown, render_terminal,
                      similarity, word_diff)
from .convert import (to_csv, to_csv_all, to_docx, to_epub, to_html, to_json,
                      to_markdown, to_pdf, to_text)
from .errors import DocumentError
from .index import DocumentIndex
from .model import Document, Section, Table, full_text, new_document
from .ocr import (ocr_available, ocr_image, ocr_languages, ocr_pdf,
                  ocr_pdf_hocr, ocr_pdf_words)
from .parsers import detect_format, parse_bytes, parse_path
from .pdf_tables import extract_text_tables, score_table
from .summarize import (bullet_digest, keyphrases, keywords, summarize,
                        summarize_abstractive, summarize_query,
                        summarize_sections, summarize_text, tldr)

__all__ = [
    "Document",
    "DocumentComparison",
    "Section",
    "Table",
    "DocumentError",
    "DocumentIndex",
    "bullet_digest",
    "compare_documents",
    "detect_format",
    "diff_documents",
    "extract_text_tables",
    "full_text",
    "keyphrases",
    "keywords",
    "new_document",
    "ocr_available",
    "ocr_image",
    "ocr_languages",
    "ocr_pdf",
    "ocr_pdf_hocr",
    "ocr_pdf_words",
    "parse_bytes",
    "parse_path",
    "render_html",
    "render_markdown",
    "render_terminal",
    "score_table",
    "similarity",
    "summarize",
    "summarize_abstractive",
    "summarize_query",
    "summarize_sections",
    "summarize_text",
    "tldr",
    "to_csv",
    "to_csv_all",
    "to_docx",
    "to_epub",
    "to_html",
    "to_json",
    "to_markdown",
    "to_pdf",
    "to_text",
    "word_diff",
]
