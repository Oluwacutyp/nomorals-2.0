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
:func:`summarize`.  Wave 2 adds the XLSX/PPTX writers
(:func:`to_xlsx` / :func:`to_pptx`, stdlib) and moves
:class:`DocumentError` into the ``NoMoralsError`` code system
(``code`` + ``kind`` on every raise; see ``documents/PARITY.md`` for
the measured fidelity rows).  The PdfCraft port adds the PDF object
model (:class:`PdfDocument`, :class:`PdfEditor` — incremental editing,
machine-checked parity TOML, SHA-256 attribution DB; see
``documents/pdfcraft/``).
"""

from __future__ import annotations

from .compare import (DocumentComparison, compare_documents, diff_documents,
                      render_html, render_markdown, render_terminal,
                      similarity, word_diff)
from .convert import (to_csv, to_csv_all, to_docx, to_epub, to_html, to_json,
                      to_markdown, to_pdf, to_pdf_embedded, to_pptx, to_text,
                      to_xlsx)
from .errors import DOCUMENT_ERROR_KINDS, DocumentError
from .index import DocumentIndex
from .model import Document, Section, Table, full_text, new_document
from .ocr import (ocr_available, ocr_image, ocr_languages, ocr_pdf,
                  ocr_pdf_hocr, ocr_pdf_words)
from .pagewise import (
    ArchiveOp,
    CompressOp,
    CropOp,
    DeleteOp,
    MergeOp,
    PagewiseJob,
    PagewiseResult,
    ProtectOp,
    RepairOp,
    ReorderOp,
    RotateOp,
    ScanOp,
    SelectOp,
    SignOp,
    StampOp,
    UnlockOp,
    archive_pdf,
    extract_page_texts,
    format_pagewise_reply,
    pdf_info,
    protect_pdf,
    repair_pdf,
    run_pagewise,
    scan_to_pdf,
    unlock_pdf,
)
from .parsers import (artifact_body_text, detect_format, extract_text,
                      parse_bytes, parse_path)
from .pdf_tables import extract_text_tables, score_table
from .pdfcraft import (
    PdfDocument,
    PdfEditor,
    run_parity as run_pdfcraft_parity,
    verify_attribution as verify_pdfcraft_attribution,
)
from .requirements import (
    Requirement,
    RequirementAnalysis,
    RequirementGraph,
    RequirementLink,
    RequirementStore,
    analyze_corpus,
    analyze_document,
    extract_requirements,
    format_requirements_reply,
    open_requirements_store,
    render_report_markdown,
    score_links,
    verify_spans,
    write_report,
)
from .summarize import (bullet_digest, keyphrases, keywords, summarize,
                        summarize_abstractive, summarize_query,
                        summarize_sections, summarize_text, tldr)

__all__ = [
    "DOCUMENT_ERROR_KINDS",
    "Document",
    "DocumentComparison",
    "Requirement",
    "RequirementAnalysis",
    "RequirementGraph",
    "RequirementLink",
    "RequirementStore",
    "Section",
    "Table",
    "DocumentError",
    "DocumentIndex",
    "PdfDocument",
    "PdfEditor",
    "ArchiveOp",
    "CompressOp",
    "CropOp",
    "DeleteOp",
    "MergeOp",
    "PagewiseJob",
    "PagewiseResult",
    "ProtectOp",
    "RepairOp",
    "ReorderOp",
    "RotateOp",
    "ScanOp",
    "SelectOp",
    "SignOp",
    "StampOp",
    "UnlockOp",
    "artifact_body_text",
    "analyze_corpus",
    "analyze_document",
    "archive_pdf",
    "bullet_digest",
    "compare_documents",
    "detect_format",
    "diff_documents",
    "extract_requirements",
    "extract_text",
    "extract_page_texts",
    "extract_text_tables",
    "format_requirements_reply",
    "full_text",
    "format_pagewise_reply",
    "keyphrases",
    "keywords",
    "new_document",
    "ocr_available",
    "ocr_image",
    "ocr_languages",
    "ocr_pdf",
    "ocr_pdf_hocr",
    "ocr_pdf_words",
    "open_requirements_store",
    "parse_bytes",
    "parse_path",
    "pdf_info",
    "protect_pdf",
    "render_html",
    "render_markdown",
    "render_report_markdown",
    "render_terminal",
    "repair_pdf",
    "run_pagewise",
    "run_pdfcraft_parity",
    "scan_to_pdf",
    "score_links",
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
    "to_pdf_embedded",
    "to_pptx",
    "to_text",
    "to_xlsx",
    "unlock_pdf",
    "verify_pdfcraft_attribution",
    "verify_spans",
    "word_diff",
    "write_report",
]
