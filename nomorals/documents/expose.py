"""Expose documents — thin wrappers over the real document engine.

Delegates to :mod:`nomorals.documents` (parsers / converters /
compare / summarize / index). Parsing is stdlib-only and works
offline on the phone: PDF, DOCX, XLSX/XLS, PPTX, HTML, Markdown,
CSV/TSV, TXT, RTF, EPUB, ODT/ODS, JSON, XML — 16 formats verified
from ``parsers._EXTENSIONS`` (see :func:`supported_formats`).

Output caveat, stated honestly: ``to_docx`` needs the optional
``python-docx`` package and fails fast with an install hint when it
is missing (``convert.py:380``). Every other converter is stdlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .compare import compare_documents as _compare_documents
from .convert import (
    to_csv as _to_csv,
    to_csv_all as _to_csv_all,
    to_docx as _to_docx,
    to_epub as _to_epub,
    to_html as _to_html,
    to_json as _to_json,
    to_markdown as _to_markdown,
    to_pdf as _to_pdf,
    to_text as _to_text,
)
from .model import Document
from .parsers import _EXTENSIONS, detect_format as _detect_format
from .parsers import parse_bytes as _parse_bytes
from .parsers import parse_path as _parse_path
from .summarize import summarize as _summarize

__all__ = [
    "SUPPORTED_FORMATS",
    "supported_formats",
    "list_formats",
    "detect_document_format",
    "read_document",
    "convert_document",
    "summarize_document",
    "compare_document_pair",
    "document_capabilities",
]

#: Distinct format keys from ``parsers._EXTENSIONS`` — computed from the
#: real table, not a hand-typed list, so it cannot drift from the parser.
SUPPORTED_FORMATS: tuple[str, ...] = tuple(sorted(set(_EXTENSIONS.values())))

#: Converter targets ``convert_document`` accepts.
CONVERT_TARGETS: tuple[str, ...] = (
    "markdown", "text", "html", "pdf", "csv", "csv_all", "json", "epub", "docx",
)

#: Notes that only apply to specific formats/targets.
_FORMAT_NOTES: dict[str, str] = {
    "csv": "delimited parser (encoding fallback); not in _PARSERS, handled by parse_bytes branch",
    "tsv": "delimited parser (encoding fallback); not in _PARSERS, handled by parse_bytes branch",
    "xls": "legacy OLE compound document parser (stdlib)",
    "pdf": "stdlib PDF text extraction + /Info metadata + whitespace table recovery",
    "docx": "stdlib zipfile XML — lists, hyperlinks, footnotes, tables (no python-docx needed to READ)",
    "pptx": "stdlib zipfile XML — no python-pptx needed",
    "epub": "stdlib zipfile",
    "odt": "stdlib ODF parser",
    "ods": "stdlib ODF parser",
    "rtf": "stdlib, incl. tables",
    "html": "with meta harvesting",
}


def supported_formats() -> list[str]:
    """Format keys the parser accepts (16, from ``parsers._EXTENSIONS``)."""
    return list(SUPPORTED_FORMATS)


def list_formats() -> list[dict[str, Any]]:
    """One entry per supported format: extensions, parser, notes.

    ``stdlib_only`` is True for parsing every format. Output is
    stdlib-only except ``docx`` output, which needs optional
    ``python-docx`` (see :func:`document_capabilities`).
    """
    exts: dict[str, list[str]] = {}
    for ext, fmt in sorted(_EXTENSIONS.items()):
        exts.setdefault(fmt, []).append(ext)
    return [
        {
            "format": fmt,
            "extensions": exts.get(fmt, []),
            "parser": True,
            "stdlib_only": True,
            "note": _FORMAT_NOTES.get(fmt, ""),
        }
        for fmt in SUPPORTED_FORMATS
    ]


def detect_document_format(data: bytes, *, filename: str = "",
                           mime: str = "") -> str:
    """Sniff a format without parsing (extension → magic → MIME → text)."""
    return _detect_format(data, filename=filename, mime=mime)


def read_document(*, path: str | Path | None = None,
                  data: bytes | None = None,
                  filename: str = "", mime: str = "") -> Document:
    """Parse a document from ``path`` or raw ``data`` (exactly one).

    Delegates to ``parse_path`` / ``parse_bytes``. Raises
    ``DocumentError`` on unknown or unparseable input — never returns
    a silently-empty Document.
    """
    if (path is None) == (data is None):
        raise ValueError("read_document needs exactly one of path= / data=")
    if path is not None:
        return _parse_path(path)
    assert data is not None
    return _parse_bytes(data, filename=filename, mime=mime)


def convert_document(doc: Document, target: str, **kw: Any) -> Any:
    """Convert a parsed :class:`Document` to ``target``.

    Targets: ``markdown``, ``text``, ``html``, ``pdf``, ``csv``
    (one table; ``table=`` index or name), ``csv_all`` (every table),
    ``json``, ``epub`` (stdlib zipfile), ``docx`` (needs optional
    ``python-docx`` — fails fast with an install hint when missing).
    """
    t = (target or "").strip().lower()
    if t == "markdown":
        return _to_markdown(doc, **kw)
    if t == "text":
        return _to_text(doc)
    if t == "html":
        return _to_html(doc, **kw)
    if t == "pdf":
        return _to_pdf(doc)
    if t == "csv":
        return _to_csv(doc, **kw)
    if t == "csv_all":
        return _to_csv_all(doc)
    if t == "json":
        return _to_json(doc, **kw)
    if t == "epub":
        return _to_epub(doc, **kw)
    if t == "docx":
        return _to_docx(doc)
    raise ValueError(
        f"unknown convert target {target!r}; expected one of "
        f"{list(CONVERT_TARGETS)}")


def summarize_document(doc: Document, *, sentences: int = 5,
                       method: str = "tf", **kw: Any) -> Any:
    """Delegate to ``documents.summarize.summarize``."""
    return _summarize(doc, sentences=sentences, method=method, **kw)


def compare_document_pair(old: Document, new: Document) -> Any:
    """Delegate to ``documents.compare.compare_documents`` (diff + score)."""
    return _compare_documents(old, new)


def document_capabilities() -> dict[str, Any]:
    """Capability card: formats, targets, stdlib/offline truth."""
    return {
        "formats": list(SUPPORTED_FORMATS),
        "format_count": len(SUPPORTED_FORMATS),
        "convert_targets": list(CONVERT_TARGETS),
        "parse_stdlib_only": True,
        "parse_offline": True,
        "optional_deps": {
            "docx_output": "python-docx (only for convert target 'docx'; "
                           "reading .docx is stdlib)",
        },
        "notes": "Every parser is stdlib (zipfile / xml / csv / json). "
                 "No network, no API key, phone-offline safe.",
    }
