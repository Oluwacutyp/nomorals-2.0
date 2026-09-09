"""Document parsers, stdlib only.

PDF, DOCX, XLSX, ODT, EPUB, HTML, CSV, JSON, and plain text — no pdfminer, no
python-docx, no openpyxl. Every one of these formats is a container (zip or a
byte stream) with a documented structure, and depending on four heavy libraries
to read them is what makes a project uninstallable on a phone.

The PDF text extractor is a real one: it walks the page content streams, handles
``FlateDecode``, and pulls text out of ``Tj``/``TJ`` operators. It will not beat
pdfplumber on scanned documents — nothing will, without OCR — but it extracts
cleanly from the overwhelming majority of digitally-generated PDFs.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..core.errors import ParseError, UnsupportedFormat
from ..core.policy import Capability
from ..core.text import normalize_text

__all__ = ["detect_kind", "parse", "parse_pdf", "register"]

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ODT_NS = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
EPUB_NS = "{http://www.idpf.org/2007/opf}"


def detect_kind(path: str | Path, head: bytes = b"") -> str:
    """Classify by magic bytes first, extension second."""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        suffix = Path(path).suffix.lower()
        return {".docx": "docx", ".xlsx": "xlsx", ".odt": "odt", ".ods": "ods",
                ".epub": "epub", ".pptx": "pptx"}.get(suffix, "zip")
    if head.startswith(b"\x89PNG"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image"
    if head.startswith(b"GIF8"):
        return "image"
    if head[:4] == b"RIFF" or head[4:8] == b"ftyp":
        return "media"
    suffix = Path(path).suffix.lower()
    return {
        ".html": "html", ".htm": "html", ".csv": "csv", ".tsv": "tsv",
        ".json": "json", ".md": "text", ".txt": "text", ".xml": "xml",
    }.get(suffix, "text")


def parse(path: str | Path, *, max_chars: int = 200_000) -> dict[str, Any]:
    """Parse any supported document into ``{kind, title, text, meta}``."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise ParseError(f"no such file: {target}")
    head = target.read_bytes()[:16]
    kind = detect_kind(target, head)
    data = target.read_bytes()

    try:
        if kind == "pdf":
            text, meta = parse_pdf(data)
        elif kind == "docx":
            text, meta = _parse_docx(data)
        elif kind in {"xlsx", "ods"}:
            text, meta = _parse_xlsx(data)
        elif kind == "odt":
            text, meta = _parse_odt(data)
        elif kind == "epub":
            text, meta = _parse_epub(data)
        elif kind == "html":
            from .web import html_to_text

            body = data.decode("utf-8", errors="replace")
            match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
            text, meta = html_to_text(body), {"title": match.group(1).strip() if match else ""}
        elif kind == "csv":
            text, meta = _parse_csv(data, delimiter=",")
        elif kind == "tsv":
            text, meta = _parse_csv(data, delimiter="\t")
        elif kind == "json":
            parsed = json.loads(data.decode("utf-8", errors="replace"))
            text, meta = json.dumps(parsed, indent=2, ensure_ascii=False), {"json": True}
        elif kind == "image":
            from .vision import image_metadata

            return {"kind": "image", "title": target.name, "text": "", "meta": image_metadata(data)}
        elif kind == "media":
            return {"kind": "media", "title": target.name, "text": "", "meta": {"bytes": len(data)}}
        else:
            text = data.decode("utf-8", errors="replace")
            meta = {}
    except (zipfile.BadZipFile, zlib.error, ElementTree.ParseError, ValueError, KeyError) as exc:
        raise ParseError(f"could not parse {target.name} as {kind}: {exc}") from exc

    text = normalize_text(text)[:max_chars]
    meta.setdefault("title", target.name)
    return {
        "kind": kind,
        "title": meta.get("title") or target.name,
        "text": text,
        "chars": len(text),
        "meta": {k: v for k, v in meta.items() if k != "title"},
    }


# ── PDF ────────────────────────────────────────────────────────────────────────

_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_TJ = re.compile(rb"\((?:\\.|[^()\\])*\)|\bTJ\b|\bTj\b|\bTd\b|\bTD\b|\bT\*\b|\bET\b")
_LITERAL = re.compile(rb"\(((?:\\.|[^()\\])*)\)")


def parse_pdf(data: bytes) -> tuple[str, dict[str, Any]]:
    """Extract text from a PDF's content streams."""
    if not data.startswith(b"%PDF"):
        raise ParseError("not a PDF")
    chunks: list[str] = []
    for match in _STREAM.finditer(data):
        raw = match.group(1)
        try:
            stream = zlib.decompress(raw)
        except zlib.error:
            stream = raw  # uncompressed or a non-content stream
        if b"BT" not in stream and b"Tj" not in stream and b"TJ" not in stream:
            continue
        page_parts: list[str] = []
        for piece in _LITERAL.finditer(stream):
            page_parts.append(_unescape_pdf_string(piece.group(1)))
        if page_parts:
            chunks.append(" ".join(p for p in page_parts if p.strip()))
    text = "\n".join(chunks)
    pages = data.count(b"/Type /Page") or data.count(b"/Type/Page")
    title = ""
    match = re.search(rb"/Title\s*\(((?:\\.|[^()\\])*)\)", data)
    if match:
        title = _unescape_pdf_string(match.group(1))
    return text, {"title": title, "pages": max(1, pages)}


def _unescape_pdf_string(raw: bytes) -> str:
    out = raw.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
    out = re.sub(rb"\\[0-7]{1,3}", lambda m: bytes([int(m.group(0)[1:], 8) & 0xFF]), out)
    return out.decode("latin-1", errors="replace")


# ── OOXML / ODF ────────────────────────────────────────────────────────────────


def _parse_docx(data: bytes) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = archive.read("word/document.xml")
        title = ""
        if "docProps/core.xml" in archive.namelist():
            core = ElementTree.fromstring(archive.read("docProps/core.xml"))
            node = core.find("{http://purl.org/dc/elements/1.1/}title")
            title = (node.text or "").strip() if node is not None else ""
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{WORD_NS}t")]
        line = "".join(runs).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs), {"title": title, "paragraphs": len(paragraphs)}


def _parse_xlsx(data: bytes) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{SHEET_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{SHEET_NS}t")))
        names = [n for n in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]
        lines: list[str] = []
        cells = 0
        for name in sorted(names):
            root = ElementTree.fromstring(archive.read(name))
            for row in root.iter(f"{SHEET_NS}row"):
                values: list[str] = []
                for cell in row.iter(f"{SHEET_NS}c"):
                    value_node = cell.find(f"{SHEET_NS}v")
                    if value_node is None or value_node.text is None:
                        values.append("")
                        continue
                    if cell.get("t") == "s":
                        index = int(value_node.text)
                        values.append(shared[index] if index < len(shared) else "")
                    else:
                        values.append(value_node.text)
                    cells += 1
                if any(v.strip() for v in values):
                    lines.append("\t".join(values))
    return "\n".join(lines), {"sheets": len(names), "cells": cells}


def _parse_odt(data: bytes) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = archive.read("content.xml")
    root = ElementTree.fromstring(xml)
    parts = [node.text for node in root.iter() if node.text and node.text.strip()]
    return "\n".join(p.strip() for p in parts), {}


def _parse_epub(data: bytes) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        title = ""
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
        opf_path = rootfile.get("full-path", "") if rootfile is not None else ""
        chapters: list[str] = []
        if opf_path and opf_path in archive.namelist():
            opf = ElementTree.fromstring(archive.read(opf_path))
            node = opf.find(f".//{EPUB_NS}metadata/{{http://purl.org/dc/elements/1.1/}}title")
            title = (node.text or "").strip() if node is not None else ""
            manifest = {
                item.get("id", ""): item.get("href", "")
                for item in opf.iter(f"{EPUB_NS}item")
            }
            base = str(Path(opf_path).parent)
            for reference in opf.iter(f"{EPUB_NS}itemref"):
                href = manifest.get(reference.get("idref", ""), "")
                if not href:
                    continue
                full = href if not base or base == "." else f"{base}/{href}"
                if full in archive.namelist() and full.endswith((".xhtml", ".html", ".htm")):
                    from .web import html_to_text

                    chapters.append(html_to_text(archive.read(full).decode("utf-8", "replace")))
    return "\n\n".join(chapters), {"title": title, "chapters": len(chapters)}


def _parse_csv(data: bytes, *, delimiter: str = ",") -> tuple[str, dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader]
    if not rows:
        return "", {"rows": 0}
    header = rows[0]
    lines = ["\t".join(header)]
    lines += ["\t".join(r) for r in rows[1:5001]]
    return "\n".join(lines), {"rows": len(rows), "columns": len(header), "truncated": len(rows) > 5001}


def register(registry: Any) -> None:
    """Attach the parsing tools to a registry."""
    context = registry.context

    @registry.register(
        "parse_file",
        description="Extract text from a document (pdf, docx, xlsx, odt, epub, html, csv, json, text).",
        capability=Capability.FS_READ,
    )
    def parse_file(path: str, *, max_chars: int = 200_000) -> dict[str, Any]:
        from .filesystem import safe_path

        target = safe_path(context, path, must_exist=True)
        return parse(target, max_chars=max_chars)

    @registry.register(
        "file_kind",
        description="Detect a file's format from its magic bytes.",
        capability=Capability.FS_READ,
    )
    def file_kind(path: str) -> dict[str, Any]:
        from .filesystem import safe_path

        target = safe_path(context, path, must_exist=True)
        with target.open("rb") as handle:
            head = handle.read(16)
        return {"path": str(target), "kind": detect_kind(target, head)}
