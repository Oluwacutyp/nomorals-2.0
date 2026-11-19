"""The ``.designcraft`` package codec.

Upstream's native format (``docs/architecture.md``: the ``format``
crate) is a zip — ``document.json``, ``assets/``, ``meta.json``,
optional ``preview.png`` — versioned. Devon's port uses the same
shape at format version 1, with the fidelity discipline of the
word/pdfcraft organs: bounded sizes, deterministic bytes (fixed
zip timestamps, canonical JSON, sorted entries) so an unedited
document writes back byte-identical.
"""

from __future__ import annotations

import io
import json
import zipfile

from .errors import LayoutError
from .model import LayoutDocument

__all__ = ["MAX_PACKAGE_BYTES", "MAX_PART_BYTES", "parse_document",
           "read_package", "to_document_bytes", "write_package"]

#: Bounds mirroring the word organ's OOXML package caps.
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 100
MAX_JSON_ELEMENTS = 500_000

_FORMAT_VERSION = 1
_DATE = (1980, 1, 1, 0, 0, 0)


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _bounded_json_loads(data: bytes):
    """Parse JSON with a depth/element budget so a hostile file
    cannot blow up the parser (same habit as the word organ's
    bounded XML)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LayoutError(f"document.json is not UTF-8: {exc}",
                          kind="parse") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LayoutError(f"document.json is not valid JSON: {exc}",
                          kind="parse") from exc
    count = [0]

    def walk(node, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise LayoutError("document.json nests too deep",
                              kind="parse")
        count[0] += 1
        if count[0] > MAX_JSON_ELEMENTS:
            raise LayoutError("document.json has too many elements",
                              kind="parse")
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(obj, 0)
    return obj


def read_package(data: bytes) -> tuple[dict, dict[str, bytes], dict]:
    """Read a ``.designcraft`` zip into (document dict, assets,
    meta dict)."""
    if len(data) > MAX_PACKAGE_BYTES:
        raise LayoutError(
            f"package is {len(data)} bytes (max {MAX_PACKAGE_BYTES})",
            kind="parse")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise LayoutError(f"not a .designcraft package: {exc}",
                          kind="parse") from exc
    with zf:
        names = zf.namelist()
        if "document.json" not in names:
            raise LayoutError("package has no document.json",
                              kind="parse")
        assets: dict[str, bytes] = {}
        for name in names:
            info = zf.getinfo(name)
            if info.file_size > MAX_PART_BYTES:
                raise LayoutError(
                    f"part {name!r} is {info.file_size} bytes "
                    f"(max {MAX_PART_BYTES})", kind="parse")
            if name.startswith("assets/"):
                assets[name] = zf.read(name)
        doc_raw = _bounded_json_loads(zf.read("document.json"))
        meta: dict = {}
        if "meta.json" in names:
            meta = _bounded_json_loads(zf.read("meta.json"))
    if not isinstance(doc_raw, dict):
        raise LayoutError("document.json is not an object", kind="parse")
    return doc_raw, assets, meta


def write_package(doc: LayoutDocument) -> bytes:
    """Serialize a document to deterministic ``.designcraft`` bytes:
    entries in order document.json, meta.json, assets/* sorted;
    fixed timestamps; canonical JSON."""
    doc_dict = doc.to_dict()
    pages = sum(len(s.pages) for s in doc.spreads)
    meta = {"format_version": _FORMAT_VERSION, "title": doc.title,
            "pages": pages}
    entries = [("document.json", _canonical_json(doc_dict)),
               ("meta.json", _canonical_json(meta))]
    entries.extend(sorted(doc.assets.items()))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buf.getvalue()


def parse_document(data: bytes) -> LayoutDocument:
    """Parse ``.designcraft`` bytes into a document."""
    doc_raw, assets, _meta = read_package(data)
    doc = LayoutDocument.from_dict(doc_raw)
    doc.assets = assets
    return doc


def to_document_bytes(doc: LayoutDocument) -> bytes:
    """Serialize a document to ``.designcraft`` bytes."""
    return write_package(doc)
