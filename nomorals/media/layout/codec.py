"""The ``.designcraft`` package codec.

Upstream's native format (``docs/architecture.md``: the ``format``
crate) is a zip — ``document.json``, ``assets/``, ``meta.json``,
optional ``preview.png`` — versioned. Devon's port uses the same
shape at format version 1, with the fidelity discipline of the
word/pdfcraft organs: bounded sizes, deterministic bytes (fixed
zip timestamps, canonical JSON, sorted entries) so an unedited
document writes back byte-identical.

The package shape itself now lives in the shared library
(``nomorals/craftlibs/codec.py``); this module keeps the document
mapping and the entry order (``document.json``, ``meta.json``,
``assets/*`` sorted — which differs from sorted order), and
delegates the zip and JSON work, translating only the error type.
"""

from __future__ import annotations

from nomorals.craftlibs.codec import (
    MAX_PACKAGE_BYTES,
    MAX_PART_BYTES,
    CodecError,
    write_package_ordered,
)
from nomorals.craftlibs.codec import (
    canonical_json as _canonical_json,
)
from nomorals.craftlibs.codec import (
    read_document_package as _shared_read,
)

from .errors import LayoutError
from .model import LayoutDocument

__all__ = ["MAX_PACKAGE_BYTES", "MAX_PART_BYTES", "parse_document",
           "read_package", "to_document_bytes", "write_package"]

_FORMAT_VERSION = 1


def read_package(data: bytes) -> tuple[dict, dict[str, bytes], dict]:
    """Read a ``.designcraft`` zip into (document dict, assets,
    meta dict)."""
    try:
        doc_raw, assets, meta = _shared_read(data)
    except CodecError as exc:
        raise LayoutError(str(exc), kind="parse") from exc
    if not isinstance(doc_raw, dict):
        raise LayoutError("document.json is not an object",
                          kind="parse")
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
    return write_package_ordered(entries)


def parse_document(data: bytes) -> LayoutDocument:
    """Parse ``.designcraft`` bytes into a document."""
    doc_raw, assets, _meta = read_package(data)
    doc = LayoutDocument.from_dict(doc_raw)
    doc.assets = assets
    return doc


def to_document_bytes(doc: LayoutDocument) -> bytes:
    """Serialize a document to ``.designcraft`` bytes."""
    return write_package(doc)
