"""PdfCraft port — the PDF object model, incremental editor, parity
runner, and attribution DB for Devon's documents organ.

Spec-decomposed like the upstream crates (``crates/cos`` → :mod:`cos`,
``crates/cos/document`` + ``crates/model`` → :mod:`model`,
``crates/edit`` → :mod:`edit`, ``parity/*.toml`` → :mod:`parity`,
``ATTRIBUTION.toml`` → :mod:`attribution`); dual MIT/Apache-2.0
upstream, attribution in
``nomorals/documents/pdfcraft-attribution.toml``.

Entry points: :class:`PdfDocument` (open + inventory),
:class:`PdfEditor` (incremental/full save), :func:`run_parity`
(machine-checked parity TOML), :func:`verify_attribution` (SHA-256
provenance check).  COS primitives (:class:`PdfName`,
:class:`PdfRef`, :class:`PdfString`, :class:`PdfStream`,
:func:`parse_object`, :func:`parse_xref`, :func:`serialize`) are
exported for callers that work at the object level.
"""

from __future__ import annotations

from .attribution import (
    DEFAULT_ATTRIBUTION_PATH,
    DEFAULT_CLONE_ROOT,
    load_attribution,
    verify_attribution,
)
from .cos import (
    PdfName,
    PdfRef,
    PdfStream,
    PdfString,
    Xref,
    XrefEntry,
    build_full,
    build_incremental,
    decode_stream,
    parse_indirect,
    parse_object,
    parse_xref,
    resolve,
    serialize,
    serialize_indirect,
)
from .edit import PdfEditor
from .model import PdfDocument, PdfPage
from .parity import DEFAULT_PARITY_PATH, load_parity, run_parity

__all__ = [
    "DEFAULT_ATTRIBUTION_PATH",
    "DEFAULT_CLONE_ROOT",
    "DEFAULT_PARITY_PATH",
    "PdfDocument",
    "PdfEditor",
    "PdfName",
    "PdfPage",
    "PdfRef",
    "PdfStream",
    "PdfString",
    "Xref",
    "XrefEntry",
    "build_full",
    "build_incremental",
    "decode_stream",
    "load_attribution",
    "load_parity",
    "parse_indirect",
    "parse_object",
    "parse_xref",
    "resolve",
    "run_parity",
    "serialize",
    "serialize_indirect",
    "verify_attribution",
]
