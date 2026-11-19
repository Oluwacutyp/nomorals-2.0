"""WordCraft port — the word-processor document model, editor, parity
runner, and attribution DB for Devon's documents organ.

Spec-decomposed like the upstream crates (``crates/doc/document`` +
``crates/doc/block`` + ``crates/doc/para`` + ``crates/doc/table`` +
``crates/doc/section`` → :mod:`model`, ``crates/doc/edit`` +
``crates/doc/styles`` + ``crates/doc/numbering`` → :mod:`edit`,
``crates/docx/read`` + ``crates/docx/write`` + ``crates/docx/package``
+ ``crates/docx/xml`` → :mod:`ooxml`, ``parity/*.toml`` → :mod:`parity`,
``ATTRIBUTION.toml`` → :mod:`attribution`); dual MIT/Apache-2.0
upstream, attribution in
``nomorals/documents/word-attribution.toml``.

Entry points: :class:`WordDocument` (open + inventory),
:class:`WordEditor` (fidelity-preserving edits), :func:`run_parity`
(machine-checked parity TOML), :func:`verify_attribution` (SHA-256
provenance check).  OOXML primitives (:class:`DocxPackage`,
:func:`parse_xml_bounded`, :func:`serialize_xml`) are exported for
callers that work at the package level.
"""

from __future__ import annotations

from .attribution import (
    DEFAULT_ATTRIBUTION_PATH,
    DEFAULT_CLONE_ROOT,
    load_attribution,
    verify_attribution,
)
from .edit import WordEditor
from .model import (
    WordCell,
    WordDocument,
    WordParagraph,
    WordRun,
    WordStyle,
    WordTable,
)
from .ooxml import (
    DocxPackage,
    parse_xml_bounded,
    read_package,
    serialize_xml,
    write_package,
)
from .parity import DEFAULT_PARITY_PATH, load_parity, run_parity

__all__ = [
    "DEFAULT_ATTRIBUTION_PATH",
    "DEFAULT_CLONE_ROOT",
    "DEFAULT_PARITY_PATH",
    "DocxPackage",
    "WordCell",
    "WordDocument",
    "WordEditor",
    "WordParagraph",
    "WordRun",
    "WordStyle",
    "WordTable",
    "load_attribution",
    "load_parity",
    "parse_xml_bounded",
    "read_package",
    "run_parity",
    "serialize_xml",
    "verify_attribution",
    "write_package",
]
