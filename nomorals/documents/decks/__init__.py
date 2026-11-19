"""DeckCraft port — the presentation document model, editor,
renderer, parity runner, and attribution DB for Devon's documents
organ.

Spec-decomposed like the upstream crates (``crates/model`` →
:mod:`model`, ``crates/pptx/read`` + ``crates/pptx/write`` +
``crates/pptx/opc`` + ``crates/pptx/xml`` → :mod:`ooxml`,
``crates/engine/cmd/*`` → :mod:`edit`, ``crates/render`` →
:mod:`render`, ``parity/*.toml`` → :mod:`parity`,
``ATTRIBUTION.toml`` → :mod:`attribution`); dual MIT/Apache-2.0
upstream, attribution in
``nomorals/documents/deckcraft-attribution.toml``.

Entry points: :class:`DeckDocument` (open + inventory),
:class:`DeckEditor` (fidelity-preserving edits),
:func:`render_slide` / :func:`render_slide_to_png` (slide → PNG),
:func:`run_parity` (machine-checked parity TOML),
:func:`verify_attribution` (SHA-256 provenance check). OOXML
primitives (:class:`PptxPackage`, :func:`parse_xml_bounded`,
:func:`serialize_xml`) are exported for callers that work at the
package level. The XML helpers themselves are shared with the
word organ (:mod:`nomorals.documents.word.ooxml`) — one seam,
not two copies.
"""

from __future__ import annotations

from .attribution import (
    DEFAULT_ATTRIBUTION_PATH,
    DEFAULT_CLONE_ROOT,
    load_attribution,
    verify_attribution,
)
from .edit import DeckEditor
from .model import (
    DeckDocument,
    DeckLayout,
    DeckMaster,
    DeckParagraph,
    DeckRun,
    DeckShape,
    DeckSlide,
)
from .ooxml import (
    PptxPackage,
    parse_xml_bounded,
    read_package,
    serialize_xml,
    write_package,
)
from .parity import DEFAULT_PARITY_PATH, load_parity, run_parity
from .render import render_slide, render_slide_to_png

__all__ = [
    "DEFAULT_ATTRIBUTION_PATH",
    "DEFAULT_CLONE_ROOT",
    "DEFAULT_PARITY_PATH",
    "DeckDocument",
    "DeckEditor",
    "DeckLayout",
    "DeckMaster",
    "DeckParagraph",
    "DeckRun",
    "DeckShape",
    "DeckSlide",
    "PptxPackage",
    "load_attribution",
    "load_parity",
    "parse_xml_bounded",
    "read_package",
    "render_slide",
    "render_slide_to_png",
    "run_parity",
    "serialize_xml",
    "verify_attribution",
    "write_package",
]
