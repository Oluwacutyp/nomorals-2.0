"""Layout — page and layout design (DesignCraft port).

Devon's layout organ: a real page/layout document model
(spreads, pages, margins, columns, guides, layers, swatches,
styles, stories, items) in upstream's ``.designcraft`` zip shape,
a faithful port of upstream's flex layout engine, real text
composition measured with the PIL font door, a page renderer to
PNG, and fidelity-preserving editing — with machine-checked parity
and SHA-256 attribution.

Port of the *design* of storytold/designcraft (an open-source
page and layout design editor in pure Rust, dual MIT/Apache-2.0,
upstream commit fb630909c4851a0d3987f0a118b1c8fd107c9c97). The
code here is Devon's own Python; the design, the flex algorithm
and the format shape follow upstream, and every ported unit is
pinned in ``layout-attribution.toml``.
"""

from __future__ import annotations

from .attribution import load_attribution, verify_attribution
from .codec import parse_document, to_document_bytes
from .compose import (
                      ComposedFrame,
                      ComposedLine,
                      ComposedStory,
                      compose_story,
                      font_for,
                      measure_text,
)
from .edit import LayoutEditor
from .errors import LayoutError, LayoutErrorKind
from .flex import Arrangement, arrange, contains_flex, layout_all, layout_item
from .model import (
                      CharRun,
                      Columns,
                      DocumentGrid,
                      Fill,
                      FlexAlign,
                      FlexDirection,
                      FlexJustify,
                      FlexLayout,
                      FlexSize,
                      Guide,
                      Item,
                      Layer,
                      LayoutDocument,
                      Margins,
                      Page,
                      ParaFormat,
                      Section,
                      Spread,
                      Story,
                      Stroke,
                      Swatch,
                      TextFrameOptions,
)
from .parity import PARITY_TOML, run_parity
from .render import render_document, render_page

__all__ = [
    "Arrangement", "CharRun", "Columns", "ComposedFrame",
    "ComposedLine", "ComposedStory", "DocumentGrid", "Fill",
    "FlexAlign", "FlexDirection", "FlexJustify", "FlexLayout",
    "FlexSize", "Guide", "Item", "Layer", "LayoutDocument",
    "LayoutEditor", "LayoutError", "LayoutErrorKind", "Margins",
    "Page", "ParaFormat", "PARITY_TOML", "Section", "Spread",
    "Story", "Stroke", "Swatch", "TextFrameOptions", "arrange",
    "compose_story", "contains_flex", "font_for", "layout_all",
    "layout_item", "load_attribution", "measure_text",
    "parse_document", "render_document", "render_page", "run_parity",
    "to_document_bytes", "verify_attribution",
]
