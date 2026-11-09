"""Devon's typography service — fonts as searchable design data.

One organ owns everything about the faces Devon renders with:
a catalogue parsed from the real font files on the host
(:mod:`fontparse`, :mod:`catalogue`), deterministic semantic
selection with honest fallback chains (:mod:`selection`), real
complex-text shaping and grapheme-safe breaking (:mod:`shaping`),
and a bounded, journaled rendered-asset cache (:mod:`cache`),
behind one facade (:mod:`service`).

It serves rendered assets — cards, posters, deeds, captions,
covers, titles — never native chat text, which stays native.
Presentation themes stay with the theme system; this organ owns
faces and text engineering.
"""

from __future__ import annotations

from .cache import RenderCache
from .catalogue import FontCatalogue, build_catalogue, default_roots
from .fontparse import FontFace, parse_font_file, script_fraction
from .selection import INTENTS, Selection, face_traits, select
from .service import TypographyService, get_service, reset_service
from .shaping import (
                      dominant_direction,
                      graphemes,
                      load_font,
                      shaped_width,
                      shaping_engine,
                      unshaped_width,
                      wrap_text,
)

__all__ = [
    "FontCatalogue", "FontFace", "INTENTS", "RenderCache", "Selection",
    "TypographyService", "build_catalogue", "default_roots",
    "dominant_direction", "face_traits", "get_service", "graphemes",
    "load_font", "parse_font_file", "reset_service", "script_fraction",
    "select", "shaped_width", "shaping_engine", "unshaped_width",
    "wrap_text",
]
