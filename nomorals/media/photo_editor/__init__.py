"""Devon Studio native photo editor — the PhotoCraft port.

The organ:

* :mod:`nomorals.media.photo_editor.psd` — the PSD format crate
  (decode + encode, all four compressions).
* :mod:`nomorals.media.photo_editor.document` — the COW-tile
  layered document model + history + ladder edits.
* :mod:`nomorals.media.photo_editor.oracle` — the corpus-oracle
  harness (real PSDs, vendor composite as the truth).
* :mod:`nomorals.media.photo_editor.store` — durable sessions.
* :mod:`nomorals.media.photo_editor.page` — the Studio photo page.
* :mod:`nomorals.media.photo_editor.compose` — the shared
  compositor (layers, blends, masks, clipping, effects).
* :mod:`nomorals.media.photo_editor.adjust` — all 15 adjustment
  layers, parsed and evaluated.
* :mod:`nomorals.media.photo_editor.blend` — the 27 Photoshop
  blend modes.
* :mod:`nomorals.media.photo_editor.effects` — layer effects
  (bevel, stroke, glow, satin, shadow).
* :mod:`nomorals.media.photo_editor.text` — TySh live text.
* :mod:`nomorals.media.photo_editor.smart` — smart objects and
  smart filters (FEid unfiltered pixels included).
* :mod:`nomorals.media.photo_editor.fill` — solid/gradient/
  pattern fill layers.
* :mod:`nomorals.media.photo_editor.gradient` — the gradient
  model (Catmull-Rom bake, Oklab/Linear/Classic/Smooth).
* :mod:`nomorals.media.photo_editor.distance` — distance fields
  for effects (Euclidean + chamfer).
* :mod:`nomorals.media.photo_editor.descriptor` — the Photoshop
  descriptor (OSType) parser.

License basis (see the survey): PhotoCraft is dual MIT OR
Apache-2.0 ("Copyright (c) 2026 ArtCraft Team and the PhotoCraft
contributors"); the corpus PSDs are the same license. The code
here is Devon's own Python reimplementation of the pattern —
document model over COW tiles, PSD-format crate decomposition,
corpus-oracle testing — written from Adobe's public Photoshop File
Formats spec and the crate's structure, not a copy of the Rust.
"""

from __future__ import annotations

from .document import PhotoDocument, PhotoEditorError, PhotoLayer
from .psd import PsdError, PsdFile

__all__ = [
    "PhotoDocument",
    "PhotoEditorError",
    "PhotoLayer",
    "PsdError",
    "PsdFile",
]
