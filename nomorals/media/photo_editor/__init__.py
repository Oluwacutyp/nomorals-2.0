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
