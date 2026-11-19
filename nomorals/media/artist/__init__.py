"""Devon's artist IDE — the ArtCraft port (storytold/artcraft,
dual MIT/Apache-2.0).

Build the scene before generating: 2D layers, 3D staging with depth
+ camera, posed characters with identity references, one consistent
location across shots — then generate through her own rungs (imggen
Studio, videogen chain, motion studio, talking-head). No paid
service, no per-clip billing, no limits.

* :mod:`nomorals.media.artist.scene` — the scene model.
* :mod:`nomorals.media.artist.compose` — 2D compositing + 3D staging.
* :mod:`nomorals.media.artist.store` — durable scene store.
* :mod:`nomorals.media.artist.generate` — orchestration over her rungs.
"""
from __future__ import annotations

from .generate import ArtistEngine
from .scene import (
    UPSTREAM,
    ArtistError,
    Camera,
    Character,
    Layer,
    Location,
    Scene,
    Shot,
    StageItem,
)
from .store import ArtistStore, artist_dir, artist_path

__all__ = [
    "ArtistEngine",
    "ArtistError",
    "ArtistStore",
    "Camera",
    "Character",
    "Layer",
    "Location",
    "Scene",
    "Shot",
    "StageItem",
    "UPSTREAM",
    "artist_dir",
    "artist_path",
]
