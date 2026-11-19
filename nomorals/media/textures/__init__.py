"""Texture organ (TextureCraft port, census 3A.15).

Sparse copy-on-write 64x64 tiles, deterministic PCG3D noise, the 32
W3C blend modes, 23 procedural patterns, a layer-stack compositor,
a brush engine, mipmaps, normal-map derivation, BFS dilation and a
.tcraft codec — ported from storytold/texturecraft (commit
37ad2112, dual MIT/Apache-2.0) into Devon's Python, with attribution
in ``nomorals/media/textures-attribution.toml`` and machine-checked
parity in ``nomorals/media/textures-parity.toml``.
"""
from __future__ import annotations

from .attribution import verify_attribution
from .model import Channel, TextureDocument
from .parity import run_parity

__all__ = ["Channel", "TextureDocument", "run_parity",
           "verify_attribution"]
