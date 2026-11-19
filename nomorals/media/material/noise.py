"""Tileable noise primitives — the material organ's generators.

MaterialCraft port of upstream's ``crates/nodes/src/noise.rs`` and
``noise3d.rs`` (Perlin gradient noise, Worley/cellular noise, fractal
sums) plus the value-noise core. Everything is deterministic per
seed: the permutation table is built from the seed with SplitMix64,
never from Python's randomised ``hash()`` or a global RNG, so the
same seed cooks the same pixels on every machine and every run.

Tiling follows upstream: lattice coordinates wrap by the cell period
when tiling is on, so a noise tile repeats seamlessly. Pixel-based
parameters scale with ``w / 256`` (upstream's preview rule) through
the caller's ``scale`` argument.

The grid engine now lives in the shared library
(``nomorals/craftlibs/noise.py``); this module is its material
name for it. Same functions, same values — the material parity
pins prove it.
"""

from __future__ import annotations

from nomorals.craftlibs.noise import grid_fbm as fbm
from nomorals.craftlibs.noise import grid_hash2 as hash2
from nomorals.craftlibs.noise import grid_perlin as perlin
from nomorals.craftlibs.noise import grid_permutation_table as permutation_table
from nomorals.craftlibs.noise import grid_value_noise as value_noise
from nomorals.craftlibs.noise import grid_warp_field as warp_field
from nomorals.craftlibs.noise import grid_worley as worley

__all__ = ["fbm", "hash2", "perlin", "permutation_table", "value_noise",
           "warp_field", "worley"]
