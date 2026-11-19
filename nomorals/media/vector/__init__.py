"""VectorCraft — Devon's vector-art organ (storytold/vectorcraft
port, census §3A.8).

A real vector document model over real SVG bytes: paths (bezier
curves, lines, arcs), shapes, fills/strokes, linear + radial
gradients, groups, layers, transforms, text and text-on-path —
parsed and written with stdlib XML, boolean path ops via
Weiler–Atherton, and a numpy + PIL rasterizer that turns a
document into a real PNG. Dual MIT/Apache-2.0 upstream; no
upstream code copied — the port is Devon's own Python following
the upstream design; provenance is machine-checked (see
``nomorals/media/vector-attribution.toml``).
"""

from __future__ import annotations

from .attribution import load_attribution, verify_attribution
from .edit import VectorEditor
from .errors import VectorError
from .geom import Affine, Path, parse_path_data, parse_transform
from .model import VectorDocument
from .parity import load_parity, run_parity
from .pathops import boolean, polygon_area
from .render import render_document, render_svg

__all__ = [
    "Affine",
    "Path",
    "VectorDocument",
    "VectorEditor",
    "VectorError",
    "boolean",
    "load_attribution",
    "load_parity",
    "parse_path_data",
    "parse_transform",
    "polygon_area",
    "render_document",
    "render_svg",
    "run_parity",
    "verify_attribution",
]
