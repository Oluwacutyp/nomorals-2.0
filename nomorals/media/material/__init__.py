"""Materials — node-based procedural PBR materials (MaterialCraft port).

Devon's materials organ: a real node-graph material designer ported
from upstream's ``crates/core`` (Package → Graph → Node /
Connection), ``crates/nodes`` (the node library: generators,
patterns, filters, PBR utilities), ``crates/eval`` (topological
cook + content-addressed cache), ``crates/formats`` (``.mcraft``
packages, ``.mcmat`` portable materials) and ``crates/render``
(CPU PBR preview with GGX + ACES). The code here is Devon's own
Python; the design, the cook/cache rules and the format shape
follow upstream, and every ported unit is pinned in
``material-attribution.toml``.
"""

from __future__ import annotations

from .attribution import load_attribution, verify_attribution
from .edit import MaterialEditor
from .errors import MaterialError, MaterialErrorKind
from .eval import CookCache, Cooked, cook, hash_of, topological_order
from .image import FloatImage
from .library import build_preset, list_presets, search
from .model import (
    BaseParams,
    Connection,
    Graph,
    MaterialDocument,
    Node,
)
from .nodes import by_id, catalog, resolve_params
from .parity import run_parity
from .pbr import render_channel, render_preview

__all__ = [
    "BaseParams", "Connection", "CookCache", "Cooked", "FloatImage",
    "Graph", "MaterialDocument", "MaterialEditor", "MaterialError",
    "MaterialErrorKind", "Node", "build_preset", "by_id", "catalog",
    "cook", "hash_of", "list_presets", "load_attribution",
    "render_channel", "render_preview", "resolve_params", "run_parity",
    "search", "topological_order", "verify_attribution",
]
