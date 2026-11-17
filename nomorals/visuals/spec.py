"""Scene spec: the structured, style-agnostic scene the brain composes.

A spec is plain data in one of three kinds:

- ``diagram`` — nodes and edges (a call graph, a prior-art map, a
  before/after reframe). Nodes carry normalized 0..1 coordinates.
- ``overlay`` — annotations pinned to coordinates over existing
  content (the Clicky [POINT]/CURSOR/RECT vocabulary, as data).
- ``scene`` — 2.5D/3D primitives (the motion-rungs scene schema).

The spec is the interchange format: the brain composes it, the
renderers turn it into deliverables, and the code-video rung
consumes the validated spec directly — the next stage never
re-authors it. Styles live in consumers; the spec carries structure
only (the 2026-10-09 architecture principle).

The brain never has to hand-author coordinates: nodes without
``x``/``y`` are placed by a deterministic layered layout
(:func:`auto_layout`), and :func:`diagram_from_parts` builds a
diagram from plain node ids and edge pairs. Validation errors name
the offending field — the error IS the authoring feedback.
"""

from __future__ import annotations

import copy
import re
from typing import Any

__all__ = [
    "SceneSpecError",
    "auto_layout",
    "diagram_from_parts",
    "validate_scene_spec",
]

#: The three spec kinds. Anything else is an authoring error.
KINDS = ("diagram", "overlay", "scene")

#: Diagram node shapes (the motion-studio explainer vocabulary).
NODE_SHAPES = ("box", "circle")

#: Overlay annotation kinds (the clickyX tag vocabulary, as data —
#: annotation is never actuation).
ANNOTATION_KINDS = ("cursor", "rect", "highlight", "caption",
                    "scribble", "shape")

#: Named colours the brain may author; everything else must be hex.
NAMED_COLOURS: dict[str, str] = {
    "red": "#EF4444",
    "green": "#22C55E",
    "blue": "#38BDF8",
    "yellow": "#FBBF24",
    "pink": "#F472B6",
    "purple": "#A78BFA",
    "orange": "#FB923C",
    "white": "#F8FAFC",
    "black": "#0F172A",
    "grey": "#94A3B8",
    "gray": "#94A3B8",
}

#: Default node colour cycle (the motion-studio palette).
PALETTE = ("#38BDF8", "#F472B6", "#A3E635", "#FBBF24",
           "#A78BFA", "#FB923C")

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class SceneSpecError(ValueError):
    """A scene spec failed validation. The message names the field."""


def _fail(field: str, problem: str) -> None:
    raise SceneSpecError(f"scene spec: {field}: {problem}")


def _colour_of(raw: Any, idx: int, field: str) -> str:
    if raw is None or raw == "":
        return PALETTE[idx % len(PALETTE)]
    text = str(raw).strip()
    if _HEX_RE.match(text):
        return text.upper() if len(text) == 7 else text
    lowered = text.lower()
    if lowered in NAMED_COLOURS:
        return NAMED_COLOURS[lowered]
    _fail(field, f"colour must be #rrggbb or one of "
                 f"{sorted(NAMED_COLOURS)}, got {raw!r}")


def _coord(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(field, f"needs a numeric coordinate, got {value!r}")
    if not 0.0 <= number <= 1.0:
        _fail(field, f"coordinate must be 0..1, got {number}")
    return number


def auto_layout(nodes: list[dict[str, Any]],
                edges: list[dict[str, Any]]
                ) -> list[dict[str, Any]]:
    """Deterministic layered positions for every node.

    Depth is the longest path from a source (a node with no incoming
    edge); a node's x grows with depth, its y spreads evenly inside
    its layer. Pure — the input dicts are never mutated, and the same
    input always yields the same positions. Cycles cannot loop it:
    the depth pass is capped at one iteration per node.
    """
    ids = [str(n.get("id", "")) for n in nodes]
    preds: dict[str, set[str]] = {i: set() for i in ids}
    for edge in edges:
        src, dst = str(edge.get("from", "")), str(edge.get("to", ""))
        if src in preds and dst in preds:
            preds[dst].add(src)
    depth: dict[str, int] = {i: 0 for i in ids}
    for _ in range(len(ids)):
        changed = False
        for node_id in ids:
            if preds[node_id]:
                candidate = 1 + max(depth[p] for p in preds[node_id])
                if candidate > depth[node_id]:
                    depth[node_id] = candidate
                    changed = True
        if not changed:
            break
    max_depth = max(depth.values()) if depth else 0
    layers: dict[int, list[str]] = {}
    for node_id in ids:
        layers.setdefault(depth[node_id], []).append(node_id)
    placed: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        layer = layers[depth[node_id]]
        item = dict(node)
        item["x"] = (depth[node_id] + 1) / (max_depth + 2)
        item["y"] = (layer.index(node_id) + 1) / (len(layer) + 1)
        placed.append(item)
    return placed


def _validate_diagram(spec: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail("diagram.nodes", "needs a non-empty 'nodes' list")
    seen: set[str] = set()
    clean_nodes: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_nodes):
        field = f"diagram.nodes[{i}]"
        if not isinstance(raw, dict):
            _fail(field, "must be a dict")
        node_id = str(raw.get("id", ""))
        if not node_id:
            _fail(field, "missing 'id'")
        if node_id in seen:
            _fail(field, f"duplicate node id {node_id!r}")
        seen.add(node_id)
        shape = str(raw.get("shape", "box"))
        if shape not in NODE_SHAPES:
            _fail(field, f"shape must be one of {NODE_SHAPES}, "
                         f"got {shape!r}")
        clean_nodes.append({
            "id": node_id,
            "label": str(raw.get("label", node_id)),
            "x": raw.get("x"),
            "y": raw.get("y"),
            "shape": shape,
            "color": _colour_of(raw.get("color"), i, field + ".color"),
        })
    raw_edges = spec.get("edges") or []
    if not isinstance(raw_edges, list):
        _fail("diagram.edges", "must be a list")
    clean_edges: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_edges):
        field = f"diagram.edges[{i}]"
        if not isinstance(raw, dict):
            _fail(field, "must be a dict")
        src, dst = str(raw.get("from", "")), str(raw.get("to", ""))
        for end in (src, dst):
            if end not in seen:
                _fail(field, f"endpoint {end!r} is not a node id")
        clean_edges.append({
            "from": src,
            "to": dst,
            "label": str(raw.get("label", "")),
        })
    # Nodes without both coordinates take the deterministic layout;
    # a node with both keeps exactly what the author wrote.
    laid_out = auto_layout(clean_nodes, clean_edges)
    nodes: list[dict[str, Any]] = []
    for authored, placed in zip(clean_nodes, laid_out, strict=True):
        if authored["x"] is None or authored["y"] is None:
            authored["x"] = placed["x"]
            authored["y"] = placed["y"]
        authored["x"] = _coord(authored["x"], f"diagram node {authored['id']}.x")
        authored["y"] = _coord(authored["y"], f"diagram node {authored['id']}.y")
        nodes.append(authored)
    return {"nodes": nodes, "edges": clean_edges}


def _validate_overlay(spec: dict[str, Any]) -> dict[str, Any]:
    raw = spec.get("annotations")
    if not isinstance(raw, list) or not raw:
        _fail("overlay.annotations", "needs a non-empty 'annotations' list")
    annotations: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        field = f"overlay.annotations[{i}]"
        if not isinstance(item, dict):
            _fail(field, "must be a dict")
        kind = str(item.get("kind", "cursor"))
        if kind not in ANNOTATION_KINDS:
            _fail(field, f"kind must be one of {ANNOTATION_KINDS}, "
                         f"got {kind!r}")
        annotations.append({
            "x": _coord(item.get("x"), field + ".x"),
            "y": _coord(item.get("y"), field + ".y"),
            "text": str(item.get("text", "")),
            "kind": kind,
        })
    out: dict[str, Any] = {"annotations": annotations}
    if spec.get("background") is not None:
        out["background"] = str(spec["background"])
    return out


def _validate_scene(spec: dict[str, Any]) -> dict[str, Any]:
    raw_objects = spec.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        _fail("scene.objects", "needs a non-empty 'objects' list")
    # The 3D object rules live in one place — the motion rungs. This
    # organ validates the envelope and delegates the objects, so the
    # two can never drift.
    from ..media.motion_rungs import validate_scene_spec as _validate_3d

    try:
        checked = _validate_3d({
            "camera": spec.get("camera") or {},
            "background": spec.get("background", "#101018"),
            "objects": raw_objects,
        })
    except ValueError as exc:
        raise SceneSpecError(f"scene spec: {exc}") from None
    return {"camera": checked["camera"],
            "background": checked["background"],
            "objects": checked["objects"]}


def validate_scene_spec(spec: Any) -> dict[str, Any]:
    """Validate + normalize a scene spec (pure data).

    Returns the normalized spec the renderers consume. Raises
    :class:`SceneSpecError` naming the offending field. The input
    dict is never mutated — the normalized copy is a fresh dict.
    """
    if not isinstance(spec, dict):
        raise SceneSpecError("scene spec: expected a dict")
    kind = str(spec.get("kind", ""))
    if kind not in KINDS:
        raise SceneSpecError(
            f"scene spec: 'kind' must be one of {KINDS}, got {kind!r}")
    working = copy.deepcopy(spec)
    out: dict[str, Any] = {
        "kind": kind,
        "title": str(working.get("title", "")),
    }
    if kind == "diagram":
        out.update(_validate_diagram(working))
    elif kind == "overlay":
        out.update(_validate_overlay(working))
    else:
        out.update(_validate_scene(working))
    return out


def diagram_from_parts(title: str,
                       nodes: list[Any],
                       edges: list[Any] = ()) -> dict[str, Any]:
    """A diagram from plain parts — the brain names nodes and edges,
    the organ lays them out.

    ``nodes`` are ids (``"a"``) or node dicts; ``edges`` are
    ``(from, to)`` pairs, ``(from, to, label)`` triples, or edge
    dicts. Returns the validated spec.
    """
    norm_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, str):
            norm_nodes.append({"id": node})
        else:
            norm_nodes.append(dict(node))
    norm_edges: list[dict[str, Any]] = []
    for edge in edges:
        if isinstance(edge, (tuple, list)) and len(edge) == 2:
            norm_edges.append({"from": edge[0], "to": edge[1]})
        elif isinstance(edge, (tuple, list)) and len(edge) == 3:
            norm_edges.append({"from": edge[0], "to": edge[1],
                               "label": edge[2]})
        else:
            norm_edges.append(dict(edge))
    return validate_scene_spec({
        "kind": "diagram",
        "title": title,
        "nodes": norm_nodes,
        "edges": norm_edges,
    })
