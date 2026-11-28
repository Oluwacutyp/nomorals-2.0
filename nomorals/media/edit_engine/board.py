"""The whiteboard organ: Excalidraw (#18) + Tldraw (#44) as one
Devon-native board.

Both references run on the same system underneath: a store of typed
shape records, edited through an ordered op list, rendered
deterministically. Tldraw adds shape-to-shape bindings (arrow→shape)
and pages; Excalidraw's delta is a *style* — the hand-drawn
``sketch`` theme over the same primitives. So one document model
serves both: :class:`BoardDocument` holds the shapes + bindings +
ops, the ``clean`` theme is the Tldraw look, the ``sketch`` theme is
the Excalidraw look.

What Devon does NOT fake: live multi-user pointer collaboration —
60fps cursors, presence, other humans dragging shapes in real time
— is a web-frontend product her chat surfaces cannot host. What she
has instead is the system underneath, fully: the brain and the owner
manipulate the board as data (create / add / move / bind / undo),
and it renders to a PNG on demand or to a live page they can keep
drawing on. The page is the board's own editor — drag, add, delete,
edit text, download the board JSON — with no server and no external
URL.

The D1 document idiom (source + ordered op list, deterministic
render) is shared with :mod:`nomorals.media.edit_engine.document`:
the op list IS the replay — apply the same ops twice and the hash is
identical — and :meth:`BoardDocument.undo` / :meth:`redo` ride a
snapshot history over board dicts, the same shape as
:class:`UndoStack` (which is typed to ``Document``).

Coordinates are board units (pixels in board space, origin top-left).
A render fits the content bounds into the output size with padding,
so the canvas is effectively unbounded — the infinite-canvas half of
the references, without pretending the chat surface is a live canvas.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.failure_store import home_dir as _devon_home_dir
from ...core.ids import new_id
from ...media.typography import get_service, pil_font_for
from ...media_edit.videos import MediaEditError
from .document import _canonical

__all__ = [
    "THEMES",
    "BoardBinding",
    "BoardDocument",
    "BoardError",
    "BoardShape",
    "BoardStore",
    "SHAPE_TYPES",
    "board_from_diagram",
    "board_live_html",
    "default_boards_dir",
    "render_board",
    "write_board_page",
]

#: The typed shape records (Tldraw's geo/arrow/draw/text/image +
#: Excalidraw's sticky note).
SHAPE_TYPES = ("rect", "ellipse", "arrow", "text", "image", "sticky",
               "draw")

#: Render themes: ``clean`` is the Tldraw look, ``sketch`` is the
#: Excalidraw hand-drawn look.
THEMES = ("clean", "sketch")

#: Op kinds. Every mutation is one of these, recorded in order.
OP_KINDS = ("add_shape", "update_shape", "remove_shape", "bind",
            "unbind")

#: Default board canvas the diagram import scales 0..1 coords into.
_DEFAULT_BOARD_W = 1280.0
_DEFAULT_BOARD_H = 720.0

#: Default sizes for shapes added without explicit dimensions.
_DEFAULT_SIZE: dict[str, tuple[float, float]] = {
    "rect": (200.0, 120.0),
    "ellipse": (180.0, 120.0),
    "arrow": (200.0, 0.0),
    "text": (240.0, 60.0),
    "image": (320.0, 240.0),
    "sticky": (180.0, 180.0),
    "draw": (0.0, 0.0),
}

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

#: Props a shape type understands. Anything else is an authoring
#: error — the error names the prop, so the brain can fix it.
_SHAPE_PROPS: dict[str, tuple[str, ...]] = {
    "rect": ("text", "color", "fill"),
    "ellipse": ("text", "color", "fill"),
    "arrow": ("label", "color", "from_shape", "to_shape"),
    "text": ("text", "color", "size"),
    "image": ("path",),
    "sticky": ("text", "color"),
    "draw": ("points", "color"),
}


class BoardError(MediaEditError):
    """A board failed validation. The message names the field."""


def _fail(field: str, problem: str) -> None:
    raise BoardError(f"board: {field}: {problem}")


def _colour(value: Any, field: str) -> str:
    text = str(value).strip()
    if not _HEX_RE.match(text):
        _fail(field, f"colour must be #rrggbb, got {value!r}")
    return text.upper() if len(text) == 7 else text


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(field, f"needs a number, got {value!r}")
    if number != number or number in (float("inf"), float("-inf")):
        _fail(field, f"needs a finite number, got {value!r}")
    return number


def _points(value: Any, field: str) -> list[list[float]]:
    if not isinstance(value, list):
        _fail(field, "points must be a list of [x, y] pairs")
    out: list[list[float]] = []
    for i, pair in enumerate(value):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2):
            _fail(f"{field}[{i}]", "must be an [x, y] pair")
        out.append([_number(pair[0], f"{field}[{i}][0]"),
                    _number(pair[1], f"{field}[{i}][1]")])
    return out


def _clean_props(shape_type: str, props: Any,
                 field: str) -> dict[str, Any]:
    if props is None:
        props = {}
    if not isinstance(props, dict):
        _fail(field, "props must be a dict")
    allowed = _SHAPE_PROPS[shape_type]
    clean: dict[str, Any] = {}
    for key, value in props.items():
        if key not in allowed:
            _fail(f"{field}.{key}",
                  f"unknown prop for {shape_type!r} "
                  f"(allowed: {sorted(allowed)})")
        if key in ("color", "fill"):
            clean[key] = _colour(value, f"{field}.{key}")
        elif key == "points":
            clean[key] = _points(value, f"{field}.points")
        elif key in ("from_shape", "to_shape", "path", "label", "text"):
            clean[key] = str(value)
        elif key == "size":
            clean[key] = _number(value, f"{field}.size")
        else:
            clean[key] = value
    if shape_type == "draw" and not clean.get("points"):
        _fail(f"{field}.points", "a draw shape needs points")
    return clean


@dataclass
class BoardShape:
    """One typed shape record on the board.

    ``x``/``y`` is the top-left corner in board units; an arrow's
    ``w``/``h`` is the delta from its start to its end (so ``w``/``h``
    may be negative, and are 0 for a point); a draw shape's geometry
    lives in ``props["points"]``. ``z`` is the draw order (ties break
    by insertion order).
    """
    id: str
    type: str
    x: float
    y: float
    w: float
    h: float
    rotation: float = 0.0
    props: dict[str, Any] = field(default_factory=dict)
    z: int = 0

    def __post_init__(self) -> None:
        if not str(self.id):
            _fail("shape.id", "missing 'id'")
        self.id = str(self.id)
        if self.type not in SHAPE_TYPES:
            _fail("shape.type",
                  f"must be one of {SHAPE_TYPES}, got {self.type!r}")
        self.x = _number(self.x, "shape.x")
        self.y = _number(self.y, "shape.y")
        self.w = _number(self.w, "shape.w")
        self.h = _number(self.h, "shape.h")
        if self.type != "arrow" and (self.w < 0 or self.h < 0):
            _fail("shape.w/h", "must be >= 0 for a non-arrow shape")
        self.rotation = _number(self.rotation, "shape.rotation")
        self.props = _clean_props(self.type, self.props, "shape.props")
        self.z = int(self.z)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "x": self.x,
                "y": self.y, "w": self.w, "h": self.h,
                "rotation": self.rotation,
                "props": copy.deepcopy(self.props), "z": self.z}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoardShape:
        return cls(
            id=str(data.get("id", "")),
            type=str(data.get("type", "")),
            x=data.get("x", 0.0), y=data.get("y", 0.0),
            w=data.get("w", 0.0), h=data.get("h", 0.0),
            rotation=data.get("rotation", 0.0),
            props=copy.deepcopy(data.get("props") or {}),
            z=int(data.get("z", 0)))


@dataclass
class BoardBinding:
    """An arrow→shape binding: which arrow connects which shapes.

    The arrow shape itself survives a binding's removal (it becomes
    a free arrow at its last coordinates); deleting a *bound* shape
    removes the binding, never the arrow.
    """
    id: str
    arrow_shape: str
    from_shape: str
    to_shape: str
    kind: str = "arrow"
    label: str = ""

    def __post_init__(self) -> None:
        if not str(self.id):
            _fail("binding.id", "missing 'id'")
        self.id = str(self.id)
        for name in ("arrow_shape", "from_shape", "to_shape"):
            if not str(getattr(self, name)):
                _fail(f"binding.{name}", "missing shape id")
            setattr(self, name, str(getattr(self, name)))
        if self.kind != "arrow":
            _fail("binding.kind", f"must be 'arrow', got {self.kind!r}")
        self.label = str(self.label or "")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "arrow_shape": self.arrow_shape,
                "from_shape": self.from_shape, "to_shape": self.to_shape,
                "kind": self.kind, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoardBinding:
        return cls(
            id=str(data.get("id", "")),
            arrow_shape=str(data.get("arrow_shape", "")),
            from_shape=str(data.get("from_shape", "")),
            to_shape=str(data.get("to_shape", "")),
            kind=str(data.get("kind", "arrow")),
            label=str(data.get("label", "")))


def _auto_layout(nodes: list[dict[str, Any]],
                 edges: list[dict[str, Any]]
                 ) -> dict[str, tuple[float, float]]:
    """Deterministic layered positions for nodes without coordinates.

    The same layered idea as the scene organ's ``auto_layout`` —
    depth is the longest path from a source, x grows with depth, y
    spreads evenly inside the layer — written here so this L4 organ
    never imports the L5 scene organ. Pure: same input, same
    positions, every time.
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
    placed: dict[str, tuple[float, float]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        layer = layers[depth[node_id]]
        placed[node_id] = (
            (depth[node_id] + 1) / (max_depth + 2),
            (layer.index(node_id) + 1) / (len(layer) + 1))
    return placed


class BoardDocument:
    """One board — shapes + bindings + the ordered op list.

    The op list is the replay: :meth:`replay` applies the same ops
    to an empty board and lands on the same hash. Ids are generated
    at the op-construction site (``add_shape`` / ``bind``), never
    inside :meth:`apply`, so a replay is byte-identical — a
    re-generated id would break determinism.
    """

    def __init__(self, *, id: str = "", title: str = "",
                 theme: str = "clean",
                 shapes: list[BoardShape | dict[str, Any]] = (),
                 bindings: list[BoardBinding | dict[str, Any]] = (),
                 ops: list[dict[str, Any]] = (),
                 version: int = 1, doc_hash: str = "") -> None:
        if theme not in THEMES:
            _fail("board.theme",
                  f"must be one of {THEMES}, got {theme!r}")
        self.id = str(id) if id else new_id()
        self.title = str(title or "")
        self.theme = theme
        self.shapes: list[BoardShape] = [
            s if isinstance(s, BoardShape) else BoardShape.from_dict(dict(s))
            for s in shapes]
        self.bindings: list[BoardBinding] = [
            b if isinstance(b, BoardBinding)
            else BoardBinding.from_dict(dict(b))
            for b in bindings]
        self.ops: list[dict[str, Any]] = [copy.deepcopy(o) for o in ops]
        self.version = int(version)
        self.doc_hash = str(doc_hash or "")
        self._past: list[dict[str, Any]] = []
        self._future: list[dict[str, Any]] = []

    # -- construction --------------------------------------------------
    @classmethod
    def create(cls, title: str = "", *,
               theme: str = "clean") -> BoardDocument:
        """A new empty board."""
        return cls(title=title, theme=theme)

    # -- identity ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "theme": self.theme,
            "shapes": [s.to_dict() for s in self.shapes],
            "bindings": [b.to_dict() for b in self.bindings],
            "ops": copy.deepcopy(self.ops),
            "version": self.version,
            "doc_hash": self.doc_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoardDocument:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            theme=str(data.get("theme", "clean")),
            shapes=[BoardShape.from_dict(s)
                    for s in data.get("shapes", [])],
            bindings=[BoardBinding.from_dict(b)
                      for b in data.get("bindings", [])],
            ops=copy.deepcopy(data.get("ops") or []),
            version=int(data.get("version", 1)),
            doc_hash=str(data.get("doc_hash", "")))

    def compute_hash(self) -> str:
        """SHA-256 of the canonical document (hash field excluded)."""
        raw = self.to_dict()
        raw.pop("doc_hash", None)
        blob = json.dumps(_canonical(raw), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def seal(self) -> BoardDocument:
        """Bump version + stamp hash. Call after any edit."""
        self.version = int(self.version) + 1
        self.doc_hash = self.compute_hash()
        return self

    def copy(self) -> BoardDocument:
        return BoardDocument.from_dict(self.to_dict())

    # -- lookup --------------------------------------------------------
    def shape(self, shape_id: str) -> BoardShape | None:
        for shape in self.shapes:
            if shape.id == shape_id:
                return shape
        return None

    def binding(self, binding_id: str) -> BoardBinding | None:
        for binding in self.bindings:
            if binding.id == binding_id:
                return binding
        return None

    def binding_for_arrow(self, arrow_id: str) -> BoardBinding | None:
        for binding in self.bindings:
            if binding.arrow_shape == arrow_id:
                return binding
        return None

    def bounds(self) -> tuple[float, float, float, float]:
        """Content bounding box (minx, miny, maxx, maxy)."""
        if not self.shapes:
            return (0.0, 0.0, 0.0, 0.0)
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for shape in self.shapes:
            if shape.type == "draw":
                pts = shape.props.get("points") or []
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if xs:
                    min_x = min(min_x, min(xs))
                    max_x = max(max_x, max(xs))
                    min_y = min(min_y, min(ys))
                    max_y = max(max_y, max(ys))
                continue
            min_x = min(min_x, shape.x)
            min_y = min(min_y, shape.y)
            max_x = max(max_x, shape.x + shape.w)
            max_y = max(max_y, shape.y + shape.h)
        if min_x == float("inf"):
            return (0.0, 0.0, 0.0, 0.0)
        return (min_x, min_y, max_x, max_y)

    # -- ops -----------------------------------------------------------
    def apply(self, op: dict[str, Any]) -> dict[str, Any]:
        """Apply one op, record it, seal. Validates BEFORE mutating.

        Returns ``{"op": <the op>, "removed_bindings": [...]}`` — the
        removed bindings matter on ``remove_shape``: deleting a bound
        shape drops its bindings (the arrow shape itself survives as
        a free arrow at its last coordinates).
        """
        if not isinstance(op, dict):
            _fail("op", "must be a dict")
        kind = str(op.get("op", ""))
        if kind not in OP_KINDS:
            _fail("op.op",
                  f"must be one of {OP_KINDS}, got {kind!r}")
        # Validate the whole op against the current document first,
        # so a rejected op can never half-apply.
        if kind == "add_shape":
            shape = BoardShape.from_dict(dict(op.get("shape") or {}))
            if self.shape(shape.id) is not None:
                _fail("op.shape.id",
                      f"duplicate shape id {shape.id!r}")
        elif kind == "update_shape":
            target = self.shape(str(op.get("id", "")))
            if target is None:
                _fail("op.id", f"no shape {op.get('id')!r}")
            changes = op.get("changes")
            if not isinstance(changes, dict) or not changes:
                _fail("op.changes", "needs a non-empty changes dict")
            for key in changes:
                if key not in ("x", "y", "w", "h", "rotation", "z",
                               "props"):
                    _fail(f"op.changes.{key}", "not an updatable field")
            if "props" in changes:
                merged = dict(target.props)
                merged.update(changes["props"])
                _clean_props(target.type, merged, "op.changes.props")
            for key in ("x", "y", "w", "h", "rotation"):
                if key in changes:
                    _number(changes[key], f"op.changes.{key}")
        elif kind == "remove_shape":
            if self.shape(str(op.get("id", ""))) is None:
                _fail("op.id", f"no shape {op.get('id')!r}")
        elif kind == "bind":
            binding = BoardBinding.from_dict(dict(op.get("binding") or {}))
            arrow = BoardShape.from_dict(dict(op.get("arrow") or {}))
            if arrow.type != "arrow":
                _fail("op.arrow.type", "a binding needs an arrow shape")
            if self.shape(arrow.id) is not None:
                _fail("op.arrow.id",
                      f"duplicate shape id {arrow.id!r}")
            if self.binding(binding.id) is not None:
                _fail("op.binding.id",
                      f"duplicate binding id {binding.id!r}")
            if binding.arrow_shape != arrow.id:
                _fail("op.binding.arrow_shape",
                      "must name the arrow shape in this op")
            for end in (binding.from_shape, binding.to_shape):
                if self.shape(end) is None:
                    _fail("op.binding", f"no shape {end!r}")
        elif (kind == "unbind"
                and self.binding(str(op.get("id", ""))) is None):
            _fail("op.id", f"no binding {op.get('id')!r}")
        # Validated — snapshot history, then mutate.
        self._past.append(self.to_dict())
        self._future.clear()
        removed: list[str] = []
        if kind == "add_shape":
            shape = BoardShape.from_dict(dict(op.get("shape") or {}))
            self.shapes.append(shape)
        elif kind == "update_shape":
            target = self.shape(str(op.get("id", "")))
            assert target is not None  # validated above
            changes = dict(op.get("changes") or {})
            for key in ("x", "y", "w", "h", "rotation"):
                if key in changes:
                    setattr(target, key, float(changes[key]))
            if "z" in changes:
                target.z = int(changes["z"])
            if "props" in changes:
                merged = dict(target.props)
                merged.update(changes["props"])
                target.props = _clean_props(target.type, merged,
                                            "op.changes.props")
        elif kind == "remove_shape":
            shape_id = str(op.get("id", ""))
            self.shapes = [s for s in self.shapes if s.id != shape_id]
            kept: list[BoardBinding] = []
            for binding in self.bindings:
                if (binding.from_shape == shape_id
                        or binding.to_shape == shape_id
                        or binding.arrow_shape == shape_id):
                    removed.append(binding.id)
                else:
                    kept.append(binding)
            self.bindings = kept
        elif kind == "bind":
            binding = BoardBinding.from_dict(dict(op.get("binding") or {}))
            arrow = BoardShape.from_dict(dict(op.get("arrow") or {}))
            self.shapes.append(arrow)
            self.bindings.append(binding)
        elif kind == "unbind":
            binding_id = str(op.get("id", ""))
            self.bindings = [b for b in self.bindings
                             if b.id != binding_id]
        self.ops.append(copy.deepcopy(op))
        self.seal()
        return {"op": copy.deepcopy(op), "removed_bindings": removed}

    # -- convenience verbs (build the op, then apply it) --------------
    def add_shape(self, shape: dict[str, Any] | BoardShape,
                  **overrides: Any) -> BoardShape:
        """Add a shape. ``overrides`` fill in missing fields
        (``x``/``y``/``w``/``h``/``props``) — a shape added without
        dimensions takes the type default."""
        raw = (shape.to_dict() if isinstance(shape, BoardShape)
               else dict(shape))
        for key, value in overrides.items():
            if key == "props":
                merged = dict(raw.get("props") or {})
                merged.update(value)
                raw["props"] = merged
            else:
                raw.setdefault(key, value)
        if "id" not in raw or not raw["id"]:
            raw["id"] = new_id()
        shape_type = str(raw.get("type", ""))
        if shape_type in _DEFAULT_SIZE:
            default_w, default_h = _DEFAULT_SIZE[shape_type]
            raw.setdefault("w", default_w)
            raw.setdefault("h", default_h)
        raw.setdefault("x", 0.0)
        raw.setdefault("y", 0.0)
        raw.setdefault("z", len(self.shapes))
        created = BoardShape.from_dict(raw)
        self.apply({"op": "add_shape", "shape": created.to_dict()})
        created = self.shape(created.id)
        assert created is not None
        return created

    def update_shape(self, shape_id: str, **changes: Any) -> BoardShape:
        """Update a shape's fields (``props`` merges, never replaces)."""
        self.apply({"op": "update_shape", "id": shape_id,
                    "changes": copy.deepcopy(changes)})
        updated = self.shape(shape_id)
        assert updated is not None
        return updated

    def remove_shape(self, shape_id: str) -> dict[str, Any]:
        """Remove a shape; its bindings drop with it (reported)."""
        return self.apply({"op": "remove_shape", "id": shape_id})

    def bind(self, from_shape: str, to_shape: str, *,
             label: str = "") -> BoardBinding:
        """Connect two shapes with a new arrow + binding.

        The arrow's stored geometry runs from the from-shape's centre
        to the to-shape's centre; the renderer draws it edge to edge.
        """
        if self.shape(from_shape) is None:
            _fail("bind.from_shape", f"no shape {from_shape!r}")
        if self.shape(to_shape) is None:
            _fail("bind.to_shape", f"no shape {to_shape!r}")
        src = self.shape(from_shape)
        dst = self.shape(to_shape)
        assert src is not None and dst is not None
        arrow_id = new_id()
        binding_id = new_id()
        start_x = src.x + src.w / 2
        start_y = src.y + src.h / 2
        end_x = dst.x + dst.w / 2
        end_y = dst.y + dst.h / 2
        arrow = BoardShape(
            id=arrow_id, type="arrow", x=start_x, y=start_y,
            w=end_x - start_x, h=end_y - start_y,
            props={"label": str(label or "")}, z=len(self.shapes))
        binding = BoardBinding(
            id=binding_id, arrow_shape=arrow_id,
            from_shape=from_shape, to_shape=to_shape,
            label=str(label or ""))
        self.apply({"op": "bind", "binding": binding.to_dict(),
                    "arrow": arrow.to_dict()})
        created = self.binding(binding_id)
        assert created is not None
        return created

    def unbind(self, binding_id: str) -> None:
        """Drop a binding; the arrow shape stays as a free arrow."""
        self.apply({"op": "unbind", "id": binding_id})

    # -- undo / redo ---------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    @property
    def history_depth(self) -> int:
        return len(self._past)

    def undo(self) -> BoardDocument:
        """Step one op back. Raises :class:`BoardError` at the start."""
        if not self._past:
            raise BoardError("board: nothing to undo")
        self._future.append(self.to_dict())
        restored = BoardDocument.from_dict(self._past.pop())
        self._restore(restored)
        return self

    def redo(self) -> BoardDocument:
        """Step one undone op forward."""
        if not self._future:
            raise BoardError("board: nothing to redo")
        self._past.append(self.to_dict())
        restored = BoardDocument.from_dict(self._future.pop())
        self._restore(restored)
        return self

    def _restore(self, other: BoardDocument) -> None:
        self.id = other.id
        self.title = other.title
        self.theme = other.theme
        self.shapes = other.shapes
        self.bindings = other.bindings
        self.ops = other.ops
        self.version = other.version
        self.doc_hash = other.doc_hash

    # -- durable history ---------------------------------------------
    def history_dict(self) -> dict[str, Any]:
        """Undo/redo stacks for the store.

        History is NOT part of the document identity — it never
        enters :meth:`compute_hash` — but it must survive a save/load
        cycle, or undo would die the moment the board is reloaded
        (every tool/CLI/chat call loads fresh). The store writes it
        beside the board, and :meth:`restore_history` reads it back.
        """
        return {"past": copy.deepcopy(self._past),
                "future": copy.deepcopy(self._future)}

    def restore_history(self, data: Any) -> None:
        """Restore undo/redo stacks saved by :meth:`history_dict`."""
        if not isinstance(data, dict):
            return
        past = data.get("past")
        future = data.get("future")
        if isinstance(past, list):
            self._past = [copy.deepcopy(d) for d in past
                          if isinstance(d, dict)]
        if isinstance(future, list):
            self._future = [copy.deepcopy(d) for d in future
                            if isinstance(d, dict)]

    # -- replay --------------------------------------------------------
    @classmethod
    def replay(cls, ops: list[dict[str, Any]], *, title: str = "",
               theme: str = "clean",
               board_id: str = "") -> BoardDocument:
        """An empty board + the ops, in order. The hash of the result
        is the replay proof: same ops, same board, every time."""
        doc = cls(id=board_id, title=title, theme=theme)
        for op in ops:
            doc.apply(copy.deepcopy(op))
        return doc


def board_from_diagram(spec: dict[str, Any], *, title: str = "",
                       theme: str = "clean") -> BoardDocument:
    """A scene-organ diagram spec (plain dict) → a board.

    Nodes become rects (circles become ellipses) with their labels;
    edges become arrows + bindings. Node coordinates are 0..1 and
    scale into board space; nodes without coordinates take the
    deterministic layered layout. The spec dict is duck-typed — this
    L4 organ never imports the L5 scene organ.
    """
    if not isinstance(spec, dict):
        raise BoardError("board: diagram spec must be a dict")
    if str(spec.get("kind", "")) != "diagram":
        raise BoardError(
            "board: board_from_diagram needs a diagram spec, got "
            f"{spec.get('kind')!r}")
    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise BoardError("board: diagram spec needs a non-empty "
                         "'nodes' list")
    raw_edges = spec.get("edges") or []
    if not isinstance(raw_edges, list):
        raise BoardError("board: diagram spec 'edges' must be a list")
    laid_out = _auto_layout(
        [dict(n) for n in raw_nodes if isinstance(n, dict)],
        [dict(e) for e in raw_edges if isinstance(e, dict)])
    doc = BoardDocument.create(title or str(spec.get("title", "")),
                               theme=theme)
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise BoardError("board: diagram node must be a dict")
        node_id = str(raw.get("id", ""))
        if not node_id:
            raise BoardError("board: diagram node missing 'id'")
        shape_type = ("ellipse" if str(raw.get("shape", "")) == "circle"
                      else "rect")
        width, height = _DEFAULT_SIZE[shape_type]
        raw_x, raw_y = raw.get("x"), raw.get("y")
        if raw_x is None or raw_y is None:
            norm_x, norm_y = laid_out[node_id]
        else:
            norm_x, norm_y = float(raw_x), float(raw_y)
        doc.add_shape({
            "id": node_id, "type": shape_type,
            "x": norm_x * _DEFAULT_BOARD_W - width / 2,
            "y": norm_y * _DEFAULT_BOARD_H - height / 2,
            "w": width, "h": height,
            "props": {"text": str(raw.get("label", node_id))},
        })
    for raw in raw_edges:
        if not isinstance(raw, dict):
            raise BoardError("board: diagram edge must be a dict")
        src, dst = str(raw.get("from", "")), str(raw.get("to", ""))
        doc.bind(src, dst, label=str(raw.get("label", "")))
    return doc


# ── durable store ────────────────────────────────────────────────────


def default_boards_dir(home: Path | str | None = None) -> Path:
    """``<home>/boards`` — where boards persist."""
    base = Path(home) if home is not None else Path(_devon_home_dir())
    return base / "boards"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


class BoardStore:
    """Durable boards: one atomic JSON file per board.

    A corrupt or missing file loads as ``None`` — a board store that
    cannot read a board says so, never raises, never invents one.
    """

    def __init__(self, home: Path | str | None = None, *,
                 dir: Path | str | None = None) -> None:
        if dir is not None:
            self.dir = Path(dir)
        else:
            self.dir = default_boards_dir(home)

    def _path(self, board_id: str) -> Path:
        return self.dir / f"{board_id}.json"

    def save(self, doc: BoardDocument) -> str:
        """Persist a board (+ its undo history); return the file path."""
        from datetime import UTC, datetime

        payload = {
            "board": doc.to_dict(),
            "history": doc.history_dict(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path = self._path(doc.id)
        _atomic_write_json(path, payload)
        return str(path)

    def load(self, board_id: str) -> BoardDocument | None:
        """Load a board by id, or ``None`` when it is not there."""
        try:
            raw = json.loads(self._path(board_id).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or not isinstance(
                raw.get("board"), dict):
            return None
        try:
            doc = BoardDocument.from_dict(raw["board"])
        except (BoardError, KeyError, TypeError, ValueError):
            return None
        doc.restore_history(raw.get("history"))
        return doc

    def list_boards(self) -> list[dict[str, Any]]:
        """One row per stored board, newest first."""
        rows: list[dict[str, Any]] = []
        if not self.dir.is_dir():
            return rows
        for path in sorted(self.dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            board = raw.get("board") if isinstance(raw, dict) else None
            if not isinstance(board, dict):
                continue
            rows.append({
                "id": str(board.get("id", path.stem)),
                "title": str(board.get("title", "")),
                "theme": str(board.get("theme", "clean")),
                "shapes": len(board.get("shapes") or []),
                "bindings": len(board.get("bindings") or []),
                "version": int(board.get("version", 1)),
                "updated_at": str(raw.get("updated_at", "")),
            })
        rows.sort(key=lambda r: (r["updated_at"], r["id"]),
                  reverse=True)
        return rows

    def delete(self, board_id: str) -> bool:
        """Delete a board file. ``False`` when it was not there."""
        try:
            self._path(board_id).unlink()
        except OSError:
            return False
        return True

    def matches(self, ref: str) -> list[str]:
        """Stored board ids matching ``ref`` exactly or by prefix."""
        ref = (ref or "").strip()
        if not ref:
            return []
        ids = [row["id"] for row in self.list_boards()]
        exact = [i for i in ids if i == ref]
        if exact:
            return exact
        return [i for i in ids if i.startswith(ref)]

    def find(self, ref: str) -> BoardDocument | None:
        """Load a board by exact id or a unique id prefix."""
        found = self.matches(ref)
        if len(found) != 1:
            return None
        return self.load(found[0])


def _seeded_rng(seed_text: str) -> random.Random:
    """Deterministic jitter: the same shape id, the same wobble."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:4], "big"))


# ── render rung: deterministic PNG ───────────────────────────────────

_log = logging.getLogger(__name__)


def _load_font(size: int, *, bold: bool = False):
    """Board text wears the card context (Inter): the whiteboard
    is house text, not the old DejaVu door (gapfix 164). A total
    miss is logged and the Pillow bitmap default carries the
    label."""
    from PIL import ImageFont

    try:
        return pil_font_for(get_service(), "card", max(8, int(size)),
                            weight=700 if bold else 400)
    except Exception as exc:  # noqa: BLE001 - named miss
        _log.warning("board font could not resolve: %s", exc)
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap by pixel width."""
    lines: list[str] = []
    for raw in (text or "").split("\n"):
        words = raw.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if font.getlength(trial) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def _arrow_endpoints(doc: BoardDocument, shape: BoardShape,
                     to_screen) -> tuple:
    """Arrow start/end in screen coords — edge to edge when bound."""
    binding = doc.binding_for_arrow(shape.id)
    if binding is None:
        return (to_screen(shape.x, shape.y),
                to_screen(shape.x + shape.w, shape.y + shape.h))
    src = doc.shape(binding.from_shape)
    dst = doc.shape(binding.to_shape)
    if src is None or dst is None:
        return (to_screen(shape.x, shape.y),
                to_screen(shape.x + shape.w, shape.y + shape.h))
    sx, sy = src.x + src.w / 2, src.y + src.h / 2
    ex, ey = dst.x + dst.w / 2, dst.y + dst.h / 2
    # Walk from each centre toward the other until the rect edge.
    def _edge(cx: float, cy: float, w: float, h: float,
              tx: float, ty: float) -> tuple[float, float]:
        dx, dy = tx - cx, ty - cy
        if dx == 0 and dy == 0:
            return (cx, cy)
        scale = min(
            (w / 2) / abs(dx) if dx else float("inf"),
            (h / 2) / abs(dy) if dy else float("inf"))
        return (cx + dx * scale, cy + dy * scale)

    start = _edge(sx, sy, src.w, src.h, ex, ey)
    end = _edge(ex, ey, dst.w, dst.h, sx, sy)
    return (to_screen(*start), to_screen(*end))


def _draw_arrowhead(draw, end: tuple, angle: float, color: str,
                    size: float = 14.0) -> None:
    import math

    x2, y2 = end
    head = [(x2, y2),
            (x2 - size * math.cos(angle - 0.4),
             y2 - size * math.sin(angle - 0.4)),
            (x2 - size * math.cos(angle + 0.4),
             y2 - size * math.sin(angle + 0.4))]
    draw.polygon(head, fill=color)


def _rough_line(draw, points: list, color: str, width: int,
                rng: random.Random) -> None:
    """A hand-drawn line: two jittered passes over the same path."""
    for _ in range(2):
        wobbled = [
            (x + rng.uniform(-2.0, 2.0), y + rng.uniform(-2.0, 2.0))
            for x, y in points]
        draw.line(wobbled, fill=color, width=width, joint="curve")


def _rough_rect(draw, box: tuple, color: str, width: int,
                rng: random.Random) -> None:
    """A hand-drawn rectangle: four jittered double edges."""
    x0, y0, x1, y1 = box
    def _mid(a: float, b: float) -> float:
        return (a + b) / 2

    edges = [
        [(x0, y0), (_mid(x0, x1), y0), (x1, y0)],
        [(x1, y0), (x1, _mid(y0, y1)), (x1, y1)],
        [(x1, y1), (_mid(x0, x1), y1), (x0, y1)],
        [(x0, y1), (x0, _mid(y0, y1)), (x0, y0)],
    ]
    for edge in edges:
        _rough_line(draw, edge, color, width, rng)


def _rough_ellipse(draw, box: tuple, color: str, width: int,
                   rng: random.Random) -> None:
    """A hand-drawn ellipse: a jittered 28-point closed polyline."""
    import math

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    points = []
    for i in range(28):
        angle = 2 * math.pi * i / 28
        points.append((
            cx + rx * math.cos(angle) + rng.uniform(-2.0, 2.0),
            cy + ry * math.sin(angle) + rng.uniform(-2.0, 2.0)))
    points.append(points[0])
    _rough_line(draw, points, color, width, rng)


def render_board(doc: BoardDocument, *, theme: str | None = None,
                 size: tuple[int, int] = (1280, 720),
                 path: str | Path | None = None,
                 out_dir: str | Path | None = None) -> dict[str, Any]:
    """Render a board to a PNG. Returns the render record.

    Deterministic: the same board renders the same bytes — the
    ``render_hash`` is the SHA-256 of the PNG file, so a replay
    proof is a byte comparison. The sketch theme's wobble is seeded
    from the shape id, never from the clock or the global RNG.
    """
    from PIL import Image, ImageDraw

    use_theme = theme or doc.theme
    if use_theme not in THEMES:
        raise BoardError(
            f"board: theme must be one of {THEMES}, got {use_theme!r}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise BoardError("board: render size must be positive")
    pad = 48
    min_x, min_y, max_x, max_y = doc.bounds()
    if not doc.shapes:
        min_x, min_y, max_x, max_y = 0.0, 0.0, _DEFAULT_BOARD_W, \
            _DEFAULT_BOARD_H
    content_w = max(1.0, max_x - min_x)
    content_h = max(1.0, max_y - min_y)
    scale = min((width - 2 * pad) / content_w,
                (height - 2 * pad) / content_h)
    off_x = (width - content_w * scale) / 2 - min_x * scale
    off_y = (height - content_h * scale) / 2 - min_y * scale

    def to_screen(x: float, y: float) -> tuple[float, float]:
        return (off_x + x * scale, off_y + y * scale)

    image = Image.new("RGB", (width, height), "#FFFFFF")
    base = ImageDraw.Draw(image, "RGBA")
    if doc.title:
        title_font = _load_font(int(26 * scale) + 10, bold=True)
        base.text((24, 16), doc.title, fill="#1E1E2E",
                  font=title_font)
    missing_images: list[str] = []
    ordered = sorted(enumerate(doc.shapes),
                     key=lambda pair: (pair[1].z, pair[0]))
    for _, shape in ordered:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        color = shape.props.get("color", "#1E1E2E")
        if shape.type == "arrow":
            start, end = _arrow_endpoints(doc, shape, to_screen)
            import math

            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            if use_theme == "sketch":
                rng = _seeded_rng(shape.id)
                mid = ((start[0] + end[0]) / 2 + rng.uniform(-3, 3),
                       (start[1] + end[1]) / 2 + rng.uniform(-3, 3))
                _rough_line(draw, [start, mid, end], color, 3, rng)
            else:
                draw.line([start, end], fill=color, width=3)
            _draw_arrowhead(draw, end, angle, color)
            label = shape.props.get("label", "")
            if label:
                font = _load_font(int(16 * scale) + 8)
                mid_x = (start[0] + end[0]) / 2 + 6
                mid_y = (start[1] + end[1]) / 2 - 10
                draw.text((mid_x, mid_y), str(label), fill="#64748B",
                          font=font)
        elif shape.type == "draw":
            pts = [to_screen(px, py)
                   for px, py in (shape.props.get("points") or [])]
            if len(pts) > 1:
                if use_theme == "sketch":
                    _rough_line(draw, pts, color, 3,
                                _seeded_rng(shape.id))
                else:
                    draw.line(pts, fill=color, width=3, joint="curve")
        else:
            x0, y0 = to_screen(shape.x, shape.y)
            x1, y1 = to_screen(shape.x + shape.w,
                               shape.y + shape.h)
            box = (x0, y0, x1, y1)
            fill = shape.props.get("fill", "")
            if shape.type == "sticky":
                fill = shape.props.get("fill", "#FEF08A")
                if use_theme == "sketch":
                    rng = _seeded_rng(shape.id)
                    _rough_rect(draw, box, "#EAB308", 2, rng)
                    draw.rectangle(box, fill=fill + "CC")
                else:
                    draw.rectangle(box, fill=fill, outline="#EAB308",
                                   width=2)
            elif shape.type == "rect":
                if fill:
                    draw.rectangle(box, fill=fill)
                if use_theme == "sketch":
                    _rough_rect(draw, box, color, 3,
                                _seeded_rng(shape.id))
                else:
                    draw.rectangle(box, outline=color, width=3)
            elif shape.type == "ellipse":
                if fill:
                    draw.ellipse(box, fill=fill)
                if use_theme == "sketch":
                    _rough_ellipse(draw, box, color, 3,
                                   _seeded_rng(shape.id))
                else:
                    draw.ellipse(box, outline=color, width=3)
            elif shape.type == "image":
                src = str(shape.props.get("path", ""))
                try:
                    pic = Image.open(src).convert("RGB")
                    pic.thumbnail((max(1, int(x1 - x0)),
                                   max(1, int(y1 - y0))))
                    layer.paste(pic, (int(x0), int(y0)))
                except (OSError, ValueError):
                    missing_images.append(src)
                    draw.rectangle(box, outline="#EF4444", width=2)
                    font = _load_font(14)
                    draw.text((x0 + 8, y0 + 8), "missing image",
                              fill="#EF4444", font=font)
                    draw.text((x0 + 8, y0 + 28),
                              Path(src).name[:30], fill="#64748B",
                              font=font)
            text = str(shape.props.get("text", ""))
            if text and shape.type in ("rect", "ellipse", "sticky",
                                       "text"):
                text_size = int(
                    float(shape.props.get("size", 20)) * scale) + 8
                font = _load_font(text_size,
                                  bold=shape.type == "sticky")
                max_w = max(10.0, (x1 - x0) - 16)
                lines = _wrap_text(text, font, max_w)
                line_h = int(text_size * 1.3) + 2
                total_h = line_h * len(lines)
                start_y = y0 + max(0.0, (y1 - y0 - total_h) / 2)
                for i, line in enumerate(lines):
                    line_w = font.getlength(line)
                    lx = x0 + max(0.0, (x1 - x0 - line_w) / 2)
                    draw.text((lx, start_y + i * line_h), line,
                              fill=color if shape.type == "text"
                              else "#1E1E2E", font=font)
        if shape.rotation:
            cx, cy = to_screen(shape.x + shape.w / 2,
                               shape.y + shape.h / 2)
            layer = layer.rotate(-shape.rotation, center=(cx, cy))
        image = Image.alpha_composite(image.convert("RGBA"), layer)
    if path is not None:
        target = Path(path)
    else:
        out = Path(out_dir) if out_dir is not None else Path.cwd()
        out.mkdir(parents=True, exist_ok=True)
        target = out / (f"board-{doc.id[:8]}-{doc.doc_hash[:8]}.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, format="PNG")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"path": str(target), "theme": use_theme,
            "size": [width, height], "shapes": len(doc.shapes),
            "bindings": len(doc.bindings),
            "missing_images": missing_images, "render_hash": digest}


# ── live page: the board's own editor ────────────────────────────────


_BOARD_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
body{margin:0;background:#f1f5f9;font-family:sans-serif}
#toolbar{padding:8px;background:#1e1e2e;color:#f8fafc;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
#toolbar button{padding:6px 10px;border:0;border-radius:6px;background:#38bdf8;color:#0f172a;cursor:pointer;font-weight:600}
#toolbar button.alt{background:#fbbf24}
#toolbar span{margin-left:8px;color:#94a3b8}
canvas{display:block;background:#ffffff;margin:12px auto;border:1px solid #cbd5e1}
#hint{text-align:center;color:#64748b;padding:0 12px 16px}
</style></head><body>
<div id="toolbar">
<button data-add="rect">+ Rect</button>
<button data-add="ellipse">+ Ellipse</button>
<button data-add="arrow">+ Arrow</button>
<button data-add="text">+ Text</button>
<button data-add="sticky">+ Sticky</button>
<button data-add="draw">+ Draw</button>
<button id="del" class="alt">Delete selected</button>
<button id="dl">Download board JSON</button>
<button id="png">Export PNG</button>
<span id="status">board __BOARD_ID__ — __THEME__ theme</span>
</div>
<canvas id="board" width="1280" height="720"></canvas>
<div id="hint">Drag a shape to move it. Double-click text, a sticky or a label to edit it. Draw mode: press, scribble, release. Edits live in this page — Download board JSON to save them.</div>
<script>
const board = __BOARD_JSON__;
const THEME = "__THEME__";
const cv = document.getElementById('board');
const ctx = cv.getContext('2d');
let selected = null, dragOff = null, drawPts = null, drawMode = false;
function hashSeed(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function view(){let mnx=1e9,mny=1e9,mxx=-1e9,mxy=-1e9;for(const s of board.shapes){if(s.type==='draw'){for(const p of (s.props.points||[])){mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1])}}else{mnx=Math.min(mnx,s.x);mny=Math.min(mny,s.y);mxx=Math.max(mxx,s.x+s.w);mxy=Math.max(mxy,s.y+s.h)}}if(!board.shapes.length){mnx=0;mny=0;mxx=1280;mxy=720}const sc=Math.min((1280-96)/Math.max(1,mxx-mnx),(720-96)/Math.max(1,mxy-mny));return{sc:sc,ox:(1280-(mxx-mnx)*sc)/2-mnx*sc,oy:(720-(mxy-mny)*sc)/2-mny*sc}}
const V=view();
function SX(x){return V.ox+x*V.sc} function SY(y){return V.oy+y*V.sc}
function BX(px){return (px-V.ox)/V.sc} function BY(py){return (py-V.oy)/V.sc}
function roughLine(pts,color,w,rng){for(let pass=0;pass<2;pass++){ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();pts.forEach((p,i)=>{const x=p[0]+(rng()-0.5)*4,y=p[1]+(rng()-0.5)*4;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}}
function drawShape(s){
  const col=s.props.color||'#1e1e2e';
  if(s.type==='arrow'){
    let a=SX(s.x),b=SY(s.y),c=SX(s.x+s.w),d=SY(s.y+s.h);
    const bind=board.bindings.find(b=>b.arrow_shape===s.id);
    if(bind){const f=board.shapes.find(x=>x.id===bind.from_shape),t=board.shapes.find(x=>x.id===bind.to_shape);if(f&&t){a=SX(f.x+f.w/2);b=SY(f.y+f.h/2);c=SX(t.x+t.w/2);d=SY(t.y+t.h/2)}}
    const ang=Math.atan2(d-b,c-a);
    if(THEME==='sketch'){roughLine([[a,b],[(a+c)/2,(b+d)/2],[c,d]],col,3,mulberry32(hashSeed(s.id)))}
    else{ctx.strokeStyle=col;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke()}
    ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(c,d);ctx.lineTo(c-14*Math.cos(ang-0.4),d-14*Math.sin(ang-0.4));ctx.lineTo(c-14*Math.cos(ang+0.4),d-14*Math.sin(ang+0.4));ctx.closePath();ctx.fill();
    const lab=s.props.label||'';if(lab){ctx.fillStyle='#64748b';ctx.font='16px sans-serif';ctx.fillText(lab,(a+c)/2+6,(b+d)/2-10)}
    return;
  }
  if(s.type==='draw'){const pts=(s.props.points||[]).map(p=>[SX(p[0]),SY(p[1])]);if(pts.length>1){if(THEME==='sketch')roughLine(pts,col,3,mulberry32(hashSeed(s.id)));else{ctx.strokeStyle=col;ctx.lineWidth=3;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke()}}return}
  const x=SX(s.x),y=SY(s.y),w=s.w*V.sc,h=s.h*V.sc;
  ctx.save();ctx.translate(x+w/2,y+h/2);if(s.rotation)ctx.rotate(s.rotation*Math.PI/180);ctx.translate(-w/2,-h/2);
  if(s.type==='sticky'){ctx.fillStyle=s.props.fill||'#fef08a';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#eab308';ctx.lineWidth=2;ctx.strokeRect(0,0,w,h)}
  else if(s.type==='rect'){if(s.props.fill){ctx.fillStyle=s.props.fill;ctx.fillRect(0,0,w,h)}if(THEME==='sketch'){roughLine([[0,0],[w/2,0],[w,0]],col,3,mulberry32(hashSeed(s.id+'a')));roughLine([[w,0],[w,h/2],[w,h]],col,3,mulberry32(hashSeed(s.id+'b')));roughLine([[w,h],[w/2,h],[0,h]],col,3,mulberry32(hashSeed(s.id+'c')));roughLine([[0,h],[0,h/2],[0,0]],col,3,mulberry32(hashSeed(s.id+'d')))}else{ctx.strokeStyle=col;ctx.lineWidth=3;ctx.strokeRect(0,0,w,h)}}
  else if(s.type==='ellipse'){ctx.beginPath();ctx.ellipse(w/2,h/2,w/2,h/2,0,0,Math.PI*2);if(s.props.fill){ctx.fillStyle=s.props.fill;ctx.fill()}ctx.strokeStyle=col;ctx.lineWidth=3;ctx.stroke()}
  else if(s.type==='image'){ctx.strokeStyle='#ef4444';ctx.lineWidth=2;ctx.strokeRect(0,0,w,h);ctx.fillStyle='#ef4444';ctx.font='14px sans-serif';ctx.fillText('image — see board JSON',8,24)}
  const text=s.props.text||'';
  if(text&&(s.type==='rect'||s.type==='ellipse'||s.type==='sticky'||s.type==='text')){
    ctx.fillStyle=s.type==='text'?col:'#1e1e2e';ctx.font=(s.type==='sticky'?'bold ':'')+Math.max(12,18*V.sc)+'px sans-serif';ctx.textAlign='center';
    const words=text.split(' ');let line='',lines=[];for(const word of words){const t=line?line+' '+word:word;if(ctx.measureText(t).width>w-16&&line){lines.push(line);line=word}else line=t}lines.push(line);
    const lh=Math.max(14,20*V.sc);lines.forEach((ln,i)=>ctx.fillText(ln,w/2,h/2+(i-(lines.length-1)/2)*lh+6));ctx.textAlign='left';
  }
  ctx.restore();
  if(selected===s.id){ctx.strokeStyle='#38bdf8';ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.strokeRect(x-4,y-4,w+8,h+8);ctx.setLineDash([])}
}
function draw(){
  ctx.fillStyle='#ffffff';ctx.fillRect(0,0,1280,720);
  ctx.fillStyle='#1e1e2e';ctx.font='bold 26px sans-serif';ctx.fillText(board.title||'board',24,40);
  const ordered=[...board.shapes].sort((a,b)=>(a.z-b.z));
  for(const s of ordered)drawShape(s);
}
function hit(px,py){
  const ordered=[...board.shapes].sort((a,b)=>(b.z-a.z));
  for(const s of ordered){
    const x=BX(px),y=BY(py);
    if(s.type==='draw'){for(const p of (s.props.points||[])){if(Math.hypot(p[0]-x,p[1]-y)<12/V.sc)return s}continue}
    if(s.type==='arrow'){continue}
    if(x>=s.x&&x<=s.x+s.w&&y>=s.y&&y<=s.y+s.h)return s;
  }
  return null;
}
function pos(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
cv.addEventListener('mousedown',e=>{
  const [px,py]=pos(e);
  if(drawMode){drawPts=[[BX(px),BY(py)]];return}
  const s=hit(px,py);selected=s?s.id:null;
  if(s)dragOff=[BX(px)-s.x,BY(py)-s.y];
  draw();
});
cv.addEventListener('mousemove',e=>{
  const [px,py]=pos(e);
  if(drawPts){drawPts.push([BX(px),BY(py)]);draw();ctx.strokeStyle='#1e1e2e';ctx.lineWidth=3;ctx.beginPath();drawPts.forEach((p,i)=>{const q=[SX(p[0]),SY(p[1])];i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke();return}
  if(selected&&dragOff){const s=board.shapes.find(x=>x.id===selected);if(s){s.x=BX(px)-dragOff[0];s.y=BY(py)-dragOff[1];draw()}}
});
cv.addEventListener('mouseup',e=>{
  if(drawPts&&drawPts.length>1){
    const id='s'+Math.random().toString(36).slice(2,10);
    board.shapes.push({id:id,type:'draw',x:0,y:0,w:0,h:0,rotation:0,props:{points:drawPts},z:board.shapes.length});
  }
  drawPts=null;dragOff=null;draw();
});
cv.addEventListener('dblclick',e=>{
  const [px,py]=pos(e);const s=hit(px,py);if(!s)return;
  const cur=s.props.text||s.props.label||'';
  const next=prompt('Edit text:',cur);if(next===null)return;
  if(s.type==='arrow')s.props.label=next;else s.props.text=next;
  draw();
});
document.querySelectorAll('[data-add]').forEach(btn=>btn.addEventListener('click',()=>{
  const type=btn.dataset.add;
  if(type==='draw'){drawMode=!drawMode;btn.style.background=drawMode?'#f472b6':'';document.getElementById('status').textContent=drawMode?'draw mode — press, scribble, release':'board '+board.id.slice(0,8)+' — '+THEME+' theme';return}
  const sizes={rect:[200,120],ellipse:[180,120],arrow:[200,0],text:[240,60],sticky:[180,180]};
  const wh=sizes[type]||[200,120];
  const id='s'+Math.random().toString(36).slice(2,10);
  const shape={id:id,type:type,x:60+Math.random()*200,y:120+Math.random()*200,w:wh[0],h:wh[1],rotation:0,props:{},z:board.shapes.length};
  if(type==='text'||type==='rect'||type==='ellipse'||type==='sticky')shape.props.text=type==='sticky'?'note':type;
  if(type==='arrow')shape.props.label='';
  board.shapes.push(shape);selected=id;draw();
}));
function deleteSelected(){
  if(!selected)return;
  board.shapes=board.shapes.filter(s=>s.id!==selected);
  board.bindings=board.bindings.filter(b=>b.from_shape!==selected&&b.to_shape!==selected&&b.arrow_shape!==selected);
  selected=null;draw();
}
document.getElementById('del').addEventListener('click',deleteSelected);
document.addEventListener('keydown',e=>{if((e.key==='Delete'||e.key==='Backspace')&&document.activeElement===document.body)deleteSelected()});
document.getElementById('dl').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify(board,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='board-'+board.id.slice(0,8)+'.json';a.click();
});
document.getElementById('png').addEventListener('click',()=>{
  const a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download='board-'+board.id.slice(0,8)+'.png';a.click();
});
draw();
</script></body></html>
"""


def board_live_html(doc: BoardDocument, *,
                    theme: str | None = None) -> str:
    """A board → a self-contained editable HTML page (a string).

    The page draws the board and edits it: drag shapes, double-click
    to edit text, add shapes from the toolbar, draw freehand, delete,
    download the board JSON or export a PNG. No server, no external
    URL — edits live in the page, and the download is the honest save
    path.
    """
    import html as _html

    use_theme = theme or doc.theme
    if use_theme not in THEMES:
        raise BoardError(
            f"board: theme must be one of {THEMES}, got {use_theme!r}")
    payload = json.dumps(doc.to_dict()).replace("</", "<\\/")
    page = _BOARD_PAGE.replace("__TITLE__",
                               _html.escape(doc.title or "board"))
    page = page.replace("__BOARD_ID__", doc.id[:8])
    page = page.replace("__THEME__", use_theme)
    page = page.replace("__BOARD_JSON__", payload)
    return page


def write_board_page(doc: BoardDocument, path: str | Path, *,
                     theme: str | None = None) -> str:
    """Write the live page to ``path``; return the path as a string."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(board_live_html(doc, theme=theme),
                      encoding="utf-8")
    return str(target)
