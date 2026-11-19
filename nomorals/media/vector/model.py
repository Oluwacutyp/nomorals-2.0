"""Vector document model — layers, nodes, fills, gradients.

Port of the *design* of the upstream ``crates/doc`` document model
(layers of nodes over artboards, graphic styles, gradient defs) into
Devon's Python. A :class:`VectorDocument` is a canvas (width/height/
viewBox) holding :class:`Layer` objects; each layer holds nodes —
paths, rects, circles, ellipses, lines, polygons, text, images,
groups, and ``<use>`` references. Nodes carry fill/stroke/opacity/
transform; every node converts to a :class:`~.geom.Path` so the
renderer and the boolean engine share one geometry. The model is
format-neutral: :mod:`nomorals.media.vector.svg` owns the SVG wire
form, this module owns the document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geom import Affine, Path, parse_path_data

__all__ = [
    "Defs", "EllipseNode", "Fill", "Gradient", "GradientStop",
    "GroupNode", "ImageNode", "Layer", "LineNode", "PolyNode",
    "CircleNode", "PathNode", "RectNode", "Stroke", "TextNode",
    "UseNode", "VectorDocument", "VectorNode",
]

#: Node kinds the model knows.
NODE_KINDS = (
    "path", "rect", "circle", "ellipse", "line", "polyline",
    "polygon", "text", "image", "group", "use",
)


@dataclass
class Fill:
    """A fill: solid colour, gradient reference, or none."""

    kind: str = "solid"  # solid | gradient | none
    color: str | None = None
    gradient: str | None = None
    opacity: float = 1.0

    def canonical(self) -> tuple:
        return (self.kind, self.color, self.gradient,
                round(self.opacity, 6))


@dataclass
class Stroke:
    """A stroke: colour, width, opacity, dash pattern."""

    color: str | None = None
    width: float = 1.0
    opacity: float = 1.0
    dash: tuple[float, ...] | None = None
    linecap: str = "butt"
    linejoin: str = "miter"

    def canonical(self) -> tuple:
        return (self.color, round(self.width, 6),
                round(self.opacity, 6), self.dash,
                self.linecap, self.linejoin)


@dataclass
class GradientStop:
    offset: float = 0.0
    color: str = "#000000"
    opacity: float = 1.0

    def canonical(self) -> tuple:
        return (round(self.offset, 6), self.color,
                round(self.opacity, 6))


@dataclass
class Gradient:
    """A gradient definition: linear or radial, object-bounding-box
    by default (``units`` is ``objectBoundingBox`` or
    ``userSpaceOnUse``)."""

    gradient_id: str = ""
    kind: str = "linear"  # linear | radial
    stops: list[GradientStop] = field(default_factory=list)
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.5
    units: str = "objectBoundingBox"
    transform: Affine = field(default_factory=Affine.identity)

    def canonical(self) -> tuple:
        return (self.gradient_id, self.kind,
                tuple(s.canonical() for s in self.stops),
                round(self.x1, 6), round(self.y1, 6),
                round(self.x2, 6), round(self.y2, 6),
                round(self.cx, 6), round(self.cy, 6),
                round(self.r, 6), self.units)


@dataclass
class Defs:
    """The ``<defs>`` block: gradients, reusable paths (textPath
    targets), reusable shapes (``<use>`` targets, by id), and raw
    XML for def children the model does not understand (preserved
    byte-for-byte through a round trip)."""

    gradients: dict[str, Gradient] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    nodes: dict[str, VectorNode] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)

    def canonical(self) -> tuple:
        return (
            tuple(sorted(
                (gid, g.canonical())
                for gid, g in self.gradients.items())),
            tuple(sorted(
                (pid, p.to_d()) for pid, p in self.paths.items())),
            tuple(sorted(
                (nid, n.canonical())
                for nid, n in self.nodes.items())),
            tuple(self.raw),
        )


@dataclass
class VectorNode:
    """One drawable node. ``kind`` names the shape; geometry lives
    on the subclass fields. ``transform`` is the node's own matrix,
    composed over its parents' matrices at render time."""

    node_id: str = ""
    kind: str = "path"
    transform: Affine = field(default_factory=Affine.identity)
    opacity: float = 1.0
    fill: Fill = field(default_factory=Fill)
    stroke: Stroke = field(default_factory=Stroke)
    fill_rule: str = "nonzero"

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        """This node's geometry as a path (identity transform —
        callers compose ``transform`` themselves)."""
        raise NotImplementedError

    def bounds(self, doc: VectorDocument | None = None
               ) -> tuple[tuple[float, float], tuple[float, float]]:
        path = self.to_path(doc)
        if not path:
            return ((0.0, 0.0), (0.0, 0.0))
        return path.transform(self.transform).bounds()

    def canonical(self) -> tuple:
        return (self.node_id, self.kind,
                (round(self.transform.a, 6), round(self.transform.b, 6),
                 round(self.transform.c, 6), round(self.transform.d, 6),
                 round(self.transform.e, 6), round(self.transform.f, 6)),
                round(self.opacity, 6), self.fill.canonical(),
                self.stroke.canonical(), self.fill_rule)


@dataclass
class PathNode(VectorNode):
    kind: str = "path"
    path: Path = field(default_factory=Path)
    d: str = ""

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        return self.path

    def canonical(self) -> tuple:
        return super().canonical() + (self.path.to_d(),)


@dataclass
class RectNode(VectorNode):
    kind: str = "rect"
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rx: float = 0.0
    ry: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        x, y, w, h = self.x, self.y, self.width, self.height
        rx = min(self.rx or self.ry, w / 2.0, h / 2.0)
        ry = min(self.ry or self.rx, w / 2.0, h / 2.0)
        if rx <= 0.0:
            d = (f"M {x:g} {y:g} L {x + w:g} {y:g} "
                 f"L {x + w:g} {y + h:g} L {x:g} {y + h:g} Z")
            return parse_path_data(d)
        k = 0.5522847498307936
        d = (
            f"M {x + rx:g} {y:g} "
            f"L {x + w - rx:g} {y:g} "
            f"C {x + w - rx + k * rx:g} {y:g}, "
            f"{x + w:g} {y + ry - k * ry:g}, {x + w:g} {y + ry:g} "
            f"L {x + w:g} {y + h - ry:g} "
            f"C {x + w:g} {y + h - ry + k * ry:g}, "
            f"{x + w - rx + k * rx:g} {y + h:g}, "
            f"{x + w - rx:g} {y + h:g} "
            f"L {x + rx:g} {y + h:g} "
            f"C {x + rx - k * rx:g} {y + h:g}, "
            f"{x:g} {y + h - ry + k * ry:g}, {x:g} {y + h - ry:g} "
            f"L {x:g} {y + ry:g} "
            f"C {x:g} {y + ry - k * ry:g}, "
            f"{x + rx - k * rx:g} {y:g}, {x + rx:g} {y:g} Z")
        return parse_path_data(d)

    def canonical(self) -> tuple:
        return super().canonical() + (
            round(self.x, 6), round(self.y, 6),
            round(self.width, 6), round(self.height, 6),
            round(self.rx, 6), round(self.ry, 6))


@dataclass
class CircleNode(VectorNode):
    kind: str = "circle"
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        return EllipseNode(cx=self.cx, cy=self.cy, rx=self.r,
                           ry=self.r).to_path()

    def canonical(self) -> tuple:
        return super().canonical() + (
            round(self.cx, 6), round(self.cy, 6), round(self.r, 6))


@dataclass
class EllipseNode(VectorNode):
    kind: str = "ellipse"
    cx: float = 0.0
    cy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        k = 0.5522847498307936
        cx, cy, rx, ry = self.cx, self.cy, self.rx, self.ry
        d = (
            f"M {cx + rx:g} {cy:g} "
            f"C {cx + rx:g} {cy + k * ry:g}, "
            f"{cx + k * rx:g} {cy + ry:g}, {cx:g} {cy + ry:g} "
            f"C {cx - k * rx:g} {cy + ry:g}, "
            f"{cx - rx:g} {cy + k * ry:g}, {cx - rx:g} {cy:g} "
            f"C {cx - rx:g} {cy - k * ry:g}, "
            f"{cx - k * rx:g} {cy - ry:g}, {cx:g} {cy - ry:g} "
            f"C {cx + k * rx:g} {cy - ry:g}, "
            f"{cx + rx:g} {cy - k * ry:g}, {cx + rx:g} {cy:g} Z")
        return parse_path_data(d)

    def canonical(self) -> tuple:
        return super().canonical() + (
            round(self.cx, 6), round(self.cy, 6),
            round(self.rx, 6), round(self.ry, 6))


@dataclass
class LineNode(VectorNode):
    kind: str = "line"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        return parse_path_data(
            f"M {self.x1:g} {self.y1:g} L {self.x2:g} {self.y2:g}")

    def canonical(self) -> tuple:
        return super().canonical() + (
            round(self.x1, 6), round(self.y1, 6),
            round(self.x2, 6), round(self.y2, 6))


@dataclass
class PolyNode(VectorNode):
    kind: str = "polyline"  # polyline | polygon
    points: list[tuple[float, float]] = field(default_factory=list)

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        if not self.points:
            return Path()
        d = "M " + " L ".join(f"{x:g} {y:g}" for x, y in self.points)
        if self.kind == "polygon":
            d += " Z"
        return parse_path_data(d)

    def canonical(self) -> tuple:
        return super().canonical() + (
            tuple((round(x, 6), round(y, 6))
                  for x, y in self.points),)


@dataclass
class TextNode(VectorNode):
    kind: str = "text"
    text: str = ""
    x: float = 0.0
    y: float = 0.0
    font_family: str = "DejaVuSans"
    font_size: float = 16.0
    font_weight: str = "normal"
    font_style: str = "normal"
    text_path: str | None = None
    start_offset: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        # Text has no outline in this model; the renderer draws it
        # with the font door. Its bounds come from the font metrics
        # at render time — here, an empty path.
        return Path()

    def bounds(self, doc: VectorDocument | None = None
               ) -> tuple[tuple[float, float], tuple[float, float]]:
        width = len(self.text) * self.font_size * 0.6
        height = self.font_size
        top_left = self.transform.apply((self.x, self.y - height))
        bottom_right = self.transform.apply(
            (self.x + width, self.y))
        return (top_left, bottom_right)

    def canonical(self) -> tuple:
        return super().canonical() + (
            self.text, round(self.x, 6), round(self.y, 6),
            self.font_family, round(self.font_size, 6),
            self.font_weight, self.font_style, self.text_path,
            round(self.start_offset, 6))


@dataclass
class ImageNode(VectorNode):
    kind: str = "image"
    href: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        return RectNode(x=self.x, y=self.y, width=self.width,
                        height=self.height).to_path()

    def canonical(self) -> tuple:
        return super().canonical() + (
            self.href, round(self.x, 6), round(self.y, 6),
            round(self.width, 6), round(self.height, 6))


@dataclass
class GroupNode(VectorNode):
    kind: str = "group"
    children: list[VectorNode] = field(default_factory=list)

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        return Path()

    def canonical(self) -> tuple:
        # A group's own attributes only — never its children.
        # Children are compared as their own entries in
        # fidelity_report; embedding them here makes one edited
        # child report its (untouched) group as changed too.
        return super().canonical()


@dataclass
class UseNode(VectorNode):
    kind: str = "use"
    href: str = ""
    x: float = 0.0
    y: float = 0.0

    def _target(self, doc: VectorDocument | None
                ) -> VectorNode | None:
        if doc is None:
            return None
        return doc.find_node(self.href.lstrip("#"))

    def to_path(self, doc: VectorDocument | None = None) -> Path:
        target = self._target(doc)
        if target is None:
            return Path()
        path = target.to_path(doc)
        shift = Affine.translate(self.x, self.y)
        return path.transform(shift * target.transform)

    def bounds(self, doc: VectorDocument | None = None
               ) -> tuple[tuple[float, float], tuple[float, float]]:
        target = self._target(doc)
        if target is None:
            return ((self.x, self.y), (self.x, self.y))
        (x0, y0), (x1, y1) = target.bounds(doc)
        return ((x0 + self.x, y0 + self.y),
                (x1 + self.x, y1 + self.y))

    def canonical(self) -> tuple:
        return super().canonical() + (
            self.href, round(self.x, 6), round(self.y, 6))


@dataclass
class Layer:
    """One layer — a named, ordered list of top-level nodes."""

    name: str = "Layer 1"
    visible: bool = True
    locked: bool = False
    children: list[VectorNode] = field(default_factory=list)

    def canonical(self) -> tuple:
        return (self.name, self.visible, self.locked,
                tuple(c.canonical() for c in self.children))


@dataclass
class VectorDocument:
    """A vector document: canvas + layers + defs + unknowns.

    ``unknown`` holds raw XML strings for top-level elements the
    model does not understand (custom namespaces, foreignObject);
    they round-trip byte-for-byte in content. ``root_attrs`` holds
    root attributes beyond width/height/viewBox (``id``,
    ``preserveAspectRatio``, …) so a round trip keeps them.
    """

    width: float = 0.0
    height: float = 0.0
    view_box: tuple[float, float, float, float] | None = None
    title: str = ""
    desc: str = ""
    layers: list[Layer] = field(default_factory=list)
    defs: Defs = field(default_factory=Defs)
    unknown: list[str] = field(default_factory=list)
    root_attrs: dict[str, str] = field(default_factory=dict)

    # ── construction ──

    @classmethod
    def from_bytes(cls, data: bytes) -> VectorDocument:
        from .svg import parse_svg
        return parse_svg(data)

    def to_bytes(self) -> bytes:
        from .svg import serialize_svg
        return serialize_svg(self)

    @classmethod
    def blank(cls, width: float = 800.0, height: float = 600.0,
              title: str = "") -> VectorDocument:
        doc = cls(width=width, height=height,
                  view_box=(0.0, 0.0, width, height), title=title)
        doc.layers.append(Layer(name="Layer 1"))
        return doc

    # ── inventory ──

    def all_nodes(self) -> list[VectorNode]:
        """Leaf nodes in document order (groups expanded, groups
        themselves excluded — use :meth:`find_node` for those)."""
        out: list[VectorNode] = []

        def _walk(nodes: list[VectorNode]) -> None:
            for node in nodes:
                if isinstance(node, GroupNode):
                    _walk(node.children)
                else:
                    out.append(node)

        for layer in self.layers:
            _walk(layer.children)
        return out

    def find_node(self, node_id: str) -> VectorNode | None:
        if node_id in self.defs.nodes:
            return self.defs.nodes[node_id]

        def _walk(nodes: list[VectorNode]) -> VectorNode | None:
            for node in nodes:
                if node.node_id == node_id:
                    return node
                if isinstance(node, GroupNode):
                    found = _walk(node.children)
                    if found is not None:
                        return found
            return None

        for layer in self.layers:
            found = _walk(layer.children)
            if found is not None:
                return found
        return None

    def find_parent(self, node_id: str
                    ) -> tuple[list[VectorNode], int] | None:
        """(sibling list, index) holding the node, or None."""

        def _walk(nodes: list[VectorNode]
                  ) -> tuple[list[VectorNode], int] | None:
            for index, node in enumerate(nodes):
                if node.node_id == node_id:
                    return (nodes, index)
                if isinstance(node, GroupNode):
                    found = _walk(node.children)
                    if found is not None:
                        return found
            return None

        for layer in self.layers:
            found = _walk(layer.children)
            if found is not None:
                return found
        return None

    def next_id(self) -> str:
        # Monotonic max+1 over EVERY live id (leaves, groups at any
        # depth, defs targets): a removed node's id is never reused,
        # so a fresh node can never be confused with a deleted one
        # — and the id is stable even while nodes are temporarily
        # detached mid-edit (group/ungroup).
        used: set[str] = set(self.defs.nodes.keys())

        def walk(nodes: list[VectorNode]) -> None:
            for node in nodes:
                used.add(node.node_id)
                if isinstance(node, GroupNode):
                    walk(node.children)

        for layer in self.layers:
            walk(layer.children)
        numbers = [int(n[1:]) for n in used
                   if n.startswith("n") and n[1:].isdigit()]
        return f"n{max(numbers, default=0) + 1}"

    def to_dict(self) -> dict[str, Any]:
        nodes = self.all_nodes()
        by_kind: dict[str, int] = {}
        for node in nodes:
            by_kind[node.kind] = by_kind.get(node.kind, 0) + 1
        return {
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "view_box": list(self.view_box)
            if self.view_box else None,
            "layers": [layer.name for layer in self.layers],
            "nodes": len(nodes),
            "by_kind": by_kind,
            "gradients": sorted(self.defs.gradients),
            "unknown": len(self.unknown),
        }

    def plain_text(self) -> str:
        return "\n".join(n.text for n in self.all_nodes()
                         if isinstance(n, TextNode))

    # ── fidelity ──

    def fidelity_report(self, other: VectorDocument, *,
                        touched: set[str] | None = None
                        ) -> dict[str, Any]:
        """Compare this document with ``other`` node by node.

        ``changed_ids`` counts node ids whose canonical form
        differs; ``untouched_changed`` counts differing ids that
        were NOT in ``touched`` (the ids the editor says it
        changed) — a round trip must report 0, and an edit must
        report 0 for everything it did not touch.
        """
        touched = touched or set()
        mine = {n.node_id: n.canonical() for n in self.all_nodes()}
        mine.update({g.node_id: g.canonical()
                     for layer in self.layers
                     for g in layer.children
                     if isinstance(g, GroupNode)})
        theirs = {n.node_id: n.canonical()
                  for n in other.all_nodes()}
        theirs.update({g.node_id: g.canonical()
                       for layer in other.layers
                       for g in layer.children
                       if isinstance(g, GroupNode)})
        changed = [nid for nid in mine
                   if nid in theirs and mine[nid] != theirs[nid]]
        added = [nid for nid in theirs if nid not in mine]
        removed = [nid for nid in mine if nid not in theirs]
        return {
            "nodes_before": len(mine),
            "nodes_after": len(theirs),
            "changed_ids": len(changed),
            "untouched_changed": len(
                [nid for nid in changed if nid not in touched]),
            "added": len(added),
            "removed": len(removed),
            "defs_changed": self.defs.canonical()
            != other.defs.canonical(),
        }

    # ── helpers ──

    def view_scale(self) -> tuple[float, float, float, float]:
        """(scale_x, scale_y, offset_x, offset_y) mapping viewBox
        user units to canvas pixels (meet, centred)."""
        if not self.view_box or self.width <= 0 or self.height <= 0:
            return (1.0, 1.0, 0.0, 0.0)
        vx, vy, vw, vh = self.view_box
        if vw <= 0 or vh <= 0:
            return (1.0, 1.0, -vx, -vy)
        scale = min(self.width / vw, self.height / vh)
        ox = (self.width - vw * scale) / 2.0 - vx * scale
        oy = (self.height - vh * scale) / 2.0 - vy * scale
        return (scale, scale, ox, oy)
