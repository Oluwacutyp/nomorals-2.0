"""Vector editor — fidelity-preserving edits over a document.

Port of the *design* of the upstream ``crates/doc`` document
editing (document_ext + pathops edit) into Devon's Python. Every
mutation goes through :class:`VectorEditor`; :meth:`to_bytes`
returns the original bytes verbatim when nothing changed, and
serialization otherwise re-emits only the model (unknown elements
and defs survive as content). The editor tracks which node ids it
touched so :meth:`VectorDocument.fidelity_report` can prove the
untouched nodes did not move.
"""

from __future__ import annotations

from .errors import VectorError
from .geom import Affine, parse_path_data
from .model import (
    CircleNode,
    EllipseNode,
    Fill,
    GroupNode,
    ImageNode,
    Layer,
    LineNode,
    PathNode,
    PolyNode,
    RectNode,
    Stroke,
    TextNode,
    VectorDocument,
    VectorNode,
)
from .pathops import boolean

__all__ = ["VectorEditor"]


class VectorEditor:
    """Edit a vector document; save back to SVG bytes."""

    def __init__(self, document: VectorDocument,
                 original: bytes | None = None):
        self.document = document
        self.original = original
        self._dirty = False
        self.touched: set[str] = set()

    @classmethod
    def from_bytes(cls, data: bytes) -> VectorEditor:
        return cls(VectorDocument.from_bytes(data), original=data)

    @classmethod
    def create(cls, width: float = 800.0, height: float = 600.0,
               title: str = "") -> VectorEditor:
        return cls(VectorDocument.blank(width, height, title))

    # ── internals ──

    def _touch(self, node_id: str = "") -> None:
        self._dirty = True
        if node_id:
            self.touched.add(node_id)

    def _layer(self) -> Layer:
        for layer in self.document.layers:
            if layer.visible and not layer.locked:
                return layer
        if not self.document.layers:
            self.document.layers.append(Layer(name="Layer 1"))
        return self.document.layers[-1]

    def _node(self, node_id: str) -> VectorNode:
        node = self.document.find_node(node_id)
        if node is None:
            raise VectorError(f"no node {node_id!r} in the document",
                              kind="not_found")
        return node

    def _add(self, node: VectorNode) -> str:
        if not node.node_id:
            node.node_id = self.document.next_id()
        self._layer().children.append(node)
        self._touch(node.node_id)
        return node.node_id

    # ── adding nodes ──

    def add_rect(self, x: float, y: float, width: float,
                 height: float, fill: str = "#000000",
                 stroke: str | None = None, stroke_width: float = 1.0,
                 opacity: float = 1.0) -> str:
        return self._add(RectNode(
            x=x, y=y, width=width, height=height,
            fill=_solid_fill(fill), stroke=_solid_stroke(
                stroke, stroke_width), opacity=opacity))

    def add_circle(self, cx: float, cy: float, r: float,
                   fill: str = "#000000",
                   stroke: str | None = None,
                   stroke_width: float = 1.0,
                   opacity: float = 1.0) -> str:
        return self._add(CircleNode(
            cx=cx, cy=cy, r=r, fill=_solid_fill(fill),
            stroke=_solid_stroke(stroke, stroke_width),
            opacity=opacity))

    def add_ellipse(self, cx: float, cy: float, rx: float, ry: float,
                    fill: str = "#000000",
                    stroke: str | None = None,
                    stroke_width: float = 1.0,
                    opacity: float = 1.0) -> str:
        return self._add(EllipseNode(
            cx=cx, cy=cy, rx=rx, ry=ry, fill=_solid_fill(fill),
            stroke=_solid_stroke(stroke, stroke_width),
            opacity=opacity))

    def add_line(self, x1: float, y1: float, x2: float, y2: float,
                 stroke: str = "#000000", stroke_width: float = 1.0,
                 opacity: float = 1.0) -> str:
        return self._add(LineNode(
            x1=x1, y1=y1, x2=x2, y2=y2,
            fill=Fill(kind="none"),
            stroke=_solid_stroke(stroke, stroke_width),
            opacity=opacity))

    def add_polygon(self, points: list[tuple[float, float]],
                    fill: str = "#000000", closed: bool = True,
                    stroke: str | None = None,
                    stroke_width: float = 1.0,
                    opacity: float = 1.0) -> str:
        return self._add(PolyNode(
            kind="polygon" if closed else "polyline",
            points=list(points), fill=_solid_fill(fill),
            stroke=_solid_stroke(stroke, stroke_width),
            opacity=opacity))

    def add_path(self, d: str, fill: str = "#000000",
                 stroke: str | None = None,
                 stroke_width: float = 1.0,
                 opacity: float = 1.0) -> str:
        try:
            path = parse_path_data(d)
        except ValueError as exc:
            raise VectorError(f"bad path data {d!r}: {exc}",
                              kind="parse") from exc
        return self._add(PathNode(
            path=path, d=d, fill=_solid_fill(fill),
            stroke=_solid_stroke(stroke, stroke_width),
            opacity=opacity))

    def add_text(self, x: float, y: float, text: str,
                 font_family: str = "DejaVuSans",
                 font_size: float = 16.0,
                 font_weight: str = "normal",
                 font_style: str = "normal",
                 fill: str = "#000000",
                 opacity: float = 1.0) -> str:
        return self._add(TextNode(
            text=text, x=x, y=y, font_family=font_family,
            font_size=font_size, font_weight=font_weight,
            font_style=font_style, fill=_solid_fill(fill),
            opacity=opacity))

    def add_image(self, href: str, x: float, y: float,
                  width: float, height: float,
                  opacity: float = 1.0) -> str:
        return self._add(ImageNode(
            href=href, x=x, y=y, width=width, height=height,
            fill=Fill(kind="none"), opacity=opacity))

    # ── restyling ──

    def set_fill(self, node_id: str, color: str | None = None, *,
                 gradient: str | None = None,
                 opacity: float | None = None) -> None:
        node = self._node(node_id)
        if gradient:
            node.fill = Fill(kind="gradient", gradient=gradient,
                             opacity=node.fill.opacity)
        elif color is not None:
            if color.strip().lower() == "none":
                node.fill = Fill(kind="none")
            else:
                node.fill = Fill(kind="solid", color=color,
                                 opacity=node.fill.opacity)
        if opacity is not None:
            node.fill.opacity = max(0.0, min(1.0, opacity))
        self._touch(node_id)

    def set_stroke(self, node_id: str, color: str | None = None, *,
                   width: float | None = None,
                   dash: tuple[float, ...] | None = None,
                   opacity: float | None = None) -> None:
        node = self._node(node_id)
        if color is not None:
            if color.strip().lower() == "none":
                node.stroke = Stroke()
            else:
                node.stroke.color = color
        if width is not None:
            node.stroke.width = width
        if dash is not None:
            node.stroke.dash = dash
        if opacity is not None:
            node.stroke.opacity = max(0.0, min(1.0, opacity))
        self._touch(node_id)

    def set_opacity(self, node_id: str, opacity: float) -> None:
        node = self._node(node_id)
        node.opacity = max(0.0, min(1.0, opacity))
        self._touch(node_id)

    def transform(self, node_id: str, affine: Affine) -> None:
        node = self._node(node_id)
        node.transform = affine * node.transform
        self._touch(node_id)

    # ── structure ──

    def group(self, node_ids: list[str], name: str = "Group") -> str:
        if len(node_ids) < 2:
            raise VectorError("group needs at least two nodes",
                              kind="invalid_argument")
        parents = [self.document.find_parent(nid)
                   for nid in node_ids]
        if any(p is None for p in parents):
            raise VectorError("group: a node was not found",
                              kind="not_found")
        first_list = parents[0][0]
        if any(p[0] is not first_list for p in parents):
            raise VectorError(
                "group: nodes must share a parent",
                kind="invalid_argument")
        siblings, _first_index = parents[0]
        indices = sorted(p[1] for p in parents)
        # Mint the id BEFORE detaching the children: next_id reads
        # the live tree, and a lowest-free scheme would hand the
        # group a child's own id while the children are detached.
        group_id = self.document.next_id()
        children = [siblings[i] for i in indices]
        for index in sorted(indices, reverse=True):
            siblings.pop(index)
        group = GroupNode(node_id=group_id, children=children)
        siblings.insert(min(indices), group)
        self._touch(group.node_id)
        return group.node_id

    def ungroup(self, group_id: str) -> None:
        node = self._node(group_id)
        if not isinstance(node, GroupNode):
            raise VectorError(f"node {group_id!r} is not a group",
                              kind="invalid_argument")
        found = self.document.find_parent(group_id)
        if found is None:
            raise VectorError(f"group {group_id!r} has no parent",
                              kind="not_found")
        siblings, index = found
        siblings.pop(index)
        for offset, child in enumerate(node.children):
            siblings.insert(index + offset, child)
        self._touch(group_id)

    def remove(self, node_id: str) -> None:
        found = self.document.find_parent(node_id)
        if found is None:
            raise VectorError(f"no node {node_id!r} in the document",
                              kind="not_found")
        siblings, index = found
        siblings.pop(index)
        self._touch(node_id)

    def add_layer(self, name: str) -> Layer:
        layer = Layer(name=name)
        self.document.layers.append(layer)
        self._dirty = True
        return layer

    def move_to_layer(self, node_id: str, layer_name: str) -> None:
        target = next((layer for layer in self.document.layers
                       if layer.name == layer_name), None)
        if target is None:
            raise VectorError(f"no layer {layer_name!r}",
                              kind="not_found")
        found = self.document.find_parent(node_id)
        if found is None:
            raise VectorError(f"no node {node_id!r} in the document",
                              kind="not_found")
        siblings, index = found
        node = siblings.pop(index)
        target.children.append(node)
        self._touch(node_id)

    def set_canvas(self, width: float, height: float,
                   view_box: tuple[float, float, float, float]
                   | None = None) -> None:
        self.document.width = width
        self.document.height = height
        if view_box is not None:
            self.document.view_box = view_box
        self._dirty = True

    # ── booleans ──

    def apply_boolean(self, op: str, a_id: str, b_id: str
                      ) -> list[str]:
        if a_id == b_id:
            raise VectorError(
                "boolean op needs two different nodes",
                kind="invalid_argument")
        a = self._node(a_id)
        b = self._node(b_id)
        if isinstance(a, (GroupNode, TextNode)) \
                or isinstance(b, (GroupNode, TextNode)):
            raise VectorError(
                "boolean ops need shape nodes (not groups/text)",
                kind="invalid_argument")
        a_path = a.to_path(self.document).transform(a.transform)
        b_path = b.to_path(self.document).transform(b.transform)
        try:
            results = boolean(op, a_path, b_path)
        except ValueError as exc:
            raise VectorError(str(exc),
                              kind="invalid_argument") from exc
        if not results:
            self.remove(a_id)
            self.remove(b_id)
            return []
        found = self.document.find_parent(a_id)
        if found is None:
            raise VectorError(f"no node {a_id!r} in the document",
                              kind="not_found")
        siblings, index = found
        siblings.pop(index)
        self.remove(b_id)
        result_ids: list[str] = []
        for offset, result in enumerate(results):
            node = PathNode(
                node_id=self.document.next_id(),
                path=result, d=result.to_d(), fill=a.fill,
                stroke=a.stroke, opacity=a.opacity,
                fill_rule=a.fill_rule)
            siblings.insert(index + offset, node)
            result_ids.append(node.node_id)
            self._touch(node.node_id)
        self._touch(a_id)
        return result_ids

    # ── output ──

    def save(self) -> bytes:
        return self.to_bytes()

    def to_bytes(self) -> bytes:
        if not self._dirty and self.original is not None:
            return self.original
        return self.document.to_bytes()


def _solid_fill(color: str) -> Fill:
    if color.strip().lower() == "none":
        return Fill(kind="none")
    return Fill(kind="solid", color=color)


def _solid_stroke(color: str | None, width: float) -> Stroke:
    if color is None:
        return Stroke()
    return Stroke(color=color, width=width)
