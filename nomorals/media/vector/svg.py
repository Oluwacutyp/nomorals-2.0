"""SVG wire form — parse real SVG bytes into the document model and
serialize back.

Port of the *design* of the upstream ``crates/svg`` import/export
(importer with CSS prepass, layers, symbols; exporter with paint
and text) into Devon's Python, stdlib ``xml.etree`` only. The
parser is bounded (32 MB, 100k elements, depth 200) and refuses
malformed XML with kind ``parse``. What the model does not
understand (custom namespaces, ``foreignObject``) is preserved as
raw XML and re-emitted on save — content survives, position does
not (re-emitted at the end of the document). CSS is honoured in
the real cascade order: presentation attributes, then tag / class /
id rules, then inline ``style``; fill, stroke and font properties
inherit down the tree; ``display:none`` nodes are dropped.
"""

from __future__ import annotations

import contextlib
import re
import xml.etree.ElementTree as ET
from typing import Any

from .errors import VectorError
from .geom import Affine, format_float, parse_path_data, parse_transform
from .model import (
    CircleNode,
    EllipseNode,
    Fill,
    Gradient,
    GradientStop,
    GroupNode,
    ImageNode,
    Layer,
    LineNode,
    PathNode,
    PolyNode,
    RectNode,
    Stroke,
    TextNode,
    UseNode,
    VectorDocument,
    VectorNode,
)

__all__ = ["parse_svg", "serialize_svg"]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
NS = {"svg": SVG_NS, "xlink": XLINK_NS, "inkscape": INKSCAPE_NS}

MAX_SVG_BYTES = 32 * 1024 * 1024
MAX_ELEMENTS = 100_000
MAX_DEPTH = 200

NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000",
    "green": "#008000", "blue": "#0000ff", "yellow": "#ffff00",
    "cyan": "#00ffff", "aqua": "#00ffff", "magenta": "#ff00ff",
    "fuchsia": "#ff00ff", "gray": "#808080", "grey": "#808080",
    "silver": "#c0c0c0", "maroon": "#800000", "olive": "#808000",
    "lime": "#00ff00", "navy": "#000080", "teal": "#008080",
    "purple": "#800080", "orange": "#ffa500", "pink": "#ffc0cb",
    "brown": "#a52a2a", "gold": "#ffd700", "indigo": "#4b0082",
    "violet": "#ee82ee", "coral": "#ff7f50", "salmon": "#fa8072",
    "khaki": "#f0e68c", "beige": "#f5f5dc", "ivory": "#fffff0",
    "transparent": "#000000",
}

# NB: "opacity" is deliberately NOT in _INHERITED — per SVG it is
# not an inherited property (a group's opacity composes through
# the rendering tree, which render.py already does). Folding it
# into children at parse time double-counts it on every round
# trip (a 0.9 group re-parsed as 0.81, then 0.729, ...). It stays
# in _STYLE_ATTRS so an element's OWN opacity attribute is read.
_INHERITED = (
    "fill", "fill-opacity", "fill-rule", "stroke",
    "stroke-width", "stroke-opacity", "stroke-dasharray",
    "stroke-linecap", "stroke-linejoin", "font-family",
    "font-size", "font-weight", "font-style", "color",
)

_STYLE_ATTRS = _INHERITED + ("opacity", "stop-color", "stop-opacity")


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _num(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    if value.endswith("px"):
        value = value[:-2]
    try:
        return float(value)
    except ValueError:
        return default


def _length(value: str | None, base: float, default: float = 0.0
            ) -> float:
    if value is None:
        return default
    value = value.strip()
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100.0 * base
        except ValueError:
            return default
    return _num(value, default)


def _opacity(value: str | None, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except ValueError:
        return default


def _norm_color(value: str | None, default: str | None = None
                ) -> str | None:
    """Normalize a colour to lowercase ``#rrggbb`` (or None)."""
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("none", "transparent"):
        return None
    if value.startswith("url("):
        return value
    if value.startswith("#"):
        hexpart = value[1:]
        if len(hexpart) == 3:
            hexpart = "".join(c * 2 for c in hexpart)
        if len(hexpart) == 6:
            try:
                int(hexpart, 16)
            except ValueError:
                return default
            return "#" + hexpart
        return default
    match = re.match(r"rgba?\(([^)]*)\)", value)
    if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        try:
            r, g, b = (int(float(parts[0])), int(float(parts[1])),
                       int(float(parts[2])))
        except (ValueError, IndexError):
            return default
        return f"#{r:02x}{g:02x}{b:02x}"
    if value in NAMED_COLORS:
        return NAMED_COLORS[value]
    return default


def _parse_css(text: str) -> list[tuple[str, str, dict[str, str]]]:
    """Parse a ``<style>`` block into (kind, key, declarations).

    Kinds: ``tag``, ``class``, ``id``, ``tag.class``. Only simple
    selectors are honoured; anything with combinators or pseudo
    classes is skipped.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules: list[tuple[str, str, dict[str, str]]] = []
    for match in re.finditer(r"([^{]+)\{([^}]*)\}", text):
        selector = match.group(1).strip()
        if any(c in selector for c in " >+~:["):
            continue
        decls: dict[str, str] = {}
        for part in match.group(2).split(";"):
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            decls[key.strip()] = val.strip()
        if not decls:
            continue
        if selector.startswith("#"):
            rules.append(("id", selector[1:], decls))
        elif selector.startswith("."):
            rules.append(("class", selector[1:], decls))
        elif "." in selector:
            tag, cls = selector.split(".", 1)
            rules.append(("tag.class", f"{tag}.{cls}", decls))
        else:
            rules.append(("tag", selector, decls))
    return rules


def _parse_style_attr(value: str) -> dict[str, str]:
    decls: dict[str, str] = {}
    for part in (value or "").split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        decls[key.strip()] = val.strip()
    return decls


def _parse_points(value: str) -> list[tuple[float, float]]:
    nums = [float(n) for n in re.findall(
        r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", value or "")]
    return [(nums[i], nums[i + 1])
            for i in range(0, len(nums) - 1, 2)]


class _Parser:
    """Stateful SVG → document conversion."""

    def __init__(self, root: ET.Element):
        self.doc = VectorDocument()
        self.counter = 0
        self.css: list[tuple[str, str, dict[str, str]]] = []
        self.root = root
        self.view_w = 0.0
        self.view_h = 0.0

    def next_id(self) -> str:
        self.counter += 1
        return f"n{self.counter}"

    # ── style ──

    def _style(self, element: ET.Element,
               parent: dict[str, str]) -> dict[str, str]:
        style = {k: v for k, v in parent.items() if k in _INHERITED}
        for attr in _STYLE_ATTRS:
            if attr in element.attrib:
                style[attr] = element.attrib[attr]
        tag = _local(element.tag)
        classes = element.attrib.get("class", "").split()
        node_id = element.attrib.get("id", "")
        # CSS specificity tiers, lowest first: tag, tag.class,
        # class, id; later rules win within a tier.
        for tier in ("tag", "tag.class", "class", "id"):
            for kind, key, decls in self.css:
                if kind != tier:
                    continue
                matches = (
                    (kind == "tag" and key == tag)
                    or (kind == "tag.class" and any(
                        key == f"{tag}.{cls}" for cls in classes))
                    or (kind == "class" and key in classes)
                    or (kind == "id" and key == node_id))
                if matches:
                    style.update(decls)
        style.update(_parse_style_attr(
            element.attrib.get("style", "")))
        return style

    def _fill(self, style: dict[str, str]) -> Fill:
        raw = style.get("fill", "#000000")
        if raw.startswith("url("):
            match = re.match(r"url\(#([^)]+)\)", raw)
            return Fill(kind="gradient",
                        gradient=match.group(1) if match else None,
                        opacity=_opacity(style.get("fill-opacity")))
        color = _norm_color(raw, "#000000")
        if raw.strip().lower() == "none":
            return Fill(kind="none",
                        opacity=_opacity(style.get("fill-opacity")))
        return Fill(kind="solid", color=color,
                    opacity=_opacity(style.get("fill-opacity")))

    def _stroke(self, style: dict[str, str]) -> Stroke:
        raw = style.get("stroke")
        if raw is None or raw.strip().lower() == "none":
            return Stroke()
        color = _norm_color(raw)
        dash: tuple[float, ...] | None = None
        dash_raw = style.get("stroke-dasharray", "").strip()
        if dash_raw and dash_raw.lower() != "none":
            try:
                dash = tuple(float(n) for n in re.findall(
                    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",
                    dash_raw))
            except ValueError:
                dash = None
        return Stroke(
            color=color,
            width=_num(style.get("stroke-width"), 1.0),
            opacity=_opacity(style.get("stroke-opacity")),
            dash=dash if dash else None,
            linecap=style.get("stroke-linecap", "butt"),
            linejoin=style.get("stroke-linejoin", "miter"))

    def _base(self, element: ET.Element, style: dict[str, str]
              ) -> dict[str, Any]:
        transform = parse_transform(
            element.attrib.get("transform", ""))
        # The node's OWN opacity only: group opacity composes at
        # render time (render.py), never by folding it into the
        # children here.
        opacity = _opacity(style.get("opacity"))
        node_id = element.attrib.get("id") or self.next_id()
        return {
            "node_id": node_id,
            "transform": transform,
            "opacity": opacity,
            "fill": self._fill(style),
            "stroke": self._stroke(style),
            "fill_rule": style.get("fill-rule", "nonzero"),
        }

    # ── shapes ──

    def _shape(self, element: ET.Element, style: dict[str, str]
               ) -> VectorNode | None:
        tag = _local(element.tag)
        base = self._base(element, style)
        if tag == "path":
            d = element.attrib.get("d", "")
            try:
                path = parse_path_data(d)
            except ValueError as exc:
                raise VectorError(f"bad path data {d!r}: {exc}",
                                  kind="parse") from exc
            return PathNode(path=path, d=d, **base)
        if tag == "rect":
            return RectNode(
                x=_length(element.attrib.get("x"), self.view_w),
                y=_length(element.attrib.get("y"), self.view_h),
                width=_length(element.attrib.get("width"),
                              self.view_w),
                height=_length(element.attrib.get("height"),
                               self.view_h),
                rx=_num(element.attrib.get("rx"), 0.0),
                ry=_num(element.attrib.get("ry"), 0.0), **base)
        if tag == "circle":
            return CircleNode(
                cx=_length(element.attrib.get("cx"), self.view_w),
                cy=_length(element.attrib.get("cy"), self.view_h),
                r=_num(element.attrib.get("r"), 0.0), **base)
        if tag == "ellipse":
            return EllipseNode(
                cx=_length(element.attrib.get("cx"), self.view_w),
                cy=_length(element.attrib.get("cy"), self.view_h),
                rx=_length(element.attrib.get("rx"), self.view_w),
                ry=_length(element.attrib.get("ry"), self.view_h),
                **base)
        if tag == "line":
            return LineNode(
                x1=_length(element.attrib.get("x1"), self.view_w),
                y1=_length(element.attrib.get("y1"), self.view_h),
                x2=_length(element.attrib.get("x2"), self.view_w),
                y2=_length(element.attrib.get("y2"), self.view_h),
                **base)
        if tag in ("polyline", "polygon"):
            return PolyNode(
                kind=tag,
                points=_parse_points(
                    element.attrib.get("points", "")), **base)
        if tag == "text":
            return self._text(element, base, style)
        if tag == "image":
            href = element.attrib.get("href") or element.attrib.get(
                f"{{{XLINK_NS}}}href", "")
            return ImageNode(
                href=href,
                x=_length(element.attrib.get("x"), self.view_w),
                y=_length(element.attrib.get("y"), self.view_h),
                width=_length(element.attrib.get("width"),
                              self.view_w),
                height=_length(element.attrib.get("height"),
                               self.view_h), **base)
        if tag == "use":
            href = element.attrib.get("href") or element.attrib.get(
                f"{{{XLINK_NS}}}href", "")
            node = UseNode(
                href=href,
                x=_length(element.attrib.get("x"), self.view_w),
                y=_length(element.attrib.get("y"), self.view_h),
                **base)
            target_id = href.lstrip("#")
            target = self.doc.defs.nodes.get(target_id)
            if target is not None:
                if "fill" not in element.attrib \
                        and "fill" not in style:
                    node.fill = target.fill
                if "stroke" not in element.attrib \
                        and "stroke" not in style:
                    node.stroke = target.stroke
            return node
        return None

    def _text(self, element: ET.Element, base: dict[str, Any],
              style: dict[str, str]) -> TextNode:
        text_path: str | None = None
        start_offset = 0.0
        text = ""
        for child in element:
            if _local(child.tag) == "textPath":
                href = child.attrib.get("href") or child.attrib.get(
                    f"{{{XLINK_NS}}}href", "")
                text_path = href.lstrip("#")
                start_offset = _num(
                    child.attrib.get("startOffset"), 0.0)
                text = "".join(child.itertext())
                break
        else:
            text = "".join(element.itertext())
        return TextNode(
            text=text.strip(),
            x=_length(element.attrib.get("x"), self.view_w),
            y=_length(element.attrib.get("y"), self.view_h),
            font_family=style.get("font-family", "DejaVuSans"),
            font_size=_num(style.get("font-size"), 16.0),
            font_weight=style.get("font-weight", "normal"),
            font_style=style.get("font-style", "normal"),
            text_path=text_path,
            start_offset=start_offset, **base)

    # ── defs ──

    def _defs(self, element: ET.Element) -> None:
        for child in element:
            tag = _local(child.tag)
            if not isinstance(child.tag, str):
                continue
            child_id = child.attrib.get("id", "")
            if tag in ("linearGradient", "radialGradient"):
                gradient = self._gradient(child, tag)
                if gradient.gradient_id:
                    self.doc.defs.gradients[gradient.gradient_id] \
                        = gradient
            elif tag == "path" and child_id:
                try:
                    self.doc.defs.paths[child_id] = parse_path_data(
                        child.attrib.get("d", ""))
                except ValueError:
                    self.doc.defs.raw.append(
                        ET.tostring(child, encoding="unicode"))
            elif tag in ("rect", "circle", "ellipse", "line",
                         "polyline", "polygon", "path", "text",
                         "image", "g"):
                style = self._style(child, {})
                if tag == "g":
                    for grand in child:
                        node = self._shape(grand, style)
                        if node is not None and node.node_id:
                            self.doc.defs.nodes[node.node_id] = node
                    continue
                node = self._shape(child, style)
                if node is not None and node.node_id:
                    self.doc.defs.nodes[node.node_id] = node
            else:
                self.doc.defs.raw.append(
                    ET.tostring(child, encoding="unicode"))

    def _gradient(self, element: ET.Element, tag: str) -> Gradient:
        stops: list[GradientStop] = []
        for stop in element:
            if _local(stop.tag) != "stop":
                continue
            stop_style = self._style(stop, {})
            color = _norm_color(
                stop.attrib.get("stop-color")
                or stop_style.get("stop-color"), "#000000")
            stops.append(GradientStop(
                offset=_length(stop.attrib.get("offset"), 1.0),
                color=color or "#000000",
                opacity=_opacity(stop.attrib.get("stop-opacity")
                                 or stop_style.get("stop-opacity"))))
        units = element.attrib.get(
            "gradientUnits", "objectBoundingBox")
        transform = parse_transform(
            element.attrib.get("gradientTransform", ""))
        if tag == "linearGradient":
            return Gradient(
                gradient_id=element.attrib.get("id", ""),
                kind="linear", stops=stops,
                x1=_length(element.attrib.get("x1"), 1.0),
                y1=_length(element.attrib.get("y1"), 1.0),
                x2=_length(element.attrib.get("x2"), 1.0, 1.0),
                y2=_length(element.attrib.get("y2"), 1.0),
                units=units, transform=transform)
        return Gradient(
            gradient_id=element.attrib.get("id", ""),
            kind="radial", stops=stops,
            cx=_length(element.attrib.get("cx"), 1.0, 0.5),
            cy=_length(element.attrib.get("cy"), 1.0, 0.5),
            r=_length(element.attrib.get("r"), 1.0, 0.5),
            units=units, transform=transform)

    # ── tree walk ──

    def _walk(self, element: ET.Element, layer: Layer,
              parent_style: dict[str, str]) -> None:
        for child in element:
            tag = _local(child.tag)
            if not isinstance(child.tag, str):
                continue
            if tag == "style":
                self.css.extend(_parse_css(child.text or ""))
                continue
            if tag == "defs":
                self._defs(child)
                continue
            if tag == "title":
                if not self.doc.title:
                    self.doc.title = (child.text or "").strip()
                continue
            if tag == "desc":
                if not self.doc.desc:
                    self.doc.desc = (child.text or "").strip()
                continue
            style = self._style(child, parent_style)
            if style.get("display", "").strip() == "none":
                continue
            if tag == "g":
                group_style = style
                group = GroupNode(
                    node_id=child.attrib.get("id") or self.next_id(),
                    transform=parse_transform(
                        child.attrib.get("transform", "")),
                    opacity=_opacity(style.get("opacity")),
                    fill=self._fill(group_style),
                    stroke=self._stroke(group_style))
                self._walk_into_group(child, group, group_style)
                layer.children.append(group)
                continue
            if tag in ("svg", "a"):
                group = GroupNode(
                    node_id=child.attrib.get("id") or self.next_id(),
                    transform=Affine.translate(
                        _length(child.attrib.get("x"), self.view_w),
                        _length(child.attrib.get("y"), self.view_h)),
                    opacity=_opacity(style.get("opacity")),
                    fill=self._fill(style),
                    stroke=self._stroke(style))
                self._walk_into_group(child, group, style)
                layer.children.append(group)
                continue
            node = self._shape(child, style)
            if node is not None:
                layer.children.append(node)
                continue
            self.doc.unknown.append(
                ET.tostring(child, encoding="unicode"))

    def _walk_into_group(self, element: ET.Element, group: GroupNode,
                         parent_style: dict[str, str]) -> None:
        pseudo = Layer(children=[])
        self._walk(element, pseudo, parent_style)
        group.children = pseudo.children

    # ── entry ──

    def parse(self) -> VectorDocument:
        root = self.root
        if _local(root.tag) != "svg":
            raise VectorError("not an SVG document (root is not "
                              "<svg>)", kind="parse")
        view_box = root.attrib.get("viewBox", "")
        if view_box:
            try:
                vx, vy, vw, vh = (float(n) for n in
                                  view_box.replace(",", " ").split())
                self.doc.view_box = (vx, vy, vw, vh)
                self.view_w, self.view_h = vw, vh
            except ValueError:
                pass
        width = _length(root.attrib.get("width"), self.view_w or 0.0)
        height = _length(root.attrib.get("height"), self.view_h or 0.0)
        if width <= 0 and self.view_w:
            width = self.view_w
        if height <= 0 and self.view_h:
            height = self.view_h
        self.doc.width, self.doc.height = width, height
        known_root = {"width", "height", "viewBox", "xmlns",
                      "xmlns:xlink", "xmlns:inkscape", "version"}
        for key, value in root.attrib.items():
            if key not in known_root and not key.startswith("xmlns:"):
                self.doc.root_attrs[key] = value
        # CSS first (a <style> anywhere applies document-wide).
        for element in root.iter():
            if isinstance(element.tag, str) \
                    and _local(element.tag) == "style":
                self.css.extend(_parse_css(element.text or ""))
        default_layer = Layer(name="Layer 1")
        self.doc.layers.append(default_layer)
        for child in root:
            tag = _local(child.tag)
            if not isinstance(child.tag, str):
                continue
            if tag == "defs":
                self._defs(child)
            elif tag == "style":
                continue  # already collected above
            elif tag == "title":
                self.doc.title = (child.text or "").strip()
            elif tag == "desc":
                self.doc.desc = (child.text or "").strip()
            elif tag == "g" and child.attrib.get(
                    f"{{{INKSCAPE_NS}}}groupmode") == "layer":
                layer = Layer(
                    name=child.attrib.get(
                        f"{{{INKSCAPE_NS}}}label", "Layer"),
                    children=[])
                self._walk(child, layer, {})
                self.doc.layers.append(layer)
            else:
                self._walk_single(child, default_layer)
        if not default_layer.children \
                and len(self.doc.layers) > 1:
            self.doc.layers.remove(default_layer)
        return self.doc

    def _walk_single(self, element: ET.Element, layer: Layer) -> None:
        tag = _local(element.tag)
        if not isinstance(element.tag, str):
            return
        style = self._style(element, {})
        if style.get("display", "").strip() == "none":
            return
        if tag == "g":
            group = GroupNode(
                node_id=element.attrib.get("id") or self.next_id(),
                transform=parse_transform(
                    element.attrib.get("transform", "")),
                opacity=_opacity(style.get("opacity")),
                fill=self._fill(style),
                stroke=self._stroke(style))
            self._walk_into_group(element, group, style)
            layer.children.append(group)
            return
        if tag in ("svg", "a"):
            group = GroupNode(
                node_id=element.attrib.get("id") or self.next_id(),
                transform=Affine.translate(
                    _length(element.attrib.get("x"), self.view_w),
                    _length(element.attrib.get("y"), self.view_h)),
                opacity=_opacity(style.get("opacity")),
                fill=self._fill(style),
                stroke=self._stroke(style))
            self._walk_into_group(element, group, style)
            layer.children.append(group)
            return
        node = self._shape(element, style)
        if node is not None:
            layer.children.append(node)
            return
        self.doc.unknown.append(
            ET.tostring(element, encoding="unicode"))


def parse_svg(data: bytes) -> VectorDocument:
    """Parse SVG bytes into a :class:`VectorDocument`."""
    if len(data) > MAX_SVG_BYTES:
        raise VectorError(
            f"SVG is {len(data)} bytes (limit {MAX_SVG_BYTES})",
            kind="unsupported")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise VectorError(f"malformed SVG XML: {exc}",
                          kind="parse") from exc
    for count, _ in enumerate(root.iter(), start=1):
        if count > MAX_ELEMENTS:
            raise VectorError(
                f"SVG has over {MAX_ELEMENTS} elements",
                kind="unsupported")
    depth = 0
    stack = [(root, 1)]
    while stack:
        element, level = stack.pop()
        depth = max(depth, level)
        if depth > MAX_DEPTH:
            raise VectorError(
                f"SVG nests deeper than {MAX_DEPTH}",
                kind="unsupported")
        stack.extend((child, level + 1) for child in element)
    # Register namespaces so serialization keeps the prefixes.
    for match in re.finditer(
            rb'xmlns:([A-Za-z_][\w.-]*)="([^"]+)"', data):
        prefix = match.group(1).decode("ascii", "ignore")
        uri = match.group(2).decode("utf-8", "ignore")
        with contextlib.suppress(ValueError):
            ET.register_namespace(prefix, uri)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    return _Parser(root).parse()


# ── serialization ───────────────────────────────────────────────────

def _fmt(value: float) -> str:
    # Exact round-trip text (see geom.format_float): "%g" keeps
    # only 6 significant digits and a re-parsed document drifts.
    return format_float(value)


def _style_attrs(node: VectorNode) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if node.fill.kind == "gradient" and node.fill.gradient:
        attrs["fill"] = f"url(#{node.fill.gradient})"
    elif node.fill.kind == "none":
        attrs["fill"] = "none"
    elif node.fill.color:
        attrs["fill"] = node.fill.color
    if node.fill.opacity != 1.0:
        attrs["fill-opacity"] = _fmt(node.fill.opacity)
    if node.stroke.color:
        attrs["stroke"] = node.stroke.color
        attrs["stroke-width"] = _fmt(node.stroke.width)
        if node.stroke.opacity != 1.0:
            attrs["stroke-opacity"] = _fmt(node.stroke.opacity)
        if node.stroke.dash:
            attrs["stroke-dasharray"] = " ".join(
                _fmt(d) for d in node.stroke.dash)
        if node.stroke.linecap != "butt":
            attrs["stroke-linecap"] = node.stroke.linecap
        if node.stroke.linejoin != "miter":
            attrs["stroke-linejoin"] = node.stroke.linejoin
    if node.fill_rule != "nonzero":
        attrs["fill-rule"] = node.fill_rule
    if node.opacity != 1.0:
        attrs["opacity"] = _fmt(node.opacity)
    if not node.transform.is_identity():
        attrs["transform"] = node.transform.to_svg()
    return attrs


def _node_element(node: VectorNode) -> ET.Element:
    tag = "g" if node.kind == "group" else node.kind
    element = ET.Element(f"{{{SVG_NS}}}{tag}")
    attrs = _style_attrs(node)
    if node.node_id:
        element.set("id", node.node_id)
    if isinstance(node, PathNode):
        element.set("d", node.d or node.path.to_d())
    elif isinstance(node, RectNode):
        element.set("x", _fmt(node.x))
        element.set("y", _fmt(node.y))
        element.set("width", _fmt(node.width))
        element.set("height", _fmt(node.height))
        if node.rx:
            element.set("rx", _fmt(node.rx))
        if node.ry:
            element.set("ry", _fmt(node.ry))
    elif isinstance(node, CircleNode):
        element.set("cx", _fmt(node.cx))
        element.set("cy", _fmt(node.cy))
        element.set("r", _fmt(node.r))
    elif isinstance(node, EllipseNode):
        element.set("cx", _fmt(node.cx))
        element.set("cy", _fmt(node.cy))
        element.set("rx", _fmt(node.rx))
        element.set("ry", _fmt(node.ry))
    elif isinstance(node, LineNode):
        element.set("x1", _fmt(node.x1))
        element.set("y1", _fmt(node.y1))
        element.set("x2", _fmt(node.x2))
        element.set("y2", _fmt(node.y2))
    elif isinstance(node, PolyNode):
        element.set("points", " ".join(
            f"{x:g},{y:g}" for x, y in node.points))
    elif isinstance(node, TextNode):
        element.set("x", _fmt(node.x))
        element.set("y", _fmt(node.y))
        element.set("font-family", node.font_family)
        element.set("font-size", _fmt(node.font_size))
        if node.font_weight != "normal":
            element.set("font-weight", node.font_weight)
        if node.font_style != "normal":
            element.set("font-style", node.font_style)
        if node.text_path:
            child = ET.SubElement(
                element, f"{{{SVG_NS}}}textPath")
            child.set("href", f"#{node.text_path}")
            if node.start_offset:
                child.set("startOffset", _fmt(node.start_offset))
            child.text = node.text
        else:
            element.text = node.text
    elif isinstance(node, ImageNode):
        element.set("href", node.href)
        element.set("x", _fmt(node.x))
        element.set("y", _fmt(node.y))
        element.set("width", _fmt(node.width))
        element.set("height", _fmt(node.height))
    elif isinstance(node, UseNode):
        element.set("href", node.href)
        element.set("x", _fmt(node.x))
        element.set("y", _fmt(node.y))
    elif isinstance(node, GroupNode):
        for child in node.children:
            element.append(_node_element(child))
    for key, value in attrs.items():
        element.set(key, value)
    return element


def _gradient_element(gradient: Gradient) -> ET.Element:
    tag = "linearGradient" if gradient.kind == "linear" \
        else "radialGradient"
    element = ET.Element(f"{{{SVG_NS}}}{tag}")
    element.set("id", gradient.gradient_id)
    if gradient.kind == "linear":
        element.set("x1", _fmt(gradient.x1))
        element.set("y1", _fmt(gradient.y1))
        element.set("x2", _fmt(gradient.x2))
        element.set("y2", _fmt(gradient.y2))
    else:
        element.set("cx", _fmt(gradient.cx))
        element.set("cy", _fmt(gradient.cy))
        element.set("r", _fmt(gradient.r))
    if gradient.units != "objectBoundingBox":
        element.set("gradientUnits", gradient.units)
    if not gradient.transform.is_identity():
        element.set("gradientTransform",
                    gradient.transform.to_svg())
    for stop in gradient.stops:
        child = ET.SubElement(element, f"{{{SVG_NS}}}stop")
        child.set("offset", _fmt(stop.offset))
        child.set("stop-color", stop.color)
        if stop.opacity != 1.0:
            child.set("stop-opacity", _fmt(stop.opacity))
    return element


def serialize_svg(doc: VectorDocument) -> bytes:
    """Serialize a :class:`VectorDocument` to SVG bytes."""
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    for raw in doc.unknown:
        for match in re.finditer(
                r'xmlns:([A-Za-z_][\w.-]*)="([^"]+)"', raw):
            with contextlib.suppress(ValueError):
                ET.register_namespace(match.group(1),
                                     match.group(2))
    root = ET.Element(f"{{{SVG_NS}}}svg")
    root.set("width", _fmt(doc.width))
    root.set("height", _fmt(doc.height))
    if doc.view_box:
        root.set("viewBox", " ".join(_fmt(v) for v in doc.view_box))
    for key, value in doc.root_attrs.items():
        root.set(key, value)
    if doc.title:
        title = ET.SubElement(root, f"{{{SVG_NS}}}title")
        title.text = doc.title
    if doc.desc:
        desc = ET.SubElement(root, f"{{{SVG_NS}}}desc")
        desc.text = doc.desc
    defs = doc.defs
    if defs.gradients or defs.paths or defs.nodes or defs.raw:
        defs_el = ET.SubElement(root, f"{{{SVG_NS}}}defs")
        for gradient in defs.gradients.values():
            defs_el.append(_gradient_element(gradient))
        for path_id, path in defs.paths.items():
            path_el = ET.SubElement(defs_el, f"{{{SVG_NS}}}path")
            path_el.set("id", path_id)
            path_el.set("d", path.to_d())
        for node in defs.nodes.values():
            defs_el.append(_node_element(node))
        for raw in defs.raw:
            try:
                defs_el.append(ET.fromstring(raw))
            except ET.ParseError:
                continue
    multi_layer = len(doc.layers) > 1 or any(
        layer.name != "Layer 1" or layer.locked
        for layer in doc.layers)
    for layer in doc.layers:
        if multi_layer:
            group = ET.SubElement(root, f"{{{SVG_NS}}}g")
            group.set(f"{{{INKSCAPE_NS}}}groupmode", "layer")
            group.set(f"{{{INKSCAPE_NS}}}label", layer.name)
            for child in layer.children:
                group.append(_node_element(child))
        else:
            for child in layer.children:
                root.append(_node_element(child))
    for raw in doc.unknown:
        try:
            root.append(ET.fromstring(raw))
        except ET.ParseError:
            continue
    text = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + text
            ).encode("utf-8")
