"""FigCraft frame renderer — frame -> PNG (lane 105).

Renders through Devon's own Pillow path (the same door the
vector organ uses). Solid fills, linear gradients, strokes
with dash patterns, rounded corners, text, vector networks,
star polygons and resolved instances are drawn; images are
drawn from the document's image parts. Effects (blur, shadow,
glow) are NOT rendered — they are named in ``unrendered``,
never silently dropped. A frame renders at its own size;
children are placed from :func:`compute_layout` when it
applies and from their transforms otherwise.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .components import resolve_instance
from .layout import compute_layout
from .model import DesignDocument

__all__ = ["render_frame"]

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = max(1, int(round(size)))
    path = _FONT_PATHS[1] if bold else _FONT_PATHS[0]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _rgba(color: dict[str, Any], opacity: float = 1.0) -> tuple[int, int, int, int]:
    a = float(color.get("a", 1.0)) * opacity
    return (int(round(float(color.get("r", 0.0)) * 255)),
            int(round(float(color.get("g", 0.0)) * 255)),
            int(round(float(color.get("b", 0.0)) * 255)),
            int(round(max(0.0, min(1.0, a)) * 255)))


def _gradient_image(width: int, height: int, fill: dict[str, Any]) -> Image.Image:
    stops = fill.get("stops") or []
    if not stops:
        return Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    from_xy = fill.get("from") or [0.0, 0.0]
    to_xy = fill.get("to") or [1.0, 0.0]
    fx, fy = float(from_xy[0]), float(from_xy[1])
    tx, ty = float(to_xy[0]), float(to_xy[1])
    dx, dy = tx - fx, ty - fy
    length = dx * dx + dy * dy
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    nx = (xs + 0.5) / max(width, 1)
    ny = (ys + 0.5) / max(height, 1)
    t = (np.zeros((height, width)) if length == 0
         else ((nx - fx) * dx + (ny - fy) * dy) / length)
    t = np.clip(t, 0.0, 1.0)
    positions = np.array([float(s.get("position", 0.0)) for s in stops])
    reds = np.array([float(s["color"].get("r", 0.0)) * 255 for s in stops])
    greens = np.array([float(s["color"].get("g", 0.0)) * 255 for s in stops])
    blues = np.array([float(s["color"].get("b", 0.0)) * 255 for s in stops])
    alphas = np.array([float(s["color"].get("a", 1.0)) * 255 for s in stops])
    r = np.interp(t, positions, reds)
    g = np.interp(t, positions, greens)
    b = np.interp(t, positions, blues)
    a = np.interp(t, positions, alphas)
    arr = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _mask_for(node: dict[str, Any], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(mask)
    radii = (node.get("corners") or {}).get("radii") or [0.0, 0.0, 0.0, 0.0]
    radius = max(float(r) for r in radii)
    if radius > 0:
        draw.rounded_rectangle([0, 0, width - 1, height - 1],
                               radius=min(radius, width / 2, height / 2),
                               fill=255)
    else:
        draw.rectangle([0, 0, width - 1, height - 1], fill=255)
    return mask


def _star_points(count: int, inner_ratio: float,
                 width: float, height: float) -> list[tuple[float, float]]:
    cx, cy = width / 2.0, height / 2.0
    outer = min(width, height) / 2.0
    inner = outer * inner_ratio
    points: list[tuple[float, float]] = []
    for i in range(count * 2):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / count
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    return points


class _Renderer:
    def __init__(self, doc: DesignDocument, layout: dict[int, dict[str, float]]):
        self.doc = doc
        self.layout = layout
        self.unrendered: list[str] = []

    # -- entry --------------------------------------------------------
    def render(self, node: dict[str, Any]) -> Image.Image:
        box = self.layout.get(int(node["id"])) or {
            "x": 0.0, "y": 0.0,
            "w": float(node.get("width") or 0.0),
            "h": float(node.get("height") or 0.0)}
        width = max(1, int(round(box["w"])))
        height = max(1, int(round(box["h"])))
        return self._draw_node(node, width, height)

    def _children_of(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        children = node.get("children")
        if isinstance(children, list) and children and isinstance(children[0], dict):
            return [c for c in children if isinstance(c, dict)]
        return [self.doc.node(int(c)) for c in (children or [])]

    def _draw_node(self, node: dict[str, Any], width: int, height: int) -> Image.Image:
        layer = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
        ntype = node.get("type")
        opacity = float(node.get("opacity", 1.0))
        if ntype == "instance":
            resolved = resolve_instance(self.doc, int(node["id"]))
            flat = resolved["tree"]
            by_id: dict[int, dict[str, Any]] = {}
            for item in flat:
                by_id[int(item["id"])] = item
            for item in flat:
                item["children"] = [by_id[int(c)] for c in (item.get("children") or [])
                                    if int(c) in by_id]
            drawn = self._draw_resolved(by_id[int(flat[0]["id"])], width, height)
            layer.alpha_composite(drawn, (0, 0))
        elif ntype == "text":
            self._draw_text(layer, node, width, height)
        elif ntype == "ellipse":
            self._draw_ellipse(layer, node, width, height)
        elif ntype == "vector":
            self._draw_vector(layer, node, width, height)
        elif ntype == "star":
            self._draw_star(layer, node, width, height)
        else:
            self._draw_shape(layer, node, width, height)
        # children
        children = self._children_of(node)
        if children:
            kids = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            for child in children:
                cbox = self.layout.get(int(child["id"]))
                if cbox is not None:
                    cx = int(round(cbox["x"] - self._origin_x(node)))
                    cy = int(round(cbox["y"] - self._origin_y(node)))
                    cw = max(1, int(round(cbox["w"])))
                    ch = max(1, int(round(cbox["h"])))
                else:
                    transform = child.get("transform") or [1, 0, 0, 1, 0, 0]
                    cx, cy = int(round(transform[4])), int(round(transform[5]))
                    cw = max(1, int(round(float(child.get("width") or 0))))
                    ch = max(1, int(round(float(child.get("height") or 0))))
                child_img = self._draw_node(child, cw, ch)
                kids.alpha_composite(child_img, (cx, cy))
            if node.get("clip", ntype in ("frame", "component", "componentSet")):
                mask = _mask_for(node, width, height)
                kids.putalpha(Image.composite(kids.split()[3], Image.new("L", kids.size, 0), mask))
            layer.alpha_composite(kids, (0, 0))
        for effect in node.get("effects") or []:
            etype = effect.get("type", "effect")
            self.unrendered.append(
                f"effect {etype} on node {node['id']} ({node.get('name', '')})")
        if opacity < 1.0:
            alpha = layer.split()[3].point(lambda v: int(v * opacity))
            layer.putalpha(alpha)
        return layer

    def _origin_x(self, node: dict[str, Any]) -> float:
        box = self.layout.get(int(node["id"]))
        return box["x"] if box else 0.0

    def _origin_y(self, node: dict[str, Any]) -> float:
        box = self.layout.get(int(node["id"]))
        return box["y"] if box else 0.0

    def _draw_resolved(self, tree: dict[str, Any],
                       width: int, height: int) -> Image.Image:
        layer = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
        self._draw_shape(layer, tree, width, height)
        for child in tree.get("children") or []:
            if not isinstance(child, dict):
                continue
            transform = child.get("transform") or [1, 0, 0, 1, 0, 0]
            cx, cy = int(round(transform[4])), int(round(transform[5]))
            cw = max(1, int(round(float(child.get("width") or 0))))
            ch = max(1, int(round(float(child.get("height") or 0))))
            child_img = self._draw_node(child, cw, ch)
            layer.alpha_composite(child_img, (cx, cy))
        return layer

    # -- shapes -------------------------------------------------------
    def _draw_shape(self, layer: Image.Image, node: dict[str, Any],
                    width: int, height: int) -> None:
        fills = node.get("fills") or []
        mask = _mask_for(node, width, height)
        for fill in fills:
            if fill.get("visible") is False:
                continue
            ftype = fill.get("type")
            if ftype == "solid":
                img = Image.new("RGBA", layer.size,
                                _rgba(fill["color"], fill.get("opacity", 1.0)))
            elif ftype == "gradient":
                img = _gradient_image(width, height, fill)
            elif ftype == "image":
                img = self._draw_image_fill(node, fill, width, height)
                if img is None:
                    continue
            else:
                self.unrendered.append(
                    f"fill {ftype} on node {node['id']} ({node.get('name', '')})")
                continue
            img.putalpha(Image.composite(img.split()[3], Image.new("L", img.size, 0), mask))
            layer.alpha_composite(img, (0, 0))
        self._draw_strokes(layer, node, width, height)

    def _draw_image_fill(self, node: dict[str, Any], fill: dict[str, Any],
                         width: int, height: int) -> Image.Image | None:
        image_hash = str(fill.get("hash") or fill.get("image") or "")
        raw = self.doc.images.get(image_hash)
        if raw is None:
            self.unrendered.append(
                f"image {image_hash or 'unknown'} on node {node['id']} "
                f"({node.get('name', '')}): missing image data")
            placeholder = Image.new("RGBA", (max(1, width), max(1, height)),
                                    (220, 220, 220, 255))
            draw = ImageDraw.Draw(placeholder)
            draw.rectangle([0, 0, width - 1, height - 1], outline=(150, 150, 150))
            draw.line([0, 0, width, height], fill=(150, 150, 150))
            draw.line([width, 0, 0, height], fill=(150, 150, 150))
            return placeholder
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            self.unrendered.append(
                f"image {image_hash} on node {node['id']}: undecodable")
            return None
        iw, ih = img.size
        scale = max(width / iw, height / ih)
        img = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))),
                         Image.LANCZOS)
        left = (img.size[0] - width) // 2
        top = (img.size[1] - height) // 2
        return img.crop((left, top, left + width, top + height))

    def _draw_strokes(self, layer: Image.Image, node: dict[str, Any],
                      width: int, height: int) -> None:
        strokes = node.get("strokes") or []
        if not strokes:
            return
        draw = ImageDraw.Draw(layer)
        radii = (node.get("corners") or {}).get("radii") or [0.0] * 4
        radius = max(float(r) for r in radii)
        for stroke in strokes:
            if stroke.get("visible") is False:
                continue
            color = _rgba(stroke.get("color") or {"r": 0, "g": 0, "b": 0, "a": 1},
                          stroke.get("opacity", 1.0))
            sw = max(1, int(round(float(stroke.get("weight", 1.0)))))
            dash = stroke.get("dash")
            if radius > 0:
                draw.rounded_rectangle([sw // 2, sw // 2, width - 1 - sw // 2,
                                        height - 1 - sw // 2],
                                       radius=min(radius, width / 2, height / 2),
                                       outline=color, width=sw)
            elif dash:
                self._draw_dashed_rect(draw, width, height, color, sw, dash)
            else:
                draw.rectangle([sw // 2, sw // 2, width - 1 - sw // 2,
                                height - 1 - sw // 2], outline=color, width=sw)

    @staticmethod
    def _draw_dashed_rect(draw: ImageDraw.ImageDraw, width: int, height: int,
                          color: tuple[int, int, int, int], sw: int,
                          dash: list[float]) -> None:
        pattern = [max(1.0, float(d)) for d in dash] or [4.0, 4.0]
        edges = [((0, 0), (width, 0)), ((width, 0), (width, height)),
                 ((width, height), (0, height)), ((0, height), (0, 0))]
        for (x0, y0), (x1, y1) in edges:
            length = abs(x1 - x0) + abs(y1 - y0)
            pos = 0.0
            idx = 0
            on = True
            while pos < length:
                seg = pattern[idx % len(pattern)]
                end = min(length, pos + seg)
                if on:
                    if x0 == x1:
                        draw.line([(x0, y0 + pos), (x0, y0 + end)], fill=color, width=sw)
                    else:
                        draw.line([(x0 + pos, y0), (x0 + end, y0)], fill=color, width=sw)
                pos = end
                idx += 1
                on = not on

    # -- text ---------------------------------------------------------
    def _draw_text(self, layer: Image.Image, node: dict[str, Any],
                   width: int, height: int) -> None:
        # A text node's fills ARE the glyph colour, never a
        # background — painting them as a rect would flood the
        # whole text box. Only strokes (rare) paint a shape.
        self._draw_strokes(layer, node, width, height)
        style = node.get("style") or {}
        size = float(style.get("font_size", 16.0))
        bold = str(style.get("font_style", "")).lower() in ("bold", "semibold", "black")
        font = _font(size, bold)
        text = str(node.get("characters") or "")
        lh = style.get("line_height") or {}
        if lh.get("unit") == "pixels":
            line_height = float(lh.get("value", size * 1.2))
        else:
            line_height = size * float(lh.get("value", 120.0)) / 100.0
        fills = node.get("fills") or []
        color = (17, 17, 17, 255)
        for fill in fills:
            if fill.get("type") == "solid":
                color = _rgba(fill["color"], fill.get("opacity", 1.0))
                break
        draw = ImageDraw.Draw(layer)
        lines: list[str] = []
        for raw_line in text.split("\n"):
            words = raw_line.split(" ")
            current = ""
            for word in words:
                trial = f"{current} {word}".strip()
                if font.getlength(trial) <= width or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        align = str(style.get("alignment", "left")).lower()
        y = 0.0
        for line in lines:
            lw = font.getlength(line)
            x = {"center": (width - lw) / 2.0, "right": width - lw}.get(align, 0.0)
            draw.text((x, y), line, font=font, fill=color)
            y += line_height

    def _draw_ellipse(self, layer: Image.Image, node: dict[str, Any],
                      width: int, height: int) -> None:
        mask = Image.new("L", (max(1, width), max(1, height)), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, width - 1, height - 1], fill=255)
        fills = node.get("fills") or []
        for fill in fills:
            if fill.get("visible") is False:
                continue
            if fill.get("type") == "solid":
                img = Image.new("RGBA", layer.size,
                                _rgba(fill["color"], fill.get("opacity", 1.0)))
            elif fill.get("type") == "gradient":
                img = _gradient_image(width, height, fill)
            else:
                self.unrendered.append(
                    f"fill {fill.get('type')} on node {node['id']} "
                    f"({node.get('name', '')})")
                continue
            img.putalpha(Image.composite(img.split()[3],
                                         Image.new("L", img.size, 0), mask))
            layer.alpha_composite(img, (0, 0))
        strokes = node.get("strokes") or []
        if strokes:
            draw = ImageDraw.Draw(layer)
            for stroke in strokes:
                if stroke.get("visible") is False:
                    continue
                color = _rgba(stroke.get("color") or
                              {"r": 0, "g": 0, "b": 0, "a": 1},
                              stroke.get("opacity", 1.0))
                sw = max(1, int(round(float(stroke.get("weight", 1.0)))))
                draw.ellipse([sw // 2, sw // 2, width - 1 - sw // 2,
                              height - 1 - sw // 2], outline=color, width=sw)

    # -- vector / star -------------------------------------------------
    def _draw_vector(self, layer: Image.Image, node: dict[str, Any],
                     width: int, height: int) -> None:
        network = node.get("network") or {}
        vertices = network.get("vertices") or []
        segments = network.get("segments") or []
        regions = network.get("regions") or []
        draw = ImageDraw.Draw(layer)
        fills = node.get("fills") or []
        for region in regions:
            loops = region.get("loops") or []
            for loop in loops:
                points = [(float(vertices[i]["x"]), float(vertices[i]["y"]))
                          for i in loop if 0 <= int(i) < len(vertices)]
                if len(points) >= 3 and fills:
                    fill = fills[0]
                    if fill.get("type") == "solid":
                        draw.polygon(points, fill=_rgba(fill["color"]))
        for segment in segments:
            a, b = int(segment.get("a", 0)), int(segment.get("b", 0))
            if 0 <= a < len(vertices) and 0 <= b < len(vertices):
                strokes = node.get("strokes") or []
                color = (0, 0, 0, 255)
                sw = 1
                if strokes:
                    color = _rgba(strokes[0].get("color") or
                                  {"r": 0, "g": 0, "b": 0, "a": 1})
                    sw = max(1, int(round(float(strokes[0].get("weight", 1.0)))))
                draw.line([(float(vertices[a]["x"]), float(vertices[a]["y"])),
                           (float(vertices[b]["x"]), float(vertices[b]["y"]))],
                          fill=color, width=sw)

    def _draw_star(self, layer: Image.Image, node: dict[str, Any],
                   width: int, height: int) -> None:
        points = _star_points(int(node.get("count", 5)),
                              float(node.get("inner_ratio", 0.5)),
                              float(width), float(height))
        draw = ImageDraw.Draw(layer)
        fills = node.get("fills") or []
        if fills and fills[0].get("type") == "solid":
            draw.polygon(points, fill=_rgba(fills[0]["color"]))
        self._draw_strokes(layer, node, width, height)


def render_frame(doc: DesignDocument, node_id: int) -> tuple[bytes, dict[str, Any]]:
    """Render one frame (or any node) to PNG bytes.

    Returns ``(png_bytes, info)`` where ``info`` carries the
    width, height and the ``unrendered`` list of effects and
    fills the renderer could not draw.
    """
    node = doc.node(node_id)
    layout = compute_layout(doc, node_id)
    renderer = _Renderer(doc, layout)
    image = renderer.render(node)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), {"width": image.size[0], "height": image.size[1],
                               "unrendered": renderer.unrendered}
