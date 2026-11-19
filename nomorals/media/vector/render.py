"""Rasterizer — vector document → PNG bytes.

Port of the *design* of the upstream ``crates/render`` (rasterize
+ encode/png) into Devon's Python over numpy + PIL (the same PIL
Devon's imggen already uses). Fills run through a real scanline
rasterizer with nonzero and even-odd rules; linear/radial
gradients are interpolated per pixel; strokes are disk-stamped
polylines (round joins/caps); text goes through the one font door
(DejaVuSans, same helper as the artist organ); ``data:`` images
paste for real, external image references draw a placeholder
frame (named gap). Nothing here shells out.
"""

from __future__ import annotations

import base64
import io
import math
from itertools import pairwise
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import VectorError
from .geom import Affine, Path
from .model import (
    Gradient,
    GroupNode,
    ImageNode,
    TextNode,
    VectorDocument,
    VectorNode,
)
from .pathops import dash_path

__all__ = ["render_document", "render_svg", "render_to_image"]


def _font(size: int, bold: bool = False) -> Any:
    size = max(8, int(size))
    names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold \
        else ("DejaVuSans.ttf",)
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _parse_rgb(color: str | None) -> tuple[float, float, float]:
    if not color:
        return (0.0, 0.0, 0.0)
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    try:
        return (int(color[0:2], 16) / 255.0,
                int(color[2:4], 16) / 255.0,
                int(color[4:6], 16) / 255.0)
    except ValueError:
        return (0.0, 0.0, 0.0)


def _scanline_mask(contours: list[list[tuple[float, float]]],
                   width: int, height: int,
                   fill_rule: str) -> np.ndarray:
    """Boolean fill mask for flattened contours (device coords)."""
    mask = np.zeros((height, width), dtype=bool)
    if width <= 0 or height <= 0:
        return mask
    # Bucket edge crossings per scanline: (x, delta) with delta
    # +1 for a downward edge, -1 for an upward edge.
    rows: list[list[tuple[float, int]]] = [[] for _ in range(height)]
    for contour in contours:
        for (x1, y1), (x2, y2) in pairwise(contour):
            if y1 == y2:
                continue
            delta = 1 if y2 > y1 else -1
            y_lo = min(y1, y2)
            y_hi = max(y1, y2)
            row_lo = max(0, int(math.floor(y_lo)))
            row_hi = min(height - 1, int(math.ceil(y_hi)) - 1)
            if row_hi < row_lo:
                continue
            slope = (x2 - x1) / (y2 - y1)
            for row in range(row_lo, row_hi + 1):
                yc = row + 0.5
                if y_lo <= yc < y_hi or (yc == y_hi and y2 > y1):
                    rows[row].append((x1 + (yc - y1) * slope, delta))
    for row, crossings in enumerate(rows):
        if not crossings:
            continue
        crossings.sort(key=lambda item: item[0])
        winding = 0
        prev_x: float | None = None
        for parity, (x, delta) in enumerate(crossings):
            if prev_x is not None:
                x0 = max(0, int(math.ceil(prev_x)))
                x1 = min(width, int(math.floor(x)))
                if x1 > x0:
                    if fill_rule == "evenodd":
                        if parity % 2 == 1:
                            mask[row, x0:x1] = True
                    elif winding != 0:
                        mask[row, x0:x1] = True
            winding += delta
            prev_x = x
    return mask


def _composite(canvas: np.ndarray, mask: np.ndarray,
               rgb: np.ndarray, alpha: float) -> None:
    """Source-over a solid colour (rgb 0..1, alpha 0..1) through
    a boolean mask, in place."""
    if alpha <= 0.0 or not mask.any():
        return
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    region = canvas[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    sa = np.where(m, alpha, 0.0)
    da = region[..., 3]
    out_a = sa + da * (1.0 - sa)
    for channel in range(3):
        src = rgb[channel] * sa
        dst = region[..., channel] * da * (1.0 - sa)
        with np.errstate(invalid="ignore", divide="ignore"):
            region[..., channel] = np.where(
                out_a > 0, (src + dst) / np.where(out_a > 0, out_a, 1.0),
                0.0)
    region[..., 3] = out_a


def _gradient_colors(gradient: Gradient,
                     xs: np.ndarray, ys: np.ndarray,
                     bbox: tuple[float, float, float, float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (rgb, alpha) arrays for a gradient over device
    coordinates, using the node's user-space bounding box."""
    bx0, by0, bx1, by1 = bbox
    bw = max(bx1 - bx0, 1e-9)
    bh = max(by1 - by0, 1e-9)
    u = (xs - bx0) / bw
    v = (ys - by0) / bh
    if not gradient.transform.is_identity():
        t = gradient.transform
        u2 = t.a * u + t.c * v + t.e
        v2 = t.b * u + t.d * v + t.f
        u, v = u2, v2
    if gradient.kind == "linear":
        ax = gradient.x1
        ay = gradient.y1
        dx = gradient.x2 - gradient.x1
        dy = gradient.y2 - gradient.y1
        denom = dx * dx + dy * dy or 1.0
        tvals = ((u - ax) * dx + (v - ay) * dy) / denom
    else:
        rx = max(gradient.r, 1e-9)
        tvals = np.sqrt(((u - gradient.cx) / rx) ** 2
                        + ((v - gradient.cy) / rx) ** 2)
    tvals = np.clip(tvals, 0.0, 1.0)
    stops = gradient.stops
    if not stops:
        return (np.zeros(xs.shape + (3,)),
                np.zeros(xs.shape))
    offsets = np.array([s.offset for s in stops])
    reds = np.array([_parse_rgb(s.color)[0] for s in stops])
    greens = np.array([_parse_rgb(s.color)[1] for s in stops])
    blues = np.array([_parse_rgb(s.color)[2] for s in stops])
    alphas = np.array([s.opacity for s in stops])
    rgb = np.stack([
        np.interp(tvals, offsets, reds),
        np.interp(tvals, offsets, greens),
        np.interp(tvals, offsets, blues)], axis=-1)
    alpha = np.interp(tvals, offsets, alphas)
    return rgb, alpha


def _stroke_mask(contours: list[list[tuple[float, float]]],
                 width: float, height: int,
                 stroke_width: float) -> np.ndarray:
    """Boolean mask of stroked polylines (disk stamping)."""
    mask = np.zeros((height, width), dtype=bool)
    if stroke_width <= 0 or width <= 0 or height <= 0:
        return mask
    radius = stroke_width / 2.0
    step = max(1.0, radius)
    for contour in contours:
        points: list[tuple[float, float]] = []
        for a, b in pairwise(contour):
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            count = max(1, int(seg_len / step) + 1)
            for i in range(count):
                t = i / count
                points.append((a[0] + (b[0] - a[0]) * t,
                               a[1] + (b[1] - a[1]) * t))
            points.append(b)
        for cx, cy in points:
            x0 = max(0, int(math.floor(cx - radius)))
            x1 = min(width, int(math.ceil(cx + radius)) + 1)
            y0 = max(0, int(math.floor(cy - radius)))
            y1 = min(height, int(math.ceil(cy + radius)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disc = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2
                    <= radius * radius)
            mask[y0:y1, x0:x1] |= disc
    return mask


class _Renderer:
    def __init__(self, doc: VectorDocument, scale: float,
                 background: tuple[int, int, int] | None):
        width = doc.width if doc.width > 0 else (
            doc.view_box[2] if doc.view_box else 0.0)
        height = doc.height if doc.height > 0 else (
            doc.view_box[3] if doc.view_box else 0.0)
        self.width = max(1, int(round(width * scale)))
        self.height = max(1, int(round(height * scale)))
        self.scale = scale
        self.doc = doc
        self.canvas = np.zeros((self.height, self.width, 4),
                               dtype=np.float64)
        if background is not None:
            self.canvas[..., 0] = background[0] / 255.0
            self.canvas[..., 1] = background[1] / 255.0
            self.canvas[..., 2] = background[2] / 255.0
            self.canvas[..., 3] = 1.0
        sx, sy, ox, oy = doc.view_scale()
        self.view = Affine(a=sx * scale, d=sy * scale,
                           e=ox * scale, f=oy * scale)
        self.overlay = Image.new("RGBA", (self.width, self.height),
                                 (0, 0, 0, 0))
        self.overlay_draw = ImageDraw.Draw(self.overlay)

    def device(self, path: Path, transform: Affine) -> Path:
        return path.transform(self.view * transform)

    def render(self) -> Image.Image:
        for layer in self.doc.layers:
            if not layer.visible:
                continue
            for node in layer.children:
                self._node(node, Affine.identity(), 1.0)
        base = Image.fromarray(
            (np.clip(self.canvas, 0.0, 1.0) * 255).astype("uint8"),
            "RGBA")
        base.alpha_composite(self.overlay)
        return base

    def _node(self, node: VectorNode, parent: Affine,
              parent_opacity: float) -> None:
        opacity = parent_opacity * node.opacity
        transform = parent * node.transform
        if isinstance(node, GroupNode):
            for child in node.children:
                self._node(child, transform, opacity)
            return
        if isinstance(node, TextNode):
            self._text(node, transform, opacity)
            return
        if isinstance(node, ImageNode):
            self._image(node, transform, opacity)
            return
        path = node.to_path(self.doc)
        if not path:
            return
        if node.fill.kind == "gradient" and node.fill.gradient:
            gradient = self.doc.defs.gradients.get(node.fill.gradient)
        else:
            gradient = None
        device = self.device(path, transform)
        contours = device.flatten(0.5)
        if node.fill.kind != "none":
            mask = _scanline_mask(contours, self.width, self.height,
                                  node.fill_rule)
            if mask.any():
                if gradient is not None:
                    self._fill_gradient(mask, gradient, node,
                                        device, opacity)
                elif node.fill.color:
                    rgb = np.array(_parse_rgb(node.fill.color))
                    _composite(self.canvas, mask, rgb,
                               node.fill.opacity * opacity)
        if node.stroke.color:
            stroke_width = node.stroke.width
            stroke_paths = [device]
            if node.stroke.dash:
                stroke_paths = dash_path(device, node.stroke.dash)
            mask = np.zeros((self.height, self.width), dtype=bool)
            for stroke_path in stroke_paths:
                mask |= _stroke_mask(
                    stroke_path.flatten(0.5), self.width, self.height,
                    stroke_width * self.scale)
            if mask.any():
                rgb = np.array(_parse_rgb(node.stroke.color))
                _composite(self.canvas, mask, rgb,
                           node.stroke.opacity * opacity)

    def _fill_gradient(self, mask: np.ndarray, gradient: Gradient,
                       node: VectorNode, device: Path,
                       opacity: float) -> None:
        (x0, y0), (x1, y1) = device.bounds()
        if x1 <= x0 or y1 <= y0:
            return
        # Work in device pixels over the node's bbox.
        ix0 = max(0, int(math.floor(x0)))
        ix1 = min(self.width, int(math.ceil(x1)))
        iy0 = max(0, int(math.floor(y0)))
        iy1 = min(self.height, int(math.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        xs = (xx + 0.5).astype(np.float64)
        ys = (yy + 0.5).astype(np.float64)
        rgb, alpha = _gradient_colors(gradient, xs, ys,
                                      (x0, y0, x1, y1))
        region = self.canvas[iy0:iy1, ix0:ix1]
        m = mask[iy0:iy1, ix0:ix1]
        sa = np.where(m, alpha * node.fill.opacity * opacity, 0.0)
        da = region[..., 3]
        out_a = sa + da * (1.0 - sa)
        for channel in range(3):
            src = rgb[..., channel] * sa
            dst = region[..., channel] * da * (1.0 - sa)
            with np.errstate(invalid="ignore", divide="ignore"):
                region[..., channel] = np.where(
                    out_a > 0,
                    (src + dst) / np.where(out_a > 0, out_a, 1.0),
                    0.0)
        region[..., 3] = out_a

    def _text(self, node: TextNode, transform: Affine,
              opacity: float) -> None:
        font = _font(int(round(node.font_size * self.scale)),
                     bold=node.font_weight in ("bold", "700", "800",
                                               "900"))
        rgb = _parse_rgb(node.fill.color)
        alpha = int(round(255 * node.fill.opacity * opacity))
        color = (int(rgb[0] * 255), int(rgb[1] * 255),
                 int(rgb[2] * 255), alpha)
        if node.text_path:
            self._text_on_path(node, transform, font, color)
            return
        # Anchor in device pixels: node transform, then the view
        # transform (which already carries the output scale).
        vx, vy = self.view.apply(transform.apply((node.x, node.y)))
        self.overlay_draw.text((vx, vy), node.text, font=font,
                               fill=color, anchor="ls")

    def _text_on_path(self, node: TextNode, transform: Affine,
                      font: Any, color: tuple[int, int, int, int]
                      ) -> None:
        path = self.doc.defs.paths.get(node.text_path or "")
        if path is None or not node.text:
            return
        device = self.device(path, transform)
        contours = device.flatten(0.5)
        if not contours or not contours[0]:
            return
        points = contours[0]
        # Cumulative arclength along the flattened path.
        lengths = [0.0]
        for a, b in pairwise(points):
            lengths.append(lengths[-1] + math.hypot(
                b[0] - a[0], b[1] - a[1]))
        total = lengths[-1]
        if total <= 0:
            return
        advance = node.start_offset
        for char in node.text:
            width = max(font.getlength(char), 1.0)
            mid = advance + width / 2.0
            if mid > total:
                break
            index = 0
            while index < len(lengths) - 2 \
                    and lengths[index + 1] < mid:
                index += 1
            a, b = points[index], points[index + 1]
            seg = lengths[index + 1] - lengths[index] or 1.0
            t = (mid - lengths[index]) / seg
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            angle = math.degrees(math.atan2(b[1] - a[1],
                                            b[0] - a[0]))
            char_img = Image.new("RGBA", (int(width) + 4,
                                          int(node.font_size
                                              * self.scale) + 8),
                                 (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((2, 4), char, font=font, fill=color,
                           anchor="ls")
            rotated = char_img.rotate(-angle, expand=True,
                                      resample=Image.BICUBIC)
            self.overlay.alpha_composite(
                rotated, (int(x - rotated.width / 2),
                          int(y - rotated.height / 2)))
            advance += width

    def _image(self, node: ImageNode, transform: Affine,
               opacity: float) -> None:
        x, y = self.view.apply((node.x, node.y))
        w = node.width * self.view.a
        h = node.height * self.view.d
        if w <= 0 or h <= 0:
            return
        href = node.href or ""
        pasted = False
        if href.startswith("data:image/"):
            try:
                payload = href.split(",", 1)[1]
                raw = base64.b64decode(payload)
                image = Image.open(io.BytesIO(raw)).convert("RGBA")
                image = image.resize((max(1, int(w)), max(1, int(h))),
                                     Image.LANCZOS)
                if opacity < 1.0:
                    alpha = image.split()[3].point(
                        lambda v: int(v * opacity))
                    image.putalpha(alpha)
                self.overlay.alpha_composite(
                    image, (int(x), int(y)))
                pasted = True
            except (ValueError, OSError):
                pasted = False
        if not pasted:
            # Placeholder frame: grey border + X.
            draw = self.overlay_draw
            box = [int(x), int(y), int(x + w), int(y + h)]
            draw.rectangle(box, outline=(128, 128, 128, 255), width=1)
            draw.line([box[0], box[1], box[2], box[3]],
                      fill=(128, 128, 128, 255), width=1)
            draw.line([box[2], box[1], box[0], box[3]],
                      fill=(128, 128, 128, 255), width=1)


def render_to_image(doc: VectorDocument, scale: float = 1.0,
                    background: tuple[int, int, int] | None = None
                    ) -> Image.Image:
    """Render a document to a PIL RGBA image."""
    if scale <= 0:
        raise VectorError("render scale must be positive",
                          kind="invalid_argument")
    return _Renderer(doc, scale, background).render()


def render_document(doc: VectorDocument, scale: float = 1.0,
                    background: tuple[int, int, int] | None = None
                    ) -> bytes:
    """Render a document to PNG bytes."""
    image = render_to_image(doc, scale=scale, background=background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_svg(data: bytes, scale: float = 1.0,
               background: tuple[int, int, int] | None = None
               ) -> bytes:
    """Parse SVG bytes and render them to PNG bytes."""
    from .svg import parse_svg
    return render_document(parse_svg(data), scale=scale,
                           background=background)
