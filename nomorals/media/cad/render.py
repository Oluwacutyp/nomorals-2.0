"""A real drawing renderer: sketch → PNG through PIL.

The drawing is fitted to the page with a margin, Y flipped (drawing
Y grows upwards, image Y grows downwards), and drawn with a light
grid behind it. Entities render in their layer colour (or black);
hidden, frozen or locked-off layers are skipped — a drawing whose
only layer is hidden renders all white, as the test pins.

What renders honestly: lines, arcs, circles, ellipses, splines
(sampled), points, polylines (filled when ``fill``), text (measured
with the real DejaVuSans font door the layout organ uses),
dimensions (extension lines, arrows, measured value), block inserts
(expanded in place, one level). What does not render is named in
``unrendered`` on the returned image info — the renderer never draws
a placeholder and calls it the thing.
"""
from __future__ import annotations

import io
import logging
import math

from PIL import Image, ImageDraw, ImageFont

from ..typography import get_service, pil_font_for
from .errors import CadError
from .geom import ccw_sweep, spline_eval
from .model import (
    Arc,
    CadDocument,
    CadPoint,
    CadText,
    Circle,
    Dimension,
    Ellipse,
    Insert,
    Line,
    Polyline,
    Spline,
)

__all__ = ["render_document", "render_to_image"]

_log = logging.getLogger(__name__)


def _font(size: float, bold: bool = False):
    """CAD text wears the cad context (JetBrains Mono): a
    dimension reads the same width it draws. A total miss is
    logged and the Pillow bitmap default carries the label —
    never a silent slide to an unrelated face. The old DejaVu
    ladder is gone (gapfix 164)."""
    size = max(6, int(round(size)))
    try:
        return pil_font_for(get_service(), "cad", size,
                            weight=700 if bold else 400)
    except Exception as exc:  # noqa: BLE001 - named miss
        _log.warning("cad font could not resolve: %s", exc)
        return ImageFont.load_default()


def _parse_color(color: str | None) -> tuple[int, int, int]:
    if not color:
        return (0, 0, 0)
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16),
                int(text[4:6], 16))
    except ValueError:
        return (0, 0, 0)


class _View:
    """Drawing → image coordinates (Y flipped, fitted with margin)."""

    def __init__(self, doc: CadDocument, width: int, height: int):
        self.width = width
        self.height = height
        margin = 24
        bounds = doc.bounds()
        if bounds is None:
            bounds = (0.0, 0.0, 100.0, 100.0)
        x0, y0, x1, y1 = bounds
        span_x = max(x1 - x0, 1e-9)
        span_y = max(y1 - y0, 1e-9)
        scale = min((width - 2 * margin) / span_x,
                    (height - 2 * margin) / span_y)
        self.scale = scale
        self.origin = (margin - x0 * scale,
                       height - margin + y0 * scale)
        self.bounds = bounds

    def __call__(self, point) -> tuple[float, float]:
        return (self.origin[0] + point[0] * self.scale,
                self.origin[1] - point[1] * self.scale)


def render_to_image(doc: CadDocument, width: int = 800,
                    height: int = 600) -> tuple[Image.Image, dict]:
    """Render the document to a PIL image. Returns ``(image, info)``
    where ``info`` carries the view scale and the ``unrendered`` list
    of entity ids that were skipped honestly (never faked)."""
    if width <= 0 or height <= 0:
        raise CadError("render size must be positive",
                       kind="invalid_argument")
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    view = _View(doc, width, height)
    hidden = {layer.name for layer in doc.layers
              if not layer.visible or layer.frozen}
    visible = [entity for entity in doc.entities.values()
               if entity.layer not in hidden]
    # The grid orients visible content; with nothing visible to
    # draw there is no grid either, so a drawing whose only layer
    # is hidden renders all white.
    if visible:
        _draw_grid(draw, view, width, height)
    unrendered: list[str] = []
    for entity in visible:
        try:
            _draw_entity(draw, view, doc, entity)
        except Exception:
            unrendered.append(entity.id)
    info = {"width": width, "height": height, "scale": view.scale,
            "bounds": list(view.bounds), "unrendered": unrendered}
    return image, info


def _draw_grid(draw: ImageDraw.ImageDraw, view: _View,
               width: int, height: int) -> None:
    """A light grid at a round drawing-unit spacing."""
    x0, y0, x1, y1 = view.bounds
    span = max(x1 - x0, y1 - y0)
    step = 10 ** math.floor(math.log10(max(span, 1.0) / 10.0))
    if step <= 0:
        step = 1.0
    # Snap the grid to multiples of the step.
    start_x = math.floor(x0 / step) * step
    start_y = math.floor(y0 / step) * step
    grid_color = (235, 235, 235)
    x = start_x
    while x <= x1:
        px, _ = view((x, y0))
        draw.line([(px, 0), (px, height)], fill=grid_color)
        x += step
    y = start_y
    while y <= y1:
        _, py = view((x0, y))
        draw.line([(0, py), (width, py)], fill=grid_color)
        y += step
    # Axes through the origin, slightly stronger.
    if x0 <= 0 <= x1:
        px, _ = view((0.0, y0))
        draw.line([(px, 0), (px, height)], fill=(220, 220, 220))
    if y0 <= 0 <= y1:
        _, py = view((x0, 0.0))
        draw.line([(0, py), (width, py)], fill=(220, 220, 220))


def _draw_entity(draw, view, doc, entity) -> None:
    color = _parse_color(entity.color)
    if entity.layer != "0":
        layer = doc.layer(entity.layer)
        if layer is not None and layer.color not in (None, "#ffffff"):
            color = _parse_color(layer.color)
    if isinstance(entity, Line):
        draw.line([view(entity.a), view(entity.b)], fill=color, width=2)
    elif isinstance(entity, Circle):
        cx, cy = view(entity.center)
        r = entity.radius * view.scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=color, width=2)
    elif isinstance(entity, Arc):
        cx, cy = view(entity.center)
        r = entity.radius * view.scale
        start = math.degrees(entity.start)
        end = math.degrees(entity.end)
        # PIL arcs run clockwise from 3 o'clock in image space; the
        # drawing is Y-flipped, so the sweep flips too.
        draw.arc([cx - r, cy - r, cx + r, cy + r],
                 start=-end, end=-start, fill=color, width=2)
    elif isinstance(entity, Ellipse):
        _draw_ellipse(draw, view, entity, color)
    elif isinstance(entity, Spline):
        pts = [spline_eval(entity.points, t / 48.0, entity.closed,
                           entity.fit) for t in range(49)]
        draw.line([view(p) for p in pts], fill=color, width=2)
    elif isinstance(entity, CadPoint):
        x, y = view(entity.p)
        draw.line([(x - 4, y), (x + 4, y)], fill=color, width=2)
        draw.line([(x, y - 4), (x, y + 4)], fill=color, width=2)
    elif isinstance(entity, Polyline):
        pts = [view(p) for p in entity.vertices]
        if entity.closed and entity.fill:
            draw.polygon(pts, fill=color)
        elif entity.closed:
            draw.line(pts + [pts[0]], fill=color, width=2)
        else:
            draw.line(pts, fill=color, width=2)
    elif isinstance(entity, CadText):
        _draw_text(draw, view, entity, color)
    elif isinstance(entity, Dimension):
        _draw_dimension(draw, view, doc, entity, color)
    elif isinstance(entity, Insert):
        _draw_insert(draw, view, doc, entity, color)
    else:
        raise CadError(f"cannot render {type(entity).__name__}",
                       kind="render")


def _draw_ellipse(draw, view, entity: Ellipse, color) -> None:
    # Sample the ellipse: centre + major·cos t + minor·sin t, with
    # minor = perp(major) · ratio.
    major = entity.major
    minor = (-major[1] * entity.ratio, major[0] * entity.ratio)
    start = entity.start
    end = entity.end
    sweep = math.tau if entity.is_full() else ccw_sweep(start, end)
    steps = 64
    pts = []
    for i in range(steps + 1):
        t = start + sweep * i / steps
        pts.append((entity.center[0] + major[0] * math.cos(t)
                    + minor[0] * math.sin(t),
                    entity.center[1] + major[1] * math.cos(t)
                    + minor[1] * math.sin(t)))
    draw.line([view(p) for p in pts], fill=color, width=2)


def _draw_text(draw, view, entity: CadText, color) -> None:
    x, y = view(entity.position)
    size = max(6.0, entity.height * view.scale)
    font = _font(size)
    if entity.rotation:
        # PIL draws text upright; rotated text is drawn on a small
        # tile and pasted back rotated.
        tile = Image.new("RGBA", (int(size * len(entity.text) + 20),
                                  int(size * 2 + 8)), (0, 0, 0, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((4, 4), entity.text, font=font,
                       fill=color + (255,))
        tile = tile.rotate(-entity.rotation, expand=True,
                           resample=Image.BICUBIC)
        draw._image.paste(tile, (int(x), int(y) - tile.height // 2),
                          tile)
    else:
        draw.text((x, y), entity.text, font=font, fill=color)


def _ref_point(doc: CadDocument, ref):
    """The drawing point a dimension reference denotes, or None."""
    from .model import Sub
    entity = doc.entities.get(ref.handle)
    if entity is None:
        return None
    sub = ref.sub
    if isinstance(entity, Line):
        if sub == Sub.START:
            return entity.a
        if sub == Sub.END:
            return entity.b
        if sub == Sub.MID:
            return ((entity.a[0] + entity.b[0]) / 2,
                    (entity.a[1] + entity.b[1]) / 2)
    elif isinstance(entity, (Circle, Ellipse)):
        if sub == Sub.CENTER:
            return entity.center
    elif isinstance(entity, CadPoint):
        return entity.p
    return None


def _draw_dimension(draw, view, doc, entity: Dimension, color) -> None:
    # Witness lines from each measured point to the dimension
    # position, an arrow at the position, and the measured value.
    # The value is the dimension's own measured value — never
    # re-typed.
    x, y = view(entity.position)
    for ref in entity.refs:
        point = _ref_point(doc, ref)
        if point is None:
            continue
        px, py = view(point)
        draw.line([(px, py), (x, y)], fill=(180, 180, 180), width=1)
    label = entity.text or _format_value(entity)
    font = _font(max(6.0, 11.0))
    draw.text((x + 4, y - 18), label, font=font, fill=color)
    for sign in (1, -1):
        draw.polygon([(x, y), (x + sign * 8, y - 3),
                      (x + sign * 8, y + 3)], fill=color)


def _format_value(entity: Dimension) -> str:
    if entity.kind == "angular":
        return f"{entity.value:.1f}°"
    if entity.kind == "diameter":
        return f"Ø{entity.value:.3f}"
    if entity.kind == "radius":
        return f"R{entity.value:.3f}"
    return f"{entity.value:.3f}"


def _draw_insert(draw, view, doc, entity: Insert, color) -> None:
    block = doc.blocks.get(entity.block)
    if block is None:
        return
    # One level of expansion: each block entity drawn translated
    # (and rotated/scaled) by the insert.
    angle = math.radians(entity.rotation)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    sx, sy = entity.scale

    def place(p):
        x, y = p[0] * sx, p[1] * sy
        x, y = (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        return (entity.position[0] + x, entity.position[1] + y)

    for sub in block.entities:
        import copy
        clone = copy.deepcopy(sub)
        _place_entity(clone, place)
        _draw_entity(draw, view, doc, clone)


def _place_entity(entity, place) -> None:
    if isinstance(entity, Line):
        entity.a, entity.b = place(entity.a), place(entity.b)
    elif isinstance(entity, (Arc, Circle, Ellipse)):
        entity.center = place(entity.center)
    elif isinstance(entity, Spline):
        entity.points = [place(p) for p in entity.points]
    elif isinstance(entity, CadPoint):
        entity.p = place(entity.p)
    elif isinstance(entity, Polyline):
        entity.vertices = [place(p) for p in entity.vertices]
    elif isinstance(entity, (CadText, Dimension)):
        entity.position = place(entity.position)


def render_document(doc: CadDocument, width: int = 800,
                    height: int = 600) -> bytes:
    """Render the document to PNG bytes."""
    image, _info = render_to_image(doc, width, height)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
