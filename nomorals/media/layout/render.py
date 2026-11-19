"""The page renderer — a page (or a whole document) to PNG.

Renders through PIL at the requested scale: shapes from the
vector organ's flattened paths, fills through swatches + tints,
strokes, placed images from the package assets, and text composed
with the real font door (threaded frames and multi-column frames
flow through :func:`compose_story`). Hidden layers, hidden items
and non-printing items are not drawn. What it does not do yet:
gradient fills render as their first stop, and a missing image
renders as a grey placeholder box — both named in the survey.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from ..vector.geom import Affine
from .compose import compose_story, font_for, measure_text
from .errors import LayoutError
from .model import Item, LayoutDocument, Page

__all__ = ["render_document", "render_page"]

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PLACEHOLDER_FILL = (220, 220, 220)
_PLACEHOLDER_EDGE = (200, 0, 0)


def _resolve_color(doc: LayoutDocument, value) -> tuple[int, int, int] | None:
    """A swatch name or ``#rrggbb`` colour as an RGB triple."""
    if value is None:
        return None
    if isinstance(value, str) and value.startswith("#"):
        text = value[1:]
        try:
            if len(text) == 6:
                return (int(text[0:2], 16), int(text[2:4], 16),
                        int(text[4:6], 16))
            if len(text) == 3:
                return (int(text[0] * 2, 16), int(text[1] * 2, 16),
                        int(text[2] * 2, 16))
        except ValueError:
            return None
        return None
    if isinstance(value, str):
        return doc.swatch_rgb(value)
    return None


def _tinted(rgb: tuple[int, int, int], tint: float
            ) -> tuple[int, int, int]:
    """A swatch colour at ``tint`` (1.0 = full colour, 0.0 =
    white) — upstream's tint rule."""
    tint = min(1.0, max(0.0, tint))
    return tuple(round(255 + (c - 255) * tint) for c in rgb)


def _fill_rgb(doc: LayoutDocument, item: Item
             ) -> tuple[int, int, int] | None:
    if item.fill.is_none():
        return None
    swatch = doc.swatch(item.fill.swatch)
    if swatch is None:
        return None
    if swatch.rgb is not None:
        return _tinted(tuple(swatch.rgb[:3]), item.fill.tint)
    if swatch.stops:
        # Gradients render as their first stop (flat) for now.
        first = swatch.stops[0] if isinstance(swatch.stops[0], dict) \
            else {}
        rgb = _resolve_color(doc, first.get("color"))
        if rgb is not None:
            return _tinted(rgb, item.fill.tint)
        return None
    rgb = swatch.resolve_rgb()
    return _tinted(rgb, item.fill.tint) if rgb is not None else None


def _stroke_rgb(doc: LayoutDocument, item: Item
               ) -> tuple[int, int, int] | None:
    if item.stroke.width <= 0:
        return None
    return _resolve_color(doc, item.stroke.color)


def _to_px(xf: Affine, scale: float, point: tuple[float, float]
           ) -> tuple[float, float]:
    x, y = xf.apply(point)
    return (x * scale, y * scale)


def _draw_shape(draw: ImageDraw.ImageDraw, doc: LayoutDocument,
                item: Item, xf: Affine, scale: float) -> None:
    fill = _fill_rgb(doc, item)
    stroke = _stroke_rgb(doc, item)
    width = max(1, int(round(item.stroke.width * scale))) if stroke else 1
    if item.shape == "oval":
        x0, y0, x1, y1 = item.local_bounds()
        (px0, py0) = _to_px(xf, scale, (x0, y0))
        (px1, py1) = _to_px(xf, scale, (x1, y1))
        draw.ellipse([px0, py0, px1, py1], fill=fill, outline=stroke,
                     width=width if stroke else 1)
        return
    if item.shape == "line":
        x0, y0, x1, y1 = item.local_bounds()
        p0 = _to_px(xf, scale, (x0, y0))
        p1 = _to_px(xf, scale, (x1, y1))
        if stroke is not None:
            draw.line([p0, p1], fill=stroke, width=width)
        return
    contours = item.path().flatten()
    if not contours:
        return
    for contour in contours:
        if len(contour) < 3:
            continue
        px = [_to_px(xf, scale, p) for p in contour]
        if fill is not None:
            draw.polygon(px, fill=fill)
        if stroke is not None:
            draw.line(px + [px[0]], fill=stroke, width=width, joint="curve")


def _draw_graphic(image: Image.Image, draw: ImageDraw.ImageDraw,
                  doc: LayoutDocument, item: Item, xf: Affine,
                  scale: float) -> None:
    # Bounds in spread space: the local bounds through the
    # composed transform.
    lx0, ly0, lx1, ly1 = item.local_bounds()
    corners = [xf.apply(p) for p in ((lx0, ly0), (lx1, ly0),
                                     (lx0, ly1), (lx1, ly1))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    px0, py0 = int(round(bx0 * scale)), int(round(by0 * scale))
    pw = max(1, int(round((bx1 - bx0) * scale)))
    ph = max(1, int(round((by1 - by0) * scale)))
    asset = item.content.get("asset")
    placed = None
    if isinstance(asset, str) and asset in doc.assets:
        try:
            placed = Image.open(io.BytesIO(doc.assets[asset])).convert("RGB")
        except Exception:
            placed = None
    if placed is None:
        draw.rectangle([px0, py0, px0 + pw - 1, py0 + ph - 1],
                       fill=_PLACEHOLDER_FILL, outline=_PLACEHOLDER_EDGE,
                       width=2)
        return
    placed = placed.resize((pw, ph), Image.LANCZOS)
    image.paste(placed, (px0, py0))


def _text_flows(doc: LayoutDocument) -> dict[str, list]:
    """Compose every text flow in the document. Frames linked by
    ``next_frame_id`` form one threaded flow (the story is composed
    across them in link order, spanning pages when the thread does);
    unlinked frames are solo flows. Returns
    ``{item_id: [ComposedLine, ...]}``."""
    items = [i for i in doc.all_items() if i.content_type() == "text"]
    by_id = {i.id: i for i in items}
    targeted = {i.text_options().next_frame_id for i in items}
    flows: dict[str, list] = {}
    chained: set[str] = set()
    for item in items:
        if item.id in targeted:
            continue  # starts mid-chain; its head drives the flow
        chain = [item]
        seen = {item.id}
        while True:
            nxt = chain[-1].text_options().next_frame_id
            if nxt is None or nxt not in by_id or nxt in seen:
                break
            chain.append(by_id[nxt])
            seen.add(nxt)
        chained.update(i.id for i in chain)
        story = doc.stories.get(item.story_id or "")
        if story is None:
            for member in chain:
                flows[member.id] = []
            continue
        frames = [_frame_boxes(doc, member) for member in chain]
        flat: list[tuple[float, float]] = []
        for boxes in frames:
            flat.extend(boxes)
        size = _story_size(story)
        composed = compose_story(story, flat, font_size=size)
        # Hand the composed lines back to their frames, column by
        # column: frame j owns composed.frames[sum(len(boxes[:j])):].
        index = 0
        for member, boxes in zip(chain, frames, strict=True):
            lines = []
            for _ in boxes:
                if index < len(composed.frames):
                    lines.extend(composed.frames[index].lines)
                index += 1
            flows[member.id] = lines
    for item in items:
        if item.id in chained:
            continue
        story = doc.stories.get(item.story_id or "")
        if story is None:
            flows[item.id] = []
            continue
        boxes = _frame_boxes(doc, item)
        composed = compose_story(story, boxes,
                                 font_size=_story_size(story))
        lines = []
        for frame in composed.frames:
            lines.extend(frame.lines)
        flows[item.id] = lines
    return flows


def _frame_boxes(doc: LayoutDocument, item: Item
                 ) -> list[tuple[float, float]]:
    """The text areas of ``item`` as (width, height) frames — one
    per column when the frame is multi-column."""
    options = item.text_options()
    x0, y0, x1, y1 = item.local_bounds()
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    columns = max(1, options.columns)
    if columns == 1:
        return [(width, height)]
    gutter = max(0.0, options.gutter)
    col_w = max(0.0, (width - gutter * (columns - 1)) / columns)
    return [(col_w, height)] * columns


def _story_size(story) -> float:
    for para in story.paras:
        for run in para.runs:
            if run.font_size is not None:
                return float(run.font_size)
    return 12.0


def _draw_text(draw: ImageDraw.ImageDraw, doc: LayoutDocument,
               item: Item, xf: Affine, scale: float,
               lines: list) -> None:
    if not lines:
        return
    story = doc.stories.get(item.story_id or "")
    if story is None:
        return
    size = _story_size(story)
    bold = False
    color_value = None
    align = "left"
    for para in story.paras:
        if para.align:
            align = para.align
        for run in para.runs:
            bold = bold or run.bold
            if run.color is not None:
                color_value = run.color
    color = _resolve_color(doc, color_value) or _BLACK
    font = font_for(size, bold)
    options = item.text_options()
    columns = max(1, options.columns)
    x0, y0, x1, y1 = item.local_bounds()
    frame_w = max(0.0, x1 - x0)
    gutter = max(0.0, options.gutter)
    col_w = (frame_w if columns == 1 else
             max(0.0, (frame_w - gutter * (columns - 1)) / columns))
    # Lines are handed back column by column (see _text_flows):
    # split them back into columns by their y order — a column's
    # lines all share one composed frame, so count the lines per
    # column from the composition by re-deriving the split: lines
    # are stored in frame order, and each column's lines are the
    # ones that fit in its box. We recover the split by counting
    # lines whose baseline exceeds the column height... simpler:
    # the flows already ordered lines frame by frame; a column
    # boundary is where a line's y restarts below the previous
    # line's y. Recompute per column by walking the lines and
    # starting a new column when y decreases.
    column_lines: list[list] = [[]]
    last_y = None
    for line in lines:
        if last_y is not None and line.y < last_y - 0.5:
            column_lines.append([])
        column_lines[-1].append(line)
        last_y = line.y
    for col_index, col in enumerate(column_lines):
        col_x = x0 + col_index * (col_w + gutter)
        for line in col:
            text_w = measure_text(line.text, size, bold)
            if align == "center":
                off = max(0.0, (col_w - text_w) / 2.0)
            elif align == "right":
                off = max(0.0, col_w - text_w)
            else:
                off = 0.0
            px, py = _to_px(xf, scale, (col_x + off, y0 + line.y))
            draw.text((px, py), line.text, font=font, fill=color,
                      anchor="ls")


def _render_page_image(doc: LayoutDocument, page: Page,
                       scale: float) -> Image.Image:
    width = max(1, int(round(page.width * scale)))
    height = max(1, int(round(page.height * scale)))
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    layers = {layer.id: layer for layer in doc.layers}
    flows = _text_flows(doc)

    def draw_item(item: Item, parent: Affine | None) -> None:
        if item.hidden or item.nonprinting:
            return
        layer = layers.get(item.layer)
        if layer is not None and not layer.visible:
            return
        xf = item.xf if parent is None else parent * item.xf
        if item.content_type() == "group":
            for child in item.content_items():
                draw_item(child, xf)
            return
        if item.content_type() == "text":
            _draw_text(draw, doc, item, xf, scale,
                       flows.get(item.id, []))
            return
        if item.content_type() == "graphic":
            _draw_graphic(image, draw, doc, item, xf, scale)
            return
        _draw_shape(draw, doc, item, xf, scale)

    for item in page.items:
        draw_item(item, None)
    return image


def render_page(doc: LayoutDocument, page_id: str, scale: float = 1.0
                ) -> bytes:
    """Render one page to PNG bytes."""
    if scale is None or scale <= 0:
        raise LayoutError(f"render scale must be positive, got {scale}",
                          kind="invalid_argument")
    page = doc.find_page(page_id)
    if page is None:
        raise LayoutError(f"no page {page_id!r}", kind="not_found")
    image = _render_page_image(doc, page, float(scale))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_document(doc: LayoutDocument, scale: float = 1.0
                    ) -> list[bytes]:
    """Render every page to PNG bytes, in document order."""
    return [render_page(doc, page.id, scale=scale)
            for spread in doc.spreads for page in spread.pages]
