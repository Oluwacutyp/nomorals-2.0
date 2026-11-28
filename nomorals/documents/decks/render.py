"""Slide renderer — slide → PNG through Devon's own PIL path.

Follows the upstream ``crates/render`` design (``geometry.rs`` +
``images.rs`` + ``text`` layout): shapes draw in document order,
text lays out run by run with the run's own formatting, pictures
paste for real, tables draw as grids. The renderer is honest
about what it does not draw: charts, SmartArt/OLE/media shapes
and animations are reported in the ``unrendered`` list (a chart
gets a labelled frame, never a faked plot) — see
:func:`render_slide`.

Units: DrawingML EMU (914400 per inch). Font sizes are
half-points in the XML; pixels come from the requested output
width and the slide's own size, so a 32pt title is 32pt at any
output width.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...media.typography import get_service, pil_font_for
from ..errors import DocumentError
from .model import EMU_PER_INCH, DeckDocument, DeckShape, DeckSlide

_log = logging.getLogger(__name__)

__all__ = ["render_slide", "render_slide_to_png"]

#: Theme scheme colours the renderer cannot resolve from the file
#: fall back to the Office defaults the fixture theme carries.
_SCHEME_DEFAULTS = {
    "dk1": "000000", "lt1": "FFFFFF", "dk2": "1F497D",
    "lt2": "EEECE1", "accent1": "4472C4", "accent2": "ED7D31",
    "accent3": "A5A5A5", "accent4": "FFC000", "accent5": "5B9BD5",
    "accent6": "70AD47", "hlink": "0563C1", "folHlink": "954F72",
}


def _font(size_px: int, bold: bool = False,
          family: str | None = None) -> Any:
    """The PIL font for ``size_px`` points.

    A named ``family`` resolves through the typography service —
    the stocked library first, then the parsed host catalogue,
    so a deck's Calibri/Arial resolves to the real face on a
    Windows host that carries it. No family wears the card
    context (Inter). A family that cannot resolve anywhere is a
    named miss (logged) and the Pillow bitmap default — never a
    silent slide to an unrelated face. The old DejaVu ladder is
    gone (gapfix 164).
    """
    size_px = max(8, int(size_px))
    try:
        if family:
            return pil_font_for(get_service(), "text", size_px,
                                family=family,
                                weight=700 if bold else 400)
        return pil_font_for(get_service(), "card", size_px,
                            weight=700 if bold else 400)
    except Exception as exc:  # noqa: BLE001 - named miss
        _log.warning("deck font %r could not resolve: %s",
                     family or "card", exc)
        try:
            return ImageFont.load_default(size=size_px)
        except TypeError:
            return ImageFont.load_default()


def _resolve_color(color: str | None,
                   theme: dict[str, str]) -> tuple[int, int, int]:
    """A fill/run colour to RGB — hex, or ``scheme:name`` via the
    theme (Office defaults when the file names no theme)."""
    if not color:
        return (0, 0, 0)
    if color.startswith("scheme:"):
        name = color.split(":", 1)[1]
        hex_value = theme.get(name, _SCHEME_DEFAULTS.get(name, "000000"))
        color = hex_value
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    try:
        return (int(color[0:2], 16), int(color[2:4], 16),
                int(color[4:6], 16))
    except (ValueError, IndexError):
        return (0, 0, 0)


class _Renderer:
    """One slide render: EMU→pixel math + PIL drawing."""

    def __init__(self, doc: DeckDocument, width: int) -> None:
        self.doc = doc
        slide_w_in = doc.slide_size[0] / EMU_PER_INCH
        slide_h_in = doc.slide_size[1] / EMU_PER_INCH
        self.px_per_inch = width / slide_w_in
        self.width = width
        self.height = max(1, int(round(slide_h_in * self.px_per_inch)))
        self.unrendered: list[str] = []

    def emu(self, value: int) -> int:
        return int(round(value / EMU_PER_INCH * self.px_per_inch))

    def font_px(self, size_half_points: int | None) -> int:
        points = (size_half_points or 36) / 2.0
        return max(8, int(round(points / 72.0 * self.px_per_inch)))

    def render(self, slide: DeckSlide) -> bytes:
        bg = (255, 255, 255)
        if slide.background_fill and slide.background_fill not in (
                "gradient", "picture"):
            bg = _resolve_color(slide.background_fill,
                                self.doc.theme_colors)
        image = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(image, "RGBA")
        for shape in slide.shapes:
            self._draw_shape(draw, image, shape)
        if slide.has_animation:
            self.unrendered.append(
                f"slide {slide.index}: animations not rendered")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    # ── shapes ────────────────────────────────────────────────────
    def _draw_shape(self, draw: ImageDraw.ImageDraw, image: Image.Image,
                    shape: DeckShape, origin: tuple[int, int] = (0, 0),
                    scale: float = 1.0) -> None:
        x = origin[0] + self.emu(int(shape.xfrm["x"] * scale))
        y = origin[1] + self.emu(int(shape.xfrm["y"] * scale))
        w = self.emu(int(shape.xfrm["cx"] * scale))
        h = self.emu(int(shape.xfrm["cy"] * scale))
        if shape.kind == "group":
            self._draw_group(draw, image, shape, (x, y))
            return
        if shape.kind == "picture":
            self._draw_picture(image, shape, (x, y, w, h))
            return
        if shape.kind == "table":
            self._draw_table(draw, shape, (x, y, w, h))
            return
        if shape.kind == "chart":
            self._draw_chart_frame(draw, shape, (x, y, w, h))
            return
        if shape.kind == "opaque":
            self.unrendered.append(
                f"shape {shape.shape_id} ({shape.name or 'opaque'}): "
                f"unmodelled shape not rendered")
            return
        fill = None
        if shape.fill and shape.fill not in ("gradient", "picture"):
            fill = _resolve_color(shape.fill, self.doc.theme_colors)
        elif shape.fill in ("gradient", "picture"):
            self.unrendered.append(
                f"shape {shape.shape_id}: {shape.fill} fill drawn as "
                f"light grey")
            fill = (230, 230, 230)
        outline = None
        outline_w = 1
        if shape.line_color:
            outline = _resolve_color(shape.line_color,
                                     self.doc.theme_colors)
            outline_w = max(1, self.emu(shape.line_width_emu or 9525))
        geom = shape.geometry or "rect"
        box = [x, y, x + max(w, 1), y + max(h, 1)]
        if geom == "ellipse":
            draw.ellipse(box, fill=fill, outline=outline,
                         width=outline_w)
        elif geom == "roundRect":
            draw.rounded_rectangle(box, radius=min(w, h) // 8,
                                   fill=fill, outline=outline,
                                   width=outline_w)
        elif geom == "triangle":
            draw.polygon([(x + w // 2, y), (x + w, y + h), (x, y + h)],
                         fill=fill, outline=outline)
        elif geom == "diamond":
            draw.polygon([(x + w // 2, y), (x + w, y + h // 2),
                          (x + w // 2, y + h), (x, y + h // 2)],
                         fill=fill, outline=outline)
        elif geom == "line":
            draw.line([x, y, x + w, y + h], fill=outline or (0, 0, 0),
                      width=outline_w)
        else:
            draw.rectangle(box, fill=fill, outline=outline,
                           width=outline_w)
        if shape.paragraphs:
            self._draw_text(draw, shape, (x, y, w, h))

    def _draw_group(self, draw: ImageDraw.ImageDraw, image: Image.Image,
                    shape: DeckShape, origin: tuple[int, int]) -> None:
        # Children live in the group's child space (chOff/chExt in
        # the XML); the model keeps the group's xfrm only, so children
        # draw at the group's origin with their own offsets. A group
        # whose child extent is zero draws its children 1:1.
        for child in shape.children:
            self._draw_shape(draw, image, child, origin)

    def _draw_picture(self, image: Image.Image, shape: DeckShape,
                      box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        if not shape.image_part:
            self.unrendered.append(
                f"shape {shape.shape_id}: picture without a media part")
            return
        data = self.doc._package.get(shape.image_part)
        if data is None:
            self.unrendered.append(
                f"shape {shape.shape_id}: media part "
                f"{shape.image_part!r} missing")
            return
        try:
            pic = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception:
            self.unrendered.append(
                f"shape {shape.shape_id}: media part "
                f"{shape.image_part!r} is not a readable image")
            return
        pic = pic.resize((max(1, w), max(1, h)))
        image.paste(pic, (x, y), pic)

    def _draw_table(self, draw: ImageDraw.ImageDraw, shape: DeckShape,
                    box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        rows = shape.table or []
        if not rows:
            return
        n_rows = len(rows)
        n_cols = max(len(r) for r in rows)
        row_h = max(1, h // n_rows)
        col_w = max(1, w // n_cols)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                cell_box = [x + c * col_w, y + r * row_h,
                            x + (c + 1) * col_w, y + (r + 1) * row_h]
                draw.rectangle(cell_box, outline=(180, 180, 180))
                if cell:
                    font = _font(self.font_px(36), bold=(r == 0))
                    draw.text((cell_box[0] + 4, cell_box[1] + 4), cell,
                              fill=(0, 0, 0), font=font)

    def _draw_chart_frame(self, draw: ImageDraw.ImageDraw,
                          shape: DeckShape,
                          box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline=(120, 120, 120),
                       width=2)
        label = shape.chart_title or "Chart"
        font = _font(self.font_px(36), bold=True)
        draw.text((x + 8, y + h // 2 - 10), f"[chart: {label}]",
                  fill=(90, 90, 90), font=font)
        self.unrendered.append(
            f"shape {shape.shape_id}: chart "
            f"{label!r} drawn as a labelled frame, not plotted")

    # ── text ──────────────────────────────────────────────────────
    def _draw_text(self, draw: ImageDraw.ImageDraw, shape: DeckShape,
                   box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        cursor_y = y + 4
        for para in shape.paragraphs:
            if not para.runs:
                cursor_y += self.font_px(36)
                continue
            # word-wrap the paragraph's runs across the box width
            lines = self._wrap_paragraph(para, w - 8)
            for line_runs in lines:
                cursor_x = x + 4
                line_h = 0
                for run, text in line_runs:
                    size = self.font_px(run.size_half_points)
                    font = _font(size, bold=run.bold, family=run.font)
                    line_h = max(line_h, size)
                    color = _resolve_color(run.color,
                                           self.doc.theme_colors)
                    draw.text((cursor_x, cursor_y), text, fill=color,
                              font=font)
                    cursor_x += int(font.getlength(text))
                cursor_y += line_h + 2
                if cursor_y > y + h:
                    return

    def _wrap_paragraph(self, para: Any, max_width: int
                        ) -> list[list[tuple[Any, str]]]:
        """Split a paragraph's runs into lines that fit max_width.

        Runs are split on word boundaries; a word longer than the
        box is hard-split. Each line is a list of (run, text)."""
        lines: list[list[tuple[Any, str]]] = []
        current: list[tuple[Any, str]] = []
        current_w = 0
        for run in para.runs:
            size = self.font_px(run.size_half_points)
            font = _font(size, bold=run.bold, family=run.font)
            words = run.text.split(" ")
            for i, word in enumerate(words):
                piece = word + (" " if i < len(words) - 1 else "")
                piece_w = int(font.getlength(piece))
                if current and current_w + piece_w > max_width:
                    lines.append(current)
                    current = []
                    current_w = 0
                    piece = piece.lstrip()
                    piece_w = int(font.getlength(piece))
                while piece_w > max_width and len(piece) > 1:
                    # hard-split a word longer than the box
                    cut = len(piece) // 2
                    while cut > 1 and int(font.getlength(
                            piece[:cut])) > max_width:
                        cut //= 2
                    current.append((run, piece[:cut]))
                    lines.append(current)
                    current = []
                    current_w = 0
                    piece = piece[cut:]
                    piece_w = int(font.getlength(piece))
                if piece:
                    current.append((run, piece))
                    current_w += piece_w
        if current:
            lines.append(current)
        return lines


def render_slide(doc: DeckDocument, index: int, *,
                 width: int = 960) -> dict[str, Any]:
    """Render one slide to PNG bytes.

    Returns ``{png, width, height, unrendered}`` — ``unrendered``
    names everything on the slide the renderer did not draw
    (charts, opaque shapes, animations, gradient/picture fills).
    """
    if width <= 0 or width > 4096:
        raise DocumentError(f"render width {width} out of range "
                            f"(1..4096)", kind="invalid_argument")
    slide = doc.slide(index)
    renderer = _Renderer(doc, width)
    png = renderer.render(slide)
    return {"png": png, "width": renderer.width,
            "height": renderer.height,
            "unrendered": renderer.unrendered}


def render_slide_to_png(doc: DeckDocument, index: int, *,
                        width: int = 960) -> bytes:
    """Render one slide to PNG bytes (the ``png`` half of
    :func:`render_slide`)."""
    return render_slide(doc, index, width=width)["png"]
