"""FigCraft auto layout — the flexbox-shaped engine (lane 105).

Follows upstream's auto-layout design (``crates/layout`` + the
auto-layout parity doc): a frame's ``layout`` says the flow —
``mode`` vertical/horizontal, ``wrap``, ``gap``/``cross_gap``,
``padding`` [top right bottom left], ``main_align``
(min/center/max/space-between) and ``cross_align`` (min/center/max)
— and each child's ``child_layout`` says its sizing per axis:
``fixed`` (the node's own width/height), ``hug`` (measured
content: text by font metrics, a frame by its own layout) or
``fill`` (an equal share of the free space). Absolute children sit
outside the flow, positioned by their transform.

Text is measured through the DejaVu font door (the same door the
vector and decks renders use — Inter is not installed on this box,
so widths are DejaVu widths, stated wherever they are pinned).
Grid mode places children by their ``grid_column``/``grid_row``;
grid auto-placement is not built (see the survey).
"""

from __future__ import annotations

from typing import Any

from PIL import ImageFont

from .model import DesignDocument

__all__ = ["compute_layout", "measure_text"]

_FONT_NAMES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")


def _font(size_px: float, bold: bool = False) -> Any:
    size = max(1, int(round(size_px)))
    names = (_FONT_NAMES[0], _FONT_NAMES[1]) if bold else (_FONT_NAMES[1],)
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def measure_text(text: str, font_family: str = "Inter", font_size: float = 16.0,
                 bold: bool = False, line_height: float = 140.0) -> tuple[float, float]:
    """Measure ``text``: (width, height) in design points.

    Width is the longest line's DejaVu advance; height is
    ``lines * font_size * line_height / 100``. ``font_family`` is
    recorded for the caller but the measurement uses the DejaVu
    door (Inter is not installed here).
    """
    font = _font(font_size, bold=bold)
    lines = str(text).split("\n") or [""]
    width = max((font.getlength(line) for line in lines), default=0.0)
    height = len(lines) * float(font_size) * float(line_height) / 100.0
    return float(width), float(height)


def _clamp(value: float, low: float | None, high: float | None) -> float:
    if low is not None:
        value = max(value, float(low))
    if high is not None:
        value = min(value, float(high))
    return value


def _text_style(node: dict[str, Any]) -> tuple[float, bool, float]:
    style = node.get("style") or {}
    size = float(style.get("font_size") or 16.0)
    bold = str(style.get("font_style") or "").lower() in ("bold", "medium", "700", "800")
    line_height = style.get("line_height") or {}
    if isinstance(line_height, dict) and line_height.get("unit") == "percent":
        lh = float(line_height.get("value") or 140.0)
    else:
        lh = 140.0
    return size, bold, lh


def _hug_size(doc: DesignDocument, node: dict[str, Any],
              boxes: dict[int, dict[str, float]]) -> tuple[float, float]:
    """Content size of a hug-sized node (text or auto-layout frame)."""
    ntype = node.get("type")
    if ntype == "text":
        size, bold, lh = _text_style(node)
        return measure_text(node.get("characters", ""), "Inter", size,
                            bold=bold, line_height=lh)
    if ntype in ("frame", "group", "component", "componentSet", "instance"):
        # A hug frame sizes to its flow children; lay it out with
        # unconstrained space and take the content extent.
        sub = _layout_children(doc, node, boxes, 0.0, 0.0, None, None)
        return sub[0], sub[1]
    return float(node.get("width") or 0.0), float(node.get("height") or 0.0)


def _layout_children(doc: DesignDocument, frame: dict[str, Any],
                     boxes: dict[int, dict[str, float]],
                     origin_x: float, origin_y: float,
                     avail_w: float | None, avail_h: float | None
                     ) -> tuple[float, float]:
    """Lay out ``frame``'s children; returns (content_w, content_h).

    Writes absolute boxes for every descendant into ``boxes``.
    ``avail_w``/``avail_h`` are the content-box sizes the parent
    offers (None = unconstrained, hug).
    """
    layout = frame.get("layout") or {}
    children = [c for c in doc.children(int(frame["id"]))]
    flow = [c for c in children
            if not (c.get("child_layout") or {}).get("absolute")]
    absolute = [c for c in children
                if (c.get("child_layout") or {}).get("absolute")]
    for child in absolute:
        transform = child.get("transform") or [1, 0, 0, 1, 0, 0]
        _place_node(doc, child, boxes,
                    origin_x + float(transform[4]),
                    origin_y + float(transform[5]),
                    None, None)
    if not layout:
        # No auto layout: children sit at their transforms.
        for child in flow:
            transform = child.get("transform") or [1, 0, 0, 1, 0, 0]
            _place_node(doc, child, boxes,
                        origin_x + float(transform[4]),
                        origin_y + float(transform[5]),
                        None, None)
        width = float(frame.get("width") or 0.0)
        height = float(frame.get("height") or 0.0)
        return width, height

    mode = layout.get("mode", "vertical")
    horizontal = mode == "horizontal"
    gap = float(layout.get("gap") or 0.0)
    cross_gap = float(layout.get("cross_gap") or gap)
    padding = layout.get("padding") or [0.0, 0.0, 0.0, 0.0]
    pad_t, _pad_r, _pad_b, pad_l = (float(padding[0]), float(padding[1]),
                                  float(padding[2]), float(padding[3]))
    main_align = layout.get("main_align", "min")
    cross_align = layout.get("cross_align", "min")
    wrap = bool(layout.get("wrap"))

    def main_size(child: dict[str, Any]) -> float:
        cl = child.get("child_layout") or {}
        sizing = cl.get("sizing_h" if horizontal else "sizing_v")
        if sizing == "hug":
            w, h = _hug_size(doc, child, boxes)
            return w if horizontal else h
        if sizing == "fill":
            return 0.0  # decided after the fixed children
        return float(child.get("width" if horizontal else "height") or 0.0)

    def cross_size(child: dict[str, Any], cross_avail: float | None) -> float:
        cl = child.get("child_layout") or {}
        sizing = cl.get("sizing_v" if horizontal else "sizing_h")
        if sizing == "fill" and cross_avail is not None:
            return _clamp(cross_avail, cl.get("min_w" if horizontal else "min_h"),
                          cl.get("max_w" if horizontal else "max_h"))
        if sizing == "hug":
            w, h = _hug_size(doc, child, boxes)
            return h if horizontal else w
        return float(child.get("height" if horizontal else "width") or 0.0)

    # ``avail_w``/``avail_h`` are CONTENT-BOX sizes (padding already
    # excluded — _place_node subtracts it before calling in).
    lines: list[list[dict[str, Any]]] = [[]]
    if wrap and avail_w is not None:
        budget = avail_w if horizontal else avail_h
        budget = budget if budget is not None else float("inf")
        run = 0.0
        for child in flow:
            size = main_size(child)
            if lines[-1] and run + gap + size > budget:
                lines.append([])
                run = 0.0
            if lines[-1]:
                run += gap
            lines[-1].append(child)
            run += size
    else:
        lines[0] = list(flow)

    content_main = avail_w if horizontal else avail_h
    total_cross = 0.0
    cross_offset = 0.0
    line_mains: list[float] = []
    line_crosses: list[float] = []
    for line in lines:
        if not line:
            continue
        fixed_main = sum(main_size(c) for c in line
                         if (c.get("child_layout") or {}).get(
                             "sizing_h" if horizontal else "sizing_v") != "fill")
        fills = [c for c in line
                 if (c.get("child_layout") or {}).get(
                     "sizing_h" if horizontal else "sizing_v") == "fill"]
        gaps = gap * max(0, len(line) - 1)
        # When the parent is hug-sized the main axis is unconstrained:
        # fill children fall back to their measured size.
        constrained = (avail_w is not None) if horizontal else (avail_h is not None)
        free = max(0.0, content_main - fixed_main - gaps) if constrained else 0.0
        fill_each = free / len(fills) if fills and free else 0.0
        line_main = fixed_main + gaps + (fill_each * len(fills) if fills else 0.0)
        if main_align == "center":
            main_start = (content_main - line_main) / 2.0 if constrained else 0.0
        elif main_align == "max":
            main_start = (content_main - line_main) if constrained else 0.0
        elif main_align == "space-between" and len(line) > 1 and constrained:
            main_start = 0.0
            gap = (content_main - fixed_main - fill_each * len(fills)) / (len(line) - 1)
        else:
            main_start = 0.0
        # Cross sizes for this line (avail is already the content box).
        cross_avail = avail_h if horizontal else avail_w
        sizes = {}
        for child in line:
            cl = child.get("child_layout") or {}
            sizing_main = cl.get("sizing_h" if horizontal else "sizing_v")
            if sizing_main == "fill" and fill_each:
                m = _clamp(fill_each,
                           cl.get("min_w" if horizontal else "min_h"),
                           cl.get("max_w" if horizontal else "max_h"))
            else:
                m = main_size(child)
                if sizing_main == "fill":
                    w, h = _hug_size(doc, child, boxes)
                    m = w if horizontal else h
            sizes[child["id"]] = (m, cross_size(child, cross_avail))
        line_cross = max((s[1] for s in sizes.values()), default=0.0)
        line_mains.append(sum(s[0] for s in sizes.values()) + gaps)
        line_crosses.append(line_cross)
        main_pos = main_start
        for child in line:
            m, c = sizes[child["id"]]
            if cross_align == "center":
                cross_pos = cross_offset + (line_cross - c) / 2.0
            elif cross_align == "max":
                cross_pos = cross_offset + line_cross - c
            else:
                cross_pos = cross_offset
            if horizontal:
                x = origin_x + pad_l + main_pos
                y = origin_y + pad_t + cross_pos
                _place_node(doc, child, boxes, x, y, m, c)
            else:
                x = origin_x + pad_l + cross_pos
                y = origin_y + pad_t + main_pos
                _place_node(doc, child, boxes, x, y, c, m)
            main_pos += m + gap
        total_cross += line_cross
        cross_offset += line_cross + cross_gap
    if line_crosses:
        total_cross -= cross_gap  # no trailing cross gap
    if horizontal:
        return max(line_mains, default=0.0), total_cross
    return total_cross, max(line_mains, default=0.0)


def _place_node(doc: DesignDocument, node: dict[str, Any],
                boxes: dict[int, dict[str, float]],
                x: float, y: float,
                width: float | None, height: float | None) -> tuple[float, float]:
    """Place ``node`` at (x, y) with the given size (None = own size).

    Returns the node's (w, h). Frames with auto layout lay their
    children out inside; other nodes keep their own size.
    """
    ntype = node.get("type")
    w = float(node.get("width") or 0.0) if width is None else float(width)
    h = float(node.get("height") or 0.0) if height is None else float(height)
    boxes[int(node["id"])] = {"x": float(x), "y": float(y), "w": w, "h": h}
    if ntype in ("frame", "group", "component", "componentSet", "instance"):
        layout = node.get("layout") or {}
        if layout:
            pad = layout.get("padding") or [0.0, 0.0, 0.0, 0.0]
            avail_w = w - float(pad[1]) - float(pad[3])
            avail_h = h - float(pad[0]) - float(pad[2])
            _layout_children(doc, node, boxes, x, y, avail_w, avail_h)
        else:
            for child in doc.children(int(node["id"])):
                transform = child.get("transform") or [1, 0, 0, 1, 0, 0]
                _place_node(doc, child, boxes,
                            x + float(transform[4]), y + float(transform[5]),
                            None, None)
    return w, h


def compute_layout(doc: DesignDocument, root_id: int
                   ) -> dict[int, dict[str, float]]:
    """Compute absolute boxes for ``root_id`` and its descendants.

    Returns ``{node_id: {"x", "y", "w", "h"}}``. A hug-sized root
    grows to its content; a fixed root keeps its own size. The
    root sits at its transform position.
    """
    root = doc.node(root_id)
    # Boxes are relative to the root: the root sits at (0, 0) in its
    # own coordinate space (its page transform is not added in).
    origin_x = 0.0
    origin_y = 0.0
    boxes: dict[int, dict[str, float]] = {}
    layout = root.get("layout") or {}
    cl = root.get("child_layout") or {}
    sizing_h = cl.get("sizing_h", "fixed")
    sizing_v = cl.get("sizing_v", "fixed")
    if layout and (sizing_h == "hug" or sizing_v == "hug"):
        content_w, content_h = _layout_children(
            doc, root, boxes, origin_x, origin_y, None, None)
        pad = layout.get("padding") or [0.0, 0.0, 0.0, 0.0]
        w = (float(root.get("width") or 0.0) if sizing_h != "hug"
             else content_w + float(pad[1]) + float(pad[3]))
        h = (float(root.get("height") or 0.0) if sizing_v != "hug"
             else content_h + float(pad[0]) + float(pad[2]))
        boxes[int(root["id"])] = {"x": origin_x, "y": origin_y, "w": w, "h": h}
        # Re-run with the resolved size so fill children settle.
        boxes = {}
        _place_node(doc, root, boxes, origin_x, origin_y, w, h)
        return boxes
    _place_node(doc, root, boxes, origin_x, origin_y, None, None)
    return boxes
