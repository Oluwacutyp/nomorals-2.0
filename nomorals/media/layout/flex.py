"""The flex layout engine.

A faithful port of upstream ``crates/doc/src/flex.rs`` — the
``arrange`` / ``layout`` / ``layout_all`` algorithm DesignCraft
uses to lay out a group's children: row/column (with reverse),
wrap, justify (flex-start, center, flex-end, space-between,
space-around, space-evenly), cross-axis align, auto sizing,
padding and gaps. Rects are ``(x0, y0, x1, y1)`` tuples in the
container's inner space; ``arrange`` returns the container's
result rect and one (dx, dy) move per child, in the same space.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..vector.geom import Affine
from .model import FlexLayout, Item

__all__ = ["Arrangement", "arrange", "contains_flex", "layout_all",
           "layout_item"]

_EPS = 1e-6


@dataclass
class Arrangement:
    """The result of ``arrange``: the container's rect and one
    (dx, dy) move per child, in the container's inner space."""

    rect: tuple[float, float, float, float]
    moves: list[tuple[float, float]]
    main_gap: float


def _finite(value: float) -> float:
    return value if value == value and abs(value) != float("inf") else 0.0


def arrange(flex: FlexLayout, rect: tuple[float, float, float, float],
            bounds: list[tuple[float, float, float, float]]
            ) -> Arrangement:
    """Lay out objects of the given bounds in ``rect`` (both in the
    container's inner space). Direct port of upstream ``arrange``."""
    row = flex.direction in ("row", "row-reverse")
    reverse = flex.reverse or flex.direction in ("row-reverse",
                                                 "column-reverse")
    gap_main = _finite(flex.gap)
    gap_cross = _finite(flex.row_gap if flex.row_gap is not None
                        else flex.gap)
    pad_top, pad_left, pad_bottom, pad_right = (
        _finite(v) for v in flex.padding)
    size = (lambda b: (b[2] - b[0], b[3] - b[1])) if row else (
        lambda b: (b[3] - b[1], b[2] - b[0]))
    main_auto = (flex.width == "auto") if row else (flex.height == "auto")
    cross_auto = (flex.height == "auto") if row else (flex.width == "auto")
    if row:
        pad_main_start, pad_main_end = pad_left, pad_right
        pad_cross_start, pad_cross_end = pad_top, pad_bottom
    else:
        pad_main_start, pad_main_end = pad_top, pad_bottom
        pad_cross_start, pad_cross_end = pad_left, pad_right
    rect_main = (rect[2] - rect[0]) if row else (rect[3] - rect[1])
    rect_cross = (rect[3] - rect[1]) if row else (rect[2] - rect[0])
    main_len = max(0.0, rect_main - pad_main_start - pad_main_end)
    # Rows: the objects in each, broken where the next one would
    # not fit (Wrap on a fixed length).
    lines: list[list[int]] = []
    used = 0.0
    for i, b in enumerate(bounds):
        m, _ = size(b)
        breaks = (flex.wrap and not main_auto and lines
                  and lines[-1]
                  and used + gap_main + m > main_len + _EPS)
        if lines and not breaks:
            last = lines[-1]
            used += m if not last else gap_main + m
            last.append(i)
        else:
            lines.append([i])
            used = m

    def line_main(line: list[int]) -> float:
        return (sum(size(bounds[i])[0] for i in line)
                + gap_main * max(0, len(line) - 1))

    def line_cross(line: list[int]) -> float:
        return max((size(bounds[i])[1] for i in line), default=0.0)

    if main_auto:
        main_len = max((line_main(line) for line in lines), default=0.0)
    content_cross = (sum(line_cross(line) for line in lines)
                     + gap_cross * max(0, len(lines) - 1))
    cross_len = (content_cross if cross_auto else
                 max(0.0, rect_cross - pad_cross_start - pad_cross_end))
    # The container: Auto sides fit the objects, from its top left.
    if row:
        w = main_len + pad_main_start + pad_main_end
        h = cross_len + pad_cross_start + pad_cross_end
    else:
        w = cross_len + pad_cross_start + pad_cross_end
        h = main_len + pad_main_start + pad_main_end
    out_rect = (rect[0], rect[1],
                rect[0] + (w if (row and main_auto)
                           or (not row and cross_auto) else rect[2] - rect[0]),
                rect[1] + (h if (not row and main_auto)
                           or (row and cross_auto) else rect[3] - rect[1]))

    def shift(free: float) -> float:
        if flex.align == "center":
            return free / 2.0
        if flex.align == "end":
            return free
        return 0.0

    moves: list[tuple[float, float]] = [(0.0, 0.0)] * len(bounds)
    cross_at = shift(cross_len - content_cross)
    main_gap = gap_main
    for li, line in enumerate(lines):
        lc = line_cross(line)
        free = main_len - line_main(line)
        n = float(len(line))
        if free <= 0.0 or flex.justify == "flex-start":
            start, gap = 0.0, gap_main
        elif flex.justify == "center":
            start, gap = free / 2.0, gap_main
        elif flex.justify == "flex-end":
            start, gap = free, gap_main
        elif flex.justify == "space-between" and len(line) > 1:
            start, gap = 0.0, gap_main + free / (n - 1.0)
        elif flex.justify == "space-between":
            start, gap = 0.0, gap_main
        elif flex.justify == "space-around":
            start, gap = free / (2.0 * n), gap_main + free / n
        elif flex.justify == "space-evenly":
            start, gap = free / (n + 1.0), gap_main + free / (n + 1.0)
        else:
            start, gap = 0.0, gap_main
        if li == 0:
            main_gap = gap
        at = start
        for i in line:
            m, c = size(bounds[i])
            main = main_len - at - m if reverse else at
            cross = cross_at + shift(lc - c)
            if row:
                x = out_rect[0] + pad_main_start + main
                y = out_rect[1] + pad_cross_start + cross
            else:
                x = out_rect[0] + pad_cross_start + cross
                y = out_rect[1] + pad_main_start + main
            moves[i] = (x - bounds[i][0], y - bounds[i][1])
            at += m + gap
        cross_at += lc + gap_cross
    return Arrangement(rect=out_rect, moves=moves, main_gap=main_gap)


def layout_item(item: Item) -> bool:
    """Lay out a container's children (and size it, for Auto
    sides). Returns whether anything moved — direct port of
    upstream ``layout``."""
    if item.flex is None:
        return False
    if item.content_type() != "group":
        return False
    flex = item.flex
    kids = item.content_items()
    rect = item.local_bounds()
    bounds = [k.bounds() for k in kids]
    result = arrange(flex, rect, bounds)
    changed = False
    if (abs(result.rect[2] - result.rect[0] - (rect[2] - rect[0])) > _EPS
            or abs(result.rect[3] - result.rect[1]
                   - (rect[3] - rect[1])) > _EPS):
        # Auto sides resize the container's own path.
        w = result.rect[2] - result.rect[0]
        h = result.rect[3] - result.rect[1]
        item.path_d = (f"M0,0 L{w:g},0 L{w:g},{h:g} L0,{h:g} Z")
        item._path_cache = None
        changed = True
    for kid, (dx, dy) in zip(kids, result.moves, strict=True):
        if abs(dx) > _EPS or abs(dy) > _EPS:
            kid.xf = Affine.translate(dx, dy) * kid.xf
            changed = True
    if changed:
        item.set_content_items(kids)
    return changed


def contains_flex(item: Item) -> bool:
    """Is ``item`` a container, or does it hold one? (upstream
    ``contains_flex``)."""
    if item.flex is not None:
        return True
    return any(contains_flex(c) for c in item.content_items())


def layout_all(items: list[Item]) -> None:
    """Lay out every container among ``items`` and the objects
    nested in them, inner containers first (an Auto container's
    size is its parent's object size). Only changed items are
    copied — direct port of upstream ``layout_all``."""
    for item in items:
        if not contains_flex(item):
            continue
        if item.content_type() == "group":
            layout_all(item.content_items())
        layout_item(item)
