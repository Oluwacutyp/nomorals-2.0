"""The layout editor — fidelity-preserving edits.

Every edit mutates the document in place and marks the touched
item dirty (``LayoutDocument.mark_dirty``), so the fidelity report
can tell a touched item from an untouched one — even a restyle
that sets the value the item already had counts as touched, never
as an untouched change. Untouched items keep their canonical
bytes, which is what ``fidelity_report`` measures.
"""

from __future__ import annotations

from ..vector.geom import Affine
from .errors import LayoutError
from .model import CharRun, Fill, Guide, Item, LayoutDocument, Page, ParaFormat, Story, Stroke

__all__ = ["LayoutEditor"]


class LayoutEditor:
    """Edit a layout document in place."""

    def __init__(self, doc: LayoutDocument) -> None:
        self.doc = doc

    # -- ids ---------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        taken = {i.id for i in self.doc.all_items()}
        n = 1
        while f"{prefix}{n}" in taken:
            n += 1
        return f"{prefix}{n}"

    def _next_story_id(self) -> str:
        n = 1
        while f"S{n}" in self.doc.stories:
            n += 1
        return f"S{n}"

    def _require_item(self, item_id: str) -> Item:
        item = self.doc.find_item(item_id)
        if item is None:
            raise LayoutError(f"no item {item_id!r}", kind="not_found")
        return item

    def _require_page(self, page_id: str) -> Page:
        page = self.doc.find_page(page_id)
        if page is None:
            raise LayoutError(f"no page {page_id!r}", kind="not_found")
        return page

    def _container(self, item_id: str) -> list[Item] | None:
        """The list holding ``item_id`` (a page's items or a group's
        children), or ``None`` when it is not on any page."""
        for spread in self.doc.spreads:
            for page in spread.pages:
                found = self._container_in(page.items, item_id)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _container_in(items: list[Item], item_id: str
                      ) -> list[Item] | None:
        for item in items:
            if item.id == item_id:
                return items
            if item.content_type() == "group":
                found = LayoutEditor._container_in(
                    item.content_items(), item_id)
                if found is not None:
                    return found
        return None

    def _page_of(self, item_id: str) -> Page | None:
        for spread in self.doc.spreads:
            for page in spread.pages:
                if self._container_in(page.items, item_id) is not None:
                    return page
        return None

    # -- adding ------------------------------------------------------
    def add_rect(self, page_id: str, x: float, y: float, width: float,
                 height: float, fill: Fill | None = None,
                 name: str = "") -> Item:
        page = self._require_page(page_id)
        item = Item(id=self._next_id("I"), name=name or "rectangle",
                    shape="rectangle",
                    path_d=(f"M0,0 L{width:g},0 L{width:g},{height:g} "
                            f"L0,{height:g} Z"),
                    xf=Affine.translate(x, y),
                    fill=fill or Fill())
        page.items.append(item)
        return item

    def add_oval(self, page_id: str, x: float, y: float, width: float,
                 height: float, fill: Fill | None = None,
                 name: str = "") -> Item:
        item = self.add_rect(page_id, x, y, width, height, fill=fill,
                             name=name or "oval")
        item.shape = "oval"
        return item

    def add_text(self, page_id: str, x: float, y: float, width: float,
                 height: float, text: str, font_size: float = 12.0,
                 name: str = "") -> Item:
        page = self._require_page(page_id)
        story_id = self._next_story_id()
        self.doc.stories[story_id] = Story(
            id=story_id,
            paras=[ParaFormat(runs=[CharRun(text=text,
                                            font_size=font_size)])])
        item = Item(id=self._next_id("I"), name=name or "text",
                    shape="rectangle",
                    path_d=(f"M0,0 L{width:g},0 L{width:g},{height:g} "
                            f"L0,{height:g} Z"),
                    xf=Affine.translate(x, y),
                    fill=Fill(swatch="None"),
                    content={"type": "text", "story_id": story_id,
                             "options": {"columns": 1, "gutter": 12.0,
                                         "vertical_justification": "top",
                                         "first_baseline": "ascent",
                                         "auto_size": "off",
                                         "next_frame_id": None}})
        page.items.append(item)
        return item

    # -- geometry ----------------------------------------------------
    def move(self, item_id: str, x: float, y: float) -> Item:
        """Move an item so its bounds' top-left sits at (x, y)."""
        item = self._require_item(item_id)
        x0, y0, _, _ = self.doc.item_bounds(item)
        item.xf = Affine.translate(x - x0, y - y0) * item.xf
        self.doc.mark_dirty(item_id)
        return item

    def resize(self, item_id: str, width: float, height: float) -> Item:
        """Resize an item to (width, height), keeping its top-left
        where it is."""
        item = self._require_item(item_id)
        x0, y0, x1, y1 = self.doc.item_bounds(item)
        old_w, old_h = x1 - x0, y1 - y0
        if old_w <= 0 or old_h <= 0:
            raise LayoutError(f"item {item_id!r} has no size to resize",
                              kind="invalid_argument")
        sx, sy = width / old_w, height / old_h
        item.xf = (Affine.translate(x0, y0)
                   * Affine.scale(sx, sy)
                   * Affine.translate(-x0, -y0)
                   * item.xf)
        self.doc.mark_dirty(item_id)
        return item

    def restyle(self, item_id: str, fill: Fill | None = None,
                stroke: Stroke | None = None,
                opacity: float | None = None) -> Item:
        item = self._require_item(item_id)
        if fill is not None:
            item.fill = fill
        if stroke is not None:
            item.stroke = stroke
        if opacity is not None:
            item.opacity = float(opacity)
        self.doc.mark_dirty(item_id)
        return item

    # -- structure ---------------------------------------------------
    def group(self, item_ids: list[str], name: str = "group") -> Item:
        if len(item_ids) < 2:
            raise LayoutError("group needs at least two items",
                              kind="invalid_argument")
        items = [self._require_item(i) for i in item_ids]
        page = self._page_of(item_ids[0])
        if page is None:
            raise LayoutError(f"item {item_ids[0]!r} is not on a page",
                              kind="not_found")
        x0s, y0s, x1s, y1s = [], [], [], []
        for item in items:
            x0, y0, x1, y1 = self.doc.item_bounds(item)
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
        min_x, min_y = min(x0s), min(y0s)
        width, height = max(x1s) - min_x, max(y1s) - min_y
        for item in items:
            item.xf = Affine.translate(-min_x, -min_y) * item.xf
            self.doc.mark_dirty(item.id)
        container = self._container(item_ids[0])
        assert container is not None
        for item in items:
            own = self._container(item.id)
            if own is not None:
                own.remove(item)
        group = Item(id=self._next_id("G"), name=name, shape="group",
                     path_d=(f"M0,0 L{width:g},0 L{width:g},{height:g} "
                             f"L0,{height:g} Z"),
                     xf=Affine.translate(min_x, min_y),
                     fill=Fill(swatch="None"),
                     content={"type": "group"})
        group.set_content_items(items)
        page.items.append(group)
        return group

    def ungroup(self, group_id: str) -> list[Item]:
        group = self._require_item(group_id)
        if group.content_type() != "group":
            raise LayoutError(f"item {group_id!r} is not a group",
                              kind="invalid_argument")
        kids = list(group.content_items())
        page = self._page_of(group_id)
        container = self._container(group_id)
        assert container is not None
        container.remove(group)
        for kid in kids:
            kid.xf = group.xf * kid.xf
            self.doc.mark_dirty(kid.id)
        if page is not None:
            page.items.extend(kids)
        return kids

    def remove(self, item_id: str) -> None:
        container = self._container(item_id)
        if container is None:
            raise LayoutError(f"no item {item_id!r}", kind="not_found")
        for item in container:
            if item.id == item_id:
                container.remove(item)
                return

    def bring_to_front(self, item_id: str) -> Item:
        item = self._require_item(item_id)
        container = self._container(item_id)
        assert container is not None
        container.remove(item)
        container.append(item)
        return item

    def send_to_back(self, item_id: str) -> Item:
        item = self._require_item(item_id)
        container = self._container(item_id)
        assert container is not None
        container.remove(item)
        container.insert(0, item)
        return item

    # -- align / distribute / snap -----------------------------------
    def align(self, item_ids: list[str], edge: str) -> list[Item]:
        """Align items to an edge: left | right | center | top |
        bottom | middle (centre on the selection's bounding box)."""
        items = [self._require_item(i) for i in item_ids]
        if not items:
            return []
        bounds = [self.doc.item_bounds(i) for i in items]
        if edge == "left":
            target = min(b[0] for b in bounds)
            deltas = [(target - b[0], 0.0) for b in bounds]
        elif edge == "right":
            target = max(b[2] for b in bounds)
            deltas = [(target - b[2], 0.0) for b in bounds]
        elif edge == "center":
            target = (min(b[0] for b in bounds)
                      + max(b[2] for b in bounds)) / 2.0
            deltas = [(target - (b[0] + b[2]) / 2.0, 0.0) for b in bounds]
        elif edge == "top":
            target = min(b[1] for b in bounds)
            deltas = [(0.0, target - b[1]) for b in bounds]
        elif edge == "bottom":
            target = max(b[3] for b in bounds)
            deltas = [(0.0, target - b[3]) for b in bounds]
        elif edge == "middle":
            target = (min(b[1] for b in bounds)
                      + max(b[3] for b in bounds)) / 2.0
            deltas = [(0.0, target - (b[1] + b[3]) / 2.0) for b in bounds]
        else:
            raise LayoutError(f"unknown align edge {edge!r}",
                              kind="invalid_argument")
        for item, (dx, dy) in zip(items, deltas, strict=True):
            if dx or dy:
                item.xf = Affine.translate(dx, dy) * item.xf
                self.doc.mark_dirty(item.id)
        return items

    def distribute(self, item_ids: list[str], axis: str) -> list[Item]:
        """Space items evenly along an axis (horizontal = x, vertical
        = y), keeping the first and last edges where they are."""
        items = [self._require_item(i) for i in item_ids]
        if len(items) < 3:
            return items
        bounds = {i.id: self.doc.item_bounds(i) for i in items}
        if axis == "horizontal":
            ordered = sorted(items, key=lambda i: bounds[i.id][0])
            first = bounds[ordered[0].id][0]
            last = bounds[ordered[-1].id][0]
            step = (last - first) / (len(ordered) - 1)
            for index, item in enumerate(ordered):
                new = first + index * step
                dx = new - bounds[item.id][0]
                if dx:
                    item.xf = Affine.translate(dx, 0.0) * item.xf
                    self.doc.mark_dirty(item.id)
        elif axis == "vertical":
            ordered = sorted(items, key=lambda i: bounds[i.id][1])
            first = bounds[ordered[0].id][1]
            last = bounds[ordered[-1].id][1]
            step = (last - first) / (len(ordered) - 1)
            for index, item in enumerate(ordered):
                new = first + index * step
                dy = new - bounds[item.id][1]
                if dy:
                    item.xf = Affine.translate(0.0, dy) * item.xf
                    self.doc.mark_dirty(item.id)
        else:
            raise LayoutError(f"unknown distribute axis {axis!r}",
                              kind="invalid_argument")
        return items

    def add_guide(self, page_id: str, axis: str,
                  position: float) -> Guide:
        page = self._require_page(page_id)
        if axis not in ("vertical", "horizontal"):
            raise LayoutError(f"unknown guide axis {axis!r}",
                              kind="invalid_argument")
        guide = Guide(axis=axis, position=float(position))
        page.guides.append(guide)
        return guide

    def snap(self, item_id: str, tolerance: float = 5.0) -> bool:
        """Snap an item's nearest edge to the nearest guide, page
        edge or sibling edge within ``tolerance`` points. Returns
        whether anything snapped."""
        item = self._require_item(item_id)
        page = self._page_of(item_id)
        if page is None:
            return False
        x0, y0, x1, y1 = self.doc.item_bounds(item)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        vertical = [g.position for g in page.guides
                    if g.axis == "vertical"]
        vertical += [0.0, page.width, page.margins.inside,
                     page.width - page.margins.outside]
        horizontal = [g.position for g in page.guides
                      if g.axis == "horizontal"]
        horizontal += [0.0, page.height, page.margins.top,
                       page.height - page.margins.bottom]
        for other in page.items:
            if other.id == item_id:
                continue
            ox0, oy0, ox1, oy1 = self.doc.item_bounds(other)
            vertical += [ox0, ox1, (ox0 + ox1) / 2.0]
            horizontal += [oy0, oy1, (oy0 + oy1) / 2.0]
        best_dx: float | None = None
        for target in vertical:
            for edge in (x0, x1, cx):
                delta = target - edge
                if abs(delta) <= tolerance and (
                        best_dx is None or abs(delta) < abs(best_dx)):
                    best_dx = delta
        best_dy: float | None = None
        for target in horizontal:
            for edge in (y0, y1, cy):
                delta = target - edge
                if abs(delta) <= tolerance and (
                        best_dy is None or abs(delta) < abs(best_dy)):
                    best_dy = delta
        if best_dx is None and best_dy is None:
            return False
        item.xf = Affine.translate(best_dx or 0.0, best_dy or 0.0) * item.xf
        self.doc.mark_dirty(item_id)
        return True

    # -- liquid layout / page resize ---------------------------------
    def set_liquid(self, item_id: str, pin_left: bool = False,
                   pin_right: bool = False, pin_top: bool = False,
                   pin_bottom: bool = False,
                   stretch_width: bool = False,
                   stretch_height: bool = False) -> Item:
        """Pin an item's edges to the page (liquid layout): a pinned
        edge keeps its distance from that page edge when the page
        is resized."""
        item = self._require_item(item_id)
        liquid = dict(item.liquid or {})
        liquid.update({"pin_left": pin_left, "pin_right": pin_right,
                       "pin_top": pin_top, "pin_bottom": pin_bottom,
                       "stretch_width": stretch_width,
                       "stretch_height": stretch_height})
        item.liquid = liquid
        self.doc.mark_dirty(item_id)
        return item

    def resize_page(self, page_id: str, width: float,
                    height: float) -> Page:
        page = self._require_page(page_id)
        old_w, old_h = page.width, page.height
        page.width, page.height = float(width), float(height)
        for item in page.items:
            if not item.liquid:
                continue
            x0, y0, x1, y1 = self.doc.item_bounds(item)
            item_w, item_h = x1 - x0, y1 - y0
            liquid = item.liquid
            new_x0, new_y0 = x0, y0
            new_w, new_h = item_w, item_h
            if liquid.get("pin_right") and liquid.get("pin_left"):
                new_w = item_w + (width - old_w)
            elif liquid.get("pin_right"):
                new_x0 = width - (old_w - x1) - item_w
            if liquid.get("pin_bottom") and liquid.get("pin_top"):
                new_h = item_h + (height - old_h)
            elif liquid.get("pin_bottom"):
                new_y0 = height - (old_h - y1) - item_h
            if liquid.get("stretch_width"):
                new_w = item_w * (width / old_w) if old_w else item_w
            if liquid.get("stretch_height"):
                new_h = item_h * (height / old_h) if old_h else item_h
            if new_w <= 0 or new_h <= 0:
                continue
            sx = new_w / item_w if item_w else 1.0
            sy = new_h / item_h if item_h else 1.0
            item.xf = (Affine.translate(new_x0, new_y0)
                       * Affine.scale(sx, sy)
                       * Affine.translate(-x0, -y0)
                       * item.xf)
            self.doc.mark_dirty(item.id)
        return page

    # -- flex ----------------------------------------------------------
    def apply_flex(self, group_id: str) -> bool:
        """Run the flex layout on a group (see ``flex.layout_item``)."""
        from .flex import layout_item
        item = self._require_item(group_id)
        changed = layout_item(item)
        if changed:
            self.doc.mark_dirty(group_id)
        return changed
