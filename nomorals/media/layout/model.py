"""Layout document model — pages, spreads, frames, stories, styles.

Port of the *design* of upstream ``crates/doc`` (DesignCraft, a
desktop-publishing page-layout studio in pure Rust, dual
MIT/Apache-2.0) into Devon's Python, stdlib + the vector organ's
geometry. Upstream shapes mirrored here:

* ``Document`` — title, settings, spreads, parent spreads,
  layers, stories, styles, swatches (``crates/doc/src/lib.rs``).
* ``Spread`` / ``Page`` — facing/non-facing pages, margins,
  columns, bleed, sections (``crates/doc/src/page.rs``).
* ``Item`` — id, layer, shape, path in inner space, ``xf`` to
  spread space, fill through a swatch + tint, stroke, opacity,
  blend, effects, wrap, content (``crates/doc/src/item.rs``).
* ``Story`` — paragraphs of character runs
  (``crates/doc/src/story.rs``).
* ``Columns`` — even splits, custom positions, mirroring for
  facing pages, gutter moves (``crates/doc/src/page.rs``).

Fidelity discipline (same habit as the vector organ): an entity
remembers which keys its source dict carried (``_present``) and
``to_dict`` emits a key when it was present in the source or its
value differs from the default — so a parse → serialize round
trip reproduces the source bytes exactly, and an edit changes
only the keys it touches. Unknown keys are kept in ``extra`` and
re-emitted, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..vector.geom import Affine, Path, parse_path_data
from .errors import LayoutError

__all__ = [
    "CharRun", "Columns", "DocumentGrid", "Fill", "FlexAlign",
    "FlexDirection", "FlexJustify", "FlexLayout", "FlexSize", "Guide",
    "Item", "Layer", "LayoutDocument", "Margins", "Page", "ParaFormat",
    "Section", "Spread", "Story", "Stroke", "Swatch",
    "TextFrameOptions",
]


class FlexDirection(StrEnum):
    """Flex main-axis direction (upstream ``flex.rs``)."""

    Row = "row"
    Column = "column"
    RowReverse = "row-reverse"
    ColumnReverse = "column-reverse"


class FlexJustify(StrEnum):
    """How free space is shared along the main axis."""

    FlexStart = "flex-start"
    Center = "center"
    FlexEnd = "flex-end"
    SpaceBetween = "space-between"
    SpaceAround = "space-around"
    SpaceEvenly = "space-evenly"


class FlexAlign(StrEnum):
    """Cross-axis alignment of a line's items."""

    Start = "start"
    Center = "center"
    End = "end"


class FlexSize(StrEnum):
    """Whether a flex container's side is fixed or fits content."""

    Fixed = "fixed"
    Auto = "auto"


def _enum_value(value):
    """The JSON value of an enum-or-string flex field."""
    return value.value if isinstance(value, StrEnum) else value

# ── small value types ─────────────────────────────────────────────

@dataclass
class Margins:
    """Page margins in points; inside/outside are left/right on a
    non-facing page (upstream ``page.rs``)."""

    top: float = 36.0
    bottom: float = 36.0
    inside: float = 36.0
    outside: float = 36.0

    def to_dict(self) -> dict:
        return {"top": self.top, "bottom": self.bottom,
                "inside": self.inside, "outside": self.outside}

    @classmethod
    def from_dict(cls, raw: dict) -> Margins:
        return cls(top=float(raw.get("top", 36.0)),
                   bottom=float(raw.get("bottom", 36.0)),
                   inside=float(raw.get("inside", 36.0)),
                   outside=float(raw.get("outside", 36.0)))


@dataclass
class Columns:
    """Column layout for a page (upstream ``page.rs``).

    ``count`` columns split the margin box evenly with ``gutter``
    between them, unless ``positions`` carries a valid custom set
    (2 × count offsets from the margin box's left edge, first 0,
    last = margin width, ascending).
    """

    count: int = 1
    gutter: float = 12.0
    positions: list[float] | None = None

    MAX_COLUMNS = 216
    POSITION_TOLERANCE = 0.01

    def to_dict(self) -> dict:
        return {"count": self.count, "gutter": self.gutter,
                "positions": self.positions}

    @classmethod
    def from_dict(cls, raw: dict) -> Columns:
        positions = raw.get("positions")
        return cls(count=int(raw.get("count", 1)),
                   gutter=float(raw.get("gutter", 12.0)),
                   positions=([float(v) for v in positions]
                              if positions is not None else None))

    def valid_positions(self, margin_width: float) -> list[float] | None:
        """The custom positions when they are usable for a margin
        box ``margin_width`` wide, else ``None`` (upstream
        ``Columns::valid_positions``)."""
        positions = self.positions
        if positions is None:
            return None
        if not (1 <= self.count <= self.MAX_COLUMNS):
            return None
        if len(positions) != 2 * self.count:
            return None
        if not all(v == v and abs(v) != float("inf") for v in positions):
            return None
        if any(positions[i + 1] < positions[i]
               for i in range(len(positions) - 1)):
            return None
        if abs(positions[0]) > self.POSITION_TOLERANCE:
            return None
        if abs(positions[-1] - margin_width) > self.POSITION_TOLERANCE:
            return None
        return positions

    def edges(self, margin_width: float) -> list[float]:
        """Every column's left and right edge as offsets from the
        margin box's left edge (upstream ``Columns::edges``)."""
        width = max(0.0, margin_width) if margin_width == margin_width else 0.0
        valid = self.valid_positions(width)
        if valid is not None:
            return [min(max(v, 0.0), width) for v in valid]
        count = max(1, self.count)
        total_gutter = self.gutter * (count - 1)
        col_w = max(0.0, (width - total_gutter) / count)
        out: list[float] = []
        at = 0.0
        for _ in range(count):
            out.append(at)
            at += col_w
            out.append(at)
            at += self.gutter
        return out

    def with_gutter_moved(self, margin_width: float, gutter: int,
                          new_left: float) -> list[float] | None:
        """The custom positions with gutter ``gutter`` moved so its
        left edge sits at ``new_left`` (upstream
        ``Columns::with_gutter_moved``); ``None`` when the move is
        not valid."""
        if new_left != new_left or abs(new_left) == float("inf"):
            return None
        pos = self.edges(margin_width)
        i = gutter * 2 + 1
        window = pos[i - 1:i + 3]
        if len(window) != 4:
            return None
        lo, left, right, hi = window
        width = right - left
        at = min(max(new_left, lo), hi - width)
        new = list(pos)
        new[i] = at
        new[i + 1] = min(at + width, hi)
        return new if self._valid_list(new, margin_width) else None

    def _valid_list(self, positions: list[float],
                    margin_width: float) -> bool:
        probe = Columns(count=self.count, gutter=self.gutter,
                        positions=positions)
        return probe.valid_positions(margin_width) is not None

    def mirrored(self, margin_width: float) -> Columns:
        """These columns reflected across a margin box ``margin_width``
        wide, as on the facing page across the spine (upstream
        ``Columns::mirrored``)."""
        valid = self.valid_positions(margin_width)
        positions = None
        if valid is not None:
            positions = [margin_width - v for v in reversed(valid)]
        return Columns(count=self.count, gutter=self.gutter,
                       positions=positions)


@dataclass
class Guide:
    """A ruler guide on a page (upstream: guides dragged from the
    rulers, snapped to margins, columns, bleed, page and object
    edges)."""

    axis: str = "vertical"  # "vertical" (x position) | "horizontal"
    position: float = 0.0

    def to_dict(self) -> dict:
        return {"axis": self.axis, "position": self.position}

    @classmethod
    def from_dict(cls, raw: dict) -> Guide:
        return cls(axis=str(raw.get("axis", "vertical")),
                   position=float(raw.get("position", 0.0)))


@dataclass
class DocumentGrid:
    """The document grid (upstream ``DocumentGrid`` in lib.rs)."""

    horizontal: float = 72.0
    vertical: float = 72.0
    subdivisions: int = 8
    color: list[int] = field(default_factory=lambda: [200, 200, 200])
    in_back: bool = True

    def to_dict(self) -> dict:
        return {"horizontal": self.horizontal, "vertical": self.vertical,
                "subdivisions": self.subdivisions, "color": self.color,
                "in_back": self.in_back}

    @classmethod
    def from_dict(cls, raw: dict) -> DocumentGrid:
        return cls(horizontal=float(raw.get("horizontal", 72.0)),
                   vertical=float(raw.get("vertical", 72.0)),
                   subdivisions=int(raw.get("subdivisions", 8)),
                   color=list(raw.get("color", [200, 200, 200])),
                   in_back=bool(raw.get("in_back", True)))


@dataclass
class Section:
    """A section (page numbering run) in the document."""

    name: str = ""
    start_page: int = 1
    numbering: str = "arabic"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"name": self.name, "start_page": self.start_page,
               "numbering": self.numbering}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Section:
        known = {"name", "start_page", "numbering"}
        return cls(name=str(raw.get("name", "")),
                   start_page=int(raw.get("start_page", 1)),
                   numbering=str(raw.get("numbering", "arabic")),
                   extra={k: v for k, v in raw.items() if k not in known})


@dataclass
class Layer:
    """A document layer (upstream: top of the list = frontmost)."""

    id: str = "L1"
    name: str = "Layer 1"
    visible: bool = True
    locked: bool = False
    color: str = "#000000"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"id": self.id, "name": self.name,
               "visible": self.visible, "locked": self.locked,
               "color": self.color}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Layer:
        known = {"id", "name", "visible", "locked", "color"}
        return cls(id=str(raw.get("id", "L1")),
                   name=str(raw.get("name", "Layer 1")),
                   visible=bool(raw.get("visible", True)),
                   locked=bool(raw.get("locked", False)),
                   color=str(raw.get("color", "#000000")),
                   extra={k: v for k, v in raw.items() if k not in known})


@dataclass
class Swatch:
    """A colour swatch (upstream ``color`` crate: process, spot,
    tints, gradients). ``rgb`` is [r, g, b] 0–255; a gradient
    swatch carries ``stops`` and ``gradient_type``."""

    name: str = "None"
    kind: str = "none"  # "process" | "spot" | "none" | "gradient"
    rgb: list[int] | None = None
    cmyk: list[float] | None = None
    stops: list[dict] | None = None
    gradient_type: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "kind": self.kind,
                     "rgb": self.rgb}
        if self.cmyk is not None:
            out["cmyk"] = self.cmyk
        if self.stops is not None:
            out["stops"] = self.stops
        if self.gradient_type is not None:
            out["gradient_type"] = self.gradient_type
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Swatch:
        known = {"name", "kind", "rgb", "cmyk", "stops", "gradient_type"}
        return cls(name=str(raw.get("name", "None")),
                   kind=str(raw.get("kind", "none")),
                   rgb=(list(raw["rgb"]) if raw.get("rgb") is not None
                        else None),
                   cmyk=(list(raw["cmyk"]) if raw.get("cmyk") is not None
                         else None),
                   stops=raw.get("stops"),
                   gradient_type=raw.get("gradient_type"),
                   extra={k: v for k, v in raw.items() if k not in known})

    def resolve_rgb(self) -> tuple[int, int, int] | None:
        """The swatch as an RGB triple, or ``None`` for the None
        swatch / a gradient (gradients resolve per stop)."""
        if self.rgb is not None:
            r, g, b = self.rgb[:3]
            return (int(r), int(g), int(b))
        if self.cmyk is not None:
            c, m, y, k = [float(v) for v in self.cmyk[:4]]
            return (round(255 * (1 - c) * (1 - k)),
                    round(255 * (1 - m) * (1 - k)),
                    round(255 * (1 - y) * (1 - k)))
        return None


@dataclass
class Fill:
    """A fill through a swatch + tint (upstream ``item/paint.rs``:
    fills always go through swatches or unnamed colours)."""

    swatch: str = "None"
    tint: float = 1.0
    gradient_angle: float | None = None
    gradient_vector: list[float] | None = None
    overprint: bool = False

    def to_dict(self) -> dict:
        out: dict = {"swatch": self.swatch, "tint": self.tint}
        if self.gradient_angle is not None:
            out["gradient_angle"] = self.gradient_angle
        if self.gradient_vector is not None:
            out["gradient_vector"] = self.gradient_vector
        if self.overprint:
            out["overprint"] = True
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Fill:
        return cls(swatch=str(raw.get("swatch", "None")),
                   tint=float(raw.get("tint", 1.0)),
                   gradient_angle=raw.get("gradient_angle"),
                   gradient_vector=raw.get("gradient_vector"),
                   overprint=bool(raw.get("overprint", False)))

    def is_none(self) -> bool:
        return self.swatch == "None"


@dataclass
class Stroke:
    """A stroke (upstream ``item/paint.rs``)."""

    color: str | None = None
    width: float = 0.0
    align: str = "center"  # center | inside | outside
    cap: str = "butt"  # butt | round | square
    join: str = "miter"  # miter | round | bevel
    dash: list[float] | None = None
    type: str = "solid"

    def to_dict(self) -> dict:
        out: dict = {"color": self.color, "width": self.width}
        if self.align != "center":
            out["align"] = self.align
        if self.cap != "butt":
            out["cap"] = self.cap
        if self.join != "miter":
            out["join"] = self.join
        if self.dash is not None:
            out["dash"] = self.dash
        if self.type != "solid":
            out["type"] = self.type
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Stroke:
        return cls(color=raw.get("color"),
                   width=float(raw.get("width", 0.0)),
                   align=str(raw.get("align", "center")),
                   cap=str(raw.get("cap", "butt")),
                   join=str(raw.get("join", "miter")),
                   dash=raw.get("dash"),
                   type=str(raw.get("type", "solid")))


@dataclass
class CharRun:
    """One run of text with one character format (upstream
    ``story.rs`` ``CharRun``)."""

    text: str = ""
    char_style: str | None = None
    font_size: float | None = None
    bold: bool = False
    italic: bool = False
    color: str | None = None
    _present: frozenset = field(default_factory=frozenset, repr=False)

    def to_dict(self) -> dict:
        # ``_present`` keeps explicit ``false``/``null`` values from
        # the source (the fixtures write "bold": false and
        # "para_style": null) so a round trip is byte-identical.
        out: dict = {"text": self.text}
        if self.char_style is not None or "char_style" in self._present:
            out["char_style"] = self.char_style
        if self.font_size is not None or "font_size" in self._present:
            out["font_size"] = self.font_size
        if self.bold or "bold" in self._present:
            out["bold"] = self.bold
        if self.italic or "italic" in self._present:
            out["italic"] = self.italic
        if self.color is not None or "color" in self._present:
            out["color"] = self.color
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> CharRun:
        return cls(text=str(raw.get("text", "")),
                   char_style=raw.get("char_style"),
                   font_size=raw.get("font_size"),
                   bold=bool(raw.get("bold", False)),
                   italic=bool(raw.get("italic", False)),
                   color=raw.get("color"),
                   _present=frozenset(raw.keys()))


@dataclass
class ParaFormat:
    """One paragraph: runs plus its paragraph style and alignment
    (upstream ``story.rs`` ``ParaFormat``)."""

    runs: list[CharRun] = field(default_factory=list)
    para_style: str | None = None
    align: str | None = None
    extra: dict = field(default_factory=dict)
    _present: frozenset = field(default_factory=frozenset, repr=False)

    def to_dict(self) -> dict:
        out: dict = {"runs": [r.to_dict() for r in self.runs]}
        if self.para_style is not None or "para_style" in self._present:
            out["para_style"] = self.para_style
        if self.align is not None or "align" in self._present:
            out["align"] = self.align
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> ParaFormat:
        known = {"runs", "para_style", "align"}
        return cls(runs=[CharRun.from_dict(r)
                         for r in raw.get("runs", [])],
                   para_style=raw.get("para_style"),
                   align=raw.get("align"),
                   extra={k: v for k, v in raw.items() if k not in known},
                   _present=frozenset(raw.keys()))

    def text(self) -> str:
        return "".join(r.text for r in self.runs)


@dataclass
class Story:
    """A threaded story (upstream ``story.rs``): paragraphs that
    flow through linked text frames."""

    id: str = "S1"
    paras: list[ParaFormat] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"id": self.id,
                     "paras": [p.to_dict() for p in self.paras]}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Story:
        known = {"id", "paras"}
        return cls(id=str(raw.get("id", "S1")),
                   paras=[ParaFormat.from_dict(p)
                          for p in raw.get("paras", [])],
                   extra={k: v for k, v in raw.items() if k not in known})

    def text(self) -> str:
        return "\n".join(p.text() for p in self.paras)


@dataclass
class TextFrameOptions:
    """Text frame options (upstream ``item/text_frame.rs``)."""

    columns: int = 1
    gutter: float = 12.0
    vertical_justification: str = "top"  # top | center | bottom | justify
    first_baseline: str = "ascent"  # ascent | cap_height | leading | fixed
    auto_size: str = "off"  # off | height | width | width_height
    next_frame_id: str | None = None

    def to_dict(self) -> dict:
        return {"columns": self.columns, "gutter": self.gutter,
                "vertical_justification": self.vertical_justification,
                "first_baseline": self.first_baseline,
                "auto_size": self.auto_size,
                "next_frame_id": self.next_frame_id}

    @classmethod
    def from_dict(cls, raw: dict) -> TextFrameOptions:
        return cls(columns=int(raw.get("columns", 1)),
                   gutter=float(raw.get("gutter", 12.0)),
                   vertical_justification=str(
                       raw.get("vertical_justification", "top")),
                   first_baseline=str(raw.get("first_baseline", "ascent")),
                   auto_size=str(raw.get("auto_size", "off")),
                   next_frame_id=raw.get("next_frame_id"))


@dataclass
class FlexLayout:
    """A flex layout on a group item (upstream ``doc/flex.rs``:
    ``FlexLayout``). ``width``/``height`` are ``"fixed"`` or
    ``"auto"``; ``padding`` is [top, left, bottom, right]. The
    direction/justify/align/size fields are str enums, so they
    compare equal to their plain-string values."""

    direction: FlexDirection = FlexDirection.Row  # row | column
    justify: FlexJustify = FlexJustify.FlexStart
    align: FlexAlign = FlexAlign.Start  # start | center | end
    wrap: bool = False
    gap: float = 0.0
    row_gap: float | None = None
    padding: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    width: FlexSize = FlexSize.Fixed  # fixed | auto
    height: FlexSize = FlexSize.Fixed
    reverse: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"direction": _enum_value(self.direction),
                     "justify": _enum_value(self.justify),
                     "align": _enum_value(self.align),
                     "wrap": self.wrap, "gap": self.gap,
                     "row_gap": self.row_gap if self.row_gap is not None
                     else self.gap,
                     "padding": self.padding,
                     "width": _enum_value(self.width),
                     "height": _enum_value(self.height)}
        if self.reverse:
            out["reverse"] = True
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> FlexLayout:
        known = {"direction", "justify", "align", "wrap", "gap", "row_gap",
                 "padding", "width", "height", "reverse"}

        def _flex_enum(enum_cls, value, default):
            try:
                return enum_cls(value)
            except ValueError:
                return default

        return cls(direction=_flex_enum(
                       FlexDirection, raw.get("direction", "row"),
                       FlexDirection.Row),
                   justify=_flex_enum(
                       FlexJustify, raw.get("justify", "flex-start"),
                       FlexJustify.FlexStart),
                   align=_flex_enum(FlexAlign, raw.get("align", "start"),
                                    FlexAlign.Start),
                   wrap=bool(raw.get("wrap", False)),
                   gap=float(raw.get("gap", 0.0)),
                   row_gap=raw.get("row_gap"),
                   padding=[float(v) for v in
                            raw.get("padding", [0.0, 0.0, 0.0, 0.0])],
                   width=_flex_enum(FlexSize, raw.get("width", "fixed"),
                                    FlexSize.Fixed),
                   height=_flex_enum(FlexSize, raw.get("height", "fixed"),
                                     FlexSize.Fixed),
                   reverse=bool(raw.get("reverse", False)),
                   extra={k: v for k, v in raw.items() if k not in known})


@dataclass
class Item:
    """A page item (upstream ``item.rs`` ``Item``): geometry in its
    inner coordinate space (``path``), mapped to spread space by
    ``xf``. Content is unassigned, text (a story + frame options),
    graphic (a placed asset), or a group of items."""

    id: str = "I1"
    name: str = ""
    layer: str = "L1"
    shape: str = "rectangle"  # rectangle | oval | polygon | line | path | group
    path_d: str = "M0,0 L100,0 L100,100 L0,100 Z"
    xf: Affine = field(default_factory=Affine.identity)
    fill: Fill = field(default_factory=Fill)
    stroke: Stroke = field(default_factory=Stroke)
    corners: dict = field(default_factory=dict)
    opacity: float = 1.0
    blend: str = "normal"
    effects: dict = field(default_factory=dict)
    wrap: dict = field(default_factory=dict)
    locked: bool = False
    hidden: bool = False
    nonprinting: bool = False
    object_style: str = ""
    content: dict = field(default_factory=lambda: {"type": "unassigned"})
    overrides: str | None = None
    label: str = ""
    alt_text: str = ""
    liquid: dict | None = None
    flex: FlexLayout | None = None
    # Group children, live objects (not raw dicts) so ``all_items``
    # returns the stored item, never a copy — an editor that moves
    # an item found through ``all_items`` mutates the document.
    children: list[Item] = field(default_factory=list, repr=False)
    extra: dict = field(default_factory=dict)
    _present: frozenset = field(default_factory=frozenset, repr=False)
    _path_cache: Path | None = field(default=None, repr=False)

    # -- geometry ---------------------------------------------------
    def path(self) -> Path:
        if self._path_cache is None:
            self._path_cache = parse_path_data(self.path_d)
        return self._path_cache

    def local_bounds(self) -> tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = self.path().bounds()
        return (x0, y0, x1, y1)

    def bounds(self, parent: Affine | None = None
               ) -> tuple[float, float, float, float]:
        """The item's bounds in spread space. A group child's ``xf``
        maps into its parent's space, so pass the parent's spread
        transform (``None`` = identity) for nested items."""
        x0, y0, x1, y1 = self.local_bounds()
        xf = self.xf if parent is None else parent * self.xf
        corners = [xf.apply(p) for p in
                   ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys))

    def content_type(self) -> str:
        return str(self.content.get("type", "unassigned"))

    @property
    def story_id(self) -> str | None:
        if self.content_type() == "text":
            return self.content.get("story_id")
        return None

    def text_options(self) -> TextFrameOptions:
        if self.content_type() == "text":
            return TextFrameOptions.from_dict(
                self.content.get("options", {}))
        return TextFrameOptions()

    def content_items(self) -> list[Item]:
        if self.content_type() == "group":
            return self.children
        return []

    def set_content_items(self, items: list[Item]) -> None:
        self.children = list(items)
        self.content = {"type": "group"}
        self._present = frozenset(set(self._present) | {"content"})

    # -- serialization ----------------------------------------------
    def to_dict(self) -> dict:
        content = dict(self.content)
        if self.content_type() == "group":
            content["items"] = [c.to_dict() for c in self.children]
        out: dict = {"id": self.id, "name": self.name, "layer": self.layer,
                     "shape": self.shape,
                     "path": {"d": self.path_d},
                     "xf": {"a": self.xf.a, "b": self.xf.b, "c": self.xf.c,
                            "d": self.xf.d, "e": self.xf.e, "f": self.xf.f},
                     "fill": self.fill.to_dict(),
                     "stroke": self.stroke.to_dict(),
                     "opacity": self.opacity,
                     "content": content}
        if self.corners or "corners" in self._present:
            out["corners"] = self.corners
        if self.blend != "normal" or "blend" in self._present:
            out["blend"] = self.blend
        if self.effects or "effects" in self._present:
            out["effects"] = self.effects
        if self.wrap or "wrap" in self._present:
            out["wrap"] = self.wrap
        if self.locked or "locked" in self._present:
            out["locked"] = self.locked
        if self.hidden or "hidden" in self._present:
            out["hidden"] = self.hidden
        if self.nonprinting or "nonprinting" in self._present:
            out["nonprinting"] = self.nonprinting
        if self.object_style or "object_style" in self._present:
            out["object_style"] = self.object_style
        if self.overrides is not None:
            out["overrides"] = self.overrides
        if self.label or "label" in self._present:
            out["label"] = self.label
        if self.alt_text or "alt_text" in self._present:
            out["alt_text"] = self.alt_text
        if self.liquid is not None:
            out["liquid"] = self.liquid
        if self.flex is not None:
            out["flex"] = self.flex.to_dict()
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Item:
        known = {"id", "name", "layer", "shape", "path", "xf", "fill",
                 "stroke", "corners", "opacity", "blend", "effects",
                 "wrap", "locked", "hidden", "nonprinting",
                 "object_style", "content", "overrides", "label",
                 "alt_text", "liquid", "flex"}
        path = raw.get("path", {})
        xf = raw.get("xf", {})
        content = dict(raw.get("content", {"type": "unassigned"}))
        children = ([Item.from_dict(r) for r in content.get("items", [])]
                    if content.get("type") == "group" else [])
        return cls(
            id=str(raw.get("id", "I1")),
            name=str(raw.get("name", "")),
            layer=str(raw.get("layer", "L1")),
            shape=str(raw.get("shape", "rectangle")),
            path_d=str(path.get("d", "M0,0 L100,0 L100,100 L0,100 Z")),
            xf=Affine(a=float(xf.get("a", 1.0)), b=float(xf.get("b", 0.0)),
                      c=float(xf.get("c", 0.0)), d=float(xf.get("d", 1.0)),
                      e=float(xf.get("e", 0.0)), f=float(xf.get("f", 0.0))),
            fill=Fill.from_dict(raw.get("fill", {})),
            stroke=Stroke.from_dict(raw.get("stroke", {})),
            corners=dict(raw.get("corners", {})),
            opacity=float(raw.get("opacity", 1.0)),
            blend=str(raw.get("blend", "normal")),
            effects=dict(raw.get("effects", {})),
            wrap=dict(raw.get("wrap", {})),
            locked=bool(raw.get("locked", False)),
            hidden=bool(raw.get("hidden", False)),
            nonprinting=bool(raw.get("nonprinting", False)),
            object_style=str(raw.get("object_style", "")),
            content=content,
            overrides=raw.get("overrides"),
            label=str(raw.get("label", "")),
            alt_text=str(raw.get("alt_text", "")),
            liquid=raw.get("liquid"),
            flex=(FlexLayout.from_dict(raw["flex"])
                  if raw.get("flex") is not None else None),
            children=children,
            extra={k: v for k, v in raw.items() if k not in known},
            _present=frozenset(raw.keys()),
        )

    def canonical(self) -> tuple:
        """The item's own attributes only — children are compared
        as their own entries (same habit as the vector organ's
        ``GroupNode.canonical``)."""
        return (self.id, self.name, self.layer, self.shape, self.path_d,
                round(self.xf.a, 6), round(self.xf.b, 6),
                round(self.xf.c, 6), round(self.xf.d, 6),
                round(self.xf.e, 6), round(self.xf.f, 6),
                self.fill.swatch, round(self.fill.tint, 6),
                self.stroke.color, round(self.stroke.width, 6),
                round(self.opacity, 6), self.blend, self.locked,
                self.hidden, self.content_type(),
                self.story_id,
                tuple(sorted((k, str(v)) for k, v in
                             self.content.items()
                             if k not in ("items", "story_id"))))


@dataclass
class Page:
    """One page (upstream ``page.rs`` ``Page``)."""

    id: str = "P1"
    name: str = "1"
    width: float = 612.0
    height: float = 792.0
    margins: Margins = field(default_factory=Margins)
    columns: Columns = field(default_factory=Columns)
    bleed: float = 0.0
    items: list[Item] = field(default_factory=list)
    guides: list[Guide] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"id": self.id, "name": self.name, "width": self.width,
               "height": self.height,
               "margins": self.margins.to_dict(),
               "columns": self.columns.to_dict(),
               "bleed": self.bleed,
               "items": [i.to_dict() for i in self.items],
               "guides": [g.to_dict() for g in self.guides]}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Page:
        known = {"id", "name", "width", "height", "margins", "columns",
                 "bleed", "items", "guides"}
        return cls(id=str(raw.get("id", "P1")),
                   name=str(raw.get("name", "1")),
                   width=float(raw.get("width", 612.0)),
                   height=float(raw.get("height", 792.0)),
                   margins=Margins.from_dict(raw.get("margins", {})),
                   columns=Columns.from_dict(raw.get("columns", {})),
                   bleed=float(raw.get("bleed", 0.0)),
                   items=[Item.from_dict(i) for i in raw.get("items", [])],
                   guides=[Guide.from_dict(g)
                           for g in raw.get("guides", [])],
                   extra={k: v for k, v in raw.items() if k not in known})

    def margin_box(self) -> tuple[float, float, float, float]:
        """The live area inside the margins: (x0, y0, x1, y1)."""
        m = self.margins
        return (m.inside, m.top, self.width - m.outside,
                self.height - m.bottom)


@dataclass
class Spread:
    """A spread: one or two pages, optionally on a parent spread
    (upstream ``page.rs`` ``Spread``)."""

    id: str = "SP1"
    parent_id: str | None = None
    pages: list[Page] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"id": self.id, "parent_id": self.parent_id,
                     "pages": [p.to_dict() for p in self.pages]}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Spread:
        known = {"id", "parent_id", "pages"}
        return cls(id=str(raw.get("id", "SP1")),
                   parent_id=raw.get("parent_id"),
                   pages=[Page.from_dict(p) for p in raw.get("pages", [])],
                   extra={k: v for k, v in raw.items() if k not in known})


# ── document ──────────────────────────────────────────────────────

@dataclass
class LayoutDocument:
    """A page-layout document (upstream ``Document`` in
    ``crates/doc/src/lib.rs``)."""

    title: str = ""
    settings: dict = field(default_factory=lambda: {"facing_pages": True,
                                                    "units": "pt"})
    layers: list[Layer] = field(default_factory=list)
    swatches: list[Swatch] = field(default_factory=list)
    styles: dict = field(default_factory=lambda: {"paragraph": [],
                                                  "character": [],
                                                  "object": []})
    stories: dict[str, Story] = field(default_factory=dict)
    spreads: list[Spread] = field(default_factory=list)
    parents: list[Spread] = field(default_factory=list)
    grid: DocumentGrid = field(default_factory=DocumentGrid)
    sections: list[Section] = field(default_factory=list)
    assets: dict[str, bytes] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    _dirty_ids: set = field(default_factory=set, repr=False)
    _removed_ids: set = field(default_factory=set, repr=False)

    # -- construction ------------------------------------------------
    @classmethod
    def blank(cls, title: str = "") -> LayoutDocument:
        page = Page(id="P1", name="1")
        spread = Spread(id="SP1", pages=[page])
        return cls(title=title, layers=[Layer()],
                   swatches=[Swatch(name="Black", kind="process",
                                    rgb=[0, 0, 0]),
                             Swatch(name="None", kind="none")],
                   spreads=[spread])

    @classmethod
    def from_bytes(cls, data: bytes) -> LayoutDocument:
        from .codec import parse_document
        return parse_document(data)

    def to_bytes(self) -> bytes:
        from .codec import to_document_bytes
        return to_document_bytes(self)

    # -- inventory ---------------------------------------------------
    def page_count(self) -> int:
        return sum(len(s.pages) for s in self.spreads)

    def item_count(self) -> int:
        return len(self.all_items())

    def story_count(self) -> int:
        return len(self.stories)

    def all_items(self) -> list[Item]:
        """Every item in document order (spread, page, z-order),
        including items nested in groups (flattened, parents
        first)."""
        out: list[Item] = []

        def walk(items: list[Item]) -> None:
            for item in items:
                out.append(item)
                if item.content_type() == "group":
                    walk(item.content_items())

        for spread in self.spreads:
            for page in spread.pages:
                walk(page.items)
        return out

    def find_item(self, item_id: str) -> Item | None:
        for item in self.all_items():
            if item.id == item_id:
                return item
        return None

    def item_bounds(self, item: Item) -> tuple[float, float, float, float]:
        """An item's bounds in spread space, composing the parent
        chain for group children (whose ``xf`` is relative to the
        group's space)."""
        for spread in self.spreads:
            for page in spread.pages:
                found = self._bounds_in(page.items, item, None)
                if found is not None:
                    return found
        return item.bounds()

    @staticmethod
    def _bounds_in(items: list[Item], target: Item,
                   parent: Affine | None
                   ) -> tuple[float, float, float, float] | None:
        for candidate in items:
            if candidate is target:
                return candidate.bounds(parent)
            if candidate.content_type() == "group":
                child_parent = (candidate.xf if parent is None
                                else parent * candidate.xf)
                found = LayoutDocument._bounds_in(
                    candidate.children, target, child_parent)
                if found is not None:
                    return found
        return None

    def find_page(self, page_id: str) -> Page | None:
        for spread in self.spreads:
            for page in spread.pages:
                if page.id == page_id:
                    return page
        return None

    def find_spread(self, spread_id: str) -> Spread | None:
        for spread in self.spreads:
            if spread.id == spread_id:
                return spread
        return None

    def story_text(self, story_id: str | None) -> str:
        if story_id is None:
            return ""
        story = self.stories.get(story_id)
        return story.text() if story is not None else ""

    def swatch_rgb(self, name: str) -> tuple[int, int, int] | None:
        for swatch in self.swatches:
            if swatch.name == name:
                return swatch.resolve_rgb()
        return None

    def swatch(self, name: str) -> Swatch | None:
        for swatch in self.swatches:
            if swatch.name == name:
                return swatch
        return None

    def mark_dirty(self, item_id: str) -> None:
        """Record that an editor touched ``item_id``. A restyle to
        the value the item already had changes no canonical byte,
        but the item was still edited — the fidelity report counts
        it as changed (and never as an untouched change)."""
        self._dirty_ids.add(item_id)

    # -- fidelity ----------------------------------------------------
    def fidelity_report(self, original_bytes: bytes) -> dict:
        """Compare this document against ``original_bytes`` at the
        canonical-item level. ``untouched_changed`` counts changed
        items the editor never touched — the number that must be 0
        for a fidelity-preserving edit."""
        from .codec import parse_document
        original = parse_document(original_bytes)
        before = {i.id: i.canonical() for i in original.all_items()}
        after = {i.id: i.canonical() for i in self.all_items()}
        common = before.keys() & after.keys()
        diff = {i for i in common if before[i] != after[i]}
        changed = sorted(diff | (set(self._dirty_ids) & common))
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        untouched = [i for i in changed if i not in self._dirty_ids]
        return {"changed_ids": changed,
                "added": len(added), "added_ids": added,
                "removed": len(removed), "removed_ids": removed,
                "untouched_changed": len(untouched),
                "untouched_changed_ids": untouched}

    # -- serialization -----------------------------------------------
    def to_dict(self) -> dict:
        out: dict = {"format_version": 1, "title": self.title,
                     "settings": self.settings,
                     "layers": [layer.to_dict() for layer in self.layers],
                     "swatches": [s.to_dict() for s in self.swatches],
                     "styles": self.styles,
                     "stories": [s.to_dict() for s in self.stories.values()],
                     "spreads": [s.to_dict() for s in self.spreads],
                     "parents": [s.to_dict() for s in self.parents],
                     "grid": self.grid.to_dict(),
                     "sections": [s.to_dict() for s in self.sections]}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> LayoutDocument:
        if raw.get("format_version") != 1:
            raise LayoutError(
                f"unsupported layout format version "
                f"{raw.get('format_version')!r} (want 1)",
                kind="unsupported")
        known = {"format_version", "title", "settings", "layers",
                 "swatches", "styles", "stories", "spreads", "parents",
                 "grid", "sections"}
        stories: dict[str, Story] = {}
        for entry in raw.get("stories", []):
            story = Story.from_dict(entry)
            stories[story.id] = story
        return cls(
            title=str(raw.get("title", "")),
            settings=dict(raw.get("settings", {"facing_pages": True,
                                               "units": "pt"})),
            layers=[Layer.from_dict(layer)
                    for layer in raw.get("layers", [])],
            swatches=[Swatch.from_dict(s)
                      for s in raw.get("swatches", [])],
            styles=dict(raw.get("styles", {"paragraph": [],
                                           "character": [], "object": []})),
            stories=stories,
            spreads=[Spread.from_dict(s) for s in raw.get("spreads", [])],
            parents=[Spread.from_dict(s) for s in raw.get("parents", [])],
            grid=DocumentGrid.from_dict(raw.get("grid", {})),
            sections=[Section.from_dict(s)
                      for s in raw.get("sections", [])],
            extra={k: v for k, v in raw.items() if k not in known},
        )
