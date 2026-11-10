"""The Devon Style: Atlas cartography as data (docs 19+21, M3).

Doc 21's product split is explicit: the renderer is a replaceable
boundary (S5); **the style is Devon's**.  So the style lives here
as a *document* — a JSON-shaped value the vector renderer consumes
and nothing in the renderer hardcodes.  Restyling Atlas means
editing this data (or handing the renderer another document); it
never means editing renderer code, and a test pins exactly that.

The default document is Devon's own dark cartography, tuned to
read as a chat photo: a deep ink background, luminous water, a
gold hierarchy for the roads that matter (the house color — the
same gold the trading desk speaks), receding greys for the roads
that don't, violet boundaries, warm gold points of interest.
Colors are named constants so a surface, a test, or a future
light/contrast variant can speak about them precisely.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .coords import MAX_ZOOM

__all__ = [
    "BACKGROUND", "BOUNDARY", "BUILDING_FILL", "BUILDING_STROKE",
    "LABEL_HALO", "LABEL_TEXT", "LANDCOVER_WOOD", "LANDUSE_RESIDENTIAL",
    "NO_DATA", "POI_FILL", "ROAD_CASING", "ROAD_MINOR", "ROAD_MOTORWAY",
    "ROAD_PRIMARY", "ROAD_SECONDARY", "WATER", "WATERWAY",
    "DEFAULT_STYLE", "KIND_FILL", "KIND_LABEL", "KIND_LINE",
    "KIND_POINT", "STYLE_FORMAT", "MapStyle", "StyleError",
    "StyleLayer", "parse_color", "style_hash",
]

#: The style document's own format version.
STYLE_FORMAT = 1

#: Layer kinds the renderer knows how to paint.
KIND_FILL = "fill"
KIND_LINE = "line"
KIND_POINT = "point"
KIND_LABEL = "label"
_KINDS = (KIND_FILL, KIND_LINE, KIND_POINT, KIND_LABEL)

# ── the Devon palette (named, so it can be spoken about) ─────────

#: The deep ink the whole map sits on.
BACKGROUND = "#0a0d13"
#: The M1 no-data color, restated in the document so the style is
#: the whole visual truth of a render (26, 26, 32).
NO_DATA = "#1a1a20"
#: Luminous water — the brightest mass on the map after the roads.
WATER = "#17648f"
#: Rivers and canals: water, one step brighter, drawn thin.
WATERWAY = "#2b84b8"
#: Woods and green landcover: deep, quiet green.
LANDCOVER_WOOD = "#142a1e"
#: Residential land: barely lifted off the background — presence,
#: not emphasis.
LANDUSE_RESIDENTIAL = "#12161f"
#: Buildings: a slate block with a slightly lighter edge.
BUILDING_FILL = "#1c2433"
BUILDING_STROKE = "#2a3648"
#: Administrative boundaries: violet, unmistakably not a road.
BOUNDARY = "#7a5fa0"
#: The road casing: a hair darker than the background ink, so a
#: bright road reads as a ribbon laid on the map.
ROAD_CASING = "#05070a"
#: The gold hierarchy (house gold first) and the receding greys.
ROAD_MOTORWAY = "#e8b34b"
ROAD_PRIMARY = "#c9cfd8"
ROAD_SECONDARY = "#8d97a5"
ROAD_MINOR = "#5b6572"
#: Points of interest wear the house gold too.
POI_FILL = "#e8b34b"
#: Place labels: near-white text on the ink as its own halo.
LABEL_TEXT = "#e8ecf1"
LABEL_HALO = "#0a0d13"

_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class StyleError(ValueError):
    """A style document defect, named in the message."""


def parse_color(text: str) -> tuple[int, int, int]:
    """``"#rgb"``/``"#rrggbb"`` → an (r, g, b) tuple, or refuse."""
    if not isinstance(text, str) or not _COLOR_RE.match(text.strip()):
        raise StyleError("malformed color: " + repr(text))
    body = text.strip()[1:]
    if len(body) == 3:
        body = "".join(ch * 2 for ch in body)
    return (int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16))


def _check_color(text: str, where: str) -> None:
    try:
        parse_color(text)
    except StyleError as exc:
        raise StyleError(where + ": " + str(exc)) from exc


@dataclass(frozen=True)
class StyleLayer:
    """One paint rule: a source layer, a kind, and how it paints.

    ``source_layer`` names the MVT layer whose features this rule
    paints; ``aliases`` are other names packs may call the same
    plane (OpenMapTiles says ``transportation``, other pipelines
    say ``road`` — the style speaks both).  Class-keyed maps read
    the feature's ``class`` tag (then ``subclass``): ``class_fills``
    / ``class_strokes`` recolor per class, ``class_widths`` sets a
    class's line width outright.  Widths are output pixels at the
    rule's ``min_zoom``, growing by ``width_per_zoom`` per zoom
    step — the standard cartographic swell as you descend.
    """

    source_layer: str
    kind: str
    aliases: tuple[str, ...] = ()
    fill: str = ""
    stroke: str = ""
    width: float = 1.0
    width_per_zoom: float = 0.0
    min_zoom: int = 0
    max_zoom: int = MAX_ZOOM
    class_fills: tuple[tuple[str, str], ...] = ()
    class_strokes: tuple[tuple[str, str], ...] = ()
    class_widths: tuple[tuple[str, float], ...] = ()
    radius: float = 3.0
    text_field: str = ""
    text_size: float = 12.0
    text_size_per_zoom: float = 1.0
    text_color: str = ""
    halo_color: str = ""
    halo_width: int = 2

    def __post_init__(self) -> None:
        where = "style layer " + repr(self.source_layer)
        if self.kind not in _KINDS:
            raise StyleError(
                where + ": unknown layer kind " + repr(self.kind)
                + " (known: " + ", ".join(_KINDS) + ")")
        if not (0 <= int(self.min_zoom) <= int(self.max_zoom)
                <= MAX_ZOOM):
            raise StyleError(
                where + ": bad zoom range " + repr(self.min_zoom)
                + ".." + repr(self.max_zoom))
        if self.fill:
            _check_color(self.fill, where + " fill")
        if self.stroke:
            _check_color(self.stroke, where + " stroke")
        if float(self.width) < 0 or float(self.width_per_zoom) < 0:
            raise StyleError(where + ": negative line width")
        if float(self.radius) < 0:
            raise StyleError(where + ": negative point radius")
        for klass, color in self.class_fills:
            _check_color(color, where + " class fill " + repr(klass))
        for klass, color in self.class_strokes:
            _check_color(color, where + " class stroke " + repr(klass))
        for klass, width in self.class_widths:
            if float(width) < 0:
                raise StyleError(
                    where + ": negative class width " + repr(klass))
        if self.kind in (KIND_FILL, KIND_POINT) and not self.fill \
                and not self.class_fills:
            raise StyleError(
                where + ": a " + self.kind + " layer needs a fill")
        if self.kind == KIND_LINE and not self.stroke \
                and not self.class_strokes:
            raise StyleError(where + ": a line layer needs a stroke")
        if self.kind == KIND_LABEL:
            if not self.text_field:
                raise StyleError(
                    where + ": a label layer needs a text_field")
            if not self.text_color:
                raise StyleError(
                    where + ": a label layer needs a text_color")
            _check_color(self.text_color, where + " text color")
            if self.halo_color:
                _check_color(self.halo_color, where + " halo color")
            if float(self.text_size) <= 0:
                raise StyleError(where + ": text_size must be positive")

    # ── lookups the renderer makes ────────────────────────────────

    def matches(self, layer_name: str) -> bool:
        return layer_name == self.source_layer \
            or layer_name in self.aliases

    def applies_at(self, zoom: int) -> bool:
        return int(self.min_zoom) <= int(zoom) <= int(self.max_zoom)

    @staticmethod
    def _class_of(tags: dict) -> str:
        return str(tags.get("class") or tags.get("subclass") or "")

    def fill_for(self, tags: dict) -> tuple[int, int, int] | None:
        klass = self._class_of(tags)
        for name, color in self.class_fills:
            if name == klass:
                return parse_color(color)
        return parse_color(self.fill) if self.fill else None

    def stroke_for(self, tags: dict) -> tuple[int, int, int] | None:
        klass = self._class_of(tags)
        for name, color in self.class_strokes:
            if name == klass:
                return parse_color(color)
        return parse_color(self.stroke) if self.stroke else None

    def width_for(self, tags: dict, zoom: int) -> float:
        klass = self._class_of(tags)
        base = float(self.width)
        for name, width in self.class_widths:
            if name == klass:
                base = float(width)
                break
        return base + float(self.width_per_zoom) \
            * max(0, int(zoom) - int(self.min_zoom))

    def text_size_for(self, zoom: int) -> float:
        return float(self.text_size) + float(self.text_size_per_zoom) \
            * max(0, int(zoom) - int(self.min_zoom))

    # ── (de)serialization: the document is data ───────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_layer": self.source_layer, "kind": self.kind,
            "aliases": list(self.aliases), "fill": self.fill,
            "stroke": self.stroke, "width": float(self.width),
            "width_per_zoom": float(self.width_per_zoom),
            "min_zoom": int(self.min_zoom),
            "max_zoom": int(self.max_zoom),
            "class_fills": [list(pair) for pair in self.class_fills],
            "class_strokes": [list(pair) for pair in self.class_strokes],
            "class_widths": [list(pair) for pair in self.class_widths],
            "radius": float(self.radius),
            "text_field": self.text_field,
            "text_size": float(self.text_size),
            "text_size_per_zoom": float(self.text_size_per_zoom),
            "text_color": self.text_color,
            "halo_color": self.halo_color,
            "halo_width": int(self.halo_width),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StyleLayer:
        return cls(
            source_layer=str(data["source_layer"]),
            kind=str(data["kind"]),
            aliases=tuple(str(a) for a in data.get("aliases", ())),
            fill=str(data.get("fill", "")),
            stroke=str(data.get("stroke", "")),
            width=float(data.get("width", 1.0)),
            width_per_zoom=float(data.get("width_per_zoom", 0.0)),
            min_zoom=int(data.get("min_zoom", 0)),
            max_zoom=int(data.get("max_zoom", MAX_ZOOM)),
            class_fills=tuple((str(k), str(v)) for k, v
                              in data.get("class_fills", ())),
            class_strokes=tuple((str(k), str(v)) for k, v
                                in data.get("class_strokes", ())),
            class_widths=tuple((str(k), float(v)) for k, v
                               in data.get("class_widths", ())),
            radius=float(data.get("radius", 3.0)),
            text_field=str(data.get("text_field", "")),
            text_size=float(data.get("text_size", 12.0)),
            text_size_per_zoom=float(
                data.get("text_size_per_zoom", 1.0)),
            text_color=str(data.get("text_color", "")),
            halo_color=str(data.get("halo_color", "")),
            halo_width=int(data.get("halo_width", 2)))


@dataclass(frozen=True)
class MapStyle:
    """A whole style document: background + ordered paint rules.

    Layers paint in document order — the list IS the z-order.
    """

    name: str
    background: str = BACKGROUND
    no_data: str = NO_DATA
    layers: tuple[StyleLayer, ...] = field(default_factory=tuple)
    format: int = STYLE_FORMAT

    def __post_init__(self) -> None:
        if int(self.format) != STYLE_FORMAT:
            raise StyleError(
                "unsupported style format: " + repr(self.format))
        if not self.name:
            raise StyleError("a style document needs a name")
        _check_color(self.background, "style background")
        _check_color(self.no_data, "style no_data")

    def to_dict(self) -> dict[str, Any]:
        return {"format": int(self.format), "name": self.name,
                "background": self.background, "no_data": self.no_data,
                "layers": [layer.to_dict() for layer in self.layers]}

    @classmethod
    def from_dict(cls, data: dict) -> MapStyle:
        return cls(
            name=str(data["name"]),
            background=str(data.get("background", BACKGROUND)),
            no_data=str(data.get("no_data", NO_DATA)),
            layers=tuple(StyleLayer.from_dict(entry)
                         for entry in data.get("layers", ())),
            format=int(data.get("format", STYLE_FORMAT)))


def style_hash(style: MapStyle) -> str:
    """SHA-256 over the document's canonical JSON (receipt evidence)."""
    canonical = json.dumps(style.to_dict(), sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── the default Devon document ────────────────────────────────────

DEFAULT_STYLE = MapStyle(
    name="devon-dark",
    background=BACKGROUND,
    no_data=NO_DATA,
    layers=(
        # Land first: quiet masses everything else sits on.
        StyleLayer(
            source_layer="landcover", kind=KIND_FILL,
            fill="#101a14",
            class_fills=(("wood", LANDCOVER_WOOD),
                         ("grass", "#152619"),
                         ("ice", "#26343e"),
                         ("sand", "#2a2517"),
                         ("wetland", "#12242a"))),
        StyleLayer(
            source_layer="landuse", kind=KIND_FILL,
            fill=LANDUSE_RESIDENTIAL,
            class_fills=(("residential", LANDUSE_RESIDENTIAL),
                         ("commercial", "#171420"),
                         ("industrial", "#191a16"),
                         ("cemetery", "#12211a"),
                         ("hospital", "#1a1620"),
                         ("school", "#161c24"))),
        # Water over land: the luminous mass.
        StyleLayer(
            source_layer="water", kind=KIND_FILL, fill=WATER),
        StyleLayer(
            source_layer="waterway", kind=KIND_LINE, stroke=WATERWAY,
            width=1.0, width_per_zoom=0.15, min_zoom=8,
            class_widths=(("river", 1.8), ("canal", 1.4),
                          ("stream", 0.8))),
        # Buildings: present at street zooms, never shouting.
        StyleLayer(
            source_layer="building", kind=KIND_FILL,
            fill=BUILDING_FILL, stroke=BUILDING_STROKE,
            min_zoom=13),
        # Boundaries under the roads they bound.
        StyleLayer(
            source_layer="boundary", kind=KIND_LINE, stroke=BOUNDARY,
            width=1.2, width_per_zoom=0.05,
            class_widths=(("country", 2.0), ("state", 1.4),
                          ("county", 0.9))),
        # Roads: a dark casing ribbon, then the hierarchy itself.
        StyleLayer(
            source_layer="transportation", kind=KIND_LINE,
            aliases=("road", "roads"), stroke=ROAD_CASING,
            width=1.6, width_per_zoom=0.35, min_zoom=6,
            class_widths=(("motorway", 3.4), ("trunk", 3.0),
                          ("primary", 2.6), ("secondary", 2.2),
                          ("tertiary", 1.8), ("minor", 1.4),
                          ("service", 1.1), ("path", 0.9),
                          ("track", 0.9), ("rail", 1.2))),
        StyleLayer(
            source_layer="transportation", kind=KIND_LINE,
            aliases=("road", "roads"), stroke=ROAD_MINOR,
            width=1.0, width_per_zoom=0.3, min_zoom=6,
            class_strokes=(("motorway", ROAD_MOTORWAY),
                           ("trunk", "#d98e4a"),
                           ("primary", ROAD_PRIMARY),
                           ("secondary", ROAD_SECONDARY),
                           ("tertiary", "#78828f"),
                           ("minor", ROAD_MINOR),
                           ("service", "#4c5560"),
                           ("path", "#3d4652"),
                           ("track", "#3d4652"),
                           ("rail", "#6a7684")),
            class_widths=(("motorway", 2.2), ("trunk", 1.9),
                          ("primary", 1.7), ("secondary", 1.4),
                          ("tertiary", 1.2), ("minor", 1.0),
                          ("service", 0.8), ("path", 0.6),
                          ("track", 0.6), ("rail", 0.8))),
        # Points of interest: small gold marks.
        StyleLayer(
            source_layer="poi", kind=KIND_POINT, fill=POI_FILL,
            radius=2.5, min_zoom=14),
        # Labels last, over everything they name.
        StyleLayer(
            source_layer="place", kind=KIND_LABEL,
            aliases=("place_label", "places"),
            text_field="name", text_size=11.0, text_size_per_zoom=0.8,
            text_color=LABEL_TEXT, halo_color=LABEL_HALO,
            halo_width=2, min_zoom=4),
        StyleLayer(
            source_layer="poi", kind=KIND_LABEL,
            text_field="name", text_size=10.0, text_size_per_zoom=0.5,
            text_color="#c8cfd8", halo_color=LABEL_HALO,
            halo_width=2, min_zoom=15),
        StyleLayer(
            source_layer="water_name", kind=KIND_LABEL,
            aliases=("water",),
            text_field="name", text_size=10.0, text_size_per_zoom=0.5,
            text_color="#7fc4e8", halo_color=LABEL_HALO,
            halo_width=2, min_zoom=8),
    ))
