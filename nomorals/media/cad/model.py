"""CAD document model — sketches, layers, blocks, dimensions,
constraints.

Port of the *design* of upstream ``crates/doc`` (``entity.rs``,
``constraint.rs``, ``anno.rs``, ``array.rs``) into Devon's Python,
stdlib only, with the fidelity discipline of the layout organ
(lane 104): every entity keeps the keys it was parsed with
(``_present``) plus any unknown keys (``extra``), so parse →
serialize reproduces source bytes exactly and
:meth:`CadDocument.fidelity_report` can count untouched entities
that changed.

Entity parameters follow upstream's solver model
(``crates/constraints/src/model.rs``): a line is ``a b``, a
circle ``center radius``, an arc ``center radius start end``
(angles in radians, counter-clockwise), a polyline its vertices,
a point ``x y``, an ellipse ``center major ratio`` plus
``start end`` for an elliptical arc, a spline its defining
points (fit points interpolated, control points B-spline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .errors import CadError
from .geom import (
    arc_point,
    ccw_sweep,
    dist,
    ellipse_at,
    spline_eval,
)

__all__ = [
    "Arc", "Block", "CadDocument", "CadPoint", "CadText", "Circle",
    "Constraint", "ConstraintKind", "Dimension", "Ellipse", "GeomRef",
    "Insert", "Layer", "Line", "Polyline", "Spline", "Sub",
]

#: A 2D point as (x, y).
Point = tuple[float, float]


def _pt(value) -> Point:
    return (float(value[0]), float(value[1]))


def _pts(value) -> list[Point]:
    return [_pt(p) for p in value]


def _num(value, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CadError(f"{what} is not a number: {value!r}",
                       kind="parse") from exc
    if not math.isfinite(out):
        raise CadError(f"{what} is not finite: {value!r}",
                       kind="parse")
    return out


def _fmt(value: float) -> str:
    """Shortest text that round-trips a float exactly."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


class Sub(StrEnum):
    """Which part of an entity a constraint refers to (upstream
    ``crates/doc/src/constraint.rs`` ``Sub``)."""

    WHOLE = "whole"
    START = "start"
    END = "end"
    MID = "mid"
    CENTER = "center"

    @classmethod
    def parse(cls, text: str) -> Sub | tuple[str, int]:
        """Parse ``start``/``end``/``mid``/``center``/``whole``,
        ``v3``/``vertex3``, ``s2``/``seg2``/``segment2``."""
        low = str(text).strip().lower()
        if low in ("", "whole", "object", "o"):
            return cls.WHOLE
        if low in ("start", "s"):
            return cls.START
        if low in ("end", "e"):
            return cls.END
        if low in ("mid", "midpoint", "m"):
            return cls.MID
        if low in ("center", "centre", "c"):
            return cls.CENTER
        for prefix, vertex in (("vertex", True), ("segment", False),
                               ("seg", False), ("v", True)):
            if low.startswith(prefix):
                rest = low[len(prefix):].strip()
                if rest.isdigit():
                    return ("vertex", int(rest)) if vertex else (
                        "segment", int(rest))
        raise CadError(f"unknown sub-reference {text!r}",
                       kind="invalid_argument")

    def label(self) -> str:
        return str(self.value)


def _sub_label(sub) -> str:
    if isinstance(sub, tuple):
        return f"{'v' if sub[0] == 'vertex' else 'seg'}{sub[1]}"
    return str(sub.value) if isinstance(sub, Sub) else str(sub)


def _sub_from(value) -> Sub | tuple[str, int]:
    if isinstance(value, tuple):
        return value
    return Sub.parse(str(value))


@dataclass
class GeomRef:
    """A reference to (part of) an entity."""

    handle: str
    sub: Sub | tuple[str, int] = Sub.WHOLE

    def __post_init__(self) -> None:
        # Callers pass plain strings ("end", "whole", "v3", "seg2");
        # normalise them here so the solver can rely on Sub values
        # and ("vertex"/"segment", n) tuples everywhere.
        if isinstance(self.sub, str):
            self.sub = _sub_from(self.sub)

    def to_dict(self) -> dict:
        return {"handle": self.handle, "sub": _sub_label(self.sub)}

    @classmethod
    def from_dict(cls, raw: dict) -> GeomRef:
        return cls(handle=str(raw["handle"]),
                   sub=_sub_from(raw.get("sub", "whole")))


class ConstraintKind(StrEnum):
    """Constraint types (upstream ``ConstraintKind``; the
    dimensional ``Distance`` axis is spelled out as three kinds
    so TOML/JSON stay flat)."""

    COINCIDENT = "coincident"
    COLLINEAR = "collinear"
    CONCENTRIC = "concentric"
    FIX = "fix"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    TANGENT = "tangent"
    SMOOTH = "smooth"
    SYMMETRIC = "symmetric"
    EQUAL = "equal"
    DISTANCE_ALIGNED = "distance_aligned"
    DISTANCE_HORIZONTAL = "distance_horizontal"
    DISTANCE_VERTICAL = "distance_vertical"
    ANGULAR = "angular"
    RADIUS = "radius"
    DIAMETER = "diameter"

    def is_dimensional(self) -> bool:
        return self in (ConstraintKind.DISTANCE_ALIGNED,
                        ConstraintKind.DISTANCE_HORIZONTAL,
                        ConstraintKind.DISTANCE_VERTICAL,
                        ConstraintKind.ANGULAR,
                        ConstraintKind.RADIUS,
                        ConstraintKind.DIAMETER)

    def name(self) -> str:
        return {
            ConstraintKind.DISTANCE_ALIGNED: "Aligned",
            ConstraintKind.DISTANCE_HORIZONTAL: "Horizontal",
            ConstraintKind.DISTANCE_VERTICAL: "Vertical",
            ConstraintKind.ANGULAR: "Angular",
            ConstraintKind.RADIUS: "Radius",
            ConstraintKind.DIAMETER: "Diameter",
        }.get(self, self.value.title())


@dataclass
class Constraint:
    """One constraint: a kind, references, an optional value
    (lengths in drawing units, angles in degrees)."""

    id: int
    kind: ConstraintKind | str
    refs: list[GeomRef] = field(default_factory=list)
    name: str = ""
    expr: str = ""
    value: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            try:
                self.kind = ConstraintKind(self.kind)
            except ValueError as exc:
                raise CadError(f"unknown constraint kind {self.kind!r}",
                               kind="invalid_argument") from exc

    def to_dict(self) -> dict:
        out: dict = {"id": self.id, "kind": str(self.kind.value),
                     "refs": [r.to_dict() for r in self.refs]}
        if self.name:
            out["name"] = self.name
        if self.expr:
            out["expr"] = self.expr
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Constraint:
        return cls(
            id=int(raw["id"]),
            kind=raw["kind"],
            refs=[GeomRef.from_dict(r) for r in raw.get("refs", [])],
            name=str(raw.get("name", "")),
            expr=str(raw.get("expr", "")),
            value=(None if raw.get("value") is None
                   else float(raw["value"])),
        )


@dataclass
class Layer:
    name: str
    color: str = "#ffffff"
    visible: bool = True
    locked: bool = False
    frozen: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "color": self.color,
                "visible": self.visible, "locked": self.locked,
                "frozen": self.frozen}

    @classmethod
    def from_dict(cls, raw: dict) -> Layer:
        return cls(name=str(raw.get("name", "0")),
                   color=str(raw.get("color", "#ffffff")),
                   visible=bool(raw.get("visible", True)),
                   locked=bool(raw.get("locked", False)),
                   frozen=bool(raw.get("frozen", False)))


# ── entities ──────────────────────────────────────────────────────

@dataclass
class Line:
    id: str = field(default="", kw_only=True)
    a: Point
    b: Point
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "line", "id": self.id, "a": list(self.a),
               "b": list(self.b), "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Line:
        known = {"kind", "id", "a", "b", "layer", "color"}
        return cls(id=str(raw["id"]), a=_pt(raw["a"]), b=_pt(raw["b"]),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        return (min(self.a[0], self.b[0]), min(self.a[1], self.b[1]),
                max(self.a[0], self.b[0]), max(self.a[1], self.b[1]))

    def length(self) -> float:
        return dist(self.a, self.b)


@dataclass
class Arc:
    id: str = field(default="", kw_only=True)
    center: Point
    radius: float
    start: float
    end: float
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "arc", "id": self.id,
               "center": list(self.center), "radius": self.radius,
               "start": self.start, "end": self.end, "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Arc:
        known = {"kind", "id", "center", "radius", "start", "end",
                 "layer", "color"}
        return cls(id=str(raw["id"]), center=_pt(raw["center"]),
                   radius=_num(raw["radius"], "radius"),
                   start=float(raw["start"]), end=float(raw["end"]),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def start_point(self) -> Point:
        return arc_point(self.center, self.radius, self.start)

    def end_point(self) -> Point:
        return arc_point(self.center, self.radius, self.end)

    def sweep(self) -> float:
        return ccw_sweep(self.start, self.end)

    def bounds(self) -> tuple[float, float, float, float]:
        pts = [self.start_point(), self.end_point()]
        for ang in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
            if arc_contains(self.start, self.end, ang):
                pts.append(arc_point(self.center, self.radius, ang))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))


def arc_contains(start: float, end: float, angle: float) -> bool:
    if abs(ccw_sweep(start, end)) < 1e-12:
        return True
    return ccw_sweep(start, angle) <= ccw_sweep(start, end) + 1e-12


@dataclass
class Circle:
    id: str = field(default="", kw_only=True)
    center: Point
    radius: float
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "circle", "id": self.id,
               "center": list(self.center), "radius": self.radius,
               "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Circle:
        known = {"kind", "id", "center", "radius", "layer", "color"}
        return cls(id=str(raw["id"]), center=_pt(raw["center"]),
                   radius=_num(raw["radius"], "radius"),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        r = self.radius
        return (self.center[0] - r, self.center[1] - r,
                self.center[0] + r, self.center[1] + r)


@dataclass
class Ellipse:
    id: str = field(default="", kw_only=True)
    center: Point
    major: Point
    ratio: float
    start: float = 0.0
    end: float = math.tau
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "ellipse", "id": self.id,
               "center": list(self.center), "major": list(self.major),
               "ratio": self.ratio, "start": self.start,
               "end": self.end, "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Ellipse:
        known = {"kind", "id", "center", "major", "ratio", "start",
                 "end", "layer", "color"}
        return cls(id=str(raw["id"]), center=_pt(raw["center"]),
                   major=_pt(raw["major"]),
                   ratio=_num(raw["ratio"], "ratio"),
                   start=float(raw.get("start", 0.0)),
                   end=float(raw.get("end", math.tau)),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def is_full(self) -> bool:
        return abs(ccw_sweep(self.start, self.end)) < 1e-12

    def bounds(self) -> tuple[float, float, float, float]:
        pts = [ellipse_at(self.center, self.major, self.ratio,
                          self.start + self.sweep() * t / 8.0)
               for t in range(9)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def sweep(self) -> float:
        return ccw_sweep(self.start, self.end)


@dataclass
class Spline:
    id: str = field(default="", kw_only=True)
    points: list[Point] = field(default_factory=list)
    closed: bool = False
    fit: bool = True
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "spline", "id": self.id,
               "points": [list(p) for p in self.points],
               "closed": self.closed, "fit": self.fit,
               "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Spline:
        known = {"kind", "id", "points", "closed", "fit", "layer",
                 "color"}
        return cls(id=str(raw["id"]), points=_pts(raw["points"]),
                   closed=bool(raw.get("closed", False)),
                   fit=bool(raw.get("fit", True)),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        pts = [spline_eval(self.points, t / 16.0, self.closed, self.fit)
               for t in range(17)] or [(0.0, 0.0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class CadPoint:
    id: str = field(default="", kw_only=True)
    p: Point
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "point", "id": self.id, "p": list(self.p),
               "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> CadPoint:
        known = {"kind", "id", "p", "layer", "color"}
        return cls(id=str(raw["id"]), p=_pt(raw["p"]),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.p[0], self.p[1], self.p[0], self.p[1])


@dataclass
class Polyline:
    id: str = field(default="", kw_only=True)
    vertices: list[Point] = field(default_factory=list)
    closed: bool = False
    fill: bool = False
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "polyline", "id": self.id,
               "vertices": [list(p) for p in self.vertices],
               "closed": self.closed, "layer": self.layer}
        if self.fill:
            out["fill"] = True
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Polyline:
        known = {"kind", "id", "vertices", "closed", "fill", "layer",
                 "color"}
        return cls(id=str(raw["id"]), vertices=_pts(raw["vertices"]),
                   closed=bool(raw.get("closed", False)),
                   fill=bool(raw.get("fill", False)),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.vertices:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class CadText:
    id: str = field(default="", kw_only=True)
    position: Point
    text: str
    height: float = 2.5
    rotation: float = 0.0
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "text", "id": self.id,
               "position": list(self.position), "text": self.text,
               "height": self.height, "rotation": self.rotation,
               "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> CadText:
        known = {"kind", "id", "position", "text", "height",
                 "rotation", "layer", "color"}
        return cls(id=str(raw["id"]), position=_pt(raw["position"]),
                   text=str(raw.get("text", "")),
                   height=_num(raw.get("height", 2.5), "height"),
                   rotation=float(raw.get("rotation", 0.0)),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        w = len(self.text) * self.height * 0.6
        return (self.position[0], self.position[1],
                self.position[0] + w, self.position[1] + self.height)


@dataclass
class Dimension:
    id: str = field(default="", kw_only=True)
    kind: str
    refs: list[GeomRef] = field(default_factory=list)
    value: float = 0.0
    position: Point = (0.0, 0.0)
    text: str = ""
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "dimension", "id": self.id,
               "dim_kind": self.kind,
               "refs": [r.to_dict() for r in self.refs],
               "value": self.value, "position": list(self.position),
               "text": self.text, "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Dimension:
        known = {"kind", "id", "dim_kind", "refs", "value", "position",
                 "text", "layer", "color"}
        return cls(id=str(raw["id"]),
                   kind=str(raw.get("dim_kind", "aligned")),
                   refs=[GeomRef.from_dict(r)
                         for r in raw.get("refs", [])],
                   value=float(raw.get("value", 0.0)),
                   position=_pt(raw.get("position", (0.0, 0.0))),
                   text=str(raw.get("text", "")),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.position[0], self.position[1],
                self.position[0], self.position[1])


@dataclass
class Insert:
    id: str = field(default="", kw_only=True)
    block: str
    position: Point
    rotation: float = 0.0
    scale: Point = (1.0, 1.0)
    layer: str = "0"
    color: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"kind": "insert", "id": self.id, "block": self.block,
               "position": list(self.position),
               "rotation": self.rotation, "scale": list(self.scale),
               "layer": self.layer}
        if self.color:
            out["color"] = self.color
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> Insert:
        known = {"kind", "id", "block", "position", "rotation",
                 "scale", "layer", "color"}
        return cls(id=str(raw["id"]), block=str(raw["block"]),
                   position=_pt(raw["position"]),
                   rotation=float(raw.get("rotation", 0.0)),
                   scale=_pt(raw.get("scale", (1.0, 1.0))),
                   layer=str(raw.get("layer", "0")),
                   color=raw.get("color"),
                   extra={k: v for k, v in raw.items()
                          if k not in known})

    def canonical(self) -> dict:
        return self.to_dict()

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.position[0], self.position[1],
                self.position[0], self.position[1])


@dataclass
class Block:
    name: str
    entities: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name,
                "entities": [e.to_dict() for e in self.entities]}

    @classmethod
    def from_dict(cls, raw: dict) -> Block:
        return cls(name=str(raw["name"]),
                   entities=[entity_from_dict(e)
                             for e in raw.get("entities", [])])


_KINDS = {
    "line": Line, "arc": Arc, "circle": Circle, "ellipse": Ellipse,
    "spline": Spline, "point": CadPoint, "polyline": Polyline,
    "text": CadText, "dimension": Dimension, "insert": Insert,
}


def entity_from_dict(raw: dict):
    kind = str(raw.get("kind", ""))
    cls = _KINDS.get(kind)
    if cls is None:
        raise CadError(f"unknown entity kind {kind!r}", kind="parse")
    return cls.from_dict(raw)


# ── document ──────────────────────────────────────────────────────

@dataclass
class CadDocument:
    title: str = "Untitled"
    units: str = "mm"
    layers: list[Layer] = field(default_factory=list)
    entities: dict[str, object] = field(default_factory=dict)
    blocks: dict[str, Block] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    _dirty_ids: set[str] = field(default_factory=set, repr=False)
    _next_id: int = field(default=1, repr=False)
    _next_constraint: int = field(default=1, repr=False)

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [Layer(name="0")]

    @classmethod
    def blank(cls, title: str = "Untitled") -> CadDocument:
        return cls(title=title)

    def new_id(self, prefix: str = "e") -> str:
        handle = f"{prefix}{self._next_id}"
        self._next_id += 1
        return handle

    def add_entity(self, entity) -> str:
        """Insert an entity, minting an ``e<N>`` id when it has none
        (or when its id is already taken, as for a clone). Returns
        the id it landed under."""
        if not entity.id or entity.id in self.entities:
            entity.id = self.new_id()
        self.entities[entity.id] = entity
        return entity.id

    def next_constraint_id(self) -> int:
        cid = self._next_constraint
        self._next_constraint += 1
        return cid

    def entity_count(self) -> int:
        return len(self.entities)

    def constraint_count(self) -> int:
        return len(self.constraints)

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def layer(self, name: str) -> Layer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def bounds(self) -> tuple[float, float, float, float] | None:
        boxes = [e.bounds() for e in self.entities.values()
                 if hasattr(e, "bounds")]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def fidelity_report(self, original_bytes: bytes) -> dict:
        """Compare this document against ``original_bytes`` at the
        canonical-entity level. ``untouched_changed`` counts changed
        entities the editor never touched — the number that must be
        0 for a fidelity-preserving edit."""
        from .codec import parse_document
        original = parse_document(original_bytes)
        before = {e.id: e.canonical()
                  for e in original.entities.values()}
        after = {e.id: e.canonical() for e in self.entities.values()}
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

    def to_dict(self) -> dict:
        out: dict = {"format_version": 1, "title": self.title,
                     "units": self.units,
                     "layers": [layer.to_dict()
                                for layer in self.layers],
                     "entities": [e.to_dict()
                                  for e in self.entities.values()],
                     "blocks": [b.to_dict()
                                for b in self.blocks.values()],
                     "constraints": [c.to_dict()
                                     for c in self.constraints]}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> CadDocument:
        if raw.get("format_version") != 1:
            raise CadError(
                f"unsupported cad format version "
                f"{raw.get('format_version')!r} (want 1)",
                kind="unsupported")
        known = {"format_version", "title", "units", "layers",
                 "entities", "blocks", "constraints"}
        doc = cls(
            title=str(raw.get("title", "Untitled")),
            units=str(raw.get("units", "mm")),
            layers=[Layer.from_dict(layer)
                    for layer in raw.get("layers", [])],
            blocks={b["name"]: Block.from_dict(b)
                    for b in raw.get("blocks", [])},
            constraints=[Constraint.from_dict(c)
                         for c in raw.get("constraints", [])],
            extra={k: v for k, v in raw.items() if k not in known},
        )
        for entry in raw.get("entities", []):
            entity = entity_from_dict(entry)
            doc.entities[entity.id] = entity
        # Keep id allocation above every id already in use.
        for handle in doc.entities:
            if handle.startswith("e") and handle[1:].isdigit():
                doc._next_id = max(doc._next_id, int(handle[1:]) + 1)
        if doc.constraints:
            doc._next_constraint = max(
                c.id for c in doc.constraints) + 1
        return doc
