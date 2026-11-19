"""Geometry — points, affine transforms, paths.

Port of the *design* of the upstream ``crates/geom`` (anchor-based
editable path model over kurbo's ``BezPath``/``CubicBez``) into
Devon's Python, stdlib only. A :class:`Path` is a list of segments —
move / line / cubic / quad / arc / close — with the full SVG
``d`` grammar (absolute + relative, implicit repeats, smooth
commands), adaptive bezier flattening, arc endpoint-to-center
conversion per the SVG spec, bounds, length, reversal and
Ramer–Douglas–Peucker simplification.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import pairwise

__all__ = [
    "Affine", "ArcTo", "ClosePath", "CubicTo", "LineTo", "MoveTo",
    "Path", "QuadTo", "Segment", "arc_to_cubic", "parse_path_data",
    "parse_transform",
]

#: A 2D point as (x, y).
Point = tuple[float, float]

_EPS = 1e-9


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ── affine transforms ───────────────────────────────────────────────

def format_float(value: float) -> str:
    """Shortest text that round-trips a float exactly.

    ``f"{value:g}"`` keeps only 6 significant digits, so a matrix
    entry like -41.421356237 became "-41.4214" on write and the
    re-parsed document differed at the 6th decimal — a round trip
    that never settles. ``repr`` is exact by construction;
    integral values lose the trailing ".0" for readability.
    """
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text


@dataclass(frozen=True)
class Affine:
    """2D affine matrix in SVG order: x' = a*x + c*y + e,
    y' = b*x + d*y + f. ``A * B`` applies B first, then A."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    @classmethod
    def identity(cls) -> Affine:
        return cls()

    @classmethod
    def translate(cls, tx: float, ty: float = 0.0) -> Affine:
        return cls(e=tx, f=ty)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine:
        return cls(a=sx, d=sy if sy is not None else sx)

    @classmethod
    def rotate(cls, degrees: float, cx: float = 0.0,
               cy: float = 0.0) -> Affine:
        rad = math.radians(degrees)
        cos = math.cos(rad)
        sin = math.sin(rad)
        rot = cls(a=cos, b=sin, c=-sin, d=cos)
        if cx == 0.0 and cy == 0.0:
            return rot
        return cls.translate(cx, cy) * rot * cls.translate(-cx, -cy)

    @classmethod
    def skew_x(cls, degrees: float) -> Affine:
        return cls(c=math.tan(math.radians(degrees)))

    @classmethod
    def skew_y(cls, degrees: float) -> Affine:
        return cls(b=math.tan(math.radians(degrees)))

    def __mul__(self, other: Affine) -> Affine:
        return Affine(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f)

    def apply(self, point: Point) -> Point:
        x, y = point
        return (self.a * x + self.c * y + self.e,
                self.b * x + self.d * y + self.f)

    def is_identity(self) -> bool:
        return (self.a, self.b, self.c, self.d, self.e, self.f) \
            == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def to_svg(self) -> str:
        return (f"matrix({format_float(self.a)} "
                f"{format_float(self.b)} {format_float(self.c)} "
                f"{format_float(self.d)} {format_float(self.e)} "
                f"{format_float(self.f)})")


_TRANSFORM_RE = re.compile(
    r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(value: str) -> Affine:
    """Parse an SVG ``transform`` attribute into one Affine."""
    result = Affine.identity()
    for match in _TRANSFORM_RE.finditer(value or ""):
        name = match.group(1)
        nums = [float(n) for n in
                re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",
                           match.group(2))]
        if name == "matrix" and len(nums) == 6:
            step = Affine(*nums)
        elif name == "translate" and nums:
            step = Affine.translate(nums[0],
                                    nums[1] if len(nums) > 1 else 0.0)
        elif name == "scale" and nums:
            step = Affine.scale(nums[0],
                                nums[1] if len(nums) > 1 else None)
        elif name == "rotate" and nums:
            step = Affine.rotate(nums[0],
                                 nums[1] if len(nums) > 1 else 0.0,
                                 nums[2] if len(nums) > 2 else 0.0)
        elif name == "skewX" and nums:
            step = Affine.skew_x(nums[0])
        elif name == "skewY" and nums:
            step = Affine.skew_y(nums[0])
        else:
            continue
        result = result * step
    return result


# ── path segments ───────────────────────────────────────────────────

@dataclass
class MoveTo:
    kind: str = field(default="M", init=False)
    p: Point = (0.0, 0.0)


@dataclass
class LineTo:
    kind: str = field(default="L", init=False)
    p: Point = (0.0, 0.0)


@dataclass
class CubicTo:
    kind: str = field(default="C", init=False)
    c1: Point = (0.0, 0.0)
    c2: Point = (0.0, 0.0)
    p: Point = (0.0, 0.0)


@dataclass
class QuadTo:
    kind: str = field(default="Q", init=False)
    c: Point = (0.0, 0.0)
    p: Point = (0.0, 0.0)


@dataclass
class ArcTo:
    kind: str = field(default="A", init=False)
    rx: float = 0.0
    ry: float = 0.0
    rotation: float = 0.0
    large_arc: int = 0
    sweep: int = 0
    p: Point = (0.0, 0.0)


@dataclass
class ClosePath:
    kind: str = field(default="Z", init=False)


Segment = MoveTo | LineTo | CubicTo | QuadTo | ArcTo | ClosePath


def arc_to_cubic(start: Point, rx: float, ry: float, rotation: float,
                 large_arc: int, sweep: int,
                 end: Point) -> list[tuple[Point, Point, Point]]:
    """Convert an SVG elliptical arc to cubic bezier pieces.

    Endpoint-to-center parameterization per SVG 1.1 appendix F.6.5.
    Returns a list of (c1, c2, end) triples; empty when the arc is
    degenerate (zero radii → the caller treats it as a line).
    """
    if rx == 0.0 or ry == 0.0 or start == end:
        return []
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rotation % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx = (start[0] - end[0]) / 2.0
    dy = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale
    num = rx * rx * ry * ry - rx * rx * y1p * y1p \
        - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) if den != 0.0 else 0.0
    if large_arc == sweep:
        co = -co
    cxp = co * rx * y1p / ry
    cyp = -co * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        ang = math.acos(max(-1.0, min(1.0, dot / length))) \
            if length else 0.0
        if ux * vy - uy * vx < 0:
            ang = -ang
        return ang

    theta1 = _angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                    (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2.0 * math.pi
    if sweep and dtheta < 0:
        dtheta += 2.0 * math.pi
    segments = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2.0))))
    step = dtheta / segments
    out: list[tuple[Point, Point, Point]] = []
    for i in range(segments):
        t0 = theta1 + i * step
        t1 = t0 + step
        alpha = math.sin(step) * (math.sqrt(4.0 + 3.0 * math.tan(
            step / 2.0) ** 2) - 1.0) / 3.0

        def _pt(t: float) -> Point:
            return (cx + rx * math.cos(t) * cos_phi
                    - ry * math.sin(t) * sin_phi,
                    cy + rx * math.cos(t) * sin_phi
                    + ry * math.sin(t) * cos_phi)

        def _der(t: float) -> Point:
            return (-rx * math.sin(t) * cos_phi
                    - ry * math.cos(t) * sin_phi,
                    -rx * math.sin(t) * sin_phi
                    + ry * math.cos(t) * cos_phi)

        p0, p1 = _pt(t0), _pt(t1)
        d0, d1 = _der(t0), _der(t1)
        c1 = (p0[0] + alpha * d0[0], p0[1] + alpha * d0[1])
        c2 = (p1[0] - alpha * d1[0], p1[1] - alpha * d1[1])
        out.append((c1, c2, p1))
    return out


def _cubic_flat(p0: Point, c1: Point, c2: Point, p1: Point,
                tolerance: float, depth: int = 0) -> list[Point]:
    """Adaptive subdivision; returns interior points + end point."""
    if depth > 20:
        return [p1]
    d1 = abs((p1[0] - p0[0]) * (p0[1] - c1[1])
             - (p1[1] - p0[1]) * (p0[0] - c1[0]))
    d2 = abs((p1[0] - p0[0]) * (p0[1] - c2[1])
             - (p1[1] - p0[1]) * (p0[0] - c2[0]))
    chord = _dist(p0, p1) or 1.0
    if max(d1, d2) / chord <= tolerance:
        return [p1]
    m01 = _lerp(p0, c1, 0.5)
    m12 = _lerp(c1, c2, 0.5)
    m23 = _lerp(c2, p1, 0.5)
    m012 = _lerp(m01, m12, 0.5)
    m123 = _lerp(m12, m23, 0.5)
    mid = _lerp(m012, m123, 0.5)
    return (_cubic_flat(p0, m01, m012, mid, tolerance, depth + 1)
            + _cubic_flat(mid, m123, m23, p1, tolerance, depth + 1))


def _quad_flat(p0: Point, c: Point, p1: Point,
               tolerance: float, depth: int = 0) -> list[Point]:
    if depth > 20:
        return [p1]
    d = abs((p1[0] - p0[0]) * (p0[1] - c[1])
            - (p1[1] - p0[1]) * (p0[0] - c[0]))
    chord = _dist(p0, p1) or 1.0
    if d / chord <= tolerance:
        return [p1]
    m01 = _lerp(p0, c, 0.5)
    m12 = _lerp(c, p1, 0.5)
    mid = _lerp(m01, m12, 0.5)
    return (_quad_flat(p0, m01, mid, tolerance, depth + 1)
            + _quad_flat(mid, m12, p1, tolerance, depth + 1))


_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|"
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_path_data(data: str) -> Path:
    """Parse an SVG ``d`` attribute into a :class:`Path`.

    Handles absolute + relative commands, implicit repeats (extra
    coordinate groups repeat the command; extra moveto pairs become
    lineto), smooth ``S``/``T`` reflection, and arc flags.
    """
    tokens = _TOKEN_RE.findall(data or "")
    segments: list[Segment] = []
    current: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    command: str | None = None
    prev_cubic: Point | None = None
    prev_quad: Point | None = None
    index = 0

    def _num() -> float:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("path data ends mid-command")
        token = tokens[index]
        index += 1
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(
                f"expected a number in path data, got {token!r}"
                ) from exc

    def _rel(point: Point, relative: bool) -> Point:
        if not relative:
            return point
        return (current[0] + point[0], current[1] + point[1])

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in ("Z", "z"):
                segments.append(ClosePath())
                current = start
                prev_cubic = prev_quad = None
                command = None
                continue
        elif command is None:
            raise ValueError(
                f"path data starts with a number {token!r}")
        letter = command.upper()
        relative = command.islower()
        if letter == "M":
            x, y = _num(), _num()
            point = _rel((x, y), relative)
            segments.append(MoveTo(point))
            current = start = point
            prev_cubic = prev_quad = None
            command = "l" if relative else "L"
        elif letter == "L":
            x, y = _num(), _num()
            point = _rel((x, y), relative)
            segments.append(LineTo(point))
            current = point
            prev_cubic = prev_quad = None
        elif letter == "H":
            x = _num()
            point = (current[0] + x, current[1]) if relative \
                else (x, current[1])
            segments.append(LineTo(point))
            current = point
            prev_cubic = prev_quad = None
        elif letter == "V":
            y = _num()
            point = (current[0], current[1] + y) if relative \
                else (current[0], y)
            segments.append(LineTo(point))
            current = point
            prev_cubic = prev_quad = None
        elif letter == "C":
            x1, y1 = _num(), _num()
            x2, y2 = _num(), _num()
            x, y = _num(), _num()
            c1 = _rel((x1, y1), relative)
            c2 = _rel((x2, y2), relative)
            point = _rel((x, y), relative)
            segments.append(CubicTo(c1, c2, point))
            prev_cubic = c2
            prev_quad = None
            current = point
        elif letter == "S":
            x2, y2 = _num(), _num()
            x, y = _num(), _num()
            if prev_cubic is not None:
                c1 = (2 * current[0] - prev_cubic[0],
                      2 * current[1] - prev_cubic[1])
            else:
                c1 = current
            c2 = _rel((x2, y2), relative)
            point = _rel((x, y), relative)
            segments.append(CubicTo(c1, c2, point))
            prev_cubic = c2
            prev_quad = None
            current = point
        elif letter == "Q":
            x1, y1 = _num(), _num()
            x, y = _num(), _num()
            ctrl = _rel((x1, y1), relative)
            point = _rel((x, y), relative)
            segments.append(QuadTo(ctrl, point))
            prev_quad = ctrl
            prev_cubic = None
            current = point
        elif letter == "T":
            x, y = _num(), _num()
            if prev_quad is not None:
                ctrl = (2 * current[0] - prev_quad[0],
                        2 * current[1] - prev_quad[1])
            else:
                ctrl = current
            point = _rel((x, y), relative)
            segments.append(QuadTo(ctrl, point))
            prev_quad = ctrl
            prev_cubic = None
            current = point
        elif letter == "A":
            rx, ry = _num(), _num()
            rotation = _num()
            large = int(_num())
            sweep = int(_num())
            x, y = _num(), _num()
            point = _rel((x, y), relative)
            segments.append(ArcTo(rx, ry, rotation, large, sweep,
                                  point))
            prev_cubic = prev_quad = None
            current = point
        else:
            raise ValueError(f"unknown path command {command!r}")
    return Path(segments)


# ── Path ────────────────────────────────────────────────────────────

@dataclass
class Path:
    """An ordered list of segments; one or more subpaths."""

    segments: list[Segment] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def start_point(self) -> Point | None:
        for seg in self.segments:
            if isinstance(seg, MoveTo):
                return seg.p
        return None

    def to_d(self) -> str:
        parts: list[str] = []
        for seg in self.segments:
            if isinstance(seg, MoveTo):
                parts.append(f"M {format_float(seg.p[0])} "
                             f"{format_float(seg.p[1])}")
            elif isinstance(seg, LineTo):
                parts.append(f"L {format_float(seg.p[0])} "
                             f"{format_float(seg.p[1])}")
            elif isinstance(seg, CubicTo):
                parts.append(f"C {format_float(seg.c1[0])} "
                             f"{format_float(seg.c1[1])}, "
                             f"{format_float(seg.c2[0])} "
                             f"{format_float(seg.c2[1])}, "
                             f"{format_float(seg.p[0])} "
                             f"{format_float(seg.p[1])}")
            elif isinstance(seg, QuadTo):
                parts.append(f"Q {format_float(seg.c[0])} "
                             f"{format_float(seg.c[1])}, "
                             f"{format_float(seg.p[0])} "
                             f"{format_float(seg.p[1])}")
            elif isinstance(seg, ArcTo):
                parts.append(f"A {format_float(seg.rx)} "
                             f"{format_float(seg.ry)} "
                             f"{format_float(seg.rotation)} "
                             f"{seg.large_arc} "
                             f"{seg.sweep} {format_float(seg.p[0])} "
                             f"{format_float(seg.p[1])}")
            elif isinstance(seg, ClosePath):
                parts.append("Z")
        return " ".join(parts)

    def contours(self) -> list[list[Segment]]:
        """Split into subpaths (each starting at a MoveTo)."""
        out: list[list[Segment]] = []
        current: list[Segment] = []
        for seg in self.segments:
            if isinstance(seg, MoveTo) and current:
                out.append(current)
                current = []
            current.append(seg)
        if current:
            out.append(current)
        return out

    def flatten(self, tolerance: float = 0.5) -> list[list[Point]]:
        """Flatten every subpath to a polyline contour.

        Closed subpaths repeat their first point at the end; open
        subpaths do not. Arcs become cubics first.
        """
        result: list[list[Point]] = []
        for contour in self.contours():
            points: list[Point] = []
            current: Point = (0.0, 0.0)
            sub_start: Point = (0.0, 0.0)
            closed = False
            for seg in contour:
                if isinstance(seg, MoveTo):
                    current = sub_start = seg.p
                    points = [seg.p]
                elif isinstance(seg, LineTo):
                    points.append(seg.p)
                    current = seg.p
                elif isinstance(seg, CubicTo):
                    points.extend(_cubic_flat(current, seg.c1, seg.c2,
                                              seg.p, tolerance))
                    current = seg.p
                elif isinstance(seg, QuadTo):
                    points.extend(_quad_flat(current, seg.c, seg.p,
                                             tolerance))
                    current = seg.p
                elif isinstance(seg, ArcTo):
                    curves = arc_to_cubic(current, seg.rx, seg.ry,
                                          seg.rotation, seg.large_arc,
                                          seg.sweep, seg.p)
                    if not curves:
                        points.append(seg.p)
                    else:
                        pen = current
                        for c1, c2, end in curves:
                            points.extend(_cubic_flat(pen, c1, c2, end,
                                                     tolerance))
                            pen = end
                    current = seg.p
                elif isinstance(seg, ClosePath):
                    if points and points[-1] != sub_start:
                        points.append(sub_start)
                    closed = True
                    current = sub_start
            if points:
                if closed and points[-1] != points[0]:
                    points.append(points[0])
                result.append(points)
        return result

    def bounds(self) -> tuple[Point, Point]:
        """((min x, min y), (max x, max y)) over flattened points."""
        contours = self.flatten(0.5)
        if not contours or not contours[0]:
            return ((0.0, 0.0), (0.0, 0.0))
        xs = [p[0] for c in contours for p in c]
        ys = [p[1] for c in contours for p in c]
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    def length(self) -> float:
        total = 0.0
        for contour in self.flatten(0.25):
            for a, b in pairwise(contour):
                total += _dist(a, b)
        return total

    def transform(self, affine: Affine) -> Path:
        def _map(p: Point) -> Point:
            return affine.apply(p)

        out: list[Segment] = []
        pen: Point = (0.0, 0.0)
        for seg in self.segments:
            if isinstance(seg, MoveTo):
                out.append(MoveTo(_map(seg.p)))
                pen = seg.p
            elif isinstance(seg, LineTo):
                out.append(LineTo(_map(seg.p)))
                pen = seg.p
            elif isinstance(seg, CubicTo):
                out.append(CubicTo(_map(seg.c1), _map(seg.c2),
                                   _map(seg.p)))
                pen = seg.p
            elif isinstance(seg, QuadTo):
                out.append(QuadTo(_map(seg.c), _map(seg.p)))
                pen = seg.p
            elif isinstance(seg, ArcTo):
                # Arcs under a general affine stop being circular;
                # expand to cubics first, then map.
                curves = arc_to_cubic(pen, seg.rx, seg.ry,
                                      seg.rotation, seg.large_arc,
                                      seg.sweep, seg.p)
                if not curves:
                    out.append(LineTo(_map(seg.p)))
                else:
                    for c1, c2, end in curves:
                        out.append(CubicTo(_map(c1), _map(c2),
                                           _map(end)))
                pen = seg.p
            elif isinstance(seg, ClosePath):
                out.append(ClosePath())
        return Path(out)

    def reverse(self) -> Path:
        """Reverse every contour; signed area flips sign."""
        out: list[Segment] = []
        for contour in self.contours():
            out.extend(_reverse_contour(contour))
        return Path(out)

    def simplify(self, tolerance: float = 0.5) -> Path:
        """Ramer–Douglas–Peucker on the flattened contours."""
        out: list[Segment] = []
        for contour in self.flatten(tolerance):
            if len(contour) <= 2:
                if contour:
                    out.append(MoveTo(contour[0]))
                    for p in contour[1:]:
                        out.append(LineTo(p))
                continue
            kept = _rdp(contour, tolerance)
            closed = kept[0] == kept[-1]
            body = kept[:-1] if closed else kept
            out.append(MoveTo(body[0]))
            for p in body[1:]:
                out.append(LineTo(p))
            if closed:
                out.append(ClosePath())
        return Path(out)


def _reverse_contour(contour: list[Segment]) -> list[Segment]:
    """Reverse one contour's segments, preserving curve shapes."""
    # Collect the endpoint chain: start + each segment's end.
    points: list[Point] = []
    for seg in contour:
        if isinstance(seg, MoveTo):
            points = [seg.p]
        elif isinstance(seg, (LineTo, CubicTo, QuadTo, ArcTo)):
            points.append(seg.p)
    if len(points) < 2:
        return list(contour)
    closed = isinstance(contour[-1], ClosePath)
    # Segment i (1-based in points) runs points[i-1] -> points[i].
    # Reversed segment j runs points[n-j] -> points[n-j-1] with the
    # same shape: swap control points for cubics/quads, keep arcs
    # (sweep flips).
    segs = [s for s in contour if not isinstance(s, (MoveTo,
                                                    ClosePath))]
    out: list[Segment] = [MoveTo(points[-1])]
    for j, seg in enumerate(reversed(segs)):
        if isinstance(seg, LineTo):
            out.append(LineTo(points[len(points) - 2 - j]))
        elif isinstance(seg, CubicTo):
            out.append(CubicTo(seg.c2, seg.c1,
                               points[len(points) - 2 - j]))
        elif isinstance(seg, QuadTo):
            out.append(QuadTo(seg.c,
                              points[len(points) - 2 - j]))
        elif isinstance(seg, ArcTo):
            out.append(ArcTo(seg.rx, seg.ry, seg.rotation,
                             seg.large_arc, 1 - seg.sweep,
                             points[len(points) - 2 - j]))
    if closed:
        out.append(ClosePath())
    return out


def _rdp(points: list[Point], tolerance: float) -> list[Point]:
    if len(points) <= 2:
        return list(points)
    first, last = points[0], points[-1]
    dx, dy = last[0] - first[0], last[1] - first[1]
    denom = math.hypot(dx, dy) or 1.0
    worst = 0.0
    worst_i = 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + last[0] * first[1]
                - last[1] * first[0]) / denom
        if d > worst:
            worst, worst_i = d, i
    if worst <= tolerance:
        return [first, last]
    left = _rdp(points[:worst_i + 1], tolerance)
    right = _rdp(points[worst_i:], tolerance)
    return left[:-1] + right
