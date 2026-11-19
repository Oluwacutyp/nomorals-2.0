"""2D geometry — port of the AniCraft ``geom`` crate's essentials.

``Vec2`` and ``Affine`` follow ``crates/geom/src/affine.rs`` (a 2D
affine matrix ``[a b c d e f]`` mapping ``(x, y)`` to
``(a*x + c*y + e, b*x + d*y + f)``); :class:`Path` follows
``crates/geom/src/path.rs`` — move/line/quad/cubic/close segments,
flattened to polylines for the renderer and the shape tween.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["Affine", "Path", "Vec2"]


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> Vec2:
        return Vec2(self.x * s, self.y * s)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Vec2) and self.x == other.x
                and self.y == other.y)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Vec2({self.x:g}, {self.y:g})"

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def angle(self) -> float:
        return math.degrees(math.atan2(self.y, self.x))


class Affine:
    """A 2D affine transform, upstream's ``[a b c d e f]`` layout."""

    __slots__ = ("a", "b", "c", "d", "e", "f")

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 0.0,
                 d: float = 1.0, e: float = 0.0, f: float = 0.0) -> None:
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.d, self.e, self.f = float(d), float(e), float(f)

    @classmethod
    def identity(cls) -> Affine:
        return cls()

    @classmethod
    def translate(cls, x: float, y: float) -> Affine:
        return cls(1.0, 0.0, 0.0, 1.0, x, y)

    @classmethod
    def scale(cls, sx: float, sy: float) -> Affine:
        return cls(sx, 0.0, 0.0, sy, 0.0, 0.0)

    @classmethod
    def rotate(cls, degrees: float) -> Affine:
        r = math.radians(degrees)
        cos, sin = math.cos(r), math.sin(r)
        return cls(cos, sin, -sin, cos, 0.0, 0.0)

    @classmethod
    def skew(cls, skew_x: float, skew_y: float) -> Affine:
        return cls(1.0, math.tan(math.radians(skew_y)),
                   math.tan(math.radians(skew_x)), 1.0, 0.0, 0.0)

    def apply(self, p: Vec2) -> Vec2:
        return Vec2(self.a * p.x + self.c * p.y + self.e,
                    self.b * p.x + self.d * p.y + self.f)

    def compose(self, other: Affine) -> Affine:
        """``self`` after ``other``: apply ``other`` first, then ``self``."""
        return Affine(
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f)

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c,
                "d": self.d, "e": self.e, "f": self.f}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Affine:
        return cls(data.get("a", 1.0), data.get("b", 0.0),
                   data.get("c", 0.0), data.get("d", 1.0),
                   data.get("e", 0.0), data.get("f", 0.0))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Affine)
                and (self.a, self.b, self.c, self.d, self.e, self.f)
                == (other.a, other.b, other.c, other.d, other.e, other.f))


class Path:
    """A vector path: move/line/quad/cubic/close segments.

    Segments are ``(op, points)`` with ``op`` in
    ``move``/``line``/``quad``/``cubic``/``close``. A path is one
    subpath list; :meth:`flatten` turns it into polylines (one per
    closed contour) for rendering and shape-tween morphing.
    """

    def __init__(self, segments: list | None = None) -> None:
        self.segments: list[tuple[str, list[Vec2]]] = list(segments or [])

    @classmethod
    def rect(cls, x: float, y: float, w: float, h: float) -> Path:
        p = cls()
        p.move_to(x, y)
        p.line_to(x + w, y)
        p.line_to(x + w, y + h)
        p.line_to(x, y + h)
        p.close()
        return p

    @classmethod
    def ellipse(cls, cx: float, cy: float, rx: float, ry: float,
                steps: int = 48) -> Path:
        p = cls()
        for i in range(steps):
            a = 2.0 * math.pi * i / steps
            x = cx + rx * math.cos(a)
            y = cy + ry * math.sin(a)
            if i == 0:
                p.move_to(x, y)
            else:
                p.line_to(x, y)
        p.close()
        return p

    @classmethod
    def from_svg(cls, data: str) -> Path:
        """Parse SVG path data (M/L/H/V/Q/C/Z, absolute + relative)."""
        from .errors import AnimationError

        tokens: list[str] = []
        cur = ""
        for ch in data.strip():
            if ch in "MLHVQCTZmlhvqctz":
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append(ch)
            elif ch in ", \t\n\r":
                if cur:
                    tokens.append(cur)
                    cur = ""
            else:
                cur += ch
        if cur:
            tokens.append(cur)
        path = cls()
        i = 0
        x = y = 0.0
        start_x = start_y = 0.0
        cmd = ""
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cmd = tok
                i += 1
                if cmd in "Zz":
                    path.close()
                    x, y = start_x, start_y
                continue
            if cmd in "Mm":
                nx = float(tokens[i]) + (x if cmd == "m" else 0.0)
                ny = float(tokens[i + 1]) + (y if cmd == "m" else 0.0)
                i += 2
                if cmd == "M":
                    path.move_to(nx, ny)
                else:
                    path.move_to(nx, ny)
                x, y = nx, ny
                start_x, start_y = x, y
                cmd = "L" if cmd == "M" else "l"
            elif cmd in "Ll":
                nx = float(tokens[i]) + (x if cmd == "l" else 0.0)
                ny = float(tokens[i + 1]) + (y if cmd == "l" else 0.0)
                i += 2
                path.line_to(nx, ny)
                x, y = nx, ny
            elif cmd in "Hh":
                nx = float(tokens[i]) + (x if cmd == "h" else 0.0)
                i += 1
                path.line_to(nx, y)
                x = nx
            elif cmd in "Vv":
                ny = float(tokens[i]) + (y if cmd == "v" else 0.0)
                i += 1
                path.line_to(x, ny)
                y = ny
            elif cmd in "Qq":
                cx = float(tokens[i]) + (x if cmd == "q" else 0.0)
                cy = float(tokens[i + 1]) + (y if cmd == "q" else 0.0)
                nx = float(tokens[i + 2]) + (x if cmd == "q" else 0.0)
                ny = float(tokens[i + 3]) + (y if cmd == "q" else 0.0)
                i += 4
                path.quad_to(cx, cy, nx, ny)
                x, y = nx, ny
            elif cmd in "Cc":
                c1x = float(tokens[i]) + (x if cmd == "c" else 0.0)
                c1y = float(tokens[i + 1]) + (y if cmd == "c" else 0.0)
                c2x = float(tokens[i + 2]) + (x if cmd == "c" else 0.0)
                c2y = float(tokens[i + 3]) + (y if cmd == "c" else 0.0)
                nx = float(tokens[i + 4]) + (x if cmd == "c" else 0.0)
                ny = float(tokens[i + 5]) + (y if cmd == "c" else 0.0)
                i += 6
                path.cubic_to(c1x, c1y, c2x, c2y, nx, ny)
                x, y = nx, ny
            else:
                raise AnimationError(f"unsupported SVG path token {tok!r}")
        return path

    def move_to(self, x: float, y: float) -> None:
        self.segments.append(("move", [Vec2(x, y)]))

    def line_to(self, x: float, y: float) -> None:
        self.segments.append(("line", [Vec2(x, y)]))

    def quad_to(self, cx: float, cy: float, x: float, y: float) -> None:
        self.segments.append(("quad", [Vec2(cx, cy), Vec2(x, y)]))

    def cubic_to(self, c1x: float, c1y: float, c2x: float, c2y: float,
                 x: float, y: float) -> None:
        self.segments.append(("cubic", [Vec2(c1x, c1y), Vec2(c2x, c2y),
                                        Vec2(x, y)]))

    def close(self) -> None:
        self.segments.append(("close", []))

    def flatten(self, steps: int = 16) -> list[list[tuple[float, float]]]:
        """Polylines, one per contour; curves subdivided evenly."""
        contours: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        cur = Vec2(0.0, 0.0)
        for op, pts in self.segments:
            if op == "move":
                if current:
                    contours.append(current)
                current = [pts[0].to_tuple()]
                cur = pts[0]
            elif op == "line":
                current.append(pts[0].to_tuple())
                cur = pts[0]
            elif op == "quad":
                c, end = pts
                for k in range(1, steps + 1):
                    u = k / steps
                    x = ((1 - u) ** 2 * cur.x + 2 * (1 - u) * u * c.x
                         + u * u * end.x)
                    y = ((1 - u) ** 2 * cur.y + 2 * (1 - u) * u * c.y
                         + u * u * end.y)
                    current.append((x, y))
                cur = end
            elif op == "cubic":
                c1, c2, end = pts
                for k in range(1, steps + 1):
                    u = k / steps
                    x = ((1 - u) ** 3 * cur.x
                         + 3 * (1 - u) ** 2 * u * c1.x
                         + 3 * (1 - u) * u * u * c2.x + u ** 3 * end.x)
                    y = ((1 - u) ** 3 * cur.y
                         + 3 * (1 - u) ** 2 * u * c1.y
                         + 3 * (1 - u) * u * u * c2.y + u ** 3 * end.y)
                    current.append((x, y))
                cur = end
            elif (op == "close" and current
                    and current[0] != current[-1]):
                current.append(current[0])
        if current:
            contours.append(current)
        return contours

    def bounds(self) -> tuple[float, float, float, float]:
        pts = [p for contour in self.flatten() for p in contour]
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_svg(self) -> str:
        parts: list[str] = []
        for op, pts in self.segments:
            if op == "move":
                parts.append(f"M{pts[0].x:g},{pts[0].y:g}")
            elif op == "line":
                parts.append(f"L{pts[0].x:g},{pts[0].y:g}")
            elif op == "quad":
                parts.append(f"Q{pts[0].x:g},{pts[0].y:g} "
                             f"{pts[1].x:g},{pts[1].y:g}")
            elif op == "cubic":
                parts.append(f"C{pts[0].x:g},{pts[0].y:g} "
                             f"{pts[1].x:g},{pts[1].y:g} "
                             f"{pts[2].x:g},{pts[2].y:g}")
            elif op == "close":
                parts.append("Z")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [
            [op, [[p.x, p.y] for p in pts]]
            for op, pts in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Path:
        path = cls()
        for op, pts in data["segments"]:
            path.segments.append(
                (op, [Vec2(float(x), float(y)) for x, y in pts]))
        return path

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Path) and self.to_dict() == other.to_dict()
