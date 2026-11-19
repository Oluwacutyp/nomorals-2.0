"""Easing curves for tweens — port of AniCraft ``crates/doc/src/easing.rs``.

An :class:`Easing` maps linear tween progress ``t in [0, 1]`` to eased
progress. Three kinds, exactly as upstream (ADR 0016, ADR 0081):

* **Presets**: no ease (linear), the Classic Ease slider (strength
  -100 … +100), and nine families, each In, Out and In Out: Quad,
  Cubic, Quart, Quint, Sine, Back, Circ, Bounce and Elastic. The
  formulas are the classic public easing equations (Robert Penner's):
  ``In(t)`` per family, ``Out(t) = 1 - In(1 - t)`` and ``InOut`` as In
  over the first half and Out over the second.
* **One cubic Bezier** (``CubicBezier``, CSS-style handles relative to
  the segment's box).
* **A custom ease** (:class:`CustomEase`): a curve of several Bezier
  segments from (0, 0) to (1, 1). Time never runs backwards (x is
  monotonic along the whole curve); values may overshoot 0..1.
* **A sampled ease** (:class:`SampledEase`, ADR 0088): a table of
  values read back by linear interpolation.

Ground truth: upstream ``easing_tests.rs`` — the 22 preset cases are
encoded in ``animation-parity.toml`` and in the test suite.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from .errors import AnimationError

__all__ = [
    "BACK",
    "CustomEase",
    "Easing",
    "EaseDir",
    "EaseFamily",
    "SampledEase",
    "list_easings",
]

BACK = 1.70158
MAX_EASE_POINTS = 64
MAX_EASE_Y = 10.0

FAMILIES = ("quad", "cubic", "quart", "quint", "sine", "back",
            "circ", "bounce", "elastic")
DIRS = ("in", "out", "in-out")


def _bounce_out(t: float) -> float:
    n = 7.5625
    d = 2.75
    if t < 1.0 / d:
        return n * t * t
    if t < 2.0 / d:
        u = t - 1.5 / d
        return n * u * u + 0.75
    if t < 2.5 / d:
        u = t - 2.25 / d
        return n * u * u + 0.9375
    u = t - 2.625 / d
    return n * u * u + 0.984375


def _ease_in(family: str, t: float) -> float:
    if family == "quad":
        return t * t
    if family == "cubic":
        return t * t * t
    if family == "quart":
        return t * t * t * t
    if family == "quint":
        return t * t * t * t * t
    if family == "sine":
        return 1.0 - math.cos(t * math.pi * 0.5)
    if family == "back":
        return (BACK + 1.0) * t * t * t - BACK * t * t
    if family == "circ":
        return 1.0 - math.sqrt(max(0.0, 1.0 - t * t))
    if family == "bounce":
        return 1.0 - _bounce_out(1.0 - t)
    if family == "elastic":
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return -(2.0 ** (10.0 * t - 10.0)) * math.sin(
            (10.0 * t - 10.75) * (2.0 * math.pi / 3.0))
    raise AnimationError(f"unknown ease family {family!r}")


def _eval_named(family: str, direction: str, t: float) -> float:
    if direction == "in":
        return _ease_in(family, t)
    if direction == "out":
        return 1.0 - _ease_in(family, 1.0 - t)
    # in-out
    if t < 0.5:
        return 0.5 * _ease_in(family, 2.0 * t)
    return 1.0 - 0.5 * _ease_in(family, 2.0 - 2.0 * t)


def _classic_strength(s: float) -> float:
    if not math.isfinite(s):
        return 0.0
    return max(-100.0, min(100.0, s)) / 100.0


def _cubic_bezier(t: float, x1: float, y1: float, x2: float,
                  y2: float) -> float:
    """CSS cubic-bezier: solve x(u) = t for u, then return y(u)."""

    def bx(u: float) -> float:
        return (3.0 * (1.0 - u) ** 2 * u * x1
                + 3.0 * (1.0 - u) * u * u * x2 + u ** 3)

    def by(u: float) -> float:
        return (3.0 * (1.0 - u) ** 2 * u * y1
                + 3.0 * (1.0 - u) * u * u * y2 + u ** 3)

    def bxd(u: float) -> float:
        return (3.0 * (1.0 - u) ** 2 * x1
                + 6.0 * (1.0 - u) * u * (x2 - x1)
                + 3.0 * u * u * (1.0 - x2))

    # Newton iterations from t, then bisection fallback.
    u = t
    for _ in range(8):
        err = bx(u) - t
        if abs(err) < 1e-7:
            return by(u)
        d = bxd(u)
        if abs(d) < 1e-7:
            break
        u -= err / d
        u = max(0.0, min(1.0, u))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bx(mid) < t:
            lo = mid
        else:
            hi = mid
    return by(0.5 * (lo + hi))


class CustomEase:
    """A curve of Bezier segments from (0, 0) to (1, 1).

    Points are ``(x, y)`` anchors; each segment is a cubic Bezier
    whose handles are derived from the neighbouring anchors (a
    Catmull-Rom-to-Bezier conversion), so the curve passes through
    every anchor. ``x`` must be strictly increasing.
    """

    def __init__(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            raise AnimationError("a custom ease needs at least 2 points")
        if len(points) > MAX_EASE_POINTS:
            raise AnimationError(
                f"a custom ease keeps at most {MAX_EASE_POINTS} points")
        xs = [p[0] for p in points]
        if xs[0] != 0.0 or xs[-1] != 1.0:
            raise AnimationError(
                "a custom ease must run from x=0 to x=1")
        if any(b <= a for a, b in itertools.pairwise(xs)):
            raise AnimationError("a custom ease's x must be monotonic")
        if any(abs(y) > MAX_EASE_Y for _, y in points):
            raise AnimationError(
                f"a custom ease's y must stay within +/-{MAX_EASE_Y}")
        self.points = [(float(x), float(y)) for x, y in points]

    def apply(self, t: float) -> float:
        t = 0.0 if not math.isfinite(t) else max(0.0, min(1.0, t))
        pts = self.points
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if t <= x1 or i == len(pts) - 2:
                # Catmull-Rom handles, converted to cubic Bezier.
                x_prev = pts[i - 1][0] if i > 0 else x0 - (x1 - x0)
                y_prev = pts[i - 1][1] if i > 0 else y0 - (y1 - y0)
                x_next = (pts[i + 2][0] if i + 2 < len(pts)
                          else x1 + (x1 - x0))
                y_next = (pts[i + 2][1] if i + 2 < len(pts)
                          else y1 + (y1 - y0))
                c1x = x0 + (x1 - x_prev) / 6.0
                c1y = y0 + (y1 - y_prev) / 6.0
                c2x = x1 - (x_next - x0) / 6.0
                c2y = y1 - (y_next - y0) / 6.0
                # Solve the Bezier's x for u, then read y at that u.
                u = self._solve_x(t, x0, c1x, c2x, x1)
                return self._bezier_y(u, y0, c1y, c2y, y1)
        return pts[-1][1]

    @staticmethod
    def _solve_x(t: float, x0: float, c1: float, c2: float,
                 x1: float) -> float:
        def bx(u: float) -> float:
            return ((1.0 - u) ** 3 * x0 + 3.0 * (1.0 - u) ** 2 * u * c1
                    + 3.0 * (1.0 - u) * u * u * c2 + u ** 3 * x1)

        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if bx(mid) < t:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _bezier_y(u: float, y0: float, c1: float, c2: float,
                  y1: float) -> float:
        return ((1.0 - u) ** 3 * y0 + 3.0 * (1.0 - u) ** 2 * u * c1
                + 3.0 * (1.0 - u) * u * u * c2 + u ** 3 * y1)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "custom",
                "points": [[x, y] for x, y in self.points]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomEase:
        return cls([(float(x), float(y)) for x, y in data["points"]])


class SampledEase:
    """A table of values, read back by linear interpolation."""

    def __init__(self, values: list[float]) -> None:
        if len(values) < 2:
            raise AnimationError("a sampled ease needs at least 2 values")
        self.values = [float(v) for v in values]

    def apply(self, t: float) -> float:
        t = 0.0 if not math.isfinite(t) else max(0.0, min(1.0, t))
        n = len(self.values) - 1
        pos = t * n
        i = min(n - 1, int(pos))
        frac = pos - i
        return self.values[i] * (1.0 - frac) + self.values[i + 1] * frac

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "sampled", "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampledEase:
        return cls([float(v) for v in data["values"]])


class Easing:
    """One easing curve. Immutable; serialises to a plain dict."""

    def __init__(self, kind: str, **kw: Any) -> None:
        self.kind = kind
        self.family = kw.get("family")
        self.direction = kw.get("direction")
        self.strength = kw.get("strength")
        self.bezier = kw.get("bezier")
        self.custom = kw.get("custom")
        self.sampled = kw.get("sampled")

    # -- constructors --------------------------------------------------
    @classmethod
    def named(cls, name: str) -> Easing:
        key = "".join(c for c in name.lower() if c not in "-_ ")
        aliases = {"none": "linear", "noease": "linear",
                   "easein": "quad-in", "easeout": "quad-out",
                   "easeinout": "quad-in-out",
                   "smooth": "bezier:0.25,0.1,0.25,1.0"}
        key = aliases.get(key, key)
        if key == "linear":
            return cls("linear")
        if key.startswith("bezier:"):
            parts = [float(p) for p in key[len("bezier:"):].split(",")]
            return cls("bezier", bezier=tuple(parts))
        for family in FAMILIES:
            for direction in DIRS:
                if key == family + direction.replace("-", ""):
                    return cls("preset", family=family,
                               direction=direction)
        raise AnimationError(
            f"unknown easing {name!r}; pick from {list_easings()}")

    @classmethod
    def classic(cls, strength: float) -> Easing:
        return cls("classic", strength=float(strength))

    @classmethod
    def bezier(cls, x1: float, y1: float, x2: float, y2: float) -> Easing:
        return cls("bezier", bezier=(float(x1), float(y1),
                                     float(x2), float(y2)))

    @classmethod
    def custom(cls, points: list[tuple[float, float]]) -> Easing:
        return cls("custom", custom=CustomEase(points))

    @classmethod
    def sampled(cls, values: list[float]) -> Easing:
        return cls("sampled", sampled=SampledEase(values))

    # -- evaluation ----------------------------------------------------
    def apply(self, t: float) -> float:
        t = 0.0 if not math.isfinite(t) else max(0.0, min(1.0, t))
        if self.kind == "linear":
            return t
        if self.kind == "classic":
            c = 0.5 + 0.5 * _classic_strength(self.strength)
            return 2.0 * (1.0 - t) * t * c + t * t
        if self.kind == "bezier":
            x1, y1, x2, y2 = self.bezier
            return _cubic_bezier(t, x1, y1, x2, y2)
        if self.kind == "custom":
            return self.custom.apply(t)
        if self.kind == "sampled":
            return self.sampled.apply(t)
        return _eval_named(self.family, self.direction, t)

    # -- naming --------------------------------------------------------
    def key(self) -> str:
        if self.kind == "linear":
            return "linear"
        if self.kind == "classic":
            return f"classic:{self.strength:+.0f}"
        if self.kind == "bezier":
            return "bezier:" + ",".join(f"{v:g}" for v in self.bezier)
        if self.kind == "custom":
            return "custom"
        if self.kind == "sampled":
            return "sampled"
        return f"{self.family}-{self.direction}"

    def label(self) -> str:
        if self.kind == "linear":
            return "Linear"
        if self.kind == "classic":
            return f"Classic {self.strength:+.0f}"
        if self.kind == "bezier":
            if self.bezier == (0.25, 0.1, 0.25, 1.0):
                return "Smooth (bezier)"
            return ("Bezier(" + ", ".join(f"{v:.2f}" for v in self.bezier)
                    + ")")
        if self.kind == "custom":
            return "Custom"
        if self.kind == "sampled":
            return "Sampled"
        direction = {"in": "In", "out": "Out",
                     "in-out": "In Out"}[self.direction]
        return f"{self.family.capitalize()} {direction}"

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if self.kind == "preset":
            return {"kind": "preset", "family": self.family,
                    "direction": self.direction}
        if self.kind == "classic":
            return {"kind": "classic", "strength": self.strength}
        if self.kind == "bezier":
            return {"kind": "bezier", "handles": list(self.bezier)}
        if self.kind == "custom":
            return self.custom.to_dict()
        if self.kind == "sampled":
            return self.sampled.to_dict()
        return {"kind": "linear"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Easing:
        kind = data.get("kind", "linear")
        if kind == "preset":
            return cls("preset", family=data["family"],
                       direction=data["direction"])
        if kind == "classic":
            return cls("classic", strength=float(data["strength"]))
        if kind == "bezier":
            h = data["handles"]
            return cls("bezier", bezier=(float(h[0]), float(h[1]),
                                         float(h[2]), float(h[3])))
        if kind == "custom":
            return cls("custom", custom=CustomEase.from_dict(data))
        if kind == "sampled":
            return cls("sampled", sampled=SampledEase.from_dict(data))
        return cls("linear")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Easing) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Easing({self.key()})"


class EaseFamily:
    """Namespace for the nine Penner families."""

    ALL = FAMILIES


class EaseDir:
    """Namespace for the three ease directions."""

    ALL = DIRS


def list_easings() -> list[str]:
    out = ["linear"]
    for family in FAMILIES:
        for direction in DIRS:
            out.append(f"{family}-{direction}")
    return out
