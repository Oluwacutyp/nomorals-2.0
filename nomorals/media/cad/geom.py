"""Geometry — points, intersections, fillets, splines.

Port of the *design* of upstream ``crates/geom`` (``vec.rs``,
``curve.rs``, ``intersect.rs``, ``fillet.rs``, ``spline.rs``,
``spline_fit.rs``, ``nurbs.rs``) into Devon's Python, stdlib only.

A point is a ``(x, y)`` tuple. The intersection routines keep
upstream's numerical habits: line–circle measures from the foot
of the perpendicular (not from a discriminant of two O(length⁴)
numbers, which loses the gap on a long line nearly tangent to a
small circle), circle–circle works at the scale of the two
circles and judges tangency against the smaller one. The fillet
follows upstream's ``fillet.rs``: the centre of a radius-``r``
circle tangent to two curves lies on an offset of each at
distance ``r`` along the normal; sampled offset curves give
candidates, Newton's method on the exact offset curves refines
them. Splines are cubic B-splines evaluated by de Boor (control
form) and Catmull–Rom interpolation through fit points (the
fit form); upstream's full NURBS rational weights are not
modelled — a weighted spline reads back as its unweighted
control form and the loss is reported by the codec, never
hidden.
"""

from __future__ import annotations

import math

__all__ = [
    "Arc", "add", "angle_of", "arc_contains", "arc_point", "ccw_sweep",
    "circle_circle", "corner_fit", "cross", "dist", "dot", "ellipse_at",
    "ellipse_derivative", "ellipse_normal", "ellipse_on", "fillet_candidates",
    "line_circle", "line_line", "line_line_infinite", "mid", "norm_angle",
    "normalize", "perp", "rotate", "scale", "spline_derivative",
    "spline_eval", "sub", "tangent_points",
]

#: A 2D point as (x, y).
Point = tuple[float, float]

_EPS = 1e-12
TAU = math.tau


# ── point arithmetic ──────────────────────────────────────────────

def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def perp(a: Point) -> Point:
    """Left normal: (-y, x)."""
    return (-a[1], a[0])


def length(a: Point) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize(a: Point) -> Point:
    n = length(a)
    if n <= _EPS:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def rotate(a: Point, angle: float, center: Point = (0.0, 0.0)) -> Point:
    c, s = math.cos(angle), math.sin(angle)
    x, y = a[0] - center[0], a[1] - center[1]
    return (center[0] + x * c - y * s, center[1] + x * s + y * c)


def angle_of(a: Point) -> float:
    return math.atan2(a[1], a[0])


def norm_angle(a: float) -> float:
    """Angle into [0, 2π)."""
    r = math.fmod(a, TAU)
    if r < 0.0:
        r += TAU
    return r if math.isfinite(r) else 0.0


def ccw_sweep(a0: float, a1: float) -> float:
    """Counter-clockwise sweep from a0 to a1, in [0, 2π)."""
    return norm_angle(a1 - a0)


def _finite(p: Point) -> bool:
    return math.isfinite(p[0]) and math.isfinite(p[1])


# ── angles on arcs ────────────────────────────────────────────────

def arc_point(center: Point, radius: float, angle: float) -> Point:
    return (center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle))


def arc_contains(start: float, end: float, angle: float) -> bool:
    """Is ``angle`` on the CCW arc from start to end (inclusive)?"""
    if abs(ccw_sweep(start, end)) < 1e-12:
        return True
    return ccw_sweep(start, angle) <= ccw_sweep(start, end) + 1e-12


class Arc:
    """A circular arc: centre, radius, start/end angles (radians, CCW)."""

    __slots__ = ("center", "radius", "start", "end")

    def __init__(self, center: Point, radius: float, start: float,
                 end: float) -> None:
        self.center = center
        self.radius = radius
        self.start = norm_angle(start)
        self.end = norm_angle(end)

    def start_point(self) -> Point:
        return arc_point(self.center, self.radius, self.start)

    def end_point(self) -> Point:
        return arc_point(self.center, self.radius, self.end)

    def sweep(self) -> float:
        return ccw_sweep(self.start, self.end)

    def contains_angle(self, angle: float) -> bool:
        return arc_contains(self.start, self.end, angle)

    def point_at(self, t: float) -> Point:
        return arc_point(self.center, self.radius,
                         self.start + self.sweep() * t)


# ── intersections (upstream intersect.rs) ─────────────────────────

def line_line_infinite(a1: Point, a2: Point, b1: Point,
                       b2: Point) -> tuple[Point, float, float] | None:
    """Intersection of two infinite lines, with parameters on each."""
    r = sub(a2, a1)
    s = sub(b2, b1)
    den = cross(r, s)
    if abs(den) < 1e-14 * max(length(r) * length(s), 1e-300):
        return None
    t = cross(sub(b1, a1), s) / den
    u = cross(sub(b1, a1), r) / den
    return (add(a1, scale(r, t)), t, u)


def line_line(a1: Point, a2: Point, b1: Point,
              b2: Point) -> Point | None:
    """Intersection of two finite segments, or None."""
    found = line_line_infinite(a1, a2, b1, b2)
    if found is None:
        return None
    p, t, u = found
    tol = 1e-9
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return p
    return None


def line_circle(a: Point, b: Point, center: Point,
                radius: float) -> list[Point]:
    """Intersections of the infinite line through a–b with a circle.

    Measured from the foot of the perpendicular, as upstream:
    a discriminant of two O(length⁴) numbers loses the gap on a
    long line nearly tangent to a small circle.
    """
    d = sub(b, a)
    f = sub(a, center)
    aa = dot(d, d)
    if not (aa >= _EPS * _EPS and math.isfinite(radius)
            and _finite(f)):
        return []
    t = -dot(f, d) / aa
    nearest = add(f, scale(d, t))
    delta = radius * radius - dot(nearest, nearest)
    tol = 1e-12 * radius * radius
    if not math.isfinite(delta) or delta < -tol:
        return []
    foot = add(center, nearest)
    if delta <= tol:
        return [foot]
    offset = math.sqrt(delta / aa)
    step = scale(d, offset)
    return [sub(foot, step), add(foot, step)]


def circle_circle(c1: Point, r1: float, c2: Point,
                  r2: float) -> list[Point]:
    """Intersections of two full circles (upstream's scaled form)."""
    delta = sub(c2, c1)
    d = length(delta)
    if not (math.isfinite(d) and math.isfinite(r1) and math.isfinite(r2)):
        return []
    if d == 0.0 or r1 < 0.0 or r2 < 0.0:
        return []
    swapped = r2 < r1
    if swapped:
        base, direction = c2, scale(delta, -1.0 / d)
    else:
        base, direction = c1, scale(delta, 1.0 / d)
    sc = max(d, r1, r2)
    ds, r1s, r2s = d / sc, r1 / sc, r2 / sc
    if ds > r1s + r2s or ds < abs(r1s - r2s):
        return []
    rs, ro = (r2s, r1s) if swapped else (r1s, r2s)
    a = ((rs - ro) * (rs + ro) + ds * ds) / (2.0 * ds)
    h2 = (rs - a) * (rs + a)
    tol = max(1e-12 * rs * rs, 8.0 * 2.220446049250313e-16 * rs / ds)
    if h2 < -tol:
        return []
    p = add(base, scale(direction, a * sc))
    if not _finite(p):
        return []
    if h2 <= tol:
        return [p]
    step = scale(perp(direction), math.sqrt(h2) * sc)
    p1, p2 = add(p, step), sub(p, step)
    if not (_finite(p1) and _finite(p2)):
        return []
    return [p2, p1] if swapped else [p1, p2]


# ── ellipses ──────────────────────────────────────────────────────

def ellipse_at(center: Point, major: Point, ratio: float,
               t: float) -> Point:
    """Point at eccentric parameter t on an ellipse.

    ``major`` is the major-axis vector; the minor axis is
    ``perp(major) * ratio``.
    """
    minor = scale(perp(major), ratio)
    return (center[0] + major[0] * math.cos(t) + minor[0] * math.sin(t),
            center[1] + major[1] * math.cos(t) + minor[1] * math.sin(t))


def ellipse_derivative(major: Point, ratio: float, t: float) -> Point:
    minor = scale(perp(major), ratio)
    return (-major[0] * math.sin(t) + minor[0] * math.cos(t),
            -major[1] * math.sin(t) + minor[1] * math.cos(t))


def ellipse_normal(center: Point, major: Point, ratio: float,
                   p: Point) -> Point:
    """Outward normal of the ellipse at point p (closest parameter)."""
    best_t, best_d = 0.0, math.inf
    for k in range(72):
        t = TAU * k / 72.0
        d = dist(ellipse_at(center, major, ratio, t), p)
        if d < best_d:
            best_t, best_d = t, d
    # One Newton polish on the distance.
    for _ in range(8):
        q = ellipse_at(center, major, ratio, best_t)
        dq = ellipse_derivative(major, ratio, best_t)
        step = dot(sub(p, q), dq) / max(dot(dq, dq), 1e-300)
        best_t += step
        if abs(step) < 1e-12:
            break
    return normalize(perp(ellipse_derivative(major, ratio, best_t)))


def ellipse_on(center: Point, major: Point, ratio: float,
               p: Point) -> float:
    """Signed implicit value of p on the ellipse (0 = on it).

    (x/a)² + (y/b)² − 1 in the ellipse's own frame.
    """
    a_len = length(major)
    if a_len <= _EPS or ratio <= 0.0:
        return math.inf
    u = normalize(major)
    v = perp(u)
    rel = sub(p, center)
    x, y = dot(rel, u), dot(rel, v)
    b_len = a_len * ratio
    return (x / a_len) ** 2 + (y / b_len) ** 2 - 1.0


# ── splines ───────────────────────────────────────────────────────

def _bspline_basis(i: int, degree: int, t: float,
                   knots: list[float]) -> float:
    """Cox–de Boor basis N(i, degree) at t (open uniform knots)."""
    if degree == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    out = 0.0
    d1 = knots[i + degree] - knots[i]
    if d1 > 0.0:
        out += (t - knots[i]) / d1 * _bspline_basis(i, degree - 1, t, knots)
    d2 = knots[i + degree + 1] - knots[i + 1]
    if d2 > 0.0:
        out += (knots[i + degree + 1] - t) / d2 * _bspline_basis(
            i + 1, degree - 1, t, knots)
    return out


def _open_knots(n: int, degree: int) -> list[float]:
    inner = n - degree
    knots = [0.0] * (degree + 1)
    knots += [float(k) for k in range(1, inner)]
    knots += [float(inner)] * (degree + 1)
    return knots


def bspline_eval(control: list[Point], degree: int = 3,
                 t: float = 0.0) -> Point:
    """Cubic (default) B-spline point at parameter t in [0, 1]."""
    n = len(control)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return control[0]
    degree = min(degree, n - 1)
    knots = _open_knots(n, degree)
    span = float(n - degree)
    u = t * span
    u = min(u, span - 1e-12)
    x = y = 0.0
    for i in range(n):
        w = _bspline_basis(i, degree, u, knots)
        x += control[i][0] * w
        y += control[i][1] * w
    return (x, y)


def bspline_derivative(control: list[Point], degree: int = 3,
                       t: float = 0.0) -> Point:
    """Derivative of the B-spline at t (central differences)."""
    h = 1e-6
    t0, t1 = max(0.0, t - h), min(1.0, t + h)
    p0 = bspline_eval(control, degree, t0)
    p1 = bspline_eval(control, degree, t1)
    if t1 - t0 <= 0.0:
        return (0.0, 0.0)
    return ((p1[0] - p0[0]) / (t1 - t0), (p1[1] - p0[1]) / (t1 - t0))


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point,
                 t: float) -> Point:
    """One Catmull–Rom segment between p1 and p2."""
    t2, t3 = t * t, t * t * t
    x = 0.5 * ((2.0 * p1[0]) + (-p0[0] + p2[0]) * t
               + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
               + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3)
    y = 0.5 * ((2.0 * p1[1]) + (-p0[1] + p2[1]) * t
               + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
               + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3)
    return (x, y)


def spline_eval(points: list[Point], t: float,
                closed: bool = False, fit: bool = True) -> Point:
    """Point at parameter t in [0, 1] on a spline.

    Fit form (default): Catmull–Rom interpolation through the
    points, so the curve passes exactly through every fit point
    (upstream's interpolating spline). Control form: cubic
    B-spline through the control polygon.
    """
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    if not fit:
        return bspline_eval(points, 3, t)
    n = len(points)
    if closed:
        ext = [points[-1]] + points + [points[0], points[1]]
        seg = min(int(t * n), n - 1)
        u = t * n - seg
        return _catmull_rom(ext[seg], ext[seg + 1], ext[seg + 2],
                            ext[seg + 3], u)
    seg = min(int(t * (n - 1)), n - 2)
    u = t * (n - 1) - seg
    p0 = points[seg - 1] if seg > 0 else points[0]
    p3 = points[seg + 2] if seg + 2 < n else points[-1]
    return _catmull_rom(p0, points[seg], points[seg + 1], p3, u)


def spline_derivative(points: list[Point], t: float,
                      closed: bool = False, fit: bool = True) -> Point:
    """Derivative of the spline at t (central differences)."""
    h = 1e-6
    t0, t1 = max(0.0, t - h), min(1.0, t + h)
    if t1 - t0 <= 0.0:
        return (0.0, 0.0)
    p0 = spline_eval(points, t0, closed, fit)
    p1 = spline_eval(points, t1, closed, fit)
    return ((p1[0] - p0[0]) / (t1 - t0), (p1[1] - p0[1]) / (t1 - t0))


# ── fillets (upstream fillet.rs) ──────────────────────────────────

def _offset_point(kind: str, data: tuple, t: float, d: float,
                  side: float) -> Point | None:
    """Point at distance ``d`` along the normal of a curve at t.

    ``kind`` is "line" (data = (a, b)), "circle" (data = (c, r))
    or "spline" (data = (points, closed, fit)). ``side`` is +1 for
    the left normal, −1 for the right.
    """
    if kind == "line":
        a, b = data
        p = add(a, scale(sub(b, a), t))
        n = normalize(perp(sub(b, a)))
        return add(p, scale(n, side * d)) if n != (0.0, 0.0) else None
    if kind == "circle":
        c, r = data
        base = arc_point(c, r, t * TAU)
        n = normalize(sub(base, c))
        return add(base, scale(n, side * d)) if n != (0.0, 0.0) else None
    points, closed, fit = data
    p = spline_eval(points, t, closed, fit)
    n = normalize(perp(spline_derivative(points, t, closed, fit)))
    return add(p, scale(n, side * d)) if n != (0.0, 0.0) else None


def corner_fit(kind1: str, data1: tuple, kind2: str, data2: tuple,
               radius: float) -> list[dict]:
    """Fillet centres tangent to two curves at distance ``radius``.

    Returns a list of dicts ``{t1, t2, p1, p2, center}`` — the
    touch parameters and points on each curve and the centre of
    the tangent circle. ``radius == 0`` gives the crossings of
    the curves themselves. Degenerate input yields no solutions,
    never a crash.
    """
    if radius < 0.0 or not math.isfinite(radius):
        return []
    solutions: list[dict] = []
    samples = 72 if radius > 0.0 else 36
    # Sample the offset curves; the closest approach of the two
    # offsets seeds a Newton refinement on the exact offsets.
    for side1 in (1.0, -1.0):
        for side2 in (1.0, -1.0):
            best: tuple[float, float, float] | None = None
            for i in range(samples):
                t1 = i / samples
                p1 = _offset_point(kind1, data1, t1, radius, side1)
                if p1 is None:
                    continue
                for j in range(samples):
                    t2 = j / samples
                    p2 = _offset_point(kind2, data2, t2, radius, side2)
                    if p2 is None:
                        continue
                    d = dist(p1, p2)
                    if best is None or d < best[0]:
                        best = (d, t1, t2)
            if best is None:
                continue
            _, t1, t2 = best
            # Newton on (offset1(t1) − offset2(t2)) = 0.
            for _ in range(40):
                p1 = _offset_point(kind1, data1, t1, radius, side1)
                p2 = _offset_point(kind2, data2, t2, radius, side2)
                if p1 is None or p2 is None:
                    break
                err = sub(p1, p2)
                if length(err) < 1e-10:
                    break
                h = 1e-7
                d1 = sub(_offset_point(kind1, data1, min(1.0, t1 + h),
                                       radius, side1) or p1,
                         _offset_point(kind1, data1, max(0.0, t1 - h),
                                       radius, side1) or p1)
                d2 = sub(_offset_point(kind2, data2, min(1.0, t2 + h),
                                       radius, side2) or p2,
                         _offset_point(kind2, data2, max(0.0, t2 - h),
                                       radius, side2) or p2)
                d1 = scale(d1, 1.0 / (2.0 * h))
                d2 = scale(d2, 1.0 / (2.0 * h))
                # Solve [d1, −d2] · (dt1, dt2) = −err (2×2).
                a, b = d1[0], -d2[0]
                c, d = d1[1], -d2[1]
                det = a * d - b * c
                if abs(det) < 1e-300:
                    break
                dt1 = (-err[0] * d + b * err[1]) / det
                dt2 = (a * -err[1] + err[0] * c) / det
                t1 = min(1.0, max(0.0, t1 + dt1))
                t2 = min(1.0, max(0.0, t2 + dt2))
                if abs(dt1) < 1e-12 and abs(dt2) < 1e-12:
                    break
            p1 = _offset_point(kind1, data1, t1, radius, side1)
            p2 = _offset_point(kind2, data2, t2, radius, side2)
            if p1 is None or p2 is None:
                continue
            if dist(p1, p2) > 1e-6:
                continue
            centre = mid(p1, p2)
            if any(dist(centre, s["center"]) < 1e-6 for s in solutions):
                continue
            # Touch points on the ORIGINAL curves, not the offsets.
            if kind1 == "line":
                a, b = data1
                tp1 = add(a, scale(sub(b, a), t1))
            elif kind1 == "circle":
                tp1 = arc_point(data1[0], data1[1], t1 * TAU)
            else:
                tp1 = spline_eval(data1[0], t1, data1[1], data1[2])
            if kind2 == "line":
                a, b = data2
                tp2 = add(a, scale(sub(b, a), t2))
            elif kind2 == "circle":
                tp2 = arc_point(data2[0], data2[1], t2 * TAU)
            else:
                tp2 = spline_eval(data2[0], t2, data2[1], data2[2])
            solutions.append({"t1": t1, "t2": t2, "p1": tp1, "p2": tp2,
                              "center": centre})
            if len(solutions) >= 64:
                return solutions
    return solutions


def fillet_candidates(kind1: str, data1: tuple, kind2: str,
                      data2: tuple, radius: float) -> list[dict]:
    """Public wrapper for :func:`corner_fit` (same signature)."""
    return corner_fit(kind1, data1, kind2, data2, radius)


def tangent_points(center: Point, radius: float,
                   external: Point) -> list[Point]:
    """Tangent points from an external point to a circle."""
    d = dist(external, center)
    if d <= radius or radius <= 0.0:
        return []
    base = math.acos(radius / d)
    ang = angle_of(sub(external, center))
    return [arc_point(center, radius, ang + base),
            arc_point(center, radius, ang - base)]
