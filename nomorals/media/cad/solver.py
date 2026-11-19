"""Constraint solver — the parametric heart of CadCraft.

Port of the *design* of upstream ``crates/constraints``
(``model.rs``, ``residual.rs``, ``lm.rs``, ``dof.rs``,
``lib.rs``) into Devon's Python, stdlib only.

* :class:`Model` maps constrained entities to a flat parameter
  vector and back — line ``ax ay bx by``, circle ``cx cy r``,
  arc ``cx cy r start end``, polyline ``x0 y0 …``, point
  ``x y``, ellipse ``cx cy mx my ratio`` (+ ``start end`` for an
  elliptical arc), spline its defining points, insert ``x y
  rotation``.
* :class:`Built` interprets each constraint once against the
  starting geometry into a residual form (point–point, point–
  line, two lines, line–circle, rays for angles, …) with a
  target value and an orientation sign chosen from the starting
  geometry, then evaluates its residuals.
* :func:`lm_solve` is upstream's damped, weighted minimum-norm
  Gauss–Newton (Levenberg–Marquardt style) with a numerical
  Jacobian: under-constrained systems move as little as
  possible.
* :func:`dof_report` is upstream's degrees-of-freedom analysis:
  per connected component, remaining DOF is free parameters
  minus the rank of the constraint Jacobian; equations beyond
  the rank are redundant (over-constraint). Rank comes from
  Householder QR with column pivoting on the balanced Jacobian;
  per-object DOF from the null-space basis restricted to the
  object's own parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import CadError
from .geom import (
    TAU,
    arc_point,
    ccw_sweep,
    cross,
    dist,
    dot,
    ellipse_at,
    ellipse_derivative,
    ellipse_normal,
    ellipse_on,
    line_line_infinite,
    mid,
    norm_angle,
    normalize,
    perp,
    spline_derivative,
    spline_eval,
    sub,
)
from .model import (
    Arc,
    CadDocument,
    CadPoint,
    Circle,
    Constraint,
    ConstraintKind,
    Ellipse,
    GeomRef,
    Insert,
    Line,
    Polyline,
    Spline,
    Sub,
)

__all__ = [
    "MAX_CONSTRAINTS", "MAX_ITER", "MAX_PARAMS", "MAX_ROWS",
    "dof_report", "lm_solve", "measure", "purge_invalid", "solve",
    "validate_constraint",
]

MAX_PARAMS = 4000
MAX_ROWS = 800
MAX_CONSTRAINTS = 5000
MAX_ITER = 200
MAX_EVAL_WORK = 2e9
MAX_NULL_COLS = 600
MAX_RANK_WORK = 4e8
TOL = 1e-9
RANK_TOL = 1e-7
COL_NOISE = 1e-6


def _sgn(v: float) -> float:
    return -1.0 if v < 0.0 else 1.0


def _wrap(a: float) -> float:
    r = (a + math.pi) % TAU - math.pi
    return r if math.isfinite(r) else 0.0


def _dir_n(a, b):
    d = sub(b, a)
    n = math.hypot(*d)
    return (d[0] / n, d[1] / n) if n > 1e-12 else (0.0, 0.0)


def _ray(a, b, rev: bool):
    return _dir_n(b, a) if rev else _dir_n(a, b)


# ── parameter model (upstream model.rs) ───────────────────────────

@dataclass
class Slot:
    handle: str
    shape: tuple
    off: int
    length: int


class Model:
    """Flat parameter vector for a set of entities."""

    def __init__(self) -> None:
        self.slots: list[Slot] = []
        self.x0: list[float] = []
        self.index: dict[str, int] = {}

    @classmethod
    def build(cls, doc: CadDocument, handles: list[str]) -> Model:
        model = cls()
        for handle in handles:
            if handle in model.index:
                continue
            entity = doc.entities.get(handle)
            if entity is None:
                continue
            off = len(model.x0)
            shape = model._add(entity, off)
            if shape is None:
                continue
            if len(model.x0) > MAX_PARAMS or any(
                    not math.isfinite(v) for v in model.x0[off:]):
                del model.x0[off:]
                continue
            length = len(model.x0) - off
            model.index[handle] = len(model.slots)
            model.slots.append(Slot(handle, shape, off, length))
        return model

    def _add(self, entity, off: int) -> tuple | None:
        if isinstance(entity, Line):
            self.x0.extend([entity.a[0], entity.a[1],
                            entity.b[0], entity.b[1]])
            return ("line",)
        if isinstance(entity, Circle):
            self.x0.extend([entity.center[0], entity.center[1],
                            entity.radius])
            return ("circle",)
        if isinstance(entity, Arc):
            self.x0.extend([entity.center[0], entity.center[1],
                            entity.radius, entity.start, entity.end])
            return ("arc",)
        if isinstance(entity, Polyline):
            if len(entity.vertices) < 2:
                return None
            if off + len(entity.vertices) * 2 > MAX_PARAMS:
                return None
            for v in entity.vertices:
                self.x0.extend([v[0], v[1]])
            return ("poly", len(entity.vertices), entity.closed)
        if isinstance(entity, CadPoint):
            self.x0.extend([entity.p[0], entity.p[1]])
            return ("point",)
        if isinstance(entity, Ellipse):
            arc = not entity.is_full()
            self.x0.extend([entity.center[0], entity.center[1],
                            entity.major[0], entity.major[1],
                            entity.ratio])
            if arc:
                self.x0.extend([entity.start, entity.end])
            return ("ellipse", arc)
        if isinstance(entity, Spline):
            pts = entity.points
            if not (2 <= len(pts) <= 500):
                return None
            if off + len(pts) * 2 > MAX_PARAMS:
                return None
            for p in pts:
                self.x0.extend([p[0], p[1]])
            ends = (not entity.closed)
            return ("spline", len(pts), entity.fit, ends)
        if isinstance(entity, Insert):
            self.x0.extend([entity.position[0], entity.position[1],
                            entity.rotation])
            return ("insert", 0)
        return None

    def slot(self, handle: str) -> Slot | None:
        idx = self.index.get(handle)
        return self.slots[idx] if idx is not None else None

    def _p2(self, x, i: int):
        if i + 1 < len(x):
            return (x[i], x[i + 1])
        return None

    def point(self, x, ref: GeomRef):
        slot = self.slot(ref.handle)
        if slot is None:
            return None
        o = slot.off
        shape = slot.shape
        sub = ref.sub
        sub_name = sub[0] if isinstance(sub, tuple) else sub.value
        if shape[0] == "line":
            if sub_name == "start":
                return self._p2(x, o)
            if sub_name == "end":
                return self._p2(x, o + 2)
            if sub_name == "mid":
                a, b = self._p2(x, o), self._p2(x, o + 2)
                return mid(a, b) if a and b else None
        if shape[0] in ("circle", "arc") and sub_name == "center":
            return self._p2(x, o)
        if shape[0] == "arc" and sub_name in ("start", "end", "mid"):
            c = self._p2(x, o)
            rad = x[o + 2]
            a0, a1 = x[o + 3], x[o + 4]
            ang = a0 if sub_name == "start" else (
                a1 if sub_name == "end" else a0 + ccw_sweep(a0, a1) / 2)
            return arc_point(c, rad, ang) if c else None
        if shape[0] == "poly":
            n = shape[1]
            if sub_name == "vertex":
                i = sub[1]
                return self._p2(x, o + 2 * i) if i < n else None
            if sub_name == "start":
                return self._p2(x, o)
            if sub_name == "end":
                return self._p2(x, o + 2 * (n - 1))
        if shape[0] == "point" and sub_name in ("whole", "start",
                                                 "center"):
            return self._p2(x, o)
        if shape[0] == "ellipse":
            if sub_name == "center":
                return self._p2(x, o)
            if sub_name == "vertex":
                ell = self.ellipse(x, ref)
                if ell is None:
                    return None
                c, mj, ratio, _, _, _ = ell
                minor = (perp(mj)[0] * ratio, perp(mj)[1] * ratio)
                axis = {0: mj, 1: minor, 2: (-mj[0], -mj[1]),
                        3: (-minor[0], -minor[1])}.get(sub[1])
                return (c[0] + axis[0], c[1] + axis[1]) if axis else None
            if shape[1] and sub_name in ("start", "end", "mid"):
                ell = self.ellipse(x, ref)
                if ell is None:
                    return None
                c, mj, ratio, t0, t1, _ = ell
                t = t0 if sub_name == "start" else (
                    t1 if sub_name == "end"
                    else t0 + ccw_sweep(t0, t1) / 2)
                return ellipse_at(c, mj, ratio, t)
        if shape[0] == "spline":
            n = shape[1]
            if sub_name == "vertex":
                i = sub[1]
                return self._p2(x, o + 2 * i) if i < n else None
            if shape[3] and sub_name == "start":
                return self._p2(x, o)
            if shape[3] and sub_name == "end":
                return self._p2(x, o + 2 * (n - 1))
        if shape[0] == "insert" and sub_name in ("whole", "start",
                                                  "center"):
            return self._p2(x, o)
        return None

    def ellipse(self, x, ref: GeomRef):
        slot = self.slot(ref.handle)
        if slot is None or slot.shape[0] != "ellipse":
            return None
        o = slot.off
        arc = slot.shape[1]
        t0 = x[o + 5] if arc else 0.0
        t1 = x[o + 6] if arc else TAU
        c = self._p2(x, o)
        mj = self._p2(x, o + 2)
        if c is None or mj is None:
            return None
        return (c, mj, x[o + 4], t0, t1, arc)

    def is_spline(self, ref: GeomRef) -> bool:
        slot = self.slot(ref.handle)
        return slot is not None and slot.shape[0] == "spline"

    def line(self, x, ref: GeomRef):
        slot = self.slot(ref.handle)
        if slot is None:
            return None
        o = slot.off
        if slot.shape[0] == "line":
            a, b = self._p2(x, o), self._p2(x, o + 2)
            return (a, b) if a and b else None
        if slot.shape[0] == "poly":
            n, closed = slot.shape[1], slot.shape[2]
            sub = ref.sub
            if isinstance(sub, tuple) and sub[0] == "segment":
                i = sub[1]
                if i + 1 < n:
                    a, b = self._p2(x, o + 2 * i), self._p2(x, o + 2 * i + 2)
                elif closed and i + 1 == n and n > 2:
                    a, b = self._p2(x, o + 2 * i), self._p2(x, o)
                else:
                    return None
                return (a, b) if a and b else None
            if n == 2:
                a, b = self._p2(x, o), self._p2(x, o + 2)
                return (a, b) if a and b else None
        return None

    def circle(self, x, ref: GeomRef):
        slot = self.slot(ref.handle)
        if slot is None or slot.shape[0] not in ("circle", "arc"):
            return None
        c = self._p2(x, slot.off)
        return (c, x[slot.off + 2]) if c else None

    def arc_sweep(self, x, ref: GeomRef):
        slot = self.slot(ref.handle)
        if slot is None or slot.shape[0] != "arc":
            return None
        return ccw_sweep(x[slot.off + 3], x[slot.off + 4])

    def end_frame(self, x, ref: GeomRef, at_end: bool):
        """(point, tangent, curvature) at an end of a curve."""
        slot = self.slot(ref.handle)
        if slot is None:
            return None
        o = slot.off
        sub = ref.sub
        sub_name = sub[0] if isinstance(sub, tuple) else sub.value
        if slot.shape[0] == "poly":
            seg = None
            if sub_name == "start":
                seg = ("segment", 0)
            elif sub_name == "end" and not slot.shape[2]:
                seg = ("segment", slot.shape[1] - 2)
            if seg is not None:
                line = self.line(x, GeomRef(ref.handle, seg))
                if line:
                    a, b = line
                    d = sub(b, a)
                    n = math.hypot(*d)
                    if n > 1e-300:
                        p = b if at_end else a
                        return (p, (d[0] / n, d[1] / n), 0.0)
        if slot.shape[0] == "line":
            line = self.line(x, ref)
            if line:
                a, b = line
                d = sub(b, a)
                n = math.hypot(*d)
                if n > 1e-300:
                    return ((b if at_end else a),
                            (d[0] / n, d[1] / n), 0.0)
        if slot.shape[0] == "arc":
            c = self._p2(x, o)
            rad = x[o + 2]
            ang = x[o + 4] if at_end else x[o + 3]
            if c and abs(rad) > 1e-300:
                p = arc_point(c, rad, ang)
                t = normalize(perp(sub(p, c)))
                return (p, t, 1.0 / rad)
        if slot.shape[0] == "ellipse" and slot.shape[1]:
            ell = self.ellipse(x, ref)
            if ell:
                c, mj, ratio, t0, t1, _ = ell
                t = t1 if at_end else t0
                p = ellipse_at(c, mj, ratio, t)
                d1 = ellipse_derivative(mj, ratio, t)
                n = math.hypot(*d1)
                if n > 1e-300:
                    return (p, (d1[0] / n, d1[1] / n), 0.0)
        if slot.shape[0] == "spline" and slot.shape[3]:
            n, fit = slot.shape[1], slot.shape[2]
            pts = [self._p2(x, o + 2 * i) for i in range(n)]
            if any(p is None for p in pts):
                return None
            t = 1.0 if at_end else 0.0
            p = spline_eval(pts, t, False, fit)
            d1 = spline_derivative(pts, t, False, fit)
            n1 = math.hypot(*d1)
            if n1 <= 1e-300:
                return None
            h = 1e-4
            d1a = spline_derivative(pts, min(1.0, t + h), False, fit)
            d1b = spline_derivative(pts, max(0.0, t - h), False, fit)
            span = (min(1.0, t + h) - max(0.0, t - h)) or 1.0
            d2 = ((d1a[0] - d1b[0]) / span, (d1a[1] - d1b[1]) / span)
            k = abs(d1[0] * d2[1] - d1[1] * d2[0]) / (n1 ** 3)
            return (p, (d1[0] / n1, d1[1] / n1), k)
        return None

    def point_refs(self, handle: str) -> list[GeomRef]:
        slot = self.slot(handle)
        if slot is None:
            return []
        from .model import Sub

        def at(s):
            return GeomRef(handle, s)

        if slot.shape[0] == "line":
            return [at(Sub.START), at(Sub.END)]
        if slot.shape[0] == "point":
            return [at(Sub.WHOLE)]
        if slot.shape[0] in ("circle", "arc", "ellipse", "insert"):
            return [at(Sub.CENTER)]
        if slot.shape[0] in ("poly", "spline"):
            return [at(("vertex", i)) for i in range(slot.shape[1])]
        return []

    def lock_indices(self, ref: GeomRef) -> list[int] | None:
        slot = self.slot(ref.handle)
        if slot is None:
            return None
        o = slot.off
        sub = ref.sub
        sub_name = sub[0] if isinstance(sub, tuple) else sub.value
        if sub_name == "whole":
            return list(range(o, o + slot.length))
        if slot.shape[0] == "line":
            if sub_name == "start":
                return [o, o + 1]
            if sub_name == "end":
                return [o + 2, o + 3]
        if slot.shape[0] in ("circle", "arc") and sub_name == "center":
            return [o, o + 1]
        if slot.shape[0] == "poly":
            n = slot.shape[1]
            if sub_name == "vertex" and sub[1] < n:
                return [o + 2 * sub[1], o + 2 * sub[1] + 1]
            if sub_name == "start":
                return [o, o + 1]
            if sub_name == "end" and n > 0:
                return [o + 2 * (n - 1), o + 2 * (n - 1) + 1]
            if sub_name == "segment":
                i = sub[1]
                if i + 1 < n:
                    return [o + 2 * i, o + 2 * i + 1,
                            o + 2 * i + 2, o + 2 * i + 3]
                if slot.shape[2] and i + 1 == n and n > 2:
                    return [o + 2 * i, o + 2 * i + 1, o, o + 1]
        if slot.shape[0] in ("ellipse", "insert") \
                and sub_name == "center":
            return [o, o + 1]
        if slot.shape[0] == "insert" and sub_name == "start":
            return [o, o + 1]
        if slot.shape[0] == "spline":
            n = slot.shape[1]
            if sub_name == "vertex" and sub[1] < n:
                return [o + 2 * sub[1], o + 2 * sub[1] + 1]
            if slot.shape[3]:
                if sub_name == "start":
                    return [o, o + 1]
                if sub_name == "end" and n > 0:
                    return [o + 2 * (n - 1), o + 2 * (n - 1) + 1]
        return None

    def spline_reconstruct_cost(self, handle: str) -> float:
        slot = self.slot(handle)
        if slot is None or slot.shape[0] != "spline":
            return 0.0
        n = float(slot.shape[1])
        return n ** 3 if slot.shape[2] else n ** 2

    def storable(self, x) -> bool:
        for slot in self.slots:
            if slot.shape[0] == "ellipse":
                ratio = x[slot.off + 4]
                if not (0.0 < ratio <= 1.0 + 1e-9):
                    return False
        return True

    def write_back(self, doc: CadDocument, x) -> list[str]:
        moved: list[str] = []
        for slot in self.slots:
            new = x[slot.off:slot.off + slot.length]
            old = self.x0[slot.off:slot.off + slot.length]
            if len(new) != len(old) or len(new) == 0:
                continue
            if any(not math.isfinite(v) for v in new):
                continue
            scale = max((abs(v) for v in old), default=1.0)
            if all(abs(a - b) <= 1e-13 * max(scale, 1.0)
                   for a, b in zip(new, old, strict=True)):
                continue
            entity = doc.entities.get(slot.handle)
            if entity is None:
                continue
            self._apply(entity, slot, new)
            doc._dirty_ids.add(slot.handle)
            moved.append(slot.handle)
        return moved

    def _apply(self, entity, slot: Slot, new: list[float]) -> None:
        shape = slot.shape
        if isinstance(entity, Line) and shape[0] == "line":
            entity.a = (new[0], new[1])
            entity.b = (new[2], new[3])
        elif isinstance(entity, Circle) and shape[0] == "circle":
            entity.center = (new[0], new[1])
            entity.radius = max(abs(new[2]), 1e-12)
        elif isinstance(entity, Arc) and shape[0] == "arc":
            turn = math.pi if new[2] < 0.0 else 0.0
            entity.center = (new[0], new[1])
            entity.radius = max(abs(new[2]), 1e-12)
            entity.start = norm_angle(new[3] + turn)
            entity.end = norm_angle(new[4] + turn)
        elif isinstance(entity, Polyline) and shape[0] == "poly":
            n = shape[1]
            if len(entity.vertices) == n:
                entity.vertices = [(new[2 * i], new[2 * i + 1])
                                   for i in range(n)]
        elif isinstance(entity, CadPoint) and shape[0] == "point":
            entity.p = (new[0], new[1])
        elif isinstance(entity, Ellipse) and shape[0] == "ellipse":
            arc = shape[1]
            major = (new[2], new[3])
            ratio = abs(new[4])
            t0, t1 = entity.start, entity.end
            if ratio > 1.0 + 1e-9:
                major = (perp(major)[0] * ratio, perp(major)[1] * ratio)
                ratio = 1.0 / ratio
                if arc:
                    t0 -= math.pi / 2
                    t1 -= math.pi / 2
            if math.hypot(*major) > 1e-12:
                entity.center = (new[0], new[1])
                entity.major = major
                entity.ratio = min(max(ratio, 1e-9), 1.0)
                if arc:
                    entity.start = norm_angle(new[5])
                    entity.end = norm_angle(new[6])
        elif isinstance(entity, Spline) and shape[0] == "spline":
            n = shape[1]
            entity.points = [(new[2 * i], new[2 * i + 1])
                             for i in range(n)]
        elif isinstance(entity, Insert) and shape[0] == "insert":
            entity.position = (new[0], new[1])
            entity.rotation = norm_angle(new[2])


# ── residual forms (upstream residual.rs) ─────────────────────────

@dataclass
class Built:
    id: int
    kind: ConstraintKind
    refs: list[GeomRef]
    form: tuple = ()
    target: float = 0.0
    sign: float = 1.0
    rows: int = 0
    aux_init: list[float] = field(default_factory=list)
    aux: int = 0

    @classmethod
    def new(cls, constraint: Constraint, model: Model, x,
            target: float) -> Built:
        kind = constraint.kind
        refs = list(constraint.refs)
        n = len(refs)
        def pt(i):
            return model.point(x, refs[i]) if i < n else None

        def ln(i):
            return model.line(x, refs[i]) if i < n else None

        def ci(i):
            return model.circle(x, refs[i]) if i < n else None

        def el(i):
            return model.ellipse(x, refs[i]) if i < n else None

        def lnx(i):
            return ln(i) or (el(i) and (
                (el(i)[0][0] - el(i)[1][0], el(i)[0][1] - el(i)[1][1]),
                (el(i)[0][0] + el(i)[1][0], el(i)[0][1] + el(i)[1][1])))

        def need(what):
            raise CadError(f"{kind.name()} needs {what}", kind="solve")

        b = cls(id=constraint.id, kind=kind, refs=refs)
        if kind == ConstraintKind.COINCIDENT:
            if n != 2:
                need("two references")
            if pt(0) is not None and pt(1) is not None:
                b.form, b.rows = ("pp",), 2
            elif pt(0) is not None and ln(1) is not None:
                b.form, b.rows = ("pl", False), 1
            elif pt(1) is not None and ln(0) is not None:
                b.form, b.rows = ("pl", True), 1
            elif pt(0) is not None and ci(1) is not None:
                b.form, b.rows = ("pc", False), 1
            elif pt(1) is not None and ci(0) is not None:
                b.form, b.rows = ("pc", True), 1
            elif pt(0) is not None and el(1) is not None:
                b.form, b.rows = ("pe", False), 1
            elif pt(1) is not None and el(0) is not None:
                b.form, b.rows = ("pe", True), 1
            else:
                need("two points, or a point and a line/circle/arc/"
                     "ellipse")
        elif kind in (ConstraintKind.HORIZONTAL,
                      ConstraintKind.VERTICAL):
            if n == 1 and lnx(0) is not None:
                b.form, b.rows = ("l1",), 1
            elif n == 2 and pt(0) is not None and pt(1) is not None:
                b.form, b.rows = ("p2",), 1
            else:
                need("a line, an ellipse or two points")
        elif kind in (ConstraintKind.PARALLEL,
                      ConstraintKind.PERPENDICULAR,
                      ConstraintKind.COLLINEAR):
            if not (n == 2 and ln(0) is not None
                    and ln(1) is not None):
                need("two lines")
            b.form = ("ll",)
            b.rows = 2 if kind == ConstraintKind.COLLINEAR else 1
        elif kind == ConstraintKind.CONCENTRIC:
            def center(i):
                return (ci(i) or (None,))[0] if ci(i) else (
                    el(i)[0] if el(i) else pt(i))

            def curve(i):
                return ci(i) is not None or el(i) is not None

            if not (n == 2 and center(0) is not None
                    and center(1) is not None
                    and (curve(0) or curve(1))):
                need("two circles, arcs or ellipses")
            b.form, b.rows = ("cc",), 2
        elif kind == ConstraintKind.EQUAL:
            if n != 2:
                need("two objects")
            if ln(0) is not None and ln(1) is not None:
                b.form, b.rows = ("ll",), 1
            elif ci(0) is not None and ci(1) is not None:
                b.form, b.rows = ("cc",), 1
            else:
                need("two lines or two circles/arcs")
        elif kind in (ConstraintKind.TANGENT, ConstraintKind.SMOOTH):
            return cls._tangent_form(b, constraint, model, x)
        elif kind == ConstraintKind.SYMMETRIC:
            if n != 3 or ln(2) is None:
                need("two objects and a symmetry line")
            if pt(0) is not None and pt(1) is not None:
                b.form, b.rows = ("pp",), 2
            elif ln(0) is not None and ln(1) is not None:
                a1, b1 = ln(0)
                a2, b2 = ln(1)
                la, lb = ln(2)
                def mirror(p, q1, q2):
                    # Mirror of p across the line q1–q2:
                    # 2 * projection − p.
                    d = _dir_n(q1, q2)
                    rel = sub(p, q1)
                    return (q1[0] + 2 * d[0] * dot(rel, d) - rel[0],
                            q1[1] + 2 * d[1] * dot(rel, d) - rel[1])
                straight = dist(mirror(a1, la, lb), a2) + dist(
                    mirror(b1, la, lb), b2)
                crossed = dist(mirror(a1, la, lb), b2) + dist(
                    mirror(b1, la, lb), a2)
                b.form, b.rows = ("symlines", crossed < straight), 4
            elif ci(0) is not None and ci(1) is not None:
                b.form, b.rows = ("cc",), 3
            else:
                need("two points, lines or circles and a symmetry "
                     "line")
        elif kind == ConstraintKind.FIX:
            if n != 1:
                need("one reference")
            ref = refs[0]
            if model.lock_indices(ref) is not None:
                b.form, b.rows = ("lock",), 0
            elif pt(0) is not None:
                b.form, b.rows = ("anchor", pt(0)), 2
            else:
                need("a point or an object")
        elif kind in (ConstraintKind.DISTANCE_ALIGNED,
                      ConstraintKind.DISTANCE_HORIZONTAL,
                      ConstraintKind.DISTANCE_VERTICAL):
            b.rows = 1
            axis = kind
            if n == 1 and ln(0) is not None:
                b.form = ("l1",)
            elif n == 2 and pt(0) is not None and pt(1) is not None:
                b.form = ("p2",)
            elif n == 2 and axis == ConstraintKind.DISTANCE_ALIGNED \
                    and pt(0) is not None and ln(1) is not None:
                b.form = ("pl", False)
            elif n == 2 and axis == ConstraintKind.DISTANCE_ALIGNED \
                    and pt(1) is not None and ln(0) is not None:
                b.form = ("pl", True)
            else:
                need("a line or two points")
            if axis != ConstraintKind.DISTANCE_ALIGNED or \
                    b.form[0] == "pl":
                b.sign = 1.0
                out: list[float] = []
                b.target = 0.0
                b.eval(model, x, out)
                b.sign = _sgn(out[0] if out else 0.0)
                b.target = target
            if not (math.isfinite(target) and target >= 0.0):
                raise CadError(
                    f"{kind.name()} value must be non-negative",
                    kind="solve")
        elif kind == ConstraintKind.ANGULAR:
            b.rows = 1
            if n == 2 and ln(0) is not None and ln(1) is not None:
                a1, b1 = ln(0)
                a2, b2 = ln(1)
                found = line_line_infinite(a1, b1, a2, b2)
                rev = [False, False]
                if found is not None:
                    _, t, u = found
                    rev = [t > 0.5, u > 0.5]
                b.form = ("rays", tuple(rev))
                d1 = _ray(a1, b1, rev[0])
                d2 = _ray(a2, b2, rev[1])
                b.sign = _sgn(math.atan2(cross(d1, d2), dot(d1, d2)))
            elif n == 1 and model.arc_sweep(x, refs[0]) is not None:
                b.form = ("arcsweep",)
            else:
                need("two lines or an arc")
            b.target = math.radians(target)
            max_ang = math.pi if b.form[0] == "rays" else TAU
            if not math.isfinite(b.target) or abs(b.target) > max_ang:
                raise CadError("angle out of range", kind="solve")
        elif kind in (ConstraintKind.RADIUS, ConstraintKind.DIAMETER):
            if not (n == 1 and ci(0) is not None):
                need("a circle or arc")
            b.form, b.rows = ("c1",), 1
            if not math.isfinite(target) or target <= 0.0:
                raise CadError(
                    f"{kind.name()} value must be positive",
                    kind="solve")
        else:  # pragma: no cover - kind enum is exhaustive
            raise CadError(f"unhandled constraint kind {kind}",
                           kind="solve")
        # Angular targets were already converted to radians in
        # their branch; every other kind takes the raw target.
        if kind != ConstraintKind.ANGULAR:
            b.target = target
        return b

    @classmethod
    def _tangent_form(cls, b: Built, constraint: Constraint,
                      model: Model, x) -> Built:
        def need(what):
            raise CadError(f"{constraint.kind.name()} needs {what}",
                           kind="solve")

        if len(b.refs) != 2:
            need("two objects")

        def ln(i):
            return model.line(x, b.refs[i])

        def ci(i):
            return model.circle(x, b.refs[i])

        def el(i):
            return model.ellipse(x, b.refs[i])

        if any(model.is_spline(r) for r in b.refs):
            return cls._spline_form(b, model, x)
        b.rows = 1
        if ln(0) is not None and ci(1) is not None:
            b.form = ("lc", False)
            a, bb = ln(0)
            c, _ = ci(1)
            b.sign = _sgn(cross(_dir_n(a, bb), sub(c, a)))
        elif ln(1) is not None and ci(0) is not None:
            b.form = ("lc", True)
            a, bb = ln(1)
            c, _ = ci(0)
            b.sign = _sgn(cross(_dir_n(a, bb), sub(c, a)))
        elif ci(0) is not None and ci(1) is not None:
            b.form = ("cc",)
            (c1, r1), (c2, r2) = ci(0), ci(1)
            d = dist(c1, c2)
            b.sign = 1.0 if d >= max(abs(r1), abs(r2)) else -1.0
            b.target = _sgn(r1 - r2)
        elif ln(0) is not None and el(1) is not None:
            b.form = ("le", False)
            a, bb = ln(0)
            b.sign = _sgn(cross(_dir_n(a, bb), sub(el(1)[0], a)))
        elif ln(1) is not None and el(0) is not None:
            b.form = ("le", True)
            a, bb = ln(1)
            b.sign = _sgn(cross(_dir_n(a, bb), sub(el(0)[0], a)))
        elif (ci(0) is not None or el(0) is not None) and \
                (ci(1) is not None or el(1) is not None):
            b.form = ("contact",)
            b.rows = 3
            guess = cls._contact_guess(b, model, x)
            if guess is None:
                raise CadError(
                    f"{constraint.kind.name()} could not find where "
                    "the curves touch", kind="solve")
            b.aux_init = guess
        elif ln(0) is not None and ln(1) is not None \
                and constraint.kind == ConstraintKind.SMOOTH:
            b.form = ("ll",)
        else:
            need("a line and a circle/arc/ellipse, two curves, or "
                 "a spline end and a curve")
        return b

    @classmethod
    def _spline_form(cls, b: Built, model: Model, x) -> Built:
        r0, r1 = b.refs[0], b.refs[1]

        def ends(r):
            if isinstance(r.sub, Sub) and r.sub == Sub.START:
                return [False]
            if isinstance(r.sub, Sub) and r.sub == Sub.END:
                return [True]
            return [False, True]

        join = b.kind == ConstraintKind.SMOOTH or (
            model.is_spline(r0) and model.is_spline(r1))
        if join:
            best = None
            for ea in ends(r0):
                for eb in ends(r1):
                    fa = model.end_frame(x, r0, ea)
                    fb = model.end_frame(x, r1, eb)
                    if fa and fb:
                        d = dist(fa[0], fb[0])
                        if best is None or d < best[0]:
                            best = (d, ea, eb)
            if best is None:
                raise CadError(
                    f"{b.kind.name()} needs an open spline and a "
                    "line, arc, elliptical arc or open spline",
                    kind="solve")
            _, a_end, b_end = best
            g2 = b.kind == ConstraintKind.SMOOTH
            b.form = ("join", a_end, b_end, g2)
            b.rows = 4 if g2 else 3
            return b
        si = 0 if model.is_spline(r0) else 1
        rs = r0 if si == 0 else r1
        ro = r1 if si == 0 else r0

        def dist_to(p):
            line = model.line(x, ro)
            if line:
                a, bb = line
                return abs(cross(_dir_n(a, bb), sub(p, a)))
            circ = model.circle(x, ro)
            if circ:
                c, rad = circ
                return abs(dist(p, c) - rad)
            ell = model.ellipse(x, ro)
            if ell:
                c, mj, ratio, _, _, _ = ell
                return abs(ellipse_on(c, mj, ratio, p))
            return None

        best = None
        for e in ends(rs):
            frame = model.end_frame(x, rs, e)
            if frame is None:
                continue
            d = dist_to(frame[0])
            if d is not None and math.isfinite(d) and (
                    best is None or d < best[0]):
                best = (d, e)
        if best is None:
            raise CadError(
                "Tangent needs an open spline and a line, circle, "
                "arc, ellipse or spline", kind="solve")
        b.form = ("endon", si, best[1])
        b.rows = 2
        return b

    def _curve_at(self, model: Model, x, i: int, t: float):
        ref = self.refs[i]
        circ = model.circle(x, ref)
        if circ:
            c, rad = circ
            return (arc_point(c, rad, t),
                    perp((math.cos(t), math.sin(t))))
        ell = model.ellipse(x, ref)
        if ell:
            c, mj, ratio, _, _, _ = ell
            return (ellipse_at(c, mj, ratio, t),
                    ellipse_derivative(mj, ratio, t))
        return None

    @classmethod
    def _contact_guess(cls, b: Built, model: Model, x):
        step = TAU / 72.0
        s0 = [(k * step, cls._curve_at_static(b, model, x, 0,
                                              k * step))
              for k in range(72)]
        s1 = [(k * step, cls._curve_at_static(b, model, x, 1,
                                              k * step))
              for k in range(72)]
        best = None
        for t0, p0 in s0:
            if p0 is None:
                continue
            for t1, p1 in s1:
                if p1 is None:
                    continue
                d = dist(p0[0], p1[0])
                if math.isfinite(d) and (best is None or d < best[0]):
                    best = (d, t0, t1)
        return [best[1], best[2]] if best else None

    @staticmethod
    def _curve_at_static(b, model, x, i, t):
        ref = b.refs[i]
        circ = model.circle(x, ref)
        if circ:
            c, rad = circ
            return (arc_point(c, rad, t),
                    perp((math.cos(t), math.sin(t))))
        ell = model.ellipse(x, ref)
        if ell:
            c, mj, ratio, _, _, _ = ell
            return (ellipse_at(c, mj, ratio, t),
                    ellipse_derivative(mj, ratio, t))
        return None

    def eval(self, model: Model, x, out: list[float]) -> None:
        start = len(out)
        self._eval_inner(model, x, out)
        while len(out) < start + self.rows:
            out.append(0.0)

    def _eval_inner(self, model: Model, x, out: list[float]) -> None:
        kind, form = self.kind, self.form

        def pt(i):
            return model.point(x, self.refs[i])

        def ln(i):
            return model.line(x, self.refs[i])

        def ci(i):
            return model.circle(x, self.refs[i])

        def el(i):
            return model.ellipse(x, self.refs[i])

        if form[0] == "lock":
            return
        if form[0] == "anchor":
            p = pt(0)
            if p is None:
                return
            out.extend([p[0] - form[1][0], p[1] - form[1][1]])
            return
        if form[0] == "pe":
            swap = form[1]
            p = pt(1) if swap else pt(0)
            e = el(0) if swap else el(1)
            if p is None or e is None:
                return
            out.append(ellipse_on(e[0], e[1], e[2], p))
            return
        if form[0] == "le":
            swap = form[1]
            line = ln(1) if swap else ln(0)
            e = el(0) if swap else el(1)
            if line is None or e is None:
                return
            a, b = line
            direction = _dir_n(a, b)
            # Support distance of the ellipse along the normal:
            # max |n·(point − c)| over the ellipse.
            n = perp(direction)
            mj = e[1]
            minor = (perp(mj)[0] * e[2], perp(mj)[1] * e[2])
            support = math.sqrt((dot(n, mj)) ** 2
                                + (dot(n, minor)) ** 2)
            out.append(self.sign * cross(direction, sub(e[0], a))
                       - support)
            return
        if form[0] == "contact":
            t0 = x[self.aux]
            t1 = x[self.aux + 1]
            c0 = self._curve_at(model, x, 0, t0)
            c1 = self._curve_at(model, x, 1, t1)
            if c0 is None or c1 is None:
                return
            (p0, d0), (p1, d1) = c0, c1
            n0 = normalize(d0)
            n1 = normalize(d1)
            out.extend([p0[0] - p1[0], p0[1] - p1[1],
                        cross(n0, n1)])
            return
        if form[0] == "endon":
            _, spline, at_end = form
            frame = model.end_frame(x, self.refs[spline], at_end)
            if frame is None:
                return
            other = 1 - min(spline, 1)
            line = ln(other)
            if line:
                a, b = line
                direction = _dir_n(a, b)
                out.extend([cross(direction, sub(frame[0], a)),
                            cross(frame[1], direction)])
                return
            circ = ci(other)
            if circ:
                c, rad = circ
                out.extend([dist(frame[0], c) - rad,
                            dot(frame[1], _dir_n(c, frame[0]))])
                return
            ell = el(other)
            if ell:
                c, mj, ratio, _, _, _ = ell
                out.extend([ellipse_on(c, mj, ratio, frame[0]),
                            dot(frame[1],
                                ellipse_normal(c, mj, ratio,
                                               frame[0]))])
                return
            return
        if form[0] == "join":
            _, a_end, b_end, g2 = form
            f1 = model.end_frame(x, self.refs[0], a_end)
            f2 = model.end_frame(x, self.refs[1], b_end)
            if f1 is None or f2 is None:
                return
            if not a_end:
                f1 = (f1[0], (-f1[1][0], -f1[1][1]), f1[2])
            if b_end:
                f2 = (f2[0], (-f2[1][0], -f2[1][1]), f2[2])
            out.extend([f1[0][0] - f2[0][0], f1[0][1] - f2[0][1],
                        cross(f1[1], f2[1])])
            if g2:
                out.append(f1[2] - f2[2])
            return
        if kind == ConstraintKind.COINCIDENT and form[0] == "pp":
            p0, p1 = pt(0), pt(1)
            if p0 is None or p1 is None:
                return
            out.extend([p0[0] - p1[0], p0[1] - p1[1]])
            return
        if kind == ConstraintKind.COINCIDENT and form[0] == "pl":
            swap = form[1]
            p = pt(1) if swap else pt(0)
            line = ln(0) if swap else ln(1)
            if p is None or line is None:
                return
            a, b = line
            out.append(cross(_dir_n(a, b), sub(p, a)))
            return
        if kind == ConstraintKind.COINCIDENT and form[0] == "pc":
            swap = form[1]
            p = pt(1) if swap else pt(0)
            circ = ci(0) if swap else ci(1)
            if p is None or circ is None:
                return
            out.append(dist(p, circ[0]) - circ[1])
            return
        if kind in (ConstraintKind.HORIZONTAL,
                    ConstraintKind.VERTICAL):
            if form[0] == "l1":
                line = ln(0)
                if line is None and el(0) is not None:
                    e = el(0)
                    line = ((e[0][0] - e[1][0], e[0][1] - e[1][1]),
                            (e[0][0] + e[1][0], e[0][1] + e[1][1]))
                if line is None:
                    return
                a, b = line
            else:
                a, b = pt(0), pt(1)
                if a is None or b is None:
                    return
            value = (a[1] - b[1]) if kind == ConstraintKind.HORIZONTAL \
                else (a[0] - b[0])
            out.append(value)
            return
        if kind in (ConstraintKind.PARALLEL, ConstraintKind.SMOOTH) \
                and form[0] == "ll":
            l0, l1 = ln(0), ln(1)
            if l0 is None or l1 is None:
                return
            out.append(cross(_dir_n(*l0), _dir_n(*l1)))
            return
        if kind == ConstraintKind.PERPENDICULAR:
            l0, l1 = ln(0), ln(1)
            if l0 is None or l1 is None:
                return
            out.append(dot(_dir_n(*l0), _dir_n(*l1)))
            return
        if kind == ConstraintKind.COLLINEAR:
            l0, l1 = ln(0), ln(1)
            if l0 is None or l1 is None:
                return
            d = _dir_n(*l0)
            out.extend([cross(d, sub(l1[0], l0[0])),
                        cross(d, sub(l1[1], l0[0]))])
            return
        if kind == ConstraintKind.CONCENTRIC:
            def center(i):
                c = ci(i)
                if c:
                    return c[0]
                e = el(i)
                if e:
                    return e[0]
                return pt(i)
            c0, c1 = center(0), center(1)
            if c0 is None or c1 is None:
                return
            out.extend([c0[0] - c1[0], c0[1] - c1[1]])
            return
        if kind == ConstraintKind.EQUAL and form[0] == "ll":
            l0, l1 = ln(0), ln(1)
            if l0 is None or l1 is None:
                return
            out.append(dist(*l0) - dist(*l1))
            return
        if kind == ConstraintKind.EQUAL:
            c0, c1 = ci(0), ci(1)
            if c0 is None or c1 is None:
                return
            out.append(c0[1] - c1[1])
            return
        if kind in (ConstraintKind.TANGENT, ConstraintKind.SMOOTH) \
                and form[0] == "lc":
            swap = form[1]
            line = ln(1) if swap else ln(0)
            circ = ci(0) if swap else ci(1)
            if line is None or circ is None:
                return
            a, b = line
            c, r = circ
            out.append(self.sign * cross(_dir_n(a, b), sub(c, a)) - r)
            return
        if kind in (ConstraintKind.TANGENT, ConstraintKind.SMOOTH) \
                and form[0] == "cc":
            c0, c1 = ci(0), ci(1)
            if c0 is None or c1 is None:
                return
            (p0, r0), (p1, r1) = c0, c1
            d = dist(p0, p1)
            if self.sign > 0:
                out.append(d - (r0 + r1))
            else:
                out.append(d - self.target * (r0 - r1))
            return
        if kind == ConstraintKind.SYMMETRIC and form[0] == "pp":
            p0, p1 = pt(0), pt(1)
            line = ln(2)
            if p0 is None or p1 is None or line is None:
                return
            d = _dir_n(*line)
            out.extend([cross(d, sub(mid(p0, p1), line[0])),
                        dot(d, sub(p1, p0))])
            return
        if kind == ConstraintKind.SYMMETRIC and form[0] == "symlines":
            crossed = form[1]
            l0, l1, line = ln(0), ln(1), ln(2)
            if l0 is None or l1 is None or line is None:
                return
            a2, b2 = l1
            if crossed:
                a2, b2 = b2, a2
            d = _dir_n(*line)
            out.extend([
                cross(d, sub(mid(l0[0], a2), line[0])),
                dot(d, sub(a2, l0[0])),
                cross(d, sub(mid(l0[1], b2), line[0])),
                dot(d, sub(b2, l0[1])),
            ])
            return
        if kind == ConstraintKind.SYMMETRIC:
            c0, c1 = ci(0), ci(1)
            line = ln(2)
            if c0 is None or c1 is None or line is None:
                return
            d = _dir_n(*line)
            out.extend([
                cross(d, sub(mid(c0[0], c1[0]), line[0])),
                dot(d, sub(c1[0], c0[0])),
                c0[1] - c1[1],
            ])
            return
        if kind in (ConstraintKind.DISTANCE_ALIGNED,
                    ConstraintKind.DISTANCE_HORIZONTAL,
                    ConstraintKind.DISTANCE_VERTICAL):
            if form[0] == "pl":
                swap = form[1]
                p = pt(1) if swap else pt(0)
                line = ln(0) if swap else ln(1)
                if p is None or line is None:
                    return
                a, b = line
                value = cross(_dir_n(a, b), sub(p, a))
            else:
                if form[0] == "l1":
                    a, b = ln(0)
                else:
                    a, b = pt(0), pt(1)
                if a is None or b is None:
                    return
                if kind == ConstraintKind.DISTANCE_ALIGNED:
                    value = dist(a, b)
                elif kind == ConstraintKind.DISTANCE_HORIZONTAL:
                    value = b[0] - a[0]
                else:
                    value = b[1] - a[1]
            out.append(self.sign * value - self.target)
            return
        if kind == ConstraintKind.ANGULAR and form[0] == "arcsweep":
            sweep = model.arc_sweep(x, self.refs[0])
            if sweep is None:
                return
            out.append(sweep - self.target)
            return
        if kind == ConstraintKind.ANGULAR and form[0] == "rays":
            rev = form[1]
            l0, l1 = ln(0), ln(1)
            if l0 is None or l1 is None:
                return
            d1 = _ray(l0[0], l0[1], rev[0])
            d2 = _ray(l1[0], l1[1], rev[1])
            residual = _wrap(math.atan2(cross(d1, d2), dot(d1, d2))
                             - self.sign * self.target)
            out.append(residual)
            return
        if kind == ConstraintKind.RADIUS:
            circ = ci(0)
            if circ is None:
                return
            out.append(circ[1] - self.target)
            return
        if kind == ConstraintKind.DIAMETER:
            circ = ci(0)
            if circ is None:
                return
            out.append(2.0 * circ[1] - self.target)
            return

    def measure(self, model: Model, x) -> float:
        clone = Built(id=self.id, kind=self.kind, refs=self.refs,
                      form=self.form, target=0.0, sign=self.sign,
                      rows=self.rows, aux_init=self.aux_init,
                      aux=self.aux)
        out: list[float] = []
        clone.eval(model, x, out)
        value = abs(out[0]) if out else 0.0
        if self.kind == ConstraintKind.ANGULAR:
            value = math.degrees(value)
        return value


# ── Levenberg–Marquardt (upstream lm.rs) ─────────────────────────

def _max_abs(v) -> float:
    best = 0.0
    for a in v:
        if math.isfinite(a):
            best = max(best, abs(a))
        else:
            return math.inf
    return best


def _norm2(v) -> float:
    return sum(a * a for a in v)


def jacobian(f, x, free, rows: int) -> list[float]:
    nf = len(free)
    jac = [0.0] * (rows * nf)
    xp = list(x)
    rp: list[float] = []
    rm: list[float] = []
    for k, j in enumerate(free):
        xj = x[j]
        h = 1e-7 * max(abs(xj), 1.0)
        xp[j] = xj + h
        rp.clear()
        f(xp, rp)
        xp[j] = xj - h
        rm.clear()
        f(xp, rm)
        xp[j] = xj
        for i in range(rows):
            d = ((rp[i] if i < len(rp) else 0.0)
                 - (rm[i] if i < len(rm) else 0.0)) / (2.0 * h)
            jac[i * nf + k] = d if math.isfinite(d) else 0.0
    return jac


def _solve_dense(a, b, n: int):
    scale = max((abs(v) for v in a), default=0.0)
    scale = max(scale, 1e-300)
    a = list(a)
    b = list(b)
    for col in range(n):
        piv, best = col, 0.0
        for r in range(col, n):
            v = abs(a[r * n + col])
            if v > best:
                best, piv = v, r
        if best <= scale * 1e-15:
            return None
        if piv != col:
            for c in range(n):
                a[piv * n + c], a[col * n + c] = \
                    a[col * n + c], a[piv * n + c]
            b[piv], b[col] = b[col], b[piv]
        p = a[col * n + col]
        for r in range(col + 1, n):
            f = a[r * n + col] / p
            if f == 0.0:
                continue
            for c in range(col, n):
                a[r * n + c] -= f * a[col * n + c]
            b[r] -= f * b[col]
    y = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = b[r]
        for c in range(r + 1, n):
            s -= a[r * n + c] * y[c]
        y[r] = s / a[r * n + r]
    return y if all(math.isfinite(v) for v in y) else None


def _reflect(v, vv, c) -> None:
    # v and c are both length-nf by construction at both call
    # sites (_qr builds reflectors of length len(cols[0]), and
    # null_space's e is length cols), so strict is exact.
    d = sum(a * b for a, b in zip(v, c, strict=True))
    f = 2.0 * d / vv
    if f != 0.0:
        for i in range(len(c)):
            c[i] -= f * v[i]


def _qr(cols, rel: float, abs_tol: float):
    """Householder QR with column pivoting. Returns (rank, reflectors)."""
    cols = [list(c) for c in cols]
    m = len(cols[0]) if cols else 0
    n = len(cols)
    refl: list[list[float]] = []
    first = None
    for k in range(min(m, n)):
        best_j, best_v = k, -1.0
        for j in range(k, n):
            v = sum(val * val for val in cols[j][k:])
            if math.isfinite(v) and v > best_v:
                best_j, best_v = j, v
        norm = math.sqrt(max(best_v, 0.0))
        lim = abs_tol if first is None else max(first * rel, abs_tol)
        if norm <= lim:
            break
        if first is None:
            first = norm
        cols[k], cols[best_j] = cols[best_j], cols[k]
        col = cols[k]
        v = [0.0] * m
        for i in range(k, m):
            v[i] = col[i]
        x0 = v[k]
        alpha = -norm if x0 >= 0.0 else norm
        v[k] -= alpha
        vv = sum(val * val for val in v)
        if vv <= 0.0:
            break
        for c in cols[k:]:
            _reflect(v, vv, c)
        refl.append(v)
    return len(refl), refl


def _balanced_rows(m, rows: int, cols: int):
    out = []
    for i in range(rows):
        row = list(m[i * cols:(i + 1) * cols])
        while len(row) < cols:
            row.append(0.0)
        n = math.sqrt(sum(v * v for v in row))
        f = 1.0 / n if n > 1e-12 and math.isfinite(n) else 0.0
        out.append([v * f for v in row])
    for c in range(cols):
        n = math.sqrt(sum(row[c] ** 2 for row in out))
        f = 1.0 / n if n > COL_NOISE and math.isfinite(n) else 0.0
        for row in out:
            row[c] *= f
    return out


def rank_qr(m, rows: int, cols: int) -> int:
    if rows == 0 or cols == 0:
        return 0
    return _qr(_balanced_rows(m, rows, cols), RANK_TOL, 1e-12)[0]


def null_space(m, rows: int, cols: int):
    """Rank and an orthonormal basis of the null space."""
    if rows == 0:
        rank, refl = 0, []
    else:
        rank, refl = _qr(_balanced_rows(m, rows, cols),
                         RANK_TOL, 1e-12)
    basis: list[list[float]] = []
    for j in range(rank, cols):
        e = [0.0] * cols
        e[j] = 1.0
        for v in reversed(refl):
            vv = sum(val * val for val in v)
            if vv > 0.0:
                _reflect(v, vv, e)
        basis.append(e)
    return rank, basis


def lm_solve(f, x0, free, weights, tol: float,
             max_iter: int) -> dict:
    """Damped weighted minimum-norm Gauss–Newton (upstream lm.rs)."""
    x = list(x0)
    r: list[float] = []
    f(x, r)
    rows = len(r)
    if rows == 0 or _max_abs(r) <= tol:
        return {"x": x, "converged": True}
    if not free or rows > MAX_ROWS:
        return {"x": x, "converged": False}
    nf = len(free)
    winv = [1.0 / max(weights[j] if j < len(weights) else 1.0, 1e-12)
            for j in free]
    mu = 1e-10
    cost = _norm2(r)
    it = 0
    rn: list[float] = []
    while it < max_iter:
        it += 1
        jac = jacobian(f, x, free, rows)
        a = [0.0] * (rows * rows)
        for i in range(rows):
            for k in range(i, rows):
                s = 0.0
                for c, w in enumerate(winv):
                    s += (jac[i * nf + c] * jac[k * nf + c] * w)
                a[i * rows + k] = s
                a[k * rows + i] = s
        diag_max = max((a[i * rows + i] for i in range(rows)),
                       default=0.0)
        diag_max = max(diag_max, 1e-12)
        improved = False
        for _ in range(12):
            ad = list(a)
            for i in range(rows):
                ad[i * rows + i] += mu * diag_max
            y = _solve_dense(ad, list(r), rows)
            if y is None:
                mu *= 10.0
                continue
            xn = list(x)
            for c, j in enumerate(free):
                s = 0.0
                for i, yi in enumerate(y):
                    s += jac[i * nf + c] * yi
                xn[j] -= winv[c] * s
            rn.clear()
            f(xn, rn)
            cn = _norm2(rn)
            if math.isfinite(cn) and cn < cost:
                x = xn
                r, rn = rn, r
                cost = cn
                mu = max(mu * 0.1, 1e-15)
                improved = True
                break
            mu *= 10.0
        if _max_abs(r) <= tol:
            return {"x": x, "converged": True}
        if not improved:
            break
    return {"x": x, "converged": _max_abs(r) <= tol}


# ── components + solve entry (upstream lib.rs) ────────────────────

@dataclass
class Comp:
    model: Model
    built: list[Built]
    weights: list[float]
    held: list[int]
    tol: float = TOL

    def free(self, skip: int | None = None) -> list[int]:
        held = set(self.held)
        return [j for j in range(len(self.model.x0))
                if j not in held]

    def rows(self, skip: int | None = None) -> int:
        return sum(b.rows for i, b in enumerate(self.built)
                   if skip is None or i != skip)

    def eval(self, x, out: list[float], skip: int | None = None) -> None:
        # aux parameters ride after the model parameters.
        full = list(x)
        for i, b in enumerate(self.built):
            if skip is not None and i == skip:
                continue
            b.eval(self.model, full, out)

    def run(self, skip: int | None = None) -> dict:
        free = self.free(skip)
        rows = self.rows(skip)
        if rows > MAX_ROWS:
            return {"x": list(self.model.x0), "converged": False}
        # Assemble the full vector: model params + aux params.
        x0 = list(self.model.x0)
        for b in self.built:
            b.aux = len(x0)
            x0.extend(b.aux_init)
        nf = len(free)
        eval_work = rows * nf * (1.0 + sum(
            self.model.spline_reconstruct_cost(s.handle)
            for s in self.model.slots))
        if eval_work > MAX_EVAL_WORK:
            return {"x": list(self.model.x0), "converged": False}
        weights = list(self.weights)
        while len(weights) < len(x0):
            weights.append(1.0)

        def f(xx, out):
            out.clear()
            self.eval(xx, out, skip)

        outcome = lm_solve(f, x0, free, weights, self.tol, MAX_ITER)
        return outcome


def _components(doc: CadDocument, constraints: list[Constraint],
                hold: set[str] | None):
    """Group constraints into connected components over handles."""
    parent: dict[str, str] = {}

    def find(h: str) -> str:
        parent.setdefault(h, h)
        while parent[h] != h:
            parent[h] = parent[parent[h]]
            h = parent[h]
        return h

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in constraints:
        handles = [r.handle for r in c.refs]
        for h in handles[1:]:
            union(handles[0], h)
    groups: dict[str, list[Constraint]] = {}
    for c in constraints:
        if not c.refs:
            continue
        groups.setdefault(find(c.refs[0].handle), []).append(c)
    return groups


def build(doc: CadDocument, constraints: list[Constraint],
          hold: set[str] | None = None):
    """Build solver components. Returns (comps, skipped) where
    skipped is [(constraint id, reason)]."""
    hold = hold or set()
    comps: list[Comp] = []
    skipped: list[tuple[int, str]] = []
    groups = _components(doc, constraints, hold)
    for group in groups.values():
        handles: list[str] = []
        for c in group:
            for r in c.refs:
                if r.handle not in handles:
                    handles.append(r.handle)
        model = Model.build(doc, handles)
        built: list[Built] = []
        held: set[int] = set()
        for handle in hold:
            slot = model.slot(handle)
            if slot is not None:
                held.update(range(slot.off, slot.off + slot.length))
        for c in group:
            target = c.value if c.value is not None else 0.0
            if c.kind == ConstraintKind.ANGULAR:
                target = c.value if c.value is not None else 0.0
            try:
                b = Built.new(c, model, model.x0, target)
            except CadError as exc:
                skipped.append((c.id, str(exc)))
                continue
            built.append(b)
            if b.form[0] == "lock":
                ref = c.refs[0]
                locked = model.lock_indices(ref)
                if locked:
                    held.update(locked)
        weights = [1.0] * len(model.x0)
        comps.append(Comp(model=model, built=built, weights=weights,
                          held=sorted(held)))
    return comps, skipped


def _point_slot(doc: CadDocument, ref: GeomRef):
    """A writable point location for a coincident reference, or
    None when the reference is not a plain point (an arc end or a
    line midpoint is derived, not stored)."""
    entity = doc.entities.get(ref.handle)
    if entity is None:
        return None
    sub = ref.sub
    if isinstance(entity, Line):
        if sub == Sub.START:
            return (entity, "a")
        if sub == Sub.END:
            return (entity, "b")
    elif isinstance(entity, (Circle, Ellipse)) and sub == Sub.CENTER:
        return (entity, "center")
    elif isinstance(entity, CadPoint):
        return (entity, "p")
    elif (isinstance(entity, Spline) and isinstance(sub, tuple)
            and sub[0] == "vertex"):
        index = sub[1]
        if 0 <= index < len(entity.points):
            return (entity, "spline", index)
    return None


def _slot_read(slot) -> tuple[float, float]:
    entity, kind = slot[0], slot[1]
    if kind == "spline":
        return entity.points[slot[2]]
    return getattr(entity, kind)


def _slot_write(slot, point: tuple[float, float]) -> None:
    entity, kind = slot[0], slot[1]
    if kind == "spline":
        entity.points[slot[2]] = point
    else:
        setattr(entity, kind, point)


def _snap_coincident_points(doc: CadDocument, comp: Comp) -> None:
    """Make coincident points EXACTLY one point.

    The solver writes each entity's parameters back independently,
    so two coincident endpoints land within the tolerance of each
    other but not bit-identical. A sketch promises that coincident
    points coincide exactly, so after a converged solve every
    connected group of coincident points is set to its centroid —
    one shared tuple for the whole group, chains included.
    """
    pairs: list[tuple[tuple, tuple]] = []
    slots_by_key: dict[tuple, tuple] = {}
    for built in comp.built:
        if built.kind != ConstraintKind.COINCIDENT \
                or len(built.refs) != 2:
            continue
        first = _point_slot(doc, built.refs[0])
        second = _point_slot(doc, built.refs[1])
        if first is None or second is None:
            continue
        key1 = (first[0].id, first[1],
                first[2] if len(first) > 2 else None)
        key2 = (second[0].id, second[1],
                second[2] if len(second) > 2 else None)
        slots_by_key[key1] = first
        slots_by_key[key2] = second
        pairs.append((key1, key2))
    parent = {key: key for key in slots_by_key}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key1, key2 in pairs:
        root1, root2 = find(key1), find(key2)
        if root1 != root2:
            parent[root1] = root2
    groups: dict[tuple, list] = {}
    for key, slot in slots_by_key.items():
        groups.setdefault(find(key), []).append(slot)
    for slots in groups.values():
        if len(slots) < 2:
            continue
        points = [_slot_read(slot) for slot in slots]
        centroid = (sum(p[0] for p in points) / len(points),
                    sum(p[1] for p in points) / len(points))
        for slot in slots:
            _slot_write(slot, centroid)
            doc._dirty_ids.add(slot[0].id)


def solve(doc: CadDocument, *, tol: float = TOL,
          max_iter: int = MAX_ITER,
          hold: set[str] | None = None) -> dict:
    """Solve every constraint component; write solved geometry
    back. Returns a SolveReport dict."""
    if len(doc.constraints) > MAX_CONSTRAINTS:
        raise CadError(
            f"too many constraints ({len(doc.constraints)} > "
            f"{MAX_CONSTRAINTS})", kind="solve")
    comps, skipped = build(doc, list(doc.constraints), hold)
    moved: list[str] = []
    conflicts: list[int] = []
    max_residual = 0.0
    converged = True
    for comp in comps:
        outcome = comp.run()
        x = outcome["x"]
        # Measure the residual at the outcome.
        probe: list[float] = []
        comp.eval(x, probe)
        res = _max_abs(probe)
        max_residual = max(max_residual, res)
        if not outcome["converged"]:
            converged = False
            # Name only the constraints that are actually violated
            # at the stalled iterate — not every constraint in the
            # component.
            for built in comp.built:
                rows: list[float] = []
                built.eval(comp.model, x, rows)
                if rows and _max_abs(rows) > 1e-7:
                    conflicts.append(built.id)
            continue
        moved.extend(comp.model.write_back(doc, x))
        _snap_coincident_points(doc, comp)
    report = dof_report(doc)
    return {"converged": converged,
            "max_residual": max_residual,
            "moved": sorted(set(moved)),
            "conflicts": sorted(set(conflicts)),
            "skipped": [{"id": i, "reason": r}
                        for i, r in skipped],
            "dof": report["dof"],
            "status": report["status"],
            "redundant": report["redundant"]}


def dof_report(doc: CadDocument,
               handles: list[str] | None = None) -> dict:
    """Degrees of freedom (upstream dof.rs)."""
    want_all = not handles
    want = set(handles or [])
    comps, _ = build(doc, list(doc.constraints))
    objects: list[dict] = []
    seen: set[str] = set()
    total: int | None = 0
    redundant = 0
    constraint_count = 0
    any_constrained = False
    for comp in comps:
        if not any(want_all or s.handle in want
                   for s in comp.model.slots):
            continue
        any_constrained = True
        free = comp.free()
        rows = comp.rows()
        nf = len(free)
        unknown = {"dof": None, "redundant": 0,
                   "objects": [None] * len(comp.model.slots)}
        if rows > MAX_ROWS or (rows * nf * min(rows, nf)
                              > MAX_RANK_WORK):
            analysis = unknown
        else:
            eval_work = rows * nf * (1.0 + sum(
                comp.model.spline_reconstruct_cost(s.handle)
                for s in comp.model.slots))
            if eval_work > MAX_EVAL_WORK:
                analysis = unknown
            else:
                def f(xx, out, _comp=comp):
                    out.clear()
                    _comp.eval(xx, out)

                jac = jacobian(f, comp.model.x0, free, rows)
                if nf > MAX_NULL_COLS:
                    rank = rank_qr(jac, rows, nf)
                    analysis = {"dof": nf - min(rank, nf),
                                "redundant": rows - rank,
                                "objects": [None]
                                * len(comp.model.slots)}
                else:
                    rank, basis = null_space(jac, rows, nf)
                    per_object: list[int | None] = []
                    for s in comp.model.slots:
                        cols = [k for k, j in enumerate(free)
                                if s.off <= j < s.off + s.length]
                        if not cols or not basis:
                            per_object.append(0)
                            continue
                        restricted = [[v[k] for k in cols]
                                      for v in basis]
                        per_object.append(
                            _qr(restricted, 0.0, 1e-6)[0])
                    analysis = {"dof": nf - min(rank, nf),
                                "redundant": rows - rank,
                                "objects": per_object}
        constraint_count += len(comp.built)
        redundant += analysis["redundant"]
        if total is not None and analysis["dof"] is not None:
            total += analysis["dof"]
        else:
            total = None
        for s, od in zip(comp.model.slots, analysis["objects"],
                         strict=True):
            if not (want_all or s.handle in want):
                continue
            if s.handle in seen or len(objects) >= 10_000:
                continue
            seen.add(s.handle)
            if analysis["redundant"] > 0:
                status = "over"
            elif od == 0:
                status = "fully"
            elif od is None:
                status = "unknown"
            else:
                status = "under"
            entity = doc.entities.get(s.handle)
            kind = type(entity).__name__ if entity else ""
            objects.append({"handle": s.handle, "type": kind,
                            "params": s.length, "dof": od,
                            "status": status})
    free_objects = False
    # Entities in no component are fully free — but the loop below
    # used to iterate `want`, which is EMPTY when every handle is
    # wanted, so a drawing with no constraints at all reported
    # dof 0 instead of its full parameter count. With want_all,
    # walk every entity in the document.
    free_handles = list(doc.entities) if want_all else list(want)
    for handle in free_handles[:10_000]:
        if handle in seen:
            continue
        model = Model.build(doc, [handle])
        if not model.slots:
            continue
        seen.add(handle)
        free_objects = True
        slot = model.slots[0]
        if total is not None:
            total += slot.length
        entity = doc.entities.get(handle)
        kind = type(entity).__name__ if entity else ""
        objects.append({"handle": handle, "type": kind,
                        "params": slot.length,
                        "dof": slot.length, "status": "unconstrained"})
    if redundant > 0:
        status = "over"
    elif not any_constrained:
        status = "unconstrained"
    elif total == 0:
        status = "fully"
    elif total is None:
        status = "unknown"
    else:
        status = "under"
    if status == "unconstrained" and not free_objects:
        message = "No constraints"
    elif status == "over":
        message = (f"Over-constrained: {redundant} redundant "
                   f"{'equation' if redundant == 1 else 'equations'}")
    elif status == "fully":
        message = "Fully constrained"
    elif total is not None:
        message = (f"{total} remaining "
                   f"{'degree of freedom' if total == 1 else 'degrees of freedom'}")
    else:
        message = "Too large to analyse"
    return {"dof": total, "status": status,
            "redundant": redundant, "constraints": constraint_count,
            "objects": objects, "message": message}


def validate_constraint(doc: CadDocument,
                        constraint: Constraint) -> str | None:
    """Interpret a constraint against the current geometry.
    Returns an error message, or None when it is valid."""
    handles = [r.handle for r in constraint.refs]
    model = Model.build(doc, handles)
    target = constraint.value if constraint.value is not None else 0.0
    try:
        Built.new(constraint, model, model.x0, target)
    except CadError as exc:
        return str(exc)
    return None


def purge_invalid(doc: CadDocument) -> list[int]:
    """Remove constraints whose references are missing or that
    cannot be interpreted. Returns the removed ids."""
    removed: list[int] = []
    kept: list[Constraint] = []
    for c in doc.constraints:
        if any(r.handle not in doc.entities for r in c.refs):
            removed.append(c.id)
            continue
        if validate_constraint(doc, c) is not None:
            removed.append(c.id)
            continue
        kept.append(c)
    doc.constraints = kept
    return removed


def measure(doc: CadDocument, constraint: Constraint) -> float:
    """The current measured value of a dimensional constraint
    (degrees for angular)."""
    handles = [r.handle for r in constraint.refs]
    model = Model.build(doc, handles)
    # A probe has no target value; Built.new still validates the
    # target for dimensional kinds, so use a positive placeholder —
    # Built.measure re-clones with target 0.0, so the placeholder
    # never reaches the result.
    target = constraint.value if constraint.value is not None else 1.0
    built = Built.new(constraint, model, model.x0, target)
    return built.measure(model, model.x0)
