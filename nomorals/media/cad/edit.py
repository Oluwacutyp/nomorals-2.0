"""The CadEditor: add/edit/move entities, apply constraints and
dimensions, and the real drafting operations (trim, extend, offset,
fillet, chamfer, mirror, array) computed from real geometry.

Every operation that can fail returns a clear answer instead of
corrupting the drawing: ``trim``/``extend`` return False when there
is nothing to cut to, ``offset``/``fillet``/``chamfer`` return None
when the geometry does not admit the operation — and the document is
left exactly as it was. ``remove`` and the bad-geometry ``add_*``
calls raise :class:`~nomorals.media.cad.errors.CadError` naming the
problem, before anything is added.
"""
from __future__ import annotations

import math
from itertools import pairwise

from .errors import CadError
from .geom import (
    arc_contains,
    circle_circle,
    dist,
    line_circle,
    line_line,
    line_line_infinite,
    norm_angle,
)
from .model import (
    Arc,
    CadDocument,
    CadPoint,
    CadText,
    Circle,
    Constraint,
    ConstraintKind,
    Dimension,
    Ellipse,
    GeomRef,
    Insert,
    Layer,
    Line,
    Polyline,
    Spline,
    Sub,
)
from .solver import measure, validate_constraint


def _point(value: object) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    raise CadError(f"expected a point (x, y), got {value!r}",
                   kind="invalid_argument")


def _angle_degrees(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CadError(f"expected an angle in degrees, got {value!r}",
                       kind="invalid_argument")
    return float(value)


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _param_on_line(a: tuple[float, float], b: tuple[float, float],
                   p: tuple[float, float]) -> float:
    """Parameter t of the projection of p onto the line a–b."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom == 0:
        return 0.0
    return ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom


class CadEditor:
    """Edits one :class:`CadDocument`, tracking touched ids for the
    fidelity report (``doc._dirty_ids``)."""

    def __init__(self, doc: CadDocument) -> None:
        self.doc = doc

    # ── adding entities ──────────────────────────────────────────

    def add_line(self, a, b) -> str:
        a, b = _point(a), _point(b)
        if a == b:
            raise CadError("zero-length line", kind="invalid_argument")
        entity = Line(a=a, b=b)
        return self._add(entity)

    def add_arc(self, center, radius: float, start: float,
                end: float) -> str:
        center = _point(center)
        if radius <= 0:
            raise CadError(f"arc radius must be positive, got {radius}",
                           kind="invalid_argument")
        entity = Arc(center=center, radius=float(radius),
                     start=float(start), end=float(end))
        return self._add(entity)

    def add_circle(self, center, radius: float) -> str:
        center = _point(center)
        if radius <= 0:
            raise CadError(
                f"circle radius must be positive, got {radius}",
                kind="invalid_argument")
        return self._add(Circle(center=center, radius=float(radius)))

    def add_ellipse(self, center, major, ratio: float) -> str:
        center, major = _point(center), _point(major)
        if math.hypot(*major) <= 0:
            raise CadError("ellipse major axis must be non-zero",
                           kind="invalid_argument")
        if ratio <= 0:
            raise CadError(f"ellipse ratio must be positive, got {ratio}",
                           kind="invalid_argument")
        return self._add(Ellipse(center=center, major=major,
                                 ratio=float(ratio)))

    def add_spline(self, points) -> str:
        pts = [_point(p) for p in points]
        if len(pts) < 2:
            raise CadError("a spline needs at least two points",
                           kind="invalid_argument")
        return self._add(Spline(points=pts))

    def add_point(self, p) -> str:
        return self._add(CadPoint(p=_point(p)))

    def add_polyline(self, vertices, closed: bool = False) -> str:
        verts = [_point(v) for v in vertices]
        if len(verts) < 2:
            raise CadError("a polyline needs at least two vertices",
                           kind="invalid_argument")
        return self._add(Polyline(vertices=verts, closed=bool(closed)))

    def add_text(self, position, text: str, height: float = 2.5) -> str:
        if height <= 0:
            raise CadError(f"text height must be positive, got {height}",
                           kind="invalid_argument")
        return self._add(CadText(position=_point(position), text=str(text),
                                 height=float(height)))

    def _add(self, entity) -> str:
        entity_id = self.doc.add_entity(entity)
        self.doc._dirty_ids.add(entity_id)
        return entity_id

    def _touch(self, entity_id: str) -> None:
        self.doc._dirty_ids.add(entity_id)

    def _get(self, entity_id: str):
        entity = self.doc.entities.get(entity_id)
        if entity is None:
            raise CadError(f"no entity {entity_id!r}",
                           kind="not_found")
        return entity

    # ── rigid transforms ─────────────────────────────────────────

    def move(self, ids, dx: float, dy: float) -> None:
        for entity_id in ids:
            entity = self._get(entity_id)
            self._move_entity(entity, float(dx), float(dy))
            self._touch(entity_id)

    @staticmethod
    def _move_entity(entity, dx: float, dy: float) -> None:
        if isinstance(entity, Line):
            entity.a = (entity.a[0] + dx, entity.a[1] + dy)
            entity.b = (entity.b[0] + dx, entity.b[1] + dy)
        elif isinstance(entity, (Arc, Circle, Ellipse)):
            entity.center = (entity.center[0] + dx, entity.center[1] + dy)
        elif isinstance(entity, Spline):
            entity.points = [(x + dx, y + dy) for x, y in entity.points]
        elif isinstance(entity, CadPoint):
            entity.p = (entity.p[0] + dx, entity.p[1] + dy)
        elif isinstance(entity, Polyline):
            entity.vertices = [(x + dx, y + dy)
                               for x, y in entity.vertices]
        elif isinstance(entity, (CadText, Dimension, Insert)):
            entity.position = (entity.position[0] + dx,
                               entity.position[1] + dy)

    def rotate(self, ids, center, angle_deg: float) -> None:
        centre = _point(center)
        angle = math.radians(_angle_degrees(angle_deg))
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def rot(p):
            x, y = p[0] - centre[0], p[1] - centre[1]
            return (centre[0] + x * cos_a - y * sin_a,
                    centre[1] + x * sin_a + y * cos_a)

        for entity_id in ids:
            entity = self._get(entity_id)
            self._rotate_entity(entity, rot, angle)
            self._touch(entity_id)

    @staticmethod
    def _rotate_entity(entity, rot, angle: float) -> None:
        if isinstance(entity, Line):
            entity.a, entity.b = rot(entity.a), rot(entity.b)
        elif isinstance(entity, Arc):
            entity.center = rot(entity.center)
            entity.start = norm_angle(entity.start + angle)
            entity.end = norm_angle(entity.end + angle)
        elif isinstance(entity, Circle):
            entity.center = rot(entity.center)
        elif isinstance(entity, Ellipse):
            entity.center = rot(entity.center)
            entity.major = (entity.major[0] * math.cos(angle)
                            - entity.major[1] * math.sin(angle),
                            entity.major[0] * math.sin(angle)
                            + entity.major[1] * math.cos(angle))
            entity.start = norm_angle(entity.start + angle)
            entity.end = norm_angle(entity.end + angle)
        elif isinstance(entity, Spline):
            entity.points = [rot(p) for p in entity.points]
        elif isinstance(entity, CadPoint):
            entity.p = rot(entity.p)
        elif isinstance(entity, Polyline):
            entity.vertices = [rot(p) for p in entity.vertices]
        elif isinstance(entity, CadText):
            entity.position = rot(entity.position)
            # CadText.rotation is stored in degrees.
            entity.rotation = (entity.rotation
                               + math.degrees(angle)) % 360.0
        elif isinstance(entity, Dimension):
            entity.position = rot(entity.position)
        elif isinstance(entity, Insert):
            entity.position = rot(entity.position)
            entity.rotation = norm_angle(entity.rotation + angle)

    def scale(self, ids, center, factor: float) -> None:
        centre = _point(center)
        if factor <= 0:
            raise CadError(f"scale factor must be positive, got {factor}",
                           kind="invalid_argument")
        factor = float(factor)

        def sc(p):
            return (centre[0] + (p[0] - centre[0]) * factor,
                    centre[1] + (p[1] - centre[1]) * factor)

        for entity_id in ids:
            entity = self._get(entity_id)
            self._scale_entity(entity, sc, factor)
            self._touch(entity_id)

    @staticmethod
    def _scale_entity(entity, sc, factor: float) -> None:
        if isinstance(entity, Line):
            entity.a, entity.b = sc(entity.a), sc(entity.b)
        elif isinstance(entity, (Arc, Circle)):
            entity.center = sc(entity.center)
            entity.radius *= factor
        elif isinstance(entity, Ellipse):
            entity.center = sc(entity.center)
            entity.major = (entity.major[0] * factor,
                            entity.major[1] * factor)
        elif isinstance(entity, Spline):
            entity.points = [sc(p) for p in entity.points]
        elif isinstance(entity, CadPoint):
            entity.p = sc(entity.p)
        elif isinstance(entity, Polyline):
            entity.vertices = [sc(p) for p in entity.vertices]
        elif isinstance(entity, CadText):
            entity.position = sc(entity.position)
            entity.height *= factor
        elif isinstance(entity, Dimension):
            entity.position = sc(entity.position)
            entity.value *= factor
        elif isinstance(entity, Insert):
            entity.position = sc(entity.position)
            entity.scale = (entity.scale[0] * factor,
                            entity.scale[1] * factor)

    # ── copy operations ──────────────────────────────────────────

    def mirror(self, ids, p1, p2) -> list[str]:
        a, b = _point(p1), _point(p2)
        if a == b:
            raise CadError("mirror line must have two distinct points",
                           kind="invalid_argument")
        out: list[str] = []
        for entity_id in ids:
            entity = self._get(entity_id)
            clone = self._clone_mirrored(entity, a, b)
            out.append(self._add(clone))
        return out

    @staticmethod
    def _mirror_point(p, a, b):
        d = (b[0] - a[0], b[1] - a[1])
        length = math.hypot(*d)
        d = (d[0] / length, d[1] / length)
        rel = (p[0] - a[0], p[1] - a[1])
        t = rel[0] * d[0] + rel[1] * d[1]
        proj = (a[0] + d[0] * t, a[1] + d[1] * t)
        return (2 * proj[0] - p[0], 2 * proj[1] - p[1])

    def _clone_mirrored(self, entity, a, b):
        import copy
        clone = copy.deepcopy(entity)

        def mirror(p):
            return self._mirror_point(p, a, b)

        if isinstance(clone, Line):
            clone.a, clone.b = mirror(clone.a), mirror(clone.b)
        elif isinstance(clone, Arc):
            clone.center = mirror(clone.center)
            # Mirroring flips the sweep: swap the ends.
            clone.start, clone.end = clone.end, clone.start
        elif isinstance(clone, Circle):
            clone.center = mirror(clone.center)
        elif isinstance(clone, Ellipse):
            clone.center = mirror(clone.center)
            # Reflect the major-axis vector across the mirror line.
            d = (b[0] - a[0], b[1] - a[1])
            length = math.hypot(*d)
            d = (d[0] / length, d[1] / length)
            m = clone.major
            t = m[0] * d[0] + m[1] * d[1]
            clone.major = (2 * d[0] * t - m[0], 2 * d[1] * t - m[1])
        elif isinstance(clone, Spline):
            clone.points = [mirror(p) for p in clone.points]
        elif isinstance(clone, CadPoint):
            clone.p = mirror(clone.p)
        elif isinstance(clone, Polyline):
            clone.vertices = [mirror(p) for p in clone.vertices]
        elif isinstance(clone, (CadText, Dimension, Insert)):
            clone.position = mirror(clone.position)
        return clone

    def array_rect(self, ids, rows: int, cols: int,
                   dx: float, dy: float) -> list[str]:
        if rows < 1 or cols < 1:
            raise CadError("array needs at least 1 row and 1 column",
                           kind="invalid_argument")
        out: list[str] = []
        for entity_id in ids:
            entity = self._get(entity_id)
            for row in range(rows):
                for col in range(cols):
                    if row == 0 and col == 0:
                        continue
                    import copy
                    clone = copy.deepcopy(entity)
                    self._move_entity(clone, col * dx, row * dy)
                    out.append(self._add(clone))
        return out

    def array_polar(self, ids, center, count: int,
                    angle_deg: float) -> list[str]:
        if count < 2:
            raise CadError("a polar array needs at least 2 copies",
                           kind="invalid_argument")
        centre = _point(center)
        step = math.radians(float(angle_deg)) / count
        out: list[str] = []
        for entity_id in ids:
            entity = self._get(entity_id)
            for i in range(1, count):
                import copy
                clone = copy.deepcopy(entity)
                angle = step * i
                cos_a, sin_a = math.cos(angle), math.sin(angle)

                def rot(p, _c=centre, _ca=cos_a, _sa=sin_a):
                    x, y = p[0] - _c[0], p[1] - _c[1]
                    return (_c[0] + x * _ca - y * _sa,
                            _c[1] + x * _sa + y * _ca)

                self._rotate_entity(clone, rot, angle)
                out.append(self._add(clone))
        return out

    # ── drafting operations ──────────────────────────────────────

    def _cut_points(self, entity_id: str) -> list[tuple[float, object]]:
        """Intersections of one entity with every other entity, as
        (parameter, point) pairs along the entity."""
        entity = self._get(entity_id)
        cuts: list[tuple[float, object]] = []
        for other_id, other in self.doc.entities.items():
            if other_id == entity_id:
                continue
            points = self._intersections(entity, other)
            for point in points:
                param = self._param(entity, point)
                if param is not None:
                    cuts.append((param, point))
        cuts.sort(key=lambda item: item[0])
        return cuts

    @staticmethod
    def _intersections(first, second) -> list[tuple[float, float]]:
        if isinstance(first, Line) and isinstance(second, Line):
            hit = line_line(first.a, first.b, second.a, second.b)
            return [hit] if hit is not None else []
        if isinstance(first, Line) and isinstance(second, Circle):
            return line_circle(first.a, first.b, second.center,
                               second.radius)
        if isinstance(first, Circle) and isinstance(second, Line):
            return line_circle(second.a, second.b, first.center,
                               first.radius)
        if isinstance(first, Circle) and isinstance(second, Circle):
            return circle_circle(first.center, first.radius,
                                 second.center, second.radius)
        if isinstance(first, Line) and isinstance(second, Arc):
            pts = line_circle(first.a, first.b, second.center,
                              second.radius)
            return [p for p in pts
                    if arc_contains(second.start, second.end,
                                    math.atan2(p[1] - second.center[1],
                                               p[0] - second.center[0]))]
        if isinstance(first, Arc) and isinstance(second, Line):
            return CadEditor._intersections(second, first)
        if isinstance(first, Line) and isinstance(second, Polyline):
            out = []
            for a, b in pairwise(second.vertices):
                hit = line_line(first.a, first.b, a, b)
                if hit is not None:
                    out.append(hit)
            return out
        if isinstance(first, Polyline) and isinstance(second, Line):
            return CadEditor._intersections(second, first)
        return []

    @staticmethod
    def _param(entity, point) -> float | None:
        if isinstance(entity, Line):
            return _param_on_line(entity.a, entity.b, point)
        if isinstance(entity, Circle):
            return math.atan2(point[1] - entity.center[1],
                              point[0] - entity.center[0])
        if isinstance(entity, Arc):
            return math.atan2(point[1] - entity.center[1],
                              point[0] - entity.center[0])
        return None

    def trim(self, entity_id: str, pick) -> bool:
        """Cut the entity at the nearest boundaries and remove the
        piece the pick point lies on. Returns False when nothing
        cuts the entity here; the document is untouched then."""
        pick = _point(pick)
        entity = self._get(entity_id)
        if isinstance(entity, Line):
            return self._trim_line(entity_id, entity, pick)
        if isinstance(entity, Circle):
            return self._trim_circle(entity_id, entity, pick)
        if isinstance(entity, Arc):
            return self._trim_arc(entity_id, entity, pick)
        return False

    def _trim_line(self, entity_id, line: Line, pick) -> bool:
        """Trim a line: cut at every intersection, drop the piece
        the pick point lies on, keep the rest."""
        cuts = [(t, p) for t, p in self._cut_points(entity_id)
                if -1e-9 <= t <= 1.0 + 1e-9]
        if not cuts:
            return False
        a, b = line.a, line.b
        pick_t = _param_on_line(a, b, pick)
        bounds = [0.0] + [t for t, _ in cuts] + [1.0]
        pieces = [(bounds[i], bounds[i + 1])
                  for i in range(len(bounds) - 1)
                  if bounds[i + 1] - bounds[i] > 1e-12]
        keep = [piece for piece in pieces
                if not (piece[0] - 1e-9 <= pick_t <= piece[1] + 1e-9)]
        if len(keep) == len(pieces):
            return False
        if not keep:
            self.remove([entity_id])
            return True
        first = keep[0]
        line.a = (a[0] + first[0] * (b[0] - a[0]),
                  a[1] + first[0] * (b[1] - a[1]))
        line.b = (a[0] + first[1] * (b[0] - a[0]),
                  a[1] + first[1] * (b[1] - a[1]))
        self._touch(entity_id)
        if len(keep) > 1:
            # The pick removed a middle piece: the kept tail becomes
            # a second line.
            second = keep[1]
            tail_a = (a[0] + second[0] * (b[0] - a[0]),
                      a[1] + second[0] * (b[1] - a[1]))
            tail_b = (a[0] + second[1] * (b[0] - a[0]),
                      a[1] + second[1] * (b[1] - a[1]))
            self.add_line(tail_a, tail_b)
        return True

    def _trim_circle(self, entity_id, circle: Circle, pick) -> bool:
        cuts = self._cut_points(entity_id)
        if not cuts:
            return False
        pick_angle = math.atan2(pick[1] - circle.center[1],
                                pick[0] - circle.center[0])
        angles = sorted(norm_angle(t) for t, _ in cuts)
        if len(angles) == 1:
            return False
        # The kept arc runs from the cut after the pick round to the
        # cut before it.
        after = [a for a in angles if a > pick_angle] or angles
        start = after[0]
        before = [a for a in angles if a < pick_angle] or angles
        end = before[-1]
        arc = Arc(center=circle.center, radius=circle.radius,
                  start=start, end=end, layer=circle.layer,
                  color=circle.color)
        arc_id = self.doc.add_entity(arc)
        self.doc._dirty_ids.add(arc_id)
        self.remove([entity_id])
        return True

    def _trim_arc(self, entity_id, arc: Arc, pick) -> bool:
        cuts = [(t, p) for t, p in self._cut_points(entity_id)
                if arc_contains(arc.start, arc.end, t)]
        if not cuts:
            return False
        pick_angle = math.atan2(pick[1] - arc.center[1],
                                pick[0] - arc.center[0])
        if not arc_contains(arc.start, arc.end, pick_angle):
            return False
        # Keep the arc piece that does not contain the pick.
        sweep = (norm_angle(arc.end) - norm_angle(arc.start)) % math.tau
        if sweep == 0:
            sweep = math.tau
        rel_pick = (pick_angle - norm_angle(arc.start)) % math.tau
        rel_cuts = sorted((norm_angle(t) - norm_angle(arc.start))
                          % math.tau for t, _ in cuts)
        bounds = [0.0] + rel_cuts + [sweep]
        pieces = [(bounds[i], bounds[i + 1])
                  for i in range(len(bounds) - 1)
                  if bounds[i + 1] - bounds[i] > 1e-12]
        keep = [piece for piece in pieces
                if not (piece[0] - 1e-9 <= rel_pick <= piece[1] + 1e-9)]
        if len(keep) == len(pieces) or not keep:
            return False
        first = keep[0]
        start0 = norm_angle(arc.start)
        arc.start = norm_angle(start0 + first[0])
        arc.end = norm_angle(start0 + first[1])
        self._touch(entity_id)
        return True

    def extend(self, entity_id: str, pick) -> bool:
        """Extend the entity's nearer end to the nearest boundary in
        the pick's direction. Returns False when no boundary lies
        that way."""
        pick = _point(pick)
        entity = self._get(entity_id)
        if not isinstance(entity, Line):
            return False
        a, b = entity.a, entity.b
        # The end nearer the pick is the one that grows.
        if _dist2(pick, b) <= _dist2(pick, a):
            origin, direction = b, (b[0] - a[0], b[1] - a[1])
            grow_b = True
        else:
            origin, direction = a, (a[0] - b[0], a[1] - b[1])
            grow_b = False
        length = math.hypot(*direction)
        if length == 0:
            return False
        direction = (direction[0] / length, direction[1] / length)
        best: tuple[float, tuple[float, float]] | None = None
        for other_id, other in self.doc.entities.items():
            if other_id == entity_id:
                continue
            for point in self._ray_hits(origin, direction, other):
                t = ((point[0] - origin[0]) * direction[0]
                     + (point[1] - origin[1]) * direction[1])
                if t > 1e-9 and (best is None or t < best[0]):
                    best = (t, point)
        if best is None:
            return False
        if grow_b:
            entity.b = best[1]
        else:
            entity.a = best[1]
        self._touch(entity_id)
        return True

    def _ray_hits(self, origin, direction, other):
        """Points where the ray from origin along direction meets
        another entity (segment bounds respected)."""
        if isinstance(other, Line):
            # The ray is infinite; the boundary segment is finite.
            found = line_line_infinite(
                origin, (origin[0] + direction[0],
                         origin[1] + direction[1]),
                other.a, other.b)
            if found is None:
                return []
            hit, t_ray, u_seg = found
            if t_ray > 1e-9 and -1e-9 <= u_seg <= 1.0 + 1e-9:
                return [hit]
            return []
        if isinstance(other, Circle):
            pts = line_circle(origin,
                              (origin[0] + direction[0] * 1e6,
                               origin[1] + direction[1] * 1e6),
                              other.center, other.radius)
            return [p for p in pts
                    if ((p[0] - origin[0]) * direction[0]
                        + (p[1] - origin[1]) * direction[1]) > 1e-9]
        if isinstance(other, Polyline):
            out = []
            for a, b in pairwise(other.vertices):
                out.extend(self._ray_hits(origin, direction,
                                          Line(a=a, b=b)))
            return out
        return []

    def offset(self, entity_id: str, distance: float,
               side_point) -> str | None:
        """Copy the entity at ``distance`` towards ``side_point``.
        Returns the new id, or None when the type cannot be offset
        (or the offset would collapse it)."""
        side = _point(side_point)
        entity = self._get(entity_id)
        if isinstance(entity, Line):
            d = (entity.b[0] - entity.a[0], entity.b[1] - entity.a[1])
            length = math.hypot(*d)
            if length == 0:
                return None
            d = (d[0] / length, d[1] / length)
            n = (-d[1], d[0])
            to_side = (side[0] - entity.a[0], side[1] - entity.a[1])
            sign = 1.0 if to_side[0] * n[0] + to_side[1] * n[1] >= 0 \
                else -1.0
            shift = (n[0] * distance * sign, n[1] * distance * sign)
            return self.add_line(
                (entity.a[0] + shift[0], entity.a[1] + shift[1]),
                (entity.b[0] + shift[0], entity.b[1] + shift[1]))
        if isinstance(entity, Circle):
            gap = dist(side, entity.center)
            if gap > entity.radius:
                radius = entity.radius + abs(distance)
            else:
                radius = entity.radius - abs(distance)
                if radius <= 0:
                    return None
            return self.add_circle(entity.center, radius)
        if isinstance(entity, Arc):
            gap = dist(side, entity.center)
            if gap > entity.radius:
                radius = entity.radius + abs(distance)
            else:
                radius = entity.radius - abs(distance)
                if radius <= 0:
                    return None
            return self.add_arc(entity.center, radius, entity.start,
                                entity.end)
        if isinstance(entity, Polyline):
            # Offset each vertex along the average of its two edge
            # normals (miter join); the side point picks the sign.
            verts = entity.vertices
            if len(verts) < 2:
                return None
            out: list[tuple[float, float]] = []
            for i, vertex in enumerate(verts):
                prev_v = verts[i - 1] if i > 0 else verts[0]
                next_v = verts[i + 1] if i < len(verts) - 1 \
                    else verts[-1]
                e1 = (vertex[0] - prev_v[0], vertex[1] - prev_v[1])
                e2 = (next_v[0] - vertex[0], next_v[1] - vertex[1])
                n1 = (-e1[1], e1[0])
                n2 = (-e2[1], e2[0])
                for n in (n1, n2):
                    length = math.hypot(*n)
                    if length:
                        n = (n[0] / length, n[1] / length)
                n = (n1[0] + n2[0], n1[1] + n2[1])
                length = math.hypot(*n)
                if length == 0:
                    return None
                n = (n[0] / length, n[1] / length)
                to_side = (side[0] - vertex[0], side[1] - vertex[1])
                sign = 1.0 if to_side[0] * n[0] + to_side[1] * n[1] >= 0 \
                    else -1.0
                out.append((vertex[0] + n[0] * distance * sign,
                            vertex[1] + n[1] * distance * sign))
            return self.add_polyline(out, closed=entity.closed)
        return None

    def fillet(self, id1: str, id2: str, radius: float,
               pick1, pick2) -> str | None:
        """Round the corner between two lines with an arc of
        ``radius``. Returns the arc id, or None when the corner does
        not admit a fillet of that radius (document unchanged)."""
        pick1, pick2 = _point(pick1), _point(pick2)
        first = self._get(id1)
        second = self._get(id2)
        if not isinstance(first, Line) or not isinstance(second, Line):
            return None
        if radius <= 0:
            return None
        corner = line_line(first.a, first.b, second.a, second.b)
        if corner is None:
            return None
        # The corner end of each line is the endpoint at the corner;
        # the kept end is the other one. The pick points sit on the
        # kept side near the corner, as in the test drawings.
        end1 = first.b if _dist2(first.b, corner) < _dist2(first.a, corner) \
            else first.a
        end2 = second.b if _dist2(second.b, corner) < _dist2(second.a, corner) \
            else second.a
        keep1 = first.a if end1 == first.b else first.b
        keep2 = second.a if end2 == second.b else second.b
        # Analytic line–line fillet: offset both lines by the radius
        # towards each other's kept side and intersect the offsets.
        u1 = (keep1[0] - corner[0], keep1[1] - corner[1])
        u2 = (keep2[0] - corner[0], keep2[1] - corner[1])
        l1, l2 = math.hypot(*u1), math.hypot(*u2)
        if l1 == 0 or l2 == 0:
            return None
        u1 = (u1[0] / l1, u1[1] / l1)
        u2 = (u2[0] / l2, u2[1] / l2)
        # Normals pointing from each line towards the other line's
        # kept end.
        n1 = (-u1[1], u1[0])
        if (n1[0] * (keep2[0] - corner[0])
                + n1[1] * (keep2[1] - corner[1])) < 0:
            n1 = (-n1[0], -n1[1])
        n2 = (-u2[1], u2[0])
        if (n2[0] * (keep1[0] - corner[0])
                + n2[1] * (keep1[1] - corner[1])) < 0:
            n2 = (-n2[0], -n2[1])
        off1_a = (corner[0] + n1[0] * radius,
                  corner[1] + n1[1] * radius)
        off1_b = (off1_a[0] + u1[0], off1_a[1] + u1[1])
        off2_a = (corner[0] + n2[0] * radius,
                  corner[1] + n2[1] * radius)
        off2_b = (off2_a[0] + u2[0], off2_a[1] + u2[1])
        # The offsets are infinite lines; the centre is where they
        # cross (the short segments above only carry direction).
        found = line_line_infinite(off1_a, off1_b, off2_a, off2_b)
        if found is None:
            # Parallel lines: no corner to round.
            return None
        centre = found[0]
        # Tangent points: feet of the perpendiculars from the centre.
        t1 = _param_on_line(first.a, first.b, centre)
        t2 = _param_on_line(second.a, second.b, centre)
        if not (0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0):
            return None
        p1 = (first.a[0] + t1 * (first.b[0] - first.a[0]),
              first.a[1] + t1 * (first.b[1] - first.a[1]))
        p2 = (second.a[0] + t2 * (second.b[0] - second.a[0]),
              second.a[1] + t2 * (second.b[1] - second.a[1]))
        # The tangent points must sit on the kept side of the corner.
        if dist(corner, p1) > dist(corner, keep1) + 1e-9 \
                or dist(corner, p2) > dist(corner, keep2) + 1e-9:
            return None
        start = math.atan2(p1[1] - centre[1], p1[0] - centre[0])
        end = math.atan2(p2[1] - centre[1], p2[0] - centre[0])
        arc_id = self.add_arc(centre, radius, start, end)
        if first.b == end1:
            first.b = p1
        else:
            first.a = p1
        if second.b == end2:
            second.b = p2
        else:
            second.a = p2
        self._touch(id1)
        self._touch(id2)
        return arc_id

    def chamfer(self, id1: str, id2: str, d1: float, d2: float,
                pick1, pick2) -> str | None:
        """Cut the corner between two lines with a straight segment
        ``d1`` from the corner along the first line and ``d2`` along
        the second. Returns the new segment's id, or None when the
        corner does not admit it (document unchanged)."""
        pick1, pick2 = _point(pick1), _point(pick2)
        first = self._get(id1)
        second = self._get(id2)
        if not isinstance(first, Line) or not isinstance(second, Line):
            return None
        if d1 <= 0 or d2 <= 0:
            return None
        corner = line_line(first.a, first.b, second.a, second.b)
        if corner is None:
            return None
        end1 = first.b if _dist2(first.b, corner) < _dist2(first.a, corner) \
            else first.a
        end2 = second.b if _dist2(second.b, corner) < _dist2(second.a, corner) \
            else second.a
        keep1 = first.b if first.b != end1 else first.a
        keep2 = second.b if second.b != end2 else second.a
        # Unit directions from the corner towards the kept ends.
        u1 = (keep1[0] - corner[0], keep1[1] - corner[1])
        u2 = (keep2[0] - corner[0], keep2[1] - corner[1])
        l1, l2 = math.hypot(*u1), math.hypot(*u2)
        if l1 == 0 or l2 == 0:
            return None
        u1 = (u1[0] / l1, u1[1] / l1)
        u2 = (u2[0] / l2, u2[1] / l2)
        if d1 > l1 or d2 > l2:
            return None
        p1 = (corner[0] + u1[0] * d1, corner[1] + u1[1] * d1)
        p2 = (corner[0] + u2[0] * d2, corner[1] + u2[1] * d2)
        seg_id = self.add_line(p1, p2)
        if first.b == end1:
            first.b = p1
        else:
            first.a = p1
        if second.b == end2:
            second.b = p2
        else:
            second.a = p2
        self._touch(id1)
        self._touch(id2)
        return seg_id

    # ── constraints + dimensions ─────────────────────────────────

    def constrain(self, kind, refs, value: float | None = None) -> int:
        """Add a constraint, validated before it lands. Returns the
        new constraint id."""
        if isinstance(kind, str):
            try:
                kind = ConstraintKind(kind)
            except ValueError as exc:
                raise CadError(f"unknown constraint kind {kind!r}",
                               kind="invalid_argument") from exc
        refs = [self._coerce_ref(ref) for ref in refs]
        constraint = Constraint(id=self.doc.next_constraint_id(),
                                kind=kind, refs=refs, value=value)
        problem = validate_constraint(self.doc, constraint)
        if problem is not None:
            raise CadError(f"invalid constraint: {problem}",
                           kind="invalid_argument")
        self.doc.constraints.append(constraint)
        return constraint.id

    @staticmethod
    def _coerce_ref(ref) -> GeomRef:
        if isinstance(ref, GeomRef):
            return ref
        if isinstance(ref, str):
            handle, _, sub = ref.partition(":")
            return GeomRef(handle, Sub.parse(sub) if sub else Sub.WHOLE)
        raise CadError(f"expected a GeomRef, got {ref!r}",
                       kind="invalid_argument")

    def add_dimension(self, kind: str, refs,
                      position) -> Dimension:
        """Add a dimension whose value is MEASURED from the geometry
        (aligned = length, angular = degrees, radius/diameter from
        the circle) — never typed in by hand."""
        refs = [self._coerce_ref(ref) for ref in refs]
        # Map the dimension kind onto the constraint kind that
        # measures the same quantity.
        kind_map = {
            "aligned": ConstraintKind.DISTANCE_ALIGNED,
            "angular": ConstraintKind.ANGULAR,
            "radius": ConstraintKind.RADIUS,
            "diameter": ConstraintKind.DIAMETER,
        }
        if kind not in kind_map:
            raise CadError(f"unknown dimension kind {kind!r}",
                           kind="invalid_argument")
        probe = Constraint(id=0, kind=kind_map[kind], refs=refs)
        value = measure(self.doc, probe)
        dim = Dimension(kind=kind, refs=refs, value=value,
                        position=_point(position))
        dim_id = self.doc.add_entity(dim)
        self.doc._dirty_ids.add(dim_id)
        return dim

    # ── layers + removal ─────────────────────────────────────────

    def add_layer(self, name: str, color: str = "#ffffff",
                  visible: bool = True) -> None:
        if self.doc.layer(name) is not None:
            raise CadError(f"layer {name!r} already exists",
                           kind="invalid_argument")
        self.doc.layers.append(Layer(name=name, color=color,
                                     visible=bool(visible)))

    def set_layer(self, ids, name: str) -> None:
        if self.doc.layer(name) is None:
            raise CadError(f"no layer {name!r}", kind="not_found")
        for entity_id in ids:
            entity = self._get(entity_id)
            entity.layer = name
            self._touch(entity_id)

    def remove(self, ids) -> None:
        for entity_id in ids:
            if entity_id not in self.doc.entities:
                raise CadError(f"no entity {entity_id!r}",
                               kind="not_found")
        for entity_id in ids:
            del self.doc.entities[entity_id]
            self.doc._dirty_ids.discard(entity_id)
