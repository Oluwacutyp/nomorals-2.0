"""Path operations — boolean ops, cleanup, dashing.

Port of the *design* of the upstream ``crates/pathops``
(boolean/pathfinder over a planar graph) into Devon's Python.
Booleans run on flattened simple polygons via the
Weiler–Atherton algorithm (proper segment crossings only;
collinear overlapping edges are ignored, and a result with a
hole comes back as one path with an outer contour plus a
reverse-wound hole contour). ``polygon_area`` sums signed contour
areas, so a holed path measures outer minus hole. Offsetting is
NOT built (named gap — see the survey).
"""

from __future__ import annotations

import math
from itertools import pairwise

from .geom import Path, parse_path_data

__all__ = ["boolean", "dash_path", "flatten_path", "polygon_area"]

_EPS = 1e-9


def flatten_path(path: Path, tolerance: float = 0.5
                 ) -> list[list[tuple[float, float]]]:
    """Flatten a path to closed contours (points, first repeated
    at the end). Open contours are closed implicitly for area
    work; the renderer keeps them open separately."""
    return path.flatten(tolerance)


def polygon_area(path: Path, tolerance: float = 0.5) -> float:
    """Signed area of a path: the sum of its contours' shoelace
    areas (holes wound opposite subtract)."""
    total = 0.0
    for contour in path.flatten(tolerance):
        area = 0.0
        # The closing edge (last point back to first) is part of
        # the polygon: without it an open path's area is wrong by
        # exactly one edge term. The two zipped lists are the same
        # length by construction, hence strict=True.
        for (x1, y1), (x2, y2) in zip(contour, contour[1:]
                                      + contour[:1], strict=True):
            area += x1 * y2 - x2 * y1
        total += area / 2.0
    return total


def _seg_intersect(a: tuple[float, float], b: tuple[float, float],
                   c: tuple[float, float], d: tuple[float, float]
                   ) -> tuple[float, float, float, float] | None:
    """Proper crossing of segments ab and cd, or None.

    Returns (x, y, t_ab, t_cd) with both parameters strictly
    inside (0, 1) — endpoint touches and collinear overlaps are
    not crossings.
    """
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < _EPS:
        return None
    qp = (c[0] - a[0], c[1] - a[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if _EPS < t < 1.0 - _EPS and _EPS < u < 1.0 - _EPS:
        return (a[0] + t * r[0], a[1] + t * r[1], t, u)
    return None


def _point_in_polygon(point: tuple[float, float],
                      poly: list[tuple[float, float]]) -> bool:
    """Even-odd point-in-polygon (points on an edge count out)."""
    x, y = point
    inside = False
    count = len(poly)
    for i in range(count):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def _clean(poly: list[tuple[float, float]]
           ) -> list[tuple[float, float]]:
    """Drop duplicate consecutive points (and a closing repeat)."""
    out: list[tuple[float, float]] = []
    for point in poly:
        if not out or _dist2(out[-1], point) > _EPS * _EPS:
            out.append(point)
    if len(out) > 1 and _dist2(out[0], out[-1]) <= _EPS * _EPS:
        out.pop()
    return out


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class _Node:
    __slots__ = ("point", "is_x", "x_id", "next", "prev")

    def __init__(self, point: tuple[float, float],
                 is_x: bool = False, x_id: int = -1):
        self.point = point
        self.is_x = is_x
        self.x_id = x_id
        self.next: _Node | None = None
        self.prev: _Node | None = None


def _build_ring(poly: list[tuple[float, float]]) -> list[_Node]:
    nodes = [_Node(p) for p in poly]
    for i, node in enumerate(nodes):
        node.next = nodes[(i + 1) % len(nodes)]
        node.prev = nodes[(i - 1) % len(nodes)]
    return nodes


def _insert_crossings(subject: list[tuple[float, float]],
                      clip: list[tuple[float, float]]
                      ) -> tuple[list[_Node], list[_Node], int]:
    """Split both rings at every proper crossing. Returns the two
    node rings and the number of crossings found."""
    # Collect crossings per edge, then rebuild each ring.
    sub_hits: dict[int, list[tuple[float, tuple[float, float], int]]] \
        = {}
    clip_hits: dict[int, list[tuple[float, tuple[float, float], int]]] \
        = {}
    x_id = 0
    for i in range(len(subject)):
        a, b = subject[i], subject[(i + 1) % len(subject)]
        for j in range(len(clip)):
            c, d = clip[j], clip[(j + 1) % len(clip)]
            hit = _seg_intersect(a, b, c, d)
            if hit is None:
                continue
            point = (hit[0], hit[1])
            t, u = hit[2], hit[3]
            sub_hits.setdefault(i, []).append((t, point, x_id))
            clip_hits.setdefault(j, []).append((u, point, x_id))
            x_id += 1
    sub_nodes: list[_Node] = []
    for i, point in enumerate(subject):
        sub_nodes.append(_Node(point))
        for _t, hit_point, hit_id in sorted(
                sub_hits.get(i, [])):
            sub_nodes.append(_Node(hit_point, True, hit_id))
    clip_nodes: list[_Node] = []
    for j, point in enumerate(clip):
        clip_nodes.append(_Node(point))
        for _u, hit_point, hit_id in sorted(
                clip_hits.get(j, [])):
            clip_nodes.append(_Node(hit_point, True, hit_id))
    for ring in (sub_nodes, clip_nodes):
        for i, node in enumerate(ring):
            node.next = ring[(i + 1) % len(ring)]
            node.prev = ring[(i - 1) % len(ring)]
    return sub_nodes, clip_nodes, x_id


def _label(ring: list[_Node],
           other: list[tuple[float, float]]) -> dict[int, bool]:
    """For each crossing id, whether the ring's following edge
    (from the crossing node) runs inside the other polygon."""
    labels: dict[int, bool] = {}
    for node in ring:
        if not node.is_x or node.next is None:
            continue
        mid = ((node.point[0] + node.next.point[0]) / 2.0,
               (node.point[1] + node.next.point[1]) / 2.0)
        labels[node.x_id] = _point_in_polygon(mid, other)
    return labels


def _crossing_nodes(ring: list[_Node]) -> dict[int, _Node]:
    return {node.x_id: node for node in ring if node.is_x}


#: Per-mode walk spec: (start label on the subject, (direction,
#: stop label) per ring). Labels say whether the ring's following
#: edge runs inside the other polygon. Verified by hand on two
#: overlapping squares:
#:   union:        S-fwd from EXIT, stop at ENTRY; C-fwd, stop at
#:                 ENTRY  →  area 175 for 10×10 squares at (0,0)/(5,5)
#:   intersection: S-fwd from ENTRY, stop at EXIT; C-fwd, stop at
#:                 EXIT   →  area 25
#:   difference:   S-fwd from EXIT, stop at ENTRY; C-backward, stop
#:                 at ENTRY → area 75
_WALK = {
    "union": {
        "start_in": False,
        "sub": ("fwd", True),
        "clip": ("fwd", True),
    },
    "intersection": {
        "start_in": True,
        "sub": ("fwd", False),
        "clip": ("fwd", False),
    },
    "difference": {
        "start_in": False,
        "sub": ("fwd", True),
        "clip": ("bwd", True),
    },
}


def _walk_result(start: _Node, start_ring: str,
                 sub: list[_Node], clip: list[_Node],
                 sub_in: dict[int, bool], clip_in: dict[int, bool],
                 mode: str) -> list[tuple[float, float]] | None:
    """Walk one result contour from a starting crossing.

    Walks each ring in its spec direction, switching rings at the
    first crossing whose label matches the spec stop value
    (crossings alternate entry/exit along a ring, so this is the
    next crossing). Stops when the walk returns to the start
    crossing on the start ring.
    """
    spec = _WALK[mode]
    points: list[tuple[float, float]] = [start.point]
    node = start
    ring = start_ring
    guard = 0
    while True:
        guard += 1
        if guard > 10000:
            return None
        direction, stop_value = spec[ring]
        labels = sub_in if ring == "sub" else clip_in
        nxt: _Node | None = None
        # Walk to the next crossing on this ring.
        while True:
            nxt = node.next if direction == "fwd" else node.prev
            if nxt is None:
                return None
            if nxt.is_x:
                break
            points.append(nxt.point)
            node = nxt
        node = nxt
        if ring == start_ring and node.x_id == start.x_id:
            break
        if labels[node.x_id] == stop_value:
            points.append(node.point)
            ring = "clip" if ring == "sub" else "sub"
            other = clip if ring == "clip" else sub
            node = _crossing_nodes(other)[node.x_id]
            if ring == start_ring and node.x_id == start.x_id:
                break
        else:
            # Degenerate input: crossings did not alternate.
            # Keep walking this ring past the crossing.
            points.append(node.point)
    return _clean(points)


def _wa(mode: str, subject: list[tuple[float, float]],
        clip: list[tuple[float, float]]
        ) -> list[list[tuple[float, float]]]:
    """Weiler–Atherton for one subject contour vs one clip
    contour. Returns result contours (possibly empty)."""
    subject = _clean(subject)
    clip = _clean(clip)
    if len(subject) < 3 or len(clip) < 3:
        return []
    sub_ring, clip_ring, crossings = _insert_crossings(subject, clip)
    if crossings == 0:
        sub_in_clip = _point_in_polygon(subject[0], clip)
        clip_in_sub = _point_in_polygon(clip[0], subject)
        if mode == "union":
            if sub_in_clip:
                return [clip]
            if clip_in_sub:
                return [subject]
            return [subject, clip]
        if mode == "intersection":
            if sub_in_clip:
                return [subject]
            if clip_in_sub:
                return [clip]
            return []
        # difference
        if clip_in_sub:
            return [subject, list(reversed(clip))]
        if sub_in_clip:
            return []
        return [subject]
    sub_in = _label(sub_ring, clip)
    clip_in = _label(clip_ring, subject)
    results: list[list[tuple[float, float]]] = []
    used: set[tuple[str, int]] = set()
    sub_x = _crossing_nodes(sub_ring)
    clip_x = _crossing_nodes(clip_ring)
    want_in = _WALK[mode]["start_in"]
    starts = [(node, "sub") for x_id, node in sub_x.items()
              if sub_in[x_id] == want_in
              and ("sub", x_id) not in used]
    for node, ring in starts:
        if (ring, node.x_id) in used:
            continue
        contour = _walk_result(node, ring, sub_ring, clip_ring,
                               sub_in, clip_in, mode)
        if contour and len(contour) >= 3:
            results.append(contour)
            # Mark every crossing this contour passed through.
            for point in contour:
                for x_id, xnode in sub_x.items():
                    if _dist2(point, xnode.point) <= _EPS * _EPS:
                        used.add(("sub", x_id))
                for x_id, xnode in clip_x.items():
                    if _dist2(point, xnode.point) <= _EPS * _EPS:
                        used.add(("clip", x_id))
    # A difference whose clip is fully inside the subject has no
    # crossings; handled above. A difference whose subject is
    # fully inside the clip returns empty — also above.
    return results


def _contours_to_path(contours: list[list[tuple[float, float]]]
                      ) -> Path:
    if not contours:
        return Path()
    parts: list[str] = []
    for contour in contours:
        if not contour:
            continue
        parts.append(
            "M " + " L ".join(f"{x:g} {y:g}" for x, y in contour)
            + " Z")
    return parse_path_data(" ".join(parts))


def boolean(op: str, subject: Path, clip: Path,
            tolerance: float = 0.5) -> list[Path]:
    """Boolean op (``union`` | ``intersection`` | ``difference`` |
    ``xor``) between two paths, on their flattened contours.

    Returns a list of paths (one per disjoint result region; a
    holed region is a single path with outer + hole contours).
    """
    if op not in ("union", "intersection", "difference", "xor"):
        raise ValueError(
            f"unknown boolean op {op!r}; use union, intersection, "
            "difference or xor")
    sub_contours = [c for c in subject.flatten(tolerance) if len(c) >= 3]
    clip_contours = [c for c in clip.flatten(tolerance) if len(c) >= 3]
    if not sub_contours:
        return []
    if not clip_contours:
        return [subject] if op in ("union", "difference", "xor") \
            else []
    if op == "xor":
        minus = boolean("difference", subject, clip, tolerance)
        plus = boolean("difference", clip, subject, tolerance)
        return minus + plus
    if op == "union":
        results: list[list[tuple[float, float]]] = list(sub_contours)
        for clip_c in clip_contours:
            new_results: list[list[tuple[float, float]]] = []
            pending: list[tuple[float, float]] | None = clip_c
            for result in results:
                if pending is None:
                    new_results.append(result)
                    continue
                pieces = _wa("union", result, pending)
                if len(pieces) == 1:
                    # Merged (or one contained in the other): the
                    # single piece replaces the result.
                    new_results.append(pieces[0])
                    pending = None
                else:
                    # Disjoint: keep the result, keep trying.
                    new_results.append(result)
            if pending is not None:
                new_results.append(pending)
            results = new_results
        return [_contours_to_path([c]) for c in results]
    if op == "intersection":
        results: list[list[tuple[float, float]]] = []
        for sub_c in sub_contours:
            for clip_c in clip_contours:
                results.extend(_wa("intersection", sub_c, clip_c))
        return [_contours_to_path([c]) for c in results]
    # difference
    results = list(sub_contours)
    for clip_c in clip_contours:
        next_results: list[list[tuple[float, float]]] = []
        for sub_c in results:
            next_results.extend(_wa("difference", sub_c, clip_c))
        results = next_results
    # Group contours into paths: a hole belongs to the path whose
    # outer contour contains it.
    paths: list[Path] = []
    remaining = list(results)
    while remaining:
        outer = remaining.pop(0)
        holes = [c for c in remaining
                 if _point_in_polygon(c[0], outer)
                 and abs(_signed_area(c)) < abs(_signed_area(outer))]
        for hole in holes:
            remaining.remove(hole)
        paths.append(_contours_to_path([outer] + holes))
    return paths


def _signed_area(contour: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in pairwise(contour):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def dash_path(path: Path, dash: tuple[float, ...],
              tolerance: float = 0.5) -> list[Path]:
    """Split a path's contours into dash segments (on/off pattern).

    Returns one path per visible dash run.
    """
    if not dash or any(d <= 0 for d in dash):
        return [path]
    out: list[Path] = []
    for contour in path.flatten(tolerance):
        pattern = list(dash)
        index = 0
        left = pattern[0]
        on = True
        run: list[tuple[float, float]] = [contour[0]]
        for a, b in pairwise(contour):
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            pos = 0.0
            while pos < seg_len:
                take = min(left, seg_len - pos)
                t0 = pos / seg_len if seg_len else 0.0
                t1 = (pos + take) / seg_len if seg_len else 0.0
                p0 = (a[0] + (b[0] - a[0]) * t0,
                      a[1] + (b[1] - a[1]) * t0)
                p1 = (a[0] + (b[0] - a[0]) * t1,
                      a[1] + (b[1] - a[1]) * t1)
                if on:
                    if not run or run[-1] != p0:
                        run.append(p0)
                    run.append(p1)
                pos += take
                left -= take
                if left <= _EPS:
                    if on and len(run) >= 2:
                        d = "M " + " L ".join(
                            f"{x:g} {y:g}" for x, y in run)
                        out.append(parse_path_data(d))
                    run = []
                    index = (index + 1) % len(pattern)
                    left = pattern[index]
                    on = not on
        if on and len(run) >= 2:
            d = "M " + " L ".join(f"{x:g} {y:g}" for x, y in run)
            out.append(parse_path_data(d))
    return out
