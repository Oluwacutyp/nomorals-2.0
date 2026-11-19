"""Distance fields for layer effects — the exact Euclidean
transform (Felzenszwalb & Huttenlocher) and the 5 x 5 chamfer
transform, as Photoshop's effect distances use them.

Effect edges (precise glows, chiselled bevels, wide strokes)
use the exact Euclidean transform; spread / choke and narrow
strokes use the 5 x 5 chamfer (step lengths 1, sqrt(2),
sqrt(5)), whose sub-pixel edge seeding Photoshop matches at
small offsets. Distances are measured from pixel centres to
the shape's edge: pixels inside the shape get <= 0, outside
pixels get (distance to the edge) - 0.5, with the edge placed
by the nearest pixel's coverage.

The Euclidean transform uses scipy when it is importable and
falls back to a transliteration of the Felzenszwalb 1D pass
otherwise, so the organ works on a box without scipy.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["INSIDE_EPS", "dist_inside", "dist_inside_chamfer",
           "dist_outside", "dist_outside_chamfer",
           "local_coverage"]

INSIDE_EPS = 0.5 / 255.0
_CHAMFER_INF = 1e20
_S2 = math.sqrt(2.0)
_S5 = math.sqrt(5.0)

try:
    from scipy.ndimage import distance_transform_edt as _edt

    def _edt_nearest(inside: np.ndarray) -> tuple[np.ndarray,
                                                  np.ndarray]:
        """(distance, flat index of the nearest inside pixel)
        for every pixel; distance is 0 on inside pixels."""
        # scipy measures to the nearest ZERO: invert so inside
        # pixels are the zeros.
        dist, inds = _edt(~inside, return_indices=True)
        h, w = inside.shape
        near = (inds[0] * w + inds[1]).astype(np.int64)
        return dist, near

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy present on dev box
    _HAVE_SCIPY = False

    def _dt1(f: list[float]) -> tuple[list[float], list[int]]:
        """1D squared distance transform (Felzenszwalb &
        Huttenlocher): for each q, min over p of (q-p)^2+f[p]."""
        n = len(f)
        v = [0] * n
        z = [0.0] * (n + 1)
        near = [0] * n
        z[0] = float("-inf")
        z[1] = float("inf")
        k = 0
        for q in range(1, n):
            while True:
                p = v[k]
                s = ((f[q] + q * q) - (f[p] + p * p)) / (
                    2.0 * q - 2.0 * p)
                if s <= z[k] and k > 0:
                    k -= 1
                    continue
                if s <= z[k]:
                    v[0] = q
                    z[0] = float("-inf")
                    z[1] = float("inf")
                    break
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = float("inf")
                break
        k = 0
        out = [0.0] * n
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            p = v[k]
            d = q - p
            out[q] = d * d + f[p]
            near[q] = p
        return out, near

    def _edt_nearest(inside: np.ndarray) -> tuple[np.ndarray,
                                                  np.ndarray]:
        h, w = inside.shape
        inf = 1e20
        # Columns first (distance along y), then rows.
        gc = np.full((h, w), inf, dtype=np.float64)
        nc = np.zeros((h, w), dtype=np.int64)
        for x in range(w):
            col = [0.0 if inside[y, x] else inf
                   for y in range(h)]
            out, near = _dt1(col)
            gc[:, x] = out
            nc[:, x] = near
        dist = np.zeros((h, w), dtype=np.float64)
        near_idx = np.zeros((h, w), dtype=np.int64)
        for y in range(h):
            row = list(gc[y, :])
            out, near = _dt1(row)
            dist[y, :] = np.sqrt(np.array(out))
            for x in range(w):
                near_idx[y, x] = nc[near[x], x] * w + near[x]
        return dist, near_idx


def dist_outside(alpha: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from each pixel centre outside
    the shape to its edge (inside pixels get -0.5)."""
    inside = alpha > INSIDE_EPS
    dist, near = _edt_nearest(inside)
    flat_alpha = np.clip(alpha.ravel(), 0.0, 1.0)
    out = np.where(inside, -0.5,
                   dist - 0.5 + (1.0 - flat_alpha[near])
                   ).reshape(alpha.shape)
    return out


def dist_inside(alpha: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from each pixel centre inside
    the shape to its edge (outside pixels get -0.5)."""
    outside = alpha <= INSIDE_EPS
    dist, _ = _edt_nearest(outside)
    return np.where(outside, -0.5, dist - 0.5)


def _chamfer_from(start: np.ndarray) -> np.ndarray:
    """Two-pass 5 x 5 chamfer transform from per-pixel start
    values (``_CHAMFER_INF`` = not a seed): the minimum over
    seeds of start value + path length. The forward pass
    relaxes each row from the rows above (vectorized) plus a
    left-to-right sweep; the backward pass mirrors it. The
    array is padded with INF so shifted reads never wrap
    around a row's edge."""
    h, w = start.shape
    d = np.pad(start, 2, mode="constant",
               constant_values=_CHAMFER_INF)
    idx = np.arange(w)
    # Forward pass: neighbours (-1,0), (-1,-1), (0,-1), (1,-1),
    # (-1,-2), (1,-2), (-2,-1), (2,-1) — all from rows above
    # except the same-row left neighbour, handled by a sweep.
    for y in range(h):
        py = y + 2
        row = d[py, 2:2 + w]
        up = d[py - 1, 2:2 + w]
        row = np.minimum(row, up + 1.0)
        row = np.minimum(row, d[py - 1, 1:1 + w] + _S2)
        row = np.minimum(row, d[py - 1, 3:3 + w] + _S2)
        row = np.minimum(row, d[py - 1, 0:0 + w] + _S5)
        row = np.minimum(row, d[py - 1, 4:4 + w] + _S5)
        row = np.minimum(row, d[py - 2, 1:1 + w] + _S5)
        row = np.minimum(row, d[py - 2, 3:3 + w] + _S5)
        # Same-row left sweep: d[x] = min over k<=x of
        # d[k] + (x-k) = x + min(d[k]-k).
        row = np.minimum.accumulate(row - idx) + idx
        d[py, 2:2 + w] = row
    # Backward pass: neighbours (+1,0), (+1,+1), (0,+1),
    # (-1,+1), (+1,+2), (-1,+2), (+2,+1), (-2,+1).
    for y in range(h - 1, -1, -1):
        py = y + 2
        row = d[py, 2:2 + w]
        dn = d[py + 1, 2:2 + w]
        row = np.minimum(row, dn + 1.0)
        row = np.minimum(row, d[py + 1, 1:1 + w] + _S2)
        row = np.minimum(row, d[py + 1, 3:3 + w] + _S2)
        row = np.minimum(row, d[py + 1, 0:0 + w] + _S5)
        row = np.minimum(row, d[py + 1, 4:4 + w] + _S5)
        row = np.minimum(row, d[py + 2, 1:1 + w] + _S5)
        row = np.minimum(row, d[py + 2, 3:3 + w] + _S5)
        # Same-row right sweep: d[x] = min over k>=x of
        # d[k] + (k-x) = -x + min(d[k]+k).
        row = (np.minimum.accumulate((row + idx)[::-1])[::-1]
               - idx)
        d[py, 2:2 + w] = row
    return d[2:2 + h, 2:2 + w]


def dist_outside_chamfer(alpha: np.ndarray) -> np.ndarray:
    """Chamfer distance outside the shape (inside pixels get
    -0.5). Seeds start at their sub-pixel edge offset
    (1 - coverage)."""
    start = np.where(alpha > INSIDE_EPS,
                     1.0 - np.clip(alpha, 0.0, 1.0),
                     _CHAMFER_INF)
    d = _chamfer_from(start)
    return np.where(alpha > INSIDE_EPS, -0.5, d - 0.5)


def local_coverage(alpha: np.ndarray) -> np.ndarray:
    """Alpha relative to its 3 x 3 neighbourhood maximum:
    separates edge coverage from interior coverage."""
    padded = np.pad(alpha, 1, mode="edge")
    neigh = np.full_like(alpha, -1.0)
    for dy in range(3):
        for dx in range(3):
            neigh = np.maximum(
                neigh, padded[dy:dy + alpha.shape[0],
                              dx:dx + alpha.shape[1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(neigh > 0, alpha / np.maximum(neigh, 1e-9),
                       0.0)
    return np.clip(out, 0.0, 1.0)


def dist_inside_chamfer(alpha: np.ndarray) -> np.ndarray:
    """Chamfer distance inside the shape (outside pixels get
    -0.5). Partly covered edge pixels seed at their local
    coverage, mirroring the outside field."""
    cov = local_coverage(alpha)
    start = np.where(
        alpha <= INSIDE_EPS, 0.0,
        np.where(cov < 1.0 - INSIDE_EPS, cov, _CHAMFER_INF))
    d = _chamfer_from(start)
    return np.where(alpha <= INSIDE_EPS, -0.5, d - 0.5)
