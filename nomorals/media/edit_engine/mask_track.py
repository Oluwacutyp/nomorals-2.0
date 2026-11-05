"""Deterministic translation tracking for regional effect masks.

A regional effect mask resolves once, against the clip's window-
start frame (see :func:`render._prepare_effects`), and used to
simply *hold* for the whole clip — if the masked subject moved, the
effect stayed behind on empty background. This module is the
tracking rung: given the frame sequence (grayscale) and the anchor
mask, it follows the masked region's content frame by frame and
reports where the mask must sit on each frame.

The tracker is deliberately small and honest about its shape:

- **Translation only.** The mask moves rigidly by the tracked
  offset of its bounding-box content. Rotation, scale and deform
  are not modelled; content that changes shape drives the match
  score down and the tracker reports the frames it cannot follow
  instead of inventing a warp.
- **Template NCC.** The anchor frame's masked bounding box (plus a
  small margin) is the template; each next frame is matched by
  normalized cross-correlation (FFT correlation + integral-image
  normalization, pure numpy — no OpenCV rung exists on the render
  hosts, and none is pretended). The search is restricted to a
  per-step radius around the last good position.
- **Loss is data.** A frame whose best score falls below
  ``min_score`` is *lost*: the mask holds at the last good offset
  for that frame, the loss is recorded, and later frames try to
  re-acquire around the last good position. The result's status is
  ``tracked`` (every step matched), ``partial`` (some steps lost)
  or ``held`` (nothing could be followed — no texture, or the very
  first step already failed). The caller decides what a status
  means; the render layer keeps the static mask on ``held`` and
  says so in the effect record.
- **Deterministic.** Same frames + mask + parameters ⇒ same
  offsets, on any machine: the correlation is one fixed numpy
  computation and ties break by scan order.

Quality is *measured*, not asserted: ``mask_track_corpus`` builds
synthetic sequences whose ground-truth motion is exact by
construction and scores this tracker against floors recorded in
``tests/golden/mask_track_corpus.json``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...media_edit.videos import MediaEditError

__all__ = [
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SEARCH_RADIUS",
    "MAX_TRACK_FRAMES",
    "TrackResult",
    "mask_centroid",
    "shift_mask",
    "track_sequence",
]

#: per-step search radius (px) around the last good position.
DEFAULT_SEARCH_RADIUS = 48
#: a step whose best NCC score falls below this is *lost*.
DEFAULT_MIN_SCORE = 0.5
#: the anchor template needs real texture: below this gray-level
#: standard deviation there is nothing to follow.
TEXTURE_STD_EPS = 2.0
#: context ring (px) of background included around the mask's
#: bounding box in the template — it pins the match spatially.
TEMPLATE_MARGIN = 8
#: hard cap on tracked frames per mask (2 min at 30 fps): beyond
#: it the render layer holds instead of tracking (named, in the
#: effect record) rather than paying unbounded extraction cost.
MAX_TRACK_FRAMES = 1800


@dataclass
class TrackResult:
    """Where the mask goes on every frame, and how sure that is.

    ``offsets[i]`` is the mask's displacement (dx, dy) in canvas
    pixels on frame i (frame 0 is the anchor: always ``(0, 0)``).
    ``scores[i]`` is the NCC score of the step that produced frame
    i's offset (1.0 on the anchor); ``lost[i]`` marks frames whose
    step failed — their offset is the last good one, held.
    """

    offsets: list[tuple[int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    lost: list[bool] = field(default_factory=list)
    status: str = "held"          # tracked | partial | held
    reason: str = ""              # why held (or the loss summary)

    @property
    def frames(self) -> int:
        return len(self.offsets)

    @property
    def lost_count(self) -> int:
        return sum(1 for x in self.lost if x)

    @property
    def max_displacement(self) -> float:
        if not self.offsets:
            return 0.0
        return max((dx * dx + dy * dy) ** 0.5 for dx, dy in self.offsets)

    def detail(self) -> str:
        """One-line provenance for the render's effect record."""
        if self.status == "held":
            return f"held: {self.reason}" if self.reason else "held"
        base = (f"{self.status}: max displacement "
                f"{self.max_displacement:.0f}px over {self.frames} "
                f"frames")
        if self.lost_count:
            base += (f"; {self.lost_count} frame(s) held at the "
                     f"last good offset")
        return base


def _ncc_map(frame: np.ndarray, templ: np.ndarray) -> np.ndarray:
    """NCC of ``templ`` at every valid top-left position in ``frame``.

    Both inputs float64 2-D. The correlation half is one FFT pass;
    the normalization half is integral images — the pair is exact
    (the corpus pins argmax positions against known translations).
    """
    F = np.asarray(frame, dtype=np.float64)
    T = np.asarray(templ, dtype=np.float64)
    H, W = F.shape
    th, tw = T.shape
    shape = (H + th - 1, W + tw - 1)
    corr = np.fft.irfft2(
        np.fft.rfft2(F, shape) * np.conj(np.fft.rfft2(T, shape)),
        shape)[:H - th + 1, :W - tw + 1]

    def integral(a: np.ndarray) -> np.ndarray:
        return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def windowed(integ: np.ndarray) -> np.ndarray:
        return (integ[th:, tw:] - integ[:-th, tw:]
                - integ[th:, :-tw] + integ[:-th, :-tw])

    sF = windowed(integral(F))
    sF2 = windowed(integral(F * F))
    n = float(th * tw)
    sT = float(T.sum())
    varT = float((T * T).sum()) - sT * sT / n
    num = corr - sF * (sT / n)
    varF = np.maximum(sF2 - sF * sF / n, 0.0)
    den = np.sqrt(varF * max(varT, 0.0))
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 1e-9)
    return out


def _bbox(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the mask's ones, plus margin, clipped."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    return x0, y0, x1, y1


def track_sequence(frames: Iterable[Any], mask: Any, *,
                   search_radius: int = DEFAULT_SEARCH_RADIUS,
                   min_score: float = DEFAULT_MIN_SCORE,
                   margin: int = TEMPLATE_MARGIN) -> TrackResult:
    """Follow ``mask``'s content through ``frames`` (grayscale).

    ``frames`` is consumed lazily (the renderer streams extracted
    frames; only offsets are kept). ``mask`` is the anchor frame's
    mask — boolean or 0..255, same HxW as the frames. Returns a
    :class:`TrackResult`; content problems (empty mask, no texture,
    unmatchable steps) are reported in the result, never raised —
    only shape misuse raises (mismatched frame sizes).
    """
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 127
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise MediaEditError(
            "mask tracking needs at least one frame") from None
    f0 = np.asarray(first, dtype=np.float64)
    if f0.ndim != 2:
        raise MediaEditError(
            f"tracking frames must be 2-D grayscale, got {f0.shape}")
    if m.shape != f0.shape:
        raise MediaEditError(
            f"mask shape {m.shape} != frame shape {f0.shape}")
    res = TrackResult(offsets=[(0, 0)], scores=[1.0], lost=[False])
    if not m.any():
        res.reason = "the resolved mask is empty — nothing to track"
        return res
    x0, y0, x1, y1 = _bbox(m, margin)
    templ = f0[y0:y1, x0:x1]
    std = float(templ.std())
    if std < TEXTURE_STD_EPS:
        res.reason = (
            f"the masked region at the anchor frame has no "
            f"trackable texture (template std {std:.2f} < "
            f"{TEXTURE_STD_EPS})")
        return res
    H, W = f0.shape
    th, tw = templ.shape
    max_px, max_py = W - tw, H - th     # valid top-left range
    pos = (x0, y0)                       # current top-left (last good)
    good_steps = 0
    for raw in it:
        F = np.asarray(raw, dtype=np.float64)
        if F.shape != f0.shape:
            raise MediaEditError(
                f"tracking frame shape {F.shape} != anchor {f0.shape}")
        scores = _ncc_map(F, templ)
        cx0 = max(0, pos[0] - search_radius)
        cy0 = max(0, pos[1] - search_radius)
        cx1 = min(max_px, pos[0] + search_radius)
        cy1 = min(max_py, pos[1] + search_radius)
        if cx1 < cx0 or cy1 < cy0:
            res.offsets.append(res.offsets[-1])
            res.scores.append(0.0)
            res.lost.append(True)
            continue
        win = scores[cy0:cy1 + 1, cx0:cx1 + 1]
        flat = int(np.argmax(win))
        wy, wx = np.unravel_index(flat, win.shape)
        best = float(win[wy, wx])
        if best >= min_score:
            pos = (cx0 + int(wx), cy0 + int(wy))
            good_steps += 1
            res.offsets.append((pos[0] - x0, pos[1] - y0))
            res.scores.append(best)
            res.lost.append(False)
        else:
            res.offsets.append(res.offsets[-1])
            res.scores.append(best)
            res.lost.append(True)
    if res.frames == 1:
        res.status = "tracked"
        res.reason = "single frame — nothing to follow"
    elif good_steps == 0:
        res.status = "held"
        res.reason = (
            f"the masked content could not be matched on the first "
            f"step (best score {max(res.scores[1:]):.2f} < "
            f"{min_score}) — scene change or content loss at the "
            f"window start")
    elif res.lost_count:
        res.status = "partial"
        res.reason = (
            f"{res.lost_count} of {res.frames - 1} step(s) fell "
            f"below the match floor {min_score}")
    else:
        res.status = "tracked"
    return res


def shift_mask(mask: Any, dx: int, dy: int) -> np.ndarray:
    """``mask`` (bool or 0..255) translated by (dx, dy), uint8 0/255.

    Pixels that leave the canvas are clipped away; pixels that
    enter are the mask's own — no wraparound, ever.
    """
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 127
    out = np.zeros(m.shape, dtype=np.uint8)
    h, w = m.shape
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = \
            m[sy0:sy1, sx0:sx1].astype(np.uint8) * 255
    return out


def mask_centroid(mask: Any) -> tuple[float, float]:
    """(cx, cy) of the mask's ones; (nan, nan) for an empty mask."""
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 127
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return (float(xs.mean()), float(ys.mean()))
