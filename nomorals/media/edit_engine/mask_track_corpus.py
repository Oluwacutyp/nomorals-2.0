"""Tracking corpus: synthetic sequences with exact ground truth.

The mask tracker's quality bar (§1.1d's discipline, applied to
motion): the tracker is *scored*, not vibes. Every fixture here is
a deterministic synthetic sequence built from seeded textures, so
the masked patch's true position on every frame is exact by
construction — including the adversarial fixtures, where the truth
is that there is *nothing to follow* (the patch vanishes, the scene
cuts, the field is featureless) and the tracker must say so.

- :data:`FIXTURES` — the scene list (name, seed, notes).
- :func:`build_sequence` — ``(frames, mask, truth, meta)``:
  grayscale uint8 frames, the anchor mask (frame 0's patch rect),
  per-frame truth offsets (``None`` once the content is gone), and
  the fixture's event frame where one exists.
- :func:`score_fixture` — runs the real tracker
  (:func:`mask_track.track_sequence`) and reports what it actually
  achieved: status, per-frame offset error and mask IoU against
  truth, centroid error, first lost frame, and whether the offset
  stayed put after the content vanished.

Floors + measured values live in
``tests/golden/mask_track_corpus.json``;
``tests/test_mask_track_corpus.py`` re-scores every fixture and
fails below a floor — a tracker regression is a test failure, not
a production surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import mask_track

__all__ = [
    "FIXTURES",
    "TrackFixture",
    "build_sequence",
    "fixture_names",
    "score_fixture",
]

_W, _H = 320, 240
_PATCH_W, _PATCH_H = 110, 80


@dataclass(frozen=True)
class TrackFixture:
    name: str
    seed: int
    notes: str


FIXTURES: tuple[TrackFixture, ...] = (
    TrackFixture(
        name="translate_diagonal", seed=21,
        notes="Textured patch drifting (3, 2) px/frame for 24 "
              "frames — the tracker's home turf; offsets should "
              "come back exact."),
    TrackFixture(
        name="translate_slow_east", seed=22,
        notes="One pixel per frame east for 16 frames — small "
              "steps, no accumulation drift."),
    TrackFixture(
        name="translate_large_step", seed=23,
        notes="(12, 8) px/frame for 12 frames — inside the "
              "per-step search radius, far outside a naive "
              "few-pixel search."),
    TrackFixture(
        name="static_scene", seed=24,
        notes="No motion at all: the tracker must report zero "
              "displacement at full confidence, not wander."),
    TrackFixture(
        name="vanish_midway", seed=25,
        notes="The patch drifts (2, 1) for 10 frames, then is "
              "gone — bare background. Truth after frame 10 is "
              "'nothing to follow': loss must be flagged and the "
              "offset must hold at the last fix, not chase "
              "background texture."),
    TrackFixture(
        name="scene_cut", seed=26,
        notes="Twelve frames of drift, then a hard cut to a "
              "different scene (new background, new patch "
              "elsewhere). The old content is simply not there: "
              "loss must be flagged from the cut."),
    TrackFixture(
        name="uniform_field", seed=27,
        notes="A featureless field and an arbitrary box mask: "
              "there is no texture to follow, so the tracker must "
              "refuse (held) and name the reason — a silent "
              "'tracked' here would be a fabricated success."),
)


def fixture_names() -> list[str]:
    return [f.name for f in FIXTURES]


def _fixture(name: str) -> TrackFixture:
    for f in FIXTURES:
        if f.name == name:
            return f
    raise ValueError(
        f"unknown tracking fixture {name!r}; use {fixture_names()}")


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w)).astype(np.uint8)


class _Painter:
    """One fixture's world: a fixed background field + one patch."""

    def __init__(self, seed: int, *, patch_seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.bg = _texture(rng, _H, _W)
        prng = np.random.default_rng(
            seed + 1000 if patch_seed is None else patch_seed)
        self.patch = _texture(prng, _PATCH_H, _PATCH_W)

    def frame(self, pos: tuple[int, int] | None) -> np.ndarray:
        f = self.bg.copy()
        if pos is not None:
            x, y = pos
            f[y:y + _PATCH_H, x:x + _PATCH_W] = self.patch
        return f


def build_sequence(name: str):
    """``(frames, mask, truth, meta)`` for fixture ``name``.

    ``truth[i]`` is the patch's true offset (dx, dy) from its
    frame-0 position, or ``None`` once the content no longer
    exists (vanish/cut). ``meta`` carries ``start`` (the patch's
    frame-0 top-left), ``event_frame`` (first frame with no
    content, or None) and the fixture notes.
    """
    fx = _fixture(name)
    frames: list[np.ndarray] = []
    truth: list[tuple[int, int] | None] = []
    event_frame: int | None = None

    if name == "uniform_field":
        start = (100, 80)
        flat = np.full((_H, _W), 128, dtype=np.uint8)
        frames = [flat.copy() for _ in range(8)]
        truth = [(0, 0)] * 8
        mask = np.zeros((_H, _W), dtype=bool)
        mask[start[1]:start[1] + _PATCH_H,
             start[0]:start[0] + _PATCH_W] = True
        return frames, mask, truth, {
            "start": start, "event_frame": None, "notes": fx.notes}

    painter = _Painter(fx.seed)
    cut_painter = (_Painter(fx.seed + 500, patch_seed=fx.seed + 900)
                   if name == "scene_cut" else None)
    start = (40, 60)
    vel = (0, 0)
    n = 12
    if name == "translate_diagonal":
        start, vel, n = (40, 60), (3, 2), 24
    elif name == "translate_slow_east":
        start, vel, n = (40, 100), (1, 0), 16
    elif name == "translate_large_step":
        start, vel, n = (30, 40), (12, 8), 12
    elif name == "static_scene":
        start, vel, n = (90, 70), (0, 0), 12
    elif name == "vanish_midway":
        start, vel, n = (40, 60), (2, 1), 20
        event_frame = 10
    elif name == "scene_cut":
        start, vel, n = (40, 60), (2, 0), 20
        event_frame = 12
    for i in range(n):
        off = (vel[0] * i, vel[1] * i)
        if event_frame is not None and i >= event_frame:
            if name == "vanish_midway":
                frames.append(painter.frame(None))
            else:  # scene_cut: a different world entirely
                frames.append(cut_painter.frame((170, 30)))
            truth.append(None)
        else:
            frames.append(painter.frame(
                (start[0] + off[0], start[1] + off[1])))
            truth.append(off)
    mask = np.zeros((_H, _W), dtype=bool)
    mask[start[1]:start[1] + _PATCH_H,
         start[0]:start[0] + _PATCH_W] = True
    return frames, mask, truth, {
        "start": start, "event_frame": event_frame, "notes": fx.notes}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


def score_fixture(name: str) -> dict[str, Any]:
    """Run the real tracker on fixture ``name``; report the truth.

    Offset/IoU/centroid errors are computed over the frames where
    ground truth exists. For the event fixtures, ``holds_after_
    event`` records whether the offset stayed at its last true fix
    once the content was gone (chasing background is the failure
    this exists to catch).
    """
    frames, mask, truth, meta = build_sequence(name)
    res = mask_track.track_sequence(frames, mask)
    offset_errs: list[float] = []
    ious: list[float] = []
    centroid_errs: list[float] = []
    for i, t in enumerate(truth):
        if t is None:
            continue
        # a held result carries only the anchor offset — the mask
        # stays there for every frame, which is what gets scored.
        got = res.offsets[i] if i < len(res.offsets) else res.offsets[-1]
        offset_errs.append(max(abs(got[0] - t[0]), abs(got[1] - t[1])))
        truth_mask = mask_track.shift_mask(mask, t[0], t[1]) > 127
        got_mask = mask_track.shift_mask(mask, got[0], got[1]) > 127
        ious.append(_iou(got_mask, truth_mask))
        cx_t = mask_track.mask_centroid(truth_mask)
        cx_g = mask_track.mask_centroid(got_mask)
        centroid_errs.append(
            ((cx_t[0] - cx_g[0]) ** 2 + (cx_t[1] - cx_g[1]) ** 2) ** 0.5)
    first_lost = next((i for i, x in enumerate(res.lost) if x), None)
    holds: bool | None = None
    ev = meta["event_frame"]
    if ev is not None:
        anchor_off = res.offsets[ev - 1]
        holds = all(
            abs(o[0] - anchor_off[0]) <= 2
            and abs(o[1] - anchor_off[1]) <= 2
            for o in res.offsets[ev:])
    return {
        "status": res.status,
        "reason": res.reason,
        "frames": res.frames,
        "mean_offset_err_px": (float(np.mean(offset_errs))
                               if offset_errs else None),
        "max_offset_err_px": (float(np.max(offset_errs))
                              if offset_errs else None),
        "mean_centroid_err_px": (float(np.mean(centroid_errs))
                                 if centroid_errs else None),
        "min_iou": float(np.min(ious)) if ious else None,
        "first_lost_frame": first_lost,
        "lost_frames": res.lost_count,
        "holds_after_event": holds,
        "max_displacement_px": res.max_displacement,
        "min_score": float(np.min(res.scores)) if res.scores else None,
    }
