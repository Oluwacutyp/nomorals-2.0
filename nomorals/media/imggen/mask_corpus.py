"""IoU-scored mask corpus: synthetic fixtures with exact ground truth.

Plan §1.1d acceptance demands mask quality be *measured, not
vibes*: corpus stills with known objects, an IoU floor per fixture
recorded in a manifest, and every mask path reporting which rung
produced it. This module is the corpus half of that:

- :data:`FIXTURES` — deterministic synthetic scenes (seeded numpy,
  drawn with PIL). Every scene is built *from* its ground-truth
  mask, so the truth is exact by construction — no downloads, no
  hand-labelled approximations.
- :func:`build_fixture` — ``(image, ground_truth, prompt)`` for a
  fixture name. Same seed → same bytes, on any machine.
- :func:`score_fixture` — runs the real ladder (:func:`auto_mask`)
  per rung against the ground truth and reports the IoU each rung
  actually achieved, plus the tier that produced it.

The recorded floors + measured values live in
``tests/golden/mask_corpus.json``; ``tests/test_mask_corpus.py``
rebuilds every fixture, re-scores the ladder, and fails when a
rung that is usable on the machine falls below its floor — a rung
regression is a test failure, not a surprise in production.

SAM is deliberately *not* scored here: it needs a checkpoint and
torch on the machine, and inventing a number for it would be the
vibe this corpus exists to replace. The manifest records that
honestly; every rung :func:`mask_status` reports usable is scored.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dep
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None; ImageDraw = None  # type: ignore[assignment]

from . import ImgGenError
from .masks import auto_mask

__all__ = [
    "CorpusFixture",
    "FIXTURES",
    "fixture_names",
    "build_fixture",
    "iou_score",
    "score_fixture",
]

_SIZE = (192, 192)


@dataclass(frozen=True)
class CorpusFixture:
    """One corpus scene: how to draw it and how it is prompted.

    ``prompt`` mirrors the :func:`auto_mask` inputs:
    ``{"kind": "point", "point": [x, y]}``,
    ``{"kind": "box", "box": [x0, y0, x1, y1]}``, or
    ``{"kind": "text", "box": [...]}`` — for text, ``box`` is what
    a locator (``box_predictor``) would return for the object; the
    corpus tests the ladder, the locator is a fixture input.
    """

    name: str
    seed: int
    prompt: dict
    notes: str


FIXTURES: tuple[CorpusFixture, ...] = (
    CorpusFixture(
        name="disc_on_field",
        seed=11,
        prompt={"kind": "point", "point": [96, 96]},
        notes="Flat disc on a flat field; the ladder's home turf — "
              "every rung should be near-exact.",
    ),
    CorpusFixture(
        name="block_on_field",
        seed=12,
        prompt={"kind": "box", "box": [48, 60, 150, 140]},
        notes="Axis-aligned block prompted with its exact box.",
    ),
    CorpusFixture(
        name="block_gradient_bg",
        seed=13,
        prompt={"kind": "point", "point": [96, 100]},
        notes="Flat block on a smooth gradient field — the "
              "background varies, the object does not.",
    ),
    CorpusFixture(
        name="speckled_disc",
        seed=14,
        prompt={"kind": "point", "point": [96, 96]},
        notes="Disc filled with two-tone speckle. Flood fill is "
              "the weak rung here — pixels whose speckle tone sits "
              "far from the seed pixel's tone never join its "
              "region (measured ~0.73 against GrabCut's ~0.98); "
              "the manifest records that gap. GrabCut's colour "
              "models are the rung for this class.",
    ),
    CorpusFixture(
        name="thin_ring",
        seed=15,
        prompt={"kind": "point", "point": [148, 96]},
        notes="Annulus prompted on the ring: the hole is ground-"
              "truth background and must stay unmasked.",
    ),
    CorpusFixture(
        name="twin_discs",
        seed=16,
        prompt={"kind": "point", "point": [58, 96]},
        notes="Two discs, two colours; the point picks one — the "
              "other disc is ground-truth background.",
    ),
    CorpusFixture(
        name="disc_via_locator",
        seed=11,
        prompt={"kind": "text", "box": [40, 40, 152, 152]},
        notes="The disc_on_field scene reached through a text "
              "prompt; the fixture's box is what a locator returns "
              "for it, so the auto path is scored end-to-end.",
    ),
)


def fixture_names() -> list[str]:
    return [f.name for f in FIXTURES]


def _fixture(name: str) -> CorpusFixture:
    for f in FIXTURES:
        if f.name == name:
            return f
    raise ImgGenError(
        f"unknown mask corpus fixture {name!r}; "
        f"use one of {fixture_names()}")


def _noisy(rng, color, amp: int = 2) -> tuple[int, int, int]:
    return tuple(
        int(min(255, max(0, c + rng.integers(-amp, amp + 1))))
        for c in color)


def _paint_disc(arr, gt, rng, center, radius, color, *, speckle=0):
    h, w = gt.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    if speckle:
        noise = rng.integers(-speckle, speckle + 1,
                             size=(h, w, 1))
        obj = np.asarray(color, dtype=np.int16) + noise
    else:
        obj = np.asarray([_noisy(rng, color)], dtype=np.int16)
    arr[inside] = obj[inside] if speckle else obj[0]
    gt |= inside
    return inside


def _paint_rect(arr, gt, rng, box, color):
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = _noisy(rng, color)
    gt[y0:y1, x0:x1] = True


def build_fixture(name: str):
    """``(image, ground_truth, prompt)`` for fixture ``name``.

    ``ground_truth`` is a boolean ``(h, w)`` array — the exact
    object pixels, by construction. Deterministic: the same name
    builds byte-identical pixels on any machine.
    """
    if np is None or Image is None:
        raise ImgGenError(
            "mask corpus needs numpy and Pillow installed")
    fx = _fixture(name)
    rng = np.random.default_rng(fx.seed)
    w, h = _SIZE
    gt = np.zeros((h, w), dtype=bool)

    if name in ("disc_on_field", "disc_via_locator"):
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:] = _noisy(rng, (34, 44, 118))
        _paint_disc(arr, gt, rng, (96, 96), 56, (214, 92, 40))
    elif name == "block_on_field":
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:] = _noisy(rng, (96, 110, 88))
        _paint_rect(arr, gt, rng, (48, 60, 150, 140), (196, 178, 52))
    elif name == "block_gradient_bg":
        grad = np.linspace(20, 150, w, dtype=np.int16)
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:, :, 0] = grad[None, :]
        arr[:, :, 1] = grad[None, :] // 2
        arr[:, :, 2] = 60
        _paint_rect(arr, gt, rng, (40, 52, 152, 148), (228, 196, 84))
    elif name == "speckled_disc":
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:] = _noisy(rng, (40, 90, 60))
        _paint_disc(arr, gt, rng, (96, 96), 58, (200, 120, 60),
                    speckle=28)
    elif name == "thin_ring":
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:] = _noisy(rng, (52, 52, 60))
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = (xx - 96) ** 2 + (yy - 96) ** 2
        ring = (d2 <= 64 ** 2) & (d2 > 40 ** 2)
        arr[ring] = _noisy(rng, (220, 214, 190))
        gt |= ring
    elif name == "twin_discs":
        arr = np.zeros((h, w, 3), dtype=np.int16)
        arr[:] = _noisy(rng, (30, 30, 44))
        _paint_disc(arr, gt, rng, (58, 96), 34, (204, 84, 52))
        other = np.zeros((h, w), dtype=bool)
        _paint_disc(arr, other, rng, (134, 96), 34, (62, 178, 92))
    else:  # pragma: no cover - FIXTURES and builder stay in sync
        raise ImgGenError(f"fixture {name!r} has no builder")

    image = Image.fromarray(arr.astype(np.uint8), "RGB")
    return image, gt, dict(fx.prompt)


def iou_score(mask, ground_truth) -> float:
    """Intersection-over-union of a mask against ground truth.

    ``mask`` is a PIL ``L`` image (white = selected) or a boolean
    array; it is thresholded at >127, matching how the ladder's
    feathered edges are consumed downstream.
    """
    if np is None:
        raise ImgGenError("iou_score needs numpy installed")
    if hasattr(mask, "convert"):  # PIL image
        pred = np.asarray(mask.convert("L")) > 127
    else:
        pred = np.asarray(mask, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ImgGenError(
            f"mask shape {pred.shape} != ground truth {gt.shape}")
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / union


def score_fixture(name: str, *, tiers=("auto", "grabcut", "flood",
                                       "box")) -> dict:
    """Run the real ladder for fixture ``name``, rung by rung.

    Returns ``{rung: {"tier": <producing tier>, "iou": float}}``
    for rungs that produced a mask, and
    ``{rung: {"error": <reason>}}`` for rungs that honestly could
    not (not usable on this machine, or not applicable to the
    fixture's prompt kind — e.g. flood needs a point).
    """
    image, gt, prompt = build_fixture(name)
    kind = prompt["kind"]
    point = (tuple(prompt["point"]) if kind == "point" else None)
    box = (tuple(prompt["box"]) if kind in ("box", "text") else None)
    predictor = None
    if kind == "text":
        want = tuple(prompt["box"])

        def predictor(image_, text, _want=want):
            return _want

    out: dict[str, dict] = {}
    for tier in tiers:
        if tier == "flood" and point is None:
            out[tier] = {"error": "flood needs a point prompt; "
                                  "fixture has none"}
            continue
        try:
            res = auto_mask(image, point=point, box=box,
                            box_predictor=predictor, tier=tier)
        except ImgGenError as exc:
            out[tier] = {"error": str(exc)}
            continue
        out[tier] = {"tier": res.tier, "iou": iou_score(res.mask, gt)}
    return out
