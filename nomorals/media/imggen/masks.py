"""Auto-mask generation: text / point / box → inpaint mask.

Tiered, honest segmentation:

1. **SAM** (Segment Anything) when importable *and* a checkpoint is
   on disk — real neural segmentation, best masks.
2. **GrabCut** via OpenCV when ``cv2`` is importable — needs a box
   (from ``box_predictor`` or a point expanded to a box) or a point.
3. **Region-grow flood fill** (pure numpy): from a point, grows over
   color-similar pixels — real segmentation for solid-ish objects,
   no dependencies.
4. **Box + feather** last resort: from ``box_predictor(image, text)``
   (a vision/LLM locator callback) or an explicit box/point.

Every path labels itself in the returned mask's ``info`` so callers
know the mask's provenance. All torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter

from . import ImgGenError

__all__ = [
    "MaskResult",
    "auto_mask",
    "box_mask",
    "point_flood_mask",
    "sam_status",
    "SAM_AVAILABLE",
    "CV2_AVAILABLE",
]


def _optional_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


_cv2 = _optional_import("cv2")
CV2_AVAILABLE = _cv2 is not None

_sam_pkg = _optional_import("segment_anything")
_ultra = _optional_import("ultralytics")
SAM_AVAILABLE = _sam_pkg is not None or _ultra is not None


def sam_status() -> dict:
    """Why SAM is or isn't usable — honest, no silent gaps."""
    import os

    candidates = [
        os.path.expanduser("~/.nomorals/models/sam_vit_h.pth"),
        os.path.expanduser("~/.nomorals/models/sam_vit_l.pth"),
        os.path.expanduser("~/.nomorals/models/sam2.1_b.pt"),
    ]
    ckpt = next((c for c in candidates if os.path.exists(c)), None)
    return {
        "package": ("segment-anything" if _sam_pkg is not None
                    else "ultralytics" if _ultra is not None else None),
        "checkpoint": ckpt,
        "usable": SAM_AVAILABLE and ckpt is not None,
        "reason": (
            "ok" if SAM_AVAILABLE and ckpt else
            "no SAM package installed (pip install segment-anything)"
            if ckpt and not SAM_AVAILABLE else
            f"no SAM checkpoint on disk; place one at {candidates[0]}"
        ),
    }


@dataclass
class MaskResult:
    """An auto-generated mask plus its provenance."""

    mask: Image.Image  # L mode, white = repaint
    tier: str          # e.g. "sam", "grabcut", "flood", "box+feather"
    detail: str        # human-readable provenance


def box_mask(image_size: tuple[int, int], box: tuple[int, int, int, int],
             feather_radius: int = 8) -> Image.Image:
    """Box (x0, y0, x1, y1) → feathered L mask."""
    x0, y0, x1, y1 = (round(v) for v in box)
    w, h = image_size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ImgGenError(f"degenerate box {box}")
    m = Image.new("L", (w, h), 0)
    m.paste(255, (x0, y0, x1, y1))
    if feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_radius))
    return m


def point_flood_mask(image: Image.Image, point: tuple[int, int],
                     tolerance: int = 32,
                     feather_radius: int = 6) -> Image.Image:
    """Region-grow from ``point`` over color-similar pixels (numpy).

    BFS over 4-neighbours with a per-pixel L1 color distance against
    the seed pixel; pixels within ``tolerance`` join the region.
    Real segmentation for solid-ish objects; leaks on busy textures —
    the tier label says so.
    """
    import numpy as np

    img = image.convert("RGB")
    w, h = img.size
    px, py = round(point[0]), round(point[1])
    if not (0 <= px < w and 0 <= py < h):
        raise ImgGenError(f"point {point} outside image {w}x{h}")
    arr = np.asarray(img).astype(np.int16)
    seed = arr[py, px]
    region = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)

    # Iterative frontier expansion (vectorized per wavefront).
    # Candidates are deduplicated each wave: without this the same
    # pixel re-enters via every neighbouring frontier pixel and the
    # wavefront grows ~4x per iteration (exponential blowup → OOM).
    frontier_y = np.array([py])
    frontier_x = np.array([px])
    region[py, px] = True
    visited[py, px] = True
    while frontier_y.size:
        nys = np.concatenate([frontier_y - 1, frontier_y + 1,
                              frontier_y, frontier_y])
        nxs = np.concatenate([frontier_x, frontier_x,
                              frontier_x - 1, frontier_x + 1])
        # Bounds first (never index `visited` out of range)...
        inb = ((nys >= 0) & (nys < h) & (nxs >= 0) & (nxs < w))
        nys, nxs = nys[inb], nxs[inb]
        if not nys.size:
            break
        # ...then drop visited, then drop duplicates.
        fresh = ~visited[nys, nxs]
        nys, nxs = nys[fresh], nxs[fresh]
        if not nys.size:
            break
        _, uniq = np.unique(nys * w + nxs, return_index=True)
        nys, nxs = nys[uniq], nxs[uniq]
        close = (np.abs(arr[nys, nxs] - seed).sum(axis=1)
                 <= tolerance * 3)
        visited[nys, nxs] = True
        take_y, take_x = nys[close], nxs[close]
        region[take_y, take_x] = True
        frontier_y, frontier_x = take_y, take_x

    m = Image.fromarray((region * 255).astype(np.uint8), "L")
    if feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_radius))
    return m


def _sam_mask(image: Image.Image, point=None,
              box=None) -> MaskResult | None:
    """SAM tier. Returns None (with no side effects) when unusable."""
    st = sam_status()
    if not st["usable"]:
        return None
    try:
        import numpy as np

        ckpt = st["checkpoint"]
        if _sam_pkg is not None:
            from segment_anything import (SamPredictor,
                                          sam_model_registry)
            kind = ("vit_h" if "vit_h" in ckpt else
                    "vit_l" if "vit_l" in ckpt else "vit_b")
            sam = sam_model_registry[kind](checkpoint=ckpt)
            predictor = SamPredictor(sam)
            predictor.set_image(np.asarray(image.convert("RGB")))
            if point is not None:
                masks, _, _ = predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([1]),
                    multimask_output=False)
            elif box is not None:
                masks, _, _ = predictor.predict(
                    box=np.array(box), multimask_output=False)
            else:
                return None
            m = Image.fromarray((masks[0] * 255).astype(np.uint8), "L")
        else:  # ultralytics SAM
            from ultralytics import SAM

            model = SAM(ckpt)
            results = model.predict(np.asarray(image.convert("RGB")),
                                    points=[point] if point else None,
                                    bboxes=[box] if box else None,
                                    verbose=False)
            r = results[0]
            if r.masks is None:
                return None
            m = Image.fromarray(
                (r.masks.data[0].cpu().numpy() * 255).astype(np.uint8),
                "L").resize(image.size, Image.BILINEAR)
        return MaskResult(mask=m, tier="sam",
                          detail=f"SAM ({st['package']}) segmentation")
    except Exception as exc:  # honest: report, don't silently degrade
        raise ImgGenError(f"SAM failed: {exc}") from exc


def _grabcut_mask(image: Image.Image, box, point,
                  feather_radius: int) -> MaskResult | None:
    """GrabCut tier via OpenCV. None when cv2 is missing."""
    if not CV2_AVAILABLE:
        return None
    import numpy as np

    img = image.convert("RGB")
    arr = np.asarray(img)
    bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    if box is not None:
        x0, y0, x1, y1 = (round(v) for v in box)
        rect = (max(0, x0), max(0, y0),
                min(w, x1) - max(0, x0), min(h, y1) - max(0, y0))
        if rect[2] <= 0 or rect[3] <= 0:
            raise ImgGenError(f"degenerate box {box}")
        _cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, 5,
                     _cv2.GC_INIT_WITH_RECT)
        seed = f"grabcut(rect {rect})"
    elif point is not None:
        px, py = round(point[0]), round(point[1])
        r = max(8, min(w, h) // 12)
        gc_mask[py, px] = _cv2.GC_FGD
        gc_mask[max(0, py - r):py + r, max(0, px - r):px + r] = \
            _cv2.GC_PR_FGD
        gc_mask[0, :] = gc_mask[-1, :] = _cv2.GC_BGD
        gc_mask[:, 0] = gc_mask[:, -1] = _cv2.GC_BGD
        _cv2.grabCut(bgr, gc_mask, None, bgd, fgd, 5,
                     _cv2.GC_INIT_WITH_MASK)
        seed = f"grabcut(point {point})"
    else:
        return None
    binm = np.where((gc_mask == _cv2.GC_FGD)
                    | (gc_mask == _cv2.GC_PR_FGD), 255, 0
                    ).astype(np.uint8)
    m = Image.fromarray(binm, "L")
    if feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_radius))
    return MaskResult(mask=m, tier="grabcut",
                      detail=f"OpenCV {seed}")


def auto_mask(image: Image.Image, *,
              text: str | None = None,
              point: tuple[float, float] | None = None,
              box: tuple[float, float, float, float] | None = None,
              box_predictor=None,
              feather_radius: int = 8,
              tier: str = "auto") -> MaskResult:
    """text / point / box → inpaint mask, best tier available.

    ``box_predictor``: optional callable ``(image, text) -> box`` —
    a vision/LLM locator that turns a text description into
    ``(x0, y0, x1, y1)``. Required when only ``text`` is given and no
    SAM checkpoint exists.

    ``tier``: ``"auto"`` (SAM → GrabCut → flood/box+feather) or force
    one of ``"sam"``, ``"grabcut"``, ``"flood"``, ``"box"``.

    Raises :class:`ImgGenError` with an honest reason when no tier
    can produce a mask (e.g. text given but no locator and no SAM).
    """
    tiers = ("sam", "grabcut", "flood", "box")
    if tier != "auto" and tier not in tiers:
        raise ImgGenError(f"unknown tier {tier!r}; use auto|"
                          + "|".join(tiers))
    wanted = tiers if tier == "auto" else (tier,)

    if box is None and text is not None and box_predictor is not None:
        box = box_predictor(image, text)
    if box is None and point is None and text is not None:
        raise ImgGenError(
            "auto_mask from text needs a box_predictor locator or a "
            "SAM checkpoint on disk — neither is available. Pass "
            "point= or box= instead.")

    for t in wanted:
        if t == "sam" and (point is not None or box is not None):
            res = _sam_mask(image, point=point, box=box)
            if res is not None:
                return res
        elif t == "grabcut" and (point is not None or box is not None):
            res = _grabcut_mask(image, box, point, feather_radius)
            if res is not None:
                return res
        elif t == "flood" and point is not None:
            m = point_flood_mask(image, point,
                                 feather_radius=feather_radius)
            return MaskResult(mask=m, tier="flood",
                              detail="numpy region-grow from point "
                                     "(solid-ish objects; may leak on "
                                     "busy texture)")
        elif t == "box":
            use_box = box
            if use_box is None and point is not None:
                # Point → adaptive box: flood to find the region,
                # take its bounding box, fall back to a fixed box.
                import numpy as np

                region = point_flood_mask(image, point,
                                          feather_radius=0)
                ys, xs = np.nonzero(np.asarray(region) > 127)
                if xs.size > 4:
                    pad = 6
                    use_box = (xs.min() - pad, ys.min() - pad,
                               xs.max() + pad, ys.max() + pad)
                else:
                    r = max(16, min(image.size) // 8)
                    use_box = (point[0] - r, point[1] - r,
                               point[0] + r, point[1] + r)
            if use_box is None:
                continue
            m = box_mask(image.size, use_box,
                         feather_radius=feather_radius)
            return MaskResult(
                mask=m, tier="box+feather",
                detail="bounding box + gaussian feather "
                       "(approximate; prefer sam/grabcut/flood)")

    raise ImgGenError(
        f"no mask tier could run (tried {wanted}; "
        f"SAM usable={sam_status()['usable']}, cv2={CV2_AVAILABLE}). "
        "Provide point= or box=, or install SAM/OpenCV.")
