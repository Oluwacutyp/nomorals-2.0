"""One-shot background removal: the composed ladder (52-websites #14).

Remove.bg is one job — a photo in, a transparent-background PNG
out. Devon already had both halves of that job in separate rooms:

- the *model* rung: :mod:`nomorals.media_edit.segment`
  (rembg, profile-aware model choice, honest ``SegmentationError``
  when rembg is missing);
- the *classical* ladder: :mod:`.masks` (SAM → GrabCut → flood →
  box+feather) with the IoU-scored corpus behind it.

This module composes them into one ladder with one answer shape.
``tier="auto"`` runs the best rung the machine can actually run —
the rembg model rung first where provisioned, then the classical
ladder — and every cutout says which rung served it (the §1.1d
honesty rule). A rung that isn't provisioned is skipped with its
reason recorded; when nothing can run, the error names every rung
and why — never a silent quality downgrade.

Judgment call, stated plainly: the report sketch says "classical
first, model rung where provisioned". This ladder runs the model
rung first *where provisioned* and the classical ladder as the
always-available foundation underneath it. The classical ladder
is what makes the tool work with zero provisioning; the model
rung is first because it is the quality rung — and because the
live chat path already uses rembg, so a classical-first default
would have been a live quality regression on EC2 the day this
shipped. The deviation is this paragraph, not a silent choice:
the parent can overrule it and the ladder order is one tuple.

Two guards keep the classical rungs honest:

- **Border-colour validation.** A classical mask is only a subject
  if its mean colour differs from the image border's mean colour
  (the border is near-certainly background). Flood-fill from the
  image centre on a ring photo would otherwise "find" the hole —
  background the same colour as the border — and serve it as the
  subject. A mask that matches the border fails the rung and the
  ladder continues; the last-resort box tier is exempt (it is
  labelled approximate, never presented as a found subject).
- **Coverage notes.** A mask covering >98% or <2% of the frame
  is reported as such — the subject may fill the frame, or may
  not be centred, or may match the background too closely for
  the classical rungs (the trained-model moat the report names).

Hair and transparent objects are a trained-model moat: a classical
cutout carries that note, and the model rung is named as the
sharper route — never faked.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

from . import ImgGenError
from .masks import MaskResult, auto_mask, mask_status

__all__ = [
    "BgRemoveResult",
    "BGREMOVE_TIERS",
    "BGREMOVE_LADDER",
    "remove_background",
    "remove_background_file",
    "cutout_from_mask",
    "cutout_coverage",
    "bgremove_status",
]

#: The full rung vocabulary, best first.
BGREMOVE_TIERS = ("rembg", "sam", "grabcut", "flood", "box")

#: ``tier="auto"`` walks this order. See the module docstring for
#: why the model rung leads: it is the quality rung, and the live
#: chat path already used it — the classical ladder underneath is
#: the always-available foundation.
BGREMOVE_LADDER = BGREMOVE_TIERS

#: A classical mask whose mean colour is within this summed
#: absolute channel distance of the border mean is background,
#: not a subject (see the module docstring). 72 = 24 per channel:
#: a subject that close to its background in mean colour is
#: inseparable by any colour method anyway.
BG_SUBJECT_MIN_BORDER_DISTANCE = 72.0


@dataclass
class BgRemoveResult:
    """A cutout plus the record of how it was made."""

    image: object  # RGBA PIL image
    mask: object  # L-mode PIL image, white = subject
    rung: str  # the tier that produced the mask
    detail: str  # human-readable provenance of the rung
    coverage: float  # fraction of the frame the mask covers
    seed: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    attempts: list = field(default_factory=list)


def _centre(image) -> tuple[float, float]:
    w, h = image.size
    return (w / 2.0, h / 2.0)


def _inset_rect(image, *, margin: float) -> tuple[float, float, float,
                                                  float]:
    w, h = image.size
    m = max(0.0, min(0.4, float(margin)))
    return (w * m, h * m, w * (1.0 - m), h * (1.0 - m))


def _border_mean(image) -> tuple[float, float, float]:
    """Mean RGB of a 4px border frame — the background prior."""
    import numpy as np

    arr = np.asarray(image.convert("RGB")).astype(np.float64)
    h, w = arr.shape[:2]
    band = max(1, min(4, min(w, h) // 4))
    frame = np.concatenate([
        arr[:band, :, :].reshape(-1, 3),
        arr[-band:, :, :].reshape(-1, 3),
        arr[:, :band, :].reshape(-1, 3),
        arr[:, -band:, :].reshape(-1, 3),
    ])
    mean = frame.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _mask_mean(image, mask) -> tuple[float, float, float]:
    """Mean RGB of the masked region."""
    import numpy as np

    arr = np.asarray(image.convert("RGB")).astype(np.float64)
    sel = np.asarray(mask.convert("L")) > 127
    if not sel.any():
        return (0.0, 0.0, 0.0)
    picked = arr[sel]
    mean = picked.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _looks_like_subject(image, mask) -> tuple[bool, str]:
    """Border-colour validation: is the mask a subject, or the
    background wearing a mask? Returns (ok, reason)."""
    mine = _mask_mean(image, mask)
    border = _border_mean(image)
    dist = sum(abs(a - b) for a, b in zip(mine, border, strict=True))
    if dist < BG_SUBJECT_MIN_BORDER_DISTANCE:
        return (False,
                f"mask mean colour {tuple(round(v) for v in mine)} "
                f"matches the border {tuple(round(v) for v in border)} "
                f"(distance {dist:.0f} < "
                f"{BG_SUBJECT_MIN_BORDER_DISTANCE:.0f}) — the "
                "region is background, not a subject")
    return (True, "")


def cutout_from_mask(image, mask):
    """Apply an L mask as the alpha channel → RGBA cutout."""
    from PIL import Image

    base = image.convert("RGB")
    out = Image.new("RGBA", base.size)
    out.paste(base, (0, 0))
    out.putalpha(mask.convert("L"))
    return out


def _coverage(mask) -> float:
    import numpy as np

    sel = np.asarray(mask.convert("L")) > 127
    return float(sel.mean()) if sel.size else 0.0


def cutout_coverage(image) -> float:
    """Fraction of a cutout's frame that is opaque — the same
    coverage the ladder reports, read back off a saved RGBA
    file (its alpha channel is the mask)."""
    return _coverage(image.convert("RGBA").split()[3])


def _rembg_cutout(image, *, model: str, model_fn=None):
    """The model rung. Returns (RGBA, mask, detail). Raises with
    the rung's own honest reason when it cannot run."""
    if model_fn is not None:
        out = model_fn(image, model=model)
        detail = "injected model rung"
    else:
        from ...media_edit import segment as _segment

        try:
            out = _segment.remove_background(image, model=model or "")
        except _segment.SegmentationError as exc:
            raise ImgGenError(f"rembg rung unavailable: {exc}") from exc
        detail = (f"rembg ({model or 'profile default'}) — "
                  "trained segmentation model")
    from PIL import Image

    rgba = out.convert("RGBA") if isinstance(out, Image.Image) \
        else Image.fromarray(out).convert("RGBA")
    mask = rgba.split()[3]
    return rgba, mask, detail


def _mask_rung(image, *, tier: str, point, box, feather: int,
               mask_fn=None) -> MaskResult:
    """One classical rung through the real ladder (or an injected
    stand-in for tests). Raises ImgGenError when it cannot run."""
    fn = mask_fn or auto_mask
    if tier == "sam":
        return fn(image, point=point, tier="sam",
                  feather_radius=feather)
    if tier == "grabcut":
        return fn(image, box=box, tier="grabcut",
                  feather_radius=feather)
    if tier == "flood":
        return fn(image, point=point, tier="flood",
                  feather_radius=feather)
    if tier == "box":
        return fn(image, box=box, tier="box",
                  feather_radius=feather)
    raise ImgGenError(f"unknown classical tier {tier!r}")


def remove_background(image, *, tier: str = "auto", model: str = "",
                      margin: float = 0.06, feather: int = 2,
                      model_fn=None, mask_fn=None) -> BgRemoveResult:
    """Photo → transparent-background cutout, best rung available.

    ``tier``: ``"auto"`` (the ladder) or one forced rung —
    ``"rembg"`` | ``"sam"`` | ``"grabcut"`` | ``"flood"`` |
    ``"box"``. A forced rung that cannot run raises naming the
    rung and why. ``model`` picks the rembg model (empty = the
    profile default from ``media_edit.segment``). ``margin`` is
    the inset fraction for the GrabCut/box rect. ``model_fn`` /
    ``mask_fn`` are injectable seams for tests (a fake model
    rung, a fake mask ladder); production callers leave them
    unset.
    """
    from PIL import Image

    if Image is None:  # pragma: no cover - Pillow is a hard dep
        raise ImgGenError("background removal needs Pillow installed")
    if not isinstance(image, Image.Image):
        raise ImgGenError(
            f"remove_background takes a PIL image, got "
            f"{type(image).__name__}")
    wanted = (tier or "auto").strip().lower()
    if wanted != "auto" and wanted not in BGREMOVE_TIERS:
        raise ImgGenError(
            f"unknown tier {wanted!r}; use auto|"
            + "|".join(BGREMOVE_TIERS))
    order = BGREMOVE_LADDER if wanted == "auto" else (wanted,)
    point = _centre(image)
    box = _inset_rect(image, margin=margin)
    seed = {"point": [round(point[0], 1), round(point[1], 1)],
            "box": [round(v, 1) for v in box]}
    attempts: list = []
    for rung in order:
        try:
            if rung == "rembg":
                rgba, mask, detail = _rembg_cutout(
                    image, model=model or "", model_fn=model_fn)
                res_mask, res_detail = mask, detail
                res_rung = "rembg"
            else:
                res = _mask_rung(image, tier=rung, point=point,
                                 box=box, feather=feather,
                                 mask_fn=mask_fn)
                res_mask, res_detail, res_rung = (
                    res.mask, res.detail, res.tier)
                rgba = None
            if (res_rung not in ("box+feather", "rembg")
                    and _coverage(res_mask) <= 0.98):
                # The model rung is a trained judgment, not a
                # colour heuristic — it is trusted as returned
                # (an all-opaque model mask is caught by the
                # coverage note below instead).
                #
                # A mask covering (nearly) the whole frame is
                # exempt too: the subject fills the frame, so
                # mask and border are the same thing by
                # construction — the coverage note below says
                # so, and rejecting it would serve a worse
                # rectangle.
                ok, why = _looks_like_subject(image, res_mask)
                if not ok:
                    attempts.append({"rung": rung, "ok": False,
                                     "reason": why})
                    continue
        except ImgGenError as exc:
            attempts.append({"rung": rung, "ok": False,
                             "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - rung honesty
            attempts.append({"rung": rung, "ok": False,
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        attempts.append({"rung": rung, "ok": True,
                         "reason": res_detail})
        if rgba is None:
            rgba = cutout_from_mask(image, res_mask)
        coverage = _coverage(res_mask)
        notes: list = []
        if coverage > 0.98:
            notes.append(
                f"mask covers {coverage:.0%} of the frame — the "
                "subject may fill the frame; nearly nothing was "
                "removed")
        elif coverage < 0.02:
            notes.append(
                f"mask covers {coverage:.0%} of the frame — almost "
                "everything was removed; the subject may not be "
                "centred, or may match the background too closely "
                "for the classical rungs")
        if res_rung != "rembg":
            notes.append(
                "classical rung — hair and transparent objects "
                "will be approximate; the rembg model rung is "
                "sharper where provisioned (see bgremove_status)")
        if res_rung == "box+feather":
            notes.append(
                "box+feather is the last resort: a feathered "
                "rectangle, not a found subject — prefer "
                "sam/grabcut/flood where the machine can run them")
        return BgRemoveResult(
            image=rgba, mask=res_mask, rung=res_rung,
            detail=res_detail, coverage=coverage, seed=seed,
            notes=notes, attempts=attempts)
    raise ImgGenError(
        "no background-removal rung could run: "
        + "; ".join(
            f"{a['rung']} ({a['reason']})" for a in attempts)
        + ". Install rembg (pip install rembg) or OpenCV "
          "(pip install opencv-python) for the sharper rungs.")


def remove_background_file(src, dst, *, tier: str = "auto",
                           model: str = "", margin: float = 0.06,
                           feather: int = 2) -> dict:
    """File in → transparent PNG out. The output is always PNG
    (alpha needs it): a non-PNG ``dst`` gets its suffix replaced
    and the record says so."""
    from ...media_edit.images import load_image, save_image

    src_path = Path(src)
    if not src_path.exists():
        raise ImgGenError(f"no such image: {src}")
    dst_path = Path(dst)
    if dst_path.suffix.lower() != ".png":
        dst_path = dst_path.with_suffix(".png")
    img = load_image(src_path)
    try:
        res = remove_background(img, tier=tier, model=model,
                                margin=margin, feather=feather)
    finally:
        with contextlib.suppress(Exception):
            img.close()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(res.image, dst_path, fmt="PNG")
    return {
        "input": str(src_path),
        "output": str(dst_path),
        "rung": res.rung,
        "detail": res.detail,
        "coverage": round(res.coverage, 4),
        "notes": list(res.notes),
        "bytes": dst_path.stat().st_size,
    }


def bgremove_status() -> dict:
    """Per-rung probe of the whole background-removal ladder —
    the mask ladder's status plus the rembg model rung, so a
    surface can say what will serve a cutout before trying."""
    import importlib.util

    status = {"ladder": list(BGREMOVE_LADDER),
              "masks": mask_status()}
    rembg_spec = importlib.util.find_spec("rembg")
    if rembg_spec is None:
        status["rembg"] = {
            "usable": False,
            "reason": "rembg not installed "
                      "(pip install rembg — the u2netp model is "
                      "only ~4.7 MB)",
            "model": None,
            "model_cached": False,
        }
        return status
    from ...media_edit.segment import MODELS, default_model

    want = default_model()
    cache = Path.home() / ".u2net" / f"{want}.onnx"
    cached = cache.exists()
    status["rembg"] = {
        "usable": True,
        "reason": ("ok" if cached else
                   f"ok — the {want} model downloads on first "
                   f"use (~{MODELS[want]['size_mb']:.0f} MB)"),
        "model": want,
        "model_cached": cached,
    }
    return status
