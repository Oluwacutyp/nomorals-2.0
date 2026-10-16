"""Image four (EDIT_GEN_UPGRADE 2.8).

* **Relighting after compositing** — :func:`relight_after_composite`
  re-lights a composited image with a directional gradient + warmth
  shift, so pasted subjects stop looking pasted. Pure PIL: an honest
  approximate relight, not a neural relighter (no neural relight
  model is installed — see :func:`neural_relight_available`).
* **Pose / structure control** — :func:`pose_control` builds a
  :class:`PoseTrack` on the existing :mod:`directed.pose_rig` rig and
  renders its first-frame skeleton as a ControlNet-class control
  image (black bg, coloured limbs). Same rig as motion — one organ.
* **Multi-ref identity** — :func:`multi_ref_identity` aggregates
  ≥2 reference faces (palette + embedding mean) into one identity
  descriptor; a single ref is refused (that is a face-swap, not an
  identity lock — see ``faceswap_guard``).
* **Text fidelity finish pass** — :func:`text_fidelity_finish` is the
  standard finish pass: detect → inpaint → render via the existing
  :mod:`media_edit.edittext` (``replace_text``), with per-replacement
  backend + match metadata reported honestly.
"""

from __future__ import annotations

from typing import Any

from ...media_edit.videos import MediaEditError

__all__ = [
    "neural_relight_available",
    "relight_after_composite",
    "pose_control",
    "pose_control_image",
    "multi_ref_identity",
    "text_fidelity_finish",
    "finish_pass",
]


def neural_relight_available() -> tuple[bool, str]:
    """Is a neural relighter installed? Honest probe (none is)."""
    return (False, "no neural relighter installed — relight_after_composite "
                   "is the deterministic PIL approximate relight")


def relight_after_composite(img: Any, *, light_dir: str = "top-left",
                            intensity: float = 0.35,
                            warmth: float = 0.0) -> Any:
    """Re-light a composited image (PIL, deterministic).

    ``light_dir``: top-left | top-right | bottom-left | bottom-right |
    top | left. ``intensity`` 0..1 — how strong the gradient is.
    ``warmth`` -1..1 — cool→warm shift on the lit side. The gradient
    multiplies brightness on the lit side and darkens the far side,
    then a warm/cool tint is blended in. Approximate by design.
    """
    from PIL import Image as _Image, ImageEnhance

    if intensity < 0 or intensity > 1:
        raise MediaEditError("relight intensity must be in [0, 1]")
    if warmth < -1 or warmth > 1:
        raise MediaEditError("relight warmth must be in [-1, 1]")
    base = img.convert("RGB")
    w, h = base.size
    dirs = {
        "top-left": (-1, -1), "top-right": (1, -1),
        "bottom-left": (-1, 1), "bottom-right": (1, 1),
        "top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0),
    }
    if light_dir not in dirs:
        raise MediaEditError(
            f"unknown light_dir {light_dir!r}; use: {sorted(dirs)}")
    dx, dy = dirs[light_dir]
    # gradient map: 1 at light corner, 1-intensity at far corner
    import math as _math
    grad = _Image.new("L", (w, h))
    px = grad.load()
    for y in range(h):
        ny = (y / max(1, h - 1)) * 2 - 1
        for x in range(w):
            nx = (x / max(1, w - 1)) * 2 - 1
            align = (nx * dx + ny * dy) / 2.0  # -1..1
            px[x, y] = int(255 * (1.0 - intensity * (1.0 - align) / 2.0))
    # multiply luminance by the gradient
    from PIL import ImageChops as _Chops
    lit = _Chops.multiply(base, _Image.merge("RGB", (grad, grad, grad)))
    out = _Image.blend(base, lit, 0.85)
    if abs(warmth) > 1e-6:
        r, g, b = out.split()
        shift = int(abs(warmth) * 28)
        if warmth > 0:
            r = r.point(lambda v: min(255, v + shift))
            b = b.point(lambda v: max(0, v - shift))
        else:
            b = b.point(lambda v: min(255, v + shift))
            r = r.point(lambda v: max(0, v - shift))
        out = _Image.merge("RGB", (r, g, b))
    return out


def pose_control(action: str, *, n_frames: int = 40, fps: float = 10.0,
                 width: int = 512, height: int = 512,
                 detected: Any = None) -> dict[str, Any]:
    """Pose/structure control for an image (ControlNet-class input).

    Builds a :class:`PoseTrack` for ``action`` on
    :mod:`directed.pose_rig` (retargeted onto ``detected`` keypoints
    when supplied), and renders the first frame as a skeleton control
    image. Returns ``{"track", "control_image", "action"}`` — the
    control image is what a ControlNet-class backend consumes; the
    track is the editable structure.
    """
    from ..directed.pose_rig import build_track, retarget
    track = build_track(action, n_frames=n_frames, fps=fps,
                        width=width, height=height)
    if detected is not None:
        track = retarget(track, detected)
    control = pose_control_image(track, frame=0)
    return {"track": track, "control_image": control, "action": action}


def pose_control_image(track: Any, *, frame: int = 0,
                       width: int | None = None,
                       height: int | None = None) -> Any:
    """One skeleton frame of a PoseTrack as a PIL control image."""
    from PIL import Image as _Image, ImageDraw as _Draw
    import numpy as _np
    from ..directed.pose_rig import BODY_LIMBS, HAND_LIMBS, N_BODY, N_HAND, R_HAND, L_HAND
    w = int(width or track.width)
    h = int(height or track.height)
    img = _Image.new("RGB", (w, h), (0, 0, 0))
    d = _Draw.Draw(img)
    kp = track.frames[int(frame)]
    pts = (kp * _np.array([w, h])).astype(int)
    for a, b in BODY_LIMBS:
        d.line([tuple(pts[a]), tuple(pts[b])], fill=(255, 255, 255), width=4)
    for sl in (R_HAND, L_HAND):
        hp = pts[sl]
        for a, b in HAND_LIMBS:
            d.line([tuple(hp[a]), tuple(hp[b])], fill=(0, 255, 255), width=2)
    for (x, y) in pts:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
    return img


def multi_ref_identity(refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Multi-reference identity: ≥2 refs → one identity descriptor.

    Each ref: ``{"palette": [[r,g,b], ...], "embedding": [float, ...],
    "wardrobe": str}``. Returns ``{"palette", "embedding", "wardrobe",
    "n_refs"}`` — mean palette + mean embedding + the modal wardrobe.
    One ref raises: identity needs corroboration, a single face is a
    swap, not a lock (see ``faceswap_guard``).
    """
    if len(refs) < 2:
        raise MediaEditError(
            "multi_ref_identity needs ≥2 references — a single face is "
            "a swap, not an identity lock")
    palettes: list[list[list[float]]] = []
    embs: list[list[float]] = []
    wardrobes: list[str] = []
    for r in refs:
        pal = r.get("palette") or []
        emb = r.get("embedding") or []
        if pal:
            palettes.append([[float(c) for c in px] for px in pal])
        if emb:
            embs.append([float(v) for v in emb])
        if r.get("wardrobe"):
            wardrobes.append(str(r["wardrobe"]))
    if not palettes and not embs:
        raise MediaEditError("multi_ref_identity refs carry no palette "
                             "or embedding")
    mean_pal: list[list[float]] = []
    if palettes:
        n = min(len(p) for p in palettes)
        for i in range(n):
            mean_pal.append([sum(p[i][c] for p in palettes) / len(palettes)
                             for c in range(3)])
    mean_emb: list[float] = []
    if embs:
        n = min(len(e) for e in embs)
        mean_emb = [sum(e[i] for e in embs) / len(embs) for i in range(n)]
    wardrobe = ""
    if wardrobes:
        wardrobe = max(set(wardrobes), key=wardrobes.count)
    return {"palette": mean_pal, "embedding": mean_emb,
            "wardrobe": wardrobe, "n_refs": len(refs)}


def text_fidelity_finish(img: Any,
                         replacements: list[tuple[str, str]], *,
                         lang: str = "eng") -> tuple[Any, dict[str, Any]]:
    """Standard finish pass: detect → inpaint → render, per replacement.

    Delegates to :func:`media_edit.edittext.replace_text` — the one
    text-replacement path. Returns ``(image, report)`` where the report
    lists each replacement with its match score + inpaint backend, so
    the owner sees exactly what was rewritten and how.
    """
    from ...media_edit.edittext import replace_text
    out = img
    applied: list[dict[str, Any]] = []
    for old, new in replacements:
        out, meta = replace_text(out, old, new, lang=lang)
        applied.append({"old": old, "new": new, **meta})
    return out, {"replacements": applied,
                 "n": len(applied)}


def finish_pass(img: Any, replacements: list[tuple[str, str]] | None = None,
                *, relight: dict[str, Any] | None = None,
                lang: str = "eng") -> tuple[Any, dict[str, Any]]:
    """Image finish: optional relight, then text-fidelity pass."""
    report: dict[str, Any] = {}
    out = img
    if relight:
        out = relight_after_composite(out, **relight)
        report["relight"] = dict(relight)
    if replacements:
        out, trep = text_fidelity_finish(out, replacements, lang=lang)
        report["text"] = trep
    return out, report
