"""Ken Burns engine — still images → cinematic camera moves.

Moves are data (the ``MOVES`` table), not code branches: each move is a
start/end pair of (center_x, center_y, zoom) in normalized image space,
animated with an easing curve. Multi-layer drift adds a parallax-ish
foreground layer moving against the background.

    from nomorals.media.motion_studio.kenburns import kenburns
    kenburns("cover.png", "out.mp4", move="zoom_in", duration=6.0)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ._core import (
    MotionStudioError,
    ease as _ease,
    profile_defaults,
    record_ledger,
    render_sequence,
    new_render_path,
)
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MOVES", "list_moves", "KenBurnsSpec", "kenburns", "slideshow_clip",
    "multilayer_drift", "add_grain", "add_vignette",
]


@dataclass(frozen=True)
class _Move:
    """Camera move as data: start/end (cx, cy, zoom)."""
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    label: str


#: every move the engine knows — pick by name, add new ones as data
MOVES: dict[str, _Move] = {
    "zoom_in":      _Move((0.50, 0.50, 1.00), (0.50, 0.50, 1.28), "slow push in"),
    "zoom_out":     _Move((0.50, 0.50, 1.30), (0.50, 0.50, 1.02), "slow pull out"),
    "pan_left":     _Move((0.62, 0.50, 1.22), (0.38, 0.50, 1.22), "pan left"),
    "pan_right":    _Move((0.38, 0.50, 1.22), (0.62, 0.50, 1.22), "pan right"),
    "pan_up":       _Move((0.50, 0.62, 1.22), (0.50, 0.38, 1.22), "tilt up"),
    "pan_down":     _Move((0.50, 0.38, 1.22), (0.50, 0.62, 1.22), "tilt down"),
    "drift":        _Move((0.46, 0.52, 1.14), (0.54, 0.48, 1.24), "organic drift"),
    "orbit":        _Move((0.42, 0.50, 1.20), (0.58, 0.50, 1.20), "wide orbit"),
    "push_tilt":    _Move((0.50, 0.60, 1.05), (0.50, 0.42, 1.30), "push + tilt"),
    "settle":       _Move((0.52, 0.48, 1.26), (0.50, 0.50, 1.18), "settle in"),
}

#: deterministic "auto" rotation for slideshows
_AUTO_ORDER = ("zoom_in", "drift", "pan_right", "push_tilt", "pan_left",
               "zoom_out", "pan_up", "settle", "pan_down", "orbit")


def list_moves() -> list[dict[str, str]]:
    return [{"name": n, "label": m.label} for n, m in MOVES.items()]


@dataclass
class KenBurnsSpec:
    image: str | os.PathLike
    out: str | os.PathLike | None = None
    duration: float = 6.0
    move: str = "zoom_in"          # or "auto" / "random"
    ease: str = "smooth"          # linear moves look robotic — don't default to it
    size: tuple[int, int] | None = None   # None → profile default (portrait)
    fps: float | None = None
    grain: bool = True
    vignette: bool = True
    letterbox: bool = False
    seed: int = 7


def _pick_move(name: str, seed: int, index: int = 0) -> _Move:
    name = (name or "zoom_in").lower()
    if name == "auto":
        return MOVES[_AUTO_ORDER[(seed + index) % len(_AUTO_ORDER)]]
    if name == "random":
        rng = np.random.default_rng(seed)
        return MOVES[rng.choice(list(MOVES))]
    if name not in MOVES:
        raise MotionStudioError(
            f"unknown move {name!r} — pick from: {', '.join(sorted(MOVES))}")
    return MOVES[name]


def _load_cover(image: str | os.PathLike, size: tuple[int, int],
                supersample: int) -> Image.Image:
    p = Path(image)
    if not p.exists():
        raise MotionStudioError(f"no such image: {image}")
    try:
        img = Image.open(p).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise MotionStudioError(f"could not open image {image}: {exc}") from exc
    # cover-resize at supersample resolution so the zoom stays sharp
    tw, th = size[0] * supersample, size[1] * supersample
    scale = max(tw / img.width, th / img.height)
    img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)),
                     Image.LANCZOS)
    return img


def _crop_view(img: Image.Image, cx: float, cy: float, zoom: float,
               size: tuple[int, int], supersample: int) -> Image.Image:
    """Crop the zoomed window around (cx, cy) and resize to output size."""
    tw, th = size[0] * supersample, size[1] * supersample
    vw, vh = tw / zoom, th / zoom
    x = (cx * img.width - vw / 2.0)
    y = (cy * img.height - vh / 2.0)
    x = min(max(x, 0.0), max(img.width - vw, 0.0))
    y = min(max(y, 0.0), max(img.height - vh, 0.0))
    box = (int(x), int(y), int(x + vw), int(y + vh))
    view = img.crop(box).resize((tw, th), Image.LANCZOS)
    if supersample > 1:
        view = view.resize(size, Image.LANCZOS)
    return view


def add_grain(frame: Image.Image, amount: float = 6.0,
              rng: np.random.Generator | None = None) -> Image.Image:
    """Animated film grain — the single cheapest "cinematic" upgrade."""
    rng = rng or np.random.default_rng()
    arr = np.asarray(frame).astype(np.float32)
    noise = rng.normal(0.0, amount, arr.shape[:2])[..., None]
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def add_vignette(frame: Image.Image, strength: float = 0.42) -> Image.Image:
    w, h = frame.size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs / w - 0.5) * 2.0
    dy = (ys / h - 0.5) * 2.0
    mask = np.clip(1.0 - (dx ** 2 + dy ** 2) * strength, 0.0, 1.0)
    arr = np.asarray(frame).astype(np.float32) * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _letterbox(frame: Image.Image, ratio: float = 0.09) -> Image.Image:
    w, h = frame.size
    bar = int(h * ratio)
    d = ImageDraw.Draw(frame)
    d.rectangle([0, 0, w, bar], fill="black")
    d.rectangle([0, h - bar, w, h], fill="black")
    return frame


def kenburns(image: str | os.PathLike, out: str | os.PathLike | None = None,
             *, duration: float = 6.0, move: str = "zoom_in",
             ease: str = "smooth", size: tuple[int, int] | None = None,
             fps: float | None = None, grain: bool = True,
             vignette: bool = True, letterbox: bool = False,
             seed: int = 7, crf: int | None = None) -> str:
    """Animate a still image with an eased camera move. Returns output path."""
    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    spec_move = _pick_move(move, seed)
    n = max(1, int(round(duration * fps)))
    img = _load_cover(image, size, int(defaults["supersample"]))
    rng = np.random.default_rng(seed)
    s0, e0 = spec_move.start, spec_move.end

    def _frame(i: int, t: float) -> np.ndarray:
        k = _ease(ease, i / max(n - 1, 1))
        cx = s0[0] + (e0[0] - s0[0]) * k
        cy = s0[1] + (e0[1] - s0[1]) * k
        z = s0[2] + (e0[2] - s0[2]) * k
        frame = _crop_view(img, cx, cy, z, size, int(defaults["supersample"]))
        if grain:
            frame = add_grain(frame, rng=rng)
        if vignette:
            frame = add_vignette(frame)
        if letterbox:
            frame = _letterbox(frame)
        return np.asarray(frame)

    out_path = Path(out) if out else new_render_path("kenburns")
    render_sequence(n, size[0], size[1], fps, out_path, _frame,
                    crf=crf or int(defaults["crf"]),
                    preset=str(defaults["preset"]))
    record_ledger({"kind": "kenburns", "path": str(out_path),
                   "move": move, "duration": duration})
    return str(out_path)


def slideshow_clip(images: Sequence[str | os.PathLike],
                   out: str | os.PathLike | None = None, *,
                   per_image: float = 4.0, move: str = "auto",
                   **kw) -> list[str]:
    """Render one Ken Burns clip per image (for :mod:`montage` assembly)."""
    clips = []
    for i, img in enumerate(images):
        clips.append(kenburns(img, duration=per_image, move=move,
                              seed=7 + i, **kw))
    return clips


def multilayer_drift(image: str | os.PathLike,
                     out: str | os.PathLike | None = None, *,
                     duration: float = 6.0, size: tuple[int, int] | None = None,
                     fps: float | None = None, depth: float = 0.35,
                     seed: int = 7) -> str:
    """Two-layer parallax drift: blurred foreground copy moves against the bg.

    ``depth`` 0..1 controls how far the foreground drifts opposite the bg.
    The foreground mask comes from the shared depth estimator
    (:func:`nomorals.media.directed.animator.estimate_depth`) — near
    regions drift as foreground — instead of a hand-rolled ellipse.
    """
    from ..directed.animator import estimate_depth
    defaults = profile_defaults()
    size = size or defaults["size"]
    fps = fps or float(defaults["fps"])
    n = max(1, int(round(duration * fps)))
    img = _load_cover(image, size, int(defaults["supersample"]))
    rng = np.random.default_rng(seed)
    fg = img.filter(ImageFilter.GaussianBlur(6)).copy()
    fg_arr_base = np.asarray(fg).astype(np.float32)
    depth = max(0.0, min(1.0, depth))
    # foreground mask from the shared pseudo-depth estimator (computed
    # once — the still doesn't change)
    _dm = estimate_depth(img)
    _dm_img = Image.fromarray((_dm * 255).astype(np.uint8)).resize(
        size, Image.BILINEAR)
    dm = np.asarray(_dm_img).astype(np.float32) / 255.0

    def _frame(i: int, t: float) -> np.ndarray:
        k = _ease("smooth", i / max(n - 1, 1))
        bg = np.asarray(_crop_view(img, 0.5, 0.5, 1.05 + 0.18 * k,
                                   size, int(defaults["supersample"])))
        shift = int(size[0] * 0.06 * depth * (k - 0.5) * 2)
        fg_arr = np.roll(fg_arr_base, shift, axis=1)
        h, w = bg.shape[:2]
        fh, fw = fg_arr.shape[:2]
        y0, x0 = (fh - h) // 2, (fw - w) // 2
        fg_crop = fg_arr[y0:y0 + h, x0:x0 + w]
        # near regions (high depth) show the drifting foreground
        mask = np.clip((dm - 0.45) * 2.2, 0, 1)[..., None]
        comp = bg.astype(np.float32) * (1 - mask) + fg_crop * mask
        frame = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))
        frame = add_grain(frame, amount=5.0, rng=rng)
        return np.asarray(add_vignette(frame))

    out_path = Path(out) if out else new_render_path("drift")
    render_sequence(n, size[0], size[1], fps, out_path, _frame,
                    crf=int(defaults["crf"]), preset=str(defaults["preset"]))
    record_ledger({"kind": "multilayer_drift", "path": str(out_path),
                   "duration": duration})
    return str(out_path)
