"""EditStudio: professional non-destructive media editing sessions.

Built ON TOP of the existing engines — :mod:`.images` (Pillow ops),
:mod:`.videos` (ffmpeg), :mod:`.jobs` (background jobs). Nothing here
re-implements them; the studio adds the session layer top-tier editors have:

- a **non-destructive op stack**: every edit appends a serializable op,
  undo/redo walks the stack, projects save/load as JSON, and ``render()``
  replays the stack from scratch (deterministic);
- **image**: layered compositing (image/text/shape layers with
  opacity + blend modes), pro color grading (lift/gamma/gain,
  temperature/tint, shadows/mids/highlights, vignette, grain), named
  filter presets, professional text (font discovery, stroke, shadow,
  letter-spacing, text boxes), collage templates, smart crop
  (edge-saliency, rule-of-thirds, optional face-detector hook),
  before/after compare exports;
- **video**: multi-clip assembly with xfade/acrossfade transitions,
  title cards, lower thirds, per-clip and timeline speed changes, audio
  ducking (sidechain), chapter markers, and export presets
  (social-vertical 9:16, youtube-4k, web-optimized, gif-preview);
- **templates**: one-call project builders (podcast-clip, quote-card,
  product-showcase, meme, slideshow);
- **batch**: run a saved project across a folder.

Image studio ops are registered into :mod:`.images`' op allowlist, so they
also work through ``nm media edit`` / the ``media_edit`` tool once this
module is imported.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .images import (
    MediaEditError,
    _require_pillow,
    _smart_crop_box,
    _parse_aspect,
    _unique_output,
    apply_chain,
    load_image,
    op_crop,
    op_grid,
    op_meme,
    op_resize,
    op_stack,
    save_image,
    validate_ops,
)
from . import images as _images
from . import generate as _generate  # noqa: F401  (registers generative_edit op)

_log = get_logger(__name__)

PROJECT_VERSION = 1


# ---------------------------------------------------------------------------
# font discovery
# ---------------------------------------------------------------------------

_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts",
    "/Library/Fonts",
    "C:\\Windows\\Fonts",
]

_font_cache: dict[str, str] | None = None


def discover_fonts(*, refresh: bool = False) -> dict[str, str]:
    """Scan system font dirs → {display name: path}. Cached after first scan."""
    global _font_cache
    if _font_cache is not None and not refresh:
        return _font_cache
    found: dict[str, str] = {}
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    name = os.path.splitext(f)[0].replace("-", " ").replace("_", " ")
                    found.setdefault(name.lower(), os.path.join(root, f))
    _font_cache = found
    return found


def find_font(query: str | None, size: int) -> Any:
    """Resolve a font by name (fuzzy), path, or fall back to DejaVu/default."""
    from PIL import ImageFont
    if query:
        q = query.strip()
        if os.path.isfile(q):
            try:
                return ImageFont.truetype(q, size)
            except OSError:  # noqa: E103 - falls through to font discovery
                pass
        fonts = discover_fonts()
        ql = q.lower()
        # exact → startswith → contains
        for key in (ql,):
            if key in fonts:
                try:
                    return ImageFont.truetype(fonts[key], size)
                except OSError:
                    break
        for name, path in fonts.items():
            if name.startswith(ql):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
        for name, path in fonts.items():
            if ql in name:
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
    # bundled fallbacks (same as images._load_font)
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# color grading engine (pure Pillow)
# ---------------------------------------------------------------------------

def _grade_lut(lift: tuple[float, float, float],
               gamma: tuple[float, float, float],
               gain: tuple[float, float, float]) -> list[int]:
    """Per-channel LUT for out = ((in * gain + lift) ^ (1/gamma))."""
    lut: list[int] = []
    for c in range(3):
        li, ga, gn = lift[c], gamma[c], gain[c]
        ga = ga if ga > 0 else 1.0
        channel = []
        for i in range(256):
            base = (i / 255.0) * gn + li
            v = base ** (1.0 / ga) if base > 0 else 0.0
            channel.append(max(0, min(255, int(round(v * 255)))))
        lut.extend(channel)
    return lut


def _temp_tint_gains(temperature: float,
                     tint: float) -> tuple[float, float, float]:
    """Fold temperature/tint into per-channel gains.

    temperature > 0 warms (red up, blue down); < 0 cools.
    tint > 0 pushes green; < 0 pushes magenta.
    """
    t = max(-1.0, min(1.0, temperature))
    ti = max(-1.0, min(1.0, tint))
    if t >= 0:
        r, g, b = 1.0 + 0.45 * t, 1.0 + 0.06 * t, 1.0 - 0.40 * t
    else:
        r, g, b = 1.0 + 0.30 * t, 1.0 + 0.05 * t, 1.0 - 0.50 * t
    g *= 1.0 + 0.25 * ti
    return (r, g, b)


def _zone_mask(lut_kind: str) -> list[int]:
    """Soft masks (0..255 LUT) for shadows / midtones / highlights."""
    lut = []
    for i in range(256):
        if lut_kind == "shadows":
            # 255 at black → 0 at 128 (cosine falloff)
            x = min(i, 128) / 128.0
            v = 0.5 + 0.5 * math.cos(math.pi * x)
        elif lut_kind == "highlights":
            # 0 at 128 → 255 at white
            x = max(0, i - 128) / 127.0
            v = 0.5 - 0.5 * math.cos(math.pi * min(1.0, x))
        else:  # midtones: peak at 128
            v = math.sin(math.pi * i / 255.0)
        lut.append(max(0, min(255, int(round(v * 255)))))
    return lut


def _apply_zone_tint(img: Any, color: tuple[int, int, int],
                     amount: float, kind: str) -> Any:
    Image = _require_pillow()
    amount = max(0.0, min(1.0, amount))
    if amount <= 0:
        return img
    rgb = img.convert("RGB")
    lum = rgb.convert("L")
    mask = lum.point(_zone_mask(kind))
    tint = Image.new("RGB", rgb.size, tuple(int(c) for c in color))
    blended = Image.blend(rgb, tint, amount)
    return Image.composite(blended, rgb, mask)


def _vignette_mask(size: tuple[int, int], amount: float) -> Any:
    Image = _require_pillow()
    w, h = size
    sw, sh = 160, max(1, round(160 * h / max(1, w)))
    cx, cy = sw / 2.0, sh / 2.0
    maxd = math.hypot(cx, cy) or 1.0
    px = []
    for y in range(sh):
        for x in range(sw):
            d = math.hypot(x - cx, y - cy) / maxd
            fall = max(0.0, min(1.0, (d - 0.5) / 0.5))
            px.append(int(255 * amount * fall * fall))
    m = Image.new("L", (sw, sh))
    m.putdata(px)
    return m.resize(size, Image.BILINEAR)


def _apply_vignette(img: Any, amount: float) -> Any:
    Image = _require_pillow()
    amount = max(0.0, min(1.0, amount))
    if amount <= 0:
        return img
    rgb = img.convert("RGB")
    mask = _vignette_mask(rgb.size, amount)
    black = Image.new("RGB", rgb.size, (0, 0, 0))
    return Image.composite(black, rgb, mask)


def _apply_grain(img: Any, amount: float, seed: int = 7) -> Any:
    """Deterministic film grain (seeded); amount ~ 0..30."""
    Image = _require_pillow()
    amount = max(0.0, min(30.0, amount))
    if amount <= 0:
        return img
    rgb = img.convert("RGB")
    w, h = rgb.size
    # generate small seeded noise, upscale (fast + deterministic)
    sw, sh = min(320, w), max(1, round(min(320, w) * h / max(1, w)))
    rng = random.Random(seed)
    sigma = 18.0 + amount * 2.5
    px = [max(0, min(255, int(rng.gauss(128, sigma)))) for _ in range(sw * sh)]
    noise = Image.new("L", (sw, sh))
    noise.putdata(px)
    noise = noise.resize((w, h), Image.BILINEAR).convert("RGB")
    alpha = min(0.35, amount / 60.0)
    return Image.blend(rgb, noise, alpha)


def op_grade(img: Any, *,
             temperature: float = 0.0,
             tint: float = 0.0,
             lift: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
             gamma: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
             gain: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
             shadows: tuple | list | None = None,
             midtones: tuple | list | None = None,
             highlights: tuple | list | None = None,
             saturation: float = 1.0,
             contrast: float = 1.0,
             vibrance: float = 0.0,
             grayscale: bool = False,
             vignette: float = 0.0,
             grain: float = 0.0,
             grain_seed: int = 7) -> Any:
    """Professional color grade. Zone tints are ((r,g,b), amount) pairs."""
    from PIL import ImageEnhance, ImageOps
    out = img
    if grayscale:
        out = ImageOps.grayscale(out).convert("RGB")
    tg = _temp_tint_gains(temperature, tint)
    lift_t = tuple(lift)
    gamma_t = tuple(gamma)
    gain_t = tuple(g * tg[c] for c, g in enumerate(gain))
    out = out.convert("RGB").point(_grade_lut(lift_t, gamma_t, gain_t))
    for zone, kind in ((shadows, "shadows"), (midtones, "midtones"),
                       (highlights, "highlights")):
        if zone:
            color, amount = zone
            out = _apply_zone_tint(out, tuple(int(c) for c in color),
                                   float(amount), kind)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    sat_eff = saturation * (1.0 + 0.5 * vibrance)
    if sat_eff != 1.0:
        out = ImageEnhance.Color(out).enhance(sat_eff)
    out = _apply_vignette(out, vignette)
    out = _apply_grain(out, grain, seed=grain_seed)
    return out


# ---------------------------------------------------------------------------
# filter presets
# ---------------------------------------------------------------------------

FILTER_PRESETS: dict[str, dict[str, Any]] = {
    "portrait": {
        "temperature": 0.08, "tint": 0.02, "saturation": 1.06,
        "contrast": 1.05, "lift": (0.02, 0.015, 0.01), "vignette": 0.15,
    },
    "cinematic": {
        "shadows": ((26, 128, 153), 0.45),
        "highlights": ((255, 153, 77), 0.30),
        "lift": (-0.03, -0.03, -0.03), "saturation": 0.90,
        "contrast": 1.10, "vignette": 0.35,
    },
    "vintage": {
        "temperature": 0.15, "lift": (0.08, 0.06, 0.04),
        "contrast": 0.85, "saturation": 0.75, "grain": 8.0, "vignette": 0.25,
    },
    "bw-drama": {
        "grayscale": True, "contrast": 1.35, "vignette": 0.40, "grain": 6.0,
    },
    "vibrant": {
        "saturation": 1.35, "contrast": 1.12, "vibrance": 0.20,
    },
    "teal-orange": {
        "shadows": ((38, 140, 153), 0.55),
        "highlights": ((255, 140, 64), 0.40), "saturation": 1.05,
    },
    "noir": {
        "grayscale": True, "lift": (-0.06, -0.06, -0.06),
        "contrast": 1.40, "vignette": 0.50,
    },
    "golden-hour": {
        "temperature": 0.22, "highlights": ((255, 191, 102), 0.40),
        "saturation": 1.10, "vignette": 0.20,
    },
    "cool-matte": {
        "temperature": -0.15, "lift": (0.03, 0.04, 0.06),
        "contrast": 0.90, "saturation": 0.90,
    },
    "warm-fade": {
        "temperature": 0.12, "lift": (0.06, 0.05, 0.04),
        "contrast": 0.88, "saturation": 0.95,
    },
}

_MULT_KEYS = {"saturation", "contrast"}
_OFF_KEYS = {"temperature", "tint", "vignette", "grain"}


def _scale_preset(params: dict[str, Any], strength: float) -> dict[str, Any]:
    """Scale a preset toward neutral by strength 0..1."""
    strength = max(0.0, min(1.0, strength))
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in _MULT_KEYS:
            out[k] = 1.0 + (v - 1.0) * strength
        elif k in _OFF_KEYS:
            out[k] = v * strength
        elif k == "lift":
            out[k] = tuple(x * strength for x in v)
        elif k in ("shadows", "midtones", "highlights"):
            color, amt = v
            out[k] = (color, amt * strength)
        elif k == "vibrance":
            out[k] = v * strength
        else:  # grayscale and friends: keep as-is
            out[k] = v
    return out


def op_filter(img: Any, preset: str, *, strength: float = 1.0) -> Any:
    """Apply a named filter preset (see FILTER_PRESETS)."""
    name = str(preset).strip().lower().replace(" ", "-").replace("_", "-")
    if name not in FILTER_PRESETS:
        raise MediaEditError(
            f"unknown filter preset {preset!r}; choose from "
            f"{sorted(FILTER_PRESETS)}")
    return op_grade(img, **_scale_preset(FILTER_PRESETS[name], strength))


def op_letterbox(img: Any, aspect: str = "21:9",
                 color: str = "black") -> Any:
    """Fit the image into ``aspect``: center-crop if too tall, pad with
    ``color`` bars if too wide (the cinematic-bars effect)."""
    Image = _require_pillow()
    target = _parse_aspect(aspect)
    have = img.width / img.height
    if abs(have - target) < 1e-6:
        return img.copy()
    if have < target:
        # too tall -> crop height (bars would be top/bottom)
        new_h = max(1, round(img.width / target))
        top = (img.height - new_h) // 2
        return img.crop((0, top, img.width, top + new_h))
    # too wide -> pad sides
    new_w = max(1, round(img.height * target))
    canvas = Image.new(img.mode if img.mode in ("RGB", "RGBA") else "RGB",
                       (new_w, img.height), color)
    base = img.convert(canvas.mode)
    canvas.paste(base, ((new_w - img.width) // 2, 0))
    return canvas


# ---------------------------------------------------------------------------
# smart crop: saliency / rule-of-thirds / face-aware placeholder
# ---------------------------------------------------------------------------

def _energy_map(img: Any) -> Any:
    """64x64 edge-energy map (same saliency signal as images._smart_crop_box)."""
    from PIL import ImageFilter
    small = img.convert("L").resize((64, 64))
    return small.filter(ImageFilter.FIND_EDGES)


def _thirds_points() -> list[tuple[float, float]]:
    return [(x / 3.0, y / 3.0) for x in (1, 2) for y in (1, 2)]


def _smart_crop_thirds(img: Any, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Pick the window maximizing edge energy, biasing the energy centroid
    toward rule-of-thirds intersections (pro composition default)."""
    energy = _energy_map(img)
    px = energy.load()
    sx, sy = img.width / 64.0, img.height / 64.0
    win_w = max(1, round(target_w / sx))
    win_h = max(1, round(target_h / sy))
    thirds = _thirds_points()
    best_score, best = float("-inf"), (0, 0)
    step = 4
    for y in range(0, 64 - win_h + 1, step):
        for x in range(0, 64 - win_w + 1, step):
            e = 0.0
            cx = cy = 0.0
            n = 0
            for yy in range(y, min(y + win_h, 64), 2):
                for xx in range(x, min(x + win_w, 64), 2):
                    v = px[xx, yy]
                    e += v
                    cx += xx * v
                    cy += yy * v
                    n += 1
            if e <= 0:
                continue
            # energy centroid of this window, in 0..1
            ccx, ccy = (cx / e) / 64.0, (cy / e) / 64.0
            d = min(math.hypot(ccx - tx, ccy - ty) for tx, ty in thirds)
            # energy dominates; thirds proximity breaks ties (~15% weight)
            score = e * (1.0 - 0.15 * min(1.0, d * 4.0))
            if score > best_score:
                best_score, best = score, (x, y)
    bx, by = best
    left = min(img.width - target_w, round(bx * sx))
    top = min(img.height - target_h, round(by * sy))
    return max(0, left), max(0, top), max(0, left) + target_w, max(0, top) + target_h


def op_smart_crop(img: Any, aspect: str, *,
                  mode: str = "saliency",
                  face_detector: Callable[[Any], list[tuple[int, int, int, int]]] | None = None) -> Any:
    """Crop to ``aspect`` with a composition-aware anchor.

    modes: center | saliency (edge-energy) | thirds (energy + rule-of-thirds)
    | faces (uses ``face_detector`` when given — a callable taking the image
    and returning [(l,t,r,b)] — otherwise falls back to saliency; no ML
    detector is bundled, so edge density is the documented placeholder).
    """
    target_ratio = _parse_aspect(aspect)
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:
        target_h, target_w = img.height, max(1, round(img.height * target_ratio))
    else:
        target_w, target_h = img.width, max(1, round(img.width / target_ratio))
    if mode == "center":
        return op_crop(img, aspect=aspect, anchor="center")
    if mode == "saliency":
        return img.crop(_smart_crop_box(img, target_w, target_h))
    if mode == "thirds":
        return img.crop(_smart_crop_thirds(img, target_w, target_h))
    if mode == "faces":
        boxes = None
        if face_detector is not None:
            try:
                boxes = face_detector(img)
            except Exception as exc:  # noqa: BLE001 - detector is optional
                _log.warning("face_detector failed, using saliency: %s", exc)
        if boxes:
            # cover the union of faces if it fits, else the largest face
            l = min(b[0] for b in boxes)
            t = min(b[1] for b in boxes)
            r = max(b[2] for b in boxes)
            b = max(b[3] for b in boxes)
            fw, fh = r - l, b - t
            if fw <= target_w and fh <= target_h:
                cx, cy = (l + r) // 2, (t + b) // 2
            else:  # largest face
                bl, bt, br, bb = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx, cy = (bl + br) // 2, (bt + bb) // 2
            left = min(img.width - target_w, max(0, cx - target_w // 2))
            top = min(img.height - target_h, max(0, cy - target_h // 2))
            return img.crop((left, top, left + target_w, top + target_h))
        _log.info("op_smart_crop faces: no detector/boxes, using saliency")
        return img.crop(_smart_crop_box(img, target_w, target_h))
    raise MediaEditError(f"unknown smart-crop mode {mode!r}; "
                         "use center/saliency/thirds/faces")


# ---------------------------------------------------------------------------
# professional text overlays
# ---------------------------------------------------------------------------

def _text_line_width(draw: Any, line: str, font: Any,
                     letter_spacing: int) -> int:
    if not letter_spacing:
        return int(draw.textlength(line, font=font))
    return int(sum(draw.textlength(ch, font=font) for ch in line)
               + letter_spacing * max(0, len(line) - 1))


def _wrap_text(draw: Any, text: str, font: Any, letter_spacing: int,
               max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _text_line_width(draw, trial, font, letter_spacing) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def render_text_layer(text: str, *,
                      font: str | None = None,
                      size: int | None = None,
                      color: str = "white",
                      stroke_width: int = 0,
                      stroke_fill: str = "black",
                      shadow: bool | dict[str, Any] = False,
                      letter_spacing: int = 0,
                      line_spacing: int = 6,
                      align: str = "center",
                      box_width: int | None = None,
                      box_bg: str | None = None,
                      box_padding: int = 16,
                      opacity: float = 1.0,
                      rotation: float = 0.0,
                      max_width_frac: float = 0.92) -> Any:
    """Render pro text to a tight RGBA layer.

    ``size=None`` auto-fits: starts large, shrinks until the longest line
    fits ``max_width_frac`` of ``box_width`` (or 92% of a 1920 default).
    ``shadow=True`` → soft offset shadow; pass a dict
    {dx, dy, blur, color, opacity} to tune. ``box_bg`` draws a rounded
    backdrop box. Returns an RGBA image (tight to the text + padding).
    """
    Image = _require_pillow()
    from PIL import ImageDraw, ImageFilter
    if not text:
        raise MediaEditError("text_layer needs non-empty text")
    if align not in ("left", "center", "right"):
        raise MediaEditError(f"unknown text align {align!r}")
    ref_w = box_width or 1920
    # auto-fit size
    if size is None:
        size = max(12, ref_w // 12)
        while size > 12:
            font_try = find_font(font, size)
            probe = Image.new("RGBA", (8, 8))
            d = ImageDraw.Draw(probe)
            lines = _wrap_text(d, text, font_try, letter_spacing,
                               int(ref_w * max_width_frac))
            widest = max(_text_line_width(d, ln, font_try, letter_spacing)
                         for ln in lines)
            if widest <= ref_w * max_width_frac:
                break
            size = max(12, size - 4)
    font_obj = find_font(font, size)
    measure = Image.new("RGBA", (8, 8))
    dm = ImageDraw.Draw(measure)
    lines = _wrap_text(dm, text, font_obj, letter_spacing,
                       int((box_width or ref_w * max_width_frac)))
    widths = [_text_line_width(dm, ln, font_obj, letter_spacing) for ln in lines]
    # line height from a representative bbox
    bbox = dm.textbbox((0, 0), "Ag", font=font_obj, stroke_width=stroke_width)
    line_h = (bbox[3] - bbox[1]) + line_spacing
    text_w = max(widths) if widths else 0
    text_h = line_h * len(lines)
    pad = box_padding if box_bg else 8
    cw, ch = text_w + pad * 2, text_h + pad * 2
    # shadow config
    sh: dict[str, Any] = {}
    if shadow:
        sh = {"dx": 3, "dy": 4, "blur": 6, "color": "black", "opacity": 0.6}
        if isinstance(shadow, dict):
            sh.update(shadow)
        cw += abs(int(sh.get("dx", 0))) + int(sh.get("blur", 0)) * 2
        ch += abs(int(sh.get("dy", 0))) + int(sh.get("blur", 0)) * 2
    layer = Image.new("RGBA", (max(1, cw), max(1, ch)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if box_bg:
        draw.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=min(24, pad),
                              fill=box_bg)

    def _draw_text(target: Any, fill: Any) -> None:
        y = pad
        for ln, lw in zip(lines, widths):
            if align == "center":
                x = (cw - lw) // 2
            elif align == "right":
                x = cw - pad - lw
            else:
                x = pad
            if letter_spacing:
                cx = x
                for ch_ in ln:
                    target.text((cx, y), ch_, font=font_obj, fill=fill,
                                stroke_width=stroke_width,
                                stroke_fill=stroke_fill)
                    cx += target.textlength(ch_, font=font_obj) + letter_spacing
            else:
                target.text((x, y), ln, font=font_obj, fill=fill,
                            stroke_width=stroke_width,
                            stroke_fill=stroke_fill)
            y += line_h

    if sh:
        glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        _draw_text(gd, sh["color"])
        glow = glow.filter(ImageFilter.GaussianBlur(sh["blur"]))
        # apply shadow opacity + offset
        alpha = glow.split()[3].point(
            lambda a: int(a * float(sh["opacity"])))
        glow.putalpha(alpha)
        shadow_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shadow_layer.alpha_composite(
            glow, (int(sh["dx"]), int(sh["dy"])))
        layer = Image.alpha_composite(shadow_layer, layer)
        draw = ImageDraw.Draw(layer)
    _draw_text(draw, color)
    if opacity < 1.0:
        alpha = layer.split()[3].point(lambda a: int(a * opacity))
        layer.putalpha(alpha)
    if rotation:
        layer = layer.rotate(rotation, expand=True,
                             resample=Image.BICUBIC)
    return layer


def _resolve_anchor(canvas: tuple[int, int], layer: tuple[int, int],
                    position: Any, margin: int) -> tuple[int, int]:
    """Anchor name or (x, y) → top-left paste coords."""
    cw, ch = canvas
    lw, lh = layer
    if isinstance(position, (list, tuple)):
        return int(position[0]), int(position[1])
    table = {
        "center": ((cw - lw) // 2, (ch - lh) // 2),
        "top": ((cw - lw) // 2, margin),
        "bottom": ((cw - lw) // 2, ch - lh - margin),
        "left": (margin, (ch - lh) // 2),
        "right": (cw - lw - margin, (ch - lh) // 2),
        "top-left": (margin, margin),
        "top-right": (cw - lw - margin, margin),
        "bottom-left": (margin, ch - lh - margin),
        "bottom-right": (cw - lw - margin, ch - lh - margin),
    }
    if position not in table:
        raise MediaEditError(
            f"unknown position {position!r}; use an anchor "
            f"{sorted(table)} or (x, y)")
    return table[position]


def op_text_layer(img: Any, text: str, *,
                  position: Any = "bottom",
                  margin: int = 24,
                  **text_kwargs: Any) -> Any:
    """Composite a professional text layer onto the image."""
    Image = _require_pillow()
    layer = render_text_layer(text, **text_kwargs)
    x, y = _resolve_anchor((img.width, img.height), layer.size,
                           position, margin)
    out = img.convert("RGBA")
    # clip layers that hang off the canvas edge
    if x < 0 or y < 0 or x + layer.width > out.width or y + layer.height > out.height:
        lx0, ly0 = max(0, -x), max(0, -y)
        lx1, ly1 = min(layer.width, out.width - x), min(layer.height, out.height - y)
        if lx1 <= lx0 or ly1 <= ly0:
            # fully off-canvas: an honest no-op, not a silent one
            _log.warning("text_layer %r is fully off-canvas; "
                         "no text was drawn", text[:40])
            return img
        layer = layer.crop((lx0, ly0, lx1, ly1))
        x, y = max(0, x), max(0, y)
    out.alpha_composite(layer, (x, y))
    return out.convert("RGB") if img.mode == "RGB" else out


# ---------------------------------------------------------------------------
# layered compositing
# ---------------------------------------------------------------------------

BLEND_MODES = ("normal", "multiply", "screen", "overlay",
               "darken", "lighten", "difference")


def _blend_fn(mode: str) -> Callable[[Any, Any], Any]:
    from PIL import ImageChops
    table = {
        "multiply": ImageChops.multiply,
        "screen": ImageChops.screen,
        "overlay": ImageChops.overlay,
        "darken": ImageChops.darker,
        "lighten": ImageChops.lighter,
        "difference": ImageChops.difference,
    }
    try:
        return table[mode]
    except KeyError:
        raise MediaEditError(f"unknown blend mode {mode!r}; "
                             f"use {list(BLEND_MODES)}") from None


def render_image_layer(canvas_size: tuple[int, int],
                       spec: dict[str, Any]) -> tuple[Any, tuple[int, int]]:
    """Render an image layer → (RGBA content, (x, y))."""
    Image = _require_pillow()
    path = spec.get("path")
    if not path:
        raise MediaEditError("image layer needs a 'path'")
    layer_img = load_image(path).convert("RGBA")
    cw, ch = canvas_size
    try:
        if spec.get("fit") == "fill":
            layer_img = op_resize(layer_img, cw, ch, mode="fill")
        elif spec.get("fit") == "fit":
            layer_img = op_resize(layer_img, cw, ch, mode="fit")
        elif "size" in spec:
            w, h = (int(v) for v in spec["size"])
            layer_img = op_resize(layer_img, w, h, mode="exact")
        elif "width" in spec:
            w = int(spec["width"])
            h = round(layer_img.height * w / layer_img.width)
            layer_img = op_resize(layer_img, w, h, mode="exact")
        else:
            scale = float(spec.get("scale", 1.0))
            if scale != 1.0:
                w = max(1, round(cw * scale))
                h = max(1, round(layer_img.height * w / layer_img.width))
                layer_img = op_resize(layer_img, w, h, mode="exact")
        rotation = float(spec.get("rotation", 0.0) or 0.0)
        if rotation:
            layer_img = layer_img.rotate(rotation, expand=True,
                                         resample=Image.BICUBIC)
        opacity = float(spec.get("opacity", 1.0))
        if not 0.0 <= opacity <= 1.0:
            raise MediaEditError("layer opacity must be 0..1")
        if opacity < 1.0:
            alpha = layer_img.split()[3].point(lambda a: int(a * opacity))
            layer_img.putalpha(alpha)
        x, y = _resolve_anchor(canvas_size, layer_img.size,
                               spec.get("position", "center"),
                               int(spec.get("margin", 20)))
        return layer_img, (x, y)
    except Exception:
        layer_img.close()
        raise


def render_shape_layer(canvas_size: tuple[int, int],
                       spec: dict[str, Any]) -> tuple[Any, tuple[int, int]]:
    """Render a shape layer → (RGBA content, (x, y))."""
    Image = _require_pillow()
    from PIL import ImageDraw
    cw, ch = canvas_size
    shape = spec.get("shape", "rect")
    box = spec.get("box")
    if box == "full":  # full-canvas (templates use this to dim a background)
        box = [0, 0, cw, ch]
    if box is None:
        # default: centered box covering the middle third
        w, h = cw // 3, ch // 3
        box = [(cw - w) // 2, (ch - h) // 2, (cw + w) // 2, (ch + h) // 2]
    l, t, r, b = (int(v) for v in box)
    w, h = max(1, r - l), max(1, b - t)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = spec.get("fill")
    outline = spec.get("outline")
    width = int(spec.get("width", 4))
    lb, tb = 0, 0
    rb, bb = w, h
    if shape in ("rect", "rectangle"):
        draw.rounded_rectangle([lb, tb, rb - 1, bb - 1],
                              radius=int(spec.get("radius", 0)),
                              fill=fill, outline=outline, width=width)
    elif shape in ("circle", "ellipse"):
        draw.ellipse([lb, tb, rb - 1, bb - 1], fill=fill,
                     outline=outline, width=width)
    elif shape == "line":
        draw.line([lb, (tb + bb) // 2, rb, (tb + bb) // 2],
                  fill=outline or fill or "white", width=width)
    elif shape == "arrow":
        y = (tb + bb) // 2
        head = max(width * 3, w // 6)
        col = outline or fill or "white"
        draw.line([lb, y, rb - head, y], fill=col, width=width)
        draw.polygon([(rb, y), (rb - head, y - head // 2),
                      (rb - head, y + head // 2)], fill=col)
    else:
        raise MediaEditError(f"unknown shape layer {shape!r}")
    opacity = float(spec.get("opacity", 1.0))
    if not 0.0 <= opacity <= 1.0:
        raise MediaEditError("layer opacity must be 0..1")
    if opacity < 1.0:
        alpha = layer.split()[3].point(lambda a: int(a * opacity))
        layer.putalpha(alpha)
    return layer, (l, t)


def composite_layers(base: Any, layers: list[dict[str, Any]]) -> Any:
    """Composite layer specs over ``base`` (pure). Layer dicts:

    {"type": "image", "path": ..., "position": "bottom-right"|(x,y),
     "scale": 0.2 | "size": [w,h] | "width": px | "fit": "fill"|"fit",
     "opacity": 0..1, "blend": "normal"|..., "rotation": deg, "margin": 20}
    {"type": "text", <render_text_layer kwargs>, "position": ..., "margin": 24}
    {"type": "shape", "shape": "rect"|"circle"|"line"|"arrow",
     "box": [l,t,r,b], "fill": color, "outline": color, "width": 4,
     "radius": 0, "opacity": 0..1}
    """
    canvas = base.convert("RGBA")
    for idx, spec in enumerate(layers):
        spec = dict(spec)
        ltype = spec.get("type", "image")
        blend = spec.get("blend", "normal")
        if blend not in BLEND_MODES:
            raise MediaEditError(f"unknown blend mode {blend!r}")
        if ltype == "image":
            content, (x, y) = render_image_layer((canvas.width, canvas.height),
                                                 spec)
        elif ltype == "text":
            text = spec.pop("text", "")
            position = spec.pop("position", "bottom")
            margin = int(spec.pop("margin", 24))
            spec.pop("type", None)
            spec.pop("blend", None)
            content = render_text_layer(text, **spec)
            x, y = _resolve_anchor((canvas.width, canvas.height),
                                   content.size, position, margin)
        elif ltype == "shape":
            content, (x, y) = render_shape_layer((canvas.width, canvas.height),
                                                 spec)
        else:
            raise MediaEditError(f"unknown layer type {ltype!r}")
        try:
            # clip to canvas
            lx0, ly0 = max(0, -x), max(0, -y)
            lx1 = min(content.width, canvas.width - x)
            ly1 = min(content.height, canvas.height - y)
            if lx1 <= lx0 or ly1 <= ly0:
                # fully off-canvas: warn so a "composite" that drew nothing
                # can never be mistaken for a successful overlay
                _log.warning("composite: layer %d (%s) is fully off-canvas; "
                             "skipped", idx, ltype)
                continue
            if lx0 or ly0 or lx1 != content.width or ly1 != content.height:
                content = content.crop((lx0, ly0, lx1, ly1))
                x, y = max(0, x), max(0, y)
            if blend == "normal":
                canvas.alpha_composite(content, (x, y))
            else:
                region = canvas.crop((x, y, x + content.width,
                                      y + content.height)).convert("RGB")
                blended = _blend_fn(blend)(region, content.convert("RGB"))
                alpha = content.split()[3]
                merged = blended.convert("RGBA")
                merged.putalpha(alpha)
                canvas.alpha_composite(merged, (x, y))
        finally:
            try:
                content.close()
            except Exception:  # noqa: BLE001
                pass
    out = canvas.convert("RGB") if base.mode == "RGB" else canvas
    return out


def op_composite(img: Any, layers: list[dict[str, Any]] | None = None,
                 **kwargs: Any) -> Any:
    """Chain-compatible layer composite. Also accepts single-layer kwargs
    (type/text/path/...) as a one-layer composite."""
    if layers is None:
        layers = [kwargs] if kwargs else []
    if not isinstance(layers, list) or not layers:
        raise MediaEditError("composite needs a non-empty 'layers' list")
    return composite_layers(img, layers)


def op_shape_layer(img: Any, **spec: Any) -> Any:
    """Chain-compatible single shape layer."""
    return composite_layers(img, [{"type": "shape", **spec}])


# ---------------------------------------------------------------------------
# collage templates
# ---------------------------------------------------------------------------

COLLAGE_TEMPLATES = ("grid", "strip", "diptych", "triptych")


def op_collage(img: Any, images: list[Any] | None = None, *,
               template: str = "grid", cols: int = 2, gap: int = 8,
               bg: str = "black", captions: list[str] | None = None,
               caption_bg: str = "black", caption_color: str = "white",
               caption_size: int = 28) -> Any:
    """Collage the base image with ``images`` (paths or Images).

    templates: grid (cols×rows), strip (horizontal), diptych, triptych.
    ``captions`` adds a labeled bar under each cell (same order as images,
    base image first).
    """
    if template not in COLLAGE_TEMPLATES:
        raise MediaEditError(f"unknown collage template {template!r}; "
                             f"use {list(COLLAGE_TEMPLATES)}")
    others = [load_image(p) if isinstance(p, (str, Path)) else p
              for p in (images or [])]
    cells = [img] + others
    try:
        if captions:
            if len(captions) != len(cells):
                raise MediaEditError(
                    f"collage needs {len(cells)} captions, got {len(captions)}")
            cells = [_caption_cell(c, cap, bg=caption_bg, color=caption_color,
                                   size=caption_size)
                     for c, cap in zip(cells, captions)]
        if template == "grid":
            return op_grid(cells, cols=cols, gap=gap, bg=bg)
        if template == "strip":
            return op_stack(cells, direction="horizontal", bg=bg, gap=gap)
        if template == "diptych":
            if len(cells) != 2:
                raise MediaEditError("diptych needs exactly 2 images "
                                     f"(base + 1), got {len(cells)}")
            return op_stack(cells, direction="horizontal", bg=bg, gap=gap)
        # triptych
        if len(cells) != 3:
            raise MediaEditError("triptych needs exactly 3 images "
                                 f"(base + 2), got {len(cells)}")
        return op_stack(cells, direction="horizontal", bg=bg, gap=gap)
    finally:
        for extra in others:
            try:
                extra.close()
            except Exception:  # noqa: BLE001
                pass


def _caption_cell(img: Any, caption: str, *, bg: str, color: str,
                  size: int) -> Any:
    Image = _require_pillow()
    bar_h = size + 24
    canvas = Image.new("RGB", (img.width, img.height + bar_h), bg)
    canvas.paste(img.convert("RGB"), (0, 0))
    label = render_text_layer(caption, size=size, color=color,
                              box_width=img.width, align="center",
                              box_padding=4)
    lw, lh = label.size
    x = (img.width - lw) // 2
    canvas.paste(label, (x, img.height + (bar_h - lh) // 2), label)
    return canvas


# ---------------------------------------------------------------------------
# before / after compare
# ---------------------------------------------------------------------------

def compare_render(before: Any, after: Any, *,
                   mode: str = "side-by-side", labels: bool = True) -> Any:
    """Render a before/after comparison image.

    modes: side-by-side (matched height), split (vertical divider at 50%),
    stacked (before over after).
    """
    Image = _require_pillow()
    if mode not in ("side-by-side", "split", "stacked"):
        raise MediaEditError(f"unknown compare mode {mode!r}")
    b = before.convert("RGB")
    a = after.convert("RGB")
    if mode == "side-by-side":
        out = op_stack([b, a], direction="horizontal", bg="black", gap=4)
    elif mode == "stacked":
        out = op_stack([b, a], direction="vertical", bg="black", gap=4)
    else:  # split
        a = op_resize(a, b.width, b.height, mode="exact")
        out = Image.new("RGB", b.size)
        out.paste(b.crop((0, 0, b.width // 2, b.height)), (0, 0))
        out.paste(a.crop((b.width // 2, 0, b.width, b.height)),
                  (b.width // 2, 0))
        from PIL import ImageDraw
        d = ImageDraw.Draw(out)
        d.line([(b.width // 2, 0), (b.width // 2, b.height)],
               fill="white", width=3)
    if labels:
        out = op_text_layer(out, "BEFORE", position="top-left", margin=12,
                            size=max(14, out.height // 40), color="white",
                            stroke_width=2)
        half = out.width // 2 if mode != "stacked" else 0
        out = op_text_layer(out, "AFTER",
                            position=(half + 12 if mode == "side-by-side"
                                      else 12, 12) if mode != "stacked"
                            else "bottom-left",
                            margin=12, size=max(14, out.height // 40),
                            color="white", stroke_width=2)
    return out


def compare_html(before_path: str | os.PathLike[str],
                 after_path: str | os.PathLike[str],
                 out_path: str | os.PathLike[str]) -> Path:
    """Self-contained before/after slider page (draggable divider)."""
    out = Path(out_path)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Before / After</title>
<style>
body {{ margin: 0; background: #111; display: flex; justify-content: center; }}
.wrap {{ position: relative; max-width: 100vw; user-select: none; }}
.wrap img {{ display: block; max-width: 100vw; max-height: 96vh; }}
.after {{ position: absolute; inset: 0; overflow: hidden; }}
.after img {{ width: 100%; height: 100%; object-fit: contain; }}
input[type=range] {{ position: absolute; inset: 0; width: 100%; opacity: 0; cursor: ew-resize; margin: 0; }}
.bar {{ position: absolute; top: 0; bottom: 0; width: 3px; background: #fff; box-shadow: 0 0 8px #000; pointer-events: none; }}
.tag {{ position: absolute; top: 10px; font: 12px sans-serif; color: #fff;
        background: rgba(0,0,0,.55); padding: 4px 10px; border-radius: 4px; pointer-events: none; }}
</style></head><body>
<div class="wrap" id="wrap">
  <img src="{Path(before_path).name}" alt="before">
  <div class="after" id="after"><img src="{Path(after_path).name}" alt="after"></div>
  <div class="bar" id="bar"></div>
  <div class="tag" style="left:10px">BEFORE</div>
  <div class="tag" style="right:10px">AFTER</div>
  <input type="range" id="slider" min="0" max="100" value="50">
</div>
<script>
const s = document.getElementById('slider');
const after = document.getElementById('after');
const bar = document.getElementById('bar');
function upd() {{
  after.style.clipPath = `inset(0 0 0 ${{s.value}}%)`;
  bar.style.left = s.value + '%';
}}
s.addEventListener('input', upd); upd();
</script></body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    # copy the two images next to the html so relative srcs resolve
    for src in (Path(before_path), Path(after_path)):
        dest = out.parent / src.name
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# register studio image ops into the images engine allowlist
# (additive — existing `nm media edit` keeps working, gains new ops)
# ---------------------------------------------------------------------------

_STUDIO_IMAGE_OPS = {
    "filter": op_filter,
    "grade": op_grade,
    "letterbox": op_letterbox,
    "smart_crop": op_smart_crop,
    "text_layer": op_text_layer,
    "composite": op_composite,
    "shape_layer": op_shape_layer,
    "collage": op_collage,
}

for _name, _fn in _STUDIO_IMAGE_OPS.items():
    _images._OP_FUNCS[_name] = _fn
    _images.OP_ALLOWLIST.add(_name)
del _name, _fn


# ---------------------------------------------------------------------------
# video: export presets + filter-graph compiler
# ---------------------------------------------------------------------------

EXPORT_PRESETS: dict[str, dict[str, Any]] = {
    "web-optimized": {
        "width": 1280, "height": 720, "fit": "fit",
        "vcodec": "libx264", "crf": 23, "preset": "medium",
        "acodec": "aac", "faststart": True, "ext": ".mp4",
    },
    "social-vertical": {
        "width": 1080, "height": 1920, "fit": "fill",
        "vcodec": "libx264", "crf": 21, "preset": "medium",
        "acodec": "aac", "faststart": True, "ext": ".mp4",
    },
    "youtube-4k": {
        "width": 3840, "height": 2160, "fit": "fit",
        "vcodec": "libx264", "crf": 18, "preset": "slow",
        "acodec": "aac", "faststart": True, "ext": ".mp4",
    },
    "source": {
        "vcodec": "libx264", "crf": 20, "preset": "medium",
        "acodec": "aac", "faststart": True, "ext": ".mp4",
    },
    "gif-preview": {
        "width": 480, "fps": 12, "ext": ".gif",
    },
}

TRANSITIONS = ("fade", "fadeblack", "fadewhite", "wipeleft", "wiperight",
               "wipeup", "wipedown", "slideleft", "slideright", "smoothleft",
               "smoothright", "circleopen", "circleclose", "dissolve")


def _atempo_chain(factor: float) -> str:
    """atempo only supports 0.5..2.0 — chain filters for wider factors."""
    f = float(factor)
    if f <= 0:
        raise MediaEditError("speed factor must be positive")
    parts = []
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def _segment_durations(segments: list[dict[str, Any]]) -> list[float]:
    """Resolve each segment's playout duration (probe + trim + speed)."""
    from .videos import video_probe, parse_time
    durs = []
    for seg in segments:
        if seg.get("kind") == "image":
            full = float(seg.get("duration", 3.0))
        else:
            info = video_probe(seg["path"])
            full = float(info.get("duration") or 0)
            if full <= 0:
                raise MediaEditError(
                    f"cannot determine duration of {seg['path']}")
        s = parse_time(seg.get("start", 0) or 0)
        e = seg.get("end")
        e = parse_time(e) if e is not None else full
        if e <= s:
            raise MediaEditError(f"segment end ({e}) must be after start ({s})")
        raw_speed = seg.get("speed", 1.0)
        try:
            speed = float(raw_speed if raw_speed is not None else 1.0)
        except (TypeError, ValueError):
            raise MediaEditError(
                f"bad segment speed {raw_speed!r}: must be a number") from None
        if speed <= 0:
            raise MediaEditError(
                f"segment speed must be positive, got {raw_speed!r}")
        durs.append((e - s) / speed)
    return durs


def compile_video(segments: list[dict[str, Any]], *,
                  transitions: list[dict[str, Any]] | None = None,
                  title: dict[str, Any] | None = None,
                  lower_thirds: list[dict[str, Any]] | None = None,
                  duck: dict[str, Any] | None = None,
                  speed_ramps: list[dict[str, Any]] | None = None,
                  timeline_trim: tuple[float | None, float | None] | None = None,
                  chapters: list[dict[str, Any]] | None = None,
                  export: dict[str, Any] | str | None = None,
                  work_dir: str | os.PathLike[str] | None = None
                  ) -> dict[str, Any]:
    """Compile studio video ops → ffmpeg inputs + filter_complex.

    Pure (no ffmpeg run): safe to unit-test. Returns {"inputs": [...],
    "filter_complex": str, "maps": [...], "extra_args": [...],
    "duration": float, "preset": str, "preset_params": dict,
    "v_label": str, "a_label": str,
    "loop_inputs": {input_index: seconds} (still images needing -loop),
    "chapters_file": path|None}.
    """
    from .videos import parse_time, video_probe
    if not segments:
        raise MediaEditError("video project needs at least one segment")
    transitions = transitions or []
    lower_thirds = lower_thirds or []
    speed_ramps = speed_ramps or []
    chapters = chapters or []

    work = Path(work_dir) if work_dir else Path.cwd()
    work.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    fc: list[str] = []
    seg_durs = _segment_durations(segments)
    # Still-image inputs (slideshow stills, title card, lower thirds) must
    # be looped at render time — a single PNG frame would starve the
    # segment filters. Maps input index -> loop duration in seconds.
    loop_inputs: dict[int, float] = {}
    full_timeline_loop: list[int] = []  # filled once total_dur is known

    # ---- per-segment normalization -------------------------------------
    for i, seg in enumerate(segments):
        inputs.append(str(seg["path"]))
        if seg.get("kind") == "image":
            dur = seg_durs[i]
            loop_inputs[i] = dur
            fc.append(
                f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
                f"trim=0:{dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v{i}]")
            fc.append(f"anullsrc=r=48000:d={dur:.3f}[a{i}]")
            continue
        s = parse_time(seg.get("start", 0) or 0)
        has_audio = any(st.get("type") == "audio"
                        for st in video_probe(seg["path"]).get("streams", []))
        speed = float(seg.get("speed", 1.0) or 1.0)
        vchain = (f"[{i}:v]trim={s:.3f}:{s + seg_durs[i] * speed:.3f},"
                  f"setpts=PTS-STARTPTS")
        if speed != 1.0:
            vchain += f",setpts=PTS/{speed:.4f}"
        vchain += (",fps=30,scale=1920:1080:force_original_aspect_ratio="
                   "decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                   f"format=yuv420p[v{i}]")
        fc.append(vchain)
        if has_audio:
            achain = (f"[{i}:a]atrim={s:.3f}:{s + seg_durs[i] * speed:.3f},"
                      f"asetpts=PTS-STARTPTS")
            if speed != 1.0:
                achain += f",{_atempo_chain(speed)}"
            achain += f",aresample=48000[a{i}]"
            fc.append(achain)
        else:
            fc.append(f"anullsrc=r=48000:d={seg_durs[i]:.3f}[a{i}]")

    vlabels = [f"[v{i}]" for i in range(len(segments))]
    alabels = [f"[a{i}]" for i in range(len(segments))]

    # ---- title card (becomes segment 0) --------------------------------
    if title:
        t_dur = float(title.get("duration", 3.0))
        t_path = work / f"studio-title-{os.getpid()}.png"
        _render_title_card(t_path, title)
        t_idx = len(inputs)
        inputs.append(str(t_path))
        full_timeline_loop.append(t_idx)
        fade = min(0.5, t_dur / 4)
        fc.append(
            f"[{t_idx}:v]scale=1920:1080,fps=30,format=yuv420p,"
            f"fade=t=in:st=0:d={fade:.2f}:alpha=1,"
            f"fade=t=out:st={t_dur - fade:.2f}:d={fade:.2f}:alpha=1[vt]")
        fc.append(f"anullsrc=r=48000:d={t_dur:.3f}[at]")
        vlabels.insert(0, "[vt]")
        alabels.insert(0, "[at]")
        seg_durs.insert(0, t_dur)

    # ---- transitions: xfade video + acrossfade audio --------------------
    n = len(vlabels)
    if n > 1:
        if transitions and len(transitions) != n - 1:
            raise MediaEditError(
                f"{n - 1} transitions needed for {n} segments, "
                f"got {len(transitions)}")
        durs_t = list(seg_durs)
        cur_v, cur_a = vlabels[0], alabels[0]
        offset = durs_t[0]
        for i in range(1, n):
            tr = transitions[i - 1] if transitions else {}
            ttype = tr.get("type", "fade")
            if ttype not in TRANSITIONS:
                raise MediaEditError(f"unknown transition {ttype!r}; "
                                     f"use {list(TRANSITIONS)}")
            td = min(float(tr.get("duration", 0.5)), durs_t[i - 1] / 2,
                     durs_t[i] / 2)
            td = max(0.1, td)
            off = offset - td
            fc.append(f"{cur_v}{vlabels[i]}xfade=transition={ttype}:"
                      f"duration={td:.3f}:offset={off:.3f}[x{i}]")
            fc.append(f"{cur_a}{alabels[i]}acrossfade=d={td:.3f}[m{i}]")
            cur_v, cur_a = f"[x{i}]", f"[m{i}]"
            offset = off + durs_t[i]
        v_out, a_out = cur_v, cur_a
        total_dur = offset
    else:
        v_out, a_out = vlabels[0], alabels[0]
        total_dur = seg_durs[0]

    # ---- lower thirds ----------------------------------------------------
    for j, lt in enumerate(lower_thirds):
        lt_path = work / f"studio-lt{j}-{os.getpid()}.png"
        _render_lower_third(lt_path, lt)
        lt_idx = len(inputs)
        inputs.append(str(lt_path))
        full_timeline_loop.append(lt_idx)
        start = float(lt.get("start", 1.0))
        dur = float(lt.get("duration", 4.0))
        fade = min(0.4, dur / 4)
        fc.append(
            f"[{lt_idx}:v]format=rgba,"
            f"fade=t=in:st=0:d={fade:.2f}:alpha=1,"
            f"fade=t=out:st={dur - fade:.2f}:d={fade:.2f}:alpha=1[lt{j}]")
        v_out_new = f"[vlt{j}]"
        fc.append(
            f"{v_out}[lt{j}]overlay=0:H-h-70:"
            f"enable='between(t,{start:.2f},{start + dur:.2f})'{v_out_new}")
        v_out = v_out_new

    # ---- timeline speed ramps -------------------------------------------
    for k, ramp in enumerate(speed_ramps):
        factor = float(ramp.get("factor", 1.0))
        rs = parse_time(ramp.get("start") or 0)
        re_raw = ramp.get("end")
        re_ = parse_time(re_raw) if re_raw is not None else total_dur
        if not 0 <= rs < re_ <= total_dur:
            raise MediaEditError(
                f"speed ramp [{rs}, {re_}] outside timeline 0..{total_dur:.1f}")
        # split → retime middle → concat
        fc.append(f"{v_out}split=3[sv{k}a][sv{k}b][sv{k}c]")
        fc.append(f"{a_out}asplit=3[sa{k}a][sa{k}b][sa{k}c]")
        mid_dur = (re_ - rs) / factor
        fc.append(f"[sv{k}b]trim={rs:.3f}:{re_:.3f},setpts=PTS-STARTPTS,"
                  f"setpts=PTS/{factor:.4f}[sv{k}bm]")
        fc.append(f"[sa{k}b]atrim={rs:.3f}:{re_:.3f},asetpts=PTS-STARTPTS,"
                  f"{_atempo_chain(factor)}[sa{k}bm]")
        fc.append(f"[sv{k}a]trim=0:{rs:.3f},setpts=PTS-STARTPTS[sv{k}ah]")
        fc.append(f"[sv{k}c]trim={re_:.3f}:{total_dur:.3f},"
                  f"setpts=PTS-STARTPTS[sv{k}ch]")
        fc.append(f"[sa{k}a]atrim=0:{rs:.3f},asetpts=PTS-STARTPTS[sa{k}ah]")
        fc.append(f"[sa{k}c]atrim={re_:.3f}:{total_dur:.3f},"
                  f"asetpts=PTS-STARTPTS[sa{k}ch]")
        fc.append(f"[sv{k}ah][sv{k}bm][sv{k}ch]concat=n=3:v=1:a=0[vsp{k}]")
        fc.append(f"[sa{k}ah][sa{k}bm][sa{k}ch]concat=n=3:v=0:a=1[asp{k}]")
        v_out, a_out = f"[vsp{k}]", f"[asp{k}]"
        total_dur = rs + mid_dur + (total_dur - re_)

    # ---- timeline trim ----------------------------------------------------
    if timeline_trim:
        ts, te = timeline_trim
        # EditStudio.cut() accepts "MM:SS" strings as well as seconds
        ts = parse_time(ts) if ts is not None else 0.0
        te = parse_time(te) if te is not None else total_dur
        if not 0 <= ts < te <= total_dur + 1e-6:
            raise MediaEditError("timeline trim out of range")
        fc.append(f"{v_out}trim={ts:.3f}:{te:.3f},setpts=PTS-STARTPTS[vt2]")
        fc.append(f"{a_out}atrim={ts:.3f}:{te:.3f},asetpts=PTS-STARTPTS[at2]")
        v_out, a_out = "[vt2]", "[at2]"
        total_dur = te - ts

    # ---- audio ducking (bgm under voice) ----------------------------------
    if duck:
        bgm = duck.get("bgm")
        if not bgm or not os.path.exists(bgm):
            raise MediaEditError(f"duck needs an existing bgm file: {bgm!r}")
        b_idx = len(inputs)
        inputs.append(str(bgm))
        bgm_gain = float(duck.get("bgm_gain", 0.5))
        duck_db = float(duck.get("amount_db", 10.0))
        # sidechaincompress thresholds in dB-ish 0..1 scale; map amount_db
        threshold = max(0.001, min(0.5, 0.05 * (duck_db / 10.0)))
        fc.append(f"[{b_idx}:a]aresample=48000,volume={bgm_gain}[bgm]")
        fc.append(f"[bgm]{a_out}sidechaincompress=threshold={threshold:.3f}:"
                  f"ratio=8:attack=15:release=400[ducked]")
        fc.append(f"[ducked]{a_out}amix=inputs=2:duration=first:"
                  f"dropout_transition=0[aout]")
        a_out = "[aout]"

    # still-image overlays (title / lower thirds) loop the full timeline
    for idx in full_timeline_loop:
        loop_inputs[idx] = total_dur

    # ---- export preset ----------------------------------------------------
    preset_name = export if isinstance(export, str) else (export or {}).get("preset", "web-optimized")
    if preset_name not in EXPORT_PRESETS:
        raise MediaEditError(f"unknown export preset {preset_name!r}; "
                             f"use {sorted(EXPORT_PRESETS)}")
    preset = dict(EXPORT_PRESETS[preset_name])
    if isinstance(export, dict):
        preset.update({k: v for k, v in export.items() if k != "preset"})
    v_final, a_final = v_out, a_out
    w, h = preset.get("width"), preset.get("height")
    if w and h:
        if preset.get("fit") == "fill":
            fc.append(f"{v_final}scale={w}:{h}:force_original_aspect_ratio="
                      f"increase,crop={w}:{h}[vpre]")
        else:
            fc.append(f"{v_final}scale={w}:{h}:force_original_aspect_ratio="
                      f"decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                      f"setsar=1[vpre]")
        v_final = "[vpre]"
    # canonical final video label: every consumer (maps, gif path) uses
    # this instead of assuming a preset-specific label exists
    fc.append(f"{v_final}null[vout]")
    v_final = "[vout]"
    maps = ["-map", v_final, "-map", a_final]
    extra = ["-shortest"] if duck else []
    chapters_file = None
    if chapters:
        chapters_file = work / f"studio-chapters-{os.getpid()}.txt"
        _write_chapters(chapters_file, chapters)
        inputs.append(str(chapters_file))
        maps += ["-map_chapters", str(len(inputs) - 1)]

    return {
        "inputs": inputs,
        "filter_complex": ";".join(fc),
        "maps": maps,
        "extra_args": extra,
        "duration": round(total_dur, 3),
        "preset": preset_name,
        "preset_params": preset,
        "v_label": v_final,
        "a_label": a_final,
        "loop_inputs": loop_inputs,
        "chapters_file": str(chapters_file) if chapters_file else None,
    }


def render_video_compiled(compiled: dict[str, Any],
                          out_path: str | os.PathLike[str],
                          *,
                          timeout: float = 600.0,
                          progress_cb: Any = None) -> dict[str, Any]:
    """Run ffmpeg for a compiled project. Writes a NEW file."""
    from .videos import run_ffmpeg, FFMPEG_TIMEOUT
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preset = compiled["preset_params"]
    loop_inputs = compiled.get("loop_inputs", {})
    args: list[str] = []
    for i, inp in enumerate(compiled["inputs"]):
        if i in loop_inputs:
            # still images (title card / lower thirds / slideshow stills):
            # loop them so the segment filters never starve for frames
            args += ["-loop", "1", "-framerate", "30",
                     "-t", str(loop_inputs[i]), "-i", inp]
        else:
            args += ["-i", inp]
    # gif-preview is a special path (palette)
    if compiled["preset"] == "gif-preview":
        return _render_gif_preview(compiled, args, out, timeout=timeout,
                                   progress_cb=progress_cb)
    args += ["-filter_complex", compiled["filter_complex"]]
    args += compiled["maps"]
    args += ["-c:v", preset.get("vcodec", "libx264")]
    if preset.get("vcodec", "libx264") in ("libx264", "libx265"):
        args += ["-preset", preset.get("preset", "medium"),
                 "-crf", str(preset.get("crf", 23))]
    args += ["-c:a", preset.get("acodec", "aac"), "-pix_fmt", "yuv420p"]
    if preset.get("faststart"):
        args += ["-movflags", "+faststart"]
    args += compiled["extra_args"]
    args += [str(out)]
    run = run_ffmpeg(args, timeout=timeout, progress_cb=progress_cb,
                     duration=compiled["duration"])
    return {"output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"], "duration": compiled["duration"],
            "preset": compiled["preset"]}


def _render_gif_preview(compiled: dict[str, Any], input_args: list[str],
                        out: Path, *, timeout: float,
                        progress_cb: Any = None) -> dict[str, Any]:
    from .videos import run_ffmpeg
    preset = compiled["preset_params"]
    width = preset.get("width", 480)
    fps = preset.get("fps", 12)
    v_label = compiled.get("v_label", "[vout]")
    a_label = compiled.get("a_label", "[aout]")
    vf = (f"{compiled['filter_complex']};"
          f"{a_label}anullsink;"  # gif has no audio: consume the chain
          f"{v_label}fps={fps},scale={width}:-2:flags=lanczos,"
          f"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse[gout]")
    args = input_args + ["-filter_complex", vf,
                         "-map", "[gout]", str(out)]
    run = run_ffmpeg(args, timeout=min(timeout, 300), progress_cb=progress_cb,
                     duration=compiled["duration"])
    return {"output": str(out), "bytes": out.stat().st_size,
            "seconds": run["seconds"], "duration": compiled["duration"],
            "preset": "gif-preview"}


def _render_title_card(path: Path, spec: dict[str, Any]) -> None:
    Image = _require_pillow()
    w, h = 1920, 1080
    bg = spec.get("bg", "#14161c")
    canvas = Image.new("RGB", (w, h), bg)
    # subtle top glow bar for style
    accent = spec.get("accent")
    if accent:
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, h // 2 - 140, w, h // 2 - 128], fill=accent)
    text = spec.get("text", "")
    sub = spec.get("subtitle", "")
    if text:
        layer = render_text_layer(
            text, font=spec.get("font"), size=spec.get("font_size", 96),
            color=spec.get("color", "white"), align="center",
            box_width=w, letter_spacing=int(spec.get("letter_spacing", 2)),
            shadow=True)
        x = (w - layer.width) // 2
        y = (h - layer.height) // 2 - (40 if sub else 0)
        canvas.paste(layer, (x, y), layer)
    if sub:
        layer = render_text_layer(
            sub, font=spec.get("font"), size=spec.get("subtitle_size", 44),
            color=spec.get("subtitle_color", "#9aa0aa"), align="center",
            box_width=w)
        x = (w - layer.width) // 2
        y = h // 2 + 60
        canvas.paste(layer, (x, y), layer)
    canvas.save(path, "PNG")


def _render_lower_third(path: Path, spec: dict[str, Any]) -> None:
    Image = _require_pillow()
    from PIL import ImageDraw
    w, h = 1920, 360
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bg = spec.get("bg", (10, 12, 16, 210))
    d.rounded_rectangle([40, 40, w - 40, h - 40], radius=24, fill=bg)
    accent = spec.get("accent", "#e8b33c")
    d.rectangle([40, 40, 52, h - 40], fill=accent)
    text = render_text_layer(
        spec.get("text", ""), font=spec.get("font"),
        size=spec.get("font_size", 54), color=spec.get("color", "white"),
        align="left", box_width=w - 240)
    layer.alpha_composite(text, (110, (h - text.height) // 2))
    layer.save(path, "PNG")


def _write_chapters(path: Path, chapters: list[dict[str, Any]]) -> None:
    lines = [";FFMETADATA1"]
    for ch in chapters:
        start_ms = int(float(ch.get("at", 0)) * 1000)
        end_ms = int(float(ch.get("end", start_ms / 1000 + 60)) * 1000)
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={ch.get('name', 'Chapter')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# EditStudio: the session class
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
               ".gif", ".avif"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class EditStudio:
    """A professional edit session: non-destructive op stack over one
    source file. Every method appends a serializable op and returns
    ``self`` (chainable). ``render()`` replays the stack from scratch —
    undo/redo, project save/load, and templates all build on that.

    Image ops run synchronously through :mod:`.images` (extended with the
    studio ops registered above). Video ops compile to a single ffmpeg
    filter graph and render via :mod:`.jobs` (background) or blocking.
    """

    def __init__(self, source: str | os.PathLike[str] | None = None, *,
                 kind: str = "auto", name: str = "untitled") -> None:
        self.name = name
        self.source = str(source) if source is not None else None
        self.kind = kind  # "auto" | "image" | "video"
        self.ops: list[dict[str, Any]] = []
        self._undone: list[dict[str, Any]] = []
        self.created = time.time()

    # -- stack mechanics -------------------------------------------------
    def _push(self, op_name: str, params: dict[str, Any],
              label: str = "") -> "EditStudio":
        self.ops.append({"op": op_name, "params": _jsonable(params),
                         "label": label or op_name})
        self._undone.clear()
        return self

    def op(self, name: str, label: str = "", **params: Any) -> "EditStudio":
        """Append any allowlisted op (escape hatch for the full engine)."""
        return self._push(name, params, label or name)

    def undo(self) -> dict[str, Any] | None:
        if not self.ops:
            return None
        op = self.ops.pop()
        self._undone.append(op)
        return op

    def redo(self) -> dict[str, Any] | None:
        if not self._undone:
            return None
        op = self._undone.pop()
        self.ops.append(op)
        return op

    def clear(self) -> "EditStudio":
        self.ops.clear()
        self._undone.clear()
        return self

    def describe(self) -> str:
        lines = [f"studio project '{self.name}' "
                 f"({self._resolve_kind()}, {len(self.ops)} ops)"]
        if self.source:
            lines.append(f"  source: {self.source}")
        for i, op in enumerate(self.ops, 1):
            params = ", ".join(f"{k}={v}" for k, v in op["params"].items())
            lines.append(f"  {i}. {op['op']}({params})")
        if self._undone:
            lines.append(f"  ({len(self._undone)} undone)")
        return "\n".join(lines)

    # -- image conveniences ----------------------------------------------
    def filter(self, preset: str, strength: float = 1.0) -> "EditStudio":
        return self._push("filter", {"preset": preset, "strength": strength},
                          f"filter:{preset}")

    def grade(self, **kwargs: Any) -> "EditStudio":
        return self._push("grade", kwargs, "grade")

    def text(self, text: str, **kwargs: Any) -> "EditStudio":
        return self._push("text_layer", {"text": text, **kwargs},
                          f"text:{text[:24]!r}")

    def image_layer(self, path: str | os.PathLike[str],
                    **kwargs: Any) -> "EditStudio":
        return self._push("composite",
                          {"layers": [{"type": "image", "path": str(path),
                                       **kwargs}]},
                          f"layer:{Path(path).name}")

    def shape_layer(self, **kwargs: Any) -> "EditStudio":
        return self._push("shape_layer", kwargs, "shape")

    def layers(self, layer_list: list[dict[str, Any]]) -> "EditStudio":
        return self._push("composite", {"layers": layer_list},
                          f"{len(layer_list)} layers")

    def collage(self, images: list[str | os.PathLike[str]],
                template: str = "grid", **kwargs: Any) -> "EditStudio":
        return self._push("collage",
                          {"images": [str(p) for p in images],
                           "template": template, **kwargs},
                          f"collage:{template}")

    def smart_crop(self, aspect: str,
                   mode: str = "saliency") -> "EditStudio":
        return self._push("smart_crop", {"aspect": aspect, "mode": mode},
                          f"smart_crop:{aspect}")

    def letterbox(self, aspect: str = "21:9",
                  color: str = "black") -> "EditStudio":
        return self._push("letterbox", {"aspect": aspect, "color": color},
                          f"letterbox:{aspect}")

    def meme(self, top: str = "", bottom: str = "") -> "EditStudio":
        return self._push("meme", {"top": top, "bottom": bottom}, "meme")

    def generative_edit(self, instruction: str, *,
                        mask: Any | None = None,
                        strength: float = 0.75,
                        seed: int | None = None,
                        backend: str | None = None,
                        **kwargs: Any) -> "EditStudio":
        """AI instruction edit (non-destructive; re-runs backend on render).

        ``mask``: (l, t, r, b) box or mask-image path (serializable, so it
        survives project save/load)."""
        params: dict[str, Any] = {"instruction": instruction,
                                  "strength": strength, "seed": seed,
                                  "backend": backend or "auto"}
        if mask is not None:
            params["mask"] = (list(mask) if isinstance(mask, (list, tuple))
                              else str(mask))
        params.update(kwargs)
        return self._push("generative_edit", params,
                          f"ai:{instruction[:40]!r}")

    # -- video conveniences ----------------------------------------------
    def cut(self, start: Any = 0, end: Any | None = None) -> "EditStudio":
        return self._push("v_trim", {"start": start, "end": end}, "cut")

    def add_clip(self, path: str | os.PathLike[str], *,
                 start: Any | None = None, end: Any | None = None,
                 speed: float = 1.0) -> "EditStudio":
        return self._push("v_clip", {"path": str(path), "start": start,
                                    "end": end, "speed": speed},
                          f"clip:{Path(path).name}")

    def add_image(self, path: str | os.PathLike[str],
                  duration: float = 3.0) -> "EditStudio":
        return self._push("v_image", {"path": str(path), "duration": duration},
                          f"still:{Path(path).name}")

    def transition(self, type: str = "fade",
                   duration: float = 0.5) -> "EditStudio":
        if type not in TRANSITIONS:
            raise MediaEditError(f"unknown transition {type!r}")
        return self._push("v_transition", {"type": type, "duration": duration},
                          f"transition:{type}")

    def title_card(self, text: str, duration: float = 3.0,
                   **style: Any) -> "EditStudio":
        return self._push("v_title", {"text": text, "duration": duration,
                                      **style}, f"title:{text[:24]!r}")

    def lower_third(self, text: str, start: float = 1.0,
                    duration: float = 4.0, **style: Any) -> "EditStudio":
        return self._push("v_lower_third",
                          {"text": text, "start": start, "duration": duration,
                           **style}, f"lower3rd:{text[:24]!r}")

    def duck(self, bgm: str | os.PathLike[str], amount_db: float = 10.0,
             bgm_gain: float = 0.5) -> "EditStudio":
        return self._push("v_duck", {"bgm": str(bgm), "amount_db": amount_db,
                                     "bgm_gain": bgm_gain}, "duck")

    def speed(self, factor: float, start: float | None = None,
              end: float | None = None) -> "EditStudio":
        return self._push("v_speed", {"factor": factor, "start": start,
                                      "end": end}, f"speed:{factor}×")

    def chapter(self, name: str, at: float,
                end: float | None = None) -> "EditStudio":
        return self._push("v_chapter", {"name": name, "at": at, "end": end},
                          f"chapter:{name}")

    def export(self, preset: str = "web-optimized") -> "EditStudio":
        if preset not in EXPORT_PRESETS:
            raise MediaEditError(f"unknown export preset {preset!r}")
        return self._push("v_export", {"preset": preset},
                          f"export:{preset}")

    # -- projects ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": PROJECT_VERSION, "name": self.name,
                "kind": self.kind, "source": self.source,
                "ops": self.ops, "created": self.created}

    def save_project(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditStudio":
        if data.get("version") != PROJECT_VERSION:
            raise MediaEditError(
                f"unsupported studio project version {data.get('version')}")
        st = cls(data.get("source"), kind=data.get("kind", "auto"),
                 name=data.get("name", "untitled"))
        st.ops = data.get("ops", [])
        st.created = data.get("created", time.time())
        return st

    @classmethod
    def load_project(cls, path: str | os.PathLike[str]) -> "EditStudio":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise MediaEditError(
                f"could not load studio project {path}: {exc}") from exc
        return cls.from_dict(data)

    # -- render ------------------------------------------------------------
    def _resolve_kind(self) -> str:
        if self.kind != "auto":
            return self.kind
        if self.source:
            ext = Path(self.source).suffix.lower()
            if ext in _VIDEO_EXTS:
                return "video"
            if ext in _IMAGE_EXTS:
                return "image"
        if any(o["op"].startswith("v_") for o in self.ops):
            return "video"
        if self.source:
            return "image"
        raise MediaEditError("studio has no source and no video ops; "
                             "pass kind='video' explicitly for slideshows")

    def render(self, *, out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "studio", fmt: str | None = None,
               quality: int = 90, wait: bool = False,
               timeout: float = 600.0) -> dict[str, Any]:
        """Replay the op stack from scratch. Images render synchronously;
        video renders as a background job unless ``wait=True``."""
        kind = self._resolve_kind()
        if kind == "image":
            return self._render_image(out_dir=out_dir, suffix=suffix, fmt=fmt,
                                      quality=quality)
        return self._render_video(out_dir=out_dir, suffix=suffix, wait=wait,
                                  timeout=timeout)

    def _render_image(self, *, out_dir: Any = None, suffix: str = "studio",
                      fmt: str | None = None,
                      quality: int = 90) -> dict[str, Any]:
        if not self.source:
            raise MediaEditError("image studio needs a source file")
        if any(o["op"].startswith("v_") for o in self.ops):
            raise MediaEditError("video ops in an image studio project")
        src = Path(self.source)
        if not src.exists():
            raise MediaEditError(f"no such image: {self.source}")
        img = load_image(src)
        try:
            chain = [{"op": o["op"], **o["params"]} for o in self.ops]
            out_img = apply_chain(img, chain) if chain else img.copy()
        except Exception:
            img.close()
            raise
        if out_img is not img:
            try:
                img.close()
            except Exception:  # noqa: BLE001
                pass
        out_ext = f".{fmt.lower()}" if fmt else src.suffix.lower()
        target = Path(out_dir) if out_dir else src.parent / "edited"
        out_path = _unique_output(src, target, suffix, out_ext)
        save_image(out_img, out_path, fmt=fmt, quality=quality)
        try:
            out_img.close()
        except Exception:  # noqa: BLE001
            pass
        return {"input": str(src), "output": str(out_path),
                "ops": [o["op"] for o in self.ops],
                "bytes": out_path.stat().st_size, "kind": "image"}

    def _collect_video_ops(self) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        if self.source:
            segments.append({"path": self.source, "kind": "video"})
        transitions: list[dict[str, Any]] = []
        lower_thirds: list[dict[str, Any]] = []
        speed_ramps: list[dict[str, Any]] = []
        chapters: list[dict[str, Any]] = []
        title = duck = timeline_trim = None
        export_spec: str = "web-optimized"
        for op in self.ops:
            n, p = op["op"], op["params"]
            if n == "v_clip":
                segments.append({"path": p["path"], "kind": "video",
                                 "start": p.get("start"), "end": p.get("end"),
                                 "speed": p.get("speed", 1.0)})
            elif n == "v_image":
                segments.append({"path": p["path"], "kind": "image",
                                 "duration": p.get("duration", 3.0)})
            elif n == "v_transition":
                transitions.append(p)
            elif n == "v_title":
                title = p
            elif n == "v_lower_third":
                lower_thirds.append(p)
            elif n == "v_duck":
                duck = p
            elif n == "v_speed":
                speed_ramps.append(p)
            elif n == "v_trim":
                timeline_trim = (p.get("start"), p.get("end"))
            elif n == "v_chapter":
                chapters.append(p)
            elif n == "v_export":
                export_spec = p.get("preset", "web-optimized")
            else:
                raise MediaEditError(
                    f"op {n!r} is not a video op (image studio?)")
        if not segments:
            raise MediaEditError("video studio needs a source or clips")
        need = len(segments) + (1 if title else 0) - 1
        if transitions and need > 1 and len(transitions) == 1:
            transitions = transitions * need
        return {"segments": segments, "transitions": transitions,
                "title": title, "lower_thirds": lower_thirds, "duck": duck,
                "speed_ramps": speed_ramps, "timeline_trim": timeline_trim,
                "chapters": chapters, "export": export_spec}

    def _render_video(self, *, out_dir: Any = None, suffix: str = "studio",
                      wait: bool = False,
                      timeout: float = 600.0) -> dict[str, Any]:
        from .videos import ffmpeg_path
        from .jobs import get_manager
        import tempfile
        ffmpeg_path()  # fail fast with the clear hint
        parts = self._collect_video_ops()
        work = Path(tempfile.mkdtemp(prefix="studio-"))
        compiled = compile_video(
            parts["segments"], transitions=parts["transitions"],
            title=parts["title"], lower_thirds=parts["lower_thirds"],
            duck=parts["duck"], speed_ramps=parts["speed_ramps"],
            timeline_trim=parts["timeline_trim"], chapters=parts["chapters"],
            export=parts["export"], work_dir=work)
        first = Path(parts["segments"][0]["path"])
        ext = compiled["preset_params"].get("ext", ".mp4")
        target = Path(out_dir) if out_dir else first.parent / "edited"
        out = _unique_output(first, target, suffix, ext)

        def _run(progress_cb: Any) -> dict[str, Any]:
            return render_video_compiled(compiled, out, timeout=timeout,
                                         progress_cb=progress_cb)

        label = f"studio:{self.name} ({len(self.ops)} ops)"
        if wait:
            mgr = get_manager()
            jid = mgr.submit("video", label, _run,
                             input_ref=str(first))
            return mgr.wait(jid, timeout=timeout)
        mgr = get_manager()
        jid = mgr.submit("video", label, _run, input_ref=str(first))
        return {"job_id": jid, "status": "queued", "label": label,
                "poll": "media_job_status"}

    # -- compare -------------------------------------------------------------
    def compare(self, mode: str = "side-by-side", *,
                out_dir: str | os.PathLike[str] | None = None
                ) -> dict[str, Any]:
        """Render, then export a before/after comparison (image studios)."""
        if self._resolve_kind() != "image":
            raise MediaEditError("compare is for image studios")
        if not self.source:
            raise MediaEditError("compare needs a source file")
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="studio-cmp-"))
        rendered = self._render_image(out_dir=tmp, suffix="render")
        before = load_image(self.source)
        after = load_image(rendered["output"])
        try:
            target = Path(out_dir) if out_dir else Path(self.source).parent / "edited"
            target.mkdir(parents=True, exist_ok=True)
            if mode == "html":
                html = target / f"{Path(self.source).stem}-compare.html"
                compare_html(self.source, rendered["output"], html)
                return {"mode": "html", "output": str(html),
                        "render": rendered["output"]}
            comp = compare_render(before, after, mode=mode)
            out = target / f"{Path(self.source).stem}-compare.png"
            save_image(comp, out)
            try:
                comp.close()
            except Exception:  # noqa: BLE001
                pass
            return {"mode": mode, "output": str(out),
                    "render": rendered["output"]}
        finally:
            before.close()
            after.close()
            shutil.rmtree(tmp, ignore_errors=True)

    # -- batch ---------------------------------------------------------------
    @classmethod
    def batch(cls, src_dir: str | os.PathLike[str],
              project: str | os.PathLike[str] | "EditStudio", *,
              pattern: str = "*.jpg",
              out_dir: str | os.PathLike[str] | None = None,
              suffix: str = "studio", wait: bool = False,
              timeout: float = 600.0) -> list[dict[str, Any]]:
        """Apply a saved project across every file in ``src_dir``."""
        proto = (cls.load_project(project) if isinstance(project, (str, Path))
                 else project)
        results = []
        for path in sorted(Path(src_dir).glob(pattern)):
            if not path.is_file():
                continue
            st = cls.from_dict(proto.to_dict())
            st.source = str(path)
            try:
                results.append(st.render(out_dir=out_dir, suffix=suffix,
                                         wait=wait, timeout=timeout))
            except MediaEditError as exc:
                _log.warning("studio batch skipped %s: %s", path, exc)
                results.append({"input": str(path), "error": str(exc)})
        return results


# ---------------------------------------------------------------------------
# project templates
# ---------------------------------------------------------------------------

def _build_podcast_clip(**p: Any) -> EditStudio:
    st = EditStudio(p["source"], kind="video", name="podcast-clip")
    if p.get("start") is not None or p.get("end") is not None:
        st.cut(p.get("start", 0), p.get("end"))
    st.lower_third(p.get("title", "Untitled"),
                   start=float(p.get("title_at", 0.5)),
                   duration=float(p.get("title_duration", 6.0)),
                   accent=p.get("accent", "#e8b33c"))
    if p.get("show"):
        st.title_card(p["show"], duration=2.0,
                      subtitle=p.get("title", ""))
    st.export("social-vertical")
    return st


def _build_quote_card(**p: Any) -> EditStudio:
    st = EditStudio(p["background"], kind="image", name="quote-card")
    st.filter(p.get("filter", "cinematic"), strength=0.9)
    # dim for readability (full-canvas shape layer)
    st.op("composite", layers=[{"type": "shape", "shape": "rect",
                                "box": "full", "fill": "black",
                                "opacity": float(p.get("dim", 0.45))}])
    st.text(p["quote"], position="center", size=None,
            box_width=int(p.get("box_width", 1400)), color="white",
            shadow=True, align="center",
            letter_spacing=int(p.get("letter_spacing", 1)))
    st.text(f"— {p.get('author', 'Unknown')}", position="bottom",
            size=int(p.get("author_size", 52)), color="#e8e8e8",
            shadow=True, margin=90)
    return st


def _build_product_showcase(**p: Any) -> EditStudio:
    images = list(p["images"])
    if not images:
        raise MediaEditError("product-showcase needs 'images'")
    st = EditStudio(images[0], kind="image", name="product-showcase")
    rest = images[1:]
    if rest:
        cols = 2 if len(rest) > 1 else 1
        st.collage(rest, template="grid", cols=cols, gap=10,
                   captions=p.get("captions"))
    st.text(p.get("title", "New Arrival"), position="top", size=72,
            color="white", stroke_width=3, shadow=True, margin=40)
    if p.get("price"):
        st.text(str(p["price"]), position="bottom-right", size=64,
                color="#ffd94d", stroke_width=2, shadow=True, margin=40)
    if p.get("logo"):
        st.image_layer(p["logo"], position="bottom-left", scale=0.14,
                       opacity=0.95, margin=40)
    return st


def _build_meme(**p: Any) -> EditStudio:
    st = EditStudio(p["image"], kind="image", name="meme")
    st.meme(top=p.get("top", ""), bottom=p.get("bottom", ""))
    return st


def _build_slideshow(**p: Any) -> EditStudio:
    images = list(p["images"])
    if not images:
        raise MediaEditError("slideshow needs 'images'")
    st = EditStudio(None, kind="video", name="slideshow")
    dur = float(p.get("duration_each", 3.0))
    for im in images:
        st.add_image(im, duration=dur)
    st.transition(p.get("transition", "fade"),
                  duration=float(p.get("transition_duration", 0.6)))
    if p.get("title"):
        st.title_card(p["title"], duration=2.5,
                      subtitle=p.get("subtitle", ""))
    if p.get("bgm"):
        st.duck(p["bgm"], amount_db=float(p.get("duck_db", 0.0)),
                bgm_gain=float(p.get("bgm_gain", 0.8)))
    st.export(p.get("export", "web-optimized"))
    return st


TEMPLATES: dict[str, dict[str, Any]] = {
    "podcast-clip": {
        "kind": "video",
        "params": ["source", "title"],
        "optional": ["show", "start", "end", "title_at", "title_duration",
                     "accent"],
        "build": _build_podcast_clip,
        "blurb": "Vertical 9:16 clip with lower-third title + show card.",
    },
    "quote-card": {
        "kind": "image",
        "params": ["background", "quote"],
        "optional": ["author", "filter", "dim", "box_width", "author_size",
                     "letter_spacing"],
        "build": _build_quote_card,
        "blurb": "Cinematic quote card: graded bg, dim, centered quote.",
    },
    "product-showcase": {
        "kind": "image",
        "params": ["images"],
        "optional": ["title", "price", "logo", "captions"],
        "build": _build_product_showcase,
        "blurb": "Collage grid + title, price tag, logo watermark.",
    },
    "meme": {
        "kind": "image",
        "params": ["image"],
        "optional": ["top", "bottom"],
        "build": _build_meme,
        "blurb": "Top/bottom meme captions with auto-fit type.",
    },
    "slideshow": {
        "kind": "video",
        "params": ["images"],
        "optional": ["duration_each", "transition", "transition_duration",
                     "title", "subtitle", "bgm", "bgm_gain", "export"],
        "build": _build_slideshow,
        "blurb": "Stills → video with transitions, title card, music.",
    },
}


def list_templates() -> dict[str, dict[str, Any]]:
    return {k: {"kind": v["kind"], "params": v["params"],
                "optional": v["optional"], "blurb": v["blurb"]}
            for k, v in TEMPLATES.items()}


def build_template(name: str, **params: Any) -> EditStudio:
    """Build a full op stack from a named template + a few params."""
    key = str(name).strip().lower().replace(" ", "-").replace("_", "-")
    if key not in TEMPLATES:
        raise MediaEditError(f"unknown template {name!r}; "
                             f"use {sorted(TEMPLATES)}")
    spec = TEMPLATES[key]
    missing = [p for p in spec["params"] if p not in params]
    if missing:
        raise MediaEditError(
            f"template {key!r} needs params {missing}; "
            f"optional: {spec['optional']}")
    return spec["build"](**params)


def studio_presets() -> dict[str, Any]:
    """Everything listable: filters, transitions, exports, templates, blends."""
    return {
        "filters": sorted(FILTER_PRESETS),
        "transitions": list(TRANSITIONS),
        "export_presets": sorted(EXPORT_PRESETS),
        "templates": list_templates(),
        "blend_modes": list(BLEND_MODES),
        "collage_templates": list(COLLAGE_TEMPLATES),
    }


__all__ = [
    "MediaEditError",
    "EditStudio",
    "FILTER_PRESETS",
    "TRANSITIONS",
    "EXPORT_PRESETS",
    "TEMPLATES",
    "BLEND_MODES",
    "COLLAGE_TEMPLATES",
    "build_template",
    "list_templates",
    "studio_presets",
    "compile_video",
    "render_video_compiled",
    "composite_layers",
    "render_text_layer",
    "compare_render",
    "compare_html",
    "discover_fonts",
    "find_font",
    "op_grade",
    "op_filter",
    "op_letterbox",
    "op_smart_crop",
    "op_text_layer",
    "op_composite",
    "op_shape_layer",
    "op_collage",
]
