"""Advanced image ops with multi-backend auto-selection.

Extends the Pillow engine in :mod:`.images`. Philosophy: Devon just works —
**no capability is limited to a single dependency**. Every operation below
tries backends in documented preference order (best free option first) and
uses the best one actually installed. The caller never picks a backend.

``pip install nomorals[media-edit]`` installs everything (Pillow +
opencv-python-headless + scikit-image), so the primary path is always
available there. With fewer packages installed, ops gracefully fall back —
every operation has a working pure-Pillow/stdlib fallback, so nothing
ever raises for a missing dependency. The one exception is automatic
document-corner detection in ``perspective_transform`` (contour finding
has no honest no-OpenCV equivalent); pass explicit corners instead.

Every function takes and returns a PIL ``Image`` (converted to/from numpy
internally), so op chains stay API-consistent. Alpha channels are preserved
where it makes sense.

Importing this module registers the ops into ``images._OP_FUNCS`` /
``images.OP_ALLOWLIST`` (additive, same pattern as :mod:`.studio`), so
``nm studio`` and ``apply_chain`` gain the new ops.

Backend-selection note for tests: backends resolve through :func:`_backend`
with a per-process cache (``_BACKENDS``); tests force fallback paths by
patching that dict, e.g. ``C._BACKENDS["cv2"] = None``.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageMath, ImageOps, ImageStat

from .images import MediaEditError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# backend resolution: best available wins, never a hard gate
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Any] = {}


def _backend(name: str) -> Any | None:
    """Import ``name`` once; return the module or None when unavailable."""
    if name not in _BACKENDS:
        try:
            _BACKENDS[name] = importlib.import_module(name)
        except ImportError:
            _BACKENDS[name] = None
    return _BACKENDS[name]


def _cv2() -> Any | None:
    return _backend("cv2")


def _ski() -> Any | None:
    return _backend("skimage")


def _np() -> Any | None:
    return _backend("numpy")


def cv2_available() -> bool:
    """True when OpenCV is importable."""
    return _cv2() is not None


def skimage_available() -> bool:
    """True when scikit-image is importable."""
    return _ski() is not None


def _backend_for(op: str) -> str:
    """Backend preference resolution shared by ops and backend_status()."""
    cv2, ski, np = _cv2(), _ski(), _np()
    has_cv = cv2 is not None and np is not None
    has_ski = ski is not None and np is not None
    if op == "denoise":
        return "opencv" if has_cv else "scikit-image" if has_ski else "pillow"
    if op == "edge_detect":
        return "opencv" if has_cv else "scikit-image" if has_ski else "pillow"
    if op == "inpaint":
        return "opencv" if has_cv else "scikit-image" if has_ski else "pillow"
    if op == "seamless_clone":
        return "opencv" if has_cv else "pillow"
    if op == "sharpen":
        return "opencv" if has_cv else "scikit-image" if has_ski else "pillow"
    if op == "cartoonize":
        return "opencv" if has_cv else "pillow"
    if op == "pencil_sketch":
        return "opencv" if has_cv else ("numpy" if np is not None
                                        else "pillow")
    if op == "perspective":
        return "opencv" if has_cv else "pillow"
    if op == "grabcut":
        return "opencv" if has_cv else "pillow"
    if op == "rescale":
        return "scikit-image" if has_ski else "opencv" if has_cv else "pillow"
    if op == "exposure":
        return "scikit-image" if has_ski else "opencv" if has_cv else "pillow"
    if op == "match_histogram":
        return "scikit-image" if has_ski else ("numpy" if np is not None
                                               else "pillow")
    if op == "adaptive_threshold":
        return "scikit-image" if has_ski else "opencv" if has_cv else (
            "numpy" if np is not None else "pillow")
    if op == "orb_features":
        return "opencv" if has_cv else ("numpy" if np is not None
                                        else "pillow")
    if op == "kmeans":
        return "opencv" if has_cv else "pillow"
    if op == "deblur":
        return "scikit-image" if has_ski else "pillow"
    if op == "superpixels":
        return "scikit-image" if has_ski else ("numpy" if np is not None
                                               else "pillow")
    if op in ("white_balance", "auto_levels"):
        return "pillow"
    if op == "color_transfer":
        return "opencv" if has_cv else "pillow"
    if op == "panorama":
        return "opencv" if has_cv else ("numpy" if np is not None
                                        else "pillow")
    if op == "retouch_smooth":
        return "opencv" if has_cv else ("numpy" if np is not None
                                        else "pillow")
    if op in ("tone_map", "detail_enhance", "stylize"):
        return "opencv" if has_cv else "pillow"
    if op == "clarity":
        return "opencv" if has_cv else ("numpy" if np is not None
                                        else "pillow")
    if op == "shadow_highlight":
        return "numpy" if np is not None else "pillow"
    if op in ("dehaze", "seam_carve", "cube_lut"):
        return "numpy" if np is not None else "pillow"
    if op in ("tilt_shift", "selective_color", "curves", "auto_levels"):
        return "pillow"
    if op == "split_tone":
        return "numpy" if np is not None else "pillow"
    if op == "replace_background":
        return "opencv" if has_cv else "pillow"
    raise MediaEditError(f"unknown op {op!r} in backend resolution")


def backend_status() -> dict[str, Any]:
    """Report installed backends and the selected backend per op."""
    ops = ["denoise", "edge_detect", "inpaint", "seamless_clone", "sharpen",
           "cartoonize", "pencil_sketch", "perspective", "grabcut",
           "rescale", "exposure", "match_histogram", "adaptive_threshold",
           "orb_features", "kmeans", "deblur", "superpixels",
           "white_balance", "auto_levels", "color_transfer", "panorama",
           "retouch_smooth", "tone_map", "detail_enhance", "stylize",
           "clarity", "shadow_highlight", "dehaze", "seam_carve",
           "tilt_shift", "selective_color", "split_tone", "curves",
           "cube_lut", "replace_background"]
    return {
        "opencv": cv2_available(),
        "scikit_image": skimage_available(),
        "numpy": _np() is not None,
        "pillow": True,
        "selected": {op: _backend_for(op) for op in ops},
    }


# ---------------------------------------------------------------------------
# PIL <-> numpy plumbing
# ---------------------------------------------------------------------------

def _rgb_array(img: Any, np: Any) -> Any:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def _gray_array(img: Any, np: Any) -> Any:
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img)


def _to_pil(arr: Any, np: Any) -> Any:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise MediaEditError(f"cannot convert array shape {arr.shape} to an image")


def _split_alpha(img: Any) -> tuple[Any, Any | None]:
    if img.mode in ("RGBA", "LA", "PA"):
        return img.convert("RGB"), img.getchannel("A")
    if img.mode == "P":
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return img, None


def _reattach_alpha(rgb_img: Any, alpha: Any | None) -> Any:
    if alpha is None:
        return rgb_img
    out = rgb_img.convert("RGBA")
    out.putalpha(alpha)
    return out


def _load_aux(aux: Any, name: str) -> Any:
    """PIL image | path | bytes -> PIL RGB image."""
    from .images import load_image  # noqa: PLC0415

    if isinstance(aux, Image.Image):
        return aux
    if isinstance(aux, (str, os.PathLike)):
        return load_image(aux).convert("RGB")
    if isinstance(aux, (bytes, bytearray)):
        import io  # noqa: PLC0415

        return Image.open(io.BytesIO(bytes(aux))).convert("RGB")
    raise MediaEditError(
        f"{name} must be a PIL image, a file path, or bytes; got "
        f"{type(aux).__name__}")


_BOX_RE = None


def parse_box(spec: Any) -> tuple[int, int, int, int] | None:
    """Parse an ``"l,t,r,b"`` box spec; return None when not a box string."""
    import re  # noqa: PLC0415

    global _BOX_RE
    if _BOX_RE is None:
        _BOX_RE = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$")
    if isinstance(spec, str) and _BOX_RE.match(spec):
        l, t, r, b = (int(v) for v in spec.split(","))
        return (l, t, r, b)
    return None


def parse_corners(spec: Any) -> list[tuple[int, int]] | None:
    """Parse ``"x1,y1;x2,y2;x3,y3;x4,y4"``; None when not a corner string."""
    if not isinstance(spec, str):
        return None
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        pts = [tuple(int(v) for v in p.split(",")) for p in parts]
    except ValueError:
        return None
    if any(len(p) != 2 for p in pts):
        return None
    return [(x, y) for x, y in pts]


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------

def _denoise_cv2(arr: Any, cv2: Any, strength: float, method: str) -> Any:
    if method == "nlmeans":
        h = float(strength)
        return cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
    s = float(strength) * 7.5
    return cv2.bilateralFilter(arr, d=9, sigmaColor=s, sigmaSpace=s)


def _denoise_skimage(arr: Any, np: Any, strength: float) -> Any:
    from skimage.restoration import denoise_nl_means  # noqa: PLC0415

    sigma = float(strength) / 55.0  # map 0..~55 strength to sigma estimate
    out = denoise_nl_means(arr, h=1.15 * sigma, fast_mode=True,
                           patch_size=5, patch_distance=6,
                           channel_axis=2)
    return (out * 255.0).astype(np.uint8)


def _denoise_pillow(img: Any, strength: float) -> Any:
    size = 3 if strength <= 12 else 5
    return img.filter(ImageFilter.MedianFilter(size=size))


def denoise(img: Any, strength: float = 10.0,
            method: str = "nlmeans") -> Any:
    """Denoise while preserving edges.

    Backends (best first): OpenCV ``fastNlMeansDenoisingColored`` /
    ``bilateralFilter`` > scikit-image ``denoise_nl_means`` > Pillow
    ``MedianFilter``.

    ``method``: ``"nlmeans"`` (best quality) or ``"bilateral"`` (faster,
    edge-preserving smoothing; Pillow fallback always uses a median
    filter). ``strength`` maps to the filter's h / sigmaColor.
    Alpha channel is preserved untouched.
    """
    if method not in ("nlmeans", "bilateral"):
        raise MediaEditError(
            f"denoise method must be 'nlmeans' or 'bilateral', "
            f"got {method!r}")
    if strength <= 0:
        raise MediaEditError("denoise strength must be > 0")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        out = _to_pil(_denoise_cv2(_rgb_array(rgb, np), cv2, strength,
                                   method), np)
    elif _ski() is not None and np is not None:
        out = _to_pil(_denoise_skimage(_rgb_array(rgb, np), np, strength),
                      np)
    else:
        _log.debug("denoise: falling back to Pillow MedianFilter")
        out = _denoise_pillow(rgb, strength)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# edge_detect
# ---------------------------------------------------------------------------

def _edges_skimage(gray: Any, np: Any, low: float, high: float) -> Any:
    from skimage.feature import canny  # noqa: PLC0415

    # map 0..255 hysteresis thresholds to skimage's 0..1 sigma-normalized
    edges = canny(gray / 255.0, sigma=1.0,
                  low_threshold=low / 255.0, high_threshold=high / 255.0)
    return (edges.astype(np.uint8)) * 255


def edge_detect(img: Any, low: float = 100.0, high: float = 200.0,
                blur: float = 1.0) -> Any:
    """Canny edge detection. Returns a grayscale (L) edge map.

    Backends (best first): OpenCV ``Canny`` > scikit-image
    ``feature.canny`` > Pillow ``FIND_EDGES``.

    A small Gaussian ``blur`` (sigma) is applied first to suppress noise
    (OpenCV/scikit-image paths); ``low``/``high`` are the hysteresis
    thresholds.
    """
    cv2, ski, np = _cv2(), _ski(), _np()
    if cv2 is not None and np is not None:
        gray = _gray_array(img, np)
        if blur and blur > 0:
            k = max(3, int(blur * 6) | 1)
            gray = cv2.GaussianBlur(gray, (k, k), blur)
        return _to_pil(cv2.Canny(gray, float(low), float(high)), np)
    if ski is not None and np is not None:
        gray = _gray_array(img, np)
        return _to_pil(_edges_skimage(gray, np, low, high), np)
    _log.debug("edge_detect: falling back to Pillow FIND_EDGES")
    return img.convert("L").filter(ImageFilter.FIND_EDGES)


# ---------------------------------------------------------------------------
# inpaint
# ---------------------------------------------------------------------------

def _mask_array(img_size: tuple[int, int], mask: Any, np: Any) -> Any:
    """Build a uint8 (0/255) damage mask from PIL/path/bytes/"l,t,r,b"."""
    box = parse_box(mask)
    if box is not None:
        l, t, r, b = box
        w, h = img_size
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise MediaEditError(
                f"inpaint_cv box {box} is outside image {(w, h)}")
        mask_img = Image.new("L", img_size, 0)
        ImageDraw.Draw(mask_img).rectangle([l, t, r, b], fill=255)
    else:
        mask_img = _load_aux(mask, "mask").convert("L")
        if mask_img.size != img_size:
            mask_img = mask_img.resize(img_size)
    arr = (np.asarray(mask_img) > 0).astype(np.uint8) * 255
    if not arr.any():
        raise MediaEditError("inpaint_cv mask is empty: nothing to inpaint")
    return arr


def _mask_pil(img_size: tuple[int, int], mask: Any,
              op: str = "mask") -> Any:
    """PIL version of :func:`_mask_array` for the no-numpy fallback path."""
    box = parse_box(mask)
    if box is not None:
        l, t, r, b = box
        w, h = img_size
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise MediaEditError(
                f"{op} box {box} is outside image {(w, h)}")
        mask_img = Image.new("L", img_size, 0)
        ImageDraw.Draw(mask_img).rectangle([l, t, r, b], fill=255)
    else:
        mask_img = _load_aux(mask, "mask").convert("L")
        if mask_img.size != img_size:
            mask_img = mask_img.resize(img_size)
    if mask_img.getbbox() is None:
        raise MediaEditError(f"{op} mask is empty: nothing to do")
    return mask_img


def _inpaint_skimage(arr: Any, mask_arr: Any, np: Any) -> Any:
    from skimage.restoration import inpaint_biharmonic  # noqa: PLC0415

    out = inpaint_biharmonic(arr, mask_arr.astype(bool), channel_axis=2)
    return (out * 255.0).astype(np.uint8)


def inpaint_cv(img: Any, mask: Any, method: str = "telea",
               radius: float = 3.0) -> Any:
    """Inpaint masked regions (reconstruct damaged areas).

    Backends (best first): OpenCV ``inpaint`` (Telea / Navier-Stokes) >
    scikit-image ``inpaint_biharmonic`` > Pillow iterative diffusion
    fill (slower, always works).

    ``mask``: a PIL image, file path, bytes, or an ``"l,t,r,b"`` box
    string; any non-zero pixel is treated as damaged. ``method``:
    ``"telea"`` (fast marching, default) or ``"ns"`` (Navier-Stokes;
    OpenCV backend only — other backends ignore it).
    """
    if method not in ("telea", "ns"):
        raise MediaEditError(
            f"inpaint_cv method must be 'telea' or 'ns', got {method!r}")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if np is None or (cv2 is None and _ski() is None):
        _log.debug("inpaint_cv: Pillow iterative-diffusion fallback")
        return _reattach_alpha(
            _inpaint_pillow(rgb, _mask_pil(rgb.size, mask,
                                           op="inpaint_cv")), alpha)
    arr = _rgb_array(rgb, np)
    mask_arr = _mask_array(rgb.size, mask, np)
    if cv2 is not None:
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        out = cv2.inpaint(arr, mask_arr, float(radius), flags)
    else:
        _log.debug("inpaint_cv: falling back to scikit-image biharmonic")
        out = _inpaint_skimage(arr, mask_arr, np)
    return _reattach_alpha(_to_pil(out, np), alpha)


# ---------------------------------------------------------------------------
# seamless_clone
# ---------------------------------------------------------------------------

def _clone_mask(fg: Any, mask: Any | None, np: Any) -> Any:
    if mask is None:
        if fg.mode in ("RGBA", "LA", "PA"):
            arr = (np.asarray(fg.getchannel("A")) > 0).astype(np.uint8)
        else:
            arr = np.ones(np.asarray(fg.convert("RGB")).shape[:2],
                          dtype=np.uint8)
    else:
        m = _load_aux(mask, "mask").convert("L")
        if m.size != fg.size:
            m = m.resize(fg.size)
        arr = (np.asarray(m) > 0).astype(np.uint8)
    if not arr.any():
        raise MediaEditError("seamless_clone mask is empty")
    return arr * 255


def seamless_clone(img: Any, background: Any, *,
                   position: tuple[int, int] | str = "center",
                   mask: Any | None = None,
                   mix: str = "normal") -> Any:
    """Paste ``img`` onto ``background`` with gradient-domain blending.

    Backends (best first): OpenCV ``seamlessClone`` (Poisson) > Pillow
    feathered paste (always works).

    ``img`` is the foreground patch, ``background`` the canvas (PIL image,
    path, or bytes). ``position`` is the center point ``(x, y)`` where the
    foreground lands, or ``"center"``. ``mask`` optionally limits which
    foreground pixels are cloned (PIL image / path / bytes; non-zero =
    clone); defaults to the full foreground (or its alpha channel when the
    foreground has one). ``mix``: ``"normal"`` or ``"monochrome"``
    (monochrome transfer — OpenCV backend only affects color handling;
    the Pillow path approximates it).
    """
    fg = _load_aux(img, "foreground")
    bg_img = _load_aux(background, "background")
    np = _np()
    if isinstance(position, str):
        if position != "center":
            raise MediaEditError(
                f"seamless_clone position must be 'center' or (x, y), "
                f"got {position!r}")
    else:
        position = (int(position[0]), int(position[1]))
    if mix not in ("normal", "monochrome"):
        raise MediaEditError(
            f"seamless_clone mix must be 'normal' or 'monochrome', "
            f"got {mix!r}")
    cv2 = _cv2()
    if cv2 is not None and np is not None:
        fg_arr = np.asarray(fg.convert("RGB"))
        bg_arr = np.asarray(bg_img.convert("RGB"))
        mask_arr = _clone_mask(fg, mask, np)
        if isinstance(position, str):
            center = (bg_arr.shape[1] // 2, bg_arr.shape[0] // 2)
        else:
            center = position
        flags = (cv2.MONOCHROME_TRANSFER if mix == "monochrome"
                 else cv2.NORMAL_CLONE)
        out = cv2.seamlessClone(fg_arr, bg_arr, mask_arr, center, flags)
        return _to_pil(out, np)
    _log.debug("seamless_clone: Pillow feathered-paste fallback")
    bw, bh = bg_img.size
    center = (bw // 2, bh // 2) if isinstance(position, str) else position
    if mask is not None:
        mask_img = _mask_pil(fg.size, mask, op="seamless_clone")
    elif fg.mode in ("RGBA", "LA", "PA"):
        mask_img = fg.getchannel("A")
    else:
        mask_img = Image.new("L", fg.size, 255)
    return _seamless_clone_pillow(fg, bg_img, center, mask_img, mix)


# ---------------------------------------------------------------------------
# sharpen
# ---------------------------------------------------------------------------

def sharpen_advanced(img: Any, amount: float = 1.0,
                     sigma: float = 1.0) -> Any:
    """Unsharp-mask sharpening.

    Backends (best first): OpenCV (GaussianBlur + addWeighted) >
    scikit-image ``filters.unsharp_mask`` > Pillow ``UnsharpMask``.

    ``amount`` controls sharpening strength (0 = no-op, ~1 = normal,
    >1 = aggressive); ``sigma`` the Gaussian radius of the blurred copy.
    """
    if amount < 0:
        raise MediaEditError("sharpen_advanced amount must be >= 0")
    if sigma <= 0:
        raise MediaEditError("sharpen_advanced sigma must be > 0")
    rgb, alpha = _split_alpha(img)
    cv2, ski, np = _cv2(), _ski(), _np()
    if cv2 is not None and np is not None:
        arr = _rgb_array(rgb, np).astype(np.float32)
        blurred = cv2.GaussianBlur(arr, (0, 0), float(sigma))
        out = cv2.addWeighted(arr, 1.0 + float(amount),
                              blurred, -float(amount), 0)
        out = _to_pil(np.clip(out, 0, 255).astype(np.uint8), np)
    elif ski is not None and np is not None:
        from skimage.filters import unsharp_mask  # noqa: PLC0415

        arr = _rgb_array(rgb, np)
        out = _to_pil(unsharp_mask(arr, radius=float(sigma),
                                   amount=float(amount),
                                   channel_axis=2) * 255.0, np)
    else:
        _log.debug("sharpen_advanced: falling back to Pillow UnsharpMask")
        out = rgb.filter(ImageFilter.UnsharpMask(
            radius=float(sigma), percent=int(float(amount) * 100),
            threshold=0))
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# cartoonize / pencil_sketch
# ---------------------------------------------------------------------------

def _cartoonize_cv2(arr: Any, cv2: Any, downscale_steps: int,
                    bilateral_d: int, edge_block: int,
                    edge_c: float) -> Any:
    h, w = arr.shape[:2]
    small = arr
    for _ in range(max(0, int(downscale_steps))):
        small = cv2.pyrDown(small)
    for _ in range(max(0, int(downscale_steps))):
        small = cv2.bilateralFilter(small, d=int(bilateral_d),
                                    sigmaColor=75, sigmaSpace=75)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    block = max(3, int(edge_block) | 1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, block, float(edge_c))
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(smooth, edges)


def _cartoonize_pillow(img: Any) -> Any:
    """Pillow approximation: median smoothing + posterize + ink edges."""
    smooth = img.filter(ImageFilter.MedianFilter(size=5))
    smooth = ImageOps.posterize(smooth, bits=4)
    edges = (img.convert("L").filter(ImageFilter.FIND_EDGES)
             .point(lambda v: 0 if v > 40 else 255).convert("RGB"))
    return ImageChops.multiply(smooth, edges)


def cartoonize(img: Any, downscale_steps: int = 2,
               bilateral_d: int = 9, edge_block: int = 9,
               edge_c: float = 2.0) -> Any:
    """Cartoon / comic stylization: smoothing + bold ink edges.

    Backends (best first): OpenCV (pyramid bilateral smoothing +
    adaptive-threshold edges) > Pillow (median smoothing + posterize +
    edge multiply).
    """
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        out = _to_pil(_cartoonize_cv2(_rgb_array(rgb, np), cv2,
                                      downscale_steps, bilateral_d,
                                      edge_block, edge_c), np)
    else:
        _log.debug("cartoonize: falling back to Pillow approximation")
        out = _cartoonize_pillow(rgb)
    return _reattach_alpha(out, alpha)


def _sketch_numpy(gray: Any, blurred: Any, np: Any, shade: float) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        sketch = np.where(
            blurred == 0, 255.0,
            np.minimum(float(shade) * gray.astype(np.float32)
                       / np.maximum(blurred, 1), 255.0))
    return sketch.astype(np.uint8)


def _pencil_sketch_pillow(img: Any, blur_sigma: float,
                          shade: float) -> Any:
    """Pillow-only dodge blend: invert + blur + ImageMath dodge."""
    from PIL import ImageMath  # noqa: PLC0415

    gray = img.convert("L")
    inv = ImageOps.invert(gray)
    blurred = inv.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    s = max(1, int(round(shade)))  # ImageMath needs int constants
    dodge = ImageMath.eval(
        "convert(min(a * %d / (256 - b), 255), 'L')" % s,
        a=gray, b=blurred)
    return dodge


def pencil_sketch(img: Any, blur_sigma: float = 21.0,
                  shade: float = 256.0) -> Any:
    """Pencil-sketch stylization. Returns a grayscale (L) sketch.

    Backends (best first): OpenCV Gaussian blur + numpy dodge blend >
    Pillow (GaussianBlur + ImageMath dodge). Both produce the classic
    invert-blur-dodge sketch; OpenCV's blur is faster on large images.
    """
    if blur_sigma <= 0:
        raise MediaEditError("pencil_sketch blur_sigma must be > 0")
    cv2, np = _cv2(), _np()
    if np is not None:
        gray = _gray_array(img, np)
        inv = 255 - gray
        if cv2 is not None:
            blurred = cv2.GaussianBlur(inv, (0, 0), float(blur_sigma))
        else:
            blurred = np.asarray(
                Image.fromarray(inv, mode="L").filter(
                    ImageFilter.GaussianBlur(radius=float(blur_sigma))))
        return _to_pil(_sketch_numpy(gray, blurred, np, shade), np)
    _log.debug("pencil_sketch: falling back to Pillow-only dodge")
    return _pencil_sketch_pillow(img, blur_sigma, shade)


# ---------------------------------------------------------------------------
# perspective_transform
# ---------------------------------------------------------------------------

def _order_corners(pts: Any, np: Any) -> Any:
    """Order 4 points as tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _order_corners_py(pts: list[tuple[float, float]]
                      ) -> list[tuple[float, float]]:
    """Pure-Python tl/tr/br/bl ordering (no numpy)."""
    s = [x + y for x, y in pts]
    d = [y - x for x, y in pts]
    return [pts[s.index(min(s))], pts[d.index(min(d))],
            pts[s.index(max(s))], pts[d.index(max(d))]]


def _detect_document_corners(gray: Any, cv2: Any, np: Any) -> Any:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32),
                                  np)
    raise MediaEditError(
        "perspective_transform could not auto-detect a 4-sided document; "
        "pass explicit corners=[tl, tr, br, bl]")


def _perspective_coeffs(src: Any, dst: Any, np: Any) -> tuple:
    """Solve Pillow PERSPECTIVE coefficients mapping dst -> src."""
    m = []
    for (x_dst, y_dst), (x_src, y_src) in zip(dst, src):
        m.append([x_dst, y_dst, 1, 0, 0, 0,
                  -x_src * x_dst, -x_src * y_dst])
        m.append([0, 0, 0, x_dst, y_dst, 1,
                  -y_src * x_dst, -y_src * y_dst])
    a = np.asarray(m, dtype=np.float64)
    b = np.asarray([p for pt in src for p in pt], dtype=np.float64)
    return tuple(np.linalg.solve(a, b).tolist())


def _perspective_pillow(img: Any, src: Any, out_w: int, out_h: int,
                        np: Any = None) -> Any:
    dst = [(0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1)]
    if np is not None:
        coeffs = _perspective_coeffs([tuple(map(float, p)) for p in src],
                                     dst, np)
    else:
        coeffs = _perspective_coeffs_pillow(
            [tuple(map(float, p)) for p in src], dst)
    return img.transform((out_w, out_h), Image.Transform.PERSPECTIVE,
                         coeffs, resample=Image.Resampling.BICUBIC)


def perspective_transform(img: Any, corners: Any | None = None,
                          width: int | None = None,
                          height: int | None = None,
                          border: tuple[int, int, int] = (0, 0, 0)) -> Any:
    """Perspective correction / document de-warping.

    Backends (best first): OpenCV ``getPerspectiveTransform`` +
    ``warpPerspective`` > Pillow ``Image.transform(PERSPECTIVE)`` with a
    pure-Python coefficient solve (always works). Auto corner detection
    needs OpenCV — pass explicit corners without it.

    ``corners``: four ``(x, y)`` points ``[tl, tr, br, bl]`` or a
    ``"x1,y1;x2,y2;x3,y3;x4,y4"`` string. When omitted, the largest
    4-sided contour is auto-detected (OpenCV backend; fails fast if none
    is found). ``width``/``height`` set the output size; when omitted
    they are derived from the corner geometry.
    """
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if corners is None:
        if cv2 is None or np is None:
            raise MediaEditError(
                "perspective_transform: automatic document corner "
                "detection needs OpenCV; pass explicit corners="
                "[tl, tr, br, bl] instead")
        arr = _rgb_array(rgb, np)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        src = [tuple(map(float, p))
               for p in _detect_document_corners(gray, cv2, np)]
    else:
        if isinstance(corners, str):
            parsed = parse_corners(corners)
            if parsed is None:
                raise MediaEditError(
                    "perspective_transform corners string must be "
                    "'x1,y1;x2,y2;x3,y3;x4,y4'")
            corners = parsed
        try:
            pts = [(float(p[0]), float(p[1])) for p in corners]
        except (TypeError, ValueError, IndexError):
            pts = []
        if len(pts) != 4:
            raise MediaEditError(
                "perspective_transform corners must be 4 (x, y) points "
                "[tl, tr, br, bl]")
        src = _order_corners_py(pts)
    (tl, tr, br, bl) = src

    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    w_top = _dist(tr, tl)
    w_bot = _dist(br, bl)
    h_left = _dist(bl, tl)
    h_right = _dist(br, tr)
    out_w = int(width) if width else max(1, int(max(w_top, w_bot)))
    out_h = int(height) if height else max(1, int(max(h_left, h_right)))
    if cv2 is not None and np is not None:
        arr = _rgb_array(rgb, np)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1],
                        [0, out_h - 1]], dtype=np.float32)
        mat = cv2.getPerspectiveTransform(
            np.asarray(src, dtype=np.float32), dst)
        warped = cv2.warpPerspective(
            arr, mat, (out_w, out_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(c) for c in border))
        out_img = _to_pil(warped, np)
        if alpha is not None:
            warped_a = cv2.warpPerspective(
                np.asarray(alpha), mat, (out_w, out_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out_img = _reattach_alpha(out_img, _to_pil(warped_a, np))
        return out_img
    _log.debug("perspective_transform: falling back to Pillow transform")
    out_img = _perspective_pillow(rgb, src, out_w, out_h, np)
    if alpha is not None:
        out_img = _reattach_alpha(
            out_img, _perspective_pillow(alpha, src, out_w, out_h, np))
    return out_img


# ---------------------------------------------------------------------------
# grabcut_segment (OpenCV is the only free implementation; documented)
# ---------------------------------------------------------------------------

def grabcut_segment(img: Any,
                    rect: tuple[int, int, int, int] | str | None = None,
                    iterations: int = 5,
                    background: str = "transparent") -> Any:
    """GrabCut foreground extraction.

    Backends (best first): OpenCV ``grabCut`` (iterative graph-cut
    segmentation) > Pillow feathered box cutout (treats ``rect`` as the
    subject with soft edges — honest fallback, always works).

    ``rect``: ``(left, top, right, bottom)`` tuple or ``"l,t,r,b"`` string
    known to contain the subject; defaults to a 5% inset of the full
    image. ``background``: ``"transparent"`` (RGBA cutout), ``"blur"``
    (portrait-mode background blur), or ``"white"``. ``iterations`` is
    used by the OpenCV backend only.
    """
    cv2, np = _cv2(), _np()
    if background not in ("transparent", "blur", "white"):
        raise MediaEditError(
            f"grabcut_segment background must be 'transparent', 'blur' or "
            f"'white', got {background!r}")
    if iterations < 1:
        raise MediaEditError("grabcut_segment iterations must be >= 1")
    if isinstance(rect, str):
        parsed = parse_box(rect)
        if parsed is None:
            raise MediaEditError(
                f"grabcut_segment rect string must be 'l,t,r,b', "
                f"got {rect!r}")
        rect = parsed
    rgb, alpha = _split_alpha(img)
    if cv2 is None or np is None:
        _log.debug("grabcut_segment: Pillow feathered box-cutout fallback")
        return _box_cutout(rgb, alpha, rect, background)
    arr = _rgb_array(rgb, np)
    h, w = arr.shape[:2]
    if rect is None:
        rect = (int(w * 0.05), int(h * 0.05),
                int(w * 0.95), int(h * 0.95))
    l, t, r, b = (int(rect[0]), int(rect[1]),
                  int(rect[2]), int(rect[3]))
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise MediaEditError(
            f"grabcut_segment rect {rect} is outside image {w}x{h}")
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(arr, mask, (l, t, r - l, b - t), bgd, fgd,
                int(iterations), cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)
    if not fg_mask.any():
        raise MediaEditError(
            "grabcut_segment found no foreground in the given rect; "
            "try a tighter rect")
    if background == "transparent":
        rgba = np.dstack([arr, fg_mask])
        if alpha is not None:
            combined = np.minimum(fg_mask.astype(np.int32),
                                  np.asarray(alpha).astype(np.int32))
            rgba = np.dstack([arr, combined.astype(np.uint8)])
        return _to_pil(rgba, np)
    bg_fill = (cv2.GaussianBlur(arr, (0, 0), 15) if background == "blur"
               else np.full_like(arr, 255))
    bg_layer = np.where(fg_mask[:, :, None] == 255, arr, bg_fill)
    return _reattach_alpha(_to_pil(bg_layer, np), alpha)


# ---------------------------------------------------------------------------
# rescale (high-quality resampling)
# ---------------------------------------------------------------------------

def _rescale_cv2(arr: Any, cv2: Any, scale: float) -> Any:
    h, w = arr.shape[:2]
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(arr, new, interpolation=interp)


def rescale_ski(img: Any, scale: float = 2.0, order: int = 3,
                anti_aliasing: bool = True) -> Any:
    """High-quality rescaling.

    Backends (best first): scikit-image ``transform.rescale`` (spline
    ``order`` 0-5; order=3 bicubic is the smoothest on gradients) >
    OpenCV ``resize`` (Lanczos4 up / area down) > Pillow ``LANCZOS``.

    ``anti_aliasing`` applies to the scikit-image path.
    """
    if scale <= 0:
        raise MediaEditError("rescale_ski scale must be > 0")
    if not 0 <= order <= 5:
        raise MediaEditError("rescale_ski order must be 0..5")
    rgb, alpha = _split_alpha(img)
    ski, cv2, np = _ski(), _cv2(), _np()
    if ski is not None and np is not None:
        from skimage.transform import rescale  # noqa: PLC0415

        arr = _rgb_array(rgb, np)
        out = _to_pil(rescale(arr, scale, order=int(order),
                              anti_aliasing=bool(anti_aliasing),
                              channel_axis=2, preserve_range=True), np)
        if alpha is not None:
            a_out = _to_pil(rescale(np.asarray(alpha), scale, order=1,
                                    anti_aliasing=bool(anti_aliasing),
                                    preserve_range=True), np)
            out = _reattach_alpha(out, a_out)
        return out
    if cv2 is not None and np is not None:
        _log.debug("rescale_ski: falling back to OpenCV resize")
        arr = _rgb_array(rgb, np)
        out = _to_pil(_rescale_cv2(arr, cv2, scale), np)
        if alpha is not None:
            a = _rescale_cv2(np.asarray(alpha), cv2, scale)
            out = _reattach_alpha(out, _to_pil(a, np))
        return out
    _log.debug("rescale_ski: falling back to Pillow LANCZOS")
    w, h = rgb.size
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    out = rgb.resize(new, resample=Image.Resampling.LANCZOS)
    if alpha is not None:
        out = _reattach_alpha(
            out, alpha.resize(new, resample=Image.Resampling.BILINEAR))
    return out


# keep the backend-neutral alias; the op name is "rescale" either way
rescale_hq = rescale_ski


# ---------------------------------------------------------------------------
# exposure_adjust
# ---------------------------------------------------------------------------

def _gamma_lut(gamma: float) -> list[int]:
    import math  # noqa: PLC0415

    return [min(255, int(round(255 * (i / 255) ** gamma))) for i in range(256)]


def _exposure_cv2(arr: Any, cv2: Any, np: Any, mode: str,
                  gamma: float) -> Any:
    if mode == "adaptive":
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    lut = np.array(_gamma_lut(float(gamma) if mode == "gamma" else 1.0),
                   dtype=np.uint8)
    if mode == "log":
        import math  # noqa: PLC0415

        c = 255.0 / math.log(256)
        lut = np.array([min(255, int(round(c * math.log(1 + i))))
                        for i in range(256)], dtype=np.uint8)
    return cv2.LUT(arr, lut)


def _exposure_pillow(img: Any, mode: str, gamma: float) -> Any:
    if mode == "gamma":
        if gamma <= 0:
            raise MediaEditError("exposure_adjust gamma must be > 0")
        return img.point(_gamma_lut(float(gamma)) * 3)
    if mode == "log":
        return img.point(_log_lut() * 3)
    # adaptive without skimage/cv2: global equalization per channel
    r, g, b = img.split()
    return Image.merge("RGB", (ImageOps.equalize(r), ImageOps.equalize(g),
                               ImageOps.equalize(b)))


def _log_lut() -> list[int]:
    import math  # noqa: PLC0415

    c = 255.0 / math.log(256)
    return [min(255, int(round(c * math.log(1 + i)))) for i in range(256)]


def exposure_adjust(img: Any, mode: str = "adaptive",
                    gamma: float = 1.0) -> Any:
    """Exposure correction.

    Backends (best first): scikit-image (``adjust_gamma`` /
    ``adjust_log`` / ``equalize_adapthist`` CLAHE) > OpenCV (LUT for
    gamma/log, ``CLAHE`` on the L channel for adaptive) > Pillow
    (point LUT for gamma/log, per-channel equalize for adaptive).

    ``mode``: ``"gamma"`` (power-law; ``gamma``<1 brightens shadows,
    >1 darkens), ``"log"`` (logarithmic brightening of dark regions),
    or ``"adaptive"`` (local contrast — the heavy lifter for washed-out
    photos).
    """
    if mode not in ("gamma", "log", "adaptive"):
        raise MediaEditError(
            f"exposure_adjust mode must be 'gamma', 'log' or 'adaptive', "
            f"got {mode!r}")
    if mode == "gamma" and gamma <= 0:
        raise MediaEditError("exposure_adjust gamma must be > 0")
    rgb, alpha = _split_alpha(img)
    ski, cv2, np = _ski(), _cv2(), _np()
    if ski is not None and np is not None:
        from skimage import exposure  # noqa: PLC0415

        arr = _rgb_array(rgb, np).astype(np.float64) / 255.0
        if mode == "gamma":
            out = exposure.adjust_gamma(arr, float(gamma))
        elif mode == "log":
            out = exposure.adjust_log(arr)
        else:
            out = exposure.equalize_adapthist(arr, clip_limit=0.03)
        return _reattach_alpha(_to_pil(out * 255.0, np), alpha)
    if cv2 is not None and np is not None:
        _log.debug("exposure_adjust: falling back to OpenCV")
        out = _to_pil(_exposure_cv2(_rgb_array(rgb, np), cv2, np, mode,
                                    gamma), np)
        return _reattach_alpha(out, alpha)
    _log.debug("exposure_adjust: falling back to Pillow")
    return _reattach_alpha(_exposure_pillow(rgb, mode, gamma), alpha)


# ---------------------------------------------------------------------------
# match_histograms
# ---------------------------------------------------------------------------

def _match_hist_numpy(src: Any, ref: Any, np: Any) -> Any:
    """CDF-based histogram matching per channel (numpy fallback)."""
    out = np.empty_like(src)
    for ch in range(src.shape[2]):
        s = src[..., ch].ravel()
        r = ref[..., ch].ravel()
        s_hist, _ = np.histogram(s, 256, (0, 255))
        r_hist, _ = np.histogram(r, 256, (0, 255))
        s_cdf = s_hist.cumsum().astype(np.float64)
        s_cdf /= s_cdf[-1]
        r_cdf = r_hist.cumsum().astype(np.float64)
        r_cdf /= r_cdf[-1]
        lut = np.searchsorted(r_cdf, s_cdf).astype(np.uint8)
        out[..., ch] = lut[src[..., ch]]
    return out


def match_histograms(img: Any, reference: Any) -> Any:
    """Match ``img``'s color distribution to a reference image.

    Backends (best first): scikit-image ``exposure.match_histograms`` >
    numpy CDF matching > Pillow CDF lookup tables (always works).

    ``reference``: PIL image, file path, or bytes (e.g. a grade still).
    Powerful for consistent looks across a shoot; keeps alpha.
    """
    rgb, alpha = _split_alpha(img)
    ref = _load_aux(reference, "reference")
    ski, np = _ski(), _np()
    if np is None:
        _log.debug("match_histograms: Pillow CDF fallback")
        return _reattach_alpha(_match_hist_pillow(rgb, ref), alpha)
    arr = _rgb_array(rgb, np)
    ref_arr = _rgb_array(ref, np)
    if ski is not None:
        from skimage import exposure  # noqa: PLC0415

        out = exposure.match_histograms(arr, ref_arr, channel_axis=-1)
    else:
        _log.debug("match_histograms: falling back to numpy CDF matching")
        out = _match_hist_numpy(arr, ref_arr, np)
    return _reattach_alpha(_to_pil(out, np), alpha)


# ---------------------------------------------------------------------------
# adaptive_threshold
# ---------------------------------------------------------------------------

def _otsu_numpy(gray: Any, np: Any) -> Any:
    """Otsu's threshold computed with numpy (Pillow-path fallback)."""
    hist, _ = np.histogram(gray.ravel(), 256, (0, 255))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b, best, thresh = 0.0, 0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thresh = between, t
    return (gray > thresh).astype(np.uint8) * 255


def _threshold_cv2(gray: Any, cv2: Any, np: Any, method: str,
                   window: int) -> Any:
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    # sauvola -> gaussian-weighted local mean; niblack -> plain local mean
    kind = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "sauvola" else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(gray, 255, kind, cv2.THRESH_BINARY,
                                 max(3, int(window) | 1), 2)


def adaptive_threshold(img: Any, method: str = "sauvola",
                       window: int = 25, k: float = 0.2) -> Any:
    """Binarize with local (adaptive) thresholding. Returns L image.

    Backends (best first): scikit-image (``threshold_sauvola`` /
    ``threshold_niblack`` / ``threshold_otsu``) > OpenCV
    (``adaptiveThreshold`` / Otsu) > numpy (Otsu from scratch) >
    Pillow (Otsu / Sauvola / Niblack via BoxBlur local stats — always
    works).

    ``method``: ``"sauvola"`` (best for uneven lighting / documents),
    ``"niblack"``, or ``"otsu"`` (global fallback). ``window`` is the
    neighborhood size (odd).
    """
    if method not in ("sauvola", "niblack", "otsu"):
        raise MediaEditError(
            f"adaptive_threshold method must be 'sauvola', 'niblack' or "
            f"'otsu', got {method!r}")
    ski, cv2, np = _ski(), _cv2(), _np()
    if np is None:
        _log.debug("adaptive_threshold: Pillow local-stats fallback")
        gray = img if img.mode == "L" else img.convert("L")
        return _adaptive_threshold_pillow(gray, method, int(window))
    gray = _gray_array(img, np)
    if ski is not None:
        from skimage.filters import (threshold_niblack,  # noqa: PLC0415
                                     threshold_otsu, threshold_sauvola)

        if method == "otsu":
            t = threshold_otsu(gray)
            binary = (gray > t).astype(np.uint8) * 255
        else:
            w = max(3, int(window) | 1)
            fn = (threshold_sauvola if method == "sauvola"
                  else threshold_niblack)
            t = fn(gray, window_size=w, k=float(k))
            binary = (gray > t).astype(np.uint8) * 255
    elif cv2 is not None:
        _log.debug("adaptive_threshold: falling back to OpenCV")
        binary = _threshold_cv2(gray, cv2, np, method, window)
    else:
        _log.debug("adaptive_threshold: falling back to numpy Otsu")
        binary = _otsu_numpy(gray, np)
    return _to_pil(binary, np)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# orb_features — free feature detection (ships in the cv2 binary, no data)
# ---------------------------------------------------------------------------

def _detect_corners(img: Any, n: int, cv2: Any, np: Any
                   ) -> list[tuple[int, int]]:
    """Corner/keypoint locations — OpenCV ORB > numpy Harris >
    Pillow edge-map local maxima. Always works."""
    gray_pil = img if img.mode == "L" else img.convert("L")
    if cv2 is not None and np is not None:
        gray = np.asarray(gray_pil)
        orb = cv2.ORB_create(nfeatures=int(n))
        return [(int(kp.pt[0]), int(kp.pt[1]))
                for kp in orb.detect(gray, None)]
    if np is not None:
        _log.debug("orb: numpy Harris fallback")
        return _harris_corners_numpy(np.asarray(gray_pil), int(n), np)
    _log.debug("orb: Pillow edge-maxima fallback")
    return _harris_corners_pillow(gray_pil, int(n))


def _draw_corners(rgb: Any, points: list[tuple[int, int]],
                  color: tuple[int, int, int] = (255, 0, 0)) -> Any:
    """Draw corner markers with ImageDraw (no OpenCV)."""
    out = rgb.copy()
    d = ImageDraw.Draw(out)
    for x, y in points:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=2)
    return out


def orb_features(img: Any, n: int = 500,
                 draw: bool = True) -> Any:
    """ORB feature detection visualization.

    Backends (best first): OpenCV ``ORB_create`` (patent-free, ships in
    the cv2 binary, no data files) > numpy Harris corner detection >
    Pillow edge-map local maxima (always works).

    Returns the image with the top-``n`` keypoints drawn; keypoint count
    scales with texture in the scene.
    """
    if n < 1:
        raise MediaEditError("orb_features n must be >= 1")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        gray = _gray_array(rgb, np)
        orb = cv2.ORB_create(nfeatures=int(n))
        kps = orb.detect(gray, None)
        if draw:
            out = cv2.drawKeypoints(
                _rgb_array(rgb, np), kps, None, color=(255, 0, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            return _reattach_alpha(_to_pil(out, np), alpha)
        # no-draw: return a black canvas with keypoint dots (countable)
        canvas = np.zeros_like(_rgb_array(rgb, np))
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
        return _to_pil(canvas, np)
    pts = _detect_corners(rgb, n, None, np)
    if draw:
        return _reattach_alpha(_draw_corners(rgb, pts), alpha)
    canvas = Image.new("RGB", rgb.size, (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    for x, y in pts:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
    return canvas


def orb_count(img: Any, n: int = 500) -> int:
    """Number of keypoints/corners detected (programmatic companion).

    Backends (best first): OpenCV ORB > numpy Harris > Pillow
    edge-maxima. Always works.
    """
    if n < 1:
        raise MediaEditError("orb_count n must be >= 1")
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        gray = _gray_array(img, np)
        return len(cv2.ORB_create(nfeatures=int(n)).detect(gray, None))
    return len(_detect_corners(img, n, None, np))


# ---------------------------------------------------------------------------
# kmeans_quantize — color quantization / poster power
# ---------------------------------------------------------------------------

def kmeans_quantize(img: Any, k: int = 8, attempts: int = 3) -> Any:
    """Reduce the image to ``k`` dominant colors (posterization++).

    Backends (best first): OpenCV ``kmeans`` (true k-means in color
    space) > Pillow ``quantize(MEDIANCUT)``.
    """
    if k < 2 or k > 256:
        raise MediaEditError("kmeans_quantize k must be 2..256")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        arr = _rgb_array(rgb, np).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    20, 1.0)
        _, labels, centers = cv2.kmeans(arr, int(k), None, criteria,
                                        int(attempts),
                                        cv2.KMEANS_PP_CENTERS)
        out = centers[labels.flatten()].reshape(
            _rgb_array(rgb, np).shape).astype(np.uint8)
        return _reattach_alpha(_to_pil(out, np), alpha)
    _log.debug("kmeans_quantize: falling back to Pillow MEDIANCUT")
    q = rgb.quantize(colors=int(k), method=Image.Quantize.MEDIANCUT)
    return _reattach_alpha(q.convert("RGB"), alpha)


# ---------------------------------------------------------------------------
# deblur — real deconvolution via scikit-image
# ---------------------------------------------------------------------------

def deblur(img: Any, psf_size: int = 5, method: str = "wiener",
           balance: float = 0.1) -> Any:
    """Deblur via deconvolution with a Gaussian point-spread function.

    Backends (best first): scikit-image ``restoration.wiener`` /
    ``richardson_lucy`` (real deconvolution) > ``sharpen_advanced``
    unsharp masking (the classic no-deps deblur stand-in — always
    works).

    ``psf_size``: assumed blur kernel diameter (odd). ``method``:
    ``"wiener"`` (fast, ``balance`` trades sharpness vs ringing) or
    ``"richardson_lucy"`` (slower, better on heavy blur; ``balance``
    maps to iteration count x10). ``method`` is used by the
    scikit-image backend; the fallback ignores it.
    """
    if method not in ("wiener", "richardson_lucy"):
        raise MediaEditError(
            f"deblur method must be 'wiener' or 'richardson_lucy', "
            f"got {method!r}")
    ski, np = _ski(), _np()
    if ski is None or np is None:
        _log.debug("deblur: unsharp-mask fallback (no scikit-image)")
        return sharpen_advanced(img, amount=1.5, sigma=1.2)
    psf_size = max(3, int(psf_size) | 1)
    rgb, alpha = _split_alpha(img)
    arr = _rgb_array(rgb, np).astype(np.float64) / 255.0
    # gaussian PSF
    ax = np.arange(psf_size) - psf_size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * (psf_size / 3) ** 2))
    psf /= psf.sum()
    if method == "wiener":
        from skimage.restoration import wiener  # noqa: PLC0415

        out = np.stack([wiener(arr[..., c], psf, float(balance))
                        for c in range(3)], axis=-1)
    else:
        from skimage.restoration import richardson_lucy  # noqa: PLC0415

        iters = max(5, int(float(balance) * 100))
        out = np.stack([richardson_lucy(arr[..., c], psf,
                                        num_iter=iters)
                        for c in range(3)], axis=-1)
    return _reattach_alpha(_to_pil(out * 255.0, np), alpha)


# ---------------------------------------------------------------------------
# superpixels — scikit-image SLIC segmentation overlay
# ---------------------------------------------------------------------------

def superpixels(img: Any, n_segments: int = 100, compactness: float = 10.0,
                overlay: bool = True) -> Any:
    """SLIC superpixel segmentation.

    Backends (best first): scikit-image ``segmentation.slic`` (the
    standard) > numpy connected-component labeling of MEDIANCUT color
    regions > uniform grid overlay (honest no-deps baseline — always
    works).

    With ``overlay`` the segment boundaries are drawn in red on the
    image; otherwise each superpixel is filled with its mean color.
    """
    if n_segments < 2:
        raise MediaEditError("superpixels n_segments must be >= 2")
    ski, np = _ski(), _np()
    if ski is None or np is None:
        _log.debug("superpixels: non-SLIC fallback")
        rgb, alpha = _split_alpha(img)
        if np is not None:
            out = _superpixels_cc(rgb, int(n_segments), np,
                                  overlay=overlay, color=(255, 0, 0))
        else:
            out = _superpixels_grid(rgb, int(n_segments),
                                    overlay=overlay, color=(255, 0, 0))
        return _reattach_alpha(out, alpha)
    from skimage.segmentation import mark_boundaries, slic  # noqa: PLC0415

    rgb, alpha = _split_alpha(img)
    arr = _rgb_array(rgb, np)
    labels = slic(arr, n_segments=int(n_segments),
                  compactness=float(compactness), start_label=0,
                  channel_axis=2)
    if overlay:
        out = _to_pil(mark_boundaries(arr / 255.0, labels,
                                      color=(1, 0, 0)) * 255.0, np)
    else:
        means = np.zeros_like(arr)
        for lab in np.unique(labels):
            m = labels == lab
            means[m] = arr[m].mean(axis=0)
        out = _to_pil(means, np)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# white_balance — gray-world / max-RGB (pure Pillow via ImageStat)
# ---------------------------------------------------------------------------

def white_balance(img: Any, method: str = "grayworld") -> Any:
    """Automatic white balance (remove color casts).

    Backends: pure Pillow (``ImageStat`` channel means) — no numpy or
    OpenCV needed, and these classic algorithms need nothing more, so
    Pillow *is* the best free option here.

    ``method``: ``"grayworld"`` (scale channels to equal means) or
    ``"maxwhite"`` (scale so the brightest channel hits 255).
    """
    if method not in ("grayworld", "maxwhite"):
        raise MediaEditError(
            f"white_balance method must be 'grayworld' or 'maxwhite', "
            f"got {method!r}")
    from PIL import ImageStat  # noqa: PLC0415

    rgb, alpha = _split_alpha(img)
    means = ImageStat.Stat(rgb).mean
    if method == "grayworld":
        target = sum(means) / 3 or 1.0
        gains = [target / (m or 1.0) for m in means]
    else:
        peak = max(means) or 1.0
        gains = [255.0 / peak] * 3
    out = Image.merge("RGB", [
        ch.point(lambda v, g=g: min(255, int(v * g)))
        for ch, g in zip(rgb.split(), gains)
    ])
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# auto_levels — Pillow autocontrast (best free option, nothing beats it)
# ---------------------------------------------------------------------------

def auto_levels(img: Any, cutoff: float = 0.5) -> Any:
    """Stretch each channel's histogram (auto levels).

    Backend: Pillow ``ImageOps.autocontrast`` — the best free option;
    ``cutoff`` % of the darkest/lightest pixels are clipped.
    """
    if not 0 <= cutoff < 50:
        raise MediaEditError("auto_levels cutoff must be 0..<50")
    rgb, alpha = _split_alpha(img)
    out = ImageOps.autocontrast(rgb, cutoff=float(cutoff))
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# color_transfer — Reinhard et al. (numpy primary, Pillow LAB fallback)
# ---------------------------------------------------------------------------

def _color_transfer_numpy(arr: Any, ref_arr: Any, np: Any) -> Any:
    cv2 = _cv2()
    src_lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = np.empty_like(src_lab)
    for c in range(3):
        s_mean, s_std = src_lab[..., c].mean(), src_lab[..., c].std() or 1.0
        r_mean, r_std = ref_lab[..., c].mean(), ref_lab[..., c].std()
        out[..., c] = ((src_lab[..., c] - s_mean) * (r_std / s_std)
                       + r_mean)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2RGB)


def _color_transfer_pillow(img: Any, ref: Any) -> Any:
    """Reinhard transfer in Pillow LAB space (no numpy needed)."""
    from PIL import ImageStat  # noqa: PLC0415

    src = img.convert("LAB")
    dst = ref.convert("LAB")
    s_stats = ImageStat.Stat(src)
    d_stats = ImageStat.Stat(dst)
    bands = []
    for i, (ch, sm, ss, dm, ds) in enumerate(zip(
            src.split(), s_stats.mean, s_stats.stddev,
            d_stats.mean, d_stats.stddev)):
        ss = ss or 1.0
        lut = [min(255, max(0, int((v - sm) * (ds / ss) + dm)))
               for v in range(256)]
        bands.append(ch.point(lut))
    return Image.merge("LAB", bands).convert("RGB")


def color_transfer(img: Any, reference: Any) -> Any:
    """Transfer the color mood of ``reference`` onto ``img`` (Reinhard).

    Backends (best first): numpy + OpenCV Lab conversion > pure Pillow
    (LAB mode + per-channel LUT). ``reference``: PIL image, file path,
    or bytes. Keeps alpha.
    """
    rgb, alpha = _split_alpha(img)
    ref = _load_aux(reference, "reference")
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        out = _to_pil(_color_transfer_numpy(_rgb_array(rgb, np),
                                            _rgb_array(ref, np), np), np)
    else:
        _log.debug("color_transfer: falling back to Pillow LAB LUT")
        out = _color_transfer_pillow(rgb, ref)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# panorama — cv2.Stitcher (ships in the binary, no data files)
# ---------------------------------------------------------------------------

def panorama(img: Any, images: list[Any] | None = None,
             mode: str = "panorama") -> Any:
    """Stitch overlapping photos into a panorama.

    Backends (best first): OpenCV ``Stitcher`` (feature-based, handles
    rotation/exposure) > translation-only stitching via FFT phase
    correlation (numpy) or coarse SAD search (pure Pillow) with a
    cross-dissolve blend (always works; best on side-by-side frames).

    ``img`` is the first frame; ``images`` the rest (PIL images, paths,
    or bytes). In op chains this composes like ``stack``/``grid``:
    ``{"op": "panorama", "images": [...]}``. ``mode``: ``"panorama"``
    or ``"scans"`` (OpenCV backend only — the fallback always stitches
    side-by-side).
    """
    if mode not in ("panorama", "scans"):
        raise MediaEditError(
            f"panorama mode must be 'panorama' or 'scans', got {mode!r}")
    frames = [_load_aux(img, "image")]
    for extra in images or []:
        frames.append(_load_aux(extra, "image"))
    if len(frames) < 2:
        raise MediaEditError("panorama needs at least 2 images")
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        _log.debug("panorama: translation-stitching fallback (no OpenCV)")
        return _panorama_pillow(frames, np)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    stitcher = cv2.Stitcher_create(
        cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(arrays)
    if status != cv2.Stitcher_OK:
        raise MediaEditError(
            f"panorama stitching failed (status {status}): images may not "
            f"overlap enough")
    # crop the black borders OpenCV leaves around the panorama
    gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pano = pano[y:y + h, x:x + w]
    return _to_pil(pano, np)


# ---------------------------------------------------------------------------
# retouch_smooth — frequency-separation skin/tone smoothing (pro retouch)
# ---------------------------------------------------------------------------

def retouch_smooth(img: Any, radius: float = 8.0,
                   amount: float = 0.7) -> Any:
    """Texture-preserving smoothing (frequency-separation retouch).

    Splits the image into tone (low frequency) and texture (high
    frequency), smooths only the tone layer, then re-adds texture
    untouched — the classic pro skin/product retouch. Blemishes and
    uneven tone melt away while pores and fabric weave stay crisp.

    Backends (best first): OpenCV ``bilateralFilter`` (edge-aware tone
    smoothing) > numpy frequency separation > Pillow ``GaussianBlur``
    frequency separation via ImageMath (always works).
    """
    if not 0 <= amount <= 1:
        raise MediaEditError("retouch_smooth amount must be 0..1")
    if radius <= 0:
        raise MediaEditError("retouch_smooth radius must be > 0")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if cv2 is not None and np is not None:
        arr = _rgb_array(rgb, np).astype(np.float32)
        d = max(3, int(radius * 3) | 1)
        low = cv2.bilateralFilter(arr, d, radius * 9, radius * 3)
        high = arr - low
        low_s = cv2.GaussianBlur(low, (0, 0), radius * 0.75)
        low_out = low * (1 - amount) + low_s * amount
        out = _to_pil(np.clip(low_out + high, 0, 255).astype(np.uint8),
                      np)
    elif np is not None:
        _log.debug("retouch_smooth: numpy GaussianBlur fallback")
        arr = _rgb_array(rgb, np).astype(np.float32)
        low = np.asarray(rgb.filter(
            ImageFilter.GaussianBlur(radius=float(radius)))).astype(
            np.float32)
        high = arr - low
        low_s = np.asarray(Image.fromarray(
            low.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=float(radius) * 0.75))).astype(
            np.float32)
        low_out = low * (1 - amount) + low_s * amount
        out = _to_pil(np.clip(low_out + high, 0, 255).astype(np.uint8),
                      np)
    else:
        _log.debug("retouch_smooth: Pillow ImageMath fallback")
        out = _retouch_pillow(rgb, float(radius), float(amount))
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# tone_map — HDR-style local tone mapping (cv2 photo module)
# ---------------------------------------------------------------------------

def tone_map(img: Any, method: str = "mantiuk",
             saturation: float = 1.0) -> Any:
    """HDR-style local tone mapping for dramatic light.

    Backends (best first): OpenCV ``createTonemapMantiuk/Drago/Reinhard``
    (photo module, ships in the cv2 binary) > Pillow large-radius
    clarity + soft highlight rolloff (the LDR essence of tone mapping —
    always works).

    ``method``: ``"mantiuk"`` (contrast-preserving, most natural),
    ``"drago"`` (punchy), ``"reinhard"`` (soft photographic). ``method``
    and ``saturation`` are used by the OpenCV backend; the Pillow
    fallback ignores them.
    """
    if method not in ("mantiuk", "drago", "reinhard"):
        raise MediaEditError(
            f"tone_map method must be 'mantiuk', 'drago' or 'reinhard', "
            f"got {method!r}")
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        _log.debug("tone_map: Pillow clarity-based fallback")
        rgb, alpha = _split_alpha(img)
        return _reattach_alpha(_tonemap_pillow(rgb), alpha)
    rgb, alpha = _split_alpha(img)
    arr = _rgb_array(rgb, np).astype(np.float32) / 255.0
    if method == "mantiuk":
        tm = cv2.createTonemapMantiuk(saturation=float(saturation))
    elif method == "drago":
        tm = cv2.createTonemapDrago(saturation=float(saturation))
    else:
        tm = cv2.createTonemapReinhard()
    out = np.clip(tm.process(arr) * 255.0, 0, 255).astype(np.uint8)
    return _reattach_alpha(_to_pil(out, np), alpha)


# ---------------------------------------------------------------------------
# detail_enhance / stylize — cv2 photo module stylization
# ---------------------------------------------------------------------------

def detail_enhance(img: Any, sigma_s: float = 10.0,
                   sigma_r: float = 0.15) -> Any:
    """Edge-preserving detail enhancement (photo module).

    Backends (best first): OpenCV ``detailEnhance`` > ``sharpen_advanced``
    unsharp masking (always works). Makes texture pop without halos.

    ``sigma_s``/``sigma_r`` are used by the OpenCV backend; the
    fallback maps them to an unsharp-mask strength.
    """
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        _log.debug("detail_enhance: unsharp-mask fallback")
        return sharpen_advanced(img, amount=1.2, sigma=1.0)
    rgb, alpha = _split_alpha(img)
    out = cv2.detailEnhance(_rgb_array(rgb, np), sigma_s=float(sigma_s),
                            sigma_r=float(sigma_r))
    return _reattach_alpha(_to_pil(out, np), alpha)


def stylize(img: Any, sigma_s: float = 60.0,
            sigma_r: float = 0.6) -> Any:
    """Watercolor-style non-photorealistic rendering.

    Backends (best first): OpenCV ``stylization`` > Pillow watercolor
    approximation (median smoothing + posterization + soft dark edges —
    always works). Distinct from ``cartoonize``: soft painted look
    instead of ink edges.

    ``sigma_s``/``sigma_r`` are used by the OpenCV backend; the
    fallback ignores them.
    """
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        _log.debug("stylize: Pillow watercolor fallback")
        rgb, alpha = _split_alpha(img)
        return _reattach_alpha(_stylize_pillow(rgb), alpha)
    rgb, alpha = _split_alpha(img)
    out = cv2.stylization(_rgb_array(rgb, np), sigma_s=float(sigma_s),
                          sigma_r=float(sigma_r))
    return _reattach_alpha(_to_pil(out, np), alpha)


# ---------------------------------------------------------------------------
# clarity — Lightroom-style local contrast
# ---------------------------------------------------------------------------

def clarity(img: Any, amount: float = 0.5, radius: float = 24.0) -> Any:
    """Local-contrast punch (Lightroom "clarity").

    Boosts midtone contrast via a wide-radius unsharp mask: detail with
    ``radius`` is extracted and re-added scaled by ``amount``
    (-1..1; negative values soften).

    Backends (best first): OpenCV bilateral base > numpy GaussianBlur
    base > Pillow ImageMath local contrast (always works).
    """
    if not -1 <= amount <= 1:
        raise MediaEditError("clarity amount must be -1..1")
    if radius <= 0:
        raise MediaEditError("clarity radius must be > 0")
    rgb, alpha = _split_alpha(img)
    cv2, np = _cv2(), _np()
    if np is None:
        _log.debug("clarity: Pillow ImageMath fallback")
        return _reattach_alpha(
            _clarity_pillow(rgb, float(amount), float(radius)), alpha)
    arr = _rgb_array(rgb, np).astype(np.float32)
    if cv2 is not None:
        d = max(3, int(radius) | 1)
        base = cv2.bilateralFilter(arr, d, radius * 3, radius)
    else:
        _log.debug("clarity: falling back to Pillow GaussianBlur base")
        base = np.asarray(rgb.filter(
            ImageFilter.GaussianBlur(radius=float(radius)))).astype(
            np.float32)
    out = np.clip(arr + float(amount) * (arr - base), 0, 255).astype(
        np.uint8)
    return _reattach_alpha(_to_pil(out, np), alpha)


# ---------------------------------------------------------------------------
# shadow_highlight — recover shadows / tame highlights
# ---------------------------------------------------------------------------

def _shadow_highlight_numpy(arr: Any, np: Any, shadows: float,
                            highlights: float) -> Any:
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
           + 0.114 * arr[..., 2]) / 255.0
    s_gain = 1.0 + float(shadows) * 2.0 * (1 - lum) ** 2
    h_gain = 1.0 - float(highlights) * 0.8 * lum ** 2
    return np.clip(arr * s_gain[..., None] * h_gain[..., None],
                   0, 255).astype(np.uint8)


def _shadow_highlight_pillow(img: Any, shadows: float,
                             highlights: float) -> Any:
    lum = img.convert("L")
    out = img
    if shadows:
        lift = img.point(lambda v: min(255, int(v * (1 + shadows))))
        mask = lum.point(lambda v: int(255 * (1 - v / 255) ** 2))
        out = Image.composite(lift, out, mask)
    if highlights:
        comp = img.point(lambda v: int(v * (1 - highlights * 0.5)))
        mask = lum.point(lambda v: int(255 * (v / 255) ** 2))
        out = Image.composite(comp, out, mask)
    return out


def shadow_highlight(img: Any, shadows: float = 0.3,
                     highlights: float = 0.3) -> Any:
    """Lift crushed shadows and recover blown highlights.

    Backends (best first): numpy luminance-weighted gain maps > Pillow
    (masked composite). ``shadows``/``highlights`` in 0..1.
    """
    if not 0 <= shadows <= 1 or not 0 <= highlights <= 1:
        raise MediaEditError(
            "shadow_highlight shadows/highlights must be 0..1")
    rgb, alpha = _split_alpha(img)
    np = _np()
    if np is not None:
        out = _to_pil(_shadow_highlight_numpy(_rgb_array(rgb, np), np,
                                               shadows, highlights), np)
    else:
        _log.debug("shadow_highlight: falling back to Pillow composite")
        out = _shadow_highlight_pillow(rgb, shadows, highlights)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# dehaze — dark-channel-prior haze removal (He et al.)
# ---------------------------------------------------------------------------

def _min_filter(arr: Any, size: int, cv2: Any | None, np: Any) -> Any:
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return cv2.erode(arr, k)
    from numpy.lib.stride_tricks import sliding_window_view  # noqa: PLC0415

    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    return sliding_window_view(padded, (size, size)).min(axis=(-2, -1))


def dehaze(img: Any, omega: float = 0.95, t0: float = 0.1) -> Any:
    """Remove haze/fog via the dark channel prior.

    Backends (best first): numpy (+ OpenCV erode when available for the
    min filter) > pure Pillow (MinFilter dark channel, ImageStat
    atmospheric light, ImageMath recovery — always works).

    Estimates atmospheric light from the haziest pixels, inverts the
    haze formation model per pixel. Dramatic on landscapes, foggy
    streets, underwater shots.
    """
    if not 0 < omega <= 1:
        raise MediaEditError("dehaze omega must be 0<omega<=1")
    np = _np()
    if np is None:
        _log.debug("dehaze: Pillow dark-channel fallback")
        rgb, alpha = _split_alpha(img)
        return _reattach_alpha(
            _dehaze_pillow(rgb, float(omega), float(t0)), alpha)
    rgb, alpha = _split_alpha(img)
    arr = _rgb_array(rgb, np).astype(np.float64) / 255.0
    dark = _min_filter(arr.min(axis=2), 15, _cv2(), np)
    # atmospheric light: max RGB among the 0.1% haziest pixels
    flat_dark = dark.ravel()
    k = max(1, int(flat_dark.size * 0.001))
    idx = np.argpartition(flat_dark, -k)[-k:]
    a = arr.reshape(-1, 3)[idx].max(axis=0)
    a = np.maximum(a, 1e-3)
    t = 1.0 - float(omega) * dark / a.max()
    t = np.maximum(t, float(t0))[..., None]
    out = np.clip((arr - a) / t + a, 0, 1)
    return _reattach_alpha(_to_pil(out * 255.0, np), alpha)


# ---------------------------------------------------------------------------
# seam_carve — content-aware resizing
# ---------------------------------------------------------------------------

def _seam_energy(gray: Any, np: Any) -> Any:
    gy, gx = np.gradient(gray.astype(np.float64))
    return np.abs(gx) + np.abs(gy)


def _find_vertical_seam(energy: Any, np: Any) -> Any:
    h, w = energy.shape
    cost = energy.copy()
    back = np.zeros((h, w), dtype=np.int8)
    for y in range(1, h):
        prev = cost[y - 1]
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        left[0], right[-1] = np.inf, np.inf
        pick = np.argmin(np.stack([left, prev, right]), axis=0) - 1
        back[y] = pick
        cost[y] += np.choose(pick + 1, [left, prev, right])
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = seam[y + 1] + int(back[y + 1, seam[y + 1]])
        seam[y] = min(max(seam[y], 0), w - 1)
    return seam


def _remove_vertical_seam(arr: Any, seam: Any, np: Any) -> Any:
    h, w = arr.shape[:2]
    keep = np.ones((h, w), dtype=bool)
    keep[np.arange(h), seam] = False
    if arr.ndim == 2:
        return arr[keep].reshape(h, w - 1)
    return arr[keep].reshape(h, w - 1, arr.shape[2])


def _seam_carve_to(arr: Any, target_w: int, np: Any) -> Any:
    out = arr
    while out.shape[1] > target_w:
        gray = out.mean(axis=2) if out.ndim == 3 else out
        seam = _find_vertical_seam(_seam_energy(gray, np), np)
        out = _remove_vertical_seam(out, seam, np)
    return out


def seam_carve(img: Any, width: int | None = None,
               height: int | None = None) -> Any:
    """Content-aware resizing (seam carving).

    Removes low-energy seams so subjects survive aggressive crops that
    would squash them with plain scaling. Only shrinking is supported
    (seam insertion is a research project, not a feature).

    Backends (best first): numpy (vectorized dynamic-programming seams)
    > pure-Python dynamic programming (slower, exact — always works).
    Pillow/OpenCV have no equivalent.
    """
    rgb, alpha = _split_alpha(img)
    w, h = rgb.size
    tw = int(width) if width else w
    th = int(height) if height else h
    if tw > w or th > h:
        raise MediaEditError(
            "seam_carve only shrinks; "
            f"got target {(tw, th)} from {(w, h)}")
    if tw < 1 or th < 1:
        raise MediaEditError("seam_carve target size must be >= 1")
    np = _np()
    if np is None:
        _log.debug("seam_carve: pure-Python DP fallback")
        out = _seam_carve_python(rgb, tw, th)
        if alpha is not None:
            alpha = _seam_carve_python(
                alpha.convert("L"), tw, th).convert("L")
        return _reattach_alpha(out, alpha)
    arr = _rgb_array(rgb, np)
    a_arr = np.asarray(alpha) if alpha is not None else None
    if tw < w:
        arr = _seam_carve_to(arr, tw, np)
        if a_arr is not None:
            a_arr = _seam_carve_to(a_arr, tw, np)
    if th < h:
        arr = _seam_carve_to(np.transpose(arr, (1, 0, 2)), th, np)
        arr = np.transpose(arr, (1, 0, 2))
        if a_arr is not None:
            a_arr = np.transpose(
                _seam_carve_to(np.transpose(a_arr, (1, 0)), th, np),
                (1, 0))
    out = _to_pil(arr, np)
    return _reattach_alpha(out, _to_pil(a_arr, np) if a_arr is not None
                           else None)


# ---------------------------------------------------------------------------
# tilt_shift — miniature-faking selective blur
# ---------------------------------------------------------------------------

def tilt_shift(img: Any, focus_center: float = 0.5,
               focus_size: float = 0.25, blur: float = 15.0,
               angle: float = 0.0) -> Any:
    """Tilt-shift miniature effect: sharp band, blurred surroundings.

    Backend: pure Pillow (gradient mask + GaussianBlur + composite) —
    nothing else needed, so Pillow *is* the best free option.
    ``focus_center``/``focus_size`` are fractions of image height;
    ``angle`` rotates the focus band (degrees).
    """
    if not 0 <= focus_center <= 1:
        raise MediaEditError("tilt_shift focus_center must be 0..1")
    if not 0 < focus_size <= 1:
        raise MediaEditError("tilt_shift focus_size must be 0<..<=1")
    rgb, alpha = _split_alpha(img)
    w, h = rgb.size
    band = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(band)
    c = int(h * focus_center)
    half = int(h * focus_size / 2)
    d.rectangle([0, c - half, w, c + half], fill=255)
    if angle:
        band = band.rotate(angle, resample=Image.Resampling.BICUBIC,
                           expand=False)
    mask = band.filter(ImageFilter.GaussianBlur(
        radius=max(1.0, h * 0.08)))
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    out = Image.composite(rgb, blurred, mask)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# selective_color — keep one hue, drop the rest
# ---------------------------------------------------------------------------

def selective_color(img: Any, hue: float = 0.0, hue_width: float = 30.0,
                    keep_saturation: float = 1.0) -> Any:
    """Selective color: one hue stays vivid, everything else goes mono.

    Backend: pure Pillow (HSV mode + smooth hue mask) — the best free
    option needs nothing more. ``hue``/``hue_width`` in degrees.
    """
    if not 0 <= hue < 360:
        raise MediaEditError("selective_color hue must be 0..<360")
    if not 0 < hue_width <= 180:
        raise MediaEditError("selective_color hue_width must be 0<..<=180")
    rgb, alpha = _split_alpha(img)
    hsv = rgb.convert("HSV")
    h_band = hsv.getchannel("H")
    center = (hue / 360.0) * 255.0
    half = (hue_width / 360.0) * 255.0

    def _dist_lut() -> list[int]:
        lut = []
        for v in range(256):
            d = abs(v - center)
            d = min(d, 255 - d)
            keep = max(0.0, 1.0 - d / half) if half else 1.0
            # smoothstep edges
            keep = keep * keep * (3 - 2 * keep)
            lut.append(int(keep * 255))
        return lut

    mask = h_band.point(_dist_lut())
    if keep_saturation != 1.0:
        from PIL import ImageEnhance  # noqa: PLC0415

        rgb = ImageEnhance.Color(rgb).enhance(float(keep_saturation))
    gray = rgb.convert("L").convert("RGB")
    out = Image.composite(rgb, gray, mask)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# split_tone — tint shadows and highlights independently
# ---------------------------------------------------------------------------

def _split_tone_numpy(arr: Any, np: Any, shadows: tuple, highlights: tuple,
                      strength: float) -> Any:
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
           + 0.114 * arr[..., 2]) / 255.0
    s_amt = ((1 - lum) * strength)[..., None]
    h_amt = (lum * strength)[..., None]
    s_col = np.asarray(shadows, dtype=np.float64)
    h_col = np.asarray(highlights, dtype=np.float64)
    out = (arr.astype(np.float64) * (1 - s_amt - h_amt)
           + s_col * s_amt + h_col * h_amt)
    return np.clip(out, 0, 255).astype(np.uint8)


def _split_tone_pillow(img: Any, shadows: tuple, highlights: tuple,
                       strength: float) -> Any:
    lum = img.convert("L")
    out = img
    s_solid = Image.new("RGB", img.size, tuple(int(c) for c in shadows))
    h_solid = Image.new("RGB", img.size, tuple(int(c) for c in highlights))
    s_mask = lum.point(lambda v: int(255 * (1 - v / 255) * strength))
    h_mask = lum.point(lambda v: int(255 * (v / 255) * strength))
    out = Image.composite(Image.blend(out, s_solid, 0.5), out, s_mask)
    out = Image.composite(Image.blend(out, h_solid, 0.5), out, h_mask)
    return out


def split_tone(img: Any, shadows: tuple[int, int, int] = (30, 60, 120),
               highlights: tuple[int, int, int] = (200, 180, 140),
               strength: float = 0.5) -> Any:
    """Split toning: tint shadows one color, highlights another.

    Backends (best first): numpy luminance-weighted blend > Pillow
    masked composite. The teal-and-orange blockbuster look lives here.
    """
    if not 0 <= strength <= 1:
        raise MediaEditError("split_tone strength must be 0..1")
    rgb, alpha = _split_alpha(img)
    np = _np()
    if np is not None:
        out = _to_pil(_split_tone_numpy(_rgb_array(rgb, np), np, shadows,
                                        highlights, strength), np)
    else:
        _log.debug("split_tone: falling back to Pillow composite")
        out = _split_tone_pillow(rgb, shadows, highlights, strength)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# apply_curves — pro tonal curves
# ---------------------------------------------------------------------------

def _curves_lut(points: list[tuple[int, int]]) -> list[int]:
    pts = sorted(points)
    if pts[0][0] != 0:
        pts = [(0, pts[0][1])] + pts
    if pts[-1][0] != 255:
        pts = pts + [(255, pts[-1][1])]
    lut: list[int] = []
    seg = 0
    for v in range(256):
        while seg < len(pts) - 2 and v > pts[seg + 1][0]:
            seg += 1
        x0, y0 = pts[seg]
        x1, y1 = pts[seg + 1]
        t = (v - x0) / (x1 - x0) if x1 != x0 else 0.0
        lut.append(min(255, max(0, int(round(y0 + t * (y1 - y0))))))
    return lut


def apply_curves(img: Any, points: list[tuple[int, int]] | None = None,
                 channel: str = "rgb") -> Any:
    """Apply a tonal curve (Lightroom/Photoshop curves).

    Backend: pure Pillow point LUT — the best free option needs nothing
    more. ``points``: ``[(in, out), ...]`` in 0..255; ``channel``:
    ``"rgb"`` or ``"r"``/``"g"``/``"b"``.
    """
    if channel not in ("rgb", "r", "g", "b"):
        raise MediaEditError(
            f"apply_curves channel must be rgb/r/g/b, got {channel!r}")
    pts = points if points else [(0, 0), (255, 255)]
    for p in pts:
        if (len(p) != 2 or not all(isinstance(v, (int, float)) for v in p)
                or not (0 <= p[0] <= 255 and 0 <= p[1] <= 255)):
            raise MediaEditError(
                f"apply_curves points must be (in, out) in 0..255, "
                f"got {p!r}")
    rgb, alpha = _split_alpha(img)
    lut = _curves_lut([(int(x), int(y)) for x, y in pts])
    if channel == "rgb":
        out = rgb.point(lut * 3)
    else:
        idx = {"r": 0, "g": 1, "b": 2}[channel]
        bands = list(rgb.split())
        bands[idx] = bands[idx].point(lut)
        out = Image.merge("RGB", bands)
    return _reattach_alpha(out, alpha)


# ---------------------------------------------------------------------------
# apply_cube_lut — professional .cube LUT files
# ---------------------------------------------------------------------------

def _parse_cube(source: Any) -> tuple[int, Any, Any]:
    """Parse a .cube LUT file -> (size, lut float array, domain).

    Parsing itself is pure-Python (see :func:`_parse_cube_lists`); only
    the returned array needs numpy.
    """
    np = _np()
    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        text = Path(source).read_text(encoding="utf-8",
                                      errors="replace")
    elif isinstance(source, (bytes, bytearray)):
        text = bytes(source).decode("utf-8", errors="replace")
    elif isinstance(source, str):
        text = source  # raw .cube text
    else:
        raise MediaEditError(
            "apply_cube_lut lut must be a .cube path, bytes, or text")
    size, _title, lut, dmin, dmax = _parse_cube_lists(text)
    return size, np.asarray(lut, dtype=np.float64), (dmin, dmax)


def _apply_trilinear(arr: Any, lut: Any, size: int,
                     domain: Any, np: Any) -> Any:
    (dmin, dmax) = domain
    dmin = np.asarray(dmin)
    scale = (size - 1) / (np.asarray(dmax) - dmin)
    # .cube red axis varies fastest
    coords = np.clip((arr - dmin) * scale, 0, size - 1)
    b = coords[..., 2]
    g = coords[..., 1]
    r = coords[..., 0]
    b0 = np.floor(b).astype(int)
    g0 = np.floor(g).astype(int)
    r0 = np.floor(r).astype(int)
    b1 = np.minimum(b0 + 1, size - 1)
    g1 = np.minimum(g0 + 1, size - 1)
    r1 = np.minimum(r0 + 1, size - 1)
    fb, fg, fr = b - b0, g - g0, r - r0
    fb, fg, fr = fb[..., None], fg[..., None], fr[..., None]
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]
    out = (c000 * (1 - fr) * (1 - fg) * (1 - fb)
           + c001 * fr * (1 - fg) * (1 - fb)
           + c010 * (1 - fr) * fg * (1 - fb)
           + c011 * fr * fg * (1 - fb)
           + c100 * (1 - fr) * (1 - fg) * fb
           + c101 * fr * (1 - fg) * fb
           + c110 * (1 - fr) * fg * fb
           + c111 * fr * fg * fb)
    return out


def apply_cube_lut(img: Any, lut: Any) -> Any:
    """Apply a professional ``.cube`` 3D LUT (photographer looks).

    ``lut``: path to a .cube file, raw .cube text, or bytes. Trilinear
    interpolation in the LUT cube.

    Backends (best first): numpy (vectorized trilinear) > pure-Python
    trilinear (slower, exact). Always works.
    """
    np = _np()
    rgb, alpha = _split_alpha(img)
    if np is None:
        if isinstance(lut, (str, os.PathLike)) and os.path.exists(lut):
            text = Path(lut).read_text(encoding="utf-8",
                                       errors="replace")
        elif isinstance(lut, (bytes, bytearray)):
            text = bytes(lut).decode("utf-8", errors="replace")
        elif isinstance(lut, str):
            text = lut
        else:
            raise MediaEditError(
                "apply_cube_lut lut must be a .cube path, bytes, or text")
        size, _title, lut_l, dmin, dmax = _parse_cube_lists(text)
        _log.debug("apply_cube_lut: pure-Python trilinear fallback")
        return _reattach_alpha(
            _cube_lut_python(rgb, size, lut_l, dmin, dmax), alpha)
    size, lut_arr, domain = _parse_cube(lut)
    arr = _rgb_array(rgb, np).astype(np.float64) / 255.0
    out = _apply_trilinear(arr, lut_arr, size, domain, np)
    return _reattach_alpha(_to_pil(out * 255.0, np), alpha)


# ---------------------------------------------------------------------------
# replace_background — grabcut subject onto a new backdrop
# ---------------------------------------------------------------------------

def replace_background(img: Any, background: Any,
                       rect: tuple[int, int, int, int] | str | None = None,
                       feather: float = 2.0) -> Any:
    """Cut the subject out (GrabCut) and drop it on a new background.

    ``background``: PIL image, path, or bytes (resized to the subject
    image). ``rect``: subject box (tuple or ``"l,t,r,b"``; auto inset
    when omitted). ``feather``: edge softening px.

    Backend: ``grabcut_segment`` (OpenCV GrabCut > Pillow feathered box
    cutout) + Pillow composite; always works.
    """
    cutout = grabcut_segment(img, rect=rect, background="transparent")
    bg_img = _load_aux(background, "background").convert("RGB")
    if bg_img.size != cutout.size:
        bg_img = bg_img.resize(cutout.size,
                               resample=Image.Resampling.LANCZOS)
    mask = cutout.getchannel("A")
    if feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    out = bg_img.copy()
    out.paste(cutout.convert("RGB"), (0, 0), mask)
    return out


# =====================================================================
# Pure-Pillow / stdlib fallback implementations.
#
# Standing rule: every op works with ZERO third-party dependencies.
# OpenCV / scikit-image / numpy are the best free primaries, but each
# helper below is a real, working implementation on top of Pillow and
# the standard library — slower or less accurate than the primaries,
# never a stub.  Each op's docstring documents its backend preference
# order; use :func:`backend_status` to see what is active here.
# =====================================================================

def _box_cutout(img, alpha, rect, background):
    """Feathered rectangular cutout — grabcut fallback without OpenCV.

    Not a real segmentation: treats ``rect`` as the subject and feathers
    its edges.  Honest and useful for simple centered subjects.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    if rect is None:
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
    l, t, r, b = (int(v) for v in rect)
    l, r = max(0, min(l, w - 1)), max(0, min(r, w))
    t, b = max(0, min(t, h - 1)), max(0, min(b, h))
    if not (l < r and t < b):
        raise MediaEditError(f"grabcut_segment: invalid rect {rect!r}")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([l, t, r, b], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))
    if background == "transparent":
        out = rgb.convert("RGBA")
        out.putalpha(mask)
        return out
    if background == "blur":
        bg_img = rgb.filter(ImageFilter.GaussianBlur(21))
    else:
        bg_img = Image.new("RGB", (w, h), (255, 255, 255))
    out = bg_img.copy()
    out.paste(rgb, (0, 0), mask)
    return out


def _inpaint_pillow(img, mask):
    """Iterative onion-peel diffusion inpaint — no OpenCV / numpy needed.

    Each pass fills the 1px frontier of the damaged area from a blurred
    copy of the working image, then grows the known region.  Converges in
    roughly ``max(damage radius)`` passes; slow on huge masks but correct.
    """
    rgb = img.convert("RGB")
    dmg = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    if dmg.getbbox() is None:
        return rgb
    known = ImageChops.invert(dmg)
    work = rgb.copy()
    for _ in range(max(rgb.size)):
        grown = known.filter(ImageFilter.MaxFilter(3))
        frontier = ImageChops.darker(grown, dmg)
        if frontier.getbbox() is None:
            break
        fill = work.filter(ImageFilter.GaussianBlur(1.5))
        work.paste(fill, mask=frontier)
        known = ImageChops.lighter(known, frontier)
    return work


def _otsu_threshold(gray):
    """Otsu's threshold from a Pillow histogram — pure stdlib."""
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 128
    sum_all = sum(i * c for i, c in enumerate(hist))
    sum_b = 0
    w_b = 0
    best = -1.0
    thresh = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thresh = between, t
    return thresh


def _local_threshold_map_pillow(gray, method, block_size, c):
    """Sauvola / Niblack threshold maps with pure Pillow.

    Local mean comes from ``BoxBlur``; local variance needs E[x^2], which
    is computed exactly by splitting each pixel into high/low nibbles
    (hi^2, hi*lo, lo^2 all fit in L so they can be box-blurred, then
    recombined as 256*E[hi^2] + 32*E[hi*lo] + E[lo^2]).
    """
    window = block_size if block_size % 2 else block_size + 1
    rad = window // 2
    mean = gray.filter(ImageFilter.BoxBlur(rad))
    hi = gray.point([(i >> 4) ** 2 for i in range(256)])
    lo = gray.point([(i & 15) ** 2 for i in range(256)])
    hl = ImageMath.eval(
        "convert(a * b, 'L')",
        a=gray.point([i >> 4 for i in range(256)]),
        b=gray.point([i & 15 for i in range(256)]),
    )
    m_hh = hi.filter(ImageFilter.BoxBlur(rad))
    m_ll = lo.filter(ImageFilter.BoxBlur(rad))
    m_hl = hl.filter(ImageFilter.BoxBlur(rad))
    meansq = ImageMath.eval("a * 256 + b * 32 + c", a=m_hh, b=m_hl, c=m_ll)
    var = ImageMath.eval("max(a - b * b, 0)", a=meansq, b=mean)
    var8 = ImageMath.eval("convert(a / 256, 'L')", a=var)
    std = var8.point([int((i * 256) ** 0.5) for i in range(256)])
    if method == "sauvola":
        # t = mean * (1 + k*(std/R - 1)), k=0.2, R=128
        #   = mean * (0.8 + std/640) = mean * (512 + std) / 640
        tmap = ImageMath.eval(
            "convert(min(max(a * (512 + b) / 640, 0), 255), 'L')",
            a=mean, b=std,
        )
    else:  # niblack, k=-0.2 -> t = mean - std/5
        tmap = ImageMath.eval(
            "convert(min(max(a - b / 5, 0), 255), 'L')", a=mean, b=std
        )
    if c:
        tmap = ImageMath.eval(
            "convert(min(max(a - C, 0), 255), 'L')", a=tmap, C=int(c)
        )
    return ImageMath.eval("convert((g > t) * 255, 'L')", g=gray, t=tmap)


def _adaptive_threshold_pillow(gray, method, block_size, c=2):
    """Adaptive thresholding on Pillow alone (Otsu / Sauvola / Niblack)."""
    if method == "otsu":
        t = _otsu_threshold(gray)
        return gray.point([0] * (t + 1) + [255] * (255 - t))
    return _local_threshold_map_pillow(gray, method, block_size, c)


def _harris_corners_pillow(gray, n):
    """Corner detection on Pillow alone: local maxima of the edge map,
    greedily thinned so no two corners sit within 5px."""
    w, h = gray.size
    edge = gray.filter(ImageFilter.FIND_EDGES)
    mx = edge.filter(ImageFilter.MaxFilter(7))
    peaks = ImageChops.difference(edge, mx).point(
        lambda v: 255 if v == 0 else 0
    )
    strong = edge.point(lambda v: 255 if v > 48 else 0)
    cand = ImageChops.darker(peaks, strong)
    cdata = cand.getdata()
    edata = edge.getdata()
    scored = [
        (edata[i], i % w, i // w) for i in range(w * h) if cdata[i]
    ]
    scored.sort(key=lambda s: s[0], reverse=True)
    kept = []
    for _, x, y in scored:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= 25 for kx, ky in kept):
            kept.append((x, y))
            if len(kept) >= n:
                break
    return kept


def _box_blur_np(a, k, np):
    """Box blur via integral image — pure numpy, no scipy needed."""
    p = k // 2
    P = np.pad(a, p)
    C = np.zeros((P.shape[0] + 1, P.shape[1] + 1), dtype=np.float64)
    C[1:, 1:] = np.cumsum(np.cumsum(P, axis=0), axis=1)
    H, W = a.shape
    # sum of P[i:i+k, j:j+k] for i in [0,H), j in [0,W)
    return (
        C[k:H + k, k:W + k] - C[0:H, k:W + k]
        - C[k:H + k, 0:W] + C[0:H, 0:W]
    ) / (k * k)


def _harris_corners_numpy(gray_arr, n, np):
    """Harris corner response with the structure tensor smoothed by an
    integral-image box blur; non-maximum suppression over 3x3."""
    g = np.asarray(gray_arr, dtype=np.float64)
    gy, gx = np.gradient(g)
    Sxx = _box_blur_np(gx * gx, 3, np)
    Syy = _box_blur_np(gy * gy, 3, np)
    Sxy = _box_blur_np(gx * gy, 3, np)
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - 0.04 * trace * trace
    H, W = R.shape
    P = np.pad(R, 1, mode="constant", constant_values=-np.inf)
    ismax = np.ones_like(R, dtype=bool)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            ismax &= R >= P[di:di + H, dj:dj + W]
    ismax &= R > (R.max() * 0.01)
    ys, xs = np.nonzero(ismax)
    if len(xs) == 0:
        return []
    order = np.argsort(-R[ys, xs], kind="stable")[:n]
    return [(int(xs[i]), int(ys[i])) for i in order]


def _superpixels_grid(img, n_segments, overlay, color):
    """Uniform-grid pseudo-superpixels — the honest no-deps baseline."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    cell = max(4, int((w * h / max(1, n_segments)) ** 0.5))
    out = rgb.copy()
    d = ImageDraw.Draw(out)
    for x in range(cell, w, cell):
        d.line([(x, 0), (x, h)], fill=color)
    for y in range(cell, h, cell):
        d.line([(0, y), (w, y)], fill=color)
    return out


def _superpixels_cc(img, n_segments, np, overlay, color):
    """Superpixels without scikit-image: MEDIANCUT color quantization
    followed by connected-component labeling (8-connectivity flood fill)
    of equal-color regions."""
    rgb = img.convert("RGB")
    small = rgb.resize((128, 128), Image.Resampling.BILINEAR)
    q = small.quantize(
        colors=min(256, max(2, n_segments * 2)),
        method=Image.Quantize.MEDIANCUT,
    ).convert("RGB")
    lab = np.asarray(q)
    H, W, _ = lab.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    cur = 0
    for y in range(H):
        for x in range(W):
            if labels[y, x] != -1:
                continue
            target = (int(lab[y, x, 0]), int(lab[y, x, 1]), int(lab[y, x, 2]))
            stack = [(y, x)]
            labels[y, x] = cur
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= H:
                        continue
                    row_l = lab[ny]
                    row_n = labels[ny]
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= W or row_n[nx] != -1:
                            continue
                        px = row_l[nx]
                        if (int(px[0]), int(px[1]), int(px[2])) == target:
                            row_n[nx] = cur
                            stack.append((ny, nx))
            cur += 1
    bnd = np.zeros((H, W), dtype=bool)
    bnd[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    bnd[:-1, :] |= labels[:-1, :] != labels[1:, :]
    bnd_img = Image.fromarray(bnd.astype(np.uint8) * 255).resize(
        rgb.size, Image.Resampling.NEAREST
    )
    if not overlay:
        full = np.array(
            Image.fromarray(labels, mode="I").resize(
                rgb.size, Image.Resampling.NEAREST
            ),
            dtype=np.int64,
        )
        arr = np.asarray(rgb).astype(np.float64)
        out = np.zeros_like(arr)
        for c in range(cur):
            m = full == c
            if m.any():
                out[m] = arr[m].mean(axis=0)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    out = rgb.copy()
    out.paste(Image.new("RGB", rgb.size, color), mask=bnd_img)
    return out


def _solve_linear_system(m):
    """Gaussian elimination with partial pivoting — pure stdlib."""
    n = len(m)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise MediaEditError("perspective: degenerate point set")
        m[col], m[piv] = m[piv], m[col]
        pivval = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pivval
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def _perspective_coeffs_pillow(src_pts, dst_pts):
    """Pillow PERSPECTIVE coefficients (dst -> src) solved in pure Python."""
    m = []
    for (xd, yd), (xs, ys) in zip(dst_pts, src_pts):
        m.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd, xs])
        m.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd, ys])
    return _solve_linear_system(m)


def _match_hist_pillow(img, ref):
    """Per-channel histogram matching via CDF lookup tables — no numpy."""
    img = img.convert("RGB")
    ref = ref.convert("RGB")
    bands = []
    for ch, rh in zip(img.split(), ref.split()):
        h1 = ch.histogram()
        h2 = rh.histogram()
        n1 = sum(h1) or 1
        n2 = sum(h2) or 1
        cdf1, cdf2 = [], []
        acc = 0
        for v in h1:
            acc += v
            cdf1.append(acc / n1)
        acc = 0
        for v in h2:
            acc += v
            cdf2.append(acc / n2)
        lut = [0] * 256
        j = 0
        for i in range(256):
            while j < 255 and cdf2[j] < cdf1[i]:
                j += 1
            lut[i] = j
        bands.append(ch.point(lut))
    return Image.merge("RGB", bands)


def _dehaze_pillow(img, omega=0.95, t0=0.1):
    """Dark-channel-prior dehazing on Pillow alone.

    Dark channel via MinFilter; atmospheric light = mean color of the
    haziest 0.1% pixels (via ImageStat with a mask); transmission map via
    a point LUT; scene radiance recovered per channel with ImageMath.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    r, g, b = rgb.split()
    dark = ImageChops.darker(ImageChops.darker(r, g), b).filter(
        ImageFilter.MinFilter(15)
    )
    hist = dark.histogram()
    total = w * h
    need = max(1, total // 1000)
    acc = 0
    thr = 255
    for v in range(255, -1, -1):
        acc += hist[v]
        if acc >= need:
            thr = v
            break
    mask = dark.point(lambda v: 255 if v >= thr else 0)
    A = ImageStat.Stat(rgb, mask).mean
    if max(A) < 1:
        return rgb
    amax = max(A)
    tmin = max(1, int(t0 * 255))
    tmap = dark.point([
        max(tmin, int(255 * (1 - omega * (dv / 255) / (amax / 255))))
        for dv in range(256)
    ])
    out = []
    for ch, ac in zip((r, g, b), A):
        ac_i = int(round(ac))
        e = ImageMath.eval(
            "convert(min(max((c - A) * 255 / t + A, 0), 255), 'L')",
            c=ch, t=tmap, A=ac_i,
        )
        out.append(e)
    return Image.merge("RGB", out)


def _seam_carve_python(img, target_w, target_h):
    """Content-aware resizing via dynamic-programming seam removal in pure
    Python.  Correct but O(seams * w * h) interpreted — fine for moderate
    images, slow for very large ones."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    px = list(rgb.getdata())
    rows = [px[i * w:(i + 1) * w] for i in range(h)]
    gray = [
        [(p[0] * 299 + p[1] * 587 + p[2] * 114) // 1000 for p in row]
        for row in rows
    ]

    def remove_vertical(rows, gray):
        h = len(rows)
        w = len(rows[0])
        nrg = [[0] * w for _ in range(h)]
        for i in range(h):
            iu = max(0, i - 1)
            id_ = min(h - 1, i + 1)
            grow = gray[i]
            for j in range(w):
                jl = max(0, j - 1)
                jr = min(w - 1, j + 1)
                nrg[i][j] = abs(grow[jl] - grow[jr]) + abs(
                    gray[iu][j] - gray[id_][j]
                )
        dp = nrg[0][:]
        parent = [[0] * w for _ in range(h)]
        for i in range(1, h):
            ndp = [0] * w
            prow = parent[i]
            for j in range(w):
                bj, bv = j, dp[j]
                if j > 0 and dp[j - 1] < bv:
                    bv, bj = dp[j - 1], j - 1
                if j < w - 1 and dp[j + 1] < bv:
                    bv, bj = dp[j + 1], j + 1
                ndp[j] = nrg[i][j] + bv
                prow[j] = bj
            dp = ndp
        j = min(range(w), key=lambda jj: dp[jj])
        for i in range(h - 1, -1, -1):
            del rows[i][j]
            del gray[i][j]
            if i > 0:
                j = parent[i][j]
        return rows, gray

    while w > target_w:
        rows, gray = remove_vertical(rows, gray)
        w -= 1
    while h > target_h:
        rows = [list(r) for r in zip(*rows)]
        gray = [list(r) for r in zip(*gray)]
        rows, gray = remove_vertical(rows, gray)
        rows = [list(r) for r in zip(*rows)]
        gray = [list(r) for r in zip(*gray)]
        h -= 1
    out = Image.new("RGB", (len(rows[0]), len(rows)))
    out.putdata([p for row in rows for p in row])
    return out


def _parse_cube_lists(text):
    """Parse an Adobe .cube LUT into nested Python lists (no numpy).

    Returns ``(size, title, lut)`` where ``lut[b][g][r] = [r, g, b]`` in
    0..1 floats, plus ``(domain_min, domain_max)``.
    """
    size = None
    title = ""
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]
        if key == "TITLE":
            title = line[5:].strip().strip('"')
        elif key == "LUT_3D_SIZE":
            size = int(parts[1])
        elif key == "DOMAIN_MIN":
            dmin = [float(v) for v in parts[1:4]]
        elif key == "DOMAIN_MAX":
            dmax = [float(v) for v in parts[1:4]]
        elif key == "LUT_1D_SIZE":
            raise MediaEditError("cube_lut: 1D LUTs are not supported")
        elif key not in ("LUT_3D_INPUT_RANGE",):
            try:
                entries.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except (ValueError, IndexError):
                pass
    if size is None:
        raise MediaEditError("cube_lut: missing LUT_3D_SIZE")
    if len(entries) != size ** 3:
        raise MediaEditError(
            f"cube_lut: expected {size ** 3} entries, found {len(entries)}"
        )
    lut = [[[None] * size for _ in range(size)] for _ in range(size)]
    it = iter(entries)
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                lut[bi][gi][ri] = next(it)
    return size, title, lut, dmin, dmax


def _cube_lut_python(img, size, lut, dmin, dmax):
    """Trilinear .cube interpolation in pure Python (slow but exact)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    px = list(rgb.getdata())
    span = [(dmax[c] - dmin[c]) or 1.0 for c in range(3)]
    s = size - 1
    out = []
    for (pr, pg, pb) in px:
        fr = min(1.0, max(0.0, (pr / 255 - dmin[0]) / span[0])) * s
        fg = min(1.0, max(0.0, (pg / 255 - dmin[1]) / span[1])) * s
        fb = min(1.0, max(0.0, (pb / 255 - dmin[2]) / span[2])) * s
        r0 = int(fr); g0 = int(fg); b0 = int(fb)
        r1 = min(r0 + 1, s); g1 = min(g0 + 1, s); b1 = min(b0 + 1, s)
        dr = fr - r0; dg = fg - g0; db = fb - b0
        c000 = lut[b0][g0][r0]; c100 = lut[b0][g0][r1]
        c010 = lut[b0][g1][r0]; c110 = lut[b0][g1][r1]
        c001 = lut[b1][g0][r0]; c101 = lut[b1][g0][r1]
        c011 = lut[b1][g1][r0]; c111 = lut[b1][g1][r1]
        ch = []
        for c in range(3):
            v = (
                c000[c] * (1 - dr) * (1 - dg) * (1 - db)
                + c100[c] * dr * (1 - dg) * (1 - db)
                + c010[c] * (1 - dr) * dg * (1 - db)
                + c110[c] * dr * dg * (1 - db)
                + c001[c] * (1 - dr) * (1 - dg) * db
                + c101[c] * dr * (1 - dg) * db
                + c011[c] * (1 - dr) * dg * db
                + c111[c] * dr * dg * db
            )
            ch.append(int(min(255, max(0, round(v * 255)))))
        out.append((ch[0], ch[1], ch[2]))
    res = Image.new("RGB", (w, h))
    res.putdata(out)
    return res


def _linear_shift_score(A, B, dx, dy, np):
    """Normalized cross-correlation of the true linear overlap when B is
    pasted at offset (dx, dy) over A.  Rejects overlaps under 20%."""
    h, w = A.shape
    x0 = max(0, dx)
    x1 = min(w, w + dx)
    y0 = max(0, dy)
    y1 = min(h, h + dy)
    if x1 - x0 < w // 5 or y1 - y0 < h // 5:
        return -2.0
    Aov = A[y0:y1, x0:x1].astype(np.float64)
    Bov = B[y0 - dy:y1 - dy, x0 - dx:x1 - dx].astype(np.float64)
    Aov -= Aov.mean()
    Bov -= Bov.mean()
    denom = float(np.sqrt((Aov ** 2).sum() * (Bov ** 2).sum()))
    if denom < 1e-9:
        return -2.0
    return float((Aov * Bov).sum() / denom)


def _estimate_shift_numpy(a, b, np):
    """Translation aligning b onto a via FFT phase correlation (numpy).

    Returns the (dx, dy) paste offset for ``b``.  The circular
    correlation peak is ambiguous modulo the frame size, so the top
    peaks are each unwrapped (± frame size) and the candidate with the
    best true linear overlap wins.
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    A = a[:h, :w].astype(np.float64)
    B = b[:h, :w].astype(np.float64)
    A -= A.mean()
    B -= B.mean()
    F = np.fft.fft2(A) * np.conj(np.fft.fft2(B))
    F /= np.abs(F) + 1e-8
    corr = np.abs(np.fft.ifft2(F))
    peaks = np.argsort(corr.ravel())[-3:][::-1]
    best, best_s = (0, 0), -2.0
    for idx in peaks:
        y, x = np.unravel_index(int(idx), corr.shape)
        if y > h // 2:
            y -= h
        if x > w // 2:
            x -= w
        for cx in (int(x) - w, int(x), int(x) + w):
            for cy in (int(y) - h, int(y), int(y) + h):
                s = _linear_shift_score(A, B, cx, cy, np)
                if s > best_s:
                    best, best_s = (cx, cy), s
    return best


def _estimate_shift_sad(a_img, b_img):
    """Coarse horizontal translation search with pure Pillow.

    Mean absolute difference on a 48px-wide downscale, searching shifts
    up to three quarters of the frame width; overlap regions as small as
    a quarter of the frame still count.  Assumes side-by-side panoramas.
    Returns the (dx, 0) paste offset for ``b`` in full-resolution px.
    """
    sw = 48
    sa = a_img.resize(
        (sw, max(1, int(a_img.height * sw / a_img.width))),
        Image.Resampling.BILINEAR,
    )
    sb = b_img.resize(
        (sw, max(1, int(b_img.height * sw / b_img.width))),
        Image.Resampling.BILINEAR,
    )
    da = list(sa.getdata())
    db = list(sb.getdata())
    w, h = sa.size
    best, best_dx = None, 0
    max_dx = w - w // 4
    for dx in range(-max_dx, max_dx + 1):
        x0 = max(0, dx)
        x1 = min(w, w + dx)
        ov = x1 - x0
        if ov < w // 4:
            continue
        s = 0
        for y in range(h):
            ba = y * w
            bb = y * w - dx
            for x in range(x0, x1):
                s += abs(da[ba + x] - db[bb + x])
        score = s / (ov * h)
        if best is None or score < best:
            best, best_dx = score, dx
    return int(round(best_dx * a_img.width / sw)), 0


def _panorama_pillow(images, np):
    """Translation-only panorama stitching without OpenCV.

    Pairwise offsets come from FFT phase correlation (numpy) or a coarse
    SAD search (pure Pillow); frames are composited with a horizontal
    cross-dissolve across each overlap.
    """
    if len(images) < 2:
        raise MediaEditError("panorama: need at least 2 images")
    grays = [im.convert("L") for im in images]
    offs = [(0, 0)]
    for prev, cur in zip(grays, grays[1:]):
        if np is not None:
            dx, dy = _estimate_shift_numpy(
                np.asarray(prev, dtype=np.float64),
                np.asarray(cur, dtype=np.float64),
                np,
            )
        else:
            dx, dy = _estimate_shift_sad(prev, cur)
        px, py = offs[-1]
        offs.append((px + dx, py + dy))
    minx = min(o[0] for o in offs)
    miny = min(o[1] for o in offs)
    offs = [(x - minx, y - miny) for x, y in offs]
    W = max(x + im.width for (x, y), im in zip(offs, images))
    H = max(y + im.height for (x, y), im in zip(offs, images))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    x0, y0 = offs[0]
    canvas.paste(images[0].convert("RGB"), (x0, y0))
    prev_rect = (x0, y0, x0 + images[0].width, y0 + images[0].height)
    for (x, y), im in zip(offs[1:], images[1:]):
        rgb = im.convert("RGB")
        pl, pt, pr, pb = prev_rect
        ox0 = max(x, pl)
        ox1 = min(x + rgb.width, pr)
        if ox1 > ox0:
            ramp = Image.new("L", (ox1 - ox0, 1))
            ramp.putdata([
                int(255 * j / max(1, ox1 - ox0 - 1))
                for j in range(ox1 - ox0)
            ])
            grad = ramp.resize((ox1 - ox0, rgb.height))
            full = Image.new("L", (W, H), 0)
            tile = Image.new("L", rgb.size, 255)
            full.paste(tile, (x, y))
            full.paste(grad, (ox0, y))
            layer = Image.new("RGB", (W, H), (0, 0, 0))
            layer.paste(rgb, (x, y))
            canvas = Image.composite(layer, canvas, full)
        else:
            canvas.paste(rgb, (x, y))
        prev_rect = (x, y, x + rgb.width, y + rgb.height)
    return canvas


def _clarity_pillow(img, amount, radius):
    """Local-contrast clarity on Pillow alone (integer ImageMath)."""
    rgb = img.convert("RGB")
    base = rgb.filter(ImageFilter.GaussianBlur(radius))
    k = int(round(amount * 100))
    out = []
    for ch, bh in zip(rgb.split(), base.split()):
        e = ImageMath.eval(
            f"convert(min(max((a * 100 + {k} * (a - b)) / 100, 0), 255), 'L')",
            a=ch, b=bh,
        )
        out.append(e)
    return Image.merge("RGB", out)


def _retouch_pillow(img, radius, amount):
    """Frequency-separation skin smoothing on Pillow alone.

    low = GaussianBlur; high = (img - low) + 128 per channel via ImageMath;
    the low band is smoothed toward a stronger blur, then recombined.
    """
    rgb = img.convert("RGB")
    low = rgb.filter(ImageFilter.GaussianBlur(radius))
    slow = low.filter(ImageFilter.GaussianBlur(radius * 0.75))
    blended = Image.blend(low, slow, amount)
    out = []
    for ch, lh, bh in zip(rgb.split(), low.split(), blended.split()):
        high = ImageMath.eval(
            "convert(min(max(a - b + 128, 0), 255), 'L')", a=ch, b=lh
        )
        e = ImageMath.eval(
            "convert(min(max(a + b - 128, 0), 255), 'L')", a=bh, b=high
        )
        out.append(e)
    return Image.merge("RGB", out)


def _stylize_pillow(img):
    """Watercolor stylization on Pillow alone: median smoothing +
    posterization with soft dark edges blended back in."""
    rgb = img.convert("RGB")
    sm = rgb.filter(ImageFilter.MedianFilter(7))
    sm = ImageOps.posterize(sm, 5).filter(ImageFilter.GaussianBlur(1.2))
    edge = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    edge = ImageOps.invert(edge).point(lambda v: int(v * 0.6 + 102))
    darkened = ImageChops.darker(sm, edge.convert("RGB"))
    return Image.blend(sm, darkened, 0.6)


def _tonemap_pillow(img):
    """Local tone mapping on Pillow alone: strong large-radius clarity
    (the essence of LDR tone mapping) plus a gentle highlight rolloff."""
    c = _clarity_pillow(img, 0.7, 48)
    r, g, b = c.split()
    out = []
    for ch in (r, g, b):
        # soft highlight compression: v -> v - max(0, v-200)^2/220
        lut = [
            min(255, max(0, v - (max(0, v - 200) ** 2) // 220))
            for v in range(256)
        ]
        out.append(ch.point(lut))
    return Image.merge("RGB", out)


def _seamless_clone_pillow(src, dst, center, mask_img, mix):
    """Feathered-paste compositing on Pillow alone (no numpy).

    ``mask_img`` is an L-mode mask the size of ``src``; it is feathered
    and positioned together with the foreground so ``center`` lands on
    ``dst``.
    """
    sw, sh = src.size
    dw, dh = dst.size
    cx, cy = (int(v) for v in center)
    x0, y0 = cx - sw // 2, cy - sh // 2
    layer = Image.new("RGB", (dw, dh), (0, 0, 0))
    layer.paste(src.convert("RGB"), (x0, y0))
    full_mask = Image.new("L", (dw, dh), 0)
    full_mask.paste(mask_img.filter(ImageFilter.GaussianBlur(4)), (x0, y0))
    if mix == "monochrome":
        lum = dst.convert("L")
        bands = []
        for ch in layer.split():
            bands.append(ImageMath.eval(
                "convert(min(a * b / 128, 255), 'L')", a=ch, b=lum
            ))
        layer = Image.merge("RGB", bands)
    out = dst.convert("RGB").copy()
    out.paste(layer, mask=full_mask)
    return out


# op-chain registration (additive; mirrors studio.py)
# ---------------------------------------------------------------------------

#: op name -> function, wired into ``images._OP_FUNCS`` / ``OP_ALLOWLIST``.
CV_IMAGE_OPS: dict[str, Any] = {
    "denoise": denoise,
    "edge_detect": edge_detect,
    "inpaint_cv": inpaint_cv,
    "seamless_clone": seamless_clone,
    "sharpen": sharpen_advanced,
    "cartoonize": cartoonize,
    "pencil_sketch": pencil_sketch,
    "perspective": perspective_transform,
    "grabcut": grabcut_segment,
    # high-quality resampling / exposure / color
    "rescale": rescale_ski,
    "exposure": exposure_adjust,
    "match_histogram": match_histograms,
    "adaptive_threshold": adaptive_threshold,
    # full capability set: detection, deconvolution, segmentation, color
    "orb_features": orb_features,
    "kmeans": kmeans_quantize,
    "deblur": deblur,
    "superpixels": superpixels,
    "white_balance": white_balance,
    "auto_levels": auto_levels,
    "color_transfer": color_transfer,
    "panorama": panorama,
    # pro retouch / color / compositing
    "retouch_smooth": retouch_smooth,
    "tone_map": tone_map,
    "detail_enhance": detail_enhance,
    "stylize": stylize,
    "clarity": clarity,
    "shadow_highlight": shadow_highlight,
    "dehaze": dehaze,
    "seam_carve": seam_carve,
    "tilt_shift": tilt_shift,
    "selective_color": selective_color,
    "split_tone": split_tone,
    "curves": apply_curves,
    "cube_lut": apply_cube_lut,
    "replace_background": replace_background,
}


def _register_ops() -> None:
    from . import images as _images  # noqa: PLC0415

    for _name, _fn in CV_IMAGE_OPS.items():
        _images._OP_FUNCS[_name] = _fn
        _images.OP_ALLOWLIST.add(_name)


_register_ops()


__all__ = [
    "MediaEditError",
    "cv2_available",
    "skimage_available",
    "backend_status",
    "denoise",
    "edge_detect",
    "inpaint_cv",
    "seamless_clone",
    "sharpen_advanced",
    "cartoonize",
    "pencil_sketch",
    "perspective_transform",
    "grabcut_segment",
    "rescale_ski",
    "rescale_hq",
    "exposure_adjust",
    "match_histograms",
    "adaptive_threshold",
    "orb_features",
    "orb_count",
    "kmeans_quantize",
    "deblur",
    "superpixels",
    "white_balance",
    "auto_levels",
    "color_transfer",
    "panorama",
    "retouch_smooth",
    "tone_map",
    "detail_enhance",
    "stylize",
    "clarity",
    "shadow_highlight",
    "dehaze",
    "seam_carve",
    "tilt_shift",
    "selective_color",
    "split_tone",
    "apply_curves",
    "apply_cube_lut",
    "replace_background",
    "parse_box",
    "parse_corners",
    "CV_IMAGE_OPS",
]
