"""God-tier image editing engine: Pillow-backed, pure composable ops.

Every transform is a pure function ``Image -> Image`` so op chains stay
predictable and testable. File IO (load/save) is separate from the pixel ops,
and :func:`edit_image` guarantees the original file is never overwritten.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)


class MediaEditError(Exception):
    """Raised for any image-edit failure: bad input, bad op, corrupt file."""


# ---------------------------------------------------------------------------
# loading / saving
# ---------------------------------------------------------------------------

def _require_pillow() -> Any:
    try:
        import PIL  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        raise MediaEditError(
            "Pillow is not installed. Install it with: pip install 'Pillow>=10.0' "
            "or: pip install nomorals[media-edit]"
        ) from exc
    return Image


def load_image(path: str | os.PathLike[str]) -> Any:
    """Load an image with EXIF auto-orientation applied (phone photos)."""
    Image = _require_pillow()
    from PIL import ImageOps
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise MediaEditError(f"could not load image {path}: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    return img


_FORMATS = {
    ".jpg": "JPEG", ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
}


def _avif_supported() -> bool:
    Image = _require_pillow()
    return "AVIF" in Image.registered_extensions().values()


def save_image(
    img: Any,
    path: str | os.PathLike[str],
    *,
    fmt: str | None = None,
    quality: int = 90,
    strip_metadata: bool = True,
) -> Path:
    """Save ``img`` to ``path``. Metadata is stripped by default (privacy)."""
    Image = _require_pillow()
    out = Path(path)
    ext = out.suffix.lower()
    if fmt is None:
        if ext == ".avif":
            if not _avif_supported():
                raise MediaEditError(
                    "AVIF output is not supported by this Pillow build; "
                    "use PNG, JPEG, or WebP instead."
                )
            fmt = "AVIF"
        else:
            fmt = _FORMATS.get(ext)
        if fmt is None:
            raise MediaEditError(f"unsupported output format for {out.name!r}")
    save_img = img
    if fmt == "JPEG" and save_img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", save_img.size, (255, 255, 255))
        background.paste(save_img, mask=save_img.split()[-1])
        save_img = background
    elif fmt == "JPEG" and save_img.mode != "RGB":
        save_img = save_img.convert("RGB")
    kwargs: dict[str, Any] = {}
    if fmt in ("JPEG", "WEBP", "AVIF"):
        kwargs["quality"] = quality
    # strip_metadata: simply do not pass exif= — Pillow drops it by default.
    if not strip_metadata:
        exif = img.getexif()
        if exif:
            kwargs["exif"] = exif
    out.parent.mkdir(parents=True, exist_ok=True)
    save_img.save(out, fmt, **kwargs)
    return out


def image_probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return dimensions, format, mode, EXIF summary, and file size."""
    Image = _require_pillow()
    p = Path(path)
    try:
        with Image.open(p) as img:
            img.load()
            exif = img.getexif()
            exif_summary = {
                "tag_count": len(exif),
                "has_gps": any(t in exif for t in (34853,)),
            }
            info = {
                "kind": "image",
                "path": str(p),
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "bytes": p.stat().st_size,
                "exif": exif_summary,
            }
    except Exception as exc:
        raise MediaEditError(f"could not probe image {path}: {exc}") from exc
    return info


# ---------------------------------------------------------------------------
# transform ops (pure Image -> Image)
# ---------------------------------------------------------------------------


def _resample(name: str) -> int:
    from PIL import Image
    table = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        return table[name.lower()]
    except KeyError:
        raise MediaEditError(f"unknown resample filter {name!r}; "
                             f"choose from {sorted(table)}") from None


def op_resize(img: Any, width: int, height: int, *,
              mode: str = "fit", resample: str = "lanczos") -> Any:
    """Resize. mode=fit keeps aspect inside the box; fill covers then crops
    center; exact stretches."""
    if width <= 0 or height <= 0:
        raise MediaEditError("resize dimensions must be positive")
    rs = _resample(resample)
    if mode == "exact":
        return img.resize((width, height), rs)
    if mode == "fit":
        ratio = min(width / img.width, height / img.height)
        return img.resize((max(1, round(img.width * ratio)),
                           max(1, round(img.height * ratio))), rs)
    if mode == "fill":
        ratio = max(width / img.width, height / img.height)
        big = img.resize((max(1, round(img.width * ratio)),
                          max(1, round(img.height * ratio))), rs)
        left = (big.width - width) // 2
        top = (big.height - height) // 2
        return big.crop((left, top, left + width, top + height))
    raise MediaEditError(f"unknown resize mode {mode!r}; use fit/fill/exact")


def _parse_aspect(aspect: str) -> float:
    try:
        w, h = aspect.replace(":", "/").split("/")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        raise MediaEditError(
            f"bad aspect ratio {aspect!r}; use e.g. '1:1' or '16:9'") from None


def _smart_crop_box(img: Any, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Pick the crop window with the highest edge energy (simple saliency)."""
    from PIL import ImageFilter
    small = img.convert("L").resize((64, 64))
    edges = small.filter(ImageFilter.FIND_EDGES)
    px = edges.load()
    # Map target window back onto the 64x64 energy map.
    sx = img.width / 64
    sy = img.height / 64
    win_w = max(1, round(target_w / sx))
    win_h = max(1, round(target_h / sy))
    best, best_xy = -1.0, (0, 0)
    step = 4
    for y in range(0, 64 - win_h + 1, step):
        row = 0.0
        for x in range(0, 64 - win_w + 1, step):
            # coarse energy: sum of edge brightness in the window
            e = 0
            for yy in range(y, min(y + win_h, 64), 2):
                for xx in range(x, min(x + win_w, 64), 2):
                    e += px[xx, yy]
            if e > best:
                best, best_xy = e, (x, y)
    bx, by = best_xy
    left = min(img.width - target_w, round(bx * sx))
    top = min(img.height - target_h, round(by * sy))
    return max(0, left), max(0, top), max(0, left) + target_w, max(0, top) + target_h


def op_crop(img: Any, *,
            box: tuple[int, int, int, int] | None = None,
            aspect: str | None = None,
            anchor: str = "center") -> Any:
    """Crop to an explicit box, or to an aspect ratio with an anchor.

    anchors: center, top, bottom, left, right, smart (edge-energy saliency).
    """
    if box is not None:
        l, t, r, b = (int(v) for v in box)
        if not (0 <= l < r <= img.width and 0 <= t < b <= img.height):
            raise MediaEditError(f"crop box {(l, t, r, b)} outside "
                                 f"{img.width}x{img.height}")
        return img.crop((l, t, r, b))
    if aspect is None:
        raise MediaEditError("crop needs box= or aspect=")
    target_ratio = _parse_aspect(aspect)
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:  # too wide -> cut sides
        target_h = img.height
        target_w = round(target_h * target_ratio)
    else:  # too tall -> cut top/bottom
        target_w = img.width
        target_h = round(target_w / target_ratio)
    target_w, target_h = max(1, target_w), max(1, target_h)
    if anchor == "center":
        left = (img.width - target_w) // 2
        top = (img.height - target_h) // 2
    elif anchor == "top":
        left, top = (img.width - target_w) // 2, 0
    elif anchor == "bottom":
        left, top = (img.width - target_w) // 2, img.height - target_h
    elif anchor == "left":
        left, top = 0, (img.height - target_h) // 2
    elif anchor == "right":
        left, top = img.width - target_w, (img.height - target_h) // 2
    elif anchor == "smart":
        return img.crop(_smart_crop_box(img, target_w, target_h))
    else:
        raise MediaEditError(f"unknown crop anchor {anchor!r}")
    return img.crop((left, top, left + target_w, top + target_h))


def op_rotate(img: Any, angle: float, *, expand: bool = True) -> Any:
    from PIL import Image
    resample = Image.Resampling.BICUBIC
    return img.rotate(angle, resample=resample, expand=expand)


def op_flip(img: Any, direction: str = "horizontal") -> Any:
    from PIL import ImageOps
    if direction == "horizontal":
        return ImageOps.mirror(img)
    if direction == "vertical":
        return ImageOps.flip(img)
    raise MediaEditError(f"unknown flip direction {direction!r}")


# ---------------------------------------------------------------------------
# enhance ops
# ---------------------------------------------------------------------------

def op_enhance(img: Any, *,
               brightness: float = 1.0,
               contrast: float = 1.0,
               sharpness: float = 1.0,
               color: float = 1.0,
               autocontrast: bool = False,
               grayscale: bool = False) -> Any:
    from PIL import ImageEnhance, ImageOps
    out = img
    if grayscale:
        out = ImageOps.grayscale(out)
    if autocontrast:
        out = ImageOps.autocontrast(out)
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
    if color != 1.0 and out.mode != "L":
        out = ImageEnhance.Color(out).enhance(color)
    return out


def op_thumbnail(img: Any, size: int = 256) -> Any:
    """Resize-aware thumbnail: downscale in one high-quality step."""
    from PIL import Image
    out = img.copy()
    out.thumbnail((size, size), Image.Resampling.LANCZOS)
    return out


# ---------------------------------------------------------------------------
# annotate ops
# ---------------------------------------------------------------------------

def _load_font(size: int) -> Any:
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_position(img: Any, text: str, font: Any, position: str,
                   margin: int) -> tuple[int, int]:
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if position == "top":
        return (img.width - tw) // 2, margin
    if position == "bottom":
        return (img.width - tw) // 2, img.height - th - margin
    if position == "center":
        return (img.width - tw) // 2, (img.height - th) // 2
    if position == "top-left":
        return margin, margin
    if position == "top-right":
        return img.width - tw - margin, margin
    if position == "bottom-left":
        return margin, img.height - th - margin
    if position == "bottom-right":
        return img.width - tw - margin, img.height - th - margin
    raise MediaEditError(f"unknown text position {position!r}")


def op_annotate_text(img: Any, text: str, *,
                     position: str = "bottom",
                     font_size: int | None = None,
                     color: str = "white",
                     stroke_width: int = 2,
                     stroke_fill: str = "black",
                     margin: int = 20) -> Any:
    from PIL import ImageDraw
    out = img.copy()
    size = font_size or max(16, out.height // 20)
    font = _load_font(size)
    draw = ImageDraw.Draw(out)
    x, y = _text_position(out, text, font, position, margin)
    draw.text((x, y), text, font=font, fill=color,
              stroke_width=stroke_width, stroke_fill=stroke_fill)
    return out


def op_annotate_shape(img: Any, shape: str, *,
                      box: tuple[int, int, int, int] | None = None,
                      outline: str = "red", width: int = 4) -> Any:
    """Draw a rectangle, circle (ellipse), or arrow. ``box`` defaults to a
    centered box covering the middle third."""
    from PIL import ImageDraw
    out = img.copy()
    draw = ImageDraw.Draw(out)
    if box is None:
        w, h = out.width // 3, out.height // 3
        box = ((out.width - w) // 2, (out.height - h) // 2,
               (out.width + w) // 2, (out.height + h) // 2)
    l, t, r, b = (int(v) for v in box)
    if shape in ("rectangle", "rect"):
        draw.rectangle([l, t, r, b], outline=outline, width=width)
    elif shape in ("circle", "ellipse"):
        draw.ellipse([l, t, r, b], outline=outline, width=width)
    elif shape == "arrow":
        # shaft from left-center to right-center with a head
        y = (t + b) // 2
        head = max(width * 3, (r - l) // 6)
        draw.line([l, y, r - head, y], fill=outline, width=width)
        draw.polygon([(r, y), (r - head, y - head // 2),
                      (r - head, y + head // 2)], fill=outline)
    else:
        raise MediaEditError(f"unknown shape {shape!r}; "
                             "use rectangle/circle/arrow")
    return out


# ---------------------------------------------------------------------------
# composite ops
# ---------------------------------------------------------------------------

def _to_rgb(img: Any) -> Any:
    return img.convert("RGB") if img.mode in ("RGBA", "LA", "PA") else img


def op_stack(images: list[Any], *, direction: str = "horizontal",
             bg: str = "black", gap: int = 0) -> Any:
    """Stack images horizontally or vertically, aligned on the long edge."""
    Image = _require_pillow()
    if not images:
        raise MediaEditError("stack needs at least one image")
    imgs = [_to_rgb(i) for i in images]
    if direction == "horizontal":
        h = max(i.height for i in imgs)
        scaled = [op_resize(i, round(i.width * h / i.height), h, mode="exact")
                  if i.height != h else i for i in imgs]
        total_w = sum(i.width for i in scaled) + gap * (len(scaled) - 1)
        canvas = Image.new("RGB", (total_w, h), bg)
        x = 0
        for i in scaled:
            canvas.paste(i, (x, (h - i.height) // 2))
            x += i.width + gap
        return canvas
    if direction == "vertical":
        w = max(i.width for i in imgs)
        scaled = [op_resize(i, w, round(i.height * w / i.width), mode="exact")
                  if i.width != w else i for i in imgs]
        total_h = sum(i.height for i in scaled) + gap * (len(scaled) - 1)
        canvas = Image.new("RGB", (w, total_h), bg)
        y = 0
        for i in scaled:
            canvas.paste(i, ((w - i.width) // 2, y))
            y += i.height + gap
        return canvas
    raise MediaEditError(f"unknown stack direction {direction!r}")


def op_grid(images: list[Any], *, cols: int = 3, bg: str = "black",
            gap: int = 4, cell: int | None = None) -> Any:
    """Contact sheet: uniform cells in a grid."""
    Image = _require_pillow()
    if not images:
        raise MediaEditError("grid needs at least one image")
    if cols <= 0:
        raise MediaEditError("grid cols must be positive")
    cell = cell or 320
    thumbs = [op_resize(_to_rgb(i), cell, cell, mode="fill") for i in images]
    rows = math.ceil(len(thumbs) / cols)
    canvas = Image.new("RGB",
                       (cols * cell + gap * (cols - 1),
                        rows * cell + gap * (rows - 1)), bg)
    for idx, th in enumerate(thumbs):
        r, c = divmod(idx, cols)
        canvas.paste(th, (c * (cell + gap), r * (cell + gap)))
    return canvas


def op_meme(img: Any, *, top: str = "", bottom: str = "",
            color: str = "white") -> Any:
    """Meme-style top/bottom captions with auto-shrinking type."""
    from PIL import ImageDraw
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for text, y_frac in ((top, 0.03), (bottom, 0.97)):
        if not text:
            continue
        size = max(16, out.height // 12)
        font = _load_font(size)
        # shrink until it fits 94% of the width
        while size > 12:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
            if bbox[2] - bbox[0] <= out.width * 0.94:
                break
            size -= 2
            font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (out.width - tw) // 2
        y = int(out.height * y_frac) if y_frac < 0.5 else int(out.height * y_frac) - th
        draw.text((x, y), text.upper(), font=font, fill=color,
                  stroke_width=max(1, size // 15), stroke_fill="black")
    return out


# ---------------------------------------------------------------------------
# op chain + file-level edit
# ---------------------------------------------------------------------------

OP_ALLOWLIST = {
    "resize", "crop", "rotate", "flip",
    "enhance", "thumbnail",
    "annotate_text", "annotate_shape",
    "stack", "grid", "meme",
    "convert",
}

_OP_FUNCS = {
    "resize": op_resize,
    "crop": op_crop,
    "rotate": op_rotate,
    "flip": op_flip,
    "enhance": op_enhance,
    "thumbnail": op_thumbnail,
    "annotate_text": op_annotate_text,
    "annotate_shape": op_annotate_shape,
    "stack": op_stack,
    "grid": op_grid,
    "meme": op_meme,
}


def validate_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate an explicit op chain against the allowlist."""
    if not isinstance(ops, list) or not ops:
        raise MediaEditError("ops must be a non-empty list of op dicts")
    clean: list[dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            raise MediaEditError(f"bad op entry {op!r}: needs an 'op' key")
        name = op["op"]
        if name not in OP_ALLOWLIST:
            raise MediaEditError(
                f"op {name!r} is not allowed; allowed: {sorted(OP_ALLOWLIST)}")
        if name == "convert":
            # convert is a save-time concern, keep it out of pixel ops
            clean.append(dict(op))
            continue
        clean.append(dict(op))
    return clean


def apply_chain(img: Any, ops: list[dict[str, Any]]) -> Any:
    """Apply a validated op chain to an image (pure, no IO)."""
    ops = validate_ops(ops)
    out = img
    extra_images: list[Any] = []
    for op in ops:
        name = op["op"]
        params = {k: v for k, v in op.items() if k != "op"}
        if name == "convert":
            continue  # handled at save time
        if name in ("stack", "grid"):
            others = params.pop("images", [])
            if not isinstance(others, list):
                raise MediaEditError(f"{name} needs an 'images' list")
            loaded = [load_image(p) if isinstance(p, (str, Path)) else p
                      for p in others]
            extra_images.extend(loaded)
            out = _OP_FUNCS[name]([out] + loaded, **params)
        else:
            out = _OP_FUNCS[name](out, **params)
    for extra in extra_images:
        try:
            extra.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
    return out


def _unique_output(src: Path, out_dir: Path, suffix: str, ext: str) -> Path:
    """Build an output path that can never equal the source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{src.stem}-{suffix}{ext}"
    candidate = out_dir / base
    n = 1
    src_resolved = src.resolve()
    while candidate.exists() or candidate.resolve() == src_resolved:
        n += 1
        candidate = out_dir / f"{src.stem}-{suffix}-{n}{ext}"
    return candidate


def edit_image(src: str | os.PathLike[str],
               ops: list[dict[str, Any]],
               *,
               out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "edited",
               fmt: str | None = None,
               ext: str | None = None,
               quality: int = 90,
               strip_metadata: bool = True) -> dict[str, Any]:
    """Apply ``ops`` to ``src`` and write a NEW file. The original is never
    touched — the output path is forced to differ from the source."""
    src_p = Path(src)
    if not src_p.exists():
        raise MediaEditError(f"no such image: {src}")
    img = load_image(src_p)
    try:
        out_img = apply_chain(img, ops)
    except Exception:
        img.close()
        raise
    if out_img is not img:
        img.close()  # pixel ops produced a new image; drop the source handle
    convert_ops = [o for o in ops if o.get("op") == "convert"]
    if fmt is None and convert_ops:
        fmt = str(convert_ops[-1].get("format", "")).upper() or None
    out_ext = ext or (f".{fmt.lower()}" if fmt else src_p.suffix.lower())
    if out_ext == ".jpg":
        out_ext = ".jpeg"
    target_dir = Path(out_dir) if out_dir else src_p.parent / "edited"
    out_path = _unique_output(src_p, target_dir, suffix, out_ext)
    save_image(out_img, out_path, fmt=fmt, quality=quality,
               strip_metadata=strip_metadata)
    try:
        out_img.close()
    except Exception:  # noqa: BLE001
        pass
    return {
        "input": str(src_p),
        "output": str(out_path),
        "ops": [o["op"] for o in ops],
        "bytes": out_path.stat().st_size,
        "suffix": suffix,
    }


def batch_edit(src_dir: str | os.PathLike[str],
               ops: list[dict[str, Any]],
               *,
               pattern: str = "*.jpg",
               out_dir: str | os.PathLike[str] | None = None,
               suffix: str = "edited") -> list[dict[str, Any]]:
    """Apply an op chain to every image in a directory matching ``pattern``."""
    src_d = Path(src_dir)
    results = []
    for path in sorted(src_d.glob(pattern)):
        if path.is_file():
            try:
                results.append(edit_image(path, ops, out_dir=out_dir,
                                          suffix=suffix))
            except MediaEditError as exc:
                _log.warning("batch_edit skipped %s: %s", path, exc)
                results.append({"input": str(path), "error": str(exc)})
    return results


@dataclass
class WatermarkSpec:
    logo_path: str
    position: str = "bottom-right"
    scale: float = 0.15
    opacity: float = 0.85
    margin: int = 20


def op_watermark(img: Any, logo: Any, *,
                 position: str = "bottom-right",
                 scale: float = 0.15,
                 opacity: float = 0.85,
                 margin: int = 20) -> Any:
    """Composite a logo onto the image. ``logo`` is an Image or a path."""
    if isinstance(logo, (str, Path)):
        logo_img = load_image(logo)
    else:
        logo_img = logo
    lw = max(1, round(img.width * scale))
    lh = max(1, round(logo_img.height * lw / logo_img.width))
    mark = op_resize(logo_img.convert("RGBA"), lw, lh, mode="exact")
    if opacity < 1.0:
        alpha = mark.split()[-1].point(lambda a: int(a * opacity))
        mark.putalpha(alpha)
    positions = {
        "top-left": (margin, margin),
        "top-right": (img.width - lw - margin, margin),
        "bottom-left": (margin, img.height - lh - margin),
        "bottom-right": (img.width - lw - margin, img.height - lh - margin),
        "center": ((img.width - lw) // 2, (img.height - lh) // 2),
    }
    if position not in positions:
        raise MediaEditError(f"unknown watermark position {position!r}")
    out = img.convert("RGBA")
    out.alpha_composite(mark, positions[position])
    result = out.convert("RGB") if img.mode == "RGB" else out
    if isinstance(logo, (str, Path)):
        logo_img.close()
    return result


_OP_FUNCS["watermark"] = op_watermark
OP_ALLOWLIST.add("watermark")


@dataclass
class EditPlan:
    """A validated, human-readable edit plan (used for dry-run output)."""
    ops: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    kind: str = "image"

    def describe(self) -> str:
        lines = [f"plan: {self.summary or self.kind + ' edit'}"]
        for i, op in enumerate(self.ops, 1):
            params = ", ".join(f"{k}={v}" for k, v in op.items() if k != "op")
            lines.append(f"  {i}. {op['op']}({params})")
        return "\n".join(lines)
