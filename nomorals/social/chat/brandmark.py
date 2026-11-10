"""The designed on-image brand mark for PUBLIC-bound images.

PDFs carry the resolved brand as a diagonal watermark and HTML
editions as a translucent layer (core.branding resolves the mark
for all of them); images carry it burned in: a small translucent
dark pill in the bottom-right corner with the brand text in white
— legible on any picture, subordinate to it, unmistakably a
designed mark rather than a caption.

Only images classified ``public`` are marked.  The classification
rides the artifact itself (``MediaRef.audience``) and is set only
by a send seam that KNOWS the surface — the makers runtime in a
group chat, a caller passing ``audience="public"`` to the
gateway.  Images made for the owner himself (his generations, his
edits) are personal and are never marked; unclassified reads as
personal, so ambiguity never marks his own media.
"""

from __future__ import annotations

import io

from ...core.branding import BRAND_TEXT

__all__ = ["BrandMarkError", "burn_brand_mark"]


class BrandMarkError(ValueError):
    """Raised when bytes cannot be marked (not a readable image)."""


def _font(size: int):
    """The card faces' ladder (makers._font's shape): the
    typography service first, the platform font files next,
    Pillow's bundled default last — a mark renders on every
    profile."""
    import os

    from PIL import ImageFont

    try:
        from ...media.typography import get_service

        picked = get_service().pil_font("text", size, weight=700)
    except Exception:  # noqa: BLE001 - the ladder below is the fallback
        picked = None
    if picked is not None:
        return picked
    names = ("DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf",
             "LiberationSans-Bold.ttf", "arialbd.ttf")
    roots = ("/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.fonts"), r"C:\Windows\Fonts")
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in names:
                if name in files:
                    try:
                        return ImageFont.truetype(
                            os.path.join(dirpath, name), size)
                    except OSError:
                        continue
    return ImageFont.load_default()


def burn_brand_mark(data: bytes, text: str = "") -> bytes:
    """Return ``data`` (PNG/JPEG/WebP bytes) with the brand mark
    burned into the bottom-right corner, in the source's own
    format.  Raises :class:`BrandMarkError` for unreadable bytes —
    callers at send seams degrade honestly (log + send unmarked)
    rather than block a delivery on the mark.
    """
    from PIL import Image, ImageDraw

    mark_text = (text or "").strip() or BRAND_TEXT
    try:
        src = Image.open(io.BytesIO(data))
        src.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is named
        raise BrandMarkError(f"not a readable image: {exc}") from exc
    fmt = (src.format or "PNG").upper()
    if fmt not in ("PNG", "JPEG", "WEBP"):
        raise BrandMarkError(
            f"brand mark supports PNG/JPEG/WebP, not {fmt}")
    img = src.convert("RGBA")
    width, height = img.size
    if width < 8 or height < 8:
        raise BrandMarkError("image too small to carry a mark")
    size = max(9, min(26, round(min(width, height) * 0.034)))
    font = _font(size)
    probe = ImageDraw.Draw(img)
    box = probe.textbbox((0, 0), mark_text, font=font)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = max(6, size // 2), max(3, size // 4)
    margin = max(6, size // 2)
    pill_w, pill_h = text_w + 2 * pad_x, text_h + 2 * pad_y
    x1, y1 = width - margin, height - margin
    x0, y0 = x1 - pill_w, y1 - pill_h
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        [x0, y0, x1, y1], radius=max(4, pill_h // 2),
        fill=(14, 14, 18, 118))
    draw.text((x0 + pad_x - box[0], y0 + pad_y - box[1]), mark_text,
              font=font, fill=(255, 255, 255, 222))
    out = Image.alpha_composite(img, overlay)
    if fmt == "JPEG":
        out = out.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format=fmt)
    return buf.getvalue()
