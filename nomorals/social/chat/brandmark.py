"""The designed on-image brand watermark for PUBLIC-bound images.

PDFs carry the resolved brand as a diagonal watermark and HTML
editions as a translucent layer (core.branding resolves the mark
for all of them); images carry it RENDERED INTO THE PIXELS — the
professional form the owner ordered (2026-11-11: "you should be
wise enough to know how advanced people do the sort of watermark
I described"): the wordmark typeset from TEXT at send time with
the house font stack, so no logo asset is ever required, sized
and placed by rule, and contrast-adaptive so it survives a white
photo and a night photo alike.  Not a caption, not metadata, not
a solid pill with a label on it — a watermark.

The designed values (one home, stated once):

* **width** — the typeset mark spans ``WIDTH_FRACTION`` (0.34)
  of the image width, clamped to [90, 460] px; the font size is
  solved from the measured text width so the fraction is real
  at every image size.  A JOINT height cap rides on top: the
  mark's height may not exceed ``MAX_HEIGHT_FRACTION`` (0.085)
  of the image height — and on LANDSCAPE frames (wider than
  tall, the banner regime) it must ALSO clear
  ``MAX_HEIGHT_WIDTH_FRACTION`` (0.03) of the image width, so
  the mark stays modest against the frame's full span.  (On
  square and portrait frames the height fraction alone
  governs: width isn't the scarce dimension there, and a
  width term would tax a small square — a 240 px QR card —
  out of a mark its height happily carries.)  On a short
  banner the width rule alone would make the mark as tall as
  the content's own type (the 0.12-of-height-only cap still
  overwhelmed a 182 px card), so the fit shrinks until the
  cap holds.  A frame whose binding cap falls below
  ``MIN_MARK_HEIGHT_PX`` (8 px) cannot carry a LEGIBLE mark
  and is refused — a smudge is not a signature.
* **anchor** — bottom-right.  The margin is the largest of
  ``MARGIN_FRACTION`` (0.03) of the smaller image dimension,
  ``MARGIN_MARK_FRACTION`` (0.75) of the mark's OWN height,
  and 8 px: clearance scales with the mark itself, so the
  mark (and its soft shadow) never crowds the frame's edge —
  on the 182 px card the old flat 8 px margin left the mark
  sitting on the bottom edge, reading clipped.
* **contrast** — the luminance of the corner region the mark
  will occupy is sampled from the source: a bright region
  (>= ``LUMINANCE_SPLIT``, 140) takes a DARK ink, a dark region
  takes a LIGHT ink, and the ink always rides a soft shadow in
  the opposite tone so the edge separates on mid-tones too.
* **strength** — ink alpha ``INK_ALPHA`` (0.78): stronger than
  the PDF underlay (branding.BrandMark.opacity, 0.15 — a
  diagonal ghost under text) because a corner brand mark is
  meant to be READ, weaker than full opacity because the photo
  still owns the frame.  Shadow alpha ``SHADOW_ALPHA`` (0.43).

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

__all__ = [
    "BrandMarkError", "burn_brand_mark",
    "WIDTH_FRACTION", "MAX_HEIGHT_FRACTION", "MAX_HEIGHT_WIDTH_FRACTION",
    "MIN_MARK_HEIGHT_PX", "MARGIN_FRACTION", "MARGIN_MARK_FRACTION",
    "INK_ALPHA", "SHADOW_ALPHA", "LUMINANCE_SPLIT",
]

#: The designed values (see the module docstring for the why).
WIDTH_FRACTION = 0.34
WIDTH_MIN_PX = 90
WIDTH_MAX_PX = 460
MAX_HEIGHT_FRACTION = 0.085
MAX_HEIGHT_WIDTH_FRACTION = 0.03
MIN_MARK_HEIGHT_PX = 8
MARGIN_FRACTION = 0.03
MARGIN_MARK_FRACTION = 0.75
INK_ALPHA = 0.78
SHADOW_ALPHA = 0.43
LUMINANCE_SPLIT = 140

_INK_DARK = (16, 16, 20)
_INK_LIGHT = (246, 246, 248)
_SHADOW_ON_DARK = (255, 255, 255)
_SHADOW_ON_LIGHT = (0, 0, 0)


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


def _corner_luminance(img, box: tuple[int, int, int, int]) -> float:
    """Mean luminance (0..255) of *box* in *img* (an RGBA image):
    the region the watermark will occupy, sampled from the
    source so the ink tone is chosen against the real corner,
    never assumed."""
    region = img.crop(box).convert("L")
    px = list(region.getdata())
    if not px:
        return 0.0
    return sum(px) / len(px)


def _fit_font(text: str, target_width: int, start: int):
    """The font whose typeset *text* measures *target_width*:
    start from a size guess, measure with the house ladder's
    font metrics, scale once, clamp to a sane range. Returns
    ``(font, (text_w, text_h))`` measured on the FINAL font."""
    from PIL import Image, ImageDraw

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    def measure(size: int):
        font = _font(size)
        box = probe.textbbox((0, 0), text, font=font)
        return font, box[2] - box[0], box[3] - box[1], box

    size = max(10, min(160, start))
    font, width, height, box = measure(size)
    if width > 0 and abs(width - target_width) > 1:
        size = max(10, min(160, round(size * target_width / width)))
        font, width, height, box = measure(size)
    return font, (width, height), box


def burn_brand_mark(data: bytes, text: str = "") -> bytes:
    """Return ``data`` (PNG/JPEG/WebP bytes) with the brand
    watermark rendered into the bottom-right corner, in the
    source's own format: the wordmark typeset from text, sized
    to the image, inked against the corner's own luminance.
    Raises :class:`BrandMarkError` for unreadable bytes —
    callers at send seams degrade honestly (log + send unmarked)
    rather than block a delivery on the mark.
    """
    from PIL import Image, ImageDraw, ImageFilter

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

    target = max(WIDTH_MIN_PX,
                 min(WIDTH_MAX_PX, round(width * WIDTH_FRACTION)))
    target = min(target, width)  # a narrow image: the mark fits it
    font, (text_w, text_h), box = _fit_font(
        mark_text, target, start=max(10, round(width * 0.045)))
    if text_w <= 0 or text_h <= 0:
        raise BrandMarkError("brand text typesets to nothing")
    # The JOINT height cap: subordinate to the frame's height —
    # and, on landscape frames, to its width too (see the
    # docstring for why square/portrait frames skip the width
    # term).
    height_cap = height * MAX_HEIGHT_FRACTION
    if width > height:
        height_cap = min(height_cap,
                         width * MAX_HEIGHT_WIDTH_FRACTION)
    if text_h > height_cap:
        if height_cap < MIN_MARK_HEIGHT_PX:
            # The frame is too small for a legible mark at the
            # designed proportions — refuse honestly rather than
            # burn in an unreadable smudge.
            raise BrandMarkError("image too small to carry a mark")
        # A short frame (a card banner): shrink the fit until the
        # mark's height is subordinate to the frame's own type.
        font, (text_w, text_h), box = _fit_font(
            mark_text, max(24, round(target * height_cap / text_h)),
            start=max(10, round(width * 0.045)))
    # Clearance scales with the mark itself: never closer to an
    # edge than MARGIN_MARK_FRACTION of the mark's own height.
    margin = max(8, round(min(width, height) * MARGIN_FRACTION),
                 round(text_h * MARGIN_MARK_FRACTION))
    # The mark's footprint: the typeset box placed bottom-right.
    x1, y1 = width - margin, height - margin
    x0, y0 = x1 - text_w, y1 - text_h
    if x0 < 0 or y0 < 0:
        raise BrandMarkError("image too small to carry a mark")

    bright = _corner_luminance(img, (x0, y0, x1, y1)) >= LUMINANCE_SPLIT
    ink = _INK_DARK if bright else _INK_LIGHT
    shadow_rgb = _SHADOW_ON_DARK if bright else _SHADOW_ON_LIGHT
    offset = max(1, round(text_h / 7))

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((x0 - box[0] + offset, y0 - box[1] + offset), mark_text,
              font=font, fill=shadow_rgb + (round(255 * SHADOW_ALPHA),))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(1, offset)))
    draw = ImageDraw.Draw(layer)
    draw.text((x0 - box[0], y0 - box[1]), mark_text,
              font=font, fill=ink + (round(255 * INK_ALPHA),))
    out = Image.alpha_composite(img, layer)
    if fmt == "JPEG":
        out = out.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format=fmt)
    return buf.getvalue()
