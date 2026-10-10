"""Text replacement in photos: detect text -> inpaint it out -> render new text.

The underserved killer feature: ``replace_text(img, "SALE", "SOLD")``
finds the word SALE in a photo, erases it, and writes SOLD back in
roughly the same size, position, and colour.

Honesty notes (read before expecting miracles):
- Font matching is APPROXIMATE: size from the OCR box height, colour
  sampled from the original text pixels, weight guessed from stroke
  thickness.  Good enough for memes and signs — not pixel-perfect
  forgery, and not trying to be.
- The inpaint ladder is ComfyUI (best) -> cv2 Telea -> PIL fill (rough).
  The PIL path fills with the surrounding background colour; on busy
  backgrounds it will look patched.  The backend used is returned in
  the result metadata so callers can say so honestly.
- OCR needs the optional ``pytesseract`` package + the tesseract
  binary.  Without them every entry point fails fast with the pip
  hint — never a silent no-op.

Public API:
    detect_text(img) -> list[TextRegion]
    replace_text(img, old_text, new_text, *, inpaint_backend="auto")
        -> tuple[PIL image, dict metadata]
    op_edittext(img, old_text, new_text, ...)  # chain op ("edittext")
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .images import MediaEditError, _load_font, _require_pillow

_log = get_logger(__name__)

#: Minimum difflib ratio for old_text -> detected region matching.
_MATCH_THRESHOLD = 0.6


@dataclass
class TextRegion:
    """One detected line of text."""
    box: tuple[int, int, int, int]  # (l, t, r, b) pixels
    text: str
    confidence: float = 0.0
    words: list[dict[str, Any]] = field(default_factory=list)


def _require_tesseract() -> Any:
    """pytesseract + binary, or a clear pip hint.  Reuses the documents
    OCR module's checker instead of duplicating it."""
    from ..documents.ocr import _require_pytesseract
    return _require_pytesseract()


def detect_text(img: Any, *, lang: str = "eng") -> list[TextRegion]:
    """Detect text lines in ``img`` -> list of TextRegion.

    Word boxes from tesseract's ``image_to_data`` are grouped into lines
    by (block, paragraph, line) ids.  Returns [] when nothing is found
    or the image is unreadable — never raises on weird images (missing
    OCR stack raises the pip-hint error instead).
    """
    Image = _require_pillow()
    pytesseract = _require_tesseract()
    try:
        rgb = img.convert("RGB")
        data = pytesseract.image_to_data(
            rgb, lang=lang, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # noqa: BLE001 - unreadable image, not a crash
        _log.warning("edittext detect failed: %s", exc)
        return []
    n = len(data.get("text", []))
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        word = str(data["text"][i] or "").strip()
        if not word:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]),
               int(data["line_num"][i]))
        lines.setdefault(key, []).append(i)
    regions: list[TextRegion] = []
    for idxs in lines.values():
        lefts, tops, rights, bottoms = [], [], [], []
        words, confs, texts = [], [], []
        for i in idxs:
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < 0:
                continue
            l, t = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            lefts.append(l)
            tops.append(t)
            rights.append(l + w)
            bottoms.append(t + h)
            words.append({"text": str(data["text"][i]).strip(),
                          "box": (l, t, l + w, t + h), "conf": conf})
            confs.append(conf)
            texts.append(str(data["text"][i]).strip())
        if not texts:
            continue
        regions.append(TextRegion(
            box=(min(lefts), min(tops), max(rights), max(bottoms)),
            text=" ".join(texts),
            confidence=sum(confs) / len(confs),
            words=words,
        ))
    # reading order: top to bottom, left to right
    regions.sort(key=lambda r: (r.box[1] // 10, r.box[0]))
    return regions


def _best_match(regions: list[TextRegion],
                old_text: str) -> TextRegion | None:
    """Best fuzzy match for old_text, or None below the threshold.

    Substring containment counts strongly: looking for "SALE" in a
    region reading "SALE TODAY" is the natural use case, and pure
    difflib under-rates it (0.57 for that pair).
    """
    want = old_text.strip().lower()
    if not want:
        return None
    best: TextRegion | None = None
    best_score = 0.0
    for region in regions:
        have = region.text.lower()
        score = difflib.SequenceMatcher(None, want, have).ratio()
        if want in have or have in want:
            score = max(score, 0.75)
        if score > best_score:
            best_score, best = score, region
    if best is None or best_score < _MATCH_THRESHOLD:
        return None
    return best


def _sample_text_color(img: Any,
                       box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Dominant text colour inside ``box``.

    Text is usually darker (or lighter) than its background: split the
    box's pixels by the median luminance and take the median RGB of the
    smaller cluster — the text strokes.  Falls back to near-black.
    """
    Image = _require_pillow()
    l, t, r, b = (max(0, int(v)) for v in box)
    crop = img.convert("RGB").crop((l, t, max(l + 1, r), max(t + 1, b)))
    px = list(crop.getdata())
    if not px:
        return (20, 20, 20)
    lums = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in px]
    med = sorted(lums)[len(lums) // 2]
    dark = [p for p, lm in zip(px, lums) if lm <= med]
    light = [p for p, lm in zip(px, lums) if lm > med]
    cluster = dark if len(dark) <= len(light) else light
    if not cluster:
        cluster = px
    n = len(cluster)
    return (sum(p[0] for p in cluster) // n,
            sum(p[1] for p in cluster) // n,
            sum(p[2] for p in cluster) // n)


def _inpaint_region(img: Any, box: tuple[int, int, int, int],
                    backend: str = "auto") -> tuple[Any, str]:
    """Erase the boxed region -> (image, backend_used).

    Ladder: comfy (ComfyUI inpaint, best) -> cv2 (Telea) -> pil
    (background-colour fill, rough).  ``backend`` pins one step;
    "auto" walks the ladder.
    """
    Image = _require_pillow()
    from PIL import ImageDraw
    l, t, r, b = (int(v) for v in box)
    # pad slightly so no glyph edge survives
    pad = max(2, (b - t) // 12)
    w, h = img.size
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(w, r + pad), min(h, b + pad)
    padded = (l, t, r, b)

    order = [backend] if backend != "auto" else ["comfy", "cv2", "pil"]
    last_error = ""
    for step in order:
        if step == "comfy":
            try:
                from .comfy import ComfyUIBackend, comfy_available
                ok, _reason = comfy_available()
                if not ok:
                    raise MediaEditError("ComfyUI not reachable")
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rectangle([l, t, r - 1, b - 1],
                                               fill=255)
                out = ComfyUIBackend().inpaint(
                    img.convert("RGB"), mask,
                    "seamless background texture, no text, no letters, "
                    "no watermark")
                return out.convert("RGB"), "comfy"
            except Exception as exc:  # noqa: BLE001 - ladder continues
                last_error = str(exc)
                _log.debug("edittext comfy inpaint unavailable: %s", exc)
                continue
        if step == "cv2":
            try:
                from .cv_ops import inpaint_cv
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rectangle([l, t, r - 1, b - 1],
                                               fill=255)
                return inpaint_cv(img.convert("RGB"), mask), "cv2"
            except Exception as exc:  # noqa: BLE001 - ladder continues
                last_error = str(exc)
                _log.debug("edittext cv2 inpaint unavailable: %s", exc)
                continue
        if step == "pil":
            # Rough path: fill with the median colour of the border ring
            # around the box (the local background).
            rgb = img.convert("RGB")
            ring = []
            px = rgb.load()
            for x in range(l, r):
                for y in (t, b - 1):
                    if 0 <= y < h:
                        ring.append(px[x, y])
            for y in range(t, b):
                for x in (l, r - 1):
                    if 0 <= x < w:
                        ring.append(px[x, y])
            if ring:
                n = len(ring)
                fill = (sum(p[0] for p in ring) // n,
                        sum(p[1] for p in ring) // n,
                        sum(p[2] for p in ring) // n)
            else:
                fill = (255, 255, 255)
            out = rgb.copy()
            ImageDraw.Draw(out).rectangle([l, t, r - 1, b - 1], fill=fill)
            return out, "pil"
    raise MediaEditError(
        f"text-region inpaint failed (backend={backend!r}): {last_error}")


def _fit_font_size(draw: Any, text: str, box_w: int, box_h: int) -> int:
    """Largest font size that fits new_text inside the old box."""
    size = max(8, int(box_h * 0.72))
    while size > 8:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= box_w * 0.98 and th <= box_h * 1.15:
            return size
        size = max(8, size - 2)
    return 8


def replace_text(img: Any, old_text: str, new_text: str, *,
                 inpaint_backend: str = "auto",
                 lang: str = "eng") -> tuple[Any, dict[str, Any]]:
    """Replace ``old_text`` with ``new_text`` in ``img``.

    Returns (new_image, metadata) where metadata names the region found,
    the match score, the inpaint backend used, and the sampled colour —
    so callers can describe honestly what happened.

    Raises MediaEditError when no text is found, nothing matches
    ``old_text`` (the error lists what WAS detected), or inpainting
    fails.  Never fakes a replacement.
    """
    Image = _require_pillow()
    from PIL import ImageDraw
    if not old_text or not old_text.strip():
        raise MediaEditError("replace_text needs the text to find")
    if not new_text or not new_text.strip():
        raise MediaEditError("replace_text needs the replacement text")
    regions = detect_text(img, lang=lang)
    if not regions:
        raise MediaEditError(
            "no text detected in the image — nothing to replace")
    region = _best_match(regions, old_text)
    if region is None:
        found = "; ".join(f"{r.text!r}" for r in regions[:8])
        raise MediaEditError(
            f"couldn't find {old_text!r} in the image. "
            f"Detected text: {found}")
    l, t, r, b = region.box
    colour = _sample_text_color(img, region.box)
    erased, used = _inpaint_region(img, region.box,
                                  backend=inpaint_backend)
    out = erased.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    box_w, box_h = r - l, b - t
    size = _fit_font_size(draw, new_text.strip(), box_w, box_h)
    font = _load_font(size)
    # stroke weight scales with size — bolder text reads bolder
    stroke = 1 if size < 28 else 2
    bbox = draw.textbbox((0, 0), new_text.strip(), font=font,
                         stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = l + (box_w - tw) // 2 - bbox[0]
    y = t + (box_h - th) // 2 - bbox[1]
    draw.text((x, y), new_text.strip(), font=font, fill=colour,
              stroke_width=stroke, stroke_fill=colour)
    meta = {
        "found": region.text,
        "box": region.box,
        "confidence": round(region.confidence, 1),
        "inpaint_backend": used,
        "font_size": size,
        "colour": colour,
    }
    return out, meta


def remove_text(img: Any, text: str, *,
                inpaint_backend: str = "auto",
                lang: str = "eng") -> tuple[Any, dict[str, Any]]:
    """Erase the region matching ``text`` — no re-rendering.

    Returns (cleaned_image, metadata). The privacy/redaction primitive:
    detect → inpaint, and stop. Raises MediaEditError when nothing is
    detected or nothing matches (the error lists what WAS detected).
    """
    if not text or not text.strip():
        raise MediaEditError("remove_text needs the text to find")
    regions = detect_text(img, lang=lang)
    if not regions:
        raise MediaEditError(
            "no text detected in the image — nothing to remove")
    region = _best_match(regions, text)
    if region is None:
        found = "; ".join(f"{r.text!r}" for r in regions[:8])
        raise MediaEditError(
            f"couldn't find {text!r} in the image. "
            f"Detected text: {found}")
    erased, used = _inpaint_region(img, region.box, backend=inpaint_backend)
    meta = {
        "removed": region.text,
        "box": region.box,
        "confidence": round(region.confidence, 1),
        "inpaint_backend": used,
    }
    _log.info("edittext: removed %r (inpaint=%s)", meta["removed"], used)
    return erased, meta


def remove_all_text(img: Any, *, inpaint_backend: str = "auto",
                    lang: str = "eng",
                    min_confidence: float = 30.0) -> tuple[Any, dict[str, Any]]:
    """Scrub every detected text region (watermark / PII cleanup).

    Erases regions with OCR confidence ≥ ``min_confidence``, largest
    first so overlapping inpaints stay stable. Returns (image, metadata
    with per-region boxes).
    """
    regions = detect_text(img, lang=lang)
    kept = [r for r in regions if r.confidence >= min_confidence]
    out = img
    erased_boxes: list[tuple[int, int, int, int]] = []
    backends: list[str] = []
    for region in sorted(kept, key=lambda r: (
            -(r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))):
        out, used = _inpaint_region(out, region.box, backend=inpaint_backend)
        erased_boxes.append(region.box)
        backends.append(used)
    meta = {"regions_erased": len(erased_boxes), "boxes": erased_boxes,
            "inpaint_backends": sorted(set(backends))}
    _log.info("edittext: scrubbed %d text region(s)", len(erased_boxes))
    return out, meta


def op_edittext(img: Any, old_text: str, new_text: str, **kwargs: Any) -> Any:
    """Chain op: text replacement in photos. Registered as ``edittext``.

    ``img`` is the current chain image; ``old_text``/``new_text`` are
    required params.  Returns the edited image (metadata is logged).
    """
    out, meta = replace_text(img, old_text, new_text, **kwargs)
    _log.info("edittext: replaced %r with %r (inpaint=%s)",
              meta["found"], new_text, meta["inpaint_backend"])
    return out


def edittext_intent_patterns() -> list[str]:
    """Human-readable pattern list (for help text)."""
    return [
        'replace "OLD" with "NEW" in this image',
        "change the text \"OLD\" to \"NEW\"",
    ]
