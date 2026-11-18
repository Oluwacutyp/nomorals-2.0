"""Composed object removal — the Cleanup.pictures loop, Devon-native.

One loop — mask, fill, compare, download — over organs Devon already
has, and nothing here executes an edit on its own:

- the mask comes from the §1.1d ladder (:mod:`nomorals.media.imggen.masks`
  — SAM → GrabCut → flood → box+feather) via a serializable
  :class:`~nomorals.media.imggen.masks.MaskRef` recipe;
- the description ("remove the man in red", "remove the watermark",
  "remove the person on the left") is turned into a box/point by
  :func:`locate_description`, built from Devon's own vision rungs
  (tesseract OCR boxes, Haar faces, contour marks) plus numpy-only
  colour-region and background-contrast rungs — no new model, no
  invented coordinates;
- the edit itself is an :class:`~nomorals.media.imggen.document.ImageDocument`
  inpaint op (:meth:`ImageDocument.remove_object`), executed by the
  document's own executor on the inpaint rung
  (:func:`nomorals.media.imggen.edit.inpaint_auto` — neural when a
  pipeline is handed in, classical CPU tiers otherwise), with
  ``preserve_outside`` compositing the source pixels back outside
  the mask inside the op;
- the Studio (:meth:`nomorals.media.imggen.studio.Studio.remove_object`)
  builds that document, renders it, and saves it with the rung
  provenance in the sidecar — the document stays open for the Studio
  to continue from (more ops, undo, replay).

Honesty rules, enforced in the op rather than documented elsewhere:

- the op record states which mask rung and which inpaint rung served
  it — a mask never travels without its provenance;
- a fill covering >= :data:`SYNTHESISED_FRACTION` of the image is
  labelled synthesised content, never a silent retouch;
- a description no rung can locate refuses in words naming every
  rung that was tried, instead of removing the wrong thing.
"""

from __future__ import annotations

import importlib.util
import io
import re

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

from . import ImgGenError

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise ImgGenError(
            "nomorals.media.imggen.cleanup is unavailable: Pillow is "
            "not installed (pip install Pillow)")


try:
    NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
except ImportError:
    NUMPY_AVAILABLE = False


def _require_numpy() -> None:
    if not NUMPY_AVAILABLE:
        raise ImgGenError(
            "nomorals.media.imggen.cleanup is unavailable: numpy is "
            "not installed (pip install numpy)")

__all__ = [
    "SYNTHESISED_FRACTION",
    "parse_description",
    "locate_description",
    "default_locator",
    "box_predictor_from_locator",
    "mask_fraction",
    "remove_object_document",
]

#: A fill covering this fraction of the image (or more) is labelled
#: synthesised content — the Studio honesty rule, not a silent retouch.
SYNTHESISED_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------

_VERBS = {"remove", "erase", "delete", "take", "get", "rid", "clean",
          "up", "out", "cut", "drop", "please"}
_ARTICLES = {"the", "a", "an", "this", "that", "it"}
_TEXT_WORDS = {"watermark", "text", "word", "words", "caption", "logo",
               "lettering", "letters", "writing"}
_PERSON_WORDS = {"person", "man", "woman", "boy", "girl", "guy", "lady",
                 "face", "people", "someone", "somebody", "child", "kid"}
_COLOURS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 180, 0),
    "yellow": (255, 220, 0), "orange": (255, 140, 0),
    "purple": (150, 0, 200), "pink": (255, 150, 200),
    "brown": (140, 90, 40), "black": (0, 0, 0),
    "white": (255, 255, 255), "grey": (128, 128, 128),
    "gray": (128, 128, 128), "cyan": (0, 200, 255),
    "magenta": (255, 0, 255),
}
# Spatial words → fraction rects (x0, y0, x1, y1); several words
# intersect, so "top-left" narrows to the top-left quadrant.
_SPATIAL = {
    "left": (0.0, 0.0, 0.34, 1.0),
    "right": (0.66, 0.0, 1.0, 1.0),
    "center": (0.33, 0.0, 0.67, 1.0),
    "centre": (0.33, 0.0, 0.67, 1.0),
    "middle": (0.33, 0.0, 0.67, 1.0),
    "top": (0.0, 0.0, 1.0, 0.34),
    "bottom": (0.0, 0.66, 1.0, 1.0),
}


def _intersect(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), min(a[3], b[3]))


def parse_description(description: str) -> dict:
    """Split a remove-what description into a locator intent.

    Returns ``{"kind", "colour", "spatial", "target_words", "quoted"}``.
    ``kind`` is "text" | "person" | "colour" | "object"; ``colour`` is
    ``(name, rgb)`` or None; ``spatial`` is a fraction rect or None;
    ``target_words`` are the content words (verbs, articles, kind
    words and spatial words stripped); ``quoted`` is text in quotes.
    """
    text = (description or "").strip()
    quoted = [q for pair in
              re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
              for q in pair if q]
    bare = re.sub(r'"[^"]*"|\'[^\']*\'', " ", text)
    words = re.findall(r"[a-z]+", bare.lower())
    spatial = None
    for w in words:
        if w in _SPATIAL:
            spatial = (_intersect(spatial, _SPATIAL[w])
                       if spatial is not None else _SPATIAL[w])
    colour = None
    for w in words:
        if w in _COLOURS:
            colour = (w, _COLOURS[w])
            break
    if quoted or any(w in _TEXT_WORDS for w in words):
        kind = "text"
    elif any(w in _PERSON_WORDS for w in words):
        kind = "person"
    elif colour is not None:
        kind = "colour"
    else:
        kind = "object"
    skip = (_VERBS | _ARTICLES | _TEXT_WORDS | _PERSON_WORDS
            | set(_COLOURS) | set(_SPATIAL))
    target = [w for w in words if w not in skip]
    return {"kind": kind, "colour": colour, "spatial": spatial,
            "target_words": target, "quoted": quoted}


# ---------------------------------------------------------------------------
# Numpy region helpers (pure numpy — no cv2, no torch)
# ---------------------------------------------------------------------------

def _flood_component(mask, seed):
    """The connected component of ``mask`` containing ``seed``.

    Vectorized wavefront BFS over 4-neighbours — the same shape as
    :func:`nomorals.media.imggen.masks.point_flood_mask`, but over an
    arbitrary boolean mask instead of a colour tolerance.
    """
    import numpy as np

    h, w = mask.shape
    sy, sx = int(seed[1]), int(seed[0])
    region = np.zeros((h, w), dtype=bool)
    if not (0 <= sx < w and 0 <= sy < h) or not mask[sy, sx]:
        return region
    visited = np.zeros((h, w), dtype=bool)
    region[sy, sx] = True
    visited[sy, sx] = True
    fy = np.array([sy])
    fx = np.array([sx])
    while fy.size:
        nys = np.concatenate([fy - 1, fy + 1, fy, fy])
        nxs = np.concatenate([fx, fx, fx - 1, fx + 1])
        inb = ((nys >= 0) & (nys < h) & (nxs >= 0) & (nxs < w))
        nys, nxs = nys[inb], nxs[inb]
        if not nys.size:
            break
        fresh = ~visited[nys, nxs]
        nys, nxs = nys[fresh], nxs[fresh]
        if not nys.size:
            break
        _, uniq = np.unique(nys * w + nxs, return_index=True)
        nys, nxs = nys[uniq], nxs[uniq]
        take = mask[nys, nxs]
        visited[nys, nxs] = True
        ty, tx = nys[take], nxs[take]
        region[ty, tx] = True
        fy, fx = ty, tx
    return region


def _densest_seed(mask, region=None):
    """A seed pixel inside the densest 8x8 cell of ``mask``.

    ``region`` is an optional fraction rect the seed must fall in.
    Returns ``(x, y)`` or None when the mask is empty in the region.
    """
    import numpy as np

    h, w = mask.shape
    if region is not None:
        rx0 = max(0, int(region[0] * w))
        ry0 = max(0, int(region[1] * h))
        rx1 = min(w, max(rx0 + 1, int(region[2] * w)))
        ry1 = min(h, max(ry0 + 1, int(region[3] * h)))
        work = np.zeros_like(mask)
        work[ry0:ry1, rx0:rx1] = mask[ry0:ry1, rx0:rx1]
    else:
        work = mask
    gh, gw = 8, 8
    ph, pw = (-h) % gh, (-w) % gw
    padded = np.pad(work, ((0, ph), (0, pw)))
    cells = padded.reshape(gh, (h + ph) // gh,
                           gw, (w + pw) // gw).sum(axis=(1, 3))
    if not cells.any():
        return None
    cy, cx = np.unravel_index(int(np.argmax(cells)), cells.shape)
    ch, cw = (h + ph) // gh, (w + pw) // gw
    y0, x0 = int(cy) * ch, int(cx) * cw
    sub = work[y0:y0 + ch, x0:x0 + cw]
    ys, xs = np.nonzero(sub)
    if not ys.size:
        return None
    i = len(ys) // 2
    return (int(x0 + xs[i]), int(y0 + ys[i]))


def _bbox_of(region, pad: int = 4, size=None):
    """Bounding box of a boolean region, padded and clipped."""
    import numpy as np

    ys, xs = np.nonzero(region)
    if not xs.size:
        return None
    x0, y0 = int(xs.min()) - pad, int(ys.min()) - pad
    x1, y1 = int(xs.max()) + pad + 1, int(ys.max()) + pad + 1
    if size is not None:
        w, h = size
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
    return (x0, y0, x1, y1)


def _box1000_to_px(box1000, size):
    w, h = size
    x0 = round(box1000["x"] / 1000 * w)
    y0 = round(box1000["y"] / 1000 * h)
    x1 = round((box1000["x"] + box1000["w"]) / 1000 * w)
    y1 = round((box1000["y"] + box1000["h"]) / 1000 * h)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


def _centre_in(box, region, size):
    """Is the centre of pixel ``box`` inside fraction rect ``region``?"""
    w, h = size
    cx = ((box[0] + box[2]) / 2) / w
    cy = ((box[1] + box[3]) / 2) / h
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


# ---------------------------------------------------------------------------
# Description → box/point
# ---------------------------------------------------------------------------

def locate_description(image, description: str, *, ocr=None, faces=None,
                       elements=None) -> dict:
    """Locate what ``description`` names in ``image``.

    Returns ``{"box", "point", "how", "detail"}`` — ``box`` is
    ``(x0, y0, x1, y1)`` pixels or None, ``point`` is ``(x, y)`` or
    None, ``how`` names the rung that located it. The injected
    callables are the vision rungs, each taking PNG ``bytes``:
    ``ocr`` → :func:`vision.native.document_layout` shape (``lines``
    with ``bbox_1000`` + ``text``), ``faces`` →
    :func:`vision.native.detect_faces` shape (``boxes`` in 0-1000),
    ``elements`` → :func:`vision.native.screen_marks` shape
    (``elements`` with ``bbox_1000`` + ``text`` + ``source``).
    A rung left as None is skipped and named in the refusal.

    Raises :class:`ImgGenError` naming every rung tried when nothing
    can locate the description — the caller removes nothing rather
    than the wrong thing.
    """
    _require_pil()
    _require_numpy()
    import numpy as np

    if not (description or "").strip():
        raise ImgGenError(
            "locate_description needs a description — "
            "what should be removed?")
    intent = parse_description(description)
    kind = intent["kind"]
    region = intent["spatial"] or (0.0, 0.0, 1.0, 1.0)
    w, h = image.size
    tried: list[str] = []
    data: bytes | None = None

    def _bytes() -> bytes:
        nonlocal data
        if data is None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
        return data

    def _union(boxes):
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        return (max(0, x0 - 3), max(0, y0 - 3),
                min(w, x1 + 3), min(h, y1 + 3))

    # -- text: OCR line boxes -------------------------------------------
    if kind == "text":
        targets = ([q.lower() for q in intent["quoted"]]
                   or intent["target_words"])
        if ocr is not None:
            try:
                layout = ocr(_bytes())
                lines = layout.get("lines") or []
                if targets:
                    matched = [
                        _box1000_to_px(ln["bbox_1000"], (w, h))
                        for ln in lines
                        if any(t in (ln.get("text") or "").lower()
                               for t in targets)]
                else:
                    matched = [_box1000_to_px(ln["bbox_1000"], (w, h))
                               for ln in lines]
                matched = [b for b in matched
                           if _centre_in(b, region, (w, h))]
                if matched:
                    box = _union(matched)
                    # Box only — no point: a flood seed at the box
                    # centre could land on background between letters
                    # and grow the mask over the page instead of the
                    # text. The box tier (or GrabCut/SAM on the box)
                    # is the right mask for text.
                    return {"box": box, "point": None,
                            "how": "ocr-text",
                            "detail": (f"OCR matched {len(matched)} "
                                       "text line(s)")}
                tried.append("ocr-text: no text line matched "
                             + (f"{targets!r}" if targets
                                else "any text"))
            except Exception as exc:  # noqa: BLE001 - rung is best-effort
                tried.append(f"ocr-text: unavailable ({exc})")
        else:
            tried.append("ocr-text: no OCR rung wired "
                         "(tesseract unavailable)")
        if elements is not None:
            try:
                marks = elements(_bytes())
                els = [e for e in (marks.get("elements") or [])
                       if (e.get("text") or "").strip()]
                if targets:
                    els = [e for e in els
                           if any(t in (e.get("text") or "").lower()
                                  for t in targets)]
                boxes = [_box1000_to_px(e["bbox_1000"], (w, h))
                         for e in els
                         if _centre_in(
                             _box1000_to_px(e["bbox_1000"], (w, h)),
                             region, (w, h))]
                if boxes:
                    box = _union(boxes)
                    return {"box": box, "point": None,
                            "how": "screen-marks-text",
                            "detail": (f"screen marks matched "
                                       f"{len(boxes)} text element(s)")}
                tried.append("screen-marks-text: no text element matched")
            except Exception as exc:  # noqa: BLE001 - rung is best-effort
                tried.append(f"screen-marks-text: unavailable ({exc})")
        else:
            tried.append("screen-marks-text: no marks rung wired")
        raise ImgGenError(
            f"could not locate {description!r} in the image — "
            f"tried: {'; '.join(tried)}")

    # -- person: faces, then contours, then contrast --------------------
    if kind == "person":
        if faces is not None:
            try:
                found = faces(_bytes())
                boxes = [_box1000_to_px(b, (w, h))
                         for b in (found.get("boxes") or [])]
                boxes = [b for b in boxes
                         if _centre_in(b, region, (w, h))]
                if len(boxes) == 1:
                    b = boxes[0]
                    return {"box": None,
                            "point": ((b[0] + b[2]) // 2,
                                      (b[1] + b[3]) // 2),
                            "how": "face-point",
                            "detail": ("one frontal face in the region — "
                                       "point-seeded segmentation; the "
                                       "face box alone is not the person")}
                if len(boxes) > 1:
                    tried.append(
                        f"faces: {len(boxes)} faces in the region — "
                        "say which one (left, center, right)")
                else:
                    tried.append("faces: no face found in the region")
            except Exception as exc:  # noqa: BLE001 - rung is best-effort
                tried.append(f"faces: unavailable ({exc})")
        else:
            tried.append("faces: no face rung wired "
                         "(OpenCV unavailable)")

    # -- colour: numpy colour-region ------------------------------------
    if kind == "colour":
        _name, rgb = intent["colour"]
        arr = np.asarray(image.convert("RGB")).astype(np.int16)
        close = (np.abs(arr - np.array(rgb, dtype=np.int16))
                 .sum(axis=2) <= 120)
        seed = _densest_seed(close, region)
        if seed is not None:
            comp = _flood_component(close, seed)
            box = _bbox_of(comp, size=(w, h))
            n = int(comp.sum())
            if box is not None and 20 <= n <= 0.6 * w * h:
                return {"box": box, "point": seed,
                        "how": "colour-region",
                        "detail": (f"largest connected region close to "
                                   f"{_name} ({n} px)")}
        tried.append(f"colour-region: no {_name} region found")
        raise ImgGenError(
            f"could not locate {description!r} in the image — "
            f"tried: {'; '.join(tried)}")

    # -- person/object: contour marks, then background-contrast ---------
    if elements is not None:
        try:
            marks = elements(_bytes())
            els = marks.get("elements") or []
            contour = [e for e in els
                       if (e.get("source") or "") == "contour"]
            pool = contour or els
            boxes = [(_box1000_to_px(e["bbox_1000"], (w, h)), e)
                     for e in pool]
            boxes = [(b, e) for b, e in boxes
                     if _centre_in(b, region, (w, h))]
            if boxes:
                boxes.sort(key=lambda be: (be[0][2] - be[0][0])
                           * (be[0][3] - be[0][1]), reverse=True)
                b, _e = boxes[0]
                return {"box": b,
                        "point": ((b[0] + b[2]) // 2,
                                  (b[1] + b[3]) // 2),
                        "how": "contour-box",
                        "detail": ("largest contour region in the "
                                   "region (approximate)")}
            tried.append("contour-box: no contour region in the region")
        except Exception as exc:  # noqa: BLE001 - rung is best-effort
            tried.append(f"contour-box: unavailable ({exc})")
    else:
        tried.append("contour-box: no marks rung wired")
    try:
        arr = np.asarray(image.convert("RGB")).astype(np.int16)
        border = np.concatenate([
            arr[:10, :, :].reshape(-1, 3), arr[-10:, :, :].reshape(-1, 3),
            arr[:, :10, :].reshape(-1, 3), arr[:, -10:, :].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        fg = (np.abs(arr - bg).sum(axis=2) > 60)
        seed = _densest_seed(fg, region)
        if seed is not None:
            comp = _flood_component(fg, seed)
            box = _bbox_of(comp, size=(w, h))
            n = int(comp.sum())
            if box is not None and 20 <= n <= 0.6 * w * h:
                return {"box": box, "point": seed,
                        "how": "background-contrast",
                        "detail": (f"largest region distinct from the "
                                   f"border background ({n} px, "
                                   "approximate)")}
        tried.append("background-contrast: no distinct region found")
    except Exception as exc:  # noqa: BLE001 - rung is best-effort
        tried.append(f"background-contrast: unavailable ({exc})")
    raise ImgGenError(
        f"could not locate {description!r} in the image — "
        f"tried: {'; '.join(tried)}")


def default_locator(image, description: str) -> dict:
    """locate_description with Devon's own vision rungs wired in.

    OCR (tesseract layout), faces (OpenCV Haar) and screen marks
    (OCR + contours) come from :mod:`nomorals.vision.native`, imported
    lazily — a rung whose dependency is missing degrades to "tried and
    unavailable" inside :func:`locate_description`, never to a crash.
    Colour and background-contrast rungs need only numpy + PIL.
    """
    try:
        from ...vision import native as _native
    except Exception as exc:
        _native = None
        _import_error = str(exc)
    else:
        _import_error = ""

    def _ocr(data: bytes):
        if _native is None:
            raise ImgGenError(
                f"vision.native unavailable ({_import_error})")
        return _native.document_layout(data)

    def _faces(data: bytes):
        if _native is None:
            raise ImgGenError(
                f"vision.native unavailable ({_import_error})")
        return _native.detect_faces(data)

    def _elements(data: bytes):
        if _native is None:
            raise ImgGenError(
                f"vision.native unavailable ({_import_error})")
        return _native.screen_marks(data)

    return locate_description(image, description, ocr=_ocr, faces=_faces,
                              elements=_elements)


def box_predictor_from_locator(locator) -> object:
    """Adapt a ``(image, text) -> locate`` result to the MaskRef
    ``box_predictor`` contract ``(image, text) -> box | None``.

    A location that only yields a point (a face) returns None here —
    the point already seeds the ladder, and a missing box must not
    fail a mask the point alone can grow.
    """
    def _predict(image, text):
        found = locator(image, text)
        return found.get("box")
    return _predict


# ---------------------------------------------------------------------------
# Mask coverage
# ---------------------------------------------------------------------------

def mask_fraction(mask) -> float:
    """Fraction of the image the mask covers (alpha >= 0.5)."""
    _require_numpy()
    import numpy as np

    a = np.asarray(mask.convert("L")).astype(np.float64) / 255.0
    return float((a >= 0.5).mean())


_INPAINT_METHODS = ("auto", "telea", "ns", "numpy", "neural")


def remove_object_document(image_path, *, mask=None, description=None,
                           prompt: str = "", method: str = "auto",
                           feather: int = 8, locator=None):
    """Build the remove-object document for ``image_path``.

    The inpaint op carries the MaskRef recipe (description + located
    box/point, or the explicit mask), so the document replays
    deterministically and undoes like any image-document op. A PIL
    mask cannot ride in a document — save it and pass its path.
    """
    from .document import ImageDocument

    _require_pil()
    if method not in _INPAINT_METHODS:
        raise ImgGenError(
            f"unknown inpaint method {method!r}; "
            "use auto|neural|telea|ns|numpy")
    if mask is not None and Image is not None \
            and isinstance(mask, Image.Image):
        raise ImgGenError(
            "remove_object_document cannot embed mask pixels — "
            "save the mask and pass its path")
    locator = locator or default_locator
    doc = ImageDocument(image_path)
    if mask is not None:
        if description:
            # An explicit mask wins; the dropped description is
            # recorded on the document, not swallowed silently.
            doc.notes.append("explicit mask given — description ignored")
        return doc.remove_object(mask=mask, prompt=prompt,
                                 method=method, feather=feather)
    if description:
        with Image.open(image_path) as _im:
            img = _im.convert("RGB")
            found = locator(img, description)
        return doc.remove_object(
            description, prompt=prompt, method=method, feather=feather,
            box=found.get("box"), point=found.get("point"))
    raise ImgGenError(
        "remove_object_document needs a mask or a description — "
        "what should be removed?")
