"""Local slider-captcha solver: she drags the puzzle herself, in-session.

The captcha estate (:mod:`nomorals.tools.captcha`) detects token
captchas and delegates them to a paid service or the owner. Slider
puzzles (TikTok's "Drag the slider to fit the puzzle", GeeTest) never
fit that shape: there is no token to buy — the solve is a *physical
act* in the live page. This module is that act, in three layers:

1. **Detection** — :func:`detect_slider_challenge` recognises slider
   markup in page HTML (ByteDance/TikTok verify, GeeTest slider,
   generic drag-puzzle structure) and returns the selectors the
   solver needs. It lives here (browser layer), not in
   ``tools/captcha.py``.
2. **Vision** — :func:`detect_gap` finds the notch in the puzzle
   image: template matching on edge maps for the two-image shape
   (background + separate piece image), outline detection for the
   single-image shape (cutout in one image). The math is numpy; image
   decode probes PIL, then OpenCV. When the numeric stack is absent
   the solver reports itself *unavailable* — it never guesses a drag.
   Below :data:`MIN_GAP_CONFIDENCE` the answer is an honest "not
   found", never a hopeful drag.
3. **Execution** — a human-shaped drag (ease-in, speed variance,
   overshoot + settle, vertical jitter; seed-deterministic) performed
   through the tab's real mouse path in the same session that hit the
   wall, with the outcome read back from the page. One retry at a
   re-detected offset; a still-present slider is a failure, reported
   as one.

Pipeline position: session/cookie state (the caller's flow) → THIS
solver → paid service backend → owner takeover. Wired as a strategy in
:mod:`nomorals.browser.challenge`, a stage in
:mod:`nomorals.tools.captcha_flow`, and registered as the
``local_slider`` backend in the captcha backend registry (via
``register_backend`` — ``tools/captcha.py`` itself is untouched).
Attempts are paced by the shared captcha :class:`SolverRateLimiter`.

Tab protocol (duck-typed; :class:`nomorals.browser.service.RenderedTab`
via the takeover adapter provides it): ``html()``, ``url``,
``element_image_bytes(selector)``, ``element_box(selector)``,
``mouse_drag_path(points)`` where points are ``(x, y, delay_ms)``
tuples in viewport CSS pixels.
"""
from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..tools.captcha import (
    CaptchaBackend,
    CaptchaKind,
    SolveResult,
    rate_limiter,
    register_backend,
)

_log = get_logger(__name__)

__all__ = [
    "SLIDER_KIND",
    "MIN_GAP_CONFIDENCE",
    "SliderChallengeInfo",
    "GapResult",
    "DragPoint",
    "SliderOutcome",
    "detect_slider_challenge",
    "vision_status",
    "detect_gap",
    "synthesize_trajectory",
    "LocalSliderSolver",
    "LocalSliderBackend",
    "solve_slider_on_tab",
    "solve_slider_kind",
    "try_local_slider_solve",
    "register_local_backend",
    "local_slider_status",
]

#: CaptchaChallenge kind for slider-class challenges. Plain string,
#: like every CaptchaKind value; tools/captcha.py needs no edit.
SLIDER_KIND = "slider"

#: Below this gap-detection confidence the solver refuses to drag.
MIN_GAP_CONFIDENCE = 0.60

#: Per-challenge drag cap: the first drag + exactly one retry.
MAX_DRAGS_PER_CHALLENGE = 2


# ── vision stack (probe-and-degrade, the repo's zero-dep posture) ────────

def _probe(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001 - absence is data, not an error
        return False


def vision_status() -> dict[str, Any]:
    """What the local vision stack can do right now, honestly."""
    have_numpy = _probe("numpy")
    have_pil = _probe("PIL")
    have_cv2 = _probe("cv2")
    available = bool(have_numpy and (have_pil or have_cv2))
    missing = []
    if not have_numpy:
        missing.append("numpy")
    if not (have_pil or have_cv2):
        missing.append("Pillow or OpenCV (image decode)")
    detail = ("numpy edge/template matching ready"
              if available else
              "local slider solving unavailable — missing: "
              + ", ".join(missing))
    return {"available": available, "numpy": have_numpy,
            "pil": have_pil, "cv2": have_cv2, "detail": detail}


def _decode_image(data: bytes) -> tuple[Any, Any] | None:
    """Image bytes -> (rgb uint8 HxWx3, alpha uint8 HxW | None).

    PIL first (PNG/JPEG everywhere), OpenCV as the fallback decoder.
    ``None`` when no decoder exists or the bytes are not an image.
    """
    if not data:
        return None
    try:
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "LA", "PA"):
            rgba = np.asarray(img.convert("RGBA"))
            return rgba[:, :, :3].copy(), rgba[:, :, 3].copy()
        return np.asarray(img.convert("RGB")).copy(), None
    except Exception:  # noqa: BLE001 - try the next decoder
        pass
    try:
        import cv2
        import numpy as np

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1), None
        if img.shape[2] == 4:
            return (cv2.cvtColor(img, cv2.COLOR_BGRA2RGB),
                    img[:, :, 3].copy())
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
    except Exception:  # noqa: BLE001 - no decoder: honest None
        return None


def _edges(gray: Any) -> Any:
    """Sobel gradient magnitude of a float grayscale image, ~0..1."""
    import numpy as np

    g = gray.astype(np.float64)
    pad = np.pad(g, 1, mode="edge")
    gx = (-pad[:-2, :-2] - 2 * pad[1:-1, :-2] - pad[2:, :-2]
          + pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
    gy = (-pad[:-2, :-2] - 2 * pad[:-2, 1:-1] - pad[:-2, 2:]
          + pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
    return np.clip(np.hypot(gx, gy) / (4.0 * 255.0), 0.0, 1.0)


# ── gap detection ────────────────────────────────────────────────────────

@dataclass
class GapResult:
    found: bool
    gap_x: float | None = None       # content left edge, bg image px
    piece_x: float | None = None     # piece content offset in piece img
    confidence: float = 0.0
    shape: str = ""                  # two_image | single_image
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"found": self.found, "gap_x": self.gap_x,
                "piece_x": self.piece_x, "confidence": round(
                    self.confidence, 4), "shape": self.shape,
                "detail": self.detail}


def _chamfer_distance(edges_bin: Any) -> Any:
    """Distance (px) from every pixel to the nearest edge pixel.

    Two-pass chamfer transform (1 / sqrt(2) weights) — close enough
    to Euclidean at these sizes, and pure numpy.
    """
    import numpy as np

    h, w = edges_bin.shape
    big = 1e6
    d = np.where(edges_bin, 0.0, big)
    diag = math.sqrt(2.0)
    for i in range(h):
        row = d[i]
        if i:
            up = d[i - 1]
            row = np.minimum(row, up + 1.0)
            row[1:] = np.minimum(row[1:], up[:-1] + diag)
            row[:-1] = np.minimum(row[:-1], up[1:] + diag)
        row[1:] = np.minimum(row[1:], row[:-1] + 1.0)
        d[i] = row
    for i in range(h - 2, -1, -1):
        row = d[i]
        down = d[i + 1]
        row = np.minimum(row, down + 1.0)
        row[1:] = np.minimum(row[1:], down[:-1] + diag)
        row[:-1] = np.minimum(row[:-1], down[1:] + diag)
        row[:-1] = np.minimum(row[:-1], row[1:] + 1.0)
        d[i] = row
    return d


def _match_piece(bg_edges: Any, piece_rgb: Any, piece_alpha: Any
                 ) -> GapResult:
    """Chamfer-match the piece's outline against the background.

    The notch in the background is the piece's silhouette drawn as an
    outline. Score = how close the template's outline pixels land to
    real background edges (gaussian over the chamfer distance),
    weighted by edge strength. Unlike magnitude-overlap correlation,
    a merely *busy* patch of photo cannot fake this: outline pixels
    far from any true edge score ~0 wherever they land. Confidence is
    the peak score discounted by prominence (a second, distant peak
    of similar height means the match is ambiguous).
    """
    import numpy as np

    bh, bw = bg_edges.shape
    p_gray = piece_rgb.astype(np.float64).mean(axis=2)
    p_edges = _edges(p_gray)
    if piece_alpha is not None:
        mask = piece_alpha > 100
    else:
        mask = p_edges > 0.05
    if not mask.any():
        return GapResult(False, detail="piece image has no visible "
                                       "content to match")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    tpl_e = p_edges[y0:y1, x0:x1]
    mbool = mask[y0:y1, x0:x1]
    # outline band: silhouette dilated minus eroded (±1px)
    eroded = mbool.copy()
    eroded[1:, :] &= mbool[:-1, :]
    eroded[:-1, :] &= mbool[1:, :]
    eroded[:, 1:] &= mbool[:, :-1]
    eroded[:, :-1] &= mbool[:, 1:]
    dilated = mbool.copy()
    dilated[1:, :] |= mbool[:-1, :]
    dilated[:-1, :] |= mbool[1:, :]
    dilated[:, 1:] |= mbool[:, :-1]
    dilated[:, :-1] |= mbool[:, 1:]
    band = dilated & ~eroded
    support = band & (tpl_e > 0.08)
    if int(support.sum()) < 40:
        support = mbool & (tpl_e > 0.10)
    if int(support.sum()) < 20:
        return GapResult(False, detail="piece outline too faint to "
                                       "match honestly")
    th, tw = tpl_e.shape
    if th > bh or tw > bw:
        return GapResult(False, detail="piece is larger than the "
                                       "background image")
    oh, ow = bh - th + 1, bw - tw + 1
    prox = np.exp(-(_chamfer_distance(bg_edges > 0.16) ** 2) / 2.0)
    weights = np.where(support, tpl_e, 0.0)
    total_w = float(weights.sum())
    score = np.zeros((oh, ow), dtype=np.float64)
    ys, xs = np.where(support)
    for dy, dx in zip(ys.tolist(), xs.tolist()):
        score += weights[dy, dx] * prox[dy:dy + oh, dx:dx + ow]
    score /= max(total_w, 1e-9)
    peak_idx = int(np.argmax(score))
    py, px = divmod(peak_idx, ow)
    peak = float(score[py, px])
    masked = score.copy()
    r = max(6, tw // 2)
    masked[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = 0.0
    second = float(masked.max()) if masked.size else 0.0
    prominence = max(0.0, min(1.0, (peak - second) / max(peak, 1e-9)))
    confidence = peak * (0.70 + 0.30 * prominence)
    found = confidence >= MIN_GAP_CONFIDENCE
    # px is where the cropped template (the piece's content) lands in
    # the background — that IS the gap's content-left edge. x0 is the
    # content's offset inside the piece image itself.
    return GapResult(
        found=found, gap_x=float(px) if found else None,
        piece_x=float(x0), confidence=confidence, shape="two_image",
        detail=(f"outline chamfer match {peak:.2f} at x={px}, "
                f"second {second:.2f}"
                if found else
                f"outline match too weak: peak {peak:.2f}, "
                f"confidence {confidence:.2f}"))


def _outline_gap(bg_rgb: Any) -> GapResult:
    """Single-image shape: find the notch by its drawn outline.

    The cutout in these widgets is a piece-shaped region with a
    visible border. Score candidate rectangles by edge coverage AND
    edge strength along their perimeter — blur/noise wash both out,
    which is what keeps degraded puzzles honestly unfound.
    """
    import numpy as np

    gray = bg_rgb.astype(np.float64).mean(axis=2)
    e = _edges(gray)
    h, w = e.shape
    # 3x3 max-filtered edges: the Sobel peak sits 1px beside the drawn
    # line as often as on it, so coverage and strength both sample
    # this tolerance map, not the raw one.
    es = e.copy()
    es[1:, :] = np.maximum(es[1:, :], e[:-1, :])
    es[:-1, :] = np.maximum(es[:-1, :], e[1:, :])
    es[:, 1:] = np.maximum(es[:, 1:], e[:, :-1])
    es[:, :-1] = np.maximum(es[:, :-1], e[:, 1:])
    strong = es > 0.20
    dil = strong.copy()
    dil[1:, :] |= strong[:-1, :]
    dil[:-1, :] |= strong[1:, :]
    dil[:, 1:] |= strong[:, :-1]
    dil[:, :-1] |= strong[:, 1:]
    col = dil.sum(axis=0).astype(np.float64)
    row = dil.sum(axis=1).astype(np.float64)

    def _peaks(profile: Any, limit: int) -> list[int]:
        idx = np.argsort(profile)[::-1]
        picked: list[int] = []
        for i in idx.tolist():
            if profile[i] < 10:
                break
            if all(abs(i - p) > 3 for p in picked):
                picked.append(i)
            if len(picked) >= limit:
                break
        return sorted(picked)

    xs = _peaks(col, 11)
    ys = _peaks(row, 11)
    best: tuple[float, int, int, int, int] | None = None
    for x1 in xs:
        for x2 in xs:
            if not (24 <= x2 - x1 <= 130):
                continue
            for y1 in ys:
                for y2 in ys:
                    if not (24 <= y2 - y1 <= 130):
                        continue
                    if abs((x2 - x1) - (y2 - y1)) > 18:
                        continue
                    top = dil[y1, x1:x2 + 1]
                    bot = dil[y2, x1:x2 + 1]
                    lft = dil[y1:y2 + 1, x1]
                    rgt = dil[y1:y2 + 1, x2]
                    cov = float(np.concatenate(
                        [top, bot, lft, rgt]).mean())
                    strength = float(np.concatenate([
                        es[y1, x1:x2 + 1], es[y2, x1:x2 + 1],
                        es[y1:y2 + 1, x1], es[y1:y2 + 1, x2]]).mean())
                    score = cov * min(1.0, strength / 0.42)
                    if best is None or score > best[0]:
                        best = (score, x1, y1, x2, y2)
    if best is None:
        return GapResult(False, shape="single_image",
                         detail="no outline candidate in image")
    score, x1, y1, x2, y2 = best
    # Refine the left edge against RAW edge columns: the tolerance
    # maps above spread the drawn line over ±3px, and the score's
    # argmax can sit on the spread's shoulder. The true line is the
    # strongest raw column near the winner.
    lo = max(0, x1 - 3)
    hi = min(e.shape[1] - 1, x1 + 3)
    colsum = e[y1:y2 + 1, lo:hi + 1].sum(axis=0)
    x1 = lo + int(np.argmax(colsum))
    found = score >= MIN_GAP_CONFIDENCE
    return GapResult(
        found=found, gap_x=float(x1) if found else None, piece_x=None,
        confidence=float(score), shape="single_image",
        detail=(f"outline at x={x1}..{x2}, y={y1}..{y2}, "
                f"score {score:.2f}" if found else
                f"best outline candidate too weak: {score:.2f}"))


def detect_gap(background: bytes, piece: bytes = b"") -> GapResult:
    """Locate the notch. ``piece`` empty -> single-image shape.

    Never raises on bad input; an undecodable or ambiguous image is
    an honest ``found=False``.
    """
    if not vision_status()["available"]:
        return GapResult(False, detail=vision_status()["detail"])
    decoded_bg = _decode_image(background)
    if decoded_bg is None:
        return GapResult(False, detail="background image undecodable "
                                       "(no decoder or bad bytes)")
    bg_rgb, _ = decoded_bg
    import numpy as np

    bg_edges = _edges(bg_rgb.astype(np.float64).mean(axis=2))
    if piece:
        decoded_piece = _decode_image(piece)
        if decoded_piece is None:
            return GapResult(False, detail="piece image undecodable")
        piece_rgb, piece_alpha = decoded_piece
        return _match_piece(bg_edges, piece_rgb, piece_alpha)
    return _outline_gap(bg_rgb)


# ── slider markup detection (browser layer, from page HTML) ─────────────

@dataclass
class SliderChallengeInfo:
    """A slider-class wall, with the selectors the solver needs."""

    kind: str = SLIDER_KIND
    vendor: str = "generic"          # tiktok | geetest | generic
    container_selector: str = ""
    background_selector: str = ""    # img/canvas holding the puzzle
    piece_selector: str = ""         # img/canvas holding the piece
    handle_selector: str = ""        # the drag handle element
    shape: str = "two_image"         # two_image | single_image
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "vendor": self.vendor,
                "container_selector": self.container_selector,
                "background_selector": self.background_selector,
                "piece_selector": self.piece_selector,
                "handle_selector": self.handle_selector,
                "shape": self.shape, "evidence": self.evidence}


_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>", re.S)
_CLASS_RE = re.compile(r"class\s*=\s*[\"']([^\"']+)[\"']", re.I)
_ID_RE = re.compile(r"\bid\s*=\s*[\"']([^\"']+)[\"']", re.I)

_HANDLE_TOKENS = (
    "drag-icon", "drag_icon", "dragger", "drag-handle", "slider_button",
    "slider-button", "slider-btn", "slide-button", "gt_slider_knob",
    "geetest_slider_button", "captcha-drag", "drag_slider",
)
_PIECE_TOKENS = ("img_slide", "img-slide", "slice", "piece", "puzzle-img",
                 "captcha_slide_img", "slide_img", "slide-img")
_PUZZLE_TEXT_RE = re.compile(
    r"drag the slider|slide to fit|drag.*fit the puzzle|fit the puzzle",
    re.I)


@dataclass
class _El:
    tag: str
    classes: str
    el_id: str
    pos: int

    def selector(self, prefer: tuple[str, ...] = ()) -> str:
        if self.el_id:
            return f"#{self.el_id}"
        tokens = self.classes.split()
        for tok in tokens:
            if any(p in tok for p in prefer):
                return f".{tok}"
        return f".{tokens[0]}" if tokens else self.tag


def _elements(html: str) -> list[_El]:
    out: list[_El] = []
    for m in _TAG_RE.finditer(html or ""):
        attrs = m.group(2) or ""
        cm = _CLASS_RE.search(attrs)
        im = _ID_RE.search(attrs)
        out.append(_El(tag=m.group(1).lower(),
                       classes=(cm.group(1) if cm else ""),
                       el_id=(im.group(1) if im else ""),
                       pos=m.start()))
    return out


def _has_token(el: _El, tokens: tuple[str, ...]) -> bool:
    cls = el.classes.lower()
    return any(t in cls for t in tokens)


def detect_slider_challenge(html: str, url: str = ""
                            ) -> SliderChallengeInfo | None:
    """Recognise a slider-puzzle wall in page HTML. None = not one.

    Families, in order: ByteDance/TikTok verify (in-house markup —
    the stock captcha detector cannot see it: no sitekey, no widget
    iframe), GeeTest slider markup, then a generic structural shape
    (verify/captcha container + drag handle + puzzle image + the
    "drag the slider" instruction family). Selectors are built from
    the class/id tokens actually present, never assumed.
    """
    if not html:
        return None
    low = html.lower()
    els = _elements(html)
    puzzle_text = bool(_PUZZLE_TEXT_RE.search(html))

    def find_handle(scope: list[_El]) -> _El | None:
        for el in scope:
            if _has_token(el, _HANDLE_TOKENS):
                return el
        return None

    def classify_images(scope: list[_El]) -> tuple[_El | None, _El | None]:
        imgs = [e for e in scope if e.tag in ("img", "canvas")]
        if not imgs:
            return None, None
        piece = next((e for e in imgs if _has_token(e, _PIECE_TOKENS)),
                     None)
        bg = next((e for e in imgs if e is not piece), None)
        if bg is None and piece is not None:
            bg, piece = piece, None  # a lone "piece" img is the puzzle
        if piece is None and len(imgs) >= 2:
            piece = imgs[1] if imgs[1] is not bg else None
        return bg, piece

    # — ByteDance / TikTok verify —
    if "captcha_verify_container" in low or "secsdk-captcha" in low:
        container = next(
            (e for e in els if "captcha_verify_container" in
             e.classes.lower()), None)
        handle = find_handle(els)
        bg, piece = classify_images(els)
        if handle is not None and bg is not None:
            return SliderChallengeInfo(
                vendor="tiktok",
                container_selector=(container.selector()
                                    if container else ""),
                background_selector=bg.selector(),
                piece_selector=piece.selector() if piece else "",
                handle_selector=handle.selector(),
                shape="two_image" if piece else "single_image",
                evidence="bytedance captcha_verify markup")
        if puzzle_text and handle is not None:
            return SliderChallengeInfo(
                vendor="tiktok",
                container_selector=(container.selector()
                                    if container else ""),
                background_selector=bg.selector() if bg else "",
                piece_selector=piece.selector() if piece else "",
                handle_selector=handle.selector(),
                shape="two_image" if piece else "single_image",
                evidence="bytedance markup + drag-the-slider text")

    # — GeeTest slider (v3 gt_* / v4 geetest_*) —
    if "geetest" in low or "gt_slider" in low or "gt_box" in low:
        handle = find_handle(els) or next(
            (e for e in els if "slider" in e.classes.lower()
             and e.tag == "div"), None)
        bg, piece = classify_images(els)
        if handle is not None and bg is not None:
            return SliderChallengeInfo(
                vendor="geetest",
                container_selector="",
                background_selector=bg.selector(),
                piece_selector=piece.selector() if piece else "",
                handle_selector=handle.selector(),
                shape="two_image" if piece else "single_image",
                evidence="geetest slider markup")

    # — generic drag-puzzle structure —
    if puzzle_text:
        container = next(
            (e for e in els
             if any(k in e.classes.lower()
                    for k in ("captcha", "verify", "puzzle"))), None)
        handle = find_handle(els)
        bg, piece = classify_images(els)
        if handle is not None and bg is not None:
            return SliderChallengeInfo(
                vendor="generic",
                container_selector=(container.selector()
                                    if container else ""),
                background_selector=bg.selector(),
                piece_selector=piece.selector() if piece else "",
                handle_selector=handle.selector(),
                shape="two_image" if piece else "single_image",
                evidence="generic drag-puzzle structure + text")
    return None


# ── trajectory synthesis ─────────────────────────────────────────────────

@dataclass
class DragPoint:
    x: float
    y: float
    delay_ms: int = 0     # wait BEFORE moving to this point

    def as_tuple(self) -> tuple[float, float, int]:
        return (self.x, self.y, self.delay_ms)


def _ease_in_out(t: float) -> float:
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def synthesize_trajectory(start_x: float, start_y: float,
                          distance: float, *, seed: Any = None,
                          duration_ms: int | None = None
                          ) -> list[DragPoint]:
    """A human-shaped drag path from (start_x, start_y) by ``distance``.

    Ease-in acceleration, mid-drag speed variance (a smooth time
    warp), an overshoot past the target with a settle-back correction,
    and small vertical jitter that returns home. Deterministic under
    an injected ``seed``. The final point is exactly
    ``start_x + distance``.
    """
    rng = random.Random(seed)
    dist = float(distance)
    if duration_ms is None:
        duration_ms = int(rng.uniform(620, 1250)
                          + min(350.0, abs(dist) * 0.55))
    duration_ms = max(280, int(duration_ms))
    n = max(24, min(150, duration_ms // 14))
    overshoot = math.copysign(
        max(3.0, min(14.0, abs(dist) * rng.uniform(0.05, 0.09))), dist)
    warp_phase = rng.uniform(0.0, math.tau)
    warp_freq = rng.uniform(7.0, 11.0)
    # vertical jitter: smoothed random walk, pinned to 0 at both ends
    walk = [rng.uniform(-1.0, 1.0) for _ in range(n + 1)]
    for _ in range(2):
        walk = [walk[0]] + [
            (walk[i - 1] + 2 * walk[i] + walk[i + 1]) / 4.0
            for i in range(1, n)] + [walk[n]]
    amp = rng.uniform(1.6, 3.4)
    peak = max(1e-6, max(abs(v) for v in walk))
    walk = [v / peak * amp for v in walk]
    walk = [v - walk[n] * (i / n) for i, v in enumerate(walk)]
    if max(walk) - min(walk) < 1.0:
        # a freak flat walk (near-perfect ramp) still needs visible
        # jitter — add a gentle arc, zero at both ends like the walk
        walk = [v + 0.9 * math.sin(math.pi * i / n)
                for i, v in enumerate(walk)]

    # per-step delays: noisy, normalised to the total duration
    raw = [rng.uniform(0.7, 1.3) for _ in range(n)]
    scale = duration_ms / max(1e-9, sum(raw))
    delays = [max(3, int(round(r * scale))) for r in raw]
    delays[-1] += duration_ms - sum(delays)

    points: list[DragPoint] = [DragPoint(float(start_x), float(start_y), 0)]
    for i in range(1, n + 1):
        t = i / n
        warp = 0.03 * math.sin(t * warp_freq + warp_phase) * t * (1 - t)
        te = min(1.0, max(0.0, t + warp))
        x = start_x + dist * _ease_in_out(te)
        if t > 0.60:
            u = (t - 0.60) / 0.40
            x += overshoot * (math.sin(math.pi * u) ** 2)
        if i == n:
            x = start_x + dist  # settle lands exactly on target
        points.append(DragPoint(round(x, 2),
                                round(start_y + walk[i], 2),
                                delays[i - 1]))
    return points


# ── in-session execution ─────────────────────────────────────────────────

@dataclass
class SliderOutcome:
    ok: bool
    detail: str = ""
    attempts: int = 0            # drags actually performed
    gap_x: float | None = None
    confidence: float = 0.0
    distance_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "detail": self.detail,
                "attempts": self.attempts, "gap_x": self.gap_x,
                "confidence": round(self.confidence, 4),
                "distance_px": round(self.distance_px, 2)}


def _tab_url(tab: Any) -> str:
    return str(getattr(tab, "url", "") or "")


class LocalSliderSolver:
    """Solve slider challenges on a live tab, in its own session."""

    def __init__(self, *, limiter: Any = None, settings: Any = None,
                 max_drags: int = MAX_DRAGS_PER_CHALLENGE,
                 seed: Any = None,
                 sleeper: Callable[[float], None] | None = None,
                 vision_check: Callable[[], tuple[bool, str]] | None = None,
                 settle_s: float = 0.35) -> None:
        self._limiter = limiter
        self._settings = settings
        self._max_drags = max(1, int(max_drags))
        self._seed = seed
        self._sleep = sleeper or time.sleep
        self._vision_check = vision_check
        self._settle_s = max(0.0, float(settle_s))

    # -- availability ------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if self._vision_check is not None:
            return self._vision_check()
        st = vision_status()
        return bool(st["available"]), str(st["detail"])

    def _limiter_or_default(self) -> Any:
        if self._limiter is not None:
            return self._limiter
        from ..tools.captcha import rate_limiter
        return rate_limiter(self._settings)

    # -- main entry ----------------------------------------------------------
    def solve_tab(self, tab: Any, *,
                  info: SliderChallengeInfo | None = None
                  ) -> SliderOutcome:
        ok_avail, reason = self.available()
        if not ok_avail:
            return SliderOutcome(False, detail=reason)
        if info is None:
            info = detect_slider_challenge(tab.html(), _tab_url(tab))
        if info is None:
            return SliderOutcome(
                False, detail="no slider challenge detected on this page")
        if not info.handle_selector or not info.background_selector:
            return SliderOutcome(
                False, detail="slider detected but its handle/background "
                              "elements could not be located")
        limiter = self._limiter_or_default()
        try:
            limiter.check()
        except Exception as exc:  # noqa: BLE001 - limiter refusal is data
            return SliderOutcome(False, detail=f"rate limiter: {exc}")
        try:
            limiter.record_submit()
        except Exception:  # noqa: BLE001 - pacing never sinks a solve
            pass

        drags = 0
        last: SliderOutcome | None = None
        while True:
            outcome = self._attempt_once(tab, info, drags)
            if outcome.ok:
                self._record(limiter, success=True)
                return outcome
            if outcome.attempts <= drags:
                # No drag was performed (capture / gap-confidence /
                # distance sanity failed). Retrying would re-read the
                # same images and fail the same way — stop honestly.
                # Only a performed drag earns the one retry.
                self._record(limiter, success=False)
                return outcome
            drags = outcome.attempts
            last = outcome
            if drags >= self._max_drags:
                break
            # the puzzle may have reset — re-detect before the retry
            fresh = detect_slider_challenge(tab.html(), _tab_url(tab))
            if fresh is None:
                self._record(limiter, success=True)
                return SliderOutcome(
                    True, attempts=drags,
                    detail="slider gone after re-check — cleared")
            info = fresh
        self._record(limiter, success=False)
        return SliderOutcome(
            False, attempts=drags,
            gap_x=last.gap_x if last else None,
            confidence=last.confidence if last else 0.0,
            distance_px=last.distance_px if last else 0.0,
            detail=f"slider still present after {drags} drag(s) — "
                   f"challenge not cleared. "
                   f"{last.detail if last else ''}".strip())

    def _record(self, limiter: Any, *, success: bool) -> None:
        try:
            if success:
                limiter.record_success()
            else:
                limiter.record_failure()
        except Exception:  # noqa: BLE001 - pacing never sinks a solve
            pass

    # -- one drag ------------------------------------------------------------
    def _attempt_once(self, tab: Any, info: SliderChallengeInfo,
                      prior_attempts: int) -> SliderOutcome:
        attempts = prior_attempts + 1
        try:
            bg_bytes = tab.element_image_bytes(info.background_selector)
            piece_bytes = (tab.element_image_bytes(info.piece_selector)
                           if info.piece_selector else b"")
            bg_box = tab.element_box(info.background_selector)
            handle_box = tab.element_box(info.handle_selector)
            piece_box = (tab.element_box(info.piece_selector)
                         if info.piece_selector else None)
        except Exception as exc:  # noqa: BLE001 - capture failure is data
            return SliderOutcome(
                False, attempts=prior_attempts,
                detail=f"could not capture the puzzle elements: {exc}")
        gap = detect_gap(bg_bytes, piece_bytes)
        if not gap.found or gap.confidence < MIN_GAP_CONFIDENCE:
            return SliderOutcome(
                False, attempts=prior_attempts, confidence=gap.confidence,
                detail=f"gap detection below confidence "
                       f"({gap.confidence:.2f} < {MIN_GAP_CONFIDENCE}) — "
                       f"not dragging. {gap.detail}")
        decoded_bg = _decode_image(bg_bytes)
        if decoded_bg is None:
            return SliderOutcome(False, attempts=prior_attempts,
                                 detail="background image undecodable")
        bg_img_w = float(decoded_bg[0].shape[1])
        scale = float(bg_box["width"]) / bg_img_w if bg_img_w else 1.0
        target_left = float(bg_box["x"]) + float(gap.gap_x or 0.0) * scale
        if piece_box is not None:
            piece_img_w = bg_img_w
            if piece_bytes:
                decoded_piece = _decode_image(piece_bytes)
                if decoded_piece is not None:
                    piece_img_w = float(decoded_piece[0].shape[1])
            piece_scale = (float(piece_box["width"]) / piece_img_w
                           if piece_img_w else scale)
            current_left = (float(piece_box["x"])
                            + float(gap.piece_x or 0.0) * piece_scale)
        elif gap.piece_x is not None:
            current_left = (float(bg_box["x"])
                            + float(gap.piece_x) * scale)
        else:
            # single-image widgets start the piece at the puzzle's
            # left edge — the standard layout for this shape.
            current_left = float(bg_box["x"])
        distance = target_left - current_left
        if abs(distance) < 4.0:
            return SliderOutcome(
                False, attempts=prior_attempts, gap_x=gap.gap_x,
                confidence=gap.confidence,
                detail=f"computed drag distance {distance:.1f}px is "
                       "implausibly small — detection and page disagree, "
                       "not dragging")
        start_x = float(handle_box["x"]) + float(handle_box["width"]) / 2.0
        start_y = float(handle_box["y"]) + float(handle_box["height"]) / 2.0
        seed = (None if self._seed is None
                else f"{self._seed}:{prior_attempts}")
        points = synthesize_trajectory(start_x, start_y, distance,
                                       seed=seed)
        try:
            tab.mouse_drag_path([p.as_tuple() for p in points])
        except Exception as exc:  # noqa: BLE001 - input failure is data
            return SliderOutcome(
                False, attempts=attempts, gap_x=gap.gap_x,
                confidence=gap.confidence, distance_px=distance,
                detail=f"drag input failed: {exc}")
        self._sleep(self._settle_s)
        try:
            after = tab.html()
        except Exception as exc:  # noqa: BLE001
            return SliderOutcome(
                False, attempts=attempts, gap_x=gap.gap_x,
                confidence=gap.confidence, distance_px=distance,
                detail=f"could not re-read the page after the drag: {exc}")
        if detect_slider_challenge(after, _tab_url(tab)) is None:
            return SliderOutcome(
                True, attempts=attempts, gap_x=gap.gap_x,
                confidence=gap.confidence, distance_px=distance,
                detail=f"slider cleared in-session: dragged "
                       f"{distance:.0f}px to gap at {gap.gap_x:.0f} "
                       f"(confidence {gap.confidence:.2f})")
        return SliderOutcome(
            False, attempts=attempts, gap_x=gap.gap_x,
            confidence=gap.confidence, distance_px=distance,
            detail="slider still present after the drag")


def solve_slider_on_tab(tab: Any, **kwargs: Any) -> SliderOutcome:
    """One-call convenience: build a solver and run it on ``tab``."""
    return LocalSliderSolver(**kwargs).solve_tab(tab)


def try_local_slider_solve(tab: Any, page_url: str = "") -> dict[str, Any]:
    """Injection-shaped local attempt (signup driver + other flows).

    Returns ``{"attempted", "ok", "detail"}``. ``attempted`` is False
    — honestly — when the page holds no slider or the tab cannot
    perform the slider protocol; the caller then continues its normal
    path (token solver → checkpoint) untouched.
    """
    try:
        html = tab.html()
    except Exception as exc:  # noqa: BLE001
        return {"attempted": False, "ok": False,
                "detail": f"could not read the page: {exc}"}
    info = detect_slider_challenge(html, page_url or _tab_url(tab))
    if info is None:
        return {"attempted": False, "ok": False,
                "detail": "no slider challenge on this page"}
    for method in ("element_image_bytes", "element_box",
                   "mouse_drag_path"):
        if not callable(getattr(tab, method, None)):
            return {"attempted": False, "ok": False,
                    "detail": f"tab cannot drive a slider "
                              f"(missing {method})"}
    solver = LocalSliderSolver()
    out = solver.solve_tab(tab, info=info)
    return {"attempted": True, "ok": out.ok, "detail": out.detail,
            "attempts": out.attempts, "confidence": out.confidence}


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalSliderBackend(CaptchaBackend):
    """In-session slider solving as a captcha backend.

    Registered under ``local_slider`` via captcha.py's public
    ``register_backend`` seam. The challenge must carry the live tab
    in ``metadata["session"]`` (or ``metadata["tab"]``) — a slider
    cannot be solved from a sitekey. Without one the backend says so,
    honestly, instead of pretending.
    """

    name = "local_slider"

    def __init__(self, settings: Any = None, *,
                 solver: LocalSliderSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        return bool(vision_status()["available"])

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        if kind not in (SLIDER_KIND, CaptchaKind.GEETEST):
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local slider backend solves slider "
                       f"challenges, not {kind!r}")
        metadata = getattr(challenge, "metadata", None) or {}
        session = metadata.get("session") or metadata.get("tab")
        if session is None:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail="local slider solving needs the live browser "
                       "session in challenge.metadata['session'] — "
                       "a slider has no token to fetch")
        solver = self._solver or LocalSliderSolver(
            settings=self._settings)
        try:
            out = solver.solve_tab(session)
        except Exception as exc:  # noqa: BLE001 - failure is data
            out = SliderOutcome(False, detail=f"{type(exc).__name__}"
                                              f": {exc}")
        return SolveResult(
            ok=out.ok, kind=kind, backend=self.name,
            elapsed_ms=int((time.time() - started) * 1000),
            detail=out.detail)


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_slider`` in the captcha backend registry.

    Uses captcha.py's public ``register_backend`` seam — the frozen
    module is never edited. Idempotent; False (with a log line) when
    registration is impossible.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_slider", LocalSliderBackend)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local slider backend registration failed: %s", exc)
        return False


def local_slider_status(settings: Any = None) -> dict[str, Any]:
    """Honest status for the local backend, for status surfaces."""
    vision = vision_status()
    from .kind_solvers import registered_kinds
    out: dict[str, Any] = {
        "backend": "local_slider",
        "registered": register_local_backend(),
        "kinds_registered": registered_kinds(),
        "available": bool(vision["available"]),
        "vision": vision,
        "browser": "per-session — a live tab is required at solve time",
        "service_key_configured": bool(os.environ.get("CAPTCHA_API_KEY")),
        "can_solve_slider": bool(vision["available"]),
        "max_drags_per_challenge": MAX_DRAGS_PER_CHALLENGE,
        "min_gap_confidence": MIN_GAP_CONFIDENCE,
    }
    try:
        out["rate_limit"] = rate_limiter(settings).status()
    except Exception:  # noqa: BLE001 - status never breaks on pacing
        out["rate_limit"] = {}
    if not vision["available"]:
        out["detail"] = vision["detail"]
    return out


def solve_slider_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point (see :mod:`nomorals.browser.kind_solvers`).

    The spine calls this for kinds ``slider`` and ``geetest``; the
    solver re-detects the puzzle from the live tab itself, so the
    challenge dict is advisory (kind/page_url) only.
    """
    if tab is None:
        return {"ok": False, "backend": "local_slider",
                "detail": "slider solving needs a live tab"}
    out = LocalSliderSolver().solve_tab(tab)
    return {"ok": out.ok, "backend": "local_slider",
            "detail": out.detail, "attempts": out.attempts,
            "confidence": out.confidence}


# Registration is a side effect of importing this module: any browser
# or flow code that pulls the solver in also makes the backend
# addressable (and visible in captcha status) in its process, and
# registers the slider/geetest kinds with the routing spine.
register_local_backend()

from .kind_solvers import register_kind_solver  # noqa: E402

register_kind_solver(SLIDER_KIND, solve_slider_kind)
register_kind_solver(CaptchaKind.GEETEST, solve_slider_kind)
