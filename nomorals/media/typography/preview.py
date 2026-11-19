"""Font preview service — the font-preview-service shape, on the
typography organ.

Upstream (storytold/font-preview-service, no licence — PATTERN
ONLY): a small Rust web service that renders font previews. It is
an early scaffold — the only endpoint is ``GET /healthz`` — but
its shape is fully declared: configuration through environment
variables (``PORT``, ``FONTS_DIR``, ``FILES_CONCURRENCY``,
``FILES_QUEUE`` with a 2 s wait then 503 + ``Retry-After``), font
sets on disk with per-family licence files kept alongside, and
the not-yet-built work named in its README: reading the fonts,
rendering preview images, and subsetting — render only the
glyphs each preview string needs.

Devon's port is that shape, written fresh against the organ Devon
already has (no upstream code or text vendored):

* request/response: :class:`PreviewRequest` (font + text + size /
  style) in, :class:`PreviewResult` (PNG + dimensions + sha256 +
  cache-hit + glyph coverage) out — the preview endpoint the
  upstream never built, on the service that already renders.
* health: :meth:`FontPreviewService.health` — the ``/healthz``
  shape (status, fonts indexed, cache stats, shaping engine).
* cache: the organ's bounded, journaled RenderCache, keyed by
  font file + text + size + style — the upstream's LRU, on disk.
* refusal: a missing family is named (:class:`FontNotStocked`,
  family + tiers searched); an unsupported format is named; an
  oversized render is named with the cap; a saturated render
  queue answers :class:`PreviewBusy` with ``retry_after``
  seconds — the upstream's 503 + ``Retry-After``, in-process.
* subsetting, Devon's honest form: the result names exactly
  which glyphs the preview string needs that the chosen face
  does not have (``missing_glyphs``), instead of silently
  boxing them.

The surface is in-process: Devon serves previews through the
craftfonts brain tool, not a separate HTTP listener, so there is
no PORT to bind — the upstream's ``PORT``/``FONTS_DIR`` knobs
map to the organ's own roots (the stocked library + catalogue
roots) and the concurrency/queue knobs below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from . import shaping
from .fontparse import FontFace
from .service import TypographyService, get_service

_log = get_logger(__name__)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_DIMENSION",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_QUEUE_TIMEOUT_S",
    "FontPreviewService",
    "PreviewBusy",
    "PreviewError",
    "PreviewRequest",
    "PreviewResult",
    "PreviewTooLarge",
    "PREVIEW_RENDERER_VERSION",
    "UnsupportedPreviewFormat",
    "get_preview_service",
    "reset_preview_service",
]

#: The one image format previews render in. Anything else is a
#: named refusal, not a silent conversion.
SUPPORTED_FORMATS = ("png",)

#: The preview service's own renderer version (the cache
#: material underneath carries the organ renderer's version).
PREVIEW_RENDERER_VERSION = "font-preview-1"

DEFAULT_MAX_SIZE = 512
DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_DIMENSION = 4096
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_QUEUE_TIMEOUT_S = 2.0

#: style name -> (weight, italic). ``weight`` wins over the
#: style's weight when the request carries one explicitly.
_STYLES: dict[str, tuple[int | None, bool]] = {
    "regular": (None, False),
    "bold": (700, False),
    "italic": (None, True),
    "bold-italic": (700, True),
}


class PreviewError(ValueError):
    """A preview request Devon refuses, naming the reason."""


class UnsupportedPreviewFormat(PreviewError):
    """The requested image format is not one previews render in."""


class PreviewTooLarge(PreviewError):
    """The request exceeds a preview cap (size, text, dimension)."""


class PreviewBusy(RuntimeError):
    """The render queue is saturated — retry after ``retry_after``.

    The upstream answers this shape as HTTP 503 with a
    ``Retry-After`` header; in-process it is this error carrying
    the same number of seconds.
    """

    def __init__(self, retry_after: float):
        self.retry_after = float(retry_after)
        super().__init__(
            "font preview service is busy — retry after "
            f"{self.retry_after:g}s")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


@dataclass
class PreviewRequest:
    """One preview: a text sample in a named font at a size/style.

    ``family`` empty means "any face the organ selects".
    ``fill``/``background`` are RGBA tuples; the default is black
    text on a white card — a preview is meant to be looked at.
    """

    family: str = ""
    text: str = ""
    size: int = 48
    weight: int | None = None
    style: str = "regular"
    format: str = "png"
    fill: tuple[int, int, int, int] = (0, 0, 0, 255)
    background: tuple[int, int, int, int] | None = (255, 255, 255, 255)
    max_width: int | None = None
    padding: int = 12

    def style_weight_italic(self) -> tuple[int | None, bool]:
        """(weight, italic) the style asks for, or a named refusal."""
        wanted = _STYLES.get(str(self.style).strip().lower())
        if wanted is None:
            raise PreviewError(
                "font preview: unknown style "
                f"{self.style!r} (supported: "
                f"{', '.join(sorted(_STYLES))})")
        return wanted


@dataclass
class PreviewResult:
    """A rendered preview: the PNG on disk plus its facts."""

    path: Path
    width: int
    height: int
    png_bytes: int
    sha256: str
    cache_hit: bool
    family: str
    subfamily: str
    size: int
    weight: int
    italic: bool
    format: str
    missing_glyphs: tuple[str, ...] = ()
    glyphs_needed: int = 0
    direction: str | None = None
    engine: str = "basic"
    renderer: str = PREVIEW_RENDERER_VERSION
    style_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "width": self.width,
            "height": self.height, "png_bytes": self.png_bytes,
            "png_sha256": self.sha256, "cache_hit": self.cache_hit,
            "family": self.family, "subfamily": self.subfamily,
            "size": self.size, "weight": self.weight,
            "italic": self.italic, "format": self.format,
            "missing_glyphs": list(self.missing_glyphs),
            "glyphs_needed": self.glyphs_needed,
            "direction": self.direction, "engine": self.engine,
            "renderer": self.renderer, "style_note": self.style_note,
        }


def _is_significant(ch: str) -> bool:
    """A character that needs a glyph of its own (the same
    neutrality rule :meth:`FontFace.covers_text` uses: whitespace,
    format controls and ASCII punctuation survive via fallback)."""
    if ch.isspace() or unicodedata.category(ch) == "Cf":
        return False
    return not (ord(ch) < 0x80 and not ch.isalnum())


def _missing_glyphs(face: FontFace, text: str) -> tuple[str, ...]:
    """The distinct significant characters ``face`` has no glyph
    for, in first-seen order — the subsetting question, answered
    honestly instead of rendered as boxes in silence."""
    missing: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch in seen or not _is_significant(ch):
            continue
        seen.add(ch)
        if not face.covers_codepoint(ord(ch)):
            missing.append(ch)
    return tuple(missing)


def _estimate_dims(text: str, font: Any, size: int,
                   max_width: int | None, padding: int) -> tuple[int, int]:
    """The pixel size the render will be, measured with the real
    font and the real shaping — the same arithmetic the service's
    own render uses, so the oversize refusal fires before the
    render instead of after it."""
    lines = shaping.wrap_text(text, font, max_width) \
        if max_width else text.split("\n")
    line_h = max(size, int(size * 1.25))
    widths = [shaping.shaped_width(line, font) for line in lines]
    text_w = int(max(widths, default=0))
    text_h = line_h * len(lines)
    return max(1, text_w + padding * 2), max(1, text_h + padding * 2)


class FontPreviewService:
    """The font-preview service shape on the typography organ.

    Configuration mirrors the upstream's env knobs: max render
    size / text length / pixel dimension (the oversize refusals),
    ``max_concurrent`` renders at once and ``queue_timeout_s``
    waiting for a slot (the upstream's ``FILES_CONCURRENCY`` /
    ``FILES_QUEUE`` 2 s wait, then 503 + ``Retry-After``).
    """

    def __init__(self, service: TypographyService | None = None, *,
                 max_size: int | None = None,
                 max_chars: int | None = None,
                 max_dimension: int | None = None,
                 max_concurrent: int | None = None,
                 queue_timeout_s: float | None = None):
        self.service = service if service is not None else get_service()
        self.max_size = max_size if max_size is not None else \
            _env_int("NM_FONT_PREVIEW_MAX_SIZE", DEFAULT_MAX_SIZE)
        self.max_chars = max_chars if max_chars is not None else \
            _env_int("NM_FONT_PREVIEW_MAX_CHARS", DEFAULT_MAX_CHARS)
        self.max_dimension = \
            max_dimension if max_dimension is not None else _env_int(
                "NM_FONT_PREVIEW_MAX_DIMENSION", DEFAULT_MAX_DIMENSION)
        self.max_concurrent = \
            max_concurrent if max_concurrent is not None else _env_int(
                "NM_FONT_PREVIEW_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT)
        self.queue_timeout_s = \
            queue_timeout_s if queue_timeout_s is not None else \
            _env_float("NM_FONT_PREVIEW_QUEUE_TIMEOUT_S",
                       DEFAULT_QUEUE_TIMEOUT_S)
        if self.max_concurrent < 1:
            raise ValueError(
                "font preview: max_concurrent must be at least 1")
        self._slots = threading.BoundedSemaphore(self.max_concurrent)

    # health ──────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        """The /healthz shape: status, fonts indexed, cache
        stats, shaping engine, concurrency knobs."""
        catalogue = self.service.catalogue()
        return {
            "status": "ok",
            "fonts_indexed": len(catalogue.faces()),
            "families": len(catalogue.families()),
            "cache": self.service.cache_stats(),
            "engine": shaping.shaping_engine(),
            "renderer": PREVIEW_RENDERER_VERSION,
            "concurrency": {
                "max_concurrent": self.max_concurrent,
                "queue_timeout_s": self.queue_timeout_s,
            },
        }

    # preview ─────────────────────────────────────────────────────
    def preview(self, request: PreviewRequest) -> PreviewResult:
        """Render one preview request, or refuse it by name."""
        text = str(request.text)
        if not text.strip():
            raise PreviewError("font preview needs non-empty text")
        fmt = str(request.format).strip().lower()
        if fmt not in SUPPORTED_FORMATS:
            raise UnsupportedPreviewFormat(
                "font preview: format "
                f"{request.format!r} is not supported "
                f"(supported: {', '.join(SUPPORTED_FORMATS)})")
        if len(text) > self.max_chars:
            raise PreviewTooLarge(
                "font preview: text is "
                f"{len(text)} characters (cap {self.max_chars})")
        try:
            size = int(request.size)
        except (TypeError, ValueError):
            raise PreviewError(
                "font preview: size "
                f"{request.size!r} is not a number") from None
        if size < 1 or size > self.max_size:
            raise PreviewTooLarge(
                f"font preview: size {size} is outside "
                f"1..{self.max_size}")
        style_weight, style_italic = request.style_weight_italic()
        weight = request.weight if request.weight is not None \
            else style_weight
        if weight is not None:
            try:
                weight = int(weight)
            except (TypeError, ValueError):
                raise PreviewError(
                    "font preview: weight "
                    f"{request.weight!r} is not a number") from None
        acquired = self._slots.acquire(timeout=self.queue_timeout_s)
        if not acquired:
            raise PreviewBusy(self.queue_timeout_s)
        try:
            return self._render_locked(request, text, fmt, size, weight,
                                       style_italic)
        finally:
            self._slots.release()

    def _render_locked(self, request: PreviewRequest, text: str,
                       fmt: str, size: int, weight: int | None,
                       style_italic: bool) -> PreviewResult:
        # The family refusal lives in the organ: select() raises
        # FontNotStocked naming the family and every tier
        # searched — never a silent slide to another face.
        selection = self.service.select(
            intent="text", text=text, weight=weight,
            family=request.family or None)
        face = selection.face
        if face is None:
            raise RuntimeError(
                "typography: no font face available on this host")
        # The requested style is authoritative: the organ's
        # semantic selection prefers italic faces for the "text"
        # intent, so a regular preview must ask for the regular
        # face explicitly — and an italic request must ask for
        # the italic one. When the family holds no face in the
        # requested slant at all, the render still happens (the
        # closest face) but the result SAYS so.
        style_note = ""
        if face.italic != style_italic:
            pool = self.service.catalogue().faces(family=face.family)
            want = weight if weight is not None else 400
            same_slant = [f for f in pool if f.italic == style_italic
                          and f.covers_text(text)]
            if not same_slant:
                same_slant = [f for f in pool
                              if f.italic == style_italic]
            if same_slant:
                same_slant.sort(key=lambda f: (
                    abs(f.weight - want), f.path, f.face_index))
                face = same_slant[0]
                selection = dataclasses.replace(
                    selection, face=face,
                    chain=(face, *selection.chain))
            else:
                style_note = (
                    f"no {'italic' if style_italic else 'regular'} "
                    f"face for {face.family!r} — rendered "
                    f"{face.subfamily}")
        missing = _missing_glyphs(face, text)
        glyphs_needed = len({ch for ch in text if _is_significant(ch)})
        font = shaping.load_font(face.path, size, face.face_index)
        est_w, est_h = _estimate_dims(
            text, font, size, request.max_width, request.padding)
        if est_w > self.max_dimension or est_h > self.max_dimension:
            raise PreviewTooLarge(
                "font preview: render would be "
                f"{est_w}x{est_h}px (cap {self.max_dimension}px per "
                "side)")
        path, hit = self.service._render(
            text, selection=selection, size=size, fill=request.fill,
            background=request.background, max_width=request.max_width,
            padding=request.padding)
        data = path.read_bytes()
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
        if width > self.max_dimension or height > self.max_dimension:
            raise PreviewTooLarge(
                "font preview: rendered "
                f"{width}x{height}px (cap {self.max_dimension}px per "
                "side)")
        return PreviewResult(
            path=path, width=width, height=height,
            png_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            cache_hit=hit, family=face.family,
            subfamily=face.subfamily, size=size, weight=face.weight,
            italic=face.italic, format=fmt, missing_glyphs=missing,
            glyphs_needed=glyphs_needed,
            direction=shaping.dominant_direction(text),
            engine=shaping.shaping_engine(),
            renderer=PREVIEW_RENDERER_VERSION, style_note=style_note)


_PREVIEW_SERVICE: FontPreviewService | None = None


def get_preview_service() -> FontPreviewService:
    global _PREVIEW_SERVICE
    if _PREVIEW_SERVICE is None:
        _PREVIEW_SERVICE = FontPreviewService()
    return _PREVIEW_SERVICE


def reset_preview_service() -> None:
    global _PREVIEW_SERVICE
    _PREVIEW_SERVICE = None
