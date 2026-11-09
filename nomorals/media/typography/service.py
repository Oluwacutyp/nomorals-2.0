"""The typography service facade — one door for rendered text.

Consumers (cards, posters, deeds, captions, covers, sticker
lettering, video titles) ask here instead of running their own
font ladders:

* :meth:`TypographyService.select` — semantic selection over the
  catalogue (see :mod:`selection`).
* :meth:`TypographyService.pil_font` — the selected face as a
  Pillow font, loaded with the shaping engine; ``None`` when the
  host has no usable face, so callers keep their last-resort
  default exactly as before.
* :meth:`TypographyService.render_text` / ``preview_text`` — a
  rendered PNG asset through the bounded, journaled cache;
  previews are the same pipeline at a capped size, cheap enough
  to generate before an expensive final.

Scope law: this service serves RENDERED ASSETS only. Native chat
text stays native — nothing here touches message text. Colours
and layout themes stay with the callers and the theme system;
this organ owns faces, shaping, measurement and the asset cache.

The process-wide instance (``get_service``) reads its roots from
the environment at first build; ``reset_service`` drops it (the
test suite and font-dir changes use it).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from . import shaping
from .cache import DEFAULT_CAP_BYTES, RENDERER_VERSION, RenderCache
from .catalogue import FontCatalogue, build_catalogue, default_roots
from .selection import Selection
from .selection import select as _select

_log = get_logger(__name__)

__all__ = ["TypographyService", "get_service", "reset_service"]

_PREVIEW_MAX_SIZE = 28


def _default_store() -> Path:
    home = os.environ.get("NM_HOME", "") or str(Path.home() / ".nomorals")
    return Path(home).expanduser() / "typography"


def _default_cap() -> int:
    raw = os.environ.get("NM_TYPOGRAPHY_CACHE_MB", "")
    try:
        return int(float(raw) * 1024 * 1024) if raw else DEFAULT_CAP_BYTES
    except ValueError:
        return DEFAULT_CAP_BYTES


class TypographyService:
    def __init__(self, roots=None, store_dir=None, *,
                 cache_cap_bytes: int | None = None):
        self._roots = [Path(r) for r in roots] if roots is not None \
            else default_roots()
        self.store_dir = Path(store_dir) if store_dir else _default_store()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._catalogue: FontCatalogue | None = None
        self.cache = RenderCache(
            self.store_dir,
            cap_bytes=cache_cap_bytes if cache_cap_bytes is not None
            else _default_cap())

    # catalogue + selection ───────────────────────────────────────
    def catalogue(self) -> FontCatalogue:
        if self._catalogue is None:
            self._catalogue = build_catalogue(
                self._roots, index_path=self.store_dir / "font_index.json")
        return self._catalogue

    def refresh(self) -> FontCatalogue:
        self._catalogue = None
        return self.catalogue()

    def select(self, intent: str = "text", *, script: str | None = None,
               text: str | None = None, weight: int | None = None,
               family: str | None = None,
               surface_family: str | None = None) -> Selection:
        """Semantic selection — the ONE seam owner style pins
        are consulted at (Living Chat P3). A pin in force for
        ``surface_family`` (or the global pin) outranks the
        semantic defaults: its font family fills an unpinned
        ``family`` argument and its intent replaces the
        requested intent. The returned Selection NAMES the pin
        (``Selection.pin`` + a note) — a pinned choice is never
        silent. Coverage gates (script/text) still apply inside
        the selection, so a pin can steer taste, never fabricate
        coverage."""
        if text and not script:
            script = _script_for_text(self.catalogue(), text)
        pin = self._pin_for(surface_family)
        if pin is not None:
            if pin.font_family and not family:
                family = pin.font_family
            if pin.intent:
                intent = pin.intent
        selection = _select(self.catalogue(), intent, script=script,
                            text=text, weight=weight, family=family)
        if pin is not None:
            selection.pin = pin.label
            selection.notes = selection.notes + (
                f"owner style pin applied ({pin.label}): "
                f"{pin.describe()}",)
        return selection

    @staticmethod
    def _pin_for(surface_family: str | None):
        """The effective owner pin for one surface family, or
        None. Pins never break selection: any store failure
        reads as no pin."""
        try:
            from ...core.style_pins import effective_pin, get_pin_store
            return effective_pin(get_pin_store(),
                                 str(surface_family or ""))
        except Exception as exc:  # noqa: BLE001 - pins are garnish
            _log.debug("style pin lookup failed: %s", exc)
            return None

    def pil_font(self, intent: str = "text", size: int = 48, *,
                 script: str | None = None, text: str | None = None,
                 weight: int | None = None,
                 family: str | None = None,
                 surface_family: str | None = None):
        """The selected face as a Pillow font, or None.

        Tries the selection's whole fallback chain before giving
        up; never raises — a consumer's own last-resort default
        stays reachable.
        """
        try:
            selection = self.select(intent, script=script, text=text,
                                    weight=weight, family=family,
                                    surface_family=surface_family)
        except Exception as exc:  # noqa: BLE001 - None is the contract
            _log.debug("typography select failed: %s", exc)
            return None
        for face in selection.chain:
            try:
                return shaping.load_font(face.path, size, face.face_index)
            except Exception as exc:  # noqa: BLE001 - next in chain
                _log.debug("typography: face %s failed to load: %s",
                           face.path, exc)
                continue
        return None

    # rendering ───────────────────────────────────────────────────
    def render_text(self, text: str, *, intent: str = "text",
                    size: int = 48, fill=(255, 255, 255, 255),
                    stroke_width: int = 0,
                    stroke_fill=(0, 0, 0, 255), background=None,
                    max_width: int | None = None,
                    script: str | None = None, weight: int | None = None,
                    family: str | None = None, padding: int = 12,
                    surface_family: str | None = None) -> Path:
        return self._render(text, intent=intent, size=size, fill=fill,
                            stroke_width=stroke_width,
                            stroke_fill=stroke_fill, background=background,
                            max_width=max_width, script=script,
                            weight=weight, family=family,
                            padding=padding, preview=False,
                            surface_family=surface_family)

    def preview_text(self, text: str, **kwargs: Any) -> Path:
        """A lightweight preview of the same render (capped font
        size, its own cache identity)."""
        return self._render(text, preview=True, **kwargs)

    def _render(self, text: str, *, intent: str = "text", size: int = 48,
                fill=(255, 255, 255, 255), stroke_width: int = 0,
                stroke_fill=(0, 0, 0, 255), background=None,
                max_width: int | None = None, script: str | None = None,
                weight: int | None = None, family: str | None = None,
                padding: int = 12, preview: bool = False,
                surface_family: str | None = None) -> Path:
        from PIL import Image, ImageDraw

        text = str(text)
        eff_size = min(size, _PREVIEW_MAX_SIZE) if preview else size
        selection = self.select(intent, script=script, text=text,
                                weight=weight, family=family,
                                surface_family=surface_family)
        face = selection.face
        if face is None:
            raise RuntimeError(
                "typography: no font face available on this host")
        material = {
            "renderer": RENDERER_VERSION, "engine": shaping.shaping_engine(),
            "text": text, "font_file": face.file_id,
            "face_index": face.face_index, "size": eff_size,
            "intent": selection.intent, "pin": selection.pin,
            "fill": list(fill),
            "stroke_width": stroke_width,
            "stroke_fill": list(stroke_fill),
            "background": list(background) if background else None,
            "max_width": max_width, "padding": padding,
            "direction": shaping.dominant_direction(text),
            "preview": preview,
        }
        key = RenderCache.key(material)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        font = shaping.load_font(face.path, eff_size, face.face_index)
        lines = shaping.wrap_text(text, font, max_width) \
            if max_width else text.split("\n")
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        line_h = max(eff_size, int(eff_size * 1.25))
        widths = [shaping.shaped_width(line, font) for line in lines]
        text_w = int(max(widths, default=0)) + 2 * stroke_width
        text_h = line_h * len(lines)
        img = Image.new("RGBA",
                        (max(1, text_w + padding * 2),
                         max(1, text_h + padding * 2)),
                        background if background else (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        direction = shaping.dominant_direction(text)
        y = padding
        for line in lines:
            bbox = probe.textbbox((0, 0), line, font=font,
                                  stroke_width=stroke_width)
            x = padding - bbox[0]
            if direction == "rtl":
                x = img.size[0] - padding - (bbox[2] - bbox[0]) - bbox[0]
            draw.text((x, y - bbox[1] + (line_h - (bbox[3] - bbox[1])) // 2),
                      line, font=font, fill=fill,
                      stroke_width=stroke_width, stroke_fill=stroke_fill)
            y += line_h
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return self.cache.put(key, buf.getvalue(), {
            "family": face.family, "intent": intent,
            "size": eff_size, "preview": preview})

    def cache_stats(self) -> dict[str, Any]:
        return self.cache.stats()


def _script_for_text(catalogue: FontCatalogue, text: str) -> str | None:
    """The catalogue script the text's first strong character
    belongs to (used to route selection when only text is given)."""
    from .fontparse import _SAMPLE_SCRIPTS, SCRIPTS
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for script, ranges in SCRIPTS.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                return script
        for script, sample in _SAMPLE_SCRIPTS.items():
            if ch in sample:
                return script
        return None
    return None


_SERVICE: TypographyService | None = None


def get_service() -> TypographyService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = TypographyService()
    return _SERVICE


def reset_service() -> None:
    global _SERVICE
    _SERVICE = None
