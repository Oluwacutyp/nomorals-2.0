"""Unified video generation — one front door for Devon Studio.

:func:`generate` checks real capability (CUDA VRAM, diffusers, profile)
and routes honestly:

- neural path (LTX 2B distilled / Wan 2.2 TI2V-5B) when the hardware is
  there — the clip is genuinely diffusion-generated;
- motion-studio path otherwise — Ken Burns / visualizer / typography,
  which is a real video, not a fake neural one.

The :class:`VideoResult` always says which path ran and why, so chat
and CLI can repeat it verbatim. Nothing here is silent.

:func:`request_hero_clip` is the integration point for the shorts
pipeline (``nomorals.media.contentops``): it asks for one neural hero
clip and gets back either a path or an honest decline the pipeline can
plan around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import VideogenError, neural_capability
from ..motion_studio._core import record_ledger
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["generate", "request_hero_clip", "capability_report", "VideoResult"]


@dataclass
class VideoResult:
    path: str
    engine: str            # "ltx" | "wan" | "motion"
    routed: str            # "neural" | "motion"
    note: str              # plain-language: why this path
    prompt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def message(self) -> str:
        icon = "🧠" if self.routed == "neural" else "🎞️"
        return (f"{icon} {self.path}\n"
                f"   engine: {self.engine} ({self.routed})\n"
                f"   {self.note}")


def capability_report(prefer: str = "auto") -> str:
    """Human-readable capability summary for chat/CLI."""
    cap = neural_capability(prefer=prefer)
    lines = [cap.explain()]
    if cap.available:
        lines.append(f"   backend: {cap.backend} · profile: {cap.profile}")
    else:
        lines.append("   fallback: motion studio (Ken Burns, visualizer, "
                     "typography — CPU-only, always works)")
    return "\n".join(lines)


def _neural_generate(prompt: str, backend: str, **kw) -> str:
    if backend == "wan":
        from .wan_backend import WanBackend
        gen = WanBackend()
    else:
        from .ltx_backend import LTXBackend
        gen = LTXBackend()
    try:
        return gen.generate(prompt, **kw)
    finally:
        gen.unload()


def _motion_fallback(prompt: str, **kw) -> tuple[str, str]:
    """Best-effort motion-studio stand-in for a neural clip.

    Returns (path, note). A prompt-only request becomes a quote-card
    style kinetic title — clearly labeled as motion, never as neural.
    """
    from ..motion_studio.typography import render_quote_card
    from ..motion_studio.kenburns import kenburns
    image = kw.get("image")
    duration = float(kw.get("duration_s", 5.0))
    size = kw.get("size")
    fps = kw.get("fps")
    if image and Path(image).exists():
        path = kenburns(image, duration=duration, move="auto",
                        letterbox=True, size=size, fps=fps)
        note = ("no neural backend on this machine — rendered as a "
                "cinematic still-animation instead")
    else:
        text = prompt if len(prompt) <= 90 else prompt[:87] + "…"
        path = render_quote_card(text, duration=duration,
                                 preset="bold_statement",
                                 size=size, fps=fps)
        note = ("no neural backend on this machine — rendered as kinetic "
                "typography instead of a diffusion clip")
    return path, note


def generate(prompt: str, *, backend: str = "auto",
             mode: str = "t2v",
             image: str | os.PathLike | None = None,
             duration_s: float = 5.0,
             seed: int = 0,
             out: str | os.PathLike | None = None,
             width: int = 0, height: int = 0,
             allow_fallback: bool = True,
             size: tuple[int, int] | None = None,
             fps: float | None = None,
             **kw) -> VideoResult:
    """Generate a video clip — neural when possible, motion otherwise.

    ``backend``: auto | ltx | wan | motion (force the motion studio).
    ``mode``: t2v | i2v. Raises :class:`VideogenError` only when neural
    was explicitly required (``allow_fallback=False``) and unavailable.
    """
    if not (prompt or "").strip() and not image:
        raise VideogenError("empty prompt and no image — nothing to generate")
    backend = (backend or "auto").lower()

    if backend != "motion":
        cap = neural_capability(prefer=backend)
        if cap.available:
            size_kw: dict[str, Any] = {}
            if width:
                size_kw["width"] = width
            if height:
                size_kw["height"] = height
            path = _neural_generate(
                prompt, cap.backend, mode=mode, image=image,
                duration_s=duration_s, seed=seed, out=out, **size_kw, **kw)
            result = VideoResult(
                path=path, engine=cap.backend, routed="neural",
                note=f"diffusion-generated with {cap.backend} "
                     f"({cap.reason})", prompt=prompt,
                meta={"vram_gb": cap.vram_gb, "model": cap.backend})
            record_ledger({"kind": "videogen.generate", "path": path,
                           "engine": cap.backend, "routed": "neural"})
            return result
        if not allow_fallback:
            raise VideogenError(f"neural unavailable: {cap.reason}")
        fallback_note = cap.reason
    else:
        fallback_note = "motion studio explicitly requested"

    path, motion_note = _motion_fallback(
        prompt or "untitled", image=image, duration_s=duration_s,
        size=size, fps=fps)
    result = VideoResult(
        path=path, engine="motion", routed="motion",
        note=f"{fallback_note}. {motion_note}", prompt=prompt,
        meta={"fallback_reason": fallback_note})
    record_ledger({"kind": "videogen.generate", "path": path,
                   "engine": "motion", "routed": "motion"})
    return result


def request_hero_clip(prompt: str, *, duration_s: float = 5.0,
                      seed: int = 0,
                      prefer: str = "auto") -> dict[str, Any]:
    """Integration point for the shorts pipeline: one neural hero clip.

    Returns ``{"ok": True, "path": …}`` on success or
    ``{"ok": False, "reason": …}`` when neural is unavailable — the
    caller (shorts pipeline) decides the fallback; this function never
    silently substitutes motion graphics for a neural clip.
    """
    cap = neural_capability(prefer=prefer)
    if not cap.available:
        return {"ok": False, "reason": cap.reason,
                "suggestion": "use motion_studio.kenburns over a generated "
                              "still for the hero shot instead"}
    try:
        path = _neural_generate(prompt, cap.backend, mode="t2v",
                                duration_s=duration_s, seed=seed)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{cap.backend} failed: {exc}",
                "suggestion": "retry, or fall back to motion studio"}
    return {"ok": True, "path": path, "backend": cap.backend,
            "vram_gb": cap.vram_gb}
