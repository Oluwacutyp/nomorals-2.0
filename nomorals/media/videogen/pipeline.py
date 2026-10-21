"""Unified video generation — one front door for Devon Studio.

:func:`generate` checks real capability (CUDA VRAM, diffusers, profile)
and routes honestly:

- neural path (LTX 2B distilled / Wan 2.2 TI2V-5B) when the hardware is
  there — the clip is genuinely diffusion-generated;
- motion-studio path otherwise — Ken Burns / visualizer / typography,
  which is a real video, not a fake neural one.

The :class:`VideoResult` always says which path ran and why, so chat
and CLI can repeat it verbatim. Nothing here is silent.

:func:`request_hero_clip` asks for one neural hero clip and gets back
either a path or an honest decline the caller can plan around. It is
exposed on the shorts surfaces (chat ``/shorts hero``, CLI
``nm shorts hero``). The contentops :class:`ShortPipeline` itself does
NOT call it — its scenes are imggen stills animated by the edit engine
today; this function is the clip-level API for consumers that want a
real neural hero shot, and it never substitutes motion graphics for
one silently.

Model and LoRA selection (``model=`` / ``lora=`` on :func:`generate`)
resolves through :mod:`nomorals.media.videogen.registry`: built-in
checkpoints, local diffusers dirs under ``~/.devon-models``, and
installed LoRA adapters are all selectable data, with the selected
model's own VRAM floor driving the capability gate.
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
    """Human-readable capability summary for chat/CLI.

    Includes the video registry: every selectable model and LoRA with
    its VRAM floor and loadable state, so chat/CLI users can pick a
    ``--model``/``--lora`` by name.
    """
    cap = neural_capability(prefer=prefer)
    lines = [cap.explain()]
    if cap.available:
        lines.append(f"   backend: {cap.backend} · profile: {cap.profile}")
    else:
        lines.append("   fallback: motion studio (Ken Burns, visualizer, "
                     "typography — CPU-only, always works)")
    try:
        from . import registry as _registry
        models = _registry.list_models()
        loras = _registry.list_loras()
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        lines.append(f"   video registry unavailable: {exc}")
        return "\n".join(lines)
    lines.append("   models (select with --model):")
    for m in models:
        floor = (f"≥{m['vram_gb']:g}GB VRAM"
                 if m.get("vram_gb") else "VRAM floor unknown")
        state = ("loadable" if m["loadable"]
                 else f"not loadable: {m['loadable_reason']}")
        dflt = " · default" if m.get("default") else ""
        lines.append(f"     • {m['name']} [{m['backend'] or '?'}] "
                     f"{m['source']} — {floor} · license: "
                     f"{m['license_key'] or 'unknown'} · {state}{dflt}")
    if loras:
        lines.append("   LoRAs (select with --lora, repeatable):")
        for l in loras:
            lines.append(f"     • {l['name']} [{l['backend']}] {l['path']}")
    else:
        lines.append("   LoRAs: none installed "
                     f"(drop adapters under {_registry.models_root()}"
                     "/loras/<backend>/<name>)")
    return "\n".join(lines)


def _resolve_selection(prefer_backend: str, model: Any,
                       lora: Any) -> dict[str, Any]:
    """Resolve a model/LoRA selection against the video registry.

    Returns ``{backend, source, label, floor, distilled, loras}`` —
    ``backend``/``floor`` are None when no model was selected (the
    caller keeps its own backend choice and stock floors). Raises
    :class:`VideogenError` for unknown model names, not-loadable
    registry entries, backend conflicts, raw repo ids without an
    explicit backend, and LoRA/backend mismatches — selection errors
    are caller errors, never silent substitutions.
    """
    from . import registry as _registry
    loras = _registry.normalize_loras(lora)
    sel: dict[str, Any] = {"backend": None, "source": "", "label": "",
                           "floor": None, "distilled": True,
                           "loras": loras}
    if not model:
        return sel
    entry = _registry.resolve(str(model))
    if entry is not None:
        if not entry["loadable"]:
            raise VideogenError(
                f"video model {entry['name']!r} is registered but not "
                f"loadable: {entry['loadable_reason']}")
        sel.update(backend=entry["backend"], source=entry["source"],
                   label=entry["name"], floor=entry["vram_gb"],
                   distilled=bool(entry.get("distilled")))
    elif os.path.isdir(str(model)):
        raise VideogenError(
            f"video model dir {model!r} has no model_index.json — "
            "not a diffusers model directory")
    elif "/" in str(model):
        if prefer_backend not in ("ltx", "wan"):
            raise VideogenError(
                f"unregistered model id {model!r} needs an explicit "
                "backend (backend='ltx' or 'wan') — or register it "
                "under ~/.devon-models so the registry knows its "
                "backend and VRAM floor")
        sel.update(backend=prefer_backend, source=str(model),
                   label=str(model), floor=None)
    else:
        known = ", ".join(m["name"] for m in _registry.list_models())
        raise VideogenError(
            f"unknown video model {model!r} — registry models: {known}; "
            "or pass an HF repo id with backend='ltx'|'wan', or a local "
            "diffusers model dir path")
    if prefer_backend in ("ltx", "wan") and sel["backend"] \
            and prefer_backend != sel["backend"]:
        raise VideogenError(
            f"backend={prefer_backend!r} conflicts with model "
            f"{sel['label']!r}, which is a {sel['backend']} model — "
            "pick one")
    for l in loras:
        if l["backend"] and sel["backend"] and l["backend"] != sel["backend"]:
            raise VideogenError(
                f"LoRA {l['name']!r} is a {l['backend']} adapter but "
                f"model {sel['label']!r} runs on {sel['backend']} — "
                "LoRAs only apply to their own backend")
    return sel


def _check_lora_backend(loras: list[dict], backend: str,
                        label: str) -> None:
    """LoRA/backend fit check once the serving backend is known."""
    for l in loras:
        if l["backend"] and l["backend"] != backend:
            raise VideogenError(
                f"LoRA {l['name']!r} is a {l['backend']} adapter but "
                f"this generation runs on {backend} ({label}) — "
                "LoRAs only apply to their own backend")


def _neural_generate(prompt: str, backend: str, *, model_id: str = "",
                     loras: Any = (), distilled: bool = True,
                     **kw) -> str:
    pairs = tuple((l["source"], l["weight"]) if isinstance(l, dict)
                  else (str(l[0]), float(l[1])) for l in loras)
    if backend == "wan":
        from .wan_backend import WanBackend
        gen = WanBackend(model_id=model_id, loras=pairs)
    else:
        from .ltx_backend import LTXBackend
        gen = LTXBackend(model_id=model_id, distilled=distilled,
                         loras=pairs)
        if pairs or model_id:
            kw.setdefault("distilled", distilled)
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
             model: str | None = None,
             lora: Any = None,
             **kw) -> VideoResult:
    """Generate a video clip — neural when possible, motion otherwise.

    ``backend``: auto | ltx | wan | motion (force the motion studio).
    ``mode``: t2v | i2v. Raises :class:`VideogenError` only when neural
    was explicitly required (``allow_fallback=False``) and unavailable.

    ``model``: a registry model name (``ltx-distilled``, ``ltx-base``,
    ``wan-ti2v-5b``, a discovered local model), a local diffusers
    model dir, or a raw HF repo id (then ``backend`` must name ltx or
    wan). The selected model's registry VRAM floor drives the
    capability gate. ``lora``: LoRA selection — a name/path, a
    ``"name:weight"`` string, or a list of those / ``(spec, weight)``
    pairs (see :func:`registry.normalize_loras`). Selection errors
    (unknown model/LoRA, backend conflict, LoRA/backend mismatch)
    raise — they are caller errors, never silent substitutions.
    """
    if not (prompt or "").strip() and not image:
        raise VideogenError("empty prompt and no image — nothing to generate")
    backend = (backend or "auto").lower()

    sel = _resolve_selection(backend, model, lora)
    if backend == "motion" and (model or sel["loras"]):
        raise VideogenError(
            "model/LoRA selection needs a neural backend — "
            "backend='motion' renders with the motion studio, which "
            "has no diffusion model to select")

    if backend != "motion":
        prefer = sel["backend"] or backend
        cap = neural_capability(prefer=prefer, min_vram_gb=sel["floor"])
        if cap.available:
            _check_lora_backend(sel["loras"], cap.backend,
                                sel["label"] or cap.backend)
            size_kw: dict[str, Any] = {}
            if width:
                size_kw["width"] = width
            if height:
                size_kw["height"] = height
            path = _neural_generate(
                prompt, cap.backend, model_id=sel["source"],
                loras=sel["loras"], distilled=sel["distilled"],
                mode=mode, image=image,
                duration_s=duration_s, seed=seed, out=out, **size_kw, **kw)
            model_label = sel["label"] or cap.backend
            result = VideoResult(
                path=path, engine=cap.backend, routed="neural",
                note=f"diffusion-generated with {model_label} "
                     f"({cap.reason})", prompt=prompt,
                meta={"vram_gb": cap.vram_gb, "model": model_label,
                      "loras": [l["name"] for l in sel["loras"]]})
            record_ledger({"kind": "videogen.generate", "path": path,
                           "engine": cap.backend, "routed": "neural",
                           "model": model_label,
                           "loras": [l["name"] for l in sel["loras"]]})
            return result
        if not allow_fallback:
            if sel["label"]:
                raise VideogenError(
                    f"neural unavailable: selected model "
                    f"{sel['label']!r}: {cap.reason}")
            raise VideogenError(f"neural unavailable: {cap.reason}")
        fallback_note = cap.reason
        if sel["label"]:
            fallback_note = f"selected model {sel['label']!r}: {cap.reason}"
    else:
        fallback_note = "motion studio explicitly requested"

    path, motion_note = _motion_fallback(
        prompt or "untitled", image=image, duration_s=duration_s,
        size=size, fps=fps)
    result = VideoResult(
        path=path, engine="motion", routed="motion",
        note=f"{fallback_note}. {motion_note}", prompt=prompt,
        meta={"fallback_reason": fallback_note,
              "requested_model": sel["label"],
              "requested_loras": [l["name"] for l in sel["loras"]]})
    record_ledger({"kind": "videogen.generate", "path": path,
                   "engine": "motion", "routed": "motion"})
    return result


def request_hero_clip(prompt: str, *, duration_s: float = 5.0,
                      seed: int = 0,
                      prefer: str = "auto",
                      model: str | None = None,
                      lora: Any = None) -> dict[str, Any]:
    """One neural hero clip for clip-level consumers (shorts surfaces).

    Exposed via chat ``/shorts hero`` and CLI ``nm shorts hero``. The
    contentops ShortPipeline does not call this — its scenes are stills
    (see the module docstring); this is the API for a caller that wants
    a genuine diffusion hero shot.

    Returns ``{"ok": True, "path": …}`` on success or
    ``{"ok": False, "reason": …}`` when neural is unavailable — the
    caller decides the fallback; this function never silently
    substitutes motion graphics for a neural clip. ``model``/``lora``
    take the same registry selection as :func:`generate`; selection
    errors come back as an ``ok: False`` dict naming the problem.
    """
    try:
        sel = _resolve_selection(prefer, model, lora)
    except VideogenError as exc:
        return {"ok": False, "reason": str(exc),
                "suggestion": "fix the model/LoRA selection, or drop it "
                              "for the stock checkpoint"}
    cap = neural_capability(prefer=sel["backend"] or prefer,
                            min_vram_gb=sel["floor"])
    if not cap.available:
        reason = cap.reason
        if sel["label"]:
            reason = f"selected model {sel['label']!r}: {cap.reason}"
        return {"ok": False, "reason": reason,
                "suggestion": "use motion_studio.kenburns over a generated "
                              "still for the hero shot instead"}
    try:
        _check_lora_backend(sel["loras"], cap.backend,
                            sel["label"] or cap.backend)
        path = _neural_generate(prompt, cap.backend,
                                model_id=sel["source"],
                                loras=sel["loras"],
                                distilled=sel["distilled"],
                                mode="t2v", duration_s=duration_s,
                                seed=seed)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{cap.backend} failed: {exc}",
                "suggestion": "retry, or fall back to motion studio"}
    return {"ok": True, "path": path, "backend": cap.backend,
            "model": sel["label"] or cap.backend,
            "loras": [l["name"] for l in sel["loras"]],
            "vram_gb": cap.vram_gb}
