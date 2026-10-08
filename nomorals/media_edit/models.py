"""License-aware image model router (build-map #22).

Sits above the generative backends (:mod:`nomorals.media_edit.generate`,
:mod:`nomorals.media_edit.comfy`) and answers one question: *which model,
on which backend, for this request?* The two axes that matter are the
**intent** (generate / edit / draft / upscale) and the **audience**:

- ``"private"`` — the owner's own DM / console. Non-commercial research
  models (the FLUX dev family) are allowed here, because the owner runs
  them for themselves.
- ``"public"`` — community surfaces, groups, anything shared. Apache 2.0
  and other open licenses ONLY. The non-commercial models are excluded
  **structurally**: the public-audience filter runs before any scoring,
  so no intent string can ever route a FLUX dev model to a public
  surface. This is not a prompt-level guardrail — it is a filter on the
  candidate list.

Model registry maintenance note: checkpoints, HF ids, and licenses
change upstream. ``IMAGE_MODELS`` is a curated snapshot (Oct 2026);
treat stale entries as bugs and update them. License fields must stay
honest — when in doubt, mark a model ``public_ok=False`` until its
license is verified.

For ambiguous requests that don't match the narrow chat patterns
(``_image_intent`` in coremind), no second router is needed: the normal
agent loop already routes through the media tools, and the tools call
:func:`route` themselves. This module documents that contract rather
than duplicating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind

_log = get_logger(__name__)


class ImageModelError(Exception):
    """No model can serve this request (license/profile/backend gap)."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
#
# Fields per model:
#   license      — the real upstream license string; keep honest.
#   public_ok    — True only for licenses that permit public/community use.
#   vram_gb      — approximate VRAM for local inference (0 = serverless/API).
#   speed        — "fast" | "medium" | "slow".
#   quality      — 1..5, relative quality ceiling.
#   backend      — "comfy" | "hf" | "diffusers".
#   workflow     — ComfyUI workflow template name (comfy backend only).
#   checkpoint   — ComfyUI checkpoint filename (comfy backend only).
#   hf_id        — Hugging Face repo id (hf/diffusers backends).
#   capabilities — intents this model serves: generate | edit | draft | upscale.
#   steps        — suggested inference steps (None = backend/model default).
#
# Maintenance: update as upstream changes (see module docstring).

IMAGE_MODELS: dict[str, dict[str, Any]] = {
    # -- private-only: non-commercial licenses ------------------------------
    "flux2-dev": {
        "license": "FLUX.2-dev Non-Commercial License",
        "license_note": "non-commercial research use only — private surfaces",
        "public_ok": False,
        "vram_gb": 24,
        "speed": "slow",
        "quality": 5,
        "backend": "comfy",
        "workflow": "txt2img",
        "checkpoint": "flux2-dev.safetensors",
        "hf_id": "black-forest-labs/FLUX.2-dev",
        "capabilities": ("generate", "draft"),
        "steps": None,
    },
    "flux1-kontext-dev": {
        "license": "FLUX.1-dev Non-Commercial License",
        "license_note": "non-commercial research use only — private surfaces",
        "public_ok": False,
        "vram_gb": 24,
        "speed": "medium",
        "quality": 5,
        "backend": "hf",
        "workflow": None,
        "checkpoint": None,
        "hf_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "capabilities": ("edit",),
        "steps": None,
    },
    # -- public-safe: open licenses ------------------------------------------
    "qwen-image-2512": {
        "license": "Apache 2.0",
        "license_note": "",
        "public_ok": True,
        "vram_gb": 20,
        "speed": "medium",
        "quality": 4,
        "backend": "comfy",
        "workflow": "txt2img",
        "checkpoint": "qwen_image_2512.safetensors",
        "hf_id": "Qwen/Qwen-Image-2512",
        "capabilities": ("generate",),
        "steps": None,
    },
    "qwen-image-edit": {
        "license": "Apache 2.0",
        "license_note": "",
        "public_ok": True,
        "vram_gb": 20,
        "speed": "medium",
        "quality": 4,
        "backend": "hf",
        "workflow": None,
        "checkpoint": None,
        "hf_id": "Qwen/Qwen-Image-Edit",
        "capabilities": ("edit",),
        "steps": None,
    },
    "z-image-turbo": {
        "license": "Apache 2.0",  # Tongyi-MAI/Z-Image-Turbo model card
        "license_note": "8-step distilled draft model",
        "public_ok": True,
        "vram_gb": 16,
        "speed": "fast",
        "quality": 3,
        "backend": "comfy",
        "workflow": "txt2img",
        "checkpoint": "z_image_turbo.safetensors",
        "hf_id": "Tongyi-MAI/Z-Image-Turbo",
        "capabilities": ("generate", "draft"),
        "steps": 8,
    },
    "flux1-schnell": {
        "license": "Apache 2.0",
        "license_note": "served via HF Inference API (no local VRAM)",
        "public_ok": True,
        "vram_gb": 0,
        "speed": "fast",
        "quality": 3,
        "backend": "hf",
        "workflow": None,
        "checkpoint": None,
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "capabilities": ("generate", "draft"),
        "steps": None,
    },
    "sdxl-base": {
        "license": "CreativeML Open RAIL-M",
        "license_note": "broad LoRA ecosystem, modest VRAM",
        "public_ok": True,
        "vram_gb": 8,
        "speed": "medium",
        "quality": 3,
        "backend": "comfy",
        "workflow": "txt2img",
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "capabilities": ("generate",),
        "steps": None,
    },
}

INTENTS = ("generate", "edit", "draft", "upscale")
AUDIENCES = ("private", "public")

_SPEED_RANK = {"fast": 0, "medium": 1, "slow": 2}


@dataclass(frozen=True)
class Route:
    """One routing decision."""

    backend: str  # "comfy" | "hf" | "diffusers" | "local"
    model: str  # registry key ("pil-lanczos" for the local upscale path)
    workflow: str | None
    reason: str
    hf_id: str | None = None
    checkpoint: str | None = None
    steps: int | None = None


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

class ImageModelRouter:
    """Pick the right model + backend for an image intent.

    License enforcement is structural: for ``audience="public"`` the
    non-commercial models are removed from the candidate list *before*
    scoring, so they can never win regardless of intent.
    """

    def __init__(self, registry: dict[str, dict[str, Any]] | None = None):
        self.registry = registry if registry is not None else IMAGE_MODELS

    # -- introspection -------------------------------------------------------
    def list_models(self, audience: str | None = None) -> list[dict[str, Any]]:
        """Describe the registry. ``audience="public"`` applies the same
        license filter routing uses — what you see is what can be picked."""
        if audience is not None and audience not in AUDIENCES:
            raise ImageModelError(
                f"unknown audience {audience!r}; use {AUDIENCES}")
        out = []
        for name in sorted(self.registry):
            m = self.registry[name]
            if audience == "public" and not m["public_ok"]:
                continue
            out.append({
                "name": name,
                "license": m["license"],
                "public_ok": m["public_ok"],
                "quality": m["quality"],
                "speed": m["speed"],
                "backend": m["backend"],
                "capabilities": list(m["capabilities"]),
                "hf_id": m["hf_id"],
            })
        return out

    # -- routing -------------------------------------------------------------
    def route(self, intent: str, *, audience: str = "private",
              profile: str | None = None) -> Route:
        """Pick (backend, model, workflow) for ``intent``.

        ``intent``: generate | edit | draft (fast) | upscale.
        ``audience``: private (owner) | public (community — open licenses only).
        ``profile``: termux | laptop | workstation (default: detected).
        Never fakes: raises ImageModelError when nothing can serve.
        """
        intent = (intent or "").strip().lower()
        if intent not in INTENTS:
            raise ImageModelError(
                f"unknown image intent {intent!r}; use {INTENTS}")
        audience = (audience or "private").strip().lower()
        if audience not in AUDIENCES:
            raise ImageModelError(
                f"unknown audience {audience!r}; use {AUDIENCES}")
        profile = ((profile or get_profile_kind()) or "").strip().lower()

        # 1. license filter — structural, first, non-bypassable.
        cands = [(name, m) for name, m in self.registry.items()
                 if intent in m["capabilities"]
                 and (audience == "private" or m["public_ok"])]

        # 2. upscale is model-free: ComfyUI host if viable, else local PIL.
        if intent == "upscale":
            return self._route_upscale(profile, audience)

        if not cands:
            raise ImageModelError(
                f"no {audience} model serves intent {intent!r} "
                f"(license/profile gap)")

        # 3. profile filter: termux has no local GPU — HF API only.
        if profile == "termux":
            cands = [(n, m) for n, m in cands if m["backend"] == "hf"]
            if not cands:
                raise ImageModelError(
                    f"no HF-served model for intent {intent!r} on termux")
            preferred = "hf"
        else:
            preferred = self._preferred_backend(profile)

        # 4. score: backend preference, then intent-shaped quality/speed.
        def _key(item: tuple[str, dict[str, Any]]) -> tuple:
            _name, m = item
            backend_penalty = 0 if m["backend"] == preferred else 1
            if intent == "draft":
                return (backend_penalty, _SPEED_RANK[m["speed"]],
                        -m["quality"])
            return (backend_penalty, -m["quality"],
                    _SPEED_RANK[m["speed"]])

        name, m = sorted(cands, key=_key)[0]
        reason = (f"{name} via {m['backend']}"
                  + (f" ({m['workflow']})" if m["workflow"] else "")
                  + f": {audience} audience, {intent} intent, {profile} "
                  + ("GPU" if profile != "termux" else "API-only profile"))
        _log.info("image route: %s", reason)
        return Route(backend=m["backend"], model=name,
                     workflow=m["workflow"], reason=reason,
                     hf_id=m["hf_id"], checkpoint=m["checkpoint"],
                     steps=m["steps"])

    # -- internals -----------------------------------------------------------
    def _preferred_backend(self, profile: str) -> str:
        """Best backend for a GPU-capable profile. Probes, never guesses.

        workstation: comfy → diffusers → hf. laptop: comfy → hf.
        Never raises — falls back to "hf" on any probe failure.
        """
        try:
            from .comfy import comfy_available
            comfy_ok, _why = comfy_available()
        except Exception:  # noqa: BLE001 - probe is best-effort
            comfy_ok = False
        if comfy_ok:
            return "comfy"
        if profile == "workstation":
            try:
                from .generate import DiffusersBackend
                if DiffusersBackend.available():
                    return "diffusers"
            except Exception:  # noqa: BLE001 - optional dep probe
                pass
        return "hf"

    def _route_upscale(self, profile: str, audience: str) -> Route:
        """Upscale is model-free (Lanczos): use the ComfyUI host when one
        is reachable, else the local PIL path. License-neutral."""
        if profile != "termux":
            try:
                from .comfy import comfy_available
                comfy_ok, _why = comfy_available()
            except Exception:  # noqa: BLE001 - probe is best-effort
                comfy_ok = False
            if comfy_ok:
                reason = (f"comfy upscale workflow: {audience} audience, "
                          f"upscale intent, {profile} host")
                return Route(backend="comfy", model="sdxl-base",
                             workflow="upscale", reason=reason,
                             checkpoint="sd_xl_base_1.0.safetensors")
        reason = ("local Lanczos upscale (model-free): no GPU needed, "
                  f"{audience} audience")
        _log.info("image route: %s", reason)
        return Route(backend="local", model="pil-lanczos",
                     workflow="upscale", reason=reason)
