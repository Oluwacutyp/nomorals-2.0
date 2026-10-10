"""Local text-to-video model router (build-map #26).

Mirrors :mod:`nomorals.media_edit.models` (the image router, #22): a
curated registry of video models, a ``VideoModelRouter`` that picks
(backend, model, workflow) for a request, and a top-level
:func:`generate_video` that routes then executes.

The lineup, by design:

- **Local (Apache 2.0, no license traps):** Wan 2.2 14B / 5B through
  ComfyUI (#21) on workstation profiles. Unlike the FLUX dev family
  (non-commercial), Wan 2.2 is Apache 2.0 — so there is no
  private/public audience split here. Documented explicitly because
  the image router needed one and this one doesn't.
- **Paid API:** Veo 3.1 (hero shots, via the existing
  :mod:`nomorals.connectors.googleflow` connector — its
  ``confirm_or_checkpoint`` gating is never bypassed), MiniMax H3
  (daily driver), Kling (motion). Unconfigured APIs raise an honest
  ``VideoModelError`` naming the env var — never a fake render.

Budgets: ``"standard"`` (default) → local Wan or MiniMax;
``"hero"`` → Veo 3.1; ``"motion"`` → Kling. A budget naming a paid
API whose key isn't set fails closed with setup steps.

Registry maintenance note: model files, VRAM numbers, and API env
vars change upstream. ``VIDEO_MODELS`` is a curated snapshot
(Oct 2026); treat stale entries as bugs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind

_log = get_logger(__name__)


class VideoModelError(Exception):
    """No video model can serve this request (profile/key/backend gap)."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
#
# Fields per model:
#   license   — the real upstream license; keep honest.
#   vram_gb   — approximate VRAM for local inference (0 = API).
#   speed     — "fast" | "medium" | "slow".
#   quality   — 1..5, relative quality ceiling.
#   backend   — "comfy" | "googleflow" | "minimax" | "kling".
#   workflow  — ComfyUI workflow template (comfy backend only).
#   model_file— ComfyUI diffusion-model filename (comfy backend only).
#   vae_file  — ComfyUI VAE filename (comfy backend only).
#   fps       — output frame rate (comfy backend only).
#   max_duration_s — longest supported clip (seconds).
#   paid      — True: costs money per render, needs owner confirmation.
#   key_env   — env vars holding the API key, in preference order (API only).
#   setup     — human setup steps when the key is missing (API only).
#   capabilities — intents served: video | video_hero | video_motion.

VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "wan22-14b": {
        "license": "Apache 2.0",
        "vram_gb": 24,
        "speed": "slow",
        "quality": 5,
        "backend": "comfy",
        "workflow": "wan22_t2v",
        "model_file": "wan2.2_t2v_high_noise_14B_fp8_e4m3fn.safetensors",
        "vae_file": "wan_2.1_vae.safetensors",
        "fps": 16,
        "max_duration_s": 10,
        "paid": False,
        "key_env": (),
        "setup": "",
        "capabilities": ("video",),
    },
    "wan22-5b": {
        "license": "Apache 2.0",
        "vram_gb": 20,
        "speed": "medium",
        "quality": 4,
        "backend": "comfy",
        "workflow": "wan22_t2v",
        "model_file": "wan2.2_t2v_high_noise_5B_fp8_e4m3fn.safetensors",
        "vae_file": "wan_2.1_vae.safetensors",
        "fps": 16,
        "max_duration_s": 10,
        "paid": False,
        "key_env": (),
        "setup": "",
        "capabilities": ("video",),
    },
    "veo-3.1": {
        "license": "Paid API (Google)",
        "vram_gb": 0,
        "speed": "medium",
        "quality": 5,
        "backend": "googleflow",
        "workflow": None,
        "model_file": None,
        "vae_file": None,
        "fps": 24,
        "max_duration_s": 8,
        "paid": True,
        "key_env": ("GOOGLE_FLOW_API_KEY", "GEMINI_API_KEY"),
        "setup": ("set GOOGLE_FLOW_API_KEY "
                  "(https://aistudio.google.com/apikey)"),
        "capabilities": ("video", "video_hero"),
    },
    "minimax-h3": {
        "license": "Paid API (MiniMax)",
        "vram_gb": 0,
        "speed": "fast",
        "quality": 4,
        "backend": "minimax",
        "workflow": None,
        "model_file": None,
        "vae_file": None,
        "fps": 24,
        "max_duration_s": 10,
        "paid": True,
        "key_env": ("MINIMAX_API_KEY",),
        "setup": "set MINIMAX_API_KEY (https://www.minimax.io)",
        "capabilities": ("video",),
    },
    "kling": {
        "license": "Paid API (Kling)",
        "vram_gb": 0,
        "speed": "medium",
        "quality": 4,
        "backend": "kling",
        "workflow": None,
        "model_file": None,
        "vae_file": None,
        "fps": 24,
        "max_duration_s": 10,
        "paid": True,
        "key_env": ("KLING_API_KEY",),
        "setup": "set KLING_API_KEY (https://klingai.com)",
        "capabilities": ("video", "video_motion"),
    },
    "ltx-2.3": {
        # Local speed king + the only open model with native synchronized
        # audio (22B DiT). Community license: free under $10M revenue.
        # No ComfyUI template ships with Devon yet — the route carries
        # workflow=None and setup text; bring your own LTX workflow JSON.
        "license": "LTX-2 Community License (free <$10M revenue)",
        "vram_gb": 24,
        "speed": "fast",
        "quality": 5,
        "backend": "comfy",
        "workflow": None,
        "model_file": "ltx-2.3-22b-distilled.safetensors",
        "vae_file": None,
        "fps": 24,
        "max_duration_s": 20,
        "paid": False,
        "key_env": (),
        "setup": ("local LTX-2.3 needs a workstation + ComfyUI with the "
                  "LTX nodes installed, and your own LTX workflow template "
                  "(none ships with Devon yet)"),
        "capabilities": ("video",),
    },
}

INTENTS = ("video", "video_hero", "video_motion")
BUDGETS = ("standard", "hero", "motion")

#: Budget → preferred paid model (used only when local is unavailable or
#: the budget explicitly asks for API quality).
_BUDGET_API = {"standard": "minimax-h3", "hero": "veo-3.1",
               "motion": "kling"}


@dataclass(frozen=True)
class VideoRoute:
    """One routing decision."""

    backend: str  # "comfy" | "googleflow" | "minimax" | "kling"
    model: str  # registry key
    workflow: str | None
    reason: str
    paid: bool
    fps: int
    max_duration_s: int


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _api_configured(model: dict[str, Any]) -> bool:
    return any(os.environ.get(k) for k in model["key_env"])


def _cuda_vram_gb() -> float | None:
    """Total VRAM of the first CUDA device, or None when unknowable.

    Best-effort: torch may not be installed, CUDA may be absent. Never
    raises — unknown VRAM is data, not an error.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.get_device_properties(0).total_memory) / 1e9
    except Exception:  # noqa: BLE001 - probe, None is a valid answer
        return None


def _comfy_reachable() -> bool:
    try:
        from .comfy import comfy_available
        ok, _why = comfy_available()
        return ok
    except Exception:  # noqa: BLE001 - probe
        return False


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

class VideoModelRouter:
    """Pick the right model + backend for a text-to-video request.

    Routing order per budget:

    - ``standard``: local Wan 2.2 on a workstation with ComfyUI up
      (14B when VRAM allows, else 5B) → MiniMax API when configured →
      error.
    - ``hero``: Veo 3.1 API (paid, needs a key) → local Wan 2.2-14B as
      the free fallback → error.
    - ``motion``: Kling API (paid, needs a key) → local Wan 2.2 → error.

    ``prefer="ltx"`` swaps the local pick to LTX-2.3 (faster, native
    synchronized audio) when the workstation + ComfyUI path is up; the
    route notes that no LTX workflow template ships yet.

    Profiles other than workstation never get the local path (no local
    GPU to speak of): termux/laptop go straight to configured APIs.
    Never fakes: raises :class:`VideoModelError` when nothing can serve.
    """

    def __init__(self,
                 registry: dict[str, dict[str, Any]] | None = None) -> None:
        self.registry = registry if registry is not None else VIDEO_MODELS

    # -- introspection -------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        """Describe the registry (no audience filter — all entries are
        Apache 2.0 or paid API; the docstring explains why)."""
        out = []
        for name in sorted(self.registry):
            m = self.registry[name]
            out.append({
                "name": name,
                "license": m["license"],
                "quality": m["quality"],
                "speed": m["speed"],
                "backend": m["backend"],
                "paid": m["paid"],
                "capabilities": list(m["capabilities"]),
                "configured": (True if not m["paid"]
                               else _api_configured(m)),
            })
        return out

    # -- routing -------------------------------------------------------------
    def route(self, intent: str = "video", *,
              profile: str | None = None,
              budget: str = "standard",
              prefer: str = "wan") -> VideoRoute:
        intent = (intent or "video").strip().lower()
        if intent not in INTENTS:
            raise VideoModelError(
                f"unknown video intent {intent!r}; use {INTENTS}")
        budget = (budget or "standard").strip().lower()
        if budget not in BUDGETS:
            raise VideoModelError(
                f"unknown video budget {budget!r}; use {BUDGETS}")
        prefer = (prefer or "wan").strip().lower()
        if prefer not in ("wan", "ltx"):
            raise VideoModelError(
                f"unknown local preference {prefer!r}; use wan|ltx")
        profile = ((profile or get_profile_kind()) or "").strip().lower()

        local_ok = profile == "workstation" and _comfy_reachable()

        # 1. local open models (free) — preferred for standard,
        #    fallback for hero/motion when the paid key is missing.
        if local_ok:
            local = self._pick_ltx() if prefer == "ltx" else self._pick_wan()
            if budget == "standard":
                return local
            # hero/motion: paid API first, local as the honest free fallback.
            api_name = _BUDGET_API[budget]
            api = self.registry[api_name]
            if _api_configured(api):
                return self._api_route(api_name, api, budget, profile,
                                       reason_suffix="key configured")
            _log.info("video route: %s not configured; falling back to %s",
                      api_name, local.model)
            return local

        # 2. no local path — configured paid API by budget.
        api_name = _BUDGET_API[budget]
        api = self.registry[api_name]
        if _api_configured(api):
            return self._api_route(api_name, api, budget, profile,
                                   reason_suffix="local unavailable, "
                                   "key configured")

        # 3. nothing can serve — say exactly why.
        problems = []
        if profile != "workstation":
            problems.append(
                f"local Wan 2.2 needs a workstation profile "
                f"(current: {profile or 'unknown'})")
        elif not _comfy_reachable():
            problems.append("ComfyUI not reachable "
                            "(start it or set COMFYUI_HOST/COMFYUI_PORT)")
        problems.append(f"{api_name}: {api['setup']}")
        raise VideoModelError(
            "no video model available — " + "; ".join(problems))

    def _pick_ltx(self) -> VideoRoute:
        """LTX-2.3 local route: fastest open model, native synced audio.

        No ComfyUI template ships with Devon, so the route's workflow is
        None and the reason says exactly what to bring. Never fakes a
        renderable path.
        """
        m = self.registry["ltx-2.3"]
        return VideoRoute(
            backend="comfy", model="ltx-2.3", workflow=None, paid=False,
            fps=m["fps"], max_duration_s=m["max_duration_s"],
            reason=("ltx-2.3 via comfy (fast, native synchronized audio): "
                    "no workflow template ships with Devon — supply your "
                    "own LTX ComfyUI workflow JSON"))

    def _pick_wan(self) -> VideoRoute:
        """wan22-14b when VRAM allows, else wan22-5b. Unprobed VRAM
        takes the safer 5B — stated in the reason, never guessed."""
        vram = _cuda_vram_gb()
        if vram is not None and vram >= 22:
            name = "wan22-14b"
            note = f"{vram:.0f}GB VRAM"
        else:
            name = "wan22-5b"
            note = (f"{vram:.0f}GB VRAM" if vram is not None
                    else "VRAM unprobed — safe pick")
        m = self.registry[name]
        return VideoRoute(
            backend="comfy", model=name, workflow=m["workflow"], paid=False,
            fps=m["fps"], max_duration_s=m["max_duration_s"],
            reason=f"{name} via comfy ({note}): free Apache 2.0 local render")

    def _api_route(self, name: str, m: dict[str, Any], budget: str,
                   profile: str, *, reason_suffix: str) -> VideoRoute:
        return VideoRoute(
            backend=m["backend"], model=name, workflow=None, paid=True,
            fps=m["fps"], max_duration_s=m["max_duration_s"],
            reason=f"{name} via {m['backend']} ({budget} budget, {profile}; "
                   f"{reason_suffix}) — PAID, needs owner confirmation")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def generate_video(prompt: str, *, duration_s: int = 5,
                   budget: str = "standard", profile: str | None = None,
                   seed: int | None = None,
                   negative_prompt: str | None = None,
                   steps: int | None = None,
                   width: int | None = None, height: int | None = None,
                   out_dir: str | os.PathLike[str] | None = None,
                   confirmed: bool = False, db: Any = None,
                   context: Any = None,
                   progress_cb: Any = None) -> Path:
    """Route and render a text-to-video clip. Returns the mp4 path.

    Local (Wan 2.2 via ComfyUI) renders immediately. Paid API routes
    go through the existing confirmation gating: ``confirmed=True``
    attests the owner approved the exact render; otherwise ``db`` opens
    a human checkpoint (see :mod:`nomorals.connectors._confirm`); with
    neither, raises :class:`VideoModelError` naming the confirmation
    needed. Paid gates are never bypassed.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise VideoModelError("text-to-video needs a prompt")
    duration_s = int(duration_s)
    if duration_s < 1:
        raise VideoModelError(
            f"duration_s must be >= 1, got {duration_s}")
    router = VideoModelRouter()
    route = router.route("video", profile=profile, budget=budget)
    if duration_s > route.max_duration_s:
        raise VideoModelError(
            f"{route.model} supports up to {route.max_duration_s}s, "
            f"asked {duration_s}s")
    _log.info("video render: %s", route.reason)

    if route.backend == "comfy":
        from .comfy import ComfyUIBackend
        be = ComfyUIBackend(timeout_s=max(1800.0, duration_s * 120.0))
        video_path = be.generate_video(
            prompt, duration_s=duration_s, fps=route.fps,
            seed=seed, negative_prompt=negative_prompt, steps=steps,
            width=width, height=height,
            model_file=VIDEO_MODELS[route.model]["model_file"],
            vae_file=VIDEO_MODELS[route.model]["vae_file"],
            progress_cb=progress_cb)
    elif route.backend == "googleflow":
        from ..connectors.googleflow import GoogleFlowConnector
        conn = GoogleFlowConnector()
        result = conn.generate_video(
            prompt, duration_seconds=min(duration_s, 8),
            negative_prompt=negative_prompt or "",
            confirmed=confirmed, db=db, context=context)
        video_path = _save_api_video(result.get("video_bytes") or b"",
                                     out_dir, route.model)
    else:  # minimax / kling — no connector yet; honest, not fake.
        m = VIDEO_MODELS[route.model]
        raise VideoModelError(
            f"{route.model} has no connector yet — {m['setup']}, then ask "
            "for the connector to be built. Nothing was rendered.")

    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        final = dest / Path(video_path).name
        if Path(video_path).resolve() != final.resolve():
            Path(video_path).replace(final)
        return final
    return Path(video_path)


def _save_api_video(data: bytes, out_dir: Any, model: str) -> Path:
    if not data:
        raise VideoModelError(
            f"{model} returned no video bytes — nothing was rendered")
    import time as _time
    dest = Path(out_dir) if out_dir else Path("data/generations")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"video-{_time.strftime('%Y%m%d-%H%M%S')}.mp4"
    path.write_bytes(data)
    return path
