"""Pluggable generative (AI) image-edit backends for the studio.

Natural-language instruction → AI-modified image ("make a bird sit on the
tree", "put him in a grand room", "make it sunset"). There is no image
generation elsewhere in the repo (``nomorals/media/`` is music/playback
only), so this module is the single home for it.

Interface::

    backend = get_backend()  # auto: diffusers → hf → clear error
    out = backend.edit(image, "make it sunset", mask=None,
                       strength=0.75, seed=42)

Backends:

- ``hf`` (:class:`HFInferenceBackend`): serverless image-to-image through
  ``huggingface_hub.InferenceClient.image_to_image``. Model from
  ``MEDIA_GEN_MODEL`` (default ``black-forest-labs/FLUX.1-Kontext-dev``,
  the documented instruction-editing model), provider from
  ``MEDIA_GEN_PROVIDER``, token from ``HF_TOKEN`` env only — never chat,
  never logs.
- ``diffusers`` (:class:`DiffusersBackend`): local pipeline, active only
  when ``diffusers`` + ``torch`` import. Model from
  ``MEDIA_GEN_DIFFUSERS_MODEL`` (default ``timbrooks/instruct-pix2pix``).
- ``off`` / unconfigured: :class:`GenerativeEditError` naming the env var —
  never a fake edit.

``mask`` targets a region: a ``(l, t, r, b)`` box, a path to a mask image,
or a PIL ``L`` image. Both backends apply it as a feathered composite of
the generated result over the original (honest region targeting; HF
providers don't take masks natively).

No backend is contacted at import time, and no real model call happens
without explicit configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .images import MediaEditError, _require_pillow

_log = get_logger(__name__)


class GenerativeEditError(MediaEditError):
    """Generative edit unavailable or failed."""


DEFAULT_HF_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_DIFFUSERS_MODEL = "timbrooks/instruct-pix2pix"


# ---------------------------------------------------------------------------
# backend interface
# ---------------------------------------------------------------------------

class GenerativeBackend:
    """edit(image, instruction, ...) -> new PIL image."""

    name = "base"

    def edit(self, image: Any, instruction: str, *,
             mask: Any | None = None,
             strength: float = 0.75,
             seed: int | None = None,
             negative_prompt: str | None = None,
             steps: int | None = None,
             guidance_scale: float | None = None) -> Any:
        raise NotImplementedError

    def describe(self) -> str:
        return f"generative backend '{self.name}'"


def _as_mask(image_size: tuple[int, int], mask: Any) -> Any:
    """Normalize mask → L-mode PIL image (white = regenerate)."""
    Image = _require_pillow()
    from PIL import ImageFilter
    w, h = image_size
    if isinstance(mask, (list, tuple)):
        m = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(m)
        d.rectangle([int(v) for v in mask], fill=255)
    elif isinstance(mask, (str, Path)):
        m = Image.open(mask).convert("L").resize((w, h), Image.BILINEAR)
    else:  # assume PIL image
        m = mask.convert("L").resize((w, h), Image.BILINEAR)
    # feather the edge so the composite doesn't show a hard seam
    return m.filter(ImageFilter.GaussianBlur(6))


def _apply_mask(original: Any, generated: Any, mask: Any | None) -> Any:
    """Composite ``generated`` over ``original`` through ``mask``."""
    Image = _require_pillow()
    gen = generated.convert("RGB").resize(original.size, Image.LANCZOS)
    if mask is None:
        return gen
    m = _as_mask(original.size, mask)
    return Image.composite(gen, original.convert("RGB"), m)


# ---------------------------------------------------------------------------
# Hugging Face inference backend
# ---------------------------------------------------------------------------

class HFInferenceBackend(GenerativeBackend):
    """Serverless img2img via huggingface_hub.InferenceClient.image_to_image."""

    name = "hf"

    def __init__(self, model: str | None = None,
                 provider: str | None = None,
                 token: str | None = None) -> None:
        self.model = model or os.environ.get("MEDIA_GEN_MODEL",
                                             DEFAULT_HF_MODEL)
        self.provider = provider or os.environ.get("MEDIA_GEN_PROVIDER")
        # env only — never accept a token from chat/logs
        self.token = token or os.environ.get("HF_TOKEN", "") or None

    def _client(self) -> Any:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise GenerativeEditError(
                "generative edit needs huggingface_hub: "
                "pip install huggingface_hub (or set MEDIA_GEN_BACKEND=diffusers "
                "with diffusers+torch installed)") from exc
        kwargs: dict[str, Any] = {}
        if self.provider:
            kwargs["provider"] = self.provider
        if self.token:
            kwargs["api_key"] = self.token
        return InferenceClient(**kwargs)

    def edit(self, image: Any, instruction: str, *,
             mask: Any | None = None,
             strength: float = 0.75,
             seed: int | None = None,
             negative_prompt: str | None = None,
             steps: int | None = None,
             guidance_scale: float | None = None) -> Any:
        if not instruction or not instruction.strip():
            raise GenerativeEditError("generative edit needs an instruction")
        client = self._client()
        params: dict[str, Any] = {"strength": float(strength)}
        if seed is not None:
            params["seed"] = int(seed)
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if steps:
            params["num_inference_steps"] = int(steps)
        if guidance_scale:
            params["guidance_scale"] = float(guidance_scale)
        _log.info("generative edit via hf model=%s instruction=%.60r",
                  self.model, instruction)
        try:
            # Documented API: image_to_image(image, prompt, model=..., **kwargs)
            # → PIL.Image. Extra params (strength/seed) pass through **kwargs.
            result = client.image_to_image(
                image, prompt=instruction, model=self.model, **params)
        except Exception as exc:
            raise GenerativeEditError(
                f"HF inference failed for model {self.model!r}: {exc}") from exc
        return _apply_mask(image, result, mask)

    def describe(self) -> str:
        return (f"generative backend 'hf' (model={self.model}, "
                f"provider={self.provider or 'default'})")


# ---------------------------------------------------------------------------
# local diffusers backend (optional)
# ---------------------------------------------------------------------------

class DiffusersBackend(GenerativeBackend):
    """Local img2img via diffusers. Activates only if diffusers+torch import."""

    name = "diffusers"

    def __init__(self, model: str | None = None,
                 device: str | None = None) -> None:
        self.model = model or os.environ.get("MEDIA_GEN_DIFFUSERS_MODEL",
                                             DEFAULT_DIFFUSERS_MODEL)
        self.device = device  # resolved lazily
        self._pipe: Any = None

    @staticmethod
    def available() -> bool:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _pipe_for(self, image: Any) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise GenerativeEditError(
                "generative edit needs diffusers+torch: "
                "pip install diffusers torch (or set MEDIA_GEN_BACKEND=hf "
                "with HF_TOKEN in the environment)") from exc
        device = self.device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _log.info("loading diffusers model %s on %s", self.model, device)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model, torch_dtype=dtype, use_safetensors=True)
        pipe = pipe.to(device)
        self._pipe = pipe
        return pipe

    def edit(self, image: Any, instruction: str, *,
             mask: Any | None = None,
             strength: float = 0.75,
             seed: int | None = None,
             negative_prompt: str | None = None,
             steps: int | None = None,
             guidance_scale: float | None = None) -> Any:
        if not instruction or not instruction.strip():
            raise GenerativeEditError("generative edit needs an instruction")
        import torch
        pipe = self._pipe_for(image)
        rgb = image.convert("RGB")
        kwargs: dict[str, Any] = {"strength": float(strength)}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if steps:
            kwargs["num_inference_steps"] = int(steps)
        if guidance_scale:
            kwargs["guidance_scale"] = float(guidance_scale)
        generator = torch.Generator().manual_seed(int(seed)) \
            if seed is not None else None
        try:
            result = pipe(instruction, image=rgb,
                          generator=generator, **kwargs).images[0]
        except Exception as exc:
            raise GenerativeEditError(
                f"diffusers edit failed for model {self.model!r}: {exc}"
            ) from exc
        return _apply_mask(image, result, mask)

    def describe(self) -> str:
        return f"generative backend 'diffusers' (model={self.model})"


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

_NO_BACKEND_MSG = (
    "no generative backend is configured. Options: "
    "pip install huggingface_hub and set HF_TOKEN (serverless, "
    "MEDIA_GEN_BACKEND=hf), or pip install diffusers torch for local "
    "inference (MEDIA_GEN_BACKEND=diffusers). "
    "Set MEDIA_GEN_BACKEND=off to silence this check."
)


def get_backend(name: str | None = None) -> GenerativeBackend:
    """Resolve a generative backend.

    ``name`` or ``MEDIA_GEN_BACKEND``: "auto" (default), "hf", "diffusers",
    "off". auto prefers local diffusers when importable, else HF when
    huggingface_hub imports, else raises a clear error (never a fake edit).
    """
    want = (name or os.environ.get("MEDIA_GEN_BACKEND", "auto")).lower()
    if want == "off":
        raise GenerativeEditError(_NO_BACKEND_MSG)
    if want == "hf":
        return HFInferenceBackend()
    if want == "diffusers":
        if not DiffusersBackend.available():
            raise GenerativeEditError(
                "MEDIA_GEN_BACKEND=diffusers but diffusers/torch are not "
                "installed")
        return DiffusersBackend()
    if want == "auto":
        if DiffusersBackend.available():
            return DiffusersBackend()
        try:
            import huggingface_hub  # noqa: F401
            return HFInferenceBackend()
        except ImportError:
            pass
        raise GenerativeEditError(_NO_BACKEND_MSG)
    raise GenerativeEditError(
        f"unknown MEDIA_GEN_BACKEND={want!r}; use auto|hf|diffusers|off")


def backend_status() -> dict[str, Any]:
    """Report which generative backends are usable (no model calls)."""
    try:
        import huggingface_hub  # noqa: F401
        hf = True
    except ImportError:
        hf = False
    return {
        "hf_installed": hf,
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "hf_model": os.environ.get("MEDIA_GEN_MODEL", DEFAULT_HF_MODEL),
        "hf_provider": os.environ.get("MEDIA_GEN_PROVIDER"),
        "diffusers_available": DiffusersBackend.available(),
        "diffusers_model": os.environ.get("MEDIA_GEN_DIFFUSERS_MODEL",
                                          DEFAULT_DIFFUSERS_MODEL),
        "selected": os.environ.get("MEDIA_GEN_BACKEND", "auto"),
    }


# ---------------------------------------------------------------------------
# studio op (registered into the images allowlist on import)
# ---------------------------------------------------------------------------

def op_generative_edit(img: Any, instruction: str, *,
                       mask: Any | None = None,
                       strength: float = 0.75,
                       seed: int | None = None,
                       backend: str | None = None,
                       negative_prompt: str | None = None,
                       steps: int | None = None,
                       guidance_scale: float | None = None) -> Any:
    """AI instruction edit as a chain op. ``mask``: (l,t,r,b) box, a mask
    image path, or a PIL L image — serializable forms survive project
    save/load; re-runs the backend on render (non-destructive)."""
    be = get_backend(backend)
    # normalize once at the op boundary: backends always get an L image
    m = _as_mask(img.size, mask) if mask is not None else None
    return be.edit(img, instruction, mask=m, strength=strength, seed=seed,
                   negative_prompt=negative_prompt, steps=steps,
                   guidance_scale=guidance_scale)


def _register() -> None:
    from . import images as _images
    _images._OP_FUNCS["generative_edit"] = op_generative_edit
    _images.OP_ALLOWLIST.add("generative_edit")


_register()
