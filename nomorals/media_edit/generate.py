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
    """edit(image, instruction, ...) -> new PIL image.
    generate(prompt, ...) -> new PIL image (text-to-image)."""

    name = "base"

    def edit(self, image: Any, instruction: str, *,
             mask: Any | None = None,
             strength: float = 0.75,
             seed: int | None = None,
             negative_prompt: str | None = None,
             steps: int | None = None,
             guidance_scale: float | None = None,
             width: int | None = None,
             height: int | None = None) -> Any:
        raise NotImplementedError

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        """Text-to-image. Returns a list of PIL images (length n)."""
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
             guidance_scale: float | None = None,
             width: int | None = None,
             height: int | None = None) -> Any:
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
        if width:
            params["width"] = int(width)
        if height:
            params["height"] = int(height)
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

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        """Text-to-image via huggingface_hub.InferenceClient.text_to_image."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        client = self._client()
        params: dict[str, Any] = {}
        if seed is not None:
            params["seed"] = int(seed)
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if steps:
            params["num_inference_steps"] = int(steps)
        if guidance_scale:
            params["guidance_scale"] = float(guidance_scale)
        if width:
            params["width"] = int(width)
        if height:
            params["height"] = int(height)
        _log.info("text-to-image via hf model=%s n=%d prompt=%.60r",
                  self.model, n, prompt)
        out: list[Any] = []
        try:
            for i in range(n):
                # Vary seed per image in batch when a seed is given.
                p = dict(params)
                if seed is not None and n > 1:
                    p["seed"] = int(seed) + i
                result = client.text_to_image(
                    prompt, model=self.model, **p)
                out.append(result)
        except Exception as exc:
            raise GenerativeEditError(
                f"HF text-to-image failed for model {self.model!r}: {exc}"
            ) from exc
        return out

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
             guidance_scale: float | None = None,
             width: int | None = None,
             height: int | None = None) -> Any:
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
        if width:
            kwargs["width"] = int(width)
        if height:
            kwargs["height"] = int(height)
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

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        """Text-to-image via diffusers AutoPipelineForText2Image."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        import torch
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise GenerativeEditError(
                "text-to-image needs diffusers: pip install diffusers torch"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _log.info("loading diffusers txt2img model %s on %s", self.model,
                  device)
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model, torch_dtype=dtype, use_safetensors=True)
        pipe = pipe.to(device)
        kwargs: dict[str, Any] = {}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if steps:
            kwargs["num_inference_steps"] = int(steps)
        if guidance_scale:
            kwargs["guidance_scale"] = float(guidance_scale)
        if width:
            kwargs["width"] = int(width)
        if height:
            kwargs["height"] = int(height)
        if n > 1:
            kwargs["num_images_per_prompt"] = n
        generators = [torch.Generator().manual_seed(int(seed) + i)
                      for i in range(n)] if seed is not None else None
        try:
            result = pipe(prompt, generator=generators, **kwargs).images
        except Exception as exc:
            raise GenerativeEditError(
                f"diffusers txt2img failed for model {self.model!r}: {exc}"
            ) from exc
        return list(result)

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
        except ImportError:  # noqa: E103 - optional backend probe, absence is handled below
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
                       guidance_scale: float | None = None,
                       width: int | None = None,
                       height: int | None = None) -> Any:
    """AI instruction edit as a chain op. ``mask``: (l,t,r,b) box, a mask
    image path, or a PIL L image — serializable forms survive project
    save/load; re-runs the backend on render (non-destructive)."""
    be = get_backend(backend)
    # normalize once at the op boundary: backends always get an L image
    m = _as_mask(img.size, mask) if mask is not None else None
    return be.edit(img, instruction, mask=m, strength=strength, seed=seed,
                   negative_prompt=negative_prompt, steps=steps,
                   guidance_scale=guidance_scale, width=width, height=height)


def op_txt2img(prompt: str, *,
               seed: int | None = None,
               backend: str | None = None,
               negative_prompt: str | None = None,
               steps: int | None = None,
               guidance_scale: float | None = None,
               width: int | None = None,
               height: int | None = None,
               n: int = 1) -> Any:
    """Text-to-image as a chain op. Returns a single PIL image (n=1) or a
    list of PIL images (n>1)."""
    be = get_backend(backend)
    images = be.generate(prompt, seed=seed, negative_prompt=negative_prompt,
                         steps=steps, guidance_scale=guidance_scale,
                         width=width, height=height, n=n)
    return images[0] if n == 1 else images


def op_upscale(img: Any, scale: float = 2.0) -> Any:
    """Upscale an image with high-quality Lanczos resampling.

    Pure PIL — no model, no network. A stepping stone to a real
    super-resolution backend; genuinely useful for enlarging generations.
    """
    Image = _require_pillow()
    if scale <= 0:
        raise MediaEditError(f"upscale scale must be > 0, got {scale}")
    w, h = img.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


def _register() -> None:
    from . import images as _images
    _images._OP_FUNCS["generative_edit"] = op_generative_edit
    _images.OP_ALLOWLIST.add("generative_edit")
    _images._OP_FUNCS["txt2img"] = op_txt2img
    _images.OP_ALLOWLIST.add("txt2img")
    _images._OP_FUNCS["upscale"] = op_upscale
    _images.OP_ALLOWLIST.add("upscale")


_register()
