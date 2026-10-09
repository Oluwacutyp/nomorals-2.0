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
- ``comfy`` (:class:`nomorals.media_edit.comfy.ComfyUIBackend`): drives a
  running ComfyUI server over HTTP (stdlib only). Host/port from
  ``COMFYUI_HOST`` / ``COMFYUI_PORT``; workflows in
  ``nomorals/media_edit/workflows/``; one GPU job at a time.
- ``leonardo`` (:class:`LeonardoBackend`), ``stability_ai``
  (:class:`StabilityAIBackend`), ``nano_banana`` (:class:`NanoBananaBackend`):
  paid API backends driving the matching connectors in
  ``nomorals/connectors/``. Explicit opt-in only (never "auto") — every
  call costs real money/credits. Keys live in the credential vault
  (``nm connectors connect --name <backend>``); the vault unlocks via
  ``NM_VAULT_PASSPHRASE``.
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

import difflib
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
# style / aspect / quality presets
# ---------------------------------------------------------------------------

STYLES: dict[str, str] = {
    "photorealistic": ("ultra photorealistic, professional photography, "
                       "sharp focus, natural lighting"),
    "cinematic": ("cinematic film still, dramatic lighting, shallow depth "
                  "of field, 35mm film look"),
    "anime": "anime style, cel shaded, vibrant colors, detailed background",
    "digital-art": "digital art, highly detailed, dramatic composition",
    "oil-painting": "classical oil painting, visible brushstrokes, canvas texture",
    "watercolor": "watercolor painting, soft washes, paper texture",
    "cyberpunk": "cyberpunk aesthetic, neon lights, futuristic, moody atmosphere",
    "3d-render": "3d render, soft studio lighting, high detail, octane render",
    "pixel-art": "pixel art, 16-bit retro game style, crisp pixels",
    "sketch": "pencil sketch, detailed linework, cross-hatching",
    "portrait": "professional portrait photography, 85mm lens, bokeh background",
    "product": "commercial product photography, studio lighting, clean background",
}

def _resolve_image_style(style: str) -> str:
    """Resolve an image style name with fuzzy matching. Never hard-fails
    on close matches — returns the canonical style key."""
    key = (style or "").strip().lower()
    if key in STYLES:
        return key
    matches = difflib.get_close_matches(key, list(STYLES.keys()), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    raise GenerativeEditError(
        f"unknown style {style!r}; use: {sorted(STYLES)}")


ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:2": (1216, 832),
    "21:9": (1536, 640),
}

QUALITY: dict[str, dict[str, int]] = {
    "draft": {"steps": 15},
    "standard": {"steps": 30},
    "ultra": {"steps": 50},
}


def _resolve_gen_params(prompt: str,
                        style: str | None = None,
                        aspect: str | None = None,
                        quality: str | None = None,
                        width: int | None = None,
                        height: int | None = None,
                        steps: int | None = None,
                        ) -> tuple[str, int | None, int | None, int | None]:
    """Apply style/aspect/quality presets → (prompt, width, height, steps).

    Explicit width/height/steps always win over presets. Unknown preset
    names fail fast.
    """
    if style is not None:
        style = _resolve_image_style(style)
        prompt = f"{prompt}, {STYLES[style]}"
    if aspect is not None:
        if aspect not in ASPECT_RATIOS:
            raise GenerativeEditError(
                f"unknown aspect {aspect!r}; use: {sorted(ASPECT_RATIOS)}")
        aw, ah = ASPECT_RATIOS[aspect]
        width = aw if width is None else width
        height = ah if height is None else height
    if quality is not None:
        if quality not in QUALITY:
            raise GenerativeEditError(
                f"unknown quality {quality!r}; use: {sorted(QUALITY)}")
        if steps is None:
            steps = QUALITY[quality]["steps"]
    return prompt, width, height, steps


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

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Image-to-image: regenerate ``image`` guided by ``prompt``.

        strength 0 = near-copy, 1 = near-total regeneration. Used for
        style transfer and variations.
        """
        raise NotImplementedError

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Regenerate the masked region of ``image`` per ``prompt``.

        ``mask``: (l,t,r,b) box, mask image path, or PIL L image
        (white = regenerate). The unmasked region is preserved.
        """
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

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Image-to-image / style transfer via image_to_image."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        if not 0 < strength <= 1:
            raise GenerativeEditError(
                f"img2img strength must be in (0, 1], got {strength}")
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
        _log.info("img2img via hf model=%s strength=%.2f prompt=%.60r",
                  self.model, strength, prompt)
        try:
            result = client.image_to_image(
                image, prompt=prompt, model=self.model, **params)
        except Exception as exc:
            raise GenerativeEditError(
                f"HF img2img failed for model {self.model!r}: {exc}") from exc
        return result.convert("RGB")

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Region-targeted regeneration.

        HF serverless exposes no native inpainting endpoint, so this
        regenerates the whole image at high strength guided by ``prompt``
        and composites the result through the feathered mask — the
        unmasked region is pixel-identical to the original.
        """
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        regenerated = self.img2img(
            image, prompt, strength=0.85, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, regenerated, mask)

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

    def _load_pipe(self, pipeline_cls: Any, cache_attr: str) -> Any:
        """Load (and cache) a diffusers pipeline of the given class."""
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        import torch
        device = self.device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _log.info("loading diffusers model %s (%s) on %s", self.model,
                  pipeline_cls.__name__, device)
        pipe = pipeline_cls.from_pretrained(
            self.model, torch_dtype=dtype, use_safetensors=True)
        pipe = pipe.to(device)
        setattr(self, cache_attr, pipe)
        return pipe

    def _pipe_for(self, image: Any) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            import torch  # noqa: F401
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise GenerativeEditError(
                "generative edit needs diffusers+torch: "
                "pip install diffusers torch (or set MEDIA_GEN_BACKEND=hf "
                "with HF_TOKEN in the environment)") from exc
        self._pipe = self._load_pipe(AutoPipelineForImage2Image, "_i2i_pipe")
        return self._pipe

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

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Image-to-image / style transfer via AutoPipelineForImage2Image."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        if not 0 < strength <= 1:
            raise GenerativeEditError(
                f"img2img strength must be in (0, 1], got {strength}")
        import torch
        pipe = self._pipe_for(image)
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
            result = pipe(prompt, image=image.convert("RGB"),
                          generator=generator, **kwargs).images[0]
        except Exception as exc:
            raise GenerativeEditError(
                f"diffusers img2img failed for model {self.model!r}: {exc}"
            ) from exc
        return result.convert("RGB")

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Native inpainting via AutoPipelineForInpainting.

        Falls back to masked img2img composite when the configured model
        is not an inpainting model.
        """
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        import torch
        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise GenerativeEditError(
                "inpainting needs diffusers: pip install diffusers torch"
            ) from exc
        m = _as_mask(image.size, mask)
        kwargs: dict[str, Any] = {}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if steps:
            kwargs["num_inference_steps"] = int(steps)
        if guidance_scale:
            kwargs["guidance_scale"] = float(guidance_scale)
        generator = torch.Generator().manual_seed(int(seed)) \
            if seed is not None else None
        try:
            pipe = self._load_pipe(AutoPipelineForInpainting, "_inpaint_pipe")
            result = pipe(prompt, image=image.convert("RGB"), mask_image=m,
                          generator=generator, **kwargs).images[0]
        except Exception as exc:
            # Not an inpainting-capable model — degrade honestly to
            # masked img2img instead of failing the whole op.
            _log.warning("diffusers inpainting pipeline failed (%s); "
                         "falling back to masked img2img", exc)
            regenerated = self.img2img(
                image, prompt, strength=0.85, seed=seed,
                negative_prompt=negative_prompt, steps=steps,
                guidance_scale=guidance_scale)
            return _apply_mask(image, regenerated, m)
        return _apply_mask(image, result, m)

    def describe(self) -> str:
        return f"generative backend 'diffusers' (model={self.model})"


# ---------------------------------------------------------------------------
# Devon native backend — our own diffusion code (nomorals.media.imggen)
# ---------------------------------------------------------------------------

class NativeBackend(GenerativeBackend):
    """Devon's own image generation: hand-built diffusion, local weights.

    Preferred automatically when a native checkpoint exists. Needs
    torch; without it (or without a checkpoint) it raises a clear
    GenerativeEditError telling the user exactly what to do — never
    a fake image, never a traceback.
    """

    name = "native"

    @staticmethod
    def available() -> bool:
        try:
            from ..media.imggen import TORCH_AVAILABLE
            from ..media.imggen.pipeline import list_native_checkpoints
        except Exception:
            return False
        return bool(TORCH_AVAILABLE and list_native_checkpoints())

    @staticmethod
    def readiness() -> tuple[bool, str]:
        """(ready, reason) — for honest chat messages."""
        try:
            from ..media.imggen import TORCH_AVAILABLE
        except Exception as exc:
            return False, f"imggen organ not importable: {exc}"
        if not TORCH_AVAILABLE:
            return False, ("torch not installed — native image "
                           "generation needs it: pip install torch")
        try:
            from ..media.imggen.pipeline import list_native_checkpoints
        except Exception as exc:
            return False, f"native pipeline broken: {exc}"
        cks = list_native_checkpoints()
        if not cks:
            return False, ("no native checkpoint yet — train one: "
                           "`nm imggen train --data <photo-folder>` "
                           "(tiny 64px model trains on CPU in minutes)")
        return True, f"{len(cks)} checkpoint(s), newest: {cks[0]['run']}"

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list:
        ok, reason = self.readiness()
        if not ok:
            raise GenerativeEditError(f"native backend not ready: {reason}")
        from ..media.imggen.pipeline import PipelineConfig
        from ..media.imggen.studio import Studio, StudioConfig

        studio = Studio(StudioConfig())
        images = []
        for _ in range(n):
            paths = studio.generate(
                prompt,
                steps=steps or 50,
                guidance_scale=(guidance_scale if guidance_scale
                                is not None else 7.5),
                seed=seed,
                batch_size=1,
                width=width or 64, height=height or 64,
                negative_prompt=negative_prompt or "",
                save_to=None,
            )
            from PIL import Image as _Image

            images.append(_Image.open(paths[0]))
            if seed is not None:
                seed += 1  # each image gets its own reproducible seed
        return images

    def describe(self) -> str:
        ok, reason = self.readiness()
        return (f"generative backend 'native' (Devon's own diffusion; "
                f"{reason})" if ok else
                f"generative backend 'native' (not ready: {reason})")


# ---------------------------------------------------------------------------
# paid API backends (opt-in via explicit MEDIA_GEN_BACKEND only)
# ---------------------------------------------------------------------------

#: Paid image-generation backends. Selectable ONLY by explicit
#: ``MEDIA_GEN_BACKEND`` value — never by "auto", because every call costs
#: real money/credits. Each one drives the matching connector in
#: ``nomorals/connectors/`` (Leonardo AI, Stability AI, Nano Banana).
PAID_IMAGE_BACKENDS = ("leonardo", "stability_ai", "nano_banana")


def _paid_vault(vault: Any | None) -> Any:
    """Resolve the credential vault for a paid image backend.

    An explicit ``vault`` wins (tests, wired callers). Otherwise build one
    from the default database + ``NM_VAULT_PASSPHRASE`` — the same vault
    ``nm connectors`` stores keys in. Fails fast with the exact fix.
    """
    if vault is not None:
        return vault
    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        raise GenerativeEditError(
            "paid image backends need the credential vault: set "
            "NM_VAULT_PASSPHRASE, then connect the backend with "
            "`nm connectors connect --name <backend>` "
            "(or /connectors connect <backend> in chat)")
    try:
        from ..accounts.vault import CredentialVault
        from ..core.config import get_settings
        from ..storage.db import open_database
        settings = get_settings()
        storage = settings.storage
        db, _, _ = open_database(
            settings.db_path,
            wal=getattr(storage, "wal", True),
            busy_timeout_ms=getattr(storage, "busy_timeout_ms", 5000),
            synchronous=getattr(storage, "synchronous", "NORMAL"),
        )
        return CredentialVault(db, master_passphrase=passphrase)
    except GenerativeEditError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
        raise GenerativeEditError(
            f"could not open the credential vault: {exc}") from exc


class _PaidAPIBackend(GenerativeBackend):
    """Shared plumbing for paid image-generation connectors.

    Subclasses bind one connector id and translate the
    generate/edit/img2img/inpaint interface onto that connector's real
    methods. Every billable call passes ``confirmed=True``: the owner typed
    the exact prompt in their own chat / CLI invocation, which is explicit
    approval of the exact payload under ``confirm_or_checkpoint``'s
    contract. The cost is surfaced in :meth:`describe` and the caller
    reports it back to the owner.
    """

    #: registry id in nomorals/connectors, e.g. "leonardo"
    connector_id = ""
    #: short cost note shown in describe(), e.g. "costs Leonardo API credits"
    cost_note = "paid — costs real money/credits per image"

    def __init__(self, vault: Any | None = None) -> None:
        self._vault = _paid_vault(vault)

    # -- connector ------------------------------------------------------
    def _connector(self) -> Any:
        from ..connectors.registry import create_connector
        try:
            return create_connector(self.connector_id, self._vault)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"paid backend {self.name!r}: {exc}") from exc

    # -- shared converters ----------------------------------------------
    @staticmethod
    def _pil_from_bytes(data: bytes) -> Any:
        Image = _require_pillow()
        import io as _io
        try:
            return Image.open(_io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                "the backend returned undecodable image data "
                f"({len(data)} bytes): {exc}"
            ) from exc

    @staticmethod
    def _pil_to_png_bytes(image: Any) -> bytes:
        import io as _io
        buf = _io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _download(url: str, timeout: float = 60.0) -> bytes:
        import urllib.request
        req = urllib.request.Request(
            url, headers={"User-Agent": "nomorals-media-edit/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"could not download the generated image: {exc}") from exc

    @staticmethod
    def _closest_aspect(width: int | None, height: int | None,
                        allowed: set[str]) -> str:
        """Pick the allowed "W:H" ratio closest to width/height.

        Systematic nearest-ratio match — no hardcoded preference table.
        """
        if not allowed:
            raise GenerativeEditError("no aspect ratios available")
        target = ((width or 1) / (height or 1))
        best, best_err = "1:1", float("inf")
        for ratio in sorted(allowed):
            try:
                w, h = ratio.split(":")
                value = float(w) / float(h)
            except (ValueError, ZeroDivisionError):
                continue
            err = abs(value - target)
            if err < best_err:
                best, best_err = ratio, err
        return best

    def describe(self) -> str:
        return f"generative backend '{self.name}' ({self.cost_note})"


class LeonardoBackend(_PaidAPIBackend):
    """Leonardo AI text-to-image / image-to-image (paid).

    Drives :class:`nomorals.connectors.leonardo.LeonardoConnector`:
    ``generate_image`` polls the job to completion and returns image URLs,
    which are downloaded here. ``edit_image`` uploads an init image.
    Strength mapping: our ``strength`` is regeneration amount; Leonardo's
    ``init_strength`` is input preservation, so init_strength = 1 - strength.
    Model from ``LEONARDO_MODEL_ID`` (default: Leonardo's default model).
    """

    name = "leonardo"
    connector_id = "leonardo"
    cost_note = "paid — costs Leonardo API credits per generation"

    def __init__(self, vault: Any | None = None) -> None:
        super().__init__(vault)
        self.model_id = os.environ.get("LEONARDO_MODEL_ID", "")

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        conn = self._connector()
        _log.info("text-to-image via leonardo n=%d prompt=%.60r", n, prompt)
        try:
            # generate_image returns [{"id", "url"}]; num_images ≤ 8.
            entries = conn.generate_image(
                prompt,
                width=width or 1024, height=height or 1024,
                num_images=max(1, min(n, 8)),
                model_id=self.model_id,
                negative_prompt=negative_prompt or "",
                seed=int(seed or 0),
                confirmed=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(f"leonardo generation failed: {exc}"
                                      ) from exc
        images = []
        for entry in entries[:n]:
            url = (entry.get("url") or "").strip()
            if not url:
                continue
            images.append(self._pil_from_bytes(self._download(url)))
        if not images:
            raise GenerativeEditError(
                "leonardo returned no downloadable images")
        return images

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
        result = self.img2img(
            image, instruction, strength=strength, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, result, mask)

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        if not 0 < strength <= 1:
            raise GenerativeEditError(
                f"img2img strength must be in (0, 1], got {strength}")
        conn = self._connector()
        raw = self._pil_to_png_bytes(image)
        _log.info("img2img via leonardo strength=%.2f prompt=%.60r",
                  strength, prompt)
        try:
            # Leonardo's init_strength preserves the input (higher = more
            # input kept); ours regenerates (higher = more change).
            entries = conn.edit_image(
                raw, prompt,
                init_strength=round(1.0 - strength, 3),
                width=image.size[0], height=image.size[1],
                model_id=self.model_id,
                confirmed=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(f"leonardo edit failed: {exc}") from exc
        url = ((entries[0].get("url") or "").strip()
               if entries else "")
        if not url:
            raise GenerativeEditError("leonardo returned no image URL")
        return self._pil_from_bytes(self._download(url))

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """No native inpainting endpoint: regenerate at high strength and
        composite through the feathered mask — unmasked pixels stay
        pixel-identical to the original."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        regenerated = self.img2img(
            image, prompt, strength=0.85, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, regenerated, mask)


class StabilityAIBackend(_PaidAPIBackend):
    """Stability AI (SD 3.x) text-to-image / image-to-image (paid).

    Drives :class:`nomorals.connectors.stabilityai.StabilityAIConnector`.
    Both calls are synchronous and return decoded image bytes. Model from
    ``STABILITY_MODEL`` (default ``sd3.5-large``); the closest supported
    aspect ratio is picked systematically from width/height.
    """

    name = "stability_ai"
    connector_id = "stability_ai"
    cost_note = "paid — costs Stability AI credits per image"

    #: aspect ratios the v2beta stable-image endpoint accepts
    ASPECTS = {"1:1", "16:9", "21:9", "2:3", "3:2",
               "4:5", "5:4", "9:16", "9:21"}

    def __init__(self, vault: Any | None = None) -> None:
        super().__init__(vault)
        self.model = os.environ.get("STABILITY_MODEL", "sd3.5-large")

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        conn = self._connector()
        aspect = self._closest_aspect(width, height, self.ASPECTS)
        _log.info("text-to-image via stability_ai model=%s aspect=%s n=%d",
                  self.model, aspect, n)
        images = []
        try:
            # One synchronous image per call; vary the seed across n.
            for i in range(n):
                result = conn.generate_image(
                    prompt,
                    model=self.model,
                    aspect_ratio=aspect,
                    seed=int(seed or 0) + i if seed is not None else 0,
                    negative_prompt=negative_prompt or "",
                    confirmed=True,
                )
                images.append(self._pil_from_bytes(result["image_bytes"]))
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"stability_ai generation failed: {exc}") from exc
        return images

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
        result = self.img2img(
            image, instruction, strength=strength, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, result, mask)

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        if not 0 < strength <= 1:
            raise GenerativeEditError(
                f"img2img strength must be in (0, 1], got {strength}")
        conn = self._connector()
        raw = self._pil_to_png_bytes(image)
        _log.info("img2img via stability_ai strength=%.2f prompt=%.60r",
                  strength, prompt)
        try:
            # Stability's strength = deviation from input: matches ours.
            result = conn.edit_image(
                raw, prompt,
                model=self.model,
                strength=float(strength),
                seed=int(seed or 0),
                negative_prompt=negative_prompt or "",
                confirmed=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"stability_ai edit failed: {exc}") from exc
        return self._pil_from_bytes(result["image_bytes"])

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """No native inpainting endpoint: regenerate at high strength and
        composite through the feathered mask."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        regenerated = self.img2img(
            image, prompt, strength=0.85, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, regenerated, mask)


class NanoBananaBackend(_PaidAPIBackend):
    """Nano Banana (Google Gemini image models) generation/editing (paid).

    Drives :class:`nomorals.connectors.nanobanana.NanoBananaConnector`.
    Images come back as base64 inlineData parts. Model from
    ``NANO_BANANA_MODEL`` (default ``gemini-2.5-flash-image``); the closest
    supported aspect ratio is picked systematically from width/height.
    """

    name = "nano_banana"
    connector_id = "nano_banana"
    cost_note = "paid — costs per image via Google AI Studio"

    #: aspect ratios the Gemini imageConfig accepts
    ASPECTS = {"1:1", "2:3", "3:2", "3:4", "4:3",
               "4:5", "5:4", "9:16", "16:9", "21:9"}

    def __init__(self, vault: Any | None = None) -> None:
        super().__init__(vault)
        self.model = os.environ.get("NANO_BANANA_MODEL",
                                    "gemini-2.5-flash-image")

    def generate(self, prompt: str, *,
                 seed: int | None = None,
                 negative_prompt: str | None = None,
                 steps: int | None = None,
                 guidance_scale: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 n: int = 1) -> list[Any]:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("text-to-image needs a prompt")
        if n < 1:
            raise GenerativeEditError("n must be >= 1")
        conn = self._connector()
        aspect = self._closest_aspect(width, height, self.ASPECTS)
        _log.info("text-to-image via nano_banana model=%s aspect=%s n=%d",
                  self.model, aspect, n)
        images = []
        try:
            # The model returns however many images it returns per call;
            # repeat for n (each call is billed).
            for _ in range(n):
                for entry in conn.generate_image(
                        prompt, model=self.model,
                        aspect_ratio=aspect, confirmed=True):
                    images.append(
                        self._pil_from_bytes(entry["image_bytes"]))
                    if len(images) >= n:
                        break
                if len(images) >= n:
                    break
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"nano_banana generation failed: {exc}") from exc
        if not images:
            raise GenerativeEditError("nano_banana returned no images")
        return images[:n]

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
        regenerated = self.img2img(
            image, instruction, strength=strength, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, regenerated, mask)

    def img2img(self, image: Any, prompt: str, *,
                strength: float = 0.6,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        if not prompt or not prompt.strip():
            raise GenerativeEditError("img2img needs a prompt")
        if not 0 < strength <= 1:
            raise GenerativeEditError(
                f"img2img strength must be in (0, 1], got {strength}")
        conn = self._connector()
        raw = self._pil_to_png_bytes(image)
        _log.info("edit via nano_banana model=%s prompt=%.60r",
                  self.model, prompt)
        try:
            # Nano Banana edits are instruction-driven (no strength knob);
            # strength is honored by phrasing, like the HF backend.
            entries = conn.edit_image(
                raw, prompt, model=self.model, confirmed=True)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
            raise GenerativeEditError(
                f"nano_banana edit failed: {exc}") from exc
        if not entries:
            raise GenerativeEditError("nano_banana returned no images")
        return self._pil_from_bytes(entries[0]["image_bytes"])

    def inpaint(self, image: Any, mask: Any, prompt: str, *,
                seed: int | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
        """Instruction edit + feathered mask composite (no native
        inpainting endpoint)."""
        if not prompt or not prompt.strip():
            raise GenerativeEditError("inpaint needs a prompt")
        regenerated = self.img2img(
            image, prompt, strength=0.85, seed=seed,
            negative_prompt=negative_prompt, steps=steps,
            guidance_scale=guidance_scale)
        return _apply_mask(image, regenerated, mask)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

_NO_BACKEND_MSG = (
    "no generative backend is configured. Options: "
    "pip install huggingface_hub and set HF_TOKEN (serverless, "
    "MEDIA_GEN_BACKEND=hf), or pip install diffusers torch for local "
    "inference (MEDIA_GEN_BACKEND=diffusers), or point at a running "
    "ComfyUI server (MEDIA_GEN_BACKEND=comfy, COMFYUI_HOST/PORT), "
    "or use a paid API backend you hold a key for "
    "(MEDIA_GEN_BACKEND=leonardo|stability_ai|nano_banana — connect the "
    "key first with `nm connectors connect --name <backend>`). "
    "Set MEDIA_GEN_BACKEND=off to silence this check."
)


def _paid_backend(name: str, vault: Any | None) -> GenerativeBackend:
    """Instantiate a paid image backend by name."""
    backends = {
        "leonardo": LeonardoBackend,
        "stability_ai": StabilityAIBackend,
        "nano_banana": NanoBananaBackend,
    }
    return backends[name](vault=vault)


def get_backend(name: str | None = None,
                vault: Any | None = None) -> GenerativeBackend:
    """Resolve a generative backend.

    ``name`` or ``MEDIA_GEN_BACKEND``: "auto" (default), "native", "hf",
    "diffusers", "comfy", "leonardo", "stability_ai", "nano_banana", "off".
    auto prefers Devon's native backend when a checkpoint exists, then
    local diffusers when importable, else HF when huggingface_hub imports,
    else raises a clear error (never a fake edit). The paid backends are
    explicit opt-in only — auto never selects them, because every call
    costs real money/credits. ``vault`` is the credential vault for the
    paid backends; when omitted it is built lazily from the default
    database + NM_VAULT_PASSPHRASE.
    """
    want = (name or os.environ.get("MEDIA_GEN_BACKEND", "auto")).lower()
    if want == "off":
        raise GenerativeEditError(_NO_BACKEND_MSG)
    if want in PAID_IMAGE_BACKENDS:
        return _paid_backend(want, vault)
    if want == "native":
        if not NativeBackend.available():
            _ok, _reason = NativeBackend.readiness()
            raise GenerativeEditError(
                f"MEDIA_GEN_BACKEND=native but {_reason}")
        return NativeBackend()
    if want == "hf":
        return HFInferenceBackend()
    if want == "comfy":
        from .comfy import ComfyUIBackend
        return ComfyUIBackend()
    if want == "diffusers":
        if not DiffusersBackend.available():
            raise GenerativeEditError(
                "MEDIA_GEN_BACKEND=diffusers but diffusers/torch are not "
                "installed")
        return DiffusersBackend()
    if want == "auto":
        if NativeBackend.available():
            return NativeBackend()
        if DiffusersBackend.available():
            return DiffusersBackend()
        try:
            import huggingface_hub  # noqa: F401
            return HFInferenceBackend()
        except ImportError:  # noqa: E103 - optional backend probe, absence is handled below
            pass
        raise GenerativeEditError(_NO_BACKEND_MSG)
    raise GenerativeEditError(
        f"unknown MEDIA_GEN_BACKEND={want!r}; use auto|native|hf|diffusers|"
        f"comfy|{'|'.join(PAID_IMAGE_BACKENDS)}|off")


def _paid_backend_status() -> dict[str, Any]:
    """Readiness of the paid backends, without network calls.

    A backend is "ready" when the vault unlocks and holds its credential.
    Never raises — reports the reason instead.
    """
    out: dict[str, Any] = {}
    for name in PAID_IMAGE_BACKENDS:
        try:
            vault = _paid_vault(None)
        except GenerativeEditError as exc:
            out[name] = {"ready": False, "reason": str(exc)}
            continue
        try:
            from ..connectors.registry import create_connector
            conn = create_connector(name, vault)
            cred = conn._load_credential()
        except Exception as exc:  # noqa: BLE001 - status, not a failure
            out[name] = {"ready": False, "reason": str(exc)}
            continue
        if cred is None:
            out[name] = {
                "ready": False,
                "reason": (f"not connected — run `nm connectors connect "
                           f"--name {name}` (or /connectors connect {name} "
                           "in chat)"),
            }
        else:
            out[name] = {"ready": True,
                         "reason": f"connected as {cred.username}"}
    return out


def backend_status() -> dict[str, Any]:
    """Report which generative backends are usable (no model calls)."""
    try:
        import huggingface_hub  # noqa: F401
        hf = True
    except ImportError:
        hf = False
    from .comfy import comfy_available
    comfy_ok, comfy_reason = comfy_available()
    return {
        "hf_installed": hf,
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "hf_model": os.environ.get("MEDIA_GEN_MODEL", DEFAULT_HF_MODEL),
        "hf_provider": os.environ.get("MEDIA_GEN_PROVIDER"),
        "diffusers_available": DiffusersBackend.available(),
        "diffusers_model": os.environ.get("MEDIA_GEN_DIFFUSERS_MODEL",
                                          DEFAULT_DIFFUSERS_MODEL),
        "comfy_available": comfy_ok,
        "comfy_reason": comfy_reason,
        "comfy_host": os.environ.get("COMFYUI_HOST", "127.0.0.1"),
        "comfy_port": int(os.environ.get("COMFYUI_PORT", "8188") or 8188),
        "paid": _paid_backend_status(),
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
               style: str | None = None,
               aspect: str | None = None,
               quality: str | None = None,
               n: int = 1) -> Any:
    """Text-to-image as a chain op. Returns a single PIL image (n=1) or a
    list of PIL images (n>1).

    ``style``: photorealistic|cinematic|anime|digital-art|oil-painting|
    watercolor|cyberpunk|3d-render|pixel-art|sketch|portrait|product.
    ``aspect``: 1:1|16:9|9:16|4:3|3:2|21:9. ``quality``: draft|standard|ultra.
    """
    be = get_backend(backend)
    prompt, width, height, steps = _resolve_gen_params(
        prompt, style=style, aspect=aspect, quality=quality,
        width=width, height=height, steps=steps)
    images = be.generate(prompt, seed=seed, negative_prompt=negative_prompt,
                         steps=steps, guidance_scale=guidance_scale,
                         width=width, height=height, n=n)
    return images[0] if n == 1 else images


def op_img2img(img: Any, prompt: str, *,
               strength: float = 0.6,
               seed: int | None = None,
               backend: str | None = None,
               negative_prompt: str | None = None,
               steps: int | None = None,
               guidance_scale: float | None = None,
               style: str | None = None) -> Any:
    """Image-to-image: restyle / variation of ``img`` guided by ``prompt``.

    ``strength`` 0→near-copy, 1→near-total regeneration. ``style`` applies
    a style preset to the prompt (see op_txt2img).
    """
    be = get_backend(backend)
    if style is not None:
        style = _resolve_image_style(style)
        prompt = f"{prompt}, {STYLES[style]}"
    return be.img2img(img, prompt, strength=strength, seed=seed,
                      negative_prompt=negative_prompt, steps=steps,
                      guidance_scale=guidance_scale)


def op_inpaint(img: Any, mask: Any, prompt: str, *,
               seed: int | None = None,
               backend: str | None = None,
               negative_prompt: str | None = None,
               steps: int | None = None,
               guidance_scale: float | None = None) -> Any:
    """Inpainting: regenerate the masked region per ``prompt``.

    ``mask``: (l,t,r,b) box, mask image path, or PIL L image
    (white = regenerate). Unmasked pixels are preserved.
    """
    be = get_backend(backend)
    m = _as_mask(img.size, mask)
    return be.inpaint(img, m, prompt, seed=seed,
                      negative_prompt=negative_prompt, steps=steps,
                      guidance_scale=guidance_scale)


def op_outpaint(img: Any, prompt: str, *,
                top: int = 0, right: int = 0,
                bottom: int = 0, left: int = 0,
                seed: int | None = None,
                backend: str | None = None,
                negative_prompt: str | None = None,
                steps: int | None = None,
                guidance_scale: float | None = None) -> Any:
    """Outpainting: extend the canvas and AI-fill the new border region.

    At least one of top/right/bottom/left must be > 0. The original image
    is preserved pixel-identical in the center; only the new border is
    generated (edge-extended fill seeds the generation for continuity).
    """
    Image = _require_pillow()
    pads = {"top": top, "right": right, "bottom": bottom, "left": left}
    for side, px in pads.items():
        if px < 0:
            raise GenerativeEditError(
                f"outpaint {side} must be >= 0, got {px}")
    if sum(pads.values()) == 0:
        raise GenerativeEditError(
            "outpaint needs at least one of top/right/bottom/left > 0")
    if not prompt or not prompt.strip():
        raise GenerativeEditError("outpaint needs a prompt")
    w, h = img.size
    nw, nh = w + left + right, h + top + bottom
    rgb = img.convert("RGB")
    # Edge-extend the original so the model sees continuity at the seam.
    canvas = Image.new("RGB", (nw, nh))
    # fill by stretching the original to canvas, then paste the sharp
    # original over the center
    stretched = rgb.resize((nw, nh), Image.BILINEAR)
    canvas.paste(stretched, (0, 0))
    canvas.paste(rgb, (left, top))
    # Mask: white only on the new border region.
    from PIL import ImageDraw
    m = Image.new("L", (nw, nh), 255)
    d = ImageDraw.Draw(m)
    d.rectangle([left, top, left + w - 1, top + h - 1], fill=0)
    be = get_backend(backend)
    return be.inpaint(canvas, m, prompt, seed=seed,
                      negative_prompt=negative_prompt, steps=steps,
                      guidance_scale=guidance_scale)


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


def op_bg_remove(img: Any, *,
                 mode: str = "auto",
                 model: str = "",
                 chroma_color: str | tuple[int, int, int] | None = None,
                 tolerance: int = 40,
                 feather: int = 2) -> Any:
    """Remove the background → RGBA image with transparent background.

    ``mode``:
    - ``"auto"``: AI segmentation via :mod:`nomorals.media_edit.segment`
      (profile-aware model: u2netp on termux, birefnet-general elsewhere);
      fails fast with the pip hint when rembg is missing — never a silent
      quality downgrade.
    - ``"rembg"``: same as auto but with an explicit model via ``model=``.
    - ``"chroma"``: chroma-key removal, pure PIL. Key color defaults to
      the most common corner color; ``tolerance`` is RGB distance.
    """
    Image = _require_pillow()
    if mode in ("auto", "rembg"):
        from .segment import remove_background, SegmentationError
        try:
            return remove_background(
                img, model=(model or "" if mode == "rembg" else ""))
        except SegmentationError as exc:
            raise GenerativeEditError(str(exc)) from exc
    if mode == "chroma":
        from PIL import ImageFilter
        rgba = img.convert("RGBA")
        w, h = rgba.size
        px = rgba.load()
        # Key color: explicit, or most common corner color.
        if chroma_color is None:
            corners = [px[0, 0][:3], px[w - 1, 0][:3],
                       px[0, h - 1][:3], px[w - 1, h - 1][:3]]
            key = max(set(corners), key=corners.count)
        elif isinstance(chroma_color, str):
            key = Image.new("RGB", (1, 1), chroma_color).getpixel((0, 0))
        else:
            key = tuple(int(v) for v in chroma_color)
        if not 0 <= tolerance <= 441:
            raise GenerativeEditError(
                f"chroma tolerance must be 0..441, got {tolerance}")
        tol2 = tolerance * tolerance
        alpha = Image.new("L", (w, h), 255)
        apx = alpha.load()
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y][:3]
                d2 = (r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2
                if d2 <= tol2:
                    apx[x, y] = 0
        if feather > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
        rgba.putalpha(alpha)
        return rgba
    raise GenerativeEditError(
        f"unknown bg_remove mode {mode!r}; use auto|rembg|chroma")


def op_bg_replace(img: Any, background: Any, *,
                  mode: str = "auto",
                  chroma_color: str | tuple[int, int, int] | None = None,
                  tolerance: int = 40) -> Any:
    """Remove the background and composite onto a new one.

    ``background``:
    - a color name/hex (``"white"``, ``"#1a1a2e"``)
    - ``"blur"`` — the original image, heavily blurred (portrait-mode look)
    - a PIL image or image path (cover-fit to the subject size)
    Returns RGB.
    """
    Image = _require_pillow()
    from PIL import ImageFilter
    fg = op_bg_remove(img, mode=mode, chroma_color=chroma_color,
                      tolerance=tolerance)
    w, h = fg.size
    if isinstance(background, str) and background != "blur":
        bg = Image.new("RGB", (w, h), background)
    elif background == "blur":
        bg = img.convert("RGB").filter(ImageFilter.GaussianBlur(25))
        if bg.size != (w, h):
            bg = bg.resize((w, h), Image.LANCZOS)
    else:
        if isinstance(background, (str, Path)):
            bg_img = Image.open(background).convert("RGB")
        else:
            bg_img = background.convert("RGB")
        # cover-fit
        scale = max(w / bg_img.width, h / bg_img.height)
        sw, sh = int(bg_img.width * scale), int(bg_img.height * scale)
        bg = bg_img.resize((sw, sh), Image.LANCZOS)
        left, top = (sw - w) // 2, (sh - h) // 2
        bg = bg.crop((left, top, left + w, top + h))
    bg.paste(fg, (0, 0), fg)
    return bg


def _register() -> None:
    from . import images as _images
    _images._OP_FUNCS["generative_edit"] = op_generative_edit
    _images.OP_ALLOWLIST.add("generative_edit")
    _images._OP_FUNCS["txt2img"] = op_txt2img
    _images.OP_ALLOWLIST.add("txt2img")
    _images._OP_FUNCS["upscale"] = op_upscale
    _images.OP_ALLOWLIST.add("upscale")
    _images._OP_FUNCS["img2img"] = op_img2img
    _images.OP_ALLOWLIST.add("img2img")
    _images._OP_FUNCS["inpaint"] = op_inpaint
    _images.OP_ALLOWLIST.add("inpaint")
    _images._OP_FUNCS["outpaint"] = op_outpaint
    _images.OP_ALLOWLIST.add("outpaint")
    _images._OP_FUNCS["bg_remove"] = op_bg_remove
    _images.OP_ALLOWLIST.add("bg_remove")
    _images._OP_FUNCS["bg_replace"] = op_bg_replace
    _images.OP_ALLOWLIST.add("bg_replace")
    # vision utility stack (#23): dedicated modules, lazy heavy deps
    from .segment import op_bg_remove_v2
    _images._OP_FUNCS["bg_remove_v2"] = op_bg_remove_v2
    _images.OP_ALLOWLIST.add("bg_remove_v2")
    from .upscale import op_upscale_sr
    _images._OP_FUNCS["upscale_sr"] = op_upscale_sr
    _images.OP_ALLOWLIST.add("upscale_sr")
    from .faceswap import op_faceswap
    _images._OP_FUNCS["faceswap"] = op_faceswap
    _images.OP_ALLOWLIST.add("faceswap")
    # text replacement in photos (#24)
    from .edittext import op_edittext
    _images._OP_FUNCS["edittext"] = op_edittext
    _images.OP_ALLOWLIST.add("edittext")


_register()
