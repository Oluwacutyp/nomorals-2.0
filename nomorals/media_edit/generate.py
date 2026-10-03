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
                 chroma_color: str | tuple[int, int, int] | None = None,
                 tolerance: int = 40,
                 feather: int = 2) -> Any:
    """Remove the background → RGBA image with transparent background.

    ``mode``:
    - ``"auto"``: use rembg (AI segmentation) if installed, else fail fast
      with the pip hint — never a silent quality downgrade.
    - ``"rembg"``: require rembg (``pip install rembg``).
    - ``"chroma"``: chroma-key removal, pure PIL. Key color defaults to
      the most common corner color; ``tolerance`` is RGB distance.
    """
    Image = _require_pillow()
    if mode == "auto":
        try:
            import rembg  # noqa: F401
            mode = "rembg"
        except ImportError:
            raise GenerativeEditError(
                "background removal needs rembg: pip install rembg "
                "(or use mode='chroma' for chroma-key removal without "
                "extra dependencies)") from None
    if mode == "rembg":
        try:
            from rembg import remove
        except ImportError as exc:
            raise GenerativeEditError(
                "background removal needs rembg: pip install rembg") from exc
        out = remove(img.convert("RGB"))
        return out.convert("RGBA")
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


_register()
