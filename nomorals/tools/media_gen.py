"""Builtin tools for native image and video GENERATION.

Wires ``nomorals.media.imggen`` (Devon's own diffusion pipeline) and
``nomorals.media.videogen`` (neural LTX/Wan + motion-studio fallback)
into the spine so the brain reaches them from plain language:
"generate an image of...", "animate this photo", "make a video of...".

Native-first: the local pipelines run when the hardware allows;
honest errors (never fake images) when it doesn't. APIs stay as
fallback, never the priority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

# Generation outputs never overwrite inputs: everything lands here.
GEN_DIR = "generated"


def _out_dir(context: Any, kind: str) -> Path:
    base = Path(getattr(context, "workspace", None) or ".")
    d = base / GEN_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(registry: Any) -> None:
    """Attach the media-generation tools to a registry."""
    context = registry.context

    @registry.register(
        "image_generate",
        description=("Generate an image from a text prompt using Devon's "
                     "native diffusion pipeline ('a cyberpunk market at night', "
                     "'portrait of a woman, oil painting'). "
                     "Options: style preset, aspect ratio, seed. "
                     "Returns the saved file path."),
        capability=Capability.FS_WRITE,
    )
    def image_generate(prompt: str, *, style: str = "",
                       aspect: str = "square", seed: int = 0,
                       steps: int = 0) -> dict[str, Any]:
        """Text-to-image via the native pipeline."""
        from ..media.imggen.studio import Studio
        if not (prompt or "").strip():
            raise ValueError("prompt is required")
        studio = Studio()
        out = _out_dir(context, "images")
        kwargs: dict[str, Any] = {"save_to": str(out)}
        if seed:
            kwargs["seed"] = seed
        if steps:
            kwargs["steps"] = steps
        if aspect and aspect != "square":
            kwargs["aspect_ratio"] = aspect
        # Style presets compose prompt modifiers — the engine stays
        # style-agnostic; styles live here at the tool surface.
        full_prompt = _apply_style_preset(prompt, style) if style else prompt
        paths = studio.generate(full_prompt, **kwargs)
        return {"paths": paths, "prompt": full_prompt,
                "style": style or "none"}

    @registry.register(
        "image_variation",
        description=("Create a variation of an existing image from a prompt "
                     "('make this look like a watercolor'). "
                     "strength 0-1 controls how much changes."),
        capability=Capability.FS_WRITE,
    )
    def image_variation(image_path: str, prompt: str, *,
                        strength: float = 0.6) -> dict[str, Any]:
        """Image-to-image variation via the native pipeline."""
        from ..media.imggen.studio import Studio
        src = _sandbox(context, image_path)
        studio = Studio()
        out = _out_dir(context, "images")
        paths = studio.img2img(str(src), prompt, strength=strength,
                               save_to=str(out))
        return {"paths": paths, "prompt": prompt, "strength": strength}

    @registry.register(
        "image_inpaint",
        description=("Inpaint a masked region of an image from a prompt. "
                     "Provide the image and a mask (white = repaint)."),
        capability=Capability.FS_WRITE,
    )
    def image_inpaint(image_path: str, mask_path: str,
                      prompt: str) -> dict[str, Any]:
        """Inpainting via the native pipeline."""
        from ..media.imggen.studio import Studio
        src = _sandbox(context, image_path)
        mask = _sandbox(context, mask_path)
        studio = Studio()
        out = _out_dir(context, "images")
        paths = studio.inpaint(str(src), str(mask), prompt,
                               save_to=str(out))
        return {"paths": paths, "prompt": prompt}

    @registry.register(
        "image_outpaint",
        description=("Extend an image beyond its borders from a prompt "
                     "('continue the landscape to the left')."),
        capability=Capability.FS_WRITE,
    )
    def image_outpaint(image_path: str, prompt: str, *,
                       direction: str = "all") -> dict[str, Any]:
        """Outpainting via the native pipeline."""
        from ..media.imggen.studio import Studio
        src = _sandbox(context, image_path)
        studio = Studio()
        out = _out_dir(context, "images")
        paths = studio.outpaint(str(src), prompt, save_to=str(out))
        return {"paths": paths, "prompt": prompt, "direction": direction}

    @registry.register(
        "image_batch",
        description=("Generate multiple images from a list of prompts in one "
                     "call. Returns all saved paths."),
        capability=Capability.FS_WRITE,
    )
    def image_batch(prompts: list[str], *, style: str = "",
                    aspect: str = "square") -> dict[str, Any]:
        """Batch text-to-image."""
        from ..media.imggen.studio import Studio
        prompts = [p for p in (prompts or []) if p and p.strip()]
        if not prompts:
            raise ValueError("at least one prompt is required")
        studio = Studio()
        out = _out_dir(context, "images")
        all_paths: list[str] = []
        for p in prompts:
            full = _apply_style_preset(p, style) if style else p
            all_paths.extend(studio.generate(
                full, save_to=str(out),
                **({"aspect_ratio": aspect} if aspect != "square" else {})))
        return {"paths": all_paths, "count": len(all_paths)}

    @registry.register(
        "video_generate",
        description=("Generate a short video clip from a text prompt "
                     "('a drone shot over Lagos at sunset'). "
                     "Neural diffusion when the hardware allows, motion-studio "
                     "fallback otherwise. duration_s in seconds."),
        capability=Capability.FS_WRITE,
    )
    def video_generate(prompt: str, *, duration_s: float = 5.0,
                       width: int = 0, height: int = 0,
                       seed: int = 0) -> dict[str, Any]:
        """Text-to-video via videogen (neural → motion fallback)."""
        from ..media.videogen.pipeline import generate
        if not (prompt or "").strip():
            raise ValueError("prompt is required")
        out = _out_dir(context, "videos")
        result = generate(prompt, duration_s=duration_s, seed=seed,
                          out=str(out / "clip.mp4"),
                          **({"width": width} if width else {}),
                          **({"height": height} if height else {}))
        return {"path": result.path, "engine": result.engine,
                "routed": result.routed, "note": result.note,
                "prompt": prompt}

    @registry.register(
        "video_animate",
        description=("Animate a still image into a video clip "
                     "(image-to-video). Ken Burns motion when neural "
                     "is unavailable."),
        capability=Capability.FS_WRITE,
    )
    def video_animate(image_path: str, *, prompt: str = "",
                      duration_s: float = 5.0) -> dict[str, Any]:
        """Image-to-video via videogen."""
        from ..media.videogen.pipeline import generate
        src = _sandbox(context, image_path)
        out = _out_dir(context, "videos")
        result = generate(prompt or "animate this image", mode="i2v",
                          image=str(src), duration_s=duration_s,
                          out=str(out / "animated.mp4"))
        return {"path": result.path, "engine": result.engine,
                "routed": result.routed, "note": result.note}


# ── style presets (tool surface; engines stay style-agnostic) ─────────────

_STYLE_PRESETS: dict[str, str] = {
    "photoreal": ", photorealistic, 85mm, natural light, sharp focus, "
                 "high detail",
    "cinematic": ", cinematic still, dramatic lighting, film grain, "
                 "anamorphic, moody atmosphere",
    "anime": ", anime style, cel shaded, vibrant, detailed background",
    "oil": ", oil painting, textured brushstrokes, classical composition",
    "cyberpunk": ", cyberpunk, neon lights, rain, futuristic, "
                 "high contrast",
    "minimal": ", minimalist, clean composition, negative space, "
               "soft tones",
}


def _apply_style_preset(prompt: str, style: str) -> str:
    key = (style or "").strip().lower()
    suffix = _STYLE_PRESETS.get(key)
    if suffix is None:
        _log.warning("unknown image style preset %r — using raw prompt", style)
        return prompt
    return prompt.rstrip() + suffix


def _sandbox(context: Any, path: str) -> Path:
    from .filesystem import safe_path
    return Path(safe_path(context, path))
