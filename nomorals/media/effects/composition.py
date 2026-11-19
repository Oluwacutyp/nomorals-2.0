"""Compositions — EffectCraft's composition/layer model, real impl.

A :class:`Composition` is a stack of :class:`Layer`\\ s (solid,
text, image, shape) composited bottom-to-top at time ``t``:
keyframed transforms (position/scale/rotation/opacity via
:class:`~.keyframes.PropertyTrack` or plain numbers), blend modes,
per-layer raster effects, and an optional 3D camera that views the
whole frame through :func:`~.camera.apply_camera`.

Frames are rendered with PIL/numpy only — no JS, no browser — and
:class:`render_composition_video` streams them through the
motion-studio ``render_sequence`` seam, so a composition of any
length renders without holding every frame in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..motion_studio._core import find_font, render_sequence
from . import effects as fx_effects
from .camera import Camera3D, apply_camera
from .keyframes import PropertyTrack

__all__ = [
    "Composition",
    "CompositionError",
    "Layer",
    "render_composition",
    "render_composition_video",
]

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "add")
LAYER_KINDS = ("solid", "text", "image", "shape")


class CompositionError(ValueError):
    """A composition document or render problem, named plainly."""


def _value(track_or_float: float | PropertyTrack, t: float) -> float:
    if isinstance(track_or_float, PropertyTrack):
        return float(track_or_float.sample(t))
    return float(track_or_float)


@dataclass
class Layer:
    """One layer of a composition.

    ``transform`` keys: ``position_x``, ``position_y`` (canvas
    pixels; default centres the layer), ``scale`` (default 1.0),
    ``rotation`` (degrees, default 0). Each value is a plain number
    or a :class:`~.keyframes.PropertyTrack`. ``opacity`` is a number
    or a track too. ``effects`` is a list of ``{"name": ..., "params":
    {...}}`` raster effects, applied bottom-to-top in order.
    """

    kind: str = "solid"
    color: tuple[int, int, int] = (0, 0, 0)
    text: str = ""
    font_size: int = 48
    image_path: str | None = None
    shape: str = "rect"
    size: tuple[int, int] | None = None
    transform: dict[str, float | PropertyTrack] = field(default_factory=dict)
    opacity: float | PropertyTrack = 1.0
    blend: str = "normal"
    effects: list[dict[str, Any]] = field(default_factory=list)
    visible: bool = True

    def __post_init__(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise CompositionError(
                f"layer kind must be one of {LAYER_KINDS}, got {self.kind!r}")
        if self.blend not in BLEND_MODES:
            raise CompositionError(
                f"blend mode must be one of {BLEND_MODES}, got {self.blend!r}")
        if len(self.color) != 3 or any(
                not isinstance(c, int) or c < 0 or c > 255 for c in self.color):
            raise CompositionError(
                f"layer color must be (r, g, b) 0..255, got {self.color!r}")
        if self.kind == "image" and not self.image_path:
            raise CompositionError("an image layer needs image_path")
        if self.kind == "text" and not self.text:
            raise CompositionError("a text layer needs text")
        if self.size is not None and (self.size[0] <= 0 or self.size[1] <= 0):
            raise CompositionError(f"layer size must be positive, got {self.size!r}")
        for key in self.transform:
            if key not in ("position_x", "position_y", "scale", "rotation"):
                raise CompositionError(
                    f"unknown transform key {key!r} — position_x, "
                    f"position_y, scale, rotation")
        for effect in self.effects:
            if effect.get("name") not in fx_effects.list_effects():
                raise CompositionError(
                    f"unknown layer effect {effect.get('name')!r}")

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def _serialise(value: float | PropertyTrack) -> Any:
            if isinstance(value, PropertyTrack):
                return {"track": value.to_dict()}
            return value

        return {
            "kind": self.kind,
            "color": list(self.color),
            "text": self.text,
            "font_size": self.font_size,
            "image_path": self.image_path,
            "shape": self.shape,
            "size": list(self.size) if self.size else None,
            "transform": {k: _serialise(v)
                          for k, v in self.transform.items()},
            "opacity": _serialise(self.opacity),
            "blend": self.blend,
            "effects": list(self.effects),
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        def _restore(value: Any) -> float | PropertyTrack:
            if isinstance(value, dict) and "track" in value:
                return PropertyTrack.from_dict(value["track"])
            return float(value)

        return cls(
            kind=str(data.get("kind", "solid")),
            color=tuple(data.get("color", (0, 0, 0))),
            text=str(data.get("text", "")),
            font_size=int(data.get("font_size", 48)),
            image_path=data.get("image_path"),
            shape=str(data.get("shape", "rect")),
            size=tuple(data["size"]) if data.get("size") else None,
            transform={k: _restore(v)
                       for k, v in (data.get("transform") or {}).items()},
            opacity=_restore(data.get("opacity", 1.0)),
            blend=str(data.get("blend", "normal")),
            effects=list(data.get("effects") or []),
            visible=bool(data.get("visible", True)),
        )


@dataclass
class Composition:
    """A motion-graphics composition: a canvas, a layer stack, a
    duration, an optional 3D camera."""

    width: int
    height: int
    fps: float = 24.0
    duration: float = 1.0
    background: tuple[int, int, int] = (0, 0, 0)
    layers: list[Layer] = field(default_factory=list)
    camera: Camera3D | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CompositionError("composition canvas must be positive")
        if self.fps <= 0:
            raise CompositionError("composition fps must be > 0")
        if self.duration <= 0:
            raise CompositionError("composition duration must be > 0")
        if self.camera is not None and not isinstance(self.camera, Camera3D):
            self.camera = Camera3D.from_dict(dict(self.camera))

    def add_layer(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        return layer

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "fps": self.fps, "duration": self.duration,
            "background": list(self.background),
            "layers": [layer.to_dict() for layer in self.layers],
            "camera": self.camera.to_dict() if self.camera else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Composition:
        return cls(
            width=int(data["width"]), height=int(data["height"]),
            fps=float(data.get("fps", 24.0)),
            duration=float(data.get("duration", 1.0)),
            background=tuple(data.get("background", (0, 0, 0))),
            layers=[Layer.from_dict(item)
                    for item in data.get("layers") or []],
            camera=Camera3D.from_dict(data["camera"])
            if data.get("camera") else None,
        )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _render_layer_bitmap(layer: Layer, comp: Composition,
                         t: float) -> Image.Image:
    """The layer's own pixels at its own size (before transform)."""
    size = layer.size or (comp.width, comp.height)
    if layer.kind == "solid":
        return Image.new("RGBA", size, (*layer.color, 255))
    if layer.kind == "shape":
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if layer.shape == "ellipse":
            draw.ellipse([0, 0, size[0] - 1, size[1] - 1],
                         fill=(*layer.color, 255))
        else:
            draw.rectangle([0, 0, size[0] - 1, size[1] - 1],
                           fill=(*layer.color, 255))
        return img
    if layer.kind == "text":
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = find_font(layer.font_size)
        draw.text((size[0] / 2, size[1] / 2), layer.text,
                  font=font, fill=(*layer.color, 255), anchor="mm")
        return img
    if layer.kind == "image":
        src = Image.open(str(layer.image_path)).convert("RGBA")
        # cover-fit: fill the layer box, centre-cropped
        scale = max(size[0] / src.size[0], size[1] / src.size[1])
        fitted = src.resize((max(1, int(src.size[0] * scale)),
                             max(1, int(src.size[1] * scale))),
                            Image.BICUBIC)
        left = (fitted.size[0] - size[0]) // 2
        top = (fitted.size[1] - size[1]) // 2
        return fitted.crop((left, top, left + size[0], top + size[1]))
    raise CompositionError(f"cannot render layer kind {layer.kind!r}")


def _blend(base: np.ndarray, layer: np.ndarray, mode: str) -> np.ndarray:
    """Blend RGB arrays (0..255 floats) under ``mode``."""
    if mode == "multiply":
        return base * layer / 255.0
    if mode == "screen":
        return 255.0 - (255.0 - base) * (255.0 - layer) / 255.0
    if mode == "overlay":
        low = 2.0 * base * layer / 255.0
        high = 255.0 - 2.0 * (255.0 - base) * (255.0 - layer) / 255.0
        return np.where(base < 128, low, high)
    if mode == "add":
        return np.clip(base + layer, 0, 255)
    return layer


def render_composition(comp: Composition, t: float) -> Image.Image:
    """The composition's frame at time ``t`` (seconds), RGB."""
    canvas = Image.new("RGBA", (comp.width, comp.height),
                       (*comp.background, 255))
    for layer in comp.layers:
        if not layer.visible:
            continue
        bitmap = _render_layer_bitmap(layer, comp, t)
        scale = _value(layer.transform.get("scale", 1.0), t)
        if scale != 1.0:
            bitmap = bitmap.resize(
                (max(1, int(bitmap.size[0] * scale)),
                 max(1, int(bitmap.size[1] * scale))), Image.BICUBIC)
        rotation = _value(layer.transform.get("rotation", 0.0), t)
        if rotation:
            bitmap = bitmap.rotate(rotation, resample=Image.BICUBIC,
                                   expand=False)
        for effect in layer.effects:
            bitmap = fx_effects.apply_effect(
                effect["name"], bitmap.convert("RGB"),
                effect.get("params") or {}).convert("RGBA")
        opacity = _value(layer.opacity, t)
        if not 0.0 <= opacity <= 1.0:
            raise CompositionError(
                f"layer opacity {opacity} out of range [0, 1]")
        if opacity < 1.0:
            alpha = np.asarray(bitmap.split()[3], dtype=np.float64)
            alpha = np.clip(alpha * opacity, 0, 255).astype(np.uint8)
            bitmap.putalpha(Image.fromarray(alpha, mode="L"))
        px = _value(layer.transform.get(
            "position_x", (comp.width - bitmap.size[0]) / 2.0), t)
        py = _value(layer.transform.get(
            "position_y", (comp.height - bitmap.size[1]) / 2.0), t)
        x0, y0 = int(round(px)), int(round(py))
        if layer.blend == "normal":
            canvas.alpha_composite(bitmap, (x0, y0))
            continue
        # non-normal blends work on the overlapped region
        x1, y1 = x0 + bitmap.size[0], y0 + bitmap.size[1]
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(comp.width, x1), min(comp.height, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        region = np.asarray(
            canvas.crop((cx0, cy0, cx1, cy1)).convert("RGB"),
            dtype=np.float64)
        over = np.asarray(
            bitmap.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)),
            dtype=np.float64)
        blended = _blend(region, over[:, :, :3], layer.blend)
        alpha = over[:, :, 3:4] / 255.0
        merged = region * (1 - alpha) + blended * alpha
        patch = Image.fromarray(
            np.clip(merged, 0, 255).astype(np.uint8), mode="RGB")
        canvas.paste(patch, (cx0, cy0))
    out = canvas.convert("RGB")
    if comp.camera is not None:
        out = apply_camera(out, comp.camera, background=comp.background)
    return out


def render_composition_video(comp: Composition, out: str | Path, *,
                             workdir: str | Path | None = None,
                             audio: str | Path | None = None) -> dict[str, Any]:
    """Render the whole composition to an MP4 via the motion-studio
    frame pipe. Returns a report dict; raises CompositionError only
    for a bad document — an encode failure raises the pipe's own
    error with ffmpeg's stderr attached."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    n = comp.n_frames
    render_sequence(n, comp.width, comp.height, comp.fps, out_path,
                    lambda i, t: np.asarray(render_composition(comp, t)),
                    audio=audio)
    return {"ok": True, "path": str(out_path), "frames": n,
            "duration": comp.duration, "fps": comp.fps,
            "width": comp.width, "height": comp.height}
