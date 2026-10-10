"""Photoshop-style layer stack for the media editor.

A real interactive layer model on top of :mod:`.studio` — the studio
already knows how to *render* layers (``composite_layers``,
``BLEND_MODES``, ``render_text_layer``, ``render_shape_layer``); this
module adds the stack editors actually work with:

- named layers with unique ids, per-layer opacity / blend mode / visibility
- image, text, shape and solid-fill layer types, positioned by
  ``(x, y)`` pixels or named anchors (``"center"``, ``"top-left"``, ...)
- ``flatten()`` composites visible layers bottom-to-top
- ``to_dict()`` / ``from_dict()`` serialize the whole stack as JSON so a
  layer stack is a saveable, replayable document (like studio projects)

Nothing here re-implements rendering or blending — ``flatten()`` builds
studio layer specs and hands them to ``composite_layers``. Only Pillow is
needed; no network.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from .images import MediaEditError, _require_pillow, load_image
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

LAYERS_VERSION = 1

LAYER_TYPES = ("image", "text", "shape", "solid")

# Named anchors understood by studio's _resolve_anchor.
POSITION_ANCHORS = (
    "center", "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right",
)

# Subset of render_text_layer kwargs exposed through add_text().
TEXT_KWARGS = (
    "stroke_width", "stroke_fill", "shadow", "letter_spacing",
    "line_spacing", "align", "box_width", "box_bg", "box_padding",
    "rotation", "max_width_frac",
)

SHAPE_TYPES = ("rect", "rectangle", "ellipse", "circle", "line", "arrow")


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _validate_blend(blend: str) -> str:
    from .studio import BLEND_MODES
    if blend not in BLEND_MODES:
        raise MediaEditError(
            f"unknown blend mode {blend!r}; use {list(BLEND_MODES)}")
    return blend


def _validate_opacity(value: Any) -> float:
    """Numbers are clamped into [0, 1]; non-numbers fail fast."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise MediaEditError(
            f"opacity must be a number in 0..1, got {value!r}") from None
    if math.isnan(f) or math.isinf(f):
        raise MediaEditError(
            f"opacity must be a finite number in 0..1, got {value!r}")
    return max(0.0, min(1.0, f))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_position(position: Any) -> Any:
    if isinstance(position, str):
        if position not in POSITION_ANCHORS:
            raise MediaEditError(
                f"unknown position anchor {position!r}; "
                f"use {list(POSITION_ANCHORS)} or (x, y)")
        return position
    if (isinstance(position, (list, tuple)) and len(position) == 2
            and all(_is_number(v) for v in position)):
        return (int(position[0]), int(position[1]))
    raise MediaEditError(
        f"position must be an (x, y) pair or one of "
        f"{list(POSITION_ANCHORS)}, got {position!r}")


def _validate_color(color: Any, what: str = "color") -> Any:
    Image = _require_pillow()
    from PIL import ImageColor
    if not isinstance(color, str):
        raise MediaEditError(f"{what} must be a color string, got {color!r}")
    try:
        ImageColor.getrgb(color)
    except ValueError:
        raise MediaEditError(f"invalid {what} {color!r}") from None
    return color


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """One entry in a :class:`LayerStack`. Plain data + spec rendering."""

    def __init__(self, lid: str, name: str, ltype: str, *,
                 opacity: float = 1.0, blend: str = "normal",
                 visible: bool = True, position: Any = (0, 0),
                 margin: int = 0, params: dict[str, Any] | None = None):
        if ltype not in LAYER_TYPES:
            raise MediaEditError(f"unknown layer type {ltype!r}")
        self.id = lid
        self.name = name or f"{ltype} {lid}"
        self.type = ltype
        self.opacity = _validate_opacity(opacity)
        self.blend = _validate_blend(blend)
        self.visible = bool(visible)
        self.position = _validate_position(position)
        self.margin = int(margin)
        self.params: dict[str, Any] = dict(params or {})

    # -- studio spec rendering -------------------------------------------

    def to_spec(self, tmpdir: str) -> dict[str, Any]:
        """Render this layer as a ``composite_layers`` spec dict.

        In-memory image layers are written to ``tmpdir`` as PNGs so
        studio's renderer (which takes paths) can consume them.
        """
        base: dict[str, Any] = {
            "opacity": self.opacity,
            "blend": self.blend,
            "position": self.position,
            "margin": self.margin,
        }
        p = self.params
        if self.type == "image":
            spec = dict(base, type="image")
            src = p.get("source_path")
            if src:
                spec["path"] = src
            else:
                img = p.get("image")
                if img is None:
                    raise MediaEditError(
                        f"image layer {self.id!r} has no image data")
                tmp = str(Path(tmpdir) / f"{self.id}.png")
                img.save(tmp, "PNG")
                spec["path"] = tmp
            for key in ("scale", "size", "width", "rotation"):
                if key in p:
                    spec[key] = p[key]
            return spec
        if self.type == "text":
            spec = dict(base, type="text", text=p["text"],
                        font=p.get("font"), size=p["font_size"],
                        color=p["color"])
            for key in TEXT_KWARGS:
                if key in p:
                    spec[key] = p[key]
            return spec
        if self.type == "shape":
            return dict(base, type="shape", shape=p["shape"], box=p["box"],
                        fill=p.get("fill"), outline=p.get("outline"),
                        width=p.get("width", 4), radius=p.get("radius", 0))
        # solid → a full-canvas rect (studio's render_shape_layer
        # understands box="full")
        return dict(base, type="shape", shape="rect", box="full",
                    fill=p["color"])

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "opacity": self.opacity,
            "blend": self.blend,
            "visible": self.visible,
            "position": list(self.position)
            if isinstance(self.position, tuple) else self.position,
            "margin": self.margin,
        }
        p = self.params
        if self.type == "image":
            src = p.get("source_path")
            if not src:
                raise MediaEditError(
                    f"layer {self.id!r} ({self.name!r}) is an in-memory "
                    "image and cannot be serialized; add it from a file "
                    "path instead to make the stack serializable")
            d["source_path"] = src
            for key in ("scale", "size", "width", "rotation"):
                if key in p:
                    d[key] = p[key]
        elif self.type == "text":
            d["text"] = p["text"]
            d["font_size"] = p["font_size"]
            d["color"] = p["color"]
            if p.get("font") is not None:
                d["font"] = p["font"]
            for key in TEXT_KWARGS:
                if key in p:
                    d[key] = p[key]
        elif self.type == "shape":
            for key in ("shape", "box", "fill", "outline", "width", "radius"):
                if key in p:
                    d[key] = p[key]
        else:  # solid
            d["color"] = p["color"]
            if p.get("size") is not None:
                d["size"] = list(p["size"])
        return d

    def info(self, canvas_size: tuple[int, int]) -> dict[str, Any]:
        """Public summary dict for :meth:`LayerStack.layer_info`."""
        size: Any = None
        p = self.params
        if self.type == "image" and p.get("image") is not None:
            size = list(p["image"].size)
        elif self.type == "shape":
            box = p["box"]
            size = [int(box[2]) - int(box[0]), int(box[3]) - int(box[1])]
        elif self.type == "solid":
            size = list(p.get("size") or canvas_size)
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "opacity": self.opacity,
            "blend": self.blend,
            "visible": self.visible,
            "position": list(self.position)
            if isinstance(self.position, tuple) else self.position,
            "size": size,
        }


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------

class LayerStack:
    """Photoshop-style layer stack.

    ``LayerStack(base)`` where ``base`` is a PIL image, a path to one, or
    a ``(width, height)`` tuple for a blank canvas (``bg=`` sets its
    color).
    """

    def __init__(self, base: Any, *, bg: str = "white"):
        Image = _require_pillow()
        self._layers: list[Layer] = []
        self._counter = 0
        if isinstance(base, (str, Path)):
            path = Path(base)
            if not path.exists():
                raise MediaEditError(f"base image not found: {path}")
            self._base_kind = "path"
            self._base_spec: dict[str, Any] = {"path": str(path)}
            img = load_image(path)
        elif isinstance(base, tuple) and len(base) == 2 and all(
                isinstance(v, int) and v > 0 for v in base):
            _validate_color(bg, "bg")
            self._base_kind = "blank"
            self._base_spec = {"size": [int(base[0]), int(base[1])], "bg": bg}
            img = Image.new("RGBA", (int(base[0]), int(base[1])), bg)
        elif isinstance(base, Image.Image):
            self._base_kind = "image"
            self._base_spec = {}
            img = base.copy()
        else:
            raise MediaEditError(
                "base must be a PIL image, a path, or a (width, height) "
                f"tuple, got {type(base).__name__}")
        self._base_mode = img.mode
        self._base_image = img.convert("RGBA")

    # -- internals ----------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"layer_{self._counter}"

    def _get(self, lid: str) -> Layer:
        for layer in self._layers:
            if layer.id == lid:
                return layer
        raise MediaEditError(f"unknown layer id {lid!r}")

    def _index(self, lid: str) -> int:
        layer = self._get(lid)
        return self._layers.index(layer)

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._base_image.size

    # -- adding layers ------------------------------------------------------

    def add_image(self, image_or_path: Any, *, name: str | None = None,
                  opacity: float = 1.0, blend: str = "normal",
                  position: Any = (0, 0), scale: float | None = None,
                  width: int | None = None,
                  size: tuple[int, int] | None = None,
                  rotation: float = 0.0, margin: int = 20) -> str:
        """Add an image layer from a path or a PIL image. Returns the id."""
        Image = _require_pillow()
        source_path: str | None = None
        if isinstance(image_or_path, (str, Path)):
            path = Path(image_or_path)
            if not path.exists():
                raise MediaEditError(f"layer image not found: {path}")
            source_path = str(path)
            img = load_image(path)
        elif isinstance(image_or_path, Image.Image):
            img = image_or_path.copy()
        else:
            raise MediaEditError(
                "image layer needs a PIL image or a path, got "
                f"{type(image_or_path).__name__}")
        if scale is not None:
            if not _is_number(scale) or scale <= 0:
                raise MediaEditError(f"scale must be a positive number, "
                                     f"got {scale!r}")
        if width is not None and (not _is_number(width) or width <= 0):
            raise MediaEditError(f"width must be a positive number, "
                                 f"got {width!r}")
        if size is not None:
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or not all(_is_number(v) and v > 0 for v in size)):
                raise MediaEditError(
                    f"size must be a (width, height) pair of positive "
                    f"numbers, got {size!r}")
            size = (int(size[0]), int(size[1]))
        if not _is_number(rotation):
            raise MediaEditError(f"rotation must be a number, got {rotation!r}")
        params: dict[str, Any] = {
            "image": img.convert("RGBA"),
            "rotation": float(rotation),
        }
        if source_path:
            params["source_path"] = source_path
        if scale is not None:
            params["scale"] = float(scale)
        if width is not None:
            params["width"] = int(width)
        if size is not None:
            params["size"] = list(size)
        layer = Layer(self._new_id(), name or "", "image",
                      opacity=_validate_opacity(opacity),
                      blend=_validate_blend(blend),
                      position=_validate_position(position),
                      margin=margin, params=params)
        self._layers.append(layer)
        return layer.id

    def add_text(self, text: str, *, name: str | None = None,
                 font_size: int = 64, color: str = "white",
                 font: str | None = None, position: Any = "center",
                 opacity: float = 1.0, blend: str = "normal",
                 margin: int = 24, **text_kwargs: Any) -> str:
        """Add a text layer. Rendering reuses studio's ``render_text_layer``
        (same fonts, stroke, shadow, letter-spacing support)."""
        if not text or not str(text).strip():
            raise MediaEditError("text layer needs non-empty text")
        if not _is_number(font_size) or font_size <= 0:
            raise MediaEditError(f"font_size must be positive, "
                                 f"got {font_size!r}")
        _validate_color(color, "color")
        unknown = set(text_kwargs) - set(TEXT_KWARGS)
        if unknown:
            raise MediaEditError(
                f"unknown text options {sorted(unknown)}; "
                f"allowed: {list(TEXT_KWARGS)}")
        params: dict[str, Any] = {
            "text": str(text),
            "font_size": int(font_size),
            "color": color,
            "font": font,
        }
        params.update(text_kwargs)
        layer = Layer(self._new_id(), name or "", "text",
                      opacity=_validate_opacity(opacity),
                      blend=_validate_blend(blend),
                      position=_validate_position(position),
                      margin=margin, params=params)
        self._layers.append(layer)
        return layer.id

    def add_shape(self, shape: str, *, name: str | None = None,
                  position: tuple[int, int] = (0, 0),
                  size: tuple[int, int] = (100, 100),
                  box: list[int] | tuple[int, ...] | None = None,
                  fill: str | None = "white", outline: str | None = None,
                  width: int = 4, radius: int = 0,
                  opacity: float = 1.0, blend: str = "normal",
                  margin: int = 0) -> str:
        """Add a shape layer (``rect``/``ellipse``/``line``/``arrow``...).

        Position with ``position`` + ``size``, or pass an explicit canvas
        ``box=[l, t, r, b]`` (studio's shape convention).
        """
        if shape not in SHAPE_TYPES:
            raise MediaEditError(f"unknown shape {shape!r}; "
                                 f"use {list(SHAPE_TYPES)}")
        if box is not None:
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(_is_number(v) for v in box)):
                raise MediaEditError(
                    f"box must be [l, t, r, b] numbers, got {box!r}")
            box = [int(v) for v in box]
        else:
            pos = _validate_position(position)
            if isinstance(pos, str):
                raise MediaEditError(
                    "shape layers need an explicit (x, y) position (or a "
                    "box=); anchors are only resolved after rendering")
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or not all(_is_number(v) and v > 0 for v in size)):
                raise MediaEditError(
                    f"size must be a (width, height) pair of positive "
                    f"numbers, got {size!r}")
            box = [int(pos[0]), int(pos[1]),
                   int(pos[0]) + int(size[0]), int(pos[1]) + int(size[1])]
        if fill is not None:
            _validate_color(fill, "fill")
        if outline is not None:
            _validate_color(outline, "outline")
        if not _is_number(width) or width <= 0:
            raise MediaEditError(f"width must be positive, got {width!r}")
        params = {"shape": shape, "box": box, "fill": fill,
                  "outline": outline, "width": int(width),
                  "radius": int(radius)}
        layer = Layer(self._new_id(), name or "", "shape",
                      opacity=_validate_opacity(opacity),
                      blend=_validate_blend(blend),
                      position=(box[0], box[1]),
                      margin=margin, params=params)
        self._layers.append(layer)
        return layer.id

    def add_solid(self, color: str, *, name: str | None = None,
                  size: tuple[int, int] | None = None,
                  opacity: float = 1.0, blend: str = "normal") -> str:
        """Add a solid color fill layer (full canvas, or ``size``)."""
        _validate_color(color, "color")
        if size is not None:
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or not all(_is_number(v) and v > 0 for v in size)):
                raise MediaEditError(
                    f"size must be a (width, height) pair of positive "
                    f"numbers, got {size!r}")
            size = (int(size[0]), int(size[1]))
        params: dict[str, Any] = {"color": color, "size": size}
        layer = Layer(self._new_id(), name or "", "solid",
                      opacity=_validate_opacity(opacity),
                      blend=_validate_blend(blend),
                      position=(0, 0), params=params)
        self._layers.append(layer)
        return layer.id

    # -- per-layer edits ----------------------------------------------------

    def set_opacity(self, lid: str, value: float) -> None:
        """Set layer opacity; out-of-range numbers are clamped to [0, 1]."""
        self._get(lid).opacity = _validate_opacity(value)

    def set_blend(self, lid: str, mode: str) -> None:
        self._get(lid).blend = _validate_blend(mode)

    def set_visible(self, lid: str, visible: bool) -> None:
        self._get(lid).visible = bool(visible)

    def move(self, lid: str, x: float, y: float) -> None:
        """Move a layer's top-left corner to ``(x, y)`` pixels."""
        if not _is_number(x) or not _is_number(y):
            raise MediaEditError(f"move needs numeric x/y, got {x!r}, {y!r}")
        layer = self._get(lid)
        layer.position = (int(x), int(y))
        if layer.type == "shape":
            box = layer.params["box"]
            w, h = box[2] - box[0], box[3] - box[1]
            layer.params["box"] = [int(x), int(y), int(x) + w, int(y) + h]

    def rename(self, lid: str, name: str) -> None:
        if not name or not str(name).strip():
            raise MediaEditError("layer name must be non-empty")
        self._get(lid).name = str(name)

    def nudge(self, lid: str, dx: float = 0, dy: float = 0) -> None:
        """Move a layer by a relative ``(dx, dy)`` offset in pixels."""
        layer = self._get(lid)
        pos = layer.position
        if isinstance(pos, str):
            raise MediaEditError(
                "cannot nudge an anchored layer — move() it to an (x, y) "
                "position first")
        self.move(lid, pos[0] + dx, pos[1] + dy)

    def scale_layer(self, lid: str, factor: float) -> None:
        """Scale an image layer by ``factor`` (relative, > 0).

        Composes with the layer's existing ``scale``/``width``/``size``
        params: an explicit ``size`` is resized in place, otherwise the
        ``scale`` multiplier is updated.
        """
        if not _is_number(factor) or factor <= 0:
            raise MediaEditError(
                f"scale factor must be a positive number, got {factor!r}")
        layer = self._get(lid)
        if layer.type != "image":
            raise MediaEditError(
                f"scale_layer needs an image layer, {lid!r} is "
                f"{layer.type}")
        p = layer.params
        if p.get("size") is not None:
            w, h = p["size"]
            p["size"] = [max(1, int(w * factor)), max(1, int(h * factor))]
        elif p.get("width") is not None:
            p["width"] = max(1, int(p["width"] * factor))
        else:
            p["scale"] = float(p.get("scale", 1.0)) * float(factor)

    def duplicate(self, lid: str, *, name: str | None = None,
                  dx: float = 0, dy: float = 0) -> str:
        """Copy a layer (params deep-copied, new id). ``dx``/``dy`` nudge
        the copy so it doesn't sit exactly on top of the original."""
        import copy as _copy
        src = self._get(lid)
        params = {k: _copy.deepcopy(v) for k, v in src.params.items()
                  if k != "image"}
        # in-memory images are copied, never shared by reference
        if src.params.get("image") is not None:
            params["image"] = src.params["image"].copy()
        dup = Layer(self._new_id(), name or f"{src.name} copy", src.type,
                    opacity=src.opacity, blend=src.blend,
                    visible=src.visible, position=src.position,
                    margin=src.margin, params=params)
        self._layers.append(dup)
        if dx or dy:
            try:
                self.nudge(dup.id, dx, dy)
            except MediaEditError:
                pass  # anchored layers stay anchored
        return dup.id

    def remove(self, lid: str) -> dict[str, Any]:
        """Remove a layer; returns its info dict."""
        idx = self._index(lid)
        layer = self._layers.pop(idx)
        return layer.info(self.canvas_size)

    def reorder(self, lid: str, new_index: int) -> int:
        """Move a layer to ``new_index`` (0 = bottom). Returns the index."""
        if not isinstance(new_index, int) or isinstance(new_index, bool):
            raise MediaEditError(f"new_index must be an int, "
                                 f"got {new_index!r}")
        idx = self._index(lid)
        layer = self._layers.pop(idx)
        new_index = max(0, min(new_index, len(self._layers)))
        self._layers.insert(new_index, layer)
        return new_index

    def move_up(self, lid: str) -> int:
        """Move a layer one step toward the top. Returns the new index."""
        idx = self._index(lid)
        if idx < len(self._layers) - 1:
            self._layers[idx], self._layers[idx + 1] = \
                self._layers[idx + 1], self._layers[idx]
            return idx + 1
        return idx

    def move_down(self, lid: str) -> int:
        """Move a layer one step toward the bottom. Returns the new index."""
        idx = self._index(lid)
        if idx > 0:
            self._layers[idx], self._layers[idx - 1] = \
                self._layers[idx - 1], self._layers[idx]
            return idx - 1
        return idx

    # -- rendering ----------------------------------------------------------

    def flatten(self) -> Any:
        """Composite visible layers bottom-to-top → a single PIL image.

        Reuses :func:`.studio.composite_layers` (opacity, blend modes and
        canvas clipping all live there). In-memory image layers are staged
        to a temp dir that is removed afterwards.
        """
        from .studio import composite_layers
        with tempfile.TemporaryDirectory(prefix="layerstack_") as tmpdir:
            specs = [layer.to_spec(tmpdir) for layer in self._layers
                     if layer.visible]
            out = composite_layers(self._base_image.copy(), specs)
        if self._base_mode == "RGB" and out.mode != "RGB":
            out = out.convert("RGB")
        return out

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stack to JSON-safe dicts.

        Image layers built from in-memory images (and an in-memory base)
        cannot be serialized — a clear :class:`MediaEditError` is raised
        naming the offending layer.
        """
        if self._base_kind == "path":
            base = {"kind": "path", "path": self._base_spec["path"]}
        elif self._base_kind == "blank":
            base = {"kind": "blank", **self._base_spec}
        else:
            raise MediaEditError(
                "cannot serialize a LayerStack built on an in-memory base "
                "image; construct it from a path or a (width, height) "
                "blank canvas instead")
        return {
            "version": LAYERS_VERSION,
            "base": base,
            "layers": [layer.to_dict() for layer in self._layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerStack":
        """Rebuild a stack serialized with :meth:`to_dict`."""
        if not isinstance(data, dict):
            raise MediaEditError("from_dict needs a dict")
        if data.get("version") != LAYERS_VERSION:
            raise MediaEditError(
                f"unsupported layer stack version "
                f"{data.get('version')!r}; expected {LAYERS_VERSION}")
        base = data.get("base") or {}
        kind = base.get("kind")
        if kind == "path":
            stack = cls(base["path"])
        elif kind == "blank":
            stack = cls(tuple(base["size"]), bg=base.get("bg", "white"))
        else:
            raise MediaEditError(f"unknown base kind {kind!r}")
        for lspec in data.get("layers", []):
            stack._layer_from_dict(dict(lspec))
        return stack

    def _layer_from_dict(self, d: dict[str, Any]) -> str:
        ltype = d.get("type")
        lid = d.get("id")
        name = d.get("name")
        opacity = d.get("opacity", 1.0)
        blend = d.get("blend", "normal")
        visible = d.get("visible", True)
        position = d.get("position", (0, 0))
        margin = d.get("margin", 0)
        if ltype == "image":
            lid2 = self.add_image(
                d["source_path"], name=name, opacity=opacity, blend=blend,
                position=position,
                scale=d.get("scale"), width=d.get("width"),
                size=tuple(d["size"]) if d.get("size") else None,
                rotation=d.get("rotation", 0.0), margin=margin)
        elif ltype == "text":
            text_kwargs = {k: d[k] for k in TEXT_KWARGS if k in d}
            lid2 = self.add_text(
                d["text"], name=name, font_size=d.get("font_size", 64),
                color=d.get("color", "white"), font=d.get("font"),
                position=position, opacity=opacity, blend=blend,
                margin=margin, **text_kwargs)
        elif ltype == "shape":
            lid2 = self.add_shape(
                d.get("shape", "rect"), name=name,
                box=list(d["box"]), fill=d.get("fill"),
                outline=d.get("outline"), width=d.get("width", 4),
                radius=d.get("radius", 0), opacity=opacity, blend=blend,
                margin=margin)
        elif ltype == "solid":
            lid2 = self.add_solid(
                d["color"], name=name,
                size=tuple(d["size"]) if d.get("size") else None,
                opacity=opacity, blend=blend)
        else:
            raise MediaEditError(f"unknown layer type {ltype!r}")
        # restore the original id (keep counter ahead to avoid reuse)
        layer = self._layers.pop()
        layer.id = lid
        layer.visible = bool(visible)
        self._layers.append(layer)
        try:
            n = int(str(lid).split("_")[-1])
            self._counter = max(self._counter, n)
        except (ValueError, IndexError):
            # Non-standard id (e.g. user-renamed): keep the counter as-is;
            # _new_id() already guarantees uniqueness against live ids.
            _log.debug("layer id %r has no numeric suffix; counter unchanged",
                       lid)
        return lid2

    # -- inspection ---------------------------------------------------------

    def layer_info(self) -> list[dict[str, Any]]:
        """Bottom-to-top list of {id, name, type, opacity, blend, visible,
        position, size} dicts."""
        return [layer.info(self.canvas_size) for layer in self._layers]

    def __len__(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"LayerStack(canvas={self.canvas_size}, "
                f"layers={len(self._layers)})")
