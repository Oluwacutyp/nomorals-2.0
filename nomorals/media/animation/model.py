"""Animation document model — port of AniCraft ``crates/doc/src/lib.rs``.

A :class:`AnimationDocument` holds scenes; a scene holds a stage and
a :class:`Timeline`; a timeline holds :class:`Layer`s; a layer holds
:class:`Keyframe`s; a keyframe holds :class:`Element`s and the
:class:`Tween` (classic or shape) or :class:`MotionTween` to the next
keyframe. Symbols live in the document library and are instanced as
elements.

The native file format is ``.acraft``: a ZIP with a deterministic
``document.json`` (sorted keys, fixed ZIP timestamp) plus optional
bitmap assets under ``assets/``. ``to_bytes`` is byte-stable: opening
an unedited document and saving it again reproduces the same bytes,
which is what the fidelity report measures.

Layer kinds follow upstream: ``normal``, ``guide`` (never in output),
``mask`` (reveals what it masks), ``folder`` (groups children),
``camera`` (at most one per timeline; its keyframes hold the camera).
``armature`` (bone rigs, ADR 0043) is NOT ported — adding one is
refused honestly, and the survey names it.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import zipfile
from dataclasses import dataclass, field
from typing import Any

from .easing import Easing
from .errors import AnimationError
from .geom import Affine, Path

__all__ = [
    "AnimationDocument",
    "Element",
    "Keyframe",
    "Layer",
    "MotionTween",
    "Paint",
    "Rgba",
    "Scene",
    "Stage",
    "Stroke",
    "Symbol",
    "Timeline",
    "Transform",
    "Tween",
]

MAX_LAYER_DEPTH = 16
MAX_SYMBOL_DEPTH = 32
LAYER_KINDS = ("normal", "guide", "mask", "folder", "camera")
TWEEN_KINDS = ("classic", "shape")
EASE_GROUPS = ("position", "rotation", "scale", "color", "filters")
LOOP_MODES = ("loop", "play-once", "single-frame")
SYMBOL_KINDS = ("graphic", "movieclip", "button")

_id_counter = itertools.count(1)


def _new_id() -> int:
    return next(_id_counter)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Colour, paint, stroke
# ---------------------------------------------------------------------------

@dataclass
class Rgba:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 1.0

    @classmethod
    def from_hex(cls, text: str) -> Rgba:
        s = text.strip().lstrip("#")
        if len(s) == 6:
            s += "ff"
        if len(s) != 8 or any(c not in "0123456789abcdefABCDEF"
                              for c in s):
            raise AnimationError(f"bad colour {text!r}; use #rrggbb")
        return cls(int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
                   int(s[4:6], 16) / 255.0, int(s[6:8], 16) / 255.0)

    def to_hex(self) -> str:
        base = (f"#{round(self.r * 255):02x}{round(self.g * 255):02x}"
                f"{round(self.b * 255):02x}")
        return base if self.a >= 1.0 else base + f"{round(self.a * 255):02x}"

    def to_tuple255(self) -> tuple[int, int, int, int]:
        return (round(self.r * 255), round(self.g * 255),
                round(self.b * 255), round(self.a * 255))

    def to_dict(self) -> dict[str, float]:
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rgba:
        return cls(float(data.get("r", 0.0)), float(data.get("g", 0.0)),
                   float(data.get("b", 0.0)), float(data.get("a", 1.0)))


@dataclass
class Paint:
    """A fill: solid colour, linear/radial gradient, or bitmap asset."""
    kind: str = "solid"  # solid | linear | radial | bitmap
    color: Rgba = field(default_factory=Rgba)
    stops: list[tuple[float, Rgba]] = field(default_factory=list)
    angle: float = 0.0  # linear gradient direction, degrees
    asset: str = ""  # bitmap asset name under assets/

    @classmethod
    def solid(cls, color: str | Rgba) -> Paint:
        if isinstance(color, str):
            color = Rgba.from_hex(color)
        return cls(kind="solid", color=color)

    @classmethod
    def linear(cls, stops: list[tuple[float, str]],
               angle: float = 0.0) -> Paint:
        return cls(kind="linear", angle=float(angle),
                   stops=[(float(t), Rgba.from_hex(c)) for t, c in stops])

    @classmethod
    def radial(cls, stops: list[tuple[float, str]]) -> Paint:
        return cls(kind="radial",
                   stops=[(float(t), Rgba.from_hex(c)) for t, c in stops])

    @classmethod
    def bitmap(cls, asset: str) -> Paint:
        return cls(kind="bitmap", asset=asset)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "color": self.color.to_dict(),
                "stops": [[t, c.to_dict()] for t, c in self.stops],
                "angle": self.angle, "asset": self.asset}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paint:
        return cls(kind=data.get("kind", "solid"),
                   color=Rgba.from_dict(data.get("color", {})),
                   stops=[(float(t), Rgba.from_dict(c))
                          for t, c in data.get("stops", [])],
                   angle=float(data.get("angle", 0.0)),
                   asset=data.get("asset", ""))


@dataclass
class Stroke:
    color: Rgba = field(default_factory=lambda: Rgba(0.0, 0.0, 0.0, 1.0))
    width: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"color": self.color.to_dict(), "width": self.width}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stroke:
        return cls(color=Rgba.from_dict(data.get("color", {})),
                   width=float(data.get("width", 1.0)))


@dataclass
class Transform:
    """Position/rotation/scale/skew/opacity, upstream's element transform."""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0  # degrees
    scale_x: float = 1.0
    scale_y: float = 1.0
    skew_x: float = 0.0
    skew_y: float = 0.0
    opacity: float = 1.0

    def to_affine(self) -> Affine:
        m = Affine.translate(self.x, self.y)
        m = m.compose(Affine.rotate(self.rotation))
        m = m.compose(Affine.skew(self.skew_x, self.skew_y))
        m = m.compose(Affine.scale(self.scale_x, self.scale_y))
        return m

    @classmethod
    def lerp(cls, a: Transform, b: Transform, t: float,
             keep_area: bool = False) -> Transform:
        def mix(u: float, v: float) -> float:
            return u + (v - u) * t

        if keep_area:
            # Geometric scale move: the area stays constant on every frame.
            sx = a.scale_x * (b.scale_x / a.scale_x) ** t \
                if a.scale_x > 0 and b.scale_x > 0 else mix(a.scale_x,
                                                            b.scale_x)
            area = a.scale_x * a.scale_y
            sy = area / sx if sx else mix(a.scale_y, b.scale_y)
        else:
            sx, sy = mix(a.scale_x, b.scale_x), mix(a.scale_y, b.scale_y)
        return cls(x=mix(a.x, b.x), y=mix(a.y, b.y),
                   rotation=mix(a.rotation, b.rotation),
                   scale_x=sx, scale_y=sy,
                   skew_x=mix(a.skew_x, b.skew_x),
                   skew_y=mix(a.skew_y, b.skew_y),
                   opacity=mix(a.opacity, b.opacity))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "rotation": self.rotation,
                "scale_x": self.scale_x, "scale_y": self.scale_y,
                "skew_x": self.skew_x, "skew_y": self.skew_y,
                "opacity": self.opacity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transform:
        return cls(x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)),
                   rotation=float(data.get("rotation", 0.0)),
                   scale_x=float(data.get("scale_x", 1.0)),
                   scale_y=float(data.get("scale_y", 1.0)),
                   skew_x=float(data.get("skew_x", 0.0)),
                   skew_y=float(data.get("skew_y", 0.0)),
                   opacity=float(data.get("opacity", 1.0)))


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

class Element:
    """One drawable: shape, symbol instance, text, bitmap, or group."""

    def __init__(self, kind: str, *, element_id: int | None = None,
                 name: str = "") -> None:
        self.id = element_id if element_id is not None else _new_id()
        self.kind = kind
        self.name = name
        self.transform = Transform()
        self.path: Path | None = None
        self.fill: Paint | None = None
        self.stroke: Stroke | None = None
        self.symbol_id: int | None = None
        self.text: str = ""
        self.font: str = ""
        self.font_size: float = 24.0
        self.text_color: Rgba = Rgba(0.0, 0.0, 0.0, 1.0)
        self.asset: str = ""
        self.width: float = 0.0
        self.height: float = 0.0
        self.children: list[Element] = []

    # -- factories -----------------------------------------------------
    @classmethod
    def rect(cls, x: float, y: float, w: float, h: float,
             fill: str = "#000000", stroke: str | None = None,
             stroke_width: float = 1.0) -> Element:
        el = cls("shape")
        el.path = Path.rect(x, y, w, h)
        el.fill = Paint.solid(fill)
        if stroke is not None:
            el.stroke = Stroke(Rgba.from_hex(stroke), stroke_width)
        return el

    @classmethod
    def ellipse(cls, cx: float, cy: float, rx: float, ry: float,
                fill: str = "#000000") -> Element:
        el = cls("shape")
        el.path = Path.ellipse(cx, cy, rx, ry)
        el.fill = Paint.solid(fill)
        return el

    @classmethod
    def path(cls, svg: str, fill: str = "#000000",
             stroke: str | None = None,
             stroke_width: float = 1.0) -> Element:
        el = cls("shape")
        el.path = Path.from_svg(svg)
        el.fill = Paint.solid(fill)
        if stroke is not None:
            el.stroke = Stroke(Rgba.from_hex(stroke), stroke_width)
        return el

    @classmethod
    def text(cls, content: str, x: float = 0.0, y: float = 0.0,
             font_size: float = 24.0, color: str = "#000000",
             font: str = "") -> Element:
        el = cls("text")
        el.text = content
        el.font_size = float(font_size)
        el.text_color = Rgba.from_hex(color)
        el.font = font
        el.transform = Transform(x=x, y=y)
        return el

    @classmethod
    def bitmap(cls, asset: str, x: float, y: float, w: float,
               h: float) -> Element:
        el = cls("bitmap")
        el.asset = asset
        el.width = float(w)
        el.height = float(h)
        el.transform = Transform(x=x, y=y)
        return el

    @classmethod
    def symbol(cls, symbol_id: int,
               transform: Transform | None = None) -> Element:
        el = cls("symbol")
        el.symbol_id = int(symbol_id)
        if transform is not None:
            el.transform = transform
        return el

    @classmethod
    def group(cls, children: list[Element]) -> Element:
        el = cls("group")
        el.children = list(children)
        return el

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "name": self.name,
            "transform": self.transform.to_dict(),
            "path": self.path.to_dict() if self.path else None,
            "fill": self.fill.to_dict() if self.fill else None,
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "symbol_id": self.symbol_id,
            "text": self.text, "font": self.font,
            "font_size": self.font_size,
            "text_color": self.text_color.to_dict(),
            "asset": self.asset, "width": self.width,
            "height": self.height,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Element:
        el = cls(data.get("kind", "shape"),
                 element_id=int(data.get("id", 0)) or None,
                 name=data.get("name", ""))
        el.transform = Transform.from_dict(data.get("transform", {}))
        if data.get("path"):
            el.path = Path.from_dict(data["path"])
        if data.get("fill"):
            el.fill = Paint.from_dict(data["fill"])
        if data.get("stroke"):
            el.stroke = Stroke.from_dict(data["stroke"])
        el.symbol_id = data.get("symbol_id")
        el.text = data.get("text", "")
        el.font = data.get("font", "")
        el.font_size = float(data.get("font_size", 24.0))
        el.text_color = Rgba.from_dict(data.get("text_color", {}))
        el.asset = data.get("asset", "")
        el.width = float(data.get("width", 0.0))
        el.height = float(data.get("height", 0.0))
        el.children = [Element.from_dict(c)
                       for c in data.get("children", [])]
        return el

    def clone(self, new_id: bool = False) -> Element:
        data = self.to_dict()
        if new_id:
            data["id"] = _new_id()
            for child in data["children"]:
                child["id"] = _new_id()
        return Element.from_dict(data)


# ---------------------------------------------------------------------------
# Tweens
# ---------------------------------------------------------------------------

class Tween:
    """A classic or shape tween from one keyframe to the next.

    ``groups`` maps an ease group (``position``/``rotation``/``scale``/
    ``color``/``filters``) to its own easing; groups without one use
    the tween's easing (upstream ADR 0081).
    """

    def __init__(self, easing: Easing | None = None,
                 kind: str = "classic", orient: bool = False,
                 groups: dict[str, Easing] | None = None,
                 keep_area: bool = False) -> None:
        if kind not in TWEEN_KINDS:
            raise AnimationError(f"unknown tween kind {kind!r}")
        self.easing = easing or Easing.named("linear")
        self.kind = kind
        self.orient = bool(orient)
        self.groups = dict(groups or {})
        for group in self.groups:
            if group not in EASE_GROUPS:
                raise AnimationError(f"unknown ease group {group!r}")
        self.keep_area = bool(keep_area)

    def ease_for(self, group: str) -> Easing:
        return self.groups.get(group, self.easing)

    def to_dict(self) -> dict[str, Any]:
        return {"easing": self.easing.to_dict(), "kind": self.kind,
                "orient": self.orient,
                "groups": {g: e.to_dict()
                           for g, e in self.groups.items()},
                "keep_area": self.keep_area}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tween:
        return cls(easing=Easing.from_dict(data.get("easing", {})),
                   kind=data.get("kind", "classic"),
                   orient=bool(data.get("orient", False)),
                   groups={g: Easing.from_dict(e)
                           for g, e in data.get("groups", {}).items()},
                   keep_area=bool(data.get("keep_area", False)))


@dataclass
class MotionTrack:
    """One animated property of a motion tween: (frame, value, easing)."""
    property: str  # x | y | rotation | scale_x | scale_y | opacity
    keys: list[tuple[int, float, Easing]] = field(default_factory=list)

    def value_at(self, frame: float) -> float:
        if not self.keys:
            raise AnimationError(
                f"motion track {self.property!r} has no keys")
        keys = sorted(self.keys)
        if frame <= keys[0][0]:
            return keys[0][1]
        if frame >= keys[-1][0]:
            return keys[-1][1]
        for (f0, v0, ease), (f1, v1, _) in itertools.pairwise(keys):
            if f0 <= frame <= f1:
                local = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
                return v0 + (v1 - v0) * ease.apply(local)
        return keys[-1][1]

    def to_dict(self) -> dict[str, Any]:
        return {"property": self.property,
                "keys": [[f, v, e.to_dict()] for f, v, e in self.keys]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionTrack:
        return cls(property=data["property"],
                   keys=[(int(f), float(v), Easing.from_dict(e))
                         for f, v, e in data.get("keys", [])])


class MotionTween:
    """A motion tween (upstream ADR 0016): per-property tracks plus an
    optional motion path the object follows, with orient-to-path."""

    def __init__(self, target: int, tracks: list[MotionTrack] | None = None,
                 path: list[tuple[float, float]] | None = None,
                 orient: bool = False,
                 pins: list[tuple[int, float]] | None = None,
                 keep_area: bool = False) -> None:
        self.target = int(target)
        self.tracks = list(tracks or [])
        self.path = [(float(x), float(y)) for x, y in (path or [])]
        self.orient = bool(orient)
        self.pins = sorted((int(f), float(p)) for f, p in (pins or []))
        self.keep_area = bool(keep_area)

    def position_at(self, frame: float, start: float,
                    end: float) -> tuple[float, float, float]:
        """(x, y, rotation) at ``frame`` inside the span [start, end]."""
        if self.path:
            total = len(self.path) - 1
            if total <= 0:
                return (self.path[0][0], self.path[0][1], 0.0)
            # Pins anchor frames to path positions; otherwise linear time.
            pos = (frame - start) / max(1.0, end - start)
            for f, p in self.pins:
                if frame <= f:
                    pos = p
                    break
            pos = max(0.0, min(1.0, pos))
            seg = min(total - 1, int(pos * total))
            local = pos * total - seg
            x0, y0 = self.path[seg]
            x1, y1 = self.path[seg + 1]
            x = x0 + (x1 - x0) * local
            y = y0 + (y1 - y0) * local
            rot = 0.0
            if self.orient:
                rot = math.degrees(math.atan2(y1 - y0, x1 - x0))
            return (x, y, rot)
        x = self._track_value("x", frame)
        y = self._track_value("y", frame)
        rot = self._track_value("rotation", frame)
        return (x, y, rot)

    def _track_value(self, prop: str, frame: float) -> float:
        for track in self.tracks:
            if track.property == prop:
                return track.value_at(frame)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target,
                "tracks": [t.to_dict() for t in self.tracks],
                "path": [[x, y] for x, y in self.path],
                "orient": self.orient,
                "pins": [[f, p] for f, p in self.pins],
                "keep_area": self.keep_area}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionTween:
        return cls(target=int(data["target"]),
                   tracks=[MotionTrack.from_dict(t)
                           for t in data.get("tracks", [])],
                   path=[(float(x), float(y))
                         for x, y in data.get("path", [])],
                   orient=bool(data.get("orient", False)),
                   pins=[(int(f), float(p))
                         for f, p in data.get("pins", [])],
                   keep_area=bool(data.get("keep_area", False)))


# ---------------------------------------------------------------------------
# Keyframes, layers, timelines
# ---------------------------------------------------------------------------

class Keyframe:
    """A keyframe and the span of frames it covers (upstream ``Keyframe``)."""

    def __init__(self, start: int, duration: int,
                 label: str = "") -> None:
        if duration < 1:
            raise AnimationError("a keyframe's duration is at least 1")
        if start < 0:
            raise AnimationError("a keyframe cannot start before frame 0")
        self.start = int(start)
        self.duration = int(duration)
        self.elements: list[Element] = []
        self.tween: Tween | None = None
        self.motion: MotionTween | None = None
        self.label = label
        self.camera: dict[str, float] | None = None

    def end(self) -> int:
        return self.start + self.duration

    def covers(self, frame: int) -> bool:
        return self.start <= frame < self.end()

    def add_element(self, element: Element) -> Element:
        self.elements.append(element)
        return element

    def tween_to(self, _next: Keyframe,
                 easing: Easing | None = None,
                 kind: str = "classic", orient: bool = False,
                 groups: dict[str, Easing] | None = None,
                 keep_area: bool = False) -> Tween:
        self.tween = Tween(easing=easing, kind=kind, orient=orient,
                           groups=groups, keep_area=keep_area)
        return self.tween

    def tween_motion(self, _next: Keyframe,
                     tracks: list[MotionTrack] | None = None,
                     path: list[tuple[float, float]] | None = None,
                     orient: bool = False,
                     pins: list[tuple[int, float]] | None = None,
                     keep_area: bool = False) -> MotionTween:
        target = self.elements[0].id if self.elements else 0
        self.motion = MotionTween(target, tracks=tracks, path=path,
                                  orient=orient, pins=pins,
                                  keep_area=keep_area)
        return self.motion

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "duration": self.duration,
                "elements": [e.to_dict() for e in self.elements],
                "tween": self.tween.to_dict() if self.tween else None,
                "motion": self.motion.to_dict() if self.motion else None,
                "label": self.label, "camera": self.camera}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keyframe:
        kf = cls(int(data["start"]), int(data["duration"]),
                 label=data.get("label", ""))
        kf.elements = [Element.from_dict(e)
                       for e in data.get("elements", [])]
        if data.get("tween"):
            kf.tween = Tween.from_dict(data["tween"])
        if data.get("motion"):
            kf.motion = MotionTween.from_dict(data["motion"])
        kf.camera = data.get("camera")
        return kf


class Layer:
    """A timeline layer (upstream ``Layer``)."""

    def __init__(self, name: str, kind: str = "normal",
                 layer_id: int | None = None) -> None:
        if kind not in LAYER_KINDS:
            raise AnimationError(
                f"unknown layer kind {kind!r}; pick from {LAYER_KINDS}")
        self.id = layer_id if layer_id is not None else _new_id()
        self.name = name
        self.kind = kind
        self.parent: int | None = None
        self.visible = True
        self.locked = False
        self.opacity = 1.0
        self.depth: float = 0.0
        self.keyframes: list[Keyframe] = []

    def add_keyframe(self, start: int, duration: int,
                     label: str = "") -> Keyframe:
        end = start + duration
        for kf in self.keyframes:
            if start < kf.end() and end > kf.start:
                raise AnimationError(
                    f"keyframe {start}..{end} overlaps keyframe "
                    f"{kf.start}..{kf.end()} on layer {self.name!r}")
        kf = Keyframe(start, duration, label=label)
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.start)
        return kf

    def keyframe_at(self, frame: int) -> Keyframe | None:
        for kf in self.keyframes:
            if kf.covers(frame):
                return kf
        return None

    def frame_count(self) -> int:
        return max((kf.end() for kf in self.keyframes), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "parent": self.parent, "visible": self.visible,
                "locked": self.locked, "opacity": self.opacity,
                "depth": self.depth,
                "keyframes": [k.to_dict() for k in self.keyframes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        layer = cls(data.get("name", ""), kind=data.get("kind", "normal"),
                    layer_id=int(data.get("id", 0)) or None)
        layer.parent = data.get("parent")
        layer.visible = bool(data.get("visible", True))
        layer.locked = bool(data.get("locked", False))
        layer.opacity = float(data.get("opacity", 1.0))
        layer.depth = float(data.get("depth", 0.0))
        layer.keyframes = [Keyframe.from_dict(k)
                           for k in data.get("keyframes", [])]
        return layer


class Timeline:
    def __init__(self, fps: float = 24.0) -> None:
        self.fps = float(fps)
        self.layers: list[Layer] = []

    def add_layer(self, name: str, kind: str = "normal") -> Layer:
        if kind == "armature":
            raise AnimationError(
                "armature layers (bone rigs, ADR 0043) are not ported "
                "yet — add a normal layer instead")
        if kind == "camera" and any(ly.kind == "camera"
                                    for ly in self.layers):
            raise AnimationError(
                "a timeline holds at most one camera layer")
        layer = Layer(name, kind=kind)
        self.layers.append(layer)
        return layer

    def frame_count(self) -> int:
        return max((ly.frame_count() for ly in self.layers), default=0)

    def layer_by_id(self, layer_id: int) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise AnimationError(f"no layer with id {layer_id}")

    def to_dict(self) -> dict[str, Any]:
        return {"fps": self.fps,
                "layers": [ly.to_dict() for ly in self.layers]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Timeline:
        tl = cls(fps=float(data.get("fps", 24.0)))
        tl.layers = [Layer.from_dict(ly)
                     for ly in data.get("layers", [])]
        return tl


# ---------------------------------------------------------------------------
# Stage, scene, symbol, document
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    width: int = 640
    height: int = 360
    background: Rgba = field(default_factory=lambda: Rgba(1.0, 1.0, 1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height,
                "background": self.background.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage:
        return cls(width=int(data.get("width", 640)),
                   height=int(data.get("height", 360)),
                   background=Rgba.from_dict(
                       data.get("background", {})))


class Scene:
    def __init__(self, name: str, stage: Stage | None = None,
                 fps: float = 24.0) -> None:
        self.name = name
        self.stage = stage or Stage()
        self.timeline = Timeline(fps=fps)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage.to_dict(),
                "timeline": self.timeline.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        scene = cls(data.get("name", ""),
                    stage=Stage.from_dict(data.get("stage", {})),
                    fps=float(data.get("timeline", {}).get("fps", 24.0)))
        scene.timeline = Timeline.from_dict(data.get("timeline", {}))
        return scene


class Symbol:
    """A library symbol: a nested timeline played by its instances."""

    def __init__(self, name: str, kind: str = "graphic",
                 symbol_id: int | None = None,
                 loop_mode: str = "loop") -> None:
        if kind not in SYMBOL_KINDS:
            raise AnimationError(f"unknown symbol kind {kind!r}")
        if loop_mode not in LOOP_MODES:
            raise AnimationError(f"unknown loop mode {loop_mode!r}")
        self.id = symbol_id if symbol_id is not None else _new_id()
        self.name = name
        self.kind = kind
        self.loop_mode = loop_mode
        self.timeline = Timeline()

    def local_frame(self, offset: int) -> int:
        total = max(1, self.timeline.frame_count())
        if self.loop_mode == "loop":
            return offset % total
        if self.loop_mode == "play-once":
            return min(offset, total - 1)
        return min(0, total - 1)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "loop_mode": self.loop_mode,
                "timeline": self.timeline.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Symbol:
        sym = cls(data.get("name", ""), kind=data.get("kind", "graphic"),
                  symbol_id=int(data.get("id", 0)) or None,
                  loop_mode=data.get("loop_mode", "loop"))
        sym.timeline = Timeline.from_dict(data.get("timeline", {}))
        return sym


class AnimationDocument:
    """One animation: scenes + a symbol library (upstream ``Document``)."""

    FORMAT = "acraft"
    VERSION = 1

    def __init__(self, name: str, fps: float = 24.0) -> None:
        self.name = name
        self.fps = float(fps)
        self.scenes: list[Scene] = []
        self.symbols: list[Symbol] = []
        self.assets: dict[str, bytes] = {}

    @classmethod
    def create(cls, name: str, width: int = 640, height: int = 360,
               fps: float = 24.0) -> AnimationDocument:
        doc = cls(name, fps=fps)
        doc.add_scene("Scene 1", width=width, height=height)
        return doc

    def add_scene(self, name: str, width: int = 640,
                  height: int = 360) -> Scene:
        scene = Scene(name, stage=Stage(width, height),
                      fps=self.fps)
        self.scenes.append(scene)
        return scene

    def scene(self, name: str | None = None) -> Scene:
        if name is None:
            if not self.scenes:
                raise AnimationError("document has no scenes")
            return self.scenes[0]
        for scene in self.scenes:
            if scene.name == name:
                return scene
        raise AnimationError(f"no scene named {name!r}")

    def add_symbol(self, name: str, kind: str = "graphic",
                   loop_mode: str = "loop") -> Symbol:
        sym = Symbol(name, kind=kind, loop_mode=loop_mode)
        self.symbols.append(sym)
        return sym

    def symbol_by_id(self, symbol_id: int) -> Symbol:
        for sym in self.symbols:
            if sym.id == symbol_id:
                return sym
        raise AnimationError(f"no symbol with id {symbol_id}")

    # -- codec -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"format": self.FORMAT, "version": self.VERSION,
                "name": self.name, "fps": self.fps,
                "scenes": [s.to_dict() for s in self.scenes],
                "symbols": [s.to_dict() for s in self.symbols]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnimationDocument:
        if data.get("format") != cls.FORMAT:
            raise AnimationError(
                f"not an .acraft document (format "
                f"{data.get('format')!r})")
        doc = cls(data.get("name", ""), fps=float(data.get("fps", 24.0)))
        doc.scenes = [Scene.from_dict(s)
                      for s in data.get("scenes", [])]
        doc.symbols = [Symbol.from_dict(s)
                       for s in data.get("symbols", [])]
        return doc

    def to_bytes(self) -> bytes:
        payload = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            info = zipfile.ZipInfo("document.json",
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)
            for name in sorted(self.assets):
                info = zipfile.ZipInfo(f"assets/{name}",
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, self.assets[name])
        return buf.getvalue()

    @classmethod
    def parse(cls, raw: bytes) -> AnimationDocument:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                if "document.json" not in names:
                    raise AnimationError(
                        ".acraft is missing document.json")
                payload = zf.read("document.json")
                doc = cls.from_dict(json.loads(payload.decode("utf-8")))
                for name in names:
                    if name.startswith("assets/"):
                        doc.assets[name[len("assets/"):]] = zf.read(name)
                return doc
        except AnimationError:
            raise
        except (zipfile.BadZipFile, ValueError, KeyError,
                UnicodeDecodeError) as exc:
            raise AnimationError(f"cannot open .acraft: {exc}") from exc

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def fidelity_report(self, other: AnimationDocument) -> dict[str, Any]:
        """What changed between this document and ``other`` — by id."""
        def layers(doc: AnimationDocument) -> list[Layer]:
            out: list[Layer] = []
            for scene in doc.scenes:
                out.extend(scene.timeline.layers)
            return out

        def keyframes(doc: AnimationDocument) -> dict[int, Keyframe]:
            out: dict[int, Keyframe] = {}
            for layer in layers(doc):
                for kf in layer.keyframes:
                    out[id(kf)] = kf
            return out

        changed_layers: list[str] = []
        changed_keyframes: list[int] = []
        changed_elements: list[int] = []
        unchanged_elements = 0
        mine = {ly.id: ly for ly in layers(self)}
        theirs = {ly.id: ly for ly in layers(other)}
        for lid, layer in mine.items():
            other_layer = theirs.get(lid)
            if other_layer is None or layer.to_dict() != other_layer.to_dict():
                changed_layers.append(layer.name)
        for kf in keyframes(self).values():
            found = None
            for other_kf in keyframes(other).values():
                if (other_kf.start == kf.start
                        and other_kf.duration == kf.duration):
                    found = other_kf
                    break
            if found is None or kf.to_dict() != found.to_dict():
                changed_keyframes.append(kf.start)
            for el in kf.elements:
                match = None
                if found is not None:
                    for other_el in found.elements:
                        if other_el.id == el.id:
                            match = other_el
                            break
                if match is None or el.to_dict() != match.to_dict():
                    changed_elements.append(el.id)
                else:
                    unchanged_elements += 1
        return {"changed_layers": sorted(set(changed_layers)),
                "changed_keyframes": sorted(set(changed_keyframes)),
                "changed_elements": sorted(set(changed_elements)),
                "unchanged_elements": unchanged_elements}
