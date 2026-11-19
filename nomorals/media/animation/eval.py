"""Timeline evaluation — port of AniCraft ``crates/doc/src/eval.rs``.

``evaluate(document, frame)`` is pure and deterministic: it resolves
every visible layer's keyframe at ``frame``, interpolates classic
tweens (per ease group) and shape tweens (same-structure paths morph;
a structure mismatch keeps the start shape and is named in
``DisplayList.notes``), follows motion tweens and motion paths,
applies the camera layer and layer depth (parallax), resolves symbol
instances (graphic/movieclip by loop mode, button as its up state),
and applies mask layers. The result is a back-to-front
:class:`DisplayList` of fully resolved display items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import AnimationError
from .geom import Affine, Path, Vec2
from .model import (
    MAX_LAYER_DEPTH,
    MAX_SYMBOL_DEPTH,
    AnimationDocument,
    Element,
    Keyframe,
    Layer,
    Scene,
    Transform,
)

__all__ = ["DisplayItem", "DisplayList", "evaluate"]


@dataclass
class DisplayItem:
    """One shape to draw, fully resolved to stage space."""
    points: list[list[tuple[float, float]]] = field(default_factory=list)
    kind: str = "shape"  # shape | text | bitmap | group
    fill: dict[str, Any] | None = None
    stroke: dict[str, Any] | None = None
    opacity: float = 1.0
    transform: Transform = field(default_factory=Transform)
    layer: int = 0
    element: int = 0
    clip: int | None = None
    text: str = ""
    font_size: float = 24.0
    text_color: str = "#000000"
    asset: str = ""
    width: float = 0.0
    height: float = 0.0


@dataclass
class DisplayList:
    items: list[DisplayItem] = field(default_factory=list)
    clips: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frame: int = 0

    def __len__(self) -> int:
        return len(self.items)


def _morph_paths(a: Path, b: Path, t: float) -> Path | None:
    """Lerp two paths vertex by vertex; None when structures differ."""
    if len(a.segments) != len(b.segments):
        return None
    out = Path()
    for (op_a, pts_a), (op_b, pts_b) in zip(
            a.segments, b.segments, strict=True):
        if op_a != op_b or len(pts_a) != len(pts_b):
            return None
        merged = [Vec2(pa.x + (pb.x - pa.x) * t,
                       pa.y + (pb.y - pa.y) * t)
                  for pa, pb in zip(pts_a, pts_b, strict=True)]
        out.segments.append((op_a, merged))
    return out


def _interpolate_element(start: Element, end: Element, t: float,
                         tween, notes: list[str]) -> Element:
    """The element at tween progress ``t`` between two keyframes."""
    out = start.clone()
    if tween is None:
        return out
    ease_pos = tween.ease_for("position")
    ease_rot = tween.ease_for("rotation")
    ease_scale = tween.ease_for("scale")
    ease_color = tween.ease_for("color")
    tp = ease_pos.apply(t)
    tr = ease_rot.apply(t)
    ts = ease_scale.apply(t)
    tc = ease_color.apply(t)
    a, b = start.transform, end.transform
    out.transform = Transform(
        x=a.x + (b.x - a.x) * tp,
        y=a.y + (b.y - a.y) * tp,
        rotation=a.rotation + (b.rotation - a.rotation) * tr,
        scale_x=a.scale_x + (b.scale_x - a.scale_x) * ts
        if not tween.keep_area else
        (a.scale_x * (b.scale_x / a.scale_x) ** ts
         if a.scale_x > 0 and b.scale_x > 0
         else a.scale_x + (b.scale_x - a.scale_x) * ts),
        scale_y=a.scale_y + (b.scale_y - a.scale_y) * ts
        if not tween.keep_area else
        (a.scale_x * a.scale_y
         / (a.scale_x * (b.scale_x / a.scale_x) ** ts)
         if a.scale_x > 0 and b.scale_x > 0
         else a.scale_y + (b.scale_y - a.scale_y) * ts),
        skew_x=a.skew_x + (b.skew_x - a.skew_x) * tr,
        skew_y=a.skew_y + (b.skew_y - a.skew_y) * tr,
        opacity=a.opacity + (b.opacity - a.opacity) * tc)
    if tween.kind == "shape" and start.path is not None \
            and end.path is not None:
        morphed = _morph_paths(start.path, end.path, tp)
        if morphed is None:
            notes.append(
                f"shape tween on element {start.id}: paths have "
                "different structures, morph refused — start shape kept")
        else:
            out.path = morphed
    if start.fill is not None and end.fill is not None \
            and start.fill.kind == "solid" and end.fill.kind == "solid":
        out.fill = start.fill
        out.fill = type(start.fill)(
            kind="solid",
            color=type(start.fill.color)(
                r=start.fill.color.r
                + (end.fill.color.r - start.fill.color.r) * tc,
                g=start.fill.color.g
                + (end.fill.color.g - start.fill.color.g) * tc,
                b=start.fill.color.b
                + (end.fill.color.b - start.fill.color.b) * tc,
                a=start.fill.color.a
                + (end.fill.color.a - start.fill.color.a) * tc))
    return out


def _camera_at(layer: Layer, frame: int) -> dict[str, float] | None:
    kf = layer.keyframe_at(frame)
    if kf is None:
        # Hold the last keyframe's camera after its span ends.
        if layer.keyframes and frame >= layer.keyframes[-1].end():
            return layer.keyframes[-1].camera
        return None
    if kf.camera is None:
        return None
    idx = layer.keyframes.index(kf)
    if kf.tween is None or idx + 1 >= len(layer.keyframes):
        return dict(kf.camera)
    nxt = layer.keyframes[idx + 1]
    if nxt.camera is None:
        return dict(kf.camera)
    local = (frame - kf.start) / max(1, kf.duration)
    eased = kf.tween.easing.apply(local)
    return {k: v + (nxt.camera.get(k, v) - v) * eased
            for k, v in kf.camera.items()}


def _resolve_keyframe(layer: Layer, frame: int) -> tuple[Keyframe, float, Keyframe | None]:
    """(keyframe, local progress, next keyframe) for ``frame``."""
    kf = layer.keyframe_at(frame)
    if kf is not None:
        idx = layer.keyframes.index(kf)
        nxt = layer.keyframes[idx + 1] \
            if idx + 1 < len(layer.keyframes) else None
        local = (frame - kf.start) / max(1, kf.duration)
        return kf, local, nxt
    if layer.keyframes and frame >= layer.keyframes[-1].end():
        last = layer.keyframes[-1]
        return last, 1.0, None
    raise AnimationError("no keyframe")


def _flatten_element(element: Element, transform: Affine,
                     opacity: float, layer_id: int,
                     clip: int | None, notes: list[str],
                     doc: AnimationDocument, frame: int,
                     depth: int, items: list[DisplayItem]) -> None:
    if element.kind == "group":
        for child in element.children:
            _flatten_element(child, transform, opacity, layer_id,
                             clip, notes, doc, frame, depth, items)
        return
    if element.kind == "symbol":
        if depth >= MAX_SYMBOL_DEPTH:
            notes.append(
                f"symbol {element.symbol_id}: nested deeper than "
                f"{MAX_SYMBOL_DEPTH}, not drawn")
            return
        try:
            symbol = doc.symbol_by_id(element.symbol_id)
        except AnimationError:
            notes.append(
                f"symbol instance {element.id}: no symbol "
                f"{element.symbol_id} in the library, not drawn")
            return
        if symbol.kind == "button":
            notes.append(
                f"symbol {symbol.name!r}: button shows its up state only")
        local = symbol.local_frame(frame)
        inner = _evaluate_timeline(
            doc, symbol.timeline, local, notes, depth + 1)
        outer = transform.compose(element.transform.to_affine())
        for item in inner:
            _apply_transform(item, outer, opacity, notes)
            item.layer = layer_id
            item.element = element.id
            if clip is not None:
                item.clip = clip
            items.append(item)
        return
    item = DisplayItem(kind=element.kind, layer=layer_id,
                       element=element.id, clip=clip)
    item.transform = element.transform
    if element.kind == "shape" and element.path is not None:
        contours = element.path.flatten()
        m = transform.compose(element.transform.to_affine())
        item.points = [[(m.apply(Vec2(x, y)).x, m.apply(Vec2(x, y)).y)
                        for x, y in contour]
                       for contour in contours]
        if element.fill is not None:
            item.fill = element.fill.to_dict()
        if element.stroke is not None:
            item.stroke = element.stroke.to_dict()
    elif element.kind == "text":
        m = transform.compose(element.transform.to_affine())
        origin = m.apply(Vec2(0.0, 0.0))
        item.points = [[(origin.x, origin.y)]]
        item.text = element.text
        item.font_size = element.font_size
        item.text_color = element.text_color.to_hex()
    elif element.kind == "bitmap":
        m = transform.compose(element.transform.to_affine())
        origin = m.apply(Vec2(0.0, 0.0))
        item.points = [[(origin.x, origin.y)]]
        item.asset = element.asset
        item.width = element.width
        item.height = element.height
    item.opacity = opacity * element.transform.opacity
    items.append(item)


def _apply_transform(item: DisplayItem, outer: Affine, opacity: float,
                     notes: list[str]) -> None:
    item.points = [[(outer.apply(Vec2(x, y)).x,
                     outer.apply(Vec2(x, y)).y)
                    for x, y in contour]
                   for contour in item.points]
    item.opacity *= opacity
    # Decompose the composed affine back into a Transform record;
    # the points above already carry the exact geometry.
    inner = item.transform.to_affine()
    composed = outer.compose(inner)
    item.transform = Transform(
        x=composed.e, y=composed.f,
        rotation=math.degrees(math.atan2(composed.b, composed.a)),
        scale_x=math.hypot(composed.a, composed.b),
        scale_y=math.hypot(composed.c, composed.d),
        opacity=item.opacity)


def _evaluate_timeline(doc: AnimationDocument, timeline,
                       frame: int, notes: list[str],
                       depth: int) -> list[DisplayItem]:
    """Evaluate one timeline (scene or symbol) at ``frame``."""
    # Camera first: it views every other layer.
    camera: dict[str, float] | None = None
    for layer in timeline.layers:
        if layer.kind == "camera" and layer.visible:
            camera = _camera_at(layer, frame)
            break
    by_id = {ly.id: ly for ly in timeline.layers}
    # Folder opacity folds into its children.
    folder_opacity: dict[int, float] = {}
    for layer in timeline.layers:
        op = 1.0
        parent = layer.parent
        seen = 0
        while parent is not None and seen < MAX_LAYER_DEPTH:
            folder = by_id.get(parent)
            if folder is None:
                break
            op *= folder.opacity
            if not folder.visible:
                op = 0.0
            parent = folder.parent
            seen += 1
        folder_opacity[layer.id] = op

    items: list[DisplayItem] = []
    # Masks are declared top-down: a mask layer reveals the normal
    # layers below it until the next non-normal layer. Pre-scan
    # top-to-bottom so the back-to-front walk below knows each
    # layer's clip before it draws it.
    masked_by: dict[int, int] = {}
    active: int | None = None
    for layer in timeline.layers:
        if layer.kind == "mask":
            try:
                kf, _local, _nxt = _resolve_keyframe(layer, frame)
            except AnimationError:
                active = None
                continue
            clip_shapes: list[list[list[tuple[float, float]]]] = []
            for el in kf.elements:
                if el.path is not None:
                    m = el.transform.to_affine()
                    clip_shapes.append(
                        [[(m.apply(Vec2(x, y)).x, m.apply(Vec2(x, y)).y)
                          for x, y in contour]
                         for contour in el.path.flatten()])
            active = len(_clips_holder[0])
            _clips_holder[0].append({"shapes": clip_shapes,
                                     "mode": "coverage"})
            continue
        if layer.kind == "normal" and active is not None:
            masked_by[layer.id] = active
        elif layer.kind != "normal":
            active = None
    # Layers are top-first; draw back-to-front, so walk in reverse.
    for layer in reversed(timeline.layers):
        if layer.kind in ("guide", "folder", "camera", "mask"):
            continue
        if not layer.visible or folder_opacity.get(layer.id, 1.0) <= 0.0:
            continue
        try:
            kf, local, nxt = _resolve_keyframe(layer, frame)
        except AnimationError:
            continue
        layer_opacity = layer.opacity * folder_opacity.get(layer.id, 1.0)
        # Depth parallax: the camera's pan moves a deeper layer less.
        view = Affine.identity()
        if camera is not None:
            factor = 1.0 / (1.0 + max(0.0, layer.depth) / 100.0)
            cx = camera.get("x", 0.0) * factor
            cy = camera.get("y", 0.0) * factor
            zoom = camera.get("zoom", 1.0)
            rot = camera.get("rotation", 0.0)
            view = Affine.translate(-cx * zoom, -cy * zoom)
            view = Affine.scale(zoom, zoom).compose(view)
            if rot:
                view = Affine.rotate(-rot).compose(view)
        clip = masked_by.get(layer.id)
        for element in kf.elements:
            resolved = element
            if kf.tween is not None and nxt is not None:
                match = next((e for e in nxt.elements
                              if e.id == element.id), None)
                if match is not None:
                    resolved = _interpolate_element(
                        element, match, local, kf.tween, notes)
            elif kf.motion is not None and nxt is not None:
                resolved = _apply_motion(element, kf.motion, frame,
                                         kf, nxt, notes)
            _flatten_element(resolved, view, layer_opacity, layer.id,
                             clip, notes, doc, frame, depth, items)
    return items


# The clips list is shared with evaluate() through a holder so the
# recursive timeline walk can append mask clips.
_clips_holder: list[list[dict[str, Any]]] = [[]]


def _apply_motion(element: Element, motion, frame: int, kf: Keyframe,
                  nxt: Keyframe, notes: list[str]) -> Element:
    out = element.clone()
    x, y, rot = motion.position_at(frame, kf.start, nxt.start)
    out.transform = Transform(
        x=x, y=y, rotation=rot,
        scale_x=element.transform.scale_x,
        scale_y=element.transform.scale_y,
        skew_x=element.transform.skew_x,
        skew_y=element.transform.skew_y,
        opacity=element.transform.opacity)
    # Per-property tracks override the path position where present.
    for track in motion.tracks:
        if track.property == "x":
            out.transform.x = track.value_at(frame)
        elif track.property == "y":
            out.transform.y = track.value_at(frame)
        elif track.property == "rotation":
            out.transform.rotation = track.value_at(frame)
        elif track.property == "scale_x":
            out.transform.scale_x = track.value_at(frame)
        elif track.property == "scale_y":
            out.transform.scale_y = track.value_at(frame)
        elif track.property == "opacity":
            out.transform.opacity = track.value_at(frame)
    return out


def evaluate(doc: AnimationDocument, frame: int,
             scene: str | Scene | None = None) -> DisplayList:
    """(document, frame) -> DisplayList. Pure, deterministic."""
    if isinstance(scene, str):
        scene_obj = doc.scene(scene)
    elif scene is None:
        scene_obj = doc.scene()
    else:
        scene_obj = scene
    notes: list[str] = []
    _clips_holder[0] = []
    items = _evaluate_timeline(doc, scene_obj.timeline, int(frame),
                               notes, 0)
    dl = DisplayList(items=items, clips=_clips_holder[0], notes=notes,
                     frame=int(frame))
    return dl
