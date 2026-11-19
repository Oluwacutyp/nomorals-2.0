"""Typed texture editing commands (TextureCraft port, census 3A.15).

Upstream TextureCraft: everything is a command, undo is snapshots,
and a command never crashes on a bad parameter (upstream AGENTS.md
§3). Every method here validates its inputs first (the per-step
guard — a bad layer id, channel, blend mode or geometry raises
:class:`TextureError` naming the step, before anything is
mutated), snapshots the document, applies the edit, and marks the
layer touched. ``undo`` restores the last snapshot (capped at 50,
like the wiki organ's revision cap).
"""
from __future__ import annotations

from collections.abc import Sequence

from .blend import BlendMode
from .compose import composite
from .dilate import NONE, DilationMap
from .errors import TextureError, TextureErrorKind
from .filters import SUPPORTED_FILTERS, apply_filter
from .model import Channel, Layer, TextureDocument, TextureSet
from .normal import height_to_normal
from .paint import Brush, fill_pattern, fill_rect, stroke
from .patterns import Procedural, ProceduralKind
from .tiles import TileImage

_MAX_UNDO = 50


class TextureEditor:
    """Typed commands over a :class:`TextureDocument`."""

    def __init__(self, doc: TextureDocument) -> None:
        self.doc = doc
        self._undo_stack: list[dict] = []

    # -- snapshot / undo ------------------------------------------------------
    def _snapshot(self) -> dict:
        state: dict = {"sets": []}
        for tset in self.doc.sets:
            layers = []
            for layer in tset.layers:
                layers.append({
                    "meta": layer.to_dict(),
                    "images": {ch: img.clone()
                               for ch, img in layer._images.items()},
                    "mask": (layer._mask_image.clone()
                             if layer._mask_image is not None else None),
                    "width": layer._width, "height": layer._height,
                })
            state["sets"].append({
                "name": tset.name, "width": tset.width,
                "height": tset.height, "channels": list(tset.channels),
                "layers": layers,
            })
        state["title"] = self.doc.title
        return state

    def _restore(self, state: dict) -> None:
        self.doc.title = state["title"]
        self.doc.sets = []
        for sdata in state["sets"]:
            tset = TextureSet(sdata["name"], sdata["width"], sdata["height"],
                              list(sdata["channels"]), [])
            for ldata in sdata["layers"]:
                layer = Layer.from_dict(ldata["meta"])
                layer._images = dict(ldata["images"])
                layer._mask_image = ldata["mask"]
                layer._width = ldata["width"]
                layer._height = ldata["height"]
                tset.layers.append(layer)
            self.doc.sets.append(tset)
        self.doc.mark_dirty()

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > _MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore the last snapshot. Returns False when the stack is
        empty."""
        if not self._undo_stack:
            return False
        self._restore(self._undo_stack.pop())
        return True

    # -- guards ---------------------------------------------------------------
    def _set(self, set_name: str) -> TextureSet:
        for s in self.doc.sets:
            if s.name == set_name:
                return s
        raise TextureError(TextureErrorKind.NOT_FOUND,
                           f"no texture set {set_name!r}")

    def _layer(self, tset: TextureSet, layer_id: int) -> Layer:
        return tset.layer_by_id(layer_id)

    def _channel(self, name: str) -> Channel:
        ch = Channel.parse(name)
        if ch is None:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"unknown channel {name!r}")
        return ch

    def _next_layer_id(self) -> int:
        ids = [layer.id for tset in self.doc.sets for layer in tset.layers]
        return max(ids) + 1 if ids else 1

    # -- layer management -----------------------------------------------------
    def add_layer(self, set_name: str, name: str, kind: str = "paint",
                  fill: dict | None = None,
                  channels: dict | None = None) -> int:
        """Add a layer to a set; returns the new layer id."""
        tset = self._set(set_name)
        if kind not in ("paint", "fill", "group", "instance"):
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"unknown layer kind {kind!r}")
        if kind == "fill" and fill is None:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               "a fill layer needs a fill definition")
        if kind == "fill" and fill is not None:
            fkind = fill.get("kind")
            if fkind not in ("uniform", "procedural"):
                raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                                   f"unknown fill kind {fkind!r}")
            if fkind == "procedural" and \
                    ProceduralKind.parse(str(fill.get("pattern", ""))) is None:
                raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                                   f"unknown pattern {fill.get('pattern')!r}")
        # Validate the channel blends BEFORE snapshotting.
        parsed_channels: dict[str, object] = {}
        if channels:
            for ch_name, cb in channels.items():
                self._channel(ch_name)
                mode = BlendMode.parse(str(cb.get("mode", "normal")))
                if mode is None:
                    raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                                       f"unknown blend mode {cb.get('mode')!r}")
                parsed_channels[ch_name] = _blend(
                    mode, float(cb.get("opacity", 1.0)))
        self._push_undo()
        layer = Layer(id=self._next_layer_id(), name=name, kind=kind,
                      fill=fill)
        layer._width = tset.width
        layer._height = tset.height
        for ch_name in tset.channels:
            layer.channels[ch_name] = _default_blend(ch_name, kind)
        layer.channels.update(parsed_channels)
        tset.layers.append(layer)
        self.doc.mark_touched(layer.id)
        self.doc.mark_dirty()
        return layer.id

    def remove_layer(self, set_name: str, layer_id: int) -> None:
        tset = self._set(set_name)
        self._layer(tset, layer_id)
        self._push_undo()
        tset.layers = [lay for lay in tset.layers if lay.id != layer_id]
        for layer in tset.layers:
            if layer_id in layer.children:
                layer.children.remove(layer_id)
        self.doc.mark_dirty()

    def rename_layer(self, set_name: str, layer_id: int, name: str) -> None:
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        if not name.strip():
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               "layer name must not be empty")
        self._push_undo()
        layer.name = name
        self.doc.mark_dirty()

    def set_visible(self, set_name: str, layer_id: int,
                    visible: bool) -> None:
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        self._push_undo()
        layer.visible = bool(visible)
        self.doc.mark_dirty()

    def move_layer(self, set_name: str, layer_id: int,
                   target: int | None = None) -> None:
        """Move a layer into a group (``target`` = the group layer
        id), or to the top of the stack when ``target`` is None."""
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        if target is not None:
            group = self._layer(tset, target)
            if group.kind != "group":
                raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                                   f"layer {target} is not a group")
            if layer_id == target:
                raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                                   "a group cannot contain itself")
            self._push_undo()
            for other in tset.layers:
                if layer_id in other.children:
                    other.children.remove(layer_id)
            group.children.append(layer_id)
        else:
            self._push_undo()
            tset.layers = [lay for lay in tset.layers
                           if lay.id != layer_id]
            tset.layers.append(layer)
            for other in tset.layers:
                if layer_id in other.children:
                    other.children.remove(layer_id)
        self.doc.mark_dirty()

    # -- blending -------------------------------------------------------------
    def set_blend(self, set_name: str, layer_id: int, channel: str,
                  mode: str, opacity: float = 1.0) -> None:
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        self._channel(channel)
        parsed = BlendMode.parse(mode)
        if parsed is None:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"unknown blend mode {mode!r}")
        if not 0.0 <= opacity <= 1.0:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"opacity {opacity} out of 0..1")
        self._push_undo()
        layer.channels[channel] = _blend(parsed, opacity)
        self.doc.mark_touched(layer_id)

    def set_opacity(self, set_name: str, layer_id: int, channel: str,
                    opacity: float) -> None:
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        self._channel(channel)
        if not 0.0 <= opacity <= 1.0:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"opacity {opacity} out of 0..1")
        self._push_undo()
        current = layer.channels.get(channel)
        mode = current.mode if current is not None else BlendMode.NORMAL
        layer.channels[channel] = _blend(mode, opacity)
        self.doc.mark_touched(layer_id)

    # -- painting -------------------------------------------------------------
    def paint_stroke(self, layer_id: int, channel: str,
                     path: Sequence[tuple[float, float]],
                     color: Sequence[int], size: float = 16.0,
                     hardness: float = 0.5, opacity: float = 1.0) -> None:
        layer = self._find_layer(layer_id)
        self._channel(channel)
        if layer.kind != "paint":
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"layer {layer_id} is not a paint layer")
        if not path:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               "stroke path is empty")
        if size <= 0:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"brush size {size} must be positive")
        self._push_undo()
        img = layer.image(channel)
        stroke(img, list(path), Brush(size=size, hardness=hardness,
                                      opacity=opacity), list(color))
        self.doc.mark_touched(layer_id)

    def paint_rect(self, layer_id: int, channel: str, x: int, y: int,
                   w: int, h: int, color: Sequence[int]) -> None:
        layer = self._find_layer(layer_id)
        self._channel(channel)
        if layer.kind != "paint":
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"layer {layer_id} is not a paint layer")
        if w <= 0 or h <= 0:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"bad rectangle {w}x{h}")
        self._push_undo()
        img = layer.image(channel)
        fill_rect(img, x, y, w, h, list(color))
        self.doc.mark_touched(layer_id)

    def fill_layer(self, layer_id: int, channel: str,
                   color: Sequence[int]) -> None:
        layer = self._find_layer(layer_id)
        self._channel(channel)
        if layer.kind != "paint":
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"layer {layer_id} is not a paint layer")
        self._push_undo()
        img = layer.image(channel)
        fill_rect(img, 0, 0, img.width, img.height, list(color))
        self.doc.mark_touched(layer_id)

    def pattern_fill(self, layer_id: int, channel: str, pattern: str,
                     seed: int = 1, scale: float = 4.0) -> None:
        layer = self._find_layer(layer_id)
        self._channel(channel)
        kind = ProceduralKind.parse(pattern)
        if kind is None:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"unknown pattern {pattern!r}")
        self._push_undo()
        img = layer.image(channel)
        fill_pattern(img, Procedural(kind, scale=scale, seed=seed))
        self.doc.mark_touched(layer_id)

    # -- transforms -----------------------------------------------------------
    def resize(self, set_name: str, width: int, height: int) -> None:
        tset = self._set(set_name)
        if width <= 0 or height <= 0 or width > 8192 or height > 8192:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"bad size {width}x{height}")
        self._push_undo()
        for layer in tset.layers:
            for ch_name, img in list(layer._images.items()):
                layer._images[ch_name] = img.resampled(width, height)
            if layer._mask_image is not None:
                layer._mask_image = layer._mask_image.resampled(width, height)
            layer._width = width
            layer._height = height
        tset.width = width
        tset.height = height
        self.doc.mark_dirty()

    def apply_filter_to_layer(self, set_name: str, layer_id: int,
                              channel: str, spec: tuple) -> None:
        tset = self._set(set_name)
        layer = self._layer(tset, layer_id)
        self._channel(channel)
        if not spec or str(spec[0]).lower() not in SUPPORTED_FILTERS:
            raise TextureError(TextureErrorKind.UNSUPPORTED,
                               f"filter not ported: {spec!r}")
        self._push_undo()
        img = layer.image(channel)
        layer.set_image(channel, apply_filter(img, tuple(spec)))
        self.doc.mark_touched(layer_id)

    # -- derived work ---------------------------------------------------------
    def dilate_channel(self, set_name: str, channel: str,
                       max_dist: int = 8) -> dict:
        """UV padding: composite the channel, then pad uncovered
        texels from their nearest covered texel (BFS dilation) and
        store the padded result as a paint layer. Returns the report."""
        tset = self._set(set_name)
        self._channel(channel)
        if max_dist < 1:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               "max_dist must be >= 1")
        comp = composite(self.doc, set_name, channel)
        comps = comp.comps - 1
        covered = [False] * (comp.width * comp.height)
        for y in range(comp.height):
            for x in range(comp.width):
                if comp.get(x, y)[comps] > 0:
                    covered[y * comp.width + x] = True
        dmap = DilationMap(covered, comp.width, comp.height,
                           max_dist=max_dist)
        flat: list[int] = []
        for y in range(comp.height):
            for x in range(comp.width):
                flat.extend(comp.get(x, y)[:comps])
        dmap.apply(flat, comps)
        padded_count = sum(1 for i, c in enumerate(covered) if not c
                           and dmap.src[i] != NONE)
        self._push_undo()
        layer_id = self._next_layer_id()
        layer = Layer(id=layer_id, name=f"{channel} padding",
                      kind="paint")
        layer._width = tset.width
        layer._height = tset.height
        for ch_name in tset.channels:
            layer.channels[ch_name] = _default_blend(ch_name, "paint")
        img = TileImage(tset.width, tset.height, comps + 1)
        for y in range(comp.height):
            for x in range(comp.width):
                vals = flat[(y * comp.width + x) * comps:
                            (y * comp.width + x + 1) * comps]
                alpha = 65535 if covered[y * comp.width + x] else 0
                img.set(x, y, list(vals) + [alpha])
        img.prune_empty()
        layer.set_image(channel, img)
        tset.layers.append(layer)
        self.doc.mark_touched(layer_id)
        return {"layer_id": layer_id, "channel": channel,
                "padded_texels": padded_count, "max_dist": max_dist}

    def derive_normal(self, set_name: str, height_channel: str = "height",
                      target: str = "normal",
                      strength: float = 1.0) -> dict:
        """Derive a normal map from the composited height channel
        and store it as a paint layer. Returns the report."""
        tset = self._set(set_name)
        self._channel(height_channel)
        self._channel(target)
        height = composite(self.doc, set_name, height_channel)
        himg = TileImage(height.width, height.height, 1)
        for y in range(height.height):
            for x in range(height.width):
                himg.set(x, y, [height.get(x, y)[0]])
        himg.prune_empty()
        normal = height_to_normal(himg, strength=strength)
        self._push_undo()
        layer_id = self._next_layer_id()
        layer = Layer(id=layer_id, name="Normal from height",
                      kind="paint")
        layer._width = tset.width
        layer._height = tset.height
        for ch_name in tset.channels:
            layer.channels[ch_name] = _default_blend(ch_name, "paint")
        # The normal image is dense; store it with full alpha.
        nimg = TileImage(normal.width, normal.height, 4)
        for y in range(normal.height):
            for x in range(normal.width):
                v = normal.get(x, y)
                nimg.set(x, y, [v[0], v[1], v[2], 65535])
        nimg.prune_empty()
        layer.set_image(target, nimg)
        tset.layers.append(layer)
        self.doc.mark_touched(layer_id)
        return {"layer_id": layer_id, "channel": target,
                "strength": strength}

    def _find_layer(self, layer_id: int) -> Layer:
        for tset in self.doc.sets:
            for layer in tset.layers:
                if layer.id == layer_id:
                    return layer
        raise TextureError(TextureErrorKind.NOT_FOUND,
                           f"no layer {layer_id}")


def _default_blend(channel: str, kind: str):
    from .model import ChannelBlend
    if kind == "group":
        return ChannelBlend(mode=BlendMode.PASSTHROUGH, opacity=1.0)
    return ChannelBlend(mode=BlendMode.NORMAL, opacity=1.0)


def _blend(mode: BlendMode, opacity: float):
    from .model import ChannelBlend
    return ChannelBlend(mode=mode, opacity=opacity)
