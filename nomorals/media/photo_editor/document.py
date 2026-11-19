"""The photo document model — PhotoCraft's document model in
Devon's idiom.

PhotoCraft's ``crates/doc`` (pure data: layers, masks, adjustments,
effects, smart objects) over ``crates/raster`` COW tiles: a document
is layers of copy-on-write 256px tiles, so a snapshot is a dict
copy — unchanged tiles are shared, never copied — and undo is a
snapshot restore. History is an ordered op list (the D2 op-spec
discipline Devon already uses in ImageDocument): every edit is
stable-id + JSON-able params, applied through the EXISTING edit
ladder (``media_edit.images`` for crop/enhance, ``imggen.develop``
for the develop stage) — this organ never re-implements a pixel op.

Layer order: ``layers[0]`` is the BOTTOM layer (PhotoCraft's
order, matching compositing order). PSD files store layers
top-first, so ``from_psd_bytes`` reverses on import and
``to_psd_bytes`` reverses on export.

Edits: ``crop`` is a canvas op — it resizes the document and
every layer through the ladder's ``op_crop``; ``enhance`` and
``develop`` are pixel ops on the active layer (the topmost visible
pixel layer, or the ``layer=`` index). A document with no pixel
layers (adjustment-only PSDs) refuses a pixel edit honestly.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import Any

from ...core.ids import new_id
from .psd import PsdFile, encode_psd

__all__ = [
    "PhotoDocument",
    "PhotoEditorError",
    "PhotoLayer",
    "Surface",
    "Tile",
    "TILE_SIZE",
]

#: Copy-on-write tile edge, in pixels (PhotoCraft's raster tiles).
TILE_SIZE = 256

#: The edit vocabulary this document speaks (all ladder ops).
EDIT_OPS = ("crop", "enhance", "develop")


class PhotoEditorError(Exception):
    """A photo-document operation that cannot be done — always with
    the reason named."""


# ---------------------------------------------------------------------------
# Copy-on-write tiles
# ---------------------------------------------------------------------------

class Tile:
    """One 256x256 RGBA tile. Immutable: the payload is ``bytes``,
    so sharing a tile between snapshots is free, and writing a
    tile always produces a new tile — the old one stays valid for
    every snapshot that holds it."""

    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        if len(data) != TILE_SIZE * TILE_SIZE * 4:
            raise PhotoEditorError(
                f"a tile holds {TILE_SIZE * TILE_SIZE * 4} bytes, "
                f"got {len(data)}")
        self.data = data

    @classmethod
    def blank(cls) -> Tile:
        return cls(bytes(TILE_SIZE * TILE_SIZE * 4))

    @classmethod
    def from_pil(cls, image: Any, tx: int, ty: int) -> Tile:
        """The tile at grid position (tx, ty) of ``image`` — padded
        with transparent black where the image ends."""
        from PIL import Image

        region = Image.new("RGBA", (TILE_SIZE, TILE_SIZE),
                           (0, 0, 0, 0))
        box = (tx * TILE_SIZE, ty * TILE_SIZE,
               (tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE)
        crop = image.crop(box)
        region.paste(crop, (0, 0))
        return cls(region.tobytes())

    def to_pil(self) -> Any:
        from PIL import Image

        return Image.frombytes(
            "RGBA", (TILE_SIZE, TILE_SIZE), self.data)

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int, int]:
        offset = (y * TILE_SIZE + x) * 4
        d = self.data
        return (d[offset], d[offset + 1], d[offset + 2],
                d[offset + 3])


class Surface:
    """A layer's pixels as a tile grid. A snapshot is a shallow
    dict copy — tiles are immutable, so the copy shares every tile
    until one is replaced."""

    def __init__(self, width: int, height: int,
                 tiles: dict[tuple[int, int], Tile] | None = None
                 ) -> None:
        self.width = max(0, width)
        self.height = max(0, height)
        self.tiles: dict[tuple[int, int], Tile] = dict(tiles or {})

    @classmethod
    def blank(cls, width: int, height: int) -> Surface:
        return cls(width, height)

    @classmethod
    def from_pil(cls, image: Any) -> Surface:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        width, height = image.size
        tiles: dict[tuple[int, int], Tile] = {}
        for ty in range((height + TILE_SIZE - 1) // TILE_SIZE):
            for tx in range((width + TILE_SIZE - 1) // TILE_SIZE):
                tiles[(tx, ty)] = Tile.from_pil(image, tx, ty)
        return cls(width, height, tiles)

    def to_pil(self) -> Any:
        from PIL import Image

        out = Image.new("RGBA", (max(1, self.width),
                                 max(1, self.height)),
                        (0, 0, 0, 0))
        for (tx, ty), tile in self.tiles.items():
            out.paste(tile.to_pil(),
                      (tx * TILE_SIZE, ty * TILE_SIZE))
        return out.crop((0, 0, self.width, self.height)) \
            if self.width and self.height else out

    def snapshot(self) -> dict[tuple[int, int], Tile]:
        return dict(self.tiles)

    def restore(self, snap: dict[tuple[int, int], Tile]) -> None:
        self.tiles = dict(snap)

    def replace_with_pil(self, image: Any) -> None:
        """Replace every tile from a PIL image (the edit path —
        the ladder renders, the surface re-tiles)."""
        fresh = Surface.from_pil(image)
        self.width = fresh.width
        self.height = fresh.height
        self.tiles = fresh.tiles


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class PhotoLayer:
    """One layer: geometry, opacity, visibility, a tile surface,
    and an optional mask (bytes + rect, as the PSD carries it).

    ``kind`` is "pixels" for a real picture layer, "adjustment" /
    "fill" for layers whose pixels are parameters, and
    "group-start" / "group-end" for section markers. The parsed
    features live on the layer too — ``adjustment`` (a parsed
    adjustment dict), ``fill_spec``, ``effects``, ``text_spec``
    (the TySh spec), ``smart`` (the SoLd/PlLd spec) — so the
    tool and the page can describe what a layer IS, and the
    flatten renders it through the shared compositor. Non-pixel
    layers keep their tagged blocks so a save writes them back
    instead of silently dropping them.

    ``dirty`` marks a layer the editor changed since load; a
    document with no dirty layer writes back byte-identical to
    the file it was opened from."""

    def __init__(self, name: str, left: int, top: int,
                 width: int, height: int, *,
                 opacity: int = 255, visible: bool = True,
                 blend: str = "norm",
                 surface: Surface | None = None,
                 mask: bytes | None = None,
                 mask_rect: tuple[int, int, int, int] | None = None,
                 layer_id: str = "",
                 kind: str = "pixels",
                 raw_blocks: list[tuple[str, bytes]] | None = None,
                 adjustment: dict | None = None,
                 fill_spec: dict | None = None,
                 effects: dict | None = None,
                 text_spec: dict | None = None,
                 smart: dict | None = None,
                 fill_opacity: float = 1.0,
                 mask_default: int = 255,
                 mask_disabled: bool = False,
                 advanced: dict | None = None,
                 clipping: int = 0,
                 dirty: bool = False,
                 ) -> None:
        self.id = layer_id or new_id()
        self.name = name
        self.left = left
        self.top = top
        self.width = max(0, width)
        self.height = max(0, height)
        self.opacity = max(0, min(255, opacity))
        self.visible = visible
        self.blend = blend
        self.surface = surface or Surface.blank(self.width,
                                                self.height)
        self.mask = mask
        self.mask_rect = mask_rect
        self.kind = kind
        self.raw_blocks: list[tuple[str, bytes]] = list(
            raw_blocks or [])
        self.adjustment = adjustment
        self.fill_spec = fill_spec
        self.effects = effects
        self.text_spec = text_spec
        self.smart = smart
        self.fill_opacity = fill_opacity
        self.mask_default = mask_default
        self.mask_disabled = mask_disabled
        self.advanced: dict = dict(advanced or {})
        self.clipping = clipping
        self.dirty = dirty
        # Opacity/visibility changed only (no pixel edit) — the
        # save can splice the source file instead of re-encoding.
        self._state_dirty = False
        # The source layer's native channel bytes (colour + alpha,
        # in the document's mode), kept so an unedited layer writes
        # back its original planes verbatim — a CMYK/Lab file
        # round-trips exactly instead of through a lossy
        # native -> RGB -> native conversion.
        self._native_channels: dict | None = None
        #: the source PSD layer record this layer came from
        #: (None for a layer built in the editor) — the flatten
        #: uses it for the FEid smart-filter mask.
        self._record: Any | None = None

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_pil(self) -> Any:
        """The layer's pixels at its own rect (RGBA)."""
        return self.surface.to_pil()

    def mask_to_pil(self) -> Any:
        from PIL import Image

        if self.mask is None or self.mask_rect is None:
            return Image.new("L", (max(1, self.width),
                                   max(1, self.height)), 255)
        mtop, mleft, mbottom, mright = self.mask_rect
        mw = max(0, mright - mleft)
        mh = max(0, mbottom - mtop)
        if not mw or not mh:
            return Image.new("L", (max(1, self.width),
                                   max(1, self.height)), 255)
        return Image.frombytes(
            "L", (mw, mh), self.mask[:mw * mh])

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "opacity": self.opacity,
            "visible": self.visible,
            "blend": self.blend,
            "has_mask": self.mask is not None,
            "dirty": self.dirty,
        }
        if self.adjustment is not None:
            out["adjustment"] = self.adjustment.get("kind")
        if self.fill_spec is not None:
            out["fill"] = self.fill_spec.get("kind")
        if self.effects is not None:
            out["effects"] = sorted(
                k for k, v in self.effects.items()
                if isinstance(v, dict) and v.get("present"))
        if self.text_spec is not None:
            out["text"] = self.text_spec.get("text", "")[:80]
        if self.smart is not None:
            filters = (self.smart.get("filters") or [])
            out["smart_filters"] = [
                f.get("name") for f in filters
                if isinstance(f, dict)]
        return out


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class PhotoDocument:
    """A layered photo document: canvas + layers + op history.

    ``layers[0]`` is the bottom layer. ``history`` is the ordered
    op journal (data, never code); ``undo``/``redo`` ride snapshots
    of the tile dicts, so undo never re-renders.
    """

    def __init__(self, width: int, height: int, *,
                 title: str = "",
                 layers: list[PhotoLayer] | None = None,
                 mode: str = "RGB", depth: int = 8) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.layers: list[PhotoLayer] = list(layers or [])
        self.history: list[dict] = []
        self._undone: list[dict] = []
        self._snapshots: list[dict] = []
        self.notes: list[str] = []
        #: the PSD colour mode / depth the document came from
        #: (the compositor's mode + depth rules).
        self.mode = mode
        self.depth = depth
        #: the exact bytes the document was opened from, when it
        #: was opened from a PSD — a document with no dirty layer
        #: and no structural change writes these back verbatim.
        self._source_psd: bytes | None = None
        #: the parsed source file (for the FEid smart-filter
        #: cache and the stable writer).
        self._psd: Any | None = None
        #: set by a canvas op (crop) — the document's shape
        #: changed, so no verbatim write-back is possible.
        self._structure_dirty = False

    # -- open / save ----------------------------------------------------
    @classmethod
    def from_psd_bytes(cls, data: bytes,
                       title: str = "") -> PhotoDocument:
        from .compose import describe_layer

        psd = PsdFile.from_bytes(data)
        layers: list[PhotoLayer] = []
        notes_flat = False
        # PSD stores layers bottom-first (file order) — the
        # document is bottom-first too, so no reversal.
        # Adjustment, fill and group layers are kept as zero-size
        # layers of their kind (their tagged blocks ride along so
        # a save writes them back) — never silently dropped.
        # The parsed features (adjustment / fill / effects / text
        # / smart) are read off describe_layer's parse of the
        # layer's own blocks, so the layer describes what it is.
        for record in psd.layers:
            kind = "pixels"
            if record.is_group_start or record.is_group_end:
                kind = "group-start" if record.is_group_start \
                    else "group-end"
            elif record.is_adjustment:
                kind = "adjustment"
            elif record.is_fill:
                kind = "fill"
            parsed = describe_layer(
                kind=kind, raw_blocks=record.raw_blocks,
                name=record.display_name,
                # No document size: the load only needs the
                # PARSED features (adjustment / fill / effects /
                # text / smart) — passing the size would render
                # every text layer's glyphs here and throw them
                # away; the flatten renders them when needed.
                doc_width=0, doc_height=0)
            if record.is_group_start or record.is_group_end:
                layers.append(PhotoLayer(
                    record.display_name, record.left, record.top,
                    record.width, record.height,
                    opacity=record.opacity, visible=record.visible,
                    blend=record.blend_key, kind=kind,
                    raw_blocks=record.raw_blocks,
                    clipping=record.clipping,
                    effects=parsed.get("effects")))
                layers[-1]._record = record
                continue
            if record.is_adjustment or record.is_fill:
                layers.append(PhotoLayer(
                    record.display_name, record.left, record.top,
                    record.width, record.height,
                    opacity=record.opacity, visible=record.visible,
                    blend=record.blend_key, kind=kind,
                    raw_blocks=record.raw_blocks,
                    clipping=record.clipping,
                    adjustment=parsed.get("adjustment"),
                    fill_spec=parsed.get("fill_spec"),
                    effects=parsed.get("effects"),
                    mask=record.mask, mask_rect=record.mask_rect,
                    mask_default=record.mask_default,
                    mask_disabled=record.mask_disabled))
                layers[-1]._record = record
                continue
            if record.width == 0 or record.height == 0:
                # A zero-size pixel layer (text on a path: its
                # glyphs live only in the TySh vector data, no
                # raster at all) is kept as a zero-size layer of
                # its kind — never silently dropped, so a save
                # writes its tagged blocks back.
                layers.append(PhotoLayer(
                    record.display_name, record.left, record.top,
                    0, 0, opacity=record.opacity,
                    visible=record.visible,
                    blend=record.blend_key,
                    raw_blocks=record.raw_blocks,
                    clipping=record.clipping,
                    effects=parsed.get("effects"),
                    text_spec=parsed.get("text"),
                    smart=parsed.get("smart")))
                layers[-1]._record = record
                continue
            rgba = _layer_rgba(record, psd.mode)
            if rgba is None:
                continue
            surface = Surface.from_pil(rgba)
            layers.append(PhotoLayer(
                record.display_name, record.left, record.top,
                record.width, record.height,
                opacity=record.opacity, visible=record.visible,
                blend=record.blend_key, surface=surface,
                clipping=record.clipping,
                mask=record.mask, mask_rect=record.mask_rect,
                mask_default=record.mask_default,
                mask_disabled=record.mask_disabled,
                raw_blocks=record.raw_blocks,
                effects=parsed.get("effects"),
                text_spec=parsed.get("text"),
                smart=parsed.get("smart")))
            layers[-1]._native_channels = dict(record.channels)
            layers[-1]._record = record
        if not layers:
            # A flat file (no layers at all — the 16/32-bit
            # corpus files, and any flattened PSD) opens as one
            # Background layer carrying the merged composite;
            # otherwise the editor would show a blank canvas for
            # a real picture.
            flat = psd.composite_rgba()
            layers.append(PhotoLayer(
                "Background", 0, 0, psd.width, psd.height,
                surface=Surface.from_pil(flat)))
            notes_flat = True
        doc = cls(psd.width, psd.height, title=title, layers=layers,
                  mode=psd.mode, depth=psd.depth)
        doc._source_psd = bytes(data)
        doc._psd = psd
        if notes_flat:
            doc.notes.append(
                "flat image — opened as a single Background layer")
        doc.notes.extend(psd.unrendered())
        doc.notes.append(
            f"opened PSD ({psd.mode}, {psd.depth}-bit, "
            f"{len(psd.layers)} layers in file, "
            f"{len(layers)} pixel layers)")
        return doc

    @classmethod
    def open_psd(cls, path: str | os.PathLike[str],
                 title: str = "") -> PhotoDocument:
        p = Path(path)
        if not p.exists():
            raise PhotoEditorError(f"no such PSD file: {p}")
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise PhotoEditorError(
                f"cannot read {p}: {exc}") from exc
        return cls.from_psd_bytes(
            data, title=title or p.stem)

    def to_psd_bytes(self) -> bytes:
        """Write the document as PSD bytes. A document opened
        from a PSD with no dirty layer and no structural change
        writes back byte-identical to the file it was opened
        from; a document whose only changes are layer opacity /
        visibility splices the source file (clean layers verbatim,
        changed layers re-encoded); anything else goes through
        the encoder."""
        if self._source_psd is not None \
                and not self._structure_dirty \
                and not any(layer.dirty or layer._state_dirty
                            for layer in self.layers):
            return self._source_psd
        if self._psd is not None and not self._structure_dirty \
                and not any(layer.dirty for layer in self.layers) \
                and any(layer._state_dirty for layer in self.layers):
            for layer in self.layers:
                if not layer._state_dirty or layer._record is None:
                    continue
                record = layer._record
                record.opacity = layer.opacity
                record.flags = 0 if layer.visible else 0x02
                record.dirty = True
            return self._psd.to_bytes()
        return encode_psd(self)

    def save_psd(self, path: str | os.PathLike[str]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(self.to_psd_bytes())
        os.replace(tmp, p)
        return str(p)

    # -- history ---------------------------------------------------------
    def _take_snapshot(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "layers": [
                {
                    "left": layer.left,
                    "top": layer.top,
                    "width": layer.width,
                    "height": layer.height,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "tiles": layer.surface.snapshot(),
                    "surface_size": (layer.surface.width,
                                     layer.surface.height),
                    "mask": layer.mask,
                    "mask_rect": layer.mask_rect,
                    "dirty": layer.dirty,
                    "state_dirty": layer._state_dirty,
                }
                for layer in self.layers
            ],
        }

    def _restore_snapshot(self, snap: dict) -> None:
        self.width = snap["width"]
        self.height = snap["height"]
        for layer, saved in zip(self.layers, snap["layers"],
                                  strict=True):
            layer.left = saved["left"]
            layer.top = saved["top"]
            layer.width = saved["width"]
            layer.height = saved["height"]
            layer.opacity = saved["opacity"]
            layer.visible = saved["visible"]
            layer.surface.width, layer.surface.height = \
                saved["surface_size"]
            layer.surface.restore(saved["tiles"])
            layer.mask = saved["mask"]
            layer.mask_rect = saved["mask_rect"]
            layer.dirty = saved["dirty"]
            layer._state_dirty = saved.get("state_dirty", False)

    def undo(self) -> dict | None:
        if not self.history:
            return None
        entry = self.history.pop()
        snap = self._snapshots.pop()
        self._undone.append(entry)
        self._restore_snapshot(snap)
        return entry

    def redo(self) -> dict | None:
        if not self._undone:
            return None
        entry = self._undone.pop()
        # Re-running a deterministic ladder op from the restored
        # before-state reproduces the after-state exactly; the
        # before-state is snapshotted first so undo still works.
        self._snapshots.append(self._take_snapshot())
        self._apply(entry["op"], entry["params"], record=False)
        self.history.append(entry)
        return entry

    # -- edits ------------------------------------------------------------
    @property
    def active_layer(self) -> PhotoLayer | None:
        """The topmost visible pixel layer (the one a pixel edit
        lands on)."""
        for layer in reversed(self.layers):
            if layer.visible and layer.width and layer.height:
                return layer
        return None

    def apply_edit(self, op: str, **params: Any) -> dict:
        """Apply one edit through the ladder and journal it. ``op``:
        crop / enhance / develop. Pixel ops take ``layer=<index>``
        (default: the active layer); crop takes ``box=(l, t, r, b)``
        in canvas coordinates."""
        if op not in EDIT_OPS:
            raise PhotoEditorError(
                f"unknown photo edit {op!r}; vocabulary: "
                f"{list(EDIT_OPS)}")
        entry = {"op": op, "params": copy.deepcopy(params)}
        self._snapshots.append(self._take_snapshot())
        try:
            self._apply(op, params, record=True)
        except Exception:
            self._snapshots.pop()
            raise
        self.history.append(entry)
        self._undone.clear()
        return entry

    def set_layer_state(self, ref: str | int, *,
                        visible: bool | None = None,
                        opacity: int | None = None) -> dict:
        """Toggle a layer's visibility / opacity, journaled like
        an edit (undoable). ``ref`` is a layer index or name.
        Opacity is 0..255. A state-only change writes back through
        the source splice, not a full re-encode."""
        if visible is None and opacity is None:
            raise PhotoEditorError(
                "layer state needs visible and/or opacity")
        if opacity is not None and not 0 <= int(opacity) <= 255:
            raise PhotoEditorError(
                f"opacity {opacity!r} is outside 0..255")
        params: dict[str, Any] = {"layer": ref}
        if visible is not None:
            params["visible"] = bool(visible)
        if opacity is not None:
            params["opacity"] = int(opacity)
        entry = {"op": "layer", "params": params}
        self._snapshots.append(self._take_snapshot())
        try:
            self._apply_layer_state(params)
        except Exception:
            self._snapshots.pop()
            raise
        self.history.append(entry)
        self._undone.clear()
        return entry

    def _apply(self, op: str, params: dict, *, record: bool) -> None:
        if op == "crop":
            self._apply_crop(params)
        elif op == "enhance":
            self._apply_pixel("enhance", params)
        elif op == "develop":
            self._apply_pixel("develop", params)
        elif op == "layer":
            self._apply_layer_state(params)
        else:  # pragma: no cover - guarded by apply_edit
            raise PhotoEditorError(f"unknown photo edit {op!r}")

    def _apply_layer_state(self, params: dict) -> None:
        ref = params.get("layer")
        target: PhotoLayer | None = None
        if isinstance(ref, int) or (
                isinstance(ref, str) and str(ref).isdigit()):
            index = int(ref)
            if 0 <= index < len(self.layers):
                target = self.layers[index]
        else:
            name = str(ref).strip().lower()
            for layer in self.layers:
                if layer.name.lower() == name:
                    target = layer
                    break
        if target is None:
            raise PhotoEditorError(f"no layer {ref!r}")
        if "visible" in params:
            target.visible = bool(params["visible"])
        if "opacity" in params:
            target.opacity = int(params["opacity"])
        target._state_dirty = True

    def _apply_crop(self, params: dict) -> None:
        box = params.get("box")
        if box is None:
            raise PhotoEditorError("crop needs box=(l, t, r, b)")
        left, top, right, bottom = (int(v) for v in box)
        if not (0 <= left < right <= self.width
                and 0 <= top < bottom <= self.height):
            raise PhotoEditorError(
                f"crop box {(left, top, right, bottom)} outside "
                f"{self.width}x{self.height}")
        from ...media_edit.images import op_crop

        for layer in self.layers:
            # Intersect the layer with the new canvas; the ladder
            # crops the layer's own pixels.
            lo_l = max(left, layer.left)
            lo_t = max(top, layer.top)
            lo_r = min(right, layer.right)
            lo_b = min(bottom, layer.bottom)
            if lo_r <= lo_l or lo_b <= lo_t:
                layer.surface = Surface.blank(0, 0)
                layer.width = 0
                layer.height = 0
                layer.left = 0
                layer.top = 0
                continue
            img = layer.to_pil()
            cropped = op_crop(img, box=(
                lo_l - layer.left, lo_t - layer.top,
                lo_r - layer.left, lo_b - layer.top))
            layer.surface.replace_with_pil(cropped)
            layer.left = lo_l - left
            layer.top = lo_t - top
            layer.width = lo_r - lo_l
            layer.height = lo_b - lo_t
            if layer.mask is not None and layer.mask_rect is not None:
                layer.mask_rect = (
                    max(layer.mask_rect[0], top) - top,
                    max(layer.mask_rect[1], left) - left,
                    min(layer.mask_rect[2], bottom) - top,
                    min(layer.mask_rect[3], right) - left)
        self.width = right - left
        self.height = bottom - top
        # A crop changes the document's shape and every layer's
        # geometry — no verbatim write-back is possible after it.
        self._structure_dirty = True
        for layer in self.layers:
            layer.dirty = True

    def _apply_pixel(self, op: str, params: dict) -> None:
        index = params.get("layer")
        if index is None:
            layer = self.active_layer
        else:
            try:
                layer = self.layers[int(index)]
            except (IndexError, TypeError, ValueError):
                raise PhotoEditorError(
                    f"no layer {index!r} — this document has "
                    f"{len(self.layers)} layers") from None
        if layer is None:
            raise PhotoEditorError(
                "no visible pixel layer to edit — this document "
                "has no layer a pixel edit can land on")
        clean = {k: v for k, v in params.items() if k != "layer"}
        img = layer.to_pil()
        if op == "enhance":
            from ...media_edit.images import apply_chain

            out = apply_chain(img, [{"op": "enhance", **clean}])
        else:
            from ..imggen.develop import develop_image, develop_settings_from_params, render_develop

            if any("." in k or k == "settings" for k in clean):
                # The full LightCraft engine: dotted controls
                # (light.exposure, wb.temp, ...) or a whole
                # settings dict.
                settings = develop_settings_from_params(clean)
                out = render_develop(img, settings)
            else:
                out = develop_image(img, **clean)
        layer.surface.replace_with_pil(out)
        layer.dirty = True

    # -- render -----------------------------------------------------------
    def flatten(self) -> Any:
        """The composited document through the shared numpy
        compositor — the same render the PSD flatten runs:
        blend modes, opacity, masks, clipping, groups,
        adjustments, fills, effects, live text and smart
        objects. A layerless document flattens to its merged
        composite."""

        from .blend import mode_name
        from .compose import (
            apply_mask,
            composite_layers,
            describe_layer,
            mask_to_canvas,
            pil_to_array,
        )

        if not self.layers:
            from PIL import Image

            if self._psd is not None:
                return self._psd.composite_rgba()
            return Image.new("RGBA", (max(1, self.width),
                                      max(1, self.height)),
                             (0, 0, 0, 0))
        depth = {8: "u8", 16: "u16", 32: "f32"}.get(self.depth,
                                                   "u8")
        dicts: list[dict] = []
        psd = self._psd
        for layer in self.layers:
            if layer.kind == "group-start":
                dicts.append(describe_layer(
                    kind="group-start",
                    blend=mode_name(layer.blend),
                    name=layer.name))
                continue
            if layer.kind == "group-end":
                dicts.append(describe_layer(kind="group-end",
                                            name=""))
                continue
            if layer.kind == "adjustment":
                dicts.append(describe_layer(
                    kind="adjustment",
                    raw_blocks=layer.raw_blocks,
                    name=layer.name, visible=layer.visible,
                    opacity=layer.opacity,
                    blend=mode_name(layer.blend),
                    mask=mask_to_canvas(
                        layer.mask, layer.mask_rect,
                        self.width, self.height,
                        mask_default=layer.mask_default,
                        disabled=layer.mask_disabled)))
                continue
            if layer.kind == "fill":
                dicts.append(describe_layer(
                    kind="fill",
                    raw_blocks=layer.raw_blocks,
                    name=layer.name, visible=layer.visible,
                    opacity=layer.opacity,
                    blend=mode_name(layer.blend),
                    mask=mask_to_canvas(
                        layer.mask, layer.mask_rect,
                        self.width, self.height,
                        mask_default=layer.mask_default,
                        disabled=layer.mask_disabled)))
                continue
            if not layer.width or not layer.height:
                # A zero-size layer has no raster of its own; a
                # text layer still renders from its TySh spec.
                if layer.text_spec is None:
                    continue
                rgba = None
            else:
                pil = layer.to_pil()
                rgba = apply_mask(
                    pil_to_array(pil), layer.mask,
                    layer.mask_rect,
                    (layer.top, layer.left, layer.bottom,
                     layer.right),
                    mask_default=layer.mask_default,
                    disabled=layer.mask_disabled)
                if psd is not None and layer._record is not None:
                    # The stored pixels of a smart object are
                    # the filtered render; the FEid mask blends
                    # filtered/unfiltered at render time.
                    rgba = psd._apply_feid(rgba, layer._record)
            dicts.append(describe_layer(
                rgba=rgba, left=layer.left, top=layer.top,
                width=layer.width, height=layer.height,
                opacity=layer.opacity,
                blend=mode_name(layer.blend),
                visible=layer.visible, kind="pixels",
                name=layer.name, raw_blocks=layer.raw_blocks,
                doc_width=self.width, doc_height=self.height))
        result = composite_layers(dicts, self.width, self.height,
                                  depth=depth, mode=self.mode)
        self.notes = [n for n in self.notes
                      if not n.startswith("unrendered: ")]
        for line in result.unrendered:
            self.notes.append(f"unrendered: {line}")
        return result.image

    def render_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.flatten().save(buf, format="PNG")
        return buf.getvalue()

    def describe(self) -> str:
        lines = [
            f"photo document '{self.title or 'untitled'}' "
            f"{self.width}x{self.height}, {len(self.layers)} layers"]
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  {i}. {layer.name} "
                f"{layer.width}x{layer.height} at "
                f"({layer.left},{layer.top}) "
                f"opacity {layer.opacity}"
                + ("" if layer.visible else " (hidden)"))
        for op in self.history:
            params = ", ".join(
                f"{k}={v}" for k, v in op["params"].items())
            lines.append(f"  edit: {op['op']}({params})")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    # -- ImageDocument bridge ---------------------------------------------
    def to_image_document(self, png_path: str | os.PathLike[str]
                          ) -> Any:
        """Compile to an ImageDocument: the current flatten as the
        source PNG, the photo op journal carried as notes (the ops
        are already in the pixels — replaying them would double
        them)."""
        from ..imggen.document import ImageDocument

        target = Path(png_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.render_png_bytes())
        doc = ImageDocument(
            str(target), title=self.title or "untitled")
        doc.notes.extend(
            f"photo op already applied to the source: {op['op']} "
            f"({', '.join(f'{k}={v}' for k, v in op['params'].items())})"
            for op in self.history)
        return doc

    @classmethod
    def from_image_document(cls, imgdoc: Any) -> PhotoDocument:
        """Rebuild from an ImageDocument as a single layer."""
        rendered = imgdoc.render()
        image = rendered.image
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        layer = PhotoLayer(
            imgdoc.title or "image", 0, 0, image.width, image.height,
            surface=Surface.from_pil(image))
        return cls(image.width, image.height,
                   title=imgdoc.title or "", layers=[layer])

    def to_dict(self) -> dict:
        return {
            "kind": "devon-photo-document",
            "version": 1,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "layers": [layer.to_dict() for layer in self.layers],
            "history": copy.deepcopy(self.history),
            "notes": list(self.notes),
        }


def _layer_rgba(record: Any, mode: str) -> Any | None:
    """A parsed PSD layer record -> an RGBA image at its own rect.
    The record's channels are already decoded 8-bit planes."""
    from PIL import Image

    width, height = record.width, record.height
    if width == 0 or height == 0:
        return None
    planes = record.channels
    if mode == "RGB":
        if any(i not in planes for i in (0, 1, 2)):
            return None
        rgb = Image.merge("RGB", [
            Image.frombytes("L", (width, height), planes[0]),
            Image.frombytes("L", (width, height), planes[1]),
            Image.frombytes("L", (width, height), planes[2]),
        ])
        alpha = Image.frombytes(
            "L", (width, height), planes[-1]) if -1 in planes \
            else Image.new("L", (width, height), 255)
        return Image.merge("RGBA", [*rgb.split(), alpha])
    if mode == "L":
        if 0 not in planes:
            return None
        gray = Image.frombytes("L", (width, height), planes[0])
        alpha = Image.frombytes(
            "L", (width, height), planes[-1]) if -1 in planes \
            else Image.new("L", (width, height), 255)
        return Image.merge("RGBA", [gray, gray, gray, alpha])
    if mode == "CMYK":
        if any(i not in planes for i in range(4)):
            return None
        cmyk = Image.merge("CMYK", [
            Image.frombytes("L", (width, height), planes[i])
            for i in range(4)])
        rgb = cmyk.convert("RGB")
        alpha = Image.frombytes(
            "L", (width, height), planes[-1]) if -1 in planes \
            else Image.new("L", (width, height), 255)
        return Image.merge("RGBA", [*rgb.split(), alpha])
    if mode == "Lab":
        if any(i not in planes for i in range(3)):
            return None
        lab = Image.merge("LAB", [
            Image.frombytes("L", (width, height), planes[i])
            for i in range(3)])
        rgb = lab.convert("RGB")
        alpha = Image.frombytes(
            "L", (width, height), planes[-1]) if -1 in planes \
            else Image.new("L", (width, height), 255)
        return Image.merge("RGBA", [*rgb.split(), alpha])
    if mode == "Indexed":
        if 0 not in planes:
            return None
        indexed = Image.frombytes(
            "P", (width, height), planes[0])
        out = indexed.convert("RGBA")
        if -1 in planes:
            out.putalpha(Image.frombytes(
                "L", (width, height), planes[-1]))
        return out
    if mode == "Bitmap":
        if 0 not in planes:
            return None
        bits = Image.frombytes("1", (width, height), planes[0])
        out = bits.convert("RGBA")
        if -1 in planes:
            out.putalpha(Image.frombytes(
                "L", (width, height), planes[-1]))
        return out
    return None
