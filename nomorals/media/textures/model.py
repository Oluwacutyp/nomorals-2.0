"""Texture document model and the .tcraft codec (TextureCraft port,
census 3A.15).

Faithful to upstream ``crates/core/src/channel.rs`` (21 channels,
their kinds, component counts and default values) and
``crates/core/src/layer.rs`` (``LayerKind`` Paint / Fill / Group /
Instance, ``ChannelBlend``). The on-disk format is Devon's own
``.tcraft`` — a deterministic ZIP, mirroring upstream's native
project extension (a project zip of project JSON + sparse lossless
pixel tiles):

- ``project.json`` — canonical JSON (sorted keys, no whitespace):
  format, version, title, texture sets with their channels and
  layer stacks.
- ``tiles/<set>/<layer_id>/<channel>.bin`` — raw little-endian u16
  tile bytes, one file per allocated 64x64 tile, named
  ``<tx>_<ty>.bin``.
- ``tiles/<set>/<layer_id>/mask.bin`` — the layer mask tile bytes
  (1 component), same naming.

An unedited document serialises to its original bytes exactly;
edits re-serialise deterministically (sorted part names, fixed ZIP
epoch), so ``open -> edit -> save -> open`` leaves every untouched
tile byte-identical — checked by :meth:`TextureDocument.fidelity_report`.

Caps (upstream never crashes on hostile input): dimensions
<= 8192, sets <= 64, layers <= 512, tiles per image <= 16384.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .blend import BlendMode
from .errors import TextureError, TextureErrorKind
from .patterns import Procedural
from .tiles import TILE_AREA, TileImage

MAX_DIMENSION = 8192
MAX_SETS = 64
MAX_LAYERS = 512
_FORMAT = "tcraft"
_VERSION = 1
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class ChannelKind(StrEnum):
    COLOR = "color"
    NORMAL_MAP = "normal_map"
    SCALAR = "scalar"
    SIGNED = "signed"


class Channel(StrEnum):
    """The 21 texture channels (upstream ``Channel``)."""

    BASE_COLOR = "base_color"
    METALLIC = "metallic"
    ROUGHNESS = "roughness"
    NORMAL = "normal"
    HEIGHT = "height"
    EMISSIVE = "emissive"
    OPACITY = "opacity"
    AMBIENT_OCCLUSION = "ambient_occlusion"
    SPECULAR_LEVEL = "specular_level"
    SPECULAR = "specular"
    GLOSSINESS = "glossiness"
    COAT_WEIGHT = "coat_weight"
    COAT_ROUGHNESS = "coat_roughness"
    SHEEN_WEIGHT = "sheen_weight"
    SHEEN_COLOR = "sheen_color"
    SHEEN_ROUGHNESS = "sheen_roughness"
    ANISOTROPY_LEVEL = "anisotropy_level"
    ANISOTROPY_ANGLE = "anisotropy_angle"
    SCATTER_WEIGHT = "scatter_weight"
    TRANSMISSION = "transmission"
    COAT_NORMAL = "coat_normal"

    @classmethod
    def all(cls) -> list[Channel]:
        return list(cls)

    def kind(self) -> ChannelKind:
        if self in (Channel.BASE_COLOR, Channel.EMISSIVE, Channel.SPECULAR,
                    Channel.SHEEN_COLOR):
            return ChannelKind.COLOR
        if self in (Channel.NORMAL, Channel.COAT_NORMAL):
            return ChannelKind.NORMAL_MAP
        if self is Channel.HEIGHT:
            return ChannelKind.SIGNED
        return ChannelKind.SCALAR

    def components(self) -> int:
        return 3 if self.kind() in (ChannelKind.COLOR,
                                    ChannelKind.NORMAL_MAP) else 1

    def default_value(self) -> list[float]:
        table = {
            Channel.BASE_COLOR: [0.5, 0.5, 0.5],
            Channel.METALLIC: [0.0, 0.0, 0.0],
            Channel.HEIGHT: [0.0, 0.0, 0.0],
            Channel.EMISSIVE: [0.0, 0.0, 0.0],
            Channel.COAT_WEIGHT: [0.0, 0.0, 0.0],
            Channel.SHEEN_WEIGHT: [0.0, 0.0, 0.0],
            Channel.ANISOTROPY_LEVEL: [0.0, 0.0, 0.0],
            Channel.ANISOTROPY_ANGLE: [0.0, 0.0, 0.0],
            Channel.SCATTER_WEIGHT: [0.0, 0.0, 0.0],
            Channel.TRANSMISSION: [0.0, 0.0, 0.0],
            Channel.COAT_ROUGHNESS: [0.1, 0.1, 0.1],
            Channel.SHEEN_COLOR: [1.0, 1.0, 1.0],
            Channel.SHEEN_ROUGHNESS: [0.5, 0.5, 0.5],
            Channel.ROUGHNESS: [0.5, 0.5, 0.5],
            Channel.NORMAL: [0.5, 0.5, 1.0],
            Channel.COAT_NORMAL: [0.5, 0.5, 1.0],
            Channel.OPACITY: [1.0, 1.0, 1.0],
            Channel.AMBIENT_OCCLUSION: [1.0, 1.0, 1.0],
            Channel.SPECULAR_LEVEL: [0.5, 0.5, 0.5],
            Channel.GLOSSINESS: [0.5, 0.5, 0.5],
            Channel.SPECULAR: [0.04, 0.04, 0.04],
        }
        return table[self]

    @classmethod
    def parse(cls, name: str) -> Channel | None:
        needle = name.strip().lower().replace(" ", "_").replace("-", "_")
        for c in cls:
            if c.value == needle:
                return c
        return None


DEFAULT_SET = [Channel.BASE_COLOR, Channel.METALLIC, Channel.ROUGHNESS,
               Channel.NORMAL, Channel.HEIGHT]


@dataclass
class ChannelBlend:
    mode: BlendMode = BlendMode.NORMAL
    opacity: float = 1.0

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "opacity": self.opacity}

    @classmethod
    def from_dict(cls, data: dict) -> ChannelBlend:
        mode = BlendMode.parse(str(data.get("mode", "normal")))
        if mode is None:
            raise TextureError(TextureErrorKind.BAD_FORMAT,
                               f"unknown blend mode: {data.get('mode')!r}")
        return cls(mode=mode, opacity=float(data.get("opacity", 1.0)))


@dataclass
class Layer:
    """One layer of a texture set (upstream ``Layer``)."""

    id: int
    name: str
    kind: str = "paint"  # paint | fill | group | instance
    visible: bool = True
    channels: dict[str, ChannelBlend] = field(default_factory=dict)
    mask: dict | None = None
    effects: list = field(default_factory=list)
    fill: dict | None = None
    children: list[int] = field(default_factory=list)
    source: int | None = None
    color_tag: int | None = None
    _images: dict[str, TileImage] = field(default_factory=dict,
                                          repr=False)
    _mask_image: TileImage | None = field(default=None, repr=False)
    _materialized_fill: bool = field(default=False, repr=False)
    _width: int = field(default=64, repr=False)
    _height: int = field(default=64, repr=False)

    def image(self, channel: str) -> TileImage:
        """The layer's pixels for ``channel`` (materialising a fill
        layer on first access)."""
        if channel not in self._images:
            if self.kind == "fill" and self.fill is not None:
                self._materialize_fill(channel)
            else:
                ch = Channel.parse(channel)
                comps = ch.components() if ch is not None else 1
                if self.kind == "paint":
                    # A paint layer carries an alpha (coverage)
                    # component after the channel's value components.
                    comps += 1
                self._images[channel] = TileImage(self._width, self._height,
                                                  comps)
        return self._images[channel]

    def set_image(self, channel: str, image: TileImage) -> None:
        self._images[channel] = image

    def mask_image(self) -> TileImage | None:
        return self._mask_image

    def _materialize_fill(self, channel: str) -> None:
        """Fill the layer's image for ``channel`` from its fill
        definition. A uniform fill writes its colour (per-channel:
        base colour takes the RGB, other channels take the channel's
        default value); a procedural fill writes the pattern value
        (replicated across the channel's components; normal channels
        stay flat)."""
        fill = self.fill or {}
        ch = Channel.parse(channel)
        comps = ch.components() if ch is not None else 1
        img = TileImage(self._width, self._height, comps)
        self._images[channel] = img
        if fill.get("kind") == "uniform":
            color = tuple(int(v) for v in fill.get("color", (0, 0, 0, 65535)))
            if channel == "base_color":
                vals = list(color[:3]) + [65535]
                for ty in range(img.tiles_y):
                    for tx in range(img.tiles_x):
                        img.tile_mut(tx, ty, vals[:3])
            elif ch is not None and ch.kind() is ChannelKind.NORMAL_MAP:
                flat = [32768, 32768, 65535]
                for ty in range(img.tiles_y):
                    for tx in range(img.tiles_x):
                        img.tile_mut(tx, ty, flat)
            return
        if fill.get("kind") == "procedural":
            pr = Procedural.from_dict({
                "kind": fill.get("pattern", "clouds"),
                "scale": fill.get("scale", 4.0),
                "seed": fill.get("seed", 1),
                "contrast": fill.get("contrast", 1.0),
                "balance": fill.get("balance", 0.0),
                "invert": fill.get("invert", False),
            })
            if ch is not None and ch.kind() is ChannelKind.NORMAL_MAP:
                flat = [32768, 32768, 65535]
                for ty in range(img.tiles_y):
                    for tx in range(img.tiles_x):
                        img.tile_mut(tx, ty, flat)
                return
            from .patterns import procedural as _procedural
            for y in range(self._height):
                for x in range(self._width):
                    v = _procedural(pr, x / self._width, y / self._height)
                    u = int(v * 65535.0 + 0.5)
                    img.set(x, y, [u] * comps)
            return

    def to_dict(self) -> dict:
        data: dict = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "visible": self.visible,
            "channels": {k: v.to_dict()
                         for k, v in sorted(self.channels.items())},
            "effects": self.effects,
        }
        if self.mask is not None:
            data["mask"] = self.mask
        if self.fill is not None:
            data["fill"] = self.fill
        if self.children:
            data["children"] = self.children
        if self.source is not None:
            data["source"] = self.source
        if self.color_tag is not None:
            data["color_tag"] = self.color_tag
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Layer:
        channels = {}
        for name, cb in (data.get("channels") or {}).items():
            channels[name] = ChannelBlend.from_dict(cb)
        return cls(
            id=int(data["id"]),
            name=str(data.get("name", "Layer")),
            kind=str(data.get("kind", "paint")),
            visible=bool(data.get("visible", True)),
            channels=channels,
            mask=data.get("mask"),
            effects=list(data.get("effects") or []),
            fill=data.get("fill"),
            children=[int(c) for c in data.get("children") or []],
            source=(int(data["source"]) if data.get("source") is not None
                    else None),
            color_tag=data.get("color_tag"),
        )


@dataclass
class TextureSet:
    """One texture set: a size, a channel list and a layer stack
    (bottom -> top)."""

    name: str
    width: int
    height: int
    channels: list[str] = field(default_factory=lambda: ["base_color"])
    layers: list[Layer] = field(default_factory=list)

    def layer_by_id(self, layer_id: int) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise TextureError(TextureErrorKind.NOT_FOUND,
                           f"no layer {layer_id} in set {self.name!r}")

    def has_layer(self, layer_id: int) -> bool:
        return any(layer.id == layer_id for layer in self.layers)

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width,
                "height": self.height, "channels": self.channels,
                "layers": [layer.to_dict() for layer in self.layers]}

    @classmethod
    def from_dict(cls, data: dict) -> TextureSet:
        return cls(
            name=str(data.get("name", "Set")),
            width=int(data["width"]),
            height=int(data["height"]),
            channels=[str(c) for c in data.get("channels") or ["base_color"]],
            layers=[Layer.from_dict(d) for d in data.get("layers") or []],
        )


class TextureDocument:
    """A .tcraft document: title + texture sets."""

    def __init__(self, title: str, sets: list[TextureSet],
                 raw: bytes | None = None) -> None:
        self.title = title
        self.sets = sets
        self._raw = raw
        self._dirty = False
        self._touched: set[int] = set()
        self._snapshot: dict[int, dict[str, dict]] = {}
        if raw is not None:
            self._snapshot = _snapshot_tiles(self)

    # -- dimensions -----------------------------------------------------------
    @property
    def width(self) -> int:
        return self.sets[0].width if self.sets else 0

    @property
    def height(self) -> int:
        return self.sets[0].height if self.sets else 0

    def mark_touched(self, layer_id: int) -> None:
        self._touched.add(layer_id)
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    # -- construction ---------------------------------------------------------
    @classmethod
    def create_blank(cls, title: str, width: int, height: int,
                     channels: list[str] | None = None) -> TextureDocument:
        if width <= 0 or height <= 0 or width > MAX_DIMENSION \
                or height > MAX_DIMENSION:
            raise TextureError(TextureErrorKind.INVALID_ARGUMENT,
                               f"bad dimensions {width}x{height}")
        channels = channels or ["base_color"]
        doc = cls(title, [TextureSet("Body", width, height, channels, [])])
        doc._dirty = True
        return doc

    @classmethod
    def open(cls, path: str | Path) -> TextureDocument:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise TextureError(TextureErrorKind.NOT_FOUND, str(exc)) from exc
        return cls.from_bytes(raw)

    @classmethod
    def from_bytes(cls, raw: bytes) -> TextureDocument:
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise TextureError(TextureErrorKind.BAD_FILE,
                               "not a .tcraft zip") from exc
        with zf:
            names = zf.namelist()
            if "project.json" not in names:
                raise TextureError(TextureErrorKind.BAD_FILE,
                                   "project.json missing")
            try:
                project = json.loads(zf.read("project.json").decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise TextureError(TextureErrorKind.BAD_FORMAT,
                                   "project.json is not valid JSON") from exc
            if project.get("format") != _FORMAT:
                raise TextureError(TextureErrorKind.BAD_FORMAT,
                                   f"not a {_FORMAT} project")
            if project.get("version") != _VERSION:
                raise TextureError(TextureErrorKind.UNSUPPORTED,
                                   f"version {project.get('version')!r}")
            title = str(project.get("title", "Untitled"))
            set_datas = project.get("sets") or []
            if not set_datas or len(set_datas) > MAX_SETS:
                raise TextureError(TextureErrorKind.BAD_FORMAT,
                                   "no texture sets")
            doc_sets: list[TextureSet] = []
            for si, set_data in enumerate(set_datas):
                tset = TextureSet.from_dict(set_data)
                if tset.width <= 0 or tset.height <= 0 \
                        or tset.width > MAX_DIMENSION \
                        or tset.height > MAX_DIMENSION:
                    raise TextureError(TextureErrorKind.BAD_FORMAT,
                                       f"bad set size {tset.width}x{tset.height}")
                if len(tset.layers) > MAX_LAYERS:
                    raise TextureError(TextureErrorKind.BAD_FORMAT,
                                       "too many layers")
                for layer in tset.layers:
                    layer._width = tset.width
                    layer._height = tset.height
                    _load_layer_tiles(zf, si, tset, layer)
                doc_sets.append(tset)
        return cls(title, doc_sets, raw=raw)

    def to_bytes(self) -> bytes:
        """Serialise. An unedited document returns its original bytes
        exactly; otherwise a deterministic fresh serialisation."""
        if not self._dirty and self._raw is not None:
            return self._raw
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED,
                            compresslevel=9) as zf:
            project = {"format": _FORMAT, "version": _VERSION,
                       "title": self.title,
                       "sets": [s.to_dict() for s in self.sets]}
            _write_part(zf, "project.json",
                        json.dumps(project, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))
            for si, tset in enumerate(self.sets):
                for layer in tset.layers:
                    _write_layer_tiles(zf, si, tset, layer)
        out = buf.getvalue()
        self._raw = out
        self._dirty = False
        self._snapshot = _snapshot_tiles(self)
        self._touched = set()
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    # -- fidelity -------------------------------------------------------------
    def fidelity_report(self) -> dict:
        """Compare the live tiles against the snapshot taken at load
        (or at the last save). ``changed_layers`` lists every layer
        whose pixels differ; ``untouched_changed`` counts tiles that
        changed in a layer no editor command touched — a pixel change
        with no recorded edit is reported, never hidden. Added layers
        are reported separately and never count as untouched changes:
        their tiles are new by construction."""
        changed_layers: list[int] = []
        changed_tiles = 0
        untouched_changed = 0
        before_ids = set(self._snapshot)
        now_ids = {layer.id for tset in self.sets for layer in tset.layers}
        added = now_ids - before_ids
        for tset in self.sets:
            for layer in tset.layers:
                now = _layer_tile_hashes(tset, layer)
                before = self._snapshot.get(layer.id, {})
                diffs = 0
                for key, h in now.items():
                    if before.get(key) != h:
                        diffs += 1
                for key in before:
                    if key not in now:
                        diffs += 1
                if diffs:
                    changed_layers.append(layer.id)
                    changed_tiles += diffs
                    if layer.id not in self._touched \
                            and layer.id not in added:
                        untouched_changed += diffs
        return {
            "changed_layers": sorted(changed_layers),
            "added_layers": sorted(added),
            "removed_layers": sorted(before_ids - now_ids),
            "changed_tiles": changed_tiles,
            "untouched_changed": untouched_changed,
            "total_tiles": sum(
                len(_layer_tile_hashes(tset, layer))
                for tset in self.sets for layer in tset.layers),
        }

    def inspect(self) -> dict:
        """Everything an agent changed, in one place (upstream
        AGENTS.md: inspect shows the agent's work)."""
        sets = []
        for tset in self.sets:
            layers = []
            for layer in tset.layers:
                layers.append({
                    "id": layer.id, "name": layer.name,
                    "kind": layer.kind, "visible": layer.visible,
                    "channels": {k: v.to_dict()
                                 for k, v in layer.channels.items()},
                    "fill": layer.fill, "mask": layer.mask,
                    "effects": layer.effects,
                    "allocated_tiles": {
                        ch: img.allocated_count()
                        for ch, img in layer._images.items()},
                })
            sets.append({"name": tset.name, "width": tset.width,
                         "height": tset.height,
                         "channels": tset.channels, "layers": layers})
        return {"title": self.title, "sets": sets,
                "fidelity": self.fidelity_report()}


def _snapshot_tiles(doc: TextureDocument) -> dict[int, dict[str, dict]]:
    snap: dict[int, dict[str, dict]] = {}
    for tset in doc.sets:
        for layer in tset.layers:
            snap[layer.id] = _layer_tile_hashes(tset, layer)
    return snap


def _layer_tile_hashes(tset: TextureSet, layer: Layer) -> dict[str, dict]:
    """Per (channel, tile) sha256 of the tile bytes, for fidelity.

    Fill layers are recorded by their fill *definition* hash, never
    by materialised tiles: a fill's pixels are a pure function of
    its definition, so materialising one on read is not a change,
    and the snapshot must not pay the cost of evaluating it."""
    out: dict[str, dict] = {}
    if layer.kind == "fill" and layer.fill is not None:
        digest = hashlib.sha256(json.dumps(
            layer.fill, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        for channel in tset.channels:
            out[channel] = {"__fill_def__": digest}
        return out
    for channel, img in layer._images.items():
        tiles = {}
        for ty in range(img.tiles_y):
            for tx in range(img.tiles_x):
                t = img.tile_handle(tx, ty)
                if t is not None:
                    tiles[f"{tx}_{ty}"] = hashlib.sha256(bytes(t)).hexdigest()
        out[channel] = tiles
    if layer._mask_image is not None:
        img = layer._mask_image
        tiles = {}
        for ty in range(img.tiles_y):
            for tx in range(img.tiles_x):
                t = img.tile_handle(tx, ty)
                if t is not None:
                    tiles[f"{tx}_{ty}"] = hashlib.sha256(bytes(t)).hexdigest()
        out["__mask__"] = tiles
    return out


def _tile_part_name(si: int, layer_id: int, channel: str,
                    tx: int, ty: int) -> str:
    return f"tiles/{si}/{layer_id}/{channel}/{tx}_{ty}.bin"


def _load_layer_tiles(zf: zipfile.ZipFile, si: int, tset: TextureSet,
                      layer: Layer) -> None:
    # Fill layers hold no stored channel tiles: their pixels are a
    # pure function of the fill definition and are materialised on
    # first access. Pre-creating empty images here would shadow that.
    # (Masks still load below — they are paintable data.)
    if layer.kind != "fill":
        for channel in tset.channels:
            ch = Channel.parse(channel)
            comps = ch.components() if ch is not None else 1
            if layer.kind == "paint":
                comps += 1  # alpha (coverage) component
            img = TileImage(tset.width, tset.height, comps)
            prefix = f"tiles/{si}/{layer.id}/{channel}/"
            for name in zf.namelist():
                if not name.startswith(prefix) or not name.endswith(".bin"):
                    continue
                stem = name[len(prefix):-4]
                try:
                    tx, ty = (int(p) for p in stem.split("_"))
                except ValueError:
                    continue
                data = zf.read(name)
                if len(data) != TILE_AREA * comps * 2:
                    raise TextureError(
                        TextureErrorKind.BAD_FORMAT,
                        f"tile {name} has the wrong size")
                img.set_tile(tx, ty, data)
            layer._images[channel] = img
    mask_prefix = f"tiles/{si}/{layer.id}/mask/"
    mask_names = [n for n in zf.namelist() if n.startswith(mask_prefix)]
    if mask_names:
        mask_img = TileImage(tset.width, tset.height, 1)
        for name in mask_names:
            stem = name[len(mask_prefix):-4]
            try:
                tx, ty = (int(p) for p in stem.split("_"))
            except ValueError:
                continue
            data = zf.read(name)
            if len(data) != TILE_AREA * 2:
                raise TextureError(TextureErrorKind.BAD_FORMAT,
                                   f"mask tile {name} has the wrong size")
            mask_img.set_tile(tx, ty, data)
        layer._mask_image = mask_img


def _write_part(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, _ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    zf.writestr(info, data)


def _write_layer_tiles(zf: zipfile.ZipFile, si: int, tset: TextureSet,
                       layer: Layer) -> None:
    # Fill layers write no channel tiles: their pixels are a pure
    # function of the fill definition in project.json and are
    # re-materialised on load. Only masks (paintable data) are stored.
    if layer.kind != "fill":
        for channel in sorted(layer._images):
            img = layer._images[channel]
            for ty in range(img.tiles_y):
                for tx in range(img.tiles_x):
                    t = img.tile_handle(tx, ty)
                    if t is not None:
                        _write_part(zf, _tile_part_name(si, layer.id, channel,
                                                        tx, ty), bytes(t))
    if layer._mask_image is not None:
        img = layer._mask_image
        for ty in range(img.tiles_y):
            for tx in range(img.tiles_x):
                t = img.tile_handle(tx, ty)
                if t is not None:
                    _write_part(zf, f"tiles/{si}/{layer.id}/mask/{tx}_{ty}.bin",
                                bytes(t))
