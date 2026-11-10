"""Docs 19+21 M3 — Devon style, the vector renderer, /map surface.

Pins:
* the MVT decoder against hand-built REAL wire bytes (a tiny
  protobuf encoder below builds them; assertions are field-by-field),
  and its named-error taxonomy on malformed payloads — no partial
  silent decode exists;
* the Devon Style as data: validation refuses named defects, and
  swapping a color in the document changes the rendered pixel with
  the renderer code untouched;
* the vector renderer really rasterizes (pixel assertions on water
  fill, road stroke color + width, POI markers), names missing and
  corrupt tiles, answers typed unavailables, and states label
  outcomes in its receipts — drawn with a working backend, skipped
  (and counted) without one;
* the /map surface: packs listed with honest verify state (a
  corrupt pack named), views dispatch by tile_format, photos carry
  the pack's attribution in their caption, no-cover answers name
  the shelf, strangers are refused, Atlas-off is stated.

Hermetic: synthetic packs generated in-test, fake gateway, no
network.  The package's no-network AST pin is re-asserted over the
new modules.
"""

from __future__ import annotations

import ast
import gzip
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.maps import (
    BBox,
    MapReceiptStore,
    TileCoord,
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)
from nomorals.maps.mvt import MvtError, decode_mvt
from nomorals.maps.pngcodec import decode_png
from nomorals.maps.receipts import OP_RENDER
from nomorals.maps.render import NO_DATA_RGB
from nomorals.maps.style import (
    DEFAULT_STYLE,
    KIND_FILL,
    KIND_LABEL,
    KIND_LINE,
    KIND_POINT,
    MapStyle,
    StyleError,
    StyleLayer,
    parse_color,
    style_hash,
)
from nomorals.maps.vector_render import LabelSpec, VectorRenderer

MAPS_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "maps"
ATTRIBUTION = "© OpenStreetMap contributors"
WORLD = BBox(west=-180.0, south=-85.0, east=180.0, north=85.0)


# ── a tiny real protobuf encoder (fixtures are wire bytes) ───────


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _fv(no: int, value: int) -> bytes:
    return _varint((no << 3) | 0) + _varint(value)


def _fb(no: int, body: bytes) -> bytes:
    return _varint((no << 3) | 2) + _varint(len(body)) + body


def _ff32(no: int, value: float) -> bytes:
    return _varint((no << 3) | 5) + struct.pack("<f", value)


def _ff64(no: int, value: float) -> bytes:
    return _varint((no << 3) | 1) + struct.pack("<d", value)


def _zz(n: int) -> int:
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _packed(nums) -> bytes:
    return b"".join(_varint(int(n)) for n in nums)


def _cmd(command: int, count: int) -> int:
    return (count << 3) | command


def _geometry(*segments, close: bool = False) -> list[int]:
    """Segments of (command, [(x, y), ...]) in ABSOLUTE tile coords,
    delta-encoded the way the wire carries them."""
    out: list[int] = []
    cx = cy = 0
    for command, points in segments:
        out.append(_cmd(command, len(points)))
        for x, y in points:
            out.append(_zz(x - cx))
            out.append(_zz(y - cy))
            cx, cy = x, y
    if close:
        out.append(_cmd(7, 1))
    return out


def _val_str(s: str) -> bytes:
    return _fb(1, s.encode("utf-8"))


def _feature(*, fid=None, tags=(), geom_type=0, geometry=()) -> bytes:
    body = b""
    if fid is not None:
        body += _fv(1, fid)
    if tags:
        body += _fb(2, _packed(tags))
    body += _fv(3, geom_type)
    if geometry:
        body += _fb(4, _packed(geometry))
    return body


def _layer(name: str, features, keys=(), values=(), extent=4096,
           version=2) -> bytes:
    body = b""
    body += _fb(1, name.encode("utf-8"))
    for feat in features:
        body += _fb(2, feat)
    for key in keys:
        body += _fb(3, key.encode("utf-8"))
    for value in values:
        body += _fb(4, value)
    if extent != 4096:
        body += _fv(5, extent)
    body += _fv(15, version)
    return body


def _tile(*layers) -> bytes:
    return b"".join(_fb(3, layer) for layer in layers)


def _water_tile() -> bytes:
    """One tile: a water polygon (centre square), a motorway line
    across the middle, a named place point at the centre."""
    water = _layer(
        "water",
        [_feature(fid=42, tags=(0, 0), geom_type=3,
                  geometry=_geometry(
                      (1, [(1024, 1024)]),
                      (2, [(3072, 1024), (3072, 3072),
                           (1024, 3072)]),
                      close=True))],
        keys=("class",), values=(_val_str("lake"),))
    road = _layer(
        "transportation",
        [_feature(tags=(0, 0), geom_type=2,
                  geometry=_geometry((1, [(0, 2048)]),
                                     (2, [(4096, 2048)])))],
        keys=("class",), values=(_val_str("motorway"),))
    place = _layer(
        "place",
        [_feature(tags=(0, 0, 1, 1), geom_type=1,
                  geometry=_geometry((1, [(2048, 2048)])))],
        keys=("name", "class"),
        values=(_val_str("Ado Ekiti"), _val_str("city")))
    poi = _layer(
        "poi",
        [_feature(geom_type=1,
                  geometry=_geometry((1, [(1024, 1024)])))])
    return _tile(water, road, place, poi)


TEST_STYLE = MapStyle(
    name="test-style",
    background="#000000",
    no_data="#1a1a20",
    layers=(
        StyleLayer("water", KIND_FILL, fill="#17648f"),
        StyleLayer("transportation", KIND_LINE, stroke="#ffffff",
                   width=2.0,
                   class_strokes=(("motorway", "#e8b34b"),),
                   class_widths=(("motorway", 6.0),)),
        StyleLayer("poi", KIND_POINT, fill="#ff00ff", radius=4.0),
        StyleLayer("place", KIND_LABEL, text_field="name",
                   text_size=12.0, text_color="#ffffff",
                   halo_color="#000000"),
    ))


def _make_pack(root: Path, tiles: dict, *, zoom_min=0, zoom_max=0,
               bbox=WORLD, tile_format="mvt", pack_id="fixture-mvt",
               name="Fixture MVT"):
    root.mkdir(parents=True, exist_ok=True)
    for (z, x, y), payload in tiles.items():
        path = root / tile_relpath(z, x, y, tile_format)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = build_manifest(
        root, pack_id=pack_id, name=name, bbox=bbox,
        zoom_min=zoom_min, zoom_max=zoom_max,
        attribution=ATTRIBUTION,
        built_at="2026-11-10T00:00:00+00:00",
        tile_format=tile_format)
    write_manifest(root, manifest)
    return load_manifest(root)


def _pixel(rgb: bytes, width: int, x: int, y: int) -> tuple:
    at = (y * width + x) * 3
    return tuple(rgb[at:at + 3])


# ── the MVT decoder ───────────────────────────────────────────────


class TestMvtDecode(unittest.TestCase):
    def test_polygon_layer_field_by_field(self):
        tile = decode_mvt(_water_tile())
        self.assertEqual(tile.layer_names(),
                         ("water", "transportation", "place", "poi"))
        water = tile.layer("water")
        self.assertEqual(water.version, 2)
        self.assertEqual(water.extent, 4096)
        feat = water.features[0]
        self.assertEqual(feat.id, 42)
        self.assertEqual(feat.geom_type, "polygon")
        self.assertEqual(feat.tags, {"class": "lake"})
        self.assertEqual(len(feat.paths), 1)
        self.assertTrue(feat.paths[0].closed)
        self.assertEqual(
            feat.paths[0].points,
            ((1024, 1024), (3072, 1024), (3072, 3072), (1024, 3072)))

    def test_point_multipoint_and_line_paths(self):
        points = _layer(
            "place",
            [_feature(geom_type=1,
                      geometry=_geometry((1, [(100, 100),
                                              (200, 200)])))])
        line = _layer(
            "transportation",
            [_feature(geom_type=2,
                      geometry=_geometry((1, [(0, 0)]),
                                         (2, [(10, 10)]),
                                         (1, [(50, 50)]),
                                         (2, [(60, 60)])))])
        tile = decode_mvt(_tile(points, line))
        place = tile.layer("place").features[0]
        self.assertEqual(place.geom_type, "point")
        self.assertEqual([p.points for p in place.paths],
                         [((100, 100),), ((200, 200),)])
        road = tile.layer("transportation").features[0]
        self.assertEqual([p.points for p in road.paths],
                         [((0, 0), (10, 10)), ((50, 50), (60, 60))])
        self.assertFalse(any(p.closed for p in road.paths))

    def test_every_value_type_resolves(self):
        values = (
            _val_str("Ado Ekiti"),                    # 0 string
            _ff32(2, 1.5),                            # 1 float
            _ff64(3, 2.25),                           # 2 double
            _fv(4, (1 << 64) - 5),                    # 3 int64 = -5
            _fv(5, 12345678901234),                   # 4 uint64
            _fv(6, _zz(-77)),                         # 5 sint64
            _fv(7, 1),                                # 6 bool
        )
        keys = ("s", "f", "d", "i", "u", "si", "b")
        tags: list[int] = []
        for at in range(len(keys)):
            tags += [at, at]
        layer = _layer(
            "place",
            [_feature(tags=tuple(tags), geom_type=1,
                      geometry=_geometry((1, [(1, 1)])))],
            keys=keys, values=values)
        feat = decode_mvt(_tile(layer)).layer("place").features[0]
        self.assertEqual(feat.tags["s"], "Ado Ekiti")
        self.assertAlmostEqual(feat.tags["f"], 1.5)
        self.assertAlmostEqual(feat.tags["d"], 2.25)
        self.assertEqual(feat.tags["i"], -5)
        self.assertEqual(feat.tags["u"], 12345678901234)
        self.assertEqual(feat.tags["si"], -77)
        self.assertIs(feat.tags["b"], True)

    def test_non_default_extent_is_carried(self):
        layer = _layer("water", [], extent=512)
        self.assertEqual(decode_mvt(_tile(layer)).layer("water").extent,
                         512)

    def test_empty_payload_is_an_empty_tile(self):
        self.assertEqual(decode_mvt(b"").layers, ())

    # ── the named-error taxonomy ──────────────────────────────────

    def test_truncated_varint_named(self):
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(b"\x80")
        self.assertIn("truncated varint", str(ctx.exception))

    def test_unknown_wire_type_named(self):
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_varint((1 << 3) | 6))
        self.assertIn("unknown wire type 6", str(ctx.exception))

    def test_group_wire_type_named(self):
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_varint((1 << 3) | 3))
        self.assertIn("unsupported wire type 3", str(ctx.exception))

    def test_length_overrun_named(self):
        blob = _varint((3 << 3) | 2) + _varint(100) + b"\x00\x01"
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(blob)
        self.assertIn("overruns the payload", str(ctx.exception))

    def test_geometry_overrun_named(self):
        feat = _feature(geom_type=2,
                        geometry=[_cmd(1, 1), _zz(0), _zz(0),
                                  _cmd(2, 3), _zz(5), _zz(5)])
        layer = _layer("transportation", [feat])
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("geometry command overrun", str(ctx.exception))

    def test_unknown_command_named(self):
        feat = _feature(geom_type=2, geometry=[_cmd(3, 1), _zz(0),
                                               _zz(0)])
        layer = _layer("transportation", [feat])
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("unknown geometry command 3", str(ctx.exception))

    def test_lineto_before_moveto_named(self):
        feat = _feature(geom_type=2,
                        geometry=[_cmd(2, 1), _zz(0), _zz(0)])
        layer = _layer("transportation", [feat])
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("LineTo before any MoveTo", str(ctx.exception))

    def test_closepath_without_path_named(self):
        feat = _feature(geom_type=3, geometry=[_cmd(7, 1)])
        layer = _layer("water", [feat])
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("ClosePath with no open path", str(ctx.exception))

    def test_tag_index_out_of_range_named(self):
        feat = _feature(tags=(5, 0), geom_type=1,
                        geometry=_geometry((1, [(1, 1)])))
        layer = _layer("place", [feat], keys=("name",),
                       values=(_val_str("x"),))
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("tag key index 5 out of range",
                      str(ctx.exception))

    def test_odd_tag_count_named(self):
        feat = _feature(tags=(0,), geom_type=1,
                        geometry=_geometry((1, [(1, 1)])))
        layer = _layer("place", [feat], keys=("name",),
                       values=(_val_str("x"),))
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_tile(layer))
        self.assertIn("odd count", str(ctx.exception))

    def test_layer_without_version_named(self):
        body = _fb(1, b"water")
        with self.assertRaises(MvtError) as ctx:
            decode_mvt(_fb(3, body))
        self.assertIn("no version field", str(ctx.exception))

    def test_gzip_bytes_are_not_decoded_as_mvt(self):
        with self.assertRaises(MvtError):
            decode_mvt(gzip.compress(_water_tile()))


# ── the style document ────────────────────────────────────────────


class TestStyle(unittest.TestCase):
    def test_default_style_shape(self):
        self.assertEqual(DEFAULT_STYLE.name, "devon-dark")
        self.assertEqual(parse_color(DEFAULT_STYLE.background),
                         (10, 13, 19))
        kinds = [layer.kind for layer in DEFAULT_STYLE.layers]
        self.assertEqual(kinds[0], KIND_FILL)
        self.assertEqual(kinds[-1], KIND_LABEL)
        self.assertIn(KIND_LINE, kinds)
        self.assertIn(KIND_POINT, kinds)
        # the style hash is stable over the document's own round-trip
        clone = MapStyle.from_dict(DEFAULT_STYLE.to_dict())
        self.assertEqual(clone, DEFAULT_STYLE)
        self.assertEqual(style_hash(clone), style_hash(DEFAULT_STYLE))

    def test_unknown_kind_refused(self):
        with self.assertRaises(StyleError) as ctx:
            StyleLayer("water", "raster", fill="#17648f")
        self.assertIn("unknown layer kind", str(ctx.exception))

    def test_malformed_colors_refused(self):
        for bad in ("red", "#12345", "17648f", "#gggggg"):
            with self.assertRaises(StyleError):
                StyleLayer("water", KIND_FILL, fill=bad)
        with self.assertRaises(StyleError):
            MapStyle(name="x", background="ink")

    def test_bad_zoom_range_refused(self):
        with self.assertRaises(StyleError) as ctx:
            StyleLayer("water", KIND_FILL, fill="#17648f",
                       min_zoom=9, max_zoom=4)
        self.assertIn("bad zoom range", str(ctx.exception))

    def test_kind_requirements_refused(self):
        with self.assertRaises(StyleError):
            StyleLayer("water", KIND_FILL)            # no fill at all
        with self.assertRaises(StyleError):
            StyleLayer("road", KIND_LINE)             # no stroke
        with self.assertRaises(StyleError):
            StyleLayer("place", KIND_LABEL, text_color="#ffffff")
        with self.assertRaises(StyleError):
            StyleLayer("place", KIND_LABEL, text_field="name")

    def test_class_lookups(self):
        roads = [layer for layer in DEFAULT_STYLE.layers
                 if layer.source_layer == "transportation"
                 and layer.kind == KIND_LINE][-1]
        self.assertEqual(roads.stroke_for({"class": "motorway"}),
                         parse_color("#e8b34b"))
        self.assertEqual(roads.stroke_for({"class": "path"}),
                         parse_color("#3d4652"))
        self.assertGreater(roads.width_for({"class": "motorway"}, 10),
                           roads.width_for({"class": "path"}, 10))
        landcover = DEFAULT_STYLE.layers[0]
        self.assertEqual(landcover.fill_for({"class": "wood"}),
                         parse_color("#142a1e"))
        self.assertEqual(landcover.fill_for({"class": "other"}),
                         parse_color("#101a14"))


# ── the vector renderer ───────────────────────────────────────────


class _RecordingBackend:
    """A text backend that works: paints a white block at each
    label anchor and remembers what it was asked to draw."""

    def __init__(self):
        self.labels: tuple[LabelSpec, ...] = ()

    def available(self) -> bool:
        return True

    def draw_labels(self, width, height, rgb, labels):
        self.labels = labels
        buf = bytearray(rgb)
        for spec in labels:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = int(spec.x) + dx, int(spec.y) + dy
                    if 0 <= x < width and 0 <= y < height:
                        at = (y * width + x) * 3
                        buf[at:at + 3] = b"\xff\xff\xff"
        return bytes(buf)


class _UnavailableBackend:
    def available(self) -> bool:
        return False

    def draw_labels(self, width, height, rgb, labels):
        raise AssertionError("must never be called when unavailable")


class _FailingBackend:
    def available(self) -> bool:
        return True

    def draw_labels(self, width, height, rgb, labels):
        return None


class TestVectorRenderer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m3-vec-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _world_pack(self, tiles=None, style_tiles=None):
        payload = _water_tile() if style_tiles is None else style_tiles
        return _make_pack(self.root,
                          {(0, 0, 0): payload} if tiles is None
                          else tiles)

    def test_water_fill_and_background_pixels(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNotNone(result.image)
        self.assertEqual(result.receipt.outcome, "ok")
        width, height, rgb = decode_png(result.image)
        self.assertEqual((width, height), (256, 256))
        # the water square covers tile coords 1024..3072 of 4096
        # → output 64..192 on both axes (the motorway crosses its
        # middle row, so interior points away from that row are
        # the fill assertions)
        self.assertEqual(_pixel(rgb, 256, 100, 100), (23, 100, 143))
        self.assertEqual(_pixel(rgb, 256, 180, 180), (23, 100, 143))
        self.assertEqual(_pixel(rgb, 256, 3, 3), (0, 0, 0))

    def test_road_stroke_color_and_width(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        _w, _h, rgb = decode_png(result.image)
        gold = (232, 179, 75)
        # the motorway runs along tile y=2048 → output row 128,
        # class width 6 → radius 3
        self.assertEqual(_pixel(rgb, 256, 10, 128), gold)
        self.assertEqual(_pixel(rgb, 256, 10, 130), gold)
        self.assertEqual(_pixel(rgb, 256, 10, 140), (0, 0, 0))

    def test_poi_marker_pixel(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        _w, _h, rgb = decode_png(result.image)
        # the poi sits at tile (1024, 1024) → output (64, 64),
        # drawn after the water fill it overlaps
        self.assertEqual(_pixel(rgb, 256, 64, 64), (255, 0, 255))

    def test_style_as_data_swap_changes_pixels(self):
        manifest = self._world_pack()
        red_style = MapStyle(
            name="test-red", background="#000000", no_data="#1a1a20",
            layers=(StyleLayer("water", KIND_FILL, fill="#ff0000"),))
        renderer = VectorRenderer(self.root, manifest, style=red_style,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        _w, _h, rgb = decode_png(result.image)
        self.assertEqual(_pixel(rgb, 256, 128, 128), (255, 0, 0))
        self.assertEqual(_pixel(rgb, 256, 3, 3), (0, 0, 0))

    def test_labels_drawn_and_receipted(self):
        manifest = self._world_pack()
        backend = _RecordingBackend()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=backend)
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertEqual([spec.text for spec in backend.labels],
                         ["Ado Ekiti"])
        self.assertIn("labels: 1 drawn", result.receipt.detail)
        _w, _h, rgb = decode_png(result.image)
        self.assertEqual(_pixel(rgb, 256, 128, 128), (255, 255, 255))

    def test_labels_skipped_when_backend_unavailable(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNotNone(result.image)
        self.assertIn("labels skipped: no text backend",
                      result.receipt.detail)
        self.assertIn("1 not drawn", result.receipt.detail)
        _w, _h, rgb = decode_png(result.image)
        # geometry still rendered — the interior is the water fill,
        # not a label block and not a crash
        self.assertEqual(_pixel(rgb, 256, 100, 100), (23, 100, 143))

    def test_labels_skipped_when_backend_draw_fails(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_FailingBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNotNone(result.image)
        self.assertIn("labels skipped: the text backend could not draw",
                      result.receipt.detail)

    def test_missing_tiles_are_no_data_and_named(self):
        manifest = _make_pack(
            self.root, {(1, 0, 0): _water_tile()},
            zoom_min=1, zoom_max=1)
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 1, size=(256, 256))
        self.assertIsNotNone(result.image)
        self.assertEqual(len(result.missing_tiles), 3)
        self.assertIn(tile_relpath(1, 1, 1, "mvt"),
                      result.missing_tiles)
        _w, _h, rgb = decode_png(result.image)
        self.assertEqual(_pixel(rgb, 256, 200, 200), NO_DATA_RGB)
        self.assertEqual(_pixel(rgb, 256, 200, 10), NO_DATA_RGB)

    def test_corrupt_tile_named_not_fatal(self):
        manifest = _make_pack(
            self.root, {(1, 0, 0): _water_tile(),
                        (1, 1, 0): b"\xff\xff\xff not mvt"},
            zoom_min=1, zoom_max=1)
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 1, size=(256, 256))
        self.assertIsNotNone(result.image)
        self.assertEqual(result.receipt.outcome, "ok")
        self.assertIn("corrupt: " + tile_relpath(1, 1, 0, "mvt"),
                      result.receipt.detail)
        self.assertIn(tile_relpath(1, 1, 0, "mvt"),
                      result.missing_tiles)

    def test_gzipped_tile_member_renders(self):
        manifest = self._world_pack(
            tiles={(0, 0, 0): gzip.compress(_water_tile())})
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNotNone(result.image)
        _w, _h, rgb = decode_png(result.image)
        self.assertEqual(_pixel(rgb, 256, 100, 100), (23, 100, 143))

    def test_empty_view_is_unavailable(self):
        manifest = _make_pack(self.root, {}, zoom_min=1, zoom_max=1)
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE)
        result = renderer.render(WORLD, 1, size=(256, 256))
        self.assertIsNone(result.image)
        self.assertEqual(result.receipt.outcome, "unavailable")
        self.assertIn("no tiles for this view", result.receipt.detail)

    def test_zoom_outside_pack_range(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE)
        result = renderer.render(WORLD, 5, size=(256, 256))
        self.assertIsNone(result.image)
        self.assertEqual(result.receipt.outcome, "unavailable")
        self.assertIn("outside pack range", result.receipt.detail)

    def test_png_pack_refused_by_vector_renderer(self):
        manifest = _make_pack(self.root, {}, tile_format="png")
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE)
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNone(result.image)
        self.assertIn("vector renderer draws mvt packs",
                      result.receipt.detail)

    def test_receipt_lands_in_store_with_style_hash(self):
        manifest = self._world_pack()
        store = MapReceiptStore(home=str(self.root / "home"))
        renderer = VectorRenderer(self.root, manifest, style=TEST_STYLE,
                                  store=store,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(128, 128))
        rows = store.recent(operation=OP_RENDER)
        self.assertEqual([r.receipt_id for r in rows],
                         [result.receipt.receipt_id])
        self.assertEqual(rows[0].pack_digests, (manifest.digest,))
        self.assertIn("style test-style (" + style_hash(TEST_STYLE)[:12],
                      rows[0].detail)
        self.assertEqual(rows[0].provider, "local")
        self.assertEqual(rows[0].posture, "offline")

    def test_default_style_renders_a_real_pack(self):
        manifest = self._world_pack()
        renderer = VectorRenderer(self.root, manifest,
                                  text_backend=_UnavailableBackend())
        result = renderer.render(WORLD, 0, size=(256, 256))
        self.assertIsNotNone(result.image)
        _w, _h, rgb = decode_png(result.image)
        self.assertEqual(_pixel(rgb, 256, 128, 128),
                         parse_color("#17648f"))
        self.assertEqual(_pixel(rgb, 256, 3, 3), (10, 13, 19))
        self.assertIn("devon-dark", result.receipt.detail)


if __name__ == "__main__":
    unittest.main()


# ── the /map surface ──────────────────────────────────────────────


def _ctx(tmp: str):
    from nomorals.agents.context import build_context
    from nomorals.core.config import load_settings

    return build_context(
        load_settings(overrides={
            "home": tmp,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        }),
        with_executor=False, with_tools=True, with_router=False)


def _msg(key="telegram:1"):
    return SimpleNamespace(chat=SimpleNamespace(key=key), text="",
                           meta={}, sender_id="")


class _Gateway:
    def __init__(self):
        self.sent: list[dict[str, Any]] = []

    def send_file(self, platform, chat, path, *, caption="",
                  kind="file", name=""):
        payload = Path(path).read_bytes()
        self.sent.append({"platform": platform, "path": path,
                          "caption": caption, "kind": kind,
                          "name": name, "bytes": payload,
                          "existed": Path(path).exists()})
        return SimpleNamespace(ok=True)


class TestMapSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m3-map-")
        self.home = Path(self.tmp.name)
        self.packs = self.home / "packs"
        self.packs.mkdir(parents=True)
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = _ctx(str(self.home))
        rt._owner_chats = {"telegram:1"}
        rt._owner_sender_ids = set()
        rt.gateway = _Gateway()
        self.rt = rt
        self.env = mock.patch.dict(os.environ, {
            "NM_MAPS_ENABLED": "1",
            "NM_MAPS_PACKS_DIR": str(self.packs),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _nigeria_pack(self):
        centre = TileCoord.from_latlon(7.6211, 5.2214, 13)
        payload = _water_tile()
        nigeria = BBox(west=-10.0, south=0.0, east=20.0, north=15.0)
        return _make_pack(
            self.packs / "nigeria",
            {(13, centre.x, centre.y): payload},
            zoom_min=0, zoom_max=14, bbox=nigeria,
            pack_id="nigeria-fixture", name="Nigeria Fixture")

    def test_packs_lists_verified_and_corrupt_named(self):
        self._nigeria_pack()
        broken_dir = self.packs / "broken"
        _make_pack(broken_dir, {(0, 0, 0): _water_tile()},
                   pack_id="broken-fixture", name="Broken Fixture")
        # tamper AFTER the manifest is written: verify must catch it
        tile_path = broken_dir / tile_relpath(0, 0, 0, "mvt")
        tile_path.write_bytes(b"tampered")
        junk = self.packs / "junk"
        junk.mkdir()
        (junk / "manifest.json").write_text("{not json",
                                            encoding="utf-8")
        out = self.rt._control_map("packs", message=_msg())
        self.assertIn("✓ verified · Nigeria Fixture", out)
        self.assertIn("z0–14", out)
        self.assertIn(ATTRIBUTION, out)
        self.assertIn("✗ corrupt · Broken Fixture", out)
        self.assertIn(tile_relpath(0, 0, 0, "mvt"), out)
        self.assertIn("✗ junk — corrupt: manifest unreadable", out)

    def test_packs_empty_shelf(self):
        out = self.rt._control_map("packs", message=_msg())
        self.assertIn("no map packs installed", out)

    def test_view_mvt_pack_sends_photo_with_attribution(self):
        self._nigeria_pack()
        out = self.rt._control_map("view 7.6211,5.2214",
                                   chat_key="telegram:1",
                                   message=_msg())
        self.assertEqual(out, "")
        self.assertEqual(len(self.rt.gateway.sent), 1)
        sent = self.rt.gateway.sent[0]
        self.assertEqual(sent["kind"], "image")
        self.assertIn("Nigeria Fixture", sent["caption"])
        self.assertIn(ATTRIBUTION, sent["caption"])
        width, height, _rgb = decode_png(sent["bytes"])
        self.assertEqual((width, height), (768, 768))
        # the temp send copy is gone once sent
        self.assertFalse(Path(sent["path"]).exists())

    def test_view_png_pack_dispatches_fixture_renderer(self):
        from nomorals.maps.pngcodec import encode_rgb_png

        solid = encode_rgb_png(256, 256,
                               bytes((9, 99, 199)) * (256 * 256))
        tiles = {(1, x, y): solid for x in (0, 1) for y in (0, 1)}
        _make_pack(self.packs / "pngworld", tiles, zoom_min=1,
                   zoom_max=1, tile_format="png",
                   pack_id="png-fixture", name="PNG Fixture")
        out = self.rt._control_map("view 0,0 1", chat_key="telegram:1",
                                   message=_msg())
        self.assertEqual(out, "")
        self.assertEqual(len(self.rt.gateway.sent), 1)
        sent = self.rt.gateway.sent[0]
        self.assertIn(ATTRIBUTION, sent["caption"])
        _w, _h, rgb = decode_png(sent["bytes"])
        self.assertEqual(_pixel(rgb, 768, 100, 100), (9, 99, 199))

    def test_view_no_cover_names_the_shelf(self):
        self._nigeria_pack()
        out = self.rt._control_map("view -33.86,151.21",
                                   chat_key="telegram:1",
                                   message=_msg())
        self.assertIn("no installed pack covers", out)
        self.assertIn("Nigeria Fixture", out)
        self.assertIn("nothing was fetched from a provider", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_view_unknown_tile_format_named(self):
        centre = TileCoord.from_latlon(7.6211, 5.2214, 13)
        _make_pack(self.packs / "pmtiles-pack",
                   {(13, centre.x, centre.y): b"whatever"},
                   zoom_min=0, zoom_max=14,
                   bbox=BBox(west=-10.0, south=0.0, east=20.0,
                             north=15.0),
                   tile_format="pmtiles", pack_id="pm-fixture",
                   name="PMTiles Fixture")
        out = self.rt._control_map("view 7.6211,5.2214",
                                   chat_key="telegram:1",
                                   message=_msg())
        self.assertIn("'pmtiles'", out)
        self.assertIn("no shipped renderer", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_stranger_refused_owner_only(self):
        self._nigeria_pack()
        out = self.rt._control_map("packs",
                                   message=_msg("telegram:999"))
        self.assertIn("owner-only", out)
        out = self.rt._control_map("view 7.6211,5.2214",
                                   chat_key="telegram:999",
                                   message=_msg("telegram:999"))
        self.assertIn("owner-only", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_atlas_off_is_stated(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_MAPS_ENABLED", None)
            out = self.rt._control_map("packs", message=_msg())
        self.assertIn("Atlas is off", out)
        self.assertIn("NM_MAPS_ENABLED=1", out)

    def test_usage_and_unknown_verb(self):
        self.assertIn("usage:", self.rt._control_map("", message=_msg()))
        self.assertIn("unknown /map verb",
                      self.rt._control_map("fly 1,2", message=_msg()))

    def test_bad_coordinates_are_named(self):
        out = self.rt._control_map("view nowhere", message=_msg())
        self.assertIn("couldn't parse coordinates", out)
        out = self.rt._control_map("view 95,200", message=_msg())
        self.assertIn("bad view point", out)

    def test_catalog_and_policy_wiring(self):
        from nomorals.social.chat.control import (
            CONTROL_COMMANDS,
            caller_may_run,
            parse_control,
        )

        self.assertIn("map", CONTROL_COMMANDS)
        command = parse_control("/map packs")
        self.assertIsNotNone(command)
        self.assertEqual(command.kind, "map")
        self.assertEqual(command.tail, "packs")
        # owner-only: not a public command, so the policy denies
        # strangers before the handler is ever reached
        self.assertTrue(caller_may_run("map", is_owner=True))
        self.assertFalse(caller_may_run("map", is_owner=False))


class TestNoNetworkPinCoversNewModules(unittest.TestCase):
    def test_new_modules_have_no_network_imports(self):
        banned = ("socket", "urllib", "requests", "http.client")
        for name in ("mvt.py", "style.py", "vector_render.py"):
            tree = ast.parse((MAPS_DIR / name).read_text(
                encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            offenders = [m for m in imported
                         if any(m == b or m.startswith(b + ".")
                                for b in banned)]
            self.assertEqual(offenders, [], f"{name}: {offenders}")
