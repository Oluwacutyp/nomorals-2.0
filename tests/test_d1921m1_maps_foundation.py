"""Docs 19+21 M1 — Devon Atlas foundation pins.

Schemas (coords/GeoJSON), the pack manifest contract, MapReceipt
minting + read-back, and the fixture renderer proven end-to-end
offline on synthetic packs generated in-test.  No network anywhere:
an AST pin keeps socket/urllib imports out of the whole package.
"""

from __future__ import annotations

import ast
import json
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

from nomorals.maps import (
    BBox,
    FixtureRenderer,
    MapReceipt,
    MapReceiptStore,
    TileCoord,
    build_manifest,
    load_manifest,
    map_settings,
    render_pack_preview,
    tile_relpath,
    validate_geojson,
    verify_pack,
    verify_pack_dir,
    verify_signature,
    write_manifest,
)
from nomorals.maps.manifest import PackError
from nomorals.maps.pngcodec import PngError, decode_png, encode_rgb_png
from nomorals.maps.receipts import OP_PACK_VERIFY, OP_RENDER
from nomorals.maps.render import NO_DATA_RGB

MAPS_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "maps"
ATLAS_BBOX = BBox(west=2.0, south=4.0, east=12.0, north=10.0)
FIXED_AT = 1_700_000_000.0


# ── coordinate schemas ────────────────────────────────────────────

class TestTileCoord:
    def test_origin_at_zoom_1(self):
        assert TileCoord.from_latlon(0.0, 0.0, 1) == TileCoord(1, 1, 1)

    def test_zoom_zero_is_one_tile(self):
        assert TileCoord.from_latlon(7.0, 3.0, 0) == TileCoord(0, 0, 0)
        assert TileCoord(0, 0, 0).bbox().contains(7.0, 3.0)

    def test_pole_clamps_to_edge_row(self):
        assert TileCoord.from_latlon(89.0, 0.0, 4).y == 0
        assert TileCoord.from_latlon(-89.0, 0.0, 4).y == 15

    def test_lon_180_clamps_to_last_column(self):
        assert TileCoord.from_latlon(0.0, 180.0, 2).x == 3

    def test_out_of_range_refused(self):
        for kwargs in ({"z": -1, "x": 0, "y": 0},
                       {"z": 23, "x": 0, "y": 0},
                       {"z": 2, "x": 4, "y": 0},
                       {"z": 2, "x": 0, "y": -1}):
            with pytest.raises(ValueError):
                TileCoord(**kwargs)
        with pytest.raises(ValueError):
            TileCoord.from_latlon(91.0, 0.0, 2)

    def test_parent_children_round_trip(self):
        tile = TileCoord(3, 5, 2)
        assert tile.parent() == TileCoord(2, 2, 1)
        assert tile in tile.parent().children()
        assert TileCoord(0, 0, 0).parent() is None

    def test_tile_bbox_contains_its_point(self):
        tile = TileCoord.from_latlon(7.5, 3.9, 6)
        assert tile.bbox().contains(7.5, 3.9)


class TestBBox:
    def test_south_above_north_refused(self):
        with pytest.raises(ValueError):
            BBox(west=0.0, south=10.0, east=5.0, north=4.0)

    def test_out_of_range_refused(self):
        with pytest.raises(ValueError):
            BBox(west=-181.0, south=0.0, east=0.0, north=1.0)
        with pytest.raises(ValueError):
            BBox(west=0.0, south=0.0, east=0.0, north=91.0)

    def test_antimeridian(self):
        box = BBox(west=170.0, south=-10.0, east=-170.0, north=10.0)
        assert box.crosses_antimeridian
        assert box.width_deg == pytest.approx(20.0)
        assert box.contains(0.0, 175.0)
        assert box.contains(0.0, -175.0)
        assert not box.contains(0.0, 0.0)
        tiles = box.tiles(2)
        assert [t.x for t in tiles if t.y == tiles[0].y] == [3, 0]

    def test_plain_box_tiles_and_contains(self):
        box = BBox(west=-10.0, south=-10.0, east=10.0, north=10.0)
        assert not box.crosses_antimeridian
        assert box.contains(0.0, 0.0)
        assert not box.contains(40.0, 0.0)
        assert box.tiles(1) == (TileCoord(1, 0, 0), TileCoord(1, 1, 0),
                                TileCoord(1, 0, 1), TileCoord(1, 1, 1))


# ── GeoJSON validation ────────────────────────────────────────────

class TestGeoJson:
    def test_valid_geometries(self):
        point = validate_geojson(
            {"type": "Point", "coordinates": [3.9, 7.5]})
        assert point.ok and point.kind == "Point"
        assert point.position_count == 1
        feature = validate_geojson({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[3.0, 7.0], [4.0, 8.0]]},
            "properties": {"name": "road"}})
        assert feature.ok and feature.feature_count == 1
        assert feature.position_count == 2

    def test_bad_position_named(self):
        verdict = validate_geojson(
            {"type": "Point", "coordinates": [200.0, 7.5]})
        assert not verdict.ok
        assert any("longitude" in e for e in verdict.errors)

    def test_non_object_and_unknown_type(self):
        assert not validate_geojson([1, 2, 3]).ok
        verdict = validate_geojson({"type": "Circle", "coordinates": []})
        assert not verdict.ok and verdict.kind == "Circle"

    def test_position_cap_refuses_bomb(self):
        from nomorals.maps.geojson import MAX_POSITIONS
        line = {"type": "LineString",
                "coordinates": [[3.0, 7.0]] * (MAX_POSITIONS + 1)}
        verdict = validate_geojson(line)
        assert not verdict.ok
        assert any("cap" in e for e in verdict.errors)

    def test_feature_cap_refuses_bomb(self):
        from nomorals.maps.geojson import MAX_FEATURES
        fc = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "geometry": None,
                            "properties": {}}] * (MAX_FEATURES + 1)}
        assert not validate_geojson(fc).ok


# ── the PNG codec, pinned independently of the renderer ───────────

def _hand_png(rows: list[tuple[int, bytes]]) -> bytes:
    """A 2-wide RGB PNG built with raw struct/zlib — no codec help."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        head = struct.pack(">I", len(body)) + kind + body
        return head + struct.pack(">I", zlib.crc32(kind + body)
                                  & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 2, len(rows), 8, 2, 0, 0, 0)
    raw = b"".join(bytes([kind]) + body for kind, body in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class TestPngCodec:
    def test_decode_filter_none_and_up(self):
        row0 = bytes([255, 0, 0, 0, 255, 0])
        # row 1 stored as deltas from the row above (filter 2, Up)
        row1 = bytes([1, 0, 255, 255, 0, 255])
        width, height, rgb = decode_png(
            _hand_png([(0, row0), (2, row1)]))
        assert (width, height) == (2, 2)
        assert rgb == bytes([255, 0, 0, 0, 255, 0,
                             0, 0, 255, 255, 255, 255])

    def test_decode_filter_sub(self):
        row = bytes([10, 20, 30, 5, 5, 5])  # second pixel is a delta
        _w, _h, rgb = decode_png(_hand_png([(1, row)]))
        assert rgb == bytes([10, 20, 30, 15, 25, 35])

    def test_encode_round_trip(self):
        rgb = bytes(range(48))
        width, height, back = decode_png(encode_rgb_png(4, 4, rgb))
        assert (width, height) == (4, 4) and back == rgb

    def test_garbage_refused(self):
        with pytest.raises(PngError):
            decode_png(b"this is not a png")

    def test_pil_agrees_when_present(self):
        Image = pytest.importorskip("PIL.Image")
        rgb = bytes([9, 8, 7] * 16)
        image = Image.frombytes("RGB", (4, 4), rgb)
        buf = Path(__file__).with_suffix(".tmp.png")
        try:
            image.save(buf)
            width, height, back = decode_png(buf.read_bytes())
        finally:
            buf.unlink(missing_ok=True)
        assert (width, height) == (4, 4) and back == rgb


# ── fixture packs ─────────────────────────────────────────────────

RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)
YELLOW = (200, 200, 30)
TILE_PX = 8


def _solid_tile(color: tuple[int, int, int]) -> bytes:
    return encode_rgb_png(TILE_PX, TILE_PX,
                          bytes(color) * (TILE_PX * TILE_PX))


def _make_pack(root: Path, tiles: dict[tuple[int, int, int],
                                      tuple[int, int, int]],
               *, pack_id: str = "fixture-a", signature: str = ""):
    for (z, x, y), color in tiles.items():
        path = root / tile_relpath(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_solid_tile(color))
    manifest = build_manifest(
        root, pack_id=pack_id, name="Fixture A", bbox=ATLAS_BBOX,
        zoom_min=1, zoom_max=1, attribution="© OpenStreetMap contributors",
        built_at="2026-11-10T00:00:00+00:00", signature=signature)
    write_manifest(root, manifest)
    return load_manifest(root)


def _world_pack(root: Path) -> None:
    _make_pack(root, {(1, 0, 0): RED, (1, 1, 0): GREEN,
                      (1, 0, 1): BLUE, (1, 1, 1): YELLOW})


def _world_bbox() -> BBox:
    north_west = TileCoord(1, 0, 0).bbox()
    south_east = TileCoord(1, 1, 1).bbox()
    return BBox(west=north_west.west, south=south_east.south,
                east=south_east.east, north=north_west.north)


def _pixel(rgb: bytes, width: int, x: int, y: int) -> tuple:
    at = (y * width + x) * 3
    return tuple(rgb[at:at + 3])


class TestManifest:
    def test_round_trip_and_clean_verify(self, tmp_path):
        manifest = _make_pack(tmp_path, {(1, 0, 0): RED})
        assert manifest.pack_id == "fixture-a"
        assert manifest.digest == manifest.compute_digest()
        loaded = load_manifest(tmp_path)
        assert loaded == manifest
        verdict = verify_pack(tmp_path)
        assert verdict.ok and not verdict.missing
        assert verdict.attribution == "© OpenStreetMap contributors"
        assert verdict.signature_state == "unsigned"

    def test_tampered_member_named(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED, (1, 1, 0): GREEN})
        target = tmp_path / tile_relpath(1, 1, 0)
        target.write_bytes(_solid_tile(BLUE))
        verdict = verify_pack(tmp_path)
        assert not verdict.ok
        assert verdict.mismatched == (tile_relpath(1, 1, 0),)

    def test_missing_member_named(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED, (1, 1, 0): GREEN})
        (tmp_path / tile_relpath(1, 0, 0)).unlink()
        verdict = verify_pack(tmp_path)
        assert not verdict.ok
        assert verdict.missing == (tile_relpath(1, 0, 0),)

    def test_extra_member_named_not_fatal(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED})
        (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
        verdict = verify_pack(tmp_path)
        assert verdict.ok
        assert verdict.extra == ("notes.txt",)

    def test_forged_digest_refused(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED})
        path = tmp_path / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["digest"] = "0" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(PackError):
            load_manifest(tmp_path)

    def test_signature_states_are_honest(self, tmp_path):
        unsigned = _make_pack(tmp_path / "u", {(1, 0, 0): RED})
        assert verify_signature(unsigned) == "unsigned"
        signed = _make_pack(tmp_path / "s", {(1, 0, 0): RED},
                            signature="c2ln")
        assert signed.signature_state == "signed"
        assert verify_signature(signed) == "unverifiable"
        assert verify_signature(signed, lambda m, s: True) == "verified"
        assert verify_signature(signed, lambda m, s: False) == "invalid"


# ── receipts ──────────────────────────────────────────────────────

class _Db:
    """The duck-typed db shape the receipt stores consume."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        self._conn.execute(sql, tuple(params))
        self._conn.commit()

    def query(self, sql, params=()):
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]


class TestReceipts:
    def test_jsonl_mint_and_read_back(self, tmp_path):
        store = MapReceiptStore(home=str(tmp_path))
        first = MapReceipt.mint(OP_PACK_VERIFY, "ok", at=FIXED_AT,
                                pack_refs=("fixture-a",),
                                detail="pack fixture-a: ok")
        second = MapReceipt.mint(OP_RENDER, "ok", at=FIXED_AT + 5.0)
        assert store.append(first) and store.append(second)
        rows = store.recent()
        assert [r.receipt_id for r in rows] == [second.receipt_id,
                                                first.receipt_id]
        assert rows[1].pack_refs == ("fixture-a",)
        only_render = store.recent(operation=OP_RENDER)
        assert [r.receipt_id for r in only_render] == [second.receipt_id]

    def test_db_store_round_trip(self):
        store = MapReceiptStore(db=_Db())
        receipt = MapReceipt.mint(OP_RENDER, "unavailable", at=FIXED_AT)
        assert store.append(receipt)
        rows = store.recent(operation=OP_RENDER)
        assert len(rows) == 1 and rows[0].outcome == "unavailable"

    def test_verify_pack_dir_mints_receipt(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED})
        store = MapReceiptStore(home=str(tmp_path / "home"))
        verdict, receipt = verify_pack_dir(tmp_path, store=store)
        assert verdict.ok and receipt.outcome == "ok"
        assert receipt.operation == OP_PACK_VERIFY
        assert "fixture-a" in receipt.detail
        assert store.recent(operation=OP_PACK_VERIFY)[0].receipt_id \
            == receipt.receipt_id


# ── the fixture renderer ──────────────────────────────────────────

class TestRenderer:
    def test_composites_the_right_tiles(self, tmp_path):
        _world_pack(tmp_path)
        result = render_pack_preview(tmp_path, _world_bbox(), 1,
                                     size=(16, 16))
        assert result.image is not None
        assert result.receipt.outcome == "ok"
        assert result.receipt.operation == OP_RENDER
        assert result.missing_tiles == ()
        width, height, rgb = decode_png(result.image)
        assert (width, height) == (16, 16)
        assert _pixel(rgb, 16, 0, 0) == RED
        assert _pixel(rgb, 16, 15, 0) == GREEN
        assert _pixel(rgb, 16, 0, 15) == BLUE
        assert _pixel(rgb, 16, 15, 15) == YELLOW
        assert _pixel(rgb, 16, 7, 7) == RED
        assert _pixel(rgb, 16, 8, 8) == YELLOW

    def test_missing_tile_is_named_no_data(self, tmp_path):
        _make_pack(tmp_path, {(1, 0, 0): RED, (1, 1, 0): GREEN,
                              (1, 0, 1): BLUE})
        result = render_pack_preview(tmp_path, _world_bbox(), 1,
                                     size=(16, 16))
        assert result.image is not None
        assert result.missing_tiles == (tile_relpath(1, 1, 1),)
        _w, _h, rgb = decode_png(result.image)
        assert _pixel(rgb, 16, 15, 15) == NO_DATA_RGB

    def test_empty_view_is_unavailable_not_an_image(self, tmp_path):
        _make_pack(tmp_path, {})
        result = render_pack_preview(tmp_path, _world_bbox(), 1)
        assert result.image is None
        assert result.receipt.outcome == "unavailable"

    def test_zoom_outside_pack_range(self, tmp_path):
        _world_pack(tmp_path)
        result = render_pack_preview(tmp_path, _world_bbox(), 5)
        assert result.image is None
        assert result.receipt.outcome == "unavailable"
        assert "outside pack range" in result.receipt.detail

    def test_backend_failure_is_a_typed_unavailable(self, tmp_path):
        _world_pack(tmp_path)

        class BoomBackend:
            def decode_png(self, data):
                raise RuntimeError("no png backend on this profile")

            def encode_png(self, width, height, rgb):
                raise RuntimeError("no png backend on this profile")

        manifest = load_manifest(tmp_path)
        renderer = FixtureRenderer(tmp_path, manifest,
                                   backend=BoomBackend())
        result = renderer.render(_world_bbox(), 1, size=(16, 16))
        assert result.image is None
        assert result.receipt.outcome == "unavailable"
        assert "image backend failed" in result.receipt.detail

    def test_render_receipt_lands_in_store(self, tmp_path):
        _world_pack(tmp_path)
        store = MapReceiptStore(home=str(tmp_path / "home"))
        result = render_pack_preview(tmp_path, _world_bbox(), 1,
                                     size=(16, 16), store=store)
        rows = store.recent(operation=OP_RENDER)
        assert rows and rows[0].receipt_id == result.receipt.receipt_id
        assert rows[0].pack_refs == ("fixture-a",)


# ── settings + the offline guarantee ──────────────────────────────

class TestSettingsAndOffline:
    def test_defaults_dark(self):
        settings = map_settings({})
        assert settings.enabled is False and settings.packs_dir == ""

    def test_env_resolution(self):
        settings = map_settings({"NM_MAPS_ENABLED": "true",
                                 "NM_MAPS_PACKS_DIR": " /packs "})
        assert settings.enabled is True
        assert settings.packs_dir == "/packs"

    def test_no_network_imports_in_package(self):
        banned = ("socket", "urllib", "http.client", "requests")
        # Census B9: providers.py is the package's ONE sanctioned
        # network home — the federation rungs (Nominatim, then
        # Overpass), keyless and fallback-only, gated by the
        # owner's NM_MAPS_GEOCODE_FEDERATION policy.  Every other
        # module stays offline; the gazetteer itself never opens
        # a socket (see TestNoNetworkPinCoversGazetteer in m4).
        allowed = {"providers.py"}
        for path in sorted(MAPS_DIR.glob("*.py")):
            if path.name in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    assert not name.startswith(banned), \
                        (path.name, name)
