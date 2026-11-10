"""Docs 19+21 M4 — the local geocoder: gazetteer, queries, surface.

Pins, in order:

* the index build: places extracted from a fixture MVT pack land
  at their known coordinates; multi-zoom duplicates collapse to
  the coarsest appearance; aliases index beside the canonical
  name; a corrupt tile is counted and named, never fatal; tiles
  deeper than the z14 band are not scanned; png packs index to
  zero, stated; an unchanged pack is a digest-keyed no-op and a
  changed pack re-indexes;
* forward matching: exact > prefix > word-boundary > substring,
  kind weight inside a match class, population as tiebreak,
  Yoruba diacritic folding both directions, honest empty results;
* reverse: nearest first with the distance stated, the ladder
  rung reported, honest emptiness beyond 100 km;
* federation (doc 21 S3): off means the local answer is final;
  on with no provider names exactly that; a wired provider
  answers only after the local index misses, and its failure is
  named, never fatal; no code path opens a socket;
* the /map find + /map near surface: first use builds the index
  and says so, a clear winner also sends its map photo, ties do
  not, strangers are refused by the standing /map gate.
"""

from __future__ import annotations

import ast
import gzip
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.maps.coords import (
    BBox,
    TileCoord,
    lat_to_y_float,
    lon_to_x_float,
)
from nomorals.maps.gazetteer import (
    GAZETTEER_MAX_ZOOM,
    Gazetteer,
    GeocodeCandidate,
    normalize_name,
)
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)
from nomorals.maps.receipts import MapReceiptStore

MAPS_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "maps"
ATTRIBUTION = "© OpenStreetMap contributors"
NIGERIA = BBox(west=-10.0, south=0.0, east=20.0, north=15.0)
EXTENT = 4096

# Known places (fixture coordinates — the pack encodes exactly
# these, and the index must hand them back).
ADO = (7.6211, 5.2214)        # Ado Ekiti — city
ILAWE = (7.5957, 5.1040)      # Ìláwé — town
LAGOS = (6.5244, 3.3792)      # Lagos — city
KADO = (9.0765, 7.4898)       # Kado — suburb (Abuja)
EKITI = (7.7000, 5.3000)      # Ekiti — state
ODE_TOWN = (7.1000, 4.9000)   # Ode — town
ODE_VILLAGE = (7.9000, 5.9000)  # Ode — village (polygon)
NEW_TOWN_BIG = (9.5000, 6.0000)   # New Town — village, pop 200000
NEW_TOWN_SMALL = (8.0000, 4.0000)  # New Town — village, pop 2000
CAFE_ILAWE = (7.5960, 5.1050)     # Cafe Ìláwé — poi


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


def _zz(n: int) -> int:
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _packed(nums) -> bytes:
    return b"".join(_varint(int(n)) for n in nums)


def _cmd(command: int, count: int) -> int:
    return (count << 3) | command


def _geometry(*segments, close: bool = False) -> list[int]:
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


def _val_uint(n: int) -> bytes:
    return _fv(5, int(n))


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


def _layer_from_specs(name: str, specs) -> bytes:
    """One layer from (tags, geom_type, geometry) specs, encoded
    against a single merged key/value table."""
    keys: list[str] = []
    values: list[bytes] = []
    features = []
    for tags, geom_type, geometry in specs:
        pairs: list[int] = []
        for key, value in tags.items():
            body = _val_uint(value) if isinstance(value, int) \
                else _val_str(str(value))
            if key not in keys:
                keys.append(key)
            if body not in values:
                values.append(body)
            pairs.extend((keys.index(key), values.index(body)))
        features.append(_feature(tags=tuple(pairs),
                                 geom_type=geom_type,
                                 geometry=geometry))
    return _layer(name, features, keys=tuple(keys),
                  values=tuple(values))


# ── fixture pack construction ─────────────────────────────────────


def _pixel(lat: float, lon: float, zoom: int) -> tuple[int, int, int, int]:
    """(tile_x, tile_y, px, py) placing (lat, lon) at ``zoom``."""
    tile = TileCoord.from_latlon(lat, lon, zoom)
    px = int(round((lon_to_x_float(lon, zoom) - tile.x) * EXTENT))
    py = int(round((lat_to_y_float(lat, zoom) - tile.y) * EXTENT))
    return tile.x, tile.y, max(0, min(EXTENT - 1, px)), \
        max(0, min(EXTENT - 1, py))


class _PackBuilder:
    """Accumulates named features into per-tile layer bytes."""

    def __init__(self):
        self.tiles: dict[tuple, dict[str, list]] = {}

    def add_point(self, layer: str, tags: dict, lat: float,
                  lon: float, zoom: int = 8, geom_type: int = 1,
                  square: bool = False):
        x, y, px, py = _pixel(lat, lon, zoom)
        if square:
            d = 200
            geometry = _geometry(
                (1, [(px - d, py - d)]),
                (2, [(px + d, py - d), (px + d, py + d),
                     (px - d, py + d)]),
                close=True)
            geom_type = 3
        else:
            geometry = _geometry((1, [(px, py)]))
        self.tiles.setdefault((zoom, x, y), {}).setdefault(
            layer, []).append((tags, geom_type, geometry))
        return (zoom, x, y)

    def build(self, root: Path, *, pack_id="nigeria-geo",
              name="Nigeria Geo", extra_members=None,
              tile_format="mvt", zoom_max=16):
        root.mkdir(parents=True, exist_ok=True)
        for (z, x, y), layers in self.tiles.items():
            tile = _tile(*(_layer_from_specs(lname, specs)
                           for lname, specs in layers.items()))
            path = root / tile_relpath(z, x, y, tile_format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile)
        for rel, payload in (extra_members or {}).items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest = build_manifest(
            root, pack_id=pack_id, name=name, bbox=NIGERIA,
            zoom_min=0, zoom_max=zoom_max, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00",
            tile_format=tile_format)
        write_manifest(root, manifest)
        return load_manifest(root)


def _geo_pack(root: Path):
    """The gazetteer fixture pack: known places, one corrupt tile,
    one out-of-band tile, one unnamed feature, one misplaced name."""
    b = _PackBuilder()
    b.add_point("place", {"name": "Ado Ekiti", "class": "city",
                          "population": 350000}, *ADO, zoom=8)
    # the same place again deeper: the index keeps the z8 sighting
    b.add_point("place", {"name": "Ado Ekiti", "class": "city",
                          "population": 350000}, *ADO, zoom=10)
    b.add_point("place", {"name": "Ìláwé", "class": "town",
                          "name:en": "Ilawe Ekiti"}, *ILAWE, zoom=8)
    lagos_tile = b.add_point(
        "place", {"name": "Lagos", "class": "city",
                  "population": 15000000}, *LAGOS, zoom=8)
    b.add_point("place", {"name": "Kado", "class": "suburb"},
                *KADO, zoom=8)
    b.add_point("place", {"name": "Ekiti", "class": "state"},
                *EKITI, zoom=8)
    b.add_point("place", {"name": "Ode", "class": "town"},
                *ODE_TOWN, zoom=8)
    b.add_point("place", {"name": "Ode", "class": "village"},
                *ODE_VILLAGE, zoom=8, square=True)
    b.add_point("place", {"name": "New Town", "class": "village",
                          "population": 200000}, *NEW_TOWN_BIG,
                zoom=8)
    b.add_point("place", {"name": "New Town", "class": "village",
                          "population": 2000}, *NEW_TOWN_SMALL,
                zoom=8)
    b.add_point("poi", {"name": "Cafe Ìláwé", "class": "cafe"},
                *CAFE_ILAWE, zoom=8)
    # deeper than the band: must NOT be indexed
    b.add_point("place", {"name": "Deep Zoom Hamlet",
                          "class": "hamlet"}, 8.5, 5.5, zoom=16)
    # an unnamed feature: nothing to index
    b.add_point("place", {"class": "village"}, 8.2, 4.2, zoom=8)
    # a name in a layer the index does not read
    b.add_point("water", {"name": "Lake Shouldnotappear",
                          "class": "lake"}, 8.8, 6.2, zoom=8)
    corrupt_rel = tile_relpath(9, 100, 60, "mvt")
    manifest = b.build(root, extra_members={
        corrupt_rel: b"this is not an mvt payload"})
    # gzip the Lagos tile: the transport form must index too
    z, x, y = lagos_tile
    lagos_path = root / tile_relpath(z, x, y, "mvt")
    lagos_path.write_bytes(gzip.compress(lagos_path.read_bytes()))
    manifest = build_manifest(
        root, pack_id="nigeria-geo", name="Nigeria Geo", bbox=NIGERIA,
        zoom_min=0, zoom_max=16, attribution=ATTRIBUTION,
        built_at="2026-11-10T00:00:00+00:00", tile_format="mvt")
    write_manifest(root, manifest)
    return load_manifest(root), corrupt_rel


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m4-gaz-")
        self.root = Path(self.tmp.name)
        self.packs = self.root / "packs"
        self.packs.mkdir(parents=True)
        self.store = MapReceiptStore(home=str(self.root / "home"))

    def tearDown(self):
        self.tmp.cleanup()

    def _gazetteer(self, **kw):
        kw.setdefault("store", self.store)
        return Gazetteer(self.packs, **kw)

    def _built(self, **kw):
        manifest, corrupt_rel = _geo_pack(self.packs / "nigeria")
        gaz = self._gazetteer(**kw)
        result = gaz.build([(self.packs / "nigeria", manifest)])
        return gaz, manifest, corrupt_rel, result


# ── normalization ─────────────────────────────────────────────────


class TestNormalize(unittest.TestCase):
    def test_yoruba_diacritics_fold(self):
        self.assertEqual(normalize_name("Ìláwé"), "ilawe")
        self.assertEqual(normalize_name("ÌLÁWÉ"), "ilawe")
        self.assertEqual(normalize_name("Ẹ̀gbá"), "egba")
        self.assertEqual(normalize_name("Ọ̀yọ́"), "oyo")
        self.assertEqual(normalize_name("Ṣaki"), "saki")

    def test_case_space_and_punctuation(self):
        self.assertEqual(normalize_name("  Ado   Ekiti "), "ado ekiti")
        self.assertEqual(normalize_name("Ado-Ekiti"), "ado ekiti")
        self.assertEqual(normalize_name(""), "")


# ── the index build ───────────────────────────────────────────────


class TestBuild(_Tmp):
    def test_entries_land_at_known_coordinates(self):
        gaz, _m, _c, result = self._built()
        self.assertEqual(result.total_entries, 11)
        res = gaz.search("ado ekiti", quiet=True)
        self.assertEqual(len(res.candidates), 1)
        cand = res.candidates[0]
        self.assertAlmostEqual(cand.lat, ADO[0], delta=0.005)
        self.assertAlmostEqual(cand.lon, ADO[1], delta=0.005)
        town = gaz.search("ilawe", quiet=True).candidates[0]
        self.assertAlmostEqual(town.lat, ILAWE[0], delta=0.005)
        self.assertAlmostEqual(town.lon, ILAWE[1], delta=0.005)

    def test_multi_zoom_sighting_collapses_to_coarsest(self):
        gaz, _m, _c, _r = self._built()
        rows = gaz._connect(create=False).execute(
            "SELECT source_zoom, COUNT(*) FROM gazetteer_entries"
            " WHERE name = 'Ado Ekiti' GROUP BY source_zoom"
        ).fetchall()
        self.assertEqual(rows, [(8, 1)])

    def test_alias_indexes_beside_canonical(self):
        gaz, _m, _c, _r = self._built()
        res = gaz.search("ilawe ekiti", quiet=True)
        self.assertEqual(len(res.candidates), 1)
        self.assertEqual(res.candidates[0].name, "Ìláwé")
        self.assertEqual(res.candidates[0].matched_as, "exact")

    def test_corrupt_tile_counted_and_named_not_fatal(self):
        _gaz, _m, corrupt_rel, result = self._built()
        pack = result.packs[0]
        self.assertIn(corrupt_rel, pack.corrupt_tiles)
        self.assertGreater(pack.tiles_scanned, 5)
        receipt = result.receipt
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.operation, "geocode")
        self.assertIn("corrupt: " + corrupt_rel, receipt.detail)

    def test_out_of_band_and_unread_layers_absent(self):
        gaz, _m, _c, _r = self._built()
        self.assertEqual(
            gaz.search("deep zoom hamlet", quiet=True).candidates, ())
        self.assertEqual(
            gaz.search("shouldnotappear", quiet=True).candidates, ())
        self.assertEqual(GAZETTEER_MAX_ZOOM, 14)

    def test_polygon_anchor_is_centroid(self):
        gaz, _m, _c, _r = self._built()
        res = gaz.search("ode", quiet=True)
        village = [c for c in res.candidates if c.kind == "village"]
        self.assertEqual(len(village), 1)
        self.assertAlmostEqual(village[0].lat, ODE_VILLAGE[0],
                               delta=0.01)
        self.assertAlmostEqual(village[0].lon, ODE_VILLAGE[1],
                               delta=0.01)

    def test_digest_noop_and_change_reindexes(self):
        gaz, manifest, _c, first = self._built()
        self.assertTrue(first.built_any)
        again = gaz.ensure([(self.packs / "nigeria", manifest)])
        self.assertFalse(again.built_any)
        self.assertIsNone(again.receipt)
        self.assertTrue(all(p.skipped for p in again.packs))
        # change the pack: one more place, rebuilt manifest
        b = _PackBuilder()
        b.add_point("place", {"name": "Ilesa", "class": "town"},
                    7.63, 4.72, zoom=8)
        extra_root = self.packs / "scratch"
        b.build(extra_root, pack_id="scratch")
        tile_src = next(extra_root.rglob("*.mvt"))
        rel = tile_src.relative_to(extra_root).as_posix()
        dest = self.packs / "nigeria" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(tile_src.read_bytes())
        manifest2 = build_manifest(
            self.packs / "nigeria", pack_id="nigeria-geo",
            name="Nigeria Geo", bbox=NIGERIA, zoom_min=0, zoom_max=16,
            attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00", tile_format="mvt")
        write_manifest(self.packs / "nigeria", manifest2)
        manifest2 = load_manifest(self.packs / "nigeria")
        changed = gaz.ensure([(self.packs / "nigeria", manifest2)])
        self.assertTrue(changed.built_any)
        self.assertEqual(
            len(gaz.search("ilesa", quiet=True).candidates), 1)

    def test_png_pack_indexes_to_zero_stated(self):
        from nomorals.maps.pngcodec import encode_rgb_png

        solid = encode_rgb_png(256, 256, bytes((1, 2, 3)) * 65536)
        root = self.packs / "pngworld"
        root.mkdir(parents=True)
        for x in (0, 1):
            for y in (0, 1):
                path = root / tile_relpath(1, x, y, "png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(solid)
        manifest = build_manifest(
            root, pack_id="png-fixture", name="PNG Fixture",
            bbox=BBox(west=-180.0, south=-85.0, east=180.0,
                      north=85.0),
            zoom_min=1, zoom_max=1, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00", tile_format="png")
        write_manifest(root, manifest)
        manifest = load_manifest(root)
        gaz = self._gazetteer()
        result = gaz.build([(root, manifest)])
        pack = result.packs[0]
        self.assertEqual(pack.entries, 0)
        self.assertIn("no feature data", pack.note)


# ── forward geocode ───────────────────────────────────────────────


class TestSearch(_Tmp):
    def setUp(self):
        super().setUp()
        self.gaz, _m, _c, _r = self._built()

    def test_exact_beats_word_match(self):
        res = self.gaz.search("ekiti", quiet=True)
        self.assertEqual(res.candidates[0].name, "Ekiti")
        self.assertEqual(res.candidates[0].matched_as, "exact")
        names = [c.name for c in res.candidates]
        self.assertIn("Ado Ekiti", names)
        ado = res.candidates[names.index("Ado Ekiti")]
        self.assertEqual(ado.matched_as, "word")

    def test_prefix_beats_substring(self):
        res = self.gaz.search("ado", quiet=True)
        self.assertEqual(res.candidates[0].name, "Ado Ekiti")
        self.assertEqual(res.candidates[0].matched_as, "prefix")
        kado = [c for c in res.candidates if c.name == "Kado"]
        self.assertEqual(len(kado), 1)
        self.assertEqual(kado[0].matched_as, "substring")

    def test_kind_weight_inside_match_class(self):
        res = self.gaz.search("ode", quiet=True)
        kinds = [c.kind for c in res.candidates]
        self.assertEqual(kinds[0], "town")
        self.assertEqual(kinds[1], "village")
        self.assertTrue(
            all(c.matched_as == "exact" for c in res.candidates))

    def test_population_breaks_ties(self):
        res = self.gaz.search("new town", quiet=True)
        self.assertEqual(len(res.candidates), 2)
        self.assertAlmostEqual(res.candidates[0].lat,
                               NEW_TOWN_BIG[0], delta=0.005)
        self.assertAlmostEqual(res.candidates[1].lat,
                               NEW_TOWN_SMALL[0], delta=0.005)
        self.assertGreater(res.candidates[0].score,
                           res.candidates[1].score)

    def test_diacritic_folding_both_directions(self):
        plain = self.gaz.search("ilawe", quiet=True)
        marked = self.gaz.search("Ìláwé", quiet=True)
        self.assertEqual(plain.candidates[0].name, "Ìláwé")
        self.assertEqual(marked.candidates[0].name, "Ìláwé")
        self.assertEqual(plain.candidates[0].lat,
                         marked.candidates[0].lat)

    def test_poi_indexed_and_ranked_below_settlement(self):
        res = self.gaz.search("cafe ilawe", quiet=True)
        self.assertEqual(res.candidates[0].name, "Cafe Ìláwé")
        self.assertEqual(res.candidates[0].kind, "cafe")
        res = self.gaz.search("ilawe", quiet=True)
        self.assertEqual(res.candidates[0].name, "Ìláwé")
        names = [c.name for c in res.candidates]
        self.assertIn("Cafe Ìláwé", names)

    def test_no_match_is_honest_empty(self):
        res = self.gaz.search("atlantis", quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertEqual(res.posture, "local")

    def test_gzipped_tile_indexed(self):
        res = self.gaz.search("lagos", quiet=True)
        self.assertEqual(len(res.candidates), 1)
        self.assertAlmostEqual(res.candidates[0].lat, LAGOS[0],
                               delta=0.005)


# ── reverse geocode ───────────────────────────────────────────────


class TestReverse(_Tmp):
    def setUp(self):
        super().setUp()
        self.gaz, _m, _c, _r = self._built()

    def test_nearest_first_with_distance_stated(self):
        res = self.gaz.reverse(ADO[0], ADO[1], quiet=True)
        self.assertEqual(res.candidates[0].name, "Ado Ekiti")
        self.assertLess(res.candidates[0].distance_km, 0.5)
        self.assertEqual(res.rung_km, 1.0)
        dists = [c.distance_km for c in res.candidates]
        self.assertEqual(dists, sorted(dists))

    def test_nearest_orders_by_distance(self):
        # a point on the far side of Ìláwé town from its cafe:
        # the town is nearer, the cafe second, both honestly
        # distanced (the cafe sits ~150 m from the town point)
        res = self.gaz.reverse(7.590, 5.098, quiet=True)
        self.assertEqual(res.candidates[0].name, "Ìláwé")
        self.assertEqual(res.candidates[1].name, "Cafe Ìláwé")
        self.assertLess(res.candidates[0].distance_km, 2.0)
        self.assertLess(res.candidates[0].distance_km,
                        res.candidates[1].distance_km)

    def test_beyond_the_ladder_is_honest_empty(self):
        res = self.gaz.reverse(-20.0, -60.0, quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertIn("within 100 km", res.note)
        self.assertEqual(res.searched_km, 100.0)

    def test_invalid_point_refused(self):
        with self.assertRaises(ValueError):
            self.gaz.reverse(95.0, 0.0, quiet=True)


# ── federation posture ────────────────────────────────────────────


class _FakeProvider:
    name = "fake-provider"

    def __init__(self, found=(), fail=False):
        self._found = found
        self._fail = fail

    def search(self, query, *, limit=8):
        if self._fail:
            raise RuntimeError("provider down")
        return tuple(self._found[:limit])

    def reverse(self, lat, lon, *, limit=8):
        if self._fail:
            raise RuntimeError("provider down")
        return ()


class TestFederation(_Tmp):
    def test_off_means_local_answer_is_final(self):
        gaz, _m, _c, _r = self._built(federation=False)
        res = gaz.search("atlantis", quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertEqual(res.posture, "local")
        self.assertIn("local-first", res.note)
        self.assertIn("federation off", res.note)

    def test_on_without_provider_names_it(self):
        gaz, _m, _c, _r = self._built(federation=True)
        res = gaz.search("atlantis", quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertEqual(res.posture, "federated")
        self.assertEqual(res.provider, "none")
        self.assertIn("no geocode provider is configured", res.note)

    def test_provider_answers_only_after_local_miss(self):
        found = (GeocodeCandidate(
            name="Atlantis", kind="city", lat=0.0, lon=0.0,
            pack_id="", score=0.0, matched_as="provider"),)
        gaz, _m, _c, _r = self._built(
            federation=True, provider=_FakeProvider(found))
        hit = gaz.search("ado ekiti")
        self.assertEqual(hit.candidates[0].name, "Ado Ekiti")
        self.assertEqual(hit.posture, "local")
        miss = gaz.search("atlantis")
        self.assertEqual(miss.candidates[0].name, "Atlantis")
        self.assertEqual(miss.posture, "federated")
        self.assertEqual(miss.provider, "fake-provider")
        self.assertIsNotNone(miss.receipt)
        self.assertEqual(miss.receipt.provider, "fake-provider")
        self.assertEqual(miss.receipt.posture, "online")

    def test_provider_failure_is_named_not_fatal(self):
        gaz, _m, _c, _r = self._built(
            federation=True, provider=_FakeProvider(fail=True))
        res = gaz.search("atlantis", quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertIn("failed", res.note)
        self.assertIn("local answer stands", res.note)

    def test_no_index_yet_is_unavailable_not_empty_guess(self):
        gaz = self._gazetteer()
        res = gaz.search("ado ekiti")
        self.assertEqual(res.candidates, ())
        self.assertIn("no gazetteer index", res.note)
        self.assertIsNotNone(res.receipt)
        self.assertEqual(res.receipt.outcome, "unavailable")

    def test_quiet_flag_suppresses_receipts(self):
        gaz, _m, _c, _r = self._built()
        before = len(self.store.recent(limit=200))
        gaz.search("ado ekiti", quiet=True)
        gaz.reverse(ADO[0], ADO[1], quiet=True)
        self.assertEqual(len(self.store.recent(limit=200)), before)
        gaz.search("ado ekiti")
        self.assertEqual(len(self.store.recent(limit=200)),
                         before + 1)

    def test_no_socket_is_ever_opened(self):
        gaz, _m, _c, _r = self._built(
            federation=True, provider=_FakeProvider(()))
        with mock.patch.object(
                socket, "socket",
                side_effect=AssertionError("network attempted")):
            res = gaz.search("atlantis", quiet=True)
            rev = gaz.reverse(-20.0, -60.0, quiet=True)
        self.assertEqual(res.candidates, ())
        self.assertEqual(rev.candidates, ())


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


def _surface_pack(packs_root: Path):
    """Places at z8 for the index + water tiles at z12 around Ado
    Ekiti so a winner's photo can really render."""
    b = _PackBuilder()
    b.add_point("place", {"name": "Ado Ekiti", "class": "city",
                          "population": 350000}, *ADO, zoom=8)
    b.add_point("place", {"name": "Ìláwé", "class": "town"},
                *ILAWE, zoom=8)
    b.add_point("place", {"name": "Ode", "class": "town"},
                *ODE_TOWN, zoom=8)
    b.add_point("place", {"name": "Ode", "class": "village"},
                *ODE_VILLAGE, zoom=8)
    pack_root = packs_root / "nigeria"
    manifest = b.build(pack_root, pack_id="nigeria-surface",
                       name="Nigeria Surface", zoom_max=14)
    # z12 water tiles over the 3x3 around Ado Ekiti's z12 tile
    centre = TileCoord.from_latlon(ADO[0], ADO[1], 12)
    water = _layer(
        "water",
        [_feature(tags=(0, 0), geom_type=3,
                  geometry=_geometry(
                      (1, [(256, 256)]),
                      (2, [(3840, 256), (3840, 3840),
                           (256, 3840)]),
                      close=True))],
        keys=("class",), values=(_val_str("lake"),))
    payload = _tile(water)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            path = pack_root / tile_relpath(
                12, centre.x + dx, centre.y + dy, "mvt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    manifest = build_manifest(
        pack_root, pack_id="nigeria-surface", name="Nigeria Surface",
        bbox=NIGERIA, zoom_min=0, zoom_max=14,
        attribution=ATTRIBUTION,
        built_at="2026-11-10T00:00:00+00:00", tile_format="mvt")
    write_manifest(pack_root, manifest)
    return load_manifest(pack_root)


class TestMapFindNearSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m4-map-")
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

    def test_find_builds_index_on_first_use_then_reuses(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("find ilawe", message=_msg())
        self.assertIn("indexed the local gazetteer first", out)
        self.assertIn("Ìláwé — town", out)
        out2 = self.rt._control_map("find ilawe", message=_msg())
        self.assertNotIn("indexed the local gazetteer first", out2)
        self.assertIn("Ìláwé — town", out2)

    def test_find_clear_winner_sends_photo_and_list(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("find ado ekiti",
                                   chat_key="telegram:1",
                                   message=_msg())
        self.assertIn("Ado Ekiti — city", out)
        self.assertIn("Nigeria Surface", out)
        self.assertEqual(len(self.rt.gateway.sent), 1)
        sent = self.rt.gateway.sent[0]
        self.assertEqual(sent["kind"], "image")
        self.assertIn(ATTRIBUTION, sent["caption"])

    def test_find_tie_sends_no_photo(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("find ode", chat_key="telegram:1",
                                   message=_msg())
        self.assertIn("Ode — town", out)
        self.assertIn("Ode — village", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_find_no_match_is_honest(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("find atlantis", message=_msg())
        self.assertIn("no place named 'atlantis'", out)
        self.assertIn("local-first", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_find_federation_on_names_missing_provider(self):
        _surface_pack(self.packs)
        with mock.patch.dict(os.environ,
                             {"NM_MAPS_GEOCODE_FEDERATION": "1"}):
            out = self.rt._control_map("find atlantis",
                                       message=_msg())
        self.assertIn("no geocode provider is configured", out)

    def test_find_without_packs_names_the_empty_shelf(self):
        out = self.rt._control_map("find ado", message=_msg())
        self.assertIn("no map packs installed", out)

    def test_near_lists_distances(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("near 7.6211,5.2214",
                                   message=_msg())
        self.assertIn("named places near", out)
        lines = [ln for ln in out.splitlines() if "Ado Ekiti" in ln]
        self.assertEqual(len(lines), 1)
        self.assertTrue(" km" in lines[0] or " m ·" in lines[0])

    def test_near_bad_input_is_named(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("near nowhere", message=_msg())
        self.assertIn("couldn't parse", out)
        out = self.rt._control_map("near 95,200", message=_msg())
        self.assertIn("bad point", out)

    def test_stranger_refused_for_find_and_near(self):
        _surface_pack(self.packs)
        out = self.rt._control_map("find ado",
                                   message=_msg("telegram:999"))
        self.assertIn("owner-only", out)
        out = self.rt._control_map("near 7.6,5.2",
                                   message=_msg("telegram:999"))
        self.assertIn("owner-only", out)

    def test_usage_mentions_find_and_near(self):
        out = self.rt._control_map("", message=_msg())
        self.assertIn("/map find <place name>", out)
        self.assertIn("/map near <lat,lon>", out)
        out = self.rt._control_map("fly 1,2", message=_msg())
        self.assertIn("unknown /map verb", out)


class TestNoNetworkPinCoversGazetteer(unittest.TestCase):
    def test_gazetteer_module_has_no_network_imports(self):
        banned = ("socket", "urllib", "requests", "http.client")
        tree = ast.parse((MAPS_DIR / "gazetteer.py").read_text(
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
        self.assertEqual(offenders, [], offenders)


if __name__ == "__main__":
    unittest.main()
