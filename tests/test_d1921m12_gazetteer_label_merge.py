"""Docs 19+21 M12 — gazetteer label merge: one place, one entry.

The defect, measured on the real Nigeria pack: a place's
decoded anchor drifts between zooms by the tile grid's
quantization grain (hundreds of metres at country zooms —
the one Port Harcourt feature's sightings sit 150–300 m
apart across z4–z6), and the build's merge window was a flat
~110 m, so the same place landed as several entry rows per
name form and find returned the same city repeatedly, the
"winner" among the duplicates decided by coordinates rather
than prominence.  The merge tolerance is now zoom-aware:
the base window plus one extent unit of the coarser zoom.

Pins:

* a place sighted at several zooms with a drifting anchor
  collapses to ONE row per name form, the coarsest sighting
  winning (the shipped prominence rule), every alias form
  landing on that same single position;
* find via the canonical name and via each alias returns
  exactly one candidate, labelled with the canonical name;
* a query matching several forms of one place yields one
  candidate for the place, carrying the best match class;
* same-stem different places never merge: a station named
  after the city stays its own entry, and two same-name
  villages kilometres apart stay two entries;
* reverse at the place returns it once;
* the tolerance itself: tight at fine zooms (where same-name
  neighbours live), wide enough at coarse zooms to absorb
  the grain the real pack exhibits;
* the real-pack evidence case, when the Nigeria pack is on
  this machine: the sampled build over the tiles containing
  Port Harcourt holds exactly one canonical row for the
  city, all its name forms at one position.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from nomorals.maps.coords import (
    BBox,
    TileCoord,
    lat_to_y_float,
    lon_to_x_float,
)
from nomorals.maps.gazetteer import (
    Gazetteer,
    _extent_unit_m,
    _merge_tolerance_m,
    normalize_name,
)
from nomorals.maps.manifest import (
    PackManifest,
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)
from nomorals.maps.receipts import MapReceiptStore

ATTRIBUTION = "© OpenStreetMap contributors"
NIGERIA = BBox(west=-10.0, south=0.0, east=20.0, north=15.0)
EXTENT = 4096

# Drift City's true position and its per-zoom label anchors —
# the separations (150–300 m of longitude) mirror the real
# pack's measured drift for a capital city across z4–z6.
DRIFT = (4.76805, 7.01889)
DRIFT_SIGHTINGS = (
    (4.76805, 7.02026, 4),
    (4.76805, 7.01752, 5),
    (4.76805, 7.01889, 6),
    (4.76805, 7.01889, 12),
)
STATION = (4.76510, 7.01647)      # Drift City Station — its own place
TWIN_BIG = (8.0200, 4.0300)       # Twin Town, pop 9000
TWIN_SMALL = (8.0000, 4.0000)     # Twin Town, pop 2000 — ~4 km away
DRIFT_VILLAGE = (5.5000, 6.5000)  # a village canonically named "Drift"


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


def _pixel(lat: float, lon: float, zoom: int):
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
                  lon: float, zoom: int = 8, geom_type: int = 1):
        x, y, px, py = _pixel(lat, lon, zoom)
        geometry = _geometry((1, [(px, py)]))
        self.tiles.setdefault((zoom, x, y), {}).setdefault(
            layer, []).append((tags, geom_type, geometry))
        return (zoom, x, y)

    def build(self, root: Path, *, pack_id="drift-geo",
              name="Drift Geo", tile_format="mvt", zoom_max=16):
        root.mkdir(parents=True, exist_ok=True)
        for (z, x, y), layers in self.tiles.items():
            tile = _tile(*(_layer_from_specs(lname, specs)
                           for lname, specs in layers.items()))
            path = root / tile_relpath(z, x, y, tile_format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile)
        manifest = build_manifest(
            root, pack_id=pack_id, name=name, bbox=NIGERIA,
            zoom_min=0, zoom_max=zoom_max, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00",
            tile_format=tile_format)
        write_manifest(root, manifest)
        return load_manifest(root)


def _drift_pack(root: Path):
    b = _PackBuilder()
    tags = {"name": "Drift City", "class": "city",
            "population": 500000, "name:pt": "Porto Drift",
            "name:de": "Drift"}
    for lat, lon, zoom in DRIFT_SIGHTINGS:
        b.add_point("place", dict(tags), lat, lon, zoom=zoom)
    b.add_point("poi", {"name": "Drift City Station",
                        "class": "railway"}, *STATION, zoom=12)
    b.add_point("place", {"name": "Twin Town", "class": "village",
                          "population": 9000}, *TWIN_BIG, zoom=10)
    b.add_point("place", {"name": "Twin Town", "class": "village",
                          "population": 2000}, *TWIN_SMALL, zoom=10)
    b.add_point("place", {"name": "Drift", "class": "village"},
                *DRIFT_VILLAGE, zoom=10)
    return b.build(root)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m12-gaz-")
        self.root = Path(self.tmp.name)
        self.packs = self.root / "packs"
        self.packs.mkdir(parents=True)
        self.store = MapReceiptStore(home=str(self.root / "home"))

    def tearDown(self):
        self.tmp.cleanup()

    def _built(self):
        manifest = _drift_pack(self.packs / "drift")
        gaz = Gazetteer(self.packs, store=self.store)
        result = gaz.build([(self.packs / "drift", manifest)])
        return gaz, result

    def _rows(self, gaz, name):
        conn = gaz._connect(create=False)
        try:
            return conn.execute(
                "SELECT indexed_name, kind, lat, lon, source_zoom"
                " FROM gazetteer_entries WHERE name = ?"
                " ORDER BY indexed_name, source_zoom",
                (name,)).fetchall()
        finally:
            conn.close()


class TestLabelMerge(_Tmp):
    def test_drifting_sightings_collapse_to_one_row_per_form(self):
        gaz, _result = self._built()
        rows = self._rows(gaz, "Drift City")
        # one row per name form (canonical + name:pt + name:de)
        self.assertEqual(
            sorted(r[0] for r in rows),
            ["Drift", "Drift City", "Porto Drift"])
        # every form at ONE position: the coarsest sighting's
        positions = {(round(r[2], 5), round(r[3], 5)) for r in rows}
        self.assertEqual(len(positions), 1)
        self.assertTrue(all(r[4] == 4 for r in rows))

    def test_find_canonical_returns_the_place_once(self):
        gaz, _result = self._built()
        res = gaz.search("drift city", quiet=True)
        names = [c.name for c in res.candidates]
        # the place itself appears exactly once, first (exact);
        # the station is a separate, weaker prefix candidate
        self.assertEqual(names.count("Drift City"), 1)
        cand = res.candidates[0]
        self.assertEqual(cand.name, "Drift City")
        self.assertEqual(cand.matched_as, "exact")
        rows = self._rows(gaz, "Drift City")
        self.assertAlmostEqual(cand.lat, rows[0][2], places=6)
        self.assertAlmostEqual(cand.lon, rows[0][3], places=6)

    def test_find_via_alias_labels_the_place_not_the_alias(self):
        gaz, _result = self._built()
        for alias in ("porto drift",):
            res = gaz.search(alias, quiet=True)
            self.assertEqual(len(res.candidates), 1)
            self.assertEqual(res.candidates[0].name, "Drift City")
            self.assertEqual(res.candidates[0].matched_as, "exact")

    def test_query_matching_several_forms_yields_one_candidate(self):
        gaz, _result = self._built()
        # "drift" exact-matches the name:de form, prefix-matches
        # the canonical form and word-matches name:pt — one
        # place, one candidate, best class, canonical label;
        # the village canonically named "Drift" is its own
        # candidate, and the station follows on kind weight.
        res = gaz.search("drift", quiet=True)
        names = [c.name for c in res.candidates]
        self.assertEqual(names.count("Drift City"), 1)
        self.assertEqual(
            names, ["Drift City", "Drift", "Drift City Station"])
        city = res.candidates[0]
        self.assertEqual(city.matched_as, "exact")

    def test_same_stem_station_stays_its_own_place(self):
        gaz, _result = self._built()
        rows = self._rows(gaz, "Drift City Station")
        self.assertEqual(len(rows), 1)
        res = gaz.search("drift city station", quiet=True)
        self.assertEqual(len(res.candidates), 1)
        self.assertEqual(res.candidates[0].name,
                         "Drift City Station")
        city = gaz.search("drift city", quiet=True)
        # the station survives only as its own weaker candidate,
        # ranked below the city it is named after — never merged
        # into it, never wearing its identity
        names = [c.name for c in city.candidates]
        self.assertEqual(names[0], "Drift City")
        self.assertEqual(names.count("Drift City Station"), 1)
        station_cand = city.candidates[names.index(
            "Drift City Station")]
        self.assertEqual(station_cand.matched_as, "prefix")
        self.assertAlmostEqual(station_cand.lat, STATION[0],
                               places=3)
        self.assertAlmostEqual(station_cand.lon, STATION[1],
                               places=3)

    def test_same_name_distinct_villages_stay_separate(self):
        gaz, _result = self._built()
        rows = self._rows(gaz, "Twin Town")
        self.assertEqual(len(rows), 2)
        res = gaz.search("twin town", quiet=True)
        self.assertEqual(len(res.candidates), 2)
        # population still breaks the tie between the two: the
        # bigger village (encoded at TWIN_BIG) ranks first
        self.assertAlmostEqual(res.candidates[0].lat, TWIN_BIG[0],
                               places=3)
        self.assertAlmostEqual(res.candidates[0].lon, TWIN_BIG[1],
                               places=3)

    def test_reverse_returns_the_place_once(self):
        gaz, _result = self._built()
        res = gaz.reverse(DRIFT[0], DRIFT[1], quiet=True)
        names = [c.name for c in res.candidates]
        self.assertEqual(names.count("Drift City"), 1)
        self.assertEqual(res.candidates[0].name, "Drift City")


class TestTolerance(unittest.TestCase):
    def test_extent_unit_matches_the_tile_grid(self):
        # z4: 40 075 016.686 m / (16 tiles × 4096 units)
        self.assertAlmostEqual(_extent_unit_m(4), 611.5, delta=1.0)
        self.assertAlmostEqual(_extent_unit_m(14), 0.6, delta=0.05)

    def test_tolerance_tight_at_fine_zooms(self):
        # where same-name neighbours actually occur, the window
        # stays near the shipped base — distinct places are safe
        self.assertLess(_merge_tolerance_m(12, 14), 200.0)
        self.assertLess(_merge_tolerance_m(10, 10), 200.0)

    def test_tolerance_absorbs_coarse_zoom_drift(self):
        # the real pack's measured z4–z6 sighting spread is
        # ~304 m; the z4-involving pairs must reach past it
        self.assertGreater(_merge_tolerance_m(4, 6), 304.0)
        self.assertGreater(_merge_tolerance_m(5, 6), 160.0)


REAL_PACK = Path.home() / "workspace" / "maps-packs" / "pack-a-nigeria"


def _tile_members(pack: Path, lat: float, lon: float):
    members = []
    for z in range(0, 15):
        n = 1 << z
        x = int((lon + 180.0) / 360.0 * n)
        lat_r = math.radians(lat)
        y = int((1.0 - math.log(
            math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi)
            / 2.0 * n)
        for fmt in ("mvt", "pbf"):
            rel = f"tiles/{z}/{x}/{y}.{fmt}"
            if (pack / rel).is_file():
                members.append(rel)
    return members


@unittest.skipUnless(
    (REAL_PACK / "manifest.json").is_file(),
    "the real Nigeria pack is not on this machine")
class TestRealPackEvidence(unittest.TestCase):
    def test_port_harcourt_is_one_place_in_the_sampled_build(self):
        manifest = PackManifest.from_dict(
            json.loads((REAL_PACK / "manifest.json").read_text()))
        with tempfile.TemporaryDirectory(prefix="m12-real-") as td:
            gaz = Gazetteer(td)
            rows, _scanned, _corrupt = gaz._scan_members(
                REAL_PACK, manifest,
                _tile_members(REAL_PACK, 4.768, 7.0175))
            merged = gaz._merge_sightings(rows)
        target = normalize_name("Port Harcourt")
        canonical = [r for r in merged if r[3] == target]
        self.assertEqual(len(canonical), 1)
        forms = [r for r in merged if r[1] == "Port Harcourt"]
        self.assertGreaterEqual(len(forms), 10)
        positions = {(round(r[5], 5), round(r[6], 5)) for r in forms}
        self.assertEqual(len(positions), 1)


if __name__ == "__main__":
    unittest.main()
