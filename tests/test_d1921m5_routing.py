"""Docs 19+21 M5 — the routing plane: graph, router, surface.

Pins, in order:

* graph extraction: nodes/arcs counted over a hand-built wire
  fixture; rail features excluded; one arc set per direction
  rule (two-way pairs, a oneway single); a corrupt tile is
  counted, never fatal; the corridor graph is cached digest-
  keyed and a changed pack drops its stale graphs;
* routing: the shortest path on the fixture grid; the oneway
  detour (the direct road is oneway the wrong way AND the rail
  shortcut is not a road — the router takes the long way round,
  distances asserted against haversine ground truth); the
  reverse direction takes the shortcut; steps merge by name and
  unnamed segments are named by class; duration comes off the
  class-speed table; cross-tile joins connect (a road split at a
  tile boundary routes through);
* refusals, typed: snap success inside the bound with the snap
  distance stated; snap-from / snap-to beyond it; disconnected
  graph; no road data; the corridor cap; a raster pack;
* receipts: every attempt mints its OP_ROUTE receipt — ok for a
  route and for the truthful "disconnected", unavailable when
  the plane could not serve;
* federation (doc 21 S3), mirroring M4: off means the local
  answer is final; on with no provider names exactly that; a
  wired provider answers only after the local graph has no
  route, and its failure is named, never fatal;
* the overlay seam: the base render without an overlay is
  byte-identical to the render before the seam; with an overlay
  the route's gold pixels are really on the map;
* the /map route surface: names resolve through the gazetteer,
  raw coordinates work, success sends summary + photo, a
  no-route is honest text with no photo, strangers are refused
  by the standing /map gate.
"""

from __future__ import annotations

import ast
import math
import os
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.location.primitives import LatLon, haversine_m
from nomorals.maps.coords import BBox, TileCoord
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)
from nomorals.maps.pngcodec import decode_png
from nomorals.maps.receipts import OP_ROUTE, MapReceiptStore
from nomorals.maps.routing import (
    GRAPH_FILENAME,
    MAX_CORRIDOR_TILES,
    REASON_CORRIDOR_TOO_LARGE,
    REASON_DISCONNECTED,
    REASON_NO_ROAD_DATA,
    REASON_NOT_VECTOR,
    REASON_SNAP_FROM,
    REASON_SNAP_TO,
    SPEED_KMH,
    Router,
)
from nomorals.maps.settings import map_settings
from nomorals.maps.vector_render import MapOverlay, VectorRenderer

MAPS_DIR = Path(__file__).resolve().parent.parent / "nomorals" / "maps"
ATTRIBUTION = "© OpenStreetMap contributors"
WIDE = BBox(west=-10.0, south=-10.0, east=20.0, north=15.0)
EXTENT = 4096
GOLD = (232, 179, 75)          # the house gold, #e8b34b

Z, TX, TY = 10, 512, 512       # the fixture tile
# Fixture nodes, in tile pixels (extent 4096, ≈9.55 m/px here):
A = (512, 1024)
B = (2048, 1024)
C = (3584, 1024)
D = (2048, 2560)
E = (512, 3584)
F = (1024, 3584)
P1 = (3000, 2048)              # boundary road, west segment start
P2 = (4096, 2048)              # the tile-boundary vertex itself
P3 = (1096, 2048)              # boundary road, east segment end (tile 513)


def _latlon(px: int, py: int, tile_x: int = TX) -> tuple[float, float]:
    """(lat, lon) of a tile pixel — ground truth, computed here
    independently of the module under test."""
    fx = (tile_x + px / EXTENT) / (1 << Z)
    fy = (TY + py / EXTENT) / (1 << Z)
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * fy)))),
            fx * 360.0 - 180.0)


A_LL = _latlon(*A)
B_LL = _latlon(*B)
C_LL = _latlon(*C)
D_LL = _latlon(*D)
E_LL = _latlon(*E)
P1_LL = _latlon(*P1)
P3_LL = _latlon(*P3, tile_x=TX + 1)


def _hav(p: tuple[float, float], q: tuple[float, float]) -> float:
    return haversine_m(LatLon(*p), LatLon(*q))


# ── a tiny real protobuf encoder (fixtures are wire bytes) ───────
# (the M3/M4 in-test encoder idiom, verbatim shapes)


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


def _line(*points) -> list[int]:
    return _geometry((1, [points[0]]), (2, list(points[1:])))


# ── fixture packs ─────────────────────────────────────────────────


class _PackBuilder:
    """Accumulates features into per-tile layer bytes."""

    def __init__(self):
        self.tiles: dict[tuple, dict[str, list]] = {}

    def add(self, layer: str, tags: dict, geom_type: int,
            geometry, tile_x: int = TX, tile_y: int = TY,
            zoom: int = Z):
        self.tiles.setdefault((zoom, tile_x, tile_y), {}) \
            .setdefault(layer, []).append((tags, geom_type, geometry))

    def build(self, root: Path, *, pack_id="net-fixture",
              name="Network Fixture", tile_format="mvt",
              zoom_max=10, extra_members=None):
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
            root, pack_id=pack_id, name=name, bbox=WIDE,
            zoom_min=0, zoom_max=zoom_max, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00",
            tile_format=tile_format)
        write_manifest(root, manifest)
        return load_manifest(root)


def _network_pack(root: Path):
    """The routing fixture: a grid with a trap in it.

    A—B—C is the primary road; B—D a secondary street; D→A a
    ONEWAY tertiary shortcut (digitized D→A); A—D also carries a
    rail line that is not a road; C—D an unnamed service road;
    E—F a lane connected to nothing; and a primary road split
    across the 512/513 tile boundary at one shared vertex.
    Places sit exactly on A, D and E for the surface tests.
    """
    b = _PackBuilder()
    roads = "transportation"
    b.add(roads, {"name": "Primary Road", "class": "primary"}, 2,
          _line(A, B, C))
    b.add(roads, {"name": "South Street", "class": "secondary"}, 2,
          _line(B, D))
    b.add(roads, {"name": "Shortcut Avenue", "class": "tertiary",
                  "oneway": 1}, 2, _line(D, A))
    b.add(roads, {"name": "Nowhere Lane", "class": "minor"}, 2,
          _line(E, F))
    b.add(roads, {"name": "Old Rail", "class": "rail"}, 2,
          _line(A, D))
    b.add(roads, {"class": "service"}, 2, _line(C, D))
    b.add(roads, {"name": "Boundary Road", "class": "primary"}, 2,
          _line(P1, P2))
    b.add(roads, {"name": "Boundary Road", "class": "primary"}, 2,
          _line((0, 2048), P3), tile_x=TX + 1)
    b.add("place", {"name": "Alpha Town", "class": "town"}, 1,
          _geometry((1, [A])))
    b.add("place", {"name": "Delta Village", "class": "village"}, 1,
          _geometry((1, [D])))
    b.add("place", {"name": "Nowhere Hamlet", "class": "hamlet"}, 1,
          _geometry((1, [E])))
    return b.build(root)


def _corrupt_pack(root: Path):
    """One good road tile + one corrupt tile inside the corridor
    the stats call below spans."""
    b = _PackBuilder()
    b.add("transportation", {"name": "Solo Road", "class": "primary"},
          2, _line((3500, 2048), (3900, 2048)))
    return b.build(root, pack_id="corrupt-fixture",
                   name="Corrupt Fixture",
                   extra_members={
                       tile_relpath(Z, TX + 1, TY, "mvt"):
                           b"this is not an mvt payload"})


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m5-route-")
        self.root = Path(self.tmp.name)
        self.packs = self.root / "packs"
        self.packs.mkdir(parents=True)
        self.store = MapReceiptStore(home=str(self.root / "home"))
        self.pack_dir = self.packs / "net"
        self.manifest = _network_pack(self.pack_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _router(self, **kw):
        kw.setdefault("store", self.store)
        return Router(self.packs, **kw)

    def _route(self, start, end, **kw):
        return self._router(**kw).route(
            self.pack_dir, self.manifest, LatLon(*start),
            LatLon(*end))

    def _route_receipts(self):
        return [r for r in self.store.recent(limit=200)
                if r.operation == OP_ROUTE]


# ── graph extraction ──────────────────────────────────────────────


class TestGraphExtraction(_Tmp):
    def test_counts_nodes_arcs_and_directions(self):
        stats = self._router().graph_stats(
            self.pack_dir, self.manifest, LatLon(*A_LL),
            LatLon(*D_LL))
        # nodes: A B C D E F + the boundary road's west pair
        self.assertEqual(stats.node_count, 8)
        # arcs: primary 4 (two-way pairs A-B, B-C) + south 2
        # + shortcut 1 (oneway D->A only) + nowhere 2 + service 2
        # + boundary-west 2 = 13; the rail line contributes none
        self.assertEqual(stats.arc_count, 13)
        self.assertEqual(stats.tiles_present, 1)
        self.assertEqual(stats.corrupt_tiles, 0)
        self.assertEqual(stats.zoom, 10)
        self.assertFalse(stats.cached)

    def test_graph_is_cached_and_digest_keyed(self):
        router = self._router()
        first = router.graph_stats(self.pack_dir, self.manifest,
                                   LatLon(*A_LL), LatLon(*C_LL))
        self.assertFalse(first.cached)
        self.assertTrue((self.packs / GRAPH_FILENAME).is_file())
        second = router.graph_stats(self.pack_dir, self.manifest,
                                    LatLon(*A_LL), LatLon(*C_LL))
        self.assertTrue(second.cached)
        self.assertEqual(first.node_count, second.node_count)

    def test_changed_pack_drops_stale_graphs(self):
        router = self._router()
        self._route(A_LL, C_LL)
        # the pack changes: an extra tile member moves the digest
        extra = self.pack_dir / tile_relpath(9, 100, 60, "mvt")
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"changed")
        manifest2 = build_manifest(
            self.pack_dir, pack_id="net-fixture",
            name="Network Fixture", bbox=WIDE, zoom_min=0,
            zoom_max=10, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00", tile_format="mvt")
        write_manifest(self.pack_dir, manifest2)
        self.assertNotEqual(manifest2.digest, self.manifest.digest)
        router.route(self.pack_dir, manifest2, LatLon(*A_LL),
                     LatLon(*C_LL))
        conn = sqlite3.connect(str(self.packs / GRAPH_FILENAME))
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM routing_graphs"
                " WHERE pack_id = 'net-fixture'").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(rows, 1)   # the stale digest's graph went

    def test_corrupt_tile_is_counted_never_fatal(self):
        pack_dir = self.packs / "corrupt"
        manifest = _corrupt_pack(pack_dir)
        start = _latlon(3500, 2048)
        end = _latlon(500, 2048, tile_x=TX + 1)
        stats = Router(self.packs).graph_stats(
            pack_dir, manifest, LatLon(*start), LatLon(*end))
        self.assertEqual(stats.tiles_present, 2)
        self.assertEqual(stats.corrupt_tiles, 1)
        self.assertEqual(stats.node_count, 2)


# ── routing ───────────────────────────────────────────────────────


class TestRouting(_Tmp):
    def test_shortest_path_merges_steps_and_estimates(self):
        res = self._route(A_LL, C_LL)
        self.assertTrue(res.ok, res.detail)
        expected = _hav(A_LL, B_LL) + _hav(B_LL, C_LL)
        self.assertAlmostEqual(res.distance_m, expected, delta=1.0)
        # A->B->C is one road: the steps merge into a single leg
        self.assertEqual(len(res.steps), 1)
        step = res.steps[0]
        self.assertEqual(step.name, "Primary Road")
        self.assertTrue(step.named)
        self.assertAlmostEqual(step.length_m, expected, delta=1.0)
        # duration is the class-speed estimate, primary = 60 km/h
        self.assertAlmostEqual(
            res.duration_s,
            expected / (SPEED_KMH["primary"] / 3.6), delta=0.5)
        self.assertEqual(res.from_snap_m, 0.0)
        self.assertEqual(res.to_snap_m, 0.0)
        self.assertEqual(res.posture, "local")
        self.assertEqual(res.provider, "local")

    def test_geometry_is_the_full_polyline(self):
        res = self._route(A_LL, C_LL)
        self.assertEqual(len(res.geometry), 3)
        for got, want in zip(res.geometry, (A_LL, B_LL, C_LL),
                             strict=True):
            self.assertAlmostEqual(got[0], want[0], places=9)
            self.assertAlmostEqual(got[1], want[1], places=9)

    def test_oneway_detour_the_long_way_round(self):
        # A->D: the direct line is a oneway the WRONG way (and the
        # other direct line is a railway) — the router detours
        # A->B->D over the primary road and the secondary street.
        res = self._route(A_LL, D_LL)
        self.assertTrue(res.ok, res.detail)
        expected = _hav(A_LL, B_LL) + _hav(B_LL, D_LL)
        self.assertAlmostEqual(res.distance_m, expected, delta=1.0)
        self.assertEqual([s.name for s in res.steps],
                         ["Primary Road", "South Street"])
        self.assertNotIn("Shortcut Avenue",
                         [s.name for s in res.steps])

    def test_oneway_forward_takes_the_shortcut(self):
        res = self._route(D_LL, A_LL)
        self.assertTrue(res.ok, res.detail)
        direct = _hav(D_LL, A_LL)
        self.assertAlmostEqual(res.distance_m, direct, delta=1.0)
        self.assertEqual([s.name for s in res.steps],
                         ["Shortcut Avenue"])
        detour = self._route(A_LL, D_LL)
        self.assertLess(res.distance_m, detour.distance_m)
        self.assertAlmostEqual(
            res.duration_s,
            direct / (SPEED_KMH["tertiary"] / 3.6), delta=0.5)

    def test_unnamed_segment_is_named_by_class(self):
        res = self._route(C_LL, D_LL)
        self.assertTrue(res.ok, res.detail)
        self.assertEqual(len(res.steps), 1)
        self.assertEqual(res.steps[0].name, "service")
        self.assertEqual(res.steps[0].road_class, "service")
        self.assertFalse(res.steps[0].named)

    def test_cross_tile_join_connects(self):
        # the boundary vertex is one node in global pixel space:
        # the road split across tiles 512/513 routes through
        res = self._route(P1_LL, P3_LL)
        self.assertTrue(res.ok, res.detail)
        self.assertAlmostEqual(res.distance_m, _hav(P1_LL, P3_LL),
                               delta=50.0)
        self.assertEqual([s.name for s in res.steps],
                         ["Boundary Road"])

    def test_snap_inside_the_bound_reports_the_gap(self):
        near_a = _latlon(A[0] + 10, A[1])   # ~95 m east of node A
        res = self._route(near_a, C_LL)
        self.assertTrue(res.ok, res.detail)
        self.assertGreater(res.from_snap_m, 40.0)
        self.assertLess(res.from_snap_m, 160.0)
        # the snap leg is reported, never folded into the distance
        self.assertAlmostEqual(
            res.distance_m,
            _hav(A_LL, B_LL) + _hav(B_LL, C_LL), delta=1.0)

    def test_snap_failure_names_the_end(self):
        far = _latlon(128, 128)   # ~9 km from the nearest node
        res = self._route(far, A_LL)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_SNAP_FROM)
        self.assertIn("cannot snap", res.detail)
        res = self._route(A_LL, far)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_SNAP_TO)

    def test_disconnected_graph_answers_no_route(self):
        res = self._route(A_LL, E_LL)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_DISCONNECTED)
        self.assertIn("no connected path", res.detail)
        self.assertEqual(res.from_snap_m, 0.0)

    def test_no_road_data_where_the_pack_holds_no_tiles(self):
        start = _latlon(1024, 1024, tile_x=600)
        end = _latlon(3072, 1024, tile_x=600)
        res = self._route(start, end)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_NO_ROAD_DATA)

    def test_corridor_cap_refuses_by_name(self):
        res = self._route((0.0, -100.0), (0.0, 100.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_CORRIDOR_TOO_LARGE)
        self.assertIn(str(MAX_CORRIDOR_TILES), res.detail)
        with self.assertRaises(ValueError):
            self._router().graph_stats(
                self.pack_dir, self.manifest, LatLon(0.0, -100.0),
                LatLon(0.0, 100.0))

    def test_raster_pack_cannot_route(self):
        pack_dir = self.packs / "raster"
        pack_dir.mkdir(parents=True)
        member = pack_dir / tile_relpath(Z, TX, TY, "png")
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(b"\x89PNG\r\n\x1a\n")
        manifest = build_manifest(
            pack_dir, pack_id="raster-fixture", name="Raster",
            bbox=WIDE, zoom_min=0, zoom_max=10,
            attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00", tile_format="png")
        write_manifest(pack_dir, manifest)
        manifest = load_manifest(pack_dir)
        res = self._router().route(pack_dir, manifest,
                                   LatLon(*A_LL), LatLon(*D_LL))
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_NOT_VECTOR)

    def test_no_store_means_no_receipt_but_a_route(self):
        res = Router(self.packs).route(
            self.pack_dir, self.manifest, LatLon(*A_LL),
            LatLon(*C_LL))
        self.assertTrue(res.ok)
        self.assertIsNone(res.receipt)


# ── receipts ──────────────────────────────────────────────────────


class TestRouteReceipts(_Tmp):
    def test_success_receipt_cites_the_pack(self):
        res = self._route(A_LL, C_LL)
        self.assertIsNotNone(res.receipt)
        receipt = res.receipt
        self.assertEqual(receipt.operation, OP_ROUTE)
        self.assertEqual(receipt.outcome, "ok")
        self.assertEqual(receipt.posture, "offline")
        self.assertEqual(receipt.provider, "local")
        self.assertEqual(receipt.pack_refs, (self.manifest.pack_id,))
        self.assertEqual(receipt.pack_digests,
                         (self.manifest.digest,))
        self.assertIn("m by road", receipt.detail)
        stored = self._route_receipts()
        self.assertEqual([r.receipt_id for r in stored],
                         [receipt.receipt_id])

    def test_disconnected_receipt_is_an_answer(self):
        self._route(A_LL, E_LL)
        stored = self._route_receipts()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].outcome, "ok")
        self.assertIn(REASON_DISCONNECTED, stored[0].detail)

    def test_snap_receipt_is_unavailable(self):
        self._route(_latlon(128, 128), A_LL)
        stored = self._route_receipts()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].outcome, "unavailable")
        self.assertIn(REASON_SNAP_FROM, stored[0].detail)


# ── federation posture ────────────────────────────────────────────


class _FakeProvider:
    name = "fake-router"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    def route(self, start, end):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


class TestRoutingFederation(_Tmp):
    def test_off_means_local_is_final(self):
        res = self._route(A_LL, E_LL)
        self.assertEqual(res.posture, "local")
        self.assertIn("Local-first (federation off)", res.detail)

    def test_on_unwired_names_the_missing_provider(self):
        res = self._route(A_LL, E_LL, federation=True)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_DISCONNECTED)
        self.assertEqual(res.posture, "federated")
        self.assertEqual(res.provider, "none")
        self.assertIn("no route provider is configured", res.detail)
        self.assertEqual(res.receipt.posture, "online")
        self.assertEqual(res.receipt.provider, "none")

    def test_wired_provider_answers_after_local_miss(self):
        from nomorals.maps.routing import RouteResult, RouteStep

        offered = RouteResult(
            ok=True, geometry=(A_LL, E_LL), distance_m=1234.0,
            duration_s=300.0,
            steps=(RouteStep(name="Provider Way",
                             road_class="primary", length_m=1234.0,
                             named=True),),
            detail="provider route")
        provider = _FakeProvider(result=offered)
        res = self._route(A_LL, E_LL, federation=True,
                          provider=provider)
        self.assertEqual(provider.calls, 1)
        self.assertTrue(res.ok)
        self.assertEqual(res.provider, "fake-router")
        self.assertEqual(res.posture, "federated")
        self.assertEqual(res.distance_m, 1234.0)
        self.assertIsNotNone(res.receipt)
        self.assertEqual(res.receipt.provider, "fake-router")
        self.assertEqual(res.receipt.outcome, "ok")

    def test_provider_without_a_route_leaves_local_standing(self):
        provider = _FakeProvider(result=None)
        res = self._route(A_LL, E_LL, federation=True,
                          provider=provider)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REASON_DISCONNECTED)
        self.assertIn("had no route either", res.detail)

    def test_provider_failure_is_named_never_fatal(self):
        provider = _FakeProvider(error=RuntimeError("down"))
        res = self._route(A_LL, E_LL, federation=True,
                          provider=provider)
        self.assertFalse(res.ok)
        self.assertIn("provider fake-router failed", res.detail)

    def test_federated_route_opens_no_socket(self):
        provider = _FakeProvider(result=None)
        with mock.patch.object(
                socket, "socket",
                side_effect=AssertionError("network attempted")):
            res = self._route(A_LL, E_LL, federation=True,
                              provider=provider)
        self.assertFalse(res.ok)


# ── the overlay seam ──────────────────────────────────────────────


def _pixel_at(rgb: bytes, width: int, x: int, y: int):
    at = (y * width + x) * 3
    return tuple(rgb[at:at + 3])


def _count_color(rgb: bytes, color) -> int:
    return sum(1 for at in range(0, len(rgb) - 2, 3)
               if tuple(rgb[at:at + 3]) == color)


class TestOverlayRender(_Tmp):
    def _render(self, overlay=None):
        renderer = VectorRenderer(self.pack_dir, self.manifest)
        bbox = TileCoord(Z, TX, TY).bbox()
        return renderer.render(bbox, Z, size=(512, 512),
                               overlay=overlay)

    def test_base_render_is_byte_identical_without_overlay(self):
        first = self._render()
        second = self._render(overlay=None)
        self.assertIsNotNone(first.image)
        self.assertEqual(first.image, second.image)
        self.assertNotIn("overlay", first.receipt.detail)

    def test_route_gold_is_really_on_the_map(self):
        res = self._route(A_LL, D_LL)
        base = self._render()
        overlay = MapOverlay(
            polylines=(res.geometry,),
            markers=(res.geometry[0], res.geometry[-1]))
        drawn = self._render(overlay=overlay)
        self.assertIsNotNone(drawn.image)
        self.assertNotEqual(drawn.image, base.image)
        self.assertIn("overlay: 1 line(s), 2 marker(s)",
                      drawn.receipt.detail)
        _w, _h, base_rgb = decode_png(base.image)
        width, _h2, rgb = decode_png(drawn.image)
        self.assertLess(_count_color(base_rgb, GOLD), 10)
        self.assertGreater(_count_color(rgb, GOLD), 500)
        # the B->D segment's midpoint (output px 256, 224) is
        # road-colored in the base render, gold in the overlay's
        self.assertNotEqual(_pixel_at(base_rgb, width, 256, 224),
                            GOLD)
        self.assertEqual(_pixel_at(rgb, width, 256, 224), GOLD)
        # endpoint A projects to (128, 128): its marker's gold
        # ring sits 5 px out from the ink centre
        self.assertEqual(_pixel_at(rgb, width, 133, 128), GOLD)

    def test_overlay_colors_are_validated(self):
        with self.assertRaises(ValueError):
            MapOverlay(line_color="not-a-color")


# ── settings ──────────────────────────────────────────────────────


class TestRoutingSettings(unittest.TestCase):
    def test_routing_federation_defaults_off(self):
        self.assertFalse(map_settings(env={}).routing_federation)
        self.assertTrue(map_settings(
            env={"NM_MAPS_ROUTING_FEDERATION": "1"})
            .routing_federation)


# ── the /map route surface ────────────────────────────────────────


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


class TestMapRouteSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m5-map-")
        self.home = Path(self.tmp.name)
        self.packs = self.home / "packs"
        self.packs.mkdir(parents=True)
        _network_pack(self.packs / "net")
        # route-by-name resolves through the gazetteer index;
        # since the universality pass it is built explicitly
        # (queries never build inline), never by the surface
        from nomorals.maps.gazetteer import Gazetteer
        from nomorals.maps.manifest import load_manifest

        Gazetteer(self.packs).build(
            [(self.packs / "net",
              load_manifest(self.packs / "net"))])
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

    def test_route_by_names_sends_summary_and_photo(self):
        out = self.rt._control_map(
            "route Alpha Town to Delta Village",
            chat_key="telegram:1", message=_msg())
        self.assertIn("Alpha Town → Delta Village", out)
        self.assertIn("Primary Road", out)
        self.assertIn("South Street", out)
        self.assertNotIn("Shortcut Avenue", out)  # oneway: no A->D
        self.assertEqual(len(self.rt.gateway.sent), 1)
        sent = self.rt.gateway.sent[0]
        self.assertEqual(sent["kind"], "image")
        self.assertIn("Alpha Town → Delta Village", sent["caption"])
        self.assertIn(ATTRIBUTION, sent["caption"])

    def test_route_reverse_takes_the_shortcut(self):
        out = self.rt._control_map(
            "route Delta Village to Alpha Town",
            chat_key="telegram:1", message=_msg())
        self.assertIn("Shortcut Avenue", out)
        self.assertEqual(len(self.rt.gateway.sent), 1)

    def test_route_by_coordinates(self):
        spec = (f"route {A_LL[0]:.6f},{A_LL[1]:.6f} to "
                f"{D_LL[0]:.6f},{D_LL[1]:.6f}")
        out = self.rt._control_map(spec, chat_key="telegram:1",
                                   message=_msg())
        self.assertIn("🧭", out)
        self.assertNotIn("no route", out)
        self.assertIn("Primary Road", out)

    def test_route_no_route_is_honest_text_no_photo(self):
        out = self.rt._control_map(
            "route Alpha Town to Nowhere Hamlet",
            chat_key="telegram:1", message=_msg())
        self.assertIn("no route", out)
        self.assertIn("no connected path", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_route_unknown_place_is_named(self):
        out = self.rt._control_map(
            "route Atlantis to Alpha Town", message=_msg())
        self.assertIn("no place named", out)
        self.assertEqual(self.rt.gateway.sent, [])

    def test_route_needs_both_ends(self):
        out = self.rt._control_map("route Alpha Town",
                                   message=_msg())
        self.assertIn("usage: /map route", out)

    def test_route_stranger_refused(self):
        out = self.rt._control_map(
            "route Alpha Town to Delta Village",
            message=_msg("telegram:999"))
        self.assertIn("owner-only", out)

    def test_usage_mentions_route(self):
        out = self.rt._control_map("", message=_msg())
        self.assertIn("/map route <place or lat,lon> to", out)


# ── the no-network pin, over the new module ───────────────────────


class TestNoNetworkPinCoversRouting(unittest.TestCase):
    def test_routing_module_has_no_network_imports(self):
        banned = ("socket", "urllib", "requests", "http.client")
        tree = ast.parse((MAPS_DIR / "routing.py").read_text(
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
