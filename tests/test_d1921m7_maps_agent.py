"""Devon Atlas M7 — the MapAgent, the map tool, overlays,
agency wiring and the /map overlay verbs.

M7 turns the shipped Atlas planes (M1 packs + receipts, M3
renderer, M4 gazetteer, M5 routing + the overlay seam, M6
broker) into an agent surface: ``agents/map.py`` (MapAgent —
typed ticket methods over the same cores the /map surface
calls), ``tools/map.py`` (ONE tool, operation dispatch, the
decoder_agent precedent — ticket dicts in the T2 idiom, receipt
ids cited), ``maps/overlays.py`` (the owner's overlay documents
as validated data, stored file-or-table in the glossary idiom,
converted to M5's MapOverlay), the agency wiring (MapAgent
composed beside its peers) and the /map overlay chat verbs.

Pins:
* the tool is registered in the builtin registry and forwards
  to the agent (tool result == agent result on shared fields);
  unknown operations and bad parameters are ToolErrors.
* every operation's ticket against one fixture network pack:
  packs (verify state + a pack_verify receipt id), find
  (exact place, honest empty), near (distances stated), route
  (steps + receipt; disconnected truthfully unrouted), view
  (a real PNG on disk + a render receipt).
* overlay store CRUD + validation refusals (bad positions
  named; feature/point caps enforced; duplicate names refused;
  delete is a real delete).
* the GeoJSON gate: a valid FeatureCollection converts
  (Point/LineString/Polygon, [lon,lat] → (lat,lon), holes
  noted); hostile GeoJSON is refused whole with the gate's
  errors.
* overlay → MapOverlay conversion: polygons arrive as CLOSED
  outlines plus an explicit no-fill note — never silently
  dropped, never pretending a fill the seam lacks.
* /map overlay verbs end to end on the PartnerRuntime harness:
  add → list → show (a photo whose pixels carry the gold
  marker; caption carries the overlay name + the pack's
  attribution; a polygon overlay's caption carries the outline
  note) → drop; strangers refused owner-only.
* the agency composition resolves the map agent + a readiness
  row, and its methods return the agent results.

Offline: fixture packs only; no sockets anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.context import build_context
from nomorals.core.config import load_settings
from nomorals.maps.coords import (
    BBox,
    tile_x_to_lon,
    tile_y_to_lat,
)
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    write_manifest,
)
from nomorals.maps.overlays import (
    MAX_OVERLAY_FEATURES,
    OverlayError,
    OverlayFeature,
    OverlayStore,
    geojson_to_features,
    overlay_to_map_overlay,
    validate_features,
)
from nomorals.maps.pngcodec import decode_png
from nomorals.storage.db import Database

MARKER_GOLD = (232, 179, 75)  # #e8b34b, the MapOverlay default


def _gold_pixels(png: bytes) -> int:
    """Pixels of exactly the overlay marker gold in a PNG."""
    _w, _h, rgb = decode_png(png)
    gold = bytes(MARKER_GOLD)
    return sum(1 for i in range(0, len(rgb) - 2, 3)
               if rgb[i:i + 3] == gold)


# ── the tiny MVT encoder + network pack (the M5 fixture
# machinery, verbatim idiom) ──────────────────────────────────


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


# ── the fixture network pack ─────────────────────────────────
#
# Tile z10 x512 y512, extent 4096.  Nodes (tile units):
#   A(512,1024) B(2048,1024) C(3584,1024) — Primary Road
#   D(2048,2560) — South Street from B; ONEWAY Shortcut D→A
#   E(512,3584) F(1024,3584) — Nowhere Lane (disconnected)

EXTENT = 4096
TX, TY = 512, 512
ATTRIBUTION = "© Network Fixture contributors"

A = (512, 1024)
B = (2048, 1024)
C = (3584, 1024)
D = (2048, 2560)
E = (512, 3584)
F = (1024, 3584)


def _latlon(px: float, py: float, tile_x: int = TX):
    lon = tile_x_to_lon(tile_x, 10) + (px / EXTENT) * (
        tile_x_to_lon(tile_x + 1, 10) - tile_x_to_lon(tile_x, 10))
    lat = tile_y_to_lat(TY + py / EXTENT, 10)
    return (lat, lon)


A_LL = _latlon(*A)
D_LL = _latlon(*D)


def _network_pack(root: Path, pack_id="netpack"):
    tiles: dict[tuple, dict[str, list]] = {}

    def add(layer, tags, geom_type, geometry):
        tiles.setdefault((10, TX, TY), {}).setdefault(
            layer, []).append((tags, geom_type, geometry))

    roads = "transportation"
    add(roads, {"name": "Primary Road", "class": "primary"}, 2,
        _line(A, B, C))
    add(roads, {"name": "South Street", "class": "secondary"}, 2,
        _line(B, D))
    add(roads, {"name": "Shortcut Avenue", "class": "tertiary",
                "oneway": 1}, 2, _line(D, A))
    add(roads, {"name": "Nowhere Lane", "class": "minor"}, 2,
        _line(E, F))
    add(roads, {"name": "Old Rail", "class": "rail"}, 2,
        _line(A, D))
    add(roads, {"class": "service"}, 2, _line(C, D))
    add("place", {"name": "Alpha Town", "class": "town"}, 1,
        _geometry((1, [A])))
    add("place", {"name": "Delta Village", "class": "village"}, 1,
        _geometry((1, [D])))
    add("place", {"name": "Nowhere Hamlet", "class": "hamlet"}, 1,
        _geometry((1, [E])))
    for (z, x, y), layers in tiles.items():
        tile = _tile(*(_layer_from_specs(lname, specs)
                       for lname, specs in layers.items()))
        path = root / f"tiles/{z}/{x}/{y}.mvt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tile)
    manifest = build_manifest(
        root, pack_id=pack_id, name="Network Fixture",
        bbox=BBox(west=-10.0, south=-10.0, east=20.0, north=15.0),
        zoom_min=0, zoom_max=10, tile_format="mvt",
        attribution=ATTRIBUTION)
    write_manifest(root, manifest)
    return load_manifest(root)


@pytest.fixture()
def shelf(tmp_path, monkeypatch):
    """A context + packs dir holding the network pack."""
    home = tmp_path / "home"
    home.mkdir()
    packs = tmp_path / "packs"
    pack_dir = packs / "netpack"
    pack_dir.mkdir(parents=True)
    _network_pack(pack_dir)
    monkeypatch.setenv("NM_MAPS_ENABLED", "1")
    monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs))
    settings = load_settings(overrides={
        "home": str(home), "partner.platforms": "local",
        "chat.local_enabled": "true"})
    ctx = build_context(settings, with_executor=False,
                        with_tools=True, with_router=False)
    return SimpleNamespace(ctx=ctx, home=home, packs=packs)


def _agent(shelf):
    from nomorals.agents.map import MapAgent

    return MapAgent(shelf.ctx)


# ── overlay store ────────────────────────────────────────────


class TestOverlayStore:
    def test_crud_round_trip_file_store(self, tmp_path):
        store = OverlayStore(home=str(tmp_path))
        overlay = store.create("stops", [
            OverlayFeature("marker", ((7.62, 5.22),), "home"),
            OverlayFeature("polyline",
                           ((7.62, 5.22), (7.63, 5.23)), "run"),
        ])
        assert overlay.overlay_id.startswith("ov-")
        assert overlay.created_at
        got = store.get("stops")
        assert got is not None and got.name == "stops"
        assert got.features[0].points == ((7.62, 5.22),)
        assert got.features[1].label == "run"
        # exact-then-case-insensitive lookup
        assert store.get("STOPS") is not None
        assert [o.name for o in store.list()] == ["stops"]
        # the document is on disk as data (id-keyed payloads)
        doc = json.loads(
            (tmp_path / "maps" / "overlays.json").read_text())
        assert [row["name"] for row in doc.values()] == ["stops"]
        # delete is a real delete
        assert store.delete("stops") is True
        assert store.get("stops") is None
        assert store.delete("stops") is False

    def test_crud_round_trip_db_store(self, tmp_path):
        db = Database(str(tmp_path / "rt.db"))
        db.migrate()
        store = OverlayStore(db=db)
        store.create("a", [OverlayFeature("marker", ((1.0, 2.0),))])
        assert store.get("a") is not None
        assert store.delete("a") is True
        assert store.get("a") is None

    def test_duplicate_name_refused_case_insensitively(
            self, tmp_path):
        store = OverlayStore(home=str(tmp_path))
        store.create("Stops", [
            OverlayFeature("marker", ((1.0, 2.0),))])
        with pytest.raises(OverlayError) as err:
            store.create("stops", [
                OverlayFeature("marker", ((3.0, 4.0),))])
        assert "already exists" in str(err.value)

    def test_validation_refusals_name_the_defect(self, tmp_path):
        # validate_features NAMES every defect (returns them);
        # create() refuses with them.
        errors = validate_features([
            OverlayFeature("marker", ((91.0, 5.0),), "")])
        assert any("latitude" in e for e in errors)
        errors = validate_features([
            OverlayFeature("marker",
                           ((1.0, 2.0), (3.0, 4.0)), "")])
        assert any("exactly 1 point" in e for e in errors)
        errors = validate_features([
            OverlayFeature("polygon", ((1.0, 2.0),), "")])
        assert any("at least 3 points" in e for e in errors)
        errors = validate_features([
            OverlayFeature("blob", ((1.0, 2.0),), "")])
        assert any("unknown kind" in e for e in errors)
        assert validate_features([])  # empty is a defect too
        store = OverlayStore(home=str(tmp_path))
        with pytest.raises(OverlayError) as err:
            store.create("bad", [
                OverlayFeature("marker", ((91.0, 5.0),), "")])
        assert "latitude" in str(err.value)
        assert store.list() == ()

    def test_caps_enforced(self):
        many = [OverlayFeature("marker", ((1.0, 2.0),), "")
                for _ in range(MAX_OVERLAY_FEATURES + 1)]
        errors = validate_features(many)
        assert any("feature count exceeds the cap" in e
                   for e in errors)
        long_line = OverlayFeature(
            "polyline",
            tuple((1.0 + i * 0.000001, 2.0) for i in range(20001)),
            "")
        errors = validate_features([long_line])
        assert any("point count exceeds the cap" in e
                   for e in errors)

    def test_corrupt_store_reads_empty_never_raises(self, tmp_path):
        path = tmp_path / "maps" / "overlays.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ not json")
        store = OverlayStore(home=str(tmp_path))
        assert store.list() == ()
        assert store.get("anything") is None


# ── the GeoJSON gate ─────────────────────────────────────────


class TestGeoJsonGate:
    def test_valid_collection_converts(self):
        doc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point",
                              "coordinates": [5.2214, 7.6211]},
                 "properties": {"name": "Home"}},
                {"type": "Feature",
                 "geometry": {"type": "LineString",
                              "coordinates": [[5.22, 7.62],
                                              [5.23, 7.63]]},
                 "properties": {"label": "run"}},
                {"type": "Feature",
                 "geometry": {"type": "Polygon",
                              "coordinates": [
                                  [[5.20, 7.60], [5.24, 7.60],
                                   [5.24, 7.64], [5.20, 7.64],
                                   [5.20, 7.60]],
                                  [[5.21, 7.61], [5.22, 7.61],
                                   [5.22, 7.62], [5.21, 7.62],
                                   [5.21, 7.61]]]},
                 "properties": {"title": "farm"}},
            ],
        }
        features, notes = geojson_to_features(doc)
        kinds = [f.kind for f in features]
        assert kinds == ["marker", "polyline", "polygon"]
        # [lon, lat] in GeoJSON becomes (lat, lon) in the store
        assert features[0].points == ((7.6211, 5.2214),)
        assert features[0].label == "Home"
        assert features[1].label == "run"
        # the polygon carries its outer ring only, holes noted
        assert len(features[2].points) == 5
        assert any("hole" in n for n in notes)

    def test_multi_geometries_fan_out(self):
        doc = {"type": "Feature",
               "geometry": {"type": "MultiPoint",
                            "coordinates": [[5.0, 7.0],
                                            [5.1, 7.1]]},
               "properties": {}}
        features, _notes = geojson_to_features(doc)
        assert [f.kind for f in features] == ["marker", "marker"]

    def test_hostile_geojson_refused_whole(self):
        # a position outside the world: the gate names it and
        # NOTHING converts — the layer is refused whole.
        doc = {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "geometry": {"type": "Point",
                          "coordinates": [5.0, 91.0]},
             "properties": {}}]}
        with pytest.raises(OverlayError) as err:
            geojson_to_features(doc)
        assert "latitude" in str(err.value)
        # an unknown geometry type is refused by name.
        doc2 = {"type": "Feature",
                "geometry": {"type": "Circle",
                             "coordinates": [[5.0, 7.0]]},
                "properties": {}}
        with pytest.raises(OverlayError) as err:
            geojson_to_features(doc2)
        assert "Circle" in str(err.value)

    def test_feature_bomb_refused_by_the_gate(self):
        bomb = {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "geometry": {"type": "Point",
                          "coordinates": [5.0, 7.0]},
             "properties": {}} for _ in range(10001)]}
        with pytest.raises(OverlayError) as err:
            geojson_to_features(bomb)
        assert "features" in str(err.value)


# ── overlay → MapOverlay conversion honesty ──────────────────


class TestOverlayConversion:
    def test_marker_polyline_polygon(self):
        from nomorals.maps.overlays import Overlay

        overlay = Overlay("ov-test", "x", 0.0, (
            OverlayFeature("marker", ((7.62, 5.22),), "home"),
            OverlayFeature("polyline",
                           ((7.62, 5.22), (7.63, 5.23)), "run"),
            OverlayFeature(
                "polygon",
                ((7.60, 5.20), (7.64, 5.20), (7.64, 5.24),
                 (7.60, 5.24)), "farm"),
        ))
        converted, notes = overlay_to_map_overlay(overlay)
        assert converted.markers == ((7.62, 5.22),)
        # polyline + the polygon's closed outline
        assert len(converted.polylines) == 2
        ring = converted.polylines[1]
        assert ring[0] == ring[-1] == (7.60, 5.20)
        assert len(ring) == 5
        # the honesty note: outline only, no fill claimed
        assert len(notes) == 1
        assert "outline" in notes[0] and "farm" in notes[0]

    def test_closed_polygon_not_double_closed(self):
        from nomorals.maps.overlays import Overlay

        ring = ((7.60, 5.20), (7.64, 5.20), (7.64, 5.24),
                (7.60, 5.20))
        overlay = Overlay("ov-test", "x", 0.0,
                          (OverlayFeature("polygon", ring, "x"),))
        converted, _notes = overlay_to_map_overlay(overlay)
        assert converted.polylines[0] == ring


# ── MapAgent operations over the fixture pack ────────────────


class TestMapAgent:
    def test_packs_ticket(self, shelf):
        ticket = _agent(shelf).packs()
        assert ticket["operation"] == "packs"
        assert ticket["outcome"] == "ok"
        assert len(ticket["packs"]) == 1
        pack = ticket["packs"][0]
        assert pack["pack_id"] == "netpack"
        assert pack["verify"]["ok"] is True
        assert pack["attribution"] == "© Network Fixture contributors"
        assert ticket["receipt_ids"]  # a pack_verify receipt rode

    def test_find_and_near(self, shelf):
        agent = _agent(shelf)
        found = agent.find("Alpha Town")
        assert found["outcome"] == "ok"
        top = found["candidates"][0]
        assert top["name"] == "Alpha Town"
        assert top["matched_as"] == "exact"
        assert abs(top["lat"] - A_LL[0]) < 0.01
        assert found["receipt_id"]  # the geocode receipt
        miss = agent.find("Zxqv Notaplace")
        assert miss["outcome"] == "unavailable"
        assert miss["candidates"] == []
        assert "no place named" in miss["detail"]
        near = agent.near(*A_LL)
        assert near["outcome"] == "ok"
        assert near["candidates"][0]["name"] == "Alpha Town"
        assert near["candidates"][0]["distance_km"] < 1.0

    def test_route(self, shelf):
        agent = _agent(shelf)
        ticket = agent.route("Alpha Town", "Delta Village")
        assert ticket["outcome"] == "ok"
        assert ticket["routed"] is True
        assert ticket["distance_m"] > 0
        roads = [s["name"] for s in ticket["steps"]]
        assert "Primary Road" in roads and "South Street" in roads
        assert len(ticket["geometry"]) > 2
        assert ticket["receipt_id"]
        # the disconnected lane truthfully does not route
        stuck = agent.route("Alpha Town", "Nowhere Hamlet")
        assert stuck["outcome"] == "unavailable"
        assert stuck["routed"] is False
        assert stuck["reason"] == "disconnected"

    def test_view_returns_a_real_png(self, shelf):
        agent = _agent(shelf)
        ticket = agent.view(A_LL[0], A_LL[1], zoom=10)
        assert ticket["outcome"] == "ok"
        path = Path(ticket["image_path"])
        assert path.is_file()
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert ticket["width"] == 768 and ticket["height"] == 768
        assert ticket["receipt_id"]  # the render receipt
        assert ticket["attribution"] == "© Network Fixture contributors"

    def test_empty_shelf_is_honest_everywhere(self, tmp_path,
                                              monkeypatch):
        packs = tmp_path / "packs"
        packs.mkdir()
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs))
        agent = _agent(SimpleNamespace(ctx=SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)),
            db=None)))
        for ticket in (agent.packs(), agent.find("Alpha Town"),
                       agent.near(7.6, 5.2),
                       agent.route("A", "B"), agent.view(7.6, 5.2)):
            assert ticket["outcome"] == "unavailable", ticket
            assert ticket["detail"]

    def test_overlay_render_ticket(self, shelf):
        agent = _agent(shelf)
        agent.overlay_store().create("town view", [
            OverlayFeature("marker", (A_LL,), "alpha")])
        ticket = agent.overlay_render("town view")
        assert ticket["outcome"] == "ok"
        assert ticket["overlay"]["name"] == "town view"
        assert Path(ticket["image_path"]).is_file()
        assert ticket["zoom"] <= 10  # clamped to the pack ceiling
        gold = _gold_pixels(Path(ticket["image_path"]).read_bytes())
        assert gold > 50  # the marker disc is painted
        missing = agent.overlay_render("nope")
        assert missing["outcome"] == "unavailable"
        assert "no overlay named" in missing["detail"]

    def test_polygon_overlay_render_notes_the_outline(self, shelf):
        agent = _agent(shelf)
        lat, lon = A_LL
        agent.overlay_store().create("farm plot", [
            OverlayFeature("polygon", (
                (lat - 0.002, lon - 0.002), (lat + 0.002, lon - 0.002),
                (lat + 0.002, lon + 0.002), (lat - 0.002, lon + 0.002)),
                "farm")])
        ticket = agent.overlay_render("farm plot")
        assert ticket["outcome"] == "ok"
        assert any("outline" in n for n in ticket["notes"])


# ── the map tool ─────────────────────────────────────────────


def _tool_registry(shelf):
    from nomorals.tools import map as map_tools
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(shelf.ctx)
    map_tools.register(registry)
    return registry


class TestMapTool:
    def test_registered_in_builtins(self, shelf):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(shelf.ctx)
        registry.register_builtins()
        assert "map" in registry.names()
        spec = registry.get("map")
        assert spec.annotations["read_only"] is True
        assert spec.annotations["idempotent"] is True

    def test_operations_round_trip(self, shelf):
        registry = _tool_registry(shelf)
        packs = registry.call("map", operation="packs")
        assert packs.ok and packs.value["outcome"] == "ok"
        found = registry.call("map", operation="find",
                              query="Alpha Town")
        assert found.ok
        assert found.value["candidates"][0]["name"] == "Alpha Town"
        assert found.value["receipt_id"]
        # the tool forwards the agent verbatim (shared fields)
        agent_found = _agent(shelf).find("Alpha Town")
        assert found.value["candidates"] == agent_found["candidates"]
        assert found.value["outcome"] == agent_found["outcome"]
        near = registry.call("map", operation="near",
                             point=f"{A_LL[0]},{A_LL[1]}")
        assert near.ok and near.value["candidates"]
        route = registry.call("map", operation="route",
                              start="Alpha Town", end="Delta Village")
        assert route.ok and route.value["routed"] is True
        view = registry.call("map", operation="view",
                             point=f"{A_LL[0]},{A_LL[1]}", zoom=10)
        assert view.ok and view.value["outcome"] == "ok"
        assert Path(view.value["image_path"]).is_file()
        listing = registry.call("map", operation="overlays")
        assert listing.ok and listing.value["overlays"] == []

    def test_overlay_operation(self, shelf):
        _agent(shelf).overlay_store().create("alpha", [
            OverlayFeature("marker", (A_LL,), "")])
        registry = _tool_registry(shelf)
        out = registry.call("map", operation="overlay",
                            overlay="alpha")
        assert out.ok and out.value["outcome"] == "ok"
        assert Path(out.value["image_path"]).is_file()

    def test_bad_calls_are_tool_errors(self, shelf):
        registry = _tool_registry(shelf)
        unknown = registry.call("map", operation="teleport")
        assert not unknown.ok
        assert "unknown map operation" in str(unknown.error)
        bad_point = registry.call("map", operation="near",
                                  point="not-a-point")
        assert not bad_point.ok
        assert "lat,lon" in str(bad_point.error)
        bad_coord = registry.call("map", operation="view",
                                  point="91,5", zoom=10)
        assert not bad_coord.ok
        assert "latitude" in str(bad_coord.error)


# ── agency wiring ────────────────────────────────────────────


class TestAgencyWiring:
    def test_compose_resolves_the_map_agent(self, shelf, tmp_path):
        from nomorals.agency.wiring import compose_agency
        from nomorals.agents.map import MapAgent

        db = Database(str(tmp_path / "agency.db"))
        db.migrate()
        ctx = SimpleNamespace(
            db=db,
            settings=SimpleNamespace(home_path=str(shelf.home)),
            gateway=None, extras={}, tools=None)
        composed = compose_agency(ctx)
        assert isinstance(composed["map_agent"], MapAgent)
        assert composed["readiness"]["maps"].startswith("ready")
        found = composed["map_agent"].find("Alpha Town")
        assert found["candidates"][0]["name"] == "Alpha Town"


# ── /map overlay verbs on the runtime harness ────────────────


class _Sent:
    def __init__(self):
        self.files = []

    def send_file(self, platform, chat, path, caption="",
                  kind="image", name=""):
        data = Path(path).read_bytes()
        self.files.append({
            "platform": platform, "path": path, "caption": caption,
            "kind": kind, "name": name, "bytes": len(data),
            "data": data})
        return SimpleNamespace(ok=True)


class TestMapOverlayVerbs:
    def _runtime(self, shelf):
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = shelf.ctx
        rt._owner_chats = {"telegram:42"}
        rt._owner_sender_ids = set()
        rt.gateway = _Sent()
        return rt

    def _ref(self, key="telegram:42"):
        # the M3/M6 surface harness: a fake inbound message whose
        # chat key the owner gate reads
        return SimpleNamespace(
            chat=SimpleNamespace(key=key), text="", meta={},
            sender_id="")

    def test_add_list_show_drop(self, shelf):
        rt = self._runtime(shelf)
        point = f"{A_LL[0]:.5f},{A_LL[1]:.5f}"
        added = rt._control_map(f"overlay add base {point} Base camp",
                                message=self._ref())
        assert "saved" in added
        listed = rt._control_map("overlay list", message=self._ref())
        assert "base" in listed and "marker" in listed
        shown = rt._control_map("overlay show base", message=self._ref())
        assert shown == ""  # a landed photo returns no narration
        assert len(rt.gateway.files) == 1
        sent = rt.gateway.files[0]
        assert "base" in sent["caption"]
        assert "© Network Fixture contributors" in sent["caption"]
        assert _gold_pixels(sent["data"]) > 50
        dropped = rt._control_map("overlay drop base", message=self._ref())
        assert "dropped" in dropped
        listed = rt._control_map("overlay list", message=self._ref())
        assert "no overlays yet" in listed
        gone = rt._control_map("overlay show base", message=self._ref())
        assert "no overlay named" in gone

    def test_duplicate_add_and_bad_point_refused(self, shelf):
        rt = self._runtime(shelf)
        point = f"{A_LL[0]:.5f},{A_LL[1]:.5f}"
        rt._control_map(f"overlay add base {point}",
                                message=self._ref())
        again = rt._control_map(f"overlay add base {point}",
                                message=self._ref())
        assert "couldn't save overlay" in again
        bad = rt._control_map("overlay add bad 91,5",
                                message=self._ref())
        assert "bad marker point" in bad

    def test_polygon_show_caption_states_the_outline(self, shelf):
        rt = self._runtime(shelf)
        lat, lon = A_LL
        # richer authoring is data-level: plant the polygon
        # through the agent's store, then show it on the surface.
        _agent(shelf).overlay_store().create("farm", [
            OverlayFeature("polygon", (
                (lat - 0.002, lon - 0.002),
                (lat + 0.002, lon - 0.002),
                (lat + 0.002, lon + 0.002),
                (lat - 0.002, lon + 0.002)), "farm")])
        shown = rt._control_map("overlay show farm", message=self._ref())
        assert shown == ""
        caption = rt.gateway.files[0]["caption"]
        assert "outline" in caption
        assert "© Network Fixture contributors" in caption

    def test_stranger_refused(self, shelf):
        rt = self._runtime(shelf)
        out = rt._control_map(
            "overlay list", message=self._ref("telegram:999"))
        assert "owner-only" in out
