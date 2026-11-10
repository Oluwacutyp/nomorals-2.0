"""Doc 19+21 Atlas — the universality pass: the managed
gazetteer index.

M4 shipped the local geocoder with a digest-keyed index, but
the build was a single all-or-nothing scan that query surfaces
triggered inline (``ensure`` on find/near/route) — a full
country scan runs minutes, so first use of /map find on a real
shelf meant a ~645 s stall inside a chat command, and an
interrupted scan left nothing behind but the wait.

This pass makes the build MANAGED:

* chunked + resumable — sightings stage chunk by chunk, the
  cursor commits with each chunk, an interrupted build resumes
  where it stopped;
* never half-served — entries swap into the serving tables in
  one finalize transaction; a mid-build pack's rows never
  answer as a complete index;
* digest-cached — an unchanged pack is a true no-op, a changed
  pack reads ``stale`` and its old rows are never served as
  current to a shelf-aware query;
* never inline — find/near/route and the marks place-hint read
  the index as it stands and NAME the absent/stale/building
  state with the one build door, ``nm maps index``.

The fixture encoder is the M4 suite's tiny wire encoder, copied
in the sibling suites' idiom (each maps suite carries its own).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.maps.coords import (
    BBox,
    TileCoord,
    lat_to_y_float,
    lon_to_x_float,
)
from nomorals.maps.gazetteer import (
    Gazetteer,
    index_status_line,
)
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    tile_relpath,
    write_manifest,
)

ATTRIBUTION = "© OpenStreetMap contributors"
NIGERIA = BBox(west=-10.0, south=0.0, east=20.0, north=15.0)
EXTENT = 4096

ALPHA = (7.6211, 5.2214)      # Alpha Town — city
BETA = (5.4820, 4.9000)       # Beta Village — village
GAMMA = (9.0765, 7.4898)      # Gamma City — city
DELTA = (6.5244, 3.3792)      # Delta Port — town
EPSILON = (12.0000, 8.5167)   # Epsilon City — city (v2 only)


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
    """Accumulates named places into per-tile layer bytes."""

    def __init__(self):
        self.tiles: dict[tuple, dict[str, list]] = {}

    def add_point(self, tags: dict, lat: float, lon: float,
                  zoom: int = 8):
        x, y, px, py = _pixel(lat, lon, zoom)
        geometry = _geometry((1, [(px, py)]))
        self.tiles.setdefault((zoom, x, y), {}).setdefault(
            "place", []).append((tags, 1, geometry))
        return (zoom, x, y)

    def build(self, root: Path, *, pack_id="fixture-pack",
              name="Fixture Pack"):
        root.mkdir(parents=True, exist_ok=True)
        for (z, x, y), layers in self.tiles.items():
            tile = _tile(*(_layer_from_specs(lname, specs)
                           for lname, specs in layers.items()))
            path = root / tile_relpath(z, x, y, "mvt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile)
        manifest = build_manifest(
            root, pack_id=pack_id, name=name, bbox=NIGERIA,
            zoom_min=0, zoom_max=14, attribution=ATTRIBUTION,
            built_at="2026-11-10T00:00:00+00:00",
            tile_format="mvt")
        write_manifest(root, manifest)
        return load_manifest(root)


def _fixture_pack(root: Path, *, pack_id="fixture-pack",
                  extra=False):
    """Four towns in four z8 tiles (so chunking has joints),
    plus — with ``extra`` — a fifth place in a fifth tile: the
    v2 pack whose digest differs from v1's."""
    b = _PackBuilder()
    b.add_point({"name": "Alpha Town", "class": "city",
                 "population": 350000}, *ALPHA)
    b.add_point({"name": "Beta Village", "class": "village"},
                *BETA)
    b.add_point({"name": "Gamma City", "class": "city",
                 "population": 900000}, *GAMMA)
    b.add_point({"name": "Delta Port", "class": "town"}, *DELTA)
    if extra:
        b.add_point({"name": "Epsilon City", "class": "city",
                     "population": 1200000}, *EPSILON)
    return b.build(root, pack_id=pack_id)


@pytest.fixture()
def shelf(tmp_path):
    """A packs root holding the fixture pack, unbuilt."""
    packs = tmp_path / "packs"
    pack_dir = packs / "fixture"
    manifest = _fixture_pack(pack_dir)
    return SimpleNamespace(
        root=packs, pack_dir=pack_dir, manifest=manifest,
        shelf=[(pack_dir, manifest)], tmp=tmp_path)


# ── digest caching: unchanged is a no-op, changed is stale ─────


class TestDigestCaching:
    def test_second_build_is_a_true_noop(self, shelf):
        gaz = Gazetteer(shelf.root)
        first = gaz.build(shelf.shelf)
        assert first.built_any and first.total_entries == 4
        again = gaz.build(shelf.shelf)
        assert not again.built_any
        assert again.packs[0].skipped
        assert again.total_entries == 4
        ensured = gaz.ensure(shelf.shelf)
        assert not ensured.built_any and ensured.receipt is None
        status = gaz.index_status(shelf.shelf)
        assert status[0].state == "current"
        assert status[0].entries == 4
        assert index_status_line(status) == ""

    def test_changed_pack_reads_stale_and_serves_nothing_current(
            self, shelf):
        gaz = Gazetteer(shelf.root)
        gaz.build(shelf.shelf)
        assert gaz.search("Alpha Town",
                          packs=shelf.shelf).candidates
        # the pack changes: a fifth place lands, the manifest
        # is rewritten, the digest moves
        manifest2 = _fixture_pack(shelf.pack_dir, extra=True)
        assert manifest2.digest != shelf.manifest.digest
        shelf2 = [(shelf.pack_dir, manifest2)]
        status = gaz.index_status(shelf2)
        assert status[0].state == "stale"
        line = index_status_line(status)
        assert "fixture-pack" in line
        assert "nm maps index" in line
        # shelf-aware queries never serve the stale rows…
        assert not gaz.search("Alpha Town",
                              packs=shelf2).candidates
        assert not gaz.search("Epsilon City",
                              packs=shelf2).candidates
        # …while the recorded index, read as it stands without
        # a shelf, still holds exactly the v1 build (documented
        # posture: staleness is judged against the shelf)
        assert gaz.search("Alpha Town").candidates
        # the rebuild serves the new pack whole
        rebuilt = gaz.build(shelf2)
        assert rebuilt.total_entries == 5
        assert gaz.search("Epsilon City",
                          packs=shelf2).candidates
        assert gaz.index_status(shelf2)[0].state == "current"


# ── resumability: interrupted builds resume, never half-serve ──


class TestResumableBuild:
    def test_interrupted_build_resumes_at_the_cursor(self, shelf):
        gaz = Gazetteer(shelf.root)
        calls: list[tuple] = []

        def interrupting_progress(pack_id, done, total):
            calls.append((done, total))
            if len(calls) >= 2:
                raise KeyboardInterrupt("operator stop")

        with pytest.raises(KeyboardInterrupt):
            gaz.build(shelf.shelf, progress=interrupting_progress,
                      chunk_members=1)
        # two chunks committed before the stop: the cursor says so
        status = gaz.index_status(shelf.shelf)
        assert status[0].state == "building"
        assert status[0].members_done == 2
        assert status[0].members_total == 4
        assert status[0].entries == 0
        line = index_status_line(status)
        assert "mid-build" in line and "2 of 4" in line
        # nothing half-built answers: not shelf-aware…
        assert not gaz.search("Alpha Town",
                              packs=shelf.shelf).candidates
        assert not gaz.reverse(*ALPHA,
                               packs=shelf.shelf).candidates
        # …and not even shelf-blind: no complete build was ever
        # recorded for this pack, so no rows exist to serve
        assert not gaz.search("Alpha Town").candidates
        # the resume finishes the job and lands the same index
        # a from-scratch build lands
        resumed = gaz.build(shelf.shelf, chunk_members=1)
        result = [p for p in resumed.packs
                  if p.pack_id == "fixture-pack"][0]
        assert result.entries == 4
        assert "resumed an interrupted build" in result.note
        assert gaz.index_status(shelf.shelf)[0].state == "current"
        found = gaz.search("Gamma City", packs=shelf.shelf)
        assert found.candidates
        assert abs(found.candidates[0].lat - GAMMA[0]) < 0.01

    def test_force_rebuild_ignores_a_stale_cursor(self, shelf):
        gaz = Gazetteer(shelf.root)

        def stop(pack_id, done, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            gaz.build(shelf.shelf, progress=stop, chunk_members=1)
        assert gaz.index_status(
            shelf.shelf)[0].state == "building"
        forced = gaz.build(shelf.shelf, force=True)
        assert forced.total_entries == 4
        status = gaz.index_status(shelf.shelf)
        assert status[0].state == "current"


# ── partial shelves: one current pack still serves ──────────────


class TestPartialShelf:
    def test_current_pack_serves_while_another_is_absent(
            self, tmp_path):
        packs = tmp_path / "packs"
        dir_a = packs / "a"
        man_a = _fixture_pack(dir_a, pack_id="pack-a")
        dir_b = packs / "b"
        man_b = _fixture_pack(dir_b, pack_id="pack-b")
        gaz = Gazetteer(packs)
        gaz.build([(dir_a, man_a)])
        shelf = [(dir_a, man_a), (dir_b, man_b)]
        states = {s.pack_id: s.state
                  for s in gaz.index_status(shelf)}
        assert states == {"pack-a": "current", "pack-b": "absent"}
        found = gaz.search("Alpha Town", packs=shelf)
        assert found.candidates
        assert found.candidates[0].pack_id == "pack-a"


# ── query paths never build (agent level) ───────────────────────


def _agent(tmp_path, packs):
    from nomorals.agents.map import MapAgent

    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=str(tmp_path)),
        db=None)
    env = mock.patch.dict(os.environ, {
        "NM_MAPS_ENABLED": "1",
        "NM_MAPS_PACKS_DIR": str(packs)})
    env.start()
    return MapAgent(ctx), env


class TestQueryPathsNeverBuild:
    def test_find_near_route_name_the_state_and_build_nothing(
            self, shelf):
        agent, env = _agent(shelf.tmp, shelf.root)
        try:
            index_file = shelf.root / "gazetteer.sqlite3"
            assert not index_file.exists()
            found = agent.find("Alpha Town")
            assert found["outcome"] == "unavailable"
            assert found["candidates"] == []
            assert found["index_state"] == [
                {"pack_id": "fixture-pack", "state": "absent"}]
            assert "nm maps index" in found["detail"]
            near = agent.near(*ALPHA)
            assert near["outcome"] == "unavailable"
            assert "nm maps index" in near["detail"]
            routed = agent.route("Alpha Town", "Beta Village")
            assert routed["outcome"] == "unavailable"
            assert "nm maps index" in routed["detail"]
            # the marks place-hint reads the same absent index
            # and stays honestly empty
            assert agent.place_hint(*ALPHA) == ""
            # and after all four query paths: still no index
            assert not index_file.exists()
        finally:
            env.stop()

    def test_agent_answers_once_the_index_is_built(self, shelf):
        Gazetteer(shelf.root).build(shelf.shelf)
        agent, env = _agent(shelf.tmp, shelf.root)
        try:
            found = agent.find("Alpha Town")
            assert found["outcome"] == "ok"
            assert "index_state" not in found
            assert found["candidates"][0]["name"] == "Alpha Town"
        finally:
            env.stop()


# ── the CLI door: nm maps index ─────────────────────────────────


class TestCliIndex:
    def _args(self, **kw):
        base = {"pack": "", "packs_dir": "", "status": False,
                "force": False, "json": False}
        base.update(kw)
        return SimpleNamespace(**base)

    def _context(self, tmp_path):
        return SimpleNamespace(
            settings=SimpleNamespace(home=str(tmp_path)))

    def test_status_build_and_noop(self, shelf, capsys):
        from nomorals.cmdline.commands.maps import _index

        args = self._args(packs_dir=str(shelf.root), status=True)
        assert _index(args, self._context(shelf.tmp)) == 0
        out = capsys.readouterr().out
        assert "fixture-pack — absent" in out
        assert not (shelf.root / "gazetteer.sqlite3").exists()

        args = self._args(packs_dir=str(shelf.root))
        assert _index(args, self._context(shelf.tmp)) == 0
        out = capsys.readouterr().out
        assert "fixture-pack — 4 places from 4 tiles" in out
        assert "total: 4 named places indexed" in out

        args = self._args(packs_dir=str(shelf.root), status=True)
        assert _index(args, self._context(shelf.tmp)) == 0
        assert "fixture-pack — current" in capsys.readouterr().out

        args = self._args(packs_dir=str(shelf.root))
        assert _index(args, self._context(shelf.tmp)) == 0
        out = capsys.readouterr().out
        assert "skipped (digest unchanged)" in out

    def test_index_with_no_packs_is_an_operator_error(
            self, tmp_path, capsys):
        from nomorals.cmdline.commands.maps import _index

        empty = tmp_path / "empty"
        empty.mkdir()
        args = self._args(packs_dir=str(empty))
        assert _index(args, self._context(tmp_path)) == 1
        assert "no map packs installed" in capsys.readouterr().out
