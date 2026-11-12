"""Docs 19+21 M2 — basemap pipeline pins.

The pipeline (acquire → build → package → verify → install),
the source/pack definitions as data, the MBTiles packaging
bridge, the Natural Earth raster builder and the release
registry — all proven OFFLINE: fake fetchers write fixture
bytes, a fake command runner emits a real tiny MBTiles, and the
Natural Earth builder paints a fixture square.  The real data
runs are operator executions, never pytest downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.cmdline.commands.maps import _cmd_maps
from nomorals.maps import get_pack, get_source
from nomorals.maps.coords import BBox
from nomorals.maps.definitions import (
    BUILDER_NE_RASTER,
    BUILDER_PLANETILER,
    PACKS,
    MapSource,
    PackDefinition,
)
from nomorals.maps.manifest import PackError, load_manifest
from nomorals.maps.mbtiles import extract_mbtiles, mbtiles_metadata
from nomorals.maps.neraster import LAND_RGB, OCEAN_RGB, build_ne_tiles
from nomorals.maps.pipeline import (
    PipelineError,
    acquire_source,
    package_pack,
    planetiler_argv,
    run_build,
    verify_and_install,
)
from nomorals.maps.pngcodec import decode_png
from nomorals.maps.receipts import (
    OP_PACK_INSTALL,
    OP_PACK_VERIFY,
    MapReceiptStore,
)
from nomorals.maps.registry import ReleaseEntry, ReleaseRegistry

# ── definitions as data ──────────────────────────────────────────


class TestDefinitions:
    def test_packs_carry_the_docs_zoom_asks(self):
        pack_a = get_pack("pack-a-nigeria")
        assert (pack_a.zoom_min, pack_a.zoom_max) == (0, 14)
        assert pack_a.stretch_zoom_max == 16
        assert pack_a.builder == BUILDER_PLANETILER
        assert pack_a.tile_format == "mvt"
        lagos = get_pack("pack-a-lagos")
        # Planetiler 0.9 caps maxzoom at 15; z16 is the stretch.
        assert (lagos.zoom_min, lagos.zoom_max) == (15, 15)
        assert lagos.stretch_zoom_max == 16
        pack_b = get_pack("pack-b-world")
        assert (pack_b.zoom_min, pack_b.zoom_max) == (0, 7)
        assert pack_b.stretch_zoom_max == 8
        assert pack_b.builder == BUILDER_NE_RASTER
        assert pack_b.tile_format == "png"

    def test_ghana_pack_carries_the_pack_a_shape(self):
        """Census B10: the first country pack beyond Nigeria —
        same Pack A shape as pack-a-nigeria, same pipeline."""
        ghana = get_pack("pack-a-ghana")
        assert (ghana.zoom_min, ghana.zoom_max) == (0, 14)
        assert ghana.stretch_zoom_max == 16
        assert ghana.builder == BUILDER_PLANETILER
        assert ghana.tile_format == "mvt"
        assert ghana.source_id == "geofabrik-ghana"
        # The bbox is the live Ghana extent (Nominatim, verified
        # 2026-11-12: west -3.2608, south 4.5393, east 1.2733,
        # north 11.1749), rounded outward to 1 decimal like
        # Nigeria's — and it holds the country's cities.
        assert ghana.bbox.as_list() == [-3.3, 4.5, 1.3, 11.2]
        for lat, lon in ((5.6037, -0.1870),   # Accra
                         (6.6885, -1.6244),   # Kumasi
                         (9.4008, -0.8393)):  # Tamale
            assert ghana.bbox.west <= lon <= ghana.bbox.east
            assert ghana.bbox.south <= lat <= ghana.bbox.north

    def test_ghana_source_is_a_rolling_geofabrik_extract(self):
        source = get_source("geofabrik-ghana")
        assert source.url == ("https://download.geofabrik.de/africa/"
                              "ghana-latest.osm.pbf")
        assert source.md5_url == source.url + ".md5"
        assert source.filename == "ghana-latest.osm.pbf"
        assert source.attribution == "© OpenStreetMap contributors"
        assert source.license == "ODbL 1.0"
        # Rolling, like Nigeria's: no pinned SHA-256, the MD5
        # sidecar is the publisher check (pin-after-first-fetch).
        assert source.sha256 == "" and source.md5_url

    def test_every_pack_source_resolves_with_attribution(self):
        for pack in PACKS:
            for source_id in pack.source_ids():
                source = get_source(source_id)
                assert source.attribution and source.license

    def test_hash_discipline_is_honest(self):
        jar = get_source("planetiler-jar")
        assert len(jar.sha256) == 64  # pinned after first fetch
        extract = get_source("geofabrik-nigeria")
        assert extract.sha256 == "" and extract.md5_url  # rolling

    def test_unknown_ids_raise_named(self):
        with pytest.raises(KeyError):
            get_pack("pack-zzz")
        with pytest.raises(KeyError):
            get_source("nope")


# ── acquire ──────────────────────────────────────────────────────


def _fetcher_writing(payloads: dict[str, bytes]):
    def fetch(url: str, dest: Path, timeout_s: float) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payloads[url])
    return fetch


class TestAcquire:
    def test_pinned_hash_match_and_mismatch(self, tmp_path):
        good = b"the-real-artifact"
        source = MapSource(
            source_id="pinned-x", name="Pinned", url="https://x/a.bin",
            attribution="X", license="test",
            sha256=hashlib.sha256(good).hexdigest())
        result = acquire_source(
            source, tmp_path,
            fetcher=_fetcher_writing({source.url: good}))
        assert result.pinned and result.sha256 == source.sha256
        with pytest.raises(PipelineError) as excinfo:
            acquire_source(
                source, tmp_path / "other",
                fetcher=_fetcher_writing({source.url: b"tampered"}))
        assert "pinned-x" in str(excinfo.value)
        assert "hash mismatch" in str(excinfo.value)

    def test_rolling_source_md5_sidecar(self, tmp_path):
        payload = b"rolling-extract-bytes"
        source = MapSource(
            source_id="rolling-y", name="Rolling",
            url="https://x/y.pbf", attribution="Y", license="test",
            md5_url="https://x/y.pbf.md5", filename="y.pbf")
        md5 = hashlib.md5(payload).hexdigest().encode() + b"  y.pbf"
        result = acquire_source(
            source, tmp_path,
            fetcher=_fetcher_writing({source.url: payload,
                                      source.md5_url: md5}))
        assert result.md5_verified and not result.pinned
        assert result.sha256 == hashlib.sha256(payload).hexdigest()

    def test_existing_sidecar_is_reused_not_refetched(self, tmp_path):
        payload = b"rolling-extract-bytes"
        source = MapSource(
            source_id="rolling-w", name="Rolling",
            url="https://x/w.pbf", attribution="W", license="test",
            md5_url="https://x/w.pbf.md5", filename="w.pbf")
        (tmp_path / "w.pbf").write_bytes(payload)
        (tmp_path / "w.pbf.md5").write_text(
            hashlib.md5(payload).hexdigest() + "  w.pbf",
            encoding="utf-8")

        def fetch(url: str, dest: Path, timeout_s: float) -> None:
            raise AssertionError("no fetch expected: " + url)

        result = acquire_source(source, tmp_path, fetcher=fetch)
        assert result.md5_verified

    def test_rolling_source_md5_mismatch_refused(self, tmp_path):
        source = MapSource(
            source_id="rolling-z", name="Rolling",
            url="https://x/z.pbf", attribution="Z", license="test",
            md5_url="https://x/z.pbf.md5", filename="z.pbf")
        with pytest.raises(PipelineError) as excinfo:
            acquire_source(
                source, tmp_path,
                fetcher=_fetcher_writing(
                    {source.url: b"bytes",
                     source.md5_url: b"0" * 32 + b"  z.pbf"}))
        assert "rolling-z" in str(excinfo.value)


# ── mbtiles bridge ───────────────────────────────────────────────


def _write_mbtiles(path: Path, tiles: list[tuple[int, int, int, bytes]]):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
    conn.execute("CREATE TABLE tiles (zoom_level INTEGER, "
                 "tile_column INTEGER, tile_row INTEGER, "
                 "tile_data BLOB)")
    conn.execute("INSERT INTO metadata VALUES ('format', 'pbf')")
    conn.executemany("INSERT INTO tiles VALUES (?, ?, ?, ?)", tiles)
    conn.commit()
    conn.close()


class TestMbtiles:
    def test_extract_flips_tms_rows_to_xyz(self, tmp_path):
        mbtiles = tmp_path / "in.mbtiles"
        payload = b"\x1f\x8bgzip-mvt-bytes"
        # z=1 has 2 rows: TMS row 0 is the SOUTH row → XYZ y=1.
        _write_mbtiles(mbtiles, [(1, 0, 0, payload), (1, 1, 1, b"n")])
        out = tmp_path / "pack"
        count = extract_mbtiles(mbtiles, out, "mvt")
        assert count == 2
        assert (out / "tiles/1/0/1.mvt").read_bytes() == payload
        assert (out / "tiles/1/1/0.mvt").read_bytes() == b"n"
        assert mbtiles_metadata(mbtiles) == {"format": "pbf"}

    def test_empty_mbtiles_is_a_named_failure(self, tmp_path):
        mbtiles = tmp_path / "empty.mbtiles"
        _write_mbtiles(mbtiles, [])
        with pytest.raises(PackError):
            extract_mbtiles(mbtiles, tmp_path / "pack", "mvt")


# ── natural earth raster builder ─────────────────────────────────


_SQUARE = {
    "type": "FeatureCollection",
    "features": [{"type": "Feature", "geometry": {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [90.0, 0.0], [90.0, 80.0],
                         [0.0, 80.0], [0.0, 0.0]]]}}],
}


class TestNeRaster:
    def test_paints_land_and_ocean(self, tmp_path):
        land = tmp_path / "land.geojson"
        land.write_text(json.dumps(_SQUARE), encoding="utf-8")
        out = tmp_path / "pack"
        bbox = BBox(0.0, 0.0, 90.0, 85.0)
        count = build_ne_tiles(
            land_path=land, out_dir=out, bbox=bbox,
            zoom_min=1, zoom_max=1)
        assert count == len(bbox.tiles(1)) == 2
        width, height, rgb = decode_png(
            (out / "tiles/1/1/0.png").read_bytes())
        assert (width, height) == (256, 256)
        center = (128 * 256 + 128) * 3
        assert tuple(rgb[center:center + 3]) == LAND_RGB
        _w, _h, ocean = decode_png(
            (out / "tiles/1/1/1.png").read_bytes())
        assert tuple(ocean[center:center + 3]) == OCEAN_RGB

    def test_missing_land_is_named(self, tmp_path):
        empty = tmp_path / "empty.geojson"
        empty.write_text('{"type": "FeatureCollection", '
                         '"features": []}', encoding="utf-8")
        with pytest.raises(PackError):
            build_ne_tiles(
                land_path=empty, out_dir=tmp_path / "pack",
                bbox=BBox(0.0, 0.0, 90.0, 85.0),
                zoom_min=1, zoom_max=1)


# ── release registry ─────────────────────────────────────────────


class TestRegistry:
    def test_round_trip_and_missing(self, tmp_path):
        registry = ReleaseRegistry(tmp_path)
        assert registry.get("pack-a-nigeria") is None
        assert registry.all() == ()
        entry = ReleaseEntry(
            pack_id="pack-a-nigeria", digest="d" * 64,
            signature_state="unsigned", built_at="2026-11-10",
            source="Geofabrik", tile_format="mvt",
            zoom_min=0, zoom_max=14)
        registry.record(entry)
        again = ReleaseRegistry(tmp_path)
        assert again.get("pack-a-nigeria") == entry
        assert [e.pack_id for e in again.all()] == ["pack-a-nigeria"]
        assert (tmp_path / "releases.json").is_file()


# ── the pipeline, end to end on fakes ────────────────────────────


def _fake_runner_mbtailes(argv: list[str], timeout_s: float):
    output = next(a.split("=", 1)[1] for a in argv
                  if a.startswith("--output="))
    _write_mbtiles(Path(output), [(15, 100, 200, b"mvt-tile-bytes")])
    return 0, "planetiler-fixture: ok"


def _geofabrik_fetcher(url: str, dest: Path, timeout_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".md5"):
        digest = hashlib.md5(b"fake-pbf").hexdigest()
        dest.write_text(digest + "  nigeria-latest.osm.pbf",
                        encoding="utf-8")
    else:
        dest.write_bytes(b"fake-pbf")


def _fake_runner_ghana_mbtailes(argv: list[str], timeout_s: float):
    """The z14 twin of the Lagos runner: one tile inside the
    Ghana pack's own zoom range (z0–z14), never above it."""
    output = next(a.split("=", 1)[1] for a in argv
                  if a.startswith("--output="))
    _write_mbtiles(Path(output), [(14, 100, 200, b"mvt-tile-bytes")])
    return 0, "planetiler-fixture: ok"


def _ghana_fetcher(url: str, dest: Path, timeout_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".md5"):
        digest = hashlib.md5(b"fake-pbf").hexdigest()
        dest.write_text(digest + "  ghana-latest.osm.pbf",
                        encoding="utf-8")
    else:
        dest.write_bytes(b"fake-pbf")


class TestRunBuild:
    def test_planetiler_pack_end_to_end(self, tmp_path):
        jar = tmp_path / "planetiler.jar"
        jar.write_bytes(b"jar")
        store = MapReceiptStore(home=str(tmp_path / "home"))
        report = run_build(
            "pack-a-lagos", build_root=tmp_path / "build",
            packs_dir=tmp_path / "packs", store=store,
            fetcher=_geofabrik_fetcher, runner=_fake_runner_mbtailes,
            env={"NM_MAPS_JAVA": "java-fixture",
                 "NM_MAPS_PLANETILER_JAR": str(jar)})
        assert report.ok, report.error
        assert report.stages == [
            "acquire:geofabrik-nigeria", "build:planetiler-mbtiles",
            "package", "verify", "install"]
        assert report.tiles == 1
        pack_dir = tmp_path / "packs" / "pack-a-lagos"
        assert (pack_dir / "manifest.json").is_file()
        manifest = load_manifest(pack_dir)
        assert manifest.digest == report.digest
        # Planetiler embeds Natural Earth at low zooms: both credits.
        assert manifest.attribution == \
            "© OpenStreetMap contributors · Natural Earth"
        assert report.signature_state == "unsigned"
        entry = ReleaseRegistry(tmp_path / "packs").get("pack-a-lagos")
        assert entry is not None and entry.digest == report.digest
        installs = store.recent(operation=OP_PACK_INSTALL)
        verifies = store.recent(operation=OP_PACK_VERIFY)
        assert len(installs) == 1 and len(verifies) == 1
        assert installs[0].receipt_id == report.install_receipt_id

    def test_ghana_pack_end_to_end(self, tmp_path):
        """Census B10: ``nm maps build pack-a-ghana`` runs the
        whole pipeline for the registered Ghana definition —
        the build command is parameterised by pack id, never a
        one-off per country."""
        jar = tmp_path / "planetiler.jar"
        jar.write_bytes(b"jar")
        store = MapReceiptStore(home=str(tmp_path / "home"))
        report = run_build(
            "pack-a-ghana", build_root=tmp_path / "build",
            packs_dir=tmp_path / "packs", store=store,
            fetcher=_ghana_fetcher, runner=_fake_runner_ghana_mbtailes,
            env={"NM_MAPS_JAVA": "java-fixture",
                 "NM_MAPS_PLANETILER_JAR": str(jar)})
        assert report.ok, report.error
        assert report.stages == [
            "acquire:geofabrik-ghana", "build:planetiler-mbtiles",
            "package", "verify", "install"]
        assert report.tiles == 1
        pack_dir = tmp_path / "packs" / "pack-a-ghana"
        assert (pack_dir / "manifest.json").is_file()
        manifest = load_manifest(pack_dir)
        assert manifest.digest == report.digest
        assert manifest.attribution == \
            "© OpenStreetMap contributors · Natural Earth"
        assert report.signature_state == "unsigned"
        entry = ReleaseRegistry(tmp_path / "packs").get("pack-a-ghana")
        assert entry is not None and entry.digest == report.digest
        assert entry.zoom_max == 14
        installs = store.recent(operation=OP_PACK_INSTALL)
        verifies = store.recent(operation=OP_PACK_VERIFY)
        assert len(installs) == 1 and len(verifies) == 1

    def test_ne_pack_end_to_end_via_seam(self, tmp_path, monkeypatch):
        import nomorals.maps.pipeline as pipeline

        tiny = PackDefinition(
            pack_id="pack-tiny-ne", name="Tiny NE", source_id="ne-land",
            builder=BUILDER_NE_RASTER, tile_format="png",
            bbox=BBox(0.0, 0.0, 90.0, 85.0), zoom_min=1, zoom_max=1,
            extra_source_ids=("ne-lakes", "ne-admin0-lines"))
        monkeypatch.setattr(pipeline, "get_pack", lambda _pid: tiny)

        def fetch(url: str, dest: Path, timeout_s: float) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(_SQUARE), encoding="utf-8")

        store = MapReceiptStore(home=str(tmp_path / "home"))
        report = run_build(
            "pack-tiny-ne", build_root=tmp_path / "build",
            packs_dir=tmp_path / "packs", store=store, fetcher=fetch)
        assert report.ok, report.error
        assert report.stages == [
            "acquire:ne-land", "acquire:ne-lakes",
            "acquire:ne-admin0-lines", "build:ne-raster",
            "package", "verify", "install"]
        assert report.tiles == 2

    def test_builder_failure_installs_nothing(self, tmp_path):
        jar = tmp_path / "planetiler.jar"
        jar.write_bytes(b"jar")
        store = MapReceiptStore(home=str(tmp_path / "home"))

        def bad_runner(argv: list[str], timeout_s: float):
            return 3, "boom: out of memory"

        report = run_build(
            "pack-a-lagos", build_root=tmp_path / "build",
            packs_dir=tmp_path / "packs", store=store,
            fetcher=_geofabrik_fetcher, runner=bad_runner,
            env={"NM_MAPS_JAVA": "java-fixture",
                 "NM_MAPS_PLANETILER_JAR": str(jar)})
        assert not report.ok
        assert "build" in report.error or "command failed" in report.error
        assert not (tmp_path / "packs" / "pack-a-lagos").exists()
        assert store.recent(operation=OP_PACK_INSTALL) == []

    def test_planetiler_argv_ghana_bounds(self):
        """The Planetiler command for the Ghana pack carries the
        Ghana bbox and zoom range — the build is the same
        command, parameterised by the pack row."""
        argv = planetiler_argv(
            "java", "p.jar", Path("in.pbf"), Path("out.mbtiles"),
            get_pack("pack-a-ghana"), Path("data"))
        assert argv[:4] == ["java", "-Xmx2g", "-jar", "p.jar"]
        assert "--osm-path=in.pbf" in argv
        assert "--bounds=-3.3,4.5,1.3,11.2" in argv
        assert "--minzoom=0" in argv
        assert "--maxzoom=14" in argv
        assert all(isinstance(item, str) for item in argv)

    def test_planetiler_argv_shape(self):
        argv = planetiler_argv(
            "java", "p.jar", Path("in.pbf"), Path("out.mbtiles"),
            get_pack("pack-a-nigeria"), Path("data"))
        assert argv[:4] == ["java", "-Xmx2g", "-jar", "p.jar"]
        assert "--osm-path=in.pbf" in argv
        assert "--bounds=2.6,4.2,14.7,13.9" in argv
        assert "--maxzoom=14" in argv
        # Tile weights are pre-seeded by the acquire stage and the
        # path pinned absolutely (the JVM default is cwd-relative);
        # tmpdir lives inside the build directory, never the cwd.
        assert "--tile-weights=data/tile_weights.tsv.gz" in argv
        assert "--tmpdir=planetiler-tmp" in argv
        assert all(isinstance(item, str) for item in argv)

    def test_jvm_proxy_args_strip_userinfo(self):
        from nomorals.maps.pipeline import _jvm_proxy_args

        args = _jvm_proxy_args(
            # credential assembled from parts (secaudit rule)
            {"https_proxy": "http://user:" + "sec" + "ret@proxy.local:3128"})
        assert args == ["-Dhttp.proxyHost=proxy.local",
                        "-Dhttps.proxyHost=proxy.local",
                        "-Dhttp.proxyPort=3128",
                        "-Dhttps.proxyPort=3128"]
        assert not any("secret" in arg for arg in args)
        assert _jvm_proxy_args({}) == []
        argv = planetiler_argv(
            "java", "p.jar", Path("in.pbf"), Path("out.mbtiles"),
            get_pack("pack-a-nigeria"), Path("data"),
            proxy_args=tuple(args))
        assert argv[1] == "-Dhttp.proxyHost=proxy.local"


class TestVerifyGate:
    def test_corrupt_pack_is_not_installed(self, tmp_path):
        definition = get_pack("pack-b-world")
        staging = tmp_path / "staging"
        tile = staging / "tiles/0/0/0.png"
        tile.parent.mkdir(parents=True)
        tile.write_bytes(b"png-bytes")
        package_pack(definition, staging)
        tile.write_bytes(b"tampered-after-packaging")
        store = MapReceiptStore(home=str(tmp_path / "home"))
        outcome = verify_and_install(
            definition, staging, tmp_path / "packs", store=store)
        assert not outcome.ok
        assert "verify failed" in outcome.error
        assert not (tmp_path / "packs" / "pack-b-world").exists()
        assert ReleaseRegistry(tmp_path / "packs").all() == ()
        assert store.recent(operation=OP_PACK_INSTALL) == []
        verifies = store.recent(operation=OP_PACK_VERIFY)
        assert len(verifies) == 1
        assert verifies[0].outcome == "failed"


# ── CLI ──────────────────────────────────────────────────────────


class TestCli:
    def _args(self, **kw):
        base = dict(maps_action="", pack="", packs_dir="",
                    build_dir="", xmx="2g", threads=2,
                    timeout=7200.0, json=True)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sources_verb(self, capsys):
        context = SimpleNamespace(settings=None)
        rc = _cmd_maps(self._args(maps_action="sources"), context)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        ids = [row["source_id"] for row in payload["sources"]]
        assert "geofabrik-nigeria" in ids
        assert "planetiler-jar" in ids

    def test_packs_verb_lists_ghana_uninstalled(self, tmp_path, capsys):
        context = SimpleNamespace(settings=None)
        rc = _cmd_maps(
            self._args(maps_action="packs",
                       packs_dir=str(tmp_path / "packs")), context)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        by_id = {row["pack_id"]: row for row in payload["packs"]}
        ghana = by_id["pack-a-ghana"]
        assert ghana["zoom_min"] == 0 and ghana["zoom_max"] == 14
        assert ghana["stretch_zoom_max"] == 16
        assert ghana["tile_format"] == "mvt"
        assert ghana["installed"] is None

    def test_packs_verb_empty_registry(self, tmp_path, capsys):
        context = SimpleNamespace(settings=None)
        rc = _cmd_maps(
            self._args(maps_action="packs",
                       packs_dir=str(tmp_path / "packs")), context)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        by_id = {row["pack_id"]: row for row in payload["packs"]}
        assert by_id["pack-b-world"]["installed"] is None
        assert by_id["pack-a-nigeria"]["zoom_max"] == 14
