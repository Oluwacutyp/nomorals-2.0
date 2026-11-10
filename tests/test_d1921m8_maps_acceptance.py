"""DOCS 19+21 M8 — the Atlas acceptance battery (programme close).

One continuous narrative over the REAL packs M2 built on this
box (``~/workspace/maps-packs``: pack-a-nigeria z0-14 MVT,
pack-a-lagos z15 MVT, pack-b-world z0-7 PNG), driven through the
shipped surfaces only — manifest verify, the release registry,
the local gazetteer, the local router, both renderers, the
MapAgent, the ``map`` tool, the /map chat surface and the
LiveLocationBroker over the real ConsentStore:

  install/verify/registry honesty → find Ilawe on real data →
  route on the real pack → render real views (+ an overlay) →
  agent/tool parity → the live-share cycle → the network proof
  → dark-by-default and empty-shelf honesty.

THE NETWORK PROOF is the programme's S1/S4 claim, machine
pinned: the narrative acts run with a socket guard that records
and raises on ANY connection attempt — no Google/OSM/any tile or
geocode call happens anywhere in the leg (M4's precedent, widened
to create_connection/getaddrinfo).

Two findings this battery surfaced, stated here because the
battery is where they belong:

1.  **MVT layer field rotation (FIXED in this pass).**  The M3
    decoder's ``_decode_layer`` read the Layer message rotated by
    one field (1: features, 2: keys, 3: values, 4: name) against
    the vector-tile 2.1 schema (1: name, 2: features, 3: keys,
    4: values) — and the M3/M4/M5/M7 fixture encoders shared the
    rotation, so the whole vector plane was self-consistent and
    green while never having decoded a real tile.  Against the
    real packs the gazetteer scan read 226,004 of 226,051 tiles
    as corrupt and indexed ZERO places.  The fix is the decoder's
    field map + its docstring + the four fixture encoders (the
    Feature and Value maps were already spec-true).  After the
    fix the same scan reads 226,051 tiles, 0 corrupt, 46,477
    name rows.

2.  **Routing seam fragmentation (RESOLVED by the
    routing-design pass; originally a NAMED REMAINDER).**
    The router joined ways only at EXACTLY equal global-pixel
    nodes (M5's stated zero-tolerance rule) and snapped an
    endpoint to its single nearest node.  Planetiler simplifies
    geometry per tile, so the same road's vertices differ by
    metres across tile seams, and a town label's nearest vertex
    can be an isolated stub.  Measured on this corridor then:
    the Ilawe→Ado Ekiti graph held 524 components; the two
    towns' components passed 19.1 m apart unjoined at (7.6166,
    5.1419); the Ado label's nearest node was a 3-node stub
    33 m away while the town's main network ran 48 m off, so
    label-to-label Ilawe→Ado Ekiti answered a truthful
    "disconnected" — pinned at the time as honest behaviour.
    The routing-design pass fixed the cause, not the answer:
    seam bridging joins dangling way ends to the segment they
    stop within 10 m of (zero-distance, labelled joins), and
    the snap is multi-candidate, so each label anchors on a
    node that routes.  The corridor now holds ~21 components
    and the label-to-label route answers with a real route
    (pinned below in its new shape); genuinely unconnected
    pairs still answer "disconnected" (the M5 fixture pins).

Bounded-extract disclosure (the brief's sanctioned shape): a
full-pack gazetteer build over pack-a-nigeria takes ~645 s
(measured; 46,477 entries, 0 corrupt tiles — probe log in the
pass's verify dir), impractical inside a test run.  The find /
near / agent narrative therefore indexes a BOUNDED EKITI REGION
EXTRACT of pack-a-nigeria — real tiles copied out of the real
pack, pinned into a manifest by the shipped build_manifest, and
scanned by the shipped gazetteer.  Nothing about the data path
is faked; only its breadth is bounded, and the extract pack is
named as an extract everywhere it appears.  Verify, routing and
rendering run against the full real packs directly.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.location.live import SCOPE_LIVE_SHARE, LiveLocationBroker
from nomorals.location.primitives import LatLon, haversine_m
from nomorals.location.privacy import ConsentStore
from nomorals.maps.coords import BBox, tile_x_to_lon, tile_y_to_lat
from nomorals.maps.gazetteer import Gazetteer
from nomorals.maps.manifest import (
    build_manifest,
    load_manifest,
    verify_signature,
    write_manifest,
)
from nomorals.maps.overlays import OverlayFeature
from nomorals.maps.packs import verify_pack_dir
from nomorals.maps.pngcodec import decode_png
from nomorals.maps.receipts import MapReceiptStore
from nomorals.maps.registry import ReleaseRegistry
from nomorals.maps.render import FixtureRenderer
from nomorals.maps.routing import Router
from nomorals.maps.style import DEFAULT_STYLE, style_hash
from nomorals.maps.vector_render import VectorRenderer

REAL_PACKS = Path(
    os.environ.get("NOMORALS_M8_REAL_PACKS")
    or Path(__file__).resolve().parents[2] / "maps-packs")
REAL_PACKS_OK = (REAL_PACKS / "releases.json").is_file() and all(
    (REAL_PACKS / pid / "manifest.json").is_file()
    for pid in ("pack-a-nigeria", "pack-a-lagos", "pack-b-world"))

#: The Ekiti box the bounded extract covers (both route towns and
#: the Ireje/Erifun villages sit inside it).
EKITI_BOX = BBox(west=4.95, south=7.45, east=5.40, north=7.80)
EXTRACT_ID = "pack-a-nigeria-ekiti-extract"

#: Ilawe-Ekiti's real neighbourhood (the gazetteer's own hit is
#: asserted inside this band, never as an exact float).
ILAWE_LAT = (7.55, 7.65)
ILAWE_LON = (5.05, 5.16)

GOLD = bytes((232, 179, 75))  # #e8b34b, the MapOverlay house gold


# ── the socket guard (the network proof's instrument) ────────────


class _SocketGuard:
    """Deny networking at the socket layer, recording attempts.

    Any connect/DNS attempt raises AssertionError AND is recorded,
    so a test can assert the count is zero at the end (a shipped
    path that tried the network fails the test either way — the
    attempt is a finding, never something to mock away).
    """

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def __enter__(self) -> _SocketGuard:
        attempts = self.attempts

        def blocked(what):
            def _raise(*args, **kwargs):
                attempts.append(what)
                raise AssertionError(
                    "network attempted via " + what)
            return _raise

        self._patches = [
            mock.patch.object(socket, "socket", blocked("socket")),
            mock.patch.object(socket, "create_connection",
                              blocked("create_connection")),
            mock.patch.object(socket, "getaddrinfo",
                              blocked("getaddrinfo")),
            mock.patch.object(socket, "gethostbyname",
                              blocked("gethostbyname")),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc) -> bool:
        for patch in self._patches:
            patch.stop()
        return False


# ── module fixtures: the real shelf + the bounded extract ───────

_S = SimpleNamespace()  # the module fixture bag


def setUpModule():
    if not REAL_PACKS_OK:
        return
    tmp = tempfile.TemporaryDirectory(prefix="m8-acceptance-")
    _S.tmp = tmp
    root = Path(tmp.name)
    _S.root = root
    _S.home = root / "home"
    _S.home.mkdir(parents=True)
    _S.store = MapReceiptStore(db=None, home=str(_S.home))

    # the real shelf: symlinks to the installed packs + the
    # registry file (copied — the registry is only read here).
    shelf = root / "shelf-real"
    shelf.mkdir()
    for pid in ("pack-a-nigeria", "pack-a-lagos", "pack-b-world"):
        (shelf / pid).symlink_to(REAL_PACKS / pid, target_is_directory=True)
    shutil.copyfile(REAL_PACKS / "releases.json",
                    shelf / "releases.json")
    _S.shelf_real = shelf

    # item 1's verify, run once through the shipped path (the
    # Nigeria pack's member hashing is the slow leg, ~2 min).
    _S.verdicts = {}
    _S.manifests = {}
    for pid in ("pack-a-nigeria", "pack-a-lagos", "pack-b-world"):
        _S.manifests[pid] = load_manifest(REAL_PACKS / pid)
        verdict, receipt = verify_pack_dir(
            REAL_PACKS / pid, store=_S.store)
        _S.verdicts[pid] = (verdict, receipt)

    # the bounded Ekiti extract of pack-a-nigeria: real tiles,
    # pinned by the shipped manifest machinery.
    src = _S.manifests["pack-a-nigeria"]
    ext_dir = root / "shelf-extract" / EXTRACT_ID
    for member in src.members:
        parts = member.path.split("/")
        if len(parts) != 4 or parts[0] != "tiles":
            continue
        z, x = int(parts[1]), int(parts[2])
        y = int(parts[3].split(".")[0])
        tw, te = tile_x_to_lon(x, z), tile_x_to_lon(x + 1, z)
        tn, ts = tile_y_to_lat(y, z), tile_y_to_lat(y + 1, z)
        if te < EKITI_BOX.west or tw > EKITI_BOX.east \
                or ts > EKITI_BOX.north or tn < EKITI_BOX.south:
            continue
        dst = ext_dir / member.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(REAL_PACKS / "pack-a-nigeria" / member.path, dst)
        except OSError:
            shutil.copyfile(
                REAL_PACKS / "pack-a-nigeria" / member.path, dst)
    ext_manifest = build_manifest(
        ext_dir, pack_id=EXTRACT_ID,
        name="Devon Atlas — Nigeria, Ekiti extract (from pack-a-nigeria)",
        bbox=EKITI_BOX, zoom_min=0, zoom_max=14,
        attribution=src.attribution, tile_format="mvt")
    write_manifest(ext_dir, ext_manifest)
    _S.extract_dir = ext_dir
    _S.extract_manifest = ext_manifest
    _S.shelf_extract = ext_dir.parent

    # the gazetteer index over the extract, built once by the
    # shipped indexer (measured ~2.5 s for 506 tiles, 2,115 rows).
    _S.gaz = Gazetteer(_S.shelf_extract, store=_S.store)
    _S.built = _S.gaz.build([(ext_dir, ext_manifest)])


def tearDownModule():
    tmp = getattr(_S, "tmp", None)
    if tmp is not None:
        tmp.cleanup()


def _candidate_near(result, lat, lon, within_km=2.5):
    """The gazetteer candidate nearest a reference point, asserted
    to actually be near it (the selection is verified, not assumed)."""
    best = None
    for cand in result.candidates:
        dist = haversine_m(LatLon(cand.lat, cand.lon),
                           LatLon(lat, lon)) / 1000.0
        if best is None or dist < best[1]:
            best = (cand, dist)
    if best is None or best[1] > within_km:
        raise AssertionError(
            "no candidate within " + str(within_km) + " km of "
            + str((lat, lon)) + ": " + repr(result.candidates))
    return best[0]


def _gold_pixels(png: bytes) -> int:
    _w, _h, rgb = decode_png(png)
    return sum(1 for i in range(0, len(rgb) - 2, 3)
               if rgb[i:i + 3] == GOLD)


def _nonbackground_share(png: bytes) -> float:
    from collections import Counter

    w, h, rgb = decode_png(png)
    counts = Counter()
    for i in range(0, len(rgb), 3):
        counts[rgb[i:i + 3]] += 1
    _bg, bg_n = counts.most_common(1)[0]
    return 1.0 - bg_n / float(w * h)


# ── 1 · install → verify → registry → /map packs ────────────────


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestRealPackVerifyAndRegistry(unittest.TestCase):

    def test_real_packs_verify_through_the_shipped_path(self):
        for pid in ("pack-a-nigeria", "pack-a-lagos", "pack-b-world"):
            verdict, receipt = _S.verdicts[pid]
            self.assertTrue(verdict.ok, verdict.summary())
            self.assertEqual(verdict.missing, ())
            self.assertEqual(verdict.mismatched, ())
            self.assertEqual(verdict.extra, ())
            self.assertEqual(verdict.pack_id, pid)
            self.assertEqual(verdict.digest, _S.manifests[pid].digest)
            self.assertEqual(receipt.operation, "pack_verify")
            self.assertEqual(receipt.outcome, "ok")
            self.assertEqual(receipt.pack_digests, (verdict.digest,))

    def test_signature_state_is_unsigned_never_verified(self):
        for pid, manifest in _S.manifests.items():
            self.assertEqual(manifest.signature_state, "unsigned")
            self.assertEqual(verify_signature(manifest), "unsigned")
            self.assertEqual(_S.verdicts[pid][0].signature_state,
                             "unsigned")

    def test_registry_reads_the_releases_as_installed(self):
        registry = ReleaseRegistry(_S.shelf_real)
        entries = {e.pack_id: e for e in registry.all()}
        self.assertEqual(sorted(entries),
                         ["pack-a-lagos", "pack-a-nigeria",
                          "pack-b-world"])
        for pid, entry in entries.items():
            self.assertEqual(entry.digest, _S.manifests[pid].digest)
            self.assertEqual(entry.signature_state, "unsigned")
            self.assertEqual(entry.tile_format,
                             _S.manifests[pid].tile_format)
        # the release digests M2 recorded (the install evidence)
        self.assertTrue(entries["pack-a-nigeria"].digest.startswith(
            "138c6e6c"))
        self.assertTrue(entries["pack-b-world"].digest.startswith(
            "54214d95"))

    def test_pack_verify_receipts_are_in_the_store(self):
        receipts = _S.store.recent(limit=50, operation="pack_verify")
        ok_ids = {r.pack_refs[0] for r in receipts
                  if r.outcome == "ok"}
        self.assertTrue(
            {"pack-a-nigeria", "pack-a-lagos", "pack-b-world"}
            <= ok_ids)


# ── runtime rig (the M6 idiom) ───────────────────────────────────


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


class _FakeAdapter:
    def __init__(self):
        self.texts: list = []
        self.pins: list = []

    def send(self, chat, text, **kw):
        self.texts.append((chat.key, text))
        return SimpleNamespace(ok=True)

    def send_location(self, chat, latitude, longitude, **kw):
        self.pins.append((chat.key, latitude, longitude))
        return SimpleNamespace(ok=True)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent: list = []

    def send_file(self, platform, chat, path, *, caption="",
                  kind="file", name=""):
        payload = Path(path).read_bytes()
        self.sent.append({"platform": platform, "path": path,
                          "caption": caption, "kind": kind,
                          "name": name, "bytes": payload,
                          "existed": Path(path).exists()})
        return SimpleNamespace(ok=True)


def _runtime(home: Path, adapter: _FakeAdapter):
    from nomorals.agents.partner.runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = _ctx(str(home))
    rt._owner_chats = {"telegram:1"}
    rt._owner_sender_ids = set()
    rt.gateway = _Gateway({"telegram": adapter})
    return rt


class _SurfaceRig(unittest.TestCase):
    """A PartnerRuntime over a tmp home + a chosen packs dir."""

    extra_env: dict = {}

    def packs_dir(self) -> Path:
        raise NotImplementedError

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m8-surface-")
        self.home = Path(self.tmp.name)
        self.adapter = _FakeAdapter()
        self.rt = _runtime(self.home, self.adapter)
        env = {"NM_MAPS_ENABLED": "1",
               "NM_MAPS_PACKS_DIR": str(self.packs_dir())}
        env.update(self.extra_env)
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestSurfacePacksList(_SurfaceRig):
    """Item 1's surface leg: /map packs over the real shelf."""

    def packs_dir(self) -> Path:
        return _S.shelf_real

    def test_map_packs_lists_all_three_real_packs_verified(self):
        out = self.rt._control_map("packs", message=_msg())
        self.assertIn("3 pack(s)", out)
        for pid in ("pack-a-nigeria", "pack-a-lagos", "pack-b-world"):
            self.assertIn(pid, out)
        self.assertEqual(out.count("✓ verified"), 3)
        self.assertEqual(out.count("signature unsigned"), 3)
        self.assertIn("© OpenStreetMap contributors", out)


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestSurfaceOnExtract(_SurfaceRig):
    """The /map find + route surfaces over the extract shelf."""

    def packs_dir(self) -> Path:
        return _S.shelf_extract

    def test_map_find_ilawe_answers_and_sends_the_winner_photo(self):
        out = self.rt._control_map("find Ilawe", message=_msg())
        self.assertIn("Ilawe-Ekiti", out)
        self.assertIn("local Atlas index", out)
        # the clear winner also gets its map photo, attribution
        # riding the caption (manifest data, a license fact)
        self.assertEqual(len(self.rt.gateway.sent), 1)
        sent = self.rt.gateway.sent[0]
        self.assertTrue(sent["existed"])
        self.assertTrue(sent["bytes"].startswith(b"\x89PNG"))
        self.assertIn("© OpenStreetMap contributors",
                      sent["caption"])

    def test_map_route_names_the_road_answer(self):
        out = self.rt._control_map(
            "route Ireje to Ekiti State Government Territory",
            message=_msg())
        # the destination label is the gazetteer's stored name
        # form (lowercase, as the mappers recorded it)
        self.assertIn("🧭 Ireje → ekiti state government territory",
                      out)
        self.assertIn("by road (local Atlas routing", out)
        self.assertEqual(len(self.rt.gateway.sent), 1)
        self.assertIn("© OpenStreetMap contributors",
                      self.rt.gateway.sent[0]["caption"])


# ── 2 · find Ilawe on the real data ──────────────────────────────


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestRealGazetteer(unittest.TestCase):

    def test_the_extract_index_built_from_real_tiles(self):
        # 506 real tiles → 2,115 name rows at probe time; the band
        # is generous, the point is a real index of real places.
        self.assertGreater(_S.built.total_entries, 1500)
        result = [p for p in _S.built.packs
                  if p.pack_id == EXTRACT_ID][0]
        self.assertEqual(result.corrupt_tiles, ())
        self.assertGreater(result.tiles_scanned, 400)

    def test_find_ilawe_resolves_to_the_real_place(self):
        result = _S.gaz.search("Ilawe")
        self.assertTrue(result.candidates)
        top = result.candidates[0]
        self.assertEqual(top.name, "Ilawe-Ekiti")
        self.assertIn(top.kind, ("town", "city", "village"))
        self.assertTrue(ILAWE_LAT[0] <= top.lat <= ILAWE_LAT[1],
                        top)
        self.assertTrue(ILAWE_LON[0] <= top.lon <= ILAWE_LON[1],
                        top)
        self.assertEqual(top.pack_id, EXTRACT_ID)
        # the answer is local, offline, and receipted as such
        self.assertEqual(result.posture, "local")
        self.assertIsNotNone(result.receipt)
        self.assertEqual(result.receipt.posture, "offline")

    def test_find_ilawe_with_diacritics_folds_to_the_same_place(self):
        plain = _S.gaz.search("Ilawe").candidates[0]
        folded = _S.gaz.search("Ìláwé").candidates[0]
        self.assertEqual(folded.name, plain.name)
        self.assertAlmostEqual(folded.lat, plain.lat, places=6)
        self.assertAlmostEqual(folded.lon, plain.lon, places=6)

    def test_reverse_at_ilawe_names_ilawe(self):
        top = _S.gaz.search("Ilawe").candidates[0]
        result = _S.gaz.reverse(top.lat, top.lon)
        self.assertTrue(result.candidates)
        nearest = result.candidates[0]
        self.assertEqual(nearest.name, "Ilawe-Ekiti")
        self.assertLess(nearest.distance_km, 1.0)
        self.assertEqual(result.posture, "local")


# ── 3 · route on the REAL pack ───────────────────────────────────


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestRealRouting(unittest.TestCase):

    def _endpoints(self):
        ireje = _candidate_near(_S.gaz.search("Ireje"),
                                7.5956, 5.2524)
        erifun = _candidate_near(_S.gaz.search("Erifun"),
                                 7.6110, 5.2650)
        return ireje, erifun

    def test_route_between_two_real_places_on_the_real_pack(self):
        ireje, erifun = self._endpoints()
        router = Router(_S.root / "routing-home", store=_S.store)
        result = router.route(
            REAL_PACKS / "pack-a-nigeria",
            _S.manifests["pack-a-nigeria"],
            LatLon(ireje.lat, ireje.lon), LatLon(erifun.lat, erifun.lon),
            from_label=ireje.name, to_label=erifun.name)
        self.assertTrue(result.ok, result.detail)
        hav = haversine_m(LatLon(ireje.lat, ireje.lon),
                          LatLon(erifun.lat, erifun.lon))
        self.assertGreaterEqual(result.distance_m, hav)
        self.assertLessEqual(result.distance_m, 2.5 * hav)
        self.assertTrue(result.steps)
        self.assertGreater(result.duration_s, 0)
        self.assertIsNotNone(result.receipt)
        self.assertEqual(result.receipt.operation, "route")
        self.assertEqual(result.receipt.outcome, "ok")
        stored = _S.store.recent(limit=20, operation="route")
        self.assertIn(result.receipt.receipt_id,
                      [r.receipt_id for r in stored])

    def test_route_ilawe_to_ado_label_anchors_now_routes(self):
        # PIN CHANGED BY THE ROUTING-DESIGN PASS, deliberately:
        # this test used to pin the label-to-label pair as
        # "truthfully disconnected" (the seam-fragmentation
        # remainder, finding 2 above).  Seam bridging +
        # multi-candidate snap removed the cause — the corridor's
        # components merged and each label now anchors on a node
        # that routes — so the honest behaviour today is a real
        # route, and this pin asserts it (distance sanity in the
        # sibling Ireje→Erifun test's shape).  A genuinely
        # unconnected pair still answers "disconnected": that
        # outcome stays pinned in the M5 fixture tests.
        ilawe = _S.gaz.search("Ilawe").candidates[0]
        ado = _S.gaz.search("Ado Ekiti").candidates[0]
        router = Router(_S.root / "routing-home", store=_S.store)
        result = router.route(
            REAL_PACKS / "pack-a-nigeria",
            _S.manifests["pack-a-nigeria"],
            LatLon(ilawe.lat, ilawe.lon), LatLon(ado.lat, ado.lon),
            from_label=ilawe.name, to_label=ado.name)
        self.assertTrue(result.ok, result.detail)
        hav = haversine_m(LatLon(ilawe.lat, ilawe.lon),
                          LatLon(ado.lat, ado.lon))
        self.assertGreaterEqual(result.distance_m, hav * 0.95)
        self.assertLessEqual(result.distance_m, 2.5 * hav)
        self.assertTrue(result.steps)
        self.assertEqual(result.provider, "local")
        self.assertIsNotNone(result.receipt)
        self.assertEqual(result.receipt.operation, "route")
        self.assertEqual(result.receipt.outcome, "ok")


# ── 4 · render real views ────────────────────────────────────────


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestRealRender(unittest.TestCase):

    def test_vector_render_of_a_real_lagos_view(self):
        box = BBox(west=3.3712, south=6.5164, east=3.3872,
                   north=6.5324)
        result = VectorRenderer(
            REAL_PACKS / "pack-a-lagos", _S.manifests["pack-a-lagos"],
            store=_S.store).render(box, 15, size=(512, 512))
        self.assertIsNotNone(result.image, result.receipt.detail)
        self.assertEqual((result.width, result.height), (512, 512))
        self.assertEqual(result.missing_tiles, ())
        w, h, _rgb = decode_png(result.image)
        self.assertEqual((w, h), (512, 512))
        # a real city view: most of the frame is drawn map, not
        # background (measured 0.75 at probe time)
        self.assertGreater(_nonbackground_share(result.image), 0.2)
        self.assertEqual(result.receipt.operation, "render")
        self.assertIn("style devon-dark ("
                      + style_hash(DEFAULT_STYLE)[:12] + ")",
                      result.receipt.detail)
        self.assertIn("tiles 6/6 present", result.receipt.detail)

    def test_raster_render_of_the_world_pack(self):
        box = BBox(west=-20.0, south=0.0, east=35.0, north=25.0)
        result = FixtureRenderer(
            REAL_PACKS / "pack-b-world", _S.manifests["pack-b-world"],
            store=_S.store).render(box, 4, size=(512, 512))
        self.assertIsNotNone(result.image, result.receipt.detail)
        w, h, _rgb = decode_png(result.image)
        self.assertEqual((w, h), (512, 512))
        # land is drawn over the ocean base (measured 0.035)
        self.assertGreater(_nonbackground_share(result.image), 0.01)
        self.assertEqual(result.receipt.operation, "render")

    def test_overlay_over_ilawe_renders_with_the_fill_honesty_note(self):
        env = mock.patch.dict(
            os.environ,
            {"NM_MAPS_PACKS_DIR": str(_S.shelf_extract)})
        env.start()
        try:
            agent = _agent_on_extract()
            self._overlay_over_ilawe(agent)
        finally:
            env.stop()

    def _overlay_over_ilawe(self, agent):
        ilawe = _S.gaz.search("Ilawe").candidates[0]
        lat, lon = ilawe.lat, ilawe.lon
        agent.overlay_store().create("ilawe farm", [
            OverlayFeature("polygon", (
                (lat - 0.004, lon - 0.004), (lat + 0.004, lon - 0.004),
                (lat + 0.004, lon + 0.004), (lat - 0.004, lon + 0.004)),
                "farm"),
            OverlayFeature("marker", ((lat, lon),), "home"),
        ])
        ticket = agent.overlay_render("ilawe farm")
        self.assertEqual(ticket["outcome"], "ok", ticket)
        self.assertTrue(Path(ticket["image_path"]).is_file())
        notes = " ".join(ticket["notes"])
        self.assertIn("outline only", notes)
        self.assertIn("no fill", notes)
        # the overlay actually paints: more house-gold pixels than
        # the same view without the overlay drawn on it
        base = agent.view(bbox=BBox(**ticket["bbox"]),
                          zoom=ticket["zoom"])
        self.assertEqual(base["outcome"], "ok", base)
        overlay_gold = _gold_pixels(
            Path(ticket["image_path"]).read_bytes())
        base_gold = _gold_pixels(
            Path(base["image_path"]).read_bytes())
        self.assertGreater(overlay_gold, base_gold)


# ── 5 · agent + tool parity on real data ─────────────────────────


def _agent_on_extract():
    from nomorals.agents.map import MapAgent

    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=str(_S.home)), db=None)
    return MapAgent(ctx)


def _tool_on_extract():
    from nomorals.tools import map as map_tools
    from nomorals.tools.registry import ToolRegistry

    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=str(_S.home)), db=None)
    registry = ToolRegistry(ctx)
    map_tools.register(registry)
    return registry


class _ExtractEnv(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ,
            {"NM_MAPS_PACKS_DIR": str(_S.shelf_extract)})
        self.env.start()

    def tearDown(self):
        self.env.stop()


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestAgentToolParity(_ExtractEnv):

    _SHARED = ("operation", "outcome", "query", "candidates",
               "posture", "provider")

    def test_find_parity_agent_vs_tool(self):
        agent = _agent_on_extract()
        registry = _tool_on_extract()
        via_agent = agent.find("Ilawe")
        via_tool = registry.call(
            "map", operation="find", query="Ilawe").value
        self.assertEqual(via_agent["outcome"], "ok")
        self.assertEqual(via_tool["outcome"], "ok")
        for field in self._SHARED:
            self.assertEqual(via_agent[field], via_tool[field],
                             field)
        self.assertEqual(via_agent["candidates"][0]["name"],
                         "Ilawe-Ekiti")
        self.assertTrue(via_agent["receipt_id"])
        self.assertTrue(via_tool["receipt_id"])

    def test_route_parity_and_the_same_distance_as_the_bare_router(self):
        ireje = _candidate_near(_S.gaz.search("Ireje"),
                                7.5956, 5.2524)
        erifun = _candidate_near(_S.gaz.search("Erifun"),
                                 7.6110, 5.2650)
        start = str(ireje.lat) + "," + str(ireje.lon)
        end = str(erifun.lat) + "," + str(erifun.lon)
        agent = _agent_on_extract()
        registry = _tool_on_extract()
        via_agent = agent.route(start, end)
        via_tool = registry.call(
            "map", operation="route", start=start, end=end).value
        self.assertEqual(via_agent["outcome"], "ok", via_agent)
        for field in ("operation", "outcome", "distance_m",
                      "duration_s", "steps", "from_label",
                      "to_label", "pack_id"):
            self.assertEqual(via_agent[field], via_tool[field],
                             field)
        # and the agent's number is the bare router's number over
        # the full real pack — one graph, one distance, all planes
        router = Router(_S.root / "routing-home", store=_S.store)
        bare = router.route(
            REAL_PACKS / "pack-a-nigeria",
            _S.manifests["pack-a-nigeria"],
            LatLon(ireje.lat, ireje.lon), LatLon(erifun.lat, erifun.lon),
            from_label=ireje.name, to_label=erifun.name)
        self.assertTrue(bare.ok)
        self.assertAlmostEqual(via_agent["distance_m"],
                               bare.distance_m, delta=1.0)

    def test_view_parity_agent_vs_tool(self):
        ilawe = _S.gaz.search("Ilawe").candidates[0]
        agent = _agent_on_extract()
        registry = _tool_on_extract()
        via_agent = agent.view(ilawe.lat, ilawe.lon, zoom=13)
        via_tool = registry.call(
            "map", operation="view",
            point=str(ilawe.lat) + "," + str(ilawe.lon),
            zoom=13).value
        self.assertEqual(via_agent["outcome"], "ok", via_agent)
        for field in ("outcome", "pack_id", "image_path", "width",
                      "height", "attribution"):
            self.assertEqual(via_agent[field], via_tool[field],
                             field)
        self.assertEqual(via_agent["pack_id"], EXTRACT_ID)
        self.assertTrue(Path(via_agent["image_path"]).is_file())


# ── 6 · the live-share cycle ─────────────────────────────────────


class _Clock:
    def __init__(self, now: float = 1_700_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Sender:
    def __init__(self):
        self.calls: list = []

    def __call__(self, chat_key: str, loc: LatLon) -> bool:
        self.calls.append((chat_key, loc))
        return True


class TestLiveShareCycleBroker(unittest.TestCase):
    """The full decision cycle over the real ConsentStore, on an
    injected clock: no-session → forwarded → consent-denied →
    forwarded → expired, every step journaled with its reason."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m8-share-")
        self.root = Path(self.tmp.name)
        self.clock = _Clock()
        self.sender = _Sender()
        self.consent = ConsentStore(self.root)
        self.broker = LiveLocationBroker(
            self.root, consent=self.consent, clock=self.clock,
            sender=self.sender)
        self.fix = LatLon(7.5989, 5.1057)  # Ilawe-Ekiti

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_cycle_journaled_with_named_reasons(self):
        broker, consent = self.broker, self.consent
        # no session: a withheld record carrying NO coordinates
        decisions = broker.evaluate("owner", self.fix)
        self.assertEqual(decisions[0]["decision"], "withheld")
        self.assertEqual(decisions[0]["reason"], "no-session")
        self.assertIsNone(decisions[0]["lat"])
        self.assertIsNone(decisions[0]["lon"])
        # grant → session → the fix forwards, coarsened only
        consent.grant("owner", SCOPE_LIVE_SHARE)
        broker.start_session("owner", ("telegram:55",),
                             ttl_s=3600.0, precision="neighborhood")
        self.clock.advance(1)
        decisions = broker.evaluate("owner", self.fix)
        self.assertEqual(decisions[0]["decision"], "forwarded")
        self.assertEqual(decisions[0]["reason"], "grant-live")
        self.assertEqual(self.sender.calls,
                         [("telegram:55", LatLon(7.60, 5.11))])
        # the journal record holds ONLY the coarsened coordinates
        self.assertEqual(decisions[0]["lat"], 7.60)
        self.assertEqual(decisions[0]["lon"], 5.11)
        # revoke lands on the very next update
        consent.revoke("owner", SCOPE_LIVE_SHARE)
        self.clock.advance(1)
        decisions = broker.evaluate("owner", self.fix)
        self.assertEqual(decisions[0]["decision"], "withheld")
        self.assertEqual(decisions[0]["reason"], "consent-denied")
        self.assertEqual(len(self.sender.calls), 1)
        # a fresh grant + a short session: forwards, then expiry
        # flips lazily and withholds on its own clock
        consent.grant("owner", SCOPE_LIVE_SHARE)
        broker.start_session("owner", ("telegram:55",),
                             ttl_s=300.0, precision="city")
        self.clock.advance(301)
        decisions = broker.evaluate("owner", self.fix)
        by_reason = {d["reason"]: d for d in decisions}
        self.assertEqual(by_reason["expired"]["decision"],
                         "withheld")
        states = {s.id: s.state
                  for s in broker.sessions("owner",
                                           include_terminal=True)}
        self.assertIn("expired", states.values())
        # the durable journal tells the whole cycle, in order
        # (the first three carry distinct injected-clock times;
        # the last two share a timestamp, so they compare as a set)
        journal = broker.recent_decisions(limit=50, subject="owner")
        reasons = [d["reason"] for d in
                   sorted(journal, key=lambda d: d["at"])]
        self.assertEqual(reasons[:3],
                         ["no-session", "grant-live",
                          "consent-denied"])
        self.assertEqual(sorted(reasons[3:]),
                         ["expired", "grant-live"])


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestLiveShareSurface(_SurfaceRig):
    """Item 6 at the surface: the /map share command itself
    records the grant; pins flow through the platform adapter;
    a ConsentStore revoke withholds the next update."""

    extra_env = {"NM_MAPS_LIVE_SHARE": "1"}

    def packs_dir(self) -> Path:
        return _S.shelf_extract

    def test_share_cycle_via_the_map_share_path(self):
        out = self.rt._control_map(
            "share telegram:55 30 neighborhood", message=_msg())
        self.assertIn("live share started", out)
        consent = ConsentStore(self.home / "location")
        self.assertTrue(
            consent.has_consent("owner", SCOPE_LIVE_SHARE))
        self.assertEqual(len(self.adapter.texts), 1)
        self.assertIn("neighborhood", self.adapter.texts[0][1])
        broker = self.rt._live_broker()
        decisions = broker.evaluate("owner", LatLon(7.5989, 5.1057))
        self.assertEqual(decisions[0]["decision"], "forwarded")
        self.assertEqual(self.adapter.pins,
                         [("telegram:55", 7.60, 5.11)])
        # revoke in the store → the next update is withheld
        consent.revoke("owner", SCOPE_LIVE_SHARE)
        decisions = broker.evaluate("owner", LatLon(7.5989, 5.1057))
        self.assertEqual(decisions[0]["decision"], "withheld")
        self.assertEqual(decisions[0]["reason"], "consent-denied")
        self.assertEqual(len(self.adapter.pins), 1)
        # and the surface stop ends the session durably
        out = self.rt._control_map("share stop", message=_msg())
        self.assertIn("stopped", out)
        self.assertEqual(broker.sessions("owner"), [])


# ── 7 · the network proof ────────────────────────────────────────


@unittest.skipUnless(REAL_PACKS_OK,
                     "M2's real packs are not installed on this box")
class TestNetworkProof(_ExtractEnv):
    """The whole narrative with networking denied at the socket
    layer: zero provider calls is asserted, not assumed."""

    def test_no_socket_is_opened_across_the_narrative(self):
        guard = _SocketGuard()
        with guard:
            # geocode + reverse on the real-data index
            found = _S.gaz.search("Ilawe")
            self.assertTrue(found.candidates)
            near = _S.gaz.reverse(found.candidates[0].lat,
                                  found.candidates[0].lon)
            self.assertTrue(near.candidates)
            # route over the real pack
            ireje = _candidate_near(_S.gaz.search("Ireje"),
                                    7.5956, 5.2524)
            erifun = _candidate_near(_S.gaz.search("Erifun"),
                                     7.6110, 5.2650)
            router = Router(_S.root / "routing-home", store=_S.store)
            routed = router.route(
                REAL_PACKS / "pack-a-nigeria",
                _S.manifests["pack-a-nigeria"],
                LatLon(ireje.lat, ireje.lon),
                LatLon(erifun.lat, erifun.lon),
                from_label=ireje.name, to_label=erifun.name)
            self.assertTrue(routed.ok)
            # renders from both real pack formats
            box = BBox(west=3.3712, south=6.5164, east=3.3872,
                       north=6.5324)
            rendered = VectorRenderer(
                REAL_PACKS / "pack-a-lagos",
                _S.manifests["pack-a-lagos"],
                store=_S.store).render(box, 15, size=(256, 256))
            self.assertIsNotNone(rendered.image)
            # agent find/view over the extract shelf
            agent = _agent_on_extract()
            self.assertEqual(agent.find("Ilawe")["outcome"], "ok")
            view = agent.view(found.candidates[0].lat,
                              found.candidates[0].lon, zoom=13)
            self.assertEqual(view["outcome"], "ok")
            # a broker evaluation (file state only)
            root = _S.root / "proof-location"
            consent = ConsentStore(root)
            consent.grant("owner", SCOPE_LIVE_SHARE)
            broker = LiveLocationBroker(
                root, consent=consent, sender=_Sender())
            broker.start_session("owner", ("telegram:55",),
                                 ttl_s=600.0, precision="exact")
            decisions = broker.evaluate(
                "owner", LatLon(7.5989, 5.1057))
            self.assertEqual(decisions[0]["decision"], "forwarded")
        self.assertEqual(guard.attempts, [])


# ── 8 · dark-by-default + empty-shelf honesty ────────────────────


class TestDarkByDefault(unittest.TestCase):
    """Maps disabled: the /map surface answers its named off
    state for every verb — no traceback, no partial surface."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m8-dark-")
        self.home = Path(self.tmp.name)
        (self.home / "packs").mkdir(parents=True)
        self.adapter = _FakeAdapter()
        self.rt = _runtime(self.home, self.adapter)
        self.env = mock.patch.dict(os.environ, {
            "NM_MAPS_ENABLED": "0",
            "NM_MAPS_LIVE_SHARE": "1",
            "NM_MAPS_PACKS_DIR": str(self.home / "packs"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_every_verb_answers_the_named_off_state(self):
        for tail in ("packs", "find Ilawe", "near 7.5989,5.1057",
                     "route Ireje to Erifun", "view 7.5989,5.1057",
                     "share telegram:55 30", "overlay list"):
            out = self.rt._control_map(tail, message=_msg())
            self.assertIn("Devon Atlas is off on this host", out,
                          tail)
            self.assertIn("NM_MAPS_ENABLED=1", out, tail)
        self.assertEqual(self.adapter.texts, [])
        self.assertEqual(self.adapter.pins, [])


class TestEmptyShelfHonesty(unittest.TestCase):
    """Packs dir empty: every surface answers its honest no-cover
    state, and no provider fallback occurs (the guard stands
    watch — a fallback attempt would raise, not pass silently)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m8-empty-")
        self.home = Path(self.tmp.name)
        self.packs = self.home / "packs"
        self.packs.mkdir(parents=True)
        self.adapter = _FakeAdapter()
        self.rt = _runtime(self.home, self.adapter)
        self.env = mock.patch.dict(os.environ, {
            "NM_MAPS_ENABLED": "1",
            "NM_MAPS_PACKS_DIR": str(self.packs),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_agent_tickets_are_unavailable_with_the_reason(self):
        from nomorals.agents.map import MapAgent

        ctx = SimpleNamespace(
            settings=SimpleNamespace(home_path=str(self.home)),
            db=None)
        agent = MapAgent(ctx)
        with _SocketGuard() as guard:
            for ticket in (agent.packs(), agent.find("Ilawe"),
                           agent.near(7.5989, 5.1057),
                           agent.route("Ireje", "Erifun"),
                           agent.view(7.5989, 5.1057)):
                self.assertEqual(ticket["outcome"], "unavailable",
                                 ticket)
                self.assertIn("no map packs installed",
                              ticket["detail"], ticket)
        self.assertEqual(guard.attempts, [])

    def test_surface_texts_name_the_empty_shelf_and_no_fallback(self):
        with _SocketGuard() as guard:
            find = self.rt._control_map("find Ilawe",
                                        message=_msg())
            route = self.rt._control_map("route Ireje to Erifun",
                                         message=_msg())
            view = self.rt._control_map("view 7.5989,5.1057",
                                        message=_msg())
        for out in (find, route, view):
            self.assertIn("no map packs installed", out)
            self.assertIn("no provider fallback", out)
        self.assertEqual(guard.attempts, [])
        self.assertEqual(self.rt.gateway.sent, [])


if __name__ == "__main__":
    unittest.main()
