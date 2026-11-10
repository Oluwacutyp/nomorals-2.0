"""Devon Atlas M9 — personal marks: /map mark, the marks
store, marks-first retrieval, and the pending-mark hook.

M9 saves where the OWNER is standing: ``/map mark <label>``
captures the location of that moment — immediately when the
command message itself carries location meta, or through a
durable PENDING mark (per platform+chat+sender, 10-minute lazy
TTL) that his next location pin in that chat completes through
the same inbound seam M6 built — ahead of the broker, which
still evaluates the same share exactly as shipped.

Pins:
* the store: save/get/list/delete, case-insensitive labels,
  label-MOVE semantics (same id, new coords/time), validation
  refusals, pending arm/replace/lazy-expiry/lapsed/clear, and
  corruption degrading to empty + a named reason (file + db
  backings).
* capture flow A: a /map mark whose message carries location
  meta completes instantly — confirmation names the label, the
  gazetteer place hint when the index covers the spot (built
  beforehand; capture never builds it), else honest coords.
* capture flow B: arm → the REAL hook completes it
  (confirmation sent, pending cleared, method recorded); a
  share with NO pending marks nothing; a stranger never
  completes (operator gate + sender-keyed pending); an expired
  pending completes nothing; a WhatsApp arm can only wait —
  no inbound WA path surfaces a location today, so its pending
  survives uncompleted rather than completing falsely.
* retrieval: marks-first find through agent AND tool (kind
  "mark" ahead of gazetteer hits; answers on an empty shelf),
  the marks tool op, view-by-mark (receipt + gold marker
  pixels), route with a mark endpoint in both directions,
  /map marks, /map unmark, /map view <label>.
* M6 coexistence: with a live share running AND a pending
  armed, one owner location completes the mark AND the broker
  still forwards the pin to the session recipient.

Offline: fixture packs only; no sockets anywhere.
"""

from __future__ import annotations

import json
import time
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
from nomorals.maps.marks import (
    METHOD_MESSAGE,
    METHOD_PENDING,
    PENDING_TTL_S,
    MarkError,
    MarkStore,
    pending_key,
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


# ── the tiny MVT encoder + network pack (the M7 fixture
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


# ── the marks store (file backing) ───────────────────────────


class TestMarkStoreFile:
    def test_save_get_list_round_trip(self, tmp_path):
        store = MarkStore(home=tmp_path)
        mark, moved = store.save(
            "Hostel", A_LL[0], A_LL[1], platform="telegram",
            method=METHOD_MESSAGE, accuracy=12.5)
        assert moved is False
        assert mark.label == "Hostel"
        assert mark.accuracy == 12.5
        got = store.get("hostel")  # case-insensitive
        assert got is not None and got.mark_id == mark.mark_id
        assert [m.label for m in store.list()] == ["Hostel"]
        # durable: a fresh store over the same home reads it back
        again = MarkStore(home=tmp_path)
        assert again.get("HOSTEL") is not None
        assert (tmp_path / "maps" / "marks.json").is_file()

    def test_label_move_semantics(self, tmp_path):
        store = MarkStore(home=tmp_path)
        first, _ = store.save("hostel", A_LL[0], A_LL[1], now=1000.0)
        moved_mark, moved = store.save(
            "HOSTEL", D_LL[0], D_LL[1], now=2000.0)
        assert moved is True
        assert moved_mark.mark_id == first.mark_id  # same identity
        assert moved_mark.created_at == 1000.0      # birth kept
        assert moved_mark.updated_at == 2000.0
        assert (moved_mark.lat, moved_mark.lon) == D_LL
        assert len(store.list()) == 1               # still one mark

    def test_validation_refusals_are_named(self, tmp_path):
        store = MarkStore(home=tmp_path)
        with pytest.raises(MarkError):
            store.save("  ", A_LL[0], A_LL[1])
        with pytest.raises(MarkError):
            store.save("x" * 200, A_LL[0], A_LL[1])
        with pytest.raises(MarkError):
            store.save("spot", 91.0, 5.0)
        with pytest.raises(MarkError):
            store.save("spot", 7.0, 181.0)
        assert store.list() == ()

    def test_delete(self, tmp_path):
        store = MarkStore(home=tmp_path)
        store.save("hostel", A_LL[0], A_LL[1])
        assert store.delete("HOSTEL") is True
        assert store.get("hostel") is None
        assert store.delete("hostel") is False

    def test_pending_arm_replace_expire_clear(self, tmp_path):
        store = MarkStore(home=tmp_path)
        key = pending_key("telegram", "telegram:42", "")
        store.arm(key, "hostel", platform="telegram",
                  chat_key="telegram:42", now=1000.0)
        pend = store.pending(key, now=1001.0)
        assert pend is not None and pend.label == "hostel"
        # re-arming replaces (one pending per key)
        store.arm(key, "school", platform="telegram",
                  chat_key="telegram:42", now=1002.0)
        assert store.pending(key, now=1003.0).label == "school"
        assert len(store.pendings(now=1003.0)) == 1
        # lazy expiry: past the TTL it reads absent and lapses
        assert store.pending(key, now=1002.0 + PENDING_TTL_S + 1) \
            is None
        store.arm(key, "late", platform="telegram",
                  chat_key="telegram:42", now=1000.0)
        gone = store.lapsed(now=1000.0 + PENDING_TTL_S + 1)
        assert [p.label for p in gone] == ["late"]
        assert store.pendings(now=1000.0 + PENDING_TTL_S + 2) == ()
        # clear returns what was armed
        store.arm(key, "gym", platform="telegram",
                  chat_key="telegram:42", now=1000.0)
        cleared = store.clear(key)
        assert cleared is not None and cleared.label == "gym"
        assert store.pending(key, now=1001.0) is None

    def test_unparseable_arm_time_is_never_immortal(self, tmp_path):
        path = tmp_path / "maps" / "marks.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "marks": {},
            "pending": {"k": {"key": "k", "label": "ghost",
                              "armed_at": "not-a-time"}}}),
            encoding="utf-8")
        store = MarkStore(home=tmp_path)
        assert store.pending("k", now=1000.0) is None
        assert store.pendings(now=1000.0) == ()

    def test_corruption_degrades_to_empty_with_a_reason(
            self, tmp_path):
        path = tmp_path / "maps" / "marks.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ not json !!", encoding="utf-8")
        store = MarkStore(home=tmp_path)
        assert store.list() == ()
        assert store.get("hostel") is None
        assert store.degraded_reason()  # named, not silent
        # and a save still lands (the corrupt file is replaced)
        store.save("hostel", A_LL[0], A_LL[1])
        assert store.get("hostel") is not None


class TestMarkStoreDb:
    def test_db_backing_round_trip(self, tmp_path):
        db = Database(str(tmp_path / "marks.db"))
        db.migrate()
        store = MarkStore(db=db)
        store.save("hostel", A_LL[0], A_LL[1], platform="telegram")
        key = pending_key("telegram", "telegram:42", "")
        store.arm(key, "school", platform="telegram",
                  chat_key="telegram:42")
        fresh = MarkStore(db=db)
        assert fresh.get("hostel") is not None
        pend = fresh.pending(key)
        assert pend is not None and pend.label == "school"
        assert fresh.delete("hostel") is True


# ── the agent plane ──────────────────────────────────────────


class TestAgentMarks:
    def test_find_resolves_the_mark_first(self, shelf):
        agent = _agent(shelf)
        agent.mark_store().save("Alpha Town", D_LL[0], D_LL[1])
        ticket = agent.find("Alpha Town")
        assert ticket["outcome"] == "ok"
        first = ticket["candidates"][0]
        assert first["kind"] == "mark"
        assert first["name"] == "Alpha Town"
        assert (first["lat"], first["lon"]) == D_LL
        # the gazetteer's own Alpha Town still answers beneath it
        kinds = [c["kind"] for c in ticket["candidates"]]
        assert "town" in kinds

    def test_find_mark_answers_on_an_empty_shelf(self, tmp_path,
                                                 monkeypatch):
        packs = tmp_path / "packs"
        packs.mkdir()
        monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs))
        agent = _agent(SimpleNamespace(ctx=SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)),
            db=None)))
        agent.mark_store().save("hostel", A_LL[0], A_LL[1])
        ticket = agent.find("hostel")
        assert ticket["outcome"] == "ok"
        assert ticket["candidates"][0]["kind"] == "mark"
        assert agent.find("nowhere")["outcome"] == "unavailable"

    def test_place_hint_fills_lazily_once_the_index_exists(
            self, shelf):
        agent = _agent(shelf)
        agent.mark_store().save("hostel", A_LL[0], A_LL[1])
        # no index yet: capture/read never builds one for a hint
        assert agent.mark_for("hostel").place_hint == ""
        agent.find("Alpha Town")  # the find path builds the index
        filled = agent.mark_for("hostel")
        assert filled.place_hint == "Alpha Town"
        # and the fill persisted in the store
        assert agent.mark_store().get("hostel").place_hint \
            == "Alpha Town"

    def test_marks_ticket_rows(self, shelf):
        agent = _agent(shelf)
        agent.mark_store().save("hostel", A_LL[0], A_LL[1],
                                platform="telegram")
        ticket = agent.marks()
        assert ticket["operation"] == "marks"
        assert ticket["outcome"] == "ok"
        row = ticket["marks"][0]
        assert row["label"] == "hostel"
        assert row["platform"] == "telegram"
        assert row["method"] == METHOD_MESSAGE

    def test_view_mark_renders_the_marker(self, shelf):
        agent = _agent(shelf)
        agent.mark_store().save("hostel", A_LL[0], A_LL[1])
        ticket = agent.view_mark("hostel")
        assert ticket["outcome"] == "ok"
        assert ticket["mark"]["label"] == "hostel"
        assert ticket["receipt_id"]  # the render receipt
        data = Path(ticket["image_path"]).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert _gold_pixels(data) > 50  # the marker disc is painted
        missing = agent.view_mark("nope")
        assert missing["outcome"] == "unavailable"
        assert "no mark named" in missing["detail"]

    def test_route_accepts_a_mark_at_either_end(self, shelf):
        agent = _agent(shelf)
        agent.mark_store().save("base", A_LL[0], A_LL[1])
        out = agent.route("base", "Delta Village")
        assert out["routed"] is True
        assert out["from_label"] == "base"
        back = agent.route("Alpha Town", "base")
        assert back["routed"] is True
        assert back["to_label"] == "base"


# ── the map tool ─────────────────────────────────────────────


def _tool_registry(shelf):
    from nomorals.tools import map as map_tools
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(shelf.ctx)
    map_tools.register(registry)
    return registry


class TestMapToolMarks:
    def test_find_and_marks_and_view_inherit_the_agent(self, shelf):
        _agent(shelf).mark_store().save("hostel", A_LL[0], A_LL[1])
        registry = _tool_registry(shelf)
        found = registry.call("map", operation="find",
                              query="hostel")
        assert found.ok
        assert found.value["candidates"][0]["kind"] == "mark"
        listing = registry.call("map", operation="marks")
        assert listing.ok
        assert listing.value["marks"][0]["label"] == "hostel"
        view = registry.call("map", operation="view",
                             query="hostel")
        assert view.ok and view.value["outcome"] == "ok"
        assert view.value["mark"]["label"] == "hostel"
        route = registry.call("map", operation="route",
                              start="hostel", end="Delta Village")
        assert route.ok and route.value["routed"] is True


# ── the /map mark verbs + the pending hook, on the runtime
# harness (the M6/M7 shape: PartnerRuntime.__new__ + owner
# chats + a recording gateway) ────────────────────────────────


class _Adapter:
    """The M6 fake adapter: texts + location pins recorded."""

    def __init__(self):
        self.texts: list[tuple[str, str]] = []
        self.pins: list[tuple[str, float, float]] = []

    def send(self, chat, text, **kw):
        self.texts.append((chat.key, text))
        return SimpleNamespace(ok=True)

    def send_location(self, chat, latitude, longitude, **kw):
        self.pins.append((chat.key, latitude, longitude))
        return SimpleNamespace(ok=True)


class _Gateway:
    def __init__(self, adapter):
        self.adapters = {"telegram": adapter, "whatsapp": adapter}
        self.files = []

    def send_file(self, platform, chat, path, caption="",
                  kind="image", name=""):
        data = Path(path).read_bytes()
        self.files.append({
            "platform": platform, "path": path, "caption": caption,
            "kind": kind, "name": name, "bytes": len(data),
            "data": data})
        return SimpleNamespace(ok=True)


def _runtime(shelf):
    from nomorals.agents.partner.runtime import PartnerRuntime

    adapter = _Adapter()
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = shelf.ctx
    rt._owner_chats = {"telegram:42"}
    rt._owner_sender_ids = set()
    rt.gateway = _Gateway(adapter)
    rt.adapter = adapter
    return rt


def _ref(key="telegram:42", *, meta=None, sender_id=""):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key), text="",
        meta={} if meta is None else meta, sender_id=sender_id)


def _loc_msg(lat, lon, *, chat_id="42", owner=True, sender_id=""):
    from nomorals.social.chat.base import ChatMessage, ChatRef

    meta = {"location": {"lat": lat, "lon": lon, "live": False}}
    if owner:
        meta["is_owner"] = True
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id=chat_id),
        incoming=True, meta=meta, sender_id=sender_id,
        ts=time.time())


def _store(shelf):
    return _agent(shelf).mark_store()


class TestMarkVerbs:
    def test_immediate_capture_from_message_meta(self, shelf):
        rt = _runtime(shelf)
        msg = _ref(meta={"location": {
            "lat": D_LL[0], "lon": D_LL[1], "accuracy": 9.0}})
        out = rt._control_map("mark hostel", "telegram:42",
                              message=msg)
        assert "hostel" in out and "marked" in out
        # no gazetteer index exists yet: honest coordinates
        assert f"{D_LL[0]:.4f}" in out
        mark = _store(shelf).get("hostel")
        assert mark is not None
        assert mark.method == METHOD_MESSAGE
        assert mark.platform == "telegram"
        assert mark.accuracy == 9.0
        assert (mark.lat, mark.lon) == D_LL

    def test_immediate_capture_names_the_place_hint(self, shelf):
        rt = _runtime(shelf)
        # a find first: the shipped path builds the local index
        rt._control_map("find Alpha Town", "telegram:42",
                        message=_ref())
        msg = _ref(meta={"location": {"lat": A_LL[0],
                                      "lon": A_LL[1]}})
        out = rt._control_map("mark hostel", "telegram:42",
                              message=msg)
        assert "near Alpha Town" in out

    def test_label_move_confirmation_says_so(self, shelf):
        rt = _runtime(shelf)
        rt._control_map("mark hostel", "telegram:42", message=_ref(
            meta={"location": {"lat": A_LL[0], "lon": A_LL[1]}}))
        out = rt._control_map(
            "mark hostel", "telegram:42", message=_ref(
                meta={"location": {"lat": D_LL[0],
                                   "lon": D_LL[1]}}))
        assert "moved your 'hostel' mark" in out
        marks = _store(shelf).list()
        assert len(marks) == 1
        assert (marks[0].lat, marks[0].lon) == D_LL

    def test_pending_arm_reply_and_marks_listing(self, shelf):
        rt = _runtime(shelf)
        out = rt._control_map("mark hostel", "telegram:42",
                              message=_ref())
        assert "drop your location pin" in out
        assert "10 minutes" in out
        listed = rt._control_map("marks", "telegram:42",
                                 message=_ref())
        assert "waiting for your location pin" in listed
        assert "hostel" in listed
        # a pending is not a mark yet
        assert _store(shelf).list() == ()
        missing = rt._control_map("unmark hostel", "telegram:42",
                                  message=_ref())
        assert "no mark named" in missing

    def test_marks_list_and_unmark(self, shelf):
        rt = _runtime(shelf)
        rt._control_map("mark hostel", "telegram:42", message=_ref(
            meta={"location": {"lat": A_LL[0], "lon": A_LL[1]}}))
        listed = rt._control_map("marks", "telegram:42",
                                 message=_ref())
        assert "your marks (1)" in listed
        assert "hostel" in listed and "via telegram" in listed
        out = rt._control_map("unmark hostel", "telegram:42",
                              message=_ref())
        assert "removed" in out
        listed = rt._control_map("marks", "telegram:42",
                                 message=_ref())
        assert "no marks yet" in listed

    def test_view_by_mark_label_sends_the_photo(self, shelf):
        rt = _runtime(shelf)
        rt._control_map("mark hostel", "telegram:42", message=_ref(
            meta={"location": {"lat": A_LL[0], "lon": A_LL[1]}}))
        out = rt._control_map("view hostel", "telegram:42",
                              message=_ref())
        assert out == ""  # a landed photo returns no narration
        sent = rt.gateway.files[-1]
        assert "hostel" in sent["caption"]
        assert "© Network Fixture contributors" in sent["caption"]
        assert _gold_pixels(sent["data"]) > 50
        bad = rt._control_map("view nowhere-place", "telegram:42",
                              message=_ref())
        assert "couldn't parse coordinates" in bad

    def test_find_surface_puts_the_mark_first(self, shelf):
        rt = _runtime(shelf)
        # a mark whose label collides with an indexed place
        _store(shelf).save("Alpha Town", D_LL[0], D_LL[1])
        out = rt._control_map("find Alpha Town", "telegram:42",
                              message=_ref())
        assert "your mark 'Alpha Town'" in out
        assert "place(s) matching" in out  # gazetteer beneath it
        assert rt.gateway.files  # the mark's photo was sent
        assert rt.gateway.files[-1]["name"].startswith("mark-")

    def test_route_surface_accepts_a_mark(self, shelf):
        rt = _runtime(shelf)
        _store(shelf).save("base", A_LL[0], A_LL[1])
        out = rt._control_map("route base to Delta Village",
                              "telegram:42", message=_ref())
        assert "🧭 base → Delta Village" in out

    def test_stranger_refused_and_usage(self, shelf):
        rt = _runtime(shelf)
        out = rt._control_map("mark hostel",
                              message=_ref("telegram:999"))
        assert "owner-only" in out
        assert _store(shelf).list() == ()
        assert _store(shelf).pendings() == ()
        usage = rt._control_map("mark", "telegram:42",
                                message=_ref())
        assert "usage: /map mark <label>" in usage
        full = rt._control_map("", "telegram:42", message=_ref())
        assert "/map mark <label>" in full
        assert "/map marks" in full


class TestPendingHook:
    def test_pending_completes_through_the_hook(self, shelf):
        rt = _runtime(shelf)
        rt._control_map("mark hostel", "telegram:42",
                        message=_ref())
        rt._maybe_pending_mark_update(_loc_msg(A_LL[0], A_LL[1]))
        mark = _store(shelf).get("hostel")
        assert mark is not None
        assert mark.method == METHOD_PENDING
        assert (mark.lat, mark.lon) == A_LL
        # pending cleared; the confirmation reached the chat
        key = pending_key("telegram", "telegram:42", "")
        assert _store(shelf).pending(key) is None
        texts = [t for _k, t in rt.adapter.texts]
        assert any("hostel" in t and "marked" in t for t in texts)

    def test_location_with_no_pending_marks_nothing(self, shelf):
        rt = _runtime(shelf)
        rt._maybe_pending_mark_update(_loc_msg(A_LL[0], A_LL[1]))
        assert _store(shelf).list() == ()
        assert rt.adapter.texts == []

    def test_stranger_location_never_completes(self, shelf):
        rt = _runtime(shelf)
        rt._control_map("mark hostel", "telegram:42",
                        message=_ref())
        # a stranger's pin in another chat: not the operator
        rt._maybe_pending_mark_update(
            _loc_msg(D_LL[0], D_LL[1], chat_id="999", owner=False))
        assert _store(shelf).list() == ()
        key = pending_key("telegram", "telegram:42", "")
        assert _store(shelf).pending(key) is not None  # unconsumed

    def test_sender_mismatch_never_completes(self, shelf):
        rt = _runtime(shelf)
        # armed by the owner's telegram id; a pin stamped owner
        # but carrying a DIFFERENT sender id must not complete it
        rt._control_map("mark hostel", "telegram:42",
                        message=_ref(sender_id="4242"))
        rt._maybe_pending_mark_update(
            _loc_msg(D_LL[0], D_LL[1], sender_id="777"))
        assert _store(shelf).list() == ()
        rt._maybe_pending_mark_update(
            _loc_msg(D_LL[0], D_LL[1], sender_id="4242"))
        mark = _store(shelf).get("hostel")
        assert mark is not None
        assert (mark.lat, mark.lon) == D_LL

    def test_expired_pending_completes_nothing(self, shelf):
        rt = _runtime(shelf)
        key = pending_key("telegram", "telegram:42", "")
        _store(shelf).arm(key, "old spot", platform="telegram",
                          chat_key="telegram:42",
                          now=time.time() - PENDING_TTL_S - 60)
        rt._maybe_pending_mark_update(_loc_msg(A_LL[0], A_LL[1]))
        assert _store(shelf).list() == ()
        assert _store(shelf).pending(key) is None  # dropped
        assert rt.adapter.texts == []

    def test_whatsapp_pending_survives_uncompleted(self, shelf):
        # WhatsApp survey truth: the bridge parses NO inbound
        # location today, so no WA message ever carries
        # meta["location"] — a WA pending can arm and wait, and
        # must expire honestly rather than complete falsely.
        rt = _runtime(shelf)
        wa_key = "whatsapp:owner@s.whatsapp.net"
        out = rt._control_map(
            "mark hostel", wa_key,
            message=_ref(wa_key, meta={"self_chat": True}))
        assert "drop your location pin" in out
        pendings = _store(shelf).pendings()
        assert len(pendings) == 1
        assert pendings[0].platform == "whatsapp"
        # an ordinary WA message (no location meta — the only
        # kind the WA inbound path produces today) completes
        # nothing and consumes nothing
        rt._maybe_pending_mark_update(_ref(
            wa_key, meta={"self_chat": True}))
        assert _store(shelf).list() == ()
        assert len(_store(shelf).pendings()) == 1


class TestBrokerCoexistence:
    def test_one_share_completes_the_mark_and_still_forwards(
            self, shelf, monkeypatch):
        monkeypatch.setenv("NM_MAPS_LIVE_SHARE", "1")
        rt = _runtime(shelf)
        started = rt._control_map("share telegram:55 30 exact",
                                  "telegram:42", message=_ref())
        assert "live share started" in started
        rt._control_map("mark hostel", "telegram:42",
                        message=_ref())
        msg = _loc_msg(7.5989, 5.1057)
        # the pipeline's order: pending-mark hook, then broker
        rt._maybe_pending_mark_update(msg)
        rt._maybe_live_share_update(msg)
        mark = _store(shelf).get("hostel")
        assert mark is not None
        assert (mark.lat, mark.lon) == (7.5989, 5.1057)
        # M6 unchanged: the same share still reached the session
        assert rt.adapter.pins == [("telegram:55", 7.5989, 5.1057)]
        broker = rt._live_broker()
        decisions = broker.recent_decisions(subject="owner")
        assert decisions[0]["decision"] == "forwarded"
