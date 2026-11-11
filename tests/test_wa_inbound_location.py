"""WhatsApp inbound location shares — the Atlas M6/M9
remainder, closed.

Until this pass a WhatsApp location share died twice: the
Go bridge's onMessage never read a LocationMessage /
LiveLocationMessage off an inbound event (the wire event
crossed as an empty text message), and this adapter never
stamped ``meta["location"]`` — the one key the partner
runtime's inbound-location seam reads.  So a WhatsApp
pending /map mark could arm and wait but never complete,
and the LiveLocationBroker never saw a WhatsApp fix.

Pins (all offline; events are fed to the REAL adapter's
_dispatch, and runtime tests run the REAL hooks in the
pipeline's order — pending-mark completion BEFORE broker
evaluation):

* the adapter stamps the seam shape {lat, lon, live,
  accuracy?} from a bridge location event — a contact's
  pin, a live share, the owner's own self-chat pin — and
  stamps NOTHING for a location-less or malformed event
  (non-location behavior unchanged).
* Flow B on WhatsApp: a pin dropped in the owner's
  self-chat (where his messages are owner turns — the same
  discipline as the Telegram userbot's Saved Messages)
  completes an armed pending mark with its label; with no
  pending armed it marks nothing; a sender-mismatched pin
  never completes it and never consumes it.
* the broker on WhatsApp: a consented live share forwards
  the fix COARSENED to the session precision (the record
  holds the coarsened values too); with no consent/session
  the evaluation records a withheld decision carrying NO
  coordinates and forwards nothing.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.context import build_context
from nomorals.core.config import load_settings
from nomorals.maps.marks import METHOD_PENDING, pending_key
from nomorals.social.chat.whatsapp import WhatsAppAdapter

OWNER = "2348012345678@s.whatsapp.net"
RECIPIENT = "2349000000000@s.whatsapp.net"
GROUP = "120363012345678901@g.us"
CONTACT = "2348077777777@s.whatsapp.net"
ILAWE = (7.5989, 5.1057)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _adapter() -> WhatsAppAdapter:
    adapter = WhatsAppAdapter()
    adapter._dispatch(
        {"type": "status", "state": "open", "user": OWNER},
        lambda m: None)
    return adapter


def _message_event(chat_id, sender, *, location=None, text="",
                   kind="dm", mid="m1"):
    obj = {
        "type": "message", "id": mid,
        "chat": {"id": chat_id, "kind": kind, "title": ""},
        "from": {"id": sender, "name": "Sender"},
        "text": text, "media": [], "reply_to": "",
        "ts": _now_ms(),
    }
    if location is not None:
        obj["location"] = location
    return obj


def _own_event(chat_id, *, location=None, text="", kind="dm",
               mid="m9"):
    obj = {
        "type": "own_message", "id": mid,
        "chat": {"id": chat_id, "kind": kind, "title": ""},
        "from": {"id": OWNER, "name": "Owner"},
        "text": text, "reply_to": "", "ts": _now_ms(),
    }
    if location is not None:
        obj["location"] = location
    return obj


def _deliver(adapter, obj):
    msgs: list = []
    adapter._dispatch(obj, msgs.append)
    assert len(msgs) == 1
    return msgs[0]


# ── the adapter seam ─────────────────────────────────────


class TestAdapterStampsLocation:
    def test_contact_pin_stamps_the_seam_shape(self):
        msg = _deliver(_adapter(), _message_event(
            CONTACT, CONTACT,
            location={"lat": ILAWE[0], "lon": ILAWE[1],
                      "live": False, "accuracy": 12,
                      "name": "Mama's shop",
                      "address": "12 Allen Avenue"}))
        assert msg.meta["location"] == {
            "lat": ILAWE[0], "lon": ILAWE[1], "live": False,
            "accuracy": 12.0}

    def test_live_share_stamps_live_true(self):
        msg = _deliver(_adapter(), _message_event(
            CONTACT, CONTACT,
            location={"lat": ILAWE[0], "lon": ILAWE[1],
                      "live": True}))
        assert msg.meta["location"] == {
            "lat": ILAWE[0], "lon": ILAWE[1], "live": True}

    def test_no_accuracy_stays_absent(self):
        msg = _deliver(_adapter(), _message_event(
            CONTACT, CONTACT,
            location={"lat": ILAWE[0], "lon": ILAWE[1],
                      "live": False}))
        assert "accuracy" not in msg.meta["location"]

    def test_locationless_message_stamps_nothing(self):
        msg = _deliver(_adapter(), _message_event(
            CONTACT, CONTACT, text="hey devon"))
        assert "location" not in msg.meta
        assert msg.text == "hey devon"

    @pytest.mark.parametrize("bad", [
        {"lat": "north", "lon": 5.1057, "live": False},
        {"lat": 91.0, "lon": 5.1057, "live": False},
        {"lat": 7.5989, "live": False},
        {"live": False},
    ])
    def test_malformed_location_stamps_nothing(self, bad):
        msg = _deliver(_adapter(), _message_event(
            CONTACT, CONTACT, location=bad))
        assert "location" not in msg.meta

    def test_own_self_chat_pin_is_an_owner_turn_with_location(self):
        msg = _deliver(_adapter(), _own_event(
            OWNER, location={"lat": ILAWE[0], "lon": ILAWE[1],
                             "live": False, "accuracy": 9}))
        assert msg.incoming
        assert msg.meta.get("self_chat") is True
        assert msg.sender_id == OWNER
        assert msg.meta["location"]["lat"] == ILAWE[0]
        assert msg.meta["location"]["accuracy"] == 9.0

    def test_own_group_pin_stays_persist_only_but_stamped(self):
        # The Telegram userbot discipline: his own messages
        # outside the self-chat are persist-only echoes.  The
        # stamp records the share; the runtime seam does not
        # act on an incoming=False echo.
        msg = _deliver(_adapter(), _own_event(
            GROUP, kind="group",
            location={"lat": ILAWE[0], "lon": ILAWE[1],
                      "live": False}))
        assert not msg.incoming
        assert msg.meta.get("own_message") is True
        assert msg.meta["location"]["lon"] == ILAWE[1]


# ── the runtime seam (M9 shelf idiom) ────────────────────


class _Adapter:
    """The M6/M9 fake adapter: texts + location pins."""

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


@pytest.fixture()
def shelf(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    packs = tmp_path / "packs"
    packs.mkdir()
    monkeypatch.setenv("NM_MAPS_ENABLED", "1")
    monkeypatch.setenv("NM_MAPS_PACKS_DIR", str(packs))
    settings = load_settings(overrides={
        "home": str(home), "partner.platforms": "local",
        "chat.local_enabled": "true"})
    ctx = build_context(settings, with_executor=False,
                        with_tools=True, with_router=False)
    return SimpleNamespace(ctx=ctx, home=home, packs=packs)


def _runtime(shelf):
    from nomorals.agents.partner.runtime import PartnerRuntime

    adapter = _Adapter()
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = shelf.ctx
    rt._owner_chats = set()
    rt._owner_sender_ids = set()
    rt.gateway = _Gateway(adapter)
    rt.adapter = adapter
    return rt


def _store(shelf):
    from nomorals.agents.map import MapAgent

    return MapAgent(shelf.ctx).mark_store()


def _self_chat_turn(text=""):
    return _deliver(_adapter(), _own_event(OWNER, text=text))


def _self_chat_pin():
    return _deliver(_adapter(), _own_event(
        OWNER, location={"lat": ILAWE[0], "lon": ILAWE[1],
                         "live": False, "accuracy": 12}))


class TestWhatsAppFlowB:
    def test_self_chat_pin_completes_armed_pending(self, shelf):
        rt = _runtime(shelf)
        arm = _self_chat_turn("/map mark hostel")
        out = rt._control_map("mark hostel", arm.chat.key,
                              message=arm)
        assert "drop your location pin" in out
        pin = _self_chat_pin()
        # the pipeline's gate + order: incoming, meta location,
        # pending-mark hook BEFORE the broker hook
        assert pin.incoming and pin.meta.get("location")
        rt._maybe_pending_mark_update(pin)
        mark = _store(shelf).get("hostel")
        assert mark is not None
        assert mark.method == METHOD_PENDING
        assert mark.platform == "whatsapp"
        assert (mark.lat, mark.lon) == ILAWE
        assert mark.accuracy == 12.0
        key = pending_key("whatsapp", arm.chat.key, OWNER)
        assert _store(shelf).pending(key) is None
        texts = [t for _k, t in rt.adapter.texts]
        assert any("hostel" in t and "marked" in t for t in texts)

    def test_pin_with_no_pending_marks_nothing(self, shelf):
        rt = _runtime(shelf)
        rt._maybe_pending_mark_update(_self_chat_pin())
        assert _store(shelf).list() == ()
        assert rt.adapter.texts == []

    def test_sender_mismatch_never_completes(self, shelf):
        rt = _runtime(shelf)
        # armed by the owner in a group (his command there is
        # an owner turn; the pending keys on HIS sender id)
        arm = _deliver(_adapter(), _own_event(
            GROUP, kind="group", text="/map mark hostel"))
        assert arm.incoming
        rt._control_map("mark hostel", arm.chat.key, message=arm)
        # a contact's pin in the same group: the gateway may
        # judge the chat an owner surface (is_owner), but the
        # pending is keyed to the owner's sender id — a
        # different sender completes nothing, consumes nothing
        pin = _deliver(_adapter(), _message_event(
            GROUP, CONTACT, kind="group",
            location={"lat": ILAWE[0], "lon": ILAWE[1],
                      "live": False}))
        pin.meta["is_owner"] = True
        rt._maybe_pending_mark_update(pin)
        assert _store(shelf).list() == ()
        key = pending_key("whatsapp", arm.chat.key, OWNER)
        assert _store(shelf).pending(key) is not None


class TestWhatsAppBroker:
    def test_consented_share_forwards_coarsened_fix(
            self, shelf, monkeypatch):
        monkeypatch.setenv("NM_MAPS_LIVE_SHARE", "1")
        rt = _runtime(shelf)
        arm = _self_chat_turn()
        started = rt._control_map(
            f"share whatsapp:{RECIPIENT} 30 neighborhood",
            arm.chat.key, message=arm)
        assert "live share started" in started
        rt._maybe_live_share_update(_self_chat_pin())
        # neighborhood precision: the recipient AND the
        # decision record hold the coarsened fix, never the
        # exact supplied one
        assert rt.adapter.pins == [
            (f"whatsapp:{RECIPIENT}", 7.6, 5.11)]
        decisions = rt._live_broker().recent_decisions(
            subject="owner")
        assert decisions[0]["decision"] == "forwarded"
        assert decisions[0]["lat"] == 7.6
        assert decisions[0]["lon"] == 5.11

    def test_unconsented_share_records_no_coordinates(
            self, shelf, monkeypatch):
        monkeypatch.setenv("NM_MAPS_LIVE_SHARE", "1")
        rt = _runtime(shelf)
        rt._maybe_live_share_update(_self_chat_pin())
        assert rt.adapter.pins == []
        decisions = rt._live_broker().recent_decisions(
            subject="owner")
        assert decisions[0]["decision"] == "withheld"
        assert decisions[0]["reason"] == "no-session"
        assert decisions[0]["lat"] is None
        assert decisions[0]["lon"] is None
