"""PARITY #9 — LOCATION + CONTACT SENDING (platform parity
census bundle #9, 2026-11-07).

The census graded four rows ABSENT: WhatsApp send location
and send contact card (no send-side hit in the adapter or
either bridge), Telegram bot send location/venue/contact
(no sendLocation/sendVenue/sendContact anywhere) and
Telegram personal send location/contact (no InputMedia
GeoPoint/InputMediaContact usage).  This suite drives the
real composed chain where one exists — owner command →
handle_control → runtime_chatops → the real WhatsAppAdapter
→ a recording fake bridge speaking the bridges' wire
protocol — and pins each other surface at its own seam:

  * /sendloc: bare coordinates fire send_location with the
    exact chat + degrees; a name/address fires the venue
    form (same bridge verb, label fields set); a landed pin
    journals location_sent; bad shapes refuse BEFORE
    traffic; the owner's standing stop gates it.
  * /sendcontact: the card fires send_contact with a real
    vCard 3.0 body built from the number; a landed card
    journals contact_shared; undialable numbers refuse
    before traffic.
  * Telegram bot: sendLocation / sendVenue / sendContact
    fire with the Bot API's own parameter shapes (first/last
    name split, vcard only when supplied).
  * Telegram personal: InputMediaGeoPoint / InputMediaVenue
    / InputMediaContact ride SendMediaRequest with the
    exact fields; a disconnected adapter answers honestly.
  * The bot-surface chatop branch posts through the bot
    adapter in the command's own chat (no personal account,
    no journal).
  * Bridge internals are pinned where they live: the Go
    verbs' dispatch + whatsmeow LocationMessage /
    ContactMessage shapes (bridge/gobridge/pin_test.go
    covers the builders), the Node verbs against Baileys'
    sendMessage { location } / { contacts } payloads, and
    the rebuilt Go binaries carry the new refusal strings.
"""
from __future__ import annotations

import asyncio
import json
import socketserver
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

PORT = 19449
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
ROOT = Path(__file__).resolve().parents[1]


# ── recording fake bridge (the bridges' wire protocol) ─────

class _BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        self.wfile.write(json.dumps({
            "type": "status", "state": "open", "user": OWNER,
            "user_name": "Owner"}).encode() + b"\n")
        self.wfile.flush()
        with srv.lock:
            srv.clients.append(self.wfile)
        try:
            for raw in self.rfile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except ValueError:
                    continue
                with srv.lock:
                    srv.commands.append(cmd)
                cid = cmd.get("id")
                if not cid:
                    continue
                reply = {"id": cid, "ok": True}
                verb = cmd.get("cmd")
                if verb == "send_location":
                    if srv.location_error:
                        reply = {"id": cid, "ok": False,
                                 "error": srv.location_error}
                    else:
                        reply["message_id"] = "LOC-1"
                elif verb == "send_contact":
                    if srv.contact_error:
                        reply = {"id": cid, "ok": False,
                                 "error": srv.contact_error}
                    else:
                        reply["message_id"] = "CARD-1"
                self.wfile.write(
                    json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port=PORT):
        self.commands: list[dict] = []
        self.clients: list = []
        self.lock = threading.Lock()
        self.location_error = ""
        self.contact_error = ""
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def close(self):
        self.shutdown()
        self.server_close()


def wait_for(pred, timeout=25.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parity9")
    bridge = FakeBridge()
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": OWNER,
        "chat.local_enabled": "true",
        "chat.whatsapp_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    rec = sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                      / f"wa-{OWNER}"),
        port=PORT, status="linked", created_at=1.0))

    def beat():
        runtime.on_message(ChatMessage(
            chat=ChatRef(platform="local", chat_id="console",
                         kind=ChatKind.DM, title="console",
                         peer="owner"),
            incoming=True, text="hello", sender="owner",
            sender_id="owner",
            message_id=f"beat-{time.time()}", ts=time.time()))

    beat()
    assert wait_for(
        lambda: runtime.gateway.adapters.get("whatsapp")
        is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {
        "runtime": runtime, "bridge": bridge, "adapter": adapter,
        "sup": sup, "rec": rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "tmp": tmp, "beat": beat,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, title="", sender_id=None, mid=""):
    if sender_id is None:
        sender_id = OWNER
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=mid or f"op-{time.time()}",
        ts=time.time(), meta={"owner_command": True})
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, account_id):
    return rig["vault"].journal(account_id)


# ── bridge source pins (both engines + the binaries) ───────

def test_go_bridge_carries_the_pin_verbs():
    dispatch = (ROOT / "bridge" / "gobridge"
                / "messages.go").read_text()
    for verb in ("send_location", "send_contact"):
        assert f'case "{verb}"' in dispatch, verb
    src = (ROOT / "bridge" / "gobridge"
           / "profile.go").read_text()
    # the whatsmeow message types that actually exist
    assert "waE2E.LocationMessage" in src
    assert "waE2E.ContactMessage" in src
    assert "DegreesLatitude" in src
    assert "DegreesLongitude" in src
    assert "DisplayName" in src
    assert "Vcard" in src
    # shape refusals stated in the bridge's own words
    assert "a location pin needs latitude between -90 and 90" in src
    assert "a contact card needs a vcard" in src


def test_baileys_bridge_carries_the_pin_verbs():
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs").read_text()
    assert "case 'send_location'" in src
    assert "case 'send_contact'" in src
    # Baileys' own sendMessage payload shapes
    assert "degreesLatitude" in src
    assert "degreesLongitude" in src
    assert "{ location }" in src
    assert "contacts: { displayName: name" in src
    assert "contacts: [{ vcard }]" in src


def test_rebuilt_binaries_carry_the_verbs():
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"a location pin needs latitude between -90 and 90" in data, name
        assert b"a contact card needs a vcard" in data, name


def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )

    for kind in ("sendloc", "sendcontact"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
    # owner-only account ops — never public commands
    from nomorals.social.chat.control import PUBLIC_COMMANDS
    assert "sendloc" not in PUBLIC_COMMANDS
    assert "sendcontact" not in PUBLIC_COMMANDS


# ── parsers ────────────────────────────────────────────────

def test_parse_sendloc_args():
    from nomorals.social.chat.chatops import parse_sendloc_args

    got = parse_sendloc_args("6.5244, 3.3792")
    assert got["ok"] and got["latitude"] == 6.5244
    assert got["longitude"] == 3.3792 and got["name"] == ""
    got = parse_sendloc_args(
        "6.5244 3.3792 | Lekki Gate | Admiralty Way")
    assert got["ok"] and got["name"] == "Lekki Gate"
    assert got["address"] == "Admiralty Way"
    got = parse_sendloc_args("-33.86, 151.21")
    assert got["ok"] and got["latitude"] == -33.86
    for bad in ("", "6.5244", "abc, def", "91, 0", "0, 181",
                "6.5, 3.3, 9"):
        got = parse_sendloc_args(bad)
        assert not got["ok"], bad
        assert got["error"]


def test_parse_sendcontact_args():
    from nomorals.social.chat.chatops import parse_sendcontact_args

    got = parse_sendcontact_args("Ada Lovelace | +2348000000000")
    assert got["ok"] and got["name"] == "Ada Lovelace"
    assert got["phone"] == "+2348000000000"
    for bad in ("", "Ada", "| +2348000000000", "Ada |",
                "Ada | 123"):
        got = parse_sendcontact_args(bad)
        assert not got["ok"], bad
        assert got["error"]


# ── adapter-level pins: WhatsApp payloads + refusals ───────

@pytest.fixture()
def wa_adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19998)
    calls: list[tuple] = []

    def fake_admin(cmd, payload, *, timeout=30.0):
        calls.append((cmd, payload))
        return {"ok": True, "message_id": "WAMID-9"}

    monkeypatch.setattr(ad, "_admin_cmd", fake_admin)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _wa_chat(chat_id=GROUP):
    return ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=ChatKind.GROUP, peer=chat_id)


def test_wa_send_location_payload(wa_adapter):
    res = wa_adapter.send_location(_wa_chat(), 6.5244, 3.3792)
    assert res.ok and res.message_id == "WAMID-9"
    assert wa_adapter.calls == [("send_location", {
        "chat": GROUP, "latitude": 6.5244, "longitude": 3.3792,
        "name": "", "address": ""})]


def test_wa_send_venue_payload(wa_adapter):
    res = wa_adapter.send_venue(
        _wa_chat(), 6.5244, 3.3792, "Lekki Gate",
        "Admiralty Way")
    assert res.ok
    assert wa_adapter.calls == [("send_location", {
        "chat": GROUP, "latitude": 6.5244, "longitude": 3.3792,
        "name": "Lekki Gate", "address": "Admiralty Way"})]
    res = wa_adapter.send_venue(_wa_chat(), 6.5, 3.3, "  ")
    assert not res.ok and "needs the place's name" in res.error
    assert len(wa_adapter.calls) == 1  # refusal before traffic


def test_wa_send_contact_payload_and_vcard(wa_adapter):
    res = wa_adapter.send_contact(
        _wa_chat(), "+2348000000000", "Ada Lovelace")
    assert res.ok and res.message_id == "WAMID-9"
    cmd, payload = wa_adapter.calls[-1]
    assert cmd == "send_contact"
    assert payload["chat"] == GROUP
    assert payload["name"] == "Ada Lovelace"
    assert payload["phone"] == "+2348000000000"
    assert payload["vcard"].startswith("BEGIN:VCARD")
    assert "FN:Ada Lovelace" in payload["vcard"]
    assert "+2348000000000" in payload["vcard"]
    # a caller-supplied richer card rides verbatim
    res = wa_adapter.send_contact(
        _wa_chat(), "+2348000000000", "Ada Lovelace",
        vcard="BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Ada\r\nEND:VCARD")
    assert res.ok
    assert wa_adapter.calls[-1][1]["vcard"].startswith(
        "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Ada")


def test_wa_refusals_before_traffic(wa_adapter):
    import math

    for lat, lon in ((91.0, 0.0), (0.0, 181.0),
                     (math.nan, 0.0), (math.inf, 0.0)):
        res = wa_adapter.send_location(_wa_chat(), lat, lon)
        assert not res.ok, (lat, lon)
    res = wa_adapter.send_contact(_wa_chat(), "123", "Ada")
    assert not res.ok and "7–15 digits" in res.error
    res = wa_adapter.send_contact(_wa_chat(), "+2348000000000",
                                  "  ")
    assert not res.ok and "needs the person's name" in res.error
    assert wa_adapter.calls == []  # nothing reached the bridge


def test_wa_unproven_and_refused_sends(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19997)
    chat = _wa_chat()
    monkeypatch.setattr(
        ad, "_admin_cmd",
        lambda cmd, payload, *, timeout=30.0: {"ok": True})
    res = ad.send_location(chat, 6.5, 3.3)
    assert not res.ok and "unproven" in res.error
    res = ad.send_contact(chat, "+2348000000000", "Ada")
    assert not res.ok and "unproven" in res.error
    monkeypatch.setattr(
        ad, "_admin_cmd",
        lambda cmd, payload, *, timeout=30.0: {
            "ok": False, "error": "bridge is not logged in"})
    res = ad.send_location(chat, 6.5, 3.3)
    assert not res.ok and res.error == "bridge is not logged in"


def test_base_adapter_reports_the_truth():
    from nomorals.social.chat.base import ChatAdapter

    class _Bare(ChatAdapter):
        name = "bare"

        def run(self, handler):  # pragma: no cover - unused
            raise NotImplementedError

        def send(self, chat, text, **kwargs):  # pragma: no cover
            from nomorals.social.chat.base import SendResult

            return SendResult(ok=False, platform=self.name,
                              error="unused in this test")

    ad = _Bare()
    chat = _wa_chat()
    for res in (ad.send_location(chat, 6.5, 3.3),
                ad.send_venue(chat, 6.5, 3.3, "Somewhere"),
                ad.send_contact(chat, "+2348000000000", "Ada")):
        assert not res.ok and "not supported" in res.error


# ── adapter-level pins: Telegram bot ───────────────────────

@pytest.fixture()
def bot_adapter(monkeypatch):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    ad = TelegramBotAdapter(
        token="123456:TEST", chat_allow="",
        media_dir="data/media/telegram-bot-test")
    calls: list[tuple] = []

    def fake_api(method, **params):
        calls.append((method, params))
        return {"message_id": 4242}

    monkeypatch.setattr(ad, "_api", fake_api)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _tg_chat(chat_id="-1001234567890"):
    return ChatRef(platform="telegram-bot", chat_id=chat_id,
                   kind=ChatKind.GROUP, peer=chat_id)


def test_bot_send_location_payload(bot_adapter):
    res = bot_adapter.send_location(_tg_chat(), 6.5244, 3.3792)
    assert res.ok and res.message_id == "4242"
    assert bot_adapter.calls == [("sendLocation", {
        "chat_id": -1001234567890, "latitude": 6.5244,
        "longitude": 3.3792})]


def test_bot_send_venue_payload(bot_adapter):
    res = bot_adapter.send_venue(
        _tg_chat(), 6.5244, 3.3792, "Lekki Gate",
        "Admiralty Way")
    assert res.ok
    assert bot_adapter.calls == [("sendVenue", {
        "chat_id": -1001234567890, "latitude": 6.5244,
        "longitude": 3.3792, "title": "Lekki Gate",
        "address": "Admiralty Way"})]
    res = bot_adapter.send_venue(_tg_chat(), 6.5, 3.3, "")
    assert not res.ok and "title" in res.error
    assert len(bot_adapter.calls) == 1


def test_bot_send_contact_payload(bot_adapter):
    res = bot_adapter.send_contact(
        _tg_chat(), "+2348000000000", "Ada Lovelace")
    assert res.ok
    assert bot_adapter.calls == [("sendContact", {
        "chat_id": -1001234567890,
        "phone_number": "+2348000000000",
        "first_name": "Ada", "last_name": "Lovelace"})]
    # one-word name: no last_name key at all; vcard only when given
    res = bot_adapter.send_contact(_tg_chat(), "+15551234567",
                                   "Madonna")
    assert res.ok
    assert "last_name" not in bot_adapter.calls[-1][1]
    assert "vcard" not in bot_adapter.calls[-1][1]
    res = bot_adapter.send_contact(
        _tg_chat(), "+15551234567", "Madonna", vcard="CARD")
    assert res.ok
    assert bot_adapter.calls[-1][1]["vcard"] == "CARD"


def test_bot_refusals_and_api_failure(bot_adapter, monkeypatch):
    res = bot_adapter.send_location(_tg_chat(), 91.0, 0.0)
    assert not res.ok and "latitude" in res.error
    res = bot_adapter.send_contact(_tg_chat(), "12", "Ada")
    assert not res.ok and "7–15 digits" in res.error
    assert bot_adapter.calls == []

    def boom(method, **params):
        raise RuntimeError("chat not found")

    monkeypatch.setattr(bot_adapter, "_api", boom)
    res = bot_adapter.send_location(_tg_chat(), 6.5, 3.3)
    assert not res.ok and res.error == "chat not found"


# ── adapter-level pins: Telegram personal (Telethon) ───────

class _FakeTLType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSendMediaRequest(_FakeTLType):
    pass


@pytest.fixture()
def tg_personal(monkeypatch, tmp_path):
    """A TelegramAdapter with a recording fake client and a
    fake telethon module tree — the TL shapes are asserted
    field by field, which is the layer a wrong field name
    would silently break live."""
    from nomorals.social.chat.telegram import TelegramAdapter

    types_mod = _types.ModuleType("telethon.tl.types")
    for cls_name in ("InputGeoPoint", "InputMediaGeoPoint",
                     "InputMediaVenue", "InputMediaContact"):
        setattr(types_mod, cls_name, type(cls_name, (_FakeTLType,), {}))
    messages_mod = _types.ModuleType(
        "telethon.tl.functions.messages")
    messages_mod.SendMediaRequest = _FakeSendMediaRequest
    for name, mod in (
            ("telethon", _types.ModuleType("telethon")),
            ("telethon.tl", _types.ModuleType("telethon.tl")),
            ("telethon.tl.types", types_mod),
            ("telethon.tl.functions",
             _types.ModuleType("telethon.tl.functions")),
            ("telethon.tl.functions.messages", messages_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    requests: list = []

    class _Msg:
        id = 777
        views = None

    class _Result:
        updates = [_types.SimpleNamespace(message=_Msg())]

    async def _client(request):
        requests.append(request)
        return _Result()

    ad._client = _client  # type: ignore[attr-defined]

    async def _resolve(chat):
        return "ENTITY-1"

    monkeypatch.setattr(ad, "_resolve", _resolve)
    monkeypatch.setattr(
        ad, "_run_on_loop",
        lambda coro, timeout=60.0: asyncio.run(coro))
    ad.requests = requests  # type: ignore[attr-defined]
    return ad


def _userbot_chat(chat_id="424242"):
    return ChatRef(platform="telegram", chat_id=chat_id,
                   kind=ChatKind.DM, peer=chat_id)


def test_userbot_send_location_media(tg_personal):
    res = tg_personal.send_location(_userbot_chat(), 6.5244,
                                    3.3792)
    assert res.ok and res.message_id == "777"
    (req,) = tg_personal.requests
    assert isinstance(req, _FakeSendMediaRequest)
    assert req.peer == "ENTITY-1"
    media = req.media
    assert type(media).__name__ == "InputMediaGeoPoint"
    assert media.geo_point.lat == 6.5244
    assert media.geo_point.long == 3.3792


def test_userbot_send_venue_media(tg_personal):
    res = tg_personal.send_venue(
        _userbot_chat(), 6.5244, 3.3792, "Lekki Gate",
        "Admiralty Way")
    assert res.ok
    (req,) = tg_personal.requests
    media = req.media
    assert type(media).__name__ == "InputMediaVenue"
    assert media.geo_point.lat == 6.5244
    assert media.title == "Lekki Gate"
    assert media.address == "Admiralty Way"
    res = tg_personal.send_venue(_userbot_chat(), 6.5, 3.3,
                                 "  ")
    assert not res.ok and "title" in res.error
    assert len(tg_personal.requests) == 1


def test_userbot_send_contact_media(tg_personal):
    res = tg_personal.send_contact(
        _userbot_chat(), "+2348000000000", "Ada Lovelace")
    assert res.ok and res.message_id == "777"
    (req,) = tg_personal.requests
    media = req.media
    assert type(media).__name__ == "InputMediaContact"
    assert media.phone_number == "+2348000000000"
    assert media.first_name == "Ada"
    assert media.last_name == "Lovelace"
    assert media.vcard == ""


def test_userbot_disconnected_and_shape_refusals(
        monkeypatch, tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    chat = _userbot_chat()
    res = ad.send_location(chat, 6.5, 3.3)
    assert not res.ok and res.error == "not connected"
    res = ad.send_contact(chat, "+2348000000000", "Ada")
    assert not res.ok and res.error == "not connected"
    # shape rules still answer before any client work
    ad._client = object()  # type: ignore[attr-defined]
    res = ad.send_location(chat, 91.0, 0.0)
    assert not res.ok and "latitude" in res.error
    res = ad.send_contact(chat, "12", "Ada")
    assert not res.ok and "7–15 digits" in res.error


# ── chatop: the bot-surface branch ─────────────────────────

def test_chatop_bot_surface_posts_without_personal_home():
    from nomorals.agents.partner.runtime_chatops import RuntimeChatOpsMixin
    from nomorals.social.chat.base import SendResult

    sent: list[tuple] = []

    class _BotAdapter:
        name = "telegram-bot"

        def send_location(self, chat, lat, lon):
            sent.append(("location", chat.chat_id, lat, lon))
            return SendResult(ok=True, platform=self.name,
                              message_id="55")

        def send_contact(self, chat, phone, name, **kw):
            sent.append(("contact", chat.chat_id, phone, name))
            return SendResult(ok=True, platform=self.name,
                              message_id="56")

    mixin = RuntimeChatOpsMixin()
    mixin._adapter_for = (  # type: ignore[method-assign]
        lambda platform: _BotAdapter())
    chat = ChatRef(platform="telegram-bot",
                   chat_id="-1001234567890",
                   kind=ChatKind.GROUP,
                   peer="-1001234567890")
    msg = ChatMessage(chat=chat, incoming=True, text="x",
                      sender="Owner", sender_id="1")
    out = mixin._chatops_sendloc("6.5244, 3.3792", msg)
    assert "location pin (6.5244, 3.3792) sent" in out
    assert sent == [("location", "-1001234567890", 6.5244,
                     3.3792)]
    out = mixin._chatops_sendcontact(
        "Ada Lovelace | +2348000000000", msg)
    assert "contact card shared: Ada Lovelace" in out
    assert sent[-1] == ("contact", "-1001234567890",
                        "+2348000000000", "Ada Lovelace")
    # a chat on no surface at all answers honestly
    odd = ChatRef(platform="local", chat_id="console",
                  kind=ChatKind.DM, peer="console")
    msg2 = ChatMessage(chat=odd, incoming=True, text="x",
                       sender="Owner", sender_id="1")
    out = mixin._chatops_sendloc("6.5, 3.3", msg2)
    assert "run it from your WhatsApp" in out


# ── composed chain: /sendloc + /sendcontact on WhatsApp ────

def test_sendloc_bare_pin_fires_and_journals(rig):
    out = _op(rig, "/sendloc 6.5244, 3.3792")
    assert "location pin (6.5244, 3.3792) sent" in out
    cmds = rig["bridge"].cmds("send_location")
    assert cmds and cmds[-1]["chat"] == PN_POSTER
    assert cmds[-1]["latitude"] == 6.5244
    assert cmds[-1]["longitude"] == 3.3792
    assert cmds[-1]["name"] == "" and cmds[-1]["address"] == ""
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "location_sent"]
    assert rows and rows[-1]["source_id"] == "LOC-1"
    assert rows[-1]["latitude"] == 6.5244


def test_sendloc_named_place_fires_venue_fields(rig):
    out = _op(rig, "/sendloc 6.5244, 3.3792 | Lekki Gate | "
                   "Admiralty Way", chat_id=GROUP,
              kind=ChatKind.GROUP, title="Test Group")
    assert "Lekki Gate (6.5244, 3.3792) sent" in out
    cmds = rig["bridge"].cmds("send_location")
    assert cmds[-1]["chat"] == GROUP
    assert cmds[-1]["name"] == "Lekki Gate"
    assert cmds[-1]["address"] == "Admiralty Way"
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "location_sent"
            and "Lekki Gate" in str(r.get("text") or "")]
    assert rows


def test_sendloc_bad_shape_refuses_before_traffic(rig):
    before = len(rig["bridge"].cmds("send_location"))
    out = _op(rig, "/sendloc 91, 0")
    assert "latitude" in out
    out = _op(rig, "/sendloc over there")
    assert "usage: /sendloc" in out
    assert len(rig["bridge"].cmds("send_location")) == before


def test_sendloc_bridge_refusal_is_named(rig):
    rig["bridge"].location_error = "bridge is not logged in"
    try:
        out = _op(rig, "/sendloc 6.5244, 3.3792")
    finally:
        rig["bridge"].location_error = ""
    assert "couldn't send the location" in out
    assert "bridge is not logged in" in out


def test_sendcontact_fires_and_journals(rig):
    out = _op(rig, "/sendcontact Ada Lovelace | "
                   "+2348000000000")
    assert "contact card shared: Ada Lovelace" in out
    cmds = rig["bridge"].cmds("send_contact")
    assert cmds and cmds[-1]["chat"] == PN_POSTER
    assert cmds[-1]["name"] == "Ada Lovelace"
    assert cmds[-1]["phone"] == "+2348000000000"
    assert cmds[-1]["vcard"].startswith("BEGIN:VCARD")
    assert "+2348000000000" in cmds[-1]["vcard"]
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "contact_shared"]
    assert rows and rows[-1]["source_id"] == "CARD-1"
    assert "Ada Lovelace" in str(rows[-1].get("text") or "")


def test_sendcontact_bad_shape_refuses_before_traffic(rig):
    before = len(rig["bridge"].cmds("send_contact"))
    out = _op(rig, "/sendcontact Ada | 123")
    assert "7–15 digits" in out
    out = _op(rig, "/sendcontact Ada")
    assert "usage: /sendcontact" in out
    assert len(rig["bridge"].cmds("send_contact")) == before


def test_send_family_honors_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before_loc = len(rig["bridge"].cmds("send_location"))
        before_card = len(rig["bridge"].cmds("send_contact"))
        out = _op(rig, "/sendloc 6.5244, 3.3792")
        assert "standing stop" in out
        out = _op(rig, "/sendcontact Ada Lovelace | "
                       "+2348000000000")
        assert "standing stop" in out
        assert len(rig["bridge"].cmds("send_location")) == before_loc
        assert len(rig["bridge"].cmds("send_contact")) == before_card
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)
