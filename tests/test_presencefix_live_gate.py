"""Presence/status family — the live-gate pass (2026-11-07).

The owner's verified reproduction: he turned status_viewing,
always_online and status_react ON with /pair toggle and saw NO sign
of any of them working — "It's dead I'm not guessing."  Tracing the
whole chain at base found the links individually sound (and this
file proves the sound ones end to end against real shapes), plus
three real breaks:

1.  THE FUNNEL ATE THE FEED.  Every WhatsApp status post — from
    every poster — lands in ONE gateway chat bucket
    (``whatsapp:status@broadcast``), and the gateway's conversational
    rate limiter caps a bucket at 60/hour (the production default).
    Past the cap the funnel drops status posts BEFORE the runtime
    sees them: viewing / reacting / saving silently starve on any
    account whose contacts post more than 60 statuses an hour.
    Service-feed messages are machine feed, not conversation — they
    now bypass the conversational limiter (the limiter itself is
    untouched for real conversation, pinned below).

2.  BUTTONS DIED AT THE LAST HOP.  Devon's send path carries
    buttons faithfully (pinned below: gateway → adapter → bridge
    wire), but both bridges rendered the LEGACY ButtonsMessage —
    the form WhatsApp deprecated for unofficial multi-device
    senders; current clients do not render it, so the message
    arrived with no buttons and no error anywhere.  The Go bridge
    now sends the modern InteractiveMessage (NativeFlow
    quick_reply) form, keeping the text-line degradation on send
    failure; the bridge reports the form it used
    (``buttons_form``) and the adapter journals any degraded form
    instead of letting it pass silently (pinned below).

3.  /pair SHOWED SET-STATE ONLY.  A toggle could be on, the
    session dead, and the display said nothing — "off" and
    "broken" looked identical (the owner's exact trap).  /pair now
    shows, per ON toggle, whether it is LIVE, with the reason when
    it is not (pinned below).

The intact links are pinned too, against real account shapes (LID
poster, phone-JID poster, the real Go wire protocol spoken by a
socket fake): toggle → registry → housekeeping beat → presence
verb on the wire; status post → view + react verbs on the wire +
the vault journal; /poststatus → status_send on the wire.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.social.pairing import AccountRecord, PairingRegistry

FAKE_PORT = 19377
OWNER = "2348012345678"
LID_POSTER = "1555123456789@lid"
PN_POSTER = "2347099999999@s.whatsapp.net"


# ── a socket fake speaking the real Go bridge wire protocol ────

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
                if cmd.get("cmd") == "send":
                    reply["message_id"] = "WAMID-TEST-1"
                    if cmd.get("buttons") and srv.send_form:
                        reply["buttons_form"] = srv.send_form
                if cmd.get("cmd") == "status_send":
                    reply["message_id"] = "WAMID-STATUS-1"
                self.wfile.write(json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port=FAKE_PORT, send_form=""):
        self.commands: list[dict] = []
        self.clients: list = []
        self.lock = threading.Lock()
        self.send_form = send_form
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        self._thread = threading.Thread(
            target=self.serve_forever, daemon=True)
        self._thread.start()

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands if c.get("cmd") == name]

    def emit(self, obj):
        line = (json.dumps(obj) + "\n").encode()
        with self.lock:
            clients = list(self.clients)
        for w in clients:
            try:
                w.write(line)
                w.flush()
            except OSError:
                pass

    def close(self):
        self.shutdown()
        self.server_close()


def wait_for(pred, timeout=15.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture()
def bridge():
    fake = FakeBridge()
    yield fake
    fake.close()


def _status_msg(i, *, service=True):
    meta = {"service": "status_post"} if service else {}
    return ChatMessage(
        chat=ChatRef(platform="whatsapp",
                     chat_id="status@broadcast", kind=ChatKind.DM),
        incoming=True, text=f"status {i}", sender="Ada",
        sender_id=LID_POSTER, message_id=f"ST-{i}",
        ts=time.time(), meta=meta)


# ── break 1: the funnel must not eat the service feed ──────────

def test_service_feed_is_not_conversation_rate_limited():
    received: list = []
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    gateway.start(received.append)
    try:
        for i in range(75):
            gateway._on_inbound(_status_msg(i))
        assert len(received) == 75
    finally:
        gateway.stop()


def test_conversation_rate_limit_still_holds():
    received: list = []
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    gateway.start(received.append)
    try:
        for i in range(75):
            gateway._on_inbound(_status_msg(i, service=False))
        assert len(received) == 60  # the cap itself is untouched
    finally:
        gateway.stop()


# ── intact link: buttons leave the send path for WhatsApp ──────

BUTTON_ROWS = [[("✅ Approve", "approve prop-1"),
                ("❌ Deny", "deny prop-1")],
               [("📋 Details", "details prop-1")]]


def test_whatsapp_buttons_reach_the_bridge_wire(bridge, tmp_path):
    adapter = WhatsAppAdapter(
        port=FAKE_PORT,
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    gateway = ChatGateway({"whatsapp": adapter}, db=None)
    gateway.start(lambda m: None)
    try:
        assert wait_for(lambda: adapter.connected.is_set())
        chat = ChatRef(platform="whatsapp", chat_id=PN_POSTER,
                       kind=ChatKind.DM)
        result = gateway.send("whatsapp", chat,
                              "Gift proposal open — decide:",
                              buttons=BUTTON_ROWS)
        assert result.ok
        sends = bridge.cmds("send")
        assert len(sends) == 1
        wire = [[tuple(pair) for pair in row]
                for row in sends[0]["buttons"]]
        assert wire == BUTTON_ROWS
    finally:
        gateway.stop()


def test_adapter_journals_a_degraded_button_form(tmp_path,
                                                  monkeypatch):
    fake = FakeBridge(send_form="text")
    try:
        adapter = WhatsAppAdapter(
            port=FAKE_PORT,
            media_dir=str(tmp_path / "media"),
            outbox_dir=str(tmp_path / "outbox"))
        gateway = ChatGateway({"whatsapp": adapter}, db=None)
        gateway.start(lambda m: None)
        try:
            assert wait_for(lambda: adapter.connected.is_set())
            notes: list = []
            monkeypatch.setattr(
                "nomorals.social.chat.whatsapp.record_failure",
                lambda site, exc: notes.append((site, exc)))
            chat = ChatRef(platform="whatsapp", chat_id=PN_POSTER,
                           kind=ChatKind.DM)
            result = gateway.send("whatsapp", chat, "decide:",
                                  buttons=BUTTON_ROWS)
            assert result.ok
            assert any("buttons" in str(site) for site, _ in notes), \
                "a degraded button form must be journaled, not silent"
        finally:
            gateway.stop()
    finally:
        fake.close()


def test_telegram_parity_for_the_same_presentation(bridge, tmp_path):
    """The same button rows a Telegram surface renders natively
    leave for WhatsApp too — one presentation, both platforms."""
    from nomorals.social.chat.base import ChatAdapter, SendResult

    class RecordingAdapter(ChatAdapter):
        name = "telegram-bot"
        supported_kinds = (ChatKind.DM, ChatKind.GROUP)

        def __init__(self):
            super().__init__(media_dir=str(tmp_path / "tgmedia"))
            self.sent: list = []

        def start(self, handler):
            return True

        def stop(self):
            return None

        def send(self, chat, text, **kwargs):
            self.sent.append((text, kwargs.get("buttons")))
            return SendResult(ok=True, platform=self.name,
                              message_id="tg-1")

    tg = RecordingAdapter()
    wa = WhatsAppAdapter(
        port=FAKE_PORT,
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    gateway = ChatGateway({"telegram-bot": tg, "whatsapp": wa},
                          db=None)
    gateway.start(lambda m: None)
    try:
        assert wait_for(lambda: wa.connected.is_set())
        chat_tg = ChatRef(platform="telegram-bot", chat_id="42",
                          kind=ChatKind.DM)
        chat_wa = ChatRef(platform="whatsapp", chat_id=PN_POSTER,
                          kind=ChatKind.DM)
        assert gateway.send("telegram-bot", chat_tg, "decide:",
                            buttons=BUTTON_ROWS).ok
        assert gateway.send("whatsapp", chat_wa, "decide:",
                            buttons=BUTTON_ROWS).ok
        assert tg.sent[0][1] == BUTTON_ROWS
        sends = bridge.cmds("send")
        wire = [[tuple(pair) for pair in row]
                for row in sends[0]["buttons"]]
        assert wire == BUTTON_ROWS
    finally:
        gateway.stop()


# ── the /pair rig (bare mixin, real supervisor + gateway) ──────

def _pair_rig(tmp_path):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.pairing import BridgeSupervisor

    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions",
        spawn=lambda argv, env: None,
        base_port=FAKE_PORT,
        binary_resolver=lambda: "/staged/whatsapp-bridge")

    chat = SimpleNamespace(
        whatsapp_enabled=True, whatsapp_host="127.0.0.1",
        whatsapp_port=8787, whatsapp_outbox_dir="",
        telegram_api_id="", telegram_api_hash="")
    settings = SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)

    class Rig(RuntimeProvisionMixin):
        def __init__(self):
            self._sup = sup
            self.settings = settings
            self.context = SimpleNamespace(settings=settings)
            self.gateway = ChatGateway(
                {"local": LocalAdapter()}, db=None)
            self.received: list = []

        def _pairing_supervisor(self):
            return self._sup

        def on_message(self, message):
            self.received.append(message)

    rig = Rig()
    rec = registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER, creds_dir=str(tmp_path / "sess"),
        port=FAKE_PORT, status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=True,
                         status_react=True, always_online=True)
    return rig, sup, registry, rec


def test_pair_list_warns_when_a_toggle_is_set_but_not_live(tmp_path):
    rig, sup, registry, rec = _pair_rig(tmp_path)
    try:
        out = rig._control_pair("")
        assert "status_viewing" in out
        assert "not live" in out  # set-on-but-dead must SAY so
    finally:
        rig.gateway.stop()


def test_pair_toggle_reply_names_the_live_state(tmp_path):
    rig, sup, registry, rec = _pair_rig(tmp_path)
    try:
        registry.set_toggles(rec.id, status_viewing=False)
        out = rig._control_pair(
            f"toggle {rec.id} status_viewing on")
        assert "status_viewing on" in out
        assert "not live" in out
    finally:
        rig.gateway.stop()


def test_pair_list_shows_live_when_the_surface_is_up(bridge,
                                                     tmp_path):
    rig, sup, registry, rec = _pair_rig(tmp_path)
    try:
        note = rig._sync_whatsapp_surface()
        assert note == ""
        adapter = rig.gateway.adapters.get("whatsapp")
        assert adapter is not None
        assert wait_for(lambda: adapter.connected.is_set())
        svc = rig._pairing_personal()
        svc["last_cycle"] = 0.0
        rig._housekeeping_beat()
        assert wait_for(
            lambda: len(bridge.cmds("presence")) > 0)
        out = rig._control_pair("")
        assert "not live" not in out
        assert "live" in out
    finally:
        rig.gateway.stop()


# ── intact links, end to end against real shapes ───────────────

def test_presence_toggle_reaches_the_wire_on_the_beat(bridge,
                                                      tmp_path):
    rig, sup, registry, rec = _pair_rig(tmp_path)
    try:
        rig._sync_whatsapp_surface()
        adapter = rig.gateway.adapters.get("whatsapp")
        assert wait_for(lambda: adapter.connected.is_set())
        svc = rig._pairing_personal()
        svc["last_cycle"] = 0.0
        rig._housekeeping_beat()
        assert wait_for(
            lambda: len(bridge.cmds("presence")) > 0)
        assert bridge.cmds("presence")[0]["state"] == "available"
        assert svc["personal"].presence_state(rec.id) == "online"
    finally:
        rig.gateway.stop()


def _runtime_rig(tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
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
        port=FAKE_PORT, status="linked", created_at=1.0))
    sup.registry.set_toggles(rec.id, status_saving=True,
                             status_viewing=True,
                             status_react=True, always_online=True)
    return runtime, sup, rec


@pytest.mark.parametrize("poster", [LID_POSTER, PN_POSTER])
def test_status_post_views_reacts_and_saves_end_to_end(bridge,
                                                       tmp_path,
                                                       poster):
    runtime, sup, rec = _runtime_rig(tmp_path)
    try:
        beat = ChatMessage(
            chat=ChatRef(platform="local", chat_id="console",
                         kind=ChatKind.DM, title="console",
                         peer="owner"),
            incoming=True, text="hello", sender="owner",
            sender_id="owner", message_id="m1", ts=time.time())
        runtime.on_message(beat)
        adapter = None
        assert wait_for(
            lambda: (runtime.gateway.adapters.get("whatsapp")
                     is not None))
        adapter = runtime.gateway.adapters["whatsapp"]
        assert wait_for(lambda: adapter.connected.is_set())
        bridge.emit({"type": "status_post", "id": "ST-E2E-1",
                     "from": {"id": poster, "name": "Ada"},
                     "text": "a status", "media": [],
                     "ts": int(time.time() * 1000)})
        assert wait_for(
            lambda: len(bridge.cmds("status_view")) > 0)
        assert bridge.cmds("status_view")[0]["from"] == poster
        assert wait_for(
            lambda: len(bridge.cmds("status_react")) > 0)
        assert bridge.cmds("status_react")[0]["from"] == poster
        personal = runtime._pairing_personal()["personal"]
        kinds = [row.get("kind") for row in personal.vault.journal()]
        assert "status" in kinds       # saved (status_saving)
        assert "status_react" in kinds
    finally:
        runtime.stop()
        runtime.gateway.stop()


def test_poststatus_reaches_the_wire(bridge, tmp_path):
    adapter = WhatsAppAdapter(
        port=FAKE_PORT,
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))
    gateway = ChatGateway({"whatsapp": adapter}, db=None)
    gateway.start(lambda m: None)
    try:
        assert wait_for(lambda: adapter.connected.is_set())
        verdict = adapter.send_status("shipping today")
        assert isinstance(verdict, dict) and verdict.get("ok")
        sends = bridge.cmds("status_send")
        assert len(sends) == 1
        assert sends[0]["text"] == "shipping today"
    finally:
        gateway.stop()
