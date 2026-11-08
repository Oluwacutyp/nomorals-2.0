"""WhatsApp live-attach + the self-chat owner surface.

The owner's live bug (2026-11-04): he paired his WhatsApp on the
Go engine (registry: linked, bridge running), then sent /help in
his WhatsApp self-chat — total silence.  Two defects, both pinned
here:

1.  Attach timing.  The conversational WhatsAppAdapter is built
    once at boot against the static config endpoint, and gateway
    start drops it when no bridge is listening yet (preflight).
    Nothing in the pairing lifecycle ever attached it afterwards,
    so a post-startup pairing stayed conversationally dark until a
    lucky restart.  ``RuntimeProvisionMixin._sync_whatsapp_surface``
    is the reconcile: resolved account + live bridge → an adapter
    on THAT bridge's port, started in the running process; account
    stopped/removed → the surface tears down.

2.  The self-chat.  Bridge ``own_message`` events were persist-only
    everywhere ("never replies to self" — the double-reply
    discipline for shared-account mode).  But a message the owner
    types in his own "Message yourself" chat can only be FOR
    Devon: it is now delivered as an incoming owner turn.  Own
    messages in every other chat keep the persist-only discipline,
    and Devon's own sends echoing back through the same event stay
    persist-only too (the sent-id guard) — she never answers
    herself.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
)

OWNER_JID = "2348012345678@s.whatsapp.net"
OTHER_JID = "2347099999999@s.whatsapp.net"
BASE_PORT = 18901


# ── fakes (the pairing suites' idiom) ───────────────────────────────

class FakeProc:
    """A bridge process that parks alive until terminated."""

    def __init__(self, lines=()):
        self.terminated = False
        self._poll = None
        self._gate = threading.Event()
        self.stdout = self._stream(list(lines))

    def _stream(self, lines):
        yield from lines
        self._gate.wait(30)

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0
        self._gate.set()


def make_supervisor(tmp_path, base_port=BASE_PORT, lines=()):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=base_port,
        binary_resolver=lambda: "/staged/whatsapp-bridge")
    return sup, registry, spawns


def make_settings(tmp_path, *, whatsapp_enabled=True):
    chat = SimpleNamespace(
        whatsapp_enabled=whatsapp_enabled,
        whatsapp_host="127.0.0.1",
        whatsapp_port=8787,  # the static config endpoint — NOT the
        whatsapp_outbox_dir="",  # record's port; the adapter must
        telegram_api_id="", telegram_api_hash="")  # follow the pool
    return SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)


class SurfaceRuntime(RuntimeProvisionMixin):
    """The provision mixin on a bare harness: the real supervisor
    and the real gateway, stub settings, a recording handler.  The
    gateway starts with NO whatsapp adapter — the owner's boot
    state after preflight dropped it."""

    def __init__(self, sup, settings):
        self._sup = sup
        self.settings = settings
        self.context = SimpleNamespace(settings=settings)
        self.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
        self.received: list = []

    def _pairing_supervisor(self):
        return self._sup

    def on_message(self, message):
        self.received.append(message)

    def teardown(self):
        self.gateway.stop()


# ── 1. live-attach ──────────────────────────────────────────────────

def test_pair_after_startup_attaches_surface_without_restart(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path))
    try:
        assert "whatsapp" not in rt.gateway.adapters
        rec = sup.pair_whatsapp("2348012345678")
        assert rec.port == BASE_PORT  # the pool's own port, not 8787
        rt._sync_whatsapp_surface()
        adapter = rt.gateway.adapters.get("whatsapp")
        assert adapter is not None
        assert adapter.port == BASE_PORT
        assert adapter._thread is not None and adapter._thread.is_alive()
        assert rt._wa_surface_account == rec.id
        # A second reconcile is a no-op: same adapter, still running.
        rt._sync_whatsapp_surface()
        assert rt.gateway.adapters["whatsapp"] is adapter
    finally:
        rt.teardown()


def test_surface_tears_down_when_account_removed(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path))
    try:
        rec = sup.pair_whatsapp("2348012345678")
        rt._sync_whatsapp_surface()
        assert "whatsapp" in rt.gateway.adapters
        sup.remove(rec.id)
        rt._sync_whatsapp_surface()
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.teardown()


def test_surface_tears_down_when_account_stopped(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path))
    try:
        rec = sup.pair_whatsapp("2348012345678")
        rt._sync_whatsapp_surface()
        assert "whatsapp" in rt.gateway.adapters
        sup.stop(rec.id)
        rt._sync_whatsapp_surface()
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.teardown()


def test_no_attach_when_platform_disabled_in_config(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path,
                                           whatsapp_enabled=False))
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        rt._sync_whatsapp_surface()  # must not raise, must not attach
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.teardown()


def test_pair_status_names_the_disabled_platform(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path,
                                           whatsapp_enabled=False))
    rec = AccountRecord(id="whatsapp:2348012345678", platform="whatsapp",
                        identity="2348012345678", status="linked",
                        port=BASE_PORT)
    registry.upsert(rec)
    reply = rt._control_pair("")
    assert "disabled in config" in reply
    assert "whatsapp_enabled=false" in reply


def test_sync_surfaces_disabled_config_on_the_record(tmp_path):
    """Beat-path surfacing: with no /pair reply to append to, the
    failure lands on the account's last_error (tagged as the
    surface's own note) — and a later successful attach clears
    exactly that note."""
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path,
                                           whatsapp_enabled=False))
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        note = rt._sync_whatsapp_surface()
        assert "disabled in config" in note
        stored = registry.get(rec.id)
        assert stored.last_error.startswith("chat surface: ")
        assert "disabled in config" in stored.last_error
        # Enable the platform: the next reconcile attaches AND
        # retires its own note.
        rt.settings = make_settings(tmp_path)
        rt.context = SimpleNamespace(settings=rt.settings)
        assert rt._sync_whatsapp_surface() == ""
        assert "whatsapp" in rt.gateway.adapters
        assert registry.get(rec.id).last_error == ""
    finally:
        rt.teardown()


def test_pair_reply_surfaces_the_disabled_platform(tmp_path):
    """The /pair whatsapp reply itself carries the surface
    failure — the owner learns it in the same message as the
    code, not by discovering silence later."""
    lines = ["PAIRING CODE for +2348012345678: WXYZ2345"]
    sup, registry, _ = make_supervisor(tmp_path, lines=lines)
    rt = SurfaceRuntime(sup, make_settings(tmp_path,
                                           whatsapp_enabled=False))
    try:
        reply = rt._control_pair("whatsapp 2348012345678")
        assert "WXYZ2345" in reply
        assert "disabled in config" in reply
        assert "whatsapp" not in rt.gateway.adapters
    finally:
        rt.teardown()


def test_pair_status_quiet_about_platform_when_enabled(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = SurfaceRuntime(sup, make_settings(tmp_path))
    rec = AccountRecord(id="whatsapp:2348012345678", platform="whatsapp",
                        identity="2348012345678", status="linked",
                        port=BASE_PORT)
    registry.upsert(rec)
    reply = rt._control_pair("")
    assert "disabled in config" not in reply


# ── 2. the self-chat owner surface (adapter level) ──────────────────

def make_adapter(tmp_path):
    return WhatsAppAdapter(
        host="127.0.0.1", port=BASE_PORT,
        media_dir=str(tmp_path / "media"),
        outbox_dir=str(tmp_path / "outbox"))


def dispatch_status(adapter, msgs, user=OWNER_JID):
    adapter._dispatch({"type": "status", "state": "open", "user": user},
                      msgs.append)


def own_event(chat_id, kind="dm", text="/help", mid="m1"):
    return {"type": "own_message", "id": mid,
            "chat": {"id": chat_id, "kind": kind, "title": ""},
            "text": text, "ts": 1_700_000_000_000}


def test_self_chat_message_is_an_incoming_owner_turn(tmp_path):
    adapter = make_adapter(tmp_path)
    msgs: list = []
    dispatch_status(adapter, msgs)
    adapter._dispatch(own_event(OWNER_JID), msgs.append)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.incoming is True
    assert msg.text == "/help"
    assert msg.chat.kind == ChatKind.DM
    assert msg.chat.chat_id == OWNER_JID
    assert msg.sender_id == OWNER_JID
    assert msg.meta.get("from_owner") is True
    assert msg.meta.get("self_chat") is True
    # The persist-only stamp must NOT ride along: the brain's
    # own-message step would swallow the turn as an echo.
    assert "own_message" not in msg.meta


def test_self_chat_recognized_with_device_part_and_c_us(tmp_path):
    adapter = make_adapter(tmp_path)
    msgs: list = []
    dispatch_status(adapter, msgs)
    adapter._dispatch(
        own_event("2348012345678:4@c.us", text="hello"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is True
    assert msgs[0].meta.get("self_chat") is True


def test_own_message_in_third_party_dm_stays_persist_only(tmp_path):
    adapter = make_adapter(tmp_path)
    msgs: list = []
    dispatch_status(adapter, msgs)
    adapter._dispatch(own_event(OTHER_JID, text="hi there"),
                      msgs.append)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.incoming is False
    assert msg.meta.get("own_message") is True
    assert "self_chat" not in msg.meta


def test_own_message_in_group_stays_persist_only(tmp_path):
    adapter = make_adapter(tmp_path)
    msgs: list = []
    dispatch_status(adapter, msgs)
    adapter._dispatch(
        own_event("120363012345678901@g.us", kind="group",
                  text="gm all"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is False
    assert msgs[0].meta.get("own_message") is True


def test_own_send_echo_in_self_chat_stays_persist_only(tmp_path):
    """Devon's own reply into the self-chat echoes back through the
    bridge as an own_message with the id her send was assigned —
    it must NOT come back in as a fresh owner turn (reply loop)."""
    adapter = make_adapter(tmp_path)
    msgs: list = []
    dispatch_status(adapter, msgs)
    adapter.connected.set()
    adapter._send_cmd = lambda cmd, timeout=25.0: {
        "id": cmd.get("id"), "ok": True, "message_id": "ECHO-1"}
    chat = ChatRef(platform="whatsapp", chat_id=OWNER_JID,
                   kind=ChatKind.DM, peer=OWNER_JID)
    result = adapter.send(chat, "the help menu")
    assert result.ok and result.message_id == "ECHO-1"
    adapter._dispatch(own_event(OWNER_JID, text="the help menu",
                                mid="ECHO-1"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is False
    assert msgs[0].meta.get("own_message") is True
    # …while the owner's own typing (an id she never sent) is a turn.
    adapter._dispatch(own_event(OWNER_JID, text="/pair", mid="OWN-9"),
                      msgs.append)
    assert len(msgs) == 2
    assert msgs[1].incoming is True
    assert msgs[1].meta.get("self_chat") is True


def test_self_chat_unknown_until_status_event(tmp_path):
    """Before the bridge reports the account's own JID there is
    no proof a chat is the SELF-CHAT — and none is claimed: no
    self_chat stamp is applied until the status event teaches
    the identity.  What changed (2026-11-08 truth fix): a
    REGISTERED COMMAND in the message still promotes, on the
    own_message event's own provenance — the bridge emits that
    event only for the paired account's own messages, the
    same authorship proof the group carve-out always trusted.
    Non-command own text stays persist-only, identity or not."""
    adapter = make_adapter(tmp_path)
    msgs: list = []
    adapter._dispatch(own_event(OWNER_JID), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is True
    assert msgs[0].meta.get("owner_command") is True
    assert msgs[0].meta.get("from_owner") is True
    # the chat is NOT classified as the self-chat: that stamp
    # still waits for the bridge's identity report
    assert "self_chat" not in msgs[0].meta
    assert "own_message" not in msgs[0].meta
    plain: list = []
    adapter._dispatch(
        own_event(OWNER_JID, text="note to self", mid="m2"),
        plain.append)
    assert len(plain) == 1
    assert plain[0].incoming is False
    assert plain[0].meta.get("own_message") is True


# ── owner verdicts downstream of the adapter ────────────────────────

def _self_chat_message():
    msg = ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=OWNER_JID,
                     kind=ChatKind.DM, peer=OWNER_JID),
        incoming=True, text="/help", sender="",
        sender_id=OWNER_JID, message_id="m1", ts=time.time())
    msg.meta["from_owner"] = True
    msg.meta["self_chat"] = True
    return msg


def test_gateway_stamps_self_chat_as_owner():
    gw = ChatGateway({"local": LocalAdapter()}, db=None)
    seen: list = []
    gw._inbound = seen.append
    msg = _self_chat_message()
    gw._on_inbound(msg)
    assert msg.meta.get("is_owner") is True
    assert seen == [msg]
    # Control: an ordinary WhatsApp DM is not the owner.
    other = ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=OTHER_JID,
                     kind=ChatKind.DM, peer=OTHER_JID),
        incoming=True, text="hi", sender="Ada",
        sender_id=OTHER_JID, message_id="m2", ts=time.time())
    gw._on_inbound(other)
    assert other.meta.get("is_owner") is False


def test_platforms_listing_includes_never_built_platforms():
    """The owner's live dead end: /platforms showed telegram
    only, because gateway.status() names just built adapters and
    the EC2 boot set was telegram-only — whatsapp was INVISIBLE,
    so /start whatsapp was undiscoverable.  The listing now names
    every catalogue platform with its state and its start verb."""
    from nomorals.agents.partner.runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    reply = rt.handle_control("/platforms", "local:owner")
    assert "local:" in reply
    assert "whatsapp: not running — start with /start whatsapp" in reply
    assert "telegram: not running — start with /start telegram" in reply
    # known_platforms() is the catalogue union, not just the boot set.
    known = rt.gateway.known_platforms()
    assert "whatsapp" in known and "local" in known


def test_runtime_operator_test_honors_self_chat():
    from nomorals.agents.partner.runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt._owner_chats = set()
    rt._owner_sender_ids = set()
    assert rt._is_operator(_self_chat_message()) is True
    plain = ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=OTHER_JID,
                     kind=ChatKind.DM, peer=OTHER_JID),
        incoming=True, text="/help", sender="Ada",
        sender_id=OTHER_JID, message_id="m3", ts=time.time())
    assert rt._is_operator(plain) is False
