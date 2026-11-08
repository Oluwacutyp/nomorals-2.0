"""PARITY G1 — WhatsApp message REACTIONS + message FORWARDING
(platform parity census gaps #1 and #2, 2026-11-07).

The census graded both ABSENT: reactions existed only for
statuses (status_react) and forwarding only as an inbound
forwarded-flag.  Neither bridge engine had a verb.  This suite
drives the real composed chain — owner command → handle_control
→ runtime_chatops → WhatsApp adapter → recording fake bridge —
and a capability counts only when its verb FIRES on the wire
with the right target (the truth-pass standard):

  * /react is the expression family's §22 command — it already
    routed to ``adapter.react()`` and WhatsApp inherited the
    base class's "reactions not supported".  G1 builds the
    verb: bridge ``react`` {chat, message_id, emoji} on both
    engines + the adapter override, so the command lands on
    WhatsApp at last.  On this surface the act is AS THE
    OWNER: a standing owner stop halts it before any wire
    traffic and a landed reaction journals
    (message_reacted).  In groups the command keeps its §33
    posture (unclaimed rooms refuse it); the verb itself is
    group-capable, pinned at the adapter.
  * /forward <chat> as a reply → bridge ``forward`` {chat,
    message_id, dest}; the destination resolves through the
    human chat-target grammar (a bare number = that person's
    WhatsApp DM).  An as-him act like /poststatus: owner stop
    halts it, landed forwards journal (message_forwarded),
    refusals answer in plain lines — no destination, an
    unresolvable destination, a cross-platform destination
    (a forward stays on its own platform).
  * Telegram's surfaces already carry native react /
    forward_post verbs — /forward rides forward_post there
    (journaled); /react's bot-hat behavior is unchanged.

Bridge-side construction (whatsmeow BuildReaction on the
message's real key; the original payload re-sent with the
forwarded mark; Baileys react/forward forms) is pinned by the
bridges' own builds — the wire here is the adapter protocol,
whose shapes mirror the sibling verbs (message_star /
status_react) field for field.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, SendResult

PORT = 19431
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "12345@g.us"
TG_OWNER = "777"


# ── recording fake bridge (the bridges' wire protocol) ─────────

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
                if cmd.get("cmd") == "forward":
                    reply["message_id"] = "WAMID-FWD-1"
                self.wfile.write(json.dumps(reply).encode() + b"\n")
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


class FakeTelegramAdapter:
    """The personal Telegram seat: native react + forward_post
    verbs that record what they were asked to do."""

    name = "telegram"

    def __init__(self):
        self.reacted: list[tuple] = []
        self.forwarded: list[tuple] = []

    def start(self, handler):
        self._handler = handler
        return True

    def stop(self):
        pass

    def react(self, chat, message_id, emoji):
        self.reacted.append((chat.chat_id, message_id, emoji))
        return SendResult(ok=True, platform=self.name,
                          message_id=str(message_id))

    def forward_post(self, from_chat, message_id, to_chat):
        self.forwarded.append((from_chat.chat_id, message_id,
                               to_chat.chat_id))
        return SendResult(ok=True, platform=self.name,
                          message_id="tg-fwd-1")


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parityg1")
    bridge = FakeBridge()
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": TG_OWNER,
        "chat.local_enabled": "true",
        "chat.whatsapp_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    tg = FakeTelegramAdapter()
    gateway = ChatGateway(
        {"local": LocalAdapter(), "telegram": tg}, db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    rec = sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                      / f"wa-{OWNER}"),
        port=PORT, status="linked", created_at=1.0))
    sup.registry.upsert(AccountRecord(
        id="telegram:777", platform="telegram", identity="777",
        creds_dir=str(ws / "data" / "telegram_sessions" / "tg-777"),
        port=0, status="linked", created_at=1.0))

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
        lambda: runtime.gateway.adapters.get("whatsapp") is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {
        "runtime": runtime, "bridge": bridge, "adapter": adapter,
        "sup": sup, "rec": rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "tg": tg, "tmp": tmp, "beat": beat,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, reply_to="", sender_id=OWNER, mid=""):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title="", peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=mid or f"op-{time.time()}",
        reply_to=reply_to, ts=time.time(),
        meta={"owner_command": True})
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_kinds(rig, account_id):
    return [r.get("kind")
            for r in rig["vault"].journal(account_id)]


# ── /react on WhatsApp ─────────────────────────────────────────
# /react is the expression family's §22 command (it predates
# G1): her persona reacts to the replied message — or, with no
# reply, to the command message itself — and picks an emoji
# when none is named.  What G1 adds is the VERB: the WhatsApp
# adapter's react() over both bridge engines, plus the
# personal-surface law on this surface (owner stop halts it,
# landed acts journal).  A landed reaction answers silently
# ("") — the reaction itself is the answer.

def test_react_fires_on_dm_with_emoji(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("react"))
    reply = _op(rig, "/react 🔥", reply_to="DM-2")
    assert reply == ""
    assert wait_for(lambda: len(bridge.cmds("react")) > before)
    cmd = bridge.cmds("react")[-1]
    assert cmd["chat"] == PN_POSTER
    assert cmd["message_id"] == "DM-2"
    assert cmd["emoji"] == "🔥"
    assert "message_reacted" in _journal_kinds(rig, rig["rec"].id)


def test_react_verb_fires_for_group_chat(rig):
    # The verb itself is group-capable (the bridge builds the
    # key with the author as participant); the composed pin
    # drives the adapter the command path drives.
    from nomorals.social.chat.base import ChatRef as _Ref
    chat = _Ref(platform="whatsapp", chat_id=GROUP,
                kind=ChatKind.GROUP, title="", peer=GROUP)
    before = len(rig["bridge"].cmds("react"))
    verdict = rig["adapter"].react_message(chat, "GRP-7", "👍")
    assert verdict.get("ok")
    assert wait_for(
        lambda: len(rig["bridge"].cmds("react")) > before)
    cmd = rig["bridge"].cmds("react")[-1]
    assert cmd["chat"] == GROUP
    assert cmd["message_id"] == "GRP-7"
    assert cmd["emoji"] == "👍"


def test_react_group_command_is_33_gated(rig):
    # In a group the §22 command is bot-system output: an
    # unclaimed room refuses it (§33) — the same posture every
    # expression command stands under, unchanged by G1.
    bridge = rig["bridge"]
    before = len(bridge.cmds("react"))
    reply = _op(rig, "/react 👍", chat_id=GROUP,
                kind=ChatKind.GROUP, reply_to="GRP-7")
    assert "isn't claimed" in reply
    assert len(bridge.cmds("react")) == before


def test_react_bare_picks_an_emoji(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("react"))
    reply = _op(rig, "/react", reply_to="DM-2")
    assert reply == ""
    assert wait_for(lambda: len(bridge.cmds("react")) > before)
    cmd = bridge.cmds("react")[-1]
    assert cmd["message_id"] == "DM-2"
    assert cmd["emoji"]  # she picked one — never an empty send


def test_react_without_reply_targets_own_message(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("react"))
    reply = _op(rig, "/react 🔥", mid="CMD-SELF-1")
    assert reply == ""
    assert wait_for(lambda: len(bridge.cmds("react")) > before)
    assert bridge.cmds("react")[-1]["message_id"] == "CMD-SELF-1"


def test_react_owner_stop_halts(rig):
    bridge = rig["bridge"]
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(bridge.cmds("react"))
        reply = _op(rig, "/react 🔥", reply_to="DM-2")
        assert "standing stop" in reply
        assert len(bridge.cmds("react")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


# ── /forward on WhatsApp ───────────────────────────────────────

def test_forward_fires_to_number_dest(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("forward"))
    reply = _op(rig, "/forward 2348012345678", reply_to="DM-2")
    assert reply.startswith("forwarded to ")
    assert wait_for(lambda: len(bridge.cmds("forward")) > before)
    cmd = bridge.cmds("forward")[-1]
    assert cmd["chat"] == PN_POSTER
    assert cmd["message_id"] == "DM-2"
    assert "2348012345678" in cmd["dest"]
    assert "message_forwarded" in _journal_kinds(
        rig, rig["rec"].id)


def test_forward_from_group_fires(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("forward"))
    reply = _op(rig, "/forward 2348012345678", chat_id=GROUP,
                kind=ChatKind.GROUP, reply_to="GRP-9")
    assert reply.startswith("forwarded to ")
    assert wait_for(lambda: len(bridge.cmds("forward")) > before)
    cmd = bridge.cmds("forward")[-1]
    assert cmd["chat"] == GROUP
    assert cmd["message_id"] == "GRP-9"


def test_forward_needs_reply_and_destination(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("forward"))
    reply = _op(rig, "/forward 2348012345678")
    assert "reply to the message" in reply
    reply = _op(rig, "/forward", reply_to="DM-2")
    assert "needs 1 argument" in reply  # the catalog arity gate
    assert len(bridge.cmds("forward")) == before


def test_forward_unresolvable_destination_answers(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("forward"))
    reply = _op(rig, "/forward Zqx Nonexistent Chat",
                reply_to="DM-2")
    assert "no chat matches" in reply
    assert len(bridge.cmds("forward")) == before


def test_forward_cross_platform_refused(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("forward"))
    reply = _op(rig, "/forward telegram:777", reply_to="DM-2")
    assert "stays on its own platform" in reply
    assert len(bridge.cmds("forward")) == before


def test_forward_owner_stop_halts(rig):
    bridge = rig["bridge"]
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(bridge.cmds("forward"))
        reply = _op(rig, "/forward 2348012345678",
                    reply_to="DM-2")
        assert "standing stop" in reply
        assert len(bridge.cmds("forward")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


# ── Telegram seat: the same commands ride the native verbs ─────

def test_react_on_telegram_rides_native_verb(rig):
    tg = rig["tg"]
    reply = _op(rig, "/react 🔥", platform="telegram",
                chat_id="555", reply_to="42", sender_id=TG_OWNER)
    assert reply == ""
    assert tg.reacted == [("555", "42", "🔥")]


def test_forward_on_telegram_rides_native_verb(rig):
    tg = rig["tg"]
    reply = _op(rig, "/forward telegram:999", platform="telegram",
                chat_id="555", reply_to="42", sender_id=TG_OWNER)
    assert reply.startswith("forwarded to ")
    assert tg.forwarded == [("555", "42", "999")]
    assert "message_forwarded" in _journal_kinds(
        rig, "telegram:777")


# ── registration: the catalog knows both commands ──────────────

def test_catalog_registration():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )

    # /forward is a chatops kind, spelled in every catalog.
    assert "forward" in CHATOPS_COMMANDS
    assert "forward" in CONTROL_COMMANDS
    assert "forward" in COMMAND_DETAILS
    assert "forward" in LIST_ONELINERS
    listed = {k for _title, keys in LIST_GROUPS for k in keys}
    assert "forward" in listed
    # /react stays the expression family's §22 kind — G1 gave
    # its WhatsApp path the missing verb; it did not move the
    # command.  Exactly one catalog entry owns it.
    assert "react" not in CHATOPS_COMMANDS
    assert "react" in CONTROL_COMMANDS
    assert "react" in COMMAND_DETAILS
    assert "react" in LIST_ONELINERS
    assert "react" in listed
