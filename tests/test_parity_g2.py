"""PARITY G2 — Telegram userbot CREATE + LEAVE and WhatsApp
JOIN VIA INVITE LINK (platform parity census bundles #3 and
#4, 2026-11-07).

The census graded both ABSENT: no CreateChannelRequest /
CreateChatRequest / LeaveChannelRequest existed anywhere in
the repo (the MTProto account could join but never create or
leave), and no bridge engine carried a group-join verb
(Devon could issue and revoke invite links but never accept
one).  This suite drives the real composed chain — owner
command → handle_control → runtime_chatops → adapter →
recording fake bridge — plus fake-Telethon adapter pins, and
a capability counts only when its verb FIRES with the right
target (the truth-pass standard):

  * /joinlink (WhatsApp): the bridge ``group_join_link`` verb
    (whatsmeow JoinGroupWithLink + GetGroupInfoFromLink;
    Baileys groupGetInviteInfo / groupAcceptInvite /
    groupMetadata) resolves the link BEFORE any join attempt
    — a revoked link and a malformed link are distinct
    failures — and the landed state is read back from
    membership, because JoinGroupWithLink answers the same
    JID for a real join and for a filed approval request:
    joined, already_member (no act, NO journal row) and
    approval_pending (the request is in; the owner is NOT a
    member) are three different answers, and only the two
    landed acts journal (group_joined / group_join_requested).
  * /newgroup · /newchannel (Telegram personal): the userbot
    verbs CreateChatRequest (users=[]) / CreateChannelRequest
    (megagroup=False); refusals (CHANNELS_TOO_MUCH) return in
    Telegram's own words; other surfaces are told where the
    command lives.
  * /leave: the chatops target grammar (this chat, or one
    named the way chats are named); WhatsApp rides
    group_leave, Telegram rides the userbot's leave_chat
    (LeaveChannelRequest for channels/supergroups,
    DeleteChatUserRequest removing "me" for basic groups —
    typed-entity and raw-marked-id paths both pinned); a DM
    answers honestly that it isn't leavable.

Bridge-side construction is pinned by source assertions on
both engines and by the rebuilt binaries carrying the new
verb's name; the wire here is the adapter protocol, whose
shapes mirror the sibling verbs field for field.
"""
from __future__ import annotations

import asyncio
import json
import socketserver
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

PORT = 19441
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
GROUP_SUBJECT = "Gold Room"
TG_OWNER = "777"
TG_GROUP = "-1001234567890"
ROOT = Path(__file__).resolve().parents[1]


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
                if cmd.get("cmd") == "group_list":
                    reply["groups"] = [{
                        "id": GROUP, "subject": GROUP_SUBJECT,
                        "size": 3, "isCommunity": False,
                        "isCommunityAnnounce": False,
                        "linkedParent": ""}]
                if cmd.get("cmd") == "group_join_link":
                    if srv.join_error == "revoked":
                        reply = {
                            "id": cid, "ok": False,
                            "error": "that invite link has "
                                     "been revoked"}
                    elif srv.join_error == "invalid":
                        reply = {
                            "id": cid, "ok": False,
                            "error": "that invite link is "
                                     "not valid"}
                    else:
                        reply = {
                            "id": cid, "ok": True,
                            "state": srv.join_state,
                            "chat": GROUP,
                            "subject": GROUP_SUBJECT}
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
        self.join_state = "joined"
        self.join_error: str | None = None
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
    """The personal Telegram seat for the composed rig: the
    G2 verbs record what they were asked to do."""

    name = "telegram"

    def __init__(self):
        self.created: list[tuple] = []
        self.left: list[str] = []
        self.create_error = ""

    def start(self, handler):
        self._handler = handler
        return True

    def stop(self):
        pass

    def create_chat(self, title):
        self.created.append(("group", title))
        if self.create_error:
            return {"ok": False, "error": self.create_error}
        return {"ok": True, "chat_id": "-4242", "title": title}

    def create_channel(self, title):
        self.created.append(("channel", title))
        if self.create_error:
            return {"ok": False, "error": self.create_error}
        return {"ok": True, "chat_id": "-100777001",
                "title": title}

    def leave_chat(self, chat):
        self.left.append(chat.chat_id)
        return {"ok": True, "chat_id": chat.chat_id}


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parityg2")
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
    tg_rec = sup.registry.upsert(AccountRecord(
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
        lambda: runtime.gateway.adapters.get("whatsapp")
        is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {
        "runtime": runtime, "bridge": bridge, "adapter": adapter,
        "sup": sup, "rec": rec, "tg_rec": tg_rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "tg": tg, "tmp": tmp, "beat": beat,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, title="", sender_id=None, mid=""):
    if sender_id is None:
        sender_id = OWNER if platform == "whatsapp" else TG_OWNER
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


def _journal_kinds(rig, account_id):
    return [r.get("kind") for r in _journal_rows(rig, account_id)]


# ── bridge source pins (both engines + the rebuilt binaries) ──

def test_go_bridge_join_link_state_machine():
    src = (ROOT / "bridge" / "gobridge" / "groups.go").read_text()
    assert 'case "group_join_link"' in src
    assert "GetGroupInfoFromLink" in src
    assert "JoinGroupWithLink" in src
    assert "ErrInviteLinkRevoked" in src
    assert "ErrInviteLinkInvalid" in src
    assert '"approval_pending"' in src
    assert '"already_member"' in src
    # membership is re-read after the accept (GetGroupInfo on
    # the joined JID) — the accept's return alone never decides
    assert src.count("GetGroupInfo(ctx") >= 2


def test_baileys_join_link_handler():
    src = (ROOT / "bridge"
           / "whatsapp-groups-admin.mjs").read_text()
    assert "group_join_link" in src
    assert "groupGetInviteInfo" in src
    assert "groupAcceptInvite" in src
    assert "groupMetadata" in src
    # dead links map distinctly BEFORE the accept
    assert src.index("groupGetInviteInfo") \
        < src.index("groupAcceptInvite")
    assert "410" in src and "revoked" in src
    assert "approval_pending" in src


def test_rebuilt_binaries_carry_the_verb():
    # Switch-case command names never survive contiguously in
    # a Go binary (the long-standing group_leave is absent
    # too — string switches compile to chunked compares), so
    # the pin reads the strings only this build's state
    # machine carries: its state names and its refusal line.
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"approval_pending" in data, name
        assert b"that invite link has been revoked" in data, name


def test_adapter_normalizes_link_before_traffic(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(port=19999)
    calls: list[tuple] = []

    def fake_admin(cmd, payload, *, timeout=30.0):
        calls.append((cmd, payload))
        return {"ok": True, "state": "joined",
                "chat": GROUP, "subject": GROUP_SUBJECT}

    monkeypatch.setattr(adapter, "_admin_cmd", fake_admin)
    verdict = adapter.group_join_link(
        "https://chat.whatsapp.com/AbC123xYz")
    assert calls == [("group_join_link", {"code": "AbC123xYz"})]
    assert verdict["state"] == "joined"
    adapter.group_join_link("AbC123xYz")
    assert calls[-1] == ("group_join_link", {"code": "AbC123xYz"})
    calls.clear()
    assert adapter.group_join_link("   ")["ok"] is False
    assert adapter.group_join_link(
        "https://chat.whatsapp.com/")["ok"] is False
    assert calls == []  # both empties refused before traffic


# ── /joinlink composed (real runtime → real adapter → bridge) ──

def test_joinlink_url_joined_fires_and_journals(rig):
    bridge = rig["bridge"]
    bridge.join_state = "joined"
    bridge.join_error = None
    before = len(bridge.cmds("group_join_link"))
    reply = _op(rig, "/joinlink https://chat.whatsapp.com/AbC123xYz")
    assert reply == f"joined “{GROUP_SUBJECT}”."
    assert wait_for(
        lambda: len(bridge.cmds("group_join_link")) > before)
    cmd = bridge.cmds("group_join_link")[-1]
    assert cmd["code"] == "AbC123xYz"  # the URL was normalized
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "group_joined"]
    assert rows and rows[-1]["counterpart_id"] == GROUP


def test_joinlink_bare_code(rig):
    bridge = rig["bridge"]
    bridge.join_state = "joined"
    bridge.join_error = None
    before = len(bridge.cmds("group_join_link"))
    reply = _op(rig, "/joinlink AbC123xYz")
    assert reply.startswith("joined")
    assert wait_for(
        lambda: len(bridge.cmds("group_join_link")) > before)
    assert bridge.cmds("group_join_link")[-1]["code"] \
        == "AbC123xYz"


def test_joinlink_already_member_journals_nothing(rig):
    bridge = rig["bridge"]
    bridge.join_state = "already_member"
    bridge.join_error = None
    rows_before = len(_journal_rows(rig, rig["rec"].id))
    reply = _op(rig, "/joinlink AbC123xYz")
    assert "already a member" in reply
    assert len(_journal_rows(rig, rig["rec"].id)) == rows_before


def test_joinlink_approval_pending_is_not_joined(rig):
    bridge = rig["bridge"]
    bridge.join_state = "approval_pending"
    bridge.join_error = None
    joined_before = _journal_kinds(rig, rig["rec"].id).count(
        "group_joined")
    reply = _op(rig, "/joinlink AbC123xYz")
    assert "not a member" in reply
    kinds = _journal_kinds(rig, rig["rec"].id)
    assert "group_join_requested" in kinds
    assert kinds.count("group_joined") == joined_before


def test_joinlink_revoked_link(rig):
    bridge = rig["bridge"]
    bridge.join_error = "revoked"
    rows_before = len(_journal_rows(rig, rig["rec"].id))
    reply = _op(rig, "/joinlink DeadCode1")
    assert "revoked" in reply
    assert len(_journal_rows(rig, rig["rec"].id)) == rows_before
    bridge.join_error = None


def test_joinlink_invalid_link_is_a_different_line(rig):
    bridge = rig["bridge"]
    bridge.join_error = "invalid"
    reply = _op(rig, "/joinlink NotACode")
    assert "not valid" in reply
    assert "revoked" not in reply
    bridge.join_error = None


def test_joinlink_usage_no_traffic(rig):
    # A bare /joinlink is answered by the catalog's arity
    # layer (the command takes 1+ arguments) with a pointer to
    # /help — before any handler or wire traffic.
    bridge = rig["bridge"]
    before = len(bridge.cmds("group_join_link"))
    reply = _op(rig, "/joinlink")
    assert "needs 1 argument" in reply
    assert "/help joinlink" in reply
    assert len(bridge.cmds("group_join_link")) == before


def test_joinlink_owner_stop_halts(rig):
    bridge = rig["bridge"]
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(bridge.cmds("group_join_link"))
        reply = _op(rig, "/joinlink AbC123xYz")
        assert "standing stop" in reply
        assert len(bridge.cmds("group_join_link")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


# ── /leave composed — WhatsApp ─────────────────────────────────

def test_leave_wa_group_here(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("group_leave"))
    reply = _op(rig, "/leave", chat_id=GROUP, kind=ChatKind.GROUP,
                title=GROUP_SUBJECT)
    assert reply == f"left {GROUP_SUBJECT}."
    assert wait_for(lambda: len(bridge.cmds("group_leave")) > before)
    assert bridge.cmds("group_leave")[-1]["chat"] == GROUP
    assert "chat_left" in _journal_kinds(rig, rig["rec"].id)


def test_leave_wa_group_by_name(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("group_leave"))
    reply = _op(rig, f"/leave {GROUP_SUBJECT}")
    assert reply == f"left {GROUP_SUBJECT}."
    assert wait_for(lambda: len(bridge.cmds("group_leave")) > before)
    assert bridge.cmds("group_leave")[-1]["chat"] == GROUP


def test_leave_wa_dm_is_honest(rig):
    bridge = rig["bridge"]
    before = len(bridge.cmds("group_leave"))
    reply = _op(rig, "/leave")
    assert "isn't leavable" in reply
    assert len(bridge.cmds("group_leave")) == before


def test_leave_owner_stop_halts(rig):
    bridge = rig["bridge"]
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(bridge.cmds("group_leave"))
        reply = _op(rig, "/leave", chat_id=GROUP,
                    kind=ChatKind.GROUP, title=GROUP_SUBJECT)
        assert "standing stop" in reply
        assert len(bridge.cmds("group_leave")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


# ── /newgroup · /newchannel · /leave composed — Telegram ──────

def test_newgroup_fires_and_journals(rig):
    reply = _op(rig, "/newgroup Think Tank", platform="telegram",
                chat_id=TG_OWNER, kind=ChatKind.DM)
    assert rig["tg"].created == [("group", "Think Tank")]
    assert reply == "created your Telegram group “Think Tank”."
    rows = [r for r in _journal_rows(rig, rig["tg_rec"].id)
            if r.get("kind") == "chat_created"]
    assert rows and rows[-1]["counterpart_id"] == "-4242"


def test_newchannel_fires(rig):
    rig["tg"].created.clear()
    reply = _op(rig, "/newchannel Devon Builds",
                platform="telegram", chat_id=TG_OWNER,
                kind=ChatKind.DM)
    assert rig["tg"].created == [("channel", "Devon Builds")]
    assert reply == "created your Telegram channel “Devon Builds”."


def test_newgroup_wrong_surface_names_its_home(rig):
    created_before = list(rig["tg"].created)
    reply = _op(rig, "/newgroup Nope")  # sent from WhatsApp
    assert "lives on your personal Telegram" in reply
    assert rig["tg"].created == created_before


def test_newchannel_usage(rig):
    # Bare /newchannel: the arity layer answers with the /help
    # pointer, exactly like /joinlink — nothing is created.
    created_before = list(rig["tg"].created)
    reply = _op(rig, "/newchannel", platform="telegram",
                chat_id=TG_OWNER, kind=ChatKind.DM)
    assert "needs 1 argument" in reply
    assert rig["tg"].created == created_before


def test_newchannel_refusal_verbatim_no_journal(rig):
    rig["tg"].create_error = "CHANNELS_TOO_MUCH"
    rows_before = len(_journal_rows(rig, rig["tg_rec"].id))
    try:
        reply = _op(rig, "/newchannel One Too Many",
                    platform="telegram", chat_id=TG_OWNER,
                    kind=ChatKind.DM)
    finally:
        rig["tg"].create_error = ""
    assert "CHANNELS_TOO_MUCH" in reply
    assert len(_journal_rows(rig, rig["tg_rec"].id)) == rows_before


def test_leave_tg_group(rig):
    reply = _op(rig, "/leave", platform="telegram",
                chat_id=TG_GROUP, kind=ChatKind.GROUP,
                title="Some Channel")
    assert rig["tg"].left == [TG_GROUP]
    assert reply == "left Some Channel."
    assert "chat_left" in _journal_kinds(rig, rig["tg_rec"].id)


def test_leave_tg_dm_is_honest(rig):
    left_before = list(rig["tg"].left)
    reply = _op(rig, "/leave", platform="telegram",
                chat_id=TG_OWNER, kind=ChatKind.DM)
    assert "isn't leavable" in reply
    assert rig["tg"].left == left_before


# ── fake-Telethon adapter pins ────────────────────────────────

class _RecordingRequest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def fake_telethon(monkeypatch):
    """A fake telethon module tree in the chanstories style:
    recording request classes, entity types, and the marked-id
    get_peer_id rule."""
    class Channel:
        pass

    class Chat:
        pass

    class PeerChannel:
        def __init__(self, channel_id):
            self.channel_id = channel_id

    class FakeChannel(Channel):
        def __init__(self, id, title=""):
            self.id = id
            self.title = title

    class FakeChat(Chat):
        def __init__(self, id, title=""):
            self.id = id
            self.title = title

    def get_peer_id(entity):
        if isinstance(entity, Channel):
            return int(f"-100{entity.id}")
        if isinstance(entity, Chat):
            return -int(entity.id)
        return int(entity.id)

    CreateChatRequest = type(
        "CreateChatRequest", (_RecordingRequest,), {})
    CreateChannelRequest = type(
        "CreateChannelRequest", (_RecordingRequest,), {})
    LeaveChannelRequest = type(
        "LeaveChannelRequest", (_RecordingRequest,), {})
    DeleteChatUserRequest = type(
        "DeleteChatUserRequest", (_RecordingRequest,), {})

    channels_mod = SimpleNamespace(
        CreateChannelRequest=CreateChannelRequest,
        LeaveChannelRequest=LeaveChannelRequest)
    messages_mod = SimpleNamespace(
        CreateChatRequest=CreateChatRequest,
        DeleteChatUserRequest=DeleteChatUserRequest)
    types_mod = SimpleNamespace(
        Channel=Channel, Chat=Chat, PeerChannel=PeerChannel)
    utils_mod = SimpleNamespace(get_peer_id=get_peer_id)
    monkeypatch.setitem(sys.modules, "telethon", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "telethon.tl",
                        SimpleNamespace())
    monkeypatch.setitem(sys.modules, "telethon.tl.functions",
                        SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "telethon.tl.functions.channels",
        channels_mod)
    monkeypatch.setitem(
        sys.modules, "telethon.tl.functions.messages",
        messages_mod)
    monkeypatch.setitem(sys.modules, "telethon.tl.types",
                        types_mod)
    monkeypatch.setitem(sys.modules, "telethon.utils", utils_mod)
    return SimpleNamespace(
        Channel=Channel, Chat=Chat, PeerChannel=PeerChannel,
        FakeChannel=FakeChannel, FakeChat=FakeChat,
        CreateChatRequest=CreateChatRequest,
        CreateChannelRequest=CreateChannelRequest,
        LeaveChannelRequest=LeaveChannelRequest,
        DeleteChatUserRequest=DeleteChatUserRequest)


class FakeTGClient:
    """The Telethon client stand-in: records requests, answers
    canned Updates, refuses raw-id input entities (so _resolve
    ends at its raw fallback, as a cold session would)."""

    def __init__(self):
        self.requests: list = []
        self.next_chats: list = []
        self.raise_with: Exception | None = None
        self.input_entity_calls: list = []

    async def __call__(self, request):
        self.requests.append(request)
        if self.raise_with is not None:
            raise self.raise_with
        return SimpleNamespace(chats=list(self.next_chats))

    async def get_input_entity(self, peer):
        self.input_entity_calls.append(peer)
        if isinstance(peer, int):
            raise ValueError("no input entity for a raw id")
        return ("input-entity", peer)

    async def get_entity(self, peer):
        return None


def _tg_adapter(tmp_path, client=None):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=12345, api_hash="fakehash",
        session_path=str(tmp_path / "tg.session"))
    if client is not None:
        adapter._client = client
        adapter._loop = object()  # only the None-check reads it

        def _drive(coro, timeout=60.0):
            return asyncio.run(coro)

        adapter._run_on_loop = _drive
    return adapter


def test_adapter_create_chat_request_shape(tmp_path, fake_telethon):
    client = FakeTGClient()
    client.next_chats = [fake_telethon.FakeChat(4242, "Think Tank")]
    adapter = _tg_adapter(tmp_path, client)
    verdict = adapter.create_chat("Think Tank")
    assert verdict["ok"] is True
    assert verdict["chat_id"] == "-4242"  # peer form: basic group
    assert verdict["title"] == "Think Tank"
    assert len(client.requests) == 1
    req = client.requests[0]
    assert isinstance(req, fake_telethon.CreateChatRequest)
    assert req.kwargs["users"] == []
    assert req.kwargs["title"] == "Think Tank"


def test_adapter_create_channel_is_not_a_megagroup(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    client.next_chats = [
        fake_telethon.FakeChannel(777001, "Devon Builds")]
    adapter = _tg_adapter(tmp_path, client)
    verdict = adapter.create_channel("Devon Builds")
    assert verdict["ok"] is True
    assert verdict["chat_id"] == "-100777001"  # peer form: channel
    req = client.requests[0]
    assert isinstance(req, fake_telethon.CreateChannelRequest)
    assert req.kwargs["megagroup"] is False
    assert req.kwargs["broadcast"] is True
    assert req.kwargs["title"] == "Devon Builds"


def test_adapter_create_channel_refusal_verbatim(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    client.raise_with = RuntimeError(
        "CHANNELS_TOO_MUCH (caused by CreateChannelRequest)")
    adapter = _tg_adapter(tmp_path, client)
    verdict = adapter.create_channel("One Too Many")
    assert verdict["ok"] is False
    assert "CHANNELS_TOO_MUCH" in verdict["error"]


def test_adapter_create_refuses_before_traffic(
        tmp_path, fake_telethon):
    adapter = _tg_adapter(tmp_path)  # never connected
    assert adapter.create_chat("X")["ok"] is False
    assert "not connected" in adapter.create_chat("X")["error"]
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    assert adapter.create_chat("   ")["ok"] is False
    assert adapter.create_channel("")["ok"] is False
    assert client.requests == []  # nothing reached the wire


def test_adapter_leave_channel_typed_entity(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    entity = fake_telethon.FakeChannel(777001, "Devon Builds")
    adapter._cache_input_entity("-100777001", entity)
    ref = ChatRef(platform="telegram", chat_id="-100777001",
                  kind=ChatKind.GROUP)
    verdict = adapter.leave_chat(ref)
    assert verdict["ok"] is True
    req = client.requests[0]
    assert isinstance(req, fake_telethon.LeaveChannelRequest)
    assert req.kwargs["channel"] is entity


def test_adapter_leave_basic_group_typed_entity(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    entity = fake_telethon.FakeChat(5150, "Old Group")
    adapter._cache_input_entity("-5150", entity)
    ref = ChatRef(platform="telegram", chat_id="-5150",
                  kind=ChatKind.GROUP)
    verdict = adapter.leave_chat(ref)
    assert verdict["ok"] is True
    req = client.requests[0]
    assert isinstance(req, fake_telethon.DeleteChatUserRequest)
    assert req.kwargs["chat_id"] == 5150  # raw positive id
    assert req.kwargs["user_id"] == "me"


def test_adapter_leave_channel_raw_id_fallback(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    ref = ChatRef(platform="telegram", chat_id="-100999888",
                  kind=ChatKind.GROUP)
    verdict = adapter.leave_chat(ref)
    assert verdict["ok"] is True
    req = client.requests[0]
    assert isinstance(req, fake_telethon.LeaveChannelRequest)
    peer = client.input_entity_calls[-1]
    assert isinstance(peer, fake_telethon.PeerChannel)
    assert peer.channel_id == 999888  # the -100 mark stripped


def test_adapter_leave_basic_group_raw_id_fallback(
        tmp_path, fake_telethon):
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    ref = ChatRef(platform="telegram", chat_id="-5150",
                  kind=ChatKind.GROUP)
    verdict = adapter.leave_chat(ref)
    assert verdict["ok"] is True
    req = client.requests[0]
    assert isinstance(req, fake_telethon.DeleteChatUserRequest)
    assert req.kwargs["chat_id"] == 5150
    assert req.kwargs["user_id"] == "me"


def test_adapter_leave_dm_refuses(tmp_path, fake_telethon):
    client = FakeTGClient()
    adapter = _tg_adapter(tmp_path, client)
    ref = ChatRef(platform="telegram", chat_id="777",
                  kind=ChatKind.DM)
    verdict = adapter.leave_chat(ref)
    assert verdict["ok"] is False
    assert "not leavable" in verdict["error"]
    assert client.requests == []


# ── catalog registration ──────────────────────────────────────

def test_catalog_registration():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    for kind in ("joinlink", "newgroup", "newchannel", "leave"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert kind not in PUBLIC_COMMANDS  # owner-only, always
    chat_privacy = dict(LIST_GROUPS)["chat & privacy — your "
                                     "chats · your accounts"]
    for kind in ("joinlink", "newgroup", "newchannel", "leave"):
        assert kind in chat_privacy
