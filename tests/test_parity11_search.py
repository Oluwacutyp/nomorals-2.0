"""PARITY #11 — SEARCH DEPTH (platform parity census
bundle #11, 2026-11-07).

The census graded three rows short of FULL: WhatsApp
in-chat search PARTIAL (a history fetch only — no search
verb in either bridge), WhatsApp cross-chat search ABSENT,
and Telegram-personal global search ABSENT (no
SearchGlobalRequest/SearchRequest anywhere).  This suite
pins the build at every seam:

  * bridges: a `search` verb on BOTH WhatsApp engines —
    a case-insensitive filter over the bridge's OWN
    remembered messages for the chat (whatsmeow/Baileys
    expose no server-side search to a linked device, so
    that memory is the honest reach, and the verdict's
    coverage line says so).  The Go filter is pinned by
    bridge/gobridge/search_test.go; here the dispatch,
    the Node payload and the rebuilt binaries are pinned.
  * WhatsApp adapter: search_messages rides the verb;
    search_global is the adapter fan-out over chats()
    (per-chat cap, merged newest-first, totals summed,
    every-chat-refused reads as a refusal).
  * Telegram personal: messages.Search for one chat and
    messages.SearchGlobal across all of them — Telegram
    searches server-side, so the coverage line names the
    real archive.  Peers resolve against the result's own
    chats/users; a global hit that cannot name its
    conversation is skipped, never printed as an id.
  * Telegram bot: no override — the Bot API has no
    message-search endpoint, and both the adapter verdict
    and the /find reply say exactly that.
  * /find + /findall: the composed chain (owner command →
    runtime → identity home → real WhatsAppAdapter →
    recording fake bridge).  Both are READS, and this
    family's standing doctrine is that reads are not acts
    as the owner (the identity home's own contract), so
    the standing stop — which gates acts like /sendloc —
    does not halt them; a pin below keeps that deliberate.
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

PORT = 19459
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
ROOT = Path(__file__).resolve().parents[1]

# The fake bridge's remembered messages, per chat — the
# store its `search` filters, in the bridges' row shape.
SEED = {
    PN_POSTER: [
        {"id": "m1", "incoming": True, "text": "morning!",
         "sender": "Poster", "media": [], "reply_to": "",
         "ts": 1762387200000},
        {"id": "m2", "incoming": True,
         "text": "the invoice for october is ready",
         "sender": "Poster", "media": [], "reply_to": "",
         "ts": 1762390800000},
        {"id": "m3", "incoming": False, "text": "got it, thanks",
         "sender": "", "media": [], "reply_to": "",
         "ts": 1762394400000},
        {"id": "m4", "incoming": True,
         "text": "INVOICE paid off today",
         "sender": "Poster", "media": [], "reply_to": "",
         "ts": 1762398000000},
    ],
    GROUP: [
        {"id": "g1", "incoming": True,
         "text": "who has the invoice template?",
         "sender": "Ada", "media": [], "reply_to": "",
         "ts": 1762401600000},
        {"id": "g2", "incoming": True, "text": "match on saturday",
         "sender": "Bola", "media": [], "reply_to": "",
         "ts": 1762405200000},
    ],
}
TITLES = {PN_POSTER: "Poster Pal", GROUP: "Test Group"}


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
                if verb == "search":
                    if srv.search_error:
                        reply = {"id": cid, "ok": False,
                                 "error": srv.search_error}
                    else:
                        rows = srv.seed.get(cmd.get("chat")
                                            or "", [])
                        needle = str(
                            cmd.get("query") or "").lower()
                        hits = [r for r in rows
                                if needle in str(
                                    r.get("text")
                                    or "").lower()]
                        limit = max(1, int(
                            cmd.get("limit") or 20))
                        page = hits[-limit:]
                        reply = {"id": cid, "ok": True,
                                 "messages": page,
                                 "total": len(hits)}
                elif verb == "chats":
                    chats = [{
                        "id": jid, "title": TITLES.get(jid, ""),
                        "kind": ("group" if jid.endswith("@g.us")
                                 else "dm"),
                        "ts": max(r["ts"] for r in rows),
                    } for jid, rows in srv.seed.items()]
                    reply = {"id": cid, "ok": True,
                             "chats": chats}
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
        self.search_error = ""
        self.seed = {k: [dict(r) for r in v]
                     for k, v in SEED.items()}
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

    tmp = tmp_path_factory.mktemp("parity11")
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
    env = {"runtime": runtime, "bridge": bridge,
           "adapter": adapter, "sup": sup, "rec": rec,
           "tmp": tmp, "beat": beat}
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


# ── bridge source pins (both engines + the binaries) ───────

def test_go_bridge_carries_the_search_verb():
    dispatch = (ROOT / "bridge" / "gobridge"
                / "messages.go").read_text()
    assert 'case "search"' in dispatch
    assert "func (b *bridge) cmdSearch" in dispatch
    assert "searchRemembered" in dispatch
    # the refusal mirrors the Python shape rule, in words
    assert "a search needs the words to look for" in dispatch


def test_baileys_bridge_carries_the_search_verb():
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs").read_text()
    assert "case 'search'" in src
    assert "a search needs the words to look for" in src
    # the filter reads the same memory `history` serves
    assert "recentByChat.get" in src
    assert "total: hits.length" in src


def test_rebuilt_binaries_carry_the_verb():
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"a search needs the words to look for" in data, name


def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    for kind in ("find", "findall"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
        # owner-only account reads — never public commands
        assert kind not in PUBLIC_COMMANDS


# ── the formatter (pure) ───────────────────────────────────

def _hit(chat_id="999888", title="Some Chat", sender="Ada",
         text="the invoice is ready", ts=1762390800.0,
         incoming=True, mid="WAMID-SECRET-1"):
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_id,
                     kind=ChatKind.GROUP, title=title),
        incoming=incoming, text=text, sender=sender,
        message_id=mid, ts=ts)


def test_format_search_results_shapes():
    from nomorals.social.chat.chatops import format_search_results

    verdict = {"ok": True, "query": "invoice", "total": 1,
               "coverage": "a test reach",
               "messages": [_hit()]}
    out = format_search_results(verdict, scope="this chat")
    assert "1 message match “invoice” in this chat:" in out
    assert "Ada" in out and "the invoice is ready" in out
    assert "(searched a test reach)" in out
    # ids never print: not the message id, not the chat id
    assert "WAMID-SECRET-1" not in out
    assert "999888" not in out


def test_format_search_results_capped_and_empty():
    from nomorals.social.chat.chatops import format_search_results

    verdict = {"ok": True, "query": "invoice", "total": 57,
               "coverage": "a test reach",
               "messages": [_hit(), _hit(mid="M2")]}
    out = format_search_results(verdict)
    assert "57 messages match “invoice”" in out
    assert "showing the 2 most recent" in out
    out = format_search_results(
        {"ok": True, "query": "zzz", "total": 0,
         "messages": [], "coverage": "a test reach"},
        scope="this chat")
    assert "no messages match “zzz” in this chat." in out
    out = format_search_results(
        {"ok": False, "error": "bridge is not logged in"})
    assert out == ("the search didn't run — "
                   "bridge is not logged in")
    # an own message reads as "you", never a blank sender
    own = format_search_results(
        {"ok": True, "query": "invoice", "total": 1,
         "messages": [_hit(incoming=False, sender="")]})
    assert "· you —" in own


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
    chat = ChatRef(platform="bare", chat_id="1")
    for verdict in (ad.search_messages(chat, "invoice"),
                    ad.search_global("invoice")):
        assert verdict["ok"] is False
        assert "not supported" in verdict["error"]


# ── adapter pins: WhatsApp ─────────────────────────────────

@pytest.fixture()
def wa_adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19996)
    ad.connected.set()
    calls: list[dict] = []
    pages = {
        PN_POSTER: {"ok": True, "total": 2, "messages": [
            {"id": "m2", "incoming": True,
             "text": "the invoice for october is ready",
             "sender": "Poster", "media": [], "reply_to": "",
             "ts": 1762390800000},
            {"id": "m4", "incoming": True,
             "text": "INVOICE paid off today",
             "sender": "Poster", "media": [], "reply_to": "",
             "ts": 1762398000000}]},
        GROUP: {"ok": True, "total": 1, "messages": [
            {"id": "g1", "incoming": True,
             "text": "who has the invoice template?",
             "sender": "Ada", "media": [], "reply_to": "",
             "ts": 1762401600000}]},
    }

    def fake_send_cmd(payload, timeout=15):
        calls.append(payload)
        if payload.get("cmd") == "search":
            return pages.get(payload.get("chat") or "",
                             {"ok": True, "total": 0,
                              "messages": []})
        if payload.get("cmd") == "chats":
            return {"ok": True, "chats": [
                {"id": PN_POSTER, "title": "Poster Pal",
                 "kind": "dm", "ts": 1762398000000},
                {"id": GROUP, "title": "Test Group",
                 "kind": "group", "ts": 1762401600000}]}
        return {"ok": True}

    monkeypatch.setattr(ad, "_send_cmd", fake_send_cmd)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _wa_chat(chat_id=PN_POSTER, kind=ChatKind.DM):
    return ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=kind, peer=chat_id)


def test_wa_search_messages_payload_and_order(wa_adapter):
    verdict = wa_adapter.search_messages(_wa_chat(), "invoice")
    assert wa_adapter.calls == [{
        "cmd": "search", "chat": PN_POSTER,
        "query": "invoice", "limit": 20}]
    assert verdict["ok"] and verdict["total"] == 2
    # bridge answers oldest-first; the verdict is newest-first
    assert [m.message_id for m in verdict["messages"]] == [
        "m4", "m2"]
    assert verdict["messages"][0].text == (
        "INVOICE paid off today")
    assert "bridge has seen this session" in verdict["coverage"]


def test_wa_search_refusals_before_traffic(wa_adapter):
    verdict = wa_adapter.search_messages(_wa_chat(), "   ")
    assert verdict["ok"] is False
    assert "words to look for" in verdict["error"]
    verdict = wa_adapter.search_global("")
    assert verdict["ok"] is False
    assert wa_adapter.calls == []  # nothing reached the bridge


def test_wa_search_disconnected_and_bridge_refusal(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19995)
    verdict = ad.search_messages(_wa_chat(), "invoice")
    assert verdict["ok"] is False
    assert verdict["error"] == "whatsapp bridge not connected"
    ad.connected.set()
    monkeypatch.setattr(
        ad, "_send_cmd",
        lambda payload, timeout=15: {
            "ok": False, "error": "bridge is not logged in"})
    verdict = ad.search_messages(_wa_chat(), "invoice")
    assert verdict["ok"] is False
    assert verdict["error"] == "bridge is not logged in"


def test_wa_search_global_fans_out(wa_adapter):
    verdict = wa_adapter.search_global("invoice", limit=20)
    assert verdict["ok"] and verdict["total"] == 3
    searches = [c for c in wa_adapter.calls
                if c.get("cmd") == "search"]
    assert len(searches) == 2
    # the fan-out's per-chat cap is what keeps one loud
    # room from crowding out the rest
    assert all(c["limit"] == 5 for c in searches)
    assert [m.message_id for m in verdict["messages"]] == [
        "g1", "m4", "m2"]  # merged newest-first
    group_hit = verdict["messages"][0]
    assert group_hit.chat.title == "Test Group"
    assert group_hit.chat.chat_id == GROUP
    assert "2 of 2 chats" in verdict["coverage"]


def test_wa_search_global_all_refused(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19994)
    ad.connected.set()

    def fake_send_cmd(payload, timeout=15):
        if payload.get("cmd") == "chats":
            return {"ok": True, "chats": [
                {"id": PN_POSTER, "title": "Poster Pal",
                 "kind": "dm", "ts": 1}]}
        return {"ok": False, "error": "unknown cmd: search"}

    monkeypatch.setattr(ad, "_send_cmd", fake_send_cmd)
    verdict = ad.search_global("invoice")
    assert verdict["ok"] is False
    assert verdict["error"] == "unknown cmd: search"


# ── adapter pins: Telegram personal (Telethon) ─────────────

class _FakeTLType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture()
def tg_personal(monkeypatch, tmp_path):
    """A TelegramAdapter with a recording fake client and a
    fake telethon module tree — the TL request shapes and
    the result mapping are asserted field by field."""
    from nomorals.social.chat.telegram import TelegramAdapter

    types_mod = _types.ModuleType("telethon.tl.types")
    for cls_name in ("InputMessagesFilterEmpty",
                     "InputPeerEmpty"):
        setattr(types_mod, cls_name,
                type(cls_name, (_FakeTLType,), {}))
    messages_mod = _types.ModuleType(
        "telethon.tl.functions.messages")
    for cls_name in ("SearchRequest", "SearchGlobalRequest"):
        setattr(messages_mod, cls_name,
                type(cls_name, (_FakeTLType,), {}))
    utils_mod = _types.ModuleType("telethon.utils")

    def _get_peer_id(obj):
        marked = getattr(obj, "marked_id", None)
        if marked is not None:
            return marked
        return int(getattr(obj, "id", 0) or 0)

    utils_mod.get_peer_id = _get_peer_id
    for name, mod in (
            ("telethon", _types.ModuleType("telethon")),
            ("telethon.tl", _types.ModuleType("telethon.tl")),
            ("telethon.tl.types", types_mod),
            ("telethon.tl.functions",
             _types.ModuleType("telethon.tl.functions")),
            ("telethon.tl.functions.messages", messages_mod),
            ("telethon.utils", utils_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    requests: list = []

    user = _types.SimpleNamespace(
        id=424242, marked_id=424242, first_name="Ada",
        last_name="Lovelace", username="ada", title=None)
    channel = _types.SimpleNamespace(
        id=1234567890, marked_id=-1001234567890,
        title="Signal Room", first_name=None,
        last_name=None, username=None)
    class _Date:
        """A Telethon message date stand-in: the adapter
        only ever calls .timestamp() on it."""

        @staticmethod
        def timestamp():
            return 1762390800.0

    when = _Date()

    def _msg(mid, peer_marked, text, *, out=False, sender=None):
        return _types.SimpleNamespace(
            id=mid, message=text, out=out, date=when,
            peer_id=_types.SimpleNamespace(
                marked_id=peer_marked),
            from_id=(_types.SimpleNamespace(
                marked_id=sender) if sender else None))

    chat_result = _types.SimpleNamespace(
        count=1, users=[user], chats=[],
        messages=[_msg(11, 424242,
                       "the invoice is ready",
                       sender=424242)])
    global_result = _types.SimpleNamespace(
        count=9, users=[user], chats=[channel],
        messages=[
            _msg(21, -1001234567890,
                 "invoice totals are up", sender=424242),
            _msg(22, 424242, "my own invoice note",
                 out=True),
            _msg(23, -1009999999999, "an unnameable hit"),
        ])

    async def _client(request):
        requests.append(request)
        if type(request).__name__ == "SearchGlobalRequest":
            return global_result
        return chat_result

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


def test_userbot_search_request_shape(tg_personal):
    verdict = tg_personal.search_messages(
        _userbot_chat(), "invoice")
    (req,) = tg_personal.requests
    assert type(req).__name__ == "SearchRequest"
    assert req.peer == "ENTITY-1"
    assert req.q == "invoice" and req.limit == 20
    assert type(req.filter).__name__ == (
        "InputMessagesFilterEmpty")
    assert verdict["ok"] and verdict["total"] == 1
    (hit,) = verdict["messages"]
    assert hit.text == "the invoice is ready"
    assert hit.sender == "Ada Lovelace"
    assert hit.chat.title == "Ada Lovelace"  # the DM itself
    assert "real archive" in verdict["coverage"]


def test_userbot_search_global_shape_and_mapping(tg_personal):
    verdict = tg_personal.search_global("invoice")
    (req,) = tg_personal.requests
    assert type(req).__name__ == "SearchGlobalRequest"
    assert req.q == "invoice" and req.offset_rate == 0
    assert type(req.offset_peer).__name__ == "InputPeerEmpty"
    assert verdict["ok"] and verdict["total"] == 9
    # the hit whose peer no list names is SKIPPED, never
    # printed as a raw id — 3 arrived, 2 are nameable
    assert len(verdict["messages"]) == 2
    first, second = verdict["messages"]
    assert first.chat.title == "Signal Room"
    assert first.chat.chat_id == "-1001234567890"
    assert first.sender == "Ada Lovelace"
    assert second.chat.title == "Ada Lovelace"
    assert second.incoming is False


def test_userbot_search_disconnected_and_shape(monkeypatch,
                                                tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    verdict = ad.search_messages(_userbot_chat(), "invoice")
    assert verdict["ok"] is False
    assert verdict["error"] == "not connected"
    verdict = ad.search_global("invoice")
    assert verdict["ok"] is False
    ad._client = object()  # type: ignore[attr-defined]
    verdict = ad.search_messages(_userbot_chat(), "  ")
    assert verdict["ok"] is False
    assert "words to look for" in verdict["error"]


def test_bot_adapter_search_is_an_honest_absence():
    from nomorals.social.chat.telegram import TelegramBotAdapter

    ad = TelegramBotAdapter(
        token="123456:TEST", chat_allow="",
        media_dir="data/media/telegram-bot-test")
    chat = ChatRef(platform="telegram-bot",
                   chat_id="-1001234567890",
                   kind=ChatKind.GROUP, peer="-1001234567890")
    for verdict in (ad.search_messages(chat, "invoice"),
                    ad.search_global("invoice")):
        assert verdict["ok"] is False
        assert "not supported" in verdict["error"]


# ── chatop: the bot-surface branch ─────────────────────────

def test_chatop_bot_surface_answers_platform_truth():
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )

    mixin = RuntimeChatOpsMixin()
    consulted: list = []
    mixin._adapter_for = (  # type: ignore[method-assign]
        lambda platform: consulted.append(platform))
    chat = ChatRef(platform="telegram-bot",
                   chat_id="-1001234567890",
                   kind=ChatKind.GROUP,
                   peer="-1001234567890")
    msg = ChatMessage(chat=chat, incoming=True, text="x",
                      sender="Owner", sender_id="1")
    out = mixin._chatops_find("find", "invoice", msg)
    assert "Bot API has no message-search" in out
    assert consulted == []  # answered before any adapter work
    out = mixin._chatops_find("findall", "invoice", msg)
    assert "Bot API has no message-search" in out


# ── composed chain: /find + /findall on WhatsApp ───────────

def test_find_in_chat_lists_matches_newest_first(rig):
    out = _op(rig, "/find invoice")
    assert "2 messages match “invoice” in this chat:" in out
    assert "INVOICE paid off today" in out
    assert "the invoice for october is ready" in out
    # newest first in the rendered list
    assert out.index("INVOICE paid off today") < out.index(
        "the invoice for october is ready")
    assert "bridge has seen this session" in out
    cmds = rig["bridge"].cmds("search")
    assert cmds and cmds[-1]["chat"] == PN_POSTER
    assert cmds[-1]["query"] == "invoice"


def test_find_no_match_and_no_words(rig):
    out = _op(rig, "/find zzz-no-hit")
    assert "no messages match “zzz-no-hit” in this chat." in out
    before = len(rig["bridge"].cmds("search"))
    # no words at all: the catalog's own arg gate refuses
    # before the handler — an honest usage refusal either way
    out = _op(rig, "/find")
    assert "needs 1 argument(s)" in out
    out = _op(rig, "/findall")
    assert "needs 1 argument(s)" in out
    assert len(rig["bridge"].cmds("search")) == before


def test_find_bridge_refusal_is_named(rig):
    rig["bridge"].search_error = "bridge is not logged in"
    try:
        out = _op(rig, "/find invoice")
    finally:
        rig["bridge"].search_error = ""
    assert "the search didn't run" in out
    assert "bridge is not logged in" in out


def test_findall_names_each_conversation(rig):
    out = _op(rig, "/findall invoice")
    assert "3 messages match “invoice”" in out
    assert "every chat on this surface" in out
    assert "Test Group" in out  # the group hit names its chat
    assert "who has the invoice template?" in out
    assert "INVOICE paid off today" in out
    # merged newest-first across chats: the group hit is newest
    assert out.index("who has the invoice template?") < (
        out.index("INVOICE paid off today"))
    assert rig["bridge"].cmds("chats")
    searched = {c["chat"] for c in rig["bridge"].cmds("search")}
    assert searched == {PN_POSTER, GROUP}


def test_find_is_a_read_not_an_act_under_owner_stop(rig):
    # The chatops family's standing doctrine (the identity
    # home's contract): the owner's standing stop gates
    # ACTS as him — sends, posts, identity changes — not
    # reads of his own messages.  /find and /findall are
    # reads, so they answer under a stop, in deliberate
    # contrast to /sendloc's pin in the #9 suite.
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        out = _op(rig, "/find invoice")
        assert "2 messages match “invoice”" in out
        out = _op(rig, "/findall invoice")
        assert "3 messages match “invoice”" in out
    finally:
        rig["sup"].registry.set_owner_stopped(
            "whatsapp", False)
