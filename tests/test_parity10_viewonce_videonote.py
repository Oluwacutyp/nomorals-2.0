"""PARITY #10 — VIEW-ONCE + VIDEO-NOTE SENDING (platform
parity census bundle #10, 2026-11-07).

The census graded four rows ABSENT:

  * WA send view-once media — send_media carried no viewOnce
    flag (inbound capture was already FULL; send side missing)
  * WA send video note — no ptv/videoNote handling in either
    bridge (0 send-side hits)
  * TG-bot send video note — no sendVideoNote in the kind map
  * TG-personal send video note — send_media never passed a
    video-note form to Telethon's send_file

The build, per surface:

  * WhatsApp: MediaRef.view_once + kind "video_note" ride
    send_media through the adapter (validated BEFORE queueing
    or wire traffic — a form the media cannot carry refuses
    honestly, and a queued send keeps its form through the
    outbox) into both bridges: Go sets the ViewOnce flag on
    the media message and wraps a note as PtvMessage (pinned
    at the proto level in bridge/gobridge/mediaforms_test.go);
    Baileys gets viewOnce/ptv content flags, which its
    generateWAMessage turns into the viewOnceMessage wrap /
    ptvMessage form (Baileys 6.7.24 lib/Utils/messages.js).
  * TG bot: kind "video_note" → sendVideoNote; a caption
    follows as its own text (the verb takes none). View-once
    refuses by name — Telegram bots have no such media.
  * TG personal: kind "video_note" → Telethon send_file
    video_note=True (the round-message document attribute).
    View-once refuses by name — MTProto has none outside
    secret chats.
  * The gateway's send_file takes view_once / kind=
    "video_note" and REFUSES rather than downgrade the form
    away over the inline cap; the file_send tool and the
    /sendonce + /sendnote chatops (owner-gated, journaled)
    reach the same pipeline.

A capability counts only when the form is on the WIRE: the
adapter payloads are pinned through recording fakes, the
chatops are driven end-to-end (owner command → runtime →
adapter → recording fake bridge), and the rebuilt Go
binaries must carry the new refusal strings.
"""
from __future__ import annotations

import asyncio
import json
import socketserver
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, MediaRef, SendResult

ROOT = Path(__file__).resolve().parents[1]
PORT = 19437
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
TG_OWNER = "777"


# ── bridge source + binary pins ────────────────────────────

def test_go_bridge_carries_the_forms():
    src = (ROOT / "bridge" / "gobridge"
           / "messages.go").read_text()
    assert 'req.boolean("view_once")' in src
    assert 'req.boolean("ptv")' in src
    assert "specialFormRefusal" in src
    assert "applyViewOnce" in src
    assert "asVideoNote" in src
    assert "PtvMessage" in src
    # the flag lands on the media messages themselves
    assert "msg.ImageMessage.ViewOnce = proto.Bool(true)" in src
    assert "msg.VideoMessage.ViewOnce = proto.Bool(true)" in src


def test_rebuilt_binaries_carry_the_forms():
    # The refusal strings are unique to this build's Go code
    # and survive in the binaries' strings (G4 precedent).
    for name in ("whatsapp-bridge-linux", "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"view-once applies to photos and videos only" in data, name
        assert b"a video note (ptv) needs a video file" in data, name


def test_baileys_bridge_carries_the_forms():
    src = (ROOT / "bridge" / "whatsapp-bridge.mjs").read_text()
    start = src.index("async function cmdSendMedia")
    region = src[start:src.index("function lastMessagesFor", start)]
    assert "view_once" in region and "ptv" in region
    # Baileys' own content flags — its generateWAMessage wraps
    # viewOnce into viewOnceMessage and ptv into ptvMessage
    assert "content.viewOnce = true" in region
    assert "content.ptv = true" in region
    assert "view-once applies to photos and videos only" in region
    assert "a video note (ptv) needs a video file" in region


# ── WhatsApp adapter payloads (recording fake) ─────────────

@pytest.fixture()
def wa_adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19998)
    ad.connected.set()
    calls: list[dict] = []

    def fake_cmd(cmd, *, timeout=25.0):
        calls.append(cmd)
        return {"ok": True, "message_id": "WAMID-P10-1"}

    monkeypatch.setattr(ad, "_send_cmd", fake_cmd)
    ad.calls = calls  # type: ignore[attr-defined]
    return ad


def _wa_chat():
    return ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)


def test_wa_view_once_photo_payload(wa_adapter, tmp_path):
    pic = tmp_path / "pic.jpg"
    pic.write_bytes(b"\xff\xd8\xff")
    res = wa_adapter.send_media(
        _wa_chat(), MediaRef(path=str(pic), kind="image",
                             view_once=True),
        caption="one look")
    assert res.ok, res.error
    cmd = wa_adapter.calls[-1]
    assert cmd["cmd"] == "send_media"
    assert cmd["view_once"] is True
    assert "ptv" not in cmd


def test_wa_view_once_video_payload(wa_adapter, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    res = wa_adapter.send_view_once(_wa_chat(), str(clip),
                                    kind="video")
    assert res.ok, res.error
    assert wa_adapter.calls[-1]["view_once"] is True


def test_wa_video_note_payload(wa_adapter, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    res = wa_adapter.send_video_note(_wa_chat(), str(clip))
    assert res.ok, res.error
    cmd = wa_adapter.calls[-1]
    assert cmd["cmd"] == "send_media"
    assert cmd["ptv"] is True
    assert cmd["media_kind"] == "video_note"
    assert "view_once" not in cmd


def test_wa_plain_send_carries_no_form_flags(wa_adapter, tmp_path):
    pic = tmp_path / "pic.jpg"
    pic.write_bytes(b"\xff\xd8\xff")
    res = wa_adapter.send_media(
        _wa_chat(), MediaRef(path=str(pic), kind="image"))
    assert res.ok, res.error
    cmd = wa_adapter.calls[-1]
    assert "view_once" not in cmd
    assert "ptv" not in cmd


def test_wa_view_once_refuses_non_visual_media(wa_adapter, tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"ID3")
    res = wa_adapter.send_media(
        _wa_chat(), MediaRef(path=str(song), kind="audio",
                             view_once=True))
    assert res.ok is False
    assert "view-once applies to photos and videos" in res.error
    assert wa_adapter.calls == []  # refused before the wire


def test_wa_view_once_refuses_a_video_note(wa_adapter, tmp_path):
    # A round note is re-openable by design on the receiving
    # client; stamping it view-once would be a false label.
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    res = wa_adapter.send_media(
        _wa_chat(), MediaRef(path=str(clip), kind="video_note",
                             view_once=True))
    assert res.ok is False
    assert "view-once applies to photos and videos" in res.error
    assert wa_adapter.calls == []


def test_wa_queued_send_keeps_its_form(wa_adapter, tmp_path):
    # Bridge down: the send queues in the outbox WITH its
    # form, and the reconnect delivery still carries it.
    pic = tmp_path / "pic.jpg"
    pic.write_bytes(b"\xff\xd8\xff")
    wa_adapter.outbox.path = str(tmp_path / "outbox.jsonl")
    wa_adapter.connected.clear()
    res = wa_adapter.send_media(
        _wa_chat(), MediaRef(path=str(pic), kind="image",
                             view_once=True))
    assert res.ok is False
    assert "queued for delivery" in res.error
    rows = [json.loads(line)
            for line in Path(wa_adapter.outbox.path)
            .read_text().splitlines() if line.strip()]
    media_rows = [r for r in rows if r.get("kind") == "media"]
    assert len(media_rows) == 1
    assert media_rows[0]["view_once"] is True
    wa_adapter.connected.set()
    assert wa_adapter._deliver_entry(media_rows[0]) is True
    assert wa_adapter.calls[-1]["view_once"] is True


# ── gateway form discipline ────────────────────────────────

class _StubAdapter:
    name = "whatsapp"

    def __init__(self):
        self.media: list[MediaRef] = []

    def send_media(self, chat, media, *, caption="",
                   parse_mode="", buttons=None):
        self.media.append(media)
        return SendResult(ok=True, platform=self.name,
                          message_id="stub-1")


@pytest.fixture()
def gateway_rig():
    from nomorals.social.chat.gateway import ChatGateway

    stub = _StubAdapter()
    gw = ChatGateway({"whatsapp": stub}, db=None)
    return gw, stub


def _chat():
    return ChatRef(platform="whatsapp", chat_id="12345",
                   kind=ChatKind.DM)


def test_gateway_view_once_rides_the_ref(gateway_rig, tmp_path):
    gw, stub = gateway_rig
    pic = tmp_path / "pic.jpg"
    pic.write_bytes(b"\xff\xd8\xff" * 10)
    res = gw.send_file("whatsapp", _chat(), str(pic),
                       kind="image", view_once=True)
    assert res.ok, res.error
    assert stub.media[-1].view_once is True
    assert stub.media[-1].kind == "image"


def test_gateway_video_note_kind_rides(gateway_rig, tmp_path):
    gw, stub = gateway_rig
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp" * 10)
    res = gw.send_file("whatsapp", _chat(), str(clip),
                       kind="video_note")
    assert res.ok, res.error
    assert stub.media[-1].kind == "video_note"


def test_gateway_view_once_refuses_a_document(gateway_rig,
                                               tmp_path):
    gw, stub = gateway_rig
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00" * 64)
    res = gw.send_file("whatsapp", _chat(), str(blob),
                       view_once=True)
    assert res.ok is False
    assert "view-once" in res.error
    assert stub.media == []  # never reached the adapter


def test_gateway_video_note_refuses_the_downgrade(gateway_rig,
                                                  tmp_path):
    # Over the inline cap the kind would resolve to
    # "document": the note form must refuse, not downgrade.
    from nomorals.social.chat.platform_limits import for_platform

    cap = for_platform("whatsapp").max_file_mb
    big = tmp_path / "big.mp4"
    with open(big, "wb") as fh:
        fh.truncate(int(cap * 1024 * 1024) + 1024)
    gw, stub = gateway_rig
    res = gw.send_file("whatsapp", _chat(), str(big),
                       kind="video_note")
    assert res.ok is False
    assert "video note" in res.error
    assert stub.media == []


# ── Telegram bot: sendVideoNote + honest view-once ─────────

class _FakeBotSession:
    """The bot adapter's session contract (pass41 pattern)."""

    def __init__(self):
        self.posts: list[dict] = []

    def post_multipart(self, url, fields=None, files=None,
                       timeout=None):
        self.posts.append({
            "url": url, "data": fields,
            "files": {e[0]: e for e in (files or [])}})
        return SimpleNamespace(
            json=lambda: {"ok": True,
                          "result": {"message_id": 9}})

    def post_json(self, url, params, timeout=None):
        self.posts.append({"url": url, "data": params,
                           "files": {}})
        return SimpleNamespace(
            json=lambda: {"ok": True,
                          "result": {"message_id": 9}})

    def get(self, url, timeout=None):
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": {}})


@pytest.fixture()
def bot_rig(tmp_path):
    from nomorals.social.chat.telegram import TelegramBotAdapter

    session = _FakeBotSession()
    adapter = TelegramBotAdapter(token="123456:secret",
                                 session=session)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    chat = ChatRef(platform="telegram-bot", chat_id="42",
                   kind=ChatKind.DM)
    return adapter, session, clip, chat


def test_bot_video_note_uses_sendvideonote(bot_rig):
    adapter, session, clip, chat = bot_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video_note"))
    assert res.ok, res.error
    assert session.posts[0]["url"].endswith("/sendVideoNote")
    assert "video_note" in session.posts[0]["files"]
    assert "caption" not in (session.posts[0]["data"] or {})


def test_bot_video_note_caption_follows_as_text(bot_rig):
    adapter, session, clip, chat = bot_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video_note"),
        caption="quick update")
    assert res.ok, res.error
    assert session.posts[0]["url"].endswith("/sendVideoNote")
    assert session.posts[1]["url"].endswith("/sendMessage")
    assert session.posts[1]["data"]["text"] == "quick update"


def test_bot_view_once_refuses_honestly(bot_rig):
    adapter, session, clip, chat = bot_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video",
                       view_once=True))
    assert res.ok is False
    assert "view-once" in res.error
    assert session.posts == []


def test_bot_plain_video_still_sendvideo(bot_rig):
    adapter, session, clip, chat = bot_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video"))
    assert res.ok, res.error
    assert session.posts[0]["url"].endswith("/sendVideo")


# ── Telegram personal: Telethon video-note form ────────────

def _run_sync(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeTelethonClient:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_file(self, entity, file, caption=None,
                        **kwargs):
        self.sent.append({"entity": entity, "file": file,
                          "caption": caption, "kwargs": kwargs})
        return SimpleNamespace(id=77)


@pytest.fixture()
def tg_rig(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    adapter._run_on_loop = lambda coro, timeout=30.0: \
        _run_sync(coro)
    client = _FakeTelethonClient()
    adapter._client = client
    chat = ChatRef(platform="telegram", chat_id="4242",
                   kind=ChatKind.DM)
    adapter._input_entity_cache.set(chat.chat_id, ("peer", "4242"))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    return adapter, client, clip, chat


def test_tg_personal_video_note_form(tg_rig):
    adapter, client, clip, chat = tg_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video_note"))
    assert res.ok, res.error
    assert client.sent[-1]["kwargs"].get("video_note") is True


def test_tg_personal_plain_video_has_no_note_form(tg_rig):
    adapter, client, clip, chat = tg_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video"))
    assert res.ok, res.error
    assert "video_note" not in client.sent[-1]["kwargs"]


def test_tg_personal_view_once_refuses_honestly(tg_rig):
    adapter, client, clip, chat = tg_rig
    res = adapter.send_media(
        chat, MediaRef(path=str(clip), kind="video",
                       view_once=True))
    assert res.ok is False
    assert "view-once" in res.error
    assert client.sent == []


def test_tg_personal_voice_note_form_intact(tg_rig, tmp_path):
    adapter, client, _clip, chat = tg_rig
    note = tmp_path / "note.ogg"
    note.write_bytes(b"OggS")
    res = adapter.send_media(
        chat, MediaRef(path=str(note), kind="voice",
                       voice_note=True))
    assert res.ok, res.error
    assert client.sent[-1]["kwargs"].get("voice_note") is True
    assert "video_note" not in client.sent[-1]["kwargs"]


# ── chatops end-to-end (the G1 rig pattern) ────────────────

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
                if cmd.get("cmd") in ("send", "send_media"):
                    reply["message_id"] = "WAMID-P10-1"
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
    """The personal Telegram seat for the chatops rig: only
    its presence matters — /sendonce must refuse on platform
    truth before ever touching it."""

    name = "telegram"

    def __init__(self):
        self.media_calls: list = []

    def start(self, handler):
        self._handler = handler
        return True

    def stop(self):
        pass

    def send_media(self, chat, media, *, caption="",
                   parse_mode=""):
        self.media_calls.append((chat, media))
        return SendResult(ok=True, platform=self.name,
                          message_id="tg-1")


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parity10")
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
    pic = tmp / "photo.jpg"
    pic.write_bytes(b"\xff\xd8\xff" * 8)
    clip = tmp / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp" * 8)

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
        "tg": tg, "tmp": tmp, "pic": pic, "clip": clip,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        media=None, meta_extra=None, sender_id=OWNER):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.DM, title="", peer=chat_id)
    meta = {"owner_command": True}
    if meta_extra:
        meta.update(meta_extra)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=f"op-{time.time()}",
        media=media or [], ts=time.time(), meta=meta)
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_kinds(rig, account_id):
    return [r.get("kind")
            for r in rig["vault"].journal(account_id)]


def test_sendonce_end_to_end(rig):
    media = [MediaRef(path=str(rig["pic"]), kind="image",
                      mime="image/jpeg")]
    reply = _op(rig, "/sendonce for your eyes only", media=media)
    assert "view-once" in reply
    cmds = rig["bridge"].cmds("send_media")
    assert cmds, "no send_media reached the bridge"
    assert cmds[-1]["view_once"] is True
    assert cmds[-1]["caption"] == "for your eyes only"
    assert "message_viewonce_sent" in _journal_kinds(
        rig, rig["rec"].id)


def test_sendnote_end_to_end(rig):
    media = [MediaRef(path=str(rig["clip"]), kind="video",
                      mime="video/mp4")]
    reply = _op(rig, "/sendnote", media=media)
    assert "video note" in reply
    cmds = rig["bridge"].cmds("send_media")
    assert cmds[-1]["ptv"] is True
    assert cmds[-1]["media_kind"] == "video_note"
    assert "message_videonote_sent" in _journal_kinds(
        rig, rig["rec"].id)


def test_sendonce_reply_form(rig):
    # The media rides the message the command REPLIES to.
    meta = {"replied_media_local": {
        "path": str(rig["pic"]), "kind": "image"}}
    reply = _op(rig, "/sendonce", meta_extra=meta)
    assert "view-once" in reply
    assert rig["bridge"].cmds("send_media")[-1]["view_once"] \
        is True


def test_sendonce_needs_media(rig):
    before = len(rig["bridge"].cmds("send_media"))
    reply = _op(rig, "/sendonce")
    assert "usage" in reply
    assert len(rig["bridge"].cmds("send_media")) == before


def test_sendonce_wrong_media_kind(rig):
    before = len(rig["bridge"].cmds("send_media"))
    media = [MediaRef(path=str(rig["clip"]), kind="audio",
                      mime="audio/mpeg")]
    reply = _op(rig, "/sendonce", media=media)
    assert "photo or a video" in reply
    assert len(rig["bridge"].cmds("send_media")) == before


def test_sendnote_wrong_media_kind(rig):
    before = len(rig["bridge"].cmds("send_media"))
    media = [MediaRef(path=str(rig["pic"]), kind="image",
                      mime="image/jpeg")]
    reply = _op(rig, "/sendnote", media=media)
    assert "needs a video" in reply
    assert len(rig["bridge"].cmds("send_media")) == before


def test_sendonce_owner_stop_halts(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        before = len(rig["bridge"].cmds("send_media"))
        media = [MediaRef(path=str(rig["pic"]), kind="image",
                          mime="image/jpeg")]
        reply = _op(rig, "/sendonce", media=media)
        assert "standing stop" in reply
        assert len(rig["bridge"].cmds("send_media")) == before
    finally:
        rig["sup"].registry.set_owner_stopped("whatsapp", False)


def test_sendonce_on_telegram_is_platform_truth(rig):
    reply = _op(rig, "/sendonce", platform="telegram",
                chat_id="777", sender_id=TG_OWNER,
                media=[MediaRef(path=str(rig["pic"]),
                                kind="image")])
    assert "no view-once media" in reply
    assert rig["tg"].media_calls == []


def test_sendnote_on_telegram_personal(rig):
    reply = _op(rig, "/sendnote", platform="telegram",
                chat_id="777", sender_id=TG_OWNER,
                media=[MediaRef(path=str(rig["clip"]),
                                kind="video")])
    assert "video note" in reply
    assert rig["tg"].media_calls, "the telegram seat never sent"
    _chat, media = rig["tg"].media_calls[-1]
    assert media.kind == "video_note"
