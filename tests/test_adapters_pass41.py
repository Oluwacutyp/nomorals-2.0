"""Pass 41 — social platform adapters + immersion surface.

Defects pinned here (each failed on pristine HEAD):

* WhatsApp outbox flush bypassed the live path's WhatsApp-markdown
  conversion + splitting — a reply queued while the bridge was down
  went out with Telegram HTML still in it, unsplit.
* WhatsApp ``SendResult.message_id`` was the bridge *request* id (the
  bridge discarded the real WhatsApp message id), so delivery receipts
  could never be matched via ``delivery_status()``.
* WhatsApp inbound normalization dropped ``sender_id`` (the bridge
  sends ``from.id``), ``message_id`` (the bridge never sent
  ``m.key.id``), and ``is_voice`` on voice-note media.
* WhatsApp ``_jid()`` mangled JIDs with unrecognized suffixes
  (``@lid``) into ``digits@c.us``.
* Discord ``_channel()`` resolved DMs via ``fetch_user(chat_id)`` —
  but this adapter's chat ids are DM *channel* ids everywhere
  (inbound stores ``channel.id``, ``start_dm`` returns ``dm.id``), so
  every DM send resolved the wrong entity and failed.
* Discord inbound attachments: any non-image became kind="file" —
  audio/video (incl. Discord voice messages) lost their kind and
  never reached STT/vision routing.
* TelegramBotAdapter.send_media: kind="voice" / voice_note=True fell
  through to sendDocument — TTS voice replies went out as file
  attachments on both Telegram bots instead of voice bubbles.
* SMS split_sms ignored the "(i/n) " prefix budget: numbered parts
  exceeded the 153/67 concatenated-segment budget, so carriers
  re-split them (double billing) — the documented SMS_CONCAT_CHARS
  design was never wired in.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, MediaRef, SendResult
from nomorals.social.chat.discord import DiscordAdapter
from nomorals.social.chat.sms import split_sms
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.whatsapp import WhatsAppAdapter


def _run_sync(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── WhatsApp: outbox flush must render like the live send path ─────────────

class TestWhatsAppFlushRendering:
    def _adapter_with_recorder(self):
        adapter = WhatsAppAdapter(outbox_enabled=False)
        adapter.connected.set()
        sent: list[tuple[str, str]] = []

        def _record(chat, text, *, reply_to=""):
            sent.append((chat.chat_id, text))
            return SendResult(ok=True, platform="whatsapp", message_id="m1")

        adapter._send_with_retry = _record  # type: ignore[method-assign]
        return adapter, sent

    def test_flush_converts_telegram_html_to_whatsapp(self):
        adapter, sent = self._adapter_with_recorder()
        ok = adapter._deliver_entry({
            "kind": "text", "chat_id": "2348012345678@c.us",
            "chat_kind": "dm", "text": "<b>hello</b> **bold** `code`",
        })
        assert ok is True
        assert len(sent) == 1
        text = sent[0][1]
        assert "<b>" not in text and "**" not in text
        assert "*hello*" in text and "*bold*" in text

    def test_flush_splits_long_text(self):
        adapter, sent = self._adapter_with_recorder()
        long_text = ("lorem ipsum dolor sit amet " * 200).strip()  # ~5400 chars
        ok = adapter._deliver_entry({
            "kind": "text", "chat_id": "2348012345678@c.us",
            "chat_kind": "dm", "text": long_text,
        })
        assert ok is True
        assert len(sent) >= 2
        assert all(len(t) <= 4000 for _, t in sent)


# ── WhatsApp: bridge fake (send replies carry the real message id) ─────────

class _FakeBridge:
    """Minimal JSON-lines bridge; script maps cmd -> response dict."""

    def __init__(self, script):
        self.script = script
        self.received: list[dict] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        buf = b""
        try:
            with conn:
                conn.settimeout(5)
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, _, buf = buf.partition(b"\n")
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line.decode("utf-8"))
                        self.received.append(obj)
                        resp = self.script.get(obj.get("cmd"))
                        if resp is None:
                            resp = {"ok": False, "error": "unknown command"}
                        # the real bridge replies {id: <request id>, ...}
                        conn.sendall((json.dumps(
                            {"id": obj.get("id"), **resp}) + "\n").encode())
        except OSError:
            pass

    def close(self):
        try:
            self._srv.close()
        except OSError:
            pass


def _wa_connected(bridge):
    adapter = WhatsAppAdapter(host="127.0.0.1", port=bridge.port,
                              outbox_enabled=False)
    adapter._sock = socket.create_connection(("127.0.0.1", bridge.port),
                                             timeout=5)
    reader = threading.Thread(target=adapter._reader_loop,
                              args=(lambda m: None,), daemon=True)
    reader.start()
    adapter.connected.set()
    return adapter


class TestWhatsAppSendMessageId:
    def test_send_returns_real_message_id_and_receipts_match(self):
        bridge = _FakeBridge({"send": {"ok": True, "message_id": "WA-REAL-1"}})
        try:
            adapter = _wa_connected(bridge)
            chat = ChatRef(platform="whatsapp",
                           chat_id="2348012345678@c.us", kind=ChatKind.DM)
            result = adapter.send(chat, "hello")
            assert result.ok, result.error
            # The receipt key space is real WhatsApp message ids (from
            # the bridge's messages.update feed) — the SendResult must
            # carry that same id, not the bridge request id.
            assert result.message_id == "WA-REAL-1"
            adapter._note_receipt("WA-REAL-1", "read")
            status = adapter.delivery_status(result.message_id)
            assert status is not None and status["status"] == "read"
        finally:
            bridge.close()

    def test_send_media_returns_real_message_id(self):
        bridge = _FakeBridge(
            {"send_media": {"ok": True, "message_id": "WA-REAL-2"}})
        try:
            adapter = _wa_connected(bridge)
            chat = ChatRef(platform="whatsapp",
                           chat_id="2348012345678@c.us", kind=ChatKind.DM)
            result = adapter.send_media(
                chat, MediaRef(path="/tmp/x.ogg", kind="voice"))
            assert result.ok, result.error
            assert result.message_id == "WA-REAL-2"
        finally:
            bridge.close()


# ── WhatsApp: inbound normalization keeps sender/message/voice identity ────

class TestWhatsAppInboundNormalization:
    def test_message_dispatch_sets_sender_id_message_id_is_voice(self):
        adapter = WhatsAppAdapter(outbox_enabled=False)
        msgs: list[ChatMessage] = []
        adapter._dispatch({
            "type": "message", "id": "WAMID-99",
            "chat": {"id": "120363@g.us", "kind": "group", "title": "Crew"},
            "from": {"id": "2348099999999@c.us", "name": "Ada"},
            "text": "",
            "media": [{"path": "/tmp/v.ogg", "mime": "audio/ogg",
                       "kind": "voice"}],
            "reply_to": "", "mentioned": False, "ts": 1759276800000,
        }, msgs.append)
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg.message_id == "WAMID-99"
        assert msg.sender_id == "2348099999999@c.us"
        assert msg.media[0].kind == "voice"
        assert msg.media[0].is_voice is True


class TestWhatsAppJid:
    def test_lid_jid_passes_through_unmangled(self):
        adapter = WhatsAppAdapter(outbox_enabled=False)
        chat = ChatRef(platform="whatsapp",
                       chat_id="123456789012345@lid", kind=ChatKind.DM)
        assert adapter._jid(chat) == "123456789012345@lid"

    def test_plain_number_still_becomes_c_us(self):
        adapter = WhatsAppAdapter(outbox_enabled=False)
        chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                       kind=ChatKind.DM)
        assert adapter._jid(chat) == "2348012345678@c.us"

    def test_known_suffixes_unchanged(self):
        adapter = WhatsAppAdapter(outbox_enabled=False)
        for jid in ("2348012345678@c.us", "2348012345678@s.whatsapp.net",
                    "120363@g.us"):
            chat = ChatRef(platform="whatsapp", chat_id=jid,
                           kind=ChatKind.GROUP if jid.endswith("@g.us")
                           else ChatKind.DM)
            assert adapter._jid(chat) == jid


# ── Discord: DM channel resolution + inbound media kinds ───────────────────

class _FakeDiscordChannel:
    def __init__(self, cid):
        self.id = cid
        self.sent: list[dict] = []

    async def send(self, content, **kwargs):
        self.sent.append({"content": content, **kwargs})
        return SimpleNamespace(id="m-snow")


def _discord_adapter():
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    from nomorals.social.chat.base import ChatAdapter as _Base
    _Base.__init__(adapter, media_dir=tempfile.mkdtemp())
    adapter.token = "tok"
    adapter.channel_allow = set()
    adapter.greet_new = False
    adapter.on_new_member = None
    adapter._bot = None
    adapter._loop = None
    adapter._me = None
    adapter._discord_mod = None
    adapter._greeted_members = set()
    return adapter


class TestDiscordDmChannelResolution:
    def test_dm_send_resolves_channel_id_not_user_id(self):
        adapter = _discord_adapter()
        channel = _FakeDiscordChannel("555")
        calls: list[str] = []

        class _Bot:
            async def fetch_channel(self, cid):
                calls.append(f"channel:{cid}")
                return channel

            async def fetch_user(self, uid):  # a channel id is not a user
                calls.append(f"user:{uid}")
                raise RuntimeError("404 unknown user")

        adapter._bot = _Bot()
        adapter._run_on_loop = lambda coro, timeout=20.0: _run_sync(coro)
        # _dc() must find a module for MessageReference etc.
        adapter._discord_mod = SimpleNamespace(
            MessageReference=lambda **kw: SimpleNamespace(**kw))
        chat = ChatRef(platform="discord", chat_id="555", kind=ChatKind.DM)
        result = adapter.send(chat, "hi ada")
        assert result.ok, result.error
        assert calls == ["channel:555"]
        assert channel.sent and channel.sent[0]["content"] == "hi ada"

    def test_dm_send_falls_back_to_user_when_no_channel(self):
        adapter = _discord_adapter()
        channel = _FakeDiscordChannel("777")

        class _User:
            async def create_dm(self):
                return channel

        class _Bot:
            async def fetch_channel(self, cid):
                raise RuntimeError("404 unknown channel")

            async def fetch_user(self, uid):
                return _User()

        adapter._bot = _Bot()
        adapter._run_on_loop = lambda coro, timeout=20.0: _run_sync(coro)
        adapter._discord_mod = SimpleNamespace(
            MessageReference=lambda **kw: SimpleNamespace(**kw))
        chat = ChatRef(platform="discord", chat_id="4242", kind=ChatKind.DM)
        result = adapter.send(chat, "hi")
        assert result.ok, result.error
        assert channel.sent and channel.sent[0]["content"] == "hi"


class _FakeAttachment:
    def __init__(self, filename, content_type, voice=False):
        self.filename = filename
        self.content_type = content_type
        self._voice = voice

    def is_voice_message(self):
        return self._voice

    async def save(self, path):
        with open(path, "wb") as fh:
            fh.write(b"bytes")


class TestDiscordInboundMediaKinds:
    def _capture_on_message(self):
        """Run DiscordAdapter.run() against a fake discord module and
        return the registered on_message handler."""
        handlers: dict = {}

        class _Client:
            def event(self, fn):
                handlers[fn.__name__] = fn
                return fn

            def run(self, token, log_handler=None):
                return None

        fake = SimpleNamespace(
            Client=_Client,
            DMChannel=type("DMChannel", (), {}),
            TextChannel=type("TextChannel", (), {}),
            Thread=type("Thread", (), {}),
        )
        adapter = _discord_adapter()
        import nomorals.social.chat.discord as dc_mod
        original = sys.modules.get("discord")
        sys.modules["discord"] = fake  # type: ignore[assignment]
        try:
            adapter.run(lambda m: None)
        finally:
            if original is None:
                sys.modules.pop("discord", None)
            else:
                sys.modules["discord"] = original
        return adapter, fake, handlers["on_message"]

    def test_voice_video_image_kinds_survive(self):
        adapter, fake, on_message = self._capture_on_message()
        adapter._me = SimpleNamespace(id="999", name="devon")
        received: list[ChatMessage] = []

        channel = object.__new__(fake.DMChannel)
        channel.id = 555
        channel.name = "ada"
        author = SimpleNamespace(id="42", name="ada", global_name="Ada",
                                 bot=False)
        message = SimpleNamespace(
            author=author, channel=channel, content="listen",
            mentions=[], id=777,
            created_at=SimpleNamespace(timestamp=lambda: time.time()),
            attachments=[
                _FakeAttachment("voice-message.ogg", "audio/ogg", voice=True),
                _FakeAttachment("clip.mp4", "video/mp4"),
                _FakeAttachment("pic.png", "image/png"),
                _FakeAttachment("song.mp3", "audio/mpeg"),
            ],
        )
        # on_message delivers through adapter._deliver — capture there.
        captured: list[ChatMessage] = []
        adapter._deliver = lambda h, m: captured.append(m)  # type: ignore
        _run_sync(on_message(message))
        assert len(captured) == 1
        kinds = [(m.kind, m.is_voice) for m in captured[0].media]
        assert kinds == [("voice", True), ("video", False),
                         ("image", False), ("audio", False)]


# ── Telegram bots: voice notes must go out as voice bubbles ────────────────

class _FakeBotSession:
    def __init__(self):
        self.posts: list[dict] = []

    def post(self, url, data=None, files=None, timeout=None):
        self.posts.append({"url": url, "data": data, "files": files})
        return SimpleNamespace(
            json=lambda: {"ok": True, "result": {"message_id": 9}})


class TestTelegramBotVoiceSend:
    def _adapter(self):
        session = _FakeBotSession()
        adapter = TelegramBotAdapter(token="123456:secret", session=session)
        return adapter, session

    def _tmp(self, suffix):
        fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        fh.write(b"bytes")
        fh.close()
        return fh.name

    def test_kind_voice_sends_as_voice_bubble(self):
        adapter, session = self._adapter()
        path = self._tmp(".ogg")
        chat = ChatRef(platform="telegram-bot", chat_id="42", kind=ChatKind.DM)
        result = adapter.send_media(
            chat, MediaRef(path=path, kind="voice", voice_note=True,
                           mime="audio/ogg"))
        assert result.ok, result.error
        assert session.posts[0]["url"].endswith("/sendVoice")
        assert "voice" in session.posts[0]["files"]

    def test_kind_voice_without_flag_sends_as_voice_bubble(self):
        adapter, session = self._adapter()
        path = self._tmp(".ogg")
        chat = ChatRef(platform="telegram-personal", chat_id="42",
                       kind=ChatKind.DM)
        result = adapter.send_media(
            chat, MediaRef(path=path, kind="voice", mime="audio/ogg"))
        assert result.ok, result.error
        assert session.posts[0]["url"].endswith("/sendVoice")
        assert "voice" in session.posts[0]["files"]

    def test_voice_note_flag_with_ogg_pin(self):
        # Behavior pin (passed pre-fix via the .ogg suffix heuristic):
        # kind="audio" + voice_note=True + .ogg already went to
        # sendVoice; the fix must not break it.
        adapter, session = self._adapter()
        path = self._tmp(".ogg")
        chat = ChatRef(platform="telegram-personal", chat_id="42",
                       kind=ChatKind.DM)
        result = adapter.send_media(
            chat, MediaRef(path=path, kind="audio", voice_note=True,
                           mime="audio/ogg"))
        assert result.ok, result.error
        assert session.posts[0]["url"].endswith("/sendVoice")

    def test_audio_file_still_send_audio(self):
        adapter, session = self._adapter()
        path = self._tmp(".mp3")
        chat = ChatRef(platform="telegram-bot", chat_id="42", kind=ChatKind.DM)
        result = adapter.send_media(
            chat, MediaRef(path=path, kind="audio", mime="audio/mpeg"))
        assert result.ok, result.error
        assert session.posts[0]["url"].endswith("/sendAudio")


# ── Menus: WhatsApp /help must serve the WhatsApp-native menu ──────────────

class TestWhatsAppHelpMenuWiring:
    """control.help_menu('whatsapp') renders a purpose-built,
    text-first WhatsApp menu — but on pristine HEAD nothing in
    production called it: the runtime's /help branch answered every
    platform with detailed_help() (Telegram-styled catalog), so the
    owner's "WhatsApp gets its OWN menu" rule was dead code."""

    def _runtime(self):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings

        tmp = tempfile.mkdtemp(prefix="nm-pass41-")
        settings = load_settings(overrides={
            "home": tmp,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)

        class _Router:
            def chat(self, *a, **k):
                raise AssertionError("test must not call the model")

        context.router = _Router()

        class _Gateway:
            adapters: dict = {}

            def status(self):
                return {}

            def set_rate_limit(self, *a, **k):
                return None

            def typing(self, *a, **k):
                return None

        return PartnerRuntime(context, gateway=_Gateway())

    def _msg(self, text, chat_key):
        platform, _, chat_id = chat_key.partition(":")
        return ChatMessage(
            chat=ChatRef(platform=platform, chat_id=chat_id,
                         kind=ChatKind.DM),
            incoming=True, text=text, sender="owner",
            sender_id="5478650254", meta={"is_owner": True})

    def test_whatsapp_help_gets_whatsapp_menu(self):
        runtime = self._runtime()
        key = "whatsapp:2348012345678@c.us"
        out = runtime.handle_control("/help", key,
                                     message=self._msg("/help", key))
        # The WhatsApp-native menu's stable markers.
        assert "text-first — no buttons" in out
        assert "GROUPS & COMMUNITIES" in out

    def test_telegram_help_unchanged(self):
        runtime = self._runtime()
        key = "telegram:999"
        out = runtime.handle_control("/help", key,
                                     message=self._msg("/help", key))
        assert "text-first — no buttons" not in out
        assert "the full catalog" in out

    def test_whatsapp_help_topic_still_detailed(self):
        runtime = self._runtime()
        key = "whatsapp:2348012345678@c.us"
        out = runtime.handle_control("/help budget", key,
                                     message=self._msg("/help budget", key))
        assert out.strip()  # a real page, not a dead end


# ── SMS: numbered parts must fit the concatenated-segment budget ───────────

class TestSmsSplitBudget:
    def test_gsm_numbered_parts_fit_concat_budget(self):
        segs = split_sms("word " * 100)  # 500 GSM chars
        assert len(segs) >= 3
        # 153 is the whole per-part budget on the wire (UDH included in
        # the 160); the "(i/n) " prefix must come OUT of it, not on top.
        assert all(len(s) <= 153 for s in segs), [len(s) for s in segs]

    def test_ucs2_numbered_parts_fit_concat_budget(self):
        segs = split_sms("héllo wörld 🎉 " * 20)  # UCS-2 via the emoji
        assert len(segs) >= 2
        assert all(len(s) <= 67 for s in segs), [len(s) for s in segs]

    def test_single_segment_unnumbered_unchanged(self):
        assert split_sms("x" * 160) == ["x" * 160]
        segs = split_sms("w " * 80 + "x")  # 161 chars -> 2 numbered parts
        assert len(segs) == 2
        assert segs[0].startswith("(1/2) ") and segs[1].startswith("(2/2) ")

    def test_reassembly_recovers_text(self):
        text = "alpha beta gamma delta " * 30
        segs = split_sms(text)
        joined = " ".join(s.split(" ", 1)[1] for s in segs)
        assert joined.replace(" ", "") == text.replace(" ", "")
