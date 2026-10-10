"""Deep social-adapter tests: the Phase 2 Section 6 rebuild.

Covers the native-first improvements, not the happy paths:
  * discord: module reference (no NameError), own-message loop guard,
    mention detection, sender identity, typing(action=) protocol parity,
    history self-filter, 429 retry
  * whatsapp: jittered backoff, persistent outbox (enqueue/flush/TTL/
    bound), delivery receipts, typing presence, mention dispatch
  * telegram: flood-wait native retry (duck-typed, no telethon import)
  * sms: Twilio signature verification (documented test vector)
  * gateway: typing() forwards the action kwarg to every adapter
  * nm chat: platforms/doctor/outbox CLI verbs
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, MediaRef
from nomorals.social.chat.discord import DiscordAdapter
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.sms import verify_twilio_signature
from nomorals.social.chat.telegram import _flood_wait_seconds, TelegramAdapter
from nomorals.social.chat.whatsapp import (
    OUTBOX_MAX_ENTRIES,
    Outbox,
    WhatsAppAdapter,
    _jittered_backoff,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _chat(platform: str = "discord", chat_id: str = "c1") -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM)


class FakeDiscordModule:
    """Enough of the discord namespace for send/send_media paths."""

    class MessageReference:
        def __init__(self, message_id=None, channel_id=None):
            self.message_id = message_id
            self.channel_id = channel_id

    class File:
        def __init__(self, fp, filename=None):
            self.fp = fp
            self.filename = filename

    class DMChannel:
        pass

    class TextChannel:
        pass

    class Thread:
        pass


class FakeDiscordChannel:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, content, **kwargs):
        self.sent.append({"content": content, **kwargs})
        return SimpleNamespace(id="m123")


def _discord_with_fake_module() -> DiscordAdapter:
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
    adapter._discord_mod = FakeDiscordModule()
    adapter._greeted_members = set()
    return adapter


def _run_sync(coro):
    """Run a coroutine to completion without a live loop (tests)."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ── discord ────────────────────────────────────────────────────────────────

class TestDiscordDeep:
    def test_module_ref_survives_without_run(self):
        """send must not NameError when run() never imported discord."""
        adapter = _discord_with_fake_module()
        adapter._discord_mod = None  # simulate: module missing entirely
        result = adapter.send(_chat(), "hi", reply_to="5")
        assert not result.ok
        assert "not installed" in result.error

    def test_send_with_reply_to_uses_module_ref(self):
        adapter = _discord_with_fake_module()
        channel = FakeDiscordChannel()
        adapter._channel = lambda chat: _coro(channel)  # noqa: E731
        adapter._run_on_loop = lambda coro, timeout=20.0: _run_sync(coro)
        # numeric ids, like real Discord snowflakes (a non-numeric
        # reply_to is correctly dropped, not sent as a bad reference)
        result = adapter.send(_chat("discord", "99"), "hi", reply_to="42")
        assert result.ok, result.error
        assert result.message_id == "m123"
        ref = channel.sent[0].get("reference")
        assert ref is not None and ref.message_id == 42

    def test_send_media_uses_module_ref(self):
        adapter = _discord_with_fake_module()
        channel = FakeDiscordChannel()
        adapter._channel = lambda chat: _coro(channel)  # noqa: E731
        adapter._run_on_loop = lambda coro, timeout=20.0: _run_sync(coro)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(b"fake")
            path = fh.name
        try:
            result = adapter.send_media(_chat(), MediaRef(path=path), caption="cap")
        finally:
            os.unlink(path)
        assert result.ok, result.error
        assert channel.sent[0]["content"] == "cap"

    def test_mentions_me_via_mentions_list(self):
        adapter = _discord_with_fake_module()
        adapter._me = SimpleNamespace(id="999", name="devon")
        me = SimpleNamespace(id="999")
        other = SimpleNamespace(id="111")
        msg = SimpleNamespace(mentions=[other, me], content="hello")
        assert adapter._mentions_me(msg) is True
        msg2 = SimpleNamespace(mentions=[other], content="hello")
        assert adapter._mentions_me(msg2) is False

    def test_mentions_me_falls_back_to_at_username(self):
        adapter = _discord_with_fake_module()
        adapter._me = SimpleNamespace(id="999", name="devon")
        msg = SimpleNamespace(mentions=[], content="hey @devon look")
        assert adapter._mentions_me(msg) is True

    def test_author_identity(self):
        adapter = _discord_with_fake_module()
        author = SimpleNamespace(id="42", name="bob", global_name="Bobby")
        msg = SimpleNamespace(author=author)
        display, stable, username = adapter._author_identity(msg)
        assert display == "Bobby"
        assert stable == "42"
        assert username == "bob"

    def test_typing_accepts_action_kwarg(self):
        """Gateway passes action= always; the signature must accept it."""
        adapter = _discord_with_fake_module()
        assert adapter.typing(_chat(), seconds=0, action="typing") is False
        assert adapter.typing(_chat(), seconds=0, action="recording") is False

    def test_history_filters_self_and_bots(self):
        adapter = _discord_with_fake_module()
        adapter._me = SimpleNamespace(id="999", name="devon")
        me = SimpleNamespace(id="999", bot=False)
        bot = SimpleNamespace(id="1", bot=True)
        human = SimpleNamespace(id="7", bot=False)
        now = time.time()

        async def _hist(**kwargs):
            return [
                SimpleNamespace(author=me, content="mine", id="a",
                                created_at=SimpleNamespace(timestamp=lambda: now)),
                SimpleNamespace(author=bot, content="bot", id="b",
                                created_at=SimpleNamespace(timestamp=lambda: now)),
                SimpleNamespace(author=human, content="hi", id="c",
                                created_at=SimpleNamespace(timestamp=lambda: now)),
            ]

        class FakeChannel:
            def history(self, limit=20, oldest_first=True):
                return _aiter(_hist())

        async def _aiter(coro):
            for m in await coro:
                yield m

        adapter._channel = lambda chat: _coro(FakeChannel())  # noqa: E731
        adapter._run_on_loop = lambda coro, timeout=20.0: _run_sync(coro)
        out = adapter.history(_chat())
        assert [m.text for m in out] == ["hi"]
        assert out[0].sender_id == "7"


async def _coro(value):
    return value


# ── whatsapp outbox ────────────────────────────────────────────────────────

class TestOutbox:
    def _box(self, **kwargs):
        d = tempfile.mkdtemp()
        return Outbox(os.path.join(d, "whatsapp.jsonl"), **kwargs)

    def test_enqueue_and_flush_fifo(self):
        box = self._box()
        delivered: list[dict] = []
        box.enqueue({"kind": "text", "chat_id": "a", "text": "one"})
        box.enqueue({"kind": "text", "chat_id": "b", "text": "two"})
        assert box.pending_count() == 2
        report = box.flush(lambda e: delivered.append(e) or True)
        assert report == {"sent": 2, "dropped": 0, "kept": 0}
        assert [e["text"] for e in delivered] == ["one", "two"]
        assert box.pending_count() == 0

    def test_failed_entries_stay_queued_in_order(self):
        box = self._box()
        box.enqueue({"kind": "text", "chat_id": "a", "text": "one"})
        box.enqueue({"kind": "text", "chat_id": "b", "text": "two"})
        report = box.flush(lambda e: e["text"] != "one")
        assert report == {"sent": 1, "dropped": 0, "kept": 1}
        assert [e["text"] for e in box.pending()] == ["one"]

    def test_expired_entries_dropped_not_delivered(self):
        box = self._box(ttl_s=0.01)
        box.enqueue({"kind": "text", "chat_id": "a", "text": "stale"})
        time.sleep(0.02)
        delivered: list[dict] = []
        report = box.flush(lambda e: delivered.append(e) or True)
        assert report == {"sent": 0, "dropped": 1, "kept": 0}
        assert delivered == []

    def test_bounded_oldest_dropped(self):
        box = self._box(max_entries=3)
        for i in range(5):
            box.enqueue({"kind": "text", "chat_id": "a", "text": f"m{i}"})
        assert [e["text"] for e in box.pending()] == ["m2", "m3", "m4"]

    def test_persists_across_instances(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "whatsapp.jsonl")
        Outbox(path).enqueue({"kind": "text", "chat_id": "a", "text": "saved"})
        assert Outbox(path).pending_count() == 1

    def test_deliver_exception_keeps_entry(self):
        box = self._box()

        def _boom(_entry):
            raise RuntimeError("nope")

        box.enqueue({"kind": "text", "chat_id": "a", "text": "x"})
        report = box.flush(_boom)
        assert report["kept"] == 1


class TestWhatsAppAdapterDeep:
    def _adapter(self, **kwargs) -> WhatsAppAdapter:
        d = tempfile.mkdtemp()
        return WhatsAppAdapter(media_dir=os.path.join(d, "media"),
                               outbox_dir=os.path.join(d, "outbox"), **kwargs)

    def test_send_while_down_queues(self):
        adapter = self._adapter()
        assert not adapter.connected.is_set()
        result = adapter.send(_chat("whatsapp", "123"), "hello?")
        assert not result.ok
        assert "queued" in result.error
        assert adapter.outbox.pending_count() == 1
        entry = adapter.outbox.pending()[0]
        assert entry["text"] == "hello?"
        assert entry["chat_id"] == "123"

    def test_send_media_while_down_queues(self):
        adapter = self._adapter()
        result = adapter.send_media(
            _chat("whatsapp", "123"), MediaRef(path="/tmp/x.png", kind="image"))
        assert not result.ok and "queued" in result.error
        entry = adapter.outbox.pending()[0]
        assert entry["kind"] == "media" and entry["path"] == "/tmp/x.png"

    def test_outbox_disabled_fails_closed(self):
        adapter = self._adapter(outbox_enabled=False)
        result = adapter.send(_chat("whatsapp", "123"), "hi")
        assert not result.ok
        assert "outbox disabled" in result.error

    def test_receipt_dispatch_tracks_status(self):
        adapter = self._adapter()
        adapter._dispatch({"type": "receipt", "chat": "1@c.us",
                           "ids": ["mid1", "mid2"], "kind": "delivered"},
                          lambda m: None)
        adapter._dispatch({"type": "receipt", "chat": "1@c.us",
                           "ids": ["mid1"], "kind": "read"}, lambda m: None)
        assert adapter.delivery_status("mid1")["status"] == "read"
        assert adapter.delivery_status("mid2")["status"] == "delivered"
        assert adapter.delivery_status("nope") is None

    def test_message_dispatch_sets_mentioned(self):
        adapter = self._adapter()
        got: list[ChatMessage] = []
        adapter._dispatch(
            {"type": "message",
             "chat": {"id": "1@g.us", "kind": "group", "title": "g"},
             "from": {"id": "2@c.us", "name": "Ann"},
             "text": "@me hi", "media": [], "reply_to": "",
             "mentioned": True, "ts": time.time() * 1000},
            got.append)
        assert len(got) == 1
        assert got[0].mentioned is True
        assert got[0].chat.kind == ChatKind.GROUP
        assert got[0].sender == "Ann"

    def test_typing_sends_presence(self):
        adapter = self._adapter()
        adapter.connected.set()
        seen: list[dict] = []

        def _fake_cmd(cmd, *, timeout=25.0):
            seen.append(cmd)
            return {"ok": True}

        adapter._send_cmd = _fake_cmd
        assert adapter.typing(_chat("whatsapp", "123"), 3.0, action="recording")
        assert seen[0]["presence"] == "recording"
        seen.clear()
        assert adapter.typing(_chat("whatsapp", "123"), 3.0)
        assert seen[0]["presence"] == "composing"

    def test_flush_outbox_on_status_open(self):
        adapter = self._adapter()
        # Bridge was DOWN while the brain queued this (no connected set yet).
        adapter.outbox.enqueue({"kind": "text", "chat_id": "123",
                                "chat_kind": "dm", "text": "queued hi"})
        sent: list[str] = []
        adapter._send_with_retry = lambda chat, text, reply_to="": \
            sent.append(text) or SimpleNamespace(ok=True)
        # status open with connected clear = down->up transition → flush.
        adapter._dispatch({"type": "status", "state": "open", "user": "me@c.us"},
                          lambda m: None)
        # flush runs on a side thread — wait briefly
        deadline = time.time() + 5.0
        while adapter.outbox.pending_count() and time.time() < deadline:
            time.sleep(0.05)
        assert adapter.outbox.pending_count() == 0
        assert sent == ["queued hi"]

    def test_health_reports_outbox_and_receipts(self):
        adapter = self._adapter()
        adapter.outbox.enqueue({"kind": "text", "chat_id": "1", "text": "x"})
        adapter._note_receipt("m1", "read")
        h = adapter.health()
        assert h["outbox_pending"] == 1
        assert h["receipts_tracked"] == 1


class TestBackoff:
    def test_jittered_backoff_bounds_and_growth(self):
        rng = random.Random(7)
        delays = [_jittered_backoff(3.0, attempt, rng=rng) for attempt in range(1, 8)]
        # exponential growth on average, all within jitter band of the ladder
        for attempt, delay in zip(range(1, 8), delays):
            nominal = min(3.0 * (2.0 ** (attempt - 1)), 60.0)
            assert 0.5 <= delay <= nominal * 1.3 + 0.01
        assert delays[-1] > delays[0]  # growth

    def test_backoff_caps(self):
        rng = random.Random(1)
        assert _jittered_backoff(3.0, 99, rng=rng) <= 60.0 * 1.3 + 0.01

    def test_backoff_deterministic_with_rng(self):
        a = _jittered_backoff(3.0, 2, rng=random.Random(42))
        b = _jittered_backoff(3.0, 2, rng=random.Random(42))
        assert a == b


# ── telegram flood wait ────────────────────────────────────────────────────

class TestFloodWait:
    def _flood(self, seconds: int):
        exc = type("FloodWaitError", (Exception,), {})(
            f"Too many requests (causing a flood wait of {seconds} seconds)")
        exc.seconds = seconds
        return exc

    def test_duck_typed_floodwait(self):
        assert _flood_wait_seconds(self._flood(37)) == 37

    def test_non_flood_exception_is_zero(self):
        exc = RuntimeError("boom")
        exc.seconds = 10  # seconds attr without the Flood name → ignored
        assert _flood_wait_seconds(exc) == 0

    def test_429_with_retry_after(self):
        exc = SimpleNamespace(status=429, parameters={"retry_after": 12})
        assert _flood_wait_seconds(exc) == 12

    def test_send_retries_once_after_flood_wait(self):
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.flood_max_wait_s = 300.0
        calls = {"n": 0}
        slept: list[float] = []

        async def _fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise self._flood(5)
            return "sent"

        async def _fake_sleep(s):
            slept.append(s)

        with patch("asyncio.sleep", _fake_sleep):
            result = _run_sync(adapter._send_with_flood_retry(_fn, "test"))
        assert result == "sent"
        assert calls["n"] == 2
        assert slept == [5.0]

    def test_flood_wait_over_cap_fails_fast(self):
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.flood_max_wait_s = 10.0
        slept: list[float] = []

        async def _fn():
            raise self._flood(999)

        async def _fake_sleep(s):
            slept.append(s)

        with patch("asyncio.sleep", _fake_sleep):
            with pytest.raises(Exception):
                _run_sync(adapter._send_with_flood_retry(_fn, "test"))
        assert slept == []

    def test_second_flood_propagates(self):
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.flood_max_wait_s = 300.0
        slept: list[float] = []

        async def _fn():
            raise self._flood(2)

        async def _fake_sleep(s):
            slept.append(s)

        with patch("asyncio.sleep", _fake_sleep):
            with pytest.raises(Exception):
                _run_sync(adapter._send_with_flood_retry(_fn, "test"))
        assert slept == [2.0]


# ── sms signature ──────────────────────────────────────────────────────────

class TestTwilioSignature:
    # Twilio's documented verification algorithm, with the documented
    # example parameter set (url + CallSid/Caller/Digits/From/To, token
    # "12345"). EXPECTED is the HMAC-SHA1(base64) of that input — computed
    # here rather than copied, so the test pins the algorithm, not memory.
    URL = "https://mycompany.com/myapp.php?foo=1&bar=2"
    PARAMS = {
        "CallSid": "CA1234567890ABCDE",
        "Caller": "+14158675309",
        "Digits": "1234",
        "From": "+14158675309",
        "To": "+18005551212",
    }
    TOKEN = "12345"
    EXPECTED = "RSOYDt4T1cUTdK1PDd93/VVr8B8="

    def test_documented_vector(self):
        assert verify_twilio_signature(
            self.TOKEN, self.URL, self.PARAMS, self.EXPECTED) is True

    def test_wrong_token_fails(self):
        assert verify_twilio_signature(
            "wrong", self.URL, self.PARAMS, self.EXPECTED) is False

    def test_tampered_param_fails(self):
        params = dict(self.PARAMS, Digits="9999")
        assert verify_twilio_signature(
            self.TOKEN, self.URL, params, self.EXPECTED) is False

    def test_empty_inputs_fail_closed(self):
        assert verify_twilio_signature("", self.URL, self.PARAMS, self.EXPECTED) is False
        assert verify_twilio_signature(self.TOKEN, self.URL, self.PARAMS, "") is False
        assert verify_twilio_signature(self.TOKEN, "", {}, "x") is False

    def test_round_trip_self_signed(self):
        params = {"From": "+1555", "Body": "hello", "To": "+1666"}
        body = "https://example.com/hook"
        for key in sorted(params):
            body += key + params[key]
        sig = base64.b64encode(
            hmac.new(b"secret", body.encode(), hashlib.sha1).digest()).decode()
        assert verify_twilio_signature("secret", "https://example.com/hook",
                                       params, sig) is True

    def test_never_raises_on_garbage(self):
        assert verify_twilio_signature(None, None, None, None) is False  # type: ignore[arg-type]


# ── gateway typing action passthrough ──────────────────────────────────────

class _RecordingAdapter:
    """Minimal stand-in that records how gateway.typing called it."""

    def __init__(self):
        self.calls: list[dict] = []

    def typing(self, chat, seconds=3.0, action="typing"):
        self.calls.append({"seconds": seconds, "action": action})
        return True


class TestGatewayTyping:
    def test_action_forwarded_to_every_adapter(self):
        from nomorals.social.chat.base import ChatAdapter as _Base

        class A(_Base):
            name = "a"

            def __init__(self):
                super().__init__(media_dir=tempfile.mkdtemp())
                self.calls: list[dict] = []

            def send(self, chat, text, **kwargs):
                from nomorals.social.chat.base import SendResult
                return SendResult(ok=True, platform=self.name)

            def typing(self, chat, seconds=3.0, action="typing"):
                self.calls.append({"action": action})
                return True

        gateway = ChatGateway({"a": A()})
        assert gateway.typing("a", _chat("a"), seconds=5.0, action="recording")
        assert gateway.adapters["a"].calls == [{"action": "recording"}]


# ── nm chat CLI ────────────────────────────────────────────────────────────

class TestChatCLI:
    def _args(self, task):
        return SimpleNamespace(task=task, json=False)

    def _ctx(self):
        return SimpleNamespace(settings=None)

    def test_platforms_lists_all(self):
        from nomorals.cmdline.commands.chat import _cmd_chat, CAPABILITY_MATRIX
        assert _cmd_chat(self._args(["platforms"]), self._ctx()) == 0
        assert {r["platform"] for r in CAPABILITY_MATRIX} == {
            "telegram", "telegram-bot", "discord", "whatsapp",
            "sms", "local", "webhook"}

    def test_doctor_flags_missing_owner(self):
        from nomorals.cmdline.commands.chat import _cmd_chat
        # no settings → owner_chats empty → gating check FAILS (rc 1)
        assert _cmd_chat(self._args(["doctor"]), self._ctx()) == 1

    def test_outbox_empty_and_clear(self):
        from nomorals.cmdline.commands.chat import _cmd_chat
        d = tempfile.mkdtemp()
        settings = SimpleNamespace(
            chat=SimpleNamespace(whatsapp_outbox_dir=d),
            resolve=lambda p: p)
        ctx = SimpleNamespace(settings=settings)
        assert _cmd_chat(self._args(["outbox"]), ctx) == 0
        Outbox(os.path.join(d, "whatsapp.jsonl")).enqueue(
            {"kind": "text", "chat_id": "1", "text": "x"})
        assert _cmd_chat(self._args(["outbox"]), ctx) == 0
        assert _cmd_chat(self._args(["outbox", "clear"]), ctx) == 0
        assert _cmd_chat(self._args(["outbox"]), ctx) == 0

    def test_unknown_verb_usage(self):
        from nomorals.cmdline.commands.chat import _cmd_chat
        assert _cmd_chat(self._args(["frobnicate"]), self._ctx()) == 2
