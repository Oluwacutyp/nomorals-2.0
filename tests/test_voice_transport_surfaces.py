"""Pass 5: per-surface voice-note transports + Saved Messages delivery.

Pins:
- The Bot API adapter's media sends run on its ONE session contract
  (core HttpClient surface: post_json / get / post_multipart) — the
  production default session included. No requests-style ``.post``
  anywhere (the ``'HttpClient' object has no attribute 'post'`` death).
- Transport selection per surface: bot → Bot API sendVoice; personal
  MTProto → Telethon send_file(voice_note=True); WhatsApp → ptt send.
- The no-cross-client rule: the MTProto transport works against an
  adapter that has no Bot API client members at all.
- Generated artifacts on the personal surface default to Saved
  Messages; surfaces without one fail honestly.
- Degradation is labelled: conversion unavailable or a rejected voice
  send still delivers the artifact as a file, marked degraded with
  the reason — in the transport result AND in the /tts reply text.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

from nomorals.social.chat import voice_transport as vt
from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef, SendResult
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter


# ── helpers ──────────────────────────────────────────────────────────────

def _tmp(suffix: str, data: bytes = b"audio-bytes") -> str:
    fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fh.write(data)
    fh.close()
    return fh.name


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _HttpClientSession:
    """The core-HttpClient session contract. NOTE: no ``.post`` —
    a requests-style call against this object is the old bug."""

    def __init__(self, fail_methods=()):
        self.calls: list[dict] = []
        self.fail_methods = set(fail_methods)

    def post_json(self, url, params, timeout=None):
        self.calls.append({"url": url, "params": params})
        return _Resp({"ok": True, "result": {"message_id": 9}})

    def get(self, url, timeout=None):
        self.calls.append({"url": url})
        return _Resp({"ok": True, "result": {}})

    def post_multipart(self, url, fields=None, files=None, timeout=None):
        self.calls.append({"url": url, "fields": fields, "files": files})
        method = url.rsplit("/", 1)[-1]
        if method in self.fail_methods:
            return _Resp({"ok": False, "description": f"{method} rejected"})
        return _Resp({"ok": True, "result": {"message_id": 9}})


def _bot_adapter(session=None, name="telegram-personal"):
    return TelegramBotAdapter(token="<redacted>", session=session,
                                adapter_name=name,
                                media_dir=tempfile.mkdtemp())


def _chat(platform="telegram-personal", chat_id="42"):
    return ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM)


def _ogg(monkeypatch, tmp_suffix=".ogg"):
    ogg = _tmp(tmp_suffix)
    monkeypatch.setattr(
        "nomorals.tools.audio.wav_to_voice_ogg",
        lambda p: (ogg, "audio/ogg"))
    return ogg


# ── Bot API adapter: the session contract (the EC2 failure, pinned) ─────

class TestBotAdapterSessionContract:
    def test_send_media_voice_goes_through_post_multipart(self):
        session = _HttpClientSession()
        assert not hasattr(session, "post")  # the old call would die here
        adapter = _bot_adapter(session)
        path = _tmp(".ogg")
        result = adapter.send_media(
            _chat("telegram-bot"), MediaRef(path=path, kind="voice",
                                            voice_note=True,
                                            mime="audio/ogg"))
        assert result.ok, result.error
        assert result.message_id == "9"
        uploads = [c for c in session.calls if "files" in c]
        assert len(uploads) == 1
        assert uploads[0]["url"].endswith("/sendVoice")
        assert uploads[0]["files"][0][0] == "voice"

    def test_send_media_default_session_is_the_httpclient_contract(
            self, monkeypatch):
        # Production constructs the adapter with NO session: _sess()
        # builds the core HttpClient. Pin that path end to end.
        session = _HttpClientSession()
        monkeypatch.setattr("nomorals.core.http.HttpClient",
                            lambda *a, **k: session)
        adapter = _bot_adapter(session=None)
        path = _tmp(".ogg")
        result = adapter.send_media(
            _chat(), MediaRef(path=path, kind="voice", voice_note=True,
                              mime="audio/ogg"))
        assert result.ok, result.error
        assert any(c["url"].endswith("/sendVoice")
                   for c in session.calls if "files" in c)

    def test_edit_media_uses_the_same_contract(self):
        session = _HttpClientSession()
        adapter = _bot_adapter(session)
        path = _tmp(".png")
        result = adapter.edit_media(_chat("telegram-bot"), "7",
                                    MediaRef(path=path, kind="image"))
        assert result.ok, result.error
        uploads = [c for c in session.calls if "files" in c]
        assert uploads and uploads[0]["url"].endswith("/editMessageMedia")
        assert uploads[0]["files"][0][0] == path.rsplit("/", 1)[-1]


# ── Bot API transport ────────────────────────────────────────────────────

class TestBotApiVoiceTransport:
    def test_voice_note_success(self, monkeypatch):
        _ogg(monkeypatch)
        session = _HttpClientSession()
        adapter = _bot_adapter(session)
        delivery = vt.send_voice_note(adapter, _chat(), _tmp(".wav"))
        assert delivery.ok and delivery.voice_note
        assert not delivery.degraded
        assert isinstance(vt.voice_transport_for(adapter),
                          vt.BotApiVoiceTransport)

    def test_conversion_unavailable_sends_nothing(self, monkeypatch):
        # No OGG/Opus → no voice bubble can exist: the transport sends
        # NOTHING and returns the reason; the caller's labelled file
        # fallback (gateway file route) does the degraded delivery.
        wav = _tmp(".wav")
        monkeypatch.setattr(
            "nomorals.tools.audio.wav_to_voice_ogg",
            lambda p: (wav, "audio/wav"))  # ffmpeg missing → stays wav
        session = _HttpClientSession()
        adapter = _bot_adapter(session)
        delivery = vt.send_voice_note(adapter, _chat(), wav)
        assert not delivery.ok and not delivery.degraded
        assert not delivery.voice_note
        assert "ffmpeg not available" in delivery.reason
        assert [c for c in session.calls if "files" in c] == []

    def test_degraded_when_voice_send_rejected(self, monkeypatch):
        _ogg(monkeypatch)
        session = _HttpClientSession(fail_methods={"sendVoice"})
        adapter = _bot_adapter(session)
        delivery = vt.send_voice_note(adapter, _chat(), _tmp(".wav"))
        assert delivery.ok and delivery.degraded
        assert "sendVoice rejected" in delivery.reason
        methods = [c["url"].rsplit("/", 1)[-1]
                   for c in session.calls if "files" in c]
        assert methods == ["sendVoice", "sendDocument"]

    def test_total_failure_reports_both_reasons(self, monkeypatch):
        _ogg(monkeypatch)
        session = _HttpClientSession(
            fail_methods={"sendVoice", "sendDocument"})
        adapter = _bot_adapter(session)
        delivery = vt.send_voice_note(adapter, _chat(), _tmp(".wav"))
        assert not delivery.ok and not delivery.degraded
        assert "sendVoice rejected" in delivery.reason
        assert "file fallback also failed" in delivery.reason


# ── MTProto (personal / userbot) transport ───────────────────────────────

class _RecordingUserbot(TelegramAdapter):
    """A real TelegramAdapter with send_media replaced by a recorder.

    It has NO Bot API members (no _sess / post_multipart / token): if
    the MTProto transport touched a Bot API client shape it would
    raise AttributeError and the delivery would fail — the
    no-cross-client rule, enforced by construction.
    """

    def __init__(self, me_id=777, fail_kinds=()):
        super().__init__(api_id=1, api_hash="x",
                         session_path=os.path.join(
                             tempfile.mkdtemp(), "s.session"),
                         media_dir=tempfile.mkdtemp())
        self._me_id = me_id
        self.sent: list[tuple] = []
        self.fail_kinds = set(fail_kinds)

    def send_media(self, chat, media, *, caption=""):
        self.sent.append((chat, media, caption))
        if media.kind in self.fail_kinds:
            return SendResult(ok=False, platform=self.name,
                              error=f"{media.kind} rejected")
        return SendResult(ok=True, platform=self.name, message_id="5")


class TestMTProtoVoiceTransport:
    def test_transport_selection_and_voice_send(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _RecordingUserbot()
        assert isinstance(vt.voice_transport_for(adapter),
                          vt.MTProtoVoiceTransport)
        chat = _chat("telegram", "1234")  # some other DM — conversational
        delivery = vt.send_voice_note(adapter, chat, _tmp(".wav"))
        assert delivery.ok and delivery.voice_note
        (sent_chat, media, _caption), = adapter.sent
        assert sent_chat is chat  # conversational addressing kept
        assert media.voice_note is True and media.kind == "voice"

    def test_saved_messages_ref(self):
        adapter = _RecordingUserbot(me_id=777)
        ref = vt.saved_messages_chat(adapter)
        assert ref is not None
        assert ref.chat_id == "777" and ref.platform == "telegram"
        assert ref.title == "Saved Messages"

    def test_saved_messages_unknown_until_connected(self):
        adapter = _RecordingUserbot(me_id=None)
        adapter._me_id = None
        assert vt.saved_messages_chat(adapter) is None

    def test_artifact_defaults_to_saved_messages(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _RecordingUserbot(me_id=777)
        delivery = vt.deliver_voice_artifact(adapter, _tmp(".wav"))
        assert delivery.ok and delivery.voice_note
        (sent_chat, _media, _caption), = adapter.sent
        assert sent_chat.chat_id == "777"
        assert sent_chat.title == "Saved Messages"

    def test_degraded_file_fallback_on_userbot(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _RecordingUserbot(fail_kinds={"voice"})
        delivery = vt.send_voice_note(adapter, _chat("telegram", "9"),
                                      _tmp(".wav"))
        assert delivery.ok and delivery.degraded
        assert "voice rejected" in delivery.reason
        assert adapter.sent[-1][1].kind == "document"


class TestArtifactTargeting:
    def test_bot_surface_has_no_saved_messages(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _bot_adapter(_HttpClientSession())
        assert vt.saved_messages_chat(adapter) is None
        delivery = vt.deliver_voice_artifact(adapter, _tmp(".wav"))
        assert not delivery.ok
        assert "Saved Messages" in delivery.reason

    def test_artifact_with_explicit_chat_uses_it(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _RecordingUserbot(me_id=777)
        chat = _chat("telegram", "555")
        delivery = vt.deliver_voice_artifact(adapter, _tmp(".wav"),
                                             chat=chat)
        assert delivery.ok
        assert adapter.sent[0][0] is chat


# ── WhatsApp transport ───────────────────────────────────────────────────

class _WhatsAppStub:
    name = "whatsapp"

    def __init__(self):
        self.voice_calls: list[tuple] = []

    def send_voice(self, chat, audio_path, *, caption=""):
        self.voice_calls.append((chat, audio_path, caption))
        return SendResult(ok=True, platform="whatsapp", message_id="w1")

    def send_media(self, chat, media, *, caption=""):
        raise AssertionError("ptt surface must use send_voice")


class TestWhatsAppVoiceTransport:
    def test_ptt_send_voice(self, monkeypatch):
        _ogg(monkeypatch)
        stub = _WhatsAppStub()
        transport = vt.WhatsAppVoiceTransport(stub)
        delivery = transport.send_voice(_chat("whatsapp", "1"),
                                        _tmp(".wav"))
        assert delivery.ok and delivery.voice_note
        assert len(stub.voice_calls) == 1


# ── TTS pipeline wiring (runtime_voice) ──────────────────────────────────

def _mixin_with_gateway(adapter, platform="telegram-personal"):
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    mixin = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    gateway = SimpleNamespace(
        adapters={platform: adapter},
        _adapter_for=lambda p: adapter,
        send=lambda *a, **k: SendResult(ok=True, platform=platform),
        send_file=lambda *a, **k: SendResult(ok=True, platform=platform),
    )
    mixin.gateway = gateway
    return mixin, gateway


class TestRuntimeVoiceWiring:
    def test_send_tts_voice_note_returns_delivery(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _bot_adapter(_HttpClientSession())
        mixin, _gw = _mixin_with_gateway(adapter)
        delivery = mixin._send_tts_voice_note(_chat(), _tmp(".wav"))
        assert delivery.ok and delivery.voice_note

    def test_control_tts_degraded_is_labelled(self, monkeypatch):
        wav = _tmp(".wav")
        monkeypatch.setattr(
            "nomorals.tools.audio.wav_to_voice_ogg",
            lambda p: (wav, "audio/wav"))
        adapter = _bot_adapter(_HttpClientSession())
        mixin, gateway = _mixin_with_gateway(adapter)
        mixin.context = SimpleNamespace(
            tools=SimpleNamespace(call=lambda name, **kw: SimpleNamespace(
                ok=True, value={"path": wav, "engine": "xtts",
                                "bytes": 2048, "seconds": 2})),
            extras={"gateway": gateway})
        chat = _chat()
        mixin._ref_from_key = lambda key: chat
        out = mixin._control_tts("hello there", chat_key="telegram-personal:42")
        # Conversion unavailable → the gateway file route delivers it,
        # labelled as a file with the ffmpeg reason — never a voice note.
        assert "sent as file instead" in out
        assert "ffmpeg not available" in out
        assert "voice note sent" not in out

    def test_control_tts_voice_success(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _bot_adapter(_HttpClientSession())
        mixin, gateway = _mixin_with_gateway(adapter)
        mixin.context = SimpleNamespace(
            tools=SimpleNamespace(call=lambda name, **kw: SimpleNamespace(
                ok=True, value={"path": _tmp(".wav"), "engine": "xtts",
                                "bytes": 2048, "seconds": 2})),
            extras={"gateway": gateway})
        mixin._ref_from_key = lambda key: _chat()
        out = mixin._control_tts("hello there", chat_key="telegram-personal:42")
        assert "voice note sent" in out
        assert "DEGRADED" not in out

    def test_deliver_voice_note_labels(self, monkeypatch):
        _ogg(monkeypatch)
        adapter = _bot_adapter(_HttpClientSession())
        mixin, gateway = _mixin_with_gateway(adapter)
        mixin.context = SimpleNamespace(extras={"gateway": gateway})
        mixin._ref_from_key = lambda key: _chat()
        out = mixin._deliver_voice_note("telegram-personal:42",
                                        _tmp(".wav"), "hi", "xtts")
        assert "voice note" in out and "DEGRADED" not in out

    def test_deliver_voice_note_degraded_label(self, monkeypatch):
        wav = _tmp(".wav")
        monkeypatch.setattr(
            "nomorals.tools.audio.wav_to_voice_ogg",
            lambda p: (wav, "audio/wav"))
        adapter = _bot_adapter(_HttpClientSession())
        mixin, gateway = _mixin_with_gateway(adapter)
        mixin.context = SimpleNamespace(extras={"gateway": gateway})
        mixin._ref_from_key = lambda key: _chat()
        out = mixin._deliver_voice_note("telegram-personal:42", wav,
                                        "hi", "xtts")
        assert "DEGRADED" in out and "not a voice note" in out
