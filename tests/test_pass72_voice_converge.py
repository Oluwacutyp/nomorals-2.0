"""Pass #72: voice transport convergence.

One canonical converter (``tools.audio.wav_to_voice_ogg``), one voice
transport (``social.chat.voice_transport``), and one honesty invariant:
no surface may report a voice note delivered unless the transport
confirms a real voice bubble landed; a file fallback is reported as a
file.

Failing-first targets:
- the morning pulse hand-built ``MediaRef(voice_note=True)`` around its
  own converter — a WAV fallback went out flagged as a voice note
  (with an ``.ogg`` name), and on WhatsApp the flag is ignored entirely
  (kind="audio" never sets ptt), so a file landed while the pulse
  reported a voice note;
- ``PartnerRuntime._send_voice_reply`` bypassed the transport — and a
  *rejected* voice send silently dropped the reply part (no text or
  file fallback, unlike the synthesis-failure path);
- ``pingpong.to_ogg`` and ``tools.audio._to_ogg`` were separate
  conversion implementations;
- ``VoiceNoteStore.send_voice_note`` only stored, while the /vnote
  reply told the user "voice note sent";
- ``voice/bridge.py`` and ``integrations/voice_integration.py`` passed
  ``voice_note=True`` for unconverted (MP3) TTS output.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.core.errors import ToolError
from nomorals.social.chat.base import ChatRef, SendResult


def _run(coro):
    return asyncio.run(coro)


def _valid_wav() -> str:
    """A REAL, convertible WAV (0.05s of silence) — ffmpeg must succeed."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 400)
    return path


# ── (a) morning pulse through the transport ─────────────────────────────

class _PulseAdapter:
    """Adapter double with scripted per-kind outcomes.

    mode: "accept" (everything ok), "reject_voice" (voice sends fail,
    file sends ok), "reject_all".
    """

    name = "telegram-bot"

    def __init__(self, mode="accept"):
        self.mode = mode
        self.attempts = []   # (media, ok_returned)

    def send_media(self, chat, media):
        is_voice = bool(getattr(media, "voice_note", False)) \
            or getattr(media, "kind", "") == "voice"
        ok = self.mode == "accept" or (self.mode == "reject_voice"
                                        and not is_voice)
        self.attempts.append((media, ok))
        return SendResult(ok=ok, platform=self.name,
                          message_id="m1" if ok else "",
                          error="" if ok else "rejected")


class _PulseGateway:
    dry_run = False

    def __init__(self, adapter):
        self._adapter = adapter

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def _adapter_for(self, plat):
        return self._adapter


def _pulse_deliver(adapter, audio):
    from nomorals.agents import morning_pulse

    ctx = SimpleNamespace(settings=SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram-bot:12345")))
    gw = _PulseGateway(adapter)
    with mock.patch("nomorals.agents.notifier.resolve_gateway",
                    return_value=gw), \
            mock.patch("nomorals.partner.chat_modes.proactive_allowed",
                       return_value=True):
        return morning_pulse._deliver_audio(ctx, audio)


def _ok_sends(adapter):
    return [m for m, ok in adapter.attempts if ok]


def test_pulse_bubble_path_claims_voice_note():
    audio = _valid_wav()
    try:
        adapter = _PulseAdapter("accept")
        assert _pulse_deliver(adapter, audio) is True
        sent = _ok_sends(adapter)
        assert len(sent) == 1
        # The delivered media IS a voice bubble claim — and it is true:
        # the transport converted the wav and sent it as one.
        assert getattr(sent[0], "voice_note", False) is True
        assert sent[0].kind == "voice"
    finally:
        os.unlink(audio)


def test_pulse_voice_rejected_lands_labelled_file_not_voice_note():
    audio = _valid_wav()
    try:
        adapter = _PulseAdapter("reject_voice")
        assert _pulse_deliver(adapter, audio) is True
        sent = _ok_sends(adapter)
        assert len(sent) == 1
        # What actually landed is a plain file — nothing that succeeded
        # may carry the voice-note claim.
        assert getattr(sent[0], "voice_note", False) is not True
        assert sent[0].kind != "voice"
    finally:
        os.unlink(audio)


def test_pulse_conversion_impossible_falls_back_to_labelled_file():
    audio = _valid_wav()
    try:
        adapter = _PulseAdapter("accept")
        with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                        return_value=(audio, "audio/wav")):
            assert _pulse_deliver(adapter, audio) is True
        sent = _ok_sends(adapter)
        assert len(sent) == 1
        # The pulse's own labelled fallback: the original WAV, honest
        # name, no voice-note flag.
        assert getattr(sent[0], "voice_note", False) is not True
        assert sent[0].path == audio
        assert sent[0].name.endswith(".wav")
    finally:
        os.unlink(audio)


def test_pulse_nothing_reachable_returns_false():
    audio = _valid_wav()
    try:
        adapter = _PulseAdapter("reject_all")
        assert _pulse_deliver(adapter, audio) is False
        assert _ok_sends(adapter) == []
    finally:
        os.unlink(audio)


# ── (b) partner runtime voice reply through the transport ───────────────

class _RtAdapter:
    name = "local"

    def __init__(self, mode="accept"):
        self.mode = mode
        self.media = []

    def send_media(self, chat, media):
        self.media.append(media)
        ok = self.mode == "accept"
        return SendResult(ok=ok, platform=self.name, message_id="v1",
                          error="" if ok else "rejected")


class _RtGateway:
    dry_run = False

    def __init__(self, adapter):
        self._adapter = adapter
        self.sent = []

    def _adapter_for(self, platform):
        return self._adapter

    def typing(self, platform, chat, seconds=0.0, **kw):
        return True

    def send(self, platform, chat, text, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform=platform, message_id="t1")


def _voice_runtime(adapter):
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.social.chat.base import ChatMessage

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.gateway = _RtGateway(adapter)
    rt._typing_enabled = lambda chat: False
    rt.context = SimpleNamespace()
    msg = ChatMessage(chat=ChatRef(platform="local", chat_id="1"),
                      incoming=True, text="", meta={"voice_in": True})
    return rt, msg


def test_runtime_reply_uses_the_transport():
    import nomorals.social.chat.voice_transport as vt

    adapter = _RtAdapter("accept")
    rt, msg = _voice_runtime(adapter)
    real = vt.send_voice_note
    with mock.patch("nomorals.voice.pingpong.synthesize_voice_reply",
                    return_value="/tmp/pass72-fake.ogg"), \
            mock.patch.object(vt, "send_voice_note",
                              side_effect=real) as spy:
        assert rt._send_voice_reply(msg, ["hello voice"]) is True
    # The send went through the shared transport, with this adapter,
    # this chat, and the synthesized ogg.
    assert spy.call_count == 1
    args = spy.call_args[0]
    assert args[0] is adapter and args[2] == "/tmp/pass72-fake.ogg"
    # …and the bubble claim that landed is the transport's voice media.
    assert len(adapter.media) == 1
    assert adapter.media[0].kind == "voice"
    assert getattr(adapter.media[0], "voice_note", False) is True
    assert rt.gateway.sent == []


def test_runtime_reply_rejected_voice_falls_back_to_text():
    adapter = _RtAdapter("reject_all")
    rt, msg = _voice_runtime(adapter)
    with mock.patch("nomorals.voice.pingpong.synthesize_voice_reply",
                    return_value="/tmp/pass72-fake.ogg"):
        assert rt._send_voice_reply(msg, ["only part"]) is True
    # The transport delivered nothing (voice AND file rejected): the
    # part must fall back to text instead of vanishing silently.
    assert rt.gateway.sent == ["only part"]


# ── (c) converters delegate to the canonical implementation ─────────────

def test_pingpong_to_ogg_delegates():
    from nomorals.tools.audio import wav_to_voice_ogg
    from nomorals.voice import pingpong as pp

    wav = _valid_wav()
    try:
        expected, mime = wav_to_voice_ogg(wav)
        assert mime == "audio/ogg"
        assert pp.to_ogg(wav) == expected
    finally:
        os.unlink(wav)


def test_pingpong_to_ogg_uses_canonical_converter():
    from nomorals.voice import pingpong as pp

    with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                    return_value=("/sentinel.ogg", "audio/ogg")) as conv:
        assert pp.to_ogg("/tmp/whatever.wav") == "/sentinel.ogg"
    assert conv.call_count == 1


def test_pingpong_to_ogg_keeps_raise_contract():
    from nomorals.voice import pingpong as pp

    with mock.patch("subprocess.run",
                    side_effect=FileNotFoundError("no ffmpeg")):
        with pytest.raises(RuntimeError) as exc:
            pp.to_ogg("/tmp/x.wav")
    assert "ffmpeg" in str(exc.value)


def test_stt_to_ogg_delegates():
    from nomorals.tools import audio as audio_mod

    wav = _valid_wav()
    aac = wav[:-4] + ".aac"
    os.rename(wav, aac)
    try:
        with mock.patch.object(audio_mod, "wav_to_voice_ogg",
                               wraps=audio_mod.wav_to_voice_ogg) as conv:
            out = audio_mod._to_ogg(Path(aac))
        assert conv.call_count == 1
        assert out.suffix == ".ogg" and out.is_file()
    finally:
        os.unlink(aac)
        ogg = aac[:-4] + ".ogg"
        if os.path.exists(ogg):
            os.unlink(ogg)


def test_stt_to_ogg_keeps_toolerror_contract():
    from nomorals.tools import audio as audio_mod

    fd, bad = tempfile.mkstemp(suffix=".aac")
    os.write(fd, b"not-audio-at-all")
    os.close(fd)
    try:
        with pytest.raises(ToolError):
            audio_mod._to_ogg(Path(bad))
    finally:
        os.unlink(bad)


def test_pulse_converter_alias_delegates():
    from nomorals.agents import morning_pulse

    with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                    return_value=("/sentinel.ogg", "audio/ogg")) as conv:
        assert morning_pulse._to_voice_ogg("/tmp/x.wav") == (
            "/sentinel.ogg", "audio/ogg")
    assert conv.call_count == 1


# ── (d) the store records; it does not claim to send ────────────────────

def _store(tmp_path):
    from nomorals.social.voice_notes import VoiceNoteStore

    return VoiceNoteStore(
        db_path=str(tmp_path / "vn.db"), vault_dir=str(tmp_path / "vault"),
        stt=lambda p, lang: {"text": "the price is firm",
                             "backend": "mock-stt"})


def test_store_records_without_a_send_claim(tmp_path):
    from nomorals.social.voice_notes import VoiceNoteStore, control_vnote

    # The old name promised a delivery the store never performs.
    assert not hasattr(VoiceNoteStore, "send_voice_note")
    store = _store(tmp_path)
    audio = _valid_wav()
    try:
        note = store.store_voice_note("grp1", "Ada", audio, language="pcm")
    finally:
        os.unlink(audio)
    assert note is not None
    assert os.path.isfile(note.audio_path)  # the vault copy — the handle
    assert note.transcript == "the price is firm"
    # The chat surface must say "recorded", never "sent".
    ctx = SimpleNamespace(voice_note_store=store)
    audio2 = _valid_wav()
    try:
        out = control_vnote(f"send grp1 {audio2} pcm", context=ctx,
                            sender="Ada")
    finally:
        os.unlink(audio2)
    assert "recorded" in out
    assert "voice note sent" not in out


def test_stored_note_is_a_delivery_handle_for_the_transport(tmp_path):
    import nomorals.social.chat.voice_transport as vt

    store = _store(tmp_path)
    audio = _valid_wav()
    try:
        note = store.store_voice_note("grp1", "Ada", audio)
    finally:
        os.unlink(audio)
    assert note is not None
    adapter = _PulseAdapter("accept")
    with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                    return_value=(note.audio_path, "audio/ogg")):
        delivery = vt.send_voice_note(
            adapter, ChatRef(platform="telegram-bot", chat_id="1"),
            note.audio_path)
    # Only THIS — a caller driving the transport — may claim the bubble.
    assert delivery.ok and delivery.voice_note


# ── bridge / legacy integration: no voice flag on unconverted audio ─────

class _FakeEngine:
    def __init__(self, path):
        self._path = path

    def speak(self, text, voice_name=None, **kwargs):
        return {"path": self._path}


class _FakeTelegramClient:
    def __init__(self):
        self.calls = []

    async def send_file(self, chat_id, path, **kwargs):
        self.calls.append((chat_id, path, kwargs))
        return SimpleNamespace(id=1)


def test_bridge_telegram_converts_before_claiming_voice():
    from nomorals.voice.bridge import VoiceBridge

    wav = _valid_wav()
    telegram = SimpleNamespace(client=_FakeTelegramClient())
    bridge = VoiceBridge(_FakeEngine(wav), telegram=telegram)
    assert _run(bridge.send_voice_telegram(123, "hi")) is True
    assert len(telegram.client.calls) == 1
    _, sent_path, kwargs = telegram.client.calls[0]
    assert kwargs.get("voice_note") is True
    assert sent_path.endswith(".ogg")  # converted — not the raw wav


def test_bridge_telegram_refuses_when_conversion_impossible():
    from nomorals.voice.bridge import VoiceBridge

    wav = _valid_wav()
    telegram = SimpleNamespace(client=_FakeTelegramClient())
    bridge = VoiceBridge(_FakeEngine(wav), telegram=telegram)
    try:
        with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                        return_value=(wav, "audio/wav")):
            assert _run(bridge.send_voice_telegram(123, "hi")) is False
        assert telegram.client.calls == []
    finally:
        if os.path.exists(wav):
            os.unlink(wav)


def test_bridge_whatsapp_converts_before_ptt():
    from nomorals.voice.bridge import VoiceBridge

    wav = _valid_wav()
    sent = []

    class _WA:
        def send_voice(self, chat, path, caption=""):
            sent.append(path)
            return SendResult(ok=True, platform="whatsapp", message_id="w1")

    bridge = VoiceBridge(_FakeEngine(wav), whatsapp=_WA())
    ref = ChatRef(platform="whatsapp", chat_id="234@x")
    assert _run(bridge.send_voice_whatsapp(ref, "hi")) is True
    assert len(sent) == 1 and sent[0].endswith(".ogg")


def test_bridge_broadcast_reports_failure_when_conversion_impossible():
    from nomorals.voice.bridge import VoiceBridge

    wav = _valid_wav()
    telegram = SimpleNamespace(client=_FakeTelegramClient())
    bridge = VoiceBridge(_FakeEngine(wav), telegram=telegram)
    ref = ChatRef(platform="whatsapp", chat_id="234@x")
    try:
        with mock.patch("nomorals.tools.audio.wav_to_voice_ogg",
                        return_value=(wav, "audio/wav")):
            out = _run(bridge.broadcast_voice(
                [("telegram", 123), ("whatsapp", ref)], "hi"))
        assert out["sent"] == 0 and out["failed"] == 2
        assert all(not r["ok"] for r in out["receipts"])
        assert telegram.client.calls == []
    finally:
        if os.path.exists(wav):
            os.unlink(wav)
