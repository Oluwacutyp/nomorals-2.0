"""Pass #13: social/chat layer — morning voice-note delivery parity.

The text briefing goes through ``Notifier.publish`` (canonical owner-target
resolution + the proactive/quiet gate). The voice note (``_deliver_audio``)
went straight to the gateway with its own naive ``platform:id`` parsing:

1. a bare numeric ``owner_chats`` entry (the documented, real-world config
   shape — ``NM_PARTNER_OWNER_CHATS=5478650254``) was silently skipped, so
   the morning voice note NEVER delivered while the text did;
2. the proactive/quiet gate was never consulted, so a chat in quiet mode
   (or with proactive off) had its text briefing held while the voice
   note still arrived.

Both are fixed by routing the voice path through the same
``resolve_owner_targets`` + ``proactive_allowed`` machinery as text.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest import mock

from nomorals.agents import morning_pulse


class _FakeResult:
    def __init__(self, ok=True):
        self.ok = ok


class _FakeAdapter:
    def __init__(self):
        self.sent = []

    def send_media(self, chat_ref, media):
        self.sent.append((chat_ref, media))
        return _FakeResult(True)


class _FakeGateway:
    def __init__(self, running=("telegram-bot",)):
        self._adapters = {p: _FakeAdapter() for p in running}
        self._running = set(running)
        self.dry_run = False

    def status(self):
        return {p: {"running_in_session": True} for p in self._running}

    def _adapter_for(self, plat):
        return self._adapters.get(plat)


def _make_context(owner_chats):
    settings = types.SimpleNamespace(
        partner=types.SimpleNamespace(owner_chats=owner_chats))
    return types.SimpleNamespace(settings=settings)


def _audio_file():
    # A REAL wav (Pass #72): delivery now goes through the shared
    # voice transport, whose conversion to OGG/Opus must genuinely
    # succeed for the voice-note assertions below to mean anything —
    # garbage bytes would (honestly) land as a labelled file instead.
    import wave
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 400)
    return path


def _run(owner_chats, gateway, proactive=True):
    ctx = _make_context(owner_chats)
    audio = _audio_file()
    try:
        with mock.patch("nomorals.agents.notifier.resolve_gateway",
                         return_value=gateway), \
             mock.patch("nomorals.partner.chat_modes.proactive_allowed",
                        return_value=proactive):
            return morning_pulse._deliver_audio(ctx, audio)
    finally:
        os.unlink(audio)


def _sent_media(gateway):
    out = []
    for adapter in gateway._adapters.values():
        out.extend(adapter.sent)
    return out


class DeliverAudioTests:
    pass


def test_bare_owner_chat_id_delivers_voice_note():
    """Bare numeric owner_chats (the documented config shape) must resolve
    like the text path does — not be silently skipped."""
    gw = _FakeGateway(running=("telegram-bot",))
    ok = _run("12345", gw, proactive=True)
    assert ok is True, "voice note not delivered for bare owner chat id"
    sent = _sent_media(gw)
    assert len(sent) == 1
    chat_ref, media = sent[0]
    assert chat_ref.chat_id == "12345"
    assert getattr(media, "voice_note", False) is True


def test_quiet_chat_skips_voice_note():
    """A chat with proactive off (quiet mode) must not get the voice note —
    the text briefing is already held for it."""
    gw = _FakeGateway(running=("telegram-bot",))
    ok = _run("telegram-bot:12345", gw, proactive=False)
    assert ok is False, "voice note bypassed the proactive/quiet gate"
    assert _sent_media(gw) == []


def test_explicit_platform_still_delivers_when_allowed():
    """Pin: explicit platform:id + proactive allowed → voice note goes out."""
    gw = _FakeGateway(running=("telegram-bot",))
    ok = _run("telegram-bot:12345", gw, proactive=True)
    assert ok is True
    sent = _sent_media(gw)
    assert len(sent) == 1
    assert sent[0][0].chat_id == "12345"


def test_missing_audio_file_returns_false():
    gw = _FakeGateway()
    ctx = _make_context("telegram-bot:12345")
    assert morning_pulse._deliver_audio(ctx, "/nonexistent/voice.wav") is False


def test_no_gateway_returns_false():
    ctx = _make_context("telegram-bot:12345")
    audio = _audio_file()
    try:
        with mock.patch("nomorals.agents.notifier.resolve_gateway",
                         return_value=None):
            assert morning_pulse._deliver_audio(ctx, audio) is False
    finally:
        os.unlink(audio)
