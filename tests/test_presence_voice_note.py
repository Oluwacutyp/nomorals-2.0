"""Presence guard (Devon Personal) + /tts voice-note delivery.

Issue 1 — replies in personal (restricted) chats must feel like a person
wrote them, not an assistant: assistant/task-framed sentences are stripped
by the presence guard, and the connector-capabilities block (which frames
Devon as a service with "services you can link, use, and act on") is
omitted from restricted-chat prompts. Owner chats are untouched — there
Devon IS the owner's assistant and task framing is correct.

Issue 2 — /tts sends as a voice note (OGG/Opus, voice-note type), not a
WAV file attachment. file=on keeps the old file behavior; a failed
conversion/send falls back to a file send labelled honestly as a file.
"""

from __future__ import annotations

import types

import pytest

from nomorals.partner.style import strip_assistant_framing


# ── Issue 1: presence guard ─────────────────────────────────────────────

# Every assistant-style reply from the owner's screenshot (gf chat).
SCREENSHOT_REPLIES = [
    "hey, what's the clip? if you want me to ID it, could you upload the audio file here?",
    "gotcha, what do you need?",
    "if you've got a file or a clip, drop it here and i'll take a look.",
    "i can only reply in text, not voice notes.",
]


@pytest.mark.parametrize("reply", SCREENSHOT_REPLIES)
def test_screenshot_assistant_replies_stripped(reply: str) -> None:
    """Each screenshot reply is pure assistant framing -> nothing left."""
    assert strip_assistant_framing(reply) == ""


def test_false_limitation_stripped_but_human_part_kept() -> None:
    draft = "I can only reply in text, not voice notes. Lol that was funny"
    # The false-limitation sentence goes; the human sentence survives.
    assert strip_assistant_framing(draft) == "Lol that was funny"


def test_human_sentences_preserved() -> None:
    human = "lol yeah that was funny. man wahala sha. you better no allow ma vex"
    assert strip_assistant_framing(human) == human
    assert strip_assistant_framing("k. mhm. wym?") == "k. mhm. wym?"


def test_assistant_conditional_stripped() -> None:
    assert strip_assistant_framing(
        "If you want me to look into it, just say so. Anyway how was your day"
    ) == "Anyway how was your day"


def test_no_assistant_framing_is_identity() -> None:
    text = "hey, how was your day? mine was long."
    assert strip_assistant_framing(text) == text


def test_capabilities_block_dropped_for_restricted_chats() -> None:
    from nomorals.partner.context import PartnerContextBuilder
    from nomorals.partner.mood import MoodEngine
    from nomorals.partner.persona import default_persona
    from nomorals.partner.relationship import Relationship

    builder = PartnerContextBuilder()
    persona = default_persona()
    mood = MoodEngine(persona.baselines)
    rel = Relationship()
    owner_system = builder.build(
        persona=persona, mood=mood, relationship=rel,
        include_capabilities=True,
    )
    personal_system = builder.build(
        persona=persona, mood=mood, relationship=rel,
        include_capabilities=False,
    )
    assert "Services you can link" in owner_system.content
    assert "Services you can link" not in personal_system.content
    # The persona block (human presence) survives in both.
    assert "real person" in personal_system.content


# ── Devon Personal identity (Q&A) — life anchors ───────────────────────


def test_life_anchors_carry_devon_personal_identity() -> None:
    """The protected anchor block is the Q&A identity, not a bio dump."""
    from nomorals.partner.context import PartnerContextBuilder

    block = PartnerContextBuilder._life_anchors_block()
    assert "freelance computer programming" in block
    assert "Gunnison, Colorado" in block
    assert "ICELab" in block
    assert "Hartman Rocks" in block
    assert "never dump" in block
    assert "never list them" in block


def test_life_anchors_present_in_built_prompt_and_protected() -> None:
    from nomorals.partner.context import (
        SHRINK_ORDER,
        PartnerContextBuilder,
    )
    from nomorals.partner.mood import MoodEngine
    from nomorals.partner.persona import default_persona
    from nomorals.partner.relationship import Relationship

    assert "life" not in SHRINK_ORDER  # protected: never shrunk away
    builder = PartnerContextBuilder()
    persona = default_persona()
    system = builder.build(
        persona=persona,
        mood=MoodEngine(persona.baselines),
        relationship=Relationship(),
        include_capabilities=False,  # restricted/personal chat shape
    )
    assert "Gunnison, Colorado" in system.content
    assert "freelance computer programming" in system.content
    assert "Hartman Rocks" in system.content


def test_devon_personal_facts_cover_identity() -> None:
    from nomorals.partner.background import DEVON_PERSONAL_FACTS

    by_id = {f.id: f.text for f in DEVON_PERSONAL_FACTS}
    assert "Lake Wales" in by_id["personal.mom"]
    assert "single about two years" in by_id["personal.status"]
    assert "Huevos Rancheros" in by_id["personal.food_favorites"]
    assert "Sledgehammer" in by_id["personal.food_favorites"]
    assert "Just Peachy" in by_id["personal.food_favorites"]
    assert "coding bootcamp" in by_id["personal.bootcamp"]
    assert "trust, honesty" in by_id["personal.values"]
    # The anchors are the constant facts — exactly the public identity.
    anchors = {f.id for f in DEVON_PERSONAL_FACTS if f.constant}
    assert anchors == {"personal.work", "personal.place", "personal.outdoors"}


def test_gated_background_serves_devon_personal_facts() -> None:
    from nomorals.partner.background import BackgroundSelector

    sel = BackgroundSelector()
    lines = sel.context("tell me about your mom", user_in_us=True,
                        romantic=False)
    assert any("Lake Wales" in line for line in lines)
    lines = sel.context("what do you do for work", user_in_us=True,
                        romantic=False)
    assert any("freelance" in line.lower() for line in lines)


def test_gated_background_still_closed_for_non_us_non_romantic() -> None:
    """The gate is unchanged: deeper facts don't leak to strangers."""
    from nomorals.partner.background import BackgroundSelector

    sel = BackgroundSelector()
    assert sel.context("tell me about your mom", user_in_us=False,
                       romantic=False) == []


def test_wav_to_voice_ogg_passthrough_for_ogg() -> None:
    from nomorals.tools.audio import wav_to_voice_ogg

    assert wav_to_voice_ogg("voice.ogg") == ("voice.ogg", "audio/ogg")
    assert wav_to_voice_ogg("voice.opus") == ("voice.opus", "audio/ogg")


def test_wav_to_voice_ogg_without_ffmpeg_returns_original_honestly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from nomorals.tools.audio import wav_to_voice_ogg

    monkeypatch.setattr("shutil.which", lambda _name: None)
    wav = tmp_path / "tts-1.wav"
    wav.write_bytes(b"fake wav")
    assert wav_to_voice_ogg(str(wav)) == (str(wav), "audio/wav")


def test_wav_to_voice_ogg_converts_with_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import os
    import shutil
    from unittest import mock

    from nomorals.tools.audio import wav_to_voice_ogg

    wav = tmp_path / "tts-1.wav"
    wav.write_bytes(b"fake wav")
    ogg = tmp_path / "tts-1.ogg"

    def _fake_run(cmd, **kwargs):
        ogg.write_bytes(b"fake ogg")
        return mock.Mock(returncode=0)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(os.path, "isfile", lambda p: str(p) == str(ogg))
    assert wav_to_voice_ogg(str(wav)) == (str(ogg), "audio/ogg")


class _FakeOutcome:
    def __init__(self, value):
        self.ok = True
        self.value = value


class _FakeTools:
    def __init__(self, info):
        self._info = info

    def call(self, name, **kwargs):
        assert name == "speak"
        return _FakeOutcome(dict(self._info))


class _FakeResult:
    def __init__(self, ok=True, error=""):
        self.ok = ok
        self.error = error


class _FakeAdapter:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent: list[dict] = []

    def send_media(self, chat, media):
        self.sent.append({"chat": chat, "media": media})
        return _FakeResult(ok=self.ok, error="" if self.ok else "boom")


class _FakeGateway:
    def __init__(self, adapter=None):
        self._adapter = adapter
        self.files_sent: list[dict] = []

    def _adapter_for(self, platform):
        return self._adapter

    def send_file(self, platform, chat, path, *, caption=""):
        self.files_sent.append({"path": path, "caption": caption})
        return _FakeResult(ok=True)


def _tts_mixin(*, adapter=None, wav_path="/tmp/tts-1.wav"):
    """A RuntimeVoiceMixin driven by fakes (no real runtime)."""
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = object.__new__(RuntimeVoiceMixin)
    obj.context = types.SimpleNamespace(
        tools=_FakeTools({"path": wav_path, "engine": "espeak_ng",
                          "bytes": 50400, "seconds": 3.2}))
    obj.gateway = _FakeGateway(adapter=adapter)
    obj._ref_from_key = lambda key: types.SimpleNamespace(
        platform="whatsapp", chat_id="123", key=key)
    return obj


def test_tts_sends_voice_note_not_file(monkeypatch, tmp_path) -> None:
    import os
    from unittest import mock

    from nomorals.tools import audio as audio_mod

    wav = tmp_path / "tts-1.wav"
    wav.write_bytes(b"fake wav")
    ogg = tmp_path / "tts-1.ogg"
    adapter = _FakeAdapter(ok=True)
    obj = _tts_mixin(adapter=adapter, wav_path=str(wav))

    def _fake_run(cmd, **kwargs):
        ogg.write_bytes(b"fake ogg")
        return mock.Mock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda _n: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(os.path, "isfile", lambda p: str(p) == str(ogg))
    reply = obj._control_tts("hello there", chat_key="whatsapp:123")
    assert "voice note sent" in reply
    assert len(adapter.sent) == 1
    media = adapter.sent[0]["media"]
    assert media.kind == "voice"
    assert media.mime == "audio/ogg"
    assert getattr(media, "voice_note", False) is True
    assert media.path == str(ogg)
    # No file send happened.
    assert obj.gateway.files_sent == []


def test_tts_file_on_forces_file_send() -> None:
    adapter = _FakeAdapter(ok=True)
    obj = _tts_mixin(adapter=adapter)
    reply = obj._control_tts("file=on hello there", chat_key="whatsapp:123")
    assert "spoken" in reply
    assert "voice note" not in reply
    assert adapter.sent == []
    assert len(obj.gateway.files_sent) == 1


def test_tts_voice_note_failure_falls_back_to_file_honestly(monkeypatch) -> None:
    """No ffmpeg -> conversion unavailable -> file send LABELLED as file."""
    monkeypatch.setattr("shutil.which", lambda _n: None)
    adapter = _FakeAdapter(ok=True)
    obj = _tts_mixin(adapter=adapter)
    reply = obj._control_tts("hello there", chat_key="whatsapp:123")
    assert "sent as file instead" in reply
    assert "ffmpeg not available" in reply
    assert adapter.sent == []
    assert len(obj.gateway.files_sent) == 1
