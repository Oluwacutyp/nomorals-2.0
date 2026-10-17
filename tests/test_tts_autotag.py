"""Tests for automatic TTS tag application (VoiceTagger, tools/audio.py).

The owner writes plain text; VoiceTagger inserts engine-appropriate tags
so pacing/expression happens without manual [pause] / [laugh] markup.
Explicit user tags are preserved and never duplicated; real words are
never eaten.
"""
from __future__ import annotations

import pytest

from nomorals.tools.audio import VoiceTagger, tts


# ── native-tag engines (Chatterbox / Bark) ────────────────────────────────

def test_native_sentence_pause():
    tagged, applied = VoiceTagger("chatterbox").tag("Hello. How are you?")
    assert tagged == "Hello. [pause:250] How are you?"
    assert applied == ["pause×1"]


def test_native_slow_profile_uses_profile_lengths():
    tagged, _ = VoiceTagger("chatterbox", pace="slow").tag("Hello. World")
    assert tagged == "Hello. [pause:450] World"


def test_native_tight_inserts_no_pause_tags():
    tagged, applied = VoiceTagger("chatterbox", pace="tight").tag(
        "Hello. World, um ok... haha")
    assert "[pause" not in tagged
    assert "..." in tagged  # original punctuation stays on tight
    assert "[laugh]" in tagged  # laughs are not pace-dependent
    assert applied == ["laugh×1"]


def test_native_laugh_after_cue_words_kept():
    tagged, applied = VoiceTagger("chatterbox").tag("haha that is funny")
    assert tagged == "haha [laugh] that is funny"
    assert applied == ["laugh×1"]


def test_native_laugh_emoji_becomes_tag():
    tagged, applied = VoiceTagger("chatterbox").tag("that's funny 😂")
    assert "[laugh]" in tagged
    assert "😂" not in tagged
    assert applied == ["laugh×1"]


def test_native_ellipsis_and_emdash_pause():
    tagged, applied = VoiceTagger("chatterbox").tag("Wait... what — really?")
    assert tagged == "Wait [pause:250] what [pause:120] really?"
    assert applied == ["pause×2"]


def test_native_comma_and_hesitation_pause():
    tagged, _ = VoiceTagger("chatterbox").tag("um hello")
    assert tagged == "um [pause:120] hello"


def test_native_paragraph_pause():
    tagged, applied = VoiceTagger("chatterbox").tag(
        "First para.\n\nSecond para.")
    # Paragraph 500 + sentence 250 merge into one 750ms break.
    assert tagged == "First para. [pause:750] Second para."
    assert applied == ["pause×1"]


def test_native_abbreviation_no_split():
    tagged, _ = VoiceTagger("chatterbox").tag(
        "Mr. Smith went home. He left.")
    assert "Mr. Smith" in tagged
    assert "Mr. [pause" not in tagged
    assert "home. [pause:250] He left." in tagged


def test_native_dedupe_merges_adjacent_pauses():
    tagged, applied = VoiceTagger("chatterbox").tag("um... wow")
    assert tagged.count("[pause") == 1
    assert tagged == "um [pause:370] wow"  # 120 hesitation + 250 ellipsis
    assert applied == ["pause×1"]


def test_explicit_pause_preserved_never_duplicated():
    tagged, applied = VoiceTagger("chatterbox").tag("Hello. [pause:500] World")
    assert tagged == "Hello. [pause:500] World"
    assert tagged.count("[pause") == 1
    assert applied == []  # the explicit tag is not counted as applied


def test_explicit_laugh_preserved_never_duplicated():
    tagged, applied = VoiceTagger("chatterbox").tag("haha [laugh] funny")
    assert tagged == "haha [laugh] funny"
    assert tagged.count("[laugh]") == 1
    assert applied == []


def test_explicit_pause_at_start_preserved():
    tagged, _ = VoiceTagger("chatterbox").tag("[pause:300] Hello. World")
    assert tagged.startswith("[pause:300] Hello.")
    assert tagged.count("[pause") == 2  # explicit + one auto sentence pause


def test_unknown_bracket_text_preserved():
    tagged, _ = VoiceTagger("chatterbox").tag("see [section 4] for details")
    assert "[section 4]" in tagged


def test_markdown_emphasis_markers_stripped_words_kept():
    tagged, _ = VoiceTagger("chatterbox").tag("Hello *world* and **bold**")
    assert "*" not in tagged
    assert "world" in tagged and "bold" in tagged


def test_words_never_eaten():
    tagged, _ = VoiceTagger("chatterbox").tag(
        "NASA launched at 12:30. See https://example.com for details. "
        "U.S. Army. my_var and 2*3 stay.")
    for word in ("NASA", "12:30", "https://example.com", "U.S.",
                 "my_var", "2*3"):
        assert word in tagged


def test_no_misfire_on_er_or_acronym():
    tagged, _ = VoiceTagger("chatterbox").tag("Go to the ER now.")
    assert "[pause" not in tagged  # "ER" is not hesitation


def test_caps_not_auto_tagged_as_shouting():
    tagged, _ = VoiceTagger("chatterbox").tag("NASA launched.")
    assert "[shouting]" not in tagged
    assert "NASA" in tagged


def test_empty_and_plain_text():
    assert VoiceTagger("chatterbox").tag("") == ("", [])
    assert VoiceTagger("chatterbox").tag("just words") == ("just words", [])


def test_bad_pace_rejected():
    with pytest.raises(Exception, match="unknown pace"):
        VoiceTagger("chatterbox", pace="bogus")


def test_unknown_engine_defaults_to_chunked():
    tagged, applied = VoiceTagger("mystery_engine").tag("haha hi. um ok")
    assert "[laugh]" not in tagged  # chunked: no sound tags
    assert "haha" in tagged
    assert applied == ["pause×1"]  # hesitation only


# ── chunked engines (espeak-ng / pico / say / edge / API / XTTS) ──────────

def test_chunked_no_sentence_pause_inserted():
    # The pacing chunker already handles sentence pauses — the tagger
    # must not duplicate them.
    tagged, applied = VoiceTagger("espeak_ng").tag("Hello. How are you?")
    assert tagged == "Hello. How are you?"
    assert applied == []


def test_chunked_hesitation_pause_inserted():
    tagged, applied = VoiceTagger("espeak_ng").tag("um hello")
    assert tagged == "um [pause:120] hello"
    assert applied == ["pause×1"]


def test_chunked_no_laugh_tag_words_kept():
    tagged, applied = VoiceTagger("espeak_ng").tag("haha that is funny")
    assert "[laugh]" not in tagged
    assert "haha" in tagged
    assert applied == []


def test_chunked_laugh_emoji_removed_not_spoken():
    tagged, _ = VoiceTagger("espeak_ng").tag("that's funny 😂")
    assert "😂" not in tagged
    assert "funny" in tagged


def test_chunked_explicit_tags_preserved():
    tagged, _ = VoiceTagger("espeak_ng").tag("Hello [pause:300] world")
    assert tagged == "Hello [pause:300] world"


# ── tts() wiring: auto_tags ON by default, opt-out ─────────────────────────

def test_tts_auto_tags_on_by_default(tmp_path, monkeypatch):
    import subprocess

    seen: dict = {}

    def _fake_run(argv, capture_output=False, timeout=None, check=False):
        seen["argv"] = list(argv)
        out = argv[argv.index("-w") + 1]
        with open(out, "wb") as f:
            f.write(b"RIFFfake")

        class _P:
            returncode = 0
            stderr = b""
            stdout = b""

        return _P()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("nomorals.tools.audio.detect_tts_engines",
                        lambda: ["espeak_ng"])
    res = tts("um hello", engine="espeak_ng", out_dir=str(tmp_path))
    assert res["auto_tags"] is True
    assert res["auto_tags_applied"] == ["pause×1"]
    ssml = seen["argv"][-1]
    assert '<break time="120ms"/>' in ssml  # hesitation pause reached espeak


def test_tts_auto_tags_off(tmp_path, monkeypatch):
    import subprocess

    seen: dict = {}

    def _fake_run(argv, capture_output=False, timeout=None, check=False):
        seen["argv"] = list(argv)
        out = argv[argv.index("-w") + 1]
        with open(out, "wb") as f:
            f.write(b"RIFFfake")

        class _P:
            returncode = 0
            stderr = b""
            stdout = b""

        return _P()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("nomorals.tools.audio.detect_tts_engines",
                        lambda: ["espeak_ng"])
    res = tts("um hello", engine="espeak_ng", out_dir=str(tmp_path),
              auto_tags=False)
    assert res["auto_tags"] is False
    assert res["auto_tags_applied"] == []
    ssml = seen["argv"][-1]
    assert '<break time="120ms"/>' not in ssml


# ── /tts handler: auto= flag + applied note ───────────────────────────────

class _FakeOutcome:
    def __init__(self, ok=True, value=None, error=None):
        self.ok = ok
        self.value = value or {}
        self.error = error


class _FakeTools:
    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def call(self, name, *args, **kwargs):
        self.calls.append(((name, args), kwargs))
        return _FakeOutcome(True, {
            "path": "/tmp/fake-tts.mp3", "engine": "espeak_ng",
            "bytes": 2048, "seconds": 1.2,
            "dropped_sounds": [], "pauses_ms": 120,
            "pauses_dropped": False, "auto_tags_applied": ["pause×1"],
        })


class _FakeContext:
    def __init__(self):
        self.tools = _FakeTools()
        self.extras = {}


class _FakeChat:
    platform = "telegram"


class _FakeSendResult:
    ok = True


class _FakeGateway:
    def send_file(self, platform, chat, path, caption=""):
        return _FakeSendResult()


def _handler():
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    obj.context = _FakeContext()
    obj.gateway = _FakeGateway()
    obj._ref_from_key = lambda key: _FakeChat()
    return obj


def test_control_tts_auto_defaults_on_and_reports_applied():
    obj = _handler()
    reply = obj._control_tts("um hello", chat_key="telegram:1")
    (_name, _args), kwargs = obj.context.tools.calls[0]
    assert kwargs["auto_tags"] is True
    assert "auto-tags applied: pause×1" in reply


def test_control_tts_auto_off():
    obj = _handler()
    obj._control_tts("auto=off um hello", chat_key="telegram:1")
    (_name, _args), kwargs = obj.context.tools.calls[0]
    assert kwargs["auto_tags"] is False


def test_control_tts_auto_bad_value_is_usage():
    obj = _handler()
    assert "usage" in obj._control_tts("auto=maybe hello", chat_key="telegram:1")
    assert obj.context.tools.calls == []


def test_tts_help_documents_auto_tagging():
    from nomorals.social.chat.control import COMMAND_DETAILS

    tts_detail = COMMAND_DETAILS["tts"]
    assert "auto" in tts_detail["usage"]
    assert "Auto-tagging" in tts_detail["what"]


# ── UniversalTTS wiring: opt-in, backend capability drives it ─────────────

def test_universal_apply_auto_tags_native_backend():
    from types import SimpleNamespace

    from nomorals.voice.tts import UniversalTTS

    engine = UniversalTTS.__new__(UniversalTTS)
    backend = SimpleNamespace(name="chatterbox", supports_native_tags=True)
    out = engine._apply_auto_tags("haha hi. How are you?", backend)
    assert "[laugh]" in out
    assert "[pause:250]" in out


def test_universal_apply_auto_tags_chunked_backend():
    from types import SimpleNamespace

    from nomorals.voice.tts import UniversalTTS

    engine = UniversalTTS.__new__(UniversalTTS)
    backend = SimpleNamespace(name="xtts", supports_native_tags=False)
    out = engine._apply_auto_tags("haha hi. um ok", backend)
    assert "[laugh]" not in out  # XTTS has no native tags
    assert "haha" in out
    assert "[pause:120]" in out  # hesitation still applied


def test_universal_apply_auto_tags_never_breaks():
    from types import SimpleNamespace

    from nomorals.voice.tts import UniversalTTS

    engine = UniversalTTS.__new__(UniversalTTS)
    backend = SimpleNamespace(name="chatterbox", supports_native_tags=True)
    # None text must not raise — falls back to empty text.
    assert engine._apply_auto_tags(None, backend) == ""
