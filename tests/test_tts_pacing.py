"""Tests for /tts pacing + expressive tags (tools/audio.py).

/tts goes through the `speak` tool -> tts() (espeak/pico/say/edge-tts/
openai_compat), NOT the neural UniversalTTS engine. These engines have no
native tag support, so tags must never reach them literally.
"""
from __future__ import annotations

import io
import wave

import pytest

from nomorals.core.errors import ToolError
from nomorals.tools.audio import (
    chunk_tts_text,
    tts,
    _concat_wav_chunks,
    _fold_empty_chunks,
)


def _wav_bytes(frames: int = 2400, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


def _speakable(chunks):
    return " ".join(t for t, _ in chunks if t.strip())


# ── chunking ────────────────────────────────────────────────────────────────

def test_sound_tag_dropped_never_spoken():
    chunks, dropped = chunk_tts_text("hello [laugh] world")
    assert dropped == ["laugh"]
    assert _speakable(chunks) == "hello world"
    assert "[laugh]" not in _speakable(chunks)


def test_emotion_tag_stripped():
    chunks, dropped = chunk_tts_text("[happy] hello there")
    assert dropped == []
    assert _speakable(chunks) == "hello there"


def test_pause_tag_is_silence():
    chunks, _ = chunk_tts_text("hello [pause:300] world")
    assert chunks == [("hello", 300), ("world", 0)]


def test_bare_pause_defaults_500ms():
    chunks, _ = chunk_tts_text("[pause] hello")
    assert chunks == [("", 500), ("hello", 0)]


def test_sentence_pause_natural():
    chunks, _ = chunk_tts_text("hello. world")
    assert chunks == [("hello.", 250), ("world", 0)]


def test_sentence_pause_tight_is_legacy():
    chunks, _ = chunk_tts_text("hello. world", pace="tight")
    assert chunks == [("hello. world", 0)]


def test_sentence_pause_slow():
    chunks, _ = chunk_tts_text("hello. world", pace="slow")
    assert chunks == [("hello.", 450), ("world", 0)]


def test_comma_pause():
    chunks, _ = chunk_tts_text("wait, stop", pace="natural")
    assert chunks == [("wait,", 120), ("stop", 0)]


def test_unknown_bracket_text_preserved():
    chunks, dropped = chunk_tts_text("[see appendix] hello")
    assert dropped == []
    assert "[see appendix]" in _speakable(chunks)


def test_abbreviation_no_split():
    chunks, _ = chunk_tts_text("Mr. Smith went home.")
    assert _speakable(chunks) == "Mr. Smith went home."
    assert len(chunks) == 1


def test_only_tags_no_speakable():
    chunks, dropped = chunk_tts_text("[laugh]")
    assert chunks == []
    assert dropped == ["laugh"]


def test_invalid_pace():
    with pytest.raises(ToolError):
        chunk_tts_text("hello", pace="warp")


# ── fold empty chunks ───────────────────────────────────────────────────────

def test_fold_leading_pause():
    assert _fold_empty_chunks([("", 500), ("hello", 0)]) == ([("hello", 0)], 500)


def test_fold_mid_pause_joins_previous():
    assert _fold_empty_chunks([("hello", 100), ("", 300), ("world", 0)]) == (
        [("hello", 400), ("world", 0)], 0)


def test_fold_all_empty():
    assert _fold_empty_chunks([("", 200), ("", 300)]) == ([], 500)


# ── wav concat ──────────────────────────────────────────────────────────────

def test_concat_wav_with_silence():
    a = _wav_bytes(2400)  # 0.1s
    b = _wav_bytes(4800)  # 0.2s
    out = _concat_wav_chunks([a, b], [500, 0], 24000, leading_silence_ms=200)
    with wave.open(io.BytesIO(out), "rb") as w:
        # 0.2s leading + 0.1s + 0.5s gap + 0.2s = 1.0s = 24000 frames
        assert w.getnframes() == 24000
        assert w.getframerate() == 24000


# ── tts() with a fake API client ────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, ok=True, body=b"", status=200, text=""):
        self.ok = ok
        self.body = body
        self.status = status
        self.text = text


class _FakeClient:
    """Records payloads; returns a tiny valid WAV for wav requests, mp3 bytes otherwise."""

    def __init__(self, reject_wav: bool = False):
        self.payloads: list[dict] = []
        self.reject_wav = reject_wav

    def __init_subclass__(cls, **kwargs):
        pass

    def __call__(self, timeout=None):
        return self

    def post_json(self, url, payload, headers=None):
        self.payloads.append(dict(payload))
        if payload.get("response_format") == "wav":
            if self.reject_wav:
                raise ToolError("TTS API 400: response_format not supported")
            return _FakeResponse(ok=True, body=_wav_bytes())
        return _FakeResponse(ok=True, body=b"fake-mp3-bytes")


@pytest.fixture
def fake_api(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("nomorals.core.http.HttpClient", client)
    return client


def test_tts_api_strips_tags_and_inserts_pauses(tmp_path, fake_api):
    res = tts("hello [laugh] world. [pause:300] goodbye",
              engine="openai_compat", base_url="https://example.test/v1",
              api_key="k", out_dir=str(tmp_path))
    assert res["engine"] == "openai_compat"
    assert res["format"] == "wav"  # chunked path (pauses > 0)
    assert res["dropped_sounds"] == ["laugh"]
    # sentence pause 250 + explicit 300 = 550
    assert res["pauses_ms"] == 550
    assert not res["pauses_dropped"]
    for payload in fake_api.payloads:
        assert "[laugh]" not in payload["input"]
        assert "[pause" not in payload["input"]
    assert res["path"] and __import__("os").path.exists(res["path"])


def test_tts_api_speed_passed(tmp_path, fake_api):
    res = tts("hello", engine="openai_compat", base_url="https://example.test/v1",
              api_key="k", out_dir=str(tmp_path), speed=1.5)
    assert res["format"] == "mp3"  # no pauses -> single-shot fast path
    assert fake_api.payloads[-1]["speed"] == 1.5


def test_tts_api_wav_rejected_falls_back(tmp_path, monkeypatch):
    client = _FakeClient(reject_wav=True)
    monkeypatch.setattr("nomorals.core.http.HttpClient", client)
    res = tts("hello. [pause:300] world", engine="openai_compat",
              base_url="https://example.test/v1", api_key="k",
              out_dir=str(tmp_path))
    assert res["format"] == "mp3"
    assert res["pauses_dropped"] is True
    assert res["pauses_ms"] == 0
    # Fallback call carried the stripped text, no tags.
    assert "[pause" not in client.payloads[-1]["input"]


def test_tts_only_tags_rejected(tmp_path, fake_api):
    with pytest.raises(ToolError, match="only tags"):
        tts("[laugh]", engine="openai_compat", base_url="https://example.test/v1",
            api_key="k", out_dir=str(tmp_path))


# ── espeak-ng SSML ──────────────────────────────────────────────────────────

def test_espeak_ssml_carries_breaks():
    from nomorals.tools.audio import _espeak_ssml

    ssml = _espeak_ssml([("hello.", 250), ("world", 0), ("", 0)])
    assert ssml.startswith("<speak")
    assert '<break time="250ms"/>' in ssml
    assert "hello." in ssml and "world" in ssml


def test_espeak_ssml_escapes_text():
    from nomorals.tools.audio import _espeak_ssml

    ssml = _espeak_ssml([("a <b> & c", 0)])
    assert "a &lt;b&gt; &amp; c" in ssml


def test_tts_espeak_uses_ssml_and_gap(tmp_path, monkeypatch):
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
    res = tts("hello. [pause:300] world [laugh]",
              engine="espeak_ng", out_dir=str(tmp_path),
              speed=1.25, pace="natural", gap_ms=30)
    argv = seen["argv"]
    assert "-m" in argv  # SSML mode
    assert "-s" in argv and argv[argv.index("-s") + 1] == "219"  # 175*1.25
    assert "-g" in argv and argv[argv.index("-g") + 1] == "3"  # 30ms -> 3 units
    ssml = argv[-1]
    # sentence pause 250 + explicit [pause:300] merge into one 550ms break
    assert '<break time="550ms"/>' in ssml
    assert "[laugh]" not in ssml
    assert res["dropped_sounds"] == ["laugh"]
    assert res["pauses_ms"] == 550
    assert res["gap_ms"] == 30
    assert res["gap_ignored"] is False


def test_tts_gap_ignored_off_espeak(tmp_path, fake_api):
    res = tts("hello", engine="openai_compat", base_url="https://example.test/v1",
              api_key="k", out_dir=str(tmp_path), gap_ms=30)
    assert res["gap_ignored"] is True

def test_edge_ssml_carries_breaks_and_rate(tmp_path, monkeypatch):
    import asyncio
    import sys
    import types

    seen: dict = {}

    class _FakeComm:
        def __init__(self, text, voice=None):
            seen["text"] = text
            seen["voice"] = voice

        def save(self, path):
            async def _go():
                with open(path, "wb") as f:
                    f.write(b"fake-mp3")
            return _go()

    fake_mod = types.ModuleType("edge_tts")
    fake_mod.Communicate = _FakeComm
    monkeypatch.setitem(sys.modules, "edge_tts", fake_mod)
    res = tts("hello. [pause:300] world [laugh]",
              engine="edge_tts", out_dir=str(tmp_path), speed=1.25)
    assert res["engine"] == "edge_tts"
    ssml = seen["text"]
    assert ssml.startswith("<speak")
    # sentence pause 250 + explicit [pause:300] merge into one 550ms break
    assert '<break time="550ms"/>' in ssml
    assert 'rate="+25%"' in ssml
    assert "[laugh]" not in ssml


# ── /tts handler: options + dropped-sound note ──────────────────────────────

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
            "path": "/tmp/fake-tts.mp3", "engine": "openai_compat",
            "bytes": 2048, "seconds": 1.2,
            "dropped_sounds": ["laugh"], "pauses_ms": 550,
            "pauses_dropped": False,
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


def test_control_tts_parses_options_and_reports_drops():
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    obj.context = _FakeContext()
    obj.gateway = _FakeGateway()
    obj._ref_from_key = lambda key: _FakeChat()
    reply = obj._control_tts("speed=1.5 pace=slow hello [laugh] world",
                             chat_key="telegram:1")
    (name, _args), kwargs = obj.context.tools.calls[0]
    assert name == "speak"
    assert kwargs["speed"] == 1.5
    assert kwargs["pace"] == "slow"
    assert "laugh" in reply and "dropped" in reply
    assert "pauses 550ms" in reply


def test_control_tts_gap_option():
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    obj.context = _FakeContext()
    obj.gateway = _FakeGateway()
    obj._ref_from_key = lambda key: _FakeChat()
    obj._control_tts("gap=30 hello", chat_key="telegram:1")
    (_name, _args), kwargs = obj.context.tools.calls[0]
    assert kwargs["gap_ms"] == 30


def test_control_tts_engines_report():
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    obj.context = _FakeContext()
    obj.gateway = _FakeGateway()
    obj._ref_from_key = lambda key: _FakeChat()
    report = obj._control_tts("engines", chat_key="telegram:1")
    assert "espeak_ng" in report
    assert "pause" in report.lower()
    assert "cannot laugh" in report or "can't laugh" in report
    assert "/speak" in report


def test_control_tts_usage():
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    obj = RuntimeVoiceMixin.__new__(RuntimeVoiceMixin)
    obj.context = _FakeContext()
    obj.gateway = _FakeGateway()
    obj._ref_from_key = lambda key: _FakeChat()
    assert "usage" in obj._control_tts("", chat_key="telegram:1")


# ── help documents tags ─────────────────────────────────────────────────────

def test_tts_help_documents_tags():
    from nomorals.social.chat.control import COMMAND_DETAILS

    tts = COMMAND_DETAILS["tts"]
    assert "pace" in tts["usage"] and "speed" in tts["usage"] and "gap" in tts["usage"]
    assert "[pause" in tts["what"]
    assert "[laugh]" in tts["what"]
    assert "engines" in tts["usage"]
