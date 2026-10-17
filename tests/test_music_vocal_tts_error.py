"""Vocal TTS failure must name the real cause — never an empty ``()``.

Regression for the owner's /music output:
``🎤 TTS vocal failed () — vocal track skipped``. The empty parens
came from a bare ``EOFError`` (``wave.open`` on 0 bytes) raised inside
the vocal TTS chain and formatted with ``f"... ({exc})"``. These tests
pin: the real reason always surfaces, the system backend prefers
espeak-ng (the proven /tts engine) on every platform, an empty
``--stdout`` falls back to ``-w`` file output, and the "sent audio"
line says instrumental-only when vocals were skipped.
"""

from __future__ import annotations

import io
import os
import sys
import wave
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media import vocal_lite
from nomorals.media.vocal_lite import _exc_str
from nomorals.voice.tts import HFEndpointBackend, SystemTTSBackend


class TestExcStr:
    def test_bare_eof_error_names_class(self):
        assert _exc_str(EOFError()) == "EOFError"

    def test_empty_message_value_error_names_class(self):
        assert _exc_str(ValueError()) == "ValueError"

    def test_message_preserved(self):
        assert _exc_str(RuntimeError("boom")) == "boom"

    def test_whitespace_only_message_names_class(self):
        assert _exc_str(RuntimeError("   ")) == "RuntimeError"


class TestDecodeWavEmpty:
    def test_empty_blob_descriptive_not_bare_eof(self):
        try:
            HFEndpointBackend._decode_wav(b"")
        except ValueError as exc:
            assert "empty audio" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("empty blob must raise")

    def test_valid_wav_still_decodes(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 10)
        samples, rate = HFEndpointBackend._decode_wav(buf.getvalue())
        assert rate == 22050
        assert len(samples) == 10


def _bare_eof_speak(self, *args, **kwargs):
    # A backend raising a bare EOFError (empty message) — the exact
    # shape that used to surface as "TTS vocal failed ()".
    raise EOFError()


def _empty_blob_speak(self, *args, **kwargs):
    # The fixed _decode_wav path: empty bytes -> descriptive ValueError.
    from nomorals.voice.tts import HFEndpointBackend
    HFEndpointBackend._decode_wav(b"")
    return {"path": ""}


class TestDefaultTtsNeverEmpty:
    def test_bare_eof_error_surfaced(self):
        with (mock.patch("nomorals.voice.tts.available_backends",
                         return_value=["system"]),
              mock.patch("nomorals.voice.tts.UniversalTTS.speak",
                         _bare_eof_speak)):
            res = vocal_lite._default_tts("What a life", "/tmp/x.wav")
        assert not res["ok"]
        assert "failed ()" not in res["reason"]
        assert "EOFError" in res["reason"]

    def test_empty_blob_error_surfaced_descriptively(self):
        with (mock.patch("nomorals.voice.tts.available_backends",
                         return_value=["system"]),
              mock.patch("nomorals.voice.tts.UniversalTTS.speak",
                         _empty_blob_speak)):
            res = vocal_lite._default_tts("What a life", "/tmp/x.wav")
        assert not res["ok"]
        assert "failed ()" not in res["reason"]
        assert "empty audio bytes" in res["reason"]

    def test_hook_vocal_reason_never_empty_parens(self):
        class Sec:
            name = "chorus"
            lyrics = ["What a life, we made it through",
                      "sun on our faces, gold in the sky"]

        class Song:
            sections = [Sec()]
            tempo = 114

        with (mock.patch("nomorals.voice.tts.available_backends",
                         return_value=["system"]),
              mock.patch("nomorals.voice.tts.UniversalTTS.speak",
                         _bare_eof_speak)):
            res = vocal_lite.render_hook_vocal(Song(), "/tmp/vocal_t")
        assert not res["ok"]
        assert "failed ()" not in res["reason"]
        assert "EOFError" in res["reason"]

    def test_add_vocal_track_reason_never_empty_parens(self):
        class Sec:
            name = "chorus"
            lyrics = ["What a life, we made it through"]

        class Song:
            sections = [Sec()]
            tempo = 114

        with (mock.patch("nomorals.voice.tts.available_backends",
                         return_value=["system"]),
              mock.patch("nomorals.voice.tts.UniversalTTS.speak",
                         _bare_eof_speak)):
            res = vocal_lite.add_vocal_track(
                Song(), "/tmp/bed.wav", "/tmp/vocal_t")
        # no bed file -> honest "no bed audio", still never empty parens
        assert "()" not in res["reason"]


class TestSystemBackendDetect:
    def test_prefers_espeak_ng_on_win32(self):
        def fake_which(name):
            return {"espeak-ng": "C:\\espeak-ng.exe"}.get(name)

        with (mock.patch.object(sys, "platform", "win32"),
              mock.patch("nomorals.voice.tts.shutil.which",
                         side_effect=fake_which)):
            assert SystemTTSBackend.detect() == (
                "espeak", "C:\\espeak-ng.exe")

    def test_powershell_fallback_on_win32_without_espeak(self):
        def fake_which(name):
            return {"powershell": "C:\\pwsh.exe"}.get(name)

        with (mock.patch.object(sys, "platform", "win32"),
              mock.patch("nomorals.voice.tts.shutil.which",
                         side_effect=fake_which)):
            assert SystemTTSBackend.detect() == (
                "powershell", "C:\\pwsh.exe")

    def test_prefers_espeak_ng_over_say_on_darwin(self):
        def fake_which(name):
            return {"espeak-ng": "/usr/bin/espeak-ng",
                    "say": "/usr/bin/say"}.get(name)

        with (mock.patch.object(sys, "platform", "darwin"),
              mock.patch("nomorals.voice.tts.shutil.which",
                         side_effect=fake_which)):
            assert SystemTTSBackend.detect() == (
                "espeak", "/usr/bin/espeak-ng")


class TestViaEspeakFallback:
    def _backend(self):
        b = SystemTTSBackend.__new__(SystemTTSBackend)
        b.lang = ""
        b._kind = "espeak"
        b._exe = "espeak-ng"
        return b

    def test_empty_stdout_retries_via_w_file(self):
        backend = self._backend()
        calls = []

        class Proc:
            def __init__(self, stdout):
                self.stdout = stdout
                self.returncode = 0

        def fake_run(cmd):
            calls.append(list(cmd))
            if "--stdout" in cmd:
                return Proc(b"")  # empty stdout -> retry
            # -w call: write a minimal valid wav to the -w target
            out = cmd[cmd.index("-w") + 1]
            with wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"\x01\x00" * 100)
            return Proc(b"")

        with mock.patch.object(SystemTTSBackend, "_run",
                               side_effect=fake_run):
            blob = backend._via_espeak("hello world")
        assert blob[:4] == b"RIFF"
        assert len(calls) == 2
        assert "-w" in calls[1]

    def test_both_empty_descriptive_error(self):
        backend = self._backend()

        class Proc:
            stdout = b""
            returncode = 0

        with mock.patch.object(SystemTTSBackend, "_run",
                               return_value=Proc()):
            try:
                backend._via_espeak("hello world")
            except RuntimeError as exc:
                assert "produced no audio" in str(exc)
            else:  # pragma: no cover
                raise AssertionError("must raise on empty output")


class TestSentLineReflectsVocals:
    """The 'sent audio + score' line must not imply a full song when
    the vocal track was skipped."""

    def _run_music(self, vocal_path, vocal_note):
        from types import SimpleNamespace
        from unittest import mock
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class Sec:
            lyrics = ["line one", "line two"]

        song = SimpleNamespace(
            title="What Life", style="pop", key="A", tempo=114,
            sections=[Sec()], synth_backend="builtin",
            synth_note="", vocal_note=vocal_note, vocal_path=vocal_path,
            audio_path="/tmp/song.wav", score_pdf_path="/tmp/score.pdf")

        rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        rt.context = SimpleNamespace()
        rt.gateway = mock.Mock()
        rt._ref_from_key = lambda key: SimpleNamespace(
            platform="telegram", chat_id="123")

        with mock.patch("nomorals.media.music.MusicCreator") as MC:
            MC.return_value.compose.return_value = song
            text = rt._control_music("What Life pop", "telegram:123")
        return text

    def test_skipped_vocal_marks_instrumental_only(self):
        text = self._run_music(
            "", "TTS vocal failed (EOFError) — vocal track skipped")
        assert "🎤 TTS vocal failed (EOFError)" in text
        assert "sent audio + score to this chat." in text
        assert "instrumental only" in text
        assert "vocal track skipped" in text

    def test_rendered_vocal_has_no_instrumental_qualifier(self):
        text = self._run_music(
            "/tmp/song_with_vocals.wav", "sung vocal: 4 line(s)")
        assert "sent audio + score to this chat." in text
        assert "instrumental only" not in text

    def test_novocals_has_no_instrumental_qualifier(self):
        from types import SimpleNamespace
        from unittest import mock
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        class Sec:
            lyrics = ["line one"]

        song = SimpleNamespace(
            title="What Life", style="pop", key="A", tempo=114,
            sections=[Sec()], synth_backend="builtin",
            synth_note="", vocal_note="", vocal_path="",
            audio_path="/tmp/song.wav", score_pdf_path="/tmp/score.pdf")
        rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        rt.context = SimpleNamespace()
        rt.gateway = mock.Mock()
        rt._ref_from_key = lambda key: SimpleNamespace(
            platform="telegram", chat_id="123")
        with mock.patch("nomorals.media.music.MusicCreator") as MC:
            MC.return_value.compose.return_value = song
            text = rt._control_music("What Life pop novocals",
                                     "telegram:123")
        assert "instrumental only" not in text
