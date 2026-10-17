"""STT API errors must surface the REAL underlying error, never a bare
generic "STT API error" that hides the cause the owner needs to act on.

Covers:
- _short_stt_reason (pingpong + brain twins) preserves specific errors:
  "STT API 401: {...}" stays as-is (no doubling), plain failures get
  "STT API error: <detail>", secrets are redacted, detail truncates.
- NM_VOICE_STT_PROVIDER=api normalizes to openai_compat; the env keys
  documented in the template (NM_AUDIO_STT_*, NM_VOICE_STT_*) actually
  map into audio settings (previously silently ignored).
- _stt_openai_compat sends the real audio MIME (audio/ogg for a .ogg
  voice note), not application/octet-stream.
- End-to-end: a voice note with a mocked STT API failure returns the
  specific reason through pingpong + brain; a mocked success
  transcribes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.partner.brain import _short_stt_reason as brain_reason
from nomorals.core.config import load_settings
from nomorals.core.errors import NoMoralsError
from nomorals.tools import audio as audio_mod
from nomorals.voice import pingpong as pp


class ShortReasonTest(unittest.TestCase):
    def test_api_status_body_preserved_no_doubling(self) -> None:
        err = ('STT API 401: {"error": {"message": "Incorrect API key", '
               '"code": "invalid_api_key"}}')
        self.assertEqual(pp._short_stt_reason(err), err)
        self.assertEqual(brain_reason(err), err)

    def test_plain_failure_gets_detail(self) -> None:
        self.assertEqual(pp._short_stt_reason("Groq timeout"),
                         "STT API error: Groq timeout")
        self.assertEqual(brain_reason("Groq timeout"),
                         "STT API error: Groq timeout")

    def test_tls_failure_preserved(self) -> None:
        err = ("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] "
               "certificate verify failed>")
        self.assertIn("CERTIFICATE_VERIFY_FAILED", pp._short_stt_reason(err))
        self.assertIn("CERTIFICATE_VERIFY_FAILED", brain_reason(err))

    def test_no_backend_still_classified(self) -> None:
        self.assertEqual(pp._short_stt_reason("no STT backend installed"),
                         "no STT backend")
        self.assertEqual(brain_reason("no STT backend installed"),
                         "no STT backend")

    def test_empty_stays_bare(self) -> None:
        self.assertEqual(pp._short_stt_reason(""), "STT API error")
        self.assertEqual(brain_reason(None), "STT API error")

    def test_typed_error_message_extracted(self) -> None:
        # The real path hands us a NoMoralsError (outcome.error), not a
        # string — the message must be extracted, not str()'d into a repr.
        err = NoMoralsError("STT API 429: rate limited")
        self.assertEqual(pp._short_stt_reason(err), "STT API 429: rate limited")
        self.assertEqual(brain_reason(err), "STT API 429: rate limited")

    def test_secrets_redacted(self) -> None:
        err = "STT API 401: Bearer sk-secret-12345 api_key=also-secret"
        out = pp._short_stt_reason(err)
        self.assertNotIn("sk-secret-12345", out)
        self.assertNotIn("also-secret", out)
        self.assertIn("Bearer ***", out)
        self.assertIn("api_key=***", out)

    def test_detail_truncates_and_single_line(self) -> None:
        err = "line one\nline two " + ("x" * 500)
        out = pp._short_stt_reason(err)
        self.assertNotIn("\n", out)
        self.assertLessEqual(len(out), 301 + len("STT API error: "))

    def test_twins_agree(self) -> None:
        cases = ["", "no backend", "STT API 500: boom",
                 "CERTIFICATE_VERIFY_FAILED", "Groq timeout",
                 "Bearer abc123", NoMoralsError("STT API 400: bad")]
        for case in cases:
            self.assertEqual(pp._short_stt_reason(case),
                             brain_reason(case), f"twins differ on {case!r}")


class ProviderNormalizationTest(unittest.TestCase):
    def test_api_normalizes(self) -> None:
        self.assertEqual(audio_mod.normalize_stt_provider("api"),
                         "openai_compat")
        self.assertEqual(audio_mod.normalize_stt_provider("API"),
                         "openai_compat")
        self.assertEqual(audio_mod.normalize_stt_provider("auto"), "auto")
        self.assertEqual(audio_mod.normalize_stt_provider("whisper_cpp"),
                         "whisper_cpp")

    def test_detect_stt_api(self) -> None:
        self.assertEqual(
            audio_mod.detect_stt("api", base_url="https://api.groq.com/openai/v1",
                                 api_key="k"), "openai_compat")
        self.assertIsNone(
            audio_mod.detect_stt("api", base_url="", api_key=""))


class OpenAiCompatMimeTest(unittest.TestCase):
    def test_ogg_sent_as_audio_ogg(self) -> None:
        captured: dict[str, Any] = {}

        class _FakeResponse:
            ok = True
            status = 200
            text = '{"text": "hello"}'

            def json(self) -> dict[str, Any]:
                return {"text": "hello"}

        class _FakeClient:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def post_multipart(self, url: str, fields: Any = None,
                               files: Any = None, **kw: Any) -> Any:
                captured["url"] = url
                captured["fields"] = fields
                captured["files"] = files
                return _FakeResponse()

        with tempfile.TemporaryDirectory(prefix="nm-stt-mime-") as tmp:
            path = os.path.join(tmp, "voice.ogg")
            with open(path, "wb") as fh:
                fh.write(b"fake-ogg-bytes")
            with mock.patch("nomorals.core.http.HttpClient", _FakeClient):
                text = audio_mod._stt_openai_compat(
                    __import__("pathlib").Path(path), "en",
                    base_url="https://api.groq.com/openai/v1",
                    api_key="k", model="")
            self.assertEqual(text, "hello")
            files = captured["files"]
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0][2], "audio/ogg")
            self.assertIn("whisper-large-v3-turbo",
                          captured["fields"]["model"])


class EnvMappingTest(unittest.TestCase):
    def test_audio_stt_env_keys_map(self) -> None:
        settings = load_settings(env={
            "NM_AUDIO_STT_BASE_URL": "https://api.groq.com/openai/v1",
            "NM_AUDIO_STT_API_KEY": "stt-key",
            "NM_VOICE_STT_PROVIDER": "api",
            "NM_VOICE_STT_MODEL": "whisper-large-v3-turbo",
            "NM_VOICE_TTS_ENGINE": "espeak_ng",
        })
        self.assertEqual(settings.audio.stt_base_url,
                         "https://api.groq.com/openai/v1")
        self.assertEqual(settings.audio.stt_api_key, "stt-key")
        self.assertEqual(settings.audio.stt_provider, "api")
        self.assertEqual(settings.audio.stt_model, "whisper-large-v3-turbo")
        self.assertEqual(settings.audio.tts_engine, "espeak_ng")


class EndToEndReasonTest(unittest.TestCase):
    def _media_msg(self) -> SimpleNamespace:
        return SimpleNamespace(
            media=[SimpleNamespace(kind="voice", path="/tmp/note.ogg",
                                   mime="audio/ogg")],
            meta={})

    def test_pingpong_failure_specific_reason(self) -> None:
        class _FailingTools:
            def call(self, *a: Any, **k: Any) -> Any:
                return SimpleNamespace(
                    ok=False,
                    error=NoMoralsError(
                        'STT API 401: {"error": {"message": '
                        '"Incorrect API key", "code": "invalid_api_key"}}'),
                    value={})

        text, reason = pp.transcribe_voice_media_with_reason(
            SimpleNamespace(tools=_FailingTools()), self._media_msg())
        self.assertEqual(text, "")
        self.assertIn("STT API 401", reason)
        self.assertIn("Incorrect API key", reason)

    def test_pingpong_success_transcribes(self) -> None:
        class _WorkingTools:
            def call(self, *a: Any, **k: Any) -> Any:
                return SimpleNamespace(ok=True, value={"text": "hello devon"})

        text, reason = pp.transcribe_voice_media_with_reason(
            SimpleNamespace(tools=_WorkingTools()), self._media_msg())
        self.assertEqual(text, "hello devon")
        self.assertEqual(reason, "")

    def test_brain_failure_specific_reason(self) -> None:
        from nomorals.agents.partner_runtime import PartnerBrain

        class _FailingTools:
            def call(self, *a: Any, **k: Any) -> Any:
                return SimpleNamespace(
                    ok=False,
                    error="STT API 400: model_not_found",
                    value={})

        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(tools=_FailingTools())
        brain._last_stt_failure_reason = ""
        note, reason = brain._transcribe_media_note_with_reason(
            SimpleNamespace(path="/tmp/note.ogg"))
        self.assertEqual(note, "")
        self.assertIn("STT API 400", reason)
        self.assertIn("model_not_found", reason)


if __name__ == "__main__":
    unittest.main()
