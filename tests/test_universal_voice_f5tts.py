"""Tests for the F5-TTS backend (nomorals/voice/tts.py).

MIT code / CC-BY-NC checkpoints: highest-fidelity single-shot cloning,
5–15s reference required. No real models: stub ``f5_tts.api``.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.tts import (
    F5TTSBackend,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
)


class _FakeF5TTS:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    def infer(self, *, ref_file, ref_text, gen_text):
        self.calls.append({"ref_file": ref_file, "ref_text": ref_text,
                           "gen_text": gen_text})
        return [0.05] * 120, 24000, None


class TestF5TTS(unittest.TestCase):
    def setUp(self):
        self._stubbed = ["f5_tts", "f5_tts.api"]
        fake = _FakeF5TTS("F5TTS_v1_Base")
        self.fake = fake
        api = types.ModuleType("f5_tts.api")
        api.F5TTS = lambda model: fake
        pkg = types.ModuleType("f5_tts")
        pkg.api = api
        sys.modules["f5_tts"] = pkg
        sys.modules["f5_tts.api"] = api
        os.environ.pop("F5TTS_MODEL", None)

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        os.environ.pop("F5TTS_MODEL", None)

    def test_init_model_id(self):
        backend = F5TTSBackend()
        self.assertEqual(backend.model_id, "F5TTS_v1_Base")

    def test_init_model_from_env(self):
        os.environ["F5TTS_MODEL"] = "F5TTS_v1_Custom"
        backend = F5TTSBackend()
        self.assertEqual(backend.model_id, "F5TTS_v1_Custom")

    def test_synthesize_passes_ref_and_text(self):
        backend = F5TTSBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav",
            prompt_text="this is the reference transcript",
            )
        out = backend.synthesize("hello world", voice)
        call = self.fake.calls[0]
        self.assertEqual(call["ref_file"], "/tmp/ref.wav")
        self.assertEqual(call["ref_text"],
                         "this is the reference transcript")
        self.assertEqual(call["gen_text"], "hello world")
        self.assertEqual(backend.sample_rate, 24000)
        self.assertEqual(len(out), 120)

    def test_missing_prompt_text_uses_asr_path(self):
        backend = F5TTSBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav",
            )
        backend.synthesize("hello", voice)
        self.assertEqual(self.fake.calls[0]["ref_text"], "")

    def test_no_reference_raises_helpful(self):
        backend = F5TTSBackend()
        voice = VoiceProfile(name="preset", preset_id="af_heart")
        with self.assertRaises(RuntimeError) as ctx:
            backend.synthesize("hello", voice)
        self.assertIn("no preset voices", str(ctx.exception))

    def test_no_voice_at_all_raises(self):
        backend = F5TTSBackend()
        with self.assertRaises(RuntimeError):
            backend.synthesize("hello", None)

    def test_consent_gate(self):
        # Consent gate removed (audit-1.0): synthesis proceeds without
        # consent_confirmed; audit logging happens in the backend instead.
        backend = F5TTSBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav")
        # Should NOT raise PermissionError for missing consent
        try:
            backend.synthesize("hello", voice)
        except PermissionError:
            self.fail("consent gate should not exist")
        except Exception:
            pass  # Other errors (missing package, etc.) are fine

    def test_missing_package_raises_helpful(self):
        sys.modules.pop("f5_tts", None)
        sys.modules.pop("f5_tts.api", None)
        with self.assertRaises(RuntimeError) as ctx:
            F5TTSBackend()
        self.assertIn("pip install f5-tts", str(ctx.exception))

    def test_registered(self):
        self.assertIs(_BACKENDS["f5tts"], F5TTSBackend)
        self.assertEqual(_BACKEND_SPECS["f5tts"], "f5_tts")

    def test_available_backends_sees_stub(self):
        self.assertIn("f5tts", available_backends())


if __name__ == "__main__":
    unittest.main()
