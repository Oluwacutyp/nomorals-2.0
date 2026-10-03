"""Tests for the Qwen3-TTS backend (nomorals/voice/tts.py).

Apache-2.0, Alibaba: Base (3s cloning) vs CustomVoice (preset speakers)
picked automatically; native [laugh]/[sigh]/[emotion] tags. No real
models: stub the ``qwen_tts`` package.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import director
from nomorals.voice.tts import (
    Qwen3TTSBackend,
    TagProcessor,
    UniversalTTS,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
)


class _FakeQwen3:
    def __init__(self) -> None:
        self.clone_calls: list[dict] = []
        self.custom_calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.clone_calls.append(kwargs)
        return [[0.3] * 60], 24000

    def generate_custom_voice(self, **kwargs):
        self.custom_calls.append(kwargs)
        return [[0.4] * 60], 24000


class TestQwen3TTS(unittest.TestCase):
    def setUp(self):
        self._stubbed = ["qwen_tts"]
        self.model = _FakeQwen3()
        self.loaded: list[dict] = []

        def from_pretrained(model_id, device_map=None, dtype=None):
            self.loaded.append({"model_id": model_id,
                                "device_map": device_map, "dtype": dtype})
            return self.model

        mod = types.ModuleType("qwen_tts")
        mod.Qwen3TTSModel = types.SimpleNamespace(
            from_pretrained=staticmethod(from_pretrained))
        sys.modules["qwen_tts"] = mod
        os.environ.pop("QWEN3_TTS_BASE_MODEL", None)
        os.environ.pop("QWEN3_TTS_MODEL", None)

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        os.environ.pop("QWEN3_TTS_BASE_MODEL", None)
        os.environ.pop("QWEN3_TTS_MODEL", None)

    def test_clone_path_with_reference(self):
        backend = Qwen3TTSBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav",
            prompt_text="the transcript", language="fr",
            )
        out = backend.synthesize("bonjour le monde", voice)
        self.assertTrue(any("Base" in entry["model_id"]
                            for entry in self.loaded))
        call = self.model.clone_calls[0]
        self.assertEqual(call["text"], "bonjour le monde")
        self.assertEqual(call["language"], "French")
        self.assertEqual(call["ref_audio"], "/tmp/ref.wav")
        self.assertEqual(call["ref_text"], "the transcript")
        self.assertEqual(backend.sample_rate, 24000)
        self.assertEqual(len(out), 60)

    def test_custom_voice_path_without_reference(self):
        backend = Qwen3TTSBackend()
        voice = VoiceProfile(name="narr", preset_id="Ryan", language="en")
        out = backend.synthesize("a tale", voice, instruct="very happy.")
        self.assertTrue(any("CustomVoice" in entry["model_id"]
                            for entry in self.loaded))
        call = self.model.custom_calls[0]
        self.assertEqual(call["speaker"], "Ryan")
        self.assertEqual(call["instruct"], "very happy.")
        self.assertEqual(call["language"], "English")
        self.assertEqual(len(out), 60)

    def test_default_speaker_when_no_preset(self):
        backend = Qwen3TTSBackend()
        backend.synthesize("hi", None)
        self.assertEqual(self.model.custom_calls[0]["speaker"], "Vivian")

    def test_language_map(self):
        self.assertEqual(Qwen3TTSBackend._language(
            VoiceProfile(name="x", language="zh")), "Chinese")
        self.assertEqual(Qwen3TTSBackend._language(None), "English")
        self.assertEqual(Qwen3TTSBackend._language(
            VoiceProfile(name="x", language="xx")), "Auto")

    def test_consent_gate(self):
        # Consent gate removed (audit-1.0): synthesis proceeds without
        # consent_confirmed; audit logging happens in the backend instead.
        backend = Qwen3TTSBackend()
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
        sys.modules.pop("qwen_tts", None)
        with self.assertRaises(RuntimeError) as ctx:
            Qwen3TTSBackend()
        self.assertIn("pip install qwen-tts", str(ctx.exception))

    def test_native_tags(self):
        self.assertTrue(Qwen3TTSBackend.supports_native_tags)
        self.assertTrue(Qwen3TTSBackend.supports_cloning)

    def test_registered(self):
        self.assertIs(_BACKENDS["qwen3tts"], Qwen3TTSBackend)
        self.assertEqual(_BACKEND_SPECS["qwen3tts"], "qwen_tts")

    def test_available_backends_sees_stub(self):
        self.assertIn("qwen3tts", available_backends())

    def test_render_for_passthrough_keeps_native_tags(self):
        text, extra = director.render_for("qwen3tts", "[laugh] oh wow [sigh]")
        self.assertIn("[laugh]", text)
        self.assertIn("[sigh]", text)
        self.assertIsNone(extra)

    def test_engine_native_path_keeps_canonical_tags(self):
        eng = UniversalTTS.__new__(UniversalTTS)
        eng.tag_processor = TagProcessor()
        backend = types.SimpleNamespace(name="qwen3tts",
                                        supports_native_tags=True)
        segments = eng.tag_processor.parse("[laugh] well hello")
        text, instruct = eng._render_native(backend, segments)
        self.assertIn("[laugh]", text)
        self.assertIn("well hello", text)
        self.assertEqual(instruct, "")


if __name__ == "__main__":
    unittest.main()
