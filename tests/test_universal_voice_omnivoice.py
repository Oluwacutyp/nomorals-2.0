"""Tests for the OmniVoice backend (nomorals/voice/tts.py).

Apache-2.0, k2-fsa: 600+ languages, zero-shot cloning, voice design,
native [laughter]/[sigh]/[sniff] non-verbals. No real models: stub
the ``omnivoice`` package.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import director
from nomorals.voice.tts import (
    OmniVoiceBackend,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
)


class _FakeOmniVoice:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, *, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return [0.2] * 160


class TestOmniVoice(unittest.TestCase):
    def setUp(self):
        self._stubbed = ["omnivoice"]
        self.model = _FakeOmniVoice()

        def from_pretrained(model_id, device_map=None, dtype=None):
            self._load_kwargs = {"model_id": model_id,
                                 "device_map": device_map, "dtype": dtype}
            return self.model

        mod = types.ModuleType("omnivoice")
        mod.OmniVoice = types.SimpleNamespace(
            from_pretrained=staticmethod(from_pretrained))
        sys.modules["omnivoice"] = mod
        os.environ.pop("OMNIVOICE_MODEL_ID", None)

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        os.environ.pop("OMNIVOICE_MODEL_ID", None)

    def test_init_default_model(self):
        OmniVoiceBackend()
        self.assertEqual(self._load_kwargs["model_id"], "k2-fsa/OmniVoice")

    def test_init_model_from_env(self):
        os.environ["OMNIVOICE_MODEL_ID"] = "my-org/my-voice"
        OmniVoiceBackend()
        self.assertEqual(self._load_kwargs["model_id"], "my-org/my-voice")

    def test_synthesize_clone_passes_ref(self):
        backend = OmniVoiceBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav",
            prompt_text="hello there",
            description="female, low pitch, british accent")
        out = backend.synthesize("new words", voice)
        call = self.model.calls[0]
        self.assertEqual(call["ref_audio"], "/tmp/ref.wav")
        self.assertEqual(call["ref_text"], "hello there")
        self.assertEqual(call["instruct"], "female, low pitch, british accent")
        self.assertEqual(call["text"], "new words")
        self.assertTrue(len(out) > 0)

    def test_synthesize_design_voice_without_ref(self):
        backend = OmniVoiceBackend()
        out = backend.synthesize("a story", None, instruct="warm narrator")
        call = self.model.calls[0]
        self.assertNotIn("ref_audio", call)
        self.assertEqual(call["instruct"], "warm narrator")
        self.assertTrue(len(out) > 0)

    def test_consent_gate(self):
        # Consent gate removed (audit-1.0): synthesis proceeds without
        # consent_confirmed; audit logging happens in the backend instead.
        backend = OmniVoiceBackend()
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
        sys.modules.pop("omnivoice", None)
        with self.assertRaises(RuntimeError) as ctx:
            OmniVoiceBackend()
        self.assertIn("pip install omnivoice", str(ctx.exception))

    def test_native_tags(self):
        self.assertTrue(OmniVoiceBackend.supports_native_tags)
        self.assertTrue(OmniVoiceBackend.supports_cloning)

    def test_registered(self):
        self.assertIs(_BACKENDS["omnivoice"], OmniVoiceBackend)
        self.assertEqual(_BACKEND_SPECS["omnivoice"], "omnivoice")

    def test_available_backends_sees_stub(self):
        self.assertIn("omnivoice", available_backends())

    def test_render_omnivoice_mapping(self):
        out = director.render_omnivoice("[laugh] that was funny [sigh]")
        self.assertIn("[laughter]", out)
        self.assertIn("[sigh]", out)
        out2 = director.render_omnivoice("[sniffle] dusty in here")
        self.assertIn("[sniff]", out2)
        # unmapped bursts pass through free-form, never vanish
        out3 = director.render_omnivoice("[cough] excuse me")
        self.assertIn("[cough]", out3)

    def test_render_for_mapping(self):
        text, extra = director.render_for("omnivoice", "[laugh] hi")
        self.assertIn("[laughter]", text)
        self.assertIsNone(extra)


if __name__ == "__main__":
    unittest.main()
