"""Tests for the Chatterbox TTS backend (nomorals/voice/tts.py).

MIT, Resemble AI: zero-shot cloning from ~5-10s of reference audio,
23 languages, emotion exaggeration — the best free cloning TTS.
No real models are loaded: stub ``chatterbox.*`` modules stand in.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import director
from nomorals.voice.tts import (
    ChatterboxBackend,
    UniversalTTS,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
)


def _make_pkg(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _FakeModel:
    def __init__(self) -> None:
        self.sr = 24000
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return [0.1] * 240


class TestChatterbox(unittest.TestCase):
    def setUp(self):
        self._stubbed = ["chatterbox", "chatterbox.mtl_tts",
                         "chatterbox.tts_turbo"]
        self.models: dict[str, _FakeModel] = {}

        def make_mtl(device=None, t3_model=None):
            model = _FakeModel()
            self.models["multilingual"] = model
            self._mtl_kwargs = {"device": device, "t3_model": t3_model}
            return model

        def make_turbo(device=None, nano=False):
            model = _FakeModel()
            self.models["turbo" if not nano else "nano"] = model
            self._turbo_kwargs = {"device": device, "nano": nano}
            return model

        pkg = _make_pkg("chatterbox")
        mtl = _make_pkg("chatterbox.mtl_tts")
        mtl.ChatterboxMultilingualTTS = types.SimpleNamespace(
            from_pretrained=staticmethod(make_mtl))
        turbo = _make_pkg("chatterbox.tts_turbo")
        turbo.ChatterboxTurboTTS = types.SimpleNamespace(
            from_pretrained=staticmethod(make_turbo))
        pkg.mtl_tts = mtl
        pkg.tts_turbo = turbo
        os.environ.pop("CHATTERBOX_VARIANT", None)
        os.environ["CHATTERBOX_DEVICE"] = "cpu"

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        os.environ.pop("CHATTERBOX_DEVICE", None)
        os.environ.pop("CHATTERBOX_VARIANT", None)

    def test_default_variant_is_multilingual(self):
        backend = ChatterboxBackend()
        self.assertEqual(backend.variant, "multilingual")
        self.assertEqual(self._mtl_kwargs["t3_model"], "v3")
        self.assertFalse(backend.supports_native_tags)

    def test_turbo_variant_and_native_tags(self):
        backend = ChatterboxBackend(variant="turbo")
        self.assertEqual(backend.variant, "turbo")
        self.assertTrue(backend.supports_native_tags)
        self.assertFalse(self._turbo_kwargs["nano"])

    def test_nano_variant(self):
        backend = ChatterboxBackend(variant="nano")
        self.assertTrue(self._turbo_kwargs["nano"])
        self.assertTrue(backend.supports_native_tags)

    def test_variant_from_env(self):
        os.environ["CHATTERBOX_VARIANT"] = "turbo"
        backend = ChatterboxBackend()
        self.assertEqual(backend.variant, "turbo")

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            ChatterboxBackend(variant="mega")

    def test_synthesize_clone_passes_language_and_ref(self):
        backend = ChatterboxBackend()
        voice = VoiceProfile(
            name="me", reference_audio_path="/tmp/ref.wav",
            language="fr", )
        out = backend.synthesize("bonjour", voice)
        call = self.models["multilingual"].calls[0]
        self.assertEqual(call["language_id"], "fr")
        self.assertEqual(call["audio_prompt_path"], "/tmp/ref.wav")
        self.assertEqual(call["text"], "bonjour")
        self.assertTrue(len(out) > 0)

    def test_synthesize_without_voice(self):
        backend = ChatterboxBackend()
        out = backend.synthesize("hello", None)
        call = self.models["multilingual"].calls[0]
        self.assertEqual(call["language_id"], "en")
        self.assertIsNone(call["audio_prompt_path"])
        self.assertTrue(len(out) > 0)

    def test_consent_gate(self):
        # Consent gate removed (audit-1.0): synthesis proceeds without
        # consent_confirmed; audit logging happens in the backend instead.
        backend = ChatterboxBackend()
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
        for name in ("chatterbox", "chatterbox.mtl_tts",
                     "chatterbox.tts_turbo"):
            sys.modules.pop(name, None)
        with self.assertRaises(RuntimeError) as ctx:
            ChatterboxBackend()
        self.assertIn("pip install chatterbox-tts", str(ctx.exception))

    def test_registered(self):
        self.assertIs(_BACKENDS["chatterbox"], ChatterboxBackend)
        self.assertEqual(_BACKEND_SPECS["chatterbox"], "chatterbox")

    def test_available_backends_sees_stub(self):
        self.assertIn("chatterbox", available_backends())

    def test_render_chatterbox_native_and_fallback(self):
        out = director.render_chatterbox("[laugh] that was funny [cough]")
        self.assertIn("[laugh]", out)
        self.assertIn("[cough]", out)
        # unknown burst degrades to speakable words, never silence
        out2 = director.render_chatterbox("[sneeze] bless you")
        self.assertIn("achoo!", out2)
        # emotions are dropped (they go through the exaggeration knob)
        out3 = director.render_chatterbox("[happy] great news")
        self.assertNotIn("[happy]", out3)
        self.assertIn("great news", out3)

    def test_render_for_mapping(self):
        text, extra = director.render_for("chatterbox", "[laugh] hi")
        self.assertIn("[laugh]", text)
        self.assertIsNone(extra)

    def test_engine_native_path_uses_render_chatterbox(self):
        eng = UniversalTTS.__new__(UniversalTTS)
        from nomorals.voice.tts import TagProcessor

        eng.tag_processor = TagProcessor()
        backend = ChatterboxBackend(variant="turbo")
        segments = eng.tag_processor.parse("[laugh] well hello there")
        text, instruct = eng._render_native(backend, segments)
        self.assertIn("[laugh]", text)
        self.assertEqual(instruct, "")


if __name__ == "__main__":
    unittest.main()
