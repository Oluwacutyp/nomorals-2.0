"""Tests for the Piper TTS backend (nomorals/voice/tts.py).

MIT, ONNX-runtime: the best free on-device (phone/CPU) TTS —
~60MB voices, 3.6× realtime on plain CPU. No real models: stub the
``piper`` package.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.tts import (
    PiperBackend,
    VoiceProfile,
    _BACKEND_SPECS,
    _BACKENDS,
    available_backends,
)


class _FakeConfig:
    sample_rate = 22050


class _FakePiperVoice:
    config = _FakeConfig()

    def __init__(self, path: str) -> None:
        self.path = path
        self.stream_calls: list[str] = []

    def synthesize_stream_raw(self, text):
        self.stream_calls.append(text)
        yield struct.pack("<8h", *([1000] * 8))


def _stub_piper():
    mod = types.ModuleType("piper")
    mod.PiperVoice = _FakePiperVoice
    mod.PiperVoice.load = staticmethod(lambda path: _FakePiperVoice(path))
    sys.modules["piper"] = mod


class TestPiper(unittest.TestCase):
    def setUp(self):
        self._stubbed = ["piper"]
        self.tmp = tempfile.mkdtemp()
        self.voice_file = os.path.join(self.tmp, "en_US-test-medium.onnx")
        with open(self.voice_file, "wb") as fh:
            fh.write(b"fake-onnx")
        _stub_piper()
        os.environ["PIPER_VOICES_DIR"] = self.tmp
        os.environ.pop("PIPER_VOICE", None)

    def tearDown(self):
        for name in self._stubbed:
            sys.modules.pop(name, None)
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("PIPER_VOICES_DIR", None)
        os.environ.pop("PIPER_VOICE", None)

    def test_init_finds_default_voice_and_rate(self):
        backend = PiperBackend()
        self.assertEqual(backend.voice_path, self.voice_file)
        self.assertEqual(backend.sample_rate, 22050)

    def test_init_explicit_path(self):
        backend = PiperBackend(voice_path=self.voice_file)
        self.assertEqual(backend.voice_path, self.voice_file)

    def test_synthesize_returns_floats(self):
        backend = PiperBackend()
        out = backend.synthesize("hello", None)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(isinstance(v, float) for v in out))
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in out))

    def test_preset_id_resolves_voice(self):
        alt = os.path.join(self.tmp, "en_GB-alan-medium.onnx")
        with open(alt, "wb") as fh:
            fh.write(b"fake")
        backend = PiperBackend()
        voice = VoiceProfile(name="alan", preset_id="en_GB-alan-medium")
        backend.synthesize("hello", voice)
        resolved = backend._resolve(voice)
        self.assertEqual(resolved.path, alt)

    def test_unknown_preset_raises_helpful(self):
        backend = PiperBackend()
        voice = VoiceProfile(name="x", preset_id="no-such-voice")
        with self.assertRaises(RuntimeError) as ctx:
            backend.synthesize("hello", voice)
        self.assertIn("download_voices", str(ctx.exception))

    def test_no_voice_anywhere_raises_helpful(self):
        os.environ["PIPER_VOICES_DIR"] = "/nonexistent-dir-xyz"
        backend = PiperBackend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.synthesize("hello", None)
        self.assertIn("piper.download_voices", str(ctx.exception))

    def test_missing_package_raises_helpful(self):
        sys.modules.pop("piper", None)
        with self.assertRaises(RuntimeError) as ctx:
            PiperBackend(voice_path=self.voice_file)
        self.assertIn("pip install piper-tts", str(ctx.exception))

    def test_empty_audio_raises(self):
        class _Silent(_FakePiperVoice):
            def synthesize_stream_raw(self, text):
                return iter([])

        sys.modules["piper"].PiperVoice.load = staticmethod(
            lambda path: _Silent(path))
        backend = PiperBackend(voice_path=self.voice_file)
        with self.assertRaises(ValueError):
            backend.synthesize("hello", None)

    def test_no_cloning_no_native_tags(self):
        self.assertFalse(PiperBackend.supports_cloning)
        self.assertFalse(PiperBackend.supports_native_tags)

    def test_registered(self):
        self.assertIs(_BACKENDS["piper"], PiperBackend)
        self.assertEqual(_BACKEND_SPECS["piper"], "piper")

    def test_available_backends_sees_stub(self):
        self.assertIn("piper", available_backends())


if __name__ == "__main__":
    unittest.main()
