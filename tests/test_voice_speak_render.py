"""Tests for UniversalTTS.speak()'s per-backend native rendering.

``speak()`` parses author tags once (TagProcessor) and then must hand
each neural backend text in *its own* vocabulary — CosyVoice instruct
tokens, Dia parens, Orpheus angle tags, Fish free-form tags — not Bark
tags for everything. No real models are loaded: fake backends record
what they were asked to synthesize.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.tts import UniversalTTS, TagProcessor


class _FakeBackend:
    """Records synthesize() calls; returns 0.1s of silence."""

    sample_rate = 24000

    def __init__(self, name: str, native: bool, fish: bool = False):
        self.name = name
        self.supports_native_tags = native
        self._fish = fish
        self.calls: list[tuple[str, object, str]] = []

    def _is_fish(self) -> bool:
        return self._fish

    def synthesize(self, text: str, voice, *, instruct: str = ""):
        self.calls.append((text, voice, instruct))
        return [0.0] * 2400  # 0.1 s of silence at 24 kHz


def _engine(backend: _FakeBackend, **kw) -> UniversalTTS:
    eng = UniversalTTS(backend="bark", **kw)
    eng._impl = backend
    eng._loaded = True
    return eng


class SegmentsToCanonicalTest(unittest.TestCase):
    def setUp(self):
        self.eng = UniversalTTS.__new__(UniversalTTS)
        self.eng.tag_processor = TagProcessor()

    def test_legacy_sound_tags_normalize(self):
        segs = self.eng.tag_processor.parse("[laughs] ha [clears_throat]")
        out = self.eng._segments_to_canonical(segs)
        self.assertIn("[laugh]", out)
        self.assertIn("[clearthroat]", out)
        self.assertNotIn("[laughs]", out)
        self.assertNotIn("[clears_throat]", out)

    def test_emotion_tag_reopened(self):
        segs = self.eng.tag_processor.parse("[happy] hi there")
        out = self.eng._segments_to_canonical(segs)
        self.assertTrue(out.startswith("[happy]"))
        self.assertIn("hi there", out)

    def test_pause_reinserted(self):
        segs = self.eng.tag_processor.parse("wait [pause:300] go")
        out = self.eng._segments_to_canonical(segs)
        self.assertIn("[pause:300]", out)
        self.assertIn("wait", out)
        self.assertIn("go", out)

    def test_canonical_burst_kept(self):
        segs = self.eng.tag_processor.parse("[sneeze]")
        out = self.eng._segments_to_canonical(segs)
        self.assertEqual(out, "[sneeze]")


class RenderNativeTest(unittest.TestCase):
    def setUp(self):
        self.eng = UniversalTTS.__new__(UniversalTTS)
        self.eng.tag_processor = TagProcessor()
        self.segs = self.eng.tag_processor.parse(
            "[happy] hello [laughs] [pause:300]")

    def test_cosyvoice_gets_instruct_tokens(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("cosyvoice", True), self.segs)
        self.assertIn("[laughter]", text)
        self.assertIn("happy", instruct)

    def test_dia_gets_parens_and_speaker(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("dia", True), self.segs)
        self.assertTrue(text.startswith("[S1]"))
        self.assertIn("(laughs)", text)
        self.assertEqual(instruct, "")

    def test_orpheus_gets_angle_tags(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("orpheus", True), self.segs)
        self.assertIn("<laugh>", text)
        self.assertEqual(instruct, "")

    def test_hf_endpoint_fish_gets_fish_tags(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("hf-endpoint", True, fish=True), self.segs)
        self.assertIn("[laughing]", text)
        self.assertEqual(instruct, "")

    def test_hf_endpoint_non_fish_falls_back_to_bark(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("hf-endpoint", True, fish=False), self.segs)
        self.assertIn("[laughs]", text)
        self.assertEqual(instruct, "")

    def test_bark_keeps_legacy_path(self):
        text, instruct = self.eng._render_native(
            _FakeBackend("bark", True), self.segs)
        self.assertIn("[laughs]", text)
        self.assertEqual(instruct, "")


class SpeakEndToEndTest(unittest.TestCase):
    def test_speak_passes_cosyvoice_native_text_and_instruct(self):
        backend = _FakeBackend("cosyvoice", True)
        with tempfile.TemporaryDirectory() as tmp:
            eng = _engine(backend, voices_dir=os.path.join(tmp, "voices"))
            out = eng.speak("[happy] hello [laughs]", out_path=os.path.join(
                tmp, "say.wav"))
            text, _voice, instruct = backend.calls[0]
            self.assertIn("[laughter]", text)
            self.assertIn("happy", instruct)
            self.assertTrue(os.path.exists(out["path"]))
            self.assertEqual(out["backend"], "cosyvoice")
            with wave.open(out["path"], "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.getnchannels(), 1)

    def test_speak_plain_backend_strips_tags_and_splices_silence(self):
        backend = _FakeBackend("xtts", False)
        with tempfile.TemporaryDirectory() as tmp:
            eng = _engine(backend, voices_dir=os.path.join(tmp, "voices"))
            out = eng.speak("hi [laughs] there [pause:500] done",
                            out_path=os.path.join(tmp, "say.wav"))
            text, _voice, instruct = backend.calls[0]
            self.assertNotIn("[laughs]", text)
            self.assertNotIn("[pause:500]", text)
            self.assertIn("hi", text)
            # 500ms of spliced silence lengthens the output beyond the
            # 0.1s of backend audio
            try:
                import numpy  # noqa: F401
            except ImportError:
                self.skipTest("numpy not installed — pause splicing "
                              "needs it")
            with wave.open(out["path"], "rb") as wav:
                self.assertGreater(wav.getnframes(), 2400)


if __name__ == "__main__":
    unittest.main()
