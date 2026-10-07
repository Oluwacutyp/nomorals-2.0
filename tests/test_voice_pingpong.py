"""Tests for voice-note ping-pong (no network, no real TTS)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nomorals.voice import pingpong as pp


def _msg(*media):
    return SimpleNamespace(media=list(media), text="", meta={},
                           chat=SimpleNamespace(key="test"))


def _media(kind="audio", path="/tmp/note.ogg", mime="audio/ogg"):
    return SimpleNamespace(kind=kind, path=path, mime=mime, name="note.ogg")


class HasVoiceMediaTests(unittest.TestCase):
    def test_voice_detected(self):
        self.assertTrue(pp.has_voice_media(_msg(_media("audio"))))
        self.assertTrue(pp.has_voice_media(_msg(_media("voice"))))

    def test_mime_detected(self):
        self.assertTrue(pp.has_voice_media(
            _msg(_media(kind="document", mime="audio/mpeg"))))

    def test_no_voice(self):
        self.assertFalse(pp.has_voice_media(_msg()))
        self.assertFalse(pp.has_voice_media(
            _msg(_media(kind="image", mime="image/jpeg"))))


class TranscribeTests(unittest.TestCase):
    def test_transcribes(self):
        ctx = MagicMock()
        ctx.tools.call.return_value = SimpleNamespace(
            ok=True, value={"text": "hello devon"})
        self.assertEqual("hello devon",
                         pp.transcribe_voice_media(ctx, _msg(_media())))
        ctx.tools.call.assert_called_once()

    def test_no_media_returns_empty(self):
        ctx = MagicMock()
        self.assertEqual("", pp.transcribe_voice_media(ctx, _msg()))
        ctx.tools.call.assert_not_called()

    def test_failure_returns_empty(self):
        ctx = MagicMock()
        ctx.tools.call.return_value = SimpleNamespace(ok=False, value={})
        self.assertEqual("", pp.transcribe_voice_media(ctx, _msg(_media())))

    def test_exception_returns_empty(self):
        ctx = MagicMock()
        ctx.tools.call.side_effect = RuntimeError("boom")
        self.assertEqual("", pp.transcribe_voice_media(ctx, _msg(_media())))


class OggTests(unittest.TestCase):
    def test_ogg_passthrough(self):
        self.assertEqual("/tmp/x.ogg", pp.to_ogg("/tmp/x.ogg"))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            pp.synthesize_voice_reply(MagicMock(), "")

    def test_no_ffmpeg(self):
        with patch("subprocess.run",
                   side_effect=FileNotFoundError("no ffmpeg")):
            with self.assertRaises(RuntimeError) as cm:
                pp.to_ogg("/tmp/x.wav")
            self.assertIn("ffmpeg", str(cm.exception))


class SynthesizeTests(unittest.TestCase):
    def test_tts_failure_raises(self):
        ctx = MagicMock()
        ctx.tools.call.return_value = SimpleNamespace(ok=False, error="down")
        with self.assertRaises(RuntimeError):
            pp.synthesize_voice_reply(ctx, "hi")


if __name__ == "__main__":
    unittest.main()
