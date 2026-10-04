"""VoiceBridge must route through the owner's own engine (UniversalTTS /
UniversalSTT) when one is attached, and keep the legacy VoiceIntegration
path working. No real backends needed — duck-typed stubs."""

from __future__ import annotations

import asyncio
import unittest


def _run(coro):
    return asyncio.run(coro)


class _NewTTS:
    """UniversalTTS-shaped: sync speak() -> {"path": ...}."""

    def __init__(self):
        self.calls = []

    def speak(self, text, voice_name=None, **kwargs):
        self.calls.append((text, voice_name))
        return {"path": "/tmp/new-engine.wav", "backend": "system"}


class _NewSTT:
    """UniversalSTT-shaped: sync transcribe() -> {"text": ...}."""

    def transcribe(self, path, language="en"):
        return {"text": f"heard:{path}", "language": language}


class _Legacy:
    """VoiceIntegration-shaped: .tts.synthesize / .stt.transcribe, no .speak."""

    def __init__(self):
        self.calls = []

    class _TTS:
        async def synthesize(self, text, voice="default", rate="+0%"):
            return "/tmp/legacy.mp3"

    class _STT:
        async def transcribe(self, path, language="en"):
            return "legacy words"

    def __init__(self):
        self.tts = self._TTS()
        self.stt = self._STT()


class VoiceBridgeTests(unittest.TestCase):
    def test_new_tts_engine_used(self):
        from nomorals.voice.bridge import VoiceBridge

        tts = _NewTTS()
        bridge = VoiceBridge(tts)
        path = _run(bridge._synthesize("hello", voice="narrator"))
        self.assertEqual(path, "/tmp/new-engine.wav")
        self.assertEqual(tts.calls, [("hello", "narrator")])

    def test_new_tts_default_voice_maps_to_none(self):
        from nomorals.voice.bridge import VoiceBridge

        tts = _NewTTS()
        bridge = VoiceBridge(tts)
        _run(bridge._synthesize("hi"))
        self.assertEqual(tts.calls, [("hi", None)])

    def test_new_stt_engine_used(self):
        from nomorals.voice.bridge import VoiceBridge

        bridge = VoiceBridge(_NewSTT())
        self.assertEqual(_run(bridge.transcribe_voice("/tmp/a.wav")),
                         "heard:/tmp/a.wav")

    def test_legacy_engine_still_works(self):
        from nomorals.voice.bridge import VoiceBridge

        bridge = VoiceBridge(_Legacy())
        self.assertEqual(_run(bridge._synthesize("hi")), "/tmp/legacy.mp3")
        self.assertEqual(_run(bridge.transcribe_voice("/tmp/a.wav")),
                         "legacy words")

    def test_no_engine_raises_clear_error(self):
        from nomorals.voice.bridge import VoiceBridge

        bridge = VoiceBridge()
        with self.assertRaises(RuntimeError) as cm:
            _run(bridge._synthesize("hi"))
        self.assertIn("no voice engine", str(cm.exception))
        # transcribe degrades to "" (never kills a caller)
        self.assertEqual(_run(bridge.transcribe_voice("/tmp/a.wav")), "")

    def test_send_without_adapter_returns_false(self):
        from nomorals.voice.bridge import VoiceBridge

        bridge = VoiceBridge(_NewTTS())
        self.assertFalse(_run(bridge.send_voice_telegram(123, "hi")))
        self.assertFalse(_run(bridge.send_voice_auto("telegram", 123, "hi")))
        self.assertFalse(_run(bridge.send_voice_auto("bogus", 123, "hi")))


if __name__ == "__main__":
    unittest.main()
