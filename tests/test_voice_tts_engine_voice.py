"""Per-engine TTS voice handling: an edge-tts / Microsoft voice id
must never reach a local engine's ``-v``.

Regression: the voice-note reply path passes its default voice
(``en-NG-EzinneNeural`` — an edge-tts id) into whatever engine
auto-detection picks. On an espeak-ng machine espeak rejected the
unknown ``-v``, TTS raised, and every voice reply silently downgraded
to text. /tts passes no voice, so it is unaffected.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.tools import audio as audio_mod

EDGE_ID = "en-NG-EzinneNeural"


class VoiceForEngineTest(unittest.TestCase):
    def test_edge_id_dropped_for_espeak(self) -> None:
        self.assertEqual("", audio_mod._voice_for_engine("espeak_ng", EDGE_ID))

    def test_edge_id_dropped_for_pico_and_say(self) -> None:
        self.assertEqual("", audio_mod._voice_for_engine("pico2wave", EDGE_ID))
        self.assertEqual("", audio_mod._voice_for_engine("say", EDGE_ID))

    def test_openai_voice_dropped_for_local_engines(self) -> None:
        self.assertEqual("", audio_mod._voice_for_engine("espeak_ng", "alloy"))
        self.assertEqual("", audio_mod._voice_for_engine("say", "nova"))

    def test_edge_id_kept_for_edge_and_api_engines(self) -> None:
        self.assertEqual(EDGE_ID,
                         audio_mod._voice_for_engine("edge_tts", EDGE_ID))
        self.assertEqual("alloy",
                         audio_mod._voice_for_engine("openai_compat", "alloy"))

    def test_local_voice_passes_through(self) -> None:
        self.assertEqual("en-gb",
                         audio_mod._voice_for_engine("espeak_ng", "en-gb"))
        self.assertEqual("en",
                         audio_mod._voice_for_engine("espeak_ng", "en"))
        self.assertEqual("Samantha",
                         audio_mod._voice_for_engine("say", "Samantha"))

    def test_empty_voice_unchanged(self) -> None:
        self.assertEqual("", audio_mod._voice_for_engine("espeak_ng", ""))


class TtsEspeakVoiceTest(unittest.TestCase):
    """tts() on a simulated espeak-ng-only machine."""

    def _run_tts(self, voice: str, tmp: str) -> tuple[dict, list[str]]:
        argv_seen: list[str] = []

        def _fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
            argv_seen.extend(str(a) for a in argv)
            # espeak writes the wav to the path after -w.
            if "-w" in argv:
                out = Path(argv[argv.index("-w") + 1])
                out.write_bytes(b"RIFF" + b"\x00" * 100)
            return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

        with mock.patch.object(audio_mod, "detect_tts_engines",
                               return_value=["espeak_ng"]), \
                mock.patch("subprocess.run", side_effect=_fake_run):
            result = audio_mod.tts(
                "hello there", engine="auto", voice=voice, out_dir=tmp)
        return result, argv_seen

    def test_default_edge_voice_produces_voice_note(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="nm-tts-") as tmp:
            result, argv = self._run_tts(EDGE_ID, tmp)
            self.assertEqual(result["engine"], "espeak_ng")
            self.assertTrue(Path(result["path"]).is_file())
            # The edge-tts id must NOT be passed verbatim to espeak-ng:
            # no -v flag at all (engine default speaks).
            self.assertNotIn("-v", argv)
            self.assertNotIn(EDGE_ID, argv)

    def test_no_voice_passes_no_v_flag(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="nm-tts-") as tmp:
            _result, argv = self._run_tts("", tmp)
            self.assertNotIn("-v", argv)

    def test_legit_espeak_voice_passes_through(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="nm-tts-") as tmp:
            _result, argv = self._run_tts("en-gb", tmp)
            self.assertIn("-v", argv)
            self.assertEqual(argv[argv.index("-v") + 1], "en-gb")


if __name__ == "__main__":
    unittest.main()
