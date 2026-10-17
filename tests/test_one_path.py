"""Infra spec P1 (§4.2): one path per job.

- One STT-failure formatter: brain + pingpong alias errors.short_stt_reason.
- One system-TTS order: TTS_ENGINE_ORDER, shared by /tts (detect_tts_engines)
  and the vocal path (SystemTTSBackend.detect).
- Transcript cache + media taxonomy already covered by
  test_voice_note_routing / test_telegram_audio_inbound (not duplicated).
"""
from __future__ import annotations

import shutil
import sys
import types
import unittest
from unittest import mock


class TestOneSttReason(unittest.TestCase):
    def test_single_implementation(self):
        from nomorals.agents.partner.brain import _short_stt_reason as brain_reason
        from nomorals.core.errors import short_stt_reason
        from nomorals.voice.pingpong import _short_stt_reason as pingpong_reason
        self.assertIs(brain_reason, short_stt_reason)
        self.assertIs(pingpong_reason, short_stt_reason)

    def test_reason_keeps_real_cause(self):
        from nomorals.core.errors import short_stt_reason
        self.assertIn("401", short_stt_reason(Exception("STT API 401: bad key")))
        self.assertEqual(short_stt_reason(Exception("no STT backend")), "no STT backend")
        self.assertIn("EOFError", short_stt_reason(EOFError()))


class TestOneTtsOrder(unittest.TestCase):
    def test_canonical_order(self):
        from nomorals.tools.audio import TTS_ENGINE_ORDER
        self.assertEqual(TTS_ENGINE_ORDER, ("espeak_ng", "pico2wave", "say", "edge_tts"))

    def test_detect_follows_canonical_order(self):
        from nomorals.tools.audio import detect_tts_engines
        fake_edge = types.ModuleType("edge_tts")

        def fake_which(name):
            return {"/usr/bin/espeak-ng": None}  # placeholder, replaced below

        present = {"espeak-ng": "/usr/bin/espeak-ng", "pico2wave": "/usr/bin/pico2wave",
                   "say": "/usr/bin/say"}

        def which(name):
            return present.get(name)

        with mock.patch.object(shutil, "which", side_effect=which), \
                mock.patch.dict(sys.modules, {"edge_tts": fake_edge}):
            self.assertEqual(detect_tts_engines(),
                             ["espeak_ng", "pico2wave", "say", "edge_tts"])
        # espeak-ng missing -> pico2wave first, order otherwise unchanged.
        present.pop("espeak-ng")
        with mock.patch.object(shutil, "which", side_effect=which), \
                mock.patch.dict(sys.modules, {"edge_tts": fake_edge}):
            self.assertEqual(detect_tts_engines(),
                             ["pico2wave", "say", "edge_tts"])

    def test_systemtts_never_picks_powershell_when_espeak_present(self):
        from nomorals.voice.tts import SystemTTSBackend

        def which(name):
            return {"espeak-ng": "C:\\espeak-ng.exe",
                    "powershell": "C:\\pwsh.exe"}.get(name)

        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(shutil, "which", side_effect=which):
            self.assertEqual(SystemTTSBackend.detect(),
                             ("espeak", "C:\\espeak-ng.exe"))

    def test_systemtts_powershell_only_when_canonical_empty(self):
        from nomorals.voice.tts import SystemTTSBackend

        def which(name):
            return {"powershell": "C:\\pwsh.exe"}.get(name)

        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(shutil, "which", side_effect=which):
            self.assertEqual(SystemTTSBackend.detect(),
                             ("powershell", "C:\\pwsh.exe"))

    def test_systemtts_say_only_on_darwin(self):
        from nomorals.voice.tts import SystemTTSBackend

        def which(name):
            return {"say": "/usr/bin/say"}.get(name)

        with mock.patch.object(sys, "platform", "plan9"), \
                mock.patch.object(shutil, "which", side_effect=which):
            self.assertEqual(SystemTTSBackend.detect(), (None, None))


class TestSttResolution(unittest.TestCase):
    def test_provider_api_needs_endpoint(self):
        from nomorals.tools.audio import detect_stt
        self.assertEqual(detect_stt("api", base_url="https://x/v1", api_key="k"),
                         "openai_compat")
        self.assertIsNone(detect_stt("api"))
        self.assertIsNone(detect_stt("bogus-provider"))

    def test_configured_provider_reaches_settings(self):
        from nomorals.core.config import load_settings
        settings = load_settings(env={"NM_VOICE_STT_PROVIDER": "api"},
                                 use_env_file=False)
        self.assertEqual(settings.audio.stt_provider, "api")


if __name__ == "__main__":
    unittest.main()
