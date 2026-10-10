"""Acoustic voice-prints: Devon's own measurement of reference clips."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from nomorals.voice.tts import (
    VoiceLibrary,
    VoiceProfile,
    probe_reference_audio,
    voice_print,
    voice_print_distance,
)


def _voice_wav(path, f0=180.0, sr=11025, secs=4.0):
    n = int(sr * secs)
    vals = array("d", [0.0]) * n
    for i in range(n):
        t = i / sr
        env = math.sin(math.pi * i / n)  # voiced swell
        vals[i] = 0.4 * math.sin(2 * math.pi * f0 * t) * env
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in vals))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestVoicePrint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.low = _voice_wav(os.path.join(self.d, "low.wav"), f0=150.0)
        self.high = _voice_wav(os.path.join(self.d, "high.wav"), f0=300.0)

    def test_print_measures_f0(self):
        vp = voice_print(self.low)
        self.assertTrue(vp["ok"], vp["warnings"])
        f0 = vp["features"]["mean_f0_hz"]
        self.assertGreater(f0, 120)
        self.assertLess(f0, 180)

    def test_print_has_verdict(self):
        vp = voice_print(self.low)
        self.assertIn(vp["verdict"],
                      ("good reference", "usable with caveats",
                       "poor reference"))

    def test_short_clip_warns(self):
        short = _voice_wav(os.path.join(self.d, "short.wav"), secs=1.5)
        vp = voice_print(short)
        self.assertTrue(vp["ok"])
        self.assertTrue(any("short" in w for w in vp["warnings"]))

    def test_unreadable_never_raises(self):
        vp = voice_print(os.path.join(self.d, "nope.wav"))
        self.assertFalse(vp["ok"])
        self.assertEqual(vp["verdict"], "unreadable")

    def test_distance_same_voice_near_zero(self):
        a = voice_print(self.low)
        b = voice_print(self.low)
        self.assertLess(voice_print_distance(a, b), 0.05)

    def test_distance_different_voices_positive(self):
        a = voice_print(self.low)
        b = voice_print(self.high)
        d = voice_print_distance(a, b)
        self.assertGreater(d, 0.1)
        # and symmetric-ish
        self.assertAlmostEqual(d, voice_print_distance(b, a), places=6)

    def test_distance_missing_features_is_inf(self):
        self.assertEqual(voice_print_distance({}, {}), float("inf"))


class TestProbeIncludesPrint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.clip = _voice_wav(os.path.join(self.d, "v.wav"))

    def test_probe_carries_voice_print(self):
        info = probe_reference_audio(self.clip)
        self.assertTrue(info["ok"])
        self.assertIn("voice_print", info)
        self.assertIn("verdict", info)
        self.assertTrue(info["voice_print"]["ok"])


class TestLibraryVoicePrint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.low = _voice_wav(os.path.join(self.d, "low.wav"), f0=150.0)
        self.high = _voice_wav(os.path.join(self.d, "high.wav"), f0=300.0)
        self.lib = VoiceLibrary(os.path.join(self.d, "lib"))

    def test_upload_computes_print(self):
        p = self.lib.upload_voice("low", self.low, backend="xtts")
        self.assertTrue((p.voice_print or {}).get("ok"))
        self.assertIn("verdict", p.voice_print)

    def test_print_persists_across_reload(self):
        self.lib.upload_voice("low", self.low, backend="xtts")
        lib2 = VoiceLibrary(os.path.join(self.d, "lib"))
        p = lib2.get("low")
        self.assertIsNotNone(p)
        self.assertTrue((p.voice_print or {}).get("ok"))
        self.assertGreater(
            (p.voice_print.get("features") or {}).get("mean_f0_hz", 0), 100)

    def test_match_voice_finds_closest(self):
        self.lib.upload_voice("low", self.low, backend="xtts")
        self.lib.upload_voice("high", self.high, backend="chatterbox")
        res = self.lib.match_voice(self.high)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["name"], "high")

    def test_match_voice_no_voices(self):
        res = self.lib.match_voice(self.low)
        self.assertFalse(res["ok"])
        self.assertIn("no cloned voices", res["reason"])

    def test_match_voice_bad_clip(self):
        self.lib.upload_voice("low", self.low, backend="xtts")
        res = self.lib.match_voice(os.path.join(self.d, "nope.wav"))
        self.assertFalse(res["ok"])

    def test_list_shows_backend(self):
        self.lib.upload_voice("low", self.low, backend="xtts")
        rows = self.lib.list()
        self.assertEqual(rows[0]["backend"], "xtts")

    def test_voice_profile_dataclass_roundtrip(self):
        p = VoiceProfile(name="x", voice_print={"ok": True})
        d = p.to_dict()
        self.assertEqual(d["voice_print"], {"ok": True})


if __name__ == "__main__":
    unittest.main()
