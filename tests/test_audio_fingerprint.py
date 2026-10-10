"""Devon's own audio fingerprinting + analysis. All offline, no API."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from nomorals.audio.fingerprint import (
    AudioAnalysis,
    AudioReadError,
    FingerprintDB,
    analyze,
    describe_audio,
    fingerprint,
    index_own,
    match_local_db,
    read_mono,
)


def _tone_wav(path, freqs, sr=11025, secs=6, noise=0.0):
    import random
    rng = random.Random(42)
    n = int(sr * secs)
    vals = array("d", [0.0]) * n
    for i in range(n):
        t = i / sr
        v = sum(math.sin(2 * math.pi * f * t) * 0.25 for f in freqs)
        if noise:
            v += rng.gauss(0, noise)
        vals[i] = v * 0.9
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in vals))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestReadMono(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_reads_wav(self):
        p = _tone_wav(os.path.join(self.d, "a.wav"), [440.0])
        samples, sr = read_mono(p)
        self.assertEqual(sr, 11025)
        self.assertGreater(len(samples), 1000)

    def test_missing_file_raises_honest_error(self):
        with self.assertRaises(AudioReadError):
            read_mono(os.path.join(self.d, "nope.wav"))

    def test_max_seconds_caps(self):
        p = _tone_wav(os.path.join(self.d, "long.wav"), [440.0], secs=10)
        samples, sr = read_mono(p, max_seconds=2.0)
        self.assertLessEqual(len(samples), int(2.2 * sr))


class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.a = _tone_wav(os.path.join(self.d, "a.wav"),
                           [220.0, 277.18, 329.63])
        self.b = _tone_wav(os.path.join(self.d, "b.wav"),
                           [196.0, 246.94, 293.66])

    def test_extracts_hashes(self):
        hs = fingerprint(self.a)
        self.assertGreater(len(hs), 50)
        for h, t in hs[:5]:
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(t, 0.0)

    def test_quiet_audio_extracts_nothing_useful(self):
        p = _tone_wav(os.path.join(self.d, "q.wav"), [440.0], noise=0.0)
        # nearly silent file
        n = int(11025 * 2)
        vals = array("d", [0.0]) * n
        pcm = array("h", (0 for _ in vals))
        with wave.open(p, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(11025)
            wf.writeframes(pcm.tobytes())
        hs = fingerprint(p)
        self.assertEqual(hs, [])


class TestFingerprintDB(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.a = _tone_wav(os.path.join(self.d, "a.wav"),
                           [220.0, 277.18, 329.63])
        self.b = _tone_wav(os.path.join(self.d, "b.wav"),
                           [196.0, 246.94, 293.66])
        self.db = FingerprintDB(os.path.join(self.d, "fp.db"))

    def tearDown(self):
        self.db.close()

    def test_add_and_self_match(self):
        res = self.db.add_track(self.a, title="Song A", artist="Devon")
        self.assertTrue(res["ok"])
        m = self.db.match(self.a)
        self.assertTrue(m["ok"])
        self.assertEqual(m["title"], "Song A")
        self.assertGreaterEqual(m["score"], 0.5)

    def test_different_song_does_not_match(self):
        self.db.add_track(self.a, title="Song A", artist="Devon")
        m = self.db.match(self.b)
        self.assertFalse(m["ok"])

    def test_noisy_version_still_matches(self):
        self.db.add_track(self.a, title="Song A", artist="Devon")
        noisy = _tone_wav(os.path.join(self.d, "a_noisy.wav"),
                          [220.0, 277.18, 329.63], noise=0.05)
        m = self.db.match(noisy)
        self.assertTrue(m["ok"], f"noisy match failed: {m}")

    def test_list_and_remove(self):
        r = self.db.add_track(self.a, title="Song A", artist="Devon")
        tracks = self.db.list_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertTrue(self.db.remove_track(r["track_id"]))
        self.assertEqual(self.db.list_tracks(), [])

    def test_match_never_raises_on_garbage(self):
        bad = os.path.join(self.d, "bad.wav")
        Path(bad).write_bytes(b"not a wav at all")
        m = self.db.match(bad)
        self.assertFalse(m["ok"])
        self.assertIn("reason", m)


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        # A-major-ish chord: analysis should find key A
        self.a = _tone_wav(os.path.join(self.d, "a.wav"),
                           [220.0, 277.18, 329.63], secs=8)

    def test_analysis_fields(self):
        an = analyze(self.a)
        self.assertTrue(an.ok, an.reason)
        self.assertGreater(an.seconds, 7)
        self.assertEqual(an.sample_rate, 11025)
        self.assertLess(an.rms_db, 0)
        self.assertGreater(an.spectral_centroid_hz, 100)

    def test_key_detection(self):
        an = analyze(self.a)
        self.assertEqual(an.key, "A")
        self.assertGreater(an.key_confidence, 0.5)

    def test_describe_is_human(self):
        line = describe_audio(self.a)
        self.assertIn("key of A", line)
        self.assertIn("heard natively", line)

    def test_missing_file_is_honest(self):
        an = analyze(os.path.join(self.d, "nope.wav"))
        self.assertFalse(an.ok)
        self.assertIn("no such file", an.reason)
        self.assertIn("couldn't listen", describe_audio(
            os.path.join(self.d, "nope.wav")))

    def test_speech_scores_lower_than_music(self):
        # bursty "speech-like": amplitude-modulated noise bursts
        import random
        rng = random.Random(1)
        sr = 11025
        n = int(sr * 6)
        vals = array("d", [0.0]) * n
        for i in range(n):
            gate = 1.0 if (i // (sr // 4)) % 2 == 0 else 0.05
            vals[i] = rng.gauss(0, 0.3) * gate
        p = os.path.join(self.d, "speech.wav")
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in vals))
        with wave.open(p, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        music_score = analyze(self.a).music_score
        speech_score = analyze(p).music_score
        self.assertLess(speech_score, music_score)


class TestIndexOwn(unittest.TestCase):
    def test_index_own_never_raises(self):
        res = index_own("/nonexistent/path.wav", title="x")
        self.assertFalse(res["ok"])


if __name__ == "__main__":
    unittest.main()
