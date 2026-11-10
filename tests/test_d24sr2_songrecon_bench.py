"""Doc 24 SR2 — SongRecon §9 benchmark + class-2 hardening pins.

The benchmark harness is the executable proof: it drives the real
engine over deterministic synthesized ground truth. These tests pin
fixture determinism, ground-truth integrity (clean excerpt found,
non-catalog query refused), the two measured hardenings (playback-
rate retry for coupled speed/pitch edits; distinct-landmark breadth
floors for acceptance), and the measured-calibration statement the
receipt is allowed to carry.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from nomorals.audio.fingerprint import (
    MEASURED_CALIBRATION,
    FingerprintDB,
)
from nomorals.audio.songrecon_bench import (
    apply_speed,
    excerpt,
    fingerprint_fixture_check,
    run_benchmark,
    synth_track,
    write_wav,
)


def _tone_wav(path, freqs, sr=11025, secs=6):
    n = int(sr * secs)
    vals = array("d", [0.0]) * n
    for i in range(n):
        t = i / sr
        vals[i] = sum(math.sin(2 * math.pi * f * t) * 0.25
                      for f in freqs) * 0.9
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in vals))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestFixtureDeterminism(unittest.TestCase):
    def test_same_seed_same_audio(self):
        a = synth_track(20000, 6.0)
        b = synth_track(20000, 6.0)
        self.assertEqual(list(a), list(b))

    def test_different_seed_different_audio(self):
        a = synth_track(20000, 6.0)
        b = synth_track(20001, 6.0)
        self.assertNotEqual(list(a), list(b))

    def test_fingerprint_fixture_check(self):
        self.assertTrue(fingerprint_fixture_check())


class TestGroundTruthAndHardening(unittest.TestCase):
    """Small real catalog, real engine, measured failure modes."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.db = FingerprintDB(os.path.join(cls.d, "fp.db"))
        cls.samples = {}
        for i in range(3):
            samples = synth_track(20000 + i, 18.0)
            cls.samples[i] = samples
            path = write_wav(os.path.join(cls.d, f"t{i}.wav"), samples)
            res = cls.db.add_track(path, title=f"Bench Track {i:02d}",
                                   artist="SongRecon Bench")
            assert res["ok"], res

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _query(self, samples, name="q.wav"):
        return write_wav(os.path.join(self.d, name), samples)

    def test_clean_excerpt_identifies(self):
        m = self.db.match(self._query(excerpt(self.samples[1], 2.0, 8.0)))
        self.assertTrue(m["ok"])
        self.assertEqual(m["title"], "Bench Track 01")

    def test_non_catalog_track_is_no_match(self):
        other = synth_track(40000, 18.0)
        m = self.db.match(self._query(excerpt(other, 2.0, 8.0)))
        self.assertFalse(m["ok"])

    def test_sped_up_excerpt_matches_with_rate_correction(self):
        query = apply_speed(excerpt(self.samples[1], 2.0, 8.0), 1.06)
        m = self.db.match(self._query(query))
        self.assertTrue(m["ok"], m.get("reason"))
        self.assertEqual(m["title"], "Bench Track 01")
        self.assertAlmostEqual(m["rate_correction"], 1.06, delta=0.011)
        alignment = m["receipt"]["alignment"]
        self.assertAlmostEqual(alignment["rate_correction"], 1.06,
                               delta=0.011)

    def test_slowed_excerpt_matches_with_rate_correction(self):
        query = apply_speed(excerpt(self.samples[2], 2.0, 8.0), 0.94)
        m = self.db.match(self._query(query))
        self.assertTrue(m["ok"], m.get("reason"))
        self.assertEqual(m["title"], "Bench Track 02")
        self.assertAlmostEqual(m["rate_correction"], 0.94, delta=0.011)

    def test_transposed_tone_chord_is_not_the_same_song(self):
        # The measured retry trap: a whole-tone-transposed three-tone
        # chord is a *different* fixture song, yet a rate hypothesis
        # maps it onto the catalog chord at fraction 1.000 on ~42
        # distinct landmarks. Strict retry breadth refuses it.
        d = tempfile.mkdtemp()
        db = FingerprintDB(os.path.join(d, "tones.db"))
        try:
            a = _tone_wav(os.path.join(d, "a.wav"),
                          [220.0, 277.18, 329.63])
            b = _tone_wav(os.path.join(d, "b.wav"),
                          [196.0, 246.94, 293.66])
            db.add_track(a, title="Song A", artist="Devon")
            self.assertTrue(db.match(a)["ok"])
            self.assertFalse(db.match(b)["ok"])
        finally:
            db.close()


class TestQuickBenchmark(unittest.TestCase):
    def test_quick_run_structure_and_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            out1 = os.path.join(td, "run1")
            report = run_benchmark(out1, n_tracks=4,
                                   track_seconds=18.0, quick=True)
            self.assertEqual(report["schema"], "songrecon.bench/v1")
            self.assertTrue(os.path.isfile(
                os.path.join(out1, "report.json")))
            self.assertTrue(os.path.isfile(
                os.path.join(out1, "report.md")))
            by_name = {c["name"]: c for c in report["conditions"]}
            self.assertEqual(by_name["clean_full"]["accept_accuracy"],
                             1.0)
            self.assertEqual(by_name["excerpt_3s"]["top1_accuracy"],
                             1.0)
            self.assertEqual(
                report["negatives"]["false_candidate_rate"], 0.0)
            # Calibration bins span top-1 fractions at/over the
            # acceptance threshold (0.12): every such row lands in
            # exactly one bin, and sub-threshold rows in none.
            binned = sum(1 for r in report["rows"]
                         if r["top1_score"] >= 0.12)
            self.assertEqual(
                sum(c["n"] for c in report["calibration"]), binned)
            out2 = os.path.join(td, "run2")
            again = run_benchmark(out2, n_tracks=4,
                                  track_seconds=18.0, quick=True)
            strip = ("generated_at", "wall_s", "median_query_s")
            def stable(rep):
                return {
                    "conditions": [
                        {k: v for k, v in c.items() if k not in strip}
                        for c in rep["conditions"]],
                    "negatives": {
                        k: v for k, v in rep["negatives"].items()
                        if k != "rows"},
                    "calibration": rep["calibration"],
                    "accepted_precision": rep["accepted_precision"],
                }
            self.assertEqual(stable(report), stable(again))


class TestMeasuredCalibration(unittest.TestCase):
    def test_receipt_carries_the_measured_statement(self):
        d = tempfile.mkdtemp()
        db = FingerprintDB(os.path.join(d, "fp.db"))
        try:
            samples = synth_track(20000, 18.0)
            path = write_wav(os.path.join(d, "t.wav"), samples)
            db.add_track(path, title="Bench Track 00",
                         artist="SongRecon Bench")
            query = write_wav(os.path.join(d, "q.wav"),
                              excerpt(samples, 2.0, 8.0))
            m = db.match(query)
            self.assertTrue(m["ok"])
            strength = m["receipt"]["match_strength"]
            self.assertIn("not a probability", strength["calibration"])
            measured = strength["measured_calibration"]
            self.assertEqual(measured, MEASURED_CALIBRATION)
            self.assertLessEqual(measured["accepted_correct"],
                                 measured["accepted"])
            self.assertEqual(measured["negatives_accepted"], 0)
            self.assertIn("synthesized", measured["fixture_scope"])
            self.assertNotIn("confidence",
                             json.dumps(m["receipt"]).lower())
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
