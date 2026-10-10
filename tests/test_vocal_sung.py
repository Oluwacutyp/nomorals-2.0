"""Melody-guided sung vocals: lyrics sung on their own melody phrases."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import wave
from array import array
from types import SimpleNamespace

from nomorals.media.vocal_lite import (
    _phrase_median_midi,
    _phrases,
    add_vocal_track,
    chorus_regions,
    render_hook_vocal,
    render_sung_vocal,
    section_regions,
    song_duration_beats,
)


def _fake_tts_factory(f0=180.0, sr=22050, secs=1.0):
    def _tts(text, out_path):
        n = int(sr * secs)
        vals = array("d", [0.0]) * n
        for i in range(n):
            t = i / sr
            vals[i] = 0.4 * math.sin(2 * math.pi * f0 * t) * math.sin(
                math.pi * i / n)
        pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                          for s in vals))
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return {"ok": True, "path": out_path, "backend": "fake"}
    return _tts


class Ev:
    def __init__(self, note, start, duration):
        self.note = note
        self.start = start
        self.duration = duration
        self.velocity = 96


def _song():
    verse = SimpleNamespace(name="verse", bars=4,
                            lyrics=["walking slow", "through the rain"])
    chorus = SimpleNamespace(name="chorus", bars=4,
                             lyrics=["we rise up", "into the light"])
    return SimpleNamespace(sections=[verse, chorus], tempo=100)


def _melody():
    # verse phrases at beats 0-4, chorus phrases at beats 16-24
    return [Ev(69, 0.0, 0.5), Ev(72, 0.5, 0.5),
            Ev(76, 4.0, 0.5), Ev(74, 4.5, 0.5),
            Ev(67, 16.0, 0.5), Ev(71, 16.5, 0.5),
            Ev(74, 20.0, 0.5), Ev(79, 20.5, 1.0)]


def _bed_wav(path, sr=22050, secs=4.0):
    n = int(sr * secs)
    vals = array("d", [0.3 * math.sin(2 * math.pi * 110 * i / sr)
                       for i in range(n)])
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in vals))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


class TestSectionRegions(unittest.TestCase):
    def test_regions_cover_song(self):
        regs = section_regions(_song())
        self.assertEqual(len(regs), 2)
        self.assertEqual(regs[0][0], "verse")
        self.assertEqual(regs[1][0], "chorus")
        # 4 bars each at 4 beats
        self.assertEqual(regs[0][1:], (0.0, 16.0))
        self.assertEqual(regs[1][1:], (16.0, 32.0))

    def test_chorus_regions_still_works(self):
        regs = chorus_regions(_song())
        self.assertEqual(regs, [(16.0, 32.0)])

    def test_duration(self):
        self.assertEqual(song_duration_beats(_song()), 32.0)


class TestPhrases(unittest.TestCase):
    def test_gap_breaks_phrase(self):
        evs = [Ev(60, 0.0, 0.5), Ev(62, 0.5, 0.5), Ev(64, 5.0, 0.5)]
        ph = _phrases(evs, gap_beats=1.0)
        self.assertEqual(len(ph), 2)
        self.assertEqual(len(ph[0]), 2)

    def test_median_midi(self):
        self.assertEqual(_phrase_median_midi(
            [Ev(60, 0, 1), Ev(64, 1, 1), Ev(67, 2, 1)]), 64.0)
        self.assertEqual(_phrase_median_midi([]), 0.0)


class TestSungVocal(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_sings_every_line(self):
        res = render_sung_vocal(_song(), self.d, _melody(),
                                tts_fn=_fake_tts_factory())
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["lines"], 4)
        self.assertTrue(os.path.isfile(res["path"]))
        self.assertGreater(os.path.getsize(res["path"]), 1000)

    def test_lines_pitch_shifted_toward_phrases(self):
        res = render_sung_vocal(_song(), self.d, _melody(),
                                tts_fn=_fake_tts_factory(f0=180.0))
        self.assertTrue(res["ok"])
        self.assertEqual(res["shifted"], 4)

    def test_no_melody_is_honest_failure(self):
        res = render_sung_vocal(_song(), self.d, [],
                                tts_fn=_fake_tts_factory())
        self.assertFalse(res["ok"])

    def test_no_lyrics_is_honest_failure(self):
        song = SimpleNamespace(sections=[], tempo=100)
        res = render_sung_vocal(song, self.d, _melody(),
                                tts_fn=_fake_tts_factory())
        self.assertFalse(res["ok"])

    def test_failing_tts_is_honest(self):
        def bad_tts(text, out):
            return {"ok": False, "reason": "no TTS backend installed"}
        res = render_sung_vocal(_song(), self.d, _melody(), tts_fn=bad_tts)
        self.assertFalse(res["ok"])

    def test_hook_still_works(self):
        res = render_hook_vocal(_song(), self.d,
                                tts_fn=_fake_tts_factory())
        self.assertTrue(res["ok"], res)


class TestAddVocalTrackModes(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.bed = _bed_wav(os.path.join(self.d, "bed.wav"))

    def test_auto_prefers_sung(self):
        res = add_vocal_track(_song(), self.bed, self.d,
                              tts_fn=_fake_tts_factory(),
                              melody_events=_melody())
        self.assertTrue(res["ok"], res)
        self.assertEqual(res.get("vocal_mode"), "sung")
        self.assertTrue(os.path.isfile(res["path"]))

    def test_explicit_hook(self):
        res = add_vocal_track(_song(), self.bed, self.d,
                              tts_fn=_fake_tts_factory(),
                              melody_events=_melody(), vocal_mode="hook")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res.get("vocal_mode"), "hook")

    def test_auto_falls_back_to_hook_without_melody(self):
        res = add_vocal_track(_song(), self.bed, self.d,
                              tts_fn=_fake_tts_factory(),
                              melody_events=None)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res.get("vocal_mode"), "hook")

    def test_auto_falls_back_to_hum_without_tts(self):
        def no_tts(text, out):
            return {"ok": False, "reason": "no TTS backend installed"}
        res = add_vocal_track(_song(), self.bed, self.d,
                              tts_fn=no_tts, melody_events=_melody())
        self.assertTrue(res["ok"], res)
        self.assertEqual(res.get("vocal_mode"), "hum")

    def test_missing_bed_is_honest(self):
        res = add_vocal_track(_song(), os.path.join(self.d, "no.wav"),
                              self.d, tts_fn=_fake_tts_factory())
        self.assertFalse(res["ok"])
        self.assertIn("no bed", res["reason"])


if __name__ == "__main__":
    unittest.main()
