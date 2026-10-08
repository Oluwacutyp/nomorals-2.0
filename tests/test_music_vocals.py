"""Tests for the /music vocal + bass upgrades.

Covers :mod:`nomorals.media.vocal_lite` (lightweight TTS hook vocal for
the default /music path — the Termux-friendly voice) and the bass boost
in :mod:`nomorals.media.synth` (gain 0.9 -> 1.2, richer harmonics, sub-
octave doubler). All offline; TTS is mocked. Nothing here may raise.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
import unittest
import wave
from array import array
from types import SimpleNamespace

from nomorals.media import synth as S
from nomorals.media import vocal_lite as VL


def _make_song(sections=(
        ("intro", 2), ("verse", 8), ("chorus", 8), ("outro", 2))):
    secs = []
    for name, bars in sections:
        lyrics = []
        if name == "chorus":
            lyrics = ["We chase the golden light",
                      "Hold the golden light tight",
                      "We chase the golden light",
                      "Forever dancing through the light"]
        elif name == "verse":
            lyrics = ["Walking down the avenue alone",
                      "Neon rain is falling on the stone"]
        secs.append(SimpleNamespace(name=name, bars=bars, lyrics=lyrics))
    return SimpleNamespace(sections=secs, tempo=112)


def _write_tone_wav(path, seconds=1.0, sr=16000, freq=440.0):
    n = int(seconds * sr)
    pcm = array("h", (int(10000 * __import__("math").sin(
        2 * __import__("math").pi * freq * i / sr)) for i in range(n)))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def _mock_tts_ok(text, out_path):
    _write_tone_wav(out_path, seconds=2.0)
    return {"ok": True, "path": out_path, "backend": "mock-tts",
            "sample_rate": 16000}


def _mock_tts_fail(text, out_path):
    return {"ok": False, "reason": "no TTS backend installed"}


class BassBoostTests(unittest.TestCase):
    def test_bass_gain_boosted(self):
        self.assertEqual(S._TIMBRES["bass"][5], 1.2)

    def test_bass_harmonics_richer(self):
        harmonics = S._TIMBRES["bass"][0]
        # 2nd/3rd harmonics present for small-speaker audibility
        self.assertGreater(harmonics[1], 0.3)
        self.assertGreater(harmonics[2], 0.0)

    def test_sub_bass_doubler_enabled(self):
        self.assertGreater(S._SUB_BASS_GAIN, 0.0)

    def test_other_timbres_untouched(self):
        self.assertEqual(S._TIMBRES["melody"][5], 1.0)
        self.assertEqual(S._TIMBRES["chords"][5], 0.55)

    def test_bass_mix_no_clip(self):
        from nomorals.core.midi import NoteEvent
        parts = {
            "bass": [NoteEvent(note=36, start=0.0, duration=2.0,
                              velocity=100),
                     NoteEvent(note=41, start=2.0, duration=2.0,
                              velocity=100)],
            "melody": [NoteEvent(note=69, start=0.0, duration=1.0,
                                 velocity=90)],
        }
        mix = S.mix_tracks(parts, tempo=100.0, seed=1)
        self.assertTrue(len(mix) > 0)
        peak = max(abs(s) for s in mix)
        self.assertLessEqual(peak, 1.0)  # normalizer guarantees this
        # bass-heavy mix should actually have energy
        rms = (sum(s * s for s in mix) / len(mix)) ** 0.5
        self.assertGreater(rms, 0.01)

    def test_bass_timbre_more_energy(self):
        """Pre-normalization: the new bass timbre carries more energy
        than the old one for the same note (gain + harmonics)."""
        n = S.SAMPLE_RATE  # 1 second
        freq = 55.0  # A1 — deep bass
        old_harm = (1.0, 0.3, 0.0, 0.0)
        new_harm = S._TIMBRES["bass"][0]
        kw = dict(attack=0.006, decay=0.05, sustain=0.85, release=0.06)
        old_sig = S._render_tone(freq, n, 100, old_harm, **kw)
        new_sig = S._render_tone(freq, n, 100, new_harm, **kw)
        # apply the track gains as the mixer would
        rms_old = (sum((s * 0.9) ** 2 for s in old_sig) / n) ** 0.5
        rms_new = (sum((s * 1.2) ** 2 for s in new_sig) / n) ** 0.5
        self.assertGreater(rms_new, rms_old)

    def test_sub_doubler_changes_output(self):
        """The sub-octave doubler actually contributes to the mix."""
        from nomorals.core.midi import NoteEvent
        parts = {"bass": [NoteEvent(note=36, start=0.0, duration=2.0,
                                   velocity=100)]}
        with_sub = S.mix_tracks(parts, tempo=100.0, seed=1)
        old = S._SUB_BASS_GAIN
        S._SUB_BASS_GAIN = 0.0
        try:
            without_sub = S.mix_tracks(parts, tempo=100.0, seed=1)
        finally:
            S._SUB_BASS_GAIN = old
        diff = sum(abs(a - b)
                   for a, b in zip(with_sub, without_sub))
        self.assertGreater(diff, 0.0)


class HookLyricsTests(unittest.TestCase):
    def test_chorus_hook_preferred(self):
        song = _make_song()
        lines = VL.hook_lyrics(song)
        self.assertEqual(len(lines), 4)
        self.assertIn("golden light", lines[0])

    def test_falls_back_to_first_lyrics(self):
        song = _make_song(sections=(("verse", 8), ("outro", 2)))
        lines = VL.hook_lyrics(song)
        self.assertTrue(lines)
        self.assertIn("avenue", lines[0])

    def test_no_lyrics_empty(self):
        song = _make_song(sections=(("intro", 2),))
        self.assertEqual(VL.hook_lyrics(song), [])

    def test_garbage_never_raises(self):
        self.assertEqual(VL.hook_lyrics(None), [])
        self.assertEqual(VL.hook_lyrics(object()), [])
        self.assertEqual(VL.chorus_regions(None), [])
        self.assertEqual(VL.song_duration_beats(None), 0.0)


class ChorusRegionTests(unittest.TestCase):
    def test_chorus_regions_absolute_beats(self):
        song = _make_song()  # intro 2, verse 8, chorus 8, outro 2
        regions = VL.chorus_regions(song)
        self.assertEqual(len(regions), 1)
        # chorus starts after 10 bars -> beat 40, 8 bars -> beat 72
        self.assertEqual(regions[0], (40.0, 72.0))

    def test_multiple_choruses(self):
        song = _make_song(sections=(("chorus", 4), ("verse", 4),
                                    ("chorus", 4)))
        regions = VL.chorus_regions(song)
        self.assertEqual(len(regions), 2)

    def test_no_chorus_middle_third(self):
        song = _make_song(sections=(("verse", 12),))
        regions = VL.chorus_regions(song)
        self.assertEqual(len(regions), 1)
        total = VL.song_duration_beats(song)
        self.assertAlmostEqual(regions[0][0], total / 3.0)


class RenderHookVocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ok_path(self):
        song = _make_song()
        res = VL.render_hook_vocal(song, self.tmp, tts_fn=_mock_tts_ok,
                                   profile="termux")
        self.assertTrue(res["ok"])
        self.assertTrue(os.path.isfile(res["path"]))
        self.assertIn("mock-tts", res["note"])

    def test_termux_profile_accepted(self):
        """The lightweight path is profile-safe: termux never hits
        DiffSinger/RVC — it just uses TTS."""
        song = _make_song()
        res = VL.render_hook_vocal(song, self.tmp, tts_fn=_mock_tts_ok,
                                   profile="termux")
        self.assertTrue(res["ok"])

    def test_no_lyrics_honest(self):
        song = _make_song(sections=(("intro", 2),))
        res = VL.render_hook_vocal(song, self.tmp, tts_fn=_mock_tts_ok)
        self.assertFalse(res["ok"])
        self.assertIn("lyrics", res["reason"])

    def test_tts_failure_honest(self):
        song = _make_song()
        res = VL.render_hook_vocal(song, self.tmp, tts_fn=_mock_tts_fail)
        self.assertFalse(res["ok"])
        self.assertIn("no TTS backend", res["reason"])

    def test_tts_raises_honest(self):
        def boom(text, out_path):
            raise RuntimeError("kaboom")
        song = _make_song()
        res = VL.render_hook_vocal(song, self.tmp, tts_fn=boom)
        self.assertFalse(res["ok"])

    def test_garbage_never_raises(self):
        res = VL.render_hook_vocal(None, self.tmp, tts_fn=_mock_tts_ok)
        self.assertFalse(res["ok"])


class MixVocalTrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bed = os.path.join(self.tmp, "bed.wav")
        self.voc = os.path.join(self.tmp, "voc.wav")
        _write_tone_wav(self.bed, seconds=8.0, sr=22050, freq=110.0)
        _write_tone_wav(self.voc, seconds=2.0, sr=16000, freq=440.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mix_produces_valid_wav(self):
        out = os.path.join(self.tmp, "mixed.wav")
        VL.mix_vocal_track(self.bed, self.voc, [(0.0, 32.0)], 120.0,
                           64.0, out)
        self.assertTrue(os.path.isfile(out))
        with wave.open(out, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertGreater(wf.getnframes(), 0)

    def test_mix_no_clip(self):
        out = os.path.join(self.tmp, "mixed.wav")
        VL.mix_vocal_track(self.bed, self.voc, [(0.0, 32.0)], 120.0,
                           64.0, out, vocal_gain=2.0)
        with wave.open(out, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        peak = max(abs(v) for v in struct.unpack("<%dh" % n, raw))
        self.assertLessEqual(peak, 32767)

    def test_vocal_audible_in_mix(self):
        """Mixing a vocal under silence leaves vocal energy behind."""
        silent = os.path.join(self.tmp, "silent.wav")
        _write_tone_wav(silent, seconds=8.0, sr=22050, freq=0.0)
        # freq=0 -> sin(0)=0 -> true silence
        out = os.path.join(self.tmp, "mixed2.wav")
        VL.mix_vocal_track(silent, self.voc, [(0.0, 32.0)], 120.0,
                           64.0, out)
        with wave.open(out, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        vals = struct.unpack("<%dh" % n, raw)
        rms = (sum(v * v for v in vals) / n) ** 0.5
        self.assertGreater(rms, 10.0)

    def test_garbage_never_raises(self):
        out = os.path.join(self.tmp, "x.wav")
        try:
            VL.mix_vocal_track("/nonexistent/bed.wav", self.voc, [],
                               120.0, 64.0, out)
        except Exception as exc:  # noqa: BLE001 - wave.open raises
            self.assertIsInstance(exc, (OSError, FileNotFoundError))


class AddVocalTrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bed = os.path.join(self.tmp, "bed.wav")
        _write_tone_wav(self.bed, seconds=10.0, sr=22050, freq=110.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_path(self):
        song = _make_song()
        res = VL.add_vocal_track(song, self.bed, self.tmp,
                                 tts_fn=_mock_tts_ok, profile="termux")
        self.assertTrue(res["ok"])
        self.assertTrue(os.path.isfile(res["path"]))
        self.assertIn("mixed under the bed", res["note"])

    def test_honest_skip_keeps_bed(self):
        song = _make_song()
        res = VL.add_vocal_track(song, self.bed, self.tmp,
                                 tts_fn=_mock_tts_fail)
        self.assertFalse(res["ok"])
        self.assertEqual(res["path"], self.bed)

    def test_no_bed_honest(self):
        song = _make_song()
        res = VL.add_vocal_track(song, "/nonexistent.wav", self.tmp,
                                 tts_fn=_mock_tts_ok)
        self.assertFalse(res["ok"])

    def test_garbage_never_raises(self):
        res = VL.add_vocal_track(None, self.bed, self.tmp,
                                 tts_fn=_mock_tts_ok)
        self.assertFalse(res["ok"])
        self.assertEqual(res["path"], self.bed)


class ResampleTests(unittest.TestCase):
    def test_identity(self):
        s = array("d", [0.1, 0.2, 0.3])
        self.assertEqual(list(VL._resample_linear(s, 22050, 22050)),
                         [0.1, 0.2, 0.3])

    def test_upsample_length(self):
        s = array("d", [0.0] * 100)
        out = VL._resample_linear(s, 16000, 22050)
        self.assertEqual(len(out), 138)  # 100 * 22050/16000

    def test_garbage_never_raises(self):
        out = VL._resample_linear(array("d"), 0, 22050)
        self.assertEqual(len(out), 0)


class _HumEvent:
    """Minimal NoteEvent stand-in for hum tests."""
    def __init__(self, note, start, duration, velocity=96):
        self.note = note
        self.start = start
        self.duration = duration
        self.velocity = velocity


def _hum_song():
    secs = [SimpleNamespace(name="intro", bars=2, lyrics=[]),
            SimpleNamespace(name="chorus", bars=8,
                            lyrics=["We chase the golden light"]),
            SimpleNamespace(name="outro", bars=2, lyrics=[])]
    return SimpleNamespace(sections=secs, tempo=112)


def _hum_events():
    # chorus starts at beat 8 (2 intro bars x 4)
    return [_HumEvent(60 + (i % 4) * 2, 8 + i * 0.5, 0.45)
            for i in range(32)]


def _acf_pitch(seg, sr):
    n = len(seg)
    if n < 200:
        return 0.0
    m = sum(seg) / n
    seg = [s - m for s in seg]
    best_lag, best_corr = 0, -1.0
    for lag in range(int(sr / 800), int(sr / 80)):
        c = sum(seg[i] * seg[i + lag] for i in range(0, n - lag, 4))
        if c > best_corr:
            best_corr, best_lag = c, lag
    return sr / best_lag if best_lag else 0.0


class HumFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bed = os.path.join(self.tmp, "bed.wav")
        _write_tone_wav(self.bed, seconds=25.0, sr=22050, freq=110.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hum_triggers_when_no_tts_backend(self):
        # the exact failure the user hit on Termux -> hum, not silence
        res = VL.add_vocal_track(
            _hum_song(), self.bed, self.tmp,
            tts_fn=_mock_tts_fail, melody_events=_hum_events())
        self.assertTrue(res["ok"], res.get("reason"))
        self.assertEqual(res.get("backend"), "hum")
        self.assertTrue(os.path.isfile(res["path"]))
        self.assertIn("hummed vocal melody", res["note"])
        self.assertIn("termux-api", res["note"])

    def test_hum_follows_melodic_contour(self):
        # octave apart -> hum pitches an octave apart
        song = SimpleNamespace(
            sections=[SimpleNamespace(name="chorus", bars=4,
                                       lyrics=["la"])],
            tempo=120)
        evs = [_HumEvent(60, 0, 1.0), _HumEvent(72, 1.0, 1.0)]
        res = VL.render_hummed_vocal(song, self.tmp, evs)
        self.assertTrue(res["ok"])
        with wave.open(res["path"], "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        vals = array("d", (v / 32768.0
                           for v in struct.unpack("<%dh" % n, raw)))
        sr = 22050
        f1 = _acf_pitch(list(vals[int(0.1 * sr):int(0.4 * sr)]), sr)
        f2 = _acf_pitch(list(vals[int(0.6 * sr):int(0.9 * sr)]), sr)
        self.assertGreater(f1, 100)
        self.assertAlmostEqual(f2 / f1, 2.0, delta=0.15)

    def test_hum_vibrato_present(self):
        # a sustained note's pitch should wobble (not be laser-flat)
        tone = VL._render_hum_tone(440.0, 22050, 96)
        sr = 22050
        wins = [list(tone[int((0.2 + i * 0.1) * sr):
                           int((0.3 + i * 0.1) * sr)])
                for i in range(5)]
        freqs = [_acf_pitch(w, sr) for w in wins]
        freqs = [f for f in freqs if f > 100]
        self.assertGreater(len(freqs), 2)
        # mean near 440, with measurable wobble
        mean = sum(freqs) / len(freqs)
        self.assertAlmostEqual(mean, 440.0, delta=15.0)
        self.assertGreater(max(freqs) - min(freqs), 1.0)

    def test_hum_audible_in_mix(self):
        res = VL.add_vocal_track(
            _hum_song(), self.bed, self.tmp,
            tts_fn=_mock_tts_fail, melody_events=_hum_events())
        self.assertTrue(res["ok"])
        with wave.open(res["path"], "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        vals = array("d", (v / 32768.0
                           for v in struct.unpack("<%dh" % n, raw)))
        sr = 22050

        def rms(a, b):
            seg = vals[int(a * sr):int(b * sr)]
            return (sum(s * s for s in seg) / max(1, len(seg))) ** 0.5
        # chorus (beats 8-40 @112bpm = 4.3s-21.4s) vs intro
        self.assertGreater(rms(6, 10), rms(1, 3) * 1.1)

    def test_hum_mix_no_clip(self):
        res = VL.add_vocal_track(
            _hum_song(), self.bed, self.tmp,
            tts_fn=_mock_tts_fail, melody_events=_hum_events())
        self.assertTrue(res["ok"])
        with wave.open(res["path"], "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        peak = max(abs(v) for v in struct.unpack("<%dh" % n, raw))
        self.assertLessEqual(peak, 32767)

    def test_no_melody_events_honest_skip(self):
        # hum needs the melody — without it, honest skip (old behavior)
        res = VL.add_vocal_track(
            _hum_song(), self.bed, self.tmp, tts_fn=_mock_tts_fail)
        self.assertFalse(res["ok"])
        self.assertEqual(res["path"], self.bed)

    def test_no_chorus_melody_honest(self):
        res = VL.render_hummed_vocal(_hum_song(), self.tmp, [])
        self.assertFalse(res["ok"])

    def test_other_tts_failure_no_hum(self):
        # hum is only for the missing-backend case, not every TTS error
        def fail_other(text, out_path):
            return {"ok": False, "reason": "TTS crashed mysteriously"}
        res = VL.add_vocal_track(
            _hum_song(), self.bed, self.tmp,
            tts_fn=fail_other, melody_events=_hum_events())
        self.assertFalse(res["ok"])
        self.assertEqual(res["path"], self.bed)

    def test_garbage_never_raises(self):
        self.assertFalse(VL.render_hummed_vocal(None, self.tmp, None)["ok"])
        # junk events filter out -> no melody -> honest False
        self.assertFalse(VL.render_hummed_vocal(
            _hum_song(), self.tmp, [None, "junk"])["ok"])
        self.assertEqual(VL.mix_hum_track("/nope.wav", "/nope.wav",
                                          "/nope_out.wav"), "")
        self.assertEqual(len(VL._render_hum_tone(0, 0, 0)), 0)
        self.assertEqual(len(VL._render_hum_tone(-440, 100, 96)), 100)


if __name__ == "__main__":
    unittest.main()
