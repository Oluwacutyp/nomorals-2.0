"""Devon's own DSP toolbox. All offline, deterministic where stated."""

from __future__ import annotations

import math
import os
import random
import tempfile
import unittest
import wave
from array import array

from nomorals.audio.dsp import (
    EFFECTS,
    EffectChain,
    compressor,
    dehum,
    echo,
    enhance,
    eq_3band,
    fade_edges,
    mix_under,
    normalize_peak,
    notch,
    pitch_shift,
    read_mono_wav,
    remove_dc,
    reverb,
    soft_limiter,
    spectral_gate,
    trim_silence,
    write_mono_wav,
)


def _sig(sr=22050, secs=2.0, freq=200.0, hum=True, noise=0.05, seed=7):
    rng = random.Random(seed)
    n = int(sr * secs)
    s = array("d", [0.0]) * n
    for i in range(n):
        t = i / sr
        v = 0.5 * math.sin(2 * math.pi * freq * t)
        if hum:
            v += 0.15 * math.sin(2 * math.pi * 60 * t)
        if noise:
            v += rng.gauss(0, noise)
        s[i] = v
    return s, sr


def _rms(x):
    return math.sqrt(sum(v * v for v in x) / max(1, len(x)))


def _band(x, sr, freq):
    n = len(x)
    re = sum(v * math.cos(2 * math.pi * freq * i / sr)
             for i, v in enumerate(x))
    im = sum(v * math.sin(2 * math.pi * freq * i / sr)
             for i, v in enumerate(x))
    return math.hypot(re, im) / n


class TestPrimitives(unittest.TestCase):
    def test_normalize_peak(self):
        s, _ = _sig()
        out = normalize_peak(s, 0.89)
        self.assertAlmostEqual(max(abs(v) for v in out), 0.89, places=2)

    def test_remove_dc(self):
        s, _ = _sig()
        s = array("d", (v + 0.3 for v in s))
        out = remove_dc(s)
        self.assertAlmostEqual(sum(out) / len(out), 0.0, places=6)

    def test_trim_silence(self):
        sr = 22050
        s, _ = _sig(secs=2.0)
        padded = array("d", [0.0]) * sr + s + array("d", [0.0]) * sr
        out = trim_silence(padded, sr, pad_s=0.0)
        self.assertLess(len(out), len(padded))
        self.assertGreater(len(out), len(s) * 0.9)

    def test_fade_edges_kills_click(self):
        s = array("d", [1.0]) * 1000
        out = fade_edges(s, 22050, fade_s=0.01)
        self.assertAlmostEqual(out[0], 0.0, places=6)
        self.assertAlmostEqual(out[-1], 0.0, places=6)
        self.assertAlmostEqual(out[500], 1.0, places=6)

    def test_notch_kills_hum(self):
        s, sr = _sig(noise=0.0)
        before = _band(s, sr, 60)
        out = notch(s, sr, 60.0)
        # measure past the filter's startup transient
        steady = out[sr // 2:]
        after = _band(steady, sr, 60)
        self.assertLess(after, before * 0.05)
        # the 200 Hz tone survives
        self.assertGreater(_band(steady, sr, 200), _band(s, sr, 200) * 0.9)

    def test_dehum(self):
        s, sr = _sig(noise=0.0)
        out = dehum(s, sr)
        steady = out[sr // 2:]
        self.assertLess(_band(steady, sr, 60), _band(s, sr, 60) * 0.05)

    def test_spectral_gate_reduces_noise(self):
        s, sr = _sig(secs=4.0, noise=0.2)
        out = spectral_gate(s, sr, reduction_db=12.0)
        # noise floor region: first 0.5s is tone too — compare quiet gaps
        self.assertLessEqual(_rms(out), _rms(s) * 1.05)

    def test_compressor_tames_peaks(self):
        s, _ = _sig()
        loud = array("d", (v * 4.0 for v in s))
        out = compressor(loud, 22050)
        self.assertLess(max(abs(v) for v in out),
                        max(abs(v) for v in loud))

    def test_soft_limiter_ceiling(self):
        s, _ = _sig()
        loud = array("d", (v * 4.0 for v in s))
        out = soft_limiter(loud, 0.95)
        self.assertLessEqual(max(abs(v) for v in out), 0.951)

    def test_reverb_adds_tail(self):
        s, sr = _sig(secs=1.0, hum=False, noise=0.0)
        out = reverb(s, sr, decay_s=0.5, wet=0.5)
        self.assertEqual(len(out), len(s))
        # wet signal differs from dry
        diff = _rms(array("d", (a - b for a, b in zip(s, out))))
        self.assertGreater(diff, 0.01)

    def test_reverb_deterministic(self):
        s, sr = _sig(secs=1.0, hum=False, noise=0.0)
        a = reverb(s, sr, decay_s=0.5, wet=0.5)
        b = reverb(s, sr, decay_s=0.5, wet=0.5)
        self.assertEqual(list(a), list(b))

    def test_echo(self):
        s, sr = _sig(secs=1.0, hum=False, noise=0.0)
        out = echo(s, sr, delay_ms=200.0, decay=0.5, repeats=2)
        self.assertEqual(len(out), len(s))

    def test_eq_3band(self):
        s, sr = _sig(noise=0.0)
        out = eq_3band(s, sr, low_db=-20.0, high_db=6.0)
        self.assertLess(_band(out, sr, 60), _band(s, sr, 60) * 0.5)

    def test_eq_noop_returns_input(self):
        s, sr = _sig()
        out = eq_3band(s, sr)
        self.assertEqual(list(out), list(s))

    def test_pitch_shift_preserves_length(self):
        s, sr = _sig(secs=1.0, hum=False, noise=0.0)
        out = pitch_shift(s, sr, 4.0)
        self.assertEqual(len(out), len(s))
        # +4 semitones: 200 Hz -> ~252 Hz
        self.assertGreater(_band(out, sr, 252), _band(out, sr, 200))

    def test_pitch_shift_zero_is_passthrough(self):
        s, sr = _sig()
        self.assertEqual(list(pitch_shift(s, sr, 0.0)), list(s))

    def test_mix_under(self):
        bed = array("d", [0.5]) * 100
        top = array("d", [0.5]) * 100
        out = mix_under(bed, top, gain=1.0)
        self.assertAlmostEqual(out[0], 1.0, places=6)


class TestIO(unittest.TestCase):
    def test_roundtrip(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "t.wav")
        s, sr = _sig(secs=1.0, hum=False, noise=0.0)
        self.assertTrue(write_mono_wav(p, s, sr))
        back, bsr = read_mono_wav(p)
        self.assertEqual(bsr, sr)
        self.assertEqual(len(back), len(s))
        self.assertLess(_rms(array("d", (a - b
                                         for a, b in zip(s, back)))),
                        0.01)

    def test_read_missing_returns_empty(self):
        back, sr = read_mono_wav("/nonexistent/x.wav")
        self.assertEqual(len(back), 0)
        self.assertEqual(sr, 0)


class TestEffectChain(unittest.TestCase):
    def test_parse_and_run(self):
        chain, unknown = EffectChain.parse_lenient(
            "dehum, normalize target=0.9, reverb wet=0.2")
        self.assertEqual(unknown, [])
        self.assertIn("dehum", chain.describe())
        s, sr = _sig()
        out = chain.run(s, sr)
        self.assertEqual(len(out), len(s))
        self.assertEqual(chain.skipped, [])

    def test_unknown_effect_reported(self):
        chain, unknown = EffectChain.parse_lenient("normalize, warpdrive")
        self.assertEqual(unknown, ["warpdrive"])
        self.assertEqual(len(chain.steps), 1)

    def test_unknown_name_raises_at_construction(self):
        with self.assertRaises(ValueError):
            EffectChain([("warpdrive", {})])

    def test_empty_chain_describes(self):
        self.assertIn("empty", EffectChain([]).describe())

    def test_all_effects_registered(self):
        for name in ("denoise", "dehum", "normalize", "compress", "limit",
                     "trim", "fade", "reverb", "echo", "eq", "pitch"):
            self.assertIn(name, EFFECTS)


class TestEnhance(unittest.TestCase):
    def test_voice_profile(self):
        # 1s silence lead + tone + hum + noise (realistic voice note)
        sr = 22050
        n = int(sr * 4)
        rng = random.Random(3)
        s = array("d", [0.0]) * n
        for i in range(n):
            t = i / sr
            if t > 1.0:
                s[i] = (0.5 * math.sin(2 * math.pi * 200 * t)
                        + 0.15 * math.sin(2 * math.pi * 60 * t)
                        + rng.gauss(0, 0.1))
        res = enhance(s, sr, "voice")
        self.assertTrue(res["ok"])
        out = res["samples"]
        self.assertLess(_band(out, sr, 60), _band(s, sr, 60) * 0.1)
        self.assertIn("denoise", res["chain"])

    def test_music_profile_lighter(self):
        s, sr = _sig()
        res = enhance(s, sr, "music")
        self.assertTrue(res["ok"])
        self.assertIn("denoise", res["chain"])

    def test_unknown_profile_falls_back_to_voice(self):
        s, sr = _sig()
        res = enhance(s, sr, "nonsense")
        self.assertTrue(res["ok"])


if __name__ == "__main__":
    unittest.main()
