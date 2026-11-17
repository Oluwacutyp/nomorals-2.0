"""Soundscape organ (MyNoise, 52-websites #32): generative soundscapes.

Devon composes soundscapes herself — noise colours with measured
spectra, binaural beats and drones, and layered scenes (rain, forest,
ocean, fire, cafe, night) under one uniform 10-slider model — renders
them to real WAV files she can send in chat, and remembers named mixes
so one word brings them back. No recordings are faked: every stem is
synthesised in code (provenance "synthesised") or a recorded stem the
owner adds himself; an unknown generator or layer is refused honestly.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import unittest
import wave

from nomorals.audio.fingerprint import has_numpy
from nomorals.audio.soundscapes.engine import (
    BAND_FREQS,
    MAX_DURATION_S,
    SR,
    animate_sliders,
    apply_eq,
    apply_hearing,
    octave_band_energies,
    render_tone,
    synthesize_noise,
)
from nomorals.audio.soundscapes.generators import (
    GENERATORS,
    PRESETS,
    resolve_generator,
)
from nomorals.audio.soundscapes.render import render_soundscape
from nomorals.audio.soundscapes.stems import STEMS
from nomorals.audio.soundscapes.store import SoundscapeStore, open_store

NUMPY = has_numpy()
if NUMPY:
    import numpy as np


def _band_db(energies: dict[float, float], freq: float) -> float:
    val = energies.get(freq, 0.0)
    if val <= 0:
        return -120.0
    return 10.0 * math.log10(val)


@unittest.skipUnless(NUMPY, "numpy needed for spectral checks")
class SpectralTests(unittest.TestCase):
    def test_white_is_flat(self):
        samples = synthesize_noise("white", 2 * SR, SR, seed=7)
        energies = octave_band_energies(samples, SR)
        dbs = [_band_db(energies, f) for f in BAND_FREQS[1:9]]
        self.assertLess(max(dbs) - min(dbs), 3.0,
                        f"white must be flat across bands, got {dbs}")

    def test_pink_falls_3db_per_octave(self):
        samples = synthesize_noise("pink", 4 * SR, SR, seed=7)
        energies = octave_band_energies(samples, SR)
        slopes = []
        for i in range(1, 8):
            slopes.append(_band_db(energies, BAND_FREQS[i + 1])
                          - _band_db(energies, BAND_FREQS[i]))
        for slope in slopes:
            self.assertAlmostEqual(slope, -3.0, delta=1.5)

    def test_brown_falls_6db_per_octave(self):
        samples = synthesize_noise("brown", 4 * SR, SR, seed=7)
        energies = octave_band_energies(samples, SR)
        slopes = []
        for i in range(2, 8):
            slopes.append(_band_db(energies, BAND_FREQS[i + 1])
                          - _band_db(energies, BAND_FREQS[i]))
        for slope in slopes:
            self.assertAlmostEqual(slope, -6.0, delta=2.0)

    def test_blue_rises_3db_per_octave(self):
        samples = synthesize_noise("blue", 4 * SR, SR, seed=7)
        energies = octave_band_energies(samples, SR)
        slopes = []
        for i in range(1, 8):
            slopes.append(_band_db(energies, BAND_FREQS[i + 1])
                          - _band_db(energies, BAND_FREQS[i]))
        for slope in slopes:
            self.assertAlmostEqual(slope, 3.0, delta=1.5)

    def test_grey_follows_equal_loudness(self):
        # Grey is shaped to equal loudness: hearing is weakest at the
        # extremes, so grey carries MORE energy at the low and high
        # ends than in the mids — the inverse of the ISO 226 contour.
        samples = synthesize_noise("grey", 4 * SR, SR, seed=7)
        energies = octave_band_energies(samples, SR)
        mid = _band_db(energies, 1000.0)
        low = _band_db(energies, 63.0)
        high = _band_db(energies, 8000.0)
        self.assertGreater(low, mid + 3.0)
        self.assertGreater(high, mid + 1.0)

    def test_eq_boost_raises_its_band(self):
        base = synthesize_noise("white", 2 * SR, SR, seed=7)
        boosted = apply_eq(base, SR, {1000.0: 12.0})
        e_base = octave_band_energies(base, SR)
        e_boost = octave_band_energies(boosted, SR)
        self.assertAlmostEqual(
            _band_db(e_boost, 1000.0) - _band_db(e_base, 1000.0),
            12.0, delta=2.0)
        # Neighbour bands move far less than the boosted one.
        self.assertLess(
            abs(_band_db(e_boost, 250.0) - _band_db(e_base, 250.0)), 6.0)

    def test_hearing_profile_shifts_band_gains(self):
        base = synthesize_noise("white", 2 * SR, SR, seed=7)
        corrected = apply_hearing(base, SR, {1000.0: 6.0, 250.0: -6.0})
        e_base = octave_band_energies(base, SR)
        e_corr = octave_band_energies(corrected, SR)
        self.assertAlmostEqual(
            _band_db(e_corr, 1000.0) - _band_db(e_base, 1000.0),
            6.0, delta=2.0)
        self.assertAlmostEqual(
            _band_db(e_corr, 250.0) - _band_db(e_base, 250.0),
            -6.0, delta=2.0)


class SynthesisTests(unittest.TestCase):
    def test_noise_is_deterministic_per_seed(self):
        a = synthesize_noise("pink", SR, SR, seed=42)
        b = synthesize_noise("pink", SR, SR, seed=42)
        self.assertEqual(list(a), list(b))
        c = synthesize_noise("pink", SR, SR, seed=43)
        self.assertNotEqual(list(a), list(c))

    def test_noise_never_clips(self):
        for color in ("white", "pink", "brown", "grey", "blue"):
            samples = synthesize_noise(color, SR, SR, seed=3)
            self.assertLessEqual(max(abs(s) for s in samples), 1.0)
            self.assertGreater(max(abs(s) for s in samples), 0.1)

    def test_unknown_colour_refused(self):
        with self.assertRaises(ValueError):
            synthesize_noise("ultraviolet", SR, SR, seed=1)

    def test_animate_drifts_sliders_slowly(self):
        base = (0.0,) * 10
        first = animate_sliders(base, 0.0, seed=5)
        later = animate_sliders(base, 60.0, seed=5)
        self.assertEqual(first, base)
        self.assertNotEqual(first, later)
        for gain in later:
            self.assertLessEqual(abs(gain), 3.5)

    def test_tone_is_a_pure_sine_at_the_band(self):
        path = render_tone(1000.0, duration_s=1.0)
        try:
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), SR)
                frames = wf.readframes(wf.getnframes())
            ints = struct.unpack(f"<{len(frames) // 2}h", frames)
            peak = max(abs(v) for v in ints)
            self.assertGreater(peak, 20000)
            if NUMPY:
                spec = np.abs(np.fft.rfft(np.array(ints, dtype=float)))
                freqs = np.fft.rfftfreq(len(ints), 1.0 / SR)
                top = float(freqs[int(np.argmax(spec[1:])) + 1])
                self.assertAlmostEqual(top, 1000.0, delta=25.0)
        finally:
            os.unlink(path)


class StemTests(unittest.TestCase):
    def test_every_scene_stem_renders_sound(self):
        for name, fn in STEMS.items():
            samples = fn(SR, SR, seed=11)
            self.assertEqual(len(samples), SR, name)
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            self.assertGreater(rms, 0.005, name)
            self.assertLessEqual(max(abs(s) for s in samples), 1.0, name)

    def test_rain_is_broadband_and_thunder_is_low(self):
        rain = STEMS["rain"](2 * SR, SR, seed=11)
        thunder = STEMS["thunder"](2 * SR, SR, seed=11)
        if not NUMPY:
            self.skipTest("numpy needed for stem spectra")
        e_rain = octave_band_energies(rain, SR)
        e_thunder = octave_band_energies(thunder, SR)
        rain_high = _band_db(e_rain, 4000.0)
        rain_low = _band_db(e_rain, 125.0)
        self.assertGreater(rain_high, rain_low - 6.0)
        thunder_low = _band_db(e_thunder, 63.0)
        thunder_high = _band_db(e_thunder, 4000.0)
        self.assertGreater(thunder_low, thunder_high + 6.0)

    def test_birds_are_sparse_chirps_not_constant_tone(self):
        birds = STEMS["birds"](4 * SR, SR, seed=11)
        windows = [birds[i * SR // 4:(i + 1) * SR // 4] for i in range(16)]
        energies = [sum(s * s for s in w) for w in windows]
        self.assertGreater(max(energies), 0.0)
        self.assertLess(min(energies), max(energies) * 0.2)


class CatalogueTests(unittest.TestCase):
    def test_every_generator_resolves(self):
        for gid in ("white", "pink", "brown", "grey", "blue",
                    "binaural", "drone", "rain", "forest", "ocean",
                    "fire", "cafe", "night"):
            spec = resolve_generator(gid)
            self.assertEqual(spec.id, gid)
            self.assertTrue(spec.sliders)
            self.assertLessEqual(len(spec.sliders), 10)

    def test_unknown_generator_refused(self):
        with self.assertRaises(ValueError):
            resolve_generator("hurricane")

    def test_presets_are_real_generators_with_valid_sliders(self):
        self.assertTrue(PRESETS)
        for preset in PRESETS.values():
            spec = resolve_generator(preset["generator"])
            self.assertEqual(len(preset["sliders"]), len(spec.sliders))
            for gain in preset["sliders"]:
                self.assertGreaterEqual(gain, -30.0)
                self.assertLessEqual(gain, 12.0)

    def test_scene_stems_are_all_real(self):
        for spec in GENERATORS.values():
            if spec.kind == "scene":
                for stem in spec.sliders:
                    self.assertIn(stem, STEMS, f"{spec.id}:{stem}")


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_render_writes_a_real_wav_with_receipt(self):
        ans = render_soundscape(self.home, "pink", duration_s=1.0, seed=9)
        self.assertTrue(ans["ok"], ans)
        self.assertTrue(os.path.exists(ans["path"]))
        self.assertEqual(len(ans["render_hash"]), 64)
        with wave.open(ans["path"], "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getframerate(), SR)
            self.assertEqual(wf.getnframes(), SR)
        self.assertGreater(ans["rms"], 0.01)
        self.assertLessEqual(ans["peak"], 1.0)

    def test_render_is_deterministic_per_seed(self):
        a = render_soundscape(self.home, "rain", duration_s=1.0, seed=4)
        b = render_soundscape(self.home, "rain", duration_s=1.0, seed=4)
        self.assertEqual(a["render_hash"], b["render_hash"])
        c = render_soundscape(self.home, "rain", duration_s=1.0, seed=5)
        self.assertNotEqual(a["render_hash"], c["render_hash"])

    def test_fade_out_tail_goes_quiet(self):
        ans = render_soundscape(self.home, "brown", duration_s=2.0,
                                seed=4, fade_s=1.0)
        with wave.open(ans["path"], "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        ints = struct.unpack(f"<{len(frames) // 2}h", frames)
        head = ints[: SR // 2]
        tail = ints[-SR // 4:]
        head_rms = math.sqrt(sum(v * v for v in head) / len(head))
        tail_rms = math.sqrt(sum(v * v for v in tail) / len(tail))
        self.assertLess(tail_rms, head_rms * 0.3)

    def test_loop_crossfade_joins_head_to_tail(self):
        ans = render_soundscape(self.home, "drone", duration_s=2.0,
                                seed=4, loop=True)
        with wave.open(ans["path"], "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        ints = struct.unpack(f"<{len(frames) // 2}h", frames)
        head = ints[: SR // 10]
        tail = ints[-SR // 10:]
        head_rms = math.sqrt(sum(v * v for v in head) / len(head))
        tail_rms = math.sqrt(sum(v * v for v in tail) / len(tail))
        self.assertGreater(tail_rms, head_rms * 0.5)

    def test_duration_cap_refused_honestly(self):
        with self.assertRaises(ValueError):
            render_soundscape(self.home, "white",
                              duration_s=MAX_DURATION_S + 1.0)

    def test_unknown_generator_refused(self):
        ans = render_soundscape(self.home, "hurricane", duration_s=1.0)
        self.assertFalse(ans["ok"])
        self.assertIn("hurricane", ans["error"])

    def test_saved_mix_name_wins_over_preset_word(self):
        # "deep focus" is the owner's mix (pink), not the "focus"
        # preset (brown) — his own words win over a preset substring.
        store = open_store(self.home)
        store.save_mix("deep focus", "pink", (0.0,) * 10)
        ans = render_soundscape(self.home, "deep focus", duration_s=1.0,
                                seed=4)
        self.assertTrue(ans["ok"], ans)
        self.assertEqual(ans["generator"], "pink")

    def test_tune_names_the_adjustment_in_the_reply(self):
        ans = render_soundscape(self.home, "rain", duration_s=1.0, seed=4,
                                adjust={"thunder": 3.0})
        self.assertTrue(ans["ok"], ans)
        self.assertIn("thunder", ans["reply"])
        self.assertIn("+3.0 dB", ans["reply"])

    def test_animate_changes_the_mix_over_time(self):
        if not NUMPY:
            self.skipTest("numpy needed for animate check")
        flat = render_soundscape(self.home, "pink", duration_s=4.0,
                                 seed=4, animate=False)
        animated = render_soundscape(self.home, "pink", duration_s=4.0,
                                     seed=4, animate=True)
        self.assertNotEqual(flat["render_hash"], animated["render_hash"])

    def test_hearing_profile_is_applied_and_named(self):
        store = open_store(self.home)
        store.set_hearing({1000.0: 6.0})
        ans = render_soundscape(self.home, "white", duration_s=1.0, seed=4)
        self.assertTrue(ans["hearing_applied"])
        self.assertIn("1000", ans["reply"])

    def test_cli_parses_and_runs(self):
        from types import SimpleNamespace

        from nomorals.cmdline.commands.soundscape import _cmd_soundscape
        from nomorals.cmdline.parser import _parser

        context = SimpleNamespace(
            settings=SimpleNamespace(home_path=self.home))
        args = _parser().parse_args(["soundscape", "list"])
        self.assertEqual(_cmd_soundscape(args, context), 0)
        args = _parser().parse_args(["soundscape", "rain", "--duration", "1"])
        self.assertEqual(_cmd_soundscape(args, context), 0)
        args = _parser().parse_args(["soundscape", "hurricane"])
        self.assertEqual(_cmd_soundscape(args, context), 1)
        args = _parser().parse_args(["soundscape"])
        self.assertEqual(_cmd_soundscape(args, context), 2)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_mix_round_trip(self):
        store = open_store(self.home)
        store.save_mix("deep focus", "brown", (0.0,) * 10,
                       animate=True, duration_s=120.0)
        mix = store.load_mix("deep focus")
        self.assertEqual(mix["generator"], "brown")
        self.assertTrue(mix["animate"])
        self.assertEqual(mix["duration_s"], 120.0)
        self.assertIn("deep focus", store.list_mixes())
        store.delete_mix("deep focus")
        self.assertNotIn("deep focus", store.list_mixes())

    def test_mix_names_are_case_insensitive(self):
        store = open_store(self.home)
        store.save_mix("Rainy Night", "rain", (0.0,) * 7)
        self.assertEqual(store.load_mix("rainy night")["generator"], "rain")

    def test_unknown_mix_is_none(self):
        store = open_store(self.home)
        self.assertIsNone(store.load_mix("no such mix"))

    def test_hearing_round_trip(self):
        store = open_store(self.home)
        store.set_hearing({125.0: 3.0, 8000.0: -2.0})
        self.assertEqual(store.hearing(), {125.0: 3.0, 8000.0: -2.0})

    def test_last_render_round_trip(self):
        store = open_store(self.home)
        store.set_last({"generator": "rain", "sliders": [0.0] * 7})
        self.assertEqual(store.last()["generator"], "rain")

    def test_store_survives_reload(self):
        store = open_store(self.home)
        store.save_mix("sleep", "pink", (0.0,) * 10)
        again = SoundscapeStore(self.home)
        self.assertIn("sleep", again.list_mixes())


class RegistrationTests(unittest.TestCase):
    def test_registered_everywhere_public(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("soundscape", CONTROL_COMMANDS)
        self.assertIn("soundscape", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["soundscape"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("soundscape", grouped)
        self.assertEqual(grouped.count("soundscape"), 1)
        self.assertIn("/soundscape", COMMAND_DETAILS["soundscape"]["usage"])
        self.assertIn("soundscape", PUBLIC_COMMANDS)
        self.assertEqual(272, len(PUBLIC_COMMANDS))
        self.assertTrue(caller_may_run("soundscape", False))
        self.assertIn("/soundscape", list_catalog())
        cmd = parse_control("/soundscape rain")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "soundscape")

    def test_tool_registers(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtins()
        names = {s["name"] for s in registry.schemas()}
        self.assertIn("soundscape", names)


if __name__ == "__main__":
    unittest.main()
