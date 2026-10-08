"""Tests for the music upgrade: 7 new styles, 808 slides, heavier bass.

Covers :mod:`nomorals.core.midi` (new drum patterns, new bass styles,
NoteEvent.slide_to) and :mod:`nomorals.media.music` (new StyleSpecs,
aliases, bass_boost, snare rolls, drop impact) plus
:mod:`nomorals.media.synth` (slide-tone renderers, both paths).
Every new style must produce real audio — never silence, never raises.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.core import midi as M
from nomorals.core.midi import (
    BASS_STYLES,
    DRUM_PATTERNS,
    CRASH,
    NoteEvent,
    SNARE,
    generate_bass_line,
    generate_drums,
)
from nomorals.media import synth
from nomorals.media.music import STYLES, MusicCreator, resolve_style

NEW_STYLES = ("uk-drill", "rap", "trap", "grime", "phonk", "jersey", "dnb")

_STYLE_EXPECT = {
    "uk-drill": ("uk_drill", "808_slide", 1.4),
    "rap": ("boombap", "roots", 1.0),
    "trap": ("trap", "sub_pulse", 1.35),
    "grime": ("grime", "sub_pulse", 1.3),
    "phonk": ("phonk", "sub_pulse", 1.4),
    "jersey": ("jersey", "riff", 1.0),
    "dnb": ("dnb", "reese", 1.3),
}


def _context():
    root = tempfile.mkdtemp(prefix="music_upgrade_")
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=root))
    ctx._root = root
    return ctx


class NewStyleSpecs(unittest.TestCase):
    def test_all_new_styles_registered(self):
        for name in NEW_STYLES:
            self.assertIn(name, STYLES, name)

    def test_drum_and_bass_keys_valid(self):
        for name, (drums, bass, _boost) in _STYLE_EXPECT.items():
            spec = STYLES[name]
            self.assertEqual(spec.drum_pattern, drums)
            self.assertEqual(spec.bass_pattern, bass)
            self.assertIn(drums, DRUM_PATTERNS, name)
            self.assertIn(bass, BASS_STYLES, name)

    def test_tempo_ranges(self):
        self.assertEqual(STYLES["uk-drill"].tempo, (138, 143))
        self.assertEqual(STYLES["dnb"].tempo, (172, 176))
        self.assertEqual(STYLES["rap"].tempo, (88, 95))

    def test_aliases(self):
        cases = {
            "uk drill": "uk-drill", "ukdrill": "uk-drill",
            "trap": "trap", "rap": "rap", "boom bap": "rap",
            "boombap": "rap", "grime": "grime", "eski": "grime",
            "phonk": "phonk", "jersey": "jersey", "jersey club": "jersey",
            "dnb": "dnb", "drum and bass": "dnb", "jungle": "dnb",
            "drill": "drill",  # old chicago drill still resolves
        }
        for alias, want in cases.items():
            self.assertEqual(resolve_style(alias).name, want, alias)


class NewDrumPatterns(unittest.TestCase):
    def test_uk_drill_sparse_and_rolling(self):
        d = generate_drums("uk_drill", bars=1, seed=1)
        self.assertTrue(d)
        kicks = [e.start for e in d if e.note == M.KICK]
        snares = [e.start for e in d if e.note == M.SNARE]
        # snare lands on the 3
        self.assertIn(2.0, snares)
        # kicks are sparse (not four-on-the-floor)
        self.assertLess(len(kicks), 8)
        # 32nd-note hat roll at the bar end
        hats = sorted(e.start for e in d if e.note == M.CLOSED_HAT
                      and e.start >= 3.6)
        self.assertGreaterEqual(len(hats), 4)
        gaps = [round(b - a, 4) for a, b in zip(hats, hats[1:])]
        self.assertTrue(any(g <= 0.0626 for g in gaps), gaps)

    def test_dnb_break_density(self):
        d = generate_drums("dnb", bars=1, seed=1)
        snares = [e for e in d if e.note == M.SNARE]
        # amen-style: main hits plus ghost snares
        self.assertGreaterEqual(len(snares), 4)

    def test_jersey_triplet_kicks(self):
        d = generate_drums("jersey", bars=1, seed=1)
        kicks = sorted(e.start for e in d if e.note == M.KICK)
        self.assertTrue(any(abs(k - 0.33) < 0.01 for k in kicks),
                        kicks)

    def test_phonk_cowbell_lead(self):
        d = generate_drums("phonk", bars=1, seed=1)
        bells = [e for e in d if e.note == M.COWBELL]
        self.assertGreaterEqual(len(bells), 4)

    def test_grime_rim_clicks(self):
        d = generate_drums("grime", bars=1, seed=1)
        self.assertTrue(any(e.note == M.RIM for e in d))


class NewBassStyles(unittest.TestCase):
    def _line(self, style):
        return generate_bass_line("A", "minor",
                                  ["i", "bVI", "bVII", "i"],
                                  style=style, seed=1)

    def test_808_slide_has_portamento(self):
        line = self._line("808_slide")
        slides = [e for e in line if e.slide_to is not None]
        self.assertTrue(slides, "808_slide must emit slide_to events")
        for e in slides:
            self.assertIsInstance(e.slide_to, int)

    def test_sub_pulse_is_pulsing(self):
        line = self._line("sub_pulse")
        self.assertGreaterEqual(len(line), 8)

    def test_reese_stacks_octave(self):
        line = self._line("reese")
        starts = {}
        for e in line:
            starts.setdefault(round(e.start, 3), []).append(e.note)
        stacks = [n for n in starts.values() if len(n) >= 2]
        self.assertTrue(stacks, "reese must double notes")

    def test_unknown_bass_style_falls_back(self):
        line = generate_bass_line("A", "minor", ["i", "V"],
                                  style="not_a_style", seed=1)
        self.assertTrue(line)


class SlideSynth(unittest.TestCase):
    def _slide_parts(self):
        bass = generate_bass_line("A", "minor", ["i", "bVI"],
                                  style="808_slide", seed=1)
        return {"bass": bass}

    def test_numpy_slide_render(self):
        wav = synth.render_wav(self._slide_parts(), tempo=140, seed=1)
        self.assertGreater(len(wav), 1000)

    def test_stdlib_slide_render(self):
        parts = self._slide_parts()
        mix = synth._mix_tracks_stdlib(parts, tempo=140, seed=1)
        self.assertTrue(any(abs(s) > 1e-6 for s in mix))

    def test_slide_renderers_direct(self):
        sig = synth._render_slide_tone(55.0, 82.5, 2205, 100,
                                       (1.0, 0.5), 0.006, 0.05, 0.85, 0.06)
        self.assertEqual(len(sig), 2205)
        self.assertTrue(any(abs(s) > 1e-9 for s in sig))

    def test_slide_never_raises(self):
        # degenerate slides: same note, out-of-range targets
        for slide_to in (None, 36, 36, 200, -10):
            ev = NoteEvent(36, 0.0, 1.0, velocity=90, channel=3,
                           slide_to=slide_to)
            wav = synth.render_wav({"bass": [ev]}, tempo=120, seed=1)
            self.assertGreater(len(wav), 100)
        # stdlib path too
        ev = NoteEvent(36, 0.0, 1.0, velocity=90, slide_to=55)
        mix = synth._mix_tracks_stdlib({"bass": [ev]}, tempo=120)
        self.assertTrue(mix)

    def test_sub_bass_gain_positive(self):
        self.assertGreater(synth._SUB_BASS_GAIN, 0)


class ArrangementUpgrades(unittest.TestCase):
    def _compose(self, style, seed=7):
        ctx = _context()
        try:
            creator = MusicCreator(ctx)
            song = creator.compose(
                "night drive", style=style, seed=seed,
                with_midi=False, with_audio=False, with_score=False,
                with_vocals=False, workdir=ctx._root)
            return creator, song, ctx
        except Exception:
            shutil.rmtree(ctx._root, ignore_errors=True)
            raise

    def _arrange(self, style, seed=7):
        creator, song, ctx = self._compose(style, seed)
        try:
            import random
            rng = random.Random(song.seed)
            parts = creator._arrange(song, STYLES[style], rng)
            return parts
        finally:
            shutil.rmtree(ctx._root, ignore_errors=True)

    def test_bass_boost_applied(self):
        boosted = self._arrange("uk-drill")["bass"]
        plain = self._arrange("rap")["bass"]
        self.assertTrue(boosted and plain)
        self.assertEqual(max(e.velocity for e in boosted), 127)
        self.assertLess(max(e.velocity for e in plain), 127)

    def test_prechorus_snare_roll(self):
        # pop has a pre-chorus: its last bar should carry the buildup
        # snare roll (dense 16th snares absent from halftime_pop)
        parts = self._arrange("pop")
        snare_offs = [e.start % 4 for e in parts["drums"]
                      if e.note == SNARE]
        dense = [o for o in snare_offs
                 if abs(o - 2.0) > 0.01]  # not the base-pattern hit
        self.assertTrue(dense, "expected snare-roll hits in pre-chorus")

    def test_drop_crash(self):
        parts = self._arrange("dnb")
        crashes = [e for e in parts["drums"] if e.note == CRASH]
        self.assertTrue(crashes, "drop sections need a downbeat crash")

    def test_808_slide_survives_arrange(self):
        parts = self._arrange("uk-drill")
        slides = [e for e in parts["bass"]
                  if getattr(e, "slide_to", None) is not None]
        self.assertTrue(slides)


class FullRenders(unittest.TestCase):
    def test_every_new_style_renders_audio(self):
        for style in NEW_STYLES:
            ctx = _context()
            try:
                creator = MusicCreator(ctx)
                song = creator.compose(
                    "neon rain", style=style, seed=11,
                    with_midi=True, with_audio=True, with_score=False,
                    with_vocals=False, workdir=ctx._root)
                self.assertTrue(song.audio_path, style)
                self.assertGreater(
                    os.path.getsize(song.audio_path), 1000, style)
                self.assertTrue(song.midi_path, style)
            finally:
                shutil.rmtree(ctx._root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
