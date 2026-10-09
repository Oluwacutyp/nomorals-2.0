"""Tests for the artist's notebook: song drafts, freestyle, perform bridge."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.media.song_draft import (
    SongDraft, DraftSection, DraftLine,
    render_notebook, save_draft, load_draft, list_drafts,
    _extract_json,
)
from nomorals.media.freestyle import (
    FreestyleSession, bar_syllable_target, start_session,
)
from nomorals.media.draft_perform import draft_melody_events


def _sample_draft() -> SongDraft:
    return SongDraft(
        title="Test Song", topic="testing", style="pop",
        tempo=100, key="C",
        about="a song about tests",
        sections=[
            DraftSection(
                name="verse 1", bars=4, energy=0.4,
                purpose="sets the scene",
                lines=[
                    DraftLine(text="first line here", syllables=4,
                              contour=["hold", "up", "down", "hold"],
                              rhyme="A"),
                    DraftLine(text="second line there", syllables=5,
                              contour=["up", "up", "hold", "down", "hold"],
                              rhyme="A"),
                ]),
            DraftSection(
                name="chorus", bars=4, energy=0.9,
                purpose="the release",
                lines=[
                    DraftLine(text="chorus hits hard", syllables=4,
                              contour=["leap", "hold", "hold", "drop"],
                              rhyme="B"),
                ]),
        ],
        rhyme_scheme="verse AAAA, chorus BB",
        why_melody_fits="rises land on the emotional words",
        energy_arc="verse low, chorus peaks",
        performance_notes="sing it like you mean it",
    )


class DraftDataTests(unittest.TestCase):
    def test_roundtrip(self):
        d = _sample_draft()
        d2 = SongDraft.from_dict(d.to_dict())
        self.assertEqual(d2.title, "Test Song")
        self.assertEqual(len(d2.sections), 2)
        self.assertEqual(d2.sections[0].lines[0].text, "first line here")
        self.assertEqual(d2.sections[1].energy, 0.9)

    def test_notebook_renders(self):
        nb = render_notebook(_sample_draft())
        self.assertIn("Test Song", nb)
        self.assertIn("first line here", nb)
        self.assertIn("why the melody fits", nb)
        self.assertIn("energy arc", nb)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            d = _sample_draft()
            p = save_draft(d, workdir=td)
            self.assertTrue(os.path.isfile(p))
            d2 = load_draft(p)
            self.assertEqual(d2.title, "Test Song")
            self.assertEqual(d2.sections[0].lines[1].rhyme, "A")
            paths = list_drafts(workdir=td)
            self.assertIn(p, paths)

    def test_extract_json(self):
        raw = '```json\n{"a": 1, "b": [1, 2]}\n```'
        self.assertEqual(_extract_json(raw), {"a": 1, "b": [1, 2]})
        raw2 = 'some preamble {"x": "y"} trailing'
        self.assertEqual(_extract_json(raw2), {"x": "y"})
        with self.assertRaises(ValueError):
            _extract_json("no json here at all")


class FreestyleTests(unittest.TestCase):
    def test_syllable_target_scales_with_bpm(self):
        lo_slow, hi_slow = bar_syllable_target(70)
        lo_fast, hi_fast = bar_syllable_target(140)
        # slower tempo → longer bar → more syllables fit
        self.assertGreater(lo_slow, lo_fast)
        self.assertGreater(hi_slow, hi_fast)

    def test_syllable_target_sane(self):
        lo, hi = bar_syllable_target(92)
        self.assertGreater(lo, 3)
        self.assertLess(hi, 40)
        self.assertLess(lo, hi)

    def test_session_requires_context(self):
        sess = start_session(bpm=92, seed="lagos nights")
        self.assertEqual(sess.bpm, 92)
        with self.assertRaises(ValueError):
            sess.spit(4, context=None)

    def test_transcript(self):
        sess = start_session(seed="test")
        sess.bars = ["bar one", "bar two"]
        t = sess.transcript()
        self.assertIn("bar one", t)
        self.assertIn("92 BPM", t)


class DraftMelodyTests(unittest.TestCase):
    def test_melody_events_from_contours(self):
        events = draft_melody_events(_sample_draft())
        # verse: 4 + 5 contour moves; chorus: 4 moves = 13 events
        self.assertEqual(len(events), 13)
        for e in events:
            self.assertTrue(hasattr(e, "note"))
            self.assertTrue(hasattr(e, "start"))
            # singable band
            self.assertGreaterEqual(e.note, 40)
            self.assertLessEqual(e.note, 90)

    def test_energy_drives_register(self):
        low = SongDraft(title="t", sections=[
            DraftSection(name="v", bars=4, energy=0.1, lines=[
                DraftLine(text="x", contour=["hold"])])])
        high = SongDraft(title="t", sections=[
            DraftSection(name="c", bars=4, energy=1.0, lines=[
                DraftLine(text="x", contour=["hold"])])])
        lo_ev = draft_melody_events(low)
        hi_ev = draft_melody_events(high)
        self.assertLess(lo_ev[0].note, hi_ev[0].note)
        self.assertLess(lo_ev[0].velocity, hi_ev[0].velocity)

    def test_empty_draft_no_events(self):
        d = SongDraft(title="empty")
        self.assertEqual(draft_melody_events(d), [])


if __name__ == "__main__":
    unittest.main()
