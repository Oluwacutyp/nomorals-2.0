"""Pass #21 follow-up — the story decides its shape (owner, 2026-10-20).

Owner's ruling: "No hard coded amount of chapter for books, it depends
on what is written. As for novels the story decides, also note a never
ending story class exists too."

Pins:
* a novel ends when its own arcs are served — never at the
  premise-complexity estimate (``decide_shape`` is advisory only);
* novel arcs actually progress as chapters are written;
* writing past a novel's end raises StoryCompleteError honestly and
  ``status`` still answers;
* a serial (the never-ending class) is never given a total_planned cap
  and ``write_novel`` never declares it done;
* an explicit owner chapter count is still honored exactly (a decision
  by the owner is not rewritten either);
* ``planned_chapters`` metadata records the estimate honestly, with
  its source.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.fiction import FictionWriter, StoryCompleteError
from nomorals.books.serial import SerialManager


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


# mystery plan_arcs: The Crime 3 + The Investigation 5 + The Resolution 2
# + Denouement 1 = 11 chapters of story structure.
MYSTERY_ARC_TOTAL = 11
# premise below scores complexity 0 -> decide_shape estimate is 10 (!= 11)
PREMISE = "a detective hunts a killer"


class TestNovelEndsByStoryNotEstimate(unittest.TestCase):
    def test_write_novel_runs_to_arc_completion_past_estimate(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_novel(PREMISE, genre="mystery")
        slug = start["slug"]
        state = mgr.fiction.load(slug)
        self.assertEqual(state.total_planned, 10)      # the estimate
        self.assertEqual(state.shape_source, "estimate")
        out = mgr.write_novel(slug)
        self.assertTrue(out["done"])
        self.assertEqual(out["chapters_written"], MYSTERY_ARC_TOTAL)
        self.assertIn("arcs", out["stopped_because"])
        state = mgr.fiction.load(slug)
        self.assertTrue(all(a["status"] == "closed" for a in state.arcs))

    def test_arcs_progress_as_chapters_are_written(self):
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        result = w.start(PREMISE, genre="mystery", mode="novel")
        slug = result["slug"]
        arcs = []
        for _ in range(4):  # ch.1 from start + 3 more -> ch.4 opens arc 2
            ch = w.write_next(slug)
            arcs.append(ch["arc"])
        self.assertEqual(arcs[-1], "The Investigation")

    def test_write_past_the_end_is_an_honest_error(self):
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        result = w.start(PREMISE, genre="mystery", mode="novel")
        slug = result["slug"]
        for _ in range(MYSTERY_ARC_TOTAL - 1):
            w.write_next(slug)
        with self.assertRaises(StoryCompleteError):
            w.write_next(slug)
        # status still answers for a completed novel
        st = w.status(slug)
        self.assertTrue(st["complete"])
        self.assertEqual(st["chapters_written"], MYSTERY_ARC_TOTAL)


class TestSerialNeverCapped(unittest.TestCase):
    def test_serial_gets_no_total_planned(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_serial(PREMISE, genre="mystery")
        state = mgr.fiction.load(start["slug"])
        self.assertEqual(state.mode, "serial")
        self.assertEqual(state.total_planned, 0)

    def test_serial_keeps_writing_past_the_old_estimate(self):
        ctx, _ = _ctx()
        w = FictionWriter(ctx)
        result = w.start(PREMISE, genre="mystery", mode="serial")
        slug = result["slug"]
        # old code stored 6 as total_planned for this premise; a serial
        # must sail past any such number without error or ending
        for _ in range(9):  # ch.1 + 9 = 10 chapters
            w.write_next(slug)
        st = w.status(slug)
        self.assertEqual(st["chapters_written"], 10)
        self.assertFalse(st["complete"])

    def test_write_novel_never_declares_a_serial_done(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_serial(PREMISE, genre="mystery")
        out = mgr.write_novel(start["slug"], max_chapters=3)
        self.assertEqual(out["kind"], "serial")
        self.assertFalse(out["done"])
        self.assertIn("never-ending", out["stopped_because"])


class TestOwnerCountStillHonored(unittest.TestCase):
    def test_explicit_chapters_stop_write_novel(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_novel(PREMISE, genre="mystery", chapters=3)
        slug = start["slug"]
        state = mgr.fiction.load(slug)
        self.assertEqual(state.total_planned, 3)
        self.assertEqual(state.shape_source, "owner")
        out = mgr.write_novel(slug)
        self.assertTrue(out["done"])
        self.assertEqual(out["chapters_written"], 3)
        self.assertIn("owner-set", out["stopped_because"])


class TestPlannedMetadataHonesty(unittest.TestCase):
    def test_estimate_is_recorded_as_estimate(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_novel(PREMISE, genre="mystery")
        meta = mgr._load_meta(start["slug"])
        self.assertEqual(meta["planned_chapters"], 10)
        self.assertEqual(meta["shape_source"], "estimate")

    def test_owner_count_recorded_as_owner(self):
        ctx, _ = _ctx()
        mgr = SerialManager(ctx)
        start = mgr.start_novel(PREMISE, genre="mystery", chapters=7)
        meta = mgr._load_meta(start["slug"])
        self.assertEqual(meta["planned_chapters"], 7)
        self.assertEqual(meta["shape_source"], "owner")


if __name__ == "__main__":
    unittest.main()
