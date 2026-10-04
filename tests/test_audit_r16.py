"""R16 audit tests.

- trial assist runs persist across restarts: in-flight rows are marked
  ``interrupted`` and reported once via the durable notifier; terminal
  rows are untouched; ``/trial status`` shows the recovered state;
- BookForge offline prose: beat-aware leads (how-to/warning/contrast/
  question beats read differently), rotating section shapes, and
  per-beat takeaways instead of one static closing block.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.storage.db import Database


def _ctx(db=None):
    return build_context(db=db or Database(":memory:"), with_tools=True,
                         with_router=False, with_memory=False)


def _seed_run(db, run_id, state, platform="github", age_s=0):
    db.migrate()  # the table comes from migration 71
    db.execute(
        "INSERT OR REPLACE INTO trial_assist_runs"
        " (run_id, platform, chat_key, started, state, note, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, platform, "telegram:123", time.time() - age_s,
         state, "", time.time() - age_s),
    )


class TrialRunPersistenceTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.trial.flow import TrialFlow
        TrialFlow._assist_recovery_done = True  # tests drive recovery explicitly

    def _flow(self, ctx):
        from nomorals.agents.trial.flow import TrialFlow
        return TrialFlow(ctx)

    def test_assist_persists_run_row(self):
        from nomorals.agents.trial.flow import TrialFlow
        ctx = _ctx()
        flow = self._flow(ctx)
        with mock.patch.object(TrialFlow, "_assist_run",
                               lambda *a, **k: None):
            with mock.patch.object(
                    TrialFlow, "_owner_identity",
                    return_value={"name": "N", "email": "e@x.com"}):
                with mock.patch.dict("os.environ",
                                      {"NM_VAULT_PASSPHRASE": "x"}):
                    out = flow.assist("github", chat_key="telegram:123")
        self.assertIn("started", out)
        rows = ctx.db.query("SELECT * FROM trial_assist_runs")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["platform"], "github")
        self.assertEqual(rows[0]["chat_key"], "telegram:123")
        self.assertEqual(rows[0]["state"], "starting")

    def test_recovery_marks_inflight_interrupted_and_reports(self):
        from nomorals.agents.trial.flow import TrialFlow
        db = Database(":memory:")
        _seed_run(db, "r-inflight", "running")
        _seed_run(db, "r-done", "done")
        _seed_run(db, "r-failed", "failed")
        ctx = _ctx(db)
        recovered = TrialFlow.recover_interrupted_runs(ctx)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["run_id"], "r-inflight")
        row = db.query_one(
            "SELECT state, note FROM trial_assist_runs WHERE run_id=?",
            ("r-inflight",))
        self.assertEqual(row["state"], "interrupted")
        self.assertIn("restart", row["note"])
        # terminal rows untouched
        for rid, state in (("r-done", "done"), ("r-failed", "failed")):
            self.assertEqual(
                db.query_one("SELECT state FROM trial_assist_runs"
                             " WHERE run_id=?", (rid,))["state"], state)
        # the report went through the durable notifier (persisted row)
        notes = db.query(
            "SELECT title FROM notifications WHERE kind='trial'")
        self.assertTrue(any("interrupted" in (n.get("title") or "")
                            for n in notes),
                        f"no trial interrupt notification persisted: {notes}")
        # second recovery is a no-op — nothing re-reported
        self.assertEqual(TrialFlow.recover_interrupted_runs(ctx), [])

    def test_status_shows_persisted_runs(self):
        from nomorals.agents.trial.flow import TrialFlow
        db = Database(":memory:")
        _seed_run(db, "r-old", "interrupted", platform="x")
        ctx = _ctx(db)
        flow = self._flow(ctx)
        status = flow.assist_status()
        self.assertIn("x", status)
        self.assertIn("interrupted", status)

    def test_prune_drops_old_terminal_runs(self):
        from nomorals.agents.trial.flow import TrialFlow
        db = Database(":memory:")
        _seed_run(db, "r-ancient", "done", age_s=8 * 24 * 3600)
        _seed_run(db, "r-fresh", "done")
        ctx = _ctx(db)
        self._flow(ctx)  # init prunes
        self.assertIsNone(db.query_one(
            "SELECT 1 FROM trial_assist_runs WHERE run_id='r-ancient'"))
        self.assertIsNotNone(db.query_one(
            "SELECT 1 FROM trial_assist_runs WHERE run_id='r-fresh'"))


class BookProseVarietyTests(unittest.TestCase):
    def _chapter(self, number=2, beats=None, slug="vq", notes=""):
        from nomorals.books.model import Book, Chapter
        from nomorals.books.write import template_chapter
        b = Book(topic="deep work", slug=slug, title="Deep Work")
        b.notes = notes
        ch = Chapter(number=number, title="Focus",
                     beats=beats or ["the first idea", "the second idea"])
        return template_chapter(b, ch, prev_tail="prev"), template_chapter

    def test_beat_kind_leads_differ(self):
        from nomorals.books.model import Book, Chapter
        from nomorals.books.write import template_chapter, _beat_kind
        # the classifier routes each beat shape to its own lead pool
        self.assertEqual(_beat_kind("how to focus for hours"), "howto")
        self.assertEqual(
            _beat_kind("common mistakes that destroy focus"), "warning")
        self.assertEqual(_beat_kind("deep work vs shallow work"), "contrast")
        self.assertEqual(_beat_kind("why does focus matter?"), "question")
        self.assertEqual(_beat_kind("the first idea"), "general")
        # and across seeds each kind-specific lead actually surfaces
        markers = ("step by honest step", "Consider yourself warned",
                   "comparison matters", "honest answer")
        found = set()
        for i in range(30):
            b = Book(topic="deep work", slug=f"kind{i}", title="Deep Work")
            ch = Chapter(number=2, title="Focus", beats=[
                "how to focus for hours",
                "common mistakes that destroy focus",
                "deep work vs shallow work",
                "why does focus matter?",
            ])
            text = template_chapter(b, ch, prev_tail="prev")
            for m in markers:
                if m in text:
                    found.add(m)
        self.assertEqual(set(markers), found)

    def test_takeaways_come_from_beats(self):
        text, _ = self._chapter(beats=["alpha principle", "beta method"])
        tail = text.split("## Key takeaways")[-1]
        self.assertIn("Alpha principle", tail)
        self.assertIn("Beta method", tail)
        # the old static closing block is gone
        self.assertNotIn("Attention beats volume", text)
        self.assertNotIn("hands off to the following chapter", text)

    def test_two_chapters_differ(self):
        t1, _ = self._chapter(number=2, beats=["the first idea"])
        t2, _ = self._chapter(number=3, beats=["the first idea"])
        self.assertNotEqual(t1, t2)

    def test_deterministic(self):
        t1, _ = self._chapter(number=2, beats=["the first idea"])
        t2, _ = self._chapter(number=2, beats=["the first idea"])
        self.assertEqual(t1, t2)

    def test_no_template_repeats_within_chapter(self):
        import re
        text, _ = self._chapter(beats=[f"principle number {i}"
                                       for i in range(5)])
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 25 and not s.strip().startswith("##")]
        # template sentences (long reusable ones) must not repeat; the
        # research-free chapter has no note sentences to collide
        seen: dict[str, int] = {}
        for s in sents:
            if "principle number" in s:
                continue  # beat echoes are expected
            seen[s] = seen.get(s, 0) + 1
        dupes = {s: n for s, n in seen.items() if n > 1}
        self.assertEqual(dupes, {})

    def test_notes_still_woven_in(self):
        text, _ = self._chapter(
            beats=["focus rituals"],
            notes="A long study of attention spans found that rituals "
                  "reduce the cost of starting focused work.")
        self.assertIn("rituals reduce the cost", text)


if __name__ == "__main__":
    unittest.main()
