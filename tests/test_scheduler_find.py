"""Scheduler._find: id/name/prefix resolution never attaches to the wrong job.

Contract (uniform with upgrade/mission/research resolvers):
exact id > exact name > unique id prefix > AmbiguousRef (never first-match).
"""
import unittest
from types import SimpleNamespace

from nomorals.agents.scheduler import Scheduler
from nomorals.core.errors import AmbiguousRef
from nomorals.storage import migrations
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db):
    return SimpleNamespace(db=db)


def _insert(db, job_id, name):
    db.execute(
        "INSERT INTO schedule_jobs (id, name, kind, spec, payload_kind, payload,"
        " enabled, next_run, created_at, updated_at)"
        " VALUES (?, ?, 'every', '60', 'message', '{}', 1, 0, 0, 0)",
        (job_id, name),
    )


class SchedulerFindTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.sched = Scheduler(_ctx(self.db))

    def _two_jobs(self):
        # ids share the prefix "job-abc"
        _insert(self.db, "job-abc111", "morning")
        _insert(self.db, "job-abc222", "evening")

    def test_exact_id_wins(self):
        self._two_jobs()
        row = self.sched._find("job-abc111")
        self.assertEqual(row["name"], "morning")

    def test_exact_name_wins(self):
        self._two_jobs()
        row = self.sched._find("evening")
        self.assertEqual(row["id"], "job-abc222")

    def test_unique_prefix_resolves(self):
        self._two_jobs()
        row = self.sched._find("job-abc1")
        self.assertEqual(row["id"], "job-abc111")

    def test_ambiguous_prefix_raises_not_first_match(self):
        self._two_jobs()
        with self.assertRaises(AmbiguousRef) as ctx:
            self.sched._find("job-abc")
        self.assertEqual(len(ctx.exception.candidates), 2)
        self.assertGreater(ctx.exception.min_prefix_len, len("job-abc"))

    def test_ambiguous_never_acts(self):
        # the dangerous old behavior: remove() would delete the first row.
        self._two_jobs()
        with self.assertRaises(AmbiguousRef):
            self.sched.remove("job-abc")
        remaining = self.db.query("SELECT id FROM schedule_jobs")
        self.assertEqual(len(remaining), 2)

    def test_no_match_returns_none(self):
        self._two_jobs()
        self.assertIsNone(self.sched._find("job-zzz"))

    def test_empty_ref_returns_none(self):
        self._two_jobs()
        self.assertIsNone(self.sched._find(""))
        self.assertIsNone(self.sched._find("   "))

    def test_like_wildcards_treated_literally(self):
        self._two_jobs()
        # "%" must not become a match-all
        self.assertIsNone(self.sched._find("job-%"))
        self.assertIsNone(self.sched._find("job_abc"))

    def test_exact_id_that_is_also_a_prefix_wins(self):
        _insert(self.db, "job-a", "short")
        _insert(self.db, "job-abc", "long")
        row = self.sched._find("job-a")
        self.assertEqual(row["name"], "short")


if __name__ == "__main__":
    unittest.main()
