"""Pass #111 pins: the identities failure ledger records every silent
fallback in the task-identity store, with fallbacks themselves intact.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.agents.identities import IdentityStore, _default_db
from nomorals.agents.identities_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


def temp_db() -> str:
    return tempfile.mktemp(prefix="nm-ident111-", suffix=".db")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        clear_failures()

    def tearDown(self):
        clear_failures()

    def test_record_and_recent_newest_first(self):
        record_failure("a", ValueError("one"))
        record_failure("b", OSError("two"))
        recent = recent_failures(10)
        self.assertEqual([f.site for f in recent], ["b", "a"])
        self.assertEqual(recent[0].exc_type, "OSError")
        self.assertEqual(recent[0].message, "two")
        self.assertEqual(failure_count(), 2)

    def test_bound_keeps_newest_200(self):
        for i in range(250):
            record_failure(f"site-{i}", ValueError(str(i)))
        self.assertEqual(len(recent_failures(1000)), 200)
        self.assertEqual(failure_count(), 250)
        self.assertEqual(recent_failures(1)[0].site, "site-249")

    def test_unprintable_exception_never_raises(self):
        class BadStr(Exception):
            def __str__(self):
                raise RuntimeError("nope")

        record_failure("bad", BadStr())
        self.assertIn("unprintable", recent_failures(1)[0].message)

    def test_format_empty_and_render(self):
        self.assertEqual(format_failures(), "no identity failures recorded.")
        record_failure("create", RuntimeError("db gone"))
        out = format_failures()
        self.assertIn("identity failures (1 total", out)
        self.assertIn("create: RuntimeError: db gone", out)


class ConvertedSiteTests(unittest.TestCase):
    def setUp(self):
        clear_failures()

    def tearDown(self):
        clear_failures()

    def test_default_db_mkdir_failure_recorded_path_still_returned(self):
        orig_mkdir = Path.mkdir

        def boom(self, *a, **k):
            raise OSError("read-only home")

        Path.mkdir = boom
        try:
            path = _default_db()
        finally:
            Path.mkdir = orig_mkdir
        self.assertTrue(path.endswith("identities.db"))
        self.assertEqual([f.site for f in recent_failures()], ["_default_db"])

    def test_init_connect_failure_recorded_store_degraded(self):
        store = IdentityStore(db_path=tempfile.mkdtemp())
        self.assertIsNone(store._db)
        self.assertEqual([f.site for f in recent_failures()], ["__init__"])

    def test_closed_connection_failures_all_recorded_fallbacks_held(self):
        store = IdentityStore(db_path=temp_db())
        ident = store.create("Devon Research")
        self.assertIsNotNone(ident)
        clear_failures()
        store._db.close()
        self.assertIsNone(store.create("Second"))
        self.assertIsNone(store.get(ident.id))
        self.assertEqual(store.list(), [])
        self.assertFalse(store.remove(ident.id))
        self.assertEqual(
            [f.site for f in recent_failures(10)],
            ["remove", "list", "get", "create"])

    def test_healthy_paths_record_nothing(self):
        store = IdentityStore(db_path=temp_db())
        ident = store.create("Devon Research", handle="@devon_research")
        self.assertIsNotNone(ident)
        self.assertEqual(store.get(ident.id).name, "Devon Research")
        self.assertEqual(len(store.list()), 1)
        self.assertTrue(store.remove(ident.id))
        self.assertIsNone(store.create(""))  # early return, not a failure
        self.assertEqual(recent_failures(), [])
        self.assertEqual(failure_count(), 0)

    def test_subsystem_scan_clean(self):
        from nomorals.tools.error_scan import scan

        here = Path(__file__).resolve()
        target = here.parent.parent / "nomorals" / "agents" / "identities.py"
        report = scan([str(target)])
        self.assertEqual(report.findings, [])
        ledger = here.parent.parent / "nomorals" / "agents" / "identities_failures.py"
        self.assertEqual(scan([str(ledger)]).findings, [])


if __name__ == "__main__":
    unittest.main()
