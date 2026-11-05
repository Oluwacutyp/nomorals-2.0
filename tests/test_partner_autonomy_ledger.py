"""The autonomy side-ledger handle (idle clock + pattern/interests).

History: the per-message block in ``PartnerRuntime._process_inner``
constructed ``Database(workspace_dir)`` fresh on EVERY message —
handing SQLite a directory, which cannot open (OperationalError
"unable to open database file"). Every message journaled the same
failure into the partner failure ledger (24k+ cumulative rows on the
owner's box) and the activity/interest writes never landed at all.

These tests pin the fixed shape:

* one lazily-opened handle for the process lifetime, opened through
  ``autonomy.idle.workspace_db`` (the canonical directory→file seam);
* healthy path records activity/interests exactly as before;
* a failed open journals ONE entry per outage episode, retries only
  after the cooldown, and recovers quietly;
* no per-message ``Database`` construction remains on the path.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.partner.failures import (
    clear_failures,
    failure_count,
    recent_failures,
)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.autonomy.idle import workspace_db

#: Test homes honour NM_TEST_TMPDIR so a runner can keep them on a
#: disk-backed dir instead of a small tmpfs.
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OPEN_SITE = "runtime.PartnerRuntime.autonomy_ledger_open"
WRITE_SITE = "runtime.PartnerRuntime.autonomy_ledger_write"


def _runtime(workspace_dir: Any) -> PartnerRuntime:
    """A bare PartnerRuntime carrying only what the ledger path needs."""
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=workspace_dir))
    return rt


def _message(text: str = "gold trading charts and lagos music",
             incoming: bool = True) -> SimpleNamespace:
    return SimpleNamespace(incoming=incoming, text=text)


def _ledger_failures() -> list:
    return [f for f in recent_failures(limit=200)
            if "autonomy_ledger" in f.site]


class HealthyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ledger-", dir=_TEST_TMPDIR)
        self.ws = Path(self.tmp.name) / "ws"  # not pre-created
        self.rt = _runtime(self.ws)

    def tearDown(self) -> None:
        handle = getattr(self.rt, "_autonomy_db_handle", None)
        if handle is not None:
            handle.close()
        self.tmp.cleanup()

    def test_records_activity_and_interests(self) -> None:
        for _ in range(5):
            self.rt._note_autonomy_activity(_message())
        # The workspace dir was ensured at open and the canonical
        # file created inside it.
        self.assertTrue((self.ws / "nomorals.db").is_file())
        db = workspace_db(self.ws)
        row = db.execute(
            "SELECT last_activity_ts FROM autonomy_activity "
            "WHERE id = 1").fetchone()
        self.assertIsNotNone(row)
        self.assertGreater(row[0], 0)
        self.assertGreaterEqual(
            db.execute("SELECT COUNT(*) FROM routine_activity"
                       ).fetchone()[0], 1)
        self.assertGreaterEqual(
            db.execute("SELECT COUNT(*) FROM interests").fetchone()[0],
            1)
        db.close()
        # A healthy ledger journals nothing.
        self.assertEqual(_ledger_failures(), [])
        self.assertEqual(failure_count(), 0)

    def test_one_handle_for_the_process(self) -> None:
        first = self.rt._autonomy_db()
        self.assertIsNotNone(first)
        for _ in range(10):
            self.rt._note_autonomy_activity(_message())
        self.assertIs(self.rt._autonomy_db(), first)

    def test_outgoing_message_notes_activity_only(self) -> None:
        self.rt._note_autonomy_activity(_message(incoming=False))
        db = workspace_db(self.ws)
        row = db.execute(
            "SELECT last_activity_ts FROM autonomy_activity "
            "WHERE id = 1").fetchone()
        self.assertGreater(row[0], 0)
        # The pattern/interest writers never ran, so their tables
        # were never even created on this file.
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
        self.assertNotIn("routine_activity", tables)
        self.assertNotIn("interests", tables)
        db.close()


class OpenFailureEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ledger-", dir=_TEST_TMPDIR)
        # A regular FILE where the workspace dir should be: the open
        # can never succeed (the DB file would live *inside* it).
        self.blocker = Path(self.tmp.name) / "blocker"
        self.blocker.write_text("not a directory", encoding="utf-8")
        self.rt = _runtime(self.blocker)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_journals_once_across_many_messages(self) -> None:
        for _ in range(50):
            self.rt._note_autonomy_activity(_message())
        failures = _ledger_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].site, OPEN_SITE)
        self.assertEqual(failure_count(), 1)
        self.assertIsNone(self.rt._autonomy_db())
        self.assertGreater(self.rt._autonomy_db_retry_at,
                           time.monotonic())

    def test_cooldown_blocks_retry_then_recovers(self) -> None:
        self.rt._note_autonomy_activity(_message())
        self.assertEqual(len(_ledger_failures()), 1)
        # Unblock: point the settings at a healthy workspace. While
        # the cooldown runs, the new path is NOT retried yet.
        good = Path(self.tmp.name) / "good"
        self.rt.context.settings.workspace_dir = good
        self.rt._note_autonomy_activity(_message())
        self.assertIsNone(getattr(self.rt, "_autonomy_db_handle", None))
        self.assertEqual(len(_ledger_failures()), 1)
        # Deadline reached: the next note retries, succeeds, and the
        # episode closes without another failure entry.
        self.rt._autonomy_db_retry_at = 0.0
        self.rt._note_autonomy_activity(_message())
        self.assertIsNotNone(self.rt._autonomy_db_handle)
        self.assertFalse(self.rt._autonomy_db_episode)
        self.assertEqual(len(_ledger_failures()), 1)
        db = workspace_db(good)
        row = db.execute(
            "SELECT last_activity_ts FROM autonomy_activity "
            "WHERE id = 1").fetchone()
        self.assertGreater(row[0], 0)
        db.close()
        self.rt._autonomy_db_handle.close()


class WriteFailureEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ledger-", dir=_TEST_TMPDIR)
        self.ws = Path(self.tmp.name) / "ws"
        self.rt = _runtime(self.ws)

    def tearDown(self) -> None:
        handle = getattr(self.rt, "_autonomy_db_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
        self.tmp.cleanup()

    def test_write_failure_drops_handle_and_journals_once(self) -> None:
        self.assertIsNotNone(self.rt._autonomy_db())

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("disk gone")

        self.rt._autonomy_db_handle = SimpleNamespace(
            execute=_boom, close=lambda: None)
        for _ in range(10):
            self.rt._note_autonomy_activity(_message())
        failures = [f for f in _ledger_failures() if f.site == WRITE_SITE]
        self.assertEqual(len(failures), 1)
        # The broken handle was dropped; the cooldown keeps later
        # messages from even attempting the open.
        self.assertIsNone(self.rt._autonomy_db_handle)
        self.assertIsNone(self.rt._autonomy_db())


class NoPerMessageConstructionTests(unittest.TestCase):
    def test_runtime_source_has_no_directory_database(self) -> None:
        source = (Path(__file__).resolve().parents[1]
                  / "nomorals" / "agents" / "partner"
                  / "runtime.py").read_text(encoding="utf-8")
        # The defect shape itself is gone from the whole runtime.
        self.assertNotIn("Database(ws_dir)", source)
        # The per-message block routes through the shared handle.
        self.assertIn("self._note_autonomy_activity(message)", source)
        self.assertIn("self._autonomy_db()", source)


if __name__ == "__main__":
    unittest.main()
