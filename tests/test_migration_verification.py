"""Migration post-condition verification (burn-down: swallowed COMMITs).

The defect: ``Database._safe("COMMIT")`` swallows a failed COMMIT
(SQLITE_FULL on a full disk), so ``migrate()`` could report success on a
half-applied schema and the failure surfaced much later as misleading
"no such table" errors. The migration runner now proves its own
post-conditions: a commit that did not land raises, the failed step is
never recorded as applied, the objects each migration's SQL created are
spot-checked, and a doctored schema is named. Ordinary transactions
keep the soft path untouched.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from nomorals.core.errors import MigrationError
from nomorals.storage.db import Database
from nomorals.storage.migrations import MIGRATIONS, latest_version
from nomorals.storage.schema import Migration, MigrationRunner

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

_MIGS = (
    Migration(
        1,
        "widgets",
        sql="CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT);\n"
        "CREATE INDEX idx_widgets_name ON widgets(name);",
    ),
    Migration(
        2,
        "gadgets",
        sql="CREATE TABLE gadgets (id INTEGER PRIMARY KEY);\n"
        "CREATE VIEW gadget_view AS SELECT id FROM gadgets;",
    ),
)


def _sabotage_commit():
    """Patch ``Database._safe`` so COMMIT silently never happens.

    Everything else (BEGIN, ROLLBACK, savepoints) runs for real — this is
    exactly the production failure shape: the soft path swallowed the
    commit and the transaction is left dangling in SQLite.
    """
    real_safe = Database._safe

    def flaky(self, sql):  # type: ignore[no-untyped-def]
        if sql.strip().upper() == "COMMIT":
            return None
        return real_safe(self, sql)

    return unittest.mock.patch.object(Database, "_safe", flaky)


class _FileDbTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=_TEST_TMPDIR)
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "test.db"

    def make_db(self) -> Database:
        db = Database(self.path)
        self.addCleanup(db.close)
        return db


class TestHealthyMigrateVerifies(_FileDbTest):
    def test_custom_migrations_verify_green(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        applied = runner.apply_all(_MIGS)
        self.assertEqual(len(applied), 2)
        self.assertEqual(
            runner.verify_applied(_MIGS, applied, expected_version=2), []
        )
        self.assertEqual(runner.verify_objects(_MIGS), [])
        self.assertTrue(db.table_exists("widgets"))
        self.assertTrue(db.table_exists("gadgets"))

    def test_full_production_migrate_verifies_green(self) -> None:
        db = self.make_db()
        summary = db.migrate()
        self.assertEqual(summary.version, latest_version())
        runner = MigrationRunner(db)
        self.assertEqual(
            runner.verify_applied(
                MIGRATIONS, summary.applied, expected_version=summary.version
            ),
            [],
        )
        # A second migrate is a no-op run and must verify just as clean.
        again = db.migrate()
        self.assertEqual(again.applied, [])
        self.assertEqual(again.version, latest_version())


class TestSwallowedCommitFailsLoudly(_FileDbTest):
    def test_commit_failure_raises_and_step_is_not_claimed(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        with _sabotage_commit():
            with self.assertRaises(MigrationError) as ctx:
                runner.apply_all(_MIGS)
        self.assertIn("did not commit", str(ctx.exception))
        # The failed step was rolled back, not silently left dangling:
        # no open transaction, no bookkeeping row, no table.
        self.assertFalse(db.connection_in_transaction)
        self.assertEqual(runner.current_version(), 0)
        self.assertEqual(runner.applied(), {})
        self.assertFalse(db.table_exists("widgets"))

    def test_clean_migrate_after_commit_failure_completes(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        with _sabotage_commit():
            with self.assertRaises(MigrationError):
                runner.apply_all(_MIGS)
        applied = runner.apply_all(_MIGS)
        self.assertEqual(len(applied), 2)
        self.assertEqual(runner.current_version(), 2)
        self.assertEqual(
            runner.verify_applied(_MIGS, applied, expected_version=2), []
        )
        self.assertTrue(db.table_exists("widgets"))
        self.assertTrue(db.table_exists("gadget_view"))


class TestDoctoredSchemaIsNamed(_FileDbTest):
    def test_dropped_table_is_named_by_object_verification(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        runner.apply_all(_MIGS)
        db.execute("DROP TABLE gadgets")
        problems = runner.verify_objects(_MIGS)
        self.assertTrue(
            any("gadgets" in p and "missing" in p for p in problems),
            f"expected the dropped table to be named, got: {problems}",
        )

    def test_missing_bookkeeping_row_is_named(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        applied = runner.apply_all(_MIGS)
        db.execute("DELETE FROM schema_migrations WHERE version = 2")
        problems = runner.verify_applied(_MIGS, applied, expected_version=2)
        self.assertTrue(
            any("0002_gadgets" in p for p in problems),
            f"expected the unrecorded migration to be named, got: {problems}",
        )

    def test_wrong_recorded_version_is_named(self) -> None:
        db = self.make_db()
        runner = MigrationRunner(db)
        applied = runner.apply_all(_MIGS)
        problems = runner.verify_applied(_MIGS, applied, expected_version=7)
        self.assertTrue(any("expected 7" in p for p in problems))


class TestSoftPathUnchanged(_FileDbTest):
    def test_swallowed_commit_outside_migrate_stays_soft(self) -> None:
        db = self.make_db()
        with _sabotage_commit():
            # The ordinary transaction path must behave exactly as before:
            # no exception escapes the context manager even though the
            # COMMIT never landed.
            with db.transaction():
                db.execute("CREATE TABLE soft_t (x INTEGER)")
            self.assertFalse(db.in_transaction)
        # Clean up the dangling transaction the soft path left behind.
        db.execute("ROLLBACK")
        self.assertFalse(db.connection_in_transaction)
        self.assertFalse(db.table_exists("soft_t"))

    def test_connection_in_transaction_reflects_sqlite_state(self) -> None:
        db = self.make_db()
        self.assertFalse(db.connection_in_transaction)
        db.execute("BEGIN")
        try:
            self.assertTrue(db.connection_in_transaction)
        finally:
            db.execute("ROLLBACK")
        self.assertFalse(db.connection_in_transaction)


if __name__ == "__main__":
    unittest.main()
