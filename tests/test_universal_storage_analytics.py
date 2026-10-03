"""Tests for the universal-upgrade analytical storage backends.

DuckDB tests run against the real engine (installed in this environment);
SQLiteAnalytics tests always run as the zero-dependency fallback path.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nomorals.compat import load_optional
from nomorals.core.errors import StorageError
from nomorals.storage.analytics import (
    DuckDBAnalytics,
    SQLiteAnalytics,
    open_analytics,
    table_row_counts,
)
from nomorals.storage.db import Database

DUCKDB_AVAILABLE = load_optional("duckdb") is not None


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def seed_memories(db: Database, count: int = 10) -> None:
    kinds = ["fact", "episode", "preference"]
    with db.transaction():
        for i in range(count):
            db.insert(
                "memories",
                {
                    "id": f"mem-{i:03d}",
                    "kind": kinds[i % 3],
                    "content": f"memory number {i}",
                    "importance": 0.1 * (i + 1),
                    "created_at": 1_700_000_000.0 + i,
                    "updated_at": 1_700_000_000.0 + i,
                },
            )


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestDuckDBAnalytics(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        seed_memories(self.db, 12)
        self.engine = DuckDBAnalytics()

    def tearDown(self) -> None:
        self.engine.close()
        self.db.close()

    def test_requires_duckdb_message_when_missing(self) -> None:
        # Constructor raises a helpful error (not ImportError) when unavailable.
        import nomorals.storage.analytics as analytics_mod

        original = analytics_mod.load_optional
        analytics_mod.load_optional = lambda name: None  # type: ignore[assignment]
        try:
            with self.assertRaises(StorageError) as ctx:
                DuckDBAnalytics()
            self.assertIn("pip install duckdb", str(ctx.exception))
        finally:
            analytics_mod.load_optional = original

    def test_load_table_syncs_rows(self) -> None:
        loaded = self.engine.load_table(self.db, "memories")
        self.assertEqual(loaded, 12)
        self.assertIn("memories", self.engine.table_names())

    def test_load_table_missing_table_raises(self) -> None:
        with self.assertRaises(StorageError):
            self.engine.load_table(self.db, "no_such_table")

    def test_load_table_with_where_filter(self) -> None:
        loaded = self.engine.load_table(
            self.db, "memories", where="kind = ?", params=("fact",)
        )
        self.assertEqual(loaded, 4)  # 12 rows, kinds cycle every 3

    def test_analytical_group_by(self) -> None:
        self.engine.load_table(self.db, "memories")
        rows = self.engine.query(
            "SELECT kind, COUNT(*) AS n, AVG(importance) AS avg_imp "
            "FROM memories GROUP BY kind ORDER BY kind"
        )
        by_kind = {r["kind"]: r for r in rows}
        self.assertEqual(by_kind["fact"]["n"], 4)
        self.assertEqual(by_kind["episode"]["n"], 4)
        self.assertEqual(by_kind["preference"]["n"], 4)
        self.assertAlmostEqual(by_kind["fact"]["avg_imp"], 0.55, places=6)

    def test_query_params_binding(self) -> None:
        self.engine.load_table(self.db, "memories")
        rows = self.engine.query(
            "SELECT id FROM memories WHERE importance > ? ORDER BY id", (1.0,)
        )
        self.assertEqual(len(rows), 2)  # importance 1.1, 1.2
        self.assertEqual(rows[0]["id"], "mem-010")

    def test_window_function(self) -> None:
        self.engine.load_table(self.db, "memories")
        rows = self.engine.query(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY kind ORDER BY created_at) AS rn "
            "FROM memories ORDER BY id"
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["rn"], 1)
        self.assertEqual(rows[3]["rn"], 2)  # mem-003 is the 2nd 'fact'

    def test_backend_is_read_only(self) -> None:
        self.engine.load_table(self.db, "memories")
        for bad in ("DELETE FROM memories", "DROP TABLE memories", "INSERT INTO memories VALUES (1)"):
            with self.assertRaises(StorageError):
                self.engine.query(bad)

    def test_refresh_picks_up_new_rows(self) -> None:
        self.engine.load_table(self.db, "memories")
        self.db.insert(
            "memories",
            {
                "id": "mem-new",
                "kind": "fact",
                "content": "new",
                "created_at": 1_800_000_000.0,
                "updated_at": 1_800_000_000.0,
            },
        )
        refreshed = self.engine.refresh(self.db, "memories")
        self.assertEqual(refreshed, 13)

    def test_parquet_round_trip(self) -> None:
        import tempfile

        self.engine.load_table(self.db, "memories")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = self.engine.to_parquet("memories", Path(tmp) / "mem.parquet")
            self.assertTrue(parquet.is_file())
            self.assertGreater(parquet.stat().st_size, 0)
            count = self.engine.from_parquet(parquet, "memories_copy")
            self.assertEqual(count, 12)
            rows = self.engine.query("SELECT COUNT(*) AS n FROM memories_copy")
            self.assertEqual(rows[0]["n"], 12)

    def test_clear_drops_tables(self) -> None:
        self.engine.load_table(self.db, "memories")
        self.engine.clear()
        self.assertEqual(self.engine.table_names(), [])

    def test_stats_snapshot(self) -> None:
        self.engine.load_table(self.db, "memories")
        self.engine.query("SELECT 1")
        snap = self.engine.stats_snapshot()
        self.assertEqual(snap["rows_loaded"], 12)
        self.assertEqual(snap["queries"], 1)
        self.assertIn("memories", snap["tables"])

    def test_table_row_counts_helper(self) -> None:
        self.engine.load_table(self.db, "memories")
        counts = table_row_counts(self.engine)
        self.assertEqual(counts["memories"], 12)


class TestSQLiteAnalytics(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        seed_memories(self.db, 6)
        self.engine = SQLiteAnalytics(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_passthrough_query(self) -> None:
        rows = self.engine.query(
            "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind ORDER BY kind"
        )
        self.assertEqual([r["n"] for r in rows], [2, 2, 2])

    def test_load_table_returns_count_without_copying(self) -> None:
        self.assertEqual(self.engine.load_table(self.db, "memories"), 6)
        self.assertEqual(
            self.engine.load_table(self.db, "memories", where="kind = ?", params=("fact",)),
            2,
        )

    def test_load_table_missing_table_raises(self) -> None:
        with self.assertRaises(StorageError):
            self.engine.load_table(self.db, "no_such_table")

    def test_read_only(self) -> None:
        with self.assertRaises(StorageError):
            self.engine.query("DELETE FROM memories")

    def test_table_names_and_clear(self) -> None:
        self.assertIn("memories", self.engine.table_names())
        self.engine.clear()  # no-op, must not raise

    def test_table_row_counts_helper(self) -> None:
        counts = table_row_counts(self.engine)
        self.assertEqual(counts["memories"], 6)


class TestOpenAnalytics(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()

    def tearDown(self) -> None:
        self.db.close()

    def test_auto_prefers_duckdb_when_available(self) -> None:
        engine = open_analytics(self.db)
        try:
            if DUCKDB_AVAILABLE:
                self.assertIsInstance(engine, DuckDBAnalytics)
            else:
                self.assertIsInstance(engine, SQLiteAnalytics)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    def test_prefer_sqlite(self) -> None:
        engine = open_analytics(self.db, prefer="sqlite")
        self.assertIsInstance(engine, SQLiteAnalytics)

    def test_prefer_duckdb(self) -> None:
        if not DUCKDB_AVAILABLE:
            with self.assertRaises(StorageError):
                open_analytics(self.db, prefer="duckdb")
        else:
            engine = open_analytics(self.db, prefer="duckdb")
            self.assertIsInstance(engine, DuckDBAnalytics)
            engine.close()

    def test_unknown_preference_raises(self) -> None:
        with self.assertRaises(StorageError):
            open_analytics(self.db, prefer="postgres")


if __name__ == "__main__":
    unittest.main()
