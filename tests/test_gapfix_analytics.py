"""R3-59: ``nm analytics`` — the analytical backend gets a command surface.

Failing-first: on the unmodified tree this file does not even collect
(no ``nomorals.cmdline.commands.analytics`` module, no ``analytics``
subparser). After the fix, ``nm analytics`` runs read-only analytical
SQL over the live store through ``open_analytics`` — DuckDB when
installed, the SQLite passthrough otherwise — with a backend-aware
table listing, describe/sample/histogram/profile helpers, and honest
refusals for non-SELECT SQL, unknown tables, and a missing database.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.cmdline.commands.analytics import _cmd_analytics
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import CLI_ALIASES, _parser
from nomorals.compat import load_optional
from nomorals.storage.db import Database

DUCKDB_AVAILABLE = load_optional("duckdb") is not None


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    db = Database(":memory:")
    db.migrate()
    seed_memories(db, 10)
    return build_context(db=db, with_tools=True,
                         with_router=False, with_memory=False)


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


def _run(fn, argv, ctx):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = fn(_args(argv), ctx)
    return rc, out.getvalue() + err.getvalue()


class _FakeDuckDB:
    """Recording stand-in for DuckDBAnalytics (duckdb not installed here)."""

    name = "duckdb"

    def __init__(self):
        self.loaded: list[str] = []
        self.stats = {"tables_loaded": 0, "rows_loaded": 0, "queries": 0}

    def load_table(self, db, table, *, where="", params=()):
        self.loaded.append(table)
        return 10

    def query(self, sql, params=()):
        self.stats["queries"] += 1
        return [{"kind": "fact", "n": 4},
                {"kind": "episode", "n": 3},
                {"kind": "preference", "n": 3}]

    def table_names(self):
        return list(self.loaded)

    def describe(self, table):
        return {"table": table, "rows": 10,
                "columns": [{"name": "kind", "type": "VARCHAR",
                             "null": True, "key": ""}]}

    def sample(self, table, n=100, *, seed=42):
        return [{"kind": "fact"}]

    def histogram(self, table, column, *, bins=20, where="", params=()):
        return [{"low": 0.1, "high": 0.55, "count": 5, "raw": None}]

    def profile(self, sql, params=()):
        return [{"explain": "fake plan"}]

    def stats_snapshot(self):
        return {**self.stats, "tables": self.table_names()}

    def format_stats(self, theme=None):
        return "analytics (duckdb)\n  tables: memories"


class AnalyticsParseTests(unittest.TestCase):
    def test_command_registered(self):
        self.assertEqual(_canonical_command("analytics"), "analytics")
        self.assertIn("analytics", CLI_ALIASES)

    def test_query_parse(self):
        args = _args(["analytics", "query", "SELECT 1"])
        self.assertEqual(args.command, "analytics")
        self.assertEqual(args.task[0], "query")


class AnalyticsCliTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_no_verb_usage(self):
        rc, _ = _run(_cmd_analytics, ["analytics"], self.ctx)
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        rc, _ = _run(_cmd_analytics, ["analytics", "frobnicate"], self.ctx)
        self.assertEqual(rc, 2)

    def test_status_names_the_live_backend(self):
        rc, out = _run(_cmd_analytics, ["analytics", "status"], self.ctx)
        self.assertEqual(rc, 0)
        # DuckDB is not installed on this box: the passthrough is the
        # honest answer, and the status must say so.
        self.assertIn("sqlite", out)

    def test_status_json(self):
        rc, out = _run(_cmd_analytics,
                       ["analytics", "status", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["backend"], "sqlite")

    def test_tables_lists_live_tables_with_counts(self):
        rc, out = _run(_cmd_analytics, ["analytics", "tables"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("memories", out)
        self.assertIn("10", out)

    def test_query_group_by_over_live_data(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query",
             "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"],
            self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("fact", out)
        self.assertIn("4", out)
        self.assertIn("episode", out)
        self.assertIn("3", out)

    def test_query_empty_result_is_honest(self):
        self.ctx.db.execute(
            "CREATE TABLE probe_empty (id TEXT, v INTEGER)")
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query", "SELECT * FROM probe_empty"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("0 rows", out)

    def test_query_json(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query",
             "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind",
             "--json"],
            self.ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["rows"], 3)
        by_kind = {r["kind"]: r["n"] for r in payload["data"]}
        self.assertEqual(by_kind, {"fact": 4, "episode": 3, "preference": 3})

    def test_query_non_select_refused(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query", "DELETE FROM memories"], self.ctx)
        self.assertEqual(rc, 1)
        self.assertIn("read-only", out.lower())

    def test_query_unknown_table_refused(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query", "SELECT * FROM no_such_table"], self.ctx)
        self.assertEqual(rc, 1)
        self.assertIn("no_such_table", out)

    def test_describe_table(self):
        rc, out = _run(
            _cmd_analytics, ["analytics", "describe", "memories"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("memories", out)
        self.assertIn("kind", out)
        self.assertIn("10", out)

    def test_sample_table(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "sample", "memories", "--rows", "2"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("sample of memories (2 rows):", out)
        # The sample is ORDER BY RANDOM(): which kinds it draws is not
        # deterministic — pin the count and that every row is a seeded
        # memory, never a particular kind.
        data = [line for line in out.splitlines() if "id=mem-" in line]
        self.assertEqual(len(data), 2)

    def test_histogram(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "histogram", "memories", "importance"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("importance", out)

    def test_profile(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "profile", "SELECT * FROM memories"], self.ctx)
        self.assertEqual(rc, 0)

    def test_no_database_refused(self):
        rc, out = _run(_cmd_analytics, ["analytics", "status"],
                       SimpleNamespace(db=None))
        self.assertEqual(rc, 1)
        self.assertIn("no database", out.lower())

    def test_tables_counts_seam_tables_from_the_seam(self):
        # The listing is backend-aware: a seam-served table is counted
        # from the seam rows, never from the SQLite fossil.
        import nomorals.cmdline.commands.analytics as mod

        real = mod.analytics_source_rows

        def _fake(db, table):
            if table == "memories":
                return [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]
            return None

        mod.analytics_source_rows = _fake
        try:
            rc, out = _run(_cmd_analytics, ["analytics", "tables"], self.ctx)
        finally:
            mod.analytics_source_rows = real
        self.assertEqual(rc, 0)
        for line in out.splitlines():
            if line.strip().startswith("memories"):
                self.assertIn("3", line)
                break
        else:
            self.fail("memories row missing from tables output")

    def test_duckdb_query_loads_referenced_tables_first(self):
        # Proves the load-then-query wiring without duckdb installed:
        # the fake DuckDB engine records which tables were synced.
        import nomorals.cmdline.commands.analytics as mod

        fake = _FakeDuckDB()
        real = mod.open_analytics
        mod.open_analytics = lambda db, prefer="auto": fake
        try:
            rc, out = _run(
                _cmd_analytics,
                ["analytics", "query",
                 "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"],
                self.ctx)
        finally:
            mod.open_analytics = real
        self.assertEqual(rc, 0)
        self.assertEqual(fake.loaded, ["memories"])
        self.assertIn("fact", out)
        self.assertIn("4", out)

    def test_duckdb_describe_loads_the_table_first(self):
        import nomorals.cmdline.commands.analytics as mod

        fake = _FakeDuckDB()
        real = mod.open_analytics
        mod.open_analytics = lambda db, prefer="auto": fake
        try:
            rc, out = _run(
                _cmd_analytics, ["analytics", "describe", "memories"],
                self.ctx)
        finally:
            mod.open_analytics = real
        self.assertEqual(rc, 0)
        self.assertEqual(fake.loaded, ["memories"])
        self.assertIn("memories", out)


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class AnalyticsDuckDBTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_query_runs_on_the_real_duckdb_engine(self):
        rc, out = _run(
            _cmd_analytics,
            ["analytics", "query",
             "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"],
            self.ctx)
        self.assertEqual(rc, 0)
        payload_rows = [line for line in out.splitlines()
                        if "fact" in line]
        self.assertTrue(payload_rows)

    def test_status_names_duckdb(self):
        rc, out = _run(_cmd_analytics, ["analytics", "status"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("duckdb", out)


if __name__ == "__main__":
    unittest.main()
