"""Analytics export + /analytics chat verb (R3-59 named remainders).

``cmdline/commands/analytics.py`` served status/tables/query/describe/
sample/histogram/profile with no export, and no ``/analytics`` chat
command existed — while the engine already had the export half
(``DuckDBAnalytics.to_parquet`` / ``to_csv``). These tests pin the
wire: ``nm analytics export`` writes a real file the owner can open
(CSV on any backend, Parquet on DuckDB only), query exports inherit
the query verb's read-only caps, and ``/analytics`` answers in chat
over the same engine, sending the file back through ``file_send``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.cmdline.commands.analytics import _cmd_analytics
from nomorals.cmdline.parser import _parser
from nomorals.compat import load_optional
from nomorals.core.config import load_settings
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    PUBLIC_COMMANDS,
    caller_may_run,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.telegram_menu import validate_menu
from nomorals.storage.db import Database
from tests.test_gapfix_analytics import _ctx, _FakeDuckDB, seed_memories
from tests.test_gapfix_museum_command import _msg
from tests.test_gapfix_museum_send import _SendRecordingGateway

DUCKDB_AVAILABLE = load_optional("duckdb") is not None


def _args(argv):
    return _parser().parse_args(argv)


def _run(fn, argv, ctx):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = fn(_args(argv), ctx)
    return rc, out.getvalue() + err.getvalue()


def _ctx_home(tmp: Path):
    """A CLI context whose home is ``tmp`` (exports land under it)."""
    db = Database(":memory:")
    db.migrate()
    seed_memories(db, 10)
    settings = load_settings(overrides={"home": str(tmp)})
    return build_context(settings, db=db, with_tools=True,
                         with_router=False, with_memory=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class _ExportFakeDuckDB(_FakeDuckDB):
    """Fake DuckDB engine that also records export calls."""

    def __init__(self) -> None:
        super().__init__()
        self.exports: list = []

    def to_csv(self, table, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("kind,content\nfact,memory number 0\n",
                          encoding="utf-8")
        self.exports.append(("csv", table, str(target)))
        return target

    def to_parquet(self, table, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PAR1fake")
        self.exports.append(("parquet", table, str(target)))
        return target

    def export_query(self, sql, path, *, format="csv", limit=1000):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if format == "parquet":
            target.write_bytes(b"PAR1fake")
        else:
            target.write_text("kind,n\nfact,4\nepisode,3\n",
                              encoding="utf-8")
        self.exports.append((format, sql, str(target)))
        return 2


class AnalyticsExportParseTests(unittest.TestCase):
    def test_export_parses(self):
        args = _args(["analytics", "export", "memories",
                      "--format", "csv"])
        self.assertEqual(args.command, "analytics")
        self.assertEqual(args.task[0], "export")
        self.assertEqual(args.format, "csv")

    def test_export_query_parses(self):
        args = _args(["analytics", "export", "query", "SELECT 1",
                      "--format", "parquet", "--out", "x.parquet"])
        self.assertEqual(args.task[1], "query")
        self.assertEqual(args.format, "parquet")
        self.assertEqual(args.out, "x.parquet")


class AnalyticsExportCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-analytics-export-",
            dir=str(Path.home() / "workspace" / "analyticsexport-verify"))
        self.ctx = _ctx()

    def tearDown(self):
        self.tmp.cleanup()

    def _out(self, name: str) -> Path:
        return Path(self.tmp.name) / name

    def test_csv_export_table_real_sqlite(self):
        out = self._out("mem.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories",
                         "--out", str(out)], self.ctx)
        self.assertEqual(rc, 0, text)
        self.assertIn("exported 10 rows", text)
        rows = _read_csv(out)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["content"], "memory number 0")

    def test_csv_export_default_path(self):
        ctx = _ctx_home(Path(self.tmp.name))
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories"], ctx)
        self.assertEqual(rc, 0, text)
        landed = list((Path(self.tmp.name) / "exports").glob(
            "analytics-memories-*.csv"))
        self.assertEqual(len(landed), 1, text)
        self.assertEqual(len(_read_csv(landed[0])), 10)

    def test_query_export_csv_real_sqlite(self):
        out = self._out("kinds.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "query",
                         "SELECT kind, COUNT(*) AS n FROM memories "
                         "GROUP BY kind",
                         "--out", str(out)], self.ctx)
        self.assertEqual(rc, 0, text)
        rows = _read_csv(out)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["kind"] for r in rows},
                         {"fact", "episode", "preference"})

    def test_query_export_cap(self):
        out = self._out("capped.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "query",
                         "SELECT * FROM memories",
                         "--out", str(out), "--limit", "2"], self.ctx)
        self.assertEqual(rc, 0, text)
        self.assertIn("capped at 2 rows", text)
        self.assertEqual(len(_read_csv(out)), 2)

    def test_parquet_refuses_on_sqlite(self):
        out = self._out("mem.parquet")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories",
                         "--format", "parquet", "--out", str(out)],
                        self.ctx)
        self.assertEqual(rc, 1, text)
        self.assertIn("sqlite passthrough", text)
        self.assertIn("duckdb", text)
        self.assertFalse(out.exists())

    def test_query_export_non_select_refused(self):
        out = self._out("del.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "query",
                         "DELETE FROM memories",
                         "--out", str(out)], self.ctx)
        self.assertEqual(rc, 1, text)
        self.assertFalse(out.exists())

    def test_export_unknown_table_refused(self):
        out = self._out("no.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "nosuchtable",
                         "--out", str(out)], self.ctx)
        self.assertEqual(rc, 1, text)
        self.assertFalse(out.exists())

    def test_export_json(self):
        out = self._out("mem.json.csv")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories",
                         "--out", str(out), "--json"], self.ctx)
        self.assertEqual(rc, 0, text)
        payload = json.loads(text)
        self.assertEqual(payload["rows"], 10)
        self.assertEqual(payload["format"], "csv")
        self.assertEqual(payload["path"], str(out))

    def test_export_no_args_usage(self):
        rc, _ = _run(_cmd_analytics, ["analytics", "export"], self.ctx)
        self.assertEqual(rc, 2)

    def test_export_unknown_format(self):
        out = self._out("mem.xls")
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories",
                         "--format", "xls", "--out", str(out)], self.ctx)
        self.assertEqual(rc, 2, text)
        self.assertFalse(out.exists())

    def test_fake_duckdb_parquet_table(self):
        fake = _ExportFakeDuckDB()
        out = self._out("mem.parquet")
        with mock.patch(
                "nomorals.cmdline.commands.analytics.open_analytics",
                return_value=fake):
            rc, text = _run(_cmd_analytics,
                            ["analytics", "export", "memories",
                             "--format", "parquet",
                             "--out", str(out)], self.ctx)
        self.assertEqual(rc, 0, text)
        self.assertIn(("parquet", "memories", str(out)), fake.exports)
        self.assertTrue(out.exists())

    def test_fake_duckdb_parquet_query(self):
        fake = _ExportFakeDuckDB()
        out = self._out("q.parquet")
        with mock.patch(
                "nomorals.cmdline.commands.analytics.open_analytics",
                return_value=fake):
            rc, text = _run(_cmd_analytics,
                            ["analytics", "export", "query",
                             "SELECT * FROM memories",
                             "--format", "parquet",
                             "--out", str(out)], self.ctx)
        self.assertEqual(rc, 0, text)
        self.assertEqual(fake.exports[0][0], "parquet")
        self.assertTrue(out.exists())


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class AnalyticsExportDuckDBTests(unittest.TestCase):
    """Real DuckDB export — runs only where duckdb is installed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-analytics-export-ddb-",
            dir=str(Path.home() / "workspace" / "analyticsexport-verify"))
        self.ctx = _ctx()

    def tearDown(self):
        self.tmp.cleanup()

    def test_parquet_round_trip(self):
        from nomorals.storage.analytics import DuckDBAnalytics

        out = Path(self.tmp.name) / "mem.parquet"
        rc, text = _run(_cmd_analytics,
                        ["analytics", "export", "memories",
                         "--format", "parquet", "--out", str(out)],
                        self.ctx)
        self.assertEqual(rc, 0, text)
        self.assertIn("exported 10 rows", text)
        engine = DuckDBAnalytics()
        try:
            engine.load_table(self.ctx.db, "memories")
            self.assertEqual(engine.from_parquet(out, "memories_rt"), 10)
        finally:
            engine.close()


class AnalyticsRegistrationTests(unittest.TestCase):
    def test_registered_everywhere(self):
        self.assertIn("analytics", CONTROL_COMMANDS)
        self.assertIn("analytics", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["analytics"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("analytics", grouped)
        self.assertEqual(grouped.count("analytics"), 1)
        self.assertIn("/analytics export",
                      COMMAND_DETAILS["analytics"]["usage"])

    def test_owner_only(self):
        self.assertNotIn("analytics", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("analytics", False))
        self.assertTrue(caller_may_run("analytics", True))

    def test_parse_control(self):
        cmd = parse_control("/analytics export memories")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "analytics")
        self.assertEqual(cmd.tail, "export memories")

    def test_listed_and_in_menu(self):
        self.assertIn("/analytics", list_catalog())
        ok, problems = validate_menu()
        self.assertTrue(ok, problems)


class AnalyticsChatTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / "workspace" / "analyticsexport-verify"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-analytics-chat-", dir=str(scratch))
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_tools=True)
        self.context.db.migrate()
        seed_memories(self.context.db, 10)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.gateway = _SendRecordingGateway()
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True,
             chat_key: str = "telegram:999") -> str:
        return self.runtime.handle_control(
            text, chat_key, message=_msg(text, chat_key=chat_key,
                                         owner=owner))

    def test_status_names_backend(self):
        reply = self._run("/analytics")
        self.assertIn("analytics", reply.lower())
        self.assertIn("sqlite", reply.lower())

    def test_tables_lists_memories(self):
        reply = self._run("/analytics tables")
        self.assertIn("memories", reply)
        self.assertIn("10 rows", reply)

    def test_query_returns_rows(self):
        reply = self._run(
            "/analytics query SELECT kind, COUNT(*) AS n FROM memories "
            "GROUP BY kind")
        self.assertIn("fact", reply)
        self.assertIn("3 rows", reply)

    def test_query_non_select_refused(self):
        reply = self._run("/analytics query DELETE FROM memories")
        self.assertIn("failed", reply)
        self.assertEqual(self.context.db.row_count("memories"), 10)

    def test_export_table_sends_file(self):
        reply = self._run("/analytics export memories")
        self.assertIn("exported 10 rows", reply)
        self.assertEqual(len(self.gateway.files), 1)
        sent = self.gateway.files[0]
        path = Path(sent["path"])
        self.assertTrue(path.suffix == ".csv")
        self.assertTrue(path.is_file())
        self.assertEqual(len(_read_csv(path)), 10)
        self.assertIn("exported 10 rows", sent["caption"])

    def test_export_parquet_refuses_on_sqlite(self):
        reply = self._run("/analytics export memories parquet")
        self.assertIn("sqlite passthrough", reply)
        self.assertIn("duckdb", reply)
        self.assertEqual(len(self.gateway.files), 0)

    def test_export_query_sends_file(self):
        reply = self._run(
            "/analytics export query SELECT * FROM memories")
        self.assertIn("exported 10 rows", reply)
        self.assertEqual(len(self.gateway.files), 1)
        path = Path(self.gateway.files[0]["path"])
        self.assertEqual(len(_read_csv(path)), 10)

    def test_export_needs_live_chat(self):
        reply = self.runtime._control_analytics("export memories",
                                                message=None)
        self.assertIn("needs a live chat", reply)

    def test_unknown_verb_usage(self):
        reply = self._run("/analytics frobnicate")
        self.assertIn("usage:", reply)

    def test_non_owner_cannot_export(self):
        reply = self._run("/analytics export memories", owner=False)
        self.assertNotIn("exported", reply)
        self.assertEqual(len(self.gateway.files), 0)


if __name__ == "__main__":
    unittest.main()
