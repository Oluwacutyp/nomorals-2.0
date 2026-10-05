"""R14: ``nm schedule`` and ``nm db`` CLI mirrors — parsing, verbs, dispatch."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.cmdline.commands.db import _cmd_db
from nomorals.cmdline.commands.schedule import _cmd_schedule
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import CLI_ALIASES, _parser
from nomorals.storage.db import Database


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    return build_context(db=Database(":memory:"), with_tools=True,
                         with_router=False, with_memory=False)


def _run(fn, argv, ctx):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn(_args(argv), ctx)
    return rc, buf.getvalue()


class ScheduleParseTests(unittest.TestCase):
    def test_aliases_registered(self):
        self.assertEqual(_canonical_command("schedule"), "schedule")
        self.assertEqual(_canonical_command("sched"), "schedule")
        self.assertEqual(_canonical_command("db"), "db")
        self.assertEqual(_canonical_command("database"), "db")
        self.assertIn("sched", CLI_ALIASES["schedule"])
        self.assertIn("database", CLI_ALIASES["db"])

    def test_schedule_parse(self):
        args = _args(["schedule", "add", "goodnight", "22:00", "message",
                      "goodnight"])
        self.assertEqual(args.command, "schedule")
        self.assertEqual(args.task[:2], ["add", "goodnight"])

    def test_db_parse(self):
        args = _args(["db", "query", "SELECT 1"])
        self.assertEqual(args.command, "db")
        self.assertEqual(args.task[0], "query")


class ScheduleCliTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_no_verb_usage(self):
        rc, _ = _run(_cmd_schedule, ["schedule"], self.ctx)
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        rc, _ = _run(_cmd_schedule, ["schedule", "frobnicate"], self.ctx)
        self.assertEqual(rc, 2)

    def test_add_and_list(self):
        rc, out = _run(_cmd_schedule,
                       ["schedule", "add", "goodnight", "22:00", "message",
                        "goodnight 🌙"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("goodnight", out)
        rc, out = _run(_cmd_schedule, ["schedule", "list"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("goodnight", out)
        self.assertIn("daily", out)

    def test_list_json(self):
        _run(_cmd_schedule,
             ["schedule", "add", "j1", "every 30m", "message", "hi"], self.ctx)
        rc, out = _run(_cmd_schedule, ["schedule", "list", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        jobs = json.loads(out)
        self.assertTrue(any(j["name"] == "j1" for j in jobs))

    def test_add_bad_spec(self):
        rc, _ = _run(_cmd_schedule,
                     ["schedule", "add", "bad", "not-a-time", "message", "x"],
                     self.ctx)
        self.assertEqual(rc, 1)

    def test_add_missing_action(self):
        rc, _ = _run(_cmd_schedule,
                     ["schedule", "add", "bad", "22:00", "nonsense"], self.ctx)
        self.assertEqual(rc, 2)

    def test_enable_disable(self):
        _run(_cmd_schedule,
             ["schedule", "add", "j2", "every 30m", "message", "hi"], self.ctx)
        rc, out = _run(_cmd_schedule, ["schedule", "disable", "j2"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("disabled", out)
        rc, out = _run(_cmd_schedule, ["schedule", "enable", "j2"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("enabled", out)

    def test_run_message_job(self):
        _run(_cmd_schedule,
             ["schedule", "add", "j3", "every 30m", "message", "hello"],
             self.ctx)
        rc, out = _run(_cmd_schedule, ["schedule", "run", "j3"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("hello", out)

    def test_rm(self):
        _run(_cmd_schedule,
             ["schedule", "add", "j4", "every 30m", "message", "hi"], self.ctx)
        rc, _ = _run(_cmd_schedule, ["schedule", "rm", "j4"], self.ctx)
        self.assertEqual(rc, 0)
        rc, _ = _run(_cmd_schedule, ["schedule", "rm", "j4"], self.ctx)
        self.assertEqual(rc, 1)

    def test_rm_unknown(self):
        rc, _ = _run(_cmd_schedule, ["schedule", "rm", "no-such-job"],
                     self.ctx)
        self.assertEqual(rc, 1)


class DbCliTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_no_verb_usage(self):
        rc, _ = _run(_cmd_db, ["db"], self.ctx)
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        rc, _ = _run(_cmd_db, ["db", "frobnicate"], self.ctx)
        self.assertEqual(rc, 2)

    def test_tables(self):
        rc, out = _run(_cmd_db, ["db", "tables"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("tables (", out)

    def test_tables_lists_notifications(self):
        # the prose list is capped at 100 rows, so a late-alphabet table like
        # "notifications" is only guaranteed visible in the uncapped JSON
        rc, out = _run(_cmd_db, ["db", "tables", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        value = json.loads(out)
        self.assertIn("notifications",
                      [t["name"] for t in value["tables"]])

    def test_tables_json(self):
        rc, out = _run(_cmd_db, ["db", "tables", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        value = json.loads(out)
        self.assertIn("schedule_jobs",
                      [t["name"] for t in value["tables"]])

    def test_schema(self):
        rc, out = _run(_cmd_db, ["db", "schema", "schedule_jobs"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("schedule_jobs", out)
        self.assertIn("name", out)

    def test_schema_missing_arg(self):
        rc, _ = _run(_cmd_db, ["db", "schema"], self.ctx)
        self.assertEqual(rc, 2)

    def test_schema_unknown_table(self):
        rc, _ = _run(_cmd_db, ["db", "schema", "no_such_table_xyz"], self.ctx)
        self.assertEqual(rc, 1)

    def test_query(self):
        rc, out = _run(_cmd_db,
                       ["db", "query", "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='schedule_jobs'"],
                       self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("schedule_jobs", out)

    def test_query_rejects_writes(self):
        rc, _ = _run(_cmd_db, ["db", "query", "DELETE FROM schedule_jobs"],
                     self.ctx)
        self.assertEqual(rc, 1)

    def test_counts(self):
        rc, out = _run(_cmd_db, ["db", "counts"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("largest tables:", out)


if __name__ == "__main__":
    unittest.main()
