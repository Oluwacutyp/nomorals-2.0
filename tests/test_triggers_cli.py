"""``nm trigger`` CLI: dispatch, verbs, aliases."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.trigger import _cmd_trigger
from nomorals.cmdline.dispatch import _canonical_command
from nomorals.cmdline.parser import CLI_ALIASES, _parser
from nomorals.storage.db import Database


def _args(argv):
    return _parser().parse_args(argv)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="trigger-cli-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=Database(":memory:"), extras={},
                           settings=settings)


def _run(argv, ctx):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_trigger(_args(argv), ctx)
    return rc, buf.getvalue()


class TriggerAliasTests(unittest.TestCase):
    def test_alias_registered(self):
        self.assertIn("trig", CLI_ALIASES["trigger"])

    def test_canonical(self):
        self.assertEqual(_canonical_command("trigger"), "trigger")
        self.assertEqual(_canonical_command("trig"), "trigger")

    def test_parse(self):
        args = _args(["trigger", "list"])
        self.assertEqual(args.command, "trigger")
        self.assertEqual(args.task, ["list"])

    def test_parse_add_flags(self):
        args = _args(["trigger", "add", "--name", "n", "--source", "price",
                      "--symbol", "BTC", "--op", "lt", "--value", "5",
                      "--action", "notify"])
        self.assertEqual(args.name, "n")
        self.assertEqual(args.value, "5")


class TriggerAddTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()

    def test_no_verb_usage(self):
        rc, _ = _run(["trigger"], self.ctx)
        self.assertEqual(rc, 2)

    def test_unknown_verb(self):
        rc, _ = _run(["trigger", "frobnicate"], self.ctx)
        self.assertEqual(rc, 2)

    def test_add_needs_flags(self):
        rc, _ = _run(["trigger", "add"], self.ctx)
        self.assertEqual(rc, 2)

    def test_add_schedule_sugar(self):
        rc, out = _run(["trigger", "add", "--name", "morn",
                        "--source", "schedule", "--cron", "0 9 * * *",
                        "--action", "notify", "--title", "gm"], self.ctx)
        self.assertEqual(rc, 0, out)
        self.assertIn("added trigger", out)

    def test_add_price_sugar(self):
        rc, out = _run(["trigger", "add", "--name", "btc",
                        "--source", "price", "--symbol", "BTC",
                        "--op", "lt", "--value", "60000",
                        "--action", "notify"], self.ctx)
        self.assertEqual(rc, 0, out)

    def test_add_condition_json(self):
        rc, out = _run(["trigger", "add", "--name", "m",
                        "--source", "message",
                        "--condition", '{"pattern": "hi+"}',
                        "--action", "notify"], self.ctx)
        self.assertEqual(rc, 0, out)

    def test_add_bad_cron_fails(self):
        rc, out = _run(["trigger", "add", "--name", "bad",
                        "--source", "schedule", "--cron", "nope",
                        "--action", "notify"], self.ctx)
        self.assertEqual(rc, 1)

    def test_add_bad_json_fails(self):
        with self.assertRaises(SystemExit) as ctx:
            _run(["trigger", "add", "--name", "x", "--source", "webhook",
                  "--condition", "{oops", "--action", "notify"], self.ctx)
        self.assertEqual(ctx.exception.code, 2)

    def test_add_sugar_conflict_fails(self):
        with self.assertRaises(SystemExit) as ctx:
            _run(["trigger", "add", "--name", "x", "--source", "price",
                  "--condition", '{"symbol": "ETH"}',
                  "--symbol", "BTC", "--op", "lt", "--value", "1",
                  "--action", "notify"], self.ctx)
        self.assertEqual(ctx.exception.code, 2)


class TriggerManageTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        rc, out = _run(["trigger", "add", "--name", "wh",
                        "--source", "webhook", "--action", "notify",
                        "--json"], self.ctx)
        self.assertEqual(rc, 0, out)
        self.tid = json.loads(out)["id"]

    def test_list(self):
        rc, out = _run(["trigger", "list"], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("wh", out)
        self.assertIn(self.tid, out)

    def test_list_json(self):
        rc, out = _run(["trigger", "list", "--json"], self.ctx)
        self.assertEqual(rc, 0)
        rows = json.loads(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], self.tid)

    def test_disable_enable(self):
        rc, out = _run(["trigger", "disable", self.tid], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("disabled", out)
        rc, out = _run(["trigger", "list"], self.ctx)
        self.assertIn("[off]", out)
        rc, out = _run(["trigger", "enable", self.tid], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("enabled", out)

    def test_enable_unknown(self):
        rc, _ = _run(["trigger", "enable", "trigger_nope"], self.ctx)
        self.assertEqual(rc, 1)

    def test_run_fires(self):
        rc, out = _run(["trigger", "run", self.tid], self.ctx)
        # notify via the real notifier with an in-memory db and no gateway:
        # persist-only delivery, must not raise
        self.assertEqual(rc, 0, out)
        self.assertIn("fired", out)

    def test_run_unknown(self):
        rc, _ = _run(["trigger", "run", "trigger_nope"], self.ctx)
        self.assertEqual(rc, 1)

    def test_history(self):
        _run(["trigger", "run", self.tid], self.ctx)
        rc, out = _run(["trigger", "history", self.tid], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("fired", out)
        rc, out = _run(["trigger", "history", "--json", "--limit", "5"],
                       self.ctx)
        rows = json.loads(out)
        self.assertTrue(any(r["outcome"] == "fired" for r in rows))

    def test_remove(self):
        rc, out = _run(["trigger", "remove", self.tid], self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("removed", out)
        rc, out = _run(["trigger", "list"], self.ctx)
        self.assertIn("no triggers", out)

    def test_remove_unknown(self):
        rc, _ = _run(["trigger", "remove", "trigger_nope"], self.ctx)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
