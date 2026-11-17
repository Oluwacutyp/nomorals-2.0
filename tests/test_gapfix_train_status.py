"""Gapfix: `nm train status` — the DM promises a command that did not exist.

`CognitiveLoop._tick_train` (agents/cognition.py) tells the owner
"I'll report when it finishes — `nm train status` any time", but `nm train`
served only --backends/--run/--continual/--quick/--export-telegram/--dataset.
The status itself already exists — `SelfImprovementStatus(context)` computes
policy + last run and already feeds the brain-side `autonomy` tool. These
tests pin the one flag + one branch over that function:

* the flag parses, and defaults off;
* never-run state reports honestly (no invented data);
* a real seeded run shows up in the runs stats;
* a due policy says so with its reasons;
* --json output parses with the same shape;
* sibling flags untouched (--backends still lists backends, plain nm train
  still says nothing to do).
"""

from __future__ import annotations

import argparse
import io
import json
import time
import unittest
from contextlib import redirect_stdout

from nomorals.agents.context import build_context
from nomorals.cmdline import _parser
from nomorals.cmdline.commands.train import _cmd_train
from nomorals.storage.db import Database


def _args(**overrides: object) -> argparse.Namespace:
    ns = argparse.Namespace(
        backends=False, run=False, backend="", base_model="",
        json=False, quick=False, status=False,
        export_telegram=False, continual=False, dataset="",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _ctx():
    db = Database(":memory:")
    db.migrate()
    return build_context(db=db, with_tools=False,
                         with_router=False, with_memory=False)


def _run_cmd(args: argparse.Namespace, ctx) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_train(args, ctx)
    return rc, buf.getvalue()


class StatusParserTests(unittest.TestCase):
    def test_train_status_parses(self) -> None:
        args = _parser().parse_args(["train", "--status"])
        self.assertTrue(args.status)

    def test_train_without_status_defaults_off(self) -> None:
        args = _parser().parse_args(["train"])
        self.assertFalse(args.status)
        self.assertFalse(args.run)

    def test_train_status_shows_in_help(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as caught, redirect_stdout(buf):
            _parser().parse_args(["train", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--status", buf.getvalue())


class StatusShapeTests(unittest.TestCase):
    def test_never_run_reports_honestly(self) -> None:
        ctx = _ctx()
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("no training run yet", text)
        self.assertIn("not due", text)
        self.assertNotIn("run None", text)

    def test_seeded_run_shows_in_runs_stats(self) -> None:
        ctx = _ctx()
        now = time.time()
        ctx.db.execute(
            "INSERT INTO training_runs (id, name, status, gate_passed,"
            " created_at, finished_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("run-seeded", "seeded", "promoted", 1, now - 3600, now - 3500),
        )
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("1 run", text)
        self.assertIn("promoted", text)
        self.assertNotIn("no training run yet", text)

    def test_due_policy_says_so_with_reasons(self) -> None:
        ctx = _ctx()
        # 150 new examples clears the 100-example growth threshold.
        ctx.db.execute(
            "INSERT INTO datasets (id, name, rows, created_at)"
            " VALUES (?, ?, ?, ?)",
            ("ds-due", "due-seed", 150, time.time()),
        )
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("due", text)
        self.assertIn("growth threshold", text)

    def test_status_json_parses_with_same_shape(self) -> None:
        ctx = _ctx()
        rc, text = _run_cmd(_args(status=True, json=True), ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(text)
        self.assertIn("should_retrain", payload)
        self.assertIn("data_available", payload)
        self.assertIn("reasons", payload)
        self.assertIn("runs", payload)
        self.assertEqual(payload["runs"]["total"], 0)


class SiblingFlagsTests(unittest.TestCase):
    def test_backends_still_lists_backends(self) -> None:
        ctx = _ctx()
        rc, text = _run_cmd(_args(backends=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("configured backend", text)

    def test_plain_train_still_nothing_to_do(self) -> None:
        ctx = _ctx()
        rc, text = _run_cmd(_args(), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", text)


if __name__ == "__main__":
    unittest.main()
