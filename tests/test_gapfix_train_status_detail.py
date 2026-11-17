"""Gapfix: `nm train status` per-run detail.

The `--status` ship (b3825033) shows aggregates only, and its survey names
why: "SelfImprovementStatus returns no per-run record." The rows were there
all along — `TrainingRegistry.stats()` iterated `runs.all()` to compute the
totals and then threw the rows away. These tests pin the fix:

* `stats()` carries a "recent" key — the last 5 runs, newest first, with
  exactly the fields a status reader needs (name, status, gate_passed,
  created_at, started_at, finished_at, error);
* the aggregates are unchanged by the addition;
* an empty registry returns an empty recent list;
* the CLI renders the recent runs under the aggregate line (and omits the
  section when there are none);
* --json carries the recent list.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import unittest
from contextlib import redirect_stdout

from nomorals.agents.context import build_context
from nomorals.cmdline.commands.train import _cmd_train
from nomorals.storage.db import Database
from nomorals.training.registry import TrainingRegistry


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


def _seed(db, run_id: str, name: str, status: str, gate: int | None,
          created: float, error: str = "") -> None:
    db.execute(
        "INSERT INTO training_runs (id, name, status, gate_passed,"
        " created_at, started_at, finished_at, error)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, name, status, gate, created, created + 10,
         created + 600, error),
    )


class StatsRecentTests(unittest.TestCase):
    def test_recent_newest_first_with_detail_fields(self) -> None:
        ctx = _ctx()
        now = time.time()
        _seed(ctx.db, "r-old", "old-run", "done", 1, now - 7200)
        _seed(ctx.db, "r-mid", "mid-run", "failed", 0, now - 3600,
              error="boom: out of memory")
        _seed(ctx.db, "r-new", "new-run", "promoted", 1, now - 60)
        stats = TrainingRegistry(ctx.db).stats()
        # Aggregates unchanged by the addition.
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["promoted"], 1)
        self.assertEqual(stats["gate_passed"], 2)
        recent = stats["recent"]
        self.assertEqual([r["name"] for r in recent],
                         ["new-run", "mid-run", "old-run"])
        self.assertEqual(set(recent[0]),
                         {"name", "status", "gate_passed", "created_at",
                          "started_at", "finished_at", "error"})
        newest = recent[0]
        self.assertEqual(newest["status"], "promoted")
        self.assertIs(newest["gate_passed"], True)
        self.assertAlmostEqual(newest["created_at"], now - 60, delta=1)
        self.assertAlmostEqual(newest["started_at"], now - 50, delta=1)
        self.assertAlmostEqual(newest["finished_at"], now + 540, delta=1)
        self.assertEqual(newest["error"], "")
        failed = recent[1]
        self.assertIs(failed["gate_passed"], False)
        self.assertEqual(failed["error"], "boom: out of memory")

    def test_empty_registry_returns_empty_recent(self) -> None:
        ctx = _ctx()
        stats = TrainingRegistry(ctx.db).stats()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["recent"], [])

    def test_recent_capped_at_five(self) -> None:
        ctx = _ctx()
        now = time.time()
        for i in range(7):
            _seed(ctx.db, f"r-{i}", f"run-{i}", "done", 1,
                  now - (6 - i) * 1000)
        stats = TrainingRegistry(ctx.db).stats()
        self.assertEqual(stats["total"], 7)
        self.assertEqual(len(stats["recent"]), 5)  # RECENT_RUNS in registry.py
        self.assertEqual([r["name"] for r in stats["recent"]],
                         ["run-6", "run-5", "run-4", "run-3", "run-2"])

    def test_recent_via_rowstore_branch(self) -> None:
        # The Mongo seam branch: _runs() returns a rowstore whose all()
        # yields stored rows; the recent list is sorted newest-first
        # there too, and the aggregates still count every row.
        ctx = _ctx()
        now = time.time()
        rows = [
            {"id": "s-old", "name": "seam-old", "status": "done",
             "gate_passed": 1, "created_at": now - 5000,
             "started_at": now - 4990, "finished_at": now - 4400,
             "error": "", "config": "{}", "metrics": "{}"},
            {"id": "s-new", "name": "seam-new", "status": "rejected",
             "gate_passed": 0, "created_at": now - 100,
             "started_at": now - 90, "finished_at": now - 10,
             "error": "gate: accuracy fell", "config": "{}",
             "metrics": "{}"},
        ]
        registry = TrainingRegistry(ctx.db)

        class _FakeRows:
            def all(self):
                return list(rows)

        registry._runs = lambda: _FakeRows()  # noqa: SLF001 — seam stub
        stats = registry.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual([r["name"] for r in stats["recent"]],
                         ["seam-new", "seam-old"])
        self.assertEqual(stats["recent"][0]["error"],
                         "gate: accuracy fell")
        self.assertIs(stats["recent"][0]["gate_passed"], False)


class CliRecentTests(unittest.TestCase):
    def test_status_renders_recent_runs(self) -> None:
        ctx = _ctx()
        now = time.time()
        _seed(ctx.db, "r-1", "nightly", "promoted", 1, now - 3600)
        _seed(ctx.db, "r-2", "manual", "failed", 0, now - 1800,
              error="disk full while saving adapter")
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("2 runs — 1 promoted, 0 rejected, 1 passed the gate",
                      text)
        self.assertIn("recent runs:", text)
        self.assertIn("nightly — promoted — gate passed", text)
        self.assertIn("manual — failed — gate failed", text)
        self.assertIn("disk full while saving adapter", text)
        self.assertRegex(text, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")
        # Newest first: the manual run (later) renders before nightly.
        self.assertLess(text.index("manual"), text.index("nightly"))

    def test_status_omits_recent_section_when_none(self) -> None:
        ctx = _ctx()
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("no training run yet", text)
        self.assertNotIn("recent runs:", text)

    def test_gate_unknown_renders_without_gate_text(self) -> None:
        ctx = _ctx()
        now = time.time()
        _seed(ctx.db, "r-1", "in-flight", "running", None, now - 300)
        rc, text = _run_cmd(_args(status=True), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("in-flight — running", text)
        self.assertNotIn("gate passed", text)
        self.assertNotIn("gate failed", text)

    def test_status_json_carries_recent(self) -> None:
        ctx = _ctx()
        now = time.time()
        _seed(ctx.db, "r-1", "nightly", "promoted", 1, now - 3600)
        rc, text = _run_cmd(_args(status=True, json=True), ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(text)
        recent = payload["runs"]["recent"]
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["name"], "nightly")
        self.assertEqual(recent[0]["status"], "promoted")
        self.assertIs(recent[0]["gate_passed"], True)
        # The aggregate line is unchanged.
        self.assertEqual(payload["runs"]["total"], 1)
        self.assertEqual(payload["runs"]["promoted"], 1)


if __name__ == "__main__":
    unittest.main()
