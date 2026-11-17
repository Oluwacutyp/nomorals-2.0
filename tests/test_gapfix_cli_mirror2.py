"""CLI-mirror bundle #2: ``nm focus`` + ``nm workflow`` + ``nm plan`` + ``nm museum``.

Failing-first: on the pristine base none of the four surfaces exists
(no ``nomorals.cmdline.commands.focus`` / ``workflow`` / ``plan`` /
``museum`` modules, no parsers), so every test here fails there.

Each command mirrors its chat verb over the same durable store:

1. ``nm focus`` — the chat /focus verbs (runtime_reach._control_focus)
   over FocusTimer at ``<home>/data/focus.json``: start/pause/resume/
   stop/status/today/week/month. Week/month render through
   reviews/review.py build_weekly_review / build_monthly_review, the
   same composer chat uses — never a second copy.
2. ``nm workflow`` — the store verbs only (list/show/status/run/cancel/
   resume) over WorkflowStore. Drafting from ``<task>`` needs the brain
   and stays chat-only; the usage text says so.
3. ``nm plan`` — the store verbs only (list/show/status) over
   PlanStore. Drafting stays chat-only, same usage note.
4. ``nm museum`` — census + find over the Museum, built from the same
   roots as the museum_census tool (books/tools._museum). ``send``
   needs a live chat and is refused in words from the console, with
   chat's exact wording.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.focus.sessions import week_bounds


def _ctx(home: Path):
    return SimpleNamespace(
        db=None,
        settings=SimpleNamespace(
            home=str(home), home_path=Path(home),
            workspace_dir=home / "workspace"),
    )


def _ns(**kwargs):
    args = argparse.Namespace(
        focus_action="status", focus_rest=[], workflow_action="list",
        workflow_rest=[], plan_action="list", plan_rest=[],
        museum_action=None, museum_rest=[], json=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def _run(fn, args, ctx):
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = fn(args, ctx)
    return rc, buf.getvalue(), err.getvalue()


def _seed_week(home: Path):
    """One kept promise + one resolved decision inside the current week."""
    from nomorals.commitments.ledger import CommitmentLedger
    from nomorals.decisions.journal import DecisionJournal

    now = time.time()
    start, _end = week_bounds(now)
    inside = start + 3600
    data = home / "data"
    data.mkdir(parents=True, exist_ok=True)
    ledger = CommitmentLedger(str(data / "commitments.json"))
    c = ledger.add(what="Ship the onboarding flow",
                   deadline=now + 86400, ts=inside)
    ledger.mark_kept(c.id, "shipped it", ts=inside + 60)
    journal = DecisionJournal(str(data / "decisions.json"))
    d = journal.record(decision="Ship the onboarding flow",
                       chosen="ship", prediction="users love it",
                       confidence=0.8, ts=inside)
    journal.resolve(d.id, outcome="shipped, no regressions", success=True)
    return c, d


class FocusCliTests(unittest.TestCase):
    def test_start_status_pause_resume_stop_round_trip(self):
        from nomorals.cmdline.commands.focus import _cmd_focus
        from nomorals.focus.sessions import FocusTimer

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            ctx = _ctx(home)
            rc, out, _ = _run(_cmd_focus,
                              _ns(focus_action="start",
                                  focus_rest=["deep work"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("focus started (deep work)", out)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="status"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("focus running (deep work)", out)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="pause"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("paused (deep work)", out)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="resume"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("resumed (deep work)", out)
            rc, out, _ = _run(_cmd_focus,
                              _ns(focus_action="stop",
                                  focus_rest=["done"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("stopped (deep work)", out)
            self.assertIn("focused", out)
            timer = FocusTimer(str(home / "data" / "focus.json"))
            self.assertEqual(len(timer.sessions), 1)
            self.assertEqual(list(timer.sessions.values())[0].status, "done")

    def test_start_twice_is_an_error_not_a_second_session(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            rc, _, _ = _run(_cmd_focus,
                            _ns(focus_action="start", focus_rest=["a"]), ctx)
            self.assertEqual(rc, 0)
            rc, _, err = _run(
                _cmd_focus, _ns(focus_action="start", focus_rest=["b"]), ctx)
            self.assertEqual(rc, 2)
            self.assertIn("already running", err)

    def test_pause_with_nothing_running_is_an_error(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(_cmd_focus, _ns(focus_action="pause"),
                              _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("no focus session is running", err)

    def test_status_with_nothing_running_names_today(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="status"),
                              _ctx(Path(td)))
            self.assertEqual(rc, 0)
            self.assertIn("no focus session running", out)
            self.assertIn("today:", out)
            self.assertIn("nm focus start [label] to begin", out)

    def test_today_counts_a_seeded_session(self):
        from nomorals.cmdline.commands.focus import _cmd_focus
        from nomorals.focus.sessions import FocusTimer

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            now = time.time()
            timer = FocusTimer(str(home / "data" / "focus.json"))
            timer.start(label="reading", ts=now - 600)
            timer.stop(ts=now - 60)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="today"),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("today:", out)
            self.assertIn("reading", out)
            self.assertIn("1 session(s)", out)

    def test_status_json_is_structured(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            ctx = _ctx(home)
            _run(_cmd_focus, _ns(focus_action="start",
                                 focus_rest=["json-test"]), ctx)
            rc, out, _ = _run(_cmd_focus,
                              _ns(focus_action="status", json=True), ctx)
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(payload["active"]["label"], "json-test")
            self.assertIn("today", payload)

    def test_week_renders_the_seeded_week(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_week(home)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="week"),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("Ship the onboarding flow", out)

    def test_month_renders_without_fabricating_weeks(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _seed_week(home)
            rc, out, _ = _run(_cmd_focus, _ns(focus_action="month"),
                              _ctx(home))
            self.assertEqual(rc, 0)
            self.assertTrue(out.strip())

    def test_unknown_verb_is_usage(self):
        from nomorals.cmdline.commands.focus import _cmd_focus

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(_cmd_focus, _ns(focus_action="explode"),
                              _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("usage: nm focus", err)


class WorkflowCliTests(unittest.TestCase):
    def _seed(self, home: Path):
        from nomorals.agents.workflow.store import (
            WorkflowStore,
            default_workflows_dir,
            default_workflows_path,
        )

        store = WorkflowStore(path=default_workflows_path(home),
                              scripts_dir=default_workflows_dir(home))
        return store.save_workflow(name="nightly", task="do nightly things",
                                   script="print('hi')\n")

    def test_list_show_status_run_cancel_resume_round_trip(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed(home)
            ctx = _ctx(home)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="list"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("nightly", out)
            self.assertIn("[queued]", out)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="show",
                                  workflow_rest=["nightly"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("print('hi')", out)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="status",
                                  workflow_rest=["nightly"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("nightly [queued]", out)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="cancel",
                                  workflow_rest=["nightly"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("cancelled", out)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="run",
                                  workflow_rest=["nightly"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("queued again", out)

    def test_resume_picks_up_a_failed_run(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            record = self._seed(home)
            from nomorals.agents.workflow.store import (
                WorkflowStore,
                default_workflows_dir,
                default_workflows_path,
            )

            store = WorkflowStore(path=default_workflows_path(home),
                                  scripts_dir=default_workflows_dir(home))
            store.mark_running(record["id"])
            store.mark_failed(record["id"], error="boom")
            ctx = _ctx(home)
            rc, out, _ = _run(_cmd_workflow,
                              _ns(workflow_action="resume",
                                  workflow_rest=["nightly"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("queued again", out)

    def test_resume_on_a_cancelled_run_is_refused(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            record = self._seed(home)
            from nomorals.agents.workflow.store import (
                WorkflowStore,
                default_workflows_dir,
                default_workflows_path,
            )

            store = WorkflowStore(path=default_workflows_path(home),
                                  scripts_dir=default_workflows_dir(home))
            store.cancel(record["id"], by="owner")
            rc, _, err = _run(_cmd_workflow,
                              _ns(workflow_action="resume",
                                  workflow_rest=["nightly"]), _ctx(home))
            self.assertEqual(rc, 2)
            self.assertIn("cancelled", err)

    def test_empty_list_is_honest(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = _run(_cmd_workflow, _ns(workflow_action="list"),
                              _ctx(Path(td)))
            self.assertEqual(rc, 0)
            self.assertIn("no workflows yet", out)

    def test_unknown_name_is_an_error(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed(home)
            rc, _, err = _run(
                _cmd_workflow,
                _ns(workflow_action="show", workflow_rest=["nope"]),
                _ctx(home))
            self.assertEqual(rc, 2)
            self.assertIn("no workflow 'nope'", err)

    def test_a_task_is_refused_to_chat(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(
                _cmd_workflow,
                _ns(workflow_action="do", workflow_rest=["the", "thing"]),
                _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("needs the brain", err)
            self.assertIn("/workflow", err)

    def test_list_json_is_structured(self):
        from nomorals.cmdline.commands.workflow import _cmd_workflow

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed(home)
            rc, out, _ = _run(
                _cmd_workflow, _ns(workflow_action="list", json=True),
                _ctx(home))
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(len(payload["workflows"]), 1)
            self.assertEqual(payload["workflows"][0]["name"], "nightly")


class PlanCliTests(unittest.TestCase):
    def _seed(self, home: Path):
        from nomorals.agents.plan.store import PlanStore, default_plans_dir, default_plans_path

        store = PlanStore(path=default_plans_path(home),
                          plans_dir=default_plans_dir(home))
        record = store.save_plan(task="rebuild the intake",
                                 survey={}, plan={}, runs=[],
                                 unassigned=[])
        store.write_artifact(record)
        return record

    def test_list_show_status_round_trip(self):
        from nomorals.cmdline.commands.plan import _cmd_plan

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            record = self._seed(home)
            ctx = _ctx(home)
            rc, out, _ = _run(_cmd_plan, _ns(plan_action="list"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn(record["id"], out)
            self.assertIn("rebuild the intake", out)
            rc, out, _ = _run(_cmd_plan,
                              _ns(plan_action="show",
                                  plan_rest=[record["id"]]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("rebuild the intake", out)
            rc, out, _ = _run(_cmd_plan,
                              _ns(plan_action="status",
                                  plan_rest=[record["id"]]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn(f"plan {record['id']}", out)
            self.assertIn("rebuild the intake", out)

    def test_empty_list_is_honest(self):
        from nomorals.cmdline.commands.plan import _cmd_plan

        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = _run(_cmd_plan, _ns(plan_action="list"),
                              _ctx(Path(td)))
            self.assertEqual(rc, 0)
            self.assertIn("no plans yet", out)

    def test_unknown_id_is_an_error(self):
        from nomorals.cmdline.commands.plan import _cmd_plan

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(
                _cmd_plan, _ns(plan_action="show", plan_rest=["nope"]),
                _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("no plan 'nope'", err)

    def test_a_task_is_refused_to_chat(self):
        from nomorals.cmdline.commands.plan import _cmd_plan

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(
                _cmd_plan,
                _ns(plan_action="rebuild", plan_rest=["the", "intake"]),
                _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("needs the brain", err)
            self.assertIn("/plan", err)

    def test_list_json_is_structured(self):
        from nomorals.cmdline.commands.plan import _cmd_plan

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            record = self._seed(home)
            rc, out, _ = _run(_cmd_plan, _ns(plan_action="list", json=True),
                              _ctx(home))
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(len(payload["plans"]), 1)
            self.assertEqual(payload["plans"][0]["id"], record["id"])


class MuseumCliTests(unittest.TestCase):
    def _seed_book(self, home: Path):
        from nomorals.books.library import Library

        ctx = _ctx(home)
        book = home / "workspace" / "seed-book.txt"
        book.parent.mkdir(parents=True, exist_ok=True)
        book.write_text(("Seed book. " * 40) + "\n", encoding="utf-8")
        result = Library(ctx).ingest(book, title="Seed Book")
        return result

    def test_empty_museum_is_honest(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = _run(_cmd_museum, _ns(), _ctx(Path(td)))
            self.assertEqual(rc, 0)
            self.assertIn("museum is empty", out)

    def test_census_lists_a_seeded_book(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_book(home)
            rc, out, _ = _run(_cmd_museum, _ns(), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("library-books", out)
            self.assertIn("Seed Book", out)

    def test_shelf_filters_and_unknown_shelf_names_the_real_ones(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_book(home)
            ctx = _ctx(home)
            rc, out, _ = _run(
                _cmd_museum,
                _ns(museum_action="library-books"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("Seed Book", out)
            rc, out, _ = _run(
                _cmd_museum, _ns(museum_action="nope"), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("no shelf 'nope' in the museum", out)
            self.assertIn("library-books", out)

    def test_find_resolves_one_card(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            result = self._seed_book(home)
            ctx = _ctx(home)
            rc, out, _ = _run(
                _cmd_museum,
                _ns(museum_action="find",
                    museum_rest=[f"library-book:{result.slug}"]), ctx)
            self.assertEqual(rc, 0)
            self.assertIn("Seed Book", out)

    def test_find_unknown_is_an_error(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _run(
                _cmd_museum,
                _ns(museum_action="find", museum_rest=["nope"]), _ctx(Path(td)))
            self.assertEqual(rc, 2)
            self.assertIn("no single card matches 'nope'", err)

    def test_send_is_refused_in_words(self):
        from nomorals.cmdline.commands.museum import _cmd_museum

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            self._seed_book(home)
            rc, out, _ = _run(
                _cmd_museum,
                _ns(museum_action="send",
                    museum_rest=["library-book:seed-book"]), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("send needs a live chat", out)
            self.assertIn("not the console", out)


class ParserTests(unittest.TestCase):
    def test_four_parsers_registered(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        actions = parser._actions
        sub = next(a for a in actions
                   if isinstance(a, argparse._SubParsersAction))
        for name in ("focus", "workflow", "plan", "museum"):
            self.assertIn(name, sub.choices)

    def test_director_plan_untouched(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        actions = parser._actions
        sub = next(a for a in actions
                   if isinstance(a, argparse._SubParsersAction))
        director = sub.choices["director"]
        d_actions = director._actions
        d_sub = next(a for a in d_actions
                     if isinstance(a, argparse._SubParsersAction))
        self.assertIn("plan", d_sub.choices)


if __name__ == "__main__":
    unittest.main()
