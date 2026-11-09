"""WS2 P2 — goal classes get their chat surface + brain context.

Pins:

* catalog: ``goals`` registered everywhere a command must be
  (CONTROL_COMMANDS / COMMAND_DETAILS / LIST_ONELINERS / exactly
  one LIST_GROUPS slot / help text), owner-only, and ``/goal``
  (finance savings) untouched beside it;
* the class board (goals.surface): per-class rollups from stats(),
  active goals with progress labels, designed empty states for
  classes with no goals — never a bare blank;
* the basis honesty contract in chat text: a provider-measured
  number renders ``(measured · <source>)``, a manual one
  ``(self-reported)`` — verbatim, everywhere;
* create-with-class (explicit ``in <class>``, ``<class>:`` prefix,
  and the deterministic title suggestion, named in the reply) and
  one-action re-file; unknown classes refuse in words listing the
  known classes;
* the previously-dead tracker methods wired to production callers:
  get_briefing ← /goals briefing, weekly_review ← /goals review,
  today_focus ← /goals focus + the board strip, pace_report ←
  /goals progress (+ get_briefing's pace check);
* the brain context section (context/engine.py): present only when
  goals exist, grouped by class, basis verbatim, bounded;
* the morning briefing's goal-classes section: per-class
  one-liners when the owner has goals, absent otherwise;
* ``nm goals`` CLI parity over the same surface functions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.morning_briefing import BriefingComposer
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.cmdline.commands.goals import _cmd_goals
from nomorals.context import ContextEngine
from nomorals.core.config import load_settings
from nomorals.goals import GoalTracker, surface
from nomorals.goals import classes as goal_classes
from nomorals.goals.classes import (
    ClassProgress,
    GoalClassError,
    register_progress_provider,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    help_text,
    parse_control,
)
from nomorals.storage.db import Database

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

UID = "owner"


def _tracker() -> GoalTracker:
    return GoalTracker(Database(":memory:"))


class CatalogTests(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("goals", CONTROL_COMMANDS)
        self.assertIn("goals", COMMAND_DETAILS)
        self.assertIn("goals", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("goals"), 1)
        self.assertIn("/goals", COMMAND_DETAILS["goals"]["usage"])
        self.assertTrue(COMMAND_DETAILS["goals"]["what"])
        self.assertTrue(COMMAND_DETAILS["goals"]["example"])
        self.assertIn("/goals", help_text())
        page = detailed_help("goals")
        self.assertIn("/goals", page)

    def test_owner_only(self) -> None:
        self.assertNotIn("goals", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("goals", False))
        self.assertTrue(caller_may_run("goals", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/goals add save in finance")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "goals")
        self.assertEqual(cmd.arg, "add")
        self.assertIsNotNone(parse_control("/goals"))

    def test_finance_goal_surface_untouched(self) -> None:
        # /goal stays the finance savings surface beside /goals.
        self.assertIn("goal", CONTROL_COMMANDS)
        self.assertEqual(COMMAND_DETAILS["goal"]["usage"], "/goal")
        self.assertIn("money goals", COMMAND_DETAILS["goal"]["what"])
        cmd = parse_control("/goal list")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "goal")


class SurfaceBoardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _tracker()
        self._providers = dict(goal_classes._PROGRESS_PROVIDERS)

    def tearDown(self) -> None:
        goal_classes._PROGRESS_PROVIDERS.clear()
        goal_classes._PROGRESS_PROVIDERS.update(self._providers)

    def test_board_groups_rollups_and_empty_states(self) -> None:
        goal, _, _ = surface.create_goal_filed(
            self.tracker, UID, "Save for a house", "finance")
        surface.run_sync(self.tracker.update_progress(goal.goal_id, 40.0))
        board = surface.render_class_board(self.tracker, UID)
        # rollup from stats(): counts + avg progress
        self.assertIn("**Finance** (`finance`)", board)
        self.assertIn("1 goal(s) · avg 40%", board)
        self.assertIn("Save for a house — 40% (self-reported)", board)
        # classes with no goals render designed empty states
        self.assertIn("**Careers** (`careers`)", board)
        self.assertIn("🗒 nothing here yet", board)
        self.assertIn("/goals add <title> in careers", board)
        # every registered class appears exactly once as a section
        for descriptor in goal_classes.list_classes():
            self.assertEqual(board.count(f"**{descriptor.title}**"), 1)

    def test_board_empty_user_still_renders_the_map(self) -> None:
        board = surface.render_class_board(self.tracker, UID)
        self.assertIn("0 goal(s) on file", board)
        self.assertIn("🗒 nothing here yet", board)

    def test_measured_basis_label_verbatim(self) -> None:
        register_progress_provider(
            "finance",
            lambda db, goal, *, now: ClassProgress(
                percent=71.0, basis="measured", source="ledger-test",
                headline="funded from the ledger"))
        goal, _, _ = surface.create_goal_filed(
            self.tracker, UID, "Save for a house", "finance")
        surface.run_sync(self.tracker.update_progress(goal.goal_id, 40.0))
        board = surface.render_class_board(self.tracker, UID)
        self.assertIn("71% (measured · ledger-test)", board)
        self.assertNotIn("40% (measured", board)
        detail = surface.render_goal_progress(self.tracker, UID, "house")
        self.assertIn("71% (measured · ledger-test)", detail)
        self.assertIn("funded from the ledger", detail)

    def test_single_class_board_and_unknown_class(self) -> None:
        surface.create_goal_filed(self.tracker, UID, "Ship the thing", "build")
        board = surface.render_class_board(
            self.tracker, UID, only_class="build")
        self.assertIn("**Build** (`build`)", board)
        self.assertNotIn("**Finance**", board)
        with self.assertRaises(GoalClassError) as cm:
            surface.render_class_board(self.tracker, UID, only_class="nope")
        self.assertIn("known classes", str(cm.exception))

    def test_overdue_called_out(self) -> None:
        past = time.time() - 3 * 86400
        surface.run_sync(self.tracker.create(
            UID, "Late thing", target_date=past, goal_class="build"))
        board = surface.render_class_board(self.tracker, UID)
        self.assertIn("⛔ overdue by 3d", board)
        self.assertIn("⛔ 1 overdue", board)


class SurfaceFilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _tracker()

    def test_create_explicit_suggested_and_general(self) -> None:
        goal, class_id, suggested = surface.create_goal_filed(
            self.tracker, UID, "Anything at all", "health")
        self.assertEqual((class_id, suggested), ("health", False))
        self.assertEqual(goal.goal_class, "health")

        goal, class_id, suggested = surface.create_goal_filed(
            self.tracker, UID, "Learn Yoruba deeply", None)
        self.assertEqual((class_id, suggested), ("learning", True))

        goal, class_id, suggested = surface.create_goal_filed(
            self.tracker, UID, "Water the plants", None)
        self.assertEqual((class_id, suggested), ("general", False))

    def test_create_unknown_class_refused_in_words(self) -> None:
        with self.assertRaises(GoalClassError) as cm:
            surface.create_goal_filed(self.tracker, UID, "X", "noscope")
        self.assertIn("unknown goal class 'noscope'", str(cm.exception))
        self.assertIn("finance", str(cm.exception))

    def test_refile_moves_and_names_both_classes(self) -> None:
        goal, _, _ = surface.create_goal_filed(
            self.tracker, UID, "Learn Yoruba deeply", None)
        out = surface.refile_goal(self.tracker, UID, goal.goal_id, "health")
        self.assertIn("from Learning to Health", out)
        moved = surface.run_sync(self.tracker.get(goal.goal_id))
        assert moved is not None
        self.assertEqual(moved.goal_class, "health")

    def test_refile_unknown_goal_lists_candidates(self) -> None:
        surface.create_goal_filed(self.tracker, UID, "Real goal", "build")
        out = surface.refile_goal(self.tracker, UID, "no-such", "health")
        self.assertIn("no single goal matches", out)
        self.assertIn("Real goal", out)

    def test_resolve_by_prefix_and_title(self) -> None:
        goal, _, _ = surface.create_goal_filed(
            self.tracker, UID, "Save for a house", "finance")
        hit, _ = surface.resolve_goal(self.tracker, UID, goal.goal_id[:12])
        self.assertIsNotNone(hit)
        hit, _ = surface.resolve_goal(self.tracker, UID, "house")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.goal_id, goal.goal_id)


class WiredMethodsTests(unittest.TestCase):
    """The four dead methods, exercised through their P2 callers."""

    def setUp(self) -> None:
        self.tracker = _tracker()
        surface.create_goal_filed(self.tracker, UID, "Save for a house",
                                  "finance")

    def test_briefing_via_surface(self) -> None:
        out = surface.render_briefing(self.tracker, UID)
        self.assertIn("Goal Progress Briefing", out)
        self.assertIn("Save for a house", out)

    def test_review_via_surface(self) -> None:
        out = surface.render_review(self.tracker, UID)
        self.assertIn("Weekly Review", out)

    def test_focus_via_surface(self) -> None:
        out = surface.render_focus(self.tracker, UID)
        self.assertIn("Focus now", out)
        self.assertIn("Save for a house", out)
        self.assertIn("(self-reported)", out)

    def test_pace_report_via_progress(self) -> None:
        future = time.time() + 30 * 86400
        goal = surface.run_sync(self.tracker.create(
            UID, "Dated goal", target_date=future, goal_class="build"))
        surface.run_sync(self.tracker.update_progress(goal.goal_id, 10.0))
        out = surface.render_goal_progress(self.tracker, UID, "Dated goal")
        self.assertIn("pace:", out)
        self.assertIn("days left", out)


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _tracker()
        self._providers = dict(goal_classes._PROGRESS_PROVIDERS)

    def tearDown(self) -> None:
        goal_classes._PROGRESS_PROVIDERS.clear()
        goal_classes._PROGRESS_PROVIDERS.update(self._providers)

    def test_snapshot_shape_and_basis(self) -> None:
        register_progress_provider(
            "finance",
            lambda db, goal, *, now: ClassProgress(
                percent=55.0, basis="measured", source="ledger-test"))
        surface.create_goal_filed(self.tracker, UID, "Save for a house",
                                  "finance")
        surface.create_goal_filed(self.tracker, UID, "Water the plants",
                                  None)
        snap = surface.goals_context_snapshot(self.tracker, UID)
        self.assertEqual(len(snap), 2)
        by_title = {e["title"]: e for e in snap}
        fin = by_title["Save for a house"]
        self.assertEqual(fin["goal_class"], "finance")
        self.assertEqual(fin["class_title"], "Finance")
        self.assertEqual(fin["basis"], "measured")
        self.assertEqual(fin["source"], "ledger-test")
        self.assertEqual(fin["percent"], 55.0)
        gen = by_title["Water the plants"]
        self.assertEqual(gen["basis"], "self-reported")

    def test_snapshot_bounded_by_limit(self) -> None:
        for i in range(10):
            surface.run_sync(self.tracker.create(UID, f"Goal {i}"))
        snap = surface.goals_context_snapshot(self.tracker, UID, limit=3)
        self.assertEqual(len(snap), 3)
        snap = surface.goals_context_snapshot(self.tracker, UID)
        self.assertEqual(len(snap), surface.SNAPSHOT_LIMIT)

    def test_snapshot_empty_when_no_goals(self) -> None:
        self.assertEqual(
            surface.goals_context_snapshot(self.tracker, UID), [])

    def test_briefing_lines(self) -> None:
        surface.create_goal_filed(self.tracker, UID, "Save for a house",
                                  "finance")
        lines = surface.class_briefing_lines(self.tracker, UID)
        self.assertEqual(len(lines), 1)
        self.assertIn("Finance: 1 active of 1", lines[0])
        self.assertEqual(
            surface.class_briefing_lines(_tracker(), UID), [])


class ContextSectionTests(unittest.TestCase):
    def _entries(self, n: int = 2) -> list[dict[str, Any]]:
        base = [
            {"title": "Save for a house", "goal_class": "finance",
             "class_title": "Finance", "percent": 71.0,
             "basis": "measured", "source": "ledger-test"},
            {"title": "Learn Yoruba", "goal_class": "learning",
             "class_title": "Learning", "percent": 10.0,
             "basis": "self-reported", "source": "manual",
             "pace": "at_risk"},
        ]
        extra = [{"title": f"Goal {i}", "goal_class": "general",
                  "class_title": "General", "percent": float(i),
                  "basis": "self-reported", "source": "manual"}
                 for i in range(max(0, n - 2))]
        return (base + extra)[:n]

    def test_section_present_grouped_with_basis(self) -> None:
        built = ContextEngine().build(system="sys", goals=self._entries())
        names = [s.name for s in built.sections]
        self.assertIn("goals", names)
        section = next(s for s in built.sections if s.name == "goals")
        self.assertIn("Finance:", section.content)
        self.assertIn("Save for a house — 71% (measured · ledger-test)",
                      section.content)
        self.assertIn("Learn Yoruba — 10% (self-reported)", section.content)
        self.assertIn("pace: at risk", section.content)
        self.assertIn("goals", built.text)

    def test_section_absent_without_goals(self) -> None:
        built = ContextEngine().build(system="sys")
        self.assertNotIn("goals", [s.name for s in built.sections])
        built = ContextEngine().build(system="sys", goals=[])
        self.assertNotIn("goals", [s.name for s in built.sections])

    def test_section_bounded(self) -> None:
        built = ContextEngine().build(system="sys", goals=self._entries(12))
        section = next(s for s in built.sections if s.name == "goals")
        goal_lines = [ln for ln in section.content.splitlines()
                      if ln.startswith("- ")]
        self.assertEqual(len(goal_lines), ContextEngine._GOALS_SECTION_MAX)
        self.assertIn("+4 more goals not shown", section.content)

    def test_section_accepts_callable(self) -> None:
        built = ContextEngine().build(
            system="sys", goals=lambda: self._entries())
        self.assertIn("goals", [s.name for s in built.sections])

    def test_section_survives_garbage(self) -> None:
        built = ContextEngine().build(system="sys", goals=object())
        self.assertNotIn("goals", [s.name for s in built.sections])

    def test_rich_step_prompt_carries_goals(self) -> None:
        from types import SimpleNamespace as NS
        mission = NS(goal="Do the mission", id="m1", status="running",
                     state={})
        step = NS(goal="Step one", name="step")
        text = ContextEngine().build_step_prompt(
            mission, step, rich=True, goals=self._entries())
        self.assertIn("Save for a house", text)


class BriefingProviderTests(unittest.TestCase):
    def test_section_present_with_class_lines(self) -> None:
        tracker = _tracker()
        surface.create_goal_filed(tracker, UID, "Save for a house",
                                  "finance")
        ctx = SimpleNamespace(db=tracker.db)
        briefing = BriefingComposer().compose(ctx, "2026-11-09")
        section = next((s for s in briefing.sections
                        if s.name == "goal_classes"), None)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(section.source, "goal-tracker")
        self.assertTrue(any("Finance:" in ln for ln in section.lines))

    def test_section_absent_without_goals_or_db(self) -> None:
        briefing = BriefingComposer().compose(
            SimpleNamespace(db=Database(":memory:")), "2026-11-09")
        self.assertNotIn("goal_classes",
                         [s.name for s in briefing.sections])
        briefing = BriefingComposer().compose(
            SimpleNamespace(db=None), "2026-11-09")
        self.assertNotIn("goal_classes",
                         [s.name for s in briefing.sections])


class ChatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-goals-p2-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        reply = self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))
        return reply.text if hasattr(reply, "text") else reply

    def test_board_is_actionable_with_grid_picker(self) -> None:
        self._run("/goals add Save for a house in finance")
        reply = self.runtime.handle_control(
            "/goals", "telegram:999",
            message=_msg("/goals", chat_key="telegram:999"))
        self.assertTrue(hasattr(reply, "buttons"))
        rows = reply.buttons
        self.assertTrue(all(len(row) <= 2 for row in rows))
        flat = [data for row in rows for _, data in row]
        self.assertIn("goals list finance", flat)
        self.assertIn("goals briefing", flat)
        self.assertIn("Save for a house — 0% (self-reported)", reply.text)
        self.assertIn("🗒 nothing here yet", reply.text)

    def test_add_forms_and_refile(self) -> None:
        out = self._run("/goals add finance: Fund the runway")
        self.assertIn("under Finance (`finance`)", out)
        out = self._run("/goals add Learn Yoruba")
        self.assertIn("filed under Learning from the title", out)
        out = self._run("/goals class Yoruba health")
        self.assertIn("from Learning to Health", out)
        out = self._run("/goals list health")
        self.assertIn("Learn Yoruba", out)

    def test_add_unknown_class_refuses_listing_classes(self) -> None:
        out = self._run("/goals add Something in noscope")
        self.assertIn("unknown goal class 'noscope'", out)
        self.assertIn("known classes", out)

    def test_wired_verbs_render(self) -> None:
        self._run("/goals add Save for a house in finance")
        self.assertIn("Goal Progress Briefing", self._run("/goals briefing"))
        self.assertIn("Weekly Review", self._run("/goals review"))
        self.assertIn("Focus now", self._run("/goals focus"))
        progress = self._run("/goals progress house")
        self.assertIn("progress:", progress)
        self.assertIn("(self-reported)", progress)
        self.assertIn("Goal classes", self._run("/goals classes"))

    def test_usage_for_bare_progress(self) -> None:
        self.assertIn("usage: /goals progress", self._run("/goals progress"))


def _cli_ctx() -> SimpleNamespace:
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, settings=SimpleNamespace(), router=None)


def _cli_ns(**overrides: object) -> argparse.Namespace:
    base = dict(action="list", arg="", arg2="", title="", goal_class="",
                user="owner", json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _cli(args: argparse.Namespace, ctx: Any) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _cmd_goals(args, ctx)
    return code, buf.getvalue()


class CliTests(unittest.TestCase):
    def test_add_list_class_flow(self) -> None:
        ctx = _cli_ctx()
        code, out = _cli(_cli_ns(action="add", arg="Save for a house",
                                 goal_class="finance"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("[class: finance]", out)
        code, out = _cli(_cli_ns(action="list"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("Save for a house", out)
        code, out = _cli(_cli_ns(action="briefing"), ctx)
        self.assertIn("Goal Progress Briefing", out)
        code, out = _cli(_cli_ns(action="class", arg="house",
                                 arg2="health"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("to Health", out)

    def test_add_unknown_class_fails(self) -> None:
        code, out = _cli(_cli_ns(action="add", arg="X",
                                 goal_class="noscope"), _cli_ctx())
        self.assertEqual(code, 1)
        self.assertIn("unknown goal class", out)


if __name__ == "__main__":
    unittest.main()
