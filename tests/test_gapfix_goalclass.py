"""Goal-class lane — custom classes as data, per-class milestones,
check-in cadence enacted, and the brain's ``goal_class`` tool.

The owner-named classes (careers, productivity, finance, interests,
"something else") already existed as a built-in registry (WS2); what
was missing — and what this suite pins — is that a class the owner
names that is NOT built in becomes a real class without a code
change, that each class carries its own milestone shape, that the
declared review cadence actually decides when a check-in is due,
and that the brain can work the user goal-class store through one
tool (registry 785 -> 786 with ``goal_class``).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import time
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.cmdline.commands.goals import _cmd_goals
from nomorals.goals import (
    GOAL_CLASSES,
    GoalTracker,
    surface,
)
from nomorals.goals import classes as goal_classes
from nomorals.goals.classes import (
    GoalClassError,
    get_class,
    list_classes,
)
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

UID = "owner"


def _db() -> Database:
    return Database(":memory:")


def _tracker() -> GoalTracker:
    return GoalTracker(_db())


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _tool(registry: ToolRegistry, **kwargs: Any) -> dict[str, Any]:
    outcome = registry.call("goal_class", **kwargs)
    assert getattr(outcome, "ok", False), outcome
    return outcome.value


def _registry(db: Database | None) -> ToolRegistry:
    ctx = SimpleNamespace(db=db) if db is not None else SimpleNamespace()
    registry = ToolRegistry(ctx)
    registry.register_builtins()
    return registry


class _Rt(RuntimeSystemMixin):
    """The system mixin with only a db-backed context (the /goals
    handler needs nothing else except the careers store, which only
    the ``/goals list careers`` path touches)."""

    def __init__(self, db: Database) -> None:
        self.context = SimpleNamespace(db=db)


def _text(reply: Any) -> str:
    return reply.text if hasattr(reply, "text") else reply


def _cli_ctx() -> SimpleNamespace:
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(db=db, settings=SimpleNamespace(), router=None)


def _cli_ns(**overrides: object) -> argparse.Namespace:
    base = dict(action="list", arg="", arg2="", title="", goal_class="",
                progress="", note="", user="owner", json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _cli(args: argparse.Namespace, ctx: Any) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _cmd_goals(args, ctx)
    return code, buf.getvalue()


class ToolCountTests(unittest.TestCase):
    def test_tool_count_is_786_after_goalclass(self):
        # 785 after C.8 added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control. 809 after this lane (moneyprinter) added make_short. 810 after this lane (VoiceStudio) added voicestudio; 811 after this lane (Animato) added animato_spark.
        # added goal_class.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 811)
        self.assertIn("goal_class", registry._tools)
        self.assertEqual(len(registry.schemas()), 811)


class BuiltinRegistryTests(unittest.TestCase):
    def test_builtin_list_still_exactly_ten(self):
        self.assertEqual(
            [c.id for c in list_classes()],
            ["careers", "productivity", "finance", "interests", "health",
             "learning", "build", "relationships", "trading", "general"])
        self.assertEqual(sorted(GOAL_CLASSES), sorted(
            ["careers", "productivity", "finance", "interests", "health",
             "learning", "build", "relationships", "trading", "general"]))

    def test_builtin_milestone_templates(self):
        self.assertEqual(
            get_class("careers").milestone_templates,
            ("Submit the first application", "Reach the interview stage",
             "Receive an offer"))
        self.assertEqual(
            get_class("finance").milestone_templates,
            ("Reach 25% funded", "Reach 50% funded", "Fully funded"))
        self.assertEqual(get_class("general").milestone_templates, ())

    def test_cadence_days(self):
        self.assertEqual(goal_classes.cadence_days("daily-brief"), 1)
        self.assertEqual(goal_classes.cadence_days("weekly"), 7)
        self.assertEqual(goal_classes.cadence_days("digest:finance"), 7)
        self.assertEqual(goal_classes.cadence_days("monthly"), 30)
        self.assertEqual(goal_classes.cadence_days("nonsense"), 7)
        self.assertEqual(goal_classes.cadence_days(None), 7)


class CustomClassTests(unittest.TestCase):
    def test_define_list_and_redefine(self):
        gt = _tracker()
        made = gt.define_class(
            "gardening", "Gardening", "Growing things.",
            milestone_templates=("First bed planted", "First harvest"))
        self.assertEqual(made.id, "gardening")
        self.assertEqual(made.progress_model, "manual")
        self.assertEqual(made.milestone_templates,
                         ("First bed planted", "First harvest"))
        ids = [c.id for c in gt.list_all_classes()]
        self.assertEqual(len(ids), 11)
        self.assertIn("gardening", ids)
        self.assertEqual(ids[:10], [c.id for c in list_classes()])
        # Re-defining a custom id replaces its descriptor (data can
        # be corrected); the count does not grow.
        again = gt.define_class("gardening", "Gardening!", "New words.")
        self.assertEqual(again.title, "Gardening!")
        self.assertEqual(len(gt.list_all_classes()), 11)

    def test_define_refusals(self):
        gt = _tracker()
        for bad_id in ("finance", "Finance", "Bad ID!", "1abc", "", "a" * 40):
            with self.assertRaises(GoalClassError, msg=bad_id):
                gt.define_class(bad_id, "X")
        with self.assertRaises(GoalClassError):
            gt.define_class("ok-id", "   ")

    def test_remove_rules(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening")
        with self.assertRaises(GoalClassError):
            gt.remove_class("finance")  # built-in: never removable
        with self.assertRaises(GoalClassError):
            gt.remove_class("nosuch")  # unknown: not a custom class
        goal = _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        with self.assertRaises(GoalClassError):
            gt.remove_class("gardening")  # in use
        _run(gt.delete(goal.goal_id))
        self.assertEqual(gt.remove_class("gardening"), 0)
        self.assertNotIn("gardening",
                         [c.id for c in gt.list_all_classes()])
        # And the id is free to define again afterwards.
        gt.define_class("gardening", "Gardening")
        self.assertIn("gardening", [c.id for c in gt.list_all_classes()])

    def test_custom_class_survives_new_tracker(self):
        db = _db()
        gt = GoalTracker(db)
        gt.define_class("gardening", "Gardening")
        gt2 = GoalTracker(db)  # same database, fresh tracker
        self.assertIn("gardening", [c.id for c in gt2.list_all_classes()])


class TrackerClassTests(unittest.TestCase):
    def test_create_list_refile_in_custom_class(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening")
        goal = _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        self.assertEqual(goal.goal_class, "gardening")
        listed = _run(gt.list_goals(UID, goal_class="gardening"))
        self.assertEqual([g.goal_id for g in listed], [goal.goal_id])
        self.assertEqual(_run(gt.list_goals(UID, goal_class="finance")), [])
        moved = _run(gt.update(goal.goal_id, goal_class="finance"))
        self.assertEqual(moved.goal_class, "finance")
        self.assertEqual(_run(gt.list_goals(UID, goal_class="gardening")), [])

    def test_unknown_class_refused_everywhere(self):
        gt = _tracker()
        with self.assertRaises(GoalClassError):
            _run(gt.create(UID, "X", goal_class="noscope"))
        goal = _run(gt.create(UID, "Y"))
        self.assertEqual(goal.goal_class, "general")
        with self.assertRaises(GoalClassError):
            _run(gt.update(goal.goal_id, goal_class="noscope"))
        with self.assertRaises(GoalClassError):
            _run(gt.list_goals(UID, goal_class="noscope"))

    def test_stats_by_class_includes_custom(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening")
        _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        _run(gt.create(UID, "Save for a plot", goal_class="finance"))
        stats = _run(gt.stats(UID))
        self.assertEqual(stats["by_class"]["gardening"]["total"], 1)
        self.assertEqual(stats["by_class"]["finance"]["total"], 1)

    def test_class_progress_labels_custom_as_self_reported(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening")
        goal = _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        _run(gt.update_progress(goal.goal_id, 40.0))
        goal = _run(gt.get(goal.goal_id))
        assert goal is not None
        reading = gt.class_progress(goal)
        self.assertEqual(reading.basis, "self-reported")
        self.assertEqual(reading.percent, 40.0)


class MilestoneTests(unittest.TestCase):
    def test_seed_from_builtin_templates_on_filed_create(self):
        gt = _tracker()
        goal, class_id, _suggested = surface.create_goal_filed(
            gt, UID, "Apply for backend roles", "careers")
        self.assertEqual(class_id, "careers")
        titles = [m.title for m in _run(gt.list_milestones(goal.goal_id))]
        self.assertEqual(titles, ["Submit the first application",
                                  "Reach the interview stage",
                                  "Receive an offer"])

    def test_seed_general_seeds_nothing(self):
        gt = _tracker()
        goal, class_id, _suggested = surface.create_goal_filed(
            gt, UID, "Do a vague thing", None)
        self.assertEqual(class_id, "general")
        self.assertEqual(_run(gt.list_milestones(goal.goal_id)), [])

    def test_seed_custom_templates_and_idempotent(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening",
                        milestone_templates=("First bed planted",
                                             "First harvest"))
        goal = _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        self.assertEqual(_run(gt.seed_class_milestones(goal.goal_id)), 2)
        self.assertEqual(_run(gt.seed_class_milestones(goal.goal_id)), 0)
        titles = [m.title for m in _run(gt.list_milestones(goal.goal_id))]
        self.assertEqual(titles, ["First bed planted", "First harvest"])

    def test_seed_skips_titles_already_present(self):
        gt = _tracker()
        goal = _run(gt.create(UID, "Apply for backend roles",
                              goal_class="careers"))
        _run(gt.add_milestone(goal.goal_id, "Receive an offer"))
        added = _run(gt.seed_class_milestones(goal.goal_id))
        self.assertEqual(added, 2)
        titles = [m.title for m in _run(gt.list_milestones(goal.goal_id))]
        self.assertEqual(titles.count("Receive an offer"), 1)


class CheckinTests(unittest.TestCase):
    def _stale_goal(self, gt, title, class_id, days_ago, now):
        goal = _run(gt.create(UID, title, goal_class=class_id))
        gt.db.execute("UPDATE goals SET created_at = ? WHERE goal_id = ?",
                      (now - days_ago * 86400, goal.goal_id))
        gt.db.execute(
            "UPDATE goal_progress_history SET recorded_at = ? "
            "WHERE goal_id = ?", (now - days_ago * 86400, goal.goal_id))
        return goal

    def test_due_by_cadence_window(self):
        gt = _tracker()
        now = time.time()
        # Finance is weekly (7d): stale at 10d, fresh at 3d.
        self._stale_goal(gt, "Save for a plot", "finance", 10, now)
        self._stale_goal(gt, "Save for a bike", "finance", 3, now)
        # Interests is daily-brief (1d): stale at 2d.
        self._stale_goal(gt, "Watch AI news", "interests", 2, now)
        due = _run(gt.class_checkins_due(UID, now=now))
        by_id = {d["class_id"]: d for d in due}
        self.assertIn("finance", by_id)
        self.assertIn("interests", by_id)
        self.assertEqual(by_id["finance"]["cadence_days"], 7)
        self.assertEqual(by_id["interests"]["cadence_days"], 1)
        self.assertEqual(by_id["finance"]["stale_goals"], 1)
        self.assertEqual(by_id["finance"]["goals"], 2)
        # Soonest-lapsed first: finance (10d) before interests (2d).
        self.assertEqual([d["class_id"] for d in due],
                         ["finance", "interests"])

    def test_fresh_and_empty_never_due(self):
        gt = _tracker()
        self.assertEqual(_run(gt.class_checkins_due(UID)), [])
        _run(gt.create(UID, "Save for a plot", goal_class="finance"))
        self.assertEqual(_run(gt.class_checkins_due(UID)), [])

    def test_checkin_clears_due_and_moves_progress(self):
        gt = _tracker()
        now = time.time()
        goal = self._stale_goal(gt, "Save for a plot", "finance", 10, now)
        self.assertEqual(len(_run(gt.class_checkins_due(UID, now=now))), 1)
        entry = _run(gt.check_in(goal.goal_id, note="paid in",
                                 progress=25.0))
        self.assertEqual(entry.note, "paid in")
        self.assertEqual(entry.progress, 25.0)
        self.assertEqual(_run(gt.class_checkins_due(UID, now=now)), [])
        self.assertEqual((_run(gt.get(goal.goal_id))).progress, 25.0)

    def test_checkin_without_progress_keeps_number(self):
        gt = _tracker()
        goal = _run(gt.create(UID, "Save for a plot", goal_class="finance"))
        _run(gt.update_progress(goal.goal_id, 30.0))
        entry = _run(gt.check_in(goal.goal_id, note="still going"))
        self.assertEqual(entry.progress, 30.0)
        self.assertEqual((_run(gt.get(goal.goal_id))).progress, 30.0)


class SurfaceTests(unittest.TestCase):
    def test_board_renders_custom_section(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening", "Growing things.")
        _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        board = surface.render_class_board(gt, UID)
        self.assertIn("Gardening", board)
        self.assertIn("Grow tomatoes", board)
        self.assertIn("`gardening`", board)
        # A class with no goals gets a designed empty state, not a blank.
        self.assertIn("nothing here yet", board)

    def test_board_only_custom_class(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening", "Growing things.")
        _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        board = surface.render_class_board(gt, UID, only_class="gardening")
        self.assertIn("Grow tomatoes", board)
        self.assertNotIn("Finance", board)

    def test_render_classes_merged_with_tracker(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening", "Growing things.")
        text = surface.render_classes(gt)
        self.assertIn("`gardening` — Gardening", text)
        self.assertIn("/goals classes add <id> <title>", text)
        # Without a tracker the legacy built-in-only picker is unchanged.
        legacy = surface.render_classes()
        self.assertNotIn("gardening", legacy)
        self.assertIn("`finance` — Finance", legacy)

    def test_briefing_lines_and_snapshot_include_custom(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening", "Growing things.")
        _run(gt.create(UID, "Grow tomatoes", goal_class="gardening"))
        lines = surface.class_briefing_lines(gt, UID)
        self.assertTrue(any("Gardening" in line for line in lines))
        snap = surface.goals_context_snapshot(gt, UID)
        self.assertTrue(any(e["goal_class"] == "gardening" for e in snap))
        self.assertTrue(all(e["basis"] == "self-reported"
                            for e in snap if e["goal_class"] == "gardening"))

    def test_checkin_goal_and_checkins_due_renders(self):
        gt = _tracker()
        now = time.time()
        goal = _run(gt.create(UID, "Save for a plot", goal_class="finance"))
        gt.db.execute("UPDATE goals SET created_at = ? WHERE goal_id = ?",
                      (now - 10 * 86400, goal.goal_id))
        gt.db.execute(
            "UPDATE goal_progress_history SET recorded_at = ? "
            "WHERE goal_id = ?", (now - 10 * 86400, goal.goal_id))
        due_text = surface.render_checkins_due(gt, UID, now=now)
        self.assertIn("Finance", due_text)
        self.assertIn("checkin", due_text)
        text = surface.check_in_goal(gt, UID, "plot", progress=20.0,
                                     note="first transfer")
        self.assertIn("checked in on", text)
        self.assertIn("20% (self-reported)", text)
        self.assertIn("first transfer", text)
        self.assertIn("no goal class is due a check-in",
                      surface.render_checkins_due(gt, UID, now=now))

    def test_render_milestones(self):
        gt = _tracker()
        goal, _cid, _s = surface.create_goal_filed(
            gt, UID, "Apply for backend roles", "careers")
        text = surface.render_milestones(gt, UID, goal.goal_id)
        self.assertIn("Apply for backend roles", text)
        self.assertIn("Submit the first application", text)
        self.assertIn("⬜", text)
        self.assertIn("no single goal matches",
                      surface.render_milestones(gt, UID, "nope-not-a-goal"))

    def test_refile_into_custom_class(self):
        gt = _tracker()
        gt.define_class("gardening", "Gardening")
        _run(gt.create(UID, "Grow tomatoes", goal_class="finance"))
        text = surface.refile_goal(gt, UID, "tomatoes", "gardening")
        self.assertIn("Gardening", text)
        with self.assertRaises(GoalClassError):
            surface.refile_goal(gt, UID, "tomatoes", "noscope")


class ToolTests(unittest.TestCase):
    def _reg(self):
        db = _db()
        return _registry(db), db

    def test_create_lists_progress_checkin_complete(self):
        reg, _db = self._reg()
        created = _tool(reg, action="create", title="Apply for backend roles",
                        goal_class="careers")
        self.assertTrue(created["ok"])
        self.assertEqual(created["goal_class"], "careers")
        self.assertEqual(created["milestones"],
                         ["Submit the first application",
                          "Reach the interview stage", "Receive an offer"])
        goal_id = created["goal"]["goal_id"]
        board = _tool(reg, action="board")
        self.assertIn("Apply for backend roles", board["board"])
        self.assertEqual(board["stats"]["total"], 1)
        progress = _tool(reg, action="progress", ref=goal_id)
        self.assertEqual(progress["reading"]["basis"], "self-reported")
        self.assertEqual(progress["reading"]["percent"], 0.0)
        checked = _tool(reg, action="checkin", ref="backend",
                        progress="40", note="applied twice")
        self.assertIn("40% (self-reported)", checked["text"])
        done = _tool(reg, action="complete", ref=goal_id)
        self.assertEqual(done["goal"]["status"], "completed")
        self.assertEqual(done["goal"]["progress"], 100.0)

    def test_suggested_filing_from_title(self):
        reg, _db = self._reg()
        created = _tool(reg, action="create",
                        title="Save for a house deposit")
        self.assertEqual(created["goal_class"], "finance")
        self.assertTrue(created["suggested"])

    def test_define_remove_class_via_tool(self):
        reg, _db = self._reg()
        made = _tool(reg, action="define_class", class_id="gardening",
                     title="Gardening",
                     milestone_templates="First bed planted, First harvest")
        self.assertTrue(made["ok"])
        self.assertEqual(made["class"]["milestone_templates"],
                         ["First bed planted", "First harvest"])
        classes = _tool(reg, action="classes")
        self.assertIn("gardening", [c["id"] for c in classes["classes"]])
        created = _tool(reg, action="create", title="Grow tomatoes",
                        goal_class="gardening")
        self.assertEqual(created["milestones"],
                         ["First bed planted", "First harvest"])
        blocked = _tool(reg, action="remove_class", class_id="gardening")
        self.assertFalse(blocked["ok"])
        self.assertIn("in use", blocked["error"])
        _tool(reg, action="complete", ref=created["goal"]["goal_id"])
        # Completing does not free the class — the goal still files there.
        still = _tool(reg, action="remove_class", class_id="gardening")
        self.assertFalse(still["ok"])

    def test_remove_unused_class_via_tool(self):
        reg, _db = self._reg()
        _tool(reg, action="define_class", class_id="gardening",
              title="Gardening")
        removed = _tool(reg, action="remove_class", class_id="gardening")
        self.assertTrue(removed["ok"])

    def test_tool_errors_are_honest(self):
        reg, _db = self._reg()
        self.assertFalse(_tool(reg, action="create", title="")["ok"])
        self.assertFalse(_tool(reg, action="progress", ref="nope")["ok"])
        self.assertFalse(
            _tool(reg, action="define_class", class_id="finance",
                  title="Dup")["ok"])
        unknown = reg.call("goal_class", action="teleport")
        self.assertTrue(getattr(unknown, "ok", False))
        self.assertFalse(unknown.value["ok"])
        self.assertIn("unknown goal_class action", unknown.value["error"])

    def test_tool_without_db_is_honest(self):
        reg = _registry(None)
        outcome = reg.call("goal_class", action="board")
        self.assertTrue(getattr(outcome, "ok", False))
        self.assertFalse(outcome.value["ok"])
        self.assertIn("no database", outcome.value["error"])

    def test_checkins_via_tool(self):
        reg, db = self._reg()
        gt = GoalTracker(db)
        now = time.time()
        goal = _run(gt.create(UID, "Save for a plot", goal_class="finance"))
        db.execute("UPDATE goals SET created_at = ? WHERE goal_id = ?",
                   (now - 10 * 86400, goal.goal_id))
        db.execute("UPDATE goal_progress_history SET recorded_at = ? "
                   "WHERE goal_id = ?", (now - 10 * 86400, goal.goal_id))
        due = _tool(reg, action="checkins")
        self.assertEqual([d["class_id"] for d in due["due"]], ["finance"])
        self.assertIn("Finance", due["text"])


class ChatTests(unittest.TestCase):
    def _rt(self):
        db = _db()
        return _Rt(db), db

    def test_classes_add_then_add_goal_then_checkin(self):
        rt, _db = self._rt()
        reply = _text(rt._control_goals("classes add gardening Gardening"))
        self.assertIn("Gardening", reply)
        self.assertIn("`gardening`", reply)
        reply = _text(rt._control_goals("add Grow tomatoes in gardening"))
        self.assertIn("Gardening", reply)
        self.assertIn("goal id:", reply)
        board = _text(rt._control_goals("list gardening"))
        self.assertIn("Grow tomatoes", board)
        self.assertNotIn("Finance", board)
        reply = _text(rt._control_goals("checkin tomatoes 40 first sowing"))
        self.assertIn("40% (self-reported)", reply)
        self.assertIn("first sowing", reply)
        reply = _text(rt._control_goals("milestones tomatoes"))
        self.assertIn("Grow tomatoes", reply)

    def test_checkin_number_above_100_stays_in_ref(self):
        rt, _db = self._rt()
        _text(rt._control_goals("add Save 1000 in finance"))
        reply = _text(rt._control_goals("checkin Save 1000"))
        self.assertIn("checked in on", reply)
        self.assertIn("Save 1000", reply)

    def test_classes_remove_refused_while_in_use(self):
        rt, _db = self._rt()
        _text(rt._control_goals("classes add gardening Gardening"))
        _text(rt._control_goals("add Grow tomatoes in gardening"))
        reply = _text(rt._control_goals("classes remove gardening"))
        self.assertIn("in use", reply)
        reply = _text(rt._control_goals("classes remove noscope"))
        self.assertIn("unknown goal class", reply)
        reply = _text(rt._control_goals("classes add finance Dup"))
        self.assertIn("built in", reply)

    def test_checkins_verb_and_usage(self):
        rt, _db = self._rt()
        self.assertIn("no goal class is due",
                      _text(rt._control_goals("checkins")))
        self.assertIn("usage: /goals checkin",
                      _text(rt._control_goals("checkin")))
        self.assertIn("usage: /goals milestones",
                      _text(rt._control_goals("milestones")))
        self.assertIn("checkins",
                      _text(rt._control_goals("bogus-verb")))


class CliTests(unittest.TestCase):
    def test_classes_add_checkin_checkins_milestones_flow(self):
        ctx = _cli_ctx()
        code, out = _cli(_cli_ns(action="classes", arg="add", arg2="gardening",
                                 title="Gardening"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("`gardening`", out)
        code, out = _cli(_cli_ns(action="add", arg="Grow tomatoes",
                                 goal_class="gardening"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("[class: gardening]", out)
        code, out = _cli(_cli_ns(action="checkin", arg="tomatoes",
                                 progress="35", note="sowed"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("35% (self-reported)", out)
        code, out = _cli(_cli_ns(action="milestones", arg="tomatoes"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("Grow tomatoes", out)
        code, out = _cli(_cli_ns(action="checkins"), ctx)
        self.assertEqual(code, 0)
        self.assertIn("no goal class is due", out)
        code, out = _cli(_cli_ns(action="classes", arg="remove",
                                 arg2="gardening"), ctx)
        self.assertEqual(code, 1)
        self.assertIn("in use", out)

    def test_checkin_bad_progress_rejected(self):
        ctx = _cli_ctx()
        _cli(_cli_ns(action="add", arg="Save for a plot",
                     goal_class="finance"), ctx)
        code, out = _cli(_cli_ns(action="checkin", arg="plot",
                                 progress="abc"), ctx)
        self.assertEqual(code, 1)
        code, out = _cli(_cli_ns(action="checkin", arg="plot",
                                 progress="150"), ctx)
        self.assertEqual(code, 1)
        code, out = _cli(_cli_ns(action="checkin", arg=""), ctx)
        self.assertEqual(code, 1)
        self.assertIn("usage", out)

    def test_classes_add_needs_title(self):
        code, out = _cli(_cli_ns(action="classes", arg="add",
                                 arg2="gardening"), _cli_ctx())
        self.assertEqual(code, 1)
        self.assertIn("usage", out)


if __name__ == "__main__":
    unittest.main()
