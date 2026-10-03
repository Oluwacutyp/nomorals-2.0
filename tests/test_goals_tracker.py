"""Goals + ideas trackers: full coverage of nomorals/goals.

Covers GoalTracker (create/get/update/status lifecycle/archive/delete,
subgoals, progress + history + streaks, templates, deadlines/reminders,
listing filters, stats, briefings) and IdeaTracker (create/get/update,
dismiss/delete, promote_to_goal, list filters, search).
"""
from __future__ import annotations

import time
import unittest

from nomorals.goals import (
    GOAL_TEMPLATES,
    GoalTracker,
    IdeaTracker,
)
from nomorals.storage.db import Database

UID = "test-user"


def _db() -> Database:
    return Database(":memory:")


class GoalCreateTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_and_get_round_trip(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Learn Python", description="py",
                               priority=2, tags=["code"], workspace="w1",
                               notes="n")
        fetched = await gt.get(goal.goal_id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.title, "Learn Python")
        self.assertEqual(fetched.description, "py")
        self.assertEqual(fetched.priority, 2)
        self.assertEqual(fetched.tags, ["code"])
        self.assertEqual(fetched.workspace, "w1")
        self.assertEqual(fetched.notes, "n")
        self.assertEqual(fetched.status, "active")
        self.assertEqual(fetched.progress, 0.0)

    async def test_create_rejects_empty_title(self):
        gt = GoalTracker(_db())
        with self.assertRaises(ValueError):
            await gt.create(UID, "   ")

    async def test_get_missing_returns_none(self):
        gt = GoalTracker(_db())
        self.assertIsNone(await gt.get("goal-nope"))


class GoalUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_fields(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Old")
        updated = await gt.update(goal.goal_id, title="New", description="d",
                                  priority=5, tags=["a"], notes="nn",
                                  workspace="w2")
        self.assertEqual(updated.title, "New")
        self.assertEqual(updated.description, "d")
        self.assertEqual(updated.priority, 5)
        self.assertEqual(updated.tags, ["a"])
        self.assertEqual(updated.notes, "nn")
        self.assertEqual(updated.workspace, "w2")

    async def test_update_target_date_and_clear(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        future = time.time() + 86400
        updated = await gt.update(goal.goal_id, target_date=future)
        self.assertEqual(updated.target_date, future)
        updated = await gt.update(goal.goal_id, clear_target_date=True)
        self.assertIsNone(updated.target_date)

    async def test_update_missing_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.update("goal-nope", title="x")

    async def test_update_rejects_empty_title(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        with self.assertRaises(ValueError):
            await gt.update(goal.goal_id, title="  ")


class GoalLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_resume_abandon_archive(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        self.assertEqual((await gt.pause(goal.goal_id)).status, "paused")
        self.assertEqual((await gt.resume(goal.goal_id)).status, "active")
        self.assertEqual((await gt.abandon(goal.goal_id)).status, "abandoned")
        archived = await gt.archive(goal.goal_id)
        self.assertEqual(archived.status, "archived")
        # archived goals are hidden from the default active listing
        self.assertEqual(await gt.list_goals(UID, status="active"), [])
        self.assertEqual((await gt.unarchive(goal.goal_id)).status, "active")

    async def test_unarchive_non_archived_raises(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        with self.assertRaises(ValueError):
            await gt.unarchive(goal.goal_id)

    async def test_set_status_rejects_unknown(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        with self.assertRaises(ValueError):
            await gt.set_status(goal.goal_id, "flying")

    async def test_set_status_missing_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.set_status("goal-nope", "paused")

    async def test_delete_cascades(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        await gt.add_subgoal(goal.goal_id, "s1")
        await gt.update_progress(goal.goal_id, 10.0)
        await gt.add_reminder(goal.goal_id, time.time() + 60)
        self.assertTrue(await gt.delete(goal.goal_id))
        self.assertIsNone(await gt.get(goal.goal_id))
        self.assertEqual(await gt.progress_history(goal.goal_id), [])
        self.assertEqual(await gt.list_reminders(UID), [])
        self.assertFalse(await gt.delete(goal.goal_id))


class SubgoalTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_complete_reopen_remove(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        s1 = await gt.add_subgoal(goal.goal_id, "one")
        s2 = await gt.add_subgoal(goal.goal_id, "two")
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(len(fetched.subgoals), 2)
        self.assertEqual(fetched.progress, 0.0)

        self.assertTrue(await gt.complete_subgoal(s1.subgoal_id))
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(fetched.progress, 50.0)

        self.assertTrue(await gt.reopen_subgoal(s1.subgoal_id))
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(fetched.progress, 0.0)

        # re-opening an already-open subgoal is a no-op success
        self.assertTrue(await gt.reopen_subgoal(s1.subgoal_id))

        self.assertTrue(await gt.remove_subgoal(s2.subgoal_id))
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(len(fetched.subgoals), 1)

    async def test_subgoal_missing_ids(self):
        gt = GoalTracker(_db())
        self.assertFalse(await gt.complete_subgoal("sub-nope"))
        self.assertFalse(await gt.reopen_subgoal("sub-nope"))
        self.assertFalse(await gt.remove_subgoal("sub-nope"))

    async def test_add_subgoal_missing_goal_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.add_subgoal("goal-nope", "x")
        with self.assertRaises(ValueError):
            goal = await gt.create(UID, "T")
            await gt.add_subgoal(goal.goal_id, "   ")


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_progress_clamps_and_records(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        await gt.update_progress(goal.goal_id, 150.0, note="over")
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(fetched.progress, 100.0)
        await gt.update_progress(goal.goal_id, -5.0)
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(fetched.progress, 0.0)

        history = await gt.progress_history(goal.goal_id)
        notes = [h.note for h in history]
        self.assertIn("over", notes)
        # newest first
        self.assertGreaterEqual(history[0].recorded_at, history[-1].recorded_at)

    async def test_update_progress_missing_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.update_progress("goal-nope", 10.0)

    async def test_complete_cascades_subgoals(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        s = await gt.add_subgoal(goal.goal_id, "s1")
        await gt.complete(goal.goal_id)
        fetched = await gt.get(goal.goal_id)
        assert fetched is not None
        self.assertEqual(fetched.status, "completed")
        self.assertEqual(fetched.progress, 100.0)
        self.assertTrue(fetched.subgoals[0].is_completed)
        self.assertTrue(await gt.complete_subgoal(s.subgoal_id))  # idempotent

    async def test_complete_missing_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.complete("goal-nope")

    async def test_streak(self):
        gt = GoalTracker(_db())
        self.assertEqual(await gt.progress_streak(UID), 0)
        goal = await gt.create(UID, "T")
        await gt.update_progress(goal.goal_id, 10.0)
        self.assertEqual(await gt.progress_streak(UID), 1)


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_templates_exist(self):
        self.assertGreaterEqual(len(GOAL_TEMPLATES), 4)
        for name, tpl in GOAL_TEMPLATES.items():
            self.assertIn("subgoals", tpl, name)
            self.assertTrue(tpl["subgoals"], name)

    async def test_create_from_template(self):
        gt = GoalTracker(_db())
        goal = await gt.create_from_template(UID, "learn-language",
                                             subject="Spanish")
        self.assertIn("Spanish", goal.title)
        self.assertEqual(len(goal.subgoals), 5)
        self.assertEqual(goal.status, "active")

    async def test_create_from_unknown_template_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.create_from_template(UID, "nope")

    async def test_create_from_template_title_override(self):
        gt = GoalTracker(_db())
        goal = await gt.create_from_template(UID, "fitness",
                                             title="My custom title",
                                             subject="x")
        self.assertEqual(goal.title, "My custom title")


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_due_goals_buckets(self):
        gt = GoalTracker(_db())
        now = time.time()
        overdue = await gt.create(UID, "overdue", target_date=now - 86400)
        soon = await gt.create(UID, "soon", target_date=now + 2 * 86400)
        far = await gt.create(UID, "far", target_date=now + 60 * 86400)
        buckets = await gt.due_goals(UID, within_days=7)
        self.assertEqual([g.goal_id for g in buckets["overdue"]], [overdue.goal_id])
        self.assertEqual([g.goal_id for g in buckets["due_soon"]], [soon.goal_id])
        self.assertNotIn(far.goal_id,
                         [g.goal_id for g in buckets["overdue"] + buckets["due_soon"]])

    async def test_due_within_days_filter(self):
        gt = GoalTracker(_db())
        now = time.time()
        await gt.create(UID, "soon", target_date=now + 86400)
        await gt.create(UID, "far", target_date=now + 60 * 86400)
        rows = await gt.list_goals(UID, due_within_days=7)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "soon")

    async def test_briefing_sections(self):
        gt = GoalTracker(_db())
        now = time.time()
        goal = await gt.create(UID, "Brief me", target_date=now - 3600)
        await gt.update_progress(goal.goal_id, 30.0)
        await gt.add_reminder(goal.goal_id, now - 10, "check in")
        text = await gt.get_briefing(UID)
        self.assertIn("Brief me", text)
        self.assertIn("30%", text)
        self.assertIn("Overdue", text)
        self.assertIn("Reminders due", text)
        self.assertIn("check in", text)

    async def test_briefing_no_goals(self):
        gt = GoalTracker(_db())
        self.assertEqual(await gt.get_briefing(UID), "No active goals.")


class ReminderTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_list_due_ack(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "T")
        now = time.time()
        r1 = await gt.add_reminder(goal.goal_id, now - 5, "past")
        r2 = await gt.add_reminder(goal.goal_id, now + 3600, "future")
        due = await gt.due_reminders(UID, now=now)
        self.assertEqual([r.reminder_id for r in due], [r1.reminder_id])
        self.assertTrue(await gt.ack_reminder(r1.reminder_id))
        due = await gt.due_reminders(UID, now=now)
        self.assertEqual(due, [])
        # acknowledged reminders are hidden by default, visible on request
        self.assertEqual(len(await gt.list_reminders(UID)), 1)
        self.assertEqual(len(await gt.list_reminders(UID, include_acknowledged=True)), 2)
        self.assertEqual(r2.message, "future")

    async def test_reminder_missing_goal_raises(self):
        gt = GoalTracker(_db())
        with self.assertRaises(KeyError):
            await gt.add_reminder("goal-nope", time.time())

    async def test_ack_missing_returns_false(self):
        gt = GoalTracker(_db())
        self.assertFalse(await gt.ack_reminder("rem-nope"))


class ListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_filters(self):
        gt = GoalTracker(_db())
        await gt.create(UID, "code goal", tags=["code"], workspace="dev")
        await gt.create(UID, "home goal", tags=["home"], workspace="home")
        rows = await gt.list_goals(UID, tags=["CODE"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "code goal")
        rows = await gt.list_goals(UID, workspace="home")
        self.assertEqual(len(rows), 1)
        rows = await gt.list_goals(UID, limit=1)
        self.assertEqual(len(rows), 1)

    async def test_stats(self):
        gt = GoalTracker(_db())
        await gt.create(UID, "a")
        b = await gt.create(UID, "b")
        await gt.complete(b.goal_id)
        stats = await gt.stats(UID)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_status"]["active"], 1)
        self.assertEqual(stats["by_status"]["completed"], 1)
        self.assertEqual(stats["by_status"]["archived"], 0)


class IdeaTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_get_update(self):
        it = IdeaTracker(_db())
        idea = await it.create(UID, "Game idea", description="rpg",
                               tags=["fun"], source="chat")
        fetched = await it.get(idea.idea_id)
        assert fetched is not None
        self.assertEqual(fetched.title, "Game idea")
        self.assertEqual(fetched.tags, ["fun"])
        self.assertEqual(fetched.source, "chat")
        self.assertEqual(fetched.status, "active")

        updated = await it.update(idea.idea_id, title="Better game",
                                  tags=["fun", "rpg"])
        self.assertEqual(updated.title, "Better game")
        self.assertEqual(updated.tags, ["fun", "rpg"])

    async def test_create_rejects_empty_title(self):
        it = IdeaTracker(_db())
        with self.assertRaises(ValueError):
            await it.create(UID, "")

    async def test_get_update_missing(self):
        it = IdeaTracker(_db())
        self.assertIsNone(await it.get("idea-nope"))
        with self.assertRaises(KeyError):
            await it.update("idea-nope", title="x")

    async def test_dismiss(self):
        it = IdeaTracker(_db())
        idea = await it.create(UID, "Meh")
        self.assertTrue(await it.dismiss(idea.idea_id, reason="not now"))
        fetched = await it.get(idea.idea_id)
        assert fetched is not None
        self.assertEqual(fetched.status, "dismissed")
        self.assertEqual(fetched.dismiss_reason, "not now")
        self.assertIsNotNone(fetched.dismissed_at)
        self.assertFalse(await it.dismiss("idea-nope"))

    async def test_delete(self):
        it = IdeaTracker(_db())
        idea = await it.create(UID, "Temp")
        self.assertTrue(await it.delete(idea.idea_id))
        self.assertIsNone(await it.get(idea.idea_id))
        self.assertFalse(await it.delete(idea.idea_id))

    async def test_promote_to_goal(self):
        db = _db()
        it = IdeaTracker(db)
        gt = GoalTracker(db)
        idea = await it.create(UID, "Build it", description="desc", tags=["x"])
        goal_id = await it.promote_to_goal(idea.idea_id, gt)
        self.assertIsNotNone(goal_id)
        assert goal_id is not None
        goal = await gt.get(goal_id)
        assert goal is not None
        self.assertEqual(goal.title, "Build it")
        self.assertEqual(goal.tags, ["x"])
        fetched = await it.get(idea.idea_id)
        assert fetched is not None
        self.assertEqual(fetched.status, "promoted_to_goal")
        self.assertEqual(fetched.metadata.get("promoted_goal_id"), goal_id)

    async def test_promote_missing_returns_none(self):
        db = _db()
        it = IdeaTracker(db)
        gt = GoalTracker(db)
        self.assertIsNone(await it.promote_to_goal("idea-nope", gt))

    async def test_list_filters_and_search(self):
        it = IdeaTracker(_db())
        await it.create(UID, "Space game", tags=["gaming"])
        await it.create(UID, "Cook pasta", tags=["food"])
        rows = await it.list_ideas(UID, tags=["GAMING"])
        self.assertEqual(len(rows), 1)
        rows = await it.list_ideas(UID, limit=1)
        self.assertEqual(len(rows), 1)
        rows = await it.search(UID, "pasta")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "Cook pasta")
        self.assertEqual(await it.search(UID, "zzz-no-match"), [])
        self.assertEqual(await it.search(UID, "   "), [])


if __name__ == "__main__":
    unittest.main()
