"""Sweep tests for nomorals/goals upgrades.

Covers: OKR key results (add/update/remove/score/roll-up), goal journal,
goal links/dependencies, milestones, Taskwarrior-style focus scoring,
Beeminder-style pace reports, check-ins, Loop-style momentum, longest streak,
stale goals, GTD weekly review, goal search, custom templates, export,
extended stats, subgoal rename/reorder, recurring reminders + snooze,
ICE idea scoring, similar-idea search, idea review queue, idea stats,
goal kinds.
"""
from __future__ import annotations

import time
import unittest

from nomorals.goals import (
    FOCUS_COEFFICIENTS,
    GOAL_KINDS,
    GoalTracker,
    IdeaTracker,
)
from nomorals.storage.db import Database

UID = "sweep-user"


def _db() -> Database:
    return Database(":memory:")


def _hist(db: Database, goal_id: str, progress: float, days_ago: float, note: str = "t"):
    """Insert a backdated progress-history row directly."""
    db.execute(
        """INSERT INTO goal_progress_history
               (history_id, goal_id, progress, note, recorded_at)
           VALUES (?, ?, ?, ?, ?)""",
        (f"h-{goal_id}-{days_ago}-{progress}", goal_id, progress, note,
         time.time() - days_ago * 86400),
    )


class KeyResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_update_remove_round_trip(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Ship it")
        kr = await gt.add_key_result(goal.goal_id, "Signups", 1000.0,
                                    baseline=100.0, unit="users", weight=2.0)
        self.assertEqual(kr.title, "Signups")
        self.assertEqual(kr.direction, "increase")
        self.assertTrue(kr.committed)
        self.assertAlmostEqual(kr.score, 0.0)

        kr = await gt.update_key_result(kr.key_result_id, 550.0)
        self.assertAlmostEqual(kr.score, 0.5)  # (550-100)/(1000-100)

        krs = await gt.list_key_results(goal.goal_id)
        self.assertEqual(len(krs), 1)
        self.assertTrue(await gt.remove_key_result(kr.key_result_id))
        self.assertEqual(await gt.list_key_results(goal.goal_id), [])
        self.assertFalse(await gt.remove_key_result("kr-nope"))

    async def test_decrease_direction(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Costs")
        kr = await gt.add_key_result(goal.goal_id, "Burn", 500.0, baseline=1000.0,
                                    direction="decrease")
        kr = await gt.update_key_result(kr.key_result_id, 750.0)
        self.assertAlmostEqual(kr.score, 0.5)  # (1000-750)/(1000-500)

    async def test_score_clamps(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Clamp")
        kr = await gt.add_key_result(goal.goal_id, "X", 10.0)
        kr = await gt.update_key_result(kr.key_result_id, 999.0)
        self.assertEqual(kr.score, 1.0)
        kr = await gt.update_key_result(kr.key_result_id, -5.0)
        self.assertEqual(kr.score, 0.0)

    async def test_okr_score_weighted(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Weighted")
        a = await gt.add_key_result(goal.goal_id, "A", 100.0, weight=3.0)
        b = await gt.add_key_result(goal.goal_id, "B", 100.0, weight=1.0)
        await gt.update_key_result(a.key_result_id, 100.0)  # score 1.0
        await gt.update_key_result(b.key_result_id, 0.0)    # score 0.0
        self.assertAlmostEqual(await gt.okr_score(goal.goal_id), 0.75)
        self.assertEqual(await gt.okr_score("goal-nope"), 0.0)

    async def test_progress_rolls_up_from_key_results(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Rollup")
        sg = await gt.add_subgoal(goal.goal_id, "step one")
        await gt.complete_subgoal(sg.subgoal_id)  # would be 100% via subgoals
        self.assertEqual((await gt.get(goal.goal_id)).progress, 100.0)
        kr = await gt.add_key_result(goal.goal_id, "KR", 100.0)
        # KRs take precedence over subgoals once present.
        self.assertEqual((await gt.get(goal.goal_id)).progress, 0.0)
        await gt.update_key_result(kr.key_result_id, 40.0)
        self.assertAlmostEqual((await gt.get(goal.goal_id)).progress, 40.0)

    async def test_validation(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "V")
        with self.assertRaises(ValueError):
            await gt.add_key_result(goal.goal_id, "   ", 10.0)
        with self.assertRaises(ValueError):
            await gt.add_key_result(goal.goal_id, "X", 10.0, direction="sideways")
        with self.assertRaises(ValueError):
            await gt.add_key_result(goal.goal_id, "X", 10.0, weight=0.0)
        with self.assertRaises(KeyError):
            await gt.add_key_result("goal-nope", "X", 10.0)
        with self.assertRaises(KeyError):
            await gt.update_key_result("kr-nope", 5.0)


class JournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_and_list(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Journal goal")
        e1 = await gt.add_journal_entry(goal.goal_id, "first note")
        e2 = await gt.add_journal_entry(goal.goal_id, "second note")
        entries = await gt.journal(goal.goal_id)
        self.assertEqual([e.text for e in entries], ["second note", "first note"])
        self.assertEqual(e1.user_id, UID)
        with self.assertRaises(ValueError):
            await gt.add_journal_entry(goal.goal_id, "  ")
        with self.assertRaises(KeyError):
            await gt.add_journal_entry("goal-nope", "x")


class LinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_link_unlink(self):
        gt = GoalTracker(_db())
        a = await gt.create(UID, "A")
        b = await gt.create(UID, "B")
        link = await gt.link_goals(a.goal_id, b.goal_id, "blocks")
        self.assertEqual(link.kind, "blocks")
        # Idempotent: linking twice returns the same link.
        again = await gt.link_goals(a.goal_id, b.goal_id, "blocks")
        self.assertEqual(again.link_id, link.link_id)
        self.assertEqual(len(await gt.goal_links(a.goal_id)), 1)
        self.assertEqual(len(await gt.goal_links(b.goal_id)), 1)

        blocking = await gt.blocking_goals(a.goal_id)
        self.assertEqual([g.goal_id for g in blocking], [b.goal_id])
        blocked_by = await gt.blocked_by_goals(b.goal_id)
        self.assertEqual([g.goal_id for g in blocked_by], [a.goal_id])

        self.assertTrue(await gt.unlink_goals(a.goal_id, b.goal_id, "blocks"))
        self.assertFalse(await gt.unlink_goals(a.goal_id, b.goal_id, "blocks"))
        self.assertEqual(await gt.blocked_by_goals(b.goal_id), [])

    async def test_link_validation(self):
        gt = GoalTracker(_db())
        a = await gt.create(UID, "A")
        with self.assertRaises(ValueError):
            await gt.link_goals(a.goal_id, a.goal_id, "blocks")
        with self.assertRaises(ValueError):
            await gt.link_goals(a.goal_id, "goal-nope", "warps")
        with self.assertRaises(KeyError):
            await gt.link_goals(a.goal_id, "goal-nope", "blocks")

    async def test_blocked_goal_loses_focus(self):
        gt = GoalTracker(_db())
        now = time.time()
        blocker = await gt.create(UID, "Blocker")
        blocked = await gt.create(UID, "Blocked")
        free = await gt.create(UID, "Free")
        # Same shape otherwise; only the link differs.
        before = gt.focus_score(blocked, now=now)
        await gt.link_goals(blocker.goal_id, blocked.goal_id, "blocks")
        after = gt.focus_score(blocked, now=now)
        self.assertAlmostEqual(before - after, -FOCUS_COEFFICIENTS["blocked"])
        self.assertEqual(gt.focus_breakdown(blocked, now=now)["blocked"],
                         FOCUS_COEFFICIENTS["blocked"])
        self.assertGreater(gt.focus_score(free, now=now), after)


class MilestoneTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Miles")
        m = await gt.add_milestone(goal.goal_id, "Alpha", target_date=time.time() + 86400)
        self.assertFalse(m.is_completed)
        self.assertTrue(await gt.complete_milestone(m.milestone_id))
        self.assertTrue((await gt.list_milestones(goal.goal_id))[0].is_completed)
        self.assertTrue(await gt.reopen_milestone(m.milestone_id))
        self.assertFalse((await gt.list_milestones(goal.goal_id))[0].is_completed)
        self.assertTrue(await gt.remove_milestone(m.milestone_id))
        self.assertEqual(await gt.list_milestones(goal.goal_id), [])
        self.assertFalse(await gt.complete_milestone("gm-nope"))
        with self.assertRaises(ValueError):
            await gt.add_milestone(goal.goal_id, "  ")


class FocusTests(unittest.IsolatedAsyncioTestCase):
    async def test_next_tag_dominates(self):
        gt = GoalTracker(_db())
        now = time.time()
        nxt = await gt.create(UID, "Next thing", tags=["next"])
        soon = await gt.create(UID, "Due soon", target_date=now + 3 * 86400, priority=3)
        self.assertGreaterEqual(gt.focus_score(nxt, now=now), FOCUS_COEFFICIENTS["next"])
        top = await gt.today_focus(UID, limit=2, now=now)
        self.assertEqual(top[0]["goal"].goal_id, nxt.goal_id)
        self.assertIn("total", top[0]["breakdown"])
        self.assertIn("next", top[0]["breakdown"])
        self.assertIn("due", top[1]["breakdown"])

    async def test_overdue_beats_future(self):
        gt = GoalTracker(_db())
        now = time.time()
        late = await gt.create(UID, "Late", target_date=now - 10 * 86400)
        future = await gt.create(UID, "Future", target_date=now + 60 * 86400)
        # ≥7d overdue pins the due term at its max (12.0 * 1.0).
        self.assertEqual(gt.focus_breakdown(late, now=now)["due"], 12.0)
        self.assertGreater(gt.focus_score(late, now=now),
                           gt.focus_score(future, now=now))

    async def test_priority_and_paused_terms(self):
        gt = GoalTracker(_db())
        now = time.time()
        hi = await gt.create(UID, "Hi", priority=5)
        lo = await gt.create(UID, "Lo", priority=1)
        b_hi = gt.focus_breakdown(hi, now=now)
        b_lo = gt.focus_breakdown(lo, now=now)
        self.assertEqual(b_hi["priority"], FOCUS_COEFFICIENTS["priority_high"])
        self.assertEqual(b_lo["priority"], FOCUS_COEFFICIENTS["priority_low"])
        await gt.pause(hi.goal_id)
        paused = await gt.get(hi.goal_id)
        assert paused is not None
        self.assertEqual(gt.focus_breakdown(paused, now=now)["waiting"],
                         FOCUS_COEFFICIENTS["waiting"])


class PaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_track(self):
        db = _db()
        gt = GoalTracker(db)
        now = time.time()
        goal = await gt.create(UID, "Pace", target_date=now + 10 * 86400)
        _hist(db, goal.goal_id, 0.0, 5.0)
        await gt.update_progress(goal.goal_id, 60.0)  # 12%/day actual
        report = await gt.pace_report(goal.goal_id, now=now)
        self.assertEqual(report["verdict"], "on_track")
        self.assertGreater(report["actual_daily_pace"], 11.0)
        self.assertAlmostEqual(report["required_daily_pace"], 4.0, places=1)
        self.assertGreater(report["safe_days"], 5.0)
        self.assertIsNotNone(report["projected_completion_date"])

    async def test_flatline_is_off_track(self):
        db = _db()
        gt = GoalTracker(db)
        now = time.time()
        goal = await gt.create(UID, "Flat", target_date=now + 10 * 86400)
        _hist(db, goal.goal_id, 50.0, 5.0)
        await gt.update_progress(goal.goal_id, 40.0)  # regressing
        report = await gt.pace_report(goal.goal_id, now=now)
        self.assertEqual(report["verdict"], "off_track")

    async def test_verdicts(self):
        gt = GoalTracker(_db())
        now = time.time()
        overdue = await gt.create(UID, "Over", target_date=now - 86400)
        self.assertEqual((await gt.pace_report(overdue.goal_id, now=now))["verdict"],
                         "overdue")
        plain = await gt.create(UID, "Plain")
        self.assertEqual((await gt.pace_report(plain.goal_id, now=now))["verdict"],
                         "no_target")
        fresh = await gt.create(UID, "Fresh", target_date=now + 10 * 86400)
        # Only the "goal created" entry: not enough history for a pace.
        self.assertEqual((await gt.pace_report(fresh.goal_id, now=now))["verdict"],
                         "no_data")
        done = await gt.create(UID, "Done")
        await gt.complete(done.goal_id)
        self.assertEqual((await gt.pace_report(done.goal_id, now=now))["verdict"],
                         "complete")
        with self.assertRaises(KeyError):
            await gt.pace_report("goal-nope")

    async def test_at_risk(self):
        db = _db()
        gt = GoalTracker(db)
        now = time.time()
        goal = await gt.create(UID, "Risky", target_date=now + 10 * 86400)
        _hist(db, goal.goal_id, 0.0, 9.0)
        await gt.update_progress(goal.goal_id, 45.0)  # ~5%/day, needs ~5.5%/day
        report = await gt.pace_report(goal.goal_id, now=now)
        self.assertEqual(report["verdict"], "at_risk")


class CheckInTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_in_keeps_number(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "CI")
        await gt.update_progress(goal.goal_id, 30.0, note="work")
        entry = await gt.check_in(goal.goal_id, note="still alive")
        self.assertEqual(entry.progress, 30.0)
        self.assertEqual(entry.note, "still alive")
        self.assertEqual((await gt.get(goal.goal_id)).progress, 30.0)
        # Two activity days in a row feed the streak.
        self.assertGreaterEqual(await gt.progress_streak(UID), 1)

    async def test_check_in_with_progress(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "CI2")
        await gt.check_in(goal.goal_id, progress=42.0, note="bump")
        self.assertEqual((await gt.get(goal.goal_id)).progress, 42.0)
        with self.assertRaises(KeyError):
            await gt.check_in("goal-nope")


class MomentumTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_user_zero(self):
        gt = GoalTracker(_db())
        self.assertEqual(await gt.momentum("nobody"), 0.0)
        self.assertEqual(await gt.longest_streak("nobody"), 0)

    async def test_perfect_run_near_one(self):
        db = _db()
        gt = GoalTracker(db)
        goal = await gt.create(UID, "Habit")
        for d in range(60):
            _hist(db, goal.goal_id, 10.0, float(d))
        self.assertGreater(await gt.momentum(UID, days=60), 0.9)
        self.assertEqual(await gt.longest_streak(UID), 60)

    async def test_gap_resets_longest_but_not_momentum(self):
        db = _db()
        gt = GoalTracker(db)
        goal = await gt.create(UID, "Gappy")
        for d in [0, 1, 2, 10, 11, 12]:
            _hist(db, goal.goal_id, 10.0, float(d))
        self.assertEqual(await gt.longest_streak(UID), 3)
        self.assertGreater(await gt.momentum(UID, days=30), 0.0)


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_goals(self):
        db = _db()
        gt = GoalTracker(db)
        now = time.time()
        quiet = await gt.create(UID, "Quiet")
        loud = await gt.create(UID, "Loud")
        # Backdate all of quiet's history (incl. the creation entry).
        db.execute("UPDATE goal_progress_history SET recorded_at = ? WHERE goal_id = ?",
                   (now - 10 * 86400, quiet.goal_id))
        stale = await gt.stale_goals(UID, days=7, now=now)
        self.assertEqual([g.goal_id for g in stale], [quiet.goal_id])
        self.assertNotIn(loud.goal_id, [g.goal_id for g in stale])

    async def test_weekly_review_shape(self):
        db = _db()
        gt = GoalTracker(db)
        ideas = IdeaTracker(db)
        now = time.time()
        done = await gt.create(UID, "Done this week")
        await gt.complete(done.goal_id)
        late = await gt.create(UID, "Late one", target_date=now - 2 * 86400)
        soon = await gt.create(UID, "Soon one", target_date=now + 3 * 86400)
        blocker = await gt.create(UID, "Blocker")
        stuck = await gt.create(UID, "Stuck")
        await gt.link_goals(blocker.goal_id, stuck.goal_id, "blocks")
        idea = await ideas.create(UID, "Someday app")
        review = await gt.weekly_review(UID, now=now)
        self.assertEqual([g.goal_id for g in review["completed_this_week"]],
                         [done.goal_id])
        self.assertEqual([g.goal_id for g in review["overdue"]], [late.goal_id])
        self.assertEqual([g.goal_id for g in review["due_soon"]], [soon.goal_id])
        self.assertEqual(len(review["blocked"]), 1)
        self.assertEqual(review["blocked"][0]["goal"].goal_id, stuck.goal_id)
        self.assertEqual(review["ideas_for_review"][0][0].idea_id, idea.idea_id)
        self.assertTrue(review["suggested_actions"])

        text = gt.format_weekly_review(review)
        for needle in ("Weekly Review", "Late one", "Stuck", "Someday app",
                       "Suggested actions"):
            self.assertIn(needle, text)


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_goals(self):
        gt = GoalTracker(_db())
        await gt.create(UID, "Learn Spanish", description="daily practice")
        await gt.create(UID, "Run marathon", notes="buy shoes")
        hits = await gt.search_goals(UID, "spanish")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Learn Spanish")
        hits = await gt.search_goals(UID, "shoes")
        self.assertEqual(hits[0].title, "Run marathon")
        self.assertEqual(await gt.search_goals(UID, "  "), [])
        self.assertEqual(await gt.search_goals(UID, "zzz-no-match"), [])


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_list_delete(self):
        gt = GoalTracker(_db())
        t = await gt.save_template(UID, "Morning Pages", "Write {subject}",
                                   description="d", priority=2,
                                   tags=["writing"], subgoals=["open doc", "write"])
        self.assertEqual(t.name, "morning-pages")
        names = [x.name for x in await gt.list_custom_templates(UID)]
        self.assertIn("morning-pages", names)
        # Upsert: saving again replaces.
        await gt.save_template(UID, "Morning Pages", "Write!", subgoals=["a", "b", "c"])
        t2 = (await gt.list_custom_templates(UID))[0]
        self.assertEqual(t2.title, "Write!")
        self.assertEqual(len(t2.subgoals), 3)
        self.assertTrue(await gt.delete_template(UID, "Morning Pages"))
        self.assertFalse(await gt.delete_template(UID, "Morning Pages"))
        with self.assertRaises(ValueError):
            await gt.save_template(UID, "  ", "X")

    async def test_create_from_custom_template(self):
        gt = GoalTracker(_db())
        await gt.save_template(UID, "Deep Work", "Deep work: {subject}",
                               subgoals=["clear desk", "90 min block"])
        goal = await gt.create_from_template(UID, "deep-work", subject="writing")
        self.assertEqual(goal.title, "Deep work: writing")
        self.assertEqual(len((await gt.get(goal.goal_id)).subgoals), 2)

    async def test_builtin_still_works(self):
        gt = GoalTracker(_db())
        goal = await gt.create_from_template(UID, "fitness", subject="running")
        self.assertIn("running", goal.title)
        with self.assertRaises(KeyError):
            await gt.create_from_template(UID, "nope-template")


class ExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_json(self):
        db = _db()
        gt = GoalTracker(db)
        ideas = IdeaTracker(db)
        goal = await gt.create(UID, "Export me", tags=["x"])
        await gt.add_subgoal(goal.goal_id, "step")
        await gt.add_key_result(goal.goal_id, "KR", 10.0)
        await gt.add_milestone(goal.goal_id, "M1")
        await gt.add_journal_entry(goal.goal_id, "note")
        await gt.add_reminder(goal.goal_id, time.time() + 60, "nudge")
        await ideas.create(UID, "An idea")
        dump = await gt.export_json(UID)
        self.assertEqual(dump["user_id"], UID)
        self.assertEqual(len(dump["goals"]), 1)
        g = dump["goals"][0]
        for key in ("key_results", "milestones", "journal", "links",
                    "reminders", "history", "subgoals"):
            self.assertIn(key, g)
        self.assertEqual(len(g["subgoals"]), 1)
        self.assertEqual(len(dump["ideas"]), 1)

    async def test_export_markdown(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "MD goal", description="desc")
        await gt.add_subgoal(goal.goal_id, "step")
        text = await gt.export_markdown(UID)
        self.assertIn("# Goals export", text)
        self.assertIn("MD goal", text)
        self.assertIn("[ ] step", text)


class StatsTests(unittest.IsolatedAsyncioTestCase):
    async def test_extended_stats(self):
        gt = GoalTracker(_db())
        a = await gt.create(UID, "A")
        await gt.create(UID, "B")
        await gt.complete(a.goal_id)
        stats = await gt.stats(UID)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["completion_rate"], 0.5)
        self.assertIsNotNone(stats["avg_completion_days"])
        self.assertGreaterEqual(stats["longest_streak"], 1)
        self.assertGreaterEqual(stats["momentum"], 0.0)


class SubgoalMgmtTests(unittest.IsolatedAsyncioTestCase):
    async def test_rename_and_reorder(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "Order")
        s1 = await gt.add_subgoal(goal.goal_id, "one")
        s2 = await gt.add_subgoal(goal.goal_id, "two")
        s3 = await gt.add_subgoal(goal.goal_id, "three")
        self.assertTrue(await gt.rename_subgoal(s2.subgoal_id, "TWO"))
        self.assertFalse(await gt.rename_subgoal("sg-nope", "x"))
        with self.assertRaises(ValueError):
            await gt.rename_subgoal(s1.subgoal_id, " ")
        self.assertTrue(await gt.reorder_subgoals(
            goal.goal_id, [s3.subgoal_id, s1.subgoal_id, s2.subgoal_id]))
        titles = [s.title for s in (await gt.get(goal.goal_id)).subgoals]
        self.assertEqual(titles, ["three", "one", "TWO"])
        with self.assertRaises(ValueError):
            await gt.reorder_subgoals(goal.goal_id, [s1.subgoal_id])
        self.assertFalse(await gt.reorder_subgoals("goal-nope", []))


class ReminderRecurrenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_daily_ack_reschedules(self):
        gt = GoalTracker(_db())
        now = time.time()
        goal = await gt.create(UID, "R")
        r = await gt.add_reminder(goal.goal_id, now - 10, "daily nudge",
                                 recurrence="daily")
        self.assertEqual(r.recurrence, "daily")
        self.assertTrue(await gt.ack_reminder(r.reminder_id, now=now))
        live = await gt.list_reminders(UID)
        self.assertEqual(len(live), 1)  # still alive, not acknowledged away
        self.assertGreater(live[0].remind_at, now)
        self.assertLess(live[0].remind_at, now + 86400 + 1)

    async def test_weekly_ack_reschedules(self):
        gt = GoalTracker(_db())
        now = time.time()
        goal = await gt.create(UID, "R")
        r = await gt.add_reminder(goal.goal_id, now - 10 * 86400, "weekly",
                                 recurrence="weekly")
        await gt.ack_reminder(r.reminder_id, now=now)
        live = await gt.list_reminders(UID)
        self.assertGreater(live[0].remind_at, now)
        self.assertLess(live[0].remind_at, now + 7 * 86400 + 1)

    async def test_snooze(self):
        gt = GoalTracker(_db())
        now = time.time()
        goal = await gt.create(UID, "R")
        r = await gt.add_reminder(goal.goal_id, now - 60, "snoozable")
        self.assertTrue(await gt.snooze_reminder(r.reminder_id, 30, now=now))
        live = await gt.list_reminders(UID)
        self.assertAlmostEqual(live[0].remind_at, now + 1800, delta=2)
        self.assertFalse(await gt.snooze_reminder("rem-nope", 5))
        with self.assertRaises(ValueError):
            await gt.snooze_reminder(r.reminder_id, 0)

    async def test_bad_recurrence(self):
        gt = GoalTracker(_db())
        goal = await gt.create(UID, "R")
        with self.assertRaises(ValueError):
            await gt.add_reminder(goal.goal_id, time.time(), "x", recurrence="minutely")


class IdeaScoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_score_and_ice(self):
        it = IdeaTracker(_db())
        idea = await it.create(UID, "App idea")
        self.assertIsNone(idea.ice_score)
        scored = await it.score_idea(idea.idea_id, impact=8, confidence=9, ease=7)
        self.assertEqual(scored.ice_score, 8.0)  # (8+9+7)/3
        self.assertEqual(scored.to_dict()["ice_score"], 8.0)
        with self.assertRaises(ValueError):
            await it.score_idea(idea.idea_id, impact=11, confidence=5, ease=5)
        with self.assertRaises(KeyError):
            await it.score_idea("idea-nope", impact=5, confidence=5, ease=5)
        cleared = await it.clear_idea_score(idea.idea_id)
        self.assertIsNone(cleared.ice_score)

    async def test_top_ideas_ranking(self):
        it = IdeaTracker(_db())
        low = await it.create(UID, "Low")
        high = await it.create(UID, "High")
        unscored = await it.create(UID, "Unscored")
        await it.score_idea(low.idea_id, impact=3, confidence=3, ease=3)
        await it.score_idea(high.idea_id, impact=9, confidence=9, ease=9)
        top = await it.top_ideas(UID)
        self.assertEqual([i.idea_id for i in top],
                         [high.idea_id, low.idea_id, unscored.idea_id])

    async def test_find_similar_ideas(self):
        it = IdeaTracker(_db())
        a = await it.create(UID, "Build a mobile game",
                            description="puzzle game for phones")
        b = await it.create(UID, "Build a mobile game app",
                            description="puzzle game for mobile phones")
        c = await it.create(UID, "Bake sourdough bread",
                            description="weekend baking project")
        sims = await it.find_similar_ideas(UID, a.idea_id)
        ids = [i.idea_id for i, _ in sims]
        self.assertIn(b.idea_id, ids)
        self.assertNotIn(c.idea_id, ids)
        self.assertNotIn(a.idea_id, ids)
        # Raw-text probe works too.
        sims2 = await it.find_similar_ideas(UID, "mobile puzzle game")
        self.assertTrue(any(i.idea_id == a.idea_id for i, _ in sims2))
        self.assertEqual(await it.find_similar_ideas(UID, "   "), [])

    async def test_review_queue_and_stats(self):
        it = IdeaTracker(_db())
        first = await it.create(UID, "First")
        await it.create(UID, "Second")
        queue = await it.review_queue(UID)
        self.assertEqual(queue[0][0].idea_id, first.idea_id)  # oldest first
        self.assertGreaterEqual(queue[0][1], 0)
        await it.dismiss(first.idea_id, reason="nah")
        stats = await it.idea_stats(UID)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_status"]["active"], 1)
        self.assertEqual(stats["by_status"]["dismissed"], 1)
        await it.score_idea(queue[1][0].idea_id, impact=6, confidence=6, ease=6)
        stats = await it.idea_stats(UID)
        self.assertEqual(stats["scored"], 1)
        self.assertEqual(stats["avg_ice"], 6.0)


class KindTests(unittest.IsolatedAsyncioTestCase):
    async def test_kind_lifecycle(self):
        gt = GoalTracker(_db())
        habit = await gt.create(UID, "Meditate", kind="habit")
        self.assertEqual(habit.kind, "habit")
        self.assertIn("habit", GOAL_KINDS)
        updated = await gt.update(habit.goal_id, kind="project")
        self.assertEqual(updated.kind, "project")
        with self.assertRaises(ValueError):
            await gt.create(UID, "X", kind="bogus")
        with self.assertRaises(ValueError):
            await gt.update(habit.goal_id, kind="bogus")
        # to_dict carries it.
        self.assertEqual((await gt.get(habit.goal_id)).to_dict()["kind"], "project")


class BriefingUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_briefing_has_focus_and_pace(self):
        gt = GoalTracker(_db())
        now = time.time()
        await gt.create(UID, "Next big thing", tags=["next"])
        await gt.create(UID, "Slipping", target_date=now - 86400)
        text = await gt.get_briefing(UID)
        self.assertIn("Focus now", text)
        self.assertIn("Next big thing", text)
        self.assertIn("Pace check", text)
        self.assertIn("Slipping", text)


if __name__ == "__main__":
    unittest.main()
