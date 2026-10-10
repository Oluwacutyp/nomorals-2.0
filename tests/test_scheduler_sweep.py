"""Scheduler sweep tests: mined-then-built upgrades.

Covers the recurrence upgrades (human descriptions, exdate/rdate,
between/next_n, natural-language parsing) and the scheduler upgrades
(jitter, pause-on-failure, skip dates, nag reminders, quiet hours, hook
cooldown, condition combinators, unified task management, preview/describe,
run_now, prune_terminal, health upcoming/stale).
"""

from __future__ import annotations

import asyncio
import time
import unittest
from datetime import datetime, timedelta

from nomorals.scheduler.recurrence import (
    describe_cron,
    describe_rrule,
    parse_natural_datetime,
    parse_natural_schedule,
    parse_rrule_set,
    prev_cron,
)
from nomorals.scheduler.scheduler import (
    Scheduler,
    _fmt_next,
    _quiet_shift,
    evaluate_conditions,
)
from nomorals.storage.db import Database


def _make_scheduler():
    db = Database(":memory:")
    sched = Scheduler(db)
    calls: list[dict] = []

    async def ok_handler(**kwargs):
        calls.append(kwargs)

    async def boom_handler(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("boom")

    sched.register_action("ok", ok_handler)
    sched.register_action("boom", boom_handler)
    return sched, db, calls


def _cron_row(sched, db, task_id):
    return dict(db.query_one(
        """SELECT c.*, t.action, t.parameters, t.metadata, t.task_type,
                  t.status, t.created_at FROM cron_jobs c
           JOIN scheduled_tasks t ON c.task_id = t.task_id
           WHERE c.task_id = ?""", (task_id,)))


def _reminder_row(sched, db, task_id):
    return dict(db.query_one(
        """SELECT r.*, t.action, t.parameters, t.metadata, t.task_type,
                  t.status FROM reminders r
           JOIN scheduled_tasks t ON r.task_id = t.task_id
           WHERE r.task_id = ?""", (task_id,)))


# ── recurrence: human descriptions ──────────────────────────────────────────

class TestDescribeCron(unittest.TestCase):
    def test_common_shapes(self):
        self.assertEqual(describe_cron("* * * * *"), "Every minute")
        self.assertEqual(describe_cron("*/5 * * * *"), "Every 5 minutes")
        self.assertEqual(describe_cron("0 * * * *"), "Every hour")
        self.assertEqual(describe_cron("0 */2 * * *"), "Every 2 hours")
        self.assertEqual(describe_cron("30 11 * * *"), "At 11:30 AM")

    def test_weekday_range(self):
        self.assertEqual(
            describe_cron("30 11 * * 1-5"),
            "At 11:30 AM, on Monday through Friday")

    def test_extended_fields(self):
        self.assertEqual(
            describe_cron("0 9 L * *"),
            "At 9:00 AM, on the last day of the month")
        self.assertEqual(
            describe_cron("0 9 LW * *"),
            "At 9:00 AM, on the last weekday of the month")
        self.assertEqual(
            describe_cron("0 9 15W * *"),
            "At 9:00 AM, on the weekday nearest day 15")
        self.assertEqual(
            describe_cron("0 9 * * 5#3"),
            "At 9:00 AM, on the 3rd Friday of the month")
        self.assertEqual(
            describe_cron("0 9 * * 5L"),
            "At 9:00 AM, on the last Friday of the month")

    def test_month_restriction(self):
        self.assertEqual(
            describe_cron("23 12 15 3 *"),
            "At 12:23 PM, on day 15th of the month, only in March")

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            describe_cron("not a cron")


class TestDescribeRrule(unittest.TestCase):
    def test_shapes(self):
        self.assertTrue(
            describe_rrule("FREQ=DAILY").startswith("Every day at "))
        self.assertTrue(
            describe_rrule("FREQ=DAILY;INTERVAL=3").startswith("Every 3 days"))
        self.assertIn("Monday", describe_rrule("FREQ=WEEKLY;BYDAY=MO"))
        self.assertTrue(
            describe_rrule("FREQ=MONTHLY;BYDAY=2TU").startswith(
                "Every month on the 2nd Tuesday"))
        self.assertIn("last Friday",
                      describe_rrule("FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1"))
        self.assertIn("5 times", describe_rrule("FREQ=DAILY;COUNT=5"))
        self.assertIn("until", describe_rrule("FREQ=DAILY;UNTIL=20270101"))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            describe_rrule("FREQ=NEVER")


class TestPrevCron(unittest.TestCase):
    def test_prev(self):
        before = datetime(2026, 10, 10, 12, 30)
        prev = datetime.fromtimestamp(prev_cron("0 9 * * *", before))
        self.assertEqual((prev.hour, prev.minute), (9, 0))
        self.assertEqual(prev.date(), before.date())

    def test_prev_strict(self):
        # exactly on a firing minute → the one before it
        before = datetime(2026, 10, 10, 9, 0)
        prev = datetime.fromtimestamp(prev_cron("0 9 * * *", before))
        self.assertEqual(prev.date(), datetime(2026, 10, 9).date())


# ── recurrence: exdate/rdate, between, next_n ───────────────────────────────

class TestRruleSet(unittest.TestCase):
    def test_exdate_skips(self):
        rule = parse_rrule_set(
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=WEEKLY;BYDAY=MO\n"
            "EXDATE:20260112T090000")
        days = [d.strftime("%m-%d")
                for d in rule.next_n(3, datetime(2026, 1, 5))]
        self.assertEqual(days, ["01-05", "01-19", "01-26"])

    def test_rdate_merges(self):
        rule = parse_rrule_set(
            "DTSTART:20260105T090000\n"
            "RRULE:FREQ=WEEKLY;BYDAY=MO\n"
            "RDATE:20260114T090000")
        days = [d.strftime("%m-%d")
                for d in rule.next_n(3, datetime(2026, 1, 5))]
        self.assertEqual(days, ["01-05", "01-12", "01-14"])

    def test_between(self):
        rule = parse_rrule_set(
            "DTSTART:20260105T090000\nRRULE:FREQ=WEEKLY;BYDAY=MO")
        days = [d.strftime("%m-%d")
                for d in rule.between(datetime(2026, 1, 6),
                                      datetime(2026, 2, 1))]
        self.assertEqual(days, ["01-12", "01-19", "01-26"])

    def test_next_n(self):
        rule = parse_rrule_set(
            "DTSTART:20260105T090000\nRRULE:FREQ=DAILY;COUNT=3")
        occ = rule.next_n(10, datetime(2026, 1, 1))
        self.assertEqual(len(occ), 3)

    def test_needs_rrule(self):
        with self.assertRaises(ValueError):
            parse_rrule_set("EXDATE:20260112T090000")


# ── recurrence: natural language ────────────────────────────────────────────

class TestParseNaturalSchedule(unittest.TestCase):
    def test_patterns(self):
        cases = {
            "every minute": ("cron", "* * * * *"),
            "every 15 minutes": ("cron", "*/15 * * * *"),
            "hourly": ("cron", "0 * * * *"),
            "every 2 hours": ("cron", "0 */2 * * *"),
            "daily at 9am": ("cron", "0 9 * * *"),
            "every day at 18:30": ("cron", "30 18 * * *"),
            "every weekday at 9am": ("cron", "0 9 * * MON-FRI"),
            "weekends at 10am": ("cron", "0 10 * * SAT,SUN"),
            "weekly on monday at 8am": ("cron", "0 8 * * 1"),
            "every friday at 5pm": ("cron", "0 17 * * 5"),
            "monthly on the 15th": ("cron", "0 9 15 * *"),
            "every 3 days": ("rrule", "FREQ=DAILY;INTERVAL=3"),
            "every 2nd tuesday": ("rrule", "FREQ=MONTHLY;BYDAY=2TU"),
            "last friday of the month": (
                "rrule", "FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1"),
            "yearly on jan 1": ("cron", "0 9 1 1 *"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_natural_schedule(text), expected)

    def test_garbage_raises_helpfully(self):
        with self.assertRaises(ValueError) as ctx:
            parse_natural_schedule("flurb the wizzle")
        self.assertIn("every weekday at 9am", str(ctx.exception))


class TestParseNaturalDatetime(unittest.TestCase):
    BASE = datetime(2026, 10, 10, 8, 0)  # a Saturday

    def p(self, text, **kw):
        return parse_natural_datetime(text, now=self.BASE, **kw)

    def test_relative(self):
        self.assertEqual(self.p("in 20 minutes"),
                         self.BASE + timedelta(minutes=20))
        self.assertEqual(self.p("in an hour"),
                         self.BASE + timedelta(hours=1))
        self.assertEqual(self.p("in 2 weeks"),
                         self.BASE + timedelta(weeks=2))

    def test_day_words(self):
        self.assertEqual(self.p("tomorrow at 8am"),
                         datetime(2026, 10, 11, 8, 0))
        self.assertEqual(self.p("tonight"), datetime(2026, 10, 10, 20, 0))
        self.assertEqual(self.p("noon"), datetime(2026, 10, 10, 12, 0))
        self.assertEqual(self.p("midnight"), datetime(2026, 10, 11, 0, 0))

    def test_weekdays(self):
        # Saturday base: "wednesday" → Oct 14, "next monday" → Oct 19
        self.assertEqual(self.p("wednesday"), datetime(2026, 10, 14, 9, 0))
        self.assertEqual(self.p("next monday at 9"),
                         datetime(2026, 10, 19, 9, 0))
        self.assertEqual(self.p("this fri at 5pm"),
                         datetime(2026, 10, 16, 17, 0))

    def test_month_day(self):
        self.assertEqual(self.p("oct 15"), datetime(2026, 10, 15, 9, 0))
        self.assertEqual(self.p("15 october"), datetime(2026, 10, 15, 9, 0))
        # past date without year rolls to next year
        self.assertEqual(self.p("jan 5").year, 2027)

    def test_bare_time(self):
        self.assertEqual(self.p("9:30pm"), datetime(2026, 10, 10, 21, 30))
        # 7am already passed at 08:00 → tomorrow
        self.assertEqual(self.p("at 7"), datetime(2026, 10, 11, 7, 0))

    def test_iso(self):
        self.assertEqual(self.p("2026-12-25 14:00"),
                         datetime(2026, 12, 25, 14, 0))

    def test_tz_aware(self):
        dt = self.p("in 20 minutes", tz="Africa/Lagos")
        self.assertIsNotNone(dt.tzinfo)

    def test_garbage_raises_helpfully(self):
        with self.assertRaises(ValueError) as ctx:
            self.p("flurb the wizzle")
        self.assertIn("in 20 minutes", str(ctx.exception))


# ── scheduler: jitter / pause-on-failure / skip dates / idempotency ─────────

class TestScheduleCronUpgrades(unittest.TestCase):
    def test_duplicate_task_id_raises(self):
        sched, _db, _calls = _make_scheduler()
        asyncio.run(sched.schedule_cron("dup1", "* * * * *", "ok"))
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(sched.schedule_cron("dup1", "* * * * *", "ok"))
        self.assertIn("already exists", str(ctx.exception))

    def test_natural_language_cron(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(
            sched.schedule_cron(None, "every weekday at 9am", "ok"))
        self.assertEqual(job.cron_expr, "0 9 * * MON-FRI")

    def test_natural_language_rrule_hint(self):
        sched, _db, _calls = _make_scheduler()
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(sched.schedule_cron(None, "every 2nd tuesday", "ok"))
        self.assertIn("schedule_rrule", str(ctx.exception))

    def test_natural_language_rrule(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(
            sched.schedule_rrule(None, "every 2nd tuesday", "ok"))
        self.assertEqual(job.cron_expr, "FREQ=MONTHLY;BYDAY=2TU")

    def test_jitter_bounded(self):
        sched, _db, _calls = _make_scheduler()
        from nomorals.scheduler.scheduler import CronParser
        nominal = CronParser.next_run("* * * * *")
        job = asyncio.run(
            sched.schedule_cron(None, "* * * * *", "ok", jitter_s=3600))
        self.assertGreaterEqual(job.next_run, nominal)
        self.assertLessEqual(job.next_run, nominal + 3600)

    def test_no_jitter_by_default(self):
        sched, _db, _calls = _make_scheduler()
        from nomorals.scheduler.scheduler import CronParser
        nominal = CronParser.next_run("* * * * *")
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        self.assertAlmostEqual(job.next_run, nominal, delta=2)

    def test_pause_on_failure(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(
            None, "* * * * *", "boom", max_attempts=3,
            pause_on_failure=True))
        asyncio.run(sched._fire_cron(_cron_row(sched, _db, job.task_id),
                                     time.time()))
        task = asyncio.run(sched.get_task(job.task_id))
        self.assertEqual(task.status.value, "paused")
        # and it resumes
        self.assertTrue(asyncio.run(sched.resume(job.task_id)))

    def test_skip_dates(self):
        sched, _db, _calls = _make_scheduler()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        job = asyncio.run(sched.schedule_cron(
            None, "0 9 * * *", "ok", skip_dates=[tomorrow]))
        nxt_date = datetime.fromtimestamp(job.next_run).strftime("%Y-%m-%d")
        self.assertNotEqual(nxt_date, tomorrow)

    def test_add_remove_skip_dates(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "0 9 * * *", "ok"))
        day_after = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        job2 = asyncio.run(sched.add_skip_dates(job.task_id, [day_after]))
        self.assertIn(day_after, job2.metadata["skip_dates"])
        job3 = asyncio.run(sched.remove_skip_dates(job.task_id, [day_after]))
        self.assertNotIn(day_after, job3.metadata["skip_dates"])

    def test_bad_skip_date_raises(self):
        sched, _db, _calls = _make_scheduler()
        with self.assertRaises(ValueError):
            asyncio.run(sched.schedule_cron(
                None, "0 9 * * *", "ok", skip_dates=["yesterday-ish"]))


# ── scheduler: reminders (NL, nag, quiet hours) ─────────────────────────────

class TestReminderUpgrades(unittest.TestCase):
    def test_natural_language_due_at(self):
        sched, _db, _calls = _make_scheduler()
        before = time.time()
        rem = asyncio.run(
            sched.create_reminder("ping", "in 20 minutes", "u1"))
        self.assertGreaterEqual(rem.due_at, before + 19 * 60)
        self.assertLessEqual(rem.due_at, before + 21 * 60)

    def test_nag_mode(self):
        sched, _db, _calls = _make_scheduler()
        rem = asyncio.run(sched.create_reminder(
            "take meds", time.time() - 1, "u1", action="ok",
            nag_every_s=60, max_nags=2))
        # first firing → still pending, nag scheduled
        asyncio.run(sched._fire_reminder(
            _reminder_row(sched, _db, rem.task_id), time.time()))
        self.assertEqual(len(_calls), 1)
        task = asyncio.run(sched.get_task(rem.task_id))
        self.assertEqual(task.status.value, "pending")
        self.assertEqual(task.metadata["nag_count"], 1)
        # make it due again → second nag
        _db.execute("UPDATE reminders SET due_at = ? WHERE task_id = ?",
                    (time.time() - 1, rem.task_id))
        asyncio.run(sched._fire_reminder(
            _reminder_row(sched, _db, rem.task_id), time.time()))
        self.assertEqual(len(_calls), 2)
        # third firing → nags exhausted → completed
        _db.execute("UPDATE reminders SET due_at = ? WHERE task_id = ?",
                    (time.time() - 1, rem.task_id))
        asyncio.run(sched._fire_reminder(
            _reminder_row(sched, _db, rem.task_id), time.time()))
        self.assertEqual(len(_calls), 3)
        task = asyncio.run(sched.get_task(rem.task_id))
        self.assertEqual(task.status.value, "completed")

    def test_quiet_hours_shift(self):
        sched, _db, _calls = _make_scheduler()
        rem = asyncio.run(sched.create_reminder(
            "shh", time.time() - 1, "u1", quiet_hours=("00:00", "23:59")))
        fired = asyncio.run(sched._fire_reminder(
            _reminder_row(sched, _db, rem.task_id), time.time()))
        self.assertFalse(fired)
        self.assertEqual(len(_calls), 0)
        task = asyncio.run(sched.get_task(rem.task_id))
        self.assertEqual(task.status.value, "pending")
        # run history records the suppression
        runs = sched.recent_runs(rem.task_id)
        self.assertIn("quiet hours", runs[0]["result"])

    def test_quiet_shift_math(self):
        base = datetime(2026, 10, 10, 2, 0)  # 02:00 local
        ts = base.timestamp()
        shifted = _quiet_shift(ts, ("23:00", "07:00"), None)
        self.assertIsNotNone(shifted)
        self.assertEqual(
            datetime.fromtimestamp(shifted).strftime("%H:%M"), "07:00")
        # outside the window → None
        noon = datetime(2026, 10, 10, 12, 0).timestamp()
        self.assertIsNone(_quiet_shift(noon, ("23:00", "07:00"), None))
        # non-overnight window
        early = datetime(2026, 10, 10, 13, 30).timestamp()
        shifted2 = _quiet_shift(early, ("13:00", "14:00"), None)
        self.assertEqual(
            datetime.fromtimestamp(shifted2).strftime("%H:%M"), "14:00")

    def test_bad_quiet_hours_raises(self):
        sched, _db, _calls = _make_scheduler()
        with self.assertRaises(ValueError):
            asyncio.run(sched.create_reminder(
                "x", time.time() + 60, "u1", quiet_hours="whenever"))

    def test_schedule_once_natural_and_quiet(self):
        sched, _db, _calls = _make_scheduler()
        task = asyncio.run(sched.schedule_once(
            None, "in 30 minutes", "ok", quiet_hours="00:00-23:59"))
        self.assertGreater(task.metadata["run_at"], time.time())
        self.assertEqual(task.metadata["quiet_hours"], ["00:00", "23:59"])
        asyncio.run(sched.schedule_once("dupx", time.time() + 5, "ok"))
        with self.assertRaises(ValueError):
            asyncio.run(sched.schedule_once("dupx", time.time() + 5, "ok"))


# ── scheduler: hooks (cooldown, combinators) ────────────────────────────────

class TestHookUpgrades(unittest.TestCase):
    def test_cooldown_throttles(self):
        sched, _db, calls = _make_scheduler()
        events = []
        sched.add_listener("throttled", lambda p: events.append(p))
        hook = asyncio.run(sched.create_event_hook(
            "price_drop", {"price": {"lt": 100}}, "ok", cooldown_s=3600))
        first = asyncio.run(
            sched.trigger_event("price_drop", {"price": 50}))
        second = asyncio.run(
            sched.trigger_event("price_drop", {"price": 40}))
        self.assertEqual(first, [hook.task_id])
        self.assertEqual(second, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["task_id"], hook.task_id)

    def test_no_cooldown_by_default(self):
        sched, _db, calls = _make_scheduler()
        asyncio.run(sched.create_event_hook(
            "ping", {}, "ok"))
        asyncio.run(sched.trigger_event("ping", {}))
        asyncio.run(sched.trigger_event("ping", {}))
        self.assertEqual(len(calls), 2)

    def test_condition_combinators(self):
        data = {"kind": "email", "spam": False, "price": 50}
        self.assertTrue(evaluate_conditions(
            {"$or": [{"kind": "sms"}, {"price": {"lt": 100}}]}, data))
        self.assertFalse(evaluate_conditions(
            {"$or": [{"kind": "sms"}, {"price": {"gt": 100}}]}, data))
        self.assertTrue(evaluate_conditions(
            {"$and": [{"kind": "email"}, {"$not": {"spam": True}}]}, data))
        self.assertFalse(evaluate_conditions(
            {"$and": [{"kind": "email"}, {"spam": True}]}, data))
        self.assertTrue(evaluate_conditions({"$not": {"spam": True}}, data))
        self.assertFalse(evaluate_conditions({"$not": {"kind": "email"}},
                                             data))
        # malformed combinators never match
        self.assertFalse(evaluate_conditions({"$or": "nope"}, data))

    def test_hook_matches_with_combinators(self):
        sched, _db, calls = _make_scheduler()
        asyncio.run(sched.create_event_hook(
            "alert", {"$or": [{"kind": "email"}, {"kind": "sms"}]}, "ok"))
        hit = asyncio.run(sched.trigger_event("alert", {"kind": "sms"}))
        miss = asyncio.run(sched.trigger_event("alert", {"kind": "push"}))
        self.assertEqual(len(hit), 1)
        self.assertEqual(miss, [])


# ── scheduler: unified management ───────────────────────────────────────────

class TestTaskManagement(unittest.TestCase):
    def _seed(self, sched):
        cron = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        rem = asyncio.run(
            sched.create_reminder("r", time.time() + 600, "u1"))
        hook = asyncio.run(
            sched.create_event_hook("ev", {}, "ok"))
        once = asyncio.run(
            sched.schedule_once(None, time.time() + 600, "ok"))
        return cron, rem, hook, once

    def test_get_task(self):
        sched, _db, _calls = _make_scheduler()
        cron, rem, hook, once = self._seed(sched)
        self.assertEqual(
            (asyncio.run(sched.get_task(cron.task_id))).cron_expr, "* * * * *")
        self.assertEqual(
            (asyncio.run(sched.get_task(rem.task_id))).text, "r")
        self.assertEqual(
            (asyncio.run(sched.get_task(hook.task_id))).event_type, "ev")
        self.assertEqual(
            (asyncio.run(sched.get_task(once.task_id))).task_type, "one_time")
        self.assertIsNone(asyncio.run(sched.get_task("nope")))

    def test_list_tasks(self):
        sched, _db, _calls = _make_scheduler()
        self._seed(sched)
        self.assertEqual(len(asyncio.run(sched.list_tasks())), 4)
        self.assertEqual(
            len(asyncio.run(sched.list_tasks(task_type="cron"))), 1)
        self.assertEqual(
            len(asyncio.run(sched.list_tasks(task_type="reminder"))), 1)

    def test_update_task(self):
        sched, _db, _calls = _make_scheduler()
        cron, _rem, _hook, _once = self._seed(sched)
        self.assertTrue(asyncio.run(sched.update_task(
            cron.task_id, parameters={"x": 1})))
        task = asyncio.run(sched.get_task(cron.task_id))
        self.assertEqual(task.parameters, {"x": 1})
        self.assertFalse(asyncio.run(
            sched.update_task("nope", action="ok")))

    def test_delete_task(self):
        sched, _db, _calls = _make_scheduler()
        cron, _rem, _hook, _once = self._seed(sched)
        self.assertTrue(asyncio.run(sched.delete_task(cron.task_id)))
        self.assertIsNone(asyncio.run(sched.get_task(cron.task_id)))
        self.assertFalse(asyncio.run(sched.delete_task(cron.task_id)))

    def test_prune_terminal(self):
        sched, _db, _calls = _make_scheduler()
        rem = asyncio.run(
            sched.create_reminder("r", time.time() + 600, "u1"))
        asyncio.run(sched.complete_reminder(rem.task_id))
        _db.execute(
            "UPDATE scheduled_tasks SET updated_at = ? WHERE task_id = ?",
            (time.time() - 100, rem.task_id))
        removed = sched.prune_terminal(older_than_s=10)
        self.assertEqual(removed, 1)
        self.assertIsNone(asyncio.run(sched.get_task(rem.task_id)))
        # fresh terminal tasks are kept
        rem2 = asyncio.run(
            sched.create_reminder("r2", time.time() + 600, "u1"))
        asyncio.run(sched.complete_reminder(rem2.task_id))
        self.assertEqual(sched.prune_terminal(older_than_s=3600), 0)


# ── scheduler: preview / describe / run_now ─────────────────────────────────

class TestPreviewDescribeRunNow(unittest.TestCase):
    def test_preview_cron(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        nxt = sched.preview(job.task_id, 3)
        self.assertEqual(len(nxt), 3)
        self.assertLess(nxt[0], nxt[1])
        self.assertLess(nxt[1], nxt[2])
        self.assertAlmostEqual(nxt[1] - nxt[0], 60, delta=2)

    def test_preview_others(self):
        sched, _db, _calls = _make_scheduler()
        rem = asyncio.run(
            sched.create_reminder("r", time.time() + 600, "u1"))
        self.assertEqual(len(sched.preview(rem.task_id)), 1)
        hook = asyncio.run(sched.create_event_hook("ev", {}, "ok"))
        self.assertEqual(sched.preview(hook.task_id), [])
        self.assertEqual(sched.preview("nope"), [])

    def test_describe(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(
            None, "30 11 * * 1-5", "ok", tz="Africa/Lagos"))
        text = asyncio.run(sched.describe(job.task_id))
        self.assertIn("11:30 AM", text)
        self.assertIn("Monday through Friday", text)
        self.assertIn("Africa/Lagos", text)
        self.assertIn("next:", text)
        rem = asyncio.run(
            sched.create_reminder("call mom", time.time() + 600, "u1"))
        rtext = asyncio.run(sched.describe(rem.task_id))
        self.assertIn("call mom", rtext)
        self.assertIn("due", rtext)
        self.assertIsNone(asyncio.run(sched.describe("nope")))

    def test_run_now(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        before = job.next_run
        self.assertTrue(asyncio.run(sched.run_now(job.task_id)))
        self.assertEqual(len(_calls), 1)
        # schedule untouched
        job2 = asyncio.run(sched.get_cron(job.task_id))
        self.assertEqual(job2.next_run, before)
        runs = sched.recent_runs(job.task_id)
        self.assertEqual(runs[0]["trigger_source"], "manual")
        self.assertFalse(asyncio.run(sched.run_now("nope")))

    def test_run_now_failure(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "boom"))
        self.assertFalse(asyncio.run(sched.run_now(job.task_id)))
        runs = sched.recent_runs(job.task_id)
        self.assertFalse(runs[0]["ok"])


# ── scheduler: health upgrades ──────────────────────────────────────────────

class TestHealthUpgrades(unittest.TestCase):
    def test_upcoming_and_stale(self):
        sched, _db, _calls = _make_scheduler()
        job = asyncio.run(sched.schedule_cron(None, "* * * * *", "ok"))
        rem = asyncio.run(
            sched.create_reminder("r", time.time() + 600, "u1"))
        h = sched.health()
        kinds = {e["kind"] for e in h["upcoming"]}
        self.assertIn("cron", kinds)
        self.assertIn("reminder", kinds)
        self.assertTrue(all(e["in_s"] >= 0 for e in h["upcoming"]))
        # silence heuristic: last firing long ago, cadence short
        _db.execute(
            "UPDATE cron_jobs SET last_run = ?, next_run = ? WHERE task_id = ?",
            (time.time() - 10000, time.time() - 9900, job.task_id))
        h2 = sched.health()
        self.assertEqual(len(h2["stale"]), 1)
        self.assertEqual(h2["stale"][0]["task_id"], job.task_id)
        self.assertFalse(h2["ok"])

    def test_fmt_next(self):
        now = time.time()
        self.assertTrue(_fmt_next(now).startswith("today"))
        self.assertTrue(
            _fmt_next(now + 86400).startswith("tomorrow"))


if __name__ == "__main__":
    unittest.main()
