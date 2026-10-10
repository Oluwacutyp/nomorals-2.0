"""Section B (scheduler) verification: the new capabilities.

Covers what Section B added on top of the earlier upgrade suites:
- recurrence engine (extended cron, tz-aware next_run, RRULE)
- agents scheduler: real concurrent dispatch, rrule kind, blackout
  dates, start jitter, run-history retention
- user-facing scheduler: snooze fix, failed-cron backoff + dead-letter,
  missed-fire/staleness, tz, rrule, hook operators, overlap, timeout,
  listeners, pause/resume/reschedule, catch-up, health, history.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from nomorals.agents.scheduler import Scheduler as AgentScheduler
from nomorals.agents.scheduler import parse_schedule_spec
from nomorals.scheduler import Scheduler as UserScheduler
from nomorals.scheduler.recurrence import (
    CronSpec,
    RRule,
    cron_matches,
    next_cron,
)
from nomorals.scheduler.scheduler import (
    CONDITION_OPS,
    CronParser,
    evaluate_conditions,
)
from nomorals.storage.db import Database


def _agent_ctx(db):
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw: SimpleNamespace(
            ok=True, value="ok", error=None))
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(profile="workstation"),
                           extras={})


def _file_db():
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "sched.db"))
    db.migrate()
    return db


# ── recurrence engine ────────────────────────────────────────────────────

class RecurrenceCronTests(unittest.TestCase):
    def test_last_day_of_month(self):
        self.assertTrue(cron_matches("0 9 L * *", datetime(2026, 2, 28, 9, 0)))
        self.assertFalse(cron_matches("0 9 L * *", datetime(2026, 2, 27, 9, 0)))
        # leap year
        self.assertTrue(cron_matches("0 9 L * *", datetime(2024, 2, 29, 9, 0)))

    def test_last_weekday(self):
        # Oct 2026: 31st is Saturday → last weekday = Fri 30th
        self.assertTrue(cron_matches("0 9 LW * *", datetime(2026, 10, 30, 9, 0)))
        self.assertFalse(cron_matches("0 9 LW * *", datetime(2026, 10, 31, 9, 0)))

    def test_nearest_weekday(self):
        # Aug 15 2026 is Saturday → Friday 14th
        self.assertTrue(cron_matches("0 9 15W * *", datetime(2026, 8, 14, 9, 0)))
        self.assertFalse(cron_matches("0 9 15W * *", datetime(2026, 8, 15, 9, 0)))

    def test_nth_and_last_dow(self):
        self.assertTrue(cron_matches("0 9 * * 5#3", datetime(2026, 10, 16, 9, 0)))
        self.assertFalse(cron_matches("0 9 * * 5#3", datetime(2026, 10, 9, 9, 0)))
        self.assertTrue(cron_matches("0 9 * * 5L", datetime(2026, 10, 30, 9, 0)))

    def test_question_mark_and_names(self):
        self.assertTrue(cron_matches("0 9 ? * MON-FRI", datetime(2026, 10, 9, 9, 0)))
        self.assertFalse(cron_matches("0 9 ? * MON-FRI", datetime(2026, 10, 10, 9, 0)))

    def test_classic_still_works(self):
        self.assertTrue(cron_matches("*/15 * * * *", datetime(2026, 1, 1, 0, 30)))
        self.assertTrue(cron_matches("0 0 29 2 *", datetime(2024, 2, 29, 0, 0)))

    def test_next_cron_extended(self):
        nxt = next_cron("0 9 L * *", datetime(2026, 10, 9, 12, 0))
        self.assertEqual(datetime.fromtimestamp(nxt).strftime("%Y-%m-%d %H:%M"),
                         "2026-10-31 09:00")

    def test_next_cron_tz_aware(self):
        nxt = next_cron("0 9 * * *", datetime(2026, 10, 9, 12, 0),
                        "America/New_York")
        self.assertEqual(
            datetime.fromtimestamp(nxt, ZoneInfo("America/New_York"))
            .strftime("%H:%M"), "09:00")

    def test_parse_validates(self):
        with self.assertRaises(ValueError):
            CronSpec("0 9 * * 8")  # bad weekday
        with self.assertRaises(ValueError):
            CronSpec("not a cron")


class RRuleTests(unittest.TestCase):
    def test_second_tuesday(self):
        r = RRule.parse("FREQ=MONTHLY;BYDAY=2TU", datetime(2026, 1, 1, 9, 0))
        self.assertEqual(r.after(datetime(2026, 10, 9)),
                         datetime(2026, 10, 13, 9, 0))

    def test_last_friday_bysetpos(self):
        r = RRule.parse("FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1",
                        datetime(2026, 1, 1, 9, 0))
        self.assertEqual(r.after(datetime(2026, 10, 9)),
                         datetime(2026, 10, 30, 9, 0))

    def test_count_exhausts(self):
        r = RRule.parse("FREQ=WEEKLY;BYDAY=MO,WE,FR;COUNT=3",
                        datetime(2026, 10, 5, 9, 0))
        self.assertEqual(r.after(datetime(2026, 10, 6)),
                         datetime(2026, 10, 7, 9, 0))
        self.assertIsNone(r.after(datetime(2026, 10, 12)))

    def test_until(self):
        r = RRule.parse("FREQ=DAILY;UNTIL=20261010T000000Z",
                        datetime(2026, 10, 1, 9, 0))
        self.assertIsNone(r.after(datetime(2026, 10, 9, 10, 0)))

    def test_negative_monthday(self):
        r = RRule.parse("FREQ=MONTHLY;BYMONTHDAY=-1",
                        datetime(2026, 1, 31, 9, 0))
        self.assertEqual(r.after(datetime(2026, 10, 9)),
                         datetime(2026, 10, 31, 9, 0))

    def test_interval(self):
        r = RRule.parse("FREQ=DAILY;INTERVAL=2", datetime(2026, 10, 1, 8, 30))
        self.assertEqual(r.after(datetime(2026, 10, 2)),
                         datetime(2026, 10, 3, 8, 30))

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            RRule.parse("FREQ=NEVER", datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            RRule.parse("INTERVAL=2", datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            RRule.parse("FREQ=WEEKLY;BYDAY=2TU", datetime(2026, 1, 1))


# ── agents scheduler: new capabilities ───────────────────────────────────

class AgentConcurrencyTests(unittest.TestCase):
    def test_tick_dispatches_concurrently(self):
        db = _file_db()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               max_concurrent=4, missed_grace_s=30)
        events = []
        orig = AgentScheduler._execute

        def slow(self, row, **kw):
            events.append(("start", row["name"], threading.current_thread().name,
                           time.time()))
            time.sleep(1.2)
            return {"id": row["id"], "name": row["name"], "ok": True,
                    "result": "ok", "seconds": 1.2,
                    "next_run": time.time() + 3600, "will_retry": False,
                    "retry_count": 0, "timed_out": False,
                    "trigger_source": "tick"}
        AgentScheduler._execute = slow
        try:
            for name in ("jobA", "jobB"):
                sched.add(name, "every 3600s", "message", {"text": "hi"})
            db.execute("UPDATE schedule_jobs SET next_run = ?",
                       (time.time() - 1,))
            t0 = time.time()
            outcomes = sched.tick()
            total = time.time() - t0
        finally:
            AgentScheduler._execute = orig
        threads = {e[2] for e in events if e[0] == "start"}
        self.assertEqual(len(outcomes), 2)
        self.assertLess(total, 2.0, "jobs did not overlap — still serial")
        self.assertEqual(len(threads), 2)

    def test_max_concurrent_bounds(self):
        db = _file_db()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               max_concurrent=1, missed_grace_s=30)
        orig = AgentScheduler._execute
        AgentScheduler._execute = lambda self, row, **kw: (
            time.sleep(0.6), {"id": row["id"], "name": row["name"],
                              "ok": True, "result": "ok", "seconds": 0.6,
                              "next_run": time.time() + 3600,
                              "will_retry": False, "retry_count": 0,
                              "timed_out": False,
                              "trigger_source": "tick"})[1]
        try:
            for name in ("jobA", "jobB"):
                sched.add(name, "every 3600s", "message", {"text": "hi"})
            db.execute("UPDATE schedule_jobs SET next_run = ?",
                       (time.time() - 1,))
            t0 = time.time()
            sched.tick()
            total = time.time() - t0
        finally:
            AgentScheduler._execute = orig
        self.assertGreaterEqual(total, 1.1, "max_concurrent=1 not enforced")


class AgentRruleTests(unittest.TestCase):
    def test_rrule_kind_parses(self):
        kind, detail = parse_schedule_spec("rrule FREQ=WEEKLY;BYDAY=MO")
        self.assertEqual(kind, "rrule")
        kind, detail = parse_schedule_spec("FREQ=DAILY;INTERVAL=2")
        self.assertEqual(kind, "rrule")
        with self.assertRaises(ValueError):
            parse_schedule_spec("rrule FREQ=NEVER")

    def test_rrule_job_fires_and_advances(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               missed_grace_s=3600)
        job = sched.add("rrule-job", "rrule FREQ=MINUTELY",
                        "message", {"text": "rrule hi"})
        self.assertEqual(job["kind"], "rrule")
        first = job["next_run"]
        db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (time.time() - 1, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])
        row = sched._find(job["id"])
        # next firing is a real future occurrence of the series
        self.assertGreater(row["next_run"], time.time() - 1)

    def test_exhausted_rrule_retires(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               missed_grace_s=3600)
        job = sched.add("one-rrule", "rrule FREQ=MINUTELY;COUNT=1",
                        "message", {"text": "once"})
        # age the series anchor so its single occurrence already passed
        db.execute("UPDATE schedule_jobs SET created_at = ?, next_run = ? "
                   "WHERE id = ?",
                   (time.time() - 180, time.time() - 1, job["id"]))
        sched.tick()
        row = sched._find(job["id"])
        self.assertFalse(row["enabled"])


class AgentBlackoutTests(unittest.TestCase):
    def test_blackout_day_skips_without_running(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               missed_grace_s=3600)
        today = datetime.now().strftime("%Y-%m-%d")
        job = sched.add("blackout-job", "every 3600s",
                        "message", {"text": "should not fire"},
                        blackout_dates=[today])
        db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (time.time() - 1, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(outcomes, [])
        runs = sched.recent_runs(job["id"])
        self.assertTrue(any("blackout" in r["result"] for r in runs))
        row = sched._find(job["id"])
        self.assertGreater(row["next_run"], time.time())

    def test_set_blackout_dates(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db))
        job = sched.add("j", "every 3600s", "message", {"text": "x"})
        fmt = sched.set_blackout_dates(job["id"], ["2026-12-25"])
        self.assertEqual(fmt["blackout_dates"], ["2026-12-25"])
        with self.assertRaises(ValueError):
            sched.set_blackout_dates(job["id"], ["not-a-date"])

    def test_start_jitter_spreads_first_firing(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db))
        sched._rng = __import__("random").Random(7)
        before = time.time()
        job = sched.add("jitter-job", "every 3600s", "message",
                        {"text": "x"}, start_jitter_s=100)
        gap = job["next_run"] - before - 3600
        self.assertGreaterEqual(gap, 0.0)
        self.assertLessEqual(gap, 100.0)


class AgentRetentionTests(unittest.TestCase):
    def test_run_history_pruned(self):
        db = Database(":memory:")
        db.migrate()
        sched = AgentScheduler(_agent_ctx(db), tick_seconds=60,
                               missed_grace_s=3600, run_history_limit=3)
        job = sched.add("j", "every 3600s", "message", {"text": "x"})
        for _ in range(5):
            db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                       (time.time() - 1, job["id"]))
            sched.tick()
        rows = db.query("SELECT COUNT(*) AS n FROM schedule_runs WHERE job_id = ?",
                        (job["id"],))
        self.assertEqual(rows[0]["n"], 3)
        self.assertEqual(sched.prune_run_history(job["id"]), 0)


# ── user-facing scheduler ────────────────────────────────────────────────

def _user_sched():
    tmp = tempfile.mkdtemp()
    return UserScheduler(Database(os.path.join(tmp, "u.db")))


class UserSnoozeTests(unittest.TestCase):
    def test_snoozed_reminder_fires(self):
        s = _user_sched()
        fired = []
        s.register_action("send_reminder",
                          lambda **kw: fired.append(kw.get("text")))
        r = asyncio.run(s.create_reminder("take meds", time.time() - 5, "u1"))
        asyncio.run(s.snooze_reminder(r.task_id, minutes=0))
        s._tick()
        self.assertIn("take meds", fired)


class UserFailureTests(unittest.TestCase):
    def _boom_sched(self):
        s = _user_sched()
        async def boom(**kw):
            raise RuntimeError("kaboom")
        s.register_action("boom", boom)
        return s

    def test_failed_cron_backs_off_then_dead_letters(self):
        s = self._boom_sched()
        asyncio.run(s.schedule_cron("c1", "* * * * *", "boom", {},
                                    max_attempts=2, retry_base_s=5))
        s.db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = 'c1'",
                     (time.time() - 5,))
        s._tick()  # attempt 1: backoff, stays pending
        row = s.db.query_one(
            "SELECT next_run FROM cron_jobs WHERE task_id = 'c1'")
        self.assertGreater(row["next_run"], time.time())
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = 'c1'")["status"], "pending")
        s.db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = 'c1'",
                     (time.time() - 1,))
        s._tick()  # attempt 2: dead-letter
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = 'c1'")["status"], "dead")
        runs = s.recent_runs("c1")
        self.assertEqual(len(runs), 2)
        self.assertFalse(any(r["ok"] for r in runs))

    def test_stale_reminder_marked_missed(self):
        s = _user_sched()
        fired = []
        s.register_action("send_reminder",
                          lambda **kw: fired.append(kw.get("text")))
        r = asyncio.run(s.create_reminder("old", time.time() - 7200, "u1",
                                          stale_after_s=60))
        s._tick()
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = ?", (r.task_id,))["status"],
            "missed")
        self.assertNotIn("old", fired)

    def test_missing_handler_dead_letters_loudly(self):
        s = _user_sched()  # no handler registered for "ghost"
        asyncio.run(s.schedule_once("t9", time.time() - 1, "ghost", {},
                                    max_attempts=1))
        s._tick()
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = 't9'")["status"], "dead")


class UserOverlapTests(unittest.TestCase):
    def test_overlap_skip(self):
        s = _user_sched()
        fired = []
        s.register_action("ok", lambda **kw: fired.append(1))
        asyncio.run(s.schedule_cron("c1", "* * * * *", "ok", {},
                                    overlap_policy="skip"))
        s.db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = 'c1'",
                     (time.time() - 1,))
        s._mark_flight("c1", True)  # simulate a run in flight
        try:
            s._tick()
        finally:
            s._mark_flight("c1", False)
        self.assertEqual(fired, [])
        runs = s.recent_runs("c1")
        self.assertTrue(any("overlap" in r["result"] for r in runs))


class UserTimeoutTests(unittest.TestCase):
    def test_slow_action_times_out(self):
        s = _user_sched()
        async def slow(**kw):
            await asyncio.sleep(5)
        s.register_action("slow", slow)
        asyncio.run(s.schedule_once("t1", time.time() - 1, "slow", {},
                                    timeout_s=0.1, max_attempts=1))
        t0 = time.time()
        s._tick()
        self.assertLess(time.time() - t0, 4.0)
        runs = s.recent_runs("t1")
        self.assertTrue(any("timed out" in r["result"] for r in runs))


class UserListenerTests(unittest.TestCase):
    def test_events_emitted(self):
        s = _user_sched()
        seen = []
        for ev in ("fired", "failed", "missed", "dead"):
            s.add_listener(ev, lambda p, ev=ev: seen.append(ev))
        s.register_action("ok", lambda **kw: None)
        asyncio.run(s.schedule_once("t1", time.time() - 1, "ok", {}))
        s._tick()
        self.assertIn("fired", seen)
        r = asyncio.run(s.create_reminder("old", time.time() - 7200, "u1",
                                          stale_after_s=60))
        s._tick()
        self.assertIn("missed", seen)
        self.assertTrue(s.remove_listener("fired", s._listeners["fired"][0]))


class UserHookOperatorTests(unittest.TestCase):
    def test_operators(self):
        self.assertTrue(evaluate_conditions(
            {"price": {"lt": 100}}, {"price": 50}))
        self.assertFalse(evaluate_conditions(
            {"price": {"lt": 100}}, {"price": 150}))
        self.assertTrue(evaluate_conditions(
            {"tags": {"contains": "x"}}, {"tags": ["a", "x"]}))
        self.assertTrue(evaluate_conditions(
            {"msg": {"icontains": "HELLO"}}, {"msg": "well hello there"}))
        self.assertTrue(evaluate_conditions(
            {"user.tier": "pro"}, {"user": {"tier": "pro"}}))
        self.assertTrue(evaluate_conditions(
            {"n": {"regex": "^a+$"}}, {"n": "aaa"}))
        self.assertFalse(evaluate_conditions(
            {"missing": {"gt": 1}}, {}))
        # bare values still mean equality (backwards compatible)
        self.assertTrue(evaluate_conditions({"k": "v"}, {"k": "v"}))

    def test_hook_end_to_end(self):
        s = _user_sched()
        fired = []
        s.register_action("ok", lambda **kw: fired.append(kw.get("text")))
        h = asyncio.run(s.create_event_hook(
            "price", {"price": {"lt": 100}, "sym": "BTC"}, "ok",
            {"text": "dip"}))
        out = asyncio.run(s.trigger_event("price", {"price": 50, "sym": "BTC"}))
        self.assertEqual(out, [h.task_id])
        self.assertIn("dip", fired)
        out = asyncio.run(s.trigger_event("price", {"price": 150, "sym": "BTC"}))
        self.assertEqual(out, [])


class UserRruleTests(unittest.TestCase):
    def test_schedule_rrule_and_fire(self):
        s = _user_sched()
        fired = []
        s.register_action("ok", lambda **kw: fired.append(kw.get("text")))
        j = asyncio.run(s.schedule_rrule("r1", "FREQ=DAILY;COUNT=2", "ok",
                                         {"text": "rrule-fire"}))
        self.assertGreater(j.next_run, time.time())
        s.db.execute("UPDATE cron_jobs SET next_run = ? WHERE task_id = 'r1'",
                     (time.time() - 1,))
        s._tick()
        self.assertIn("rrule-fire", fired)
        with self.assertRaises(ValueError):
            asyncio.run(s.schedule_rrule("r2", "FREQ=NEVER", "ok", {}))


class UserTzTests(unittest.TestCase):
    def test_tz_cron_next_run_in_zone(self):
        s = _user_sched()
        s.register_action("ok", lambda **kw: None)
        j = asyncio.run(s.schedule_cron("c9", "0 9 * * *", "ok", {},
                                        tz="Africa/Lagos"))
        Lagos = ZoneInfo("Africa/Lagos")
        self.assertEqual(
            datetime.fromtimestamp(j.next_run, Lagos).strftime("%H:%M"),
            "09:00")
        with self.assertRaises(ValueError):
            asyncio.run(s.schedule_cron("cb", "0 9 * * *", "ok", {},
                                        tz="Not/AZone"))

    def test_naive_datetime_uses_tz(self):
        s = _user_sched()
        s.register_action("ok", lambda **kw: None)
        r = asyncio.run(s.create_reminder("x", datetime(2026, 10, 10, 9, 0),
                                          "u1", tz="Africa/Lagos"))
        Lagos = ZoneInfo("Africa/Lagos")
        self.assertEqual(
            datetime.fromtimestamp(r.due_at, Lagos).strftime("%H:%M"), "09:00")


class UserLifecycleTests(unittest.TestCase):
    def test_pause_resume_reschedule(self):
        s = _user_sched()
        s.register_action("ok", lambda **kw: None)
        asyncio.run(s.schedule_cron("c1", "0 9 * * *", "ok", {}))
        self.assertTrue(asyncio.run(s.pause("c1")))
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = 'c1'")["status"], "paused")
        self.assertTrue(asyncio.run(s.resume("c1")))
        self.assertEqual(
            s.db.query_one("SELECT status FROM scheduled_tasks "
                           "WHERE task_id = 'c1'")["status"], "pending")
        j = asyncio.run(s.reschedule_cron("c1", cron_expr="0 10 * * *"))
        self.assertEqual(j.cron_expr, "0 10 * * *")
        self.assertIsNone(asyncio.run(s.reschedule_cron("nope",
                                                        cron_expr="0 1 * * *")))

    def test_catch_up_fires_missed_one_time(self):
        s = _user_sched()
        fired = []
        s.register_action("ok", lambda **kw: fired.append(kw.get("text")))
        asyncio.run(s.schedule_once("t1", time.time() - 30, "ok",
                                    {"text": "caught"}))
        got = s.catch_up_on_startup()
        self.assertEqual(got, ["t1"])
        self.assertIn("caught", fired)

    def test_health(self):
        s = _user_sched()
        s.register_action("ok", lambda **kw: None)
        asyncio.run(s.schedule_once("t1", time.time() - 1, "ok", {},
                                    max_attempts=1))
        # leave it failing-free: register nothing for ghost -> dead
        asyncio.run(s.schedule_once("t2", time.time() - 1, "ghost", {},
                                    max_attempts=1))
        s._tick()
        h = s.health()
        self.assertIn("tasks", h)
        self.assertEqual(h["tasks"].get("dead"), 1)
        self.assertTrue(h["recent_runs"])
        self.assertIn("ok", h)

    def test_cron_parser_backwards_compat(self):
        parsed = CronParser.parse("0 9 * * MON-FRI")
        self.assertEqual(parsed["minute"], {0})
        self.assertEqual(parsed["hour"], {9})
        self.assertIn(1, parsed["day_of_week"])
        self.assertEqual(CronParser.DAY_NAMES["MON"], 1)
        nxt = CronParser.next_run("0 9 * * *")
        self.assertGreater(nxt, time.time())
        # extended fields validate too
        CronParser.parse("0 9 L * *")
        CronParser.parse("0 9 * * 5#3")


if __name__ == "__main__":
    unittest.main()
