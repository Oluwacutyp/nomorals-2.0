"""Scheduler upgrades: timezones, cron, dependencies, retries, catch-up."""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from nomorals.agents.scheduler import (
    Scheduler,
    _cron_matches,
    _next_cron,
    _next_daily,
    _parse_cron,
    parse_schedule_spec,
)
from nomorals.storage.db import Database


def _ctx():
    """Minimal context with an in-memory DB and a stub tool registry."""
    db = Database(":memory:")
    db.migrate()  # creates schedule_jobs with the new columns
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw: SimpleNamespace(ok=True, value="ok", error=None)
    )
    return SimpleNamespace(db=db, tools=tools)


class ParseSpecTests(unittest.TestCase):
    def test_cron_prefixed(self):
        kind, detail = parse_schedule_spec("cron 0 22 * * *")
        self.assertEqual(kind, "cron")
        self.assertEqual(detail, "0 22 * * *")

    def test_cron_bare(self):
        kind, detail = parse_schedule_spec("0 22 * * *")
        self.assertEqual(kind, "cron")

    def test_cron_complex(self):
        kind, detail = parse_schedule_spec("*/15 9-17 * * 1-5")
        self.assertEqual(kind, "cron")
        self.assertEqual(detail, "*/15 9-17 * * 1-5")

    def test_cron_invalid(self):
        with self.assertRaises(ValueError):
            parse_schedule_spec("cron 99 99 * * *")

    def test_daily_with_timezone(self):
        kind, detail = parse_schedule_spec("daily 22:00 America/New_York")
        self.assertEqual(kind, "daily")
        self.assertEqual(detail, "22:00 America/New_York")

    def test_daily_bare_with_timezone(self):
        kind, detail = parse_schedule_spec("22:00 Europe/London")
        self.assertEqual(kind, "daily")
        self.assertIn("Europe/London", detail)

    def test_daily_bad_timezone(self):
        with self.assertRaises(ValueError):
            parse_schedule_spec("daily 22:00 Not/AZone")

    def test_old_specs_still_work(self):
        self.assertEqual(parse_schedule_spec("every 30m")[0], "every")
        self.assertEqual(parse_schedule_spec("22:00")[0], "daily")
        kind, _ = parse_schedule_spec("at 2030-01-01 09:00")
        self.assertEqual(kind, "at")


class CronMathTests(unittest.TestCase):
    def test_matches_simple(self):
        dt = datetime(2026, 10, 5, 22, 0)  # Monday
        self.assertTrue(_cron_matches("0 22 * * *", dt))
        self.assertFalse(_cron_matches("0 21 * * *", dt))

    def test_matches_weekday(self):
        monday = datetime(2026, 10, 5, 9, 0)  # Monday
        sunday = datetime(2026, 10, 4, 9, 0)  # Sunday
        self.assertTrue(_cron_matches("0 9 * * 1", monday))
        self.assertFalse(_cron_matches("0 9 * * 1", sunday))
        self.assertTrue(_cron_matches("0 9 * * 0", sunday))
        self.assertTrue(_cron_matches("0 9 * * 7", sunday))  # 7 = Sunday too

    def test_matches_step(self):
        self.assertTrue(_cron_matches("*/15 * * * *", datetime(2026, 1, 1, 0, 30)))
        self.assertFalse(_cron_matches("*/15 * * * *", datetime(2026, 1, 1, 0, 31)))

    def test_next_cron(self):
        now = datetime(2026, 10, 5, 21, 30)  # Monday 21:30
        nxt = _next_cron("0 22 * * *", now)
        expected = datetime(2026, 10, 5, 22, 0).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=1)

    def test_next_cron_next_day(self):
        now = datetime(2026, 10, 5, 23, 30)  # Monday 23:30
        nxt = _next_cron("0 22 * * *", now)
        expected = datetime(2026, 10, 6, 22, 0).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=1)

    def test_next_cron_weekly(self):
        # Monday 10:00, next Friday 9:00
        now = datetime(2026, 10, 5, 10, 0)
        nxt = _next_cron("0 9 * * 5", now)
        expected = datetime(2026, 10, 9, 9, 0).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=1)


class TimezoneDailyTests(unittest.TestCase):
    def test_next_daily_with_tz(self):
        # 22:00 New York. 2026-10-05 21:30 UTC = 17:30 EDT.
        # Next 22:00 EDT = 2026-10-06 02:00 UTC.
        now_utc = datetime(2026, 10, 5, 21, 30, tzinfo=ZoneInfo("UTC"))
        nxt = _next_daily("22:00 America/New_York", now_utc)
        expected = datetime(2026, 10, 6, 2, 0, tzinfo=ZoneInfo("UTC")).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=1)

    def test_next_daily_tz_next_day(self):
        # 2026-10-05 23:30 EDT = 2026-10-06 03:30 UTC. Next 22:00 EDT is tomorrow.
        now_utc = datetime(2026, 10, 6, 3, 30, tzinfo=ZoneInfo("UTC"))
        nxt = _next_daily("22:00 America/New_York", now_utc)
        expected = datetime(2026, 10, 7, 2, 0, tzinfo=ZoneInfo("UTC")).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=1)

    def test_next_daily_no_tz_unchanged(self):
        now = datetime(2026, 10, 5, 21, 30)
        nxt = _next_daily("22:00", now)
        expected = datetime(2026, 10, 5, 22, 0).timestamp()
        self.assertAlmostEqual(nxt, expected, delta=60)


class SchedulerUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.sched = Scheduler(self.ctx)

    def test_add_with_timezone(self):
        job = self.sched.add("ny-night", "daily 22:00 America/New_York",
                             "message", {"text": "goodnight"},
                             timezone="America/New_York")
        row = self.sched._find(job["id"])
        self.assertEqual(row["timezone"], "America/New_York")
        # next_run should be a future 22:00 EDT
        nxt = datetime.fromtimestamp(job["next_run"], tz=ZoneInfo("America/New_York"))
        self.assertEqual((nxt.hour, nxt.minute), (22, 0))

    def test_add_with_cron(self):
        job = self.sched.add("cronjob", "cron 0 9 * * 1-5",
                             "message", {"text": "weekday"})
        self.assertEqual(job["kind"], "cron")
        row = self.sched._find(job["id"])
        self.assertEqual(row["spec"], "0 9 * * 1-5")

    def test_add_bad_timezone_rejected(self):
        with self.assertRaises(ValueError):
            self.sched.add("bad", "daily 22:00", "message", {"text": "x"},
                           timezone="Not/AZone")

    def test_add_bad_dependency_rejected(self):
        with self.assertRaises(ValueError):
            self.sched.add("child", "every 1h", "message", {"text": "x"},
                           depends_on="nonexistent-job-id")

    def test_dependency_blocks_until_parent_succeeds(self):
        parent = self.sched.add("parent", "every 1h", "message", {"text": "p"})
        child = self.sched.add("child", "every 1h", "message", {"text": "c"},
                               depends_on=parent["id"])
        child_row = self.sched._find(child["id"])
        # parent never ran → dependency unmet
        self.assertFalse(self.sched._dependency_ok(child_row))
        # force parent to be due and run it
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1, parent["id"]))
        self.sched.tick()
        child_row = self.sched._find(child["id"])
        self.assertTrue(self.sched._dependency_ok(child_row))

    def test_dependency_blocks_on_parent_failure(self):
        # parent with a tool payload that will fail
        tools = SimpleNamespace(
            call=lambda tool, capabilities=None, **kw: SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message="boom")))
        ctx = SimpleNamespace(db=self.ctx.db, tools=tools)
        sched = Scheduler(ctx)
        parent = sched.add("failparent", "every 1h", "tool",
                           {"tool": "nope", "args": {}})
        child = sched.add("child", "every 1h", "message", {"text": "c"},
                          depends_on=parent["id"])
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id IN (?, ?)",
            (time.time() - 1, parent["id"], child["id"]))
        sched.tick()  # parent fails
        child_row = sched._find(child["id"])
        self.assertFalse(sched._dependency_ok(child_row))

    def test_retry_on_failure(self):
        tools = SimpleNamespace(
            call=lambda tool, capabilities=None, **kw: SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message="boom")))
        ctx = SimpleNamespace(db=self.ctx.db, tools=tools)
        sched = Scheduler(ctx)
        job = sched.add("retryme", "every 1h", "tool",
                        {"tool": "nope", "args": {}},
                        max_retries=2, retry_delay=60)
        outcome = sched.run_now(job["id"])
        self.assertFalse(outcome["ok"])
        self.assertTrue(outcome["will_retry"])
        self.assertEqual(outcome["retry_count"], 1)
        row = sched._find(job["id"])
        self.assertEqual(row["retry_count"], 1)
        # next_run should be ~60s out (linear backoff × 1)
        self.assertGreater(row["next_run"], time.time() + 30)

    def test_retry_exhaustion_reports_failure(self):
        tools = SimpleNamespace(
            call=lambda tool, capabilities=None, **kw: SimpleNamespace(
                ok=False, value=None,
                error=SimpleNamespace(message="boom")))
        ctx = SimpleNamespace(db=self.ctx.db, tools=tools)
        sched = Scheduler(ctx)
        job = sched.add("retryme", "every 1h", "tool",
                        {"tool": "nope", "args": {}},
                        max_retries=1, retry_delay=60)
        sched.run_now(job["id"])  # fail 1 → retry scheduled
        row = sched._find(job["id"])
        self.assertEqual(row["retry_count"], 1)
        # force the retry to be due and run again → retries exhausted
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1, job["id"]))
        outcome = sched.tick()[0]
        self.assertFalse(outcome["ok"])
        self.assertFalse(outcome["will_retry"])
        row = sched._find(job["id"])
        # after exhaustion it goes back to the normal schedule
        self.assertGreater(row["next_run"], time.time())

    def test_retry_resets_on_success(self):
        calls = {"n": 0}

        def fake_call(tool, capabilities=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(ok=False, value=None,
                                       error=SimpleNamespace(message="boom"))
            return SimpleNamespace(ok=True, value="fine", error=None)

        tools = SimpleNamespace(call=fake_call)
        ctx = SimpleNamespace(db=self.ctx.db, tools=tools)
        sched = Scheduler(ctx)
        job = sched.add("flaky", "every 1h", "tool",
                        {"tool": "maybe", "args": {}},
                        max_retries=3, retry_delay=60)
        sched.run_now(job["id"])  # fail → retry 1
        row = sched._find(job["id"])
        self.assertEqual(row["retry_count"], 1)
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1, job["id"]))
        sched.tick()  # success → retry_count resets
        row = sched._find(job["id"])
        self.assertEqual(row["retry_count"], 0)

    def test_catch_up_runs_missed_jobs(self):
        job = self.sched.add("missed", "every 1h", "message", {"text": "hello"})
        # pretend the bot was down for 30 minutes past the due time
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 1800, job["id"]))
        caught = self.sched.catch_up_on_startup(max_age_hours=24)
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0]["id"], job["id"])

    def test_catch_up_skips_stale_jobs(self):
        job = self.sched.add("stale", "every 1h", "message", {"text": "hello"})
        # missed 48h ago — beyond the 24h max age
        self.ctx.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 48 * 3600, job["id"]))
        caught = self.sched.catch_up_on_startup(max_age_hours=24)
        self.assertEqual(len(caught), 0)
        # but it got rescheduled forward
        row = self.sched._find(job["id"])
        self.assertGreater(row["next_run"], time.time())

    def test_format_shows_new_fields(self):
        parent = self.sched.add("p", "every 1h", "message", {"text": "p"})
        job = self.sched.add("c", "daily 22:00 America/New_York",
                             "message", {"text": "c"},
                             depends_on=parent["id"], max_retries=3)
        formatted = self.sched._format_job(self.sched._find(job["id"]))
        self.assertEqual(formatted["depends_on"], parent["id"])
        self.assertEqual(formatted["max_retries"], 3)
        self.assertEqual(formatted["timezone"], "America/New_York")


if __name__ == "__main__":
    unittest.main()
