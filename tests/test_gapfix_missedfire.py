"""Missed-fire fix (owner order 2026-11-20): a missed time-anchored job
is SKIPPED, never fired late, and the next firing stays on the clock.

Root 1 (the one biting the pulses): ``morning_pulse.PULSE_POLICIES``
carried ``missed_fire_policy=fire_now`` — a morning/evening pulse
missed by hours fired late on the next tick, and the boot catch-up
(``check_pulse_catchup`` / ``check_slot_catchup``) fired it late again
on every boot.  Both are now skips, journaled by name.

Root 2 (the drift behind the skip): the scheduler's ``skip`` advanced
``every`` jobs from *now* (``now + interval``), moving the phase to
the late moment — a 30-minute job missed by 40 minutes came back 40
minutes off its grid, and every later firing inherited the shift.
``skip`` (and ``next_only`` for multi-period misses, which could land
``next_run`` in the past) now advances to the first firing strictly
after now on the schedule's own grid.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database


def _ctx(**kw):
    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw2: SimpleNamespace(
            ok=True, value="ok", error=None))
    ctx = SimpleNamespace(db=db, tools=tools, settings=SimpleNamespace(),
                          extras={})
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def _pulse_ctx(tmpdir, **pulse_kw):
    pulse = SimpleNamespace(enabled=True, time="23:00",
                            timezone="America/Denver")
    for k, v in pulse_kw.items():
        setattr(pulse, k, v)
    return SimpleNamespace(
        db=_ctx().db,
        settings=SimpleNamespace(workspace_dir=tmpdir, pulse=pulse),
        extras={})


class MissedDailyPulseTests(unittest.TestCase):
    """A missed daily pulse-shaped job never fires late and stays on
    the clock."""

    def test_missed_daily_skip_never_fires(self):
        sched = Scheduler(_ctx())
        job = sched.add("pulse-shaped", "daily 23:00 America/Denver",
                        "message", {"text": "x"},
                        missed_fire_policy="skip")
        missed = time.time() - 7200  # missed 2h ago
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (missed, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 0)
        row = sched._find(job["id"])
        # on the clock: the next firing is 23:00 Denver, not "now + a day"
        dt = datetime.fromtimestamp(row["next_run"],
                                    tz=ZoneInfo("America/Denver"))
        self.assertEqual((dt.hour, dt.minute), (23, 0))
        self.assertGreater(row["next_run"], time.time())

    def test_missed_daily_skip_journals_policy_and_lateness(self):
        sched = Scheduler(_ctx())
        job = sched.add("pulse-shaped", "daily 23:00 America/Denver",
                        "message", {"text": "x"},
                        missed_fire_policy="skip")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 7200, job["id"]))
        sched.tick()
        runs = sched.recent_runs(job["id"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "skipped")
        self.assertIn("missed_fire_policy=skip", runs[0]["result"])
        self.assertIn("7200s", runs[0]["result"])

    def test_missed_daily_next_only_stays_on_clock(self):
        sched = Scheduler(_ctx())
        job = sched.add("pulse-shaped", "daily 23:00 America/Denver",
                        "message", {"text": "x"},
                        missed_fire_policy="next_only")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 7200, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 0)
        row = sched._find(job["id"])
        dt = datetime.fromtimestamp(row["next_run"],
                                    tz=ZoneInfo("America/Denver"))
        self.assertEqual((dt.hour, dt.minute), (23, 0))

    def test_missed_daily_next_only_multi_period_lands_in_future(self):
        # The old next_only took the single next occurrence after the
        # missed firing — missed by 3 days, that lands 2 days in the
        # past and the job re-misses on every tick.
        sched = Scheduler(_ctx())
        job = sched.add("pulse-shaped", "daily 23:00 America/Denver",
                        "message", {"text": "x"},
                        missed_fire_policy="next_only")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 3 * 86400, job["id"]))
        sched.tick()
        row = sched._find(job["id"])
        self.assertGreater(row["next_run"], time.time())


class MissedEveryPhaseTests(unittest.TestCase):
    """A missed ``every`` job keeps its phase — the drift root."""

    def test_skip_keeps_hourly_phase(self):
        sched = Scheduler(_ctx())
        job = sched.add("hourly", "every 1h", "message", {"text": "x"},
                        missed_fire_policy="skip")
        missed = time.time() - 3590  # missed ~1h ago, not on the grid edge
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (missed, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 0)
        row = sched._find(job["id"])
        nxt = row["next_run"]
        self.assertGreater(nxt, time.time())
        self.assertLess(nxt, time.time() + 3600)
        # on the grid anchored at the missed firing (within 2s of exact)
        self.assertAlmostEqual((nxt - missed) % 3600, 0, delta=2.0)

    def test_skip_keeps_half_hour_phase(self):
        sched = Scheduler(_ctx())
        job = sched.add("half-hour", "every 30m", "message",
                        {"text": "x"}, missed_fire_policy="skip")
        missed = time.time() - 2400  # missed 40m ago: the old skip
        # came back at now+30m, 40 minutes off the :00/:30 grid
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (missed, job["id"]))
        sched.tick()
        row = sched._find(job["id"])
        nxt = row["next_run"]
        self.assertGreater(nxt, time.time())
        self.assertAlmostEqual((nxt - missed) % 1800, 0, delta=2.0)

    def test_skip_multi_period_every_lands_in_future(self):
        sched = Scheduler(_ctx())
        job = sched.add("half-hour", "every 30m", "message",
                        {"text": "x"}, missed_fire_policy="skip")
        missed = time.time() - 3 * 86400  # missed 3 days ago
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (missed, job["id"]))
        sched.tick()
        row = sched._find(job["id"])
        self.assertGreater(row["next_run"], time.time())
        self.assertLess(row["next_run"], time.time() + 1800)
        self.assertAlmostEqual((row["next_run"] - missed) % 1800,
                               0, delta=2.0)


class MissedCronTests(unittest.TestCase):
    def test_missed_hourly_cron_next_only_lands_in_future(self):
        # Old next_only: single next occurrence after the missed firing
        # (missed 3h ago -> 2h in the past).  Now: next on-clock firing.
        sched = Scheduler(_ctx())
        job = sched.add("hourly-cron", "cron 0 * * * *", "message",
                        {"text": "x"}, missed_fire_policy="next_only")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 3 * 3600, job["id"]))
        sched.tick()
        row = sched._find(job["id"])
        self.assertGreater(row["next_run"], time.time())
        dt = datetime.fromtimestamp(row["next_run"])
        self.assertEqual((dt.minute, dt.second), (0, 0))


class GraceAndFireNowTests(unittest.TestCase):
    def test_merely_late_runs_normally(self):
        # Inside the 300s grace a job is merely late, not missed: it
        # runs.  The grace absorbs tick jitter, not a missed firing.
        sched = Scheduler(_ctx())
        job = sched.add("late", "every 1h", "message", {"text": "x"},
                        missed_fire_policy="skip")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 30, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])

    def test_fire_now_still_fires_late(self):
        # fire_now's meaning is unchanged for jobs that genuinely want
        # it (pollers catching up): a missed firing runs once, now.
        sched = Scheduler(_ctx())
        job = sched.add("catchup", "every 1h", "message", {"text": "x"},
                        missed_fire_policy="fire_now")
        sched.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (time.time() - 3600, job["id"]))
        outcomes = sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])


class PulsePolicyTests(unittest.TestCase):
    def test_pulse_policies_skip_missed(self):
        from nomorals.agents.morning_pulse import PULSE_POLICIES
        self.assertEqual(PULSE_POLICIES["missed_fire_policy"], "skip")

    def test_ensure_pulse_job_registers_skip(self):
        from unittest.mock import MagicMock, patch

        from nomorals.agents.morning_pulse import ensure_pulse_job
        sched = MagicMock()
        sched.list_jobs.return_value = []
        sched.add.return_value = {"id": "job-1"}
        with patch("nomorals.agents.scheduler.Scheduler",
                   return_value=sched):
            ensure_pulse_job(_ctx())
        _, kwargs = sched.add.call_args
        self.assertEqual(kwargs.get("missed_fire_policy"), "skip")


class BootCatchupTests(unittest.TestCase):
    """The boot catch-up never fires a missed pulse late; only an
    explicit catch-up does."""

    def test_automatic_catchup_skips_missed_morning(self):
        import nomorals.agents.morning_pulse as mp
        ctx = _pulse_ctx(tempfile.mkdtemp())
        with patch.object(mp, "_pulse_due_today",
                          return_value=time.time() - 3600), \
             patch.object(mp, "run_pulse",
                          side_effect=AssertionError("must not run")):
            res = mp.check_pulse_catchup(ctx)
        self.assertFalse(res["catchup"])
        self.assertIn("missed: skipped", res["reason"])
        self.assertIn("skip", res["reason"])
        # the miss is journaled in the autonomy ledger, by name
        rows = ctx.db.query(
            "SELECT * FROM autonomy_ledger WHERE kind = 'skipped' "
            "ORDER BY rowid DESC LIMIT 1")
        self.assertEqual(len(rows), 1)
        self.assertIn("morning", rows[0]["summary"])
        self.assertIn("missed: skipped", rows[0]["summary"])

    def test_explicit_catchup_still_fires_missed_morning(self):
        import nomorals.agents.morning_pulse as mp
        ctx = _pulse_ctx(tempfile.mkdtemp())
        ran = []

        def fake_pulse(c):
            ran.append(1)
            return {"ok": True, "delivered_text": True}

        with patch.object(mp, "_pulse_due_today",
                          return_value=time.time() - 3600), \
             patch.object(mp, "run_pulse", side_effect=fake_pulse):
            res = mp.check_pulse_catchup(ctx, explicit=True)
        self.assertTrue(res["catchup"])
        self.assertEqual(len(ran), 1)

    def test_automatic_catchup_skips_missed_evening_slot(self):
        import nomorals.agents.morning_pulse as mp
        ctx = _pulse_ctx(tempfile.mkdtemp(), evening_time="19:00")
        with patch.object(mp, "_slot_due_today",
                          return_value=time.time() - 7200), \
             patch.object(mp, "run_slot",
                          side_effect=AssertionError("must not run")):
            res = mp.check_slot_catchup(ctx, "evening")
        self.assertFalse(res["catchup"])
        self.assertIn("missed: skipped", res["reason"])
        rows = ctx.db.query(
            "SELECT * FROM autonomy_ledger WHERE kind = 'skipped' "
            "ORDER BY rowid DESC LIMIT 1")
        self.assertEqual(len(rows), 1)
        self.assertIn("evening", rows[0]["summary"])

    def test_explicit_catchup_still_fires_missed_evening_slot(self):
        import nomorals.agents.morning_pulse as mp
        ctx = _pulse_ctx(tempfile.mkdtemp(), evening_time="19:00")
        ran = []

        def fake_slot(c, slot):
            ran.append(slot)
            return {"ok": True, "delivered_text": True}

        with patch.object(mp, "_slot_due_today",
                          return_value=time.time() - 7200), \
             patch.object(mp, "run_slot", side_effect=fake_slot):
            res = mp.check_slot_catchup(ctx, "evening", explicit=True)
        self.assertTrue(res["catchup"])
        self.assertEqual(ran, ["evening"])


if __name__ == "__main__":
    unittest.main()
