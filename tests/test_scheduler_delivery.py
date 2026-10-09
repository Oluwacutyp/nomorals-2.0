"""Scheduler delivery foundation: retry with backoff, dead-lettering,
Termux fallback, restart persistence, timezone correctness, never-raises.

Covers the delivery-reliability work on the scheduler + notifier:
- failed deliveries retry with exponential backoff (not silently dropped)
- exhausted retries dead-letter instead of spinning forever
- a Termux system notification is the last-resort channel when no chat
  adapter is live in the session
- jobs survive process restarts (DB persistence + catch-up)
- the scheduler tick drives the redelivery queue on its own
- /schedule health reports the loop, next fire, and delivery queue
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from nomorals.agents import notifier as notmod
from nomorals.agents.notifier import (
    DELIVERY_MAX_ATTEMPTS,
    Notifier,
    delivery_backoff,
)
from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database


class _SendResult:
    def __init__(self, ok=True, error=""):
        self.ok = ok
        self.error = error


class FakeGateway:
    """Chat gateway double.  ``live`` = platforms running in session."""

    def __init__(self, live=(), fail_status=False, fail_send=False):
        self.live = set(live)
        self.fail_status = fail_status
        self.fail_send = fail_send
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        if self.fail_status:
            raise RuntimeError("status exploded")
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
        if self.fail_send:
            raise RuntimeError("send exploded")
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


def _ctx(gateway=None, owner_chats="telegram:111", db=None):
    db = db or Database(":memory:")
    db.migrate()
    partner = SimpleNamespace(owner_chats=owner_chats)
    settings = SimpleNamespace(partner=partner)
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw: SimpleNamespace(
            ok=True, value="ok", error=None))
    return SimpleNamespace(db=db, settings=settings, tools=tools,
                           extras={"gateway": gateway} if gateway else {})


def _row(db, nid):
    return db.query_one("SELECT * FROM notifications WHERE id = ?", (nid,))


class BackoffShapeTests(unittest.TestCase):
    def test_backoff_grows_exponentially(self):
        b1 = delivery_backoff(1)
        b2 = delivery_backoff(2)
        b3 = delivery_backoff(3)
        self.assertGreaterEqual(b1, 60.0)
        self.assertLess(b1, 60.0 + 30.0 + 1)
        self.assertGreaterEqual(b2, 120.0)
        self.assertGreaterEqual(b3, 240.0)

    def test_backoff_capped(self):
        self.assertLessEqual(delivery_backoff(100), 6 * 3600.0 + 30.0 + 1)

    def test_backoff_never_raises_on_garbage(self):
        self.assertGreater(delivery_backoff(0), 0)
        self.assertGreater(delivery_backoff(-5), 0)


class DeliveryRetryTests(unittest.TestCase):
    def setUp(self):
        notmod._termux_notify_bin = None  # reset the which() cache

    def tearDown(self):
        notmod._termux_notify_bin = None

    def _dead_notifier(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        return Notifier(ctx, gateway=ctx.extras["gateway"])

    def test_failed_publish_parks_with_zero_retry(self):
        n = self._dead_notifier()
        res = n.publish("schedule", "t", "b")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")
        row = _row(n.db, res["id"])
        self.assertEqual(row["retry_count"], 0)
        self.assertEqual(row["next_retry_at"], 0.0)

    def test_redeliver_attempt_burns_and_backs_off(self):
        n = self._dead_notifier()
        res = n.publish("schedule", "t", "b")
        self.assertEqual(n.redeliver(), 0)  # nothing live — attempt fails
        row = _row(n.db, res["id"])
        self.assertEqual(row["retry_count"], 1)
        self.assertEqual(row["delivery_state"], "failed")
        # backoff engaged: the next attempt is ~2 minutes out, not now
        self.assertGreater(row["next_retry_at"], time.time() + 60)

    def test_backoff_blocks_immediate_second_attempt(self):
        n = self._dead_notifier()
        n.publish("schedule", "t", "b")
        n.redeliver()  # attempt 1 → backs off
        row = n.db.query_one(
            "SELECT retry_count FROM notifications WHERE delivered = 0")
        self.assertEqual(row["retry_count"], 1)
        n.redeliver()  # backoff not elapsed → no attempt consumed
        row = n.db.query_one(
            "SELECT retry_count FROM notifications WHERE delivered = 0")
        self.assertEqual(row["retry_count"], 1)

    def test_retry_succeeds_once_channel_returns(self):
        n = self._dead_notifier()
        res = n.publish("schedule", "t", "b")
        n.redeliver()  # fails, backs off
        # the platform comes back; backoff elapsed
        n.gateway = FakeGateway(live=("telegram",))
        n.db.execute(
            "UPDATE notifications SET next_retry_at = 0 WHERE id = ?",
            (res["id"],))
        self.assertEqual(n.redeliver(), 1)
        row = _row(n.db, res["id"])
        self.assertTrue(row["delivered"])
        self.assertEqual(row["delivery_state"], "sent")
        self.assertEqual(row["channel"], "telegram")

    def test_exhausted_retries_dead_letter(self):
        n = self._dead_notifier()
        res = n.publish("schedule", "t", "b")
        n.db.execute(
            "UPDATE notifications SET retry_count = ?, next_retry_at = 0 "
            "WHERE id = ?",
            (DELIVERY_MAX_ATTEMPTS - 1, res["id"]))
        self.assertEqual(n.redeliver(), 0)
        row = _row(n.db, res["id"])
        self.assertEqual(row["delivery_state"], "dead")
        self.assertFalse(row["delivered"])
        # dead rows are out of the retry queue and out of pending()
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(n.pending(), [])
        dead = n.dead_letters()
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["id"], res["id"])

    def test_queue_depth_buckets(self):
        n = self._dead_notifier()
        r1 = n.publish("schedule", "one", "b")
        r2 = n.publish("schedule", "two", "b")
        n.db.execute(
            "UPDATE notifications SET delivery_state = 'dead' WHERE id = ?",
            (r2["id"],))
        depth = n.queue_depth()
        self.assertEqual(depth["retryable"], 1)
        self.assertEqual(depth["dead"], 1)
        self.assertEqual(depth["held"], 0)
        self.assertEqual(_row(n.db, r1["id"])["delivery_state"], "failed")


class TermuxFallbackTests(unittest.TestCase):
    def setUp(self):
        notmod._termux_notify_bin = None

    def tearDown(self):
        notmod._termux_notify_bin = None

    def _termux_env(self, run=None):
        run = run if run is not None else SimpleNamespace(returncode=0,
                                                          stderr=b"")
        return (patch.dict(os.environ, {"NM_TERMUX_NOTIFY": "1"}),
                patch("nomorals.core.profile.is_termux", return_value=True),
                patch("shutil.which", return_value="/bin/termux-notification"),
                patch("subprocess.run", return_value=run))

    def test_termux_fallback_delivers_when_no_chat_live(self):
        env, termux, which, run = self._termux_env()
        with env, termux, which, run:
            ctx = _ctx(gateway=FakeGateway(live=()))
            n = Notifier(ctx, gateway=ctx.extras["gateway"])
            self.assertTrue(n.termux_fallback_available())
            res = n.publish("schedule", "job done", "all good")
        self.assertTrue(res["delivered"])
        self.assertEqual(res["delivery_state"], "sent")
        self.assertEqual(res["channel"], "termux")

    def test_termux_fallback_disabled_by_env(self):
        env, termux, which, run = self._termux_env()
        with patch.dict(os.environ, {"NM_TERMUX_NOTIFY": "0"}), \
                termux, which, run:
            ctx = _ctx(gateway=FakeGateway(live=()))
            n = Notifier(ctx, gateway=ctx.extras["gateway"])
            self.assertFalse(n.termux_fallback_available())
            res = n.publish("schedule", "job done", "all good")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")

    def test_termux_fallback_not_on_non_termux(self):
        with patch("nomorals.core.profile.is_termux", return_value=False):
            ctx = _ctx(gateway=FakeGateway(live=()))
            n = Notifier(ctx, gateway=ctx.extras["gateway"])
            self.assertFalse(n.termux_fallback_available())

    def test_termux_fallback_never_raises(self):
        env, termux, which, run = self._termux_env(
            run=RuntimeError("boom"))
        # subprocess.run itself raising must not break publish
        with env, termux, which, \
                patch("subprocess.run", side_effect=RuntimeError("boom")):
            ctx = _ctx(gateway=FakeGateway(live=()))
            n = Notifier(ctx, gateway=ctx.extras["gateway"])
            res = n.publish("schedule", "t", "b")
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")

    def test_redeliver_uses_termux_fallback(self):
        env, termux, which, run = self._termux_env()
        ctx = _ctx(gateway=FakeGateway(live=()))
        n = Notifier(ctx, gateway=ctx.extras["gateway"])
        res = n.publish("schedule", "t", "b")  # parked: failed
        self.assertFalse(res["delivered"])
        with env, termux, which, run:
            self.assertEqual(n.redeliver(), 1)
        row = _row(n.db, res["id"])
        self.assertTrue(row["delivered"])
        self.assertEqual(row["channel"], "termux")


class SchedulerTickRedeliversTests(unittest.TestCase):
    def test_tick_drives_redelivery_queue(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        sched._last_redeliver_at = 0.0  # force the throttled sweep
        n = sched.notifier
        res = n.publish("schedule", "t", "b")
        self.assertFalse(res["delivered"])
        sched.tick()  # no jobs due; the sweep still runs
        row = _row(n.db, res["id"])
        self.assertEqual(row["retry_count"], 1)

    def test_redelivery_sweep_is_throttled(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"],
                          redeliver_interval=3600)
        sched._last_redeliver_at = time.time()  # just swept
        n = sched.notifier
        res = n.publish("schedule", "t", "b")
        sched.tick()
        row = _row(n.db, res["id"])
        self.assertEqual(row["retry_count"], 0)  # sweep skipped


class RestartPersistenceTests(unittest.TestCase):
    def test_jobs_survive_across_scheduler_instances(self):
        tmp = tempfile.mkdtemp(prefix="sched-restart-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "devon.db")

        db1 = Database(path)
        db1.migrate()
        ctx1 = _ctx(db=db1)
        s1 = Scheduler(ctx1)
        job = s1.add("persist-me", "every 1h", "message",
                     {"text": "still here"})
        db1.close() if hasattr(db1, "close") else None

        # "process restart": a brand-new Scheduler on the same DB file
        db2 = Database(path)
        db2.migrate()
        ctx2 = _ctx(db=db2)
        s2 = Scheduler(ctx2)
        found = s2._find(job["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "persist-me")
        self.assertTrue(found["enabled"])

    def test_due_job_fires_after_restart(self):
        tmp = tempfile.mkdtemp(prefix="sched-restart-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "devon.db")
        db = Database(path)
        db.migrate()
        gw = FakeGateway(live=("telegram",))
        ctx = _ctx(db=db, gateway=gw, owner_chats="telegram:111")
        s1 = Scheduler(ctx, gateway=gw)
        job = s1.add("fire-after-restart", "every 1h", "message",
                     {"text": "hello again"})
        # simulate downtime: the job became due while "down"
        db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (time.time() - 5, job["id"]))

        s2 = Scheduler(ctx, gateway=gw)  # fresh instance, same DB
        outcomes = s2.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])
        self.assertIn("sent via telegram", outcomes[0]["result"])
        self.assertEqual(len(gw.sent), 1)

    def test_catch_up_runs_missed_jobs(self):
        ctx = _ctx(gateway=FakeGateway(live=("telegram",)))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        job = sched.add("missed", "every 1h", "message", {"text": "catch me"})
        ctx.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                       (time.time() - 60, job["id"]))
        caught = sched.catch_up_on_startup()
        self.assertEqual(len(caught), 1)
        self.assertTrue(caught[0]["ok"])

    def test_stale_missed_jobs_reschedule_without_running(self):
        ctx = _ctx(gateway=FakeGateway(live=("telegram",)))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        job = sched.add("ancient", "every 1h", "message", {"text": "nope"})
        ctx.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                       (time.time() - 48 * 3600, job["id"]))
        caught = sched.catch_up_on_startup(max_age_hours=24.0)
        self.assertEqual(caught, [])
        row = sched._find(job["id"])
        # rescheduled forward, not executed
        self.assertGreater(row["next_run"], time.time())
        self.assertIsNone(row["last_run"])


class TimezoneCorrectnessTests(unittest.TestCase):
    def test_cron_with_timezone_fires_at_tz_wallclock(self):
        ctx = _ctx()
        sched = Scheduler(ctx)
        job = sched.add("ny-morning", "cron 0 9 * * *", "message",
                        {"text": "good morning"},
                        timezone="America/New_York")
        nxt = datetime.fromtimestamp(job["next_run"],
                                     tz=ZoneInfo("America/New_York"))
        self.assertEqual((nxt.hour, nxt.minute), (9, 0))

    def test_daily_with_timezone_fires_at_tz_wallclock(self):
        ctx = _ctx()
        sched = Scheduler(ctx)
        job = sched.add("pulse", "daily 23:00 America/Denver", "message",
                        {"text": "night pulse"})
        nxt = datetime.fromtimestamp(job["next_run"],
                                     tz=ZoneInfo("America/Denver"))
        self.assertEqual((nxt.hour, nxt.minute), (23, 0))

    def test_daily_without_timezone_uses_server_local(self):
        ctx = _ctx()
        sched = Scheduler(ctx)
        job = sched.add("local", "daily 07:30", "message", {"text": "x"})
        nxt = datetime.fromtimestamp(job["next_run"]).astimezone()
        local = datetime.now().astimezone().tzinfo
        nxt_local = datetime.fromtimestamp(job["next_run"], tz=local)
        self.assertEqual((nxt_local.hour, nxt_local.minute), (7, 30))


class NeverRaisesTests(unittest.TestCase):
    def test_publish_with_exploding_gateway(self):
        ctx = _ctx(gateway=FakeGateway(live=("telegram",), fail_send=True,
                                        fail_status=True))
        n = Notifier(ctx, gateway=ctx.extras["gateway"])
        res = n.publish("schedule", "t", "b")  # must not raise
        self.assertFalse(res["delivered"])
        self.assertEqual(res["delivery_state"], "failed")

    def test_tick_with_exploding_gateway(self):
        ctx = _ctx(gateway=FakeGateway(live=("telegram",), fail_send=True))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        job = sched.add("boom", "every 1h", "message", {"text": "x"})
        ctx.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                       (time.time() - 1, job["id"]))
        outcomes = sched.tick()  # must not raise
        self.assertEqual(len(outcomes), 1)

    def test_health_with_no_db(self):
        ctx = SimpleNamespace(db=None, settings=SimpleNamespace(
            partner=SimpleNamespace(owner_chats="")), tools=None, extras={})
        sched = Scheduler(ctx)
        h = sched.health()  # must not raise
        self.assertIsInstance(h, dict)
        self.assertIn("running", h)
        self.assertIn("delivery", h)

    def test_redeliver_with_no_db(self):
        ctx = SimpleNamespace(db=None, settings=None, extras={})
        self.assertEqual(Notifier(ctx).redeliver(), 0)

    def test_health_never_raises_with_broken_gateway(self):
        ctx = _ctx(gateway=FakeGateway(fail_status=True))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        h = sched.health()
        self.assertIsInstance(h, dict)


class SchedulerHealthTests(unittest.TestCase):
    def test_health_reports_loop_next_and_last(self):
        gw = FakeGateway(live=("telegram",))
        ctx = _ctx(gateway=gw, owner_chats="telegram:111")
        sched = Scheduler(ctx, gateway=gw)
        job = sched.add("healthcheck-job", "every 1h", "message",
                        {"text": "ping"})
        ctx.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                       (time.time() - 1, job["id"]))
        sched.tick()
        h = sched.health()
        self.assertFalse(h["running"])  # tick loop not started in tests
        self.assertIsNotNone(h["last_tick_at"])
        self.assertEqual(h["jobs"]["total"], 1)
        self.assertEqual(h["jobs"]["enabled"], 1)
        self.assertIsNotNone(h["next_job"])
        self.assertEqual(h["next_job"]["name"], "healthcheck-job")
        self.assertIsNotNone(h["last_job"])
        self.assertTrue(h["last_job"]["ok"])
        self.assertEqual(h["delivery"]["live_owner_channels"], ["telegram"])
        # nothing stuck: queue empty, channel live → ok
        self.assertTrue(h["ok"])
        self.assertEqual(h["reasons"], [])

    def test_health_flags_stuck_queue(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        n = sched.notifier
        for i in range(6):
            n.publish("schedule", f"stuck {i}", "b")
        with patch("nomorals.core.profile.is_termux", return_value=False):
            h = sched.health()
        self.assertFalse(h["ok"])
        self.assertTrue(any("stuck retrying" in r for r in h["reasons"]))
        self.assertEqual(h["delivery"]["queue"]["retryable"], 6)

    def test_health_flags_dead_letters(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        n = sched.notifier
        res = n.publish("schedule", "doomed", "b")
        n.db.execute(
            "UPDATE notifications SET delivery_state = 'dead' WHERE id = ?",
            (res["id"],))
        with patch("nomorals.core.profile.is_termux", return_value=False):
            h = sched.health()
        self.assertFalse(h["ok"])
        self.assertTrue(any("dead-lettered" in r for r in h["reasons"]))


class RunMessageChannelTests(unittest.TestCase):
    def test_run_message_reports_channel(self):
        gw = FakeGateway(live=("telegram",))
        ctx = _ctx(gateway=gw, owner_chats="telegram:111")
        sched = Scheduler(ctx, gateway=gw)
        job = sched.add("ping", "every 1h", "message", {"text": "hello"})
        outcome = sched.run_now(job["id"])
        self.assertTrue(outcome["ok"])
        self.assertTrue(outcome["result"].startswith("sent via telegram:"))

    def test_run_message_reports_stored_state(self):
        ctx = _ctx(gateway=FakeGateway(live=()))
        sched = Scheduler(ctx, gateway=ctx.extras["gateway"])
        job = sched.add("ping", "every 1h", "message", {"text": "hello"})
        outcome = sched.run_now(job["id"])
        self.assertTrue(outcome["ok"])  # the job ran; delivery parked
        self.assertTrue(outcome["result"].startswith("stored (failed):"))


if __name__ == "__main__":
    unittest.main()
