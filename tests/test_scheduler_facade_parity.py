"""Facade behaviour-parity harness — Echelon Wave 5.

The Wave 5 consolidation wires Devon's scheduling surfaces behind
``nomorals.scheduler.facade.SchedulerFacade`` WITHOUT behaviour
change.  This suite is the proof instrument: every operation runs
twice — once on the incumbent directly, once through the facade on
a twin store — and the outcomes must agree:

* spec parsing: with the agents parser injected, the facade's
  ``parse_spec`` is the incumbent parser — same result, or the same
  refusal, on every corpus spec;
* agent jobs: ``add`` outcomes, stored policy fields, policy
  validation, and missed-fire outcomes (``fire_now`` vs ``skip``
  vs ``next_only``) agree on twin DBs; the overlap policy
  round-trips identically;
* unified jobs: first ``next_run`` computation and missed-policy
  due verdicts (``catch_up_once`` / ``skip`` / ``backfill``) agree
  exactly on twin in-memory schedulers (jitter pinned to 0 so the
  comparison is exact, not statistical);
* cron honesty: impossible expressions and unknown zones are
  refused BY NAME through the facade and at every schedule-time
  surface (unified job construction, user-facing ``schedule_cron``).
"""

from __future__ import annotations

import asyncio
import random
import time
import unittest
from types import SimpleNamespace

from nomorals.agents import scheduler as agent_mod
from nomorals.scheduler import unified_scheduler as unified_mod
from nomorals.scheduler.facade import SchedulerFacade
from nomorals.scheduler.recurrence import (
    CronNeverMatchesError,
    CronTimezoneError,
)
from nomorals.storage.db import Database


# ── agents Scheduler twin rig (idiom from test_scheduler_deep) ──────

def _ctx() -> SimpleNamespace:
    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw: SimpleNamespace(
            ok=True, value="ok", error=None)
    )
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(), extras={})


def _agent_pair() -> tuple[agent_mod.Scheduler, SchedulerFacade]:
    direct = agent_mod.Scheduler(_ctx())
    injected = agent_mod.Scheduler(_ctx())
    facade = SchedulerFacade(agent_scheduler=injected,
                             spec_parser=agent_mod.parse_schedule_spec)
    return direct, facade


def _row(sched: agent_mod.Scheduler, job_id: str) -> dict:
    for row in sched.list_jobs():
        if row["id"] == job_id:
            return row
    raise AssertionError(f"job {job_id} not in list_jobs()")


def _stale(sched: agent_mod.Scheduler, job_id: str,
           seconds: float = 7200.0) -> None:
    sched.db.execute(
        "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
        (time.time() - seconds, job_id),
    )


def _outcome(fn, *args, **kwargs):
    """('ok', result) or ('raise', exception-type-name) — parity
    compares refusals as first-class outcomes."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - compared by type name
        return ("raise", type(exc).__name__)


class ParseSpecParityTests(unittest.TestCase):
    CORPUS = [
        "every 30m", "every 2h", "every 1d", "30m", "2h",
        "daily 09:30", "daily 22:00 America/New_York",
        "weekly MON 08:00", "monthly 15 07:45", "yearly 6 1 0:00",
        "cron 0 9 * * MON-FRI", "cron */15 * * * * Europe/London",
        "at 2030-01-01 09:00", "at 2030-06-15T18:30:00+00:00",
        "RRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=30",
        "daily 09:00 Europe/Berlin",
        "flurb the wizzle", "", "every blue moon",
    ]

    def test_injected_parser_matches_incumbent_exactly(self):
        _, facade = _agent_pair()
        for spec in self.CORPUS:
            with self.subTest(spec=spec):
                self.assertEqual(
                    _outcome(facade.parse_spec, spec),
                    _outcome(agent_mod.parse_schedule_spec, spec))

    def test_uninjected_facade_is_minimal_and_named(self):
        """Without injection the facade keeps its documented minimal
        subset — and the missing agent scheduler refuses by name
        instead of silently diverging."""
        facade = SchedulerFacade()
        kind, payload = facade.parse_spec("every 30m")
        self.assertEqual(kind, "interval")  # documented local shape
        self.assertEqual(payload["seconds"], 1800)
        with self.assertRaises(RuntimeError) as cm:
            _ = facade.agent_scheduler
        self.assertIn("agent_scheduler", str(cm.exception))


class AgentJobParityTests(unittest.TestCase):
    def test_add_outcomes_and_stored_fields_agree(self):
        direct, facade = _agent_pair()
        cases = [
            ("standup", "every 30m", "message", {"text": "hi"}, {}),
            ("report", "cron 0 9 * * MON-FRI", "message",
             {"text": "r"}, {"timezone": "Africa/Lagos"}),
            ("once", "at 2030-01-01 09:00", "message", {"text": "o"},
             {}),
            ("pol", "every 1h", "message", {"text": "p"},
             {"missed_fire_policy": "skip", "overlap_policy": "queue"}),
        ]
        for name, spec, kind, payload, kw in cases:
            with self.subTest(name=name):
                a = direct.add(name, spec, kind, payload, **kw)
                b = facade.add_agent_job(name, spec, kind, payload, **kw)
                self.assertEqual(a["name"], b["name"])
                self.assertEqual(a["kind"], b["kind"])
                self.assertAlmostEqual(a["next_run"], b["next_run"],
                                       delta=2.0)
                ra, rb = _row(direct, a["id"]), _row(
                    facade.agent_scheduler, b["id"])
                for field in ("kind", "spec", "missed_fire_policy",
                              "overlap_policy", "timezone", "enabled",
                              "payload_kind"):
                    self.assertEqual(ra[field], rb[field], field)

    def test_policy_validation_agrees(self):
        direct, facade = _agent_pair()
        for kw in ({"missed_fire_policy": "nope"},
                   {"overlap_policy": "nope"},
                   {"backoff": "nope"},
                   {"depends_policy": "nope"}):
            with self.subTest(kw=kw):
                self.assertEqual(
                    _outcome(direct.add, "x", "every 1h", "message",
                             {"text": "x"}, **kw),
                    _outcome(facade.add_agent_job, "x", "every 1h",
                             "message", {"text": "x"}, **kw))

    def test_missed_fire_outcomes_agree(self):
        """The plan's named parity surface: fire_now | skip |
        next_only must behave identically via facade and direct."""
        for policy in ("fire_now", "skip", "next_only"):
            with self.subTest(policy=policy):
                direct, facade = _agent_pair()
                a = direct.add("j", "every 1h", "message",
                               {"text": "x"}, missed_fire_policy=policy)
                b = facade.add_agent_job(
                    "j", "every 1h", "message", {"text": "x"},
                    missed_fire_policy=policy)
                _stale(direct, a["id"])
                _stale(facade.agent_scheduler, b["id"])
                out_a = direct.tick()
                out_b = facade.agent_scheduler.tick()
                self.assertEqual(len(out_a), len(out_b))
                if policy == "fire_now":
                    self.assertEqual(len(out_a), 1)
                else:
                    self.assertEqual(len(out_a), 0)
                ra = _row(direct, a["id"])
                rb = _row(facade.agent_scheduler, b["id"])
                # next fire stepped forward the same way (same window)
                self.assertAlmostEqual(ra["next_run"], rb["next_run"],
                                       delta=2.0)
                self.assertGreater(ra["next_run"], time.time())

    def test_remove_and_enable_agree(self):
        direct, facade = _agent_pair()
        a = direct.add("j", "every 1h", "message", {"text": "x"})
        b = facade.add_agent_job("j", "every 1h", "message",
                                 {"text": "x"})
        facade.set_agent_enabled(b["id"], False)
        self.assertFalse(
            _row(facade.agent_scheduler, b["id"])["enabled"])
        self.assertTrue(facade.remove_agent_job(b["id"]))
        remaining = {r["id"] for r in facade.list_agent_jobs()}
        self.assertNotIn(b["id"], remaining)
        self.assertEqual(len(facade.list_agent_jobs()),
                         len(direct.list_jobs()) - 1)
        self.assertIsNotNone(_row(direct, a["id"]))


# ── unified scheduler parity ─────────────────────────────────────────

NOW = 1_767_225_600.0  # fixed footing: 2026-01-01T00:00:00Z


def _unified_job(job_id: str, recurrence: dict, **kw) -> unified_mod.UnifiedJob:
    kw.setdefault("jitter_frac", 0.0)  # exact comparisons, no dice
    return unified_mod.UnifiedJob(
        job_id=job_id, name=job_id,
        kind=str(recurrence.get("type") or "cron"),
        recurrence=recurrence, created_at=NOW, **kw)


def _unified_pair():
    direct = unified_mod.UnifiedScheduler()
    via = unified_mod.UnifiedScheduler()
    facade = SchedulerFacade(unified=via)
    return direct, facade, via


def _seed_unified(sched_add, policy: str = "catch_up_once"):
    """Three jobs through an add callable with identical seeds."""
    jobs = []
    specs = [
        ("iv", {"type": "interval", "spec": 3600}, {}),
        ("cr", {"type": "cron", "spec": "0 9 * * *"},
         {"timezone": "Africa/Lagos"}),
        ("anchor", {"type": "cron", "spec": "0 6 * * *"},
         {"timezone": "Africa/Lagos"}),
    ]
    for job_id, recurrence, kw in specs:
        job = _unified_job(job_id, recurrence, missed_policy=policy,
                           **kw)
        jobs.append(sched_add(job, now=NOW, rng=random.Random(42)))
    return jobs


class UnifiedParityTests(unittest.TestCase):
    def test_next_run_agrees_exactly(self):
        direct, facade, _via = _unified_pair()
        jobs_a = _seed_unified(direct.add_job)
        jobs_b = _seed_unified(facade.add_job)
        for a, b in zip(jobs_a, jobs_b):
            self.assertEqual(a.next_run, b.next_run, a.job_id)
            self.assertEqual(a.recurrence, b.recurrence)
        self.assertEqual(
            [j.job_id for j in facade.list_jobs()],
            [j.job_id for j in direct.list_jobs()])

    def test_missed_policy_verdicts_agree(self):
        for policy in ("catch_up_once", "skip", "backfill"):
            with self.subTest(policy=policy):
                direct, facade, via = _unified_pair()
                _seed_unified(direct.add_job, policy)
                _seed_unified(facade.add_job, policy)
                later = NOW + 5 * 3600  # five hours on
                due_a = direct.due_jobs(now=later)
                due_b = facade.due_jobs(now=later)
                self.assertEqual(
                    [(d["job"].job_id, d["action"], d["fire"])
                     for d in due_a],
                    [(d["job"].job_id, d["action"], d["fire"])
                     for d in due_b])
                after_a = {j.job_id: j.next_run
                           for j in direct.list_jobs()}
                after_b = {j.job_id: j.next_run for j in via.list_jobs()}
                self.assertEqual(after_a, after_b)
                if policy == "skip":
                    self.assertTrue(
                        all(d["action"] == "skipped" for d in due_a))
                if policy == "backfill":
                    iv = next(d for d in due_a
                              if d["job"].job_id == "iv")
                    self.assertEqual(iv["action"], "backfill")
                    self.assertEqual(len(iv["fire"]), 5)

    def test_missed_policy_round_trip(self):
        direct, facade, _via = _unified_pair()
        a = _unified_job("j", {"type": "interval", "spec": 60},
                         missed_policy="backfill")
        b = _unified_job("j", {"type": "interval", "spec": 60},
                         missed_policy="backfill")
        direct.add_job(a, now=NOW)
        facade.add_job(b, now=NOW)
        self.assertEqual(facade.get_job("j").missed_policy, "backfill")
        self.assertEqual(facade.get_job("j").next_run, a.next_run)
        with self.assertRaises(ValueError):
            _unified_job("bad", {"type": "interval", "spec": 60},
                         missed_policy="nope")


# ── cron honesty at every schedule-time surface ──────────────────────

class CronHonestyTests(unittest.TestCase):
    def test_facade_passthroughs(self):
        facade = SchedulerFacade()
        from datetime import datetime, timezone
        pivot = datetime(2025, 3, 1, tzinfo=timezone.utc)
        got = facade.next_cron("0 0 29 2 *", pivot)
        self.assertEqual(
            datetime.fromtimestamp(got, timezone.utc),
            datetime(2028, 2, 29, tzinfo=timezone.utc))
        with self.assertRaises(CronNeverMatchesError):
            facade.validate_cron("0 0 31 2 *")
        with self.assertRaises(CronTimezoneError):
            facade.validate_cron("0 9 * * *", "America/Denverr")

    def test_unified_job_refuses_impossible_at_construction(self):
        with self.assertRaises(CronNeverMatchesError):
            _unified_job("x", {"type": "cron", "spec": "0 0 31 2 *"})
        with self.assertRaises(CronNeverMatchesError):
            _unified_job("x", {"type": "cron", "spec": "0 0 30 2 *"},
                         missed_policy="skip")
        with self.assertRaises(CronTimezoneError):
            _unified_job("x", {"type": "cron", "spec": "0 9 * * *"},
                         timezone="America/Denverr")
        with self.assertRaises(ValueError) as cm:
            _unified_job("x", {"type": "cron", "spec": "banana"})
        self.assertNotIsInstance(cm.exception, CronNeverMatchesError)
        # Feb 29 is satisfiable and must construct fine.
        job = _unified_job("x", {"type": "cron", "spec": "0 0 29 2 *"})
        self.assertEqual(job.recurrence["type"], "cron")

    def test_user_facing_schedule_cron_refuses_by_name(self):
        from nomorals.scheduler.scheduler import Scheduler as UserSched

        db = Database(":memory:")
        sched = UserSched(db)

        async def noop(**kwargs):
            return None

        sched.register_action("noop", noop)
        with self.assertRaises(CronNeverMatchesError):
            asyncio.run(sched.schedule_cron(
                "impossible", "0 0 31 2 *", "noop", {}))
        with self.assertRaises(CronTimezoneError):
            asyncio.run(sched.schedule_cron(
                "badzone", "0 9 * * *", "noop", {},
                tz="America/Denverr"))
        job = asyncio.run(sched.schedule_cron(
            "leap", "0 0 29 2 *", "noop", {}))
        self.assertIsNotNone(job)
        self.assertEqual(job.cron_expr, "0 0 29 2 *")


if __name__ == "__main__":
    unittest.main()
