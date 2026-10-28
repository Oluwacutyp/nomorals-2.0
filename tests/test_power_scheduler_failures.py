"""Pins for nomorals/power/scheduler.py (tail-17): behavior + NOT-recording.

The power package has no failures ledger and scheduler.py records
nothing, so NOT-recording is pinned against the module logger ``_log``
(tail-7/tail-10/tail-11 precedent): ``scheduler._log`` is patched with a
MagicMock around each trigger and ``method_calls`` must stay empty.

This file contains NO except handlers (AST-checked): every trigger is
either a non-raising call or a mock side_effect raised inside the
target's own handler, never in this file.
"""
import unittest
from unittest import mock

from nomorals.power import scheduler as sched_mod
from nomorals.power.monitor import PowerMonitor
from nomorals.power.scheduler import PowerAwareScheduler
from nomorals.storage.db import Database
from nomorals.storage.migrations import MIGRATIONS
from nomorals.storage.schema import MigrationRunner


def _db():
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


def _sched(resources=None):
    return PowerAwareScheduler(_db(), monitor=PowerMonitor(),
                               resources=resources)


class _RaisingBudgets:
    def acquire(self, *a, **k):
        raise RuntimeError("budgets boom")

    def release(self, *a, **k):
        raise RuntimeError("budgets boom")

    def snapshot(self, *a, **k):
        raise RuntimeError("budgets boom")


class _Budgets:
    def __init__(self):
        self.released = []

    def acquire(self, *a, **k):
        return False

    def release(self, sub, **k):
        self.released.append(sub)

    def snapshot(self, *a, **k):
        return {"llm": {"limit": 1}}


class _RaisingProperty:
    @property
    def budgets(self):
        raise RuntimeError("property boom")


class _StubDbQuery:
    def query(self, *a, **k):
        raise RuntimeError("db boom")


class SchedulerFailurePins(unittest.TestCase):
    def test_178_budgets_property_failure_is_none(self):
        s = _sched(resources=_RaisingProperty())
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertIsNone(s._budgets())
            self.assertEqual(log.method_calls, [])
        b = _Budgets()
        s2 = _sched(resources=type("R", (), {"budgets": b})())
        self.assertIs(s2._budgets(), b)

    def test_190_acquire_failure_fails_open(self):
        s = _sched(resources=type("R", (), {"budgets": _RaisingBudgets()})())
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertIs(s.try_acquire("llm"), True)
            self.assertEqual(log.method_calls, [])
        s2 = _sched(resources=type("R", (), {"budgets": _Budgets()})())
        self.assertIs(s2.try_acquire("llm"), False)
        self.assertIs(_sched(resources=None).try_acquire("llm"), True)

    def test_200_release_failure_is_silent_none(self):
        s = _sched(resources=type("R", (), {"budgets": _RaisingBudgets()})())
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertIsNone(s.release_budget("llm"))
            self.assertEqual(log.method_calls, [])
        b = _Budgets()
        s2 = _sched(resources=type("R", (), {"budgets": b})())
        s2.release_budget("llm")
        self.assertEqual(b.released, ["llm"])

    def test_207_snapshot_failure_is_empty_dict(self):
        s = _sched(resources=type("R", (), {"budgets": _RaisingBudgets()})())
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertEqual(s.budget_snapshot(), {})
            self.assertEqual(log.method_calls, [])
        b = _Budgets()
        s2 = _sched(resources=type("R", (), {"budgets": b})())
        self.assertEqual(s2.budget_snapshot(), {"llm": {"limit": 1}})

    def test_476_release_failure_does_not_block_completion(self):
        s = _sched(resources=type("R", (), {"budgets": _RaisingBudgets()})())
        jid = s.dispatch("probe_task", subsystem="llm")
        with mock.patch.object(sched_mod, "_log") as log:
            s.complete(jid)
            self.assertEqual(log.method_calls, [])
        self.assertEqual(s.queue.get(jid).status, "done")
        b = _Budgets()
        s2 = _sched(resources=type("R", (), {"budgets": b})())
        jid2 = s2.dispatch("probe_task", subsystem="llm")
        s2.complete(jid2)
        self.assertEqual(b.released, ["llm"])

    def test_517_energy_ledger_failure_is_advisory(self):
        s = _sched()
        jid = s.dispatch("probe_task")
        with mock.patch.object(s.energy, "record",
                               side_effect=RuntimeError("ledger boom")):
            with mock.patch.object(sched_mod, "_log") as log:
                self.assertIsNone(s._record_energy(jid))
                self.assertEqual(log.method_calls, [])
        s._record_energy(jid)
        self.assertIsInstance(s.energy.summary(), dict)

    def test_535_deferred_count_failure_is_zero(self):
        s = _sched()
        s.db = _StubDbQuery()
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertEqual(s._deferred_count("power:heavy"), 0)
            self.assertEqual(log.method_calls, [])
        self.assertEqual(_sched()._deferred_count("power:heavy"), 0)

    def test_576_deferred_tasks_failure_is_empty(self):
        s = _sched()
        s.db = _StubDbQuery()
        with mock.patch.object(sched_mod, "_log") as log:
            self.assertEqual(s.deferred_tasks(), [])
            self.assertEqual(log.method_calls, [])
        s2 = _sched()
        jid = s2.dispatch("probe_task")
        self.assertTrue(any(w["job_id"] == jid for w in s2.deferred_tasks()))

    def test_610_per_topic_pending_failure_is_zero_for_that_topic(self):
        s = _sched()
        orig = s.queue.pending

        def pending(topic):
            if topic == "power:heavy":
                raise RuntimeError("pending boom")
            return orig(topic)

        with mock.patch.object(s.queue, "pending", side_effect=pending):
            with mock.patch.object(sched_mod, "_log") as log:
                st = s.stats()
                self.assertEqual(log.method_calls, [])
        self.assertEqual(st["pending"]["heavy"], 0)
        self.assertIsInstance(st["pending"]["light"], int)

    def test_623_stats_failure_is_fallback_dict(self):
        s = _sched()
        with mock.patch.object(s.energy, "summary",
                               side_effect=RuntimeError("summary boom")):
            with mock.patch.object(sched_mod, "_log") as log:
                st = s.stats()
                self.assertEqual(log.method_calls, [])
        self.assertEqual(st, {"pending": {}, "deferred_heavy": 0,
                              "deferred_medium": 0, "budgets": {}})
        st2 = _sched().stats()
        self.assertEqual(set(st2["pending"]), {"light", "medium", "heavy"})


if __name__ == "__main__":
    unittest.main()
