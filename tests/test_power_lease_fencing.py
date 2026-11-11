"""PowerScheduler lease fencing (doc 18 P3 remainder).

Pins: every task ``poll()`` hands out carries a generation-fenced
claim from ``core.leases`` on ``power-task:<job_id>``, and the
terminal acts (complete / fail / cancel) revalidate the generation
at the act seam.  Two scheduler instances over ONE lease store
drive every contention case — the fence is never mocked:

(a) two instances contend: the loser's act is fenced and recorded
    as ``power.task.fenced`` (never as the work's outcome), the
    winner's work runs and lands;
(b) a generation bump between claim and act fences the act;
(c) an expired lease's act is refused even with no successor —
    a holder that cannot renew in time does not finish "one last
    job" on a lapsed claim;
(d) release + reclaim proceeds normally: the successor's result
    lands, its lease is released after, generations are never
    reused;
(e) renewal keeps a live claim alive past its original expiry,
    and a refused renewal fences the claim;
(f) an executor presenting a superseded claim's queue token is
    fenced even against its own instance's newer claim;
(g) a deferred job's claim is released with the deferral, so the
    next claim of the job starts at the next generation;
(h) single-instance behavior is unchanged: poll → complete/fail/
    cancel land as before, and completing a job this instance never
    claimed stays the administrative path.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import dataclass

from nomorals.core.events import global_bus
from nomorals.core.leases import LeaseStore
from nomorals.power.monitor import PowerMonitor
from nomorals.power.scheduler import PowerAwareScheduler
from nomorals.storage.db import Database
from nomorals.storage.migrations import MIGRATIONS
from nomorals.storage.schema import MigrationRunner

SHORT_TTL = 0.3
PAST_TTL = 0.5


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "power.db"))
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


class _ZeroRng:
    """Deterministic RNG: always the lowest possible delay."""

    def uniform(self, a, b):
        return a


@dataclass
class _FakeSample:
    battery_percent: float | None = None
    thermal_state: str | None = None
    battery_status: str = ""
    idle: bool | None = None


class _FakeManager:
    """Deterministic ResourceManager stand-in (test_power_sweep's)."""

    def __init__(self, degradation_level=0, degradation_name="full",
                 allowed_tiers=None, throttled=False, ok=True):
        self._sample = _FakeSample()
        self._advisory = {
            "throttled": throttled, "ok": ok, "reasons": [],
            "degradation_level": degradation_level,
            "degradation": {
                "level": degradation_level, "name": degradation_name,
                "allowed_tiers": allowed_tiers or
                ["critical", "important", "background", "bulk"],
            },
            "battery_forecast": {},
        }
        self.budgets = None

    def sample(self):
        return self._sample

    def consult(self, mission=None):
        return dict(self._advisory)


class _FencedEvents:
    """Collect power.task.fenced events for one test."""

    def __init__(self, test: unittest.TestCase):
        self.events: list = []
        self._sub = global_bus.subscribe(
            "power.task.fenced",
            lambda e: self.events.append(e), sync=True)
        test.addCleanup(global_bus.unsubscribe, self._sub)

    def phases(self) -> list:
        return [e.data.get("phase") for e in self.events]


def _sched(db: Database, **kw) -> PowerAwareScheduler:
    kw.setdefault("monitor", PowerMonitor())
    return PowerAwareScheduler(db, **kw)


class ContentionTests(unittest.TestCase):

    def test_loser_act_fenced_winner_lands(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        b = _sched(db)
        jid = a.dispatch("train", {}, power_class="light")
        tasks_a = a.poll("worker-a", lease_seconds=SHORT_TTL)
        self.assertEqual(len(tasks_a), 1)
        self.assertEqual(tasks_a[0].lease_generation, 1)
        time.sleep(PAST_TTL)  # A's queue lease AND claim lease lapse
        tasks_b = b.poll("worker-b")
        self.assertEqual(len(tasks_b), 1)
        self.assertEqual(tasks_b[0].job_id, jid)
        self.assertEqual(tasks_b[0].lease_generation, 2)
        # The stale holder's completion must not land.
        a.complete(jid, result="stale-result")
        row = a.queue.get(jid)
        self.assertEqual(row.status, "leased")
        self.assertEqual(len(fenced.events), 1)
        self.assertEqual(fenced.events[0].data["phase"], "complete")
        self.assertEqual(fenced.events[0].data["reason"],
                         "stale_generation")
        self.assertEqual(fenced.events[0].data["job_id"], jid)
        # The rightful holder's completion lands, with its result.
        b.complete(jid, result="winner-result")
        self.assertEqual(b.queue.get(jid).status, "done")
        self.assertEqual(b.queue.result_of(jid), "winner-result")
        self.assertEqual(len(fenced.events), 1)  # nothing else fenced

    def test_generation_bump_between_claim_and_act_fences_fail(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        jid = a.dispatch("sync", {}, power_class="light")
        task = a.poll("worker-a")[0]  # default TTL: claim stays live
        self.assertEqual(task.lease_generation, 1)
        # An outside hand releases the claim and takes the resource
        # at the next generation (the doc §6 fencing story).
        store = LeaseStore(db)
        released = store.release(task.lease_id, task.lease_generation)
        self.assertTrue(released["released"])
        granted = store.acquire(task.lease_resource, "intruder", 600)
        self.assertTrue(granted["granted"])
        self.assertEqual(granted["lease"]["generation"], 2)
        # A's late failure must not requeue/dead-letter the job.
        a.fail(jid, error="stale boom")
        row = a.queue.get(jid)
        self.assertEqual(row.status, "leased")
        self.assertEqual(fenced.phases(), ["fail"])
        self.assertEqual(fenced.events[0].data["reason"],
                         "stale_generation")

    def test_expired_lease_act_refused_without_successor(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        jid = a.dispatch("backup", {}, power_class="light")
        a.poll("worker-a", lease_seconds=SHORT_TTL)
        time.sleep(PAST_TTL)  # lapsed; nobody else ever claims it
        a.complete(jid, result="too-late")
        row = a.queue.get(jid)
        self.assertNotEqual(row.status, "done")
        self.assertEqual(fenced.phases(), ["complete"])
        self.assertEqual(fenced.events[0].data["reason"], "expired")
        # A fenced claim stays fenced: a second act does not fall
        # through to the unfenced administrative path.
        a.fail(jid, error="too-late")
        self.assertNotEqual(a.queue.get(jid).status, "dead")
        self.assertEqual(fenced.phases(), ["complete", "fail"])

    def test_reclaim_after_expiry_proceeds_and_releases(self):
        db = _file_db(self)
        a = _sched(db)
        b = _sched(db)
        jid = a.dispatch("report", {}, power_class="light")
        task_a = a.poll("worker-a", lease_seconds=SHORT_TTL)[0]
        time.sleep(PAST_TTL)
        task_b = b.poll("worker-b")[0]
        self.assertEqual(task_b.lease_generation, 2)
        b.complete(jid, result="b-result")
        self.assertEqual(b.queue.get(jid).status, "done")
        self.assertEqual(b.queue.result_of(jid), "b-result")
        store = LeaseStore(db)
        # The winner's lease was released by its landed act...
        self.assertIsNone(store.current(task_a.lease_resource))
        # ...and the history keeps both generations, never reused.
        history = store.leases_for_resource(
            task_a.lease_resource, include_history=True)
        self.assertEqual([h["generation"] for h in history], [2, 1])
        self.assertEqual(history[0]["state"], "released")
        self.assertEqual(history[1]["state"], "expired")


class RenewalTests(unittest.TestCase):

    def test_renew_keeps_claim_alive_past_original_expiry(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        jid = a.dispatch("long-job", {}, power_class="light")
        a.poll("worker-a", lease_seconds=0.4)
        self.assertTrue(a.renew_claim(jid, extra_seconds=60.0))
        time.sleep(0.55)  # past the ORIGINAL claim/queue expiry
        a.complete(jid, result="finished")
        self.assertEqual(a.queue.get(jid).status, "done")
        self.assertEqual(fenced.events, [])

    def test_renew_refused_after_takeover_fences_claim(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        b = _sched(db)
        jid = a.dispatch("train", {}, power_class="light")
        a.poll("worker-a", lease_seconds=SHORT_TTL)
        time.sleep(PAST_TTL)
        b.poll("worker-b")
        self.assertFalse(a.renew_claim(jid))
        self.assertEqual(fenced.phases(), ["renew"])
        # A's later act fences too; B's lands.
        a.complete(jid, result="stale")
        self.assertEqual(fenced.phases(), ["renew", "complete"])
        b.complete(jid, result="b")
        self.assertEqual(b.queue.get(jid).status, "done")

    def test_renew_unclaimed_job_is_false(self):
        db = _file_db(self)
        a = _sched(db)
        jid = a.dispatch("never-polled", {}, power_class="light")
        self.assertFalse(a.renew_claim(jid))


class SupersededTokenTests(unittest.TestCase):

    def test_superseded_claim_token_fences_act(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        a = _sched(db)
        jid = a.dispatch("sync", {}, power_class="light")
        first = a.poll("worker-a", lease_seconds=SHORT_TTL)[0]
        self.assertTrue(first.lease_token)
        time.sleep(PAST_TTL)
        # The same instance re-claims its own lapsed job: a NEW
        # claim (generation 2, fresh queue token) supersedes.
        second = a.poll("worker-a")[0]
        self.assertEqual(second.lease_generation, 2)
        self.assertNotEqual(second.lease_token, first.lease_token)
        # The first executor's act, pinned to its superseded token,
        # is fenced; the current claim's act lands.
        a.complete(jid, result="first", lease_token=first.lease_token)
        self.assertEqual(a.queue.get(jid).status, "leased")
        self.assertEqual(fenced.phases(), ["complete"])
        a.complete(jid, result="second",
                   lease_token=second.lease_token)
        self.assertEqual(a.queue.get(jid).status, "done")
        self.assertEqual(a.queue.result_of(jid), "second")


class DeferralReleasesClaimTests(unittest.TestCase):

    def test_deferred_job_claim_released_next_claim_next_generation(
            self):
        db = _file_db(self)
        shedding = PowerMonitor(sampler=lambda: _FakeManager(
            degradation_level=3, degradation_name="critical",
            allowed_tiers=["critical"]))
        a = PowerAwareScheduler(db, monitor=shedding,
                                backoff_rng=_ZeroRng())
        jid = a.dispatch("bulk-job", {}, power_class="light",
                         tier="bulk")
        self.assertEqual(a.poll("worker-a"), [])  # tier-shed deferral
        store = LeaseStore(db)
        resource = f"power-task:{jid}"
        # The deferral returned the job AND released the claim taken
        # for it — nothing of A's fences the next claim.
        self.assertIsNone(store.current(resource))
        history = store.leases_for_resource(resource,
                                            include_history=True)
        self.assertEqual([h["generation"] for h in history], [1])
        self.assertEqual(history[0]["state"], "released")
        # A permissive instance claims it at the NEXT generation.
        b = _sched(db)
        tasks = b.poll("worker-b")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].lease_generation, 2)
        b.complete(jid)
        self.assertEqual(b.queue.get(jid).status, "done")


class SingleInstanceParityTests(unittest.TestCase):

    def test_poll_complete_fail_cancel_land_as_before(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        s = _sched(db)
        # complete after poll
        jid = s.dispatch("t", {}, power_class="light")
        task = s.poll()[0]
        self.assertEqual(task.lease_generation, 1)
        self.assertEqual(task.lease_resource, f"power-task:{jid}")
        self.assertTrue(task.lease_id)
        self.assertTrue(task.lease_token)
        s.complete(task.job_id)
        self.assertEqual(s.queue.get(jid).status, "done")
        # the landed act released the claim
        self.assertIsNone(LeaseStore(db).current(task.lease_resource))
        # fail with retry returns the job to the pool
        jid2 = s.dispatch("t2", {}, power_class="light")
        s.poll()
        s.fail(jid2, error="boom")
        self.assertEqual(s.queue.get(jid2).status, "ready")
        # cancel moves a claimed job to dead
        jid3 = s.dispatch("t3", {}, power_class="light")
        s.poll()
        s.cancel(jid3)
        self.assertEqual(s.queue.get(jid3).status, "dead")
        self.assertEqual(fenced.events, [])

    def test_complete_unclaimed_job_stays_administrative(self):
        db = _file_db(self)
        fenced = _FencedEvents(self)
        s = _sched(db)
        jid = s.dispatch("probe_task", {}, power_class="light")
        s.complete(jid)  # never polled: no claim, no fence
        self.assertEqual(s.queue.get(jid).status, "done")
        self.assertEqual(fenced.events, [])


if __name__ == "__main__":
    unittest.main()
