"""Doc 15 P4 — outcome cases + the steward core.

Pins: (a) an unknown-outcome operation opens exactly one case across
two sweeps; (b) a case whose domain records establish the outcome
resolves with the evidence attached, through a real delegation
contract that completed; (c) the unknown discipline — no evidence,
no resolution, and no store API can force one; (d) generation
fencing fences stale workers; (e) the steward never calls the
existing mechanisms' mutating paths — it observes records only.
"""

from __future__ import annotations

import os
import threading
import time
import unittest

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.outcome_cases import (
    CaseError,
    OutcomeCaseStore,
    StaleCaseGeneration,
)
from nomorals.agents.outcome_steward import (
    OutcomeSteward,
    ensure_outcome_steward_job,
    inspect_case,
    render_case,
    run_stewardship_cycle,
)
from nomorals.core.jobs import JobRegistry
from nomorals.missions.mission import (
    Mission,
    MissionStatus,
    MissionStore,
)
from nomorals.storage.db import Database

DEAD_PID = 999999


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _mission(store: MissionStore, *, status: str, state=None,
             age_s: float = 0.0, goal: str = "do the thing") -> Mission:
    now = time.time()
    mission = Mission(
        goal=goal, status=status, state=dict(state or {}),
        created_at=now - age_s, updated_at=now - age_s)
    return store.create(mission)


def _dead_heartbeat(age_s: float = 100000.0) -> dict:
    return {"heartbeat": {"at": time.time() - age_s, "pid": DEAD_PID}}


def _liveness_failed(store: MissionStore) -> Mission:
    state = _dead_heartbeat()
    state["last_error"] = ("runner died — no heartbeat for 27h "
                           "(pid 999999 is gone)")
    state["status_note"] = f"reconciled: {state['last_error']}"
    return _mission(store, status=MissionStatus.FAILED, state=state,
                    goal="render the report")


class CaseStoreDisciplineTests(unittest.TestCase):
    def setUp(self):
        self.cases = OutcomeCaseStore(_db())

    def _open(self):
        case, created = self.cases.open_case(
            domain="missions", kind="mission", subject="a mission",
            subject_key="mission:m-1", reason="unknown-outcome",
            detail={"mission_id": "m-1"})
        assert created
        return case

    def test_resolve_requires_real_evidence_on_this_case(self):
        case = self._open()
        case, generation = self.cases.claim(case.id)
        # A fabricated receipt id is refused.
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id="ocr-nope",
                               generation=generation)
        # A non-evidence receipt (the run-claimed one) is refused.
        claimed = [r for r in self.cases.receipts(case.id)
                   if r.kind == "run-claimed"][0]
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id=claimed.id,
                               generation=generation)
        # outcome_unknown is not a resolution, even with evidence.
        evidence = self.cases.add_receipt(
            case.id, kind="evidence", source="missions.store",
            note="record read", generation=generation)
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="outcome_unknown",
                               evidence_receipt_id=evidence.id,
                               generation=generation)
        # An evidence receipt from ANOTHER case is refused.
        other = self._open_other()
        other_evidence = self.cases.add_receipt(
            other.id, kind="evidence", source="missions.store",
            note="other case's evidence")
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id=other_evidence.id,
                               generation=generation)
        # The honest path works and is receipted.
        closed = self.cases.resolve(case.id, verdict="failed",
                                    evidence_receipt_id=evidence.id,
                                    generation=generation)
        self.assertEqual(closed.state, "resolved")
        self.assertEqual(closed.verdict, "failed")
        kinds = [r.kind for r in self.cases.receipts(case.id)]
        self.assertEqual(kinds[0], "opened")
        self.assertIn("resolution", kinds)
        # Terminal cases refuse every further transition.
        with self.assertRaises(CaseError):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id=evidence.id,
                               generation=generation)
        with self.assertRaises(CaseError):
            self.cases.close_unknown(case.id, note="again",
                                     generation=generation)

    def _open_other(self):
        case, _ = self.cases.open_case(
            domain="missions", kind="mission", subject="another",
            subject_key="mission:m-2", reason="orphaned",
            detail={"mission_id": "m-2"})
        return case

    def test_fencing_stale_generation_cannot_write_or_transition(self):
        case = self._open()
        _case, gen1 = self.cases.claim(case.id)
        case, gen2 = self.cases.claim(case.id)
        self.assertEqual(gen2, gen1 + 1)
        # Stale-generation receipt: refused.
        with self.assertRaises(StaleCaseGeneration):
            self.cases.add_receipt(case.id, kind="evidence",
                                   source="missions.store", note="stale",
                                   generation=gen1)
        evidence = self.cases.add_receipt(
            case.id, kind="evidence", source="missions.store",
            note="fresh evidence", generation=gen2)
        # Stale-generation transition: refused, state untouched.
        with self.assertRaises(StaleCaseGeneration):
            self.cases.resolve(case.id, verdict="failed",
                               evidence_receipt_id=evidence.id,
                               generation=gen1)
        self.assertEqual(self.cases.get(case.id).state, "investigating")
        # The owning generation resolves fine.
        closed = self.cases.resolve(case.id, verdict="failed",
                                    evidence_receipt_id=evidence.id,
                                    generation=gen2)
        self.assertEqual(closed.state, "resolved")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.missions = MissionStore(self.db)
        self.registry = JobRegistry()
        self.steward = OutcomeSteward(self.db, registry=self.registry)

    def test_unknown_outcome_mission_one_case_across_two_sweeps(self):
        _liveness_failed(self.missions)
        first = self.steward.sweep()
        self.assertEqual(first["opened"], 1)
        second = self.steward.sweep()
        self.assertEqual(second["opened"], 0)
        self.assertEqual(second["existing"], 1)
        self.assertEqual(len(self.steward.cases.list_cases()), 1)

    def test_orphaned_running_mission_discovered_live_one_not(self):
        _mission(self.missions, status=MissionStatus.RUNNING,
                 state=_dead_heartbeat(), goal="orphaned work")
        live_state = {"heartbeat": {"at": time.time(), "pid": os.getpid()}}
        _mission(self.missions, status=MissionStatus.RUNNING,
                 state=live_state, goal="healthy work")
        report = self.steward.sweep()
        self.assertEqual(report["opened"], 1)
        case = self.steward.cases.list_cases()[0]
        self.assertEqual(case.reason, "orphaned")

    def test_genuine_failure_and_done_open_no_case(self):
        _mission(self.missions, status=MissionStatus.FAILED,
                 state={"last_error": "step 2 raised ValueError"},
                 goal="honestly failed work")
        _mission(self.missions, status=MissionStatus.DONE,
                 goal="finished work")
        report = self.steward.sweep()
        self.assertEqual(report["candidates"], 0)

    def test_stalled_pending_mission_discovered(self):
        _mission(self.missions, status=MissionStatus.PENDING,
                 age_s=90000.0, goal="never started")
        report = self.steward.sweep()
        self.assertEqual(report["opened"], 1)
        self.assertEqual(self.steward.cases.list_cases()[0].reason,
                         "stalled")


class StewardRunTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.missions = MissionStore(self.db)
        self.registry = JobRegistry()
        self.steward = OutcomeSteward(self.db, registry=self.registry)

    def _done_uncollected_job(self):
        job = self.registry.submit(lambda: {"report": "ok"},
                                   tool="render")
        self.registry.collect(job.id, timeout=10)
        job.ended_at = time.time() - 700.0
        return job

    def _presumed_dead_job(self, fn):
        job = self.registry.submit(fn, tool="upload",
                                   heartbeat_interval_s=1.0)
        # The heartbeat thread keeps refreshing last_heartbeat_at
        # while the worker lives; backdate + reap until the
        # dead-man's switch wins a tick.
        for _ in range(100):
            job.last_heartbeat_at = time.time() - 100.0
            self.registry.reap()
            if job.status == "presumed-dead":
                break
            time.sleep(0.02)
        assert job.status == "presumed-dead"
        return job

    def test_uncollected_job_resolves_succeeded_through_contract(self):
        self._done_uncollected_job()
        self.steward.sweep()
        case = self.steward.cases.list_cases()[0]
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "succeeded")
        # The cited evidence is a real evidence receipt on this case.
        cited = self.steward.cases.get_receipt(closed.evidence_receipt_id)
        self.assertIsNotNone(cited)
        self.assertEqual(cited.kind, "evidence")
        self.assertEqual(cited.case_id, case.id)
        # The delegation contract completed (verifier-gated).
        delegation = DelegationStore(self.db)
        contract = delegation.get_contract(closed.contract_id)
        self.assertIsNotNone(contract)
        self.assertEqual(contract.status, "completed")
        self.assertNotEqual(contract.worker, contract.verifier)
        view = inspect_case(self.steward.cases, delegation, case.id)
        self.assertEqual(view["contract"]["status"], "completed")

    def test_mission_that_completed_resolves_from_its_record(self):
        mission = _mission(self.missions, status=MissionStatus.PENDING,
                           age_s=90000.0, goal="slow to start")
        self.steward.sweep()
        case = self.steward.cases.list_cases()[0]
        mission.status = MissionStatus.DONE
        self.missions.save(mission)
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(self.steward.cases.get(case.id).verdict,
                         "succeeded")

    def test_orphaned_mission_closes_unknown_and_domain_untouched(self):
        mission = _mission(self.missions, status=MissionStatus.RUNNING,
                           state=_dead_heartbeat(), goal="lost runner")
        self.steward.sweep()
        case = self.steward.cases.list_cases()[0]
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "closed-unknown")
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "outcome_unknown")
        kinds = [r.kind for r in self.steward.cases.receipts(case.id)]
        self.assertIn("evidence", kinds)
        self.assertIn("closed-unknown", kinds)
        # The steward observed the mission; it did not reconcile it.
        self.assertEqual(self.missions.get(mission.id).status,
                         MissionStatus.RUNNING)

    def test_liveness_failure_is_not_adopted_as_the_verdict(self):
        _liveness_failed(self.missions)
        self.steward.sweep()
        case = self.steward.cases.list_cases()[0]
        result = self.steward.run_case(case.id)
        # reconcile() said FAILED from liveness alone; the steward's
        # case must NOT resolve to failed on that basis (G6).
        self.assertEqual(result["outcome"], "closed-unknown")
        self.assertEqual(self.steward.cases.get(case.id).verdict,
                         "outcome_unknown")

    def test_presumed_dead_job_without_result_stays_unknown(self):
        gate = threading.Event()
        job = self._presumed_dead_job(lambda: gate.wait(30))
        try:
            self.steward.sweep()
            case = self.steward.cases.list_cases()[0]
            self.assertEqual(case.reason, "unknown-outcome")
            result = self.steward.run_case(case.id)
            self.assertEqual(result["outcome"], "closed-unknown")
        finally:
            gate.set()
            self.registry.collect(job.id, timeout=10)

    def test_presumed_dead_job_late_result_resolves_succeeded(self):
        gate = threading.Event()

        def work():
            gate.wait(30)
            return {"uploaded": True}

        job = self._presumed_dead_job(work)
        gate.set()
        # collect() short-circuits on terminal jobs (presumed-dead is
        # terminal), so wait for the worker thread to record its late
        # result on its own.
        for _ in range(200):
            if job.late_result:
                break
            time.sleep(0.05)
        assert job.late_result and job.result == {"uploaded": True}
        self.steward.sweep()
        case = self.steward.cases.list_cases()[0]
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(self.steward.cases.get(case.id).verdict,
                         "succeeded")

    def test_full_cycle_report(self):
        _liveness_failed(self.missions)
        self._done_uncollected_job()
        report = run_stewardship_cycle(self.db, registry=self.registry)
        self.assertEqual(report["sweep"]["opened"], 2)
        self.assertEqual(report["runs"]["resolved"], 1)
        self.assertEqual(report["runs"]["closed-unknown"], 1)
        counts = report["counts"]
        self.assertEqual(counts["resolved"], 1)
        self.assertEqual(counts["closed-unknown"], 1)


class MechanismsUntouchedTests(unittest.TestCase):
    def test_steward_never_calls_mutating_domain_paths(self):
        db = _db()
        missions = MissionStore(db)
        registry = JobRegistry()
        _mission(missions, status=MissionStatus.RUNNING,
                 state=_dead_heartbeat(), goal="observe only")
        job = registry.submit(lambda: 1, tool="render")
        registry.collect(job.id, timeout=10)
        job.ended_at = time.time() - 700.0

        def forbidden(name):
            def _raise(*a, **kw):
                raise AssertionError(f"steward called {name}")
            return _raise

        patched = []
        for owner, names in (
                (MissionStore, ("reconcile", "save", "set_status",
                                "create", "create_new")),
                (JobRegistry, ("cancel", "collect", "collect_ready",
                               "submit", "sweep"))):
            for name in names:
                patched.append((owner, name, getattr(owner, name)))
                setattr(owner, name, forbidden(f"{owner.__name__}.{name}"))
        try:
            steward = OutcomeSteward(db, registry=registry)
            report = steward.sweep()
            self.assertEqual(report["opened"], 2)
            runs = steward.run_pending()
            self.assertEqual(runs["ran"], 2)
        finally:
            for owner, name, original in patched:
                setattr(owner, name, original)


class WiringAndSurfaceTests(unittest.TestCase):
    def test_ensure_job_idempotent_and_points_at_the_tool(self):
        class FakeScheduler:
            def __init__(self):
                self.jobs = []
                self.added = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, schedule, kind, payload, **kw):
                job = {"id": "job-1", "name": name, "schedule": schedule,
                       "kind": kind, "payload": payload, **kw}
                self.jobs.append(job)
                self.added.append(job)
                return job

        scheduler = FakeScheduler()
        first = ensure_outcome_steward_job(scheduler)
        second = ensure_outcome_steward_job(scheduler)
        self.assertTrue(first[0]["scheduled"])
        self.assertTrue(second[0]["already_scheduled"])
        self.assertEqual(len(scheduler.added), 1)
        self.assertEqual(scheduler.added[0]["payload"]["tool"],
                         "outcome_steward")

    def test_render_case_prints_no_internal_ids(self):
        db = _db()
        missions = MissionStore(db)
        _liveness_failed(missions)
        steward = OutcomeSteward(db, registry=JobRegistry())
        steward.sweep()
        case = steward.cases.list_cases()[0]
        steward.run_case(case.id)
        case = steward.cases.get(case.id)
        text = render_case(case, steward.cases.receipts(case.id))
        self.assertIn(case.subject, text)
        self.assertIn("outcome unknown", text)
        self.assertNotIn(case.id, text)
        self.assertNotIn(case.contract_id, text)
        self.assertNotIn(case.subject_key, text)


if __name__ == "__main__":
    unittest.main()
