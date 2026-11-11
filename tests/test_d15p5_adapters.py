"""Doc 15 P5 — domain adapters + the mission effect-outcome projection.

Pins: (a) the projection — a liveness-FAILED mission whose own
records establish its effect reads that effect in the projection
while its mission status stays FAILED (both facts coexist); no
effect evidence reads effect-unknown, and the projection NEVER
upgrades an unevidenced effect; (b) the steward's missions discovery
consumes the projection — an evidenced effect opens no
unknown-outcome case, an unknown one still does, and an existing
case resolves citing the projection evidence; (c) per shipped
adapter (contracts, scheduler, outbox): a planted overdue/unknown
record in the domain's REAL store opens exactly one case across two
sweeps and gathers evidence receipts; a healthy domain opens none;
(d) one domain's discovery raising leaves the others' cases intact
and the failure recorded in the sweep report; (e) the no-mutation
discipline for every new domain — monkeypatched mutators never fire
for the observed records, and the scheduler/outbox tables are
byte-identical after a full cycle. (Browser + render were
unavailable at P5; doc 18 P1 built their durable records and their
adapters now read working. Transfers + artifacts were likewise
unavailable until the delivery record registry
(storage/delivery_records) journaled transfer entries and
delivery/publication receipts at the hand-off seams — their
adapters now read working too. The health pins below state the
current truth.)
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.delegation import DelegationStore
from nomorals.agents.outcome_steward import (
    ContractsAdapter,
    OutcomeSteward,
    adapter_health,
    register as register_steward_tool,
    run_stewardship_cycle,
)
from nomorals.agents.scheduler import Scheduler
from nomorals.core.jobs import JobRegistry
from nomorals.missions.idempotency import (
    IdempotencyStore,
    dedupe,
    step_idempotency_key,
)
from nomorals.missions.mission import (
    Mission,
    MissionStatus,
    MissionStore,
)
from nomorals.social.chat.outbox import Outbox
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.storage.queue import WorkQueue

DEAD_PID = 999999
NOW = time.time()


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _mission(store: MissionStore, *, status: str, state=None,
             goal: str = "do the thing") -> Mission:
    now = time.time()
    mission = Mission(goal=goal, status=status,
                      state=dict(state or {}),
                      created_at=now, updated_at=now)
    return store.create(mission)


def _liveness_state(extra: dict | None = None) -> dict:
    state = {"heartbeat": {"at": time.time() - 100000.0,
                           "pid": DEAD_PID},
             "last_error": ("runner died — no heartbeat for 27h "
                            "(pid 999999 is gone)")}
    state["status_note"] = f"reconciled: {state['last_error']}"
    state.update(extra or {})
    return state


def _complete_contract(store: DelegationStore, contract) -> None:
    """Drive a planted contract to completion through the real gate:
    verifier sign-off, then the completion decision."""
    store.sign_off(contract.id, verifier=contract.verifier)
    decision = store.complete(contract.id, by=contract.verifier)
    assert decision.ok, decision.reasons


def _plant_contract(store: DelegationStore, **kw):
    base = dict(goal="ship the report", title="ship the report",
                worker="cell:writer", verifier="cell:checker")
    base.update(kw)
    return store.create_contract(**base)


# ── (a) the projection ────────────────────────────────────────────────


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.missions = MissionStore(self.db)

    def test_evidenced_effect_coexists_with_failed_status(self):
        plan = {"steps": [
            {"name": "gather", "goal": "gather data", "role": "scout"},
            {"name": "write", "goal": "write it", "role": "writer"}]}
        mission = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({
                "plan": plan,
                "completed_steps": ["gather", "write"],
                "outputs": {"gather": {"ok": True},
                            "write": {"ok": True}}}),
            goal="render the report")
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.effect, "succeeded")
        self.assertEqual(projection.failure_basis, "liveness")
        self.assertEqual(projection.steps_total, 2)
        self.assertEqual(projection.steps_completed, 2)
        self.assertTrue(any(e["source"] == "missions.plan"
                            for e in projection.evidence))
        # Both facts coexist: the runner status is untouched.
        self.assertEqual(self.missions.get(mission.id).status,
                         MissionStatus.FAILED)
        self.assertEqual(projection.status, MissionStatus.FAILED)

    def test_no_effect_evidence_reads_unknown(self):
        mission = _mission(self.missions, status=MissionStatus.FAILED,
                           state=_liveness_state(),
                           goal="render the report")
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.effect, "unknown")
        self.assertEqual(projection.failure_basis, "liveness")
        self.assertEqual(projection.steps_total, 0)

    def test_recorded_failure_basis_is_not_liveness(self):
        mission = _mission(
            self.missions, status=MissionStatus.FAILED,
            state={"last_error": "step 2 raised ValueError"})
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.failure_basis, "recorded")

    def test_failed_step_claim_only_reads_failed(self):
        plan = {"steps": [
            {"name": "push", "goal": "push it", "role": "ops"}]}
        mission = _mission(self.missions, status=MissionStatus.FAILED,
                           state=_liveness_state({"plan": plan}))
        idem = IdempotencyStore(self.db)
        step = SimpleNamespace(name="push", goal="push it", role="ops")
        key = step_idempotency_key(mission.id, step)

        def _boom():
            raise RuntimeError("push failed")

        with self.assertRaises(RuntimeError):
            dedupe(idem, key, _boom, owner="test")
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.claims_failed, 1)
        self.assertEqual(projection.effect, "failed")

    def test_completed_claims_establish_steps(self):
        plan = {"steps": [
            {"name": "a", "goal": "ga", "role": "scout"},
            {"name": "b", "goal": "gb", "role": "scout"}]}
        mission = _mission(self.missions, status=MissionStatus.FAILED,
                           state=_liveness_state({"plan": plan}))
        idem = IdempotencyStore(self.db)
        for name, goal in (("a", "ga"), ("b", "gb")):
            step = SimpleNamespace(name=name, goal=goal, role="scout")
            dedupe(idem, step_idempotency_key(mission.id, step),
                   lambda: {"ok": True}, owner="test")
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.claims_completed, 2)
        self.assertEqual(projection.steps_completed, 2)
        self.assertEqual(projection.effect, "succeeded")

    def test_partial_progress_reads_partial(self):
        plan = {"steps": [
            {"name": "a", "goal": "ga", "role": "scout"},
            {"name": "b", "goal": "gb", "role": "scout"}]}
        mission = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({"plan": plan,
                                   "completed_steps": ["a"]}))
        idem = IdempotencyStore(self.db)
        step = SimpleNamespace(name="b", goal="gb", role="scout")

        def _boom():
            raise RuntimeError("b failed")

        with self.assertRaises(RuntimeError):
            dedupe(idem, step_idempotency_key(mission.id, step),
                   _boom, owner="test")
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.steps_completed, 1)
        self.assertEqual(projection.claims_failed, 1)
        self.assertEqual(projection.effect, "partial")

    def test_verification_record_decides(self):
        passed = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({"verification": {"passed": True}}))
        self.assertEqual(
            self.missions.effect_outcome(passed.id).effect,
            "succeeded")
        failed = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({
                "verification": {
                    "passed": False,
                    "required_criteria_failed": ["report_exists"]}}))
        projection = self.missions.effect_outcome(failed.id)
        self.assertEqual(projection.effect, "failed")
        self.assertEqual(projection.verification, "failed")
        self.assertTrue(any("report_exists" in e["summary"]
                            for e in projection.evidence))

    def test_fully_delegated_mission_reads_succeeded(self):
        mission = _mission(self.missions, status=MissionStatus.FAILED,
                           state=_liveness_state())
        delegation = DelegationStore(self.db)
        contract = _plant_contract(delegation, mission_id=mission.id)
        _complete_contract(delegation, contract)
        projection = self.missions.effect_outcome(mission.id)
        self.assertEqual(projection.contracts_total, 1)
        self.assertEqual(projection.contracts_completed, 1)
        self.assertEqual(projection.effect, "succeeded")

    def test_projection_computes_and_writes_nothing(self):
        mission = _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({
                "plan": {"steps": [
                    {"name": "a", "goal": "ga", "role": "scout"}]},
                "completed_steps": ["a"]}))
        before = self.db.query_one(
            "SELECT * FROM missions WHERE id = ?", (mission.id,))
        self.missions.effect_outcome(mission.id)
        self.missions.effect_outcome(mission.id)
        after = self.db.query_one(
            "SELECT * FROM missions WHERE id = ?", (mission.id,))
        self.assertEqual(dict(before), dict(after))


class ProjectionArtifactTests(unittest.TestCase):
    def test_artifacts_only_reads_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "devon.db"))
            db.migrate()
            missions = MissionStore(db)
            mission = _mission(missions, status=MissionStatus.FAILED,
                               state=_liveness_state())
            store = ArtifactStore(
                db, BlobStore(db, Path(tmp) / "blobs"))
            store.put(b"the report", type="report",
                      creator="test", mission_id=mission.id)
            projection = missions.effect_outcome(mission.id)
            self.assertEqual(projection.artifacts, 1)
            self.assertEqual(projection.effect, "partial")
            db.close()


# ── (b) the steward consumes the projection ───────────────────────────


class StewardProjectionTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.missions = MissionStore(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())

    def _evidenced_mission(self) -> Mission:
        return _mission(
            self.missions, status=MissionStatus.FAILED,
            state=_liveness_state({
                "plan": {"steps": [
                    {"name": "only", "goal": "g", "role": "scout"}]},
                "completed_steps": ["only"],
                "outputs": {"only": {"ok": True}}}),
            goal="landed work")

    def test_evidenced_effect_opens_no_unknown_case(self):
        self._evidenced_mission()
        report = self.steward.sweep()
        self.assertEqual(report["opened"], 0)
        self.assertEqual(self.steward.cases.list_cases(), [])

    def test_unknown_effect_still_opens_case(self):
        _mission(self.missions, status=MissionStatus.FAILED,
                 state=_liveness_state(), goal="mystery work")
        report = self.steward.sweep()
        self.assertEqual(report["opened"], 1)
        case = self.steward.cases.list_cases()[0]
        self.assertEqual(case.reason, "unknown-outcome")

    def test_existing_case_resolves_citing_projection(self):
        mission = self._evidenced_mission()
        case, created = self.steward.cases.open_case(
            domain="missions", kind="mission", subject=mission.goal,
            subject_key=f"mission:{mission.id}",
            reason="unknown-outcome",
            detail={"mission_id": mission.id})
        assert created
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "succeeded")
        cited = [r for r in self.steward.cases.receipts(case.id)
                 if r.id == closed.evidence_receipt_id]
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0].source, "missions.effect_outcome")


# ── (c) contracts adapter ─────────────────────────────────────────────


class ContractsAdapterTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.delegation = DelegationStore(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())

    def _contracts_cases(self):
        return [c for c in self.steward.cases.list_cases()
                if c.domain == "contracts"]

    def test_overdue_contract_one_case_across_two_sweeps(self):
        _plant_contract(self.delegation, deadline=NOW - 100.0)
        first = self.steward.sweep()
        second = self.steward.sweep()
        cases = self._contracts_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "overdue")
        self.assertEqual(first["opened"] + second["opened"], 1)

    def test_budget_spent_contract_resolves_when_completed(self):
        contract = _plant_contract(self.delegation, step_budget=1)
        self.db.execute(
            "UPDATE delegation_contracts SET steps_used = 1"
            " WHERE id = ?", (contract.id,))
        self.steward.sweep()
        case = self._contracts_cases()[0]
        self.assertEqual(case.reason, "overdue")
        _complete_contract(self.delegation, contract)
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "resolved")
        closed = self.steward.cases.get(case.id)
        self.assertEqual(closed.verdict, "succeeded")
        sources = [r.source for r in self.steward.cases.receipts(case.id)
                   if r.kind == "evidence"]
        self.assertIn("delegation.contracts", sources)

    def test_open_overdue_contract_closes_unknown_not_resolved(self):
        _plant_contract(self.delegation, deadline=NOW - 100.0)
        self.steward.sweep()
        case = self._contracts_cases()[0]
        result = self.steward.run_case(case.id)
        self.assertEqual(result["outcome"], "closed-unknown")
        self.assertEqual(self.steward.cases.get(case.id).verdict,
                         "outcome_unknown")

    def test_healthy_contract_opens_none(self):
        _plant_contract(self.delegation, deadline=NOW + 86400.0)
        self.steward.sweep()
        self.assertEqual(self._contracts_cases(), [])


# ── (c) scheduler adapter ─────────────────────────────────────────────


def _scheduler(db: Database) -> Scheduler:
    ctx = SimpleNamespace(
        db=db,
        tools=SimpleNamespace(
            call=lambda tool, capabilities=None, **kw: SimpleNamespace(
                ok=True, value="ok", error=None)),
        settings=SimpleNamespace(), extras={})
    return Scheduler(ctx)


class SchedulerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.scheduler = _scheduler(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())

    def _scheduler_cases(self):
        return [c for c in self.steward.cases.list_cases()
                if c.domain == "scheduler"]

    def test_overdue_job_one_case_across_two_sweeps(self):
        job = self.scheduler.add("nightly digest", "every 1h",
                                 "message", {"text": "digest"})
        self.db.execute(
            "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
            (NOW - 7200.0, job["id"]))
        self.steward.sweep()
        self.steward.sweep()
        cases = self._scheduler_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "overdue")
        result = self.steward.run_case(cases[0].id)
        self.assertEqual(result["outcome"], "closed-unknown")
        notes = [r.note for r in
                 self.steward.cases.receipts(cases[0].id)
                 if r.kind == "evidence"]
        self.assertTrue(any("no run was ever recorded" in n
                            for n in notes))

    def test_abandoned_one_shot_resolves_failed(self):
        job = self.scheduler.add("one-off alert", "in 5 minutes",
                                 "message", {"text": "alert"})
        self.db.execute(
            "UPDATE schedule_jobs SET next_run = NULL WHERE id = ?",
            (job["id"],))
        self.scheduler._record_run(
            job["id"], NOW - 7200.0, 1.0, False, "gateway down",
            0, "tick")
        self.steward.sweep()
        cases = self._scheduler_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "abandoned")
        result = self.steward.run_case(cases[0].id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(self.steward.cases.get(cases[0].id).verdict,
                         "failed")

    def test_healthy_job_opens_none(self):
        self.scheduler.add("future job", "every 1h", "message",
                           {"text": "later"})
        self.steward.sweep()
        self.assertEqual(self._scheduler_cases(), [])


# ── (c) outbox adapter ────────────────────────────────────────────────


class OutboxAdapterTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.outbox = Outbox(self.db)
        self.steward = OutcomeSteward(self.db, registry=JobRegistry())

    def _outbox_cases(self):
        return [c for c in self.steward.cases.list_cases()
                if c.domain == "outbox"]

    def _backdate(self, job_id: str, age_s: float = 7200.0) -> None:
        self.db.execute(
            "UPDATE work_queue SET created_at = ? WHERE id = ?",
            (time.time() - age_s, job_id))

    def test_stuck_send_case_then_delivered_resolution(self):
        job_id = self.outbox.submit(
            platform="telegram", chat="2348000000000",
            text="your report is ready")
        self._backdate(job_id)
        self.steward.sweep()
        self.steward.sweep()
        cases = self._outbox_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "overdue")
        # The subject names platform + the message's own words —
        # never the chat key (here a phone-shaped string).
        self.assertIn("telegram message: your report is ready",
                      cases[0].subject)
        self.assertNotIn("2348000000000", cases[0].subject)
        WorkQueue(self.db).complete(job_id, {"delivered": True})
        result = self.steward.run_case(cases[0].id)
        self.assertEqual(result["outcome"], "resolved")
        self.assertEqual(self.steward.cases.get(cases[0].id).verdict,
                         "delivered")

    def test_expired_lease_is_unknown_outcome(self):
        job_id = self.outbox.submit(
            platform="whatsapp", chat="chat-9", text="ping")
        queue = WorkQueue(self.db)
        job = queue.lease_one("chat_outbox:whatsapp:chat-9",
                              worker="test")
        assert job is not None and job.id == job_id
        self.db.execute(
            "UPDATE work_queue SET lease_until = ? WHERE id = ?",
            (time.time() - 60.0, job_id))
        self._backdate(job_id)
        self.steward.sweep()
        cases = self._outbox_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].reason, "unknown-outcome")

    def test_fresh_send_opens_none(self):
        self.outbox.submit(platform="telegram", chat="chat-1",
                           text="just now")
        self.steward.sweep()
        self.assertEqual(self._outbox_cases(), [])


# ── (d) containment ───────────────────────────────────────────────────


class ContainmentTests(unittest.TestCase):
    def test_one_domain_raising_leaves_others_intact(self):
        db = _db()
        scheduler = _scheduler(db)
        job = scheduler.add("nightly digest", "every 1h", "message",
                            {"text": "digest"})
        db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (NOW - 7200.0, job["id"]))
        steward = OutcomeSteward(db, registry=JobRegistry())
        original = ContractsAdapter.discover

        def _boom(self, now):
            raise RuntimeError("contracts store unreadable")

        ContractsAdapter.discover = _boom
        try:
            report = steward.sweep()
        finally:
            ContractsAdapter.discover = original
        self.assertEqual(report["opened"], 1)
        self.assertEqual(
            [c.domain for c in steward.cases.list_cases()],
            ["scheduler"])
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["domain"], "contracts")
        self.assertIn("unreadable", report["errors"][0]["error"])


# ── (e) no-mutation discipline ────────────────────────────────────────


class NoMutationTests(unittest.TestCase):
    def test_contract_mutators_never_touch_the_observed_contract(self):
        db = _db()
        delegation = DelegationStore(db)
        observed = _plant_contract(delegation, deadline=NOW - 100.0)
        before = delegation.get_contract(observed.id)
        guarded = ("complete", "waive", "report_failure", "replan",
                   "sign_off", "submit_evidence", "challenge")
        originals = {name: getattr(DelegationStore, name)
                     for name in guarded}

        def _wrap(name, fn):
            def _guarded(self, contract_id, *args, **kwargs):
                if contract_id == observed.id:
                    raise AssertionError(
                        f"steward called {name} on the observed "
                        "contract")
                return fn(self, contract_id, *args, **kwargs)
            return _guarded

        try:
            for name, fn in originals.items():
                setattr(DelegationStore, name, _wrap(name, fn))
            steward = OutcomeSteward(db, registry=JobRegistry())
            steward.sweep()
            steward.run_pending()
        finally:
            for name, fn in originals.items():
                setattr(DelegationStore, name, fn)
        after = delegation.get_contract(observed.id)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.steps_used, before.steps_used)
        self.assertEqual(after.generation, before.generation)
        self.assertEqual(after.updated_at, before.updated_at)

    def test_scheduler_and_outbox_tables_byte_identical(self):
        db = _db()
        scheduler = _scheduler(db)
        job = scheduler.add("nightly digest", "every 1h", "message",
                            {"text": "digest"})
        db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (NOW - 7200.0, job["id"]))
        outbox = Outbox(db)
        send_id = outbox.submit(platform="telegram", chat="chat-1",
                                text="stuck send")
        db.execute("UPDATE work_queue SET created_at = ? WHERE id = ?",
                   (NOW - 7200.0, send_id))

        def _dump():
            return {
                table: [dict(r) for r in db.query(
                    f"SELECT * FROM {table} ORDER BY id")]
                for table in ("schedule_jobs", "schedule_runs",
                              "work_queue")}

        before = _dump()
        mutators = []
        for owner, name in (
                (Scheduler, "add"), (Scheduler, "remove"),
                (Scheduler, "set_enabled"), (WorkQueue, "enqueue"),
                (WorkQueue, "lease"), (WorkQueue, "complete"),
                (WorkQueue, "fail"), (Outbox, "submit")):
            mutators.append((owner, name, getattr(owner, name)))

            def _raise(self, *a, _name=name, **kw):
                raise AssertionError(f"domain mutator called: {_name}")

            setattr(owner, name, _raise)
        try:
            steward = OutcomeSteward(db, registry=JobRegistry())
            steward.sweep()
            steward.run_pending()
        finally:
            for owner, name, fn in mutators:
                setattr(owner, name, fn)
        self.assertEqual(_dump(), before)


# ── adapter health surface ────────────────────────────────────────────


class AdapterHealthTests(unittest.TestCase):
    def test_health_names_every_domain(self):
        health = {row["domain"]: row for row in adapter_health()}
        for domain in ("missions", "jobs", "contracts", "scheduler",
                       "outbox", "browser", "render", "transfers",
                       "artifacts"):
            self.assertEqual(health[domain]["status"], "working",
                             domain)
            self.assertTrue(health[domain]["note"], domain)
        self.assertTrue(all(row["status"] == "working"
                            for row in health.values()))
        self.assertIn("browser_run_records",
                      health["browser"]["note"])
        self.assertIn("media_job_records", health["render"]["note"])
        self.assertIn("transfer_entries",
                      health["transfers"]["note"])
        self.assertIn("delivery_receipts",
                      health["artifacts"]["note"])
        self.assertIn("publication_receipts",
                      health["artifacts"]["note"])

    def test_cycle_and_tool_status_carry_adapter_health(self):
        db = _db()
        report = run_stewardship_cycle(db, registry=JobRegistry())
        domains = {row["domain"] for row in report["adapters"]}
        self.assertIn("contracts", domains)
        self.assertIn("browser", domains)

        captured = {}

        class _Registry:
            context = SimpleNamespace(db=db)

            def register(self, name, **kw):
                def _deco(fn):
                    captured[name] = fn
                    return fn
                return _deco

        register_steward_tool(_Registry())
        status = captured["outcome_steward"](action="status")
        self.assertTrue(status["ok"])
        tool_domains = {row["domain"] for row in status["adapters"]}
        self.assertEqual(tool_domains, domains)


if __name__ == "__main__":
    unittest.main()
