"""Doc 18 P6 / G9 — planner-created contracts at the execution seam.

Pins: (a) the consequential rule composes the shipped signals, each
with a named reason (declared-consequential · declared writes ·
debate flag · heavy class via the shipped manifests resolver), and
a step meeting none of them is not consequential; (b) at the
orchestrator's dispatch seam a consequential step's contract exists
OPEN in the ledger BEFORE its handler runs (ordering asserted from
inside the handler), keyed by the run id; (c) the step's completion
flows through the contract — evidence per required kind, standing
skeptic challenge resolved, verifier sign-off, steward completion —
and the outcome is stamped on the result; (d) a declared-writes
step whose recorded result is empty is a concrete defect: the gate
refuses, the stamp names it, the work product stands; (e) PARITY:
sub-consequential work is byte-identical — no contract, no stamp;
(f) fail-open: contract creation raising still executes the step,
records the failure in the core ledger, and stamps the gap on the
result; a run without a database is today's behavior exactly;
(g) step failures (raise / reported error) land on the contract as
report_failure — one contract per dispatch attempt, since the
supervisor's retry re-dispatches and each attempt's record stands
on its own; (h) the seam functions are reusable off the orchestrator
(duck-typed step mappings in, outcomes out).

Isolation: OrchestratorSeamTests runs the core failure ledger
against a per-test tmp home (NM_HOME override + ring clears) —
the fail-open pin counts ledger entries, and the ring caps at
200, so a saturated live home could never let the count grow.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import plan_contracts
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.runtime import HybridExecutor
from nomorals.core import failures as core_failures
from nomorals.storage.db import Database


def _file_db(test: unittest.TestCase) -> Database:
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(tmp.cleanup)
    db = Database(os.path.join(tmp.name, "p6.db"))
    db.migrate()
    return db


def _orch(db):
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(), router=None, executor=None,
        blackboard=None, emit=lambda *a, **k: None)
    return MasterOrchestrator(ctx, executor=HybridExecutor())


class RuleTests(unittest.TestCase):
    def test_no_signal_not_consequential(self):
        ok, reasons = plan_contracts.step_consequentiality(
            {"name": "look", "role": "research", "payload": {}})
        self.assertFalse(ok)
        self.assertEqual(reasons, [])

    def test_each_signal_named(self):
        cases = [
            ({"role": "research", "payload": {"consequential": True}},
             "declares itself consequential"),
            ({"role": "research", "writes": ["a.txt"]}, "declares writes"),
            ({"role": "research", "payload": {"writes": ["a.txt"]}},
             "declares writes"),
            ({"role": "research", "payload": {"debate": True}},
             "debate-flagged"),
        ]
        for step, fragment in cases:
            ok, reasons = plan_contracts.step_consequentiality(step)
            self.assertTrue(ok, step)
            self.assertTrue(any(fragment in r for r in reasons),
                            (step, reasons))

    def test_heavy_class_signal_via_shipped_resolver(self):
        # With the shipped _ROLE_TO_CLASS no planner role resolves to
        # a heavy class today; the branch is pinned against the
        # resolver the seam actually composes.
        with mock.patch("nomorals.agents.manifests.class_for_role",
                        return_value="graphics_3d"):
            ok, reasons = plan_contracts.step_consequentiality(
                {"role": "render", "payload": {}})
        self.assertTrue(ok)
        self.assertTrue(any("heavy class" in r for r in reasons))

    def test_empty_writes_is_not_a_signal(self):
        ok, _ = plan_contracts.step_consequentiality(
            {"role": "research", "writes": [], "payload": {}})
        self.assertFalse(ok)


class SeamFunctionTests(unittest.TestCase):
    def setUp(self):
        self.db = _file_db(self)
        self.store = DelegationStore(self.db)
        self.step = {"name": "write", "goal": "write the file",
                     "role": "execution", "writes": ["out.txt"],
                     "payload": {"writes": ["out.txt"]}}

    def test_open_and_settle_completes(self):
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        self.assertTrue(contract.consequential)
        self.assertEqual(contract.scope, ("out.txt",))
        self.assertEqual(contract.verifier, plan_contracts.VERIFIER)
        self.assertNotEqual(contract.worker, contract.verifier)
        outcome = plan_contracts.settle_step_contract(
            self.store, contract, step=self.step,
            result={"did": "write"}, mission_id="run-x")
        self.assertTrue(outcome["completed"], outcome)
        self.assertEqual(outcome["evidence_ref"],
                         "plan-step:run-x/write")
        fresh = self.store.get_contract(contract.id)
        self.assertEqual(fresh.status, "completed")
        kinds = [c.kind for c in self.store.claims(contract.id)]
        self.assertIn("challenge", kinds)
        self.assertIn("signoff", kinds)

    def test_defect_refuses_completion(self):
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        outcome = plan_contracts.settle_step_contract(
            self.store, contract, step=self.step, result={},
            mission_id="run-x")
        self.assertFalse(outcome["completed"])
        self.assertTrue(outcome["reasons"])
        fresh = self.store.get_contract(contract.id)
        self.assertNotEqual(fresh.status, "completed")

    def test_fail_records_on_ledger(self):
        contract = plan_contracts.open_step_contract(
            self.store, mission_id="run-x", step=self.step)
        plan_contracts.fail_step_contract(
            self.store, contract, detail="step blew up")
        fresh = self.store.get_contract(contract.id)
        self.assertEqual(fresh.status, "failed")
        kinds = [c.kind for c in self.store.claims(contract.id)]
        self.assertIn("failure", kinds)


class OrchestratorSeamTests(unittest.TestCase):
    def setUp(self):
        self.db = _file_db(self)
        self.orch = _orch(self.db)
        self.store = DelegationStore(self.db)
        # Core-ledger isolation (Batch A extension adjudication,
        # row 6): the fail-open pin counts ledger entries before /
        # after, and the ledger ring caps at 200 — against a
        # saturated live home the count can never grow. Redirect
        # the store with the app's own NM_HOME override (per-test
        # tmp home; the store resolves it fresh on every operation)
        # and clear the ring: the property-failure suites' idiom,
        # in setUp form so it also holds under unittest.main(),
        # where pytest fixtures never run.
        self._nm_home_tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(self._nm_home_tmp.cleanup)
        self._nm_home_before = os.environ.get("NM_HOME")
        self.addCleanup(self._restore_nm_home)
        os.environ["NM_HOME"] = os.path.join(self._nm_home_tmp.name,
                                             "home")
        core_failures.clear_failures()
        self.addCleanup(self._clear_tmp_ledger)

    def _clear_tmp_ledger(self):
        # Re-pin before the teardown clear: if a test changed or
        # deleted NM_HOME mid-test, clear_failures() resolves the
        # store path at call time — it must still land on the tmp
        # home, never unlink the live home's ledger.
        os.environ["NM_HOME"] = os.path.join(self._nm_home_tmp.name,
                                             "home")
        core_failures.clear_failures()

    def _restore_nm_home(self):
        if self._nm_home_before is None:
            os.environ.pop("NM_HOME", None)
        else:
            os.environ["NM_HOME"] = self._nm_home_before

    def _run(self, plan, handler):
        return self.orch.run("goal", plan=plan, default_handler=handler,
                             reflect=False)

    def test_contract_exists_before_execution(self):
        seen = {}

        def handler(task):
            seen[task.name] = [
                (c.id, c.status) for c in
                self.store.contracts_for_mission(self.orch._run_id)]
            return {"did": task.name}

        plan = Plan(goal="g", steps=[
            PlanStep(name="look", goal="read", role="research"),
            PlanStep(name="write", goal="write", role="execution",
                     depends_on=["look"], payload={"writes": ["o.txt"]}),
        ])
        result = self._run(plan, handler)
        self.assertTrue(result.ok)
        self.assertEqual(seen["look"], [])
        self.assertEqual(len(seen["write"]), 1)
        self.assertEqual(seen["write"][0][1], "open")
        contracts = self.store.contracts_for_mission(self.orch._run_id)
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].status, "completed")
        stamped = result.report.results["write"]["delegation_contract"]
        self.assertEqual(stamped["contract_id"], contracts[0].id)
        self.assertTrue(stamped["completed"])

    def test_sub_consequential_parity(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="look", goal="read", role="research"),
            PlanStep(name="judge", goal="judge", role="critic",
                     depends_on=["look"]),
        ])
        result = self._run(plan, lambda task: {"did": task.name})
        self.assertTrue(result.ok)
        self.assertEqual(
            self.store.contracts_for_mission(self.orch._run_id), [])
        self.assertEqual(result.report.results["look"], {"did": "look"})
        self.assertEqual(result.report.results["judge"], {"did": "judge"})

    def test_declared_consequential_without_writes(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="audit", goal="audit", role="research",
                     payload={"consequential": True}),
        ])
        result = self._run(plan, lambda task: "audited")
        self.assertTrue(result.ok)
        contracts = self.store.contracts_for_mission(self.orch._run_id)
        self.assertEqual(len(contracts), 1)
        out = result.report.results["audit"]
        self.assertEqual(out["output"], "audited")
        self.assertTrue(out["delegation_contract"]["completed"])

    def test_handler_raise_reports_failure_on_contract(self):
        def handler(task):
            raise RuntimeError("disk gone")

        plan = Plan(goal="g", steps=[
            PlanStep(name="write", goal="write", role="execution",
                     payload={"writes": ["o.txt"]}),
        ])
        result = self._run(plan, handler)
        self.assertFalse(result.ok)
        contracts = self.store.contracts_for_mission(self.orch._run_id)
        # The supervisor retries the step; every attempt dispatches
        # fresh, so every attempt carries its OWN contract — and
        # every one of them records the failure (none completes,
        # none is left silently open).
        self.assertGreaterEqual(len(contracts), 1)
        for contract in contracts:
            self.assertEqual(contract.status, "failed")
            kinds = [c.kind for c in self.store.claims(contract.id)]
            self.assertIn("failure", kinds)

    def test_reported_error_reports_failure_on_contract(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="write", goal="write", role="execution",
                     payload={"writes": ["o.txt"]}),
        ])
        result = self._run(plan, lambda task: {"error": "no space"})
        self.assertFalse(result.ok)
        contracts = self.store.contracts_for_mission(self.orch._run_id)
        self.assertEqual(contracts[0].status, "failed")

    def test_creation_failure_fails_open(self):
        before = len(core_failures.recent_failures(limit=200))
        plan = Plan(goal="g", steps=[
            PlanStep(name="write", goal="write", role="execution",
                     payload={"writes": ["o.txt"]}),
        ])
        with mock.patch.object(plan_contracts, "open_step_contract",
                               side_effect=RuntimeError("substrate down")):
            result = self._run(plan, lambda task: {"did": "write"})
        self.assertTrue(result.ok)
        out = result.report.results["write"]
        self.assertEqual(out["did"], "write")
        self.assertIn("substrate down",
                      out["delegation_contract"]["error"])
        after = core_failures.recent_failures(limit=200)
        self.assertGreater(len(after), before)
        self.assertTrue(any(
            f.site == "agents.orchestrator.plan_contract"
            for f in after))

    def test_no_database_is_todays_behavior(self):
        orch = _orch(None)
        plan = Plan(goal="g", steps=[
            PlanStep(name="write", goal="write", role="execution",
                     payload={"writes": ["o.txt"]}),
        ])
        result = orch.run("goal", plan=plan,
                          default_handler=lambda task: {"did": "write"},
                          reflect=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.report.results["write"], {"did": "write"})

    def test_debate_step_also_contracts(self):
        from nomorals.agents.debate import Critique

        self.orch.debate_critic = lambda artifact, rubric: Critique(
            verdict="approve", score=9.0, notes="fine")
        plan = Plan(goal="g", steps=[
            PlanStep(name="draft", goal="draft", role="coding",
                     payload={"debate": True}),
        ])
        result = self._run(plan, lambda task: {"did": "draft"})
        self.assertTrue(result.ok)
        contracts = self.store.contracts_for_mission(self.orch._run_id)
        self.assertEqual(len(contracts), 1)
        out = result.report.results["draft"]
        self.assertEqual(out["debate_verdict"], "approved")
        self.assertTrue(out["delegation_contract"]["completed"])


if __name__ == "__main__":
    unittest.main()
