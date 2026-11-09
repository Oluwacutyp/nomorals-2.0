"""Acceptance projects A–F (doc 5 §10) as end-to-end tests.

Each project drives the real machinery — specialist manifests cut work
contracts, workers produce evidence, skeptics challenge, independent
verifiers sign, completion lands in the claim ledger, and (project E)
the real mission lifecycle enforces the contract gate.  Workers are
deterministic stand-ins writing real files, so evidence refs point at
real artifacts the witness can check — no model in the loop.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents import manifests
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.delegation import (
    DelegationStore,
    StaleRevisionError,
    contract_for_step,
)
from nomorals.agents.worldmodel import KIND_OBSERVED, WorldModel
from nomorals.missions.idempotency import IdempotencyStore
from nomorals.storage.db import Database

VERIFIER = "verifier-amy"
SKEPTIC = "skeptic-sue"
SUPERVISOR = "supervisor-sam"


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _ProjectBase(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.store = DelegationStore(self.db)
        self.idem = IdempotencyStore(self.db)
        self.workdir = Path(tempfile.mkdtemp(
            dir=os.environ.get("NM_TEST_TMPDIR") or None))
        self.addCleanup(lambda: shutil.rmtree(self.workdir, True))

    def produce(self, name: str, content: str) -> str:
        path = self.workdir / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def fulfil(self, contract, *, verifier=VERIFIER):
        """Worker produces every required evidence kind as a real
        file; the witness checks the file exists; the verifier —
        never the worker — marks each claim supported."""
        for kind in contract.required_evidence:
            ref = self.produce(
                f"{contract.id}-{kind.replace(' ', '_')}.txt",
                f"evidence: {kind} for {contract.title}")
            self.assertTrue(Path(ref).is_file())
            claim = self.store.submit_evidence(
                contract.id, kind=kind, claim=f"{kind} delivered",
                producer=contract.worker, evidence_ref=ref)
            self.store.verify_claim(claim.id, validator=verifier,
                                    verdict="supported")
        self.store.sign_off(contract.id, verifier=verifier)

    def complete(self, contract):
        decision = self.store.complete(contract.id, by=VERIFIER,
                                       idempotency=self.idem)
        self.assertTrue(decision.ok, decision.reasons)
        return decision


class ProjectAFullStack(_ProjectBase):
    """A — full-stack application: decomposed by manifest, contracted,
    evidenced, verified, completed in the ledger."""

    def test_full_stack_project(self):
        steps = [
            {"name": "api", "goal": "build the service",
             "role": "coding"},
            {"name": "ui", "goal": "build the page", "role": "coding"},
            {"name": "verify", "goal": "test and audit", "role": "critic"},
        ]
        plan = manifests.plan_for_steps(steps)
        self.assertEqual([p["class_id"] for p in plan],
                         ["backend", "backend", "quality_resilience"])
        contracts = []
        for step in steps:
            contract = contract_for_step(
                self.store, mission_id="proj-a", step=step,
                verifier=VERIFIER, skeptic=SKEPTIC,
                recovery_supervisor=SUPERVISOR,
                consequential=step["name"] == "api")
            contracts.append(contract)
        # The consequential API contract takes a skeptic pass first.
        api = contracts[0]
        self.fulfil(api)
        challenge = self.store.challenge(
            api.id, challenger=SKEPTIC,
            claim="error paths are untested")
        self.assertFalse(self.store.complete(api.id, by=VERIFIER).ok)
        self.store.resolve_challenge(
            challenge.id, resolver=api.worker, verdict="supported",
            note="error-path tests added and attached")
        self.complete(api)
        for contract in contracts[1:]:
            self.fulfil(contract)
            self.complete(contract)
        self.assertEqual(self.store.open_contracts("proj-a"), [])
        completions = [c for c in self.store.claims(limit=100)
                       if c.kind == "completion"]
        self.assertEqual(len(completions), 3)


class ProjectBDesignToFrontend(_ProjectBase):
    """B — designer arts → 3D → frontend: chained artifacts, and a
    stale write against a moved revision is FENCED."""

    def test_design_chain_with_fenced_stale_write(self):
        picked = manifests.select_specialists(
            "design the look, model a 3d scene in blender, build the "
            "frontend viewer", limit=4)
        ids = {row["class_id"] for row in picked}
        self.assertIn("graphics_3d", ids)
        self.assertIn("frontend", ids)

        design = self.store.create_contract(
            goal="art direction + scene spec", worker="visual:vision",
            verifier=VERIFIER, required_evidence=("asset manifest",))
        self.store.claim_artifact(design.id, "scene-spec")
        self.store.write_artifact(design.id, "scene-spec",
                                  base_revision=0,
                                  payload_ref=self.produce(
                                      "scene-v1.json", "{}"))
        world = WorldModel(Blackboard())
        world.assert_fact("asset:scene-spec", {"revision": 1},
                          kind=KIND_OBSERVED, source="design contract")
        # A downstream worker holding the OLD revision tries to
        # overwrite the spec — the fence refuses it.
        with self.assertRaises(StaleRevisionError):
            self.store.write_artifact(design.id, "scene-spec",
                                      base_revision=0,
                                      payload_ref="stale overwrite")
        # The current-revision write lands.
        out = self.store.write_artifact(
            design.id, "scene-spec", base_revision=1,
            payload_ref=self.produce("scene-v2.json", "{}"))
        self.assertEqual(out["revision"], 2)
        world.assert_fact("asset:scene-spec", {"revision": 2},
                          kind=KIND_OBSERVED, source="design contract")
        self.fulfil(design)
        self.complete(design)
        self.assertEqual(world.read_fact("asset:scene-spec").version, 2)


class ProjectCRepair(_ProjectBase):
    """C — existing-system repair: a worker's unsupported claim is
    challenged, completion is REFUSED, then real evidence completes."""

    def test_repair_with_refused_completion(self):
        contract = self.store.create_contract(
            goal="repair the reconnect loop", worker="backend:coding",
            verifier=VERIFIER, skeptic=SKEPTIC, consequential=True,
            required_evidence=("test run", "failure reproduction"))
        # Worker claims success with NO executed evidence attached.
        hollow = self.store.submit_evidence(
            contract.id, kind="test run", claim="all tests pass",
            producer="backend:coding", evidence_ref="")
        challenge = self.store.challenge(
            contract.id, challenger=SKEPTIC,
            claim="no executed evidence backs the pass claim",
            target_claim_id=hollow.id)
        self.store.resolve_challenge(
            challenge.id, resolver=VERIFIER, verdict="contradicted",
            note="claim has no artifact; contradicted")
        self.store.sign_off(contract.id, verifier=VERIFIER)
        decision = self.store.complete(contract.id, by=VERIFIER)
        self.assertFalse(decision.ok, "hollow evidence must refuse")
        self.assertTrue(any("test run" in r for r in decision.reasons))
        # The repair is then actually evidenced and completes.
        self.fulfil(contract)
        final = self.store.complete(contract.id, by=VERIFIER,
                                    idempotency=self.idem)
        self.assertTrue(final.ok, final.reasons)
        verdicts = {c.verdict for c in self.store.claims(contract.id)}
        self.assertIn("contradicted", verdicts)


class ProjectDParallelIntegration(_ProjectBase):
    """D — parallel integration: a deliberate API field change must
    SURFACE — evidence pinned to the old shape goes stale, the
    integration only completes after re-verification at the new one."""

    def test_api_change_surfaces_as_stale_evidence(self):
        world = WorldModel(Blackboard())
        world.assert_fact("api:user", {"fields": ["id", "name"]},
                          kind=KIND_OBSERVED, source="openapi dump")
        api = self.store.create_contract(
            goal="user api", worker="backend:coding", verifier=VERIFIER,
            required_evidence=("api transcript",))
        client = self.store.create_contract(
            goal="user client", worker="frontend:coding",
            verifier=VERIFIER, required_evidence=("build output",))
        # Client evidences against API v1 and is verified.
        claim_v1 = self.store.submit_evidence(
            client.id, kind="build output",
            claim="client built against api:user v1",
            producer="frontend:coding", revision="api@1",
            evidence_ref=self.produce("client-v1.txt", "ok"))
        self.store.verify_claim(claim_v1.id, validator=VERIFIER,
                                verdict="supported")
        # The API ships a field change mid-flight.
        world.assert_fact("api:user",
                          {"fields": ["id", "name", "email"]},
                          kind=KIND_OBSERVED, source="openapi dump")
        self.assertEqual(world.read_fact("api:user").version, 2)
        # The integration owner detects the version skew and marks the
        # client's evidence stale — the change surfaces in the ledger.
        self.store.verify_claim(claim_v1.id,
                                validator="integration-owner-ivy",
                                verdict="stale")
        self.store.sign_off(client.id, verifier=VERIFIER)
        blocked = self.store.complete(client.id, by=VERIFIER)
        self.assertFalse(blocked.ok)
        self.assertTrue(any("stale" in r for r in blocked.reasons))
        # Re-evidence at v2, verify, complete both, then integrate.
        claim_v2 = self.store.submit_evidence(
            client.id, kind="build output",
            claim="client rebuilt against api:user v2",
            producer="frontend:coding", revision="api@2",
            evidence_ref=self.produce("client-v2.txt", "ok"))
        self.store.verify_claim(claim_v2.id, validator=VERIFIER,
                                verdict="supported")
        self.complete(client)
        self.fulfil(api)
        self.complete(api)
        integration = contract_for_step(
            self.store, mission_id="proj-d",
            step={"name": "integrate", "goal": "integrate api + client",
                  "role": "reflection"},
            verifier=VERIFIER)
        self.assertEqual(integration.specialist_class,
                         "integration_maintenance")
        self.fulfil(integration)
        self.complete(integration)


class ProjectERecovery(_ProjectBase):
    """E — recovery: worker fails mid-work, the recovery supervisor
    re-plans, the stale generation is fenced, the mission lifecycle
    itself refuses to finish while a contract is open."""

    def test_recovery_replan_and_mission_gate(self):
        contract = self.store.create_contract(
            goal="migrate the ledger", worker="data:memory",
            verifier=VERIFIER, recovery_supervisor=SUPERVISOR,
            mission_id="proj-e", required_evidence=("integrity check",))
        self.store.claim_artifact(contract.id, "ledger.db")
        self.store.report_failure(contract.id, by="data:memory",
                                  detail="worker died at row 40k")
        replanned = self.store.replan(
            contract.id, supervisor=SUPERVISOR,
            new_worker="data:memory-2", note="resume from checkpoint")
        self.assertEqual(replanned.generation, 2)
        # The dead worker's late write is fenced by generation.
        from nomorals.agents.delegation import StaleGenerationError
        with self.assertRaises(StaleGenerationError):
            self.store.write_artifact(
                contract.id, "ledger.db", base_revision=0,
                payload_ref="zombie write", generation=1)
        self.fulfil(replanned)
        self.complete(replanned)

    def _run_one_step_mission(self, goal: str):
        """A real MissionRunner run of a one-step mission on the
        project db, with the step stubbed successful."""
        from types import SimpleNamespace

        from nomorals.agents.orchestrator import PlanStep
        from nomorals.core.tasks import TaskKind
        from nomorals.missions import MissionStore
        from nomorals.missions.runner import (
            MissionRunner,
            StepOutcome,
            _serialize_plan,
        )

        ctx = SimpleNamespace(db=self.db, tools=SimpleNamespace(),
                              memory=None, settings=SimpleNamespace(),
                              extras={})
        store = MissionStore(self.db)
        mission = store.create_new(goal, name="proj-e-gate")
        step = PlanStep(name="build", goal="build it", role="execution",
                        kind=TaskKind.IO)
        mission.state["plan"] = _serialize_plan(
            SimpleNamespace(steps=[step]))
        store.save(mission)
        runner = MissionRunner(
            ctx, store=store, milestones=False,
            orchestrator_factory=lambda: mock.MagicMock())
        runner._execute_step = lambda m, s: StepOutcome(
            step=s.name, ok=True, seconds=0.1, payload={"done": s.name})
        return mission, runner.run(mission, reflect=False)

    def test_mission_cannot_finish_with_open_contracts(self):
        tmp_home = tempfile.mkdtemp(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(lambda: shutil.rmtree(tmp_home, True))
        with mock.patch.dict(os.environ, {"NM_HOME": tmp_home}):
            # No contracts at all → the historical direct path: DONE.
            _m0, result0 = self._run_one_step_mission("ungated mission")
            self.assertEqual(result0.status, "done", result0.error)

            # A contract cut BEFORE the run, still open → refused.
            from nomorals.missions import MissionStore
            store = MissionStore(self.db)
            m_open = store.create_new("pre-gated mission",
                                      name="proj-e-open")
            self.store.create_contract(
                goal="pre work", worker="w", verifier=VERIFIER,
                mission_id=m_open.id)
            from types import SimpleNamespace

            from nomorals.agents.orchestrator import PlanStep
            from nomorals.core.tasks import TaskKind
            from nomorals.missions.runner import (
                MissionRunner,
                StepOutcome,
                _serialize_plan,
            )
            ctx = SimpleNamespace(db=self.db, tools=SimpleNamespace(),
                                  memory=None,
                                  settings=SimpleNamespace(), extras={})
            step = PlanStep(name="build", goal="build it",
                            role="execution", kind=TaskKind.IO)
            m_open.state["plan"] = _serialize_plan(
                SimpleNamespace(steps=[step]))
            store.save(m_open)
            runner = MissionRunner(
                ctx, store=store, milestones=False,
                orchestrator_factory=lambda: mock.MagicMock())
            runner._execute_step = lambda m, s: StepOutcome(
                step=s.name, ok=True, seconds=0.1, payload={})
            refused = runner.run(m_open, reflect=False)
            self.assertEqual(refused.status, "failed")
            self.assertIn("delegation incomplete", refused.error)

            # Complete the contract → a fresh mission under a completed
            # contract finishes DONE.
            contract = self.store.contracts_for_mission(m_open.id)[0]
            self.store.sign_off(contract.id, verifier=VERIFIER)
            done = self.store.complete(contract.id, by=VERIFIER)
            self.assertTrue(done.ok, done.reasons)
            m_done = store.create_new("post-completion mission",
                                      name="proj-e-done")
            carried = self.store.create_contract(
                goal="carried work", worker="w", verifier=VERIFIER,
                mission_id=m_done.id)
            self.store.sign_off(carried.id, verifier=VERIFIER)
            self.assertTrue(
                self.store.complete(carried.id, by=VERIFIER).ok)
            m_done.state["plan"] = _serialize_plan(
                SimpleNamespace(steps=[step]))
            store.save(m_done)
            runner2 = MissionRunner(
                ctx, store=store, milestones=False,
                orchestrator_factory=lambda: mock.MagicMock())
            runner2._execute_step = runner._execute_step
            finished = runner2.run(m_done, reflect=False)
            self.assertEqual(finished.status, "done", finished.error)


class ProjectFPortability(_ProjectBase):
    """F — platform/resource portability: the plan DECLARES the actual
    execution location per class under a constrained profile."""

    def test_termux_plan_declares_locations(self):
        picked = manifests.select_specialists(
            "render a 3d scene and package the release build",
            limit=5, profile="termux", remote_available=False)
        by_id = {row["class_id"]: row for row in picked}
        self.assertEqual(by_id["graphics_3d"]["execution_location"],
                         "deferred")
        self.assertEqual(by_id["platform_delivery"]["execution_location"],
                         "deferred")
        picked_remote = manifests.select_specialists(
            "render a 3d scene and package the release build",
            limit=5, profile="termux", remote_available=True)
        by_id = {row["class_id"]: row for row in picked_remote}
        self.assertEqual(by_id["graphics_3d"]["execution_location"],
                         "remote-worker")
        picked_local = manifests.select_specialists(
            "render a 3d scene", limit=3, profile="")
        self.assertTrue(all(row["execution_location"] == "local"
                            for row in picked_local))
        # Ordinary steps on the constrained profile still plan locally.
        plan = manifests.plan_for_steps(
            [{"name": "api", "goal": "build", "role": "coding"}],
            profile="termux")
        self.assertEqual(plan[0]["execution_location"], "local")


if __name__ == "__main__":
    unittest.main()
