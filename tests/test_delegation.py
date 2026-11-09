"""Accountable delegation: contracts, independence, claim ledger,
completion gating, revision fencing, generation fencing, recovery —
plus the world model and the specialist manifests."""

from __future__ import annotations

import unittest

from nomorals.agents.blackboard import Blackboard
from nomorals.agents.delegation import (
    DelegationStore,
    IndependenceError,
    OwnershipError,
    StaleGenerationError,
    StaleRevisionError,
    contract_for_step,
)
from nomorals.agents.worldmodel import KIND_OBSERVED, WorldModel
from nomorals.missions.idempotency import IdempotencyStore
from nomorals.storage.db import Database


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _contract(store: DelegationStore, **kw):
    base = dict(goal="do the job", worker="worker-wren",
                verifier="verifier-amy", witness="witness-jo",
                required_evidence=("test run",))
    base.update(kw)
    return store.create_contract(**base)


def _fulfil(store: DelegationStore, contract, *, kind="test run"):
    claim = store.submit_evidence(
        contract.id, kind=kind, claim=f"{kind} attached",
        producer=contract.worker, evidence_ref=f"file:///{kind}")
    store.verify_claim(claim.id, validator=contract.verifier,
                       verdict="supported")
    return claim


class ContractRuleTests(unittest.TestCase):
    def setUp(self):
        self.store = DelegationStore(_db())

    def test_verifier_cannot_be_the_worker(self):
        with self.assertRaises(IndependenceError):
            self.store.create_contract(goal="g", worker="sam",
                                       verifier="sam")

    def test_producer_cannot_verify_own_claim(self):
        contract = _contract(self.store)
        claim = self.store.submit_evidence(
            contract.id, kind="test run", claim="tests pass",
            producer="worker-wren")
        with self.assertRaises(IndependenceError):
            self.store.verify_claim(claim.id, validator="worker-wren",
                                    verdict="supported")

    def test_worker_cannot_sign_off(self):
        contract = _contract(self.store)
        with self.assertRaises(IndependenceError):
            self.store.sign_off(contract.id, verifier="worker-wren")

    def test_completion_blocked_until_evidence_and_signoff(self):
        contract = _contract(self.store)
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertFalse(decision.ok)
        self.assertTrue(any("missing required evidence" in r
                            for r in decision.reasons))
        _fulfil(self.store, contract)
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertFalse(decision.ok)
        self.assertTrue(any("sign-off" in r for r in decision.reasons))
        self.store.sign_off(contract.id, verifier="verifier-amy")
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertTrue(decision.ok, decision.reasons)
        self.assertEqual(self.store.get_contract(contract.id).status,
                         "completed")

    def test_completion_is_idempotent_through_batch_b(self):
        db = _db()
        store = DelegationStore(db)
        contract = _contract(store)
        _fulfil(store, contract)
        store.sign_off(contract.id, verifier="verifier-amy")
        idem = IdempotencyStore(db)
        first = store.complete(contract.id, by="verifier-amy",
                               idempotency=idem)
        second = store.complete(contract.id, by="verifier-amy",
                                idempotency=idem)
        self.assertTrue(first.ok and second.ok)
        completions = [c for c in store.claims(contract.id)
                       if c.kind == "completion"]
        self.assertEqual(len(completions), 1,
                         "completion must record exactly once")


class SkepticTests(unittest.TestCase):
    def setUp(self):
        self.store = DelegationStore(_db())

    def test_consequential_contract_needs_a_skeptic_pass(self):
        contract = _contract(self.store, consequential=True,
                             skeptic="skeptic-sue")
        _fulfil(self.store, contract)
        self.store.sign_off(contract.id, verifier="verifier-amy")
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertFalse(decision.ok)
        self.assertTrue(any("skeptic" in r for r in decision.reasons))

    def test_unresolved_challenge_blocks_then_resolution_frees(self):
        contract = _contract(self.store, consequential=True,
                             skeptic="skeptic-sue")
        claim = _fulfil(self.store, contract)
        challenge = self.store.challenge(
            contract.id, challenger="skeptic-sue",
            claim="the test run predates the final patch",
            target_claim_id=claim.id)
        self.store.sign_off(contract.id, verifier="verifier-amy")
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertFalse(decision.ok)
        self.store.resolve_challenge(challenge.id, resolver="worker-wren",
                                     verdict="supported",
                                     note="re-ran after the patch")
        decision = self.store.complete(contract.id, by="verifier-amy")
        self.assertTrue(decision.ok, decision.reasons)

    def test_contradicted_evidence_blocks_until_superseded(self):
        contract = _contract(self.store)
        bad = self.store.submit_evidence(
            contract.id, kind="test run", claim="tests pass",
            producer="worker-wren")
        self.store.verify_claim(bad.id, validator="verifier-amy",
                                verdict="contradicted")
        self.store.sign_off(contract.id, verifier="verifier-amy")
        self.assertFalse(
            self.store.complete(contract.id, by="verifier-amy").ok)
        _fulfil(self.store, contract)  # fresh, supported evidence
        self.assertTrue(
            self.store.complete(contract.id, by="verifier-amy").ok)


class FencingTests(unittest.TestCase):
    def setUp(self):
        self.store = DelegationStore(_db())
        self.contract = _contract(self.store)

    def test_stale_revision_write_is_refused(self):
        self.store.claim_artifact(self.contract.id, "report.md")
        out = self.store.write_artifact(
            self.contract.id, "report.md", base_revision=0,
            payload_ref="v1")
        self.assertEqual(out["revision"], 1)
        with self.assertRaises(StaleRevisionError):
            self.store.write_artifact(
                self.contract.id, "report.md", base_revision=0,
                payload_ref="stale overwrite")
        out = self.store.write_artifact(
            self.contract.id, "report.md", base_revision=1,
            payload_ref="v2")
        self.assertEqual(out["revision"], 2)

    def test_non_owner_write_and_rival_claim_refused(self):
        other = _contract(self.store, worker="worker-oth")
        self.store.claim_artifact(self.contract.id, "report.md")
        with self.assertRaises(OwnershipError):
            self.store.write_artifact(other.id, "report.md",
                                      base_revision=0, payload_ref="x")
        with self.assertRaises(OwnershipError):
            self.store.claim_artifact(other.id, "report.md")

    def test_replan_fences_the_old_generation(self):
        contract = _contract(self.store,
                             recovery_supervisor="supervisor-sam")
        self.store.claim_artifact(contract.id, "draft.md")
        self.store.report_failure(contract.id, by="worker-wren",
                                  detail="worker crashed mid-write")
        self.assertEqual(self.store.get_contract(contract.id).status,
                         "failed")
        from nomorals.agents.delegation import DelegationError
        with self.assertRaises(DelegationError):
            self.store.replan(contract.id, supervisor="worker-wren")
        replanned = self.store.replan(
            contract.id, supervisor="supervisor-sam",
            new_worker="worker-nia")
        self.assertEqual(replanned.generation, 2)
        self.assertEqual(replanned.status, "open")
        with self.assertRaises(StaleGenerationError):
            self.store.submit_evidence(
                contract.id, kind="test run", claim="late evidence",
                producer="worker-wren", generation=1)
        with self.assertRaises(StaleGenerationError):
            self.store.write_artifact(
                contract.id, "draft.md", base_revision=0,
                payload_ref="late write", generation=1)
        out = self.store.write_artifact(
            contract.id, "draft.md", base_revision=0,
            payload_ref="replacement write", generation=2)
        self.assertEqual(out["revision"], 1)


class WaiveTests(unittest.TestCase):
    def test_only_the_arbiter_waives(self):
        store = DelegationStore(_db())
        contract = _contract(store, arbiter="arbiter-ada")
        from nomorals.agents.delegation import DelegationError
        with self.assertRaises(DelegationError):
            store.waive(contract.id, by="worker-wren", reason="skip")
        waived = store.waive(contract.id, by="arbiter-ada",
                             reason="scope withdrawn by the owner")
        self.assertEqual(waived.status, "waived")
        self.assertEqual(store.open_contracts(), [])


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldModel(Blackboard())

    def test_facts_carry_provenance_and_versions(self):
        snap = self.world.assert_fact(
            "repo:head", "abc123", kind=KIND_OBSERVED,
            source="git rev-parse", author="scout")
        self.assertEqual(snap.version, 1)
        self.assertEqual(snap.world_version, 1)
        newer = self.world.assert_fact(
            "repo:head", "def456", kind=KIND_OBSERVED,
            source="git rev-parse", author="scout")
        self.assertEqual(newer.version, 2)
        read = self.world.read_fact("repo:head")
        self.assertEqual(read.value, "def456")
        self.assertEqual(read.version, 2)
        self.assertEqual(read.kind, "observed")
        self.assertEqual(len(self.world.history("repo:head")), 2)

    def test_provenance_is_mandatory(self):
        with self.assertRaises(ValueError):
            self.world.assert_fact("x", 1, kind="observed", source="")
        with self.assertRaises(ValueError):
            self.world.assert_fact("x", 1, kind="rumoured", source="s")

    def test_invalidate_is_a_journaled_tombstone(self):
        self.world.assert_fact("api:shape", {"fields": ["id"]},
                               kind=KIND_OBSERVED, source="openapi dump")
        tomb = self.world.invalidate("api:shape",
                                     reason="api v2 shipped", by="lead")
        self.assertTrue(tomb.invalidated)
        self.assertEqual(tomb.invalidate_reason, "api v2 shipped")
        self.assertEqual(self.world.version(), 2)
        self.assertEqual([f.key for f in self.world.facts()], ["api:shape"])


class ManifestTests(unittest.TestCase):
    def test_seventeen_classes_with_real_bindings(self):
        from nomorals.agents import manifests
        from nomorals.agents.role_specs import default_registry
        from nomorals.agents.roles import ROLES
        from nomorals.agents.subagents import ROSTER

        self.assertEqual(len(manifests.MANIFESTS), 17)
        spec_names = set(default_registry().names())
        for m in manifests.MANIFESTS:
            for key in m.subagents:
                self.assertIn(key, ROSTER, f"{m.class_id}:{key}")
            for key in m.agent_roles:
                self.assertIn(key, ROLES, f"{m.class_id}:{key}")
            for key in m.role_specs:
                self.assertIn(key, spec_names, f"{m.class_id}:{key}")
            self.assertTrue(m.evidence_duties, m.class_id)

    def test_every_step_role_resolves_to_a_class(self):
        from nomorals.agents import manifests
        from nomorals.agents.roles import ROLES

        for role in ROLES:
            self.assertIn(manifests.class_for_role(role),
                          manifests.manifest_index())

    def test_selection_is_deterministic_and_grounded(self):
        from nomorals.agents import manifests

        first = manifests.select_specialists(
            "write integration tests and audit the auth security",
            limit=4)
        second = manifests.select_specialists(
            "write integration tests and audit the auth security",
            limit=4)
        self.assertEqual([r["class_id"] for r in first],
                         [r["class_id"] for r in second])
        ids = {r["class_id"] for r in first}
        self.assertIn("quality_resilience", ids)
        self.assertIn("security_privacy", ids)

    def test_contract_for_step_uses_manifest_evidence_duties(self):
        store = DelegationStore(_db())
        contract = contract_for_step(
            store, mission_id="m1",
            step={"name": "api", "goal": "build the service",
                  "role": "coding"},
            verifier="verifier-amy")
        self.assertEqual(contract.specialist_class, "backend")
        self.assertEqual(contract.required_evidence,
                         ("test run", "api transcript"))
        self.assertEqual(contract.mission_id, "m1")


if __name__ == "__main__":
    unittest.main()
