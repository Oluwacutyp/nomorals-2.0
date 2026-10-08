"""Tests for H.O.T-Jarvis test-proof skill evolution (skill_proving).

Offline: no network, no real LLM (mock llm_fn), proof tests run in a
sandboxed subprocess via the repo's own pytest.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import skill_proving as P
from nomorals.agents.skill_distillation import SkillDraft
from nomorals.skills.registry import SkillRegistry
from nomorals.storage.db import Database


def _draft(name="distilled_test_skill", tools=("web_search", "summarize"),
           workflow=None):
    return SkillDraft(
        name=name,
        description="test a thing then summarize",
        tools=list(tools),
        workflow=workflow or (
            "1. Search with web_search for the topic\n"
            "2. Summarize the results with summarize"))


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _installed(reg, name):
    """Fetch any installed version (registry.get(name) only resolves the
    active pin)."""
    versions = reg.versions(name)
    assert versions, f"nothing installed for {name}"
    return reg.get(name, versions[-1])


class TablesTests(unittest.TestCase):
    def test_ensure_tables(self):
        db = _db()
        self.assertTrue(P.ensure_tables(db))
        row = db.query_one(
            "SELECT name FROM sqlite_master WHERE name='skill_proofs'")
        self.assertIsNotNone(row)
        row = db.query_one(
            "SELECT name FROM sqlite_master WHERE name='skill_flags'")
        self.assertIsNotNone(row)


class GenerateTestTests(unittest.TestCase):
    def test_structural_fallback_no_llm(self):
        code = P.generate_test(_draft())
        self.assertIsNotNone(code)
        self.assertIn("def test_", code)
        self.assertIn("web_search", code)

    def test_llm_code_used_when_valid(self):
        llm = lambda prompt: "def test_llm_one():\n    assert True\n"
        code = P.generate_test(_draft(), llm_fn=llm)
        self.assertIn("test_llm_one", code)

    def test_llm_fences_stripped(self):
        llm = lambda prompt: "```python\ndef test_fenced():\n    assert True\n```"
        code = P.generate_test(_draft(), llm_fn=llm)
        self.assertIn("def test_fenced", code)
        self.assertNotIn("```", code)

    def test_llm_garbage_falls_back_to_structural(self):
        llm = lambda prompt: "not python at all, just words"
        code = P.generate_test(_draft(), llm_fn=llm)
        self.assertIn("def test_", code)  # structural fallback

    def test_llm_raises_falls_back(self):
        def boom(prompt):
            raise RuntimeError("nope")
        code = P.generate_test(_draft(), llm_fn=boom)
        self.assertIn("def test_", code)

    def test_bad_draft_none(self):
        self.assertIsNone(P.generate_test(
            SkillDraft(name="", description="", workflow="")))
        self.assertIsNone(P.generate_test(None))


class RunProofTestTests(unittest.TestCase):
    def test_passing(self):
        passed, out = P.run_proof_test(
            "def test_ok():\n    assert 1 + 1 == 2\n")
        self.assertTrue(passed)
        self.assertIn("passed", out.lower())

    def test_failing(self):
        passed, out = P.run_proof_test(
            "def test_bad():\n    assert 1 + 1 == 3\n")
        self.assertFalse(passed)

    def test_no_test_functions(self):
        passed, out = P.run_proof_test("x = 1\n")
        self.assertFalse(passed)
        self.assertIn("no test functions", out)

    def test_garbage_never_raises(self):
        passed, out = P.run_proof_test("")
        self.assertFalse(passed)
        passed, out = P.run_proof_test(None)
        self.assertFalse(passed)

    def test_syntax_error_fails(self):
        passed, _ = P.run_proof_test("def test_broken(:\n    pass\n")
        self.assertFalse(passed)


class ProveSkillTests(unittest.TestCase):
    def test_prove_passes_good_draft(self):
        db = _db()
        proof = P.prove_skill(_draft(), db)
        self.assertTrue(proof.passed)
        self.assertEqual(proof.skill_name, "distilled_test_skill")
        self.assertTrue(P.has_proof("distilled_test_skill", db))

    def test_prove_fails_ungrounded_draft(self):
        db = _db()
        draft = _draft(workflow="1. Do something magical\n2. Profit somehow")
        proof = P.prove_skill(draft, db)
        self.assertFalse(proof.passed)
        self.assertFalse(P.has_proof(draft.name, db))

    def test_prove_with_explicit_test_code(self):
        db = _db()
        proof = P.prove_skill(
            _draft(), db,
            test_code="def test_explicit():\n    assert True\n")
        self.assertTrue(proof.passed)

    def test_prove_records_row(self):
        db = _db()
        P.prove_skill(_draft(), db)
        row = db.query_one(
            "SELECT * FROM skill_proofs WHERE skill_name=?",
            ("distilled_test_skill",))
        self.assertIsNotNone(row)
        self.assertEqual(row["passed"], 1)


class FlagTests(unittest.TestCase):
    def test_flag_and_get(self):
        db = _db()
        reg = SkillRegistry(db)
        reg.install(_draft().to_manifest())
        self.assertTrue(P.flag_skill("distilled_test_skill", "no proof", db, reg))
        flag = P.get_flag("distilled_test_skill", db)
        self.assertIsNotNone(flag)
        self.assertIn("no proof", flag["reason"])
        # disabled => runner refuses
        installed = _installed(reg, "distilled_test_skill")
        self.assertFalse(installed.enabled)

    def test_unflag_reenables_without_activating(self):
        db = _db()
        reg = SkillRegistry(db)
        reg.install(_draft().to_manifest())
        reg.deactivate("distilled_test_skill")
        P.flag_skill("distilled_test_skill", "bad", db, reg)
        P.unflag_skill("distilled_test_skill", db, reg)
        self.assertIsNone(P.get_flag("distilled_test_skill", db))
        installed = _installed(reg, "distilled_test_skill")
        self.assertTrue(installed.enabled)
        self.assertFalse(installed.active)

    def test_refuse_untested(self):
        db = _db()
        self.assertTrue(P.refuse_untested("distilled_test_skill", db))
        P.prove_skill(_draft(), db)
        self.assertFalse(P.refuse_untested("distilled_test_skill", db))


class PromoteTests(unittest.TestCase):
    def test_pass_activates(self):
        db = _db()
        reg = SkillRegistry(db)
        draft = _draft()
        reg.install(draft.to_manifest())
        reg.deactivate(draft.name)
        version = draft.to_manifest()["version"]
        proof = P.ProofResult(draft.name, version, True)
        P.record_proof(db, proof)
        outcome = P.promote_on_proof(draft, proof, reg, db)
        self.assertEqual(outcome, "active")
        self.assertTrue(reg.get(draft.name).active)

    def test_fail_flags_and_disables(self):
        db = _db()
        reg = SkillRegistry(db)
        draft = _draft()
        reg.install(draft.to_manifest())
        reg.deactivate(draft.name)
        version = draft.to_manifest()["version"]
        proof = P.ProofResult(draft.name, version, False,
                              reason="proof test failed")
        outcome = P.promote_on_proof(draft, proof, reg, db)
        self.assertEqual(outcome, "flagged")
        installed = _installed(reg, draft.name)
        self.assertFalse(installed.enabled)
        self.assertIsNotNone(P.get_flag(draft.name, db))


class LifecycleTests(unittest.TestCase):
    def test_unknown(self):
        self.assertEqual(P.skill_lifecycle("nope", _db()), "unknown")

    def test_draft_then_active(self):
        db = _db()
        reg = SkillRegistry(db)
        draft = _draft()
        reg.install(draft.to_manifest())
        reg.deactivate(draft.name)
        self.assertEqual(P.skill_lifecycle(draft.name, db, reg), "draft")
        P.prove_skill(draft, db)
        proof = P.ProofResult(draft.name, draft.to_manifest()["version"], True)
        P.record_proof(db, proof)
        P.promote_on_proof(draft, proof, reg, db)
        self.assertEqual(P.skill_lifecycle(draft.name, db, reg), "active")

    def test_flagged(self):
        db = _db()
        reg = SkillRegistry(db)
        draft = _draft()
        reg.install(draft.to_manifest())
        P.flag_skill(draft.name, "bad", db, reg)
        self.assertEqual(P.skill_lifecycle(draft.name, db, reg), "flagged")


class EnsureCapabilityTests(unittest.TestCase):
    def _llm(self, text):
        return lambda prompt: text

    def test_present_when_proven_active(self):
        db = _db()
        reg = SkillRegistry(db)
        draft = _draft(name="distilled_cap_x")
        reg.install(draft.to_manifest())
        version = draft.to_manifest()["version"]
        proof = P.ProofResult(draft.name, version, True)
        P.record_proof(db, proof)
        reg.pin(draft.name, version)
        out = P.ensure_capability("cap_x", "does x", db, reg)
        self.assertEqual(out["status"], "present")

    def test_drafts_proves_activates(self):
        db = _db()
        reg = SkillRegistry(db)
        text = ("NAME: cap_y\nDESCRIPTION: does y\nTOOLS: web_search\n"
                "WORKFLOW:\n1. Search with web_search\n")
        out = P.ensure_capability("cap_y", "does y", db, reg,
                                  llm_fn=self._llm(text))
        self.assertEqual(out["status"], "active")
        self.assertTrue(out["proof_passed"])
        self.assertTrue(reg.get("distilled_cap_y").active)

    def test_failed_proof_flags(self):
        db = _db()
        reg = SkillRegistry(db)
        # workflow steps don't name any tool -> structural test fails
        text = ("NAME: cap_z\nDESCRIPTION: does z\nTOOLS: web_search\n"
                "WORKFLOW:\n1. Think really hard\n2. Hope for the best\n")
        out = P.ensure_capability("cap_z", "does z", db, reg,
                                  llm_fn=self._llm(text))
        self.assertEqual(out["status"], "flagged")
        self.assertIsNotNone(P.get_flag("distilled_cap_z", db))

    def test_no_llm_fails(self):
        db = _db()
        out = P.ensure_capability("cap_w", "does w", db, None, llm_fn=None)
        self.assertEqual(out["status"], "failed")

    def test_empty_name_fails(self):
        db = _db()
        out = P.ensure_capability("", "x", db)
        self.assertEqual(out["status"], "failed")

    def test_never_raises(self):
        out = P.ensure_capability("cap_v", "x", None)
        self.assertEqual(out["status"], "failed")


class CanaryGateTests(unittest.TestCase):
    def _rollout(self, db):
        from nomorals.agents.skill_canary import CanaryRollout
        return CanaryRollout(SimpleNamespace(db=db))

    def test_proof_check_true_false(self):
        db = _db()
        check = P.canary_proof_check(db)
        self.assertFalse(check("s", "h123"))
        P.record_proof(db, P.ProofResult("s", "1", True, version_hash="h123"))
        self.assertTrue(check("s", "h123"))
        self.assertFalse(check("s", "other"))

    def test_evaluate_refuses_untested_promotion(self):
        db = _db()
        rollout = self._rollout(db)
        started = rollout.start("skill_a", canary_body="new body",
                                baseline_body="old body", fraction=1.0)
        self.assertTrue(started["ok"])
        run_id = started["id"]
        # force readiness: shrink thresholds via observations
        for _ in range(20):
            rollout.observe("skill_a", started["canary_hash"], True)
        res = rollout.evaluate(
            "skill_a", proof_check=P.canary_proof_check(db))
        self.assertEqual(res["decision"], "revert")
        self.assertIn("proof_gate", res)

    def test_evaluate_promotes_with_proof(self):
        db = _db()
        rollout = self._rollout(db)
        started = rollout.start("skill_b", canary_body="new body",
                                baseline_body="old body", fraction=1.0)
        P.record_proof(db, P.ProofResult(
            "skill_b", "1", True, version_hash=started["canary_hash"]))
        for _ in range(20):
            rollout.observe("skill_b", started["canary_hash"], True)
        res = rollout.evaluate(
            "skill_b", proof_check=P.canary_proof_check(db))
        self.assertEqual(res["decision"], "promote")

    def test_evaluate_backward_compatible_no_gate(self):
        db = _db()
        rollout = self._rollout(db)
        started = rollout.start("skill_c", canary_body="new body",
                                baseline_body="old body", fraction=1.0)
        for _ in range(20):
            rollout.observe("skill_c", started["canary_hash"], True)
        res = rollout.evaluate("skill_c")  # no proof_check -> old behavior
        self.assertEqual(res["decision"], "promote")


class MaybeDistillProveTests(unittest.TestCase):
    def _ctx(self, db):
        return SimpleNamespace(db=db, router=None)

    def _mem(self):
        steps = [SimpleNamespace(tool_name="web_search", thought="s")
                 for _ in range(6)]
        return SimpleNamespace(steps=steps, user_message="find gigs")

    def _res(self):
        return SimpleNamespace(success=True,
                               tools_called=[f"tool_{i}" for i in range(6)])

    def test_prove_true_activates_on_pass(self):
        from nomorals.agents import skill_distillation as D
        db = _db()
        llm = (lambda prompt:
               "NAME: gig_hunt\nDESCRIPTION: hunt gigs\n"
               "TOOLS: tool_0, tool_1\nWORKFLOW:\n"
               "1. Call tool_0 to search\n2. Call tool_1 to summarize\n")
        with mock.patch.object(
                D, "should_distill", return_value=True):
            draft = D.maybe_distill(self._res(), self._mem(),
                                    self._ctx(db), llm_fn=llm, db=db,
                                    prove=True)
        self.assertIsNotNone(draft)
        self.assertEqual(P.skill_lifecycle(draft.name, db,
                                           SkillRegistry(db)), "active")

    def test_prove_true_flags_on_fail(self):
        from nomorals.agents import skill_distillation as D
        db = _db()
        llm = (lambda prompt:
               "NAME: gig_hunt\nDESCRIPTION: hunt gigs\n"
               "TOOLS: tool_0\nWORKFLOW:\n"
               "1. Ponder deeply\n2. Hope for results\n")
        with mock.patch.object(
                D, "should_distill", return_value=True):
            draft = D.maybe_distill(self._res(), self._mem(),
                                    self._ctx(db), llm_fn=llm, db=db,
                                    prove=True)
        self.assertIsNotNone(draft)
        # proof LLM also gets the bad workflow -> structural test fails
        state = P.skill_lifecycle(draft.name, db, SkillRegistry(db))
        self.assertEqual(state, "flagged")

    def test_prove_false_keeps_old_behavior(self):
        from nomorals.agents import skill_distillation as D
        db = _db()
        llm = (lambda prompt:
               "NAME: gig_hunt\nDESCRIPTION: hunt gigs\n"
               "TOOLS: tool_0\nWORKFLOW:\n1. Call tool_0\n")
        with mock.patch.object(
                D, "should_distill", return_value=True):
            draft = D.maybe_distill(self._res(), self._mem(),
                                    self._ctx(db), llm_fn=llm, db=db)
        self.assertIsNotNone(draft)
        # default path: installed inactive, no proof attempted
        self.assertEqual(P.skill_lifecycle(draft.name, db,
                                           SkillRegistry(db)), "draft")


if __name__ == "__main__":
    unittest.main()
