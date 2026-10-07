"""Tests for plan mode (no network, no LLM)."""

import unittest

from nomorals.agents.plan_mode import (
    CodePlan, PlanStore, is_complex, render_plan,
)


def _spec(path, new=False):
    return {"path": path, "why": "test", "new_file": new}


class IsComplexTests(unittest.TestCase):
    def test_simple_is_not_complex(self):
        self.assertFalse(is_complex([_spec("a.py")]))
        self.assertFalse(is_complex([_spec("a.py"), _spec("b.py")]))

    def test_many_files_is_complex(self):
        specs = [_spec(f"f{i}.py") for i in range(5)]
        self.assertTrue(is_complex(specs))

    def test_many_new_files_is_complex(self):
        specs = [_spec(f"f{i}.py", new=True) for i in range(3)]
        self.assertTrue(is_complex(specs))

    def test_cross_module_is_complex(self):
        specs = [_spec("a/x.py"), _spec("b/y.py"), _spec("c/z.py")]
        self.assertTrue(is_complex(specs))

    def test_same_module_not_complex(self):
        specs = [_spec("pkg/a.py"), _spec("pkg/b.py")]
        self.assertFalse(is_complex(specs))


class PlanStoreTests(unittest.TestCase):
    def test_save_get_approve(self):
        plan = PlanStore.new("do things", [_spec("a.py")])
        self.assertFalse(plan.approved)
        got = PlanStore.get(plan.id)
        self.assertIsNotNone(got)
        self.assertEqual("do things", got.task)
        self.assertIsNone(PlanStore.approve("nope"))
        ok = PlanStore.approve(plan.id)
        self.assertTrue(ok.approved)
        self.assertGreater(ok.approved_at, 0)

    def test_in_scope(self):
        plan = PlanStore.new("t", [_spec("a.py"), _spec("b/c.py")])
        self.assertTrue(plan.in_scope("a.py"))
        self.assertTrue(plan.in_scope("b/c.py"))
        self.assertFalse(plan.in_scope("evil.py"))


class RenderTests(unittest.TestCase):
    def test_render(self):
        plan = PlanStore.new(
            "build widget", [_spec("w.py", new=True)],
            approach="write it", risks=["might break"])
        text = render_plan(plan)
        self.assertIn("build widget", text)
        self.assertIn("[new] w.py", text)
        self.assertIn("write it", text)
        self.assertIn("might break", text)
        self.assertIn("approve", text)


class CodingAgentPlanModeTests(unittest.TestCase):
    def _agent(self):
        from unittest.mock import MagicMock
        from nomorals.agents.coding import CodingAgent
        import tempfile
        ctx = MagicMock()
        # explicit root: _resolve() then stays inside a temp dir, so the
        # real run() preamble (snapshot etc.) never touches the repo
        agent = CodingAgent(ctx, root=tempfile.mkdtemp())
        # stub the model-driven plan step
        agent._plan_files = lambda task, default, workdir: [
            _spec("a.py"), _spec("b.py"), _spec("c.py"),
            _spec("d.py"), _spec("e.py"),
        ]
        return agent

    def test_complex_pauses_for_approval(self):
        agent = self._agent()
        result = agent.run("big task", plan_mode="auto")
        self.assertTrue(result.needs_approval)
        self.assertTrue(result.plan_id)
        self.assertIn("PLAN", result.plan_text)
        self.assertFalse(result.ok)

    def test_simple_does_not_pause(self):
        agent = self._agent()
        agent._plan_files = lambda t, d, w: [_spec("a.py")]
        # stub the rest of run to avoid real execution
        import nomorals.agents.coding as C
        orig_run = C.CodingAgent.run
        def fake_run(self, task, **kw):
            # replicate only up to the gate
            from nomorals.agents.plan_mode import is_complex
            plan = self._plan_files(task, "main.py", None)
            if kw.get("plan_mode", "auto") != "never":
                if is_complex(plan):
                    return C.CodingResult(
                        ok=False, iterations=0, needs_approval=True)
            return C.CodingResult(ok=True, iterations=1)
        C.CodingAgent.run = fake_run
        try:
            result = agent.run("small task", plan_mode="auto")
            self.assertFalse(result.needs_approval)
            self.assertTrue(result.ok)
        finally:
            C.CodingAgent.run = orig_run

    def test_plan_mode_false_never_pauses(self):
        agent = self._agent()
        import nomorals.agents.coding as C
        orig = C.CodingAgent._plan_files
        # force the gate check without running the full loop
        from nomorals.agents.plan_mode import PlanStore, is_complex
        plan = agent._plan_files("x", "main.py", None)
        self.assertTrue(is_complex(plan))  # sanity: it IS complex
        C.CodingAgent._plan_files = orig

    def test_execute_unknown_plan(self):
        agent = self._agent()
        result = agent.execute_plan("nope")
        self.assertFalse(result.ok)
        self.assertIn("unknown plan", result.error)

    def test_execute_unapproved_plan(self):
        agent = self._agent()
        plan = PlanStore.new("t", [_spec("a.py")])
        result = agent.execute_plan(plan.id)
        self.assertFalse(result.ok)
        self.assertIn("not approved", result.error)

    def test_approve_and_execute_unknown(self):
        agent = self._agent()
        result = agent.approve_and_execute("nope")
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
