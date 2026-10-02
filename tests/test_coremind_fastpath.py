"""Fast-path discipline for the Core Mind (wave D stream A).

- simple interactive requests never call the model router and return fast
- a stalled router can never block the chat thread: _model_check is bounded
  by MODEL_CHECK_TIMEOUT_S and the deterministic intent stands on timeout —
  loudly (logged + counted), never silently swallowed
- heuristic planning always carries an explicit plan_error (planner.py +
  devon.py)
- timeout budgets are named constants, interactive short / background long
- mission execution runs off the chat thread
"""
from __future__ import annotations

import asyncio
import time
import types
import unittest

from nomorals.agents.coremind import (
    CoreMind,
    CODING_JOB_TIMEOUT_S,
    MODEL_CHECK_TIMEOUT_S,
)
from nomorals.agents.planner import AgentPlanner


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class FakeContext:
    def __init__(self, router=None):
        self.settings = FakeSettings()
        self.extras = {}
        self.memory = None
        self.router = router


class SpyRouter:
    """Explodes if touched — the fast path must never need the model."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, params=None, **kw):
        self.calls += 1
        raise AssertionError("fast path must not call the router")


class SlowRouter:
    """Simulates a stalled provider chain."""

    def __init__(self, hang=30.0):
        self.hang = hang

    def complete(self, prompt, params=None, **kw):
        time.sleep(self.hang)
        return types.SimpleNamespace(ok=True, text='{"kind": "chat"}')


class ChattyRouter:
    """Returns a research verdict with high confidence."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, params=None, **kw):
        self.calls += 1
        return types.SimpleNamespace(
            ok=True,
            text='{"kind": "research", "target": "fusion reactors", '
                 '"confidence": 0.95, "why": "user asked to research"}')


class FailingRouter:
    def complete(self, prompt, params=None, **kw):
        raise RuntimeError("provider down")


def owner_dm():
    chat = types.SimpleNamespace(key="x:console", kind="dm")
    return types.SimpleNamespace(chat=chat)


class _FailingLLM:
    async def generate(self, prompt, max_tokens=2000):
        raise RuntimeError("hub is down")


class _GarbageLLM:
    async def generate(self, prompt, max_tokens=2000):
        return types.SimpleNamespace(content="not json at all")


class _GoodLLM:
    async def generate(self, prompt, max_tokens=2000):
        return types.SimpleNamespace(content=(
            '{"steps": [{"step_id": "s1", "name": "Wait a beat", '
            '"description": "pause", "action": "wait", '
            '"parameters": {"seconds": 0}, "depends_on": []}]}'))


# ── the fast path ───────────────────────────────────────────────────────────

class FastPathTest(unittest.TestCase):
    def test_simple_inputs_never_call_router_and_stay_fast(self):
        router = SpyRouter()
        mind = CoreMind(FakeContext(router=router), runtime=None)
        for text in ["hello", "what time is it", "research fusion reactors",
                     "build me a todo app", "status"]:
            t0 = time.perf_counter()
            mind.decide(text)
            dt = time.perf_counter() - t0
            self.assertLess(dt, 1.0, f"{text!r} took {dt:.2f}s on the fast path")
        self.assertEqual(router.calls, 0)
        self.assertEqual(mind._router_calls, 0)

    def test_chat_falls_through_fast(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        t0 = time.perf_counter()
        reply = mind.handle("hello", message=owner_dm(), chat_key="x:console")
        dt = time.perf_counter() - t0
        self.assertIsNone(reply)  # chat -> normal conversation flow
        self.assertLess(dt, 1.0)


class ModelCheckBoundTest(unittest.TestCase):
    def test_timeout_stays_deterministic(self):
        mind = CoreMind(FakeContext(router=SlowRouter(hang=30.0)), runtime=None)
        t0 = time.perf_counter()
        intent = mind.decide("build something")  # conf 0.6 -> model band
        dt = time.perf_counter() - t0
        self.assertEqual(intent.kind, "build",
                         "timeout must keep the deterministic intent")
        self.assertAlmostEqual(intent.confidence, 0.6)
        self.assertLess(dt, MODEL_CHECK_TIMEOUT_S + 3.0,
                        f"chat thread stalled {dt:.1f}s")
        self.assertEqual(mind._router_timeouts, 1)
        self.assertTrue(mind._model_check_note,
                        "timeout must be surfaced, not swallowed")

    def test_success_still_routes(self):
        mind = CoreMind(FakeContext(router=ChattyRouter()), runtime=None)
        intent = mind.decide("build something")
        self.assertEqual(intent.kind, "research")
        self.assertEqual(intent.route, "research_swarm")

    def test_failure_returns_deterministic(self):
        mind = CoreMind(FakeContext(router=FailingRouter()), runtime=None)
        t0 = time.perf_counter()
        intent = mind.decide("build something")
        dt = time.perf_counter() - t0
        self.assertEqual(intent.kind, "build")
        self.assertEqual(mind._router_timeouts, 0)  # failure, not timeout
        self.assertLess(dt, 5.0)

    def test_status_reports_model_timeouts(self):
        mind = CoreMind(FakeContext(router=SlowRouter(hang=30.0)), runtime=None)
        mind.decide("build something")
        text = mind.status()
        self.assertIn("timed out 1×", text)
        self.assertIn("last model check", text)


class TimeoutBudgetsTest(unittest.TestCase):
    def test_constants_sane(self):
        # interactive = fail fast (seconds); background jobs get a long budget
        self.assertGreater(MODEL_CHECK_TIMEOUT_S, 0)
        self.assertLessEqual(MODEL_CHECK_TIMEOUT_S, 15,
                             f"interactive budget too big: {MODEL_CHECK_TIMEOUT_S}s")
        self.assertGreaterEqual(CODING_JOB_TIMEOUT_S, 60,
                                f"background budget too small: {CODING_JOB_TIMEOUT_S}s")
        self.assertGreater(CODING_JOB_TIMEOUT_S, MODEL_CHECK_TIMEOUT_S * 4)


class MissionDispatchTest(unittest.TestCase):
    def test_mission_run_does_not_block_chat_thread(self):
        mind = CoreMind(FakeContext(router=None), runtime=None)

        class SlowDirectives:
            def run(self, did):
                time.sleep(5.0)
                return {"ok": True, "id": "abc12345", "result": "done"}

        mind._directives = lambda: SlowDirectives()
        t0 = time.perf_counter()
        reply = mind.handle("run my mission", message=owner_dm(),
                            chat_key="x:console")
        dt = time.perf_counter() - t0
        self.assertLess(dt, 2.0,
                        f"mission run blocked the chat thread {dt:.1f}s")
        self.assertIn("running the mission", reply or "")


# ── plan_error ──────────────────────────────────────────────────────────────

class PlannerPlanErrorTest(unittest.TestCase):
    def test_template_plan_sets_plan_error_no_llm(self):
        planner = AgentPlanner(llm_router=None)
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertTrue(result.plan_error,
                        "template fallback must set plan_error")
        self.assertIn("template", result.summary.lower())
        self.assertEqual(result.to_dict()["plan_error"], result.plan_error)

    def test_template_plan_sets_plan_error_when_llm_fails(self):
        planner = AgentPlanner(llm_router=_FailingLLM())
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertTrue(result.plan_error,
                        "LLM failure must surface as plan_error")
        self.assertIn("template", result.plan_error.lower())

    def test_template_plan_sets_plan_error_when_llm_garbles(self):
        planner = AgentPlanner(llm_router=_GarbageLLM())
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertIn("no JSON", result.plan_error)

    def test_model_plan_has_no_plan_error(self):
        planner = AgentPlanner(llm_router=_GoodLLM())
        result = asyncio.run(planner.execute(goal="do it", account="owner"))
        self.assertEqual(result.plan_error, "")
        self.assertNotIn("template", result.summary.lower())


class DevonPlanErrorTest(unittest.TestCase):
    def test_empty_task_carries_plan_error(self):
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(FakeContext())
        result = agent.run("")
        self.assertEqual(result.planned_by, "heuristic")
        self.assertTrue(result.plan_error,
                        "heuristic plan must explain itself")

    def test_heuristic_fallback_never_empty_error(self):
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(FakeContext())  # no router -> heuristic path
        result = agent.run("check the repo state")
        self.assertIn(result.planned_by, ("heuristic", "reasoning", "llm"))
        if result.planned_by == "heuristic":
            self.assertTrue(result.plan_error,
                            "silent heuristic success is forbidden")


if __name__ == "__main__":
    unittest.main()
