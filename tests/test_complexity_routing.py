"""Build-map #18: multi-model routing by task complexity.

Covers: the heuristic complexity classifier (buckets, confidence
ordering, never-raises default), the TaskRouter complexity bias,
per-call cost logging (token math, USD estimates, JSONL read-back),
the Architect/Editor coding split (different tiers with routing on,
identical behavior with routing off), route_by_complexity end to end,
and the research-budget measured-cost true-up.  All offline with fake
providers — nothing here may touch the network or the real home dir.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import nomorals.llm.router as router_mod
from nomorals.agents.complexity import COMPLEXITIES, classify_complexity
from nomorals.agents.router_select import ModelProfile, TaskRouter
from nomorals.llm.base import Message
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import (
    CostLog,
    LLMRouter,
    estimate_cost,
    get_cost_log,
    log_llm_call,
    total_spend,
)


# ── fakes ────────────────────────────────────────────────────────────────────

def _router_two() -> LLMRouter:
    """mock (local/free) + groq (cloud/paid), mock primary."""
    r = LLMRouter()
    r.add(MockProvider(model="mock-7b"), name="mock", primary=True)
    r.add(MockProvider(model="llama-3.3-70b-versatile"), name="groq")
    return r


def _context(router: LLMRouter, intelligent: str = "on") -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(router_intelligent=intelligent),
        router=router,
        db=SimpleNamespace(),  # CodingAgent only stores the reference
    )


class _CostLogHome(unittest.TestCase):
    """Redirect the module-global cost log at ~/.nomorals into tmp_path."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._old = router_mod._cost_log
        router_mod._cost_log = CostLog(
            f"{self._tmp.name}/cost.jsonl")

    def tearDown(self):
        router_mod._cost_log = self._old
        self._tmp.cleanup()


# ── classifier ───────────────────────────────────────────────────────────────

class ClassifierTests(unittest.TestCase):
    def test_easy_bucket(self):
        for text in ("what time is it", "translate hello to Spanish",
                     "hi", "good morning"):
            level, conf = classify_complexity(text)
            self.assertEqual(level, "easy", text)
            self.assertGreaterEqual(conf, 0.5)

    def test_medium_bucket(self):
        for text in ("Write a function that sorts a list of dicts by key",
                     "how do I debug a segfault in my C extension?",
                     "explain how TCP handshakes work"):
            level, _ = classify_complexity(text)
            self.assertEqual(level, "medium", text)

    def test_hard_bucket(self):
        for text in (
            "Design the architecture for a distributed task queue; "
            "discuss the trade-offs",
            "Refactor the auth module across 12 files",
            "Design the rollout strategy for migrating 40 services to "
            "Kubernetes, with trade-offs",
        ):
            level, conf = classify_complexity(text)
            self.assertEqual(level, "hard", text)
            self.assertGreater(conf, 0.7)

    def test_unknown_defaults_to_medium(self):
        self.assertEqual(classify_complexity("")[0], "medium")
        self.assertEqual(classify_complexity("   ")[0], "medium")
        level, conf = classify_complexity("tell me about dogs")
        self.assertEqual(level, "medium")
        self.assertEqual(conf, 0.5)

    def test_confidence_ordering(self):
        _, strong = classify_complexity("what time is it")
        _, vague = classify_complexity("tell me about dogs")
        self.assertGreater(strong, vague)

    def test_never_raises(self):
        self.assertEqual(classify_complexity(None)[0], "medium")  # type: ignore[arg-type]
        self.assertEqual(classify_complexity(12345)[0], "medium")  # type: ignore[arg-type]

    def test_task_type_hint(self):
        level, _ = classify_complexity("refactor this", task_type="coding")
        # "refactor" is a hard signal; the coding hint must not demote it
        self.assertIn(level, ("medium", "hard"))

    def test_complexities_constant(self):
        self.assertEqual(COMPLEXITIES, ("easy", "medium", "hard"))


# ── TaskRouter complexity bias ───────────────────────────────────────────────

class ComplexityBiasTests(unittest.TestCase):
    def setUp(self):
        self.tr = TaskRouter(_context(_router_two(), "on"))

    def test_easy_prefers_local_free(self):
        choice = self.tr.select(task_type="chat", objective="balanced",
                                complexity="easy")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.name, "mock")

    def test_hard_prefers_quality(self):
        choice = self.tr.select(task_type="chat", objective="balanced",
                                complexity="hard")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.name, "groq")

    def test_medium_unchanged_scoring(self):
        mock = next(p for p in self.tr.profiles(refresh=False)
                    if p.name == "mock")
        groq = next(p for p in self.tr.profiles(refresh=False)
                    if p.name == "groq")
        # complexity=None must score exactly like the old two-arg form
        self.assertEqual(
            self.tr.score(mock, "chat", "balanced"),
            self.tr.score(mock, "chat", "balanced", complexity=None))
        self.assertEqual(
            self.tr.score(groq, "chat", "balanced", complexity="medium"),
            self.tr.score(groq, "chat", "balanced"))

    def test_score_bias_magnitude(self):
        mock = ModelProfile(name="mock", local=True, free=True,
                            capabilities=frozenset({"chat"}))
        cloud = ModelProfile(name="groq",
                             capabilities=frozenset({"chat"}))
        base_mock = self.tr.score(mock, "chat", "balanced")
        base_cloud = self.tr.score(cloud, "chat", "balanced")
        self.assertEqual(
            self.tr.score(mock, "chat", "balanced", complexity="easy"),
            base_mock + 2.0)
        self.assertEqual(
            self.tr.score(cloud, "chat", "balanced", complexity="hard"),
            base_cloud + 2.0)
        # medium adds nothing
        self.assertEqual(
            self.tr.score(mock, "chat", "balanced", complexity="medium"),
            base_mock)

    def test_decision_reports_complexity(self):
        d = self.tr.decision("chat", "balanced", complexity="hard")
        self.assertEqual(d["complexity"], "hard")
        self.assertEqual(d["choice"], "groq")

    def test_routing_off_ignores_complexity(self):
        tr = TaskRouter(_context(_router_two(), "off"))
        self.assertIsNone(tr.select("chat", "balanced", complexity="hard"))


# ── route_by_complexity ──────────────────────────────────────────────────────

class RouteByComplexityTests(unittest.TestCase):
    def test_end_to_end_with_fake_providers(self):
        tr = TaskRouter(_context(_router_two(), "on"))
        name, level = tr.route_by_complexity("what time is it")
        self.assertEqual((name, level), ("mock", "easy"))
        name, level = tr.route_by_complexity(
            "Design the architecture for a distributed task queue; "
            "discuss the trade-offs")
        self.assertEqual((name, level), ("groq", "hard"))

    def test_routing_off_returns_active_chain(self):
        tr = TaskRouter(_context(_router_two(), "off"))
        name, level = tr.route_by_complexity("what time is it")
        self.assertEqual(level, "easy")   # classification still reported
        self.assertEqual(name, "mock")    # chain's active provider

    def test_task_type_override(self):
        tr = TaskRouter(_context(_router_two(), "on"))
        name, level = tr.route_by_complexity("do it", task_type="coding")
        self.assertEqual(level, "medium")
        self.assertTrue(name)


# ── cost logging ─────────────────────────────────────────────────────────────

class EstimateCostTests(unittest.TestCase):
    def test_token_math(self):
        # groq: $0.35 / $0.40 per 1M tokens
        self.assertAlmostEqual(
            estimate_cost("groq", "llama-3.3-70b-versatile", 1_000_000, 1_000_000),
            0.75)

    def test_unknown_falls_back_to_flat_estimate(self):
        from nomorals.research.pipeline import COST_TABLE
        self.assertAlmostEqual(
            estimate_cost("mystery", "mystery-model", 100, 100),
            COST_TABLE["llm_call"])

    def test_free_tiers_cost_nothing(self):
        self.assertEqual(
            estimate_cost("llama_cpp", "qwen2.5.gguf", 1_000_000, 1_000_000), 0.0)
        self.assertEqual(
            estimate_cost("mock", "mock-7b", 500, 500), 0.0)

    def test_never_raises(self):
        self.assertGreaterEqual(estimate_cost(None, None, -1, -1), 0.0)  # type: ignore[arg-type]


class CostLogTests(_CostLogHome):
    def test_dispatch_logs_call(self):
        r = _router_two()
        resp = r.chat([Message.user("hello")], tier="easy")
        self.assertTrue(resp.ok)
        rows = get_cost_log()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["provider"], "mock")
        self.assertEqual(row["model"], "mock-7b")
        self.assertEqual(row["tier"], "easy")
        self.assertEqual(row["operation"], "chat")
        self.assertGreater(row["prompt_tokens"], 0)
        self.assertGreater(row["completion_tokens"], 0)
        self.assertEqual(row["cost_usd"], 0.0)  # mock is free

    def test_total_spend_sums_and_filters(self):
        r = _router_two()
        before = total_spend()
        r.chat([Message.user("one")])
        r.chat([Message.user("two")])
        # free provider: spend stays flat but rows accumulate
        self.assertEqual(total_spend(), before)
        self.assertEqual(len(get_cost_log()), 2)
        # since=future → nothing
        import time
        self.assertEqual(get_cost_log(since=time.time() + 3600), [])
        self.assertEqual(total_spend(since=time.time() + 3600), 0.0)

    def test_log_llm_call_returns_usd(self):
        r = _router_two()
        resp = r.chat([Message.user("hi")])
        # _dispatch already logged it; direct helper call adds one more row
        cost = log_llm_call("groq", "llama-3.3-70b-versatile", resp,
                            tier="hard", operation="chat")
        self.assertGreater(cost, 0.0)
        rows = get_cost_log()
        self.assertEqual(rows[-1]["tier"], "hard")
        self.assertGreater(rows[-1]["cost_usd"], 0.0)

    def test_complete_logs_too(self):
        r = _router_two()
        resp = r.complete("finish this", tier="medium")
        self.assertTrue(resp.ok)
        rows = get_cost_log()
        self.assertEqual(rows[-1]["operation"], "complete")
        self.assertEqual(rows[-1]["tier"], "medium")

    def test_cost_log_to_explicit_path(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
        log = CostLog(path)
        log.record(provider="groq", model="m", tier="hard",
                   prompt_tokens=1_000_000, completion_tokens=1_000_000,
                   cost_usd=0.75)
        self.assertEqual(log.total_spend(), 0.75)
        self.assertEqual(len(log.entries()), 1)
        # malformed lines are skipped, missing file reads empty
        with open(path, "a") as fh:
            fh.write("not json\n")
        self.assertEqual(len(log.entries()), 1)
        self.assertEqual(CostLog(path + ".missing").total_spend(), 0.0)


# ── Architect/Editor split ───────────────────────────────────────────────────

class ArchitectEditorTests(_CostLogHome):
    def _agent(self, intelligent: str):
        from nomorals.agents.coding import CodingAgent
        return CodingAgent(_context(_router_two(), intelligent))

    def test_routing_off_selects_default_chain(self):
        agent = self._agent("off")
        self.assertIs(agent._select_model("plan"), agent.router)
        self.assertIs(agent._select_model("edit"), agent.router)

    def test_routing_on_splits_tiers(self):
        agent = self._agent("on")
        architect = agent._select_model("plan")
        editor = agent._select_model("edit")
        # plan → hard/quality → the strong cloud model
        self.assertEqual(architect.name, "groq")
        # edit → medium/cost → the free local model
        self.assertEqual(editor.name, "mock")
        self.assertIsNot(architect, editor)

    def test_phase_chat_falls_back_to_chain(self):
        from nomorals.agents.coding import CodingAgent

        class _AlwaysFails(MockProvider):
            def chat(self, messages, params=None, **kw):
                from nomorals.llm.base import LLMResponse
                return LLMResponse(text="", model=self.model,
                                   error="simulated failure")

        r = LLMRouter()
        r.add(MockProvider(model="mock-7b"), name="mock", primary=True)
        r.add(_AlwaysFails(model="big"), name="groq")
        agent = CodingAgent(_context(r, "on"))
        # plan phase picks groq (hard/quality) which fails → chain serves via mock
        resp = agent._phase_chat("plan", [Message.user("hi")], tier="hard")
        self.assertTrue(resp.ok)

    def test_select_model_never_raises(self):
        agent = self._agent("on")
        agent.context.settings = None  # hostile context
        self.assertIs(agent._select_model("plan"), agent.router)


# ── research budget measured-cost true-up ────────────────────────────────────

class BudgetTrueUpTests(_CostLogHome):
    def test_measured_cheaper_than_estimate_refunds(self):
        from nomorals.research.pipeline import (
            COST_TABLE, ResearchBudget, _budgeted_llm, _router_llm_fn)
        r = _router_two()
        llm_fn = _router_llm_fn(r)
        budget = ResearchBudget(10.0)
        bllm = _budgeted_llm(llm_fn, budget)
        bllm("test prompt")
        # mock costs $0.00 < $0.0008 estimate → estimate refunded in full
        self.assertEqual(budget.spent, 0.0)
        self.assertFalse(budget.exhausted)

    def test_measured_pricier_than_estimate_charges_delta(self):
        from nomorals.research.pipeline import (
            COST_TABLE, ResearchBudget, _budgeted_llm)
        usage = SimpleNamespace(prompt_tokens=1_000_000,
                                completion_tokens=1_000_000)
        resp = SimpleNamespace(provider="groq",
                               model="llama-3.3-70b-versatile",
                               usage=usage)

        def llm_fn(prompt: str) -> str:
            return "ok"
        llm_fn.last_response = resp  # type: ignore[attr-defined]

        budget = ResearchBudget(10.0)
        bllm = _budgeted_llm(llm_fn, budget)
        bllm("x")
        # pre-charge 0.0008 + delta (0.75 - 0.0008) = 0.75 exactly once
        self.assertAlmostEqual(budget.spent, 0.75)

    def test_no_usage_keeps_flat_estimate(self):
        from nomorals.research.pipeline import (
            COST_TABLE, ResearchBudget, _budgeted_llm)

        def llm_fn(prompt: str) -> str:
            return "ok"  # no last_response attribute at all

        budget = ResearchBudget(10.0)
        _budgeted_llm(llm_fn, budget)("x")
        self.assertAlmostEqual(budget.spent, COST_TABLE["llm_call"])

    def test_refund_clamps_at_zero(self):
        from nomorals.research.pipeline import ResearchBudget
        b = ResearchBudget(1.0)
        b.charge("web_search")
        self.assertAlmostEqual(b.refund(0.0005), 0.0005)
        self.assertEqual(b.refund(99.0), 0.0)
        self.assertEqual(b.spent, 0.0)

    def test_exhaustion_still_guards(self):
        from nomorals.research.pipeline import (
            ResearchBudget, _BudgetExhausted, _budgeted_llm)
        budget = ResearchBudget(0.0001)  # can't afford even the estimate
        bllm = _budgeted_llm(lambda p: "ok", budget)
        with self.assertRaises(_BudgetExhausted):
            bllm("x")
        self.assertTrue(budget.exhausted)


if __name__ == "__main__":
    unittest.main()
