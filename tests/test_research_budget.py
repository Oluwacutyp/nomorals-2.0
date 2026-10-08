"""Build-map #7 — budgeted research dial: spend as the stopping condition.

Offline: registries are faked with scripted ``call``/``call_many`` outcomes.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from nomorals.core.result import Err, Ok
from nomorals.research.pipeline import (
    COST_TABLE,
    ResearchBudget,
    ResearchContext,
    ResearchJob,
    research_deep,
    run_job,
)
from nomorals.storage.db import Database


class _ScriptedRegistry:
    """Fake registry with order-preserving call_many."""

    def __init__(self, search=None, fetch_text="fetched full text"):
        self.search = search or {}
        self.fetch_text = fetch_text
        self.calls = []

    def call(self, name, *, actor="system", **kwargs):
        self.calls.append((name, kwargs))
        if name == "web_search":
            return Ok({"results": self.search.get(kwargs.get("query", ""), [])})
        if name == "web_fetch":
            return Ok({"url": kwargs.get("url", ""), "text": self.fetch_text})
        return Err(ValueError(f"unknown tool {name}"))

    def call_many(self, calls, *, max_workers=1, **common):
        return [self.call(name, **{**common, **kwargs})
                for name, kwargs in calls]


def _rctx(registry):
    return ResearchContext(db=Database(":memory:"), registry=registry)


def _results(*urls):
    return [{"title": f"Title for {u} which is long enough",
             "url": u,
             "snippet": f"Snippet about {u} with enough words to be real."}
            for u in urls]


def _job(queries, **kw):
    return ResearchJob(id="j1", topic="t", queries=list(queries), **kw)


# ── ResearchBudget ───────────────────────────────────────────────────────────

class BudgetMathTests(unittest.TestCase):
    def test_charge_spent_remaining(self):
        b = ResearchBudget(0.01)
        self.assertTrue(b.charge("web_search"))
        self.assertAlmostEqual(b.spent, COST_TABLE["web_search"])
        self.assertAlmostEqual(b.remaining, 0.01 - COST_TABLE["web_search"])
        self.assertFalse(b.exhausted)

    def test_exhaustion_never_goes_negative(self):
        b = ResearchBudget(0.002)
        self.assertTrue(b.charge("web_search"))   # 0.001 spent
        self.assertTrue(b.charge("web_search"))   # 0.002 spent
        # next charge would exceed: refused, balance untouched, latched
        self.assertFalse(b.charge("web_search"))
        self.assertAlmostEqual(b.spent, 0.002)
        self.assertEqual(b.remaining, 0.0)
        self.assertTrue(b.exhausted)
        # stays refused once exhausted
        self.assertFalse(b.charge("web_search"))
        self.assertAlmostEqual(b.spent, 0.002)

    def test_explicit_amount(self):
        b = ResearchBudget(1.0)
        self.assertTrue(b.charge("custom", amount=0.25))
        self.assertAlmostEqual(b.spent, 0.25)

    def test_unknown_op_raises(self):
        b = ResearchBudget(1.0)
        with self.assertRaises(KeyError):
            b.charge("no_such_op")

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            ResearchBudget(-1.0)

    def test_zero_budget_charges_nothing(self):
        b = ResearchBudget(0.0)
        self.assertFalse(b.charge("web_search"))
        self.assertTrue(b.exhausted)

    def test_spent_usd_rounding(self):
        b = ResearchBudget(1.0)
        b.charge("web_search")
        self.assertEqual(b.spent_usd(), round(COST_TABLE["web_search"], 6))

    def test_thread_safety(self):
        # 100 threads x 20 charges of 0.001 on a $1.00 budget: exactly
        # 1000 charges may succeed; the balance must never exceed the cap.
        b = ResearchBudget(1.0)
        ok_count = [0]
        count_lock = threading.Lock()

        def hammer():
            local = 0
            for _ in range(20):
                if b.charge("web_search"):
                    local += 1
            with count_lock:
                ok_count[0] += local

        threads = [threading.Thread(target=hammer) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(b.spent, 1.0)
        self.assertAlmostEqual(b.spent, ok_count[0] * COST_TABLE["web_search"])
        self.assertTrue(b.exhausted)


# ── run_job with a budget ────────────────────────────────────────────────────

class RunJobBudgetTests(unittest.TestCase):
    def test_stops_issuing_searches_when_exhausted_but_keeps_findings(self):
        reg = _ScriptedRegistry(search={
            "q1": _results("https://a.example/1"),
            "q2": _results("https://a.example/2"),
            "q3": _results("https://a.example/3"),
            "q4": _results("https://a.example/4"),
        })
        budget = ResearchBudget(0.002)  # exactly two web_search calls
        findings = run_job(_job(["q1", "q2", "q3", "q4"]), _rctx(reg),
                           budget=budget)
        search_calls = [c for c in reg.calls if c[0] == "web_search"]
        self.assertEqual(len(search_calls), 2)
        self.assertEqual(len(findings), 2)
        # fetch phase had nothing left: no fetches issued, findings kept
        fetch_calls = [c for c in reg.calls if c[0] == "web_fetch"]
        self.assertEqual(fetch_calls, [])
        self.assertTrue(budget.exhausted)

    def test_budget_exhausted_before_any_search_returns_empty_no_raise(self):
        reg = _ScriptedRegistry(search={"q1": _results("https://a.example/1")})
        budget = ResearchBudget(0.0001)  # less than one search
        findings = run_job(_job(["q1"]), _rctx(reg), budget=budget)
        self.assertEqual(findings, [])
        self.assertEqual(reg.calls, [])
        self.assertTrue(budget.exhausted)

    def test_unlimited_budget_unchanged(self):
        reg = _ScriptedRegistry(search={
            "q1": _results("https://a.example/1"),
            "q2": _results("https://a.example/2"),
        })
        findings = run_job(_job(["q1", "q2"]), _rctx(reg))
        self.assertEqual(len(findings), 2)
        self.assertEqual(len([c for c in reg.calls if c[0] == "web_search"]), 2)

    def test_job_budget_usd_honored_without_explicit_budget(self):
        reg = _ScriptedRegistry(search={
            "q1": _results("https://a.example/1"),
            "q2": _results("https://a.example/2"),
        })
        findings = run_job(_job(["q1", "q2"], budget_usd=0.001), _rctx(reg))
        self.assertEqual(len(findings), 1)
        self.assertEqual(len([c for c in reg.calls if c[0] == "web_search"]), 1)

    def test_explicit_budget_wins_over_job_budget(self):
        reg = _ScriptedRegistry(search={
            "q1": _results("https://a.example/1"),
            "q2": _results("https://a.example/2"),
        })
        budget = ResearchBudget(10.0)
        findings = run_job(_job(["q1", "q2"], budget_usd=0.001), _rctx(reg),
                           budget=budget)
        self.assertEqual(len(findings), 2)
        self.assertLess(budget.spent, 10.0)
        self.assertFalse(budget.exhausted)

    def test_fetch_phase_capped_separately(self):
        reg = _ScriptedRegistry(search={
            "q1": _results("https://a.example/1", "https://a.example/2"),
        })
        # one search + one fetch affordable
        budget = ResearchBudget(COST_TABLE["web_search"] + COST_TABLE["web_fetch"])
        findings = run_job(_job(["q1"], fetch_top=2), _rctx(reg), budget=budget)
        self.assertEqual(len(findings), 2)
        fetch_calls = [c for c in reg.calls if c[0] == "web_fetch"]
        self.assertEqual(len(fetch_calls), 1)
        self.assertTrue(findings[0].detail)
        self.assertFalse(findings[1].detail)


# ── research_deep with a budget ──────────────────────────────────────────────

class ResearchDeepBudgetTests(unittest.TestCase):
    Q = "best noise cancelling headphones 2026"  # sharp: heuristic clarify → []

    def _reg(self):
        return _ScriptedRegistry(search={
            "best noise cancelling headphones 2026": _results("https://a.example/1"),
            "best noise cancelling headphones 2026 best practices how to":
                _results("https://b.example/2"),
        })

    def test_tiny_budget_caps_sub_queries(self):
        # $0.004 ≈ one search + one fetch: templates want 4 queries, capped to 2
        report = research_deep(self.Q, _rctx(self._reg()), budget_usd=0.004)
        self.assertFalse(report.needs_clarification)
        self.assertLessEqual(len(report.sub_queries), 2)
        self.assertGreaterEqual(len(report.sub_queries), 1)
        self.assertTrue(report.budget_exhausted)
        self.assertLessEqual(report.spent_usd, 0.004)
        self.assertTrue(report.synthesis)  # degraded, not dead

    def test_exhausted_run_still_synthesizes_without_raising(self):
        report = research_deep(self.Q, _rctx(self._reg()), budget_usd=0.0001)
        self.assertEqual(report.findings, [])
        self.assertTrue(report.budget_exhausted)
        self.assertEqual(report.spent_usd, 0.0)
        # nothing to synthesize from: honest empty marker, no exception
        self.assertIn("SYNTHESIS_EMPTY", report.synthesis)

    def test_generous_budget_reports_spend(self):
        reg = self._reg()
        report = research_deep(self.Q, _rctx(reg), budget_usd=1.0,
                               max_queries=2)
        self.assertFalse(report.needs_clarification)
        self.assertFalse(report.budget_exhausted)
        n_search = len([c for c in reg.calls if c[0] == "web_search"])
        n_fetch = len([c for c in reg.calls if c[0] == "web_fetch"])
        expected = (n_search * COST_TABLE["web_search"]
                    + n_fetch * COST_TABLE["web_fetch"])
        self.assertAlmostEqual(report.spent_usd, round(expected, 6))
        self.assertGreater(report.spent_usd, 0.0)

    def test_llm_phases_charge_and_degrade(self):
        # llm clarify call exhausts the budget; the run degrades to
        # templates + extractive instead of dying.
        def llm(prompt):
            return "CLEAR"
        reg = self._reg()
        report = research_deep(self.Q, _rctx(reg), llm_fn=llm,
                               budget_usd=COST_TABLE["llm_call"])
        self.assertFalse(report.needs_clarification)
        self.assertTrue(report.budget_exhausted)
        # clarify charged the only affordable llm_call; searches never ran
        self.assertEqual(report.findings, [])
        self.assertIn("SYNTHESIS_EMPTY", report.synthesis)

    def test_no_budget_is_today_behavior(self):
        reg = self._reg()
        report = research_deep(self.Q, _rctx(reg), max_queries=2)
        self.assertEqual(report.spent_usd, 0.0)
        self.assertFalse(report.budget_exhausted)
        self.assertEqual(len(report.sub_queries), 2)


# ── tool registration ──────────────────────────────────────────────────────

class BudgetToolTests(unittest.TestCase):
    def _registry(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(context=SimpleNamespace(
            db=Database(":memory:"), router=None, memory=None, gateway=None))

        def fake_search(query: str = "", max_results: int = 5):
            return {"results": _results("https://a.example/1")}

        def fake_fetch(url: str = "", max_chars: int = 6000):
            return {"url": url, "text": "fetched article text"}

        registry.register("web_search", fake_search, capability="net.out")
        registry.register("web_fetch", fake_fetch, capability="net.out")
        from nomorals.research.pipeline import register
        register(registry)
        return registry

    def test_tool_accepts_budget_usd(self):
        registry = self._registry()
        outcome = registry.call(
            "research_deep", actor="system",
            question="current 2026 Nigeria AI startup news",
            max_queries=1, budget_usd=0.01,
        )
        self.assertTrue(outcome.ok, f"tool failed: {outcome.error}")
        result = outcome.value
        self.assertFalse(result["needs_clarification"])
        self.assertIn("spent_usd", result)
        self.assertIn("budget_exhausted", result)
        self.assertLessEqual(result["spent_usd"], 0.01)

    def test_tool_budget_exhaustion_reported_not_raised(self):
        registry = self._registry()
        outcome = registry.call(
            "research_deep", actor="system",
            question="current 2026 Nigeria AI startup news",
            max_queries=1, budget_usd=0.0001,
        )
        self.assertTrue(outcome.ok, f"tool failed: {outcome.error}")
        self.assertTrue(outcome.value["budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
