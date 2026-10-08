"""Build-map #6 — query decomposition + concurrent sub-tasks.

Offline: registries are faked with scripted ``call``/``call_many`` outcomes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.core.result import Err, Ok
from nomorals.research.pipeline import (
    DeepReport,
    ResearchContext,
    ResearchFinding,
    ResearchJob,
    clarify,
    decompose,
    research_deep,
    run_job,
    synthesize,
)
from nomorals.storage.db import Database


def _finding(title, url, snippet="snippet text here"):
    return ResearchFinding(job_id="j", title=title, url=url, snippet=snippet)


class _ScriptedRegistry:
    """Fake registry with order-preserving call_many."""

    def __init__(self, search=None, fetch_text="fetched full text",
                 fail_search=(), fail_fetch=()):
        self.search = search or {}
        self.fetch_text = fetch_text
        self.fail_search = set(fail_search)
        self.fail_fetch = set(fail_fetch)
        self.calls = []
        self.max_workers_seen = None

    def call(self, name, *, actor="system", **kwargs):
        self.calls.append((name, kwargs))
        if name == "web_search":
            q = kwargs.get("query", "")
            if q in self.fail_search:
                return Err(ValueError("boom"))
            return Ok({"results": self.search.get(q, [])})
        if name == "web_fetch":
            url = kwargs.get("url", "")
            if url in self.fail_fetch:
                return Err(ValueError("boom"))
            return Ok({"url": url, "text": self.fetch_text})
        return Err(ValueError(f"unknown tool {name}"))

    def call_many(self, calls, *, max_workers=1, **common):
        self.max_workers_seen = max_workers
        return [self.call(name, **{**common, **kwargs})
                for name, kwargs in calls]


class _SerialOnlyRegistry(_ScriptedRegistry):
    """No call_many — exercises the serial fallback."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # hide call_many entirely (del on instance won't remove the method,
        # so shadow the class attribute lookup via __getattribute__)
    call_many = None  # type: ignore[assignment]


def _rctx(registry):
    db = Database(":memory:")
    return ResearchContext(db=db, registry=registry)


def _results(*urls):
    return [{"title": f"Title for {u}", "url": u,
             "snippet": f"Snippet about {u} with enough words to be real."}
            for u in urls]


# ── decompose ────────────────────────────────────────────────────────────────

class DecomposeTests(unittest.TestCase):
    def test_llm_parses_lines_strips_blanks_and_numbering(self):
        def llm(prompt):
            self.assertIn("QUESTION", prompt)
            return ("1. qlora learning rate guide\n"
                    "\n"
                    "2) qlora rank selection\n"
                    "- qlora vram requirements\n"
                    "\n")
        out = decompose("qlora settings?", llm_fn=llm)
        self.assertEqual(out, ["qlora learning rate guide",
                              "qlora rank selection",
                              "qlora vram requirements"])

    def test_llm_capped_at_max_queries(self):
        def llm(prompt):
            return "\n".join(f"query number {i}" for i in range(10))
        out = decompose("x", llm_fn=llm, max_queries=4)
        self.assertEqual(len(out), 4)

    def test_fallback_templates_without_llm(self):
        out = decompose("fine-tuning small llms")
        self.assertEqual(out[0], "fine-tuning small llms")
        self.assertIn("fine-tuning small llms best practices how to", out)
        self.assertIn("fine-tuning small llms criticism problems limitations", out)
        self.assertIn("fine-tuning small llms recent developments", out)
        self.assertLessEqual(len(out), 6)

    def test_fallback_capped(self):
        out = decompose("fine-tuning small llms", max_queries=2)
        self.assertEqual(len(out), 2)

    def test_fallback_on_llm_failure(self):
        def boom(prompt):
            raise RuntimeError("llm down")
        out = decompose("fine-tuning small llms", llm_fn=boom)
        self.assertEqual(out[0], "fine-tuning small llms")

    def test_fallback_on_unparseable_llm(self):
        out = decompose("fine-tuning small llms",
                        llm_fn=lambda p: "\n\n  \n")
        self.assertEqual(out[0], "fine-tuning small llms")

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            decompose("   ")


# ── concurrent run_job ───────────────────────────────────────────────────────

class ConcurrentRunJobTests(unittest.TestCase):
    def test_preserves_query_order_and_dedups_urls(self):
        reg = _ScriptedRegistry(search={
            "q one": _results("https://a.example/1", "https://shared.example/x"),
            "q two": _results("https://shared.example/x", "https://b.example/2"),
        })
        job = ResearchJob(id="j1", topic="t", queries=["q one", "q two"],
                          max_results=5, fetch_top=0)
        findings = run_job(job, _rctx(reg))
        urls = [f.url for f in findings]
        self.assertEqual(urls, ["https://a.example/1",
                                "https://shared.example/x",
                                "https://b.example/2"])
        self.assertIsNotNone(reg.max_workers_seen)
        self.assertGreaterEqual(reg.max_workers_seen, 1)

    def test_raises_on_total_failure(self):
        reg = _ScriptedRegistry(fail_search={"q one", "q two"})
        job = ResearchJob(id="j1", topic="t", queries=["q one", "q two"])
        with self.assertRaises(RuntimeError):
            run_job(job, _rctx(reg))

    def test_partial_failure_keeps_good_findings(self):
        reg = _ScriptedRegistry(
            search={"q one": _results("https://a.example/1")},
            fail_search={"q two"})
        job = ResearchJob(id="j1", topic="t", queries=["q one", "q two"],
                          fetch_top=0)
        findings = run_job(job, _rctx(reg))
        self.assertEqual([f.url for f in findings], ["https://a.example/1"])

    def test_serial_fallback_without_call_many(self):
        reg = _SerialOnlyRegistry(search={
            "q one": _results("https://a.example/1"),
            "q two": _results("https://b.example/2"),
        })
        job = ResearchJob(id="j1", topic="t", queries=["q one", "q two"],
                          fetch_top=0)
        findings = run_job(job, _rctx(reg))
        self.assertEqual([f.url for f in findings],
                         ["https://a.example/1", "https://b.example/2"])
        # serial path still calls each query exactly once
        searches = [c for c in reg.calls if c[0] == "web_search"]
        self.assertEqual(len(searches), 2)

    def test_fetch_phase_runs_and_attaches_detail(self):
        reg = _ScriptedRegistry(
            search={"q": _results("https://a.example/1", "https://b.example/2")},
            fetch_text="FULL ARTICLE")
        job = ResearchJob(id="j1", topic="t", queries=["q"],
                          max_results=5, fetch_top=1)
        findings = run_job(job, _rctx(reg))
        self.assertEqual(findings[0].detail, "FULL ARTICLE")
        self.assertEqual(findings[1].detail, "")

    def test_progress_callback_phases(self):
        reg = _ScriptedRegistry(
            search={"q one": _results("https://a.example/1")})
        seen = []
        job = ResearchJob(id="j1", topic="t", queries=["q one"], fetch_top=1)
        run_job(job, _rctx(reg), progress=lambda ph, it: seen.append((ph, it)))
        phases = [p for p, _ in seen]
        self.assertIn("search", phases)
        self.assertIn("fetch", phases)
        self.assertEqual(seen[0], ("search", "q one"))
        self.assertEqual(seen[1], ("fetch", "https://a.example/1"))

    def test_progress_never_breaks_run(self):
        reg = _ScriptedRegistry(search={"q": _results("https://a.example/1")})
        job = ResearchJob(id="j1", topic="t", queries=["q"], fetch_top=0)

        def bad(ph, it):
            raise RuntimeError("progress exploded")

        findings = run_job(job, _rctx(reg), progress=bad)
        self.assertEqual(len(findings), 1)

    def test_empty_queries_raises(self):
        reg = _ScriptedRegistry()
        with self.assertRaises(ValueError):
            run_job(ResearchJob(id="j", topic="t", queries=[]), _rctx(reg))


# ── synthesize ───────────────────────────────────────────────────────────────

class SynthesizeTests(unittest.TestCase):
    def _findings(self):
        return [
            _finding("Alpha report", "https://a.example/1",
                     "Alpha findings about qlora ranks."),
            _finding("Beta analysis", "https://b.example/2",
                     "Beta findings about learning rates."),
        ]

    def test_llm_citations_remapped_and_invented_stripped(self):
        def llm(prompt):
            return ("Ranks matter [S2]. Also see [S9] and [S1].")
        out = synthesize("what matters?", self._findings(), llm_fn=llm)
        # [S2] -> [1] (first appearance), [S9] invented -> stripped,
        # [S1] -> [2]; Sources section lists both real sources.
        self.assertIn("[1]", out)
        self.assertIn("[2]", out)
        self.assertNotIn("[S9]", out)
        self.assertNotIn("S9", out.replace("Sources:", ""))
        self.assertIn("Sources:", out)
        self.assertIn("https://b.example/2", out)
        self.assertIn("https://a.example/1", out)

    def test_synthesis_empty_passthrough(self):
        out = synthesize("q?", self._findings(),
                         llm_fn=lambda p: "SYNTHESIS_EMPTY")
        self.assertEqual(out, "SYNTHESIS_EMPTY")

    def test_extractive_fallback(self):
        out = synthesize("what matters?", self._findings(), llm_fn=None)
        self.assertIn("[1] Alpha report", out)
        self.assertIn("[2] Beta analysis", out)
        self.assertIn("Sources:", out)
        self.assertIn("https://a.example/1", out)

    def test_extractive_on_llm_failure(self):
        def boom(prompt):
            raise RuntimeError("down")
        out = synthesize("what matters?", self._findings(), llm_fn=boom)
        self.assertIn("[1] Alpha report", out)

    def test_no_valid_citations_falls_back_to_extractive(self):
        # LLM answers but cites nothing real -> honest extractive brief.
        out = synthesize("what matters?", self._findings(),
                         llm_fn=lambda p: "All made up, no sources cited here.")
        self.assertIn("[1] Alpha report", out)

    def test_empty_findings(self):
        self.assertEqual(synthesize("q?", [], llm_fn=None), "SYNTHESIS_EMPTY")

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            synthesize("  ", self._findings())


# ── clarify ──────────────────────────────────────────────────────────────────

class ClarifyTests(unittest.TestCase):
    def test_vague_gets_generic_clarification(self):
        out = clarify("tell me about stuff")
        self.assertEqual(len(out), 1)
        self.assertIn("angle", out[0].lower())

    def test_short_question_gets_clarification(self):
        out = clarify("AI news")
        self.assertEqual(len(out), 1)

    def test_sharp_anchored_query_returns_empty(self):
        out = clarify("What are the current NITDA NCAIR 2026 application "
                      "requirements for Nigerian developers?")
        self.assertEqual(out, [])

    def test_llm_clear_returns_empty(self):
        out = clarify("anything", llm_fn=lambda p: "CLEAR")
        self.assertEqual(out, [])

    def test_llm_questions_parsed_max_three(self):
        def llm(prompt):
            return ("1. What time frame?\n2. Which country?\n"
                    "3. How deep?\n4. What format?")
        out = clarify("vague topic", llm_fn=llm)
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith("What time frame"))

    def test_llm_failure_returns_empty(self):
        def boom(prompt):
            raise RuntimeError("down")
        self.assertEqual(clarify("vague topic", llm_fn=boom), [])

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            clarify("  ")


# ── research_deep ────────────────────────────────────────────────────────────

class ResearchDeepTests(unittest.TestCase):
    def test_short_circuits_on_clarification(self):
        reg = _ScriptedRegistry()
        report = research_deep("tell me about stuff", _rctx(reg))
        self.assertIsInstance(report, DeepReport)
        self.assertTrue(report.needs_clarification)
        self.assertEqual(len(report.clarifications), 1)
        self.assertEqual(report.findings, [])
        self.assertEqual(report.synthesis, "")
        # the run was NOT burned: no tool calls at all
        self.assertEqual(reg.calls, [])

    def test_full_orchestration_no_llm(self):
        from nomorals.research.pipeline import _template_queries
        q = ("What are the current NITDA NCAIR 2026 application requirements "
             "for Nigerian developers?")
        templates = _template_queries(q, 2)
        reg = _ScriptedRegistry(search={
            templates[0]: _results("https://a.example/1"),
            templates[1]: _results("https://b.example/2"),
        })
        report = research_deep(q, _rctx(reg), max_queries=2)
        self.assertFalse(report.needs_clarification)
        self.assertEqual(len(report.sub_queries), 2)
        self.assertEqual(len(report.findings), 2)
        self.assertIn("[1]", report.synthesis)
        self.assertIn("Sources:", report.synthesis)

    def test_full_orchestration_with_llm(self):
        seen_prompts = []

        def llm(prompt):
            seen_prompts.append(prompt)
            if "scoping" in prompt:
                return "CLEAR"
            if "Break this research question" in prompt:
                return "qlora rank guide\nqlora vram needs"
            return "Use rank 16 [S1] with enough VRAM [S2]."

        # Snippets honestly contain the facts the fake LLM cites —
        # build-map #20 strips sentences whose citations don't verify
        # against the source texts.
        reg = _ScriptedRegistry(search={
            "qlora rank guide": [{
                "title": "QLoRA rank guide",
                "url": "https://a.example/1",
                "snippet": "Use rank 16 for QLoRA fine-tuning.",
            }],
            "qlora vram needs": [{
                "title": "QLoRA VRAM needs",
                "url": "https://b.example/2",
                "snippet": "You need enough VRAM for training runs.",
            }],
        })
        report = research_deep(
            "What are the current best qlora settings for 2026?",
            _rctx(reg), llm_fn=llm)
        self.assertFalse(report.needs_clarification)
        self.assertEqual(report.sub_queries,
                         ["qlora rank guide", "qlora vram needs"])
        self.assertIn("rank 16 [1]", report.synthesis)
        self.assertIn("VRAM [2]", report.synthesis)
        self.assertGreaterEqual(len(seen_prompts), 3)

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            research_deep("  ", _rctx(_ScriptedRegistry()))

    def test_total_search_failure_raises(self):
        from nomorals.research.pipeline import _template_queries
        # sharp enough to skip clarification: anchored + content words
        q = "current 2026 Nigeria remote AI training gig platforms compared"

        def llm(prompt):
            if "scoping" in prompt:
                return "CLEAR"
            # decompose prompt: return the template queries verbatim
            return "\n".join(_template_queries(q, 6))

        reg = _ScriptedRegistry(fail_search=set(_template_queries(q, 6)))
        with self.assertRaises(RuntimeError):
            research_deep(q, _rctx(reg), llm_fn=llm)


# ── tool registration ────────────────────────────────────────────────────────

class RegisterTests(unittest.TestCase):
    def test_research_deep_tool_registered(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(context=SimpleNamespace(
            db=Database(":memory:"), router=None, memory=None, gateway=None))
        from nomorals.research.pipeline import register
        register(registry)
        spec = registry.get("research_deep")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.capability, "net.out")

    def test_wired_via_agent_tool_bridge(self):
        # the AGENT_TOOL_MODULES mechanism (nomorals.* fallback) must pick
        # up research.pipeline next to research_swarm
        from nomorals.tools import agents as bridge

        self.assertIn("research.pipeline", bridge.AGENT_TOOL_MODULES)
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(context=SimpleNamespace(
            db=Database(":memory:"), router=None, memory=None, gateway=None))
        bridge.register(registry)
        self.assertIsNotNone(registry.get("research_deep"))

    def test_tool_call_end_to_end_through_registry(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(context=SimpleNamespace(
            db=Database(":memory:"), router=None, memory=None, gateway=None))

        def fake_search(query: str = "", max_results: int = 5):
            return {"results": _results("https://a.example/1")}

        def fake_fetch(url: str = "", max_chars: int = 6000):
            return {"url": url, "text": "fetched article text"}

        registry.register("web_search", fake_search,
                          capability="net.out")
        registry.register("web_fetch", fake_fetch,
                          capability="net.out")
        from nomorals.research.pipeline import register
        register(registry)

        outcome = registry.call(
            "research_deep", actor="system",
            question="current 2026 Nigeria AI news",
            max_queries=1,
        )
        self.assertTrue(outcome.ok, f"tool failed: {outcome.error}")
        result = outcome.value
        self.assertFalse(result["needs_clarification"])
        self.assertIn("[1]", result["synthesis"])
        self.assertEqual(len(result["findings"]), 1)

    def test_tool_rejects_empty_question(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(context=SimpleNamespace(
            db=Database(":memory:"), router=None, memory=None, gateway=None))
        from nomorals.research.pipeline import register
        register(registry)
        outcome = registry.call("research_deep", actor="system", question="  ")
        self.assertFalse(outcome.ok)


if __name__ == "__main__":
    unittest.main()
