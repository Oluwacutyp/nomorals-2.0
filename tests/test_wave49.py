"""Wave 49 — the agent benchmark + universal reasoning pre-flight.

- agents/benchmark.py: four hermetic dimensions (reasoning, planning,
  tool use, self-correction) that measure the SYSTEM, scored 0-1
- the evolution promotion gate now runs it as a regression check:
  records a baseline on first measurable run, blocks promotions that
  regress it, raises it on improvement, and skips when unmeasurable
  (mock/offline) — an unmeasurable benchmark never blocks
- the universal reasoning hook (NM_REASONING_MODE off|auto|always,
  always-on in power mode) is wired into the orchestrator's plans,
  Devon's tool sequences, and the coding bot's drafts

Fully hermetic: a scripted router answers every model call by matching
a marker in the prompt; the self-correction dimension runs real broken
code in the local sandbox to capture its actual traceback, then runs
the fix. No network anywhere.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tests.test_partner_runtime import _make_context

from nomorals.agents.benchmark import (
    _provider_name,
    measurable,
    run_benchmark,
)
from nomorals.agents.reasoning import (
    reasoning_enabled,
    revise_text,
    review_text,
)
from nomorals.llm.base import LLMResponse


# ── scripted model ───────────────────────────────────────────────────────────


class FakeRouter:
    """Returns the first marker that appears in the last (user) prompt."""

    def __init__(self, script: dict[str, str] | None = None,
                 default: str = "") -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls = 0
        self.prompts: list[str] = []

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content if messages else ""
        self.calls += 1
        self.prompts.append(prompt)
        for marker, reply in self.script.items():
            if marker in prompt:
                return LLMResponse(text=reply, model="fake")
        return LLMResponse(text=self.default, model="fake")


class Ctx:
    """Minimal context with the attributes the agents read."""

    def __init__(self, router, settings=None) -> None:
        self.router = router
        self.settings = settings or _FakeSettings()
        self.executor = None
        self.blackboard = None
        self.db = None
        self.tools = None
        self.extras = {}


class _FakeSettings:
    def __init__(self, provider="testbed", offline=False,
                 reasoning_mode="auto", evolution_benchmark="on") -> None:
        class _LLM:
            pass
        class _Evo:
            pass
        self.llm = _LLM()
        self.llm.provider = provider
        self.offline = offline
        self.reasoning_mode = reasoning_mode
        self.evolution = _Evo()
        self.evolution.benchmark = evolution_benchmark


def _measurable_ctx(script=None, **settings):
    settings.setdefault("provider", "testbed")
    router = FakeRouter(script)
    return Ctx(router, _FakeSettings(**settings)), router


# ── measurability ────────────────────────────────────────────────────────────


class MeasurabilityTest(unittest.TestCase):
    def test_mock_provider_unmeasurable(self):
        ctx, _ = _measurable_ctx(provider="mock")
        self.assertFalse(measurable(ctx))

    def test_missing_router_unmeasurable(self):
        ctx, _ = _measurable_ctx()
        ctx.router = None
        self.assertFalse(measurable(ctx))

    def test_real_provider_measurable(self):
        ctx, _ = _measurable_ctx(provider="testbed")
        self.assertTrue(measurable(ctx))

    def test_offline_unmeasurable(self):
        ctx, _ = _measurable_ctx(provider="testbed", offline=True)
        self.assertFalse(measurable(ctx))

    def test_provider_name(self):
        ctx, _ = _measurable_ctx(provider="testbed")
        self.assertEqual(_provider_name(ctx), "testbed")


# ── the full benchmark report ────────────────────────────────────────────────


class BenchmarkReportTest(unittest.TestCase):
    def test_unmeasurable_scores_none(self):
        ctx, _ = _measurable_ctx(provider="mock")
        report = run_benchmark(ctx)
        self.assertFalse(report.measurable)
        self.assertIsNone(report.overall)
        for dim in report.scores.values():
            self.assertIsNone(dim.score)
        d = report.as_dict()
        self.assertIn("dimensions", d)
        self.assertIsNone(d["overall"])

    def test_measurable_runs_all_dimensions(self):
        # a router that answers "well enough" — exact correctness is
        # covered per-dimension below; here we check the report shape
        ctx, _ = _measurable_ctx({
            "Decompose the goal": json.dumps({"steps": []}),
            "Available tools": json.dumps({"steps": []}),
            "QUESTION:": "ANSWER: x\nCONFIDENCE: 0.5",
        })
        report = run_benchmark(ctx, limit=1)
        self.assertTrue(report.measurable)
        self.assertIsNotNone(report.overall)
        for name in ("reasoning", "planning", "tool_use", "self_correction"):
            self.assertIn(name, report.scores)
        # self-correction always runs real code, so it's scored regardless
        self.assertIsNotNone(report.scores["self_correction"].score)


# ── dimension: planning ──────────────────────────────────────────────────────


def _mkplan(steps):
    """Build a Plan from (name, goal, role, depends_on) tuples."""
    from nomorals.agents.orchestrator import Plan, PlanStep
    from nomorals.agents.tasks import TaskKind
    return Plan(
        goal="g",
        steps=[PlanStep(name=n, goal=g, role=r, kind=TaskKind.IO,
                        depends_on=d)
               for n, g, r, d in steps])


class PlanningDimensionTest(unittest.TestCase):
    def test_valid_plans_pass(self):
        from nomorals.agents.benchmark import _judge_plan

        # backup then verify (verify depends on backup)
        ok, _ = _judge_plan(_mkplan([
            ("backup", "copy the data/ directory to remote", "execution", []),
            ("verify", "restore the backup and check it", "critic",
             ["backup"]),
        ]), min_steps=2, must_mention=[r"backup|remote|data/"],
            order=(r"verif|restor", r"backup|copy|transfer"))
        self.assertTrue(ok, _)
        # investigate a slow API across three independent causes
        ok, _ = _judge_plan(_mkplan([
            ("db", "profile the database queries", "research", []),
            ("net", "trace the network path", "research", []),
            ("load", "measure the server load", "research", []),
            ("fix", "fix the bottleneck", "execution", ["db", "net", "load"]),
        ]), min_steps=3,
            coverage=[(r"database|quer",), (r"networ",),
                      (r"load|server|cpu")])
        self.assertTrue(ok, _)
        # ship a CLI command: implement -> test -> document, ordered
        ok, _ = _judge_plan(_mkplan([
            ("impl", "write the command code", "coding", []),
            ("tests", "add unit tests for the command", "execution",
             ["impl"]),
            ("docs", "update the help text", "execution", ["tests"]),
        ]), min_steps=3, must_mention=[r"test", r"doc|help|readme"],
            order=(r"test", r"implement|writ|code"))
        self.assertTrue(ok, _)

    def test_plan_missing_coverage_fails(self):
        from nomorals.agents.benchmark import _judge_plan

        # generic steps that cover none of the required sub-problems
        ok, detail = _judge_plan(_mkplan([
            ("a", "look around the code", "research", []),
            ("b", "read some files", "research", []),
            ("c", "write a note", "execution", ["a", "b"]),
        ]), min_steps=3,
            coverage=[(r"database|quer",), (r"networ",),
                      (r"load|server|cpu")])
        self.assertFalse(ok)
        self.assertIn("covers none", detail)

    def test_bad_ordering_fails(self):
        from nomorals.agents.benchmark import _judge_plan

        # verify step that does NOT depend on the backup step
        ok, detail = _judge_plan(_mkplan([
            ("verify", "restore the snapshot and check it", "critic", []),
            ("backup", "copy the data/ directory to remote", "execution",
             []),
        ]), min_steps=2, must_mention=[r"backup|remote|data/"],
            order=(r"verif|restor", r"copy|transfer"))
        self.assertFalse(ok)
        self.assertIn("does not depend", detail)

    def test_too_few_steps_fails(self):
        from nomorals.agents.benchmark import _judge_plan

        ok, detail = _judge_plan(_mkplan([
            ("only", "do the thing", "execution", []),
        ]), min_steps=3)
        self.assertFalse(ok)
        self.assertIn("only 1 steps", detail)

    def test_integration_router_plan(self):
        # end-to-end: the router returns a good plan, the dimension passes
        script = {
            "backs up the data/ directory": json.dumps({"rationale": "r",
                "steps": [
                    {"name": "backup", "goal": "copy the data/ directory "
                     "to remote", "role": "execution"},
                    {"name": "verify", "goal": "restore the backup and "
                     "check it", "role": "critic",
                     "depends_on": ["backup"]},
                ]}),
        }
        ctx, _ = _measurable_ctx(script, reasoning_mode="off")
        report = run_benchmark(ctx, dimensions=["planning"], limit=1)
        self.assertEqual(report.scores["planning"].score, 1.0)


# ── dimension: tool use ──────────────────────────────────────────────────────


class ToolUseDimensionTest(unittest.TestCase):
    def test_valid_sequences_pass(self):
        script = {
            "login endpoint": json.dumps({"steps": [
                {"tool": "grep", "args": {"pattern": "def login",
                                          "path": "src"}, "why": "find it"},
                {"tool": "read_file", "args": {"path": "src/auth.py"},
                 "why": "read it"},
                {"tool": "run_tests", "args": {"pattern": "test_auth"},
                 "why": "run tests"},
            ]}),
            "api.example.com/health": json.dumps({"steps": [
                {"tool": "http_get",
                 "args": {"url": "https://api.example.com/health"}},
                {"tool": "db_query",
                 "args": {"sql": "SELECT * FROM errors LIMIT 1"}},
                {"tool": "notify", "args": {"message": "summarizing"}},
            ]}),
            "Kubernetes": json.dumps({"steps": [
                {"tool": "notify",
                 "args": {"message": "no deploy tool available here"}},
            ]}),
        }
        ctx, _ = _measurable_ctx(script)
        report = run_benchmark(ctx, dimensions=["tool_use"], limit=3)
        self.assertEqual(report.scores["tool_use"].score, 1.0)

    def test_hallucinated_tool_fails(self):
        # a model that invents a tool that doesn't exist in the registry
        script = {
            "Kubernetes": json.dumps({"steps": [
                {"tool": "kubectl_apply", "args": {"manifest": "a.yaml"}},
            ]}),
        }
        ctx, _ = _measurable_ctx(script)
        report = run_benchmark(ctx, dimensions=["tool_use"], limit=1)
        self.assertEqual(report.scores["tool_use"].score, 0.0)

    def test_wrong_order_fails(self):
        # notify before it has gathered the two results
        script = {
            "api.example.com/health": json.dumps({"steps": [
                {"tool": "notify", "args": {"message": "done"}},
                {"tool": "http_get",
                 "args": {"url": "https://api.example.com/health"}},
                {"tool": "db_query", "args": {"sql": "SELECT 1"}},
            ]}),
        }
        ctx, _ = _measurable_ctx(script)
        report = run_benchmark(ctx, dimensions=["tool_use"], limit=1)
        self.assertEqual(report.scores["tool_use"].score, 0.0)


# ── dimension: self-correction (real sandbox runs) ──────────────────────────


class SelfCorrectionDimensionTest(unittest.TestCase):
    def test_fixes_that_run_pass(self):
        script = {
            "last person in the list": (
                "```python\n"
                "def last_name(people):\n"
                "    return people[len(people) - 1][\"name\"]\n\n"
                "print(last_name([{\"name\": \"ada\"}]))\n"
                "```"),
            "empty list averages to 0": (
                "```python\n"
                "def average(values):\n"
                "    if not values:\n"
                "        return 0\n"
                "    return sum(values) / len(values)\n\n"
                "print(average([]))\n"
                "```"),
            "top k scores": (
                "```python\n"
                "def top_scores(scores, k=3):\n"
                "    return sorted(scores, reverse=True)[:k]\n\n"
                "print(top_scores([90, 75, 88], 3))\n"
                "```"),
        }
        ctx, _ = _measurable_ctx(script)
        report = run_benchmark(ctx, dimensions=["self_correction"], limit=3)
        dim = report.scores["self_correction"]
        self.assertEqual(dim.score, 1.0)
        self.assertEqual(dim.passed, 3)

    def test_fix_that_still_breaks_fails(self):
        # returns the original (still-broken) code — the NameError remains
        script = {
            "top k scores": (
                "```python\n"
                "def top_scores(scores, k=3):\n"
                "    return sorted(scores, reverse=True)[:k]\n\n"
                "print(top_scores([90, 75, 88], top_k))\n"
                "```"),
        }
        ctx, _ = _measurable_ctx(script)
        report = run_benchmark(ctx, dimensions=["self_correction"], limit=1)
        self.assertEqual(report.scores["self_correction"].score, 0.0)


# ── the evolution promotion gate ─────────────────────────────────────────────


def _temp_repo(tmp: str) -> None:
    os.makedirs(os.path.join(tmp, "tests"), exist_ok=True)
    open(os.path.join(tmp, "tests", "__init__.py"), "w").close()
    with open(os.path.join(tmp, "mod.py"), "w") as fh:
        fh.write("def answer():\n    return 41\n")
    with open(os.path.join(tmp, "tests", "test_mod.py"), "w") as fh:
        fh.write(
            "import unittest\nfrom mod import answer\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_answer(self):\n"
            "        self.assertEqual(answer(), 41)\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "baseline"]):
        subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)


class _ScriptedEvolver:
    """Answers the evolver's plan prompt with one docstring edit."""

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text=json.dumps({
            "rationale": "docstring",
            "edits": [{"path": "mod.py",
                       "old": "def answer():\n    return 41",
                       "new": 'def answer():\n    """The answer."""\n    '
                              'return 41'}]}), model="fake")


class _Report:
    def __init__(self, overall, measurable=True):
        self.overall = overall
        self.measurable = measurable
        self.provider = "testbed"
        self.seconds = 0.0
        self.scores = {}

    def as_dict(self):
        return {"overall": self.overall, "measurable": self.measurable}


class EvolutionBenchmarkGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="evo-w49-")
        _temp_repo(self.tmp)
        from nomorals.agents.evolution import EvolutionAgent
        self.agent = EvolutionAgent(self.ctx, repo_root=self.tmp)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plan(self) -> str:
        self.ctx.router = _ScriptedEvolver()
        return self.agent.plan("add a docstring to answer()").id

    def test_baseline_recorded_on_first_run(self):
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark",
                        return_value=_Report(0.8)) as m:
            out = self.agent.apply(self._plan_id())
        self.assertTrue(out["applied"], out)
        self.assertTrue(m.called)
        baseline = self.agent._load_benchmark_baseline()
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 0.8)

    def test_regression_blocks_apply(self):
        self.agent._save_benchmark_baseline(0.9)
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark",
                        return_value=_Report(0.5)):
            out = self.agent.apply(self._plan_id())
        self.assertFalse(out["applied"])
        self.assertEqual(out["status"], "reverted")
        self.assertIn("regression", out["reason"])
        # the working tree was restored exactly
        with open(os.path.join(self.tmp, "mod.py")) as fh:
            self.assertNotIn('"""The answer."""', fh.read())

    def test_improvement_raises_baseline(self):
        self.agent._save_benchmark_baseline(0.5)
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark",
                        return_value=_Report(0.9)):
            out = self.agent.apply(self._plan_id())
        self.assertTrue(out["applied"], out)
        self.assertAlmostEqual(self.agent._load_benchmark_baseline(), 0.9)

    def test_tolerance_allows_small_dip(self):
        self.agent._save_benchmark_baseline(0.9)
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark",
                        return_value=_Report(0.87)):  # within 0.05
            out = self.agent.apply(self._plan_id())
        self.assertTrue(out["applied"], out)

    def test_unmeasurable_skips_gate(self):
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark",
                        return_value=_Report(None, measurable=False)):
            out = self.agent.apply(self._plan_id())
        self.assertTrue(out["applied"], out)
        self.assertIsNone(self.agent._load_benchmark_baseline())

    def test_disabled_gate_skips(self):
        self.ctx.settings.evolution.benchmark = "off"
        self.ctx.router = _ScriptedEvolver()
        with mock.patch("nomorals.agents.benchmark.run_benchmark") as m:
            out = self.agent.apply(self._plan_id())
        self.assertTrue(out["applied"], out)
        self.assertFalse(m.called)

    def _plan_id(self) -> str:
        self.ctx.router = _ScriptedEvolver()
        return self.agent.plan("add a docstring to answer()").id


# ── the universal reasoning hook ─────────────────────────────────────────────


class ReasoningHookTest(unittest.TestCase):
    def test_disabled_by_off(self):
        ctx, _ = _measurable_ctx(reasoning_mode="off")
        self.assertFalse(reasoning_enabled(ctx, complex_ok=True))

    def test_auto_respects_complexity(self):
        ctx, _ = _measurable_ctx(reasoning_mode="auto")
        self.assertFalse(reasoning_enabled(ctx, complex_ok=False))
        self.assertTrue(reasoning_enabled(ctx, complex_ok=True))

    def test_always_on(self):
        ctx, _ = _measurable_ctx(reasoning_mode="always")
        self.assertTrue(reasoning_enabled(ctx, complex_ok=False))

    def test_power_mode_forces_on(self):
        ctx, _ = _measurable_ctx(reasoning_mode="auto")
        # a power-mode instance set in context.extras
        from nomorals.agents.power import PowerMode

        class _CtxWithExtras:
            pass
        pctx = _CtxWithExtras()
        pctx.settings = ctx.settings
        power = PowerMode(pctx)
        object.__setattr__(power, "_active", True)
        pctx.extras = {"power": power}
        # reasoning_enabled reads context.settings and context.extras
        self.assertTrue(reasoning_enabled(pctx, complex_ok=False))

    def test_review_returns_flaws(self):
        ctx, _ = _measurable_ctx({
            "harsh reviewer": json.dumps({
                "passed": False, "flaws": ["missing the error case",
                                           "assumes disk is free"]}),
        })
        flaws = review_text(ctx, "a plan that skips error handling")
        self.assertEqual(len(flaws), 2)
        self.assertIn("missing the error case", flaws)

    def test_review_passes_clean(self):
        ctx, _ = _measurable_ctx({
            "harsh reviewer": json.dumps({"passed": True, "flaws": []}),
        })
        self.assertEqual(review_text(ctx, "a solid plan"), [])

    def test_review_never_raises_on_garbage(self):
        ctx, _ = _measurable_ctx({
            "harsh reviewer": "not json at all",
        })
        self.assertEqual(review_text(ctx, "some text"), [])

    def test_review_never_raises_on_error(self):
        class BoomRouter:
            def chat(self, *a, **k):
                raise RuntimeError("provider down")
        ctx = Ctx(BoomRouter(), _FakeSettings())
        self.assertEqual(review_text(ctx, "text"), [])

    def test_revise_returns_revised(self):
        ctx, _ = _measurable_ctx({
            "Revise the artifact": "the revised plan, fixed",
        })
        out = revise_text(ctx, "original", ["flaw one"],
                          focus="same shape")
        self.assertEqual(out, "the revised plan, fixed")

    def test_revise_falls_back_on_short(self):
        ctx, _ = _measurable_ctx({"Revise the artifact": "no"})
        out = revise_text(ctx, "original text that is long enough",
                          ["flaw"])
        self.assertEqual(out, "original text that is long enough")


# ── hook integration: orchestrator / devon / coding ──────────────────────────


class OrchestratorReviewIntegrationTest(unittest.TestCase):
    def test_flawed_plan_gets_revised(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        good_plan = json.dumps({"rationale": "r", "steps": [
            {"name": "research", "goal": "gather evidence",
             "role": "research"},
            {"name": "execute", "goal": "do the thing", "role": "execution",
             "depends_on": ["research"]},
            {"name": "verify", "goal": "verify the result against the goal",
             "role": "critic", "depends_on": ["execute"]},
        ]})
        ctx, _ = _measurable_ctx({
            "Decompose the goal": good_plan,
            "harsh reviewer": json.dumps({
                "passed": False, "flaws": ["verify step does not run tests"]}),
            "Revise the artifact": good_plan,
        }, reasoning_mode="always")
        orch = MasterOrchestrator(ctx)
        plan = orch.plan("investigate and fix the broken pipeline and verify")
        # the plan survived with steps (the revision re-parsed cleanly)
        self.assertTrue(plan.steps)
        self.assertTrue(any("verify" in s.goal for s in plan.steps))

    def test_clean_plan_is_untouched(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        plan_json = json.dumps({"rationale": "r", "steps": [
            {"name": "research", "goal": "gather evidence",
             "role": "research"},
        ]})
        ctx, _ = _measurable_ctx({
            "Decompose the goal": plan_json,
            "harsh reviewer": json.dumps({"passed": True, "flaws": []}),
        }, reasoning_mode="always")
        orch = MasterOrchestrator(ctx)
        plan = orch.plan("gather evidence about the incident")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].goal, "gather evidence")

    def test_review_skipped_when_off(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        plan_json = json.dumps({"rationale": "r", "steps": [
            {"name": "research", "goal": "gather evidence",
             "role": "research"},
        ]})
        ctx, router = _measurable_ctx({
            "Decompose the goal": plan_json,
            "harsh reviewer": json.dumps({"passed": True, "flaws": []}),
        }, reasoning_mode="off")
        orch = MasterOrchestrator(ctx)
        orch.plan("gather evidence about the incident")
        # no review call was made when the mode is off
        self.assertFalse(any("harsh reviewer" in p for p in router.prompts))


class CodingReviewIntegrationTest(unittest.TestCase):
    def _agent(self, ctx):
        from nomorals.agents.coding import CodingAgent
        return CodingAgent(ctx)

    def test_buggy_draft_gets_revised(self):
        # the review finds the undefined name and the revision fixes it
        ctx, _ = _measurable_ctx({
            "harsh reviewer": json.dumps({
                "passed": False, "flaws": ["'top_k' is not defined"]}),
            "Revise the artifact": (
                "```python\n"
                "def top_scores(scores, k=3):\n"
                "    return sorted(scores, reverse=True)[:k]\n"
                "```\n"),
        }, reasoning_mode="always")
        # the Ctx needs a db for the journal; use a real context
        real, tmp = _make_context()
        real.router = ctx.router
        real.settings.reasoning_mode = "always"
        agent = self._agent(real)
        try:
            out = agent._reason_review_code(
                "print the top k scores",
                "def top_scores(scores, k=3):\n"
                "    return sorted(scores, reverse=True)[:k]\n"
                "print(top_scores([90, 75, 88], top_k))\n")
            self.assertNotIn("top_k", out)
        finally:
            real.close()
            tmp.cleanup()

    def test_clean_draft_untouched(self):
        real, tmp = _make_context()
        real.router = FakeRouter({
            "harsh reviewer": json.dumps({"passed": True, "flaws": []}),
        })
        real.settings.reasoning_mode = "always"
        agent = self._agent(real)
        try:
            code = "print(1 + 1)\n"
            self.assertEqual(agent._reason_review_code("add two numbers",
                                                       code), code)
        finally:
            real.close()
            tmp.cleanup()


# ── wiring ───────────────────────────────────────────────────────────────────


class WiringTest(unittest.TestCase):
    def test_registry_registers_benchmark(self):
        from nomorals.tools.registry import ToolRegistry
        self.assertIn("benchmark",
                      set(ToolRegistry().register_builtins()._tools))

    def test_devon_catalog_and_handler(self):
        from nomorals.agents.devon import TOOL_CATALOG, DevonAgent
        self.assertIn("benchmark", {n for n, _ in TOOL_CATALOG})
        self.assertTrue(hasattr(DevonAgent, "_tool_benchmark"))

    def test_control_benchmark_registered(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS, parse_control
        self.assertIn("benchmark", CONTROL_COMMANDS)
        c = parse_control("/benchmark")
        self.assertEqual(c.kind, "benchmark")
        c2 = parse_control("/benchmark planning")
        self.assertIn("planning", c2.tail)

    def test_env_maps(self):
        from nomorals.core.config import env_var_path
        self.assertEqual(env_var_path("NM_REASONING_MODE"), "reasoning_mode")
        self.assertEqual(env_var_path("NM_EVOLUTION_BENCHMARK"),
                         "evolution.benchmark")

    def test_settings_defaults_and_validation(self):
        from nomorals.core.config import Settings, _validate
        # wave 85: the reasoning pre-flight is permanently on by default
        # (each hook stays budget-capped); "auto"/"off" still validate.
        self.assertEqual(Settings().reasoning_mode, "always")
        self.assertEqual(Settings().evolution.benchmark, "on")
        s = Settings()
        _validate(s)  # defaults are valid
        bad = Settings()
        bad.reasoning_mode = "bogus"
        with self.assertRaises(Exception):
            _validate(bad)


if __name__ == "__main__":
    unittest.main()
