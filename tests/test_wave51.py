"""Wave 51 — the autonomous cascade.

Hermetic end-to-end tests for the wave-51 stack:

  1. goal → project cascade   — one command spawns an autonomous project
                                that executes the goal; progress and status
                                sync both ways (migration 18)
  2. KG-aware reasoning       — the knowledge graph is injected into the
                                universal reasoning pre-flight (every system
                                that reasons now reasons over stored knowledge)
  3. the cognitive loop       — one heartbeat that drives goals → improvement
                                → personal-model fine-tune, fault-isolated
  4. power-mode autonomy dial — one unlock cascades the whole autonomous
                                stack (cognitive loop + improvement auto-tick
                                + multi-model router when a local model exists)
  5. scheduler glue           — the "cognitive loop" job is registered
                                idempotently on boot

No network egress. LLM paths run through a scripted :class:`FakeRouter`;
anything that would otherwise call a model is deterministic.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nomorals.core.config import load_settings          # noqa: E402
from nomorals.core.errors import ConfigError            # noqa: E402
from nomorals.agents.context import build_context       # noqa: E402
from nomorals.llm.base import LLMResponse               # noqa: E402
from nomorals.tools.registry import ToolRegistry        # noqa: E402


class FakeRouter:
    """Scripted router: first marker found in the last user prompt wins."""

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


def _make_context() -> tuple[Any, "tempfile.TemporaryDirectory"]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-wave51-")
    settings = load_settings(
        overrides={"home": tmp.name, "partner.platforms": "local",
                   "chat.local_enabled": "true"}
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    return context, tmp


def _registry(context) -> ToolRegistry:
    reg = ToolRegistry()
    reg.context = context
    reg.register_builtins()
    return reg


def _ok_executor(description: str) -> str:
    return "ok: " + description


# ── 1. migration 18 + goal → project cascade ─────────────────────────────────

class Migration18Test(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_goals_project_id_column(self) -> None:
        cols = [r["name"] for r in self.context.db.query(
            "PRAGMA table_info(goals)")]
        self.assertIn("project_id", cols)
        idx = [r["name"] for r in self.context.db.query(
            "SELECT name FROM sqlite_master WHERE type='index'")]
        self.assertIn("idx_goals_project", idx)

    def test_latest_version(self) -> None:
        from nomorals.storage.migrations import latest_version

        self.assertGreaterEqual(latest_version(), 18)


class GoalProjectCascadeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_spawn_links_goal_and_project(self) -> None:
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.projects import ProjectManager

        gs = GoalSystem(self.context)
        g = gs.create("Ship the report",
                      description="write it. verify it. deliver it.")
        sp = gs.spawn_project(g.id)
        self.assertFalse(sp["existing"])
        self.assertIn("project_id", sp)
        # the goal is linked
        self.assertEqual(gs.get(g.id).project_id, sp["project_id"])
        # the project knows its goal
        pm = ProjectManager(self.context)
        st = pm.status(sp["project_id"])
        self.assertTrue(st["ok"])
        self.assertEqual(st.get("goal_id") or st.get("project", {}).get("goal_id", ""),
                         g.id)

    def test_spawn_is_idempotent(self) -> None:
        from nomorals.agents.goals import GoalSystem

        gs = GoalSystem(self.context)
        g = gs.create("Idempotent", description="a. b.")
        first = gs.spawn_project(g.id)
        second = gs.spawn_project(g.id)
        self.assertEqual(first["project_id"], second["project_id"])
        self.assertTrue(second["existing"])

    def test_advance_drives_linked_project(self) -> None:
        from nomorals.agents.goals import GoalSystem

        gs = GoalSystem(self.context)
        g = gs.create("Drive", description="one. two. three.")
        gs.spawn_project(g.id)
        before = gs.get(g.id).progress
        after = gs.advance(g.id, executor=_ok_executor)
        self.assertGreater(after.progress, before)
        self.assertEqual(after.status, "active")

    def test_project_done_completes_goal(self) -> None:
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.projects import ProjectManager

        gs = GoalSystem(self.context)
        g = gs.create("Finish", description="a. b. c.")
        sp = gs.spawn_project(g.id)
        pm = ProjectManager(self.context)
        res = pm.run(sp["project_id"], executor=_ok_executor, max_steps=10)
        self.assertEqual(res["status"], "done")
        goal = gs.get(g.id)
        self.assertEqual(goal.status, "done")
        self.assertEqual(goal.progress, 1.0)
        self.assertTrue(all(s.status == "done" for s in goal.steps))

    def test_project_failed_pauses_goal(self) -> None:
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.projects import ProjectManager

        gs = GoalSystem(self.context)
        g = gs.create("Doomed", description="impossible. step.")
        sp = gs.spawn_project(g.id)

        def always_fail(d):
            raise RuntimeError("nope")

        pm = ProjectManager(self.context)
        res = pm.run(sp["project_id"], executor=always_fail,
                     max_attempts=2, max_steps=5)
        self.assertEqual(res["status"], "failed")
        goal = gs.get(g.id)
        self.assertEqual(goal.status, "paused")

    def test_goal_tool_spawn_project_action(self) -> None:
        reg = _registry(self.context)
        out = reg.call("goal", action="create", title="Tool cascade",
                       description="a. b. c.")
        self.assertTrue(out.ok)
        value = out.value
        gid = value.get("id") or (value.get("goal") or {}).get("id")
        out2 = reg.call("goal", action="spawn_project", goal_id=gid)
        self.assertTrue(out2.ok)
        self.assertIn("project_id", out2.value)
        out3 = reg.call("goal", action="spawn_project", goal_id=gid)
        self.assertTrue(out3.ok)
        self.assertTrue(out3.value.get("existing"))


# ── 2. KG-aware reasoning ────────────────────────────────────────────────────

class KnowledgeContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.kg import KnowledgeGraph

        self.kg = KnowledgeGraph(self.context.db)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def _seed(self) -> None:
        self.kg.upsert_node("Django", type="concept",
                            properties={"language": "python"})
        self.kg.upsert_node("Maria", type="person")
        self.kg.link("Maria", "Django", "uses")

    def test_block_lists_known_knowledge(self) -> None:
        from nomorals.agents.kg import knowledge_context

        self._seed()
        block = knowledge_context(self.context,
                                  "how does Maria use Django in the web app")
        self.assertIn("Django", block)
        self.assertIn("Maria", block)

    def test_empty_graph_is_empty(self) -> None:
        from nomorals.agents.kg import knowledge_context

        self.assertEqual(
            knowledge_context(self.context,
                              "a perfectly ordinary question about nothing"),
            "")

    def test_short_query_is_skipped(self) -> None:
        from nomorals.agents.kg import knowledge_context

        self._seed()
        self.assertEqual(knowledge_context(self.context, "hi"), "")

    def test_gate_off_disables_injection(self) -> None:
        from dataclasses import replace
        from nomorals.agents.kg import knowledge_context

        self._seed()
        self.context.settings = replace(self.context.settings,
                                        reasoning_knowledge="off")
        self.assertEqual(
            knowledge_context(self.context,
                              "how does Maria use Django in the web app"),
            "")
        self.context.settings = replace(self.context.settings,
                                        reasoning_knowledge="on")


class ReasoningKgInjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.kg import KnowledgeGraph

        kg = KnowledgeGraph(self.context.db)
        kg.upsert_node("Django", type="concept")
        kg.upsert_node("migrate command", type="concept")
        kg.link("Django", "migrate command", "uses")

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_reasoning_prompt_contains_knowledge(self) -> None:
        from nomorals.agents.reasoning import ReasoningEngine

        router = FakeRouter(default="answer: run the migrate command")
        self.context.router = router
        engine = ReasoningEngine(self.context, max_llm_calls=6,
                                 max_seconds=60.0)
        engine.reason("How do I apply Django database schema changes?")
        self.assertTrue(router.calls)
        joined = "\n".join(router.prompts)
        self.assertIn("knowledge graph", joined)
        self.assertIn("Django", joined)

    def test_reasoning_trace_notes_the_injection(self) -> None:
        from nomorals.agents.reasoning import ReasoningEngine

        self.context.router = FakeRouter(default="answer: yes")
        engine = ReasoningEngine(self.context, max_llm_calls=6,
                                 max_seconds=60.0)
        result = engine.reason("How do I apply Django database schema changes?")
        self.assertTrue(any(
            s.kind == "note" and "stored knowledge" in s.text
            for s in result.trace
        ))

    def test_injection_suppressed_when_off(self) -> None:
        from dataclasses import replace
        from nomorals.agents.reasoning import ReasoningEngine

        self.context.settings = replace(self.context.settings,
                                        reasoning_knowledge="off")
        router = FakeRouter(default="answer: yes")
        self.context.router = router
        engine = ReasoningEngine(self.context, max_llm_calls=6,
                                 max_seconds=60.0)
        engine.reason("How do I apply Django database schema changes?")
        self.assertTrue(router.calls)
        self.assertFalse(any("knowledge graph" in p for p in router.prompts))


# ── 3. the cognitive loop ────────────────────────────────────────────────────

class CognitiveLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_tick_runs_all_stages(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop

        tick = CognitiveLoop(self.context).tick()
        self.assertEqual(set(tick["stages"].keys()),
                         {"goals", "improvement", "training"})
        # improvement is off by default → skipped, not an error
        self.assertIn("skipped", tick["stages"]["improvement"])
        # no training data yet → skipped with the policy decision attached
        self.assertIn("skipped", tick["stages"]["training"])
        self.assertIsInstance(tick["stages"]["goals"].get("active"), int)

    def test_tick_drives_project_goals_to_done(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.projects import ProjectManager

        gs = GoalSystem(self.context)
        g = gs.create("Cascade", description="a. b. c.")
        sp = gs.spawn_project(g.id)
        loop = CognitiveLoop(self.context)
        for _ in range(8):
            if gs.get(g.id).status == "done":
                break
            loop.tick(executor=_ok_executor)
        goal = gs.get(g.id)
        self.assertEqual(goal.status, "done")
        self.assertEqual(goal.progress, 1.0)
        proj = ProjectManager(self.context).status(sp["project_id"])
        self.assertEqual(proj["status"], "done")

    def test_tick_reports_project_driven_goals(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop
        from nomorals.agents.goals import GoalSystem

        gs = GoalSystem(self.context)
        g = gs.create("Track", description="a. b.")
        gs.spawn_project(g.id)
        tick = CognitiveLoop(self.context).tick(executor=_ok_executor)
        self.assertIn(g.id, tick["stages"]["goals"]["project_driven"])

    def test_stage_failure_is_isolated(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.improvement import ImprovementLoop

        from dataclasses import replace

        # improvement must actually run (mode != off) for the fault to fire
        self.context.settings = replace(
            self.context.settings,
            improvement=replace(self.context.settings.improvement,
                                mode="autonomous"))
        gs = GoalSystem(self.context)
        g = gs.create("Survivor", description="a. b.")
        gs.spawn_project(g.id)
        with mock.patch.object(ImprovementLoop, "run_cycle",
                               side_effect=RuntimeError("boom")):
            tick = CognitiveLoop(self.context).tick(executor=_ok_executor)
        # the improvement stage captured the fault…
        self.assertIn("error", tick["stages"]["improvement"])
        self.assertIn("boom", tick["stages"]["improvement"]["error"])
        # …but the other stages still ran
        self.assertIn(g.id, tick["stages"]["goals"].get("advanced", []))
        self.assertIn("skipped", tick["stages"]["training"])

    def test_stage_toggles(self) -> None:
        from dataclasses import replace
        from nomorals.agents.cognition import CognitiveLoop

        self.context.settings = replace(
            self.context.settings,
            autonomy=replace(self.context.settings.autonomy,
                             tick_goals=False, tick_improvement=False,
                             tick_train=False))
        tick = CognitiveLoop(self.context).tick()
        for stage in tick["stages"].values():
            self.assertIn("skipped", stage)

    def test_status_reports_state(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop

        st = CognitiveLoop(self.context).status()
        self.assertIn("enabled", st)
        self.assertIn("interval_hours", st)
        self.assertIn("goals", st)
        self.assertIn("improvement", st)
        self.assertIn("training", st)
        self.assertEqual(st["improvement"]["mode"], "off")


class AutonomyDialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_set_enabled_persists_across_processes(self) -> None:
        from nomorals.agents.cognition import (
            autonomy_enabled, set_autonomy_enabled,
        )

        self.assertFalse(autonomy_enabled(self.context))
        self.assertTrue(set_autonomy_enabled(self.context, True))
        self.assertTrue(autonomy_enabled(self.context))
        # a fresh context over the same home reads the durable flag
        settings = load_settings(overrides={"home": self.tmp.name,
                                            "partner.platforms": "local",
                                            "chat.local_enabled": "true"})
        fresh = build_context(settings, with_executor=False,
                              with_tools=False)
        try:
            self.assertTrue(autonomy_enabled(fresh))
            self.assertFalse(fresh.settings.autonomy.enabled)  # config
            self.assertTrue(set_autonomy_enabled(fresh, False))
            self.assertFalse(autonomy_enabled(fresh))
        finally:
            fresh.close()

    def test_tool_tick_and_status(self) -> None:
        reg = _registry(self.context)
        out = reg.call("autonomy", action="status")
        self.assertTrue(out.ok)
        self.assertIn("enabled", out.value)
        out2 = reg.call("autonomy", action="tick")
        self.assertTrue(out2.ok)
        self.assertEqual(set(out2.value["stages"].keys()),
                         {"goals", "improvement", "training"})


# ── 4. power-mode autonomy dial ──────────────────────────────────────────────

class PowerAutonomyCascadeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def _power(self):
        from nomorals.agents.power import PowerMode

        self.context.settings.partner.owner_key = "k"
        return PowerMode(self.context)

    def test_unlock_cascades_the_stack(self) -> None:
        power = self._power()
        report = power.unlock("k")
        self.assertTrue(report.get("ok"))
        fields = {c["field"] for c in report["changes"]}
        # the two root-level cascade switches are both widened…
        self.assertIn("autonomy", fields)
        self.assertIn("improvement", fields)
        self.assertTrue(self.context.settings.autonomy.enabled)
        self.assertTrue(self.context.settings.improvement.auto_tick)
        # …but the router stays OFF without a downloaded local model
        self.assertEqual(self.context.settings.router_intelligent, "off")
        power.lock()
        self.assertFalse(self.context.settings.autonomy.enabled)
        self.assertFalse(self.context.settings.improvement.auto_tick)
        self.assertEqual(self.context.settings.router_intelligent, "off")

    def test_router_on_when_local_model_present(self) -> None:
        from dataclasses import replace

        self.context.settings = replace(self.context.settings,
                                        llm=replace(self.context.settings.llm,
                                                    local_model="model.gguf"))
        power = self._power()
        report = power.unlock("k")
        fields = {c["field"] for c in report["changes"]}
        self.assertIn("router_intelligent", fields)
        self.assertEqual(self.context.settings.router_intelligent, "on")
        power.lock()
        self.assertEqual(self.context.settings.router_intelligent, "off")

    def test_restore_is_exact(self) -> None:
        power = self._power()
        before = (self.context.settings.autonomy,
                  self.context.settings.improvement,
                  self.context.settings.partner.max_parallel_chats)
        power.unlock("k")
        self.assertNotEqual(self.context.settings.partner.max_parallel_chats,
                            before[2])
        power.lock()
        self.assertEqual(self.context.settings.autonomy, before[0])
        self.assertEqual(self.context.settings.improvement, before[1])
        self.assertEqual(self.context.settings.partner.max_parallel_chats,
                         before[2])


# ── 5. scheduler glue ────────────────────────────────────────────────────────

class _FakeGateway:
    """Just enough gateway surface for the runtime's boot path."""

    def start(self, callback) -> list[str]:
        return ["fake"]

    def stop(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {"_stats": {}}

    def send(self, *a, **k):
        return SimpleNamespace(ok=True, error="")

    def send_file(self, *a, **k):
        return SimpleNamespace(ok=False, error="no")

    def typing(self, *a, **k):
        pass

    def set_rate_limit(self, n) -> None:
        pass


class SchedulerGlueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.cognition import set_autonomy_enabled

        set_autonomy_enabled(self.context, True)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def _runtime(self):
        from nomorals.agents.partner_runtime import PartnerRuntime

        return PartnerRuntime(self.context, gateway=_FakeGateway(),
                              dry_run=False)

    def _cognitive_jobs(self, runtime) -> list[dict[str, Any]]:
        return [j for j in runtime._scheduler.list_jobs()
                if j.get("name") == "cognitive loop"]

    def test_job_registered_once_and_idempotent(self) -> None:
        rt1 = self._runtime()
        try:
            rt1.start()
            jobs = self._cognitive_jobs(rt1)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["kind"], "every")
            self.assertEqual(jobs[0]["payload_kind"], "tool")
            raw = self.context.db.query_one(
                "SELECT payload FROM schedule_jobs "
                "WHERE name='cognitive loop'")
            self.assertEqual(json.loads(raw["payload"]),
                             {"tool": "autonomy", "args": {"action": "tick"}})
            # a second boot over the same home must not duplicate the job
            rt2 = self._runtime()
            try:
                rt2.start()
                self.assertEqual(len(self._cognitive_jobs(rt2)), 1)
            finally:
                rt2.stop()
        finally:
            rt1.stop()

    def test_job_not_registered_when_autonomy_off(self) -> None:
        from nomorals.agents.cognition import set_autonomy_enabled

        set_autonomy_enabled(self.context, False)
        rt = self._runtime()
        try:
            rt.start()
            self.assertEqual(self._cognitive_jobs(rt), [])
        finally:
            rt.stop()


# ── CLI ──────────────────────────────────────────────────────────────────────

class CliAutonomyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-wave51-cli-")
        self._old_home = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = self.tmp.name

    def tearDown(self) -> None:
        try:
            if self._old_home is None:
                os.environ.pop("NM_HOME", None)
            else:
                os.environ["NM_HOME"] = self._old_home
        finally:
            self.tmp.cleanup()

    def _run(self, *argv: str) -> tuple[int, str]:
        import io

        from nomorals.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_goal_create_with_project_flag(self) -> None:
        code, out = self._run("goal", "create", "Ship the blog",
                              "--description", "design it. build it. launch it.",
                              "--project")
        self.assertEqual(code, 0)
        self.assertIn("project", out)
        self.assertIn("spawned", out)

    def test_autonomy_status_on_tick(self) -> None:
        code, out = self._run("autonomy", "status")
        self.assertEqual(code, 0)
        self.assertIn("autonomy: off", out)
        code, out = self._run("autonomy", "on")
        self.assertEqual(code, 0)
        self.assertIn("persisted", out)
        # the next process sees it on
        code, out = self._run("autonomy", "status", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["enabled"])
        # tick works and reports every stage
        code, out = self._run("autonomy", "tick")
        self.assertEqual(code, 0)
        self.assertIn("cognitive loop tick", out)
        # off again, persistently
        code, out = self._run("autonomy", "off")
        self.assertEqual(code, 0)
        code, out = self._run("autonomy", "status", "--json")
        self.assertFalse(json.loads(out)["enabled"])


# ── config validation ────────────────────────────────────────────────────────

class ConfigWave51Test(unittest.TestCase):
    def test_defaults(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-wave51-cfg-")
        try:
            s = load_settings(overrides={"home": tmp.name})
            self.assertFalse(s.autonomy.enabled)
            self.assertEqual(s.autonomy.interval_hours, 6.0)
            self.assertTrue(s.autonomy.tick_goals)
            self.assertTrue(s.autonomy.tick_improvement)
            self.assertTrue(s.autonomy.tick_train)
            self.assertEqual(s.reasoning_knowledge, "on")
        finally:
            tmp.cleanup()

    def test_interval_validation(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"autonomy.interval_hours": "0.1"})

    def test_reasoning_knowledge_rejects_bad_value(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"reasoning_knowledge": "sometimes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
