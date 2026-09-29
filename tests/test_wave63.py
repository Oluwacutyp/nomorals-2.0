"""Wave 63 — autonomous replanning, cross-goal intelligence, cost-aware
cadence.

Hermetic end-to-end tests:

  1. cross-goal intelligence — finished goals write what they produced
                     into the knowledge graph; goals that build on them
                     start from the real artifacts (injected into the
                     executor's prompt)
  2. autonomous replanning   — a project that keeps failing is not just
                     reworded: from the Nth heal onwards it is RE-PLANNED
                     with failure context (what was tried + what the
                     failure analyzer learned), keeping done steps
  3. cost-aware cadence      — the cognitive loop watches its own
                     telemetry and tightens the heartbeat when busy,
                     relaxes it when idle (applied live to the scheduler)

No network egress. LLM paths run through a scripted :class:`FakeRouter`.
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
from typing import Any

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
    tmp = tempfile.TemporaryDirectory(prefix="nm-wave63-")
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


def _ok(d: str) -> str:
    return "ok: " + d


def _port_fail(d: str) -> str:
    raise RuntimeError("Address already in use on 127.0.0.1:8080")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.projects import ProjectManager

        self.gs = GoalSystem(self.context)
        self.pm = ProjectManager(self.context)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()


# ── 1. cross-goal intelligence ──────────────────────────────────────────────

class CrossGoalIntelligenceTest(_Base):
    def _finish_goal(self, g, note="collected the base data") -> None:
        for _ in range(6):
            if self.gs.get(g.id).status == "done":
                return
            self.gs.advance(g.id, executor=lambda d, n=note: n)

    def test_completion_writes_kg_node(self) -> None:
        g = self.gs.create("Base data", description="collect it",
                           plan=["collect the data"])
        self._finish_goal(g, note="the base dataset is ready")
        rows = self.context.db.query(
            "SELECT * FROM kg_nodes WHERE type='goal' AND label=?",
            ("Base data",))
        self.assertEqual(len(rows), 1)
        props = json.loads(rows[0]["properties"])
        self.assertEqual(props["id"], g.id)
        self.assertIn("base dataset", str(props.get("results", "")))

    def test_completion_links_dependents(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"])
        self.gs.add_dependency(top.id, base.id)
        self._finish_goal(base)
        edges = self.context.db.query(
            "SELECT * FROM kg_edges WHERE relation='depends_on'")
        self.assertEqual(len(edges), 1)
        self._finish_goal(top)
        # finishing the dependent adds no extra edge (it's done now)
        edges = self.context.db.query(
            "SELECT * FROM kg_edges WHERE relation='depends_on'")
        self.assertEqual(len(edges), 1)

    def test_upstream_knowledge(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"])
        self.gs.add_dependency(top.id, base.id)
        # nothing done yet
        self.assertEqual(self.gs.upstream_knowledge(top.id), "")
        self._finish_goal(base, note="the schema is in place")
        block = self.gs.upstream_knowledge(top.id)
        self.assertIn("base", block)
        self.assertIn("schema is in place", block)

    def test_executor_receives_upstream_knowledge(self) -> None:
        # a project-backed goal that depends on a finished goal must SEE
        # what the dependency produced in its execution prompt
        base = self.gs.create("upstream work", plan=["x"])
        self._finish_goal(base, note="the API is live at /v1")
        g = self.gs.create("downstream work", plan=["y"])
        self.gs.add_dependency(g.id, base.id)
        sp = self.gs.spawn_project(g.id)
        router = FakeRouter(default="step done")
        self.context.router = router
        self.pm.run(sp["project_id"], executor=None, max_steps=10)
        self.assertTrue(router.calls)
        joined = "\n".join(router.prompts)
        self.assertIn("Completed upstream goals", joined)
        self.assertIn("API is live at /v1", joined)

    def test_completion_idempotent(self) -> None:
        g = self.gs.create("once", plan=["x"])
        self._finish_goal(g)
        self.gs._record_completion_knowledge(g.id)
        self.gs._record_completion_knowledge(g.id)
        n = self.context.db.scalar(
            "SELECT COUNT(*) FROM kg_nodes WHERE type='goal' "
            "AND label=?", ("once",))
        self.assertEqual(int(n), 1)


# ── 2. autonomous replanning ────────────────────────────────────────────────

class _CliHome:
    """CLI tests get an isolated NM_HOME (never the real one)."""

    def setUp(self) -> None:
        super().setUp()
        self._cli_tmp = tempfile.TemporaryDirectory(prefix="nm-wave63-cli-")
        self._old_home = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = self._cli_tmp.name

    def tearDown(self) -> None:
        try:
            if self._old_home is None:
                os.environ.pop("NM_HOME", None)
            else:
                os.environ["NM_HOME"] = self._old_home
        finally:
            self._cli_tmp.cleanup()
            super().tearDown()


class ReplanningTest(_CliHome, _Base):
    def test_replan_keeps_done_and_plans_fresh(self) -> None:
        g = self.gs.create("stuck", description="a. b. c.",
                           plan=["a", "b", "c"])
        sp = self.gs.spawn_project(g.id)
        pid = sp["project_id"]
        # get one step done, then fail the rest
        self.pm.advance(pid, executor=_ok)

        self.pm.run(pid, executor=_port_fail, max_attempts=2, max_steps=8)
        self.assertEqual(self.pm.status(pid)["status"], "failed")
        out = self.pm.replan(pid, context_note="the port was busy")
        self.assertTrue(out["ok"])
        self.assertEqual(out["kept_steps"], 1)
        self.assertGreaterEqual(out["new_steps"], 1)
        st = self.pm.status(pid)
        self.assertEqual(st["status"], "running")
        self.assertEqual(st["steps"][0]["status"], "done")

    def test_replan_refuses_done_project(self) -> None:
        g = self.gs.create("fine", plan=["x"])
        sp = self.gs.spawn_project(g.id)
        self.pm.run(sp["project_id"], executor=_ok, max_steps=10)
        out = self.pm.replan(sp["project_id"])
        self.assertFalse(out["ok"])
        self.assertIn("done", out["error"])

    def test_replan_prompt_carries_failure_context(self) -> None:
        # a failed step (learned) + replan: the planning prompt must carry
        # what was tried and the learned lessons
        g = self.gs.create("context", plan=["bind to 127.0.0.1:8080"])
        sp = self.gs.spawn_project(g.id)
        pid = sp["project_id"]
        self.pm.run(pid, executor=_port_fail, max_attempts=2, max_steps=5)
        router = FakeRouter(
            script={"Objective": '["do it differently", "verify it"]'},
            default="fallback")
        self.context.router = router
        out = self.pm.replan(pid)
        self.assertTrue(out["ok"])
        self.assertTrue(router.calls)
        prompt = "\n".join(router.prompts)
        self.assertIn("Context from past attempts", prompt)
        self.assertIn("8080", prompt)

    def test_heal_escalates_to_replan(self) -> None:
        g = self.gs.create("escalate", plan=["bind to the port"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        self.pm.run(pid, executor=_port_fail, max_attempts=2, max_steps=5)
        h1 = self.pm.heal(pid)
        self.assertEqual(h1["mode"], "retry")
        # burn the retry budget: the goal has been healed twice already
        self.context.db.execute(
            "UPDATE agent_goals SET heals=? WHERE id=?", (2, g.id))
        self.pm.run(pid, executor=_port_fail, max_attempts=2, max_steps=5)
        h2 = self.pm.heal(pid)
        self.assertEqual(h2["mode"], "replan")
        self.assertGreaterEqual(h2["new_steps"], 1)

    def test_replan_never_when_disabled(self) -> None:
        from dataclasses import replace

        self.context.settings = replace(
            self.context.settings,
            autonomy=replace(self.context.settings.autonomy,
                             replan_after_heals=0))
        g = self.gs.create("no replan", plan=["bind to the port"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        self.context.db.execute(
            "UPDATE agent_goals SET heals=? WHERE id=?", (9, g.id))
        self.pm.run(pid, executor=_port_fail, max_attempts=2, max_steps=5)
        h = self.pm.heal(pid)
        self.assertEqual(h["mode"], "retry")

    def test_tool_replan(self) -> None:
        reg = _registry(self.context)
        g = self.gs.create("tool replan", plan=["x", "y"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        out = reg.call("project", action="replan", project_id=pid)
        self.assertTrue(out.ok)
        self.assertTrue(out.value["ok"])

    def test_cli_replan(self) -> None:
        code, out = self._run_cli("project", "create", "CliReplan",
                                  "bind to 127.0.0.1:8080")
        self.assertEqual(code, 0)
        code, out = self._run_cli("project", "list")
        pid = out.strip().split()[0]
        code, out = self._run_cli("project", "replan", pid)
        self.assertEqual(code, 0)
        self.assertIn("replanned", out)

    def _run_cli(self, *argv: str) -> tuple[int, str]:
        import io

        from nomorals.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


# ── 3. cost-aware cadence ───────────────────────────────────────────────────

class AdaptiveCadenceTest(_Base):
    def _loop(self):
        from nomorals.agents.cognition import CognitiveLoop

        return CognitiveLoop(self.context)

    def _seed_tick(self, *, advanced=0, healed=0,
                   improvement="skipped", training="skipped") -> None:
        from nomorals.core.ids import new_short_id

        imp = ({"skipped": "improvement mode off"}
               if improvement == "skipped" else {"ran": True})
        tr = ({"skipped": "no training data"}
              if training == "skipped" else {"ran": True})
        self.context.db.execute(
            "INSERT INTO cognition_log (id, ts, stages, seconds, note) "
            "VALUES (?,?,?,?,?)",
            (new_short_id("cog"), time.time(),
             json.dumps({"goals": {"advanced": list(range(advanced)),
                                   "healed": list(range(healed))},
                         "improvement": imp, "training": tr}),
             0.1, ""))

    def test_idle_relaxes(self) -> None:
        loop = self._loop()
        for _ in range(4):
            self._seed_tick()
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(loop.adaptive_interval(), min(48.0, base * 4))

    def test_busy_tightens(self) -> None:
        loop = self._loop()
        for _ in range(4):
            self._seed_tick(advanced=1)
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(loop.adaptive_interval(),
                         max(0.25, min(base, base * 0.25)))

    def test_mixed_holds_base(self) -> None:
        loop = self._loop()
        self._seed_tick(advanced=1)
        self._seed_tick()
        self._seed_tick()
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(loop.adaptive_interval(), base)

    def test_off_is_always_base(self) -> None:
        from dataclasses import replace

        self.context.settings = replace(
            self.context.settings,
            autonomy=replace(self.context.settings.autonomy,
                             adaptive_cadence=False))
        loop = self._loop()
        for _ in range(4):
            self._seed_tick()
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(loop.adaptive_interval(), base)
        self.assertEqual(loop.effective_interval(), base)

    def test_idle_cap_48h(self) -> None:
        from dataclasses import replace

        self.context.settings = replace(
            self.context.settings,
            autonomy=replace(self.context.settings.autonomy,
                             interval_hours=20.0))
        loop = self._loop()
        for _ in range(4):
            self._seed_tick()
        self.assertEqual(loop.adaptive_interval(), 48.0)

    def test_tick_persists_cadence(self) -> None:
        loop = self._loop()
        tick = loop.tick()
        self.assertIn("next_interval_hours", tick)
        row = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key='autonomy.adaptive_interval'")
        self.assertIsNotNone(row)
        data = json.loads(row["value"])
        self.assertIn("interval_hours", data)
        self.assertEqual(loop.effective_interval(),
                         data["interval_hours"])

    def test_live_application_to_scheduler(self) -> None:
        calls: list[tuple[str, int]] = []

        class _FakeScheduler:
            def set_interval(self, ref, seconds):
                calls.append((ref, int(seconds)))
                return {}

        self.context.extras["scheduler"] = _FakeScheduler()
        loop = self._loop()
        for _ in range(4):
            self._seed_tick(advanced=1)
        loop.tick(executor=_ok)
        self.assertTrue(calls)
        ref, seconds = calls[-1]
        self.assertEqual(ref, "cognitive loop")
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(seconds, int(max(0.25, base * 0.25) * 3600))

    def test_scheduler_set_interval(self) -> None:
        from nomorals.agents.scheduler import Scheduler

        sch = Scheduler(self.context)
        sch.add("cognitive loop", "every 6h", "tool",
                {"tool": "autonomy", "args": {"action": "tick"}})
        row = sch.set_interval("cognitive loop", 5400)
        self.assertIsNotNone(row)
        jobs = {j["name"]: j for j in sch.list_jobs()}
        self.assertEqual(jobs["cognitive loop"]["spec"], "every 5400s")
        # one-shot and daily jobs are left alone
        sch.add("oneshot", "at 2099-01-01 00:00", "message",
                {"text": "hi"})
        self.assertIsNone(sch.set_interval("oneshot", 60))
        self.assertIsNone(sch.set_interval("ghost job", 60))
        # clamped to the scheduler's own limits
        row = sch.set_interval("cognitive loop", 10**9)
        jobs = {j["name"]: j for j in sch.list_jobs()}
        self.assertEqual(jobs["cognitive loop"]["spec"],
                         f"every {int(48 * 3600)}s")


# ── 4. config + views ───────────────────────────────────────────────────────

class ConfigWave63Test(unittest.TestCase):
    def test_defaults(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-wave63-cfg-")
        try:
            s = load_settings(overrides={"home": tmp.name})
            self.assertEqual(s.autonomy.replan_after_heals, 2)
            self.assertTrue(s.autonomy.adaptive_cadence)
        finally:
            tmp.cleanup()

    def test_replan_validation(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"autonomy.replan_after_heals": "-1"})


class MissionCadenceViewTest(_CliHome, _Base):
    def test_plan_reports_cadence(self) -> None:
        from nomorals.agents.mission import MissionControl

        plan = MissionControl(self.context).plan()
        self.assertIn("cadence", plan)
        self.assertEqual(plan["cadence"]["base_hours"],
                         self.context.settings.autonomy.interval_hours)
        self.assertTrue(plan["cadence"]["adaptive"])
        self.assertIsNotNone(plan["cadence"]["effective_hours"])

    def test_mission_cli_shows_heartbeat(self) -> None:
        code, out = self._run_cli("mission", "plan")
        self.assertEqual(code, 0)
        self.assertIn("heartbeat", out)

    def _run_cli(self, *argv: str) -> tuple[int, str]:
        import io

        from nomorals.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


if __name__ == "__main__":
    unittest.main(verbosity=2)
