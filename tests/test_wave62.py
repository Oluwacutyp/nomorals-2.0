"""Wave 62 — mission control, autonomy telemetry, self-healing projects.

Hermetic end-to-end tests:

  1. mission control   — goal priorities + dependencies, the portfolio
                         plan (ready / blocked / paused / next)
  2. autonomy telemetry — every heartbeat journaled into cognition_log,
                         per-stage outcome reports
  3. self-healing      — a failed project feeds the failure analyzer,
                         ``heal`` rewords the exhausted steps with the
                         learned lessons and re-runs; the cognitive loop
                         auto-heals (bounded by ``max_project_heals``)

No network egress. LLM paths run through a scripted :class:`FakeRouter`;
everything that would otherwise call a model is deterministic.
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
    tmp = tempfile.TemporaryDirectory(prefix="nm-wave62-")
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


def _ok(description: str) -> str:
    return "ok: " + description


def _always_fail(description: str) -> str:
    raise RuntimeError("Address already in use on 127.0.0.1:8080")


# ── 1. migration 19 + mission control ───────────────────────────────────────

class Migration19Test(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_new_goal_columns(self) -> None:
        cols = [r["name"] for r in self.context.db.query(
            "PRAGMA table_info(goals)")]
        for col in ("priority", "depends_on", "heals"):
            self.assertIn(col, cols)

    def test_cognition_log_table(self) -> None:
        row = self.context.db.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='cognition_log'")
        self.assertIsNotNone(row)

    def test_latest_version(self) -> None:
        from nomorals.storage.migrations import latest_version

        # wave 62 introduced schema v19; later waves may add more
        self.assertGreaterEqual(latest_version(), 19)


class GoalPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.goals import GoalSystem

        self.gs = GoalSystem(self.context)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_set_priority_and_ordering(self) -> None:
        low = self.gs.create("low", plan=["a"])
        high = self.gs.create("high", plan=["b"])
        self.gs.set_priority(high.id, 10)
        nxt = self.gs.next_goal()
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.id, high.id)

    def test_create_with_priority(self) -> None:
        g = self.gs.create("born high", plan=["a"], priority=7)
        self.assertEqual(g.priority, 7)
        self.assertEqual(self.gs.get(g.id).priority, 7)

    def test_tie_breaks_by_age(self) -> None:
        older = self.gs.create("older", plan=["a"])
        time.sleep(0.01)
        younger = self.gs.create("younger", plan=["b"])
        nxt = self.gs.next_goal()
        self.assertEqual(nxt.id, older.id)


class GoalDependencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.goals import GoalSystem

        self.gs = GoalSystem(self.context)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_blocked_until_dependency_done(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"], priority=100)
        self.gs.add_dependency(top.id, base.id)
        # top is higher priority but blocked
        nxt = self.gs.next_goal()
        self.assertEqual(nxt.id, base.id)
        self.assertFalse(self.gs.dependencies_met(self.gs.get(top.id)))
        # finish the base goal
        for _ in range(5):
            if self.gs.get(base.id).status == "done":
                break
            self.gs.advance(base.id, executor=_ok)
        self.assertTrue(self.gs.dependencies_met(self.gs.get(top.id)))
        self.assertEqual(self.gs.next_goal().id, top.id)

    def test_cycle_rejected(self) -> None:
        a = self.gs.create("A", plan=["x"])
        b = self.gs.create("B", plan=["y"])
        self.gs.add_dependency(b.id, a.id)
        result = self.gs.add_dependency(a.id, b.id)
        # a must NOT now depend on b (that would be a cycle)
        self.assertNotIn(b.id, result.depends_on)

    def test_self_dependency_rejected(self) -> None:
        a = self.gs.create("solo", plan=["x"])
        result = self.gs.add_dependency(a.id, a.id)
        self.assertNotIn(a.id, result.depends_on)

    def test_remove_dependency(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"])
        self.gs.add_dependency(top.id, base.id)
        self.assertEqual(self.gs.get(top.id).depends_on, [base.id])
        self.gs.remove_dependency(top.id, base.id)
        self.assertEqual(self.gs.get(top.id).depends_on, [])

    def test_tick_skips_blocked_goals(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"])
        self.gs.add_dependency(top.id, base.id)
        ticked = self.gs.tick(executor=_ok)
        ids = [g.id for g in ticked]
        self.assertIn(base.id, ids)
        self.assertNotIn(top.id, ids)


class MissionControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.goals import GoalSystem
        from nomorals.agents.mission import MissionControl

        self.gs = GoalSystem(self.context)
        self.mc = MissionControl(self.context)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_plan_partitions(self) -> None:
        base = self.gs.create("base", plan=["x"])
        top = self.gs.create("top", plan=["y"], priority=5)
        self.gs.add_dependency(top.id, base.id)
        done = self.gs.create("finished", plan=["z"])
        self.gs.complete(done.id)
        plan = self.mc.plan()
        self.assertEqual(plan["next"]["id"], base.id)
        ready = {g["id"] for g in plan["ready"]}
        blocked = {g["id"] for g in plan["blocked"]}
        self.assertIn(base.id, ready)
        self.assertIn(top.id, blocked)
        self.assertIn(done.id, [g["id"] for g in plan["done"]])
        waiting = next(g["waiting_on"] for g in plan["blocked"]
                       if g["id"] == top.id)
        self.assertEqual(waiting, [base.id])
        self.assertEqual(plan["counts"]["ready"], 1)
        self.assertEqual(plan["counts"]["blocked"], 1)

    def test_plan_shows_live_projects(self) -> None:
        g = self.gs.create("with project", plan=["x"])
        self.gs.spawn_project(g.id)
        plan = self.mc.plan()
        self.assertEqual(plan["counts"]["live_projects"], 1)
        self.assertEqual(plan["projects"][0]["goal_id"], g.id)

    def test_next_action(self) -> None:
        self.assertEqual(self.mc.next()["next"], None)
        g = self.gs.create("the one", plan=["x"])
        out = self.mc.next()
        self.assertEqual(out["next"]["id"], g.id)

    def test_tool(self) -> None:
        reg = _registry(self.context)
        g = self.gs.create("tool goal", plan=["x"])
        out = reg.call("mission", action="plan")
        self.assertTrue(out.ok)
        self.assertEqual(out.value["next"]["id"], g.id)
        out2 = reg.call("mission", action="next")
        self.assertTrue(out2.ok)


# ── 2. autonomy telemetry ───────────────────────────────────────────────────

class TelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        from nomorals.agents.cognition import CognitiveLoop

        self.loop = CognitiveLoop(self.context)

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_tick_is_journaled(self) -> None:
        self.loop.tick()
        self.loop.tick()
        rows = self.context.db.query(
            "SELECT * FROM cognition_log ORDER BY ts ASC")
        self.assertEqual(len(rows), 2)
        self.assertIn("goals", json.loads(rows[0]["stages"]))

    def test_recent(self) -> None:
        self.loop.tick()
        recent = self.loop.recent(limit=5)
        self.assertEqual(len(recent), 1)
        self.assertIn("stages", recent[0])
        self.assertIn("note", recent[0])

    def test_report_aggregates_stages(self) -> None:
        self.loop.tick()
        rep = self.loop.report()
        self.assertEqual(rep["ticks"], 1)
        self.assertEqual(rep["stages"]["goals"]["ran"], 1)
        self.assertEqual(rep["stages"]["improvement"]["skipped"], 1)
        self.assertEqual(rep["stages"]["training"]["skipped"], 1)
        self.assertEqual(len(rep["recent"]), 1)

    def test_report_counts_errors(self) -> None:
        # simulate an errored stage row directly (the loop catches real
        # errors the same way)
        from nomorals.core.ids import new_short_id

        self.context.db.execute(
            "INSERT INTO cognition_log (id, ts, stages, seconds, note) "
            "VALUES (?,?,?,?,?)",
            (new_short_id("cog"), time.time(),
             json.dumps({"goals": {"advanced": []},
                         "improvement": {"error": "boom"},
                         "training": {"skipped": "no data"}}),
             0.5, "improvement error"))
        rep = self.loop.report()
        self.assertEqual(rep["ticks"], 1)
        self.assertEqual(rep["stages"]["improvement"]["error"], 1)

    def test_tool_report(self) -> None:
        reg = _registry(self.context)
        self.loop.tick()
        out = reg.call("autonomy", action="report")
        self.assertTrue(out.ok)
        self.assertEqual(out.value["ticks"], 1)


# ── 3. self-healing projects ────────────────────────────────────────────────

class SelfHealingTest(unittest.TestCase):
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

    def test_failure_is_learned(self) -> None:
        from nomorals.agents.failure import FailureAnalyzer

        g = self.gs.create("flaky port", plan=["bind to 127.0.0.1:8080"])
        sp = self.gs.spawn_project(g.id)
        res = self.pm.run(sp["project_id"], executor=_always_fail,
                          max_attempts=2, max_steps=5)
        self.assertEqual(res["status"], "failed")
        # a coding_log row was journaled and a lesson persisted
        row = self.context.db.query_one(
            "SELECT * FROM coding_log WHERE exit_code != 0 LIMIT 1")
        self.assertIsNotNone(row)
        analyzer = FailureAnalyzer(self.context)
        ctx = analyzer.prevention_context("bind to 127.0.0.1:8080")
        self.assertIn("8080", ctx)

    def test_heal_revives_failed_project(self) -> None:
        g = self.gs.create("revive", plan=["bind to the port"])
        sp = self.gs.spawn_project(g.id)
        self.pm.run(sp["project_id"], executor=_always_fail,
                    max_attempts=2, max_steps=5)
        self.assertEqual(self.gs.get(g.id).status, "paused")
        out = self.pm.heal(sp["project_id"])
        self.assertTrue(out["ok"])
        self.assertEqual(out["healed_steps"], 1)
        st = self.pm.status(sp["project_id"])
        self.assertEqual(st["status"], "running")
        step = self.pm._load(sp["project_id"]).steps[0]
        self.assertEqual(step.status, "pending")
        self.assertEqual(step.attempts, 0)
        # the linked goal is active again
        self.assertEqual(self.gs.get(g.id).status, "active")
        # and now it runs to completion
        res = self.pm.run(sp["project_id"], executor=_ok, max_steps=10)
        self.assertEqual(res["status"], "done")
        self.assertEqual(self.gs.get(g.id).status, "done")

    def test_heal_refuses_completed(self) -> None:
        g = self.gs.create("already done", plan=["x"])
        sp = self.gs.spawn_project(g.id)
        self.pm.run(sp["project_id"], executor=_ok, max_steps=10)
        out = self.pm.heal(sp["project_id"])
        self.assertFalse(out["ok"])

    def test_loop_auto_heals_bounded(self) -> None:
        from nomorals.agents.cognition import CognitiveLoop

        g = self.gs.create("auto heal", plan=["auto step"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        self.pm.run(pid, executor=_always_fail, max_attempts=2, max_steps=5)
        self.assertEqual(self.gs.get(g.id).status, "paused")
        loop = CognitiveLoop(self.context)
        tick1 = loop.tick(executor=_ok)
        self.assertIn(g.id, tick1["stages"]["goals"]["healed"])
        self.assertEqual(self.gs.get(g.id).heals, 1)
        self.assertEqual(self.gs.get(g.id).status, "active")
        # exhaust the heal budget and fail again
        self.context.db.execute(
            "UPDATE goals SET heals=? WHERE id=?",
            (self.context.settings.autonomy.max_project_heals, g.id))
        self.pm.run(pid, executor=_always_fail, max_attempts=2, max_steps=5)
        tick2 = loop.tick(executor=_ok)
        self.assertNotIn(g.id, tick2["stages"]["goals"].get("healed", []))
        self.assertEqual(self.gs.get(g.id).status, "paused")

    def test_heal_disabled_by_config(self) -> None:
        from dataclasses import replace

        from nomorals.agents.cognition import CognitiveLoop

        self.context.settings = replace(
            self.context.settings,
            autonomy=replace(self.context.settings.autonomy,
                             max_project_heals=0))
        g = self.gs.create("no heal", plan=["x"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        self.pm.run(pid, executor=_always_fail, max_attempts=2, max_steps=5)
        tick = CognitiveLoop(self.context).tick(executor=_ok)
        self.assertEqual(tick["stages"]["goals"].get("healed", []), [])
        self.assertEqual(self.gs.get(g.id).status, "paused")

    def test_project_tool_heal(self) -> None:
        reg = _registry(self.context)
        g = self.gs.create("tool heal", plan=["x"])
        pid = self.gs.spawn_project(g.id)["project_id"]
        self.pm.run(pid, executor=_always_fail, max_attempts=2, max_steps=5)
        out = reg.call("project", action="heal", project_id=pid)
        self.assertTrue(out.ok)
        self.assertTrue(out.value["ok"])
        self.assertEqual(out.value["healed_steps"], 1)

    def test_goal_tool_priority_and_depends(self) -> None:
        reg = _registry(self.context)
        g = self.gs.create("t", plan=["x"])
        d = self.gs.create("d", plan=["y"])
        out = reg.call("goal", action="priority", goal_id=g.id,
                       priority="3")
        self.assertTrue(out.ok)
        self.assertEqual(out.value["goal"]["priority"], 3)
        out2 = reg.call("goal", action="depends", goal_id=g.id,
                        depends_on=d.id)
        self.assertTrue(out2.ok)
        self.assertIn(d.id, out2.value["goal"]["depends_on"])
        out3 = reg.call("goal", action="depends", goal_id=g.id,
                        depends_on=d.id, remove="1")
        self.assertTrue(out3.ok)
        self.assertEqual(out3.value["goal"]["depends_on"], [])
        out4 = reg.call("goal", action="next")
        self.assertTrue(out4.ok)


# ── CLI ──────────────────────────────────────────────────────────────────────

class CliWave62Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-wave62-cli-")
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

    def _goal_ids(self) -> dict[str, str]:
        code, out = self._run("goal", "list", "--json")
        self.assertEqual(code, 0)
        return {g["title"]: g["id"] for g in json.loads(out)["goals"]}

    def test_goal_priority_and_depends(self) -> None:
        code, _ = self._run("goal", "create", "Base", "--description", "b")
        self.assertEqual(code, 0)
        code, _ = self._run("goal", "create", "Top", "--description", "t",
                            "--priority", "5")
        self.assertEqual(code, 0)
        ids = self._goal_ids()
        code, out = self._run("goal", "depends", ids["Top"], ids["Base"])
        self.assertEqual(code, 0)
        code, out = self._run("goal", "next")
        self.assertEqual(code, 0)
        self.assertIn(ids["Base"], out)
        code, out = self._run("goal", "priority", ids["Top"], "9")
        self.assertEqual(code, 0)
        code, out = self._run("goal", "depends", ids["Top"], ids["Base"],
                              "--remove")
        self.assertEqual(code, 0)
        code, out = self._run("goal", "next")
        self.assertIn(ids["Top"], out)

    def test_mission_plan_and_next(self) -> None:
        code, _ = self._run("goal", "create", "Base2", "--description", "b")
        code, _ = self._run("goal", "create", "Top2", "--description", "t",
                            "--priority", "5")
        ids = self._goal_ids()
        self._run("goal", "depends", ids["Top2"], ids["Base2"])
        code, out = self._run("mission", "plan")
        self.assertEqual(code, 0)
        self.assertIn("ready 1", out)
        self.assertIn("blocked 1", out)
        self.assertIn(ids["Base2"], out)
        code, out = self._run("mission", "next")
        self.assertIn(ids["Base2"], out)

    def test_autonomy_report(self) -> None:
        code, _ = self._run("autonomy", "tick")
        self.assertEqual(code, 0)
        code, out = self._run("autonomy", "report")
        self.assertEqual(code, 0)
        self.assertIn("heartbeat", out)
        self.assertIn("goals: ran=1", out)
        code, out = self._run("autonomy", "report", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["ticks"], 1)

    def test_project_heal(self) -> None:
        code, out = self._run("project", "create", "Healable",
                              "bind to 127.0.0.1:8080")
        self.assertEqual(code, 0)
        code, out = self._run("project", "list")
        pid = out.strip().split()[0]
        # seed the failed state (a step that already gave up)
        from nomorals.agents.projects import ProjectManager
        from nomorals.core.config import load_settings
        settings = load_settings(overrides={"home": self.tmp.name,
                                            "partner.platforms": "local",
                                            "chat.local_enabled": "true"})
        from nomorals.agents.context import build_context
        with build_context(settings) as context:
            mgr = ProjectManager(context)
            p = mgr.plan(pid)  # give it a step to fail
            p.steps[0].status = "failed"
            p.steps[0].result = "failed after 3 attempts: port busy"
            p.status = "failed"
            mgr._upsert_row(p)
        code, out = self._run("project", "heal", pid)
        self.assertEqual(code, 0)
        self.assertIn("healed 1 step", out)
        data = json.loads(self._run("project", "status", pid, "--json")[1])
        self.assertEqual(data["status"], "running")


# ── config validation ────────────────────────────────────────────────────────

class ConfigWave62Test(unittest.TestCase):
    def test_defaults(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-wave62-cfg-")
        try:
            s = load_settings(overrides={"home": tmp.name})
            self.assertEqual(s.autonomy.max_project_heals, 3)
        finally:
            tmp.cleanup()

    def test_heals_validation(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"autonomy.max_project_heals": "-1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
