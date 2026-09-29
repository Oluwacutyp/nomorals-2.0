"""Wave 64 — reflective checkpointing, the autonomy budget governor, and
portfolio risk scoring.

Hermetic end-to-end tests:

  1. reflective checkpointing — a finished goal is distilled (model pass
                     when one is available, deterministic fallback when
                     not) into a knowledge-graph node, a reusable skill
                     (when there was a failure to learn from), and a
                     durable, idempotent reflection record
  2. autonomy budget governor — the cognitive loop meters the model calls
                     of every tick against a durable daily ledger; with a
                     cap set it skips model stages once exhausted and
                     self-throttles the heartbeat near the limit
  3. portfolio risk scoring   — ``mission plan`` ranks the portfolio by
                     expected value (priority tempered by dependency
                     depth, heal history, and remaining size) and the
                     cognitive loop works goals in EV order

No network egress. LLM paths run through a scripted :class:`FakeRouter`.
"""

from __future__ import annotations

import io
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
    tmp = tempfile.TemporaryDirectory(prefix="nm-wave64-")
    settings = load_settings(
        overrides={"home": tmp.name, "partner.platforms": "local",
                   "chat.local_enabled": "true"})
    context = build_context(settings, with_executor=False, with_tools=False)
    return context, tmp


def _registry(context) -> ToolRegistry:
    reg = ToolRegistry()
    reg.context = context
    reg.register_builtins()
    return reg


def _ok(d: str) -> str:
    return "ok: " + d


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

    def _finish_goal(self, g, note: str = "step done") -> None:
        for _ in range(30):
            if self.gs.get(g.id).status == "done":
                return
            self.gs.advance(g.id, executor=lambda d, n=note: n)
        self.fail(f"goal {g.id} never reached done")


# ── 1. reflective checkpointing ────────────────────────────────────────────

class ReflectionTest(_Base):
    def test_heuristic_on_completion(self) -> None:
        self.context.router = None  # force the deterministic path
        g = self.gs.create("Smoke goal", plan=["step one", "step two"])
        self._finish_goal(g, note="the thing is built")
        row = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key=?",
            ("goal.reflection." + g.id,))
        self.assertIsNotNone(row, "reflection record was not stored")
        rec = json.loads(row["value"])
        self.assertEqual(rec["source"], "heuristic")
        self.assertTrue(rec["what_worked"])
        self.assertIn("2/2 step(s) completed", "; ".join(rec["what_worked"]))
        # KG node + edge
        nodes = self.context.db.query(
            "SELECT * FROM kg_nodes WHERE label=? AND type='fact'",
            ("reflection:Smoke goal",))
        self.assertEqual(len(nodes), 1)
        props = json.loads(nodes[0]["properties"])
        self.assertEqual(props["goal_id"], g.id)
        edges = self.context.db.query(
            "SELECT * FROM kg_edges WHERE relation='reflects'")
        self.assertEqual(len(edges), 1)
        # a clean first-pass run produces no skill (nothing to prevent)
        from nomorals.agents.skills import SkillLibrary

        self.assertIsNone(SkillLibrary(self.context.db)
                          .get_by_name("lesson-smoke-goal"))

    def test_heuristic_with_heal_writes_skill(self) -> None:
        self.context.router = None
        g = self.gs.create("Flaky thing", plan=["bind the port", "verify it"])
        self.context.db.execute(
            "UPDATE agent_goals SET heals=1 WHERE id=?", (g.id,))
        self.gs.advance(g.id, executor=_ok)  # 1/2 — still active
        self.gs.complete(g.id)  # completes the goal WITH the heal on record
        rec = json.loads(self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key=?",
            ("goal.reflection." + g.id,))["value"])
        self.assertEqual(rec["source"], "heuristic")
        self.assertTrue(any("self-heal" in d for d in rec["what_didnt_work"]))
        self.assertTrue(rec["skill_name"])
        from nomorals.agents.skills import SkillLibrary

        skill = SkillLibrary(self.context.db).get_by_name("lesson-flaky-thing")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.kind, "prevention")
        self.assertEqual(skill.source, "reflection")

    def test_model_reflection_parses_json_and_saves_skill(self) -> None:
        g = self.gs.create("Model goal", plan=["bind to 8080", "verify"])
        self._finish_goal(g, note="ok")
        router = FakeRouter(script={
            "Distill what actually worked": (
                '{"what_worked": ["the retry path recovered the bind"], '
                '"what_didnt_work": ["hardcoded port 8080"], '
                '"lessons": ["bind to an ephemeral port"], '
                '"reusable_skill": {"name": "ephemeral-bind", '
                '"kind": "strategy", '
                '"body": "bind port 0 and read back the assigned port"}}'),
        })
        self.context.router = router
        from nomorals.agents.reflection import GoalReflector

        out = GoalReflector(self.context).reflect(g.id, force=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "model")
        self.assertEqual(out["skill_name"], "ephemeral-bind")
        self.assertIn("the retry path recovered the bind", out["what_worked"])
        self.assertEqual(router.calls, 1)
        prompt = router.prompts[0]
        self.assertIn("bind to 8080", prompt)
        self.assertIn("SELF-HEALS", prompt)
        from nomorals.agents.skills import SkillLibrary

        s = SkillLibrary(self.context.db).get_by_name("ephemeral-bind")
        self.assertIsNotNone(s)
        self.assertEqual(s.kind, "strategy")
        self.assertIn("bind port 0", s.body)

    def test_model_vacuous_reply_falls_back_to_heuristic(self) -> None:
        g = self.gs.create("Vacuous", plan=["x"])
        self._finish_goal(g)
        # a parseable JSON dict with none of the reflection fields
        self.context.router = FakeRouter(
            default='{"answer": "done", "confidence": 0.7}')
        from nomorals.agents.reflection import GoalReflector

        out = GoalReflector(self.context).reflect(g.id, force=True)
        self.assertEqual(out["source"], "heuristic")
        self.assertTrue(out["what_worked"])

    def test_reflect_idempotent(self) -> None:
        self.context.router = None
        g = self.gs.create("Once", plan=["x"])
        self._finish_goal(g)
        from nomorals.agents.reflection import GoalReflector

        rf = GoalReflector(self.context)
        first = rf.reflect(g.id, force=True)
        second = rf.reflect(g.id)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        n = self.context.db.scalar(
            "SELECT COUNT(*) FROM kg_nodes WHERE label='reflection:Once'")
        self.assertEqual(int(n), 1)

    def test_reflect_refuses_not_done(self) -> None:
        from nomorals.agents.reflection import GoalReflector

        g = self.gs.create("Not done", plan=["x", "y"])
        out = GoalReflector(self.context).reflect(g.id)
        self.assertFalse(out["ok"])
        self.assertIn("not done", out["error"])

    def test_reflect_disabled_in_settings(self) -> None:
        self.context.router = None
        self.context.settings.autonomy.reflect_on_completion = False
        g = self.gs.create("No reflect", plan=["x"])
        self._finish_goal(g)
        row = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key=?",
            ("goal.reflection." + g.id,))
        self.assertIsNone(row, "auto-reflection should be off")
        # an explicit pass still works
        from nomorals.agents.reflection import GoalReflector

        out = GoalReflector(self.context).reflect(g.id, force=True)
        self.assertTrue(out["ok"])

    def test_upstream_knowledge_in_evidence(self) -> None:
        base = self.gs.create("Base dep", plan=["x"])
        self._finish_goal(base, note="the schema is in place")
        top = self.gs.create("Dependent goal", plan=["y"])
        self.gs.add_dependency(top.id, base.id)
        self.gs.advance(top.id, executor=_ok)
        router = FakeRouter(script={
            "Distill what actually worked":
            '{"what_worked": ["used the schema"], "lessons": ["reuse"]}'})
        self.context.router = router
        from nomorals.agents.reflection import GoalReflector

        out = GoalReflector(self.context).reflect(top.id, force=True)
        self.assertTrue(out["ok"])
        prompt = router.prompts[0]
        self.assertIn("Completed upstream goals", prompt)
        self.assertIn("schema is in place", prompt)

    def test_tool_reflect_and_list(self) -> None:
        self.context.router = None
        g = self.gs.create("Tool goal", plan=["x"])
        self._finish_goal(g)
        reg = _registry(self.context)
        out = reg.call("reflection", action="reflect", goal_id=g.id)
        self.assertTrue(out.ok)
        self.assertTrue(out.value["ok"])
        last = reg.call("reflection", action="last")
        self.assertTrue(last.ok)
        self.assertEqual(last.value["last"]["goal_id"], g.id)
        rows = reg.call("reflection", action="list")
        self.assertTrue(rows.ok)
        self.assertEqual(len(rows.value["reflections"]), 1)
        # also reachable from the goal tool
        via_goal = reg.call("goal", action="reflect", goal_id=g.id)
        self.assertTrue(via_goal.ok)
        self.assertTrue(via_goal.value["ok"])

    def _run_cli(self, *argv: str) -> tuple[int, str]:
        from nomorals.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


class _CliHome:
    """CLI tests get an isolated NM_HOME (never the real one)."""

    def setUp(self) -> None:
        super().setUp()
        self._cli_tmp = tempfile.TemporaryDirectory(prefix="nm-wave64-cli-")
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

    def _run_cli(self, *argv: str) -> tuple[int, str]:
        from nomorals.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


class ReflectionCliTest(_CliHome, _Base):
    def test_cli_reflect(self) -> None:
        code, out = self._run_cli("goal", "create", "CLI goal")
        self.assertEqual(code, 0)
        code, out = self._run_cli("goal", "list")
        gid = out.strip().splitlines()[0].split()[0]
        for _ in range(30):
            code, out = self._run_cli("goal", "list")
            if "[done]" in out.splitlines()[0]:
                break
            self._run_cli("goal", "advance", gid)
        code, out = self._run_cli("goal", "reflect", gid)
        self.assertEqual(code, 0)
        self.assertIn("reflection for", out)
        self.assertIn("kg:", out)
        code, out = self._run_cli("goal", "reflect", "ghost")
        self.assertEqual(code, 0)
        self.assertIn("could not reflect", out)


# ── 2. autonomy budget governor ───────────────────────────────────────────

class BudgetTest(_Base):
    def _budget(self):
        from nomorals.agents.cognition import ModelBudget

        return ModelBudget(self.context)

    def _set_cap(self, cap: int) -> None:
        # in-place mutation — the context holds this exact settings object
        self.context.settings.autonomy.daily_model_calls = int(cap)

    def test_unlimited_by_default(self) -> None:
        rep = self._budget().report()
        self.assertTrue(rep["unlimited"])
        self.assertEqual(rep["cap"], 0)
        self.assertEqual(rep["remaining"], -1)
        self.assertEqual(rep["pressure"], 0.0)
        ok, why = self._budget().allow("goals")
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_record_and_ledger(self) -> None:
        self._set_cap(5)
        b = self._budget()
        b.record(2, 10)
        b.record(3, 5)
        usage = b.usage()
        self.assertEqual(usage["calls"], 5)
        self.assertEqual(usage["tokens"], 15)
        self.assertEqual(usage["ticks"], 2)
        self.assertEqual(b.remaining, 0)
        # the ledger is durable
        row = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key='autonomy.budget'")
        self.assertIsNotNone(row)
        self.assertEqual(json.loads(row["value"])["calls"], 5)

    def test_allow_when_exhausted(self) -> None:
        self._set_cap(1)
        self._budget().record(1, 0)
        ok, why = self._budget().allow("goals")
        self.assertFalse(ok)
        self.assertIn("budget", why)
        self.assertIn("goals", why)

    def test_daily_rollover(self) -> None:
        self._set_cap(5)
        self._budget().record(5, 0)
        self.assertEqual(self._budget().remaining, 0)
        # simulate a previous day in the ledger
        row = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key='autonomy.budget'")
        data = json.loads(row["value"])
        data["date"] = "1999-01-01"
        self.context.db.execute(
            "UPDATE kv_store SET value=? WHERE key='autonomy.budget'",
            (json.dumps(data),))
        self.assertEqual(self._budget().usage()["calls"], 0)
        self.assertEqual(self._budget().remaining, 5)

    def test_tick_meters_model_calls(self) -> None:
        # a project-backed goal drives real router calls inside the tick
        g = self.gs.create("Metered", plan=["do the thing"])
        self.gs.spawn_project(g.id)
        tick = self._loop().tick()
        goals = tick["stages"]["goals"]
        self.assertNotIn("error", goals)
        self.assertGreaterEqual(goals.get("model_calls", 0), 1)
        self.assertGreaterEqual(tick["budget"]["calls_this_tick"], 1)
        self.assertGreaterEqual(tick["budget"]["used"], 1)
        self.assertTrue(tick["budget"]["unlimited"])

    def test_tick_skips_model_stages_when_exhausted(self) -> None:
        self._set_cap(1)
        self._budget().record(1, 0)  # burn the day
        g = self.gs.create("Starved", plan=["x"])
        tick = self._loop().tick()
        goals = tick["stages"]["goals"]
        self.assertIn("skipped", goals)
        self.assertIn("budget", goals["skipped"])
        self.assertIn("skipped", tick["stages"]["improvement"])
        # the goal was NOT advanced
        self.assertEqual(self.gs.get(g.id).status, "active")
        self.assertFalse(any(s.status == "done" for s in
                             self.gs.get(g.id).steps))

    def test_tick_throttles_cadence_near_cap(self) -> None:
        self._set_cap(2)
        self._budget().record(2, 0)  # pressure = 1.0
        calls: list[tuple[str, int]] = []

        class _FakeScheduler:
            def set_interval(self, ref, seconds):
                calls.append((ref, int(seconds)))
                return {}

        self.context.extras["scheduler"] = _FakeScheduler()
        tick = self._loop().tick()
        self.assertTrue(tick["budget"]["throttled"])
        self.assertTrue(calls)
        ref, seconds = calls[-1]
        base = self.context.settings.autonomy.interval_hours
        self.assertEqual(ref, "cognitive loop")
        self.assertGreaterEqual(seconds, int(base * 2 * 3600))

    def test_no_throttle_when_unlimited(self) -> None:
        calls: list[tuple[str, int]] = []

        class _FakeScheduler:
            def set_interval(self, ref, seconds):
                calls.append((ref, int(seconds)))
                return {}

        self.context.extras["scheduler"] = _FakeScheduler()
        tick = self._loop().tick()
        self.assertFalse(tick["budget"]["throttled"])

    def test_status_and_report(self) -> None:
        self._set_cap(10)
        self._budget().record(9, 0)  # pressure 0.9
        rep = self._budget().report()
        self.assertEqual(rep["used"], 9)
        self.assertEqual(rep["remaining"], 1)
        self.assertTrue(rep["throttled"])
        status = self._loop().status()
        self.assertIn("budget", status)
        self.assertEqual(status["budget"]["cap"], 10)

    def _loop(self):
        from nomorals.agents.cognition import CognitiveLoop

        return CognitiveLoop(self.context)


class BudgetCliTest(_CliHome, _Base):
    def test_cli_budget_unlimited(self) -> None:
        code, out = self._run_cli("autonomy", "budget")
        self.assertEqual(code, 0)
        self.assertIn("model budget", out)
        self.assertIn("unlimited", out)

    def test_cli_budget_with_cap_and_ledger(self) -> None:
        os.environ["NM_AUTONOMY_DAILY_MODEL_CALLS"] = "5"
        try:
            # seed today's ledger directly in the CLI's home
            settings = load_settings(
                overrides={"home": self._cli_tmp.name,
                           "partner.platforms": "local",
                           "chat.local_enabled": "true"})
            ctx = build_context(settings, with_executor=False,
                                with_tools=False)
            try:
                from nomorals.agents.cognition import ModelBudget

                ModelBudget(ctx).record(4, 900)
            finally:
                ctx.close()
            code, out = self._run_cli("autonomy", "budget")
            self.assertEqual(code, 0)
            self.assertIn("cap 5 calls/day", out)
            self.assertIn("used 4", out)
            self.assertIn("remaining 1", out)
        finally:
            os.environ.pop("NM_AUTONOMY_DAILY_MODEL_CALLS", None)


# ── 3. portfolio risk scoring ─────────────────────────────────────────────

class EvRankingTest(_Base):
    def _mc(self):
        from nomorals.agents.mission import MissionControl

        return MissionControl(self.context)

    def test_priority_dominates_size(self) -> None:
        a = self.gs.create("High prio small", plan=["x"], priority=9)
        b = self.gs.create("Low prio huge",
                           plan=[f"s{i}" for i in range(12)], priority=1)
        scores = self._mc().ev_scores([a.id, b.id])
        self.assertGreater(scores[a.id]["expected_value"],
                           scores[b.id]["expected_value"])

    def test_blocked_goal_loses_value(self) -> None:
        a = self.gs.create("Independent", plan=["x"], priority=9)
        base = self.gs.create("Base", plan=["y"], priority=9)
        b = self.gs.create("Waiting", plan=["z"], priority=9)
        self.gs.add_dependency(b.id, base.id)
        scores = self._mc().ev_scores([a.id, b.id, base.id])
        self.assertGreater(scores[a.id]["expected_value"],
                           scores[b.id]["expected_value"])
        self.assertEqual(scores[b.id]["unmet_dependency_depth"], 1)
        self.assertEqual(scores[base.id]["unmet_dependency_depth"], 0)
        self.assertEqual(scores[a.id]["risk"], "low")

    def test_chain_depth_is_high_risk(self) -> None:
        x = self.gs.create("Chain base", plan=["x"])
        a = self.gs.create("Chain mid", plan=["y"])
        b = self.gs.create("Chain top", plan=["z"])
        self.gs.add_dependency(a.id, x.id)
        self.gs.add_dependency(b.id, a.id)
        scores = self._mc().ev_scores([b.id])
        self.assertEqual(scores[b.id]["unmet_dependency_depth"], 2)
        self.assertEqual(scores[b.id]["risk"], "high")
        # finishing the base shortens the chain by one hop
        g = x
        self._finish_goal(g)
        scores = self._mc().ev_scores([b.id])
        self.assertEqual(scores[b.id]["unmet_dependency_depth"], 1)

    def test_heals_penalty(self) -> None:
        a = self.gs.create("Steady", plan=["x"], priority=5)
        b = self.gs.create("Wounded", plan=["x"], priority=5)
        self.context.db.execute(
            "UPDATE agent_goals SET heals=2 WHERE id=?", (b.id,))
        scores = self._mc().ev_scores([a.id, b.id])
        self.assertGreater(scores[a.id]["expected_value"],
                           scores[b.id]["expected_value"])
        self.assertEqual(scores[b.id]["risk"], "high")
        self.assertAlmostEqual(scores[b.id]["factors"]["reliability"],
                               1.0 / 2.0, places=3)

    def test_size_penalty(self) -> None:
        a = self.gs.create("Tiny", plan=["x"], priority=5)
        b = self.gs.create("Sprawling",
                           plan=[f"s{i}" for i in range(20)], priority=5)
        scores = self._mc().ev_scores([a.id, b.id])
        self.assertGreater(scores[a.id]["expected_value"],
                           scores[b.id]["expected_value"])
        self.assertEqual(scores[a.id]["pending_steps"], 1)
        self.assertEqual(scores[b.id]["pending_steps"], 20)

    def test_plan_ranking_sorted_with_factors(self) -> None:
        self.gs.create("P1", plan=["x"], priority=1)
        self.gs.create("P9", plan=["x"], priority=9)
        done = self.gs.create("Finished", plan=["x"], priority=99)
        self._finish_goal(done)
        plan = self._mc().plan()
        ranked = plan["ranking"]
        self.assertIn("ranking", plan)
        ids = [r["id"] for r in ranked]
        self.assertNotIn(done.id, ids, "done goals are not ranked")
        evs = [r["expected_value"] for r in ranked]
        self.assertEqual(evs, sorted(evs, reverse=True))
        top = ranked[0]
        for key in ("id", "title", "expected_value", "risk", "priority",
                    "unmet_dependency_depth", "heals", "pending_steps",
                    "factors"):
            self.assertIn(key, top)
        for key in ("base", "depth", "reliability", "size"):
            self.assertIn(key, top["factors"])

    def test_ev_scores_ignores_done_and_unknown(self) -> None:
        g = self.gs.create("Ghost check", plan=["x"])
        self._finish_goal(g)
        scores = self._mc().ev_scores([g.id, "goal-does-not-exist"])
        self.assertEqual(scores, {})

    def test_loop_orders_by_ev(self) -> None:
        # EV flips raw priority: the small P0 goal (EV ~40.9) outranks the
        # P5 goal with 20 pending steps (EV ~33.3)
        small = self.gs.create("Small", plan=["work-small"], priority=0)
        huge = self.gs.create("Huge",
                              plan=[f"work-huge-{i}" for i in range(20)],
                              priority=5)
        # both active: EV flips raw priority (small 40.9 > huge 33.3)
        scores = self._mc().ev_scores([small.id, huge.id])
        self.assertGreater(scores[small.id]["expected_value"],
                           scores[huge.id]["expected_value"])
        order: list[str] = []
        self._loop().tick(executor=lambda d: order.append(d) or "ok: done")
        self.assertTrue(order, "tick advanced nothing")
        self.assertEqual(order[0], "work-small", "tick did not work EV order")

    def _loop(self):
        from nomorals.agents.cognition import CognitiveLoop

        return CognitiveLoop(self.context)


class EvRankingCliTest(_CliHome, _Base):
    def test_mission_cli_prints_ranking(self) -> None:
        code, out = self._run_cli("goal", "create", "Ranked",
                                  "--priority", "3")
        self.assertEqual(code, 0)
        code, out = self._run_cli("mission", "plan")
        self.assertEqual(code, 0)
        self.assertIn("ranking (expected value):", out)
        self.assertIn("EV", out)
        self.assertIn("Ranked", out)


class ConfigWave64Test(unittest.TestCase):
    def test_defaults(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-wave64-cfg-")
        try:
            s = load_settings(overrides={"home": tmp.name})
            self.assertEqual(s.autonomy.daily_model_calls, 0)
            self.assertTrue(s.autonomy.reflect_on_completion)
        finally:
            tmp.cleanup()

    def test_budget_validation(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings(overrides={"autonomy.daily_model_calls": "-1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
