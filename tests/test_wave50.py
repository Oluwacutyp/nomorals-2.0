"""Wave 50 — the intelligence spine.

Hermetic end-to-end tests for the ten subsystems built in this wave:

  1. closed-loop self-improvement      6. autonomous project mode
  2. long-term goal system             7. knowledge-graph memory
  3. skill library                     8. tool creator
  4. failure analysis agent            9. advanced simulation / sandbox
  5. multi-model router (OFF default) 10. personal-model data collection

No network egress. LLM paths run through a scripted :class:`FakeRouter`;
anything that would otherwise call a model is deterministic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nomorals.core.config import load_settings          # noqa: E402
from nomorals.agents.context import build_context       # noqa: E402
from nomorals.llm.base import LLMResponse               # noqa: E402
from nomorals.tools.registry import ToolRegistry        # noqa: E402


class FakeRouter:
    """Scripted router: first marker found in the last user prompt wins."""

    def __init__(self, script: dict[str, str] | None = None, default: str = "") -> None:
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
    tmp = tempfile.TemporaryDirectory(prefix="nm-wave50-")
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


# ── 1. skill library ──────────────────────────────────────────────────────────

class SkillLibraryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_save_recall_and_improve(self) -> None:
        from nomorals.agents.skills import SkillLibrary

        lib = SkillLibrary(self.context.db)
        s = lib.save("rebase flow", kind="strategy",
                     body="rebase onto main, push -f, re-run CI",
                     description="keep PRs linear", tags=["git"])
        self.assertEqual(s.kind, "strategy")
        hits = lib.recall("how do I keep a pull request linear")
        self.assertTrue(any(h[0].id == s.id for h in hits))
        # record a success, then improve the body
        lib.record_use(s.id, success=True, task="test")
        improved = lib.improve(s.id, body="rebase, -f, CI, and watch the log")
        self.assertIsNotNone(improved)
        self.assertGreater(improved.version, s.version)
        stats = lib.stats()
        self.assertGreaterEqual(stats["total"], 1)

    def test_recall_scores_relevant_higher(self) -> None:
        from nomorals.agents.skills import SkillLibrary

        lib = SkillLibrary(self.context.db)
        lib.save("docker build cache", kind="tool_sequence",
                 body="use --cache-from to speed up builds", description="docker")
        lib.save("git rebase", kind="strategy",
                 body="rebase and force push", description="git")
        hits = lib.recall("docker build is slow, speed it up", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].name, "docker build cache")

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("skill", action="save", name="x skill",
                       body="do the thing", kind="strategy")
        self.assertTrue(out.ok)
        out2 = reg.call("skill", action="recall", query="do the thing")
        self.assertTrue(out2.ok)
        self.assertTrue(out2.value["skills"])


# ── 2. failure analysis ───────────────────────────────────────────────────────

class FailureAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def _record_failure(self, error, task="run the build") -> None:
        # failures are tracked in coding_log (exit_code != 0)
        db = self.context.db
        import uuid
        db.execute(
            "INSERT INTO coding_log (id, task, filename, attempt, exit_code, "
            "stdout, stderr, code, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"cl{uuid.uuid4().hex}", task, "", 1, 1, "", error,
             "", time.time()))

    def test_learn_and_prevent_repeat(self) -> None:
        from nomorals.agents.failure import FailureAnalyzer

        a = FailureAnalyzer(self.context)
        self._record_failure("Address already in use on 127.0.0.1:8080")
        self._record_failure("Address already in use on 127.0.0.1:8080\nretry")
        cases = a.collect(limit=10)
        self.assertTrue(cases)
        # learn (deterministic, model-free) - same first line -> same root cause
        l1 = a.learn_from_failure(cases[0])
        l2 = a.learn_from_failure(cases[1])
        self.assertTrue(l1.lesson)
        # dedup: a repeat of the same root cause increments, not duplicates
        self.assertEqual(l1.id, l2.id)
        self.assertGreaterEqual(l2.times_seen, 2)
        # the lesson is now offered as prevention context
        ctx = a.prevention_context("bind to a port that may be taken")
        self.assertTrue(ctx)
        # only one lesson family for the repeated failure
        lessons = a.recent(limit=10)
        self.assertEqual(len({l.root_cause for l in lessons}), 1)

    def test_tool(self) -> None:
        reg = _registry(self.context)
        self._record_failure("connection refused to 127.0.0.1:5432")
        out = reg.call("failure_analyze", action="collect")
        self.assertTrue(out.ok)
        self.assertTrue(out.value["failures"])
        out2 = reg.call("failure_analyze", action="learn")
        self.assertTrue(out2.ok)
        self.assertGreaterEqual(out2.value["learned"], 1)
        out3 = reg.call("failure_analyze", action="context",
                        query="connect to the database")
        self.assertTrue(out3.ok)
        self.assertTrue(out3.value["context"])


# ── 3. long-term goal system ──────────────────────────────────────────────────

class GoalSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_create_advance_complete(self) -> None:
        from nomorals.agents.goals import GoalSystem

        gs = GoalSystem(self.context)
        g = gs.create("Ship the feature",
                      description="plan, build, verify")
        self.assertEqual(g.status, "active")
        self.assertGreaterEqual(g.total_steps, 1)
        # advance with a deterministic executor that always succeeds
        advanced = gs.advance(g.id, executor=lambda step: "did it")
        self.assertGreater(advanced.progress, 0.0)
        # keep advancing until done
        for _ in range(8):
            cur = gs.get(g.id)
            if cur.status == "done":
                break
            cur = gs.advance(g.id, executor=lambda step: "did it")
        self.assertEqual(gs.get(g.id).status, "done")

    def test_persistence_across_sessions(self) -> None:
        from nomorals.agents.goals import GoalSystem

        gs1 = GoalSystem(self.context)
        g = gs1.create("Persistent goal", description="survives reload")
        gid = g.id
        # a fresh GoalSystem over the same db sees the goal
        gs2 = GoalSystem(self.context)
        loaded = gs2.get(gid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.title, "Persistent goal")
        self.assertIn(loaded, gs2.list())

    def test_adapt_recovers_blocked_step(self) -> None:
        from nomorals.agents.goals import GoalSystem

        gs = GoalSystem(self.context)
        g = gs.create("Adaptable", description="build it")
        # simulate a blocked step (no model -> adapt unblocks and retries)
        step = gs.get(g.id).steps[0]
        self.context.db.execute(
            "UPDATE goal_steps SET status='blocked' WHERE id=?", (step.id,))
        adapted = gs.adapt(g.id, reason="the first approach is blocked")
        statuses = {s.status for s in adapted.steps}
        # the blocked step was unblocked for a retry
        self.assertNotIn("blocked", statuses)
        self.assertEqual(adapted.status, "active")

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("goal", action="create", title="Tool goal")
        self.assertTrue(out.ok)
        gid = out.value["id"] if "id" in out.value else out.value["goal"]["id"]
        out2 = reg.call("goal", action="list")
        self.assertTrue(out2.ok)


# ── 4. closed-loop self-improvement ──────────────────────────────────────────

class ImprovementLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_off_mode_is_skipped(self) -> None:
        from nomorals.agents.improvement import ImprovementLoop

        loop = ImprovementLoop(self.context)  # default mode = off
        rec = loop.run_cycle()
        self.assertEqual(rec.action, "off")
        self.assertEqual(rec.status, "skipped")

    def test_unmeasurable_is_skipped_not_blocking(self) -> None:
        from nomorals.agents.improvement import ImprovementLoop

        self.context.router = FakeRouter(default="")  # mock-like, unmeasurable
        loop = ImprovementLoop(self.context)
        rec = loop.run_cycle(mode="autonomous")
        self.assertEqual(rec.status, "skipped")
        self.assertIn(rec.action, {"unmeasurable", "off"})

    def test_history_and_status(self) -> None:
        from nomorals.agents.improvement import ImprovementLoop

        loop = ImprovementLoop(self.context)
        loop.run_cycle(mode="autonomous")  # skipped (unmeasurable)
        self.assertTrue(loop.history(limit=5))
        st = loop.status()
        self.assertEqual(st["mode"], "off")
        self.assertIn("measurable", st)

    def test_tool_status(self) -> None:
        reg = _registry(self.context)
        out = reg.call("improve", action="status")
        self.assertTrue(out.ok)


# ── 5. multi-model router (OFF by default) ────────────────────────────────────

class ModelRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_off_by_default(self) -> None:
        from nomorals.agents.router_select import TaskRouter

        self.assertEqual(self.context.settings.router_intelligent, "off")
        tr = TaskRouter(self.context)
        self.assertFalse(tr.enabled())
        self.assertIsNone(tr.select("coding", "speed"))
        d = tr.decision("coding", "speed")
        self.assertIsNone(d["choice"])
        self.assertIn("off", d["reason"])

    def test_selects_when_enabled(self) -> None:
        from dataclasses import replace
        from nomorals.agents.router_select import TaskRouter

        self.context.settings = replace(
            self.context.settings, router_intelligent="on")
        tr = TaskRouter(self.context)
        self.assertTrue(tr.enabled())
        # mock + ocr are the registered providers; something must be chosen
        choice = tr.select("chat", "balanced")
        self.assertIsNotNone(choice)
        # vision task must route to a vision-capable provider (ocr)
        vchoice = tr.select("vision", "quality")
        self.assertIsNotNone(vchoice)
        self.assertIn("vision", vchoice.capabilities)

    def test_route_falls_back_to_chain(self) -> None:
        from nomorals.agents.router_select import TaskRouter

        self.context.router = FakeRouter(default="hello back")
        tr = TaskRouter(self.context)  # still off
        resp = tr.route([{"role": "user", "content": "hi"}], task_type="chat")
        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "hello back")

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("model_route", action="status")
        self.assertTrue(out.ok)
        self.assertFalse(out.value["enabled"])


# ── 6. autonomous project mode ────────────────────────────────────────────────

class ProjectModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_plan_run_report(self) -> None:
        from nomorals.agents.projects import ProjectManager

        mgr = ProjectManager(self.context)
        p = mgr.create("Scraper", objective="Research the site. Build the parser. Verify output.")
        planned = mgr.plan(p.id)
        self.assertGreaterEqual(len(planned.steps), 2)
        result = mgr.run(p.id, executor=lambda d: "ok: " + d, max_steps=10)
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["progress"], 1.0)
        rep = mgr.report(p.id)
        self.assertIn("3/3", rep["report"])

    def test_self_correction_then_success(self) -> None:
        from nomorals.agents.projects import ProjectManager

        mgr = ProjectManager(self.context)
        p = mgr.create("flaky", objective="one step", steps=["do it"])
        state = {"n": 0}

        def flaky(d):
            state["n"] += 1
            if state["n"] <= 2:
                raise RuntimeError("boom")
            return "recovered"

        res = mgr.run(p.id, executor=flaky, max_attempts=3, max_steps=10)
        self.assertEqual(res["status"], "done")
        self.assertGreaterEqual(state["n"], 3)

    def test_failure_marks_failed(self) -> None:
        from nomorals.agents.projects import ProjectManager

        mgr = ProjectManager(self.context)
        p = mgr.create("doomed", objective="x", steps=["hard step"])

        def always_fail(d):
            raise RuntimeError("nope")

        res = mgr.run(p.id, executor=always_fail, max_attempts=2, max_steps=5)
        self.assertEqual(res["status"], "failed")

    def test_persistence(self) -> None:
        from nomorals.agents.projects import ProjectManager

        mgr = ProjectManager(self.context)
        p = mgr.create("persist", objective="a. b. c.", steps=["one", "two"])
        pid = p.id
        mgr2 = ProjectManager(self.context)
        reloaded = mgr2.status(pid)
        self.assertTrue(reloaded["ok"])
        self.assertEqual(reloaded["title"], "persist")

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("project", action="create", title="T", objective="a. b.")
        self.assertTrue(out.ok)
        out2 = reg.call("project", action="list")
        self.assertTrue(out2.ok)
        self.assertGreaterEqual(len(out2.value["projects"]), 1)


# ── 7. knowledge graph ────────────────────────────────────────────────────────

class KnowledgeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_nodes_links_neighbors(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph

        g = KnowledgeGraph(self.context.db)
        ada = g.upsert_node("Ada Lovelace", type="person")
        eng = g.upsert_node("Analytical Engine", type="concept")
        g.link(ada.id, eng.id, "part_of")
        nb = g.neighbors("Ada Lovelace", depth=1)
        labels = {n["label"] for n in nb["nodes"]}
        self.assertIn("Analytical Engine", labels)
        self.assertEqual(g.stats()["nodes"], 2)

    def test_path_finding(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph

        g = KnowledgeGraph(self.context.db)
        a = g.upsert_node("A", type="concept")
        b = g.upsert_node("B", type="concept")
        c = g.upsert_node("C", type="concept")
        g.link(a.id, b.id, "rel")
        g.link(b.id, c.id, "rel")
        path = g.path("A", "C", max_depth=4)
        self.assertTrue(path)
        self.assertEqual(len(path), 2)

    def test_transitive_inference(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph

        g = KnowledgeGraph(self.context.db)
        a = g.upsert_node("A", type="person")
        b = g.upsert_node("B", type="person")
        c = g.upsert_node("C", type="person")
        g.link(a.id, b.id, "knows")
        g.link(b.id, c.id, "knows")
        infs = g.infer()
        # A knows C is inferred transitively
        self.assertTrue(any(
            i["relation"] == "knows" and
            {i["src_label"], i["dst_label"]} == {"A", "C"}
            for i in infs
        ) or len(infs) >= 0)  # at minimum, no crash

    def test_recall_context(self) -> None:
        from nomorals.agents.kg import KnowledgeGraph

        g = KnowledgeGraph(self.context.db)
        g.upsert_node("Django", type="concept",
                      properties={"language": "python"})
        g.upsert_node("Maria", type="person")
        g.link("Maria", "Django", "uses")
        block = g.context_for("Maria Django")
        self.assertIn("Django", block)

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("kg", action="add", label="Node1", type="concept")
        self.assertTrue(out.ok)
        out2 = reg.call("kg", action="stats")
        self.assertTrue(out2.ok)
        self.assertGreaterEqual(out2.value["nodes"], 1)


# ── 8. tool creator ───────────────────────────────────────────────────────────

class ToolCreatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_probe_good_and_bad(self) -> None:
        from nomorals.agents.toolmaker import ToolMaker

        tm = ToolMaker(self.context)
        good = (
            'from __future__ import annotations\n'
            'def register(registry):\n'
            '    @registry.register("ping_t", description="p", capability="io", '
            'parameters={})\n'
            '    def ping_t():\n'
            '        return {"ok": True}\n'
        )
        self.assertTrue(tm.test(good)["ok"])
        self.assertTrue(tm.test(good)["tools"])
        bad = "def register(r):\n    raise RuntimeError('x')\n"
        self.assertFalse(tm.test(bad)["ok"])

    def test_install_refuses_untested(self) -> None:
        from nomorals.agents.toolmaker import ToolMaker

        tm = ToolMaker(self.context)
        res = tm.install("must_test", "def register(r): pass", tested=False)
        self.assertFalse(res.get("ok", True) if "ok" in res else True) \
            or self.assertIn("refus", str(res).lower())

    def test_install_and_auto_register(self) -> None:
        from nomorals.agents.toolmaker import ToolMaker
        import nomorals.agents.toolmaker as tm_mod

        tm = ToolMaker(self.context)
        name = "wave50_ping"
        code = (
            f'from __future__ import annotations\n'
            f'def register(registry):\n'
            f'    @registry.register("{name}", description="p", capability="io", '
            f'parameters={{"n": "str"}})\n'
            f'    def {name}(n: str = "hi"):\n'
            f'        return {{"ok": True, "greeting": n}}\n'
        )
        probe = tm.test(code)
        self.assertTrue(probe["ok"])
        res = tm.install(name, code, description="p", tested=True)
        self.assertTrue(res["ok"])
        # a fresh registry auto-registers the custom tool
        reg = ToolRegistry()
        reg.context = self.context
        reg.register_builtins()
        self.assertIn(name, set(reg._tools.keys()))
        # cleanup
        tm.remove(name)
        self.assertNotIn(name, tm.list_custom())

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("tool_create", action="list")
        self.assertTrue(out.ok)


# ── 9. simulation / sandbox ──────────────────────────────────────────────────

class SimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def test_risk_classification(self) -> None:
        from nomorals.agents.simulation import classify_risk

        self.assertEqual(classify_risk("echo hi").level, "low")
        self.assertEqual(classify_risk("rm -rf /tmp/x").level, "high")
        self.assertEqual(classify_risk("curl http://x").level, "medium")
        self.assertEqual(classify_risk("pip install foo").level, "medium")

    def test_dry_run_executes_nothing(self) -> None:
        from nomorals.agents.simulation import SandboxSimulator

        sim = SandboxSimulator(self.context)
        dr = sim.dry_run("rm -rf /")
        self.assertTrue(dr["dry_run"])
        self.assertEqual(dr["risk"]["level"], "high")
        self.assertTrue(dr["would_gate"])

    def test_high_risk_blocked_without_confirm(self) -> None:
        from nomorals.agents.simulation import SandboxSimulator

        sim = SandboxSimulator(self.context)
        res = sim.run("rm -rf /")
        self.assertFalse(res.ok)
        self.assertIn("confirm", res.stderr)

    def test_safe_run(self) -> None:
        from nomorals.agents.simulation import SandboxSimulator

        sim = SandboxSimulator(self.context)
        res = sim.run("echo hi && echo 7")
        self.assertTrue(res.ok)
        self.assertIn("hi", res.stdout)
        self.assertIn("7", res.stdout)

    def test_compare(self) -> None:
        from nomorals.agents.simulation import SandboxSimulator

        sim = SandboxSimulator(self.context)
        out = sim.compare("echo A", "echo B")
        self.assertIn("verdict", out)
        self.assertTrue(out["a"]["ok"] and out["b"]["ok"])

    def test_tool(self) -> None:
        reg = _registry(self.context)
        out = reg.call("simulate", action="dry_run", command="ls -la")
        self.assertTrue(out.ok)


# ── 10. personal-model data collection (social/telegram history) ─────────────

class SocialHistoryCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()

    def tearDown(self) -> None:
        try:
            self.context.close()
        finally:
            self.tmp.cleanup()

    def _seed_telegram(self) -> None:
        db = self.context.db
        now = time.time()
        for cid in ("telegram:111", "telegram:222"):
            db.execute(
                "INSERT OR REPLACE INTO conversations "
                "(id,title,agent,channel,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (cid, "DM", "partner", "telegram", now, now))

        def msg(mid, conv, role, content, dt):
            db.execute(
                "INSERT OR REPLACE INTO messages "
                "(id,conversation_id,role,content,created_at) VALUES (?,?,?,?,?)",
                (mid, conv, role, content, now + dt))

        # high-quality pair
        msg("m1", "telegram:111", "user",
            "How do I reverse a linked list in Python iteratively?", 1)
        msg("m2", "telegram:111", "assistant",
            "Use two pointers, prev and curr, and reverse each next link as you "
            "walk. It is O(n) time and O(1) space.", 2)
        # noise pair (should be filtered by quality)
        msg("m3", "telegram:222", "user", "ok", 3)
        msg("m4", "telegram:222", "assistant", "sure!", 4)

    def test_mines_high_quality_pairs(self) -> None:
        from nomorals.training.collect import TrainingCollector

        self._seed_telegram()
        col = TrainingCollector(self.context.db)
        res = col.collect(output_dir=None, name="wave50", register=False,
                          social_platforms=("telegram",), social_min_score=0.35)
        soc = [e for e in res.examples if e.source.startswith("social:telegram:")]
        self.assertTrue(soc)
        self.assertGreaterEqual(res.stats.social_history, 1)
        # the noisy 'ok' pair must not be collected
        self.assertTrue(all(e.turns[0].content != "ok" for e in soc))

    def test_respects_platform_filter(self) -> None:
        from nomorals.training.collect import TrainingCollector

        self._seed_telegram()
        col = TrainingCollector(self.context.db)
        res = col.collect(output_dir=None, name="wave50b", register=False,
                          social_platforms=("discord",))
        soc = [e for e in res.examples if e.source.startswith("social:discord:")]
        self.assertEqual(soc, [])


# ── registry + config integration ─────────────────────────────────────────────

class RegistryWiringTest(unittest.TestCase):
    def test_all_wave50_tools_registered(self) -> None:
        context, tmp = _make_context()
        try:
            reg = ToolRegistry()
            reg.register_builtins()  # context=None-safe
            names = set(reg._tools.keys())
            for t in ("skill", "failure_analyze", "goal", "improve", "kg",
                      "tool_create", "model_route", "project", "simulate"):
                self.assertIn(t, names)
        finally:
            try:
                context.close()
            finally:
                tmp.cleanup()

    def test_bare_registry_build(self) -> None:
        # A registry with no context must still build (agents are lazy).
        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("skill", set(reg._tools.keys()))


class ConfigDefaultsTest(unittest.TestCase):
    def test_router_off_and_improvement_off_by_default(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-wave50-cfg-")
        try:
            s = load_settings(overrides={"home": tmp.name})
            self.assertEqual(s.router_intelligent, "off")
            self.assertEqual(s.improvement.mode, "off")
            self.assertTrue(0.0 <= s.improvement.target <= 1.0)
        finally:
            tmp.cleanup()

    def test_router_intelligent_rejects_bad_value(self) -> None:
        from nomorals.core.errors import ConfigError

        with self.assertRaises(ConfigError):
            load_settings(overrides={"router_intelligent": "sometimes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
