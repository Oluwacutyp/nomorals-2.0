"""Wave 68 — real reasoning everywhere, missions that actually run and
answer queries, a structuring sub-agent, a clean proxy file, /list, and
the path to a local fine-tuned primary brain.

Hermetic by design:

* scripted routers stand in for every model call (no network, no keys)
* mission steps execute through a faked ``_execute_step`` and a
  pre-seeded persisted plan (no orchestrator model call)
* the CLI end-to-end cases run in a subprocess with a private NM_HOME

Covered:
* F1 — ReasoningAgent: advise degrades to "" when the model is down,
       returns real advice with a model, plan_tools parses strict JSON,
       drops unknown tools, degrades on garbage
* P1 — Devon plan ladder: llm plan; unparseable → strict retry → success;
       unparseable (twice) → reasoning engine plans → "reasoning";
       model down → heuristic with "model-error"; no router → "no-router";
       plan_error always honest on the heuristic path
* F2 — BriefAgent: short asks skip; heuristic floor without a model;
       strict-JSON model path merged over the floor; as_goal rendering
* F5 — Mission.progress / store.detail / latest_checkpoints / set_status
       (pause, cancel, invalid) — the queryable, inspectable contract
* F5 — runner honours cross-process pause (parks, waits, continues) and
       cancel between steps; max_iterations=0 = unlimited
* F3 — ProxyLab.active_file: clean file, dead/stale dropped,
       fastest first; empty pool leaves no stale copy; proxy_file tool
       registered and callable
* F4 — /list + /commands + /menu: categorized catalog, group filter,
       unknown-group hint; nm commands CLI mirror
* P2 — nm models --promote-local writes the .env contract (provider,
       model, auto-start) that build_context picks up next session
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.test_partner_runtime import FakeRouter, _make_context

from nomorals.agents.brief import BriefAgent, MissionBrief, should_brief
from nomorals.agents.context import build_context
from nomorals.agents.devon import DevonAgent
from nomorals.agents.reasoning import ReasoningAgent
from nomorals.core.config import Settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.missions import (Mission, MissionRunner, MissionStatus,
                               MissionStore)
from nomorals.missions.runner import StepOutcome

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── scripted routers ─────────────────────────────────────────────────────────


class LadderRouter(FakeRouter):
    """Scripted planner: garbage first, then (optionally) a real plan."""

    def __init__(self, plan_after_garbage: bool = False,
                 reasoning_plan: list[dict] | None = None) -> None:
        super().__init__(replies=[])
        self.plan_after_garbage = plan_after_garbage
        self.reasoning_plan = reasoning_plan
        self.planner_calls = 0

    def chat(self, messages, params=None, **kw):
        self.calls.append(list(messages))
        blob = " ".join(m.content for m in messages)
        if "autonomous engineering & debug agent" in blob:
            self.planner_calls += 1
            if self.planner_calls == 1:
                return LLMResponse(text="sorry, I can only plan in words",
                                   model="fake-planner")
            if self.plan_after_garbage:
                return LLMResponse(
                    text=json.dumps({"steps": [
                        {"tool": "logs_tail", "args": {"lines": "20"},
                         "why": "see the recent errors"}]}),
                    model="fake-planner")
            return LLMResponse(text="still no json", model="fake-planner")
        if "Plan the tool calls that answer this task" in blob:
            if self.reasoning_plan is not None:
                return LLMResponse(
                    text=json.dumps({"steps": self.reasoning_plan}),
                    model="fake-reasoning")
            return LLMResponse(text="nope, prose only", model="fake-reasoning")
        return super().chat(messages, params, **kw)


class BrokenRouter:
    """A model that is completely down — every call raises."""

    def chat(self, messages, params=None, **kw) -> LLMResponse:
        raise RuntimeError("connection refused (simulated vendor outage)")


def _context_with(router, *, with_tools: bool = False):
    context, tmp = _make_context()
    context.router = router
    return context, tmp


# ── F1: the permanent reasoning agent ────────────────────────────────────────


class ReasoningAgentTest(unittest.TestCase):
    def setUp(self):
        self.context, self.tmp = _context_with(FakeRouter(
            replies=["Use the faster tool first, then verify with the log."]))
        self.addCleanup(self.tmp.cleanup)

    def test_advise_returns_real_advice(self) -> None:
        agent = ReasoningAgent(self.context)
        advice = agent.advise("step 'execute' failed with timeout",
                              focus="how the next attempt should differ")
        self.assertIn("faster tool", advice)

    def test_advise_degrades_to_empty_when_model_down(self) -> None:
        self.context.router = BrokenRouter()
        agent = ReasoningAgent(self.context)
        self.assertEqual(agent.advise("deciding now: X"), "")

    def test_advise_empty_on_blank(self) -> None:
        self.assertEqual(ReasoningAgent(self.context).advise("   "), "")

    def test_plan_tools_parses_strict_json(self) -> None:
        self.context.router = FakeRouter(replies=[
            "thinking out loud… " + json.dumps({"steps": [
                {"tool": "log_tail", "args": {"lines": "50"},
                 "why": "recent errors"},
                {"tool": "web_search", "args": {"query": "llama 3 quant"},
                 "why": "research"},
                {"tool": "not_a_real_tool", "args": {}, "why": "hallucinated"},
            ]})
        ])
        steps = ReasoningAgent(self.context).plan_tools(
            "debug the crash", [("log_tail", "x"), ("web_search", "y")])
        self.assertEqual([s["tool"] for s in steps], ["log_tail", "web_search"])
        self.assertEqual(steps[0]["args"], {"lines": "50"})

    def test_plan_tools_degrades_on_garbage(self) -> None:
        self.context.router = FakeRouter(replies=["no json here, sorry"])
        self.assertEqual(ReasoningAgent(self.context).plan_tools(
            "task", [("log_tail", "x")]), [])

    def test_plan_tools_empty_catalog_is_empty(self) -> None:
        self.assertEqual(ReasoningAgent(self.context).plan_tools("task", []),
                         [])


# ── P1: the devon plan ladder ────────────────────────────────────────────────


class DevonPlanLadderTest(unittest.TestCase):
    def _run(self, router, task="check the recent log errors"):
        context, tmp = _context_with(router)
        self.addCleanup(tmp.cleanup)
        return DevonAgent(context).run(task)

    def test_llm_plan_is_used_when_json_is_valid(self) -> None:
        result = self._run(LadderRouter(plan_after_garbage=True))
        self.assertEqual(result.planned_by, "llm")
        self.assertEqual(result.plan_error, "")
        self.assertEqual(result.steps[0].tool, "logs_tail")

    def test_reasoning_engine_plans_when_model_wont_parse(self) -> None:
        result = self._run(LadderRouter(reasoning_plan=[
            {"tool": "logs_tail", "args": {"lines": "30"},
             "why": "reasoned: the errors are in the log"}]))
        self.assertEqual(result.planned_by, "reasoning")
        self.assertEqual(result.model, "reasoning-engine")
        self.assertIn("unparseable", result.plan_error)

    def test_heuristic_is_last_resort_with_an_honest_reason(self) -> None:
        result = self._run(LadderRouter())  # garbage, garbage, garbage
        self.assertEqual(result.planned_by, "heuristic")
        self.assertIn("unparseable (twice)", result.plan_error)

    def test_model_down_degrades_with_model_error_reason(self) -> None:
        result = self._run(BrokenRouter())
        self.assertEqual(result.planned_by, "heuristic")
        self.assertIn("model-error", result.plan_error)

    def test_no_router_degrades_with_no_router_reason(self) -> None:
        result = self._run(None)
        self.assertEqual(result.planned_by, "heuristic")
        self.assertIn("no-router", result.plan_error)
        # the honest reason is surfaced to the owner in the digest too
        self.assertIn("fallback plan", result.digest)
        self.assertIn("no-router", result.digest)


# ── F2: the structuring sub-agent ────────────────────────────────────────────


class BriefAgentTest(unittest.TestCase):
    LONG = ("Gather the latest HuggingFace dataset rankings, compare the top "
            "three against what we already store, and write a summary with "
            "license notes into the workspace")

    def test_short_ask_skips_structuring(self) -> None:
        context, tmp = _context_with(None)
        self.addCleanup(tmp.cleanup)
        brief = BriefAgent(context).refine("check the weather")
        self.assertEqual(brief.by, "skip")
        self.assertEqual(brief.objective, "check the weather")

    def test_should_brief_thresholds(self) -> None:
        self.assertFalse(should_brief("hello there"))
        self.assertTrue(should_brief(self.LONG))  # >= 80 chars
        self.assertTrue(should_brief(
            "gather the data, then compare it, and then write notes up"))

    def test_heuristic_floor_without_a_model(self) -> None:
        context, tmp = _context_with(None)
        self.addCleanup(tmp.cleanup)
        brief = BriefAgent(context).refine(self.LONG)
        self.assertEqual(brief.by, "heuristic")
        self.assertTrue(brief.objective)
        self.assertTrue(brief.steps)
        self.assertIn("MISSION:", brief.as_goal())
        self.assertIn("Done means:", brief.as_goal())

    def test_model_path_structures_and_merges(self) -> None:
        spec = {
            "objective": "Produce a license-annotated HF dataset comparison",
            "constraints": ["read-only access", "no downloads over 100MB"],
            "success_criteria": ["summary file exists in workspace",
                                 "each dataset lists its license"],
            "steps": ["fetch rankings", "compare top three",
                      "write the summary"],
            "persona": "",
            "tool_hints": ["web_fetch"],
        }
        context, tmp = _context_with(FakeRouter(
            replies=["Here you go: " + json.dumps(spec)]))
        self.addCleanup(tmp.cleanup)
        brief = BriefAgent(context).refine(self.LONG)
        self.assertEqual(brief.by, "model")
        self.assertEqual(brief.objective, spec["objective"])
        self.assertEqual(brief.steps[0], "fetch rankings")
        self.assertIn("license-annotated", brief.as_goal())

    def test_model_garbage_falls_back_to_heuristic(self) -> None:
        context, tmp = _context_with(FakeRouter(replies=["no json here"]))
        self.addCleanup(tmp.cleanup)
        brief = BriefAgent(context).refine(self.LONG)
        self.assertEqual(brief.by, "heuristic")
        self.assertTrue(brief.objective)

    def test_to_dict_round_trips(self) -> None:
        brief = MissionBrief(objective="x", constraints=["c"],
                             success_criteria=["s"], steps=["1", "2"],
                             persona="p", tool_hints=["t"], by="model")
        d = brief.to_dict()
        self.assertEqual(d["objective"], "x")
        self.assertEqual(d["by"], "model")


# ── F5: mission progress, detail, and operator status control ────────────────


def _plan_state(names: list[str]) -> list[dict]:
    return [{"name": n, "goal": f"do {n}", "role": "execution",
             "kind": "io", "depends_on": []} for n in names]


class MissionProgressControlTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w68-ms-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)
        self.mission = self.store.create_new("a then b then c then d")
        self.mission.state["plan"] = _plan_state(["a", "b", "c", "d"])
        self.store.save(self.mission)

    def tearDown(self):
        try:
            self.context.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def test_progress_derives_from_persisted_state(self) -> None:
        p = self.store.progress(self.mission.id)
        self.assertEqual(p["total_steps"], 4)
        self.assertEqual(p["steps_done"], 0)
        self.assertEqual(p["percent"], 0.0)
        self.assertEqual(p["current_step"], "a")

        self.mission.state["completed_steps"] = ["a", "b"]
        self.mission.iterations = 2
        self.store.save(self.mission)
        p = self.store.progress(self.mission.id)
        self.assertEqual(p["steps_done"], 2)
        self.assertEqual(p["percent"], 50.0)
        self.assertEqual(p["current_step"], "c")

    def test_detail_includes_checkpoints_and_progress(self) -> None:
        self.store.checkpoint(self.mission, label="after:a")
        detail = self.store.detail(self.mission.id)
        self.assertIn("progress", detail)
        self.assertEqual(detail["progress"]["total_steps"], 4)
        self.assertTrue(any(c["label"] == "after:a"
                            for c in detail["recent_checkpoints"]))

    def test_latest_checkpoints_orders_newest_first(self) -> None:
        self.store.checkpoint(self.mission, label="one")
        time.sleep(0.01)
        self.store.checkpoint(self.mission, label="two")
        rows = self.store.latest_checkpoints(self.mission.id)
        self.assertEqual([r["label"] for r in rows][:2], ["two", "one"])

    def test_set_status_pause_and_resume(self) -> None:
        updated = self.store.set_status(self.mission.id, "paused",
                                        "paused by operator")
        self.assertEqual(updated.status, MissionStatus.PAUSED)
        self.assertEqual(self.store.get(self.mission.id).status,
                         MissionStatus.PAUSED)
        self.assertIn("pause_note", self.store.get(self.mission.id).state)

        updated = self.store.set_status(self.mission.id, "running")
        self.assertEqual(updated.status, MissionStatus.RUNNING)
        self.assertNotIn("pause_note", self.store.get(self.mission.id).state)

    def test_set_status_cancel_is_terminal(self) -> None:
        updated = self.store.set_status(self.mission.id, "cancelled",
                                        "user changed mind")
        self.assertEqual(updated.status, MissionStatus.CANCELLED)
        self.assertTrue(updated.finished_at > 0)

    def test_set_status_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            self.store.set_status(self.mission.id, "frozen")


# ── F5: the runner honours cross-process pause and cancel between steps ─────


class RunnerCrossProcessControlTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w68-run-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)

    def tearDown(self):
        try:
            self.context.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def _runner(self, names: list[str]):
        mission = self.store.create_new(
            "then".join(f" {n} " for n in names).strip())
        mission.state["plan"] = _plan_state(names)
        self.store.save(mission)
        runner = MissionRunner(self.context, store=self.store)

        def fake_execute(m, step):
            time.sleep(0.01)
            return StepOutcome(step=step.name, ok=True, seconds=0.01,
                               payload={"did": step.name})

        runner._execute_step = fake_execute  # type: ignore[method-assign]
        return mission, runner

    def test_unlimited_iterations_complete_every_step(self) -> None:
        names = [f"s{i}" for i in range(1, 11)]  # 10 steps
        mission, runner = self._runner(names)
        started = time.monotonic()
        result = runner.run(mission, max_iterations=0, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(mission.iterations, 10)
        self.assertEqual(sorted(mission.state["completed_steps"]),
                         sorted(names))
        self.assertLess(time.monotonic() - started, 30.0)

    def test_iteration_limit_still_stops(self) -> None:
        mission, runner = self._runner([f"s{i}" for i in range(1, 11)])
        result = runner.run(mission, max_iterations=3, reflect=False)
        self.assertEqual(mission.iterations, 3)
        self.assertNotEqual(result.status, MissionStatus.DONE)

    def test_pause_parks_the_runner_then_resumes(self) -> None:
        mission, runner = self._runner(["a", "b", "c", "d"])
        parked = threading.Event()
        seen_paused = threading.Event()

        def on_step(m, outcome):
            if outcome.step == "a":
                # the operator, from another "process", pauses the mission
                self.store.set_status(m.id, "paused", "operator pause")
                parked.set()

        def operator_resume():
            # the pause is real: it sticks until the operator lifts it.
            # The park loop checks at t≈0 and t≈2s; lift the pause at
            # t≈1.25s so the runner provably parks, then continues.
            self.assertTrue(parked.wait(5.0))
            time.sleep(1.25)
            self.assertEqual(
                self.store.get(mission.id).status, MissionStatus.PAUSED)
            seen_paused.set()
            self.store.set_status(mission.id, "running", "operator resume")

        threading.Thread(target=operator_resume, daemon=True).start()
        runner.on_step = on_step
        started = time.monotonic()
        result = runner.run(mission, max_iterations=10, reflect=False)
        elapsed = time.monotonic() - started
        # it parked (the row said PAUSED while it waited) and then
        # finished all four steps after the operator's resume
        self.assertTrue(seen_paused.is_set())
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual(mission.iterations, 4)
        self.assertGreaterEqual(elapsed, 1.5)  # it really waited it out

    def test_cancel_between_steps_stops_the_run(self) -> None:
        mission, runner = self._runner(["a", "b", "c"])

        def on_step(m, outcome):
            if outcome.step == "a":
                self.store.set_status(m.id, "cancelled", "operator cancel")

        runner.on_step = on_step
        result = runner.run(mission, max_iterations=10, reflect=False)
        self.assertEqual(result.status, MissionStatus.CANCELLED)
        self.assertEqual(mission.iterations, 1)  # b and c never ran
        self.assertIn("cancelled", result.error)

    def test_step_failure_gets_reasoning_advice_attempt(self) -> None:
        # model down → advise degrades to "" but the mission records the
        # failure and stops cleanly (the advice hook must never break it)
        self.context.router = BrokenRouter()
        mission, runner = self._runner(["a", "b"])

        def failing(m, step):
            return StepOutcome(step=step.name, ok=False,
                               detail="simulated step failure", seconds=0.01)

        runner._execute_step = failing  # type: ignore[method-assign]
        result = runner.run(mission, max_iterations=10, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertIn("simulated step failure",
                      mission.state.get("last_error", ""))


# ── F4: devon answers "where is my mission?" from persisted state ────────────


class DevonMissionProgressTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w68-devon-")
        self.context = build_context(Settings(home=self.home))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)
        self.agent = DevonAgent(self.context)

    def tearDown(self):
        try:
            self.context.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def test_progress_query_reports_real_state(self) -> None:
        mission = self.store.create_new("gather the dataset", name="mywork")
        mission.state["plan"] = _plan_state(["fetch", "compare", "write"])
        mission.state["completed_steps"] = ["fetch"]
        mission.iterations = 1
        mission.state["last_error"] = ""
        self.store.save(mission)
        self.store.checkpoint(mission, label="after:fetch")

        out = self.agent._mission_progress_report("mywork")
        self.assertIn("mission progress:", out)
        self.assertIn("mywork", out)
        self.assertIn("1/3 steps", out)
        self.assertIn("(33%)", out)
        self.assertIn("current: compare", out)
        self.assertIn("after:fetch", out)

    def test_progress_word_routes_to_report_not_new_mission(self) -> None:
        mission = self.store.create_new("do the thing", name="job1")
        mission.state["plan"] = _plan_state(["a", "b"])
        self.store.save(mission)
        before = len(self.store.list())
        out = self.agent._tool_mission(
            {"goal": "progress on job1", "name": "job1"})
        self.assertIn("mission progress:", out)
        self.assertNotIn("mission started", out)
        self.assertEqual(len(self.store.list()), before)  # nothing spawned

    def test_unknown_mission_lists_the_active_ones(self) -> None:
        self.store.create_new("do the thing", name="job1")
        out = self.agent._mission_progress_report("zzz-not-there")
        self.assertIn("no mission matching 'zzz-not-there'", out)
        self.assertIn("job1", out)


# ── F3: the clean proxy file ─────────────────────────────────────────────────


class ProxyFileTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w68-proxy-")
        from nomorals.tools.proxylab import Proxy, ProxyStore

        self.Proxy = Proxy
        self.store = ProxyStore(self.home)

    def _seed(self, proxies: list) -> None:
        self.store.working_json.write_text(
            json.dumps([p.to_dict() for p in proxies]), encoding="utf-8")

    def test_active_file_is_clean_and_fastest_first(self) -> None:
        now = time.time()
        fast = self.Proxy(host="1.1.1.1", port=80, alive=True,
                          tested_at=now, latency_ms=40)
        slow = self.Proxy(host="2.2.2.2", port=8080, alive=True,
                          tested_at=now, latency_ms=400, scheme="https")
        dead = self.Proxy(host="3.3.3.3", port=3128, alive=False,
                          tested_at=now, latency_ms=10)
        stale = self.Proxy(host="4.4.4.4", port=80, alive=True,
                           tested_at=now - 48 * 3600, latency_ms=10)
        self._seed([stale, dead, slow, fast])

        from nomorals.tools.proxylab import ProxyLab
        context, tmp = _make_context()
        self.addCleanup(tmp.cleanup)
        # point the lab at the seeded store's directory
        lab = ProxyLab(context)
        lab.store = self.store
        info = lab.active_file()
        self.assertTrue(info["written"])
        self.assertEqual(info["count"], 2)
        self.assertEqual(info["fastest"], fast.url)
        text = Path(info["path"]).read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertEqual(lines, [fast.url, slow.url])

    def test_active_file_empty_pool_writes_nothing(self) -> None:
        from nomorals.tools.proxylab import ProxyLab
        context, tmp = _make_context()
        self.addCleanup(tmp.cleanup)
        lab = ProxyLab(context)
        lab.store = self.store
        info = lab.active_file()
        self.assertFalse(info["written"])
        self.assertEqual(info["count"], 0)
        self.assertIn("note", info)
        self.assertFalse(Path(info["path"]).exists())

    def test_proxy_file_tool_registered_and_callable(self) -> None:
        context, tmp = _make_context()
        self.addCleanup(tmp.cleanup)
        # tools on, everything else minimal
        ctx2 = build_context(Settings(home=tmp.name), with_router=False,
                             with_memory=False, with_tools=True,
                             with_executor=False)
        ctx2.__enter__()
        self.addCleanup(ctx2.__exit__)
        # seed the pool in the directory THIS context's lab actually uses
        from nomorals.tools.proxylab import ProxyLab
        now = time.time()
        lab_store = ProxyLab(ctx2).store
        lab_store.working_json.write_text(
            json.dumps([self.Proxy(host="9.9.9.9", port=80, alive=True,
                                   tested_at=now,
                                   latency_ms=12).to_dict()]),
            encoding="utf-8")
        outcome = ctx2.tools.call("proxy_file")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.value["count"], 1)
        file_text = Path(outcome.value["path"]).read_text(encoding="utf-8")
        self.assertEqual(file_text.strip(), "http://9.9.9.9:80")


# ── F4: /list — every executable command, categorized ────────────────────────


class CommandCatalogTest(unittest.TestCase):
    def test_full_catalog_covers_the_system(self) -> None:
        from nomorals.social.chat.control import list_catalog

        out = list_catalog()
        for cmd in ("/devon", "/proxy", "/evolve", "/searchdeep",
                    "/think", "/code", "/list"):
            self.assertIn(cmd, out)
        for group in ("her day", "search & research",
                      "building for real", "memory & thinking",
                      "voice & vision", "tools & automation"):
            self.assertIn(group, out)

    def test_group_filter(self) -> None:
        from nomorals.social.chat.control import list_catalog

        out = list_catalog("building")
        self.assertIn("/code", out)
        self.assertIn("/devon", out)
        self.assertNotIn("/dns", out)  # another group stays out

    def test_unknown_group_lists_the_options(self) -> None:
        from nomorals.social.chat.control import list_catalog

        out = list_catalog("not-a-group")
        self.assertIn("available:", out)

    def test_aliases_parse(self) -> None:
        from nomorals.social.chat.control import CONTROL_COMMANDS, \
            parse_control

        for kind in ("list", "commands", "menu"):
            self.assertIn(kind, CONTROL_COMMANDS)
            cmd = parse_control(f"/{kind}")
            self.assertIsNotNone(cmd)
            self.assertEqual(cmd.kind, kind)  # dispatch maps all three
        self.assertEqual(parse_control("/list building").tail, "building")


# ── CLI end-to-end: private NM_HOME, subprocess ──────────────────────────────


def _nm(args: list[str], home: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = home
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "nomorals"] + args,
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env=env, timeout=120,
    )


class CLIE2ETest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w68-cli-")

    def test_commands_cli_full_and_filtered(self) -> None:
        proc = _nm(["commands"], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("building for real", proc.stdout)
        self.assertIn("/devon", proc.stdout)
        proc = _nm(["commands", "building"], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("/code", proc.stdout)
        self.assertNotIn("/dns", proc.stdout)

    def test_promote_local_writes_the_env_contract(self) -> None:
        proc = _nm(["models", "--promote-local", "/data/models/ft.gguf"],
                   self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("local model promoted", proc.stdout)
        env_text = (Path(self.home) / ".env").read_text(encoding="utf-8")
        for key in ("NM_LLM_PROVIDER=llama_cpp",
                    "NM_LLM_LOCAL_MODEL=/data/models/ft.gguf",
                    "NM_LLM_LOCAL_AUTO_START=1"):
            self.assertIn(key, env_text)

    def test_missions_pause_cancel_show(self) -> None:
        # create a real mission row in the same home, then control it
        # through the CLI — the cross-process operator contract.
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from nomorals.core.config import load_settings\n"
            "from nomorals.agents.context import build_context\n"
            "from nomorals.missions import MissionStore\n"
            "with build_context(load_settings()) as ctx:\n"
            "    s = MissionStore(ctx.db)\n"
            "    m = s.create_new('alpha then beta then gamma')\n"
            "    m.state['plan'] = [{'name': 'alpha', 'goal': 'a', "
            "'role': 'execution', 'kind': 'io', 'depends_on': []},\n"
            "     {'name': 'beta', 'goal': 'b', 'role': 'execution', "
            "'kind': 'io', 'depends_on': []},\n"
            "     {'name': 'gamma', 'goal': 'c', 'role': 'execution', "
            "'kind': 'io', 'depends_on': []}]\n"
            "    s.save(m)\n"
            "    print(m.id)\n" % str(REPO_ROOT)
        )
        env = dict(os.environ)
        env["NM_HOME"] = self.home
        env["PYTHONPATH"] = str(REPO_ROOT)
        created = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True,
                                 cwd=str(REPO_ROOT), env=env, timeout=120)
        self.assertEqual(created.returncode, 0, created.stderr)
        mid = created.stdout.strip().splitlines()[-1]

        proc = _nm(["missions", "--pause", mid], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("paused", proc.stdout)

        proc = _nm(["missions", "--show", mid], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("[paused]", proc.stdout)
        self.assertIn("steps", proc.stdout)

        proc = _nm(["missions", "--cancel", mid], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("cancelled", proc.stdout)

        proc = _nm(["missions", "--show", mid], self.home)
        self.assertIn("[cancelled]", proc.stdout)

    def test_missions_resume_status(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from nomorals.core.config import load_settings\n"
            "from nomorals.agents.context import build_context\n"
            "from nomorals.missions import MissionStore\n"
            "with build_context(load_settings()) as ctx:\n"
            "    s = MissionStore(ctx.db)\n"
            "    m = s.create_new('alpha then beta')\n"
            "    print(m.id)\n" % str(REPO_ROOT)
        )
        env = dict(os.environ)
        env["NM_HOME"] = self.home
        env["PYTHONPATH"] = str(REPO_ROOT)
        created = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True,
                                 cwd=str(REPO_ROOT), env=env, timeout=120)
        self.assertEqual(created.returncode, 0, created.stderr)
        mid = created.stdout.strip().splitlines()[-1]
        _nm(["missions", "--pause", mid], self.home)
        proc = _nm(["missions", "--resume-status", mid], self.home)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("running", proc.stdout)


if __name__ == "__main__":
    unittest.main()
