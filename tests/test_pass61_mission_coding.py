"""Pass #61 (W3): mission coding runs on the real CodingAgent loop.

The defect, from the 2026-10-20 coding-agent survey: mission coding
steps routed through ``missions/runner.build_agent`` to a SHADOW
``CodingAgent`` in ``agents/roles`` — single-shot generate → fs_write
→ run the file. No plan, no tests, no lint, no review, no rollback —
and its name collided with the real ``agents/coding.CodingAgent``,
hiding the substitution in logs.

These tests pin the unified contract:

- the ``coding`` role registry entry is a bridge
  (:class:`MissionCodingAgent`) that drives the real loop, and no
  class named ``CodingAgent`` remains importable from ``agents.roles``;
- the demoted single-shot writer survives under an honest name
  (``SnippetWriterAgent``, role ``snippet``), unregistered, still
  doing its lightweight job and still failing honestly with no model;
- a mission step's carried ``accept`` command (plan payload → the
  runner's serialization whitelist → rehydrated step → ``bind_step``)
  becomes the loop run's accept check, and the mission budget defaults
  are the deliberate 5 iterations / 300s with plan_mode forced off;
- outcome honesty end-to-end through the mission runner: loop green →
  step DONE with the loop's evidence (files, per-step status,
  verification) in the payload; loop red → step FAILED with the
  loop's reason in the detail, nothing recorded as an output;
- one true end-to-end: bridge → REAL loop (scripted model) → a file
  on disk that satisfies the step's accept command.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.coding as coding_module
import nomorals.agents.roles as roles
from nomorals.agents.coding import CodingAgent, CodingResult
from nomorals.agents.orchestrator import Plan, PlanStep
from nomorals.agents.roles import (
    MissionCodingAgent,
    SnippetWriterAgent,
    build_agent,
)
from nomorals.core.result import Err, Ok
from nomorals.core.tasks import Task
from nomorals.missions import MissionStatus, MissionStore
from nomorals.missions.mission import known_step_roles
from nomorals.missions.runner import MissionRunner, _serialize_plan
from nomorals.storage.db import Database


# ── fakes ────────────────────────────────────────────────────────────


class FakeRegistry:
    """Tool registry stand-in with the real call() outcome contract."""

    def __init__(self):
        self.calls = []
        self.writes = {}

    def call(self, name, *, actor="system", capabilities=None,
             idempotency_key=None, **kwargs):
        self.calls.append((name, kwargs))
        if name == "fs_write":
            self.writes[kwargs.get("path", "")] = kwargs.get("content", "")
            return Ok({"path": kwargs.get("path", "")})
        if name == "shell_run":
            return Ok({"exit_code": 0, "stdout": "ran", "stderr": ""})
        return Err(ValueError(f"unknown tool {name}"))


def _ctx(router=None, tools=None):
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db, tools=tools if tools is not None else FakeRegistry(),
        memory=None, router=router,
        settings=SimpleNamespace(), extras={})


class FakeLoop:
    """Stand-in for the real CodingAgent, at the exact boundary the
    bridge depends on: constructed with (context, root=...), run with
    the loop's keyword contract, returning a real CodingResult."""

    scripted: CodingResult | None = None
    raises: Exception | None = None
    instances: list["FakeLoop"] = []

    def __init__(self, context, root=None):
        self.context = context
        self.root = root
        self.run_calls: list[tuple[str, dict]] = []
        FakeLoop.instances.append(self)

    def run(self, task, **kwargs):
        self.run_calls.append((task, kwargs))
        if FakeLoop.raises is not None:
            raise FakeLoop.raises
        assert FakeLoop.scripted is not None
        return FakeLoop.scripted


def _green_result() -> CodingResult:
    return CodingResult(
        ok=True, iterations=2, files=["hello.py"],
        output="all good", verification="tests",
        steps=[{"id": "s1", "path": "hello.py", "files": ["hello.py"],
                "status": "done", "reason": "", "attempts": 1,
                "instructions": "create it", "acceptance": "tests pass",
                "depends_on": []}],
        replans=0)


def _red_result() -> CodingResult:
    return CodingResult(
        ok=False, iterations=3, files=["hello.py"],
        output="", error="", verification="tests",
        steps=[{"id": "s1", "path": "hello.py", "files": ["hello.py"],
                "status": "failed", "reason": "tests failed: 1 assertion",
                "attempts": 3, "instructions": "create it",
                "acceptance": "tests pass", "depends_on": []}],
        replans=1)


class _FakeLoopCase(unittest.TestCase):
    def setUp(self):
        FakeLoop.scripted = None
        FakeLoop.raises = None
        FakeLoop.instances = []
        patcher = mock.patch.object(coding_module, "CodingAgent", FakeLoop)
        patcher.start()
        self.addCleanup(patcher.stop)

    @property
    def loop(self) -> FakeLoop:
        self.assertEqual(len(FakeLoop.instances), 1)
        return FakeLoop.instances[0]


# ── the rename / registry ────────────────────────────────────────────


class RegistryRenameTests(unittest.TestCase):
    def test_coding_role_is_the_bridge(self):
        self.assertIs(roles.ROLES["coding"], MissionCodingAgent)
        agent = build_agent("coding", context=_ctx())
        self.assertIsInstance(agent, MissionCodingAgent)

    def test_no_shadow_under_the_colliding_name(self):
        self.assertFalse(hasattr(roles, "CodingAgent"))
        self.assertNotIn("CodingAgent", roles.__all__)
        # the one true CodingAgent lives in agents.coding
        self.assertEqual(CodingAgent.__module__, "nomorals.agents.coding")

    def test_snippet_writer_demoted_and_unregistered(self):
        self.assertEqual(SnippetWriterAgent.role, "snippet")
        self.assertNotIn(SnippetWriterAgent, roles.ROLES.values())
        self.assertNotIn("snippet", roles.ROLES)

    def test_coding_stays_a_valid_mission_step_role(self):
        self.assertIn("coding", known_step_roles())

    def test_swarm_coder_delegation_lands_on_the_bridge(self):
        from nomorals.agents.role_specs import SwarmAgent

        legacy = SwarmAgent.LEGACY_DELEGATION["coder"]
        self.assertIs(roles.ROLES[legacy], MissionCodingAgent)


# ── the bridge: wiring into the real loop ────────────────────────────


class BridgeWiringTests(_FakeLoopCase):
    def test_green_loop_evidence_no_error(self):
        FakeLoop.scripted = _green_result()
        agent = MissionCodingAgent(context=_ctx())
        out = agent.work("build the thing")
        self.assertNotIn("error", out)
        self.assertTrue(out["ok"])
        self.assertEqual(out["engine"], "coding_agent")
        self.assertEqual(out["files"], ["hello.py"])
        self.assertEqual(out["verification"], "tests")
        self.assertEqual(out["steps"][0]["status"], "done")
        task, kwargs = self.loop.run_calls[0]
        self.assertEqual(task, "build the thing")
        # mission budget, chosen deliberately (see roles module):
        self.assertEqual(kwargs["max_iterations"], 5)
        self.assertEqual(kwargs["timeout"], 300.0)
        self.assertEqual(kwargs["accept"], "")
        # a mission step cannot answer a plan-approval pause:
        self.assertIs(kwargs["plan_mode"], False)

    def test_step_payload_accept_becomes_the_accept_check(self):
        FakeLoop.scripted = _green_result()
        agent = MissionCodingAgent(context=_ctx())
        agent.bind_step(PlanStep(
            name="build", goal="build it", role="coding",
            payload={"accept": "python3 -m pytest -q"}))
        agent.work("the assembled step prompt")
        _, kwargs = self.loop.run_calls[0]
        self.assertEqual(kwargs["accept"], "python3 -m pytest -q")

    def test_explicit_spec_overrides_step_payload_and_budget(self):
        FakeLoop.scripted = _green_result()
        agent = MissionCodingAgent(context=_ctx())
        agent.bind_step(PlanStep(
            name="build", goal="build it", role="coding",
            payload={"accept": "from-step", "max_iterations": 9}))
        out = agent.work(Task(name="code", payload={
            "goal": "build it", "accept": "make check",
            "max_iterations": 3, "timeout": 42}))
        self.assertNotIn("error", out)
        _, kwargs = self.loop.run_calls[0]
        self.assertEqual(kwargs["accept"], "make check")
        self.assertEqual(kwargs["max_iterations"], 3)
        self.assertEqual(kwargs["timeout"], 42.0)

    def test_root_rides_the_spec(self):
        FakeLoop.scripted = _green_result()
        agent = MissionCodingAgent(context=_ctx())
        agent.work({"goal": "build it", "root": "/tmp/somewhere"})
        self.assertEqual(self.loop.root, "/tmp/somewhere")

    def test_serialize_plan_carries_accept_in_policy(self):
        plan = Plan(goal="g", steps=[PlanStep(
            name="build", goal="build it", role="coding",
            payload={"accept": "python3 -m pytest -q", "retries": 2})])
        entry = _serialize_plan(plan)[0]
        self.assertEqual(entry["policy"]["accept"],
                         "python3 -m pytest -q")
        self.assertEqual(entry["policy"]["retries"], 2)


# ── the bridge: outcome honesty ──────────────────────────────────────


class BridgeHonestyTests(_FakeLoopCase):
    def test_red_loop_names_its_failed_step(self):
        FakeLoop.scripted = _red_result()
        agent = MissionCodingAgent(context=_ctx())
        out = agent.work("build the thing")
        self.assertFalse(out["ok"])
        self.assertIn("s1", out["error"])
        self.assertIn("hello.py", out["error"])
        self.assertIn("tests failed", out["error"])
        self.assertEqual(out["steps"][0]["status"], "failed")
        self.assertNotIn("code", out)

    def test_red_loop_error_text_wins(self):
        FakeLoop.scripted = CodingResult(
            ok=False, iterations=1, error="accept command failed (exit 1)")
        agent = MissionCodingAgent(context=_ctx())
        out = agent.work("build the thing")
        self.assertEqual(out["error"], "accept command failed (exit 1)")

    def test_loop_crash_is_an_honest_error_not_an_exception(self):
        FakeLoop.raises = RuntimeError("db gone")
        agent = MissionCodingAgent(context=_ctx())
        out = agent.work("build the thing")
        self.assertIn("error", out)
        self.assertIn("db gone", out["error"])

    def test_no_goal_is_an_error(self):
        agent = MissionCodingAgent(context=_ctx())
        out = agent.work(Task(name="", payload={}))
        self.assertIn("error", out)
        self.assertEqual(FakeLoop.instances, [])


# ── end to end through the mission runner ────────────────────────────


def _mission_with_plan(store, plan_entries, goal="test goal"):
    mission = store.create_new(goal, name="m")
    mission.state["plan"] = plan_entries
    mission.metadata["replan_policy"] = "off"
    store.save(mission)
    return mission


_CODING_ENTRY = {
    "name": "build", "goal": "build the thing", "role": "coding",
    "kind": "io", "depends_on": [],
    "policy": {"accept": "python3 -m pytest -q"},
}


class MissionRunnerEndToEndTests(_FakeLoopCase):
    def setUp(self):
        super().setUp()
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def _run(self):
        mission = _mission_with_plan(self.store, [dict(_CODING_ENTRY)])
        runner = MissionRunner(self.ctx, store=self.store,
                               milestones=False)
        return runner.run(mission), mission

    def test_green_loop_settles_done_with_evidence(self):
        FakeLoop.scripted = _green_result()
        result, mission = self._run()
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertTrue(result.steps[0].ok)
        out = self.store.get(mission.id).state["outputs"]["build"]
        self.assertEqual(out["engine"], "coding_agent")
        self.assertEqual(out["files"], ["hello.py"])
        self.assertEqual(out["verification"], "tests")
        self.assertEqual(out["steps"][0]["status"], "done")
        # the step's accept rode the whole chain into the loop call:
        _, kwargs = self.loop.run_calls[0]
        self.assertEqual(kwargs["accept"], "python3 -m pytest -q")

    def test_red_loop_settles_failed_with_reason(self):
        FakeLoop.scripted = CodingResult(
            ok=False, iterations=3, error="accept command failed (exit 1)",
            verification="accept")
        result, mission = self._run()
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertFalse(result.steps[0].ok)
        self.assertIn("accept command failed", result.steps[0].detail)
        saved = self.store.get(mission.id)
        self.assertNotIn("build", saved.state.get("outputs") or {})


# ── the demoted snippet writer still works ───────────────────────────


class SnippetWriterTests(unittest.TestCase):
    def test_no_model_no_fake_code(self):
        agent = SnippetWriterAgent(context=_ctx(router=None))
        out = agent.work(Task(name="code",
                              payload={"goal": "write fizzbuzz"}))
        self.assertIn("error", out)
        self.assertNotIn("code", out)

    def test_generated_snippet_is_written_and_run(self):
        registry = FakeRegistry()
        agent = SnippetWriterAgent(context=_ctx(tools=registry))
        with mock.patch.object(SnippetWriterAgent, "_generate",
                               return_value="print('hi')\n"):
            out = agent.work(Task(name="code",
                                  payload={"goal": "print hi"}))
        self.assertTrue(out["executed"])
        self.assertEqual(out["exit_code"], 0)
        self.assertEqual(list(registry.writes.values()),
                         ["print('hi')\n"])
        self.assertTrue(out["path"].startswith("generated/"))


# ── true end-to-end: bridge → REAL loop → file on disk ──────────────


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class _RuleRouter:
    """Content-routed fake LLM (pattern from test_pass60): answers by
    what the call is asking, so call order never matters."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        system = texts[0] if texts else ""
        user = texts[-1] if texts else ""
        self.calls.append((system, user))
        text = self.handler(system, user)
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")


class _StubDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class RealLoopEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass61_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        _git("init", "-q", cwd=self.root)
        _git("config", "user.email", "t@example.com", cwd=self.root)
        _git("config", "user.name", "T", cwd=self.root)

    def test_bridge_drives_the_real_loop_to_a_verified_file(self):
        def handler(system, user):
            if "build planner" in system:
                return "```json\n" + json.dumps({"steps": [{
                    "path": "hello.py", "why": "entry point",
                    "new_file": True,
                    "instructions": "create hello.py that prints hi",
                    "acceptance": "running hello.py prints hi",
                }]}) + "\n```"
            if "Write complete, runnable Python" in system:
                return "```python\nprint('hi')\n```\n"
            return ""

        ctx = SimpleNamespace(
            router=_RuleRouter(handler), db=_StubDB(),
            settings=SimpleNamespace(reasoning_mode="off"))
        agent = MissionCodingAgent(context=ctx)
        out = agent.work(Task(name="build", payload={
            "goal": "create hello.py that prints hi",
            "root": str(self.root),
            "accept": f'"{sys.executable}" hello.py',
        }))
        self.assertNotIn("error", out)
        self.assertTrue(out["ok"])
        self.assertEqual(out["verification"], "accept")
        self.assertEqual((self.root / "hello.py").read_text(),
                         "print('hi')\n")
        self.assertEqual(out["steps"][0]["status"], "done")
        self.assertIn("hello.py", out["files"])


if __name__ == "__main__":
    unittest.main()
