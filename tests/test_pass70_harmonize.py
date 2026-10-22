"""Pass #70: two bridges, one settlement — and status routing by
judgment, not keywords.

The survey (2026-10-21 coding-agent survey, items from passes #61/#66
+ the #67 swept lead, site #14):

- TWO bridges drive the one real coding loop
  (``agents/coding.py``'s ``CodingAgent``): ``MissionCodingAgent``
  (mission runner path) and ``CodingRoleAgent`` (orchestrator/swarm
  path).  Their settlements drifted: the mission settlement carried
  steps / iterations / verification / replans / seconds / usage; the
  orchestrator output carried none of those, named the loop's output
  tail ``summary`` where the mission (and ``CodingResult.to_dict``)
  say ``output``, and put the RAW ``result.error`` on the dict always
  — so a run that failed with an empty error string (e.g. a
  plan-approval pause, which the orchestrator bridge never opted out
  of) read as SUCCESS to the orchestrator's ``result.get("error")``
  dispatch check while the same run settled FAILED through the
  mission bridge.  The mission bridge also forwarded step payloads
  (``accept``, ``max_tokens``) and forced ``plan_mode=False`` (an
  autonomous step has no approver); the orchestrator bridge ignored
  both payloads and left plan mode at the loop's interactive default.
- The two coding allowlists drifted too: ``tools/agents.py``'s
  ``CODING_TOOLS`` (12 names) vs the ``coder`` RoleSpec allowlist
  (15) — each has read/navigation names the other lacks, for the
  same role driving the same loop.
- ``devon.py``'s mission tool routed "status query vs new mission"
  by a keyword list + a 120-char length cap: a long status question
  about an existing mission spawned a duplicate mission instead of
  answering.

The contract pinned here:

- ONE shared settlement builder, ``coding_settlement`` in
  ``agents/coding.py``: both bridges emit the same coding-account
  keys with the same values for the same run (goal, engine, ok,
  files, iterations, verification, steps, replans, review, output,
  seconds, usage, scope_denials, step_account when non-empty,
  capped when the run hit its cap, and — only on failure — an
  ``error`` that is NEVER empty: the loop's error, else the
  plan-approval reason, else the failed step named, else a generic
  honest reason).  Consumers (mission runner, orchestrator dispatch,
  ``payload_text``) read the harmonized shape unchanged.
- The bridges invoke the loop identically for the same payload:
  ``plan_mode=False``, ``accept``/``max_tokens`` forwarded, the same
  defaults (5 iterations x 300s), background tests on.
- ``CODING_TOOLS`` == the ``coder`` RoleSpec allowlist, and the
  loop's governed surface (the ``_require_scope`` probes in
  coding.py + the explore ``fs_read``) stays inside it — a drift
  pin, so the two lists cannot silently diverge again.  Every name
  exists in a built registry with its capability covered by
  ``CODING_CAPABILITIES`` (a name the grant doesn't cover would
  silently vanish from the scoped registry's schemas).
- devon mission routing: structure first (no missions in the store
  -> the message cannot be a status query -> spawn, no brain call);
  else the brain judges (one bounded structured call over the
  mission roster); with no judgment available the fallback REPORTS
  instead of spawning — a wrong report costs the owner one
  re-issued message, a wrong spawn starts an unlimited autonomous
  mission nobody asked for.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nomorals.agents.coding as coding_module
from nomorals.agents.coding import CodingResult
from nomorals.agents.role_specs import BUILTIN_ROLES
from nomorals.agents.roles import MissionCodingAgent
from nomorals.missions.progress import payload_text
from nomorals.storage.db import Database
from nomorals.tools.agents import (
    CODING_CAPABILITIES,
    CODING_TOOLS,
    CodingRoleAgent,
)
from nomorals.tools.registry import ToolRegistry


# ── fabricated results (pass #69 shapes) ─────────────────────────────


def _verify_ok():
    return {"ran": True, "ok": True, "tests": ["tests/test_a.py::test_a"],
            "failed": [], "checks": 1, "executed_total": 1, "note": ""}


def _step(sid, status, *, instructions="", path="", attempts=1,
          reason="", verify=None, depends_on=None):
    d = {"id": sid, "path": path, "files": [path] if path else [],
         "status": status, "reason": reason, "attempts": attempts,
         "instructions": instructions, "acceptance": "",
         "depends_on": list(depends_on or [])}
    if verify is not None:
        d["verify"] = verify
    return d


def _rich_result(**overrides):
    kw = dict(
        ok=True, iterations=3, files=["calc.py", "main.py"],
        output="loop output tail", error="", seconds=12.345,
        verification="tests", replans=1,
        review={"passed": True, "available": True, "rounds": 1,
                "objections": []},
        steps=[
            _step("s1", "done", instructions="add the config parser",
                  path="calc.py", verify=_verify_ok()),
            _step("s2", "done", instructions="wire the parser into main",
                  path="main.py", verify=_verify_ok()),
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 40,
               "total_tokens": 140, "calls": 2, "usage_unreported": 0,
               "by_phase": {}},
        scope_denials=[{"tool": "fs_write",
                        "reason": "outside the role scope"}],
    )
    kw.update(overrides)
    return CodingResult(**kw)


class FakeLoop:
    """CodingAgent double: records construction + run kwargs, returns
    the scripted result (the pass #61/#66 pattern)."""

    scripted: CodingResult | None = None
    instances: list["FakeLoop"] = []

    def __init__(self, context, **kwargs):
        self.context = context
        self.init_kwargs = kwargs
        self.run_calls: list = []
        FakeLoop.instances.append(self)

    def run(self, goal, **kwargs):
        self.run_calls.append((goal, kwargs))
        return FakeLoop.scripted


def _role_bridge_output(result, payload=None):
    FakeLoop.scripted = result
    FakeLoop.instances = []
    registry = ToolRegistry()
    registry.register_builtins()
    ctx = SimpleNamespace(db=None, tools=registry, memory=None,
                          router=None, settings=SimpleNamespace(),
                          extras={})
    registry.context = ctx
    agent = CodingRoleAgent(ctx, registry)
    return agent.run(payload or {"task": "build the thing"})


def _mission_bridge_output(result, spec=None):
    FakeLoop.scripted = result
    FakeLoop.instances = []
    ctx = SimpleNamespace(db=None, tools=FakeInner(), memory=None,
                          router=None, settings=SimpleNamespace(),
                          extras={})
    agent = MissionCodingAgent(context=ctx)
    payload = {"goal": "build the thing"}
    payload.update(spec or {})
    return agent.work(payload)


class FakeInner:
    """Minimal context.tools stand-in (unbound mission agent never
    touches it; bound paths are covered by pass #66's suites)."""

    def get(self, name):
        return None

    def names(self):
        return []


# ── settlement parity ────────────────────────────────────────────────


class SettlementParityTests(unittest.TestCase):

    def test_builder_is_the_one_shape(self):
        from nomorals.agents.coding import coding_settlement

        out = coding_settlement(_rich_result(), goal="build the thing")
        self.assertEqual(out["engine"], "coding_agent")
        self.assertEqual(out["goal"], "build the thing")
        self.assertEqual(out["output"], "loop output tail")
        self.assertNotIn("summary", out)
        self.assertNotIn("error", out)  # green run: no error key
        self.assertEqual(out["iterations"], 3)
        self.assertEqual(out["verification"], "tests")
        self.assertEqual(out["replans"], 1)
        self.assertEqual(out["seconds"], 12.35)
        self.assertEqual(out["usage"]["total_tokens"], 140)
        self.assertIn("step_account", out)
        self.assertNotIn("capped", out)

    def test_both_bridges_settle_the_same_run_identically(self):
        result = _rich_result()
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            role_out = _role_bridge_output(result).output
            mission_out = _mission_bridge_output(result)
        self.assertEqual(role_out, mission_out)

    def test_failure_error_is_synthesized_and_identical(self):
        # The orchestrator hole: ok=False with an EMPTY raw error read
        # as success through the role bridge (dispatch checks
        # result.get("error")) while the mission bridge settled it
        # failed with the failed step named.
        steps = [
            _step("s1", "done", instructions="add the parser",
                  path="calc.py", verify=_verify_ok()),
            _step("s2", "failed", instructions="render the title",
                  path="report.py", attempts=2,
                  reason="verify failed: tests/test_r.py::test_t"),
        ]
        result = _rich_result(ok=False, error="", steps=steps)
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            role_out = _role_bridge_output(result).output
            mission_out = _mission_bridge_output(result)
        self.assertEqual(role_out, mission_out)
        self.assertTrue(role_out["error"])  # never empty on failure
        self.assertIn("s2", role_out["error"])
        # the exact predicates both consumers apply:
        for out in (role_out, mission_out):
            self.assertEqual(bool(out.get("error")), not out["ok"])

    def test_needs_approval_pause_settles_failed_on_both(self):
        result = _rich_result(ok=False, error="", needs_approval=True,
                              steps=[])
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            role_out = _role_bridge_output(result).output
            mission_out = _mission_bridge_output(result)
        self.assertEqual(role_out["error"], mission_out["error"])
        self.assertIn("plan approval", role_out["error"])

    def test_capped_rides_both_settlements(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15, "calls": 1, "usage_unreported": 0,
                 "by_phase": {}, "cap": 12, "capped": True}
        result = _rich_result(ok=False, error="spend cap reached",
                              usage=usage)
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            role_out = _role_bridge_output(result).output
            mission_out = _mission_bridge_output(result)
        self.assertEqual(role_out, mission_out)
        self.assertIs(role_out["capped"], True)
        self.assertEqual(role_out["usage"]["cap"], 12)

    def test_consumers_read_the_harmonized_shape(self):
        result = _rich_result()
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            role_out = _role_bridge_output(result).output
            mission_out = _mission_bridge_output(result)
        # missions' results renderer: the account leads, the loop's
        # output follows as evidence — for BOTH bridges' dicts now.
        for out in (role_out, mission_out):
            text = payload_text(out)
            self.assertTrue(text.startswith(out["step_account"]))
            self.assertIn("loop output tail", text)


class InvocationParityTests(unittest.TestCase):
    """The same step payload must drive the loop the same way
    through either bridge (pass #61's contract, extended to the
    orchestrator bridge)."""

    PAYLOAD = {"accept": "python3 -m pytest -q", "max_tokens": 5000}

    def test_role_bridge_forwards_payload_and_forces_plan_mode_off(self):
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            _role_bridge_output(_rich_result(),
                                {"task": "build the thing",
                                 **self.PAYLOAD})
        _, kwargs = FakeLoop.instances[-1].run_calls[0]
        self.assertEqual(kwargs["accept"], "python3 -m pytest -q")
        self.assertEqual(kwargs["max_tokens"], 5000)
        self.assertIs(kwargs["plan_mode"], False)
        self.assertIs(kwargs["background_tests"], True)

    def test_both_bridges_call_the_loop_identically(self):
        with mock.patch.object(coding_module, "CodingAgent", FakeLoop):
            _role_bridge_output(_rich_result(),
                                {"task": "build the thing",
                                 **self.PAYLOAD})
            role_kwargs = FakeLoop.instances[-1].run_calls[0][1]
            _mission_bridge_output(_rich_result(), dict(self.PAYLOAD))
            mission_kwargs = FakeLoop.instances[-1].run_calls[0][1]
        shared = ("accept", "max_iterations", "timeout", "plan_mode",
                  "max_tokens", "background_tests")
        for key in shared:
            self.assertEqual(role_kwargs[key], mission_kwargs[key],
                             f"bridge divergence on {key}")
        # the harmonized defaults: the mission bridge's documented
        # 5 x 300s for autonomous steps.
        self.assertEqual(role_kwargs["max_iterations"], 5)
        self.assertEqual(role_kwargs["timeout"], 300.0)


# ── allowlist drift pin ──────────────────────────────────────────────


class AllowlistDriftPinTests(unittest.TestCase):

    def test_governed_surface_stays_inside_the_allowlist(self):
        # Derive the loop's governed surface from its own source:
        # every _require_scope("...") probe, plus the explore phase's
        # fs_read (which flows through call_many, not _require_scope).
        src = (Path(coding_module.__file__)
               .read_text(encoding="utf-8"))
        governed = set(re.findall(r'_require_scope\("([a-z_]+)"', src))
        self.assertIn("fs_read", src)  # explore reads exist
        governed.add("fs_read")
        self.assertEqual(governed, {"fs_read", "edit_file",
                                    "apply_patch", "run_tests",
                                    "lint", "shell_run"})
        missing = governed - set(CODING_TOOLS)
        self.assertEqual(missing, set(),
                         f"governed tools missing from CODING_TOOLS: "
                         f"{sorted(missing)}")

    def test_the_two_coding_allowlists_are_one_list(self):
        spec_tools = set(BUILTIN_ROLES["coder"].tool_allowlist)
        self.assertEqual(set(CODING_TOOLS), spec_tools,
                         "CODING_TOOLS and the coder RoleSpec allowlist "
                         "drifted apart — one role, one surface")

    def test_every_allowlisted_tool_exists_and_is_capability_covered(self):
        registry = ToolRegistry()
        registry.register_builtins()
        for name in CODING_TOOLS:
            spec = registry.get(name)
            self.assertIsNotNone(spec, f"{name} not a registered tool")
            self.assertIn(spec.capability, CODING_CAPABILITIES,
                          f"{name} needs {spec.capability}, outside "
                          f"CODING_CAPABILITIES — it would silently "
                          f"vanish from the scoped registry's schemas")


# ── devon mission routing ────────────────────────────────────────────

from nomorals.agents import devon as devon_mod  # noqa: E402
from nomorals.agents.devon import DevonAgent  # noqa: E402
from nomorals.missions.mission import MissionStore  # noqa: E402


class ScriptedRouter:
    """Duck-typed router (the pass #61 pattern): answers the routing
    judgment with the scripted JSON, records every call."""

    def __init__(self, script):
        self.script = script
        self.calls: list = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        self.calls.append(texts)
        text = json.dumps(self.script) if self.script is not None else ""
        resp = SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "no script")
        resp.usage = SimpleNamespace(prompt_tokens=10,
                                     completion_tokens=5,
                                     total_tokens=15)
        return resp


class FakeThread:
    started: list = []

    def __init__(self, *args, **kwargs):
        self.target = kwargs.get("target") or (
            args[0] if args else None)
        FakeThread.started.append(self)

    def start(self):
        pass  # never run the mission in a routing test


class FakeBriefAgent:
    def __init__(self, context):
        self.context = context

    def refine(self, goal, kind="mission"):
        return SimpleNamespace(
            as_goal=lambda: goal,
            to_dict=lambda: {"by": "fake"})


class DevonMissionRoutingTests(unittest.TestCase):

    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = MissionStore(self.db)
        FakeThread.started = []

    def _agent(self, router):
        ctx = SimpleNamespace(db=self.db, tools=None, memory=None,
                              router=router,
                              settings=SimpleNamespace(), extras={})
        return DevonAgent(ctx)

    def _mission(self, name, goal="do the thing"):
        return self.store.create_new(goal, name=name)

    def _call(self, agent, goal, name="devon-mission"):
        with mock.patch.object(devon_mod.threading, "Thread",
                               FakeThread), \
                mock.patch("nomorals.agents.brief.BriefAgent",
                           FakeBriefAgent):
            return agent._tool_mission({"goal": goal, "name": name})

    def test_no_missions_means_spawn_without_a_brain_call(self):
        # Structure first: with nothing in the store, a status-shaped
        # message cannot be a status query — and no model call is
        # spent deciding it.
        router = ScriptedRouter({"intent": "status_query",
                                 "mission": ""})
        out = self._call(self._agent(router),
                         "what is the status of my missions?")
        self.assertIn("mission started in background", out)
        self.assertEqual(router.calls, [])
        self.assertEqual(len(FakeThread.started), 1)

    def test_short_status_question_routes_to_the_report(self):
        self._mission("data-scrape", "scrape the rates nightly")
        router = ScriptedRouter({"intent": "status_query",
                                 "mission": "data-scrape"})
        out = self._call(self._agent(router),
                         "status of the data-scrape mission?")
        self.assertTrue(out.startswith("mission progress:"))
        self.assertIn("data-scrape", out)
        self.assertEqual(FakeThread.started, [])

    def test_clear_new_task_still_spawns(self):
        self._mission("data-scrape", "scrape the rates nightly")
        router = ScriptedRouter({"intent": "new_mission",
                                 "mission": ""})
        out = self._call(
            self._agent(router),
            "gather the Q3 invoices into a spreadsheet and total them")
        self.assertIn("mission started in background", out)
        self.assertEqual(len(FakeThread.started), 1)

    def test_long_status_question_routes_to_the_report(self):
        # The #67 lead: the old keyword+length gate sent exactly
        # this shape (a status question over 120 chars) to spawn a
        # duplicate mission.
        self._mission("data-scrape", "scrape the rates nightly")
        router = ScriptedRouter({"intent": "status_query",
                                 "mission": "data-scrape"})
        goal = ("please tell me how the data-scrape mission has been "
                "getting on since yesterday evening, whether the "
                "nightly scrape finished, and if anything is stuck")
        self.assertGreater(len(goal), 120)
        out = self._call(self._agent(router), goal)
        self.assertTrue(out.startswith("mission progress:"))
        self.assertIn("data-scrape", out)
        self.assertEqual(FakeThread.started, [])

    def test_long_new_task_still_spawns(self):
        self._mission("data-scrape", "scrape the rates nightly")
        router = ScriptedRouter({"intent": "new_mission",
                                 "mission": ""})
        goal = ("build a brand new importer that pulls the warehouse "
                "stock levels every morning, reconciles them against "
                "the storefront counts, and writes the differences "
                "into a fresh report file")
        self.assertGreater(len(goal), 120)
        out = self._call(self._agent(router), goal)
        self.assertIn("mission started in background", out)
        self.assertEqual(len(FakeThread.started), 1)

    def test_no_brain_fallback_reports_instead_of_spawning(self):
        # No router -> no judgment available.  The fallback is the
        # progress report: a wrong report costs one re-issued
        # message, a wrong spawn starts an unlimited autonomous
        # mission nobody asked for.
        self._mission("data-scrape", "scrape the rates nightly")
        out = self._call(self._agent(None), "how is the scrape going?")
        self.assertTrue(out.startswith("mission progress:"))
        self.assertEqual(FakeThread.started, [])

    def test_brain_named_mission_filters_the_report(self):
        self._mission("alpha-scrape", "scrape alpha")
        self._mission("beta-audit", "audit beta")
        router = ScriptedRouter({"intent": "status_query",
                                 "mission": "beta-audit"})
        out = self._call(self._agent(router), "how is beta-audit doing?")
        self.assertTrue(out.startswith("mission progress:"))
        self.assertIn("beta-audit", out)
        self.assertNotIn("alpha-scrape", out)
        self.assertEqual(FakeThread.started, [])


if __name__ == "__main__":
    unittest.main()
