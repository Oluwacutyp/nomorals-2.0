"""Hollow-mission fix: steps execute or fail honestly.

The owner's live failure (2026-10-21): a chat research mission
"completed" 3/3 in 0s with every step payload an echo —
``{'goal': <step prompt>, 'status': 'no action specified'}``. The
chain was: plan entries carried no usable role → everything became
the shell-only execution agent → its no-command return settled as a
successful step. These tests pin the fix at every layer:

- role agents return a top-level ``error`` for no-op outcomes, and
  the mission runner settles that shape as a FAILED step;
- a step's role is required and validated (construction, fresh
  decomposition, replan) — never defaulted;
- persisted legacy role-less steps settle FAILED with
  "no role recorded for step";
- the chat research template assigns real roles, so the owner's
  scenario now produces research payloads, not echoes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.orchestrator import Plan, PlanStep
from nomorals.agents.partner import PartnerRuntime
from nomorals.agents.roles import (
    CodingAgent,
    CriticAgent,
    ExecutionAgent,
    MemoryAgent,
    ReflectionAgent,
    ResearchAgent,
    SocialAgent,
    TrainingAgent,
    VisionAgent,
)
from nomorals.core.errors import ValidationError
from nomorals.core.result import Err, Ok
from nomorals.core.tasks import Task, TaskKind
from nomorals.missions import MissionStatus, MissionStore
from nomorals.missions.mission import MISSION_TEMPLATES, validate_plan_roles
from nomorals.missions.runner import MissionRunner, _rehydrate_plan
from nomorals.storage.db import Database


# ── fakes ─────────────────────────────────────────────────────────────


class FakeRegistry:
    """Tool registry stand-in with the real call() outcome contract."""

    def __init__(self, hits=None, fetch_text="fetched body text"):
        self.hits = list(hits if hits is not None else [
            {"url": "https://example.com/a", "title": "Alpha"},
            {"url": "https://example.com/b", "title": "Beta"},
        ])
        self.fetch_text = fetch_text
        self.calls = []

    def call(self, name, *, actor="system", capabilities=None,
             idempotency_key=None, **kwargs):
        self.calls.append((name, kwargs))
        if name == "web_search":
            return Ok({"results": list(self.hits)})
        if name == "web_fetch":
            return Ok({"url": kwargs.get("url", ""),
                       "title": "Fetched",
                       "text": self.fetch_text})
        if name == "shell_run":
            return Ok({"exit_code": 0, "stdout": "ran", "stderr": ""})
        return Err(ValueError(f"unknown tool {name}"))


def _ctx(tools=None, memory=None, router=None):
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db, tools=tools if tools is not None else FakeRegistry(),
        memory=memory, router=router,
        settings=SimpleNamespace(), extras={})


def _runner(ctx, store, **kw):
    return MissionRunner(ctx, store=store, milestones=False, **kw)


def _mission_with_plan(store, plan_entries, goal="test goal", **meta):
    mission = store.create_new(goal, name="m")
    mission.state["plan"] = plan_entries
    for key, value in meta.items():
        mission.metadata[key] = value
    store.save(mission)
    return mission


def _step_entry(name, role=None, goal=None):
    entry = {"name": name, "goal": goal or f"do {name}",
             "kind": "io", "depends_on": []}
    if role is not None:
        entry["role"] = role
    return entry


# ── role agents: hollow returns became failures ──────────────────────


class RoleAgentHollowTests(unittest.TestCase):
    def test_execution_goal_only_is_an_error_not_an_echo(self):
        agent = ExecutionAgent(context=_ctx())
        out = agent.work(Task(name="gather sources",
                              payload={"goal": "gather sources"}))
        self.assertIn("error", out)
        self.assertIn("no action specified", out["error"])
        self.assertNotEqual(out.get("status"), "no action specified")

    def test_execution_with_command_still_works(self):
        agent = ExecutionAgent(context=_ctx())
        result = agent.run(Task(name="run it",
                                payload={"command": "echo hi"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.output["exit_code"], 0)

    def test_research_zero_hits_is_an_error(self):
        agent = ResearchAgent(context=_ctx(tools=FakeRegistry(hits=[])))
        out = agent.work("research money")
        self.assertIn("error", out)
        self.assertIn("no sources retrieved", out["error"])

    def test_research_with_hits_returns_real_payload(self):
        agent = ResearchAgent(context=_ctx())
        out = agent.work("research money")
        self.assertNotIn("error", out)
        self.assertEqual(len(out["sources"]), 2)
        self.assertTrue(out["summary"])

    def test_social_missing_inputs_are_errors(self):
        agent = SocialAgent(context=_ctx())
        out = agent.work(Task(name="post", payload={}))
        self.assertIn("error", out)
        self.assertIn("platform", out["error"])
        self.assertIn("content", out["error"])
        out = agent.work(Task(name="post", payload={"platform": "x"}))
        self.assertIn("content", out["error"])

    def test_training_without_dataset_is_an_error(self):
        agent = TrainingAgent(context=_ctx())
        out = agent.work(Task(name="train", payload={}))
        self.assertIn("error", out)
        self.assertIn("dataset", out["error"])

    def test_vision_without_image_is_an_error(self):
        agent = VisionAgent(context=_ctx())
        out = agent.work(Task(name="look", payload={}))
        self.assertIn("error", out)

    def test_critic_without_model_does_not_fabricate_a_score(self):
        agent = CriticAgent(context=_ctx(router=None))
        out = agent.work(Task(name="verify", payload={"output": "x"}))
        self.assertIn("error", out)
        self.assertNotIn("score", out)

    def test_reflection_without_memory_is_an_error(self):
        agent = ReflectionAgent(context=_ctx(memory=None))
        out = agent.work(Task(name="reflect", payload={}))
        self.assertIn("error", out)

    def test_memory_without_subsystem_is_an_error(self):
        agent = MemoryAgent(context=_ctx(memory=None))
        out = agent.work(Task(name="remember", payload={}))
        self.assertEqual(out.get("error"), "no memory subsystem")

    def test_coding_without_model_produces_no_fake_code(self):
        agent = CodingAgent(context=_ctx(router=None))
        out = agent.work(Task(name="code",
                              payload={"goal": "write fizzbuzz"}))
        self.assertIn("error", out)
        self.assertNotIn("code", out)


# ── role validation: construction / decomposition ────────────────────


class RoleValidationTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_validator_rejects_roleless_plan_naming_the_step(self):
        with self.assertRaises(ValidationError) as cm:
            validate_plan_roles([_step_entry("gather sources")])
        self.assertIn("gather sources", str(cm.exception))

    def test_validator_rejects_unknown_role_naming_the_step(self):
        with self.assertRaises(ValidationError) as cm:
            validate_plan_roles([_step_entry("mystery", role="wizard")])
        self.assertIn("mystery", str(cm.exception))
        self.assertIn("wizard", str(cm.exception))

    def test_create_new_accepts_valid_plan(self):
        mission = self.store.create_new(
            "g", state={"plan": [_step_entry("gather", role="research")]})
        self.assertEqual(mission.state["plan"][0]["role"], "research")

    def test_store_funnel_copies_legacy_plans_verbatim(self):
        # The store funnel is also the copy path (redrive / retry copy
        # a recorded plan into a fresh mission): a legacy role-less
        # plan must remain copyable — the runner's execution gate is
        # what settles its steps as failures (see LegacyPlanTests).
        mission = self.store.create_new(
            "g", state={"plan": [_step_entry("old step")]})
        self.assertNotIn("role", mission.state["plan"][0])

    def test_builtin_templates_construct_with_valid_roles(self):
        for key in MISSION_TEMPLATES:
            mission = self.store.create_from_template(key, "some goal")
            validate_plan_roles(mission.state["plan"])

    def test_chat_template_specs_carry_valid_declared_roles(self):
        for template in ("research", "build", "fix"):
            spec = PartnerRuntime._mission_template_spec(template, "money")
            self.assertIsNotNone(spec)
            validate_plan_roles(spec["state"]["plan"])
            # construction through the store must accept them too
            self.store.create_new(**spec)
        research = PartnerRuntime._mission_template_spec("research", "money")
        roles = {s["name"]: s["role"] for s in research["state"]["plan"]}
        self.assertEqual(roles["gather sources"], "research")

    def test_runner_rejects_roleless_fresh_decomposition(self):
        mission = self.store.create_new("research money", name="m")
        bad_plan = Plan(goal="research money", steps=[
            PlanStep(name="gather sources", goal="gather", role="")])
        runner = _runner(self.ctx, self.store,
                         orchestrator_factory=lambda: SimpleNamespace(
                             plan=lambda goal: bad_plan))
        with self.assertRaises(ValidationError) as cm:
            runner.run(mission)
        self.assertIn("gather sources", str(cm.exception))

    def test_runner_rejects_unknown_role_fresh_decomposition(self):
        mission = self.store.create_new("research money", name="m")
        bad_plan = Plan(goal="research money", steps=[
            PlanStep(name="divine", goal="divine", role="wizard")])
        runner = _runner(self.ctx, self.store,
                         orchestrator_factory=lambda: SimpleNamespace(
                             plan=lambda goal: bad_plan))
        with self.assertRaises(ValidationError) as cm:
            runner.run(mission)
        self.assertIn("wizard", str(cm.exception))


# ── runner settle behavior ────────────────────────────────────────────


class RunnerSettleTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_execution_goal_only_step_settles_failed(self):
        mission = _mission_with_plan(
            self.store, [_step_entry("do thing", role="execution")],
            replan_policy="off")
        result = _runner(self.ctx, self.store).run(mission)
        self.assertEqual(result.status, MissionStatus.FAILED)
        step = result.steps[0]
        self.assertFalse(step.ok)
        self.assertIn("no action specified", step.detail)
        saved = self.store.get(mission.id)
        self.assertNotIn("do thing", saved.state.get("outputs") or {})

    def test_research_step_with_sources_settles_done(self):
        mission = _mission_with_plan(
            self.store, [_step_entry("gather", role="research")],
            replan_policy="off")
        result = _runner(self.ctx, self.store).run(mission)
        self.assertEqual(result.status, MissionStatus.DONE)
        out = self.store.get(mission.id).state["outputs"]["gather"]
        self.assertEqual(len(out["sources"]), 2)
        self.assertTrue(out["summary"])

    def test_research_step_with_zero_hits_settles_failed(self):
        ctx = _ctx(tools=FakeRegistry(hits=[]))
        store = MissionStore(ctx.db)
        mission = _mission_with_plan(
            store, [_step_entry("gather", role="research")],
            replan_policy="off")
        result = _runner(ctx, store).run(mission)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertIn("no sources retrieved", result.steps[0].detail)

    def test_memory_error_dict_settles_failed(self):
        # MemoryAgent has returned {"error": ...} all along; the
        # runner now honors that shape as a failure.
        mission = _mission_with_plan(
            self.store, [_step_entry("remember", role="memory")],
            replan_policy="off")
        result = _runner(self.ctx, self.store).run(mission)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertIn("no memory subsystem", result.steps[0].detail)


# ── legacy persisted plans ────────────────────────────────────────────


class LegacyPlanTests(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.store = MissionStore(self.ctx.db)

    def test_rehydrate_preserves_missing_role(self):
        steps = _rehydrate_plan("g", [_step_entry("a")])
        self.assertEqual(steps[0].role, "")

    def test_roleless_persisted_step_settles_failed_with_reason(self):
        mission = _mission_with_plan(
            self.store, [_step_entry("gather sources")],
            replan_policy="off")
        result = _runner(self.ctx, self.store).run(mission)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertFalse(result.steps[0].ok)
        self.assertEqual(result.steps[0].detail,
                         "no role recorded for step")

    def test_unknown_role_persisted_step_settles_failed(self):
        mission = _mission_with_plan(
            self.store, [_step_entry("divine", role="wizard")],
            replan_policy="off")
        result = _runner(self.ctx, self.store).run(mission)
        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertIn("unknown role recorded for step",
                      result.steps[0].detail)


# ── replan validation ─────────────────────────────────────────────────


class ReplanRoleTests(unittest.TestCase):
    def test_replan_with_roleless_steps_is_rejected(self):
        ctx = _ctx()
        store = MissionStore(ctx.db)
        mission = store.create_new("g", name="m")
        mission.metadata["replan_policy"] = "always"
        store.save(mission)
        bad_plan = Plan(goal="g", steps=[
            PlanStep(name="retry gather", goal="gather", role="")])
        runner = _runner(ctx, store,
                         orchestrator_factory=lambda: SimpleNamespace(
                             plan=lambda goal: bad_plan))
        out = runner._maybe_replan(
            mission,
            [PlanStep(name="gather", goal="g", role="research",
                      kind=TaskKind.IO)],
            "gather", "it failed", set(), {"gather"})
        self.assertIsNone(out)
        self.assertFalse(mission.state.get("replans"))


# ── the owner's scenario, end to end ──────────────────────────────────


class OwnerScenarioTests(unittest.TestCase):
    def test_chat_research_mission_produces_findings_not_echoes(self):
        ctx = _ctx()
        store = MissionStore(ctx.db)
        spec = PartnerRuntime._mission_template_spec("research", "money")
        mission = store.create_new(**spec)
        result = _runner(ctx, store).run(mission)
        self.assertEqual(result.status, MissionStatus.DONE)
        outputs = self.store_outputs(store, mission)
        self.assertEqual(set(outputs), {"gather sources",
                                        "synthesize findings",
                                        "write report"})
        for name, payload in outputs.items():
            self.assertNotEqual(set(payload), {"goal", "status"},
                                f"{name} settled as the hollow echo")
            self.assertTrue(payload.get("sources"),
                            f"{name} produced no sources")
            self.assertTrue(payload.get("summary"),
                            f"{name} produced no summary")

    @staticmethod
    def store_outputs(store, mission):
        return store.get(mission.id).state.get("outputs") or {}


if __name__ == "__main__":
    unittest.main()
