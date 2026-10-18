"""Tests for the three selectable agent modes: adaptive / autonomous / planning.

Failing-first: each test pins a DISTINCT, measurable behaviour per mode —
adaptive learns from outcomes, autonomous self-initiates with no prompt,
planning plans before executing and replans on failure — plus selection,
per-chat defaults, and mode combinations.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nomorals.agents import modes as M


# ── selection: /agent args ──────────────────────────────────────────────

def test_parse_agent_command_single_mode_sets_exactly():
    op = M.parse_agent_command("planning")
    assert op["op"] == "set"
    assert op["modes"] == frozenset({"planning"})


def test_parse_agent_command_combination():
    op = M.parse_agent_command("adaptive+planning")
    assert op["op"] == "set"
    assert op["modes"] == frozenset({"adaptive", "planning"})
    op2 = M.parse_agent_command("autonomous, planning")
    assert op2["modes"] == frozenset({"autonomous", "planning"})


def test_parse_agent_command_add_remove_off_status():
    assert M.parse_agent_command("+autonomous")["op"] == "add"
    assert M.parse_agent_command("-planning")["op"] == "remove"
    assert M.parse_agent_command("off")["op"] == "clear"
    assert M.parse_agent_command("")["op"] == "status"
    assert M.parse_agent_command("status")["op"] == "status"
    assert M.parse_agent_command("bogus-mode")["op"] == "usage"


def test_parse_mode_phrase_natural_language():
    assert M.parse_mode_phrase("use planning mode") == frozenset({"planning"})
    assert M.parse_mode_phrase("go autonomous") == frozenset({"autonomous"})
    assert M.parse_mode_phrase("adapt to me") == frozenset({"adaptive"})
    assert M.parse_mode_phrase("plan first, then execute") == frozenset({"planning"})
    assert M.parse_mode_phrase("work on your own") == frozenset({"autonomous"})
    assert M.parse_mode_phrase("use autonomous and planning modes") == frozenset(
        {"autonomous", "planning"})
    assert M.parse_mode_phrase("use all three modes") == frozenset(M.MODES)


def test_parse_mode_phrase_no_false_positive():
    assert M.parse_mode_phrase("I like planning my day") is None
    assert M.parse_mode_phrase("tell me about autonomous cars") is None
    assert M.parse_mode_phrase("hello") is None


# ── per-chat defaults + persistence ──────────────────────────────────────

def test_per_chat_default_isolated_and_persisted(tmp_path):
    store = M.ModeStore(tmp_path / "modes.json")
    store.set("telegram:1", {"planning"})
    assert store.get("telegram:1") == frozenset({"planning"})
    assert store.get("telegram:2") == frozenset()
    # a fresh store over the same file remembers chat 1's default
    store2 = M.ModeStore(tmp_path / "modes.json")
    assert store2.get("telegram:1") == frozenset({"planning"})
    assert store2.get("telegram:2") == frozenset()


def test_modes_apply_command_status_text(tmp_path):
    store = M.ModeStore(tmp_path / "modes.json")
    agent = M.AgentModes(store=store)
    reply = agent.apply_command("adaptive+planning", "telegram:1")
    assert "adaptive" in reply and "planning" in reply
    assert agent.modes_for("telegram:1") == frozenset({"adaptive", "planning"})
    assert "adaptive" in agent.apply_command("status", "telegram:1")
    agent.apply_command("off", "telegram:1")
    assert agent.modes_for("telegram:1") == frozenset()


# ── adaptive: learns from outcomes ───────────────────────────────────────

def test_adaptive_learns_from_outcomes_not_fixed_defaults():
    learner = M.AdaptiveLearner()
    strategies = ["direct", "stepwise", "verify-each"]
    # fresh learner: no preference — equal weights (nothing learned yet)
    before = learner.weights(strategies, "ctx")
    assert max(before.values()) - min(before.values()) < 1e-9
    # real results: stepwise wins 4/5, direct loses 4/5
    for _ in range(4):
        learner.record_outcome("ctx", "stepwise", True)
        learner.record_outcome("ctx", "direct", False)
    learner.record_outcome("ctx", "stepwise", False)
    learner.record_outcome("ctx", "direct", True)
    ranked = learner.choose(strategies, "ctx")
    assert ranked[0][0] == "stepwise"
    assert ranked[-1][0] == "direct"
    after = learner.weights(strategies, "ctx")
    assert after["stepwise"] > after["direct"]
    assert abs(sum(after.values()) - 1.0) < 1e-9


def test_adaptive_avoids_failed_strategy_in_new_context_via_global():
    learner = M.AdaptiveLearner()
    # "direct" fails everywhere; a brand-new context still avoids it
    for _ in range(3):
        learner.record_outcome("ctx-a", "direct", False)
        learner.record_outcome("ctx-a", "stepwise", True)
    ranked = learner.choose(["direct", "stepwise"], "ctx-brand-new")
    assert ranked[0][0] == "stepwise"


def test_adaptive_style_hints_from_chat_profile():
    hints = M.AdaptiveLearner.style_hints({
        "vibe_label": "warm",
        "activity_label": "buzzing",
        "topics": ["trading", "music"],
    })
    assert hints["vibe"] == "warm"
    assert hints["pace"] == "buzzing"
    assert hints["topics"] == ["trading", "music"]
    assert M.AdaptiveLearner.style_hints({}) == {}


# ── autonomous: self-initiates without a prompt ──────────────────────────

def test_autonomous_tick_self_initiates_without_prompt():
    engine = M.AutonomousEngine()
    calls: list[str] = []

    def _source(ctx):
        calls.append("ran")
        return {"label": "check inbox", "action": lambda c: "inbox ok"}

    engine.add_source("inbox-watch", _source, priority=0.9)
    done = engine.tick({})
    assert calls == ["ran"]  # nobody prompted it — tick() alone started work
    assert len(done) == 1
    assert done[0].status == "done"
    assert done[0].result == "inbox ok"
    assert engine.pending() == []


def test_autonomous_lifecycle_failed_initiative_and_rotation_adapts():
    engine = M.AutonomousEngine()
    log: list[str] = []

    def _flaky(ctx):
        return {"label": "flaky", "action": lambda c: (_ for _ in ()).throw(
            RuntimeError("boom"))}

    def _steady(ctx):
        return {"label": "steady", "action": lambda c: "ok"}

    engine.add_source("flaky", _flaky, priority=0.9)
    engine.add_source("steady", _steady, priority=0.5)
    engine.tick({})
    failed = [i for i in engine.history() if i.source == "flaky"][0]
    assert failed.status == "failed"
    assert "boom" in (failed.error or "")
    # rotation learned: the failing source drops below the steady one
    order = engine.rotation_order(["flaky", "steady"], "autonomous")
    assert order[0] == "steady"


def test_autonomous_tick_with_no_sources_does_nothing():
    engine = M.AutonomousEngine()
    assert engine.tick({}) == []


# ── planning: plan first, verify each step, replan on failure ───────────

def _decompose(goal, evidence=None):
    return [
        {"name": "gather", "description": "gather inputs", "action": "wait",
         "parameters": {"seconds": 0}, "depends_on": []},
        {"name": "build", "description": "build it", "action": "wait",
         "parameters": {"seconds": 0}, "depends_on": ["step_1"]},
    ]


def _execute(step, ctx):
    ctx.setdefault("executed", []).append(step.name)
    return f"{step.name}-done"


def test_planning_plan_emitted_before_any_execution():
    runner = M.PlanningRunner()
    events: list[str] = []

    def _on_plan(plan):
        events.append(f"plan:{len(plan.steps)}")

    def _exec(step, ctx):
        events.append(f"exec:{step.name}")
        return "ok"

    result = asyncio.run(runner.run(
        "do the thing", decompose=_decompose, execute_step=_exec,
        on_plan=_on_plan))
    assert result.success is True
    assert events[0].startswith("plan:")
    assert events[1] == "exec:gather"
    assert events[2] == "exec:build"
    # dependency order: gather before build
    assert events.index("exec:gather") < events.index("exec:build")


def test_planning_replans_when_verification_fails():
    runner = M.PlanningRunner()
    attempts = {"n": 0}
    plans_seen: list[int] = []

    def _decompose_retry(goal, evidence=None):
        attempts["n"] += 1
        if evidence is None:
            return [{"name": "first-try", "description": "bad approach",
                     "action": "wait", "parameters": {"seconds": 0},
                     "depends_on": []}]
        # replan sees the failure evidence and picks a different approach
        assert "first-try" in json.dumps(evidence)
        return [{"name": "second-try", "description": "fixed approach",
                 "action": "wait", "parameters": {"seconds": 0},
                 "depends_on": []}]

    def _verify(step, result):
        return step.name != "first-try"  # first approach fails verification

    result = asyncio.run(runner.run(
        "do it", decompose=_decompose_retry, execute_step=_execute,
        verify_step=_verify,
        on_plan=lambda p: plans_seen.append(len(p.steps))))
    assert result.success is True
    assert attempts["n"] == 2  # planned, failed verify, replanned
    assert len(plans_seen) == 2


def test_planning_invalid_plan_never_executes():
    runner = M.PlanningRunner()
    executed: list[str] = []

    def _cyclic(goal, evidence=None):
        return [
            {"name": "a", "description": "a", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": ["step_2"]},
            {"name": "b", "description": "b", "action": "wait",
             "parameters": {"seconds": 0}, "depends_on": ["step_1"]},
        ]

    with pytest.raises(ValueError, match="cycle"):
        asyncio.run(runner.run(
            "bad", decompose=_cyclic,
            execute_step=lambda s, c: executed.append(s.name)))
    assert executed == []


# ── combinations ─────────────────────────────────────────────────────────

def test_autonomous_plus_planning_tick_plans_first():
    agent = M.AgentModes()
    agent.set_modes("telegram:1", {"autonomous", "planning"})
    events: list[str] = []

    def _decompose(goal, evidence=None):
        return [{"name": "step-one", "description": "one", "action": "wait",
                 "parameters": {"seconds": 0}, "depends_on": []}]

    def _source(ctx):
        return {"label": "initiative work", "goal": "do initiative work",
                "decompose": _decompose,
                "execute_step": lambda s, c: events.append(f"exec:{s.name}"),
                "on_plan": lambda p: events.append("plan")}

    agent.autonomous.add_source("worker", _source, priority=0.9)
    done = agent.tick("telegram:1", {})
    assert len(done) == 1
    assert done[0].status == "done"
    assert events[0] == "plan"  # plan-first even for self-initiated work
    assert events[1] == "exec:step-one"


def test_autonomous_without_mode_does_not_tick():
    agent = M.AgentModes()
    agent.set_modes("telegram:1", {"planning"})
    agent.autonomous.add_source(
        "worker", lambda ctx: {"label": "x", "action": lambda c: "y"})
    assert agent.tick("telegram:1", {}) == []


def test_adaptive_plus_autonomous_tick_feeds_learner():
    agent = M.AgentModes()
    agent.set_modes("telegram:1", {"adaptive", "autonomous"})

    def _bad(ctx):
        return {"label": "bad", "action": lambda c: (_ for _ in ()).throw(
            RuntimeError("nope"))}

    agent.autonomous.add_source("bad", _bad, priority=0.9)
    agent.tick("telegram:1", {})
    ranked = agent.learner.choose(["bad", "other"], "autonomous")
    assert ranked[0][0] == "other"


def test_handle_task_planning_vs_direct_paths_differ(tmp_path):
    agent = M.AgentModes(store=M.ModeStore(tmp_path / "m.json"))
    events: list[str] = []

    def _decompose(goal, evidence=None):
        return [{"name": "planned-step", "description": "d", "action": "wait",
                 "parameters": {"seconds": 0}, "depends_on": []}]

    def _exec(step, ctx):
        events.append(f"exec:{step.name}")
        return "ok"

    # planning mode: plan emitted first
    agent.set_modes("c1", {"planning"})
    res = asyncio.run(agent.handle_task(
        "goal", "c1", decompose=_decompose, execute_step=_exec,
        on_plan=lambda p: events.append("plan")))
    assert res.success is True
    assert events == ["plan", "exec:planned-step"]
    # no modes: direct single-step execution, no plan event
    events.clear()
    res = asyncio.run(agent.handle_task(
        "goal", "c2", decompose=_decompose, execute_step=_exec,
        on_plan=lambda p: events.append("plan")))
    assert res.success is True
    assert events == ["exec:goal"]


def test_handle_task_adaptive_picks_learned_strategy(tmp_path):
    agent = M.AgentModes(store=M.ModeStore(tmp_path / "m.json"))
    agent.set_modes("c1", {"adaptive"})
    for _ in range(3):
        agent.learner.record_outcome("default", "careful", True)
        agent.learner.record_outcome("default", "quick", False)
    seen: list[str] = []

    def _exec(step, ctx):
        seen.append(ctx.get("strategy"))
        return "ok"

    asyncio.run(agent.handle_task(
        "goal", "c1", execute_step=_exec,
        strategies=["quick", "careful"]))
    assert seen == ["careful"]


# ── coremind NL routing ──────────────────────────────────────────────────

def test_coremind_mode_intent():
    from nomorals.agents.coremind import understand

    intents = understand("use planning mode")
    assert intents and intents[0].kind == "mode"
    assert intents[0].target == "planning"
    intents = understand("go autonomous")
    assert intents and intents[0].kind == "mode"
    assert intents[0].target == "autonomous"
