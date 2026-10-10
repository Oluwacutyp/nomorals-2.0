"""Phase 9 Slice E — god-tier agent orchestration.

Real tests for breakable behavior (no model, no network):

- Blackboard reactive triggers (link/cancel/failure isolation)
- Blackboard persistence (save/load) + condition-based wait_for
- Supervisor blackboard events + supervised parallel team runs
- PanelDebate (N-position) + Debate.settle auto-arbitration
- CheckpointStore generic agent-state snapshots
- MasterOrchestrator checkpoint/resume + per-step debate
- MissionControl.launch portfolio → execution glue
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from nomorals.agents.base import Agent, AgentResult
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.checkpoints import CheckpointStore
from nomorals.agents.debate import (
    Critique,
    Debate,
    Issue,
    PanelDebate,
    Position,
    WorkArtifact,
)
from nomorals.agents.mission import MissionControl
from nomorals.agents.orchestrator import (
    MasterOrchestrator,
    Plan,
    PlanStep,
)
from nomorals.agents.runtime import HybridExecutor
from nomorals.agents.supervisor import RestartPolicy, Supervisor
from nomorals.agents.swarm import SwarmAgent, SwarmResult


# ── fakes ────────────────────────────────────────────────────────────────


class ScriptedAgent(Agent):
    """Agent whose work() follows a script: values succeed, exceptions fail."""

    role = "test"

    def __init__(self, name: str, script: list, **kw):
        super().__init__(name=name, **kw)
        self.script = list(script)
        self.calls = 0

    def work(self, task_input):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _null_context():
    ctx = SimpleNamespace()
    ctx.emit = lambda *a, **k: None
    return ctx


def _orch(**kw):
    kw.setdefault("executor", HybridExecutor(threads=4, use_processes=False))
    return MasterOrchestrator(None, **kw)


def _plan(*names: str, debate: bool = False) -> Plan:
    steps = []
    for i, name in enumerate(names):
        payload = {"debate": True} if debate else {}
        steps.append(PlanStep(
            name=name, goal=f"do {name}", role="execution",
            depends_on=[names[i - 1]] if i else [],
            payload=payload,
        ))
    return Plan(goal="test goal", steps=steps)


# ── Blackboard reactive triggers ─────────────────────────────────────────


def test_link_fires_on_match_only():
    board = Blackboard()
    seen: list[str] = []
    event = threading.Event()
    sub = board.link("mission.*.done", lambda entry: (seen.append(entry.key), event.set()))
    try:
        board.post("mission.alpha.done", {"ok": True}, author="a")
        board.post("mission.alpha.started", {"ok": True}, author="a")
        assert event.wait(timeout=5)
        time.sleep(0.2)  # let any stray firing land
        assert seen == ["mission.alpha.done"]
        assert sub.fired == 1
    finally:
        sub.cancel()
        board.close()


def test_link_callback_failure_does_not_break_post():
    board = Blackboard()
    good: list[str] = []
    done = threading.Event()

    def bad(entry):
        raise RuntimeError("agent exploded")

    def also_good(entry):
        good.append(entry.key)
        done.set()

    s1 = board.link("x.*", bad)
    s2 = board.link("x.*", also_good)
    try:
        entry = board.post("x.1", {"v": 1}, author="t")
        assert entry.key == "x.1"  # the post itself survived
        assert done.wait(timeout=5)
        assert good == ["x.1"]
        assert s1.fired == 0  # a raising callback is not counted
        assert s2.fired == 1
    finally:
        s1.cancel()
        s2.cancel()
        board.close()


def test_link_cancel_stops_firing():
    board = Blackboard()
    seen: list[str] = []
    sub = board.link("y.*", lambda entry: seen.append(entry.key))
    fired_event = threading.Event()

    # wait until the first firing lands, then cancel
    board.post("y.1", 1, author="t")
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen == ["y.1"]
    sub.cancel()
    assert sub.cancelled
    board.post("y.2", 2, author="t")
    time.sleep(0.3)
    assert seen == ["y.1"]
    board.close()


def test_wait_for_wakes_on_post():
    board = Blackboard()

    def poster():
        time.sleep(0.15)
        board.post("awaited.key", "hello", author="t")

    thread = threading.Thread(target=poster, daemon=True)
    thread.start()
    started = time.perf_counter()
    value = board.wait_for("awaited.key", timeout=5.0)
    elapsed = time.perf_counter() - started
    thread.join(timeout=5)
    assert value == "hello"
    assert elapsed < 4.0  # woke on the post, not on the timeout
    board.close()


def test_wait_for_timeout_returns_none():
    board = Blackboard()
    assert board.wait_for("never.posted", timeout=0.2) is None
    board.close()


def test_blackboard_save_load_roundtrip(tmp_path):
    board = Blackboard()
    board.post("k1", {"a": 1}, author="amy", topic="t1")
    board.post("k2", [1, 2, 3], author="bob", topic="t1", ttl=60.0)
    board.post("k3", {"non": "json", "obj": object()}, author="x")  # coerced
    path = tmp_path / "board.json"
    saved = board.save(path)
    assert saved["ok"] and saved["entries"] == 3

    board2 = Blackboard()
    loaded = board2.load(path)
    assert loaded["ok"] and loaded["entries"] == 3
    assert board2.get("k1") == {"a": 1}
    assert board2.get("k2") == [1, 2, 3]
    assert isinstance(board2.get("k3")["obj"], str)  # coerced, not crashed
    board.close()
    board2.close()


def test_blackboard_load_does_not_fire_watchers(tmp_path):
    board = Blackboard()
    board.post("w.1", "v", author="t")
    path = tmp_path / "snap.json"
    assert board.save(path)["ok"]
    board2 = Blackboard()
    fired: list[str] = []
    board2.watch("w.*", lambda entry: fired.append(entry.key))
    board2.load(path)
    assert board2.get("w.1") == "v"
    assert fired == []  # restore is recovery, not new work
    board.close()
    board2.close()


# ── Supervisor ───────────────────────────────────────────────────────────


def test_supervisor_events_posted_to_blackboard():
    board = Blackboard()
    policy = RestartPolicy(max_restarts=1, backoff_base=0.0)
    sup = Supervisor(policy=policy)
    sup.attach_blackboard(board, topic="supv")
    agent = ScriptedAgent("flaky", [RuntimeError("boom"), "recovered"])
    result = sup.run_agent(agent)
    assert result.ok
    assert sup.stats["restarts"] == 1
    keys = board.keys("supervisor.restart.*")
    assert keys, "expected a restart event on the blackboard"
    payload = board.get(keys[0])
    assert payload["kind"] == "restart"
    assert payload["subject"] == "flaky"
    board.close()


def test_supervisor_run_all_watches_the_team():
    policy = RestartPolicy(max_restarts=2, backoff_base=0.0)
    sup = Supervisor(policy=policy)
    agents = {
        "steady": ScriptedAgent("steady", ["fine"]),
        "flaky": ScriptedAgent("flaky",
                               [RuntimeError("first try"), "second try ok"]),
        "doomed": ScriptedAgent("doomed", [RuntimeError("always")]),
    }
    team = sup.run_all(agents)
    assert set(team.results) == {"steady", "flaky", "doomed"}
    assert team.results["steady"].ok
    assert team.results["flaky"].ok  # restarted once, then recovered
    assert not team.results["doomed"].ok  # gave up within policy
    assert team.succeeded == 2 and team.failed == 1
    assert not team.ok
    # doomed: 1 initial + 2 restarts, then give_up
    assert agents["doomed"].calls == 3
    assert sup.stats["give_ups"] == 1


def test_supervisor_run_all_uses_factories_for_restart():
    policy = RestartPolicy(max_restarts=1, backoff_base=0.0)
    sup = Supervisor(policy=policy)
    created: list[str] = []

    def factory():
        created.append("fresh")
        return ScriptedAgent("reborn", ["fresh ok"])

    stale = ScriptedAgent("reborn", [RuntimeError("stale state")])
    team = sup.run_all({"reborn": stale}, factories={"reborn": factory})
    assert team.results["reborn"].ok
    assert created == ["fresh"]


# ── Debate ───────────────────────────────────────────────────────────────


def _position(label: str, score_marker: str) -> Position:
    return Position(label=label,
                    artifact=WorkArtifact(content=f"content-{label}",
                                          summary=f"summary-{label}"),
                    claim=f"claim-{label}", evidence=f"ev-{label}",
                    risk=f"risk-{label}")


def test_panel_debate_picks_highest_scored_position():
    scores = {"a": 90.0, "b": 70.0, "c": 80.0}

    def critic(artifact: WorkArtifact, rubric) -> Critique:
        label = artifact.content.split("content-")[1]
        return Critique(verdict="approve", score=scores[label])

    panel = PanelDebate(
        position_fns={label: (lambda l=label: _position(l, l))
                      for label in ("a", "b", "c")},
        critic_fn=critic,
        max_rounds=1,
    )
    result = panel.run("pick the best")
    assert result.verdict == "decided"
    assert result.decided
    assert result.winner is not None and result.winner.label == "a"
    assert result.scores["a"] == 90.0


def test_panel_debate_tie_escalates_to_judge():
    def critic(artifact: WorkArtifact, rubric) -> Critique:
        return Critique(verdict="approve", score=80.0)  # exact tie

    def judge(positions, rubric):
        # judge picks "b" deterministically
        return {"decision": {"label": "b"}, "rationale": "judge says b"}

    panel = PanelDebate(
        position_fns={"a": lambda: _position("a", "a"),
                      "b": lambda: _position("b", "b")},
        critic_fn=critic,
        max_rounds=1,
        tie_margin=5.0,
        judge_fn=judge,
    )
    result = panel.run("break the tie")
    assert result.verdict == "tie_arbitrated"
    assert result.winner is not None and result.winner.label == "b"
    assert result.arbitration is not None


def test_panel_debate_no_contenders():
    def dead():
        raise RuntimeError("nope")

    panel = PanelDebate(position_fns={"a": dead}, critic_fn=lambda a, r: None)
    result = panel.run("nothing to debate")
    assert result.verdict == "no_contenders"
    assert result.winner is None
    assert not result.decided


def test_debate_settle_arbitrates_unresolved():
    def coder(brief, feedback, artifact):
        return artifact or WorkArtifact(content="v1", summary="s1")

    def critic(artifact, rubric):
        return Critique(verdict="request_changes", score=40.0,
                        notes="not good enough",
                        issues=[Issue(id="i1", severity="major",
                                      location="l1", detail="d1")])

    debate = Debate(coder_fn=coder, critic_fn=critic, max_rounds=1)
    result, arbitration = debate.settle(
        "do it",
        judge_fn=lambda positions, rubric: {
            "decision": positions[0]["label"],
            "rationale": "coder wins by default"},
    )
    assert result.verdict == "unresolved"
    assert arbitration is not None
    assert arbitration.decision == "coder"


# ── CheckpointStore agent-state snapshots ────────────────────────────────


def test_agent_state_roundtrip(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    payload = {"goal": "g", "tasks": {"a": {"state": "done"}}}
    saved = store.save_state("run-1", payload, label="mid")
    assert saved["ok"] and saved["run_id"] == "run-1"

    states = store.list_states("run-1")
    assert len(states) == 1
    assert states[0]["label"] == "mid"

    record = store.load_state(states[0]["state_id"])
    assert record is not None
    assert record["payload"] == payload
    assert record["version"] == 1

    assert store.delete_state(states[0]["state_id"])
    assert store.list_states("run-1") == []


def test_agent_state_version_mismatch_refused(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    d = tmp_path / "ckpts" / "agent_state"
    d.mkdir(parents=True)
    (d / "state_run-1_20990101_000000_ab.json").write_text(json.dumps({
        "version": 999, "state_id": "x", "run_id": "run-1",
        "payload": {}}))
    assert store.load_state("state_run-1_20990101_000000_ab") is None


def test_agent_state_prunes_to_keep(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    for i in range(8):
        store.save_state("run-9", {"i": i})
    states = store.list_states("run-9")
    assert len(states) == 5  # AGENT_STATE_KEEP


# ── MasterOrchestrator checkpoint / resume ──────────────────────────────


def test_run_persists_checkpoints(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    orch = _orch()
    calls: list[str] = []

    def handler(task):
        calls.append(task.name)
        return {"step": task.name}

    result = orch.run("goal", plan=_plan("a", "b"),
                      default_handler=handler,
                      checkpoint_store=store, run_id="run-abc",
                      reflect=False)
    assert result.ok
    states = store.list_states("run-abc")
    assert len(states) >= 2  # one snapshot per settled task
    record = store.load_state(states[0]["state_id"])
    payload = record["payload"]
    assert payload["goal"] == "goal"
    assert payload["tasks"]["a"]["state"] == "done"
    assert payload["tasks"]["b"]["state"] == "done"
    assert len(payload["steps"]) == 2
    assert calls == ["a", "b"]


def test_resume_skips_done_tasks(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    calls: list[str] = []

    def failing(task):
        calls.append(task.name)
        if task.name == "b":
            raise RuntimeError("boom")
        return {"step": task.name}

    orch1 = _orch()
    result1 = orch1.run("goal", plan=_plan("a", "b"),
                        default_handler=failing,
                        checkpoint_store=store, run_id="run-xyz",
                        reflect=False)
    assert not result1.ok  # b failed (initial + supervised retries)
    assert calls.count("a") == 1
    b_calls_run1 = calls.count("b")
    assert b_calls_run1 == 4  # 1 initial + 3 supervised retries (policy default)

    # New orchestrator (fresh supervisor), b now succeeds.
    def fixed(task):
        calls.append(task.name)
        return {"step": task.name, "fixed": True}

    orch2 = _orch()
    result2 = orch2.resume("run-xyz", store, default_handler=fixed,
                           reflect=False)
    assert result2.ok
    assert calls.count("a") == 1, "a was DONE — resume must not re-run it"
    assert calls.count("b") == b_calls_run1 + 1  # one supervised retry of b


def test_resume_unknown_run_raises(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "ckpts")
    orch = _orch()
    with pytest.raises(ValueError, match="no checkpoint snapshots"):
        orch.resume("nope", store)


# ── MasterOrchestrator per-step debate ──────────────────────────────────


def test_step_debate_approves_and_stamps():
    def critic(artifact: WorkArtifact, rubric) -> Critique:
        assert rubric  # rubric actually passed through
        return Critique(verdict="approve", score=95.0)

    orch = _orch(debate_critic=critic)
    result = orch.run("goal", plan=_plan("a", debate=True),
                      default_handler=lambda task: {"answer": 42},
                      reflect=False)
    assert result.ok
    stamped = result.report.results["a"]
    assert stamped["answer"] == 42
    assert stamped["debate_verdict"] == "approved"
    assert stamped["debate_score"] == 95.0


def test_step_debate_rejects_fails_the_step():
    def critic(artifact: WorkArtifact, rubric) -> Critique:
        return Critique(verdict="reject", score=5.0, notes="terrible")

    orch = _orch(debate_critic=critic)
    result = orch.run("goal", plan=_plan("a", debate=True),
                      default_handler=lambda task: {"answer": 42},
                      reflect=False)
    assert not result.ok
    assert "debate rejected" in result.report.failures["a"]


def test_step_debate_without_model_passes_through_unstamped():
    orch = _orch()  # context=None → no router → no model critic
    result = orch.run("goal", plan=_plan("a", debate=True),
                      default_handler=lambda task: {"answer": 7},
                      reflect=False)
    assert result.ok
    assert result.report.results["a"] == {"answer": 7}


# ── SwarmAgent state save / resume ──────────────────────────────────────


def test_swarm_save_and_resume_without_rerunning_legs(tmp_path):
    agent = SwarmAgent(None, wall_seconds=60.0)
    first = SwarmResult(goal="g", subtasks=["s1", "s2"])
    first.legs = [
        {"subtask": "s1", "ok": True, "digest": "d1",
         "planned_by": "x", "steps": 1, "seconds": 1.0},
        {"subtask": "s2", "ok": True, "digest": "d2",
         "planned_by": "x", "steps": 1, "seconds": 1.0},
    ]
    agent._last_result = first
    path = str(tmp_path / "swarm.json")
    saved = agent.save_state(path)
    assert saved["ok"] and saved["legs"] == 2

    # Resume must not instantiate any DevonAgent: every leg is done.
    resumed = agent.resume(path, workers=2)
    assert resumed.ok
    assert [leg["digest"] for leg in resumed.legs] == ["d1", "d2"]
    assert "d1" in resumed.synthesis and "d2" in resumed.synthesis


def test_swarm_resume_rejects_foreign_state(tmp_path):
    agent = SwarmAgent(None)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 12345}))
    with pytest.raises(ValueError, match="not a swarm state file"):
        agent.resume(str(bad))


# ── MissionControl.launch ───────────────────────────────────────────────


def _mission_context():
    from nomorals.storage.db import Database

    ctx = SimpleNamespace()
    ctx.db = Database(":memory:")
    ctx.db.migrate()
    ctx.settings = SimpleNamespace()
    ctx.emit = lambda *a, **k: None
    return ctx


def test_mission_launch_executes_and_writes_back():
    ctx = _mission_context()
    mc = MissionControl(ctx)
    goal = mc.goals.create("test mission", plan=["first thing", "second thing"])

    orch = MasterOrchestrator(None, executor=HybridExecutor(
        threads=4, use_processes=False))
    ran: list[str] = []

    def handler(task):
        ran.append(task.name)
        return {"did": task.name}

    out = mc.launch(goal.id, orchestrator=orch,
                    default_handler=handler, reflect=False)
    assert out["ok"], out
    assert out["run_ok"]
    assert ran == ["gstep-0", "gstep-1"]
    assert [s["status"] for s in out["steps"]] == ["done", "done"]
    assert out["goal_completed"]

    refreshed = mc.goals.get(goal.id)
    assert refreshed.status == "done"


def test_mission_launch_records_failed_steps():
    ctx = _mission_context()
    mc = MissionControl(ctx)
    goal = mc.goals.create("failing mission", plan=["ok step", "bad step"])

    orch = MasterOrchestrator(None, executor=HybridExecutor(
        threads=4, use_processes=False))

    def handler(task):
        if task.name == "gstep-1":
            raise RuntimeError("kaput")
        return {"did": task.name}

    out = mc.launch(goal.id, orchestrator=orch,
                    default_handler=handler, reflect=False)
    assert out["ok"]
    assert not out["run_ok"]
    statuses = {s["id"]: s["status"] for s in out["steps"]}
    assert set(statuses.values()) == {"done", "blocked"}
    assert not out["goal_completed"]
    # the blocked step stays retryable: still has work left
    refreshed = mc.goals.get(goal.id)
    assert refreshed.status == "active"


def test_mission_launch_rejects_unknown_goal():
    ctx = _mission_context()
    mc = MissionControl(ctx)
    out = mc.launch("goal-nope")
    assert not out["ok"] and "no goal" in out["error"]
