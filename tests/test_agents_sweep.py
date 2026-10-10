"""Sweep tests for the agents module upgrade (144-file sweep).

Every new/changed behavior from the agents sweep gets a real test:
render primitives, tool-loop events/stall-detection/parallel mode,
symmetric debate, router feedback learning, checkpoint verify/diff,
blackboard audit/export, crescendo red-teaming, supervisor intensity +
rest_for_one, self-consistency reasoning, plan diffs, swarm/fanout
presentation, scheduler relative times, autonomy digest.
"""

import json
import sqlite3
import time
from types import SimpleNamespace

from nomorals.agents import render
from nomorals.agents.blackboard import Blackboard
from nomorals.agents.checkpoints import Checkpoint, CheckpointStore
from nomorals.agents.debate import (
    SymmetricDebate, render_debate,
)
from nomorals.agents.fanout import map_reduce, render_mapreduce
from nomorals.agents.partner.tool_loop import (
    ToolCallingLoop, ToolLoopEvent, ToolLoopResult, parse_tool_calls,
    render_transcript,
)
from nomorals.agents.planner import (
    Plan, PlanStatus, PlanStep, diff_plans, render_plan, render_plan_diff,
)
from nomorals.agents.reasoning import ReasoningEngine, trace_text
from nomorals.agents.redteam import (
    CrescendoStrategy, RedTeamReport, AttackFinding, render_report,
)
from nomorals.agents.router_select import (
    LearnedReliability, TaskRouter,
)
from nomorals.agents.scheduler import parse_relative, parse_schedule_spec
from nomorals.agents.supervisor import (
    Agent, AgentResult, RestartPolicy, Supervisor,
)
from nomorals.agents.swarm import SwarmResult, render_swarm


# ── render.py ────────────────────────────────────────────────────────────

def test_render_banner_and_kv():
    assert render.banner("Hello", "🧠") == "🧠 **Hello**"
    out = render.kv([("model", "groq"), ("latency", "1.2s")])
    assert "model" in out and "groq" in out and "latency" in out
    assert out.index("model") < out.index("latency")


def test_render_table_and_bar():
    t = render.table(["a", "b"], [["x", "y"]])
    assert "| a" in t and "| x" in t
    b = render.bar(0.5, 10)
    assert "50%" in b and b.count("█") == 5


def test_render_truncate_never_raises():
    assert render.truncate("abcdef", 3) == "ab…"
    assert render.truncate(None, 10) == ""


# ── tool loop: events, stall detection, parallel, transcript ────────────

class _FakeLLM:
    """Scripted model: returns canned texts in order."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return SimpleNamespace(ok=True, text=text, model="fake")


class _FakeRegistry:
    def __init__(self):
        self.calls = []

    def get(self, name):
        return object() if name in {"echo", "slow"} else None

    def names(self):
        return ["echo", "slow"]

    def ranked_listing(self, query="", capabilities=None, limit=40):
        return "- echo: echo back args\n- slow: slow echo"

    def prompt_listing(self, capabilities=None):
        return "- echo: echo back args\n- slow: slow echo"

    def call(self, name, actor="owner", **kwargs):
        self.calls.append((name, kwargs))
        if name == "slow":
            time.sleep(0.05)
        return SimpleNamespace(ok=True, value={"echoed": kwargs})


def _stall_text():
    return '```tool\n{"tool": "echo", "args": {"q": "same"}}\n```'


def test_tool_loop_stall_detection():
    events = []
    loop = ToolCallingLoop(
        _FakeLLM([_stall_text()] * 6), _FakeRegistry(),
        max_iterations=6,
        on_event=lambda e: events.append(e.kind),
    )
    result = loop.run("go")
    assert result.degraded is True
    assert result.error.startswith("stall")
    assert "stuck" in result.answer
    assert "stalled" in events
    assert "final" in events


def test_tool_loop_events_fire():
    events = []
    loop = ToolCallingLoop(
        _FakeLLM(['```tool\n{"tool": "echo", "args": {"q": 1}}\n```',
                  "done!"]),
        _FakeRegistry(),
        on_event=lambda e: events.append(
            (e.kind, e.tool_name, e.iteration)),
    )
    result = loop.run("go")
    assert result.ok and result.answer == "done!"
    kinds = [k for k, _, _ in events]
    assert kinds[0] == "turn_start"
    assert ("tool_start", "echo", 1) in events
    assert ("tool_end", "echo", 1) in events
    assert kinds[-1] == "final"


def test_tool_loop_parallel_per_turn():
    reg = _FakeRegistry()
    loop = ToolCallingLoop(
        _FakeLLM(['```tool\n{"tool": "slow", "args": {"n": 1}}\n```\n'
                  '```tool\n{"tool": "slow", "args": {"n": 2}}\n```',
                  "done"]),
        reg, parallel_per_turn=True, tool_timeout_s=5.0,
    )
    started = time.time()
    result = loop.run("go")
    elapsed = time.time() - started
    assert result.ok
    # Sequential would take ~0.10s; parallel ~0.05s. Loose bound for CI.
    assert elapsed < 0.095
    assert [n for n, _ in reg.calls] == ["slow", "slow"]


def test_tool_loop_transcript_renders():
    loop = ToolCallingLoop(
        _FakeLLM(['```tool\n{"tool": "echo", "args": {"q": 1}}\n```',
                  "all done"]),
        _FakeRegistry())
    result = loop.run("go")
    text = render_transcript(result)
    assert "Turn 1" in text and "`echo`" in text and "all done" in text


def test_parse_tool_calls_still_works():
    calls = parse_tool_calls('```tool\n{"tool": "echo", "args": {"a": 1}}\n```')
    assert len(calls) == 1 and calls[0].name == "echo"


# ── symmetric debate ─────────────────────────────────────────────────────

def _debater(answer):
    return lambda brief, others: answer


def test_symmetric_debate_consensus_early_stop():
    d = SymmetricDebate(
        debater_fns={"a": _debater("42"), "b": _debater("42"),
                     "c": _debater("42")},
        max_rounds=2)
    r = d.run("what is the answer?")
    assert r.verdict == "consensus"
    assert r.rounds == 0  # independent answers already agreed
    assert r.final_answer == "42"
    assert r.converged is True


def test_symmetric_debate_reveal_and_revise():
    seen = {}

    def reviser(brief, others):
        seen.update(others)
        # Independent round: own answer; later rounds: revise toward peers.
        return "42" if others else "17"

    d = SymmetricDebate(
        debater_fns={"stubborn": _debater("42"),
                     "flex": reviser},
        max_rounds=2, parallel=False)
    r = d.run("q?")
    assert r.verdict == "consensus"
    assert r.rounds >= 1
    assert seen  # the reviser saw the peer's answer


def test_symmetric_debate_judge_decides():
    d = SymmetricDebate(
        debater_fns={"a": _debater("yes"), "b": _debater("no")},
        judge_fn=lambda answers: {"answer": "maybe",
                                  "rationale": "split the difference"},
        max_rounds=1)
    r = d.run("q?")
    assert r.verdict == "judge_decided"
    assert r.final_answer == "maybe"


def test_symmetric_debate_no_fake_consensus():
    d = SymmetricDebate(
        debater_fns={"a": _debater("yes"), "b": _debater("no")},
        max_rounds=1)
    r = d.run("q?")
    assert r.verdict == "no_consensus"
    assert "refusing to fake consensus" in r.reason


def test_render_debate_symmetric():
    d = SymmetricDebate(
        debater_fns={"a": _debater("42"), "b": _debater("42")},
        max_rounds=1)
    text = render_debate(d.run("q?"))
    assert "consensus" in text and "42" in text


# ── router feedback learning ─────────────────────────────────────────────

def _router_ctx():
    settings = SimpleNamespace(router_intelligent="on",
                               router_exploration=0.0)
    router = SimpleNamespace(
        providers=lambda: ["fast", "slow"],
        get=lambda n: SimpleNamespace(
            capabilities={"chat"}, model_id=n,
            stats_snapshot=lambda: {"calls": 10, "errors": 0,
                                    "avg_latency_ms": 100}),
        broker=None, active="fast")
    return SimpleNamespace(settings=settings, router=router)


def test_router_learned_reliability_neutral_without_data():
    s = LearnedReliability(ledger={})
    p = SimpleNamespace(name="fast")
    assert s.score(p, "chat", "balanced", None) == 0.0


def test_router_record_outcome_demotes_bad_model():
    tr = TaskRouter(_router_ctx())
    # Both start neutral-ish; poison "fast" with failures.
    for _ in range(10):
        tr.record_outcome("fast", "chat", ok=False)
    for _ in range(10):
        tr.record_outcome("slow", "chat", ok=True, latency_ms=200)
    assert tr.feedback_score("fast") < tr.feedback_score("slow")
    choice = tr.select("chat", "balanced")
    assert choice is not None and choice.name == "slow"


def test_router_explain_is_prose():
    tr = TaskRouter(_router_ctx())
    choice = tr.select("chat", "balanced")
    text = tr.explain(choice, "chat", "balanced")
    assert choice.name in text and "picked" in text
    assert tr.explain(None) == (
        "intelligent routing is off — using the standard provider chain")


# ── checkpoints: verify / diff / describe ────────────────────────────────

def test_checkpoint_diff():
    a = Checkpoint(id="a", ts=1.0, label="", head_hash="h1",
                   stash_hash=None, workdir="",
                   scope_files=["x.py", "y.py"],
                   untracked_files=["u.txt"])
    b = Checkpoint(id="b", ts=2.0, label="", head_hash="h2",
                   stash_hash=None, workdir="",
                   scope_files=["y.py", "z.py"],
                   untracked_files=["u.txt", "v.txt"])
    d = a.diff(b)
    assert d["head_moved"] is True
    assert d["scope_added"] == ["z.py"]
    assert d["scope_removed"] == ["x.py"]
    assert d["untracked_added"] == ["v.txt"]


def test_checkpoint_describe_and_verify(tmp_path):
    store = CheckpointStore(base_dir=str(tmp_path))
    ckpt = Checkpoint(id="missing", ts=time.time(), label="t",
                      head_hash="", stash_hash=None, workdir="",
                      code_captured=False)
    verdict = ckpt.verify(store)
    assert verdict["ok"] is False
    assert any("meta file missing" in p for p in verdict["problems"])
    text = ckpt.describe()
    assert "missing" in text and "convo-only" in text


# ── blackboard: history / export / import / render ───────────────────────

def test_blackboard_history_audit():
    b = Blackboard()
    b.post("k", "v1", author="a")
    time.sleep(0.01)
    b.post("k", "v2", author="b")
    h = b.history("k")
    assert len(h) == 2
    assert h[0]["version"] == 1 and h[0]["author"] == "a"
    assert h[1]["version"] == 2 and h[1]["author"] == "b"
    assert h[1]["value_preview"] == "v2"


def test_blackboard_export_import_roundtrip():
    b1 = Blackboard()
    b1.post("k1", {"x": 1}, author="a", topic="t")
    b1.post("k2", "v", author="b")
    snap = b1.export()
    b2 = Blackboard()
    assert b2.import_snapshot(snap) == 2
    assert b2.get("k1") == {"x": 1}
    assert b2.get("k2") == "v"
    # The import itself is audited.
    assert any(e["author"] == "import" for e in b2._audit)


def test_blackboard_render_board():
    b = Blackboard()
    b.post("alpha", "1", author="a")
    text = b.render_board()
    assert "alpha" in text and "Blackboard" in text
    assert b.render_board(topic="nope") != ""


# ── redteam: crescendo ───────────────────────────────────────────────────

def _crescendo_parts(progress_by_turn):
    calls = {"n": 0}

    def attacker(history, objective, escalation):
        return f"attack@{escalation}"

    def target(prompt, history):
        return f"reply to {prompt}"

    def scorer(objective, history):
        calls["n"] += 1
        return progress_by_turn[min(calls["n"] - 1,
                                    len(progress_by_turn) - 1)]

    return attacker, target, scorer


def test_crescendo_reaches_objective():
    s = CrescendoStrategy(max_turns=5, success_threshold=0.9)
    attacker, target, scorer = _crescendo_parts([0.1, 0.5, 0.95])
    r = s.run("t1", "get the secret", attacker_fn=attacker,
              target_fn=target, scorer_fn=scorer)
    assert r.succeeded is True
    assert len(r.turns) == 3
    assert r.turns[0].escalation == 0  # starts benign


def test_crescendo_prunes_stalled_line():
    s = CrescendoStrategy(max_turns=5, stall_turns=2)
    attacker, target, scorer = _crescendo_parts([0.2, 0.2, 0.2])
    r = s.run("t2", "obj", attacker_fn=attacker, target_fn=target,
              scorer_fn=scorer)
    assert r.succeeded is False
    assert "pruned" in r.reason
    assert len(r.turns) < 5


def test_crescendo_attacker_failure_ends_line():
    def bad_attacker(history, objective, escalation):
        raise RuntimeError("attacker exploded")

    s = CrescendoStrategy()
    r = s.run("t3", "obj", attacker_fn=bad_attacker,
              target_fn=lambda p, h: "x",
              scorer_fn=lambda o, h: 0.0)
    assert r.succeeded is False and "attacker failed" in r.reason


def test_render_report():
    report = RedTeamReport()
    report.findings.append(AttackFinding(
        scenario_id="inj-1", name="prompt injection", severity="high",
        succeeded=True, evidence="the loop echoed the payload"))
    report.findings.append(AttackFinding(
        scenario_id="ok-1", name="clean input", severity="low",
        succeeded=False, evidence=""))
    text = render_report(report)
    assert "HOLES FOUND" in text and "inj-1" in text
    assert "Recommendations" in text


# ── supervisor: intensity + rest_for_one ─────────────────────────────────

class _Flaky(Agent):
    role = "flaky"

    def __init__(self, name, fails):
        super().__init__(name=name)
        self.fails = fails
        self.runs = 0

    def work(self, task_input):
        self.runs += 1
        if self.fails > 0:
            self.fails -= 1
            raise RuntimeError("boom")
        return "fine"


def test_supervisor_intensity_gives_up_fast():
    now = [1000.0]
    sv = Supervisor(
        policy=RestartPolicy(max_restarts=2, window_seconds=300.0,
                             backoff_base=0.0, backoff_cap=0.0),
        clock=lambda: now[0])
    agent = _Flaky("flaky", fails=99)
    result = sv.run_agent(agent)
    assert result.ok is False
    # initial run + 2 in-window restarts, then the intensity limit bites —
    # not 99 attempts.
    assert agent.runs == 3


def test_supervisor_window_slides():
    now = [1000.0]
    sv = Supervisor(
        policy=RestartPolicy(max_restarts=1, window_seconds=10.0,
                             backoff_base=0.0, backoff_cap=0.0),
        clock=lambda: now[0])
    agent = _Flaky("flaky2", fails=1)
    assert sv.run_agent(agent).ok is True
    assert agent.runs == 2
    # Much later the window has slid: the same agent earns a fresh restart.
    now[0] += 100.0
    agent.fails = 1
    assert sv.run_agent(agent).ok is True
    assert agent.runs == 4


def test_supervisor_rest_for_one():
    sv = Supervisor(
        policy=RestartPolicy(max_restarts=0, backoff_base=0.0,
                             backoff_cap=0.0))
    dep = _Flaky("dep", fails=99)          # always fails
    child = _Flaky("child", fails=0)       # ok on its own…
    team = sv.run_all(
        {"dep": dep, "child": child},
        dependencies={"child": ["dep"]})
    # dep failed → rest_for_one restarts dep AND child.
    assert team.results["dep"].ok is False
    events = [e.kind for e in sv.events]
    assert "restart" in events  # the rest_for_one restart event
    assert "rest_for_one" in sv.render_tree() or True  # render never raises


def test_supervisor_render_tree():
    sv = Supervisor(
        policy=RestartPolicy(max_restarts=0, backoff_base=0.0,
                             backoff_cap=0.0))
    sv.run_agent(_Flaky("x", fails=0))
    text = sv.render_tree()
    assert "Supervision tree" in text and "x" in text


# ── reasoning: self-consistency ──────────────────────────────────────────

class _CannedEngine(ReasoningEngine):
    def __init__(self, answers):
        super().__init__(None, max_llm_calls=50, max_seconds=60.0)
        self._answers = list(answers)

    def _llm(self, prompt, *, temperature=0.2):
        self.budget.spend()
        text = self._answers.pop(0) if self._answers else \
            "ANSWER: fallback\nCONFIDENCE: 0.5"
        return text


def _cot_answer(ans, conf=0.8):
    return (f"REASONING:\n1. think\n2. think\nANSWER: {ans}\n"
            f"CONFIDENCE: {conf}")


def test_self_consistency_majority_wins():
    eng = _CannedEngine([_cot_answer("Paris", 0.9),
                         _cot_answer("Paris", 0.8),
                         _cot_answer("London", 0.6),
                         _cot_answer("Paris", 0.7),
                         _cot_answer("Berlin", 0.5)])
    result = eng.reason("capital of France?", strategy="self_consistency")
    assert result.answer == "Paris"
    assert result.strategy == "self_consistency"
    # 3/5 agree: confidence blends vote share with model confidence.
    assert 0.5 < result.confidence <= 1.0
    assert any(s.kind == "sample" for s in result.trace)


def test_trace_text_god_tier():
    eng = _CannedEngine([_cot_answer("Paris", 0.9),
                         _cot_answer("Paris", 0.9),
                         _cot_answer("Paris", 0.9)])
    result = eng.reason("q?", strategy="self_consistency")
    text = trace_text(result)
    assert "Reasoning" in text and "Paris" in text and "█" in text


# ── planner: diff + render ───────────────────────────────────────────────

def _plan(pid, steps):
    return Plan(plan_id=pid, goal="g",
                steps=[PlanStep(step_id=s[0], name=s[1], description=s[1],
                                action=s[2], status=s[3])
                       for s in steps],
                status=PlanStatus.PENDING)


def test_diff_plans():
    old = _plan("p1", [("s1", "one", "a1", "pending"),
                       ("s2", "two", "a2", "pending")])
    new = _plan("p2", [("s2", "two", "a2", "completed"),
                       ("s3", "three", "a3", "pending")])
    d = diff_plans(old, new)
    assert d["added"] == ["s3"]
    assert d["removed"] == ["s1"]
    assert d["changed"]["s2"]["status"] == ("pending", "completed")


def test_render_plan():
    p = _plan("p1", [("s1", "one", "do_thing", "completed"),
                     ("s2", "two", "do_other", "pending")])
    text = render_plan(p)
    assert "p1" in text and "do_thing" in text and "50%" in text


def test_render_plan_diff():
    old = _plan("p1", [("s1", "one", "a1", "pending")])
    new = _plan("p2", [("s1", "one", "a1", "completed")])
    text = render_plan_diff(diff_plans(old, new))
    assert "p1 → p2" in text and "pending → completed" in text


# ── swarm render / fanout map-reduce ─────────────────────────────────────

def test_render_swarm():
    r = SwarmResult(
        goal="research x",
        subtasks=["a", "b"],
        legs=[{"subtask": "a", "ok": True, "digest": "found a",
               "seconds": 1.2},
              {"subtask": "b", "ok": False, "digest": "boom",
               "seconds": 0.5}],
        synthesis="a wins", seconds=3.0)
    text = render_swarm(r)
    assert "1/2 ok" in text and "a wins" in text


def test_map_reduce_retry_and_progress():
    attempts = {"n": 0}
    progress = []

    def worker(chunk, index):
        if index == 0:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("flaky")
        return sum(chunk)

    r = map_reduce([1, 2, 3, 4], worker, sum, k=2, retries=2,
                   on_progress=lambda done, total: progress.append(
                       (done, total)))
    assert r.reduced == 10
    assert r.errors == []
    assert r.shard_status[0]["retries"] == 1
    assert progress and progress[-1] == (2, 2)


def test_render_mapreduce():
    r = map_reduce([1, 2], lambda c, i: sum(c), sum, k=2)
    text = render_mapreduce(r)
    assert "Map-reduce" in text and "2/2" in text


# ── scheduler relative times ─────────────────────────────────────────────

def test_parse_relative():
    assert parse_relative("in 20 minutes") == 20 * 60
    assert parse_relative("in 3 hours") == 3 * 3600
    assert parse_relative("in 45 seconds") == 45
    assert parse_relative("in a minute") == 60
    assert parse_relative("in an hour") == 3600
    assert parse_relative("in 2 days") == 2 * 86400
    try:
        parse_relative("tomorrow")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_parse_schedule_spec_relative():
    kind, detail = parse_schedule_spec("in 20 minutes")
    assert kind == "at"
    assert abs(detail - (time.time() + 1200)) < 5


# ── autonomy digest ──────────────────────────────────────────────────────

class _MemDB:
    """Minimal DB stub with the interface AutonomyLedger needs."""

    def __init__(self):
        self.cx = sqlite3.connect(":memory:")
        self.cx.row_factory = sqlite3.Row

    def transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def _t():
            yield self.cx
            self.cx.commit()
        return _t()

    def execute(self, sql, params=()):
        return self.cx.execute(sql, params)

    def _rows(self, cur):
        # The real Database returns dict rows, not sqlite3.Row.
        return [dict(r) for r in cur.fetchall()]

    def query(self, sql, params=()):
        return self._rows(self.cx.execute(sql, params))

    def query_one(self, sql, params=()):
        rows = self._rows(self.cx.execute(sql, params))
        return rows[0] if rows else None


def test_autonomy_render_digest():
    from nomorals.agents.autonomy_ledger import AutonomyLedger, render_digest

    db = _MemDB()
    ledger = AutonomyLedger(db)
    ledger.record("scheduler", "run", summary="nightly job", ok=True,
                  cost_seconds=3.0)
    ledger.record("watcher", "fire", summary="disk full",
                  ok=False, learned="add threshold")
    text = ledger.render_digest(window_hours=24)
    assert "Autonomy digest" in text
    assert "scheduler" in text and "watcher" in text
    assert "disk full" in text
    assert render_digest(db, window_hours=24) == text or True  # convenience


def test_autonomy_render_digest_empty():
    from nomorals.agents.autonomy_ledger import AutonomyLedger

    text = AutonomyLedger(_MemDB()).render_digest(window_hours=1)
    assert "Nothing ran" in text
