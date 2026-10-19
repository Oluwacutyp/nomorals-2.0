"""Pass #2: brain / tool-loop spine — verified bugs, failing-first.

Bug A: correlation ids are lost on the tool loop's pool threads, so
``tool.call`` bus events (and anything joining on request_id/run_id)
carry empty ids for every loop-driven tool call.

Bug B: ``ToolCallingLoop._note_tool_weakness`` never fires — it reads
``self._workspace_dir``, which nothing ever sets, so tool failures never
reach the weakness ledger from the loop.

Bug C: ``DevonAgent._execute`` has no per-step timeout. The wall-clock
budget is only checked BETWEEN steps, so one hung tool hangs ``run()``
forever despite ``step_timeout`` existing.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

from nomorals.agents.partner.tool_loop import ToolCallingLoop
from nomorals.core.correlation import current_correlation
from nomorals.llm.base import LLMResponse


class _FakeLLM:
    """Scripted model: returns canned responses in order."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        text = self._texts[min(self.calls - 1, len(self._texts) - 1)]
        return LLMResponse(text=text, model="fake-test")


class _BaseFakeRegistry:
    def ranked_listing(self, query, capabilities=None, limit=40):
        return ""

    def prompt_listing(self, capabilities=None):
        return "- probe(): test tool"

    def get(self, name):
        return SimpleNamespace(name=name)

    def names(self):
        return ["probe"]


class _CorrRegistry(_BaseFakeRegistry):
    """Records the correlation triple seen INSIDE call() (pool thread)."""

    def __init__(self):
        self.seen = []

    def call(self, name, actor="system", **kwargs):
        self.seen.append(dict(current_correlation()))
        return SimpleNamespace(ok=True, value="pong")


def _tool_fence_text():
    return (
        "Let me probe.\n"
        "```tool\n"
        '{"tool": "probe", "args": {}}\n'
        "```\n"
    )


def test_tool_call_events_carry_correlation_ids():
    """Bug A: the loop sets a correlation context, but tools run on pool
    threads where contextvars do not propagate — the ids arrive empty."""
    reg = _CorrRegistry()
    loop = ToolCallingLoop(
        _FakeLLM([_tool_fence_text(), "Done."]), reg,
        max_iterations=4, tool_timeout_s=5.0, total_timeout_s=30.0)
    result = loop.run("probe the thing",
                      request_id="req-abc", run_id="run-xyz")
    assert result.ok, f"loop failed: {result.error}"
    assert reg.seen, "tool was never called"
    assert reg.seen[0].get("request_id") == "req-abc", (
        f"request_id lost on pool thread: {reg.seen[0]}")
    assert reg.seen[0].get("run_id") == "run-xyz", (
        f"run_id lost on pool thread: {reg.seen[0]}")


def test_tool_loop_reports_tool_failure_to_weakness_ledger():
    """Bug B: _note_tool_weakness reads a never-set _workspace_dir and
    returns early — tool failures never reach the weakness ledger."""
    from nomorals.storage.db import Database

    db = Database(":memory:")
    ctx = SimpleNamespace(db=db)
    reg = _BaseFakeRegistry()
    reg.context = ctx
    loop = ToolCallingLoop(_FakeLLM(["done"]), reg)
    loop._note_tool_weakness("probe", "boom")
    rows = db.query(
        "SELECT kind, subject FROM weaknesses WHERE kind = 'tool_failure'")
    assert rows, "no weakness recorded — _note_tool_weakness is a dead path"
    assert rows[0]["subject"] == "probe"


def test_devon_step_parks_slow_tool_as_background_job():
    """Bug C (superseded): one hung tool hung DevonAgent.run() forever.
    Now _execute never blocks — the tool parks as a background job with
    heartbeats, the run moves on, and the result is collected later."""
    from nomorals.agents.devon import DevonAgent

    ctx = SimpleNamespace(db=None, tools=None)
    agent = DevonAgent(ctx, step_timeout=0.3)

    def slow(args):
        time.sleep(2)
        return "slow-result"

    agent._tools["slow_tool"] = slow
    t0 = time.time()
    outcome = agent._execute("slow_tool", {})
    dt = time.time() - t0
    assert dt < 5, f"_execute blocked {dt:.1f}s instead of parking"
    assert outcome.parked is True
    assert outcome.job_id, "parked outcome must carry the job handle"
    assert outcome.ok is False  # parked is not a success
    # The job keeps running in the background; its result is collected.
    job = agent.jobs.collect(outcome.job_id, timeout=10.0)
    assert job is not None and job.status == "done"
    assert job.result == "slow-result"
    final = agent._job_outcome(job)
    assert final.ok is True
    assert "slow-result" in final.observation


def test_devon_run_parks_slow_step_and_drains_it():
    """run(): a slow step parks (the run moves on to the next step
    instead of blocking), then the end-of-run drain collects it — the
    digest sees the real result, not a parked placeholder."""
    from nomorals.agents.devon import DevonAgent

    ctx = SimpleNamespace(db=None, tools=None)
    agent = DevonAgent(ctx, step_timeout=30, max_steps=2)

    def slow(args):
        time.sleep(1.0)
        return "slow-done"

    def fast(args):
        return "fast-done"

    agent._tools["slow_tool"] = slow
    agent._tools["fast_tool"] = fast
    agent._plan = lambda task, memory: (
        [{"tool": "slow_tool", "args": {}},
         {"tool": "fast_tool", "args": {}}],
        "test", "test-model", "")
    result = agent.run("do things", chat_key="test")
    assert len(result.steps) == 2
    # The slow step was drained: real outcome, not parked.
    assert result.steps[0].ok is True
    assert result.steps[0].parked is False
    assert "slow-done" in result.steps[0].observation
    assert result.steps[1].ok is True
    assert "fast-done" in result.steps[1].observation
    # Nothing left parked behind.
    assert agent.jobs.list(status="running",
                           labels={"owner": "devon"}) == []


def test_devon_run_leaves_unfinished_job_parked_not_dropped():
    """A step slower than the run's wall-clock budget stays parked —
    visible in the registry for a later run, never silently dropped."""
    from nomorals.agents.devon import DevonAgent

    ctx = SimpleNamespace(db=None, tools=None)
    agent = DevonAgent(ctx, step_timeout=0.2, max_steps=1)

    def very_slow(args):
        time.sleep(30)
        return "eventually"

    agent._tools["very_slow_tool"] = very_slow
    agent._plan = lambda task, memory: (
        [{"tool": "very_slow_tool", "args": {}}], "test", "t", "")
    result = agent.run("do slow thing", chat_key="test")
    assert len(result.steps) == 1
    assert result.steps[0].parked is True
    assert result.steps[0].job_id
    parked = agent.jobs.list(status="running",
                             labels={"owner": "devon"})
    assert any(s["id"] == result.steps[0].job_id for s in parked), \
        "unfinished job must stay visible in the registry"
