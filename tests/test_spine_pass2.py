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


def test_devon_step_timeout_is_honest():
    """Bug C: one hung tool hangs DevonAgent.run() forever — _execute has
    no per-step timeout even though step_timeout exists."""
    from nomorals.agents.devon import DevonAgent

    ctx = SimpleNamespace(db=None, tools=None)
    agent = DevonAgent(ctx, step_timeout=0.3)

    def slow(args):
        time.sleep(10)
        return "never"

    agent._tools["slow_tool"] = slow
    t0 = time.time()
    outcome = agent._execute("slow_tool", {})
    dt = time.time() - t0
    assert dt < 5, f"no per-step timeout: blocked {dt:.1f}s on one tool"
    assert outcome.ok is False
    assert "timed out" in outcome.error
    assert "unknown" in outcome.error.lower(), (
        f"timeout must be marked outcome-unknown, got: {outcome.error!r}")
