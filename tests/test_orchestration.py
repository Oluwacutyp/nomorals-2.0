"""Unit tests for the agentic orchestration loop.

Uses mock LLM + mock tools — no network, no real model calls.
"""

import json

from nomorals.agents.orchestration.context import LoopMemory, StepRecord
from nomorals.agents.orchestration.loop import AgenticLoop, run_agentic
from nomorals.agents.orchestration.tools import ToolAdapter
from nomorals.core.policy import CapabilitySet
from nomorals.core.result import Err, Ok
from nomorals.llm.base import Message


# ── fakes ─────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeLLM:
    """Returns scripted think outputs in order."""

    def __init__(self, scripts: list[dict]):
        self.scripts = scripts
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        idx = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return FakeResponse(json.dumps(self.scripts[idx]))


class FakeRegistry:
    """Minimal registry stand-in with two tools."""

    def __init__(self):
        self._tools = {}
        self.calls = []

    def register(self, name, fn, description="", parameters=None):
        self._tools[name] = {
            "name": name,
            "fn": fn,
            "description": description,
            "capability": "",
            "parameters": parameters or {},
        }

    def get(self, name):
        return self._tools.get(name)

    def schemas(self, capabilities=None):
        return list(self._tools.values())

    def call(self, name, actor="system", capabilities=None, **kwargs):
        self.calls.append((name, kwargs))
        spec = self._tools.get(name)
        if spec is None:
            return Err(Exception(f"unknown tool {name!r}"))
        try:
            return Ok(spec["fn"](**kwargs))
        except Exception as exc:  # noqa: BLE001
            return Err(exc)


def make_loop(scripts, tools_setup=None):
    reg = FakeRegistry()
    (tools_setup or (lambda r: None))(reg)
    llm = FakeLLM(scripts)
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(llm, adapter, step_budget=10)
    return loop, reg, llm


# ── tests ─────────────────────────────────────────────────────────────


def test_think_act_observe_cycle():
    """One tool call, then respond with the result."""
    reg_scripts = [
        {"thought": "need the time", "action": "tool",
         "tool": "get_time", "args": {}},
        {"thought": "have the time", "action": "respond",
         "response": "It is noon."},
    ]
    loop, reg, llm = make_loop(
        reg_scripts,
        lambda r: r.register("get_time", lambda: "12:00", "current time"),
    )
    result = loop.run("what time is it?")
    assert result.success
    assert not result.budget_exhausted
    assert result.steps_taken == 2
    assert result.tools_called == ["get_time"]
    assert "noon" in result.response
    assert ("get_time", {}) in reg.calls


def test_tool_chaining():
    """Second tool uses the first tool's output."""
    seen_args = {}

    def search(query):
        return "result-42"

    def fetch(doc_id):
        seen_args["doc_id"] = doc_id
        return "full text"

    loop, reg, llm = make_loop(
        [
            {"thought": "search first", "action": "tool",
             "tool": "search", "args": {"query": "prices"}},
            {"thought": "fetch the doc", "action": "tool",
             "tool": "fetch", "args": {"doc_id": "result-42"}},
            {"thought": "done", "action": "respond",
             "response": "Got the full text."},
        ],
        lambda r: (
            r.register("search", search, "search docs", {"query": {}}),
            r.register("fetch", fetch, "fetch doc", {"doc_id": {}}),
        ),
    )
    result = loop.run("find prices")
    assert result.success
    assert result.tools_called == ["search", "fetch"]
    assert seen_args["doc_id"] == "result-42"


def test_failure_recovery_tries_different_tool():
    """After a tool fails, the loop tries another approach, not a blind retry."""

    def boom():
        raise RuntimeError("service down")

    loop, reg, llm = make_loop(
        [
            {"thought": "try primary", "action": "tool",
             "tool": "primary", "args": {}},
            {"thought": "primary failed, try backup", "action": "tool",
             "tool": "backup", "args": {}},
            {"thought": "done", "action": "respond",
             "response": "Used the backup."},
        ],
        lambda r: (
            r.register("primary", boom, "primary source"),
            r.register("backup", lambda: "backup data", "backup source"),
        ),
    )
    result = loop.run("get data")
    assert result.success
    assert result.tools_called == ["primary", "backup"]
    assert reg.calls == [("primary", {}), ("backup", {})]


def test_blind_retry_is_blocked():
    """The loop refuses to retry the identical failed call."""

    def boom(x):
        raise RuntimeError("always fails")

    # Script stubbornly retries the same call twice
    loop, reg, llm = make_loop(
        [
            {"thought": "try it", "action": "tool",
             "tool": "flaky", "args": {"x": 1}},
            {"thought": "try again same way", "action": "tool",
             "tool": "flaky", "args": {"x": 1}},
            {"thought": "give up gracefully", "action": "respond",
             "response": "Could not do it."},
        ],
        lambda r: r.register("flaky", boom, "flaky tool", {"x": {}}),
    )
    result = loop.run("do the thing")
    assert result.success
    # registry saw only ONE real call — the second was blocked by the loop
    assert reg.calls.count(("flaky", {"x": 1})) == 1


def test_step_budget_enforced():
    """A model that never responds hits the budget and gets an honest summary."""
    scripts = [
        {"thought": f"step {i}", "action": "tool",
         "tool": "noop", "args": {}}
        for i in range(20)
    ]
    loop, reg, llm = make_loop(
        scripts,
        lambda r: r.register("noop", lambda: "ok", "does nothing"),
    )
    loop.step_budget = 4
    result = loop.run("never-ending task")
    assert result.budget_exhausted
    assert not result.success
    assert result.steps_taken == 4
    assert "ran out of steps" in result.response


def test_ask_action_pauses_for_user():
    loop, reg, llm = make_loop(
        [
            {"thought": "ambiguous", "action": "ask",
             "response": "Which city?"},
        ]
    )
    result = loop.run("book a flight")
    assert result.asked_user
    assert result.question == "Which city?"


def test_unknown_tool_is_failure_not_crash():
    loop, reg, llm = make_loop(
        [
            {"thought": "hallucinate a tool", "action": "tool",
             "tool": "nope_not_real", "args": {}},
            {"thought": "ok that failed, answer anyway", "action": "respond",
             "response": "I don't have that tool."},
        ]
    )
    result = loop.run("do impossible")
    assert result.success
    assert "don't have that tool" in result.response


def test_garbage_model_output_recovers():
    """Non-JSON model output doesn't crash the loop."""

    class GarbageLLM(FakeLLM):
        def chat(self, messages, params=None, **kw):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("this is not json at all")
            return FakeResponse(json.dumps(
                {"thought": "recovered", "action": "respond",
                 "response": "Recovered fine."}))

    reg = FakeRegistry()
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(GarbageLLM([]), adapter, step_budget=5)
    result = loop.run("hello")
    assert result.success
    assert "Recovered fine" in result.response


def test_run_agentic_convenience_wrapper():
    reg = FakeRegistry()
    reg.register("echo", lambda text: text.upper(), "echo", {"text": {}})
    llm = FakeLLM([
        {"thought": "echo it", "action": "tool",
         "tool": "echo", "args": {"text": "hi"}},
        {"thought": "done", "action": "respond", "response": "HI"},
    ])
    result = run_agentic("say hi", llm=llm, registry=reg,
                         history=[("user", "hello"), ("assistant", "hey")])
    assert result.success
    assert result.response == "HI"


def test_memory_renders_compactly():
    mem = LoopMemory(user_message="test")
    mem.add_history("user", "old message")
    mem.record_step(StepRecord(step=1, thought="t", action="tool",
                               tool_name="x", observation="y" * 5000))
    rendered = mem.render()
    assert "USER REQUEST: test" in rendered
    assert "truncated" in rendered  # long observation was compacted
    assert len(rendered) < 5000
