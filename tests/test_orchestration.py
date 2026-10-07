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
    def __init__(self, text: str, ok: bool = True):
        self.text = text
        self.ok = ok


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


# ── relevance-ranked tool selection ───────────────────────────────────


def test_describe_for_ranks_relevant_tools_first():
    """With 50+ tools, the task-relevant tool surfaces to the top."""
    reg = FakeRegistry()
    for i in range(50):
        reg.register(f"zzz_tool_{i:02d}", lambda: "x",
                     f"unrelated utility number {i}")
    reg.register("weather_lookup", lambda: "sunny",
                 "get the current weather forecast for a city",
                 {"city": {}})
    # Pin to 40 to test pagination (profile default varies by machine).
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all(),
                          max_tools_in_prompt=40)
    listing = adapter.describe_for("what is the weather like today in Lagos?")
    lines = listing.splitlines()
    assert lines[0].startswith("- weather_lookup("), \
        f"relevant tool not ranked first: {lines[0]}"
    # all 51 tools still reachable across the listing pages
    assert "… and 11 more tools" in listing


def test_describe_for_empty_query_keeps_original_order():
    """No query → same behavior as the old describe()."""
    reg = FakeRegistry()
    reg.register("bravo", lambda: "b", "second tool")
    reg.register("alpha", lambda: "a", "first tool")
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    assert adapter.describe() == adapter.describe_for("")


def test_think_sees_relevant_tools():
    """The think prompt carries the relevance-ranked listing."""
    reg = FakeRegistry()
    for i in range(45):
        reg.register(f"aaa_filler_{i:02d}", lambda: "f", "filler tool")
    reg.register("btc_price", lambda: "90000", "get BTC price in USD")
    seen_prompts = []

    class SpyLLM(FakeLLM):
        def chat(self, messages, params=None, **kw):
            seen_prompts.append(messages[1].content)
            self.calls += 1
            return FakeResponse(json.dumps(
                {"thought": "done", "action": "respond",
                 "response": "ok"}))

    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(SpyLLM([]), adapter, step_budget=2)
    loop.run("check BTC price and alert me")
    prompt = seen_prompts[0]
    btc_pos = prompt.find("btc_price")
    filler_pos = prompt.find("aaa_filler_00")
    assert btc_pos != -1 and filler_pos != -1
    assert btc_pos < filler_pos, "relevant tool must appear before filler"


# ── parallel tool calls ───────────────────────────────────────────────


def test_parallel_tools_action():
    """One 'tools' action dispatches independent calls together."""
    loop, reg, llm = make_loop(
        [
            {"thought": "need both", "action": "tools",
             "calls": [
                 {"tool": "get_time", "args": {}},
                 {"tool": "get_weather", "args": {"city": "Lagos"}},
             ]},
            {"thought": "have both", "action": "respond",
             "response": "Time and weather fetched."},
        ],
        lambda r: (
            r.register("get_time", lambda: "12:00", "current time"),
            r.register("get_weather", lambda city: f"sunny in {city}",
                       "weather", {"city": {}}),
        ),
    )
    result = loop.run("time and weather")
    assert result.success
    assert result.tools_called == ["get_time", "get_weather"]
    # both calls happened, but only ONE budget step was consumed for the batch
    assert result.steps_taken == 2
    assert ("get_time", {}) in reg.calls
    assert ("get_weather", {"city": "Lagos"}) in reg.calls


def test_parallel_batch_uses_registry_call_many():
    """When the registry supports call_many, the batch goes through it."""

    class ParallelRegistry(FakeRegistry):
        def __init__(self):
            super().__init__()
            self.call_many_used = False

        def call_many(self, calls, max_workers=1, **common):
            from nomorals.core.result import Ok
            self.call_many_used = True
            return [Ok(f"parallel-{name}") for name, _ in calls]

    reg = ParallelRegistry()
    reg.register("alpha", lambda: "a", "alpha tool")
    reg.register("beta", lambda: "b", "beta tool")
    llm = FakeLLM([
        {"thought": "batch it", "action": "tools",
         "calls": [{"tool": "alpha", "args": {}},
                    {"tool": "beta", "args": {}}]},
        {"thought": "done", "action": "respond", "response": "batched"},
    ])
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(llm, adapter, step_budget=5)
    result = loop.run("batch test")
    assert result.success
    assert reg.call_many_used, "registry.call_many was not used"
    assert result.tools_called == ["alpha", "beta"]


def test_parallel_batch_skips_blind_retry():
    """A repeated failed call inside a batch is skipped, others proceed."""

    def boom():
        raise RuntimeError("down")

    loop, reg, llm = make_loop(
        [
            {"thought": "try solo", "action": "tool",
             "tool": "flaky", "args": {"x": 1}},
            {"thought": "batch with a repeat", "action": "tools",
             "calls": [
                 {"tool": "flaky", "args": {"x": 1}},  # blind retry → skip
                 {"tool": "solid", "args": {}},
             ]},
            {"thought": "done", "action": "respond", "response": "ok"},
        ],
        lambda r: (
            r.register("flaky", boom, "flaky", {"x": {}}),
            r.register("solid", lambda: "fine", "solid tool"),
        ),
    )
    result = loop.run("batch retry test")
    assert result.success
    # flaky called once (the batch repeat was blocked), solid ran
    assert reg.calls.count(("flaky", {"x": 1})) == 1
    assert ("solid", {}) in reg.calls


def test_malformed_tools_action_is_guided_failure():
    loop, reg, llm = make_loop(
        [
            {"thought": "bad batch", "action": "tools"},  # no calls list
            {"thought": "recover", "action": "respond", "response": "recovered"},
        ]
    )
    result = loop.run("bad batch test")
    assert result.success
    assert result.response == "recovered"


# ── cross-message plan persistence ────────────────────────────────────


def test_ask_snapshot_resumes_on_next_message():
    """ask → snapshot → resume_from: the plan continues, not restarts."""
    reg = FakeRegistry()
    reg.register("signup", lambda platform: f"started {platform}",
                 "start signup", {"platform": {}})

    llm1 = FakeLLM([
        {"thought": "need the platform", "action": "ask",
         "plan": "sign up for trial once the platform is known",
         "response": "Which platform?"},
    ])
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop1 = AgenticLoop(llm1, adapter, step_budget=5)
    r1 = loop1.run("help me sign up for a trial")
    assert r1.asked_user
    assert r1.question == "Which platform?"
    assert r1.memory_snapshot, "ask must return a memory snapshot"

    # user answers on the next message; a fresh loop resumes the plan
    llm2 = FakeLLM([
        {"thought": "platform known, proceed", "action": "tool",
         "tool": "signup", "args": {"platform": "netflix"}},
        {"thought": "done", "action": "respond",
         "response": "Signup started for netflix."},
    ])
    loop2 = AgenticLoop(llm2, adapter, step_budget=5)
    r2 = run_agentic("netflix", llm=llm2, registry=reg,
                     resume_from=r1.memory_snapshot)
    assert r2.success
    assert r2.tools_called == ["signup"]
    assert ("signup", {"platform": "netflix"}) in reg.calls

    # the resumed memory carries the plan, the Q&A, and both runs' steps
    mem = LoopMemory.from_dict(r2.memory_snapshot)
    assert "trial" in mem.plan
    assert ("assistant", "Which platform?") in mem.history
    assert ("user", "netflix") in mem.history
    assert len(mem.steps) == 3  # ask + tool + respond


def test_memory_roundtrip_is_lossless():
    mem = LoopMemory(user_message="original request")
    mem.add_history("user", "hello")
    mem.set_plan("do the thing in two steps")
    mem.record_step(StepRecord(step=1, thought="t1", action="tool",
                               tool_name="x", tool_args={"a": 1},
                               observation="out", failed=False))
    mem.record_step(StepRecord(step=2, thought="t2", action="ask",
                               question="sure?"))
    restored = LoopMemory.from_dict(mem.to_dict())
    assert restored.user_message == "original request"
    assert restored.history == [("user", "hello")]
    assert restored.plan == "do the thing in two steps"
    assert len(restored.steps) == 2
    assert restored.steps[0].tool_args == {"a": 1}
    assert restored.last_question() == "sure?"
    assert restored.pending_ask()


def test_memory_from_dict_is_defensive():
    assert LoopMemory.from_dict(None).user_message == ""
    assert LoopMemory.from_dict({}).user_message == ""
    assert LoopMemory.from_dict({"steps": "garbage"}).steps == []
    # fresh message without a snapshot still works
    r = run_agentic("hi", llm=FakeLLM([
        {"thought": "t", "action": "respond", "response": "hey"}]),
        registry=FakeRegistry(), resume_from=None)
    assert r.success


def test_plan_field_updates_memory():
    loop, reg, llm = make_loop(
        [
            {"thought": "step one", "action": "tool", "tool": "noop",
             "args": {}, "plan": "first check X, then do Y"},
            {"thought": "done", "action": "respond", "response": "finished"},
        ],
        lambda r: r.register("noop", lambda: "ok", "does nothing"),
    )
    result = loop.run("do the thing")
    assert result.success
    mem = LoopMemory.from_dict(result.memory_snapshot)
    assert mem.plan == "first check X, then do Y"


# ── planner bridge: unified orchestration ──────────────────────────────

from nomorals.agents.orchestration.bridge import (
    is_model_available,
    maybe_run_agentic,
)
from nomorals.agents.orchestration.planner_bridge import (
    PlannerBridge,
    model_steps_to_plan,
    observation_from_result,
)


def _bridge_registry():
    reg = FakeRegistry()
    reg.register("get_time", lambda: "12:00", "current time")
    reg.register("web_search", lambda query="": f"results for {query}",
                 "search the web", parameters={"query": ""})
    return reg


def test_model_steps_to_plan_valid():
    reg = _bridge_registry()
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    steps = [
        {"name": "t1", "goal": "get the time", "tool": "get_time",
         "args": {}, "role": "data_collection", "depends_on": []},
        {"name": "t2", "goal": "search for noon", "tool": "web_search",
         "args": {"query": "noon"}, "role": "research", "depends_on": ["t1"]},
    ]
    plan = model_steps_to_plan("check time and search", steps, adapter)
    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.steps[0].name == "t1"
    assert plan.steps[0].payload["tool"] == "get_time"
    assert plan.steps[1].depends_on == ["t1"]
    assert plan.steps[1].role == "research"


def test_model_steps_to_plan_rejects_unknown_tool():
    reg = _bridge_registry()
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    steps = [
        {"name": "t1", "goal": "do magic", "tool": "nonexistent_tool",
         "args": {}},
    ]
    assert model_steps_to_plan("do magic", steps, adapter) is None


def test_model_steps_to_plan_rejects_bad_deps():
    reg = _bridge_registry()
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    steps = [
        {"name": "t1", "goal": "step one", "tool": "get_time",
         "args": {}, "depends_on": ["ghost"]},
    ]
    assert model_steps_to_plan("bad deps", steps, adapter) is None


def test_model_steps_to_plan_rejects_duplicates():
    reg = _bridge_registry()
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    steps = [
        {"name": "t1", "goal": "one", "tool": "get_time", "args": {}},
        {"name": "t1", "goal": "two", "tool": "get_time", "args": {}},
    ]
    assert model_steps_to_plan("dupes", steps, adapter) is None


def test_loop_emits_plan_action():
    """The loop routes a plan action through the bridge and records it."""
    reg = _bridge_registry()
    llm = FakeLLM([
        {"thought": "complex task, needs planning", "action": "plan",
         "goal": "check time then search",
         "steps": [
             {"name": "t1", "goal": "get the time", "tool": "get_time",
              "args": {}, "role": "data_collection", "depends_on": []},
         ],
         "plan": "get time via planned execution"},
        {"thought": "got the planned result", "action": "respond",
         "response": "planned done"},
    ])
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(llm, adapter, step_budget=10)
    result = loop.run("complex request")
    assert result.success
    assert "plan" in result.tools_called
    # The plan step was recorded
    mem = LoopMemory.from_dict(result.memory_snapshot)
    plan_steps = [s for s in mem.steps if s.action == "plan"]
    assert len(plan_steps) == 1
    assert "planned execution" in plan_steps[0].observation.lower()


def test_planned_result_feeds_back_as_observation():
    """A plan action's observation contains the orchestrated outcome."""
    reg = _bridge_registry()
    llm = FakeLLM([
        {"thought": "needs orchestration", "action": "plan",
         "goal": "get time",
         "steps": [
             {"name": "t1", "goal": "get the time", "tool": "get_time",
              "args": {}, "depends_on": []},
         ]},
        {"thought": "done", "action": "respond", "response": "ok"},
    ])
    adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
    loop = AgenticLoop(llm, adapter, step_budget=10)
    result = loop.run("do it planned")
    mem = LoopMemory.from_dict(result.memory_snapshot)
    plan_step = next(s for s in mem.steps if s.action == "plan")
    assert not plan_step.failed
    assert "12:00" in plan_step.observation or "succeeded" in plan_step.observation


def test_plan_lessons_land_in_memory():
    """Reflection lessons from planned execution are visible to the loop."""
    mem = LoopMemory(user_message="test")
    mem.add_lesson("always check the time first")
    mem.add_lesson("always check the time first")  # dedup
    assert len(mem.lessons) == 1
    rendered = mem.render()
    assert "LESSONS FROM PAST RUNS" in rendered
    assert "always check the time first" in rendered
    # Serialization roundtrip
    restored = LoopMemory.from_dict(mem.to_dict())
    assert restored.lessons == mem.lessons


def test_is_model_available():
    assert not is_model_available(None)

    class Dead:
        available = False
    assert not is_model_available(Dead())

    class NoCreds:
        api_key = ""
    assert not is_model_available(NoCreds())

    class Live:
        available = True
    assert is_model_available(Live())
    # Plain object with no flags → assumed usable
    assert is_model_available(object())


def test_maybe_run_agentic_returns_none_without_model(monkeypatch):
    """No model → None → rigid pipeline handles it. Exact commands bypass."""
    import os
    monkeypatch.setenv("NM_AGENTIC_MODE", "1")
    reg = _bridge_registry()
    # llm=None must fall through, not crash
    assert maybe_run_agentic("hello", llm=None, registry=reg) is None
    # Disabled mode also falls through (exact /commands parse first anyway)
    monkeypatch.setenv("NM_AGENTIC_MODE", "0")
    llm = FakeLLM([{"thought": "t", "action": "respond", "response": "hi"}])
    assert maybe_run_agentic("/model groq", llm=llm, registry=reg) is None


def test_observation_from_result():
    from nomorals.agents.orchestrator import OrchestrationResult, Plan
    from nomorals.agents.runtime import ExecutionReport
    report = ExecutionReport()
    report.done = 2
    report.failed = 0
    result = OrchestrationResult(
        goal="test goal",
        plan=Plan(goal="test goal"),
        report=report,
        answer="the answer",
        lessons=["lesson one"],
        score=0.9,
        seconds=1.5,
    )
    obs = observation_from_result(result)
    assert "succeeded" in obs
    assert "the answer" in obs
    assert "lesson one" in obs
