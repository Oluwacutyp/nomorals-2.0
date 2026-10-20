"""Pass #44 — idempotency keys wired into the production callers.

Pass #40 verified the registry machinery: ``ToolRegistry.call`` accepts
an ``idempotency_key`` and replays a recorded result (same tool, key,
actor, args digest, tool version) instead of re-executing — durably,
across restarts. But NO production caller passed a key, so every
retry / replan / redelivery re-executed side effects.

This pass wires deterministic keys at the chokepoints where the SAME
intended effect can be attempted twice:

* partner tool loop — key from (chat identity, inbound message id,
  tool, args); no message id → no key (honest degrade, never invented).
* partner control commands (``_tool_reply``) — same chat scope, set by
  ``PartnerRuntime._process`` for the duration of one inbound message.
* mission steps — scope ``mission:<id>:step:<step_idempotency_key>``
  set inside ``MissionRunner._run_step_agent`` (the one place that runs
  on every attempt path, incl. the raw-thread timed path); consumed by
  ``RoleAgent._call_tool``, the funnel role agents call tools through.
* planning (PlanningRunner) — scope ``plan:<run>:<phase>:<step
  fingerprint>`` (the pass-#15 ``_step_spec_key``) set around each step
  job submission, so a replanned identical step replays its tool
  effects while different work gets a different key and executes.
* orchestration ReAct loop — ``LoopMemory`` gains a durable ``run_id``
  (survives the resume snapshot); ``ToolAdapter`` keys on it.
* MCP — the JSON-RPC request id, scoped to principal + server session,
  so a client retry of one request replays and nothing else can.

Every key folds the canonical args into the digest, so two DISTINCT
intended effects (different tool or different args under the same
scope) always get different keys and both execute — the registry's
conflict Err can never fire from these callers by construction.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nomorals.core.idempotency_keys import (
    current_idempotency_scope,
    derive_idempotency_key,
    idempotency_scope,
    scoped_idempotency_key,
)
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


# ── helpers ─────────────────────────────────────────────────────────────

def _registry(db=None, calls: list | None = None) -> ToolRegistry:
    """A registry whose ``effect`` tool counts its real executions."""
    if calls is None:
        calls = []
    reg = ToolRegistry(context=SimpleNamespace(db=db), enforce=False)

    def effect(amount: int = 0) -> dict:
        calls.append({"amount": amount})
        return {"applied": amount, "n": len(calls)}

    reg.register("effect", effect, capability="fs.write", version="1")
    reg.calls_log = calls  # type: ignore[attr-defined]
    return reg


def _file_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "app.db"))
    db.migrate()
    return db


# ── (e) pre-fix reproduction: unkeyed calls double-execute ──────────────

class TestUnkeyedBaseline:
    """The old behavior, pinned so the wiring tests mean something."""

    def test_direct_call_without_key_double_executes(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        first = reg.call("effect", actor="alice", amount=10)
        second = reg.call("effect", actor="alice", amount=10)
        assert first.ok and second.ok
        assert len(calls) == 2  # no key → no protection → effect ran twice

    def test_derived_keys_are_deterministic_and_distinct(self) -> None:
        scope = "chat:telegram:42:m-1"
        k1 = derive_idempotency_key(scope, "effect", {"amount": 10})
        k2 = derive_idempotency_key(scope, "effect", {"amount": 10})
        k3 = derive_idempotency_key(scope, "effect", {"amount": 20})
        k4 = derive_idempotency_key("chat:telegram:42:m-2", "effect",
                                    {"amount": 10})
        assert k1 and k1 == k2
        assert k3 != k1  # different args → different key → executes
        assert k4 != k1  # different message → different key → executes
        assert derive_idempotency_key("", "effect", {"amount": 10}) == ""


# ── (d) chat path: the partner tool loop ────────────────────────────────

class TestChatToolLoop:
    def _loop(self, reg: ToolRegistry):
        from nomorals.agents.partner.tool_loop import (
            ToolCall,
            ToolCallingLoop,
        )

        return ToolCallingLoop(llm=None, registry=reg), ToolCall

    def _ctx(self, message_id: str | None = "m-100") -> dict:
        ctx = {
            "platform": "telegram",
            "chat_key": "telegram:42",
            "chat_kind": "dm",
            "is_owner": True,
            "sender_id": "7",
        }
        if message_id is not None:
            ctx["message_id"] = message_id
        return ctx

    def test_redelivered_message_replays_identical_call(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        loop, ToolCall = self._loop(reg)
        ctx = self._ctx()
        first = loop._invoke(ToolCall("effect", {"amount": 10}),
                             actor="owner", loop_ctx=ctx)
        # The same inbound message is delivered again (gateway retry /
        # restart): the loop runs again and the model re-issues the
        # identical call. The effect must NOT run twice.
        second = loop._invoke(ToolCall("effect", {"amount": 10}),
                              actor="owner", loop_ctx=ctx)
        assert len(calls) == 1
        assert "[tool ok]" in first and "[tool ok]" in second
        assert '"n": 1' in second  # the recorded result, not a fresh run

    def test_distinct_call_in_same_message_executes(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        loop, ToolCall = self._loop(reg)
        ctx = self._ctx()
        loop._invoke(ToolCall("effect", {"amount": 10}),
                     actor="owner", loop_ctx=ctx)
        loop._invoke(ToolCall("effect", {"amount": 20}),
                     actor="owner", loop_ctx=ctx)
        assert len(calls) == 2  # different args = different intended effect

    def test_new_message_same_call_executes(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        loop, ToolCall = self._loop(reg)
        loop._invoke(ToolCall("effect", {"amount": 10}),
                     actor="owner", loop_ctx=self._ctx("m-100"))
        loop._invoke(ToolCall("effect", {"amount": 10}),
                     actor="owner", loop_ctx=self._ctx("m-101"))
        assert len(calls) == 2  # a fresh instruction is a fresh effect

    def test_missing_message_id_degrades_to_unkeyed(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        loop, ToolCall = self._loop(reg)
        ctx = self._ctx(message_id=None)
        loop._invoke(ToolCall("effect", {"amount": 10}),
                     actor="owner", loop_ctx=ctx)
        loop._invoke(ToolCall("effect", {"amount": 10}),
                     actor="owner", loop_ctx=ctx)
        # No message id → no key may be invented: today's behavior,
        # honestly (a guessed key could swallow a distinct effect).
        assert len(calls) == 2


# ── chat path: control commands (_process scope + _tool_reply) ──────────

class TestChatControlPath:
    def _runtime(self, reg: ToolRegistry):
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(tools=reg)
        rt._send_long_checked = lambda *a, **k: ""  # delivery not under test
        return rt

    def test_process_sets_chat_scope_for_the_message(self) -> None:
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        seen: list[str] = []
        rt._process_inner = lambda msg: seen.append(  # type: ignore[attr-defined]
            current_idempotency_scope())
        message = SimpleNamespace(
            chat=SimpleNamespace(key="telegram:42"), message_id="m-9")
        rt._process(message)
        assert seen == ["chat:telegram:42:m-9"]
        assert current_idempotency_scope() == ""  # restored afterwards

    def test_process_without_message_id_sets_no_scope(self) -> None:
        from nomorals.agents.partner.runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        seen: list[str] = []
        rt._process_inner = lambda msg: seen.append(  # type: ignore[attr-defined]
            current_idempotency_scope())
        message = SimpleNamespace(
            chat=SimpleNamespace(key="telegram:42"), message_id="")
        rt._process(message)
        assert seen == [""]

    def test_tool_reply_replays_under_chat_scope(self) -> None:
        calls: list = []
        reg = _registry(calls=calls)
        rt = self._runtime(reg)
        with idempotency_scope("chat:telegram:42:m-9"):
            first = rt._tool_reply("effect", "telegram:42", amount=10)
            second = rt._tool_reply("effect", "telegram:42", amount=10)
            third = rt._tool_reply("effect", "telegram:42", amount=20)
        assert "failed" not in first
        assert len(calls) == 2  # identical replayed; distinct executed
        assert first == second


# ── (a)+(b) mission/step path ───────────────────────────────────────────

def _role_agent(name: str, ctx):  # noqa: ANN202 - test helper
    """A concrete RoleAgent so tests exercise the real _call_tool."""
    from nomorals.agents.roles import RoleAgent

    class _Bare(RoleAgent):
        def work(self, task_input):  # noqa: ANN001, ANN202
            return {}

    return _Bare(name=name, context=ctx)


class _EffectAgent:
    """Stand-in for a role agent: its work is one registry tool call."""

    def __init__(self, context, script: list[str], amount: int = 10):
        self.context = context
        self._script = list(script)
        self._amount = amount

    def run(self, prompt):  # noqa: ANN001, ANN202 - mirrors Agent.run's shape
        mode = self._script.pop(0) if self._script else "ok"
        agent = _role_agent("fx", self.context)
        value = agent._call_tool("effect", amount=self._amount)
        if mode == "boom":
            # The effect landed, THEN the step died (crash, 503, kill):
            # the mission retry must not re-apply the effect.
            return SimpleNamespace(ok=False, output={},
                                   error="RuntimeError: 503 after effect",
                                   tokens=0)
        return SimpleNamespace(ok=True, output={"tool": value},
                               error="", tokens=0)


def _mission_setup(tmp_path, monkeypatch, script: list[str],
                   amount: int = 10):
    from nomorals.core.tasks import TaskKind
    from nomorals.agents.orchestrator import PlanStep
    from nomorals.missions import (
        IdempotencyStore,
        MissionRunner,
        MissionStore,
    )
    import nomorals.agents.roles as roles_mod

    db = _file_db(tmp_path)
    calls: list = []
    reg = _registry(db=db, calls=calls)
    ctx = SimpleNamespace(db=db, memory=None, tools=reg)
    store = MissionStore(db)
    mission = store.create_new("test goal", name="m")
    step = PlanStep(name="a", goal="do a", role="execution",
                    kind=TaskKind.IO,
                    payload={"retries": 1, "retry_backoff_s": 0.01,
                             "retry_on": "any"})
    runner = MissionRunner(ctx, store=store, milestones=False,
                           idempotency=IdempotencyStore(db),
                           orchestrator_factory=lambda: MagicMock())
    agent = _EffectAgent(ctx, script, amount=amount)
    monkeypatch.setattr(roles_mod, "build_agent",
                        lambda role, name="", context=None: agent)
    return runner, mission, step, calls, ctx


class TestMissionStepPath:
    def test_retried_step_does_not_reapply_tool_effect(
            self, tmp_path, monkeypatch) -> None:
        runner, mission, step, calls, _ctx = _mission_setup(
            tmp_path, monkeypatch, ["boom", "ok"])
        outcome = runner._execute_step(mission, step)
        assert outcome.ok
        assert mission.state["step_attempts"]["a"] == 2  # step ran twice
        assert len(calls) == 1  # …but the tool effect ran ONCE

    def test_registry_replay_survives_restart(
            self, tmp_path, monkeypatch) -> None:
        # First "process": the step's agent runs the tool once.
        runner, mission, step, calls, ctx = _mission_setup(
            tmp_path, monkeypatch, ["ok"])
        started = runner._clock()
        outcome1 = runner._run_step_agent(mission, step, started,
                                          {"timeout_s": 0.0})
        assert outcome1.ok and len(calls) == 1

        # Second "process": a brand-new registry over the SAME db file.
        # The mission step runs again (e.g. resumed mission); the tool
        # call must replay from the durable table, not re-execute.
        from nomorals.missions import MissionRunner, MissionStore
        import nomorals.agents.roles as roles_mod

        db2 = Database(str(tmp_path / "app.db"))
        calls2: list = []
        reg2 = _registry(db=db2, calls=calls2)
        ctx2 = SimpleNamespace(db=db2, memory=None, tools=reg2)
        runner2 = MissionRunner(ctx2, store=MissionStore(db2),
                                milestones=False,
                                orchestrator_factory=lambda: MagicMock())
        agent2 = _EffectAgent(ctx2, ["ok"])
        monkeypatch.setattr(roles_mod, "build_agent",
                            lambda role, name="", context=None: agent2)
        outcome2 = runner2._run_step_agent(mission, step, runner2._clock(),
                                           {"timeout_s": 0.0})
        assert outcome2.ok
        assert len(calls2) == 0  # replayed from the durable record

    def test_distinct_step_and_distinct_args_execute(
            self, tmp_path, monkeypatch) -> None:
        from nomorals.core.tasks import TaskKind
        from nomorals.agents.orchestrator import PlanStep
        import nomorals.agents.roles as roles_mod

        runner, mission, step, calls, ctx = _mission_setup(
            tmp_path, monkeypatch, ["ok"])
        started = runner._clock()
        assert runner._run_step_agent(mission, step, started,
                                      {"timeout_s": 0.0}).ok
        assert len(calls) == 1
        # A different step in the same mission: different scope → executes.
        step_b = PlanStep(name="b", goal="do b", role="execution",
                          kind=TaskKind.IO, payload={})
        assert runner._run_step_agent(mission, step_b, started,
                                      {"timeout_s": 0.0}).ok
        assert len(calls) == 2
        # Same step, different args: different key → executes.
        agent99 = _EffectAgent(ctx, ["ok"], amount=99)
        monkeypatch.setattr(roles_mod, "build_agent",
                            lambda role, name="", context=None: agent99)
        assert runner._run_step_agent(mission, step, started,
                                      {"timeout_s": 0.0}).ok
        assert len(calls) == 3


# ── planning path: PlanningRunner replans ───────────────────────────────

class TestPlanningPath:
    def _runner_with_agent(self, reg: ToolRegistry, ctx) -> tuple:
        from nomorals.agents.modes import PlanningRunner

        return PlanningRunner(), _role_agent("plan-fx", ctx)

    def test_replanned_identical_step_replays_tool(self, tmp_path) -> None:
        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        ctx = SimpleNamespace(db=db, tools=reg)
        runner, agent = self._runner_with_agent(reg, ctx)

        plans = [
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 10}, "depends_on": []}],
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 10}, "depends_on": []}],
        ]
        verify_calls = {"n": 0}

        def decompose(goal, evidence=None):  # noqa: ANN001, ANN202
            return plans.pop(0) if plans else []

        def execute_step(step, ctx):  # noqa: ANN001, ANN202
            return agent._call_tool(
                "effect", amount=step.parameters["amount"])

        def verify_step(step, _ctx):  # noqa: ANN001, ANN202
            verify_calls["n"] += 1
            return verify_calls["n"] > 1  # fail once → replan → pass

        result = runner.run_sync("goal", decompose=decompose,
                                 execute_step=execute_step,
                                 verify_step=verify_step)
        assert result.success
        assert len(calls) == 1  # replan re-ran the step, not the effect

    def test_replanned_reworded_step_executes(self, tmp_path) -> None:
        """Same tool + same tool args, but the STEP's spec changed.

        The pass-#15 fingerprint (action, name, canonical params) is part
        of the key, so reworded work is different work — it executes
        instead of replaying the earlier step's effect.
        """
        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        ctx = SimpleNamespace(db=db, tools=reg)
        runner, agent = self._runner_with_agent(reg, ctx)

        plans = [
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 10}, "depends_on": []}],
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 10, "note": "retry harder"},
              "depends_on": []}],
        ]
        verify_calls = {"n": 0}

        def decompose(goal, evidence=None):  # noqa: ANN001, ANN202
            return plans.pop(0) if plans else []

        def execute_step(step, ctx):  # noqa: ANN001, ANN202
            return agent._call_tool(
                "effect", amount=step.parameters["amount"])

        def verify_step(step, _ctx):  # noqa: ANN001, ANN202
            verify_calls["n"] += 1
            return verify_calls["n"] > 1

        result = runner.run_sync("goal", decompose=decompose,
                                 execute_step=execute_step,
                                 verify_step=verify_step)
        assert result.success
        assert len(calls) == 2

    def test_replanned_different_step_executes(self, tmp_path) -> None:
        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        ctx = SimpleNamespace(db=db, tools=reg)
        runner, agent = self._runner_with_agent(reg, ctx)

        plans = [
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 10}, "depends_on": []}],
            [{"name": "s1", "action": "wait",
              "parameters": {"amount": 20}, "depends_on": []}],
        ]
        verify_calls = {"n": 0}

        def decompose(goal, evidence=None):  # noqa: ANN001, ANN202
            return plans.pop(0) if plans else []

        def execute_step(step, ctx):  # noqa: ANN001, ANN202
            return agent._call_tool(
                "effect", amount=step.parameters["amount"])

        def verify_step(step, result):  # noqa: ANN001, ANN202
            verify_calls["n"] += 1
            return verify_calls["n"] > 1

        result = runner.run_sync("goal", decompose=decompose,
                                 execute_step=execute_step,
                                 verify_step=verify_step)
        assert result.success
        assert len(calls) == 2  # different work = different key = executes


# ── orchestration path: ToolAdapter + LoopMemory run identity ───────────

class TestOrchestrationPath:
    def test_loop_memory_run_id_survives_snapshot(self) -> None:
        from nomorals.agents.orchestration.context import LoopMemory

        mem = LoopMemory(user_message="hi")
        assert mem.run_id
        restored = LoopMemory.from_dict(mem.to_dict())
        assert restored.run_id == mem.run_id
        fresh = LoopMemory(user_message="hi")
        assert fresh.run_id != mem.run_id

    def test_adapter_replays_within_a_run_only(self, tmp_path) -> None:
        from nomorals.agents.orchestration.context import LoopMemory
        from nomorals.agents.orchestration.tools import ToolAdapter

        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        mem = LoopMemory(user_message="do it")
        adapter = ToolAdapter(reg, idempotency_scope=f"orch:{mem.run_id}")
        ok1, obs1 = adapter.call("effect", {"amount": 10})
        ok2, obs2 = adapter.call("effect", {"amount": 10})
        assert ok1 and ok2 and obs1 == obs2
        assert len(calls) == 1
        ok3, _ = adapter.call("effect", {"amount": 20})
        assert ok3 and len(calls) == 2  # distinct args execute
        # A different run (fresh memory) is a different intended effect.
        other = ToolAdapter(
            reg, idempotency_scope=f"orch:{LoopMemory('again').run_id}")
        ok4, _ = other.call("effect", {"amount": 10})
        assert ok4 and len(calls) == 3

    def test_agentic_loop_stamps_scope_from_memory(self, tmp_path) -> None:
        from nomorals.agents.orchestration.context import LoopMemory
        from nomorals.agents.orchestration.loop import AgenticLoop
        from nomorals.agents.orchestration.tools import ToolAdapter
        from nomorals.core.policy import CapabilitySet

        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)

        class _Resp:
            def __init__(self, payload: dict):
                self.text = json.dumps(payload)
                self.content = self.text
                self.ok = True

        class _LLM:
            def __init__(self) -> None:
                self.n = 0

            def chat(self, messages, params=None, **kw):  # noqa: ANN001, ANN202
                self.n += 1
                if self.n == 1:
                    return _Resp({"thought": "apply it", "action": "tool",
                                  "tool": "effect", "args": {"amount": 10}})
                return _Resp({"thought": "done", "action": "respond",
                              "response": "applied"})

        adapter = ToolAdapter(reg, capabilities=CapabilitySet.all())
        loop = AgenticLoop(_LLM(), adapter, step_budget=5)
        mem = LoopMemory(user_message="apply 10")
        result = loop.run("apply 10", mem)
        assert result.success
        assert adapter.idempotency_scope == f"orch:{mem.run_id}"
        assert len(calls) == 1
        # Resume the SAME run from its snapshot and re-issue the call:
        # the effect must not land twice.
        resumed = LoopMemory.from_dict(mem.to_dict())
        adapter2 = ToolAdapter(reg, capabilities=CapabilitySet.all())
        loop2 = AgenticLoop(_LLM(), adapter2, step_budget=5)
        result2 = loop2.run("apply 10", resumed)
        assert result2.success
        assert len(calls) == 1


# ── MCP path: JSON-RPC request id ───────────────────────────────────────

class TestMcpPath:
    def _server(self, reg: ToolRegistry):
        from nomorals.api.mcp_server import MCPServer
        from nomorals.api.server import DEFAULT_PRINCIPAL

        return MCPServer(registry=reg, principal=DEFAULT_PRINCIPAL)

    def _tools_call(self, srv, mid, args: dict):
        msg = {"jsonrpc": "2.0", "method": "tools/call",
               "params": {"name": "tool_call",
                          "arguments": {"tool": "effect",
                                        "arguments": args}}}
        if mid is not None:
            msg["id"] = mid
        return srv.dispatch(msg, lambda n: None)

    def test_client_retry_with_same_request_id_replays(self, tmp_path) -> None:
        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        srv = self._server(reg)
        first = self._tools_call(srv, 1, {"amount": 10})
        assert first is not None and not first["result"]["isError"]
        # Client timed out and retried the SAME request (same id):
        retry = self._tools_call(srv, 1, {"amount": 10})
        assert retry is not None and not retry["result"]["isError"]
        assert len(calls) == 1
        # Same id carrying a different request is a different effect:
        other = self._tools_call(srv, 1, {"amount": 99})
        assert other is not None and not other["result"]["isError"]
        assert len(calls) == 2
        # A notification (no id) can never be keyed:
        note = self._tools_call(srv, None, {"amount": 10})
        assert note is None
        assert len(calls) == 3


# ── call_many: keys flow per call, contextvars intact ───────────────────

class TestCallManyKeys:
    def test_keyed_parallel_duplicates_execute_once(self, tmp_path) -> None:
        calls: list = []
        db = _file_db(tmp_path)
        reg = _registry(db=db, calls=calls)
        outcomes = reg.call_many(
            [("effect", {"amount": 10, "idempotency_key": "batch-1"}),
             ("effect", {"amount": 10, "idempotency_key": "batch-1"}),
             ("effect", {"amount": 20, "idempotency_key": "batch-2"})],
            max_workers=4, actor="alice")
        assert all(o.ok for o in outcomes)
        assert len(calls) == 2
        assert outcomes[0].value == outcomes[1].value

    def test_unkeyed_parallel_duplicates_still_execute(self, tmp_path) -> None:
        # The coding agent's explore reads ride this path unkeyed by
        # design (reads must stay fresh); pin that nothing changed.
        calls: list = []
        reg = _registry(calls=calls)
        outcomes = reg.call_many(
            [("effect", {"amount": 10}), ("effect", {"amount": 10})],
            max_workers=4, actor="alice")
        assert all(o.ok for o in outcomes)
        assert len(calls) == 2


# ── scoped helper: no scope → no key (honest degrade everywhere) ────────

class TestScopedHelper:
    def test_no_scope_no_key(self) -> None:
        assert current_idempotency_scope() == ""
        assert scoped_idempotency_key("effect", {"amount": 10}) == ""

    def test_scope_roundtrip(self) -> None:
        with idempotency_scope("chat:telegram:1:m-1"):
            key = scoped_idempotency_key("effect", {"amount": 10})
            assert key.startswith("chat:effect:")
        assert scoped_idempotency_key("effect", {"amount": 10}) == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
