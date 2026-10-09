"""Tests for the brain-driven tool-calling loop (the spine)."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.agents.partner.tool_loop import (
    ToolCallingLoop,
    ToolLoopResult,
    parse_tool_calls,
    _strip_tool_blocks,
)
from nomorals.llm.base import Message, LLMResponse
from nomorals.tools.registry import ToolRegistry


class FakeLLM:
    """Scripted model: returns canned responses in order."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0
        self.seen_messages = []

    def chat(self, messages, **kw):
        self.calls += 1
        self.seen_messages.append(messages)
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return LLMResponse(text=text, model="fake")


def _registry():
    r = ToolRegistry()
    r.register_builtins()
    return r


class TestParse:
    def test_fenced_block(self):
        text = 'Let me check.\n```tool\n{"tool": "foo", "args": {"a": 1}}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "foo"
        assert calls[0].args == {"a": 1}

    def test_multiple_blocks(self):
        text = ('```tool\n{"tool": "a", "args": {}}\n```\n'
                '```tool\n{"tool": "b", "args": {"x": "y"}}\n```')
        calls = parse_tool_calls(text)
        assert [c.name for c in calls] == ["a", "b"]

    def test_no_calls(self):
        assert parse_tool_calls("just a plain answer") == []

    def test_malformed_json_skipped(self):
        text = '```tool\n{not json}\n```\nplain text'
        assert parse_tool_calls(text) == []

    def test_strip_leaves_prose(self):
        text = 'thinking\n```tool\n{"tool": "a", "args": {}}\n```\nfinal words'
        stripped = _strip_tool_blocks(text)
        assert "final words" in stripped
        assert "tool" not in stripped.replace("final words", "")


class TestLoop:
    def test_direct_answer_no_tools(self):
        llm = FakeLLM(["Hello! How can I help?"])
        loop = ToolCallingLoop(llm, _registry(), max_iterations=3)
        result = loop.run("hi", actor="owner")
        assert result.ok
        assert "Hello!" in result.answer
        assert result.tools_used == []
        assert result.iterations == 1

    def test_tool_then_answer(self):
        # Register a tiny test tool via a fresh registry to avoid side effects.
        from nomorals.tools.registry import ToolRegistry
        from nomorals.core.result import Ok
        r = ToolRegistry()
        r.register("echo_test", lambda text="": f"echo:{text}",
                   description="echoes text", parameters={"text": "str"})
        llm = FakeLLM([
            'Let me echo that.\n```tool\n{"tool": "echo_test", "args": {"text": "hi"}}\n```',
            "Done — the tool said echo:hi.",
        ])
        loop = ToolCallingLoop(llm, r, max_iterations=3)
        result = loop.run("echo hi", actor="owner")
        assert result.ok
        assert result.tools_used == ["echo_test"]
        assert "echo:hi" in result.answer or "Done" in result.answer
        # The observation must have reached the model on turn 2.
        assert llm.calls == 2
        second_turn = llm.seen_messages[1]
        obs_text = " ".join(m.content for m in second_turn)
        assert "echo:hi" in obs_text

    def test_unknown_tool_reported_not_faked(self):
        llm = FakeLLM([
            '```tool\n{"tool": "nope_not_real", "args": {}}\n```',
            "That tool doesn't exist, so I can't do that.",
        ])
        loop = ToolCallingLoop(llm, _registry(), max_iterations=3)
        result = loop.run("do the thing", actor="owner")
        assert result.ok
        assert "nope_not_real" in result.tools_used  # attempted, honestly
        obs = " ".join(m.content for m in llm.seen_messages[1])
        assert "unknown tool" in obs

    def test_iteration_budget(self):
        # Model keeps calling tools forever; loop must stop.
        llm = FakeLLM([
            '```tool\n{"tool": "echo_test", "args": {}}\n```'
        ] * 20)
        from nomorals.tools.registry import ToolRegistry
        r = ToolRegistry()
        r.register("echo_test", lambda: "x", description="x", parameters={})
        loop = ToolCallingLoop(llm, r, max_iterations=3)
        result = loop.run("go", actor="owner")
        assert result.iterations == 3
        assert result.degraded
        assert not result.ok

    def test_failing_tool_fed_back(self):
        from nomorals.tools.registry import ToolRegistry
        r = ToolRegistry()
        def _boom():
            raise RuntimeError("kaput")
        r.register("boom", _boom, description="always fails", parameters={})
        llm = FakeLLM([
            '```tool\n{"tool": "boom", "args": {}}\n```',
            "The tool failed with 'kaput', so I can't complete that.",
        ])
        loop = ToolCallingLoop(llm, r, max_iterations=3)
        result = loop.run("break it", actor="owner")
        assert result.ok
        obs = " ".join(m.content for m in llm.seen_messages[1])
        assert "kaput" in obs or "failed" in obs


class TestAudit:
    def test_all_tools_reachable(self):
        r = _registry()
        loop = ToolCallingLoop(FakeLLM(["done"]), r)
        audit = loop.audit_reachability()
        assert audit["total"] == 282, f"expected 282, got {audit['total']}"
        assert audit["listed"] == 282, f"missing: {audit['missing'][:10]}"
        assert audit["missing"] == []

    def test_restricted_filters(self):
        from nomorals.core.policy import CapabilitySet, Capability
        r = _registry()
        loop = ToolCallingLoop(FakeLLM(["done"]), r)
        restricted = CapabilitySet.of(Capability.MODEL_CALL,
                                      Capability.MEM_READ,
                                      Capability.FS_READ)
        audit = loop.audit_reachability(capabilities=restricted)
        # Restricted sees fewer tools than the full set.
        assert audit["listed"] < audit["total"]
        assert audit["listed"] > 0
