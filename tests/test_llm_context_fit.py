"""Context fitting: strategies, budgets, task-kind chains. Offline."""
from __future__ import annotations

import pytest

from nomorals.llm.base import Message, estimate_messages
from nomorals.llm.context_fit import (
    TASK_FIT_CHAINS,
    Compact,
    FitResult,
    SummarizeMiddle,
    TruncateOldest,
    fit_messages,
    fit_prompt,
    strategy_for,
)


def _msgs(n: int, size: int = 60) -> list[Message]:
    out = [Message.system("you are a test system prompt")]
    for i in range(n):
        out.append(Message.user(f"user turn {i} " + "x" * size))
        out.append(Message.assistant(f"assistant turn {i} " + "y" * size))
    return out


def test_noop_when_under_budget():
    msgs = _msgs(2)
    res = fit_messages(msgs, 8192, "chat")
    assert res.ok and res.applied == [] and res.messages == msgs


def test_truncate_oldest_keeps_system_and_latest():
    msgs = _msgs(10)
    # a realistic budget: enough for ~half the conversation
    budget = estimate_messages(msgs[:12])
    res = fit_messages(msgs, budget, "chat")
    assert res.ok
    assert "truncate_oldest" in res.applied
    assert res.messages[0].role == "system"
    assert res.messages[-1].content == msgs[-1].content  # latest turn sacred
    assert res.dropped_turns > 0
    assert res.tokens <= max(256, int(budget * 0.85))


def test_summarize_middle_for_judge_kind():
    msgs = _msgs(10)
    budget = estimate_messages(msgs[:8])
    calls: list[str] = []
    res = fit_messages(msgs, budget, "judge",
                       summarizer=lambda t: calls.append(t) or "decisions made")
    assert res.ok
    assert res.summarized
    assert calls, "summarizer should have been consulted"
    assert any("summarized" in m.content for m in res.messages)
    assert res.messages[0].role == "system"
    assert res.messages[-1].content == msgs[-1].content


def test_summarize_middle_degrades_without_summarizer():
    msgs = _msgs(10)
    budget = estimate_messages(msgs[:8])
    res = fit_messages(msgs, budget, "judge", summarizer=None)
    assert res.ok and not res.summarized
    assert res.tokens <= max(256, int(budget * 0.85))


def test_summarizer_failure_falls_back():
    msgs = _msgs(10)
    budget = estimate_messages(msgs[:6])

    def boom(_t: str) -> str:
        raise RuntimeError("nope")

    res = fit_messages(msgs, budget, "judge", summarizer=boom)
    assert res.ok  # never raises, never stuck


def test_compact_first_pass():
    msgs = [Message.user("hello    world\n\n\nagain")]
    out = Compact().apply(msgs, 8192, {})
    assert "  " not in out[0].content
    # and fit_messages runs it as the chain's first strategy
    res = fit_messages([Message.user("a  b " * 2000)], 8192, "chat")
    assert res.ok


def test_fit_prompt_keeps_tail():
    prompt = "context: " + "x" * 5000 + "\nQUESTION: what is 2+2?"
    out = fit_prompt(prompt, 600, "chat")
    assert "what is 2+2?" in out
    assert out.startswith("…[truncated]…")


def test_task_chains_cover_common_kinds():
    for kind in ("chat", "code", "judge", "research", "summarize",
                 "intent", "creative", "vision", "plan"):
        assert kind in TASK_FIT_CHAINS
        assert TASK_FIT_CHAINS[kind]  # non-empty


def test_strategy_for_unknown_is_none():
    assert strategy_for("nope") is None
    assert strategy_for("truncate_oldest").name == "truncate_oldest"


def test_overflow_flag_when_chain_exhausted():
    msgs = [Message.system("s"), Message.user("u" * 10000)]
    res = fit_messages(msgs, 100, "chat")
    assert res.overflow  # honest: can't fit 10k chars into 100 tokens


def test_never_raises_on_garbage():
    res = fit_messages([], 100, "chat")
    assert isinstance(res, FitResult)
    fit_prompt("", 100)
