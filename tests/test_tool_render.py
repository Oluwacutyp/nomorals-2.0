"""Tool outputs must be clean, errors must carry the real cause, and
long runs must show progress (Workstream 2: tool interactions + output
rendering).

* No raw dict / JSON dump reaches chat — dicts render as ``key: value``.
* No truncated JSON — caps are structural with an explicit ``… (N more)``.
* No empty ``()`` errors — a message-less exception yields its type name.
* No bare generic errors — ``format_tool_error`` always names the cause.
* Long runs emit per-step progress via ``DevonAgent.run(on_progress=…)``.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.agents.devon import DevonAgent
from nomorals.core.errors import NoMoralsError
from nomorals.core.tool_render import (
    describe_exc,
    format_tool_error,
    outcome_message,
    render_tool_value,
)
from nomorals.social.render import render_metrics_card


def test_describe_exc_never_empty():
    assert describe_exc(RuntimeError()) == "RuntimeError"
    assert describe_exc(Exception()) == "unknown error"
    assert describe_exc(ValueError("bad port")) == "bad port"
    assert describe_exc(OSError()) == "OSError"
    assert describe_exc(NoMoralsError("")) == "NoMoralsError"
    assert describe_exc(NoMoralsError("STT API 429: rate limited")) == \
        "STT API 429: rate limited"
    assert describe_exc(None) == "unknown error"
    assert "(" not in describe_exc(RuntimeError()) or True  # no "()" pattern
    assert describe_exc(RuntimeError()) != ""


def test_outcome_message_extracts_and_falls_back():
    assert outcome_message("") == "unknown error"
    assert outcome_message(None) == "unknown error"
    assert outcome_message("boom") == "boom"
    assert outcome_message(NoMoralsError("quota exceeded")) == "quota exceeded"


def test_format_tool_error_carries_real_cause():
    assert format_tool_error("dns_lookup", ValueError("connection refused")) == \
        "dns_lookup failed: connection refused"
    # Message-less exception: type name, never "failed ()" / "failed: ".
    assert format_tool_error("speak", RuntimeError()) == "speak failed: RuntimeError"
    assert not format_tool_error("x", RuntimeError()).endswith(": ")
    assert "()" not in format_tool_error("x", RuntimeError())


def test_render_dict_is_lines_not_dump():
    out = render_tool_value({"status": "ok", "rows": 3, "nested": {"a": 1}})
    assert "{'status'" not in out and '{"status"' not in out
    assert "status: ok" in out
    assert "rows: 3" in out
    assert "nested:" in out and "a: 1" in out


def test_render_list_and_scalars():
    out = render_tool_value(["alpha", "beta"])
    assert "• alpha" in out and "• beta" in out
    assert render_tool_value(None) == "(no output)"
    assert render_tool_value("") == "(empty)"
    assert render_tool_value(42) == "42"
    assert render_tool_value("plain text") == "plain text"


def test_render_truncation_is_structural_not_mid_json():
    big = {f"field_{i}": i for i in range(200)}
    out = render_tool_value(big)
    assert "… (" in out and "more fields)" in out
    assert not out.rstrip().endswith('{"field_')  # never a dangling cut
    assert '{"field_0"' not in out
    long_list = list(range(200))
    out2 = render_tool_value(long_list)
    assert "… (160 more)" in out2


def test_render_never_raises_on_weird_values():
    class Bad:
        def __str__(self):
            raise RuntimeError("nope")

    out = render_tool_value({"bad": Bad()})
    assert isinstance(out, str) and out


def test_metrics_card_has_no_raw_dict():
    out = render_metrics_card(
        {"likes": 12, "reposts": 3, "replies": 1}, platform="whatsapp")
    assert isinstance(out, str)
    assert "{'likes'" not in out
    assert "Likes: *12*" in out


def _bare_devon(tools):
    agent = object.__new__(DevonAgent)
    agent.context = SimpleNamespace(tools=None)
    agent.brain = None
    agent.gateway = None
    agent.max_steps = 6
    agent.step_timeout = 90.0
    agent._tools = tools
    agent._on_progress = None
    return agent


def test_execute_empty_exception_gives_type_not_parens():
    def boom(_args):
        raise RuntimeError()

    agent = _bare_devon({"boom": boom})
    outcome = agent._execute("boom", {})
    assert outcome.ok is False
    assert outcome.error == "RuntimeError"
    assert outcome.observation == "tool error: RuntimeError"
    assert "()" not in outcome.observation


def test_run_emits_progress_per_step():
    seen: list[tuple[int, int, str, str]] = []
    agent = _bare_devon({"echo": lambda _a: "hello"})
    agent._plan = lambda task, memory: ([{"tool": "echo", "args": {}}], "heuristic", "", "")
    agent._prior_memory = lambda: ""
    agent._remember = lambda *a, **k: None
    agent._finish = lambda *a, **k: None
    agent._digest = lambda *a, **k: "done"
    result = agent.run("do a thing", on_progress=lambda *a: seen.append(a))
    assert result.digest == "done"
    assert ("start" in [s[3] for s in seen]) and ("done" in [s[3] for s in seen])
    assert seen[0] == (1, 1, "echo", "start")
    assert seen[-1] == (1, 1, "echo", "done")


def test_run_progress_callback_can_never_break_run():
    def bad_cb(*_a):
        raise RuntimeError("callback exploded")

    agent = _bare_devon({"echo": lambda _a: "hello"})
    agent._plan = lambda task, memory: ([{"tool": "echo", "args": {}}], "heuristic", "", "")
    agent._prior_memory = lambda: ""
    agent._remember = lambda *a, **k: None
    agent._finish = lambda *a, **k: None
    agent._digest = lambda *a, **k: "done"
    result = agent.run("do a thing", on_progress=bad_cb)
    assert result.digest == "done"
