"""Tests for /tour — the command smoke-tester.

The tour probes every registered command with safe read-only probes.
It must never raise, never execute destructive commands, and must
capture broken commands instead of dying on them.
"""

import pytest

from nomorals.social.chat.control import CONTROL_COMMANDS, parse_control
from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin


class FakeRuntime(RuntimeMetaMixin):
    """Minimal stand-in: real tour logic, scripted handle_control."""

    def __init__(self, behavior=None):
        # behavior: kind -> "ok" | "broken" | "slow" | "empty"
        self.behavior = behavior or {}

    def handle_control(self, text, chat_key="", message=None):
        kind = text.strip().lstrip("/").split()[0].lower()
        b = self.behavior.get(kind, "ok")
        if b == "broken":
            raise UnboundLocalError(
                "cannot access local variable 'chat' where it is not "
                "associated with a value")
        if b == "slow":
            import time
            time.sleep(30)  # the probe timeout must cut this off
        if b == "empty":
            return ""
        return f"{kind} usage: /{kind} [args]"


def test_tour_registered():
    assert "tour" in CONTROL_COMMANDS
    min_args, max_args = CONTROL_COMMANDS["tour"]
    assert min_args == 0 and max_args == 1
    cmd = parse_control("/tour")
    assert cmd is not None and cmd.kind == "tour"
    cmd = parse_control("/tour research")
    assert cmd is not None and cmd.kind == "tour" and cmd.tail == "research"


def test_tour_runs_without_raising():
    rt = FakeRuntime()
    out = rt._control_tour("", chat_key="test:1")
    assert "tour" in out
    assert "OK" in out


def test_tour_report_format():
    rt = FakeRuntime()
    out = rt._control_tour("", chat_key="test:1")
    assert "commands probed" in out
    assert "broken" in out
    assert "skipped" in out
    assert "/tour <command>" in out


def test_tour_denylist_never_probed():
    """Denylisted commands must be skipped, never dispatched."""
    probed = []

    class SpyRuntime(FakeRuntime):
        def handle_control(self, text, chat_key="", message=None):
            probed.append(text.strip().lstrip("/").split()[0].lower())
            return "ok"

    rt = SpyRuntime()
    rt._control_tour("", chat_key="test:1")
    for kind in RuntimeMetaMixin._TOUR_DENYLIST:
        assert kind not in probed, f"denylisted /{kind} was probed!"


def test_tour_denylist_covers_dangerous():
    dl = RuntimeMetaMixin._TOUR_DENYLIST
    for dangerous in ["exec", "send", "dm", "email", "quit", "upgrade",
                      "train", "trial", "mission", "money", "bet",
                      "redteam", "benchmark", "music", "video", "tour"]:
        assert dangerous in dl, f"/{dangerous} missing from denylist"


def test_tour_captures_broken_not_raises():
    rt = FakeRuntime(behavior={"research": "broken"})
    out = rt._control_tour("", chat_key="test:1")
    assert "1 broken" in out
    assert "/research" in out
    assert "UnboundLocalError" in out


def test_tour_single_broken():
    rt = FakeRuntime(behavior={"research": "broken"})
    out = rt._control_tour("research", chat_key="test:1")
    assert "BROKEN" in out
    assert "UnboundLocalError" in out


def test_tour_single_ok():
    rt = FakeRuntime()
    out = rt._control_tour("status", chat_key="test:1")
    assert "OK" in out
    assert "help:" in out


def test_tour_single_unknown():
    rt = FakeRuntime()
    out = rt._control_tour("nosuchcommand", chat_key="test:1")
    assert "no such command" in out


def test_tour_single_denylisted():
    rt = FakeRuntime()
    out = rt._control_tour("exec", chat_key="test:1")
    assert "SKIPPED" in out
    assert "denylisted" in out


def test_tour_needs_args_skipped():
    rt = FakeRuntime()
    # 'search' requires args per CONTROL_COMMANDS
    out = rt._control_tour("search", chat_key="test:1")
    assert "SKIPPED" in out
    assert "needs" in out and "arg" in out


def test_tour_slow_is_skipped_not_hung():
    rt = FakeRuntime(behavior={"news": "slow"})
    # shrink the timeout so the test is fast
    rt._TOUR_PROBE_TIMEOUT = 0.3
    out = rt._control_tour("news", chat_key="test:1")
    assert "SKIPPED" in out
    assert "slow" in out


def test_tour_probe_never_raises_on_garbage():
    rt = FakeRuntime(behavior={"status": "broken"})
    # even the probe helper itself must not raise
    res = rt._tour_probe("status", "test:1")
    assert res["verdict"] == "broken"
    assert "UnboundLocalError" in res["error"]
    res = rt._tour_probe("definitely-not-a-command", "test:1")
    assert res["verdict"] == "skip"
