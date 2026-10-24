"""Pass #102: partner runtime failures stop vanishing.

Pins for the agents/partner failure ledger: the ledger itself, real
converted sites (relationship-graph load/save, tool-loop rendering),
the status() key, the /status control line, and the subsystem scan.
"""

from types import SimpleNamespace

import pytest

from nomorals.agents.partner import failures as partner_failures
from nomorals.agents.partner.failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


class _BadStr:
    def __str__(self):
        raise RuntimeError("cannot print me")


def test_ledger_record_and_format():
    record_failure("tool_loop.test", ValueError("bad arg"))
    assert failure_count() == 1
    items = recent_failures()
    assert len(items) == 1
    assert items[0].site == "tool_loop.test"
    assert items[0].exc_type == "ValueError"
    assert items[0].message == "bad arg"
    out = format_failures()
    assert "partner failures (1 total" in out
    assert "tool_loop.test: ValueError: bad arg" in out


def test_ledger_bounded():
    for _ in range(250):
        record_failure("tool_loop.test", ValueError("x"))
    assert failure_count() == 250
    assert len(recent_failures(limit=1000)) == 200
    assert "250 total" in format_failures()


def test_ledger_empty_and_clear():
    assert format_failures() == "no partner failures recorded."
    record_failure("tool_loop.test", ValueError("x"))
    clear_failures()
    assert failure_count() == 0
    assert recent_failures() == []
    assert format_failures() == "no partner failures recorded."


def test_ledger_unprintable_exc_never_raises():
    record_failure("tool_loop.test", _BadStr())
    items = recent_failures()
    assert len(items) == 1
    assert "unprintable" in items[0].message


def test_games_corrupt_graph_load_records_and_falls_back(tmp_path):
    from nomorals.agents.partner.runtime_games import _charagent_graph

    store = SimpleNamespace(dir=tmp_path)
    (tmp_path / "_relationships.json").write_text("{not json", encoding="utf-8")
    graph = _charagent_graph(store)
    assert graph is not None  # fresh graph, not a raise
    items = recent_failures()
    assert len(items) == 1
    assert items[0].site == "runtime_games._charagent_graph"


def test_games_failed_save_records_without_raising(tmp_path):
    from nomorals.agents.partner.runtime_games import _charagent_save_graph
    from nomorals.characters.relationships import RelationshipGraph

    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    store = SimpleNamespace(dir=blocker)  # dir is a file: write fails
    _charagent_save_graph(store, RelationshipGraph())  # must not raise
    items = recent_failures()
    assert len(items) == 1
    assert items[0].site == "runtime_games._charagent_save_graph"


def test_tool_loop_unprintable_renders_and_records():
    from nomorals.agents.partner.tool_loop import ToolCallingLoop

    assert ToolCallingLoop._render_value(_BadStr()) == "(unprintable result)"
    items = recent_failures()
    assert len(items) == 1
    assert items[0].site == "tool_loop.ToolCallingLoop._render_value:877"


def test_partner_failures_status_shape():
    from nomorals.agents.partner.runtime import _partner_failures_status

    record_failure("runtime.test", KeyError("missing"))
    st = _partner_failures_status()
    assert st["total"] == 1
    assert st["recent"][0]["site"] == "runtime.test"
    assert st["recent"][0]["exc_type"] == "KeyError"


def _control_status_instance(monkeypatch):
    from nomorals.agents import power as power_mod
    from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

    inst = object.__new__(RuntimeMetaMixin)
    mood = SimpleNamespace(label="warm", values={"affection": 50,
                                                 "energy": 60,
                                                 "frustration": 5})
    rel = SimpleNamespace(stage="close", trust=80, milestones=[1, 2],
                          fights=[])
    inst.brain = SimpleNamespace(mood=SimpleNamespace(current=lambda: mood),
                                 relationship=rel)
    inst._autonomy = None
    inst.gateway = SimpleNamespace(status=lambda: {})
    inst.context = SimpleNamespace()
    inst._stats_snapshot = lambda: {"messages": 3, "replies": 2,
                                    "errors": 0, "controls": 1}
    inst._degradation_status = lambda: {"degraded": []}
    inst._model_status_line = lambda: "model: test"
    monkeypatch.setattr(power_mod, "power_mode_for",
                        lambda ctx: SimpleNamespace(active=True))
    return inst


def test_control_status_shows_failures_only_when_present(monkeypatch):
    inst = _control_status_instance(monkeypatch)
    out = inst._control_status()
    assert "partner failures" not in out
    record_failure("runtime.test", ValueError("boom"))
    out = inst._control_status()
    assert "partner failures (1 total" in out
    assert "runtime.test: ValueError: boom" in out


def test_scan_subsystem_zero():
    from nomorals.tools.error_scan import scan

    rep = scan(["nomorals/agents/partner"])
    assert rep.findings == []
