"""Pass #104 pins: console failures stop vanishing.

The console subsystem is never-raises by design — every best-effort
fallback below used to swallow its exception silently. Each pin drives
a real converted site and asserts BOTH halves of the contract: the
safe fallback still holds AND the failure lands in the ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nomorals.console import failures as cf
from nomorals.console.cockpit import Cockpit
from nomorals.console.commands import ConsoleCommands
from nomorals.console.dashboard import _WIDTH, _resolve_width
from nomorals.console.tui_facade import TUIFacade


@pytest.fixture(autouse=True)
def _clean_ledger():
    cf.clear_failures()
    yield
    cf.clear_failures()


def _boom():
    raise RuntimeError("boom")


def test_ledger_basics():
    cf.record_failure("console.test.site", ValueError("bad"))
    items = cf.recent_failures()
    assert len(items) == 1
    assert items[0].site == "console.test.site"
    assert items[0].exc_type == "ValueError"
    assert items[0].message == "bad"
    assert cf.failure_count() == 1


def test_ledger_bounded():
    for i in range(250):
        cf.record_failure(f"console.test.{i}", RuntimeError("x"))
    assert len(cf.recent_failures(limit=1000)) == 200
    assert cf.failure_count() == 250
    # Ring keeps the NEWEST: the oldest surviving entry is #50.
    assert cf.recent_failures(limit=1000)[-1].site == "console.test.50"


def test_ledger_unprintable_exc():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    cf.record_failure("console.test.bad", _BadStr())
    assert cf.recent_failures()[0].message.startswith("<unprintable")


def test_format_render_and_empty():
    assert cf.format_failures() == "no console failures recorded."
    cf.record_failure("console.test.site", ValueError("bad"))
    rendered = cf.format_failures()
    assert "console failures (1 total" in rendered
    assert "console.test.site: ValueError: bad" in rendered


def test_cockpit_missions_fallback_records():
    cockpit = Cockpit(missions=_boom)
    assert cockpit.missions() == []
    assert cf.recent_failures()[0].site == "cockpit.Cockpit.missions"


def test_cockpit_budget_fallback_records():
    cockpit = Cockpit(budget=_boom)
    assert cockpit.budget_state() == {}
    assert cf.recent_failures()[0].site == "cockpit.Cockpit.budget_state"


def test_commands_snapshot_fallback_records():
    cmds = ConsoleCommands(snapshot_provider=_boom)
    assert cmds._safe_snapshot() == {}
    assert cf.recent_failures()[0].site == "commands.ConsoleCommands._safe_snapshot"


def test_dashboard_width_fallback_records(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", _boom)
    assert _resolve_width(None) == _WIDTH
    assert cf.recent_failures()[0].site == "dashboard._resolve_width"


def test_godconsole_size_fallback_records(monkeypatch):
    import os

    from nomorals.console.godconsole import GodConsole

    monkeypatch.setattr(os, "get_terminal_size", _boom)
    assert GodConsole._get_size(None) == (80, 24)
    assert cf.recent_failures()[0].site == "godconsole.GodConsole._get_size"


def test_facade_list_commands_fallback_records(monkeypatch):
    import nomorals.console.tui_facade as facade_mod

    monkeypatch.setattr(facade_mod, "cmdline_commands", _boom)
    facade = TUIFacade()
    cmds = facade.list_commands()
    assert "help" in cmds  # console-native commands still listed
    assert cf.recent_failures()[0].site == "tui_facade.TUIFacade.list_commands"
    # The facade is the read-back surface for the same ledger.
    assert "tui_facade.TUIFacade.list_commands" in facade.failures()


def test_scan_subsystem_only_deliberate_sites_remain():
    """After conversion, the ONLY error_scan findings left in console
    are the 7 documented deliberate sites: 6 Ctrl-C quits in the
    watch loop (a documented way out, not a failure) and the
    EventBuffer.drain queue.Empty idiom (the drain loop's normal
    exit signal, raised on every drain). Line numbers are NOT
    asserted — record insertions above a site shift them; the
    (file, rule, context) identity is what must hold."""
    from collections import Counter

    from nomorals.tools.error_scan import scan

    root = Path(__file__).parents[1] / "nomorals" / "console"
    report = scan([str(root)])
    got = Counter((f.file.split("nomorals/console/")[-1], f.rule)
                  for f in report.findings)
    assert got == Counter({("widgets.py", "E106"): 6,
                           ("godconsole.py", "E103"): 1})
    assert all(f.rule == "E106" and "KeyboardInterrupt" in f.context
               for f in report.findings if f.file.endswith("widgets.py"))
    drain = [f for f in report.findings if f.file.endswith("godconsole.py")]
    assert len(drain) == 1 and drain[0].rule == "E103"
    assert "queue.Empty" in drain[0].context
