"""Tests for the error-system gap closures.

- Persistent journal at boot (Gap 1)
- Spine tools registered (Gap 2)
- Heartbeat helpers zero-cost when unwired + real ratios when wired (Gap 3)
- Ladders registered, supervisor watchdog (Gap 4)
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

from nomorals.core.error_system import (
    ErrorSystem,
    build_error_system,
    get_error_system,
    heartbeat,
    record_error,
    set_error_system,
)


def _tmp_home():
    d = tempfile.mkdtemp(prefix="es_test_")
    s = MagicMock()
    s.home = d
    return s, d


def test_journal_persists_across_rebuilds():
    settings, home = _tmp_home()
    es = build_error_system(settings)
    assert es.journal._path == os.path.join(home, "data", "error_journal.db")
    assert os.path.exists(es.journal._path)
    es.journal.record_incident(ValueError("boom"), subsystem="telegram")
    es.close()

    es2 = build_error_system(settings)
    rows = es2.journal.recent_incidents(limit=5)
    assert len(rows) == 1
    assert rows[0]["subsystem"] == "telegram"
    es2.close()


def test_heartbeat_noop_when_unwired():
    set_error_system(None)
    heartbeat("llm/groq", True)  # must not raise
    record_error("x", ValueError("y"))  # must not raise
    assert get_error_system() is None


def test_heartbeat_feeds_real_ratios():
    es = build_error_system(None)
    set_error_system(es)
    try:
        heartbeat("llm/groq", True)
        heartbeat("llm/groq", True)
        heartbeat("llm/groq", False)
        fails, total, ratio = es.journal.subsystem_ratio("llm/groq", 3600)
        assert total == 3 and fails == 1
        assert abs(ratio - 1 / 3) < 1e-6
    finally:
        set_error_system(None)
        es.close()


def test_default_ladders_registered():
    es = build_error_system(None)
    try:
        names = sorted(es.ladders.status_all().keys())
        assert names == ["llm", "telegram", "voice"], names
        assert es.ladders.degraded_subsystems() == []
    finally:
        es.close()


def test_health_snapshot():
    es = build_error_system(None)
    try:
        h = es.health()
        assert h["ok"] is True
        assert "budgets" in h and "ladders" in h
        assert "top_failing_24h" in h
    finally:
        es.close()


def test_supervisor_watchdog_restarts_dead_thread():
    import threading
    import time

    es = build_error_system(None)
    try:
        sup = es.supervisor_for("scheduler")
        state = {"alive": True, "restarts": 0}

        def is_alive():
            return state["alive"]

        def restart():
            state["restarts"] += 1
            state["alive"] = True

        sup.watch("tick_loop", is_alive=is_alive, restart=restart,
                  check_interval_s=0.05, strategy="one_for_one")
        time.sleep(0.15)
        assert state["restarts"] == 0  # healthy: no restart
        state["alive"] = False
        time.sleep(0.3)
        assert state["restarts"] >= 1  # dead: restarted
        assert state["alive"] is True
        sup.stop_all()
    finally:
        es.close()


def test_healer_for_creates_per_subsystem():
    es = build_error_system(None)
    try:
        h1 = es.healer_for("telegram")
        h2 = es.healer_for("telegram")
        h3 = es.healer_for("voice")
        assert h1 is h2
        assert h1 is not h3
        assert h1.subsystem == "telegram"
    finally:
        es.close()


def test_spine_tools_registered():
    from nomorals.tools import errorsys
    from nomorals.tools.registry import ToolRegistry

    ctx = MagicMock()
    ctx.error_system = None
    reg = ToolRegistry(ctx)
    # Register just this module directly — immune to other modules'
    # import-time failures polluting the full builtin registration.
    errorsys.register(reg)
    names = {t.name for t in reg._tools.values()}
    assert {"system_health", "error_budget_status", "incident_history"} <= names


def test_tools_degrade_gracefully_without_boot():
    from nomorals.tools import errorsys
    from nomorals.tools.registry import ToolRegistry

    ctx = MagicMock()
    ctx.error_system = None
    reg = ToolRegistry(ctx)
    errorsys.register(reg)
    set_error_system(None)
    tools = {t.name: t for t in reg._tools.values()}
    out = tools["system_health"].fn()
    assert out["ok"] is False and "not initialized" in out["error"]
