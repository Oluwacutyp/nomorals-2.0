"""Proactive fallback learning is durable (proactivelearn remainder).

``with_production_sources`` built a memory-only ``ProactiveEngine``:
the CLI (``nm agent tick``) and the scheduler tick-tool fallback
learned in-process only, while the runtime boot passed the real home
and learned durably under ``<home>/data/proactive_patterns.json``.
These tests pin the fix: a home passed to the fallback path wires
the same durable store the boot uses, and a missing home stays
memory-only without a crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import modes as M


def _seed_store(home: Path, *, action: str = "file the report") -> Path:
    """Write a patterns file the way the durable engine saves it."""
    path = home / "data" / "proactive_patterns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "patterns": {"owner": [{
            "pattern_id": "pattern-seed",
            "user_id": "owner",
            "action": action,
            "context": {"hour": {"9": 5}},
            "occurrences": 5,
            "last_seen": 1700000000.0,
            "confidence": 0.5,
        }]},
        "rejections": {"owner": {action: {
            "count": 1, "last_rejected": 1700000000.0, "context": {}}}},
    }), encoding="utf-8")
    return path


def _fallback_engine(home: Path):
    """A fallback-path engine: the object ``with_production_sources`` builds."""
    modes = M.with_production_sources(M.AgentModes(), home=home)
    return modes.autonomous.proactive_engine


# ── 1. a home wires the same durable store the boot uses ──────────────────

def test_with_production_sources_home_loads_seeded_store(tmp_path):
    seed = _seed_store(tmp_path)
    assert seed.exists()
    engine = _fallback_engine(tmp_path)
    assert engine._store_path == seed
    loaded = engine._patterns.get("owner", [])
    assert [p.action for p in loaded] == ["file the report"]
    assert loaded[0].occurrences == 5
    assert engine._is_rejected("owner", "file the report") is True


def test_denial_through_fallback_persists_and_suppresses(tmp_path):
    first = _fallback_engine(tmp_path)
    # Two different actions: accepting the same action would clear its
    # rejection (the owner did it anyway) — that is the engine's own
    # rule, not a persistence failure.
    first.record_rejected("owner", "file the report")
    first.record_accepted("owner", "water the plants")
    stored = json.loads(
        (tmp_path / "data" / "proactive_patterns.json").read_text(
            encoding="utf-8"))
    assert stored["rejections"]["owner"]["file the report"]["count"] == 1
    assert stored["patterns"]["owner"][0]["occurrences"] == 1

    # A fresh engine built from the same home sees both learnings.
    second = _fallback_engine(tmp_path)
    assert second._is_rejected("owner", "file the report") is True
    assert second._patterns["owner"][0].occurrences == 1


# ── 2. no home stays memory-only ──────────────────────────────────────────

def test_with_production_sources_no_home_stays_memory_only(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = M.with_production_sources(M.AgentModes()).autonomous.proactive_engine
    assert engine._store_path is None
    engine.record_rejected("owner", "file the report")
    assert engine._is_rejected("owner", "file the report") is True
    assert list(tmp_path.rglob("proactive_patterns.json")) == []


# ── 3. home resolution: the home, never the workspace dir ────────────────

def test_home_from_settings_prefers_home_path_over_workspace(tmp_path):
    settings = SimpleNamespace(
        workspace_dir=str(tmp_path / "workspace"),
        home_path=tmp_path,
        home="~/.nomorals")
    assert M._home_from_settings(settings) == tmp_path


def test_home_from_settings_falls_back_to_home_string(tmp_path):
    assert (M._home_from_settings(SimpleNamespace(home="~/.nomorals"))
            == "~/.nomorals")
    assert M._home_from_settings(SimpleNamespace()) is None
    assert M._home_from_settings(None) is None


# ── 4. the CLI call site passes the home ──────────────────────────────────

class _FakeDB:
    """Records the SQL the proposal sink files."""

    def __init__(self) -> None:
        self.executed: list = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return None


def _recording_fallback(monkeypatch, namespace: str):
    """Patch ``with_production_sources`` at ``namespace`` with a recorder
    that delegates to the real one and remembers the home it was given."""
    real = M.with_production_sources
    seen: dict = {}

    def recorder(modes, *, proposal_sink=None, home=None):
        seen["home"] = home
        seen["modes"] = modes
        return real(modes, proposal_sink=proposal_sink, home=home)

    monkeypatch.setattr(namespace, recorder)
    return seen


def test_cli_agent_passes_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_AGENT_MODES_STORE",
                       str(tmp_path / "agent-modes.json"))
    from nomorals.cmdline.commands.agent import _cmd_agent

    seen = _recording_fallback(
        monkeypatch, "nomorals.agents.modes.with_production_sources")
    context = SimpleNamespace(
        db=_FakeDB(), settings=SimpleNamespace(home_path=tmp_path))
    args = argparse.Namespace(chat="cli", mode_args=["status"])
    assert _cmd_agent(args, context) == 0
    assert seen["home"] == tmp_path
    engine = seen["modes"].autonomous.proactive_engine
    assert engine._store_path == tmp_path / "data" / "proactive_patterns.json"


def test_cli_agent_without_settings_degrades_silently(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_AGENT_MODES_STORE",
                       str(tmp_path / "agent-modes.json"))
    from nomorals.cmdline.commands.agent import _cmd_agent

    seen = _recording_fallback(
        monkeypatch, "nomorals.agents.modes.with_production_sources")
    context = SimpleNamespace(db=_FakeDB())
    args = argparse.Namespace(chat="cli", mode_args=["status"])
    assert _cmd_agent(args, context) == 0
    assert seen["home"] is None
    assert seen["modes"].autonomous.proactive_engine._store_path is None


# ── 5. the scheduler tick-tool fallback passes the home ───────────────────

class _FakeRegistry:
    """Registry stand-in: register() supports both the plain call and
    the decorator style used by tools/autonomy.register()."""

    def __init__(self, context):
        self.context = context
        self.fns = {}

    def register(self, name, fn=None, **kw):
        if fn is None:
            def deco(f):
                self.fns[name] = f
                return f
            return deco
        self.fns[name] = fn
        return fn


def _run_tick_tool(tmp_path, monkeypatch, *, settings):
    from nomorals.tools import autonomy as ta

    seen = _recording_fallback(
        monkeypatch, "nomorals.agents.modes.with_production_sources")
    ctx = SimpleNamespace(settings=settings, gateway=None, extras={})
    reg = _FakeRegistry(ctx)
    ta.register(reg)
    out = reg.fns["__agent_modes_tick__"]()
    return out, seen


def test_scheduler_fallback_passes_home(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        workspace_dir=str(tmp_path / "ws"), home_path=tmp_path)
    out, seen = _run_tick_tool(tmp_path, monkeypatch, settings=settings)
    assert out["ok"] is True
    assert seen["home"] == tmp_path
    engine = seen["modes"].autonomous.proactive_engine
    assert engine._store_path == tmp_path / "data" / "proactive_patterns.json"


def test_scheduler_fallback_without_home_stays_memory_only(
        tmp_path, monkeypatch):
    settings = SimpleNamespace(workspace_dir=str(tmp_path / "ws"))
    out, seen = _run_tick_tool(tmp_path, monkeypatch, settings=settings)
    assert out["ok"] is True
    assert seen["home"] is None
    assert seen["modes"].autonomous.proactive_engine._store_path is None
