"""Tail-86 pins: nomorals/autonomy/coordinator.py census sites.

160/165 — _pending_depth estimate guards: a crashed depth probe
falls back to the neutral prior 1.0, the same value the constant-
work organs (memory/presence/wisdom) are assigned by design. The
function is a documented rough estimator for budget slices; no
surface reports the value as a measured count.
256 — inhibitor check crash (CONVERT, tail-86 amendment): a
crashed check is NOT "no inhibitors held". The cycle defers —
no tick runs — and the deferral is recorded: a warning naming
the failed check, a cycle_deferred ledger entry whose metadata
names the error (inhibitors: [] — no fabricated names), and
stats["inhibitor_error"] carrying the cause. Healthy paths are
unchanged: held inhibitors defer by name; none held -> the
cycle runs.
330 — the cycle_failed ledger write inside the failure handler
failing is swallowed: the primary failure is already logged at
warning with exc_info; _ledger is fail-open by design.
450 — _tick_research pending_count crash -> out["queued"] = "?",
an explicit unknown marker, distinguishable from every real count.
"""
from __future__ import annotations

import time

import pytest

import nomorals.autonomy.idle as idle_mod
import nomorals.autonomy.weakness as weakness_mod
from nomorals.autonomy import coordinator as coord_mod
from nomorals.autonomy.coordinator import IdleCoordinator


def _raise(exc):
    raise exc


# ── 160 / 165: _pending_depth fallbacks ─────────────────────────────

def test_160_pending_depth_research_crash_defaults_neutral(monkeypatch, tmp_path):
    c = IdleCoordinator(tmp_path)
    monkeypatch.setattr(coord_mod._organs, "pending_count",
                        lambda db, dst=None: _raise(RuntimeError("boom")))
    depth = c._pending_depth(object())
    assert depth["research"] == 1.0
    # control: a working probe's count is used verbatim
    monkeypatch.setattr(coord_mod._organs, "pending_count",
                        lambda db, dst=None: 4)
    assert c._pending_depth(object())["research"] == 4.0


def test_165_pending_depth_weakness_crash_defaults_neutral(monkeypatch, tmp_path):
    c = IdleCoordinator(tmp_path)
    monkeypatch.setattr(weakness_mod, "open_weaknesses",
                        lambda db: _raise(RuntimeError("boom")))
    depth = c._pending_depth(object())
    assert depth["weakness"] == 1.0
    # control: a working probe's length is used verbatim
    monkeypatch.setattr(weakness_mod, "open_weaknesses",
                        lambda db: [object(), object(), object()])
    assert c._pending_depth(object())["weakness"] == 3.0


# ── 256: inhibitor check ────────────────────────────────────────────

def _run_cycle(monkeypatch, tmp_path, inhibited):
    c = IdleCoordinator(tmp_path)
    c._idle = False
    c.ledger_calls = []
    c.tick_calls = []
    monkeypatch.setattr(idle_mod, "workspace_db", lambda d: object())
    monkeypatch.setattr(idle_mod, "idle_inhibited", inhibited)
    monkeypatch.setattr(idle_mod, "inhibitors",
                        lambda db: [{"name": "probe-inhibitor"}])

    def _record_ledger(self, db, kind, ref_id, summary, **kwargs):
        self.ledger_calls.append({"kind": kind, "ref_id": ref_id,
                                  "summary": summary, **kwargs})

    def _record_tick(self, ticker, db, started, budget):
        self.tick_calls.append(ticker)
        return {"ok": False, "skipped": "probe"}

    monkeypatch.setattr(IdleCoordinator, "_ledger", _record_ledger)
    monkeypatch.setattr(IdleCoordinator, "_run_tick", _record_tick)
    c._maintenance_cycle()
    return c


def test_256_inhibitor_check_crash_defers_and_records(
        monkeypatch, tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING,
                         logger="nomorals.autonomy.coordinator"):
        c = _run_cycle(monkeypatch, tmp_path,
                       lambda db: _raise(RuntimeError("boom")))
    # the cycle deferred: no tick ran, no steps were produced
    assert c.tick_calls == []
    assert c._last_cycle is not None
    assert c._last_cycle["steps"] == {}
    # the cause is carried in stats — and no inhibitor name is
    # fabricated into the names list
    assert c._last_cycle["inhibitor_error"] == "boom"
    assert "deferred" not in c._last_cycle
    # the deferral was ledgered with the error named
    deferred = [call for call in c.ledger_calls
                if call["kind"] == "cycle_deferred"]
    assert len(deferred) == 1
    assert deferred[0]["metadata"]["error"] == "boom"
    assert deferred[0]["metadata"]["inhibitors"] == []
    assert "inhibitor check failed" in deferred[0]["summary"]
    # ...and warned about, naming the failed check
    assert any("inhibitor check failed" in rec.message
               and rec.levelno == logging.WARNING
               for rec in caplog.records)
    # control: a genuinely held inhibitor still defers by name,
    # with no error field
    c2 = _run_cycle(monkeypatch, tmp_path, lambda db: True)
    assert c2.tick_calls == []
    assert c2._last_cycle.get("deferred") == ["probe-inhibitor"]
    assert "inhibitor_error" not in c2._last_cycle
    # control: no inhibitors held -> the cycle runs its ticks
    c3 = _run_cycle(monkeypatch, tmp_path, lambda db: False)
    assert c3.tick_calls != []
    assert "deferred" not in c3._last_cycle
    assert "inhibitor_error" not in c3._last_cycle


# ── 330: failure-of-the-failure-record ──────────────────────────────

def test_330_cycle_failed_ledger_failure_swallowed(monkeypatch, tmp_path):
    c = IdleCoordinator(tmp_path)
    monkeypatch.setattr(idle_mod, "workspace_db",
                        lambda d: _raise(RuntimeError("db gone")))
    # the body raises at db acquisition; the handler's own ledger
    # attempt hits the same dead workspace_db and is swallowed
    c._maintenance_cycle()
    assert c._last_cycle is not None
    assert c._last_cycle["cycle"].startswith("cycle-")


# ── 450: research tick queued marker ────────────────────────────────

def test_450_tick_research_pending_count_crash_marks_unknown(monkeypatch, tmp_path):
    c = IdleCoordinator(tmp_path)
    c._idle = True
    monkeypatch.setattr(coord_mod._organs, "pending_count",
                        lambda db, dst=None: _raise(RuntimeError("boom")))
    out = c._tick_research(object(), time.time())
    assert out["queued"] == "?"
