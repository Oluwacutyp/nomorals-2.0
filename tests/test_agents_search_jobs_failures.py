"""Tail-105 pins: nomorals/agents/search/jobs.py census sites.

All three sites are held as exclusions. The module's storage
design is dual-store by construction: the SQLite table is
primary and ``_FALLBACK`` is the in-memory mirror — ``_store``
writes the fallback on any DB failure (and logs it at debug),
and the read side degrades to the mirror the same way.

95 — ``_load``: a ``query_one`` crash degrades to the
``_FALLBACK`` copy, or None when no copy exists. The consumer
surface stays honest: ``status()`` answers an error-shaped
``{"state": "unknown", "error": "no research job ..."}`` —
a false "not found", never a fabricated job or result.

130 — ``_list``: a ``query`` crash yields the fallback-only
list. A display surface (``list_jobs`` wraps ``_list`` for
status listings); the jobs it returns are real jobs it could
read, and the dual-store design means jobs persisted during
the outage are exactly the ones in the mirror. The weakest
of this file's three: unlike ``_store``, the read side logs
nothing — observability gap, shelved as a sketch (mirror
``_store``'s debug line), not a convert: nothing on the
surface claims the list is complete, and no decision
consumer dedupes against it (``start()`` never reads it).

201 — ``_power_active``: a ``power_mode_for`` crash returns
True (assume power active). A FAIL-OPEN, but fenced: the
docstring names the fence and the code proves it — the
worker ``_work`` re-checks authoritatively at jobs.py:312
(``power_mode_for(context).active``) before doing deep-mode
work, so the pre-gate's wrong True only lets a job reach the
worker that would have reached it anyway; the worker's own
check decides. Bounded fail-open with a named hard fence.

NOT-recording is pinned with the snapshot method: the read
paths above call no failures-ledger record function, so both
ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.agents.power as power_mod
from nomorals.agents.search import jobs as J


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("db down")


class _RaisingDB:
    def query_one(self, *a, **k):
        raise RuntimeError("db down")

    def query(self, *a, **k):
        raise RuntimeError("db down")


class _RowDB:
    def __init__(self, row):
        self._row = row

    def query_one(self, *a, **k):
        return self._row

    def query(self, *a, **k):
        return [self._row]


_ROW = {"id": "pin-db-job", "query": "gold etf flows", "mode": "quick",
        "params": "{}", "state": "done", "progress": "",
        "result": "{}", "error": "", "created_at": 1.0,
        "updated_at": 2.0}

_FALLBACK_JOB = {"id": "pin-fallback-job", "query": "q", "mode": "quick",
                 "params": {}, "state": "done", "progress": "",
                 "report": {}, "error": "", "created_at": 3.0,
                 "updated_at": 4.0}


def _preserve_fallback():
    return dict(J._FALLBACK)


def _restore_fallback(saved):
    with J._LOCK:
        J._FALLBACK.clear()
        J._FALLBACK.update(saved)


# ── 95: load crash -> memory-fallback copy (or None) ────────────

def test_95_load_db_crash_degrades_to_fallback():
    before = _snap()
    saved = _preserve_fallback()
    try:
        J._store(SimpleNamespace(db=None), dict(_FALLBACK_JOB))
        ctx = SimpleNamespace(db=_RaisingDB())
        got = J._load(ctx, "pin-fallback-job")
        assert got is not None and got["id"] == "pin-fallback-job"
        assert got["query"] == "q"
        # no fallback copy -> None (status() then answers "unknown")
        assert J._load(ctx, "pin-missing") is None
        # control: a healthy DB row loads for real
        got2 = J._load(SimpleNamespace(db=_RowDB(_ROW)), "pin-db-job")
        assert got2 is not None and got2["query"] == "gold etf flows"
        assert got2["state"] == "done"
    finally:
        _restore_fallback(saved)
    assert _snap() == before


# ── 130: list crash -> fallback-only list ───────────────────────

def test_130_list_db_crash_returns_fallback_only():
    before = _snap()
    saved = _preserve_fallback()
    try:
        with J._LOCK:
            J._FALLBACK.clear()
        J._store(SimpleNamespace(db=None), dict(_FALLBACK_JOB))
        out = J._list(SimpleNamespace(db=_RaisingDB()))
        assert [j["id"] for j in out] == ["pin-fallback-job"]
        # control: a healthy DB contributes its rows too
        out2 = J._list(SimpleNamespace(db=_RowDB(_ROW)))
        assert {j["id"] for j in out2} == {"pin-fallback-job",
                                           "pin-db-job"}
    finally:
        _restore_fallback(saved)
    assert _snap() == before


# ── 201: power check crash -> True, fenced by the worker ────────

def test_201_power_active_crash_defaults_true():
    before = _snap()
    ctx = SimpleNamespace()
    with mock.patch.object(power_mod, "power_mode_for", _raise):
        assert J._power_active(ctx) is True
    # controls: the real answer passes through both ways
    with mock.patch.object(
            power_mod, "power_mode_for",
            lambda c: SimpleNamespace(active=False)):
        assert J._power_active(ctx) is False
    with mock.patch.object(
            power_mod, "power_mode_for",
            lambda c: SimpleNamespace(active=True)):
        assert J._power_active(ctx) is True
    assert _snap() == before
