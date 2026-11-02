"""Tail-109 pins for nomorals/missions/idempotency.py — 4 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was really taken (trigger == fallback AND
control != fallback, or the guarded error proven real outside the guard).

The module imports no failures-ledger record function (grep-proven), so
nothing here can record to an external ledger; its only recording
substrate is its own table, which the 469 pin reads back directly.

Sites:
* 178 E103 — __init__ additive-column ALTER loop: a duplicate-column
  error is the expected pre-sweep-database case and is swallowed so the
  store always constructs; a schema-level failure (CREATE TABLE) is NOT
  guarded and still propagates.
* 275 E104 — failed_since(): a crashed listing query returns [], the
  empty operator redrive list. No in-repo consumer; redrive() itself
  re-reads via get() and refuses non-failed keys, so the empty list can
  drive no wrong action. The stored records are untouched.
* 303 E104 — stats(): the by-label breakdown query (a column older
  databases lack) degrades to {} while by_status/total stay truthful.
* 469 E102 — dedupe()'s except BaseException is not a swallow: the
  failure is written to the key's record (status FAILED + error text)
  and the exception re-raises, so even KeyboardInterrupt leaves an
  honest failed record and the key can be re-driven.
"""

from __future__ import annotations

import types

import pytest

from nomorals.missions.idempotency import (
    COMPLETED,
    FAILED,
    IdempotencyStore,
    dedupe,
)
from nomorals.storage.db import Database


def _store() -> IdempotencyStore:
    return IdempotencyStore(Database(":memory:"))


def _fail_value() -> None:
    raise ValueError("downstream broke")


# ── 178: guarded ALTERs in __init__ ──────────────────────────────────────────

def test_178_reinit_alter_swallowed_and_store_functional() -> None:
    db = Database(":memory:")
    IdempotencyStore(db)
    # Control: outside the guard, the duplicate ALTER genuinely raises —
    # the swallow's trigger is real, not hypothetical.
    with pytest.raises(Exception) as excinfo:
        db.execute(
            "ALTER TABLE idempotency_keys ADD COLUMN label TEXT NOT NULL DEFAULT ''")
    assert isinstance(excinfo.value, Exception)
    # Trigger: re-constructing the store on the same database swallows
    # the per-column duplicate errors and yields a working store.
    store = IdempotencyStore(db)
    result = dedupe(store, "k1", lambda: {"v": 1}, label="probe")
    assert result.status == COMPLETED and result.executed


def test_178_schema_failure_still_propagates() -> None:
    class BoomDB:
        def execute_statements(self, *args, **kwargs):
            raise RuntimeError("schema boom")

        def execute(self, *args, **kwargs):
            raise RuntimeError("schema boom")

    # The guard covers only the additive ALTERs; the CREATE TABLE step
    # is unguarded, so a dead database fails construction loudly.
    with pytest.raises(RuntimeError):
        IdempotencyStore(BoomDB())


# ── 275: failed_since degrade-to-empty ───────────────────────────────────────

def test_275_failed_since_query_crash_returns_empty() -> None:
    store = _store()
    with pytest.raises(ValueError):
        dedupe(store, "k-failed", _fail_value)
    # Control: the failed key is listed while the database answers.
    rows = store.failed_since(0)
    assert any(row["key"] == "k-failed" for row in rows)
    # Trigger: a crashed listing query returns the empty list, and the
    # stored record is still intact afterwards (read-path, non-destructive).
    real_db = store.db
    store.db = types.SimpleNamespace(
        query=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    try:
        assert store.failed_since(0) == []
    finally:
        store.db = real_db
    assert store.get("k-failed")["status"] == FAILED


# ── 303: stats() by-label degradation ────────────────────────────────────────

def test_303_stats_label_query_crash_degrades_partial() -> None:
    store = _store()
    dedupe(store, "k-ok", lambda: 1, label="probe")
    with pytest.raises(ValueError):
        dedupe(store, "k-bad", _fail_value)
    # Control: both breakdowns populated while the label query answers.
    healthy = store.stats()
    assert healthy["by_label"].get("probe") == 1
    assert healthy["by_status"].get(FAILED) == 1

    class LabelBoomDB:
        def __init__(self, real):
            self.real = real

        def query(self, sql, params=()):
            if "label" in sql:
                raise RuntimeError("no label column")
            return self.real.query(sql, params)

    real_db = store.db
    store.db = LabelBoomDB(real_db)
    try:
        degraded = store.stats()
    finally:
        store.db = real_db
    # Trigger: only the label breakdown empties; the counts stay truthful.
    assert degraded["by_label"] == {}
    assert degraded["by_status"].get(FAILED) == 1
    assert degraded["total"] == healthy["total"]


# ── 469: BaseException is recorded, then re-raised ───────────────────────────

def test_469_baseexception_recorded_and_reraised() -> None:
    store = _store()

    def fn_boom():
        raise KeyboardInterrupt("ctrl-c")

    with pytest.raises(KeyboardInterrupt):
        dedupe(store, "k-int", fn_boom)
    record = store.get("k-int")
    assert record is not None and record["status"] == FAILED
    assert "KeyboardInterrupt" in (record.get("error") or "")
    # The recorded failure is a real failed key: the next call re-executes.
    result = dedupe(store, "k-int", lambda: "recovered")
    assert result.executed and result.value == "recovered"
