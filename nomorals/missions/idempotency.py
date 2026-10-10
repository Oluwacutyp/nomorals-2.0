"""Idempotency keys for mission and task execution (Wave J, L5).

A step that runs twice after a crash must not duplicate its side effects.
Every mission/task creation and every step execution can carry an
idempotency key: a stable hash of ``(kind, params, scope)``. The
key→result mapping lives in SQLite next to the mission tables (created with
``CREATE TABLE IF NOT EXISTS``, the same no-migration pattern the os
timeline uses).

Contract of :func:`dedupe`:

* key never seen, or key previously **failed** → run ``fn`` and persist
  the outcome.
* key previously **completed** → return the stored result; ``fn`` never
  runs again.
* two threads racing on the same key → exactly one executes; the other
  waits and receives the same result ("collapse to one execution").
* a key stuck ``running`` (dead owner) is stolen after ``stale_after``
  seconds, so a crashed process cannot wedge a key forever.

This module never imports ``nomorals.os`` (L6): missions is L5 and lower
layers must never reach upward.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "COMPLETED",
    "FAILED",
    "RUNNING",
    "DedupResult",
    "DedupeTimeout",
    "IdempotencyStore",
    "create_mission_once",
    "dedupe",
    "idempotency_key",
    "mission_idempotency_key",
    "step_idempotency_key",
]

_log = get_logger(__name__)

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

_STATUSES = frozenset({RUNNING, COMPLETED, FAILED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key         TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    owner       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idempotency_status ON idempotency_keys (status);
"""


class DedupeTimeout(Exception):
    """A concurrent duplicate waited too long for the in-flight execution."""


@dataclass
class DedupResult:
    """What :func:`dedupe` decided."""

    value: Any
    executed: bool  # True when ``fn`` actually ran in this call
    status: str  # "completed" | "failed"


# ── keys ─────────────────────────────────────────────────────────────────────


def _canonical(params: Mapping[str, Any]) -> str:
    """Deterministic JSON: same logical params → same bytes, always."""
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def idempotency_key(kind: str, params: Mapping[str, Any],
                    scope: str = "") -> str:
    """Stable key for ``(kind, params, scope)``.

    ``kind`` names what is being deduplicated ("mission", "mission_step",
    "task", ...). ``params`` must be JSON-serializable (anything else is
    stringified). ``scope`` disambiguates identical params in different
    scopes — e.g. the same step name in two different missions.
    """
    payload = _canonical(
        {"kind": str(kind), "params": dict(params), "scope": str(scope)})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"idem_{digest}"


def mission_idempotency_key(goal: str, name: str = "",
                            scope: str = "") -> str:
    """Key for a mission creation request."""
    return idempotency_key("mission", {"goal": goal, "name": name},
                           scope=scope)


def step_idempotency_key(mission_id: str, step: Any) -> str:
    """Key for one step execution inside a mission's retry loop."""
    return idempotency_key(
        "mission_step",
        {
            "step": str(getattr(step, "name", "") or ""),
            "goal": str(getattr(step, "goal", "") or ""),
            "role": str(getattr(step, "role", "") or ""),
        },
        scope=str(mission_id),
    )


# ── store ────────────────────────────────────────────────────────────────────


class IdempotencyStore:
    """SQLite key→result mapping, created on demand next to mission state.

    ``db`` is the shared :class:`~nomorals.storage.db.Database` — the
    table is created with ``CREATE TABLE IF NOT EXISTS`` so this store can
    never collide with a storage migration a sibling lands.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        # execute_statements is the transaction-safe multi-statement runner
        # (Database.execute only accepts a single statement).
        self.db.execute_statements(_SCHEMA)

    def get(self, key: str) -> dict[str, Any] | None:
        """The raw record, with ``value`` deserialized, or None."""
        row = self.db.query_one(
            "SELECT key, status, result_json, error, owner, created_at,"
            " updated_at FROM idempotency_keys WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        record = dict(row)
        try:
            record["value"] = json.loads(record.get("result_json") or "{}")
        except (TypeError, ValueError):
            record["value"] = {"_raw": record.get("result_json")}
        return record

    def status(self, key: str) -> str | None:
        """The key's status, or None when the key was never seen."""
        row = self.db.query_one(
            "SELECT status FROM idempotency_keys WHERE key = ?", (key,))
        return str(row["status"]) if row is not None else None

    def clear(self, key: str) -> bool:
        """Forget a key (operator escape hatch). True when one existed."""
        cursor = self.db.execute(
            "DELETE FROM idempotency_keys WHERE key = ?", (key,))
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM idempotency_keys GROUP BY status")
        by_status = {str(r["status"]): int(r["n"]) for r in rows}
        return {
            "total": sum(by_status.values()),
            "by_status": by_status,
        }

    def purge(self, older_than_seconds: float = 7 * 86400) -> int:
        """Delete settled records older than the cutoff.

        Only ``completed``/``failed`` records go — a ``running`` key is
        never deleted, because dropping it could let a duplicate execution
        through while the first is still in flight. Returns the number of
        rows deleted.
        """
        cutoff = time.time() - max(0.0, float(older_than_seconds))
        cursor = self.db.execute(
            "DELETE FROM idempotency_keys WHERE status != ? AND updated_at < ?",
            (RUNNING, cutoff),
        )
        return int(cursor.rowcount or 0)


# ── in-process claim tracking ────────────────────────────────────────────────
#
# Concurrent duplicates inside one process collapse via a condition
# variable: the first thread registers the in-flight claim, executes, and
# wakes the waiters with the outcome. Duplicates from *other* processes see
# the "running" row and poll until it resolves or goes stale.


@dataclass
class _InFlight:
    cond: threading.Condition = field(
        default_factory=threading.Condition)
    done: bool = False
    value: Any = None
    executed_status: str = COMPLETED
    error: BaseException | None = None


_registry_lock = threading.Lock()
_IN_FLIGHT: dict[str, _InFlight] = {}


def _owner_tag() -> str:
    return f"pid:{os.getpid()}:tid:{threading.get_ident()}"


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str,
                      sort_keys=True)


def _deserialize(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": raw}


# ── dedupe ───────────────────────────────────────────────────────────────────


def dedupe(
    store: IdempotencyStore,
    key: str,
    fn: Callable[[], Any],
    *,
    owner: str = "",
    succeeded: Callable[[Any], bool] | None = None,
    stale_after: float = 300.0,
    poll_interval: float = 0.05,
    wait_timeout: float = 3600.0,
) -> DedupResult:
    """Run ``fn`` at most once per ``key``.

    Returns a :class:`DedupResult`. When the key already completed, the
    stored value is returned with ``executed=False`` and ``fn`` never runs.
    When the key failed (or was never seen), ``fn`` runs; its return value
    is persisted as JSON.

    ``succeeded(value)`` decides whether a returned value counts as a
    success. When it returns False the value is still handed back to the
    caller, but the key is recorded as ``failed`` so a later call retries
    (this is how the mission runner treats a step that ran but reported
    ``ok=False``). When ``fn`` itself raises, the key is recorded as
    failed and the exception propagates.
    """
    tag = owner or _owner_tag()

    # 1. Fast path: completed keys never re-execute.
    record = store.get(key)
    if record is not None and record["status"] == COMPLETED:
        return DedupResult(value=record["value"], executed=False,
                           status=COMPLETED)

    # 2. Somebody in this process already owns the key → collapse onto it.
    with _registry_lock:
        flight = _IN_FLIGHT.get(key)
    if flight is not None:
        return _await_flight(flight, wait_timeout=wait_timeout)

    # 3. Claim the key (or wait for / steal a foreign claim).
    claimed = _claim(store, key, tag, stale_after=stale_after,
                     poll_interval=poll_interval, wait_timeout=wait_timeout)
    if not claimed.executed:
        return claimed  # a foreign completed claim resolved while polling

    # 4. We own it: register in-flight, run, persist, wake waiters.
    flight = _InFlight()
    with _registry_lock:
        _IN_FLIGHT[key] = flight
    try:
        value = fn()
        ok = succeeded(value) if succeeded is not None else True
        store.db.execute(
            "UPDATE idempotency_keys SET status = ?, result_json = ?,"
            " error = ?, updated_at = ? WHERE key = ?",
            (COMPLETED if ok else FAILED,
             _serialize(value),
             "" if ok else f"succeeded() returned False for {type(value).__name__}",
             time.time(), key),
        )
        _resolve_flight(key, flight, value, COMPLETED if ok else FAILED,
                        error=None)
        return DedupResult(value=value, executed=True,
                           status=COMPLETED if ok else FAILED)
    except BaseException as exc:  # noqa: BLE001 - the failure *is* the record
        store.db.execute(
            "UPDATE idempotency_keys SET status = ?, error = ?,"
            " updated_at = ? WHERE key = ?",
            (FAILED, f"{type(exc).__name__}: {exc}", time.time(), key),
        )
        _resolve_flight(key, flight, None, FAILED, error=exc)
        raise
    finally:
        with _registry_lock:
            _IN_FLIGHT.pop(key, None)


def _await_flight(flight: _InFlight, *, wait_timeout: float) -> DedupResult:
    """Block until the owning thread finishes, then take its outcome."""
    deadline = time.monotonic() + wait_timeout
    with flight.cond:
        while not flight.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DedupeTimeout(
                    "timed out waiting for the in-flight idempotent execution")
            flight.cond.wait(timeout=min(remaining, 5.0))
        if flight.error is not None:
            raise flight.error
        return DedupResult(value=flight.value, executed=False,
                           status=flight.executed_status)


def _resolve_flight(key: str, flight: _InFlight, value: Any, status: str,
                    error: BaseException | None) -> None:
    with flight.cond:
        flight.value = value
        flight.executed_status = status
        flight.error = error
        flight.done = True
        flight.cond.notify_all()
    _log.debug("idempotency key %s resolved: %s", key, status)


def _claim(store: IdempotencyStore, key: str, tag: str, *,
           stale_after: float, poll_interval: float,
           wait_timeout: float) -> DedupResult:
    """Take ownership of ``key``. Returns a DedupResult only when a foreign
    claim resolved to ``completed`` while we polled — otherwise raises
    AssertionError's cousin: it returns a marker that we own the key now.
    """
    deadline = time.monotonic() + wait_timeout
    while True:
        record = store.get(key)
        now = time.time()
        if record is None:
            cursor = store.db.execute(
                "INSERT OR IGNORE INTO idempotency_keys"
                " (key, status, result_json, error, owner, created_at,"
                "  updated_at) VALUES (?, ?, '{}', '', ?, ?, ?)",
                (key, RUNNING, tag, now, now),
            )
            if cursor.rowcount == 1:
                return DedupResult(value=None, executed=True, status=RUNNING)
            continue  # lost the race: re-read and decide
        status = record["status"]
        if status == COMPLETED:
            return DedupResult(value=record["value"], executed=False,
                               status=COMPLETED)
        if status == FAILED:
            # A failed key is retryable: take it over.
            cursor = store.db.execute(
                "UPDATE idempotency_keys SET status = ?, owner = ?,"
                " error = '', updated_at = ? WHERE key = ? AND status = ?",
                (RUNNING, tag, now, key, FAILED),
            )
            if cursor.rowcount == 1:
                return DedupResult(value=None, executed=True, status=RUNNING)
            continue  # lost the race: re-read
        # status == RUNNING: owned by someone else. Wait for it, then steal
        # it once it is older than stale_after.
        age = now - float(record.get("updated_at") or 0.0)
        if age >= stale_after:
            cursor = store.db.execute(
                "UPDATE idempotency_keys SET status = ?, owner = ?,"
                " updated_at = ? WHERE key = ? AND status = ?"
                " AND updated_at <= ?",
                (RUNNING, tag, now, key, RUNNING, now - stale_after),
            )
            if cursor.rowcount == 1:
                _log.warning("idempotency key %s stolen from stale owner %s",
                             key, record.get("owner"))
                return DedupResult(value=None, executed=True, status=RUNNING)
            continue
        if time.monotonic() >= deadline:
            raise DedupeTimeout(
                f"idempotency key {key} stayed running for the whole wait")
        time.sleep(poll_interval)


# ── mission-creation helper ──────────────────────────────────────────────────


def create_mission_once(mission_store: Any, idem: IdempotencyStore, key: str,
                        goal: str, **kwargs: Any) -> tuple[Any, bool]:
    """Create a mission exactly once per ``key``.

    Returns ``(mission, created)``. A duplicate submission with the same
    key returns the original mission without creating a second row —
    ``created`` tells the caller which happened.
    """
    def _create() -> dict[str, Any]:
        mission = mission_store.create_new(goal, **kwargs)
        return {"mission_id": mission.id}

    result = dedupe(idem, key, _create, owner="mission-create")
    mission_id = result.value.get("mission_id") if isinstance(
        result.value, dict) else None
    if not mission_id:
        raise RuntimeError(
            f"idempotency record for {key} has no mission_id")
    return mission_store.get(mission_id), result.executed
