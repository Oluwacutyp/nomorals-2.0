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
    "IdempotencyConflict",
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

#: Additive columns, applied with guarded ALTER TABLE so databases created
#: before this sweep gain them without a storage migration.
_EXTRA_COLUMNS = (
    ("expires_at", "REAL NOT NULL DEFAULT 0"),
    ("request_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("label", "TEXT NOT NULL DEFAULT ''"),
)


class DedupeTimeout(Exception):
    """A concurrent duplicate waited too long for the in-flight execution."""


class IdempotencyConflict(Exception):
    """The Stripe 422 rule: the same key was reused with a *different*
    request fingerprint.

    Replaying a stored outcome is only safe when the request is the same
    operation. A fingerprint mismatch means the caller built the key wrong
    (e.g. a constant key reused across different steps) — executing would
    be wrong, replaying would be wrong, so we fail fast and loud.
    """


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
        # Additive columns for pre-sweep databases: guarded ALTERs, never
        # a migration — a missing column is simply added.
        for column, ddl in _EXTRA_COLUMNS:
            try:
                self.db.execute(
                    f"ALTER TABLE idempotency_keys ADD COLUMN {column} {ddl}")
            except Exception:  # noqa: BLE001 - column already there
                pass

    def _expired(self, record: dict[str, Any]) -> bool:
        """True when the record's TTL has passed (Stripe's 24h rule, but
        per-key configurable). Expired keys read as never-seen."""
        try:
            expires_at = float(record.get("expires_at") or 0.0)
        except (TypeError, ValueError):
            return False
        return bool(expires_at) and time.time() >= expires_at

    def _get_raw(self, key: str) -> dict[str, Any] | None:
        """The raw record *including* expired rows (no TTL filter).

        Used by ``_claim``: an expired row still occupies the primary key,
        so claiming must see it to delete it first — otherwise the
        ``INSERT OR IGNORE`` loses forever and the claim spins.
        """
        row = self.db.query_one(
            "SELECT key, status, result_json, error, owner, created_at,"
            " updated_at, expires_at, request_json, label"
            " FROM idempotency_keys WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        record = dict(row)
        try:
            record["value"] = json.loads(record.get("result_json") or "{}")
        except (TypeError, ValueError):
            record["value"] = {"_raw": record.get("result_json")}
        try:
            record["request"] = json.loads(record.get("request_json") or "{}")
        except (TypeError, ValueError):
            record["request"] = {}
        return record

    def get(self, key: str) -> dict[str, Any] | None:
        """The raw record, with ``value`` deserialized, or None.

        Expired records (TTL passed) read as never-seen, mirroring Stripe's
        retention rule — the next call with the key is a new operation.
        """
        row = self.db.query_one(
            "SELECT key, status, result_json, error, owner, created_at,"
            " updated_at, expires_at, request_json, label"
            " FROM idempotency_keys WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        record = dict(row)
        if self._expired(record):
            return None
        try:
            record["value"] = json.loads(record.get("result_json") or "{}")
        except (TypeError, ValueError):
            record["value"] = {"_raw": record.get("result_json")}
        try:
            record["request"] = json.loads(record.get("request_json") or "{}")
        except (TypeError, ValueError):
            record["request"] = {}
        return record

    def peek(self, key: str) -> dict[str, Any] | None:
        """Non-claiming status read: ``{"status", "owner", "age_s"}`` or
        None. For operators and UIs that must not disturb a claim."""
        record = self.get(key)
        if record is None:
            return None
        try:
            age = max(0.0, time.time() - float(record.get("updated_at") or 0.0))
        except (TypeError, ValueError):
            age = 0.0
        return {"status": record["status"], "owner": record.get("owner") or "",
                "age_s": round(age, 1), "label": record.get("label") or "",
                "error": record.get("error") or ""}

    def status(self, key: str) -> str | None:
        """The key's status, or None when the key was never seen."""
        record = self.get(key)
        return str(record["status"]) if record is not None else None

    def failed_since(self, since: float) -> list[dict[str, Any]]:
        """Failed key records updated after ``since`` (unix timestamp).

        The operator redrive list — "replay everything that failed since
        14:00" — without touching running or completed keys.
        """
        try:
            rows = self.db.query(
                "SELECT key, error, owner, label, updated_at"
                " FROM idempotency_keys WHERE status = ? AND updated_at >= ?"
                " ORDER BY updated_at DESC",
                (FAILED, float(since)),
            )
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    def redrive(self, key: str) -> bool:
        """Forget one failed key so the next call re-executes (operator
        escape hatch for a fixed downstream). True when a failed key was
        cleared; refuses to touch running/completed keys."""
        record = self.get(key)
        if record is None or record["status"] != FAILED:
            return False
        return self.clear(key)

    def clear(self, key: str) -> bool:
        """Forget a key (operator escape hatch). True when one existed."""
        cursor = self.db.execute(
            "DELETE FROM idempotency_keys WHERE key = ?", (key,))
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM idempotency_keys GROUP BY status")
        by_status = {str(r["status"]): int(r["n"]) for r in rows}
        try:
            label_rows = self.db.query(
                "SELECT label, COUNT(*) AS n FROM idempotency_keys"
                " WHERE label != '' GROUP BY label")
            by_label = {str(r["label"]): int(r["n"]) for r in label_rows}
        except Exception:  # noqa: BLE001
            by_label = {}
        return {
            "total": sum(by_status.values()),
            "by_status": by_status,
            "by_label": by_label,
        }

    def purge(self, older_than_seconds: float = 7 * 86400) -> int:
        """Delete settled records older than the cutoff.

        Only ``completed``/``failed`` records go — a ``running`` key is
        never deleted, because dropping it could let a duplicate execution
        through while the first is still in flight. Expired keys (TTL
        passed) are deleted regardless of status — except ``running``,
        which keeps its claim until it settles. Returns the number of
        rows deleted.
        """
        cutoff = time.time() - max(0.0, float(older_than_seconds))
        now = time.time()
        cursor = self.db.execute(
            "DELETE FROM idempotency_keys WHERE"
            " (status != ? AND updated_at < ?)"
            " OR (status != ? AND expires_at > 0 AND expires_at < ?)",
            (RUNNING, cutoff, RUNNING, now),
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


def _fingerprint_of(fingerprint: Any) -> str:
    """Stable string for the request fingerprint, or "" when none given."""
    if fingerprint is None:
        return ""
    if isinstance(fingerprint, str):
        return fingerprint
    return _canonical({"fp": fingerprint})


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
    fingerprint: Any = None,
    ttl_seconds: float = 0.0,
    label: str = "",
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

    ``fingerprint`` is the Stripe 422 rule: a hashable description of the
    request (params, step content). When a *completed* key is replayed with
    a different fingerprint than the original execution, an
    :class:`IdempotencyConflict` is raised instead of silently replaying a
    stale outcome — the key was built wrong and must be fixed, not hidden.

    ``ttl_seconds`` sets a retention TTL on the key (0 = keep until
    purged). After expiry the key reads as never-seen, mirroring Stripe's
    24h retention rule. ``label`` is a free-form kind tag surfaced by
    :meth:`IdempotencyStore.stats`.
    """
    tag = owner or _owner_tag()
    want_fp = _fingerprint_of(fingerprint)

    # 1. Fast path: completed keys never re-execute.
    record = store.get(key)
    if record is not None and record["status"] == COMPLETED:
        _check_fingerprint(key, record, want_fp)
        return DedupResult(value=record["value"], executed=False,
                           status=COMPLETED)

    # 2. Somebody in this process already owns the key → collapse onto it.
    with _registry_lock:
        flight = _IN_FLIGHT.get(key)
    if flight is not None:
        result = _await_flight(flight, wait_timeout=wait_timeout)
        if result.status == COMPLETED:
            _check_fingerprint(key, store.get(key) or {}, want_fp)
        return result

    # 3. Claim the key (or wait for / steal a foreign claim).
    claimed = _claim(store, key, tag, stale_after=stale_after,
                     poll_interval=poll_interval, wait_timeout=wait_timeout,
                     fingerprint=want_fp, ttl_seconds=ttl_seconds,
                     label=label or "")
    if not claimed.executed:
        if claimed.status == COMPLETED:
            _check_fingerprint(key, store.get(key) or {}, want_fp)
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


def _check_fingerprint(key: str, record: dict[str, Any],
                       want_fp: str) -> None:
    """Enforce the fingerprint rule on a completed-key replay.

    No fingerprint on either side → nothing to check (historical keys and
    callers that don't opt in keep working). A mismatch → fail fast:
    the caller is reusing a key for a different operation.
    """
    if not want_fp:
        return
    have_fp = str((record.get("request") or {}).get("fingerprint") or "")
    if have_fp and have_fp != want_fp:
        raise IdempotencyConflict(
            f"idempotency key {key} was recorded for a different request — "
            "same key, different fingerprint. Fix the key; refusing to "
            "replay a stale outcome.")


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
           wait_timeout: float, fingerprint: str = "",
           ttl_seconds: float = 0.0, label: str = "") -> DedupResult:
    """Take ownership of ``key``. Returns a DedupResult only when a foreign
    claim resolved to ``completed`` while we polled — otherwise raises
    AssertionError's cousin: it returns a marker that we own the key now.
    """
    deadline = time.monotonic() + wait_timeout
    expires_at = (time.time() + max(0.0, float(ttl_seconds or 0.0))
                  if ttl_seconds else 0.0)
    request_json = _serialize({"fingerprint": fingerprint} if fingerprint else {})
    while True:
        record = store.get(key)
        now = time.time()
        if record is None:
            raw = store._get_raw(key)
            if raw is not None:
                # expired row still occupies the key: drop it so the
                # INSERT below can claim the name (a second racer just
                # loops and re-reads — no spin, no lost claim).
                store.db.execute(
                    "DELETE FROM idempotency_keys WHERE key = ?", (key,))
                continue
            cursor = store.db.execute(
                "INSERT OR IGNORE INTO idempotency_keys"
                " (key, status, result_json, error, owner, created_at,"
                "  updated_at, expires_at, request_json, label)"
                " VALUES (?, ?, '{}', '', ?, ?, ?, ?, ?, ?)",
                (key, RUNNING, tag, now, now, expires_at, request_json,
                 label or ""),
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
                " error = '', updated_at = ?, expires_at = ?,"
                " request_json = ?, label = ? WHERE key = ? AND status = ?",
                (RUNNING, tag, now, expires_at, request_json, label or "",
                 key, FAILED),
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
