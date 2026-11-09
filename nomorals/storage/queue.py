"""Durable work queue in SQLite.

At-least-once delivery with leases. A worker takes a job by writing its own id and
a deadline into the row; if the worker dies, the lease expires and another worker
reclaims the job. This is what lets a long-running mission survive ``kill -9``
without losing in-flight work.

Why not a real broker: a personal AI on a phone has no Redis. SQLite gives
durability, atomicity, and zero daemons. The topic/payload shape means swapping
in Redis or NATS later is a contained change.
"""

from __future__ import annotations

import contextlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.errors import NotFound, StorageError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table
from ..scheduler.recurrence import (
    CronNeverMatchesError,
    CronSpec,
    CronTimezoneError,
    next_cron,
    validate_cron,
)
from .db import Database
from .failures import record_failure

__all__ = [
    "CronNeverMatchesError",
    "CronTimezoneError",
    "Job",
    "QueueFull",
    "WorkQueue",
    "next_cron_fire",
    "parse_cron",
]

_log = get_logger(__name__)


class QueueFull(StorageError):
    """Raised by :meth:`WorkQueue.enqueue` when a topic exceeds ``max_depth``."""

    code = "storage.queue_full"
    retryable = False


@dataclass
class Job:
    id: str
    topic: str
    payload: dict[str, Any]
    priority: int
    attempts: int
    max_attempts: int
    status: str
    created_at: float
    available_at: float
    lease_until: float = 0.0
    lease_owner: str = ""
    queueing_lock: str = ""
    #: F02 lease generation: a fresh random token per lease. Terminal
    #: writes (complete / fail / requeue) presented with this token
    #: land only while this exact lease still owns the job — the
    #: owner STRING is not fencing (the same worker name can hold a
    #: later generation), the token is.
    lease_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "status": self.status,
            "queueing_lock": self.queueing_lock,
            "lease_token": self.lease_token,
        }


class WorkQueue:
    """Topic-partitioned durable queue."""

    TABLE = "work_queue"

    def __init__(
        self,
        db: Database,
        *,
        default_lease_seconds: float = 300.0,
        backoff_base: float = 2.0,
        backoff_cap: float = 600.0,
    ) -> None:
        self.db = db
        self.default_lease_seconds = default_lease_seconds
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.stats = {"enqueued": 0, "leased": 0, "completed": 0, "failed": 0, "dead": 0, "reclaimed": 0}
        #: Migration 88 adds ``queueing_lock``; older databases keep working
        #: with the lock feature quietly disabled.
        self._lock_column: bool | None = None
        #: ``lease_token`` rides the same guarded-ALTER precedent: it is
        #: added lazily on first lease, never via the frozen migrations.
        self._token_column: bool | None = None
        #: Mongo program P4: the seam-backed tables, resolved lazily per
        #: call (the rowstore pickup rule) and cached per attached store.
        self._seam_cache: dict[Any, Any] = {}
        self._seam_build_lock = threading.Lock()

    def _has_lock_column(self) -> bool:
        if self._lock_column is None:
            try:
                cols = {r["name"] for r in self.db.table_info(self.TABLE)}
            except Exception as e:  # table may not exist yet
                record_failure('queue.WorkQueue._has_lock_column', e)
                cols = set()
            self._lock_column = "queueing_lock" in cols
        return self._lock_column

    def _has_token_column(self) -> bool:
        if self._token_column is None:
            try:
                cols = {r["name"] for r in self.db.table_info(self.TABLE)}
            except Exception as e:  # table may not exist yet
                record_failure('queue.WorkQueue._has_token_column', e)
                cols = set()
            self._token_column = "lease_token" in cols
        return self._token_column

    def _ensure_token_column(self) -> bool:
        """Add ``lease_token`` on databases predating it (guarded
        ALTER, the queueing_lock precedent). Returns False when the
        column is unavailable — fencing then fails closed for
        token-presenting callers rather than guessing."""
        if self._has_token_column():
            return True
        try:
            self.db.execute(
                f"ALTER TABLE {self.TABLE} "
                "ADD COLUMN lease_token TEXT NOT NULL DEFAULT ''")
            self._token_column = True
        except Exception as e:  # table may not exist yet
            record_failure('queue.WorkQueue._ensure_token_column', e)
            self._token_column = False
        return self._token_column

    # ── document-seam plumbing (Mongo program P4) ────────────────────────
    #
    # When boot attached a DocumentStore to the Database (Mongo, or the
    # shadow pair), every WorkQueue operation runs over the seam in
    # carry-v1 document shape; on the SQLite default ``db.document_store``
    # is None and the native SQL path below runs byte-for-byte as before.
    # The claim semantics are the point of the port: a lease is ONE
    # compare-and-swap on the job document (see RowTable.cas_row) whose
    # expected fields are the job's status + lease token + lease
    # deadline + attempt count as read — atomic on both seam rungs, so
    # exactly one competing claimer wins per job, and every terminal
    # write is fenced by the lease token the same way. No operation
    # needs a multi-document transaction (the seam advertises
    # ``multi_doc_transactions=False`` on Mongo and the port honors
    # that): compositions are per-document writes whose order is
    # chosen so a crash lands in a state the next operation can read
    # truthfully (the queueing lock is claimed before the job document
    # is written, so a lock can point at a not-yet-written job — the
    # claim helper polls that window, then treats the lock as stale —
    # but never at a job some other enqueue created).

    #: Collection holding queueing-lock claims on the seam.
    LOCK_COLLECTION = "work_queue_locks"

    def _seam_rows(self, table: str, *,
                   extra_columns: tuple[str, ...] = (),
                   extra_defaults: dict[str, Any] | None = None):
        """The seam-backed ``table`` for this queue's db, or None.

        None is the SQLite default (no store attached): the caller runs
        its native SQL path. Resolution is per call (organs built
        before boot attachment still route correctly); the built
        RowTable is cached per (table, store).
        """
        store = getattr(self.db, "document_store", None)
        if store is None:
            return None
        # The built RowTable is cached on the Database handle (the
        # attached_rows pattern), not on this WorkQueue: construction
        # reads PRAGMA schema once, and claimers in worker threads —
        # each typically holding its own WorkQueue — share it.
        cache = getattr(self.db, "_queue_seam_tables", None)
        if cache is None:
            cache = self._seam_cache
            with contextlib.suppress(Exception):
                # A handle that refuses attribute writes keeps the
                # per-instance cache instead — same behaviour, slower.
                self.db._queue_seam_tables = cache
        key = (table, extra_columns, id(store))
        cached = cache.get(key)
        if cached is not None:
            return cached
        # Build under a lock: concurrent first use from several
        # claimer threads must construct the RowTable once (the
        # build reads PRAGMA schema from the Database handle).
        with self._seam_build_lock:
            cached = cache.get(key)
            if cached is not None:
                return cached
            from .rowstore import RowTable, column_defaults, mapping_for

            defaults = column_defaults(self.db, table)
            for name in extra_columns:
                defaults.setdefault(
                    name, (extra_defaults or {}).get(name))
            rows = RowTable(
                store,
                mapping_for(self.db, table, extra_columns=extra_columns),
                defaults=defaults)
            cache[key] = rows
            return rows

    def _queue_rows(self):
        """The seam-backed work_queue table, or None on the native path."""
        return self._seam_rows(
            self.TABLE,
            extra_columns=("queueing_lock", "lease_token"),
            extra_defaults={"queueing_lock": "", "lease_token": ""})

    def _recurring_rows(self):
        self._ensure_recurring_table()
        return self._seam_rows(self.RECURRING_TABLE)

    def _lock_claim(self, topic: str, queueing_lock: str,
                    job_id: str) -> str | None:
        """Claim ``(topic, queueing_lock)`` for ``job_id`` on the seam.

        Returns the job id that owns the lock when it is not (or no
        longer) ``job_id`` — i.e. the dedup winner to answer with —
        and None when this call owns the lock. Insert-if-absent
        (``cas`` with ``expected=None``) is atomic on both rungs, so
        concurrent enqueues cannot both win; a lock whose job is
        missing or no longer live is stale and is taken over by a
        second cas fenced on the stale job id (the native partial
        unique index has exactly this live-only scope).
        """
        store = getattr(self.db, "document_store", None)
        rows = self._queue_rows()
        lock_id = f"{topic}\x1f{queueing_lock}"
        if store.cas(self.LOCK_COLLECTION, lock_id, None,
                     {"job_id": job_id}):
            return None
        for _ in range(3):
            current = store.get(self.LOCK_COLLECTION, lock_id)
            if current is None:  # released between the two reads
                if store.cas(self.LOCK_COLLECTION, lock_id, None,
                             {"job_id": job_id}):
                    return None
                continue
            owner = str(current.get("job_id") or "")
            if owner == job_id:
                return None
            owner_row = rows.get(owner) if owner else None
            if owner_row is None and owner:
                # The winner claims the lock BEFORE writing its job
                # document (so a lock never points at a job another
                # enqueue created): a missing document usually means
                # the winner is mid-write. Poll briefly before calling
                # the lock stale — the two cases are indistinguishable
                # at read time, and the native transaction never has
                # to tell them apart.
                deadline = time.time() + 0.5
                while owner_row is None and time.time() < deadline:
                    time.sleep(0.01)
                    owner_row = rows.get(owner)
            if owner_row is not None and owner_row["status"] in (
                    "ready", "leased"):
                return owner
            # Stale lock (job gone, terminal, or never written): take
            # it over, fenced on the job id just observed.
            if store.cas(self.LOCK_COLLECTION, lock_id,
                         {"job_id": owner}, {"job_id": job_id}):
                return None
        current = store.get(self.LOCK_COLLECTION, lock_id)
        return str((current or {}).get("job_id") or job_id)

    def _lock_release(self, row: dict[str, Any]) -> None:
        """Release ``row``'s queueing lock on the seam when the job
        leaves the live states — fenced on the job id, so a lock since
        taken over by a newer job is never deleted from under it."""
        lock = row.get("queueing_lock") or ""
        if not lock:
            return
        store = getattr(self.db, "document_store", None)
        lock_id = f"{row['topic']}\x1f{lock}"
        current = store.get(self.LOCK_COLLECTION, lock_id)
        if current is not None \
                and str(current.get("job_id") or "") == str(row["id"]):
            store.delete(self.LOCK_COLLECTION, lock_id)

    # ── producer ─────────────────────────────────────────────────────────────
    def enqueue(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
        queueing_lock: str = "",
        max_depth: int | None = None,
    ) -> str:
        """Enqueue one job. Returns the job id.

        ``queueing_lock`` is the idempotency key (Procrastinate's
        ``queueing_lock``, TaskTiger's unique tasks): when a *live* job
        (ready/leased) already holds the lock on this topic, its id is
        returned instead of creating a duplicate. Pass ``max_depth`` for
        backpressure — raises :class:`QueueFull` instead of growing the
        topic without bound.
        """
        rows = self._queue_rows()
        if rows is not None:
            return self._enqueue_seam(
                rows, topic, payload, priority=priority, delay=delay,
                max_attempts=max_attempts, queueing_lock=queueing_lock,
                max_depth=max_depth)
        now = time.time()
        if queueing_lock:
            # Lazily add the column on databases that predate it (migration
            # 88 never shipped as a numbered migration; the ALTER is guarded).
            self._ensure_lock_column()
        with self.db.transaction():
            if queueing_lock and self._has_lock_column():
                existing = self.db.query_one(
                    f"SELECT id FROM {self.TABLE} WHERE topic = ? "
                    f"AND queueing_lock = ? AND status IN ('ready','leased') "
                    f"ORDER BY created_at LIMIT 1",
                    (topic, queueing_lock),
                )
                if existing:
                    return str(existing["id"])
            if max_depth is not None:
                depth = int(
                    self.db.scalar(
                        f"SELECT COUNT(*) FROM {self.TABLE} WHERE topic = ? "
                        f"AND status IN ('ready','leased')",
                        (topic,),
                        default=0,
                    )
                )
                if depth >= max_depth:
                    raise QueueFull(
                        f"topic {topic!r} has {depth} live jobs "
                        f"(max_depth={max_depth})"
                    )
            job_id = new_id()
            row = {
                "id": job_id,
                "topic": topic,
                "payload": json.dumps(payload or {}, default=str),
                "priority": priority,
                "status": "ready",
                "attempts": 0,
                "max_attempts": max_attempts,
                "available_at": now + max(0.0, delay),
                "lease_until": 0.0,
                "lease_owner": "",
                "created_at": now,
                "updated_at": now,
            }
            if self._has_lock_column():
                row["queueing_lock"] = queueing_lock
            self.db.insert(self.TABLE, row)
        self.stats["enqueued"] += 1
        return job_id

    def enqueue_many(self, topic: str, payloads: list[dict[str, Any]], **kwargs: Any) -> list[str]:
        if self._queue_rows() is not None:
            # Seam: no multi-document transaction exists to wrap the
            # batch in (advertised, not hidden) — each enqueue is its
            # own atomic write, exactly the per-job guarantee the
            # native path's callers rely on.
            return [self.enqueue(topic, payload, **kwargs)
                    for payload in payloads]
        ids: list[str] = []
        with self.db.transaction():
            for payload in payloads:
                ids.append(self.enqueue(topic, payload, **kwargs))
        return ids

    # ── consumer ─────────────────────────────────────────────────────────────
    def lease(
        self,
        topic: str,
        worker: str = "worker",
        *,
        lease_seconds: float | None = None,
        batch: int = 1,
    ) -> list[Job]:
        """Claim up to ``batch`` ready jobs atomically.

        Priority first, then oldest. Expired leases on this topic are also
        eligible, which is how crashed workers' jobs get picked back up.
        """
        rows = self._queue_rows()
        if rows is not None:
            return self._lease_seam(
                rows, topic, worker, lease_seconds=lease_seconds,
                batch=batch)
        ttl = lease_seconds if lease_seconds is not None else self.default_lease_seconds
        now = time.time()
        has_token = self._ensure_token_column()
        claimed: list[Job] = []
        with self.db.transaction():
            rows = self.db.query(
                f"""
                SELECT * FROM {self.TABLE}
                WHERE topic = ?
                  AND (status = 'ready' AND available_at <= ?
                       OR status = 'leased' AND lease_until <= ?)
                ORDER BY priority DESC, created_at
                LIMIT ?
                """,
                (topic, now, now, batch),
            )
            for row in rows:
                # Every lease is a new generation: a fresh token the
                # worker must present to commit terminal state (F02).
                token = new_id() if has_token else ""
                if has_token:
                    self.db.execute(
                        f"""
                        UPDATE {self.TABLE}
                           SET status = 'leased', lease_owner = ?,
                               lease_until = ?, lease_token = ?,
                               attempts = attempts + 1, updated_at = ?
                         WHERE id = ?
                        """,
                        (worker, now + ttl, token, now, row["id"]),
                    )
                else:
                    self.db.execute(
                        f"""
                        UPDATE {self.TABLE}
                           SET status = 'leased', lease_owner = ?, lease_until = ?,
                               attempts = attempts + 1, updated_at = ?
                         WHERE id = ?
                        """,
                        (worker, now + ttl, now, row["id"]),
                    )
                claimed.append(self._to_job(
                    row, status="leased", attempts=row["attempts"] + 1,
                    lease_owner=worker, lease_until=now + ttl,
                    lease_token=token))
        self.stats["leased"] += len(claimed)
        return claimed

    def lease_one(self, topic: str, worker: str = "worker", **kwargs: Any) -> Job | None:
        jobs = self.lease(topic, worker, batch=1, **kwargs)
        return jobs[0] if jobs else None

    def complete(self, job_id: str, result: Any = None, *,
                 lease_token: str | None = None) -> bool:
        """Mark a job done. Returns True when the write landed.

        Fencing (F02): with ``lease_token`` — the token :meth:`lease`
        handed the worker — the update lands only while that exact
        lease generation still owns the job; a stale worker's late
        completion returns False and mutates nothing (it must not
        overwrite the current owner's state or result). Without a
        token this remains the deliberate administrative completion
        path (operator tooling, tests); workers never use it.
        """
        rows = self._queue_rows()
        if rows is not None:
            return self._complete_seam(rows, job_id, result,
                                       lease_token=lease_token)
        now = time.time()
        payload = json.dumps(result, default=str) \
            if result is not None else ""
        if lease_token:
            self._ensure_token_column()
            cursor = self.db.execute(
                f"""
                UPDATE {self.TABLE}
                   SET status = 'done', result = ?, error = '',
                       lease_owner = '', lease_until = 0,
                       lease_token = '', updated_at = ?
                 WHERE id = ? AND status = 'leased' AND lease_token = ?
                """,
                (payload, now, job_id, lease_token),
            )
            if not cursor.rowcount:
                return False
        else:
            self.db.execute(
                f"""
                UPDATE {self.TABLE}
                   SET status = 'done', result = ?, error = '', lease_owner = '',
                       lease_until = 0, updated_at = ?
                 WHERE id = ?
                """,
                (payload, now, job_id),
            )
        self.stats["completed"] += 1
        return True

    def fail(self, job_id: str, error: str = "", *, retry: bool = True,
             lease_token: str | None = None) -> str:
        """Record a failure. Re-queues with exponential backoff, or moves to dead.

        With ``lease_token`` the write is fenced like
        :meth:`complete`: a stale generation gets ``"stale"`` back
        and the current owner's job is untouched.
        """
        rows = self._queue_rows()
        if rows is not None:
            return self._fail_seam(rows, job_id, error, retry=retry,
                                   lease_token=lease_token)
        row = self.db.query_one(f"SELECT * FROM {self.TABLE} WHERE id = ?", (job_id,))
        if row is None:
            raise NotFound(f"job {job_id} not found")
        fence = ""
        fence_params: tuple[Any, ...] = ()
        if lease_token:
            self._ensure_token_column()
            if row["status"] != "leased" \
                    or str(row.get("lease_token") or "") != lease_token:
                return "stale"
            fence = " AND status = 'leased' AND lease_token = ?"
            fence_params = (lease_token,)
        now = time.time()
        attempts = int(row["attempts"])
        if retry and attempts < int(row["max_attempts"]):
            # Full jitter: avoids a thundering herd when a whole batch fails together.
            delay = min(self.backoff_cap, self.backoff_base ** attempts)
            delay = random.uniform(0.0, delay)
            cursor = self.db.execute(
                f"""
                UPDATE {self.TABLE}
                   SET status = 'ready', error = ?, available_at = ?,
                       lease_owner = '', lease_until = 0, lease_token = '',
                       updated_at = ?
                 WHERE id = ?{fence}
                """,
                (error[:2000], now + delay, now, job_id, *fence_params),
            )
            if fence and not cursor.rowcount:
                return "stale"
            self.stats["failed"] += 1
            return "retrying"
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'dead', error = ?, lease_owner = '',
                   lease_until = 0, lease_token = '', updated_at = ?
             WHERE id = ?{fence}
            """,
            (error[:2000], now, job_id, *fence_params),
        )
        if fence and not cursor.rowcount:
            return "stale"
        self.stats["dead"] += 1
        return "dead"

    def requeue(self, job_id: str, *, delay: float = 0.0,
                lease_token: str | None = None) -> bool:
        """Return a job to the ready pool. With ``lease_token`` the
        release is fenced to the current lease generation (a stale
        worker cannot release another generation's job); returns
        whether the write landed."""
        rows = self._queue_rows()
        if rows is not None:
            return self._requeue_seam(rows, job_id, delay=delay,
                                      lease_token=lease_token)
        if lease_token:
            self._ensure_token_column()
            cursor = self.db.execute(
                f"""
                UPDATE {self.TABLE}
                   SET status = 'ready', available_at = ?, lease_owner = '',
                       lease_until = 0, lease_token = '', updated_at = ?
                 WHERE id = ? AND status = 'leased' AND lease_token = ?
                """,
                (time.time() + delay, time.time(), job_id, lease_token),
            )
            return bool(cursor.rowcount)
        self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'ready', available_at = ?, lease_owner = '', lease_until = 0,
                   updated_at = ?
             WHERE id = ?
            """,
            (time.time() + delay, time.time(), job_id),
        )
        return True

    def reclaim_expired(self) -> int:
        """Return expired-lease jobs to the ready pool."""
        rows = self._queue_rows()
        if rows is not None:
            return self._reclaim_expired_seam(rows)
        now = time.time()
        token_clear = ", lease_token = ''" \
            if self._ensure_token_column() else ""
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'ready', lease_owner = '', lease_until = 0,
                   updated_at = ?{token_clear}
             WHERE status = 'leased' AND lease_until <= ?
            """,
            (now, now),
        )
        count = cursor.rowcount or 0
        self.stats["reclaimed"] += count
        if count:
            _log.debug("reclaimed %d expired leases", count)
        return count

    def extend_lease(self, job_id: str, worker: str, extra_seconds: float,
                     *, lease_token: str | None = None) -> bool:
        """Heartbeat a long-running job: push its lease deadline out.

        Only the lease owner can extend (a stolen/reclaimed lease returns
        False — the worker must stop, not keep working on a job another
        worker may have picked up). With ``lease_token`` the heartbeat
        is fenced to the lease generation too: the same worker NAME
        holding a later generation does not revive this one (F02).
        """
        rows = self._queue_rows()
        if rows is not None:
            return self._extend_lease_seam(
                rows, job_id, worker, extra_seconds,
                lease_token=lease_token)
        now = time.time()
        fence = ""
        params: tuple[Any, ...] = (
            now + max(1.0, extra_seconds), now, job_id, worker, now)
        if lease_token:
            self._ensure_token_column()
            fence = " AND lease_token = ?"
            params = (*params, lease_token)
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET lease_until = ?, updated_at = ?
             WHERE id = ? AND status = 'leased' AND lease_owner = ?
               AND lease_until > ?{fence}
            """,
            params,
        )
        return bool(cursor.rowcount)

    def cancel(self, job_id: str) -> bool:
        """Move a live job to ``cancelled`` (terminal; workers skip it)."""
        rows = self._queue_rows()
        if rows is not None:
            return self._cancel_seam(rows, job_id)
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'cancelled', lease_owner = '', lease_until = 0,
                   updated_at = ?
             WHERE id = ? AND status IN ('ready','leased')
            """,
            (time.time(), job_id),
        )
        return bool(cursor.rowcount)

    def result_of(self, job_id: str) -> Any:
        """The stored result of a finished job (None when unfinished/missing)."""
        rows = self._queue_rows()
        if rows is not None:
            row = rows.get(job_id)
            if row is None or row["status"] not in ("done", "dead"):
                return None
            try:
                return json.loads(row["result"]) if row["result"] else None
            except (json.JSONDecodeError, TypeError):
                return row["result"]
        row = self.db.query_one(
            f"SELECT status, result FROM {self.TABLE} WHERE id = ?", (job_id,)
        )
        if row is None or row["status"] not in ("done", "dead"):
            return None
        try:
            return json.loads(row["result"]) if row["result"] else None
        except (json.JSONDecodeError, TypeError):
            return row["result"]

    def wait_for(self, job_id: str, timeout: float = 30.0,
                 poll: float = 0.2) -> Any:
        """Block until ``job_id`` reaches a terminal state; return its result.

        Raises :class:`TimeoutError` on timeout and re-raises the job's error
        as :class:`StorageError` when the job died.
        """
        rows = self._queue_rows()
        if rows is not None:
            deadline = time.time() + max(0.0, timeout)
            while True:
                row = rows.get(job_id)
                if row is None:
                    raise NotFound(f"job {job_id} not found")
                if row["status"] == "done":
                    return self.result_of(job_id)
                if row["status"] == "dead":
                    raise StorageError(
                        f"job {job_id} died: {str(row['error'])[:500]}")
                if row["status"] == "cancelled":
                    raise StorageError(f"job {job_id} was cancelled")
                if time.time() >= deadline:
                    raise TimeoutError(f"timed out waiting for job {job_id}")
                time.sleep(min(poll, max(0.0, deadline - time.time())))
        deadline = time.time() + max(0.0, timeout)
        while True:
            row = self.db.query_one(
                f"SELECT status, result, error FROM {self.TABLE} WHERE id = ?",
                (job_id,),
            )
            if row is None:
                raise NotFound(f"job {job_id} not found")
            if row["status"] == "done":
                return self.result_of(job_id)
            if row["status"] == "dead":
                raise StorageError(f"job {job_id} died: {row['error'][:500]}")
            if row["status"] == "cancelled":
                raise StorageError(f"job {job_id} was cancelled")
            if time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for job {job_id}")
            time.sleep(min(poll, max(0.0, deadline - time.time())))

    def retry_dead(self, job_id: str, *, delay: float = 0.0) -> bool:
        """Replay a dead-letter job: back to ``ready`` with attempts reset.

        The QueueForge replayable-DLQ primitive; the operator has presumably
        fixed whatever killed it.
        """
        rows = self._queue_rows()
        if rows is not None:
            row = rows.get(job_id)
            if row is None or row["status"] != "dead":
                return False
            now = time.time()
            new_row = {**row, "status": "ready", "attempts": 0,
                       "error": "", "available_at": now + max(0.0, delay),
                       "lease_owner": "", "lease_until": 0.0,
                       "lease_token": "", "updated_at": now}
            return rows.cas_row(
                job_id, {"status": "dead", "updated_at": row["updated_at"]},
                new_row)
        now = time.time()
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'ready', attempts = 0, error = '',
                   available_at = ?, lease_owner = '', lease_until = 0,
                   updated_at = ?
             WHERE id = ? AND status = 'dead'
            """,
            (now + max(0.0, delay), now, job_id),
        )
        return bool(cursor.rowcount)

    def dead_jobs(self, topic: str | None = None, limit: int = 100) -> list[Job]:
        """Inspect the dead-letter queue."""
        rows = self._queue_rows()
        if rows is not None:
            found = rows.find({"status": "dead", **(
                {"topic": topic} if topic else {})})
            found.sort(key=lambda r: float(r["updated_at"]), reverse=True)
            return [self._to_job(r) for r in found[:limit]]
        clause = "status = 'dead'"
        params: list[Any] = []
        if topic:
            clause += " AND topic = ?"
            params.append(topic)
        rows = self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE {clause} "
            f"ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        )
        return [self._to_job(r) for r in rows]

    # ── introspection ────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Job | None:
        rows = self._queue_rows()
        if rows is not None:
            row = rows.get(job_id)
            return self._to_job(row) if row else None
        row = self.db.query_one(f"SELECT * FROM {self.TABLE} WHERE id = ?", (job_id,))
        return self._to_job(row) if row else None

    def pending(self, topic: str | None = None) -> int:
        rows = self._queue_rows()
        if rows is not None:
            live = {"status": {"$in": ["ready", "leased"]}}
            if topic:
                live["topic"] = topic
            return rows.count(live)
        if topic:
            return int(
                self.db.scalar(
                    f"SELECT COUNT(*) FROM {self.TABLE} WHERE topic = ? AND status IN ('ready','leased')",
                    (topic,),
                    default=0,
                )
            )
        return int(
            self.db.scalar(
                f"SELECT COUNT(*) FROM {self.TABLE} WHERE status IN ('ready','leased')", default=0
            )
        )

    def topics(self) -> list[dict[str, Any]]:
        rows = self._queue_rows()
        if rows is not None:
            counts: dict[tuple[str, str], int] = {}
            for row in rows.find(None):
                key = (str(row["topic"]), str(row["status"]))
                counts[key] = counts.get(key, 0) + 1
            return [
                {"topic": topic, "status": status, "n": n}
                for (topic, status), n in sorted(counts.items())
            ]
        return self.db.query(
            f"""
            SELECT topic, status, COUNT(*) AS n
              FROM {self.TABLE}
             GROUP BY topic, status
             ORDER BY topic, status
            """
        )

    def purge(self, topic: str | None = None, *, only_finished: bool = True) -> int:
        rows = self._queue_rows()
        if rows is not None:
            victims = rows.find(
                {"status": {"$in": ["done", "dead"]}} if only_finished
                else None)
            if topic:
                victims = [r for r in victims if r["topic"] == topic]
            deleted = 0
            for row in victims:
                if rows.delete(row["id"]):
                    self._lock_release(row)
                    deleted += 1
            return deleted
        clause = "status IN ('done','dead')" if only_finished else "1=1"
        if topic:
            return self.db.delete(self.TABLE, f"topic = ? AND {clause}", (topic,))
        return self.db.delete(self.TABLE, clause)

    def stats_snapshot(self) -> dict[str, Any]:
        return {**self.stats, "pending": self.pending()}

    # ── durability guards ────────────────────────────────────────────────────

    def reconcile(self, receipts: list[str]) -> list[str]:
        """Return receipt ids with no matching row in any status.

        A producer keeps the ids :meth:`enqueue` returned; if a job was
        dropped (row deleted or never written) it shows up here.  An empty
        list means every enqueued job is accounted for.
        """
        rows = self._queue_rows()
        if rows is not None:
            return [job_id for job_id in receipts
                    if rows.get(job_id) is None]
        missing: list[str] = []
        for job_id in receipts:
            row = self.db.query_one(
                f"SELECT id FROM {self.TABLE} WHERE id = ?", (job_id,)
            )
            if row is None:
                missing.append(job_id)
        return missing

    def detect_duplicates(self, topic: str | None = None) -> list[dict[str, Any]]:
        """Find live duplicate deliveries: same topic+payload more than once.

        Returns groups of ``{"topic": ..., "payload": ..., "ids": [...]}``
        for jobs that are still ``ready``/``leased`` (terminal states are
        the legitimate history of a retried job, not a duplicate).
        """
        rows = self._queue_rows()
        if rows is not None:
            groups: dict[tuple[str, str], list[str]] = {}
            for row in rows.find({"status": {"$in": ["ready", "leased"]}}):
                if topic and row["topic"] != topic:
                    continue
                key = (str(row["topic"]), str(row["payload"]))
                groups.setdefault(key, []).append(str(row["id"]))
            return [
                {"topic": t, "payload": p, "ids": ids}
                for (t, p), ids in groups.items() if len(ids) > 1
            ]
        clause = "status IN ('ready','leased')"
        params: list[Any] = []
        if topic:
            clause += " AND topic = ?"
            params.append(topic)
        rows = self.db.query(
            f"""
            SELECT topic, payload, GROUP_CONCAT(id) AS ids, COUNT(*) AS n
              FROM {self.TABLE}
             WHERE {clause}
             GROUP BY topic, payload
            HAVING n > 1
            """,
            tuple(params),
        )
        return [
            {"topic": r["topic"], "payload": r["payload"],
             "ids": str(r["ids"]).split(",")}
            for r in rows
        ]

    # ── worker loop ──────────────────────────────────────────────────────────
    def run_worker(
        self,
        topic: str,
        handler: Callable[[Job], Any],
        *,
        worker: str = "worker",
        max_jobs: int | None = None,
        idle_sleep: float = 0.25,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Drain ``topic`` until empty (or ``max_jobs`` reached). Returns jobs handled."""
        processed = 0
        while should_stop is None or not should_stop():
            self.reclaim_expired()
            job = self.lease_one(topic, worker)
            if job is None:
                if max_jobs is None:
                    time.sleep(idle_sleep)
                    continue
                break
            try:
                result = handler(job)
            except Exception as exc:  # noqa: BLE001 - worker must not die on one job
                self.fail(job.id, f"{type(exc).__name__}: {exc}",
                          lease_token=job.lease_token)
            else:
                # Fenced (F02): if the lease was reclaimed mid-flight,
                # the completion is refused and the current owner's
                # state stands — this worker's result is dropped.
                self.complete(job.id, result,
                              lease_token=job.lease_token)
            processed += 1
            if max_jobs is not None and processed >= max_jobs:
                break
        return processed

    def run_worker_forever(
        self,
        topic: str,
        handler: Callable[[Job], Any],
        *,
        worker: str = "worker",
        idle_sleep: float = 0.25,
        should_stop: Callable[[], bool] | None = None,
        tick_recurring: bool = True,
        timezone: str = "",
    ) -> int:
        """Like :meth:`run_worker` but never exits on empty; also fires cron.

        The daemon shape: one thread runs this per topic, ``tick_recurring``
        fires due recurring schedules inline so cron jobs don't need their
        own loop. ``timezone`` (IANA name) is passed through to the tick;
        empty computes recurring schedules on the system-local wall clock.
        """
        processed = 0
        while should_stop is None or not should_stop():
            if tick_recurring:
                try:
                    self.tick_recurring(timezone=timezone)
                except Exception as exc:  # noqa: BLE001 - cron must not kill the worker
                    _log.error("recurring tick failed: %s", exc)
            self.reclaim_expired()
            job = self.lease_one(topic, worker)
            if job is None:
                time.sleep(idle_sleep)
                continue
            try:
                result = handler(job)
            except Exception as exc:  # noqa: BLE001 - worker must not die on one job
                self.fail(job.id, f"{type(exc).__name__}: {exc}",
                          lease_token=job.lease_token)
            else:
                self.complete(job.id, result,
                              lease_token=job.lease_token)
            processed += 1
        return processed

    # ── seam implementations (Mongo program P4) ──────────────────────────
    #
    # Each mirrors its native counterpart's contract exactly; where the
    # native path leans on a conditional UPDATE, the seam leans on
    # ``cas_row`` with the fence fields taken verbatim from the row as
    # read — the write lands only while the job is still in the state
    # the decision was made from. A cas that loses a race fails CLOSED
    # (the completion/heartbeat is refused), which is the F02
    # philosophy: never guess at another generation's state.

    def _enqueue_seam(self, rows: Any, topic: str,
                      payload: dict[str, Any] | None, *,
                      priority: int, delay: float, max_attempts: int,
                      queueing_lock: str,
                      max_depth: int | None) -> str:
        now = time.time()
        job_id = new_id()
        if queueing_lock:
            # Native order: the dedup answer comes before backpressure.
            winner = self._lock_claim(topic, queueing_lock, job_id)
            if winner is not None:
                # Dedup hit: the native path returns the existing id
                # without counting an enqueue; mirror both.
                return winner
        if max_depth is not None:
            depth = rows.count({
                "topic": topic,
                "status": {"$in": ["ready", "leased"]}})
            if depth >= max_depth:
                if queueing_lock:
                    # Backpressure refused the job: hand the lock back
                    # (fenced on this call's job id) so the lock never
                    # points at a job that was never written.
                    store = getattr(self.db, "document_store", None)
                    lock_id = f"{topic}\x1f{queueing_lock}"
                    current = store.get(self.LOCK_COLLECTION, lock_id)
                    if current is not None and str(
                            current.get("job_id") or "") == job_id:
                        store.delete(self.LOCK_COLLECTION, lock_id)
                raise QueueFull(
                    f"topic {topic!r} has {depth} live jobs "
                    f"(max_depth={max_depth})")
        row = {
            "id": job_id,
            "topic": topic,
            "payload": json.dumps(payload or {}, default=str),
            "priority": priority,
            "status": "ready",
            "attempts": 0,
            "max_attempts": max_attempts,
            "available_at": now + max(0.0, delay),
            "lease_until": 0.0,
            "lease_owner": "",
            "queueing_lock": queueing_lock,
            "lease_token": "",
            "created_at": now,
            "updated_at": now,
        }
        rows.put(row)
        self.stats["enqueued"] += 1
        return job_id

    def _lease_seam(self, rows: Any, topic: str, worker: str, *,
                    lease_seconds: float | None, batch: int) -> list[Job]:
        ttl = lease_seconds if lease_seconds is not None \
            else self.default_lease_seconds
        now = time.time()
        claimed: list[Job] = []
        claimed_ids: set[str] = set()
        for _round in range(3):
            if len(claimed) >= batch:
                break
            candidates = [
                row for row in rows.find({"topic": topic})
                if (row["status"] == "ready"
                    and float(row["available_at"]) <= now)
                or (row["status"] == "leased"
                    and float(row["lease_until"]) <= now)
            ]
            # Native order: priority first, then oldest.
            candidates.sort(
                key=lambda r: (-int(r["priority"]), float(r["created_at"])))
            progressed = False
            for row in candidates:
                if len(claimed) >= batch:
                    break
                if row["id"] in claimed_ids:
                    continue
                # THE CLAIM: one cas. Every field that changes hands
                # is in the expected set, so of N competing claimers
                # exactly one wins this job; the rest re-read and move
                # on. Attempts increments inside the same write, as
                # the native UPDATE does.
                token = new_id()
                new_row = {
                    **row,
                    "status": "leased",
                    "lease_owner": worker,
                    "lease_until": now + ttl,
                    "lease_token": token,
                    "attempts": int(row["attempts"]) + 1,
                    "updated_at": now,
                }
                expected = {
                    "status": row["status"],
                    "lease_token": row.get("lease_token"),
                    "lease_until": row["lease_until"],
                    "attempts": row["attempts"],
                }
                if rows.cas_row(row["id"], expected, new_row):
                    claimed.append(self._to_job(new_row))
                    claimed_ids.add(str(row["id"]))
                    progressed = True
            if not progressed:
                break
        self.stats["leased"] += len(claimed)
        return claimed

    def _complete_seam(self, rows: Any, job_id: str, result: Any, *,
                       lease_token: str | None) -> bool:
        now = time.time()
        payload = json.dumps(result, default=str) \
            if result is not None else ""
        if lease_token:
            row = rows.get(job_id)
            if row is None or row["status"] != "leased" \
                    or (row.get("lease_token") or "") != lease_token:
                return False
            new_row = {
                **row, "status": "done", "result": payload, "error": "",
                "lease_owner": "", "lease_until": 0.0, "lease_token": "",
                "updated_at": now,
            }
            if not rows.cas_row(
                    job_id,
                    {"status": row["status"],
                     "lease_token": row.get("lease_token")},
                    new_row):
                return False
            self._lock_release(new_row)
        else:
            # Unfenced administrative completion: the native UPDATE by
            # id lands unconditionally (a missing job is a no-op that
            # still reports True); mirror it, releasing the lock when
            # the row is there to release.
            rows.update(job_id, {
                "status": "done", "result": payload, "error": "",
                "lease_owner": "", "lease_until": 0.0,
                "lease_token": "", "updated_at": now,
            })
            row = rows.get(job_id)
            if row is not None:
                self._lock_release(row)
        self.stats["completed"] += 1
        return True

    def _fail_seam(self, rows: Any, job_id: str, error: str, *,
                   retry: bool, lease_token: str | None) -> str:
        row = rows.get(job_id)
        if row is None:
            raise NotFound(f"job {job_id} not found")
        fence: dict[str, Any] | None = None
        if lease_token:
            if row["status"] != "leased" \
                    or (row.get("lease_token") or "") != lease_token:
                return "stale"
            fence = {"status": row["status"],
                     "lease_token": row.get("lease_token")}
        now = time.time()
        attempts = int(row["attempts"])
        if retry and attempts < int(row["max_attempts"]):
            delay = min(self.backoff_cap, self.backoff_base ** attempts)
            delay = random.uniform(0.0, delay)
            new_row = {
                **row, "status": "ready", "error": error[:2000],
                "available_at": now + delay, "lease_owner": "",
                "lease_until": 0.0, "lease_token": "", "updated_at": now,
            }
            if fence is not None:
                if not rows.cas_row(job_id, fence, new_row):
                    return "stale"
            else:
                rows.update(job_id, {
                    k: new_row[k] for k in (
                        "status", "error", "available_at", "lease_owner",
                        "lease_until", "lease_token", "updated_at")})
            self.stats["failed"] += 1
            return "retrying"
        new_row = {
            **row, "status": "dead", "error": error[:2000],
            "lease_owner": "", "lease_until": 0.0, "lease_token": "",
            "updated_at": now,
        }
        if fence is not None:
            if not rows.cas_row(job_id, fence, new_row):
                return "stale"
        else:
            rows.update(job_id, {
                k: new_row[k] for k in (
                    "status", "error", "lease_owner", "lease_until",
                    "lease_token", "updated_at")})
        self._lock_release(new_row)
        self.stats["dead"] += 1
        return "dead"

    def _requeue_seam(self, rows: Any, job_id: str, *, delay: float,
                      lease_token: str | None) -> bool:
        now = time.time()
        fields = {
            "status": "ready", "available_at": now + delay,
            "lease_owner": "", "lease_until": 0.0, "lease_token": "",
            "updated_at": now,
        }
        if lease_token:
            row = rows.get(job_id)
            if row is None or row["status"] != "leased" \
                    or (row.get("lease_token") or "") != lease_token:
                return False
            return rows.cas_row(
                job_id,
                {"status": row["status"],
                 "lease_token": row.get("lease_token")},
                {**row, **fields})
        return rows.update(job_id, fields)

    def _reclaim_expired_seam(self, rows: Any) -> int:
        now = time.time()
        count = 0
        for row in rows.find({"status": "leased"}):
            if float(row["lease_until"]) > now:
                continue
            new_row = {
                **row, "status": "ready", "lease_owner": "",
                "lease_until": 0.0, "lease_token": "", "updated_at": now,
            }
            if rows.cas_row(
                    row["id"],
                    {"status": row["status"],
                     "lease_token": row.get("lease_token"),
                     "lease_until": row["lease_until"]},
                    new_row):
                count += 1
        self.stats["reclaimed"] += count
        if count:
            _log.debug("reclaimed %d expired leases", count)
        return count

    def _extend_lease_seam(self, rows: Any, job_id: str, worker: str,
                           extra_seconds: float, *,
                           lease_token: str | None) -> bool:
        now = time.time()
        row = rows.get(job_id)
        if row is None or row["status"] != "leased" \
                or row["lease_owner"] != worker \
                or float(row["lease_until"]) <= now:
            return False
        expected: dict[str, Any] = {
            "status": row["status"],
            "lease_owner": row["lease_owner"],
            "lease_until": row["lease_until"],
        }
        if lease_token:
            if (row.get("lease_token") or "") != lease_token:
                return False
            expected["lease_token"] = row.get("lease_token")
        new_row = {
            **row,
            "lease_until": now + max(1.0, extra_seconds),
            "updated_at": now,
        }
        return rows.cas_row(job_id, expected, new_row)

    def _cancel_seam(self, rows: Any, job_id: str) -> bool:
        row = rows.get(job_id)
        if row is None or row["status"] not in ("ready", "leased"):
            return False
        new_row = {
            **row, "status": "cancelled", "lease_owner": "",
            "lease_until": 0.0, "updated_at": time.time(),
        }
        if not rows.cas_row(
                job_id,
                {"status": row["status"],
                 "lease_token": row.get("lease_token")},
                new_row):
            return False
        self._lock_release(new_row)
        return True

    def _tick_recurring_seam(self, rec: Any, now: float, *,
                             timezone: str = "") -> list[str]:
        """The seam twin of :meth:`tick_recurring`: same healing rules,
        same canonical cron engine, same idempotent fire (the
        queueing lock dedups a double tick — and here the fire and the
        schedule-row update are separate writes, so the lock is doing
        real work, exactly as designed)."""
        fired: list[str] = []
        for row in rec.find({"enabled": 1}):
            name = row["name"]
            stored_next = float(row["next_fire_at"])
            if stored_next <= 0.0:
                try:
                    stored_next = next_cron_fire(
                        row["cron"], after=now, timezone=timezone)
                except ValueError as exc:
                    self._note_dead_schedule(name, exc)
                    continue
                rec.update(name, {"next_fire_at": stored_next})
            else:
                anchor = max(float(row["last_fired_at"]),
                             float(row["created_at"]))
                try:
                    canonical = next_cron_fire(
                        row["cron"], after=anchor, timezone=timezone)
                except ValueError as exc:
                    self._note_dead_schedule(name, exc)
                    continue
                if canonical < stored_next:
                    stored_next = canonical
                    rec.update(name, {"next_fire_at": stored_next})
            if stored_next > now:
                continue
            lock = row["queueing_lock"] or f"recurring:{name}"
            try:
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except json.JSONDecodeError:
                payload = {}
            self.enqueue(
                row["topic"], payload,
                priority=int(row["priority"]),
                max_attempts=int(row["max_attempts"]),
                queueing_lock=lock,
            )
            try:
                nxt = next_cron_fire(row["cron"], after=now,
                                     timezone=timezone)
            except ValueError as exc:
                self._note_dead_schedule(name, exc)
                nxt = 0.0
            rec.update(name, {"last_fired_at": now, "next_fire_at": nxt})
            fired.append(name)
        return fired

    # ── recurring (cron) jobs ────────────────────────────────────────────
    #
    # huey/Procrastinate periodic tasks, stdlib edition: schedules live in a
    # ``recurring_jobs`` table (created lazily), ``tick_recurring()`` fires
    # whatever is due. Each firing enqueues with a ``queueing_lock`` of
    # ``recurring:<name>`` so a double tick can't double-fire.

    RECURRING_TABLE = "recurring_jobs"

    def _ensure_recurring_table(self) -> None:
        self.db.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.RECURRING_TABLE}" ('
            "name TEXT PRIMARY KEY, "
            "topic TEXT NOT NULL, "
            "payload TEXT NOT NULL DEFAULT '{}', "
            "cron TEXT NOT NULL, "
            "priority INTEGER NOT NULL DEFAULT 0, "
            "max_attempts INTEGER NOT NULL DEFAULT 5, "
            "queueing_lock TEXT NOT NULL DEFAULT '', "
            "last_fired_at REAL NOT NULL DEFAULT 0, "
            "next_fire_at REAL NOT NULL DEFAULT 0, "
            "created_at REAL NOT NULL, "
            "enabled INTEGER NOT NULL DEFAULT 1)"
        )

    def _ensure_lock_column(self) -> None:
        """Add ``queueing_lock`` to ``work_queue`` on databases predating it.

        Guarded ALTER (the wave-50 column-guard precedent): a no-op when the
        column already exists. The partial unique index makes the idempotency
        check race-free even without the application-level SELECT.
        """
        cols = {r["name"] for r in self.db.table_info(self.TABLE)}
        if "queueing_lock" not in cols:
            self.db.execute(
                f"ALTER TABLE {self.TABLE} ADD COLUMN queueing_lock TEXT NOT NULL DEFAULT ''"
            )
        self.db.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_work_queue_lock "
            f"ON {self.TABLE}(topic, queueing_lock) "
            f"WHERE queueing_lock != '' AND status IN ('ready','leased')"
        )
        self._lock_column = True

    def schedule_recurring(
        self,
        name: str,
        topic: str,
        cron: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 0,
        max_attempts: int = 5,
        queueing_lock: str = "",
        enabled: bool = True,
        timezone: str = "",
    ) -> float:
        """Register (or update) a cron schedule. Returns the next fire time.

        The expression is *proven* schedulable at registration time
        (:func:`~nomorals.scheduler.recurrence.validate_cron`): garbage
        raises ``ValueError``, an unknown ``timezone`` raises
        :class:`CronTimezoneError`, and an expression that can never
        match raises :class:`CronNeverMatchesError` — a dead schedule
        is refused by name instead of being stored with
        ``next_fire_at = 0`` (the retired engine's silent fallback).
        ``timezone`` is an IANA name; empty computes on the
        system-local wall clock. The zone is a computation-time
        parameter of the registering/ticking caller — it is not
        stored (the ``recurring_jobs`` schema is unchanged).
        """
        validate_cron(cron, timezone)  # refuse unschedulable by name
        self._ensure_recurring_table()
        rec = self._recurring_rows()
        if rec is not None:
            now = time.time()
            nxt = next_cron_fire(cron, after=now, timezone=timezone)
            # The native upsert preserves last_fired_at / created_at on
            # re-registration; a seam replace must do the same by hand.
            existing = rec.get(name) or {}
            rec.put({
                "name": name, "topic": topic,
                "payload": json.dumps(payload or {}, default=str),
                "cron": cron, "priority": priority,
                "max_attempts": max_attempts,
                "queueing_lock": queueing_lock,
                "last_fired_at": float(existing.get("last_fired_at") or 0.0),
                "next_fire_at": nxt,
                "created_at": float(existing.get("created_at") or now),
                "enabled": 1 if enabled else 0,
            })
            return nxt
        now = time.time()
        nxt = next_cron_fire(cron, after=now, timezone=timezone)
        with self.db.transaction():
            self.db.execute(
                f'INSERT INTO "{self.RECURRING_TABLE}" '
                "(name, topic, payload, cron, priority, max_attempts, "
                " queueing_lock, last_fired_at, next_fire_at, created_at, enabled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "topic = excluded.topic, payload = excluded.payload, "
                "cron = excluded.cron, priority = excluded.priority, "
                "max_attempts = excluded.max_attempts, "
                "queueing_lock = excluded.queueing_lock, "
                "next_fire_at = excluded.next_fire_at, "
                "enabled = excluded.enabled",
                (
                    name, topic, json.dumps(payload or {}, default=str), cron,
                    priority, max_attempts, queueing_lock, nxt, now,
                    1 if enabled else 0,
                ),
            )
        return nxt

    def unschedule_recurring(self, name: str) -> bool:
        self._ensure_recurring_table()
        rec = self._recurring_rows()
        if rec is not None:
            return rec.delete(name)
        return (
            self.db.delete(self.RECURRING_TABLE, "name = ?", (name,)) > 0
        )

    def list_recurring(self) -> list[dict[str, Any]]:
        self._ensure_recurring_table()
        rec = self._recurring_rows()
        if rec is not None:
            out = []
            for row in sorted(rec.find(None), key=lambda r: str(r["name"])):
                try:
                    payload = json.loads(row["payload"]) if row["payload"] else {}
                except json.JSONDecodeError:
                    payload = {}
                out.append({**row, "payload": payload})
            return out
        rows = self.db.query(
            f'SELECT * FROM "{self.RECURRING_TABLE}" ORDER BY name'
        )
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except json.JSONDecodeError:
                payload = {}
            out.append({**row, "payload": payload})
        return out

    def tick_recurring(self, now: float | None = None, *,
                       timezone: str = "") -> list[str]:
        """Enqueue every due recurring job. Returns the fired schedule names.

        Rows written before the cron consolidation are healed in
        place (the ``recurring_jobs`` schema is unchanged):

        * a row stored dead with ``next_fire_at = 0`` — the retired
          engine's silent ``or 0.0`` fallback — is recomputed; when
          its expression can never match, the failure is recorded by
          name in the failure ledger and the row stays dead instead
          of being silently re-skipped on every tick;
        * a row whose stored next fire came from the retired
          AND-semantics engine is tightened to the canonical (OR)
          next fire whenever that is earlier — canonical semantics
          can only move a fire earlier than AND from the same anchor,
          never later, so a stored value is only ever corrected
          toward the true schedule.

        ``timezone`` is an IANA name; empty computes on the
        system-local wall clock.
        """
        self._ensure_recurring_table()
        self._ensure_lock_column()
        now = time.time() if now is None else now
        rec = self._recurring_rows()
        if rec is not None:
            return self._tick_recurring_seam(rec, now, timezone=timezone)
        fired: list[str] = []
        with self.db.transaction():
            rows = self.db.query(
                f'SELECT * FROM "{self.RECURRING_TABLE}" WHERE enabled = 1'
            )
            for row in rows:
                name = row["name"]
                stored_next = float(row["next_fire_at"])
                if stored_next <= 0.0:
                    # Stored dead by the retired engine's silent
                    # fallback: recompute, or record by name.
                    try:
                        stored_next = next_cron_fire(
                            row["cron"], after=now, timezone=timezone)
                    except ValueError as exc:
                        self._note_dead_schedule(name, exc)
                        continue
                    self.db.execute(
                        f'UPDATE "{self.RECURRING_TABLE}" '
                        f"SET next_fire_at = ? WHERE name = ?",
                        (stored_next, name),
                    )
                else:
                    # Tighten-only correction, anchored at the last
                    # instant the stored value was computed against
                    # (the last firing, else registration).
                    anchor = max(float(row["last_fired_at"]),
                                 float(row["created_at"]))
                    try:
                        canonical = next_cron_fire(
                            row["cron"], after=anchor, timezone=timezone)
                    except ValueError as exc:
                        self._note_dead_schedule(name, exc)
                        continue
                    if canonical < stored_next:
                        stored_next = canonical
                        self.db.execute(
                            f'UPDATE "{self.RECURRING_TABLE}" '
                            f"SET next_fire_at = ? WHERE name = ?",
                            (stored_next, name),
                        )
                if stored_next > now:
                    continue
                lock = row["queueing_lock"] or f"recurring:{name}"
                try:
                    payload = json.loads(row["payload"]) if row["payload"] else {}
                except json.JSONDecodeError:
                    payload = {}
                # Idempotent fire: the queueing lock dedups a double tick.
                self.enqueue(
                    row["topic"], payload,
                    priority=int(row["priority"]),
                    max_attempts=int(row["max_attempts"]),
                    queueing_lock=lock,
                )
                try:
                    nxt = next_cron_fire(row["cron"], after=now,
                                         timezone=timezone)
                except ValueError as exc:
                    # It fired (a due row always fires); no next
                    # occurrence exists — record by name, park dead.
                    self._note_dead_schedule(name, exc)
                    nxt = 0.0
                self.db.execute(
                    f'UPDATE "{self.RECURRING_TABLE}" SET last_fired_at = ?, '
                    f"next_fire_at = ? WHERE name = ?",
                    (now, nxt, name),
                )
                fired.append(name)
        return fired

    def _note_dead_schedule(self, name: str, exc: ValueError) -> None:
        """Record a recurring schedule that can never fire, by name.

        The failure ledger + error log are the surface: the schedule
        is *reported* dead, never silently skipped (the retired
        engine stored ``next_fire_at = 0`` without a word).
        """
        record_failure(f"queue.WorkQueue.tick_recurring:{name}", exc)
        _log.error("recurring schedule %r can never fire: %s", name, exc)

    def format_status(self, theme: Any = None) -> str:
        """Human-readable queue overview through the shared style layer."""
        theme = theme or active_theme()
        topics = self.topics()
        rows = [
            (t["topic"], t["status"], t["n"]) for t in topics
        ]
        snap = self.stats_snapshot()
        parts = [
            header("work queue", theme=theme),
            *kv_lines(
                {
                    "pending": snap["pending"],
                    "enqueued": snap["enqueued"],
                    "leased": snap["leased"],
                    "completed": snap["completed"],
                    "failed": snap["failed"],
                    "dead": snap["dead"],
                    "reclaimed": snap["reclaimed"],
                },
                theme=theme,
            ),
        ]
        if rows:
            parts.append(styled_table(["topic", "status", "jobs"], rows,
                                      theme=theme))
        return "\n".join(parts)

    # ── internals ────────────────────────────────────────────────────────────
    def _to_job(self, row: dict[str, Any], **overrides: Any) -> Job:
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except json.JSONDecodeError:
            payload = {"raw": row["payload"]}
        return Job(
            id=row["id"],
            topic=row["topic"],
            payload=payload,
            priority=row["priority"],
            attempts=overrides.get("attempts", row["attempts"]),
            max_attempts=row["max_attempts"],
            status=overrides.get("status", row["status"]),
            created_at=row["created_at"],
            available_at=row["available_at"],
            lease_until=overrides.get("lease_until", row["lease_until"]),
            lease_owner=overrides.get("lease_owner", row["lease_owner"]),
            queueing_lock=row.get("queueing_lock") or "",
            lease_token=overrides.get(
                "lease_token", row.get("lease_token") or ""),
        )


# ── cron (the canonical engine) ──────────────────────────────────────────────
#
# This module used to run a private minimal cron engine: AND semantics
# for day-of-month + day-of-week, system-local time only, its own
# minute-stepping search capped at a year, and None when nothing fired
# (which ``schedule_recurring`` then stored as ``next_fire_at = 0`` —
# a schedule that silently never fired). That engine is retired: the
# wrappers below delegate to the one canonical engine,
# ``nomorals.scheduler.recurrence``, that every scheduling surface
# uses (Echelon Wave 5 + its follow-up). Deliberate behaviour deltas:
#
# * dom+dow both restricted → classic cron OR (was: AND);
# * the timezone is explicit — ``timezone`` is an IANA name and the
#   search runs on that zone's wall clock (DST-correct); the empty
#   default keeps the caller's footing, the system-local wall clock
#   via a naive local pivot (the unified scheduler's idiom). An
#   unknown name raises CronTimezoneError; nothing falls back to UTC
#   silently;
# * an expression that can never match raises CronNeverMatchesError
#   by name (was: None → a silently dead stored schedule);
# * the grammar is the canonical extended one: month/dow names, ``L``
#   / ``LW`` / ``nW``, ``n#k`` / ``nL``, dow 0 and 7 both Sunday, and
#   Vixie ``a/n`` steps (``a``..max step n, not just ``{a}``).


def parse_cron(expr: str) -> CronSpec:
    """Parse a 5-field cron expression → the canonical :class:`CronSpec`.

    Raises :class:`ValueError` on garbage. (Before the consolidation
    this returned five frozensets from the private parser; the parsed
    representation is now the shared ``CronSpec``, as on every other
    surface.)
    """
    return CronSpec(expr)


def next_cron_fire(expr: str, *, after: float | None = None,
                   timezone: str = "") -> float:
    """Next fire time (unix seconds) strictly after ``after``.

    ``timezone`` is an IANA name; empty (the default) computes on the
    system-local wall clock. Raises :class:`CronNeverMatchesError`
    when the expression can never match (e.g. ``0 0 30 2 *``),
    :class:`CronTimezoneError` for an unknown zone, and
    :class:`ValueError` for garbage — never a silent ``None``.
    """
    base = after if after is not None else time.time()
    return next_cron(expr, datetime.fromtimestamp(base), timezone)
