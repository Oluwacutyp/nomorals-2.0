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

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.errors import NotFound, StorageError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table
from .db import Database

__all__ = ["Job", "QueueFull", "WorkQueue", "next_cron_fire", "parse_cron"]

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

    def _has_lock_column(self) -> bool:
        if self._lock_column is None:
            try:
                cols = {r["name"] for r in self.db.table_info(self.TABLE)}
            except Exception:  # noqa: BLE001 - table may not exist yet
                cols = set()
            self._lock_column = "queueing_lock" in cols
        return self._lock_column

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
        ttl = lease_seconds if lease_seconds is not None else self.default_lease_seconds
        now = time.time()
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
                self.db.execute(
                    f"""
                    UPDATE {self.TABLE}
                       SET status = 'leased', lease_owner = ?, lease_until = ?,
                           attempts = attempts + 1, updated_at = ?
                     WHERE id = ?
                    """,
                    (worker, now + ttl, now, row["id"]),
                )
                claimed.append(self._to_job(row, status="leased", attempts=row["attempts"] + 1))
        self.stats["leased"] += len(claimed)
        return claimed

    def lease_one(self, topic: str, worker: str = "worker", **kwargs: Any) -> Job | None:
        jobs = self.lease(topic, worker, batch=1, **kwargs)
        return jobs[0] if jobs else None

    def complete(self, job_id: str, result: Any = None) -> None:
        now = time.time()
        self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'done', result = ?, error = '', lease_owner = '',
                   lease_until = 0, updated_at = ?
             WHERE id = ?
            """,
            (json.dumps(result, default=str) if result is not None else "", now, job_id),
        )
        self.stats["completed"] += 1

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> str:
        """Record a failure. Re-queues with exponential backoff, or moves to dead."""
        row = self.db.query_one(f"SELECT * FROM {self.TABLE} WHERE id = ?", (job_id,))
        if row is None:
            raise NotFound(f"job {job_id} not found")
        now = time.time()
        attempts = int(row["attempts"])
        if retry and attempts < int(row["max_attempts"]):
            # Full jitter: avoids a thundering herd when a whole batch fails together.
            delay = min(self.backoff_cap, self.backoff_base ** attempts)
            delay = random.uniform(0.0, delay)
            self.db.execute(
                f"""
                UPDATE {self.TABLE}
                   SET status = 'ready', error = ?, available_at = ?,
                       lease_owner = '', lease_until = 0, updated_at = ?
                 WHERE id = ?
                """,
                (error[:2000], now + delay, now, job_id),
            )
            self.stats["failed"] += 1
            return "retrying"
        self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'dead', error = ?, lease_owner = '', lease_until = 0, updated_at = ?
             WHERE id = ?
            """,
            (error[:2000], now, job_id),
        )
        self.stats["dead"] += 1
        return "dead"

    def requeue(self, job_id: str, *, delay: float = 0.0) -> None:
        self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'ready', available_at = ?, lease_owner = '', lease_until = 0,
                   updated_at = ?
             WHERE id = ?
            """,
            (time.time() + delay, time.time(), job_id),
        )

    def reclaim_expired(self) -> int:
        """Return expired-lease jobs to the ready pool."""
        now = time.time()
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET status = 'ready', lease_owner = '', lease_until = 0, updated_at = ?
             WHERE status = 'leased' AND lease_until <= ?
            """,
            (now, now),
        )
        count = cursor.rowcount or 0
        self.stats["reclaimed"] += count
        if count:
            _log.debug("reclaimed %d expired leases", count)
        return count

    def extend_lease(self, job_id: str, worker: str, extra_seconds: float) -> bool:
        """Heartbeat a long-running job: push its lease deadline out.

        Only the lease owner can extend (a stolen/reclaimed lease returns
        False — the worker must stop, not keep working on a job another
        worker may have picked up).
        """
        now = time.time()
        cursor = self.db.execute(
            f"""
            UPDATE {self.TABLE}
               SET lease_until = ?, updated_at = ?
             WHERE id = ? AND status = 'leased' AND lease_owner = ?
               AND lease_until > ?
            """,
            (now + max(1.0, extra_seconds), now, job_id, worker, now),
        )
        return bool(cursor.rowcount)

    def cancel(self, job_id: str) -> bool:
        """Move a live job to ``cancelled`` (terminal; workers skip it)."""
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
        row = self.db.query_one(f"SELECT * FROM {self.TABLE} WHERE id = ?", (job_id,))
        return self._to_job(row) if row else None

    def pending(self, topic: str | None = None) -> int:
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
        return self.db.query(
            f"""
            SELECT topic, status, COUNT(*) AS n
              FROM {self.TABLE}
             GROUP BY topic, status
             ORDER BY topic, status
            """
        )

    def purge(self, topic: str | None = None, *, only_finished: bool = True) -> int:
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
                self.fail(job.id, f"{type(exc).__name__}: {exc}")
            else:
                self.complete(job.id, result)
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
    ) -> int:
        """Like :meth:`run_worker` but never exits on empty; also fires cron.

        The daemon shape: one thread runs this per topic, ``tick_recurring``
        fires due recurring schedules inline so cron jobs don't need their
        own loop.
        """
        processed = 0
        while should_stop is None or not should_stop():
            if tick_recurring:
                try:
                    self.tick_recurring()
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
                self.fail(job.id, f"{type(exc).__name__}: {exc}")
            else:
                self.complete(job.id, result)
            processed += 1
        return processed

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
    ) -> float:
        """Register (or update) a cron schedule. Returns the next fire time."""
        parse_cron(cron)  # validate eagerly
        self._ensure_recurring_table()
        now = time.time()
        nxt = next_cron_fire(cron, after=now) or 0.0
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
        return (
            self.db.delete(self.RECURRING_TABLE, "name = ?", (name,)) > 0
        )

    def list_recurring(self) -> list[dict[str, Any]]:
        self._ensure_recurring_table()
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

    def tick_recurring(self, now: float | None = None) -> list[str]:
        """Enqueue every due recurring job. Returns the fired schedule names."""
        self._ensure_recurring_table()
        self._ensure_lock_column()
        now = time.time() if now is None else now
        fired: list[str] = []
        with self.db.transaction():
            due = self.db.query(
                f'SELECT * FROM "{self.RECURRING_TABLE}" '
                f"WHERE enabled = 1 AND next_fire_at > 0 AND next_fire_at <= ?",
                (now,),
            )
            for row in due:
                name = row["name"]
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
                nxt = next_cron_fire(row["cron"], after=now) or 0.0
                self.db.execute(
                    f'UPDATE "{self.RECURRING_TABLE}" SET last_fired_at = ?, '
                    f"next_fire_at = ? WHERE name = ?",
                    (now, nxt, name),
                )
                fired.append(name)
        return fired

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
            lease_until=row["lease_until"],
            lease_owner=row["lease_owner"],
            queueing_lock=row.get("queueing_lock") or "",
        )


# ── minimal cron ─────────────────────────────────────────────────────────────
#
# Five fields: minute hour dom month dow. Each field accepts ``*``, ``*/n``,
# ``a-b``, ``a-b/n``, ``a,b,c``. Month/dow names are not supported (numbers
# only); dow is 0–6 with 0 = Sunday. Stdlib only — no croniter dependency.

_CRON_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


def _parse_cron_field(field: str, low: int, high: int) -> frozenset[int]:
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty cron field part in {field!r}")
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"bad cron step in {field!r}")
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(part)
        if not (low <= start <= high and low <= end <= high and start <= end):
            raise ValueError(f"cron value out of range in {field!r}")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError(f"empty cron field {field!r}")
    return frozenset(values)


def parse_cron(expr: str) -> tuple[frozenset[int], ...]:
    """Parse a 5-field cron expression. Raises :class:`ValueError` if bad."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron needs 5 fields (minute hour dom month dow), got {len(fields)}: {expr!r}"
        )
    try:
        return tuple(
            _parse_cron_field(field, low, high)
            for field, (low, high) in zip(fields, _CRON_BOUNDS)
        )
    except ValueError as exc:
        raise ValueError(f"invalid cron {expr!r}: {exc}") from exc


def next_cron_fire(expr: str, *, after: float | None = None) -> float | None:
    """Next fire time (unix seconds, local time) strictly after ``after``.

    Returns None when nothing fires within a year (e.g. Feb 30).
    """
    minutes, hours, doms, months, dows = parse_cron(expr)
    base = after if after is not None else time.time()
    # Start at the next minute boundary.
    cursor = int(base // 60) * 60 + 60
    limit = cursor + 366 * 24 * 3600
    while cursor <= limit:
        local = time.localtime(cursor)
        cron_dow = (local.tm_wday + 1) % 7  # cron: 0 = Sunday
        if (
            local.tm_min in minutes
            and local.tm_hour in hours
            and local.tm_mday in doms
            and local.tm_mon in months
            and cron_dow in dows
        ):
            return float(cursor)
        cursor += 60
    return None
