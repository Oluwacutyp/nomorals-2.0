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

from ..core.errors import NotFound
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .db import Database

__all__ = ["Job", "WorkQueue"]

_log = get_logger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "status": self.status,
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

    # ── producer ─────────────────────────────────────────────────────────────
    def enqueue(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
    ) -> str:
        job_id = new_id()
        now = time.time()
        self.db.insert(
            self.TABLE,
            {
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
            },
        )
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
        )
