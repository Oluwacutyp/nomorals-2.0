"""Mesh task dispatch over the durable work queue.

Wraps :class:`nomorals.storage.queue.WorkQueue`. Targeting uses topic
namespaces: ``mesh:broadcast`` for tasks any node may claim,
``mesh:node:<id>`` for one node. A node polls its own topic first, then
broadcast. No lease-and-filter, no wasted attempts.

Broadcast semantics are work-stealing, not fan-out: a broadcast task is
claimed by the *first* node that polls it (at-least-once per task, like
every queue job). To have every node run something, dispatch one targeted
task per node from :meth:`NodeRegistry.list_active`.

Beyond the queue's own primitives this layer adds the Temporal/Celery
gold: declarative :class:`RetryPolicy` (backoff + non-retryable errors),
task heartbeats with progress checkpoints and lease extension, idempotent
dispatch via dedupe keys, schedule-to-start expiry, capability-aware
routing, and Flower-lite queue introspection.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue
from .errors import MeshError, TaskNotFound

__all__ = [
    "MeshTask",
    "MeshTasks",
    "RetryPolicy",
    "BROADCAST_TOPIC",
    "MESH_MAX_PAYLOAD_BYTES",
    "format_tasks_table",
]

_log = get_logger(__name__)


# Canonical DDL for the work queue lives in migration V5
# (nomorals.storage.migrations); this mirrors it so a MeshTasks built on a
# fresh Database (CLI, tests) works without running the full migration
# suite. IF NOT EXISTS keeps it a no-op on migrated databases.
WORK_QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS work_queue (
    id           TEXT PRIMARY KEY,
    topic        TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready',
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    available_at REAL NOT NULL DEFAULT 0,
    lease_until  REAL NOT NULL DEFAULT 0,
    lease_owner  TEXT NOT NULL DEFAULT '',
    result       TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_queue_ready
    ON work_queue(topic, status, priority DESC, available_at);
CREATE INDEX IF NOT EXISTS idx_work_queue_lease
    ON work_queue(status, lease_until);
"""

#: Task heartbeats land here: the Temporal-style progress checkpoint a
#: retried attempt can read to resume instead of restarting from zero.
PROGRESS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS mesh_task_progress (
    job_id     TEXT PRIMARY KEY,
    detail     TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0
);
"""


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break mesh task handling (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

BROADCAST_TOPIC = "mesh:broadcast"

#: Task payloads are envelopes, not file transfers — a runaway producer
#: embedding megabytes would bloat the queue DB and slow every poll.
#: Anything bigger belongs in the artifact store with a reference here.
MESH_MAX_PAYLOAD_BYTES = 1 * 1024 * 1024

#: How long a task heartbeat extends the lease when the caller doesn't
#: say otherwise (Temporal's heartbeat-timeout idea, worker-side).
DEFAULT_HEARTBEAT_LEASE_SECONDS = 300.0


def _node_topic(node_id: str) -> str:
    return f"mesh:node:{node_id}"


@dataclass
class RetryPolicy:
    """Declarative retry policy (Temporal-style).

    ``non_retryable_errors`` are case-insensitive substrings matched
    against the failure message: a validation error should go straight
    to the dead-letter queue instead of burning attempts.
    """

    initial_delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 600.0
    max_attempts: int = 5
    non_retryable_errors: list[str] = field(default_factory=list)

    def next_delay(self, attempt: int) -> float:
        """Full-jitter backoff for ``attempt`` (0-based), à la AWS:
        ``uniform(0, min(max_delay, initial_delay * backoff**attempt))``.
        """
        cap = min(self.max_delay, self.initial_delay * (self.backoff ** max(0, attempt)))
        return random.uniform(0.0, max(0.0, cap))

    def is_retryable(self, error: str) -> bool:
        lowered = (error or "").lower()
        return not any(
            marker.lower() in lowered for marker in self.non_retryable_errors
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_delay": self.initial_delay,
            "backoff": self.backoff,
            "max_delay": self.max_delay,
            "max_attempts": self.max_attempts,
            "non_retryable_errors": list(self.non_retryable_errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetryPolicy | None":
        if not data:
            return None
        try:
            return cls(
                initial_delay=float(data.get("initial_delay", 1.0)),
                backoff=float(data.get("backoff", 2.0)),
                max_delay=float(data.get("max_delay", 600.0)),
                max_attempts=int(data.get("max_attempts", 5)),
                non_retryable_errors=[str(e) for e in (data.get("non_retryable_errors") or [])],
            )
        except (TypeError, ValueError):
            return None


@dataclass
class MeshTask:
    job_id: str
    task_type: str
    payload: dict[str, Any]
    target_node: str | None
    origin_node: str
    priority: int = 0
    attempts: int = 0
    status: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "target_node": self.target_node,
            "origin_node": self.origin_node,
            "priority": self.priority,
            "attempts": self.attempts,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshTask":
        """Rebuild from :meth:`to_dict` output (hub HTTP wire format)."""
        return cls(
            job_id=str(data.get("job_id") or ""),
            task_type=str(data.get("task_type") or ""),
            payload=dict(data.get("payload") or {}),
            target_node=data.get("target_node"),
            origin_node=str(data.get("origin_node") or ""),
            priority=int(data.get("priority") or 0),
            attempts=int(data.get("attempts") or 0),
            status=str(data.get("status") or ""),
            created_at=float(data.get("created_at") or 0.0),
        )

    def describe(self) -> str:
        """One-line human summary for CLI/status output."""
        target = f"→{self.target_node[:8]}" if self.target_node else "→broadcast"
        state = f" · {self.status}" if self.status else ""
        return (
            f"◆ {self.job_id[:8]} · {self.task_type} {target} "
            f"· pri {self.priority} · attempt {self.attempts}{state}"
        )


def format_tasks_table(tasks: list[MeshTask]) -> str:
    """God-tier task table for CLI/dashboard rendering."""
    if not tasks:
        return "no mesh tasks"
    rows = [
        (
            f"◆ {t.job_id[:8]}",
            t.task_type,
            (t.target_node[:8] if t.target_node else "broadcast"),
            t.origin_node[:8] if t.origin_node else "—",
            str(t.priority),
            str(t.attempts),
            t.status or "—",
        )
        for t in tasks
    ]
    headers = ("job", "type", "target", "origin", "pri", "tries", "status")
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("─" * w for w in widths),
    ]
    lines.extend(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


class MeshTasks:
    """Dispatch and poll for mesh tasks."""

    def __init__(
        self,
        db: Database,
        queue: WorkQueue | None = None,
        registry: Any | None = None,
    ) -> None:
        self.db = db
        self.queue = queue or WorkQueue(db)
        # WorkQueue itself doesn't create its table (migrations do); a mesh
        # built on a fresh database must still work.
        self.db.executescript(WORK_QUEUE_DDL)
        self.db.executescript(PROGRESS_TABLE_DDL)
        #: Optional NodeRegistry for capability-aware routing. Kept as Any
        #: to avoid a hard import cycle (node.py doesn't import tasks.py,
        #: but staying loose keeps the seam test-friendly).
        self.registry = registry

    # ── dispatch ─────────────────────────────────────────────────────

    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        origin_node: str,
        target_node: str | None = None,
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
        dedupe_key: str | None = None,
        retry_policy: RetryPolicy | None = None,
        expire_after: float | None = None,
        target_capabilities: list[str] | None = None,
        target_labels: dict[str, str] | None = None,
        max_depth: int | None = None,
    ) -> str:
        """Queue a task. ``target_node=None`` leaves the task unclaimed by
        any specific node: the first node to poll takes it (work-stealing).

        ``dedupe_key`` makes dispatch idempotent (Temporal workflow-ID
        reuse idea): while a live task holds the key, re-dispatch returns
        the existing job id instead of queueing a duplicate.

        ``retry_policy`` is a :class:`RetryPolicy`: backoff shape plus
        non-retryable error markers. Its ``max_attempts`` wins over the
        ``max_attempts`` argument.

        ``expire_after`` is the schedule-to-start timeout in seconds: a
        task nobody picks up in time is reaped as expired instead of
        running stale.

        ``target_capabilities`` / ``target_labels`` route by selector
        (Kubernetes nodeSelector style) instead of naming a node: the
        freshest ready node matching the selector becomes the target.
        Needs a registry (pass one to the constructor).
        """
        if not task_type:
            raise ValueError("task_type is required")
        if not origin_node:
            raise ValueError("origin_node is required")
        if target_node and (target_capabilities or target_labels):
            raise ValueError(
                "target_node and target_capabilities/target_labels are "
                "mutually exclusive"
            )
        payload = payload or {}
        size = len(json.dumps(payload, default=str))
        if size > MESH_MAX_PAYLOAD_BYTES:
            from .errors import PayloadTooLarge
            raise PayloadTooLarge(
                f"task payload is {size} bytes (limit "
                f"{MESH_MAX_PAYLOAD_BYTES}); put large blobs in the artifact "
                "store and reference them instead")
        if target_capabilities or target_labels:
            target_node = self._resolve_selector(
                target_capabilities, target_labels)
        if retry_policy is not None:
            max_attempts = retry_policy.max_attempts
        envelope = {
            "task_type": task_type,
            "payload": payload,
            "target_node": target_node,
            "origin_node": origin_node,
            "enqueued_at": time.time(),
            "retry_policy": retry_policy.to_dict() if retry_policy else None,
        }
        if expire_after is not None:
            if expire_after <= 0:
                raise ValueError("expire_after must be positive")
            envelope["expire_at"] = time.time() + expire_after
        topic = BROADCAST_TOPIC if target_node is None else _node_topic(target_node)
        job_id = self.queue.enqueue(
            topic,
            envelope,
            priority=priority,
            delay=delay,
            max_attempts=max_attempts,
            queueing_lock=f"mesh:{dedupe_key}" if dedupe_key else "",
            max_depth=max_depth,
        )
        _log.info(
            "mesh task %s dispatched: %s -> %s",
            job_id, origin_node, target_node or "broadcast",
        )
        _emit("mesh.task.dispatched", {
            "job_id": job_id,
            "task_type": task_type,
            "origin_node": origin_node,
            "target_node": target_node,
            "dedupe_key": dedupe_key,
        })
        return job_id

    def dispatch_many(
        self,
        task_type: str,
        payloads: list[dict[str, Any]],
        *,
        origin_node: str,
        target_node: str | None = None,
        priority: int = 0,
        **kwargs: Any,
    ) -> list[str]:
        """Fan out one task type over many payloads (batch enqueue)."""
        return [
            self.dispatch(task_type, p, origin_node=origin_node,
                          target_node=target_node, priority=priority, **kwargs)
            for p in payloads
        ]

    def _resolve_selector(
        self,
        capabilities: list[str] | None,
        labels: dict[str, str] | None,
    ) -> str:
        if self.registry is None:
            raise ValueError(
                "target_capabilities/target_labels need a NodeRegistry "
                "(pass registry= to MeshTasks)")
        matches = self.registry.select(
            capabilities=capabilities, labels=labels, limit=1)
        if not matches:
            raise MeshError(
                f"no active node matches capabilities={capabilities or []} "
                f"labels={labels or {}}")
        node = matches[0]
        _log.info("mesh selector routed task to node %s (%s)",
                  node.node_id, node.name)
        return node.node_id

    # ── poll / lifecycle ─────────────────────────────────────────────

    def poll(
        self,
        node_id: str,
        *,
        batch: int = 5,
        lease_seconds: float | None = None,
    ) -> list[MeshTask]:
        """Claim up to ``batch`` tasks for ``node_id``.

        Checks the node's own topic first, then broadcast. Leases are
        atomic; tasks for other nodes are never touched.
        """
        if not node_id:
            raise ValueError("node_id is required")
        tasks: list[MeshTask] = []
        worker = f"mesh:{node_id}"
        for topic in (_node_topic(node_id), BROADCAST_TOPIC):
            if len(tasks) >= batch:
                break
            jobs = self.queue.lease(
                topic, worker=worker, lease_seconds=lease_seconds,
                batch=batch - len(tasks),
            )
            tasks.extend(self._to_task(j) for j in jobs)
        for task in tasks:
            _emit("mesh.task.claimed", {
                "job_id": task.job_id,
                "task_type": task.task_type,
                "node_id": node_id,
            })
        return tasks

    def heartbeat(
        self,
        job_id: str,
        node_id: str,
        detail: dict[str, Any] | None = None,
        extend_seconds: float = DEFAULT_HEARTBEAT_LEASE_SECONDS,
    ) -> bool:
        """Temporal-style activity heartbeat for a running task.

        Records ``detail`` as the progress checkpoint (a retry can resume
        from it instead of restarting) and extends the job's lease so a
        long task isn't reclaimed while it's still working. Returns False
        when the lease is already lost — the worker must stop, another
        worker may have picked the task up.
        """
        self.db.execute(
            """INSERT INTO mesh_task_progress (job_id, detail, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE
               SET detail=excluded.detail, updated_at=excluded.updated_at""",
            (job_id, json.dumps(detail or {}, default=str), time.time()),
        )
        alive = self.queue.extend_lease(
            job_id, f"mesh:{node_id}", extend_seconds)
        _emit("mesh.task.progress", {
            "job_id": job_id, "node_id": node_id,
            "lease_alive": alive, "detail_keys": sorted((detail or {}).keys()),
        })
        return alive

    def progress(self, job_id: str) -> dict[str, Any]:
        """Last heartbeat checkpoint for a task ({} when never heartbeated)."""
        row = self.db.query_one(
            "SELECT detail FROM mesh_task_progress WHERE job_id=?", (job_id,))
        if not row or not row["detail"]:
            return {}
        try:
            return json.loads(row["detail"])
        except (ValueError, TypeError):
            return {}

    def complete(self, job_id: str, result: Any = None) -> None:
        self.queue.complete(job_id, result=result)
        _emit("mesh.task.completed", {"job_id": job_id})

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        """Record a failure. Honors the task's :class:`RetryPolicy`:
        non-retryable errors go straight to the dead-letter queue."""
        job = self.queue.get(job_id)
        if job is None:
            raise TaskNotFound(f"no task with job id {job_id}")
        policy = RetryPolicy.from_dict((job.payload or {}).get("retry_policy"))
        if policy is not None and not policy.is_retryable(error):
            retry = False
            _log.info("mesh task %s hit non-retryable error; not retrying",
                      job_id)
        self.queue.fail(job_id, error=error, retry=retry)
        _emit("mesh.task.failed", {"job_id": job_id, "error": error,
                                   "retry": retry})

    def cancel(self, job_id: str) -> bool:
        """Cancel a live task. Returns False when already terminal/missing."""
        cancelled = self.queue.cancel(job_id)
        if cancelled:
            _emit("mesh.task.cancelled", {"job_id": job_id})
        return cancelled

    def result(self, job_id: str) -> Any:
        """Stored result of a finished task. Raises TaskNotFound when the
        job doesn't exist; returns None when unfinished."""
        if self.queue.get(job_id) is None:
            raise TaskNotFound(f"no task with job id {job_id}")
        return self.queue.result_of(job_id)

    def wait_for_result(self, job_id: str, timeout: float = 30.0) -> Any:
        """Block until the task finishes; return its result.

        Raises TaskNotFound for unknown ids, TimeoutError on timeout, and
        MeshError when the task died or was cancelled.
        """
        if self.queue.get(job_id) is None:
            raise TaskNotFound(f"no task with job id {job_id}")
        try:
            return self.queue.wait_for(job_id, timeout=timeout)
        except TimeoutError:
            raise
        except Exception as exc:  # StorageError from the queue layer
            raise MeshError(str(exc)) from exc

    def retry_dead(self, job_id: str, *, delay: float = 0.0) -> bool:
        """Replay a dead-letter task (operator fixed the cause)."""
        replayed = self.queue.retry_dead(job_id, delay=delay)
        if replayed:
            _emit("mesh.task.replayed", {"job_id": job_id})
        return replayed

    def dead(self, limit: int = 100) -> list[MeshTask]:
        """Inspect the dead-letter queue (mesh topics only)."""
        return [
            self._to_task(j)
            for j in self.queue.dead_jobs(limit=limit)
            if (j.topic or "").startswith("mesh:")
        ]

    def reclaim(self) -> int:
        """Return expired-lease tasks to the ready pool (crashed workers)."""
        return self.queue.reclaim_expired()

    def reap_expired(self) -> int:
        """Fail tasks whose schedule-to-start deadline passed unclaimed.

        Expired tasks go to the dead-letter queue with an explanatory
        error instead of running stale. Returns the count reaped.
        """
        now = time.time()
        rows = self.db.query(
            f"""SELECT id, payload FROM {self.queue.TABLE}
                WHERE topic LIKE 'mesh:%' AND status = 'ready'"""
        )
        reaped = 0
        for row in rows:
            try:
                payload = json.loads(row["payload"] or "{}")
            except (ValueError, TypeError):
                continue
            expire_at = payload.get("expire_at")
            if expire_at and float(expire_at) <= now:
                self.queue.fail(
                    row["id"],
                    error=f"expired: not picked up within schedule-to-start deadline",
                    retry=False,
                )
                _emit("mesh.task.expired", {"job_id": row["id"]})
                reaped += 1
        return reaped

    # ── introspection ────────────────────────────────────────────────

    def pending_count(self, node_id: str | None = None) -> int:
        """Count ready tasks: one node's topics, or all mesh topics."""
        if node_id is None:
            rows = self.db.query(
                f"SELECT COUNT(*) AS n FROM {self.queue.TABLE} "
                "WHERE topic LIKE 'mesh:%' AND status='ready'"
            )
        else:
            rows = self.db.query(
                f"SELECT COUNT(*) AS n FROM {self.queue.TABLE} "
                "WHERE topic IN (?, ?) AND status='ready'",
                (_node_topic(node_id), BROADCAST_TOPIC),
            )
        return int(rows[0]["n"]) if rows else 0

    def stats(self) -> dict[str, Any]:
        """Flower-lite queue introspection: per-topic and total counts by
        status across all mesh topics."""
        rows = self.db.query(
            f"""SELECT topic, status, COUNT(*) AS n FROM {self.queue.TABLE}
                WHERE topic LIKE 'mesh:%'
                GROUP BY topic, status"""
        )
        topics: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
        for row in rows:
            topic = str(row["topic"])
            status = str(row["status"])
            n = int(row["n"])
            topics.setdefault(topic, {})[status] = n
            totals[status] = totals.get(status, 0) + n
        return {"topics": topics, "totals": totals}

    def list_live(self, node_id: str | None = None, limit: int = 50) -> list[MeshTask]:
        """Newest live (ready/leased) tasks, optionally scoped to a node."""
        if node_id is None:
            clause = "topic LIKE 'mesh:%'"
            params: tuple[Any, ...] = ()
        else:
            clause = "topic IN (?, ?)"
            params = (_node_topic(node_id), BROADCAST_TOPIC)
        rows = self.db.query(
            f"""SELECT * FROM {self.queue.TABLE}
                WHERE {clause} AND status IN ('ready', 'leased')
                ORDER BY created_at DESC LIMIT ?""",
            (*params, limit),
        )
        return [self._to_task(self.queue._to_job(r)) for r in rows]

    @staticmethod
    def _to_task(job: Job) -> MeshTask:
        p = job.payload or {}
        return MeshTask(
            job_id=job.id,
            task_type=str(p.get("task_type") or ""),
            payload=dict(p.get("payload") or {}),
            target_node=p.get("target_node"),
            origin_node=str(p.get("origin_node") or ""),
            priority=job.priority,
            attempts=job.attempts,
            status=job.status,
            created_at=job.created_at,
        )
