"""Mesh task dispatch over the durable work queue.

Wraps :class:`nomorals.storage.queue.WorkQueue`. Targeting uses topic
namespaces: ``mesh:broadcast`` for tasks any node may claim,
``mesh:node:<id>`` for one node. A node polls its own topic first, then
broadcast. No lease-and-filter, no wasted attempts.

Broadcast semantics are work-stealing, not fan-out: a broadcast task is
claimed by the *first* node that polls it (at-least-once per task, like
every queue job). To have every node run something, dispatch one targeted
task per node from :meth:`NodeRegistry.list_active`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue

__all__ = ["MeshTask", "MeshTasks"]

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


def _node_topic(node_id: str) -> str:
    return f"mesh:node:{node_id}"


@dataclass
class MeshTask:
    job_id: str
    task_type: str
    payload: dict[str, Any]
    target_node: str | None
    origin_node: str
    priority: int = 0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "target_node": self.target_node,
            "origin_node": self.origin_node,
            "priority": self.priority,
            "attempts": self.attempts,
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
        )


class MeshTasks:
    """Dispatch and poll for mesh tasks."""

    def __init__(self, db: Database, queue: WorkQueue | None = None) -> None:
        self.db = db
        self.queue = queue or WorkQueue(db)
        # WorkQueue itself doesn't create its table (migrations do); a mesh
        # built on a fresh database must still work.
        self.db.executescript(WORK_QUEUE_DDL)

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
    ) -> str:
        """Queue a task. ``target_node=None`` leaves the task unclaimed by
        any specific node: the first node to poll takes it (work-stealing).
        """
        if not task_type:
            raise ValueError("task_type is required")
        if not origin_node:
            raise ValueError("origin_node is required")
        payload = payload or {}
        size = len(json.dumps(payload, default=str))
        if size > MESH_MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"task payload is {size} bytes (limit "
                f"{MESH_MAX_PAYLOAD_BYTES}); put large blobs in the artifact "
                "store and reference them instead")
        topic = BROADCAST_TOPIC if target_node is None else _node_topic(target_node)
        job_id = self.queue.enqueue(
            topic,
            {
                "task_type": task_type,
                "payload": payload,
                "target_node": target_node,
                "origin_node": origin_node,
                "enqueued_at": time.time(),
            },
            priority=priority,
            delay=delay,
            max_attempts=max_attempts,
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
        })
        return job_id

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
        return tasks

    def complete(self, job_id: str, result: Any = None) -> None:
        self.queue.complete(job_id, result=result)
        _emit("mesh.task.completed", {"job_id": job_id})

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        self.queue.fail(job_id, error=error, retry=retry)
        _emit("mesh.task.failed", {"job_id": job_id, "error": error,
                                   "retry": retry})

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
        )
