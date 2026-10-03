"""Mesh task dispatch over the durable work queue.

Wraps :class:`nomorals.storage.queue.WorkQueue`. Targeting uses topic
namespaces: ``mesh:broadcast`` for all nodes, ``mesh:node:<id>`` for one
node. A node polls both. No lease-and-filter, no wasted attempts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue

__all__ = ["MeshTask", "MeshTasks"]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break mesh task handling (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

BROADCAST_TOPIC = "mesh:broadcast"


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


class MeshTasks:
    """Dispatch and poll for mesh tasks."""

    def __init__(self, db: Database, queue: WorkQueue | None = None) -> None:
        self.db = db
        self.queue = queue or WorkQueue(db)

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
        """Queue a task. ``target_node=None`` broadcasts to all nodes."""
        if not task_type:
            raise ValueError("task_type is required")
        if not origin_node:
            raise ValueError("origin_node is required")
        topic = BROADCAST_TOPIC if target_node is None else _node_topic(target_node)
        job_id = self.queue.enqueue(
            topic,
            {
                "task_type": task_type,
                "payload": payload or {},
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
