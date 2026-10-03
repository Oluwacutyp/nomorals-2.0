"""Power-aware scheduler: heavy work waits for good power.

Wraps :class:`nomorals.storage.queue.WorkQueue` with power-class topics:
``power:light`` always flows; ``power:heavy`` is only leased when
:meth:`PowerMonitor.status` says conditions are good. Heavy tasks queued
during a brownout wait durably — nothing is dropped, just deferred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue
from .monitor import PowerMonitor

__all__ = ["PowerTask", "PowerAwareScheduler"]

_log = get_logger(__name__)

LIGHT_TOPIC = "power:light"
HEAVY_TOPIC = "power:heavy"


@dataclass
class PowerTask:
    job_id: str
    task_type: str
    payload: dict[str, Any]
    power_class: str  # "light" | "heavy"
    priority: int = 0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "power_class": self.power_class,
            "priority": self.priority,
            "attempts": self.attempts,
        }


class PowerAwareScheduler:
    """Defer heavy work when the battery is low or the device is hot."""

    def __init__(
        self,
        db: Database,
        monitor: PowerMonitor | None = None,
        queue: WorkQueue | None = None,
    ) -> None:
        self.db = db
        # Default monitor has no sampler (fails open). L7 callers inject
        # a ResourceManager-backed sampler for real readings.
        self.monitor = monitor or PowerMonitor()
        self.queue = queue or WorkQueue(db)

    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        power_class: str = "light",
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
    ) -> str:
        """Queue a task. ``power_class`` is ``"light"`` or ``"heavy"``."""
        if power_class not in ("light", "heavy"):
            raise ValueError(f"power_class must be light|heavy, got {power_class!r}")
        if not task_type:
            raise ValueError("task_type is required")
        topic = HEAVY_TOPIC if power_class == "heavy" else LIGHT_TOPIC
        job_id = self.queue.enqueue(
            topic,
            {
                "task_type": task_type,
                "payload": payload or {},
                "power_class": power_class,
                "enqueued_at": time.time(),
            },
            priority=priority,
            delay=delay,
            max_attempts=max_attempts,
        )
        _log.info("power task %s queued: %s (%s)", job_id, task_type, power_class)
        return job_id

    def poll(
        self,
        worker: str = "worker",
        *,
        batch: int = 5,
        lease_seconds: float | None = None,
        force_heavy: bool = False,
    ) -> list[PowerTask]:
        """Claim tasks. Heavy tasks flow only when power allows.

        Set ``force_heavy=True`` to override (e.g., plugged in and the
        operator explicitly wants the work now).
        """
        if not worker:
            raise ValueError("worker is required")
        tasks: list[PowerTask] = []
        topics = [LIGHT_TOPIC]
        if force_heavy or not self.monitor.status().should_defer_heavy:
            topics.append(HEAVY_TOPIC)
        else:
            _log.info("power constrained: deferring heavy tasks")
        for topic in topics:
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

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        self.queue.fail(job_id, error=error, retry=retry)

    def deferred_heavy_count(self) -> int:
        rows = self.db.query(
            f"SELECT COUNT(*) AS n FROM {self.queue.TABLE} "
            "WHERE topic=? AND status='ready'",
            (HEAVY_TOPIC,),
        )
        return int(rows[0]["n"]) if rows else 0

    @staticmethod
    def _to_task(job: Job) -> PowerTask:
        p = job.payload or {}
        return PowerTask(
            job_id=job.id,
            task_type=str(p.get("task_type") or ""),
            payload=dict(p.get("payload") or {}),
            power_class=str(p.get("power_class") or "light"),
            priority=job.priority,
            attempts=job.attempts,
        )
