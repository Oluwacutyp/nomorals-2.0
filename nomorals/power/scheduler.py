"""Power-aware scheduler: heavy work waits for good power.

Wraps :class:`nomorals.storage.queue.WorkQueue` with power-class topics:
``power:light`` always flows; ``power:medium`` flows while degradation is
mild (level <= 1); ``power:heavy`` only flows at degradation level 0 with
a clean advisory.  Deferred tasks wait durably — nothing is dropped.

On top of power classes, every task carries a *tier*
(``critical`` | ``important`` | ``background`` | ``bulk``) and an optional
*subsystem* (``llm``, ``media``, ``scheduler``, ``missions``, ``bulk``).
The degradation ladder sheds tiers bottom-up, and — when a
resource-manager-like ``resources`` object is injected — subsystem
budgets gate admission: a task whose subsystem budget is exhausted is
requeued with backoff instead of being started.

This module never imports ``nomorals.os`` (layer rule): the optional
``resources`` object is duck-typed (needs ``.budgets`` with
``acquire``/``release``/``admission``/``snapshot``), and L7 entry points
wire in the real ``ResourceManager``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.queue import Job, WorkQueue
from .monitor import PowerMonitor

__all__ = [
    "PowerTask",
    "PowerAwareScheduler",
    "LIGHT_TOPIC",
    "MEDIUM_TOPIC",
    "HEAVY_TOPIC",
    "POWER_CLASSES",
    "TASK_TIERS",
]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break power-aware scheduling (fail-open telemetry,
    fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

LIGHT_TOPIC = "power:light"
MEDIUM_TOPIC = "power:medium"
HEAVY_TOPIC = "power:heavy"
POWER_CLASSES = ("light", "medium", "heavy")
TASK_TIERS = ("critical", "important", "background", "bulk")


@dataclass
class PowerTask:
    job_id: str
    task_type: str
    payload: dict[str, Any]
    power_class: str  # "light" | "medium" | "heavy"
    priority: int = 0
    attempts: int = 0
    subsystem: str | None = None   # budget owner: llm|media|scheduler|...
    tier: str = "background"       # critical|important|background|bulk
    mem_mb: float = 0.0            # expected memory need (budget admission)
    cpu_pct: float = 0.0           # expected cpu need (budget admission)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": dict(self.payload),
            "power_class": self.power_class,
            "priority": self.priority,
            "attempts": self.attempts,
            "subsystem": self.subsystem,
            "tier": self.tier,
            "mem_mb": self.mem_mb,
            "cpu_pct": self.cpu_pct,
        }


class PowerAwareScheduler:
    """Defer heavy work when power, degradation, or budgets say so."""

    def __init__(
        self,
        db: Database,
        monitor: PowerMonitor | None = None,
        queue: WorkQueue | None = None,
        resources: Any | None = None,
        *,
        deferral_backoff_s: float = 30.0,
        budget_backoff_s: float = 60.0,
    ) -> None:
        self.db = db
        # Default monitor has no sampler (fails open). L7 callers inject
        # a ResourceManager-backed sampler for real readings.
        if monitor is None and resources is not None:
            # Build a real monitor from the injected resource manager when
            # it quacks like one (duck-typed: .sample() + .consult()).
            if callable(getattr(resources, "sample", None)) and callable(
                    getattr(resources, "consult", None)):
                monitor = PowerMonitor(sampler=lambda: resources)
        self.monitor = monitor or PowerMonitor()
        self.queue = queue or WorkQueue(db)
        # Duck-typed resource manager for subsystem budget admission.
        # None = budgets unavailable -> fail-open (tasks flow).
        self.resources = resources
        self.deferral_backoff_s = max(0.0, float(deferral_backoff_s))
        self.budget_backoff_s = max(0.0, float(budget_backoff_s))

    # -- budget helpers --------------------------------------------------
    def _budgets(self) -> Any | None:
        try:
            b = getattr(self.resources, "budgets", None)
            return b
        except Exception:  # noqa: BLE001 - never raises
            return None

    def try_acquire(self, subsystem: str, *, mem_mb: float = 0.0,
                    cpu_pct: float = 0.0) -> bool:
        """Admit a subsystem load against its budget.  Fail-open when no
        resource manager is wired (returns True).  Never raises."""
        try:
            b = self._budgets()
            if b is None:
                return True
            return bool(b.acquire(subsystem, mem_mb=mem_mb, cpu_pct=cpu_pct))
        except Exception:  # noqa: BLE001 - never raises
            return True

    def release_budget(self, subsystem: str, *, mem_mb: float = 0.0,
                       cpu_pct: float = 0.0) -> None:
        """Release tracked budget usage.  Never raises."""
        try:
            b = self._budgets()
            if b is not None:
                b.release(subsystem, mem_mb=mem_mb, cpu_pct=cpu_pct)
        except Exception:  # noqa: BLE001 - never raises
            pass

    def budget_snapshot(self) -> dict[str, Any]:
        try:
            b = self._budgets()
            return b.snapshot() if b is not None else {}
        except Exception:  # noqa: BLE001 - never raises
            return {}

    # -- dispatch ---------------------------------------------------------
    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        power_class: str = "light",
        priority: int = 0,
        delay: float = 0.0,
        max_attempts: int = 5,
        subsystem: str | None = None,
        tier: str = "background",
        mem_mb: float = 0.0,
        cpu_pct: float = 0.0,
    ) -> str:
        """Queue a task.

        ``power_class`` is ``"light"`` | ``"medium"`` | ``"heavy"``;
        ``tier`` is ``"critical"`` | ``"important"`` | ``"background"`` |
        ``"bulk"``.  ``subsystem``/``mem_mb``/``cpu_pct`` feed budget
        admission at poll time (dispatch itself never blocks).
        """
        if power_class not in POWER_CLASSES:
            raise ValueError(
                f"power_class must be one of {POWER_CLASSES}, got {power_class!r}")
        if not task_type:
            raise ValueError("task_type is required")
        tier = str(tier).lower()
        if tier not in TASK_TIERS:
            raise ValueError(
                f"tier must be one of {TASK_TIERS}, got {tier!r}")
        topic = {LIGHT_TOPIC: LIGHT_TOPIC, "light": LIGHT_TOPIC,
                 "medium": MEDIUM_TOPIC, "heavy": HEAVY_TOPIC}[power_class]
        job_id = self.queue.enqueue(
            topic,
            {
                "task_type": task_type,
                "payload": payload or {},
                "power_class": power_class,
                "subsystem": subsystem,
                "tier": tier,
                "mem_mb": max(0.0, float(mem_mb)),
                "cpu_pct": max(0.0, float(cpu_pct)),
                "enqueued_at": time.time(),
            },
            priority=priority,
            delay=delay,
            max_attempts=max_attempts,
        )
        _log.info("power task %s queued: %s (%s/%s)", job_id, task_type,
                  power_class, tier)
        _emit("power.task.dispatched", {
            "job_id": job_id,
            "task_type": task_type,
            "power_class": power_class,
            "subsystem": subsystem,
            "tier": tier,
        })
        return job_id

    # -- poll --------------------------------------------------------------
    def poll(
        self,
        worker: str = "worker",
        *,
        batch: int = 5,
        lease_seconds: float | None = None,
        force_heavy: bool = False,
    ) -> list[PowerTask]:
        """Claim tasks.  Heavy tasks flow only when power allows; medium
        tasks flow while degradation is mild; tasks whose tier is shed or
        whose subsystem budget is exhausted are requeued with backoff.

        Set ``force_heavy=True`` to override (e.g., plugged in and the
        operator explicitly wants the work now).
        """
        if not worker:
            raise ValueError("worker is required")
        status = self.monitor.status()
        plan = [LIGHT_TOPIC]
        if status.ok and status.degradation_level <= 1:
            plan.append(MEDIUM_TOPIC)
        elif status.should_defer_medium:
            _log.info("degradation level %d: deferring medium tasks",
                      status.degradation_level)
        if force_heavy or (status.degradation_level == 0
                           and not status.should_defer_heavy):
            plan.append(HEAVY_TOPIC)
        else:
            _log.info("power constrained: deferring heavy tasks")

        tasks: list[PowerTask] = []
        for topic in plan:
            if len(tasks) >= batch:
                break
            jobs = self.queue.lease(
                topic, worker=worker, lease_seconds=lease_seconds,
                batch=batch - len(tasks),
            )
            for job in jobs:
                task = self._to_task(job)
                if not status.tier_allowed(task.tier):
                    self.queue.requeue(job.id, delay=self.deferral_backoff_s)
                    _log.info("tier %r shed at degradation level %d: "
                              "deferring task %s", task.tier,
                              status.degradation_level, job.id)
                    _emit("power.task.deferred",
                          {"job_id": job.id, "reason": "tier_shed",
                           "tier": task.tier,
                           "degradation_level": status.degradation_level})
                    continue
                if task.subsystem and not self.try_acquire(
                        task.subsystem, mem_mb=task.mem_mb,
                        cpu_pct=task.cpu_pct):
                    self.queue.requeue(job.id, delay=self.budget_backoff_s)
                    _log.info("budget %r exhausted: deferring task %s",
                              task.subsystem, job.id)
                    _emit("power.task.deferred",
                          {"job_id": job.id, "reason": "budget_exhausted",
                           "subsystem": task.subsystem})
                    continue
                tasks.append(task)
                if len(tasks) >= batch:
                    break
        return tasks

    # -- completion ---------------------------------------------------------
    def _release_for_job(self, job_id: str) -> None:
        """Release budget usage tracked for a finished job, if any."""
        try:
            job = self.queue.get(job_id)
            if job is None:
                return
            p = job.payload or {}
            sub = p.get("subsystem")
            if sub:
                self.release_budget(
                    str(sub),
                    mem_mb=float(p.get("mem_mb") or 0.0),
                    cpu_pct=float(p.get("cpu_pct") or 0.0))
        except Exception:  # noqa: BLE001 - never raises
            pass

    def complete(self, job_id: str, result: Any = None) -> None:
        self._release_for_job(job_id)
        self.queue.complete(job_id, result=result)
        _emit("power.task.completed", {"job_id": job_id})

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        self._release_for_job(job_id)
        self.queue.fail(job_id, error=error, retry=retry)
        _emit("power.task.failed", {"job_id": job_id, "error": error,
                                    "retry": retry})

    # -- inspection ----------------------------------------------------------
    def deferred_heavy_count(self) -> int:
        return self._deferred_count(HEAVY_TOPIC)

    def deferred_medium_count(self) -> int:
        return self._deferred_count(MEDIUM_TOPIC)

    def _deferred_count(self, topic: str) -> int:
        try:
            rows = self.db.query(
                f"SELECT COUNT(*) AS n FROM {self.queue.TABLE} "
                "WHERE topic=? AND status='ready'",
                (topic,),
            )
            return int(rows[0]["n"]) if rows else 0
        except Exception:  # noqa: BLE001 - never raises
            return 0

    def stats(self) -> dict[str, Any]:
        """Pending/deferred counts per class plus budget snapshot."""
        try:
            pending: dict[str, int] = {}
            for cls, topic in (("light", LIGHT_TOPIC),
                               ("medium", MEDIUM_TOPIC),
                               ("heavy", HEAVY_TOPIC)):
                try:
                    pending[cls] = int(self.queue.pending(topic))
                except Exception:  # noqa: BLE001 - per-topic best-effort
                    pending[cls] = 0
            return {
                "pending": pending,
                "deferred_heavy": self.deferred_heavy_count(),
                "deferred_medium": self.deferred_medium_count(),
                "budgets": self.budget_snapshot(),
            }
        except Exception:  # noqa: BLE001 - never raises
            return {"pending": {}, "deferred_heavy": 0,
                    "deferred_medium": 0, "budgets": {}}

    @staticmethod
    def _to_task(job: Job) -> PowerTask:
        p = job.payload or {}
        tier = str(p.get("tier") or "background").lower()
        if tier not in TASK_TIERS:
            tier = "background"
        pc = str(p.get("power_class") or "light")
        if pc not in POWER_CLASSES:
            pc = "light"
        sub = p.get("subsystem")
        try:
            mem_mb = max(0.0, float(p.get("mem_mb") or 0.0))
        except (TypeError, ValueError):
            mem_mb = 0.0
        try:
            cpu_pct = max(0.0, float(p.get("cpu_pct") or 0.0))
        except (TypeError, ValueError):
            cpu_pct = 0.0
        return PowerTask(
            job_id=job.id,
            task_type=str(p.get("task_type") or ""),
            payload=dict(p.get("payload") or {}),
            power_class=pc,
            priority=job.priority,
            attempts=job.attempts,
            subsystem=str(sub) if sub else None,
            tier=tier,
            mem_mb=mem_mb,
            cpu_pct=cpu_pct,
        )
