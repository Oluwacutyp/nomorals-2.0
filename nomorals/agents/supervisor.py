"""Supervisor: watchdog, restart policy, and budget enforcement.

An autonomous system needs something whose only job is to notice that something
else has gone wrong. The supervisor watches agents and tasks, restarts them within
a restart budget, escalates when restarts stop helping, and enforces the global
mission budget so a runaway agent cannot spend forever.

The escalation ladder matters: restart once, restart twice, then stop and report.
A supervisor that restarts forever turns a deterministic bug into an infinite
loop with a token bill attached.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import BudgetExceeded, DeadlineExceeded, classify
from ..core.logging_setup import get_logger
from .base import Agent, AgentResult, AgentState
from .tasks import Task, TaskGraph, TaskState

__all__ = ["RestartPolicy", "Supervisor", "SupervisorEvent"]

_log = get_logger(__name__)


_BUDGET_ERRORS = frozenset({"BudgetExceeded", "DeadlineExceeded"})


@dataclass
class RestartPolicy:
    """How aggressively to retry a failed unit of work."""

    max_restarts: int = 3
    window_seconds: float = 300.0
    backoff_base: float = 0.5
    backoff_cap: float = 30.0
    escalate_after: int = 2
    restart_on_budget: bool = False

    def delay_for(self, attempt: int) -> float:
        return min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempt - 1)))


@dataclass
class SupervisorEvent:
    kind: str  # restart | escalate | give_up | budget
    subject: str
    detail: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail, "ts": self.ts}


@dataclass
class _Record:
    attempts: int = 0
    failures: int = 0
    restarts: int = 0
    window_start: float = field(default_factory=time.time)
    last_error: str = ""
    given_up: bool = False


class Supervisor:
    """Watches and restarts agents and tasks."""

    def __init__(
        self,
        *,
        policy: RestartPolicy | None = None,
        budget: Any = None,
        on_event: Callable[[SupervisorEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or RestartPolicy()
        self.budget = budget
        self.on_event = on_event
        self._clock = clock
        self._records: dict[str, _Record] = {}
        self._lock = threading.RLock()
        self.events: list[SupervisorEvent] = []
        self.stats = {"watched": 0, "restarts": 0, "escalations": 0, "give_ups": 0, "budget_stops": 0}

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def _record(self, subject: str) -> _Record:
        with self._lock:
            record = self._records.get(subject)
            if record is None:
                record = _Record(window_start=self._clock())
                self._records[subject] = record
            elif self._clock() - record.window_start > self.policy.window_seconds:
                # The failure window has rolled over; forget old failures.
                record.attempts = 0
                record.failures = 0
                record.window_start = self._clock()
            return record

    def _emit(self, event: SupervisorEvent) -> None:
        with self._lock:
            self.events.append(event)
        _log.warning("supervisor: %s %s — %s", event.kind, event.subject, event.detail)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 - callback must not kill the supervisor
                _log.debug("supervisor on_event raised: %s", exc)

    # ── budget ───────────────────────────────────────────────────────────────
    def check_budget(self) -> None:
        """Raise if the global budget is exhausted."""
        if self.budget is None:
            return
        try:
            self.budget.check()
        except (BudgetExceeded, DeadlineExceeded) as exc:
            self.stats["budget_stops"] += 1
            self._emit(SupervisorEvent("budget", "mission", exc.message))
            raise

    @property
    def budget_exhausted(self) -> bool:
        if self.budget is None:
            return False
        try:
            self.budget.check()
        except (BudgetExceeded, DeadlineExceeded):
            return True
        return False

    # ── agents ───────────────────────────────────────────────────────────────
    def run_agent(self, agent: Agent, task_input: Any = None, *, factory: Callable[[], Agent] | None = None) -> AgentResult:
        """Run an agent, restarting it within policy if it fails.

        ``factory`` builds a fresh instance for a restart, which matters because an
        agent that died holding partial state should not be reused as-is.
        """
        current = agent
        while True:
            self.check_budget()
            record = self._record(current.name)
            record.attempts += 1
            self.stats["watched"] += 1
            result = current.run(task_input)
            if result.ok:
                return result

            record.failures += 1
            record.last_error = result.error
            is_budget = result.metadata.get("error_type") in _BUDGET_ERRORS or (
                not result.metadata.get("error_type") and "budget" in result.error.lower()
            )
            if is_budget and not self.policy.restart_on_budget:
                self._emit(SupervisorEvent("give_up", current.name, f"budget stop: {result.error}"))
                self.stats["give_ups"] += 1
                return result

            if record.restarts >= self.policy.max_restarts or record.given_up:
                self._emit(SupervisorEvent("give_up", current.name, result.error))
                self.stats["give_ups"] += 1
                return result

            if record.failures >= self.policy.escalate_after:
                self.stats["escalations"] += 1
                self._emit(
                    SupervisorEvent(
                        "escalate",
                        current.name,
                        f"{record.failures} failures; restarting {record.restarts + 1}/{self.policy.max_restarts}",
                    )
                )

            record.restarts += 1
            self.stats["restarts"] += 1
            self._emit(
                SupervisorEvent(
                    "restart",
                    current.name,
                    f"attempt {record.restarts}/{self.policy.max_restarts} after: {result.error[:200]}",
                )
            )
            delay = self.policy.delay_for(record.restarts)
            if delay:
                time.sleep(delay)
            if factory is not None:
                current = factory()
            else:
                current.state = AgentState.IDLE
                current.cancel_event.clear()

    # ── tasks ────────────────────────────────────────────────────────────────
    def retry_task(self, graph: TaskGraph, task: Task, run: Callable[[Task], Any]) -> Task:
        """Re-run a failed task in place, honouring the restart policy."""
        record = self._record(f"task:{task.name}")
        while record.restarts < self.policy.max_restarts and not record.given_up:
            if self.budget_exhausted:
                self._emit(SupervisorEvent("give_up", task.name, "budget exhausted"))
                self.stats["give_ups"] += 1
                break
            record.restarts += 1
            self.stats["restarts"] += 1
            self._emit(
                SupervisorEvent("restart", task.name, f"attempt {record.restarts}: {task.error[:200]}")
            )
            time.sleep(self.policy.delay_for(record.restarts))
            task.state = TaskState.PENDING
            task.error = ""
            task.mark_running()
            try:
                task.mark_done(run(task))
                return task
            except Exception as exc:  # noqa: BLE001
                error = classify(exc)
                task.mark_failed(f"{error.code}: {error.message}")
                record.failures += 1
        record.given_up = True
        self.stats["give_ups"] += 1
        self._emit(SupervisorEvent("give_up", task.name, task.error))
        return task

    def watch_graph(
        self,
        graph: TaskGraph,
        run: Callable[[Task], Any],
        *,
        max_retries_per_task: int | None = None,
    ) -> dict[str, int]:
        """Retry every failed task in a completed graph. Returns retry counts."""
        limit = max_retries_per_task if max_retries_per_task is not None else self.policy.max_restarts
        retried = 0
        for task in list(graph.tasks.values()):
            if task.state is not TaskState.FAILED:
                continue
            record = self._record(f"task:{task.name}")
            if record.restarts >= limit:
                continue
            self.retry_task(graph, task, run)
            retried += 1
        return {"retried": retried}

    # ── reporting ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.stats,
                "subjects": {
                    name: {
                        "attempts": r.attempts,
                        "failures": r.failures,
                        "restarts": r.restarts,
                        "given_up": r.given_up,
                        "last_error": r.last_error[:200],
                    }
                    for name, r in self._records.items()
                },
                "events": [e.to_dict() for e in self.events[-20:]],
            }

    def reset(self, subject: str | None = None) -> None:
        with self._lock:
            if subject is None:
                self._records.clear()
            else:
                self._records.pop(subject, None)
