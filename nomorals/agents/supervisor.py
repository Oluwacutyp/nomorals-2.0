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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import BudgetExceeded, DeadlineExceeded, classify
from ..core.logging_setup import get_logger
from .base import Agent, AgentResult, AgentState
from ..core.tasks import Task, TaskGraph, TaskState

__all__ = ["RestartPolicy", "Supervisor", "SupervisorEvent", "TeamResult"]

_log = get_logger(__name__)


_BUDGET_ERRORS = frozenset({"BudgetExceeded", "DeadlineExceeded"})


@dataclass
class RestartPolicy:
    """How aggressively to retry a failed unit of work.

    Erlang/OTP intensity semantics: ``max_restarts`` restarts are allowed
    per ``window_seconds`` — exceed that and the supervisor gives up
    instead of hammering a permanently broken agent forever. The window
    slides: old restarts age out, so a flaky-but-recovering agent is not
    punished for yesterday's failures.
    """

    max_restarts: int = 3
    window_seconds: float = 300.0
    backoff_base: float = 0.5
    backoff_cap: float = 30.0
    escalate_after: int = 2
    restart_on_budget: bool = False

    def delay_for(self, attempt: int) -> float:
        return min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempt - 1)))

    def intensity_exceeded(self, restart_times: list[float],
                           now: float) -> bool:
        """True when restarts within the window hit the intensity limit."""
        window = max(1.0, float(self.window_seconds))
        recent = [t for t in restart_times if now - t <= window]
        return len(recent) >= max(1, int(self.max_restarts))


@dataclass
class SupervisorEvent:
    kind: str  # restart | escalate | give_up | budget
    subject: str
    detail: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail, "ts": self.ts}


@dataclass
class TeamResult:
    """Outcome of :meth:`Supervisor.run_all`: one supervised agent per entry.

    ``results`` maps agent name → the agent's final :class:`AgentResult`
    (a failed-then-given-up agent maps to its last failed result — failures
    are records, never raises).
    """

    results: dict[str, AgentResult]
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results.values())

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results.values() if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results.values() if not r.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "seconds": round(self.seconds, 3),
            "results": {name: r.to_dict() for name, r in self.results.items()},
        }


@dataclass
class _Record:
    attempts: int = 0
    failures: int = 0
    restarts: int = 0
    window_start: float = field(default_factory=time.time)
    last_error: str = ""
    given_up: bool = False
    #: Monotonic timestamps of every restart — the OTP intensity window.
    restart_times: list[float] = field(default_factory=list)


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
        # Optional shared blackboard: when attached, every supervisor event
        # is also posted there so other agents can observe and react to a
        # teammate's restarts / escalations / give-ups.
        self._board: Any = None
        self._board_topic: str = "supervisor"

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

    def attach_blackboard(self, board: Any, *, topic: str = "supervisor") -> None:
        """Post every supervisor event to a shared blackboard as well.

        Other agents can ``watch``/``link`` the topic to react — e.g. a
        coordinator that re-plans when a worker is given up on. Posting
        never breaks supervision: board failures are swallowed.
        """
        with self._lock:
            self._board = board
            self._board_topic = topic or "supervisor"

    def detach_blackboard(self) -> None:
        with self._lock:
            self._board = None

    def _emit(self, event: SupervisorEvent) -> None:
        with self._lock:
            self.events.append(event)
            board, topic = self._board, self._board_topic
        _log.warning("supervisor: %s %s — %s", event.kind, event.subject, event.detail)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 - callback must not kill the supervisor
                _log.debug("supervisor on_event raised: %s", exc)
        if board is not None:
            try:
                board.post(
                    f"supervisor.{event.kind}.{event.subject}",
                    event.to_dict(),
                    author="supervisor",
                    topic=topic,
                    metadata={"kind": event.kind, "subject": event.subject},
                )
            except Exception:  # noqa: BLE001 - telemetry never breaks supervision
                pass

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

            # OTP intensity (replaces the old lifetime cap): restarts are
            # only allowed max_restarts per window_seconds. A persistently
            # broken agent trips this within seconds (backoff delays are
            # short); a flaky-but-recovering one ages out of the window and
            # keeps its restart budget. Hammering forever is the failure
            # mode this prevents.
            now = self._clock()
            if record.given_up:
                self._emit(SupervisorEvent("give_up", current.name,
                                           "already given up"))
                self.stats["give_ups"] += 1
                return result
            if self.policy.intensity_exceeded(record.restart_times, now):
                detail = (f"restart intensity exceeded: "
                          f"{self.policy.max_restarts} restarts in "
                          f"{self.policy.window_seconds:g}s — giving up")
                self._emit(SupervisorEvent("give_up", current.name, detail))
                self.stats["give_ups"] += 1
                record.given_up = True
                return result
            record.restart_times.append(now)

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

    # ── teams ────────────────────────────────────────────────────────────────
    def run_all(
        self,
        agents: list[Agent] | dict[str, Agent],
        task_input: Any = None,
        *,
        factories: dict[str, Callable[[], Agent]] | None = None,
        max_workers: int | None = None,
        dependencies: dict[str, list[str]] | None = None,
    ) -> TeamResult:
        """Run a team of agents in parallel, each under the restart policy.

        Every agent is supervised exactly like :meth:`run_agent` (restart
        within policy, escalate, give up) — the difference is they run
        concurrently on a thread pool instead of one at a time. A dead
        agent becomes a failed :class:`AgentResult` in the map; it never
        kills its teammates. ``agents`` may be a list (names come from
        ``agent.name``) or a name → agent dict; ``factories`` optionally
        gives per-name fresh-instance factories for restarts.

        ``dependencies`` maps agent name → names it depends on. When given,
        Erlang/OTP ``rest_for_one`` semantics apply after the team run: a
        failed dependency restarts itself AND its dependents, because a
        dependent that ran on a broken dependency's output can't be
        trusted.
        """
        started = time.perf_counter()
        if isinstance(agents, dict):
            items = list(agents.items())
        else:
            items = [(a.name, a) for a in agents]
        factories = factories or {}
        by_name = dict(items)
        results: dict[str, AgentResult] = {}

        def _one(name: str, agent: Agent) -> tuple[str, AgentResult]:
            try:
                result = self.run_agent(
                    agent, task_input,
                    factory=factories.get(name),
                )
            except Exception as exc:  # noqa: BLE001 - a dead agent is a record
                _log.exception("supervised team agent %r crashed", name)
                result = AgentResult(
                    agent_id=name, role=getattr(agent, "role", "generic"),
                    ok=False, error=f"{type(exc).__name__}: {exc}")
            return name, result

        workers = max_workers or max(1, len(items))
        with ThreadPoolExecutor(max_workers=workers,
                               thread_name_prefix="sv-team") as pool:
            futures = {pool.submit(_one, name, agent): name
                       for name, agent in items}
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result

        # rest_for_one: a failed dependency poisons its dependents.
        if dependencies:
            dependents = self._dependents(dependencies)
            failed_deps = {name for name, r in results.items()
                           if not r.ok and name in dependents}
            poisoned = set(failed_deps)
            for dep in failed_deps:
                poisoned.update(dependents.get(dep, ()))
            for name in sorted(poisoned):
                agent = by_name.get(name)
                if agent is None:
                    continue
                _log.info("rest_for_one: restarting %r (dependency failed)",
                          name)
                self._emit(SupervisorEvent(
                    "restart", name,
                    "rest_for_one: dependency failed, restarting"))
                _, results[name] = _one(name, agent)

        team = TeamResult(results=results,
                          seconds=time.perf_counter() - started)
        _log.info("supervised team run: %d/%d agents ok (%.1fs)",
                  team.succeeded, len(results), team.seconds)
        return team

    @staticmethod
    def _dependents(dependencies: dict[str, list[str]]) -> dict[str, list[str]]:
        """Invert name → [deps] into dep → [dependents]."""
        out: dict[str, list[str]] = {}
        for name, deps in (dependencies or {}).items():
            for dep in deps or []:
                out.setdefault(dep, []).append(name)
        return out

    # ── presentation ─────────────────────────────────────────────────────
    def render_tree(self) -> str:
        """Human-readable supervision status: who restarted, who gave up."""
        from .render import ICONS, banner, kv, table, truncate

        try:
            lines = [banner("Supervision tree", ICONS["shield"]),
                     kv({"watched": self.stats.get("watched", 0),
                         "restarts": self.stats.get("restarts", 0),
                         "escalations": self.stats.get("escalations", 0),
                         "give_ups": self.stats.get("give_ups", 0),
                         "budget stops": self.stats.get("budget_stops", 0)}
                        .items())]
            rows = []
            for name, rec in sorted(self._records.items()):
                if rec.given_up:
                    icon, state = ICONS["fail"], "given up"
                elif rec.restarts:
                    icon, state = ICONS["retry"], f"{rec.restarts} restart(s)"
                elif rec.failures:
                    icon, state = ICONS["warn"], f"{rec.failures} failure(s)"
                else:
                    icon, state = ICONS["ok"], "healthy"
                rows.append([f"{icon} {name}", state,
                             truncate(rec.last_error, 60)])
            if rows:
                lines.append("")
                lines.append(table(["agent", "state", "last error"], rows))
            return "\n".join(lines)
        except Exception:  # noqa: BLE001 — rendering never breaks callers
            return "supervision tree (render failed)"

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
