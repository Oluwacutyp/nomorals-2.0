"""Task graph: the unit of parallel work.

A task declares *what kind of work* it is, and the runtime uses that to choose a
substrate:

* ``io``    → thread pool (HTTP, disk, subprocess)
* ``cpu``   → process pool (tokenizing, embedding, dataset dedup, training)
* ``async`` → event loop (high fan-out concurrent I/O)

Dependencies form a DAG. The scheduler runs everything whose dependencies have
completed, in parallel, and propagates cancellation downward — so a mission that
is cancelled does not leave 40 orphaned subtasks burning tokens.

Retries follow the scheduler discipline: timeout → retry with exponential
backoff + full jitter (idempotent ops only) → bounded attempts AND a bounded
retry budget for the whole graph → poison tasks become diagnosable
dead letters instead of busy loops. Instant retries against a dying external
dependency are the #1 homegrown scheduler bug; :class:`RetryPolicy` makes the
delay explicit and the executor's retry path should call
:meth:`TaskGraph.retry_later` with it rather than requeueing immediately.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import new_id

__all__ = [
    "AcceptanceCriterion",
    "AcceptanceResult",
    "RetryBudget",
    "RetryPolicy",
    "Task",
    "TaskGraph",
    "TaskKind",
    "TaskResult",
    "TaskState",
    "cycle_in",
]


@dataclass
class RetryPolicy:
    """How a task retries, in one place the executor can read.

    * ``max_retries`` — bounded attempts. Validation errors are *permanent*
      and must never be retried; use ``retryable`` to name the transient
      exception classes.
    * ``backoff`` — ``base * factor**attempt``, capped at ``max_delay``,
      plus full uniform jitter (fixed multiples build thundering herds).
    * ``retry_after`` — when an exception carries a ``retry_after`` attribute
      (HTTP 429), it wins over the computed backoff.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    factor: float = 2.0
    max_delay: float = 300.0
    jitter: bool = True
    retryable: tuple[type[BaseException], ...] = (Exception,)
    rng: random.Random | None = None

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Attempt is the number of the attempt that just failed (1-based)."""
        if attempt > self.max_retries:
            return False
        return isinstance(exc, self.retryable)

    def delay_for(self, attempt: int, exc: BaseException | None = None) -> float:
        """Seconds to wait before attempt ``attempt + 1``."""
        if exc is not None:
            hinted = getattr(exc, "retry_after", None)
            if hinted:
                try:
                    return min(max(0.0, float(hinted)), self.max_delay)
                except (TypeError, ValueError):
                    pass
        delay = self.base_delay * (self.factor ** max(0, attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            rng = self.rng or random
            delay = rng.uniform(0, delay)
        return delay


@dataclass
class RetryBudget:
    """A bounded retry *spend* for a whole graph run.

    Bounded attempts per task is not enough: 500 tasks each retrying 3 times
    against a dead dependency is 1500 doomed calls. The budget caps total
    retries; when exhausted, tasks fail fast instead of retrying.
    """

    max_retries: int = 100
    spent: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def spend(self) -> bool:
        """Consume one retry. Returns False when the budget is exhausted."""
        with self._lock:
            if self.spent >= self.max_retries:
                return False
            self.spent += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_retries - self.spent)

    def reset(self) -> None:
        with self._lock:
            self.spent = 0


@dataclass
class AcceptanceCriterion:
    """One check that must hold for a task to count as *correct*, not merely
    finished.  ``spec`` is serializable; built-in kinds:

    * ``{"metric": name, "gte": x}`` (also ``lte`` / ``eq``)
    * ``{"artifact": true}`` — the result references at least one artifact
    * ``{"assertion": name}`` — a named assertion in the result passed
    * ``{"test_suite": name}`` — ``result.tests[name]`` reports success
    * ``{"manual": true}`` — passes only with ``evidence["approved_by"]``
    """

    name: str
    description: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "spec": dict(self.spec), "required": self.required}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceptanceCriterion:
        return cls(name=data["name"], description=data.get("description", ""),
                   spec=dict(data.get("spec") or {}),
                   required=bool(data.get("required", True)))


@dataclass
class AcceptanceResult:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail, "required": self.required}


@dataclass
class TaskResult:
    """What a task produced — evidence, not just a status flag.

    Completion (``Task.state == DONE``) means the work *ran*; ``passed``
    means the acceptance criteria *held*.  Autonomous agents must check the
    second before claiming the goal.
    """

    status: str = "done"
    artifacts: list[str] = field(default_factory=list)  # artifact:// URIs
    evidence: dict[str, Any] = field(default_factory=dict)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    tests: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    acceptance_results: list[AcceptanceResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        required = [r for r in self.acceptance_results if r.required]
        return bool(required) and all(r.passed for r in required)

    def verify(self, criteria: Sequence[AcceptanceCriterion]) -> list[AcceptanceResult]:
        """Evaluate ``criteria`` against this result (built-in spec kinds)."""
        results: list[AcceptanceResult] = []
        for crit in criteria:
            passed, detail = _evaluate_criterion(crit, self)
            results.append(AcceptanceResult(name=crit.name, passed=passed,
                                           detail=detail, required=crit.required))
        self.acceptance_results = results
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifacts": list(self.artifacts),
            "evidence": dict(self.evidence),
            "assertions": [dict(a) for a in self.assertions],
            "tests": dict(self.tests),
            "metrics": dict(self.metrics),
            "acceptance_results": [r.to_dict() for r in self.acceptance_results],
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(
            status=data.get("status", "done"),
            artifacts=list(data.get("artifacts") or []),
            evidence=dict(data.get("evidence") or {}),
            assertions=[dict(a) for a in data.get("assertions") or []],
            tests=dict(data.get("tests") or {}),
            metrics=dict(data.get("metrics") or {}),
            acceptance_results=[
                AcceptanceResult(name=r["name"], passed=bool(r["passed"]),
                                 detail=r.get("detail", ""),
                                 required=bool(r.get("required", True)))
                for r in data.get("acceptance_results") or []
            ],
        )


def _evaluate_criterion(crit: AcceptanceCriterion,
                        result: TaskResult) -> tuple[bool, str]:
    spec = crit.spec or {}
    try:
        if "metric" in spec:
            name = spec["metric"]
            value = result.metrics.get(name)
            if value is None:
                return False, f"metric {name!r} missing"
            for op, target in (("gte", None), ("lte", None), ("eq", None)):
                if op in spec:
                    target = spec[op]
                    ok = (value >= target if op == "gte"
                          else value <= target if op == "lte"
                          else value == target)
                    return bool(ok), f"metric {name}={value} {op} {target}"
            return False, f"metric {name!r}: no comparison in spec"
        if spec.get("artifact"):
            ok = bool(result.artifacts)
            return ok, f"{len(result.artifacts)} artifact(s) referenced"
        if "assertion" in spec:
            name = spec["assertion"]
            for a in result.assertions:
                if a.get("name") == name:
                    passed = bool(a.get("passed"))
                    return passed, a.get("detail", "")
            return False, f"assertion {name!r} not recorded"
        if "test_suite" in spec:
            name = spec["test_suite"]
            suite = result.tests.get(name)
            if suite is None:
                return False, f"test suite {name!r} not recorded"
            if isinstance(suite, dict):
                passed = bool(suite.get("passed", suite.get("ok", False)))
                detail = str(suite.get("detail", suite.get("summary", "")))
            else:
                passed, detail = bool(suite), ""
            return passed, detail
        if spec.get("manual"):
            approver = result.evidence.get("approved_by", "")
            return bool(approver), f"approved_by={approver!r}"
        return False, f"unknown spec kind: {sorted(spec)}"
    except Exception as exc:  # noqa: BLE001 - a broken check fails closed
        return False, f"evaluation error: {type(exc).__name__}: {exc}"


class TaskState(str, Enum):  # noqa: UP042 - StrEnum needs 3.11+; CI tests 3.10
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskKind(str, Enum):  # noqa: UP042 - StrEnum needs 3.11+; CI tests 3.10
    IO = "io"
    CPU = "cpu"
    ASYNC = "async"


#: Terminal states: no further transitions out of these.
TERMINAL = frozenset(
    {TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED, TaskState.SKIPPED}
)


@dataclass
class Task:
    """One node in the task graph."""

    name: str
    fn: Callable[..., Any] | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: TaskKind = TaskKind.IO
    id: str = field(default_factory=new_id)
    deps: list[str] = field(default_factory=list)
    priority: int = 0
    timeout: float | None = None
    retries: int = 0
    role: str = ""
    mission_id: str = ""
    parent_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    state: TaskState = TaskState.PENDING
    result: Any = None
    error: str = ""
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Acceptance criteria: what must hold for this task to count as
    #: *correct*.  Empty means "ran to completion" is the only bar — and
    #: :attr:`verified` says so honestly.
    acceptance: list[AcceptanceCriterion] = field(default_factory=list)
    #: Retry discipline for this task. When set, the executor should use
    #: :meth:`TaskGraph.retry_later` with :meth:`RetryPolicy.delay_for`
    #: instead of requeueing immediately. ``retries`` (the legacy plain
    #: count) is the fallback when no policy is set.
    retry_policy: RetryPolicy | None = None
    #: Idempotency key: at-least-once schedulers must not run two tasks with
    #: the same key twice. :meth:`TaskGraph.add` rejects duplicates.
    idempotency_key: str = ""
    #: Monotonic timestamp before which the task is not eligible for
    #: :meth:`TaskGraph.ready` — the backoff window between retries.
    not_before: float = 0.0

    @property
    def duration(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def ok(self) -> bool:
        return self.state is TaskState.DONE

    @property
    def max_attempts(self) -> int:
        """Total attempts allowed: 1 initial + retries."""
        if self.retry_policy is not None:
            return 1 + self.retry_policy.max_retries
        return 1 + max(0, self.retries)

    def mark_running(self) -> None:
        self.state = TaskState.RUNNING
        self.started_at = time.time()
        self.attempts += 1

    def mark_done(self, result: Any) -> None:
        self.state = TaskState.DONE
        self.result = result
        self.finished_at = time.time()
        # Completion is not correctness: when the result carries evidence,
        # evaluate the acceptance criteria immediately so `verified` is
        # meaningful without a second pass.
        if isinstance(result, TaskResult) and self.acceptance:
            result.verify(self.acceptance)

    @property
    def verified(self) -> bool:
        """True when the task is both finished AND its required acceptance
        criteria held.  With no criteria defined, falls back to state —
        completion is the only bar, and the empty criteria list says so."""
        if self.state is not TaskState.DONE:
            return False
        if not self.acceptance:
            return True
        result = self.result
        if not isinstance(result, TaskResult):
            return False
        if not result.acceptance_results:
            result.verify(self.acceptance)
        return result.passed

    def verify(self) -> list[AcceptanceResult]:
        """(Re-)evaluate this task's acceptance criteria against its result."""
        result = self.result
        if not isinstance(result, TaskResult):
            result = TaskResult(status="done" if self.state is TaskState.DONE else "unknown")
            self.result = result
        return result.verify(self.acceptance)

    def mark_failed(self, error: str) -> None:
        self.state = TaskState.FAILED
        self.error = error
        self.finished_at = time.time()

    def mark_cancelled(self, reason: str = "cancelled") -> None:
        self.state = TaskState.CANCELLED
        self.error = reason
        self.finished_at = time.time()

    def mark_skipped(self, reason: str = "skipped") -> None:
        self.state = TaskState.SKIPPED
        self.error = reason
        self.finished_at = time.time()

    def to_dict(self, *, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "state": self.state.value,
            "deps": list(self.deps),
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "duration": round(self.duration, 4),
            "error": self.error,
            "role": self.role,
            "idempotency_key": self.idempotency_key,
            "acceptance": [c.to_dict() for c in self.acceptance],
            "verified": self.verified if self.state is TaskState.DONE else False,
        }
        if include_result:
            result = self.result
            payload["result"] = (result.to_dict() if isinstance(result, TaskResult)
                                 else result)
        return payload


def cycle_in(deps: dict[str, Sequence[str]]) -> list[str]:
    """Return one dependency cycle as a list of task names, or ``[]`` if acyclic.

    Kahn's algorithm: whatever is left after peeling every zero-in-degree node is
    part of a cycle.
    """
    remaining = {name: set(d) for name, d in deps.items()}
    resolved: set[str] = set()
    progress = True
    while progress:
        progress = False
        for name in list(remaining):
            if remaining[name] <= resolved:
                resolved.add(name)
                del remaining[name]
                progress = True
    if not remaining:
        return []
    # Walk the leftover subgraph to produce a readable cycle path.
    start = next(iter(remaining))
    path = [start]
    current = start
    for _ in range(len(remaining)):
        nxt = next((d for d in remaining[current] if d in remaining), None)
        if nxt is None:
            break
        path.append(nxt)
        if nxt == start:
            break
        current = nxt
    return path


class TaskGraph:
    """A DAG of tasks with a readiness query the scheduler polls.

    Thread-safe: producers may add tasks while the scheduler is draining.

    State transitions that matter to operators (failures, retries, skips,
    cancellations) are announced to ``on_state_change`` hooks registered with
    :meth:`add_hook` — wire them to metrics/logging there, not inside the
    graph.
    """

    def __init__(
        self,
        *,
        name: str = "graph",
        mission_id: str = "",
        retry_budget: RetryBudget | None = None,
    ) -> None:
        self.name = name
        self.mission_id = mission_id
        self.tasks: dict[str, Task] = {}
        self.retry_budget = retry_budget or RetryBudget()
        self._by_name: dict[str, str] = {}
        self._by_key: dict[str, str] = {}
        self._dependents: dict[str, set[str]] = {}
        self._hooks: list[Callable[[Task, TaskState, TaskState], None]] = []
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self.created_at = time.time()

    # ── hooks ────────────────────────────────────────────────────────────────
    def add_hook(self, fn: Callable[[Task, TaskState, TaskState], None]) -> None:
        """Register ``fn(task, old_state, new_state)``. Hooks never break the
        graph: a raising hook is logged, not propagated."""
        with self._lock:
            self._hooks.append(fn)

    def notify(self, task: Task, old: TaskState, new: TaskState) -> None:
        """Announce a state transition the graph didn't initiate itself
        (the executor marks tasks; it should call this afterwards)."""
        with self._lock:
            hooks = list(self._hooks)
        for hook in hooks:
            try:
                hook(task, old, new)
            except Exception:  # noqa: BLE001 - hooks must never break the graph
                import logging

                logging.getLogger("nomorals.tasks").debug(
                    "state hook failed for task %s", task.name, exc_info=True
                )

    def _transition(self, task: Task, new: TaskState, note: str = "") -> None:
        old = task.state
        if new is TaskState.DONE:
            task.mark_done(task.result)
        elif new is TaskState.FAILED:
            task.mark_failed(note or task.error)
        elif new is TaskState.CANCELLED:
            task.mark_cancelled(note)
        elif new is TaskState.SKIPPED:
            task.mark_skipped(note)
        elif new is TaskState.RUNNING:
            task.mark_running()
        else:
            task.state = new
        self.notify(task, old, task.state)

    # ── construction ─────────────────────────────────────────────────────────
    def add(self, task: Task, *, depends_on: Sequence[str | Task] = ()) -> Task:
        """Register a task. ``depends_on`` accepts ids, names, or Task objects."""
        with self._lock:
            if task.name in self._by_name:
                raise ValidationError(f"duplicate task name {task.name!r}")
            if task.id in self.tasks:
                raise ValidationError(f"duplicate task id {task.id!r}")
            if task.idempotency_key:
                owner = self._by_key.get(task.idempotency_key)
                if owner is not None and owner != task.id:
                    raise ValidationError(
                        f"duplicate idempotency key {task.idempotency_key!r} "
                        f"(task {task.name!r} vs {self.tasks[owner].name!r})"
                    )
            for dep in depends_on:
                key = dep.id if isinstance(dep, Task) else str(dep)
                resolved = self.tasks.get(key) or self.tasks.get(self._by_name.get(str(dep), ""))
                if resolved is None:
                    raise ValidationError(
                        f"task {task.name!r} depends on unknown task {dep!r}"
                    )
                if resolved.id not in task.deps:
                    task.deps.append(resolved.id)
            self.tasks[task.id] = task
            self._by_name[task.name] = task.id
            if task.idempotency_key:
                self._by_key[task.idempotency_key] = task.id
            self._dependents.setdefault(task.id, set())
            for dep_id in task.deps:
                self._dependents.setdefault(dep_id, set()).add(task.id)
            self._assert_acyclic()
            return task

    def add_task(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        depends_on: Sequence[str | Task] = (),
        **kwargs: Any,
    ) -> Task:
        """Convenience: build a Task inline."""
        task_kwargs = {
            k: kwargs.pop(k)
            for k in ("kind", "priority", "timeout", "retries", "role", "payload",
                      "mission_id", "retry_policy", "idempotency_key")
            if k in kwargs
        }
        task = Task(name=name, fn=fn, args=args, kwargs=kwargs, **task_kwargs)
        return self.add(task, depends_on=depends_on)

    def _assert_acyclic(self) -> None:
        deps = {t.id: list(t.deps) for t in self.tasks.values()}
        cycle = cycle_in(deps)
        if cycle:
            names = [self.tasks[c].name if c in self.tasks else c for c in cycle]
            raise ValidationError(f"dependency cycle: {' -> '.join(names)}")

    # ── queries ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> Task | None:
        with self._lock:
            return self.tasks.get(key) or self.tasks.get(self._by_name.get(key, ""))

    def require(self, key: str) -> Task:
        task = self.get(key)
        if task is None:
            raise ValidationError(f"unknown task {key!r}")
        return task

    def ready(self) -> list[Task]:
        """Tasks whose dependencies are all satisfied and whose backoff
        window (``not_before``) has passed, in priority order."""
        now = time.monotonic()
        with self._lock:
            out: list[Task] = []
            for task in self.tasks.values():
                if task.state is not TaskState.PENDING:
                    continue
                if task.not_before > now:
                    continue
                if all(
                    self.tasks[d].state is TaskState.DONE
                    for d in task.deps
                    if d in self.tasks
                ):
                    out.append(task)
            out.sort(key=lambda t: (-t.priority, t.created_at))
            return out

    def blocked_by_failure(self) -> list[Task]:
        """Tasks that can never run because an ancestor failed or was cancelled."""
        with self._lock:
            doomed: set[str] = set()
            for task in self.tasks.values():
                if task.state in {TaskState.FAILED, TaskState.CANCELLED, TaskState.SKIPPED}:
                    doomed.add(task.id)
            changed = True
            while changed:
                changed = False
                for task in self.tasks.values():
                    if task.id in doomed or task.state in TERMINAL:
                        continue
                    if any(d in doomed for d in task.deps):
                        doomed.add(task.id)
                        changed = True
            return [self.tasks[i] for i in doomed if self.tasks[i].state is TaskState.PENDING]

    def cascade_skip(self) -> int:
        """Mark every task blocked by a failed/cancelled ancestor as SKIPPED,
        naming the failed ancestor. Failure must be *visible*, never silently
        dropped. Returns how many were skipped."""
        count = 0
        with self._lock:
            doomed = self.blocked_by_failure()
            failed_names = {
                t.name for t in self.tasks.values() if t.state is TaskState.FAILED
            }
        for task in doomed:
            culprits = sorted(
                failed_names & {self.tasks[d].name for d in task.deps if d in self.tasks}
            )
            reason = (
                f"skipped: dependency failed ({', '.join(culprits)})"
                if culprits
                else "skipped: ancestor failed or cancelled"
            )
            self._transition(task, TaskState.SKIPPED, reason)
            count += 1
        return count

    def is_complete(self) -> bool:
        with self._lock:
            return all(t.is_terminal for t in self.tasks.values())

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self.tasks.values() if not t.is_terminal)

    def counts(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for task in self.tasks.values():
                out[task.state.value] = out.get(task.state.value, 0) + 1
            return out

    def topological_order(self) -> list[Task]:
        """A valid serial execution order (for debugging and for dry runs)."""
        with self._lock:
            remaining = dict(self.tasks)
            done: set[str] = set()
            order: list[Task] = []
            while remaining:
                batch = [
                    t for t in remaining.values() if set(t.deps) <= done
                ]
                if not batch:  # pragma: no cover - acyclicity is enforced on add
                    raise ValidationError("graph became cyclic")
                batch.sort(key=lambda t: (-t.priority, t.created_at))
                for task in batch:
                    order.append(task)
                    done.add(task.id)
                    del remaining[task.id]
            return order

    def depth(self) -> int:
        """Length of the critical path — the minimum possible wall-clock steps."""
        memo: dict[str, int] = {}

        def walk(task_id: str, seen: set[str]) -> int:
            if task_id in memo:
                return memo[task_id]
            if task_id in seen:  # pragma: no cover - acyclicity enforced
                return 0
            seen.add(task_id)
            task = self.tasks[task_id]
            value = 1 + max((walk(d, seen) for d in task.deps if d in self.tasks), default=0)
            seen.discard(task_id)
            memo[task_id] = value
            return value

        with self._lock:
            return max((walk(t, set()) for t in self.tasks), default=0)

    # ── control ──────────────────────────────────────────────────────────────
    def cancel(self, reason: str = "cancelled") -> int:
        """Cancel every non-terminal task. Returns how many were cancelled."""
        self._cancel.set()
        with self._lock:
            tasks = [t for t in self.tasks.values() if not t.is_terminal]
        for task in tasks:
            self._transition(task, TaskState.CANCELLED, reason)
        return len(tasks)

    def retry_later(self, task: Task, delay: float, reason: str = "") -> bool:
        """Requeue a task for retry after ``delay`` seconds.

        Respects the graph-wide :class:`RetryBudget`: when the budget is
        exhausted the task fails fast instead. Returns True when requeued.
        """
        with self._lock:
            if task.state not in (TaskState.FAILED, TaskState.PENDING):
                return False
            if not self.retry_budget.spend():
                self._transition(
                    task, TaskState.FAILED,
                    f"retry budget exhausted: {reason or task.error}",
                )
                return False
            old = task.state
            task.state = TaskState.PENDING
            task.not_before = time.monotonic() + max(0.0, delay)
            if reason:
                task.error = reason
            self.notify(task, old, TaskState.PENDING)
            return True

    def requeue_dead(self, key: str) -> Task:
        """Operational escape hatch: reset a FAILED task to PENDING so it can
        be replayed after the cause is fixed (the manual DLQ replay)."""
        task = self.require(key)
        with self._lock:
            if task.state is not TaskState.FAILED:
                raise ValidationError(
                    f"task {task.name!r} is {task.state.value}, not failed"
                )
            old = task.state
            task.state = TaskState.PENDING
            task.error = ""
            task.not_before = 0.0
            task.attempts = 0
            self.notify(task, old, TaskState.PENDING)
        return task

    def dead_letters(self) -> list[dict[str, Any]]:
        """Every FAILED task as a diagnosable record — the dead-letter queue."""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "id": t.id,
                    "error": t.error,
                    "attempts": t.attempts,
                    "duration": round(t.duration, 4),
                    "deps": list(t.deps),
                    "replayable": True,
                }
                for t in self.tasks.values()
                if t.state is TaskState.FAILED
            ]

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def reset(self) -> None:
        """Return every task to PENDING (for re-running a graph)."""
        self._cancel.clear()
        self.retry_budget.reset()
        with self._lock:
            tasks = list(self.tasks.values())
        for task in tasks:
            old = task.state
            task.state = TaskState.PENDING
            task.result = None
            task.error = ""
            task.attempts = 0
            task.not_before = 0.0
            task.started_at = 0.0
            task.finished_at = 0.0
            self.notify(task, old, TaskState.PENDING)

    # ── execution ────────────────────────────────────────────────────────────
    def dry_run(self, *, fail_fast: bool = False) -> dict[str, Any]:
        """Execute the graph serially in topological order, without threads.

        Dependents of a failed task are SKIPPED (visible, like the real
        scheduler's :meth:`cascade_skip`). Returns a summary; raises nothing
        the tasks themselves didn't raise. Use it to validate a mission plan
        before spending tokens on parallel execution.
        """
        ran = failed = skipped = 0
        started = time.monotonic()
        for task in self.topological_order():
            if self.cancelled:
                break
            if any(
                self.tasks[d].state is not TaskState.DONE
                for d in task.deps
                if d in self.tasks
            ):
                self._transition(task, TaskState.SKIPPED, "dry run: dependency not done")
                skipped += 1
                if fail_fast:
                    break
                continue
            self._transition(task, TaskState.RUNNING)
            try:
                result = task.fn(*task.args, **task.kwargs) if task.fn else None
            except Exception as exc:  # noqa: BLE001 - dry run records, like the executor
                self._transition(task, TaskState.FAILED, f"{type(exc).__name__}: {exc}")
                failed += 1
                if fail_fast:
                    break
                continue
            task.result = result
            self._transition(task, TaskState.DONE)
            ran += 1
        return {
            "ran": ran,
            "failed": failed,
            "skipped": skipped,
            "wall_s": round(time.monotonic() - started, 4),
            "failures": self.failures(),
        }

    # ── reporting ────────────────────────────────────────────────────────────
    def results(self) -> dict[str, Any]:
        with self._lock:
            return {t.name: t.result for t in self.tasks.values() if t.state is TaskState.DONE}

    def failures(self) -> dict[str, str]:
        with self._lock:
            return {t.name: t.error for t in self.tasks.values() if t.state is TaskState.FAILED}

    def audit(self) -> list[str]:
        """Check the graph for silent corruption after a run.

        Returns a list of problems (empty = the run's bookkeeping is
        honest).  Catches: DONE tasks with no result, FAILED tasks with
        no recorded error, and tasks that ran (or are queued to run)
        despite a failed dependency — i.e. a task that died mid-run must
        never have its dependents silently marked complete.
        """
        problems: list[str] = []
        with self._lock:
            tasks = dict(self.tasks)
        for task in tasks.values():
            if task.state is TaskState.DONE and task.result is None:
                problems.append(f"{task.name}: DONE with no result")
            if task.state is TaskState.FAILED and not task.error:
                problems.append(f"{task.name}: FAILED with no recorded error")
        failed_ids = {
            t.id for t in tasks.values() if t.state is TaskState.FAILED
        }
        for task in tasks.values():
            if not failed_ids:
                break
            if set(task.deps) & failed_ids and task.state not in (
                TaskState.SKIPPED,
                TaskState.CANCELLED,
                TaskState.FAILED,
                TaskState.PENDING,
            ):
                problems.append(
                    f"{task.name}: state {task.state.value} despite failed dependency"
                )
        return problems

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "mission_id": self.mission_id,
                "counts": self.counts(),
                "depth": self.depth(),
                "retry_budget_remaining": self.retry_budget.remaining(),
                "tasks": [t.to_dict() for t in self.topological_order()],
            }

    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
