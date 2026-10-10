"""Virtual CPUs (wave 84): independent execution engines.

A :class:`VirtualCPU` is a dedicated daemon worker thread with its own
priority queue, its own isolated execution context (nothing shared
with other VCPUs except the task's payload), and its own resource
accounting.  It is the unit the Workspace scales and the unit the
Orchestrator assigns work to.

Lifecycle::

    OFFLINE ──start()──▶ IDLE ──submit()──▶ BUSY ──queue empty──▶ IDLE
      ▲                      │                    │
      │                     pause()             crash (×3)
      │                      ▼                    ▼
    OFFLINE ◀──stop()──  PAUSED ──resume()──▶ BUSY      OFFLINE (ERROR)

Task results come back as standard :class:`concurrent.futures.Future`
objects, so a VCPU can sit anywhere an executor would — including as
the substrate of the HybridExecutor's IO lane.

Pause semantics: a PAUSED VCPU keeps ACCEPTING work (it goes on the
queue) but does not execute until :meth:`resume` — the operator can
drain a core's load by pausing it, and :meth:`stop` with
``drain=True`` runs the queued work before going offline.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

__all__ = ["VirtualCPU", "VcpuStatus"]


class VcpuStatus:
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"


#: VCPU kinds — the affinity a task asks for
KINDS = ("balanced", "io", "cpu")

#: how many consecutive worker-level (not task-level) crashes before a
#: VCPU gives up and goes offline
MAX_RESTARTS = 3


@dataclass
class _WorkItem:
    seq: int
    priority: int          # higher runs first (0 = normal)
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    future: Future
    submitted_at: float = field(default_factory=time.time)
    name: str = ""         # human label, for observability
    timeout: float = 0.0   # soft per-task timeout in seconds (0 = none)


class VirtualCPU:
    """One independent execution core."""

    def __init__(self, index: int, *, kind: str = "balanced", name: str = "",
                 max_queue: int = 256, auto_start: bool = True,
                 max_tasks: int = 0) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown VCPU kind {kind!r}; expected {KINDS}")
        self.index = int(index)
        self.kind = kind
        self.name = name or f"vcpu{index}"
        self.max_queue = max(1, int(max_queue))
        #: Celery-style rebirth: restart the worker thread after this many
        #: completed tasks (0 = never).  Fights slow memory leaks in
        #: long-lived agents without losing queued work.
        self.max_tasks = max(0, int(max_tasks))

        self.status = VcpuStatus.OFFLINE
        self._queue: list[_WorkItem] = []          # heap by (-priority, seq)
        self._cond = threading.Condition()
        self._seq = itertools.count()
        self._stop = False
        self._pause = False
        self._thread: threading.Thread | None = None
        self._restarts = 0
        self._last_error = ""
        self._current_task = ""          # name of the running task, if any
        self._orphans = 0               # timed-out tasks still running detached

        # resource awareness
        self._busy_seconds = 0.0
        self._idle_seconds = 0.0
        self._busy_ema = 0.0                        # smoothed busy fraction
        self._window_start = time.monotonic()
        #: rolling per-task durations (seconds) for latency percentiles
        self._durations: collections.deque[float] = collections.deque(
            maxlen=256)

        # stats
        self.stats = {
            "tasks_run": 0,
            "tasks_failed": 0,
            "tasks_queued": 0,
            "tasks_aborted": 0,
            "tasks_timed_out": 0,
            "tasks_stolen_in": 0,
            "tasks_stolen_out": 0,
            "task_seconds": 0.0,
            "queue_peak": 0,
            "queue_wait_seconds": 0.0,
            "restarts": 0,
            "rebirths": 0,
        }
        self._lock = threading.Lock()
        if auto_start:
            self.start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> "VirtualCPU":
        with self._cond:
            if self.status == VcpuStatus.OFFLINE and self._thread is not None \
                    and self._thread.is_alive():
                return self
            self._stop = False
            self._window_start = time.monotonic()
            self._thread = threading.Thread(
                target=self._run, name=f"nm-{self.name}", daemon=True)
            self.status = VcpuStatus.IDLE
            self._thread.start()
        return self

    def stop(self, *, drain: bool = False, timeout: float = 5.0) -> None:
        """Take the core offline.  ``drain=True`` runs queued work first."""
        with self._cond:
            if self._thread is None:
                self.status = VcpuStatus.OFFLINE
                return
            self._stop = True
            self._pause = False
            self._cond.notify_all()
        if drain:
            self.drain()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            self.status = VcpuStatus.OFFLINE

    def drain(self) -> int:
        """Run everything currently queued, inline on the calling thread.
        Used by stop(drain=True) and scale-down: the work is finished,
        not lost.  Returns how many items ran."""
        n = 0
        while True:
            with self._cond:
                if not self._queue:
                    return n
                item = heapq.heappop(self._queue)[2]
            self._execute(item)
            n += 1

    def pause(self) -> None:
        with self._cond:
            self._pause = True
            if self.status in (VcpuStatus.IDLE, VcpuStatus.BUSY):
                self.status = VcpuStatus.PAUSED

    def resume(self) -> None:
        with self._cond:
            self._pause = False
            if self.status == VcpuStatus.PAUSED:
                self.status = VcpuStatus.IDLE
                self._cond.notify_all()

    def abort_pending(self) -> int:
        """Drop everything queued (not running).  Returns how many."""
        with self._cond:
            n = len(self._queue)
            for entry in self._queue:
                item = entry[2]
                if not item.future.done():
                    item.future.set_exception(
                        RuntimeError(f"aborted on {self.name}"))
            self._queue.clear()
            with self._lock:
                self.stats["tasks_aborted"] += n
            self._cond.notify_all()
        return n

    # ── work ─────────────────────────────────────────────────────────────────
    def submit(self, fn: Callable[..., Any], *args: Any,
               priority: int = 0, name: str = "",
               timeout: float = 0.0, **kwargs: Any) -> Future:
        """Queue a task.  Returns a standard Future (result or exception).

        ``name`` labels the task for observability (visible in
        :meth:`to_dict` as ``current_task`` while running).  ``timeout``
        is a *soft* per-task limit in seconds: if the task has not
        finished in time the Future resolves with :class:`TimeoutError`
        while the runaway thread keeps going detached (a daemon — Python
        cannot kill threads).  ``stats["tasks_timed_out"]`` counts them.

        Blocks (bounded by backpressure) when the queue is full rather
        than dropping work — a full VCPU is a signal the Workspace
        should scale, not to lose the task.
        """
        if self.status == VcpuStatus.OFFLINE:
            raise RuntimeError(f"VCPU {self.name} is offline")
        future: Future = Future()
        item = _WorkItem(seq=next(self._seq), priority=int(priority),
                         fn=fn, args=args, kwargs=kwargs, future=future,
                         name=str(name or getattr(fn, "__name__", ""))[:80],
                         timeout=max(0.0, float(timeout or 0.0)))
        with self._cond:
            if self.status == VcpuStatus.OFFLINE:
                raise RuntimeError(f"VCPU {self.name} went offline")
            deadline = time.monotonic() + max(1.0, self.max_queue * 0.5)
            while len(self._queue) >= self.max_queue:
                if not self._cond.wait(timeout=0.25):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"VCPU {self.name} queue full (backpressure)")
            heapq.heappush(self._queue, (-item.priority, item.seq, item))
            with self._lock:
                self.stats["tasks_queued"] += 1
                self.stats["queue_peak"] = max(
                    self.stats["queue_peak"], len(self._queue))
            self._cond.notify_all()
        return future

    def submit_many(self, items: list[tuple | dict]) -> list[Future]:
        """Batch-submit tasks with a single lock pass.

        Each element is either a ``(fn, *args)`` tuple or a dict with
        ``fn`` plus optional ``args``/``kwargs``/``priority``/``name``/
        ``timeout`` keys.  Returns the Futures in order.
        """
        futures: list[Future] = []
        with self._cond:
            if self.status == VcpuStatus.OFFLINE:
                raise RuntimeError(f"VCPU {self.name} is offline")
            for spec in items:
                if isinstance(spec, dict):
                    fn = spec["fn"]
                    args = tuple(spec.get("args", ()))
                    kwargs = dict(spec.get("kwargs", {}))
                    priority = int(spec.get("priority", 0))
                    name = str(spec.get("name", ""))
                    timeout = float(spec.get("timeout", 0.0) or 0.0)
                else:
                    fn, *args = spec
                    kwargs, priority, name, timeout = {}, 0, "", 0.0
                future: Future = Future()
                item = _WorkItem(
                    seq=next(self._seq), priority=priority, fn=fn,
                    args=tuple(args), kwargs=kwargs, future=future,
                    name=name or getattr(fn, "__name__", ""),
                    timeout=max(0.0, timeout))
                heapq.heappush(self._queue, (-item.priority, item.seq, item))
                futures.append(future)
            with self._lock:
                self.stats["tasks_queued"] += len(futures)
                self.stats["queue_peak"] = max(
                    self.stats["queue_peak"], len(self._queue))
            self._cond.notify_all()
        return futures

    def steal_from(self, other: "VirtualCPU", max_n: int = 0) -> int:
        """Work-stealing (Tokio-style): take up to ``max_n`` queued tasks
        from a busier sibling.  ``max_n=0`` takes half the victim's queue.

        Steals the *oldest, lowest-priority* items first (the cold end of
        the victim's queue) so the victim keeps its hot work.  Returns
        how many items moved.
        """
        if other is self:
            return 0
        # lock ordering: always victim first, then self — every stealer
        # follows the same order, so no deadlock cycle is possible
        first, second = (other, self) if id(other) < id(self) else (self, other)
        with first._cond:
            with second._cond:
                if other.status == VcpuStatus.OFFLINE or not other._queue:
                    return 0
                n = max_n if max_n > 0 else max(1, len(other._queue) // 2)
                n = min(n, len(other._queue))
                # cold end = lowest priority, oldest seq = heap-largest
                cold = heapq.nlargest(n, other._queue)
                cold_ids = {id(entry[2]) for entry in cold}
                other._queue = [e for e in other._queue
                                if id(e[2]) not in cold_ids]
                heapq.heapify(other._queue)
                for entry in cold:
                    item = entry[2]
                    item.seq = next(self._seq)  # re-sequence on arrival
                    heapq.heappush(self._queue,
                                   (-item.priority, item.seq, item))
                other._cond.notify_all()
                self._cond.notify_all()
        with other._lock:
            other.stats["tasks_stolen_out"] += n
        with self._lock:
            self.stats["tasks_stolen_in"] += n
            self.stats["queue_peak"] = max(
                self.stats["queue_peak"], len(self._queue))
        return n

    def queue_depth(self) -> int:
        with self._cond:
            return len(self._queue)

    # ── resource awareness ───────────────────────────────────────────────────
    @property
    def busy_fraction(self) -> float:
        """Smoothed fraction of time the core has been executing."""
        return round(min(1.0, self._busy_ema), 3)

    @property
    def queue_pressure(self) -> float:
        with self._cond:
            return min(1.0, len(self._queue) / self.max_queue)

    @property
    def load(self) -> float:
        """Composite dispatch load, 0 (free) → 1 (saturated)."""
        return round(min(1.0,
                         0.7 * self.busy_fraction
                         + 0.3 * self.queue_pressure), 3)

    def utilization(self) -> float:
        now = time.monotonic()
        span = max(0.001, now - self._window_start)
        return self._busy_seconds / span

    def _note_work(self, busy_dt: float, waited: float) -> None:
        self._busy_seconds += busy_dt
        # refresh the EWMA busy fraction (alpha 0.25 per task)
        instant = min(1.0, busy_dt / max(0.001, busy_dt + waited))
        self._busy_ema = 0.75 * self._busy_ema + 0.25 * instant
        self._durations.append(busy_dt)

    def latency_stats(self) -> dict[str, Any]:
        """Rolling per-task latency percentiles (last ≤256 tasks)."""
        with self._lock:
            ds = sorted(self._durations)
        if not ds:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0,
                    "max_ms": 0.0, "mean_ms": 0.0}

        def pct(q: float) -> float:
            i = min(len(ds) - 1, int(q * len(ds)))
            return round(ds[i] * 1000, 2)

        return {"count": len(ds), "p50_ms": pct(0.50), "p95_ms": pct(0.95),
                "max_ms": round(ds[-1] * 1000, 2),
                "mean_ms": round(sum(ds) / len(ds) * 1000, 2)}

    # ── the worker loop ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stop:
                        self.status = VcpuStatus.OFFLINE
                        return
                    if self._queue and not self._pause:
                        break
                    if self._queue and self._pause:
                        self.status = VcpuStatus.PAUSED
                    elif not self._queue:
                        self.status = VcpuStatus.IDLE
                    if not self._cond.wait(timeout=0.5):
                        continue
                item = heapq.heappop(self._queue)[2]
            self.status = VcpuStatus.BUSY
            try:
                self._execute(item)
            except BaseException as exc:  # noqa: BLE001 — worker-level crash
                # the task's future already carries the exception; if we
                # get HERE the worker itself is broken (e.g. a C-level
                # fault) — restart the core, a few times, then give up
                with self._lock:
                    self.stats["restarts"] += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"[:200]
                self._restarts += 1
                if self._stop:
                    return
                if self._restarts >= MAX_RESTARTS:
                    with self._lock:
                        self.status = VcpuStatus.ERROR
                    return
                self.start()
                return
            # Celery-style rebirth: retire this worker thread after
            # max_tasks completions so slow leaks die with it.  Queued
            # work is untouched — the replacement thread picks it up.
            if self.max_tasks and not self._stop:
                with self._lock:
                    done = (self.stats["tasks_run"]
                            + self.stats["tasks_failed"])
                if done >= self.max_tasks:
                    with self._lock:
                        self.stats["rebirths"] += 1
                    self.start()   # spawns the replacement thread
                    return         # ...and this one exits

    def _execute(self, item: _WorkItem) -> None:
        """Run one item with the task error kept on its Future only.

        With a per-task ``timeout`` the task runs in a helper thread and
        the Future resolves with TimeoutError when the limit expires —
        the runaway thread keeps going detached (daemon); Python cannot
        kill threads, so this is a *soft* limit, honestly reported via
        ``stats["tasks_timed_out"]`` and the ``orphan_threads`` count.
        """
        self._current_task = item.name
        try:
            if item.timeout > 0:
                self._execute_with_timeout(item)
            else:
                self._execute_inline(item)
        finally:
            self._current_task = ""

    def _execute_inline(self, item: _WorkItem) -> None:
        waited = time.time() - item.submitted_at
        started = time.perf_counter()
        try:
            result = item.fn(*item.args, **item.kwargs)
        except BaseException as exc:  # noqa: BLE001
            item.future.set_exception(exc)
            with self._lock:
                self.stats["tasks_failed"] += 1
                self.stats["task_seconds"] += time.perf_counter() - started
                self._note_work(time.perf_counter() - started, waited)
            _log_task_error(self.name, item, exc)
            return
        if not item.future.set_running_or_notify_cancel():
            return
        item.future.set_result(result)
        with self._lock:
            self.stats["tasks_run"] += 1
            self.stats["task_seconds"] += time.perf_counter() - started
            self._note_work(time.perf_counter() - started, waited)

    def _execute_with_timeout(self, item: _WorkItem) -> None:
        waited = time.time() - item.submitted_at
        started = time.perf_counter()
        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["result"] = item.fn(*item.args, **item.kwargs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        helper = threading.Thread(target=_target,
                                  name=f"nm-{self.name}-task",
                                  daemon=True)
        helper.start()
        helper.join(timeout=item.timeout)
        elapsed = time.perf_counter() - started
        if helper.is_alive():
            # soft timeout: the future dies, the thread lives on detached
            with self._lock:
                self.stats["tasks_timed_out"] += 1
                self.stats["task_seconds"] += elapsed
                self._note_work(elapsed, waited)
                self._orphans += 1
            if item.future.set_running_or_notify_cancel():
                item.future.set_exception(TimeoutError(
                    f"task {item.name or 'unnamed'} on {self.name} exceeded "
                    f"{item.timeout:.1f}s (soft timeout — still running "
                    "detached)"))
            _log_task_error(self.name, item,
                            TimeoutError("soft task timeout"))
            return
        if "error" in box:
            exc = box["error"]
            item.future.set_exception(exc)
            with self._lock:
                self.stats["tasks_failed"] += 1
                self.stats["task_seconds"] += elapsed
                self._note_work(elapsed, waited)
            _log_task_error(self.name, item, exc)
            return
        if not item.future.set_running_or_notify_cancel():
            return
        item.future.set_result(box.get("result"))
        with self._lock:
            self.stats["tasks_run"] += 1
            self.stats["task_seconds"] += elapsed
            self._note_work(elapsed, waited)

    # ── reporting ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            orphans = self._orphans
        return {
            "id": self.name,
            "index": self.index,
            "kind": self.kind,
            "status": self.status,
            "load": self.load,
            "busy_fraction": self.busy_fraction,
            "queue": self.queue_depth(),
            "max_queue": self.max_queue,
            "current_task": self._current_task,
            "orphan_threads": orphans,
            "latency": self.latency_stats(),
            "restarts": self.stats["restarts"],
            "last_error": self._last_error,
            "stats": dict(self.stats),
        }


def _log_task_error(name: str, item: _WorkItem, exc: BaseException) -> None:
    try:
        from ..core.logging_setup import get_logger
        get_logger(__name__).debug(
            "vcpu %s task failed: %s: %s",
            name, type(exc).__name__, str(exc)[:160])
    except Exception:  # noqa: BLE001
        pass
