"""Hybrid execution runtime: threads + processes + asyncio, one scheduler.

Why three substrates instead of one:

* The **GIL** makes threads useless for CPU-bound work. Tokenizing 100k documents,
  embedding batches, and deduping a corpus are CPU-bound, so they need processes.
* **Processes** are expensive to spawn and can only exchange picklable data, so
  they are wrong for HTTP-bound work.
* **asyncio** is the right tool for very high fan-out I/O — 200 concurrent
  downloads — but it cannot host blocking calls without stalling the loop.

Tasks declare their kind; the runtime places them. That single rule is what keeps
a mixed system comprehensible.

Robustness notes that cost real bugs to learn:

* If the process pool cannot be created (Android/Termux, no fork, sandbox), CPU
  tasks **fall back to threads** instead of failing the mission.
* If a task's callable is not picklable, it falls back to a thread rather than
  raising at submit time — the work still gets done.
* Cancellation is cooperative *and* enforced: running tasks get a deadline, and
  the scheduler stops dispatching immediately.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import multiprocessing
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..core.errors import DeadlineExceeded, TaskCancelled, TaskFailed, classify
from ..core.logging_setup import get_logger
from ..core.platform import detect_platform
from ..core.tasks import Task, TaskGraph, TaskKind, TaskState

__all__ = ["ExecutionReport", "HybridExecutor", "run_pickled"]

_log = get_logger(__name__)


def run_pickled(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Module-level trampoline so process-pool payloads stay picklable."""
    return fn(*args, **kwargs)


@dataclass
class ExecutionReport:
    """What happened during a graph run."""

    total: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    skipped: int = 0
    wall_seconds: float = 0.0
    #: Sum of per-task elapsed time. NOT CPU time: tasks overlap, so this can
    #: exceed wall_seconds * cores. It is the serial-equivalent workload.
    task_seconds: float = 0.0
    by_kind: dict[str, int] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def speedup_vs_serial(self) -> float:
        """Serial-equivalent workload divided by wall time.

        Bounded by the machine's core count, so 2.0 on a 2-core box means the
        scheduler is keeping both cores busy — not that it is underperforming.
        """
        return round(self.task_seconds / self.wall_seconds, 2) if self.wall_seconds else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "skipped": self.skipped,
            "wall_seconds": round(self.wall_seconds, 3),
            "task_seconds": round(self.task_seconds, 3),
            "speedup_vs_serial": self.speedup_vs_serial,
            "by_kind": self.by_kind,
            "failures": self.failures,
        }


class _AsyncRunner:
    """Owns one event loop on a dedicated thread."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._spin, args=(self._loop,), name="nm-asyncio", daemon=True
            )
            self._thread.start()
            return self._loop

    @staticmethod
    def _spin(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def submit(self, coro_factory: Callable[[], Any]) -> cf.Future:
        loop = self._ensure()
        return asyncio.run_coroutine_threadsafe(coro_factory(), loop)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            loop, self._loop = self._loop, None
            thread, self._thread = self._thread, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)


class HybridExecutor:
    """Runs a :class:`TaskGraph` across threads, processes, and an event loop."""

    def __init__(
        self,
        *,
        threads: int = 0,
        processes: int = 0,
        use_processes: bool | None = None,
        max_in_flight: int = 0,
        poll_interval: float = 0.01,
        mp_context: str = "",
    ) -> None:
        cpu = os.cpu_count() or 2
        self.threads = threads or min(32, cpu * 4)
        if use_processes is None:
            # Ask the platform backend instead of sniffing uname: Termux and
            # Android declare process_pool=False (fork is unreliable there),
            # desktop/server platforms declare True.
            use_processes = detect_platform().supports("process_pool")
        self.use_processes = use_processes
        self.processes = processes if processes else max(1, cpu // 2)
        self.max_in_flight = max_in_flight or max(self.threads, self.processes) * 2
        self.poll_interval = poll_interval

        self._thread_pool = cf.ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="nm-io"
        )
        self._process_pool: cf.ProcessPoolExecutor | None = None
        self._process_error = ""
        if self.use_processes:
            self._process_pool, self._process_error = self._build_process_pool(mp_context)
        self._async = _AsyncRunner()
        self._lock = threading.RLock()
        self.stats = {
            "submitted": 0,
            "thread_tasks": 0,
            "process_tasks": 0,
            "async_tasks": 0,
            "process_fallbacks": 0,
        }

    def _build_process_pool(
        self, mp_context: str
    ) -> tuple[cf.ProcessPoolExecutor | None, str]:
        """Build a process pool that works on this platform's start method.

        ``max_tasks_per_child`` is rejected under ``fork`` (Linux/Termux default),
        so it is only offered to ``spawn``/``forkserver``. Getting this wrong
        silently degrades CPU tasks to threads — the graph still "succeeds" while
        delivering no parallelism at all, which is why it is handled explicitly
        rather than left to a broad except.
        """
        try:
            context = multiprocessing.get_context(mp_context or None)
        except ValueError:
            context = multiprocessing.get_context()
        kwargs: dict[str, Any] = {"max_workers": self.processes, "mp_context": context}
        if context.get_start_method() in {"spawn", "forkserver"}:
            kwargs["max_tasks_per_child"] = 64
        try:
            return cf.ProcessPoolExecutor(**kwargs), ""
        except Exception as first:  # noqa: BLE001 - retry with the minimal argument set
            _log.debug("process pool with %s failed: %s", sorted(kwargs), first)
        try:
            return (
                cf.ProcessPoolExecutor(max_workers=self.processes, mp_context=context),
                "",
            )
        except Exception as exc:  # noqa: BLE001 - degrade, do not die
            message = f"{type(exc).__name__}: {exc}"
            _log.warning("process pool unavailable (%s); CPU tasks will use threads", message)
            return None, message

    # ── lifecycle ────────────────────────────────────────────────────────────
    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        self._thread_pool.shutdown(wait=wait)
        if self._process_pool is not None:
            try:
                self._process_pool.shutdown(wait=wait, cancel_futures=True)
            except TypeError:  # pragma: no cover - Python < 3.9
                self._process_pool.shutdown(wait=wait)
            except Exception as exc:  # noqa: BLE001
                _log.debug("process pool shutdown: %s", exc)
        self._async.shutdown(timeout=timeout)

    def __enter__(self) -> "HybridExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ── submission ───────────────────────────────────────────────────────────
    def submit(self, task: Task) -> cf.Future:
        """Place a task on the substrate its ``kind`` implies."""
        self.stats["submitted"] += 1
        if task.kind is TaskKind.CPU and self._process_pool is not None:
            # Probe picklability HERE. ProcessPoolExecutor.submit() pickles lazily
            # on its queue-management thread, so a bad payload does not raise at
            # submit time — it surfaces later as a broken pool, taking every other
            # in-flight task down with it. A cheap pickle.dumps up front turns that
            # into a clean per-task fallback.
            if not self._picklable(task):
                self.stats["process_fallbacks"] += 1
                _log.debug("task %s is not picklable; running it on a thread", task.name)
            else:
                try:
                    future = self._process_pool.submit(run_pickled, task.fn, task.args, task.kwargs)
                    self.stats["process_tasks"] += 1
                    return future
                except (pickle.PicklingError, TypeError, AttributeError, ValueError) as exc:
                    self.stats["process_fallbacks"] += 1
                    _log.debug("task %s rejected by the pool (%s); using a thread", task.name, exc)
                except Exception as exc:  # noqa: BLE001 - broken pool: retire it
                    self.stats["process_fallbacks"] += 1
                    _log.warning("process pool failed (%s); retiring it", exc)
                    self._retire_process_pool()

        if task.kind is TaskKind.ASYNC:
            self.stats["async_tasks"] += 1
            factory = task.fn
            return self._async.submit(lambda: factory(*task.args, **task.kwargs))

        self.stats["thread_tasks"] += 1
        return self._thread_pool.submit(task.fn, *task.args, **task.kwargs)

    @staticmethod
    def _picklable(task: Task) -> bool:
        """Can this task's callable and arguments cross a process boundary?"""
        try:
            pickle.dumps((task.fn, task.args, task.kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - any failure means "not picklable"
            return False
        return True

    def _retire_process_pool(self) -> None:
        pool, self._process_pool = self._process_pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    # ── graph execution ──────────────────────────────────────────────────────
    def run(
        self,
        graph: TaskGraph,
        *,
        on_task_done: Callable[[Task], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        deadline: float | None = None,
        fail_fast: bool = False,
    ) -> ExecutionReport:
        """Drive the graph to completion, honouring dependencies and cancellation.

        ``deadline`` is an absolute :func:`time.monotonic` timestamp. When it
        passes, undispatched tasks are cancelled and running ones are abandoned.
        """
        started = time.perf_counter()
        report = ExecutionReport(total=len(graph))
        in_flight: dict[cf.Future, Task] = {}
        task_seconds = 0.0

        try:
            while True:
                if graph.cancelled or (should_stop is not None and should_stop()):
                    graph.cancel("cancelled by caller")
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    graph.cancel("deadline exceeded")
                    break
                if graph.is_complete():
                    break

                # Retire finished futures before deciding what to dispatch.
                if in_flight:
                    done, _ = cf.wait(
                        list(in_flight),
                        timeout=self.poll_interval if not graph.ready() else 0,
                        return_when=cf.FIRST_COMPLETED,
                    )
                    for future in done:
                        task = in_flight.pop(future)
                        task_seconds += self._settle(task, future)
                        if on_task_done is not None:
                            try:
                                on_task_done(task)
                            except Exception as exc:  # noqa: BLE001 - callback must not kill the run
                                _log.warning("on_task_done raised for %s: %s", task.name, exc)
                        if fail_fast and task.state is TaskState.FAILED:
                            graph.cancel(f"fail-fast after {task.name!r}")

                # Skip tasks that can never run because an ancestor failed.
                for doomed in graph.blocked_by_failure():
                    doomed.state = TaskState.SKIPPED
                    doomed.error = "dependency failed"
                    doomed.finished_at = time.time()

                slots = self.max_in_flight - len(in_flight)
                if slots > 0:
                    for task in graph.ready()[:slots]:
                        task.mark_running()
                        try:
                            in_flight[self.submit(task)] = task
                        except Exception as exc:  # noqa: BLE001
                            task.mark_failed(f"submit failed: {classify(exc).message}")

                if not in_flight and not graph.ready():
                    # Nothing running and nothing runnable: the graph is stuck on
                    # skipped/failed nodes. Leave rather than spin.
                    break

            # Settle whatever is still in flight. On cancellation this must NOT
            # wait: a cancelled mission has to return promptly, not sit through
            # the slowest abandoned task. Python cannot kill a running thread, so
            # abandonment is cooperative — the result is discarded and the thread
            # finishes on its own.
            for future, task in list(in_flight.items()):
                if graph.cancelled:
                    future.cancel()
                    task.mark_cancelled("cancelled while running")
                    continue
                task_seconds += self._settle(task, future, wait=True)
        finally:
            pass

        counts = graph.counts()
        report.done = counts.get(TaskState.DONE.value, 0)
        report.failed = counts.get(TaskState.FAILED.value, 0)
        report.cancelled = counts.get(TaskState.CANCELLED.value, 0)
        report.skipped = counts.get(TaskState.SKIPPED.value, 0)
        report.wall_seconds = time.perf_counter() - started
        report.task_seconds = task_seconds
        report.results = graph.results()
        report.failures = graph.failures()
        with self._lock:
            for task in graph.tasks.values():
                report.by_kind[task.kind.value] = report.by_kind.get(task.kind.value, 0) + 1
        return report

    def _settle(self, task: Task, future: cf.Future, *, wait: bool = False) -> float:
        """Transfer a future's outcome onto the task. Returns its CPU seconds."""
        if not wait and not future.done():
            return 0.0
        if task.timeout is not None and not future.done():
            future.cancel()
        try:
            result = future.result(timeout=task.timeout if wait else None)
        except cf.TimeoutError:
            future.cancel()
            task.mark_failed(f"timed out after {task.timeout}s")
            return task.duration
        except cf.CancelledError:
            task.mark_cancelled("cancelled while running")
            return task.duration
        except cf.process.BrokenProcessPool as exc:
            # The pool died (OOM kill, segfault in a native extension). Retire it
            # so later tasks use threads, and requeue this one rather than
            # counting infrastructure death as a task failure.
            _log.warning("process pool broke (%s); requeueing %s on a thread", exc, task.name)
            self._retire_process_pool()
            self.stats["process_fallbacks"] += 1
            task.state = TaskState.PENDING
            task.kind = TaskKind.IO
            task.error = "requeued after broken process pool"
            return 0.0
        except Exception as exc:  # noqa: BLE001 - task errors are data, not crashes
            error = classify(exc)
            if task.attempts <= task.retries:
                task.state = TaskState.PENDING
                task.error = f"retrying ({task.attempts}/{task.retries}): {error.message}"
                return 0.0
            task.mark_failed(f"{error.code}: {error.message}")
            return task.duration
        task.mark_done(result)
        return task.duration

    # ── convenience ──────────────────────────────────────────────────────────
    def map_parallel(
        self,
        fn: Callable[[Any], Any],
        items: Sequence[Any],
        *,
        kind: TaskKind = TaskKind.IO,
        chunk_size: int = 1,
        ordered: bool = True,
        timeout: float | None = None,
    ) -> list[Any]:
        """Apply ``fn`` to every item in parallel, preserving input order.

        This is the workhorse for the data-collection and tokenization paths:
        thousands of independent items, no dependencies between them.
        """
        if not items:
            return []
        graph = TaskGraph(name="map")
        if chunk_size > 1:
            chunks = [tuple(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]
            for index, chunk in enumerate(chunks):
                graph.add(
                    Task(
                        name=f"chunk{index}",
                        fn=_apply_chunk,
                        args=(fn, chunk),
                        kind=kind,
                        timeout=timeout,
                        metadata={"width": len(chunk)},
                    )
                )
            self.run(graph)
            out: list[Any] = []
            for task in graph.topological_order():
                if task.ok:
                    out.extend(task.result)
                else:
                    out.extend([None] * int(task.metadata.get("width", 0)))
            return out

        for index, item in enumerate(items):
            graph.add(Task(name=f"item{index}", fn=fn, args=(item,), kind=kind, timeout=timeout))
        self.run(graph)
        return [t.result if t.ok else None for t in graph.topological_order()]

    def run_async(self, coro_factory: Callable[[], Any], timeout: float | None = None) -> Any:
        """Run a single coroutine on the managed loop."""
        future = self._async.submit(coro_factory)
        return future.result(timeout=timeout)

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "threads": self.threads,
            "processes": self.processes if self._process_pool else 0,
            "process_pool_error": self._process_error or None,
        }


def _apply_chunk(fn: Callable[[Any], Any], chunk: tuple) -> list[Any]:
    return [fn(item) for item in chunk]


