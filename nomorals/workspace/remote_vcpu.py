"""Remote VCPUs: one farm across phone / laptop / workstation / Kaggle GPU.

The local :class:`~nomorals.workspace.vcpu.VirtualCPU` farm runs on one
machine.  This module extends it across machines: a :class:`RemoteFarm`
holds :class:`RemoteWorker` descriptors (host, capability set, GPU flag)
and routes each submission to the best worker for the job, with
fallback to local execution when the remote worker fails.

Honesty rules (this module is a scheduler, not a transport):

* A worker executes through its ``runner`` callable —
  ``runner(fn, args, kwargs) -> result``.  The default runner runs the
  callable in a local thread pool, which is how tests and single-machine
  deployments use the farm (each "remote" worker is a real, isolated
  execution slot with its own load accounting).  A real deployment
  injects a transport runner (SSH / mesh / HTTP) that ships the
  callable to the host named in the descriptor — the farm never
  pretends a local run happened on another machine.
* :meth:`RemoteFarm.submit` returns a ``Future[RemoteResult]``.  The
  result always says which worker ran it and whether it fell back to
  local — result routing is observable, never implied.
* Fallback is fail-closed on accounting: a failed remote attempt is
  counted (``stats["remote_failures"]``) and the local re-run is
  recorded as ``fallback=True`` on the result, so a caller can tell a
  degraded run from a clean remote one.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "RemoteWorker",
    "RemoteResult",
    "RemoteFarm",
    "CAPABILITY_CPU",
    "CAPABILITY_GPU",
    "CAPABILITY_IO",
    "CAPABILITY_TRAIN",
]

CAPABILITY_CPU = "cpu"
CAPABILITY_GPU = "gpu"
CAPABILITY_IO = "io"
CAPABILITY_TRAIN = "train"

#: Default thread-pool size per simulated worker slot.
_DEFAULT_POOL = 2


@dataclass(frozen=True)
class RemoteWorker:
    """One machine in the farm.

    ``host`` is the address the transport runner dials (``"local"`` for
    the in-process slot).  ``capabilities`` names what the host can do
    (``cpu`` / ``gpu`` / ``io`` / ``train``).  ``gpu`` is kept as its
    own flag as well because GPU presence is the most common routing
    question.  ``latency_ms`` is the measured/known round-trip cost and
    feeds the picker: a far GPU loses to a near CPU for tiny jobs.
    """

    name: str
    host: str = "local"
    capabilities: frozenset[str] = frozenset({CAPABILITY_CPU})
    gpu: bool = False
    latency_ms: float = 0.0
    max_concurrent: int = _DEFAULT_POOL

    def can_run(self, required: frozenset[str]) -> bool:
        """True when this worker covers every required capability."""
        return required <= set(self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "capabilities": sorted(self.capabilities),
            "gpu": self.gpu,
            "latency_ms": self.latency_ms,
            "max_concurrent": self.max_concurrent,
        }


@dataclass
class RemoteResult:
    """A completed remote submission, with its routing attached."""

    result: Any = None
    worker: str = ""
    host: str = ""
    route: str = "local"  # "remote" | "local" | "local-fallback"
    fallback: bool = False
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "host": self.host,
            "route": self.route,
            "fallback": self.fallback,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "has_result": self.result is not None,
        }


class RemoteFarm:
    """Route work across :class:`RemoteWorker` hosts, fallback local.

    Usage::

        farm = RemoteFarm()
        farm.register(RemoteWorker("kaggle", host="kaggle-gpu",
        ...                        capabilities=frozenset({"cpu", "gpu", "train"}),
        ...                        gpu=True, latency_ms=120))
        fut = farm.submit(train_step, batch, require={"train", "gpu"})
        res = fut.result()          # RemoteResult — check res.route
        print(res.result)

    The ``runner`` per worker defaults to a local thread pool.  Pass a
    real transport (``runner=(fn, args, kwargs) -> result``) to
    :meth:`register` to execute on the actual host.
    """

    def __init__(self, *, local_workers: int = 1,
                 fallback_local: bool = True) -> None:
        self.fallback_local = bool(fallback_local)
        self._workers: dict[str, RemoteWorker] = {}
        self._runners: dict[str, Callable[..., Any]] = {}
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._inflight: dict[str, int] = {}
        self._lock = threading.Lock()
        self.stats = {
            "submitted": 0,
            "remote_ok": 0,
            "remote_failures": 0,
            "local_fallback": 0,
            "local_ok": 0,
        }
        # The local slot is always present — the fallback target and the
        # worker of last resort when no remote covers the capability.
        self._local_pool = ThreadPoolExecutor(
            max_workers=max(1, int(local_workers)),
            thread_name_prefix="remote-farm-local")
        self._local_inflight = 0

    # ── registration ────────────────────────────────────────────────────

    def register(self, worker: RemoteWorker,
                 *, runner: Callable[..., Any] | None = None) -> None:
        """Add (or replace) a worker. ``runner`` is the transport; when
        omitted the worker runs in its own local thread pool."""
        with self._lock:
            self._workers[worker.name] = worker
            self._inflight.setdefault(worker.name, 0)
            if runner is not None:
                self._runners[worker.name] = runner
            else:
                pool = ThreadPoolExecutor(
                    max_workers=max(1, int(worker.max_concurrent)),
                    thread_name_prefix=f"remote-farm-{worker.name}")
                self._pools[worker.name] = pool

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._workers:
                return False
            del self._workers[name]
            self._runners.pop(name, None)
            pool = self._pools.pop(name, None)
            self._inflight.pop(name, None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        return True

    def workers(self) -> list[RemoteWorker]:
        with self._lock:
            return list(self._workers.values())

    # ── picking ─────────────────────────────────────────────────────────

    def pick_worker(self, require: frozenset[str] | set[str] | None = None,
                    *, prefer_gpu: bool = False) -> RemoteWorker | None:
        """Best worker for the requirement, or None when no worker
        covers it (the caller then falls back to local).

        Score: workers that cover the requirement rank by
        ``inflight / max_concurrent`` (load) plus a latency penalty, so
        a busy GPU loses to an idle CPU for small jobs and a far host
        loses to a near one.  ``prefer_gpu`` breaks ties toward GPU
        hosts when the job can use one.
        """
        required = frozenset(require or ())
        with self._lock:
            candidates = [w for w in self._workers.values()
                          if w.can_run(required)]
            if prefer_gpu:
                gpu_only = [w for w in candidates if w.gpu]
                if gpu_only:
                    candidates = gpu_only
            if not candidates:
                return None

            def score(w: RemoteWorker) -> tuple[float, float, str]:
                load = self._inflight.get(w.name, 0) / max(
                    1, w.max_concurrent)
                # latency penalty: 100ms ≈ 0.1 load — near hosts win ties
                return (load + w.latency_ms / 1000.0,
                        w.latency_ms, w.name)

            return min(candidates, key=score)

    # ── submission ──────────────────────────────────────────────────────

    def submit(self, fn: Callable[..., Any], *args: Any,
               require: frozenset[str] | set[str] | None = None,
               prefer_gpu: bool = False, **kwargs: Any) -> "Future[RemoteResult]":
        """Route ``fn(*args, **kwargs)`` to the best worker.

        Returns a ``Future[RemoteResult]``.  When the picked remote
        worker raises and ``fallback_local`` is set, the task re-runs
        locally and the result carries ``route="local-fallback"`` and
        ``fallback=True``.  With fallback disabled the remote exception
        surfaces on the result's ``error`` and ``result`` stays None —
        the Future itself never raises, so routing failures are data.
        """
        with self._lock:
            self.stats["submitted"] += 1
        worker = self.pick_worker(require, prefer_gpu=prefer_gpu)
        future: Future = Future()
        if worker is None:
            self._run_local(future, fn, args, kwargs, route="local")
            return future
        self._run_remote(future, worker, fn, args, kwargs)
        return future

    def _run_remote(self, future: Future, worker: RemoteWorker,
                    fn: Callable[..., Any], args: tuple,
                    kwargs: dict) -> None:
        def _do() -> None:
            started = time.perf_counter()
            with self._lock:
                self._inflight[worker.name] = \
                    self._inflight.get(worker.name, 0) + 1
            try:
                runner = self._runners.get(worker.name)
                if runner is not None:
                    result = runner(fn, args, kwargs)
                else:
                    pool = self._pools[worker.name]
                    result = pool.submit(fn, *args, **kwargs).result()
            except Exception as exc:  # noqa: BLE001 — fallback path below
                with self._lock:
                    self._inflight[worker.name] -= 1
                    self.stats["remote_failures"] += 1
                if self.fallback_local:
                    with self._lock:
                        self.stats["local_fallback"] += 1
                    self._run_local(future, fn, args, kwargs,
                                    route="local-fallback",
                                    fallback_error=str(exc))
                    return
                future.set_result(RemoteResult(
                    worker=worker.name, host=worker.host,
                    route="remote", error=f"{type(exc).__name__}: {exc}",
                    duration_s=time.perf_counter() - started))
                return
            with self._lock:
                self._inflight[worker.name] -= 1
                self.stats["remote_ok"] += 1
            future.set_result(RemoteResult(
                result=result, worker=worker.name, host=worker.host,
                route="remote",
                duration_s=time.perf_counter() - started))

        threading.Thread(target=_do,
                         name=f"remote-farm-dispatch-{worker.name}",
                         daemon=True).start()

    def _run_local(self, future: Future, fn: Callable[..., Any],
                   args: tuple, kwargs: dict, *, route: str,
                   fallback_error: str = "") -> None:
        def _do() -> None:
            started = time.perf_counter()
            with self._lock:
                self._local_inflight += 1
            try:
                result = self._local_pool.submit(
                    fn, *args, **kwargs).result()
            except Exception as exc:  # noqa: BLE001 — reported on result
                with self._lock:
                    self._local_inflight -= 1
                future.set_result(RemoteResult(
                    worker="local", host="local", route=route,
                    fallback=(route == "local-fallback"),
                    error=(fallback_error + " | " if fallback_error else "")
                    + f"{type(exc).__name__}: {exc}",
                    duration_s=time.perf_counter() - started))
                return
            with self._lock:
                self._local_inflight -= 1
                self.stats["local_ok"] += 1
            future.set_result(RemoteResult(
                result=result, worker="local", host="local", route=route,
                fallback=(route == "local-fallback"),
                error=fallback_error,
                duration_s=time.perf_counter() - started))

        threading.Thread(target=_do, name="remote-farm-dispatch-local",
                         daemon=True).start()

    # ── reporting ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "workers": [w.to_dict() for w in self._workers.values()],
                "inflight": dict(self._inflight),
                "local_inflight": self._local_inflight,
                "fallback_local": self.fallback_local,
                "stats": dict(self.stats),
            }

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop the pools. ``wait=False`` (default) returns immediately;
        in-flight work already handed to a pool still completes."""
        self._local_pool.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
        for pool in pools:
            pool.shutdown(wait=wait, cancel_futures=True)
