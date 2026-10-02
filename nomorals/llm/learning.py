"""Trajectory + benchmark learning in the serving path.

Wave I built the learner's brain (:class:`TrajectoryStore
<nomorals.cognition.trajectories.TrajectoryStore>`,
:class:`BenchmarkDB <nomorals.llm.benchmarks.BenchmarkDB>`,
:class:`ModelBroker <nomorals.llm.broker.ModelBroker>`) but nothing fed it
live serving outcomes — the broker ranked on priors forever.  This module
closes that gap: :func:`attach_learning` wires a router's dispatch path to
a trajectory store and a benchmark DB, so every provider *attempt* becomes
one trajectory row (task_kind / capability / model / success / latency /
error) and one ``source='live'`` benchmark row.

Design rules:

* **Never block the hot path.**  Outcomes are enqueued and written by a
  single daemon thread; a slow or wedged SQLite never delays a chat reply.
* **Best-effort, always.**  If the trajectory DB is missing, corrupt, or
  unwritable, the attachment degrades to a no-op and the router behaves
  exactly as before (logged, never raised).
* **No behaviour change without evidence — almost.**  When the router has
  no broker, one is built from its providers via ``build_from_router`` so
  recorded history has something to steer.  The boot primary is
  deliberately *not* promoted: a promotion would pin it forever,
  preventing trajectory/benchmark evidence from ever changing selection
  (and it would fight later explicit ``set_active`` calls).  Consequence:
  with zero history every card scores the 0.5 prior and ties break by
  card id, so the very first selections are alphabetical by provider
  name.  In the standard layout (``hf_serverless`` primary,
  ``openrouter``/``ocr`` fallbacks) the primary still wins those ties;
  as soon as live rows land, evidence takes over.
* **Existing brokers are respected.**  A hand-attached broker only gets a
  trajectory store injected if it doesn't already have one; its benchmark
  DB is never replaced.

Layer: L3 (``nomorals.llm``).  The same-layer import of
``nomorals.cognition.trajectories`` is allowed by the layering test —
only *upward* imports are violations.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from ..cognition.trajectories import TrajectoryStore
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .benchmarks import BenchmarkDB

__all__ = ["LearningAttachment", "attach_learning", "default_learning_db_path"]

_log = get_logger(__name__)

#: Router operation → capability value recorded on trajectory/benchmark
#: rows.  Mirrors ``ModelBroker.capability_for_operation`` so the rows land
#: in exactly the slice the broker's ``success_rate`` / ``summary`` lookups
#: query.
_OPERATION_CAPABILITY = {
    "chat": "chat",
    "complete": "chat",
    "vision": "vision",
    "embed": "embed",
}

_LIVE_SOURCE = "live"
_QUEUE_MAXSIZE = 4096


def default_learning_db_path() -> Path:
    """Where learning tables live when no ``db``/``db_path`` is given.

    The main state DB — the same file the agent context, lifecycle
    registry, and ``nm models`` benchmark commands use — so live rows are
    visible to ``nm models select`` immediately.
    """
    from ..core.config import get_settings

    return get_settings().db_path


class LearningAttachment:
    """Live handle returned by :func:`attach_learning`.

    Owns the trajectory store, the benchmark DB, the background writer
    thread, and (when it had to build one) the broker.  Call
    :meth:`flush` in tests to wait for queued outcomes to land, and
    :meth:`detach` to unhook the router.
    """

    def __init__(
        self,
        router: Any,
        trajectories: TrajectoryStore | None,
        benchmarks: BenchmarkDB | None,
        broker: Any | None,
        *,
        degraded: bool = False,
        reason: str = "",
    ) -> None:
        self.router = router
        self.trajectories = trajectories
        self.benchmarks = benchmarks
        self.broker = broker
        self.degraded = degraded
        self.degraded_reason = reason
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="llm-learning", daemon=True
        )
        self._worker.start()

    # ── hot-path entry (called by LLMRouter._dispatch) ────────────────────
    def note(
        self,
        *,
        operation: str,
        provider_name: str,
        success: bool,
        latency_s: float,
        error: str = "",
    ) -> None:
        """Enqueue one attempt outcome.  Never blocks, never raises."""
        if self.degraded:
            return
        try:
            self._queue.put_nowait(
                (operation, provider_name, bool(success),
                 float(latency_s), error or "")
            )
        except queue.Full:
            _log.warning(
                "learning queue full; dropping outcome for %s", provider_name
            )
        except Exception:  # noqa: BLE001 — learning must never break routing
            _log.debug("learning note failed", exc_info=True)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every queued outcome has been written.  Test seam."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def detach(self) -> None:
        """Unhook the router and stop the writer.  Idempotent."""
        try:
            setter = getattr(self.router, "set_learning", None)
            if callable(setter):
                setter(None)
            else:  # duck-typed router without the setter
                self.router._learning = None
        except Exception:  # noqa: BLE001
            _log.debug("learning detach failed", exc_info=True)
        try:
            marker = getattr(self.router, "_learning_attachment", None)
            if marker is self:
                delattr(self.router, "_learning_attachment")
        except Exception:  # noqa: BLE001
            _log.debug("learning marker cleanup failed", exc_info=True)
        self._stop.set()
        # No join(): the worker may be mid-insert on a wedged disk and we
        # must never hang teardown; it is a daemon, so process exit is
        # unaffected.

    # ── background writer ────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._write(item)
            except Exception:  # noqa: BLE001 — the worker must never die
                _log.warning("learning write failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _write(self, item: tuple) -> None:
        operation, provider_name, success, latency_s, error = item
        capability = _OPERATION_CAPABILITY.get(operation, operation)
        if self.trajectories is not None:
            try:
                self.trajectories.record(
                    task_kind=operation,
                    capability=capability,
                    model_id=provider_name,
                    success=success,
                    latency_s=latency_s,
                    error=error,
                )
            except Exception:  # noqa: BLE001
                _log.warning(
                    "trajectory record failed for %s", provider_name,
                    exc_info=True,
                )
        if self.benchmarks is not None:
            try:
                self.benchmarks.record(
                    provider_name,
                    capability,
                    latency_s,
                    success=success,
                    source=_LIVE_SOURCE,
                    task_kind=operation,
                )
            except Exception:  # noqa: BLE001
                _log.warning(
                    "benchmark record failed for %s", provider_name,
                    exc_info=True,
                )


def _ensure_broker(
    router: Any,
    trajectories: TrajectoryStore,
    benchmarks: BenchmarkDB,
) -> Any | None:
    """Return the router's broker, building one from its providers if needed.

    The auto-built broker is deliberately *not* given an operator override:
    promoting the boot primary would pin it forever, so trajectory and
    benchmark evidence could never change selection (and the pin would
    fight later explicit ``set_active`` calls).  With no history every
    card scores the 0.5 prior and ties break deterministically by card
    id; live rows take over from there.

    A pre-existing broker only gets a trajectory store injected when it
    has none — its own benchmark DB is never replaced.
    """
    from .broker import ModelBroker

    try:
        broker = router.broker
    except Exception:  # noqa: BLE001 — duck-typed router
        broker = None
    if broker is None:
        broker = ModelBroker(benchmarks=benchmarks, trajectories=trajectories)
        try:
            broker.build_from_router(router)
        except Exception:  # noqa: BLE001
            _log.warning(
                "learning: could not build broker cards from router",
                exc_info=True,
            )
        try:
            router.set_broker(broker)
        except Exception:  # noqa: BLE001
            _log.warning(
                "learning: could not attach broker to router", exc_info=True
            )
            return None
        return broker
    if getattr(broker, "trajectories", None) is None:
        broker.trajectories = trajectories
    return broker


def attach_learning(
    router: Any,
    db_path: str | Path | None = None,
    *,
    db: Database | None = None,
) -> LearningAttachment:
    """Attach trajectory/benchmark learning to a router's dispatch path.

    * Builds a :class:`TrajectoryStore` and :class:`BenchmarkDB` — on the
      given ``db``/``db_path``, else the default state-dir DB — and hooks
      outcome recording into every ``_dispatch`` attempt.
    * Injects the store into the router's broker, or builds one from the
      router's providers (no operator override — history decides) when
      none is attached.

    Idempotent per router: a second call returns the existing attachment.
    If the store cannot be opened (missing/corrupt DB), the returned
    attachment is ``degraded`` — recording is a no-op and the router works
    exactly as before.  Never raises for store problems.
    """
    existing = getattr(router, "_learning_attachment", None)
    if isinstance(existing, LearningAttachment):
        return existing

    trajectories: TrajectoryStore | None = None
    benchmarks: BenchmarkDB | None = None
    degraded_reason = ""
    try:
        database = (
            db
            if db is not None
            else Database(str(db_path) if db_path else str(default_learning_db_path()))
        )
        trajectories = TrajectoryStore(database)
        benchmarks = BenchmarkDB(database)
    except Exception as exc:  # noqa: BLE001 — learning is best-effort
        degraded_reason = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "learning store unavailable (%s); routing continues without learning",
            degraded_reason,
        )

    broker: Any | None = None
    if trajectories is not None and benchmarks is not None:
        broker = _ensure_broker(router, trajectories, benchmarks)

    attachment = LearningAttachment(
        router,
        trajectories,
        benchmarks,
        broker,
        degraded=trajectories is None,
        reason=degraded_reason,
    )
    router._learning_attachment = attachment
    if not attachment.degraded:
        try:
            router.set_learning(attachment.note)
        except Exception:  # noqa: BLE001
            _log.warning(
                "learning: could not hook into router dispatch", exc_info=True
            )
    else:
        _log.warning(
            "learning degraded (%s): dispatch hook not installed",
            degraded_reason or "unknown",
        )
    return attachment
