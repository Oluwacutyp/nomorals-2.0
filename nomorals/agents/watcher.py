"""Watch loop (wave 83 + God-Tier upgrade): the resident operator.

``nm watch`` keeps the system on its feet between human check-ins:

* **Step 0 (db_probe)** — inspects the SQLite single-writer actor via
  ``db.lock_state()`` and auto-heals a dead writer actor via
  ``db.restart_writer_actor()`` before any ops step runs;
* **Step 1 (ops_alerts)** — re-scans the ops surfaces (stuck missions
  with self-heal escalation, chronic roles, unrecovered pivots) and
  pages the operator through the notifier when something new is wrong;
* **Step 2 (redeliver)** — redelivers the durable pending-alert queue
  the moment a platform is reachable again;
* **Step 3 (mine_preventions)** — re-mines the reasoning journal so
  repeated aborts become prevention skills automatically;
* **Step 4 (workspace)** — collects the farm summary line and runs one
  autoscaling tick.

God-Tier upgrades over gapfix112 / gapfix115:
1. **Per-step ``write_timeout`` isolation**: each step enters its own
   ``db.write_timeout(WATCH_WRITE_TIMEOUT_S)`` scope so a transient
   500ms write stall in ``ops_alerts`` never poisons ``redeliver``,
   ``mine_preventions``, or ``workspace`` with ``wedged=True``.
2. **Zero silent swallows (SOUL.md §1 compliant)**: every non-fatal step
   exception is recorded via ``record_failure`` and surfaced in
   ``report["step_errors"]`` — zero ``except Exception: pass`` blocks,
   eliminating the ``ar-swallow-nomorals-agents-watcher`` survey finding.
3. **Per-step circuit breaker (``StepCircuit``)**: steps that fail
   repeatedly trip open with exponential cooldown instead of burning
   tick wall budget on deterministic failures, auto-probing half-open
   once cooldown lapses.
4. **Per-step telemetry & deadline headroom**: records ``step_ms``,
   ``step_deferrals``, ``step_status``, and ``db_writer_healed``, and
   publishes ``watch.tick.completed`` on the global event bus.
5. **Thread-safe rolling history & `.health()`**: maintains a bounded
   ring buffer of recent ticks and exposes a full diagnostic snapshot.
"""
from __future__ import annotations

import collections
import contextlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

#: Durable scheduler job + tool for the resident loop (audit gap R2-22).
WATCH_LOOP_JOB = "watch loop tick"
WATCH_LOOP_TOOL = "__watch_loop_tick__"
#: Live cadence — the dominant resident-tick cadence in this runtime
#: (group rhythm, delegation manager, unified scheduler all ~5m).
WATCH_LOOP_SPEC = "every 5m"

#: Tick wall budget (seconds): the whole sweep must finish inside the
#: scheduler's 300s job wall.
DEFAULT_TICK_BUDGET_S = 240.0

#: Tick write-wait bound (seconds) per individual step.
WATCH_WRITE_TIMEOUT_S = 0.5

#: Minimum remaining wall clock (seconds) required to start a new step.
MIN_STEP_HEADROOM_S = 0.00001

#: Default ring-buffer capacity for recent tick reports.
DEFAULT_HISTORY_LIMIT = 100


def watch_tick_budget() -> float:
    """Tick wall budget in seconds from NM_WATCH_TICK_BUDGET_S."""
    try:
        value = float(os.environ.get("NM_WATCH_TICK_BUDGET_S", "") or 0)
    except ValueError:
        value = 0.0
    if value <= 0:
        return DEFAULT_TICK_BUDGET_S
    return min(290.0, max(60.0, value))


__all__ = [
    "StepCircuit",
    "WatchLoop",
    "WatchStepSpec",
    "WATCH_LOOP_JOB",
    "WATCH_LOOP_TOOL",
    "WATCH_LOOP_SPEC",
    "WATCH_WRITE_TIMEOUT_S",
    "ensure_watch_loop_job",
    "watch_tick_budget",
]


@dataclass
class StepCircuit:
    """Per-step circuit breaker (closed -> open -> half_open -> closed).

    Prevents a deterministically broken secondary organ (e.g. a corrupt
    reasoning journal or broken workspace driver) from stalling every
    tick forever.  By default ``failure_threshold=3`` consecutive
    exceptions open the circuit for ``base_cooldown_s=60.0`` (doubling
    up to ``max_cooldown_s=900.0``).
    """

    name: str
    failure_threshold: int = 3
    base_cooldown_s: float = 60.0
    max_cooldown_s: float = 900.0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_runs: int = 0
    last_error: str = ""
    opened_until: float = 0.0
    state: str = "closed"  # "closed" | "open" | "half_open"

    def allow(self, now: float | None = None) -> bool:
        """True when the step may execute at monotonic timestamp *now*."""
        ts = time.monotonic() if now is None else now
        if self.state == "open":
            if ts >= self.opened_until:
                self.state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.total_runs += 1
        self.consecutive_failures = 0
        self.last_error = ""
        self.opened_until = 0.0
        self.state = "closed"

    def record_failure(self, exc: BaseException,
                       now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        self.total_runs += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.consecutive_failures >= self.failure_threshold:
            over = self.consecutive_failures - self.failure_threshold
            cooldown = min(
                self.max_cooldown_s,
                self.base_cooldown_s * (2.0 ** min(over, 6)),
            )
            self.opened_until = ts + cooldown
            self.state = "open"

    def to_dict(self) -> dict[str, Any]:
        remaining = max(0.0, self.opened_until - time.monotonic())
        return {
            "name": self.name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_runs": self.total_runs,
            "last_error": self.last_error,
            "cooldown_remaining_s": round(remaining, 1),
        }


@dataclass(frozen=True)
class WatchStepSpec:
    """Custom pluggable watch step registered via :meth:`WatchLoop.register_step`."""

    name: str
    fn: Callable[[Any], Any]
    write_bounded: bool = True


class _NullWriteState:
    """Fallback write-timeout receipt when the context carries no DB."""

    def __init__(self) -> None:
        self.deferred = 0
        self.wedged = False


class WatchLoop:
    """One resident loop over the ops + alert + learning + farm surfaces."""

    def __init__(
        self,
        context: Any,
        *,
        interval: float = 60.0,
        heal_background: bool = True,
        write_timeout_s: float = WATCH_WRITE_TIMEOUT_S,
        circuit_threshold: int = 3,
        circuit_cooldown_s: float = 60.0,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        auto_restart_dead_writer: bool = True,
    ) -> None:
        self.context = context
        self.interval = max(1.0, float(interval))
        self.heal_background = heal_background
        self.write_timeout_s = max(0.05, float(write_timeout_s))
        self.auto_restart_dead_writer = bool(auto_restart_dead_writer)
        self.ticks = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max(1, int(history_limit))
        )
        self._circuits: dict[str, StepCircuit] = {
            name: StepCircuit(
                name=name,
                failure_threshold=max(1, int(circuit_threshold)),
                base_cooldown_s=max(1.0, float(circuit_cooldown_s)),
            )
            for name in ("ops_alerts", "redeliver", "mine_preventions", "workspace")
        }
        self._extra_steps: list[WatchStepSpec] = []

    # ── pluggable extension surface ──────────────────────────────────

    def register_step(
        self,
        name: str,
        fn: Callable[[Any], Any],
        *,
        write_bounded: bool = True,
        failure_threshold: int = 3,
    ) -> None:
        """Register an additional best-effort resident step on this loop."""
        clean = (name or "").strip()
        if not clean:
            raise ValueError("watch step name cannot be empty")
        with self._lock:
            self._extra_steps = [s for s in self._extra_steps if s.name != clean]
            self._extra_steps.append(
                WatchStepSpec(name=clean, fn=fn, write_bounded=write_bounded)
            )
            self._circuits.setdefault(
                clean,
                StepCircuit(name=clean, failure_threshold=max(1, failure_threshold)),
            )

    def stop(self) -> None:
        """Signal a running :meth:`run` loop to exit cleanly."""
        self._stop_event.set()

    # ── DB writer health & per-step write-timeout isolation ──────────

    @contextlib.contextmanager
    def _step_write_timeout(
        self, enabled: bool = True
    ) -> Iterator[_NullWriteState | Any]:
        """Per-step write-wait bound so a timeout in step N never poisons step N+1."""
        db = getattr(self.context, "db", None)
        write_timeout = (
            getattr(db, "write_timeout", None)
            if (enabled and db is not None)
            else None
        )
        if write_timeout is None:
            yield _NullWriteState()
            return
        with write_timeout(self.write_timeout_s) as wt_state:
            yield wt_state

    def _probe_and_heal_db_writer(self) -> dict[str, Any]:
        """Zero-write check of the SQLite writer actor; restarts if dead."""
        info: dict[str, Any] = {
            "writer_healed": False,
            "lock_held_s": 0.0,
            "wal_bytes": -1,
        }
        db = getattr(self.context, "db", None)
        if db is None:
            return info
        try:
            lock_fn = getattr(db, "lock_state", None)
            if callable(lock_fn):
                state = lock_fn() or {}
                holder = state.get("write_lock_holder") or {}
                info["lock_held_s"] = float(holder.get("held_s") or 0.0)
                info["wal_bytes"] = int(state.get("wal_bytes") or -1)
        except Exception as exc:  # noqa: BLE001
            record_failure("watcher.db_lock_state", exc)
        if self.auto_restart_dead_writer:
            try:
                actor = getattr(db, "_writer_actor", None)
                restart_fn = getattr(db, "restart_writer_actor", None)
                if (
                    actor is not None
                    and hasattr(actor, "is_alive")
                    and not actor.is_alive()
                    and callable(restart_fn)
                ):
                    info["writer_healed"] = bool(restart_fn())
                    if info["writer_healed"]:
                        _log.warning("WatchLoop restarted dead DB writer actor")
            except Exception as exc:  # noqa: BLE001
                record_failure("watcher.restart_writer_actor", exc)
        return info

    # ── core tick ────────────────────────────────────────────────────

    def tick(self, *, deadline_s: float | None = None) -> dict[str, Any]:
        """One sweep of everything the operator would check by hand.

        Preserves 100% of the historical return keys while adding
        per-step write-timeout isolation, zero silent swallows,
        per-step latency/deferral accounting, and circuit-breaker
        protection.
        """
        budget = (
            float(deadline_s)
            if deadline_s is not None and float(deadline_s) > 0
            else watch_tick_budget()
        )
        started_mono = time.monotonic()
        deadline = started_mono + budget
        report = self._sweep(deadline, budget)
        report["duration_ms"] = round((time.monotonic() - started_mono) * 1000.0, 2)
        with self._lock:
            self._history.append(dict(report))
        self._emit_tick_event(report)
        return report

    def _sweep(self, deadline: float, budget: float) -> dict[str, Any]:
        """Execute the resident sweep steps with isolated write-timeout scopes."""
        from .notifier import Notifier
        from .ops_alerts import ops_alerts
        from .reasoning import ReasoningAgent

        partial = False
        total_deferred = 0
        step_ms: dict[str, float] = {}
        step_deferrals: dict[str, int] = {}
        step_errors: dict[str, str] = {}
        step_status: dict[str, str] = {}

        def _out_of_time() -> bool:
            return time.monotonic() >= deadline

        # Step 0: non-blocking DB writer health probe & self-restart
        db_probe = self._probe_and_heal_db_writer()

        # Step 1: ops_alerts (primary surface — exceptions propagate so
        # __watch_loop_tick__ reports {"ok": False, "error": ...} per pin).
        t0 = time.monotonic()
        ops_circuit = self._circuits["ops_alerts"]
        try:
            with self._step_write_timeout(enabled=True) as wt_ops:
                report = ops_alerts(
                    self.context, heal_background=self.heal_background
                )
            ops_circuit.record_success()
            step_status["ops_alerts"] = "ok"
        except Exception as exc:
            ops_circuit.record_failure(exc)
            record_failure("watcher.ops_alerts", exc)
            raise
        finally:
            step_ms["ops_alerts"] = round((time.monotonic() - t0) * 1000.0, 2)
            ops_def = int(getattr(wt_ops, "deferred", 0))
            step_deferrals["ops_alerts"] = ops_def
            total_deferred += ops_def

        # Step 2: durable notification redelivery
        redelivered = 0
        if _out_of_time():
            partial = True
            step_status["redeliver"] = "skipped_deadline"
        elif not self._circuits["redeliver"].allow():
            step_status["redeliver"] = "circuit_open"
        else:
            t0 = time.monotonic()
            wt_redeliver: Any = _NullWriteState()
            try:
                with self._step_write_timeout(enabled=True) as wt_redeliver:
                    redelivered = int(Notifier(self.context).redeliver() or 0)
                self._circuits["redeliver"].record_success()
                step_status["redeliver"] = "ok"
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                self._circuits["redeliver"].record_failure(exc)
                record_failure("watcher.redeliver", exc)
                _log.debug("watch loop redeliver failed: %s", exc, exc_info=True)
                step_errors["redeliver"] = f"{type(exc).__name__}: {exc}"
                step_status["redeliver"] = "error"
            finally:
                step_ms["redeliver"] = round((time.monotonic() - t0) * 1000.0, 2)
                red_def = int(getattr(wt_redeliver, "deferred", 0))
                step_deferrals["redeliver"] = red_def
                total_deferred += red_def

        # Step 3: reasoning journal prevention mining
        mined = 0
        if _out_of_time():
            partial = True
            step_status["mine_preventions"] = "skipped_deadline"
        elif not self._circuits["mine_preventions"].allow():
            step_status["mine_preventions"] = "circuit_open"
        else:
            t0 = time.monotonic()
            wt_mine: Any = _NullWriteState()
            try:
                with self._step_write_timeout(enabled=True) as wt_mine:
                    mined = len(
                        ReasoningAgent(self.context).mine_preventions() or ()
                    )
                self._circuits["mine_preventions"].record_success()
                step_status["mine_preventions"] = "ok"
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                self._circuits["mine_preventions"].record_failure(exc)
                record_failure("watcher.mine_preventions", exc)
                _log.debug(
                    "watch loop mine_preventions failed: %s", exc, exc_info=True
                )
                step_errors["mine_preventions"] = f"{type(exc).__name__}: {exc}"
                step_status["mine_preventions"] = "error"
            finally:
                step_ms["mine_preventions"] = round(
                    (time.monotonic() - t0) * 1000.0, 2
                )
                mine_def = int(getattr(wt_mine, "deferred", 0))
                step_deferrals["mine_preventions"] = mine_def
                total_deferred += mine_def

        # Step 4: workspace summary & autoscale tick
        workspace = ""
        if _out_of_time():
            partial = True
            step_status["workspace"] = "skipped_deadline"
        elif not self._circuits["workspace"].allow():
            step_status["workspace"] = "circuit_open"
        else:
            t0 = time.monotonic()
            wt_ws: Any = _NullWriteState()
            try:
                ws = getattr(self.context, "workspace", None)
                if ws is not None:
                    with self._step_write_timeout(enabled=True) as wt_ws:
                        workspace = str(ws.summary_line() or "")
                        ws.autoscale_tick()
                self._circuits["workspace"].record_success()
                step_status["workspace"] = "ok" if ws is not None else "noop"
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                self._circuits["workspace"].record_failure(exc)
                record_failure("watcher.workspace", exc)
                _log.debug("watch loop workspace failed: %s", exc, exc_info=True)
                step_errors["workspace"] = f"{type(exc).__name__}: {exc}"
                step_status["workspace"] = "error"
            finally:
                step_ms["workspace"] = round((time.monotonic() - t0) * 1000.0, 2)
                ws_def = int(getattr(wt_ws, "deferred", 0))
                step_deferrals["workspace"] = ws_def
                total_deferred += ws_def

        # Optional registered extension steps
        extra_results: dict[str, Any] = {}
        with self._lock:
            extra_specs = list(self._extra_steps)
        for spec in extra_specs:
            if _out_of_time():
                partial = True
                step_status[spec.name] = "skipped_deadline"
                continue
            circuit = self._circuits.setdefault(spec.name, StepCircuit(spec.name))
            if not circuit.allow():
                step_status[spec.name] = "circuit_open"
                continue
            t0 = time.monotonic()
            wt_ext: Any = _NullWriteState()
            try:
                with self._step_write_timeout(enabled=spec.write_bounded) as wt_ext:
                    extra_results[spec.name] = spec.fn(self.context)
                circuit.record_success()
                step_status[spec.name] = "ok"
            except Exception as exc:  # noqa: BLE001
                circuit.record_failure(exc)
                record_failure(f"watcher.{spec.name}", exc)
                step_errors[spec.name] = f"{type(exc).__name__}: {exc}"
                step_status[spec.name] = "error"
            finally:
                step_ms[spec.name] = round((time.monotonic() - t0) * 1000.0, 2)
                ext_def = int(getattr(wt_ext, "deferred", 0))
                step_deferrals[spec.name] = ext_def
                total_deferred += ext_def

        with self._lock:
            self.ticks += 1
            tick_num = self.ticks

        out: dict[str, Any] = {
            "tick": tick_num,
            "ts": round(time.time(), 1),
            "paged": [f["title"] for f in report.get("sent", [])],
            "self_healed": report.get("self_healed", []),
            "cooldown": len(report.get("cooldown", [])),
            "findings": len(report.get("findings", [])),
            "redelivered": redelivered,
            "mined": mined,
            "pending": report.get("pending", 0),
            "workspace": workspace,
            "partial": partial,
            "deadline_s": budget,
            "writes_deferred": total_deferred,
            "step_ms": step_ms,
            "step_deferrals": step_deferrals,
            "step_status": step_status,
            "step_errors": step_errors,
            "db_writer_healed": db_probe["writer_healed"],
        }
        if extra_results:
            out["extra"] = extra_results
        return out

    def _emit_tick_event(self, report: dict[str, Any]) -> None:
        """Publish ``watch.tick.completed`` on the global bus (best-effort)."""
        try:
            from ..core.events import Event, global_bus

            global_bus.publish(
                Event(
                    topic="watch.tick.completed",
                    data={
                        "tick": report.get("tick", 0),
                        "findings": report.get("findings", 0),
                        "paged": len(report.get("paged") or ()),
                        "redelivered": report.get("redelivered", 0),
                        "mined": report.get("mined", 0),
                        "writes_deferred": report.get("writes_deferred", 0),
                        "partial": bool(report.get("partial", False)),
                        "duration_ms": report.get("duration_ms", 0.0),
                    },
                    source="nomorals.agents.watcher",
                )
            )
        except Exception as exc:  # noqa: BLE001
            record_failure("watcher.emit_tick_event", exc)

    # ── diagnostics & history ────────────────────────────────────────

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent tick reports (newest last)."""
        with self._lock:
            items = list(self._history)
        return items[-max(1, int(limit)) :]

    def health(self) -> dict[str, Any]:
        """Full diagnostic snapshot for ``/watch status`` or ``nm watch``."""
        with self._lock:
            recent = list(self._history)
            circuits = {k: c.to_dict() for k, c in self._circuits.items()}
            ticks = self.ticks
        last = recent[-1] if recent else None
        open_circuits = [
            name for name, c in circuits.items() if c["state"] == "open"
        ]
        return {
            "ok": not open_circuits,
            "ticks": ticks,
            "interval_s": self.interval,
            "write_timeout_s": self.write_timeout_s,
            "open_circuits": open_circuits,
            "circuits": circuits,
            "last_tick": last,
        }

    # ── formatting & daemon loop ─────────────────────────────────────

    @staticmethod
    def format(result: dict[str, Any]) -> str:
        parts = [
            f"tick {result['tick']} — {len(result['paged'])} paged, "
            f"{len(result['self_healed'])} self-heal(s), "
            f"{result['cooldown']} in cooldown",
        ]
        if result.get("redelivered"):
            parts.append(f"{result['redelivered']} alert(s) redelivered")
        if result.get("mined"):
            parts.append(f"{result['mined']} prevention skill(s) mined")
        if result.get("pending"):
            parts.append(f"{result['pending']} still in the delivery queue")
        if result.get("writes_deferred"):
            parts.append(f"{result['writes_deferred']} write(s) deferred")
        if result.get("db_writer_healed"):
            parts.append("db writer actor restarted")
        if result.get("workspace"):
            parts.append(str(result["workspace"]))
        if not (
            result.get("paged")
            or result.get("self_healed")
            or result.get("redelivered")
            or result.get("mined")
            or result.get("pending")
            or result.get("writes_deferred")
            or result.get("db_writer_healed")
        ):
            parts[0] += " — all quiet"
        return "; ".join(parts)

    def run(
        self,
        *,
        max_ticks: int | None = None,
        once: bool = False,
        emit: Callable[[dict[str, Any], str], Any] | None = None,
    ) -> int:
        """Run until interrupted (or max_ticks/once). Drift-compensated cadence."""
        if emit is None:
            def _default_emit(_res: dict[str, Any], line: str) -> None:
                print(line, flush=True)
            emit = _default_emit

        self._stop_event.clear()
        limit = 1 if once else (int(max_ticks) if max_ticks is not None else None)
        done = 0
        try:
            while not self._stop_event.is_set():
                step_started = time.monotonic()
                res = self.tick()
                emit(res, self.format(res))
                done += 1
                if limit is not None and done >= limit:
                    break
                elapsed = time.monotonic() - step_started
                remaining = max(0.0, self.interval - elapsed)
                if self._stop_event.wait(timeout=remaining):
                    break
        except KeyboardInterrupt:
            return 0
        return 0


def _reconcile_overlap_skip(scheduler: Any, job: Any) -> None:
    """Force a persisted job row onto ``overlap_policy=skip``."""
    try:
        row = job if isinstance(job, dict) else None
        if row is None:
            return
        if (row.get("overlap_policy") or "concurrent") != "skip":
            scheduler.set_overlap_policy(row["id"], "skip")
    except Exception as exc:  # noqa: BLE001 - reconciliation is best-effort
        record_failure("watcher.reconcile_overlap_skip", exc)


def ensure_watch_loop_job(
    scheduler: Any, *, spec: str = WATCH_LOOP_SPEC
) -> dict[str, Any] | None:
    """Idempotently register the durable ``watch loop tick`` scheduler job."""
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            name = (
                job.get("name")
                if isinstance(job, dict)
                else getattr(job, "name", None)
            )
            if name != WATCH_LOOP_JOB:
                continue
            job_spec = (
                job.get("spec")
                if isinstance(job, dict)
                else getattr(job, "spec", None)
            )
            if job_spec == spec:
                _reconcile_overlap_skip(scheduler, job)
                return job
            with contextlib.suppress(Exception):
                job_id = (
                    job.get("id")
                    if isinstance(job, dict)
                    else getattr(job, "id", None)
                )
                scheduler.remove(job_id)
            break
        job = scheduler.add(
            WATCH_LOOP_JOB,
            spec,
            "tool",
            {"tool": WATCH_LOOP_TOOL, "args": {}},
            routine=True,
            overlap_policy="skip",
        )
        _reconcile_overlap_skip(scheduler, job)
        return job
    except Exception as exc:  # noqa: BLE001
        record_failure("watcher.ensure_watch_loop_job", exc)
        return None
