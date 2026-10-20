"""Background job execution with heartbeats — the replacement for blunt timeouts.

A tool step runs as a :class:`Job` on a daemon worker thread. A second
daemon thread — the heartbeat — emits ``tool.heartbeat`` bus events while
the worker is alive, so "slow but alive" is distinguishable from dead.
The caller never blocks on a slow job: it parks the job handle, moves on
to other work, and collects the result when it completes (or on a later
poll).

The ONLY timer left is the dead-man's switch: a job whose heartbeat goes
silent for ``MAX_MISSED_HEARTBEATS`` intervals is presumed dead — an
honest ``outcome_unknown`` ("no heartbeat for Xs; may still be running;
check state before retrying"), never a fake failure. A late result is
still collected and recorded, never presented as a fresh success.

Threads cannot be killed, so a presumed-dead job's worker may still be
running; that is reported honestly instead of hidden.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

#: How often a running job's heartbeat thread reports liveness.
HEARTBEAT_INTERVAL_S = 10.0
#: Missed heartbeat intervals before a silent job is presumed dead.
#: 5 × 10s = ~50s of silence — generous on purpose; this is a
#: lost-visibility backstop, not a performance knob.
MAX_MISSED_HEARTBEATS = 5
#: Terminal jobs older than this are swept from the registry (debugging
#: visibility without unbounded growth).
TERMINAL_JOB_TTL_S = 1800.0

TERMINAL_STATUSES = frozenset({"done", "failed", "presumed-dead"})

#: The job currently executing on this thread (set by the worker
#: wrapper). Tools call :func:`report_progress` to attach a note.
_current_job: ContextVar[Optional["Job"]] = ContextVar(
    "nomorals_current_job", default=None)


def report_progress(note: str) -> None:
    """Attach a progress note to the currently running job (if any).

    Tools call this voluntarily; a progress report also counts as a
    heartbeat. Never raises — a tool without a job context is unaffected.
    """
    job = _current_job.get()
    if job is None:
        return
    try:
        with job.lock:
            job.last_progress = str(note)[:300]
            job.last_heartbeat_at = time.time()
    except Exception:  # noqa: BLE001 - progress is best-effort
        pass


def current_job() -> Optional["Job"]:
    """The job running on this thread, or None."""
    return _current_job.get()


@dataclass
class Job:
    """One background tool execution."""

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending|running|done|failed|presumed-dead
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    result: Any = None
    error: str = ""
    outcome_unknown: bool = False
    last_heartbeat_at: float = field(default_factory=time.time)
    heartbeat_count: int = 0
    last_progress: str = ""
    #: True when the worker finished AFTER the job was presumed dead.
    late_result: bool = False
    collected: bool = False
    #: Cooperative cancellation: set by :meth:`JobRegistry.cancel`.
    #: Threads cannot be killed, so workers poll this flag at their
    #: progress points and stop themselves; a job that never checks
    #: the flag is reported as still running, never as cancelled.
    cancel_requested: bool = False
    #: Heartbeat cadence for this job; the dead-man's switch fires after
    #: ``heartbeat_interval_s * MAX_MISSED_HEARTBEATS`` of silence.
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S
    lock: threading.RLock = field(default_factory=threading.RLock,
                                  repr=False, compare=False)

    def done(self) -> bool:
        """True when the job reached a terminal status."""
        with self.lock:
            return self.status in TERMINAL_STATUSES

    @property
    def elapsed(self) -> float:
        end = self.ended_at if self.ended_at else time.time()
        return max(0.0, end - self.started_at)

    def snapshot(self) -> dict[str, Any]:
        """Visible state for debugging — nothing about a job is hidden."""
        with self.lock:
            return {
                "id": self.id,
                "tool": self.tool,
                "status": self.status,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_s": round(self.elapsed, 3),
                "error": self.error,
                "outcome_unknown": self.outcome_unknown,
                "last_heartbeat_at": self.last_heartbeat_at,
                "heartbeat_count": self.heartbeat_count,
                "heartbeat_interval_s": self.heartbeat_interval_s,
                "last_progress": self.last_progress,
                "late_result": self.late_result,
                "collected": self.collected,
                "cancel_requested": self.cancel_requested,
                "labels": dict(self.labels),
            }


def _publish(topic: str, data: dict[str, Any]) -> None:
    """Best-effort bus publish — a broken bus never breaks job execution."""
    try:
        from .events import Event, global_bus

        global_bus.publish(Event(topic=topic, data=data,
                                 source="nomorals.core.jobs"))
    except Exception:  # noqa: BLE001 - events never break jobs
        _log.debug("job bus publish failed", exc_info=True)


class JobRegistry:
    """In-memory registry of background jobs. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        # job_id -> (worker thread, heartbeat thread, stop event)
        self._threads: dict[str, tuple[threading.Thread,
                                       threading.Thread,
                                       threading.Event]] = {}

    # ── submission ────────────────────────────────────────────────

    def submit(self, fn: Callable[[], Any], *, tool: str = "",
               args: dict[str, Any] | None = None,
               labels: dict[str, str] | None = None,
               heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S) -> Job:
        """Start ``fn`` as a background job. Returns immediately."""
        from .ids import new_short_id

        job = Job(id=new_short_id("job"), tool=tool,
                  args=dict(args or {}), labels=dict(labels or {}),
                  status="running")
        interval = max(1.0, float(heartbeat_interval_s))
        job.heartbeat_interval_s = interval
        stop = threading.Event()
        submit_ctx = contextvars.copy_context()

        def _run_with_job() -> Any:
            # Set inside the copied context so report_progress() sees it.
            token = _current_job.set(job)
            try:
                return fn()
            finally:
                _current_job.reset(token)

        def _worker() -> None:
            try:
                outcome: tuple[str, Any] = (
                    "ok", submit_ctx.run(_run_with_job))
            except Exception as exc:  # noqa: BLE001 - failure is data
                outcome = ("err", exc)
            with job.lock:
                kind, val = outcome
                if job.status == "presumed-dead":
                    # Late finish after the dead-man's switch fired: record
                    # the result honestly, never overwrite the verdict or
                    # present it as a fresh success.
                    job.late_result = True
                    if kind == "ok":
                        job.result = val
                    else:
                        job.error = (f"late {type(val).__name__}: {val}")[:500]
                elif job.status == "running":
                    if kind == "ok":
                        job.status = "done"
                        job.result = val
                    else:
                        job.status = "failed"
                        job.error = f"{type(val).__name__}: {val}"[:500]
                job.ended_at = time.time()
            stop.set()
            _publish("tool.job", {"job_id": job.id, "tool": job.tool,
                                  "phase": job.status,
                                  "late_result": job.late_result,
                                  "elapsed_s": round(job.elapsed, 3)})

        def _heartbeat(worker: threading.Thread) -> None:
            while not stop.wait(interval):
                if not worker.is_alive():
                    break
                with job.lock:
                    if job.status != "running":
                        break
                    job.last_heartbeat_at = time.time()
                    job.heartbeat_count += 1
                    progress = job.last_progress
                    count = job.heartbeat_count
                _publish("tool.heartbeat", {
                    "job_id": job.id, "tool": job.tool, "alive": True,
                    "progress": progress, "heartbeat": count,
                    "elapsed_s": round(time.time() - job.started_at, 1),
                })

        worker = threading.Thread(target=_worker,
                                  name=f"job-{job.id}", daemon=True)
        hb = threading.Thread(target=_heartbeat, args=(worker,),
                              name=f"job-{job.id}-hb", daemon=True)
        with self._lock:
            self._jobs[job.id] = job
            self._threads[job.id] = (worker, hb, stop)
        worker.start()
        hb.start()
        _publish("tool.job", {"job_id": job.id, "tool": job.tool,
                              "phase": "submitted"})
        return job

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation of a live job.

        Sets ``cancel_requested`` on the job; workers that poll the
        flag at progress points (the download backends do, via
        :func:`current_job`) stop themselves and finish as failed
        with a cancellation error.  Returns False for unknown or
        already-terminal jobs — a finished job cannot be cancelled
        and this never pretends otherwise.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        with job.lock:
            if job.status in TERMINAL_STATUSES:
                return False
            job.cancel_requested = True
        _publish("tool.job", {"job_id": job.id, "tool": job.tool,
                              "phase": "cancel-requested"})
        return True

    # ── dead-man's switch ─────────────────────────────────────────

    def _evaluate(self, job: Job, now: float) -> None:
        """Presume a silent running job dead. Honest UNKNOWN, never failure.

        Silence means the heartbeat stopped arriving — lost visibility
        into the job — not "the tool is slow". A slow job that keeps
        heartbeating runs as long as it needs.
        """
        with job.lock:
            if job.status != "running":
                return
            silent = now - job.last_heartbeat_at
            limit = job.heartbeat_interval_s * MAX_MISSED_HEARTBEATS
            if silent < limit:
                return
            job.status = "presumed-dead"
            job.outcome_unknown = True
            job.error = (
                f"no heartbeat for {silent:.0f}s "
                f"({MAX_MISSED_HEARTBEATS} missed intervals) — outcome "
                f"UNKNOWN (outcome_unknown): the job may still be running; "
                f"check state before retrying, never re-issue blindly")
        _log.warning("job %s (%s) presumed dead after %.0fs without heartbeat",
                     job.id, job.tool, silent)
        _publish("tool.job", {"job_id": job.id, "tool": job.tool,
                              "phase": "presumed-dead",
                              "silent_s": round(silent, 1)})

    def reap(self) -> int:
        """Evaluate the dead-man's switch on all running jobs."""
        now = time.time()
        flagged = 0
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            before = job.status
            self._evaluate(job, now)
            if job.status != before:
                flagged += 1
        return flagged

    # ── collection ────────────────────────────────────────────────

    def poll(self, job_id: str) -> Job | None:
        """Current state of one job (evaluates the dead-man's switch)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        self._evaluate(job, time.time())
        return job

    def collect(self, job_id: str,
                timeout: float | None = None) -> Job | None:
        """Wait for one job up to ``timeout`` seconds (None = forever).

        Returns the job in whatever state it is in when the wait ends —
        callers check ``job.done()`` / ``job.status``. Never raises for a
        missing job (returns None).
        """
        job = self.poll(job_id)
        if job is None or job.done():
            return job
        with self._lock:
            threads = self._threads.get(job_id)
        worker = threads[0] if threads else None
        if worker is not None:
            worker.join(timeout if timeout is None else max(0.0, timeout))
        job = self.poll(job_id)
        return job

    def collect_ready(self, labels: dict[str, str] | None = None
                      ) -> list[Job]:
        """Non-blocking harvest of uncollected terminal jobs.

        ``labels`` filters (e.g. ``{"owner": "devon"}``). Harvested jobs
        are marked collected so a later poll never returns them twice.
        """
        self.reap()
        out: list[Job] = []
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job.lock:
                if not job.done() or job.collected:
                    continue
                if labels and any(job.labels.get(k) != v
                                 for k, v in labels.items()):
                    continue
                job.collected = True
            out.append(job)
        return out

    # ── visibility ────────────────────────────────────────────────

    def list(self, status: str | None = None,
             labels: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Every job's visible state — nothing disappears silently."""
        self.reap()
        with self._lock:
            jobs = list(self._jobs.values())
        out = []
        for job in jobs:
            snap = job.snapshot()
            if status and snap["status"] != status:
                continue
            if labels and any(snap["labels"].get(k) != v
                             for k, v in labels.items()):
                continue
            out.append(snap)
        return out

    def sweep(self) -> int:
        """Drop terminal jobs older than TERMINAL_JOB_TTL_S. Returns count."""
        now = time.time()
        dropped = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                with job.lock:
                    terminal = job.status in TERMINAL_STATUSES
                    age = now - (job.ended_at or job.started_at)
                if terminal and age > TERMINAL_JOB_TTL_S:
                    del self._jobs[job_id]
                    self._threads.pop(job_id, None)
                    dropped += 1
        return dropped


_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def shared_registry() -> JobRegistry:
    """Process-wide job registry — one visible place for all parked jobs."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry
