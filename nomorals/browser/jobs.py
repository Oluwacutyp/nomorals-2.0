"""Background jobs for ``/browser run``.

``/browser run <url> [goal]`` used to block the chat command
synchronously while the loop waited for the owner (up to the 900s
takeover wait). A chat command must never hold the conversation hostage
for a wall: the command now starts a :class:`BrowserRunJob` and returns
immediately with the job id, the loop runs detached on a daemon thread,
and the owner gets Telegram updates on the key events (wall hit ->
escalation, cleared -> resumed, blocked -> stopped, done) plus a
``/browser run status <job>`` view of steps / stopped-reason /
final URL / wall outcome.

Dedupe: starting a run for a URL+goal that already has a live job
returns the existing job instead of a second loop — unless the owner
passes ``force=True`` (``--new`` on the verb), which starts a fresh
job explicitly.

Threading notes: each job owns its thread. Everything the loop touches
(tab, challenge memory, decider) is built INSIDE the worker thread —
the Playwright tab and the sqlite-backed challenge memory are both
thread-bound, so building them on the command thread and using them
on the worker would break. The takeover manager is the exception: it
is lock-guarded and shared (its grants must be visible to both the
takeover verb and the loop). Jobs are in-memory only — they do not
survive a process restart.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "BrowserJobManager",
    "BrowserRunJob",
]

#: Job statuses. Terminal: done / blocked / failed.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_WAITING_OWNER = "waiting_owner"
STATUS_DONE = "done"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"

_ACTIVE = frozenset({STATUS_QUEUED, STATUS_RUNNING, STATUS_WAITING_OWNER})

#: ``runner(job) -> loop result dict``. Runs on the job's thread.
Runner = Callable[["BrowserRunJob"], dict[str, Any]]
#: ``(message, url) -> None`` — the owner-Telegram sender.
Notifier = Callable[[str, str], None]


def _norm_key(url: str, goal: str) -> tuple[str, str]:
    return (str(url or "").strip(),
            " ".join(str(goal or "").split()).casefold())


@dataclass
class BrowserRunJob:
    """One detached ``/browser run``: state, events, and the result."""

    job_id: str
    url: str
    goal: str
    status: str = STATUS_QUEUED
    steps: int = 0
    stopped_reason: str = ""
    final_url: str = ""
    #: Human wall outcome, e.g. "no wall", "cleared via retry",
    #: "cleared via takeover:owner_done", "blocked: takeover:still_blocked".
    wall_outcome: str = "no wall"
    wall_detection: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                  repr=False, compare=False)
    _done: threading.Event = field(default_factory=threading.Event,
                                    repr=False, compare=False)
    _notify: Notifier | None = field(default=None, repr=False, compare=False)

    # -- event intake --------------------------------------------------------
    def note(self, kind: str, message: str) -> None:
        """Append an event. Key lifecycle events also ping the owner."""
        with self._lock:
            self.events.append({"kind": kind, "message": message,
                                "at": time.time()})
        if kind in ("wall_hit", "wall_escalated", "wall_cleared",
                    "wall_blocked", "done", "blocked", "failed"):
            self._send(message)

    def _send(self, message: str) -> None:
        notify = self._notify
        if notify is None:
            return
        try:
            notify(message, self.url)
        except Exception:  # noqa: BLE001 - notify failure is reported
            _log.warning("browser run job notify failed", exc_info=True)

    def wall_event(self, kind: str, info: dict[str, Any] | None = None) -> None:
        """Consume one wall-lifecycle event from the loop's handler.

        Kinds: ``wall_hit`` (a wall appeared), ``wall_escalated``
        (ladder exhausted, owner takeover requested), ``wall_cleared``
        (``info["via"]`` names the winning strategy), ``wall_blocked``
        (still blocked after every rung + takeover).
        """
        info = dict(info or {})
        via = str(info.get("via", "") or "")
        detection = str(info.get("detection", "") or "")
        if kind == "wall_hit":
            with self._lock:
                self.wall_detection = detection or self.wall_detection
                self.wall_outcome = f"wall hit: {detection}" \
                    if detection else "wall hit"
            self.note("wall_hit",
                      f"🧱 browser run {self.job_id}: wall hit "
                      f"({detection or 'challenge'}) on {self.url} — "
                      "working the ladder…")
        elif kind == "wall_escalated":
            with self._lock:
                self.status = STATUS_WAITING_OWNER
            self.note("wall_escalated",
                      f"🧱 browser run {self.job_id}: self-solve failed — "
                      "escalation sent, waiting for you to solve it.")
        elif kind == "wall_cleared":
            with self._lock:
                self.status = STATUS_RUNNING
                self.wall_outcome = f"cleared via {via}" if via \
                    else "cleared"
            self.note("wall_cleared",
                      f"✅ browser run {self.job_id}: wall cleared via "
                      f"{via or 'unknown'} — resuming.")
        elif kind == "wall_blocked":
            with self._lock:
                self.wall_outcome = f"blocked: {via}" if via else "blocked"
            # No ping here: finish() pings "blocked -> stopped" once the
            # loop actually stops, so the owner gets one stopped update,
            # not two.
            with self._lock:
                self.events.append({"kind": "wall_blocked",
                                    "message": self.wall_outcome,
                                    "at": time.time()})

    def note_progress(self, progress: dict[str, Any]) -> None:
        """Live step count from the loop's ``on_progress`` hook."""
        try:
            step = int(progress.get("step", 0) or 0)
        except (TypeError, ValueError):
            step = 0
        with self._lock:
            self.steps = max(self.steps, step)
            if progress.get("url"):
                self.final_url = str(progress["url"])

    # -- terminal --------------------------------------------------------------
    def finish(self, result: dict[str, Any] | None) -> None:
        result = dict(result or {})
        with self._lock:
            self.steps = int(result.get("steps", self.steps) or 0)
            self.stopped_reason = str(result.get("stopped_reason", "") or "")
            self.final_url = str(result.get("final_url", "") or self.final_url)
            trace = result.get("trace")
            if isinstance(trace, list):
                self.trace = trace
            self.finished_at = time.time()
            blocked = self.stopped_reason.startswith("blocked:")
            self.status = STATUS_BLOCKED if blocked else STATUS_DONE
        # Notify BEFORE signalling done: a waiter that wakes on
        # ``wait()`` must see the owner update already delivered,
        # not racing the send.
        if blocked:
            self.note("blocked",
                      f"⛔ browser run {self.job_id} stopped: "
                      f"{self.stopped_reason} — final URL: "
                      f"{self.final_url or self.url}")
        else:
            self.note("done",
                      f"🏁 browser run {self.job_id} done: "
                      f"{self.steps} steps, stopped: "
                      f"{self.stopped_reason}, final URL: "
                      f"{self.final_url or self.url}, "
                      f"wall: {self.wall_outcome}")
        with self._lock:
            self._done.set()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self.error = str(exc)
            self.stopped_reason = f"job failed: {exc}"
            self.finished_at = time.time()
            self.status = STATUS_FAILED
        # Notify before signalling done (see finish()).
        self.note("failed",
                  f"⛔ browser run {self.job_id} failed: {exc}")
        with self._lock:
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """True when the job reached a terminal status."""
        return self._done.wait(timeout)

    # -- views ------------------------------------------------------------------
    def status_text(self) -> str:
        with self._lock:
            lines = [
                f"browser run {self.job_id} [{self.status}]",
                f"url: {self.url}",
                f"goal: {self.goal}",
                f"steps: {self.steps}",
                f"stopped: {self.stopped_reason or '(still running)'}",
                f"final URL: {self.final_url or self.url}",
                f"wall: {self.wall_outcome}",
            ]
            recent = self.events[-5:]
        if recent:
            lines.append("events:")
            for event in recent:
                lines.append(f"  {event.get('kind')}: "
                             f"{event.get('message', '')}")
        return "\n".join(lines)


class BrowserJobManager:
    """Registry + threads for browser-run jobs. One per runtime."""

    def __init__(self, *, notifier: Notifier | None = None) -> None:
        self._notifier = notifier
        self._lock = threading.Lock()
        self._jobs: dict[str, BrowserRunJob] = {}
        self._keys: dict[tuple[str, str], str] = {}

    def start(self, url: str, goal: str, runner: Runner, *,
              force: bool = False) -> tuple[BrowserRunJob, bool]:
        """Start a job on a daemon thread.

        Returns ``(job, duplicate)`` — ``duplicate`` is True when an
        already-live job for the same URL+goal was returned instead of
        starting a second loop.
        """
        key = _norm_key(url, goal)
        with self._lock:
            if not force:
                existing_id = self._keys.get(key)
                existing = self._jobs.get(existing_id or "")
                if existing is not None and existing.status in _ACTIVE:
                    return existing, True
            job_id = f"br-{uuid.uuid4().hex[:12]}"
            job = BrowserRunJob(job_id=job_id, url=url.strip(),
                                goal=" ".join(str(goal or "").split()),
                                final_url=url.strip(),
                                _notify=self._notifier)
            self._jobs[job_id] = job
            self._keys[key] = job_id
        thread = threading.Thread(target=self._run, args=(job, runner),
                                  name=f"browser-run-{job_id}",
                                  daemon=True)
        thread.start()
        return job, False

    def _run(self, job: BrowserRunJob, runner: Runner) -> None:
        try:
            with job._lock:
                job.status = STATUS_RUNNING
            result = runner(job)
        except BaseException as exc:  # noqa: BLE001 - recorded, not hidden
            _log.warning("browser run job %s failed", job.job_id,
                         exc_info=True)
            job.fail(exc)
            return
        job.finish(result)

    def get(self, job_id_or_prefix: str) -> BrowserRunJob | None:
        """Exact id, or a unique prefix (``run status br-abc`` works)."""
        needle = (job_id_or_prefix or "").strip()
        if not needle:
            return None
        with self._lock:
            if needle in self._jobs:
                return self._jobs[needle]
            matches = [job for jid, job in self._jobs.items()
                       if jid.startswith(needle)]
        return matches[0] if len(matches) == 1 else None

    def list_recent(self, limit: int = 5) -> list[BrowserRunJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.created_at, reverse=True)
        return jobs[:max(1, limit)]

    def status_text(self, job_id_or_prefix: str) -> str:
        job = self.get(job_id_or_prefix)
        if job is None:
            return (f"no browser run job {job_id_or_prefix!r} — "
                    "/browser run status lists recent jobs.")
        return job.status_text()
