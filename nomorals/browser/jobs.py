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
takeover verb and the loop). The live job objects are in-memory only,
but when the manager is given a database every run also lands in the
durable ``browser_run_records`` table (:class:`BrowserRunRecordStore`,
doc 18 P1): a snapshot is taken at run start and on every status
transition and written in order by the manager's record-writer
thread (a job thread never blocks on the database on its way into
the runner), so a process that dies mid-run leaves a record stuck at
its last state — exactly what the outcome steward's browser adapter
reads. Record writes are evidence, never a gate: a failed write is
recorded via :func:`record_failure` and the run proceeds unchanged.

Cancellation: :meth:`BrowserJobManager.cancel` is cooperative, in
the media jobs idiom. A queued run is stopped before its runner is
ever invoked (confirmed). A running run only gets the request —
runners poll :meth:`BrowserRunJob.cancel_requested` and stop
themselves, returning a ``cancelled: …`` stopped reason; the record
distinguishes ``requested`` from ``confirmed``, and a run that ends
without honoring the request is recorded ``unhonored``, never as a
clean cancel. The production runner (the wired ``AgentLoop``) polls
at its step boundaries and re-checks once more as it closes out;
its result carries a ``cancel_poll_armed`` marker, and when a
request is standing at :meth:`BrowserRunJob.finish` for such a
runner — it landed in the sliver after the loop's final poll — the
close lands the run ``cancelled``/``confirmed`` rather than a
revoked completion. A runner that never polls carries no marker
and keeps the ``unhonored`` reading when it ends on its own terms.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "BrowserJobManager",
    "BrowserRunJob",
    "BrowserRunRecordStore",
]

#: Job statuses. Terminal: done / blocked / failed / cancelled.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_WAITING_OWNER = "waiting_owner"
STATUS_DONE = "done"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_ACTIVE = frozenset({STATUS_QUEUED, STATUS_RUNNING, STATUS_WAITING_OWNER})
_TERMINAL = frozenset({STATUS_DONE, STATUS_BLOCKED, STATUS_FAILED,
                       STATUS_CANCELLED})

#: Cancel lifecycle on the durable record: "" (never requested),
#: "requested" (asked, not yet honored), "confirmed" (the run landed
#: cancelled), "unhonored" (the run ended on its own terms despite
#: the request).
CANCEL_REQUESTED = "requested"
CANCEL_CONFIRMED = "confirmed"
CANCEL_UNHONORED = "unhonored"

#: Query parameter names whose values never reach a durable record.
_SECRET_QUERY_KEYS = ("token", "secret", "password", "passwd", "auth",
                      "session", "sig", "key", "api_key", "apikey",
                      "access_token", "code")

_RECORD_DDL = """
CREATE TABLE IF NOT EXISTS browser_run_records (
  run_id TEXT PRIMARY KEY,
  url TEXT NOT NULL DEFAULT '',
  goal TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT '',
  steps INTEGER NOT NULL DEFAULT 0,
  stopped_reason TEXT NOT NULL DEFAULT '',
  final_url TEXT NOT NULL DEFAULT '',
  wall_outcome TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  cancel_state TEXT NOT NULL DEFAULT '',
  cancel_requested_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0,
  started_at REAL NOT NULL DEFAULT 0,
  finished_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_browser_run_records_status
  ON browser_run_records(status, updated_at);
"""


def _redact_url(url: str) -> str:
    """A record-safe form of a URL: userinfo stripped, secret-shaped
    query values masked. Host + path survive — the record must still
    say WHERE the run went."""
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text[:300]
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    query = ""
    if parts.query:
        pairs = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            lowered = key.lower()
            if any(marker in lowered for marker in _SECRET_QUERY_KEYS):
                value = "***"
            pairs.append((key, value))
        query = urlencode(pairs)
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))[:300]


class BrowserRunRecordStore:
    """Durable per-run records for browser runs (doc 18 P1).

    Runtime-created table over the shared ``Database`` (Batch B
    pattern — DDL in code, migrations untouched). One row per run,
    upserted at run start and on every status transition; the row is
    a snapshot of the job's own state, redacted at the boundary.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECORD_DDL)

    def record(self, snapshot: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO browser_run_records (run_id, url, goal,"
            " status, steps, stopped_reason, final_url, wall_outcome,"
            " error, cancel_state, cancel_requested_at, created_at,"
            " started_at, finished_at, updated_at) VALUES (?, ?, ?,"
            " ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id) DO UPDATE SET url = excluded.url,"
            " goal = excluded.goal, status = excluded.status,"
            " steps = excluded.steps,"
            " stopped_reason = excluded.stopped_reason,"
            " final_url = excluded.final_url,"
            " wall_outcome = excluded.wall_outcome,"
            " error = excluded.error,"
            " cancel_state = excluded.cancel_state,"
            " cancel_requested_at = excluded.cancel_requested_at,"
            " started_at = excluded.started_at,"
            " finished_at = excluded.finished_at,"
            " updated_at = excluded.updated_at",
            (snapshot["run_id"], snapshot["url"], snapshot["goal"],
             snapshot["status"], snapshot["steps"],
             snapshot["stopped_reason"], snapshot["final_url"],
             snapshot["wall_outcome"], snapshot["error"],
             snapshot["cancel_state"], snapshot["cancel_requested_at"],
             snapshot["created_at"], snapshot["started_at"],
             snapshot["finished_at"], snapshot["updated_at"]))

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM browser_run_records WHERE run_id = ?",
            (run_id,))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM browser_run_records"
            " ORDER BY created_at DESC LIMIT ?", (int(limit),))

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
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    #: Cancel lifecycle (see module docstring): "" until requested.
    cancel_state: str = ""
    cancel_requested_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                  repr=False, compare=False)
    _done: threading.Event = field(default_factory=threading.Event,
                                    repr=False, compare=False)
    _notify: Notifier | None = field(default=None, repr=False, compare=False)
    _cancel_requested: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False)
    #: Manager-installed snapshot hook (durable record writes).
    _on_change: Callable[[], None] | None = field(
        default=None, repr=False, compare=False)

    # -- cancellation ---------------------------------------------------------
    def cancel_requested(self) -> bool:
        """Runners poll this: True once a cancel has been requested."""
        return self._cancel_requested.is_set()

    def _changed(self) -> None:
        """Notify the manager a recordable transition just landed."""
        hook = self._on_change
        if hook is not None:
            hook()

    # -- event intake --------------------------------------------------------
    def note(self, kind: str, message: str) -> None:
        """Append an event. Key lifecycle events also ping the owner."""
        with self._lock:
            self.events.append({"kind": kind, "message": message,
                                "at": time.time()})
        if kind in ("wall_hit", "wall_escalated", "wall_cleared",
                    "wall_blocked", "done", "blocked", "failed",
                    "cancelled"):
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
            self._changed()
            # The solver's own escalation message carries the link, but
            # this job-scoped ping is what the owner sees in the run's
            # context — it must be actionable on its own, not "a link
            # was sent somewhere".
            link = str(info.get("takeover_url") or "")
            message = (f"🧱 browser run {self.job_id}: self-solve failed — "
                       "escalation sent, waiting for you to solve it.")
            if link:
                message += f"\nTakeover link: {link}"
            self.note("wall_escalated", message)
        elif kind == "wall_cleared":
            with self._lock:
                self.status = STATUS_RUNNING
                self.wall_outcome = f"cleared via {via}" if via \
                    else "cleared"
            self._changed()
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
            cancelled = self._cancel_requested.is_set() \
                and self.stopped_reason.startswith("cancelled")
            if not cancelled and not blocked \
                    and self._cancel_requested.is_set() \
                    and result.get("cancel_poll_armed"):
                # Close-out re-check: the runner polls
                # cooperatively (its result carries the armed
                # marker), but this request landed after its
                # final poll, in the sliver before this close —
                # the record lands cancelled, never a clean
                # completion the owner had already revoked.
                # Runners that never poll carry no marker and
                # keep the unhonored reading below.
                cancelled = True
                prior = self.stopped_reason
                self.stopped_reason = (
                    "cancelled: request landed before the record "
                    f"closed (loop stopped: {prior})" if prior else
                    "cancelled: request landed before the record "
                    "closed")
            if cancelled:
                self.status = STATUS_CANCELLED
            elif blocked:
                self.status = STATUS_BLOCKED
            else:
                self.status = STATUS_DONE
            if self.cancel_state == CANCEL_REQUESTED:
                self.cancel_state = CANCEL_CONFIRMED if cancelled \
                    else CANCEL_UNHONORED
        # Notify BEFORE signalling done: a waiter that wakes on
        # ``wait()`` must see the owner update already delivered,
        # not racing the send.
        if self.status == STATUS_CANCELLED:
            self.note("cancelled",
                      f"✋ browser run {self.job_id} cancelled: "
                      f"{self.stopped_reason}")
        elif blocked:
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
        self._changed()
        with self._lock:
            self._done.set()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self.error = str(exc)
            self.stopped_reason = f"job failed: {exc}"
            self.finished_at = time.time()
            self.status = STATUS_FAILED
            if self.cancel_state == CANCEL_REQUESTED:
                # The run died on its own terms; the cancel never
                # took effect, and the record must not pretend it did.
                self.cancel_state = CANCEL_UNHONORED
        # Notify before signalling done (see finish()).
        self.note("failed",
                  f"⛔ browser run {self.job_id} failed: {exc}")
        self._changed()
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

    def __init__(self, *, notifier: Notifier | None = None,
                 db: Any = None) -> None:
        self._notifier = notifier
        self._lock = threading.Lock()
        self._jobs: dict[str, BrowserRunJob] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._records: BrowserRunRecordStore | None = None
        if db is not None:
            try:
                self._records = BrowserRunRecordStore(db)
            except Exception as exc:  # noqa: BLE001 - records never gate
                record_failure("browser.jobs.record_store", exc)
        # Record writes run on ONE writer thread fed by a FIFO queue:
        # snapshots are built at each transition (cheap, in-memory)
        # and written in order, so a job thread never blocks on the
        # database on its way into the runner, and transitions can
        # never land out of order.
        import queue as _queue

        self._record_queue: _queue.Queue = _queue.Queue()
        self._record_writer: threading.Thread | None = None
        self._record_writer_lock = threading.Lock()

    # -- durable records ------------------------------------------------------
    def _record_job(self, job: BrowserRunJob) -> None:
        """Snapshot one job for its durable record. Fail-open: the
        record is evidence, not a gate — a failed write is recorded
        and the run proceeds exactly as before."""
        if self._records is None:
            return
        with job._lock:
            snapshot = {
                "run_id": job.job_id,
                "url": _redact_url(job.url),
                "goal": job.goal[:200],
                "status": job.status,
                "steps": job.steps,
                "stopped_reason": job.stopped_reason[:300],
                "final_url": _redact_url(job.final_url),
                "wall_outcome": job.wall_outcome[:200],
                "error": job.error[:300],
                "cancel_state": job.cancel_state,
                "cancel_requested_at": job.cancel_requested_at,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "updated_at": time.time(),
            }
        self._ensure_record_writer()
        self._record_queue.put(snapshot)

    def _ensure_record_writer(self) -> None:
        with self._record_writer_lock:
            if self._record_writer is None:
                writer = threading.Thread(
                    target=self._record_writer_loop,
                    name="browser-run-records", daemon=True)
                writer.start()
                self._record_writer = writer

    def _record_writer_loop(self) -> None:
        store = self._records
        while True:
            item = self._record_queue.get()
            try:
                if isinstance(item, threading.Event):
                    item.set()  # flush barrier
                    continue
                if store is not None:
                    store.record(item)
            except Exception as exc:  # noqa: BLE001 - fail-open
                record_failure("browser.jobs.record", exc)
            finally:
                self._record_queue.task_done()

    def flush_records(self, timeout: float = 5.0) -> bool:
        """Wait until every queued record snapshot has been written
        (tests + orderly shutdown; production never needs to wait —
        the writer drains within milliseconds)."""
        if self._records is None:
            return True
        barrier = threading.Event()
        self._ensure_record_writer()
        self._record_queue.put(barrier)
        return barrier.wait(timeout)

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
            job._on_change = lambda j=job: self._record_job(j)
            self._jobs[job_id] = job
            self._keys[key] = job_id
        self._record_job(job)
        thread = threading.Thread(target=self._run, args=(job, runner),
                                  name=f"browser-run-{job_id}",
                                  daemon=True)
        thread.start()
        return job, False

    def _run(self, job: BrowserRunJob, runner: Runner) -> None:
        if job.cancel_requested():
            # Cancelled while queued: the runner is never invoked,
            # so this confirmation is the manager's own, not a hope.
            with job._lock:
                job.status = STATUS_CANCELLED
                job.stopped_reason = "cancelled before start"
                job.finished_at = time.time()
                if job.cancel_state == CANCEL_REQUESTED:
                    job.cancel_state = CANCEL_CONFIRMED
            job.note("cancelled",
                     f"✋ browser run {job.job_id} cancelled before "
                     "it started.")
            job._changed()
            with job._lock:
                job._done.set()
            return
        try:
            with job._lock:
                job.status = STATUS_RUNNING
                job.started_at = time.time()
            job._changed()
            result = runner(job)
        except BaseException as exc:  # noqa: BLE001 - recorded, not hidden
            _log.warning("browser run job %s failed", job.job_id,
                         exc_info=True)
            job.fail(exc)
            return
        job.finish(result)

    # -- cancellation -----------------------------------------------------------
    def cancel(self, job_id_or_prefix: str) -> dict[str, Any]:
        """Request cancellation of a run. Typed result, never a
        bool: ``state`` is one of ``not_found`` / ``already_terminal``
        / ``requested``. A request is not a confirmation — for a
        running run the durable record carries the final word
        (``confirmed`` when the runner honored it and landed
        cancelled, ``unhonored`` when the run ended on its own
        terms). A queued run's request takes effect before its
        runner is invoked."""
        job = self.get(job_id_or_prefix)
        if job is None:
            return {"ok": False, "state": "not_found",
                    "job_id": str(job_id_or_prefix or "")}
        with job._lock:
            status = job.status
            if status in _TERMINAL:
                return {"ok": True, "state": "already_terminal",
                        "job_id": job.job_id, "status": status,
                        "cancel_state": job.cancel_state}
            job._cancel_requested.set()
            if not job.cancel_state:
                job.cancel_state = CANCEL_REQUESTED
                job.cancel_requested_at = time.time()
            takes_effect = ("before the run starts" if status
                            == STATUS_QUEUED else
                            "when the runner next notices the request")
        job._changed()
        return {"ok": True, "state": "requested",
                "job_id": job.job_id, "status": status,
                "cancel_state": job.cancel_state,
                "takes_effect": takes_effect}

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
