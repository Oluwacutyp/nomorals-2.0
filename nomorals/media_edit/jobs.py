"""Background job model for media editing.

Images edit synchronously (fast). Video ops enqueue a :class:`MediaJob` and
run on a worker thread; :func:`job_status` polls, and completion records the
artifact path plus any error. Artifacts live under an ``edited/`` directory
next to the source — originals are never overwritten.

When the manager is given a database, every job also lands in the
durable ``media_job_records`` table (:class:`MediaJobRecordStore`,
doc 18 P1): written at submit and on every status transition, so a
process that dies mid-render leaves a record stuck at its last
state — exactly what the outcome steward's render adapter reads.
Record writes are evidence, never a gate: a failed write is recorded
via :func:`record_failure` and the job proceeds unchanged.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.failures import record_failure
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .images import MediaEditError

_log = get_logger(__name__)


class _JobCancelled(Exception):
    """Internal: the progress callback noticed a cancel request."""


_RECORD_DDL = """
CREATE TABLE IF NOT EXISTS media_job_records (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT '',
  label TEXT NOT NULL DEFAULT '',
  backend TEXT NOT NULL DEFAULT '',
  input_ref TEXT NOT NULL DEFAULT '',
  output_ref TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  cancel_requested_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0,
  started_at REAL NOT NULL DEFAULT 0,
  finished_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_media_job_records_status
  ON media_job_records(status, updated_at);
"""


class MediaJobRecordStore:
    """Durable per-job records for media edit/render jobs (doc 18 P1).

    Runtime-created table over the shared ``Database`` (Batch B
    pattern — DDL in code, migrations untouched). One row per job,
    upserted at submit and on every status transition; the row
    carries the job's identity + outcome summary only — never the
    raw result payload.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECORD_DDL)

    def record(self, snapshot: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO media_job_records (job_id, kind, label,"
            " backend, input_ref, output_ref, status, error,"
            " cancel_requested_at, created_at, started_at,"
            " finished_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(job_id) DO UPDATE SET kind = excluded.kind,"
            " label = excluded.label, backend = excluded.backend,"
            " input_ref = excluded.input_ref,"
            " output_ref = excluded.output_ref,"
            " status = excluded.status, error = excluded.error,"
            " cancel_requested_at = excluded.cancel_requested_at,"
            " started_at = excluded.started_at,"
            " finished_at = excluded.finished_at,"
            " updated_at = excluded.updated_at",
            (snapshot["job_id"], snapshot["kind"], snapshot["label"],
             snapshot["backend"], snapshot["input_ref"],
             snapshot["output_ref"], snapshot["status"],
             snapshot["error"], snapshot["cancel_requested_at"],
             snapshot["created_at"], snapshot["started_at"],
             snapshot["finished_at"], snapshot["updated_at"]))

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM media_job_records WHERE job_id = ?",
            (job_id,))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM media_job_records"
            " ORDER BY created_at DESC LIMIT ?", (int(limit),))


@dataclass
class MediaJob:
    id: str
    kind: str  # "image" | "video"
    label: str
    status: str = "queued"  # queued|running|done|failed|cancelled
    progress: float | None = None  # 0.0..1.0 while running
    input_ref: str = ""
    output_ref: str = ""
    backend: str = ""  # "" = current behavior; "comfy"|"auto" = gen routing hint
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "progress": self.progress,
            "input_ref": self.input_ref,
            "output_ref": self.output_ref,
            "backend": self.backend,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": (round((self.finished_at or time.time())
                              - (self.started_at or self.created_at), 2)
                        if self.status in ("running", "done", "failed") else None),
        }


class JobManager:
    """Tiny in-process background runner. One worker thread, FIFO queue,
    thread-safe status reads. Process-scoped: the live job objects do
    not survive restarts (the artifact files do; re-submit to redo) —
    but with a database attached, each job's durable record does
    (:class:`MediaJobRecordStore`)."""

    def __init__(self, db: Any = None) -> None:
        self._jobs: dict[str, MediaJob] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._records: MediaJobRecordStore | None = None
        if db is not None:
            self.attach_db(db)
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="media-jobs")
        self._worker.start()

    def attach_db(self, db: Any) -> None:
        """Attach the durable record store (first db wins). Fail-open:
        a store that cannot be built is recorded, never raised."""
        if db is None or self._records is not None:
            return
        try:
            self._records = MediaJobRecordStore(db)
        except Exception as exc:  # noqa: BLE001 - records never gate
            record_failure("media_edit.jobs.record_store", exc)

    # -- durable records ------------------------------------------------------
    def _record(self, job: MediaJob) -> None:
        """Snapshot one job into its durable record. Callers hold
        ``self._lock`` (the worker's terminal writes do; submit and
        cancel build the snapshot under it too). Fail-open by design."""
        store = self._records
        if store is None:
            return
        try:
            store.record({
                "job_id": job.id,
                "kind": job.kind,
                "label": job.label[:200],
                "backend": job.backend,
                "input_ref": job.input_ref[:300],
                "output_ref": job.output_ref[:300],
                "status": job.status,
                "error": job.error[:300],
                "cancel_requested_at": float(
                    getattr(job, "_cancel_requested_at", 0.0) or 0.0),
                "created_at": job.created_at,
                "started_at": job.started_at or 0.0,
                "finished_at": job.finished_at or 0.0,
                "updated_at": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            record_failure("media_edit.jobs.record", exc)

    # -- public API ---------------------------------------------------------
    def submit(self, kind: str, label: str,
               fn: Callable[[Callable[[float], None]], dict[str, Any]],
               *, input_ref: str = "", backend: str = "") -> str:
        """Enqueue ``fn``. ``fn`` receives a progress callback and returns the
        op result dict (must include an ``output`` path on success).
        ``backend`` is a routing hint recorded on the job ("" = current
        behavior)."""
        job = MediaJob(id=new_id("mjob"), kind=kind, label=label,
                       input_ref=input_ref, backend=backend)
        job._fn = fn  # worker picks this up; not part of the dataclass state
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._record(job)
            self._cond.notify()
        _log.info("media job %s queued: %s", job.id, label)
        return job.id

    def submit_gen(self, label: str, op: str, params: dict[str, Any],
                   *, input_ref: str = "",
                   backend: str = "auto") -> str:
        """Enqueue a generative image job.

        ``op``: "txt2img" | "img2img" | "inpaint" | "generative_edit".
        ``params`` are the op's kwargs (prompt, image, ...). The backend is
        resolved at *run* time: "auto" picks ComfyUI when
        :func:`nomorals.media_edit.comfy.comfy_available` says so, else
        falls back to :func:`get_backend` (existing behavior — the ffmpeg /
        Pillow worker path is untouched). The chosen backend is recorded in
        the result dict; artifacts land in ``edited/`` next to the input
        (or ``./edited/`` when there is no input file).
        """
        from .generate import (op_generative_edit, op_img2img, op_inpaint,
                               op_txt2img)
        op_fns = {
            "txt2img": op_txt2img,
            "img2img": op_img2img,
            "inpaint": op_inpaint,
            "generative_edit": op_generative_edit,
        }
        op_fn = op_fns.get(op)
        if op_fn is None:
            raise MediaEditError(
                f"unknown gen op {op!r}; use one of {sorted(op_fns)}")
        cell: dict[str, str] = {}

        def _run(progress_cb: Callable[[float], None]) -> dict[str, Any]:
            from .comfy import comfy_available
            chosen = backend
            if chosen == "auto":
                ok, _reason = comfy_available()
                chosen = "comfy" if ok else ""
            p = dict(params)
            p["backend"] = chosen or None
            result = op_fn(**p)
            images = result if isinstance(result, list) else [result]
            from pathlib import Path as _Path
            from .images import save_image
            base = _Path(input_ref).parent if input_ref else _Path(".")
            out_dir = base / "edited"
            paths: list[str] = []
            for i, img in enumerate(images):
                out = out_dir / f"gen_{cell['id']}_{i}.png"
                save_image(img, out)
                paths.append(str(out))
            return {"output": paths[0] if len(paths) == 1 else paths,
                    "backend": chosen or "default",
                    "op": op}

        job_id = self.submit("image", label, _run, input_ref=input_ref,
                             backend=backend)
        cell["id"] = job_id
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise MediaEditError(f"unknown media job {job_id!r}")
            return job.as_dict()

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.created_at, reverse=True)
            return [j.as_dict() for j in jobs[:limit]]

    def wait(self, job_id: str, timeout: float = 900.0) -> dict[str, Any]:
        """Block until the job finishes (for CLI/tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self.status(job_id)
            if info["status"] in ("done", "failed", "cancelled"):
                return info
            time.sleep(0.25)
        raise MediaEditError(f"timed out waiting for job {job_id}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job.

        Queued jobs are dropped before they start. Running jobs are
        asked to stop at their next progress report (cooperative — a job
        that never reports progress can't be interrupted mid-flight).
        Returns the job's status dict.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise MediaEditError(f"unknown media job {job_id!r}")
            if job.status == "queued":
                if job_id in self._queue:
                    self._queue.remove(job_id)
                job.status = "cancelled"
                job.finished_at = time.time()
                self._record(job)
                _log.info("media job %s cancelled while queued", job_id)
            elif job.status == "running":
                job._cancel_requested = True  # worker notices at progress
                job._cancel_requested_at = time.time()
                self._record(job)
                _log.info("media job %s cancel requested (running)",
                          job_id)
            # done/failed/cancelled: no-op, still report
            return job.as_dict()

    def retry(self, job_id: str) -> str:
        """Re-queue a failed/cancelled/done job with the same function.

        Returns the (same) job id. The job goes back to "queued" with
        progress/error cleared.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise MediaEditError(f"unknown media job {job_id!r}")
            if job.status not in ("failed", "cancelled", "done"):
                raise MediaEditError(
                    f"can only retry a finished job, {job_id} is "
                    f"{job.status}")
            fn = getattr(job, "_fn", None)
            if fn is None:
                raise MediaEditError(
                    f"job {job_id} has no function to retry")
            job.status = "queued"
            job.progress = None
            job.error = ""
            job.result = {}
            job.output_ref = ""
            job.started_at = None
            job.finished_at = None
            if hasattr(job, "_cancel_requested"):
                delattr(job, "_cancel_requested")
            if hasattr(job, "_cancel_requested_at"):
                delattr(job, "_cancel_requested_at")
            self._record(job)
            self._queue.append(job_id)
            self._cond.notify()
            _log.info("media job %s retried", job_id)
            return job_id

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._cond.wait()
                job_id = self._queue.pop(0)
                job = self._jobs[job_id]
                if getattr(job, "_cancel_requested", False):
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._record(job)
                    continue
                job.status = "running"
                job.started_at = time.time()
                self._record(job)

            def _progress(p: float, _job: MediaJob = job,
                          _jid: str = job_id) -> None:
                if getattr(_job, "_cancel_requested", False):
                    raise _JobCancelled(f"job {_jid} cancelled")
                self._set_progress(_jid, p)

            fn = getattr(job, "_fn", None)
            try:
                assert fn is not None
                result = fn(_progress)
                with self._lock:
                    job.status = "done"
                    job.progress = 1.0
                    job.result = result or {}
                    job.output_ref = str((result or {}).get("output", ""))
                    job.finished_at = time.time()
                    self._record(job)
                _log.info("media job %s done: %s", job_id, job.output_ref)
            except _JobCancelled:
                _log.info("media job %s cancelled mid-run", job_id)
                with self._lock:
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._record(job)
            except Exception as exc:  # noqa: BLE001 - job failure is data
                _log.warning("media job %s failed: %s", job_id, exc)
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()
                    self._record(job)

    def _set_progress(self, job_id: str, value: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.progress = max(0.0, min(1.0, value))


# Module-level default manager used by the tool layer. Tests may build their
# own JobManager for isolation.
_default_manager: JobManager | None = None
_default_lock = threading.Lock()


def get_manager(db: Any = None) -> JobManager:
    """The process-wide default manager. Passing a ``db`` attaches
    the durable record store (first db wins) — the tool layer passes
    its context database so records land in the shared store the
    outcome steward reads."""
    global _default_manager
    with _default_lock:
        if _default_manager is None:
            _default_manager = JobManager(db=db)
        elif db is not None:
            _default_manager.attach_db(db)
        return _default_manager


def job_status(job_id: str) -> dict[str, Any]:
    return get_manager().status(job_id)
