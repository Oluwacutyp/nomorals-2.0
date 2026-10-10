"""Background job model for media editing.

Images edit synchronously (fast). Video ops enqueue a :class:`MediaJob` and
run on a worker thread; :func:`job_status` polls, and completion records the
artifact path plus any error. Artifacts live under an ``edited/`` directory
next to the source — originals are never overwritten.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .images import MediaEditError

_log = get_logger(__name__)


class _JobCancelled(Exception):
    """Internal: the progress callback noticed a cancel request."""


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
    thread-safe status reads. Process-scoped: jobs do not survive restarts
    (the artifact files do; re-submit to redo)."""

    def __init__(self) -> None:
        self._jobs: dict[str, MediaJob] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="media-jobs")
        self._worker.start()

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
                _log.info("media job %s cancelled while queued", job_id)
            elif job.status == "running":
                job._cancel_requested = True  # worker notices at progress
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
                    continue
                job.status = "running"
                job.started_at = time.time()

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
                _log.info("media job %s done: %s", job_id, job.output_ref)
            except _JobCancelled:
                _log.info("media job %s cancelled mid-run", job_id)
                with self._lock:
                    job.status = "cancelled"
                    job.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001 - job failure is data
                _log.warning("media job %s failed: %s", job_id, exc)
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()

    def _set_progress(self, job_id: str, value: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.progress = max(0.0, min(1.0, value))


# Module-level default manager used by the tool layer. Tests may build their
# own JobManager for isolation.
_default_manager: JobManager | None = None
_default_lock = threading.Lock()


def get_manager() -> JobManager:
    global _default_manager
    with _default_lock:
        if _default_manager is None:
            _default_manager = JobManager()
        return _default_manager


def job_status(job_id: str) -> dict[str, Any]:
    return get_manager().status(job_id)
