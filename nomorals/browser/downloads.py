"""Browser download queue, wired to the existing downloader stack.

Jobs queue up, run through the real downloader, resume after interruption,
and get integrity-checked. The transport priority is honest and visible
in every job's ``backend`` field:

1. ``nomorals.media.download_kit.ResumeDownloader`` when importable —
   Range resume with ETag/Last-Modified validator checks (a changed file
   restarts from byte 0, never splices) plus a sha256 of the result.
2. ``urllib`` fallback with the same validator discipline when the kit
   is not importable — implemented here, not claimed to be the kit.

Integrity: when a job carries ``expected_sha256``, the finished file's
hash must match or the job fails with ``integrity_mismatch`` — a
mismatched file is never reported as done. Every completed job is also
appended to the kit's ``IntegrityLedger`` when available.

A custom ``fetcher`` callable (``(job_dict) -> dict``) may be injected —
tests use it; production uses the downloader wiring above. The queue
persists to JSON so jobs survive a restart.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.ids import ulid_now

__all__ = [
    "DownloadError",
    "DownloadJob",
    "DownloadQueue",
    "sha256_file",
    "JOB_QUEUED",
    "JOB_RUNNING",
    "JOB_DONE",
    "JOB_FAILED",
    "JOB_PAUSED",
]

_log = get_logger(__name__)

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_PAUSED = "paused"

#: Backends the queue can honestly report.
BACKEND_KIT = "download_kit.ResumeDownloader"
BACKEND_FALLBACK = "urllib-fallback"


class DownloadError(ValueError):
    """A download job failed (fetch error, integrity mismatch, bad job)."""


def sha256_file(path: str | Path) -> str:
    """sha256 hex of a file, streamed (never loads the whole file)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _try_import_kit() -> tuple[Any | None, Any | None]:
    """(ResumeDownloader class, IntegrityLedger class) or (None, None).

    Import failure is a normal, reported condition — never a crash.
    """
    try:
        from ..media.download_kit import IntegrityLedger, ResumeDownloader
    except Exception as exc:  # noqa: BLE001 - optional dependency
        _log.debug("download queue: download_kit unavailable: %s", exc)
        return None, None
    return ResumeDownloader, IntegrityLedger


@dataclass
class DownloadJob:
    """One queued download."""

    job_id: str = ""
    url: str = ""
    dest: str = ""
    status: str = JOB_QUEUED
    bytes_downloaded: int = 0
    total_bytes: int = 0
    sha256: str = ""
    expected_sha256: str = ""
    error: str = ""
    attempts: int = 0
    backend: str = ""
    resumed: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not (self.url or "").strip():
            errs.append("url is required")
        elif not self.url.startswith(("http://", "https://")):
            errs.append(f"url must be http(s), got {self.url!r}")
        if not (self.dest or "").strip():
            errs.append("dest is required")
        if self.expected_sha256 and len(self.expected_sha256) != 64:
            errs.append("expected_sha256 must be a 64-char hex digest")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DownloadJob":
        raw = dict(raw or {})
        job = cls()
        for key in asdict(cls()).keys():
            if key in raw:
                setattr(job, key, raw[key])
        job.bytes_downloaded = int(job.bytes_downloaded or 0)
        job.total_bytes = int(job.total_bytes or 0)
        job.attempts = int(job.attempts or 0)
        job.created_at = float(job.created_at or 0.0)
        job.updated_at = float(job.updated_at or 0.0)
        return job


class DownloadQueue:
    """A persistent queue of download jobs.

    ``fetcher`` (optional) overrides the transport for one job run:
    ``fetcher(job_dict) -> {"ok": bool, "path": str, "size_bytes": int,
    "sha256": str, "resumed": bool, "error": str, "backend": str}``.
    When no fetcher is given, the downloader wiring runs.
    """

    def __init__(
        self,
        queue_path: str | Path | None = None,
        *,
        fetcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        ledger_path: str | Path | None = None,
    ) -> None:
        self.queue_path = Path(queue_path).expanduser() if queue_path else None
        self.fetcher = fetcher
        self._ledger_path = Path(ledger_path).expanduser() if ledger_path else None
        self.jobs: dict[str, DownloadJob] = {}
        if self.queue_path is not None and self.queue_path.is_file():
            self._load_from_disk()

    # -- persistence --------------------------------------------------------
    def _persist(self) -> None:
        if self.queue_path is None:
            return
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")
        payload = {"version": 1,
                   "jobs": [j.to_dict() for j in self.jobs.values()]}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.queue_path)

    def _load_from_disk(self) -> None:
        assert self.queue_path is not None
        try:
            raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DownloadError(f"corrupt download queue {self.queue_path}: {exc}") from exc
        for item in (raw or {}).get("jobs") or []:
            job = DownloadJob.from_dict(item)
            if job.status == JOB_RUNNING:
                # Died mid-download: the job is NOT done — it resumes.
                job.status = JOB_PAUSED
                job.error = "interrupted mid-download; will resume"
            self.jobs[job.job_id] = job

    # -- queue ops --------------------------------------------------------------
    def enqueue(self, url: str, dest: str | Path,
                *, expected_sha256: str = "") -> DownloadJob:
        job = DownloadJob(
            job_id=ulid_now(), url=(url or "").strip(),
            dest=str(dest), expected_sha256=(expected_sha256 or "").strip().lower(),
            created_at=time.time(), updated_at=time.time())
        errs = job.validate()
        if errs:
            raise DownloadError("invalid download job: " + "; ".join(errs))
        self.jobs[job.job_id] = job
        self._persist()
        return job

    def get(self, job_id: str) -> DownloadJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self, status: str = "") -> list[DownloadJob]:
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at)

    def pause(self, job_id: str) -> DownloadJob:
        job = self.get(job_id)
        if job is None:
            raise DownloadError(f"unknown download job {job_id!r}")
        if job.status not in (JOB_QUEUED, JOB_RUNNING):
            raise DownloadError(f"cannot pause a {job.status} job")
        job.status = JOB_PAUSED
        job.updated_at = time.time()
        self._persist()
        return job

    # -- transport ---------------------------------------------------------------
    def _run_with_kit(self, job: DownloadJob) -> dict[str, Any]:
        """Fetch via download_kit.ResumeDownloader (Range resume +
        validator-checked restart + sha256)."""
        ResumeDownloader, _Ledger = _try_import_kit()
        if ResumeDownloader is None:  # pragma: no cover - import failure path
            raise DownloadError("download_kit unavailable")
        report = ResumeDownloader().download(job.url, job.dest)
        if not report.ok:
            return {"ok": False, "error": report.error or "resume download failed",
                    "backend": BACKEND_KIT, "resumed": bool(report.resumed)}
        return {"ok": True, "path": report.path, "size_bytes": report.size_bytes,
                "sha256": report.sha256, "resumed": bool(report.resumed),
                "backend": BACKEND_KIT}

    def _run_fallback(self, job: DownloadJob) -> dict[str, Any]:
        """urllib Range resume with the same validator discipline as the
        kit: a partial is only resumed when saved ETag/Last-Modified
        match the server's current validators; otherwise restart from 0.
        Never splices across a changed file."""
        dest = Path(job.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        meta_path = dest.with_suffix(dest.suffix + ".resume.json")
        try:
            req = urllib.request.Request(
                job.url, method="HEAD",
                headers={"User-Agent": "NoMoralsBrowser/0.1 (download)"})
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                validators = {"etag": resp.headers.get("ETag", ""),
                              "last_modified": resp.headers.get("Last-Modified", "")}
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    total = 0
        except Exception as exc:
            return {"ok": False, "error": f"HEAD failed: {exc}",
                    "backend": BACKEND_FALLBACK}
        partial = dest.read_bytes() if dest.is_file() else b""
        saved: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - unreadable meta = no proof
                saved = {}
        restarted = False
        if partial and (not saved or saved.get("etag") != validators["etag"]
                        or saved.get("last_modified") != validators["last_modified"]):
            partial, restarted = b"", True  # never splice blind
        start = len(partial)
        if total and start >= total and partial:
            digest = hashlib.sha256(partial).hexdigest()
            dest.write_bytes(partial)
            return {"ok": True, "path": str(dest), "size_bytes": len(partial),
                    "sha256": digest, "resumed": False, "backend": BACKEND_FALLBACK}
        headers = {"User-Agent": "NoMoralsBrowser/0.1 (download)"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            req = urllib.request.Request(job.url, headers=headers)
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                if start and resp.status == 200:
                    # Server ignored Range: it sent the whole file, so the
                    # partial must be discarded, not prepended.
                    partial, restarted = b"", True
                chunk = resp.read()
        except Exception as exc:
            try:
                dest.write_bytes(partial)
                meta_path.write_text(json.dumps(validators), encoding="utf-8")
            except OSError:
                pass
            return {"ok": False, "error": f"fetch failed: {exc}",
                    "backend": BACKEND_FALLBACK}
        data = partial + (chunk or b"")
        dest.write_bytes(data)
        try:
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": True, "path": str(dest), "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "resumed": start > 0 and not restarted,
                "backend": BACKEND_FALLBACK}

    def _record_ledger(self, job: DownloadJob) -> None:
        """Append to the kit's IntegrityLedger when available. Best-effort
        — a ledger failure must not fail a good download."""
        _, Ledger = _try_import_kit()
        if Ledger is None:
            return
        try:
            Ledger(self._ledger_path).append(
                url=job.url, sha256=job.sha256,
                size_bytes=job.bytes_downloaded, source="browser-downloads",
                dest=job.dest)
        except Exception as exc:  # noqa: BLE001 - ledger is bookkeeping
            _log.debug("integrity ledger append failed: %r", exc)

    # -- execution ---------------------------------------------------------------
    def run_job(self, job_id: str) -> DownloadJob:
        """Run one job to completion (done/failed). Resume semantics:
        a paused or interrupted job re-runs from its partial file."""
        job = self.get(job_id)
        if job is None:
            raise DownloadError(f"unknown download job {job_id!r}")
        if job.status == JOB_DONE:
            return job
        if job.status not in (JOB_QUEUED, JOB_PAUSED, JOB_FAILED):
            raise DownloadError(f"cannot run a {job.status} job")
        job.status = JOB_RUNNING
        job.attempts += 1
        job.error = ""
        job.updated_at = time.time()
        self._persist()
        try:
            if self.fetcher is not None:
                outcome = self.fetcher(job.to_dict())
            else:
                ResumeDownloader, _ = _try_import_kit()
                if ResumeDownloader is not None:
                    outcome = self._run_with_kit(job)
                else:
                    outcome = self._run_fallback(job)
        except Exception as exc:  # noqa: BLE001 - a transport crash is a job failure
            job.status = JOB_FAILED
            job.error = f"transport error: {exc}"
            job.updated_at = time.time()
            self._persist()
            return job
        if not isinstance(outcome, dict) or not outcome.get("ok"):
            job.status = JOB_FAILED
            job.error = str((outcome or {}).get("error") or "download failed")
            job.backend = str((outcome or {}).get("backend") or job.backend)
            job.updated_at = time.time()
            self._persist()
            return job
        job.backend = str(outcome.get("backend") or job.backend)
        job.resumed = bool(outcome.get("resumed"))
        job.bytes_downloaded = int(outcome.get("size_bytes") or 0)
        # Integrity: recompute the hash from the finished file — never
        # trust a transport-reported digest for the verdict.
        path = outcome.get("path") or job.dest
        try:
            actual = sha256_file(path)
        except OSError as exc:
            job.status = JOB_FAILED
            job.error = f"finished file unreadable for integrity check: {exc}"
            job.updated_at = time.time()
            self._persist()
            return job
        job.sha256 = actual
        if job.expected_sha256 and job.expected_sha256 != actual:
            job.status = JOB_FAILED
            job.error = ("integrity_mismatch: sha256 of finished file "
                         f"{actual[:16]}… != expected {job.expected_sha256[:16]}…")
            job.updated_at = time.time()
            self._persist()
            return job
        job.status = JOB_DONE
        job.error = ""
        job.updated_at = time.time()
        self._persist()
        self._record_ledger(job)
        return job

    def resume(self, job_id: str) -> DownloadJob:
        """Re-run a paused/failed job (same as run_job, but refuses to
        re-run a finished one — resume means continue, not redo)."""
        job = self.get(job_id)
        if job is None:
            raise DownloadError(f"unknown download job {job_id!r}")
        if job.status == JOB_DONE:
            return job
        if job.status not in (JOB_PAUSED, JOB_FAILED, JOB_QUEUED):
            raise DownloadError(f"cannot resume a {job.status} job")
        return self.run_job(job_id)

    def run_all(self) -> list[DownloadJob]:
        """Run every queued/paused job in enqueue order."""
        out: list[DownloadJob] = []
        for job in self.list_jobs(JOB_QUEUED) + self.list_jobs(JOB_PAUSED):
            out.append(self.run_job(job.job_id))
        return out

    def verify(self, job_id: str) -> tuple[bool, str]:
        """Recompute the finished file's hash and compare with the
        recorded one. Returns (ok, reason). Never raises."""
        job = self.get(job_id)
        if job is None:
            return False, f"unknown download job {job_id!r}"
        if job.status != JOB_DONE:
            return False, f"job is {job.status}, not done"
        try:
            actual = sha256_file(job.dest)
        except OSError as exc:
            return False, f"file unreadable: {exc}"
        if actual != job.sha256:
            return False, "hash mismatch — file changed since download"
        return True, "hash matches recorded sha256"
