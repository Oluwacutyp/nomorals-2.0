"""Schedules autonomous research jobs on Devon's own tick loop.

Jobs are defined in code (see default_jobs()) and their last-run state is
persisted in the database, so a restart doesn't re-fire everything at once.
Cadences are conservative on purpose: background research earns attention by
being rare and good, not frequent and noisy.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .pipeline import JobReport, ResearchContext, ResearchJob, ensure_schema, execute_job

_log = get_logger(__name__)


def default_jobs() -> list[ResearchJob]:
    """The out-of-the-box research watchlist. Two jobs, 24h cadence each."""
    return [
        ResearchJob(
            id="money-watch",
            topic="AI-training gigs and expert platforms hiring Nigerians",
            queries=[
                "AI training jobs hiring Nigeria 2026",
                "Outlier AI expert roles coding",
                "Mindrift hiring AI trainer",
            ],
            cadence_hours=24.0,
            goals=["money"],
            max_results=6,
            fetch_top=2,
        ),
        ResearchJob(
            id="builder-watch",
            topic="Tools and releases relevant to the owner's builds",
            queries=[
                "llama.cpp release phone inference",
                "Android face swap open source",
                "local LLM fine-tune tools",
            ],
            cadence_hours=24.0,
            goals=["devon", "virtual_cam"],
            max_results=6,
            fetch_top=1,
        ),
    ]


class ResearchScheduler:
    """Owns research jobs and runs the ones that are due.

    The tick loop runs on a daemon thread. Job definitions live in memory;
    ``research_job_state`` in the DB records last-run times so restarts don't
    stampede. All delivery anti-spam lives in pipeline.deliver().
    """

    def __init__(
        self,
        db: Any,
        rctx: ResearchContext,
        *,
        tick_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from ..core.profiles import profile_value
        if tick_seconds is None:
            tick_seconds = float(profile_value("tick_seconds", 300.0))
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive")
        ensure_schema(db)
        self.db = db
        self.rctx = rctx
        self.tick_seconds = float(tick_seconds)
        self.clock = clock
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── job registry ─────────────────────────────────────────────────────────

    def add_job(self, job: ResearchJob) -> None:
        if not job.id or not job.queries:
            raise ValueError("research job needs an id and at least one query")
        with self._lock:
            self._jobs[job.id] = job
        _log.info("research job added: %s (every %.1fh)", job.id, job.cadence_hours)

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def list_jobs(self) -> list[ResearchJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    # ── due logic ────────────────────────────────────────────────────────────

    def _last_run(self, job_id: str) -> float:
        row = self.db.query_one(
            "SELECT last_run FROM research_job_state WHERE job_id = ?",
            (job_id,),
        )
        return float(row["last_run"]) if row else 0.0

    def _record_run(self, job_id: str, status: str) -> None:
        self.db.execute(
            "INSERT INTO research_job_state (job_id, last_run, last_status)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(job_id) DO UPDATE SET"
            " last_run = excluded.last_run, last_status = excluded.last_status",
            (job_id, self.clock(), status),
        )

    def due_jobs(self, now: float | None = None) -> list[ResearchJob]:
        now = self.clock() if now is None else now
        due: list[ResearchJob] = []
        for job in self.list_jobs():
            if not job.enabled:
                continue
            if now - self._last_run(job.id) >= job.cadence_hours * 3600.0:
                due.append(job)
        return due

    def run_due(self, now: float | None = None) -> list[JobReport]:
        """Run every due job once. Per-job failures are captured in the
        report; a job that can't run at all still records its run so a
        broken job doesn't hot-loop."""
        reports: list[JobReport] = []
        for job in self.due_jobs(now):
            try:
                report = execute_job(job, self.rctx)
                status = (
                    f"ok d={report.delivered} s={report.skipped}"
                    if not report.errors
                    else f"errors: {report.errors[0][:100]}"
                )
            except Exception as exc:  # noqa: BLE001 - one dead job != dead scheduler
                _log.warning("research job %s failed: %s", job.id, exc)
                report = JobReport(
                    job_id=job.id, findings=0, delivered=0, skipped=0,
                    errors=[str(exc)],
                )
                status = f"failed: {exc}"[:200]
            self._record_run(job.id, status)
            reports.append(report)
        return reports

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("research scheduler already running")
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._tick_loop,
                name="research-scheduler",
                daemon=True,
            )
            self._thread.start()
        _log.info(
            "research scheduler started (%d jobs, tick %.0fs)",
            len(self._jobs), self.tick_seconds,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            try:
                reports = self.run_due()
                for report in reports:
                    _log.info(
                        "research job %s done: %d findings, %d delivered, %d skipped",
                        report.job_id, report.findings,
                        report.delivered, report.skipped,
                    )
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                _log.warning("research tick failed: %s", exc)
            self._stop.wait(self.tick_seconds)
