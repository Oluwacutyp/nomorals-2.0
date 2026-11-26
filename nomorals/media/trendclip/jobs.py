"""TrendClip job store — ported from the source's ``trendclip/jobs.py``.

One JSON file per job under ``<trendclip root>/jobs/<job_id>.json``, written
atomically (temp file + rename) so a killed process never leaves a torn
receipt. The stages mirror the source's stage machine; the receipt is the
single source of truth for what a job did, which rung picked what, and
where its artifacts live.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["STAGES", "TrendclipJob", "JobStore", "new_job_id"]

#: The job stage machine, in order. A job moves forward through these;
#: ``failed`` and ``cancelled`` are terminal from any stage.
STAGES = (
    "queued",
    "ingesting",
    "transcribing",
    "analyzing",
    "ready",
    "rendering",
    "validating",
    "complete",
    "failed",
    "cancelled",
)

TERMINAL = frozenset(("complete", "failed", "cancelled"))


def new_job_id() -> str:
    """A trendclip job id. The ``tc_`` prefix is the publisher sweep's
    signal that a planned post's video is already rendered (see
    ``nomorals/agents/shorts_publisher.py``)."""
    return "tc_" + uuid.uuid4().hex[:12]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TrendclipJob:
    """One trendclip job and its receipt."""

    def __init__(self, job_id: str, *, topic: str = "",
                 source_url: str = "", stage: str = "queued",
                 created_at: str = "", updated_at: str = "",
                 error: str = "", receipt: dict[str, Any] | None = None):
        self.job_id = job_id
        self.topic = topic
        self.source_url = source_url
        self.stage = stage
        self.created_at = created_at or _utcnow()
        self.updated_at = updated_at or self.created_at
        self.error = error
        self.receipt: dict[str, Any] = dict(receipt or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "source_url": self.source_url,
            "stage": self.stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "receipt": self.receipt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendclipJob:
        return cls(
            job_id=str(data.get("job_id") or ""),
            topic=str(data.get("topic") or ""),
            source_url=str(data.get("source_url") or ""),
            stage=str(data.get("stage") or "queued"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            error=str(data.get("error") or ""),
            receipt=dict(data.get("receipt") or {}),
        )


class JobStore:
    """Atomic JSON job store under ``<root>/jobs/``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / (job_id + ".json")

    def create(self, *, topic: str = "", source_url: str = "") -> TrendclipJob:
        job = TrendclipJob(new_job_id(), topic=topic, source_url=source_url)
        self.save(job)
        return job

    def save(self, job: TrendclipJob) -> None:
        job.updated_at = _utcnow()
        path = self._path(job.job_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)

    def get(self, job_id: str) -> TrendclipJob:
        path = self._path(job_id)
        if not path.is_file():
            raise KeyError(f"unknown trendclip job {job_id!r}")
        return TrendclipJob.from_dict(
            json.loads(path.read_text(encoding="utf-8")))

    def set_stage(self, job: TrendclipJob, stage: str, *,
                  error: str = "") -> TrendclipJob:
        if stage not in STAGES:
            raise ValueError(f"unknown trendclip stage {stage!r}")
        job.stage = stage
        if error:
            job.error = error
        self.save(job)
        return job

    def list(self, *, stage: str = "") -> list[TrendclipJob]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("tc_*.json")):
            try:
                jobs.append(TrendclipJob.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        if stage:
            jobs = [j for j in jobs if j.stage == stage]
        return jobs
