"""Scheduled weekly review job (small remainders bundle).

``build_weekly_review`` (review.py) already builds the review and
``/focus week`` renders it on demand — but nothing ever ran it on its
own clock. This module is the scheduled half: a durable weekly job
(Sundays 20:00 local — the week is still the one ending then; a
Monday-morning run would review a week only hours old) whose tick
builds the review from the real ledgers (commitments, decisions,
focus) and publishes it through the Notifier, the same destination the
on-demand path uses.

Guarantees, in order:

* once per week — a watermark (``data/weekly_review.json``) records
  the reviewed week; a second tick in the same week publishes nothing;
* empty weeks stay silent — a week with no wins, no slipped promises,
  no blocked items, no focus sessions and no resolved decisions is not
  published and is NOT watermarked (the next tick may still deliver);
* never raises — the tick is a scheduler routine; any failure is
  returned as ``ok: False`` with the error named, never raised into
  the scheduler.

Boot wiring lives beside the other focus/review jobs in
``agents/partner/runtime.py``; the tool itself is registered in
``tools/autonomy.py`` beside the group-digest tick.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = [
    "WEEKLY_REVIEW_JOB",
    "WEEKLY_REVIEW_SPEC",
    "WEEKLY_REVIEW_TOOL",
    "ensure_weekly_review_job",
    "run_weekly_review_tick_once",
]

WEEKLY_REVIEW_JOB = "weekly review"
WEEKLY_REVIEW_TOOL = "__weekly_review_tick__"
#: Sundays 20:00 local. ``week_bounds`` is Monday 00:00 UTC → +7d, so
#: at Sunday evening the week is still the one ending.
WEEKLY_REVIEW_SPEC = "cron 0 20 * * SUN"
_WATERMARK_NAME = "weekly_review.json"


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _read_watermark(path: Path) -> float | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return float(raw.get("last_week_start") or 0.0) or None
    except (TypeError, ValueError):
        return None


def _write_watermark(path: Path, week_start: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_week_start": float(week_start)}),
                   encoding="utf-8")
    tmp.replace(path)


def _has_substance(review: Any) -> bool:
    """A week worth delivering: something happened, slipped, blocked,
    was decided, or was focused on. Anything else is silence."""
    focus = review.focus_totals or {}
    return bool(
        review.wins
        or review.slipped
        or review.blocked
        or (focus.get("sessions") or 0)
        or review.kept
        or review.broken
    )


def ensure_weekly_review_job(scheduler: Any, *,
                             spec: str = WEEKLY_REVIEW_SPEC) -> Any | None:
    """Register the weekly review job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job that
    already matches is returned untouched. Best-effort — a scheduler
    that cannot store the job returns None rather than raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"), job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != WEEKLY_REVIEW_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            WEEKLY_REVIEW_JOB, spec, "tool",
            {"tool": WEEKLY_REVIEW_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


def run_weekly_review_tick_once(
    *,
    context: Any = None,
    notifier: Any = None,
    now: float | None = None,
    ledger: Any = None,
    journal: Any = None,
    timer: Any = None,
    watermark_path: str | Path | None = None,
) -> dict[str, Any]:
    """One weekly review tick. Never raises.

    Builds the review from the real ledgers under
    ``<home>/data/`` (commitments, decisions, focus), publishes it
    via the Notifier when a notifier is given (and returns the payload
    in ``posted`` either way), and watermarks the week. An empty week
    is skipped without watermarking; a week already delivered is
    skipped as already-delivered.
    """
    now_f = float(now if now is not None else time.time())
    summary: dict[str, Any] = {"ok": True, "posted": [], "skipped": "",
                               "error": ""}
    try:
        home = _home_path(context)
        if home is None:
            return {"ok": False, "posted": [], "skipped": "",
                    "error": "no home path on context"}
        data = Path(home) / "data"
        if ledger is None:
            from ..commitments.ledger import CommitmentLedger

            ledger = CommitmentLedger(str(data / "commitments.json"))
        if journal is None:
            from ..decisions.journal import DecisionJournal

            journal = DecisionJournal(str(data / "decisions.json"))
        if timer is None:
            from ..focus.sessions import FocusTimer

            timer = FocusTimer(str(data / "focus.json"))
        from .review import build_weekly_review

        review = build_weekly_review(commitments=ledger, decisions=journal,
                                     focus=timer, now=now_f)
        week_start = float(review.week_start or 0.0)
        mark_path = (Path(watermark_path) if watermark_path is not None
                     else data / _WATERMARK_NAME)
        if week_start and _read_watermark(mark_path) == week_start:
            summary["skipped"] = (
                f"already delivered for week {review.week_label}")
            return summary
        if not _has_substance(review):
            summary["skipped"] = "nothing to review this week"
            return summary
        title = f"weekly review — {review.week_label}"
        body = review.render_markdown()
        payload = {"title": title, "body": body}
        if notifier is not None:
            notifier.publish("weekly_review", title=title, body=body)
        summary["posted"] = [payload]
        if week_start:
            _write_watermark(mark_path, week_start)
        return summary
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {"ok": False, "posted": [], "skipped": "",
                "error": f"{type(exc).__name__}: {exc}"}
