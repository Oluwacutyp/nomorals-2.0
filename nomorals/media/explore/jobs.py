"""Exploration autonomy — WS4 Pass 3: the durable sweep jobs and
the scheduled sweep cycle.

The census (``~/workspace/ws4-video-census/CENSUS.md`` §6 P3, §7)
settles the posture: the exploration loop rides the durable
scheduler as an extracted, binding-testable ``ensure_*`` family —
the third sibling of ``media/source_jobs.py`` (the source health
loop) and ``agents/news_jobs.py`` (the news poll). Two sweeps a
day, morning + evening, feeding the briefing cycle; marked
``routine`` — findings reach the owner through the morning
briefing and the ``/explore`` board, not job alerts (a FAILED run
still records as a failed run: pass #71 semantics, routine
silences success noise, never failure records).

Jobs:

* ``exploration sweep (morning)`` — daily 08:30 owner-local.
* ``exploration sweep (evening)`` — daily 18:30 owner-local.

Both fire the ``video_explore`` tool's ``sweep`` action. The slot
hours sit just outside the DEFAULT partner quiet window
(22:00–08:00) so the loop actually runs under default settings:
the morning slot opens the owner's day (after quiet hours lift),
the evening slot lands before the 19:00 day wrap, so its claims
are fresh for tomorrow's briefing. Timezone, ``missed_fire_policy``
(fire_now), ``overlap_policy`` (skip) and the 1800s run timeout
mirror the day derivation job (``agents/dayrecord.py``) — the
freshest daily-slot sibling.

The cycle (:func:`run_sweep_cycle`) is ONE entry point for every
trigger: :func:`~.engine.run_exploration` (P1's sweep entry point,
unchanged) then :func:`~.trends.evaluate_trends` (P2), so a
scheduled sweep ends with fresh claims, not just observations.
The durable job fires it with ``manual=False``; the owner's own
``/explore run`` fires it with ``manual=True``.

Discipline (census §7 — the news poll keeps its discipline inside
the poll; the cycle keeps its inside itself):

* **Quiet hours** — a SCHEDULED sweep that lands inside the
  partner quiet window defers: no sweep begins, no observations
  are recorded, and the run report says the sweep deferred and
  why. The window is ``settings.partner``'s quiet_start/quiet_end
  read with the notifier's own semantics (``agents/notifier.py``
  ``_in_quiet_hours_now``) — mirrored here, not imported, because
  agents sits above media in the layer stack; the config read and
  the hour math are the same.
* **Per-day cap** — at most :data:`MAX_SWEEPS_PER_DAY` scheduled
  sweeps per owner-local day. The two daily slots spend exactly
  the cap; a catch-up refire past it defers and says so. (The
  per-SWEEP caps — beats, results, candidates, zero exemplar
  fetches — are the engine's own, stated in every sweep report.)
* Manual sweeps are never gated by either discipline and never
  consume the cap: the owner's own hand is not the loop's budget.

A sweep that dies mid-run records itself ``failed`` in the store
(engine semantics) and the exception propagates through the tool,
so the job run records failed — never a silent success.

Named dependency (census §7): exploration rides the free proxy
tier + direct. The free pool's catalogue activation
(``refresh_sources``, doc 17) is QUEUED and still has no
production caller, so sweeps run direct + whatever the pool
already holds; the ``/explore`` board says so until that lands.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ...core.logging_setup import get_logger
from ..failures import record_failure
from .engine import explore_for_context
from .trends import evaluate_trends

_log = get_logger(__name__)

__all__ = [
    "EVENING_TIME",
    "EXPLORE_SWEEP_JOB_EVENING",
    "EXPLORE_SWEEP_JOB_MORNING",
    "EXPLORE_SWEEP_JOBS",
    "MAX_SWEEPS_PER_DAY",
    "MORNING_TIME",
    "ensure_explore_jobs",
    "in_quiet_hours",
    "next_sweep_due",
    "owner_tz",
    "run_sweep_cycle",
]

EXPLORE_SWEEP_JOB_MORNING = "exploration sweep (morning)"
EXPLORE_SWEEP_JOB_EVENING = "exploration sweep (evening)"
EXPLORE_SWEEP_JOBS = (EXPLORE_SWEEP_JOB_MORNING,
                      EXPLORE_SWEEP_JOB_EVENING)

#: The two daily slots (owner-local wall clock, "HH:MM").
MORNING_TIME = "08:30"
EVENING_TIME = "18:30"

#: The per-day cap on SCHEDULED sweeps (census §7 budget model:
#: two sweeps/day). Manual sweeps never count against it.
MAX_SWEEPS_PER_DAY = 2

_JOBS = (
    (EXPLORE_SWEEP_JOB_MORNING, MORNING_TIME),
    (EXPLORE_SWEEP_JOB_EVENING, EVENING_TIME),
)


def owner_tz() -> tuple[str, Any]:
    """(name, tzinfo) for the owner's clock — the same resolution
    the day derivation job anchors to (``pulses.timezone``;
    Africa/Lagos by default)."""
    from ...pulses.timezone import resolve_owner_tz

    name, tz, _src = resolve_owner_tz()
    return name, tz


def in_quiet_hours(context: Any, now: float) -> bool:
    """True when ``now`` falls in the partner quiet window.

    Mirrors ``agents.notifier._in_quiet_hours_now`` semantics over
    the same config (``settings.partner.quiet_start`` /
    ``quiet_end``, defaults 22 / 08; equal bounds mean no window),
    evaluated at the given instant in the owner's timezone. No
    partner settings → no window (offline tests, CLI)."""
    partner = getattr(getattr(context, "settings", None),
                      "partner", None)
    if partner is None:
        return False
    try:
        start = int(getattr(partner, "quiet_start", 22) or 0)
        end = int(getattr(partner, "quiet_end", 8) or 0)
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    _name, tz = owner_tz()
    hour = datetime.fromtimestamp(now, tz).hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _scheduled_sweeps_today(store: Any, now: float) -> int:
    """Scheduled sweeps already begun on the owner-local day
    containing ``now`` — the count the per-day cap reads. Any
    status counts: a sweep that began spent its slot."""
    _name, tz = owner_tz()
    day = datetime.fromtimestamp(now, tz).date()
    count = 0
    for row in store.sweeps(limit=100):
        if row.get("trigger") != "scheduled":
            continue
        started = float(row.get("started_at") or 0)
        if started and datetime.fromtimestamp(started, tz).date() == day:
            count += 1
    return count


def run_sweep_cycle(context: Any = None, *,
                    manual: bool = False,
                    store: Any = None,
                    beats: list[dict[str, Any]] | None = None,
                    profile: dict[str, Any] | None = None,
                    watches: list[str] | None = None,
                    settings: Any = None,
                    instruments: list[Any] | None = None,
                    now: float = 0.0) -> dict[str, Any]:
    """One full exploration cycle: discipline gates → sweep →
    trend evaluation. THE scheduled entry point (the durable jobs'
    ``video_explore`` ``sweep`` action calls it with
    ``manual=False``); the owner's ``/explore run`` calls it with
    ``manual=True`` — same entry point, his hand ungated.

    Returns the cycle report. A deferred cycle carries
    ``deferred`` (``"quiet_hours"`` / ``"daily_cap"``) and no
    sweep. A sweep that dies propagates (the store has already
    recorded it failed) so a job fire records a failed run."""
    now = now or time.time()
    store, engine, resolved = explore_for_context(
        context, store=store, profile=profile, watches=watches,
        settings=settings, instruments=instruments)
    cycle: dict[str, Any] = {"ok": True, "manual": bool(manual),
                             "at": now, "deferred": "",
                             "sweep": None, "evaluation": None}
    if not manual:
        if in_quiet_hours(context, now):
            cycle["deferred"] = "quiet_hours"
            cycle["note"] = (
                "inside the partner quiet window — the sweep "
                "defers to the next slot; nothing was swept")
            return cycle
        spent = _scheduled_sweeps_today(store, now)
        if spent >= MAX_SWEEPS_PER_DAY:
            cycle["deferred"] = "daily_cap"
            cycle["note"] = (
                f"today's scheduled-sweep cap "
                f"({MAX_SWEEPS_PER_DAY}) is already spent "
                f"({spent} begun) — the sweep defers")
            return cycle
    report = engine.run_sweep(
        beats if beats is not None else resolved,
        trigger="manual" if manual else "scheduled", now=now)
    evaluation = evaluate_trends(store, now=time.time())
    verdicts = evaluation.get("candidates") or {}
    cycle["sweep"] = report
    cycle["evaluation"] = {
        "claims_written": int(evaluation.get("claims_written") or 0),
        "claims_current": len(evaluation.get("claims_current") or []),
        "promoted": len(verdicts.get("promoted") or []),
        "shed": len(verdicts.get("shed") or []),
        "declined": len(evaluation.get("declined") or []),
    }
    if evaluation.get("error"):
        cycle["evaluation"]["error"] = str(evaluation["error"])
    return cycle


def ensure_explore_jobs(scheduler: Any) -> list[dict[str, Any]]:
    """Register the two daily exploration sweep jobs (idempotent
    by name, the social/jobs.py pattern): daily at the morning +
    evening slots, owner-local, firing the ``video_explore``
    tool's ``sweep`` action on the durable scheduler. Marked
    ``routine`` per the census: the loop is self-reporting
    bookkeeping whose findings land in the briefing and the
    board — but a failed run still records as failed."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    tz_name, _tz = owner_tz()
    for name, at in _JOBS:
        if name in have:
            out.append({"name": name, "already_scheduled": True})
            continue
        job = scheduler.add(
            name, f"daily {at}", "tool",
            {"tool": "video_explore", "args": {"action": "sweep"}},
            timezone=tz_name, missed_fire_policy="fire_now",
            overlap_policy="skip", run_timeout_s=1800, routine=True)
        out.append({"name": name, "scheduled": True,
                    "job_id": job.get("id")})
    return out


def next_sweep_due(scheduler: Any) -> dict[str, Any] | None:
    """The next scheduled sweep as ``{"name", "at"}`` from the
    registered job rows (the board's "next sweep" line), or None
    when the sweep jobs are not registered / carry no next run.
    Never raises — a board garnish, not a gate."""
    if scheduler is None:
        return None
    try:
        rows = scheduler.list_jobs()
    except Exception as exc:  # noqa: BLE001 — garnish, contained
        record_failure("explore.next_sweep_due", exc)
        return None
    due = []
    for row in rows:
        if row.get("name") not in EXPLORE_SWEEP_JOBS:
            continue
        if not row.get("enabled", 1):
            continue
        at = float(row.get("next_run") or 0)
        if at > 0:
            due.append((at, str(row.get("name") or "")))
    if not due:
        return None
    at, name = min(due)
    return {"name": name, "at": at}
