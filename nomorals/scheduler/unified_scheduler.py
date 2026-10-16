"""One scheduler, four views, one store.

Consolidation for the four time systems that previously each kept
their own timing:

* ``nomorals.scheduler`` (user-facing cron / reminders / event hooks)
* ``nomorals.agents.scheduler`` (agent-internal jobs: every / daily /
  cron / rrule / one-shot)
* ``nomorals.missions.runner`` (mission scheduling: steps queued,
  schedule-to-start timeouts)
* ``nomorals.autonomy.idle`` (idle-stage ticks that unlock maintenance)

Those four stay as they are (facades preserve their existing APIs).
This module is the *single* store + *single* ``next_run()`` they can
all converge on: one JSON job store, one recurrence computation, and
four read-only views (cron / interval / mission / organ-tick) that all
derive from the same rows, so the views can never disagree about when
a job fires next.

Design rules:

* **One store.** Jobs live in one JSON file (atomic write: temp file
  in the same directory + fsync + rename). No second table, no second
  in-memory copy that can drift.
* **One ``next_run()``.** Every view calls the same
  :meth:`UnifiedScheduler.next_run`. A view is a filter over the store,
  never a recomputation.
* **Missed-run policy is per-job and explicit.** ``catch_up_once``
  (coalesce all missed firings into one run now), ``skip`` (advance
  past the missed firings without running), ``backfill`` (enumerate
  the missed firings, capped, so the caller can replay them in order).
  There is no implicit default that differs per view.
* **Jitter is decorrelated and bounded.** ``next = nominal +
  U(-jitter_frac, +jitter_frac) * period`` with ``jitter_frac`` default
  0.10 (±10%). The jitter is computed once, at schedule time, and
  stored — views read the stored value, so jitter can never make two
  views disagree. Jitter never pushes a firing into the past: the
  result is clamped to ``> now`` for a freshly computed next run, and
  the bound ``|jitter| <= jitter_frac * period`` always holds.
* **Recurrence is delegated, not re-implemented.** Cron math comes
  from :mod:`nomorals.scheduler.recurrence` (the calendar-grade engine
  the user-facing scheduler already uses). Import is lazy so this
  module stays importable standalone.

Job kinds: ``cron`` | ``interval`` | ``once`` | ``mission`` |
``organ_tick``. A ``mission`` / ``organ_tick`` job carries its
recurrence in ``recurrence`` (``{"type": "cron"|"interval"|"once",
"spec": ...}``) — a mission is a scheduled job with a plan, an organ
tick is a scheduled job with an organ name; neither gets its own
clock.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "JOB_KINDS",
    "MISSED_POLICIES",
    "DEFAULT_JITTER_FRAC",
    "MAX_BACKFILL",
    "UnifiedJob",
    "UnifiedScheduler",
    "apply_missed_policy",
    "decorrelated_jitter",
    "missed_firings",
    "next_occurrence",
]

#: Job kinds understood by the unified store.
JOB_KINDS = frozenset({"cron", "interval", "once", "mission", "organ_tick"})

#: Missed-run policies. ``catch_up_once`` = fire once now (coalesce);
#: ``skip`` = advance without firing; ``backfill`` = replay each missed
#: firing in order (capped at MAX_BACKFILL).
MISSED_POLICIES = frozenset({"catch_up_once", "skip", "backfill"})

#: Default decorrelated jitter: ±10% of the period.
DEFAULT_JITTER_FRAC = 0.10

#: Hard cap on backfilled firings — a job down for a year must not
#: replay a million runs.
MAX_BACKFILL = 100

#: Cron scan cap: reuse the recurrence engine's own horizon (366 days).
_CRON_HORIZON_DAYS = 366


def decorrelated_jitter(period_s: float, jitter_frac: float,
                        rng: random.Random | None = None) -> float:
    """One decorrelated jitter draw, bounded by ``±jitter_frac*period``.

    Uniform in ``[-frac*period, +frac*period]``. Pure function of
    ``(period_s, jitter_frac, rng)`` — no wall clock, no hidden state,
    so the bound is checkable: ``abs(result) <= jitter_frac*period``.
    """
    frac = max(0.0, min(1.0, float(jitter_frac)))
    period = max(0.0, float(period_s))
    if frac == 0.0 or period == 0.0:
        return 0.0
    r = rng or random
    return r.uniform(-frac * period, frac * period)


def _cron_next(expr: str, after_ts: float, tz: str = "") -> float | None:
    """Next cron firing strictly after ``after_ts`` (unix ts)."""
    from .recurrence import next_cron

    try:
        return next_cron(expr, datetime.fromtimestamp(after_ts), tz or "")
    except ValueError:
        return None


def _cron_period(expr: str, after_ts: float, tz: str = "") -> float:
    """Estimated cron period: gap between the next two firings."""
    first = _cron_next(expr, after_ts, tz)
    if first is None:
        return 3600.0
    second = _cron_next(expr, first, tz)
    if second is None or second <= first:
        return 3600.0
    return float(second - first)


def next_occurrence(recurrence: dict[str, Any], after_ts: float,
                    *, tz: str = "") -> float | None:
    """Next nominal (unjittered) occurrence strictly after ``after_ts``.

    ``recurrence`` is ``{"type": "cron", "spec": "<5-field>"}``,
    ``{"type": "interval", "spec": <seconds>}`` or
    ``{"type": "once", "spec": <unix ts>}``. Returns None when the
    series is exhausted (a one-shot in the past).
    """
    rtype = str((recurrence or {}).get("type") or "").lower()
    spec = (recurrence or {}).get("spec")
    if rtype == "cron":
        return _cron_next(str(spec or ""), after_ts, tz)
    if rtype == "interval":
        try:
            interval = max(1.0, float(spec))
        except (TypeError, ValueError):
            raise ValueError(f"bad interval spec: {spec!r}")
        # Phase-anchored: occurrences at ... anchor + k*interval.
        # Anchor defaults to 0 (epoch) unless the caller set one.
        anchor = float((recurrence or {}).get("anchor") or 0.0)
        if after_ts < anchor:
            return anchor
        k = math.floor((after_ts - anchor) / interval) + 1
        return anchor + k * interval
    if rtype == "once":
        try:
            at = float(spec)
        except (TypeError, ValueError):
            raise ValueError(f"bad once spec: {spec!r}")
        return at if at > after_ts else None
    raise ValueError(f"unknown recurrence type: {rtype!r}")


def recurrence_period(recurrence: dict[str, Any], after_ts: float,
                      *, tz: str = "") -> float:
    """Period (seconds) a jitter draw is scaled against."""
    rtype = str((recurrence or {}).get("type") or "").lower()
    if rtype == "interval":
        try:
            return max(1.0, float((recurrence or {}).get("spec")))
        except (TypeError, ValueError):
            return 3600.0
    if rtype == "once":
        return 3600.0
    if rtype == "cron":
        return _cron_period(str((recurrence or {}).get("spec") or ""),
                             after_ts, tz)
    return 3600.0


@dataclass
class UnifiedJob:
    """One row in the unified store."""

    job_id: str
    name: str = ""
    kind: str = "cron"  # cron | interval | once | mission | organ_tick
    recurrence: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    missed_policy: str = "catch_up_once"
    jitter_frac: float = DEFAULT_JITTER_FRAC
    timezone: str = ""
    enabled: bool = True
    next_run: float | None = None  # stored, already jittered
    last_run: float | None = None
    run_count: int = 0
    created_at: float = field(default_factory=time.time)
    # Mission view: the plan (list of steps) for mission jobs.
    plan: list[dict[str, Any]] = field(default_factory=list)
    # Organ-tick view: which organ this tick belongs to.
    organ: str = ""

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(JOB_KINDS)}, got {self.kind!r}")
        if self.missed_policy not in MISSED_POLICIES:
            raise ValueError(
                f"missed_policy must be one of {sorted(MISSED_POLICIES)}, "
                f"got {self.missed_policy!r}")
        self.jitter_frac = max(0.0, min(1.0, float(self.jitter_frac)))
        if not self.recurrence:
            raise ValueError("recurrence is required (type + spec)")
        # Validate the recurrence eagerly — fail fast, not at 3am.
        next_occurrence(self.recurrence, self.created_at,
                        tz=self.timezone)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedJob":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def missed_firings(job: UnifiedJob, now: float,
                   *, cap: int = MAX_BACKFILL) -> list[float]:
    """Nominal firing times in ``(last_run_or_creation, now]``.

    The missed set a ``backfill`` policy replays. For interval jobs
    this walks the phase grid; for cron jobs it walks the cron series
    (bounded by the 366-day scan horizon of the recurrence engine and
    by ``cap``). One-shot jobs have at most one missed firing.
    """
    start = job.last_run if job.last_run is not None else job.created_at
    # A firing exactly at creation is not "missed" — start strictly after.
    out: list[float] = []
    cursor = start
    while len(out) < max(1, int(cap)):
        nxt = next_occurrence(job.recurrence, cursor, tz=job.timezone)
        if nxt is None or nxt > now:
            break
        out.append(nxt)
        cursor = nxt
        # One-shot: at most one occurrence, ever.
        if str(job.recurrence.get("type")) == "once":
            break
    return out


def apply_missed_policy(job: UnifiedJob, now: float,
                        *, cap: int = MAX_BACKFILL) -> dict[str, Any]:
    """Apply ``job.missed_policy`` to a stale ``job.next_run``.

    Returns ``{"action": ..., "fire": [...], "next_run": ...}`` where
    ``action`` is one of ``fire_once`` / ``skipped`` / ``backfill`` /
    ``on_time`` and ``fire`` lists the nominal firing times the caller
    should execute (empty for ``skip`` / ``on_time``). The returned
    ``next_run`` is the job's new stored next run (jittered); the
    caller persists it via :meth:`UnifiedScheduler.save` / by mutating
    the job in place (this function mutates ``job.next_run`` /
    ``job.last_run`` so the store and the decision cannot diverge).

    Honouring the policy:

    * ``catch_up_once`` — fire exactly once now, no matter how many
      periods were missed (coalesce), then reschedule from ``now``.
    * ``skip`` — fire nothing, reschedule from ``now``.
    * ``backfill`` — fire each missed nominal time in order (capped),
      then reschedule from ``now``.
    * Not stale (``next_run`` in the future) → ``on_time``, no change.
    """
    if job.next_run is not None and job.next_run > now:
        return {"action": "on_time", "fire": [], "next_run": job.next_run}
    missed = missed_firings(job, now, cap=cap)
    policy = job.missed_policy
    if policy == "catch_up_once":
        fire = [now] if missed else []
        action = "fire_once" if fire else "on_time"
    elif policy == "skip":
        fire, action = [], "skipped"
    elif policy == "backfill":
        fire, action = list(missed), "backfill" if missed else "on_time"
    else:  # pragma: no cover - validated in __post_init__
        raise ValueError(f"unknown missed_policy: {policy!r}")
    # Reschedule from now (not from the missed firing): the next nominal
    # occurrence after now, plus one fresh decorrelated jitter draw.
    # Jitter here is bounded by construction (see decorrelated_jitter).
    nominal = next_occurrence(job.recurrence, now, tz=job.timezone)
    if nominal is None:
        job.next_run = None
        job.enabled = False
    else:
        period = recurrence_period(job.recurrence, now, tz=job.timezone)
        job.next_run = nominal + decorrelated_jitter(period, job.jitter_frac)
        if job.next_run <= now:
            job.next_run = now + 1.0
    if fire:
        job.last_run = fire[-1]
    return {"action": action, "fire": fire, "next_run": job.next_run}


class UnifiedScheduler:
    """One scheduler, four views, one JSON store.

    Usage::

        sched = UnifiedScheduler(Path("jobs.json"))
        sched.add_job(UnifiedJob(job_id="daily", kind="cron",
            recurrence={"type": "cron", "spec": "0 9 * * *"}))
        sched.next_run("daily")          # single source of truth
        sched.cron_view()                # [{"job_id": ..., "next_run": ...}]
        sched.interval_view()            # same next_run for interval jobs
        sched.mission_view()             # missions, with plans
        sched.organ_tick_view()          # organ ticks, with organ names
        sched.save()                     # atomic JSON persist
    """

    def __init__(self, store_path: str | Path | None = None,
                 *, jobs: list[UnifiedJob] | None = None) -> None:
        self.store_path = Path(store_path) if store_path else None
        self._jobs: dict[str, UnifiedJob] = {}
        if jobs:
            for job in jobs:
                self._jobs[job.job_id] = job
        if self.store_path is not None and self.store_path.exists():
            self.load()

    # ── store ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load jobs from the JSON store (replaces in-memory state)."""
        if self.store_path is None:
            return
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(
                raw.get("jobs"), list):
            raise ValueError(f"bad unified-scheduler store: {self.store_path}")
        self._jobs = {j["job_id"]: UnifiedJob.from_dict(j)
                      for j in raw["jobs"]}

    def save(self) -> Path | None:
        """Persist jobs as JSON (atomic: temp + fsync + rename)."""
        if self.store_path is None:
            return None
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "jobs": [j.to_dict() for j in self._jobs.values()],
                   "saved_at": time.time()}
        fd, tmp = tempfile.mkstemp(
            dir=str(self.store_path.parent),
            prefix=self.store_path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.store_path)
            # fsync the directory so the rename itself is durable.
            try:
                dfd = os.open(str(self.store_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:  # pragma: no cover - platform without dir fsync
                pass
        except BaseException:
            with __import__("contextlib").suppress(OSError):
                os.unlink(tmp)
            raise
        return self.store_path

    # ── jobs ─────────────────────────────────────────────────────────

    def add_job(self, job: UnifiedJob, *,
                now: float | None = None,
                rng: random.Random | None = None) -> UnifiedJob:
        """Add a job; computes + stores its first jittered ``next_run``.

        ``now`` is injectable for tests; ``rng`` seeds the first jitter
        draw (default: system randomness).
        """
        if job.job_id in self._jobs:
            raise ValueError(f"job_id {job.job_id!r} already exists")
        at = now if now is not None else time.time()
        nominal = next_occurrence(job.recurrence, at, tz=job.timezone)
        if nominal is None:
            job.next_run = None
            job.enabled = False
        else:
            period = recurrence_period(job.recurrence, at, tz=job.timezone)
            job.next_run = nominal + decorrelated_jitter(
                period, job.jitter_frac, rng)
            if job.next_run <= at:
                job.next_run = at + 1.0
        self._jobs[job.job_id] = job
        return job

    def remove_job(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def get_job(self, job_id: str) -> UnifiedJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[UnifiedJob]:
        return list(self._jobs.values())

    # ── the one next_run ─────────────────────────────────────────────

    def next_run(self, job_id: str, *,
                 now: float | None = None) -> float | None:
        """The single source of truth for when ``job_id`` fires next.

        Returns the stored (jittered) ``next_run`` when it is still in
        the future. When it is stale, the job's missed-run policy
        decides (see :func:`apply_missed_policy`) and the stored value
        is refreshed — so a stale read never returns a stale time, and
        every view sees the same refreshed value afterwards.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id!r}")
        at = now if now is not None else time.time()
        if job.next_run is not None and job.next_run > at:
            return job.next_run
        decision = apply_missed_policy(job, at)
        return decision["next_run"]

    # ── four views — filters over the one store, one next_run ────────

    def _view_row(self, job: UnifiedJob, *,
                  now: float | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "job_id": job.job_id,
            "name": job.name,
            "kind": job.kind,
            "missed_policy": job.missed_policy,
            "enabled": job.enabled,
            "next_run": self.next_run(job.job_id, now=now),
            "last_run": job.last_run,
            "run_count": job.run_count,
        }
        if job.kind == "mission":
            row["plan"] = job.plan
        if job.kind == "organ_tick":
            row["organ"] = job.organ
        return row

    def cron_view(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Cron-kind jobs (user cron + agent cron + daily), by next run."""
        rows = [self._view_row(j, now=now) for j in self._jobs.values()
                if j.kind == "cron" and j.enabled]
        return sorted(rows, key=lambda r: (r["next_run"] is None,
                                            r["next_run"] or 0.0))

    def interval_view(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Interval-kind jobs (``every N seconds``), by next run."""
        rows = [self._view_row(j, now=now) for j in self._jobs.values()
                if j.kind == "interval" and j.enabled]
        return sorted(rows, key=lambda r: (r["next_run"] is None,
                                            r["next_run"] or 0.0))

    def mission_view(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Mission jobs: scheduled work with a plan, by next run."""
        rows = [self._view_row(j, now=now) for j in self._jobs.values()
                if j.kind == "mission" and j.enabled]
        return sorted(rows, key=lambda r: (r["next_run"] is None,
                                            r["next_run"] or 0.0))

    def organ_tick_view(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Organ-tick jobs (idle / presence / heartbeat ticks), by next run."""
        rows = [self._view_row(j, now=now) for j in self._jobs.values()
                if j.kind == "organ_tick" and j.enabled]
        return sorted(rows, key=lambda r: (r["next_run"] is None,
                                            r["next_run"] or 0.0))

    # ── due / advance ────────────────────────────────────────────────

    def due_jobs(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Jobs whose firing is due at ``now``, with missed-policy verdicts.

        Each entry: ``{"job": ..., "action": ..., "fire": [...]}``.
        ``fire`` is non-empty exactly when the caller should execute
        (once for ``catch_up_once``, once per missed firing for
        ``backfill``, never for ``skip``).
        """
        at = now if now is not None else time.time()
        out: list[dict[str, Any]] = []
        for job in list(self._jobs.values()):
            if not job.enabled or job.next_run is None:
                continue
            if job.next_run <= at:
                verdict = apply_missed_policy(job, at)
                out.append({"job": job, **verdict})
        return out

    def mark_fired(self, job_id: str, *, now: float | None = None) -> None:
        """Record a firing: bump ``last_run`` / ``run_count``."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id!r}")
        at = now if now is not None else time.time()
        job.last_run = at
        job.run_count += 1
