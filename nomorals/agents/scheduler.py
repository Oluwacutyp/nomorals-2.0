"""Scheduler: durable cron-style jobs, run in-process.

Schedule kinds, parsed from one flexible spec string:

* ``at 2026-09-12 09:00`` (or ISO ``2026-09-12T09:00``) — one-shot
* ``every 30m`` / ``2h`` / ``45s`` — repeating interval
* ``daily 22:00`` (or just ``22:00``) — repeating wall-clock time
* ``daily 22:00 America/New_York`` — daily in a specific IANA timezone
* ``cron 0 22 * * *`` (or bare ``0 22 * * *``) — standard cron expression,
  extended: ``L`` (last day of month), ``LW`` (last weekday), ``15W``
  (weekday nearest the 15th), ``5#3`` (3rd Friday), ``5L`` (last Friday),
  ``?`` (no specific value), day names (``MON-FRI``)
* ``rrule FREQ=WEEKLY;BYDAY=MO,WE`` — RFC 5545 recurrence (calendar-grade:
  "2nd Tuesday" = ``FREQ=MONTHLY;BYDAY=2TU``, "last Friday" =
  ``FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1``)

Three payload kinds:

* ``message`` — send text to the owner on every live channel (via Notifier)
* ``tool``    — call any registered tool with JSON args
* ``command`` — run a shell command through the sandboxed shell tool

Execution policies (per-job data, not code branches):

* **Dependencies** — ``depends_on`` accepts one job id or a list; it fires
  when the dependency gate opens under ``depends_policy``:
  ``all_ok`` (every dependency's last run succeeded),
  ``any_ok`` (at least one did), ``latest_ok`` (the most recent run among
  them succeeded).
* **Retries** — ``max_retries`` + ``retry_delay`` with a selectable
  ``backoff`` strategy: ``constant`` | ``linear`` | ``exponential``
  (default), capped at ``backoff_max_s``, with ``backoff_jitter``
  fractional jitter.  Native implementation, no external deps.
* **Missed-fire policy** — when a firing is missed (bot down, tick
  stalled, pressure deferral), ``missed_fire_policy`` decides:
  ``fire_now`` (coalesce and run once), ``skip`` (drop it, reschedule
  from now), ``next_only`` (advance the schedule past the missed firing
  without running).
* **Overlap policy** — when a job is due while its previous run is still
  executing: ``concurrent`` (run anyway), ``skip`` (drop this firing),
  ``queue`` (leave for the next tick).
* **Run timeout** — ``run_timeout_s`` caps one execution's wall clock
  (0 = the scheduler default).  A timed-out run is marked failed and its
  worker abandoned; the tick loop is never blocked forever.
* **Resource-aware** — pass a ``resources`` advisor (duck-typed
  ``consult() -> dict``, e.g. the injected ``ResourceManager``); heavy
  jobs (``heavy=True`` or ``weight >= HEAVY_WEIGHT_THRESHOLD``) defer
  under pressure — never dropped, retried on later ticks.
* **Real concurrency** — each tick dispatches due jobs onto a bounded
  worker pool (``max_concurrent`` threads); jobs genuinely overlap, a
  stuck job no longer stalls the tick loop or the redelivery sweep, and
  ``overlap_policy=concurrent`` actually runs concurrently.  The pool is
  per-tick (no leaked threads, no exit hangs); the tick still returns
  every outcome, in dispatch order.
* **Run-history retention** — ``schedule_runs`` is pruned to the last
  ``run_history_limit`` rows per job (default 200), so years of uptime
  don't grow the table forever.
* **Blackout dates** — per-job ``blackout_dates`` (``YYYY-MM-DD`` list,
  evaluated in the job's timezone): a due firing on a blackout day is
  skipped without running, journaled, and the schedule advances.
* **Start jitter** — ``start_jitter_s`` spreads a job's first firing by a
  random 0..N seconds (systemd ``RandomizedDelaySec``), so a fleet of
  jobs created together doesn't thundering-herd the first tick.

Jobs live in the ``schedule_jobs`` table (migration 15, upgraded by 81
and 85): a restart resumes them, a disabled job stays put, and one-shot
jobs disable themselves after firing.  Every execution lands one row in
``schedule_runs``; every run is journaled to the autonomy ledger; and
``scheduler.job.started`` / ``scheduler.job.finished`` events go out on
the global event bus so other systems (triggers, the ledger, the
timeline) can react — cross-system triggering for real.  The tick loop
runs on a daemon thread and every due job's outcome is published through
the Notifier, so alerts are durable and multi-channel.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.errors import AmbiguousRef
from ..core.events import Event, global_bus
from ..core.ids import min_unique_prefix_len, new_id, resolve_id_prefix
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet
from ..scheduler.recurrence import (
    CronSpec,
    cron_matches as _rec_cron_matches,
    next_cron as _rec_next_cron,
    parse_rrule as _rec_parse_rrule,
)
from .autonomy_ledger import record_ledger
from .notifier import Notifier

_log = get_logger(__name__)

__all__ = [
    "Scheduler",
    "parse_schedule_spec",
    "HEAVY_WEIGHT_THRESHOLD",
    "MISSED_FIRE_POLICIES",
    "OVERLAP_POLICIES",
    "BACKOFF_STRATEGIES",
    "DEPENDS_POLICIES",
]

# ── execution policies (per-job data) ────────────────────────────────────────

#: What to do with a missed firing.
MISSED_FIRE_POLICIES = frozenset({"fire_now", "skip", "next_only"})
#: What to do when a job is due but its previous run is still executing.
OVERLAP_POLICIES = frozenset({"concurrent", "skip", "queue"})
#: Retry-delay strategies.
BACKOFF_STRATEGIES = frozenset({"constant", "linear", "exponential"})
#: How a multi-job depends_on list gates firing.
DEPENDS_POLICIES = frozenset({"all_ok", "any_ok", "latest_ok"})

#: Metadata ``weight`` at or above this value marks a job as heavy.
HEAVY_WEIGHT_THRESHOLD = 1.0

#: A job this late (seconds) is merely late, not missed — it runs normally.
#: Older than this, the job's ``missed_fire_policy`` decides.
MISSED_GRACE_SECONDS = 300.0


def _backoff_delay(
    strategy: str,
    base_delay: float,
    attempt: int,
    *,
    max_s: float = 3600.0,
    jitter: float = 0.25,
    rng: random.Random | None = None,
) -> float:
    """Native retry-delay computation for one attempt (1-based).

    ``constant`` → base every time; ``linear`` → base × attempt;
    ``exponential`` → base × 2^(attempt-1).  The result is capped at
    ``max_s`` and perturbed by ±``jitter`` fractionally so a fleet of
    failed jobs does not retry in lockstep.  Never raises; floors at 1s.
    """
    strategy = (strategy or "exponential").strip().lower()
    base = max(1.0, float(base_delay or 1.0))
    attempt = max(1, int(attempt or 1))
    if strategy == "constant":
        delay = base
    elif strategy == "linear":
        delay = base * attempt
    else:  # exponential (and any unknown value degrades to it, loudly)
        if strategy != "exponential":
            _log.warning("unknown backoff strategy %r — using exponential",
                         strategy)
        delay = base * (2.0 ** (attempt - 1))
    delay = min(delay, max(1.0, float(max_s or 1.0)))
    jitter = min(0.9, max(0.0, float(jitter or 0.0)))
    if jitter > 0:
        r = rng if rng is not None else random
        factor = 1.0 + (r.random() * 2.0 - 1.0) * jitter
        delay = delay * max(0.1, factor)
    return max(1.0, delay)


def _parse_depends(raw: Any) -> list[str]:
    """Normalize ``depends_on`` to a list of job ids.

    Accepts a single id, a comma-separated string, a JSON list string, or
    a real list.  Never raises — garbage becomes an empty list (and the
    gate then treats "no dependencies" as open, matching history).
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, TypeError):
            pass
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]

_INTERVAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|m|h|d|sec|min|mins|hr|hrs|hours?|days?|seconds?|minutes?)\s*$", re.IGNORECASE)
_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{1,2})\s*$")
_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
# cron: 5 fields (minute hour day month weekday)
_CRON_RE = re.compile(
    r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)
# IANA timezone-ish: letters, digits, underscore, slash, hyphen, plus
_TZ_RE = re.compile(r"^[A-Za-z0-9_+\-]+(/[A-Za-z0-9_+\-]+)*$")


def parse_schedule_spec(spec: str) -> tuple[str, Any]:
    """Parse a schedule spec → (kind, detail).

    kind: ``at`` (detail = unix ts) | ``every`` (detail = seconds) |
    ``daily`` (detail = "HH:MM" or "HH:MM <tz>") |
    ``cron`` (detail = cron expression string) |
    ``rrule`` (detail = RRULE string, e.g. ``FREQ=WEEKLY;BYDAY=MO``)
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("empty schedule spec")
    lowered = s.lower()

    if lowered.startswith("at "):
        ts = _parse_timestamp(s[3:].strip())
        return "at", ts
    if lowered.startswith("every "):
        return "every", _parse_interval(s[6:].strip())
    if lowered.startswith("daily "):
        return "daily", _parse_daily_spec(s[6:].strip())
    if lowered.startswith("cron "):
        return "cron", _parse_cron(s[5:].strip())
    if lowered.startswith("rrule "):
        return "rrule", _parse_rrule_spec(s[6:].strip())
    # bare forms: "22:00" → daily, "30m" → every, ISO timestamp → at,
    # "0 22 * * *" → cron
    if _TIME_RE.match(s):
        return "daily", _parse_hhmm(s)
    # bare "HH:MM <tz>"
    parts = s.split()
    if len(parts) == 2 and _TIME_RE.match(parts[0]) and _TZ_RE.match(parts[1]):
        return "daily", _parse_daily_spec(s)
    if _INTERVAL_RE.match(s):
        return "every", _parse_interval(s)
    if _CRON_RE.match(s):
        try:
            return "cron", _parse_cron(s)
        except ValueError:
            _log.debug("scheduler: %r matched cron shape but did not parse; trying timestamp", s)
    # bare RRULE: "FREQ=WEEKLY;BYDAY=MO" (or "RRULE:FREQ=...")
    if re.match(r"(?i)^(?:RRULE:)?FREQ=", s):
        return "rrule", _parse_rrule_spec(s)
    ts = _parse_timestamp(s)
    return "at", ts


def _parse_timestamp(text: str) -> float:
    text = text.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.year < 1000:  # "%m/%d/%Y" already has a year
                parsed = parsed.replace(year=datetime.now().year)
            return parsed.timestamp()
        except ValueError:
            continue
    raise ValueError(f"could not parse a time from {text!r}")


def _parse_interval(text: str) -> float:
    match = _INTERVAL_RE.match(text)
    if not match:
        raise ValueError(f"could not parse an interval from {text!r}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"unknown interval unit {unit!r}")
    seconds = value * _UNIT_SECONDS[unit]
    if seconds < 10:
        raise ValueError("intervals must be at least 10 seconds")
    return seconds


def _parse_hhmm(text: str) -> str:
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"expected HH:MM, got {text!r}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        raise ValueError(f"not a valid time: {text!r}")
    return f"{hour:02d}:{minute:02d}"  # pads "9:5" → "09:05"


def _parse_daily_spec(text: str) -> str:
    """Parse ``HH:MM [timezone]`` → ``"HH:MM"`` or ``"HH:MM <tz>"``.

    The timezone must be a valid IANA name (validated eagerly so a
    typo fails at schedule time, not at 3am).
    """
    parts = text.strip().split()
    if not parts:
        raise ValueError("expected HH:MM [timezone]")
    hhmm = _parse_hhmm(parts[0])
    if len(parts) == 1:
        return hhmm
    tz = " ".join(parts[1:])
    # allow one slash-separated IANA name; reject junk early
    if not _TZ_RE.match(tz):
        raise ValueError(f"not a valid timezone: {tz!r}")
    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        _log.warning("scheduler: unknown timezone %r (tzdata missing?), using local", tz)
        return hhmm
    return f"{hhmm} {tz}"


def _split_daily_detail(detail: str) -> tuple[str, str]:
    """Split a daily detail ``"HH:MM"`` / ``"HH:MM <tz>"`` → (hhmm, tz)."""
    parts = str(detail or "").strip().split(None, 1)
    hhmm = parts[0] if parts else "00:00"
    tz = parts[1].strip() if len(parts) > 1 else ""
    return hhmm, tz


def _next_daily(hhmm: str, now: datetime, timezone: str = "") -> float:
    """Next wall-clock ``HH:MM`` at/after ``now``.

    ``hhmm`` may itself carry a trailing IANA timezone
    (``"22:00 America/New_York"``); the explicit ``timezone`` argument
    wins when both are given.  An empty timezone means server-local.
    """
    # detail may embed the tz: "22:00 America/New_York"
    embedded_hhmm, embedded_tz = _split_daily_detail(hhmm)
    tz_name = (timezone or "").strip() or embedded_tz
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            _log.warning("scheduler: unknown timezone %r, using local", tz_name)
            tz = None
        if tz is not None:
            # do the wall-clock math in the target zone
            now_tz = now.astimezone(tz)
            hour, minute = (int(x) for x in embedded_hhmm.split(":"))
            candidate = now_tz.replace(hour=hour, minute=minute,
                                       second=0, microsecond=0)
            if candidate <= now_tz:
                candidate += timedelta(days=1)
            return candidate.timestamp()
    hour, minute = (int(x) for x in embedded_hhmm.split(":"))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def _rrule_next(spec: str, dtstart_ts: float, after_ts: float,
                timezone: str = "") -> float | None:
    """Next RRULE occurrence strictly after ``after_ts`` (unix ts).

    The series is anchored at ``dtstart_ts`` (the job's creation time);
    both are evaluated in ``timezone`` when given.  Returns None when the
    series is exhausted (COUNT reached / UNTIL passed).
    """
    tz: Any = None
    if (timezone or "").strip():
        try:
            tz = ZoneInfo(timezone.strip())
        except ZoneInfoNotFoundError:
            _log.warning("scheduler: unknown timezone %r, using local",
                         timezone)
    dtstart = datetime.fromtimestamp(dtstart_ts, tz=tz)
    after = datetime.fromtimestamp(after_ts, tz=tz)
    rule = _rec_parse_rrule(spec, dtstart)
    nxt = rule.after(after)
    return nxt.timestamp() if nxt is not None else None


_BLACKOUT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_blackout_dates(raw: Any) -> list[str]:
    """Validate ``blackout_dates`` → sorted list of ``YYYY-MM-DD``.

    Accepts a list/tuple, a comma-separated string, or a JSON list
    string.  Raises ValueError on malformed dates (fail at schedule
    time, not on a blackout morning).
    """
    if not raw:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                items = parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, TypeError):
                raise ValueError(f"bad blackout_dates JSON: {raw!r}")
        else:
            items = [p.strip() for p in text.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError(f"bad blackout_dates: {raw!r}")
    out: list[str] = []
    for item in items:
        day = str(item or "").strip()
        if not day:
            continue
        if not _BLACKOUT_RE.match(day):
            raise ValueError(f"blackout date must be YYYY-MM-DD, got {day!r}")
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"not a real date: {day!r}")
        out.append(day)
    return sorted(set(out))


# ── cron ─────────────────────────────────────────────────────────────────

def _parse_cron_field(field: str, lo: int, hi: int) -> set[int]:
    """Parse one cron field → set of matching ints.

    Supports ``*``, ``*/n``, ``a,b,c``, ``a-b``, ``a-b/n``, and literals.
    """
    field = field.strip()
    out: set[int] = set()
    if field == "*":
        return set(range(lo, hi + 1))
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"bad cron step in {field!r}")
        if part == "*" or part == "":
            lo_p, hi_p = lo, hi
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            lo_p, hi_p = int(a_s), int(b_s)
        else:
            lo_p = hi_p = int(part)
        if lo_p < lo or hi_p > hi or lo_p > hi_p:
            raise ValueError(f"cron value out of range in {field!r}")
        out.update(range(lo_p, hi_p + 1, step))
    if not out:
        raise ValueError(f"empty cron field {field!r}")
    return out


def _parse_cron(expr: str) -> str:
    """Validate a 5-field cron expression → normalized string.

    Fields: minute hour day month weekday.  Weekday 0 and 7 both mean
    Sunday.  Extended fields supported: ``L``, ``LW``, ``nW``, ``n#k``,
    ``nL``, ``?``, day names.  Returns the canonical
    ``"m h dom mon dow"`` string.
    """
    return CronSpec(expr).expression


def _cron_matches(expr: str, dt: datetime) -> bool:
    """True if ``dt`` (minute precision) matches the cron expression."""
    return _rec_cron_matches(expr, dt)


def _next_cron(expr: str, now: datetime, timezone: str = "") -> float:
    """Next minute strictly after ``now`` matching the cron expression.

    The expression is parsed once (not per probed minute) and the
    wall-clock scan runs in ``timezone`` when given (DST-safe).
    """
    return _rec_next_cron(expr, now, timezone)


def _parse_rrule_spec(text: str) -> str:
    """Validate an RRULE body → normalized rule string.

    Raises ValueError on garbage.  A leading ``RRULE:`` (iCalendar
    property form) is tolerated and stripped.
    """
    text = (text or "").strip()
    if text.upper().startswith("RRULE:"):
        text = text[6:].strip()
    _rec_parse_rrule(text, datetime.now())  # validate eagerly
    return text


def _like_escape(text: str) -> str:
    """Escape SQL LIKE wildcards so a job ref is matched literally."""
    return (text.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_"))


def _profile_name(context: Any) -> str:
    """Runtime profile: termux | laptop | workstation (default).

    ``NM_PROFILE`` wins, then ``settings.profile`` — the same dial the
    profile-aware runtime uses.  The scheduler uses it to gate default
    parallelism instead of designing down to the weakest machine.
    """
    env = (os.environ.get("NM_PROFILE") or "").strip().lower()
    if env:
        return env
    try:
        prof = getattr(getattr(context, "settings", None), "profile", "")
        if prof:
            return str(prof).strip().lower()
    except Exception:  # noqa: BLE001
        pass
    return "workstation"


class Scheduler:
    """Owns the schedule_jobs table and the tick loop."""

    def __init__(self, context: Any, *, gateway: Any = None,
                 tick_seconds: float = 20.0, max_concurrent: int | None = None,
                 wall_seconds: float = 300.0,
                 redeliver_interval: float = 120.0,
                 resources: Any = None,
                 missed_grace_s: float = MISSED_GRACE_SECONDS,
                 run_history_limit: int = 200) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.notifier = Notifier(context, gateway=gateway)
        self.tick_seconds = max(5.0, float(tick_seconds))
        # Profile-gated parallelism instead of designing down: the phone
        # (termux) runs one job at a time; bigger machines keep the default.
        # An explicit max_concurrent always wins.
        if max_concurrent is None:
            max_concurrent = 1 if _profile_name(context) == "termux" else 2
        self.max_concurrent = max(1, int(max_concurrent))
        #: Default per-execution wall-clock cap (seconds).  Enforced for
        #: real in _execute — a job may override it with run_timeout_s.
        self.wall_seconds = max(10.0, float(wall_seconds))
        #: how often the tick loop sweeps the notification redelivery
        #: queue (throttled: one sweep per interval, not per tick).
        self.redeliver_interval = max(15.0, float(redeliver_interval))
        #: Optional resource advisor (duck-typed: consult() -> dict with
        #: ok/throttled/reasons).  Injected by L7 entry points; heavy jobs
        #: defer (never drop) while it reports pressure.
        self._resources = resources
        #: Grace window: a job this late is merely late, not missed.
        self.missed_grace_s = max(0.0, float(missed_grace_s))
        #: Per-job run-history retention: schedule_runs keeps this many
        #: rows per job (K8s successfulJobsHistoryLimit, but per job).
        self.run_history_limit = max(1, int(run_history_limit or 200))
        self._ensure_blackout_column()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running_jobs = 0
        self._lock = threading.Lock()
        #: job ids with a run in flight right now (overlap policies).
        self._active: set[str] = set()
        #: job id -> consecutive deferrals under resource pressure.
        self._deferrals: dict[str, int] = {}
        #: jitter source for backoff (seedable in tests).
        self._rng = random.Random()
        #: health bookkeeping — the numbers behind ``/schedule health``
        self._started_at: float | None = None
        self._last_tick_at: float | None = None
        self._last_redeliver_at = 0.0
        self._tick_errors = 0

    def _ensure_blackout_column(self) -> None:
        """Lazily add ``schedule_jobs.blackout_dates`` on older databases.

        The column postdates migrations 15/81/85; rather than touching the
        shared migrations module (another section's file), the scheduler
        ensures its own column at construction.  Idempotent and
        best-effort: a failure here must never stop the scheduler.
        """
        if self.db is None:
            return
        try:
            self.db.execute(
                "ALTER TABLE schedule_jobs ADD COLUMN blackout_dates "
                "TEXT NOT NULL DEFAULT ''")
        except Exception as exc:  # noqa: BLE001
            if "duplicate column" not in str(exc).lower():
                _log.debug("scheduler blackout column ensure failed: %s", exc)

    # ── cross-system wiring (bus + ledger; both fail-open) ────────────────

    def _emit_bus(self, topic: str, data: dict[str, Any]) -> None:
        """Publish a scheduler event.  Telemetry/control for other
        systems — a broken bus or subscriber must never break a job."""
        try:
            global_bus.publish(Event(topic=topic, data=data,
                                     source="nomorals.agents.scheduler"))
        except Exception:  # noqa: BLE001
            _log.debug("scheduler bus publish %s failed", topic,
                       exc_info=True)

    def _ledger(self, kind: str, ref_id: str, summary: str, *,
                cost_seconds: float = 0.0, cost_tokens: int = 0,
                ok: bool = True, learned: str = "",
                metadata: dict[str, Any] | None = None) -> str:
        """Journal one entry to the unified autonomy ledger. Never raises."""
        try:
            return record_ledger(self.db, "scheduler", kind, ref_id,
                                 summary, cost_seconds=cost_seconds,
                                 cost_tokens=cost_tokens, ok=ok,
                                 learned=learned, metadata=metadata)
        except Exception:  # noqa: BLE001
            _log.debug("scheduler ledger write failed", exc_info=True)
            return ""

    # ── resource pressure ────────────────────────────────────────────────

    def _pressure_gate(self) -> tuple[bool, list[str]]:
        """Consult the injected resource advisor once per tick.

        Returns ``(defer_heavy, reasons)``.  Heavy jobs defer when the
        advisory reports ``throttled`` or ``ok == False``.  Never raises
        and never blocks: with no advisor there is no gating, and a
        failing advisor fails closed for heavy work while light jobs
        still run.
        """
        advisor = self._resources
        if advisor is None:
            return False, []
        try:
            advisory = advisor.consult()
        except Exception as exc:  # noqa: BLE001
            _log.warning("scheduler resource consult failed (%s); "
                         "deferring heavy jobs this tick", exc)
            return True, [f"consult failed: {exc}"]
        if not isinstance(advisory, dict):
            return True, ["consult returned non-dict advisory"]
        throttled = bool(advisory.get("throttled", False))
        ok = bool(advisory.get("ok", True))
        reasons = [str(r) for r in (advisory.get("reasons") or [])]
        if throttled or not ok:
            return True, reasons
        return False, reasons

    @staticmethod
    def _is_heavy(row: dict[str, Any]) -> bool:
        """True when the job row marks heavy work.

        Explicit ``heavy`` wins; otherwise a numeric ``weight`` at or
        above :data:`HEAVY_WEIGHT_THRESHOLD` counts as heavy.  The
        default (neither set) is light.
        """
        try:
            if int(row.get("heavy") or 0):
                return True
        except (TypeError, ValueError):
            pass
        try:
            weight = float(row.get("weight") or 0.0)
        except (TypeError, ValueError):
            return False
        return weight >= HEAVY_WEIGHT_THRESHOLD

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def add(
        self,
        name: str,
        spec: str,
        payload_kind: str,
        payload: dict[str, Any],
        *,
        timezone: str = "",
        depends_on: str | list[str] = "",
        depends_policy: str = "all_ok",
        max_retries: int = 0,
        retry_delay: float = 60.0,
        backoff: str = "exponential",
        backoff_max_s: float = 3600.0,
        backoff_jitter: float = 0.25,
        missed_fire_policy: str = "fire_now",
        overlap_policy: str = "concurrent",
        run_timeout_s: float = 0.0,
        heavy: bool = False,
        weight: float = 0.0,
        blackout_dates: list[str] | str | None = None,
        start_jitter_s: float = 0.0,
    ) -> dict[str, Any]:
        if self.db is None:
            raise RuntimeError("scheduler needs a database context")
        kind, detail = parse_schedule_spec(spec)
        if payload_kind not in {"message", "tool", "command"}:
            raise ValueError("payload_kind must be message, tool, or command")
        if payload_kind == "message" and not str(payload.get("text") or "").strip():
            raise ValueError("message payloads need a 'text'")
        if payload_kind == "tool" and not str(payload.get("tool") or "").strip():
            raise ValueError("tool payloads need a 'tool' name")
        if payload_kind == "command" and not str(payload.get("command") or "").strip():
            raise ValueError("command payloads need a 'command'")
        # dependency: one id or a list; every named job must exist
        dep_ids = _parse_depends(depends_on)
        canonical_deps: list[str] = []
        for dep in dep_ids:
            row = self._find(dep)
            if row is None:
                raise ValueError(f"depends_on: no job {dep!r}")
            canonical_deps.append(row["id"])  # canonicalize to the full id
        depends_policy = (depends_policy or "all_ok").strip().lower()
        if depends_policy not in DEPENDS_POLICIES:
            raise ValueError(
                f"depends_policy must be one of {sorted(DEPENDS_POLICIES)}, "
                f"got {depends_policy!r}")
        max_retries = max(0, int(max_retries))
        retry_delay = max(10.0, float(retry_delay))
        backoff = (backoff or "exponential").strip().lower()
        if backoff not in BACKOFF_STRATEGIES:
            raise ValueError(
                f"backoff must be one of {sorted(BACKOFF_STRATEGIES)}, "
                f"got {backoff!r}")
        backoff_max_s = max(60.0, float(backoff_max_s or 3600.0))
        backoff_jitter = min(0.9, max(0.0, float(backoff_jitter or 0.0)))
        missed_fire_policy = (missed_fire_policy or "fire_now").strip().lower()
        if missed_fire_policy not in MISSED_FIRE_POLICIES:
            raise ValueError(
                f"missed_fire_policy must be one of "
                f"{sorted(MISSED_FIRE_POLICIES)}, got {missed_fire_policy!r}")
        overlap_policy = (overlap_policy or "concurrent").strip().lower()
        if overlap_policy not in OVERLAP_POLICIES:
            raise ValueError(
                f"overlap_policy must be one of {sorted(OVERLAP_POLICIES)}, "
                f"got {overlap_policy!r}")
        run_timeout_s = max(0.0, float(run_timeout_s or 0.0))
        blackout = _parse_blackout_dates(blackout_dates)
        start_jitter_s = max(0.0, float(start_jitter_s or 0.0))
        # explicit timezone arg wins; daily spec may also embed one
        timezone = (timezone or "").strip()
        if timezone:
            try:
                ZoneInfo(timezone)
            except ZoneInfoNotFoundError:
                raise ValueError(f"unknown timezone: {timezone!r}")
        elif kind == "daily":
            # spec like "daily 22:00 America/New_York" embeds the tz in detail
            _, embedded_tz = _split_daily_detail(str(detail))
            timezone = embedded_tz
        now = time.time()
        if kind == "at" and detail <= now:
            raise ValueError("one-shot time is in the past")
        next_run = self._initial_next_run(kind, detail, now, timezone)
        if next_run is None:
            raise ValueError("schedule yields no future occurrences")
        if start_jitter_s > 0 and kind != "at":
            # systemd RandomizedDelaySec: spread the first firing so jobs
            # created together don't thundering-herd the first tick.
            next_run += self._rng.uniform(0, start_jitter_s)
        job_id = new_id()
        # store the detail; for daily keep "HH:MM [tz]" so the tz survives
        if kind == "every":
            spec_str = str(int(detail))
        elif kind == "daily" and timezone and " " not in str(detail):
            spec_str = f"{detail} {timezone}"
        else:
            spec_str = str(detail)
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO schedule_jobs (id, name, kind, spec, payload_kind, payload, "
                "enabled, next_run, last_run, last_result, created_at, updated_at, "
                "timezone, depends_on, depends_policy, max_retries, retry_delay, "
                "retry_count, backoff, backoff_max_s, backoff_jitter, "
                "missed_fire_policy, overlap_policy, run_timeout_s, heavy, weight, "
                "blackout_dates) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, NULL, '', ?, ?, ?, ?, ?, ?, ?, "
                "0, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id, (name or "").strip() or "job", kind,
                    spec_str,
                    payload_kind, json.dumps(payload),
                    next_run, now, now,
                    timezone, json.dumps(canonical_deps), depends_policy,
                    max_retries, retry_delay,
                    backoff, backoff_max_s, backoff_jitter,
                    missed_fire_policy, overlap_policy, run_timeout_s,
                    1 if heavy else 0, float(weight or 0.0),
                    json.dumps(blackout),
                ),
            )
        self._ledger("schedule", job_id,
                     f"scheduled {name!r} ({kind} {spec_str})",
                     metadata={"kind": kind, "spec": spec_str,
                               "payload_kind": payload_kind})
        return {"id": job_id, "name": (name or "").strip() or "job", "kind": kind,
                "next_run": next_run}

    def list_jobs(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT * FROM schedule_jobs ORDER BY COALESCE(next_run, 0) ASC"
            )
        except Exception:  # noqa: BLE001
            return []
        jobs = []
        for row in rows:
            if not include_disabled and not row.get("enabled"):
                continue
            jobs.append(self._format_job(row))
        return jobs

    def remove(self, ref: str) -> bool:
        row = self._find(ref)
        if not row:
            return False
        with self.db.transaction():
            self.db.execute("DELETE FROM schedule_jobs WHERE id = ?", (row["id"],))
        return True

    def set_interval(self, ref: str, seconds: float) -> dict[str, Any] | None:
        """Reschedule a recurring (``every``) job with a new interval.

        Used by the cost-aware cognitive cadence (wave 63) to tighten or
        relax the heartbeat live, without a restart.  One-shot (``at``)
        and daily jobs are left alone (returns None).  ``seconds`` is
        clamped like everything else the scheduler runs.
        """
        row = self._find(ref)
        if row is None or row.get("kind") != "every":
            return None
        seconds = min(48 * 3600.0, max(10.0, float(seconds)))
        now = time.time()
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET spec=?, next_run=?, updated_at=? "
                "WHERE id=?",
                (str(int(seconds)), now + seconds, now, row["id"]))
        return self._find(ref)

    def set_enabled(self, ref: str, enabled: bool) -> dict[str, Any]:
        row = self._find(ref)
        if not row:
            raise LookupError(f"no job {ref!r}")
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, time.time(), row["id"]),
            )
        return self._format_job(self._find(ref))

    def set_blackout_dates(self, ref: str,
                           dates: list[str] | str | None) -> dict[str, Any]:
        """Replace a job's exclusion-calendar blackout dates.

        ``dates`` is a list of ``YYYY-MM-DD`` (or comma-separated string);
        empty/None clears the calendar.  Invalid dates raise ValueError.
        """
        row = self._find(ref)
        if not row:
            raise LookupError(f"no job {ref!r}")
        blackout = _parse_blackout_dates(dates)
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET blackout_dates = ?, updated_at = ? "
                "WHERE id = ?",
                (json.dumps(blackout), time.time(), row["id"]),
            )
        return self._format_job(self._find(ref))

    def _find(self, ref: str) -> dict[str, Any] | None:
        """Resolve a job by id, name, or id prefix — never guesses.

        Exact id wins, then exact name, then a unique id prefix.  A prefix
        matching two or more jobs raises :class:`AmbiguousRef` instead of
        silently acting on the first row.
        """
        if self.db is None:
            return None
        ref = (ref or "").strip()
        if not ref:
            return None
        row = self.db.query_one("SELECT * FROM schedule_jobs WHERE id = ?", (ref,))
        if row:
            return row
        row = self.db.query_one("SELECT * FROM schedule_jobs WHERE name = ?", (ref,))
        if row:
            return row
        rows = self.db.query(
            "SELECT * FROM schedule_jobs WHERE id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at DESC",
            (_like_escape(ref) + "%",),
        )
        if not rows:
            return None
        res = resolve_id_prefix(ref, [r["id"] for r in rows])
        if res.outcome == "ambiguous":
            by_id = {r["id"]: r for r in rows}
            raise AmbiguousRef(
                ref,
                [(c, by_id[c].get("name") or c) for c in res.matches],
                min_unique_prefix_len([r["id"] for r in rows]),
                entity="scheduled job",
            )
        if res.outcome in ("exact", "unique"):
            want = res.matches[0]
            for r in rows:
                if r["id"] == want:
                    return r
        return None

    # ── scheduling math ──────────────────────────────────────────────────────
    def _initial_next_run(self, kind: str, detail: Any, now: float,
                          timezone: str = "") -> float | None:
        if kind == "at":
            return float(detail)
        if kind == "every":
            return now + float(detail)
        if kind == "cron":
            return _next_cron(str(detail), datetime.fromtimestamp(now), timezone)
        if kind == "rrule":
            return _rrule_next(str(detail), now, now, timezone)
        return _next_daily(str(detail), datetime.fromtimestamp(now), timezone)

    def _next_after_run(self, row: dict[str, Any], now: float) -> tuple[float | None, bool]:
        """(next_run, still_enabled) after a job fires."""
        kind = row["kind"]
        tz = str(row.get("timezone") or "")
        if kind == "at":
            return None, False
        if kind == "every":
            return now + float(row["spec"]), True
        if kind == "cron":
            return _next_cron(str(row["spec"]), datetime.fromtimestamp(now), tz), True
        if kind == "rrule":
            nxt = _rrule_next(str(row["spec"]),
                              float(row.get("created_at") or now), now, tz)
            # an exhausted RRULE (COUNT/UNTIL) retires the job like a
            # one-shot — it stays in the table, disabled, for the record.
            return (nxt, True) if nxt is not None else (None, False)
        return _next_daily(str(row["spec"]), datetime.fromtimestamp(now), tz), True

    def _dependency_ok(self, row: dict[str, Any]) -> bool:
        """True when the job's ``depends_on`` gate is open.

        ``depends_on`` is a list of job ids (a lone id is one entry).
        ``depends_policy`` decides the gate:

        * ``all_ok`` — every dependency's last run succeeded;
        * ``any_ok`` — at least one dependency's last run succeeded;
        * ``latest_ok`` — the most recently-run dependency succeeded.

        A deleted dependency, or one that never ran, keeps the gate shut —
        the job waits instead of silently running.
        """
        dep_ids = _parse_depends(row.get("depends_on"))
        if not dep_ids:
            return True
        if self.db is None:
            return False
        policy = str(row.get("depends_policy") or "all_ok").strip().lower()
        if policy not in DEPENDS_POLICIES:
            policy = "all_ok"
        results: list[tuple[bool, float]] = []  # (succeeded, last_run)
        for dep_id in dep_ids:
            dep = self.db.query_one(
                "SELECT last_result, last_run FROM schedule_jobs WHERE id = ?",
                (dep_id,))
            if not dep:
                # dependency was deleted — treat as unmet, don't silently run
                return False
            last = str(dep.get("last_result") or "")
            # never ran counts as unmet; only an explicit success opens the gate
            succeeded = bool(last) and not last.startswith("job failed")
            try:
                last_run = float(dep.get("last_run") or 0.0)
            except (TypeError, ValueError):
                last_run = 0.0
            results.append((succeeded, last_run))
        if policy == "any_ok":
            return any(ok for ok, _ in results)
        if policy == "latest_ok":
            # the most recently executed dependency decides
            latest = max(results, key=lambda r: r[1])
            return bool(latest[0]) and latest[1] > 0
        return all(ok for ok, _ in results)  # all_ok

    # ── missed-fire policies ───────────────────────────────────────────────

    def _apply_missed_fire(
        self,
        row: dict[str, Any],
        now: float,
        *,
        cutoff: float | None = None,
        trigger_source: str = "tick",
    ) -> dict[str, Any] | None:
        """Apply the job's ``missed_fire_policy`` to a stale ``next_run``.

        Returns the execution outcome when the policy fired the job
        (``fire_now``), else None (the schedule was advanced without
        running).  ``cutoff`` (used by catch-up) degrades ``fire_now`` to
        ``skip`` for firings older than the bound — a job missed by days
        is stale, not merely late.
        """
        job_id = row["id"]
        next_run = float(row.get("next_run") or now)
        lateness = now - next_run
        if self._is_blackout_day(row, now):
            self._skip_blackout(row, now)
            return None
        policy = str(row.get("missed_fire_policy") or "fire_now").strip().lower()
        if policy not in MISSED_FIRE_POLICIES:
            policy = "fire_now"
        too_stale = cutoff is not None and next_run < cutoff
        if policy == "fire_now" and not too_stale:
            # coalesce: one run now, no matter how many periods were missed
            _log.info("scheduler: job %s (%s) missed its firing by %ds — "
                      "firing now (missed_fire_policy=fire_now)",
                      job_id, row.get("name"), int(lateness))
            return self._execute(row, trigger_source=trigger_source)
        # skip / next_only / stale fire_now: advance without executing
        self._advance_past_missed(row, now, policy)
        reason = (f"skipped: firing missed by {int(lateness)}s "
                  f"(missed_fire_policy={policy}"
                  f"{', too stale to fire' if too_stale else ''})")
        self._record_run(job_id, now, 0.0, True, reason, 0, trigger_source)
        self._ledger("skipped", job_id,
                     f"{row.get('name')}: {reason}",
                     metadata={"policy": policy, "lateness_s": int(lateness)})
        _log.info("scheduler: job %s (%s) %s", job_id, row.get("name"), reason)
        return None

    def _advance_past_missed(self, row: dict[str, Any], now: float,
                             policy: str) -> None:
        """Move the schedule past a missed firing without executing.

        ``skip`` reschedules from *now*; ``next_only`` keeps the
        schedule's phase by stepping forward from the missed firing
        itself (the next firing after the missed one).  One-shot jobs
        simply disable — a missed one-shot never fires late.
        """
        kind = row["kind"]
        if kind == "at":
            next_run: float | None = None
            still_enabled = False
        elif kind == "every":
            interval = max(10.0, float(row["spec"] or 10.0))
            if policy == "skip":
                next_run, still_enabled = now + interval, True
            else:  # next_only — keep phase
                nxt = float(row.get("next_run") or now)
                while nxt <= now:
                    nxt += interval
                next_run, still_enabled = nxt, True
        elif kind == "cron":
            tz = str(row.get("timezone") or "")
            base = (datetime.fromtimestamp(now)
                    if policy == "skip"
                    else datetime.fromtimestamp(float(row.get("next_run") or now)))
            next_run = _next_cron(str(row["spec"]), base, tz)
            still_enabled = True
        elif kind == "rrule":
            tz = str(row.get("timezone") or "")
            base_ts = (now if policy == "skip"
                       else float(row.get("next_run") or now))
            nxt = _rrule_next(str(row["spec"]),
                              float(row.get("created_at") or now), base_ts, tz)
            next_run, still_enabled = ((nxt, True) if nxt is not None
                                       else (None, False))
        else:  # daily
            tz = str(row.get("timezone") or "")
            base = (datetime.fromtimestamp(now)
                    if policy == "skip"
                    else datetime.fromtimestamp(float(row.get("next_run") or now)))
            next_run = _next_daily(str(row["spec"]), base, tz)
            still_enabled = True
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET next_run = ?, enabled = ?, "
                "retry_count = 0, updated_at = ? WHERE id = ?",
                (next_run, 1 if still_enabled else 0, time.time(), row["id"]),
            )

    def _is_blackout_day(self, row: dict[str, Any], now: float) -> bool:
        """True when ``now`` falls on one of the job's blackout dates.

        Evaluated in the job's timezone (Quartz exclusion-calendar
        semantics): a due firing on a blackout day is skipped without
        running — the schedule advances, the skip is journaled.
        """
        try:
            dates = json.loads(row.get("blackout_dates") or "[]")
        except (ValueError, TypeError):
            return False
        if not dates:
            return False
        tz_name = str(row.get("timezone") or "").strip()
        try:
            tz = ZoneInfo(tz_name) if tz_name else None
        except ZoneInfoNotFoundError:
            tz = None
        today = datetime.fromtimestamp(now, tz=tz).strftime("%Y-%m-%d")
        return today in {str(d) for d in dates}

    def _skip_blackout(self, row: dict[str, Any], now: float) -> None:
        """Advance past a blackout-day firing without executing."""
        job_id = row["id"]
        next_run, still_enabled = self._next_after_run(row, now)
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET next_run = ?, enabled = ?, "
                "retry_count = 0, updated_at = ? WHERE id = ?",
                (next_run, 1 if still_enabled else 0, time.time(), job_id),
            )
        reason = "skipped: blackout date (exclusion calendar)"
        self._record_run(job_id, now, 0.0, True, reason, 0, "tick")
        self._ledger("skipped", job_id, f"{row.get('name')}: {reason}")
        _log.info("scheduler: job %s (%s) %s", job_id, row.get("name"), reason)

    def catch_up_on_startup(self, *, max_age_hours: float = 24.0) -> list[dict[str, Any]]:
        """Run jobs whose ``next_run`` passed while the bot was down.

        Each missed job's own ``missed_fire_policy`` decides what happens
        (``fire_now`` / ``skip`` / ``next_only``); ``max_age_hours`` is the
        staleness bound beyond which even ``fire_now`` degrades to
        ``skip``.  Returns the outcomes of the jobs that actually fired.
        """
        if self.db is None:
            return []
        now = time.time()
        cutoff = now - max(0.0, float(max_age_hours)) * 3600
        try:
            rows = self.db.query(
                "SELECT * FROM schedule_jobs WHERE enabled = 1 "
                "AND next_run IS NOT NULL AND next_run <= ? "
                "ORDER BY next_run ASC",
                (now,),
            )
        except Exception:  # noqa: BLE001
            return []
        results = []
        for row in rows:
            row = dict(row)
            if not self._dependency_ok(row):
                continue
            if self._is_blackout_day(row, now):
                self._skip_blackout(row, now)
                continue
            # every due job goes through its missed-fire policy — a merely
            # late job with the default fire_now policy simply fires, which
            # is the historical catch-up contract.
            try:
                outcome = self._apply_missed_fire(
                    row, now, cutoff=cutoff, trigger_source="catchup")
                if outcome is not None:
                    results.append(outcome)
            except Exception:  # noqa: BLE001
                _log.exception("scheduler catch-up crashed: %s", row.get("id"))
        return results

    # ── execution ────────────────────────────────────────────────────────────
    def tick(self) -> list[dict[str, Any]]:
        """Run everything due right now. Returns the outcomes (for tests too).

        Due jobs are dispatched onto a bounded worker pool
        (``max_concurrent`` threads) instead of running inline: jobs
        genuinely overlap now, so a stuck job no longer stalls the tick
        loop, the redelivery sweep, or the other due jobs — and
        ``overlap_policy=concurrent`` runs concurrently for real.  The
        pool is per-tick (no leaked threads, no interpreter-exit hangs);
        the tick still waits for every dispatched job and returns the
        outcomes in dispatch order.  Each job's wall clock is still capped
        by its ``run_timeout_s`` (else the scheduler default).

        Also drives the notification redelivery queue (throttled to
        ``redeliver_interval``) so a scheduled job whose delivery failed
        keeps retrying with backoff on its own — the scheduler no longer
        depends on the watch loop for that.
        """
        if self.db is None:
            return []
        self._last_tick_at = time.time()
        now = self._last_tick_at
        _tick_start_errors = self._tick_errors
        try:
            due = self.db.query(
                "SELECT * FROM schedule_jobs WHERE enabled = 1 AND next_run IS NOT NULL "
                "AND next_run <= ? ORDER BY next_run ASC",
                (now,),
            )
        except Exception:  # noqa: BLE001
            return []
        defer_heavy, pressure_reasons = self._pressure_gate()
        results: list[dict[str, Any]] = []
        runnable: list[dict[str, Any]] = []
        ran_job_ids: set[str] = set()
        for row in due:
            row = dict(row)
            job_id = row["id"]
            if not self._dependency_ok(row):
                # dependency hasn't succeeded yet — leave for a later tick
                continue
            if self._is_blackout_day(row, now):
                # exclusion calendar: skip without running, advance
                self._skip_blackout(row, now)
                continue
            # missed firing?  the job's own policy decides (fire/skip/advance)
            lateness = now - float(row.get("next_run") or now)
            if lateness > self.missed_grace_s:
                try:
                    outcome = self._apply_missed_fire(row, now)
                    if outcome is not None:
                        results.append(outcome)
                        ran_job_ids.add(job_id)
                except Exception:  # noqa: BLE001
                    self._tick_errors += 1
                    _log.exception("scheduler missed-fire handling crashed: %s",
                                   job_id)
                continue
            # overlap: a previous run of THIS job is still executing
            with self._lock:
                overlapping = job_id in self._active
            if overlapping:
                policy = str(row.get("overlap_policy") or "concurrent").strip().lower()
                if policy == "skip":
                    self._skip_overlap(row, now)
                    continue
                if policy == "queue":
                    # leave it for the next tick rather than dropping it
                    continue
                # concurrent: fall through and run alongside the active one
            # resource pressure: heavy jobs defer (never drop) this tick
            if defer_heavy and self._is_heavy(row):
                self._deferrals[job_id] = self._deferrals.get(job_id, 0) + 1
                n = self._deferrals[job_id]
                _log.warning(
                    "scheduler: deferring heavy job %s (%s, deferral #%d) — "
                    "resource pressure [%s]; stays pending, retried next tick",
                    job_id, row.get("name"), n,
                    "; ".join(pressure_reasons) if pressure_reasons else "no reasons")
                if n == 1 or n % 10 == 0:
                    self._ledger("deferred", job_id,
                                 f"{row.get('name')}: deferred under resource "
                                 f"pressure (#{n})",
                                 metadata={"reasons": pressure_reasons})
                continue
            runnable.append(row)
        if runnable:
            # the pool IS the max_concurrent bound: at most this many run
            # at once, the rest queue inside the pool; the tick waits for
            # all of them so the outcomes contract is unchanged.
            if self.db.is_memory:
                # SQLite :memory: databases are per-connection — a worker
                # thread would see an empty database.  In-memory DBs only
                # exist in tests/CLI; production always uses a file DB.
                # Run inline (serial) instead of dispatching.
                for row in runnable:
                    try:
                        results.append(self._execute(row))
                        ran_job_ids.add(row["id"])
                    except Exception:  # noqa: BLE001 - one job must not kill the tick
                        self._tick_errors += 1
                        _log.exception("scheduler job crashed: %s",
                                       row.get("id"))
            else:
                with self._lock:
                    self._running_jobs += len(runnable)
                try:
                    with ThreadPoolExecutor(
                            max_workers=self.max_concurrent,
                            thread_name_prefix="sched-job") as pool:
                        futures = [pool.submit(self._execute, row)
                                   for row in runnable]
                        for row, fut in zip(runnable, futures):
                            try:
                                results.append(fut.result())
                                ran_job_ids.add(row["id"])
                            except Exception:  # noqa: BLE001 - one job must not kill the tick
                                self._tick_errors += 1
                                _log.exception("scheduler job crashed: %s",
                                               row.get("id"))
                finally:
                    with self._lock:
                        self._running_jobs -= len(runnable)
        for job_id in ran_job_ids:
            self._prune_runs(job_id)
        self._maybe_redeliver()
        # Error-budget heartbeat: a clean tick is success; any job crash
        # or tick-level error is failure. Feeds real scheduler ratios.
        try:
            from ..core.error_system import heartbeat
            heartbeat("scheduler", self._tick_errors == _tick_start_errors)
        except Exception:  # noqa: BLE001 - telemetry never breaks the tick
            pass
        return results

    def _skip_overlap(self, row: dict[str, Any], now: float) -> None:
        """Drop this firing because the previous run is still active
        (``overlap_policy=skip``); the schedule advances normally."""
        job_id = row["id"]
        next_run, still_enabled = self._next_after_run(row, now)
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET next_run = ?, enabled = ?, "
                "updated_at = ? WHERE id = ?",
                (next_run, 1 if still_enabled else 0, time.time(), job_id),
            )
        reason = "skipped: previous run still active (overlap_policy=skip)"
        self._record_run(job_id, now, 0.0, True, reason, 0, "tick")
        self._ledger("skipped", job_id, f"{row.get('name')}: {reason}")
        _log.info("scheduler: job %s (%s) %s", job_id, row.get("name"), reason)

    def _record_run(self, job_id: str, started: float, seconds: float,
                    ok: bool, result: str, retry_attempt: int,
                    trigger_source: str) -> None:
        """Land one row in ``schedule_runs`` — the durable per-execution
        history.  Best-effort: a broken history table must never break a
        job's bookkeeping."""
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO schedule_runs
                       (id, job_id, started_at, finished_at, seconds, ok,
                        result, retry_attempt, trigger_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_id("run"), job_id, started, started + seconds,
                     seconds, 1 if ok else 0, (result or "")[:2000],
                     int(retry_attempt or 0), trigger_source),
                )
        except Exception:  # noqa: BLE001
            _log.debug("schedule_runs record failed for %s", job_id,
                       exc_info=True)

    def _prune_runs(self, job_id: str) -> None:
        """Trim ``schedule_runs`` to the newest ``run_history_limit`` rows.

        K8s ``successfulJobsHistoryLimit`` as per-job data: years of
        uptime must not grow the history table forever.  Best-effort —
        pruning must never break a job's bookkeeping.
        """
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "DELETE FROM schedule_runs WHERE job_id = ? AND id NOT IN "
                    "(SELECT id FROM schedule_runs WHERE job_id = ? "
                    "ORDER BY started_at DESC, rowid DESC LIMIT ?)",
                    (job_id, job_id, self.run_history_limit),
                )
        except Exception:  # noqa: BLE001
            _log.debug("schedule_runs prune failed for %s", job_id,
                       exc_info=True)

    def prune_run_history(self, ref: str | None = None) -> int:
        """Prune run history now.  Returns rows deleted (best-effort).

        ``ref`` limits pruning to one job (id/name/prefix); None prunes
        every job with history.
        """
        if self.db is None:
            return 0
        job_ids: list[str]
        if ref:
            row = self._find(ref)
            job_ids = [row["id"]] if row else []
        else:
            try:
                job_ids = [r["job_id"] for r in self.db.query(
                    "SELECT DISTINCT job_id FROM schedule_runs")]
            except Exception:  # noqa: BLE001
                return 0
        deleted = 0
        for job_id in job_ids:
            try:
                with self.db.transaction():
                    cur = self.db.execute(
                        "DELETE FROM schedule_runs WHERE job_id = ? AND id NOT IN "
                        "(SELECT id FROM schedule_runs WHERE job_id = ? "
                        "ORDER BY started_at DESC, rowid DESC LIMIT ?)",
                        (job_id, job_id, self.run_history_limit))
                    deleted += cur.rowcount or 0
            except Exception:  # noqa: BLE001
                _log.debug("schedule_runs prune failed for %s", job_id,
                           exc_info=True)
        return deleted

    def recent_runs(self, ref: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Per-execution history for one job, newest first."""
        row = self._find(ref)
        if row is None or self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT * FROM schedule_runs WHERE job_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (row["id"], max(1, int(limit))),
            )
        except Exception:  # noqa: BLE001
            return []
        return [
            {
                "job_id": r["job_id"],
                "started_at": r["started_at"],
                "seconds": round(float(r.get("seconds") or 0.0), 2),
                "ok": bool(r.get("ok")),
                "result": (r.get("result") or "")[:300],
                "retry_attempt": int(r.get("retry_attempt") or 0),
                "trigger_source": r.get("trigger_source") or "",
            }
            for r in rows
        ]

    def _maybe_redeliver(self) -> int:
        """Throttled sweep of the notification redelivery queue.

        Runs at most once per ``redeliver_interval`` — the backoff clock
        on each row decides what's actually attempted, so a dead channel
        is never hammered.  Never raises.
        """
        now = time.time()
        if now - self._last_redeliver_at < self.redeliver_interval:
            return 0
        self._last_redeliver_at = now
        try:
            redelivered = self.notifier.redeliver()
        except Exception:  # noqa: BLE001 - delivery is best-effort
            _log.debug("scheduler redeliver sweep failed", exc_info=True)
            return 0
        if redelivered:
            _log.info("scheduler redelivered %d queued notification(s)",
                      redelivered)
        return redelivered

    def run_now(self, ref: str) -> dict[str, Any]:
        row = self._find(ref)
        if not row:
            raise LookupError(f"no job {ref!r}")
        return self._execute(dict(row), trigger_source="manual")

    def _run_payload_guarded(
        self, row: dict[str, Any], timeout_s: float
    ) -> tuple[bool, str, bool]:
        """Execute the job payload with a wall-clock cap.

        Returns ``(ok, summary, timed_out)``.  With ``timeout_s <= 0``
        the payload runs inline (no cap); otherwise it runs on a daemon
        worker thread and a join past the cap abandons the worker — the
        tick loop is never blocked forever by a stuck tool or command.
        """
        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                payload = json.loads(row.get("payload") or "{}")
                if row["payload_kind"] == "message":
                    box["result"] = self._run_message(payload)
                elif row["payload_kind"] == "tool":
                    box["result"] = self._run_tool(payload)
                else:
                    box["result"] = self._run_command(payload)
            except Exception as exc:  # noqa: BLE001 - a job failure is a result
                box["error"] = exc

        if timeout_s <= 0:
            _target()
        else:
            worker = threading.Thread(
                target=_target, daemon=True,
                name=f"sched-job-{str(row.get('id') or '?')[:8]}")
            worker.start()
            worker.join(timeout_s)
            if worker.is_alive():
                return (False,
                        f"job failed: timed out after {timeout_s:g}s "
                        "(worker abandoned; the job is marked failed but the "
                        "stuck call may still be running)",
                        True)
        if "error" in box:
            return False, f"job failed: {box['error']}", False
        return True, str(box.get("result") or ""), False

    def _execute(self, row: dict[str, Any], *,
                 trigger_source: str = "tick") -> dict[str, Any]:
        started = time.time()
        now = started
        job_id = row["id"]
        job_name = row["name"]
        # a real run clears any pressure-deferral streak
        self._deferrals.pop(job_id, None)
        self._emit_bus("scheduler.job.started", {
            "job_id": job_id, "name": job_name, "kind": row.get("kind"),
            "payload_kind": row.get("payload_kind"),
            "trigger_source": trigger_source,
        })
        with self._lock:
            self._active.add(job_id)
        try:
            # the per-job cap wins; 0 falls back to the scheduler default
            timeout_s = float(row.get("run_timeout_s") or 0.0) or self.wall_seconds
            ok, summary, timed_out = self._run_payload_guarded(row, timeout_s)
        finally:
            with self._lock:
                self._active.discard(job_id)
        # -- retry policy -------------------------------------------------
        # On failure, if retries remain, schedule the retry with the job's
        # backoff strategy (constant | linear | exponential, capped,
        # jittered) instead of reporting the failure.  retry_count tracks
        # consecutive failures; it resets on success.
        max_retries = int(row.get("max_retries") or 0)
        retry_delay = max(10.0, float(row.get("retry_delay") or 60))
        backoff = str(row.get("backoff") or "exponential")
        backoff_max_s = max(60.0, float(row.get("backoff_max_s") or 3600.0))
        backoff_jitter = min(0.9, max(0.0, float(row.get("backoff_jitter") or 0.0)))
        retry_count = int(row.get("retry_count") or 0)
        will_retry = False
        if not ok and retry_count < max_retries:
            will_retry = True
            retry_count += 1
            delay = _backoff_delay(backoff, retry_delay, retry_count,
                                  max_s=backoff_max_s, jitter=backoff_jitter,
                                  rng=self._rng)
            next_run = now + delay
            still_enabled = True
            # mark the in-flight retry in the result so the owner sees it
            summary = (f"{summary} (retry {retry_count}/{max_retries} "
                       f"in {int(delay)}s, {backoff} backoff)")
        else:
            if ok:
                retry_count = 0  # success resets the streak
            next_run, still_enabled = self._next_after_run(row, now)
        # transition-only failure alerts: a job that keeps failing pages
        # ONCE (on the first failure of the streak), not on every run.
        # The previous result is read before the row is updated.
        prev_result = str(row.get("last_result") or "")
        prev_failed = prev_result.startswith("job failed")
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET last_run = ?, last_result = ?, next_run = ?, "
                "enabled = ?, retry_count = ?, updated_at = ? WHERE id = ?",
                (now, summary[:2000], next_run, 1 if still_enabled else 0,
                 retry_count, time.time(), row["id"]),
            )
        seconds = round(time.time() - started, 2)
        # durable per-execution history + unified autonomy ledger
        self._record_run(job_id, started, seconds, ok, summary,
                         retry_count if will_retry else 0, trigger_source)
        learned = ""
        if will_retry:
            learned = (f"failed; retry {retry_count}/{max_retries} scheduled "
                       f"({backoff} backoff)")
        elif not ok and not will_retry and max_retries:
            learned = "retries exhausted - owner alerted"
        elif timed_out:
            learned = "run timed out; worker abandoned"
        self._ledger("run", job_id,
                     f"{job_name}: {summary[:160]}",
                     cost_seconds=seconds, ok=ok, learned=learned,
                     metadata={"kind": row.get("kind"),
                               "payload_kind": row.get("payload_kind"),
                               "trigger_source": trigger_source,
                               "will_retry": will_retry,
                               "timed_out": timed_out})
        self._emit_bus("scheduler.job.finished", {
            "job_id": job_id, "name": job_name, "kind": row.get("kind"),
            "ok": ok, "seconds": seconds, "will_retry": will_retry,
            "retry_count": retry_count, "timed_out": timed_out,
            "trigger_source": trigger_source,
        })
        # alert the owner - durable + multi-channel via the notifier.
        # Failures alert on the failure transition only (a stuck job must
        # not page every run). Successes notify too, except for routine
        # tick/heartbeat jobs which would spam.  Message payloads are the
        # exception the other way round: the message itself was already
        # delivered to the owner's DM by _run_message - a second "job ran"
        # alert would double-send.
        lname = job_name.lower()
        is_routine_tick = 'tick' in lname or 'heartbeat' in lname or 'sweep' in lname
        alert = True
        if will_retry:
            alert = False  # retry pending - don't page until retries exhaust
        if not ok and prev_failed:
            alert = False  # still failing - the owner already knows
        if ok and is_routine_tick:
            alert = False  # routine tick succeeded - silent
        if ok and row.get("payload_kind") == "message":
            alert = False  # the message itself already went out
        if alert:
            try:
                self.notifier.publish(
                    "schedule",
                    f"{'✅' if ok else '❌'} scheduled: {job_name}",
                    summary[:1500],
                )
            except Exception:  # noqa: BLE001
                _log.debug("scheduler notification failed")
        return {
            "id": job_id, "name": job_name, "ok": ok,
            "result": summary[:2000], "seconds": seconds,
            "next_run": next_run, "will_retry": will_retry,
            "retry_count": retry_count, "timed_out": timed_out,
            "trigger_source": trigger_source,
        }

    def _run_message(self, payload: dict[str, Any]) -> str:
        """Deliver the literal message text to the owner's DM.

        The old behavior returned ``"sent: ..."`` without pushing anything
        to a chat surface.  Now the text goes through the Notifier — every
        live owner channel receives it — and the notification row stays in
        the DB (pending redelivery) when no channel is live.  ``force=True``:
        the owner explicitly scheduled this message, so it bypasses dedupe
        and the notifier feature flag.  Empty payloads still raise (that is
        a broken job, not a quiet no-op).
        """
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("empty message payload")
        outcome = self.notifier.publish("message", text, force=True)
        if outcome.get("delivered"):
            channel = outcome.get("channel") or "chat"
            return f"sent via {channel}: {text[:200]}"
        state = outcome.get("delivery_state") or "pending"
        _log.warning("scheduled message stored undelivered (%s): %r",
                     state, text[:80])
        return f"stored ({state}): {text[:200]}"

    def _run_tool(self, payload: dict[str, Any]) -> str:
        tool = str(payload.get("tool") or "").strip()
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("tool payload 'args' must be an object")
        tools = getattr(self.context, "tools", None)
        if tools is None:
            raise RuntimeError("no tool registry on this context")
        outcome = tools.call(tool, capabilities=CapabilitySet.all(), **args)
        if not outcome.ok:
            raise RuntimeError(f"tool {tool} failed: {getattr(outcome.error, 'message', outcome.error)}")
        value = outcome.value
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return f"{tool}: {text[:1500]}"

    def _run_command(self, payload: dict[str, Any]) -> str:
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ValueError("empty command payload")
        tools = getattr(self.context, "tools", None)
        if tools is None:
            raise RuntimeError("no tool registry on this context")
        outcome = tools.call(
            "shell_run", command=command,
            capabilities=CapabilitySet.all(),
        )
        if not outcome.ok:
            raise RuntimeError(f"command failed: {getattr(outcome.error, 'message', outcome.error)}")
        value = outcome.value or {}
        stdout = str(value.get("stdout") or "")[:1200]
        return f"exit={value.get('exit_code')}\n{stdout}".strip()

    # ── loop ─────────────────────────────────────────────────────────────────
    def start(self, *, catch_up: bool = True,
              catch_up_max_age_hours: float = 24.0) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        if catch_up:
            try:
                caught = self.catch_up_on_startup(
                    max_age_hours=catch_up_max_age_hours)
                if caught:
                    _log.info("scheduler caught up %d missed job(s)", len(caught))
            except Exception:  # noqa: BLE001
                _log.exception("scheduler catch-up failed")
        self._stop.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._thread.start()
        _log.info("scheduler started (tick %.0fs)", self.tick_seconds)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.tick_seconds + 5)
            self._thread = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        """Scheduler + delivery health snapshot.  Never raises.

        Powers ``/schedule health``: is the tick loop alive, when did it
        last run, what fires next, what ran last, and is anything stuck
        in the delivery queue (retryable / dead-lettered).  ``ok`` is
        False when the loop died after start or the delivery queue is
        backing up — the owner-visible answer to "is my scheduler alive".
        """
        out: dict[str, Any] = {
            "ok": True,
            "reasons": [],
            "notes": [],
            "running": False,
            "tick_seconds": self.tick_seconds,
            "started_at": self._started_at,
            "last_tick_at": self._last_tick_at,
            "last_tick_age_s": None,
            "tick_errors": self._tick_errors,
            "jobs": {"total": 0, "enabled": 0, "due_now": 0},
            "next_job": None,
            "last_job": None,
            "overlaps": {"active": [], "count": 0},
            "deferred": {},
            "recent_runs": [],
            "delivery": {
                "queue": {"retryable": 0, "held": 0, "dead": 0},
                "live_owner_channels": [],
                "termux_fallback": False,
            },
        }
        try:
            alive = self.running()
            out["running"] = alive
            # loop state: running | not-started (this instance never spun
            # the thread — CLI/test contexts) | stopped (start() was
            # called but the thread died — a real degradation).
            out["loop_state"] = ("running" if alive
                                 else ("stopped" if self._started_at
                                       else "not-started"))
            if self._last_tick_at:
                out["last_tick_age_s"] = round(time.time() - self._last_tick_at, 1)
            jobs = self.list_jobs()
            now = time.time()
            enabled = [j for j in jobs if j.get("enabled")]
            out["jobs"] = {
                "total": len(jobs),
                "enabled": len(enabled),
                "due_now": sum(1 for j in enabled
                               if (j.get("next_run") or 0) <= now),
            }
            upcoming = sorted(
                (j for j in enabled if j.get("next_run")),
                key=lambda j: j["next_run"])
            if upcoming:
                nxt = upcoming[0]
                out["next_job"] = {
                    "name": nxt["name"],
                    "next_run_iso": nxt.get("next_run_iso"),
                    "in_s": max(0, round(nxt["next_run"] - now)),
                }
            ran = [j for j in jobs if j.get("last_run")]
            if ran:
                last = max(ran, key=lambda j: j["last_run"])
                out["last_job"] = {
                    "name": last["name"],
                    "last_result": (last.get("last_result") or "")[:120],
                    "ok": not str(last.get("last_result") or "")
                            .startswith("job failed"),
                }
            try:
                out["delivery"]["queue"] = self.notifier.queue_depth()
            except Exception:  # noqa: BLE001 - queue depth is best-effort
                pass
            out["delivery"]["live_owner_channels"] = self._live_owner_channels()
            try:
                out["delivery"]["termux_fallback"] = bool(
                    self.notifier.termux_fallback_available())
            except Exception:  # noqa: BLE001
                pass
            # execution depth: what is running now, what pressure deferred,
            # and the last few executions across all jobs.
            try:
                with self._lock:
                    active_ids = sorted(self._active)
                by_id = {j["id"]: j.get("name", j["id"]) for j in jobs}
                out["overlaps"] = {
                    "active": [by_id.get(i, i) for i in active_ids],
                    "count": len(active_ids),
                }
            except Exception:  # noqa: BLE001
                pass
            out["deferred"] = dict(self._deferrals)
            try:
                rows = self.db.query(
                    "SELECT r.*, j.name AS job_name FROM schedule_runs r "
                    "LEFT JOIN schedule_jobs j ON j.id = r.job_id "
                    "ORDER BY r.started_at DESC LIMIT 5",
                ) if self.db is not None else []
                out["recent_runs"] = [
                    {
                        "job": r.get("job_name") or r["job_id"],
                        "ok": bool(r.get("ok")),
                        "seconds": round(float(r.get("seconds") or 0.0), 2),
                        "result": (r.get("result") or "")[:100],
                        "trigger_source": r.get("trigger_source") or "",
                    }
                    for r in rows
                ]
            except Exception:  # noqa: BLE001 - history is best-effort
                pass
            # verdict — "stopped" degrades; "not-started" is informational
            # (CLI/test contexts never spin the loop; the live runtime does).
            reasons = out["reasons"]
            if out["loop_state"] == "stopped":
                reasons.append("tick loop died after start() — jobs only "
                               "fire on manual /schedule run; restart the "
                               "runtime")
            if out["loop_state"] == "not-started":
                out["notes"].append("tick loop not started in this process — "
                                    "automatic firing needs the live runtime")
            if out["delivery"]["queue"]["dead"]:
                reasons.append(
                    f"{out['delivery']['queue']['dead']} notification(s) "
                    "dead-lettered (delivery retries exhausted)")
            if (out["delivery"]["queue"]["retryable"] >= 5
                    and not out["delivery"]["live_owner_channels"]
                    and not out["delivery"]["termux_fallback"]):
                reasons.append(
                    f"{out['delivery']['queue']['retryable']} notification(s) "
                    "stuck retrying with no live channel and no fallback")
            out["ok"] = not reasons
        except Exception as exc:  # noqa: BLE001 - health must never raise
            out["ok"] = False
            out["reasons"].append(f"health check errored: {exc}")
        return out

    def _live_owner_channels(self) -> list[str]:
        """Owner-chat platforms running in this session right now."""
        try:
            gw = getattr(self.notifier, "gateway", None)
            if gw is None:
                return []
            partner = getattr(getattr(self.context, "settings", None),
                              "partner", None)
            raw = str(getattr(partner, "owner_chats", "") or "") if partner else ""
            status = gw.status() or {}
            live = []
            for key in raw.split(","):
                plat, _, cid = key.strip().partition(":")
                if (plat and cid
                        and status.get(plat, {}).get("running_in_session")
                        and plat.strip().lower() not in live):
                    live.append(plat.strip().lower())
            return sorted(live)
        except Exception:  # noqa: BLE001 - best-effort
            return []

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the loop must outlive any tick bug
                _log.exception("scheduler tick crashed")
            self._stop.wait(self.tick_seconds)

    # ── formatting ───────────────────────────────────────────────────────────
    def _format_job(self, row: dict[str, Any]) -> dict[str, Any]:
        spec = row.get("spec", "")
        if row.get("kind") == "every":
            spec = f"every {int(float(spec))}s" if spec else spec
        elif row.get("kind") == "cron":
            spec = f"cron {spec}" if spec else spec
        elif row.get("kind") == "rrule":
            spec = f"rrule {spec}" if spec else spec
        tz = str(row.get("timezone") or "")
        if tz and row.get("kind") in ("daily", "cron", "rrule") and tz not in str(spec):
            spec = f"{spec} [{tz}]"
        next_run = row.get("next_run")
        try:
            blackout = json.loads(row.get("blackout_dates") or "[]")
            if not isinstance(blackout, list):
                blackout = []
        except (ValueError, TypeError):
            blackout = []
        return {
            "id": row["id"],
            "name": row.get("name", ""),
            "kind": row.get("kind", ""),
            "spec": spec,
            "payload_kind": row.get("payload_kind", ""),
            "enabled": bool(row.get("enabled")),
            "next_run": next_run,
            "next_run_iso": (
                datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M") if next_run else None
            ),
            "last_run": row.get("last_run"),
            "last_result": (row.get("last_result") or "")[:300],
            "timezone": tz,
            "depends_on": _parse_depends(row.get("depends_on")),
            "depends_policy": str(row.get("depends_policy") or "all_ok"),
            "max_retries": int(row.get("max_retries") or 0),
            "retry_delay": float(row.get("retry_delay") or 60),
            "retry_count": int(row.get("retry_count") or 0),
            "backoff": str(row.get("backoff") or "exponential"),
            "backoff_max_s": float(row.get("backoff_max_s") or 3600.0),
            "backoff_jitter": float(row.get("backoff_jitter") or 0.0),
            "missed_fire_policy": str(row.get("missed_fire_policy") or "fire_now"),
            "overlap_policy": str(row.get("overlap_policy") or "concurrent"),
            "run_timeout_s": float(row.get("run_timeout_s") or 0.0),
            "heavy": bool(int(row.get("heavy") or 0)),
            "weight": float(row.get("weight") or 0.0),
            "blackout_dates": blackout,
        }


# ── persona maintenance jobs ───────────────────────────────────────────────

PERSONA_REBUILD_JOB = "persona rebuild"
PERSONA_CURATE_JOB = "persona curate"


def ensure_persona_jobs(context: Any) -> dict[str, Any]:
    """Register the daily persona rebuild + curation jobs (idempotent).

    Moved here from memory.persona (2026-10-01): job registration belongs
    with the scheduler.  memory/ must not import agents/ (layering).
    """
    out: dict[str, Any] = {}
    for name, spec, action in (
            (PERSONA_REBUILD_JOB, "daily 03:30", "rebuild"),
            (PERSONA_CURATE_JOB, "daily 04:00", "curate")):
        try:
            sched = Scheduler(context)
            have = [j for j in sched.list_jobs() if j.get("name") == name]
        except Exception:  # noqa: BLE001 — scheduler table may not exist yet
            have = []
        if have:
            out[name] = {"already_scheduled": True}
            continue
        try:
            job = sched.add(name, spec, "tool",
                            {"tool": "memory",
                             "args": {"action": action}})
            out[name] = {"scheduled": True, "job_id": job.get("id")}
            _log.info("scheduled persona job: %s", name)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
    return out

