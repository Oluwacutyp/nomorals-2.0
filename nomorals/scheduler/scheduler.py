"""Scheduler implementation with cron, reminders, and event hooks.

Production-grade scheduler with:
- Persistent SQLite storage (+ per-execution history with retention)
- Calendar-grade cron (L/LW/nW/n#k/nL/?, day names) and RFC 5545 RRULE
- Timezone-aware scheduling (IANA zones, DST-safe)
- Missed-fire policies (fire_now / skip) with staleness bounds + startup catch-up
- Overlap protection, run timeouts, retries with exponential backoff,
  dead-lettering after max failures
- Background worker thread
- Event hooks with predicate operators (eq/gt/lt/contains/in/regex/…)
- Reminder lifecycle (open/closed/snoozed/missed)
- Goal-owned scheduled tasks
- Observability: health() snapshot + listener events
  (fired/failed/missed/dead/deferred)
- Resource-aware execution: heavy tasks are deferred (not dropped) while the
  machine is under pressure, light tasks always run.  The resource advisor is
  injected (dependency injection, never an upward import) — see
  :class:`ResourceAdvisor`.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .recurrence import (
    CronSpec,
    RRule,
    describe_cron as _describe_cron,
    describe_rrule as _describe_rrule,
    next_cron as _rec_next_cron,
    parse_natural_datetime as _parse_natural_datetime,
    parse_natural_schedule as _parse_natural_schedule,
    parse_rrule as _rec_parse_rrule,
)

__all__ = [
    "Scheduler",
    "CronJob",
    "Reminder",
    "EventHook",
    "ScheduledTask",
    "TaskStatus",
    "ResourceAdvisor",
    "HEAVY_WEIGHT_THRESHOLD",
    "MISSED_FIRE_POLICIES",
    "OVERLAP_POLICIES",
    "CONDITION_OPS",
]

#: Missed-fire policies for user-facing tasks.
MISSED_FIRE_POLICIES = frozenset({"fire_now", "skip"})
#: Overlap policies for user-facing tasks.
OVERLAP_POLICIES = frozenset({"concurrent", "skip", "queue"})
#: Predicate operators for event-hook conditions.
CONDITION_OPS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "icontains", "in", "nin",
    "startswith", "endswith", "regex",
})

_log = get_logger(__name__)


#: Metadata ``weight`` at or above this value marks a task as heavy.
#: Heavy tasks are deferred (skipped for the tick, retried later) while the
#: resource advisory reports pressure.  Default task weight is light.
HEAVY_WEIGHT_THRESHOLD = 1.0


class ResourceAdvisor(Protocol):
    """The resource-advisory interface the scheduler needs.

    Duck-typed: ``nomorals.os.resources.ResourceManager`` satisfies this
    protocol, and is injected by the caller (L7 entry points) — the
    scheduler (L4) must never import ``nomorals.os`` (L6) itself.
    """

    def consult(self, mission: Any = None) -> dict[str, Any]:
        """Advisory consult -> ``{ok, throttled, reasons, pressure, sample}``."""
        ...


class TaskStatus(str, Enum):
    """Status of a scheduled task."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SNOOZED = "snoozed"
    MISSED = "missed"      # due firing went stale past its staleness bound
    DEAD = "dead"          # retries exhausted — dead-lettered, needs a human
    PAUSED = "paused"      # suspended by the owner; resumes on demand


@dataclass
class ScheduledTask:
    """Base class for all scheduled items."""
    
    task_id: str
    task_type: str = ""  # cron, reminder, event_hook, one_time (set by subclass __post_init__)
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "action": self.action,
            "parameters": self.parameters,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class CronJob(ScheduledTask):
    """A recurring cron job."""
    
    cron_expr: str = ""  # e.g., "0 9 * * MON-FRI"
    next_run: float = 0.0
    last_run: Optional[float] = None
    run_count: int = 0
    max_runs: Optional[int] = None  # None = unlimited
    goal_id: Optional[str] = None  # If owned by a goal
    
    def __post_init__(self):
        self.task_type = "cron"


@dataclass
class Reminder(ScheduledTask):
    """A reminder with lifecycle."""
    
    text: str = ""
    due_at: float = 0.0
    user_id: str = ""
    snooze_count: int = 0
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        self.task_type = "reminder"
    
    def is_overdue(self) -> bool:
        """Check if reminder is overdue."""
        return self.status == TaskStatus.PENDING and time.time() > self.due_at


@dataclass
class EventHook(ScheduledTask):
    """Fires when specific data arrives."""
    
    event_type: str = ""  # e.g., "email_received", "price_drop"
    conditions: dict[str, Any] = field(default_factory=dict)
    trigger_count: int = 0
    max_triggers: Optional[int] = None
    
    def __post_init__(self):
        self.task_type = "event_hook"
    
    def matches(self, event_data: dict[str, Any]) -> bool:
        """Check if event data matches conditions.

        Bare values mean equality (``{"kind": "email"}``); dict values
        select a predicate operator (``{"price": {"lt": 100}}``).  See
        :func:`evaluate_conditions`.
        """
        return evaluate_conditions(self.conditions, event_data)


def _match_condition(actual: Any, expected: Any) -> bool:
    """One predicate: bare value = equality, dict = operator."""
    if not isinstance(expected, dict):
        return actual == expected
    for op, want in expected.items():
        op = str(op).lower()
        try:
            if op == "eq":
                if not (actual == want):
                    return False
            elif op == "ne":
                if not (actual != want):
                    return False
            elif op == "gt":
                if not (actual is not None and actual > want):
                    return False
            elif op == "gte":
                if not (actual is not None and actual >= want):
                    return False
            elif op == "lt":
                if not (actual is not None and actual < want):
                    return False
            elif op == "lte":
                if not (actual is not None and actual <= want):
                    return False
            elif op == "contains":
                if want not in (actual or ""):
                    return False
            elif op == "icontains":
                if str(want).lower() not in str(actual or "").lower():
                    return False
            elif op == "in":
                if actual not in (want or []):
                    return False
            elif op == "nin":
                if actual in (want or []):
                    return False
            elif op == "startswith":
                if not str(actual or "").startswith(str(want)):
                    return False
            elif op == "endswith":
                if not str(actual or "").endswith(str(want)):
                    return False
            elif op == "regex":
                if not re.search(str(want), str(actual or "")):
                    return False
            else:
                _log.warning("unknown event-hook condition op %r — no match", op)
                return False
        except TypeError:
            # incomparable types (e.g. None > 5) simply don't match
            return False
    return True


def evaluate_conditions(conditions: dict[str, Any],
                        event_data: dict[str, Any]) -> bool:
    """True when every condition matches the event data.

    Keys support dotted paths (``{"user.tier": "pro"}``).  A missing key
    never matches an operator predicate (only ``ne``/``nin`` against a
    present-but-unequal value can pass — absent stays absent).

    Boolean combinators (MongoDB query-language style) compose groups::

        {"$or": [{"kind": "email"}, {"price": {"lt": 100}}]}
        {"$and": [{"kind": "email"}, {"$not": {"spam": True}}]}
        {"$not": {"user.tier": "free"}}
    """
    for key, expected in conditions.items():
        if key == "$or":
            if (not isinstance(expected, list) or not any(
                    isinstance(sub, dict)
                    and evaluate_conditions(sub, event_data)
                    for sub in expected)):
                return False
            continue
        if key == "$and":
            if (not isinstance(expected, list) or not all(
                    isinstance(sub, dict)
                    and evaluate_conditions(sub, event_data)
                    for sub in expected)):
                return False
            continue
        if key == "$not":
            if (not isinstance(expected, dict)
                    or evaluate_conditions(expected, event_data)):
                return False
            continue
        actual: Any = event_data
        for part in str(key).split("."):
            if isinstance(actual, dict) and part in actual:
                actual = actual[part]
            else:
                actual = None
                break
        if not _match_condition(actual, expected):
            return False
    return True


def _interpret_naive(dt: datetime, tz: str | None) -> datetime:
    """Attach ``tz`` to a naive datetime (aware datetimes pass through)."""
    if dt.tzinfo is not None:
        return dt
    if tz:
        from zoneinfo import ZoneInfo
        return dt.replace(tzinfo=ZoneInfo(tz))
    return dt


def _validate_tz(tz: str | None) -> str | None:
    """Validate an IANA timezone name (None/'' stays None).  Raises ValueError."""
    tz = (tz or "").strip()
    if not tz:
        return None
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
    except Exception:
        raise ValueError(f"unknown timezone: {tz!r}")
    return tz


def _validate_policy(value: str, allowed: frozenset, name: str) -> str:
    """Validate a policy knob → normalized value.  Raises ValueError."""
    value = (value or "").strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _parse_quiet_hours(value: Any) -> tuple[str, str] | None:
    """Validate ``quiet_hours`` → (start, end) "HH:MM" strings.  Raises ValueError."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("-")]
        if len(parts) != 2:
            raise ValueError(
                f"quiet_hours must be 'HH:MM-HH:MM' or a (start, end) pair, "
                f"got {value!r}")
        start, end = parts
    else:
        try:
            start, end = value
        except (TypeError, ValueError):
            raise ValueError(
                f"quiet_hours must be 'HH:MM-HH:MM' or a (start, end) pair, "
                f"got {value!r}")
        start, end = str(start).strip(), str(end).strip()
    for label, hhmm in (("start", start), ("end", end)):
        if not re.fullmatch(r"\d{1,2}:\d{2}", hhmm):
            raise ValueError(
                f"quiet_hours {label} must be HH:MM, got {hhmm!r}")
        h, m = int(hhmm.split(":")[0]), int(hhmm.split(":")[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(
                f"quiet_hours {label} out of range: {hhmm!r}")
    return start, end


def _quiet_shift(now_ts: float, quiet: tuple[str, str] | None,
                 tz: str | None) -> float | None:
    """End-of-window timestamp when ``now_ts`` falls inside quiet hours.

    Returns None when not quiet right now.  Overnight windows
    (``23:00``–``07:00``) are handled.  Times are wall-clock in ``tz``
    (IANA) or server-local when None.
    """
    if not quiet:
        return None
    start_s, end_s = quiet
    sh, sm = int(start_s.split(":")[0]), int(start_s.split(":")[1])
    eh, em = int(end_s.split(":")[0]), int(end_s.split(":")[1])
    zone = None
    if tz:
        try:
            from zoneinfo import ZoneInfo
            zone = ZoneInfo(tz)
        except Exception:
            zone = None
    now_dt = datetime.fromtimestamp(now_ts, tz=zone)
    start_dt = now_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = now_dt.replace(hour=eh, minute=em, second=0, microsecond=0)
    if end_dt <= start_dt:
        # overnight window: which side of midnight is "now" on?
        if now_dt >= start_dt:
            end_dt += timedelta(days=1)    # today 23:00 → tomorrow 07:00
        else:
            start_dt -= timedelta(days=1)  # yesterday 23:00 → today 07:00
    if start_dt <= now_dt < end_dt:
        return end_dt.timestamp()
    return None


def _ts_date(ts: float, tz: str | None) -> Any:
    """Calendar date of a unix timestamp in ``tz`` (or server-local)."""
    zone = None
    if tz:
        try:
            from zoneinfo import ZoneInfo
            zone = ZoneInfo(tz)
        except Exception:
            zone = None
    return datetime.fromtimestamp(ts, tz=zone).date()


def _normalize_skip_dates(dates: Any) -> list[str]:
    """Skip-dates input → sorted ISO 'YYYY-MM-DD' strings.  Raises ValueError."""
    if not dates:
        return []
    if isinstance(dates, (str, datetime)):
        dates = [dates]
    out: set[str] = set()
    from datetime import date as _date
    for d in dates:
        if isinstance(d, datetime):
            out.add(d.date().isoformat())
        elif isinstance(d, _date):
            out.add(d.isoformat())
        elif isinstance(d, str):
            try:
                out.add(datetime.strptime(d.strip(), "%Y-%m-%d").date()
                        .isoformat())
            except ValueError:
                raise ValueError(
                    f"skip_dates entries must be YYYY-MM-DD, got {d!r}")
        else:
            raise ValueError(
                f"skip_dates entries must be dates or YYYY-MM-DD, got {d!r}")
    return sorted(out)


def _fmt_clock(dt: datetime) -> str:
    """'9:05 AM' — no platform-dependent strftime codes."""
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} " \
        f"{'AM' if dt.hour < 12 else 'PM'}"


def _fmt_next(ts: float, tz: str | None = None) -> str:
    """Human firing label: 'today 9:00 AM', 'tomorrow 6:30 PM', …"""
    zone = None
    if tz:
        try:
            from zoneinfo import ZoneInfo
            zone = ZoneInfo(tz)
        except Exception:
            zone = None
    dt = datetime.fromtimestamp(ts, tz=zone)
    today = datetime.now(tz=zone).date()
    if dt.date() == today:
        day = "today"
    elif dt.date() == today + timedelta(days=1):
        day = "tomorrow"
    elif dt.date() == today - timedelta(days=1):
        day = "yesterday"
    elif dt.date() < today + timedelta(days=7):
        day = dt.strftime("%A")  # "Monday"
    else:
        day = dt.strftime("%b %d")  # "Oct 21"
    return f"{day} {_fmt_clock(dt)}"


def _dtstart(now_ts: float, tz: str | None) -> datetime:
    """RRULE anchor datetime for ``now`` in ``tz`` (naive when no tz)."""
    if tz:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz))
    return datetime.fromtimestamp(now_ts)


def _next_rrule(rule_text: str, dtstart_ts: float, after_ts: float,
                tz: str | None) -> float | None:
    """Next RRULE occurrence strictly after ``after_ts`` (unix ts)."""
    rule = _rec_parse_rrule(rule_text, _dtstart(dtstart_ts, tz))
    return rule.next_timestamp(after_ts, tz or "")


class CronParser:
    """Parse cron expressions — now calendar-grade.

    Classic 5-field cron (minute hour day_of_month month day_of_week) plus
    the extended fields from :mod:`nomorals.scheduler.recurrence`:

    - ``L`` / ``LW`` / ``nW`` in day-of-month (last day, last weekday,
      nearest weekday)
    - ``n#k`` / ``nL`` in day-of-week (k-th / last *n* weekday)
    - ``?`` (no specific value), day names (``MON-FRI``)

    Examples:
        "0 9 * * *"        - 9am every day
        "0 9 * * MON-FRI"  - 9am weekdays
        "*/15 * * * *"     - Every 15 minutes
        "0 0 1 * *"        - Midnight on 1st of each month
        "0 9 L * *"        - 9am on the last day of each month
        "0 9 * * 5#3"      - 9am on the 3rd Friday of each month
    """

    DAY_NAMES = {
        "SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
        "THU": 4, "FRI": 5, "SAT": 6,
    }

    @classmethod
    def parse(cls, expr: str) -> dict[str, Any]:
        """Parse (and validate) a cron expression.

        Returns the classic component dict (``minute``/``hour``/
        ``day_of_month``/``month``/``day_of_week``); extended day fields
        that can't be expressed as sets come back as their original
        string, and ``spec`` always holds the parsed
        :class:`CronSpec`.  Raises ValueError on garbage.
        """
        spec = CronSpec(expr)  # validates eagerly
        out: dict[str, Any] = {"spec": spec}
        minute_s, hour_s, dom_s, month_s, dow_s = (
            p.strip() for p in spec.expression.split())
        out["minute"] = cls._parse_field(minute_s, 0, 59)
        out["hour"] = cls._parse_field(hour_s, 0, 23)
        out["month"] = cls._parse_field(month_s, 1, 12)
        out["day_of_month"] = (cls._parse_field(dom_s, 1, 31)
                               if _is_plain_field(dom_s) else dom_s)
        out["day_of_week"] = (cls._parse_day_field(dow_s)
                              if _is_plain_field(dow_s) else dow_s)
        return out

    @classmethod
    def _parse_field(cls, field: str, min_val: int, max_val: int) -> set[int]:
        """Parse a single classic cron field."""
        if field == "*":
            return set(range(min_val, max_val + 1))

        if field.startswith("*/"):
            step = int(field[2:])
            return set(range(min_val, max_val + 1, step))

        if "-" in field:
            start, end = field.split("-")
            start = cls._resolve_day_name(start)
            end = cls._resolve_day_name(end)
            return set(range(int(start), int(end) + 1))

        if "," in field:
            return {int(cls._resolve_day_name(x)) for x in field.split(",")}

        return {int(cls._resolve_day_name(field))}

    @classmethod
    def _resolve_day_name(cls, token: str) -> str:
        token = token.strip().upper()
        return str(cls.DAY_NAMES[token]) if token in cls.DAY_NAMES else token

    @classmethod
    def _parse_day_field(cls, field: str) -> set[int]:
        """Parse day of week field (supports names)."""
        if field == "*":
            return set(range(7))

        # Replace day names with numbers
        for name, num in cls.DAY_NAMES.items():
            field = field.replace(name, str(num))

        return cls._parse_field(field, 0, 6)

    @classmethod
    def next_run(cls, expr: str, after: float | None = None,
                 tz: str | None = None) -> float:
        """Next run time for a cron expression, strictly after ``after``.

        ``tz`` is an optional IANA timezone name — the wall-clock math
        runs in that zone (DST-safe).  Without it, server-local time.
        """
        after = after if after is not None else time.time()
        return _rec_next_cron(
            expr, datetime.fromtimestamp(after), tz or "")


def _is_plain_field(field: str) -> bool:
    """True when a cron field uses only classic syntax (set-expressible)."""
    return not re.search(r"[LW#?]", field.upper())


class Scheduler:
    """Production-grade scheduler with cron, reminders, and event hooks.

    Resource-aware: pass a ``resources`` advisor (e.g. an injected
    ``nomorals.os.resources.ResourceManager``) and each tick consults it
    once.  While the advisory reports pressure (throttled or not ok),
    heavy tasks are deferred to the next tick — never dropped — while
    light tasks (reminders, notifications) still run.  A task is heavy
    when its metadata sets ``heavy: true`` or a numeric ``weight`` at or
    above :data:`HEAVY_WEIGHT_THRESHOLD`; the default is light.
    """

    def __init__(
        self,
        db: Database,
        resources: ResourceAdvisor | None = None,
        run_history_limit: int = 100,
    ) -> None:
        self.db = db
        # Injected advisor (L6 wiring happens at the entry point, never here).
        # None = ungated scheduling, fully backwards compatible.
        self._resources = resources
        # task_id -> number of times deferred under resource pressure
        self._deferrals: dict[str, int] = {}
        self._action_handlers: dict[str, Callable] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        #: task_ids with an execution in flight right now (overlap policies).
        self._in_flight: set[str] = set()
        self._flight_lock = threading.Lock()
        #: event -> list of listener callbacks (sync or async).
        self._listeners: dict[str, list[Callable]] = {}
        self._listeners_lock = threading.Lock()
        #: per-task execution-history retention (task_runs table).
        self.run_history_limit = max(1, int(run_history_limit or 100))
        self._started_at: float | None = None
        self._tick_errors = 0
        self._ensure_schema()
        _log.info("Scheduler initialized")
    
    def _ensure_schema(self) -> None:
        """Create scheduler tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    parameters TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    task_id TEXT PRIMARY KEY,
                    cron_expr TEXT NOT NULL,
                    next_run REAL NOT NULL,
                    last_run REAL,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    max_runs INTEGER,
                    goal_id TEXT,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    task_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    due_at REAL NOT NULL,
                    user_id TEXT NOT NULL,
                    snooze_count INTEGER NOT NULL DEFAULT 0,
                    completed_at REAL,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS event_hooks (
                    task_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    conditions TEXT NOT NULL DEFAULT '{}',
                    trigger_count INTEGER NOT NULL DEFAULT 0,
                    max_triggers INTEGER,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON scheduled_tasks(status)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cron_next_run
                ON cron_jobs(next_run)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                ON reminders(due_at)
            """)

            # Per-execution history: one row per firing, pruned to
            # run_history_limit per task (observability + "what ran when").
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    seconds REAL NOT NULL DEFAULT 0,
                    ok INTEGER NOT NULL DEFAULT 0,
                    result TEXT NOT NULL DEFAULT '',
                    trigger_source TEXT NOT NULL DEFAULT ''
                )
            """)

            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_runs_task
                ON task_runs(task_id, started_at DESC)
            """)
    
    # ── Cron Jobs ────────────────────────────────────────────────────────────
    
    async def schedule_cron(
        self,
        task_id: str | None,
        cron_expr: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_runs: int | None = None,
        goal_id: str | None = None,
        heavy: bool = False,
        weight: float = 0.0,
        tz: str | None = None,
        missed_fire_policy: str = "fire_now",
        stale_after_s: float = 24 * 3600,
        overlap_policy: str = "skip",
        timeout_s: float = 0.0,
        max_attempts: int = 1,
        retry_base_s: float = 60.0,
        jitter_s: float = 0.0,
        pause_on_failure: bool = False,
        skip_dates: Any = None,
    ) -> CronJob:
        """Schedule a recurring cron job.

        Args:
            task_id: Unique task ID (generated if None)
            cron_expr: Cron expression (e.g., "0 9 * * MON-FRI"; extended:
                "0 9 L * *", "0 9 * * 5#3") — or natural language like
                "every weekday at 9am" (anything :func:`parse_natural_schedule`
                maps to cron; rrule-shaped phrases raise and point at
                :meth:`schedule_rrule`)
            action: Action to execute
            parameters: Action parameters
            max_runs: Maximum executions (None = unlimited)
            goal_id: Optional goal that owns this job
            heavy: Mark as heavy work (deferred while resources are pressured)
            weight: Numeric weight; >= HEAVY_WEIGHT_THRESHOLD counts as heavy
            tz: IANA timezone for the schedule (DST-safe); None = server local
            missed_fire_policy: "fire_now" (coalesce one run) or "skip"
            stale_after_s: a firing older than this is stale (skipped)
            overlap_policy: "concurrent" | "skip" | "queue" when a previous
                run is still in flight
            timeout_s: per-execution wall-clock cap (0 = no cap)
            max_attempts: tries before dead-lettering (>= 1)
            retry_base_s: base delay for exponential retry backoff
            jitter_s: uniform random delay in [0, jitter_s] added to every
                firing (systemd ``RandomizedDelaySec`` / Temporal ``--jitter``
                — spreads the thundering herd when many jobs share a slot)
            pause_on_failure: pause the schedule on first failure (Temporal
                ``--pause-on-failure``) instead of retry-spamming; resume with
                :meth:`resume`
            skip_dates: dates the job never fires on ("2026-12-25",
                ``datetime``/``date`` objects, or lists thereof — holidays,
                maintenance windows)

        Returns:
            CronJob object

        Raises:
            ValueError: bad expression/timezone/policy, or ``task_id``
                already exists (idempotent callers: reuse the id).
        """
        task_id = task_id or new_id("cron")
        self._raise_if_exists(task_id)
        parameters = parameters or {}
        tz = _validate_tz(tz)
        missed_fire_policy = _validate_policy(
            missed_fire_policy, MISSED_FIRE_POLICIES, "missed_fire_policy")
        overlap_policy = _validate_policy(
            overlap_policy, OVERLAP_POLICIES, "overlap_policy")
        skips = _normalize_skip_dates(skip_dates)
        cron_expr = self._resolve_cron_expr(cron_expr)
        metadata = {
            "heavy": heavy, "weight": weight, "tz": tz or "",
            "missed_fire_policy": missed_fire_policy,
            "stale_after_s": max(60.0, float(stale_after_s or 3600)),
            "overlap_policy": overlap_policy,
            "timeout_s": max(0.0, float(timeout_s or 0.0)),
            "max_attempts": max(1, int(max_attempts or 1)),
            "retry_base_s": max(5.0, float(retry_base_s or 60.0)),
            "jitter_s": max(0.0, float(jitter_s or 0.0)),
            "pause_on_failure": bool(pause_on_failure),
            "skip_dates": skips,
            "fail_count": 0,
            "recurrence": "cron",
        }

        # Validate cron expression (raises on garbage); tz-aware.
        next_run = self._next_valid(
            {"cron_expr": cron_expr, "task_id": task_id,
             "created_at": time.time()},
            metadata, time.time())

        # Create task
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'cron', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), time.time(), time.time(),
                  json.dumps(metadata)))
            
            self.db.execute("""
                INSERT INTO cron_jobs (task_id, cron_expr, next_run, last_run, run_count, max_runs, goal_id)
                VALUES (?, ?, ?, NULL, 0, ?, ?)
            """, (task_id, cron_expr, next_run, max_runs, goal_id))
        
        job = CronJob(
            task_id=task_id,
            action=action,
            parameters=parameters,
            metadata=metadata,
            cron_expr=cron_expr,
            next_run=next_run,
            max_runs=max_runs,
            goal_id=goal_id,
        )
        
        _log.info(f"Scheduled cron job: {task_id} ({cron_expr})")
        return job

    async def schedule_rrule(
        self,
        task_id: str | None,
        rrule: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_runs: int | None = None,
        goal_id: str | None = None,
        heavy: bool = False,
        weight: float = 0.0,
        tz: str | None = None,
        missed_fire_policy: str = "fire_now",
        stale_after_s: float = 24 * 3600,
        overlap_policy: str = "skip",
        timeout_s: float = 0.0,
        max_attempts: int = 1,
        retry_base_s: float = 60.0,
        jitter_s: float = 0.0,
        pause_on_failure: bool = False,
        skip_dates: Any = None,
    ) -> CronJob:
        """Schedule a recurring task from an RFC 5545 RRULE.

        Calendar-grade recurrence cron can't express, e.g.
        ``FREQ=MONTHLY;BYDAY=2TU`` (2nd Tuesday) or
        ``FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1`` (last Friday).  Stored in
        the cron_jobs table (RRULE text in ``cron_expr``, marker in
        metadata).  Same policy knobs as :meth:`schedule_cron`, plus
        natural-language schedules (``"every 2nd tuesday"``) via
        :func:`parse_natural_schedule`.
        """
        task_id = task_id or new_id("rrule")
        self._raise_if_exists(task_id)
        parameters = parameters or {}
        tz = _validate_tz(tz)
        missed_fire_policy = _validate_policy(
            missed_fire_policy, MISSED_FIRE_POLICIES, "missed_fire_policy")
        overlap_policy = _validate_policy(
            overlap_policy, OVERLAP_POLICIES, "overlap_policy")
        rule_text = (rrule or "").strip()
        if rule_text.upper().startswith("RRULE:"):
            rule_text = rule_text[6:].strip()
        if not rule_text or "FREQ=" not in rule_text.upper():
            # maybe natural language ("every 2nd tuesday", "last friday …")
            try:
                kind, expr = _parse_natural_schedule(rule_text)
            except ValueError:
                kind, expr = None, rule_text
            if kind == "rrule":
                rule_text = expr
            elif kind == "cron":
                raise ValueError(
                    f"{rrule!r} is a plain cron schedule — "
                    "use schedule_cron()")
        now_ts = time.time()
        dtstart = _dtstart(now_ts, tz)
        _rec_parse_rrule(rule_text, dtstart)  # raises on garbage
        skips = _normalize_skip_dates(skip_dates)
        metadata = {
            "heavy": heavy, "weight": weight, "tz": tz or "",
            "missed_fire_policy": missed_fire_policy,
            "stale_after_s": max(60.0, float(stale_after_s or 3600)),
            "overlap_policy": overlap_policy,
            "timeout_s": max(0.0, float(timeout_s or 0.0)),
            "max_attempts": max(1, int(max_attempts or 1)),
            "retry_base_s": max(5.0, float(retry_base_s or 60.0)),
            "jitter_s": max(0.0, float(jitter_s or 0.0)),
            "pause_on_failure": bool(pause_on_failure),
            "skip_dates": skips,
            "fail_count": 0,
            "recurrence": "rrule",
        }
        nxt = self._next_valid(
            {"cron_expr": rule_text, "task_id": task_id,
             "created_at": now_ts},
            metadata, now_ts)
        if nxt is None:
            raise ValueError(
                f"RRULE yields no future occurrences: {rrule!r}")
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'cron', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), now_ts, now_ts,
                  json.dumps(metadata)))
            self.db.execute("""
                INSERT INTO cron_jobs (task_id, cron_expr, next_run, last_run, run_count, max_runs, goal_id)
                VALUES (?, ?, ?, NULL, 0, ?, ?)
            """, (task_id, rule_text, nxt, max_runs, goal_id))
        job = CronJob(
            task_id=task_id, action=action, parameters=parameters,
            metadata=metadata, cron_expr=rule_text, next_run=nxt,
            max_runs=max_runs, goal_id=goal_id)
        _log.info(f"Scheduled rrule task: {task_id} ({rule_text})")
        return job

    async def cancel_cron(self, task_id: str) -> bool:
        """Cancel a cron job."""
        with self.db.transaction():
            self.db.execute(
                "UPDATE scheduled_tasks SET status = 'cancelled', updated_at = ? WHERE task_id = ?",
                (time.time(), task_id)
            )
        _log.info(f"Cancelled cron job: {task_id}")
        return True

    async def pause(self, task_id: str) -> bool:
        """Suspend a task without deleting it (K8s ``suspend``).

        Scheduling stops; in-flight runs are unaffected.  Works for
        cron, reminder, and one-time tasks.
        """
        return await self._set_status(task_id, TaskStatus.PAUSED,
                                      ("pending", "snoozed"))

    async def resume(self, task_id: str) -> bool:
        """Resume a paused task; recomputes a cron next_run from now."""
        row = self.db.query_one(
            "SELECT * FROM scheduled_tasks WHERE task_id = ?", (task_id,))
        if not row or row["status"] != TaskStatus.PAUSED.value:
            return False
        meta = self._row_metadata(row)
        if row["task_type"] == "cron":
            cron = self.db.query_one(
                "SELECT * FROM cron_jobs WHERE task_id = ?", (task_id,))
            if cron:
                nxt = self._next_recurrence(dict(cron), meta, time.time())
                if nxt is not None:
                    with self.db.transaction():
                        self.db.execute(
                            "UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                            (nxt, task_id))
        return await self._set_status(task_id, TaskStatus.PENDING, ("paused",))

    async def _set_status(self, task_id: str, status: TaskStatus,
                          from_statuses: tuple[str, ...]) -> bool:
        """Transition a task's status from one of ``from_statuses``."""
        allowed = ", ".join("?" for _ in from_statuses)
        with self.db.transaction():
            cur = self.db.execute(
                "UPDATE scheduled_tasks SET status = ?, updated_at = ? "
                f"WHERE task_id = ? AND status IN ({allowed})",
                (status.value, time.time(), task_id, *from_statuses))
            return bool(cur.rowcount)

    async def get_cron(self, task_id: str) -> CronJob | None:
        """Fetch one cron/rrule job by id."""
        row = self.db.query_one(
            """SELECT t.*, c.cron_expr, c.next_run, c.last_run, c.run_count,
                      c.max_runs, c.goal_id
               FROM scheduled_tasks t JOIN cron_jobs c ON t.task_id = c.task_id
               WHERE t.task_id = ?""", (task_id,))
        if not row:
            return None
        return CronJob(
            task_id=row["task_id"], action=row["action"],
            parameters=json.loads(row["parameters"]),
            status=TaskStatus(row["status"]),
            metadata=self._row_metadata(row),
            cron_expr=row["cron_expr"], next_run=row["next_run"],
            last_run=row["last_run"], run_count=row["run_count"],
            max_runs=row["max_runs"], goal_id=row["goal_id"])

    async def reschedule_cron(self, task_id: str, cron_expr: str | None = None,
                              rrule: str | None = None) -> CronJob | None:
        """Change a cron job's recurrence; recomputes next_run from now.

        Pass ``cron_expr`` or ``rrule`` (not both).  Returns the updated
        job, or None when the task doesn't exist / isn't a cron task.
        """
        if (cron_expr is None) == (rrule is None):
            raise ValueError("pass exactly one of cron_expr / rrule")
        row = await self.get_cron(task_id)
        if not row:
            return None
        meta = dict(row.metadata)
        tz = meta.get("tz") or None
        now_ts = time.time()
        if cron_expr is not None:
            new_expr = self._resolve_cron_expr(cron_expr)
            recurrence = "cron"
            nxt = self._next_valid(
                {"cron_expr": new_expr, "task_id": task_id,
                 "created_at": now_ts},
                {**meta, "recurrence": "cron"}, now_ts)
        else:
            rule_text = (rrule or "").strip()
            if rule_text.upper().startswith("RRULE:"):
                rule_text = rule_text[6:].strip()
            _rec_parse_rrule(rule_text, _dtstart(now_ts, tz))
            new_expr, recurrence = rule_text, "rrule"
            nxt = self._next_valid(
                {"cron_expr": rule_text, "task_id": task_id,
                 "created_at": now_ts},
                {**meta, "recurrence": "rrule"}, now_ts)
            if nxt is None:
                raise ValueError(
                    f"RRULE yields no future occurrences: {rrule!r}")
        meta["recurrence"] = recurrence
        with self.db.transaction():
            self.db.execute(
                "UPDATE cron_jobs SET cron_expr = ?, next_run = ? WHERE task_id = ?",
                (new_expr, nxt, task_id))
            self.db.execute(
                "UPDATE scheduled_tasks SET metadata = ?, updated_at = ? "
                "WHERE task_id = ?",
                (json.dumps(meta), now_ts, task_id))
        return await self.get_cron(task_id)

    def _next_recurrence(self, cron_row: dict[str, Any],
                         meta: dict[str, Any], after_ts: float) -> float | None:
        """Next firing for a cron/rrule row strictly after ``after_ts``."""
        tz = meta.get("tz") or None
        expr = cron_row["cron_expr"]
        if meta.get("recurrence") == "rrule":
            created = float(cron_row.get("created_at") or after_ts)
            task = self.db.query_one(
                "SELECT created_at FROM scheduled_tasks WHERE task_id = ?",
                (cron_row["task_id"],))
            if task and task["created_at"]:
                created = float(task["created_at"])
            return _next_rrule(expr, created, after_ts, tz)
        return CronParser.next_run(expr, after_ts, tz=tz)

    def _raise_if_exists(self, task_id: str) -> None:
        """Idempotent-add guard: a clear error beats sqlite IntegrityError."""
        row = self.db.query_one(
            "SELECT task_id FROM scheduled_tasks WHERE task_id = ?",
            (task_id,))
        if row:
            raise ValueError(f"task_id {task_id!r} already exists")

    @staticmethod
    def _resolve_cron_expr(cron_expr: str) -> str:
        """Accept a cron expression or natural language ('every weekday at
        9am').  Natural-language phrases that map to RRULE raise a
        ValueError pointing at :meth:`schedule_rrule`."""
        text = (cron_expr or "").strip()
        if not text:
            raise ValueError("empty cron expression")
        try:
            CronParser.parse(text)  # validates eagerly
            return text
        except ValueError:
            pass
        kind, expr = _parse_natural_schedule(text)  # raises when unknown
        if kind != "cron":
            raise ValueError(
                f"{text!r} needs an RRULE schedule — use schedule_rrule()")
        return expr

    @staticmethod
    def _apply_jitter(ts: float, meta: dict[str, Any]) -> float:
        """Uniform [0, jitter_s] delay per firing (Temporal --jitter /
        systemd RandomizedDelaySec gold).  0 (default) = exact."""
        jitter = max(0.0, float(meta.get("jitter_s") or 0.0))
        return ts + random.uniform(0, jitter) if jitter > 0 else ts

    def _next_valid(self, cron_row: dict[str, Any], meta: dict[str, Any],
                    after_ts: float, *, apply_jitter: bool = True,
                    ) -> float | None:
        """Next firing strictly after ``after_ts`` with skip-dates + jitter.

        Firings landing on a ``skip_dates`` calendar day (in the job's tz)
        are skipped forward (cap: 400 skips, then the series is treated as
        exhausted).  Jitter is applied last; ``preview()`` passes
        ``apply_jitter=False`` for nominal times.
        """
        tz = meta.get("tz") or None
        skips = set(meta.get("skip_dates") or [])
        nxt = self._next_recurrence(cron_row, meta, after_ts)
        guard = 0
        while nxt is not None and skips and guard < 400:
            if _ts_date(nxt, tz).isoformat() not in skips:
                break
            nxt = self._next_recurrence(cron_row, meta, nxt)
            guard += 1
        if nxt is None:
            return None
        return self._apply_jitter(nxt, meta) if apply_jitter else nxt

    async def list_cron_jobs(self, *, active_only: bool = True) -> list[CronJob]:
        """List all cron jobs."""
        query = """
            SELECT t.*, c.* FROM scheduled_tasks t
            JOIN cron_jobs c ON t.task_id = c.task_id
        """
        if active_only:
            query += " WHERE t.status IN ('pending', 'paused')"
        
        rows = self.db.query(query)
        
        jobs = []
        for row in rows:
            jobs.append(CronJob(
                task_id=row["task_id"],
                action=row["action"],
                parameters=json.loads(row["parameters"]),
                status=TaskStatus(row["status"]),
                metadata=self._row_metadata(row),
                cron_expr=row["cron_expr"],
                next_run=row["next_run"],
                last_run=row["last_run"],
                run_count=row["run_count"],
                max_runs=row["max_runs"],
                goal_id=row["goal_id"],
            ))
        
        return jobs
    
    # ── Reminders ────────────────────────────────────────────────────────────
    
    async def create_reminder(
        self,
        text: str,
        due_at: datetime | float | str,
        user_id: str,
        *,
        action: str = "send_reminder",
        parameters: dict[str, Any] | None = None,
        heavy: bool = False,
        weight: float = 0.0,
        tz: str | None = None,
        missed_fire_policy: str = "fire_now",
        stale_after_s: float = 6 * 3600,
        overlap_policy: str = "skip",
        timeout_s: float = 0.0,
        max_attempts: int = 1,
        retry_base_s: float = 60.0,
        nag_every_s: float = 0.0,
        max_nags: int = 0,
        quiet_hours: Any = None,
    ) -> Reminder:
        """Create a reminder.

        Args:
            text: Reminder text
            due_at: When reminder is due — a datetime, unix timestamp, or
                natural language ("in 20 minutes", "tomorrow at 8am",
                "next monday at 9").  Naive datetimes are interpreted in
                ``tz`` when given, else server-local.
            user_id: User to remind
            action: Action to execute (default: send_reminder)
            parameters: Additional parameters
            heavy: Mark as heavy work (deferred while resources are pressured)
            weight: Numeric weight; >= HEAVY_WEIGHT_THRESHOLD counts as heavy
            tz: IANA timezone for naive ``due_at`` datetimes
            missed_fire_policy: "fire_now" (late reminders still fire) or
                "skip" (drop them)
            stale_after_s: a reminder older than this is marked "missed"
            overlap_policy: "concurrent" | "skip" | "queue"
            timeout_s: per-execution wall-clock cap (0 = no cap)
            max_attempts: tries before dead-lettering (>= 1)
            retry_base_s: base delay for exponential retry backoff
            nag_every_s: re-fire every N seconds until acknowledged
                (Due-app "auto snooze"); 0 = fire once.  Each nag is a real
                firing (recorded + listener event) until ``max_nags`` or
                :meth:`complete_reminder`.
            max_nags: cap on nag re-fires (0 with nag_every_s set = 1 nag)
            quiet_hours: "23:00-07:00" or ("23:00", "07:00") — a firing
                inside the window is shifted to the window end (Do Not
                Disturb), recorded, and a ``deferred`` event fires.

        Returns:
            Reminder object
        """
        task_id = new_id("reminder")
        parameters = parameters or {}
        parameters.setdefault("text", text)
        parameters.setdefault("user_id", user_id)
        tz = _validate_tz(tz)
        missed_fire_policy = _validate_policy(
            missed_fire_policy, MISSED_FIRE_POLICIES, "missed_fire_policy")
        overlap_policy = _validate_policy(
            overlap_policy, OVERLAP_POLICIES, "overlap_policy")
        quiet = _parse_quiet_hours(quiet_hours)
        nag_every_s = max(0.0, float(nag_every_s or 0.0))
        max_nags = max(0, int(max_nags or 0))
        if nag_every_s > 0 and max_nags == 0:
            max_nags = 1
        metadata = {
            "heavy": heavy, "weight": weight, "tz": tz or "",
            "missed_fire_policy": missed_fire_policy,
            "stale_after_s": max(60.0, float(stale_after_s or 3600)),
            "overlap_policy": overlap_policy,
            "timeout_s": max(0.0, float(timeout_s or 0.0)),
            "max_attempts": max(1, int(max_attempts or 1)),
            "retry_base_s": max(5.0, float(retry_base_s or 60.0)),
            "fail_count": 0,
            "nag_every_s": nag_every_s,
            "max_nags": max_nags,
            "nag_count": 0,
            "quiet_hours": list(quiet) if quiet else [],
        }

        if isinstance(due_at, str):
            due_at = _parse_natural_datetime(due_at, tz=tz).timestamp()
        elif isinstance(due_at, datetime):
            due_at = _interpret_naive(due_at, tz).timestamp()
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'reminder', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), time.time(), time.time(),
                  json.dumps(metadata)))
            
            self.db.execute("""
                INSERT INTO reminders (task_id, text, due_at, user_id, snooze_count, completed_at)
                VALUES (?, ?, ?, ?, 0, NULL)
            """, (task_id, text, due_at, user_id))
        
        reminder = Reminder(
            task_id=task_id,
            action=action,
            parameters=parameters,
            metadata=metadata,
            text=text,
            due_at=due_at,
            user_id=user_id,
        )
        
        _log.info(f"Created reminder: {text} (due: {datetime.fromtimestamp(due_at)})")
        return reminder
    
    async def complete_reminder(self, task_id: str) -> bool:
        """Mark reminder as completed."""
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                UPDATE scheduled_tasks SET status = 'completed', updated_at = ? WHERE task_id = ?
            """, (now, task_id))
            self.db.execute("""
                UPDATE reminders SET completed_at = ? WHERE task_id = ?
            """, (now, task_id))
        _log.info(f"Completed reminder: {task_id}")
        return True
    
    async def snooze_reminder(
        self,
        task_id: str,
        minutes: int = 15,
    ) -> bool:
        """Snooze a reminder."""
        new_due = time.time() + (minutes * 60)
        now = time.time()
        
        with self.db.transaction():
            self.db.execute("""
                UPDATE reminders SET due_at = ?, snooze_count = snooze_count + 1 WHERE task_id = ?
            """, (new_due, task_id))
            self.db.execute("""
                UPDATE scheduled_tasks SET status = 'snoozed', updated_at = ? WHERE task_id = ?
            """, (now, task_id))
        
        _log.info(f"Snoozed reminder {task_id} for {minutes} minutes")
        return True
    
    async def list_reminders(
        self,
        user_id: str,
        *,
        include_completed: bool = False,
    ) -> list[Reminder]:
        """List reminders for a user."""
        query = """
            SELECT t.*, r.* FROM scheduled_tasks t
            JOIN reminders r ON t.task_id = r.task_id
            WHERE r.user_id = ?
        """
        params: list[Any] = [user_id]
        
        if not include_completed:
            query += " AND t.status IN ('pending', 'snoozed')"
        
        query += " ORDER BY r.due_at ASC"
        
        rows = self.db.query(query, params)
        
        reminders = []
        for row in rows:
            reminders.append(Reminder(
                task_id=row["task_id"],
                action=row["action"],
                parameters=json.loads(row["parameters"]),
                status=TaskStatus(row["status"]),
                text=row["text"],
                due_at=row["due_at"],
                user_id=row["user_id"],
                snooze_count=row["snooze_count"],
                completed_at=row["completed_at"],
            ))
        
        return reminders
    
    # ── Event Hooks ──────────────────────────────────────────────────────────
    
    async def create_event_hook(
        self,
        event_type: str,
        conditions: dict[str, Any],
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_triggers: int | None = None,
        heavy: bool = False,
        weight: float = 0.0,
        timeout_s: float = 0.0,
        max_attempts: int = 1,
        overlap_policy: str = "skip",
        cooldown_s: float = 0.0,
    ) -> EventHook:
        """Create an event hook.

        Args:
            event_type: Event type to listen for
            conditions: Conditions that must match.  Bare values mean
                equality (``{"kind": "email"}``); dict values select a
                predicate operator — ``eq``/``ne``/``gt``/``gte``/``lt``/
                ``lte``/``contains``/``icontains``/``in``/``nin``/
                ``startswith``/``endswith``/``regex``
                (``{"price": {"lt": 100}}``).  Keys support dotted paths
                (``{"user.tier": "pro"}``).
            action: Action to execute when triggered
            parameters: Action parameters
            max_triggers: Maximum triggers (None = unlimited)
            heavy: Mark as heavy work (deferred while resources are pressured)
            weight: Numeric weight; >= HEAVY_WEIGHT_THRESHOLD counts as heavy
            timeout_s: per-execution wall-clock cap (0 = no cap)
            max_attempts: failures before dead-lettering (>= 1)
            overlap_policy: "concurrent" | "skip" | "queue" when a previous
                hook run is still in flight
            cooldown_s: debounce — at most one firing per this many seconds;
                excess triggers are recorded as throttled (never silently
                dropped)

        Returns:
            EventHook object
        """
        task_id = new_id("hook")
        parameters = parameters or {}
        overlap_policy = _validate_policy(
            overlap_policy, OVERLAP_POLICIES, "overlap_policy")
        metadata = {
            "heavy": heavy, "weight": weight,
            "timeout_s": max(0.0, float(timeout_s or 0.0)),
            "max_attempts": max(1, int(max_attempts or 1)),
            "overlap_policy": overlap_policy,
            "cooldown_s": max(0.0, float(cooldown_s or 0.0)),
            "last_trigger_ts": 0.0,
            "fail_count": 0,
        }

        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'event_hook', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), time.time(), time.time(),
                  json.dumps(metadata)))

            self.db.execute("""
                INSERT INTO event_hooks (task_id, event_type, conditions, trigger_count, max_triggers)
                VALUES (?, ?, ?, 0, ?)
            """, (task_id, event_type, json.dumps(conditions), max_triggers))

        hook = EventHook(
            task_id=task_id,
            action=action,
            parameters=parameters,
            metadata=metadata,
            event_type=event_type,
            conditions=conditions,
            max_triggers=max_triggers,
        )

        _log.info(f"Created event hook: {event_type} -> {action}")
        return hook

    async def trigger_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
    ) -> list[str]:
        """Trigger an event and fire matching hooks.

        Args:
            event_type: Event type
            event_data: Event data

        Returns:
            List of triggered hook IDs
        """
        # Find matching hooks
        rows = self.db.query("""
            SELECT h.*, t.action, t.parameters, t.metadata FROM event_hooks h
            JOIN scheduled_tasks t ON h.task_id = t.task_id
            WHERE h.event_type = ? AND t.status = 'pending'
        """, (event_type,))

        triggered = []
        now = time.time()

        for row in rows:
            row = dict(row)
            conditions = json.loads(row["conditions"])
            if not evaluate_conditions(conditions, event_data or {}):
                continue
            task_id = row["task_id"]
            meta = self._row_metadata(row)
            # cooldown debounce: at most one firing per window
            cooldown = max(0.0, float(meta.get("cooldown_s") or 0.0))
            last_ts = float(meta.get("last_trigger_ts") or 0.0)
            if cooldown > 0 and last_ts and now - last_ts < cooldown:
                wait = int(cooldown - (now - last_ts))
                self._record_run(
                    task_id, now, 0.0, True,
                    f"throttled: cooldown {cooldown:g}s "
                    f"({wait}s remaining)", "event")
                self._emit("throttled", {"task_id": task_id,
                                         "action": row["action"],
                                         "event_type": event_type,
                                         "cooldown_s": cooldown,
                                         "retry_in_s": wait})
                _log.info("hook %s throttled (cooldown %ds)",
                          task_id, wait)
                continue
            if self._overlap_decision(task_id, meta) != "run":
                _log.info("hook %s skipped (overlap)", task_id)
                continue
            action = row["action"]
            parameters = json.loads(row["parameters"])
            timeout_s = float(meta.get("timeout_s") or 0.0)
            started = time.time()
            self._mark_flight(task_id, True)
            try:
                await self._run_guarded(action, parameters, timeout_s)
            except Exception as exc:  # noqa: BLE001
                seconds = time.time() - started
                self._mark_flight(task_id, False)
                self._record_run(task_id, started, seconds, False,
                                 f"hook failed: {str(exc)[:300]}", "event")
                fail_count = int(meta.get("fail_count") or 0) + 1
                max_attempts = max(1, int(meta.get("max_attempts") or 1))
                if fail_count >= max_attempts:
                    with self.db.transaction():
                        self.db.execute(
                            "UPDATE scheduled_tasks SET status = 'dead', "
                            "updated_at = ? WHERE task_id = ?",
                            (time.time(), task_id))
                    self._emit("dead", {"task_id": task_id, "action": action,
                                        "fail_count": fail_count,
                                        "error": str(exc)[:300]})
                    _log.error("hook %s dead-lettered after %d failures",
                               task_id, fail_count)
                else:
                    self._write_meta(task_id, {"fail_count": fail_count})
                    self._emit("failed", {"task_id": task_id, "action": action,
                                          "fail_count": fail_count,
                                          "error": str(exc)[:300]})
                continue
            seconds = time.time() - started
            self._mark_flight(task_id, False)
            meta_patch: dict[str, Any] = {"last_trigger_ts": time.time()}
            if int(meta.get("fail_count") or 0):
                meta_patch["fail_count"] = 0
            self._write_meta(task_id, meta_patch)
            self._record_run(task_id, started, seconds, True,
                             f"event {event_type}", "event")

            # Update trigger count
            with self.db.transaction():
                self.db.execute("""
                    UPDATE event_hooks SET trigger_count = trigger_count + 1 WHERE task_id = ?
                """, (task_id,))

                # Check if max triggers reached
                if row["max_triggers"] and (row["trigger_count"] + 1) >= row["max_triggers"]:
                    self.db.execute("""
                        UPDATE scheduled_tasks SET status = 'completed' WHERE task_id = ?
                    """, (task_id,))

            triggered.append(task_id)
            self._emit("fired", {"task_id": task_id, "action": action,
                                 "seconds": round(seconds, 2),
                                 "event_type": event_type})
            _log.info(f"Triggered event hook: {task_id}")

        return triggered
    
    # ── One-Time Tasks ───────────────────────────────────────────────────────
    
    async def schedule_once(
        self,
        task_id: str | None,
        run_at: datetime | float | str,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        heavy: bool = False,
        weight: float = 0.0,
        tz: str | None = None,
        missed_fire_policy: str = "fire_now",
        stale_after_s: float = 3600,
        overlap_policy: str = "skip",
        timeout_s: float = 0.0,
        max_attempts: int = 1,
        retry_base_s: float = 60.0,
        quiet_hours: Any = None,
    ) -> ScheduledTask:
        """Schedule a one-time task.

        Args:
            task_id: Unique task ID
            run_at: When to run — a datetime, unix timestamp, or natural
                language ("in 20 minutes", "tomorrow at 8am").  Naive
                datetimes are interpreted in ``tz``.
            action: Action to execute
            parameters: Action parameters
            heavy: Mark as heavy work (deferred while resources are pressured)
            weight: Numeric weight; >= HEAVY_WEIGHT_THRESHOLD counts as heavy
            tz: IANA timezone for naive ``run_at`` datetimes
            missed_fire_policy: "fire_now" or "skip" when the time passed
                while down
            stale_after_s: older than this → marked "missed"
            overlap_policy: "concurrent" | "skip" | "queue"
            timeout_s: per-execution wall-clock cap (0 = no cap)
            max_attempts: tries before dead-lettering (>= 1)
            retry_base_s: base delay for exponential retry backoff
            quiet_hours: "23:00-07:00" or ("23:00", "07:00") — a firing
                inside the window is shifted to the window end.

        Returns:
            ScheduledTask object

        Raises:
            ValueError: bad time/policy, or ``task_id`` already exists.
        """
        task_id = task_id or new_id("task")
        self._raise_if_exists(task_id)
        parameters = parameters or {}
        tz = _validate_tz(tz)
        missed_fire_policy = _validate_policy(
            missed_fire_policy, MISSED_FIRE_POLICIES, "missed_fire_policy")
        overlap_policy = _validate_policy(
            overlap_policy, OVERLAP_POLICIES, "overlap_policy")
        quiet = _parse_quiet_hours(quiet_hours)

        if isinstance(run_at, str):
            run_at = _parse_natural_datetime(run_at, tz=tz).timestamp()
        elif isinstance(run_at, datetime):
            run_at = _interpret_naive(run_at, tz).timestamp()

        metadata = {
            "run_at": run_at, "heavy": heavy, "weight": weight, "tz": tz or "",
            "missed_fire_policy": missed_fire_policy,
            "stale_after_s": max(60.0, float(stale_after_s or 3600)),
            "overlap_policy": overlap_policy,
            "timeout_s": max(0.0, float(timeout_s or 0.0)),
            "max_attempts": max(1, int(max_attempts or 1)),
            "retry_base_s": max(5.0, float(retry_base_s or 60.0)),
            "fail_count": 0,
            "quiet_hours": list(quiet) if quiet else [],
        }
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'one_time', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), time.time(), time.time(),
                  json.dumps(metadata)))
        
        task = ScheduledTask(
            task_id=task_id,
            task_type="one_time",
            action=action,
            parameters=parameters,
            metadata=metadata,
        )
        
        _log.info(f"Scheduled one-time task: {task_id} at {datetime.fromtimestamp(run_at)}")
        return task
    
    # ── Task management (unified across types) ───────────────────────────────

    def _hydrate(self, row: dict[str, Any]) -> ScheduledTask:
        """scheduled_tasks row → CronJob / Reminder / EventHook / ScheduledTask."""
        meta = self._row_metadata(row)
        try:
            params = json.loads(row.get("parameters") or "{}")
        except (ValueError, TypeError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        base: dict[str, Any] = {
            "task_id": row["task_id"],
            "action": row.get("action") or "",
            "parameters": params,
            "status": TaskStatus(row.get("status") or "pending"),
            "metadata": meta,
            "created_at": float(row.get("created_at") or 0.0),
            "updated_at": float(row.get("updated_at") or 0.0),
        }
        ttype = row.get("task_type")
        if ttype == "cron":
            c = self.db.query_one(
                "SELECT * FROM cron_jobs WHERE task_id = ?",
                (row["task_id"],)) or {}
            return CronJob(
                **base, cron_expr=c.get("cron_expr") or "",
                next_run=float(c.get("next_run") or 0.0),
                last_run=c.get("last_run"),
                run_count=int(c.get("run_count") or 0),
                max_runs=c.get("max_runs"), goal_id=c.get("goal_id"))
        if ttype == "reminder":
            r = self.db.query_one(
                "SELECT * FROM reminders WHERE task_id = ?",
                (row["task_id"],)) or {}
            return Reminder(
                **base, text=r.get("text") or "",
                due_at=float(r.get("due_at") or 0.0),
                user_id=r.get("user_id") or "",
                snooze_count=int(r.get("snooze_count") or 0),
                completed_at=r.get("completed_at"))
        if ttype == "event_hook":
            h = self.db.query_one(
                "SELECT * FROM event_hooks WHERE task_id = ?",
                (row["task_id"],)) or {}
            try:
                conds = json.loads(h.get("conditions") or "{}")
            except (ValueError, TypeError):
                conds = {}
            return EventHook(
                **base, event_type=h.get("event_type") or "",
                conditions=conds if isinstance(conds, dict) else {},
                trigger_count=int(h.get("trigger_count") or 0),
                max_triggers=h.get("max_triggers"))
        base["task_type"] = ttype or "one_time"
        return ScheduledTask(**base)

    async def get_task(self, task_id: str) -> ScheduledTask | None:
        """Fetch any task by id — cron, reminder, event hook, or one-time."""
        row = self.db.query_one(
            "SELECT * FROM scheduled_tasks WHERE task_id = ?", (task_id,))
        return self._hydrate(dict(row)) if row else None

    async def list_tasks(self, *, active_only: bool = True,
                         task_type: str | None = None,
                         ) -> list[ScheduledTask]:
        """List tasks across all types, newest first.

        ``active_only`` keeps pending/paused/snoozed; ``task_type`` filters
        to one of "cron" | "reminder" | "event_hook" | "one_time".
        """
        rows = self.db.query(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC")
        out: list[ScheduledTask] = []
        for r in rows:
            r = dict(r)
            if task_type and r.get("task_type") != task_type:
                continue
            if active_only and r.get("status") not in (
                    "pending", "paused", "snoozed"):
                continue
            out.append(self._hydrate(r))
        return out

    def _delete_task_sync(self, task_id: str) -> bool:
        row = self.db.query_one(
            "SELECT task_type FROM scheduled_tasks WHERE task_id = ?",
            (task_id,))
        if not row:
            return False
        table = {"cron": "cron_jobs", "reminder": "reminders",
                 "event_hook": "event_hooks"}.get(row["task_type"])
        with self.db.transaction():
            if table:
                self.db.execute(
                    f"DELETE FROM {table} WHERE task_id = ?", (task_id,))
            self.db.execute(
                "DELETE FROM task_runs WHERE task_id = ?", (task_id,))
            self.db.execute(
                "DELETE FROM scheduled_tasks WHERE task_id = ?", (task_id,))
        _log.info("Deleted task: %s", task_id)
        return True

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task and its run history.  Returns False when unknown."""
        return self._delete_task_sync(task_id)

    async def update_task(self, task_id: str, *,
                          action: str | None = None,
                          parameters: dict[str, Any] | None = None) -> bool:
        """Change a task's action and/or parameters.  Schedule untouched."""
        row = self.db.query_one(
            "SELECT task_id, action, parameters FROM scheduled_tasks "
            "WHERE task_id = ?", (task_id,))
        if not row:
            return False
        new_action = action if action is not None else row["action"]
        new_params = (parameters if parameters is not None
                      else json.loads(row["parameters"] or "{}"))
        with self.db.transaction():
            self.db.execute(
                "UPDATE scheduled_tasks SET action = ?, parameters = ?, "
                "updated_at = ? WHERE task_id = ?",
                (new_action, json.dumps(new_params), time.time(), task_id))
        return True

    def prune_terminal(self, older_than_s: float = 30 * 86400) -> int:
        """Delete terminal tasks (completed/cancelled/missed/dead) older
        than ``older_than_s`` — the K8s ``ttlSecondsAfterFinished`` /
        history-limit gold.  Run history goes with them.  Returns the
        number of tasks removed.  Never raises."""
        try:
            cutoff = time.time() - max(0.0, float(older_than_s))
            rows = self.db.query(
                "SELECT task_id FROM scheduled_tasks "
                "WHERE status IN ('completed', 'cancelled', 'missed', 'dead') "
                "AND updated_at < ?",
                (cutoff,))
            n = 0
            for r in rows:
                if self._delete_task_sync(r["task_id"]):
                    n += 1
            if n:
                _log.info("pruned %d terminal task(s)", n)
            return n
        except Exception:  # noqa: BLE001 - pruning is housekeeping
            _log.debug("prune_terminal failed", exc_info=True)
            return 0

    def preview(self, task_id: str, n: int = 5) -> list[float]:
        """Next ``n`` firing timestamps (unix) for a task.

        Cron/RRULE jobs expand their recurrence (skip-dates applied;
        jitter NOT applied — preview shows nominal times).  Reminders and
        one-time tasks return their single due time while still pending.
        Event hooks are event-driven → ``[]``.  ``[]`` when unknown.
        """
        n = max(1, min(int(n or 5), 100))
        row = self.db.query_one(
            "SELECT * FROM scheduled_tasks WHERE task_id = ?", (task_id,))
        if not row:
            return []
        row = dict(row)
        meta = self._row_metadata(row)
        ttype = row.get("task_type")
        if ttype == "cron":
            c = self.db.query_one(
                "SELECT * FROM cron_jobs WHERE task_id = ?", (task_id,)) or {}
            expr = c.get("cron_expr") or ""
            if not expr:
                return []
            cron_row = {"cron_expr": expr, "task_id": task_id,
                        "created_at": row.get("created_at")}
            out: list[float] = []
            cursor = time.time()
            for _ in range(n):
                nxt = self._next_valid(cron_row, meta, cursor,
                                       apply_jitter=False)
                if nxt is None:
                    break
                out.append(nxt)
                cursor = nxt
            return out
        if ttype == "reminder":
            r = self.db.query_one(
                "SELECT due_at FROM reminders WHERE task_id = ?",
                (task_id,)) or {}
            if row.get("status") in ("pending", "snoozed") and r.get("due_at"):
                return [float(r["due_at"])]
            return []
        if ttype == "one_time":
            run_at = float(meta.get("run_at") or 0.0)
            if row.get("status") == "pending" and run_at:
                return [run_at]
            return []
        return []

    async def describe(self, task_id: str) -> str | None:
        """Chat-ready one-line summary of a task.

        e.g. ``"⏰ Every weekday at 9:00 AM · Africa/Lagos · next: today
        9:00 AM"``.  Returns None when the task doesn't exist.
        """
        task = await self.get_task(task_id)
        if task is None:
            return None
        meta = task.metadata or {}
        tz = meta.get("tz") or None
        status = (f" [{task.status.value}]"
                  if task.status != TaskStatus.PENDING else "")
        if isinstance(task, CronJob):
            try:
                if meta.get("recurrence") == "rrule":
                    sched = _describe_rrule(task.cron_expr)
                else:
                    sched = _describe_cron(task.cron_expr)
            except ValueError:
                sched = task.cron_expr
            nxt = (f" · next: {_fmt_next(task.next_run, tz)}"
                   if task.next_run else "")
            tzs = f" · {tz}" if tz else ""
            return f"⏰ {sched}{tzs}{nxt}{status}"
        if isinstance(task, Reminder):
            return (f"🔔 Reminder: {task.text} · "
                    f"due {_fmt_next(task.due_at, tz)}{status}")
        if isinstance(task, EventHook):
            return f"⚡ Hook: {task.event_type} → {task.action}{status}"
        run_at = float(meta.get("run_at") or 0.0)
        when = f" · at {_fmt_next(run_at, tz)}" if run_at else ""
        return f"📋 One-time: {task.action}{when}{status}"

    async def run_now(self, task_id: str) -> bool:
        """Manually fire a task's action right now (Quartz ``triggerJob``).

        The schedule is untouched — a cron job's ``next_run`` is NOT
        advanced and a reminder is NOT completed.  Goes through the
        guarded path (timeout + registered-handler check), records a
        ``manual`` run, emits ``fired``.  Returns False when unknown.
        """
        row = self.db.query_one(
            "SELECT * FROM scheduled_tasks WHERE task_id = ?", (task_id,))
        if not row:
            return False
        row = dict(row)
        meta = self._row_metadata(row)
        action = row["action"]
        try:
            parameters = json.loads(row.get("parameters") or "{}")
        except (ValueError, TypeError):
            parameters = {}
        timeout_s = float(meta.get("timeout_s") or 0.0)
        started = time.time()
        try:
            await self._run_guarded(
                action, parameters if isinstance(parameters, dict) else {},
                timeout_s)
        except Exception as exc:  # noqa: BLE001
            seconds = time.time() - started
            self._record_run(task_id, started, seconds, False,
                             f"manual run failed: {str(exc)[:300]}",
                             "manual")
            self._emit("failed", {"task_id": task_id, "action": action,
                                  "manual": True, "error": str(exc)[:300]})
            return False
        seconds = time.time() - started
        self._record_run(task_id, started, seconds, True, "manual run",
                         "manual")
        self._emit("fired", {"task_id": task_id, "action": action,
                             "seconds": round(seconds, 2), "manual": True})
        _log.info("manual run: %s", task_id)
        return True

    async def add_skip_dates(self, task_id: str, dates: Any) -> CronJob | None:
        """Add dates a cron/rrule job never fires on; recomputes next_run."""
        job = await self.get_cron(task_id)
        if job is None:
            return None
        skips = sorted(set(job.metadata.get("skip_dates") or [])
                       | set(_normalize_skip_dates(dates)))
        meta = {**job.metadata, "skip_dates": skips}
        self._write_meta(task_id, {"skip_dates": skips})
        nxt = self._next_valid(
            {"cron_expr": job.cron_expr, "task_id": task_id,
             "created_at": job.created_at},
            meta, time.time())
        with self.db.transaction():
            if nxt is None:
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'completed', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            else:
                self.db.execute(
                    "UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                    (nxt, task_id))
        return await self.get_cron(task_id)

    async def remove_skip_dates(self, task_id: str, dates: Any) -> CronJob | None:
        """Remove dates from a job's skip list; recomputes next_run."""
        job = await self.get_cron(task_id)
        if job is None:
            return None
        skips = sorted(set(job.metadata.get("skip_dates") or [])
                       - set(_normalize_skip_dates(dates)))
        meta = {**job.metadata, "skip_dates": skips}
        self._write_meta(task_id, {"skip_dates": skips})
        nxt = self._next_valid(
            {"cron_expr": job.cron_expr, "task_id": task_id,
             "created_at": job.created_at},
            meta, time.time())
        with self.db.transaction():
            if nxt is not None:
                self.db.execute(
                    "UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                    (nxt, task_id))
        return await self.get_cron(task_id)

    # ── Worker ───────────────────────────────────────────────────────────────
    
    def start(self, *, catch_up: bool = True) -> None:
        """Start the scheduler worker thread.

        ``catch_up`` runs :meth:`catch_up_on_startup` first so firings
        missed while the process was down go through each task's
        missed-fire policy instead of piling up silently.
        """
        if self._running:
            return

        self._running = True
        self._started_at = time.time()
        if catch_up:
            try:
                fired = self.catch_up_on_startup()
                if fired:
                    _log.info("scheduler caught up %d missed task(s)", len(fired))
            except Exception:  # noqa: BLE001
                _log.exception("scheduler catch-up failed")
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        _log.info("Scheduler worker started")

    def stop(self) -> None:
        """Stop the scheduler worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        _log.info("Scheduler worker stopped")

    def catch_up_on_startup(self) -> list[str]:
        """Fire tasks whose time passed while the scheduler was down.

        Each task goes through its own missed-fire policy (``fire_now``
        coalesces one run; ``skip`` drops it) and staleness bound (older
        than ``stale_after_s`` is marked ``missed``).  Returns the ids of
        tasks that actually fired.
        """
        now = time.time()
        fired: list[str] = []
        # cron jobs
        try:
            cron_rows = self.db.query("""
                SELECT c.*, t.action, t.parameters, t.metadata, t.task_type,
                       t.status, t.created_at FROM cron_jobs c
                JOIN scheduled_tasks t ON c.task_id = t.task_id
                WHERE c.next_run <= ? AND t.status = 'pending'
            """, (now,))
        except Exception:  # noqa: BLE001
            cron_rows = []
        for row in cron_rows:
            row = dict(row)
            if asyncio.run(self._fire_cron(row, now)):
                fired.append(row["task_id"])
        # reminders
        try:
            reminder_rows = self.db.query("""
                SELECT r.*, t.action, t.parameters, t.metadata, t.task_type,
                       t.status FROM reminders r
                JOIN scheduled_tasks t ON r.task_id = t.task_id
                WHERE r.due_at <= ? AND t.status IN ('pending', 'snoozed')
            """, (now,))
        except Exception:  # noqa: BLE001
            reminder_rows = []
        for row in reminder_rows:
            row = dict(row)
            if asyncio.run(self._fire_reminder(row, now)):
                fired.append(row["task_id"])
        # one-time tasks
        try:
            task_rows = self.db.query("""
                SELECT * FROM scheduled_tasks
                WHERE task_type = 'one_time' AND status = 'pending'
            """)
        except Exception:  # noqa: BLE001
            task_rows = []
        for row in task_rows:
            meta = self._row_metadata(row)
            if float(meta.get("run_at") or 0) <= now:
                row = dict(row)
                if asyncio.run(self._fire_task(row, now)):
                    fired.append(row["task_id"])
        return fired

    def _run_count(self, task_id: str) -> int:
        """task_runs rows for a task (best-effort)."""
        try:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?",
                (task_id,))
            return int((row or {}).get("n") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def health(self) -> dict[str, Any]:
        """Scheduler health snapshot.  Never raises.

        Answers "is my scheduler alive": worker state, task counts by
        status, what's due now, what fires next, what's in flight, recent
        runs, and dead-lettered tasks.  ``ok`` is False when the worker
        died after start or tasks are dead-lettered.
        """
        out: dict[str, Any] = {
            "ok": True,
            "reasons": [],
            "running": self._running,
            "worker_alive": False,
            "started_at": self._started_at,
            "tick_errors": self._tick_errors,
            "tasks": {},
            "due_now": 0,
            "next_task": None,
            "in_flight": [],
            "deferred": {},
            "dead": [],
            "recent_runs": [],
            "upcoming": [],
            "stale": [],
        }
        try:
            out["worker_alive"] = (self._worker_thread is not None
                                   and self._worker_thread.is_alive())
            if self._started_at and not out["worker_alive"]:
                out["reasons"].append("worker thread died after start()")
            try:
                rows = self.db.query(
                    "SELECT status, COUNT(*) AS n FROM scheduled_tasks "
                    "GROUP BY status")
                out["tasks"] = {r["status"]: r["n"] for r in rows}
            except Exception:  # noqa: BLE001
                pass
            now = time.time()
            try:
                row = self.db.query_one(
                    """SELECT t.task_id, t.task_type, c.next_run AS at_ts
                       FROM scheduled_tasks t LEFT JOIN cron_jobs c
                         ON c.task_id = t.task_id
                       WHERE t.status = 'pending' AND t.task_type = 'cron'
                         AND c.next_run IS NOT NULL
                       ORDER BY c.next_run ASC LIMIT 1""")
                if row:
                    out["next_task"] = {
                        "task_id": row["task_id"], "kind": "cron",
                        "in_s": max(0, round(float(row["at_ts"]) - now)),
                    }
                    out["due_now"] = sum(
                        1 for r in self.db.query(
                            "SELECT task_id FROM cron_jobs WHERE next_run <= ?",
                            (now,)))
            except Exception:  # noqa: BLE001
                pass
            with self._flight_lock:
                out["in_flight"] = sorted(self._in_flight)
            out["deferred"] = dict(self._deferrals)
            # upcoming: next 5 firings across cron / reminder / one-time
            try:
                upcoming: list[dict[str, Any]] = []
                for r in self.db.query(
                        "SELECT t.task_id, t.task_type, c.next_run AS at_ts "
                        "FROM scheduled_tasks t JOIN cron_jobs c "
                        "ON c.task_id = t.task_id "
                        "WHERE t.status = 'pending' ORDER BY c.next_run ASC "
                        "LIMIT 5"):
                    upcoming.append({"task_id": r["task_id"], "kind": "cron",
                                     "at": float(r["at_ts"]),
                                     "in_s": max(0, int(float(r["at_ts"])
                                                        - now))})
                for r in self.db.query(
                        "SELECT t.task_id, r.due_at AS at_ts FROM "
                        "scheduled_tasks t JOIN reminders r "
                        "ON r.task_id = t.task_id "
                        "WHERE t.status IN ('pending', 'snoozed') "
                        "ORDER BY r.due_at ASC LIMIT 5"):
                    upcoming.append({"task_id": r["task_id"],
                                     "kind": "reminder",
                                     "at": float(r["at_ts"]),
                                     "in_s": max(0, int(float(r["at_ts"])
                                                        - now))})
                upcoming.sort(key=lambda e: e["at"])
                out["upcoming"] = upcoming[:5]
            except Exception:  # noqa: BLE001
                pass
            # stale: elapsed-silence heuristic.  A job can show a healthy
            # *future* next_run while nothing actually executes (the trigger
            # keeps advancing); the signal that catches it is silence vs
            # the job's own cadence: no firing for 2x the last interval.
            try:
                stale = []
                for r in self.db.query(
                        "SELECT t.task_id, c.next_run, c.last_run FROM "
                        "scheduled_tasks t JOIN cron_jobs c "
                        "ON c.task_id = t.task_id "
                        "WHERE t.status = 'pending' AND c.last_run IS NOT "
                        "NULL LIMIT 50"):
                    cadence = float(r["next_run"]) - float(r["last_run"])
                    silence = now - float(r["last_run"])
                    if cadence > 0 and silence > 2 * cadence:
                        stale.append({
                            "task_id": r["task_id"],
                            "silence_s": int(silence),
                            "cadence_s": int(cadence),
                        })
                out["stale"] = stale[:10]
                if stale:
                    out["reasons"].append(
                        f"{len(stale)} cron job(s) silent beyond 2x cadence")
            except Exception:  # noqa: BLE001
                pass
            try:
                dead = self.db.query(
                    "SELECT task_id, action, updated_at FROM scheduled_tasks "
                    "WHERE status = 'dead' ORDER BY updated_at DESC LIMIT 10")
                out["dead"] = [
                    {"task_id": r["task_id"], "action": r["action"],
                     "since": r["updated_at"]} for r in dead]
                if out["dead"]:
                    out["reasons"].append(
                        f"{len(out['dead'])} task(s) dead-lettered")
            except Exception:  # noqa: BLE001
                pass
            try:
                runs = self.db.query(
                    "SELECT task_id, started_at, ok, result, trigger_source "
                    "FROM task_runs ORDER BY started_at DESC LIMIT 5")
                out["recent_runs"] = [
                    {"task_id": r["task_id"], "ok": bool(r["ok"]),
                     "result": (r["result"] or "")[:100],
                     "trigger_source": r["trigger_source"] or ""}
                    for r in runs]
            except Exception:  # noqa: BLE001
                pass
            out["ok"] = not out["reasons"]
        except Exception as exc:  # noqa: BLE001 - health never raises
            out["ok"] = False
            out["reasons"].append(f"health check errored: {exc}")
        return out
    
    def _worker_loop(self) -> None:
        """Background worker loop."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                _log.error(f"Scheduler tick failed: {e}")
            
            time.sleep(1)  # Check every second
    
    def _tick(self) -> None:
        """Process due tasks.

        When a resource advisor is injected and reports pressure, heavy
        tasks are deferred this tick (they stay pending and are retried on
        the next tick); light tasks still run.  Every firing goes through
        its missed-fire / overlap / timeout / retry / dead-letter policy;
        every execution lands one row in ``task_runs``.
        """
        now = time.time()
        defer_heavy, pressure_reasons = self._pressure_gate()
        deferred = 0

        # Process due cron jobs
        try:
            cron_rows = self.db.query("""
                SELECT c.*, t.action, t.parameters, t.metadata, t.task_type,
                       t.status, t.created_at FROM cron_jobs c
                JOIN scheduled_tasks t ON c.task_id = t.task_id
                WHERE c.next_run <= ? AND t.status = 'pending'
            """, (now,))
        except Exception:  # noqa: BLE001
            cron_rows = []
        for row in cron_rows:
            if self._maybe_defer(row, defer_heavy, pressure_reasons):
                deferred += 1
                continue
            asyncio.run(self._fire_cron(dict(row), now))

        # Process due reminders (pending AND snoozed — a snoozed
        # reminder whose new due time arrived must fire).
        try:
            reminder_rows = self.db.query("""
                SELECT r.*, t.action, t.parameters, t.metadata, t.task_type,
                       t.status FROM reminders r
                JOIN scheduled_tasks t ON r.task_id = t.task_id
                WHERE r.due_at <= ? AND t.status IN ('pending', 'snoozed')
            """, (now,))
        except Exception:  # noqa: BLE001
            reminder_rows = []
        for row in reminder_rows:
            if self._maybe_defer(row, defer_heavy, pressure_reasons):
                deferred += 1
                continue
            asyncio.run(self._fire_reminder(dict(row), now))

        # Process one-time tasks
        try:
            task_rows = self.db.query("""
                SELECT * FROM scheduled_tasks
                WHERE task_type = 'one_time' AND status = 'pending'
            """)
        except Exception:  # noqa: BLE001
            task_rows = []
        for row in task_rows:
            metadata = self._row_metadata(row)
            run_at = metadata.get("run_at", 0)
            if run_at <= now:
                if self._maybe_defer(row, defer_heavy, pressure_reasons):
                    deferred += 1
                    continue
                asyncio.run(self._fire_task(dict(row), now))

        if deferred:
            _log.info(
                "Scheduler tick deferred %d heavy task(s) under resource pressure "
                "[%s]; they remain pending and will be retried next tick",
                deferred,
                "; ".join(pressure_reasons) if pressure_reasons else "no reasons reported",
            )
            self._emit("deferred", {"count": deferred,
                                    "reasons": pressure_reasons})

    # ── firing (missed-fire / overlap / timeout / retry / dead-letter) ──

    def _overlap_decision(self, task_id: str,
                          meta: dict[str, Any]) -> str:
        """'run' | 'skip' | 'queue' based on in-flight state + policy."""
        with self._flight_lock:
            in_flight = task_id in self._in_flight
        if not in_flight:
            return "run"
        policy = str(meta.get("overlap_policy") or "skip").strip().lower()
        if policy not in OVERLAP_POLICIES:
            policy = "skip"
        return "run" if policy == "concurrent" else policy

    def _mark_flight(self, task_id: str, on: bool) -> None:
        with self._flight_lock:
            if on:
                self._in_flight.add(task_id)
            else:
                self._in_flight.discard(task_id)

    @staticmethod
    def _retry_delay(base_s: float, fail_count: int) -> float:
        """Exponential backoff with a touch of jitter: base * 2^(n-1)."""
        delay = max(5.0, float(base_s or 60.0)) * (2.0 ** max(0, fail_count - 1))
        delay = min(delay, 24 * 3600.0)
        return delay * (0.75 + random.random() * 0.5)

    async def _run_guarded(self, action: str, parameters: dict[str, Any],
                           timeout_s: float) -> Any:
        """Run an action handler with an optional wall-clock cap.

        Raises RuntimeError when no handler is registered (a broken task,
        not a quiet no-op — it goes through the retry/dead-letter path),
        and on timeout.
        """
        handler = self._action_handlers.get(action)
        if handler is None:
            raise RuntimeError(f"no handler registered for action {action!r}")
        if asyncio.iscoroutinefunction(handler):
            coro = handler(**parameters)
            if timeout_s and timeout_s > 0:
                try:
                    return await asyncio.wait_for(coro, timeout_s)
                except (asyncio.TimeoutError, TimeoutError):
                    raise RuntimeError(
                        f"action {action!r} timed out after {timeout_s:g}s")
            return await coro
        # sync handler: run directly (no timeout available)
        return handler(**parameters)

    async def _fire_cron(self, row: dict[str, Any], now: float) -> bool:
        """Fire one due cron/rrule job through its full policy chain.

        Returns True when the action actually executed."""
        task_id = row["task_id"]
        meta = self._row_metadata(row)
        action = row["action"]
        parameters = json.loads(row["parameters"])
        timeout_s = float(meta.get("timeout_s") or 0.0)

        decision = self._overlap_decision(task_id, meta)
        if decision == "skip":
            self._advance_cron(row, meta, now)
            self._record_run(task_id, now, 0.0, True,
                             "skipped: previous run still in flight "
                             "(overlap_policy=skip)", "tick")
            _log.info("cron %s skipped (overlap)", task_id)
            return False
        if decision == "queue":
            return False  # leave it for the next tick

        # missed-fire policy
        lateness = now - float(row.get("next_run") or now)
        stale_after = float(meta.get("stale_after_s") or 24 * 3600)
        if lateness > stale_after:
            self._advance_cron(row, meta, now)
            self._record_run(task_id, now, 0.0, True,
                             f"missed: firing {int(lateness)}s stale "
                             f"(>{int(stale_after)}s)", "tick")
            self._emit("missed", {"task_id": task_id, "action": action,
                                  "lateness_s": int(lateness)})
            _log.warning("cron %s missed (stale %ds) — advanced",
                         task_id, int(lateness))
            return False
        if lateness > 0 and meta.get("missed_fire_policy") == "skip":
            self._advance_cron(row, meta, now)
            self._record_run(task_id, now, 0.0, True,
                             f"skipped: missed by {int(lateness)}s "
                             "(missed_fire_policy=skip)", "tick")
            self._emit("missed", {"task_id": task_id, "action": action,
                                  "lateness_s": int(lateness)})
            return False
        # fire_now (default): coalesce — one run now, however late.

        started = time.time()
        self._mark_flight(task_id, True)
        try:
            await self._run_guarded(action, parameters, timeout_s)
        except Exception as exc:  # noqa: BLE001
            seconds = time.time() - started
            self._mark_flight(task_id, False)
            self._cron_failed(row, meta, now, seconds, str(exc))
            return True
        seconds = time.time() - started
        self._mark_flight(task_id, False)
        # success resets the failure streak
        if int(meta.get("fail_count") or 0):
            self._write_meta(task_id, {"fail_count": 0})
        run_count = int(row.get("run_count") or 0) + 1
        next_run = self._advance_cron(row, meta, now)
        self._record_run(task_id, started, seconds, True,
                         f"run {run_count}", "tick")
        self._emit("fired", {"task_id": task_id, "action": action,
                             "seconds": round(seconds, 2),
                             "run_count": run_count})
        _log.info(f"Executed cron job: {task_id} (run {run_count})")
        if row.get("max_runs") and run_count >= int(row["max_runs"]):
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'completed', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
        return True

    def _advance_cron(self, row: dict[str, Any], meta: dict[str, Any],
                      now: float) -> float | None:
        """Move a cron/rrule job to its next firing.  Returns next_run.

        Goes through :meth:`_next_valid`, so skip-dates are honored and
        ``jitter_s`` is applied to every firing.
        """
        task_id = row["task_id"]
        nxt = self._next_valid(row, meta, now)
        with self.db.transaction():
            if nxt is None:
                # exhausted RRULE (COUNT/UNTIL) — retire like a one-shot
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'completed', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
                self.db.execute(
                    "UPDATE cron_jobs SET last_run = ? WHERE task_id = ?",
                    (now, task_id))
            else:
                self.db.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, "
                    "run_count = run_count + 1 WHERE task_id = ?",
                    (now, nxt, task_id))
        return nxt

    def _cron_failed(self, row: dict[str, Any], meta: dict[str, Any],
                     now: float, seconds: float, error: str) -> None:
        """Failure path: backoff + retry, or dead-letter at max_attempts.

        The schedule ALWAYS advances — a failing job never hot-loops the
        tick.  Retries wait out an exponential backoff; after
        ``max_attempts`` the task is dead-lettered (status 'dead') for a
        human to inspect instead of spamming the log every second.
        """
        task_id = row["task_id"]
        fail_count = int(meta.get("fail_count") or 0) + 1
        max_attempts = max(1, int(meta.get("max_attempts") or 1))
        self._record_run(task_id, now - seconds, seconds, False,
                         f"failed (attempt {fail_count}): {error[:300]}",
                         "tick")
        if meta.get("pause_on_failure") and fail_count < max_attempts:
            # Temporal --pause-on-failure gold: stop the schedule on the
            # first failure so a human can inspect; resume() restarts it.
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'paused', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            self._write_meta(task_id, {"fail_count": fail_count})
            self._emit("failed", {"task_id": task_id, "action": row["action"],
                                  "fail_count": fail_count,
                                  "paused": True,
                                  "error": error[:300]})
            _log.warning("cron %s paused after failure (pause_on_failure): %s",
                         task_id, error[:200])
            return
        if fail_count >= max_attempts:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'dead', updated_at = ? "
                    "WHERE task_id = ?", (time.time(), task_id))
            self._write_meta(task_id, {"fail_count": fail_count})
            self._emit("dead", {"task_id": task_id, "action": row["action"],
                                "fail_count": fail_count, "error": error[:300]})
            _log.error("cron %s dead-lettered after %d failures: %s",
                       task_id, fail_count, error[:200])
            return
        delay = self._retry_delay(float(meta.get("retry_base_s") or 60.0),
                                  fail_count)
        retry_at = now + delay
        with self.db.transaction():
            self.db.execute(
                "UPDATE cron_jobs SET next_run = ? WHERE task_id = ?",
                (retry_at, task_id))
        self._write_meta(task_id, {"fail_count": fail_count})
        self._emit("failed", {"task_id": task_id, "action": row["action"],
                              "fail_count": fail_count,
                              "max_attempts": max_attempts,
                              "retry_in_s": int(delay), "error": error[:300]})
        _log.warning("cron %s failed (attempt %d/%d) — retry in %ds: %s",
                     task_id, fail_count, max_attempts, int(delay),
                     error[:200])

    async def _fire_reminder(self, row: dict[str, Any], now: float) -> bool:
        """Fire one due reminder through its policy chain.

        Returns True when the action actually executed."""
        task_id = row["task_id"]
        meta = self._row_metadata(row)
        action = row["action"]
        parameters = json.loads(row["parameters"])
        timeout_s = float(meta.get("timeout_s") or 0.0)

        decision = self._overlap_decision(task_id, meta)
        if decision == "skip":
            self._record_run(task_id, now, 0.0, True,
                             "skipped: previous run still in flight "
                             "(overlap_policy=skip)", "tick")
            return False
        if decision == "queue":
            return False

        # quiet hours: shift the firing to the window end (DND), don't drop it
        quiet = meta.get("quiet_hours") or []
        shifted = _quiet_shift(
            now, tuple(quiet) if len(quiet) == 2 else None,
            meta.get("tz") or None)
        if shifted is not None:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE reminders SET due_at = ? WHERE task_id = ?",
                    (shifted, task_id))
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'pending', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            self._record_run(
                task_id, now, 0.0, True,
                f"suppressed: quiet hours → shifted to "
                f"{_fmt_next(shifted, meta.get('tz') or None)}", "tick")
            self._emit("deferred", {"task_id": task_id, "action": action,
                                    "reason": "quiet_hours",
                                    "shifted_to": shifted})
            _log.info("reminder %s in quiet hours — shifted to %s",
                      task_id, _fmt_next(shifted, meta.get("tz") or None))
            return False

        due_at = float(row.get("due_at") or now)
        lateness = now - due_at
        stale_after = float(meta.get("stale_after_s") or 6 * 3600)
        if lateness > stale_after or (
                lateness > 0 and meta.get("missed_fire_policy") == "skip"):
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'missed', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            self._record_run(task_id, now, 0.0, True,
                             f"missed: reminder {int(lateness)}s overdue",
                             "tick")
            self._emit("missed", {"task_id": task_id, "action": action,
                                  "lateness_s": int(lateness)})
            _log.warning("reminder %s missed (%ds overdue)", task_id,
                         int(lateness))
            return False

        started = time.time()
        self._mark_flight(task_id, True)
        try:
            await self._run_guarded(action, parameters, timeout_s)
        except Exception as exc:  # noqa: BLE001
            seconds = time.time() - started
            self._mark_flight(task_id, False)
            self._reminder_failed(row, meta, now, seconds, str(exc), due_at)
            return True
        seconds = time.time() - started
        self._mark_flight(task_id, False)
        if int(meta.get("fail_count") or 0):
            self._write_meta(task_id, {"fail_count": 0})
        # nag mode (Due-app "auto snooze"): re-fire until acknowledged
        nag_every = float(meta.get("nag_every_s") or 0.0)
        max_nags = int(meta.get("max_nags") or 0)
        nag_count = int(meta.get("nag_count") or 0)
        if nag_every > 0 and nag_count < max_nags:
            nag_count += 1
            next_due = time.time() + nag_every
            with self.db.transaction():
                self.db.execute(
                    "UPDATE reminders SET due_at = ? WHERE task_id = ?",
                    (next_due, task_id))
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'pending', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            self._write_meta(task_id, {"nag_count": nag_count})
            self._record_run(task_id, started, seconds, True,
                             f"fired (nag {nag_count}/{max_nags})", "tick")
            self._emit("fired", {"task_id": task_id, "action": action,
                                 "seconds": round(seconds, 2),
                                 "lateness_s": int(lateness),
                                 "nag": nag_count, "max_nags": max_nags})
            _log.info("reminder %s fired (nag %d/%d)", task_id,
                      nag_count, max_nags)
            return True
        await self.complete_reminder(task_id)
        note = (f"fired {int(lateness)}s late"
                if lateness > 1 else "fired on time")
        self._record_run(task_id, started, seconds, True, note, "tick")
        self._emit("fired", {"task_id": task_id, "action": action,
                             "seconds": round(seconds, 2),
                             "lateness_s": int(lateness)})
        _log.info(f"Executed reminder: {task_id}")
        return True

    def _reminder_failed(self, row: dict[str, Any], meta: dict[str, Any],
                         now: float, seconds: float, error: str,
                         due_at: float) -> None:
        task_id = row["task_id"]
        fail_count = int(meta.get("fail_count") or 0) + 1
        max_attempts = max(1, int(meta.get("max_attempts") or 1))
        self._record_run(task_id, now - seconds, seconds, False,
                         f"failed (attempt {fail_count}): {error[:300]}",
                         "tick")
        if fail_count >= max_attempts:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'dead', updated_at = ? "
                    "WHERE task_id = ?", (time.time(), task_id))
            self._write_meta(task_id, {"fail_count": fail_count})
            self._emit("dead", {"task_id": task_id, "action": row["action"],
                                "fail_count": fail_count, "error": error[:300]})
            _log.error("reminder %s dead-lettered after %d failures: %s",
                       task_id, fail_count, error[:200])
            return
        delay = self._retry_delay(float(meta.get("retry_base_s") or 60.0),
                                  fail_count)
        with self.db.transaction():
            self.db.execute(
                "UPDATE reminders SET due_at = ? WHERE task_id = ?",
                (now + delay, task_id))
            self.db.execute(
                "UPDATE scheduled_tasks SET status = 'pending', updated_at = ? "
                "WHERE task_id = ?", (time.time(), task_id))
        self._write_meta(task_id, {"fail_count": fail_count})
        self._emit("failed", {"task_id": task_id, "action": row["action"],
                              "fail_count": fail_count,
                              "max_attempts": max_attempts,
                              "retry_in_s": int(delay), "error": error[:300]})
        _log.warning("reminder %s failed (attempt %d/%d) — retry in %ds: %s",
                     task_id, fail_count, max_attempts, int(delay),
                     error[:200])

    async def _fire_task(self, row: dict[str, Any], now: float) -> bool:
        """Fire one due one-time task through its policy chain.

        Returns True when the action actually executed."""
        task_id = row["task_id"]
        meta = self._row_metadata(row)
        action = row["action"]
        parameters = json.loads(row["parameters"])
        timeout_s = float(meta.get("timeout_s") or 0.0)
        run_at = float(meta.get("run_at") or now)

        decision = self._overlap_decision(task_id, meta)
        if decision != "run":
            # skip or queue: leave it; the next tick retries
            if decision == "skip":
                self._record_run(task_id, now, 0.0, True,
                                 "skipped: previous run still in flight "
                                 "(overlap_policy=skip)", "tick")
            return False

        # quiet hours: shift the firing to the window end (DND), don't drop it
        quiet = meta.get("quiet_hours") or []
        shifted = _quiet_shift(
            now, tuple(quiet) if len(quiet) == 2 else None,
            meta.get("tz") or None)
        if shifted is not None:
            self._write_meta(task_id, {"run_at": shifted})
            self._record_run(
                task_id, now, 0.0, True,
                f"suppressed: quiet hours → shifted to "
                f"{_fmt_next(shifted, meta.get('tz') or None)}", "tick")
            self._emit("deferred", {"task_id": task_id, "action": action,
                                    "reason": "quiet_hours",
                                    "shifted_to": shifted})
            _log.info("one-time task %s in quiet hours — shifted to %s",
                      task_id, _fmt_next(shifted, meta.get("tz") or None))
            return False

        lateness = now - run_at
        stale_after = float(meta.get("stale_after_s") or 3600)
        if lateness > stale_after or (
                lateness > 0 and meta.get("missed_fire_policy") == "skip"):
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET status = 'missed', "
                    "updated_at = ? WHERE task_id = ?",
                    (time.time(), task_id))
            self._record_run(task_id, now, 0.0, True,
                             f"missed: one-time task {int(lateness)}s overdue",
                             "tick")
            self._emit("missed", {"task_id": task_id, "action": action,
                                  "lateness_s": int(lateness)})
            return False

        started = time.time()
        self._mark_flight(task_id, True)
        try:
            await self._run_guarded(action, parameters, timeout_s)
        except Exception as exc:  # noqa: BLE001
            seconds = time.time() - started
            self._mark_flight(task_id, False)
            fail_count = int(meta.get("fail_count") or 0) + 1
            max_attempts = max(1, int(meta.get("max_attempts") or 1))
            self._record_run(task_id, started, seconds, False,
                             f"failed (attempt {fail_count}): {str(exc)[:300]}",
                             "tick")
            if fail_count >= max_attempts:
                with self.db.transaction():
                    self.db.execute(
                        "UPDATE scheduled_tasks SET status = 'dead', "
                        "updated_at = ? WHERE task_id = ?",
                        (time.time(), task_id))
                self._write_meta(task_id, {"fail_count": fail_count})
                self._emit("dead", {"task_id": task_id, "action": action,
                                    "fail_count": fail_count,
                                    "error": str(exc)[:300]})
                _log.error("one-time task %s dead-lettered after %d failures",
                           task_id, fail_count)
                return
            delay = self._retry_delay(float(meta.get("retry_base_s") or 60.0),
                                      fail_count)
            self._write_meta(task_id, {"fail_count": fail_count,
                                       "run_at": now + delay})
            self._emit("failed", {"task_id": task_id, "action": action,
                                  "fail_count": fail_count,
                                  "max_attempts": max_attempts,
                                  "retry_in_s": int(delay),
                                  "error": str(exc)[:300]})
            _log.warning("one-time task %s failed (attempt %d/%d) — "
                         "retry in %ds", task_id, fail_count, max_attempts,
                         int(delay))
            return True
        seconds = time.time() - started
        self._mark_flight(task_id, False)
        with self.db.transaction():
            self.db.execute(
                "UPDATE scheduled_tasks SET status = 'completed', "
                "updated_at = ? WHERE task_id = ?", (time.time(), task_id))
        self._record_run(task_id, started, seconds, True, "completed", "tick")
        self._emit("fired", {"task_id": task_id, "action": action,
                             "seconds": round(seconds, 2)})
        _log.info(f"Executed one-time task: {task_id}")
        return True

    async def _execute_action(self, action: str,
                              parameters: dict[str, Any]) -> None:
        """Execute an action (legacy entry point — no timeout).

        Prefer :meth:`_run_guarded`.  Kept because event-hook triggering
        and external callers use it.
        """
        await self._run_guarded(action, parameters, 0.0)

    # ── Resource-aware gating ──────────────────────────────────────────────

    def _pressure_gate(self) -> tuple[bool, list[str]]:
        """Consult the injected resource advisor once per tick.

        Returns ``(defer_heavy, reasons)``.  Heavy tasks are deferred when
        the advisory reports ``throttled`` or ``ok == False``.  Never
        raises and never blocks: with no advisor injected there is no
        gating, and a failing advisor fails closed for heavy work while
        light tasks still run.
        """
        advisor = self._resources
        if advisor is None:
            return False, []
        try:
            advisory = advisor.consult()
        except Exception as exc:  # noqa: BLE001 - advisor must never break the tick
            _log.warning(
                "Resource consult failed (%s); deferring heavy tasks this tick", exc
            )
            return True, [f"consult failed: {exc}"]
        if not isinstance(advisory, dict):
            _log.warning(
                "Resource consult returned %s; deferring heavy tasks this tick",
                type(advisory).__name__,
            )
            return True, ["consult returned non-dict advisory"]
        throttled = bool(advisory.get("throttled", False))
        ok = bool(advisory.get("ok", True))
        reasons = [str(r) for r in (advisory.get("reasons") or [])]
        if throttled or not ok:
            return True, reasons
        return False, reasons

    def _maybe_defer(
        self,
        row: dict[str, Any],
        defer_heavy: bool,
        pressure_reasons: list[str],
    ) -> bool:
        """Defer a heavy task while pressure is high; log the deferral.

        Returns True when the task was deferred (caller must skip it).
        Deferred tasks are NOT dropped — they stay pending and are picked
        up on a later tick when pressure eases.
        """
        if not defer_heavy:
            return False
        metadata = self._row_metadata(row)
        if not self._is_heavy(metadata):
            return False
        task_id = str(row.get("task_id", "?"))
        task_type = str(row.get("task_type", "task"))
        action = str(row.get("action", "?"))
        self._deferrals[task_id] = self._deferrals.get(task_id, 0) + 1
        _log.warning(
            "Deferring heavy %s task %s (action=%s, deferral #%d): "
            "resource pressure high [%s] — task stays pending, retried next tick",
            task_type,
            task_id,
            action,
            self._deferrals[task_id],
            "; ".join(pressure_reasons) if pressure_reasons else "no reasons reported",
        )
        return True

    @staticmethod
    def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
        """Parse a task row's metadata column into a dict (never raises)."""
        raw = row.get("metadata")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _write_meta(self, task_id: str, patch: dict[str, Any]) -> None:
        """Merge ``patch`` into a task's metadata JSON (never raises)."""
        try:
            row = self.db.query_one(
                "SELECT metadata FROM scheduled_tasks WHERE task_id = ?",
                (task_id,))
            meta = self._row_metadata(row or {})
            meta.update(patch)
            with self.db.transaction():
                self.db.execute(
                    "UPDATE scheduled_tasks SET metadata = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (json.dumps(meta), time.time(), task_id))
        except Exception:  # noqa: BLE001 - metadata is advisory, never fatal
            _log.debug("scheduler metadata write failed for %s", task_id,
                       exc_info=True)

    # ── listeners (observability hooks) ──────────────────────────────────

    def add_listener(self, event: str, callback: Callable) -> None:
        """Subscribe to a scheduler event.

        Events: ``fired`` | ``failed`` | ``missed`` | ``dead`` |
        ``deferred`` | ``snoozed``.  The callback receives one dict
        payload; it may be a plain function or a coroutine function.
        A raising listener never breaks the scheduler.
        """
        with self._listeners_lock:
            self._listeners.setdefault(event, []).append(callback)

    def remove_listener(self, event: str, callback: Callable) -> bool:
        """Unsubscribe.  Returns True when the callback was registered."""
        with self._listeners_lock:
            cbs = self._listeners.get(event, [])
            if callback in cbs:
                cbs.remove(callback)
                return True
            return False

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        """Fire listener callbacks for ``event``.  Never raises.

        Runs in the caller's context: coroutine callbacks are scheduled
        with ``asyncio.run`` only when no loop is running, otherwise they
        are left to the caller's loop via ``create_task`` guarded by a
        try/except — a listener must never break a firing.
        """
        with self._listeners_lock:
            callbacks = list(self._listeners.get(event, []))
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None:
                        loop.create_task(cb(dict(payload)))
                    else:
                        asyncio.run(cb(dict(payload)))
                else:
                    cb(dict(payload))
            except Exception:  # noqa: BLE001 - listeners are telemetry
                _log.debug("scheduler listener %s failed", event, exc_info=True)

    # ── execution history ────────────────────────────────────────────────

    def _record_run(self, task_id: str, started: float, seconds: float,
                    ok: bool, result: str, trigger_source: str) -> None:
        """Land one row in ``task_runs``.  Never raises."""
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO task_runs
                       (id, task_id, started_at, finished_at, seconds, ok,
                        result, trigger_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (new_id("run"), task_id, started, started + seconds,
                     seconds, 1 if ok else 0, (result or "")[:2000],
                     trigger_source),
                )
            self._prune_runs(task_id)
        except Exception:  # noqa: BLE001
            _log.debug("task_runs record failed for %s", task_id, exc_info=True)

    def _prune_runs(self, task_id: str) -> None:
        """Trim a task's history to ``run_history_limit`` rows."""
        try:
            with self.db.transaction():
                self.db.execute(
                    "DELETE FROM task_runs WHERE task_id = ? AND id NOT IN "
                    "(SELECT id FROM task_runs WHERE task_id = ? "
                    "ORDER BY started_at DESC LIMIT ?)",
                    (task_id, task_id, self.run_history_limit))
        except Exception:  # noqa: BLE001
            _log.debug("task_runs prune failed for %s", task_id, exc_info=True)

    def recent_runs(self, task_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Per-execution history for one task, newest first."""
        try:
            rows = self.db.query(
                "SELECT * FROM task_runs WHERE task_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (task_id, max(1, int(limit))))
        except Exception:  # noqa: BLE001
            return []
        return [
            {
                "task_id": r["task_id"],
                "started_at": r["started_at"],
                "seconds": round(float(r.get("seconds") or 0.0), 2),
                "ok": bool(r.get("ok")),
                "result": (r.get("result") or "")[:300],
                "trigger_source": r.get("trigger_source") or "",
            }
            for r in rows
        ]

    @staticmethod
    def _is_heavy(metadata: dict[str, Any]) -> bool:
        """True when task metadata marks the task as heavy work.

        Explicit ``heavy: true`` wins; otherwise a numeric ``weight`` at
        or above :data:`HEAVY_WEIGHT_THRESHOLD` counts as heavy.  The
        default (no keys) is light, so reminders and notifications keep
        running under pressure.
        """
        if metadata.get("heavy"):
            return True
        try:
            weight = float(metadata.get("weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return weight >= HEAVY_WEIGHT_THRESHOLD

    @property
    def deferral_counts(self) -> dict[str, int]:
        """Per-task deferral counts for this session (heavy skips under pressure)."""
        return dict(self._deferrals)

    def eta_for(self, task_type: str,
                segments: dict[str, float] | None = None) -> str:
        """Honest banded ETA for a task type (#91).

        Every time promise ships a band, not a point:
        "ETA 4:30pm (range 4:15–4:50)". Bands widen after misses.
        Never raises.
        """
        try:
            from ..planning.estimates import eta_text
            return eta_text(task_type, segments)
        except Exception:  # noqa: BLE001
            return "⏱️ estimate unavailable"

    def record_outcome(self, task_type: str, predicted_minutes: float,
                       actual_minutes: float) -> bool:
        """Feed a real task outcome into the estimate model (#91)."""
        try:
            from ..planning.estimates import record_actual
            return record_actual(task_type, predicted_minutes, actual_minutes)
        except Exception:  # noqa: BLE001
            return False
    
    def register_action(self, name: str, handler: Callable) -> None:
        """Register an action handler.

        Args:
            name: Action name
            handler: Async function to handle the action
        """
        self._action_handlers[name] = handler
        _log.info(f"Registered action handler: {name}")
