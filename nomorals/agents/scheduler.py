"""Scheduler: durable cron-style jobs, run in-process.

Schedule kinds, parsed from one flexible spec string:

* ``at 2026-09-12 09:00`` (or ISO ``2026-09-12T09:00``) — one-shot
* ``every 30m`` / ``2h`` / ``45s`` — repeating interval
* ``daily 22:00`` (or just ``22:00``) — repeating wall-clock time
* ``daily 22:00 America/New_York`` — daily in a specific IANA timezone
* ``cron 0 22 * * *`` (or bare ``0 22 * * *``) — standard cron expression

Three payload kinds:

* ``message`` — send text to the owner on every live channel (via Notifier)
* ``tool``    — call any registered tool with JSON args
* ``command`` — run a shell command through the sandboxed shell tool

Advanced features:

* **Dependencies** — a job can declare ``depends_on`` (another job's id);
  it only fires when the dependency's last run succeeded.
* **Retries** — ``max_retries`` + ``retry_delay``: failed jobs retry with
  linear backoff before the failure is reported.
* **Missed-job catch-up** — ``catch_up_on_startup()`` runs jobs whose
  ``next_run`` passed while the bot was down (within a max age).

Jobs live in the ``schedule_jobs`` table (migration 15, upgraded by 81):
a restart resumes them, a disabled job stays put, and one-shot jobs
disable themselves after firing. The tick loop runs on a daemon thread
and every due job's outcome is published through the Notifier, so alerts
are durable and multi-channel.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.errors import AmbiguousRef
from ..core.ids import min_unique_prefix_len, new_id, resolve_id_prefix
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet
from .notifier import Notifier

_log = get_logger(__name__)

__all__ = ["Scheduler", "parse_schedule_spec"]

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
    ``cron`` (detail = cron expression string)
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
        raise ValueError(f"unknown timezone: {tz!r}")
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
    Sunday.  Returns the canonical ``"m h dom mon dow"`` string.
    """
    m = _CRON_RE.match(expr.strip())
    if not m:
        raise ValueError(f"not a 5-field cron expression: {expr!r}")
    minute, hour, dom, month, dow = m.groups()
    # validate each field eagerly (raises on garbage)
    _parse_cron_field(minute, 0, 59)
    _parse_cron_field(hour, 0, 23)
    _parse_cron_field(dom, 1, 31)
    _parse_cron_field(month, 1, 12)
    _parse_cron_field(dow, 0, 7)
    return f"{minute} {hour} {dom} {month} {dow}"


def _cron_matches(expr: str, dt: datetime) -> bool:
    """True if ``dt`` (minute precision) matches the cron expression."""
    minute, hour, dom, month, dow = _parse_cron(expr).split()
    mins = _parse_cron_field(minute, 0, 59)
    hrs = _parse_cron_field(hour, 0, 23)
    doms = _parse_cron_field(dom, 1, 31)
    mons = _parse_cron_field(month, 1, 12)
    dows = _parse_cron_field(dow, 0, 7)
    # cron: dow 0 and 7 both Sunday
    py_dow = (dt.weekday() + 1) % 7  # Monday=0 → Sunday=6 → map to 0
    dow_match = py_dow in dows or (py_dow == 0 and 7 in dows)
    # classic cron semantics: dom AND dow both restricted → OR them;
    # otherwise each restricted field must match.
    dom_star = dom.strip() == "*"
    dow_star = dow.strip() == "*"
    if not dom_star and not dow_star:
        day_ok = (dt.day in doms) or dow_match
    else:
        day_ok = (dt.day in doms) and dow_match
    return (dt.minute in mins and dt.hour in hrs
            and dt.month in mons and day_ok)


def _next_cron(expr: str, now: datetime, timezone: str = "") -> float:
    """Next minute at/after ``now`` matching the cron expression.

    Scans forward minute-by-minute (cap: 366 days) — no external deps.
    """
    tz: Any = None
    if (timezone or "").strip():
        try:
            tz = ZoneInfo(timezone.strip())
        except ZoneInfoNotFoundError:
            _log.warning("scheduler: unknown timezone %r, using local",
                         timezone)
    probe = now.astimezone(tz) if tz else now
    # start at the next minute boundary strictly after now
    probe = probe.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = probe + timedelta(days=366)
    while probe <= limit:
        if _cron_matches(expr, probe):
            return probe.timestamp()
        probe += timedelta(minutes=1)
    raise ValueError(f"cron expression never matches within a year: {expr!r}")


def _like_escape(text: str) -> str:
    """Escape SQL LIKE wildcards so a job ref is matched literally."""
    return (text.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_"))


class Scheduler:
    """Owns the schedule_jobs table and the tick loop."""

    def __init__(self, context: Any, *, gateway: Any = None,
                 tick_seconds: float = 20.0, max_concurrent: int = 2,
                 wall_seconds: float = 300.0,
                 redeliver_interval: float = 120.0) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.notifier = Notifier(context, gateway=gateway)
        self.tick_seconds = max(5.0, float(tick_seconds))
        self.max_concurrent = max(1, int(max_concurrent))
        self.wall_seconds = float(wall_seconds)
        #: how often the tick loop sweeps the notification redelivery
        #: queue (throttled: one sweep per interval, not per tick).
        self.redeliver_interval = max(15.0, float(redeliver_interval))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running_jobs = 0
        self._lock = threading.Lock()
        #: health bookkeeping — the numbers behind ``/schedule health``
        self._started_at: float | None = None
        self._last_tick_at: float | None = None
        self._last_redeliver_at = 0.0
        self._tick_errors = 0

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def add(
        self,
        name: str,
        spec: str,
        payload_kind: str,
        payload: dict[str, Any],
        *,
        timezone: str = "",
        depends_on: str = "",
        max_retries: int = 0,
        retry_delay: float = 60.0,
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
        # dependency must name an existing job
        depends_on = (depends_on or "").strip()
        if depends_on:
            dep = self._find(depends_on)
            if dep is None:
                raise ValueError(f"depends_on: no job {depends_on!r}")
            depends_on = dep["id"]  # canonicalize to the full id
        max_retries = max(0, int(max_retries))
        retry_delay = max(10.0, float(retry_delay))
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
                "timezone, depends_on, max_retries, retry_delay, retry_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, NULL, '', ?, ?, ?, ?, ?, ?, 0)",
                (
                    job_id, (name or "").strip() or "job", kind,
                    spec_str,
                    payload_kind, json.dumps(payload),
                    next_run, now, now,
                    timezone, depends_on, max_retries, retry_delay,
                ),
            )
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
                          timezone: str = "") -> float:
        if kind == "at":
            return float(detail)
        if kind == "every":
            return now + float(detail)
        if kind == "cron":
            return _next_cron(str(detail), datetime.fromtimestamp(now), timezone)
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
        return _next_daily(str(row["spec"]), datetime.fromtimestamp(now), tz), True

    def _dependency_ok(self, row: dict[str, Any]) -> bool:
        """True when the job's ``depends_on`` target last succeeded (or none)."""
        dep_id = str(row.get("depends_on") or "").strip()
        if not dep_id:
            return True
        if self.db is None:
            return False
        dep = self.db.query_one(
            "SELECT last_result FROM schedule_jobs WHERE id = ?", (dep_id,))
        if not dep:
            # dependency was deleted — treat as unmet, don't silently run
            return False
        last = str(dep.get("last_result") or "")
        # never ran counts as unmet; only an explicit success opens the gate
        return bool(last) and not last.startswith("job failed")

    def catch_up_on_startup(self, *, max_age_hours: float = 24.0) -> list[dict[str, Any]]:
        """Run jobs whose ``next_run`` passed while the bot was down.

        Only catches up jobs missed within ``max_age_hours`` — anything
        older is assumed stale and just gets rescheduled forward.
        Returns the outcomes of the catch-up runs.
        """
        if self.db is None:
            return []
        now = time.time()
        cutoff = now - max(0.0, float(max_age_hours)) * 3600
        try:
            rows = self.db.query(
                "SELECT * FROM schedule_jobs WHERE enabled = 1 "
                "AND next_run IS NOT NULL AND next_run <= ? "
                "AND next_run >= ? ORDER BY next_run ASC",
                (now, cutoff),
            )
        except Exception:  # noqa: BLE001
            return []
        results = []
        for row in rows:
            row = dict(row)
            if not self._dependency_ok(row):
                continue
            try:
                results.append(self._execute(row))
            except Exception:  # noqa: BLE001
                _log.exception("scheduler catch-up crashed: %s", row.get("id"))
        # reschedule anything missed *before* the cutoff (too stale to run)
        try:
            with self.db.transaction():
                stale = self.db.query(
                    "SELECT * FROM schedule_jobs WHERE enabled = 1 "
                    "AND next_run IS NOT NULL AND next_run < ?",
                    (cutoff,),
                )
                for row in stale:
                    nxt, _ = self._next_after_run(dict(row), now)
                    self.db.execute(
                        "UPDATE schedule_jobs SET next_run = ?, updated_at = ? "
                        "WHERE id = ?",
                        (nxt, now, row["id"]),
                    )
        except Exception:  # noqa: BLE001
            _log.debug("scheduler stale reschedule failed")
        return results

    # ── execution ────────────────────────────────────────────────────────────
    def tick(self) -> list[dict[str, Any]]:
        """Run everything due right now. Returns the outcomes (for tests too).

        Also drives the notification redelivery queue (throttled to
        ``redeliver_interval``) so a scheduled job whose delivery failed
        keeps retrying with backoff on its own — the scheduler no longer
        depends on the watch loop for that.
        """
        if self.db is None:
            return []
        self._last_tick_at = time.time()
        now = self._last_tick_at
        try:
            due = self.db.query(
                "SELECT * FROM schedule_jobs WHERE enabled = 1 AND next_run IS NOT NULL "
                "AND next_run <= ? ORDER BY next_run ASC",
                (now,),
            )
        except Exception:  # noqa: BLE001
            return []
        results = []
        for row in due:
            row = dict(row)
            if not self._dependency_ok(row):
                # dependency hasn't succeeded yet — leave for a later tick
                continue
            with self._lock:
                if self._running_jobs >= self.max_concurrent:
                    # leave it for the next tick rather than dropping it
                    continue
                self._running_jobs += 1
            try:
                results.append(self._execute(row))
            except Exception:  # noqa: BLE001 - one job must not kill the tick
                self._tick_errors += 1
                _log.exception("scheduler job crashed: %s", row.get("id"))
            finally:
                with self._lock:
                    self._running_jobs -= 1
        self._maybe_redeliver()
        return results

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
        return self._execute(dict(row))

    def _execute(self, row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        now = started
        try:
            payload = json.loads(row.get("payload") or "{}")
            if row["payload_kind"] == "message":
                summary = self._run_message(payload)
            elif row["payload_kind"] == "tool":
                summary = self._run_tool(payload)
            else:
                summary = self._run_command(payload)
            ok = True
        except Exception as exc:  # noqa: BLE001
            summary = f"job failed: {exc}"
            ok = False
        # ── retry policy ──────────────────────────────────────────────
        # On failure, if retries remain, schedule the retry with linear
        # backoff instead of reporting the failure.  retry_count tracks
        # consecutive failures; it resets on success.
        max_retries = int(row.get("max_retries") or 0)
        retry_delay = max(10.0, float(row.get("retry_delay") or 60))
        retry_count = int(row.get("retry_count") or 0)
        will_retry = False
        if not ok and retry_count < max_retries:
            will_retry = True
            retry_count += 1
            next_run = now + retry_delay * retry_count  # linear backoff
            still_enabled = True
            # mark the in-flight retry in the result so the owner sees it
            summary = (f"{summary} (retry {retry_count}/{max_retries} "
                       f"in {int(retry_delay * retry_count)}s)")
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
        # alert the owner — durable + multi-channel via the notifier.
        # Failures alert on the failure transition only (a stuck job must
        # not page every run). Successes notify too, except for routine
        # tick/heartbeat jobs which would spam.  Message payloads are the
        # exception the other way round: the message itself was already
        # delivered to the owner's DM by _run_message — a second "job ran"
        # alert would double-send.
        job_name = row['name'].lower()
        is_routine_tick = 'tick' in job_name or 'heartbeat' in job_name or 'sweep' in job_name
        alert = True
        if will_retry:
            alert = False  # retry pending — don't page until retries exhaust
        if not ok and prev_failed:
            alert = False  # still failing — the owner already knows
        if ok and is_routine_tick:
            alert = False  # routine tick succeeded — silent
        if ok and row.get("payload_kind") == "message":
            alert = False  # the message itself already went out
        if alert:
            try:
                self.notifier.publish(
                    "schedule",
                    f"{'✅' if ok else '❌'} scheduled: {row['name']}",
                    summary[:1500],
                )
            except Exception:  # noqa: BLE001
                _log.debug("scheduler notification failed")
        return {
            "id": row["id"], "name": row["name"], "ok": ok,
            "result": summary[:2000], "seconds": round(time.time() - started, 2),
            "next_run": next_run, "will_retry": will_retry,
            "retry_count": retry_count,
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
        tz = str(row.get("timezone") or "")
        if tz and row.get("kind") in ("daily", "cron") and tz not in str(spec):
            spec = f"{spec} [{tz}]"
        next_run = row.get("next_run")
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
            "depends_on": str(row.get("depends_on") or ""),
            "max_retries": int(row.get("max_retries") or 0),
            "retry_delay": float(row.get("retry_delay") or 60),
            "retry_count": int(row.get("retry_count") or 0),
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

