"""Scheduler: durable cron-style jobs, run in-process.

Three schedule kinds, parsed from one flexible spec string:

* ``at 2026-09-12 09:00`` (or ISO ``2026-09-12T09:00``) — one-shot
* ``every 30m`` / ``2h`` / ``45s`` — repeating interval
* ``daily 22:00`` (or just ``22:00``) — repeating wall-clock time

Three payload kinds:

* ``message`` — send text to the owner on every live channel (via Notifier)
* ``tool``    — call any registered tool with JSON args
* ``command`` — run a shell command through the sandboxed shell tool

Jobs live in the ``schedule_jobs`` table (migration 15): a restart resumes
them, a disabled job stays put, and one-shot jobs disable themselves after
firing. The tick loop runs on a daemon thread and every due job's outcome is
published through the Notifier, so alerts are durable and multi-channel.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any

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


def parse_schedule_spec(spec: str) -> tuple[str, Any]:
    """Parse a schedule spec → (kind, detail).

    kind: ``at`` (detail = unix ts) | ``every`` (detail = seconds) | ``daily`` (detail = "HH:MM")
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
        return "daily", _parse_hhmm(s[6:].strip())
    # bare forms: "22:00" → daily, "30m" → every, ISO timestamp → at
    if _TIME_RE.match(s):
        return "daily", _parse_hhmm(s)
    if _INTERVAL_RE.match(s):
        return "every", _parse_interval(s)
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


def _next_daily(hhmm: str, now: datetime) -> float:
    hour, minute = (int(x) for x in hhmm.split(":"))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def _like_escape(text: str) -> str:
    """Escape SQL LIKE wildcards so a job ref is matched literally."""
    return (text.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_"))


class Scheduler:
    """Owns the schedule_jobs table and the tick loop."""

    def __init__(self, context: Any, *, gateway: Any = None,
                 tick_seconds: float = 20.0, max_concurrent: int = 2,
                 wall_seconds: float = 300.0) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.notifier = Notifier(context, gateway=gateway)
        self.tick_seconds = max(5.0, float(tick_seconds))
        self.max_concurrent = max(1, int(max_concurrent))
        self.wall_seconds = float(wall_seconds)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running_jobs = 0
        self._lock = threading.Lock()

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def add(
        self,
        name: str,
        spec: str,
        payload_kind: str,
        payload: dict[str, Any],
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
        now = time.time()
        if kind == "at" and detail <= now:
            raise ValueError("one-shot time is in the past")
        next_run = self._initial_next_run(kind, detail, now)
        job_id = new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO schedule_jobs (id, name, kind, spec, payload_kind, payload, "
                "enabled, next_run, last_run, last_result, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, NULL, '', ?, ?)",
                (
                    job_id, (name or "").strip() or "job", kind,
                    str(detail) if kind != "every" else str(int(detail)),
                    payload_kind, json.dumps(payload),
                    next_run, now, now,
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
    def _initial_next_run(self, kind: str, detail: Any, now: float) -> float:
        if kind == "at":
            return float(detail)
        if kind == "every":
            return now + float(detail)
        return _next_daily(str(detail), datetime.fromtimestamp(now))

    def _next_after_run(self, row: dict[str, Any], now: float) -> tuple[float | None, bool]:
        """(next_run, still_enabled) after a job fires."""
        kind = row["kind"]
        if kind == "at":
            return None, False
        if kind == "every":
            return now + float(row["spec"]), True
        return _next_daily(str(row["spec"]), datetime.fromtimestamp(now)), True

    # ── execution ────────────────────────────────────────────────────────────
    def tick(self) -> list[dict[str, Any]]:
        """Run everything due right now. Returns the outcomes (for tests too)."""
        if self.db is None:
            return []
        now = time.time()
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
            with self._lock:
                if self._running_jobs >= self.max_concurrent:
                    # leave it for the next tick rather than dropping it
                    continue
                self._running_jobs += 1
            try:
                results.append(self._execute(dict(row)))
            except Exception:  # noqa: BLE001 - one job must not kill the tick
                _log.exception("scheduler job crashed: %s", row.get("id"))
            finally:
                with self._lock:
                    self._running_jobs -= 1
        return results

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
        next_run, still_enabled = self._next_after_run(row, now)
        # transition-only failure alerts: a job that keeps failing pages
        # ONCE (on the first failure of the streak), not on every run.
        # The previous result is read before the row is updated.
        prev_result = str(row.get("last_result") or "")
        prev_failed = prev_result.startswith("job failed")
        with self.db.transaction():
            self.db.execute(
                "UPDATE schedule_jobs SET last_run = ?, last_result = ?, next_run = ?, "
                "enabled = ?, updated_at = ? WHERE id = ?",
                (now, summary[:2000], next_run, 1 if still_enabled else 0,
                 time.time(), row["id"]),
            )
        # alert the owner — durable + multi-channel via the notifier.
        # Failures alert on the failure transition only (a stuck job must
        # not page every run). Successes notify too, except for routine
        # tick/heartbeat jobs which would spam.
        job_name = row['name'].lower()
        is_routine_tick = 'tick' in job_name or 'heartbeat' in job_name or 'sweep' in job_name
        alert = True
        if not ok and prev_failed:
            alert = False  # still failing — the owner already knows
        if ok and is_routine_tick:
            alert = False  # routine tick succeeded — silent
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
            "next_run": next_run,
        }

    def _run_message(self, payload: dict[str, Any]) -> str:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("empty message payload")
        return f"sent: {text[:200]}"

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
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
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

