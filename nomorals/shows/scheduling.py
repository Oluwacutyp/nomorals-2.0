"""Show schedules on their own clock (owner order 2026-11-16).

The shows lane (audit gap R1-4) left automatic episode scheduling
as an open call: a show's schedule was free text and no scheduler
read it. The owner corrected that — an autonomous system does
everything itself — so the schedule field is REAL now. One durable
scheduler job (``__show_schedule_tick__``, every 15m, the same
shape as the roster-diff tick) reads each show's schedule and, when
an episode is due, composes it from the morning pulse and research
through the composer's own segment grammar.

Deliberate boundaries, all by design:
- Episodes are created as DRAFTS. The job never publishes and
  never calls ``set_episode_published`` — publishing stays the
  owner's tap on /show publish, through the publicface gate.
- Times are WAT (Africa/Lagos). WAT has no DST, so the module
  uses a fixed +01:00 offset — exact for Lagos, no tzdata
  dependency, and it never assumes the box clock's zone.
- A show with no schedule is skipped silently. A schedule the
  parser cannot read is noted on the show row once per distinct
  text (the row is the durable journal) and the show is left
  alone until its schedule changes. A due slot with no material
  to read (no briefing, no research) is skipped the same way and
  retried on the next tick — the watermark never advances past a
  slot with no episode.
- Catch-up is bounded: the first run after a show is created
  makes at most ONE episode (today's slot if its time passed,
  else nothing — never a backlog), and a later run makes at most
  seven per show per tick; the rest follow on later ticks.
- One bad show never kills the pass; the tick returns a summary
  and logs the failure.

Layering: this module is layer 0 (``shows`` is unmapped) — stdlib
plus same-package imports only. The morning-pulse and research
readers live in the tool wrapper (tools/autonomy.py, pinned L5),
which passes them in as callables; the tick itself never imports
the briefing or the explore store.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .composer import Show, compose_episode
from .store import ShowStore

__all__ = [
    "SHOW_SCHEDULE_JOB",
    "SHOW_SCHEDULE_SPEC",
    "SHOW_SCHEDULE_TOOL",
    "WAT",
    "ParsedSchedule",
    "parse_schedule",
    "schedule_error",
    "due_slots",
    "build_show_segments",
    "run_show_schedule_tick_once",
    "ensure_show_schedule_job",
]

_log = logging.getLogger(__name__)

#: The durable scheduler job + tool names (roster-diff shape).
SHOW_SCHEDULE_JOB = "show schedule tick"
SHOW_SCHEDULE_SPEC = "every 15m"
SHOW_SCHEDULE_TOOL = "__show_schedule_tick__"

#: WAT — Africa/Lagos is UTC+1 year-round (no DST), so a fixed
#: offset IS the zone. Never the box clock's zone.
WAT = timezone(timedelta(hours=1))

#: Most episodes one tick may compose per show (catch-up bound).
MAX_CATCHUP_EPISODES = 7
#: Most script segments the pulse / research readers may supply.
MAX_PULSE_SEGMENTS = 8
MAX_RESEARCH_SEGMENTS = 3
#: Longest single segment text (the composer estimates duration
#: from words; an unbounded briefing would make an unbounded show).
MAX_SEGMENT_CHARS = 2000

#: Smallest interval worth scheduling (a 1-minute show is spam,
#: not a programme) and the largest (a week — longer is a date,
#: not a rhythm).
MIN_INTERVAL_SECONDS = 5 * 60
MAX_INTERVAL_SECONDS = 7 * 24 * 3600

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")
_DAILY = re.compile(r"^(?:daily|every\s+day)\s+(\d{1,2}):(\d{2})$")
_WEEKLY = re.compile(r"^weekly\s+([a-z]+)\s+(\d{1,2}):(\d{2})$")
_INTERVAL = re.compile(r"^every\s+(\d+)\s*(h|hour|hours|m|min|mins|minute|minutes)$")


@dataclass
class ParsedSchedule:
    """A schedule the parser understood."""

    kind: str  # "daily" | "weekly" | "interval"
    hour: int = 0
    minute: int = 0
    weekday: int = 0  # weekly only: 0 = Monday
    interval_seconds: int = 0  # interval only


def _hhmm(hour: str, minute: str) -> tuple[int, int] | None:
    """A validated (hour, minute) pair, else None."""
    try:
        h, m = int(hour), int(minute)
    except ValueError:
        return None
    if 0 <= h <= 23 and 0 <= m <= 59:
        return h, m
    return None


def parse_schedule(text: str) -> ParsedSchedule | None:
    """Parse a show schedule, leniently. None when it is not a schedule.

    Accepted shapes — the ones a person would write::

        daily 06:00 | every day 06:00 [WAT]
        weekly Mon 06:00 | weekly Monday 18:30 [WAT]
        every 6h | every 90m | every 2 hours

    A trailing ``WAT`` is allowed and ignored (schedules are always
    WAT). Anything else — a bare ``daily``, an impossible hour, an
    interval under five minutes — is refused, not guessed.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered.endswith(" wat"):
        lowered = lowered[:-4].rstrip()
    match = _DAILY.match(lowered)
    if match:
        hhmm = _hhmm(match.group(1), match.group(2))
        if hhmm is None:
            return None
        return ParsedSchedule(kind="daily", hour=hhmm[0],
                              minute=hhmm[1])
    match = _WEEKLY.match(lowered)
    if match:
        weekday = _WEEKDAYS.get(match.group(1))
        if weekday is None:
            return None
        hhmm = _hhmm(match.group(2), match.group(3))
        if hhmm is None:
            return None
        return ParsedSchedule(kind="weekly", hour=hhmm[0],
                              minute=hhmm[1], weekday=weekday)
    match = _INTERVAL.match(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        seconds = amount * 3600 if unit.startswith("h") else amount * 60
        if MIN_INTERVAL_SECONDS <= seconds <= MAX_INTERVAL_SECONDS:
            return ParsedSchedule(kind="interval",
                                  interval_seconds=seconds)
        return None
    return None


def schedule_error(text: str) -> str:
    """The note a refused schedule leaves on its show row."""
    return (
        f"schedule not understood: {text!r} — try 'daily 06:00', "
        f"'weekly Mon 06:00' or 'every 6h'"
    )


def _slot_on(day: Any, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute,
                    tzinfo=WAT)


def _as_wat(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=WAT)
    return moment.astimezone(WAT)


def due_slots(parsed: ParsedSchedule, now: datetime,
              last_slot: datetime | None) -> list[datetime]:
    """Due slots in (last_slot, now] — all WAT, oldest first.

    With no watermark the first run makes at most ONE episode:
    today's slot when its time passed (daily / weekly-on-today),
    else nothing; an interval's first slot is now. A later run
    lists every missed slot, capped at ``MAX_CATCHUP_EPISODES``
    so a long-dormant show never bursts a backlog in one tick.
    """
    now_wat = _as_wat(now)
    if parsed.kind == "interval":
        if last_slot is None:
            return [now_wat.replace(second=0, microsecond=0)]
        slots: list[datetime] = []
        candidate = _as_wat(last_slot) + timedelta(
            seconds=parsed.interval_seconds)
        while (candidate <= now_wat
               and len(slots) < MAX_CATCHUP_EPISODES):
            slots.append(candidate)
            candidate += timedelta(seconds=parsed.interval_seconds)
        return slots
    if last_slot is None:
        if parsed.kind == "weekly" and now_wat.weekday() != parsed.weekday:
            return []
        slot = _slot_on(now_wat.date(), parsed.hour, parsed.minute)
        return [slot] if slot <= now_wat else []
    last_wat = _as_wat(last_slot)
    slots = []
    if parsed.kind == "daily":
        candidate = _slot_on(last_wat.date(), parsed.hour,
                             parsed.minute)
        if candidate <= last_wat:
            candidate += timedelta(days=1)
        step = timedelta(days=1)
    else:  # weekly
        day = last_wat.date()
        while day.weekday() != parsed.weekday:
            day += timedelta(days=1)
        candidate = _slot_on(day, parsed.hour, parsed.minute)
        if candidate <= last_wat:
            candidate += timedelta(days=7)
        step = timedelta(days=7)
    while candidate <= now_wat and len(slots) < MAX_CATCHUP_EPISODES:
        slots.append(candidate)
        candidate += step
    return slots


def build_show_segments(*, pulse: Any = None, research: Any = None,
                        show: Show | None = None) -> list[dict[str, Any]]:
    """Script segments from the pulse + research readers, in the
    composer's own grammar, plus the show's bed track as a bed
    segment. Readers are zero-arg callables returning
    ``[{"title": ..., "text": ...}]`` lists; a reader that raises
    contributes nothing (logged, never fatal). The composer's
    kinds are honoured: only script/voice/ad carry spoken words,
    and the readers supply script segments.
    """
    segments: list[dict[str, Any]] = []
    for reader, cap in ((pulse, MAX_PULSE_SEGMENTS),
                        (research, MAX_RESEARCH_SEGMENTS)):
        if reader is None:
            continue
        try:
            items = reader() or []
        except Exception as exc:  # noqa: BLE001 - a dead source skips
            _log.warning("show schedule segment source failed: %s", exc)
            continue
        taken = 0
        for item in items:
            if taken >= cap:
                break
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            segments.append({"kind": "script",
                             "text": text[:MAX_SEGMENT_CHARS]})
            taken += 1
    if show is not None and str(getattr(show, "bed_track", "") or "").strip():
        segments.append({"kind": "bed",
                         "ref": str(show.bed_track).strip()})
    return segments


def _store_from_context(context: Any) -> ShowStore | None:
    """ShowStore on the context's home path (the /show surface's path)."""
    try:
        settings = getattr(context, "settings", None)
        home = getattr(settings, "home_path", None)
        if home is None:
            return None
        return ShowStore(Path(home) / "data" / "shows.json")
    except Exception:  # noqa: BLE001 - no store, no tick
        return None


def _parse_watermark(raw: Any) -> datetime | None:
    """A stored ISO watermark back to a datetime, else None."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return _as_wat(parsed)


def _tick_one_show(store: ShowStore, row: dict[str, Any],
                   now_wat: datetime, pulse: Any, research: Any,
                   summary: dict[str, Any]) -> None:
    """One show's due slots — compose drafts, advance the watermark."""
    show_id = str(row.get("id") or "")
    schedule = str(row.get("schedule") or "").strip()
    if not schedule:
        summary["skipped"].append({"show": show_id,
                                   "reason": "no schedule"})
        return
    parsed = parse_schedule(schedule)
    if parsed is None:
        note = schedule_error(schedule)
        if row.get("sched_note") != note:
            row["sched_note"] = note
            store.put_show(row)
            _log.warning("show %s: %s", show_id, note)
        summary["skipped"].append({"show": show_id,
                                   "reason": "unparseable schedule"})
        return
    slots = due_slots(parsed, now_wat, _parse_watermark(
        row.get("sched_last_slot")))
    if not slots:
        summary["skipped"].append({"show": show_id,
                                   "reason": "nothing due"})
        return
    show = Show(**row)
    for slot in slots:
        segments = build_show_segments(pulse=pulse, research=research,
                                       show=show)
        spoken = [s for s in segments
                  if s.get("kind") in ("script", "voice", "ad")
                  and str(s.get("text") or "").strip()]
        if not spoken:
            note = (f"no material for {slot:%Y-%m-%d %H:%M} — no "
                    f"briefing or research to read yet")
            if row.get("sched_note") != note:
                row["sched_note"] = note
                store.put_show(row)
            summary["skipped"].append(
                {"show": show_id, "reason": "no material",
                 "slot": slot.isoformat()})
            return  # watermark stays before this slot: retry next tick
        episode = compose_episode(
            show, segments,
            title=f"{show.title} — {slot:%Y-%m-%d %H:%M}",
            created_ts=slot.timestamp())
        stored = store.put_episode(episode.to_dict())
        row["sched_last_slot"] = slot.isoformat()
        row["sched_note"] = ""
        store.put_show(row)
        summary["created"].append(stored["id"])


def run_show_schedule_tick_once(*, context: Any = None,
                                store: ShowStore | None = None,
                                now: datetime | None = None,
                                pulse: Any = None,
                                research: Any = None) -> dict[str, Any]:
    """One scheduled show pass: every show whose schedule is due
    gets a DRAFT episode composed from the pulse + research — never
    a publish. ``pulse`` / ``research`` are zero-arg callables
    returning ``[{"title": ..., "text": ...}]`` (the tool wrapper
    supplies the briefing + explore readers); when absent the pass
    finds no material and skips. ``now`` is an aware datetime
    (tests drive it); when absent it is now in WAT. One bad show
    never kills the pass.
    """
    now_wat = _as_wat(now) if now is not None else datetime.now(WAT)
    if store is None:
        store = _store_from_context(context)
        if store is None:
            return {"ok": False,
                    "error": "no show store available"}
    summary: dict[str, Any] = {"ok": True, "shows": 0, "created": [],
                               "skipped": [], "failed": []}
    for row in sorted(store.list_shows(),
                      key=lambda s: str(s.get("id") or "")):
        summary["shows"] += 1
        try:
            _tick_one_show(store, row, now_wat, pulse, research,
                           summary)
        except Exception as exc:  # noqa: BLE001 - one bad show
            _log.warning("show schedule tick failed for %s: %s",
                         row.get("id"), exc)
            summary["failed"].append(
                {"show": str(row.get("id") or ""), "error": str(exc)})
    return summary


def ensure_show_schedule_job(scheduler: Any, *,
                             spec: str = SHOW_SCHEDULE_SPEC) -> Any:
    """Register the durable show-schedule tick job. Idempotent —
    mirrors ``ensure_group_roster_diff_job``: an existing job at
    the requested cadence is returned as-is, a stale one is
    replaced. Best-effort: never raises.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == SHOW_SCHEDULE_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(SHOW_SCHEDULE_JOB, spec, "tool",
                             {"tool": SHOW_SCHEDULE_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
