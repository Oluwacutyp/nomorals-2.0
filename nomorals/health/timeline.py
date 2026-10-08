"""Patient-side health timeline — "your health story in one place."

TRACKING ONLY. Devon is NOT a doctor. This module records what the user
reports — symptoms, visits, medications, measurements — and never
interprets medically. It names no conditions, replaces no therapy,
and gives no medication advice. Ever.

Privacy: owner-scoped ONLY. The health DB lives at
``~/.nomorals/health/timeline.db`` and is never touched by community /
social code. ``HealthTimeline`` refuses to initialize in a community
context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "HealthTimeline",
    "HealthEvent",
    "EVENT_TYPES",
    "HealthEvent",
    "HealthTimeline",
    "health_db_path",
    "parse_health_note",
    "BANNED_PHRASES",
]

#: Valid event types.
EVENT_TYPES = (
    "symptom",      # "knee hurt today", severity 1-5
    "visit",        # doctor/hospital visit, notes
    "medication",   # name, dose, started/stopped — TRACKING only
    "measurement",  # BP, weight, temperature — value + unit
    "mood",         # 1-5 + note
    "sleep",        # hours + quality
    "note",         # freeform
)

#: Phrases that must NEVER appear in any output of this module.
#: Tested in test_health_timeline.py.
BANNED_PHRASES = (
    "sounds like",
    "this could be",
    "you might have",
    "you should take",
    "diagnos",
    "prescrib",
    "i recommend you take",
    "try taking",
)


def health_db_path() -> Path:
    """Owner-scoped health DB path. Never under community scope."""
    return Path.home() / ".nomorals" / "health" / "timeline.db"


@dataclass
class HealthEvent:
    """One logged health event. The user's own words, verbatim."""
    id: str
    event_type: str
    text: str              # user's own words, verbatim
    ts: float = 0.0        # unix timestamp
    severity: int | None = None   # 1-5, symptom/mood only
    value: str = ""        # measurement value, e.g. "120/80"
    unit: str = ""         # measurement unit, e.g. "mmHg"
    source: str = "chat"   # chat | voice | manual
    extra: dict[str, Any] = field(default_factory=dict)

    def when_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))


class HealthTimeline:
    """Owner-scoped health event timeline. Tracking only — never interprets.

    Pass ``community=True`` and construction raises: health data is never
    available in community/social contexts.
    """

    def __init__(self, db_path: str | Path | None = None, *,
                 community: bool = False) -> None:
        if community:
            raise PermissionError(
                "health timeline is owner-scoped — not available in "
                "community context")
        from ..storage.db import Database
        path = Path(db_path) if db_path else health_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = Database(path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS health_events (
                   id TEXT PRIMARY KEY,
                   event_type TEXT NOT NULL,
                   text TEXT NOT NULL,
                   ts REAL NOT NULL,
                   severity INTEGER,
                   value TEXT NOT NULL DEFAULT '',
                   unit TEXT NOT NULL DEFAULT '',
                   source TEXT NOT NULL DEFAULT 'chat',
                   extra TEXT NOT NULL DEFAULT '{}'
               )""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_ts ON health_events(ts)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_type "
            "ON health_events(event_type)")

    # ── write ──────────────────────────────────────────────────────────

    def log(self, event_type: str, text: str, *,
            severity: int | None = None,
            value: str = "", unit: str = "",
            source: str = "chat",
            ts: float | None = None,
            **extra: Any) -> HealthEvent:
        """Log one event. Never raises; validates severity."""
        import json
        import uuid
        event_type = (event_type or "").strip().lower()
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"unknown health event type {event_type!r}; "
                f"valid: {', '.join(EVENT_TYPES)}")
        text = (text or "").strip()
        if not text:
            raise ValueError("health event text must not be empty")
        if severity is not None:
            severity = int(severity)
            if not 1 <= severity <= 5:
                raise ValueError("severity must be 1-5")
        ev = HealthEvent(
            id=uuid.uuid4().hex[:12], event_type=event_type, text=text,
            ts=ts if ts is not None else time.time(),
            severity=severity, value=value, unit=unit,
            source=source, extra=dict(extra))
        self._db.execute(
            "INSERT INTO health_events "
            "(id, event_type, text, ts, severity, value, unit, source, extra)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (ev.id, ev.event_type, ev.text, ev.ts, ev.severity,
             ev.value, ev.unit, ev.source, json.dumps(ev.extra)))
        return ev

    # ── read ───────────────────────────────────────────────────────────

    def timeline(self, *, since: float | None = None,
                 event_type: str | None = None,
                 limit: int = 200) -> list[HealthEvent]:
        """Chronological events, oldest first. Never raises."""
        try:
            q = "SELECT * FROM health_events"
            clauses: list[str] = []
            args: list[Any] = []
            if since is not None:
                clauses.append("ts >= ?")
                args.append(since)
            if event_type:
                clauses.append("event_type = ?")
                args.append(event_type.strip().lower())
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY ts ASC LIMIT ?"
            args.append(limit)
            return [self._row_to_event(r)
                    for r in self._db.query(q, tuple(args))]
        except Exception:  # noqa: BLE001
            _log.debug("health timeline read failed", exc_info=True)
            return []

    def summary(self, days: int = 30) -> str:
        """Human-readable recap — for the user, or to show a doctor.

        Pure record-keeping. No interpretation, no advice.
        """
        since = time.time() - days * 86400
        events = self.timeline(since=since)
        if not events:
            return f"nothing logged in the last {days} days."
        lines = [f"🩺 health recap — last {days} days "
                 f"({len(events)} entries):"]
        for ev in events:
            sev = f" [{ev.severity}/5]" if ev.severity else ""
            val = f" — {ev.value} {ev.unit}".rstrip() if ev.value else ""
            lines.append(f"• {ev.when_str()} · {ev.event_type}{sev}: "
                         f"{ev.text}{val}")
        lines.append("")
        lines.append("this is your own log — show it to your doctor "
                     "for medical guidance.")
        return "\n".join(lines)

    def _row_to_event(self, row: Any) -> HealthEvent:
        import json
        get = row.get if hasattr(row, "get") else row.__getitem__
        try:
            extra = json.loads(get("extra") or "{}")
        except Exception:  # noqa: BLE001
            extra = {}
        sev = get("severity")
        return HealthEvent(
            id=str(get("id")), event_type=str(get("event_type")),
            text=str(get("text")), ts=float(get("ts")),
            severity=int(sev) if sev is not None else None,
            value=str(get("value") or ""), unit=str(get("unit") or ""),
            source=str(get("source") or "chat"),
            extra=extra if isinstance(extra, dict) else {})

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass


# ── natural-language ingestion ──────────────────────────────────────────
# "log: headache, 3pm" / "my knee hurt today" → structured events.
# Heuristic, tracking-only: we record what the user said, verbatim.

import re as _re

_SYMPTOM_WORDS = _re.compile(
    r"\b(headache|migraine|pain|hurt|ache|nausea|dizzy|dizziness|fever|"
    r"cough|sore|tired|fatigue|cramp|sick|vomit|diarrhea|rash|swell|"
    r"bleed|numb|anxious|anxiety|stress)\b", _re.IGNORECASE)

_MEDICATION_WORDS = _re.compile(
    r"\b(took|taking|started|stopped|dose|pill|tablet|mg|prescription)\b",
    _re.IGNORECASE)

_MEASUREMENT_WORDS = _re.compile(
    r"\b(bp|blood pressure|weight|weigh|temperature|temp|"
    r"\d{2,3}\s*/\s*\d{2,3}|kg\b|lbs?)\b", _re.IGNORECASE)

_SLEEP_WORDS = _re.compile(
    r"\b(slept|sleep|insomnia|hours? of sleep)\b", _re.IGNORECASE)

_MOOD_WORDS = _re.compile(
    r"\b(mood|feeling|feel|happy|sad|depressed|okay|fine|great|"
    r"terrible|awful)\b", _re.IGNORECASE)

_FEELING_RE = _re.compile(r"\bfeel(ing)?\b", _re.IGNORECASE)

_VISIT_WORDS = _re.compile(
    r"\b(doctor|hospital|clinic|appointment|dentist|checkup|"
    r"saw dr|visited)\b", _re.IGNORECASE)

_SEVERITY_RE = _re.compile(r"\b([1-5])\s*/\s*5\b|\bseverity\s*([1-5])\b",
                           _re.IGNORECASE)


def parse_health_note(text: str) -> dict[str, Any]:
    """Turn free text into a structured health event draft.

    Returns {"event_type": ..., "text": <verbatim>, "severity": ...}.
    The original text is ALWAYS preserved verbatim.
    """
    raw = (text or "").strip()
    body = raw
    # strip a leading "log:" / "health log:" prefix
    m = _re.match(r"^(?:health\s+)?log\s*:\s*(.+)$", body, _re.IGNORECASE)
    if m:
        body = m.group(1).strip()

    severity: int | None = None
    sm = _SEVERITY_RE.search(body)
    if sm:
        severity = int(sm.group(1) or sm.group(2))

    if _VISIT_WORDS.search(body):
        etype = "visit"
    elif _MEDICATION_WORDS.search(body):
        etype = "medication"
    elif _MEASUREMENT_WORDS.search(body):
        etype = "measurement"
    elif _SLEEP_WORDS.search(body):
        etype = "sleep"
    elif _FEELING_RE.search(body) and _MOOD_WORDS.search(body):
        # "feeling anxious" — the feeling-frame marks mood, not symptom
        etype = "mood"
    elif _SYMPTOM_WORDS.search(body):
        etype = "symptom"
    elif _MOOD_WORDS.search(body):
        etype = "mood"
    else:
        etype = "note"
    return {"event_type": etype, "text": raw, "severity": severity}
