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
    "VitalPoint",
    "VitalTrend",
    "SymptomStat",
    "sparkline",
    "parse_measurement",
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

    # ── search ───────────────────────────────────────────────────────

    def search(self, query: str, *, days: int | None = None,
               limit: int = 50) -> list[HealthEvent]:
        """Full-text search across event text. Never raises."""
        try:
            q = (query or "").strip()
            if not q:
                return []
            since = time.time() - days * 86400 if days else 0.0
            like = f"%{q}%"
            rows = self._db.query(
                "SELECT * FROM health_events "
                "WHERE text LIKE ? AND ts >= ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, since, max(1, limit)))
            return [self._row_to_event(r) for r in rows]
        except Exception:  # noqa: BLE001
            _log.debug("health timeline search failed", exc_info=True)
            return []

    # ── vitals trends (measurement events → series + sparklines) ──────

    def vitals_trend(self, days: int = 30) -> list[VitalTrend]:
        """Parse measurement events into per-metric time series.

        Understands "120/80 mmHg" (BP), "72kg"/"160lbs" (weight),
        "37.2" °C (temperature). Tracking only — no interpretation.
        Never raises.
        """
        try:
            since = time.time() - days * 86400
            events = self.timeline(since=since, event_type="measurement",
                                   limit=1000)
            series: dict[str, list[VitalPoint]] = {}
            for ev in events:
                parsed = parse_measurement(ev.value or ev.text,
                                           ev.unit or "")
                for metric, val, unit in parsed:
                    series.setdefault(metric, []).append(
                        VitalPoint(ts=ev.ts, value=val, unit=unit,
                                   label=ev.text))
            trends = []
            for metric, pts in series.items():
                pts.sort(key=lambda p: p.ts)
                vals = [p.value for p in pts]
                trends.append(VitalTrend(
                    metric=metric, unit=pts[0].unit, points=pts,
                    latest=vals[-1], average=sum(vals) / len(vals),
                    minimum=min(vals), maximum=max(vals)))
            trends.sort(key=lambda t: t.metric)
            return trends
        except Exception:  # noqa: BLE001
            _log.debug("vitals_trend failed", exc_info=True)
            return []

    # ── symptom stats (frequency, severity, trend) ────────────────────

    def symptom_stats(self, days: int = 30) -> list[SymptomStat]:
        """Per-symptom frequency, average severity and trend arrow.

        Symptoms are grouped by their first significant word group —
        tracking math only, never a medical read. Never raises.
        """
        try:
            since = time.time() - days * 86400
            events = self.timeline(since=since, event_type="symptom",
                                   limit=1000)
            groups: dict[str, list[HealthEvent]] = {}
            for ev in events:
                key = _symptom_key(ev.text)
                groups.setdefault(key, []).append(ev)
            out = []
            for key, evs in groups.items():
                evs.sort(key=lambda e: e.ts)
                sevs = [e.severity for e in evs if e.severity]
                half = max(1, len(evs) // 2)
                first = [e.severity for e in evs[:half] if e.severity]
                second = [e.severity for e in evs[half:] if e.severity]
                if first and second:
                    a1 = sum(first) / len(first)
                    a2 = sum(second) / len(second)
                    arrow = "↑" if a2 > a1 + 0.4 else (
                        "↓" if a2 < a1 - 0.4 else "→")
                else:
                    arrow = "→"
                out.append(SymptomStat(
                    symptom=key, count=len(evs),
                    avg_severity=(round(sum(sevs) / len(sevs), 1)
                                  if sevs else None),
                    trend=arrow,
                    last_seen=evs[-1].ts,
                    example=evs[-1].text))
            out.sort(key=lambda s: (-s.count, s.symptom))
            return out
        except Exception:  # noqa: BLE001
            _log.debug("symptom_stats failed", exc_info=True)
            return []

    # ── logging streak (retention) ────────────────────────────────────

    def logging_streak(self, *, now: float | None = None) -> int:
        """Consecutive days (ending today/yesterday) with ≥1 logged event."""
        try:
            now = now if now is not None else time.time()
            rows = self._db.query(
                "SELECT DISTINCT date(ts, 'unixepoch', 'localtime') AS d "
                "FROM health_events ORDER BY d DESC LIMIT 400")
            days = {r["d"] for r in rows}
            if not days:
                return 0
            streak = 0
            cursor = time.localtime(now)
            # allow today to be missing (streak counts through yesterday)
            today = time.strftime("%Y-%m-%d", cursor)
            probe = now if today in days else now - 86400
            while True:
                key = time.strftime("%Y-%m-%d", time.localtime(probe))
                if key in days:
                    streak += 1
                    probe -= 86400
                else:
                    break
            return streak
        except Exception:  # noqa: BLE001
            _log.debug("logging_streak failed", exc_info=True)
            return 0

    # ── doctor report (printable handover) ────────────────────────────

    def export_report(self, days: int = 30) -> str:
        """Markdown handover: walk in with a month of facts.

        Timeline + symptom stats + vitals trends + meds. Pure record —
        the doctor interprets, Devon doesn't. Never raises.
        """
        try:
            lines = [f"# Health log — last {days} days",
                     f"_exported {time.strftime('%Y-%m-%d %H:%M')}_", ""]
            stats = self.symptom_stats(days=days)
            if stats:
                lines.append("## Symptoms")
                for s in stats:
                    sev = f", avg severity {s.avg_severity}/5" \
                        if s.avg_severity is not None else ""
                    lines.append(f"- **{s.symptom}** — {s.count}×{sev}, "
                                 f"trend {s.trend}")
                lines.append("")
            trends = self.vitals_trend(days=days)
            if trends:
                lines.append("## Measurements")
                for t in trends:
                    lines.append(f"- **{t.metric}**: latest {t.latest:g} "
                                 f"{t.unit} (avg {t.average:.1f}, range "
                                 f"{t.minimum:g}–{t.maximum:g}) "
                                 f"`{sparkline([p.value for p in t.points])}`")
                lines.append("")
            meds = self.timeline(
                since=time.time() - days * 86400, event_type="medication",
                limit=100)
            if meds:
                lines.append("## Medications (as logged)")
                for e in meds:
                    lines.append(f"- {e.when_str()}: {e.text}")
                lines.append("")
            lines.append("## Full log")
            lines.append(self.summary(days=days))
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            _log.debug("export_report failed", exc_info=True)
            return "couldn't build the report right now."

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


# ── vitals parsing + presentation helpers ────────────────────────────────
# Tracking math only: parse what the user logged, chart it, never interpret.

@dataclass
class VitalPoint:
    ts: float
    value: float
    unit: str = ""
    label: str = ""


@dataclass
class VitalTrend:
    metric: str            # "blood_pressure_systolic", "weight", ...
    unit: str
    points: list[VitalPoint] = field(default_factory=list)
    latest: float = 0.0
    average: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0


@dataclass
class SymptomStat:
    symptom: str
    count: int
    avg_severity: float | None
    trend: str             # ↑ | → | ↓ (severity direction)
    last_seen: float = 0.0
    example: str = ""


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Tiny ASCII trend chart. Pure; never raises."""
    try:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return ""
        if len(vals) == 1:
            return _SPARK[3]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return _SPARK[3] * len(vals)
        return "".join(
            _SPARK[min(7, int((v - lo) / (hi - lo) * 7))] for v in vals)
    except Exception:  # noqa: BLE001
        return ""


_BP_RE = _re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
_WEIGHT_RE = _re.compile(
    r"(\d{2,3}(?:\.\d+)?)\s*(kg|kgs|kilo|kilogram|lbs?|pounds?)\b", _re.I)
_TEMP_RE = _re.compile(r"(3[4-9]|4[0-2])(?:\.(\d))?\s*°?\s*[cC]?\b")
_HR_RE = _re.compile(r"(\d{2,3})\s*(bpm|beats?\s*(?:per|/)\s*min)\b", _re.I)


def parse_measurement(value: str, unit: str = "") -> list[tuple[str, float,
                                                               str]]:
    """Parse a measurement string into (metric, value, unit) triples.

    Handles "120/80" BP, "72kg"/"160 lbs" weight, "37.2" temperature,
    "62 bpm" heart rate. Returns [] when nothing parses. Pure.
    """
    out: list[tuple[str, float, str]] = []
    text = f"{value or ''} {unit or ''}".strip()
    if not text:
        return out
    try:
        m = _BP_RE.search(text)
        if m:
            out.append(("blood_pressure_systolic", float(m.group(1)),
                        "mmHg"))
            out.append(("blood_pressure_diastolic", float(m.group(2)),
                        "mmHg"))
            return out
        m = _WEIGHT_RE.search(text)
        if m:
            v, u = float(m.group(1)), m.group(2).lower()
            if u.startswith("lb") or u.startswith("pound"):
                out.append(("weight", round(v * 0.453592, 1), "kg"))
            else:
                out.append(("weight", v, "kg"))
            return out
        m = _HR_RE.search(text)
        if m:
            out.append(("heart_rate", float(m.group(1)), "bpm"))
            return out
        m = _TEMP_RE.search(text)
        if m:
            whole, frac = m.group(1), m.group(2) or "0"
            out.append(("temperature", float(f"{whole}.{frac}"), "°C"))
            return out
    except Exception:  # noqa: BLE001
        pass
    return out


_SYMPTOM_KEY_WORDS = _re.compile(
    r"\b(headache|migraine|knee|back|neck|shoulder|ankle|wrist|elbow|hip|"
    r"stomach|throat|chest|tooth|ear|eye|skin|joint|muscle|cramp|nausea|"
    r"dizz\w*|fever|cough|sore|tired|fatigue|anxiet\w*|stress\w*|"
    r"insomnia|sleep\w*)\b", _re.I)


def _symptom_key(text: str) -> str:
    """Group symptom text by its most significant word(s)."""
    words = _SYMPTOM_KEY_WORDS.findall(text or "")
    if words:
        seen: list[str] = []
        for w in words:
            wl = w.lower()
            if wl not in seen:
                seen.append(wl)
        return " ".join(seen[:2])
    # fallback: first 4 meaningful words
    toks = [t for t in _re.findall(r"[a-z]{3,}", (text or "").lower())
            if t not in ("the", "and", "with", "for", "today", "this")]
    return " ".join(toks[:4]) or "unspecified"
