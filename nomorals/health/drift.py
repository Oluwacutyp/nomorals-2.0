"""Drift detection + recovery actions (build-map #56) — intervene before burnout.

The 3-day rolling watch: sleep, mood, HRV and activity trends are checked
for degradation. When the trend says trouble is coming, the report forecasts
it ("tonight will be rough because your last 3 nights averaged 4.5h sleep")
and a gentle recovery plan is proposed — suggestions the user approves,
never auto-executed.

Safety rules, all test-enforced:
- Conservative: needs 3 full days of data before any verdict (``MIN_DAYS``).
- Suggestions, not actions: ``schedule_recovery`` only registers reminders
  the user approved.
- Max one proactive ping per day — no nagging.
- No diagnostic language: every public string passes
  :func:`coach.guard_coaching`.
- Very low mood → crisis resources, always.
- Owner-scoped: ``community=True`` raises, mirroring HealthTimeline.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .coach import guard_coaching
from .previsit import CRISIS_RESOURCES

_log = logging.getLogger(__name__)

__all__ = [
    "MIN_DAYS",
    "SLEEP_DEBT_H",
    "SLEEP_SEVERE_H",
    "SLEEP_DECLINE_H",
    "LOW_MOOD",
    "CRISIS_MOOD",
    "HRV_DECLINE_RATIO",
    "ACTIVITY_DROP_RATIO",
    "MAX_PINGS_PER_DAY",
    "DriftSignal",
    "DriftReport",
    "RecoveryAction",
    "DriftMonitor",
    "recovery_plan",
]

# ── thresholds (conservative by design; documented for tuning) ───────────────

#: Minimum days of data before any drift verdict. Two days of bad sleep is
#: a bad weekend; three is a trend.
MIN_DAYS = 3

#: 3-day average sleep at or below this → sleep-debt signal.
SLEEP_DEBT_H = 6.0

#: 3-day average sleep at or below this → severe (auto-escalates to "act").
SLEEP_SEVERE_H = 5.0

#: Total sleep drop across the 3-day window that counts as a decline
#: (latest night vs earliest night).
SLEEP_DECLINE_H = 1.5

#: Average mood (1-5) at or below this → mood-drop signal.
LOW_MOOD = 2.5

#: Any single mood log at or below this → crisis resources included.
CRISIS_MOOD = 2

#: 3-day average HRV below this fraction of the prior baseline → decline.
HRV_DECLINE_RATIO = 0.90

#: 3-day average activity below this fraction of the prior baseline → drop.
ACTIVITY_DROP_RATIO = 0.50

#: Never more than one proactive drift ping per calendar day.
MAX_PINGS_PER_DAY = 1

_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", re.I)


def _sleep_hours(text: str) -> float | None:
    try:
        m = _HOURS_RE.search(text or "")
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


# ── report types ─────────────────────────────────────────────────────────────


@dataclass
class DriftSignal:
    """One detected degradation signal."""
    kind: str  # sleep_debt | sleep_decline | mood_drop | hrv_decline | activity_drop
    detail: str
    days: int = MIN_DAYS
    value: float | None = None  # the measured number behind the signal


@dataclass
class DriftReport:
    """The 3-day verdict. ``severity`` is "watch" (heads-up) or "act"
    (worth a proactive nudge today)."""
    severity: str
    signals: list[DriftSignal] = field(default_factory=list)
    forecast: str = ""
    crisis: bool = False  # very low mood seen → crisis resources included

    def format(self) -> str:
        lines = []
        head = ("🔶 heads-up: I'm seeing some drift" if self.severity == "watch"
                else "🟠 worth acting on: the trend isn't great")
        lines.append(head)
        for s in self.signals:
            lines.append(f"• {s.detail}")
        if self.forecast:
            lines.append(f"\n{self.forecast}")
        if self.crisis:
            lines.append("\nIf things feel heavy, please reach out — "
                         "you don't have to sit with it alone:")
            lines.extend(CRISIS_RESOURCES)
        return guard_coaching("\n".join(lines))


@dataclass
class RecoveryAction:
    """One gentle recovery suggestion. Nothing executes until the user
    approves it; ``schedules`` marks actions that register a reminder
    once approved."""
    kind: str  # wind_down | bedtime_nudge | light_day | rest_workout
    text: str
    detail: str = ""
    schedules: bool = False


def recovery_plan(report: DriftReport) -> list[RecoveryAction]:
    """Gentle, concrete recovery suggestions for a drift report.

    Pure function — returns suggestions only. Approval and scheduling
    happen in :meth:`DriftMonitor.schedule_recovery`.
    """
    kinds = {s.kind for s in report.signals}
    actions: list[RecoveryAction] = []
    if kinds & {"sleep_debt", "sleep_decline"}:
        actions.append(RecoveryAction(
            kind="wind_down",
            text="🛌 wind-down block at 10pm tonight",
            detail="A quiet hour before bed — no screens, low light. "
                   "I can put a reminder on your calendar.",
            schedules=True))
        actions.append(RecoveryAction(
            kind="bedtime_nudge",
            text="⏰ earlier-bedtime nudge at 9:30pm",
            detail="A gentle ping so bedtime doesn't slip again.",
            schedules=True))
    if kinds & {"mood_drop"}:
        actions.append(RecoveryAction(
            kind="light_day",
            text="🌤️ tomorrow: keep it light",
            detail="I'll note it in your morning briefing — fewer heavy "
                   "blocks, room to breathe."))
    if kinds & {"hrv_decline", "activity_drop"}:
        actions.append(RecoveryAction(
            kind="rest_workout",
            text="🧘 skip the intense workout",
            detail="Your recovery signals are down — an easy walk beats "
                   "a hard session today."))
    # De-dupe while preserving order (signals can overlap).
    seen: set[str] = set()
    out: list[RecoveryAction] = []
    for a in actions:
        if a.kind not in seen:
            seen.add(a.kind)
            out.append(a)
    return out


# ── the monitor ──────────────────────────────────────────────────────────────


class DriftMonitor:
    """3-day rolling drift detection over the health timeline.

    Owner-scoped: ``community=True`` raises. HRV and activity come from
    optional callables so the module stays offline-testable; when absent,
    those signals are simply skipped (never fabricated).
    """

    def __init__(self, timeline: Any, *,
                 db_path: str | Path | None = None,
                 community: bool = False,
                 hrv_series: Callable[[], list[tuple[float, float]]] | None = None,
                 activity_series: Callable[[], list[tuple[float, float]]] | None = None,
                 ) -> None:
        if community:
            raise PermissionError(
                "drift monitoring is owner-scoped — not available in "
                "community context")
        from ..storage.db import Database
        self.timeline = timeline
        self.hrv_series = hrv_series
        self.activity_series = activity_series
        path = Path(db_path) if db_path else (
            Path.home() / ".nomorals" / "health" / "drift.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = Database(path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS drift_pings (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   day TEXT NOT NULL,
                   severity TEXT NOT NULL,
                   ts REAL NOT NULL
               )""")

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass

    # ── data gathering ───────────────────────────────────────────────

    def _series(self, event_type: str, days: int,
                now: float | None = None) -> list[tuple[float, float]]:
        """Per-day values for the last ``days`` days, oldest first.

        Sleep → parsed hours; mood → severity (1-5). Days with no data
        are skipped.
        """
        now = now if now is not None else time.time()
        events = self.timeline.timeline(since=now - days * 86400,
                                       event_type=event_type, limit=500)
        by_day: dict[str, list[float]] = {}
        for ev in events:
            if event_type == "sleep":
                v = _sleep_hours(ev.text)
            elif event_type == "mood":
                v = float(ev.severity) if ev.severity else None
            else:
                v = None
            if v is None:
                continue
            by_day.setdefault(_day(ev.ts), []).append(v)
        out = [(time.mktime(time.strptime(d, "%Y-%m-%d")),
                sum(v) / len(v)) for d, v in sorted(by_day.items())]
        return out

    # ── detection ──────────────────────────────────────────────────

    def check(self, *, now: float | None = None) -> DriftReport | None:
        """Run the 3-day drift check. Returns None when stable or when
        there isn't enough data — conservative by design."""
        now = now if now is not None else time.time()
        signals: list[DriftSignal] = []
        crisis = False

        # --- sleep ---
        sleeps = self._series("sleep", MIN_DAYS, now)
        if len(sleeps) >= MIN_DAYS:
            hours = [h for _, h in sleeps]
            avg = sum(hours) / len(hours)
            if avg <= SLEEP_SEVERE_H:
                signals.append(DriftSignal(
                    "sleep_debt",
                    f"your last {MIN_DAYS} nights averaged {avg:.1f}h sleep "
                    f"(target 7–9h)",
                    value=avg))
            elif avg <= SLEEP_DEBT_H:
                signals.append(DriftSignal(
                    "sleep_debt",
                    f"your last {MIN_DAYS} nights averaged {avg:.1f}h sleep "
                    f"(target 7–9h)",
                    value=avg))
            decline = hours[0] - hours[-1]
            if decline >= SLEEP_DECLINE_H:
                signals.append(DriftSignal(
                    "sleep_decline",
                    f"sleep has dropped {decline:.1f}h across the last "
                    f"{MIN_DAYS} nights ({hours[0]:.1f}h → {hours[-1]:.1f}h)"))

        # --- mood ---
        moods = self._series("mood", MIN_DAYS, now)
        if len(moods) >= MIN_DAYS:
            vals = [v for _, v in moods]
            avg_mood = sum(vals) / len(vals)
            if any(v <= CRISIS_MOOD for v in vals):
                crisis = True
            if avg_mood <= LOW_MOOD:
                signals.append(DriftSignal(
                    "mood_drop",
                    f"mood has averaged {avg_mood:.1f}/5 over the last "
                    f"{MIN_DAYS} days"))
            elif all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                signals.append(DriftSignal(
                    "mood_drop",
                    f"mood has slipped each day for {MIN_DAYS} days "
                    f"({vals[0]:.0f} → {vals[-1]:.0f})"))

        # --- HRV (optional series) ---
        hrv = self._hrv_3day(now)
        if hrv is not None:
            avg3, baseline = hrv
            if baseline > 0 and avg3 / baseline < HRV_DECLINE_RATIO:
                signals.append(DriftSignal(
                    "hrv_decline",
                    "your recovery signal (HRV) is running about "
                    f"{(1 - avg3 / baseline) * 100:.0f}% below its recent "
                    "baseline"))

        # --- activity (optional series) ---
        act = self._activity_3day(now)
        if act is not None:
            avg3, baseline = act
            if baseline > 0 and avg3 / baseline < ACTIVITY_DROP_RATIO:
                signals.append(DriftSignal(
                    "activity_drop",
                    "movement is way down the last few days — about "
                    f"{(1 - avg3 / baseline) * 100:.0f}% below your norm"))

        if not signals and not crisis:
            return None

        severity = self._severity(signals, crisis)
        report = DriftReport(
            severity=severity,
            signals=signals,
            forecast=self._forecast(signals),
            crisis=crisis)
        # guard_coaching runs inside format(); validate eagerly so a
        # report can never be constructed with banned phrasing.
        guard_coaching(" ".join(s.detail for s in signals) + report.forecast)
        return report

    def _severity(self, signals: list[DriftSignal], crisis: bool) -> str:
        kinds = {s.kind for s in signals}
        if crisis:
            return "act"
        if "sleep_debt" in kinds and "hrv_decline" in kinds:
            return "act"  # poor recovery + poor sleep is the burnout combo
        if any(s.kind == "sleep_debt" and s.value is not None
               and s.value <= SLEEP_SEVERE_H for s in signals):
            return "act"  # severe sleep debt
        if len(kinds) >= 2:
            return "act"
        return "watch"

    def _forecast(self, signals: list[DriftSignal]) -> str:
        kinds = {s.kind for s in signals}
        if "sleep_debt" in kinds:
            avg_txt = next(
                (s.detail for s in signals if s.kind == "sleep_debt"), "")
            m = re.search(r"averaged ([\d.]+h)", avg_txt)
            avg = m.group(1) if m else "less than usual"
            return (f"Tonight will likely be rough — your last {MIN_DAYS} "
                    f"nights averaged {avg} sleep. An early night would "
                    f"change the trajectory.")
        if "mood_drop" in kinds:
            return ("The next few days tend to follow the same slope — a "
                    "lighter schedule and an early night are the two "
                    "highest-leverage moves right now.")
        if "hrv_decline" in kinds:
            return ("Recovery is trending down, so today is a good day to "
                    "go easy — intensity can wait until the signal "
                    "rebounds.")
        return ""

    # ── optional biometric series ───────────────────────────────────

    def _hrv_3day(self, now: float) -> tuple[float, float] | None:
        """(3-day avg, prior 7-day baseline) or None when no data."""
        if self.hrv_series is None:
            return None
        try:
            pts = [(ts, v) for ts, v in self.hrv_series()
                   if v is not None and v > 0]
        except Exception:  # noqa: BLE001
            _log.debug("hrv series failed", exc_info=True)
            return None
        recent = [v for ts, v in pts if now - 3 * 86400 < ts <= now]
        base = [v for ts, v in pts if now - 10 * 86400 < ts <= now - 3 * 86400]
        if len(recent) < 2 or len(base) < 2:
            return None
        return sum(recent) / len(recent), sum(base) / len(base)

    def _activity_3day(self, now: float) -> tuple[float, float] | None:
        if self.activity_series is None:
            return None
        try:
            pts = [(ts, v) for ts, v in self.activity_series()
                   if v is not None and v >= 0]
        except Exception:  # noqa: BLE001
            _log.debug("activity series failed", exc_info=True)
            return None
        recent = [v for ts, v in pts if now - 3 * 86400 < ts <= now]
        base = [v for ts, v in pts if now - 10 * 86400 < ts <= now - 3 * 86400]
        if len(recent) < 2 or len(base) < 2:
            return None
        return sum(recent) / len(recent), sum(base) / len(base)

    # ── recovery scheduling (approval-gated) ─────────────────────────

    def schedule_recovery(self, actions: list[RecoveryAction], *,
                          approved_kinds: set[str] | None,
                          schedule_fn: Callable[..., Any] | None = None,
                          now: float | None = None) -> list[str]:
        """Register reminders for the APPROVED recovery actions only.

        ``approved_kinds`` is the user's explicit approval — None or an
        empty set schedules nothing. Returns the kinds that were
        scheduled. ``schedule_fn(task_id, run_at, action, parameters)``
        defaults to a no-op recorder; production passes the scheduler.
        """
        now = now if now is not None else time.time()
        if not approved_kinds:
            return []
        approved = {a.kind for a in actions} & set(approved_kinds)
        scheduled: list[str] = []
        for action in actions:
            if action.kind not in approved or not action.schedules:
                continue
            run_at = self._nudge_time(action.kind, now)
            try:
                if schedule_fn is not None:
                    schedule_fn(
                        task_id=f"drift-{action.kind}-{int(now)}",
                        run_at=run_at,
                        action="drift.nudge",
                        parameters={"kind": action.kind,
                                    "text": action.text,
                                    "detail": action.detail})
                scheduled.append(action.kind)
            except Exception:  # noqa: BLE001
                _log.debug("drift schedule failed for %s", action.kind,
                           exc_info=True)
        return scheduled

    @staticmethod
    def _nudge_time(kind: str, now: float) -> float:
        """Same-evening nudge times: wind-down 10pm, bedtime 9:30pm."""
        lt = time.localtime(now)
        target_hour = {"wind_down": 22, "bedtime_nudge": 21.5}.get(kind, 21)
        hour, minute = int(target_hour), int((target_hour % 1) * 60)
        target = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                              hour, minute, 0, 0, 0, lt.tm_isdst))
        if target <= now:
            target += 86400  # already past today → tomorrow
        return target

    # ── proactive delivery (max once per day) ────────────────────────

    def pings_today(self, *, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM drift_pings WHERE day = ?",
            (_day(now),))
        return int(rows[0]["n"]) if rows else 0

    def maybe_notify(self, report: DriftReport | None, sender: Callable[[str], Any],
                     *, now: float | None = None) -> bool:
        """Surface an "act"-level report through the proactive path.

        Returns True if a ping was sent. "watch" reports stay quiet
        (they belong in the briefing, not a ping). Never more than
        :data:`MAX_PINGS_PER_DAY` per day.
        """
        if report is None or report.severity != "act":
            return False
        now = now if now is not None else time.time()
        if self.pings_today(now=now) >= MAX_PINGS_PER_DAY:
            return False
        try:
            message = report.format()
            plan = recovery_plan(report)
            if plan:
                message += ("\n\nWant me to set any of these up? " +
                            " / ".join(a.text for a in plan))
            sender(message)
        except Exception:  # noqa: BLE001
            _log.debug("drift notify failed", exc_info=True)
            return False
        self._db.execute(
            "INSERT INTO drift_pings (day, severity, ts) VALUES (?,?,?)",
            (_day(now), report.severity, now))
        return True
