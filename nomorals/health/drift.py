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
    "IllnessWatch",
    "illness_watch",
    "projected_recovery",
    "CHRONIC_DAYS",
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

#: Chronic window (days) for the slow-drift check — the Oura-resilience
#: pattern: weighted average of the last 14 days, recent days weighted
#: more, minimum data required before any verdict.
CHRONIC_DAYS = 14
CHRONIC_MIN_POINTS = 5

#: 14-day average sleep at or below this → chronic sleep-debt signal.
CHRONIC_SLEEP_DEBT_H = 6.5

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
    if kinds & {"sleep_debt", "sleep_decline", "chronic_sleep_debt"}:
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
    if kinds & {"mood_drop", "chronic_mood_slide"}:
        actions.append(RecoveryAction(
            kind="light_day",
            text="🌤️ tomorrow: keep it light",
            detail="I'll note it in your morning briefing — fewer heavy "
                   "blocks, room to breathe."))
    if "chronic_sleep_debt" in kinds or "chronic_mood_slide" in kinds:
        actions.append(RecoveryAction(
            kind="reset_week",
            text="🔁 a real reset week",
            detail="Two weeks of drift is structural, not a bad weekend — "
                   "a lighter week plus a check-in with your doctor or "
                   "coach is the honest fix."))
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


# ── illness early-warning (WHOOP pattern) ────────────────────────────────
# RHR elevation + HRV drop + sleep disturbance moving TOGETHER is the
# classic "something coming on" signature. Gentle, never diagnostic.


@dataclass
class IllnessWatch:
    """Early-warning: recovery signals moving together the wrong way."""
    active: bool
    detail: str = ""
    signals: list[str] = field(default_factory=list)


def illness_watch(*, hrv_ratio: float | None = None,
                  rhr_delta_bpm: float | None = None,
                  sleep_hours: float | None = None,
                  baseline_sleep: float | None = None) -> IllnessWatch:
    """Combine RHR/HRV/sleep into a gentle early-warning. Pure function.

    Needs ≥2 of 3 signals to fire — a single bad night is not illness.
    Coaching language only; every string passes guard_coaching.
    """
    try:
        hits: list[str] = []
        if (rhr_delta_bpm is not None
                and rhr_delta_bpm >= 5):
            hits.append(f"resting heart rate up {rhr_delta_bpm:.0f} bpm "
                        f"vs your baseline")
        if hrv_ratio is not None and hrv_ratio < 0.85:
            hits.append(f"recovery signal (HRV) down "
                        f"{(1 - hrv_ratio) * 100:.0f}% vs baseline")
        if (sleep_hours is not None and baseline_sleep is not None
                and sleep_hours < baseline_sleep - 1.5):
            hits.append(f"sleep {sleep_hours:.1f}h vs your usual "
                        f"{baseline_sleep:.1f}h")
        elif sleep_hours is not None and sleep_hours <= 5.0:
            hits.append(f"only {sleep_hours:.1f}h sleep")
        if len(hits) >= 2:
            detail = ("a few of your recovery signals are moving the "
                      "wrong way together — " + "; ".join(hits) + ". "
                      "Might be worth taking it easy today and seeing "
                      "how you feel tomorrow.")
            guard_coaching(detail)
            return IllnessWatch(active=True, detail=detail, signals=hits)
        return IllnessWatch(active=False)
    except Exception:  # noqa: BLE001
        return IllnessWatch(active=False)


def projected_recovery(values: list[float], *,
                       target: float | None = None) -> str:
    """'At this slope, ~N more nights to baseline.' Honest forecast.

    ``values``: oldest-first daily metric (sleep hours, HRV, mood…).
    Returns a plain-language band, never false precision. Pure.
    """
    try:
        vals = [float(v) for v in values if v is not None]
        if len(vals) < 4:
            return "not enough data to project recovery yet."
        target = float(target) if target is not None else sum(
            vals[:3]) / 3
        recent = sum(vals[-3:]) / 3
        if recent >= target:
            return "you're back at baseline — nice."
        # slope per day over the window
        n = len(vals)
        xs = list(range(n))
        xbar, ybar = sum(xs) / n, sum(vals) / n
        denom = sum((x - xbar) ** 2 for x in xs) or 1.0
        slope = sum((x - xbar) * (y - ybar)
                    for x, y in zip(xs, vals)) / denom
        if slope <= 0:
            return ("the trend is still heading the wrong way — an "
                    "early night and a light day are the highest-"
                    "leverage moves.")
        gap = target - recent
        nights = gap / slope if slope > 0 else float("inf")
        if nights <= 1.5:
            when = "about one more good night"
        elif nights <= 4:
            when = f"roughly {nights:.0f} more good nights"
        else:
            when = "several more good nights"
        text = (f"at the current slope, {when} should bring this "
                f"back to baseline.")
        guard_coaching(text)
        return text
    except Exception:  # noqa: BLE001
        return "couldn't project recovery right now."


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
                 rhr_series: Callable[[], list[tuple[float, float]]] | None = None,
                 ) -> None:
        if community:
            raise PermissionError(
                "drift monitoring is owner-scoped — not available in "
                "community context")
        from ..storage.db import Database
        self.timeline = timeline
        self.hrv_series = hrv_series
        self.activity_series = activity_series
        self.rhr_series = rhr_series
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
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS drift_reports (
                   day TEXT PRIMARY KEY,
                   severity TEXT NOT NULL,
                   kinds TEXT NOT NULL DEFAULT '',
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

        # --- chronic window (14-day, Oura-resilience-weighted) ---
        signals.extend(self._chronic_signals(now))

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
        self._record_report(report, now=now)
        return report

    # ── chronic (14-day) window ──────────────────────────────────────

    @staticmethod
    def _weighted_avg(pts: list[tuple[float, float]]) -> float | None:
        """Recency-weighted average (Oura resilience pattern).

        Linear weights: most recent point counts most. Needs
        CHRONIC_MIN_POINTS before any verdict.
        """
        if len(pts) < CHRONIC_MIN_POINTS:
            return None
        pts = sorted(pts, key=lambda p: p[0])
        total_w = sum(range(1, len(pts) + 1))
        return sum(v * (i + 1) for i, (_, v) in enumerate(pts)) / total_w

    def _chronic_signals(self, now: float) -> list[DriftSignal]:
        """Slow-drift signals over CHRONIC_DAYS. Conservative: needs
        CHRONIC_MIN_POINTS of data, never fires on thin history."""
        signals: list[DriftSignal] = []
        try:
            sleeps = self._series("sleep", CHRONIC_DAYS, now)
            if len(sleeps) >= CHRONIC_MIN_POINTS:
                wavg = self._weighted_avg(sleeps)
                if wavg is not None and wavg <= CHRONIC_SLEEP_DEBT_H:
                    signals.append(DriftSignal(
                        "chronic_sleep_debt",
                        f"sleep has averaged {wavg:.1f}h over the last "
                        f"{CHRONIC_DAYS} days — that's a slow-burning "
                        f"debt (target 7–9h)",
                        value=wavg))
            moods = self._series("mood", CHRONIC_DAYS, now)
            if len(moods) >= CHRONIC_MIN_POINTS:
                vals = [v for _, v in moods]
                wavg = self._weighted_avg(moods)
                if wavg is not None and wavg <= LOW_MOOD:
                    signals.append(DriftSignal(
                        "chronic_mood_slide",
                        f"mood has averaged {wavg:.1f}/5 over the last "
                        f"{CHRONIC_DAYS} days — a slow slide worth "
                        f"noticing"))
        except Exception:  # noqa: BLE001
            _log.debug("chronic signals failed", exc_info=True)
        return signals

    # ── illness early-warning ────────────────────────────────────────

    def illness_watch_check(self, *, now: float | None = None
                            ) -> IllnessWatch:
        """RHR + HRV + sleep moving together the wrong way → gentle flag.

        Needs ≥2 of 3 signals. Never diagnostic, never raises.
        """
        try:
            now = now if now is not None else time.time()
            hrv_ratio = rhr_delta = sleep_h = base_sleep = None
            hrv = self._hrv_3day(now)
            if hrv is not None:
                avg3, baseline = hrv
                hrv_ratio = avg3 / baseline if baseline else None
            if self.rhr_series is not None:
                try:
                    pts = [(ts, v) for ts, v in self.rhr_series()
                           if v is not None and v > 0]
                    recent = [v for ts, v in pts
                              if now - 3 * 86400 < ts <= now]
                    base = [v for ts, v in pts
                            if now - 10 * 86400 < ts <= now - 3 * 86400]
                    if len(recent) >= 2 and len(base) >= 2:
                        rhr_delta = (sum(recent) / len(recent)
                                     - sum(base) / len(base))
                except Exception:  # noqa: BLE001
                    _log.debug("rhr series failed", exc_info=True)
            sleeps = self._series("sleep", MIN_DAYS, now)
            if len(sleeps) >= 2:
                sleep_h = sleeps[-1][1]
                base_sleep = sum(v for _, v in sleeps[:-1]
                                 ) / max(1, len(sleeps) - 1)
            return illness_watch(hrv_ratio=hrv_ratio,
                                 rhr_delta_bpm=rhr_delta,
                                 sleep_hours=sleep_h,
                                 baseline_sleep=base_sleep)
        except Exception:  # noqa: BLE001
            _log.debug("illness_watch_check failed", exc_info=True)
            return IllnessWatch(active=False)

    # ── report history + escalation ──────────────────────────────────

    def _record_report(self, report: DriftReport,
                       *, now: float) -> None:
        try:
            kinds = ",".join(sorted({s.kind for s in report.signals}))
            self._db.execute(
                "INSERT OR REPLACE INTO drift_reports "
                "(day, severity, kinds, ts) VALUES (?,?,?,?)",
                (_day(now), report.severity, kinds, now))
        except Exception:  # noqa: BLE001
            _log.debug("record_report failed", exc_info=True)

    def consecutive_act_days(self, *, now: float | None = None) -> int:
        """How many days in a row ended 'act' (escalation input)."""
        try:
            now = now if now is not None else time.time()
            rows = self._db.query(
                "SELECT day, severity FROM drift_reports "
                "ORDER BY day DESC LIMIT 14")
            streak = 0
            probe = _day(now)
            by_day = {r["day"]: r["severity"] for r in rows}
            while by_day.get(probe) == "act":
                streak += 1
                probe = time.strftime(
                    "%Y-%m-%d",
                    time.localtime(
                        time.mktime(time.strptime(probe, "%Y-%m-%d"))
                        - 86400))
            return streak
        except Exception:  # noqa: BLE001
            return 0

    def escalation_note(self, *, now: float | None = None) -> str:
        """Stronger guidance when 'act' persists 3+ days. Never raises."""
        try:
            n = self.consecutive_act_days(now=now)
            if n < 3:
                return ""
            text = (f"this is day {n} of 'act' drift in a row. At this "
                    f"point the kindest move is structural — talk to "
                    f"your doctor or a coach about what's going on, "
                    f"rather than pushing through another week.")
            return guard_coaching(text)
        except Exception:  # noqa: BLE001
            return ""

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
            esc = self.escalation_note(now=now)
            if esc:
                message += f"\n\n{esc}"
            sender(message)
        except Exception:  # noqa: BLE001
            _log.debug("drift notify failed", exc_info=True)
            return False
        self._db.execute(
            "INSERT INTO drift_pings (day, severity, ts) VALUES (?,?,?)",
            (_day(now), report.severity, now))
        return True
