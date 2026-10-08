"""Ask-your-data coaching — HealthKit / Health Connect (build-map #53).

"Ask your biometrics questions." Plain-language answers over the user's own
synced health data: sleep, recovery/readiness, activity.

CRITICAL POSITIONING: coaching, NOT diagnosis. "Your HRV is trending down
and sleep was short — consider a light day" is coaching. Naming a condition,
suggesting a disease, or interpreting symptoms is forbidden and
test-enforced via :data:`COACH_BANNED_PHRASES` / :func:`guard_coaching`.

Data comes from ``health-cli`` (Apple HealthKit or Google Health Connect —
whichever has data). No data synced → an honest "no health data" message.
Numbers are never fabricated.

Privacy: owner-scoped ONLY. ``HealthCoach`` refuses to initialize in a
community context, mirroring :class:`HealthTimeline`.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .timeline import BANNED_PHRASES as _TIMELINE_BANNED

_log = logging.getLogger(__name__)

__all__ = [
    "HealthCoach",
    "HealthDataSource",
    "Readiness",
    "Answer",
    "COACH_BANNED_PHRASES",
    "guard_coaching",
    "parse_intent",
    "ReadinessBriefingProvider",
]

#: Phrases that must NEVER appear in coach output. Coaching, not diagnosis.
#: Extends the timeline's list (which already bans "sounds like",
#: "you might have", "diagnos", "prescrib", ...).
COACH_BANNED_PHRASES = _TIMELINE_BANNED + (
    "you have a ",
    "you have an ",
    "you suffer",
    "suffering from",
    "disease",
    "disorder",
    "syndrome",
    "deficiency",
    "medical condition",
    "chronic ",
)


def guard_coaching(text: str) -> str:
    """Raise if ``text`` contains a banned diagnostic phrase.

    Every public coach output passes through this before returning.
    Test-enforced in tests/test_health_coach.py.
    """
    lowered = (text or "").lower()
    for phrase in COACH_BANNED_PHRASES:
        if phrase.lower() in lowered:
            raise ValueError(
                f"coach output contains banned diagnostic phrase: {phrase!r}")
    return text


# ── data source ────────────────────────────────────────────────────────────

_METRIC_FIELDS = (
    "step_count,distance_walking_running_meters,active_energy_burned_kcal,"
    "heart_rate_variability_ms,resting_hr_average_bpm,vo2_max"
)


class HealthDataSource:
    """Thin wrapper over ``health-cli``. Provider-agnostic: picks whichever
    of healthkit / healthconnect actually has data. Never raises on CLI
    failure — returns empty and lets the caller be honest about it."""

    PROVIDERS = ("healthkit", "healthconnect")

    def __init__(self, binary: str | None = None,
                 timeout_secs: int = 20) -> None:
        self.binary = binary or shutil.which("health-cli") or "health-cli"
        self.timeout = timeout_secs
        self._provider: str | None = None

    # -- low-level ------------------------------------------------------

    def _run(self, *args: str) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [self.binary, *args], capture_output=True, text=True,
                timeout=self.timeout)
            if proc.returncode != 0:
                _log.debug("health-cli failed (%s): %s",
                           proc.returncode, (proc.stderr or "")[:200])
                return {}
            return json.loads(proc.stdout or "{}")
        except Exception as exc:  # noqa: BLE001 — CLI trouble → no data
            _log.debug("health-cli error: %s", exc)
            return {}

    def _records(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        recs = payload.get("records") or []
        return [r for r in recs if isinstance(r, dict)]

    # -- provider resolution ---------------------------------------------

    def status(self, provider: str) -> dict[str, Any]:
        return self._run("status", "--provider", provider,
                         "--timeout-secs", str(self.timeout))

    def resolve_provider(self) -> str | None:
        """Whichever provider has data (most synced records wins)."""
        if self._provider:
            return self._provider
        best, best_n = None, 0
        for p in self.PROVIDERS:
            st = self.status(p)
            n = sum(int(c.get("record_count") or 0)
                    for c in (st.get("categories") or [])
                    if isinstance(c, dict))
            if n > best_n:
                best, best_n = p, n
        self._provider = best
        return best

    @property
    def provider(self) -> str | None:
        return self.resolve_provider()

    def has_data(self) -> bool:
        return self.provider is not None

    # -- queries ---------------------------------------------------------

    def _q(self, *args: str) -> list[dict[str, Any]]:
        p = self.provider
        if not p:
            return []
        return self._records(self._run(*args, "--provider", p))

    def metrics(self, start: date, end: date) -> list[dict[str, Any]]:
        return self._q(
            "query", "metrics",
            "--start-date", start.isoformat(), "--end-date", end.isoformat(),
            "--fields", _METRIC_FIELDS,
            "--timeout-secs", str(self.timeout))

    def sleep_sessions(self, start: date, end: date) -> list[dict[str, Any]]:
        return self._q(
            "query", "sessions", "--category", "sleep",
            "--start-date", start.isoformat(), "--end-date", end.isoformat(),
            "--timeout-secs", str(self.timeout))

    def workouts(self, start: date, end: date) -> list[dict[str, Any]]:
        return self._q(
            "query", "sessions", "--category", "workout",
            "--start-date", start.isoformat(), "--end-date", end.isoformat(),
            "--timeout-secs", str(self.timeout))


# ── intent parsing ─────────────────────────────────────────────────────────

_INTENTS: list[tuple[str, re.Pattern[str]]] = [
    ("recovery", re.compile(
        r"\b(recover\w*|readiness|readiness score|hrv|overtrain|"
        r"rest day|take it easy|push today)\b", re.IGNORECASE)),
    ("sleep", re.compile(
        r"\b(sleep|slept|sleeping|nap|naps|insomnia|bedtime|"
        r"\bwake\b|\bwoke\b|sleep quality)\b", re.IGNORECASE)),
    ("activity", re.compile(
        r"\b(active|activity|steps|\bstep\b|workout|workouts|exercise|"
        r"training|train|run|ran|running|walk|walked|calories|distance|"
        r"move|gym)\b", re.IGNORECASE)),
]

_WINDOW_RE = re.compile(
    r"\b(this week|last week|past week|this month|last month|past month|"
    r"yesterday|today|tonight|last night)\b", re.IGNORECASE)


def parse_intent(question: str) -> tuple[str, int]:
    """→ (intent, window_days). intent ∈ sleep|recovery|activity|overview."""
    q = (question or "").lower()
    intent = "overview"
    for name, rx in _INTENTS:
        if rx.search(q):
            intent = name
            break
    m = _WINDOW_RE.search(q)
    days = 7
    if m:
        w = m.group(1).lower()
        if w in ("yesterday", "today", "tonight", "last night"):
            days = 1
        elif "month" in w:
            days = 30
    return intent, days


# ── formatting helpers ─────────────────────────────────────────────────────

def _fmt_dur(hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}h {m:02d}m"


def _fmt_num(n: float) -> str:
    return f"{int(round(n)):,}"


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _sleep_hours(sess: dict[str, Any]) -> float | None:
    """Asleep hours from a sleep session record."""
    in_bed = _as_float(sess.get("sleep_in_bed_duration_sec"))
    awake = _as_float(sess.get("sleep_awake_duration_sec"))
    if in_bed is not None:
        asleep = in_bed - (awake or 0.0)
        return max(0.0, asleep / 3600.0)
    total = sum(_as_float(sess.get(k)) or 0.0 for k in (
        "sleep_core_duration_sec", "sleep_deep_duration_sec",
        "sleep_rem_duration_sec", "sleep_asleep_unspecified_duration_sec"))
    return total / 3600.0 if total > 0 else None


def _efficiency(sess: dict[str, Any]) -> float | None:
    e = _as_float(sess.get("sleep_efficiency"))
    if e is None:
        return None
    return e / 100.0 if e > 1.0 else e


def _session_time(sess: dict[str, Any], key: str) -> str:
    v = sess.get(key) or ""
    # "2026-10-07T22:45:00-04:00" → "10:45 PM"
    m = re.search(r"T(\d{2}):(\d{2})", str(v))
    if not m:
        return ""
    h, mi = int(m.group(1)), m.group(2)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{mi} {suffix}"


# ── public result types ────────────────────────────────────────────────────

@dataclass
class Answer:
    text: str
    intent: str
    numbers: dict[str, Any] = field(default_factory=dict)
    has_data: bool = True


@dataclass
class Readiness:
    level: str  # high | moderate | low | unknown (no data)
    score: float  # 0-100
    reasons: list[str] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    has_data: bool = True

    def format(self) -> str:
        if not self.has_data:
            return NO_DATA_MSG
        emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}.get(
            self.level, "⚪")
        lines = [f"{emoji} readiness: **{self.level}** "
                 f"({self.score:.0f}/100)"]
        for r in self.reasons:
            lines.append(f"• {r}")
        if self.suggestion:
            lines.append(f"\n{self.suggestion}")
        return guard_coaching("\n".join(lines))


NO_DATA_MSG = (
    "no health data synced yet — I can't answer from real numbers until "
    "your device shares them. On iPhone: Health app → profile → Apps → "
    "allow the sync; on Android: Health Connect → App permissions → allow. "
    "Then ask me again and I'll work from your actual data.")


# ── the coach ──────────────────────────────────────────────────────────────

class HealthCoach:
    """Ask-your-data coaching over synced biometrics.

    Owner-scoped: ``community=True`` raises, mirroring HealthTimeline.
    """

    def __init__(self, source: HealthDataSource | None = None,
                 timeline: Any | None = None,
                 community: bool = False) -> None:
        if community:
            raise PermissionError(
                "health coaching is owner-scoped — not available in "
                "community context")
        self.source = source or HealthDataSource()
        self._timeline = timeline  # HealthTimeline or None (lazy)

    # -- timeline logging ------------------------------------------------

    def _log_insight(self, text: str) -> None:
        """Coaching insights land in the health timeline (build-map #51)."""
        try:
            tl = self._timeline
            if tl is None:
                from .timeline import HealthTimeline
                tl = self._timeline = HealthTimeline()
            tl.log("note", text, source="coach")
        except Exception:  # noqa: BLE001 — logging never breaks coaching
            _log.debug("coach timeline log failed", exc_info=True)

    # -- Q&A -------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Answer a natural question from the user's real data."""
        question = (question or "").strip()
        if not question:
            return Answer(
                "ask me about your sleep, recovery, or activity — e.g. "
                "\"how did I sleep this week?\" or \"am I recovering?\"",
                intent="overview", has_data=self.source.has_data())
        if not self.source.has_data():
            return Answer(NO_DATA_MSG, intent="overview", has_data=False)
        intent, days = parse_intent(question)
        if intent == "sleep":
            return self._answer_sleep(question, days)
        if intent == "recovery":
            r = self.readiness(log=False)
            return Answer(r.format(), intent="recovery",
                          numbers=r.numbers, has_data=r.has_data)
        if intent == "activity":
            return self._answer_activity(question, days)
        return self._answer_overview()

    # -- sleep -----------------------------------------------------------

    def _sleep_window(self, days: int) -> tuple[list[dict], date, date]:
        end = date.today()
        start = end - timedelta(days=days)
        return self.source.sleep_sessions(start, end), start, end

    def _answer_sleep(self, question: str, days: int) -> Answer:
        sessions, start, end = self._sleep_window(days)
        sessions = sorted(sessions,
                          key=lambda s: str(s.get("end_datetime") or ""))
        if not sessions:
            return Answer(
                f"no sleep sessions synced for the last {days} day(s) — "
                "your device may not track sleep, or it hasn't synced yet.",
                intent="sleep", has_data=True)
        hours = [h for s in sessions if (h := _sleep_hours(s)) is not None]
        last = sessions[-1]
        last_h = _sleep_hours(last)
        eff = _efficiency(last)
        nums: dict[str, Any] = {"sessions": len(sessions)}
        lines = []
        if last_h is not None:
            bed = _session_time(last, "start_datetime")
            wake = _session_time(last, "end_datetime")
            span = f" ({bed} → {wake})" if bed and wake else ""
            lines.append(f"last night: **{_fmt_dur(last_h)}**{span}")
            nums["last_night_hours"] = round(last_h, 2)
        if eff is not None:
            lines.append(f"sleep efficiency: **{eff * 100:.0f}%**")
            nums["last_efficiency"] = round(eff, 3)
        aw = _as_float(last.get("number_of_awakenings"))
        if aw is not None and aw > 0:
            lines.append(f"awakenings: {int(aw)}")
        if hours and len(hours) > 1:
            avg = sum(hours) / len(hours)
            nums["avg_hours"] = round(avg, 2)
            under = sum(1 for h in hours if h < 7.0)
            lines.append(
                f"{days}-day average: **{_fmt_dur(avg)}** "
                f"({under}/{len(hours)} nights under 7h)")
            if avg < 7.0:
                lines.append("that's below the 7–9h target — an earlier "
                             "bedtime would help more than anything else.")
            elif avg > 9.5:
                lines.append("that's a lot — consistently over 9.5h is "
                             "worth mentioning at your next checkup.")
        return Answer(guard_coaching("\n".join(lines) or
                                     "sleep data is thin — not enough to "
                                     "summarize yet."),
                      intent="sleep", numbers=nums, has_data=True)

    # -- activity --------------------------------------------------------

    def _answer_activity(self, question: str, days: int) -> Answer:
        end = date.today()
        start = end - timedelta(days=days)
        recs = self.source.metrics(start, end)
        workouts = self.source.workouts(start, end)
        if not recs and not workouts:
            return Answer(
                f"no activity data for the last {days} day(s) — nothing "
                "synced from your device in that window.",
                intent="activity", has_data=True)
        steps = [_as_float(r.get("step_count")) for r in recs]
        steps = [s for s in steps if s is not None]
        nums: dict[str, Any] = {}
        lines = []
        if steps:
            total, avg = sum(steps), sum(steps) / len(steps)
            nums["total_steps"] = int(total)
            nums["avg_steps"] = int(avg)
            nums["days"] = len(steps)
            lines.append(
                f"**{_fmt_num(total)}** steps over {len(steps)} day(s) "
                f"(avg **{_fmt_num(avg)}**/day)")
            if avg >= 10000:
                lines.append("that's a strong daily average — nice.")
            elif avg < 5000:
                lines.append("on the low side — a daily walk would move "
                             "this a lot.")
        dist = [_as_float(r.get("distance_walking_running_meters"))
                for r in recs]
        dist = [d for d in dist if d is not None]
        if dist:
            km = sum(dist) / 1000.0
            nums["distance_km"] = round(km, 1)
            lines.append(f"distance: **{km:.1f} km**")
        cals = [_as_float(r.get("active_energy_burned_kcal")) for r in recs]
        cals = [c for c in cals if c is not None]
        if cals:
            nums["active_kcal"] = int(sum(cals))
            lines.append(f"active energy: **{_fmt_num(sum(cals))} kcal**")
        if workouts:
            by_type: dict[str, int] = {}
            mins = 0.0
            for w in workouts:
                t = str(w.get("workout_type") or "workout").replace("_", " ")
                by_type[t] = by_type.get(t, 0) + 1
                d = _as_float(w.get("active_duration_sec"))
                if d:
                    mins += d / 60.0
            nums["workouts"] = len(workouts)
            nums["workout_minutes"] = round(mins, 1)
            types = ", ".join(f"{c}× {t}" for t, c in
                              sorted(by_type.items(), key=lambda kv: -kv[1]))
            lines.append(f"workouts: **{len(workouts)}** ({types})"
                         + (f", {mins:.0f} min total" if mins else ""))
        return Answer(guard_coaching("\n".join(lines)),
                      intent="activity", numbers=nums, has_data=True)

    # -- overview --------------------------------------------------------

    def _answer_overview(self) -> Answer:
        parts, nums = [], {}
        end = date.today()
        sessions = self.source.sleep_sessions(end - timedelta(days=2), end)
        sessions = sorted(sessions,
                          key=lambda s: str(s.get("end_datetime") or ""))
        if sessions:
            h = _sleep_hours(sessions[-1])
            if h is not None:
                parts.append(f"😴 last night: {_fmt_dur(h)}")
                nums["last_night_hours"] = round(h, 2)
        recs = self.source.metrics(end - timedelta(days=1), end)
        steps = [_as_float(r.get("step_count")) for r in recs]
        steps = [s for s in steps if s is not None]
        if steps:
            parts.append(f"👟 yesterday: {_fmt_num(steps[-1])} steps")
            nums["yesterday_steps"] = int(steps[-1])
        hrv = self._hrv_series(30)
        if len(hrv) >= 7:
            recent = sum(hrv[-7:]) / 7
            base = sum(hrv) / len(hrv)
            arrow = "↓" if recent < base * 0.95 else ("↑" if recent >
                                                      base * 1.05 else "→")
            parts.append(f"💓 HRV 7-day: {recent:.0f}ms {arrow} "
                         f"(30-day {base:.0f}ms)")
            nums["hrv_7d"] = round(recent, 1)
            nums["hrv_30d"] = round(base, 1)
        if not parts:
            return Answer(NO_DATA_MSG, intent="overview", has_data=False)
        parts.append("\nask \"how did I sleep this week?\", "
                     "\"am I recovering?\", or \"how active was I?\" "
                     "for detail.")
        return Answer(guard_coaching("\n".join(parts)),
                      intent="overview", numbers=nums, has_data=True)

    # -- readiness -------------------------------------------------------

    def _hrv_series(self, days: int) -> list[float]:
        end = date.today()
        recs = self.source.metrics(end - timedelta(days=days), end)
        out = []
        for r in recs:
            v = _as_float(r.get("heart_rate_variability_ms"))
            if v is not None and v > 0:
                out.append(v)
        return out

    def readiness(self, *, log: bool = True) -> Readiness:
        """Sleep + HRV trend + recent strain → high|moderate|low.

        Coaching only: the score describes recovery signals in your data,
        never a medical state.
        """
        if not self.source.has_data():
            return Readiness(level="unknown", score=0.0, has_data=False,
                             reasons=["no health data synced yet"])
        end = date.today()
        reasons, numbers = [], {}
        parts: list[float] = []
        weights: list[float] = []

        # 1. sleep (last night) — 50%
        sessions = self.source.sleep_sessions(end - timedelta(days=2), end)
        sessions = sorted(sessions,
                          key=lambda s: str(s.get("end_datetime") or ""))
        if sessions:
            h = _sleep_hours(sessions[-1])
            if h is not None:
                numbers["sleep_hours"] = round(h, 2)
                if 7.0 <= h <= 9.0:
                    s_score = 100.0
                elif h < 7.0:
                    s_score = max(0.0, 100.0 - (7.0 - h) * 30.0)
                else:
                    s_score = max(0.0, 100.0 - (h - 9.0) * 20.0)
                parts.append(s_score)
                weights.append(0.5)
                eff = _efficiency(sessions[-1])
                eff_txt = f", efficiency {eff * 100:.0f}%" \
                    if eff is not None else ""
                reasons.append(
                    f"sleep {_fmt_dur(h)} last night{eff_txt} "
                    f"(target 7–9h)")
                if eff is not None:
                    numbers["sleep_efficiency"] = round(eff, 3)

        # 2. HRV trend — 30%
        hrv = self._hrv_series(30)
        if len(hrv) >= 7:
            recent = sum(hrv[-7:]) / 7
            base = sum(hrv) / len(hrv)
            numbers["hrv_7d_ms"] = round(recent, 1)
            numbers["hrv_30d_ms"] = round(base, 1)
            ratio = recent / base if base else 1.0
            numbers["hrv_ratio"] = round(ratio, 3)
            if ratio >= 1.0:
                h_score = 100.0
                trend = "steady or up"
            elif ratio >= 0.95:
                h_score = 85.0
                trend = "steady"
            elif ratio >= 0.90:
                h_score = 65.0
                trend = "dipping"
            elif ratio >= 0.85:
                h_score = 45.0
                trend = "down"
            else:
                h_score = 25.0
                trend = "down a lot"
            parts.append(h_score)
            weights.append(0.3)
            reasons.append(
                f"HRV 7-day avg {recent:.0f}ms vs 30-day {base:.0f}ms "
                f"— {trend}")

        # 3. recent strain — 20%
        workouts = self.source.workouts(end - timedelta(days=2), end)
        n_w = len(workouts)
        numbers["workouts_48h"] = n_w
        strain = max(0.0, 100.0 - min(40.0, 20.0 * n_w))
        parts.append(strain)
        weights.append(0.2)
        if n_w:
            reasons.append(f"{n_w} workout(s) in the last 48h")
        else:
            reasons.append("no workouts in the last 48h")

        if not parts:
            return Readiness(level="low", score=0.0, has_data=False)
        total_w = sum(weights)
        score = sum(p * w for p, w in zip(parts, weights)) / total_w
        level = "high" if score >= 70 else ("moderate" if score >= 45
                                           else "low")
        suggestion = {
            "high": "green light — good day to push if you want to.",
            "moderate": "decent shape — keep it moderate if you're "
                        "training hard today.",
            "low": "recovery looks low — consider a light day: easy "
                   "walk, stretch, early night.",
        }[level]
        r = Readiness(level=level, score=round(score, 1), reasons=reasons,
                      numbers=numbers, suggestion=suggestion, has_data=True)
        if log:
            self._log_insight(
                f"readiness check → {level} ({score:.0f}/100): "
                + "; ".join(reasons))
        return r

    # -- weekly recap ----------------------------------------------------

    def weekly_recap(self, *, log: bool = True) -> Answer:
        """The week's movement / sleep / recovery in one message."""
        if not self.source.has_data():
            return Answer(NO_DATA_MSG, intent="activity", has_data=False)
        end = date.today()
        start = end - timedelta(days=7)
        recs = self.source.metrics(start, end)
        sessions = self.source.sleep_sessions(start, end)
        workouts = self.source.workouts(start, end)
        lines = ["📊 **this week**"]
        nums: dict[str, Any] = {}

        steps = [s for r in recs
                 if (s := _as_float(r.get("step_count"))) is not None]
        if steps:
            avg = sum(steps) / len(steps)
            nums["avg_steps"] = int(avg)
            nums["total_steps"] = int(sum(steps))
            lines.append(f"👟 avg {_fmt_num(avg)} steps/day "
                         f"({_fmt_num(sum(steps))} total)")

        hours = [h for s in sessions
                 if (h := _sleep_hours(s)) is not None]
        if hours:
            avg_h = sum(hours) / len(hours)
            under = sum(1 for h in hours if h < 7.0)
            nums["avg_sleep_hours"] = round(avg_h, 2)
            nums["nights_tracked"] = len(hours)
            lines.append(f"😴 avg {_fmt_dur(avg_h)}/night over "
                         f"{len(hours)} nights ({under} under 7h)")

        if workouts:
            by_type: dict[str, int] = {}
            for w in workouts:
                t = str(w.get("workout_type") or "workout").replace("_", " ")
                by_type[t] = by_type.get(t, 0) + 1
            nums["workouts"] = len(workouts)
            top = sorted(by_type.items(), key=lambda kv: -kv[1])[:3]
            lines.append("🏋️ " + f"{len(workouts)} workouts: "
                         + ", ".join(f"{c}× {t}" for t, c in top))

        hrv = self._hrv_series(14)
        if len(hrv) >= 7:
            first = sum(hrv[:7]) / 7
            last_w = sum(hrv[-7:]) / 7
            arrow = "↓" if last_w < first * 0.95 else ("↑" if last_w >
                                                       first * 1.05 else "→")
            nums["hrv_trend"] = arrow
            lines.append(f"💓 HRV trend {arrow} "
                         f"({first:.0f} → {last_w:.0f}ms)")

        # one coaching line from readiness
        r = self.readiness(log=False)
        if r.has_data:
            lines.append(f"\nreadiness now: **{r.level}** "
                         f"({r.score:.0f}/100) — {r.suggestion}")

        text = guard_coaching("\n".join(lines))
        if log:
            self._log_insight("weekly recap: " + "; ".join(
                l.strip("📊👟😴🏋️💓 *") for l in lines[1:4] if l))
        return Answer(text, intent="activity", numbers=nums, has_data=True)


# ── morning-briefing hook ──────────────────────────────────────────────────
# When readiness is low the briefing adjusts ("recovery looks low —
# consider a light day"). Additive: moderate/high → no section.

class ReadinessBriefingProvider:
    """Briefing provider: low-readiness adjustment. Never raises."""

    name = "readiness"
    title = "💪 Recovery"
    priority = 15  # right after overnight alerts; this is timely
    source = "health"

    def __init__(self, coach: "HealthCoach | None" = None) -> None:
        # Injectable for tests; the composer constructs with no args.
        self._coach = coach

    def collect(self, ctx: Any, since: float) -> Any | None:
        try:
            return self._collect(ctx, since)
        except Exception as exc:  # noqa: BLE001 — readiness is optional
            _log.debug("readiness provider skipped: %s", exc)
            return None

    def _collect(self, ctx: Any, since: float) -> Any | None:
        from ..agents.morning_briefing import BriefingSection
        coach = self._coach or HealthCoach()
        if not coach.source.has_data():
            return None
        r = coach.readiness(log=False)
        if not r.has_data or r.level != "low":
            return None
        lines = [f"• 🔴 recovery looks **low** ({r.score:.0f}/100) — "
                 "consider a light day"]
        for reason in r.reasons[:3]:
            lines.append(f"  — {reason}")
        items = [{"id": "readiness-low", "title": "low recovery",
                  "body": "; ".join(r.reasons)}]
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source=self.source, items=items, lines=lines)
