"""Mood-pattern detection from memory. Wellness companionship — never therapy.

Correlations from the owner's own health timeline, surfaced proactively:
"3 of your last 4 low-mood days followed nights under 6h sleep."

Hard boundaries (test-enforced):
- NO diagnostic language — never "depression", "disorder", "clinical".
- Correlation language only: "followed", "tend to", "often come after".
  Never "causes", "because", "leads to".
- Crisis resources ship with EVERY low-mood output.
- No pattern claims on fewer than 3 data points.
- Owner-scoped: HealthTimeline raises in community contexts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .previsit import CRISIS_RESOURCES

_log = get_logger(__name__)

__all__ = [
    "Pattern",
    "detect_patterns",
    "format_patterns",
    "PatternsBriefingProvider",
    "BANNED_PHRASES",
    "MoodStats",
    "mood_stats",
    "mood_map",
    "best_worst_days",
    "detect_lag_patterns",
    "FACTOR_DEFS",
]

#: Diagnostic / medicalizing language that must never appear in output.
#: Extends the timeline module's list with pattern-specific terms.
BANNED_PHRASES = (
    "depression", "depressive", "depressed disorder",
    "disorder", "bipolar", "clinical", "diagnos",
    "anxiety disorder", "patholog", "mental illness",
    "sounds like", "this could be", "you might have",
    "you should take", "i recommend you take", "try taking", "prescrib",
)

#: Causation language that must never appear — correlation only.
CAUSATION_PHRASES = (
    "causes", "caused by", "because of", "leads to", "led to",
    "the reason", "makes you", "making you",
)

LOW_MOOD = 2            # severity <= 2 counts as a low-mood day
GOOD_MOOD = 4           # severity >= 4 counts as a good-mood day
SHORT_SLEEP_H = 6.0     # under 6h counts as a short night
MIN_EVIDENCE = 3        # never claim a pattern on fewer data points
STRONG_MIN = 0.6        # strength threshold for proactive surfacing

_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", re.I)
_ACTIVITY_RE = re.compile(
    r"\b(workout|exercise|gym|ran|running|walk|walked|yoga|swim|"
    r"football|jog|cycle|cycling|training)\b", re.I)
_CAFFEINE_RE = re.compile(r"\b(coffee|caffeine|espresso|energy drink)\b", re.I)

# ── factor library (Bearable pattern) ────────────────────────────────────
# Each factor: (kind, regex, description template). Mined from timeline
# text across ALL event types — the broader the factors, the subtler
# the dots it can connect.

FACTOR_DEFS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("sleep-short", re.compile(
        r"\b(slept|sleep)\b.{0,30}?\b([1-5](?:\.\d+)?)\s*h\b", re.I),
     "short night"),
    ("activity", _ACTIVITY_RE, "active day"),
    ("caffeine", _CAFFEINE_RE, "caffeine"),
    ("alcohol", re.compile(r"\b(beer|wine|alcohol|drank|drinking|"
                           r"cocktail|stout)\b", re.I), "alcohol"),
    ("medication", re.compile(
        r"\b(took|taking|started)\b.{0,40}?\b(mg|pill|tablet|dose|"
        r"supplement|vitamin)\b", re.I), "medication/supplement"),
    ("social", re.compile(
        r"\b(friends|family|party|hangout|date night|visited|"
        r"called mom|called dad)\b", re.I), "social time"),
    ("screen-late", re.compile(
        r"\b(phone|scrolling|netflix|youtube|gaming|tv)\b.{0,30}?\b("
        r"late|midnight|2am|3am)\b|\b(2am|3am|midnight)\b.{0,30}?\b("
        r"phone|scrolling|screen)\b", re.I), "late screens"),
    ("stress", re.compile(
        r"\b(deadline|overtime|argument|fight|bills|exam|interview|"
        r"traffic|stressful)\b", re.I), "stressful day"),
    ("good-sleep", re.compile(
        r"\b(slept|sleep)\b.{0,30}?\b([78](?:\.\d+)?)\s*h\b", re.I),
     "solid night"),
)


def _factor_days(events: list[Any]) -> dict[str, set[str]]:
    """factor kind → set of YYYY-MM-DD days it appeared."""
    out: dict[str, set[str]] = {}
    for e in events:
        text = getattr(e, "text", "") or ""
        day = _day(e.ts)
        for kind, rx, _desc in FACTOR_DEFS:
            if rx.search(text):
                # sleep-short needs the hours check
                if kind == "sleep-short":
                    m = rx.search(text)
                    try:
                        if m and float(m.group(2)) >= SHORT_SLEEP_H:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                if kind == "good-sleep":
                    m = rx.search(text)
                    try:
                        if not m or float(m.group(2)) < 7.0:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                out.setdefault(kind, set()).add(day)
    return out


def _factor_mood_pattern(kind: str, desc: str, factor_days: set[str],
                         moods: list[Any], want: str
                         ) -> Pattern | None:
    """Correlate a factor with low or good mood days."""
    if want == "low":
        days = [m for m in moods
                if (m.severity or 0) > 0 and m.severity <= LOW_MOOD]
        involves_low = True
    else:
        days = [m for m in moods if (m.severity or 0) >= GOOD_MOOD]
        involves_low = False
    if len(days) < MIN_EVIDENCE:
        return None
    matched = 0
    examples: list[str] = []
    for m in days:
        if _day(m.ts) in factor_days or _day(m.ts - 86400) in factor_days:
            matched += 1
            examples.append(_day(m.ts))
    if matched < MIN_EVIDENCE:
        return None
    strength = matched / len(days)
    mood_word = "low-mood" if want == "low" else "good-mood"
    direction = "followed" if want == "low" else "came after"
    desc_text = (f"{matched} of your last {len(days)} {mood_word} days "
                 f"{direction} days with {desc}.")
    _assert_safe(desc_text)
    return Pattern(description=desc_text, strength=round(strength, 2),
                   examples=examples, involves_low_mood=involves_low,
                   kind=f"factor-{kind}-{want}")


def detect_lag_patterns(timeline: Any, *, days: int = 30,
                        max_lag: int = 2) -> list[Pattern]:
    """Factor↔mood correlations at 0..max_lag day lags (Bearable pattern).

    "Good-mood days tend to come a day after active days." Never raises.
    """
    try:
        since = time.time() - days * 86400
        events = timeline.timeline(since=since, limit=1000)
    except Exception:  # noqa: BLE001
        _log.debug("lag pattern read failed", exc_info=True)
        return []
    moods = [e for e in events if e.event_type == "mood"]
    factors = _factor_days(events)
    out: list[Pattern] = []
    for kind, _rx, desc in FACTOR_DEFS:
        fdays = factors.get(kind, set())
        if not fdays:
            continue
        for lag in range(max_lag + 1):
            for want in ("low", "good"):
                if want == "low":
                    days_m = [m for m in moods
                              if (m.severity or 0) > 0
                              and m.severity <= LOW_MOOD]
                else:
                    days_m = [m for m in moods
                              if (m.severity or 0) >= GOOD_MOOD]
                if len(days_m) < MIN_EVIDENCE:
                    continue
                matched = sum(
                    1 for m in days_m
                    if _day(m.ts - lag * 86400) in fdays)
                if matched < MIN_EVIDENCE:
                    continue
                strength = matched / len(days_m)
                if strength < STRONG_MIN:
                    continue
                lag_txt = ("the same day" if lag == 0
                           else f"{lag} day{'s' if lag > 1 else ''} "
                                f"after")
                mood_word = "low-mood" if want == "low" else "good-mood"
                desc_text = (
                    f"{matched} of your last {len(days_m)} {mood_word} "
                    f"days came {lag_txt} days with {desc}.")
                _assert_safe(desc_text)
                out.append(Pattern(
                    description=desc_text, strength=round(strength, 2),
                    examples=[_day(m.ts) for m in days_m[:5]],
                    involves_low_mood=(want == "low"),
                    kind=f"lag-{kind}-{want}-{lag}"))
    # de-dupe by description, keep strongest
    seen: dict[str, Pattern] = {}
    for p in sorted(out, key=lambda p: -p.strength):
        seen.setdefault(p.description, p)
    return sorted(seen.values(), key=lambda p: -p.strength)[:6]


@dataclass
class Pattern:
    """One detected correlation. Strength is matched/total, 0-1."""
    description: str
    strength: float
    examples: list[str] = field(default_factory=list)  # date strings
    involves_low_mood: bool = False
    kind: str = ""  # "sleep-mood" | "activity-mood" | "caffeine-sleep"


def _assert_safe(text: str) -> None:
    lowered = text.lower()
    for phrase in BANNED_PHRASES + CAUSATION_PHRASES:
        if phrase in lowered:
            raise AssertionError(
                f"banned phrase in pattern output: {phrase!r}")


def _sleep_hours(ev: Any) -> float | None:
    """Hours slept, parsed from a sleep event's text. None if unparseable."""
    try:
        m = _HOURS_RE.search(getattr(ev, "text", "") or "")
        if m:
            return float(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return None


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _prior_night_sleep(sleeps: list[Any], mood_ts: float) -> Any | None:
    """The sleep event most likely covering the night before a mood log."""
    best = None
    for s in sleeps:
        # sleep logged within the ~20h before the mood entry
        if 0 < mood_ts - s.ts <= 20 * 3600:
            if best is None or s.ts > best.ts:
                best = s
    return best


def _sleep_mood_pattern(moods: list[Any],
                        sleeps: list[Any]) -> Pattern | None:
    """Low-mood days that followed short nights."""
    low = [m for m in moods
           if (m.severity or 0) > 0 and m.severity <= LOW_MOOD]
    if len(low) < MIN_EVIDENCE:
        return None
    followed = 0
    examples: list[str] = []
    for m in low:
        night = _prior_night_sleep(sleeps, m.ts)
        hours = _sleep_hours(night) if night else None
        if hours is not None and hours < SHORT_SLEEP_H:
            followed += 1
            examples.append(_day(m.ts))
    if followed < MIN_EVIDENCE:
        return None
    strength = followed / len(low)
    desc = (
        f"{followed} of your last {len(low)} low-mood days followed "
        f"nights under {SHORT_SLEEP_H:g}h sleep.")
    _assert_safe(desc)
    return Pattern(description=desc, strength=round(strength, 2),
                   examples=examples, involves_low_mood=True,
                   kind="sleep-mood")


def _activity_mood_pattern(moods: list[Any],
                           events: list[Any]) -> Pattern | None:
    """Good-mood days that came after active days."""
    good = [m for m in moods
            if (m.severity or 0) >= GOOD_MOOD]
    if len(good) < MIN_EVIDENCE:
        return None
    active_days = {_day(e.ts) for e in events
                   if _ACTIVITY_RE.search(getattr(e, "text", "") or "")}
    followed = 0
    examples: list[str] = []
    for m in good:
        # active the day before the good-mood log
        prev_day = _day(m.ts - 86400)
        if prev_day in active_days:
            followed += 1
            examples.append(_day(m.ts))
    if followed < MIN_EVIDENCE:
        return None
    strength = followed / len(good)
    desc = (
        f"{followed} of your last {len(good)} good-mood days came after "
        f"days you were active.")
    _assert_safe(desc)
    return Pattern(description=desc, strength=round(strength, 2),
                   examples=examples, involves_low_mood=False,
                   kind="activity-mood")


def _caffeine_sleep_pattern(sleeps: list[Any],
                            events: list[Any]) -> Pattern | None:
    """Short nights that followed late caffeine mentions."""
    short = [s for s in sleeps
             if (_sleep_hours(s) or 99) < SHORT_SLEEP_H]
    if len(short) < MIN_EVIDENCE:
        return None
    caffeine_days = {_day(e.ts) for e in events
                     if _CAFFEINE_RE.search(getattr(e, "text", "") or "")}
    followed = 0
    examples: list[str] = []
    for s in short:
        if _day(s.ts) in caffeine_days:
            followed += 1
            examples.append(_day(s.ts))
    if followed < MIN_EVIDENCE:
        return None
    strength = followed / len(short)
    desc = (
        f"{followed} of your last {len(short)} short nights came on days "
        f"you mentioned caffeine.")
    _assert_safe(desc)
    return Pattern(description=desc, strength=round(strength, 2),
                   examples=examples, involves_low_mood=False,
                   kind="caffeine-sleep")


def detect_patterns(timeline: Any, *, days: int = 30) -> list[Pattern]:
    """Correlate mood/sleep/activity/factors from the timeline.

    Never raises. Returns strong patterns only (strength >=
    STRONG_MIN). Language is correlation-only; no pattern on fewer
    than MIN_EVIDENCE points.
    """
    try:
        since = time.time() - days * 86400
        events = timeline.timeline(since=since, limit=1000)
    except Exception:  # noqa: BLE001
        _log.debug("pattern detection read failed", exc_info=True)
        return []
    moods = [e for e in events if e.event_type == "mood"]
    sleeps = [e for e in events if e.event_type == "sleep"]
    out: list[Pattern] = []
    for fn in (_sleep_mood_pattern(moods, sleeps),
               _activity_mood_pattern(moods, events),
               _caffeine_sleep_pattern(sleeps, events)):
        if fn is not None and fn.strength >= STRONG_MIN:
            out.append(fn)
    # broader factor library (Bearable pattern)
    factors = _factor_days(events)
    for kind, _rx, desc in FACTOR_DEFS:
        fdays = factors.get(kind, set())
        if not fdays:
            continue
        for want in ("low", "good"):
            # skip the ones the classic detectors already cover
            if (kind, want) in {("sleep-short", "low"),
                                ("activity", "good")}:
                continue
            p = _factor_mood_pattern(kind, desc, fdays, moods, want)
            if p is not None and p.strength >= STRONG_MIN:
                out.append(p)
    # de-dupe by description, strongest first
    seen: dict[str, Pattern] = {}
    for p in sorted(out, key=lambda p: -p.strength):
        seen.setdefault(p.description, p)
    return sorted(seen.values(), key=lambda p: -p.strength)


def format_patterns(patterns: list[Pattern]) -> str:
    """Human-readable patterns. Crisis resources included whenever low
    mood is involved. Never raises."""
    try:
        if not patterns:
            return ("no strong patterns in your recent data yet — keep "
                    "logging mood and sleep and I'll watch for them 🩺")
        lines = ["🧠 patterns from your recent logs:"]
        low_mood_involved = False
        for p in patterns:
            lines.append(f"• {p.description}")
            if p.involves_low_mood:
                low_mood_involved = True
        lines.append("")
        lines.append("correlation, not causation — just things that tend "
                     "to show up together.")
        if low_mood_involved:
            lines.append("")
            lines.append("💚 if the low days are weighing on you, support "
                         "is always here:")
            lines.extend(CRISIS_RESOURCES)
        text = "\n".join(lines)
        _assert_safe(text)
        return text
    except Exception:  # noqa: BLE001
        _log.debug("format_patterns failed", exc_info=True)
        return "couldn't format patterns right now."


# ── Daylio-style views: stats, mood map, best/worst days ─────────────────


@dataclass
class MoodStats:
    """Weekly/monthly aggregates. Pure record, never raises."""
    days: int
    entries: int = 0
    avg_mood: float | None = None
    low_days: int = 0
    good_days: int = 0
    best_day: str = ""
    worst_day: str = ""

    def format(self) -> str:
        try:
            if not self.entries:
                return (f"no mood entries in the last {self.days} days — "
                        f"log your mood and I'll chart it. 🙂")
            lines = [f"🙂 **mood — last {self.days} days** "
                     f"({self.entries} entries):"]
            if self.avg_mood is not None:
                lines.append(f"• average: {self.avg_mood:.1f}/5")
            lines.append(f"• low days (≤2): {self.low_days} · "
                         f"good days (≥4): {self.good_days}")
            if self.best_day:
                lines.append(f"• best: {self.best_day} 🌟")
            if self.worst_day:
                lines.append(f"• toughest: {self.worst_day} 💚")
            text = "\n".join(lines)
            _assert_safe(text)
            return text
        except Exception:  # noqa: BLE001
            return "couldn't build mood stats."


def mood_stats(timeline: Any, *, days: int = 30) -> MoodStats:
    """Aggregate mood entries. Never raises."""
    stats = MoodStats(days=days)
    try:
        since = time.time() - days * 86400
        moods = [e for e in timeline.timeline(since=since, limit=1000)
                 if e.event_type == "mood" and e.severity]
        if not moods:
            return stats
        stats.entries = len(moods)
        vals = [m.severity for m in moods]
        stats.avg_mood = round(sum(vals) / len(vals), 1)
        stats.low_days = sum(1 for v in vals if v <= LOW_MOOD)
        stats.good_days = sum(1 for v in vals if v >= GOOD_MOOD)
        by_day: dict[str, list[int]] = {}
        for m in moods:
            by_day.setdefault(_day(m.ts), []).append(m.severity)
        day_avg = {d: sum(v) / len(v) for d, v in by_day.items()}
        stats.best_day = max(day_avg, key=lambda d: day_avg[d])
        stats.worst_day = min(day_avg, key=lambda d: day_avg[d])
        return stats
    except Exception:  # noqa: BLE001
        _log.debug("mood_stats failed", exc_info=True)
        return stats


_MOOD_BLOCKS = {1: "🟥", 2: "🟧", 3: "🟨", 4: "🟩", 5: "💚"}


def mood_map(timeline: Any, *, weeks: int = 12) -> str:
    """Year-in-pixels style mood calendar (ASCII/emoji weeks). Never raises."""
    try:
        since = time.time() - weeks * 7 * 86400
        moods = [e for e in timeline.timeline(since=since, limit=2000)
                 if e.event_type == "mood" and e.severity]
        if not moods:
            return "no mood entries yet — your pixels go here. 🙂"
        by_day: dict[str, list[int]] = {}
        for m in moods:
            by_day.setdefault(_day(m.ts), []).append(m.severity)
        # build week rows, Monday-first
        today = time.time()
        monday = today - (time.localtime(today).tm_wday * 86400)
        monday -= (weeks - 1) * 7 * 86400
        rows = []
        for w in range(weeks):
            cells = []
            for d in range(7):
                ts = monday + (w * 7 + d) * 86400
                if ts > today + 86400:
                    cells.append("⬜")
                    continue
                key = time.strftime("%Y-%m-%d", time.localtime(ts))
                vals = by_day.get(key)
                if not vals:
                    cells.append("⬜")
                else:
                    avg = round(sum(vals) / len(vals))
                    cells.append(_MOOD_BLOCKS.get(
                        max(1, min(5, avg)), "⬜"))
            rows.append("".join(cells))
        legend = "🟥1 🟧2 🟨3 🟩4 💚5"
        text = ("🗓️ **mood map** — last "
                f"{weeks} weeks:\n" + "\n".join(rows)
                + f"\n_{legend}_")
        _assert_safe(text)
        return text
    except Exception as exc:  # noqa: BLE001
        _log.debug("mood_map failed: %s", exc)
        return "couldn't build the mood map."


def best_worst_days(timeline: Any, *, days: int = 30,
                    limit: int = 3) -> str:
    """The Daylio retention view: your best and toughest days, with what
    you logged around them. Never raises."""
    try:
        since = time.time() - days * 86400
        events = timeline.timeline(since=since, limit=1000)
        moods = [e for e in events
                 if e.event_type == "mood" and e.severity]
        if len(moods) < 2:
            return "log a few more mood entries and I'll show your best " \
                "and toughest days. 🙂"
        by_day: dict[str, list[Any]] = {}
        for m in moods:
            by_day.setdefault(_day(m.ts), []).append(m)
        day_avg = {d: sum(m.severity for m in v) / len(v)
                   for d, v in by_day.items()}
        best = sorted(day_avg, key=lambda d: -day_avg[d])[:limit]
        worst = sorted(day_avg, key=lambda d: day_avg[d])[:limit]
        others = [e for e in events if e.event_type != "mood"]

        def _context(day: str) -> str:
            bits = []
            for e in others:
                if _day(e.ts) == day and e.event_type in (
                        "sleep", "note", "activity"):
                    bits.append(e.text[:60])
                if len(bits) >= 2:
                    break
            return (" — " + "; ".join(bits)) if bits else ""
        lines = ["🌟 **your best days** recently:"]
        for d in best:
            lines.append(f"• {d} ({day_avg[d]:.1f}/5){_context(d)}")
        lines.append("")
        lines.append("🌧️ **toughest days** recently:")
        for d in worst:
            lines.append(f"• {d} ({day_avg[d]:.1f}/5){_context(d)}")
        lines.append("")
        lines.append("correlation, not causation — just what the days "
                     "looked like.")
        text = "\n".join(lines)
        _assert_safe(text)
        return text
    except Exception as exc:  # noqa: BLE001
        _log.debug("best_worst_days failed: %s", exc)
        return "couldn't build best/worst days."

class PatternsBriefingProvider:
    """Surfaces strong health patterns in the morning briefing.

    Follows the _Provider protocol structurally (collect(ctx, since)).
    Only strong patterns (strength >= STRONG_MIN); crisis resources ride
    along whenever low mood is involved. Never raises.
    """
    name = "health-patterns"
    title = "Health patterns"
    priority = 90  # after urgent content, before trivia

    def collect(self, ctx: Any, since: float) -> Any | None:
        try:
            from ..agents.morning_briefing import BriefingSection
        except Exception:  # noqa: BLE001
            return None
        try:
            from .timeline import HealthTimeline
            tl = HealthTimeline()
        except Exception:  # noqa: BLE001
            return None
        patterns = detect_patterns(tl, days=30)
        if not patterns:
            return None
        lines = [p.description for p in patterns]
        if any(p.involves_low_mood for p in patterns):
            lines.append("")
            lines.append("💚 support, always here:")
            lines.extend(CRISIS_RESOURCES)
        return BriefingSection(name=self.name, title=self.title,
                               priority=self.priority, source="health",
                               lines=lines)
