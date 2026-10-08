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
    """Correlate mood/sleep/activity from the timeline. Never raises.

    Returns strong patterns only (strength >= STRONG_MIN). Language is
    correlation-only; no pattern on fewer than MIN_EVIDENCE points.
    """
    try:
        since = time.time() - days * 86400
        events = timeline.timeline(since=since, limit=500)
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
    out.sort(key=lambda p: -p.strength)
    return out


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


# ── morning-briefing provider ─────────────────────────────────────────────

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
