"""Proactive suggestions engine.

Analyzes user patterns and context to suggest actions before they ask.
This is what makes the bot feel intelligent and Muse-like.

Features:
- Pattern recognition (e.g., "You usually order coffee on Mondays")
- Context-aware suggestions (e.g., "Your flight is tomorrow, check in?")
- Reminder optimization
- Habit tracking
- Time-based suggestions
- Event-driven suggestions

Usage:
    proactive = ProactiveEngine(memory, calendar, email, shopping)
    
    # Get suggestions for current context
    suggestions = await proactive.get_suggestions(user_id="user123")
    
    for suggestion in suggestions:
        print(f"💡 {suggestion.text}")
        print(f"   Confidence: {suggestion.confidence}")
        print(f"   Action: {suggestion.action}")
    
    # Record an action for pattern learning
    await proactive.record_action(
        user_id="user123",
        action="order_coffee",
        context={"day": "monday", "time": "09:00"},
    )
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..core.logging_setup import get_logger
from ..memory.manager import MemoryManager

__all__ = [
    "ProactiveEngine",
    "Suggestion",
    "SuggestionType",
    "Pattern",
    "UserHabit",
]

_log = get_logger(__name__)


@dataclass
class Suggestion:
    """A proactive suggestion."""
    
    suggestion_id: str
    text: str
    suggestion_type: str  # pattern, context, reminder, habit
    action: str  # Action to execute if accepted
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1
    priority: int = 0  # Higher = more important
    expires_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "suggestion_id": self.suggestion_id,
            "text": self.text,
            "type": self.suggestion_type,
            "action": self.action,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "priority": self.priority,
        }
    
    def to_message(self) -> str:
        """Format as user-facing message."""
        return f"💡 {self.text}"


@dataclass
class Pattern:
    """A recognized user pattern."""
    
    pattern_id: str
    user_id: str
    action: str
    frequency: Counter = field(default_factory=Counter)
    context: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    occurrences: int = 0
    last_seen: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "pattern_id": self.pattern_id,
            "user_id": self.user_id,
            "action": self.action,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
            "context": {k: dict(v) for k, v in self.context.items()},
        }


@dataclass
class UserHabit:
    """A tracked user habit."""
    
    habit_id: str
    user_id: str
    name: str
    description: str
    frequency: str  # daily, weekly, monthly
    last_completed: Optional[float] = None
    streak: int = 0
    total_completions: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "habit_id": self.habit_id,
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "last_completed": self.last_completed,
            "streak": self.streak,
            "total_completions": self.total_completions,
        }


class ProactiveEngine:
    """Proactive suggestion engine.
    
    Analyzes user behavior and context to suggest actions proactively.
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        calendar: Any = None,
        email: Any = None,
        shopping: Any = None,
    ) -> None:
        self.memory = memory
        self.calendar = calendar
        self.email = email
        self.shopping = shopping
        
        # Pattern storage
        self._patterns: dict[str, list[Pattern]] = defaultdict(list)  # user_id -> patterns
        self._habits: dict[str, list[UserHabit]] = defaultdict(list)  # user_id -> habits
        
        # Suggestion generators
        self._suggestion_generators = [
            self._generate_pattern_suggestions,
            self._generate_calendar_suggestions,
            self._generate_time_suggestions,
            self._generate_habit_suggestions,
            self._generate_context_suggestions,
        ]
        
        _log.info("Proactive engine initialized")
    
    async def get_suggestions(
        self,
        user_id: str,
        *,
        limit: int = 5,
        min_confidence: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> list[Suggestion]:
        """Get proactive suggestions for a user.
        
        Args:
            user_id: User ID
            limit: Maximum suggestions to return
            min_confidence: Minimum confidence threshold
            context: Current context (time, location, etc.)
            
        Returns:
            List of Suggestion objects sorted by priority
        """
        context = context or {}
        context.setdefault("timestamp", time.time())
        context.setdefault("hour", datetime.now().hour)
        context.setdefault("day_of_week", datetime.now().strftime("%A").lower())
        
        all_suggestions: list[Suggestion] = []
        
        # Run all suggestion generators
        for generator in self._suggestion_generators:
            try:
                suggestions = await generator(user_id, context)
                all_suggestions.extend(suggestions)
            except Exception as e:
                _log.error(f"Suggestion generator failed: {e}")
        
        # Filter and sort
        filtered = [
            s for s in all_suggestions
            if s.confidence >= min_confidence
        ]
        
        # Sort by priority, then confidence
        filtered.sort(key=lambda s: (-s.priority, -s.confidence))
        
        return filtered[:limit]
    
    async def record_action(
        self,
        user_id: str,
        action: str,
        *,
        context: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a user action for pattern learning.
        
        Args:
            user_id: User ID
            action: Action name
            context: Context when action occurred
            parameters: Action parameters
        """
        context = context or {}
        parameters = parameters or {}
        
        # Add temporal context
        now = datetime.now()
        context.setdefault("hour", now.hour)
        context.setdefault("day_of_week", now.strftime("%A").lower())
        context.setdefault("month", now.month)
        
        # Find or create pattern
        pattern = self._find_pattern(user_id, action)
        if not pattern:
            from ..core.ids import new_id
            pattern = Pattern(
                pattern_id=new_id("pattern"),
                user_id=user_id,
                action=action,
            )
            self._patterns[user_id].append(pattern)
        
        # Update pattern
        pattern.occurrences += 1
        pattern.last_seen = time.time()
        
        # Update context frequency
        for key, value in context.items():
            pattern.context[key][str(value)] += 1
        
        # Recalculate confidence
        pattern.confidence = min(1.0, pattern.occurrences / 10.0)
        
        _log.debug(f"Recorded action: {action} (occurrences: {pattern.occurrences})")
    
    async def accept_suggestion(self, suggestion: Suggestion) -> None:
        """Record that a suggestion was accepted.
        
        Args:
            suggestion: The accepted suggestion
        """
        # This could be used to improve future suggestions
        _log.info(f"Suggestion accepted: {suggestion.text}")
    
    async def reject_suggestion(self, suggestion: Suggestion) -> None:
        """Record that a suggestion was rejected.
        
        Args:
            suggestion: The rejected suggestion
        """
        # Lower confidence for similar suggestions in the future
        _log.info(f"Suggestion rejected: {suggestion.text}")
    
    # ── Suggestion Generators ────────────────────────────────────────────────
    
    async def _generate_pattern_suggestions(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate suggestions based on user patterns."""
        suggestions = []
        patterns = self._patterns.get(user_id, [])
        
        for pattern in patterns:
            if pattern.confidence < 0.5:
                continue
            
            # Check if current context matches pattern
            match_score = self._context_matches(pattern, context)
            
            if match_score > 0.6:
                from ..core.ids import new_id
                
                suggestions.append(Suggestion(
                    suggestion_id=new_id("suggestion"),
                    text=self._format_pattern_suggestion(pattern, context),
                    suggestion_type="pattern",
                    action=pattern.action,
                    parameters=self._extract_pattern_params(pattern, context),
                    confidence=pattern.confidence * match_score,
                    priority=2,
                ))
        
        return suggestions
    
    async def _generate_calendar_suggestions(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate suggestions based on calendar events."""
        suggestions = []
        
        if not self.calendar:
            return suggestions
        
        try:
            # Check for upcoming events
            events = await self.calendar.list_events(
                account=f"{user_id}@gmail.com",
                days_ahead=1,
            )
            
            from ..core.ids import new_id
            
            for event in events:
                # Suggest preparation for meetings
                if "meeting" in event.title.lower() or "call" in event.title.lower():
                    suggestions.append(Suggestion(
                        suggestion_id=new_id("suggestion"),
                        text=f"You have '{event.title}' coming up. Want me to prepare notes?",
                        suggestion_type="context",
                        action="prepare_meeting_notes",
                        parameters={"event_id": event.event_id, "title": event.title},
                        confidence=0.8,
                        priority=3,
                    ))
                
                # Suggest travel time for events with location
                if event.location:
                    suggestions.append(Suggestion(
                        suggestion_id=new_id("suggestion"),
                        text=f"Event at {event.location} - want me to check travel time?",
                        suggestion_type="context",
                        action="check_travel_time",
                        parameters={"event_id": event.event_id, "location": event.location},
                        confidence=0.7,
                        priority=2,
                    ))
        except Exception as e:
            _log.error(f"Calendar suggestions failed: {e}")
        
        return suggestions
    
    async def _generate_time_suggestions(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate time-based suggestions."""
        suggestions = []
        hour = context.get("hour", 12)
        
        from ..core.ids import new_id
        
        # Morning suggestions
        if 7 <= hour <= 9:
            suggestions.append(Suggestion(
                suggestion_id=new_id("suggestion"),
                text="Good morning! Want me to summarize your emails and today's schedule?",
                suggestion_type="context",
                action="morning_briefing",
                confidence=0.6,
                priority=1,
            ))
        
        # Lunch time
        if 11 <= hour <= 13:
            suggestions.append(Suggestion(
                suggestion_id=new_id("suggestion"),
                text="Lunchtime! Want me to order food or find nearby restaurants?",
                suggestion_type="context",
                action="lunch_suggestion",
                confidence=0.5,
                priority=1,
            ))
        
        # Evening wind-down
        if 20 <= hour <= 22:
            suggestions.append(Suggestion(
                suggestion_id=new_id("suggestion"),
                text="Evening! Want me to summarize what you accomplished today?",
                suggestion_type="context",
                action="daily_summary",
                confidence=0.5,
                priority=1,
            ))
        
        return suggestions
    
    async def _generate_habit_suggestions(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate suggestions based on tracked habits."""
        suggestions = []
        habits = self._habits.get(user_id, [])
        
        from ..core.ids import new_id
        
        for habit in habits:
            # Check if habit is due
            if self._is_habit_due(habit):
                suggestions.append(Suggestion(
                    suggestion_id=new_id("suggestion"),
                    text=f"Time for your habit: {habit.name}! (Streak: {habit.streak} days)",
                    suggestion_type="habit",
                    action="complete_habit",
                    parameters={"habit_id": habit.habit_id},
                    confidence=0.9,
                    priority=3,
                ))
        
        return suggestions
    
    async def _generate_context_suggestions(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate context-aware suggestions."""
        suggestions = []
        
        from ..core.ids import new_id
        
        # Check for unread emails
        if self.email:
            try:
                messages = await self.email.read_inbox(
                    account=f"{user_id}@gmail.com",
                    limit=5,
                    unread_only=True,
                )
                
                if len(messages) > 3:
                    suggestions.append(Suggestion(
                        suggestion_id=new_id("suggestion"),
                        text=f"You have {len(messages)} unread emails. Want me to summarize them?",
                        suggestion_type="context",
                        action="summarize_emails",
                        parameters={"count": len(messages)},
                        confidence=0.7,
                        priority=2,
                    ))
            except Exception as e:
                _log.debug("unread-email suggestion failed: %s", e)
        
        return suggestions
    
    # ── Helper Methods ───────────────────────────────────────────────────────
    
    def _find_pattern(self, user_id: str, action: str) -> Optional[Pattern]:
        """Find a pattern by user and action."""
        patterns = self._patterns.get(user_id, [])
        for pattern in patterns:
            if pattern.action == action:
                return pattern
        return None
    
    def _context_matches(self, pattern: Pattern, context: dict[str, Any]) -> float:
        """Calculate how well current context matches pattern."""
        if not pattern.context:
            return 0.5
        
        matches = 0
        total = 0
        
        for key, value in context.items():
            if key in pattern.context:
                total += 1
                freq = pattern.context[key]
                if str(value) in freq:
                    # Weight by frequency
                    count = freq[str(value)]
                    total_count = sum(freq.values())
                    matches += count / total_count
        
        if total == 0:
            return 0.5
        
        return matches / total
    
    def _format_pattern_suggestion(self, pattern: Pattern, context: dict[str, Any]) -> str:
        """Format a pattern as a suggestion text."""
        action = pattern.action.replace("_", " ").title()
        
        # Get most common context value
        day = context.get("day_of_week", "today")
        
        return f"You usually {action.lower()} on {day}s. Want me to do it now?"
    
    def _extract_pattern_params(self, pattern: Pattern, context: dict[str, Any]) -> dict[str, Any]:
        """Extract parameters from pattern for action execution."""
        params = {}
        
        # Get most common values for each context key
        for key, freq in pattern.context.items():
            if freq:
                most_common = freq.most_common(1)[0][0]
                params[key] = most_common
        
        return params
    
    def _is_habit_due(self, habit: UserHabit) -> bool:
        """Check if a habit is due."""
        if not habit.last_completed:
            return True
        
        now = time.time()
        elapsed = now - habit.last_completed
        
        frequency_seconds = {
            "daily": 24 * 3600,
            "weekly": 7 * 24 * 3600,
            "monthly": 30 * 24 * 3600,
        }
        
        threshold = frequency_seconds.get(habit.frequency, 24 * 3600)
        return elapsed > threshold
    
    # ── Habit Management ─────────────────────────────────────────────────────
    
    async def add_habit(
        self,
        user_id: str,
        name: str,
        *,
        description: str = "",
        frequency: str = "daily",
    ) -> UserHabit:
        """Add a new habit to track.
        
        Args:
            user_id: User ID
            name: Habit name
            description: Habit description
            frequency: How often (daily, weekly, monthly)
            
        Returns:
            Created UserHabit object
        """
        from ..core.ids import new_id
        
        habit = UserHabit(
            habit_id=new_id("habit"),
            user_id=user_id,
            name=name,
            description=description,
            frequency=frequency,
        )
        
        self._habits[user_id].append(habit)
        _log.info(f"Added habit: {name}")
        
        return habit
    
    async def complete_habit(self, user_id: str, habit_id: str) -> bool:
        """Mark a habit as completed.
        
        Args:
            user_id: User ID
            habit_id: Habit ID
            
        Returns:
            True if successful
        """
        habits = self._habits.get(user_id, [])
        
        for habit in habits:
            if habit.habit_id == habit_id:
                # Update streak
                if habit.last_completed:
                    elapsed = time.time() - habit.last_completed
                    if elapsed < 2 * 24 * 3600:  # Within 2 days
                        habit.streak += 1
                    else:
                        habit.streak = 1
                else:
                    habit.streak = 1
                
                habit.last_completed = time.time()
                habit.total_completions += 1
                
                _log.info(f"Completed habit: {habit.name} (streak: {habit.streak})")
                return True
        
        return False
    
    def get_habits(self, user_id: str) -> list[UserHabit]:
        """Get all habits for a user."""
        return self._habits.get(user_id, [])
    
    def get_patterns(self, user_id: str) -> list[Pattern]:
        """Get all patterns for a user."""
        return self._patterns.get(user_id, [])


# ── relationship cadence ───────────────────────────────────────────────────
# Tracks per-person contact cadence from markdown people pages (the store —
# no new infrastructure). Powers neglect detection, birthday surfacing, the
# morning briefing's People section, and the /memory trust command.

#: Default nudge thresholds (days without contact) per closeness tier.
DEFAULT_CADENCE_THRESHOLDS: dict[str, float] = {
    "close": 7.0,
    "normal": 30.0,
    "distant": 90.0,
}

#: Ranking weights per closeness tier.
CLOSENESS_WEIGHTS: dict[str, float] = {"close": 3.0, "normal": 1.0, "distant": 0.4}

#: People pages live here unless a people_dir is given explicitly.
DEFAULT_PEOPLE_DIR = "people"

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
)
_MONTHDAY_FORMATS = ("%m-%d", "%m/%d", "%b %d", "%B %d", "%d %b", "%d %B")

_LAST_CONTACT_KEYS = ("last contact", "last spoke", "last seen", "last talked")


def _parse_date(value: str) -> float | None:
    """Parse a full date string → epoch seconds. None when unparseable."""
    value = value.strip()
    if not value:
        return None
    try:  # ISO first (fromisoformat handles most variants)
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return None


def _parse_month_day(value: str) -> tuple[int, int] | None:
    """Parse a month/day birthday → (month, day). Year is ignored."""
    value = value.strip()
    if not value:
        return None
    for fmt in _MONTHDAY_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return (dt.month, dt.day)
        except ValueError:
            continue
    # numeric "02-29"/"2/29" — strptime rejects Feb 29 in non-leap 1900,
    # so fall back to a direct parse that tolerates it.
    import re as _re
    m = _re.fullmatch(r"(\d{1,2})[-/](\d{1,2})", value)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31 and not (month == 2 and day > 29):
            return (month, day)
    # ISO-ish "1990-05-04" → take month/day
    ts = _parse_date(value)
    if ts is not None:
        dt = datetime.fromtimestamp(ts)
        return (dt.month, dt.day)
    return None


def _slugify(name: str) -> str:
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "person"


@dataclass
class PersonRecord:
    """One person parsed from a people page. Malformed pages are skipped,
    never raised — fields fall back to None/'normal'."""
    name: str
    last_contact_ts: float | None = None
    closeness: str = "normal"  # close | normal | distant
    birthday: tuple[int, int] | None = None  # (month, day)
    anniversary: tuple[int, int] | None = None
    path: Any = None

    def days_since_contact(self, now: float) -> float | None:
        if self.last_contact_ts is None:
            return None
        return max(0.0, (now - self.last_contact_ts) / 86400.0)


@dataclass
class PersonNudge:
    """A relationship nudge: neglect or upcoming birthday."""
    kind: str  # "neglect" | "birthday" | "anniversary"
    name: str
    days_overdue: float = 0.0      # neglect: days past threshold
    days_until: float = 0.0        # birthday: days until the date
    closeness: str = "normal"
    score: float = 0.0             # closeness weight × days overdue
    detail: str = ""
    total_days: float | None = None  # neglect: total days since contact

    def text(self) -> str:
        if self.kind == "neglect":
            if self.total_days is None:
                return f"No contact recorded with {self.name} — worth checking in"
            return f"Haven't talked to {self.name} in {int(round(self.total_days))} days"
        label = "birthday" if self.kind == "birthday" else "anniversary"
        when = "today" if self.days_until < 1 else f"in {int(self.days_until)} days"
        return f"{self.name}'s {label} {when}"


class RelationshipCadence:
    """Per-person contact cadence over markdown people pages.

    Page format (all fields optional except the name):
        # Adaeze
        Closeness: close
        Birthday: May 4
        Anniversary: 2020-12-12
        Last contact: 2026-09-20

    Closeness: explicit ``Closeness:`` line wins; otherwise derived from
    ``INDEX.md`` ordering (top third → close, middle → normal, rest →
    distant); missing → "normal".
    """

    def __init__(
        self,
        people_dir: Any = None,
        *,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        from pathlib import Path as _Path
        if people_dir is None:
            people_dir = _Path.home() / ".devon" / DEFAULT_PEOPLE_DIR
        self.people_dir = _Path(people_dir)
        self.thresholds = dict(DEFAULT_CADENCE_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    # -- parsing -----------------------------------------------------------
    def _index_order(self) -> list[str]:
        """Names in INDEX.md order (closest first). [] when absent."""
        index = self.people_dir / "INDEX.md"
        names: list[str] = []
        try:
            for line in index.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("- **"):
                    name = line[4:].split("**")[0].strip()
                    if name:
                        names.append(name)
        except OSError:
            pass
        except Exception as exc:  # noqa: BLE001 — malformed index, ignore
            _log.debug("people INDEX.md unreadable: %s", exc)
        return names

    def _parse_page(self, path: Any, index_rank: dict[str, int],
                    index_total: int) -> PersonRecord | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        except Exception as exc:  # noqa: BLE001
            _log.debug("skipping unreadable people page %s: %s", path, exc)
            return None
        name: str | None = None
        fields: dict[str, str] = {}
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# ") and name is None:
                name = s[2:].strip()
            elif ":" in s and not s.startswith("#"):
                key, _, val = s.partition(":")
                fields[key.strip().lower()] = val.strip()
        if not name:
            name = path.stem.replace("-", " ").replace("_", " ").strip() or "person"

        last_contact: float | None = None
        for key in _LAST_CONTACT_KEYS:
            if key in fields:
                last_contact = _parse_date(fields[key])
                if last_contact is not None:
                    break

        closeness = fields.get("closeness", "").lower()
        if closeness not in ("close", "normal", "distant"):
            rank = index_rank.get(name.lower())
            if rank is not None and index_total > 0:
                third = index_total / 3.0
                closeness = ("close" if rank < third
                             else "normal" if rank < 2 * third else "distant")
            else:
                closeness = "normal"

        return PersonRecord(
            name=name,
            last_contact_ts=last_contact,
            closeness=closeness,
            birthday=_parse_month_day(fields.get("birthday", "")),
            anniversary=_parse_month_day(fields.get("anniversary", "")),
            path=path,
        )

    def people(self) -> list[PersonRecord]:
        """All parseable people. Never raises — malformed pages are skipped."""
        try:
            pages = sorted(self.people_dir.glob("*.md"))
        except OSError:
            return []
        order = self._index_order()
        rank = {n.lower(): i for i, n in enumerate(order)}
        out: list[PersonRecord] = []
        for page in pages:
            if page.name.upper() == "INDEX.MD":
                continue
            try:
                rec = self._parse_page(page, rank, len(order))
            except Exception as exc:  # noqa: BLE001 — fail closed per page
                _log.debug("skipping people page %s: %s", page, exc)
                continue
            if rec is not None:
                out.append(rec)
        return out

    # -- neglect -----------------------------------------------------------
    def neglected(self, now: float | None = None, *, limit: int = 5) -> list[PersonNudge]:
        """People past their closeness threshold, ranked by
        closeness-weight × days overdue. People with no recorded contact
        are treated as maximally overdue."""
        now = time.time() if now is None else now
        nudges: list[PersonNudge] = []
        for person in self.people():
            threshold = self.thresholds.get(person.closeness, 30.0)
            days = person.days_since_contact(now)
            if days is None:
                overdue = threshold  # unknown → nudge once, gently
                total = threshold
            else:
                overdue = days - threshold
                total = days
            if overdue <= 0:
                continue
            weight = CLOSENESS_WEIGHTS.get(person.closeness, 1.0)
            nudges.append(PersonNudge(
                kind="neglect", name=person.name, days_overdue=overdue,
                closeness=person.closeness, score=weight * overdue,
                detail=f"last contact {total:.0f} days ago (threshold {threshold:.0f})",
                total_days=None if days is None else total,
            ))
        nudges.sort(key=lambda n: -n.score)
        return nudges[:limit]

    # -- birthdays ---------------------------------------------------------
    @staticmethod
    def _days_until(month: int, day: int, now: float) -> float:
        from datetime import date as _date
        today = _date.fromtimestamp(now)
        year = today.year
        # Feb 29 → Feb 28 in non-leap years
        try:
            candidate = _date(year, month, day)
        except ValueError:
            candidate = _date(year, 2, 28)
        if candidate < today:
            try:
                candidate = _date(year + 1, month, day)
            except ValueError:
                candidate = _date(year + 1, 2, 28)
        return (candidate - today).days

    def upcoming_birthdays(self, now: float | None = None,
                           *, within_days: int = 14) -> list[PersonNudge]:
        """Birthdays/anniversaries within the window, soonest first."""
        now = time.time() if now is None else now
        out: list[PersonNudge] = []
        for person in self.people():
            for kind, md in (("birthday", person.birthday),
                             ("anniversary", person.anniversary)):
                if md is None:
                    continue
                days = self._days_until(md[0], md[1], now)
                if 0 <= days <= within_days:
                    out.append(PersonNudge(
                        kind=kind, name=person.name, days_until=float(days),
                        closeness=person.closeness,
                        score=CLOSENESS_WEIGHTS.get(person.closeness, 1.0) * 2.0,
                        detail=f"{kind} in {int(days)} days",
                    ))
        out.sort(key=lambda n: (n.days_until, -n.score))
        return out

    # -- recording ---------------------------------------------------------
    def record_contact(self, name: str, ts: float | None = None) -> Any:
        """Update (or create) a person's page with today's last-contact line.
        Idempotent: same timestamp → byte-identical page. Returns the page path."""
        from pathlib import Path as _Path
        ts = time.time() if ts is None else ts
        stamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        self.people_dir.mkdir(parents=True, exist_ok=True)
        want = _slugify(name)
        path = self.people_dir / f"{want}.md"
        if not path.exists():
            # Match an existing page case-insensitively: on case-sensitive
            # filesystems "Ada.md" and "ada.md" are different files, and we
            # must update the page the user actually has, not fork a second one.
            try:
                for cand in self.people_dir.glob("*.md"):
                    if _slugify(cand.stem) == want:
                        path = cand
                        break
            except OSError:
                pass
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            lines = text.splitlines()
            replaced = False
            for i, line in enumerate(lines):
                key = line.strip().split(":")[0].strip().lower()
                if key in _LAST_CONTACT_KEYS:
                    lines[i] = f"Last contact: {stamp}"
                    replaced = True
                    break
            if not replaced:
                # insert after the heading when there is one
                insert_at = 1 if lines and lines[0].startswith("#") else 0
                lines.insert(insert_at, f"Last contact: {stamp}")
            new_text = "\n".join(lines) + "\n"
        else:
            new_text = f"# {name.strip()}\nLast contact: {stamp}\n"
        # idempotent write: skip when nothing changed
        try:
            if path.exists() and path.read_text(encoding="utf-8") == new_text:
                return path
        except OSError:
            pass
        path.write_text(new_text, encoding="utf-8")
        _log.info("recorded contact with %s", name)
        return path


def control_memory(arg: str, context: Any) -> str:
    """`/memory` — trust view: what Devon remembers. Summaries only,
    never raw memory dumps. Slash commands are already owner-only
    (runtime gates control commands to owner chats)."""
    arg = (arg or "").strip()
    people_dir = getattr(getattr(context, "settings", None), "people_dir", None)
    cadence = RelationshipCadence(people_dir=people_dir)
    people = cadence.people()

    if arg:
        wanted = arg.lower()
        match = next((p for p in people if p.name.lower() == wanted), None)
        if match is None:
            match = next((p for p in people if wanted in p.name.lower()), None)
        if match is None:
            return f"No person page for '{arg}'. I track {len(people)} people."
        lines = [f"**{match.name}**", f"Closeness: {match.closeness}"]
        if match.last_contact_ts:
            days = (time.time() - match.last_contact_ts) / 86400.0
            lines.append(f"Last contact: {days:.0f} days ago")
        else:
            lines.append("Last contact: not recorded")
        if match.birthday:
            lines.append(f"Birthday: {match.birthday[1]:02d}-{match.birthday[0]:02d}")
        if match.anniversary:
            lines.append(f"Anniversary: {match.anniversary[1]:02d}-{match.anniversary[0]:02d}")
        return "\n".join(lines)

    # summary view
    mem = getattr(context, "memory", None)
    counts: dict[str, int] = {}
    total = 0
    if mem is not None:
        try:
            counts = mem.counts_by_kind() or {}
            total = sum(counts.values())
        except Exception as exc:  # noqa: BLE001
            _log.debug("/memory counts unavailable: %s", exc)
    recent = sorted(
        (p for p in people if p.last_contact_ts),
        key=lambda p: -p.last_contact_ts,  # type: ignore[operator]
    )[:5]
    lines = [
        f"I track **{len(people)} people** and **{total} long-term memories**"
        + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})" if counts else "")
        + ".",
    ]
    if recent:
        lines.append("Recent contact:")
        for p in recent:
            days = (time.time() - p.last_contact_ts) / 86400.0  # type: ignore[operator]
            lines.append(f"• {p.name} — {days:.0f} days ago")
    else:
        lines.append("No contact history recorded yet.")
    lines.append("Ask `/memory <name>` for one person's summary.")
    return "\n".join(lines)


__all__ += [
    "RelationshipCadence",
    "PersonRecord",
    "PersonNudge",
    "DEFAULT_CADENCE_THRESHOLDS",
    "CLOSENESS_WEIGHTS",
    "control_memory",
]
