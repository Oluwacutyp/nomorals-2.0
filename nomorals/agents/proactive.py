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
