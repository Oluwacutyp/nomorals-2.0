"""Proactive gateway - wires ProactiveEngine to chat platforms via Notifier.

Closes the loop: proactive engine generates suggestions → gateway delivers them
to WhatsApp/Telegram based on user preferences.

Features:
- Preference store (what to notify about, when, how often)
- Quiet hours (don't ping at 3am)
- Rate limiting (max N suggestions per day)
- Channel selection (WhatsApp vs Telegram vs both)
- Acceptance tracking (did user act on suggestion?)

Usage:
    gateway = ProactiveGateway(proactive_engine, notifier, scheduler, db)
    
    # Set up scheduled checks
    await gateway.setup_schedule()  # Every 30 minutes
    
    # Or trigger manually
    await gateway.check_and_notify(user_id="user123")
    
    # Update preferences
    await gateway.set_preferences(
        user_id="user123",
        quiet_hours_start=23,  # 11pm
        quiet_hours_end=7,     # 7am
        max_suggestions_per_day=10,
        channels=["whatsapp", "telegram"],
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..agents.notifier import Notifier
from ..agents.proactive import ProactiveEngine, Suggestion
from ..scheduler import Scheduler
from ..storage.db import Database

__all__ = ["ProactiveGateway", "ProactivePreferences"]

_log = get_logger(__name__)


@dataclass
class ProactivePreferences:
    """User preferences for proactive notifications."""
    
    user_id: str
    quiet_hours_start: int = 23  # 11pm
    quiet_hours_end: int = 7     # 7am
    max_suggestions_per_day: int = 10
    channels: list[str] = field(default_factory=lambda: ["whatsapp"])
    enabled_categories: list[str] = field(default_factory=lambda: [
        "pattern", "context", "reminder", "habit"
    ])
    min_confidence: float = 0.6
    enabled: bool = True
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "max_suggestions_per_day": self.max_suggestions_per_day,
            "channels": self.channels,
            "enabled_categories": self.enabled_categories,
            "min_confidence": self.min_confidence,
            "enabled": self.enabled,
        }
    
    def is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        hour = datetime.now().hour
        if self.quiet_hours_start > self.quiet_hours_end:
            # Crosses midnight (e.g., 23:00 - 07:00)
            return hour >= self.quiet_hours_start or hour < self.quiet_hours_end
        else:
            # Same day (e.g., 13:00 - 15:00)
            return self.quiet_hours_start <= hour < self.quiet_hours_end


class ProactiveGateway:
    """Wires ProactiveEngine to chat platforms via Notifier."""
    
    def __init__(
        self,
        proactive_engine: ProactiveEngine,
        notifier: Notifier,
        scheduler: Scheduler,
        db: Database,
    ) -> None:
        self.proactive = proactive_engine
        self.notifier = notifier
        self.scheduler = scheduler
        self.db = db
        self._ensure_schema()
        _log.info("Proactive gateway initialized")
    
    def _ensure_schema(self) -> None:
        """Create proactive gateway tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS proactive_preferences (
                    user_id TEXT PRIMARY KEY,
                    quiet_hours_start INTEGER NOT NULL DEFAULT 23,
                    quiet_hours_end INTEGER NOT NULL DEFAULT 7,
                    max_suggestions_per_day INTEGER NOT NULL DEFAULT 10,
                    channels TEXT NOT NULL DEFAULT '["whatsapp"]',
                    enabled_categories TEXT NOT NULL DEFAULT '["pattern","context","reminder","habit"]',
                    min_confidence REAL NOT NULL DEFAULT 0.6,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS proactive_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    suggestion_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    delivered_at REAL NOT NULL,
                    accepted INTEGER NOT NULL DEFAULT 0,
                    rejected INTEGER NOT NULL DEFAULT 0,
                    acted_on INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS proactive_daily_counts (
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, date)
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_user ON proactive_deliveries(user_id, delivered_at)")
    
    async def setup_schedule(self, *, interval_minutes: int = 30) -> str:
        """Set up scheduled proactive checks.
        
        Args:
            interval_minutes: How often to check (default: 30 min)
            
        Returns:
            Cron job ID
        """
        cron_expr = f"*/{interval_minutes} * * * *"  # Every N minutes
        
        job = await self.scheduler.schedule_cron(
            task_id=None,
            cron_expr=cron_expr,
            action="proactive_check",
            parameters={},
        )
        
        _log.info(f"Scheduled proactive checks: every {interval_minutes} minutes")
        return job.task_id
    
    async def _handle_scheduled_check(self, params: dict[str, Any]) -> None:
        """Handler for scheduled proactive checks."""
        # Get all users with proactive enabled
        users = self.db.query("""
            SELECT user_id FROM proactive_preferences WHERE enabled = 1
        """)
        
        for row in users:
            try:
                await self.check_and_notify(row["user_id"])
            except Exception as e:
                _log.error(f"Proactive check failed for {row['user_id']}: {e}")
    
    async def check_and_notify(self, user_id: str) -> list[Suggestion]:
        """Check for proactive suggestions and notify user.
        
        Args:
            user_id: User to check for
            
        Returns:
            List of suggestions that were delivered
        """
        # Get preferences
        prefs = await self.get_preferences(user_id)
        
        if not prefs.enabled:
            return []
        
        # Check quiet hours
        if prefs.is_quiet_hours():
            _log.debug(f"Quiet hours for {user_id}, skipping proactive check")
            return []
        
        # Check daily limit
        today = datetime.now().strftime("%Y-%m-%d")
        count_row = self.db.query_one("""
            SELECT count FROM proactive_daily_counts
            WHERE user_id = ? AND date = ?
        """, (user_id, today))
        
        current_count = count_row["count"] if count_row else 0
        
        if current_count >= prefs.max_suggestions_per_day:
            _log.debug(f"Daily limit reached for {user_id} ({current_count}/{prefs.max_suggestions_per_day})")
            return []
        
        # Get suggestions from engine
        suggestions = await self.proactive.get_suggestions(user_id=user_id)
        
        # Filter by preferences
        filtered = [
            s for s in suggestions
            if s.suggestion_type in prefs.enabled_categories
            and s.confidence >= prefs.min_confidence
        ]
        
        if not filtered:
            return []
        
        # Deliver top suggestions
        delivered = []
        remaining_quota = prefs.max_suggestions_per_day - current_count
        
        for suggestion in filtered[:remaining_quota]:
            success = await self._deliver_suggestion(user_id, suggestion, prefs)
            if success:
                delivered.append(suggestion)
        
        # Update daily count
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO proactive_daily_counts (user_id, date, count)
                VALUES (?, ?, ?)
            """, (user_id, today, current_count + len(delivered)))
        
        _log.info(f"Delivered {len(delivered)} proactive suggestions to {user_id}")
        return delivered
    
    async def _deliver_suggestion(
        self,
        user_id: str,
        suggestion: Suggestion,
        prefs: ProactivePreferences,
    ) -> bool:
        """Deliver a suggestion via configured channels."""
        delivery_id = new_id("delivery")
        delivered = False
        
        for channel in prefs.channels:
            try:
                # Format message
                message = self._format_suggestion(suggestion)
                
                # Send via notifier
                self.notifier.publish(
                    kind="proactive",
                    title=f"💡 {suggestion.text[:50]}",
                    body=message,
                    force=False,
                )
                
                # Record delivery
                with self.db.transaction():
                    self.db.execute("""
                        INSERT INTO proactive_deliveries
                        (delivery_id, user_id, suggestion_id, channel, delivered_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (delivery_id, user_id, suggestion.suggestion_id, channel, time.time()))
                
                delivered = True
                break  # Only deliver once even if multiple channels configured
                
            except Exception as e:
                _log.error(f"Failed to deliver suggestion via {channel}: {e}")
        
        return delivered
    
    def _format_suggestion(self, suggestion: Suggestion) -> str:
        """Format suggestion as user-friendly message."""
        lines = [suggestion.text]
        
        if suggestion.action:
            lines.append(f"\nAction: {suggestion.action}")
        
        if suggestion.confidence > 0:
            lines.append(f"Confidence: {suggestion.confidence:.0%}")
        
        return "\n".join(lines)
    
    async def get_preferences(self, user_id: str) -> ProactivePreferences:
        """Get user preferences (creates defaults if not set)."""
        row = self.db.query_one(
            "SELECT * FROM proactive_preferences WHERE user_id = ?",
            (user_id,)
        )
        
        if not row:
            # Create defaults
            prefs = ProactivePreferences(user_id=user_id)
            await self.set_preferences(prefs)
            return prefs
        
        return ProactivePreferences(
            user_id=row["user_id"],
            quiet_hours_start=row["quiet_hours_start"],
            quiet_hours_end=row["quiet_hours_end"],
            max_suggestions_per_day=row["max_suggestions_per_day"],
            channels=json.loads(row["channels"]),
            enabled_categories=json.loads(row["enabled_categories"]),
            min_confidence=row["min_confidence"],
            enabled=bool(row["enabled"]),
            updated_at=row["updated_at"],
        )
    
    async def set_preferences(self, prefs: ProactivePreferences) -> None:
        """Update user preferences."""
        prefs.updated_at = time.time()
        
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO proactive_preferences
                (user_id, quiet_hours_start, quiet_hours_end, max_suggestions_per_day,
                 channels, enabled_categories, min_confidence, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prefs.user_id,
                prefs.quiet_hours_start,
                prefs.quiet_hours_end,
                prefs.max_suggestions_per_day,
                json.dumps(prefs.channels),
                json.dumps(prefs.enabled_categories),
                prefs.min_confidence,
                int(prefs.enabled),
                prefs.updated_at,
            ))
        
        _log.info(f"Updated proactive preferences for {prefs.user_id}")
    
    async def record_acceptance(self, delivery_id: str) -> None:
        """Record that user accepted a suggestion."""
        with self.db.transaction():
            self.db.execute("""
                UPDATE proactive_deliveries SET accepted = 1 WHERE delivery_id = ?
            """, (delivery_id,))
    
    async def record_rejection(self, delivery_id: str) -> None:
        """Record that user rejected a suggestion."""
        with self.db.transaction():
            self.db.execute("""
                UPDATE proactive_deliveries SET rejected = 1 WHERE delivery_id = ?
            """, (delivery_id,))
    
    async def record_action(self, delivery_id: str) -> None:
        """Record that user acted on a suggestion."""
        with self.db.transaction():
            self.db.execute("""
                UPDATE proactive_deliveries SET acted_on = 1 WHERE delivery_id = ?
            """, (delivery_id,))
    
    async def get_delivery_stats(self, user_id: str, *, days: int = 30) -> dict[str, Any]:
        """Get delivery statistics for a user."""
        since = time.time() - (days * 24 * 3600)
        
        stats = self.db.query_one("""
            SELECT
                COUNT(*) as total,
                SUM(accepted) as accepted,
                SUM(rejected) as rejected,
                SUM(acted_on) as acted_on
            FROM proactive_deliveries
            WHERE user_id = ? AND delivered_at >= ?
        """, (user_id, since))
        
        total = stats["total"] or 0
        accepted = stats["accepted"] or 0
        rejected = stats["rejected"] or 0
        acted_on = stats["acted_on"] or 0
        
        return {
            "total_delivered": total,
            "accepted": accepted,
            "rejected": rejected,
            "acted_on": acted_on,
            "acceptance_rate": (accepted / total) if total > 0 else 0,
            "action_rate": (acted_on / total) if total > 0 else 0,
            "period_days": days,
        }
