"""Goals and ideas tracking system.

Supports:
- Durable goals with workspaces and subgoals
- Progress tracking and briefings
- Idea cards with dismissal tracking
- Community inspiration feed

Usage:
    goals = GoalTracker(db)
    
    # Create a goal
    goal = await goals.create(
        user_id="user123",
        title="Learn Python",
        description="Master Python programming",
        target_date=datetime(2026, 12, 31),
    )
    
    # Add subgoal
    subgoal = await goals.add_subgoal(goal.goal_id, "Complete basics tutorial")
    
    # Update progress
    await goals.update_progress(goal.goal_id, progress=25.0)
    
    # Create idea
    idea = await ideas.create(
        user_id="user123",
        title="Build a game",
        description="Create a simple RPG game in Python",
        tags=["gaming", "python"],
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["GoalTracker", "IdeaTracker", "Goal", "Subgoal", "Idea"]

_log = get_logger(__name__)


@dataclass
class Subgoal:
    """A subgoal within a larger goal."""
    
    subgoal_id: str
    goal_id: str
    title: str
    is_completed: bool = False
    completed_at: Optional[float] = None
    order: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "is_completed": self.is_completed,
            "completed_at": self.completed_at,
            "order": self.order,
        }


@dataclass
class Goal:
    """A durable goal with progress tracking."""
    
    goal_id: str
    user_id: str
    title: str
    description: str = ""
    status: str = "active"  # active, completed, paused, abandoned
    progress: float = 0.0  # 0-100
    priority: int = 0
    target_date: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    tags: list[str] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    notes: str = ""
    workspace: str = "default"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "priority": self.priority,
            "target_date": self.target_date,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "tags": self.tags,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "workspace": self.workspace,
        }


@dataclass
class Idea:
    """An idea card."""
    
    idea_id: str
    user_id: str
    title: str
    description: str = ""
    status: str = "active"  # active, dismissed, promoted_to_goal
    created_at: float = field(default_factory=time.time)
    dismissed_at: Optional[float] = None
    dismiss_reason: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""  # Where the idea came from
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "tags": self.tags,
            "source": self.source,
        }


class GoalTracker:
    """Tracks goals with subgoals and progress."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()
        _log.info("Goal tracker initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    progress REAL NOT NULL DEFAULT 0.0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    target_date REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '',
                    workspace TEXT NOT NULL DEFAULT 'default'
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS subgoals (
                    subgoal_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    is_completed INTEGER NOT NULL DEFAULT 0,
                    completed_at REAL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_subgoals_goal ON subgoals(goal_id)")
    
    async def create(
        self,
        user_id: str,
        title: str,
        *,
        description: str = "",
        target_date: float | None = None,
        priority: int = 0,
        tags: list[str] | None = None,
        workspace: str = "default",
    ) -> Goal:
        """Create a new goal."""
        goal_id = new_id("goal")
        now = time.time()
        tags = tags or []
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goals (goal_id, user_id, title, description, status, progress, priority,
                    target_date, created_at, updated_at, tags, workspace)
                VALUES (?, ?, ?, ?, 'active', 0.0, ?, ?, ?, ?, ?, ?)
            """, (goal_id, user_id, title, description, priority, target_date, now, now,
                  json.dumps(tags), workspace))
        
        goal = Goal(
            goal_id=goal_id,
            user_id=user_id,
            title=title,
            description=description,
            target_date=target_date,
            priority=priority,
            tags=tags,
            workspace=workspace,
        )
        
        _log.info(f"Created goal: {title}")
        return goal
    
    async def add_subgoal(
        self,
        goal_id: str,
        title: str,
        *,
        order: int = 0,
    ) -> Subgoal:
        """Add a subgoal to a goal."""
        subgoal_id = new_id("subgoal")
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO subgoals (subgoal_id, goal_id, title, is_completed, sort_order)
                VALUES (?, ?, ?, 0, ?)
            """, (subgoal_id, goal_id, title, order))
        
        subgoal = Subgoal(
            subgoal_id=subgoal_id,
            goal_id=goal_id,
            title=title,
            order=order,
        )
        
        # Auto-recalculate progress
        await self._recalculate_progress(goal_id)
        
        return subgoal
    
    async def complete_subgoal(self, subgoal_id: str) -> bool:
        """Mark a subgoal as completed."""
        now = time.time()
        
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT goal_id FROM subgoals WHERE subgoal_id = ?",
                (subgoal_id,)
            )
            if not row:
                return False
            
            self.db.execute("""
                UPDATE subgoals SET is_completed = 1, completed_at = ? WHERE subgoal_id = ?
            """, (now, subgoal_id))
        
        # Recalculate progress
        await self._recalculate_progress(row["goal_id"])
        return True
    
    async def update_progress(self, goal_id: str, progress: float) -> None:
        """Manually update goal progress."""
        with self.db.transaction():
            self.db.execute("""
                UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?
            """, (max(0.0, min(100.0, progress)), time.time(), goal_id))
    
    async def complete(self, goal_id: str) -> None:
        """Mark a goal as completed."""
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                UPDATE goals SET status = 'completed', progress = 100.0, completed_at = ?, updated_at = ?
                WHERE goal_id = ?
            """, (now, now, goal_id))
    
    async def list_goals(
        self,
        user_id: str,
        *,
        status: str | None = None,
        workspace: str | None = None,
    ) -> list[Goal]:
        """List goals for a user."""
        query = "SELECT * FROM goals WHERE user_id = ?"
        params: list[Any] = [user_id]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if workspace:
            query += " AND workspace = ?"
            params.append(workspace)
        
        query += " ORDER BY priority DESC, created_at DESC"
        
        rows = self.db.query(query, params)
        
        goals = []
        for row in rows:
            # Fetch subgoals
            subgoal_rows = self.db.query(
                "SELECT * FROM subgoals WHERE goal_id = ? ORDER BY sort_order",
                (row["goal_id"],)
            )
            
            subgoals = [
                Subgoal(
                    subgoal_id=s["subgoal_id"],
                    goal_id=s["goal_id"],
                    title=s["title"],
                    is_completed=bool(s["is_completed"]),
                    completed_at=s["completed_at"],
                    order=s["sort_order"],
                )
                for s in subgoal_rows
            ]
            
            goals.append(Goal(
                goal_id=row["goal_id"],
                user_id=row["user_id"],
                title=row["title"],
                description=row["description"],
                status=row["status"],
                progress=row["progress"],
                priority=row["priority"],
                target_date=row["target_date"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                tags=json.loads(row["tags"]),
                subgoals=subgoals,
                notes=row["notes"],
                workspace=row["workspace"],
            ))
        
        return goals
    
    async def get_briefing(self, user_id: str) -> str:
        """Generate a progress briefing for all active goals."""
        goals = await self.list_goals(user_id, status="active")
        
        if not goals:
            return "No active goals."
        
        lines = ["📊 **Goal Progress Briefing:**\n"]
        
        for goal in goals:
            status_emoji = "🟢" if goal.progress >= 75 else "🟡" if goal.progress >= 25 else "🔴"
            lines.append(f"{status_emoji} **{goal.title}** — {goal.progress:.0f}%")
            
            if goal.subgoals:
                completed = sum(1 for s in goal.subgoals if s.is_completed)
                lines.append(f"   Subgoals: {completed}/{len(goal.subgoals)} completed")
            
            if goal.target_date:
                from datetime import datetime
                days_left = (goal.target_date - time.time()) / 86400
                if days_left > 0:
                    lines.append(f"   ⏰ {days_left:.0f} days remaining")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def _recalculate_progress(self, goal_id: str) -> None:
        """Recalculate goal progress from subgoals."""
        subgoals = self.db.query(
            "SELECT * FROM subgoals WHERE goal_id = ?",
            (goal_id,)
        )
        
        if not subgoals:
            return
        
        completed = sum(1 for s in subgoals if s["is_completed"])
        progress = (completed / len(subgoals)) * 100.0
        
        with self.db.transaction():
            self.db.execute("""
                UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?
            """, (progress, time.time(), goal_id))


class IdeaTracker:
    """Tracks ideas with dismissal tracking."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()
        _log.info("Idea tracker initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS ideas (
                    idea_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    dismissed_at REAL,
                    dismiss_reason TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_ideas_user ON ideas(user_id, status)")
    
    async def create(
        self,
        user_id: str,
        title: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        source: str = "",
    ) -> Idea:
        """Create a new idea."""
        idea_id = new_id("idea")
        now = time.time()
        tags = tags or []
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO ideas (idea_id, user_id, title, description, status, created_at, tags, source)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """, (idea_id, user_id, title, description, now, json.dumps(tags), source))
        
        return Idea(
            idea_id=idea_id,
            user_id=user_id,
            title=title,
            description=description,
            tags=tags,
            source=source,
        )
    
    async def dismiss(self, idea_id: str, *, reason: str = "") -> bool:
        """Dismiss an idea."""
        with self.db.transaction():
            self.db.execute("""
                UPDATE ideas SET status = 'dismissed', dismissed_at = ?, dismiss_reason = ?
                WHERE idea_id = ?
            """, (time.time(), reason, idea_id))
        return True
    
    async def promote_to_goal(self, idea_id: str, goal_tracker: GoalTracker) -> Optional[str]:
        """Promote an idea to a goal."""
        row = self.db.query_one("SELECT * FROM ideas WHERE idea_id = ?", (idea_id,))
        if not row:
            return None
        
        goal = await goal_tracker.create(
            user_id=row["user_id"],
            title=row["title"],
            description=row["description"],
            tags=json.loads(row["tags"]),
        )
        
        with self.db.transaction():
            self.db.execute("""
                UPDATE ideas SET status = 'promoted_to_goal' WHERE idea_id = ?
            """, (idea_id,))
        
        return goal.goal_id
    
    async def list_ideas(
        self,
        user_id: str,
        *,
        status: str | None = None,
    ) -> list[Idea]:
        """List ideas for a user."""
        query = "SELECT * FROM ideas WHERE user_id = ?"
        params: list[Any] = [user_id]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        rows = self.db.query(query, params)
        
        return [
            Idea(
                idea_id=row["idea_id"],
                user_id=row["user_id"],
                title=row["title"],
                description=row["description"],
                status=row["status"],
                created_at=row["created_at"],
                dismissed_at=row["dismissed_at"],
                dismiss_reason=row["dismiss_reason"],
                tags=json.loads(row["tags"]),
                source=row["source"],
            )
            for row in rows
        ]
