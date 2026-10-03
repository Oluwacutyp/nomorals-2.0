"""Goals and ideas tracking system.

Durable per-user goals with subgoals, progress history, deadlines and
reminders, goal templates, and progress briefings — plus an idea card
tracker with dismissal tracking and promotion into goals.

Usage:
    goals = GoalTracker(db)

    # Create a goal
    goal = await goals.create(
        user_id="user123",
        title="Learn Python",
        description="Master Python programming",
        target_date=time.time() + 90 * 86400,
    )

    # Add subgoal
    subgoal = await goals.add_subgoal(goal.goal_id, "Complete basics tutorial")

    # Update progress (records a history entry)
    await goals.update_progress(goal.goal_id, 25.0, note="finished chapter 3")

    # From a template
    goal = await goals.create_from_template("user123", "learn-language")

    # Deadlines and reminders
    due = await goals.due_goals("user123", within_days=7)
    await goals.add_reminder(goal.goal_id, time.time() + 86400, "check in")

    # Ideas
    ideas = IdeaTracker(db)
    idea = await ideas.create(user_id="user123", title="Build a game")
    goal_id = await ideas.promote_to_goal(idea.idea_id, goals)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "GoalTracker",
    "IdeaTracker",
    "Goal",
    "Subgoal",
    "Idea",
    "Reminder",
    "ProgressEntry",
    "GOAL_TEMPLATES",
]

_log = get_logger(__name__)

GOAL_STATUSES = ("active", "completed", "paused", "abandoned", "archived")
IDEA_STATUSES = ("active", "dismissed", "promoted_to_goal")

#: Built-in goal templates. Each entry supplies a title, description,
#: default priority/tags and a starter subgoal list.
GOAL_TEMPLATES: dict[str, dict[str, Any]] = {
    "learn-language": {
        "title": "Learn {subject}",
        "description": "Reach conversational fluency through daily practice.",
        "priority": 2,
        "tags": ["learning", "language"],
        "subgoals": [
            "Pick a course or app and finish the onboarding",
            "Learn the 500 most common words",
            "Hold a 5-minute conversation",
            "Watch a show without subtitles",
            "Pass a mock proficiency test",
        ],
    },
    "read-books": {
        "title": "Read {count} books",
        "description": "Finish the reading list, one book at a time.",
        "priority": 1,
        "tags": ["learning", "reading"],
        "subgoals": [
            "Build the reading list",
            "Read 25% of the list",
            "Read 50% of the list",
            "Read 75% of the list",
            "Finish the list and write notes",
        ],
    },
    "fitness": {
        "title": "Get fit: {subject}",
        "description": "Build a sustainable training habit.",
        "priority": 3,
        "tags": ["health", "fitness"],
        "subgoals": [
            "Set a baseline (measurements / test workout)",
            "Train 3x per week for a month",
            "Dial in sleep and nutrition",
            "Hit the first milestone",
            "Re-test and set the next target",
        ],
    },
    "ship-project": {
        "title": "Ship {subject}",
        "description": "Take the project from idea to launched.",
        "priority": 3,
        "tags": ["build", "project"],
        "subgoals": [
            "Write the one-paragraph spec",
            "Build the minimum viable version",
            "Test it end to end",
            "Polish the rough edges",
            "Launch and announce",
        ],
    },
    "save-money": {
        "title": "Save {subject}",
        "description": "Hit the savings target on schedule.",
        "priority": 2,
        "tags": ["money", "finance"],
        "subgoals": [
            "Set the target amount and deadline",
            "Open / pick the savings vehicle",
            "Automate the first transfer",
            "Reach 50% of the target",
            "Reach 100% of the target",
        ],
    },
    "declutter": {
        "title": "Declutter {subject}",
        "description": "Clear the space, one zone at a time.",
        "priority": 1,
        "tags": ["home", "organization"],
        "subgoals": [
            "Pick the first zone",
            "Sort: keep / donate / trash",
            "Clear the first zone",
            "Clear the remaining zones",
            "Set up a maintenance routine",
        ],
    },
}


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
class ProgressEntry:
    """One recorded progress event for a goal."""

    history_id: str
    goal_id: str
    progress: float
    note: str
    recorded_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_id": self.history_id,
            "goal_id": self.goal_id,
            "progress": self.progress,
            "note": self.note,
            "recorded_at": self.recorded_at,
        }


@dataclass
class Reminder:
    """A scheduled nudge attached to a goal."""

    reminder_id: str
    goal_id: str
    user_id: str
    remind_at: float
    message: str = ""
    acknowledged: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "goal_id": self.goal_id,
            "user_id": self.user_id,
            "remind_at": self.remind_at,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "created_at": self.created_at,
        }


@dataclass
class Goal:
    """A durable goal with progress tracking."""

    goal_id: str
    user_id: str
    title: str
    description: str = ""
    status: str = "active"  # active, completed, paused, abandoned, archived
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
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "tags": self.tags,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "notes": self.notes,
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
            "dismissed_at": self.dismissed_at,
            "dismiss_reason": self.dismiss_reason,
            "tags": self.tags,
            "source": self.source,
            "metadata": self.metadata,
        }


class GoalTracker:
    """Tracks goals with subgoals, progress history, deadlines and reminders."""

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

            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_progress_history (
                    history_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    progress REAL NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    recorded_at REAL NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)

            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_reminders (
                    reminder_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    remind_at REAL NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)

            self.db.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_goals_target ON goals(user_id, target_date)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_subgoals_goal ON subgoals(goal_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_progress_goal ON goal_progress_history(goal_id, recorded_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user ON goal_reminders(user_id, remind_at)")

    # ── row mapping ───────────────────────────────────────────────────────

    def _row_to_subgoal(self, s: dict[str, Any]) -> Subgoal:
        return Subgoal(
            subgoal_id=s["subgoal_id"],
            goal_id=s["goal_id"],
            title=s["title"],
            is_completed=bool(s["is_completed"]),
            completed_at=s["completed_at"],
            order=s["sort_order"],
        )

    def _row_to_goal(self, row: dict[str, Any], subgoals: list[Subgoal] | None = None) -> Goal:
        return Goal(
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
            tags=json.loads(row["tags"] or "[]"),
            subgoals=subgoals if subgoals is not None else [],
            notes=row["notes"],
            workspace=row["workspace"],
        )

    def _attach_subgoals(self, rows: list[dict[str, Any]]) -> list[Goal]:
        """Batch-map goal rows to Goal objects with their subgoals (one query)."""
        if not rows:
            return []
        goal_ids = [r["goal_id"] for r in rows]
        placeholders = ",".join("?" * len(goal_ids))
        subgoal_rows = self.db.query(
            f"SELECT * FROM subgoals WHERE goal_id IN ({placeholders}) ORDER BY sort_order",
            goal_ids,
        )
        by_goal: dict[str, list[Subgoal]] = {r["goal_id"]: [] for r in rows}
        for s in subgoal_rows:
            if s["goal_id"] in by_goal:
                by_goal[s["goal_id"]].append(self._row_to_subgoal(s))
        return [self._row_to_goal(r, by_goal[r["goal_id"]]) for r in rows]

    # ── create / read ─────────────────────────────────────────────────────

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
        notes: str = "",
    ) -> Goal:
        """Create a new goal."""
        title = (title or "").strip()
        if not title:
            raise ValueError("goal title must not be empty")
        goal_id = new_id("goal")
        now = time.time()
        tags = list(tags or [])

        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goals (goal_id, user_id, title, description, status, progress, priority,
                    target_date, created_at, updated_at, tags, notes, workspace)
                VALUES (?, ?, ?, ?, 'active', 0.0, ?, ?, ?, ?, ?, ?, ?)
            """, (goal_id, user_id, title, description, priority, target_date, now, now,
                  json.dumps(tags), notes, workspace))
            self._record_history(goal_id, 0.0, "goal created")

        _log.info(f"Created goal: {title}")
        return await self.get(goal_id) or Goal(
            goal_id=goal_id, user_id=user_id, title=title, description=description,
            target_date=target_date, priority=priority, tags=tags,
            notes=notes, workspace=workspace)

    async def create_from_template(
        self,
        user_id: str,
        template_name: str,
        *,
        title: str | None = None,
        subject: str = "",
        count: str = "",
        target_date: float | None = None,
        priority: int | None = None,
        tags: list[str] | None = None,
        workspace: str = "default",
    ) -> Goal:
        """Create a goal from a built-in template (see GOAL_TEMPLATES).

        ``{subject}`` / ``{count}`` placeholders in the template title are
        filled from the keyword arguments; pass ``title`` to override.
        """
        template = GOAL_TEMPLATES.get(template_name)
        if template is None:
            raise KeyError(
                f"unknown goal template {template_name!r} "
                f"(available: {', '.join(sorted(GOAL_TEMPLATES))})")
        filled = (title if title is not None else template["title"]).format(
            subject=subject, count=count)
        goal = await self.create(
            user_id,
            filled,
            description=template["description"],
            target_date=target_date,
            priority=template["priority"] if priority is None else priority,
            tags=list(tags) if tags is not None else list(template["tags"]),
            workspace=workspace,
        )
        for order, step in enumerate(template["subgoals"]):
            await self.add_subgoal(goal.goal_id, step.format(subject=subject, count=count),
                                   order=order)
        return await self.get(goal.goal_id) or goal

    async def get(self, goal_id: str) -> Goal | None:
        """Fetch a single goal by id, with subgoals attached."""
        row = self.db.query_one("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
        if not row:
            return None
        return self._attach_subgoals([row])[0]

    async def _require(self, goal_id: str) -> Goal:
        goal = await self.get(goal_id)
        if goal is None:
            raise KeyError(f"no goal {goal_id!r}")
        return goal

    # ── update ────────────────────────────────────────────────────────────

    async def update(
        self,
        goal_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        priority: int | None = None,
        tags: list[str] | None = None,
        target_date: float | None = None,
        clear_target_date: bool = False,
        notes: str | None = None,
        workspace: str | None = None,
    ) -> Goal:
        """Update goal fields. Raises KeyError when the goal does not exist."""
        await self._require(goal_id)
        updates: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("goal title must not be empty")
            updates["title"] = title
        if description is not None:
            updates["description"] = description
        if priority is not None:
            updates["priority"] = int(priority)
        if tags is not None:
            updates["tags"] = json.dumps(list(tags))
        if clear_target_date:
            updates["target_date"] = None
        elif target_date is not None:
            updates["target_date"] = target_date
        if notes is not None:
            updates["notes"] = notes
        if workspace is not None:
            updates["workspace"] = workspace
        if updates:
            updates["updated_at"] = time.time()
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            with self.db.transaction():
                self.db.execute(
                    f"UPDATE goals SET {set_clause} WHERE goal_id = ?",
                    (*updates.values(), goal_id))
        return await self.get(goal_id) or await self._require(goal_id)

    async def set_status(self, goal_id: str, status: str) -> Goal:
        """Set the goal status. Raises KeyError / ValueError on bad input."""
        if status not in GOAL_STATUSES:
            raise ValueError(f"unknown status {status!r} (expected one of {GOAL_STATUSES})")
        await self._require(goal_id)
        with self.db.transaction():
            self.db.execute(
                "UPDATE goals SET status = ?, updated_at = ? WHERE goal_id = ?",
                (status, time.time(), goal_id))
        return await self._require(goal_id)

    async def pause(self, goal_id: str) -> Goal:
        return await self.set_status(goal_id, "paused")

    async def resume(self, goal_id: str) -> Goal:
        return await self.set_status(goal_id, "active")

    async def abandon(self, goal_id: str) -> Goal:
        return await self.set_status(goal_id, "abandoned")

    async def archive(self, goal_id: str) -> Goal:
        """Archive a goal: out of the active list, kept for history."""
        return await self.set_status(goal_id, "archived")

    async def unarchive(self, goal_id: str) -> Goal:
        """Return an archived goal to active."""
        goal = await self._require(goal_id)
        if goal.status != "archived":
            raise ValueError(f"goal {goal_id!r} is not archived (status={goal.status})")
        return await self.set_status(goal_id, "active")

    async def delete(self, goal_id: str) -> bool:
        """Hard-delete a goal and everything attached to it.

        Returns False when the goal does not exist.
        """
        with self.db.transaction():
            row = self.db.query_one("SELECT goal_id FROM goals WHERE goal_id = ?", (goal_id,))
            if not row:
                return False
            self.db.execute("DELETE FROM goal_progress_history WHERE goal_id = ?", (goal_id,))
            self.db.execute("DELETE FROM goal_reminders WHERE goal_id = ?", (goal_id,))
            self.db.execute("DELETE FROM subgoals WHERE goal_id = ?", (goal_id,))
            self.db.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
        _log.info(f"Deleted goal {goal_id}")
        return True

    # ── subgoals ──────────────────────────────────────────────────────────

    async def add_subgoal(
        self,
        goal_id: str,
        title: str,
        *,
        order: int = 0,
    ) -> Subgoal:
        """Add a subgoal to a goal. Raises KeyError when the goal is missing."""
        await self._require(goal_id)
        title = (title or "").strip()
        if not title:
            raise ValueError("subgoal title must not be empty")
        subgoal_id = new_id("subgoal")

        with self.db.transaction():
            self.db.execute("""
                INSERT INTO subgoals (subgoal_id, goal_id, title, is_completed, sort_order)
                VALUES (?, ?, ?, 0, ?)
            """, (subgoal_id, goal_id, title, order))

        # Auto-recalculate progress
        await self._recalculate_progress(goal_id)

        return Subgoal(
            subgoal_id=subgoal_id,
            goal_id=goal_id,
            title=title,
            order=order,
        )

    async def complete_subgoal(self, subgoal_id: str) -> bool:
        """Mark a subgoal as completed. Returns False when it does not exist."""
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
        await self._recalculate_progress(row["goal_id"], note="subgoal completed")
        return True

    async def reopen_subgoal(self, subgoal_id: str) -> bool:
        """Re-open a completed subgoal. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT goal_id, is_completed FROM subgoals WHERE subgoal_id = ?",
                (subgoal_id,)
            )
            if not row:
                return False
            if not row["is_completed"]:
                return True
            self.db.execute("""
                UPDATE subgoals SET is_completed = 0, completed_at = NULL WHERE subgoal_id = ?
            """, (subgoal_id,))

        await self._recalculate_progress(row["goal_id"], note="subgoal re-opened")
        return True

    async def remove_subgoal(self, subgoal_id: str) -> bool:
        """Delete a subgoal. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT goal_id FROM subgoals WHERE subgoal_id = ?",
                (subgoal_id,)
            )
            if not row:
                return False
            self.db.execute("DELETE FROM subgoals WHERE subgoal_id = ?", (subgoal_id,))
        await self._recalculate_progress(row["goal_id"], note="subgoal removed")
        return True

    # ── progress ──────────────────────────────────────────────────────────

    async def update_progress(self, goal_id: str, progress: float, *, note: str = "") -> None:
        """Manually update goal progress and record it in the history.

        Raises KeyError when the goal does not exist.
        """
        await self._require(goal_id)
        progress = max(0.0, min(100.0, float(progress)))
        with self.db.transaction():
            self.db.execute("""
                UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?
            """, (progress, time.time(), goal_id))
            self._record_history(goal_id, progress, note or "progress updated")

    async def complete(self, goal_id: str) -> None:
        """Mark a goal as completed (subgoals cascade to completed).

        Raises KeyError when the goal does not exist.
        """
        await self._require(goal_id)
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                UPDATE subgoals SET is_completed = 1, completed_at = ?
                WHERE goal_id = ? AND is_completed = 0
            """, (now, goal_id))
            self.db.execute("""
                UPDATE goals SET status = 'completed', progress = 100.0, completed_at = ?, updated_at = ?
                WHERE goal_id = ?
            """, (now, now, goal_id))
            self._record_history(goal_id, 100.0, "goal completed")

    def _record_history(self, goal_id: str, progress: float, note: str) -> None:
        self.db.execute("""
            INSERT INTO goal_progress_history (history_id, goal_id, progress, note, recorded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (new_id("ghist"), goal_id, float(progress), note or "", time.time()))

    async def progress_history(self, goal_id: str, *, limit: int = 50) -> list[ProgressEntry]:
        """Progress events for a goal, newest first."""
        rows = self.db.query("""
            SELECT * FROM goal_progress_history WHERE goal_id = ?
            ORDER BY recorded_at DESC LIMIT ?
        """, (goal_id, max(1, limit)))
        return [
            ProgressEntry(
                history_id=r["history_id"],
                goal_id=r["goal_id"],
                progress=r["progress"],
                note=r["note"],
                recorded_at=r["recorded_at"],
            )
            for r in rows
        ]

    async def progress_streak(self, user_id: str) -> int:
        """Current streak: consecutive days (ending today) with any progress recorded."""
        rows = self.db.query("""
            SELECT DISTINCT date(recorded_at, 'unixepoch') AS day
            FROM goal_progress_history h
            JOIN goals g ON g.goal_id = h.goal_id
            WHERE g.user_id = ?
            ORDER BY day DESC
        """, (user_id,))
        if not rows:
            return 0
        import datetime as _dt
        days = [r["day"] for r in rows]
        today = _dt.date.today().isoformat()
        # Streak may start yesterday if nothing recorded yet today.
        cursor = _dt.date.today()
        if days[0] != today:
            cursor -= _dt.timedelta(days=1)
            if days[0] != cursor.isoformat():
                return 0
        streak = 0
        day_set = set(days)
        while cursor.isoformat() in day_set:
            streak += 1
            cursor -= _dt.timedelta(days=1)
        return streak

    # ── listing / filtering ───────────────────────────────────────────────

    async def list_goals(
        self,
        user_id: str,
        *,
        status: str | None = None,
        workspace: str | None = None,
        tags: list[str] | None = None,
        due_within_days: int | None = None,
        limit: int | None = None,
    ) -> list[Goal]:
        """List goals for a user with optional filters."""
        query = "SELECT * FROM goals WHERE user_id = ?"
        params: list[Any] = [user_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        if workspace:
            query += " AND workspace = ?"
            params.append(workspace)

        if due_within_days is not None:
            query += " AND target_date IS NOT NULL AND target_date <= ?"
            params.append(time.time() + due_within_days * 86400)

        query += " ORDER BY priority DESC, created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, limit))

        rows = self.db.query(query, params)
        goals = self._attach_subgoals(rows)

        if tags:
            wanted = {t.lower() for t in tags}
            goals = [g for g in goals if wanted & {t.lower() for t in g.tags}]

        return goals

    async def due_goals(
        self,
        user_id: str,
        *,
        within_days: int = 7,
        now: float | None = None,
    ) -> dict[str, list[Goal]]:
        """Active goals split into overdue and due-soon buckets."""
        now = time.time() if now is None else now
        goals = await self.list_goals(user_id, status="active")
        overdue = [g for g in goals if g.target_date is not None and g.target_date < now]
        horizon = now + within_days * 86400
        due_soon = [g for g in goals
                    if g.target_date is not None and now <= g.target_date <= horizon]
        due_soon.sort(key=lambda g: g.target_date or 0.0)
        return {"overdue": overdue, "due_soon": due_soon}

    async def stats(self, user_id: str) -> dict[str, Any]:
        """Aggregate counts for a user's goals."""
        rows = self.db.query("""
            SELECT status, COUNT(*) AS n, COALESCE(AVG(progress), 0) AS avg_progress
            FROM goals WHERE user_id = ? GROUP BY status
        """, (user_id,))
        by_status = {r["status"]: r["n"] for r in rows}
        total = sum(by_status.values())
        active_avg = next((r["avg_progress"] for r in rows if r["status"] == "active"), 0.0)
        due = await self.due_goals(user_id)
        return {
            "total": total,
            "by_status": {s: by_status.get(s, 0) for s in GOAL_STATUSES},
            "active_avg_progress": round(float(active_avg), 1),
            "overdue": len(due["overdue"]),
            "due_soon": len(due["due_soon"]),
        }

    # ── reminders ─────────────────────────────────────────────────────────

    async def add_reminder(
        self,
        goal_id: str,
        remind_at: float,
        message: str = "",
    ) -> Reminder:
        """Schedule a reminder for a goal. Raises KeyError when the goal is missing."""
        goal = await self._require(goal_id)
        reminder_id = new_id("remind")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_reminders (reminder_id, goal_id, user_id, remind_at, message, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (reminder_id, goal_id, goal.user_id, float(remind_at), message or "", now))
        return Reminder(
            reminder_id=reminder_id, goal_id=goal_id, user_id=goal.user_id,
            remind_at=float(remind_at), message=message or "", created_at=now)

    async def list_reminders(
        self,
        user_id: str,
        *,
        include_acknowledged: bool = False,
    ) -> list[Reminder]:
        """All reminders for a user, soonest first."""
        query = "SELECT * FROM goal_reminders WHERE user_id = ?"
        if not include_acknowledged:
            query += " AND acknowledged = 0"
        query += " ORDER BY remind_at ASC"
        rows = self.db.query(query, (user_id,))
        return [
            Reminder(
                reminder_id=r["reminder_id"],
                goal_id=r["goal_id"],
                user_id=r["user_id"],
                remind_at=r["remind_at"],
                message=r["message"],
                acknowledged=bool(r["acknowledged"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def due_reminders(
        self,
        user_id: str,
        *,
        now: float | None = None,
    ) -> list[Reminder]:
        """Unacknowledged reminders whose time has come."""
        now = time.time() if now is None else now
        return [r for r in await self.list_reminders(user_id) if r.remind_at <= now]

    async def ack_reminder(self, reminder_id: str) -> bool:
        """Acknowledge a reminder. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT reminder_id FROM goal_reminders WHERE reminder_id = ?",
                (reminder_id,))
            if not row:
                return False
            self.db.execute(
                "UPDATE goal_reminders SET acknowledged = 1 WHERE reminder_id = ?",
                (reminder_id,))
        return True

    # ── briefing ──────────────────────────────────────────────────────────

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
                days_left = (goal.target_date - time.time()) / 86400
                if days_left > 0:
                    lines.append(f"   ⏰ {days_left:.0f} days remaining")
                else:
                    lines.append(f"   ⛔ OVERDUE by {-days_left:.0f} days")

            lines.append("")

        # Deadline watch
        due = await self.due_goals(user_id)
        if due["overdue"]:
            lines.append("⛔ **Overdue:**")
            for g in due["overdue"]:
                days = int((time.time() - (g.target_date or time.time())) / 86400)
                lines.append(f"   • {g.title} ({days}d overdue)")
            lines.append("")
        if due["due_soon"]:
            lines.append("⏰ **Due soon:**")
            for g in due["due_soon"]:
                days = int(((g.target_date or time.time()) - time.time()) / 86400)
                lines.append(f"   • {g.title} (in {days}d)")
            lines.append("")

        # Reminders
        reminders = await self.due_reminders(user_id)
        if reminders:
            titles = {g.goal_id: g.title for g in goals}
            lines.append("🔔 **Reminders due:**")
            for r in reminders:
                lines.append(f"   • {titles.get(r.goal_id, r.goal_id)}"
                             + (f" — {r.message}" if r.message else ""))
            lines.append("")

        streak = await self.progress_streak(user_id)
        if streak:
            lines.append(f"🔥 Progress streak: {streak} day(s)")

        return "\n".join(lines).rstrip()

    async def _recalculate_progress(self, goal_id: str, *, note: str = "") -> None:
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
            self._record_history(goal_id, progress, note)


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

    def _row_to_idea(self, row: dict[str, Any]) -> Idea:
        return Idea(
            idea_id=row["idea_id"],
            user_id=row["user_id"],
            title=row["title"],
            description=row["description"],
            status=row["status"],
            created_at=row["created_at"],
            dismissed_at=row["dismissed_at"],
            dismiss_reason=row["dismiss_reason"],
            tags=json.loads(row["tags"] or "[]"),
            source=row["source"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def create(
        self,
        user_id: str,
        title: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Idea:
        """Create a new idea."""
        title = (title or "").strip()
        if not title:
            raise ValueError("idea title must not be empty")
        idea_id = new_id("idea")
        now = time.time()
        tags = list(tags or [])
        metadata = dict(metadata or {})

        with self.db.transaction():
            self.db.execute("""
                INSERT INTO ideas (idea_id, user_id, title, description, status, created_at, tags, source, metadata)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """, (idea_id, user_id, title, description, now, json.dumps(tags), source,
                  json.dumps(metadata)))

        return Idea(
            idea_id=idea_id,
            user_id=user_id,
            title=title,
            description=description,
            tags=tags,
            source=source,
            metadata=metadata,
        )

    async def get(self, idea_id: str) -> Idea | None:
        """Fetch a single idea by id."""
        row = self.db.query_one("SELECT * FROM ideas WHERE idea_id = ?", (idea_id,))
        return self._row_to_idea(row) if row else None

    async def _require(self, idea_id: str) -> Idea:
        idea = await self.get(idea_id)
        if idea is None:
            raise KeyError(f"no idea {idea_id!r}")
        return idea

    async def update(
        self,
        idea_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> Idea:
        """Update idea fields. Raises KeyError when the idea does not exist."""
        await self._require(idea_id)
        updates: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("idea title must not be empty")
            updates["title"] = title
        if description is not None:
            updates["description"] = description
        if tags is not None:
            updates["tags"] = json.dumps(list(tags))
        if source is not None:
            updates["source"] = source
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            with self.db.transaction():
                self.db.execute(
                    f"UPDATE ideas SET {set_clause} WHERE idea_id = ?",
                    (*updates.values(), idea_id))
        return await self._require(idea_id)

    async def dismiss(self, idea_id: str, *, reason: str = "") -> bool:
        """Dismiss an idea. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT idea_id FROM ideas WHERE idea_id = ?", (idea_id,))
            if not row:
                return False
            self.db.execute("""
                UPDATE ideas SET status = 'dismissed', dismissed_at = ?, dismiss_reason = ?
                WHERE idea_id = ?
            """, (time.time(), reason or "", idea_id))
        return True

    async def delete(self, idea_id: str) -> bool:
        """Hard-delete an idea. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT idea_id FROM ideas WHERE idea_id = ?", (idea_id,))
            if not row:
                return False
            self.db.execute("DELETE FROM ideas WHERE idea_id = ?", (idea_id,))
        return True

    async def promote_to_goal(self, idea_id: str, goal_tracker: GoalTracker) -> Optional[str]:
        """Promote an idea to a goal.

        Records the created goal id back onto the idea's metadata and
        returns it. Returns None when the idea does not exist.
        """
        idea = await self.get(idea_id)
        if idea is None:
            return None

        goal = await goal_tracker.create(
            user_id=idea.user_id,
            title=idea.title,
            description=idea.description,
            tags=list(idea.tags),
        )

        metadata = dict(idea.metadata)
        metadata["promoted_goal_id"] = goal.goal_id
        with self.db.transaction():
            self.db.execute("""
                UPDATE ideas SET status = 'promoted_to_goal', metadata = ? WHERE idea_id = ?
            """, (json.dumps(metadata), idea_id))

        return goal.goal_id

    async def list_ideas(
        self,
        user_id: str,
        *,
        status: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Idea]:
        """List ideas for a user with optional filters."""
        query = "SELECT * FROM ideas WHERE user_id = ?"
        params: list[Any] = [user_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, limit))

        rows = self.db.query(query, params)
        ideas = [self._row_to_idea(r) for r in rows]

        if tags:
            wanted = {t.lower() for t in tags}
            ideas = [i for i in ideas if wanted & {t.lower() for t in i.tags}]

        return ideas

    async def search(self, user_id: str, query: str, *, limit: int = 20) -> list[Idea]:
        """Full-text-ish search over idea titles and descriptions."""
        query = (query or "").strip()
        if not query:
            return []
        like = f"%{query}%"
        rows = self.db.query("""
            SELECT * FROM ideas
            WHERE user_id = ? AND (title LIKE ? OR description LIKE ?)
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, like, like, max(1, limit)))
        return [self._row_to_idea(r) for r in rows]
