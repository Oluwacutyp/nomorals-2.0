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
import math
import re
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
    "KeyResult",
    "JournalEntry",
    "GoalLink",
    "Milestone",
    "GoalTemplate",
    "GOAL_TEMPLATES",
    "GOAL_STATUSES",
    "IDEA_STATUSES",
    "GOAL_KINDS",
    "GOAL_LINK_KINDS",
    "REMINDER_RECURRENCES",
    "FOCUS_COEFFICIENTS",
]

_log = get_logger(__name__)


def _ensure_column(db: Database, table: str, column: str, ddl: str) -> None:
    """Add a column to an existing table when it is missing (idempotent)."""
    cols = {r["name"] for r in db.query(f"PRAGMA table_info({table})")}
    if column not in cols:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

GOAL_STATUSES = ("active", "completed", "paused", "abandoned", "archived")
IDEA_STATUSES = ("active", "dismissed", "promoted_to_goal")

#: Goal kinds (Habitica's habit/daily/todo trichotomy, mapped onto goals:
#: habits are cadence-based, outcomes are target-based, projects are
#: checklist-driven).
GOAL_KINDS = ("outcome", "habit", "project")

#: Directed goal-link kinds. ``blocks`` is directional (A blocks B);
#: ``relates`` is informational.
GOAL_LINK_KINDS = ("blocks", "relates")

#: Reminder recurrence modes (Loop Habit Tracker-style per-goal nudges).
REMINDER_RECURRENCES = ("once", "daily", "weekly")

#: Focus-score coefficients, adapted from Taskwarrior's urgency algorithm
#: (https://taskwarrior.org/docs/urgency/). ``next`` is the special tag that
#: dominates; ``due`` uses Taskwarrior's graded due formula; ``blocked``
#: penalizes goals that cannot move; ``stale`` is our own addition —
#: Beeminder's pessimistic-presumption insight that silence is signal.
FOCUS_COEFFICIENTS: dict[str, float] = {
    "next": 15.0,
    "due": 12.0,
    "blocking": 8.0,
    "priority_high": 6.0,
    "priority_medium": 3.9,
    "priority_low": 1.8,
    "started": 4.0,
    "stale": 3.0,
    "age": 2.0,
    "annotations": 1.0,
    "tags": 1.0,
    "waiting": -3.0,   # paused goals
    "blocked": -5.0,   # goals blocked by another active goal
}

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
    recurrence: str = "once"  # once, daily, weekly
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "goal_id": self.goal_id,
            "user_id": self.user_id,
            "remind_at": self.remind_at,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "recurrence": self.recurrence,
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
    kind: str = "outcome"  # outcome, habit, project

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
            "kind": self.kind,
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
    # ICE scores (Sean Ellis): 1-10 each, or None when unscored.
    impact: Optional[float] = None
    confidence: Optional[float] = None
    ease: Optional[float] = None

    @property
    def ice_score(self) -> Optional[float]:
        """ICE score = (Impact + Confidence + Ease) / 3, or None if unscored."""
        if self.impact is None or self.confidence is None or self.ease is None:
            return None
        return round((self.impact + self.confidence + self.ease) / 3.0, 2)

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
            "impact": self.impact,
            "confidence": self.confidence,
            "ease": self.ease,
            "ice_score": self.ice_score,
        }


@dataclass
class KeyResult:
    """One measurable key result on a goal (OKR style).

    A key result is a metric with a baseline and a target — an outcome, not
    an activity. Graded 0.0-1.0 Google-style.
    """

    key_result_id: str
    goal_id: str
    title: str
    unit: str = ""
    baseline: float = 0.0
    target: float = 100.0
    current: float = 0.0
    weight: float = 1.0
    direction: str = "increase"  # increase | decrease
    committed: bool = True  # committed vs aspirational (mark up front)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """0.0-1.0 grade: fraction of the baseline→target span achieved."""
        if self.direction == "decrease":
            span = self.baseline - self.target
            done = self.baseline - self.current
        else:
            span = self.target - self.baseline
            done = self.current - self.baseline
        if span == 0:
            return 1.0 if done >= 0 else 0.0
        return max(0.0, min(1.0, done / span))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_result_id": self.key_result_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "unit": self.unit,
            "baseline": self.baseline,
            "target": self.target,
            "current": self.current,
            "weight": self.weight,
            "direction": self.direction,
            "committed": self.committed,
            "score": round(self.score, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class JournalEntry:
    """A timestamped journal note attached to a goal."""

    entry_id: str
    goal_id: str
    user_id: str
    text: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "goal_id": self.goal_id,
            "user_id": self.user_id,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class GoalLink:
    """A directed link between two goals (``blocks`` / ``relates``)."""

    link_id: str
    from_goal_id: str
    to_goal_id: str
    kind: str = "relates"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "from_goal_id": self.from_goal_id,
            "to_goal_id": self.to_goal_id,
            "kind": self.kind,
            "created_at": self.created_at,
        }


@dataclass
class Milestone:
    """A dated checkpoint inside a goal (the "milestone KR" for phased work)."""

    milestone_id: str
    goal_id: str
    title: str
    target_date: Optional[float] = None
    is_completed: bool = False
    completed_at: Optional[float] = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone_id": self.milestone_id,
            "goal_id": self.goal_id,
            "title": self.title,
            "target_date": self.target_date,
            "is_completed": self.is_completed,
            "completed_at": self.completed_at,
            "order": self.order,
        }


@dataclass
class GoalTemplate:
    """A user-saved goal template (built-ins live in GOAL_TEMPLATES)."""

    template_id: str
    user_id: str  # "" = shared/global
    name: str
    title: str
    description: str = ""
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    subgoals: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "user_id": self.user_id,
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "tags": self.tags,
            "subgoals": self.subgoals,
            "created_at": self.created_at,
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

            # Sweep upgrade: OKR key results, journal, links, milestones, custom templates.
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_key_results (
                    key_result_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    unit TEXT NOT NULL DEFAULT '',
                    baseline REAL NOT NULL DEFAULT 0.0,
                    target REAL NOT NULL DEFAULT 100.0,
                    current REAL NOT NULL DEFAULT 0.0,
                    weight REAL NOT NULL DEFAULT 1.0,
                    direction TEXT NOT NULL DEFAULT 'increase',
                    committed INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_journal (
                    entry_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_links (
                    link_id TEXT PRIMARY KEY,
                    from_goal_id TEXT NOT NULL,
                    to_goal_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'relates',
                    created_at REAL NOT NULL,
                    UNIQUE (from_goal_id, to_goal_id, kind),
                    FOREIGN KEY (from_goal_id) REFERENCES goals(goal_id),
                    FOREIGN KEY (to_goal_id) REFERENCES goals(goal_id)
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_milestones (
                    milestone_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    target_date REAL,
                    is_completed INTEGER NOT NULL DEFAULT 0,
                    completed_at REAL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS goal_templates (
                    template_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]',
                    subgoals TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    UNIQUE (user_id, name)
                )
            """)
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_kr_goal ON goal_key_results(goal_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_journal_goal ON goal_journal(goal_id, created_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_links_from ON goal_links(from_goal_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_links_to ON goal_links(to_goal_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_milestones_goal ON goal_milestones(goal_id)")

            # Column migrations for databases created before this sweep.
            _ensure_column(self.db, "goals", "kind", "TEXT NOT NULL DEFAULT 'outcome'")
            _ensure_column(self.db, "goal_reminders", "recurrence", "TEXT NOT NULL DEFAULT 'once'")
            tables = {r["name"] for r in
                      self.db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "ideas" in tables:
                _ensure_column(self.db, "ideas", "impact", "REAL")
                _ensure_column(self.db, "ideas", "confidence", "REAL")
                _ensure_column(self.db, "ideas", "ease", "REAL")

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
            kind=row.get("kind") or "outcome",
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
        kind: str = "outcome",
    ) -> Goal:
        """Create a new goal."""
        title = (title or "").strip()
        if not title:
            raise ValueError("goal title must not be empty")
        if kind not in GOAL_KINDS:
            raise ValueError(f"unknown goal kind {kind!r} (expected one of {GOAL_KINDS})")
        goal_id = new_id("goal")
        now = time.time()
        tags = list(tags or [])

        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goals (goal_id, user_id, title, description, status, progress, priority,
                    target_date, created_at, updated_at, tags, notes, workspace, kind)
                VALUES (?, ?, ?, ?, 'active', 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (goal_id, user_id, title, description, priority, target_date, now, now,
                  json.dumps(tags), notes, workspace, kind))
            self._record_history(goal_id, 0.0, "goal created")

        _log.info(f"Created goal: {title}")
        return await self.get(goal_id) or Goal(
            goal_id=goal_id, user_id=user_id, title=title, description=description,
            target_date=target_date, priority=priority, tags=tags,
            notes=notes, workspace=workspace, kind=kind)

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
        template = self._resolve_template(user_id, template_name)
        if template is None:
            custom = [t.name for t in await self.list_custom_templates(user_id)]
            available = sorted(set(GOAL_TEMPLATES) | set(custom))
            raise KeyError(
                f"unknown goal template {template_name!r} "
                f"(available: {', '.join(available)})")
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
        kind: str | None = None,
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
        if kind is not None:
            if kind not in GOAL_KINDS:
                raise ValueError(f"unknown goal kind {kind!r} (expected one of {GOAL_KINDS})")
            updates["kind"] = kind
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
        completed = by_status.get("completed", 0)
        comp_rows = self.db.query("""
            SELECT created_at, completed_at FROM goals
            WHERE user_id = ? AND status = 'completed' AND completed_at IS NOT NULL
        """, (user_id,))
        avg_days = (sum((r["completed_at"] - r["created_at"]) for r in comp_rows)
                    / len(comp_rows) / 86400.0) if comp_rows else None
        return {
            "total": total,
            "by_status": {s: by_status.get(s, 0) for s in GOAL_STATUSES},
            "active_avg_progress": round(float(active_avg), 1),
            "overdue": len(due["overdue"]),
            "due_soon": len(due["due_soon"]),
            "completion_rate": round(completed / total, 3) if total else 0.0,
            "avg_completion_days": None if avg_days is None else round(avg_days, 1),
            "longest_streak": await self.longest_streak(user_id),
            "momentum": await self.momentum(user_id),
        }

    # ── reminders ─────────────────────────────────────────────────────────

    async def add_reminder(
        self,
        goal_id: str,
        remind_at: float,
        message: str = "",
        *,
        recurrence: str = "once",
    ) -> Reminder:
        """Schedule a reminder for a goal. Raises KeyError when the goal is missing.

        ``recurrence`` is ``once`` | ``daily`` | ``weekly`` — recurring
        reminders reschedule themselves on acknowledge instead of dying.
        """
        if recurrence not in REMINDER_RECURRENCES:
            raise ValueError(
                f"unknown recurrence {recurrence!r} (expected one of {REMINDER_RECURRENCES})")
        goal = await self._require(goal_id)
        reminder_id = new_id("remind")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_reminders
                    (reminder_id, goal_id, user_id, remind_at, message, recurrence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (reminder_id, goal_id, goal.user_id, float(remind_at), message or "",
                  recurrence, now))
        return Reminder(
            reminder_id=reminder_id, goal_id=goal_id, user_id=goal.user_id,
            remind_at=float(remind_at), message=message or "",
            recurrence=recurrence, created_at=now)

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
                recurrence=r.get("recurrence") or "once",
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

    async def ack_reminder(self, reminder_id: str, *, now: float | None = None) -> bool:
        """Acknowledge a reminder. Returns False when it does not exist.

        Recurring reminders (daily/weekly) are not killed — they roll forward
        to their next occurrence.
        """
        now = time.time() if now is None else now
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT reminder_id, remind_at, recurrence FROM goal_reminders WHERE reminder_id = ?",
                (reminder_id,))
            if not row:
                return False
            recurrence = row["recurrence"] or "once"
            if recurrence in ("daily", "weekly"):
                step = 86400.0 if recurrence == "daily" else 7 * 86400.0
                nxt = float(row["remind_at"])
                while nxt <= now:
                    nxt += step
                self.db.execute(
                    "UPDATE goal_reminders SET remind_at = ? WHERE reminder_id = ?",
                    (nxt, reminder_id))
            else:
                self.db.execute(
                    "UPDATE goal_reminders SET acknowledged = 1 WHERE reminder_id = ?",
                    (reminder_id,))
        return True

    async def snooze_reminder(self, reminder_id: str, minutes: float,
                             *, now: float | None = None) -> bool:
        """Push a reminder ``minutes`` into the future. Returns False when missing."""
        if minutes <= 0:
            raise ValueError("snooze minutes must be positive")
        now = time.time() if now is None else now
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT remind_at FROM goal_reminders WHERE reminder_id = ?",
                (reminder_id,))
            if not row:
                return False
            new_at = max(float(row["remind_at"]), now) + minutes * 60.0
            self.db.execute(
                "UPDATE goal_reminders SET remind_at = ? WHERE reminder_id = ?",
                (new_at, reminder_id))
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

        # Focus: what deserves attention right now (Taskwarrior-style urgency).
        focus = await self.today_focus(user_id, limit=3)
        if focus:
            lines.append("")
            lines.append("🎯 **Focus now:**")
            for i, entry in enumerate(focus, 1):
                g = entry["goal"]
                lines.append(f"   {i}. {g.title} — {g.progress:.0f}% (score {entry['score']:.1f})")

        # Pace check for dated goals (Beeminder-style verdicts).
        verdict_label = {
            "on_track": "on track ✅", "at_risk": "at risk ⚠️",
            "off_track": "off track 🔻", "overdue": "overdue ⛔",
            "no_data": "no data yet", "no_target": "no target",
            "complete": "complete 🎉",
        }
        pace_lines = []
        for goal in goals:
            if goal.target_date is None or goal.status != "active":
                continue
            report = await self.pace_report(goal.goal_id)
            if report["verdict"] in ("on_track", "complete"):
                continue
            req = report["required_daily_pace"]
            act = report["actual_daily_pace"]
            detail = ""
            if req is not None and act is not None:
                detail = f" — needs {req:.1f}%/day, doing {act:.1f}%/day"
            elif req is not None:
                detail = f" — needs {req:.1f}%/day"
            pace_lines.append(
                f"   • {goal.title}: {verdict_label[report['verdict']]}{detail}")
        if pace_lines:
            lines.append("")
            lines.append("📈 **Pace check:**")
            lines.extend(pace_lines)

        # Quiet goals (GTD: every project needs a next action).
        quiet = await self.stale_goals(user_id, days=7)
        if quiet:
            lines.append("")
            lines.append(f"💤 **Gone quiet:** {len(quiet)} goal(s) with no activity in 7+ days")

        return "\n".join(lines).rstrip()

    async def _recalculate_progress(self, goal_id: str, *, note: str = "") -> None:
        """Recalculate goal progress.

        Precedence: key results (OKR roll-up) → subgoals (completion ratio) →
        manual (left untouched when neither exists).
        """
        krs = await self.list_key_results(goal_id)
        if krs:
            progress = (await self.okr_score(goal_id)) * 100.0
            with self.db.transaction():
                self.db.execute("""
                    UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?
                """, (progress, time.time(), goal_id))
                self._record_history(goal_id, progress, note or "key results updated")
            return

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

    async def rename_subgoal(self, subgoal_id: str, title: str) -> bool:
        """Rename a subgoal. Returns False when it does not exist."""
        title = (title or "").strip()
        if not title:
            raise ValueError("subgoal title must not be empty")
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT subgoal_id FROM subgoals WHERE subgoal_id = ?", (subgoal_id,))
            if not row:
                return False
            self.db.execute("UPDATE subgoals SET title = ? WHERE subgoal_id = ?",
                            (title, subgoal_id))
        return True

    async def reorder_subgoals(self, goal_id: str, ordered_ids: list[str]) -> bool:
        """Reorder a goal's subgoals (Habitica-style drag-and-drop).

        ``ordered_ids`` must contain exactly the goal's subgoal ids in the
        desired order. Returns False when the goal does not exist.
        """
        goal = await self.get(goal_id)
        if goal is None:
            return False
        current = {s.subgoal_id for s in goal.subgoals}
        if set(ordered_ids) != current or len(ordered_ids) != len(current):
            raise ValueError("ordered_ids must list exactly the goal's subgoal ids")
        with self.db.transaction():
            for order, sid in enumerate(ordered_ids):
                self.db.execute("UPDATE subgoals SET sort_order = ? WHERE subgoal_id = ?",
                                (order, sid))
        return True
    # ── key results (OKR) ───────────────────────────────────────────────

    def _row_to_key_result(self, r: dict[str, Any]) -> KeyResult:
        return KeyResult(
            key_result_id=r["key_result_id"],
            goal_id=r["goal_id"],
            title=r["title"],
            unit=r["unit"],
            baseline=r["baseline"],
            target=r["target"],
            current=r["current"],
            weight=r["weight"],
            direction=r["direction"],
            committed=bool(r["committed"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    async def add_key_result(
        self,
        goal_id: str,
        title: str,
        target: float,
        *,
        baseline: float = 0.0,
        current: float | None = None,
        unit: str = "",
        weight: float = 1.0,
        direction: str = "increase",
        committed: bool = True,
    ) -> KeyResult:
        """Add a measurable key result to a goal (OKR style).

        A key result tracks baseline → target (an outcome, not an activity).
        When a goal has key results, its progress rolls up from them.
        """
        await self._require(goal_id)
        title = (title or "").strip()
        if not title:
            raise ValueError("key result title must not be empty")
        if direction not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")
        if weight <= 0:
            raise ValueError("weight must be positive")
        kr_id = new_id("kr")
        now = time.time()
        current = baseline if current is None else float(current)
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_key_results
                    (key_result_id, goal_id, title, unit, baseline, target, current,
                     weight, direction, committed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (kr_id, goal_id, title, unit, float(baseline), float(target),
                  current, float(weight), direction, int(committed), now, now))
        await self._recalculate_progress(goal_id, note="key result added")
        row = self.db.query_one(
            "SELECT * FROM goal_key_results WHERE key_result_id = ?", (kr_id,))
        assert row is not None
        return self._row_to_key_result(row)

    async def update_key_result(
        self, key_result_id: str, current: float, *, note: str = ""
    ) -> KeyResult:
        """Record a new measured value for a key result (rolls up to the goal)."""
        row = self.db.query_one(
            "SELECT * FROM goal_key_results WHERE key_result_id = ?", (key_result_id,))
        if not row:
            raise KeyError(f"no key result {key_result_id!r}")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                UPDATE goal_key_results SET current = ?, updated_at = ?
                WHERE key_result_id = ?
            """, (float(current), now, key_result_id))
        await self._recalculate_progress(
            row["goal_id"], note=note or f"key result updated: {row['title']}")
        updated = self.db.query_one(
            "SELECT * FROM goal_key_results WHERE key_result_id = ?", (key_result_id,))
        assert updated is not None
        return self._row_to_key_result(updated)

    async def remove_key_result(self, key_result_id: str) -> bool:
        """Delete a key result. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT goal_id FROM goal_key_results WHERE key_result_id = ?",
                (key_result_id,))
            if not row:
                return False
            self.db.execute("DELETE FROM goal_key_results WHERE key_result_id = ?",
                            (key_result_id,))
        await self._recalculate_progress(row["goal_id"], note="key result removed")
        return True

    async def list_key_results(self, goal_id: str) -> list[KeyResult]:
        """Key results for a goal, in creation order."""
        rows = self.db.query(
            "SELECT * FROM goal_key_results WHERE goal_id = ? ORDER BY created_at ASC",
            (goal_id,))
        return [self._row_to_key_result(r) for r in rows]

    async def okr_score(self, goal_id: str) -> float:
        """Weight-averaged 0.0-1.0 grade across the goal's key results."""
        krs = await self.list_key_results(goal_id)
        if not krs:
            return 0.0
        total_weight = sum(kr.weight for kr in krs)
        if total_weight <= 0:
            return 0.0
        return sum(kr.score * kr.weight for kr in krs) / total_weight

    # ── journal ───────────────────────────────────────────────────────────

    async def add_journal_entry(self, goal_id: str, text: str) -> JournalEntry:
        """Append a timestamped journal note to a goal."""
        goal = await self._require(goal_id)
        text = (text or "").strip()
        if not text:
            raise ValueError("journal entry text must not be empty")
        entry_id = new_id("gjournal")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_journal (entry_id, goal_id, user_id, text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (entry_id, goal_id, goal.user_id, text, now))
        return JournalEntry(entry_id=entry_id, goal_id=goal_id,
                            user_id=goal.user_id, text=text, created_at=now)

    async def journal(self, goal_id: str, *, limit: int = 50) -> list[JournalEntry]:
        """Journal entries for a goal, newest first."""
        rows = self.db.query("""
            SELECT * FROM goal_journal WHERE goal_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (goal_id, max(1, limit)))
        return [
            JournalEntry(entry_id=r["entry_id"], goal_id=r["goal_id"],
                         user_id=r["user_id"], text=r["text"],
                         created_at=r["created_at"])
            for r in rows
        ]

    # ── goal links / dependencies ─────────────────────────────────────────

    def _row_to_link(self, r: dict[str, Any]) -> GoalLink:
        return GoalLink(
            link_id=r["link_id"],
            from_goal_id=r["from_goal_id"],
            to_goal_id=r["to_goal_id"],
            kind=r["kind"],
            created_at=r["created_at"],
        )

    async def link_goals(
        self, from_goal_id: str, to_goal_id: str, kind: str = "relates"
    ) -> GoalLink:
        """Link two goals. ``blocks`` is directional: from blocks to."""
        if kind not in GOAL_LINK_KINDS:
            raise ValueError(f"unknown link kind {kind!r} (expected one of {GOAL_LINK_KINDS})")
        if from_goal_id == to_goal_id:
            raise ValueError("a goal cannot link to itself")
        await self._require(from_goal_id)
        await self._require(to_goal_id)
        link_id = new_id("glink")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                INSERT OR IGNORE INTO goal_links
                    (link_id, from_goal_id, to_goal_id, kind, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (link_id, from_goal_id, to_goal_id, kind, now))
            row = self.db.query_one("""
                SELECT * FROM goal_links
                WHERE from_goal_id = ? AND to_goal_id = ? AND kind = ?
            """, (from_goal_id, to_goal_id, kind))
        assert row is not None
        return self._row_to_link(row)

    async def unlink_goals(
        self, from_goal_id: str, to_goal_id: str, kind: str = "relates"
    ) -> bool:
        """Remove a goal link. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one("""
                SELECT link_id FROM goal_links
                WHERE from_goal_id = ? AND to_goal_id = ? AND kind = ?
            """, (from_goal_id, to_goal_id, kind))
            if not row:
                return False
            self.db.execute("DELETE FROM goal_links WHERE link_id = ?",
                            (row["link_id"],))
        return True

    async def goal_links(self, goal_id: str) -> list[GoalLink]:
        """All links touching a goal (either direction)."""
        rows = self.db.query("""
            SELECT * FROM goal_links
            WHERE from_goal_id = ? OR to_goal_id = ?
            ORDER BY created_at ASC
        """, (goal_id, goal_id))
        return [self._row_to_link(r) for r in rows]

    async def blocking_goals(self, goal_id: str) -> list[Goal]:
        """Active goals that ``goal_id`` blocks (outgoing ``blocks`` links)."""
        rows = self.db.query("""
            SELECT g.* FROM goals g
            JOIN goal_links l ON l.to_goal_id = g.goal_id
            WHERE l.from_goal_id = ? AND l.kind = 'blocks' AND g.status = 'active'
        """, (goal_id,))
        return self._attach_subgoals(rows)

    async def blocked_by_goals(self, goal_id: str) -> list[Goal]:
        """Active goals blocking ``goal_id`` (incoming ``blocks`` links)."""
        rows = self.db.query("""
            SELECT g.* FROM goals g
            JOIN goal_links l ON l.from_goal_id = g.goal_id
            WHERE l.to_goal_id = ? AND l.kind = 'blocks' AND g.status = 'active'
        """, (goal_id,))
        return self._attach_subgoals(rows)

    # ── milestones ────────────────────────────────────────────────────────

    def _row_to_milestone(self, r: dict[str, Any]) -> Milestone:
        return Milestone(
            milestone_id=r["milestone_id"],
            goal_id=r["goal_id"],
            title=r["title"],
            target_date=r["target_date"],
            is_completed=bool(r["is_completed"]),
            completed_at=r["completed_at"],
            order=r["sort_order"],
        )

    async def add_milestone(
        self,
        goal_id: str,
        title: str,
        *,
        target_date: float | None = None,
        order: int = 0,
    ) -> Milestone:
        """Add a dated checkpoint inside a goal."""
        await self._require(goal_id)
        title = (title or "").strip()
        if not title:
            raise ValueError("milestone title must not be empty")
        milestone_id = new_id("gmile")
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_milestones
                    (milestone_id, goal_id, title, target_date, sort_order)
                VALUES (?, ?, ?, ?, ?)
            """, (milestone_id, goal_id, title, target_date, order))
        row = self.db.query_one(
            "SELECT * FROM goal_milestones WHERE milestone_id = ?", (milestone_id,))
        assert row is not None
        return self._row_to_milestone(row)

    async def complete_milestone(self, milestone_id: str) -> bool:
        """Mark a milestone completed. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT milestone_id FROM goal_milestones WHERE milestone_id = ?",
                (milestone_id,))
            if not row:
                return False
            self.db.execute("""
                UPDATE goal_milestones SET is_completed = 1, completed_at = ?
                WHERE milestone_id = ?
            """, (time.time(), milestone_id))
        return True

    async def reopen_milestone(self, milestone_id: str) -> bool:
        """Re-open a completed milestone. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT milestone_id FROM goal_milestones WHERE milestone_id = ?",
                (milestone_id,))
            if not row:
                return False
            self.db.execute("""
                UPDATE goal_milestones SET is_completed = 0, completed_at = NULL
                WHERE milestone_id = ?
            """, (milestone_id,))
        return True

    async def remove_milestone(self, milestone_id: str) -> bool:
        """Delete a milestone. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT milestone_id FROM goal_milestones WHERE milestone_id = ?",
                (milestone_id,))
            if not row:
                return False
            self.db.execute("DELETE FROM goal_milestones WHERE milestone_id = ?",
                            (milestone_id,))
        return True

    async def list_milestones(self, goal_id: str) -> list[Milestone]:
        """Milestones for a goal, in order."""
        rows = self.db.query("""
            SELECT * FROM goal_milestones WHERE goal_id = ?
            ORDER BY sort_order ASC, milestone_id ASC
        """, (goal_id,))
        return [self._row_to_milestone(r) for r in rows]

    # ── focus scoring (Taskwarrior urgency, adapted) ──────────────────────

    @staticmethod
    def _due_value(target_date: float | None, now: float) -> float:
        """Taskwarrior's graded due term: 1.0 when ≥7d overdue, 0.2 when far out."""
        if target_date is None:
            return 0.0
        days_overdue = (now - target_date) / 86400.0
        if days_overdue >= 7:
            return 1.0
        if days_overdue >= -14:
            return ((days_overdue + 14.0) * 0.8 / 21.0) + 0.2
        return 0.2

    @staticmethod
    def _graded_count(n: int) -> float:
        """Taskwarrior's 0 / 0.8 / 0.9 / 1.0 count grading (tags, annotations)."""
        if n <= 0:
            return 0.0
        if n == 1:
            return 0.8
        if n == 2:
            return 0.9
        return 1.0

    def _last_activity(self, goal: Goal) -> float:
        row = self.db.query_one(
            "SELECT MAX(recorded_at) AS m FROM goal_progress_history WHERE goal_id = ?",
            (goal.goal_id,))
        last = row["m"] if row and row["m"] else None
        return last or goal.created_at

    def focus_breakdown(self, goal: Goal, *, now: float | None = None) -> dict[str, float]:
        """Per-term focus contributions for a goal (Taskwarrior-style urgency).

        Returns the nonzero term contributions plus ``"total"``.
        """
        now = time.time() if now is None else now
        c = FOCUS_COEFFICIENTS
        terms: dict[str, float] = {}

        tags = {t.lower() for t in goal.tags}
        if "next" in tags:
            terms["next"] = c["next"]
        due_v = self._due_value(goal.target_date, now)
        if due_v:
            terms["due"] = round(c["due"] * due_v, 2)
        if goal.priority >= 3:
            terms["priority"] = c["priority_high"]
        elif goal.priority == 2:
            terms["priority"] = c["priority_medium"]
        elif goal.priority == 1:
            terms["priority"] = c["priority_low"]
        if goal.progress > 0:
            terms["started"] = c["started"]
        age_days = max(0.0, (now - goal.created_at) / 86400.0)
        age_v = min(1.0, age_days / 365.0)
        if age_v:
            terms["age"] = round(c["age"] * age_v, 2)
        stale_days = (now - self._last_activity(goal)) / 86400.0
        if stale_days > 7:
            terms["stale"] = round(c["stale"] * min(1.0, stale_days / 30.0), 2)
        open_subgoals = sum(1 for s in goal.subgoals if not s.is_completed)
        ann_v = self._graded_count(open_subgoals)
        if ann_v:
            terms["checklist"] = round(c["annotations"] * ann_v, 2)
        tag_v = self._graded_count(len(goal.tags))
        if tag_v:
            terms["tags"] = round(c["tags"] * tag_v, 2)
        if goal.status == "paused":
            terms["waiting"] = c["waiting"]

        active_blocking = self.db.query_one("""
            SELECT 1 AS x FROM goal_links l JOIN goals g ON g.goal_id = l.to_goal_id
            WHERE l.from_goal_id = ? AND l.kind = 'blocks' AND g.status = 'active' LIMIT 1
        """, (goal.goal_id,))
        if active_blocking:
            terms["blocking"] = c["blocking"]
        active_blocked = self.db.query_one("""
            SELECT 1 AS x FROM goal_links l JOIN goals g ON g.goal_id = l.from_goal_id
            WHERE l.to_goal_id = ? AND l.kind = 'blocks' AND g.status = 'active' LIMIT 1
        """, (goal.goal_id,))
        if active_blocked:
            terms["blocked"] = c["blocked"]

        terms["total"] = round(sum(terms.values()), 2)
        return terms

    def focus_score(self, goal: Goal, *, now: float | None = None) -> float:
        """Single urgency number for a goal — higher means "work on this now"."""
        return self.focus_breakdown(goal, now=now)["total"]

    async def today_focus(
        self, user_id: str, *, limit: int = 5, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Top-N active goals by focus score, each with its term breakdown."""
        now = time.time() if now is None else now
        goals = await self.list_goals(user_id, status="active")
        ranked = [
            {"goal": g, "score": self.focus_score(g, now=now),
             "breakdown": self.focus_breakdown(g, now=now)}
            for g in goals
        ]
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:max(1, limit)]

    # ── pace / forecast (Beeminder-style) ─────────────────────────────────

    async def pace_report(self, goal_id: str, *, now: float | None = None) -> dict[str, Any]:
        """Pace analysis: required vs actual daily pace, projection, verdict.

        Verdicts: complete | no_target | overdue | no_data | on_track |
        at_risk | off_track. ``safe_days`` is the Beeminder-style buffer —
        how many days you can coast before the deadline slips.
        """
        goal = await self._require(goal_id)
        now = time.time() if now is None else now
        history = await self.progress_history(goal_id, limit=500)
        oldest_first = sorted(history, key=lambda h: h.recorded_at)

        actual: float | None = None
        if len(oldest_first) >= 2:
            first, last = oldest_first[0], oldest_first[-1]
            span_days = (last.recorded_at - first.recorded_at) / 86400.0
            if span_days >= 1.0:
                actual = (last.progress - first.progress) / span_days

        days_left: float | None = None
        required: float | None = None
        if goal.target_date is not None:
            days_left = (goal.target_date - now) / 86400.0
            if days_left > 0:
                required = (100.0 - goal.progress) / days_left

        projected_date: float | None = None
        safe_days = 0.0
        if actual is not None and actual > 0 and goal.progress < 100:
            days_needed = (100.0 - goal.progress) / actual
            projected_date = now + days_needed * 86400.0
            if days_left is not None:
                safe_days = max(0.0, days_left - days_needed)

        if goal.status == "completed" or goal.progress >= 100:
            verdict = "complete"
        elif goal.target_date is None:
            verdict = "no_target"
        elif days_left is not None and days_left < 0:
            verdict = "overdue"
        elif actual is None:
            verdict = "no_data"
        elif actual <= 0:
            verdict = "off_track"  # flatline: Beeminder derails on silence
        elif projected_date is not None and projected_date <= goal.target_date:
            verdict = "on_track"
        else:
            total_span = max(1.0, (goal.target_date or now) - goal.created_at)
            slack = 0.3 * total_span
            if projected_date is not None and projected_date <= (goal.target_date or 0) + slack:
                verdict = "at_risk"
            else:
                verdict = "off_track"

        return {
            "goal_id": goal.goal_id,
            "title": goal.title,
            "progress": goal.progress,
            "target_date": goal.target_date,
            "days_left": None if days_left is None else round(days_left, 1),
            "required_daily_pace": None if required is None else round(required, 2),
            "actual_daily_pace": None if actual is None else round(actual, 2),
            "projected_completion_date": projected_date,
            "safe_days": round(safe_days, 1),
            "verdict": verdict,
        }

    async def check_in(
        self, goal_id: str, *, note: str = "", progress: float | None = None
    ) -> ProgressEntry:
        """Record a lightweight check-in datapoint.

        Without ``progress`` this logs activity at the current progress value —
        it keeps the streak/momentum alive without moving the number (the
        Beeminder "enter a 0" for do-less goals). With ``progress`` it also
        updates the goal.
        """
        goal = await self._require(goal_id)
        value = goal.progress if progress is None else max(0.0, min(100.0, float(progress)))
        entry_id = new_id("ghist")
        now = time.time()
        text = note or "check-in"
        with self.db.transaction():
            if progress is not None and value != goal.progress:
                self.db.execute(
                    "UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?",
                    (value, now, goal_id))
            self.db.execute("""
                INSERT INTO goal_progress_history
                    (history_id, goal_id, progress, note, recorded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (entry_id, goal_id, value, text, now))
        return ProgressEntry(history_id=entry_id, goal_id=goal_id,
                             progress=value, note=text, recorded_at=now)

    # ── momentum (Loop Habit Tracker exponential smoothing) ──────────────

    async def momentum(self, user_id: str, *, days: int = 60, now: float | None = None) -> float:
        """Habit-strength 0.0-1.0 via exponential smoothing (uhabits' formula).

        Every day with any recorded progress counts as completed; recent days
        weigh more than old ones, so one miss after a long run barely dents
        the score while frequent misses sink it.
        """
        now = time.time() if now is None else now
        import datetime as _dt
        days = max(1, days)
        start = now - days * 86400
        rows = self.db.query("""
            SELECT DISTINCT date(h.recorded_at, 'unixepoch') AS day
            FROM goal_progress_history h
            JOIN goals g ON g.goal_id = h.goal_id
            WHERE g.user_id = ? AND h.recorded_at >= ?
        """, (user_id, start))
        done = {r["day"] for r in rows}
        multiplier = 0.5 ** (math.sqrt(1.0) / 13.0)  # uhabits Score, daily frequency
        score = 0.0
        base = _dt.datetime.fromtimestamp(start, tz=_dt.timezone.utc).date()
        for i in range(days):
            day = (base + _dt.timedelta(days=i)).isoformat()
            score = score * multiplier + (1.0 if day in done else 0.0) * (1 - multiplier)
        return round(score, 3)

    async def longest_streak(self, user_id: str) -> int:
        """Longest run of consecutive days with any recorded progress."""
        rows = self.db.query("""
            SELECT DISTINCT date(h.recorded_at, 'unixepoch') AS day
            FROM goal_progress_history h
            JOIN goals g ON g.goal_id = h.goal_id
            WHERE g.user_id = ?
            ORDER BY day ASC
        """, (user_id,))
        import datetime as _dt
        best = run = 0
        prev: _dt.date | None = None
        for r in rows:
            day = _dt.date.fromisoformat(r["day"])
            if prev is not None and (day - prev).days == 1:
                run += 1
            else:
                run = 1
            best = max(best, run)
            prev = day
        return best

    # ── stale goals / weekly review (GTD) ─────────────────────────────────

    async def stale_goals(
        self, user_id: str, *, days: int = 7, now: float | None = None
    ) -> list[Goal]:
        """Active goals with no recorded activity in ``days`` days.

        GTD's "every project needs a next action" smell test.
        """
        now = time.time() if now is None else now
        cutoff = now - days * 86400
        goals = await self.list_goals(user_id, status="active")
        stale = [g for g in goals if self._last_activity(g) < cutoff]
        stale.sort(key=lambda g: self._last_activity(g))
        return stale

    async def weekly_review(self, user_id: str, *, now: float | None = None) -> dict[str, Any]:
        """GTD-style weekly review: get clear / get current / get creative.

        Returns completed-this-week, overdue, due-soon, stale (quiet) goals,
        blocked goals, ideas awaiting review (the someday/maybe list), and
        concrete suggested actions.
        """
        now = time.time() if now is None else now
        week_ago = now - 7 * 86400

        completed = [g for g in await self.list_goals(user_id, status="completed")
                     if (g.completed_at or 0) >= week_ago]
        due = await self.due_goals(user_id, within_days=7, now=now)
        stale = await self.stale_goals(user_id, days=7, now=now)
        active = await self.list_goals(user_id, status="active")
        blocked: list[dict[str, Any]] = []
        for g in active:
            blockers = await self.blocked_by_goals(g.goal_id)
            if blockers:
                blocked.append({"goal": g, "blocked_by": blockers})

        ideas = IdeaTracker(self.db)
        review_ideas = await ideas.review_queue(user_id, limit=10)

        actions: list[str] = []
        for g in due["overdue"]:
            days = int((now - (g.target_date or now)) / 86400)
            actions.append(
                f"'{g.title}' is {days}d overdue — renegotiate the date or shrink the scope.")
        for entry in blocked:
            names = ", ".join(b.title for b in entry["blocked_by"])
            actions.append(f"'{entry['goal'].title}' is blocked by {names} — unblock or re-plan.")
        for g in stale:
            quiet = int((now - self._last_activity(g)) / 86400)
            actions.append(
                f"'{g.title}' went quiet {quiet}d ago — what's the next action?")
        for idea, waiting in review_ideas:
            actions.append(
                f"Idea '{idea.title}' has waited {waiting}d — promote it, schedule it, or dismiss it.")

        return {
            "completed_this_week": completed,
            "overdue": due["overdue"],
            "due_soon": due["due_soon"],
            "stale": stale,
            "blocked": blocked,
            "ideas_for_review": review_ideas,
            "suggested_actions": actions,
        }

    def format_weekly_review(self, review: dict[str, Any]) -> str:
        """Render a ``weekly_review()`` dict as markdown."""
        lines = ["📋 **Weekly Review**\n"]
        done = review["completed_this_week"]
        lines.append(f"✅ **Completed this week:** {len(done)}")
        for g in done:
            lines.append(f"   • {g.title}")
        lines.append("")
        if review["overdue"]:
            lines.append("⛔ **Overdue:**")
            for g in review["overdue"]:
                lines.append(f"   • {g.title}")
            lines.append("")
        if review["due_soon"]:
            lines.append("⏰ **Due soon:**")
            for g in review["due_soon"]:
                lines.append(f"   • {g.title}")
            lines.append("")
        if review["stale"]:
            lines.append("💤 **Gone quiet (no activity in 7+ days):**")
            for g in review["stale"]:
                lines.append(f"   • {g.title} — {g.progress:.0f}%")
            lines.append("")
        if review["blocked"]:
            lines.append("🧱 **Blocked:**")
            for entry in review["blocked"]:
                names = ", ".join(b.title for b in entry["blocked_by"])
                lines.append(f"   • {entry['goal'].title} (blocked by {names})")
            lines.append("")
        if review["ideas_for_review"]:
            lines.append("💡 **Ideas awaiting review:**")
            for idea, waiting in review["ideas_for_review"]:
                lines.append(f"   • {idea.title} ({waiting}d parked)")
            lines.append("")
        if review["suggested_actions"]:
            lines.append("🎯 **Suggested actions:**")
            for i, action in enumerate(review["suggested_actions"], 1):
                lines.append(f"   {i}. {action}")
        else:
            lines.append("🎯 Nothing needs attention. Stay the course.")
        return "\n".join(lines).rstrip()

    # ── goal search ───────────────────────────────────────────────────────

    async def search_goals(
        self, user_id: str, query: str, *, limit: int = 20,
        include_archived: bool = False,
    ) -> list[Goal]:
        """Text search over goal titles, descriptions and notes."""
        query = (query or "").strip()
        if not query:
            return []
        like = f"%{query}%"
        archived = "" if include_archived else "AND status != 'archived'"
        rows = self.db.query(f"""
            SELECT * FROM goals
            WHERE user_id = ? {archived}
              AND (title LIKE ? OR description LIKE ? OR notes LIKE ?)
            ORDER BY updated_at DESC LIMIT ?
        """, (user_id, like, like, like, max(1, limit)))
        return self._attach_subgoals(rows)

    # ── custom templates ────────────────────────────────────────────────

    def _row_to_template(self, r: dict[str, Any]) -> GoalTemplate:
        return GoalTemplate(
            template_id=r["template_id"],
            user_id=r["user_id"],
            name=r["name"],
            title=r["title"],
            description=r["description"],
            priority=r["priority"],
            tags=json.loads(r["tags"] or "[]"),
            subgoals=json.loads(r["subgoals"] or "[]"),
            created_at=r["created_at"],
        )

    async def save_template(
        self,
        user_id: str,
        name: str,
        title: str,
        *,
        description: str = "",
        priority: int = 0,
        tags: list[str] | None = None,
        subgoals: list[str] | None = None,
    ) -> GoalTemplate:
        """Save a reusable goal template (upserts on name)."""
        name = (name or "").strip().lower().replace(" ", "-")
        if not name:
            raise ValueError("template name must not be empty")
        template_id = new_id("gtemplate")
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO goal_templates
                    (template_id, user_id, name, title, description, priority, tags, subgoals, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, name) DO UPDATE SET
                    title = excluded.title, description = excluded.description,
                    priority = excluded.priority, tags = excluded.tags,
                    subgoals = excluded.subgoals
            """, (template_id, user_id, name, title, description, int(priority),
                  json.dumps(list(tags or [])), json.dumps(list(subgoals or [])), now))
            row = self.db.query_one(
                "SELECT * FROM goal_templates WHERE user_id = ? AND name = ?",
                (user_id, name))
        assert row is not None
        return self._row_to_template(row)

    async def list_custom_templates(self, user_id: str) -> list[GoalTemplate]:
        """User templates plus shared (global) ones."""
        rows = self.db.query("""
            SELECT * FROM goal_templates
            WHERE user_id = ? OR user_id = ''
            ORDER BY name ASC
        """, (user_id,))
        return [self._row_to_template(r) for r in rows]

    async def delete_template(self, user_id: str, name: str) -> bool:
        """Delete a user template. Returns False when it does not exist."""
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT template_id FROM goal_templates WHERE user_id = ? AND name = ?",
                (user_id, (name or "").strip().lower().replace(" ", "-")))
            if not row:
                return False
            self.db.execute("DELETE FROM goal_templates WHERE template_id = ?",
                            (row["template_id"],))
        return True

    def _resolve_template(self, user_id: str, template_name: str) -> dict[str, Any] | None:
        """Custom template first (user, then global), else a built-in."""
        slug = (template_name or "").strip().lower().replace(" ", "-")
        row = self.db.query_one("""
            SELECT * FROM goal_templates
            WHERE name = ? AND (user_id = ? OR user_id = '')
            ORDER BY user_id DESC LIMIT 1
        """, (slug, user_id))
        if row:
            t = self._row_to_template(row)
            return {"title": t.title, "description": t.description,
                    "priority": t.priority, "tags": t.tags, "subgoals": t.subgoals}
        builtin = GOAL_TEMPLATES.get(template_name)
        if builtin:
            return dict(builtin)
        return None

    # ── export ──────────────────────────────────────────────────────────

    async def export_json(self, user_id: str) -> dict[str, Any]:
        """Full machine-readable dump of a user's goals and ideas."""
        goals = await self.list_goals(user_id)
        out_goals = []
        for g in goals:
            d = g.to_dict()
            d["key_results"] = [kr.to_dict() for kr in await self.list_key_results(g.goal_id)]
            d["milestones"] = [m.to_dict() for m in await self.list_milestones(g.goal_id)]
            d["journal"] = [e.to_dict() for e in await self.journal(g.goal_id, limit=1000)]
            d["links"] = [lnk.to_dict() for lnk in await self.goal_links(g.goal_id)]
            d["reminders"] = [r.to_dict() for r in await self.list_reminders(
                user_id, include_acknowledged=True) if r.goal_id == g.goal_id]
            d["history"] = [h.to_dict() for h in await self.progress_history(
                g.goal_id, limit=1000)]
            out_goals.append(d)
        ideas = await IdeaTracker(self.db).list_ideas(user_id)
        return {
            "user_id": user_id,
            "exported_at": time.time(),
            "goals": out_goals,
            "ideas": [i.to_dict() for i in ideas],
        }

    async def export_markdown(self, user_id: str) -> str:
        """Human-readable markdown dump of a user's goals and ideas."""
        import datetime as _dt
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Goals export — {stamp}\n"]
        goals = await self.list_goals(user_id)
        for g in goals:
            bar = "█" * int(g.progress / 10) + "░" * (10 - int(g.progress / 10))
            lines.append(f"## {g.title} `{g.status}` {bar} {g.progress:.0f}%")
            if g.description:
                lines.append(f"_{g.description}_")
            if g.target_date:
                dt = _dt.datetime.fromtimestamp(g.target_date).strftime("%Y-%m-%d")
                lines.append(f"Target: {dt}")
            krs = await self.list_key_results(g.goal_id)
            for kr in krs:
                lines.append(
                    f"- KR: {kr.title} — {kr.current:g}{kr.unit} / {kr.target:g}{kr.unit} "
                    f"(score {kr.score:.2f})")
            for m in await self.list_milestones(g.goal_id):
                box = "[x]" if m.is_completed else "[ ]"
                lines.append(f"- {box} milestone: {m.title}")
            for s in g.subgoals:
                box = "[x]" if s.is_completed else "[ ]"
                lines.append(f"- {box} {s.title}")
            lines.append("")
        ideas = await IdeaTracker(self.db).list_ideas(user_id, status="active")
        if ideas:
            lines.append("## Ideas")
            for i in ideas:
                ice = f" (ICE {i.ice_score})" if i.ice_score is not None else ""
                lines.append(f"- {i.title}{ice}")
        return "\n".join(lines).rstrip() + "\n"


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
            _ensure_column(self.db, "ideas", "impact", "REAL")
            _ensure_column(self.db, "ideas", "confidence", "REAL")
            _ensure_column(self.db, "ideas", "ease", "REAL")

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
            impact=row.get("impact"),
            confidence=row.get("confidence"),
            ease=row.get("ease"),
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

    # ── ICE scoring (Sean Ellis) ──────────────────────────────────────────

    async def score_idea(
        self,
        idea_id: str,
        *,
        impact: float,
        confidence: float,
        ease: float,
    ) -> Idea:
        """Score an idea on Impact / Confidence / Ease (1-10 each).

        ICE = (I + C + E) / 3 — built for fast triage of idea backlogs.
        Calibrate honestly: confidence inflates on pet ideas, and easy
        low-impact work beats hard high-impact work if you let it.
        """
        for name, value in (("impact", impact), ("confidence", confidence), ("ease", ease)):
            if not 1 <= float(value) <= 10:
                raise ValueError(f"{name} must be between 1 and 10 (got {value!r})")
        await self._require(idea_id)
        with self.db.transaction():
            self.db.execute("""
                UPDATE ideas SET impact = ?, confidence = ?, ease = ?
                WHERE idea_id = ?
            """, (float(impact), float(confidence), float(ease), idea_id))
        return await self._require(idea_id)

    async def clear_idea_score(self, idea_id: str) -> Idea:
        """Remove an idea's ICE scores. Raises KeyError when missing."""
        await self._require(idea_id)
        with self.db.transaction():
            self.db.execute("""
                UPDATE ideas SET impact = NULL, confidence = NULL, ease = NULL
                WHERE idea_id = ?
            """, (idea_id,))
        return await self._require(idea_id)

    async def top_ideas(self, user_id: str, *, limit: int = 10) -> list[Idea]:
        """Active ideas ranked by ICE score (unscored sink to the bottom)."""
        ideas = await self.list_ideas(user_id, status="active")
        ideas.sort(
            key=lambda i: (i.ice_score is None, -(i.ice_score or 0.0), -i.created_at))
        return ideas[:max(1, limit)]

    # ── similarity / dedupe ─────────────────────────────────────────────

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}

    async def find_similar_ideas(
        self,
        user_id: str,
        idea_or_text: str,
        *,
        threshold: float = 0.25,
        limit: int = 10,
    ) -> list[tuple[Idea, float]]:
        """Find ideas with overlapping vocabulary (Jaccard similarity).

        Pass an idea id or raw text. Useful for catching duplicates before
        the someday/maybe pile grows teeth.
        """
        idea = await self.get(idea_or_text)
        if idea is not None and idea.user_id == user_id:
            probe = f"{idea.title} {idea.description}"
            exclude = idea.idea_id
        else:
            probe = idea_or_text
            exclude = ""
        probe_tokens = self._tokens(probe)
        if not probe_tokens:
            return []
        scored: list[tuple[Idea, float]] = []
        for other in await self.list_ideas(user_id):
            if other.idea_id == exclude:
                continue
            other_tokens = self._tokens(f"{other.title} {other.description}")
            if not other_tokens:
                continue
            sim = len(probe_tokens & other_tokens) / len(probe_tokens | other_tokens)
            if sim >= threshold:
                scored.append((other, round(sim, 3)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:max(1, limit)]

    # ── review queue (GTD someday/maybe) ──────────────────────────────────

    async def review_queue(self, user_id: str, *, limit: int = 20) -> list[tuple[Idea, int]]:
        """Active ideas oldest-first with days parked — the review pile."""
        ideas = await self.list_ideas(user_id, status="active")
        now = time.time()
        queued = [(i, int((now - i.created_at) / 86400)) for i in ideas]
        queued.sort(key=lambda p: p[0].created_at)
        return queued[:max(1, limit)]

    async def idea_stats(self, user_id: str) -> dict[str, Any]:
        """Aggregate counts for a user's ideas."""
        rows = self.db.query("""
            SELECT status, COUNT(*) AS n FROM ideas WHERE user_id = ? GROUP BY status
        """, (user_id,))
        by_status = {r["status"]: r["n"] for r in rows}
        scored = self.db.query_one("""
            SELECT COUNT(*) AS n, COALESCE(AVG((impact + confidence + ease) / 3.0), 0) AS avg_ice
            FROM ideas
            WHERE user_id = ? AND status = 'active'
              AND impact IS NOT NULL AND confidence IS NOT NULL AND ease IS NOT NULL
        """, (user_id,))
        return {
            "total": sum(by_status.values()),
            "by_status": {s: by_status.get(s, 0) for s in IDEA_STATUSES},
            "scored": scored["n"] if scored else 0,
            "avg_ice": round(float(scored["avg_ice"]), 2) if scored else 0.0,
        }
