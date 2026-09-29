"""Scheduler implementation with cron, reminders, and event hooks.

Production-grade scheduler with:
- Persistent SQLite storage
- Cron expression parsing
- Background worker thread
- Event hooks (fire on data arrival)
- Reminder lifecycle (open/closed/snoozed)
- Goal-owned scheduled tasks
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "Scheduler",
    "CronJob",
    "Reminder",
    "EventHook",
    "ScheduledTask",
    "TaskStatus",
]

_log = get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of a scheduled task."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SNOOZED = "snoozed"


@dataclass
class ScheduledTask:
    """Base class for all scheduled items."""
    
    task_id: str
    task_type: str = ""  # cron, reminder, event_hook, one_time (set by subclass __post_init__)
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "action": self.action,
            "parameters": self.parameters,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class CronJob(ScheduledTask):
    """A recurring cron job."""
    
    cron_expr: str = ""  # e.g., "0 9 * * MON-FRI"
    next_run: float = 0.0
    last_run: Optional[float] = None
    run_count: int = 0
    max_runs: Optional[int] = None  # None = unlimited
    goal_id: Optional[str] = None  # If owned by a goal
    
    def __post_init__(self):
        self.task_type = "cron"


@dataclass
class Reminder(ScheduledTask):
    """A reminder with lifecycle."""
    
    text: str = ""
    due_at: float = 0.0
    user_id: str = ""
    snooze_count: int = 0
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        self.task_type = "reminder"
    
    def is_overdue(self) -> bool:
        """Check if reminder is overdue."""
        return self.status == TaskStatus.PENDING and time.time() > self.due_at


@dataclass
class EventHook(ScheduledTask):
    """Fires when specific data arrives."""
    
    event_type: str = ""  # e.g., "email_received", "price_drop"
    conditions: dict[str, Any] = field(default_factory=dict)
    trigger_count: int = 0
    max_triggers: Optional[int] = None
    
    def __post_init__(self):
        self.task_type = "event_hook"
    
    def matches(self, event_data: dict[str, Any]) -> bool:
        """Check if event data matches conditions."""
        for key, expected in self.conditions.items():
            actual = event_data.get(key)
            if actual != expected:
                return False
        return True


class CronParser:
    """Parse cron expressions.
    
    Supports: minute hour day_of_month month day_of_week
    
    Examples:
        "0 9 * * *"      - 9am every day
        "0 9 * * MON-FRI" - 9am weekdays
        "*/15 * * * *"   - Every 15 minutes
        "0 0 1 * *"      - Midnight on 1st of each month
    """
    
    DAY_NAMES = {
        "SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
        "THU": 4, "FRI": 5, "SAT": 6,
    }
    
    @classmethod
    def parse(cls, expr: str) -> dict[str, Any]:
        """Parse cron expression into components."""
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr}")
        
        return {
            "minute": cls._parse_field(parts[0], 0, 59),
            "hour": cls._parse_field(parts[1], 0, 23),
            "day_of_month": cls._parse_field(parts[2], 1, 31),
            "month": cls._parse_field(parts[3], 1, 12),
            "day_of_week": cls._parse_day_field(parts[4]),
        }
    
    @classmethod
    def _parse_field(cls, field: str, min_val: int, max_val: int) -> set[int]:
        """Parse a single cron field."""
        if field == "*":
            return set(range(min_val, max_val + 1))
        
        if field.startswith("*/"):
            step = int(field[2:])
            return set(range(min_val, max_val + 1, step))
        
        if "-" in field:
            start, end = field.split("-")
            return set(range(int(start), int(end) + 1))
        
        if "," in field:
            return {int(x) for x in field.split(",")}
        
        return {int(field)}
    
    @classmethod
    def _parse_day_field(cls, field: str) -> set[int]:
        """Parse day of week field (supports names)."""
        if field == "*":
            return set(range(7))
        
        # Replace day names with numbers
        for name, num in cls.DAY_NAMES.items():
            field = field.replace(name, str(num))
        
        return cls._parse_field(field, 0, 6)
    
    @classmethod
    def next_run(cls, expr: str, after: float | None = None) -> float:
        """Calculate next run time for cron expression."""
        parsed = cls.parse(expr)
        after = after or time.time()
        dt = datetime.fromtimestamp(after)
        
        # Start from next minute
        dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Try next 366 days
        for _ in range(366 * 24 * 60):  # Minutes in a year
            if (
                dt.minute in parsed["minute"]
                and dt.hour in parsed["hour"]
                and dt.day in parsed["day_of_month"]
                and dt.month in parsed["month"]
                and dt.weekday() in parsed["day_of_week"]
            ):
                return dt.timestamp()
            dt += timedelta(minutes=1)
        
        raise ValueError(f"No valid run time found for: {expr}")


class Scheduler:
    """Production-grade scheduler with cron, reminders, and event hooks."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._action_handlers: dict[str, Callable] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._ensure_schema()
        _log.info("Scheduler initialized")
    
    def _ensure_schema(self) -> None:
        """Create scheduler tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    parameters TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    task_id TEXT PRIMARY KEY,
                    cron_expr TEXT NOT NULL,
                    next_run REAL NOT NULL,
                    last_run REAL,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    max_runs INTEGER,
                    goal_id TEXT,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    task_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    due_at REAL NOT NULL,
                    user_id TEXT NOT NULL,
                    snooze_count INTEGER NOT NULL DEFAULT 0,
                    completed_at REAL,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS event_hooks (
                    task_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    conditions TEXT NOT NULL DEFAULT '{}',
                    trigger_count INTEGER NOT NULL DEFAULT 0,
                    max_triggers INTEGER,
                    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id)
                )
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON scheduled_tasks(status)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cron_next_run
                ON cron_jobs(next_run)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                ON reminders(due_at)
            """)
    
    # ── Cron Jobs ────────────────────────────────────────────────────────────
    
    async def schedule_cron(
        self,
        task_id: str | None,
        cron_expr: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_runs: int | None = None,
        goal_id: str | None = None,
    ) -> CronJob:
        """Schedule a recurring cron job.
        
        Args:
            task_id: Unique task ID (generated if None)
            cron_expr: Cron expression (e.g., "0 9 * * MON-FRI")
            action: Action to execute
            parameters: Action parameters
            max_runs: Maximum executions (None = unlimited)
            goal_id: Optional goal that owns this job
            
        Returns:
            CronJob object
        """
        task_id = task_id or new_id("cron")
        parameters = parameters or {}
        
        # Validate cron expression
        next_run = CronParser.next_run(cron_expr)
        
        # Create task
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'cron', ?, ?, 'pending', ?, ?, '{}')
            """, (task_id, action, json.dumps(parameters), time.time(), time.time()))
            
            self.db.execute("""
                INSERT INTO cron_jobs (task_id, cron_expr, next_run, last_run, run_count, max_runs, goal_id)
                VALUES (?, ?, ?, NULL, 0, ?, ?)
            """, (task_id, cron_expr, next_run, max_runs, goal_id))
        
        job = CronJob(
            task_id=task_id,
            action=action,
            parameters=parameters,
            cron_expr=cron_expr,
            next_run=next_run,
            max_runs=max_runs,
            goal_id=goal_id,
        )
        
        _log.info(f"Scheduled cron job: {task_id} ({cron_expr})")
        return job
    
    async def cancel_cron(self, task_id: str) -> bool:
        """Cancel a cron job."""
        with self.db.transaction():
            self.db.execute(
                "UPDATE scheduled_tasks SET status = 'cancelled', updated_at = ? WHERE task_id = ?",
                (time.time(), task_id)
            )
        _log.info(f"Cancelled cron job: {task_id}")
        return True
    
    async def list_cron_jobs(self, *, active_only: bool = True) -> list[CronJob]:
        """List all cron jobs."""
        query = """
            SELECT t.*, c.* FROM scheduled_tasks t
            JOIN cron_jobs c ON t.task_id = c.task_id
        """
        if active_only:
            query += " WHERE t.status = 'pending'"
        
        rows = self.db.query(query)
        
        jobs = []
        for row in rows:
            jobs.append(CronJob(
                task_id=row["task_id"],
                action=row["action"],
                parameters=json.loads(row["parameters"]),
                status=TaskStatus(row["status"]),
                cron_expr=row["cron_expr"],
                next_run=row["next_run"],
                last_run=row["last_run"],
                run_count=row["run_count"],
                max_runs=row["max_runs"],
                goal_id=row["goal_id"],
            ))
        
        return jobs
    
    # ── Reminders ────────────────────────────────────────────────────────────
    
    async def create_reminder(
        self,
        text: str,
        due_at: datetime | float,
        user_id: str,
        *,
        action: str = "send_reminder",
        parameters: dict[str, Any] | None = None,
    ) -> Reminder:
        """Create a reminder.
        
        Args:
            text: Reminder text
            due_at: When reminder is due
            user_id: User to remind
            action: Action to execute (default: send_reminder)
            parameters: Additional parameters
            
        Returns:
            Reminder object
        """
        task_id = new_id("reminder")
        parameters = parameters or {}
        parameters.setdefault("text", text)
        parameters.setdefault("user_id", user_id)
        
        if isinstance(due_at, datetime):
            due_at = due_at.timestamp()
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'reminder', ?, ?, 'pending', ?, ?, '{}')
            """, (task_id, action, json.dumps(parameters), time.time(), time.time()))
            
            self.db.execute("""
                INSERT INTO reminders (task_id, text, due_at, user_id, snooze_count, completed_at)
                VALUES (?, ?, ?, ?, 0, NULL)
            """, (task_id, text, due_at, user_id))
        
        reminder = Reminder(
            task_id=task_id,
            action=action,
            parameters=parameters,
            text=text,
            due_at=due_at,
            user_id=user_id,
        )
        
        _log.info(f"Created reminder: {text} (due: {datetime.fromtimestamp(due_at)})")
        return reminder
    
    async def complete_reminder(self, task_id: str) -> bool:
        """Mark reminder as completed."""
        now = time.time()
        with self.db.transaction():
            self.db.execute("""
                UPDATE scheduled_tasks SET status = 'completed', updated_at = ? WHERE task_id = ?
            """, (now, task_id))
            self.db.execute("""
                UPDATE reminders SET completed_at = ? WHERE task_id = ?
            """, (now, task_id))
        _log.info(f"Completed reminder: {task_id}")
        return True
    
    async def snooze_reminder(
        self,
        task_id: str,
        minutes: int = 15,
    ) -> bool:
        """Snooze a reminder."""
        new_due = time.time() + (minutes * 60)
        now = time.time()
        
        with self.db.transaction():
            self.db.execute("""
                UPDATE reminders SET due_at = ?, snooze_count = snooze_count + 1 WHERE task_id = ?
            """, (new_due, task_id))
            self.db.execute("""
                UPDATE scheduled_tasks SET status = 'snoozed', updated_at = ? WHERE task_id = ?
            """, (now, task_id))
        
        _log.info(f"Snoozed reminder {task_id} for {minutes} minutes")
        return True
    
    async def list_reminders(
        self,
        user_id: str,
        *,
        include_completed: bool = False,
    ) -> list[Reminder]:
        """List reminders for a user."""
        query = """
            SELECT t.*, r.* FROM scheduled_tasks t
            JOIN reminders r ON t.task_id = r.task_id
            WHERE r.user_id = ?
        """
        params: list[Any] = [user_id]
        
        if not include_completed:
            query += " AND t.status IN ('pending', 'snoozed')"
        
        query += " ORDER BY r.due_at ASC"
        
        rows = self.db.query(query, params)
        
        reminders = []
        for row in rows:
            reminders.append(Reminder(
                task_id=row["task_id"],
                action=row["action"],
                parameters=json.loads(row["parameters"]),
                status=TaskStatus(row["status"]),
                text=row["text"],
                due_at=row["due_at"],
                user_id=row["user_id"],
                snooze_count=row["snooze_count"],
                completed_at=row["completed_at"],
            ))
        
        return reminders
    
    # ── Event Hooks ──────────────────────────────────────────────────────────
    
    async def create_event_hook(
        self,
        event_type: str,
        conditions: dict[str, Any],
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_triggers: int | None = None,
    ) -> EventHook:
        """Create an event hook.
        
        Args:
            event_type: Event type to listen for
            conditions: Conditions that must match
            action: Action to execute when triggered
            parameters: Action parameters
            max_triggers: Maximum triggers (None = unlimited)
            
        Returns:
            EventHook object
        """
        task_id = new_id("hook")
        parameters = parameters or {}
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'event_hook', ?, ?, 'pending', ?, ?, '{}')
            """, (task_id, action, json.dumps(parameters), time.time(), time.time()))
            
            self.db.execute("""
                INSERT INTO event_hooks (task_id, event_type, conditions, trigger_count, max_triggers)
                VALUES (?, ?, ?, 0, ?)
            """, (task_id, event_type, json.dumps(conditions), max_triggers))
        
        hook = EventHook(
            task_id=task_id,
            action=action,
            parameters=parameters,
            event_type=event_type,
            conditions=conditions,
            max_triggers=max_triggers,
        )
        
        _log.info(f"Created event hook: {event_type} -> {action}")
        return hook
    
    async def trigger_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
    ) -> list[str]:
        """Trigger an event and fire matching hooks.
        
        Args:
            event_type: Event type
            event_data: Event data
            
        Returns:
            List of triggered hook IDs
        """
        # Find matching hooks
        rows = self.db.query("""
            SELECT h.*, t.action, t.parameters FROM event_hooks h
            JOIN scheduled_tasks t ON h.task_id = t.task_id
            WHERE h.event_type = ? AND t.status = 'pending'
        """, (event_type,))
        
        triggered = []
        
        for row in rows:
            conditions = json.loads(row["conditions"])
            
            # Check if conditions match
            matches = all(
                event_data.get(k) == v
                for k, v in conditions.items()
            )
            
            if matches:
                # Fire the hook
                action = row["action"]
                parameters = json.loads(row["parameters"])
                
                await self._execute_action(action, parameters)
                
                # Update trigger count
                with self.db.transaction():
                    self.db.execute("""
                        UPDATE event_hooks SET trigger_count = trigger_count + 1 WHERE task_id = ?
                    """, (row["task_id"],))
                    
                    # Check if max triggers reached
                    if row["max_triggers"] and (row["trigger_count"] + 1) >= row["max_triggers"]:
                        self.db.execute("""
                            UPDATE scheduled_tasks SET status = 'completed' WHERE task_id = ?
                        """, (row["task_id"],))
                
                triggered.append(row["task_id"])
                _log.info(f"Triggered event hook: {row['task_id']}")
        
        return triggered
    
    # ── One-Time Tasks ───────────────────────────────────────────────────────
    
    async def schedule_once(
        self,
        task_id: str | None,
        run_at: datetime | float,
        action: str,
        parameters: dict[str, Any] | None = None,
    ) -> ScheduledTask:
        """Schedule a one-time task.
        
        Args:
            task_id: Unique task ID
            run_at: When to run
            action: Action to execute
            parameters: Action parameters
            
        Returns:
            ScheduledTask object
        """
        task_id = task_id or new_id("task")
        parameters = parameters or {}
        
        if isinstance(run_at, datetime):
            run_at = run_at.timestamp()
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO scheduled_tasks (task_id, task_type, action, parameters, status, created_at, updated_at, metadata)
                VALUES (?, 'one_time', ?, ?, 'pending', ?, ?, ?)
            """, (task_id, action, json.dumps(parameters), time.time(), time.time(),
                  json.dumps({"run_at": run_at})))
        
        task = ScheduledTask(
            task_id=task_id,
            task_type="one_time",
            action=action,
            parameters=parameters,
            metadata={"run_at": run_at},
        )
        
        _log.info(f"Scheduled one-time task: {task_id} at {datetime.fromtimestamp(run_at)}")
        return task
    
    # ── Worker ───────────────────────────────────────────────────────────────
    
    def start(self) -> None:
        """Start the scheduler worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        _log.info("Scheduler worker started")
    
    def stop(self) -> None:
        """Stop the scheduler worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        _log.info("Scheduler worker stopped")
    
    def _worker_loop(self) -> None:
        """Background worker loop."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                _log.error(f"Scheduler tick failed: {e}")
            
            time.sleep(1)  # Check every second
    
    def _tick(self) -> None:
        """Process due tasks."""
        now = time.time()
        
        # Process due cron jobs
        cron_rows = self.db.query("""
            SELECT c.*, t.action, t.parameters FROM cron_jobs c
            JOIN scheduled_tasks t ON c.task_id = t.task_id
            WHERE c.next_run <= ? AND t.status = 'pending'
        """, (now,))
        
        for row in cron_rows:
            asyncio.run(self._execute_cron(row))
        
        # Process due reminders
        reminder_rows = self.db.query("""
            SELECT r.*, t.action, t.parameters FROM reminders r
            JOIN scheduled_tasks t ON r.task_id = t.task_id
            WHERE r.due_at <= ? AND t.status = 'pending'
        """, (now,))
        
        for row in reminder_rows:
            asyncio.run(self._execute_reminder(row))
        
        # Process one-time tasks
        task_rows = self.db.query("""
            SELECT * FROM scheduled_tasks
            WHERE task_type = 'one_time' AND status = 'pending'
        """)
        
        for row in task_rows:
            metadata = json.loads(row["metadata"])
            run_at = metadata.get("run_at", 0)
            if run_at <= now:
                asyncio.run(self._execute_task(row))
    
    async def _execute_cron(self, row: dict[str, Any]) -> None:
        """Execute a cron job."""
        task_id = row["task_id"]
        action = row["action"]
        parameters = json.loads(row["parameters"])
        
        try:
            await self._execute_action(action, parameters)
            
            # Update run count and next run
            next_run = CronParser.next_run(row["cron_expr"])
            run_count = row["run_count"] + 1
            
            with self.db.transaction():
                self.db.execute("""
                    UPDATE cron_jobs SET last_run = ?, next_run = ?, run_count = ? WHERE task_id = ?
                """, (time.time(), next_run, run_count, task_id))
                
                # Check if max runs reached
                if row["max_runs"] and run_count >= row["max_runs"]:
                    self.db.execute("""
                        UPDATE scheduled_tasks SET status = 'completed' WHERE task_id = ?
                    """, (task_id,))
            
            _log.info(f"Executed cron job: {task_id} (run {run_count})")
        except Exception as e:
            _log.error(f"Cron job failed: {task_id} - {e}")
    
    async def _execute_reminder(self, row: dict[str, Any]) -> None:
        """Execute a reminder."""
        task_id = row["task_id"]
        action = row["action"]
        parameters = json.loads(row["parameters"])
        
        try:
            await self._execute_action(action, parameters)
            _log.info(f"Executed reminder: {task_id}")
        except Exception as e:
            _log.error(f"Reminder failed: {task_id} - {e}")
    
    async def _execute_task(self, row: dict[str, Any]) -> None:
        """Execute a one-time task."""
        task_id = row["task_id"]
        action = row["action"]
        parameters = json.loads(row["parameters"])
        
        try:
            await self._execute_action(action, parameters)
            
            with self.db.transaction():
                self.db.execute("""
                    UPDATE scheduled_tasks SET status = 'completed', updated_at = ? WHERE task_id = ?
                """, (time.time(), task_id))
            
            _log.info(f"Executed one-time task: {task_id}")
        except Exception as e:
            _log.error(f"Task failed: {task_id} - {e}")
    
    async def _execute_action(self, action: str, parameters: dict[str, Any]) -> None:
        """Execute an action."""
        handler = self._action_handlers.get(action)
        
        if handler:
            await handler(**parameters)
        else:
            _log.warning(f"No handler for action: {action}")
    
    def register_action(self, name: str, handler: Callable) -> None:
        """Register an action handler.
        
        Args:
            name: Action name
            handler: Async function to handle the action
        """
        self._action_handlers[name] = handler
        _log.info(f"Registered action handler: {name}")
