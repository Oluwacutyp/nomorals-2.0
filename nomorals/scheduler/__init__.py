"""Scheduler system for cron jobs, reminders, and event hooks.

This is the user-facing scheduler API. For agent-internal task scheduling,
see nomorals.agents.scheduler.

Supports:
- One-time scheduled tasks
- Recurring cron jobs (minute, hourly, daily, weekly, monthly)
- Event hooks (fire when data arrives)
- Reminders with open/close lifecycle
- Goal-owned crons (check-ins, nudges)

Usage:
    from nomorals.scheduler import Scheduler, CronJob, Reminder
    
    scheduler = Scheduler(db)
    
    # One-time task
    await scheduler.schedule_once(
        task_id="send_report",
        run_at=datetime(2026, 9, 30, 14, 0),
        action="send_email",
        parameters={"to": "boss@company.com", "subject": "Weekly Report"},
    )
    
    # Recurring cron
    await scheduler.schedule_cron(
        task_id="daily_standup",
        cron_expr="0 9 * * MON-FRI",  # 9am weekdays
        action="send_reminder",
        parameters={"message": "Daily standup in 15 minutes"},
    )
    
    # Reminder with lifecycle
    reminder = await scheduler.create_reminder(
        text="Call mom",
        due_at=datetime(2026, 9, 29, 18, 0),
        user_id="user123",
    )
    
    # Later: mark complete
    await scheduler.complete_reminder(reminder.reminder_id)
"""

from .scheduler import Scheduler, CronJob, Reminder, EventHook, ScheduledTask

__all__ = [
    "Scheduler",
    "CronJob",
    "Reminder",
    "EventHook",
    "ScheduledTask",
]
