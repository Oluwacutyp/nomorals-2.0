"""Event-condition-action automation for Devon.

A trigger is ``when <source> says so, do <action>``:

* **sources** — ``schedule`` (cron/interval, via the existing scheduler),
  ``file`` (path change), ``price`` (keyless market-data threshold),
  ``message`` (regex on inbound chat), ``webhook`` (HTTP POST).
* **actions** — ``notify`` (owner-only channel), ``message`` (send to a
  chat), ``command`` (run an ``nm`` command), ``mission`` (start a mission).

Definitions persist in SQLite (``storage.db.Database``, the same pattern
the scheduler uses) and survive restarts.  Every evaluation outcome —
fired, no-match, skipped, error — lands in the history table; nothing is
dropped silently.  Invalid definitions fail fast at creation time.

Layer: L5, peers with scheduler/missions.  Only imports downward (L4 and
below) and sideways (L5 peers: agents, integrations, missions, scheduler).
"""

from __future__ import annotations

from .engine import TriggerEngine, attach, message_hook
from .models import (
    ACTION_MESSAGE,
    ACTION_COMMAND,
    ACTION_MISSION,
    ACTION_NOTIFY,
    ACTIONS,
    SOURCE_FILE,
    SOURCE_MESSAGE,
    SOURCE_PRICE,
    SOURCE_SCHEDULE,
    SOURCE_WEBHOOK,
    SOURCES,
    Trigger,
    TriggerError,
    validate_definition,
)
from .store import TriggerStore

__all__ = [
    "TriggerEngine",
    "TriggerStore",
    "Trigger",
    "TriggerError",
    "attach",
    "message_hook",
    "validate_definition",
    "SOURCES",
    "ACTIONS",
    "SOURCE_SCHEDULE",
    "SOURCE_FILE",
    "SOURCE_PRICE",
    "SOURCE_MESSAGE",
    "SOURCE_WEBHOOK",
    "ACTION_NOTIFY",
    "ACTION_MESSAGE",
    "ACTION_COMMAND",
    "ACTION_MISSION",
]
