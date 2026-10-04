"""Daily hunt: one special arena win per day pays double XP.

Every day the arena posts a hunt — win any arena battle before midnight
and the XP doubles.  It resets at midnight local time, one completion per
player per day.  The ``/daily`` command shows today's status.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "hunt_date", "daily_hunt_done", "complete_daily_hunt",
    "daily_hunt_status",
]

_log = get_logger(__name__)


def hunt_date(now: float | None = None) -> str:
    """Today's hunt date as ``YYYY-MM-DD`` (local time)."""
    return datetime.fromtimestamp(now or time.time()).strftime("%Y-%m-%d")


def _ensure(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS daily_hunts ("
            "player_key TEXT NOT NULL, hunt_date TEXT NOT NULL, "
            "completed_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, hunt_date))")
    except Exception:  # noqa: BLE001
        _log.debug("daily_hunts ensure failed", exc_info=True)


def daily_hunt_done(db: Any, player_key: str,
                    date: str | None = None) -> bool:
    """True if this player already completed the hunt for ``date``."""
    _ensure(db)
    try:
        rows = db.query(
            "SELECT 1 FROM daily_hunts WHERE player_key = ? "
            "AND hunt_date = ? LIMIT 1",
            (player_key, date or hunt_date()))
        return bool(rows)
    except Exception:  # noqa: BLE001
        _log.debug("daily_hunts check failed", exc_info=True)
        return False


def complete_daily_hunt(db: Any, player_key: str) -> bool:
    """Mark today's hunt complete. True if newly completed."""
    _ensure(db)
    try:
        cur = db.execute(
            "INSERT OR IGNORE INTO daily_hunts "
            "(player_key, hunt_date, completed_at) VALUES (?, ?, ?)",
            (player_key, hunt_date(), time.time()))
        return cur.rowcount > 0
    except Exception:  # noqa: BLE001
        _log.debug("daily_hunts complete failed", exc_info=True)
        return False


def daily_hunt_status(db: Any, player_key: str) -> str:
    """One-liner describing today's hunt for ``/daily``."""
    if daily_hunt_done(db, player_key):
        return ("🎯 today's hunt is done — the arena rests. "
                "Come back tomorrow for double XP.")
    return ("🎯 today's hunt: win any arena battle before midnight "
            "for DOUBLE XP. The hunt waits.")
