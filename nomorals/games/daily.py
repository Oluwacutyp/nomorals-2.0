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
    # streaks
    "STREAK_MILESTONES", "get_streak", "bump_streak", "grant_freeze",
    "repair_streak", "streak_calendar",
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


# ── streaks: the Duolingo retention engine ──────────────────────────────────
#
# A binary done/not-done hunt doesn't retain anyone. Streaks do:
# consecutive days of hunt completion, a streak freeze that forgives one
# missed day, coin-based streak repair, and milestone celebrations at
# 7 / 30 / 100 / 365 days. The calendar render shows the week at a glance.

STREAK_MILESTONES: tuple[tuple[int, str], ...] = (
    (7, "🔥 week of fire — 7-day streak!"),
    (30, "🌟 a full month — 30-day streak, unstoppable!"),
    (100, "💯 CENTURY — 100 days. Legendary."),
    (365, "👑 A FULL YEAR — 365-day streak. Immortal."),
)


def _ensure_streaks(db: Any) -> None:
    _ensure(db)
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS daily_streaks ("
            "player_key TEXT PRIMARY KEY, "
            "streak INTEGER NOT NULL DEFAULT 0, "
            "best INTEGER NOT NULL DEFAULT 0, "
            "last_date TEXT NOT NULL DEFAULT '', "
            "freezes INTEGER NOT NULL DEFAULT 0)")
    except Exception:  # noqa: BLE001
        _log.debug("daily_streaks ensure failed", exc_info=True)


def _yesterday(date: str) -> str:
    from datetime import datetime, timedelta
    d = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def get_streak(db: Any, player_key: str) -> dict[str, Any]:
    """``{'streak','best','freezes','last_date'}`` — never raises."""
    _ensure_streaks(db)
    out = {"streak": 0, "best": 0, "freezes": 0, "last_date": ""}
    try:
        row = db.query_one(
            "SELECT streak, best, last_date, freezes FROM daily_streaks "
            "WHERE player_key = ?", (player_key,))
        if row:
            out.update(streak=int(row.get("streak") or 0),
                       best=int(row.get("best") or 0),
                       freezes=int(row.get("freezes") or 0),
                       last_date=str(row.get("last_date") or ""))
    except Exception:  # noqa: BLE001
        pass
    return out


def _save_streak(db: Any, player_key: str, info: dict[str, Any]) -> None:
    try:
        db.execute(
            "INSERT INTO daily_streaks "
            "(player_key, streak, best, last_date, freezes) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(player_key) DO UPDATE SET "
            "streak = excluded.streak, best = excluded.best, "
            "last_date = excluded.last_date, freezes = excluded.freezes",
            (player_key, info["streak"], info["best"],
             info["last_date"], info["freezes"]))
    except Exception:  # noqa: BLE001
        _log.debug("streak save failed", exc_info=True)


def _milestone_hit(streak: int) -> str:
    for days, msg in STREAK_MILESTONES:
        if streak == days:
            return msg
    return ""


def bump_streak(db: Any, player_key: str,
                date: str | None = None) -> dict[str, Any]:
    """Record a hunt completion for ``date`` (default today).

    Returns ``{'streak','best','milestone','continued'}``. A gap of one
    day auto-consumes a freeze if the player holds one; a longer gap
    resets the streak (see :func:`repair_streak` for the paid way back).
    """
    date = date or hunt_date()
    info = get_streak(db, player_key)
    last = info["last_date"]
    continued = True
    if last == date:
        return {"streak": info["streak"], "best": info["best"],
                "milestone": "", "continued": True, "already": True}
    if not last:
        info["streak"] = 1
    elif last == _yesterday(date):
        info["streak"] += 1
    else:
        # gap: exactly one missed day → burn a freeze if held,
        # otherwise the streak resets
        if last == _yesterday(_yesterday(date)) and info["freezes"] > 0:
            info["freezes"] -= 1
            info["streak"] += 1
        else:
            info["streak"] = 1
            continued = False
    info["best"] = max(info["best"], info["streak"])
    info["last_date"] = date
    _save_streak(db, player_key, info)
    milestone = _milestone_hit(info["streak"])
    return {"streak": info["streak"], "best": info["best"],
            "milestone": milestone, "continued": continued,
            "already": False}


def grant_freeze(db: Any, player_key: str, n: int = 1) -> int:
    """Give streak freezes (gem shop / milestones). Returns new count."""
    info = get_streak(db, player_key)
    info["freezes"] = max(0, info["freezes"] + n)
    _save_streak(db, player_key, info)
    return info["freezes"]


def repair_streak(db: Any, player_key: str, cost: int = 200) -> tuple[bool, str]:
    """Restore yesterday's broken streak for coins. The streak value is
    kept as it was — the missed day is forgiven, not replayed."""
    info = get_streak(db, player_key)
    today = hunt_date()
    if info["last_date"] in (today, _yesterday(today)):
        return False, "your streak is intact — nothing to repair."
    if info["streak"] <= 0:
        return False, "no streak to repair — start a new one today."
    # charge via the player store when available; signature is
    # (db, player_key, cost) → bool provided by the caller through
    # the ``_coin_charger`` hook to avoid a hard import cycle.
    charger = globals().get("_coin_charger")
    if charger is not None:
        try:
            if not charger(db, player_key, cost):
                return False, (f"streak repair costs {cost}c — "
                               f"not enough coins.")
        except Exception:  # noqa: BLE001
            return False, "couldn't charge the repair — try again."
    info["last_date"] = _yesterday(today)
    _save_streak(db, player_key, info)
    return True, (f"🧊 streak repaired — you're back at "
                  f"{info['streak']} days. Don't miss today!")


def streak_calendar(db: Any, player_key: str) -> str:
    """Week view: ``🔥🔥🔥⬜⬜⬜⬜`` + streak stats."""
    info = get_streak(db, player_key)
    done = daily_hunt_done(db, player_key)
    today_idx = datetime.now().weekday()  # Mon=0
    cells = []
    for i in range(7):
        if i < today_idx:
            cells.append("🔥" if info["streak"] > today_idx - i else "⬜")
        elif i == today_idx:
            cells.append("✅" if done else "⏳")
        else:
            cells.append("·")
    lines = [f"🔥 streak: {info['streak']} days (best {info['best']})",
             " ".join(cells) + "  M T W T F S S"]
    if info["freezes"]:
        lines.append(f"🧊 {info['freezes']} freeze(s) banked — "
                     f"one missed day forgiven each")
    if info["streak"] and not done:
        lines.append("today's hunt is still open — keep it alive. /daily")
    return "\n".join(lines)
