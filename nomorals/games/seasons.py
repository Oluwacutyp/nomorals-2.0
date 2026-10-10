"""Seasonal events: a native, deterministic rotating event calendar.

Every week the tables run under one seasonal event — combat pits pay
extra under the Blood Moon, the casino gilds its payouts during Golden
Week, puzzles pay double XP in Mind Games week. No external API, no
cron, no config: the active event derives from the ISO calendar week,
so every deployment agrees on what season it is without talking to
anyone.

Each event carries coin/XP multipliers for the games it touches. The
engine applies them at finish time (coins + XP) and announces the
season at table open. ``/game events`` shows the current season and
what's coming.

An owner override lives in the DB (``season_override``: event id +
expiry) — set it directly when you want to force a season for an
occasion; the calendar resumes when it lapses.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

_log = logging.getLogger("nomorals.games.seasons")

__all__ = [
    "SEASON_ROSTER",
    "active_event",
    "apply_event",
    "describe_seasons",
    "event_blurb",
    "event_for_week",
]

#: id, name, emoji, games it touches, coin mult, xp mult, one-line blurb.
SEASON_ROSTER: tuple[dict[str, Any], ...] = (
    {
        "id": "blood_moon",
        "name": "Blood Moon",
        "emoji": "🌙",
        "games": ("arena", "pvp", "raid"),
        "coin_mult": 1.25,
        "xp_mult": 1.25,
        "blurb": "the Blood Moon rises — combat pits pay +25% coins & XP.",
    },
    {
        "id": "golden_week",
        "name": "Golden Week",
        "emoji": "🪙",
        "games": ("blackjack", "roulette", "slots", "poker", "craps"),
        "coin_mult": 1.5,
        "xp_mult": 1.0,
        "blurb": "Golden Week — the casino pays +50% coins on every game.",
    },
    {
        "id": "mind_games",
        "name": "Mind Games",
        "emoji": "🧠",
        "games": ("sudoku", "anagram", "cryptogram", "wordle", "case",
                  "trivia", "duel", "20q", "hangman"),
        "coin_mult": 1.0,
        "xp_mult": 2.0,
        "blurb": "Mind Games week — puzzles & quizzes pay DOUBLE XP.",
    },
    {
        "id": "grand_melee",
        "name": "Grand Melee",
        "emoji": "⚔️",
        "games": ("pvp", "raid", "arena", "duel"),
        "coin_mult": 2.0,
        "xp_mult": 1.5,
        "blurb": "the Grand Melee — duels pay DOUBLE coins, +50% XP.",
    },
    {
        "id": "harvest",
        "name": "Harvest Festival",
        "emoji": "🌾",
        "games": ("world", "shop", "auction"),
        "coin_mult": 1.5,
        "xp_mult": 1.5,
        "blurb": "Harvest Festival — builders & traders earn +50%.",
    },
    {
        "id": "quiet",
        "name": "Quiet Season",
        "emoji": "🌫️",
        "games": (),
        "coin_mult": 1.0,
        "xp_mult": 1.0,
        "blurb": "quiet season — the tables rest, no bonuses this week.",
    },
)

_BY_ID = {e["id"]: e for e in SEASON_ROSTER}


def event_for_week(iso_year: int, iso_week: int) -> dict[str, Any]:
    """Deterministic season for an ISO week — no I/O, no API."""
    idx = (int(iso_year) * 53 + int(iso_week)) % len(SEASON_ROSTER)
    return SEASON_ROSTER[idx]


def _override(db: Any) -> dict[str, Any] | None:
    """Owner-forced season, or None when no override is live."""
    if db is None:
        return None
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS season_override ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "event_id TEXT NOT NULL, until REAL NOT NULL DEFAULT 0)")
        row = db.query_one(
            "SELECT event_id, until FROM season_override WHERE id = 1")
        if row and float(row.get("until") or 0) > time.time():
            return _BY_ID.get(str(row.get("event_id")))
    except Exception:  # noqa: BLE001
        _log.debug("season override read failed", exc_info=True)
    return None


def active_event(db: Any = None, now: float | None = None) -> dict[str, Any]:
    """This week's seasonal event (override wins over the calendar)."""
    forced = _override(db)
    if forced is not None:
        return forced
    dt = datetime.fromtimestamp(now if now is not None else time.time())
    iso_year, iso_week, _ = dt.isocalendar()
    return event_for_week(iso_year, iso_week)


def event_touches(event: dict[str, Any], game: str) -> bool:
    return (game or "") in (event.get("games") or ())


def event_blurb(db: Any, game: str,
                now: float | None = None) -> str | None:
    """Season announcement for a table open, or None when the season
    doesn't touch this game."""
    try:
        event = active_event(db, now)
    except Exception:  # noqa: BLE001
        return None
    if not event_touches(event, game):
        return None
    return f"{event['emoji']} **{event['name']}** — {event['blurb']}"


def apply_event(db: Any, game: str, kind: str, amount: int,
                now: float | None = None) -> tuple[int, str]:
    """Apply the season's multiplier to a coin/XP payout.

    Returns ``(new_amount, note)`` — note is "" when nothing applied.
    """
    try:
        event = active_event(db, now)
    except Exception:  # noqa: BLE001
        return amount, ""
    if not event_touches(event, game):
        return amount, ""
    key = {"coins": "coin_mult", "xp": "xp_mult"}.get(kind, f"{kind}_mult")
    mult = float(event.get(key, 1.0) or 1.0)
    if mult == 1.0:
        return amount, ""
    new_amount = int(round(amount * mult))
    return new_amount, f" · {event['emoji']} {event['name']} x{mult:g}"


def describe_seasons(db: Any = None, now: float | None = None) -> str:
    """``/game events`` text: now + the next two weeks."""
    try:
        cur = active_event(db, now)
    except Exception:  # noqa: BLE001
        return "the seasons are unreadable right now."
    now_ts = now if now is not None else time.time()
    lines = ["🗓️ seasons — the tables turn with the weeks:"]
    lines.append(f"  now: {cur['emoji']} **{cur['name']}** — {cur['blurb']}")
    seen = {cur["id"]}
    # peek at the next 8 weeks, show the next 2 distinct seasons
    upcoming: list[dict[str, Any]] = []
    week_ts = now_ts
    for _ in range(56):
        week_ts += 86400
        dt = datetime.fromtimestamp(week_ts)
        iso_year, iso_week, _ = dt.isocalendar()
        ev = event_for_week(iso_year, iso_week)
        if ev["id"] not in seen:
            seen.add(ev["id"])
            upcoming.append(ev)
            if len(upcoming) >= 2:
                break
    for ev in upcoming:
        lines.append(f"  next: {ev['emoji']} {ev['name']} — {ev['blurb']}")
    lines.append("  seasons rotate weekly — plan your grind.")
    return "\n".join(lines)
