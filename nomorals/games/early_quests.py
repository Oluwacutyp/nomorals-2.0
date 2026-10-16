"""Early quests: the first three days of guided play.

Day 1: play 1 easy math, play 1 trivia, join 1 party -> chest.
Day 2: win 1 duel, complete 1 lesson, invite 1 friend -> chest.
Day 3: streak 3, hard math, group digest read -> chest.

Every chest pays coins + XP + one cosmetic. Chests are **never**
an investment, never a cash-out promise, never real money: the
coins are in-game only, the cosmetic is cosmetic, and the chest
text says so. There is no "invest to unlock" path anywhere here.

Mechanics:

* Progress is tracked per objective (``record_progress``), capped
  at the objective's target — extra completions do not overfill.
* A chest is claimable only when *every* objective of its quest is
  complete, and only once (``claim_chest``). Partial progress never
  pays.
* Chest payout goes through an injected ``reward_fn(player_key, xp,
  coins, reason)`` — the caller owns the economy — and is *also*
  written to the ``early_quest_ledger`` table, so every chest is
  auditable.
* Universal: objective kinds are game actions, not cultures.
  Trivia/lesson/digest *content* comes from the caller's pack
  (``pack_id`` / ``nation``); Nigeria is a default pack selection,
  never a boundary.

Objective ids are globally unique (``d1_...`` / ``d2_...`` /
``d3_...``) so one ``record_progress`` call can never credit the
wrong day's quest.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "EARLY_QUESTS", "CHEST_DISCLAIMER", "RewardFn",
    "early_quests", "quest_for_day", "quest_by_id",
    "objective_by_id", "record_progress", "quest_progress",
    "all_early_progress", "claim_chest", "chest_claimed",
    "quest_status_text", "ledger_entries", "ensure_early_quest_tables",
]

_log = get_logger(__name__)

#: Reward callback: (player_key, xp, coins, reason) -> anything.
RewardFn = Callable[[str, int, int, str], Any]

#: Printed on every chest. In-game coins are not money.
CHEST_DISCLAIMER = (
    "In-game coins and cosmetics only — never an investment, "
    "never real money, never a cash-out promise."
)


def _objective(objective_id: str, kind: str, description: str,
               target: int = 1) -> dict[str, Any]:
    return {
        "objective_id": objective_id,
        "kind": kind,
        "description": description,
        "target": int(target),
    }


def _quest(quest_id: str, day: int, title: str,
           objectives: list[dict[str, Any]],
           chest_coins: int, chest_xp: int,
           cosmetic_slug: str, cosmetic_name: str) -> dict[str, Any]:
    return {
        "quest_id": quest_id,
        "day": int(day),
        "title": title,
        "objectives": objectives,
        "chest": {
            "coins": int(chest_coins),
            "xp": int(chest_xp),
            "cosmetic": {
                "slug": cosmetic_slug,
                "name": cosmetic_name,
                "kind": "cosmetic",
                "power_attack": 0,
                "power_defense": 0,
                "power": 0,
                "tradeable": False,
                "cash_value": 0,
            },
            "disclaimer": CHEST_DISCLAIMER,
        },
    }


EARLY_QUESTS: tuple[dict[str, Any], ...] = (
    _quest(
        "early_day1", 1, "Day 1 — First steps",
        [
            _objective("d1_math_easy", "math_easy",
                       "Play 1 easy /math problem"),
            _objective("d1_trivia", "trivia",
                       "Play 1 trivia round"),
            _objective("d1_party_join", "party_join",
                       "Join 1 party"),
        ],
        chest_coins=60, chest_xp=60,
        cosmetic_slug="day1-ribbon", cosmetic_name="Day-1 Ribbon"),
    _quest(
        "early_day2", 2, "Day 2 — Prove it",
        [
            _objective("d2_duel_win", "duel_win",
                       "Win 1 duel"),
            _objective("d2_lesson", "lesson_complete",
                       "Complete 1 lesson"),
            _objective("d2_invite", "friend_invite",
                       "Invite 1 friend"),
        ],
        chest_coins=90, chest_xp=90,
        cosmetic_slug="day2-torch", cosmetic_name="Day-2 Torch"),
    _quest(
        "early_day3", 3, "Day 3 — Keep the fire",
        [
            _objective("d3_streak", "streak_3",
                       "Reach a 3-day streak", target=3),
            _objective("d3_math_hard", "math_hard",
                       "Play 1 hard /math problem"),
            _objective("d3_digest", "digest_read",
                       "Read 1 group digest"),
        ],
        chest_coins=120, chest_xp=120,
        cosmetic_slug="day3-crown", cosmetic_name="Day-3 Crown (cosmetic)"),
)

_OBJECTIVES: dict[str, tuple[str, dict[str, Any]]] = {}
for _q in EARLY_QUESTS:
    for _o in _q["objectives"]:
        _OBJECTIVES[_o["objective_id"]] = (_q["quest_id"], _o)


# ── storage ─────────────────────────────────────────────────────────────────

def ensure_early_quest_tables(db: Any) -> None:
    """Create the early-quest tables if they do not exist yet."""
    if db is None:
        return
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS early_quest_progress ("
            "player_key TEXT NOT NULL, quest_id TEXT NOT NULL, "
            "objective_id TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, "
            "updated_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, objective_id))")
        db.execute(
            "CREATE TABLE IF NOT EXISTS early_quest_chests ("
            "player_key TEXT NOT NULL, quest_id TEXT NOT NULL, "
            "claimed_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, quest_id))")
        db.execute(
            "CREATE TABLE IF NOT EXISTS early_quest_ledger ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "player_key TEXT NOT NULL, kind TEXT NOT NULL, "
            "xp INTEGER NOT NULL DEFAULT 0, "
            "coins INTEGER NOT NULL DEFAULT 0, "
            "detail TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL DEFAULT 0)")
    except Exception:  # noqa: BLE001
        _log.debug("early quest tables ensure failed", exc_info=True)


def _ledger(db: Any, player_key: str, kind: str, xp: int, coins: int,
            detail: str = "") -> None:
    ensure_early_quest_tables(db)
    if db is None:
        return
    try:
        db.execute(
            "INSERT INTO early_quest_ledger "
            "(player_key, kind, xp, coins, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (player_key, kind, int(xp), int(coins), detail, time.time()))
    except Exception:  # noqa: BLE001
        _log.debug("early quest ledger write failed", exc_info=True)


def ledger_entries(db: Any, player_key: str) -> list[dict[str, Any]]:
    """Audit trail of early-quest payouts for one player, oldest first."""
    ensure_early_quest_tables(db)
    if db is None:
        return []
    try:
        return list(db.query(
            "SELECT * FROM early_quest_ledger WHERE player_key = ? "
            "ORDER BY id ASC", (player_key,)))
    except Exception:  # noqa: BLE001
        _log.debug("early quest ledger read failed", exc_info=True)
        return []


# ── catalog ─────────────────────────────────────────────────────────────────

def early_quests() -> list[dict[str, Any]]:
    """Fresh deep-ish copy of the early-quest catalog (never shared)."""
    out: list[dict[str, Any]] = []
    for quest in EARLY_QUESTS:
        copy = dict(quest)
        copy["objectives"] = [dict(o) for o in quest["objectives"]]
        copy["chest"] = dict(quest["chest"])
        copy["chest"]["cosmetic"] = dict(quest["chest"]["cosmetic"])
        out.append(copy)
    return out


def quest_by_id(quest_id: str) -> dict[str, Any] | None:
    """One quest by id, or None."""
    for quest in early_quests():
        if quest["quest_id"] == quest_id:
            return quest
    return None


def quest_for_day(day: int) -> dict[str, Any] | None:
    """The early quest for day 1/2/3, or None for other days."""
    for quest in early_quests():
        if quest["day"] == int(day):
            return quest
    return None


def objective_by_id(objective_id: str) -> dict[str, Any] | None:
    """One objective by id, with its quest id attached, or None."""
    hit = _OBJECTIVES.get(objective_id)
    if hit is None:
        return None
    quest_id, objective = hit
    out = dict(objective)
    out["quest_id"] = quest_id
    return out


# ── progress ────────────────────────────────────────────────────────────────

def record_progress(db: Any, player_key: str, objective_id: str,
                    amount: int = 1,
                    now: float | None = None) -> dict[str, Any]:
    """Advance one objective by ``amount`` (default 1), capped at target.

    Returns the objective's new progress plus whether its quest just
    became fully complete (``quest_completed``) and whether its chest
    is now ready (``chest_ready``). Recording progress never pays —
    only :func:`claim_chest` pays. Unknown objectives are refused;
    over-target completions are capped, not banked.
    """
    hit = _OBJECTIVES.get(objective_id)
    if hit is None:
        return {"ok": False, "reason": "unknown_objective",
                "objective_id": objective_id}
    quest_id, objective = hit
    target = int(objective["target"])
    if db is None:
        return {"ok": False, "reason": "no_db"}
    ensure_early_quest_tables(db)
    try:
        row = db.query_one(
            "SELECT progress FROM early_quest_progress "
            "WHERE player_key = ? AND objective_id = ?",
            (player_key, objective_id))
        current = int(row["progress"]) if row is not None else 0
        new = min(target, current + max(0, int(amount)))
        db.execute(
            "INSERT INTO early_quest_progress "
            "(player_key, quest_id, objective_id, progress, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(player_key, objective_id) DO UPDATE SET "
            "progress = excluded.progress, updated_at = excluded.updated_at",
            (player_key, quest_id, objective_id, new,
             time.time() if now is None else float(now)))
    except Exception:  # noqa: BLE001
        _log.debug("early quest progress write failed", exc_info=True)
        return {"ok": False, "reason": "storage_error"}
    progress = quest_progress(db, player_key, quest_id)
    return {
        "ok": True,
        "objective_id": objective_id,
        "quest_id": quest_id,
        "progress": new,
        "target": target,
        "objective_done": new >= target,
        "quest_completed": progress["completed"],
        "chest_ready": progress["chest_ready"],
    }


def quest_progress(db: Any, player_key: str,
                   quest_id: str) -> dict[str, Any]:
    """Progress of one quest: per-objective progress, completion, chest."""
    quest = quest_by_id(quest_id)
    if quest is None:
        return {"ok": False, "reason": "unknown_quest", "quest_id": quest_id}
    stored: dict[str, int] = {}
    if db is not None:
        ensure_early_quest_tables(db)
        try:
            for row in db.query(
                    "SELECT objective_id, progress FROM early_quest_progress "
                    "WHERE player_key = ? AND quest_id = ?",
                    (player_key, quest_id)):
                stored[str(row["objective_id"])] = int(row["progress"])
        except Exception:  # noqa: BLE001
            _log.debug("early quest progress read failed", exc_info=True)
    objectives = []
    for obj in quest["objectives"]:
        made = stored.get(obj["objective_id"], 0)
        objectives.append({
            **obj,
            "progress": made,
            "done": made >= int(obj["target"]),
        })
    completed = all(o["done"] for o in objectives)
    claimed = chest_claimed(db, player_key, quest_id)
    return {
        "ok": True,
        "quest_id": quest_id,
        "day": quest["day"],
        "title": quest["title"],
        "objectives": objectives,
        "completed": completed,
        "chest_claimed": claimed,
        # Ready = fully complete AND not yet claimed.
        "chest_ready": completed and not claimed,
        "chest": quest["chest"],
    }


def all_early_progress(db: Any, player_key: str) -> list[dict[str, Any]]:
    """Progress for all three early quests, day order."""
    return [quest_progress(db, player_key, q["quest_id"])
            for q in EARLY_QUESTS]


# ── chest ───────────────────────────────────────────────────────────────────

def chest_claimed(db: Any, player_key: str, quest_id: str) -> bool:
    """True if this player's chest for this quest is already claimed."""
    if db is None:
        return False
    ensure_early_quest_tables(db)
    try:
        row = db.query_one(
            "SELECT 1 AS x FROM early_quest_chests "
            "WHERE player_key = ? AND quest_id = ?",
            (player_key, quest_id))
        return row is not None
    except Exception:  # noqa: BLE001
        _log.debug("chest claimed check failed", exc_info=True)
        return False


def claim_chest(db: Any, player_key: str, quest_id: str,
                reward_fn: RewardFn | None = None,
                now: float | None = None) -> dict[str, Any]:
    """Claim a quest's chest. Only on full completion, only once.

    Partial progress pays nothing (``ok: False, reason:
    "not_complete"``); a second claim pays nothing (``reason:
    "already_claimed"``). On success the reward is paid via
    ``reward_fn`` *and* recorded in the ledger in the same call —
    there is no deferred/pending payout state.
    """
    quest = quest_by_id(quest_id)
    if quest is None:
        return {"ok": False, "reason": "unknown_quest", "quest_id": quest_id}
    progress = quest_progress(db, player_key, quest_id)
    if not progress["completed"]:
        missing = [o["objective_id"] for o in progress["objectives"]
                   if not o["done"]]
        return {"ok": False, "reason": "not_complete",
                "quest_id": quest_id, "missing": missing,
                "chest": None}
    if progress["chest_claimed"]:
        return {"ok": False, "reason": "already_claimed",
                "quest_id": quest_id, "chest": None}
    chest = quest["chest"]
    xp, coins = int(chest["xp"]), int(chest["coins"])
    if db is not None:
        try:
            db.execute(
                "INSERT OR IGNORE INTO early_quest_chests "
                "(player_key, quest_id, claimed_at) VALUES (?, ?, ?)",
                (player_key, quest_id,
                 time.time() if now is None else float(now)))
        except Exception:  # noqa: BLE001
            _log.debug("chest claim failed", exc_info=True)
            return {"ok": False, "reason": "storage_error"}
    _ledger(db, player_key, f"chest:{quest_id}", xp, coins,
            f"Early quest chest: {quest['title']} "
            f"(+{chest['cosmetic']['name']}). {CHEST_DISCLAIMER}")
    if reward_fn is not None:
        reward_fn(player_key, xp, coins, f"chest:{quest_id}")
    return {
        "ok": True,
        "quest_id": quest_id,
        "chest": chest,
        "paid": {"xp": xp, "coins": coins,
                 "cosmetic": chest["cosmetic"]},
    }


def quest_status_text(db: Any, player_key: str, quest_id: str) -> str:
    """One-line status for a quest, e.g. for the profile screen."""
    progress = quest_progress(db, player_key, quest_id)
    if not progress.get("ok"):
        return f"Unknown quest {quest_id}."
    done = sum(1 for o in progress["objectives"] if o["done"])
    total = len(progress["objectives"])
    if progress["chest_claimed"]:
        return (f"{progress['title']}: chest claimed. {CHEST_DISCLAIMER}")
    if progress["chest_ready"]:
        return (f"{progress['title']}: {done}/{total} done — "
                "chest ready to claim!")
    return (f"{progress['title']}: {done}/{total} objectives done.")
