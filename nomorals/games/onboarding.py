"""New-player onboarding boost + tutorial quest chain.

Fresh-start onboarding for the game. A new player gets a **7-day
boost** — XP x2, coins x1.5, a first-win-of-the-day bonus, a starter
gear pack, and one free reroll — plus a 5-step tutorial chain that
teaches ``/math``, ``/mix``, duel, party, and quest *in that order*,
each step paying out immediately on completion.

Design rules (hard, not decorative):

* **7 days exactly.** Day 1 is the day the boost starts; day 7 is the
  last boosted day. On day 8 the multipliers are 1.0 — no tail, no
  grace extension, no "one more day".
* **Multipliers are exact.** XP x2.0, coins x1.5 while the boost is
  active. The anti-farming control is a *decaying daily cap on the
  boosted bonus* (see :func:`daily_boost_cap_xp`), never a quiet
  change to the advertised multipliers.
* **Starter gear is earned, not pay-to-win.** Every starter piece is
  cosmetic plus a *small* XP bonus (a few percent). Attack and
  defense are always 0 — starter gear can never buy power.
* **No dark patterns.** The profile line states the facts — which day
  of 7, the multipliers, how many tutorial quests are left, and the
  exact end date. No countdown pressure, no "hurry", no "you'll lose
  this forever", no fake scarcity.
* **Anti-farming: one boost per verified identity, not per alt.**
  A verified-identity fingerprint (hashed, never stored raw) may
  claim a boost exactly once. A second player key presenting the
  same fingerprint gets no boost. Caps + the decaying daily bonus
  cap bound what a farm can extract even from a legitimately
  verified identity.
* **Universal.** Nothing here is Nigeria-specific. Content packs
  (trivia, lessons) are selected by the caller via
  ``pack_id`` / ``nation``; Nigeria is a default pack selection,
  never a boundary — the same boost, tutorial, and quests run for
  every nation/culture.

Payouts go through an injected reward function so the game never
hardcodes an economy::

    def reward(player_key: str, xp: int, coins: int, reason: str) -> None: ...

Every payout is also written to the ``onboarding_ledger`` table, so
a player's onboarding earnings are auditable even when the caller
supplies no reward function.

Storage: functions accept the shared ``Database`` (``db.execute`` /
``db.query`` / ``db.query_one`` returning dict rows) and create their
own tables with ``CREATE TABLE IF NOT EXISTS``. They never raise on
storage errors — they log and return a ``{"ok": False, ...}`` result.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "BOOST_DAYS", "BOOST_SECONDS", "XP_MULTIPLIER", "COIN_MULTIPLIER",
    "FIRST_WIN_OF_DAY_BONUS_XP", "FIRST_WIN_OF_DAY_BONUS_COINS",
    "STARTER_GEAR_PACK", "TUTORIAL_STEPS", "RewardFn",
    "normalize_fingerprint", "fingerprint_claimed_by",
    "register_identity", "start_boost", "get_boost",
    "boost_day", "is_boost_active", "xp_multiplier_for_day",
    "coin_multiplier_for_day", "daily_boost_cap_xp",
    "daily_boost_cap_coins", "apply_boost",
    "apply_boost_for_player", "claim_first_win_of_day",
    "first_win_claimed_today", "starter_gear_pack",
    "grant_starter_gear", "starter_gear_granted",
    "free_reroll_available", "claim_free_reroll",
    "tutorial_steps", "tutorial_progress", "complete_tutorial_step",
    "boost_profile_line", "profile_boost_summary",
    "onboarding_status", "ledger_entries", "ensure_onboarding_tables",
]

_log = get_logger(__name__)

#: Boost length in days — exactly 7, no tail.
BOOST_DAYS = 7
BOOST_SECONDS = BOOST_DAYS * 24 * 3600

#: Advertised multipliers while the boost is active. Exact.
XP_MULTIPLIER = 2.0
COIN_MULTIPLIER = 1.5

#: First win of the day, while the boost is active.
FIRST_WIN_OF_DAY_BONUS_XP = 50
FIRST_WIN_OF_DAY_BONUS_COINS = 100

#: Reward callback: (player_key, xp, coins, reason) -> anything.
RewardFn = Callable[[str, int, int, str], Any]


# ── starter gear ────────────────────────────────────────────────────────────
# Cosmetic + small XP bonus only. ``power_attack`` / ``power_defense``
# are 0 on every piece by construction — see _STARTER_GEAR_VALIDATED.

def _starter_piece(slug: str, name: str, slot: str, cosmetic: str,
                   xp_bonus_pct: float) -> dict[str, Any]:
    return {
        "slug": slug,
        "name": name,
        "slot": slot,
        "kind": "starter-cosmetic",
        "cosmetic": cosmetic,
        # Earned, never bought, never power.
        "power_attack": 0,
        "power_defense": 0,
        "power": 0,
        "xp_bonus_pct": float(xp_bonus_pct),
        "tradeable": False,
        "cash_value": 0,
        "description": (
            f"{cosmetic}. Earned starter gear — cosmetic + "
            f"+{xp_bonus_pct:g}% XP. No combat power, never pay-to-win."
        ),
    }


STARTER_GEAR_PACK: tuple[dict[str, Any], ...] = (
    _starter_piece("starter-ribbon", "Welcome Ribbon", "trinket",
                   "A bright welcome ribbon", 5.0),
    _starter_piece("starter-cap", "Rookie Cap", "armor",
                   "A cap that says 'new here, learning fast'", 3.0),
    _starter_piece("starter-charm", "First-Steps Charm", "trinket",
                   "A small charm for your first week", 3.0),
    _starter_piece("starter-banner", "Newcomer Banner", "trinket",
                   "A banner in your colours", 2.0),
)

# Total starter XP bonus stays small (<= 15%) by construction.
_STARTER_XP_TOTAL = sum(p["xp_bonus_pct"] for p in STARTER_GEAR_PACK)
assert _STARTER_XP_TOTAL <= 15.0, "starter XP bonus must stay small"
assert all(p["power_attack"] == 0 and p["power_defense"] == 0
           and p["power"] == 0 for p in STARTER_GEAR_PACK), \
    "starter gear must never grant power"


# ── tutorial chain ──────────────────────────────────────────────────────────
# Order is the teaching order: /math first (solo, safe), /mix second
# (crafting), duel third (PvP), party fourth (social), quest fifth
# (the loop that ties them together). Order is enforced, not suggested.

def _tutorial_step(position: int, step_id: str, command: str, title: str,
                   description: str, xp: int, coins: int) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "command": command,
        "title": title,
        "description": description,
        "reward_xp": int(xp),
        "reward_coins": int(coins),
        "position": int(position),
    }


TUTORIAL_STEPS_ORDER: tuple[str, ...] = (
    "tutorial_math", "tutorial_mix", "tutorial_duel",
    "tutorial_party", "tutorial_quest",
)

TUTORIAL_STEPS: tuple[dict[str, Any], ...] = (
    _tutorial_step(
        1,         "tutorial_math", "/math", "Learn /math",
        "Play one easy /math problem. Solo, no stakes — learn the shape.",
        20, 10),
    _tutorial_step(
        2,         "tutorial_mix", "/mix", "Learn /mix",
        "Use /mix once. Combine what you have; see how crafting works.",
        20, 10),
    _tutorial_step(
        3,         "tutorial_duel", "duel", "Try a duel",
        "Play one duel against another player or the house. Win or lose, you learn.",
        30, 20),
    _tutorial_step(
        4,         "tutorial_party", "party", "Join a party",
        "Join one party. Games are better with people.",
        30, 20),
    _tutorial_step(
        5,         "tutorial_quest", "quest", "Take a quest",
        "Accept and finish one quest. This is the loop everything else feeds.",
        40, 30),
)


# ── storage ─────────────────────────────────────────────────────────────────

def ensure_onboarding_tables(db: Any) -> None:
    """Create the onboarding tables if they do not exist yet."""
    if db is None:
        return
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_boosts ("
            "player_key TEXT PRIMARY KEY, "
            "started_at REAL NOT NULL, ends_at REAL NOT NULL, "
            "fingerprint_hash TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL DEFAULT 0)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_identities ("
            "fingerprint_hash TEXT PRIMARY KEY, "
            "player_key TEXT NOT NULL, "
            "claimed_at REAL NOT NULL DEFAULT 0)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_ledger ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "player_key TEXT NOT NULL, kind TEXT NOT NULL, "
            "xp INTEGER NOT NULL DEFAULT 0, "
            "coins INTEGER NOT NULL DEFAULT 0, "
            "detail TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL DEFAULT 0)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_tutorial ("
            "player_key TEXT NOT NULL, step_id TEXT NOT NULL, "
            "completed_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, step_id))")
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_first_wins ("
            "player_key TEXT NOT NULL, win_date TEXT NOT NULL, "
            "claimed_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, win_date))")
        db.execute(
            "CREATE TABLE IF NOT EXISTS onboarding_grants ("
            "player_key TEXT NOT NULL, grant_id TEXT NOT NULL, "
            "granted_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, grant_id))")
    except Exception:  # noqa: BLE001 - onboarding must never crash the game
        _log.debug("onboarding tables ensure failed", exc_info=True)


def _ledger(db: Any, player_key: str, kind: str, xp: int, coins: int,
            detail: str = "") -> None:
    ensure_onboarding_tables(db)
    if db is None:
        return
    try:
        db.execute(
            "INSERT INTO onboarding_ledger "
            "(player_key, kind, xp, coins, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (player_key, kind, int(xp), int(coins), detail, time.time()))
    except Exception:  # noqa: BLE001
        _log.debug("onboarding ledger write failed", exc_info=True)


def ledger_entries(db: Any, player_key: str) -> list[dict[str, Any]]:
    """Audit trail of onboarding payouts for one player, oldest first."""
    ensure_onboarding_tables(db)
    if db is None:
        return []
    try:
        return list(db.query(
            "SELECT * FROM onboarding_ledger WHERE player_key = ? "
            "ORDER BY id ASC", (player_key,)))
    except Exception:  # noqa: BLE001
        _log.debug("onboarding ledger read failed", exc_info=True)
        return []


def _now(now: float | None) -> float:
    return float(now) if now is not None else time.time()


def _date_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# ── verified-identity fingerprint ───────────────────────────────────────────
# The fingerprint is a caller-supplied verified-identity token: e.g.
# "telegram:123456" for a Telegram-verified account, or a verified
# phone/email identity string. It is SHA-256 hashed before storage;
# the raw identity is never persisted here.

def normalize_fingerprint(raw: str | None) -> str | None:
    """Hash a verified-identity token. None/blank -> None (unverified)."""
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def fingerprint_claimed_by(db: Any, fingerprint: str | None) -> str | None:
    """Player key that already claimed this fingerprint, or None."""
    fp = normalize_fingerprint(fingerprint)
    if fp is None or db is None:
        return None
    ensure_onboarding_tables(db)
    try:
        row = db.query_one(
            "SELECT player_key FROM onboarding_identities "
            "WHERE fingerprint_hash = ?", (fp,))
    except Exception:  # noqa: BLE001
        _log.debug("fingerprint lookup failed", exc_info=True)
        return None
    if row is None:
        return None
    return str(row["player_key"])


def register_identity(db: Any, fingerprint: str | None,
                      player_key: str, now: float | None = None) -> dict[str, Any]:
    """Claim a fingerprint for a player key.

    Returns ``{"ok": True}`` when the fingerprint is new or already
    belongs to this same player key, else ``{"ok": False,
    "reason": "fingerprint_already_used", "claimed_by": ...}``.
    Unverified (no fingerprint) identities are never registered —
    they simply get no anti-farming claim, and callers that require
    verification should treat ``ok: False, reason: unverified`` as
    "boost not available until verified".
    """
    fp = normalize_fingerprint(fingerprint)
    if fp is None:
        return {"ok": False, "reason": "unverified",
                "detail": "No verified identity supplied; boost requires verification."}
    if db is None:
        return {"ok": False, "reason": "no_db"}
    ensure_onboarding_tables(db)
    owner = fingerprint_claimed_by(db, fingerprint)
    if owner is not None and owner != player_key:
        return {"ok": False, "reason": "fingerprint_already_used",
                "claimed_by": owner,
                "detail": "This verified identity already claimed a boost."}
    try:
        db.execute(
            "INSERT OR IGNORE INTO onboarding_identities "
            "(fingerprint_hash, player_key, claimed_at) VALUES (?, ?, ?)",
            (fp, player_key, _now(now)))
    except Exception:  # noqa: BLE001
        _log.debug("identity register failed", exc_info=True)
        return {"ok": False, "reason": "storage_error"}
    return {"ok": True, "fingerprint_hash": fp}


# ── boost lifecycle ─────────────────────────────────────────────────────────

def boost_day(started_at: float, now: float | None = None) -> int:
    """1-based boost day for a start timestamp. 0 = not started yet."""
    ts = _now(now)
    if ts < started_at:
        return 0
    return int((ts - started_at) // 86400) + 1


def is_boost_active(started_at: float, now: float | None = None) -> bool:
    """True on days 1..7 inclusive; False on day 8+."""
    return 1 <= boost_day(started_at, now) <= BOOST_DAYS


def xp_multiplier_for_day(day: int) -> float:
    """Exact XP multiplier: 2.0 on days 1-7, 1.0 otherwise."""
    return XP_MULTIPLIER if 1 <= int(day) <= BOOST_DAYS else 1.0


def coin_multiplier_for_day(day: int) -> float:
    """Exact coin multiplier: 1.5 on days 1-7, 1.0 otherwise."""
    return COIN_MULTIPLIER if 1 <= int(day) <= BOOST_DAYS else 1.0


def daily_boost_cap_xp(day: int) -> int:
    """Daily cap on *boosted bonus* XP — decays so farms taper out.

    Day 1: 1000 bonus XP, then -100/day to 400 on day 7. Day 8+ the
    cap is 0 because there is no boost. The cap bounds the *bonus*
    (the extra XP above base), never the base earnings themselves,
    and never changes the advertised x2 multiplier.
    """
    if not 1 <= int(day) <= BOOST_DAYS:
        return 0
    return 1000 - (int(day) - 1) * 100


def daily_boost_cap_coins(day: int) -> int:
    """Daily cap on *boosted bonus* coins — decays 500 -> 200."""
    if not 1 <= int(day) <= BOOST_DAYS:
        return 0
    return 500 - (int(day) - 1) * 50


def apply_boost(base_xp: int, base_coins: int, *, day: int,
                already_boosted_xp: int = 0,
                already_boosted_coins: int = 0) -> dict[str, Any]:
    """Apply the boost multipliers to base earnings, honouring the cap.

    Pure function (no DB): ``already_boosted_*`` is how much bonus
    the player already banked today, so the decaying cap is enforced
    across multiple payouts in one day.
    """
    base_xp = max(0, int(base_xp))
    base_coins = max(0, int(base_coins))
    xp_mult = xp_multiplier_for_day(day)
    coin_mult = coin_multiplier_for_day(day)
    raw_bonus_xp = int(round(base_xp * (xp_mult - 1.0)))
    raw_bonus_coins = int(round(base_coins * (coin_mult - 1.0)))
    room_xp = max(0, daily_boost_cap_xp(day) - max(0, int(already_boosted_xp)))
    room_coins = max(0, daily_boost_cap_coins(day) - max(0, int(already_boosted_coins)))
    bonus_xp = min(raw_bonus_xp, room_xp)
    bonus_coins = min(raw_bonus_coins, room_coins)
    return {
        "base_xp": base_xp, "base_coins": base_coins,
        "xp_multiplier": xp_mult, "coin_multiplier": coin_mult,
        "bonus_xp": bonus_xp, "bonus_coins": bonus_coins,
        "capped_xp": raw_bonus_xp - bonus_xp,
        "capped_coins": raw_bonus_coins - bonus_coins,
        "total_xp": base_xp + bonus_xp,
        "total_coins": base_coins + bonus_coins,
        "day": int(day),
    }


def start_boost(db: Any, player_key: str, fingerprint: str | None = None,
                now: float | None = None) -> dict[str, Any]:
    """Start (or return the existing) boost for a player.

    When ``fingerprint`` is given, the boost is claimed against that
    verified identity: a second player key with the same fingerprint
    gets ``{"ok": False, "reason": "fingerprint_already_used"}`` and
    no boost. When no fingerprint is given, the boost is keyed to
    the player key alone (callers that need anti-farming must pass
    the verified fingerprint).
    """
    ts = _now(now)
    if db is None:
        return {"ok": False, "reason": "no_db"}
    ensure_onboarding_tables(db)
    existing = get_boost(db, player_key, now=ts)
    if existing is not None:
        return {"ok": True, "already": True, "boost": existing}
    if fingerprint is not None:
        reg = register_identity(db, fingerprint, player_key, now=ts)
        if not reg["ok"]:
            return {"ok": False, "reason": reg["reason"],
                    "claimed_by": reg.get("claimed_by"),
                    "detail": reg.get("detail", "")}
    ends = ts + BOOST_SECONDS
    try:
        db.execute(
            "INSERT OR IGNORE INTO onboarding_boosts "
            "(player_key, started_at, ends_at, fingerprint_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (player_key, ts, ends,
             normalize_fingerprint(fingerprint) or "", ts))
    except Exception:  # noqa: BLE001
        _log.debug("boost start failed", exc_info=True)
        return {"ok": False, "reason": "storage_error"}
    boost = get_boost(db, player_key, now=ts)
    return {"ok": True, "already": False, "boost": boost}


def get_boost(db: Any, player_key: str,
              now: float | None = None) -> dict[str, Any] | None:
    """Boost state for a player, or None when they never started one."""
    if db is None:
        return None
    ensure_onboarding_tables(db)
    try:
        row = db.query_one(
            "SELECT * FROM onboarding_boosts WHERE player_key = ?",
            (player_key,))
    except Exception:  # noqa: BLE001
        _log.debug("boost read failed", exc_info=True)
        return None
    if row is None:
        return None
    ts = _now(now)
    started = float(row["started_at"])
    ends = float(row["ends_at"])
    day = boost_day(started, ts)
    active = 1 <= day <= BOOST_DAYS
    return {
        "player_key": player_key,
        "started_at": started,
        "ends_at": ends,
        "end_date": datetime.fromtimestamp(ends).strftime("%Y-%m-%d"),
        "day": day,
        "active": active,
        "xp_multiplier": xp_multiplier_for_day(day) if active else 1.0,
        "coin_multiplier": coin_multiplier_for_day(day) if active else 1.0,
        "days_left": max(0, BOOST_DAYS - day + 1) if active else 0,
    }


def apply_boost_for_player(db: Any, player_key: str, base_xp: int,
                           base_coins: int,
                           now: float | None = None) -> dict[str, Any]:
    """Boost-aware payout for one player, cap-aware across the day.

    Sums today's already-banked boosted bonus from the ledger so the
    decaying daily cap holds across many small payouts, not just one.
    """
    boost = get_boost(db, player_key, now=now)
    day = boost["day"] if boost is not None else 0
    already_xp = already_coins = 0
    if boost is not None and boost["active"] and db is not None:
        today = _date_of(_now(now))
        try:
            for entry in ledger_entries(db, player_key):
                if entry.get("kind") == "boost_bonus" and _date_of(
                        float(entry.get("created_at") or 0)) == today:
                    already_xp += int(entry.get("xp") or 0)
                    already_coins += int(entry.get("coins") or 0)
        except Exception:  # noqa: BLE001
            _log.debug("boost cap ledger scan failed", exc_info=True)
    result = apply_boost(base_xp, base_coins, day=day,
                         already_boosted_xp=already_xp,
                         already_boosted_coins=already_coins)
    if result["bonus_xp"] or result["bonus_coins"]:
        _ledger(db, player_key, "boost_bonus",
                result["bonus_xp"], result["bonus_coins"],
                f"boost day {day} bonus (cap-aware)")
    return result


# ── first win of the day ────────────────────────────────────────────────────

def first_win_claimed_today(db: Any, player_key: str,
                            now: float | None = None) -> bool:
    """True if today's first-win bonus is already claimed."""
    if db is None:
        return False
    ensure_onboarding_tables(db)
    try:
        row = db.query_one(
            "SELECT 1 AS x FROM onboarding_first_wins "
            "WHERE player_key = ? AND win_date = ?",
            (player_key, _date_of(_now(now))))
        return row is not None
    except Exception:  # noqa: BLE001
        _log.debug("first-win check failed", exc_info=True)
        return False


def claim_first_win_of_day(db: Any, player_key: str,
                           reward_fn: RewardFn | None = None,
                           now: float | None = None) -> dict[str, Any]:
    """Claim today's first-win bonus. Once per player per calendar day,
    only while the boost is active. Second claim the same day pays 0.
    """
    ts = _now(now)
    boost = get_boost(db, player_key, now=ts)
    if boost is None or not boost["active"]:
        return {"ok": False, "reason": "boost_not_active",
                "xp": 0, "coins": 0}
    if first_win_claimed_today(db, player_key, now=ts):
        return {"ok": False, "reason": "already_claimed_today",
                "xp": 0, "coins": 0}
    if db is not None:
        try:
            db.execute(
                "INSERT OR IGNORE INTO onboarding_first_wins "
                "(player_key, win_date, claimed_at) VALUES (?, ?, ?)",
                (player_key, _date_of(ts), ts))
        except Exception:  # noqa: BLE001
            _log.debug("first-win claim failed", exc_info=True)
            return {"ok": False, "reason": "storage_error", "xp": 0, "coins": 0}
    xp, coins = FIRST_WIN_OF_DAY_BONUS_XP, FIRST_WIN_OF_DAY_BONUS_COINS
    _ledger(db, player_key, "first_win_of_day", xp, coins, "first win of day")
    if reward_fn is not None:
        reward_fn(player_key, xp, coins, "first_win_of_day")
    return {"ok": True, "xp": xp, "coins": coins,
            "day": boost["day"]}


# ── starter gear + free reroll ──────────────────────────────────────────────

def starter_gear_pack() -> list[dict[str, Any]]:
    """A fresh copy of the starter gear pack (never the shared tuple)."""
    return [dict(item) for item in STARTER_GEAR_PACK]


def starter_gear_granted(db: Any, player_key: str) -> bool:
    """True if this player already received the starter gear pack."""
    if db is None:
        return False
    ensure_onboarding_tables(db)
    try:
        row = db.query_one(
            "SELECT 1 AS x FROM onboarding_grants "
            "WHERE player_key = ? AND grant_id = 'starter_gear'",
            (player_key,))
        return row is not None
    except Exception:  # noqa: BLE001
        _log.debug("starter gear check failed", exc_info=True)
        return False


def grant_starter_gear(db: Any, player_key: str,
                       grant_fn: Callable[[str, dict[str, Any]], Any] | None = None
                       ) -> dict[str, Any]:
    """Grant the starter gear pack exactly once. Idempotent.

    ``grant_fn(player_key, item)`` is called once per piece when
    supplied (e.g. to write into the real gear store); the pack is
    recorded in ``onboarding_grants`` either way, so a second call
    returns ``already: True`` and grants nothing.
    """
    if starter_gear_granted(db, player_key):
        return {"ok": True, "already": True, "granted": []}
    pack = starter_gear_pack()
    if grant_fn is not None:
        for item in pack:
            grant_fn(player_key, item)
    if db is not None:
        try:
            db.execute(
                "INSERT OR IGNORE INTO onboarding_grants "
                "(player_key, grant_id, granted_at) VALUES (?, ?, ?)",
                (player_key, "starter_gear", time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("starter gear grant record failed", exc_info=True)
    _ledger(db, player_key, "starter_gear", 0, 0,
            "starter gear pack: " + ",".join(i["slug"] for i in pack))
    return {"ok": True, "already": False, "granted": pack}


def free_reroll_available(db: Any, player_key: str) -> bool:
    """True until the one free first reroll is claimed."""
    if db is None:
        return False
    ensure_onboarding_tables(db)
    try:
        row = db.query_one(
            "SELECT 1 AS x FROM onboarding_grants "
            "WHERE player_key = ? AND grant_id = 'free_reroll'",
            (player_key,))
        return row is None
    except Exception:  # noqa: BLE001
        _log.debug("free reroll check failed", exc_info=True)
        return False


def claim_free_reroll(db: Any, player_key: str) -> dict[str, Any]:
    """Claim the one free first reroll. Second claim is refused."""
    if not free_reroll_available(db, player_key):
        return {"ok": False, "reason": "already_claimed"}
    if db is not None:
        try:
            db.execute(
                "INSERT OR IGNORE INTO onboarding_grants "
                "(player_key, grant_id, granted_at) VALUES (?, ?, ?)",
                (player_key, "free_reroll", time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("free reroll claim failed", exc_info=True)
            return {"ok": False, "reason": "storage_error"}
    _ledger(db, player_key, "free_reroll", 0, 0, "free first reroll")
    return {"ok": True, "rerolls": 1}


# ── tutorial chain ──────────────────────────────────────────────────────────

def tutorial_steps() -> list[dict[str, Any]]:
    """The tutorial chain in teaching order (a fresh copy)."""
    return [dict(step) for step in TUTORIAL_STEPS]


def tutorial_progress(db: Any, player_key: str) -> dict[str, Any]:
    """Tutorial progress: completed steps, next step, remaining count."""
    completed: list[str] = []
    if db is not None:
        ensure_onboarding_tables(db)
        try:
            rows = db.query(
                "SELECT step_id FROM onboarding_tutorial "
                "WHERE player_key = ? ORDER BY completed_at ASC",
                (player_key,))
            completed = [str(r["step_id"]) for r in rows]
        except Exception:  # noqa: BLE001
            _log.debug("tutorial progress read failed", exc_info=True)
    # Only the in-order prefix counts: a step is "done" when every
    # step before it is also done (order is enforced on completion).
    done: list[str] = []
    for step_id in TUTORIAL_STEPS_ORDER:
        if step_id in completed:
            done.append(step_id)
        else:
            break
    remaining = [s for s in tutorial_steps()
                 if s["step_id"] not in done]
    nxt = remaining[0] if remaining else None
    return {
        "completed": done,
        "next": nxt,
        "next_step_id": nxt["step_id"] if nxt else None,
        "remaining": len(remaining),
        "total": len(TUTORIAL_STEPS_ORDER),
        "done": not remaining,
    }


def complete_tutorial_step(db: Any, player_key: str, step_id: str,
                           reward_fn: RewardFn | None = None,
                           now: float | None = None) -> dict[str, Any]:
    """Complete one tutorial step — in order — and pay out immediately.

    Out-of-order attempts are refused (``ok: False, reason:
    "out_of_order"``) with the expected next step named; replays of a
    completed step pay nothing. On success the reward is paid via
    ``reward_fn`` *and* recorded in the ledger in the same call, so
    "each step pays out immediately" is literally true — there is no
    deferred/pending payout state anywhere in this module.
    """
    progress = tutorial_progress(db, player_key)
    if step_id not in TUTORIAL_STEPS_ORDER:
        return {"ok": False, "reason": "unknown_step", "step_id": step_id,
                "expected": progress["next_step_id"]}
    if step_id in progress["completed"]:
        return {"ok": False, "reason": "already_completed",
                "step_id": step_id, "xp": 0, "coins": 0,
                "next": progress["next"]}
    if step_id != progress["next_step_id"]:
        return {"ok": False, "reason": "out_of_order", "step_id": step_id,
                "expected": progress["next_step_id"],
                "detail": f"Complete {progress['next_step_id']} first."}
    step = next(s for s in TUTORIAL_STEPS if s["step_id"] == step_id)
    xp, coins = int(step["reward_xp"]), int(step["reward_coins"])
    if db is not None:
        try:
            db.execute(
                "INSERT OR IGNORE INTO onboarding_tutorial "
                "(player_key, step_id, completed_at) VALUES (?, ?, ?)",
                (player_key, step_id, _now(now)))
        except Exception:  # noqa: BLE001
            _log.debug("tutorial complete failed", exc_info=True)
            return {"ok": False, "reason": "storage_error"}
    _ledger(db, player_key, f"tutorial:{step_id}", xp, coins,
            f"Tutorial: {step['title']} — paid immediately")
    if reward_fn is not None:
        reward_fn(player_key, xp, coins, f"tutorial:{step_id}")
    after = tutorial_progress(db, player_key)
    return {"ok": True, "step_id": step_id, "xp": xp, "coins": coins,
            "paid": {"xp": xp, "coins": coins},
            "next": after["next"],
            "remaining": after["remaining"],
            "tutorial_done": after["done"]}


# ── profile surface ─────────────────────────────────────────────────────────

def boost_profile_line(db: Any, player_key: str,
                       quests_left: int | None = None,
                       now: float | None = None) -> str:
    """One factual profile line about the boost. No dark patterns.

    Active: ``"Day 1/7 boost: XP x2, 3 quests left — ends 2026-10-23"``.
    The end date is always shown; there is no countdown, no urgency
    language, no "hurry" — the player can read the facts and decide.
    """
    boost = get_boost(db, player_key, now=now)
    if boost is None:
        return "No new-player boost yet — it starts on your first day."
    if not boost["active"]:
        return (f"New-player boost ended {boost['end_date']} — "
                "your XP and coins now earn at the normal rate.")
    quests = (f", {int(quests_left)} quests left" if quests_left is not None
              else "")
    return (f"Day {boost['day']}/7 boost: XP x2, coins x1.5{quests} "
            f"— ends {boost['end_date']}")


def profile_boost_summary(db: Any, player_key: str,
                          now: float | None = None) -> dict[str, Any]:
    """Structured profile block: boost + tutorial + reroll + gear."""
    boost = get_boost(db, player_key, now=now)
    tut = tutorial_progress(db, player_key)
    return {
        "boost": boost,
        "profile_line": boost_profile_line(
            db, player_key, quests_left=tut["remaining"], now=now),
        "tutorial": tut,
        "free_reroll_available": free_reroll_available(db, player_key),
        "starter_gear_granted": starter_gear_granted(db, player_key),
        "first_win_claimed_today": first_win_claimed_today(
            db, player_key, now=now),
    }


def onboarding_status(db: Any, player_key: str,
                      now: float | None = None) -> dict[str, Any]:
    """Everything the profile screen needs, in one call."""
    return profile_boost_summary(db, player_key, now=now)
