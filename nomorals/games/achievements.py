"""Achievements system: cross-game unlocks that persist.

Each achievement has an id, name, description, and optional rarity tier.
The engine awards achievements on game events (win, score threshold, etc.).
Players can view their unlocks with ``nm achievements`` or in-game.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..storage.db import Database

__all__ = ["Achievement", "ACHIEVEMENTS", "ACHIEVEMENT_UNLOCKS",
           "UNLOCK_SOURCE", "unlock_achievement", "get_achievements",
           "get_unlocks", "has_unlock", "progression_capabilities",
           # live rarity / progress / repeatables / showcase
           "RARITY_ORDER", "RARITY_EMOJI", "TRACKED_ACHIEVEMENTS",
           "REPEATABLE_MILESTONES", "SHOWCASE_SIZE",
           "live_rarity", "rarity_badge", "track_progress", "get_progress",
           "progress_text", "repeatable_check",
           "achievement_anniversaries", "set_showcase", "get_showcase",
           "render_showcase"]


@dataclass(frozen=True)
class Achievement:
    """One unlockable achievement."""
    id: str
    name: str
    description: str
    rarity: str = "common"  # common, rare, epic, legendary


# ── catalog ──────────────────────────────────────────────────────────────────

ACHIEVEMENTS: tuple[Achievement, ...] = (
    # 2048
    Achievement("2048_win", "Tile Master", "Reach 2048", "rare"),
    Achievement("2048_4096", "Beyond 2048", "Reach 4096", "epic"),
    Achievement("2048_score_5k", "High Scorer", "Score 5000+ in 2048", "common"),
    # Snake
    Achievement("snake_50", "Growing Up", "Reach score 50 in Snake", "common"),
    Achievement("snake_200", "Long Boi", "Reach score 200 in Snake", "rare"),
    Achievement("snake_no_crash_20", "Careful Mover", "Survive 20 moves without crashing", "common"),
    # Connect Four
    Achievement("connect4_win", "Four in a Row", "Win a game of Connect Four", "common"),
    Achievement("connect4_quick", "Speed Demon", "Win Connect Four in under 10 moves", "rare"),
    # Battleship
    Achievement("battleship_win", "Admiral", "Sink all enemy ships", "common"),
    Achievement("battleship_perfect", "Sharpshooter", "Win Battleship with 80%+ accuracy", "epic"),
    # Poker
    Achievement("poker_win", "Card Shark", "Win a hand of Poker", "common"),
    Achievement("poker_straight", "Straight Shooter", "Make a straight or better", "rare"),
    Achievement("poker_allin_win", "All-In Legend", "Win after going all-in", "epic"),
    # Hangman
    Achievement("hangman_5_wins", "Word Wizard", "Win 5 Hangman games", "common"),
    Achievement("hangman_perfect", "No Mistakes", "Win Hangman with 0 wrong guesses", "rare"),
    # Case game
    Achievement("case_first", "Rookie Sleuth", "Solve your first case", "common"),
    Achievement("case_streak_3", "Hot Trail", "Solve 3 cases in a row", "rare"),
    Achievement("case_streak_5", "Untouchable", "Solve 5 cases in a row", "epic"),
    Achievement("case_expert", "Master Detective", "Solve an expert-tier case", "epic"),
    Achievement("case_clean", "Clean Solve", "Solve a case with no strikes and no hints", "rare"),
    Achievement("case_timed", "Beat the Clock", "Solve a timed case with a speed bonus", "rare"),
    # World
    Achievement("world_50_pop", "Booming Town", "Reach 50 population in World", "rare"),
    Achievement("world_all_buildings", "Master Builder", "Build every type of building", "epic"),
    # Arena
    Achievement("arena_win", "Gladiator", "Win a Battle Arena match", "common"),
    Achievement("arena_crit_kill", "Critical Finish", "Win Arena with a crit", "rare"),
    Achievement("arena_streak_5", "On Fire", "Win 5 arena battles in a row", "rare"),
    Achievement("arena_streak_10", "Unstoppable", "Win 10 arena battles in a row", "epic"),
    Achievement("arena_streak_15", "Relentless", "Win 15 arena battles in a row", "epic"),
    Achievement("arena_streak_20", "War Machine", "Win 20 arena battles in a row", "legendary"),
    Achievement("arena_streak_25", "Immortal", "Win 25 arena battles in a row", "legendary"),
    Achievement("arena_s_rank", "Dragonslayer", "Defeat an S-rank hunter", "epic"),
    Achievement("arena_ss_rank", "Stormbreaker", "Defeat an SS-rank hunter", "epic"),
    Achievement("arena_x_rank", "Legend Killer", "Defeat an X-rank hunter", "legendary"),
    Achievement("arena_myth_foe", "Mythslayer", "Defeat a myth-foe hunter", "epic"),
    Achievement("arena_flawless", "Untouched", "Win an arena battle without taking damage", "rare"),
    Achievement("arena_upset", "Giant Slayer", "Defeat a stronger opponent in the arena", "rare"),
    Achievement("arena_skill_kill", "Technique Finish", "Win an arena battle with a skill as the killing blow", "common"),
    Achievement("arena_brutal", "Brutal", "Land a brutal finish (2× overkill) in the arena", "rare"),
    Achievement("arena_forbidden", "Seen the Forbidden", "Survive an enemy's forbidden technique", "common"),
    Achievement("arena_pvp_win", "Duelist", "Win a PvP duel", "rare"),
    Achievement("arena_pvp_streak_3", "Duel Master", "Win 3 PvP duels in a row", "rare"),
    Achievement("arena_raid_win", "Boss Hunter", "Defeat a raid boss", "epic"),
    Achievement("arena_raid_mvp", "Raid MVP", "Deal the most damage in a raid", "epic"),
    Achievement("arena_100_wins", "Centurion", "Win 100 arena battles", "epic"),
    Achievement("arena_tier3", "Master of Arts", "Upgrade a skill to tier III", "epic"),
    Achievement("arena_myth_set", "Myth Forged", "Equip a full myth-tier set", "epic"),
    Achievement("arena_no_potion", "Purist", "Win an arena battle without using a potion", "rare"),
    Achievement("arena_comeback", "From the Brink", "Win an arena battle from below 20% HP", "rare"),
    # ── unlikely scenarios: the strange glories ──
    Achievement("arena_1hp_win", "Last Breath", "Win an arena battle with exactly 1 HP", "epic"),
    Achievement("arena_lose_10", "Never Give Up", "Lose 10 arena battles in a row", "rare"),
    Achievement("arena_underdog_5", "Underdog Spirit", "Win 5 times as the weaker fighter", "epic"),
    Achievement("arena_skills_only", "Pure Technique", "Win using only skills, never a basic attack", "epic"),
    Achievement("arena_fast_win", "Blitz", "Win an arena battle in 3 turns or fewer", "rare"),
    Achievement("arena_marathon", "Endurance", "Survive 20 turns in one arena battle", "rare"),
    Achievement("arena_dual_10", "Weaver", "Land 10 successful dual-casts", "epic"),
    Achievement("arena_lucky_dual", "Against All Odds", "Win a dual-cast with under 30% success", "legendary"),
    Achievement("arena_dual_kill", "Twin Fang Finish", "Win with a dual-cast as the killing blow", "rare"),
    Achievement("arena_mana_starved", "On Empty", "Win after running out of mana", "rare"),
    Achievement("arena_no_gear", "Barehanded", "Win with no gear equipped", "epic"),
    Achievement("arena_perfect_dual", "Perfect Weave", "Land a dual-cast at 95%+ success", "common"),
    # RPG
    Achievement("rpg_finish", "Campaign Complete", "Finish the RPG campaign", "common"),
    Achievement("rpg_level_5", "Veteran", "Reach level 5 in the RPG", "rare"),
    # General
    Achievement("games_10", "Game Night", "Play 10 games", "common"),
    Achievement("games_100", "Game Master", "Play 100 games", "rare"),
    Achievement("wins_25", "Winner Circle", "Win 25 games", "rare"),
    Achievement("wins_100", "Champion", "Win 100 games", "epic"),
    # Casino
    Achievement("blackjack_win", "Card Counter", "Win a hand of Blackjack", "common"),
    Achievement("blackjack_21", "Perfect 21", "Hit exactly 21 in Blackjack", "rare"),
    Achievement("roulette_number", "Lucky Number", "Win a straight-up number bet in Roulette", "epic"),
    Achievement("slots_jackpot", "Jackpot!", "Hit three diamonds in Slots", "legendary"),
    # Sudoku
    Achievement("sudoku_win", "Grid Filler", "Solve a Sudoku", "common"),
    Achievement("sudoku_hard", "Puzzle Master", "Solve a hard/expert Sudoku", "epic"),
    Achievement("sudoku_clean", "Flawless Logic", "Solve a Sudoku with no strikes and no hints", "rare"),
    # Anagram
    Achievement("anagram_win", "Word Unscrambler", "Win an Anagram match", "common"),
    Achievement("anagram_ace", "Perfect Rounds", "Win every round of an Anagram match", "epic"),
    # Cryptogram
    Achievement("cryptogram_win", "Master Codebreaker", "Win a Cryptogram match", "common"),
    Achievement("cryptogram_perfect", "Clean Decode", "Crack a quote with no wrong letter guesses", "rare"),
    # Wordle / Mines / Memory / Craps
    Achievement("wordle_win", "Five-Letter Sleuth", "Solve a Wordle", "common"),
    Achievement("wordle_ace", "Three and Done", "Solve a Wordle in 3 guesses or fewer", "rare"),
    Achievement("mines_win", "Mine Sweeper", "Clear a Minesweeper board", "rare"),
    Achievement("memory_win", "Total Recall", "Find every pair in Concentration", "common"),
    Achievement("memory_sharp", "Eagle Eyes", "Win Concentration in 24 moves or fewer", "rare"),
    Achievement("craps_win", "Hot Roller", "Finish a Craps table on top", "common"),
    Achievement("craps_high_roller", "High Roller", "Finish Craps with 200+ chips", "epic"),
    # Inbox classics
    Achievement("reversi_win", "Flank Master", "Win a game of Reversi", "common"),
    Achievement("checkers_win", "Crowned", "Win a game of Checkers", "common"),
    Achievement("gomoku_win", "Five Alive", "Win a game of Gomoku", "common"),
    # Duel / trivia / ttt
    Achievement("duel_win", "Duelist", "Win a Quiz Duel", "common"),
    Achievement("duel_flawless", "Untouchable", "Win a Quiz Duel 5–0", "epic"),
    Achievement("trivia_win", "Quiz Night Champion", "Win a Trivia Royale", "common"),
    Achievement("ttt_draw", "Held the Line", "Draw against the perfect Tic-Tac-Toe house", "rare"),
    # Social deduction & co.
    Achievement("mafia_win", "Survivor", "Survive the Mafia's five nights", "rare"),
    Achievement("escape_win", "Escapist", "Escape the room", "rare"),
    Achievement("political_win", "Landslide", "Win the Political campaign", "rare"),
    Achievement("spy_win", "Double Agent", "Win a game of Spy", "common"),
    Achievement("auction_win", "Top Bidder", "Win the Auction on profit", "common"),
    Achievement("twentyq_win", "Mind Reader", "Win a game of 20 Questions", "common"),
    Achievement("bulls_win", "Code Cracker", "Crack the Bulls & Cows code", "common"),
    Achievement("numberguess_win", "Sharpshooter", "Win the Number Guess battle", "common"),
    Achievement("king_win", "Hill King", "Win King of the Hill", "common"),
)


def _achievement_map() -> dict[str, Achievement]:
    return {a.id: a for a in ACHIEVEMENTS}


# ── progression unlocks: beating challenges unlocks commands/capabilities ──
#
# Grant strings use the ``unlock:<kind>:<name>`` format:
#   unlock:command:predict      — unlocks the /predict chat command
#   unlock:capability:<cap>     — grants a policy capability string
#
# Every achievement_id here MUST exist in the ACHIEVEMENTS catalog above
# (verified by tests/test_progression_unlocks.py).

ACHIEVEMENT_UNLOCKS: dict[str, list[str]] = {
    # Casino high-rollers earn the prediction pit.
    "craps_high_roller": ["unlock:command:predict"],
    # Boss hunters earn the prediction pit too — bosses unlock commands.
    "arena_raid_win": ["unlock:command:predict"],
    # Master detectives earn deeper research tools.
    "case_expert": ["unlock:capability:research.advanced"],
    # Puzzle masters earn deeper research tools.
    "sudoku_hard": ["unlock:capability:research.advanced"],
}

#: Reverse map: unlock grant string -> achievement id that grants it.
#: Used for locked-command messages ("earn 'High Roller' to unlock /predict").
UNLOCK_SOURCE: dict[str, str] = {}
for _aid, _grants in ACHIEVEMENT_UNLOCKS.items():
    for _g in _grants:
        UNLOCK_SOURCE.setdefault(_g, _aid)
del _aid, _g, _grants


def _ensure_unlocks_table(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS progression_unlocks ("
            "player_key TEXT NOT NULL, "
            "unlock_id TEXT NOT NULL, "
            "unlocked_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, unlock_id))"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_progression_unlocks_player "
            "ON progression_unlocks(player_key)"
        )
    except Exception:  # noqa: BLE001 - defensive: no unlocks, no crash
        pass


def _persist_unlocks(db: Any, player_key: str, achievement_id: str) -> None:
    """Persist the progression grants for a newly unlocked achievement."""
    grants = ACHIEVEMENT_UNLOCKS.get(achievement_id)
    if not grants:
        return
    _ensure_unlocks_table(db)
    now = time.time()
    try:
        for grant in grants:
            db.execute(
                "INSERT OR IGNORE INTO progression_unlocks "
                "(player_key, unlock_id, unlocked_at) VALUES (?, ?, ?)",
                (player_key, grant, now),
            )
    except Exception:  # noqa: BLE001
        pass


def get_unlocks(db: Any, player_key: str) -> list[str]:
    """All progression unlock grant strings for a player. Never raises."""
    _ensure_unlocks_table(db)
    try:
        cursor = db.execute(
            "SELECT unlock_id FROM progression_unlocks WHERE player_key = ? "
            "ORDER BY unlocked_at ASC",
            (player_key,),
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception:  # noqa: BLE001 - missing table etc: no unlocks
        return []


def has_unlock(db: Any, player_key: str, unlock_id: str) -> bool:
    """True when the player holds this unlock grant. Never raises."""
    _ensure_unlocks_table(db)
    try:
        cursor = db.execute(
            "SELECT 1 FROM progression_unlocks WHERE player_key = ? "
            "AND unlock_id = ? LIMIT 1",
            (player_key, unlock_id),
        )
        return cursor.fetchone() is not None
    except Exception:  # noqa: BLE001
        return False


def progression_capabilities(db: Any, player_key: str) -> set[str]:
    """Capability strings granted by a player's progression unlocks.

    Filters the player's unlock grants to ``unlock:capability:*`` and
    returns the bare capability strings. This is what callers pass as
    the ``progression`` callable to :class:`nomorals.core.policy.Policy`.
    Never raises.
    """
    caps: set[str] = set()
    try:
        for grant in get_unlocks(db, player_key):
            if grant.startswith("unlock:capability:"):
                cap = grant.split("unlock:capability:", 1)[1].strip()
                if cap:
                    caps.add(cap)
    except Exception:  # noqa: BLE001
        pass
    return caps


# ── unlock / query ───────────────────────────────────────────────────────────

def unlock_achievement(db: Database, player_key: str, achievement_id: str) -> bool:
    """Unlock an achievement for a player. Returns True if newly unlocked."""
    catalog = _achievement_map()
    if achievement_id not in catalog:
        return False
    now = time.time()
    try:
        cursor = db.execute(
            "INSERT OR IGNORE INTO achievements (player_key, achievement_id, unlocked_at) "
            "VALUES (?, ?, ?)",
            (player_key, achievement_id, now),
        )
        # INSERT OR IGNORE inserts exactly one row on a new unlock and
        # zero when the player already has it — no timestamp guessing.
        new = cursor.rowcount > 0
    except Exception:  # noqa: BLE001
        return False
    if new:
        # achievements unlock titles — check what this one earned
        try:
            from .titles import TitleStore
            TitleStore(db).check_unlocks(player_key)
        except Exception:  # noqa: BLE001
            pass
        # achievements unlock progression grants (commands / capabilities)
        try:
            _persist_unlocks(db, player_key, achievement_id)
        except Exception:  # noqa: BLE001
            pass
    return new


def get_achievements(db: Database, player_key: str) -> list[dict[str, Any]]:
    """Return all unlocked achievements for a player, with metadata."""
    catalog = _achievement_map()
    cursor = db.execute(
        "SELECT achievement_id, unlocked_at FROM achievements "
        "WHERE player_key = ? ORDER BY unlocked_at DESC",
        (player_key,),
    )
    rows = cursor.fetchall()
    result = []
    for aid, unlocked_at in rows:
        ach = catalog.get(aid)
        if ach:
            result.append({
                "id": ach.id,
                "name": ach.name,
                "description": ach.description,
                "rarity": ach.rarity,
                "unlocked_at": unlocked_at,
            })
    return result


def get_all_achievements(db: Database, player_key: str) -> list[dict[str, Any]]:
    """Return all achievements, with unlock status."""
    unlocked = {a["id"] for a in get_achievements(db, player_key)}
    return [
        {
            "id": a.id,
            "name": a.name,
            "description": a.description,
            "rarity": a.rarity,
            "unlocked": a.id in unlocked,
        }
        for a in ACHIEVEMENTS
    ]


# ── leaderboards ─────────────────────────────────────────────────────────────

def record_score(db: Database, game_name: str, player_key: str,
                 player_name: str, score: int) -> int:
    """Record a game score. Returns the rank (1-based) in the game's leaderboard."""
    now = time.time()
    db.execute(
        "INSERT INTO leaderboards (game_name, player_key, player_name, score, played_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_name, player_key, player_name, score, now),
    )
    # Get rank
    cursor = db.execute(
        "SELECT COUNT(*) + 1 FROM leaderboards "
        "WHERE game_name = ? AND score > ?",
        (game_name, score),
    )
    row = cursor.fetchone()
    return row[0] if row else 1


def get_leaderboard(db: Database, game_name: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get top scores for a game."""
    cursor = db.execute(
        "SELECT player_name, score, played_at FROM leaderboards "
        "WHERE game_name = ? ORDER BY score DESC, played_at DESC LIMIT ?",
        (game_name, limit),
    )
    return [
        {"rank": i + 1, "player": row[0] or "Anonymous", "score": row[1], "played_at": row[2]}
        for i, row in enumerate(cursor.fetchall())
    ]


def get_player_rank(db: Database, game_name: str, player_key: str) -> int | None:
    """Get a player's best rank in a game, or None if they haven't played."""
    cursor = db.execute(
        "SELECT score FROM leaderboards "
        "WHERE game_name = ? AND player_key = ? "
        "ORDER BY score DESC LIMIT 1",
        (game_name, player_key),
    )
    row = cursor.fetchone()
    if not row:
        return None
    best_score = row[0]
    cursor = db.execute(
        "SELECT COUNT(*) + 1 FROM leaderboards "
        "WHERE game_name = ? AND score > ?",
        (game_name, best_score),
    )
    row = cursor.fetchone()
    return row[0] if row else 1


# ── game statistics ──────────────────────────────────────────────────────────

def update_game_stats(db: Database, player_key: str, game_name: str,
                      won: bool, score: int, duration: float = 0.0) -> None:
    """Update a player's stats for a game. Called on game end.

    Single atomic upsert — two games finishing at once can't lose one
    from the counts (the old SELECT-then-UPDATE could).
    """
    now = time.time()
    try:
        db.execute(
            "INSERT INTO game_stats (player_key, game_name, games_played, "
            "games_won, total_score, best_score, total_time, created_at, "
            "updated_at) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(player_key, game_name) DO UPDATE SET "
            "games_played = games_played + 1, "
            "games_won = games_won + excluded.games_won, "
            "total_score = total_score + excluded.total_score, "
            "best_score = max(best_score, excluded.best_score), "
            "total_time = total_time + excluded.total_time, "
            "updated_at = excluded.updated_at",
            (player_key, game_name, 1 if won else 0,
             score, score, duration, now, now),
        )
    except Exception:  # noqa: BLE001
        _log.debug("game_stats update failed", exc_info=True)


def get_game_stats(db: Database, player_key: str) -> list[dict[str, Any]]:
    """Return all game stats for a player."""
    cursor = db.execute(
        "SELECT game_name, games_played, games_won, total_score, best_score, "
        "total_time FROM game_stats WHERE player_key = ? ORDER BY games_played DESC",
        (player_key,),
    )
    return [
        {
            "game": row[0],
            "played": row[1],
            "won": row[2],
            "win_rate": row[2] / row[1] if row[1] else 0.0,
            "total_score": row[3],
            "best_score": row[4],
            "total_time": row[5],
        }
        for row in cursor.fetchall()
    ]


# ── live rarity: Xbox/Steam-style, computed from real unlock rates ───────────
#
# Static rarity strings lie. The live tier is the fraction of all players
# who hold the achievement: <1% legendary, <5% epic, <10% rare (the Xbox
# "Rare Achievement" cutoff with its distinct fanfare), <25% uncommon,
# else common. ``rarity_badge`` renders it for profiles and unlock
# announcements.

RARITY_ORDER = ("common", "uncommon", "rare", "epic", "legendary")

RARITY_EMOJI = {
    "common": "▫️",
    "uncommon": "🟢",
    "rare": "🔵",
    "epic": "🟣",
    "legendary": "🟠",
}


def _ensure_rarity_tables(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS achievements ("
            "player_key TEXT NOT NULL, "
            "achievement_id TEXT NOT NULL, "
            "unlocked_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, achievement_id))")
    except Exception:  # noqa: BLE001
        pass


def live_rarity(db: Any, achievement_id: str) -> tuple[str, float]:
    """(tier, unlock_fraction) across all players holding any achievement.

    ``unlock_fraction`` is the share of the player base that holds this
    achievement. Under 30 total players the sample is too small to mean
    anything — falls back to the catalog rarity.
    """
    catalog = _achievement_map()
    fallback = (catalog.get(achievement_id).rarity
                if catalog.get(achievement_id) else "common")
    _ensure_rarity_tables(db)
    try:
        rows = db.query(
            "SELECT COUNT(DISTINCT player_key) AS n FROM achievements")
        total = int((rows[0] or {}).get("n") or 0) if rows else 0
        if total < 30:
            return fallback, 0.0
        rows = db.query(
            "SELECT COUNT(DISTINCT player_key) AS n FROM achievements "
            "WHERE achievement_id = ?", (achievement_id,))
        holders = int((rows[0] or {}).get("n") or 0) if rows else 0
        frac = holders / total
    except Exception:  # noqa: BLE001
        return fallback, 0.0
    if frac < 0.01:
        tier = "legendary"
    elif frac < 0.05:
        tier = "epic"
    elif frac < 0.10:
        tier = "rare"
    elif frac < 0.25:
        tier = "uncommon"
    else:
        tier = "common"
    return tier, frac


def rarity_badge(db: Any, achievement_id: str) -> str:
    tier, frac = live_rarity(db, achievement_id)
    pct = f" · {frac:.1%} hold this" if frac else ""
    return f"{RARITY_EMOJI[tier]} {tier}{pct}"


# ── tracked achievements: multi-step progress, not just binary ──────────────
#
# "Win 25 games" as a boolean is a dead end — nobody sees how close they
# are. Tracked achievements keep a progress counter; the unlock fires
# when the counter hits the goal, and ``progress_text`` renders the bar.

#: achievement_id → (goal, unit). Only count-based catalog entries here.
TRACKED_ACHIEVEMENTS: dict[str, tuple[int, str]] = {
    "games_10": (10, "games played"),
    "games_100": (100, "games played"),
    "wins_25": (25, "wins"),
    "wins_100": (100, "wins"),
    "arena_100_wins": (100, "arena wins"),
    "arena_streak_5": (5, "arena win streak"),
    "arena_streak_10": (10, "arena win streak"),
    "arena_streak_15": (15, "arena win streak"),
    "arena_streak_20": (20, "arena win streak"),
    "arena_streak_25": (25, "arena win streak"),
    "arena_dual_10": (10, "dual-casts"),
    "hangman_5_wins": (5, "hangman wins"),
    "case_streak_3": (3, "cases in a row"),
    "case_streak_5": (5, "cases in a row"),
    "arena_underdog_5": (5, "underdog wins"),
    "rpg_level_5": (5, "RPG levels"),
}


def _ensure_progress_table(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS achievement_progress ("
            "player_key TEXT NOT NULL, "
            "achievement_id TEXT NOT NULL, "
            "progress INTEGER NOT NULL DEFAULT 0, "
            "updated_at REAL NOT NULL DEFAULT 0, "
            "PRIMARY KEY (player_key, achievement_id))")
    except Exception:  # noqa: BLE001
        pass


def track_progress(db: Any, player_key: str, achievement_id: str,
                   delta: int = 1) -> tuple[int, int, bool]:
    """Add ``delta`` progress toward a tracked achievement.

    Returns ``(progress, goal, newly_unlocked)``. Unknown ids return
    ``(0, 0, False)``. The unlock itself goes through
    :func:`unlock_achievement` so titles and grants fire normally.
    """
    goal_unit = TRACKED_ACHIEVEMENTS.get(achievement_id)
    if not goal_unit:
        return 0, 0, False
    goal, _unit = goal_unit
    _ensure_progress_table(db)
    try:
        row = db.query_one(
            "SELECT progress FROM achievement_progress "
            "WHERE player_key = ? AND achievement_id = ?",
            (player_key, achievement_id))
        progress = int((row or {}).get("progress") or 0) + max(0, delta)
        db.execute(
            "INSERT INTO achievement_progress "
            "(player_key, achievement_id, progress, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(player_key, achievement_id) DO UPDATE SET "
            "progress = excluded.progress, "
            "updated_at = excluded.updated_at",
            (player_key, achievement_id, progress, time.time()))
    except Exception:  # noqa: BLE001
        return 0, goal, False
    newly = False
    if progress >= goal:
        newly = unlock_achievement(db, player_key, achievement_id)
    return progress, goal, newly


def get_progress(db: Any, player_key: str,
                 achievement_id: str) -> tuple[int, int]:
    """(progress, goal) for a tracked achievement; (0, 0) if untracked."""
    goal_unit = TRACKED_ACHIEVEMENTS.get(achievement_id)
    if not goal_unit:
        return 0, 0
    _ensure_progress_table(db)
    try:
        row = db.query_one(
            "SELECT progress FROM achievement_progress "
            "WHERE player_key = ? AND achievement_id = ?",
            (player_key, achievement_id))
        return int((row or {}).get("progress") or 0), goal_unit[0]
    except Exception:  # noqa: BLE001
        return 0, goal_unit[0]


def progress_text(db: Any, player_key: str, achievement_id: str,
                  width: int = 10) -> str:
    """``War Machine ██████░░░░ 14/20`` — the bar players check."""
    progress, goal = get_progress(db, player_key, achievement_id)
    catalog = _achievement_map()
    name = catalog[achievement_id].name if achievement_id in catalog else achievement_id
    if not goal:
        return name
    filled = min(width, int(round(width * progress / goal)))
    bar = "█" * filled + "░" * (width - filled)
    return f"{name} {bar} {min(progress, goal)}/{goal}"


# ── repeatable milestones: re-earn the glory ────────────────────────────────
#
# Xbox players asked for it explicitly: if you earned "1000 kills", you
# should re-earn it at 2000. Repeatable milestones fire every ``every``
# completions and announce the new tier ("Champion ×3").

#: achievement_id → re-earn every N completions.
REPEATABLE_MILESTONES: dict[str, int] = {
    "wins_100": 100,
    "games_100": 100,
    "arena_100_wins": 100,
    "case_streak_5": 5,
}


def repeatable_check(db: Any, player_key: str, achievement_id: str,
                     total: int) -> int:
    """Record ``total`` completions; returns the milestone tier reached
    (0 = none). Tier k means the player just earned it for the k-th time.
    Announce ``f"{name} ×{tier}"`` when > 0."""
    every = REPEATABLE_MILESTONES.get(achievement_id)
    if not every or total < every:
        return 0
    tier = total // every
    _ensure_progress_table(db)
    try:
        row = db.query_one(
            "SELECT progress FROM achievement_progress "
            "WHERE player_key = ? AND achievement_id = ?",
            (player_key, f"{achievement_id}:rep"))
        seen = int((row or {}).get("progress") or 0)
        if tier > seen:
            db.execute(
                "INSERT INTO achievement_progress "
                "(player_key, achievement_id, progress, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(player_key, achievement_id) DO UPDATE SET "
                "progress = excluded.progress, "
                "updated_at = excluded.updated_at",
                (player_key, f"{achievement_id}:rep", tier, time.time()))
            return tier
    except Exception:  # noqa: BLE001
        pass
    return 0


# ── anniversaries: celebrate past glories ───────────────────────────────────

def achievement_anniversaries(db: Any, player_key: str,
                              window_days: float = 3.0) -> list[dict[str, Any]]:
    """Achievements unlocked ~N years ago today (± window). The "remember
    when" ping — Xbox players explicitly asked for this."""
    _ensure_rarity_tables(db)
    out: list[dict[str, Any]] = []
    try:
        rows = db.query(
            "SELECT achievement_id, unlocked_at FROM achievements "
            "WHERE player_key = ?", (player_key,)) or []
    except Exception:  # noqa: BLE001
        return out
    now = time.time()
    catalog = _achievement_map()
    for r in rows:
        try:
            unlocked = float(r.get("unlocked_at") or 0)
        except Exception:  # noqa: BLE001
            continue
        if not unlocked:
            continue
        age_days = (now - unlocked) / 86400.0
        years = round(age_days / 365.25)
        if years < 1:
            continue
        if abs(age_days - years * 365.25) <= window_days:
            ach = catalog.get(r.get("achievement_id"))
            if ach:
                out.append({"id": ach.id, "name": ach.name,
                            "years": years,
                            "rarity": ach.rarity})
    return out


# ── showcase: the player's chosen three ────────────────────────────────────
#
# Steam/Xbox let players curate what their profile shows. The showcase
# is three pinned achievements — the ones that *represent* the player,
# not just the ones the platform counted.

SHOWCASE_SIZE = 3


def _ensure_showcase_table(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS achievement_showcase ("
            "player_key TEXT NOT NULL, "
            "slot INTEGER NOT NULL, "
            "achievement_id TEXT NOT NULL, "
            "PRIMARY KEY (player_key, slot))")
    except Exception:  # noqa: BLE001
        pass


def set_showcase(db: Any, player_key: str,
                 achievement_ids: list[str]) -> tuple[bool, str]:
    """Pin up to 3 unlocked achievements to the profile showcase."""
    _ensure_showcase_table(db)
    ids = [i for i in (achievement_ids or []) if i][:SHOWCASE_SIZE]
    if not ids:
        return False, "pick 1–3 achievements to showcase."
    catalog = _achievement_map()
    unlocked = {a["id"] for a in get_achievements(db, player_key)}
    bad = [i for i in ids if i not in catalog]
    if bad:
        return False, f"unknown achievement: {', '.join(bad)}"
    locked = [i for i in ids if i not in unlocked]
    if locked:
        return False, ("you haven't unlocked "
                       f"{', '.join(catalog[i].name for i in locked)} yet.")
    try:
        db.execute("DELETE FROM achievement_showcase WHERE player_key = ?",
                   (player_key,))
        for slot, aid in enumerate(ids):
            db.execute(
                "INSERT INTO achievement_showcase "
                "(player_key, slot, achievement_id) VALUES (?, ?, ?)",
                (player_key, slot, aid))
    except Exception:  # noqa: BLE001
        return False, "couldn't save your showcase."
    names = ", ".join(catalog[i].name for i in ids)
    return True, f"📌 showcase set: {names}"


def get_showcase(db: Any, player_key: str) -> list[dict[str, Any]]:
    _ensure_showcase_table(db)
    catalog = _achievement_map()
    try:
        rows = db.query(
            "SELECT achievement_id FROM achievement_showcase "
            "WHERE player_key = ? ORDER BY slot ASC",
            (player_key,)) or []
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows:
        ach = catalog.get(r.get("achievement_id"))
        if ach:
            tier, _frac = live_rarity(db, ach.id)
            out.append({"id": ach.id, "name": ach.name,
                        "description": ach.description,
                        "rarity": tier,
                        "badge": f"{RARITY_EMOJI[tier]} {tier}"})
    return out


def render_showcase(db: Any, player_key: str, player_name: str = "") -> str:
    items = get_showcase(db, player_key)
    who = player_name or player_key
    if not items:
        return (f"📌 {who} hasn't pinned a showcase yet — "
                "/achievements showcase <id> … (up to 3).")
    lines = [f"📌 {who}'s showcase:"]
    for it in items:
        lines.append(f"  {it['badge']} **{it['name']}** — "
                     f"{it['description']}")
    return "\n".join(lines)
