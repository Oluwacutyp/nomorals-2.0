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

__all__ = ["Achievement", "ACHIEVEMENTS", "unlock_achievement", "get_achievements"]


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
        return cursor.rowcount > 0
    except Exception:  # noqa: BLE001
        return False


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
    """Update a player's stats for a game. Called on game end."""
    now = time.time()
    # Insert or update the stat row
    cursor = db.execute(
        "SELECT games_played, games_won, total_score, best_score, total_time "
        "FROM game_stats WHERE player_key = ? AND game_name = ?",
        (player_key, game_name),
    )
    row = cursor.fetchone()
    if row:
        played, wins, total_score, best_score, total_time = row
        db.execute(
            "UPDATE game_stats SET games_played = ?, games_won = ?, "
            "total_score = ?, best_score = ?, total_time = ?, updated_at = ? "
            "WHERE player_key = ? AND game_name = ?",
            (played + 1, wins + (1 if won else 0),
             total_score + score, max(best_score, score),
             total_time + duration, now, player_key, game_name),
        )
    else:
        db.execute(
            "INSERT INTO game_stats (player_key, game_name, games_played, "
            "games_won, total_score, best_score, total_time, created_at, updated_at) "
            "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (player_key, game_name, 1 if won else 0,
             score, score, duration, now, now),
        )


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
