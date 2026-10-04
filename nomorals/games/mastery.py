"""Per-game mastery tiers: persistent mastery ranks for the non-arena games.

Arena has levels, gear, skills, ranks, achievements, titles — the rest of
the suite had just win/lose and flat rewards.  This module closes the
gap without copying arena's combat system:

* every game gets a 7-tier mastery ladder (custom names per game, e.g.
  sudoku: Novice → Solver → Tactician → Expert → Master → Grandmaster →
  Legend);
* tiers unlock from cumulative performance already tracked by
  :func:`achievements.update_game_stats` — wins, games played, and best
  score for *that specific game*;
* tiers gate game-specific unlockables (harder sudoku boards, bigger
  gomoku boards, …) — see :data:`UNLOCKS`;
* the engine announces the player's tier at game start, fans a fanfare
  on tier-up at game finish, and ``/mastery`` shows the whole board.

Mastery points: ``wins × 100 + played × 10 + best-score share``.  The
best-score share is normalized per game (a 20k 2048 run and a 5-move
tic-tac-toe win both count, on their own scale) so no single game's
score inflation dominates.  Thresholds are uniform: 0 / 200 / 500 /
1000 / 2000 / 3500 / 5500.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "THRESHOLDS", "TIER_LADDERS", "UNLOCKS",
    "mastery_points", "mastery_tier", "mastery_progress",
    "get_one_game_stats", "mastery_line", "unlocks_for", "new_unlocks",
    "describe_mastery", "describe_game_mastery",
]

#: points needed for tier index 0..6.  Index 0 (Novice) is free.
THRESHOLDS: tuple[int, ...] = (0, 200, 500, 1000, 2000, 3500, 5500)

_DEFAULT_LADDER: tuple[str, ...] = (
    "Novice", "Apprentice", "Adept", "Expert", "Master", "Grandmaster",
    "Legend",
)

#: game → its 7 tier names.  Games not listed use the default ladder.
TIER_LADDERS: dict[str, tuple[str, ...]] = {
    "sudoku": ("Novice", "Solver", "Tactician", "Expert", "Master",
               "Grandmaster", "Legend"),
    "anagram": ("Novice", "Unscrambler", "Wordsmith", "Lexicist",
                "Master", "Grandmaster", "Legend"),
    "cryptogram": ("Novice", "Decoder", "Codebreaker", "Cryptanalyst",
                   "Master", "Grandmaster", "Legend"),
    "ttt": ("Novice", "Player", "Tactician", "Strategist", "Master",
            "Grandmaster", "Legend"),
    "gomoku": ("Novice", "Placer", "Fighter", "Strategist", "Master",
               "Grandmaster", "Legend"),
    "reversi": ("Novice", "Flanker", "Outflanker", "Strategist",
                "Master", "Grandmaster", "Legend"),
    "checkers": ("Novice", "Jumper", "Crowner", "Strategist", "Master",
                 "Grandmaster", "Legend"),
    "connect4": ("Novice", "Dropper", "Connector", "Strategist",
                 "Master", "Grandmaster", "Legend"),
    "battleship": ("Novice", "Gunner", "Spotter", "Captain", "Master",
                   "Grandmaster", "Legend"),
    "2048": ("Novice", "Slider", "Merger", "Tilelord", "Master",
             "Grandmaster", "Legend"),
    "snake": ("Novice", "Wiggler", "Hunter", "Survivor", "Master",
              "Grandmaster", "Legend"),
    "hangman": ("Novice", "Guesser", "Speller", "Wordhunter", "Master",
                "Grandmaster", "Legend"),
    "trivia": ("Novice", "Contestant", "Quizzer", "Savant", "Master",
               "Grandmaster", "Legend"),
    "wordle": ("Novice", "Guesser", "Sleuth", "Deductionist", "Master",
               "Grandmaster", "Legend"),
    "poker": ("Novice", "Caller", "Bluffer", "Shark", "Master",
              "Grandmaster", "Legend"),
    "blackjack": ("Novice", "Hitter", "Counter", "Shark", "Master",
                  "Grandmaster", "Legend"),
    "duel": ("Novice", "Challenger", "Duelist", "Champion", "Master",
             "Grandmaster", "Legend"),
}

#: game → score that counts as a "perfect" best for normalization.
#: A best_score at/above the cap contributes the full 100 score points.
_GAME_SCORE_CAPS: dict[str, int] = {
    "sudoku": 1400, "2048": 20000, "snake": 300, "trivia": 220,
    "hangman": 220, "ttt": 120, "gomoku": 400, "reversi": 64,
    "checkers": 40, "connect4": 60, "battleship": 120,
    "anagram": 300, "cryptogram": 200, "wordle": 60, "poker": 1000,
    "blackjack": 500, "craps": 400, "slots": 1000, "roulette": 1000,
    "mines": 200, "memory": 100, "bulls": 120, "digits": 200,
    "20q": 40, "rps": 30, "numberguess": 100, "duel": 200,
    "king": 200, "mafia": 100, "case": 500, "wordchain": 100,
    "spy": 50, "auction": 500, "story": 100, "escape": 300,
}
_DEFAULT_SCORE_CAP = 300

#: game → [(tier_index, unlock description)].  Tier 0 entries are
#: informational (available to everyone); higher tiers are genuinely
#: gated by the games' ``new_state``.
UNLOCKS: dict[str, list[tuple[int, str]]] = {
    "sudoku": [
        (0, "📅 daily puzzle — /game sudoku daily (same board for everyone)"),
        (1, "⏱️ timed mode — /game sudoku timed (speed bonus on the score)"),
        (2, "🔥 hard boards — /game sudoku hard"),
        (3, "⚡ expert boards — /game sudoku expert"),
    ],
    "gomoku": [
        (2, "⬛ big board 13×13 — /game gomoku big"),
        (4, "⬛ huge board 19×19 — /game gomoku huge"),
    ],
    "2048": [
        (3, "⬛ marathon board 5×5 — /game 2048 big"),
    ],
    "hangman": [
        (2, "📖 long words — /game hangman long (9+ letters)"),
    ],
    "trivia": [
        (2, "💀 sudden death — /game trivia sudden (1 life, double points)"),
    ],
}

#: emoji per tier index, for the /mastery board.
_TIER_EMOJI = ("🌱", "🌿", "🛡️", "⚔️", "👑", "💎", "🌟")


def _ladder(game: str) -> tuple[str, ...]:
    return TIER_LADDERS.get(game, _DEFAULT_LADDER)


def mastery_points(game: str, stats: dict[str, Any] | None) -> int:
    """Mastery points from one game's cumulative stats dict.

    ``stats`` is shaped like :func:`achievements.get_game_stats` rows
    (``played``/``won``/``best_score``).  Missing/empty stats → 0.
    """
    if not stats:
        return 0
    played = int(stats.get("played", 0) or 0)
    won = int(stats.get("won", 0) or 0)
    best = int(stats.get("best_score", 0) or 0)
    cap = _GAME_SCORE_CAPS.get(game, _DEFAULT_SCORE_CAP)
    score_pts = int(100 * min(max(best, 0), cap) / cap)
    return won * 100 + played * 10 + score_pts


def mastery_tier(game: str,
                 stats: dict[str, Any] | None) -> tuple[str, int]:
    """(tier_name, tier_index) for a game's stats.  Index 0..6."""
    points = mastery_points(game, stats)
    index = 0
    for i, need in enumerate(THRESHOLDS):
        if points >= need:
            index = i
    return _ladder(game)[index], index


def mastery_progress(game: str,
                      stats: dict[str, Any] | None) -> dict[str, Any]:
    """Points, current tier, and distance to the next tier."""
    points = mastery_points(game, stats)
    _, index = mastery_tier(game, stats)
    if index + 1 < len(THRESHOLDS):
        need = THRESHOLDS[index + 1]
        to_go = need - points
        pct = min(99, int(100 * (points - THRESHOLDS[index])
                          / max(1, need - THRESHOLDS[index])))
        next_name = _ladder(game)[index + 1]
    else:
        need, to_go, pct, next_name = points, 0, 100, ""
    return {
        "points": points,
        "tier_index": index,
        "tier_name": _ladder(game)[index],
        "next_threshold": need,
        "to_next": to_go,
        "pct_to_next": pct,
        "next_name": next_name,
    }


def get_one_game_stats(db: Any, player_key: str,
                       game: str) -> dict[str, Any]:
    """The :func:`achievements.get_game_stats` row for one game, or {}."""
    try:
        cursor = db.execute(
            "SELECT games_played, games_won, total_score, best_score, "
            "total_time FROM game_stats "
            "WHERE player_key = ? AND game_name = ?",
            (player_key, game),
        )
        row = cursor.fetchone()
    except Exception:  # noqa: BLE001
        return {}
    if not row:
        return {}
    played, won, total_score, best_score, total_time = row
    return {
        "game": game,
        "played": played,
        "won": won,
        "win_rate": won / played if played else 0.0,
        "total_score": total_score,
        "best_score": best_score,
        "total_time": total_time,
    }


def mastery_line(game: str, stats: dict[str, Any] | None) -> str:
    """Compact tier line for game-start banners."""
    prog = mastery_progress(game, stats)
    emoji = _TIER_EMOJI[prog["tier_index"]]
    return (f"{emoji} mastery: {prog['tier_name']} · "
            f"{prog['points']:,} pts")


def unlocks_for(game: str, tier_index: int) -> list[str]:
    """Unlock descriptions available at (or below) ``tier_index``."""
    return [desc for need, desc in UNLOCKS.get(game, [])
            if need <= tier_index]


def new_unlocks(game: str, old_index: int,
                new_index: int) -> list[str]:
    """Unlocks freshly earned moving from ``old_index`` to ``new_index``."""
    return [desc for need, desc in UNLOCKS.get(game, [])
            if old_index < need <= new_index]


def _fmt_num(n: int) -> str:
    return f"{n:,}"


def describe_game_mastery(db: Any, player_key: str, game: str) -> str:
    """Detail view for one game: tier, stats, progress, unlocks."""
    stats = get_one_game_stats(db, player_key, game)
    prog = mastery_progress(game, stats)
    emoji = _TIER_EMOJI[prog["tier_index"]]
    lines = [f"{emoji} {game} — {prog['tier_name']}"]
    if stats:
        wr = stats["win_rate"]
        lines.append(
            f"{_fmt_num(prog['points'])} pts · {stats['played']} played · "
            f"{stats['won']} won ({wr:.0%}) · best {stats['best_score']:,}")
    else:
        lines.append("no games yet — this is where the legend starts.")
    if prog["next_name"]:
        bar = "▓" * (prog["pct_to_next"] // 10) + \
              "░" * (10 - prog["pct_to_next"] // 10)
        lines.append(
            f"next: {prog['next_name']} at "
            f"{_fmt_num(prog['next_threshold'])} "
            f"({_fmt_num(prog['to_next'])} to go) {bar}")
    else:
        lines.append("max tier reached — untouchable. 🌟")
    have = unlocks_for(game, prog["tier_index"])
    locked = [desc for need, desc in UNLOCKS.get(game, [])
              if need > prog["tier_index"]]
    if have or locked:
        lines.append("unlocks:")
        lines.extend(f"  ✅ {d}" for d in have)
        lines.extend(f"  🔒 {d}" for d in locked)
    return "\n".join(lines)


def describe_mastery(db: Any, player_key: str,
                     player_name: str = "you") -> str:
    """The ``/mastery`` board: every played game's tier + next unlocks."""
    try:
        from .achievements import get_game_stats
        rows = get_game_stats(db, player_key)
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        return ("🏅 no mastery yet — play any game and your tiers start "
                "building. /game list to pick your poison.")
    scored = []
    for row in rows:
        game = row["game"]
        prog = mastery_progress(game, row)
        scored.append((prog["points"], game, prog, row))
    scored.sort(key=lambda t: -t[0])
    lines = [f"🏅 {player_name}'s mastery"]
    shown = 0
    for _pts, game, prog, row in scored:
        if shown >= 12:
            break
        emoji = _TIER_EMOJI[prog["tier_index"]]
        nxt = (f" → {prog['next_name']} "
               f"({_fmt_num(prog['to_next'])} to go)"
               if prog["next_name"] else " · MAX")
        lines.append(
            f"  {emoji} {game} — {prog['tier_name']} "
            f"({_fmt_num(prog['points'])} pts, {row['won']}/{row['played']} won)"
            f"{nxt}")
        shown += 1
    rest = len(scored) - shown
    if rest > 0:
        lines.append(f"  …and {rest} more game(s).")
    # next unlocks across the top games, so the board always points forward
    upcoming: list[str] = []
    for _pts, game, prog, _row in scored[:6]:
        for need, desc in UNLOCKS.get(game, []):
            if need == prog["tier_index"] + 1:
                upcoming.append(f"{game}: {desc}")
                break
    if upcoming:
        lines.append("next unlocks:")
        lines.extend(f"  🔜 {u}" for u in upcoming[:4])
    lines.append("detail: /mastery <game>")
    return "\n".join(lines)
