"""Game recommender: "because you play X, try Y".

Suggests games from the player's actual play history (``per_game`` stats
on their profile), not from a static list. Similar games cluster by
mechanics; the recommender picks unplayed or underplayed games in the
clusters the player engages with most, with anti-repeat so it doesn't
nag about the same game twice in a row.
"""
from __future__ import annotations

import random
import time
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = ["GameRecommender", "GAME_CLUSTERS"]

#: mechanic clusters — games that scratch the same itch.
GAME_CLUSTERS: dict[str, list[str]] = {
    "word": ["wordchain", "hangman", "wordle", "anagram", "cryptogram",
             "twentyquestions"],
    "trivia": ["trivia", "duel", "quizduel", "two_truths", "wyrr"],
    "social_deduction": ["mafia", "spy", "investigation", "case"],
    "strategy": ["chess", "checkers", "gomoku", "reversi", "connect4",
                 "battleship", "tictactoe"],
    "luck": ["slots", "roulette", "craps", "blackjack", "poker"],
    "action": ["duel", "raid", "arena", "kingofhill"],
    "puzzle": ["sudoku", "2048", "minesweeper", "bullscows",
               "concentration", "digitmemory"],
    "creative": ["storychain", "rpg", "escaperoom", "world", "political"],
    "party": ["auction", "numberguess", "rps", "shop"],
}


def _clusters_for(game: str) -> list[str]:
    g = game.lower()
    return [c for c, games in GAME_CLUSTERS.items() if g in games]


class GameRecommender:
    """Suggest the next game from play history."""

    def __init__(self, db: Any = None, seed: int | None = None) -> None:
        self._kv = KVStore(db) if db is not None else None
        self.rng = random.Random(seed)

    def _last_suggested(self, player_key: str) -> str:
        if self._kv is None:
            return ""
        try:
            return str((self._kv.get(f"games.recommender.{player_key}") or {})
                       .get("last", ""))
        except Exception:  # noqa: BLE001
            return ""

    def _save_suggested(self, player_key: str, game: str) -> None:
        if self._kv is None:
            return
        try:
            self._kv.set(f"games.recommender.{player_key}",
                         {"last": game, "at": time.time()})
        except Exception:  # noqa: BLE001
            pass

    def recommend(self, player: Any, available: list[str],
                  player_key: str = "") -> dict[str, Any] | None:
        """Pick a game. Returns {"game", "reason"} or None."""
        per_game: dict[str, dict] = getattr(player, "per_game", {}) or {}
        if not per_game:
            # new player: suggest something accessible
            pick = self.rng.choice([g for g in available
                                    if g in ("trivia", "wordle", "rps",
                                             "numberguess")]
                                   or available[:1])
            return {"game": pick,
                    "reason": "a good first game — easy to pick up"}
        # rank clusters by play count
        cluster_plays: dict[str, int] = {}
        for game, stats in per_game.items():
            n = int((stats or {}).get("played", 0) or 0)
            for c in _clusters_for(game):
                cluster_plays[c] = cluster_plays.get(c, 0) + n
        if not cluster_plays:
            return None
        # favorite cluster → least-played game in it that we haven't
        # just suggested
        top = max(cluster_plays, key=cluster_plays.get)
        played = {g.lower() for g in per_game}
        last = self._last_suggested(player_key).lower()
        candidates = [g for g in GAME_CLUSTERS[top]
                      if g in available and g.lower() not in played
                      and g.lower() != last]
        if not candidates:
            # favorite cluster exhausted: try every other cluster,
            # most-played first, then completely fresh clusters
            tried = {top}
            for cluster, _ in sorted(cluster_plays.items(),
                                     key=lambda kv: -kv[1]):
                if cluster in tried:
                    continue
                tried.add(cluster)
                candidates = [g for g in GAME_CLUSTERS[cluster]
                              if g in available and g.lower() not in played
                              and g.lower() != last]
                if candidates:
                    top = cluster
                    break
            if not candidates:
                # nothing in played clusters: suggest from fresh clusters
                for cluster, games in GAME_CLUSTERS.items():
                    if cluster in tried:
                        continue
                    candidates = [g for g in games
                                  if g in available and g.lower() != last]
                    if candidates:
                        top = cluster
                        break
        if not candidates:
            return None
        pick = self.rng.choice(candidates)
        # name the game they play most for the reason line
        fav_game = max(per_game, key=lambda g: int((per_game[g] or {})
                                                  .get("played", 0) or 0))
        self._save_suggested(player_key, pick)
        return {"game": pick,
                "reason": f"because you play {fav_game} a lot — "
                          f"same {top} energy"}
