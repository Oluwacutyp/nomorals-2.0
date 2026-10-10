"""The social game engine — multiplayer games in every chat, on every
platform, with persistent players, a leaderboard, and an economy.

    engine = GameEngine(context, send=lambda chat, text: gateway.send(...))
    room, msgs = engine.start("telegram:123", "wordchain",
                              Player.from_sender("telegram", "456", "Ada"))
    for msg in engine.move("telegram:123", "elephant",
                           Player.from_sender("telegram", "456", "Ada")):
        send("telegram:123", msg)

See :mod:`nomorals.games.engine` for the engine,
:mod:`nomorals.games.players` for profiles/leaderboard,
:mod:`nomorals.games.economy` for the shop, and
:mod:`nomorals.games.games` for the 41 games themselves.
"""
from __future__ import annotations

from .ai import GameMind
from .economy import DEFAULT_SHOP, GameEconomy, ShopItem
from .engine import GameEngine, SendFn
from .fairness import FAIR_GAMES
from .film import (FilmBreakdown, FilmStore, FingerprintStore, GAMES,
                   analyze_vod, control_film, export_breakdown)
from .gamemaster import DM_MOODS, GameMaster, feed as dm_feed
from .matchmaking import QUEUE_GAMES, RANKED_GAMES, Matchmaker
from .players import AI_PLAYER, Leaderboard, Player, PlayerStore
from .seasons import SEASON_ROSTER, active_event

__all__ = [
    "AI_PLAYER", "DEFAULT_SHOP", "DM_MOODS", "FAIR_GAMES",
    "FilmBreakdown", "FilmStore", "FingerprintStore", "GAMES",
    "GameEconomy", "GameEngine", "GameMaster", "GameMind",
    "Leaderboard", "Matchmaker", "Player", "PlayerStore",
    "QUEUE_GAMES", "RANKED_GAMES", "SEASON_ROSTER",
    "SendFn", "ShopItem", "active_event", "analyze_vod",
    "control_film", "dm_feed", "export_breakdown",
]
