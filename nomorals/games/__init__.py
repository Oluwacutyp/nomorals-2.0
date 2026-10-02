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
:mod:`nomorals.games.games` for the 39 games themselves.
"""
from __future__ import annotations

from .ai import GameMind
from .economy import DEFAULT_SHOP, GameEconomy, ShopItem
from .engine import GameEngine, SendFn
from .players import AI_PLAYER, Leaderboard, Player, PlayerStore

__all__ = [
    "AI_PLAYER", "DEFAULT_SHOP", "GameEconomy", "GameEngine", "GameMind",
    "Leaderboard", "Player", "PlayerStore", "SendFn", "ShopItem",
]
