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

# Math Class + difficulty-reward ladder (appended; new modules only).
from .math_class import Duel, MathClass, Problem, TIERS as MATH_TIERS
from .math_commands import MathCommandHandler
from .math_economy import MATH_COINS_DISCLAIMER, MathEconomy
from .reward_ladder import REWARD_TABLE, reward_for

__all__ += [
    "Duel", "MATH_COINS_DISCLAIMER", "MATH_TIERS", "MathClass",
    "MathCommandHandler", "MathEconomy", "Problem", "REWARD_TABLE",
    "reward_for",
]

# New-player onboarding boost + early quests + fresh start (appended; new modules only).
from .early_quests import (
    EARLY_QUESTS as EARLY_QUESTS,
    claim_chest as claim_early_chest,
    early_quests as list_early_quests,
    quest_progress as early_quest_progress,
    record_progress as record_early_progress,
)
from .fresh_start import (
    SCHEMA_VERSION as GAME_SCHEMA_VERSION,
    backup_old as backup_old_game_db,
    fresh_start as fresh_start_game_db,
    verify_clean as verify_fresh_start,
)
from .onboarding import (
    BOOST_DAYS as ONBOARDING_BOOST_DAYS,
    COIN_MULTIPLIER as ONBOARDING_COIN_MULTIPLIER,
    STARTER_GEAR_PACK as STARTER_GEAR_PACK,
    TUTORIAL_STEPS as TUTORIAL_STEPS,
    XP_MULTIPLIER as ONBOARDING_XP_MULTIPLIER,
    boost_profile_line as onboarding_profile_line,
    complete_tutorial_step as complete_tutorial_step,
    get_boost as get_onboarding_boost,
    grant_starter_gear as grant_starter_gear,
    start_boost as start_onboarding_boost,
)

__all__ += [
    "EARLY_QUESTS", "GAME_SCHEMA_VERSION", "ONBOARDING_BOOST_DAYS",
    "ONBOARDING_COIN_MULTIPLIER", "ONBOARDING_XP_MULTIPLIER",
    "STARTER_GEAR_PACK", "TUTORIAL_STEPS",
    "backup_old_game_db", "onboarding_profile_line",
    "claim_early_chest", "complete_tutorial_step",
    "early_quest_progress", "fresh_start_game_db", "list_early_quests",
    "get_onboarding_boost", "grant_starter_gear",
    "record_early_progress", "start_onboarding_boost",
    "verify_fresh_start",
]
