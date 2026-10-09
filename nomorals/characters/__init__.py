"""Character agents: persistent entities that converse, collaborate,
and play games with the brain, the owner, and each other."""
from .character import Character
from .store import CharacterStore, default_character_dir
from .dialogue import Dialogue, converse, character_initiate
from .match import (
    AgentSeat, MatchResult, run_agent_match,
    character_seat_decider, register_with_engine,
    CHAR_KEY_PREFIX, BRAIN_KEY,
)

__all__ = [
    "Character", "CharacterStore", "default_character_dir",
    "Dialogue", "converse", "character_initiate",
    "AgentSeat", "MatchResult", "run_agent_match",
    "character_seat_decider", "register_with_engine",
    "CHAR_KEY_PREFIX", "BRAIN_KEY",
]
