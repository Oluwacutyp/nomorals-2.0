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
from .relationships import (
    RelationshipGraph, Edge, DIMS, BRAIN_ID, OWNER_ID,
    EVENT_DELTAS,
)
from .arcs import (
    add_belief, challenge_belief, drop_belief, milestone,
    arc_summary, grow_from_interaction,
)
from .casting import cast_for, cast_preset, chemistry, ROLE_PRESETS
from .ensemble import Scene, run_scene, podcast_episode
from .processing import (
    process_session, SessionEvent, events_from_dialogue, decay_moods,
)

__all__ = [
    "Character", "CharacterStore", "default_character_dir",
    "Dialogue", "converse", "character_initiate",
    "AgentSeat", "MatchResult", "run_agent_match",
    "character_seat_decider", "register_with_engine",
    "CHAR_KEY_PREFIX", "BRAIN_KEY",
    "RelationshipGraph", "Edge", "DIMS", "BRAIN_ID", "OWNER_ID",
    "EVENT_DELTAS",
    "add_belief", "challenge_belief", "drop_belief", "milestone",
    "arc_summary", "grow_from_interaction",
    "cast_for", "cast_preset", "chemistry", "ROLE_PRESETS",
    "Scene", "run_scene", "podcast_episode",
    "process_session", "SessionEvent", "events_from_dialogue",
    "decay_moods",
]
