"""Character agents: persistent entities that converse, collaborate,
and play games with the brain, the owner, and each other."""
from .character import Character, derive_ocean, OCEAN_TRAITS
from .store import CharacterStore, default_character_dir
from .dialogue import (
    Dialogue, converse, character_initiate,
    proactive_pulse, icebreakers,
)
from .match import (
    AgentSeat, MatchResult, run_agent_match, match_intro,
    character_seat_decider, register_with_engine,
    CHAR_KEY_PREFIX, BRAIN_KEY,
)
from .relationships import (
    RelationshipGraph, Edge, DIMS, BRAIN_ID, OWNER_ID,
    EVENT_DELTAS,
)
from .arcs import (
    add_belief, challenge_belief, drop_belief, milestone,
    arc_summary, arc_story, grow_from_interaction,
)
from .casting import (
    cast_for, cast_preset, cast_against, cast_for_audience,
    chemistry, ROLE_PRESETS,
)
from .ensemble import (
    Scene, run_scene, podcast_episode,
    DramaDirector, Beat, default_beats, SCENE_FORMATS,
)
from .processing import (
    process_session, SessionEvent, events_from_dialogue, decay_moods,
)
from .memory import CharacterMemory
from .voice import (
    VoiceFingerprint, fingerprint, consistency, distinctness,
    function_word_profile, style_report,
)
from .context import CharacterContextBuilder
from .render import (
    render_card, render_cast, render_scene, render_web, render_arc,
    THEMES,
)

__all__ = [
    "Character", "derive_ocean", "OCEAN_TRAITS",
    "CharacterStore", "default_character_dir",
    "Dialogue", "converse", "character_initiate",
    "proactive_pulse", "icebreakers",
    "AgentSeat", "MatchResult", "run_agent_match", "match_intro",
    "character_seat_decider", "register_with_engine",
    "CHAR_KEY_PREFIX", "BRAIN_KEY",
    "RelationshipGraph", "Edge", "DIMS", "BRAIN_ID", "OWNER_ID",
    "EVENT_DELTAS",
    "add_belief", "challenge_belief", "drop_belief", "milestone",
    "arc_summary", "arc_story", "grow_from_interaction",
    "cast_for", "cast_preset", "cast_against", "cast_for_audience",
    "chemistry", "ROLE_PRESETS",
    "Scene", "run_scene", "podcast_episode",
    "DramaDirector", "Beat", "default_beats", "SCENE_FORMATS",
    "process_session", "SessionEvent", "events_from_dialogue",
    "decay_moods",
    "CharacterMemory",
    "VoiceFingerprint", "fingerprint", "consistency", "distinctness",
    "function_word_profile", "style_report",
    "CharacterContextBuilder",
    "render_card", "render_cast", "render_scene", "render_web",
    "render_arc", "THEMES",
]
