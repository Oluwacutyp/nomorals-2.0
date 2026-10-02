"""L3 — partner: the romantic-companion cognition core.

The persona is data, the mood is a persistent state machine, the relationship
is a long-arc record, and the background knowledge is gated context — nothing
in this package *talks*. Talking happens in L5 (``agents/partner_runtime.py``)
through the LLM router; this package decides *who she is and how she feels*.

    from nomorals.partner import (
        Persona, MoodEngine, Relationship, BackgroundSelector,
        PartnerResponder, PartnerContextBuilder, detect_signals,
    )
"""

from __future__ import annotations

from .background import BackgroundFact, BackgroundPack, BackgroundSelector, GATE_MODES
from .context import PLATFORM_NOTES, PartnerContextBuilder
from .lexicon_feed import CATEGORIES as LEXICON_CATEGORIES
from .lexicon_feed import LEXICON_MODULE, LexiconFeed, seed_partner_lexicon
from .mood import DIMENSIONS, EVENT_TABLE, MOOD_LABELS, MoodEngine, MoodEvent, MoodState
from .persona import DEFAULT_BASLINES, Persona, SpeechProfile, default_persona, persona_from_dict
from .relationship import STAGES, Relationship
from .responder import FALLBACK_LINES, PartnerResponder, ReplyBundle, Signal, detect_signals
from .style import (
    GuardVerdict,
    clamp_to_budget,
    length_budget,
    parrot_check,
    should_answer_short,
    split_messages,
    strip_robotic,
)

__all__ = [
    "BackgroundFact",
    "BackgroundPack",
    "BackgroundSelector",
    "DEFAULT_BASLINES",
    "DIMENSIONS",
    "EVENT_TABLE",
    "FALLBACK_LINES",
    "GATE_MODES",
    "GuardVerdict",
    "LEXICON_CATEGORIES",
    "LEXICON_MODULE",
    "LexiconFeed",
    "MOOD_LABELS",
    "PLATFORM_NOTES",
    "PartnerContextBuilder",
    "PartnerResponder",
    "Persona",
    "ReplyBundle",
    "Relationship",
    "Signal",
    "STAGES",
    "SpeechProfile",
    "clamp_to_budget",
    "default_persona",
    "detect_signals",
    "length_budget",
    "parrot_check",
    "persona_from_dict",
    "seed_partner_lexicon",
    "should_answer_short",
    "split_messages",
    "strip_robotic",
]
