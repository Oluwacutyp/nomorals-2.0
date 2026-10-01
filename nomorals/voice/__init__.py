"""The bot's voice: neural TTS backends behind one interface.

`nomorals/voice/tts.py` is the port of the owner's universal_tts.py:
swappable free/open-source backends (Bark, XTTS v2, Kokoro, CosyVoice,
Orpheus, Dia) with a tag system for emotion / pauses / non-speech
sounds, persisted voice profiles with consent gating for cloning, and
a mood bridge into the partner's existing mood system.

`nomorals/voice/director.py` is the performance director: 30 vocal
bursts, 53 emotion tags, delivery styles, fillers, stutters, pacing —
rendered per backend in its native vocabulary (Fish free-form tags,
Dia parens, Orpheus angle tags, CosyVoice instruct tokens, Bark tags,
or speakable onomatopoeia for plain backends).
"""
from __future__ import annotations

from .director import (
    CANONICAL_BURSTS,
    CANONICAL_DELIVERY,
    CANONICAL_EMOTIONS,
    CANONICAL_FILLERS,
    ONOMATOPOEIA,
    direct,
    render_bark,
    render_cosyvoice,
    render_dia,
    render_fish,
    render_for,
    render_orpheus,
    render_plain,
)
from .tts import (
    TagProcessor,
    UniversalTTS,
    VoiceLibrary,
    VoiceProfile,
    available_backends,
    mood_to_tagged_text,
    probe_reference_audio,
)

__all__ = [
    "CANONICAL_BURSTS",
    "CANONICAL_DELIVERY",
    "CANONICAL_EMOTIONS",
    "CANONICAL_FILLERS",
    "ONOMATOPOEIA",
    "TagProcessor",
    "UniversalTTS",
    "VoiceLibrary",
    "VoiceProfile",
    "available_backends",
    "direct",
    "mood_to_tagged_text",
    "probe_reference_audio",
    "render_bark",
    "render_cosyvoice",
    "render_dia",
    "render_fish",
    "render_for",
    "render_orpheus",
    "render_plain",
]
