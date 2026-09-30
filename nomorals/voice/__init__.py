"""The bot's voice: neural TTS backends behind one interface.

`nomorals/voice/tts.py` is the port of the owner's universal_tts.py:
swappable free/open-source backends (Bark, XTTS v2, Kokoro) with a tag
system for emotion / pauses / non-speech sounds, persisted voice
profiles with consent gating for cloning, and a mood bridge into the
partner's existing mood system.
"""
from __future__ import annotations

from .tts import (
    TagProcessor,
    UniversalTTS,
    VoiceLibrary,
    VoiceProfile,
    available_backends,
    mood_to_tagged_text,
)

__all__ = [
    "TagProcessor",
    "UniversalTTS",
    "VoiceLibrary",
    "VoiceProfile",
    "available_backends",
    "mood_to_tagged_text",
]
