"""The bot's voice: neural TTS backends behind one interface.

`nomorals/voice/tts.py` is the port of the owner's universal_tts.py:
swappable free/open-source backends (Chatterbox, F5-TTS, OmniVoice,
Qwen3-TTS, Bark, XTTS v2, Kokoro, Piper, CosyVoice, Orpheus, Dia,
hf-endpoint, and the stdlib-only `system` fallback) with
a tag system for emotion / pauses / non-speech sounds, persisted voice
profiles with consent gating for cloning, and a mood bridge into the
partner's existing mood system.

`nomorals/voice/director.py` is the performance director: 30 vocal
bursts, 53 emotion tags, delivery styles, fillers, stutters, pacing —
rendered per backend in its native vocabulary (Fish free-form tags,
Dia parens, Orpheus angle tags, Chatterbox bracket tags, OmniVoice
non-verbals, CosyVoice instruct tokens, Bark tags, or speakable
onomatopoeia for plain backends).

`nomorals/voice/stt.py` is the transcription mirror: faster-whisper
(primary), NVIDIA Parakeet (fastest free English dictation),
whisper.cpp, classic whisper, and an HF serverless fallback behind
``UniversalSTT``. :func:`nomorals.voice.session.make_local_stt`
adapts it straight into the live ``VoiceSession`` loop — no cloud, no
bridge.
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
    render_chatterbox,
    render_cosyvoice,
    render_dia,
    render_fish,
    render_for,
    render_omnivoice,
    render_orpheus,
    render_plain,
)
from .session import make_local_stt
from .stt import UniversalSTT, available_stt_backends, make_session_stt
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
    "UniversalSTT",
    "UniversalTTS",
    "VoiceLibrary",
    "VoiceProfile",
    "available_backends",
    "available_stt_backends",
    "direct",
    "make_local_stt",
    "make_session_stt",
    "mood_to_tagged_text",
    "probe_reference_audio",
    "render_bark",
    "render_chatterbox",
    "render_cosyvoice",
    "render_dia",
    "render_fish",
    "render_for",
    "render_omnivoice",
    "render_orpheus",
    "render_plain",
]
