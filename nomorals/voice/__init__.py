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

from .accent import convert_accent, list_accents, normalize_accent
from .ambience import (
    AmbienceScene,
    apply_room,
    describe_scene,
    generate_ambience,
    mix_under,
    with_ambience,
)
from .biometrics import (
    check_liveness_response,
    decide,
    liveness_challenge,
    register_owner_voice,
    spoof_score,
    verify_voice,
)
from .catalogue import VoiceCatalogue, default_catalogue
from .design import (
    VoiceDesign,
    blend_voices,
    describe_to_params,
    design_voice,
)
from .dialogue import (
    dialogue_needs_backend_split,
    estimate_duration,
    parse_dialogue,
    render_dialogue,
)
from .director import (
    CANONICAL_BURSTS,
    CANONICAL_DELIVERY,
    CANONICAL_EMOTIONS,
    CANONICAL_FILLERS,
    ONOMATOPOEIA,
    STYLE_PRESETS,
    direct,
    render_bark,
    render_chatterbox,
    render_cosyvoice,
    render_dia,
    render_fish,
    render_for,
    render_kitten,
    render_omnivoice,
    render_orpheus,
    render_plain,
    render_spark,
    render_zonos,
    supported_backends,
)
from .efficiency import RESOURCE_BUDGETS, SegmentCache
from .fetch import (
    fetch_chatterbox_model,
    fetch_kitten_model,
    fetch_kokoro_model,
    fetch_model,
    fetch_spark_model,
    fetch_zonos_model,
)
from .longform import synthesize_long
from .mastering import master
from .money import (
    format_confirmation,
    handle_voice_money,
    parse_voice_money,
)
from .neural_emotion import render_emotional
from .nl_director import Direction, parse_direction
from .pingpong import (
    has_voice_media,
    ogg_duration,
    synthesize_voice_reply,
    to_ogg,
    transcribe_voice_media,
    voice_note_info,
)
from .rvc_bridge import (
    RVCUnavailable,
    convert as rvc_convert,
    detect_rvc,
    list_models as rvc_list_models,
    model_info as rvc_model_info,
)
from .session import (
    EnergyVAD,
    SessionReport,
    SileroVAD,
    VoiceSession,
    make_local_stt,
    make_vad,
)
from .singing import (
    Note,
    export_midi,
    humanize,
    parse_melody,
    sing,
)
from .stt import (
    StreamingSTT,
    UniversalSTT,
    available_stt_backends,
    diarize,
    format_transcript,
    make_session_stt,
    to_srt,
    to_vtt,
)
from .tts import (
    TagProcessor,
    UniversalTTS,
    VoiceLibrary,
    VoiceProfile,
    available_backends,
    mood_to_tagged_text,
    probe_reference_audio,
    voice_print,
    voice_print_distance,
)

__all__ = [
    "CANONICAL_BURSTS",
    "CANONICAL_DELIVERY",
    "CANONICAL_EMOTIONS",
    "CANONICAL_FILLERS",
    "ONOMATOPOEIA",
    "RESOURCE_BUDGETS",
    "RVCUnavailable",
    "STYLE_PRESETS",
    "AmbienceScene",
    "Direction",
    "EnergyVAD",
    "Note",
    "SegmentCache",
    "SessionReport",
    "SileroVAD",
    "StreamingSTT",
    "TagProcessor",
    "UniversalSTT",
    "UniversalTTS",
    "VoiceCatalogue",
    "VoiceDesign",
    "VoiceLibrary",
    "VoiceProfile",
    "VoiceSession",
    "apply_room",
    "available_backends",
    "available_stt_backends",
    "blend_voices",
    "check_liveness_response",
    "convert_accent",
    "decide",
    "default_catalogue",
    "describe_scene",
    "describe_to_params",
    "design_voice",
    "detect_rvc",
    "dialogue_needs_backend_split",
    "diarize",
    "direct",
    "estimate_duration",
    "export_midi",
    "fetch_chatterbox_model",
    "fetch_kitten_model",
    "fetch_kokoro_model",
    "fetch_model",
    "fetch_spark_model",
    "fetch_zonos_model",
    "format_confirmation",
    "format_transcript",
    "generate_ambience",
    "handle_voice_money",
    "has_voice_media",
    "humanize",
    "list_accents",
    "liveness_challenge",
    "make_local_stt",
    "make_session_stt",
    "make_vad",
    "master",
    "mix_under",
    "mood_to_tagged_text",
    "normalize_accent",
    "ogg_duration",
    "parse_dialogue",
    "parse_direction",
    "parse_melody",
    "parse_voice_money",
    "probe_reference_audio",
    "register_owner_voice",
    "render_bark",
    "render_chatterbox",
    "render_cosyvoice",
    "render_dia",
    "render_emotional",
    "render_fish",
    "render_for",
    "render_kitten",
    "render_omnivoice",
    "render_orpheus",
    "render_plain",
    "render_spark",
    "render_zonos",
    "rvc_convert",
    "rvc_list_models",
    "rvc_model_info",
    "sing",
    "spoof_score",
    "supported_backends",
    "synthesize_long",
    "synthesize_voice_reply",
    "to_ogg",
    "to_srt",
    "to_vtt",
    "transcribe_voice_media",
    "verify_voice",
    "voice_note_info",
    "voice_print",
    "voice_print_distance",
    "with_ambience",
]
