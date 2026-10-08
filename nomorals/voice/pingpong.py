"""Voice-note ping-pong: talk to Devon with voice notes, get voice back.

The async voice loop — no real-time audio stack needed, works on every
profile including Termux:

    inbound voice note → STT → brain → TTS → .ogg → sendVoice

Inbound transcription reuses the brain's existing path
(``_transcribe_media_note``); this module owns the *reply* side and the
feature wiring.  Nothing here auto-enables: the ``voice`` feature flag
gates it, and a voice reply only goes out when the incoming message
actually was a voice note.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "has_voice_media",
    "transcribe_voice_media",
    "synthesize_voice_reply",
    "to_ogg",
]

_VOICE_KINDS = ("audio", "voice")


def has_voice_media(message: Any) -> bool:
    """True when the inbound message carries a voice note / audio."""
    for media in getattr(message, "media", None) or []:
        kind = str(getattr(media, "kind", "") or "").lower()
        mime = str(getattr(media, "mime", "") or "").lower()
        if kind in _VOICE_KINDS or mime.startswith("audio"):
            return True
    return False


def transcribe_voice_media(context: Any, message: Any) -> str:
    """Transcribe the first voice attachment.  Returns \"\" on any failure."""
    for media in getattr(message, "media", None) or []:
        kind = str(getattr(media, "kind", "") or "").lower()
        mime = str(getattr(media, "mime", "") or "").lower()
        if kind not in _VOICE_KINDS and not mime.startswith("audio"):
            continue
        path = getattr(media, "path", None)
        if not path:
            continue
        try:
            outcome = context.tools.call("transcribe", path=str(path),
                                         provider="auto")
            if outcome.ok:
                text = ((outcome.value or {}).get("text", "") or "").strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001 - hearing is a bonus, never fatal
            _log.debug("ping-pong transcription failed", exc_info=True)
    return ""


def _private_voice_reply(context: Any, text: str) -> str:
    """Devon's own cloned voice (XTTS v2) on capable profiles.

    Returns the wav path, or "" when unavailable (wrong profile, no
    voice profile registered, backend missing) — the caller falls back
    to the cloud voice.  Never raises.
    """
    try:
        settings = getattr(context, "settings", None)
        profile = (getattr(settings, "profile", "") or "").lower()
        if profile in ("termux", "phone", "mobile"):
            return ""
        from .tts import UniversalTTS
        tts = UniversalTTS(backend="xtts")
        # "devon" is the registered private voice profile; skip silently
        # when the owner hasn't cloned one yet.
        names = {p.get("name") for p in tts.voices.list()}
        if "devon" not in names:
            return ""
        result = tts.speak(text, voice_name="devon")
        path = result.get("path", "")
        return path if path and Path(path).is_file() else ""
    except Exception:  # noqa: BLE001 - private voice is a bonus, never a failure
        return ""


def synthesize_voice_reply(context: Any, text: str) -> str:
    """TTS the reply text → path to an .ogg voice note.  Raises on failure."""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to synthesize")
    # keep voice notes short — cap at ~30s of speech
    text = text[:600]
    # Private voice first: on workstation-class profiles, use Devon's own
    # cloned voice (XTTS v2) when a voice profile is registered.  Phone /
    # termux falls back to the cloud voice below — profile-gated, never
    # designed down.
    private_path = _private_voice_reply(context, text)
    if private_path:
        return to_ogg(private_path)
    # voice from settings (NM_AUDIO_TTS_VOICE); Nigerian English default
    # suits Devon better than edge-tts's flat default.
    voice = ""
    try:
        settings = getattr(context, "settings", None)
        voice = (getattr(getattr(settings, "audio", None), "tts_voice", "")
                 or "en-NG-EzinneNeural")
    except Exception:  # noqa: BLE001
        voice = "en-NG-EzinneNeural"
    out = context.tools.call("speak", text=text, voice=voice)
    if not out.ok:
        raise RuntimeError(f"TTS failed: {getattr(out, 'error', out)}")
    value = out.value or {}
    path = value.get("path") if isinstance(value, dict) else str(value)
    if not path or not Path(str(path)).is_file():
        raise RuntimeError("TTS returned no audio file")
    return to_ogg(str(path))


def to_ogg(src_path: str) -> str:
    """Convert any audio file to opus .ogg (Telegram sendVoice format)."""
    src = Path(src_path)
    if src.suffix.lower() == ".ogg":
        return str(src)
    dest = Path(tempfile.gettempdir()) / f"devon_reply_{os.getpid()}.ogg"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:a", "libopus", "-b:a", "48k", str(dest)],
            timeout=60, capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not installed — cannot make voice notes") from exc
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            f"ffmpeg conversion failed: {proc.stderr.decode()[:200]}")
    return str(dest)
