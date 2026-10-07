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


def synthesize_voice_reply(context: Any, text: str) -> str:
    """TTS the reply text → path to an .ogg voice note.  Raises on failure."""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to synthesize")
    # keep voice notes short — cap at ~30s of speech
    text = text[:600]
    out = context.tools.call("speak", text=text)
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
