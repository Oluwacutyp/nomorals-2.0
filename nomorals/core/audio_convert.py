"""Audio format conversion for voice notes (L1 leaf).

The canonical voice-note converter lives here, at the bottom of the
layer stack, because its consumers span every layer: the L1-pinned
``integrations/voice_integration`` adapter, the L4 voice transports
(``social/chat/voice_transport``, ``voice/bridge``, ``voice/pingpong``,
``tools/audio``) and L5 callers (``agents/morning_pulse``).  It used
to live in ``tools/audio.py`` (L4), which forced the L1 adapter into
an upward import; ``tools/audio`` now re-exports it from here so the
historical import path (``nomorals.tools.audio.wav_to_voice_ogg``)
keeps working for every consumer and test double.

Pure stdlib: the conversion itself is an ``ffmpeg`` subprocess call,
and the no-ffmpeg path is a pure extension lookup.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["wav_to_voice_ogg"]


def wav_to_voice_ogg(path: str | Path) -> tuple[str, str]:
    """Convert a TTS audio file to OGG/Opus for voice-note bubbles.

    Voice notes on WhatsApp and Telegram are OGG/Opus (``voice`` kind /
    ``voice_note=True``), NOT WAV files. Sending the raw ``tts-*.wav``
    as a file shows up as a document, not a playable voice-note bubble.

    Returns ``(path, mime)``. When ffmpeg is missing or conversion
    fails, returns the ORIGINAL file + its mime — the caller must say
    so honestly (sent as a file, not a voice note), never pretend a
    file is a voice note.
    """
    import os
    import shutil
    import subprocess

    src = str(path)
    if src.lower().endswith((".ogg", ".opus")):
        return src, "audio/ogg"
    mime_by_ext = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                   ".aiff": "audio/aiff", ".m4a": "audio/mp4"}
    fallback_mime = mime_by_ext.get(os.path.splitext(src)[1].lower(),
                                    "audio/wav")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src, fallback_mime
    ogg_path = os.path.splitext(src)[0] + ".ogg"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", src, "-c:a", "libopus", "-b:a", "64k",
             ogg_path],
            check=True, capture_output=True, timeout=120)
        if os.path.isfile(ogg_path):
            return ogg_path, "audio/ogg"
    except Exception:  # noqa: BLE001 - caller falls back honestly
        pass
    return src, fallback_mime
