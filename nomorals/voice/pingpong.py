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

import subprocess
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "has_voice_media",
    "transcribe_voice_media",
    "transcribe_voice_media_with_reason",
    "synthesize_voice_reply",
    "to_ogg",
    "voice_note_info",
    "get_cached_transcript",
    "cache_transcript",
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


# One STT-failure formatter (infra spec §4.2): canonical implementation in
# nomorals.core.errors.short_stt_reason; alias so the copies never drift.
from ..core.errors import short_stt_reason as _short_stt_reason


def get_cached_transcript(meta: Any, path: str) -> str | None:
    """Transcript cached for this exact media path, or None.

    The transcript cache is keyed per media path: two voice notes in
    one message must never share a transcript (the old single
    ``meta["voice_transcript"]`` string let the last note's words
    overwrite the first's). The legacy single key is kept as a
    fallback for single-media messages.
    """
    if not isinstance(meta, dict):
        return None
    by_path = meta.get("voice_transcripts")
    if isinstance(by_path, dict):
        text = by_path.get(str(path or ""))
        if isinstance(text, str) and text.strip():
            return text
    legacy = meta.get("voice_transcript")
    if isinstance(legacy, str) and legacy.strip():
        return legacy
    return None


def cache_transcript(meta: Any, path: str, text: str) -> None:
    """Cache a transcript under its media path (+ legacy single key).

    ``meta["voice_transcripts"]`` is ``{path: text}``; the legacy
    ``meta["voice_transcript"]`` keeps the FIRST transcript (the
    single-media fast path and older readers). Never raises.
    """
    try:
        if not isinstance(meta, dict):
            return
        clean = (text or "").strip()
        if not clean:
            return
        by_path = meta.get("voice_transcripts")
        if not isinstance(by_path, dict):
            by_path = {}
            meta["voice_transcripts"] = by_path
        by_path[str(path or "")] = clean
        if not meta.get("voice_transcript"):
            meta["voice_transcript"] = clean
    except Exception as e: # meta bookkeeping never breaks chat
        record_failure('pingpong.cache_transcript:96', e)


def _record_stt_reason(message: Any, reason: str) -> None:
    """Record/clear the STT failure reason on ``message.meta``.

    The runtime reads this to answer an untranscribable voice note
    honestly instead of dropping it silently. Never raises.
    """
    try:
        meta = getattr(message, "meta", None)
        if isinstance(meta, dict):
            if reason:
                meta["stt_failure_reason"] = reason
            else:
                meta.pop("stt_failure_reason", None)
    except Exception as e: # meta bookkeeping never breaks chat
        record_failure('pingpong._record_stt_reason:113', e)


def transcribe_voice_media_with_reason(
        context: Any, message: Any) -> tuple[str, str]:
    """Transcribe the first voice attachment.

    Returns ``(text, reason)`` — ``reason`` is "" on success (or when
    there was no voice media to attempt), otherwise a short reason:
    "no STT backend" / "STT API error" / "empty transcript" /
    "no audio file". On failure the reason is also recorded on
    ``message.meta["stt_failure_reason"]``; on success any stale
    reason is cleared.
    """
    reason = ""
    saw_voice = False
    for media in getattr(message, "media", None) or []:
        kind = str(getattr(media, "kind", "") or "").lower()
        mime = str(getattr(media, "mime", "") or "").lower()
        if kind not in _VOICE_KINDS and not mime.startswith("audio"):
            continue
        saw_voice = True
        # A transcript an earlier stage already produced (_spoken_command
        # runs before pingpong) is reused, never re-transcribed: a later
        # attempt can transiently fail and lose a transcript that had
        # already succeeded. Keyed per media path — two voice notes in
        # one message keep their own transcripts.
        path = getattr(media, "path", None)
        _meta = getattr(message, "meta", None)
        if isinstance(_meta, dict):
            _cached = get_cached_transcript(_meta, path)
            if _cached:
                _record_stt_reason(message, "")
                _meta.pop("voice_untranscribed_reason", None)
                return _cached, ""
        if not path:
            reason = "no audio file"
            continue
        try:
            outcome = context.tools.call("transcribe", path=str(path),
                                         provider="auto")
            if outcome.ok:
                text = ((outcome.value or {}).get("text", "") or "").strip()
                if text:
                    _record_stt_reason(message, "")
                    # Cache for _media_notes, which runs after this —
                    # keyed by path so a second note keeps its own words.
                    _meta2 = getattr(message, "meta", None)
                    cache_transcript(_meta2, path, text)
                    try:
                        if isinstance(_meta2, dict):
                            _meta2.pop("voice_untranscribed_reason", None)
                    except Exception as e: # meta bookkeeping never breaks chat
                        record_failure('pingpong.transcribe_voice_media_with_reason:166', e)
                    return text, ""
                reason = "empty transcript"
            else:
                reason = _short_stt_reason(
                    getattr(outcome, "error", "") or "STT error")
        except Exception as exc:  # noqa: BLE001 - hearing is a bonus, never fatal
            _log.debug("ping-pong transcription failed", exc_info=True)
            reason = _short_stt_reason(exc)
    if saw_voice and reason:
        _record_stt_reason(message, reason)
    return "", reason


def transcribe_voice_media(context: Any, message: Any) -> str:
    """Transcribe the first voice attachment.  Returns \"\" on any failure."""
    text, _reason = transcribe_voice_media_with_reason(context, message)
    return text


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
    except Exception as e: # private voice is a bonus, never a failure
        record_failure('pingpong._private_voice_reply:209', e)
        return ""


def synthesize_voice_reply(context: Any, text: str,
                           progress_cb: Any = None) -> str:
    """TTS the reply text → path to an .ogg voice note.  Raises on failure.

    ``progress_cb(stage)`` fires with "synthesizing" / "converting"
    so chats can show a typing/voice indicator during the (slow) TTS.
    """
    def _progress(stage: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage)
            except Exception as e: # progress never breaks
                record_failure('pingpong.synthesize_voice_reply._progress:224', e)

    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to synthesize")
    # keep voice notes short — cap at ~30s of speech
    text = text[:600]
    _progress("synthesizing")
    # Private voice first: on workstation-class profiles, use Devon's own
    # cloned voice (XTTS v2) when a voice profile is registered.  Phone /
    # termux falls back to the cloud voice below — profile-gated, never
    # designed down.
    private_path = _private_voice_reply(context, text)
    if private_path:
        _progress("converting")
        return to_ogg(private_path)
    # voice from settings (NM_AUDIO_TTS_VOICE); Nigerian English default
    # suits Devon better than edge-tts's flat default.
    voice = ""
    try:
        settings = getattr(context, "settings", None)
        voice = (getattr(getattr(settings, "audio", None), "tts_voice", "")
                 or "en-NG-EzinneNeural")
    except Exception as e:
        record_failure('pingpong.synthesize_voice_reply:248', e)
        voice = "en-NG-EzinneNeural"
    out = context.tools.call("speak", text=text, voice=voice)
    if not out.ok:
        raise RuntimeError(f"TTS failed: {getattr(out, 'error', out)}")
    value = out.value or {}
    path = value.get("path") if isinstance(value, dict) else str(value)
    if not path or not Path(str(path)).is_file():
        raise RuntimeError("TTS returned no audio file")
    _progress("converting")
    return to_ogg(str(path))


def voice_note_info(ogg_path: str) -> dict[str, Any]:
    """Describe a voice note: duration, size, format. Never raises."""
    p = Path(ogg_path)
    info: dict[str, Any] = {
        "path": str(p),
        "exists": p.is_file(),
        "duration_s": 0.0,
        "size_kb": 0.0,
    }
    if not p.is_file():
        return info
    try:
        info["size_kb"] = round(p.stat().st_size / 1024, 1)
    except OSError as e:
        record_failure('pingpong.voice_note_info:274', e)
    info["duration_s"] = round(ogg_duration(str(p)), 1)
    return info


def ogg_duration(ogg_path: str) -> float:
    """Length of an .ogg voice note in seconds (ffprobe).  0.0 on failure."""
    import subprocess
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", ogg_path],
            capture_output=True, timeout=10)
        return max(0.0, float((proc.stdout or b"").decode().strip() or 0))
    except Exception as e:
        record_failure('pingpong.ogg_duration:289', e)
        return 0.0


def to_ogg(src_path: str) -> str:
    """Convert any audio file to opus .ogg (Telegram sendVoice format).

    The conversion itself is the canonical
    :func:`nomorals.tools.audio.wav_to_voice_ogg`; this wrapper keeps
    the ping-pong contract — raise RuntimeError when no OGG could be
    produced, so callers fall back to text instead of sending an
    unconverted file dressed as a voice note.
    """
    src = Path(src_path)
    if src.suffix.lower() == ".ogg":
        return str(src)
    from ..tools.audio import wav_to_voice_ogg

    converted, mime = wav_to_voice_ogg(str(src))
    if mime == "audio/ogg":
        return converted
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not installed — cannot make voice notes")
    raise RuntimeError(
        f"ffmpeg conversion failed: {src} did not convert to OGG/Opus")
