"""Music voice notes: detect music in a voice note, then ask Find vs Create.

Flow (owner-only ask — see :func:`pending` / the runtime hook):

1. A bare voice note arrives with no text. STT runs as usual; speech
   transcribes and flows on untouched — this module never sees it.
2. When transcription fails (typically "empty transcript" — music has
   no words), the runtime calls :func:`detect_music` on the clip.
   Detection is CONSERVATIVE: only a clip the native analyzer scores
   as clearly music (``music_score >= 0.6``, the same cutoff
   ``audio.fingerprint.describe_audio`` uses to call a clip "music")
   that is long enough to judge and not mostly silence counts. Speech
   (score <= 0.4), mixed/uncertain clips (0.4–0.6), silence, and
   too-short blips are NOT music — they fall through to the existing
   honest "couldn't transcribe" reply, never to the ask.
3. Music detected + OWNER → the runtime sends :data:`ASK_TEXT` with
   :func:`ask_buttons` (Find / Create) and stashes the clip in the
   pending store. The ask IS the gate: no production runs until the
   owner taps Create. Tapping Find runs the explicit /sham music-ID
   path on the stashed clip; tapping Create runs music production
   from the clip. Either tap consumes the pending entry.
4. Music detected + NON-OWNER DM → NO ask, NO buttons. Devon decides:
   the note is re-presented to the brain as music context
   (``[voice note — music playing: …]``) and the conversation proceeds
   normally — she may reply in voice or text, her choice.

This module never raises, never touches the normal speech path, and
never auto-runs music-ID or production: both run only on the owner's
explicit tap (or an explicit /sham / /music command).
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = [
    "ASK_TEXT",
    "CREATE_CALLBACK",
    "FIND_CALLBACK",
    "MUSIC_SCORE_MIN",
    "PendingMusicNote",
    "PENDING",
    "ask_buttons",
    "detect_music",
    "music_context_text",
]

#: Conservative music bar — matches describe_audio()'s "music" cutoff.
MUSIC_SCORE_MIN = 0.6
#: A clip shorter than this is a blip, not a song worth asking about.
MIN_SECONDS = 1.0
#: Mostly-silence clips are not music, whatever the score says.
MAX_SILENCE_RATIO = 0.5
#: A pending ask expires: a tap an hour later must not run against a
#: clip from a different conversation.
PENDING_TTL_S = 600.0

#: The ask, verbatim. Short, one question, two buttons.
ASK_TEXT = (
    "That's music, not speech 🎵\n"
    "Find this music (like Shazam) or create a new song from it?"
)

#: Button callbacks — real routable commands (``/musicnote find`` /
#: ``/musicnote create``), well inside the 47-byte callback budget.
FIND_CALLBACK = "musicnote find"
CREATE_CALLBACK = "musicnote create"


def ask_buttons() -> list[list[tuple[str, str]]]:
    """The Find / Create keyboard, via the adaptive-buttons system.

    The kind handler in :mod:`nomorals.social.chat.adaptive_buttons`
    owns the button definitions; the literal fallback only guards
    against the ask ever going out buttonless if that handler is
    somehow unavailable.
    """
    try:
        from ..social.chat.adaptive_buttons import adaptive_buttons_for

        kb = adaptive_buttons_for(output_kind="music-ask")
        if kb:
            return kb
    except Exception:  # noqa: BLE001 - the ask must never go out buttonless
        pass
    return [[("🔍 Find this music", FIND_CALLBACK),
             ("🎵 Create a song", CREATE_CALLBACK)]]


def music_context_text(heard: str) -> str:
    """How a music voice note is re-presented to the brain (non-owner).

    Bracketed system description, never presented as user speech: the
    model sees that music is playing and what it sounds like, and
    decides its own reply from there.
    """
    detail = (heard or "").strip()
    if detail:
        return f"[voice note — music playing: {detail}]"
    return "[voice note — music playing]"


def detect_music(
    path: str,
    *,
    analyze: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Is this clip music (not speech)? Conservative. Never raises.

    Returns ``{"is_music": bool, "heard": str, "score": float,
    "seconds": float}`` — ``heard`` is a one-line description of the
    music ("" when not music). ``analyze`` is injectable for tests;
    by default it is :func:`nomorals.audio.fingerprint.analyze`.
    """
    out: dict[str, Any] = {"is_music": False, "heard": "",
                           "score": 0.0, "seconds": 0.0}
    if not path:
        return out
    try:
        if analyze is None:
            from ..audio.fingerprint import analyze as analyze
        a = analyze(path)
        ok = bool(getattr(a, "ok", False))
        score = float(getattr(a, "music_score", 0.0) or 0.0)
        seconds = float(getattr(a, "seconds", 0.0) or 0.0)
        silence = float(getattr(a, "silence_ratio", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - detection failure means "not music"
        return out
    out["score"] = score
    out["seconds"] = seconds
    if not ok:
        return out
    if score < MUSIC_SCORE_MIN:
        return out
    if seconds < MIN_SECONDS:
        return out
    if silence > MAX_SILENCE_RATIO:
        return out
    parts = [f"{seconds:.0f}s of music"]
    tempo = float(getattr(a, "tempo_bpm", 0.0) or 0.0)
    tempo_conf = float(getattr(a, "tempo_confidence", 0.0) or 0.0)
    if tempo and tempo_conf >= 0.15:
        parts.append(f"~{tempo:.0f} BPM")
    key = str(getattr(a, "key", "") or "")
    key_conf = float(getattr(a, "key_confidence", 0.0) or 0.0)
    if key and key_conf >= 0.55:
        parts.append(f"key of {key}")
    rms = float(getattr(a, "rms_db", 0.0) or 0.0)
    parts.append(f"loudness {rms:.0f} dBFS")
    out["is_music"] = True
    out["heard"] = " · ".join(parts)
    return out


class PendingMusicNote:
    """Per-chat pending music voice note, waiting on the owner's tap.

    One entry per chat (a second music note replaces the first —
    the ask on screen is always about the latest clip). Entries
    expire after :data:`PENDING_TTL_S`; a tap consumes its entry,
    so a double-tap can never run Find/Create twice.
    """

    def __init__(self) -> None:
        self._by_chat: dict[str, dict[str, Any]] = {}

    def remember(self, chat_key: str, path: str, heard: str = "") -> None:
        self._by_chat[str(chat_key or "")] = {
            "path": str(path or ""),
            "heard": str(heard or ""),
            "ts": time.time(),
        }

    def take(self, chat_key: str) -> dict[str, Any] | None:
        """Pop the pending entry for this chat (None when none/expired)."""
        entry = self._by_chat.pop(str(chat_key or ""), None)
        if entry is None:
            return None
        if time.time() - float(entry.get("ts", 0.0)) > PENDING_TTL_S:
            return None
        return entry

    def peek(self, chat_key: str) -> dict[str, Any] | None:
        """Read without consuming (tests / diagnostics)."""
        entry = self._by_chat.get(str(chat_key or ""))
        if entry is None:
            return None
        if time.time() - float(entry.get("ts", 0.0)) > PENDING_TTL_S:
            return None
        return dict(entry)

    def clear(self) -> None:
        self._by_chat.clear()


#: The process-wide pending store. The runtime hook writes it; the
#: /musicnote handler reads it. Tests use their own
#: :class:`PendingMusicNote` instance instead.
PENDING = PendingMusicNote()
