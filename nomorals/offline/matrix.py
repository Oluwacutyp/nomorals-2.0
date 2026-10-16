"""Degraded-capability matrix: feature -> offline level, as a DATA table.

Levels:

* ``full`` — works offline exactly as online (local compute / local data).
* ``degraded`` — works offline with reduced freshness / scope, or the
  action is queued for sync (never silently dropped).
* ``unavailable`` — cannot work offline; the caller must say so
  honestly instead of faking a result. Trading execution and payments
  are deliberately ``unavailable`` offline: money never moves on
  stale state.

This is data, not code branching: callers look a feature up with
``offline_level(feature)`` and degrade accordingly. Unknown features
return ``"unavailable"`` (fail-closed) — the matrix is the source of
truth for what offline *can* do, and an unlisted feature claims
nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["OFFLINE_MATRIX", "LEVELS", "offline_level", "features_at", "matrix_table"]

LEVELS: tuple[str, ...] = ("full", "degraded", "unavailable")

#: feature -> {level, notes}. Keep sorted by feature for readability.
OFFLINE_MATRIX: dict[str, dict[str, str]] = {
    "chat_reply": {
        "level": "full",
        "notes": "On-device model answers from local context; no cloud needed.",
    },
    "memory_read": {
        "level": "full",
        "notes": "Local store reads work offline.",
    },
    "memory_write": {
        "level": "degraded",
        "notes": "Writes queue for sync; reads see them locally at once.",
    },
    "reminders": {
        "level": "full",
        "notes": "Local reminders fire on local clocks.",
    },
    "location_triggers": {
        "level": "full",
        "notes": "Geofence evaluation is pure geometry over supplied updates.",
    },
    "location_reminders": {
        "level": "full",
        "notes": "Fire from fence events; no network involved.",
    },
    "family_items": {
        "level": "full",
        "notes": "Per-person files are local; sharing is household-local.",
    },
    "media_playback": {
        "level": "full",
        "notes": "Cached / downloaded media plays offline.",
    },
    "voice_tts": {
        "level": "full",
        "notes": "Local TTS engine; no cloud voice needed.",
    },
    "voice_stt": {
        "level": "full",
        "notes": "Local speech recognition; no cloud transcription.",
    },
    "note_taking": {
        "level": "degraded",
        "notes": "Local notes queue for sync; search covers local notes only.",
    },
    "email_send": {
        "level": "degraded",
        "notes": "Composed offline, queued, sent on reconnect.",
    },
    "email_read": {
        "level": "degraded",
        "notes": "Cached inbox only; no new mail until reconnect.",
    },
    "social_post": {
        "level": "degraded",
        "notes": "Drafted + queued; posted on reconnect, never faked as sent.",
    },
    "social_read": {
        "level": "degraded",
        "notes": "Cached feeds only; no fresh timeline until reconnect.",
    },
    "file_sync": {
        "level": "degraded",
        "notes": "Local edits queue; remote copies sync on reconnect.",
    },
    "maps": {
        "level": "degraded",
        "notes": "Cached tiles / saved places only; no live routing.",
    },
    "music_discovery": {
        "level": "degraded",
        "notes": "Local library + cached recommendations; no fresh catalogue.",
    },
    "trading_signals": {
        "level": "degraded",
        "notes": "Signals from last-known prices, stamped STALE; never live.",
    },
    "calendar": {
        "level": "degraded",
        "notes": "Local calendar works; shared calendars sync on reconnect.",
    },
    "web_search": {
        "level": "unavailable",
        "notes": "Needs the network. Say so; do not fabricate results.",
    },
    "news": {
        "level": "unavailable",
        "notes": "Needs fresh feeds. Last cached briefing may be shown, stamped.",
    },
    "media_generation": {
        "level": "unavailable",
        "notes": "Cloud image/video models need the network.",
    },
    "trading_execution": {
        "level": "unavailable",
        "notes": "Money never moves on stale state. Orders queue as INTENTS only.",
    },
    "payments": {
        "level": "unavailable",
        "notes": "Payments need live confirmation. Never confirm offline.",
    },
    "cloud_backup": {
        "level": "unavailable",
        "notes": "Backups resume on reconnect.",
    },
    "software_updates": {
        "level": "unavailable",
        "notes": "Updates download only when online.",
    },
}


def offline_level(feature: str) -> str:
    """Offline level for ``feature``. Unknown features → ``unavailable``."""
    entry = OFFLINE_MATRIX.get(feature)
    if entry is None:
        return "unavailable"
    return entry["level"]


def features_at(level: str) -> list[str]:
    """Sorted features whose offline level is exactly ``level``."""
    if level not in LEVELS:
        raise ValueError(f"unknown offline level {level!r}; expected {LEVELS}")
    return sorted(f for f, e in OFFLINE_MATRIX.items() if e["level"] == level)


def matrix_table() -> list[dict[str, Any]]:
    """The whole matrix as sorted rows: feature, level, notes."""
    return [{"feature": f, "level": e["level"], "notes": e["notes"]}
            for f, e in sorted(OFFLINE_MATRIX.items())]
