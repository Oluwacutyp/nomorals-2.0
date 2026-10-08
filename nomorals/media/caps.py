"""Write-size caps for generated media (rex-disk mitigation).

Every media write path in :mod:`nomorals.media` checks the *expected* byte
count against the cap for its media type **before** opening the file.  When
the cap would be exceeded the write is refused honestly:

* nothing is written (no partial/truncated file),
* a human-readable reason is logged and returned,
* nothing raises — callers get ``None`` / ``False`` and the reason.

Caps are deliberately generous for legitimate use: 500 MiB is ~99 minutes
of 16-bit mono audio at 44.1 kHz, so full audiobook chapters and long
mixes fit comfortably (raw WAV itself tops out at 4 GiB).
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

#: Cap for generated audio (WAV/MP3/…) files: 500 MiB.
MAX_AUDIO_WRITE_BYTES = 500 * 1024 * 1024

#: Cap for generated video files: 4 GiB.
MAX_VIDEO_WRITE_BYTES = 4 * 1024 * 1024 * 1024

#: Cap for generated image files: 200 MiB.
MAX_IMAGE_WRITE_BYTES = 200 * 1024 * 1024

#: Generic cap used by helpers that do not know the media type.
MAX_WRITE_BYTES = MAX_AUDIO_WRITE_BYTES

#: Bytes of RIFF/WAVE container overhead added by ``wave`` around raw frames.
WAV_HEADER_BYTES = 44


def check_write_size(nbytes: int, cap: int = MAX_WRITE_BYTES) -> tuple[bool, str]:
    """Return ``(True, "")`` if ``nbytes`` fits under ``cap``.

    Otherwise return ``(False, reason)`` where ``reason`` is a
    human-readable refusal explaining the limit.  Never raises.
    """
    try:
        n = int(nbytes)
    except (TypeError, ValueError):
        n = -1
    if n >= 0 and n <= cap:
        return True, ""
    mib = cap / (1024 * 1024)
    return False, (
        f"write refused: {nbytes!s} byte(s) requested exceeds the "
        f"{mib:.0f} MiB ({cap:,} bytes) write cap"
    )


def wav_expected_bytes(frames_bytes: int) -> int:
    """Total on-disk bytes for a ``wave``-written frame buffer."""
    return int(frames_bytes) + WAV_HEADER_BYTES


def refuse_write(what: str, reason: str) -> None:
    """Log an honest refusal; shared so every site phrases it the same."""
    _log.warning("%s: %s; nothing was written", what, reason)
