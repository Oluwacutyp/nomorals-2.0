"""SoundCraft errors — typed kinds, honest refusals (lane 107).

Every failure names what went wrong and what to do about it; a
refusal is never silent success.
"""

from __future__ import annotations

__all__ = ["SoundError", "SoundErrorKind"]


class SoundErrorKind:
    BAD_FILE = "bad_file"
    BAD_FORMAT = "bad_format"
    UNSUPPORTED = "unsupported"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    TOO_LARGE = "too_large"
    RENDER_FAILED = "render_failed"
    PLUGIN_FAILED = "plugin_failed"
    REALTIME_VIOLATION = "realtime_violation"
    OTHER = "other"


class SoundError(Exception):
    """A SoundCraft failure carrying a typed kind."""

    def __init__(self, message: str, kind: str = SoundErrorKind.OTHER):
        super().__init__(message)
        self.kind = kind
