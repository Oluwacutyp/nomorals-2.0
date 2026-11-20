"""Face mocap errors — one error type, kinds named.

The bevy-facial-mocap pattern (storytold/bevy-mocap, pattern
only) names every failure at the point it happens — a bad
timestamp, a missing morph, a malformed mapping. Devon's organ
does the same: a refused file names the file and the reason,
never a bare traceback.
"""

from __future__ import annotations

__all__ = ["FaceMocapError"]


class FaceMocapError(Exception):
    """A face-mocap refusal, with a machine-readable kind."""

    def __init__(self, message: str, *, kind: str = "face_error"):
        super().__init__(message)
        self.kind = kind
