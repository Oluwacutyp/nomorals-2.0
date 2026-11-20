"""Rig errors — refusals that name the bone, file, or value.

Every failure in the rig organ is a :class:`RigError` carrying a
machine-readable ``kind`` and a message that names what is wrong,
never a bare ``ValueError`` from three calls up.
"""

from __future__ import annotations

__all__ = ["RigError"]


class RigError(Exception):
    """A rig refusal: what failed, and the kind of failure."""

    def __init__(self, message: str, *, kind: str = "rig_error") -> None:
        super().__init__(message)
        self.kind = kind
