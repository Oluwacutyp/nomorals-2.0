"""VTuber show errors — every refusal names what it needs."""

from __future__ import annotations

__all__ = ["VtuberError"]


class VtuberError(Exception):
    """A vtuber refusal, with a machine-readable kind."""

    def __init__(self, message: str, *, kind: str = "vtuber_error"):
        super().__init__(message)
        self.kind = kind
