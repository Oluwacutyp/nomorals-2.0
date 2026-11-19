"""Vector errors — typed failures for the vector organ.

Mirrors the typed-error habit of the develop engine (lane 98):
every failure names its kind so callers and the tool can report
honestly instead of raising a bare ``ValueError``.
"""

from __future__ import annotations

__all__ = ["VectorError", "VectorErrorKind"]

#: Every kind a vector operation can fail with.
VectorErrorKind = (
    "parse", "io", "unsupported", "invalid_argument", "render",
    "not_found",
)


class VectorError(Exception):
    """A vector-organ failure with a machine-readable kind."""

    def __init__(self, message: str, kind: str = "invalid_argument"):
        super().__init__(message)
        if kind not in VectorErrorKind:
            raise ValueError(f"unknown vector error kind {kind!r}")
        self.kind = kind
