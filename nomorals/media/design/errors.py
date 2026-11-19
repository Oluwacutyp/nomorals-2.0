"""FigCraft errors — typed failures for the design organ.

Mirrors the typed-error habit of the vector organ (lane 101):
every failure names its kind so callers and the tool can report
honestly instead of raising a bare ``ValueError``.
"""

from __future__ import annotations

__all__ = ["DesignError", "DesignErrorKind"]

#: Every kind a design operation can fail with.
DesignErrorKind = (
    "bad_file", "bad_format", "unsupported", "invalid_argument",
    "not_found", "cycle", "permission", "render", "io", "cancelled",
    "other",
)


class DesignError(Exception):
    """A design-organ failure with a machine-readable kind."""

    def __init__(self, message: str, kind: str = "invalid_argument"):
        super().__init__(message)
        if kind not in DesignErrorKind:
            raise ValueError(f"unknown design error kind {kind!r}")
        self.kind = kind
