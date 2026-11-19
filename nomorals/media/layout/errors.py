"""Layout errors — typed failures for the layout organ.

Mirrors the typed-error habit of the vector organ (lane 101):
every failure names its kind so callers and the tool can report
honestly instead of raising a bare ``ValueError``.
"""

from __future__ import annotations

__all__ = ["LayoutError", "LayoutErrorKind"]

#: Every kind a layout operation can fail with.
LayoutErrorKind = (
    "parse", "io", "unsupported", "invalid_argument", "render",
    "not_found",
)


class LayoutError(Exception):
    """A layout-organ failure with a machine-readable kind."""

    def __init__(self, message: str, kind: str = "invalid_argument"):
        super().__init__(message)
        if kind not in LayoutErrorKind:
            raise ValueError(f"unknown layout error kind {kind!r}")
        self.kind = kind
