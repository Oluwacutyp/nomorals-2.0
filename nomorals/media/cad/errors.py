"""CadCraft errors — the typed error for the CAD organ.

Port of the *design* of the layout/vector organs' error kinds
(lanes 101/104): one exception, one ``kind`` string, so callers
can tell a bad file from a bad argument from a solver that did
not converge — and so a refusal can name what is missing instead
of crashing.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["CadError", "CadErrorKind"]


class CadErrorKind(StrEnum):
    PARSE = "parse"
    IO = "io"
    UNSUPPORTED = "unsupported"
    INVALID_ARGUMENT = "invalid_argument"
    RENDER = "render"
    NOT_FOUND = "not_found"
    SOLVE = "solve"
    INTERNAL = "internal"


class CadError(Exception):
    """A CAD-organ failure with a machine-readable kind."""

    def __init__(self, message: str,
                 kind: CadErrorKind | str = CadErrorKind.INVALID_ARGUMENT
                 ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = (kind if isinstance(kind, CadErrorKind)
                     else CadErrorKind(kind))

    def __str__(self) -> str:
        return self.message
