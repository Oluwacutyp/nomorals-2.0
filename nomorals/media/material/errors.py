"""Material errors — typed failures for the materials organ.

MaterialCraft port: every failure is a named kind with a message the
owner or an agent can act on. A malformed package, a hostile command
or an odd parameter produces an error, never a crash.
"""

from __future__ import annotations

import enum


class MaterialErrorKind(enum.StrEnum):
    """Typed failure kinds for material work."""

    bad_file = "bad_file"
    bad_format = "bad_format"
    unsupported = "unsupported"
    not_found = "not_found"
    invalid_argument = "invalid_argument"
    too_large = "too_large"
    cook_failed = "cook_failed"
    cycle = "cycle"
    other = "other"


class MaterialError(Exception):
    """A material operation failed, with a named kind."""

    def __init__(self, kind: MaterialErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:  # noqa: D105
        return f"{self.kind.value}: {self.message}"
