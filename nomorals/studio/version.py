"""Versions: plain ``major.minor.patch``, no pre-release tags in v1.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
version.rs``, dual MIT/Apache) — the version is a string, so it
compares and serialises like one, and orders by its numbers.
"""

from __future__ import annotations

from functools import total_ordering

__all__ = ["Version", "VersionError"]


class VersionError(ValueError):
    """A version does not look like 1.2.3."""


@total_ordering
class Version(str):
    """A validated ``major.minor.patch`` version, as a string."""

    def __new__(cls, major: int, minor: int, patch: int) -> Version:
        text = f"{major}.{minor}.{patch}"
        obj = super().__new__(cls, text)
        obj.major = major
        obj.minor = minor
        obj.patch = patch
        return obj

    @classmethod
    def parse(cls, text: str) -> Version:
        """Parse ``1.2.3`` or raise VersionError."""
        parts = text.split(".")
        if len(parts) != 3:
            raise VersionError("a version looks like 1.2.3")
        numbers: list[int] = []
        for part in parts:
            if (not part or len(part) > 10
                    or not part.isascii() or not part.isdigit()
                    or (len(part) > 1 and part.startswith("0"))):
                raise VersionError("a version looks like 1.2.3")
            try:
                value = int(part)
            except ValueError:
                raise VersionError("a version looks like 1.2.3") from None
            if value > 0xFFFFFFFF:
                raise VersionError("a version looks like 1.2.3")
            numbers.append(value)
        return cls(numbers[0], numbers[1], numbers[2])

    def _key(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            return self._key() == other._key()
        if isinstance(other, str):
            try:
                return self._key() == Version.parse(other)._key()
            except VersionError:
                return False
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Version):
            return self._key() < other._key()
        if isinstance(other, str):
            return self._key() < Version.parse(other)._key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))
