"""Versions: ``X.Y.Z[-prerelease]`` with semver ordering.

Ported from upstream ``crates/core/src/version.rs`` — the launcher's
release chooser compares versions the same way: numeric parts first,
a release beats any prerelease of the same numbers, prerelease labels
compare per semver (numeric labels sort before alphanumeric ones and
compare numerically). A version that does not parse is refused, never
guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering

__all__ = ["Version", "is_newer", "is_stable"]

_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z.-]+))?$"
)


def _label_key(label: str) -> tuple[int, object]:
    # Semver §11: numeric identifiers compare numerically and sort
    # before alphanumeric ones; alphanumeric ones compare lexically.
    if label.isdigit():
        return (0, int(label))
    return (1, label)


@total_ordering
@dataclass(frozen=True)
class Version:
    """An immutable ``X.Y.Z[-prerelease]`` version."""

    major: int
    minor: int
    patch: int
    prerelease: str = ""

    @classmethod
    def parse(cls, text: str) -> Version:
        """Parse ``text`` or raise ValueError naming what was wrong."""
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise ValueError(f"not a version (want X.Y.Z[-label]): {text!r}")
        major, minor, patch, pre = match.groups()
        return cls(int(major), int(minor), int(patch), pre or "")

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        mine = (self.major, self.minor, self.patch)
        theirs = (other.major, other.minor, other.patch)
        if mine != theirs:
            return mine < theirs
        if self.prerelease == other.prerelease:
            return False
        if not self.prerelease:
            return False  # a release beats any prerelease of the same numbers
        if not other.prerelease:
            return True
        mine_labels = self.prerelease.split(".")
        theirs_labels = other.prerelease.split(".")
        for mine_label, theirs_label in zip(
                mine_labels, theirs_labels, strict=False):
            if mine_label == theirs_label:
                continue
            return _label_key(mine_label) < _label_key(theirs_label)
        return len(mine_labels) < len(theirs_labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch, self.prerelease) == (
            other.major, other.minor, other.patch, other.prerelease)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))


def is_stable(version: Version) -> bool:
    """True when ``version`` carries no prerelease label."""
    return not version.prerelease


def is_newer(candidate: str, current: str) -> bool:
    """True when ``candidate`` is a strictly newer version than ``current``."""
    return Version.parse(candidate) > Version.parse(current)
