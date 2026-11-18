"""Ids: who an extension is and which app it targets.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
id.rs``, dual MIT/Apache): ids are lowercase ASCII from a tiny
alphabet so look-alikes (case, Unicode homoglyphs) cannot exist, and
a validated id can be used as a folder name on every platform.

Devon's first-party prefixes are her own namespaces — a package a
person adds from a file may never claim one (gate G1 upstream).
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "AppId",
    "ExtensionId",
    "FIRST_PARTY_PREFIXES",
    "IdError",
    "IdErrorKind",
]

MAX_EXTENSION_ID = 64
MAX_APP_ID = 32

#: Prefixes only the first-party tier may use. A third-party package
#: claiming one is refused.
FIRST_PARTY_PREFIXES: tuple[str, ...] = (
    "ai.devon.",
    "org.devon.",
    "io.github.oluwacutyp.",
)


class IdErrorKind(Enum):
    """Why an id was refused."""

    EMPTY = "the id is empty"
    TOO_LONG = "the id is too long"
    BAD_CHARACTER = ("the id may only use lowercase letters, digits, "
                     "dots, dashes and underscores")
    BAD_SHAPE = ("the id must be dotted segments such as "
                 "org.example.tone, each starting with a letter or digit")


class IdError(ValueError):
    """An id was refused, with the reason on ``.kind``."""

    def __init__(self, kind: IdErrorKind) -> None:
        super().__init__(kind.value)
        self.kind = kind


def _check_extension_shape(text: str) -> None:
    if not text:
        raise IdError(IdErrorKind.EMPTY)
    if len(text) > MAX_EXTENSION_ID:
        raise IdError(IdErrorKind.TOO_LONG)
    if not all(c.isascii() and (c.islower() or c.isdigit()
                                or c in ".-_") for c in text):
        raise IdError(IdErrorKind.BAD_CHARACTER)
    segments = text.split(".")
    if len(segments) < 2:
        raise IdError(IdErrorKind.BAD_SHAPE)
    for segment in segments:
        edge_ok = (segment[:1].isascii()
                   and (segment[:1].islower() or segment[:1].isdigit())
                   and (segment[-1:].isascii())
                   and (segment[-1:].islower() or segment[-1:].isdigit()))
        if not edge_ok:
            raise IdError(IdErrorKind.BAD_SHAPE)


class ExtensionId(str):
    """A validated extension id such as ``org.example.tone``."""

    @classmethod
    def new(cls, text: str) -> ExtensionId:
        """Validate and return the id, or raise IdError."""
        _check_extension_shape(text)
        return cls(text)

    @classmethod
    def try_new(cls, text: str) -> ExtensionId | None:
        """Validate and return the id, or None when it is refused."""
        try:
            return cls.new(text)
        except IdError:
            return None

    def in_first_party_namespace(self) -> bool:
        """True when the id sits under a prefix reserved for Devon."""
        return any(self.startswith(p) for p in FIRST_PARTY_PREFIXES)


class AppId(str):
    """A validated app id such as ``studio``."""

    @classmethod
    def new(cls, text: str) -> AppId:
        """Validate and return the id, or raise IdError."""
        if not text:
            raise IdError(IdErrorKind.EMPTY)
        if len(text) > MAX_APP_ID:
            raise IdError(IdErrorKind.TOO_LONG)
        if not all(c.isascii() and (c.islower() or c.isdigit() or c == "-")
                   for c in text):
            raise IdError(IdErrorKind.BAD_CHARACTER)
        if not text[0].islower() or text.endswith("-"):
            raise IdError(IdErrorKind.BAD_SHAPE)
        return cls(text)

    @classmethod
    def try_new(cls, text: str) -> AppId | None:
        """Validate and return the id, or None when it is refused."""
        try:
            return cls.new(text)
        except IdError:
            return None
