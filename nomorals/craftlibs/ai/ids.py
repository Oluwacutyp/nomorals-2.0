"""Validated string types (Craft-Libs port of ai-core ``ids.rs``).

Every identifier that crosses a trust boundary is a validated
newtype, so a malformed value is an error at the edge and never a
surprise deeper in. Validation is by character class and length,
with no indexing, so hostile input cannot panic.
"""

from __future__ import annotations

__all__ = [
    "MAX_ID_LEN",
    "AppId",
    "ConnectionId",
    "IdError",
    "JobId",
    "ModelId",
    "RoleId",
    "SecretRef",
    "validate_id",
    "validate_token",
]

MAX_ID_LEN = 128


class IdError(ValueError):
    """Why a string was refused as an identifier."""


def _check(value: str, ok) -> None:
    if not value:
        raise IdError("value is empty")
    if len(value.encode("utf-8")) > MAX_ID_LEN:
        raise IdError(
            f"value is {len(value.encode('utf-8'))} bytes, "
            f"the limit is {MAX_ID_LEN}")
    for char in value:
        if not ok(char):
            raise IdError(f"character {char!r} is not allowed")


def validate_id(value: str) -> None:
    """General identifiers: ASCII letters and digits plus
    ``- _ . : /`` (model names such as ``vendor/model-1.5``)."""
    _check(value, lambda c: c.isascii() and c.isalnum()
          or c in "-_.:/")


def validate_token(value: str) -> None:
    """Tokens: ASCII letters and digits plus ``- _``. No dot, so
    ``modality.task`` splits unambiguously."""
    _check(value, lambda c: c.isascii() and c.isalnum() or c in "-_")


class _Validated(str):
    """A validated string newtype. ``new`` validates and wraps;
    the value itself behaves like a plain ``str``."""

    _validator = staticmethod(validate_id)

    def __new__(cls, value: str):
        return super().__new__(cls, value)

    @classmethod
    def new(cls, value):
        text = str(value)
        cls._validator(text)
        return cls(text)

    def as_str(self) -> str:
        return str(self)


class ConnectionId(_Validated):
    """A connection's id."""


class ModelId(_Validated):
    """A model's id."""


class RoleId(_Validated):
    """A role's id."""


class AppId(_Validated):
    """An app's id."""


class JobId(_Validated):
    """A job's id."""


class SecretRef(_Validated):
    """A locator for a secret value, never the value itself."""
