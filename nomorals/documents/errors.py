"""Document engine errors."""

from __future__ import annotations

__all__ = ["DocumentError"]


class DocumentError(Exception):
    """Raised when a document cannot be parsed or converted.

    Always carries a human-readable reason — callers should surface
    ``str(exc)`` rather than guessing what went wrong.
    """
