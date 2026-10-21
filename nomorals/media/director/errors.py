"""Director pipeline errors — always plain-language, never silent."""

from __future__ import annotations

__all__ = ["DirectorError"]


class DirectorError(Exception):
    """A director-pipeline failure the caller can act on."""
