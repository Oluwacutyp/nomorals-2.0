"""AniCraft port — typed failures.

Every failure names what is wrong and where; nothing raises a bare
``ValueError`` out of the organ boundary — the tool translates
``AnimationError`` into a plain reply.
"""

from __future__ import annotations

__all__ = ["AnimationError"]


class AnimationError(ValueError):
    """A document, timeline, render or publish problem — named, not silent."""
