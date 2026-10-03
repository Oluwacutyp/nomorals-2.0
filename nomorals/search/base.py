"""Shared adapter contract for federated search sources.

Kept in its own leaf module so both ``sources.py`` (local adapters) and
``web.py`` (live web backends) can share the base class without an
import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import SearchResult

__all__ = ["SourceAdapter"]


class SourceAdapter:
    """One searchable subsystem. ``probe`` returns None when searchable,
    otherwise a human note explaining why the source is skipped."""

    name: str = ""
    result_type: str = ""
    description: str = ""

    def probe(self) -> str | None:
        return None

    def search(
        self,
        query: str,
        *,
        limit: int,
        since: float | None = None,
        before: float | None = None,
    ) -> list["SearchResult"]:
        raise NotImplementedError
