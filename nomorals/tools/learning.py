"""Learning tool: ``free_learning`` — the brain's free-learning finder.

Devon is asked "I want to learn X, free" and answers from her curated
collection first — vetted entries (checked live, with the check date),
then unchecked ones (labelled as such), flagged ones (shown with the
reason, never recommended) — and live search second, clearly
separated. The tool is a thin surface over
:mod:`nomorals.sources.learning`; the organ owns the collection, the
health probes, and the live-search rungs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..sources.learning import (
    LearningCollection,
    LiveLearningSearch,
    learn_free,
)

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home", None)
    if home:
        return str(home)
    home_path = getattr(settings, "home_path", None)
    if home_path:
        return str(Path(home_path))
    return None


def register(registry: Any) -> None:
    """Attach the ``free_learning`` tool to a registry."""
    context = registry.context

    @registry.register(
        "free_learning",
        description=(
            "Find free ways to learn a subject (OpenCulture-curated "
            "collection of free courses, textbooks, audiobooks, ebooks, "
            "movies and language lessons, grown by live search over "
            "OpenCulture, Archive.org and Open Library). Use this when "
            "the user says 'I want to learn X, free' or asks for free "
            "courses, textbooks, audiobooks or lessons on a subject. "
            "subject=<what to learn>; format=<course|textbook|audiobook|"
            "ebook|movie|lesson> filters by kind; level=<beginner|"
            "intermediate|advanced> filters by level; limit caps results; "
            "recheck=true re-probes entries whose health check is stale. "
            "Returns vetted entries first (checked live, with the check "
            "date), then unchecked ones (labelled 'not yet checked'), "
            "then flagged ones (dead or blocked, with the reason, never "
            "recommended), then live-search results, clearly separated."
        ),
        capability=Capability.NET_OUT,
    )
    def free_learning_tool(
        subject: str = "", format: str = "", level: str = "",
        limit: int = 5, recheck: bool = False,
    ) -> dict[str, Any]:
        if not subject or not str(subject).strip():
            raise ValueError("free_learning needs a subject to learn")
        home = _home_from_context(context)
        collection = LearningCollection(home)
        if recheck:
            from ..sources.learning import refresh_health

            refresh_health(collection)
        searcher = LiveLearningSearch()
        return learn_free(
            collection, str(subject).strip(),
            format=format.strip() or None,
            level=level.strip() or None,
            limit=limit, live_search=searcher)
