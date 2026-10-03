"""Adaptive result-count inference for search surfaces.

Hardcoded ``max_results`` defaults (8 here, 6 there, 10 elsewhere) treat a
one-word price check and a "compare the best options" research question
identically.  This module infers a sensible result count from the query
itself — cheap, deterministic, offline-safe — so every search surface
(``SearchEngine.search``, deep-research fan-out, the research swarm, role
agents, video/product finders) scales its breadth to the question.

Heuristic, not model-based, on purpose: deciding *how many* results to
fetch must be free and instant, and it runs on every query including
offline ones.  The model still decides what the results *mean*.
"""

from __future__ import annotations

import re

#: words/phrases that signal the user wants breadth, not one answer
_BREADTH_HINTS = frozenset({
    "best", "top", "list", "guide", "tutorial", "review", "reviews",
    "alternatives", "options", "ideas", "ways", "examples", "resources",
    "sites", "tools", "services", "platforms", "jobs", "gigs",
    "compare", "comparison", "versus", "pros", "cons", "ranking",
    "cheapest", "free", "discount", "deals", "news", "latest",
})

#: question starters — an interrogative wants an explained answer, which
#: benefits from a couple more sources
_QUESTION_STARTERS = frozenset({
    "what", "which", "who", "where", "when", "why", "how",
})

_COMPARE_RE = re.compile(r"\bvs\.?\b|\bversus\b|\bcompare\b|\bcomparison\b")


def adaptive_result_limit(query: str, *, base: int = 8,
                          floor: int = 4, ceiling: int = 24) -> int:
    """Infer how many results a query deserves.

    ``base`` is the neutral default; the query's length, interrogative
    shape, and breadth signals nudge it up or down, clamped to
    ``[floor, ceiling]`` (the ceiling is a fetch-cost safety bound, not a
    quality knob — see the audit notes).
    """
    q = (query or "").strip().lower()
    if not q:
        return base
    tokens = q.split()
    n = len(tokens)
    limit = base
    if n <= 2:
        # "bitcoin price", "lagos weather" — a couple of hits suffice
        limit -= 3
    elif n <= 5:
        pass
    elif n <= 10:
        limit += 2
    else:
        # long, specific research questions deserve deeper coverage
        limit += 4
    if q.endswith("?") or (tokens and tokens[0] in _QUESTION_STARTERS):
        limit += 2
    if any(h in q for h in _BREADTH_HINTS):
        limit += 4
    if _COMPARE_RE.search(q):
        limit += 2
    return max(floor, min(ceiling, limit))
