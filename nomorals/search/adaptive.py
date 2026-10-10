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

#: freshness signals — the user wants *recent* information, so recency
#: should bias ranking and news/category routing should kick in
_FRESHNESS_HINTS = frozenset({
    "latest", "newest", "recent", "breaking", "today", "yesterday",
    "this week", "this month", "2025", "2026", "2027", "news",
    "update", "updates", "now",
})

#: navigational intent — "take me to X", one right answer
_NAV_HINTS = frozenset({
    "login", "sign in", "homepage", "website", "official site", "download",
    "docs", "documentation", "github", "repo",
})

#: transactional intent — the user wants to *do* something
_TRANSACTION_HINTS = frozenset({
    "buy", "price", "prices", "cheap", "coupon", "order", "book",
    "hire", "subscribe", "signup", "sign up", "rent",
})

_JUNK_RE = re.compile(r"[^\w\s@.+\-/:?!,']+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([?!.,])\1+")


def normalize_query(query: str) -> str:
    """Clean a raw query before it reaches any backend.

    Collapses whitespace, strips junk punctuation (keeping the
    characters queries legitimately use: ``@`` emails, ``+``/``-``/``/``
    paths and operators, ``:``/``?``/``!``/``,``/``'``), and drops
    empty tokens. Deterministic and backend-agnostic.
    """
    q = (query or "").strip()
    if not q:
        return ""
    q = _JUNK_RE.sub(" ", q)
    q = _REPEAT_PUNCT_RE.sub(r"\1", q)  # "what?!" stays, "!!!" → "!"
    q = _WS_RE.sub(" ", q).strip()
    return q


def freshness_intent(query: str) -> bool:
    """Does the query ask for *recent* information?

    True when it carries explicit recency signals ("latest", "news",
    "2026", "this week"...). Callers use this to route to news
    categories/verticals and to bias toward fresh timestamps instead of
    treating a 2021 blog post and today's news as equals.
    """
    q = (query or "").lower()
    return any(h in q for h in _FRESHNESS_HINTS)


def detect_intent(query: str) -> str:
    """Classify the query's search intent (heuristic, deterministic).

    One of ``"navigational"`` (one right destination), ``"transactional"``
    (do/buy something), ``"osint"`` (username/email/domain/IP/phone shape),
    or ``"informational"`` (the default — learn about something). Used to
    pick result categories and sources, not to gate anything.
    """
    q = (query or "").strip().lower()
    if not q:
        return "informational"
    # OSINT shapes first — they are the most specific
    bare = q.lstrip("@")
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", q):
        return "osint"
    if re.fullmatch(r"\+?\d{7,15}", re.sub(r"[^\d+]", "", q)):
        return "osint"
    if re.fullmatch(r"[a-z0-9.\-]+\.[a-z]{2,}", bare) and " " not in bare:
        return "osint"
    if re.fullmatch(r"[A-Za-z0-9_.\-]{2,39}", bare) and " " not in q:
        return "osint"
    if any(h in q for h in _NAV_HINTS):
        return "navigational"
    if any(h in q for h in _TRANSACTION_HINTS):
        return "transactional"
    return "informational"


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
    if freshness_intent(q):
        # recency-seeking queries churn: a couple more slots so fresh
        # hits aren't crowded out by evergreen ones
        limit += 2
    return max(floor, min(ceiling, limit))
