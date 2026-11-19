"""Wiki search: plain token ranking over title, tags and body.

No index to build, no provider to call — the wiki is small and the
search is honest about it. A title hit outranks a tag hit, a tag hit
outranks a body hit, and an exact phrase outranks all three.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["search_pages"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _score(page: Any, query_tokens: set[str], phrase: str) -> float:
    title = _tokens(page.title)
    tags = {t.lower() for t in (page.tags or [])}
    body = _tokens(page.body)
    score = 0.0
    for token in query_tokens:
        if token in title:
            score += 3.0
        elif token in tags:
            score += 2.0
        elif token in body:
            score += 1.0
        else:
            return 0.0
    haystack = f"{page.title}\n{page.body}".lower()
    if phrase and len(query_tokens) > 1 and phrase in haystack:
        # A multi-word phrase outranks single-token hits; a lone
        # token must not double-count itself as a "phrase".
        score += 5.0
    return score


def search_pages(store: Any, query: str, *, limit: int = 20) -> list[Any]:
    """Rank the pages matching ``query`` — every query token must hit.

    Returns best-first; an empty query or no match returns ``[]``,
    never a fabricated page.
    """
    query = str(query or "").strip()
    if not query:
        return []
    tokens = _tokens(query)
    if not tokens:
        return []
    phrase = query.lower()
    scored: list[tuple[float, Any]] = []
    for page in store.list_pages():
        score = _score(page, tokens, phrase)
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda pair: (-pair[0], pair[1].title.lower()))
    return [page for _, page in scored[:max(1, int(limit))]]
