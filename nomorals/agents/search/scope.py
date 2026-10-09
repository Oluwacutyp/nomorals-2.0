"""Dual-scope awareness for search: Nigeria + US, served together.

The owner lives in Nigeria and also operates in a US context, so research
questions that are scope-sensitive (prices, news, jobs, law, shopping,
banking, "best X") must pull BOTH Nigerian and US sources — a US-centric
result set silently answers the wrong question.

Design, per the standing rules (dynamic, not hardcoded booleans):

* :func:`detect_scope` — explicit geography in the query wins
  (``"ng"`` / ``"us"`` / ``"both"`` / ``"auto"``).
* :func:`scope_relevance` — a 0.0–1.0 score for how scope-sensitive the
  query is, built from weighted term *categories* (finance, news, jobs,
  legal, consumer, local-intent). More category hits → higher score.
* :func:`regional_variants` — the actual query plan: the original query
  plus region-flavoured variants when (and only when) the query is
  scope-sensitive. Pure factoids ("what is photosynthesis") stay single-shot.

Everything is deterministic and pure — no I/O — so it is fully
unit-testable.
"""

from __future__ import annotations

import re

__all__ = [
    "detect_scope",
    "scope_relevance",
    "regional_variants",
    "NG_SIGNALS",
    "US_SIGNALS",
    "SCOPE_CATEGORIES",
]

#: Explicit geography that pins the query to Nigeria.
NG_SIGNALS = frozenset({
    "nigeria", "nigerian", "naija", "lagos", "abuja", "ibadan", "kano",
    "port harcourt", "enugu", "benin city", "ekiti", "naira", "₦",
    "yoruba", "hausa", "igbo", "pidgin",
})

#: Explicit geography that pins the query to the US.
US_SIGNALS = frozenset({
    "united states", "u.s.", "usa", "america", "american", "new york",
    "los angeles", "california", "texas", "chicago", "washington dc",
    "florida", "dollar", "$",
})

#: scope-sensitive term categories → weight. A query's scope relevance is
#: the capped sum of weights of the distinct categories it touches.
SCOPE_CATEGORIES: dict[str, tuple[float, frozenset[str]]] = {
    "finance": (0.50, frozenset({
        "price", "prices", "cost", "costs", "cheap", "cheapest", "expensive",
        "salary", "salaries", "pay", "wage", "bank", "banking", "loan",
        "mortgage", "forex", "exchange rate", "naira", "dollar", "investment",
        "stock", "stocks", "crypto", "bitcoin", "transfer", "remittance",
        "insurance", "pension",
    })),
    "news": (0.60, frozenset({
        "news", "latest", "today", "breaking", "election", "president",
        "governor", "government", "minister", "policy", "headline",
        "headlines", "happening",
    })),
    "jobs": (0.55, frozenset({
        "job", "jobs", "hiring", "career", "gig", "gigs", "freelance",
        "remote work", "vacancy", "recruitment", "internship",
    })),
    "legal": (0.50, frozenset({
        "law", "laws", "legal", "visa", "immigration", "tax", "taxes",
        "regulation", "court", "license", "permit", "citizenship",
    })),
    "consumer": (0.45, frozenset({
        "buy", "buying", "best", "review", "reviews", "shop", "shopping",
        "store", "delivery", "price", "deal", "deals", "discount",
        "where to",
    })),
    "local": (0.70, frozenset({
        "near me", "nearby", "local", "in my area", "around me",
        "closest", "nearest",
    })),
}

#: A query at or above this relevance fans out to both regions.
SCOPE_THRESHOLD = 0.40

#: Region-flavoured query templates used for the fan-out (both regions
#: covered, phrased the way each region's press actually writes them).
_REGION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ng": ("{q} Nigeria", "{q} Lagos"),
    "us": ("{q} United States", "{q} US"),
}


def _contains_any(text: str, signals: frozenset[str]) -> bool:
    return any(sig in text for sig in signals)


def detect_scope(query: str) -> str:
    """``"ng"`` / ``"us"`` / ``"both"`` / ``"auto"`` — explicit geography
    in the query pins the scope; otherwise ``"auto"`` (scope decided by
    relevance + fan-out)."""
    q = (query or "").lower()
    has_ng = _contains_any(q, NG_SIGNALS)
    has_us = _contains_any(q, US_SIGNALS)
    if has_ng and has_us:
        return "both"
    if has_ng:
        return "ng"
    if has_us:
        return "us"
    return "auto"


def scope_relevance(query: str) -> float:
    """0.0–1.0: how scope-sensitive is this query?

    Capped sum of the weights of the distinct term categories the query
    touches — category-based, not a single boolean switch.
    """
    q = (query or "").lower()
    if not q.strip():
        return 0.0
    score = 0.0
    for _name, (weight, terms) in SCOPE_CATEGORIES.items():
        if _contains_any(q, terms):
            score += weight
    return round(min(1.0, score), 3)


def regional_variants(
    query: str,
    *,
    scope: str | None = None,
    max_variants: int = 5,
) -> list[tuple[str, str]]:
    """The search plan: ``[(variant_query, region_label), ...]``.

    * scope-sensitive + ``"auto"`` → original + Nigerian + US variants.
    * explicit ``"ng"`` / ``"us"`` / ``"both"`` → original plus that
      region's flavour variants (both regions for ``"both"``).
    * ``"global"`` → the original query only (explicit opt-out).
    * not scope-sensitive → the original query only.

    ``region_label`` is ``"global"``, ``"ng"`` or ``"us"`` — consumers use
    it to guarantee both regions appear in the final result set.
    """
    query = (query or "").strip()
    if not query:
        return []
    pinned = (scope or "").strip().lower()
    if pinned not in {"", "auto", "ng", "us", "both", "global"}:
        pinned = "auto"
    if pinned == "":
        pinned = "auto"

    relevance = scope_relevance(query)
    detected = detect_scope(query) if pinned == "auto" else pinned

    variants: list[tuple[str, str]] = [(query, "global")]

    if detected == "global":
        return variants  # explicit opt-out of regional fan-out
    if pinned == "auto" and relevance < SCOPE_THRESHOLD:
        return variants  # a pure factoid: single-shot, no regional noise

    def _add_region_templates(region: str) -> None:
        for template in _REGION_TEMPLATES.get(region, ()):
            variant = template.format(q=query)
            if variant.lower() == query.lower():
                continue
            if any(variant.lower() == v[0].lower() for v in variants):
                continue
            variants.append((variant, region))

    if detected in {"auto", "both"}:
        # both regions must be well-served
        _add_region_templates("ng")
        _add_region_templates("us")
    elif detected == "ng":
        # query already names Nigeria: one reinforcement variant at most
        _add_region_templates("ng")
    elif detected == "us":
        _add_region_templates("us")
    return variants[: max(1, max_variants)]


def scope_label_for_variant(query: str, variant: str) -> str:
    """Recompute the region label for a variant (cheap helper for tests
    and ranking)."""
    v = (variant or "").lower()
    if "nigeria" in v or "lagos" in v:
        return "ng"
    if "united states" in v or re.search(r"\bus\b", v):
        return "us"
    return "global"
