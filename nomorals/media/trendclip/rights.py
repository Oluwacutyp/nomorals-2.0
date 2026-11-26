"""Rights records for trendclip sources — ported from the source's
``trendclip/rights.py``.

The source's rule, kept verbatim in spirit: third-party media is a review
draft, never creator-owned, and anything without a rights record is blocked
from rendering. The owner approves every outward step; a clip of someone
else's video is a draft until he says otherwise.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RIGHTS_CATEGORIES", "validate_rights"]

#: The rights categories the source recognizes. ``unknown`` and
#: ``research_only`` are the blocked ones — a source with no usable rights
#: record never renders.
RIGHTS_CATEGORIES = (
    "third_party_review_draft",
    "user_authorized",
    "creator_owned",
    "licensed",
    "authorized_campaign",
    "unknown",
    "research_only",
)

#: Categories that may be rendered. Everything else is analysis-only.
RENDERABLE = frozenset((
    "third_party_review_draft",
    "user_authorized",
    "creator_owned",
    "licensed",
    "authorized_campaign",
))


def validate_rights(rights: dict[str, Any] | None,
                    *, source_url: str = "") -> dict[str, Any]:
    """Validate a rights record and say what may be done with the source.

    Returns ``{"ok": bool, "category": str, "label": str, "reasons": [...]}``.
    ``ok`` is False when the record is missing, the category is unknown, or
    the category is not renderable. The label is what the clip and the
    museum card carry — third-party media is always labeled as a review
    draft, never as creator-owned.
    """
    reasons: list[str] = []
    if not rights:
        return {
            "ok": False,
            "category": "unknown",
            "label": "rights unknown — render blocked",
            "reasons": ["no rights record supplied for "
                        + (source_url or "this source")],
        }
    category = str(rights.get("category") or "unknown")
    if category not in RIGHTS_CATEGORIES:
        reasons.append(f"unknown rights category {category!r}")
        category = "unknown"
    if category == "research_only":
        reasons.append("research-only source — analysis only, never render")
    if category == "unknown":
        reasons.append("rights unknown — render blocked")
    label = {
        "third_party_review_draft": "review draft — third-party media",
        "user_authorized": "user-authorized media",
        "creator_owned": "creator-owned media",
        "licensed": "licensed media",
        "authorized_campaign": "authorized campaign media",
        "unknown": "rights unknown — render blocked",
        "research_only": "research-only — render blocked",
    }[category]
    return {
        "ok": category in RENDERABLE,
        "category": category,
        "label": label,
        "reasons": reasons,
    }
