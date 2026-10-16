"""Kid-safe content filtering: locale-aware, role-enforced, in code.

Rules, enforced by ``store.put_item`` and ``filter_for_role`` — never
by prompt:

* CHILD: any flagged term → content REFUSED (``allowed=False``).
  The child never sees, stores, or receives the flagged text.
* TEEN: flagged terms are MASKED (``***``) and the rest passes.
* ADULT / OWNER: content passes unchanged; flags are still reported
  so callers can log / warn.

Matching is word-boundary, case-insensitive, over the base term list
plus the caller's locale pack extras (``locales.blocked_terms_for``).
The filter is a guardrail for family content, not a moral judgement
on adults: adults keep full access; children get a filtered view.
"""

from __future__ import annotations

import re

from . import locales as locales_mod
from . import roles as roles_mod

__all__ = ["scan_text", "is_kid_safe", "filter_for_role", "mask_text"]


def _pattern_for(code: str | None) -> re.Pattern[str]:
    terms = locales_mod.blocked_terms_for(code)
    escaped = [re.escape(t) for t in terms if t]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def scan_text(text: str, locale: str | None = None) -> list[str]:
    """Terms from the locale pack found in ``text`` (lowercased, unique)."""
    if not text:
        return []
    pattern = _pattern_for(locale)
    found: list[str] = []
    for m in pattern.finditer(text):
        term = m.group(0).lower()
        if term not in found:
            found.append(term)
    return found


def is_kid_safe(text: str, locale: str | None = None) -> bool:
    """True iff ``text`` contains no flagged terms for ``locale``."""
    return not scan_text(text, locale)


def mask_text(text: str, locale: str | None = None) -> str:
    """Replace each flagged term with ``***``, preserving the rest."""
    if not text:
        return text
    return _pattern_for(locale).sub("***", text)


def filter_for_role(text: str, role: str, locale: str | None = None) -> dict[str, object]:
    """Apply the role's content rule. Returns a result dict, always.

    ``{"allowed": bool, "text": str, "flags": [...], "role": role}``
    * child + flags → allowed=False, text="" (refused, not masked —
      a child must not receive a fill-in-the-blank of the bad word).
    * teen + flags → allowed=True, text masked.
    * adult/owner → allowed=True, text unchanged, flags reported.
    """
    role = roles_mod.validate_role(role)
    flags = scan_text(text, locale)
    if role == roles_mod.CHILD and flags:
        return {"allowed": False, "text": "", "flags": flags, "role": role}
    if role == roles_mod.TEEN and flags:
        return {"allowed": True, "text": mask_text(text, locale),
                "flags": flags, "role": role}
    return {"allowed": True, "text": text, "flags": flags, "role": role}
