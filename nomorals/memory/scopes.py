"""Memory scopes: per-project / per-person / per-goal memory spaces.

One memory store, many rooms. A scope is a named space — ``project:devon-arena``,
``person:ada``, ``goal:earn-500`` — that a record can live in. Global records
(no scope) are visible everywhere; scoped records are visible only inside
their own scope.

The mechanism is deliberately boring: a scope is a reserved tag
(``scope:<name>``) written at remember time and filtered at recall time.
No schema migration, no second database — and the anti-leak rule is one
check in the recall loop: a scoped record is returned only when the
recall asks for that exact scope. Forgetting the rule is impossible
because there is no other path.

Conventions (not enforcement — scopes are user-defined strings):
- ``project:<slug>`` — one project's working memory
- ``person:<name>`` — memories about one person
- ``goal:<slug>`` — memories serving one goal
"""

from __future__ import annotations

import re
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "SCOPE_PREFIX",
    "normalize_scope",
    "scope_tag",
    "scope_of",
    "record_matches_scope",
    "scopes_summary",
]

#: tag prefix that marks a scope tag.  Reserved: user tags may not start
#: with it (``join_tags`` callers should strip them; the manager does).
SCOPE_PREFIX = "scope:"

_SCOPE_CLEAN = re.compile(r"[^a-z0-9_:\-]+")


def normalize_scope(name: str) -> str:
    """Canonicalize a scope name: lowercase, spaces → ``-``, junk dropped.

    ``"Devon Arena"`` → ``"devon-arena"``; ``"project:Devon Arena"`` →
    ``"project:devon-arena"``.  Empty input → ``""`` (global).
    """
    cleaned = _SCOPE_CLEAN.sub("-", (name or "").strip().lower().replace(" ", "-"))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-:")
    return cleaned


def scope_tag(name: str) -> str:
    """The tag a scope is stored as. ``""`` when the name is empty."""
    normalized = normalize_scope(name)
    return f"{SCOPE_PREFIX}{normalized}" if normalized else ""


def scope_of(record: Any) -> str:
    """The scope a record belongs to, or ``""`` when it is global.

    Reads the record's tags; the first ``scope:`` tag wins.  Never raises.
    """
    try:
        tags = (getattr(record, "tags", "") or "")
        for part in tags.split(","):
            tag = part.strip().lower()
            if tag.startswith(SCOPE_PREFIX) and len(tag) > len(SCOPE_PREFIX):
                return tag[len(SCOPE_PREFIX):]
    except Exception:  # noqa: BLE001
        pass
    return ""


def record_matches_scope(record: Any, scope: str) -> bool:
    """The anti-leak check used by recall.

    - no scope requested → everything matches (historical behaviour);
    - scope requested → the record matches when it is global (no scope
      tag) or carries exactly that scope.  A record scoped to a *different*
      space never leaks in.
    """
    wanted = normalize_scope(scope)
    if not wanted:
        return True
    mine = scope_of(record)
    return not mine or mine == wanted


def scopes_summary(manager: Any) -> dict[str, int]:
    """``{scope: record_count}`` over the whole store, plus ``"<global>"``.

    Never raises — a broken query yields ``{}`` rather than breaking chat.
    """
    try:
        result = manager.recall("", limit=10000, include_private=True)
    except Exception as exc:  # noqa: BLE001
        _log.debug("scopes summary failed: %s", exc)
        return {}
    counts: dict[str, int] = {}
    for record in result.records:
        key = scope_of(record) or "<global>"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
