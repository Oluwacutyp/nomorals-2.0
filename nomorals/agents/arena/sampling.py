"""Adaptive challenge sampler for the arena.

Composes three signals on top of the topic sampler in ``topics.py``:

1. **Interest profile** (``activity.interest_profile``) — what the
   user actually does. This stays the base signal, never replaced.
2. **Coverage boost** (``scoring.coverage_weights``) — categories with
   few scored runs get multiplied up, so the arena explores instead
   of camping on one domain.
3. **Difficulty target** (``scoring.difficulty_target``) — categories
   the arena aces get grade-3 challenges, ones it struggles with get
   grade-1, everything else grade-2.

``sample_challenge`` returns ``(category, topic_text, full_entry)`` —
the full entry carries the challenge schema (``t``/``d``/``tags``/
``verify``/``kind``) so callers can render the acceptance criterion
and dispatch on kind. The last decision is persisted to
``kv_store`` under ``arena.sampling_state`` for observability.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from . import activity
from . import scoring
from . import topics

__all__ = [
    "sample_challenge",
    "sampling_state",
]

#: kv_store key holding the last sampling decision.
STATE_KEY = "arena.sampling_state"

_FALLBACK_KIND = "code"


def _coverage_boosts(db: Any) -> dict[str, float]:
    try:
        return scoring.coverage_weights(db)
    except Exception:  # noqa: BLE001
        return {}


def sample_challenge(db: Any = None, *, category: str | None = None,
                     rng: random.Random | None = None,
                     profile: dict[str, float] | None = None,
                     anti_repeat: int | None = None) -> tuple[str, str, dict]:
    """Pick ``(category, topic_text, full_entry)`` adaptively.

    * ``profile`` overrides the interest base; otherwise
      ``activity.interest_profile(db)``. Each base weight is
      multiplied by the category's coverage boost.
    * ``category`` forces one; otherwise the difficulty grade comes
      from the highest-weight category's ``difficulty_target``.
    * The full entry is looked up by topic text in
      ``topics.topics_in(cat)``; when the bank lookup misses, a
      synthetic entry with the challenge schema is returned.
    * The decision (ts/category/difficulty/weights) is persisted to
      ``kv_store`` as ``arena.sampling_state``. Never raises on
      db=None.
    """
    base = (dict(profile) if profile is not None
            else activity.interest_profile(db))
    boosts = _coverage_boosts(db)
    adjusted: dict[str, float] = {}
    for cat in topics.all_categories():
        try:
            adjusted[cat] = max(0.01, float(base.get(cat, 1.0))
                               * float(boosts.get(cat, 1.0)))
        except (TypeError, ValueError):
            adjusted[cat] = 1.0

    if category:
        difficulty = scoring.difficulty_target(db, category)
    else:
        top = (max(adjusted, key=lambda c: adjusted[c]) if adjusted
               else topics.CATEGORIES[0])
        difficulty = scoring.difficulty_target(db, top)

    cat, text = topics.sample_topic(
        db, category, rng=rng, profile=adjusted,
        difficulty=difficulty, anti_repeat=anti_repeat)

    entry = next((dict(e) for e in topics.topics_in(cat)
                  if topics.topic_text(e) == text), None)
    if entry is None:
        entry = {"t": text, "d": difficulty, "tags": (),
                 "verify": "", "kind": _FALLBACK_KIND}

    _persist_state(db, {"ts": time.time(), "category": cat,
                        "difficulty": difficulty, "weights": adjusted})
    return cat, text, entry


def _persist_state(db: Any, state: dict[str, Any]) -> bool:
    if db is None:
        return False
    try:
        with db.transaction():
            db.execute(
                """INSERT INTO kv_store (key, value, kind, updated_at)
                   VALUES (?, ?, 'json', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (STATE_KEY, json.dumps(state), time.time()),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def sampling_state(db: Any) -> dict[str, Any]:
    """The last persisted sampling decision, ``{}`` when absent/db=None."""
    if db is None:
        return {}
    try:
        row = db.query_one(
            "SELECT value FROM kv_store WHERE key = ?", (STATE_KEY,))
        if row:
            data = json.loads(row["value"])
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}
