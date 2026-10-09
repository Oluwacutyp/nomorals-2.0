"""Time / pattern / interest modeling — she learns, predicts, acts.

Three models, all learned from evidence, none hardcoded:

* **RoutineModel** — when is the owner active? Hour-of-day and
  day-of-week activity histograms. Predicts the next active window.
* **InterestModel** — what does the owner care about? Topics extracted
  from conversations, scored with time decay. Rising interests get
  proactive research; fading ones decay away.
* **PatternModel** — what sequences recur? "Checks news after the
  morning pulse", "asks for music on Friday nights". When a trigger
  pattern fires, she prepares before being asked.

All three feed the presence layer (``presence.py``), which decides
what to actually do. Models never act directly — they inform.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Days for interest half-life.
INTEREST_HALF_LIFE_DAYS = 14.0

#: Minimum activity samples before routine predictions are trusted.
MIN_ROUTINE_SAMPLES = 20


def _hour_key(ts: float, tz_offset: float = 0.0) -> int:
    return int((ts + tz_offset * 3600) // 3600) % 24


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS routine_activity (
            hour INTEGER NOT NULL,
            dow INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (hour, dow)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS interests (
            topic TEXT PRIMARY KEY,
            score REAL NOT NULL DEFAULT 0,
            last_seen REAL NOT NULL DEFAULT 0,
            sightings INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS patterns (
            trigger_sig TEXT NOT NULL,
            action_sig TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            last_seen REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (trigger_sig, action_sig)
        )
        """
    )


# ── routine ──────────────────────────────────────────────────────────

def record_activity(db: Any, ts: float | None = None,
                    tz_offset: float = 0.0) -> None:
    """Log one owner-activity tick (message, command, tool use)."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    import datetime as _dt
    dt = _dt.datetime.fromtimestamp(now + tz_offset * 3600,
                                    tz=_dt.timezone.utc)
    db.execute(
        "INSERT INTO routine_activity (hour, dow, count) VALUES (?, ?, 1) "
        "ON CONFLICT (hour, dow) DO UPDATE SET count = count + 1",
        (dt.hour, dt.weekday()),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def routine_histogram(db: Any) -> dict[str, Any]:
    """Hour × day-of-week activity counts."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT hour, dow, count FROM routine_activity").fetchall()
    total = sum(r[2] for r in rows)
    return {
        "total": total,
        "cells": [{"hour": r[0], "dow": r[1], "count": r[2]} for r in rows],
    }


def predict_active_windows(db: Any, tz_offset: float = 0.0,
                           top_n: int = 3) -> list[dict[str, Any]]:
    """Predict when the owner is next likely active.

    Returns top-N (hour, dow, score) windows. Empty until enough data.
    """
    hist = routine_histogram(db)
    if hist["total"] < MIN_ROUTINE_SAMPLES:
        return []
    import datetime as _dt
    now = _dt.datetime.utcnow() + _dt.timedelta(hours=tz_offset)
    scored = []
    for cell in hist["cells"]:
        # Weight by recency of the weekday (upcoming days score higher).
        dow_dist = (cell["dow"] - now.weekday()) % 7
        recency = 1.0 / (1.0 + dow_dist * 0.3)
        scored.append({
            "hour": cell["hour"], "dow": cell["dow"],
            "score": round(cell["count"] * recency, 2),
        })
    scored.sort(key=lambda c: -c["score"])
    return scored[:top_n]


# ── interests ────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "the a an and or but of to in on for with at by from as is are was "
    "were be been it its this that these those i you he she we they me "
    "him her us them my your his our their what when where how why who "
    "which do does did can could should would will just now then so very "
    "really get got make made".split()
)


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Crude topic extraction: significant non-stopword tokens/bigrams."""
    words = re.findall(r"[a-z][a-z\-']{2,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS]
    if not words:
        return []
    unigrams = Counter(words)
    bigrams = Counter(
        f"{a} {b}" for a, b in zip(words, words[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS
    )
    scored = Counter()
    for w, c in unigrams.items():
        scored[w] += c
    for b, c in bigrams.items():
        scored[b] += c * 1.5  # bigrams are more topical
    return [t for t, _ in scored.most_common(max_topics)]


def record_interests(db: Any, text: str,
                     ts: float | None = None) -> list[str]:
    """Extract topics from text, bump their interest scores."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    topics = extract_topics(text)
    for topic in topics:
        db.execute(
            "INSERT INTO interests (topic, score, last_seen, sightings) "
            "VALUES (?, 1.0, ?, 1) "
            "ON CONFLICT (topic) DO UPDATE SET "
            "score = score + 1.0, last_seen = excluded.last_seen, "
            "sightings = sightings + 1",
            (topic, now),
        )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return topics


def current_interests(db: Any, limit: int = 10,
                      ts: float | None = None) -> list[dict[str, Any]]:
    """Top interests with exponential time decay applied."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    rows = db.execute(
        "SELECT topic, score, last_seen, sightings FROM interests").fetchall()
    scored = []
    for topic, score, last_seen, sightings in rows:
        age_days = max(0.0, (now - last_seen) / 86400.0)
        decayed = score * math.exp(
            -age_days * math.log(2) / INTEREST_HALF_LIFE_DAYS)
        if decayed > 0.1:
            scored.append({
                "topic": topic,
                "score": round(decayed, 2),
                "sightings": sightings,
                "age_days": round(age_days, 1),
            })
    scored.sort(key=lambda i: -i["score"])
    return scored[:limit]


def rising_interests(db: Any, limit: int = 5) -> list[dict[str, Any]]:
    """Interests gaining momentum — candidates for proactive research."""
    all_now = current_interests(db, limit=50)
    # Rising = high sightings in a short span (young + active).
    rising = [i for i in all_now
              if i["sightings"] >= 3 and i["age_days"] <= 7.0]
    rising.sort(key=lambda i: (-i["sightings"], -i["score"]))
    return rising[:limit]


# ── patterns ─────────────────────────────────────────────────────────

def record_sequence(db: Any, trigger_sig: str, action_sig: str,
                    ts: float | None = None) -> None:
    """Log a trigger→action sequence (e.g. 'morning-pulse' → 'news-ask')."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    db.execute(
        "INSERT INTO patterns (trigger_sig, action_sig, count, last_seen) "
        "VALUES (?, ?, 1, ?) "
        "ON CONFLICT (trigger_sig, action_sig) DO UPDATE SET "
        "count = count + 1, last_seen = excluded.last_seen",
        (trigger_sig, action_sig, now),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def likely_next(db: Any, trigger_sig: str,
                min_count: int = 3) -> list[dict[str, Any]]:
    """Given a trigger, what actions usually follow?"""
    ensure_schema(db)
    rows = db.execute(
        "SELECT action_sig, count, last_seen FROM patterns "
        "WHERE trigger_sig = ? AND count >= ? "
        "ORDER BY count DESC",
        (trigger_sig, min_count),
    ).fetchall()
    return [{"action": r[0], "count": r[1], "last_seen": r[2]} for r in rows]
