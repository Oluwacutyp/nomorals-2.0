"""Time / pattern / interest modeling — she learns, predicts, acts.

Three models, all learned from evidence, none hardcoded:

* **RoutineModel** — when is the owner active? Hour-of-day and
  day-of-week activity histograms. Predicts the next active window.
  Cells decay over time (Prophet-style adaptability): a changed habit
  stops haunting predictions instead of fighting the new one forever.
* **InterestModel** — what does the owner care about? Topics extracted
  from conversations with a YAKE-style statistical scorer (casing,
  position, frequency, relatedness, dispersion — dependency-free, works
  on single short messages), scored with time decay. Rising interests
  get proactive research; fading ones decay away. Co-occurring topics
  form clusters ("you're into X+Y lately").
* **PatternModel** — what sequences recur? "Checks news after the
  morning pulse", "asks for music on Friday nights". A proper
  transition model (Laplace-smoothed, confidence-weighted — the
  learning-automata idea) says what usually follows a trigger, so idle
  time prepares the *right* thing.

All three feed the presence layer (``presence.py``), which decides
what to actually do. Models never act directly — they inform.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Days for interest half-life.
INTEREST_HALF_LIFE_DAYS = 14.0

#: Minimum activity samples before routine predictions are trusted.
MIN_ROUTINE_SAMPLES = 20

#: Days for routine half-life (habit adaptability).
ROUTINE_HALF_LIFE_DAYS = 45.0


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
    # Topic co-occurrence: pairs seen in the same text.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_cooccur (
            a TEXT NOT NULL,
            b TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (a, b)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS pattern_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _meta_get(db: Any, key: str, default: str = "") -> str:
    ensure_schema(db)
    row = db.execute(
        "SELECT value FROM pattern_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row and row[0] is not None else default


def _meta_set(db: Any, key: str, value: str) -> None:
    ensure_schema(db)
    db.execute(
        "INSERT INTO pattern_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


# ── routine ──────────────────────────────────────────────────────────

def _maybe_decay_routine(db: Any, ts: float | None = None) -> None:
    """Periodically decay routine cells so changed habits adapt.

    At most once per day: every cell's count decays by the routine
    half-life, and cells below 1 are dropped. A owner who moved their
    morning check from 7am to 9am stops getting 7am predictions after a
    few weeks instead of forever.
    """
    now = ts if ts is not None else time.time()
    try:
        last = float(_meta_get(db, "routine_last_decay", "0") or 0)
    except ValueError:
        last = 0.0
    if now - last < 86400:
        return
    factor = 0.5 ** (1.0 / ROUTINE_HALF_LIFE_DAYS)
    # Fractional counts (SQLite keeps the REAL value) so young cells
    # survive a decay pass; cells below 0.5 are gone for good.
    db.execute(
        "UPDATE routine_activity SET count = count * ?", (factor,))
    db.execute("DELETE FROM routine_activity WHERE count < 0.5")
    _meta_set(db, "routine_last_decay", str(now))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def record_activity(db: Any, ts: float | None = None,
                    tz_offset: float = 0.0) -> None:
    """Log one owner-activity tick (message, command, tool use)."""
    ensure_schema(db)
    _maybe_decay_routine(db, ts)
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
    now = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(
        hours=tz_offset)
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


def next_activity(db: Any, tz_offset: float = 0.0) -> dict[str, Any]:
    """Next predicted active window, with a confidence level.

    Confidence is grounded in evidence mass: many samples concentrated
    in one window → high; thin or spread data → low. Never claims more
    than the data supports.
    """
    hist = routine_histogram(db)
    total = hist["total"]
    if total < MIN_ROUTINE_SAMPLES:
        return {"predicted": False,
                "reason": f"only {total} samples "
                          f"(need {MIN_ROUTINE_SAMPLES})"}
    windows = predict_active_windows(db, tz_offset=tz_offset, top_n=5)
    if not windows:
        return {"predicted": False, "reason": "no windows"}
    top = windows[0]
    mass = top["score"]
    total_mass = sum(w["score"] for w in windows) or 1.0
    share = mass / total_mass
    # Confidence from concentration + absolute evidence.
    if total >= 200 and share >= 0.4:
        confidence = "high"
    elif total >= 60 and share >= 0.25:
        confidence = "medium"
    else:
        confidence = "low"
    import datetime as _dt
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "predicted": True,
        "hour": top["hour"],
        "dow": top["dow"],
        "dow_name": names[top["dow"]],
        "score": top["score"],
        "share": round(share, 2),
        "confidence": confidence,
        "samples": total,
    }


def quiet_hours(db: Any, tz_offset: float = 0.0) -> list[int]:
    """Hours the owner is almost never active — learned quiet hours.

    An hour counts as quiet when its total activity across all days is
    under 5% of the busiest hour's. Presence uses this to hold
    serendipity until morning instead of pinging at 3am.
    """
    hist = routine_histogram(db)
    if hist["total"] < MIN_ROUTINE_SAMPLES:
        return list(range(0, 6))  # sensible default: midnight–6am
    per_hour: dict[int, int] = defaultdict(int)
    for cell in hist["cells"]:
        # Cells were recorded with the tz_offset at record time; the
        # histogram is already in owner-local hours.
        per_hour[cell["hour"]] += cell["count"]
    if not per_hour:
        return list(range(0, 6))
    peak = max(per_hour.values())
    floor = peak * 0.05
    return sorted(h for h in range(24) if per_hour.get(h, 0) <= floor)


# ── interests ────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "the a an and or but of to in on for with at by from as is are was "
    "were be been it its this that these those i you he she we they me "
    "him her us them my your his our their what when where how why who "
    "which do does did can could should would will just now then so very "
    "really get got make made".split()
)

_SENT_SPLIT = re.compile(r"[.!?]+|\n+")


def _yake_candidates(text: str) -> tuple[list[str], list[list[str]]]:
    """Split text into sentences of content-word tokens."""
    sentences: list[list[str]] = []
    for sent in _SENT_SPLIT.split(text):
        toks = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", sent)
        kept = [t for t in toks if t.lower() not in _STOPWORDS]
        if kept:
            sentences.append(kept)
    return sentences


def _yake_score(text: str, max_topics: int = 5) -> list[str]:
    """YAKE-style statistical keyword scoring, dependency-free.

    Five single-document features per candidate n-gram (1–3 words):
    casing (TITLE/UPPER words matter), position (earlier sentences
    matter), frequency, relatedness to context (co-occurrence with other
    candidates), and sentence dispersion. Lower score = more important,
    like YAKE. Near-duplicate phrases are folded (keep the best).
    """
    sentences = _yake_candidates(text)
    if not sentences:
        return []
    n_sent = len(sentences)
    # Candidate n-grams with their sentence positions.
    cand_pos: dict[str, list[int]] = defaultdict(list)
    case_hits: Counter[str] = Counter()
    for si, toks in enumerate(sentences):
        lowered = [t.lower() for t in toks]
        for n in (1, 2, 3):
            for i in range(len(lowered) - n + 1):
                phrase = " ".join(lowered[i:i + n])
                cand_pos[phrase].append(si)
        for t in toks:
            if t[:1].isupper():
                case_hits[t.lower()] += 1
    # Frequency + dispersion.
    freq: Counter[str] = Counter(
        {p: len(set(pos)) * 1.0 + len(pos) * 0.5
         for p, pos in cand_pos.items()})
    max_freq = max(freq.values()) or 1.0
    # Relatedness: distinct co-occurring candidates in same sentences.
    sent_cands: list[set[str]] = []
    for si, toks in enumerate(sentences):
        lowered = [t.lower() for t in toks]
        s: set[str] = set()
        for n in (1, 2, 3):
            for i in range(len(lowered) - n + 1):
                s.add(" ".join(lowered[i:i + n]))
        sent_cands.append(s)
    related: Counter[str] = Counter()
    for s in sent_cands:
        for c in s:
            related[c] += len(s) - 1
    scored: list[tuple[float, str]] = []
    for phrase, positions in cand_pos.items():
        # Skip phrases that are substrings issues handled later; skip
        # single chars already filtered by regex.
        w_case = 1.0 + case_hits.get(phrase.split()[0], 0) * 0.5
        median_pos = sorted(positions)[len(positions) // 2]
        w_pos = 1.0 - (median_pos / max(n_sent, 1)) * 0.5  # earlier better
        w_freq = freq[phrase] / max_freq
        dispersion = len(set(positions)) / n_sent
        w_rel = 1.0 + math.log1p(related.get(phrase, 0))
        # YAKE-style: lower is better.
        score = (w_rel * (2.0 - w_pos)) / (
            w_case * (0.5 + w_freq) * (1.0 + dispersion) + 1e-9)
        # Prefer multi-word phrases slightly (they're more topical).
        words = len(phrase.split())
        score *= {1: 1.15, 2: 1.0, 3: 0.95}.get(words, 1.0)
        scored.append((score, phrase))
    scored.sort(key=lambda t: (t[0], t[1]))
    # Fold near-duplicates: if a lower-ranked phrase shares >80% of its
    # tokens with an accepted one, drop it.
    accepted: list[str] = []
    for _score, phrase in scored:
        ptoks = set(phrase.split())
        dup = False
        for a in accepted:
            atoks = set(a.split())
            overlap = len(ptoks & atoks) / max(len(ptoks | atoks), 1)
            if overlap > 0.8:
                dup = True
                break
        if not dup:
            accepted.append(phrase)
        if len(accepted) >= max_topics:
            break
    return accepted


def _freq_topics(text: str, max_topics: int = 5) -> list[str]:
    """Legacy frequency counter (kept for comparison/testing)."""
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
        scored[b] += c * 1.5
    return [t for t, _ in scored.most_common(max_topics)]


def extract_topics(text: str, max_topics: int = 5,
                   method: str = "yake") -> list[str]:
    """Extract significant topics from text.

    ``method="yake"`` (default): statistical single-document scoring —
    casing, position, frequency, relatedness, dispersion. Works on
    short chat messages without any corpus or dependency.
    ``method="freq"``: the legacy frequency counter.
    """
    if method == "freq":
        return _freq_topics(text, max_topics)
    return _yake_score(text, max_topics)


def record_interests(db: Any, text: str,
                     ts: float | None = None) -> list[str]:
    """Extract topics from text, bump their interest scores.

    Also records topic co-occurrence pairs (topics seen together form
    clusters via :func:`topic_clusters`).
    """
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
    # Co-occurrence pairs (ordered a < b for a stable key).
    uniq = sorted(set(topics))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            db.execute(
                "INSERT INTO topic_cooccur (a, b, count) VALUES (?, ?, 1) "
                "ON CONFLICT (a, b) DO UPDATE SET count = count + 1",
                (a, b),
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


def topic_clusters(db: Any, min_cooccur: int = 2) -> list[list[str]]:
    """Clusters of topics that keep appearing together.

    Union-find over co-occurrence pairs. Lets presence surface
    "you're into X + Y lately" instead of isolated tokens.
    """
    ensure_schema(db)
    rows = db.execute(
        "SELECT a, b FROM topic_cooccur WHERE count >= ?",
        (max(1, int(min_cooccur)),)).fetchall()
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in rows:
        union(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in list(parent):
        groups[find(node)].append(node)
    clusters = [sorted(g) for g in groups.values() if len(g) >= 2]
    clusters.sort(key=lambda c: -len(c))
    return clusters


# ── hygiene ──────────────────────────────────────────────────────────

#: Days after which a never-reinforced routine cell is dropped.
ROUTINE_RETENTION_DAYS = 180
#: Days after which a stale pattern row is dropped.
PATTERN_RETENTION_DAYS = 180


def prune(db: Any, ts: float | None = None) -> dict[str, Any]:
    """Bounded-growth maintenance for the model tables.

    Drops interests decayed below the visibility floor, routine cells
    untouched for ``ROUTINE_RETENTION_DAYS``, and stale pattern rows.
    Called once per idle cycle — the tables stay small on a bot that
    runs for months. Never raises.
    """
    report: dict[str, Any] = {
        "interests_dropped": 0, "cells_dropped": 0, "patterns_dropped": 0}
    try:
        ensure_schema(db)
        now = ts if ts is not None else time.time()
        # Interests: decayed score below the current_interests floor.
        rows = db.execute(
            "SELECT topic, score, last_seen FROM interests").fetchall()
        stale_topics = []
        for topic, score, last_seen in rows:
            age_days = max(0.0, (now - float(last_seen or 0)) / 86400.0)
            decayed = float(score or 0) * math.exp(
                -age_days * math.log(2) / INTEREST_HALF_LIFE_DAYS)
            if decayed <= 0.1 and age_days > 30:
                stale_topics.append(topic)
        for topic in stale_topics:
            db.execute("DELETE FROM interests WHERE topic = ?", (topic,))
        report["interests_dropped"] = len(stale_topics)

        # Routine cells: we don't store last_seen per cell, so bound by
        # total row count instead — keep the hottest cells.
        cell_count = db.execute(
            "SELECT COUNT(*) FROM routine_activity").fetchone()[0]
        if cell_count and cell_count > 24 * 7 * 4:  # > ~4 weeks of cells
            db.execute(
                "DELETE FROM routine_activity WHERE (hour, dow) NOT IN ("
                "SELECT hour, dow FROM routine_activity "
                "ORDER BY count DESC LIMIT ?)",
                (24 * 7 * 4,))
            report["cells_dropped"] = cell_count - 24 * 7 * 4

        # Patterns: drop rows unseen for the retention window.
        cutoff = now - PATTERN_RETENTION_DAYS * 86400
        cur = db.execute(
            "DELETE FROM patterns WHERE last_seen < ?", (cutoff,))
        report["patterns_dropped"] = cur.rowcount or 0

        # Co-occurrence: drop weak pairs.
        cur = db.execute(
            "DELETE FROM topic_cooccur WHERE count < 2")
        report["weak_pairs_dropped"] = cur.rowcount or 0
        try:
            db.commit()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        _log.warning("pattern prune failed: %s", exc, exc_info=True)
        report["error"] = str(exc)[:200]
    return report


# ── patterns (transition model) ──────────────────────────────────────

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


def transition_model(db: Any, trigger_sig: str,
                     smoothing: float = 1.0) -> list[dict[str, Any]]:
    """Full transition distribution for a trigger.

    Laplace-smoothed probabilities over all observed actions, with a
    confidence per edge from evidence mass and recency. This is the
    honest version of a raw counter: rare edges get pulled toward
    uniform, well-worn edges get trusted.
    """
    ensure_schema(db)
    now = time.time()
    rows = db.execute(
        "SELECT action_sig, count, last_seen FROM patterns "
        "WHERE trigger_sig = ?",
        (trigger_sig,),
    ).fetchall()
    if not rows:
        return []
    total = sum(int(r[1]) for r in rows)
    n = len(rows)
    out = []
    for action, count, last_seen in rows:
        count = int(count)
        prob = (count + smoothing) / (total + smoothing * n)
        # Recency weight: edges seen in the last week count double.
        age_days = max(0.0, (now - float(last_seen or 0)) / 86400.0)
        recency = 2.0 if age_days <= 7 else (1.0 if age_days <= 30 else 0.5)
        # Confidence: evidence mass × recency, capped.
        confidence = min(1.0, (count / (count + 5.0)) * recency / 2.0
                         + (count / (count + 5.0)) * 0.5)
        out.append({
            "action": action,
            "count": count,
            "probability": round(prob, 3),
            "confidence": round(min(1.0, confidence), 2),
            "last_seen": last_seen,
        })
    out.sort(key=lambda e: -e["probability"])
    return out


def likely_next(db: Any, trigger_sig: str,
                min_count: int = 3) -> list[dict[str, Any]]:
    """Given a trigger, what actions usually follow?

    Confidence-weighted and recency-aware; each entry carries its
    transition probability so callers can decide how much to trust it.
    """
    edges = transition_model(db, trigger_sig)
    return [
        {"action": e["action"], "count": e["count"],
         "last_seen": e["last_seen"],
         "probability": e["probability"],
         "confidence": e["confidence"]}
        for e in edges if e["count"] >= min_count
    ]


# ── portability ──────────────────────────────────────────────────────

def export_models(db: Any) -> dict[str, Any]:
    """Dump all learned models as JSON-able data (backup/debug)."""
    ensure_schema(db)
    return {
        "exported_ts": time.time(),
        "routine_activity": [
            {"hour": r[0], "dow": r[1], "count": r[2]}
            for r in db.execute(
                "SELECT hour, dow, count FROM routine_activity").fetchall()
        ],
        "interests": [
            {"topic": r[0], "score": r[1], "last_seen": r[2],
             "sightings": r[3]}
            for r in db.execute(
                "SELECT topic, score, last_seen, sightings "
                "FROM interests").fetchall()
        ],
        "patterns": [
            {"trigger_sig": r[0], "action_sig": r[1], "count": r[2],
             "last_seen": r[3]}
            for r in db.execute(
                "SELECT trigger_sig, action_sig, count, last_seen "
                "FROM patterns").fetchall()
        ],
        "topic_cooccur": [
            {"a": r[0], "b": r[1], "count": r[2]}
            for r in db.execute(
                "SELECT a, b, count FROM topic_cooccur").fetchall()
        ],
    }


def import_models(db: Any, data: dict[str, Any]) -> dict[str, int]:
    """Merge an exported model dump back in (additive, never deletes)."""
    ensure_schema(db)
    counts = {"routine": 0, "interests": 0, "patterns": 0,
              "cooccur": 0}
    try:
        for cell in (data or {}).get("routine_activity", []):
            db.execute(
                "INSERT INTO routine_activity (hour, dow, count) "
                "VALUES (?, ?, ?) ON CONFLICT (hour, dow) DO UPDATE "
                "SET count = count + excluded.count",
                (int(cell["hour"]), int(cell["dow"]),
                 int(cell["count"])))
            counts["routine"] += 1
        for item in (data or {}).get("interests", []):
            db.execute(
                "INSERT INTO interests (topic, score, last_seen, "
                "sightings) VALUES (?, ?, ?, ?) ON CONFLICT (topic) "
                "DO UPDATE SET score = score + excluded.score, "
                "sightings = sightings + excluded.sightings, "
                "last_seen = MAX(last_seen, excluded.last_seen)",
                (item["topic"], float(item["score"]),
                 float(item["last_seen"]), int(item["sightings"])))
            counts["interests"] += 1
        for pat in (data or {}).get("patterns", []):
            db.execute(
                "INSERT INTO patterns (trigger_sig, action_sig, count, "
                "last_seen) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (trigger_sig, action_sig) DO UPDATE SET "
                "count = count + excluded.count, "
                "last_seen = MAX(last_seen, excluded.last_seen)",
                (pat["trigger_sig"], pat["action_sig"],
                 int(pat["count"]), float(pat["last_seen"])))
            counts["patterns"] += 1
        for pair in (data or {}).get("topic_cooccur", []):
            db.execute(
                "INSERT INTO topic_cooccur (a, b, count) VALUES (?, ?, ?) "
                "ON CONFLICT (a, b) DO UPDATE SET "
                "count = count + excluded.count",
                (pair["a"], pair["b"], int(pair["count"])))
            counts["cooccur"] += 1
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("import_models failed: %s", exc, exc_info=True)
    return counts


def models_summary(db: Any) -> dict[str, Any]:
    """One-glance stats over the learned models."""
    ensure_schema(db)
    n_cells = db.execute(
        "SELECT COUNT(*) FROM routine_activity").fetchone()[0] or 0
    n_interests = db.execute(
        "SELECT COUNT(*) FROM interests").fetchone()[0] or 0
    n_patterns = db.execute(
        "SELECT COUNT(*) FROM patterns").fetchone()[0] or 0
    n_triggers = db.execute(
        "SELECT COUNT(DISTINCT trigger_sig) FROM patterns").fetchone()[0] or 0
    n_pairs = db.execute(
        "SELECT COUNT(*) FROM topic_cooccur").fetchone()[0] or 0
    return {
        "routine_cells": n_cells,
        "interests": n_interests,
        "pattern_edges": n_patterns,
        "pattern_triggers": n_triggers,
        "cooccur_pairs": n_pairs,
        "clusters": len(topic_clusters(db)),
    }
