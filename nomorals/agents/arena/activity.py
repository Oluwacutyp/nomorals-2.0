"""What the user does most → arena topic weights.

A lightweight activity log plus an interest profiler. Signals:

* **arena digest history** — categories the arena already researched
  (the owner kept the loop running on them).
* **custom topics** — owner-added topics, keyword-matched to categories.
* **recent searches** — ``SearchEngine.history()`` queries, keyword-matched.
* **goals** — goal titles, keyword-matched.
* **command usage** — ``record("command", "<verb>")`` calls from the
  control dispatcher; arena/game-adjacent verbs count double.

``interest_profile()`` returns ``{category: weight}`` — base 1.0,
boosted by the signals above. The topic sampler in ``topics.py``
consumes it. Everything degrades gracefully when a signal is missing.
"""

from __future__ import annotations

import re
import time
from typing import Any

__all__ = [
    "CATEGORY_KEYWORDS",
    "interest_profile",
    "match_category",
    "record",
    "recent_commands",
]

# Keyword → category hints. Kept broad on purpose: a search for
# "postgres indexes" should nudge the data category, "rust borrow
# checker" the languages one.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "web": ("http", "cdn", "browser", "frontend", "websocket", "website",
            "web ", "css", "javascript", "react", "ssr", "edge"),
    "security": ("security", "hack", "exploit", "malware", "phish",
                 "vulnerab", "mfa", "pentest", "rootkit", "sandbox"),
    "networking": ("network", "bgp", "dns", "tcp", "quic", "routing",
                   "wifi", "5g", "latency", "packet"),
    "data": ("database", "postgres", "sql", "kafka", "vector db",
             "data ", "etl", "parquet", "redis", "index"),
    "ai": ("ai", "llm", "gpt", "model", "training", "inference",
           "diffusion", "rag", "agent", "token", "prompt"),
    "automation": ("ci", "cd", "pipeline", "deploy", "terraform",
                   "kubernetes", "k8s", "cron", "sre", "devops"),
    "systems": ("linux", "kernel", "container", "docker", "process",
                "memory", "concurrency", "unix"),
    "history": ("history", "retro", "1980", "1990", "mainframe",
                "usenet", "vintage"),
    "crypto": ("crypto", "bitcoin", "blockchain", "ethereum", "defi",
               "wallet", "zk", "zero knowledge", "smart contract", "web3"),
    "mobile": ("mobile", "ios", "android", "iphone", "app ", "flutter",
               "push notification"),
    "devtools": ("ide", "vscode", "debug", "lsp", "build", "cli",
                 "git", "package", "compiler"),
    "os": ("windows", "macos", "boot", "filesystem", "kernel",
           "operating system"),
    "hardware": ("cpu", "gpu", "ram", "ssd", "chip", "raspberry",
                 "server", "datacenter", "nvidia"),
    "languages": ("rust", "python", "go ", "golang", "typescript",
                  "javascript", "java ", "c++", "zig", "erlang",
                  "programming language"),
}

# Control verbs that smell like a category interest (verb → category).
VERB_CATEGORY: dict[str, str] = {
    "search": "web", "investigate": "security", "monitor": "automation",
    "structure": "data", "decode": "security", "cipher": "crypto",
    "music": "mobile", "play": "mobile", "arena": "ai", "game": "ai",
    "briefing": "automation", "watch": "automation", "room": "devtools",
    "code": "devtools", "vision": "ai", "media": "mobile",
}


def _ensure_table(db: Any) -> bool:
    if db is None:
        return False
    try:
        with db.transaction():
            db.execute(
                """CREATE TABLE IF NOT EXISTS arena_activity
                   (ts REAL, kind TEXT, detail TEXT)""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_arena_activity_ts "
                "ON arena_activity (ts)")
        return True
    except Exception:  # noqa: BLE001
        return False


def record(db: Any, kind: str, detail: str = "") -> None:
    """Log one activity event. Never raises, never blocks the caller."""
    if not _ensure_table(db):
        return
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO arena_activity (ts, kind, detail) VALUES (?, ?, ?)",
                (time.time(), str(kind)[:40], str(detail)[:300]),
            )
    except Exception:  # noqa: BLE001
        pass


def recent_commands(db: Any, limit: int = 200) -> list[str]:
    """Most recent recorded control verbs, newest first."""
    if db is None:
        return []
    try:
        rows = db.query(
            "SELECT detail FROM arena_activity WHERE kind = 'command' "
            "ORDER BY ts DESC LIMIT ?", (max(1, min(int(limit), 2000)),))
        return [str(r.get("detail", "")) for r in rows]
    except Exception:  # noqa: BLE001
        return []


def match_category(text: str) -> str | None:
    """Best-guess category for free text, or None.

    Keywords match on word boundaries, so "ram" doesn't fire on
    "programming" and "go" doesn't fire on "golang".
    """
    lowered = text.lower()
    best: str | None = None
    best_hits = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            kw = kw.strip().lower()
            if not kw:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])",
                         lowered):
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best = cat
    return best


def _bump(scores: dict[str, float], category: str | None, amount: float) -> None:
    if category:
        scores[category] = scores.get(category, 0.0) + amount


def interest_profile(db: Any = None,
                     search_queries: list[str] | None = None,
                     goal_texts: list[str] | None = None) -> dict[str, float]:
    """Build ``{category: weight}`` from every available signal.

    Base weight is 1.0 everywhere; signals add on top (capped per
    signal so no single source dominates). Missing signals are
    skipped silently.
    """
    from .topics import CATEGORIES

    scores: dict[str, float] = {c: 1.0 for c in CATEGORIES}

    # 1. arena digest history — what the loop already researched.
    if db is not None:
        try:
            rows = db.query(
                "SELECT category, COUNT(*) AS n FROM arena_knowledge "
                "GROUP BY category")
            for row in rows:
                cat = str(row.get("category", ""))
                if cat in scores:
                    scores[cat] += min(float(row.get("n", 0)) * 0.4, 3.0)
        except Exception:  # noqa: BLE001
            pass
        # 2. custom topics — explicit owner interest.
        try:
            row = db.query_one(
                "SELECT value FROM kv_store WHERE key = 'arena.custom_topics'")
            if row:
                import json

                topics = json.loads(row["value"]).get("topics", [])
                for t in topics[:50]:
                    _bump(scores, match_category(str(t)), 1.0)
        except Exception:  # noqa: BLE001
            pass
        # 3. command usage — what the owner actually runs.
        try:
            for verb in recent_commands(db, 300):
                _bump(scores, VERB_CATEGORY.get(verb), 0.15)
        except Exception:  # noqa: BLE001
            pass

    # 4. recent searches — strongest topical signal.
    for q in (search_queries or [])[:60]:
        _bump(scores, match_category(q), 0.6)

    # 5. goals — durable interests.
    for g in (goal_texts or [])[:40]:
        _bump(scores, match_category(g), 0.8)

    # Hard cap so one obsession can't starve the other 13 categories.
    return {c: min(w, 6.0) for c, w in scores.items()}
