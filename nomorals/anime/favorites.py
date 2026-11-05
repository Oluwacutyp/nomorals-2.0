"""Per-user anime favorites — Devon's OWN saved list.

Not an AniList/MyAnimeList account action: rows live in Devon's
database, keyed by the caller's person key (platform + sender id),
so one member's list is never another's.  Only titles that a source
actually returned can be saved — a favorite is a stored reference
to a real record (source + its id + the title as shown), never
free-typed fiction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

__all__ = ["Favorite", "FavoritesStore"]

_TABLE = """
CREATE TABLE IF NOT EXISTS anime_favorites (
    user_key TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'anime',
    title TEXT NOT NULL DEFAULT '',
    added_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (user_key, source, source_id)
)
"""


@dataclass
class Favorite:
    source: str
    source_id: int
    kind: str
    title: str
    added_at: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.source}:{self.source_id}"


class FavoritesStore:
    """Favorites over anything with the Database execute/query API."""

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ready = False

    def _ensure(self) -> bool:
        if self._db is None:
            return False
        if not self._ready:
            self._db.execute(_TABLE)
            self._ready = True
        return True

    def add(self, user_key: str, *, source: str, source_id: int,
            kind: str, title: str) -> bool:
        """Save one favorite; False when it was already there."""
        if not self._ensure():
            raise RuntimeError("favorites storage unavailable")
        cur = self._db.execute(
            "INSERT OR IGNORE INTO anime_favorites "
            "(user_key, source, source_id, kind, title, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_key, source, int(source_id), kind, title, time.time()))
        return bool(getattr(cur, "rowcount", 0))

    def list(self, user_key: str) -> list[Favorite]:
        if not self._ensure():
            return []
        rows = self._db.query(
            "SELECT source, source_id, kind, title, added_at "
            "FROM anime_favorites WHERE user_key = ? "
            "ORDER BY added_at ASC", (user_key,))
        return [Favorite(source=str(r["source"]),
                         source_id=int(r["source_id"]),
                         kind=str(r["kind"]), title=str(r["title"]),
                         added_at=float(r["added_at"] or 0))
                for r in rows]

    def remove(self, user_key: str, *, source: str = "",
               source_id: int | None = None,
               title: str = "") -> Favorite | None:
        """Remove by record ref or by title (case-insensitive).

        Returns the removed row, or None when nothing matched."""
        if not self._ensure():
            return None
        rows = self.list(user_key)
        target: Favorite | None = None
        for fav in rows:
            if source and fav.source == source \
                    and fav.source_id == source_id:
                target = fav
                break
            if title and fav.title.lower() == title.lower():
                target = fav
                break
        if target is None:
            return None
        self._db.execute(
            "DELETE FROM anime_favorites WHERE user_key = ? "
            "AND source = ? AND source_id = ?",
            (user_key, target.source, target.source_id))
        return target
