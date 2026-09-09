"""FTS5 lexical search.

Lexical search is not redundant with vector search — they fail differently. A
vector search finds "documents about the same idea"; a bm25 search finds "the
document containing this exact identifier, filename, or error string". Recall is
built by merging both (see :mod:`nomorals.memory.working`).

If the SQLite build lacks FTS5 the index degrades to unavailable and callers get
empty results rather than an exception.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..core.logging_setup import get_logger
from .db import Database

__all__ = ["FTSIndex", "SearchHit", "escape_fts"]

_log = get_logger(__name__)

#: Reserved characters in the FTS5 query syntax that must be quoted.
#: A string, not a set: ``str.strip`` only accepts a string of characters.
_SPECIAL = '"+-^*():<>{}[]'


def escape_fts(term: str) -> str:
    """Quote a user-supplied term so it is treated as a literal phrase."""
    if not term:
        return '""'
    return '"' + term.replace('"', '""') + '"'


def build_match_query(text: str, *, prefix: bool = True, operator: str = "AND") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    ``prefix=True`` appends ``*`` to each token so partial words match — the
    behaviour users expect from a search box.
    """
    tokens = [t for t in text.replace('"', " ").split() if t.strip(_SPECIAL + " ")]
    if not tokens:
        return ""
    parts = [escape_fts(t) + ("*" if prefix else "") for t in tokens]
    joiner = f" {operator} " if operator.upper() in {"AND", "OR", "NOT"} else " AND "
    return joiner.join(parts)


@dataclass
class SearchHit:
    """One ranked result."""

    rowid: int
    score: float
    columns: dict[str, str]
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rowid": self.rowid,
            "score": self.score,
            "snippet": self.snippet,
            **self.columns,
        }


class FTSIndex:
    """Wrapper over a standalone FTS5 table.

        index = FTSIndex(db, "memories_fts", columns=["content"])
        index.put(memory_rowid, ["the cat sat on the mat"])
        hits = index.search("cat mat", limit=5)
    """

    def __init__(
        self,
        db: Database,
        table: str,
        *,
        columns: Sequence[str],
        available: bool | None = None,
    ) -> None:
        self.db = db
        self.table = table
        self.columns = list(columns)
        if available is None:
            available = db.table_exists(table)
            if not available:
                _log.debug("FTS table %s absent; lexical search disabled", table)
        self.available = available

    # ── writes ───────────────────────────────────────────────────────────────
    def put(self, rowid: int, values: Sequence[str]) -> None:
        """Insert or replace the indexed text for ``rowid``."""
        if not self.available:
            return
        if len(values) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} values, got {len(values)}")
        self.delete(rowid)
        quoted = ", ".join('"' + c + '"' for c in self.columns)
        self.db.execute(
            f'INSERT INTO "{self.table}" (rowid, {quoted}) VALUES (?, {", ".join("?" * len(self.columns))})',
            [rowid, *values],
        )

    def put_many(self, pairs: Iterable[tuple[int, Sequence[str]]]) -> int:
        count = 0
        with self.db.transaction():
            for rowid, values in pairs:
                self.put(rowid, values)
                count += 1
        return count

    def delete(self, rowid: int) -> None:
        if not self.available:
            return
        self.db.execute(f'DELETE FROM "{self.table}" WHERE rowid = ?', (rowid,))

    def rebuild(self) -> None:
        if not self.available:
            return
        self.db.execute(f'DELETE FROM "{self.table}"')

    def optimize(self) -> None:
        if not self.available:
            return
        try:
            self.db.execute(f"INSERT INTO \"{self.table}\"(\"{self.table}\", rank) VALUES('optimize', 0)")
        except sqlite3.Error as exc:  # pragma: no cover - older sqlite
            _log.debug("fts optimize failed: %s", exc)

    # ── reads ────────────────────────────────────────────────────────────────
    def search(
        self,
        text: str,
        *,
        limit: int = 10,
        prefix: bool = True,
        operator: str = "AND",
        snippet_column: str | None = None,
    ) -> list[SearchHit]:
        """Ranked search. Scores are negated bm25 so higher is better."""
        if not self.available:
            return []
        match = build_match_query(text, prefix=prefix, operator=operator)
        if not match:
            return []
        snippet_col = snippet_column or self.columns[0]
        columns_sql = ", ".join('"' + c + '"' for c in self.columns)
        sql = (
            f'SELECT rowid, bm25("{self.table}") AS rank, {columns_sql}, '
            f'snippet("{self.table}", {self.columns.index(snippet_col)}, \'[\', \']\', \'…\', 12) AS snip '
            f'FROM "{self.table}" WHERE "{self.table}" MATCH ? ORDER BY rank LIMIT ?'
        )
        try:
            rows = self.db.query(sql, (match, limit))
        except sqlite3.OperationalError as exc:
            # Malformed MATCH expression from exotic input: fall back to OR.
            _log.debug("fts match failed (%s); retrying with OR", exc)
            if operator.upper() == "AND":
                return self.search(
                    text, limit=limit, prefix=prefix, operator="OR", snippet_column=snippet_col
                )
            return []
        hits: list[SearchHit] = []
        for row in rows:
            hits.append(
                SearchHit(
                    rowid=int(row["rowid"]),
                    score=-float(row["rank"]),
                    columns={c: (row[c] or "") for c in self.columns},
                    snippet=row["snip"] or "",
                )
            )
        return hits

    def count(self) -> int:
        if not self.available:
            return 0
        return int(self.db.scalar(f'SELECT COUNT(*) FROM "{self.table}"', default=0))
