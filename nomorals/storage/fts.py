"""FTS5 lexical search.

Lexical search is not redundant with vector search — they fail differently. A
vector search finds "documents about the same idea"; a bm25 search finds "the
document containing this exact identifier, filename, or error string". Recall is
built by merging both (see :mod:`nomorals.memory.working`).

If the SQLite build lacks FTS5 the index degrades to unavailable and callers get
empty results rather than an exception.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines
from .db import Database

__all__ = [
    "FTSIndex",
    "SearchHit",
    "build_match_query",
    "escape_fts",
    "near_query",
    "phrase_query",
]

_log = get_logger(__name__)

#: Reserved characters in the FTS5 query syntax that must be quoted.
#: A string, not a set: ``str.strip`` only accepts a string of characters.
_SPECIAL = '"+-^*():<>{}[]'


def escape_fts(term: str) -> str:
    """Quote a user-supplied term so it is treated as a literal phrase."""
    if not term:
        return '""'
    return '"' + term.replace('"', '""') + '"'


#: ``col:term`` facet syntax (omnibus F1.1: ``author:``, ``tag:``). Unknown
#: facets fall through as free-text terms rather than erroring.
_FACET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.+)$")


def build_match_query(
    text: str,
    *,
    prefix: bool = True,
    operator: str = "AND",
    columns: Sequence[str] | None = None,
) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    ``prefix=True`` appends ``*`` to each token so partial words match — the
    behaviour users expect from a search box. ``col:term`` tokens become
    column filters (``{title : "term"*}`); when ``columns`` is given, facets
    naming other columns fall through as free text.
    """
    tokens = [t for t in text.replace('"', " ").split() if t.strip(_SPECIAL + " ")]
    if not tokens:
        return ""
    allowed = set(columns or ())
    parts: list[str] = []
    for token in tokens:
        match = _FACET_RE.match(token)
        if match and (not allowed or match.group(1) in allowed):
            col, term = match.group(1), match.group(2).strip(_SPECIAL + " ")
            if term:
                parts.append(
                    "{" + col + " : " + escape_fts(term)
                    + ("*" if prefix else "") + "}"
                )
                continue
        parts.append(escape_fts(token) + ("*" if prefix else ""))
    if not parts:
        return ""
    joiner = f" {operator} " if operator.upper() in {"AND", "OR", "NOT"} else " AND "
    return joiner.join(parts)


def phrase_query(text: str) -> str:
    """Exact-phrase MATCH expression (``"word1 word2"``)."""
    tokens = [t for t in text.replace('"', " ").split() if t.strip(_SPECIAL + " ")]
    if not tokens:
        return ""
    return '"' + " ".join(t.replace('"', '""') for t in tokens) + '"'


def near_query(terms: Sequence[str], distance: int = 10) -> str:
    """NEAR MATCH expression: terms within ``distance`` tokens of each other."""
    clean = [escape_fts(t) for t in terms if t.strip(_SPECIAL + " ")]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return f"NEAR({' '.join(clean)}, {max(1, distance)})"


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

    ``tokenizer`` picks the FTS5 tokenizer: ``"unicode61"`` (default,
    ``remove_diacritics 2`` folds accents — the conversational-English
    recommendation), ``"porter"`` (unicode61 + English stemming), or
    ``"trigram"`` (substring matching for CJK/fuzzy fallback; 3–5x index
    bloat, queries under 3 chars degrade — don't use it for English).
    """

    def __init__(
        self,
        db: Database,
        table: str,
        *,
        columns: Sequence[str],
        available: bool | None = None,
        tokenizer: str = "unicode61",
        tokenize_args: str = "remove_diacritics 2",
        auto_create: bool = False,
    ) -> None:
        self.db = db
        self.table = table
        self.columns = list(columns)
        self.tokenizer = tokenizer
        self.tokenize_args = tokenize_args
        if available is None:
            available = db.table_exists(table)
            if not available and auto_create:
                self.ensure_table()
                available = db.table_exists(table)
            if not available:
                _log.debug("FTS table %s absent; lexical search disabled", table)
        self.available = available

    # ── schema ─────────────────────────────────────────────────────────────
    def build_ddl(self) -> str:
        """``CREATE VIRTUAL TABLE`` for this index (migrations can embed it).

        Stops hand-written FTS DDL from drifting away from the index's own
        tokenizer/column configuration.
        """
        cols = ", ".join('"' + c + '"' for c in self.columns)
        # NOTE: FTS5 tokenizer arguments are space-separated and must NOT be
        # quoted — `tokenize='unicode61 "remove_diacritics 2"'` is a parse
        # error (verified against SQLite 3.45.1).
        directive = self.tokenizer
        if self.tokenize_args:
            directive += f" {self.tokenize_args}"
        return (
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{self.table}" '
            f"USING fts5({cols}, tokenize='{directive}')"
        )

    def ensure_table(self) -> bool:
        """Create the FTS table if missing. Returns True when available."""
        try:
            self.db.execute(self.build_ddl())
        except Exception as exc:  # noqa: BLE001 - Database raises StorageError
            _log.warning("cannot create FTS table %s: %s", self.table, exc)
            return False
        self.available = self.db.table_exists(self.table)
        return self.available

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
        except Exception as exc:  # noqa: BLE001 - older sqlite / StorageError
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
        weights: dict[str, float] | None = None,
        highlight_tags: tuple[str, str] | None = None,
    ) -> list[SearchHit]:
        """Ranked search. Scores are negated bm25 so higher is better.

        ``weights`` maps column → weight for per-column bm25 relevance
        (a hit in ``title`` can count more than one in ``body``).
        ``highlight_tags`` (e.g. ``("<b>", "</b>")``) wraps matches in the
        snippet via FTS5 ``highlight()`` instead of the default ``[``/``]``.
        """
        if not self.available:
            return []
        match = build_match_query(text, prefix=prefix, operator=operator,
                                  columns=self.columns)
        if not match:
            return []
        snippet_col = snippet_column or self.columns[0]
        columns_sql = ", ".join('"' + c + '"' for c in self.columns)
        if weights:
            bm25_args = ", ".join(
                str(float(weights.get(c, 1.0))) for c in self.columns
            )
            rank_sql = f'bm25("{self.table}", {bm25_args})'
        else:
            rank_sql = f'bm25("{self.table}")'
        if highlight_tags:
            open_tag, close_tag = highlight_tags
            snip_sql = (
                f'highlight("{self.table}", {self.columns.index(snippet_col)}, '
                f"'{open_tag}', '{close_tag}') AS snip"
            )
        else:
            snip_sql = (
                f'snippet("{self.table}", {self.columns.index(snippet_col)}, '
                f"'[', ']', '…', 12) AS snip"
            )
        sql = (
            f'SELECT rowid, {rank_sql} AS rank, {columns_sql}, {snip_sql} '
            f'FROM "{self.table}" WHERE "{self.table}" MATCH ? ORDER BY rank LIMIT ?'
        )
        try:
            rows = self.db.query(sql, (match, limit))
        except sqlite3.OperationalError as exc:
            # Malformed MATCH expression from exotic input: fall back to OR.
            _log.debug("fts match failed (%s); retrying with OR", exc)
            if operator.upper() == "AND":
                return self.search(
                    text, limit=limit, prefix=prefix, operator="OR",
                    snippet_column=snippet_col, weights=weights,
                    highlight_tags=highlight_tags,
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

    def suggest(self, prefix: str, *, limit: int = 8) -> list[str]:
        """Term completions for ``prefix`` from the ``fts5vocab`` table.

        The stdlib SQLite has no spellfix1; vocabulary-driven suggestion is
        the portable fallback (trigram-overlap idea, exact-prefix edition).
        NOTE: fts5vocab tables reject ``MATCH`` ("unable to use function
        MATCH in the requested context") — prefix filtering goes through
        ``LIKE`` instead.
        """
        if not self.available:
            return []
        clean = prefix.strip().lower()
        if len(clean) < 2:
            return []
        vocab = f"{self.table}_vocab"
        # Escape LIKE metacharacters in the user prefix.
        like_prefix = clean.replace("\\", "\\\\").replace("%", "\\%").replace(
            "_", "\\_")
        try:
            rows = self.db.query(
                f"SELECT DISTINCT term FROM \"{vocab}\" "
                f"WHERE term LIKE ? ESCAPE '\\' ORDER BY term LIMIT ?",
                (like_prefix + "%", limit),
            )
        except Exception:  # noqa: BLE001 - vocab table missing / StorageError
            # fts5vocab table not created for this index.
            return []
        return [r["term"] for r in rows]

    def ensure_vocab(self) -> bool:
        """Create the ``fts5vocab`` table backing :meth:`suggest`."""
        if not self.available:
            return False
        try:
            self.db.execute(
                f'CREATE VIRTUAL TABLE IF NOT EXISTS "{self.table}_vocab" '
                f'USING fts5vocab("{self.table}", "row")'
            )
        except Exception as exc:  # noqa: BLE001 - Database raises StorageError
            _log.debug("fts5vocab for %s failed: %s", self.table, exc)
            return False
        return True

    def check_drift(self, source_count: int) -> dict[str, Any]:
        """Detect index/source drift: one index row per source row.

        A source row changed without re-indexing silently stops being
        findable — this is the self-check that catches it (procure).
        """
        indexed = self.count()
        return {
            "table": self.table,
            "indexed": indexed,
            "source": source_count,
            "drift": indexed - source_count,
            "healthy": indexed == source_count,
        }

    def count(self) -> int:
        if not self.available:
            return 0
        return int(self.db.scalar(f'SELECT COUNT(*) FROM "{self.table}"', default=0))

    def format_stats(self, theme: Any = None) -> str:
        """Human-readable index overview through the shared style layer."""
        theme = theme or active_theme()
        return "\n".join([
            header("fts index", theme=theme),
            *kv_lines(
                {
                    "table": self.table,
                    "columns": ", ".join(self.columns),
                    "tokenizer": f"{self.tokenizer} ({self.tokenize_args})",
                    "available": self.available,
                    "documents": self.count(),
                },
                theme=theme,
            ),
        ])
