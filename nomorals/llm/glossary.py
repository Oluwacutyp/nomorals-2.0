"""The translation glossary store + enforcement (doc 19 §4, T2).

Devon's glossary rule, in the doc's own words: *when a glossary
term appears in the source, the term is translated FIRST and the
sentence is built around it.*  The mechanism that carries that rule
through a model rung is **protect-with-verified-restore**:

#. Every occurrence of a glossary source term in the text is
   replaced by a numbered placeholder token (``⟦G0⟧``, ``⟦G1⟧`` …)
   — collision-proof glyphs in the translator's own sentinel
   family, one token per occurrence.
#. The protected text goes through the ordinary ladder.
#. Restore verifies each token survived exactly once, then
   substitutes the glossary's TARGET term for it.

A rung that mangles or drops a placeholder fails verification; the
ticket path then re-translates that unit **unprotected** and names
the unenforced terms in its result notes.  A term is never silently
dropped and a placeholder never reaches the reader — the failure is
disclosed, the words are the rung's own.

The store itself is durable and per-scope, in the organ's Batch-B
idiom: a runtime-created table beside ``translate_receipts`` when a
db exists; without one, a JSON document under
``<home>/translate/`` (the receipts file's own fallback posture).
Terms carry the shipped shape — {source_term, target_term, pair,
scope}; pair languages are Devon's codes, with ``""`` meaning "any
language on that side".  Scopes follow the platform's standing
precedence: **owner beats domain beats global** when two rows name
the same source term for the same pair.

Reads are fail-open (a broken glossary store yields no terms, and
translation proceeds — the ticket notes say the glossary could not
be read); writes report honestly with a bool.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "GlossaryStore", "GlossaryTerm", "protect_terms", "restore_terms",
    "SCOPES",
]

SCOPES = ("global", "domain", "owner")
_SCOPE_RANK = {"global": 0, "domain": 1, "owner": 2}

_DDL = (
    "CREATE TABLE IF NOT EXISTS translate_glossary ("
    "glossary_id TEXT NOT NULL, source_lang TEXT NOT NULL, "
    "target_lang TEXT NOT NULL, source_term TEXT NOT NULL, "
    "target_term TEXT NOT NULL, scope TEXT NOT NULL, "
    "updated_at REAL, "
    "PRIMARY KEY (glossary_id, source_lang, target_lang, source_term))"
)


@dataclass(frozen=True)
class GlossaryTerm:
    source_term: str
    target_term: str
    source_lang: str = ""
    target_lang: str = ""
    scope: str = "global"

    def to_dict(self) -> dict[str, str]:
        return {"source_term": self.source_term,
                "target_term": self.target_term,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "scope": self.scope}


def _pair_matches(term: GlossaryTerm, source_lang: str,
                  target_lang: str) -> bool:
    return (term.source_lang in ("", source_lang)
            and term.target_lang in ("", target_lang))


class GlossaryStore:
    """Durable per-glossary term sets.  Reads never raise."""

    def __init__(self, db: Any = None, home: str = "") -> None:
        self._db = db
        self._table_ready = False
        self._path = (Path(home) / "translate" / "glossary.json"
                      if db is None and home else None)

    # -- storage backends ------------------------------------------
    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        self._db.execute(_DDL)
        self._table_ready = True

    def _load_file(self) -> dict[str, list[dict[str, Any]]]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001 — reads fail open
            record_failure("glossary.load_file", exc)
            return {}

    def _save_file(self, data: dict[str, list[dict[str, Any]]]) -> bool:
        if self._path is None:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self._path)
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("glossary.save_file", exc)
            return False

    def _rows(self, glossary_id: str) -> list[GlossaryTerm]:
        if not glossary_id:
            return []
        if self._db is not None:
            try:
                self._ensure_table()
                rows = self._db.query(
                    "SELECT source_term, target_term, source_lang, "
                    "target_lang, scope FROM translate_glossary "
                    "WHERE glossary_id = ?", (glossary_id,))
                return [GlossaryTerm(str(r["source_term"]),
                                     str(r["target_term"]),
                                     str(r["source_lang"] or ""),
                                     str(r["target_lang"] or ""),
                                     str(r["scope"] or "global"))
                        for r in rows]
            except Exception as exc:  # noqa: BLE001 — reads fail open
                record_failure("glossary.rows", exc)
                _log.debug("glossary: read failed", exc_info=True)
                return []
        terms = []
        for row in self._load_file().get(glossary_id, []):
            if isinstance(row, dict) and row.get("source_term"):
                terms.append(GlossaryTerm(
                    str(row["source_term"]), str(row.get("target_term", "")),
                    str(row.get("source_lang", "") or ""),
                    str(row.get("target_lang", "") or ""),
                    str(row.get("scope", "global") or "global")))
        return terms

    # -- writes ------------------------------------------------------
    def put_term(self, glossary_id: str, source_term: str,
                 target_term: str, *, source_lang: str = "",
                 target_lang: str = "", scope: str = "global") -> bool:
        """Add or replace one term.  True when it landed."""
        glossary_id = (glossary_id or "").strip()
        source_term = (source_term or "").strip()
        if not glossary_id or not source_term or not target_term:
            return False
        scope = scope if scope in _SCOPE_RANK else "global"
        term = GlossaryTerm(source_term, target_term,
                            (source_lang or "").strip(),
                            (target_lang or "").strip(), scope)
        if self._db is not None:
            try:
                self._ensure_table()
                self._db.execute(
                    "INSERT OR REPLACE INTO translate_glossary "
                    "(glossary_id, source_lang, target_lang, "
                    "source_term, target_term, scope, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (glossary_id, term.source_lang, term.target_lang,
                     term.source_term, term.target_term, term.scope,
                     time.time()))
                return True
            except Exception as exc:  # noqa: BLE001
                record_failure("glossary.put_term", exc)
                return False
        data = self._load_file()
        rows = [r for r in data.get(glossary_id, [])
                if not (isinstance(r, dict)
                        and r.get("source_term") == term.source_term
                        and r.get("source_lang", "") == term.source_lang
                        and r.get("target_lang", "") == term.target_lang)]
        rows.append(term.to_dict())
        data[glossary_id] = rows
        return self._save_file(data)

    def remove_term(self, glossary_id: str, source_term: str, *,
                    source_lang: str = "", target_lang: str = ""
                    ) -> bool:
        """Remove one term; True when a row was actually removed."""
        glossary_id = (glossary_id or "").strip()
        source_term = (source_term or "").strip()
        if not glossary_id or not source_term:
            return False
        if self._db is not None:
            try:
                self._ensure_table()
                before = self._rows(glossary_id)
                self._db.execute(
                    "DELETE FROM translate_glossary WHERE "
                    "glossary_id = ? AND source_term = ? AND "
                    "source_lang = ? AND target_lang = ?",
                    (glossary_id, source_term, source_lang, target_lang))
                return len(self._rows(glossary_id)) < len(before)
            except Exception as exc:  # noqa: BLE001
                record_failure("glossary.remove_term", exc)
                return False
        data = self._load_file()
        rows = data.get(glossary_id, [])
        kept = [r for r in rows
                if not (isinstance(r, dict)
                        and r.get("source_term") == source_term
                        and r.get("source_lang", "") == source_lang
                        and r.get("target_lang", "") == target_lang)]
        if len(kept) == len(rows):
            return False
        data[glossary_id] = kept
        return self._save_file(data)

    # -- reads -------------------------------------------------------
    def terms_for(self, glossary_id: str, source_lang: str,
                  target_lang: str) -> list[GlossaryTerm]:
        """The resolved term set for one pair: pair-matching rows,
        scope conflicts settled (owner > domain > global), longest
        source terms first so protection prefers the longest match.
        """
        resolved: dict[str, GlossaryTerm] = {}
        for term in self._rows(glossary_id):
            if not _pair_matches(term, source_lang, target_lang):
                continue
            key = term.source_term.casefold()
            current = resolved.get(key)
            if (current is None or _SCOPE_RANK.get(term.scope, 0)
                    > _SCOPE_RANK.get(current.scope, 0)):
                resolved[key] = term
        return sorted(resolved.values(),
                      key=lambda t: -len(t.source_term))

    def fingerprint(self, glossary_id: str, source_lang: str,
                    target_lang: str) -> str:
        """A hash of the glossary id + its resolved term set for
        this pair.  Any term change — add, remove, re-target,
        re-scope — changes the fingerprint, and with it the
        translation cache key: a changed glossary can never be
        served yesterday's cached answer."""
        terms = self.terms_for(glossary_id, source_lang, target_lang)
        canon = json.dumps(
            [glossary_id, source_lang, target_lang,
             [t.to_dict() for t in terms]],
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24]

    # -- enumeration (the management surface's reads) ----------------
    def describe_storage(self) -> str:
        """Where this store's entries live, in owner's words."""
        if self._db is not None:
            return "the translate_glossary table in Devon's database"
        if self._path is not None:
            return "the file " + str(self._path)
        return ("no store at all — neither a database nor a home "
                "directory is configured, so nothing persists")

    def glossary_ids(self) -> list[str]:
        """Every glossary name the store holds, sorted.  A file
        glossary whose last entry was removed still names itself
        here (its key remains, holding no rows); a database
        glossary exists exactly while it has rows.  Reads fail
        open, like every read on this store."""
        if self._db is not None:
            try:
                self._ensure_table()
                rows = self._db.query(
                    "SELECT DISTINCT glossary_id FROM "
                    "translate_glossary ORDER BY glossary_id")
                return [str(r["glossary_id"]) for r in rows
                        if r["glossary_id"]]
            except Exception as exc:  # noqa: BLE001 — reads fail open
                record_failure("glossary.glossary_ids", exc)
                return []
        return sorted(self._load_file().keys())

    def entries(self, glossary_id: str) -> list[GlossaryTerm]:
        """Every stored row of one glossary, unresolved — all
        scopes, all pairs — in display order (owner scope first,
        then domain, then global; terms alphabetical inside a
        scope).  This is what the management surface lists; the
        translator's own read stays :meth:`terms_for`."""
        rows = self._rows(glossary_id)
        return sorted(
            rows,
            key=lambda t: (-_SCOPE_RANK.get(t.scope, 0),
                           t.source_term.casefold(),
                           t.source_lang, t.target_lang))


# ----------------------------------------------------------------------
# protect / restore
# ----------------------------------------------------------------------

_TOKEN_FMT = "⟦G{}⟧"


def protect_terms(text: str, terms: list[GlossaryTerm]
                  ) -> tuple[str, list[tuple[str, GlossaryTerm]]]:
    """Replace every glossary-term occurrence with a placeholder
    token.  Returns the protected text and the occurrence list
    ``[(token, term)]`` in text order.  Matching is
    case-insensitive on word boundaries; longer terms win overlaps
    (``terms`` arrives longest-first from :meth:`terms_for`)."""
    if not text or not terms:
        return text, []
    candidates: list[tuple[int, int, GlossaryTerm]] = []
    for term in terms:
        if not term.source_term:
            continue
        pattern = re.compile(
            r"(?<!\w)" + re.escape(term.source_term) + r"(?!\w)",
            re.IGNORECASE)
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), term))
    if not candidates:
        return text, []
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    accepted: list[tuple[int, int, GlossaryTerm]] = []
    occupied_until = -1
    for start, end, term in candidates:
        if start >= occupied_until:
            accepted.append((start, end, term))
            occupied_until = end
    out: list[str] = []
    occurrences: list[tuple[str, GlossaryTerm]] = []
    cursor = 0
    for idx, (start, end, term) in enumerate(accepted):
        token = _TOKEN_FMT.format(idx)
        out.append(text[cursor:start])
        out.append(token)
        occurrences.append((token, term))
        cursor = end
    out.append(text[cursor:])
    return "".join(out), occurrences


def restore_terms(text: str, occurrences: list[tuple[str, GlossaryTerm]]
                  ) -> tuple[str | None, list[GlossaryTerm]]:
    """Substitute each placeholder's TARGET term back in — only
    when every token survived the rung exactly once.  Returns
    ``(restored_text, [])`` on success, ``(None, [terms whose
    placeholder did not survive])`` on failure; the caller decides
    the honest fallback (the ticket path re-translates unprotected
    and discloses)."""
    if not occurrences:
        return text, []
    lost: list[GlossaryTerm] = []
    out = text
    for token, term in occurrences:
        if out.count(token) != 1:
            lost.append(term)
    if lost:
        return None, lost
    for token, term in occurrences:
        out = out.replace(token, term.target_term)
    return out, []
