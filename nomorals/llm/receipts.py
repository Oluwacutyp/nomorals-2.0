"""Translation receipts: the durable record of what the organ did.

Doc 19 §4 (adopted 2026-11-10, pass T1): every translation the
organ serves leaves a receipt — which rung answered, which language
pair, whether the quality gate ran and what it said, whether the
answer came from the in-process cache, and who asked (when the call
path carries an attribution).  Receipts are evidence, not a gate:
writing one is fail-open end to end — a broken store records the
failure in the core failure ledger and the translation still
returns.

Storage follows the organ's own idiom: when the context carries the
shared db, receipts land in a runtime-created table beside
``translate_prefs`` (the Batch-B pattern — migrations are frozen);
with no db they append to ``<home>/translate/receipts.jsonl`` (the
doc-16/17 journal shape).  The full receipt rides in a JSON payload
column / the JSONL line itself, so new fields never need a schema
change; the filterable fields (pair, rung) are real columns on the
table path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["TranslationReceiptStore", "new_receipt_id"]

_DDL = (
    "CREATE TABLE IF NOT EXISTS translate_receipts ("
    "receipt_id TEXT PRIMARY KEY, at REAL, source_lang TEXT, "
    "target_lang TEXT, rung TEXT, payload TEXT)"
)

#: The JSONL fallback scans at most this many newest lines on a
#: filtered read — receipts are small and the file is an evidence
#: journal, not an archive.
_JSONL_SCAN = 5000


def new_receipt_id() -> str:
    """A receipt id (internal — never rendered on owner surfaces)."""
    import uuid

    return "tr-" + uuid.uuid4().hex[:16]


class TranslationReceiptStore:
    """Append + read for translation receipts.  Never raises."""

    def __init__(self, db: Any = None, home: str = "") -> None:
        self._db = db
        self._table_ready = False
        self._path: Path | None = None
        if db is None and home:
            self._path = Path(home) / "translate" / "receipts.jsonl"

    # ── write ────────────────────────────────────────────────────

    def append(self, receipt: dict[str, Any]) -> bool:
        """Persist one receipt; True when it landed.  Never raises:
        a failed write is ledgered and the caller carries on."""
        try:
            if self._db is not None:
                self._ensure_table()
                self._db.execute(
                    "INSERT OR REPLACE INTO translate_receipts "
                    "(receipt_id, at, source_lang, target_lang, rung, "
                    "payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(receipt.get("receipt_id", "")),
                     float(receipt.get("at", 0.0)),
                     str(receipt.get("source_lang", "")),
                     str(receipt.get("target_lang", "")),
                     str(receipt.get("rung", "")),
                     json.dumps(receipt, ensure_ascii=False)))
                return True
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(receipt, ensure_ascii=False))
                    fh.write("\n")
                return True
            return False
        except Exception as exc:  # noqa: BLE001 — receipts never gate
            record_failure("translate.receipts_append", exc)
            _log.debug("translate: receipt append failed",
                       exc_info=True)
            return False

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        self._db.execute(_DDL)
        self._table_ready = True

    # ── read ─────────────────────────────────────────────────────

    def recent(self, *, limit: int = 50, source_lang: str = "",
               target_lang: str = "",
               rung: str = "") -> list[dict[str, Any]]:
        """Newest-first receipts, optionally filtered by language
        pair and/or rung.  A broken store reads as empty."""
        limit = max(1, int(limit))
        try:
            if self._db is not None:
                return self._recent_db(limit, source_lang,
                                       target_lang, rung)
            rows = self._read_jsonl()
        except Exception as exc:  # noqa: BLE001
            record_failure("translate.receipts_read", exc)
            _log.debug("translate: receipt read failed", exc_info=True)
            return []
        out: list[dict[str, Any]] = []
        for receipt in rows:
            if source_lang and receipt.get("source_lang") != source_lang:
                continue
            if target_lang and receipt.get("target_lang") != target_lang:
                continue
            if rung and receipt.get("rung") != rung:
                continue
            out.append(receipt)
            if len(out) >= limit:
                break
        return out

    def _recent_db(self, limit: int, source_lang: str,
                   target_lang: str,
                   rung: str) -> list[dict[str, Any]]:
        self._ensure_table()
        clauses: list[str] = []
        params: list[Any] = []
        if source_lang:
            clauses.append("source_lang = ?")
            params.append(source_lang)
        if target_lang:
            clauses.append("target_lang = ?")
            params.append(target_lang)
        if rung:
            clauses.append("rung = ?")
            params.append(rung)
        sql = "SELECT payload FROM translate_receipts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY at DESC LIMIT ?"
        params.append(limit)
        out: list[dict[str, Any]] = []
        for row in self._db.query(sql, params):
            try:
                out.append(json.loads(row.get("payload") or "{}"))
            except (TypeError, ValueError):
                continue
        return out

    def _read_jsonl(self) -> list[dict[str, Any]]:
        """Newest-first receipts from the JSONL fallback."""
        if self._path is None or not self._path.is_file():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in reversed(lines[-_JSONL_SCAN:]):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out
