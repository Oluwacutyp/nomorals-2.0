"""The durable translation cache (doc 19 §4, pass T2).

T1's cache was the organ's in-process dict: successes only, gone at
restart.  Doc 19 demands a persistent cache keyed on the normalized
source hash + target + model + glossary + revision, with
invalidation — a translation served yesterday by a rung that has
since changed must not silently stand in for today's answer.

This store is that durable layer, in the organ's own Batch-B idiom:
a runtime-created table beside ``translate_receipts`` /
``translate_prefs`` (migrations are frozen), the full entry riding
in a JSON payload column so new fields never need a schema change.
The key is a hash of (source text, source lang, target lang, key
extra) — the ticket path folds its glossary/preserve fingerprint
into the key extra, so changing a glossary changes the key and the
old entry is simply never read again.

Honesty rules:

- **TTL.** Entries older than :data:`CACHE_TTL_S` (30 days — doc
  21's news-cache posture, applied organ-wide) read as misses.
  Nothing stale is served silently.
- **Model invalidation.** An entry records the local model
  reference that produced it; the organ refuses a ``local``-rung
  entry whose recorded reference differs from the configured one.
- **Successes only** is the organ's rule and stays the organ's —
  this store persists whatever it is handed and the organ only
  hands it successes.
- **Fail-open.** A cache is an optimization: every failure here
  reads as a miss / writes as a no-op and lands in the core failure
  ledger.  A broken cache can slow translation down; it can never
  break or falsify one.

Db-less translators keep T1's semantics exactly — the in-process
dict is their whole cache (receipts carry the JSONL evidence
fallback; a cache, unlike a receipt, is not evidence).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["TranslationCacheStore", "cache_key_for", "CACHE_TTL_S"]

_DDL = (
    "CREATE TABLE IF NOT EXISTS translate_cache ("
    "cache_key TEXT PRIMARY KEY, source_lang TEXT, "
    "target_lang TEXT, rung TEXT, stored_at REAL, payload TEXT)"
)

#: Entries older than this read as misses (doc 21's 30-day cache
#: posture for news, applied to the organ's durable cache as a
#: whole — no translation is served stale in silence).
CACHE_TTL_S = 30.0 * 24 * 3600


def cache_key_for(text: str, source_lang: str, target_lang: str,
                  key_extra: str = "") -> str:
    """The durable key: a hash of the normalized source text, the
    language pair, and the caller's key extra (the ticket path's
    glossary/preserve fingerprint rides there)."""
    raw = json.dumps([text, source_lang, target_lang, key_extra],
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TranslationCacheStore:
    """Point lookups over the durable cache.  Never raises."""

    def __init__(self, db: Any = None) -> None:
        self._db = db
        self._table_ready = False

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        self._db.execute(_DDL)
        self._table_ready = True

    def get(self, key: str) -> dict[str, Any] | None:
        """The payload stored under ``key``, or None — absent,
        expired, unparsable, or store-less all read as a miss."""
        if self._db is None or not key:
            return None
        try:
            self._ensure_table()
            row = self._db.query_one(
                "SELECT payload FROM translate_cache "
                "WHERE cache_key = ?", (key,))
            if not row:
                return None
            payload = json.loads(row.get("payload") or "{}")
            if not isinstance(payload, dict) or "outcome" not in payload:
                return None
            stored_at = float(payload.get("stored_at", 0.0))
            if stored_at <= 0 or time.time() - stored_at > CACHE_TTL_S:
                return None  # expired — a miss, never a stale serve
            return payload
        except Exception as exc:  # noqa: BLE001 — a cache never gates
            record_failure("translate.cache_get", exc)
            _log.debug("translate: cache read failed", exc_info=True)
            return None

    def put(self, key: str, *, source_lang: str, target_lang: str,
            rung: str, payload: dict[str, Any]) -> bool:
        """Persist one entry; True when it landed.  Never raises."""
        if self._db is None or not key:
            return False
        try:
            self._ensure_table()
            body = dict(payload)
            body.setdefault("stored_at", time.time())
            self._db.execute(
                "INSERT OR REPLACE INTO translate_cache "
                "(cache_key, source_lang, target_lang, rung, "
                "stored_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (key, source_lang, target_lang, rung,
                 float(body["stored_at"]),
                 json.dumps(body, ensure_ascii=False)))
            return True
        except Exception as exc:  # noqa: BLE001
            record_failure("translate.cache_put", exc)
            _log.debug("translate: cache write failed", exc_info=True)
            return False
