"""MapReceipt: the durable record of what Atlas did (doc 21 §6).

Every map operation returns a ticket, the translator-ticket idiom:
which operation ran, which packs (and digests) it touched, whether
it worked offline or online, which provider answered (``"local"``
for Devon's own planes — providers are optional adapters under doc
21, never the default identity), and the outcome, honestly graded.
Later passes (geocoder M4, router M5, broker M6, agent M7) mint
their receipts through this one store — there is no second receipt
system.

Storage follows the shipped Batch-B idiom (``llm/receipts.py``):
when the context carries the shared db, receipts land in the
runtime-created ``map_receipts`` table (migrations are frozen);
with no db they append to ``<home>/maps/receipts.jsonl``.  The full
receipt rides in a JSON payload, so new fields never need a schema
change; operation + outcome are real columns for filtering.
Writing a receipt is fail-open end to end: a broken store records
the failure in the core failure ledger and the operation's own
result still returns.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "OP_GEOCODE", "OP_PACK_INSTALL", "OP_PACK_VERIFY", "OP_RENDER",
    "OP_ROUTE", "OPERATIONS", "OUTCOME_FAILED", "OUTCOME_OK",
    "OUTCOME_UNAVAILABLE", "POSTURE_OFFLINE", "POSTURE_ONLINE",
    "MapReceipt", "MapReceiptStore", "new_receipt_id",
]

#: The operations a MapReceipt can name.  M1 mints pack_verify and
#: render; the rest are the vocabulary later passes mint through.
OP_PACK_INSTALL = "pack_install"
OP_PACK_VERIFY = "pack_verify"
OP_GEOCODE = "geocode"
OP_ROUTE = "route"
OP_RENDER = "render"
OPERATIONS = (OP_PACK_INSTALL, OP_PACK_VERIFY, OP_GEOCODE,
              OP_ROUTE, OP_RENDER)

#: Where the answer came from: her own planes or the network.
POSTURE_OFFLINE = "offline"
POSTURE_ONLINE = "online"

#: Outcome grades.  "unavailable" is a first-class, honest answer —
#: the capability is not there — never dressed up as ok or failed.
OUTCOME_OK = "ok"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_FAILED = "failed"

_DDL = (
    "CREATE TABLE IF NOT EXISTS map_receipts ("
    "receipt_id TEXT PRIMARY KEY, at REAL, operation TEXT, "
    "outcome TEXT, payload TEXT)"
)

#: The JSONL fallback scans at most this many newest lines on a
#: filtered read — receipts are evidence, not an archive.
_JSONL_SCAN = 5000


def new_receipt_id() -> str:
    """A receipt id (internal — never rendered on owner surfaces)."""
    return "map-" + uuid.uuid4().hex[:16]


@dataclass(frozen=True)
class MapReceipt:
    """One operation's ticket: what ran, on what, with what result."""

    receipt_id: str
    at: float
    operation: str
    outcome: str
    posture: str = POSTURE_OFFLINE
    provider: str = "local"
    pack_refs: tuple[str, ...] = ()
    pack_digests: tuple[str, ...] = ()
    detail: str = ""

    @classmethod
    def mint(cls, operation: str, outcome: str, *,
             posture: str = POSTURE_OFFLINE, provider: str = "local",
             pack_refs: tuple[str, ...] = (),
             pack_digests: tuple[str, ...] = (),
             detail: str = "", at: float | None = None) -> MapReceipt:
        return cls(receipt_id=new_receipt_id(),
                   at=time.time() if at is None else float(at),
                   operation=operation, outcome=outcome, posture=posture,
                   provider=provider, pack_refs=tuple(pack_refs),
                   pack_digests=tuple(pack_digests), detail=detail)

    def to_dict(self) -> dict[str, Any]:
        return {"receipt_id": self.receipt_id, "at": self.at,
                "operation": self.operation, "outcome": self.outcome,
                "posture": self.posture, "provider": self.provider,
                "pack_refs": list(self.pack_refs),
                "pack_digests": list(self.pack_digests),
                "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict) -> MapReceipt:
        return cls(receipt_id=str(data.get("receipt_id", "")),
                   at=float(data.get("at", 0.0)),
                   operation=str(data.get("operation", "")),
                   outcome=str(data.get("outcome", "")),
                   posture=str(data.get("posture", POSTURE_OFFLINE)),
                   provider=str(data.get("provider", "local")),
                   pack_refs=tuple(data.get("pack_refs", ())),
                   pack_digests=tuple(data.get("pack_digests", ())),
                   detail=str(data.get("detail", "")))


class MapReceiptStore:
    """Append + read for map receipts.  Never raises."""

    def __init__(self, db: Any = None, home: str = "") -> None:
        self._db = db
        self._table_ready = False
        self._path: Path | None = None
        if db is None and home:
            self._path = Path(home) / "maps" / "receipts.jsonl"

    # ── write ────────────────────────────────────────────────────

    def append(self, receipt: MapReceipt) -> bool:
        """Persist one receipt; True when it landed.  Never raises."""
        try:
            payload = receipt.to_dict()
            if self._db is not None:
                self._ensure_table()
                self._db.execute(
                    "INSERT OR REPLACE INTO map_receipts "
                    "(receipt_id, at, operation, outcome, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (receipt.receipt_id, receipt.at, receipt.operation,
                     receipt.outcome,
                     json.dumps(payload, ensure_ascii=False)))
                return True
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False))
                    fh.write("\n")
                return True
            return False
        except Exception as exc:  # noqa: BLE001 — receipts never gate
            record_failure("maps.receipts_append", exc)
            _log.debug("maps: receipt append failed", exc_info=True)
            return False

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        self._db.execute(_DDL)
        self._table_ready = True

    # ── read ─────────────────────────────────────────────────────

    def recent(self, *, limit: int = 50,
               operation: str = "") -> list[MapReceipt]:
        """Newest-first receipts, optionally filtered by operation.

        A broken store reads as empty.
        """
        limit = max(1, int(limit))
        try:
            if self._db is not None:
                return self._recent_db(limit, operation)
            rows = self._read_jsonl()
        except Exception as exc:  # noqa: BLE001
            record_failure("maps.receipts_read", exc)
            _log.debug("maps: receipt read failed", exc_info=True)
            return []
        out: list[MapReceipt] = []
        for payload in rows:
            if operation and payload.get("operation") != operation:
                continue
            out.append(MapReceipt.from_dict(payload))
            if len(out) >= limit:
                break
        return out

    def _recent_db(self, limit: int,
                   operation: str) -> list[MapReceipt]:
        self._ensure_table()
        sql = "SELECT payload FROM map_receipts"
        params: list[Any] = []
        if operation:
            sql += " WHERE operation = ?"
            params.append(operation)
        sql += " ORDER BY at DESC LIMIT ?"
        params.append(limit)
        out: list[MapReceipt] = []
        for row in self._db.query(sql, params):
            try:
                out.append(MapReceipt.from_dict(
                    json.loads(row.get("payload") or "{}")))
            except (TypeError, ValueError):
                continue
        return out

    def _read_jsonl(self) -> list[dict[str, Any]]:
        """Newest-first receipt payloads from the JSONL fallback."""
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
