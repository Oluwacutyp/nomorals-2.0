"""Parity proof between two document-store rungs (Mongo program P1).

The carry/cutover discipline of the Mongo program is: *parity proven before
any read flips*. This module is the prover. It compares two
:class:`~nomorals.storage.docstore.DocumentStore` implementations over one
collection — counts, an ordered SHA-256 over a canonical serialization of
every document (the same checksum thinking as the backup manifests and the
migration tamper guard), and the exact set of divergent ids — and returns a
:class:`ParityReport` an operator (or a test) can gate on.

:func:`run_kv_parity_housekeeping` is the housekeeping entry point: when the
KV store is running in shadow mode it sweeps expired documents on both
rungs and runs the comparison, journaling any divergence to the failure
ledger. It returns ``None`` when there is no shadow pair to check, so a
caller on any beat can invoke it unconditionally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

__all__ = [
    "ParityReport",
    "check_document_parity",
    "run_kv_parity_housekeeping",
]

_log = get_logger(__name__)

#: Report at most this many divergent ids in detail; counts stay exact.
_MAX_REPORTED = 50


@dataclass
class ParityReport:
    """The verdict of one two-rung comparison over one collection."""

    collection: str
    count_a: int
    count_b: int
    checksum_a: str
    checksum_b: str
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    differing: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.count_a == self.count_b
            and self.checksum_a == self.checksum_b
            and not self.only_in_a
            and not self.only_in_b
            and not self.differing
            and not self.truncated
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "ok": self.ok,
            "count_a": self.count_a,
            "count_b": self.count_b,
            "checksum_a": self.checksum_a,
            "checksum_b": self.checksum_b,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "differing": self.differing,
            "truncated": self.truncated,
        }


def _canonical(doc: dict[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)


def _digest(docs: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(_canonical(doc).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def check_document_parity(store_a: Any, store_b: Any, collection: str = "kv_store",
                          *, limit: int = 100_000) -> ParityReport:
    """Compare ``collection`` across two stores; never raises on divergence.

    Divergence is data for the report, not an exception. ``limit`` bounds
    the comparison; hitting it marks the report ``truncated`` (and therefore
    not ok) instead of silently passing on a partial read.
    """
    docs_a = store_a.find(collection, include_expired=True, limit=limit + 1)
    docs_b = store_b.find(collection, include_expired=True, limit=limit + 1)
    truncated = len(docs_a) > limit or len(docs_b) > limit
    docs_a, docs_b = docs_a[:limit], docs_b[:limit]
    by_id_a = {d["_id"]: d for d in docs_a}
    by_id_b = {d["_id"]: d for d in docs_b}
    only_in_a = sorted(set(by_id_a) - set(by_id_b))
    only_in_b = sorted(set(by_id_b) - set(by_id_a))
    differing = sorted(
        doc_id for doc_id in set(by_id_a) & set(by_id_b)
        if _canonical(by_id_a[doc_id]) != _canonical(by_id_b[doc_id])
    )
    return ParityReport(
        collection=collection,
        count_a=len(docs_a),
        count_b=len(docs_b),
        checksum_a=_digest(docs_a),
        checksum_b=_digest(docs_b),
        only_in_a=only_in_a[:_MAX_REPORTED],
        only_in_b=only_in_b[:_MAX_REPORTED],
        differing=differing[:_MAX_REPORTED],
        truncated=truncated,
    )


def run_kv_parity_housekeeping(db: Any, settings: Any) -> ParityReport | None:
    """Sweep + parity-check the KV shadow pair, if one is configured.

    Returns the :class:`ParityReport` when the attached store is a
    :class:`~nomorals.storage.docstore.ShadowDocumentStore`; ``None`` when
    the install is not shadowing (the normal case — nothing to prove).
    Divergences are journaled; a clean report is logged at debug level.
    """
    from .docstore import ShadowDocumentStore

    resolution = getattr(db, "store_resolution", None)
    store = getattr(db, "document_store", None)
    if resolution is None or not isinstance(store, ShadowDocumentStore):
        return None
    store.sweep_expired("kv_store")
    report = check_document_parity(store.primary, store.secondary, "kv_store")
    if report.ok:
        _log.debug("kv parity ok: %d documents, checksum %s",
                   report.count_a, report.checksum_a[:12])
    else:
        record_failure(
            "storage.parity",
            f"kv parity divergence: counts {report.count_a}/{report.count_b} "
            f"only_primary={len(report.only_in_a)} only_shadow={len(report.only_in_b)} "
            f"differing={len(report.differing)} truncated={report.truncated}")
        _log.warning("kv parity divergence: %s", report.to_dict())
    return report
