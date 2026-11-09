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

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

__all__ = [
    "ParityReport",
    "check_document_parity",
    "check_document_parity_streaming",
    "collection_checksum",
    "ensure_shadow_parity_job",
    "run_kv_parity_housekeeping",
    "SHADOW_PARITY_JOB_NAME",
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


def _paged_docs(store: Any, collection: str, *, page: int):
    """Yield every document of ``collection`` in ``_id`` order, one page
    in memory at a time (the seam's ``after_id`` cursor)."""
    after: str | None = None
    while True:
        docs = store.find(collection, include_expired=True,
                          after_id=after, limit=page)
        if not docs:
            return
        yield from docs
        after = docs[-1]["_id"]
        if len(docs) < page:
            return


class _MergeSide:
    """One side of a streaming merge: a paged cursor plus the rolling
    count + checksum over everything it has yielded."""

    def __init__(self, store: Any, collection: str, page: int) -> None:
        self._docs = _paged_docs(store, collection, page=page)
        self._digest = hashlib.sha256()
        self.count = 0
        self.current: dict[str, Any] | None = None
        self.advance()

    def advance(self) -> None:
        doc = next(self._docs, None)
        if doc is not None:
            if self.current is not None and doc["_id"] <= self.current["_id"]:
                # The seam contract is _id-ordered find; a store that
                # breaks it would make the merge report garbage. Fail
                # loudly instead of issuing a false verdict.
                raise ValueError(
                    f"store returned documents out of _id order: "
                    f"{doc['_id']!r} after {self.current['_id']!r}")
            self._digest.update(_canonical(doc).encode("utf-8"))
            self._digest.update(b"\n")
            self.count += 1
        self.current = doc

    @property
    def checksum(self) -> str:
        return self._digest.hexdigest()


def check_document_parity_streaming(store_a: Any, store_b: Any,
                                    collection: str, *,
                                    page: int = 1000) -> ParityReport:
    """Compare ``collection`` across two stores with bounded memory.

    The streaming twin of :func:`check_document_parity`: a two-pointer
    merge over both stores' ``_id``-ordered cursors. Only one page per
    side is ever held, so collections far beyond the in-memory
    comparator's ``limit`` compare exactly. Verdict semantics are the
    same contract — counts, the same ordered canonical checksums
    (digested incrementally, byte-identical to :func:`_digest` over
    the same documents), the same divergent-id lists capped at
    ``_MAX_REPORTED`` — and ``truncated`` is always False: the merge
    reads both collections to exhaustion, so there is no partial-read
    state to confess. Divergence is data for the report, never an
    exception.
    """
    side_a = _MergeSide(store_a, collection, page)
    side_b = _MergeSide(store_b, collection, page)
    only_in_a: list[str] = []
    only_in_b: list[str] = []
    differing: list[str] = []
    while side_a.current is not None or side_b.current is not None:
        a, b = side_a.current, side_b.current
        if b is None or (a is not None and a["_id"] < b["_id"]):
            if len(only_in_a) < _MAX_REPORTED:
                only_in_a.append(a["_id"])
            side_a.advance()
        elif a is None or b["_id"] < a["_id"]:
            if len(only_in_b) < _MAX_REPORTED:
                only_in_b.append(b["_id"])
            side_b.advance()
        else:
            if _canonical(a) != _canonical(b) \
                    and len(differing) < _MAX_REPORTED:
                differing.append(a["_id"])
            side_a.advance()
            side_b.advance()
    return ParityReport(
        collection=collection,
        count_a=side_a.count,
        count_b=side_b.count,
        checksum_a=side_a.checksum,
        checksum_b=side_b.checksum,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        differing=differing,
        truncated=False,
    )


def collection_checksum(store: Any, collection: str, *,
                        page: int = 1000) -> tuple[int, str]:
    """``(count, ordered SHA-256)`` over one collection, read in pages.

    The digest is byte-identical to the one :func:`check_document_parity`
    computes over the same documents — the canonical serialization lives
    here and only here; callers (the carry tool's progress journal, the
    parity gate) reuse it instead of growing a second canonical form.
    """
    h = hashlib.sha256()
    count = 0
    after: str | None = None
    while True:
        docs = store.find(collection, include_expired=True,
                          after_id=after, limit=page)
        if not docs:
            break
        for doc in docs:
            h.update(_canonical(doc).encode("utf-8"))
            h.update(b"\n")
        count += len(docs)
        after = docs[-1]["_id"]
        if len(docs) < page:
            break
    return count, h.hexdigest()


#: Durable scheduler job name for the shadow-parity housekeeping beat.
SHADOW_PARITY_JOB_NAME = "storage-shadow-parity"

_DEFAULT_PARITY_INTERVAL_SECONDS = 3600.0


def _parity_interval_seconds(context: Any) -> float:
    settings = getattr(context, "settings", None)
    storage = getattr(settings, "storage", None)
    value = getattr(storage, "shadow_parity_interval_seconds", None)
    try:
        return float(value) if value is not None else _DEFAULT_PARITY_INTERVAL_SECONDS
    except (TypeError, ValueError):
        return _DEFAULT_PARITY_INTERVAL_SECONDS


def ensure_shadow_parity_job(
    context: Any,
    *,
    scheduler_factory: Any = None,
) -> dict[str, Any]:
    """Register the single durable ``storage-shadow-parity`` job (idempotent).

    The beat's payload is the ``db_shadow_parity`` tool, which runs
    :func:`run_kv_parity_housekeeping` — itself inert (returns ``None``)
    whenever the install is not dual-writing in shadow mode, so the job is
    safe to register unconditionally and costs one no-op tool call per
    interval on non-shadow installs. Spec ``every <Ns>`` from
    ``storage.shadow_parity_interval_seconds`` (default 3600). Safe to
    call on every boot; a changed interval replaces the job. Never raises.

    ``scheduler_factory`` builds the scheduler (``Scheduler(context)``,
    injected — storage is L2 and may not import the scheduler). None means
    the job is not scheduled, reported honestly.
    """
    try:
        if scheduler_factory is None:
            return {"name": SHADOW_PARITY_JOB_NAME,
                    "scheduled": False,
                    "reason": "no scheduler factory injected"}
        interval = _parity_interval_seconds(context)
        if interval <= 0:
            return {"name": SHADOW_PARITY_JOB_NAME,
                    "scheduled": False, "reason": "disabled"}
        spec = f"every {int(interval)}s"
        sched = scheduler_factory(context)
        try:
            have = [j for j in sched.list_jobs()
                    if j.get("name") == SHADOW_PARITY_JOB_NAME]
        except Exception:  # noqa: BLE001 — scheduler table may not exist yet
            have = []
        if have and have[0].get("spec") == spec:
            return {"name": SHADOW_PARITY_JOB_NAME,
                    "already_scheduled": True,
                    "job_id": have[0].get("id")}
        for job in have:  # stale interval → replace
            with contextlib.suppress(Exception):
                sched.remove(job["id"])
        job = sched.add(SHADOW_PARITY_JOB_NAME, spec, "tool",
                        {"tool": "db_shadow_parity", "args": {}})
        _log.info("scheduled storage-shadow-parity job: %s", spec)
        return {"name": SHADOW_PARITY_JOB_NAME, "scheduled": True,
                "job_id": job.get("id"), "spec": spec}
    except Exception as exc:  # noqa: BLE001 — optional wiring never breaks boot
        _log.warning("shadow parity job not registered: %s", exc)
        return {"name": SHADOW_PARITY_JOB_NAME, "scheduled": False,
                "reason": str(exc)[:200]}


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
