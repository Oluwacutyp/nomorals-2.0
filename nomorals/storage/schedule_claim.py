"""Seam-backed claim book for the scheduler's coordination tables.

Mongo program P4. ``schedule_jobs`` / ``schedule_runs`` were excluded
from the P2 carry as coordination residue: their data is not the
point, their *claim* is. This module ports the claim semantics onto
the document seam, in carry-v1 shape, with the same fencing law as
the work queue (Batch B / F02):

* a claim is ONE compare-and-swap on the job document
  (:meth:`RowTable.cas_row`) whose expected fields are the job's
  ``next_run`` + claim fields exactly as read — atomic on both seam
  rungs, so of N competing claimers exactly one wins the firing;
* completion is fenced by the claim token: a stale claimant's
  completion is refused and mutates nothing;
* an expired claim (``claim_until`` in the past) is reclaimable by
  the next claimer, which is how a crashed claimant's firing is
  picked back up — the lease pattern, not a lock that can wedge;
* no multi-document transaction is used or needed (the seam
  advertises ``multi_doc_transactions=False`` on Mongo): the job
  flip lands first, then the ``schedule_runs`` history row is
  appended — the same write order the native scheduler uses (its
  job UPDATE and its run INSERT are separate transactions too).

What this is NOT: the live scheduler. ``agents/scheduler.py``'s
``Scheduler`` keeps its native SQL path in P4 — its claim is
single-process by design (an in-process active set plus a per-tick
pool), and six other modules read and write these tables natively,
so a claim-only seam wiring here would fork the record between two
stores. The claim primitive lives on the seam, proven by the P4
concurrency suite on both rungs, ready for the scheduler domain
port; the live-path disposition is recorded in
:data:`nomorals.storage.carry.PORT_RESIDUE`. The work queue — the
organ other workers genuinely contend for — is fully seam-backed
in :mod:`nomorals.storage.queue`.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.ids import new_id
from .rowstore import RowTable, column_defaults, mapping_for

__all__ = ["ScheduleClaimStore", "CLAIM_COLUMNS"]

#: Runtime claim fields carried on the job document beside the
#: table's own columns (the queueing_lock/lease_token precedent:
#: coordination fields are added by the coordination layer, never by
#: the frozen migrations).
CLAIM_COLUMNS = ("claim_token", "claim_owner", "claim_until")

_CLAIM_DEFAULTS: dict[str, Any] = {
    "claim_token": "", "claim_owner": "", "claim_until": 0.0}


class ScheduleClaimStore:
    """Claims over ``schedule_jobs`` documents on the attached seam.

    Construct with the central ``Database`` after boot attachment —
    the same pickup rule as every P3 port: when no store is attached
    (the SQLite default) :meth:`attached` returns None and there is
    nothing to claim through (the native scheduler owns the tables).
    """

    JOBS_TABLE = "schedule_jobs"
    RUNS_TABLE = "schedule_runs"

    def __init__(self, db: Any, store: Any) -> None:
        self._db = db
        self._store = store
        # RowTables are cached on the Database handle (the
        # attached_rows pattern): construction reads PRAGMA schema,
        # which must happen once — claimers in worker threads share
        # the built tables instead of re-deriving them per thread.
        cache = getattr(db, "_schedule_claim_tables", None)
        if cache is None:
            cache = {}
            try:
                db._schedule_claim_tables = cache
            except Exception:  # noqa: BLE001 - a handle that refuses attrs
                cache = {}
        jobs_key = (self.JOBS_TABLE, id(store))
        if jobs_key not in cache:
            cache[jobs_key] = RowTable(
                store,
                mapping_for(db, self.JOBS_TABLE,
                            extra_columns=CLAIM_COLUMNS),
                defaults={**column_defaults(db, self.JOBS_TABLE),
                          **_CLAIM_DEFAULTS})
        runs_key = (self.RUNS_TABLE, id(store))
        if runs_key not in cache:
            cache[runs_key] = RowTable(
                store,
                mapping_for(db, self.RUNS_TABLE),
                defaults=column_defaults(db, self.RUNS_TABLE))
        self._jobs = cache[jobs_key]
        self._runs = cache[runs_key]

    @classmethod
    def attached(cls, db: Any) -> ScheduleClaimStore | None:
        """The claim store for ``db``'s attached seam, or None."""
        store = getattr(db, "document_store", None)
        if store is None:
            return None
        return cls(db, store)

    # ── job records ──────────────────────────────────────────────────────
    def put_job(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace one job row (registration / seeding)."""
        return self._jobs.put(row)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def due_jobs(self, now: float | None = None, *,
                 limit: int | None = None) -> list[dict[str, Any]]:
        """Enabled jobs whose firing is due AND whose claim is free
        (unclaimed, or the claim's lease has expired), in native
        dispatch order (``next_run`` ascending)."""
        now = time.time() if now is None else now
        due = []
        for row in self._jobs.find({"enabled": 1}):
            next_run = row.get("next_run")
            if next_run is None or float(next_run) > now:
                continue
            if not self._claim_free(row, now):
                continue
            due.append(row)
        due.sort(key=lambda r: float(r["next_run"]))
        return due[:limit] if limit is not None else due

    @staticmethod
    def _claim_free(row: dict[str, Any], now: float) -> bool:
        token = row.get("claim_token") or ""
        if not token:
            return True
        return float(row.get("claim_until") or 0.0) <= now

    # ── the claim ────────────────────────────────────────────────────────
    def claim(self, job_id: str, worker: str, *,
              lease_seconds: float = 300.0,
              now: float | None = None) -> str | None:
        """Claim one due firing of ``job_id`` for ``worker``.

        Returns the claim token on success, None when the job is not
        claimable (missing, disabled, not due, or already claimed by
        a live lease — including the race where another claimer's cas
        landed first).
        """
        now = time.time() if now is None else now
        row = self._jobs.get(job_id)
        if row is None or not int(row.get("enabled") or 0):
            return None
        next_run = row.get("next_run")
        if next_run is None or float(next_run) > now:
            return None
        if not self._claim_free(row, now):
            return None
        token = new_id()
        new_row = {
            **row,
            "claim_token": token,
            "claim_owner": worker,
            "claim_until": now + lease_seconds,
        }
        expected = {
            "next_run": row.get("next_run"),
            "claim_token": row.get("claim_token"),
            "claim_until": row.get("claim_until"),
        }
        if not self._jobs.cas_row(job_id, expected, new_row):
            return None
        return token

    def complete(self, job_id: str, token: str, updates: dict[str, Any], *,
                 run: dict[str, Any] | None = None) -> bool:
        """Land the post-run job state, fenced by the claim token.

        ``updates`` is the caller's post-run row state (``last_run``,
        ``next_run``, ``enabled``, ``retry_count`` …). A stale token
        — the claim expired and another claimer took the firing —
        returns False and mutates nothing. When ``run`` is given, the
        ``schedule_runs`` history row is appended after the flip
        (the native write order).
        """
        if not token:
            return False
        row = self._jobs.get(job_id)
        if row is None or (row.get("claim_token") or "") != token:
            return False
        new_row = {
            **row, **updates,
            "claim_token": "", "claim_owner": "", "claim_until": 0.0,
        }
        if not self._jobs.cas_row(
                job_id,
                {"claim_token": row.get("claim_token"),
                 "next_run": row.get("next_run")},
                new_row):
            return False
        if run is not None:
            self._runs.put(run)
        return True

    def release(self, job_id: str, token: str) -> bool:
        """Give a claim back without completing (the firing stays due
        for the next claimer). Fenced by the token, like completion."""
        if not token:
            return False
        row = self._jobs.get(job_id)
        if row is None or (row.get("claim_token") or "") != token:
            return False
        new_row = {
            **row, "claim_token": "", "claim_owner": "", "claim_until": 0.0,
        }
        return self._jobs.cas_row(
            job_id, {"claim_token": row.get("claim_token")}, new_row)

    def runs_for(self, job_id: str) -> list[dict[str, Any]]:
        """The recorded run history for one job, oldest first."""
        runs = self._runs.find({"job_id": job_id})
        runs.sort(key=lambda r: float(r.get("started_at") or 0.0))
        return runs
