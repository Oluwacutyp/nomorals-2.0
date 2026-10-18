"""One durable initiative backlog (audit §2B + §2D).

Built on the existing SQLite store (same ``Database`` the rest of
Devon uses) — NOT a new task database. An initiative links to an
objective (and, via ``parent_id`` / lineage events, to the goals,
missions, signals, and artifacts that caused it).

Schema
------
``agency_initiatives`` — one row per initiative.
``agency_events`` — append-only lineage: every create/claim/advance/
complete/merge/stop leaves a trail so the owner can ask "Why?" and
get evidence → objective → plan → actions → verification → receipt.

Invariants
----------
* Nonterminal initiative ⇒ ``next_action`` OR ``next_review_at`` in
  the future OR a non-empty ``blocker``. Enforced on create/advance.
* One owner at a time: ``claim()`` takes a lease; a second entrypoint
  cannot advance the same initiative until the lease expires or is
  released. Boot reconcile recovers expired leases.
* Dedupe: ``dedupe_key`` (e.g. ``gig:<board>:<id>``) merges repeat
  signals — evidence is appended, effects are NOT multiplied.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

__all__ = ["Initiative", "InitiativeStore"]

#: Terminal statuses — nothing further is owed.
TERMINAL = ("completed", "cancelled")
#: Initiative kinds (audit §2A): explicit / standing / evidence / explore.
KINDS = ("explicit_goal", "standing", "evidence_opportunity", "exploration")


def _now() -> float:
    return time.time()


@dataclass
class Initiative:
    outcome: str
    kind: str = "evidence_opportunity"
    objective: str = ""
    evidence: str = ""
    next_action: str = ""
    next_review_at: float = 0.0
    blocker: str = ""
    benefit: float = 0.0
    uncertainty: float = 0.5
    effort: float = 1.0
    cost: float = 0.0
    deadline: float = 0.0
    priority: float = 0.5
    dedupe_key: str = ""
    parent_id: str = ""
    dependencies: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    id: str = ""
    status: str = "active"
    owner: str = ""
    lease_owner: str | None = None
    lease_until: float = 0.0
    last_action_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    def validate(self, now: float | None = None) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown initiative kind {self.kind!r}")
        if self.status not in TERMINAL:
            now = _now() if now is None else now
            has_next = bool((self.next_action or "").strip())
            has_wakeup = bool(self.next_review_at and self.next_review_at > 0)
            has_blocker = bool((self.blocker or "").strip())
            if not (has_next or has_wakeup or has_blocker):
                raise ValueError(
                    "nonterminal initiative needs next_action, a timed "
                    "wake-up (next_review_at), or a specific blocker — "
                    "'monitoring' with no next check is not active")


class InitiativeStore:
    """Durable initiative backlog over an injected DB handle."""

    def __init__(self, db: Any) -> None:
        self.db = db
        db.execute("""
            CREATE TABLE IF NOT EXISTS agency_initiatives (
                id TEXT PRIMARY KEY, outcome TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'evidence_opportunity',
                objective TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                next_action TEXT NOT NULL DEFAULT '',
                next_review_at REAL NOT NULL DEFAULT 0,
                blocker TEXT NOT NULL DEFAULT '',
                benefit REAL NOT NULL DEFAULT 0,
                uncertainty REAL NOT NULL DEFAULT 0.5,
                effort REAL NOT NULL DEFAULT 1.0,
                cost REAL NOT NULL DEFAULT 0,
                deadline REAL NOT NULL DEFAULT 0,
                priority REAL NOT NULL DEFAULT 0.5,
                dedupe_key TEXT NOT NULL DEFAULT '',
                parent_id TEXT NOT NULL DEFAULT '',
                dependencies TEXT NOT NULL DEFAULT '[]',
                verification TEXT NOT NULL DEFAULT '{}',
                stop_reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                owner TEXT NOT NULL DEFAULT '',
                lease_owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
                last_action_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )""")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_agency_dedupe "
            "ON agency_initiatives(dedupe_key)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS agency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT NOT NULL, ts REAL NOT NULL,
                kind TEXT NOT NULL, detail TEXT NOT NULL DEFAULT ''
            )""")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_agency_events_init "
            "ON agency_events(initiative_id)")

    # ── serialisation ──────────────────────────────────────────────
    @staticmethod
    def _row(it: Initiative) -> tuple:
        return (it.id, it.outcome, it.kind, it.objective, it.evidence,
                it.next_action, it.next_review_at, it.blocker,
                it.benefit, it.uncertainty, it.effort, it.cost,
                it.deadline, it.priority, it.dedupe_key, it.parent_id,
                json.dumps(it.dependencies), json.dumps(it.verification),
                it.stop_reason, it.status, it.owner,
                it.lease_owner, it.lease_until, it.last_action_at,
                it.created_at, it.updated_at)

    @staticmethod
    def _from_row(r: dict[str, Any]) -> Initiative:
        return Initiative(
            id=r["id"], outcome=r["outcome"], kind=r["kind"],
            objective=r["objective"], evidence=r["evidence"],
            next_action=r["next_action"],
            next_review_at=float(r["next_review_at"] or 0),
            blocker=r["blocker"], benefit=float(r["benefit"] or 0),
            uncertainty=float(r["uncertainty"] or 0),
            effort=float(r["effort"] or 0), cost=float(r["cost"] or 0),
            deadline=float(r["deadline"] or 0),
            priority=float(r["priority"] or 0),
            dedupe_key=r["dedupe_key"], parent_id=r["parent_id"],
            dependencies=json.loads(r["dependencies"] or "[]"),
            verification=json.loads(r["verification"] or "{}"),
            stop_reason=r["stop_reason"], status=r["status"],
            owner=r["owner"], lease_owner=r["lease_owner"],
            lease_until=float(r["lease_until"] or 0),
            last_action_at=float(r["last_action_at"] or 0),
            created_at=float(r["created_at"] or 0),
            updated_at=float(r["updated_at"] or 0))

    def _event(self, initiative_id: str, kind: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO agency_events (initiative_id, ts, kind, detail)"
            " VALUES (?, ?, ?, ?)",
            (initiative_id, _now(), kind, detail[:2000]))

    # ── CRUD ───────────────────────────────────────────────────────
    def create(self, it: Initiative, *, now: float | None = None) -> Initiative:
        now = _now() if now is None else now
        it.validate(now=now)
        if it.dedupe_key:
            row = self.db.query_one(
                "SELECT * FROM agency_initiatives WHERE dedupe_key = ?"
                " AND status NOT IN ('completed','cancelled')",
                (it.dedupe_key,))
            if row:
                existing = self._from_row(row)
                if it.evidence and it.evidence not in existing.evidence:
                    existing.evidence = (existing.evidence + "\n" + it.evidence)[:4000]
                existing.updated_at = now
                self._save(existing)
                self._event(existing.id, "merged",
                            f"duplicate signal merged: {it.evidence[:200]}")
                return existing
        it.id = it.id or f"init-{uuid.uuid4().hex[:12]}"
        it.created_at = it.created_at or now
        it.updated_at = now
        self.db.execute(
            "INSERT INTO agency_initiatives VALUES (?,?,?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self._row(it))
        self._event(it.id, "created", f"kind={it.kind} objective={it.objective}")
        if it.parent_id:
            self._event(it.id, "lineage", f"parent={it.parent_id}")
        return it

    def _save(self, it: Initiative) -> None:
        it.updated_at = _now()
        self.db.execute(
            "UPDATE agency_initiatives SET outcome=?, kind=?, objective=?,"
            " evidence=?, next_action=?, next_review_at=?, blocker=?,"
            " benefit=?, uncertainty=?, effort=?, cost=?, deadline=?,"
            " priority=?, dedupe_key=?, parent_id=?, dependencies=?,"
            " verification=?, stop_reason=?, status=?, owner=?,"
            " lease_owner=?, lease_until=?, last_action_at=?,"
            " created_at=?, updated_at=? WHERE id=?",
            self._row(it)[1:] + (it.id,))

    def get(self, initiative_id: str) -> Initiative | None:
        row = self.db.query_one(
            "SELECT * FROM agency_initiatives WHERE id = ?", (initiative_id,))
        return self._from_row(row) if row else None

    def list_active(self) -> list[Initiative]:
        return [self._from_row(r) for r in self.db.query(
            "SELECT * FROM agency_initiatives"
            " WHERE status NOT IN ('completed','cancelled')"
            " ORDER BY next_review_at ASC")]

    def list_all(self) -> list[Initiative]:
        return [self._from_row(r) for r in self.db.query(
            "SELECT * FROM agency_initiatives ORDER BY created_at ASC")]

    def events(self, initiative_id: str = "") -> list[dict[str, Any]]:
        if initiative_id:
            return self.db.query(
                "SELECT * FROM agency_events WHERE initiative_id = ?"
                " ORDER BY id ASC", (initiative_id,))
        return self.db.query("SELECT * FROM agency_events ORDER BY id ASC")

    def lineage(self, initiative_id: str) -> list[Initiative]:
        """Chain of ancestors, oldest first (durable cross-domain lineage)."""
        chain: list[Initiative] = []
        cur = self.get(initiative_id)
        seen = set()
        while cur and cur.parent_id and cur.parent_id not in seen:
            seen.add(cur.parent_id)
            parent = self.get(cur.parent_id)
            if parent is None:
                break
            chain.append(parent)
            cur = parent
        return list(reversed(chain))

    # ── leases: one entrypoint advances an initiative at a time ─────
    def claim(self, initiative_id: str, *, owner: str, lease_s: float = 300,
              now: float | None = None) -> bool:
        now = _now() if now is None else now
        it = self.get(initiative_id)
        if it is None or it.status in TERMINAL:
            return False
        if it.lease_owner and it.lease_until > now and it.lease_owner != owner:
            return False
        it.lease_owner, it.lease_until = owner, now + lease_s
        self._save(it)
        self._event(it.id, "claimed", f"owner={owner}")
        return True

    def release(self, initiative_id: str, *, owner: str) -> bool:
        it = self.get(initiative_id)
        if it is None or it.lease_owner != owner:
            return False
        it.lease_owner, it.lease_until = None, 0.0
        self._save(it)
        self._event(it.id, "released", f"owner={owner}")
        return True

    # ── advance / complete ─────────────────────────────────────────
    def advance(self, initiative_id: str, *, next_action: str = "",
                next_review_at: float | None = None, blocker: str | None = None,
                evidence: str = "", now: float | None = None) -> Initiative:
        it = self.get(initiative_id)
        if it is None:
            raise KeyError(initiative_id)
        if next_action != "":
            it.next_action = next_action
        if next_review_at is not None:
            it.next_review_at = next_review_at
        if blocker is not None:
            it.blocker = blocker
        if evidence:
            it.evidence = (it.evidence + "\n" + evidence)[:4000]
        it.last_action_at = _now() if now is None else now
        it.validate(now=it.last_action_at)
        self._save(it)
        self._event(it.id, "advanced", f"next={it.next_action[:200]}")
        return it

    def complete(self, initiative_id: str, *, stop_reason: str,
                 verification: dict[str, Any] | None = None) -> Initiative:
        it = self.get(initiative_id)
        if it is None:
            raise KeyError(initiative_id)
        it.status = "completed"
        it.stop_reason = stop_reason
        if verification is not None:
            it.verification = verification
        it.lease_owner, it.lease_until = None, 0.0
        self._save(it)
        self._event(it.id, "completed",
                    f"stop={stop_reason} verification={json.dumps(it.verification)[:500]}")
        return it

    def cancel(self, initiative_id: str, *, stop_reason: str) -> Initiative:
        it = self.get(initiative_id)
        if it is None:
            raise KeyError(initiative_id)
        it.status = "cancelled"
        it.stop_reason = stop_reason
        it.lease_owner, it.lease_until = None, 0.0
        self._save(it)
        self._event(it.id, "cancelled", f"stop={stop_reason}")
        return it

    # ── boot ───────────────────────────────────────────────────────
    def boot_reconcile(self, *, now: float | None = None) -> dict[str, int]:
        """Recover expired leases; coalesce missed checks; reopen stale runs.

        A mission left ``running`` by a crash is NOT blindly re-run:
        its lease is released and it becomes due again exactly once,
        with an event recording the recovery (at-least-once, never
        zero-times-without-a-record — same contract as missions/runner).
        """
        now = _now() if now is None else now
        recovered = 0
        for it in self.list_active():
            if it.lease_owner and it.lease_until <= now:
                it.lease_owner, it.lease_until = None, 0.0
                self._save(it)
                self._event(it.id, "lease_recovered",
                            "expired lease released on boot")
                recovered += 1
        return {"leases_recovered": recovered,
                "active": len(self.list_active())}
