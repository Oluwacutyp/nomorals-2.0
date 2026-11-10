"""Accountable delegation (L5) — work contracts, a claim ledger, fencing.

Doc 6's governing layer for delegated work, built on the organs that
already exist: persistence is runtime-created tables on the shared
``Database`` (the Batch B pattern — ``storage/migrations.py`` is
frozen), and completion is recorded through Batch B's
``missions.idempotency`` claim machinery so recording a completion
twice replays the receipt instead of double-recording.

The rules that make delegation *accountable*, enforced here in code:

* **Work contracts.** Every delegated job states its goal, deliverable,
  scope, required evidence kinds, step budget, and the named holders of
  the six contract roles: worker, witness, skeptic, verifier,
  integration owner, recovery supervisor (arbiter resolves disputes).
* **Independence.** A contract's verifier is never its worker, and a
  claim's validator is never its producer. The store raises rather
  than recording a self-verified claim.
* **Claim ledger.** Every evidence item, challenge, resolution, verdict
  and completion is a durable ledger row: who claimed what, on what
  evidence, at what revision, verified by whom, with a verdict from
  doc 6's vocabulary (supported / contradicted / partial / stale /
  unverified / unavailable / uncertain).
* **Skeptic pass.** A consequential contract cannot complete while any
  skeptic challenge stands unresolved or any claim stands contradicted.
* **Revision-aware fenced ownership.** An artifact is owned at a
  revision; a write must name the current revision as its base or it
  is refused (:class:`StaleRevisionError`). When a worker fails and the
  recovery supervisor re-plans, the contract's **generation** bumps —
  late evidence and late writes carrying the old generation are fenced
  (:class:`StaleGenerationError`), so a superseded worker can never
  overwrite its replacement's accepted result.
* **Mission gate.** ``MissionRunner`` consults :func:`open_contracts`
  at its terminal point: a mission with non-terminal contracts cannot
  finish DONE.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VERDICTS",
    "CONTRACT_ROLES",
    "TERMINAL_STATUSES",
    "WorkContract",
    "Claim",
    "CompletionDecision",
    "DelegationStore",
    "DelegationError",
    "IndependenceError",
    "StaleRevisionError",
    "StaleGenerationError",
    "OwnershipError",
    "contract_for_step",
]

#: Doc 6 claim verdicts.
VERDICTS = ("supported", "contradicted", "partial", "stale",
            "unverified", "unavailable", "uncertain")

#: The six accountable roles every contract names (arbiter included).
CONTRACT_ROLES = ("worker", "witness", "skeptic", "verifier",
                  "integration_owner", "recovery_supervisor", "arbiter")

TERMINAL_STATUSES = ("completed", "waived", "refused", "cancelled")

_DDL = """
CREATE TABLE IF NOT EXISTS delegation_contracts (
  id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  goal TEXT NOT NULL DEFAULT '',
  deliverable TEXT NOT NULL DEFAULT '',
  worker TEXT NOT NULL DEFAULT '',
  witness TEXT NOT NULL DEFAULT '',
  skeptic TEXT NOT NULL DEFAULT '',
  verifier TEXT NOT NULL DEFAULT '',
  integration_owner TEXT NOT NULL DEFAULT '',
  recovery_supervisor TEXT NOT NULL DEFAULT '',
  arbiter TEXT NOT NULL DEFAULT '',
  specialist_class TEXT NOT NULL DEFAULT '',
  consequential INTEGER NOT NULL DEFAULT 0,
  required_evidence TEXT NOT NULL DEFAULT '[]',
  scope TEXT NOT NULL DEFAULT '[]',
  base_revision TEXT NOT NULL DEFAULT '',
  step_budget INTEGER NOT NULL DEFAULT 0,
  steps_used INTEGER NOT NULL DEFAULT 0,
  deadline REAL NOT NULL DEFAULT 0,
  generation INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  completed_at REAL NOT NULL DEFAULT 0,
  refusal TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS claim_ledger (
  id TEXT PRIMARY KEY,
  contract_id TEXT NOT NULL DEFAULT '',
  mission_id TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'evidence',
  claim TEXT NOT NULL DEFAULT '',
  scope TEXT NOT NULL DEFAULT '',
  producer TEXT NOT NULL DEFAULT '',
  validator TEXT NOT NULL DEFAULT '',
  verdict TEXT NOT NULL DEFAULT 'unverified',
  evidence_ref TEXT NOT NULL DEFAULT '',
  revision TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS artifact_ownership (
  artifact_key TEXT PRIMARY KEY,
  contract_id TEXT NOT NULL DEFAULT '',
  revision INTEGER NOT NULL DEFAULT 0,
  generation INTEGER NOT NULL DEFAULT 1,
  payload_ref TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_contracts_mission
  ON delegation_contracts(mission_id);
CREATE INDEX IF NOT EXISTS idx_claims_contract
  ON claim_ledger(contract_id);
"""


class DelegationError(Exception):
    """Base for delegation refusals."""


class IndependenceError(DelegationError):
    """A verifier/validator would be judging their own work."""


class StaleRevisionError(DelegationError):
    """A write named a base revision that is no longer current."""


class StaleGenerationError(DelegationError):
    """Work arrived from a superseded contract generation."""


class OwnershipError(DelegationError):
    """An artifact write by a contract that does not own the artifact."""


@dataclass
class WorkContract:
    id: str
    mission_id: str = ""
    title: str = ""
    goal: str = ""
    deliverable: str = ""
    worker: str = ""
    witness: str = ""
    skeptic: str = ""
    verifier: str = ""
    integration_owner: str = ""
    recovery_supervisor: str = ""
    arbiter: str = ""
    specialist_class: str = ""
    consequential: bool = False
    required_evidence: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()
    base_revision: str = ""
    step_budget: int = 0
    steps_used: int = 0
    deadline: float = 0.0
    generation: int = 1
    status: str = "open"
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float = 0.0
    refusal: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def role_holder(self, role: str) -> str:
        return str(getattr(self, role, "") or "")


@dataclass
class Claim:
    id: str
    contract_id: str = ""
    mission_id: str = ""
    kind: str = "evidence"       # evidence|challenge|resolution|verdict|
    #                              completion|replan|failure|signoff|
    #                              cancellation
    claim: str = ""
    scope: str = ""
    producer: str = ""
    validator: str = ""
    verdict: str = "unverified"
    evidence_ref: str = ""
    revision: str = ""
    generation: int = 1
    created_at: float = 0.0


@dataclass
class CompletionDecision:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    contract_id: str = ""
    replayed: bool = False


def _row_contract(row: Mapping[str, Any]) -> WorkContract:
    return WorkContract(
        id=row["id"], mission_id=row["mission_id"], title=row["title"],
        goal=row["goal"], deliverable=row["deliverable"],
        worker=row["worker"], witness=row["witness"],
        skeptic=row["skeptic"], verifier=row["verifier"],
        integration_owner=row["integration_owner"],
        recovery_supervisor=row["recovery_supervisor"],
        arbiter=row["arbiter"], specialist_class=row["specialist_class"],
        consequential=bool(row["consequential"]),
        required_evidence=tuple(json.loads(row["required_evidence"]
                                           or "[]")),
        scope=tuple(json.loads(row["scope"] or "[]")),
        base_revision=row["base_revision"], step_budget=row["step_budget"],
        steps_used=row["steps_used"], deadline=row["deadline"],
        generation=row["generation"], status=row["status"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        completed_at=row["completed_at"], refusal=row["refusal"])


def _row_claim(row: Mapping[str, Any]) -> Claim:
    return Claim(
        id=row["id"], contract_id=row["contract_id"],
        mission_id=row["mission_id"], kind=row["kind"], claim=row["claim"],
        scope=row["scope"], producer=row["producer"],
        validator=row["validator"], verdict=row["verdict"],
        evidence_ref=row["evidence_ref"], revision=row["revision"],
        generation=row["generation"], created_at=row["created_at"])


class DelegationStore:
    """Contracts + claim ledger + artifact ownership over a Database."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_DDL)

    # -- contracts ------------------------------------------------------

    def create_contract(self, *, goal: str, worker: str, verifier: str,
                        mission_id: str = "", title: str = "",
                        deliverable: str = "", witness: str = "",
                        skeptic: str = "", integration_owner: str = "",
                        recovery_supervisor: str = "", arbiter: str = "",
                        specialist_class: str = "",
                        consequential: bool = False,
                        required_evidence: Sequence[str] = (),
                        scope: Sequence[str] = (),
                        base_revision: str = "", step_budget: int = 0,
                        deadline: float = 0.0) -> WorkContract:
        if not worker or not verifier:
            raise DelegationError(
                "a contract names both a worker and a verifier")
        if worker == verifier:
            raise IndependenceError(
                f"verifier {verifier!r} is the worker — verification "
                "must be independent of the work")
        now = time.time()
        contract = WorkContract(
            id=f"dc-{uuid.uuid4().hex[:12]}", mission_id=mission_id,
            title=title or goal[:60], goal=goal, deliverable=deliverable,
            worker=worker, witness=witness, skeptic=skeptic,
            verifier=verifier, integration_owner=integration_owner,
            recovery_supervisor=recovery_supervisor, arbiter=arbiter,
            specialist_class=specialist_class,
            consequential=bool(consequential),
            required_evidence=tuple(required_evidence),
            scope=tuple(scope), base_revision=base_revision,
            step_budget=int(step_budget or 0), deadline=float(deadline or 0),
            created_at=now, updated_at=now)
        self.db.execute(
            "INSERT INTO delegation_contracts (id, mission_id, title,"
            " goal, deliverable, worker, witness, skeptic, verifier,"
            " integration_owner, recovery_supervisor, arbiter,"
            " specialist_class, consequential, required_evidence, scope,"
            " base_revision, step_budget, steps_used, deadline,"
            " generation, status, created_at, updated_at, completed_at,"
            " refusal) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?)",
            (contract.id, contract.mission_id, contract.title,
             contract.goal, contract.deliverable, contract.worker,
             contract.witness, contract.skeptic, contract.verifier,
             contract.integration_owner, contract.recovery_supervisor,
             contract.arbiter, contract.specialist_class,
             int(contract.consequential),
             json.dumps(list(contract.required_evidence)),
             json.dumps(list(contract.scope)), contract.base_revision,
             contract.step_budget, contract.steps_used, contract.deadline,
             contract.generation, contract.status, contract.created_at,
             contract.updated_at, contract.completed_at, contract.refusal))
        return contract

    def get_contract(self, contract_id: str) -> WorkContract | None:
        row = self.db.query_one(
            "SELECT * FROM delegation_contracts WHERE id = ?",
            (contract_id,))
        return _row_contract(row) if row else None

    def contracts_for_mission(self, mission_id: str) -> list[WorkContract]:
        rows = self.db.query(
            "SELECT * FROM delegation_contracts WHERE mission_id = ?"
            " ORDER BY created_at", (mission_id,))
        return [_row_contract(r) for r in rows]

    def recent_contracts(self, limit: int = 20) -> list[WorkContract]:
        rows = self.db.query(
            "SELECT * FROM delegation_contracts ORDER BY created_at DESC"
            " LIMIT ?", (int(limit),))
        return [_row_contract(r) for r in rows]

    def open_contracts(self, mission_id: str = "") -> list[WorkContract]:
        if mission_id:
            pool = self.contracts_for_mission(mission_id)
        else:
            pool = self.recent_contracts(limit=500)
        return [c for c in pool if not c.terminal]

    def _save_contract(self, contract: WorkContract) -> None:
        contract.updated_at = time.time()
        self.db.execute(
            "UPDATE delegation_contracts SET worker = ?, witness = ?,"
            " skeptic = ?, verifier = ?, integration_owner = ?,"
            " recovery_supervisor = ?, arbiter = ?, generation = ?,"
            " status = ?, steps_used = ?, updated_at = ?,"
            " completed_at = ?, refusal = ? WHERE id = ?",
            (contract.worker, contract.witness, contract.skeptic,
             contract.verifier, contract.integration_owner,
             contract.recovery_supervisor, contract.arbiter,
             contract.generation, contract.status, contract.steps_used,
             contract.updated_at, contract.completed_at, contract.refusal,
             contract.id))

    def _contract_or_raise(self, contract_id: str) -> WorkContract:
        contract = self.get_contract(contract_id)
        if contract is None:
            raise DelegationError(f"no contract {contract_id!r}")
        return contract

    def _check_generation(self, contract: WorkContract,
                          generation: int | None) -> None:
        if generation is not None and int(generation) != contract.generation:
            raise StaleGenerationError(
                f"contract {contract.id} is at generation "
                f"{contract.generation}; work arrived from generation "
                f"{generation} — the superseded worker's output is fenced")

    # -- claim ledger -----------------------------------------------------

    def _record(self, contract: WorkContract, *, kind: str, claim: str,
                producer: str, scope: str = "", validator: str = "",
                verdict: str = "unverified", evidence_ref: str = "",
                revision: str = "") -> Claim:
        entry = Claim(
            id=f"cl-{uuid.uuid4().hex[:12]}", contract_id=contract.id,
            mission_id=contract.mission_id, kind=kind, claim=claim,
            scope=scope, producer=producer, validator=validator,
            verdict=verdict, evidence_ref=evidence_ref, revision=revision,
            generation=contract.generation, created_at=time.time())
        self.db.execute(
            "INSERT INTO claim_ledger (id, contract_id, mission_id, kind,"
            " claim, scope, producer, validator, verdict, evidence_ref,"
            " revision, generation, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry.id, entry.contract_id, entry.mission_id, entry.kind,
             entry.claim, entry.scope, entry.producer, entry.validator,
             entry.verdict, entry.evidence_ref, entry.revision,
             entry.generation, entry.created_at))
        return entry

    def claims(self, contract_id: str = "", *, limit: int = 200
               ) -> list[Claim]:
        if contract_id:
            rows = self.db.query(
                "SELECT * FROM claim_ledger WHERE contract_id = ?"
                " ORDER BY created_at LIMIT ?",
                (contract_id, int(limit)))
        else:
            rows = self.db.query(
                "SELECT * FROM claim_ledger ORDER BY created_at DESC"
                " LIMIT ?", (int(limit),))
        return [_row_claim(r) for r in rows]

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self.db.query_one(
            "SELECT * FROM claim_ledger WHERE id = ?", (claim_id,))
        return _row_claim(row) if row else None

    def submit_evidence(self, contract_id: str, *, kind: str, claim: str,
                        producer: str, evidence_ref: str = "",
                        revision: str = "",
                        generation: int | None = None) -> Claim:
        """Witness-recorded evidence: a claim enters the ledger
        unverified; only an independent validator can move its verdict."""
        contract = self._contract_or_raise(contract_id)
        self._check_generation(contract, generation)
        return self._record(contract, kind="evidence", claim=claim,
                            producer=producer, scope=kind,
                            evidence_ref=evidence_ref, revision=revision)

    def verify_claim(self, claim_id: str, *, validator: str,
                     verdict: str) -> Claim:
        """Move a claim's verdict — never by its own producer."""
        if verdict not in VERDICTS:
            raise DelegationError(f"unknown verdict {verdict!r}")
        entry = self.get_claim(claim_id)
        if entry is None:
            raise DelegationError(f"no claim {claim_id!r}")
        if validator == entry.producer:
            raise IndependenceError(
                f"validator {validator!r} produced claim {claim_id} — "
                "a claim is never verified by its producer")
        self.db.execute(
            "UPDATE claim_ledger SET verdict = ?, validator = ?"
            " WHERE id = ?", (verdict, validator, claim_id))
        entry.verdict = verdict
        entry.validator = validator
        return entry

    def challenge(self, contract_id: str, *, challenger: str, claim: str,
                  target_claim_id: str = "") -> Claim:
        """The skeptic's instrument: a challenge stands in the ledger
        until a resolution answers it."""
        contract = self._contract_or_raise(contract_id)
        return self._record(contract, kind="challenge", claim=claim,
                            producer=challenger, scope=target_claim_id,
                            verdict="uncertain")

    def resolve_challenge(self, challenge_id: str, *, resolver: str,
                          verdict: str, note: str = "") -> Claim:
        """Answer a challenge.  ``supported`` = the challenged claim
        survived scrutiny; ``contradicted`` = it did not, and the
        target claim's verdict moves with it."""
        if verdict not in ("supported", "contradicted", "partial"):
            raise DelegationError(
                "a challenge resolves supported, contradicted or partial")
        challenge = self.get_claim(challenge_id)
        if challenge is None or challenge.kind != "challenge":
            raise DelegationError(f"no challenge {challenge_id!r}")
        contract = self._contract_or_raise(challenge.contract_id)
        resolution = self._record(
            contract, kind="resolution",
            claim=note or f"challenge {challenge_id} resolved {verdict}",
            producer=resolver, scope=challenge_id, verdict=verdict)
        self.db.execute(
            "UPDATE claim_ledger SET verdict = ? WHERE id = ?",
            (verdict, challenge_id))
        if verdict == "contradicted" and challenge.scope:
            self.db.execute(
                "UPDATE claim_ledger SET verdict = 'contradicted'"
                " WHERE id = ?", (challenge.scope,))
        return resolution

    def sign_off(self, contract_id: str, *, verifier: str,
                 note: str = "") -> Claim:
        """The verifier's completion signature — must be the contract's
        named verifier, and never its worker."""
        contract = self._contract_or_raise(contract_id)
        if verifier == contract.worker:
            raise IndependenceError(
                "the worker cannot sign off their own contract")
        if contract.verifier and verifier != contract.verifier:
            raise DelegationError(
                f"sign-off must come from the contract verifier "
                f"{contract.verifier!r}, not {verifier!r}")
        return self._record(contract, kind="signoff",
                            claim=note or "verifier sign-off",
                            producer=verifier, verdict="supported")

    # -- failure + recovery ---------------------------------------------

    def report_failure(self, contract_id: str, *, by: str, detail: str,
                       generation: int | None = None) -> WorkContract:
        """Record a worker failure.  The contract stays open; only a
        recovery-supervisor re-plan (which bumps the generation) or a
        completion can move it on."""
        contract = self._contract_or_raise(contract_id)
        self._check_generation(contract, generation)
        self._record(contract, kind="failure", claim=detail, producer=by,
                     verdict="supported")
        contract.status = "failed"
        self._save_contract(contract)
        return contract

    def replan(self, contract_id: str, *, supervisor: str,
               new_worker: str = "", note: str = "") -> WorkContract:
        """Recovery-supervisor re-plan: generation bumps, the contract
        reopens, and the old generation's late work is fenced."""
        contract = self._contract_or_raise(contract_id)
        if contract.recovery_supervisor \
                and supervisor != contract.recovery_supervisor:
            raise DelegationError(
                f"re-plan authority is the recovery supervisor "
                f"{contract.recovery_supervisor!r}, not {supervisor!r}")
        old_worker = contract.worker
        if new_worker:
            if new_worker == contract.verifier:
                raise IndependenceError(
                    "the replacement worker cannot be the verifier")
            contract.worker = new_worker
        contract.generation += 1
        contract.status = "open"
        self._save_contract(contract)
        self._record(contract, kind="replan",
                     claim=note or (f"re-planned: {old_worker} → "
                                    f"{contract.worker}"),
                     producer=supervisor, verdict="supported")
        return contract

    # -- fenced artifact ownership ----------------------------------------

    def claim_artifact(self, contract_id: str, artifact_key: str) -> dict:
        contract = self._contract_or_raise(contract_id)
        row = self.db.query_one(
            "SELECT * FROM artifact_ownership WHERE artifact_key = ?",
            (artifact_key,))
        if row is not None:
            owner = self.get_contract(row["contract_id"])
            if row["contract_id"] != contract_id and owner is not None \
                    and not owner.terminal:
                raise OwnershipError(
                    f"artifact {artifact_key!r} is owned by active "
                    f"contract {row['contract_id']}")
            if row["contract_id"] == contract_id:
                return dict(row)
        now = time.time()
        if row is None:
            self.db.execute(
                "INSERT INTO artifact_ownership (artifact_key,"
                " contract_id, revision, generation, payload_ref,"
                " updated_at) VALUES (?,?,0,?, '', ?)",
                (artifact_key, contract_id, contract.generation, now))
            return {"artifact_key": artifact_key,
                    "contract_id": contract_id, "revision": 0,
                    "generation": contract.generation, "payload_ref": "",
                    "updated_at": now}
        self.db.execute(
            "UPDATE artifact_ownership SET contract_id = ?,"
            " generation = ?, updated_at = ? WHERE artifact_key = ?",
            (contract_id, contract.generation, now, artifact_key))
        row = self.db.query_one(
            "SELECT * FROM artifact_ownership WHERE artifact_key = ?",
            (artifact_key,))
        return dict(row)

    def artifact(self, artifact_key: str) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM artifact_ownership WHERE artifact_key = ?",
            (artifact_key,))
        return dict(row) if row else None

    def write_artifact(self, contract_id: str, artifact_key: str, *,
                       base_revision: int, payload_ref: str,
                       generation: int | None = None) -> dict:
        """The fenced write: owner + current generation + exact base
        revision, or refusal.  Success bumps the revision by one."""
        contract = self._contract_or_raise(contract_id)
        self._check_generation(contract, generation)
        row = self.db.query_one(
            "SELECT * FROM artifact_ownership WHERE artifact_key = ?",
            (artifact_key,))
        if row is None or row["contract_id"] != contract_id:
            raise OwnershipError(
                f"contract {contract_id} does not own artifact "
                f"{artifact_key!r}")
        current = int(row["revision"])
        if int(base_revision) != current:
            raise StaleRevisionError(
                f"artifact {artifact_key!r} is at revision {current}; "
                f"the write named base revision {base_revision} — a "
                "superseded revision cannot overwrite the current one")
        new_revision = current + 1
        cursor = self.db.execute(
            "UPDATE artifact_ownership SET revision = ?, payload_ref = ?,"
            " generation = ?, updated_at = ? WHERE artifact_key = ?"
            " AND revision = ?",
            (new_revision, payload_ref, contract.generation, time.time(),
             artifact_key, current))
        if not cursor.rowcount:
            raise StaleRevisionError(
                f"artifact {artifact_key!r} moved under the write — "
                "re-read the current revision and retry")
        return {"artifact_key": artifact_key, "contract_id": contract_id,
                "revision": new_revision, "payload_ref": payload_ref,
                "generation": contract.generation}

    # -- completion ------------------------------------------------------

    def completion_blockers(self, contract: WorkContract) -> list[str]:
        """Why this contract cannot complete yet — the honest list."""
        reasons: list[str] = []
        entries = self.claims(contract.id, limit=1000)
        # The ledger's current position per evidence kind is its LATEST
        # claim: a contradicted claim that the worker repaired and
        # re-evidenced is superseded, not a permanent block.
        latest: dict[str, Claim] = {}
        for entry in entries:
            if entry.kind == "evidence":
                latest[entry.scope] = entry
        for kind in contract.required_evidence:
            current = latest.get(kind)
            if current is None:
                reasons.append(f"missing required evidence: {kind}")
            elif current.verdict != "supported":
                reasons.append(
                    f"evidence not verified as supported: {kind} "
                    f"(latest verdict: {current.verdict})")
        contradicted = [c for c in latest.values()
                        if c.verdict == "contradicted"
                        and c.scope not in contract.required_evidence]
        if contradicted:
            reasons.append(
                f"{len(contradicted)} evidence claim(s) stand "
                "contradicted")
        if not any(c.kind == "signoff" and c.verdict == "supported"
                   for c in entries):
            reasons.append("no verifier sign-off recorded")
        if contract.consequential:
            open_challenges = [
                c for c in entries if c.kind == "challenge"
                and c.verdict == "uncertain"]
            if open_challenges:
                reasons.append(
                    f"{len(open_challenges)} skeptic challenge(s) "
                    "unresolved on a consequential contract")
            if contract.skeptic and not any(
                    c.kind == "challenge" for c in entries):
                reasons.append(
                    "consequential contract has no skeptic challenge "
                    "on record — the skeptic pass has not happened")
        if contract.deadline and time.time() > contract.deadline \
                and contract.status != "completed":
            reasons.append("deadline passed without completion")
        return reasons

    def complete(self, contract_id: str, *, by: str,
                 idempotency: Any = None) -> CompletionDecision:
        """Record completion — gated, and idempotent via Batch B.

        With an ``IdempotencyStore`` the completion effect runs inside
        ``dedupe``: a second completion of the same contract generation
        replays the stored receipt and records nothing twice.
        """
        contract = self._contract_or_raise(contract_id)
        if contract.status == "completed":
            return CompletionDecision(ok=True, reasons=[],
                                      contract_id=contract_id,
                                      replayed=True)
        if contract.terminal:
            # A late completion arriving after the contract reached
            # a terminal state (cancelled / waived / refused) must
            # not move it: the terminal record stands, and the late
            # result is refused with the state named.
            return CompletionDecision(
                ok=False,
                reasons=[f"contract is {contract.status} (terminal) "
                         "— a late completion cannot move a "
                         "terminal contract"],
                contract_id=contract_id)
        blockers = self.completion_blockers(contract)
        if blockers:
            return CompletionDecision(ok=False, reasons=blockers,
                                      contract_id=contract_id)

        def _effect() -> dict:
            fresh = self._contract_or_raise(contract_id)
            fresh.status = "completed"
            fresh.completed_at = time.time()
            self._save_contract(fresh)
            self._record(fresh, kind="completion",
                         claim=f"contract completed by {by}",
                         producer=by, verdict="supported")
            return {"contract_id": contract_id, "recorded": True}

        if idempotency is not None:
            from ..missions.idempotency import dedupe

            result = dedupe(
                idempotency,
                f"delegation-complete:{contract_id}:{contract.generation}",
                _effect, owner=by, label="delegation completion")
            return CompletionDecision(
                ok=True, reasons=[], contract_id=contract_id,
                replayed=not bool(result.executed))
        _effect()
        return CompletionDecision(ok=True, reasons=[],
                                  contract_id=contract_id)

    def waive(self, contract_id: str, *, by: str, reason: str) -> WorkContract:
        """The arbiter's waiver: terminal, journaled, never silent."""
        contract = self._contract_or_raise(contract_id)
        if contract.arbiter and by != contract.arbiter:
            raise DelegationError(
                f"only the arbiter {contract.arbiter!r} may waive this "
                "contract")
        contract.status = "waived"
        contract.refusal = reason
        contract.completed_at = time.time()
        self._save_contract(contract)
        self._record(contract, kind="verdict", claim=f"waived: {reason}",
                     producer=by, verdict="supported")
        return contract

    def cancel(self, contract_id: str, *, by: str, reason: str = "",
               generation: int | None = None) -> WorkContract:
        """Cancel a live contract: the ``cancelled`` terminal.

        Fenced like every other write. A cancel naming a stale
        generation is refused (:class:`StaleGenerationError` —
        the caller is acting on a superseded contract), and a
        terminal contract refuses with its state named. The
        cancellation lands on the claim ledger as a ``cancellation``
        claim that counts what it leaves open.

        Claims disposition follows the ledger's standing rule: the
        ledger is append-only history and terminal transitions
        never rewrite claims (completion and waiver behave the
        same way), so open skeptic challenges and unverified
        evidence are NOT resolved or erased by the cancel — they
        remain exactly as recorded, named in the cancellation
        claim. The cancelled contract stops gating its mission
        because the mission gate reads non-terminal contracts
        only."""
        contract = self._contract_or_raise(contract_id)
        self._check_generation(contract, generation)
        if contract.terminal:
            raise DelegationError(
                f"contract {contract.id} is already "
                f"{contract.status} — a terminal contract cannot "
                "be cancelled")
        entries = self.claims(contract.id, limit=1000)
        open_challenges = [c for c in entries
                           if c.kind == "challenge"
                           and c.verdict == "uncertain"]
        unverified = [c for c in entries if c.kind == "evidence"
                      and c.verdict == "unverified"]
        contract.status = "cancelled"
        contract.completed_at = time.time()
        self._save_contract(contract)
        text = f"cancelled by {by}"
        if reason:
            text += f": {reason}"
        text += (f" — {len(open_challenges)} skeptic challenge(s) "
                 f"and {len(unverified)} unverified evidence "
                 "claim(s) remain on the ledger unresolved "
                 "(append-only: terminal transitions do not "
                 "rewrite claims; the cancelled contract no longer "
                 "gates its mission)")
        self._record(contract, kind="cancellation", claim=text,
                     producer=by, verdict="supported")
        return contract

    # -- cockpit ---------------------------------------------------------

    def summary(self, contract: WorkContract) -> dict[str, Any]:
        entries = self.claims(contract.id, limit=1000)
        verdicts: dict[str, int] = {}
        for entry in entries:
            verdicts[entry.verdict] = verdicts.get(entry.verdict, 0) + 1
        return {
            "id": contract.id, "title": contract.title,
            "mission_id": contract.mission_id, "status": contract.status,
            "worker": contract.worker, "verifier": contract.verifier,
            "specialist_class": contract.specialist_class,
            "generation": contract.generation,
            "consequential": contract.consequential,
            "claims": len(entries), "verdicts": verdicts,
            "blockers": [] if contract.terminal
            else self.completion_blockers(contract),
        }


def contract_for_step(store: DelegationStore, *, mission_id: str,
                      step: Mapping[str, Any], verifier: str,
                      skeptic: str = "", witness: str = "",
                      integration_owner: str = "",
                      recovery_supervisor: str = "", arbiter: str = "",
                      consequential: bool = False) -> WorkContract:
    """Cut a work contract for one mission plan step.

    The dispatcher seam: the step's role resolves through the
    specialist manifests to a governing class, whose evidence duties
    become the contract's required evidence.
    """
    from .manifests import class_for_role, manifest

    role = str(step.get("role") or "")
    class_id = class_for_role(role)
    spec = manifest(class_id)
    worker = f"{class_id}:{role}" if role else class_id
    return store.create_contract(
        goal=str(step.get("goal") or step.get("name") or "step"),
        mission_id=mission_id,
        title=str(step.get("name") or ""),
        worker=worker, verifier=verifier, skeptic=skeptic,
        witness=witness, integration_owner=integration_owner,
        recovery_supervisor=recovery_supervisor, arbiter=arbiter,
        specialist_class=class_id, consequential=consequential,
        required_evidence=tuple(spec.evidence_duties if spec else ()),
        scope=tuple(str(x) for x in (step.get("writes") or ())),
        step_budget=int(step.get("step_budget") or 0))
