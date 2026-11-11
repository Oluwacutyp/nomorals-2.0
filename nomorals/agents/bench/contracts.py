"""Contract-first decomposition (Lane 4, owner extension).

The bench's answer to tightly-coupled work: when several
cursors build against one shared interface, the interface is
decided ONCE and promoted like any other candidate — before
the fan-out, not discovered by collision during it.

A **contract** is an ordinary bench task (role ``contract``)
whose write-set is exactly one artifact — the interface /
invariant document (an interface module, a schema spec, a
protocol description). Its lifecycle rides the shipped organs
end to end: a cursor claims it, writes the artifact in its
staging area, submits; the verifier judges it; the integration
queue promotes it. What this module adds is the *generation*
bookkeeping on top of that lifecycle:

- **Generations.** Each promotion of the contract task
  publishes a new contract generation (1, 2, 3, …). The
  generation's content is the artifact's bytes at the revision
  its promotion landed — read back through the staging
  manager's revision reads, so a generation is an artifact,
  never a recollection. Old generations stay readable forever:
  :meth:`BenchContracts.at_generation` resolves any published
  generation's bytes.
- **Bindings.** An implementation task binds to a contract
  (:meth:`BenchContracts.bind`); the binding records the
  generation the task works against. Implementations also list
  the contract task in their ``deps`` — the monitor's
  promotion-gated readiness then refuses a claim until the
  contract's first generation exists.
- **Freezing.** While a contract stands unchallenged, bound
  tasks claim and build against their bound generation. The
  contract cannot move underneath them: its author revising it
  is a new claim cycle, and publication is a new generation —
  never an edit to the one in force.
- **Challenge.** A cursor that finds the contract wrong files
  a challenge (:meth:`BenchContracts.challenge`) — it never
  edits the contract itself. The challenge is journaled
  (``contract_challenged``), every bound task is suspended
  (``task_suspended`` journaled; the monitor's claim guard —
  wired in :mod:`.wiring` — refuses their claims naming the
  contract), and the contract task returns to ``requeued`` so
  its author can claim it and revise. When the revision
  promotes, :meth:`BenchContracts.sync_promotions` publishes
  the new generation (``contract_revised``) and **rebases**
  every binding onto it (``task_rebased`` journaled per task,
  with the suspension lift recorded) — affected tasks then
  claim against the new generation.

The tables are runtime-created on the shared home ``Database``
(the Batch-B pattern — migrations stay frozen), beside the
Lane 0 bench tables this module composes.

Layer: agents (L5) — composes the bench model/monitor/staging
peers only.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .model import (
    MONITOR_REF,
    TASK_PROMOTED,
    TASK_REQUEUED,
    BenchError,
    BenchStore,
)

__all__ = [
    "BenchContracts",
    "CONTRACT_DDL",
    "CONTRACT_DRAFT",
    "CONTRACT_CURRENT",
    "CONTRACT_CHALLENGED",
    "CONTRACT_ROLE",
]

#: The bench role a contract task registers under. The
#: verifier's check table reads it as the feature class (its
#: acceptance carries the executable bar).
CONTRACT_ROLE = "contract"

CONTRACT_DRAFT = "draft"
CONTRACT_CURRENT = "current"
CONTRACT_CHALLENGED = "challenged"

#: Task states a challenge leaves alone when suspending: a
#: promoted binding is history, not a suspension candidate.
_TERMINAL_FOR_SUSPENSION = frozenset({TASK_PROMOTED})

CONTRACT_DDL = """
CREATE TABLE IF NOT EXISTS bench_contracts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  key TEXT NOT NULL DEFAULT '',
  task_id TEXT NOT NULL DEFAULT '',
  artifact_path TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'draft',
  current_revision TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bench_contracts_run_key
  ON bench_contracts(run_id, key);
CREATE TABLE IF NOT EXISTS bench_contract_generations (
  contract_id TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 0,
  revision TEXT NOT NULL DEFAULT '',
  artifact_path TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (contract_id, generation)
);
CREATE TABLE IF NOT EXISTS bench_task_contracts (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  contract_key TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
"""


def _contract_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"], "run_id": row["run_id"], "key": row["key"],
        "task_id": row["task_id"],
        "artifact_path": row["artifact_path"],
        "generation": int(row["generation"] or 0),
        "state": row["state"],
        "current_revision": row["current_revision"],
        "created_at": float(row["created_at"] or 0),
        "updated_at": float(row["updated_at"] or 0),
    }


class BenchContracts:
    """Contract generations, bindings, challenges and rebases
    for one bench store. Composes the monitor (contract tasks
    are bench tasks) and the staging manager (generation
    content is read at promoted revisions)."""

    def __init__(self, db: Any, *, staging: Any = None,
                 monitor: Any = None) -> None:
        self.db = db
        self.store = BenchStore(db)
        self.staging = staging
        self.monitor = monitor
        self.db.executescript(CONTRACT_DDL)

    def attach_monitor(self, monitor: Any) -> None:
        """Wire the monitor after construction (the monitor's
        claim guard points back at this organ, so the two are
        built in either order and attached)."""
        self.monitor = monitor

    # -- definition ---------------------------------------------------

    def define(self, run_id: str, key: str, artifact_path: str, *,
               instructions: str = "",
               acceptance: tuple[str, ...] = (),
               now: float | None = None) -> dict[str, Any]:
        """Register a contract on a run: a bench task whose
        write-set is exactly the artifact, plus the contract
        record standing at generation 0 (``draft`` — nothing is
        in force until the first promotion publishes
        generation 1).

        The default acceptance is executable by construction —
        the artifact must exist, be non-empty, and (for a
        ``.py`` artifact) compile — because the verifier judges
        a contract like any feature candidate, and prose alone
        earns no verdict."""
        if self.monitor is None:
            raise BenchError(
                "defining a contract needs the bench monitor "
                "(the contract is a bench task)")
        now = time.time() if now is None else float(now)
        artifact_path = str(artifact_path or "").strip()
        if not artifact_path:
            raise BenchError("a contract needs an artifact path")
        if self.get(run_id, key) is not None:
            raise BenchError(
                "contract already defined on this run: " + str(key))
        if not acceptance:
            if artifact_path.endswith(".py"):
                acceptance = (
                    "cmd:python3 -m py_compile " + artifact_path,)
            else:
                acceptance = ("cmd:test -s " + artifact_path,)
        from .model import TaskSpec
        spec = TaskSpec(
            instructions=(instructions or (
                "Author the shared contract artifact "
                + artifact_path + " — the interface/invariants "
                "the bound tasks build against.")),
            acceptance=tuple(acceptance), role=CONTRACT_ROLE,
            write_set=(artifact_path,))
        task = self.monitor.register_task(run_id, key, spec, now=now)
        contract_id = "bc-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO bench_contracts (id, run_id, key, task_id,"
            " artifact_path, generation, state, current_revision,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0,"
            " ?, '', ?, ?)",
            (contract_id, str(run_id), str(key), task["id"],
             artifact_path, CONTRACT_DRAFT, now, now))
        return self.get(run_id, key) or {}

    def get(self, run_id: str, key: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM bench_contracts WHERE run_id = ?"
            " AND key = ?", (str(run_id), str(key)))
        return _contract_dict(row) if row is not None else None

    def list_contracts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM bench_contracts WHERE run_id = ?"
            " ORDER BY created_at ASC", (str(run_id),))
        return [_contract_dict(row) for row in rows]

    # -- bindings -------------------------------------------------------

    def bind(self, run_id: str, task_key: str, contract_key: str, *,
             now: float | None = None) -> dict[str, Any]:
        """Bind a task to a contract at the contract's CURRENT
        generation (0 while the contract is still a draft — the
        binding rebases onto generation 1 when it publishes).
        One binding per task; re-binding replaces it."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, task_key)
        if task is None:
            raise BenchError("unknown bench task: " + str(task_key))
        contract = self.get(run_id, contract_key)
        if contract is None:
            raise BenchError(
                "unknown contract on this run: " + str(contract_key))
        self.db.execute(
            "INSERT OR REPLACE INTO bench_task_contracts"
            " (task_id, run_id, contract_key, generation,"
            " updated_at) VALUES (?, ?, ?, ?, ?)",
            (task["id"], str(run_id), str(contract_key),
             int(contract["generation"]), now))
        return {"task_id": task["id"], "task_key": str(task_key),
                "contract_key": str(contract_key),
                "generation": int(contract["generation"])}

    def binding_for(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM bench_task_contracts WHERE task_id = ?",
            (str(task_id),))
        if row is None:
            return None
        return {"task_id": row["task_id"], "run_id": row["run_id"],
                "contract_key": row["contract_key"],
                "generation": int(row["generation"] or 0)}

    def bindings(self, run_id: str,
                 contract_key: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM bench_task_contracts WHERE run_id = ?"
            " AND contract_key = ? ORDER BY task_id ASC",
            (str(run_id), str(contract_key)))
        return [{"task_id": row["task_id"], "run_id": row["run_id"],
                 "contract_key": row["contract_key"],
                 "generation": int(row["generation"] or 0)}
                for row in rows]

    # -- the claim guard -------------------------------------------------

    def claim_guard(self, run_id: str,
                    task: dict[str, Any]) -> dict[str, Any] | None:
        """The monitor's claim guard (wired by :mod:`.wiring`):
        a task bound to a CHALLENGED contract is not claimable —
        the refusal names the contract and the generation in
        force when it was challenged. Every other task passes
        (``None``). The contract task itself carries no binding,
        so it stays claimable for its revision throughout."""
        binding = self.binding_for(str(task.get("id") or ""))
        if binding is None:
            return None
        contract = self.get(run_id, binding["contract_key"])
        if contract is None or contract["state"] != CONTRACT_CHALLENGED:
            return None
        return {"reason": "contract_challenged",
                "contract": contract["key"],
                "generation": int(contract["generation"]),
                "artifact_path": contract["artifact_path"]}

    # -- challenge --------------------------------------------------------

    def challenge(self, run_id: str, key: str, *,
                  challenger: str, reason: str = "",
                  now: float | None = None) -> dict[str, Any]:
        """File a challenge against the contract in force. The
        challenger never edits the contract: the contract task
        returns to ``requeued`` for its author's revision, every
        bound task is suspended (journaled, and claim-refused by
        the guard) until the revision publishes a new
        generation."""
        now = time.time() if now is None else float(now)
        contract = self.get(run_id, key)
        if contract is None:
            raise BenchError(
                "unknown contract on this run: " + str(key))
        if contract["state"] != CONTRACT_CURRENT:
            raise BenchError(
                "only a contract in force can be challenged — "
                + str(key) + " is " + contract["state"]
                + " (generation "
                + str(contract["generation"]) + ")")
        task = self.store.get_task_by_id(contract["task_id"])
        self.db.execute(
            "UPDATE bench_contracts SET state = ?, updated_at = ?"
            " WHERE id = ?",
            (CONTRACT_CHALLENGED, now, contract["id"]))
        self.store.append_event(
            run_id, contract["task_id"], "contract_challenged",
            str(challenger), reason=str(reason or ""),
            detail={"contract": str(key),
                    "generation": int(contract["generation"]),
                    "artifact_path": contract["artifact_path"]},
            now=now)
        if task is not None and task["state"] == TASK_PROMOTED:
            moved = self.store.transition(
                task["id"], [TASK_PROMOTED], TASK_REQUEUED, now=now,
                candidate_ref="")
            if moved:
                self.store.append_event(
                    run_id, task["id"], "task_requeued", MONITOR_REF,
                    reason="contract_challenge",
                    detail={"contract": str(key),
                            "generation":
                                int(contract["generation"])},
                    now=now)
        suspended: list[str] = []
        for binding in self.bindings(run_id, key):
            bound = self.store.get_task_by_id(binding["task_id"])
            if bound is None \
                    or bound["state"] in _TERMINAL_FOR_SUSPENSION:
                continue
            self.store.append_event(
                run_id, bound["id"], "task_suspended", MONITOR_REF,
                reason="contract_challenged",
                detail={"contract": str(key),
                        "generation":
                            int(contract["generation"])},
                now=now)
            suspended.append(bound["key"])
        out = self.get(run_id, key) or {}
        out["suspended"] = suspended
        return out

    # -- publication ------------------------------------------------------

    def sync_promotions(self, run_id: str, *,
                        now: float | None = None) -> list[dict[str, Any]]:
        """Publish generations for contract promotions the
        journal has recorded since the last sync: each
        ``task_promoted`` event on a contract task beyond the
        recorded generations publishes the next generation,
        rebases every binding onto it, and (after a challenge)
        lifts the suspension. Idempotent — a second sync over
        the same journal publishes nothing."""
        now = time.time() if now is None else float(now)
        published: list[dict[str, Any]] = []
        for contract in self.list_contracts(run_id):
            recorded = self.db.query(
                "SELECT generation FROM bench_contract_generations"
                " WHERE contract_id = ?", (contract["id"],))
            done = {int(row["generation"]) for row in recorded}
            promotions = self.store.events(
                run_id, contract["task_id"], "task_promoted")
            # Generations publish in promotion order, one per
            # promotion; the count already recorded says how
            # many promotions an earlier sync consumed.
            fresh = promotions[len([g for g in done if g > 0]):]
            for event in fresh:
                revision = str(event["detail"].get("revision") or "")
                published.append(self._publish(
                    contract, revision, now=now))
                contract = self.get(run_id, contract["key"]) \
                    or contract
        return published

    def _publish(self, contract: dict[str, Any], revision: str, *,
                 now: float) -> dict[str, Any]:
        generation = int(contract["generation"]) + 1
        was_challenged = contract["state"] == CONTRACT_CHALLENGED
        self.db.execute(
            "INSERT OR REPLACE INTO bench_contract_generations"
            " (contract_id, generation, revision, artifact_path,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (contract["id"], generation, revision,
             contract["artifact_path"], now))
        self.db.execute(
            "UPDATE bench_contracts SET generation = ?, state = ?,"
            " current_revision = ?, updated_at = ? WHERE id = ?",
            (generation, CONTRACT_CURRENT, revision, now,
             contract["id"]))
        self.store.append_event(
            contract["run_id"], contract["task_id"],
            "contract_defined" if generation == 1
            else "contract_revised",
            MONITOR_REF,
            detail={"contract": contract["key"],
                    "generation": generation,
                    "revision": revision,
                    "artifact_path": contract["artifact_path"]},
            now=now)
        for binding in self.bindings(contract["run_id"],
                                     contract["key"]):
            if int(binding["generation"]) >= generation:
                continue
            self.db.execute(
                "UPDATE bench_task_contracts SET generation = ?,"
                " updated_at = ? WHERE task_id = ?",
                (generation, now, binding["task_id"]))
            bound = self.store.get_task_by_id(binding["task_id"])
            self.store.append_event(
                contract["run_id"], binding["task_id"],
                "task_rebased", MONITOR_REF,
                detail={"contract": contract["key"],
                        "task": bound["key"] if bound else "",
                        "from_generation":
                            int(binding["generation"]),
                        "to_generation": generation,
                        "suspension_lifted": bool(was_challenged)},
                now=now)
        return {"contract": contract["key"],
                "generation": generation, "revision": revision}

    # -- reading generations ------------------------------------------------

    def current(self, run_id: str, key: str) -> dict[str, Any]:
        """The contract in force: its generation, the revision
        it published at, and the artifact's bytes AT that
        revision — the frozen text bound tasks build against."""
        contract = self.get(run_id, key)
        if contract is None:
            raise BenchError(
                "unknown contract on this run: " + str(key))
        if int(contract["generation"]) < 1:
            raise BenchError(
                "contract " + str(key) + " has no generation in "
                "force yet — it is still a draft")
        return self._read(contract, int(contract["generation"]),
                          contract["current_revision"])

    def at_generation(self, run_id: str, key: str,
                      generation: int) -> dict[str, Any]:
        """One published generation, read back at its own
        revision — old generations stay readable after a
        revision supersedes them."""
        contract = self.get(run_id, key)
        if contract is None:
            raise BenchError(
                "unknown contract on this run: " + str(key))
        row = self.db.query_one(
            "SELECT * FROM bench_contract_generations"
            " WHERE contract_id = ? AND generation = ?",
            (contract["id"], int(generation)))
        if row is None:
            raise BenchError(
                "contract " + str(key) + " has no generation "
                + str(generation))
        return self._read(contract, int(generation),
                          str(row["revision"]))

    def _read(self, contract: dict[str, Any], generation: int,
              revision: str) -> dict[str, Any]:
        if self.staging is None:
            raise BenchError(
                "reading a contract generation needs the staging "
                "manager (generation content lives at promoted "
                "revisions)")
        content = self.staging.read_at_revision(
            revision, contract["artifact_path"])
        return {"contract": contract["key"],
                "generation": generation,
                "state": contract["state"],
                "artifact_path": contract["artifact_path"],
                "revision": revision, "content": content}
