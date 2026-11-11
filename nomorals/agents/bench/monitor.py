"""The bench Task Monitor — claiming, heartbeat, watchdog (Lane 1).

The monitor is the ONLY writer of bench state transitions.
Cursors never talk to each other and never touch bench tables;
they talk to the monitor, and the monitor composes the organs
that already exist rather than inventing parallel machinery:

- **Ownership** is ``core/leases.py``: a claim acquires a task
  lease (``bench-task:<task_id>``) plus one file lease per
  write-set entry (``bench-file:<entry>``), all inside ONE
  ``Database.transaction()`` — the database's re-entrant
  transactions make the leases' own transactions savepoints of
  the claim's, so a contention refusal rolls back every partial
  grant: a claim is atomic, never half-held. Lease generations
  are the fence: they increment on every acquisition and are
  never reused, so a superseded claimant is refused by
  ``LeaseStore.check`` on every later mutation (heartbeat,
  submit, release) — by the store, not by politeness.
- **Admission** additionally scans the durable held-task
  population for write-set overlap (exact paths AND subtree
  globs, via :func:`model.write_sets_overlap`): lease names alone
  cannot see that ``nomorals/agents/**`` covers
  ``nomorals/agents/swarm.py``, so the scan is what makes
  overlapping write-sets *structurally* unclaimable while a
  conflicting claim lives. A refusal names the holder, its task,
  and the lease expiry — never a silent wait, never a steal.
- **Readiness** is promotion-gated (dossier §2.4): a task whose
  deps are not all ``promoted`` is not claimable, full stop. Task
  structure is validated through ``core/tasks.py`` TaskGraph at
  registration (duplicates, unknown deps and cycles are the
  graph's own refusals).
- **Accountability** is ``agents/delegation.py``: the first claim
  of a task records its WorkContract (worker = cursor, verifier =
  the bench verifier pool, integration owner = this monitor), so
  the independence law and the claim ledger apply to bench work
  from the first minute.
- **Death** follows the mission run-claim pattern at one grain
  finer: leases expire at read (no sweep to trust), the watchdog
  additionally treats a recorded pid that is dead as a dead claim
  however fresh its stamp, and expiry requeues the task with the
  evidence journaled. The dead cursor's late submit hits the
  generation fence and is refused by the store.
- **The watchdog** runs as a ``core/jobs.py`` JobRegistry job,
  inheriting that registry's heartbeat/reap plumbing.

Promotion itself is Lane 2's integration queue
(:mod:`.integration`): :meth:`BenchMonitor.submit` moves a task
to ``submitted`` (leases retained, per dossier §3.4) — with a
staging manager attached, submit first captures the staging
diff as the candidate (:mod:`.staging`) — and
:meth:`BenchMonitor.promotion_candidates` is the ordered input
the queue consumes. Nothing here promotes, and the store's read
view (``BenchStore.peer_view``) keeps an unpromoted candidate
invisible to peers meanwhile. With a staging manager attached,
claims also create the task's staging area, releases discard
it, expiries quarantine it, and the watchdog's stall detection
reads artifact change in the staging tree as progress.

Lane 4 adds two monitor powers, both exercised through the
same claim machinery: an optional **claim guard** (a predicate
the wiring layer injects — the contract organ's, refusing
claims on tasks bound to a challenged contract; the monitor
stays the only writer, the guard only ever says no) and
**swarm spawning** (:meth:`BenchMonitor.spawn`): a cursor
subdivides its own fenced write-set among child tasks the
monitor registers, the partition checked disjoint and contained
before any child exists.

Layer: agents (L5) — imports core (L1) and the agents/missions
peers it composes.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from ...core.errors import ValidationError
from ...core.jobs import JobRegistry, shared_registry
from ...core.leases import LeaseStore
from ...core.tasks import Task, TaskGraph
from ..delegation import DelegationStore
from .model import (
    CLAIMABLE_STATES,
    HELD_STATES,
    MONITOR_REF,
    TASK_CLAIMED,
    TASK_PENDING,
    TASK_PROMOTED,
    TASK_REQUEUED,
    TASK_RETURNED,
    TASK_SUBMITTED,
    TASK_VERIFYING,
    TASK_WORKING,
    BenchError,
    BenchStore,
    TaskSpec,
    contract_kwargs,
    entry_within,
    overlapping_entries,
)

__all__ = ["BenchMonitor", "task_resource", "file_resource"]

#: Default claim lease TTL: two minutes between heartbeats is
#: generous for a live cursor and bounds how long a dead one can
#: mute its write-set (the command-claim window logic, one grain
#: finer). Owner/profile code may pass a different value.
DEFAULT_LEASE_TTL_S = 120.0

#: A live claim that shows no progress for this long earns a
#: ``stalled`` journal event (dossier §2.5). Progress in Lane 1
#: is a heartbeat that says so; Lane 2 refines the signal to
#: artifact change in the staging worktree.
DEFAULT_STALL_AFTER_S = 600.0

#: Bench stall reasons — concrete codes in the missions/progress
#: idiom (a stall is never a bare "stalled").
STALL_NO_ARTIFACT_CHANGE = "no_artifact_change"
STALL_PID_DEAD = "pid_dead"


def task_resource(task_id: str) -> str:
    """The lease resource naming one task's claim."""
    return "bench-task:" + str(task_id)


def file_resource(entry: str) -> str:
    """The lease resource naming one write-set entry's claim."""
    return "bench-file:" + str(entry)


class _ClaimRefused(Exception):
    """Internal: carries a typed refusal out of the claim
    transaction so the outer transaction rolls back every
    partial lease grant before the refusal is journaled."""

    def __init__(self, refusal: dict[str, Any]) -> None:
        super().__init__(str(refusal.get("reason") or "refused"))
        self.refusal = refusal


def _pid_alive(pid: int) -> bool:
    """True when ``pid`` names a live process. The mission
    run-claim liveness rule: a dead pid is a dead claim however
    fresh its stamp. Our own pid always reads alive; unknown or
    unprobeable pids read alive too (the lease expiry remains
    the backstop — a pid is evidence, never the only fence)."""
    if pid <= 0 or pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return True
    return True


class BenchMonitor:
    """Owns bench runs: registration, claiming, heartbeat,
    release, submit, and the watchdog that expires dead claims."""

    def __init__(self, db: Any, *,
                 lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
                 stall_after_s: float = DEFAULT_STALL_AFTER_S,
                 staging: Any = None,
                 claim_guard: Callable[..., dict[str, Any] | None]
                 | None = None) -> None:
        self.db = db
        self.store = BenchStore(db)
        self.leases = LeaseStore(db)
        self.delegation = DelegationStore(db)
        self.lease_ttl_s = float(lease_ttl_s)
        self.stall_after_s = float(stall_after_s)
        #: Lane 4: an optional claim guard, ``guard(run_id,
        #: task) -> refusal | None``, consulted before any claim
        #: is granted and in readiness. The wiring layer injects
        #: the contract organ's guard (a task bound to a
        #: challenged contract is suspended); a monitor without
        #: one claims exactly as Lanes 1–3 did. The guard never
        #: writes — refusal is its only power.
        self.claim_guard = claim_guard
        #: Lane 2: the staging manager (``bench/staging.py``), or
        #: None — a monitor without one claims exactly as Lane 1
        #: did; with one, claims create staging areas, submits
        #: capture candidates, and the watchdog reads artifact
        #: progress. Attached at construction by the wiring layer.
        self.staging = staging
        self._watchdog_stop = threading.Event()
        self._watchdog_job: Any = None

    # -- runs + registration -----------------------------------------

    def open_run(self, goal: str, base_revision: str,
                 *, now: float | None = None) -> dict[str, Any]:
        """Open a bench run at ``base_revision`` (the promoted
        HEAD tasks will base on until promotions move it)."""
        return self.store.create_run(goal, base_revision, now=now)

    def close_run(self, run_id: str,
                  *, now: float | None = None) -> dict[str, Any] | None:
        return self.store.close_run(run_id, now=now)

    def register_task(self, run_id: str, key: str, spec: TaskSpec,
                      *, now: float | None = None) -> dict[str, Any]:
        """Register one task on a run. The spec is validated
        (write-set + acceptance are mandatory), deps must already
        be registered, and the run's structure is re-validated
        through a TaskGraph — duplicates, unknown deps and cycles
        are refused with the graph's own reasons."""
        spec.validate()
        run = self.store.get_run(run_id)
        if run is None:
            raise BenchError("unknown bench run: " + str(run_id))
        if run["state"] != "open":
            raise BenchError("bench run is closed: " + str(run_id))
        if self.store.get_task(run_id, key) is not None:
            raise BenchError(
                "task key already registered on this run: " + str(key))
        graph = TaskGraph(name="bench:" + str(run_id),
                          mission_id=str(run_id))
        existing = self.store.list_tasks(run_id)
        # Existing tasks enter the graph in dependency order:
        # the store lists by (created_at, key), and a task whose
        # key sorts before its dep's would otherwise be added
        # before the dep exists and refused as unknown — the
        # graph is a validation view, so it is built in the
        # order the deps demand (Lane 4, found by swarm
        # spawning: children sort before their parent).
        added: set[str] = set()
        pending = list(existing)
        while pending:
            progressed = False
            for task in list(pending):
                dep_keys = [self._dep_key(existing, d)
                            for d in task["deps"]]
                if any(dep_key not in added
                       and dep_key != task["key"]
                       for dep_key in dep_keys):
                    continue
                graph.add(Task(name=task["key"], role=task["role"],
                               mission_id=str(run_id)),
                          depends_on=dep_keys)
                added.add(task["key"])
                pending.remove(task)
                progressed = True
            if not progressed:
                # Unreachable for store-written tasks (deps were
                # validated at their own registration); add the
                # rest in stored order and let the graph's own
                # refusal speak, exactly as before.
                for task in pending:
                    graph.add(
                        Task(name=task["key"], role=task["role"],
                             mission_id=str(run_id)),
                        depends_on=[self._dep_key(existing, d)
                                    for d in task["deps"]])
                break
        try:
            graph.add(Task(name=str(key), role=spec.role,
                           mission_id=str(run_id),
                           idempotency_key=str(run_id) + ":" + str(key)),
                      depends_on=list(spec.deps))
        except ValidationError as exc:
            raise BenchError(str(exc)) from exc
        task = self.store.insert_task(run_id, key, spec, now=now)
        self.store.append_event(
            run_id, task["id"], "task_registered", MONITOR_REF,
            detail={"key": str(key), "role": spec.role,
                    "write_set": list(spec.write_set),
                    "deps": list(spec.deps)},
            now=now)
        return task

    @staticmethod
    def _dep_key(tasks: list[dict[str, Any]], dep_id: str) -> str:
        for task in tasks:
            if task["id"] == dep_id:
                return task["key"]
        return dep_id

    def ready_tasks(self, run_id: str) -> list[dict[str, Any]]:
        """Claimable tasks whose deps are ALL promoted — the
        promotion-gated readiness rule (dossier §2.4). A dep that
        is merely done or submitted does not unblock. A task the
        claim guard refuses (Lane 4: bound to a challenged
        contract) is not ready either — suspension is a
        readiness fact, not a surprise at claim time."""
        tasks = self.store.list_tasks(run_id)
        by_key = {t["key"]: t for t in tasks}
        out: list[dict[str, Any]] = []
        for task in tasks:
            if task["state"] not in CLAIMABLE_STATES:
                continue
            if all(by_key.get(dep, {}).get("state") == TASK_PROMOTED
                   for dep in task["deps"]):
                if self.claim_guard is not None \
                        and self.claim_guard(run_id, task):
                    continue
                out.append(task)
        return out

    # -- claiming ------------------------------------------------------

    def claim(self, run_id: str, key: str, cursor_id: str, *,
              pid: int = 0, now: float | None = None) -> dict[str, Any]:
        """Claim one task AND generation-fenced leases over its
        whole write-set, as one transaction.

        Granted: ``{"claimed": True, "task_id", "generation",
        "leases", "base_revision", "contract_id"}`` (plus
        ``"already_held": True`` for the same cursor's idempotent
        re-claim). Refused: ``{"claimed": False, "reason", ...}``
        — ``deps_unpromoted`` names the unpromoted deps;
        ``write_set_held`` names the holding cursor, its task and
        the lease expiry; ``held`` is the lease authority's own
        contention record. A refusal is journaled; nothing is
        half-granted, ever."""
        now = time.time() if now is None else float(now)
        cursor_id = str(cursor_id or "").strip()
        if not cursor_id:
            raise BenchError("a claim needs a cursor id")
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"claimed": False, "reason": "unknown_task",
                    "run_id": str(run_id), "key": str(key)}
        run = self.store.get_run(run_id)
        if run is None or run["state"] != "open":
            return self._refuse(task, cursor_id, now,
                                {"claimed": False, "reason": "run_closed",
                                 "run_id": str(run_id)})
        if task["state"] in (TASK_CLAIMED, TASK_WORKING) \
                and task["holder"] == cursor_id \
                and self.leases.check(task_resource(task["id"]),
                                      cursor_id,
                                      task["claim_generation"],
                                      now=now):
            # Idempotent re-claim by the live claimant: the fence
            # checked out at the store.
            self._ensure_staging(run, task, now)
            return {"claimed": True, "already_held": True,
                    "task_id": task["id"],
                    "generation": task["claim_generation"],
                    "leases": task["leases"],
                    "base_revision": task["base_revision"],
                    "contract_id": task["contract_id"]}
        # A claimant whose own claim has lapsed falls through:
        # the block below expires the dead claim inline and this
        # call becomes a fresh claim at the next generation (the
        # old generation stays fenced for any late submit).
        if task["state"] in (TASK_CLAIMED, TASK_WORKING):
            # Another cursor's claim stands — unless it is already
            # dead at read (lease expiry is COMPUTED, never swept,
            # in the lease authority): a lapsed claim is expired
            # inline here and the task becomes claimable in this
            # same call, rather than bouncing until a watchdog
            # pass happens to run.
            lease = self.leases.current(task_resource(task["id"]),
                                        now=now)
            live = (lease is not None
                    and lease["holder_ref"] == task["holder"]
                    and int(lease["generation"])
                    == int(task["claim_generation"]))
            if live:
                return self._refuse(
                    task, cursor_id, now,
                    {"claimed": False, "reason": "not_claimable",
                     "state": task["state"], "task_id": task["id"],
                     "holder": task["holder"],
                     "expires_at": lease["expires_at"]})
            history = self.leases.leases_for_resource(
                task_resource(task["id"]), include_history=True,
                now=now)
            self._expire_claim(
                task, "lease_expired", now,
                evidence={"expires_at": (history[0]["expires_at"]
                                         if history else 0.0),
                          "inline": True})
            task = self.store.get_task(run_id, key) or task
        if task["state"] not in CLAIMABLE_STATES:
            return self._refuse(
                task, cursor_id, now,
                {"claimed": False, "reason": "not_claimable",
                 "state": task["state"], "task_id": task["id"],
                 "holder": task["holder"]})
        if self.claim_guard is not None:
            # Lane 4: the injected guard's refusal (e.g. the
            # task's contract stands challenged) is consulted
            # BEFORE the dep check and journaled like every
            # other refusal — suspension is the specific,
            # actionable reason (a challenged contract also
            # unpromotes its dep edge, and "deps_unpromoted"
            # would bury the cause). Suspension is a fact on
            # the record, never a silent wait.
            guard_refusal = self.claim_guard(run_id, task)
            if guard_refusal:
                refusal = {"claimed": False,
                           "task_id": task["id"]}
                refusal.update(guard_refusal)
                return self._refuse(task, cursor_id, now, refusal)
        unpromoted = self._unpromoted_deps(task)
        if unpromoted:
            return self._refuse(
                task, cursor_id, now,
                {"claimed": False, "reason": "deps_unpromoted",
                 "task_id": task["id"], "deps": unpromoted})
        try:
            with self.db.transaction():
                granted = self._claim_in_transaction(
                    run, task, cursor_id, pid=int(pid or 0), now=now)
        except _ClaimRefused as refused:
            return self._refuse(task, cursor_id, now, refused.refusal)
        self.store.append_event(
            run_id, task["id"], "claim_granted", cursor_id,
            generation=granted["generation"],
            detail={"leases": granted["leases"],
                    "base_revision": granted["base_revision"],
                    "attempt": granted["attempt"]},
            now=now)
        try:
            self._ensure_staging(
                run, self.store.get_task(run_id, key) or task, now)
        except BenchError:
            # The claim landed but its staging could not be
            # created: yield the claim rather than hand the
            # cursor a claim it cannot work in.
            self.release(run_id, key, cursor_id,
                         granted["generation"],
                         reason="staging_failed", now=now)
            raise
        return granted

    def _ensure_staging(self, run: dict[str, Any],
                        task: dict[str, Any], now: float) -> None:
        """Lane 2: create the claimed task's staging area
        (idempotent) and journal its creation. A monitor without
        a staging manager skips this entirely (Lane 1 shape)."""
        if self.staging is None:
            return
        area, created = self.staging.create(run["id"], task, now=now)
        if created:
            self.store.append_event(
                run["id"], task["id"], "staging_created", MONITOR_REF,
                generation=int(task.get("claim_generation") or 0),
                detail={"mechanism": area.mechanism,
                        "base_revision": area.base_revision,
                        "path": str(area.path)},
                now=now)

    def _claim_in_transaction(self, run: dict[str, Any],
                              task: dict[str, Any], cursor_id: str,
                              *, pid: int, now: float) -> dict[str, Any]:
        """The claim's atomic core. Runs inside the caller's
        BEGIN IMMEDIATE transaction; raising _ClaimRefused rolls
        back every lease granted so far in this attempt."""
        fresh = self.store.get_task(run["id"], task["key"])
        if fresh is None or fresh["state"] not in CLAIMABLE_STATES:
            raise _ClaimRefused(
                {"claimed": False, "reason": "not_claimable",
                 "state": fresh["state"] if fresh else "missing",
                 "task_id": task["id"]})
        task = fresh
        # Admission: no live claim anywhere on this store may hold
        # an overlapping write-set. A held-state task whose lease
        # has already lapsed (expiry is computed at read — the
        # watchdog's requeue may not have run yet) does not block:
        # its claim no longer lives.
        for other in self.store.held_tasks():
            if other["id"] == task["id"]:
                continue
            other_lease = self.leases.current(
                task_resource(other["id"]), now=now)
            if other_lease is None \
                    or other_lease["holder_ref"] != other["holder"] \
                    or int(other_lease["generation"]) \
                    != int(other["claim_generation"]):
                continue
            hits = overlapping_entries(task["write_set"],
                                       other["write_set"])
            if hits:
                raise _ClaimRefused({
                    "claimed": False, "reason": "write_set_held",
                    "task_id": task["id"], "files": hits,
                    "holder": other["holder"],
                    "holder_task": other["key"],
                    "holder_run": other["run_id"],
                    "expires_at": other_lease["expires_at"]})
        # Leases: the task lease first, then one per write-set
        # entry. Any contention rolls the whole claim back.
        lease_records: list[dict[str, Any]] = []
        task_lease = self._acquire(task_resource(task["id"]),
                                   cursor_id, now)
        lease_records.append(task_lease)
        for entry in task["write_set"]:
            lease_records.append(
                self._acquire(file_resource(entry), cursor_id, now))
        generation = int(task_lease["generation"])
        # Accountability: the delegation contract is recorded on
        # the FIRST claim; a re-claim after requeue works under
        # the same contract (its claim ledger carries the new
        # attempt) — the bench fence is the lease generation.
        contract_id = task["contract_id"]
        base_revision = str(run["promoted_revision"]
                            or run["base_revision"])
        if not contract_id:
            spec = TaskSpec(
                instructions=task["instructions"],
                acceptance=tuple(task["acceptance"]),
                role=task["role"],
                write_set=tuple(task["write_set"]),
                read_set=tuple(task["read_set"]),
                deps=tuple(task["deps"]))
            contract = self.delegation.create_contract(
                **contract_kwargs(run_id=run["id"],
                                   task_key=task["key"], spec=spec,
                                   cursor_id=cursor_id,
                                   base_revision=base_revision))
            contract_id = contract.id
        attempt = int(task["attempts"]) + 1
        moved = self.store.transition(
            task["id"], sorted(CLAIMABLE_STATES), TASK_CLAIMED, now=now,
            holder=cursor_id, pid=int(pid), claim_generation=generation,
            leases=lease_records, contract_id=contract_id,
            base_revision=base_revision, attempts=attempt,
            last_seen=now, last_progress_at=now, stalled=0,
            candidate_ref="")
        if not moved:
            raise _ClaimRefused(
                {"claimed": False, "reason": "not_claimable",
                 "state": task["state"], "task_id": task["id"]})
        return {"claimed": True, "task_id": task["id"],
                "generation": generation, "leases": lease_records,
                "base_revision": base_revision,
                "contract_id": contract_id, "attempt": attempt}

    def _acquire(self, resource: str, cursor_id: str,
                 now: float) -> dict[str, Any]:
        result = self.leases.acquire(resource, cursor_id,
                                     self.lease_ttl_s, now=now)
        if not result.get("granted"):
            raise _ClaimRefused({
                "claimed": False, "reason": "held",
                "resource": resource,
                "holder": result.get("holder_ref", ""),
                "expires_at": result.get("expires_at", 0.0),
                "generation": result.get("generation", 0)})
        lease = result["lease"]
        return {"resource": resource, "lease_id": lease["lease_id"],
                "generation": int(lease["generation"]),
                "expires_at": float(lease["expires_at"])}

    def _unpromoted_deps(self, task: dict[str, Any]) -> list[str]:
        by_key = {t["key"]: t
                  for t in self.store.list_tasks(task["run_id"])}
        return [dep for dep in task["deps"]
                if by_key.get(dep, {}).get("state") != TASK_PROMOTED]

    def _refuse(self, task: dict[str, Any], cursor_id: str,
                now: float, refusal: dict[str, Any]) -> dict[str, Any]:
        self.store.append_event(
            task["run_id"], task["id"], "claim_refused", cursor_id,
            reason=str(refusal.get("reason") or ""),
            detail={k: v for k, v in refusal.items()
                    if k not in ("claimed", "reason")},
            now=now)
        return refusal

    # -- swarm spawning (Lane 4) ----------------------------------------

    def spawn(self, run_id: str, parent_key: str, spawner_id: str,
              generation: int,
              children: Sequence[tuple[str, TaskSpec]], *,
              now: float | None = None) -> dict[str, Any]:
        """Subdivide a claimed task's fenced write-set among
        child tasks (swarm spawning): the spawner keeps its own
        task; the children are ordinary bench tasks on the same
        run, each depending on the parent (so none is claimable
        until the parent's own candidate — the decomposition
        itself, or the work it still does — is promoted and its
        file leases are released).

        The partition is enforced BEFORE any child exists:

        - the spawner must be the parent's live claimant at
          ``generation`` (the lease store's fence — a superseded
          spawner spawns nothing);
        - every child write-set entry must lie WITHIN the
          parent's write-set (:func:`model.entry_within`) — an
          escaping entry is refused, named;
        - the children's write-sets must be pairwise DISJOINT —
          an overlap is refused naming both children and the
          colliding entries;
        - every child's role must resolve in the role registry
          (``BUILTIN_ROLES``, aliases included) — the stored
          spec carries the canonical role name.

        Refusals are returned (``{"spawned": False, "reason",
        ...}``) and nothing is half-registered: validation of
        the whole partition precedes the first registration.
        Malformed specs raise :class:`BenchError`, as
        registration does."""
        now = time.time() if now is None else float(now)
        spawner_id = str(spawner_id or "").strip()
        parent = self.store.get_task(run_id, parent_key)
        if parent is None:
            return {"spawned": False, "reason": "unknown_task",
                    "run_id": str(run_id), "key": str(parent_key)}
        if parent["holder"] != spawner_id or not self.leases.check(
                task_resource(parent["id"]), spawner_id,
                int(generation), now=now):
            return {"spawned": False, "reason": "stale_generation",
                    "task_id": parent["id"]}
        if not children:
            return {"spawned": False, "reason": "no_children",
                    "task_id": parent["id"]}
        from ..role_specs import BUILTIN_ROLES, ROLE_ALIASES
        prepared: list[tuple[str, TaskSpec]] = []
        seen_keys: set[str] = set()
        for key, spec in children:
            key = str(key)
            if key == str(parent_key) or key in seen_keys \
                    or self.store.get_task(run_id, key) is not None:
                return {"spawned": False, "reason": "duplicate_key",
                        "child": key, "task_id": parent["id"]}
            seen_keys.add(key)
            spec.validate()
            role = str(spec.role or "")
            canonical = role if role in BUILTIN_ROLES \
                else ROLE_ALIASES.get(role)
            if canonical is None:
                return {"spawned": False, "reason": "unknown_role",
                        "child": key, "role": role,
                        "task_id": parent["id"]}
            escapes = [entry for entry in spec.write_set
                       if not entry_within(entry,
                                           parent["write_set"])]
            if escapes:
                return {"spawned": False,
                        "reason": "write_set_escape",
                        "child": key, "entries": escapes,
                        "parent_write_set":
                            list(parent["write_set"]),
                        "task_id": parent["id"]}
            prepared.append((key, TaskSpec(
                instructions=spec.instructions,
                acceptance=tuple(spec.acceptance), role=canonical,
                write_set=tuple(spec.write_set),
                read_set=tuple(spec.read_set),
                deps=tuple(spec.deps))))
        for index, (key_a, spec_a) in enumerate(prepared):
            for key_b, spec_b in prepared[index + 1:]:
                hits = overlapping_entries(spec_a.write_set,
                                           spec_b.write_set)
                if hits:
                    return {"spawned": False,
                            "reason": "partition_overlap",
                            "children": [key_a, key_b],
                            "files": hits,
                            "task_id": parent["id"]}
        # Registration in dependency order (a child may depend
        # on a sibling): repeat passes until every child is in;
        # a pass with no progress means a cycle or an unknown
        # dep — registration's own refusal, raised as BenchError.
        registered: dict[str, str] = {}
        pending = list(prepared)
        while pending:
            progressed = False
            for key, spec in list(pending):
                blockers = [dep for dep in spec.deps
                            if dep in seen_keys
                            and dep not in registered]
                if blockers:
                    continue
                deps = tuple(
                    dict.fromkeys(
                        (str(parent_key),)
                        + tuple(d for d in spec.deps
                                if d != str(parent_key))))
                child = self.register_task(
                    run_id, key,
                    TaskSpec(instructions=spec.instructions,
                             acceptance=spec.acceptance,
                             role=spec.role,
                             write_set=spec.write_set,
                             read_set=spec.read_set, deps=deps),
                    now=now)
                registered[key] = child["id"]
                self.store.append_event(
                    run_id, child["id"], "task_spawned", spawner_id,
                    generation=int(generation),
                    detail={"parent": str(parent_key),
                            "parent_task_id": parent["id"],
                            "write_set": list(spec.write_set)},
                    now=now)
                pending.remove((key, spec))
                progressed = True
            if not progressed:
                raise BenchError(
                    "spawn partition has unresolvable child "
                    "dependencies (cycle or unknown dep): "
                    + ", ".join(key for key, _ in pending))
        return {"spawned": True, "parent": str(parent_key),
                "task_id": parent["id"], "children": registered}

    # -- heartbeat / progress ------------------------------------------

    def heartbeat(self, run_id: str, key: str, cursor_id: str,
                  generation: int, *, progress: bool = False,
                  now: float | None = None) -> dict[str, Any]:
        """Renew the claim's leases at the recorded generations.
        A fenced claimant (lapsed + re-claimed, or released) is
        refused ``stale_generation`` by the lease store itself.
        ``progress=True`` is the cursor's word that work advanced
        — it resets the stall clock."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"renewed": False, "reason": "unknown_task"}
        if task["state"] not in HELD_STATES or not task["leases"]:
            return {"renewed": False, "reason": "not_claimed",
                    "state": task["state"]}
        if task["holder"] != str(cursor_id):
            return self._heartbeat_refused(task, cursor_id, generation,
                                           "not_holder", now)
        if int(generation) != int(task["claim_generation"]):
            return self._heartbeat_refused(task, cursor_id, generation,
                                           "stale_generation", now)
        expires_at = 0.0
        for record in task["leases"]:
            renewed = self.leases.renew(
                record["lease_id"], int(record["generation"]), now=now)
            if not renewed.get("renewed"):
                return self._heartbeat_refused(
                    task, cursor_id, generation,
                    str(renewed.get("reason") or "stale_generation"),
                    now)
            expires_at = max(expires_at,
                             float(renewed["lease"]["expires_at"]))
        fields: dict[str, Any] = {"last_seen": now}
        if progress:
            fields["last_progress_at"] = now
            fields["stalled"] = 0
        self.store.transition(task["id"], [task["state"]],
                              task["state"], now=now, **fields)
        self.store.append_event(
            run_id, task["id"], "heartbeat", str(cursor_id),
            generation=int(generation),
            detail={"progress": bool(progress),
                    "expires_at": expires_at},
            now=now)
        return {"renewed": True, "task_id": task["id"],
                "expires_at": expires_at}

    def _heartbeat_refused(self, task: dict[str, Any], cursor_id: str,
                           generation: int, reason: str,
                           now: float) -> dict[str, Any]:
        self.store.append_event(
            task["run_id"], task["id"], "heartbeat_refused",
            str(cursor_id), generation=int(generation), reason=reason,
            now=now)
        return {"renewed": False, "reason": reason,
                "task_id": task["id"]}

    # -- submit (the Lane 2 seam) ---------------------------------------

    def submit(self, run_id: str, key: str, cursor_id: str,
               generation: int, *, candidate_ref: str = "",
               now: float | None = None) -> dict[str, Any]:
        """Hand the task's candidate to the bench: state moves
        claimed/working -> submitted — and, for a task the
        verifier returned (Lane 3), returned -> submitted with
        the rework's NEW candidate ref (the old verdict, keyed
        to the old ref, has no power over the new one). The
        record then waits for the verifier; promotion happens
        only behind a supported verdict (the gate lives in
        integration.py). The fencing
        check is the lease store's: a claimant whose generation
        was superseded (its claim expired and the task was
        re-claimed) is refused ``stale_generation`` HERE, by the
        store — its late work cannot enter the queue."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"submitted": False, "reason": "unknown_task"}
        if not self.leases.check(task_resource(task["id"]),
                                 str(cursor_id), int(generation),
                                 now=now):
            self.store.append_event(
                run_id, task["id"], "submit_refused", str(cursor_id),
                generation=int(generation),
                reason="stale_generation", now=now)
            return {"submitted": False, "reason": "stale_generation",
                    "task_id": task["id"]}
        if self.staging is not None and not candidate_ref:
            # Lane 2: submit = capture. The staging diff becomes
            # the candidate NOW, while the fencing above still
            # holds; a capture failure refuses the submit with
            # the reason journaled — no phantom candidate.
            try:
                capture = self.staging.capture(run_id, task["id"])
            except BenchError as exc:
                self.store.append_event(
                    run_id, task["id"], "submit_refused",
                    str(cursor_id), generation=int(generation),
                    reason="capture_failed",
                    detail={"error": str(exc)}, now=now)
                return {"submitted": False,
                        "reason": "capture_failed",
                        "task_id": task["id"]}
            candidate_ref = str(capture["candidate_ref"])
            self.store.append_event(
                run_id, task["id"], "candidate_captured",
                str(cursor_id), generation=int(generation),
                detail={"candidate_ref": candidate_ref,
                        "files": capture["files"],
                        "empty": bool(capture["empty"]),
                        "base_revision": capture["base_revision"]},
                now=now)
        moved = self.store.transition(
            task["id"], [TASK_CLAIMED, TASK_WORKING, TASK_RETURNED],
            TASK_SUBMITTED,
            now=now, candidate_ref=str(candidate_ref or ""),
            submitted_at=now, last_seen=now)
        if not moved:
            return {"submitted": False, "reason": "not_claimed",
                    "state": task["state"], "task_id": task["id"]}
        self.store.append_event(
            run_id, task["id"], "task_submitted", str(cursor_id),
            generation=int(generation),
            detail={"candidate_ref": str(candidate_ref or "")},
            now=now)
        return {"submitted": True, "task_id": task["id"],
                "state": TASK_SUBMITTED}

    def promotion_candidates(self, run_id: str) -> list[dict[str, Any]]:
        """Tasks past submit, in submit order: ``submitted``
        (awaiting verification) then ``verifying``. This is the
        monitor's observability list; the integration queue's
        actual input is narrower — Lane 3's verdict gate in
        ``BenchIntegrator.pending`` admits only verifying tasks
        with a supported verdict for the current candidate ref.
        Nothing on either list is visible to peers as code
        until it is promoted (``BenchStore.peer_view``)."""
        tasks = [t for t in self.store.list_tasks(run_id)
                 if t["state"] in (TASK_SUBMITTED, TASK_VERIFYING)]
        return sorted(tasks,
                      key=lambda t: (t["submitted_at"], t["key"]))

    # -- release ---------------------------------------------------------

    def release(self, run_id: str, key: str, cursor_id: str,
                generation: int, *, reason: str = "",
                now: float | None = None) -> dict[str, Any]:
        """The claimant yields its claim: leases released at
        their generations (a fenced ex-holder's release is
        refused by the store and the successor's leases are
        untouched), the task returns to ``pending`` for another
        cursor."""
        now = time.time() if now is None else float(now)
        task = self.store.get_task(run_id, key)
        if task is None:
            return {"released": False, "reason": "unknown_task"}
        if not self.leases.check(task_resource(task["id"]),
                                 str(cursor_id), int(generation),
                                 now=now):
            return {"released": False, "reason": "stale_generation",
                    "task_id": task["id"]}
        for record in task["leases"]:
            self.leases.release(record["lease_id"],
                                int(record["generation"]), now=now)
        moved = self.store.transition(
            task["id"], [TASK_CLAIMED, TASK_WORKING], TASK_PENDING,
            now=now, holder="", pid=0, claim_generation=0, leases=[],
            last_seen=now)
        if moved:
            self.store.append_event(
                run_id, task["id"], "claim_released", str(cursor_id),
                generation=int(generation), reason=str(reason or ""),
                now=now)
            if self.staging is not None:
                # A voluntary yield abandons the staging area;
                # its content was never captured, so it goes.
                self.staging.discard(run_id, task["id"])
        return {"released": bool(moved), "task_id": task["id"]}

    # -- watchdog ---------------------------------------------------------

    def sweep(self, *, now: float | None = None) -> dict[str, Any]:
        """One watchdog pass over every held task on this store:
        expire dead claims (lease lapsed, or recorded pid dead)
        and requeue their tasks; journal a ``stalled`` event for
        live claims past the progress horizon. Returns what it
        did — the pass is idempotent and safe to run any time."""
        now = time.time() if now is None else float(now)
        expired: list[str] = []
        stalled: list[str] = []
        for task in self.store.held_tasks():
            if task["state"] not in (TASK_CLAIMED, TASK_WORKING):
                # Submitted+ claims are the verifier phase's
                # responsibility (Lane 3), not the watchdog's.
                continue
            lease = self.leases.current(task_resource(task["id"]),
                                        now=now)
            live = (lease is not None
                    and lease["holder_ref"] == task["holder"]
                    and int(lease["generation"])
                    == int(task["claim_generation"]))
            if live and task["pid"] and not _pid_alive(int(task["pid"])):
                self._force_release(task, now)
                self._expire_claim(task, STALL_PID_DEAD, now,
                                   evidence={"pid": int(task["pid"])})
                expired.append(task["key"])
                continue
            if not live:
                history = self.leases.leases_for_resource(
                    task_resource(task["id"]), include_history=True,
                    now=now)
                self._expire_claim(
                    task, "lease_expired", now,
                    evidence={"expires_at": (history[0]["expires_at"]
                                             if history else 0.0)})
                expired.append(task["key"])
                continue
            # Lane 2 refines the progress signal: a live claim
            # whose STAGING ARTIFACT changed since the last pass
            # has made progress whether or not its cursor said
            # so in a heartbeat — the artifact is the evidence,
            # the heartbeat was only the claim. The ``stalled``
            # event is the same event, now carrying the artifact
            # evidence it was judged on.
            artifact: dict[str, Any] | None = None
            if self.staging is not None:
                artifact = self.staging.observe_progress(
                    task["run_id"], task["id"])
                if artifact["changed"]:
                    self.store.transition(
                        task["id"], [task["state"]], task["state"],
                        now=now, last_progress_at=now, stalled=0)
                    task = {**task, "last_progress_at": now,
                            "stalled": False}
            if not task["stalled"] and self.stall_after_s > 0 and \
                    now - float(task["last_progress_at"] or 0) \
                    > self.stall_after_s:
                self.store.transition(
                    task["id"], [task["state"]], task["state"],
                    now=now, stalled=1)
                detail: dict[str, Any] = {
                    "stall_after_s": self.stall_after_s,
                    "last_progress_at": task["last_progress_at"]}
                if artifact is not None:
                    detail["artifact"] = {
                        "digest": artifact["digest"],
                        "files": artifact["files"],
                        "last_change_at":
                            artifact["last_change_at"]}
                self.store.append_event(
                    task["run_id"], task["id"], "stalled", MONITOR_REF,
                    generation=int(task["claim_generation"]),
                    reason=STALL_NO_ARTIFACT_CHANGE,
                    detail=detail,
                    now=now)
                stalled.append(task["key"])
        return {"expired": expired, "stalled": stalled}

    def _force_release(self, task: dict[str, Any], now: float) -> None:
        """Monitor-side release of a dead claimant's leases,
        using the generations recorded on the task row (the
        lease store's generation-CAS is the only authority
        consulted — a lease already re-granted refuses cleanly
        and is left alone)."""
        for record in task["leases"]:
            self.leases.release(record["lease_id"],
                                int(record["generation"]), now=now)

    def _expire_claim(self, task: dict[str, Any], reason: str,
                      now: float, *, evidence: dict[str, Any]) -> None:
        generation = int(task["claim_generation"])
        moved = self.store.transition(
            task["id"], [TASK_CLAIMED, TASK_WORKING], TASK_REQUEUED,
            now=now, holder="", pid=0, claim_generation=0, leases=[],
            candidate_ref="", last_seen=now)
        if not moved:
            return
        if self.staging is not None:
            # The dead cursor's staging is quarantined: kept for
            # inspection, never promoted, never a starting point.
            area = self.staging.quarantine(task["run_id"], task["id"])
            if area is not None and area.state == "quarantined":
                self.store.append_event(
                    task["run_id"], task["id"],
                    "staging_quarantined", MONITOR_REF,
                    generation=generation, reason=reason,
                    detail={"path": str(area.path)}, now=now)
        self.store.append_event(
            task["run_id"], task["id"], "claim_expired", MONITOR_REF,
            generation=generation, reason=reason, detail=evidence,
            now=now)
        self.store.append_event(
            task["run_id"], task["id"], "task_requeued", MONITOR_REF,
            generation=generation, reason=reason,
            detail={"attempts": int(task["attempts"]),
                    "dead_holder": task["holder"]},
            now=now)

    def start_watchdog(self, registry: JobRegistry | None = None, *,
                       interval_s: float = 30.0) -> Any:
        """Run :meth:`sweep` on a cadence inside a JobRegistry
        job (its heartbeat/reap plumbing included). Idempotent:
        a second call returns the already-running job."""
        if self._watchdog_job is not None \
                and not self._watchdog_job.done():
            return self._watchdog_job
        self._watchdog_stop.clear()
        stop = self._watchdog_stop

        def _loop() -> str:
            while not stop.wait(max(1.0, float(interval_s))):
                self.sweep()
            return "stopped"

        registry = registry if registry is not None \
            else shared_registry()
        self._watchdog_job = registry.submit(
            _loop, tool="bench-watchdog",
            labels={"bench": "watchdog"})
        return self._watchdog_job

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
