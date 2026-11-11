"""Multi-cursor bench — contracts and durable state (Lane 0).

The bench lets several coding cursors work one job in parallel
without ever seeing each other's unfinished work. Three organs
make that true, and this module is the first: the *contracts* —
what a bench task IS (a spec with a declared write-set, read-set,
acceptance criteria and dependencies), the state machines runs
and tasks move through, and the durable tables the Task Monitor
(:mod:`nomorals.agents.bench.monitor`) is the only writer of.

Tables are runtime-created on the shared home ``Database`` (the
Batch-B pattern — DDL in code, ``storage/migrations.py`` stays
frozen and untouched), exactly like ``command_claims`` and the
delegation tables:

- ``bench_runs`` — one row per bench run: the goal, the base
  revision it started from, and the current *promoted* revision
  (the only revision peers may resolve — see the read-view rule
  on :class:`BenchStore`).
- ``bench_tasks`` — one row per task: its spec, its state, its
  current claimant and the generation-fenced lease records that
  fence it (the lease authority itself is ``core/leases.py``;
  this table records WHICH leases a claim holds).
- ``bench_events`` — the monitor's append-only journal: every
  claim, refusal, heartbeat lapse, expiry, requeue, submit and
  stall, with actor, generation and reason. This is the owner's
  inspectable record of what the bench did and why.
- ``bench_verdicts`` — the verifier's journal (written by the
  Lane 3 verifier cursor; the table and verdict vocabulary are
  Lane 0 contracts so the monitor's promotion queue can consume
  them without a schema change later).

Task state machine (design dossier §2.1)::

    pending -> claimed -> working -> submitted -> verifying
        -> promoted | returned | conflicted
    claimed | working -> requeued -> claimed (next attempt)

``requeued`` is claimable exactly like ``pending``; the attempt
counter on the row says how many claims the task has burned.

Delegation mapping: every claimed task carries a
``delegation.WorkContract`` (worker = the claiming cursor,
verifier = the bench verifier pool, integration owner = the
monitor itself, recovery supervisor = the bench recovery role,
arbiter = the owner) — see :func:`contract_kwargs`. The
independence law is the delegation store's own: a contract whose
verifier equals its worker cannot be created, so a cursor can
never be its own verifier on a bench task.

Layer: agents (L5) — imports core (L1) only; the monitor composes
the peer organs (leases, delegation, jobs) from here.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BENCH_DDL",
    "BenchError",
    "BenchStore",
    "TaskSpec",
    "RUN_OPEN",
    "RUN_CLOSED",
    "TASK_PENDING",
    "TASK_CLAIMED",
    "TASK_WORKING",
    "TASK_SUBMITTED",
    "TASK_VERIFYING",
    "TASK_PROMOTED",
    "TASK_RETURNED",
    "TASK_CONFLICTED",
    "TASK_REQUEUED",
    "TASK_STATES",
    "CLAIMABLE_STATES",
    "HELD_STATES",
    "EVENT_KINDS",
    "VERDICTS",
    "MONITOR_REF",
    "VERIFIER_REF",
    "RECOVERY_REF",
    "ARBITER_REF",
    "contract_kwargs",
    "normalize_entry",
    "write_sets_overlap",
    "overlapping_entries",
]

# ── run + task states ─────────────────────────────────────────────

RUN_OPEN = "open"
RUN_CLOSED = "closed"

TASK_PENDING = "pending"
TASK_CLAIMED = "claimed"
TASK_WORKING = "working"
TASK_SUBMITTED = "submitted"
TASK_VERIFYING = "verifying"
TASK_PROMOTED = "promoted"
TASK_RETURNED = "returned"
TASK_CONFLICTED = "conflicted"
TASK_REQUEUED = "requeued"

TASK_STATES = frozenset({
    TASK_PENDING, TASK_CLAIMED, TASK_WORKING, TASK_SUBMITTED,
    TASK_VERIFYING, TASK_PROMOTED, TASK_RETURNED, TASK_CONFLICTED,
    TASK_REQUEUED,
})

#: States from which a cursor may claim the task.
CLAIMABLE_STATES = frozenset({TASK_PENDING, TASK_REQUEUED})

#: States in which a live claim holds the task's write-set: a
#: conflicting claim by another cursor is refused while any of
#: these stands (submitted/verifying/returned included — the
#: dossier §3.4 rule: the leases stay with the claimant while it
#: remains the live claimant, through verification and return).
HELD_STATES = frozenset({
    TASK_CLAIMED, TASK_WORKING, TASK_SUBMITTED, TASK_VERIFYING,
    TASK_RETURNED,
})

#: The bench_events vocabulary. Every monitor transition lands
#: here; nothing about a claim is true unless it is journaled.
EVENT_KINDS = frozenset({
    "run_opened", "run_closed", "task_registered",
    "claim_granted", "claim_refused", "claim_released",
    "claim_expired", "task_requeued", "heartbeat",
    "heartbeat_refused", "task_submitted", "submit_refused",
    "stalled",
    # Lane 2 — staging + integration queue.
    "staging_created", "staging_quarantined",
    "candidate_captured", "task_promoted", "task_conflicted",
    "monitor_defect",
    # Lane 3 — the verifier cursor.
    "verification_started", "verdict_recorded", "task_returned",
    "verification_escalated", "verification_timeout",
})

#: The verdict vocabulary — the delegation claim verdicts, reused
#: verbatim so a bench verdict IS a delegation verdict.
VERDICTS = frozenset({
    "supported", "contradicted", "partial", "stale",
    "unverified", "unavailable", "uncertain",
})

#: Role holders on every bench delegation contract. The verifier
#: is a POOL ref (Lane 3 instances register under it); the monitor
#: is the integration owner; the owner is the arbiter of last
#: resort (dossier §3.4 escalation).
MONITOR_REF = "bench-monitor"
VERIFIER_REF = "bench-verifier"
RECOVERY_REF = "bench-recovery"
ARBITER_REF = "owner"


class BenchError(Exception):
    """A bench contract violation — refused, never papered over."""


# ── task spec ─────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    """One bench task as data: what to do, what proves it, and
    exactly which files the doing may touch.

    ``write_set`` entries are repo-relative paths or subtree
    globs (``nomorals/agents/**`` claims the whole subtree — use
    sparingly, it serialises the layer). ``deps`` names task keys
    in the same run that must be PROMOTED (not merely finished)
    before this task may be claimed (dossier §2.4).
    """

    instructions: str
    acceptance: tuple[str, ...] = ()
    role: str = "coder"
    write_set: tuple[str, ...] = ()
    read_set: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()

    def validate(self) -> None:
        """The monitor refuses a spec it cannot fence: no
        write-set or no acceptance means the task's blast radius
        or its bar is undefined (dossier §2.7 — the monitor never
        invents either)."""
        if not str(self.instructions or "").strip():
            raise BenchError("a bench task needs instructions")
        if not self.write_set:
            raise BenchError(
                "a bench task needs a declared write-set — "
                "unbounded writes cannot be fenced")
        if not self.acceptance:
            raise BenchError(
                "a bench task needs acceptance criteria — the "
                "verifier judges against them, not against vibes")
        for entry in self.write_set:
            if not str(entry or "").strip():
                raise BenchError("write-set entries must be non-empty")
        for entry in self.read_set:
            if not str(entry or "").strip():
                raise BenchError("read-set entries must be non-empty")


# ── write-set overlap ─────────────────────────────────────────────

def normalize_entry(entry: str) -> tuple[str, str]:
    """Normalise one write-set entry to ``(kind, value)``:
    ``("file", path)`` for an exact path, ``("subtree", prefix)``
    for a glob/dir claim covering everything under ``prefix``,
    ``("all", "")`` for a claim covering the whole tree."""
    e = str(entry or "").replace("\\", "/").strip()
    while e.startswith("./"):
        e = e[2:]
    if e in ("", "*", "**"):
        return ("all", "")
    if e.endswith("/**"):
        return ("subtree", e[:-3].rstrip("/"))
    if e.endswith("/*"):
        # One-level glob, treated conservatively as the subtree:
        # over-refusing a claim is safe, under-refusing is the
        # collision this bench exists to prevent.
        return ("subtree", e[:-2].rstrip("/"))
    if "*" in e:
        prefix = e.split("*", 1)[0].rstrip("/")
        return ("subtree", prefix) if prefix else ("all", "")
    if e.endswith("/"):
        return ("subtree", e.rstrip("/"))
    return ("file", e)


def _entries_overlap(a: tuple[str, str], b: tuple[str, str]) -> bool:
    kind_a, val_a = a
    kind_b, val_b = b
    if kind_a == "all" or kind_b == "all":
        return True
    if kind_a == "file" and kind_b == "file":
        return val_a == val_b
    if kind_a == "file":
        return val_a == val_b or val_a.startswith(val_b + "/")
    if kind_b == "file":
        return val_b == val_a or val_b.startswith(val_a + "/")
    # subtree vs subtree
    return (val_a == val_b or val_a.startswith(val_b + "/")
            or val_b.startswith(val_a + "/"))


def overlapping_entries(mine: Sequence[str],
                        theirs: Sequence[str]) -> list[str]:
    """The entries of ``mine`` that collide with ``theirs`` —
    the admission check's evidence, named in the refusal."""
    theirs_norm = [normalize_entry(e) for e in theirs]
    hits: list[str] = []
    for entry in mine:
        norm = normalize_entry(entry)
        if any(_entries_overlap(norm, other) for other in theirs_norm):
            hits.append(str(entry))
    return hits


def write_sets_overlap(mine: Sequence[str],
                       theirs: Sequence[str]) -> bool:
    """True when two declared write-sets can touch the same file."""
    return bool(overlapping_entries(mine, theirs))


# ── delegation mapping ────────────────────────────────────────────

def contract_kwargs(*, run_id: str, task_key: str, spec: TaskSpec,
                    cursor_id: str, base_revision: str) -> dict[str, Any]:
    """The WorkContract a claim records for its task (dossier
    §2.2): the claiming cursor works, the verifier pool verifies,
    the monitor owns integration. Passed verbatim to
    ``DelegationStore.create_contract``."""
    return {
        "goal": spec.instructions,
        "mission_id": run_id,
        "title": task_key,
        "deliverable": "promoted candidate for " + task_key,
        "worker": cursor_id,
        "verifier": VERIFIER_REF,
        "integration_owner": MONITOR_REF,
        "recovery_supervisor": RECOVERY_REF,
        "arbiter": ARBITER_REF,
        "specialist_class": spec.role,
        "consequential": True,
        "required_evidence": tuple(spec.acceptance),
        "scope": tuple(spec.write_set),
        "base_revision": base_revision,
    }


# ── durable state ─────────────────────────────────────────────────

BENCH_DDL = """
CREATE TABLE IF NOT EXISTS bench_runs (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL DEFAULT '',
  base_revision TEXT NOT NULL DEFAULT '',
  promoted_revision TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL DEFAULT 0,
  closed_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS bench_tasks (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  key TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT '',
  instructions TEXT NOT NULL DEFAULT '',
  acceptance TEXT NOT NULL DEFAULT '[]',
  write_set TEXT NOT NULL DEFAULT '[]',
  read_set TEXT NOT NULL DEFAULT '[]',
  deps TEXT NOT NULL DEFAULT '[]',
  state TEXT NOT NULL DEFAULT 'pending',
  holder TEXT NOT NULL DEFAULT '',
  pid INTEGER NOT NULL DEFAULT 0,
  claim_generation INTEGER NOT NULL DEFAULT 0,
  leases TEXT NOT NULL DEFAULT '[]',
  contract_id TEXT NOT NULL DEFAULT '',
  base_revision TEXT NOT NULL DEFAULT '',
  candidate_ref TEXT NOT NULL DEFAULT '',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_seen REAL NOT NULL DEFAULT 0,
  last_progress_at REAL NOT NULL DEFAULT 0,
  stalled INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  submitted_at REAL NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bench_tasks_run_key
  ON bench_tasks(run_id, key);
CREATE INDEX IF NOT EXISTS idx_bench_tasks_run_state
  ON bench_tasks(run_id, state);
CREATE TABLE IF NOT EXISTS bench_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL DEFAULT '',
  task_id TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  actor TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 0,
  reason TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bench_events_run
  ON bench_events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_bench_events_task
  ON bench_events(task_id, id);
CREATE TABLE IF NOT EXISTS bench_verdicts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  task_id TEXT NOT NULL DEFAULT '',
  candidate_ref TEXT NOT NULL DEFAULT '',
  verifier TEXT NOT NULL DEFAULT '',
  verdict TEXT NOT NULL DEFAULT 'unverified',
  evidence_ref TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bench_verdicts_task
  ON bench_verdicts(task_id, created_at);
"""


def _run_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"], "goal": row["goal"],
        "base_revision": row["base_revision"],
        "promoted_revision": row["promoted_revision"],
        "state": row["state"],
        "created_at": float(row["created_at"] or 0),
        "closed_at": float(row["closed_at"] or 0),
    }


def _task_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"], "run_id": row["run_id"], "key": row["key"],
        "role": row["role"], "instructions": row["instructions"],
        "acceptance": tuple(json.loads(row["acceptance"] or "[]")),
        "write_set": tuple(json.loads(row["write_set"] or "[]")),
        "read_set": tuple(json.loads(row["read_set"] or "[]")),
        "deps": tuple(json.loads(row["deps"] or "[]")),
        "state": row["state"], "holder": row["holder"],
        "pid": int(row["pid"] or 0),
        "claim_generation": int(row["claim_generation"] or 0),
        "leases": json.loads(row["leases"] or "[]"),
        "contract_id": row["contract_id"],
        "base_revision": row["base_revision"],
        "candidate_ref": row["candidate_ref"],
        "attempts": int(row["attempts"] or 0),
        "last_seen": float(row["last_seen"] or 0),
        "last_progress_at": float(row["last_progress_at"] or 0),
        "stalled": bool(row["stalled"]),
        "created_at": float(row["created_at"] or 0),
        "updated_at": float(row["updated_at"] or 0),
        "submitted_at": float(row["submitted_at"] or 0),
    }


def _event_dict(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]), "run_id": row["run_id"],
        "task_id": row["task_id"], "kind": row["kind"],
        "actor": row["actor"],
        "generation": int(row["generation"] or 0),
        "reason": row["reason"],
        "detail": json.loads(row["detail"] or "{}"),
        "created_at": float(row["created_at"] or 0),
    }


@dataclass
class BenchStore:
    """The bench tables over the shared home ``Database``.

    This store records facts; it makes no scheduling decisions.
    Every state transition the monitor performs goes through
    :meth:`transition` (a compare-and-swap on the current state,
    so a lost race writes nothing) and lands in the journal via
    :meth:`append_event`.

    Read-view rule (dossier §2.3, enforced here at the data
    layer): a peer cursor resolving a run sees task *states* and
    PROMOTED revisions; an unpromoted task's ``candidate_ref`` is
    visible only to its own claimant and to the monitor/verifier
    roles. There is no view onto another cursor's unfinished work.
    """

    db: Any

    def __post_init__(self) -> None:
        self.db.executescript(BENCH_DDL)

    # -- runs -------------------------------------------------------

    def create_run(self, goal: str, base_revision: str,
                   *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        run_id = "br-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO bench_runs (id, goal, base_revision,"
            " promoted_revision, state, created_at, closed_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, str(goal or ""), str(base_revision or ""),
             str(base_revision or ""), RUN_OPEN, now, 0.0))
        self.append_event(run_id, "", "run_opened", MONITOR_REF,
                          reason=str(base_revision or ""), now=now)
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM bench_runs WHERE id = ?", (str(run_id),))
        return _run_dict(row) if row is not None else None

    def close_run(self, run_id: str,
                  *, now: float | None = None) -> dict[str, Any] | None:
        now = time.time() if now is None else float(now)
        self.db.execute(
            "UPDATE bench_runs SET state = ?, closed_at = ?"
            " WHERE id = ?",
            (RUN_CLOSED, now, str(run_id)))
        self.append_event(run_id, "", "run_closed", MONITOR_REF, now=now)
        return self.get_run(run_id)

    def note_promoted_revision(self, run_id: str, revision: str) -> None:
        """Advance the run's promoted revision — called ONLY by
        the Lane 2 promotion step, in the same transaction as the
        task's promotion. Peers resolve this revision and no
        other."""
        self.db.execute(
            "UPDATE bench_runs SET promoted_revision = ? WHERE id = ?",
            (str(revision), str(run_id)))

    # -- tasks ------------------------------------------------------

    def insert_task(self, run_id: str, key: str, spec: TaskSpec,
                    *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        task_id = "bt-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO bench_tasks (id, run_id, key, role,"
            " instructions, acceptance, write_set, read_set, deps,"
            " state, created_at, updated_at) VALUES (?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, str(run_id), str(key), spec.role,
             spec.instructions, json.dumps(list(spec.acceptance)),
             json.dumps(list(spec.write_set)),
             json.dumps(list(spec.read_set)),
             json.dumps(list(spec.deps)), TASK_PENDING, now, now))
        return self.get_task(run_id, key) or {}

    def get_task(self, run_id: str, key: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM bench_tasks WHERE run_id = ? AND key = ?",
            (str(run_id), str(key)))
        return _task_dict(row) if row is not None else None

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM bench_tasks WHERE id = ?", (str(task_id),))
        return _task_dict(row) if row is not None else None

    def list_tasks(self, run_id: str = "") -> list[dict[str, Any]]:
        if run_id:
            rows = self.db.query(
                "SELECT * FROM bench_tasks WHERE run_id = ?"
                " ORDER BY created_at ASC, key ASC", (str(run_id),))
        else:
            rows = self.db.query(
                "SELECT * FROM bench_tasks ORDER BY created_at ASC")
        return [_task_dict(row) for row in rows]

    def held_tasks(self) -> list[dict[str, Any]]:
        """Every task currently holding its write-set, across all
        runs on this store — the admission check's population
        (write-set claims fence the shared tree, not one run)."""
        marks = ",".join("?" for _ in HELD_STATES)
        rows = self.db.query(
            "SELECT * FROM bench_tasks WHERE state IN (" + marks + ")"
            " ORDER BY created_at ASC",
            tuple(sorted(HELD_STATES)))
        return [_task_dict(row) for row in rows]

    def transition(self, task_id: str, from_states: Sequence[str],
                   to_state: str, *, now: float | None = None,
                   **fields: Any) -> bool:
        """Compare-and-swap a task's state (+ columns): the update
        lands only when the task is still in one of
        ``from_states``. Returns False when the race was lost —
        the caller re-reads and decides; nothing is half-written."""
        now = time.time() if now is None else float(now)
        allowed = {"holder", "pid", "claim_generation", "leases",
                   "contract_id", "base_revision", "candidate_ref",
                   "attempts", "last_seen", "last_progress_at",
                   "stalled", "submitted_at"}
        unknown = set(fields) - allowed
        if unknown:
            raise BenchError(
                "transition fields not writable: "
                + ", ".join(sorted(unknown)))
        sets = ["state = ?", "updated_at = ?"]
        params: list[Any] = [to_state, now]
        for name, value in fields.items():
            sets.append(name + " = ?")
            params.append(
                json.dumps(value) if name == "leases" else value)
        marks = ",".join("?" for _ in from_states)
        params.append(str(task_id))
        params.extend(from_states)
        cursor = self.db.execute(
            "UPDATE bench_tasks SET " + ", ".join(sets)
            + " WHERE id = ? AND state IN (" + marks + ")",
            tuple(params))
        return bool(getattr(cursor, "rowcount", 0))

    # -- journal ------------------------------------------------------

    def append_event(self, run_id: str, task_id: str, kind: str,
                     actor: str, *, generation: int = 0,
                     reason: str = "", detail: dict[str, Any] | None = None,
                     now: float | None = None) -> None:
        """One journal row. The journal is append-only: events are
        never updated or deleted, so the record of a refusal or an
        expiry survives whatever happens next."""
        if kind not in EVENT_KINDS:
            raise BenchError("unknown bench event kind: " + str(kind))
        now = time.time() if now is None else float(now)
        self.db.execute(
            "INSERT INTO bench_events (run_id, task_id, kind, actor,"
            " generation, reason, detail, created_at) VALUES (?, ?,"
            " ?, ?, ?, ?, ?, ?)",
            (str(run_id), str(task_id), kind, str(actor),
             int(generation), str(reason),
             json.dumps(detail or {}), now))

    def events(self, run_id: str = "", task_id: str = "",
               kind: str = "") -> list[dict[str, Any]]:
        """Journal rows, oldest first, filtered by any of
        run/task/kind — the owner's inspectable record."""
        sql = "SELECT * FROM bench_events"
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(str(run_id))
        if task_id:
            clauses.append("task_id = ?")
            params.append(str(task_id))
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC"
        return [_event_dict(row)
                for row in self.db.query(sql, tuple(params))]

    # -- verdicts (Lane 0 contract; written by Lane 3) ---------------

    def record_verdict(self, run_id: str, task_id: str, *,
                       candidate_ref: str, verifier: str,
                       verdict: str, evidence_ref: str = "",
                       detail: dict[str, Any] | None = None,
                       now: float | None = None) -> dict[str, Any]:
        """Append one verifier verdict row. Verdicts are facts
        the promotion queue consumes; they are never edited."""
        if verdict not in VERDICTS:
            raise BenchError("unknown verdict: " + str(verdict))
        now = time.time() if now is None else float(now)
        verdict_id = "bv-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO bench_verdicts (id, run_id, task_id,"
            " candidate_ref, verifier, verdict, evidence_ref,"
            " detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?,"
            " ?, ?)",
            (verdict_id, str(run_id), str(task_id),
             str(candidate_ref), str(verifier), verdict,
             str(evidence_ref), json.dumps(detail or {}), now))
        return {"id": verdict_id, "run_id": str(run_id),
                "task_id": str(task_id),
                "candidate_ref": str(candidate_ref),
                "verifier": str(verifier), "verdict": verdict,
                "evidence_ref": str(evidence_ref),
                "detail": detail or {}, "created_at": now}

    def latest_verdict(self, task_id: str,
                       candidate_ref: str) -> dict[str, Any] | None:
        """The newest verdict recorded for EXACTLY this candidate
        — the integration gate's only input (Lane 3). A verdict
        written against an earlier candidate has no power over a
        resubmitted one: the refs differ, so it cannot match."""
        rows = [row for row in self.verdicts(task_id)
                if row["candidate_ref"] == str(candidate_ref)]
        return rows[-1] if rows else None

    def verdicts(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM bench_verdicts WHERE task_id = ?"
            " ORDER BY created_at ASC, id ASC", (str(task_id),))
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                "id": row["id"], "run_id": row["run_id"],
                "task_id": row["task_id"],
                "candidate_ref": row["candidate_ref"],
                "verifier": row["verifier"], "verdict": row["verdict"],
                "evidence_ref": row["evidence_ref"],
                "detail": json.loads(row["detail"] or "{}"),
                "created_at": float(row["created_at"] or 0),
            })
        return out

    # -- the read view ------------------------------------------------

    def peer_view(self, run_id: str,
                  for_cursor: str = "") -> list[dict[str, Any]]:
        """What a cursor may resolve about a run: every task's
        state and spec, the run's promoted revision — and a
        ``candidate_ref`` ONLY for promoted tasks and for the
        requesting cursor's own tasks. Another cursor's
        submitted-but-unpromoted candidate is structurally absent
        (empty ref + ``candidate_visible: False``), not merely
        discouraged."""
        run = self.get_run(run_id)
        out: list[dict[str, Any]] = []
        for task in self.list_tasks(run_id):
            visible = (task["state"] == TASK_PROMOTED
                       or (for_cursor != ""
                           and task["holder"] == for_cursor))
            view = dict(task)
            view["candidate_visible"] = visible
            if not visible:
                view["candidate_ref"] = ""
            view["promoted_revision"] = (
                run["promoted_revision"] if run else "")
            out.append(view)
        return out
