"""Delegation manager: async spawn / track / steer / resume / collect.

Today Devon can only run subagents synchronously in one shot
(``agents.subagents.run_parallel`` blocks until every worker
returns; ``agents.orchestrator.MasterOrchestrator.run`` executes a
plan end to end in the caller's thread). This manager turns that
into async managed delegation the way the assistant assigns
subagents: a spawn returns at once, the worker runs on a background
thread, every state change lands on a durable row, and the result
comes back into the owner's chat and mission loop.

The governance layer is NOT reimplemented here — it is reused:
every spawn cuts a work contract through the shipped
``agents.plan_contracts.open_step_contract`` seam (which itself
rides ``agents.delegation.contract_for_step``), worker results flow
through the shipped ``settle_step_contract`` gate (evidence per
required kind → verifier verifies → one standing skeptic challenge
raised and resolved → sign-off → completion), and failures flow
through ``fail_step_contract``. ``agents/delegation.py`` itself is
untouched. Every manager-spawned contract is consequential by
construction: async delegated work whose output lands in the
mission/chat loop is integration-risk work, so the full gate
applies — the settle path runs the skeptic pass every time.

Durability (the Batch B pattern — ``storage/migrations.py`` is
frozen, so tables are runtime-created on the shared ``Database``):

* ``delegation_runs`` — one row per spawned run: the contract it
  serves, the roster role, the worker input, steering notes, the
  result, the delivery state. The row is the record across turns
  and restarts; the in-memory future is only a handle.
* ``delegation_contracts`` / ``claim_ledger`` — owned by the
  untouched ``DelegationStore``; the run row points at its contract.

Restart semantics: a second manager over the same database (a
restart) finds no live futures, so ``recover()`` marks every
orphaned running row ``interrupted`` with a failure claim on its
contract, and ``resume()`` re-plans through the store (generation
bumps, fencing the old worker) and re-submits the stored input
with the accumulated steering applied. A late finalize from the
superseded worker is fenced twice: by the generation it started
with, and by the row's status claim — it can never overwrite its
replacement's accepted result.

Steering is honest about what it can do: a note is recorded
durably on the run row and on the claim ledger at once, and it
reaches the worker's input on the next (re)invocation, merged into
the role's direction field (planner task, implementer instruction,
tester focus, refactorer goal, api_designer feature). Roles with
no direction field (reviewer diff, dep_hunter name) keep the note
ledger-only — the note is never silently pretended into an input
it cannot enter. Steering a live run cannot rewrite an input
already handed to the worker; it applies at the next boundary.

Collection: ``collect()`` delivers a completed run's result into
the chat via the notifier (durable-first; persist-only when no
channel is live) and marks the run collected. The contract itself
is completed by the settle gate at finalize time, not by collect —
so completing the contract unblocks the mission loop
(``missions.mission_effect_outcome`` reads open contracts) whether
or not a chat channel is live. Collecting twice replays without a
second delivery. A run whose gate refused completion (defect in
the recorded result) cannot collect: the blockers are named, and
the owner steers and resumes instead.

The scheduler tick (``__delegation_manager_tick__``, every 5m)
calls ``poll()``: recover orphans, resume what the restart
interrupted, finalize finished futures, collect completed runs.
The tick reuses one process-long manager (``_tick_manager``) so a
tick never leaks a thread pool per run; ``from_context`` always
builds a fresh manager, which is what restart recovery needs.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .delegation import DelegationError, DelegationStore, WorkContract
from .failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "DelegationError",
    "DelegationManager",
    "DelegationRun",
    "RUN_STATUSES",
    "PLAN_ROLE_TO_ROSTER",
    "DELEGATION_JOB",
    "DELEGATION_SPEC",
    "DELEGATION_TOOL",
    "ensure_delegation_manager_job",
    "run_delegation_manager_tick_once",
]

#: Run states. ``finalizing`` is transient: the claim that exactly
#: one finalizer owns a finished worker (a dead process can leave a
#: row there; recover() treats it as orphaned like ``running``).
RUN_STATUSES = ("running", "steering", "finalizing", "completed",
                "failed", "interrupted", "collected", "cancelled")
_LIVE_STATUSES = ("running", "steering", "finalizing")
_RESUMABLE_STATUSES = ("interrupted", "failed")

#: Plan-step role -> roster worker. Only roles with a real roster
#: worker map; anything else is reported unassigned by
#: :meth:`DelegationManager.plan_and_spawn` with the reason named —
#: a step is never silently run by the wrong worker.
PLAN_ROLE_TO_ROSTER: dict[str, str] = {
    "research": "planner",
    "reflection": "planner",
    "coding": "implementer",
    "critic": "reviewer",
}

#: Roster role -> the input field that carries direction, so steering
#: notes merge into a field the worker actually reads. Roles absent
#: here (reviewer ``diff``, dep_hunter ``name``) carry no direction:
#: their steering stays ledger-only.
_PRIMARY_TEXT_FIELD: dict[str, str] = {
    "planner": "task",
    "implementer": "instruction",
    "tester": "focus",
    "refactorer": "goal",
    "api_designer": "feature",
}

#: Roster role -> input fields that must be non-empty. Tester needs
#: none (``focus`` is optional); implementer/refactorer need a target
#: file, so a bare goal cannot spawn them — the refusal names the
#: missing field instead of guessing a file.
_REQUIRED_INPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "planner": ("task",),
    "implementer": ("path", "instruction"),
    "reviewer": ("diff",),
    "tester": (),
    "refactorer": ("path", "goal"),
    "dep_hunter": ("name",),
    "api_designer": ("feature",),
}

#: Roster role -> minimal input built from a bare goal. Implementer
#: and refactorer are absent: a goal alone names no target file.
_MINIMAL_INPUT: dict[str, Callable[[str], dict[str, Any]]] = {
    "planner": lambda goal: {"task": goal},
    "reviewer": lambda goal: {"diff": goal},
    "tester": lambda goal: {"focus": goal},
    "dep_hunter": lambda goal: {"name": goal},
    "api_designer": lambda goal: {"feature": goal},
}

_DDL = """
CREATE TABLE IF NOT EXISTS delegation_runs (
  id TEXT PRIMARY KEY,
  contract_id TEXT NOT NULL DEFAULT '',
  mission_id TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT '',
  input_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'running',
  generation INTEGER NOT NULL DEFAULT 1,
  worker TEXT NOT NULL DEFAULT '',
  result_json TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  notes_json TEXT NOT NULL DEFAULT '[]',
  chat_key TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  completed_at REAL NOT NULL DEFAULT 0,
  delivered INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_contract
  ON delegation_runs(contract_id);
CREATE INDEX IF NOT EXISTS idx_runs_mission
  ON delegation_runs(mission_id);
CREATE INDEX IF NOT EXISTS idx_runs_status
  ON delegation_runs(status);
"""


@dataclass
class DelegationRun:
    """One spawned subagent run: the durable record across turns."""

    id: str
    contract_id: str = ""
    mission_id: str = ""
    role: str = ""
    input_json: str = "{}"
    status: str = "running"
    generation: int = 1
    worker: str = ""
    result_json: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    chat_key: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float = 0.0
    delivered: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "collected",
                               "cancelled")

    def input_dict(self) -> dict[str, Any]:
        try:
            data = json.loads(self.input_json or "{}")
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def result_dict(self) -> dict[str, Any]:
        try:
            data = json.loads(self.result_json or "")
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return data


def _row_run(row: Any) -> DelegationRun:
    try:
        notes = json.loads(row["notes_json"] or "[]")
    except (ValueError, TypeError):
        notes = []
    return DelegationRun(
        id=row["id"], contract_id=row["contract_id"],
        mission_id=row["mission_id"], role=row["role"],
        input_json=row["input_json"], status=row["status"],
        generation=int(row["generation"]), worker=row["worker"],
        result_json=row["result_json"], error=row["error"],
        notes=list(notes) if isinstance(notes, list) else [],
        chat_key=row["chat_key"], created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        completed_at=float(row["completed_at"]),
        delivered=bool(row["delivered"]))


class DelegationManager:
    """Async managed delegation over a :class:`DelegationStore`."""

    def __init__(self, store: DelegationStore, db: Any = None, *,
                 context: Any = None,
                 executor: Any = None,
                 run_agent: Callable[[str, dict[str, Any]],
                                     dict[str, Any]] | None = None,
                 notifier: Callable[..., dict[str, Any]] | None = None,
                 max_workers: int = 2) -> None:
        if store is None:
            raise DelegationError(
                "delegation manager needs a DelegationStore")
        self.store = store
        self.db = db if db is not None else getattr(store, "db", None)
        if self.db is None:
            raise DelegationError(
                "delegation manager needs a database — no db on "
                "the store")
        self.db.executescript(_DDL)
        self.context = context
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="delegate")
        self._owns_executor = executor is None
        self._run_agent = run_agent or self._run_roster_agent
        self._notifier = notifier
        #: run id -> (future, generation the worker started with).
        #: The generation is the fence: a late finalize from a
        #: superseded worker carries the old generation and is
        #: dropped, never recorded on the contract.
        self._futures: dict[str, tuple[Any, int]] = {}
        self._futures_lock = threading.Lock()

    @classmethod
    def from_context(cls, context: Any, **kw: Any) -> DelegationManager:
        """Build a fresh manager from a runtime context.

        Always fresh — a restart means a NEW manager over the same
        database, which is exactly what ``recover()`` needs. The
        scheduler tick uses the process-long ``_tick_manager``
        instead, so it never leaks a thread pool per tick.
        """
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            raise DelegationError(
                "delegation manager needs the database, which this "
                "context doesn't carry")
        return cls(DelegationStore(db), db=db, context=context, **kw)

    # -- roster ---------------------------------------------------------

    @staticmethod
    def _roster() -> dict[str, Any]:
        from .subagents import ROSTER

        return ROSTER

    def _build_input(self, role: str, goal: str,
                     input: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(input or {})
        if not data:
            minimal = _MINIMAL_INPUT.get(role)
            if minimal is None:
                missing = ", ".join(_REQUIRED_INPUT_FIELDS.get(role, ()))
                raise DelegationError(
                    f"role {role!r} needs input ({missing}) — a goal "
                    "alone names no target file; pass the input "
                    "explicitly")
            data = minimal(goal)
        missing = [f for f in _REQUIRED_INPUT_FIELDS.get(role, ())
                   if not str(data.get(f) or "").strip()]
        if missing:
            raise DelegationError(
                f"role {role!r} needs input field(s): "
                f"{', '.join(missing)}")
        return data

    @staticmethod
    def _apply_steering(role: str, input_dict: dict[str, Any],
                        notes: list[str]) -> tuple[dict[str, Any], bool]:
        """Merge steering notes into the worker's input.

        Returns ``(merged, applied)``: ``applied`` is False for roles
        with no direction field — the notes stay ledger-only there,
        never pretended into an input they cannot enter.
        """
        if not notes:
            return dict(input_dict), False
        field = _PRIMARY_TEXT_FIELD.get(role)
        if not field:
            return dict(input_dict), False
        merged = dict(input_dict)
        direction = ("\n\nDirection from the owner:\n"
                     + "\n".join(f"- {note}" for note in notes))
        merged[field] = (str(merged.get(field) or "") + direction).strip()
        return merged, True

    # -- durable rows -----------------------------------------------------

    def _insert_run(self, run: DelegationRun) -> None:
        self.db.execute(
            "INSERT INTO delegation_runs (id, contract_id, mission_id,"
            " role, input_json, status, generation, worker,"
            " result_json, error, notes_json, chat_key, created_at,"
            " updated_at, completed_at, delivered)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.id, run.contract_id, run.mission_id, run.role,
             run.input_json, run.status, run.generation, run.worker,
             run.result_json, run.error, json.dumps(list(run.notes)),
             run.chat_key, run.created_at, run.updated_at,
             run.completed_at, int(run.delivered)))

    def _update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {"status", "generation", "result_json", "error",
                   "notes_json", "delivered", "completed_at", "worker",
                   "input_json"}
        sets = [f"{key} = ?" for key in fields if key in allowed]
        values = [fields[key] for key in fields if key in allowed]
        if not sets:
            return
        sets.append("updated_at = ?")
        values.append(time.time())
        values.append(run_id)
        self.db.execute(
            f"UPDATE delegation_runs SET {', '.join(sets)} WHERE id = ?",
            tuple(values))

    def _claim_finalize(self, run_id: str) -> bool:
        """Exactly one finalizer owns a finished worker.

        The conditional UPDATE is the claim: the first caller flips
        the row to ``finalizing``; a racing second caller (tick vs
        done-callback) sees zero rows and stands down.
        """
        cursor = self.db.execute(
            "UPDATE delegation_runs SET status = 'finalizing',"
            " updated_at = ? WHERE id = ? AND status IN"
            " ('running', 'steering')",
            (time.time(), run_id))
        return bool(getattr(cursor, "rowcount", 0))

    def get_run(self, run_id: str) -> DelegationRun | None:
        row = self.db.query_one(
            "SELECT * FROM delegation_runs WHERE id = ?",
            (str(run_id or ""),))
        return _row_run(row) if row else None

    def list_runs(self, *, mission_id: str = "", status: str = "",
                  limit: int = 20) -> list[DelegationRun]:
        if mission_id and status:
            rows = self.db.query(
                "SELECT * FROM delegation_runs WHERE mission_id = ?"
                " AND status = ? ORDER BY created_at DESC LIMIT ?",
                (mission_id, status, int(limit)))
        elif mission_id:
            rows = self.db.query(
                "SELECT * FROM delegation_runs WHERE mission_id = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (mission_id, int(limit)))
        elif status:
            rows = self.db.query(
                "SELECT * FROM delegation_runs WHERE status = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (status, int(limit)))
        else:
            rows = self.db.query(
                "SELECT * FROM delegation_runs ORDER BY created_at DESC"
                " LIMIT ?", (int(limit),))
        return [_row_run(r) for r in rows]

    # -- spawn ------------------------------------------------------------

    def spawn(self, *, role: str, goal: str,
              input: dict[str, Any] | None = None, mission_id: str = "",
              scope: Any = (), step_budget: int = 0, chat_key: str = "",
              name: str = "") -> DelegationRun:
        """Cut a contract and run the roster worker asynchronously.

        Returns at once with the run in ``running`` state; the worker
        executes on a background thread and finalizes through the
        settle gate when it finishes. Raises
        :class:`DelegationError` for an unknown role or an input that
        cannot run (the missing field is named, never guessed).
        """
        role = (role or "").strip().lower()
        roster = self._roster()
        if role not in roster:
            raise DelegationError(
                f"unknown roster role {role!r}; roster: "
                f"{sorted(roster)}")
        goal = " ".join(str(goal or "").split()).strip()
        if not goal:
            raise DelegationError("spawn needs a goal — what is the "
                                  "worker meant to do?")
        data = self._build_input(role, goal, input)
        from .plan_contracts import open_step_contract

        step = {"name": name or f"delegate-{role}", "goal": goal,
                "role": role, "writes": list(scope or ()),
                "step_budget": int(step_budget or 0)}
        contract = open_step_contract(
            self.store, mission_id=mission_id, step=step)
        now = time.time()
        run = DelegationRun(
            id=f"dr-{uuid.uuid4().hex[:12]}",
            contract_id=contract.id, mission_id=mission_id, role=role,
            input_json=json.dumps(data), status="running",
            generation=contract.generation, worker=contract.worker,
            chat_key=chat_key, created_at=now, updated_at=now)
        self._insert_run(run)
        self._submit(run, data)
        return run

    def _submit(self, run: DelegationRun,
                input_dict: dict[str, Any] | None = None) -> None:
        data = (dict(input_dict) if input_dict is not None
                else run.input_dict())
        future = self._executor.submit(self._execute, run.id,
                                       run.role, data)
        with self._futures_lock:
            self._futures[run.id] = (future, int(run.generation))
        future.add_done_callback(
            lambda fut, run_id=run.id: self._on_done(run_id, fut))

    def _execute(self, run_id: str, role: str,
                 input_dict: dict[str, Any]) -> dict[str, Any]:
        """The worker invocation: never raises out of the thread."""
        try:
            outcome = self._run_agent(role, input_dict)
        except Exception as exc:  # noqa: BLE001 - failures are records
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(outcome, dict):
            return outcome
        return {"ok": True, "result": outcome}

    @staticmethod
    def _run_roster_agent(role: str,
                          input_dict: dict[str, Any]) -> dict[str, Any]:
        """The default worker: a real roster subagent from
        ``agents.subagents``. Mirrors the ``subagent_run`` tool's
        Input construction (``<ClassName>Input`` + field filter), so
        the manager and the tool never disagree about what runs."""
        from . import subagents as _subagents

        cls = _subagents.ROSTER.get(role)
        if cls is None:
            return {"ok": False,
                    "error": f"unknown roster role {role!r}"}
        try:
            agent = cls()
            # Find the Input dataclass: <Role>Input naming
            # convention first, then any dataclass ending in
            # "Input" whose name starts with the role class's
            # first four letters (Planner -> PlanInput) — the same
            # lookup the subagent_run tool uses, so the manager and
            # the tool never disagree about what runs.
            input_cls = getattr(_subagents, f"{cls.__name__}Input", None)
            if input_cls is None or not dataclasses.is_dataclass(
                    input_cls):
                prefix = cls.__name__.lower()[:4]
                for _name, _value in vars(_subagents).items():
                    if (_name.endswith("Input")
                            and dataclasses.is_dataclass(_value)
                            and _name.lower().startswith(prefix)):
                        input_cls = _value
                        break
            if input_cls is None or not dataclasses.is_dataclass(
                    input_cls):
                return {"ok": False,
                        "error": f"no Input dataclass for role {role!r}"}
            fields = {f.name for f in dataclasses.fields(input_cls)}
            kwargs = {k: v for k, v in (input_dict or {}).items()
                      if k in fields}
            result = agent.run(input_cls(**kwargs))
            if dataclasses.is_dataclass(result):
                return {"ok": True,
                        "result": dataclasses.asdict(result)}
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - failures are records
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    # -- finalize -----------------------------------------------------------

    def _on_done(self, run_id: str, future: Any) -> bool:
        try:
            outcome = self._outcome_from_future(future)
        except Exception as exc:  # noqa: BLE001 - finalize never raises
            record_failure("delegation_manager.outcome", exc)
            outcome = {"ok": False,
                       "error": f"{type(exc).__name__}: {exc}"}
        try:
            done = self._finalize(run_id, outcome)
        except Exception as exc:  # noqa: BLE001 - finalize never raises
            record_failure("delegation_manager.finalize", exc)
            done = False
        with self._futures_lock:
            self._futures.pop(run_id, None)
        return done

    @staticmethod
    def _outcome_from_future(future: Any) -> dict[str, Any]:
        try:
            outcome = future.result()
        except Exception as exc:  # noqa: BLE001 - failures are records
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(outcome, dict):
            return outcome
        return {"ok": True, "result": outcome}

    def _step_for_run(self, run: DelegationRun,
                      contract: WorkContract) -> dict[str, Any]:
        """The duck-typed step mapping the settle seam consumes."""
        writes = list(contract.scope or ())
        return {"name": f"delegate-{run.role}", "goal": contract.goal,
                "role": run.role, "writes": writes,
                "payload": {"writes": writes},
                "step_budget": contract.step_budget}

    def _finalize(self, run_id: str,
                  outcome: dict[str, Any]) -> bool:
        """Flow one finished worker through its contract.

        Exactly one finalizer wins (the status claim); a finalize
        carrying a superseded generation is dropped before it can
        touch the contract. Returns True when this call settled the
        run.
        """
        run = self.get_run(run_id)
        if run is None:
            return False
        contract = self.store.get_contract(run.contract_id)
        if contract is None:
            record_failure(
                "delegation_manager.finalize",
                ValueError(f"run {run_id} has no contract "
                           f"{run.contract_id!r}"))
            return False
        with self._futures_lock:
            entry = self._futures.get(run_id)
        if entry is not None and int(entry[1]) != int(contract.generation):
            # A superseded worker's late output: dropped, never
            # recorded on the contract. The replan claim already
            # names the supersede; this is only the log line.
            record_failure(
                "delegation_manager.stale_finalize",
                ValueError(
                    f"run {run_id} finished at generation "
                    f"{entry[1]}; contract is at generation "
                    f"{contract.generation} — late output dropped"))
            return False
        if not self._claim_finalize(run_id):
            return False
        now = time.time()
        if outcome.get("ok"):
            result = outcome.get("result")
            try:
                from .plan_contracts import settle_step_contract

                settle_step_contract(
                    self.store, contract,
                    step=self._step_for_run(run, contract),
                    result=result, mission_id=run.mission_id)
            except Exception as exc:  # noqa: BLE001 - settle never kills
                record_failure("delegation_manager.settle", exc)
                self._update_run(run_id, status="failed",
                                 result_json=json.dumps(outcome),
                                 error=f"contract settle failed: {exc}",
                                 completed_at=now)
                return True
            self._update_run(run_id, status="completed",
                             result_json=json.dumps(outcome),
                             completed_at=now)
            return True
        detail = str(outcome.get("error") or
                     "worker failed without a reason")
        try:
            from .plan_contracts import fail_step_contract

            fail_step_contract(self.store, contract, detail=detail)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            record_failure("delegation_manager.fail_contract", exc)
        self._update_run(run_id, status="failed",
                         result_json=json.dumps(outcome), error=detail,
                         completed_at=now)
        return True

    # -- track / steer ------------------------------------------------------

    def track(self, run_id: str) -> DelegationRun | None:
        """One run's durable record (alias for :meth:`get_run`)."""
        return self.get_run(run_id)

    def steer(self, run_id: str, note: str, *,
              by: str = "owner") -> DelegationRun:
        """Record direction for a live, failed, interrupted, or
        gate-refused run.

        The note lands durably on the run row and on the claim ledger
        at once. It reaches the worker's input on the next
        (re)invocation — a live worker's input cannot be rewritten
        mid-flight, so a running run flips to ``steering`` and the
        note applies at the next boundary. Steering a run whose
        contract already settled (collected / cancelled) is refused
        with the state named.
        """
        run = self.get_run(run_id)
        if run is None:
            raise DelegationError(f"no delegation run {run_id!r}")
        if run.status in ("running", "steering"):
            new_status = "steering"
        elif run.status in ("interrupted", "failed"):
            # The fail -> steer -> resume flow: direction is
            # recorded now and reaches the worker when it re-runs.
            new_status = run.status
        elif run.status == "completed":
            # A gate-refused run (completed, contract still open)
            # takes steering for its resume; a run whose contract
            # already settled does not.
            contract = self.store.get_contract(run.contract_id)
            if contract is not None and contract.terminal:
                raise DelegationError(
                    f"run {run_id} already completed its contract "
                    "— collect it instead of steering")
            new_status = "completed"
        else:
            raise DelegationError(
                f"run {run_id} is {run.status} — steering a finished "
                "run changes nothing; resume it to run again with "
                "new direction")
        note = " ".join(str(note or "").split()).strip()
        if not note:
            raise DelegationError(
                "steer needs a direction — /delegate steer <run> "
                "<note>")
        notes = list(run.notes) + [note]
        self._update_run(run_id, notes_json=json.dumps(notes),
                         status=new_status)
        contract = self.store.get_contract(run.contract_id)
        if contract is not None:
            # The ledger's append entry point: the public
            # submit_evidence hardcodes kind="evidence", and a steer
            # is not evidence — it is direction, on the audit trail.
            self.store._record(  # noqa: SLF001 - same package, ledger seam
                contract, kind="steer",
                claim=f"steered by {by}: {note}", producer=by,
                verdict="supported")
        return self.get_run(run_id) or run

    # -- resume -------------------------------------------------------------

    def resume(self, run_id: str, *, by: str = "owner") -> DelegationRun:
        """Re-run an interrupted, failed, or gate-refused run.

        Re-plans through the store (generation bumps, fencing the
        old worker) and re-submits the stored input with the
        accumulated steering applied. A live run cannot resume; a
        completed contract cannot resume either — it collects.
        """
        run = self.get_run(run_id)
        if run is None:
            raise DelegationError(f"no delegation run {run_id!r}")
        contract = self.store.get_contract(run.contract_id)
        if contract is None:
            raise DelegationError(
                f"run {run_id} has no contract {run.contract_id!r}")
        if run.status in ("running", "steering", "finalizing"):
            raise DelegationError(
                f"run {run_id} is {run.status} — only interrupted, "
                "failed, or gate-refused runs resume")
        if run.status in ("collected", "cancelled"):
            raise DelegationError(
                f"run {run_id} is {run.status} — terminal runs do "
                "not resume")
        if run.status == "completed" and contract.terminal:
            raise DelegationError(
                f"run {run_id} already completed its contract — "
                "collect it instead of resuming")
        from .plan_contracts import SUPERVISOR

        self.store.replan(
            contract.id, supervisor=SUPERVISOR,
            note=f"resumed by {by} after {run.status}")
        fresh = self.store.get_contract(run.contract_id)
        assert fresh is not None  # contract existed a line above
        notes = list(run.notes)
        merged, _applied = self._apply_steering(
            run.role, run.input_dict(), notes)
        # The row keeps the PRISTINE input: steering merges at submit
        # time only, so a second resume never duplicates the
        # direction lines.
        self._update_run(run_id, generation=fresh.generation,
                         status="running", result_json="", error="")
        self._submit(self.get_run(run_id) or run, merged)
        return self.get_run(run_id) or run

    def resume_all_interrupted(self, *, by: str = "owner"
                               ) -> list[DelegationRun]:
        """Recover, then resume every interrupted run."""
        resumed: list[DelegationRun] = []
        for run in self.recover():
            try:
                resumed.append(self.resume(run.id, by=by))
            except Exception as exc:  # noqa: BLE001 - one bad resume
                record_failure("delegation_manager.resume_all", exc)
        return resumed

    # -- recover / poll -------------------------------------------------------

    def recover(self) -> list[DelegationRun]:
        """Mark orphaned running rows interrupted (restart recovery).

        A row is orphaned when this process holds no live future for
        it: a done future is finalized first (the worker may have
        finished as the process died), anything else left running is
        interrupted with a failure claim on its contract. Never
        raises; per-row failures are recorded, not fatal.
        """
        recovered: list[DelegationRun] = []
        try:
            rows = self.list_runs(limit=500)
        except Exception as exc:  # noqa: BLE001 - recovery never raises
            record_failure("delegation_manager.recover.list", exc)
            return recovered
        for run in rows:
            if run.status not in _LIVE_STATUSES:
                continue
            with self._futures_lock:
                entry = self._futures.get(run.id)
            if entry is not None:
                future, _generation = entry
                if not future.done():
                    continue
                try:
                    self._on_done(run.id, future)
                except Exception as exc:  # noqa: BLE001
                    record_failure("delegation_manager.recover.done",
                                   exc)
                continue
            try:
                contract = self.store.get_contract(run.contract_id)
                if contract is not None and not contract.terminal:
                    self.store._record(  # noqa: SLF001 - ledger seam
                        contract, kind="failure",
                        claim=(f"delegation run {run.id} interrupted "
                               "by restart — no live worker; resume "
                               "to re-run"),
                        producer="delegation-manager",
                        verdict="supported")
            except Exception as exc:  # noqa: BLE001 - recorded
                record_failure("delegation_manager.recover.claim", exc)
            self._update_run(run.id, status="interrupted",
                             error="interrupted by restart — resume "
                                   "to re-run")
            fresh = self.get_run(run.id)
            if fresh is not None:
                recovered.append(fresh)
        return recovered

    def poll(self) -> dict[str, Any]:
        """One tick: heal, finalize, collect. Never raises.

        The scheduler tick's whole job: recover orphans, resume what
        the restart interrupted, finalize finished futures, collect
        completed runs into the chat. A run whose gate refused
        completion is left for the owner (collect would refuse it
        with its blockers named).
        """
        out: dict[str, Any] = {"recovered": [], "resumed": [],
                               "finalized": [], "collected": []}
        try:
            recovered = self.recover()
        except Exception as exc:  # noqa: BLE001 - the tick never dies
            record_failure("delegation_manager.poll.recover", exc)
            recovered = []
        out["recovered"] = [r.id for r in recovered]
        for run in recovered:
            try:
                resumed = self.resume(run.id, by="delegation-tick")
                out["resumed"].append(resumed.id)
            except Exception as exc:  # noqa: BLE001
                record_failure("delegation_manager.poll.resume", exc)
        try:
            rows = self.list_runs(limit=500)
        except Exception as exc:  # noqa: BLE001
            record_failure("delegation_manager.poll.list", exc)
            rows = []
        for run in rows:
            if run.status not in _LIVE_STATUSES:
                continue
            with self._futures_lock:
                entry = self._futures.get(run.id)
            if entry is None or not entry[0].done():
                continue
            try:
                if self._on_done(run.id, entry[0]):
                    out["finalized"].append(run.id)
            except Exception as exc:  # noqa: BLE001
                record_failure("delegation_manager.poll.finalize", exc)
        try:
            rows = self.list_runs(limit=500)
        except Exception as exc:  # noqa: BLE001
            record_failure("delegation_manager.poll.list2", exc)
            rows = []
        for run in rows:
            if run.status != "completed":
                continue
            try:
                outcome = self.collect(run.id)
                if outcome.get("ok"):
                    out["collected"].append(run.id)
            except Exception as exc:  # noqa: BLE001
                record_failure("delegation_manager.poll.collect", exc)
        return out

    # -- collect --------------------------------------------------------------

    def _notify(self, kind: str, title: str,
                body: str) -> dict[str, Any]:
        """Deliver through the injected notifier, else the context's.

        No context and no injected notifier means no channel: the
        delivery is reported undelivered with the reason named, and
        the result stays on the run record — never lost, never
        pretended sent.
        """
        if self._notifier is not None:
            outcome = self._notifier(kind, title, body)
            return outcome if isinstance(outcome, dict) \
                else {"delivered": bool(outcome)}
        if self.context is None:
            return {"delivered": False,
                    "reason": "no context — no chat channel to "
                              "deliver to; the result stays on the "
                              "run record"}
        from .notifier import notify

        outcome = notify(self.context, kind, title, body)
        return outcome if isinstance(outcome, dict) \
            else {"delivered": bool(outcome)}

    def collect(self, run_id: str, *, by: str = "") -> dict[str, Any]:
        """Deliver a completed run's result into the chat.

        Requires the run completed AND its contract settled by the
        gate (finalize settles; a gate refusal leaves the contract
        open and collect refuses with the blockers named). Delivery
        goes through the notifier; a dead channel does not block the
        mission loop — the run still collects, with the delivery
        failure named. Idempotent: a second collect replays without
        a second delivery.
        """
        run = self.get_run(run_id)
        if run is None:
            return {"ok": False, "run_id": run_id, "contract_id": "",
                    "delivered": False,
                    "reasons": [f"no delegation run {run_id!r}"],
                    "replayed": False}
        if run.status == "collected":
            return {"ok": True, "run_id": run.id,
                    "contract_id": run.contract_id,
                    "delivered": bool(run.delivered), "reasons": [],
                    "replayed": True}
        if run.status != "completed":
            return {"ok": False, "run_id": run.id,
                    "contract_id": run.contract_id, "delivered": False,
                    "reasons": [f"run is {run.status} — only a "
                                "completed run collects"],
                    "replayed": False}
        contract = self.store.get_contract(run.contract_id)
        if contract is None:
            return {"ok": False, "run_id": run.id,
                    "contract_id": run.contract_id, "delivered": False,
                    "reasons": [f"contract {run.contract_id} is gone"],
                    "replayed": False}
        if not contract.terminal:
            blockers = self.store.completion_blockers(contract)
            return {"ok": False, "run_id": run.id,
                    "contract_id": contract.id, "delivered": False,
                    "reasons": blockers or
                    [f"contract is {contract.status}"],
                    "replayed": False}
        result = run.result_dict().get("result", "")
        summary = json.dumps(result, default=str)[:1000]
        title = f"delegated work done: {contract.title}"
        lines = [
            f"run {run.id} · contract {contract.id} · "
            f"role {run.role} · generation {run.generation}",
            f"worker: {run.worker}",
            f"result: {summary}",
        ]
        if run.mission_id:
            lines.append(f"mission: {run.mission_id}")
        if run.notes:
            lines.append("steering applied: "
                         + "; ".join(run.notes))
        body = "\n".join(lines)
        reasons: list[str] = []
        delivered = False
        try:
            outcome = self._notify("delegation", title, body)
            delivered = bool(outcome.get("delivered"))
            if not delivered:
                detail = (outcome.get("reason")
                          or outcome.get("delivery_state")
                          or "no channel")
                reasons.append(
                    f"delivery did not reach a channel ({detail}) "
                    "— the result stays on the run record")
        except Exception as exc:  # noqa: BLE001 - delivery never blocks
            record_failure("delegation_manager.collect.notify", exc)
            reasons.append(
                f"delivery failed: {exc} — the result stays on the "
                "run record")
        self._update_run(run.id, status="collected",
                         delivered=1 if delivered else 0)
        return {"ok": True, "run_id": run.id,
                "contract_id": contract.id, "delivered": delivered,
                "reasons": reasons, "replayed": False}

    # -- plan -------------------------------------------------------------------

    def plan_and_spawn(self, goal: str, *, mission_id: str = "",
                       chat_key: str = "",
                       context_hint: str = "") -> dict[str, Any]:
        """Decompose a goal and spawn what the roster can run.

        Planning reuses ``MasterOrchestrator.plan`` (model when one
        is configured, the named template plan otherwise — the
        degradation rides in ``plan_error``, never silent). Each
        step maps through :data:`PLAN_ROLE_TO_ROSTER`; steps with no
        roster worker, or with no runnable input (an implementer
        step naming no target file, a critic step naming no diff),
        are reported ``unassigned`` with the reason named — never
        silently run by the wrong worker. Steps are spawned
        independently and in parallel; ordering between steps (a
        verify step reviewing an implement step's diff) is the
        orchestrator's own run, not this manager's job.
        """
        from .orchestrator import MasterOrchestrator

        goal = " ".join(str(goal or "").split()).strip()
        if not goal:
            raise DelegationError("plan needs a goal — what is the "
                                  "work?")
        plan = MasterOrchestrator(self.context).plan(
            goal, context_hint=context_hint)
        runs: list[DelegationRun] = []
        unassigned: list[dict[str, Any]] = []
        for step in plan.steps:
            plan_role = (step.role or "").strip().lower()
            roster_role = PLAN_ROLE_TO_ROSTER.get(plan_role)
            if roster_role is None:
                unassigned.append(
                    {"step": step.name, "role": step.role,
                     "reason": f"no roster worker for plan role "
                               f"{step.role!r}"})
                continue
            try:
                input_dict = self._plan_step_input(roster_role, step)
            except DelegationError as exc:
                unassigned.append({"step": step.name, "role": step.role,
                                   "reason": str(exc)})
                continue
            try:
                runs.append(self.spawn(
                    role=roster_role, goal=step.goal,
                    input=input_dict, mission_id=mission_id,
                    scope=tuple(step.payload.get("writes") or ()),
                    chat_key=chat_key, name=step.name))
            except DelegationError as exc:
                unassigned.append({"step": step.name, "role": step.role,
                                   "reason": str(exc)})
        return {"plan": plan.to_dict(), "plan_error": plan.plan_error,
                "runs": runs, "unassigned": unassigned}

    @staticmethod
    def _plan_step_input(roster_role: str, step: Any) -> dict[str, Any]:
        goal = str(step.goal or "")
        if roster_role == "planner":
            return {"task": goal}
        if roster_role == "implementer":
            writes = list(step.payload.get("writes") or ())
            if not writes:
                raise DelegationError(
                    f"implementer step {step.name!r} names no target "
                    "file (no writes) — it cannot run")
            return {"path": str(writes[0]), "instruction": goal,
                    "file_content": ""}
        if roster_role == "reviewer":
            raise DelegationError(
                f"reviewer step {step.name!r} names no diff to "
                "review — it cannot run until a worker produces one")
        raise DelegationError(
            f"no plan-step input for roster role {roster_role!r}")

    # -- presentation -------------------------------------------------------------

    def summary(self, run: DelegationRun) -> dict[str, Any]:
        data = run.to_dict()
        data["result"] = run.result_dict()
        return data

    def status_text(self, run_id: str) -> str:
        run = self.get_run(run_id)
        if run is None:
            return f"no delegation run {run_id!r} — /delegate runs lists them."
        contract = self.store.get_contract(run.contract_id)
        lines = [
            f"run {run.id} [{run.status}] — role {run.role} · "
            f"worker {run.worker or '?'} · generation {run.generation}",
            f"contract {run.contract_id} "
            f"[{contract.status if contract else 'gone'}]",
        ]
        if contract is not None:
            lines.append(f"goal: {contract.goal}")
            blockers = [] if contract.terminal \
                else self.store.completion_blockers(contract)
            if blockers:
                lines.append("contract blocked by: " + "; ".join(blockers))
        if run.notes:
            lines.append("steering: " + "; ".join(run.notes))
        if run.error:
            lines.append(f"error: {run.error[:400]}")
        result = run.result_dict().get("result", "")
        if result != "":
            lines.append(
                "result: " + json.dumps(result, default=str)[:600])
        lines.append(
            f"delivered: {'yes' if run.delivered else 'no'}")
        return "\n".join(lines)

    def wait(self, run_id: str, *, timeout: float = 30.0
             ) -> DelegationRun:
        """Block (bounded) until a run leaves the live states."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            run = self.get_run(run_id)
            if run is None:
                raise DelegationError(f"no delegation run {run_id!r}")
            if run.status not in _LIVE_STATUSES:
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"delegation run {run_id} still {run.status} "
                    f"after {timeout}s")
            time.sleep(0.05)

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the manager's executor (tests / shutdown only)."""
        if self._owns_executor:
            with contextlib.suppress(Exception):
                self._executor.shutdown(wait=wait, cancel_futures=True)


# -- scheduler tick ---------------------------------------------------------

DELEGATION_JOB = "delegation manager tick"
DELEGATION_SPEC = "every 5m"
DELEGATION_TOOL = "__delegation_manager_tick__"

_tick_managers: dict[int, DelegationManager] = {}
_tick_managers_lock = threading.Lock()


def _tick_manager(context: Any) -> DelegationManager:
    """The process-long manager the tick reuses.

    One manager per database: a fresh manager per tick would leak a
    thread pool per tick (workers run async past the tick's return).
    ``from_context`` stays fresh — restart recovery needs a NEW
    manager, and the tick's singleton is exactly the long-lived
    process shape, not a restart.
    """
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        raise DelegationError(
            "delegation manager needs the database, which this "
            "context doesn't carry")
    key = id(db)
    with _tick_managers_lock:
        manager = _tick_managers.get(key)
        if manager is None:
            manager = DelegationManager.from_context(context)
            _tick_managers[key] = manager
        return manager


def ensure_delegation_manager_job(scheduler: Any, *,
                                  spec: str = DELEGATION_SPEC) -> Any:
    """Register the durable delegation-manager tick job.

    Idempotent — mirrors ``ensure_group_roster_diff_job``: an
    existing job at the requested cadence is returned as-is, a stale
    one is replaced. Best-effort: never raises.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == DELEGATION_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(DELEGATION_JOB, spec, "tool",
                             {"tool": DELEGATION_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def run_delegation_manager_tick_once(context: Any) -> dict[str, Any]:
    """Run one delegation-manager tick. Never raises."""
    try:
        manager = _tick_manager(context)
    except DelegationError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("delegation_manager.tick.manager", exc)
        return {"ok": False,
                "error": f"delegation tick failed: {exc}"}
    try:
        out = manager.poll()
        return {"ok": True, **out}
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("delegation_manager.tick", exc)
        return {"ok": False,
                "error": f"delegation tick failed: {exc}"}
