"""Mission state and persistence.

A mission is a goal that outlives the process working on it. That single property
drives every design decision here:

- **State is a JSON blob, not columns.** A mission's working state is whatever the
  planner and orchestrator happened to produce. Modelling it as columns would mean
  a migration every time the plan shape changes.
- **Checkpoints are append-only.** Rewriting a checkpoint in place means a crash
  during the write loses both the old and the new state. Appending means the worst
  case is a torn final row, which is discarded on load.
- **Budgets live in the row.** Spent wall-clock and tokens are persisted so a
  resumed mission cannot launder its budget by restarting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.tasks import AcceptanceCriterion
from ..storage.repository import Repository
from .progress import (
    clear_stall as _clear_stall_entry,
    estimate_eta,
    eta_breakdown,
    fmt_duration,
    real_plan_steps,
    record_stall as _record_stall_entry,
)
from .progress import _step_name as _plan_step_name

__all__ = [
    "MissionStatus",
    "Mission",
    "MissionStore",
    "Checkpoint",
    "ACCEPTANCE_STATE_KEY",
    "normalize_acceptance",
    "mission_liveness",
    "MISSION_TEMPLATES",
    "list_mission_templates",
]

_log = get_logger(__name__)

#: State key under which a mission's acceptance criteria live. Criteria are
#: stored as plain JSON (``MissionAcceptance.to_dict()`` shape) so no storage
#: migration is needed — the runner verifies against them at finish time and
#: transitions RUNNING -> VERIFYING -> COMPLETED/FAILED instead of completing
#: blind. Missions without this key keep the historical direct path.
ACCEPTANCE_STATE_KEY = "acceptance"

#: How long a mission may claim "running" without a runner heartbeat
#: before ``MissionStore.reconcile`` treats the runner as dead (seconds).
#: Overridable per call and via NM_MISSION_STALE_HEARTBEAT_S (floor 60s).
STALE_HEARTBEAT_SECONDS = 900.0


def _stale_heartbeat_after() -> float:
    try:
        return max(
            60.0,
            float(os.environ.get("NM_MISSION_STALE_HEARTBEAT_S", "")
                  or STALE_HEARTBEAT_SECONDS),
        )
    except (TypeError, ValueError):
        return STALE_HEARTBEAT_SECONDS


def _process_alive(pid: int) -> bool:
    """True when the OS still has this pid. Never raises."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    except (OSError, ValueError, TypeError):
        return False
    return True


def mission_liveness(mission: "Mission",
                     *, stale_after: float | None = None) -> dict[str, Any]:
    """Heartbeat/process liveness probe for one mission.

    The single source of truth shared by ``MissionStore.reconcile`` and
    ``MissionRunner.self_heal``: a worker counts as alive when its process
    is alive *or* its heartbeat is fresh. Returns ``{"alive",
    "heartbeat_age_s", "process_alive"}``. Never raises.
    """
    limit = stale_after if stale_after is not None else _stale_heartbeat_after()
    hb = (mission.state or {}).get("heartbeat") or {}
    try:
        age = max(0.0, time.time() - float(hb.get("at") or 0.0))
    except (TypeError, ValueError):
        age = float("inf")
    pid = hb.get("pid")
    process_alive = _process_alive(pid) if isinstance(pid, int) and pid > 0 else False
    return {
        "alive": bool(process_alive or age <= limit),
        "heartbeat_age_s": round(age, 1),
        "process_alive": process_alive,
    }


def normalize_acceptance(acceptance: Any) -> dict[str, Any]:
    """Validate + canonicalize acceptance criteria into the persisted shape.

    Accepts a plain dict in ``MissionAcceptance.to_dict()`` shape (or any
    object with a ``to_dict()`` of that shape — duck-typed so missions/L5
    never imports ``nomorals.os.mission_state``/L6). Criteria are
    canonicalized through ``AcceptanceCriterion`` (L1). Raises
    :class:`ValidationError` on any bad shape — fail fast at creation, not
    at verification time. An empty criteria set is rejected: a mission with
    nothing to verify must not pay the VERIFYING detour.
    """
    to_dict = getattr(acceptance, "to_dict", None)
    data = to_dict() if callable(to_dict) else acceptance
    if not isinstance(data, dict):
        raise ValidationError(
            f"acceptance must be a dict, got {type(acceptance).__name__}",
            field="acceptance",
        )
    raw_criteria = data.get("criteria") or []
    if not isinstance(raw_criteria, list):
        raise ValidationError("acceptance.criteria must be a list",
                              field="acceptance")

    criteria: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_criteria):
        if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
            raise ValidationError(
                f"acceptance.criteria[{index}] needs a name",
                field="acceptance",
            )
        try:
            spec = raw.get("spec")
            if spec is not None and not isinstance(spec, dict):
                raise ValidationError(
                    f"acceptance.criteria[{index}].spec must be a dict",
                    field="acceptance",
                )
            criteria.append(AcceptanceCriterion.from_dict(raw).to_dict())
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize or reject
            raise ValidationError(
                f"acceptance.criteria[{index}] is malformed: {exc}",
                field="acceptance",
            ) from exc
    required_types = data.get("required_artifact_types") or []
    if (not isinstance(required_types, (list, tuple))
            or any(not str(t or "").strip() for t in required_types)):
        raise ValidationError(
            "acceptance.required_artifact_types must be a list of non-empty strings",
            field="acceptance",
        )
    if not criteria and not required_types:
        raise ValidationError(
            "acceptance needs at least one criterion or required artifact type",
            field="acceptance",
        )
    return {
        "criteria": criteria,
        "required_artifact_types": [str(t) for t in required_types],
    }


class MissionStatus:
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({DONE, FAILED, CANCELLED})
    ALL = frozenset({PENDING, RUNNING, PAUSED, DONE, FAILED, CANCELLED})


@dataclass
class Mission:
    """A long-running goal with a budget and resumable state.

    ``priority`` / ``tags`` / ``parent_id`` ride inside the ``metadata``
    JSON column (zero-migration): ``to_row`` merges them in, ``from_row``
    pops them back out, so the persisted shape never changes.
    """

    goal: str
    name: str = ""
    id: str = field(default_factory=new_id)
    status: str = MissionStatus.PENDING
    state: dict[str, Any] = field(default_factory=dict)
    budget_wall: float = 0.0
    budget_tokens: int = 0
    spent_wall: float = 0.0
    spent_tokens: int = 0
    iterations: int = 0
    success: float | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: scheduling priority — higher runs first in priority-aware listings.
    priority: int = 0
    #: free-form labels ("nightly", "research", "trading", ...).
    tags: list[str] = field(default_factory=list)
    #: id of the parent mission when this is a sub-mission.
    parent_id: str = ""

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValidationError("a mission needs a goal", field="goal")
        self.name = self.name or self.goal[:60]
        try:
            self.priority = int(self.priority or 0)
        except (TypeError, ValueError):
            self.priority = 0
        self.tags = [str(t) for t in (self.tags or []) if str(t or "").strip()]
        self.parent_id = str(self.parent_id or "")

    @property
    def terminal(self) -> bool:
        return self.status in MissionStatus.TERMINAL

    @property
    def acceptance(self) -> dict[str, Any] | None:
        """Persisted acceptance criteria (``MissionAcceptance`` dict shape),
        or None when the mission completes without verification."""
        data = (self.state or {}).get(ACCEPTANCE_STATE_KEY)
        return dict(data) if isinstance(data, dict) else None

    @property
    def budget_exhausted(self) -> bool:
        if self.budget_wall and self.spent_wall >= self.budget_wall:
            return True
        if self.budget_tokens and self.spent_tokens >= self.budget_tokens:
            return True
        return False

    @property
    def wall_remaining(self) -> float:
        if not self.budget_wall:
            return float("inf")
        return max(0.0, self.budget_wall - self.spent_wall)

    def charge(self, *, wall: float = 0.0, tokens: int = 0) -> None:
        self.spent_wall += max(0.0, wall)
        self.spent_tokens += max(0, tokens)

    def to_row(self) -> dict[str, Any]:
        # priority/tags/parent_id persist inside the metadata JSON column —
        # no storage migration needed for the new fields.
        metadata = dict(self.metadata or {})
        metadata["priority"] = int(self.priority or 0)
        metadata["tags"] = list(self.tags or [])
        metadata["parent_id"] = str(self.parent_id or "")
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "status": self.status,
            "state": self.state,
            "budget_wall": self.budget_wall,
            "budget_tokens": self.budget_tokens,
            "spent_wall": self.spent_wall,
            "spent_tokens": self.spent_tokens,
            "iterations": self.iterations,
            "success": self.success,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "metadata": metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Mission":
        metadata = _json(row.get("metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)
        priority = metadata.pop("priority", 0)
        tags = metadata.pop("tags", [])
        parent_id = metadata.pop("parent_id", "")
        try:
            priority = int(priority or 0)
        except (TypeError, ValueError):
            priority = 0
        return cls(
            id=row["id"],
            name=row.get("name") or "",
            goal=row.get("goal") or "",
            status=row.get("status") or MissionStatus.PENDING,
            state=_json(row.get("state"), {}),
            budget_wall=float(row.get("budget_wall") or 0.0),
            budget_tokens=int(row.get("budget_tokens") or 0),
            spent_wall=float(row.get("spent_wall") or 0.0),
            spent_tokens=int(row.get("spent_tokens") or 0),
            iterations=int(row.get("iterations") or 0),
            success=None if row.get("success") is None else float(row["success"]),
            created_at=float(row.get("created_at") or 0.0),
            updated_at=float(row.get("updated_at") or 0.0),
            finished_at=None if row.get("finished_at") is None else float(row["finished_at"]),
            metadata=metadata,
            priority=priority,
            tags=list(tags) if isinstance(tags, (list, tuple)) else [],
            parent_id=str(parent_id or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_row(), "terminal": self.terminal,
                "budget_exhausted": self.budget_exhausted}

    @property
    def is_child(self) -> bool:
        """True when this mission was spawned by a parent mission."""
        return bool(self.parent_id)

    def summary_card(self) -> str:
        """One-glance boxed card for chat/terminal (see progress)."""
        from .progress import render_mission_card

        return render_mission_card(self)


@dataclass
class Checkpoint:
    """An immutable snapshot of mission state."""

    mission_id: str
    state: dict[str, Any]
    label: str = ""
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "label": self.label,
            "state": self.state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Checkpoint":
        return cls(
            id=row["id"],
            mission_id=row["mission_id"],
            label=row.get("label") or "",
            state=_json(row.get("state"), {}),
            created_at=float(row.get("created_at") or 0.0),
        )


def _json(raw: Any, default: Any) -> Any:
    """Decode a JSON column, tolerating the row already being decoded."""
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        _log.warning("could not decode JSON column, using default: %r", str(raw)[:80])
        return default


#: Canned mission shapes. Each template ships a name, default tags, and a
#: plan skeleton — a list of step dicts in the persisted plan shape
#: (``name`` / ``goal`` / ``role`` / ``kind`` / ``depends_on`` / ``policy``).
#: The shapes mirror the golden drills, so a templated mission exercises a
#: known-good pattern instead of an ad-hoc plan. ``create_from_template``
#: materializes one into ``state["plan"]`` so the runner executes it
#: without calling the planner.
MISSION_TEMPLATES: dict[str, dict[str, Any]] = {
    "research_write_verify": {
        "name": "Research → write → verify",
        "tags": ["research"],
        "steps": [
            {"name": "collect", "goal": "gather facts and sources",
             "role": "research", "kind": "io", "depends_on": [],
             "policy": {"retries": 2, "retry_on": "transient"}},
            {"name": "draft", "goal": "write the deliverable from the collected facts",
             "role": "execution", "kind": "io", "depends_on": ["collect"],
             "policy": {"retries": 1}},
            {"name": "verify", "goal": "verify the deliverable covers every collected fact",
             "role": "critic", "kind": "io", "depends_on": ["draft"],
             "policy": {"on_failure": "fail_fast"}},
        ],
    },
    "build_test_fix": {
        "name": "Build → test → fix",
        "tags": ["build"],
        "steps": [
            {"name": "scaffold", "goal": "write the code under construction",
             "role": "execution", "kind": "io", "depends_on": [],
             "policy": {"retries": 1}},
            {"name": "test", "goal": "run the test suite and report failures",
             "role": "execution", "kind": "cpu", "depends_on": ["scaffold"],
             "policy": {"retries": 1, "retry_on": "transient",
                        "timeout_s": 600}},
            {"name": "fix", "goal": "fix failing tests until green",
             "role": "execution", "kind": "io", "depends_on": ["test"],
             "policy": {"on_failure": "fail_fast"}},
        ],
    },
    "audit_remediate_rescan": {
        "name": "Audit → remediate → re-scan",
        "tags": ["audit"],
        "steps": [
            {"name": "audit", "goal": "scan the target and list every issue found",
             "role": "research", "kind": "io", "depends_on": [],
             "policy": {"retries": 1}},
            {"name": "remediate", "goal": "fix every issue the audit found",
             "role": "execution", "kind": "io", "depends_on": ["audit"],
             "policy": {"retries": 2}},
            {"name": "rescan", "goal": "re-scan and prove zero issues remain",
             "role": "critic", "kind": "io", "depends_on": ["remediate"],
             "policy": {"on_failure": "fail_fast"}},
        ],
    },
    "monitor_watch": {
        "name": "Watch → report",
        "tags": ["monitor"],
        "steps": [
            {"name": "watch", "goal": "observe the target and collect observations",
             "role": "research", "kind": "async", "depends_on": [],
             "policy": {"retries": 3, "retry_on": "transient",
                        "retry_backoff_s": 30, "timeout_s": 3600}},
            {"name": "report", "goal": "summarize the observations into a report",
             "role": "execution", "kind": "io", "depends_on": ["watch"],
             "policy": {}},
        ],
    },
}


def list_mission_templates() -> list[dict[str, Any]]:
    """Template catalog: key, name, tags, step names."""
    return [
        {"key": key, "name": spec.get("name", key),
         "tags": list(spec.get("tags") or []),
         "steps": [s.get("name", "") for s in spec.get("steps", [])]}
        for key, spec in MISSION_TEMPLATES.items()
    ]


class MissionStore:
    """Persistence for missions and their checkpoints."""

    def __init__(self, db: Any, *, checkpoint_limit: int = 50) -> None:
        self.db = db
        self.missions = Repository(db, "missions", json_columns=("state", "metadata"))
        # Checkpoints are append-only and immutable, so the table has no
        # updated_at column. The Repository default would inject one and fail.
        self.checkpoints = Repository(
            db,
            "mission_checkpoints",
            json_columns=("state",),
            timestamp_columns=("created_at",),
        )
        self.checkpoint_limit = checkpoint_limit

    # ── missions ─────────────────────────────────────────────────────────────

    def create(self, mission: Mission) -> Mission:
        self.missions.create(mission.to_row())
        _log.info("mission %s created: %s", mission.id, mission.goal[:60])
        return mission

    def create_new(
        self,
        goal: str,
        *,
        name: str = "",
        budget_wall: float = 0.0,
        budget_tokens: int = 0,
        state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        acceptance: dict[str, Any] | Any | None = None,
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: str = "",
    ) -> Mission:
        """Create + persist a mission.

        ``acceptance`` (optional) is validated up front by
        :func:`normalize_acceptance` and stored under
        ``state["acceptance"]`` — the runner then verifies the mission
        through VERIFYING before it may complete.
        """
        state = dict(state or {})
        if acceptance is not None:
            state[ACCEPTANCE_STATE_KEY] = normalize_acceptance(acceptance)
        return self.create(
            Mission(
                goal=goal,
                name=name,
                budget_wall=budget_wall,
                budget_tokens=budget_tokens,
                state=state,
                metadata=metadata or {},
                priority=priority,
                tags=tags or [],
                parent_id=parent_id,
            )
        )

    def create_from_template(
        self,
        template_key: str,
        goal: str,
        **kwargs: Any,
    ) -> Mission:
        """Materialize a canned mission shape (see ``MISSION_TEMPLATES``).

        The template's plan skeleton is written straight into
        ``state["plan"]`` in the persisted shape, so the runner executes it
        without consulting the planner. Raises :class:`ValidationError`
        for an unknown template key.
        """
        spec = MISSION_TEMPLATES.get(template_key)
        if spec is None:
            raise ValidationError(
                f"unknown mission template {template_key!r} — one of: "
                f"{sorted(MISSION_TEMPLATES)}",
                field="template",
            )
        state = dict(kwargs.pop("state", None) or {})
        plan = []
        for entry in spec.get("steps", []):
            step = {
                "name": entry.get("name", "step"),
                "goal": entry.get("goal", ""),
                "role": entry.get("role", "execution"),
                "kind": entry.get("kind", "io"),
                "depends_on": list(entry.get("depends_on") or []),
            }
            if entry.get("policy"):
                step["policy"] = dict(entry["policy"])
            plan.append(step)
        state["plan"] = plan
        tags = list(kwargs.pop("tags", None) or []) or list(spec.get("tags") or [])
        name = kwargs.pop("name", "") or spec.get("name", template_key)
        mission = self.create_new(
            goal, name=name, state=state, tags=tags, **kwargs)
        mission.state["template"] = template_key
        return self.save(mission)

    def spawn_child(self, parent_id: str, goal: str, **kwargs: Any) -> Mission:
        """Create a sub-mission under ``parent_id`` (Temporal child-workflow
        style). The child inherits the parent's tags (plus ``"child"``)
        and gets its own fresh budget; ``children()`` / ``descendants()``
        walk the tree. Raises :class:`NotFound` for an unknown parent and
        :class:`ValidationError` when the parent is terminal.
        """
        parent = self.get(parent_id)  # raises NotFound when unknown
        if parent.terminal:
            raise ValidationError(
                f"mission {parent_id} is {parent.status}: "
                "a terminal mission cannot spawn children")
        tags = list(kwargs.pop("tags", None) or []) or list(parent.tags)
        if "child" not in tags:
            tags.append("child")
        return self.create_new(goal, parent_id=parent.id, tags=tags, **kwargs)

    def children(self, mission_id: str) -> list[Mission]:
        """Direct sub-missions of ``mission_id``."""
        self.get(mission_id)  # raises NotFound when unknown
        try:
            rows = self.db.query(
                "SELECT * FROM missions ORDER BY updated_at DESC")
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in rows:
            try:
                mission = Mission.from_row(dict(r))
            except Exception:  # noqa: BLE001 - a bad row is not our problem
                continue
            if mission.parent_id == mission_id:
                out.append(mission)
        return out

    def descendants(self, mission_id: str) -> list[Mission]:
        """All sub-missions transitively (breadth-first)."""
        seen: set[str] = set()
        out: list[Mission] = []
        queue = [mission_id]
        while queue:
            current = queue.pop(0)
            for child in self.children(current):
                if child.id in seen:
                    continue
                seen.add(child.id)
                out.append(child)
                queue.append(child.id)
        return out

    def tree(self, mission_id: str) -> dict[str, Any]:
        """The mission plus its whole sub-mission tree as nested dicts."""
        root = self.get(mission_id)  # raises NotFound when unknown

        def _node(mission: Mission) -> dict[str, Any]:
            return {
                "mission": mission.to_dict(),
                "children": [_node(c) for c in self.children(mission.id)],
            }

        return _node(root)

    def get(self, mission_id: str) -> Mission:
        row = self.missions.get(mission_id)
        if row is None:
            raise NotFound(f"no mission {mission_id!r}")
        return Mission.from_row(row)

    def save(self, mission: Mission) -> Mission:
        """Persist the whole row. Callers set status and spent counters first."""
        mission.updated_at = time.time()
        if mission.terminal and mission.finished_at is None:
            mission.finished_at = mission.updated_at
        self.missions.update(mission.id, mission.to_row())
        return mission

    def progress(self, mission_id: str) -> dict[str, Any]:
        """Step progress for the CLI: done / total / percent / current step.

        Total comes from the persisted plan (``state["plan"]``), minus the
        ``__plan_error__`` degradation marker which is not a step. When no
        plan was stored yet the mission simply has no measurable total.
        Also carries the spend counters and last error so the chat status
        command and the Devon agent's progress report read from one place.
        """
        mission = self.get(mission_id)  # raises NotFound when unknown
        plan = real_plan_steps(mission.state.get("plan"))
        completed = set(mission.state.get("completed_steps") or [])
        total = len(plan)
        done = len([s for s in plan
                    if _plan_step_name(s) in completed]) if total else len(completed)
        current = ""
        for step in plan:
            name = _plan_step_name(step)
            if name and name not in completed:
                current = name
                break
        return {"steps_done": done, "total_steps": total,
                "percent": (100.0 * done / total) if total else 0.0,
                "current_step": current,
                "spent_wall_seconds": mission.spent_wall,
                "spent_tokens": mission.spent_tokens,
                "last_error": str(mission.state.get("last_error") or "")}

    def detail(self, mission_id: str) -> dict[str, Any]:
        """Everything the chat status command needs in one call.

        ``progress`` + an honest ETA + the stall record (if any) + recent
        checkpoints + the milestone push log. Powers ``/mission status``
        and the Devon agent's progress report.
        """
        mission = self.get(mission_id)  # raises NotFound when unknown
        eta_seconds, eta_note = estimate_eta(mission, self.db)
        return {
            "mission": mission.to_dict(),
            "progress": self.progress(mission_id),
            "eta_seconds": eta_seconds,
            "eta_note": eta_note,
            # structured ETA: method ("ema" | "run-average" | "history" |
            # "none"), per-step rate, remaining count — powers the card
            # renderer and any consumer that wants more than a number.
            "eta_breakdown": eta_breakdown(mission, self.db),
            "stall": mission.state.get("stall"),
            # real step executions per step (idempotency replays don't
            # count) — powers "/mission status" retry visibility.
            "attempts": dict(mission.state.get("step_attempts") or {}),
            "recent_checkpoints": [
                c.to_row() for c in self.checkpoint_history(mission_id, limit=5)
            ],
            "milestone_log": list(mission.state.get("milestones") or []),
        }

    # ── stall tracking ───────────────────────────────────────────────────────
    #
    # A stalled mission says WHY it is stalled: a StallCode plus a concrete
    # message, the step it was on, and the timestamp. Never a bare
    # "waiting". The runner records stalls automatically (consecutive
    # failures, budget exhaustion); mark_stalled is the explicit operator
    # API for "waiting on provider X" / "blocked on approval Y".

    def mark_stalled(
        self,
        mission_id: str,
        code: str,
        message: str,
        *,
        step: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a concrete stall reason. Returns the entry + whether it
        changed (a repeat of the identical stall is not a new event)."""
        mission = self.get(mission_id)  # raises NotFound when unknown
        if mission.terminal:
            raise ValidationError(
                f"mission {mission_id} is {mission.status}: "
                "a terminal mission cannot stall")
        changed = _record_stall_entry(mission, code, message, step=step, extra=extra)
        self.save(mission)
        return {"mission_id": mission_id, "changed": changed,
                "stall": mission.state.get("stall")}

    def clear_stall(self, mission_id: str) -> bool:
        """Drop the stall record. Returns True when one was present."""
        mission = self.get(mission_id)  # raises NotFound when unknown
        cleared = _clear_stall_entry(mission)
        if cleared:
            self.save(mission)
        return cleared

    def stall(self, mission_id: str) -> dict[str, Any] | None:
        """The current stall record, or None when the mission isn't stalled."""
        return self.get(mission_id).state.get("stall")

    def set_status(self, mission_id: str, status: str, note: str = "") -> Mission:
        """Cross-process status flip (pause / cancel / resume-status).

        Refuses to move a terminal mission back to a live state — resume
        goes through ``MissionRunner.resume`` so checkpoints are honoured.
        """
        if status not in MissionStatus.ALL:
            raise ValidationError(f"unknown mission status {status!r}")
        mission = self.get(mission_id)  # raises NotFound when unknown
        if mission.terminal and status not in MissionStatus.TERMINAL:
            raise ValidationError(
                f"mission {mission_id} is {mission.status}: "
                "terminal missions cannot be reactivated")
        mission.status = status
        if note:
            mission.state["status_note"] = note
        return self.save(mission)

    def reconcile(
        self, mission_id: str, *, stale_after: float | None = None
    ) -> dict[str, Any]:
        """Verify a mission that claims to be running is actually alive.

        A mission survives ``kill -9``, so a crash can leave
        ``status="running"`` with nobody driving it. On read we check the
        runner heartbeat (``MissionRunner`` writes ``state["heartbeat"]``
        — pid + timestamp — at start and after every step) plus the OS
        process: a stale heartbeat *and* a gone process means the runner
        died, so the mission flips to ``failed`` with an explicit reason
        instead of reporting "running" forever. A stale heartbeat with a
        *live* process is just a long step — it stays running.

        Returns ``{"mission_id", "changed", "status", "reason"}``;
        ``changed`` is True only when the status actually moved. Never
        touches non-running missions.
        """
        mission = self.get(mission_id)  # raises NotFound when unknown
        if mission.status != MissionStatus.RUNNING:
            return {"mission_id": mission_id, "changed": False,
                    "status": mission.status, "reason": ""}
        live = mission_liveness(mission, stale_after=stale_after)
        if live["alive"]:
            return {"mission_id": mission_id, "changed": False,
                    "status": MissionStatus.RUNNING,
                    "heartbeat_age_s": live["heartbeat_age_s"],
                    "process_alive": live["process_alive"], "reason": ""}
        pid = ((mission.state or {}).get("heartbeat") or {}).get("pid")
        reason = (
            f"runner died — no heartbeat for {fmt_duration(live['heartbeat_age_s'])}"
            + (f" (pid {pid} is gone)" if pid else " (no heartbeat ever recorded)")
        )
        mission.status = MissionStatus.FAILED
        mission.state["last_error"] = reason
        mission.state["status_note"] = f"reconciled: {reason}"
        self.save(mission)
        _log.warning("mission %s reconciled running->failed: %s",
                     mission.id, reason)
        return {"mission_id": mission_id, "changed": True,
                "status": MissionStatus.FAILED, "reason": reason}

    def list(
        self,
        *,
        status: str = "",
        active_only: bool = False,
        limit: int = 50,
        tag: str = "",
        min_priority: int | None = None,
        search: str = "",
        order: str = "updated",
    ) -> list[Mission]:
        """List missions, newest first.

        ``tag`` / ``min_priority`` / ``search`` filter in Python over the
        metadata JSON (zero-migration: no SQL JSON operators needed).
        ``search`` matches name + goal case-insensitively. ``order`` is
        ``"updated"`` (default) or ``"priority"`` (priority desc, then
        updated desc).
        """
        if active_only:
            placeholders = ",".join("?" for _ in MissionStatus.TERMINAL)
            rows = self.db.query(
                f"SELECT * FROM missions WHERE status NOT IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT ?",
                (*sorted(MissionStatus.TERMINAL), limit),
            )
        elif status:
            rows = self.db.query(
                "SELECT * FROM missions WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM missions ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        missions = []
        for r in rows:
            try:
                missions.append(Mission.from_row(dict(r)))
            except Exception:  # noqa: BLE001 - a bad row is not our problem
                continue
        tag = (tag or "").strip().lower()
        needle = (search or "").strip().lower()
        if tag:
            missions = [m for m in missions
                        if any(t.lower() == tag for t in m.tags)]
        if min_priority is not None:
            try:
                floor = int(min_priority)
            except (TypeError, ValueError):
                floor = 0
            missions = [m for m in missions if m.priority >= floor]
        if needle:
            missions = [m for m in missions
                        if needle in (m.name or "").lower()
                        or needle in (m.goal or "").lower()]
        if order == "priority":
            missions.sort(key=lambda m: (-m.priority, -(m.updated_at or 0.0)))
        return missions[: max(0, limit)]

    def search(self, query: str, *, status: str = "",
               limit: int = 50) -> list[Mission]:
        """Full-text-ish search over mission name + goal."""
        if not (query or "").strip():
            raise ValidationError("search needs a query", field="query")
        return self.list(status=status, search=query, limit=limit)

    def bulk_set_status(
        self,
        mission_ids: list[str],
        status: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Flip many missions at once. Returns ``{"updated", "skipped"}``
        where skipped maps id → reason. One bad id never aborts the batch.
        """
        if status not in MissionStatus.ALL:
            raise ValidationError(f"unknown mission status {status!r}")
        updated: list[str] = []
        skipped: dict[str, str] = {}
        for mission_id in mission_ids:
            try:
                mission = self.get(mission_id)  # raises NotFound when unknown
            except NotFound:
                skipped[mission_id] = "not found"
                continue
            if mission.terminal and status not in MissionStatus.TERMINAL:
                skipped[mission_id] = (
                    f"terminal ({mission.status}): cannot reactivate")
                continue
            mission.status = status
            if note:
                mission.state["status_note"] = note
            self.save(mission)
            updated.append(mission_id)
        return {"updated": updated, "skipped": skipped,
                "updated_count": len(updated)}

    def bulk_cancel(self, *, status: str = "", tag: str = "") -> dict[str, Any]:
        """Cancel every live mission (optionally filtered)."""
        live = [m for m in self.list(status=status, tag=tag, limit=10000)
                if not m.terminal]
        return self.bulk_set_status(
            [m.id for m in live], MissionStatus.CANCELLED,
            note="bulk cancel")

    def archive(self, older_than_seconds: float = 30 * 86400) -> dict[str, Any]:
        """Delete terminal missions (and their checkpoints + reflections)
        finished longer than ``older_than_seconds`` ago.

        Retention hygiene: a personal agent that never forgets a finished
        mission grows its DB forever. Returns counts per deleted table.
        Never touches live missions.
        """
        cutoff = time.time() - max(0.0, float(older_than_seconds))
        rows = self.db.query(
            "SELECT id, finished_at, updated_at FROM missions WHERE status IN "
            "('done', 'failed', 'cancelled')")
        doomed = [
            str(r["id"]) for r in rows
            if float(r["finished_at"] or r["updated_at"] or 0.0) < cutoff
        ]
        deleted = {"missions": 0, "checkpoints": 0, "reflections": 0}
        for mission_id in doomed:
            try:
                with self.db.transaction():
                    cp = self.db.execute(
                        "DELETE FROM mission_checkpoints WHERE mission_id = ?",
                        (mission_id,))
                    deleted["checkpoints"] += int(cp.rowcount or 0)
                    try:
                        rf = self.db.execute(
                            "DELETE FROM reflections WHERE mission_id = ?",
                            (mission_id,))
                        deleted["reflections"] += int(rf.rowcount or 0)
                    except Exception:  # noqa: BLE001 - table may not exist yet
                        pass
                    ms = self.db.execute(
                        "DELETE FROM missions WHERE id = ?", (mission_id,))
                    deleted["missions"] += int(ms.rowcount or 0)
            except Exception:  # noqa: BLE001 - one bad id never aborts the batch
                _log.warning("archive failed for mission %s", mission_id,
                             exc_info=True)
        _log.info("archived %d missions (%d checkpoints, %d reflections)",
                  deleted["missions"], deleted["checkpoints"],
                  deleted["reflections"])
        return deleted

    def resumable(self) -> list[Mission]:
        """Missions that were interrupted: running or paused, never terminal."""
        return self.list(active_only=True)

    def live_with_lock(self, lock_key: str) -> list[Mission]:
        """Live (non-terminal) missions holding this lock key.

        The runner uses it to refuse a second concurrent mission on the
        same lock — two missions racing the same goal is how duplicate
        side effects happen.  The lock key lives in ``metadata`` as a
        JSON string (``{"lock_key": ...}``); an empty key matches nothing.
        """
        lock_key = (lock_key or "").strip()
        if not lock_key:
            return []
        try:
            rows = self.db.query(
                "SELECT * FROM missions WHERE status NOT IN "
                "('done', 'failed', 'cancelled')"
            )
        except Exception:  # noqa: BLE001
            return []
        out = []
        for r in rows:
            try:
                mission = Mission.from_row(dict(r))
            except Exception:  # noqa: BLE001 - a bad row is not our problem
                continue
            if str((mission.metadata or {}).get("lock_key") or "") == lock_key:
                out.append(mission)
        return out

    def stats(self) -> dict[str, Any]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM missions GROUP BY status")
        by_status = {r["status"]: int(r["n"]) for r in rows}
        total = int(self.db.scalar("SELECT COUNT(*) FROM missions", default=0) or 0)
        avg = self.db.scalar(
            "SELECT AVG(success) FROM missions WHERE success IS NOT NULL", default=None
        )
        return {
            "total": total,
            "by_status": by_status,
            "active": sum(v for k, v in by_status.items() if k not in MissionStatus.TERMINAL),
            "avg_success": None if avg is None else round(float(avg), 4),
            "checkpoints": int(
                self.db.scalar("SELECT COUNT(*) FROM mission_checkpoints", default=0) or 0
            ),
        }

    # ── checkpoints ──────────────────────────────────────────────────────────

    def checkpoint(self, mission: Mission, *, label: str = "") -> Checkpoint:
        """Append a snapshot. Never rewrites: a crash mid-write must not lose both."""
        point = Checkpoint(
            mission_id=mission.id,
            state=dict(mission.state),
            label=label,
        )
        with self.db.transaction():
            self.checkpoints.create(point.to_row())
            self._trim_checkpoints(mission.id)
        _log.debug("mission %s checkpointed (%s)", mission.id, label or "unnamed")
        return point

    def latest_checkpoint(self, mission_id: str) -> Checkpoint | None:
        row = self.db.query_one(
            "SELECT * FROM mission_checkpoints WHERE mission_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (mission_id,),
        )
        return Checkpoint.from_row(dict(row)) if row is not None else None

    def checkpoint_history(self, mission_id: str, *, limit: int = 20) -> list[Checkpoint]:
        rows = self.db.query(
            "SELECT * FROM mission_checkpoints WHERE mission_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (mission_id, limit),
        )
        return [Checkpoint.from_row(dict(r)) for r in rows]

    def _trim_checkpoints(self, mission_id: str) -> None:
        """Keep the newest N. Unbounded history is a disk leak on a long mission."""
        if self.checkpoint_limit <= 0:
            return
        self.db.execute(
            "DELETE FROM mission_checkpoints WHERE mission_id = ? AND id NOT IN ("
            "  SELECT id FROM mission_checkpoints WHERE mission_id = ? "
            "  ORDER BY created_at DESC, rowid DESC LIMIT ?"
            ")",
            (mission_id, mission_id, self.checkpoint_limit),
        )

    # ── reflections ──────────────────────────────────────────────────────────

    def record_reflection(
        self,
        mission_id: str,
        *,
        score: float,
        summary: str = "",
        lessons: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> str:
        reflection_id = new_id()
        self.db.insert(
            "reflections",
            {
                "id": reflection_id,
                "mission_id": mission_id,
                "score": float(score),
                "summary": summary,
                "lessons": json.dumps(lessons or []),
                "weights": json.dumps(weights or {}),
                "created_at": time.time(),
            },
        )
        return reflection_id

    def reflections(self, mission_id: str = "", *, limit: int = 20) -> list[dict[str, Any]]:
        if mission_id:
            rows = self.db.query(
                "SELECT * FROM reflections WHERE mission_id = ? ORDER BY created_at DESC LIMIT ?",
                (mission_id, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["lessons"] = _json(data.get("lessons"), [])
            data["weights"] = _json(data.get("weights"), {})
            out.append(data)
        return out
