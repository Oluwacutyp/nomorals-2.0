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
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.repository import Repository

__all__ = ["MissionStatus", "Mission", "MissionStore", "Checkpoint"]

_log = get_logger(__name__)


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
    """A long-running goal with a budget and resumable state."""

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

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValidationError("a mission needs a goal", field="goal")
        self.name = self.name or self.goal[:60]

    @property
    def terminal(self) -> bool:
        return self.status in MissionStatus.TERMINAL

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
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Mission":
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
            metadata=_json(row.get("metadata"), {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_row(), "terminal": self.terminal, "budget_exhausted": self.budget_exhausted}


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
    ) -> Mission:
        return self.create(
            Mission(
                goal=goal,
                name=name,
                budget_wall=budget_wall,
                budget_tokens=budget_tokens,
                state=state or {},
                metadata=metadata or {},
            )
        )

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

        Total comes from the persisted plan (``state["plan"]``); when no
        plan was stored yet the mission simply has no measurable total.
        """
        mission = self.get(mission_id)  # raises NotFound when unknown
        plan = mission.state.get("plan") or []
        completed = set(mission.state.get("completed_steps") or [])
        total = len(plan)
        done = len([s for s in plan
                    if (s.get("name") if isinstance(s, dict) else s)
                    in completed]) if total else len(completed)
        current = ""
        for step in plan:
            name = step.get("name", "") if isinstance(step, dict) else str(step)
            if name and name not in completed:
                current = name
                break
        return {"steps_done": done, "total_steps": total,
                "percent": (100.0 * done / total) if total else 0.0,
                "current_step": current}

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

    def list(
        self,
        *,
        status: str = "",
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Mission]:
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
        return [Mission.from_row(dict(r)) for r in rows]

    def resumable(self) -> list[Mission]:
        """Missions that were interrupted: running or paused, never terminal."""
        return self.list(active_only=True)

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
