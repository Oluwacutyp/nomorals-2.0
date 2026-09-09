"""Training run tracking, and the gate between a trained model and production.

This is where the self-improvement loop either earns its keep or quietly rots the
system. A run is recorded as ``pending`` the moment it starts, so a crash leaves
evidence rather than a mystery. On completion the run is evaluated, and the
result decides promotion.

The gate is not advisory. ``promote()`` refuses to activate a model that did not
pass ``ModelRegistry.beats_incumbent()`` unless the caller explicitly overrides
with ``force=True`` — and an override is recorded in the run row, because a human
deciding to ship a worse model should be visible in the history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..llm.registry import ModelRegistry
from ..storage.repository import Repository

__all__ = ["TrainingRun", "TrainingRegistry"]

_log = get_logger(__name__)


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PROMOTED = "promoted"
    REJECTED = "rejected"

    TERMINAL = frozenset({DONE, FAILED, PROMOTED, REJECTED})


@dataclass
class TrainingRun:
    """One training attempt and everything needed to judge it."""

    name: str = ""
    base_model: str = ""
    output_model: str = ""
    dataset_id: str = ""
    backend: str = "native"
    id: str = field(default_factory=new_id)
    status: str = RunStatus.PENDING
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    epochs: float = 0.0
    output_path: str = ""
    gate_passed: bool | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Map to the training_runs columns.

        The table has no updated_at and no metadata column; it has started_at,
        finished_at, and error. Anything that does not fit a column is folded
        into the config JSON rather than dropped, because a forced promotion or a
        crash reason is exactly the history worth keeping.
        """
        return {
            "id": self.id, "name": self.name, "base_model": self.base_model,
            "output_model": self.output_model, "dataset_id": self.dataset_id,
            "backend": self.backend, "status": self.status,
            "config": {**self.config, "extra": self.metadata} if self.metadata else self.config,
            "metrics": self.metrics, "steps": self.steps, "epochs": self.epochs,
            "output_path": self.output_path,
            "gate_passed": None if self.gate_passed is None else int(self.gate_passed),
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TrainingRun":
        gate = row.get("gate_passed")
        config = _decode(row.get("config"))
        return cls(
            id=row["id"], name=row.get("name") or "",
            base_model=row.get("base_model") or "",
            output_model=row.get("output_model") or "",
            dataset_id=row.get("dataset_id") or "",
            backend=row.get("backend") or "native",
            status=row.get("status") or RunStatus.PENDING,
            config={k: v for k, v in config.items() if k != "extra"},
            metrics=_decode(row.get("metrics")),
            steps=int(row.get("steps") or 0), epochs=float(row.get("epochs") or 0.0),
            output_path=row.get("output_path") or "",
            gate_passed=None if gate is None else bool(gate),
            created_at=float(row.get("created_at") or 0.0),
            started_at=float(row.get("started_at") or 0.0),
            finished_at=float(row.get("finished_at") or 0.0),
            error=row.get("error") or "",
            metadata=config.get("extra") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_row()


def _decode(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class TrainingRegistry:
    """Run bookkeeping plus the promotion decision."""

    def __init__(self, db: Any, *, models: ModelRegistry | None = None) -> None:
        self.db = db
        self.repo = Repository(
            db, "training_runs", json_columns=("config", "metrics"),
            timestamp_columns=("created_at",),
        )
        self.models = models or ModelRegistry(db)

    def start(
        self,
        name: str = "",
        *,
        base_model: str = "",
        output_model: str = "",
        dataset_id: str = "",
        backend: str = "native",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        run = TrainingRun(
            name=name or f"run-{int(time.time())}",
            base_model=base_model, output_model=output_model or name or "personal",
            dataset_id=dataset_id, backend=backend,
            status=RunStatus.RUNNING, config=config or {}, metadata=metadata or {},
            started_at=time.time(),
        )
        self.repo.create(run.to_row())
        _log.info("training run %s started (backend=%s)", run.name, backend)
        return run

    def get(self, run_id: str) -> TrainingRun:
        row = self.repo.get(run_id) or self.repo.find_one(name=run_id)
        if row is None:
            raise NotFound(f"no training run {run_id!r}")
        return TrainingRun.from_row(row)

    def save(self, run: TrainingRun) -> TrainingRun:
        now = time.time()
        if run.status == RunStatus.RUNNING and not run.started_at:
            run.started_at = now
        if run.status in RunStatus.TERMINAL and not run.finished_at:
            run.finished_at = now
        self.repo.update(run.id, run.to_row())
        return run

    def complete(self, run: TrainingRun, *, metrics: dict[str, Any], output_path: str = "") -> TrainingRun:
        run.status = RunStatus.DONE
        run.metrics = metrics
        run.steps = int(metrics.get("steps") or run.steps)
        run.epochs = float(metrics.get("epochs") or run.epochs)
        if output_path:
            run.output_path = output_path
        return self.save(run)

    def fail(self, run: TrainingRun, *, error: str) -> TrainingRun:
        run.status = RunStatus.FAILED
        run.error = error[:500]
        _log.warning("training run %s failed: %s", run.name, error[:200])
        return self.save(run)

    def evaluate(self, run: TrainingRun, *, metric: str = "score", tolerance: float = 0.0) -> bool:
        """Record the metrics on the model and ask the gate whether it wins."""
        scores = {k: v for k, v in run.metrics.items() if isinstance(v, (int, float))}
        if not scores:
            raise ValidationError("run has no numeric metrics to evaluate", field="metrics")
        self.models.register(
            run.output_model,
            kind="finetune",
            base_model=run.base_model,
            path=run.output_path,
            metadata={"training_run": run.id, "backend": run.backend},
        )
        self.models.record_eval(run.output_model, scores)
        passed = self.models.beats_incumbent(run.output_model, metric, tolerance=tolerance)
        run.gate_passed = passed
        return self.save(run).gate_passed is True

    def promote(self, run: TrainingRun, *, force: bool = False) -> bool:
        """Activate the trained model. Refuses unless the gate passed."""
        if run.gate_passed is None:
            raise ValidationError(
                f"run {run.name} was never evaluated; call evaluate() first", field="gate_passed"
            )
        if not run.gate_passed and not force:
            run.status = RunStatus.REJECTED
            self.save(run)
            _log.info(
                "run %s rejected by the promotion gate (%s)",
                run.name, run.metrics.get("score"),
            )
            return False
        self.models.activate(run.output_model)
        run.status = RunStatus.PROMOTED
        if force and not run.gate_passed:
            # A human overrode the gate. That belongs in the record.
            run.metadata = {**run.metadata, "forced_promotion": True, "forced_at": time.time()}
            _log.warning("run %s force-promoted despite failing the gate", run.name)
        return self.save(run).status == RunStatus.PROMOTED

    def list(self, *, status: str = "", limit: int = 50) -> list[TrainingRun]:
        # Not repo.find(limit=...): that becomes `WHERE "limit" = 50` and returns
        # nothing, because find() treats every kwarg as a column filter.
        query = self.repo.query()
        if status:
            query.where("status = ?", status)
        rows = self.db.query(*query.order_by("created_at DESC").limit(limit).build())
        return [TrainingRun.from_row(dict(r)) for r in rows]

    def interrupted(self) -> list[TrainingRun]:
        """Runs left in flight by a crash. Startup should mark these failed."""
        rows = self.db.query(
            "SELECT * FROM training_runs WHERE status IN ('pending','running') "
            "ORDER BY created_at DESC LIMIT 100"
        )
        return [TrainingRun.from_row(dict(r)) for r in rows]

    def reap_interrupted(self) -> int:
        """Mark crashed runs failed so they do not look in-progress forever."""
        count = 0
        for run in self.interrupted():
            run.status = RunStatus.FAILED
            run.error = "process died before completion"
            self.save(run)
            count += 1
        if count:
            _log.warning("reaped %d interrupted training run(s)", count)
        return count

    def stats(self) -> dict[str, Any]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM training_runs GROUP BY status")
        by_status = {r["status"]: int(r["n"]) for r in rows}
        return {
            "total": int(self.db.scalar("SELECT COUNT(*) FROM training_runs", default=0) or 0),
            "by_status": by_status,
            "promoted": by_status.get(RunStatus.PROMOTED, 0),
            "rejected": by_status.get(RunStatus.REJECTED, 0),
            "gate_passed": int(
                self.db.scalar(
                    "SELECT COUNT(*) FROM training_runs WHERE gate_passed = 1", default=0
                ) or 0
            ),
        }
