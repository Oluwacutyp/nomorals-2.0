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

__all__ = ["TrainingRun", "TrainingRegistry", "RunStatus", "PROMOTION_STAGES"]

_log = get_logger(__name__)


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    # ── staged rollout + approval (mined from MLflow/champion-challenger
    # practice): a model can serve at 0% (shadow) or partial (canary)
    # traffic before full promotion, and promotion can require a human.
    SHADOW = "shadow"
    CANARY = "canary"
    AWAITING_APPROVAL = "awaiting_approval"

    TERMINAL = frozenset({DONE, FAILED, PROMOTED, REJECTED})


#: Valid promotion stages for :meth:`TrainingRegistry.promote`.
PROMOTION_STAGES = ("shadow", "canary", "full")


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

    def _model_score(self, model_name: str, metric: str) -> float | None:
        """The recorded ``metric`` for a registered model, or None."""
        if not model_name:
            return None
        record = self.models.by_name(model_name)
        if record is None:
            return None
        scores = getattr(record, "eval_scores", None) or {}
        if isinstance(scores, str):
            try:
                scores = json.loads(scores or "{}")
            except (json.JSONDecodeError, TypeError):
                scores = {}
        value = scores.get(metric) if isinstance(scores, dict) else None
        return float(value) if isinstance(value, (int, float)) else None

    def evaluate(
        self,
        run: TrainingRun,
        *,
        metric: str = "score",
        tolerance: float = 0.0,
        min_gain: float = 0.0,
    ) -> bool:
        """Record the metrics on the model and ask the gate whether it wins.

        ``min_gain`` is the champion-challenger margin: the challenger must
        beat the champion by at least this much, so a +0.0001 noise win
        never promotes.  The full comparison (scores, margin, champion
        name) is stored on ``run.metadata["gate"]`` for the audit trail.
        """
        scores = {k: v for k, v in run.metrics.items() if isinstance(v, (int, float))}
        if not scores:
            raise ValidationError("run has no numeric metrics to evaluate", field="metrics")
        champion = self.models.active()
        champion_name = champion.name if champion else ""
        champion_score = self._model_score(champion_name, metric)
        self.models.register(
            run.output_model,
            kind="finetune",
            base_model=run.base_model,
            path=run.output_path,
            metadata={"training_run": run.id, "backend": run.backend},
        )
        self.models.record_eval(run.output_model, scores)
        beats = self.models.beats_incumbent(run.output_model, metric, tolerance=tolerance)
        challenger_score = self._model_score(run.output_model, metric)
        margin: float | None = None
        if challenger_score is not None and champion_score is not None:
            margin = challenger_score - champion_score
        passed = beats and (margin is None or margin >= min_gain)
        run.gate_passed = passed
        run.metadata = {
            **run.metadata,
            "gate": {
                "metric": metric, "tolerance": tolerance, "min_gain": min_gain,
                "challenger": run.output_model, "challenger_score": challenger_score,
                "champion": champion_name or None, "champion_score": champion_score,
                "margin": margin, "beats_incumbent": beats,
                "evaluated_at": time.time(),
            },
        }
        self.save(run)
        _log.info(
            "gate %s for run %s (margin=%s, min_gain=%s)",
            "PASSED" if passed else "FAILED", run.name, margin, min_gain,
        )
        return passed

    def gate_report(self, run: TrainingRun) -> dict[str, Any]:
        """The gate evidence shaped for ``style.render_gate_report``."""
        gate = run.metadata.get("gate") or {}
        return {
            "challenger": gate.get("challenger") or run.output_model,
            "champion": gate.get("champion"),
            "challenger_score": gate.get("challenger_score") or 0.0,
            "champion_score": gate.get("champion_score"),
            "margin": gate.get("margin") or 0.0,
            "min_gain": gate.get("min_gain") or 0.0,
            "passed": bool(run.gate_passed),
            "reasons": [
                f"metric={gate.get('metric', 'score')}",
                f"beats_incumbent={gate.get('beats_incumbent')}",
            ],
        }

    def compare(
        self,
        challenger: str,
        champion: str = "",
        *,
        metric: str = "score",
    ) -> dict[str, Any]:
        """Head-to-head: challenger vs. champion on one metric.

        ``champion`` defaults to the active model.  Returns scores, margin,
        and a recommendation — the artifact a human (or the agent) reads
        before approving a promotion.
        """
        champion = champion or (self.models.active().name if self.models.active() else "")
        challenger_score = self._model_score(challenger, metric)
        champion_score = self._model_score(champion, metric)
        margin = (challenger_score - champion_score
                  if challenger_score is not None and champion_score is not None
                  else None)
        if margin is None:
            recommendation = "cannot compare — missing scores"
        elif margin > 0:
            recommendation = f"promote: challenger wins by {margin:.6f}"
        elif margin == 0:
            recommendation = "tie: no reason to promote"
        else:
            recommendation = f"reject: challenger loses by {-margin:.6f}"
        return {
            "metric": metric, "challenger": challenger, "champion": champion or None,
            "challenger_score": challenger_score, "champion_score": champion_score,
            "margin": margin, "recommendation": recommendation,
        }

    def promote(
        self, run: TrainingRun, *, force: bool = False, stage: str = "full"
    ) -> bool:
        """Promote the trained model through ``stage``.

        Stages (the progressive-rollout ladder): ``shadow`` (registered,
        serves 0% traffic — observation only), ``canary`` (partial traffic),
        ``full`` (activated — the old behavior).  Refuses unless the gate
        passed, unless ``force=True`` (recorded).  The displaced model is
        recorded so :meth:`demote` can roll back to it.
        """
        if stage not in PROMOTION_STAGES:
            raise ValidationError(
                f"unknown promotion stage {stage!r} (valid: {', '.join(PROMOTION_STAGES)})",
                field="stage",
            )
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
        previous = self.models.active()
        previous_name = previous.name if previous and previous.name != run.output_model else ""
        if stage == "full":
            self.models.activate(run.output_model)
            run.status = RunStatus.PROMOTED
        else:
            # shadow/canary: the model is registered and marked, but the
            # champion keeps serving until a full promotion.
            run.status = RunStatus.SHADOW if stage == "shadow" else RunStatus.CANARY
            run.metadata = {**run.metadata, "stage": stage, "staged_at": time.time()}
            _log.info("run %s staged as %s (champion keeps serving)", run.name, stage)
        run.metadata = {
            **run.metadata,
            "promoted_over": previous_name or None,
            "promotion_stage": stage,
        }
        if force and not run.gate_passed:
            # A human overrode the gate. That belongs in the record.
            run.metadata = {**run.metadata, "forced_promotion": True, "forced_at": time.time()}
            _log.warning("run %s force-promoted despite failing the gate", run.name)
        return self.save(run).status in {RunStatus.PROMOTED, RunStatus.SHADOW, RunStatus.CANARY}

    def demote(self, run: TrainingRun, *, reason: str = "") -> bool:
        """Roll back a promotion: reactivate the model this run displaced.

        The run keeps its evaluation history; its status returns to DONE
        with a demotion record, so the audit trail shows the full
        promote → demote arc instead of a deletion.
        """
        previous = run.metadata.get("promoted_over")
        if run.status not in {RunStatus.PROMOTED, RunStatus.SHADOW, RunStatus.CANARY}:
            raise ValidationError(
                f"run {run.name} is not promoted (status={run.status}); nothing to demote",
                field="status",
            )
        if previous:
            try:
                self.models.activate(previous)
            except Exception as exc:  # noqa: BLE001 — the old artifact may be gone
                _log.warning("demote: could not reactivate %s: %s", previous, exc)
        run.status = RunStatus.DONE
        run.metadata = {
            **run.metadata,
            "demoted": True, "demoted_at": time.time(),
            "demote_reason": reason, "restored_model": previous,
        }
        _log.warning("run %s demoted%s", run.name, f": {reason}" if reason else "")
        return self.save(run).status == RunStatus.DONE

    # ── approval flow ─────────────────────────────────────────────────────

    def request_approval(self, run: TrainingRun, *, approver: str = "") -> TrainingRun:
        """Park a gated run for human approval before promotion."""
        if run.gate_passed is None:
            raise ValidationError(
                f"run {run.name} was never evaluated; call evaluate() first", field="gate_passed"
            )
        run.status = RunStatus.AWAITING_APPROVAL
        run.metadata = {
            **run.metadata,
            "approval": {"requested_at": time.time(), "approver": approver,
                         "decision": None},
        }
        return self.save(run)

    def approve(self, run: TrainingRun, *, approver: str = "") -> TrainingRun:
        """Human approval: the run may now be promoted."""
        if run.status != RunStatus.AWAITING_APPROVAL:
            raise ValidationError(
                f"run {run.name} is not awaiting approval (status={run.status})",
                field="status",
            )
        run.status = RunStatus.DONE
        approval = dict(run.metadata.get("approval") or {})
        approval.update({"decision": "approved", "decided_at": time.time(),
                         "approver": approver or approval.get("approver", "")})
        run.metadata = {**run.metadata, "approval": approval}
        return self.save(run)

    def reject_approval(self, run: TrainingRun, *, reason: str = "") -> TrainingRun:
        """Human rejection: the run is dead, with the reason on record."""
        if run.status != RunStatus.AWAITING_APPROVAL:
            raise ValidationError(
                f"run {run.name} is not awaiting approval (status={run.status})",
                field="status",
            )
        run.status = RunStatus.REJECTED
        approval = dict(run.metadata.get("approval") or {})
        approval.update({"decision": "rejected", "decided_at": time.time(),
                         "reason": reason})
        run.metadata = {**run.metadata, "approval": approval}
        return self.save(run)

    def model_card(self, run: TrainingRun) -> str:
        """A model card for the run: lineage, metrics, gate history.

        The standard attached document for a promotion decision — what the
        model is, what trained it, how it scored, and who overrode what.
        """
        gate = run.metadata.get("gate") or {}
        approval = run.metadata.get("approval") or {}
        lines = [
            f"# Model card: {run.output_model or run.name}",
            "",
            f"- **run**: {run.name} (`{run.id}`)",
            f"- **status**: {run.status}",
            f"- **base model**: {run.base_model or '—'}",
            f"- **dataset**: {run.dataset_id or '—'}",
            f"- **backend**: {run.backend}",
            f"- **output**: {run.output_path or '—'}",
            f"- **created**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(run.created_at))}",
            "",
            "## Training config",
            "",
            "```json",
            json.dumps(run.config, indent=2, ensure_ascii=False)[:2000],
            "```",
            "",
            "## Metrics",
            "",
        ]
        for key, value in run.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"- **{key}**: {value}")
        lines += ["", "## Promotion gate", ""]
        if gate:
            lines += [
                f"- **metric**: {gate.get('metric')}",
                f"- **challenger score**: {gate.get('challenger_score')}",
                f"- **champion**: {gate.get('champion') or 'none'} "
                f"({gate.get('champion_score')})",
                f"- **margin**: {gate.get('margin')} (min_gain={gate.get('min_gain')})",
                f"- **verdict**: {'PASS' if run.gate_passed else 'FAIL'}",
            ]
        else:
            lines.append("- not evaluated")
        if run.metadata.get("forced_promotion"):
            lines.append("- ⚠️ **force-promoted despite failing the gate**")
        if approval:
            lines.append(f"- **approval**: {approval.get('decision')} "
                         f"by {approval.get('approver') or '—'}")
        if run.metadata.get("demoted"):
            lines.append(f"- **demoted**: {run.metadata.get('demote_reason') or 'no reason given'}")
        if run.error:
            lines += ["", "## Error", "", f"```\n{run.error[:1000]}\n```"]
        lines += ["", "## Limitations", "",
                  "- Scores are from the pipeline's own eval; they are a "
                  "regression signal, not a capability benchmark.",
                  "- A promoted model is the best *measured* option, not a "
                  "guaranteed improvement on every prompt."]
        return "\n".join(lines)

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
