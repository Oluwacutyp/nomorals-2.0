"""Crash-resumable automatic self-improvement pipeline.

This is the small orchestration boundary between L3 training and L6 missions.
The pipeline is represented as a normal :class:`MissionRunner` mission, so each
of collect, curate, train, evaluate, and promote is checkpointed in SQLite.  A
SIGKILL may repeat the step that was in flight, but completed steps are skipped
from their persisted state and a training run is reused rather than duplicated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .missions import MissionRunner, MissionStatus, MissionStore, StepOutcome
from .training.collect import CollectionResult, TrainingCollector
from .training.dataset import DatasetRegistry
from .training.policy import PolicyDecision, RetrainingPolicy, policy_status
from .training.preprocess import prepare
from .training.registry import RunStatus, TrainingRegistry

__all__ = [
    "PIPELINE_STEPS",
    "TrainingJobResult",
    "SelfImprovementJob",
    "AutomaticTrainingJob",
]

PIPELINE_STEPS = ("collect", "curate", "train", "evaluate", "promote")


@dataclass
class TrainingJobResult:
    """Stable result shape for CLI, schedulers, and offline tests."""

    status: str
    mission_id: str = ""
    run_id: str = ""
    dataset_id: str = ""
    promoted: bool = False
    skipped: bool = False
    reason: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.skipped or self.status == MissionStatus.DONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "mission_id": self.mission_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "promoted": self.promoted,
            "skipped": self.skipped,
            "reason": self.reason,
            "steps": self.steps,
            "decision": self.decision,
            "error": self.error,
        }


class _PipelineRunner(MissionRunner):
    """MissionRunner with idempotent, persisted pipeline step handlers."""

    def __init__(self, job: "SelfImprovementJob", *args: Any, **kwargs: Any) -> None:
        self.job = job
        super().__init__(*args, **kwargs)
        self.stop_on_failure = True

    def _execute_step(self, mission: Any, step: Any) -> StepOutcome:
        started = self._clock()
        handler = getattr(self.job, f"_step_{step.name}", None)
        if handler is None:
            return StepOutcome(step=step.name, ok=False, detail=f"unknown training step {step.name}")
        try:
            payload = handler(mission) or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            # Persist the side effect before returning.  MissionRunner also saves
            # after this method; this inner save closes the crash window between a
            # durable artifact and the normal checkpoint.
            self.store.save(mission)
            return StepOutcome(
                step=step.name,
                ok=True,
                seconds=self._clock() - started,
                payload={k: v for k, v in list(payload.items())[:20]},
            )
        except Exception as exc:  # noqa: BLE001 - a pipeline step is a mission result
            return StepOutcome(
                step=step.name,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                seconds=self._clock() - started,
            )


class SelfImprovementJob:
    """Run the complete loop when :class:`RetrainingPolicy` says it is due."""

    def __init__(
        self,
        context: Any,
        *,
        policy: RetrainingPolicy | None = None,
        provider: str = "mock",
        collector: TrainingCollector | None = None,
    ) -> None:
        self.context = context
        self.db = context.db
        self.policy = policy or RetrainingPolicy()
        self.provider = provider or "mock"
        self.collector = collector or TrainingCollector(self.db)
        self.datasets = DatasetRegistry(self.db)
        self.runs = TrainingRegistry(self.db)
        self.store = MissionStore(self.db)
        self._collection: CollectionResult | None = None

    @property
    def data_dir(self) -> Path:
        settings = self.context.settings
        return settings.resolve(settings.training.data_dir)

    @property
    def output_dir(self) -> Path:
        settings = self.context.settings
        return settings.resolve(settings.training.output_dir)

    def status(self) -> dict[str, Any]:
        state = policy_status(self.db, policy=self.policy)
        decision: PolicyDecision = state.pop("decision")
        state["policy"] = dict(vars(self.policy))
        state.update(
            {
                "decision": decision.to_dict(),
                "runs": self.runs.stats(),
                "provider": self.provider,
            }
        )
        return state

    def collect(self, *, limit: int = 0) -> CollectionResult:
        """Expose the safe harvest for ``nm train --collect``."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        result = self.collector.collect(
            limit=limit,
            output_dir=self.data_dir,
            name=f"collected-{int(time.time())}",
            register=True,
        )
        self._collection = result
        return result

    def run(
        self,
        *,
        force: bool = False,
        max_iterations: int = 8,
        budget_wall: float = 0.0,
        budget_tokens: int = 0,
        dataset_id: str = "",
        backend: str = "",
        base_model: str = "",
    ) -> TrainingJobResult:
        """Create and execute one durable pipeline mission.

        ``force`` bypasses only the trigger policy.  It never bypasses the model
        regression gate; promotion remains controlled by ``TrainingRegistry``.

        ``dataset_id`` seeds the mission with a specific registered corpus
        (e.g. an account-history export) instead of collecting live-use rows.

        ``backend`` selects the training engine (``native`` | ``unsloth`` |
        ``llama_factory``); empty falls back to ``TrainingSettings.backend``.
        ``base_model`` is the HF id the external backends finetune; required
        for anything but ``native``.
        """
        # A scheduler invocation after a process restart first gives interrupted
        # pipeline missions their durable continuation.  This is important when
        # the dataset was created just before the crash: trigger metrics alone
        # cannot tell an unfinished run from a completed one.
        resumed = self.resume_all(max_iterations=max_iterations)
        if resumed:
            return resumed[0]

        status = self.status()
        decision = status["decision"]
        if not decision["data_available"] or (not force and not decision["should_retrain"]):
            return TrainingJobResult(
                status="skipped",
                skipped=True,
                reason="; ".join(decision["reasons"]),
                decision=decision,
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        store = self.store
        # A forced corpus (e.g. an account-history export) is seeded straight
        # into the pipeline state; _step_collect reuses it without collecting.
        # A backend/base model pinned for this run rides the same state so a
        # crash-resume re-runs the SAME engine, not whatever the config says.
        pipeline_state: dict[str, Any] = {"raw_dataset_id": dataset_id} if dataset_id else {}
        if backend:
            pipeline_state["backend"] = backend
        if base_model:
            pipeline_state["base_model"] = base_model
        mission = store.create_new(
            "self-improvement training loop",
            name="automatic-training",
            budget_wall=budget_wall,
            budget_tokens=budget_tokens,
            state={"plan": _pipeline_plan(), "pipeline": pipeline_state, "provider": self.provider},
            metadata={"pipeline": "collect-curate-train-evaluate-promote", "provider": self.provider},
        )
        runner = _PipelineRunner(self, self.context, store=store)
        mission_result = runner.run(mission, max_iterations=max_iterations, reflect=True)
        return self._result(mission_result, decision)

    def resume(self, mission_id: str, *, max_iterations: int = 8) -> TrainingJobResult:
        """Resume a previously interrupted pipeline mission after a crash."""
        runner = _PipelineRunner(self, self.context, store=self.store)
        mission_result = runner.resume(mission_id, max_iterations=max_iterations, reflect=True)
        return self._result(mission_result, self.status()["decision"])

    def resume_all(self, *, max_iterations: int = 8) -> list[TrainingJobResult]:
        """Resume every interrupted automatic-training mission, newest first."""
        results: list[TrainingJobResult] = []
        for mission in self.store.resumable():
            if mission.metadata.get("pipeline") != "collect-curate-train-evaluate-promote":
                continue
            runner = _PipelineRunner(self, self.context, store=self.store)
            mission_result = runner.resume(mission.id, max_iterations=max_iterations, reflect=True)
            results.append(self._result(mission_result, self.status()["decision"]))
        return results

    # ── pipeline steps -----------------------------------------------------
    def _step_collect(self, mission: Any) -> dict[str, Any]:
        prior = mission.state.get("pipeline") or {}
        if prior.get("raw_dataset_id"):
            dataset = self.datasets.get(prior["raw_dataset_id"])  # NotFound -> clear error
            if dataset.rows <= 0:
                raise ValueError(f"dataset {dataset.name!r} has no examples to train on")
            return {"dataset_id": dataset.id, "reused": True}
        result = self.collector.collect(
            output_dir=self.data_dir,
            name=f"collected-{mission.id}",
            register=True,
        )
        if not result.dataset_id:
            # A manual ``--collect`` may have already registered every source
            # row.  Reuse the newest durable corpus for an initial run instead
            # of manufacturing a duplicate; subsequent runs are controlled by
            # the policy and still receive the promotion gate.
            available = self.datasets.list(limit=1)
            if not available:
                raise ValueError("collection produced no new examples")
            prior["raw_dataset_id"] = available[0].id
            prior["collected"] = result.to_dict()
            mission.state["pipeline"] = prior
            return {"dataset_id": available[0].id, "count": 0, "reused": True}
        self._collection = result
        prior["raw_dataset_id"] = result.dataset_id
        prior["collected"] = result.to_dict()
        mission.state["pipeline"] = prior
        return {"dataset_id": result.dataset_id, "count": result.count}

    def _step_curate(self, mission: Any) -> dict[str, Any]:
        pipeline = mission.state.setdefault("pipeline", {})
        if pipeline.get("train_dataset_id"):
            return {"dataset_id": pipeline["train_dataset_id"], "reused": True}
        raw = self.datasets.get(pipeline["raw_dataset_id"])
        train, evaluation, stats = prepare(
            raw.examples(),
            eval_fraction=float(self.context.settings.training.eval_split),
            seed=1234,
        )
        if not train:
            raise ValueError("curation produced an empty training split")
        train_ds = self.datasets.register_examples(
            f"curated-train-{mission.id}",
            train,
            self.data_dir,
            metadata={"curated": True, "source_dataset": raw.id, "stats": stats.to_dict()},
        )
        eval_id = ""
        if evaluation:
            eval_ds = self.datasets.register_examples(
                f"curated-eval-{mission.id}",
                evaluation,
                self.data_dir,
                kind="eval",
                metadata={"curated": True, "source_dataset": raw.id, "stats": stats.to_dict()},
            )
            eval_id = eval_ds.id
        pipeline.update({"train_dataset_id": train_ds.id, "eval_dataset_id": eval_id, "curate_stats": stats.to_dict()})
        return {"dataset_id": train_ds.id, "eval_dataset_id": eval_id, "stats": stats.to_dict()}

    def _step_train(self, mission: Any) -> dict[str, Any]:
        from .training.backends import get_backend

        pipeline = mission.state.setdefault("pipeline", {})
        train_ds = self.datasets.get(pipeline["train_dataset_id"])
        eval_ds = self.datasets.get(pipeline["eval_dataset_id"]) if pipeline.get("eval_dataset_id") else None
        run_id = pipeline.get("run_id")
        run = self.runs.get(run_id) if run_id else None
        model_dir = self.output_dir / mission.id / "model"
        # A DONE run with any artifact is reused on resume — model.json for the
        # native backend, adapter dirs / GGUF for the external ones.
        if run is not None and run.status == RunStatus.DONE and _dir_has_artifact(model_dir):
            return {"run_id": run.id, "model_path": str(model_dir), "reused": True}

        settings = self.context.settings.training
        backend_name = (
            pipeline.get("backend") or settings.backend or "native"
        ).strip().lower()
        base_model_override = (
            pipeline.get("base_model") or settings.base_model or ""
        ).strip()

        if run is None:
            active = self.runs.models.active()
            base_model = base_model_override or (
                active.name if active else (self.context.settings.llm.active_model or self.provider)
            )
            run = self.runs.start(
                f"auto-{mission.id}",
                base_model=base_model,
                output_model=f"personal-{mission.id}",
                dataset_id=train_ds.id,
                backend=backend_name,
                config={"provider": self.provider},
                metadata={"mission_id": mission.id, "automatic": True},
            )
            pipeline["run_id"] = run.id
            self.store.save(mission)

        train_examples = list(train_ds.examples())
        eval_examples = list(eval_ds.examples()) if eval_ds is not None else []

        # External backends finetune an HF base model — a local model NAME is
        # not a loadable id, so require an explicit one rather than letting the
        # HF API explain the mistake.
        if backend_name != "native" and not base_model_override:
            error = (
                f"backend {backend_name!r} needs a base model id (HF): set "
                "NM_TRAINING_BASE_MODEL or pass --base-model, e.g. "
                "cognitivecomputations/dolphin-2.9-llama3-8b"
            )
            self.runs.fail(run, error=error)
            raise RuntimeError(error)

        backend = get_backend(backend_name)
        ok, reason = backend.available()
        if not ok:
            error = f"backend {backend_name!r} is not available on this machine: {reason}"
            self.runs.fail(run, error=error)
            raise RuntimeError(error)

        result = backend.train(
            train_examples, eval_examples,
            output_dir=model_dir,
            base_model=base_model_override or run.base_model,
            settings=settings,
        )
        self.runs.complete(run, metrics=result.metrics, output_path=result.output_path)
        pipeline["run_id"] = run.id
        pipeline["model_path"] = result.output_path
        return {
            "run_id": run.id,
            "model_path": result.output_path,
            "metrics": result.metrics,
            "info": result.info,
        }

    def _step_evaluate(self, mission: Any) -> dict[str, Any]:
        pipeline = mission.state.setdefault("pipeline", {})
        run = self.runs.get(pipeline["run_id"])
        if run.gate_passed is not None:
            return {"run_id": run.id, "gate_passed": run.gate_passed, "reused": True}
        passed = self.runs.evaluate(
            run,
            metric="score",
            tolerance=float(self.context.settings.training.regression_tolerance),
        )
        pipeline["gate_passed"] = passed
        return {"run_id": run.id, "gate_passed": passed, "metrics": run.metrics}

    def _step_promote(self, mission: Any) -> dict[str, Any]:
        pipeline = mission.state.setdefault("pipeline", {})
        run = self.runs.get(pipeline["run_id"])
        if run.status == RunStatus.PROMOTED:
            pipeline["promoted"] = True
            return {"run_id": run.id, "promoted": True, "reused": True}
        if run.status == RunStatus.REJECTED:
            pipeline["promoted"] = False
            return {"run_id": run.id, "promoted": False, "rejected": True, "reused": True}
        promoted = self.runs.promote(run)
        pipeline["promoted"] = promoted
        return {"run_id": run.id, "promoted": promoted, "gate_passed": run.gate_passed}

    def _result(self, mission_result: Any, decision: dict[str, Any]) -> TrainingJobResult:
        mission = self.store.get(mission_result.mission_id)
        pipeline = mission.state.get("pipeline") or {}
        return TrainingJobResult(
            status=mission_result.status,
            mission_id=mission_result.mission_id,
            run_id=str(pipeline.get("run_id") or ""),
            dataset_id=str(pipeline.get("train_dataset_id") or pipeline.get("raw_dataset_id") or ""),
            promoted=bool(pipeline.get("promoted")),
            decision=decision,
            steps=[step.to_dict() for step in mission_result.steps],
            error=mission_result.error,
        )


AutomaticTrainingJob = SelfImprovementJob


def _dir_has_artifact(model_dir: Path) -> bool:
    """True when a trained model has been persisted into ``model_dir``.

    The native backend writes ``model.json``; external backends write a PEFT
    adapter directory, a tokenizer, or a GGUF.  Any non-empty directory
    counts, which keeps crash-resume from retraining a run that already
    produced an artifact.
    """
    if not model_dir.is_dir():
        return False
    return any(model_dir.iterdir())


def _pipeline_plan() -> list[dict[str, Any]]:
    return [
        {"name": name, "goal": f"{name} training pipeline stage", "role": "training", "kind": "cpu", "depends_on": []}
        for name in PIPELINE_STEPS
    ]
