"""The mock provider (Craft-Libs port of ai-core ``mock.rs``).

A scripted provider for tests: how many polls a job spends
running, which text arrives per poll, and whether it succeeds
or fails. It answers the same for any conforming provider, so
the lifecycle, cancel and rejection checks run against it.

A job id is ``mock-N``. One text chunk is emitted per poll
until they run out; while chunks remain the job stays
running, however many polls it was scripted to run. While
running, percent is ``polls * 100 // (running_polls + 1)``,
capped at 99. A finished job stays finished: later polls
return the same state with no events.
"""

from __future__ import annotations

from .error import AiError
from .job import JobSnapshot, JobState
from .payload import Limits, Payload
from .provider import Provider

__all__ = ["MockProvider"]


class MockProvider(Provider):
    """A provider that follows a script."""

    def __init__(self, models: list, script: dict | None = None
                 ) -> None:
        self._models = list(models)
        self._script = dict(script or {})
        self._jobs: dict[str, dict] = {}
        self._next = 0

    def models(self) -> list:
        return list(self._models)

    def estimate(self, model, request) -> dict:
        spec = self._spec_for(request)
        return {"cost_micros": 0, "currency": "USD",
                "capability": request.capability.as_str(),
                "max_outputs": spec.max_outputs if spec else 1}

    def submit(self, model, request) -> str:
        info = self._model_for(model)
        if info is None:
            raise AiError.invalid(f"unknown model {self._model_id(model)}")
        spec = info.spec_for(request.capability)
        if spec is None:
            raise AiError.unsupported(
                f"model {self._model_id(model)} cannot do "
                f"{request.capability.as_str()}")
        request.validate(spec, Limits())
        self._next += 1
        job_id = f"mock-{self._next}"
        self._jobs[job_id] = {"request": request, "polls": 0,
                              "chunks_sent": 0, "cancelled": False,
                              "done": None}
        return job_id

    def poll(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        if job["done"] is not None:
            return JobSnapshot(job["done"], []).to_dict()
        if job["cancelled"]:
            job["done"] = JobState.CANCELLED
            return JobSnapshot(JobState.CANCELLED, []).to_dict()
        job["polls"] += 1
        polls = job["polls"]
        events: list = []
        chunks = self._script.get("text_chunks", [])
        if job["chunks_sent"] < len(chunks):
            events.append({"type": "text_delta",
                           "text": chunks[job["chunks_sent"]]})
            job["chunks_sent"] += 1
        streaming_left = job["chunks_sent"] < len(chunks)
        running_polls = int(self._script.get("running_polls", 2))
        if polls <= running_polls or streaming_left:
            percent = min(polls * 100 // (running_polls + 1), 99)
            return JobSnapshot(JobState.running(percent), events) \
                .to_dict()
        outcome = self._script.get("outcome")
        if outcome is None:
            state = JobState.SUCCEEDED
            events.append({"type": "output",
                           "payload": Payload.text("").to_dict()})
        else:
            kind, value = outcome
            if kind == "fail":
                error = value if isinstance(value, AiError) \
                    else AiError.invalid(str(value))
                state = JobState.failed(error)
            else:
                state = JobState.SUCCEEDED
                for payload in value:
                    events.append({"type": "output",
                                   "payload": payload.to_dict()})
        job["done"] = state
        return JobSnapshot(state, events).to_dict()

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        if job["done"] is None:
            job["cancelled"] = True

    def _model_id(self, model) -> str:
        if hasattr(model, "as_str"):
            return model.as_str()
        info_id = getattr(model, "id", None)
        if info_id is not None and hasattr(info_id, "as_str"):
            return info_id.as_str()
        return str(model)

    def _model_for(self, model):
        wanted = self._model_id(model)
        for info in self._models:
            if info.id.as_str() == wanted:
                return info
        return None

    def _spec_for(self, request):
        for model in self._models:
            spec = model.spec_for(request.capability)
            if spec is not None:
                return spec
        return None
