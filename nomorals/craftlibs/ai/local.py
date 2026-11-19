"""The local motion provider (the anicraft suggestion engine,
re-homed).

The motion suggestion engine that used to live in
``nomorals/media/animation/ai.py`` now runs here, as a real
:class:`~craftlibs.ai.provider.Provider`: it declares the models
it offers, checks each request against their specs, and answers
synchronously. Nothing about what it suggests has changed — the
same keywords pick the same preset and easing — only where the
engine lives, and what it answers through.
"""

from __future__ import annotations

import json

from .capability import CapabilityId
from .job import JobSnapshot, JobState
from .model import CapabilitySpec, InputRole, ModelInfo
from .payload import Limits, Payload
from .provider import Provider
from .request import Request

__all__ = [
    "LOCAL_CAPABILITIES",
    "LocalMotionProvider",
    "match_motion",
    "suggest_easing",
    "suggest_motion",
]

LOCAL_CAPABILITIES = ("motion.suggest", "motion.ease")

# Words in a description, and the motion they ask for.
_KEYWORDS = [
    (("bounc", "hop", "jump"), "bounce", "bounce-out"),
    (("walk", "stroll", "march"), "walk-cycle", "sine-in-out"),
    (("fade", "appear", "dissolve"), "fade-in", "sine-in-out"),
    (("slide", "enter", "arrive"), "slide-in", "quad-out"),
    (("pulse", "beat", "throb", "breathe"), "pulse", "sine-in-out"),
    (("type", "write", "spell"), "typewriter", "linear"),
    (("spin", "rotat", "twirl"), "pulse", "quad-in-out"),
]


def match_motion(prompt: str):
    """The keyword row a description matches: ``(preset, easing)``,
    or ``None`` when no row matches."""
    text = prompt.lower()
    for words, preset, easing in _KEYWORDS:
        if any(word in text for word in words):
            return preset, easing
    return None


def suggest_motion(prompt: str) -> str:
    """The preset a description asks for, by its words."""
    matched = match_motion(prompt)
    if matched is not None:
        return matched[0]
    return "slide-in"


def suggest_easing(prompt: str) -> str:
    """The easing a description asks for, by its words."""
    text = prompt.lower()
    if any(word in text for word in ("bounc", "spring", "elastic")):
        return "bounce-out"
    if any(word in text for word in ("gentle", "smooth", "soft",
                                     "slow")):
        return "sine-in-out"
    if any(word in text for word in ("fast", "snap", "quick",
                                     "sharp")):
        return "quad-out"
    return "quad-in-out"


def _motion_spec(capability: str) -> CapabilitySpec:
    spec = CapabilitySpec.new(CapabilityId.parse(capability))
    spec = spec.accepting(InputRole.PROMPT, 1)
    spec = spec.requiring(InputRole.PROMPT)
    return spec.with_max_outputs(1)


class LocalMotionProvider(Provider):
    """The local motion engine, as a provider. It answers
    immediately: submit runs the suggestion and poll reports it
    done."""

    def __init__(self) -> None:
        self._models = [
            ModelInfo.new("local-motion", "Local motion suggestions")
            .with_spec(_motion_spec("motion.suggest")),
            ModelInfo.new("local-motion-ease", "Local easing suggestions")
            .with_spec(_motion_spec("motion.ease")),
        ]
        self._jobs: dict[str, dict] = {}
        self._next = 0

    def models(self) -> list:
        return list(self._models)

    def estimate(self, model, request: Request) -> dict:
        spec = self._spec_for(request)
        return {"cost_micros": 0, "currency": "USD",
                "capability": request.capability.as_str(),
                "max_outputs": spec.max_outputs if spec else 1}

    def submit(self, model, request: Request) -> str:
        from .error import AiError
        info = self._model_for(model)
        if info is None:
            raise AiError.invalid(f"unknown model {self._model_id(model)}")
        spec = info.spec_for(request.capability)
        if spec is None:
            raise AiError.unsupported(
                f"model {self._model_id(model)} cannot do "
                f"{request.capability.as_str()}")
        request.validate(spec, Limits())
        prompt = request.inputs[0][1].as_text() or "" \
            if request.inputs else ""
        # One suggestion dict, whatever the capability asked for:
        # the preset the words ask for and the easing they ask
        # for, as JSON text in the output payload.
        result = {"preset": suggest_motion(prompt),
                  "easing": suggest_easing(prompt)}
        self._next += 1
        job_id = f"local-{self._next}"
        self._jobs[job_id] = {"result": result, "polled": False,
                              "cancelled": False, "done": None}
        return job_id

    def poll(self, job_id: str) -> dict:
        from .error import AiError
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        if job["done"] is not None:
            # A terminal state is final: a later cancel cannot
            # flip a finished job to cancelled.
            return JobSnapshot(job["done"], []).to_dict()
        if job["cancelled"]:
            job["done"] = JobState.CANCELLED
            return JobSnapshot(JobState.CANCELLED, []).to_dict()
        job["polled"] = True
        job["done"] = JobState.SUCCEEDED
        return JobSnapshot(
            JobState.SUCCEEDED,
            [{"type": "output",
              "payload": Payload.text(
                  json.dumps(job["result"])).to_dict()}]) \
            .to_dict()

    def cancel(self, job_id: str) -> None:
        from .error import AiError
        if job_id not in self._jobs:
            raise AiError.invalid(f"unknown job {job_id}")
        if self._jobs[job_id]["done"] is None:
            self._jobs[job_id]["cancelled"] = True

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

    def _spec_for(self, request: Request):
        for model in self._models:
            spec = model.spec_for(request.capability)
            if spec is not None:
                return spec
        return None
