"""AI-client seam — a Python port of the shape of
craft-ai-integrations ``crates/ai-client`` (commit 02068b0).

An app declares its :class:`Role`s, builds one :class:`AiClient`,
and for each feature runs: ``status`` (can it run?), ``plan`` (what
would be sent?), consent if needed, ``start`` (go), then polls the
:class:`Job`. That flow is upstream's, verbatim.

The roles, jobs and the local motion engine behind this seam now
live in the shared library (``nomorals/craftlibs/ai/``): a role is
a shared :class:`~craftlibs.ai.role.Role`, a job carries a shared
:class:`~craftlibs.ai.job.JobSnapshot`, and the local suggestion
engine runs as a shared :class:`~craftlibs.ai.local.LocalMotionProvider`.
This module is the app-facing seam over them — the keyword table,
the preset choice and the easing choice live in exactly one place.
Two honest facts about this port:

* AniCraft at the ported commit (bcfea81) does NOT depend on
  ai-client — the census line's "the one app consuming
  craft-ai-integrations' ai-client" is stale against the current
  upstream manifest (its root Cargo.toml lists only craft-libs git
  deps; a repo-wide grep for ``ai-client``/``ai_client``/``craft-ai``
  finds nothing). The seam is therefore built from the
  craft-ai-integrations crate itself, not from an AniCraft call site.
* Devon has no ai-client and no external AI provider wired into
  this organ. The local engine runs motion suggestions
  deterministically, on this machine, with no network and no cost;
  every capability that would need a provider reports
  ``Unavailable`` with the reason named, and cloud roles need
  consent before they may even start — after consent, the job still
  reports unavailable, because there is no provider to call.
"""

from __future__ import annotations

import time
from typing import Any

from nomorals.craftlibs.ai import (
    AiError,
    CapabilityId,
    InputRole,
    JobSnapshot,
    JobState,
    ModelId,
    Payload,
    Request,
)
from nomorals.craftlibs.ai.local import LocalMotionProvider, match_motion
from nomorals.craftlibs.ai.local import suggest_easing as _shared_suggest_easing
from nomorals.craftlibs.ai.role import Role as SharedRole

from .errors import AnimationError

__all__ = ["AiClient", "ConsentLedger", "Job", "Role"]

# Capabilities the local engine can run, no provider, no consent.
LOCAL_CAPABILITIES = ("motion.suggest", "motion.ease")
# Capabilities that would send the prompt to a cloud provider.
CLOUD_CAPABILITIES = ("motion.suggest-cloud",)


class Role:
    """One feature's AI role: a shared role, with this organ's
    string-facing view over it."""

    def __init__(self, id: str, label: str, capability: str,
                 inputs: list[str] | None = None) -> None:
        self._shared = SharedRole(id, label, capability,
                                  inputs or ["prompt"])

    @property
    def id(self) -> str:
        return self._shared.id.as_str()

    @property
    def label(self) -> str:
        return self._shared.label

    @property
    def capability(self) -> str:
        return self._shared.capability.as_str()

    @property
    def inputs(self) -> list[str]:
        return [i.as_str() for i in self._shared.inputs]

    @classmethod
    def suggest_motion(cls) -> Role:
        return cls("suggest-motion", "Suggest motion",
                   "motion.suggest", ["prompt"])

    @classmethod
    def suggest_easing(cls) -> Role:
        return cls("suggest-easing", "Suggest easing",
                   "motion.ease", ["prompt"])

    def to_dict(self) -> dict[str, Any]:
        return self._shared.to_dict()


class ConsentLedger:
    """Granted disclosures (upstream ``ai-core/src/consent.rs``).

    This ledger stays app-specific: its disclosures are free-text
    strings ("runs locally; nothing leaves this machine"), a
    different concept from the shared library's structured
    per-kind consent ledger.
    """

    def __init__(self) -> None:
        self._grants: dict[str, float | None] = {}

    def needs_consent(self, disclosure: str) -> bool:
        expires = self._grants.get(disclosure)
        granted = (expires is None and disclosure in self._grants) or (
            expires is not None and expires > time.time())
        return not granted

    def grant(self, disclosure: str) -> None:
        self._grants[disclosure] = None

    def grant_until(self, disclosure: str, expires_at: float) -> None:
        self._grants[disclosure] = float(expires_at)


class Job:
    """A started job: a shared job snapshot, with this organ's
    ``done`` / ``cancelled`` / ``unavailable`` view over it. A
    job that fails — no provider to call — surfaces as
    ``unavailable``, never as a silent empty."""

    def __init__(self, job_id: str, result: dict[str, Any] | None,
                 reason: str | None = None,
                 snapshot: JobSnapshot | None = None) -> None:
        self.id = job_id
        self._result = result
        self._reason = reason
        if snapshot is not None:
            self._snapshot = snapshot
        elif result is not None:
            self._snapshot = JobSnapshot(JobState.SUCCEEDED, [])
        else:
            self._snapshot = JobSnapshot(JobState.failed(
                AiError.unsupported(
                    reason or "no provider configured")), [])
        self._cancelled = False

    def poll(self) -> dict[str, Any]:
        if self._cancelled:
            return {"status": "cancelled", "job_id": self.id}
        state = self._snapshot.state
        if state.kind == "succeeded":
            return {"status": "done", "job_id": self.id,
                    "result": self._result}
        if state.kind == "cancelled":
            return {"status": "cancelled", "job_id": self.id}
        reason = self._reason
        if state.error is not None:
            reason = str(state.error)
        return {"status": "unavailable", "job_id": self.id,
                "reason": reason or "no provider configured"}

    def cancel(self) -> None:
        self._cancelled = True


def suggest_motion(prompt: str) -> dict[str, Any]:
    """Prompt -> a motion plan, by keyword. Deterministic, local.
    The keyword row comes from the shared engine; the plan shape
    (preset, easing, frames, note) is this organ's."""
    matched = match_motion(prompt)
    if matched is not None:
        preset, easing = matched
        return {"preset": preset, "easing": easing,
                "duration_frames": 24,
                "note": f"matched {preset!r} from the prompt"}
    return {"preset": "slide-in", "easing": "quad-out",
            "duration_frames": 24,
            "note": "no preset matched the prompt; slide-in is the "
                    "default"}


# The shared easing choice has exactly four outcomes, each with
# its own note here.
_EASING_NOTES = {
    "bounce-out": "springy motion wants bounce-out",
    "sine-in-out": "gentle motion wants sine-in-out",
    "quad-out": "snappy motion wants quad-out",
    "quad-in-out": "no cue in the prompt; quad-in-out is the default",
}


def suggest_easing(prompt: str) -> dict[str, Any]:
    """Prompt -> an easing name, by keyword. Deterministic, local."""
    easing = _shared_suggest_easing(prompt)
    return {"easing": easing, "note": _EASING_NOTES[easing]}


class AiClient:
    """The app-facing entry point of the AI layer (upstream ``AiClient``)."""

    def __init__(self, app_id: str) -> None:
        self.app_id = app_id
        self._jobs = 0

    @classmethod
    def standard(cls, app_id: str) -> AiClient:
        return cls(app_id)

    def status(self, role: Role) -> dict[str, Any]:
        """Whether a role can run right now (upstream ``RoleStatus``)."""
        if role.capability in LOCAL_CAPABILITIES:
            return {"available": True, "engine": "local",
                    "connection": "local", "model": "local-motion",
                    "locality": "local", "billing": "none",
                    "role": role.id}
        if role.capability in CLOUD_CAPABILITIES:
            return {"available": False, "engine": "cloud",
                    "reason": "no cloud provider configured for "
                              f"capability {role.capability!r}",
                    "role": role.id}
        return {"available": False, "engine": "none",
                "reason": "no provider configured for capability "
                          f"{role.capability!r}; this app can't run the "
                          "chosen model yet",
                "role": role.id}

    def plan(self, role: Role, request: dict[str, Any]) -> dict[str, Any]:
        """What would be sent where (upstream ``Plan``)."""
        if role.capability in LOCAL_CAPABILITIES:
            return {"capability": role.capability,
                    "role_id": role.id, "engine": "local",
                    "needs_consent": False,
                    "disclosure": "runs locally; nothing leaves this "
                                  "machine",
                    "request": dict(request)}
        if role.capability in CLOUD_CAPABILITIES:
            prompt = str(request.get("prompt", ""))
            return {"capability": role.capability,
                    "role_id": role.id, "engine": "cloud",
                    "needs_consent": True,
                    "disclosure": f"send the prompt {prompt!r} to a "
                                  "cloud provider for a motion plan",
                    "request": dict(request)}
        raise AnimationError(
            f"cannot plan role {role.id!r}: no provider configured for "
            f"capability {role.capability!r}")

    def start(self, plan: dict[str, Any],
              ledger: ConsentLedger) -> Job:
        """Start a planned job; consent is enforced before anything runs."""
        if plan.get("needs_consent") and ledger.needs_consent(
                plan["disclosure"]):
            raise AnimationError(
                "consent required before this job may start: "
                f"{plan['disclosure']}")
        self._jobs += 1
        job_id = f"{self.app_id}-job-{self._jobs}"
        if plan.get("engine") == "local":
            prompt = str(plan.get("request", {}).get("prompt", ""))
            provider = LocalMotionProvider()
            request = Request(
                CapabilityId.parse(plan["capability"]),
                [(InputRole.PROMPT, Payload.text(prompt))])
            model_id = ("local-motion"
                        if plan["capability"] == "motion.suggest"
                        else "local-motion-ease")
            provider_job = provider.submit(ModelId.new(model_id),
                                           request)
            snapshot = JobSnapshot.from_dict(
                provider.poll(provider_job))
            if plan["capability"] == "motion.suggest":
                return Job(job_id, suggest_motion(prompt),
                           snapshot=snapshot)
            return Job(job_id, suggest_easing(prompt),
                       snapshot=snapshot)
        return Job(job_id, None,
                   reason="no cloud provider configured for capability "
                          f"{plan.get('capability')!r}")


    def list_roles(self) -> list[dict[str, Any]]:
        return [Role.suggest_motion().to_dict(),
                Role.suggest_easing().to_dict()]


__all__ += ["suggest_easing", "suggest_motion"]
