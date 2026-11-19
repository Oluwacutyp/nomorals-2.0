"""AI-client seam — a Python port of the shape of
craft-ai-integrations ``crates/ai-client`` (commit 02068b0).

An app declares its :class:`Role`s, builds one :class:`AiClient`,
and for each feature runs: ``status`` (can it run?), ``plan`` (what
would be sent?), consent if needed, ``start`` (go), then polls the
:class:`Job`. That flow is upstream's, verbatim.

Two honest facts about this port:

* AniCraft at the ported commit (bcfea81) does NOT depend on
  ai-client — the census line's "the one app consuming
  craft-ai-integrations' ai-client" is stale against the current
  upstream manifest (its root Cargo.toml lists only craft-libs git
  deps; a repo-wide grep for ``ai-client``/``ai_client``/``craft-ai``
  finds nothing). The seam is therefore built from the
  craft-ai-integrations crate itself, not from an AniCraft call site.
* Devon has no ai-client and no external AI provider wired into
  this organ. The local engine below runs motion suggestions
  deterministically, on this machine, with no network and no cost;
  every capability that would need a provider reports
  ``Unavailable`` with the reason named, and cloud roles need
  consent before they may even start — after consent, the job still
  reports unavailable, because there is no provider to call.
"""

from __future__ import annotations

import time
from typing import Any

from .errors import AnimationError

__all__ = ["AiClient", "ConsentLedger", "Job", "Role"]

# Capabilities the local engine can run, no provider, no consent.
LOCAL_CAPABILITIES = ("motion.suggest", "motion.ease")
# Capabilities that would send the prompt to a cloud provider.
CLOUD_CAPABILITIES = ("motion.suggest-cloud",)


class Role:
    """One feature's AI role (upstream ``ai-core/src/role.rs``)."""

    def __init__(self, id: str, label: str, capability: str,
                 inputs: list[str] | None = None) -> None:
        self.id = id
        self.label = label
        self.capability = capability
        self.inputs = list(inputs or ["prompt"])

    @classmethod
    def suggest_motion(cls) -> Role:
        return cls("suggest-motion", "Suggest motion",
                   "motion.suggest", ["prompt"])

    @classmethod
    def suggest_easing(cls) -> Role:
        return cls("suggest-easing", "Suggest easing",
                   "motion.ease", ["prompt"])

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label,
                "capability": self.capability, "inputs": self.inputs}


class ConsentLedger:
    """Granted disclosures (upstream ``ai-core/src/consent.rs``)."""

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
    """A started job (upstream ``ai-client/src/client.rs``)."""

    def __init__(self, job_id: str, result: dict[str, Any] | None,
                 reason: str | None = None) -> None:
        self.id = job_id
        self._result = result
        self._reason = reason
        self._cancelled = False

    def poll(self) -> dict[str, Any]:
        if self._cancelled:
            return {"status": "cancelled", "job_id": self.id}
        if self._result is not None:
            return {"status": "done", "job_id": self.id,
                    "result": self._result}
        return {"status": "unavailable", "job_id": self.id,
                "reason": self._reason or "no provider configured"}

    def cancel(self) -> None:
        self._cancelled = True


# -- the local motion engine (deterministic, no network) ---------------
_KEYWORDS: list[tuple[tuple[str, ...], str, str]] = [
    (("bounc", "hop", "jump"), "bounce", "bounce-out"),
    (("walk", "stroll", "march"), "walk-cycle", "sine-in-out"),
    (("fade", "appear", "dissolve"), "fade-in", "sine-in-out"),
    (("slide", "enter", "arrive"), "slide-in", "quad-out"),
    (("pulse", "beat", "throb", "breathe"), "pulse", "sine-in-out"),
    (("type", "write", "spell"), "typewriter", "linear"),
    (("spin", "rotat", "twirl"), "pulse", "quad-in-out"),
]


def suggest_motion(prompt: str) -> dict[str, Any]:
    """Prompt -> a motion plan, by keyword. Deterministic, local."""
    text = prompt.lower()
    for keywords, preset, easing in _KEYWORDS:
        if any(k in text for k in keywords):
            return {"preset": preset, "easing": easing,
                    "duration_frames": 24,
                    "note": f"matched {preset!r} from the prompt"}
    return {"preset": "slide-in", "easing": "quad-out",
            "duration_frames": 24,
            "note": "no preset matched the prompt; slide-in is the "
                    "default"}


def suggest_easing(prompt: str) -> dict[str, Any]:
    """Prompt -> an easing name, by keyword. Deterministic, local."""
    text = prompt.lower()
    if any(k in text for k in ("bounc", "spring", "elastic")):
        return {"easing": "bounce-out",
                "note": "springy motion wants bounce-out"}
    if any(k in text for k in ("gentle", "smooth", "soft", "slow")):
        return {"easing": "sine-in-out",
                "note": "gentle motion wants sine-in-out"}
    if any(k in text for k in ("fast", "snap", "quick", "sharp")):
        return {"easing": "quad-out",
                "note": "snappy motion wants quad-out"}
    return {"easing": "quad-in-out",
            "note": "no cue in the prompt; quad-in-out is the default"}


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
            if plan["capability"] == "motion.suggest":
                return Job(job_id, suggest_motion(prompt))
            return Job(job_id, suggest_easing(prompt))
        return Job(job_id, None,
                   reason="no cloud provider configured for capability "
                          f"{plan.get('capability')!r}")

    def list_roles(self) -> list[dict[str, Any]]:
        return [Role.suggest_motion().to_dict(),
                Role.suggest_easing().to_dict()]


__all__ += ["suggest_easing", "suggest_motion"]
