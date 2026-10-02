"""Formal mission state machine (Wave H2, L6).

The persisted :class:`~nomorals.missions.mission.Mission` carries a coarse
cross-process ``status`` (pending/running/paused/done/failed/cancelled).
This module layers a finer, formally checked state machine on top of it:
a move that is not in ``LEGAL_TRANSITIONS`` raises
:class:`InvalidTransition` instead of silently corrupting the mission's
lifecycle.

The os-side state lives in ``mission.state["os_state"]`` plus an
append-only ``mission.state["transition_log"]``, so no storage migration
is needed; the coarse ``MissionStatus`` is kept in sync through
``MissionStore.set_status``.

Wiring to the runner is duck-typed: :func:`attach_runner` sets
``runner._os_transition_hook`` to a plain callable, so missions (L5) never
import this package (L6) and vice versa at runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..core.tasks import AcceptanceCriterion, TaskResult

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime cost
    from ..missions.mission import Mission, MissionStore
    from ..storage.artifacts import ArtifactStore

__all__ = [
    "CREATED",
    "PLANNED",
    "RUNNING",
    "VERIFYING",
    "PAUSED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TERMINAL_STATES",
    "LEGAL_TRANSITIONS",
    "OS_STATE_TO_MISSION_STATUS",
    "InvalidTransition",
    "MissionAcceptance",
    "current_state",
    "transition",
    "attach_runner",
    "evaluate_acceptance",
    "verify_repair_loop",
]

_log = get_logger(__name__)

# ── states ───────────────────────────────────────────────────────────────────

CREATED = "CREATED"
PLANNED = "PLANNED"
RUNNING = "RUNNING"
VERIFYING = "VERIFYING"
PAUSED = "PAUSED"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELLED})

#: Legal moves. Terminal states have no outgoing edges. RUNNING may complete
#: directly (missions with no acceptance criteria skip VERIFYING); when
#: acceptance is wired, the RUNNING -> VERIFYING -> COMPLETED path is used.
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    CREATED: frozenset({PLANNED, CANCELLED}),
    PLANNED: frozenset({RUNNING, CANCELLED}),
    RUNNING: frozenset({VERIFYING, COMPLETED, FAILED, PAUSED, CANCELLED}),
    VERIFYING: frozenset({COMPLETED, FAILED, RUNNING}),
    PAUSED: frozenset({RUNNING, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

#: Coarse status kept in sync on the persisted Mission row.
#: (String literals — MissionStatus lives in missions/L5 and is only
#: imported here under TYPE_CHECKING to avoid a runtime cycle.)
OS_STATE_TO_MISSION_STATUS: dict[str, str] = {
    CREATED: "pending",
    PLANNED: "pending",
    RUNNING: "running",
    VERIFYING: "running",
    PAUSED: "paused",
    COMPLETED: "done",
    FAILED: "failed",
    CANCELLED: "cancelled",
}

_OS_STATE_KEY = "os_state"
_TRANSITION_LOG_KEY = "transition_log"


class InvalidTransition(Exception):
    """Raised when a mission move is not in ``LEGAL_TRANSITIONS``."""


# ── transitions ──────────────────────────────────────────────────────────────

def current_state(mission: "Mission") -> str:
    """The mission's os state; ``CREATED`` when never transitioned."""
    state = str(mission.state.get(_OS_STATE_KEY) or CREATED).upper()
    return state if state in LEGAL_TRANSITIONS else CREATED


def transition(store: "MissionStore", mission_id: str, to: str,
               note: str = "") -> "Mission":
    """Move ``mission_id`` to ``to``.

    Raises :class:`InvalidTransition` on an illegal move. Transitioning to
    the current state is a no-op success (idempotent — safe to call twice).
    Persists the os state + appends ``{from, to, ts, note}`` to the
    mission's ``transition_log``, syncs the coarse ``MissionStatus`` via
    ``store.set_status``, and emits ``mission.transition`` on the bus.
    """
    target = str(to or "").upper()
    if target not in LEGAL_TRANSITIONS:
        raise InvalidTransition(f"unknown os state {to!r}")
    mission = store.get(mission_id)  # raises NotFound when unknown
    current = current_state(mission)
    if current == target:
        return mission  # idempotent: already there, nothing to do
    if target not in LEGAL_TRANSITIONS[current]:
        raise InvalidTransition(
            f"illegal mission transition {current} -> {target}")
    mission.state[_OS_STATE_KEY] = target
    log = mission.state.setdefault(_TRANSITION_LOG_KEY, [])
    entry = {"from": current, "to": target, "ts": time.time(), "note": note}
    log.append(entry)
    # Save the os fields first: set_status re-reads the row and would drop
    # in-memory-only mutations.
    store.save(mission)
    mission = store.set_status(mission_id, OS_STATE_TO_MISSION_STATUS[target],
                               note=note or f"os: {current}->{target}")
    _emit_transition(mission_id, current, target, note)
    return mission


def _emit_transition(mission_id: str, from_state: str, to_state: str,
                     note: str) -> None:
    try:
        global_bus.publish(Event(
            topic="mission.transition",
            data={"mission_id": mission_id, "from_state": from_state,
                  "to_state": to_state, "note": note},
            source="nomorals.os.mission_state",
        ))
    except Exception:  # noqa: BLE001 - events never break transitions
        _log.debug("mission.transition event failed", exc_info=True)


def attach_runner(runner: Any, store: "MissionStore") -> Callable[..., None]:
    """Bind a ``MissionRunner`` to this state machine.

    Duck-typed: ``runner`` is never imported (missions is L5, os is L6 —
    and the runner must not import os either). Sets
    ``runner._os_transition_hook`` to ``(mission_id, to_state, note="")``.
    Illegal moves raise :class:`InvalidTransition` to the hook's caller;
    the runner itself invokes the hook defensively and never breaks a run
    over it. Returns the installed hook.
    """
    def hook(mission_id: str, to_state: str, note: str = "") -> "Mission":
        return transition(store, mission_id, to_state, note=note)

    runner._os_transition_hook = hook
    return hook


# ── acceptance ───────────────────────────────────────────────────────────────

@dataclass
class MissionAcceptance:
    """What must hold for a mission to count as *correct*.

    ``criteria`` reuse the :class:`~nomorals.core.tasks.AcceptanceCriterion`
    spec kinds (metric gte/lte/eq, artifact, assertion, test_suite, manual);
    ``required_artifact_types`` names artifact types that must exist in
    ``store.for_mission(mission.id)``.
    """

    criteria: list[AcceptanceCriterion] = field(default_factory=list)
    required_artifact_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": [c.to_dict() for c in self.criteria],
            "required_artifact_types": list(self.required_artifact_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionAcceptance":
        return cls(
            criteria=[AcceptanceCriterion.from_dict(c)
                      for c in data.get("criteria") or []],
            required_artifact_types=list(
                data.get("required_artifact_types") or []),
        )


def evaluate_acceptance(mission: "Mission",
                        artifact_store: "ArtifactStore",
                        acceptance: MissionAcceptance
                        ) -> tuple[bool, dict[str, Any]]:
    """Check a mission against ``acceptance``.

    Criteria are evaluated through the same ``TaskResult.verify`` patterns
    tasks use: the result is assembled from the mission's persisted state
    (``state["evidence"]`` / ``state["assertions"]`` / ``state["tests"]`` /
    ``state["metrics"]``, plus every artifact URI the mission produced).
    Returns ``(passed, details)``.
    """
    artifacts = artifact_store.for_mission(mission.id)
    state = mission.state or {}
    result = TaskResult(
        status="done" if str(mission.status) == "done" else str(mission.status),
        artifacts=[a.uri for a in artifacts],
        evidence=dict(state.get("evidence") or {}),
        assertions=list(state.get("assertions") or []),
        tests=dict(state.get("tests") or {}),
        metrics=dict(state.get("metrics") or {}),
    )
    criterion_results = result.verify(acceptance.criteria)
    required_failed = [r for r in criterion_results
                       if r.required and not r.passed]
    present_types = sorted({a.type for a in artifacts})
    missing_types = [t for t in acceptance.required_artifact_types
                     if t not in present_types]
    passed = not required_failed and not missing_types
    details: dict[str, Any] = {
        "passed": passed,
        "criteria": [r.to_dict() for r in criterion_results],
        "required_criteria_failed": [r.name for r in required_failed],
        "artifact_types": {
            "required": list(acceptance.required_artifact_types),
            "present": present_types,
            "missing": missing_types,
        },
    }
    return passed, details


# ── verify / repair loop ─────────────────────────────────────────────────────

def verify_repair_loop(
    *,
    verifier: Any,
    planner: Callable[[], dict[str, Any]],
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    repairer: Callable[[dict[str, Any], Any], dict[str, Any]],
    max_repair_rounds: int = 3,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """plan -> execute -> observe -> verify -> repair, until it passes.

    ``verifier`` is duck-typed: anything with ``.verify(target: dict)``
    returning an object with ``.passed`` (the :class:`Verdict` in
    ``nomorals.os.verifiers`` is the canonical one). The verifier sees
    ``{**target, "plan", "observation"}``. On failure the ``repairer``
    turns ``(observation, verdict)`` into a fresh plan and the loop runs
    again, at most ``max_repair_rounds`` repairs after the first attempt.

    Returns a report dict: ``passed``, ``rounds`` (attempts run),
    ``repairs`` (how many repairs were applied), ``verdicts`` (one per
    attempt), plus the final ``plan`` and ``observation``.
    """
    base_target = dict(target or {})
    verdicts: list[dict[str, Any]] = []
    plan = planner()
    observation: dict[str, Any] = {}
    repairs = 0
    passed = False

    for attempt in range(max_repair_rounds + 1):
        observation = executor(plan)
        verdict = verifier.verify({**base_target, "plan": plan,
                                   "observation": observation})
        verdicts.append(_verdict_dict(verdict))
        if bool(getattr(verdict, "passed", False)):
            passed = True
            break
        if attempt >= max_repair_rounds:
            break
        plan = repairer(observation, verdict)
        repairs += 1

    return {
        "passed": passed,
        "rounds": len(verdicts),
        "repairs": repairs,
        "verdicts": verdicts,
        "plan": plan,
        "observation": observation,
    }


def _verdict_dict(verdict: Any) -> dict[str, Any]:
    to_dict = getattr(verdict, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:  # noqa: BLE001 - fall through to the manual shape
            pass
    return {
        "passed": bool(getattr(verdict, "passed", False)),
        "details": str(getattr(verdict, "details", "")),
        "artifacts": list(getattr(verdict, "artifacts", []) or []),
    }
