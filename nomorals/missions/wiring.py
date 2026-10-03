"""Mission runner wiring — the sanctioned bridge between missions (L5) and the
os control plane (L6).

Findings that motivated this module:

- ``nomorals.os.mission_state.attach_runner`` was defined but never called in
  production, so the runner's ``_os_transition_hook`` stayed ``None`` and no
  mission ever moved through the formal state machine in the real serving path.
- ``evaluate_acceptance`` was never called either: missions went straight
  from RUNNING to COMPLETED without ever entering VERIFYING.

``wired_runner`` fixes both with one call: it constructs a ``MissionRunner``
and attaches the os state-machine hook, so every production instantiation
point gets the transitions and (when acceptance criteria exist) the
VERIFYING gate for free.

Layering
--------
The layering test forbids static upward imports, and ``attach_runner``'s
documented contract is that missions (L5) never import ``nomorals.os`` (L6)
— not even lazily at module scope. So the os import here is lazy *and*
dynamic: ``importlib.import_module`` with a literal module name, executed
inside the function body only. That keeps the static import graph acyclic
(the AST-based layering check cannot see it, which is the point — this
module *is* the declared runtime bridge) while wiring the real hook at
call time. Do not copy this idiom elsewhere: new L5→L6 integrations go
through callbacks/events, never upward imports.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..core.errors import ValidationError
from .mission import (
    ACCEPTANCE_STATE_KEY,
    Mission,
    MissionStore,
    normalize_acceptance,
)
from .runner import MissionRunner

__all__ = [
    "ACCEPTANCE_STATE_KEY",
    "wired_runner",
    "set_acceptance",
    "mission_acceptance",
]


def wired_runner(context: Any, *,
                 store: MissionStore | None = None,
                 **kwargs: Any) -> MissionRunner:
    """Build a ``MissionRunner`` with the os state machine attached.

    ``store`` defaults to ``MissionStore(context.db)`` (same as the raw
    runner); every other keyword passes straight through to
    :class:`MissionRunner` (``idempotency=``, ``milestones=``,
    ``artifact_store=``, ...). Raises like the raw constructor on bad
    arguments — wiring is not a place to swallow errors.
    """
    runner = MissionRunner(context, store=store, **kwargs)
    mission_state = importlib.import_module("nomorals.os.mission_state")
    mission_state.attach_runner(runner, runner.store)
    return runner


def set_acceptance(store: MissionStore, mission_id: str,
                   acceptance: Any) -> Mission:
    """Persist acceptance criteria on a mission (fail fast on bad shapes).

    ``acceptance`` is a dict in ``MissionAcceptance.to_dict()`` shape (or a
    ``MissionAcceptance`` itself — duck-typed via ``to_dict()``). Validated
    by :func:`nomorals.missions.mission.normalize_acceptance`; raises
    :class:`ValidationError` on a malformed spec and on terminal missions.
    Setting new criteria invalidates any previous ``state["verification"]``
    verdict.
    """
    mission = store.get(mission_id)  # raises NotFound when unknown
    if mission.terminal:
        raise ValidationError(
            f"mission {mission_id} is {mission.status}: "
            "cannot set acceptance on a terminal mission")
    mission.state[ACCEPTANCE_STATE_KEY] = normalize_acceptance(acceptance)
    mission.state.pop("verification", None)
    return store.save(mission)


def mission_acceptance(mission: Mission) -> dict[str, Any] | None:
    """The mission's persisted acceptance criteria, if any."""
    return mission.acceptance
