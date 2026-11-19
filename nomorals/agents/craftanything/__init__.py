"""Craft-Anything — the general tools/agents surface (census §3A.18).

Port of the *design* of storytold/craft-anything (an open-source clean-room
AI build engine in pure Rust, dual MIT/Apache-2.0, upstream commit
078cf0c4842c72d0b3d86a8c208515e2223510cc): the single ToolSpec contract
every surface dispatches through, declared side effects with consent
derived from them, toolsets that keep an agent's context small, one
Engine.call every caller goes through, and the agent loop that plans
multi-step work over that engine. The code here is Devon's own Python;
the design follows upstream, and every ported unit is pinned in
``craftanything-attribution.toml``.

This organ GENERALISES Devon's command registry — it does not replace
it. :mod:`nomorals.tools.registry` stays the registry of record (it
owns registration, capability gating, the audit trail and idempotency);
this organ adds what it never had: declared effects on each ToolSpec,
a consent gate per effect class with a journal, toolsets, one
caller-aware dispatch (``CraftEngine.call``), evidence retention for
over-limit results, and an agent that plans and runs multi-step tasks
through that dispatch. The engine takes the live registry as an
injected object (duck-typed ``get``/``call``) — it never imports the
registry, so the organ stays a leaf over ``core`` and the registry
stays the single source of tools.
"""

from __future__ import annotations

from . import agent as agent
from . import consent as consent
from . import effects as effects
from . import engine as engine
from . import specs as specs
from . import toolsets as toolsets
from .agent import CraftAgent, Plan, PlanStep
from .consent import Consent, ConsentDecision, ConsentGate
from .effects import (
    ACCESSES_NETWORK,
    CARRIES_TARGET_CONTENT,
    CHANGES_UI_STATE,
    EVIDENCE,
    EXECUTES_FOREIGN_CODE,
    IDEMPOTENT,
    LAUNCHES_PROCESS,
    MAY_DISCARD_DATA,
    MUTATES_DEVICE,
    MUTATES_SESSION,
    MUTATES_TARGET,
    NATIVE_EVIDENCE,
    NETWORK_EVIDENCE,
    READ_ONLY,
    SEND_MESSAGE,
    SPENDS_MONEY,
    WRITES_FILESYSTEM,
)
from .engine import Caller, CallOptions, CancelToken, CraftEngine, Progress
from .toolsets import Toolset

__all__ = [
    "agent", "consent", "effects", "engine", "specs", "toolsets",
    "CraftAgent", "Plan", "PlanStep",
    "Consent", "ConsentDecision", "ConsentGate",
    "MUTATES_SESSION", "MUTATES_TARGET", "WRITES_FILESYSTEM",
    "LAUNCHES_PROCESS", "ACCESSES_NETWORK", "CHANGES_UI_STATE",
    "MAY_DISCARD_DATA", "EXECUTES_FOREIGN_CODE", "MUTATES_DEVICE",
    "IDEMPOTENT", "CARRIES_TARGET_CONTENT", "SEND_MESSAGE", "SPENDS_MONEY",
    "EVIDENCE", "NATIVE_EVIDENCE", "NETWORK_EVIDENCE", "READ_ONLY",
    "Caller", "CallOptions", "CancelToken", "CraftEngine", "Progress",
    "Toolset",
]
