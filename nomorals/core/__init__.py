"""L1 kernel primitives.

Nothing in this subpackage may import from any other ``nomorals`` subpackage.
"""

from __future__ import annotations

from .clock import Clock, MonotonicClock, SystemClock
from .errors import (
    NoMoralsError,
    BudgetExceeded,
    CapabilityDenied,
    ConfigError,
    ModelError,
    NotFound,
    ProviderError,
    StorageError,
    TaskCancelled,
    ToolError,
    ValidationError,
)
from .events import Event, EventBus
from .ids import new_id, new_short_id, ulid_now
from .result import Err, Ok, Outcome, outcome_from
from .policy import Capability, CapabilitySet, Policy, PolicyDecision

__all__ = [
    "BudgetExceeded",
    "Capability",
    "CapabilityDenied",
    "CapabilitySet",
    "Clock",
    "ConfigError",
    "Err",
    "Event",
    "EventBus",
    "ModelError",
    "MonotonicClock",
    "NoMoralsError",
    "NotFound",
    "Ok",
    "Outcome",
    "Policy",
    "PolicyDecision",
    "ProviderError",
    "StorageError",
    "SystemClock",
    "TaskCancelled",
    "ToolError",
    "ValidationError",
    "new_id",
    "new_short_id",
    "outcome_from",
    "ulid_now",
]
