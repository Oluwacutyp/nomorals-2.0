"""Battery/thermal/degradation-aware scheduling.

Layer L5. Wraps :class:`nomorals.os.resources.ResourceManager` (which does
the actual sampling; injected, never imported — L5 must not import L6)
and :class:`nomorals.storage.queue.WorkQueue` (which does the durable
queuing).

Three power classes: ``light`` always flows; ``medium`` flows while
degradation is mild (level <= 1); ``heavy`` flows only at degradation
level 0 with a clean advisory.  Tasks also carry a tier
(``critical`` | ``important`` | ``background`` | ``bulk``) shed
bottom-up by the degradation ladder, and an optional subsystem budget
(``llm``, ``media``, ``scheduler``, ...) gating admission.  Deferred
work waits durably — nothing is dropped, just deferred with jittered
exponential backoff.
"""

from __future__ import annotations

from .backoff import STRATEGIES, BackoffPolicy
from .energy import BASE_WATTS, WATTS_PER_MEM_MB, EnergyLedger
from .errors import (
    InvalidPowerSpecError,
    PowerBudgetExhaustedError,
    PowerConstrainedError,
    PowerError,
    PowerSamplerUnavailableError,
    TaskDeferredError,
)
from .monitor import (
    DEGRADATION_TIERS,
    SYSFS_POWER_SUPPLY,
    THERMAL_BANDS,
    PowerMonitor,
    PowerStatus,
    local_sampler,
)
from .scheduler import (
    CONSTRAINT_KEYS,
    HEAVY_TOPIC,
    LIGHT_TOPIC,
    MEDIUM_TOPIC,
    POWER_CLASSES,
    TASK_TIERS,
    PowerAwareScheduler,
    PowerTask,
)
from .telemetry import emit_event

__all__ = [
    "PowerError",
    "TaskDeferredError",
    "PowerConstrainedError",
    "PowerBudgetExhaustedError",
    "PowerSamplerUnavailableError",
    "InvalidPowerSpecError",
    "BackoffPolicy",
    "STRATEGIES",
    "EnergyLedger",
    "BASE_WATTS",
    "WATTS_PER_MEM_MB",
    "PowerMonitor",
    "PowerStatus",
    "local_sampler",
    "SYSFS_POWER_SUPPLY",
    "THERMAL_BANDS",
    "DEGRADATION_TIERS",
    "PowerAwareScheduler",
    "PowerTask",
    "LIGHT_TOPIC",
    "MEDIUM_TOPIC",
    "HEAVY_TOPIC",
    "POWER_CLASSES",
    "TASK_TIERS",
    "CONSTRAINT_KEYS",
    "emit_event",
]
