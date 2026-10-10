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
work waits durably — nothing is dropped, just deferred with backoff.
"""

from __future__ import annotations

from .errors import PowerError
from .monitor import PowerMonitor, PowerStatus
from .scheduler import (
    HEAVY_TOPIC,
    LIGHT_TOPIC,
    MEDIUM_TOPIC,
    POWER_CLASSES,
    TASK_TIERS,
    PowerAwareScheduler,
    PowerTask,
)

__all__ = [
    "PowerError",
    "PowerMonitor",
    "PowerStatus",
    "PowerAwareScheduler",
    "PowerTask",
    "LIGHT_TOPIC",
    "MEDIUM_TOPIC",
    "HEAVY_TOPIC",
    "POWER_CLASSES",
    "TASK_TIERS",
]
