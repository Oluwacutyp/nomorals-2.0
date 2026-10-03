"""Battery/thermal-aware scheduling.

Layer L5. Wraps :class:`nomorals.os.resources.ResourceManager` (which does
the actual sampling) and :class:`nomorals.storage.queue.WorkQueue` (which
does the durable queuing). Heavy work is deferred when the battery is low
or the device is thermally throttled; light work always flows.
"""

from __future__ import annotations

from .errors import PowerError
from .monitor import PowerMonitor, PowerStatus
from .scheduler import PowerAwareScheduler

__all__ = ["PowerError", "PowerMonitor", "PowerStatus", "PowerAwareScheduler"]
