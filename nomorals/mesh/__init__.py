"""Device mesh: phone-to-cloud worker coordination.

Layer L5. Lets the user's phone (Termux) and cloud nodes dispatch work to
each other and track presence via heartbeats.

The task queue reuses :class:`nomorals.storage.queue.WorkQueue` (durable,
leased, at-least-once). Node targeting is a payload field; a node polls for
tasks addressed to it, or untargeted (broadcast) tasks, which are claimed
by the first node that polls — work-stealing, not fan-out.
"""

from __future__ import annotations

from .errors import MeshError, NodeUnknown, TransportError
from .http_transport import HttpTransport
from .node import MeshNode, NodeRegistry
from .tasks import MeshTask, MeshTasks
from .transport import LocalTransport, Transport

__all__ = [
    "MeshError",
    "NodeUnknown",
    "TransportError",
    "MeshNode",
    "NodeRegistry",
    "MeshTask",
    "MeshTasks",
    "Transport",
    "LocalTransport",
    "HttpTransport",
]
