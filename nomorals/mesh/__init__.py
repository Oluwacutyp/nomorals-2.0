"""Device mesh: phone-to-cloud worker coordination.

Layer L5. Lets the user's phone (Termux) and cloud nodes dispatch work to
each other and track presence via heartbeats.

The task queue reuses :class:`nomorals.storage.queue.WorkQueue` (durable,
leased, at-least-once). Node targeting is a payload field; a node polls for
tasks addressed to it, or untargeted (broadcast) tasks, which are claimed
by the first node that polls — work-stealing, not fan-out.

Presence is three-state (``ready`` → ``suspect`` → ``gone``), and tasks
support Temporal-style heartbeats, declarative retry policies, dedupe
keys, schedule-to-start expiry, and capability-aware routing.
"""

from __future__ import annotations

from .errors import (
    AuthError,
    CircuitOpen,
    HubUnreachable,
    MeshError,
    NodeSuspect,
    NodeUnknown,
    PayloadTooLarge,
    TaskCancelled,
    TaskExpired,
    TaskNotFound,
    TransportError,
)
from .http_transport import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    CircuitBreaker,
    HttpTransport,
)
from .node import MeshNode, NodeRegistry, format_nodes_table
from .tasks import (
    BROADCAST_TOPIC,
    MESH_MAX_PAYLOAD_BYTES,
    MeshTask,
    MeshTasks,
    RetryPolicy,
    format_tasks_table,
)
from .transport import LocalTransport, Transport

__all__ = [
    "MeshError",
    "NodeUnknown",
    "NodeSuspect",
    "TransportError",
    "HubUnreachable",
    "AuthError",
    "CircuitOpen",
    "TaskNotFound",
    "TaskExpired",
    "TaskCancelled",
    "PayloadTooLarge",
    "MeshNode",
    "NodeRegistry",
    "format_nodes_table",
    "MeshTask",
    "MeshTasks",
    "RetryPolicy",
    "format_tasks_table",
    "BROADCAST_TOPIC",
    "MESH_MAX_PAYLOAD_BYTES",
    "Transport",
    "LocalTransport",
    "HttpTransport",
    "CircuitBreaker",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
]
