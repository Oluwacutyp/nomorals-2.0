"""Mesh transports: how nodes talk to the shared mesh state.

The local transport shares one database (single machine, or a shared
volume). The transport interface is what a future HTTP transport (phone
polling a cloud endpoint) implements — same calls, remote execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..storage.db import Database
from .node import MeshNode, NodeRegistry
from .tasks import MeshTask, MeshTasks

__all__ = ["Transport", "LocalTransport"]


class Transport(ABC):
    """What a mesh node needs from the outside world."""

    @abstractmethod
    def register(
        self, name: str, platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        ...

    @abstractmethod
    def heartbeat(self, node_id: str) -> None:
        ...

    @abstractmethod
    def active_nodes(self) -> list[MeshNode]:
        ...

    @abstractmethod
    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        origin_node: str,
        target_node: str | None = None,
        priority: int = 0,
    ) -> str:
        ...

    @abstractmethod
    def poll(self, node_id: str, *, batch: int = 5) -> list[MeshTask]:
        ...

    @abstractmethod
    def complete(self, job_id: str, result: Any = None) -> None:
        ...

    @abstractmethod
    def fail(self, job_id: str, error: str = "") -> None:
        ...


class LocalTransport(Transport):
    """All nodes share one database. Production-grade on one machine;
    the reference implementation every remote transport must match."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.nodes = NodeRegistry(db)
        self.tasks = MeshTasks(db)

    def register(
        self, name: str, platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        return self.nodes.register(
            name, platform=platform, capabilities=capabilities, node_id=node_id
        )

    def heartbeat(self, node_id: str) -> None:
        self.nodes.heartbeat(node_id)

    def active_nodes(self) -> list[MeshNode]:
        return self.nodes.list_active()

    def dispatch(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        *,
        origin_node: str,
        target_node: str | None = None,
        priority: int = 0,
    ) -> str:
        return self.tasks.dispatch(
            task_type, payload,
            origin_node=origin_node, target_node=target_node, priority=priority,
        )

    def poll(self, node_id: str, *, batch: int = 5) -> list[MeshTask]:
        return self.tasks.poll(node_id, batch=batch)

    def complete(self, job_id: str, result: Any = None) -> None:
        self.tasks.complete(job_id, result=result)

    def fail(self, job_id: str, error: str = "") -> None:
        self.tasks.fail(job_id, error=error)
