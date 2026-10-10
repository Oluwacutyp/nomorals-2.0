"""Mesh transports: how nodes talk to the shared mesh state.

The local transport shares one database (single machine, or a shared
volume). The transport interface is what a future HTTP transport (phone
polling a cloud endpoint) implements — same calls, remote execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..storage.db import Database
from .errors import TaskNotFound
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
    def heartbeat(self, node_id: str,
                  info: dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    def deregister(self, node_id: str) -> None:
        """Graceful leave: remove the node immediately."""

    @abstractmethod
    def active_nodes(self) -> list[MeshNode]:
        ...

    @abstractmethod
    def ping(self) -> float:
        """Liveness probe of the mesh backend. Returns round-trip seconds."""

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
    def poll(self, node_id: str, *, batch: int = 5,
             wait: float = 0.0) -> list[MeshTask]:
        """Claim tasks. ``wait`` > 0 long-polls: keep asking until tasks
        arrive or the deadline passes (near-instant dispatch without
        hammering the backend)."""

    @abstractmethod
    def complete(self, job_id: str, result: Any = None) -> None:
        ...

    @abstractmethod
    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a live task. False when already terminal/missing."""

    @abstractmethod
    def result(self, job_id: str) -> Any:
        """Stored result of a finished task (None when unfinished)."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Queue introspection: per-topic and total counts by status."""

    def close(self) -> None:
        """Release transport resources. Default: nothing to release."""

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class LocalTransport(Transport):
    """All nodes share one database. Production-grade on one machine;
    the reference implementation every remote transport must match."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.nodes = NodeRegistry(db)
        self.tasks = MeshTasks(db, registry=self.nodes)

    def register(
        self, name: str, platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        return self.nodes.register(
            name, platform=platform, capabilities=capabilities, node_id=node_id
        )

    def heartbeat(self, node_id: str,
                  info: dict[str, Any] | None = None) -> None:
        self.nodes.heartbeat(node_id, info=info)

    def deregister(self, node_id: str) -> None:
        self.nodes.deregister(node_id)

    def active_nodes(self) -> list[MeshNode]:
        return self.nodes.list_active()

    def ping(self) -> float:
        import time
        start = time.perf_counter()
        self.db.query_one("SELECT 1 AS ok")
        return time.perf_counter() - start

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

    def poll(self, node_id: str, *, batch: int = 5,
             wait: float = 0.0) -> list[MeshTask]:
        if wait > 0:
            import time
            deadline = time.monotonic() + wait
            while True:
                tasks = self.tasks.poll(node_id, batch=batch)
                if tasks:
                    return tasks
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                time.sleep(min(0.25, remaining))
        return self.tasks.poll(node_id, batch=batch)

    def complete(self, job_id: str, result: Any = None) -> None:
        self.tasks.complete(job_id, result=result)

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        try:
            self.tasks.fail(job_id, error=error, retry=retry)
        except TaskNotFound:
            # Historical behavior: failing an unknown job id was a silent
            # no-op at the queue layer. Keep it lenient here so remote and
            # local transports agree.
            pass

    def cancel(self, job_id: str) -> bool:
        return self.tasks.cancel(job_id)

    def result(self, job_id: str) -> Any:
        return self.tasks.result(job_id)

    def stats(self) -> dict[str, Any]:
        return self.tasks.stats()
