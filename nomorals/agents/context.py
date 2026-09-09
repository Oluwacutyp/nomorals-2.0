"""The dependency container.

No global singletons. Everything an agent can touch — database, event bus,
executor, policy, model router, memory, tools — is handed to it through one
explicit object. That is what makes the whole system testable in parallel: two
tests can build two independent contexts and they cannot interfere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.config import Settings, get_settings
from ..core.events import EventBus
from ..core.logging_setup import get_logger
from ..core.observability import Metrics, Tracer
from ..core.policy import CapabilitySet, Policy
from ..core.ratelimit import SemaphorePool
from ..storage.db import Database

__all__ = ["AgentContext"]

_log = get_logger(__name__)


@dataclass
class AgentContext:
    """Everything an agent is allowed to reach, in one object.

    Constructed once per process by :func:`build_context` and then threaded
    through the orchestrator into every agent and sub-agent.
    """

    settings: Settings
    db: Database
    bus: EventBus
    metrics: Metrics = field(default_factory=Metrics)
    tracer: Tracer = field(default_factory=Tracer)
    policy: Policy | None = None
    permits: SemaphorePool | None = None
    executor: Any = None
    router: Any = None
    memory: Any = None
    tools: Any = None
    blackboard: Any = None

    started_at: float = field(default_factory=time.time)
    actor: str = "system"
    capabilities: CapabilitySet = field(default_factory=CapabilitySet.all)
    extras: dict[str, Any] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── policy ───────────────────────────────────────────────────────────────
    def ensure_policy(self) -> Policy:
        if self.policy is None:
            with self._lock:
                if self.policy is None:
                    self.policy = Policy(default_grant=self.capabilities)
        return self.policy

    def check(self, capability: str, *, actor: str = "", confirmation: str | None = None) -> Any:
        """Ask the policy whether ``capability`` may be exercised."""
        return self.ensure_policy().check(
            capability, actor=actor or self.actor, grant=self.capabilities, confirmation=confirmation
        )

    def require(self, capability: str, *, actor: str = "", confirmation: str | None = None) -> None:
        self.ensure_policy().require(
            capability, actor=actor or self.actor, grant=self.capabilities, confirmation=confirmation
        )

    # ── concurrency ──────────────────────────────────────────────────────────
    def ensure_permits(self) -> SemaphorePool:
        if self.permits is None:
            with self._lock:
                if self.permits is None:
                    self.permits = SemaphorePool(
                        {
                            "network": self.settings.concurrency.network_permits,
                            "disk": self.settings.concurrency.disk_permits,
                            "gpu": self.settings.concurrency.gpu_permits,
                        }
                    )
        return self.permits

    def permit(self, resource: str, timeout: float | None = None):
        """``with ctx.permit('network'): ...``"""
        return self.ensure_permits().slot(resource, timeout)

    # ── child contexts ───────────────────────────────────────────────────────
    def child(
        self,
        *,
        actor: str = "",
        capabilities: CapabilitySet | None = None,
        **extras: Any,
    ) -> "AgentContext":
        """Derive a scoped context for a sub-agent.

        Capabilities can only narrow: the child receives the intersection of what
        it asked for and what this context holds.
        """
        grant = self.capabilities if capabilities is None else capabilities.intersect(self.capabilities)
        return AgentContext(
            settings=self.settings,
            db=self.db,
            bus=self.bus,
            metrics=self.metrics,
            tracer=self.tracer,
            policy=self.policy,
            permits=self.permits,
            executor=self.executor,
            router=self.router,
            memory=self.memory,
            tools=self.tools,
            blackboard=self.blackboard,
            started_at=time.time(),
            actor=actor or f"{self.actor}.child",
            capabilities=grant,
            extras={**self.extras, **extras},
        )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def emit(self, topic: str, **data: Any) -> None:
        self.bus.emit(topic, actor=self.actor, **data)

    def close(self) -> None:
        """Release resources this context owns. Idempotent."""
        executor, self.executor = self.executor, None
        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                _log.debug("executor shutdown: %s", exc)
        try:
            self.db.close()
        except Exception as exc:  # noqa: BLE001
            _log.debug("db close: %s", exc)
        self.bus.stop()

    def __enter__(self) -> "AgentContext":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── reporting ────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "uptime": round(time.time() - self.started_at, 2),
            "capabilities": self.capabilities.as_list(),
            "db": self.db.stats_snapshot(),
            "bus": self.bus.stats(),
            "metrics": self.metrics.snapshot(),
            "permits": self.permits.snapshot() if self.permits else {},
            "policy": self.policy.stats() if self.policy else {},
        }


def build_context(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    bus: EventBus | None = None,
    with_executor: bool = True,
    with_router: bool = True,
    with_memory: bool = True,
    with_tools: bool = True,
    capabilities: CapabilitySet | None = None,
    actor: str = "system",
) -> AgentContext:
    """Assemble a fully wired context.

    Sub-systems are optional so a test can build a context with a database and
    nothing else. Imports are deferred: constructing a context must not pull in
    torch or yt-dlp.
    """
    settings = settings or get_settings()
    database = db or Database(
        settings.db_path,
        wal=settings.storage.wal,
        busy_timeout_ms=settings.storage.busy_timeout_ms,
        synchronous=settings.storage.synchronous,
    )
    database.migrate()
    event_bus = bus or EventBus().start()

    context = AgentContext(
        settings=settings,
        db=database,
        bus=event_bus,
        policy=Policy(default_grant=capabilities or CapabilitySet.all()),
        capabilities=capabilities or CapabilitySet.all(),
        actor=actor,
    )
    context.ensure_permits()

    if with_executor:
        from .runtime import HybridExecutor

        context.executor = HybridExecutor(
            threads=settings.concurrency.threads,
            processes=settings.concurrency.processes,
            use_processes=settings.concurrency.use_processes,
            max_in_flight=settings.concurrency.max_subagents,
        )

    if with_router:
        context.router = _build_router(settings, event_bus)

    if with_memory:
        from ..memory.manager import MemoryManager

        context.memory = MemoryManager(context)

    if with_tools:
        from ..tools.registry import ToolRegistry

        context.tools = ToolRegistry(context)

    from .blackboard import Blackboard

    context.blackboard = Blackboard()
    return context


def _build_router(settings: Settings, bus: EventBus) -> Any:
    """Build the provider chain, wiring each backend's own credentials.

    Each provider kind needs different arguments; passing one flat kwargs dict
    would either drop the HF token or send an api_key to a provider that does not
    use one.
    """
    from ..llm.providers import build_provider
    from ..llm.router import LLMRouter

    llm = settings.llm

    def kwargs_for(kind: str) -> dict[str, Any]:
        kind = kind.lower()
        common = {"timeout": llm.timeout}
        if kind in {"hf", "hf_serverless", "huggingface", "hf_endpoint"}:
            return {
                **common,
                "token": llm.hf_token,
                "model": llm.hf_model,
                "base_url": llm.hf_base_url,
                "endpoint_url": llm.hf_endpoint_url,
            }
        if kind in {"llama_cpp", "llamacpp", "gguf"}:
            return {**common, "base_url": llm.llama_cpp_url, "model": llm.openai_model or "local"}
        if kind in {"mock", "offline", "test"}:
            return {**common, "model": llm.openai_model or "mock-7b"}
        return {
            **common,
            "base_url": llm.openai_base_url,
            "api_key": llm.openai_api_key,
            "model": llm.openai_model,
        }

    router = LLMRouter(bus=bus)
    registered: set[str] = set()

    def add(kind: str, *, primary: bool = False) -> None:
        if kind in registered:
            return
        try:
            router.add(build_provider(kind, **kwargs_for(kind)), primary=primary, name=kind)
            registered.add(kind)
        except Exception as exc:  # noqa: BLE001 - one bad backend must not block startup
            _log.warning("could not register provider %s: %s", kind, exc)

    add(llm.provider, primary=True)
    for fallback in llm.fallback_chain:
        add(fallback)
    if not router.providers():
        # Never leave the system without a model: the offline mock always works.
        add("mock", primary=True)
    return router
