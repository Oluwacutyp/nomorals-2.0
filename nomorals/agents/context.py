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
from typing import Any

from ..core.config import Settings, get_settings
from ..core.events import EventBus
from ..core.logging_setup import get_logger
from ..core.observability import Metrics, Tracer
from ..core.policy import CapabilitySet, Policy
from ..core.ratelimit import SemaphorePool
from ..storage.db import Database, open_database

__all__ = ["AgentContext", "build_context", "build_router",
           "persist_provider_override", "ensure_local_gguf"]

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
    ) -> AgentContext:
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
                # Graceful, bounded shutdown: cancel pending futures and
                # reap process-pool children. The old wait=False orphaned
                # forked workers whenever a command exited with pool work
                # still queued (battery/CPU drain on the phone). wait=True
                # matches HybridExecutor.__exit__; running thread tasks are
                # waited the same way interpreter exit would wait for them.
                shutdown = getattr(executor, "shutdown", None)
                if shutdown is not None:
                    try:
                        shutdown(wait=True, timeout=10.0)
                    except TypeError:
                        shutdown()  # exotic executor without kwargs
            except Exception as exc:  # noqa: BLE001
                _log.debug("executor shutdown: %s", exc)
        try:
            self.db.close()
        except Exception as exc:  # noqa: BLE001
            _log.debug("db close: %s", exc)
        self.bus.stop()

    def __enter__(self) -> AgentContext:
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
    # Ensure data directories exist before the database tries to open.
    # (Fresh installs and new profiles won't have them yet.)
    settings.ensure_dirs()
    # open_database quarantines a corrupt file (moved aside, never deleted)
    # instead of bricking the boot — a killed-mid-checkpoint phone DB must
    # not require the owner to SSH in and hand-delete it.
    db_recovered = False
    db_backup_path: str | None = None
    if db is None:
        database, db_recovered, db_backup_path = open_database(
            settings.db_path,
            wal=settings.storage.wal,
            busy_timeout_ms=settings.storage.busy_timeout_ms,
            synchronous=settings.storage.synchronous,
        )
    else:
        database = db
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
        context.router = build_router(settings, event_bus, db=context.db)
        # Startup provider verification: probe each provider with a cheap
        # call to catch config errors (bad model names, bad keys) at boot
        # instead of at chat time. Best-effort — never breaks startup.
        if getattr(settings.llm, "verify_at_startup", True):
            try:
                report = context.router.verify()
                if not report.get("ok"):
                    _log.error(
                        "startup: no working LLM providers! "
                        "Devon will use fallbacks until a provider is fixed."
                    )
            except Exception as exc:  # noqa: BLE001 - verification is advisory
                _log.warning("startup provider verification failed: %s", exc)
        # Local GGUF auto-start: if the operator promoted a local model
        # (NM_LLM_PROVIDER=llama_cpp + NM_LLM_LOCAL_AUTO_START=1), make sure
        # llama-server is actually running before any chat traffic hits it.
        # Without this the router points at a dead localhost:8080.
        if settings.llm.provider.lower() in {"llama_cpp", "llamacpp", "gguf"}:
            try:
                ensure_local_gguf(settings)
            except Exception as exc:  # noqa: BLE001 — cloud fallbacks still work
                _log.warning("local GGUF auto-start failed: %s", exc)

    if with_memory:
        from ..memory.manager import MemoryManager

        context.memory = MemoryManager(context)

    if with_tools:
        from ..tools.registry import ToolRegistry

        context.tools = ToolRegistry(context)
        # Register eagerly. Without this, only the CLI ever populated the
        # registry and every agent ran with zero tools — the research step
        # failed with "unknown tool 'web_search'; available: []" while the CLI
        # looked perfectly healthy. register_builtins() is idempotent.
        context.tools.register_builtins()

        # Wire up the SideChatManager for the side_chat tool. Without this,
        # the tool is registered but raises "SideChatManager not initialized"
        # on every call.
        try:
            from ..social.chat.side_chats import SideChatManager
            from ..tools.side_chats import init_side_chats

            if getattr(context, "db", None) is not None:
                init_side_chats(SideChatManager(context.db))
        except Exception:  # noqa: BLE001 — side chats are optional
            pass

    from .blackboard import Blackboard

    context.blackboard = Blackboard()
    if db_recovered:
        # Surfaced in status/diagnostics: the owner should know their data
        # was quarantined, and where the old file went.
        context.extras["db_recovered_from_corrupt"] = True
        context.extras["db_corrupt_backup"] = db_backup_path
        _log.warning(
            "database was corrupt at boot; quarantined to %s and started "
            "fresh — old data is preserved in the quarantine file",
            db_backup_path,
        )
    return context


PROVIDER_OVERRIDE_KEY = "llm.provider_override"
FALLBACK_CHAIN_KEY = "llm.fallback_chain"


def persist_provider_override(db: Any, provider: str,
                              chain: list[str] | None = None) -> None:
    """Durably store the live provider switch (the /model control command).

    Lives in kv_store, not the .env file: the .env is the user's hand —
    rewriting it from chat would surprise them. The store is applied again
    at every boot (:func:`_apply_provider_override`), so a switch made mid-
    conversation survives a restart.
    """
    import json as _json

    provider = (provider or "").strip().lower()
    if not provider:
        raise ValueError("provider name required")
    import time as _time

    for key, payload in (
        (PROVIDER_OVERRIDE_KEY, {"provider": provider}),
        (FALLBACK_CHAIN_KEY, {"chain": [c.strip().lower() for c in (chain or []) if c.strip()]}),
    ):
        db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, _json.dumps(payload), _time.time()),
        )


def _apply_provider_override(db: Any, settings: Settings) -> None:
    """kv_store wins over the .env defaults for provider + fallback chain."""
    import json as _json

    try:
        row = db.query_one(
            "SELECT value FROM kv_store WHERE key = ?", (PROVIDER_OVERRIDE_KEY,))
        if row and row["value"]:
            provider = str(_json.loads(row["value"]).get("provider") or "").strip().lower()
            if provider:
                settings.llm.provider = provider
        crow = db.query_one(
            "SELECT value FROM kv_store WHERE key = ?", (FALLBACK_CHAIN_KEY,))
        if crow and crow["value"]:
            chain = [str(c).strip().lower() for c in
                     (_json.loads(crow["value"]).get("chain") or []) if str(c).strip()]
            if chain:
                settings.llm.fallback_chain = chain
    except Exception as exc:  # noqa: BLE001 — a broken override must not stop the boot
        _log.debug("provider override lookup failed; using .env defaults: %s", exc)


def build_router(settings: Settings, bus: EventBus, *, db: Any | None = None,
                 context: Any | None = None) -> Any:
    """Build the provider chain, wiring each backend's own credentials.

    Each provider kind needs different arguments; passing one flat kwargs dict
    would either drop the HF token or send an api_key to a provider that does not
    use one.

    Public entry point for the same builder ``build_context`` uses: the CLI
    setup wizard (``nm setup``) rebuilds the router after writing new
    credentials, and partner-runtime hot-swaps use it for live model changes.
    """
    from ..llm.providers import build_provider
    from ..llm.router import LLMRouter

    if db is not None:
        # a live /model switch (or `nm models --set-provider`) beats .env
        _apply_provider_override(db, settings)
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
        if kind == "groq":
            return {
                **common,
                "base_url": llm.groq_base_url,
                "api_key": llm.groq_api_key,
                "model": llm.groq_model,
            }
        if kind == "openrouter":
            return {
                **common,
                "base_url": llm.openrouter_base_url,
                "api_key": llm.openrouter_api_key,
                "model": llm.openrouter_model,
                # OpenRouter asks apps to identify themselves; harmless, helps
                # with their abuse handling and analytics.
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/Oluwacutyp/nomorals-2.0",
                    "X-Title": "Devon",
                },
            }
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
        # Keyed cloud fallbacks with no key configured are skipped quietly —
        # a provider that can only fail at call time just adds noise to the
        # failover chain. Set the key and it registers on next boot.
        if kind == "groq" and not llm.groq_api_key:
            _log.info("skipping groq fallback: NM_GROQ_API_KEY not set")
            return
        if kind == "openrouter" and not llm.openrouter_api_key:
            _log.info("skipping openrouter fallback: NM_OPENROUTER_API_KEY not set")
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
        if llm.allow_mock_fallback:
            add("mock", primary=True)
        else:
            # No silent fake: a misconfigured chain boots model-less, and the
            # error surfaces where it belongs (loudly, at call time) instead
            # of a scripted mock answering "how are you?" with fake warmth.
            _log.warning(
                "no usable model provider configured — running model-less. "
                "Fix NM_LLM_PROVIDER/NM_LLM_FALLBACK_CHAIN, or opt into the "
                "scripted test mock with NM_LLM_ALLOW_MOCK_FALLBACK=1.")
    # The OCR floor is always in the chain: free, offline, and the only
    # "vision" that works without a VLM. It is a fallback, never primary —
    # real models describe first, tesseract reads the pixels when they can't.
    if "ocr" not in registered:
        try:
            router.add(
                build_provider(
                    "ocr",
                    binary=settings.vision.ocr_binary,
                    language=settings.vision.ocr_language,
                ),
                name="ocr",
            )
            registered.add("ocr")
        except Exception as exc:  # noqa: BLE001 — the floor itself may be absent
            _log.warning("could not register ocr fallback: %s", exc)
    # Capability-based routing: attach a ModelBroker so the router can select
    # the best provider per operation (not just failover order).
    # Best-effort — broker must never break boot or routing.
    #
    # The broker is built FIRST so attach_learning() below finds it on
    # router.broker and injects the trajectory store into it. The reverse
    # order used to build a second, store-less broker here that silently
    # replaced the learning-attached one — live serving outcomes never
    # reached the broker's ranking evidence.
    try:
        from ..llm.broker import ModelBroker
        broker = ModelBroker()
        broker.build_from_router(router)
        # Also register lifecycle-managed models (local GGUFs etc.)
        try:
            from ..llm.lifecycle import ModelLifecycle
            from ..cmdline.commands.models import _card_for
            lc = ModelLifecycle(db)
            for model in lc.list():
                try:
                    broker.register(_card_for(model))
                except Exception:
                    pass
            primary = lc.primary
            if primary and broker.card(primary) is not None:
                broker.promote(primary)
        except Exception:
            _log.warning("lifecycle model registration failed", exc_info=True)
        router.set_broker(broker)
    except Exception:  # noqa: BLE001
        _log.warning("broker attach failed; continuing without it", exc_info=True)
    # Wave J: feed live serving outcomes into the trajectory store and
    # benchmark DB so the broker ranks on evidence instead of priors.
    # Best-effort — learning must never break boot or routing.
    try:
        from ..llm.learning import attach_learning
        attach_learning(router, db=db)
    except Exception:  # noqa: BLE001
        _log.warning("learning attach failed; continuing without it", exc_info=True)
    # Console debug view (key d): record LLM call traces. Chain-safe —
    # the learning hook above keeps working; this only adds telemetry.
    # Best-effort — must never break boot or routing.
    try:
        from ..console.debug import DebugHub
        DebugHub.install_llm_hook(router)
    except Exception:  # noqa: BLE001
        pass
    return router


# Compatibility alias: partner runtime and older call sites import the
# private name.  Both names refer to the same builder.
_build_router = build_router


def ensure_local_gguf(model_path_or_settings, **kwargs):
    """Ensure a local GGUF model is running and return the server manager.

    Args:
        model_path_or_settings: Path to the GGUF model file or Settings object
        **kwargs: Additional parameters (ignored)

    Returns:
        GGUFServerManager if auto-start is enabled and server is running/healed,
        None otherwise, or the model path string if it's just a path
    """
    import os

    # Handle Settings object
    model_path = model_path_or_settings
    if hasattr(model_path_or_settings, 'llm'):
        model_path = getattr(model_path_or_settings.llm, 'local_model', '')
        if not model_path:
            return None

        # Check if auto-start is enabled
        auto_start = getattr(model_path_or_settings.llm, 'local_auto_start', False)
        if auto_start:
            # Try to start or heal the server
            try:
                from nomorals.llm import local_server as ls
                port = getattr(model_path_or_settings.llm, 'local_port', 8080)
                host = getattr(model_path_or_settings.llm, 'local_host', '127.0.0.1')

                # Check if port is in use
                if ls.port_in_use(host, port):
                    # Port is in use, check if healthy
                    mgr = ls.GGUFServerManager(
                        model_path=model_path,
                        port=port,
                        host=host,
                    )
                    if mgr._healthy():
                        # Already healthy, don't interfere
                        return None
                    # Not healthy, try to heal
                    diagnosis = mgr.heal()
                    if diagnosis and hasattr(diagnosis, 'model_path'):
                        mgr.model_path = diagnosis.model_path
                    return mgr
                else:
                    # Port is free, start a new server
                    mgr = ls.GGUFServerManager(
                        model_path=model_path,
                        port=port,
                        host=host,
                    )
                    diagnosis = mgr.heal()
                    if diagnosis and diagnosis.ok:
                        if hasattr(diagnosis, 'model_path'):
                            mgr.model_path = diagnosis.model_path
                        return mgr
            except Exception as e:
                _log.warning("GGUF server heal/start failed for %s: %s",
                             model_path, e)

        return model_path if os.path.exists(model_path) else None

    # Just a path string
    if not model_path:
        return None

    if os.path.exists(model_path):
        return model_path
    return None
