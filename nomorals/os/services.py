"""Service registry for the OS control plane.

Named services are constructed lazily: the factory runs once, on first
: meth:`lookup`, and the instance is cached.  Every descriptor may carry a
health check, which the :class:`~nomorals.os.kernel.OSKernel` wires into
the :class:`~nomorals.os.health.HealthMonitor` at start-up.

The four built-in descriptors — ``browser``, ``model``, ``bridge``,
``queue`` — are sensible defaults whose factories import their
implementations lazily (function-level imports, so importing this module
never pulls in the browser, the LLM stack, or the queue).  Layer-7 entry
points may replace any descriptor with a fully-configured factory via
:meth:`register` (``replace=True``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .health import HealthCheck, as_health_check

__all__ = [
    "ServiceNotFound",
    "ServiceDescriptor",
    "ServiceRegistry",
    "default_db_path",
]


class ServiceNotFound(LookupError):
    """Raised by :meth:`ServiceRegistry.lookup` for an unknown service name."""


def default_db_path() -> Path:
    """Default on-disk home for OS-owned service state (``~/.nomorals``)."""
    return Path("~/.nomorals").expanduser() / "os-services.db"


# ── default lazy factories (function-level imports: no import-time cost) ──

def _factory_browser() -> Any:
    from ..tools.browser import BrowserSession
    return BrowserSession()


def _factory_model() -> Any:
    from ..llm.router import LLMRouter
    return LLMRouter()


def _factory_bridge() -> Any:
    from ..social.chat.whatsapp import WhatsAppAdapter
    return WhatsAppAdapter()


_WORK_QUEUE_DDL = """
-- Mirrors migrations v5 (durable work queue); idempotent by construction.
CREATE TABLE IF NOT EXISTS work_queue (
    id           TEXT PRIMARY KEY,
    topic        TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready',
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    available_at REAL NOT NULL DEFAULT 0,
    lease_until  REAL NOT NULL DEFAULT 0,
    lease_owner  TEXT NOT NULL DEFAULT '',
    result       TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_queue_ready
    ON work_queue(topic, status, priority DESC, available_at);
CREATE INDEX IF NOT EXISTS idx_work_queue_lease
    ON work_queue(status, lease_until);
"""


def _factory_queue(db_path: str | None) -> Any:
    from ..storage.db import Database
    from ..storage.queue import WorkQueue
    path = db_path or str(default_db_path())
    db = Database(path)
    db.executescript(_WORK_QUEUE_DDL)
    return WorkQueue(db)


# ── registry ─────────────────────────────────────────────────────────────

@dataclass
class ServiceDescriptor:
    """How to build (and health-check) one named service."""

    name: str
    factory: Callable[[], Any]
    health_check: HealthCheck | Callable[[], Any] | None = None
    description: str = ""

    def normalized_health_check(self) -> HealthCheck | None:
        if self.health_check is None:
            return None
        return as_health_check(f"service:{self.name}", self.health_check)


def _check_browser() -> tuple[bool, str]:
    try:
        from ..tools.browser import BrowserSession  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"browser import failed: {exc!r}"
    return True, "BrowserSession importable"


def _check_model() -> tuple[bool, str]:
    try:
        from ..llm.router import LLMRouter  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"llm router import failed: {exc!r}"
    return True, "LLMRouter importable"


def _check_bridge() -> tuple[bool, str]:
    try:
        from ..social.chat.whatsapp import WhatsAppAdapter  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"bridge import failed: {exc!r}"
    return True, "WhatsAppAdapter importable"


def _check_queue() -> tuple[bool, str]:
    try:
        from ..storage.queue import WorkQueue  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"queue import failed: {exc!r}"
    return True, "WorkQueue importable"


class ServiceRegistry:
    """Thread-safe registry of lazily-constructed named services."""

    def __init__(self, *, db_path: str | None = None) -> None:
        self._descriptors: dict[str, ServiceDescriptor] = {}
        self._instances: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._db_path = db_path
        self._register_defaults()

    # ── built-ins ────────────────────────────────────────────────────────
    def _register_defaults(self) -> None:
        db_path = self._db_path
        self.register("browser", _factory_browser,
                      health_check=_check_browser,
                      description="Cookie-kept web session (tools/browser.BrowserSession)")
        self.register("model", _factory_model,
                      health_check=_check_model,
                      description="LLM provider chain router (llm/router.LLMRouter)")
        self.register("bridge", _factory_bridge,
                      health_check=_check_bridge,
                      description="Messaging bridge client (social/chat/whatsapp.WhatsAppAdapter)")
        self.register("queue", lambda: _factory_queue(db_path),
                      health_check=_check_queue,
                      description="Durable SQLite work queue (storage/queue.WorkQueue)")

    # ── management ───────────────────────────────────────────────────────
    def register(self, name: str, factory: Callable[[], Any], *,
                 health_check: HealthCheck | Callable[[], Any] | None = None,
                 description: str = "",
                 replace: bool = False) -> ServiceDescriptor:
        """Register (or, with ``replace=True``, replace) a service descriptor.

        Registering a name that already has a *constructed* instance does
        not disturb that instance unless ``replace=True``, which also drops
        the cached instance.
        """
        name = str(name)
        with self._lock:
            if name in self._descriptors and not replace:
                raise ValueError(f"service {name!r} already registered")
            descriptor = ServiceDescriptor(name=name, factory=factory,
                                           health_check=health_check,
                                           description=description)
            self._descriptors[name] = descriptor
            if replace:
                self._instances.pop(name, None)
            return descriptor

    def lookup(self, name: str) -> Any:
        """Return the service instance, building it once via its factory."""
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                raise ServiceNotFound(f"unknown service {name!r}")
            factory = descriptor.factory
        # Build outside the lock: factories may be slow and may re-enter.
        instance = factory()
        with self._lock:
            # Double-checked: a concurrent lookup may have built it first.
            return self._instances.setdefault(name, instance)

    def list_services(self) -> list[str]:
        with self._lock:
            return sorted(self._descriptors)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._descriptors

    def is_constructed(self, name: str) -> bool:
        """True when the factory has already run for ``name``."""
        with self._lock:
            return name in self._instances

    def unregister(self, name: str) -> bool:
        """Remove a descriptor and drop its cached instance. Returns True
        when something was removed."""
        with self._lock:
            removed = self._descriptors.pop(name, None) is not None
            self._instances.pop(name, None)
            return removed

    def health_check_for(self, name: str) -> HealthCheck | None:
        """The normalized health check for a registered service, if any."""
        with self._lock:
            descriptor = self._descriptors.get(name)
        return descriptor.normalized_health_check() if descriptor else None

    def describe(self, name: str) -> ServiceDescriptor:
        with self._lock:
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                raise ServiceNotFound(f"unknown service {name!r}")
            return descriptor
