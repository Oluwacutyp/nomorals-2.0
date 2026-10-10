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

Descriptors carry lifecycle metadata borrowed from production supervisors:

* ``scope`` — ``"singleton"`` (build once, cache) or ``"factory"``
  (fresh instance per lookup).
* ``depends_on`` — names of services that must start first; ``start_all``
  / ``stop_all`` order topologically, stop reverses start.
* ``on_start`` / ``on_stop`` — lifecycle hooks run around the cached
  instance (e.g. connect / disconnect).
* ``tags`` — free-form labels for discovery (``find_by_tag``).
* ``override()`` — a context manager for tests that swaps an instance
  without touching the descriptor.
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .health import HealthCheck, as_health_check

__all__ = [
    "ServiceNotFound",
    "ServiceDescriptor",
    "ServiceRegistry",
    "default_db_path",
    "SINGLETON",
    "FACTORY",
]

#: Descriptor scopes.
SINGLETON = "singleton"
FACTORY = "factory"


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


def _factory_browser_service() -> Any:
    from ..browser import BrowserService
    return BrowserService()


def _check_browser_service() -> tuple[bool, str]:
    try:
        from ..browser import BrowserService  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"browser service import failed: {exc!r}"
    return True, "BrowserService importable"


# ── registry ─────────────────────────────────────────────────────────────

@dataclass
class ServiceDescriptor:
    """How to build (and health-check) one named service."""

    name: str
    factory: Callable[[], Any]
    health_check: HealthCheck | Callable[[], Any] | None = None
    description: str = ""
    #: "singleton" (build once, cache) or "factory" (fresh per lookup).
    scope: str = SINGLETON
    #: Services that must be started before this one.
    depends_on: tuple[str, ...] = ()
    #: Lifecycle hooks: on_start(instance) after build, on_stop(instance)
    #: before the cached instance is dropped.
    on_start: Callable[[Any], Any] | None = None
    on_stop: Callable[[Any], Any] | None = None
    #: Free-form labels for discovery (see find_by_tag).
    tags: tuple[str, ...] = ()
    #: When True, start_all() starts this service automatically.
    autostart: bool = False

    def normalized_health_check(self) -> HealthCheck | None:
        if self.health_check is None:
            return None
        return as_health_check(f"service:{self.name}", self.health_check)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "depends_on": list(self.depends_on),
            "tags": list(self.tags),
            "has_health_check": self.health_check is not None,
            "has_on_start": self.on_start is not None,
            "has_on_stop": self.on_stop is not None,
            "autostart": self.autostart,
        }


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
        self._started: set[str] = set()
        self._overrides: dict[str, Any] = {}
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
        self.register("browser_service", _factory_browser_service,
                      health_check=_check_browser_service,
                      description="Wave K browser service: sessions/tabs/downloads/"
                                  "screenshots as artifacts (browser.BrowserService)")

    # ── management ───────────────────────────────────────────────────────
    def register(self, name: str, factory: Callable[[], Any], *,
                 health_check: HealthCheck | Callable[[], Any] | None = None,
                 description: str = "",
                 replace: bool = False,
                 scope: str = SINGLETON,
                 depends_on: tuple[str, ...] | list[str] = (),
                 on_start: Callable[[Any], Any] | None = None,
                 on_stop: Callable[[Any], Any] | None = None,
                 tags: tuple[str, ...] | list[str] = (),
                 autostart: bool = False) -> ServiceDescriptor:
        """Register (or, with ``replace=True``, replace) a service descriptor.

        Registering a name that already has a *constructed* instance does
        not disturb that instance unless ``replace=True``, which also drops
        the cached instance.
        """
        name = str(name)
        if scope not in (SINGLETON, FACTORY):
            raise ValueError(f"unknown scope {scope!r} for service {name!r}")
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid service name {name!r}")
        if not callable(factory):
            raise ValueError(f"factory for service {name!r} is not callable")
        with self._lock:
            if name in self._descriptors and not replace:
                raise ValueError(f"service {name!r} already registered")
            descriptor = ServiceDescriptor(
                name=name, factory=factory, health_check=health_check,
                description=description, scope=scope,
                depends_on=tuple(depends_on), on_start=on_start,
                on_stop=on_stop, tags=tuple(tags), autostart=autostart)
            self._descriptors[name] = descriptor
            if replace:
                self._instances.pop(name, None)
                self._started.discard(name)
            return descriptor

    def lookup(self, name: str) -> Any:
        """Return the service instance.

        Singleton-scoped services are built once via their factory and
        cached; factory-scoped services get a fresh instance per lookup.
        An active :meth:`override` wins over both.
        """
        with self._lock:
            if name in self._overrides:
                return self._overrides[name]
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                raise ServiceNotFound(f"unknown service {name!r}")
            if descriptor.scope == SINGLETON and name in self._instances:
                return self._instances[name]
            factory = descriptor.factory
        # Build outside the lock: factories may be slow and may re-enter.
        instance = factory()
        with self._lock:
            if descriptor.scope == SINGLETON:
                # Double-checked: a concurrent lookup may have built it first.
                return self._instances.setdefault(name, instance)
            return instance

    @contextmanager
    def override(self, name: str, instance: Any) -> Iterator[None]:
        """Temporarily swap the instance returned for ``name`` (tests).

        Restores the previous instance (or lack of one) on exit, even on
        exception.  Nested overrides of the same name stack correctly.
        """
        with self._lock:
            if name not in self._descriptors:
                raise ServiceNotFound(f"unknown service {name!r}")
            sentinel = object()
            previous = self._overrides.get(name, sentinel)
            self._overrides[name] = instance
        try:
            yield
        finally:
            with self._lock:
                if previous is sentinel:
                    self._overrides.pop(name, None)
                else:
                    self._overrides[name] = previous

    # ── lifecycle ────────────────────────────────────────────────────────
    def start_service(self, name: str) -> Any:
        """Build (if needed) and run the on_start hook for ``name``.

        Dependencies (``depends_on``) are started first.  Idempotent.
        """
        with self._lock:
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                raise ServiceNotFound(f"unknown service {name!r}")
            if name in self._started:
                return self._instances.get(name)
        for dep in descriptor.depends_on:
            self.start_service(dep)
        instance = self.lookup(name)
        if descriptor.on_start is not None:
            descriptor.on_start(instance)
        with self._lock:
            self._started.add(name)
        return instance

    def stop_service(self, name: str) -> bool:
        """Run the on_stop hook and drop the cached instance.

        Returns False when the service was not started.  Dependents are
        stopped first (reverse of start order).
        """
        with self._lock:
            dependents = [n for n, d in self._descriptors.items()
                          if name in d.depends_on and n in self._started]
        for dependent in dependents:
            self.stop_service(dependent)
        with self._lock:
            if name not in self._started:
                return False
            descriptor = self._descriptors.get(name)
            instance = self._instances.pop(name, None)
            self._started.discard(name)
        if descriptor is not None and descriptor.on_stop is not None \
                and instance is not None:
            try:
                descriptor.on_stop(instance)
            except Exception:  # noqa: BLE001 — stop must not fail the registry
                pass
        return True

    def start_all(self, *, only_autostart: bool = True) -> list[str]:
        """Start services in dependency order. Returns start order.

        By default only ``autostart`` services are started — services stay
        lazily constructed unless the operator opts in.
        """
        with self._lock:
            names = [n for n in self._topo_order()
                     if not only_autostart or self._descriptors[n].autostart]
        for name in names:
            try:
                self.start_service(name)
            except Exception:  # noqa: BLE001 — one bad service ≠ dead registry
                pass
        return names

    def stop_all(self) -> list[str]:
        """Stop every started service in reverse dependency order."""
        order = self._topo_order()
        stopped: list[str] = []
        for name in reversed(order):
            if self.stop_service(name):
                stopped.append(name)
        return stopped

    def restart(self, name: str) -> Any:
        """Recycle a service: stop, drop the cached instance, start again."""
        self.stop_service(name)
        return self.start_service(name)

    def _topo_order(self) -> list[str]:
        """Dependency-first ordering of all registered services."""
        with self._lock:
            names = sorted(self._descriptors)
            deps = {n: [d for d in self._descriptors[n].depends_on
                        if d in self._descriptors] for n in names}
        order: list[str] = []
        visited: dict[str, int] = {}  # 0=visiting, 1=done

        def visit(node: str) -> None:
            state = visited.get(node)
            if state == 1:
                return
            if state == 0:
                raise ValueError(f"circular service dependency at {node!r}")
            visited[node] = 0
            for dep in deps.get(node, ()):  # unknown deps start lazily
                if dep in deps:
                    visit(dep)
            visited[node] = 1
            order.append(node)

        for name in names:
            visit(name)
        return order

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

    def is_started(self, name: str) -> bool:
        """True when ``start_service`` has run for ``name``."""
        with self._lock:
            return name in self._started

    def started_services(self) -> list[str]:
        with self._lock:
            return sorted(self._started)

    def find_by_tag(self, tag: str) -> list[str]:
        """Names of all services carrying ``tag``."""
        with self._lock:
            return sorted(n for n, d in self._descriptors.items()
                          if tag in d.tags)

    def describe_all(self) -> list[dict[str, Any]]:
        """Rich descriptor listing for dashboards and CLIs."""
        with self._lock:
            items = [d.to_dict() for d in
                     (self._descriptors[n] for n in sorted(self._descriptors))]
            for item in items:
                item["constructed"] = item["name"] in self._instances
                item["started"] = item["name"] in self._started
                item["overridden"] = item["name"] in self._overrides
        return items

    def render(self) -> str:
        """Plain-text service registry dashboard."""
        lines = ["services"]
        for item in self.describe_all():
            if item["overridden"]:
                mark = "◈"
            elif item["started"]:
                mark = "▶"
            elif item["constructed"]:
                mark = "✓"
            else:
                mark = "·"
            tags = f" [{','.join(item['tags'])}]" if item["tags"] else ""
            deps = (f" → needs {','.join(item['depends_on'])}"
                    if item["depends_on"] else "")
            lines.append(f"  {mark} {item['name']} ({item['scope']})"
                         f"{tags}{deps}")
            if item["description"]:
                lines.append(f"      {item['description']}")
        return "\n".join(lines)

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
