"""Scope file, global budget, per-host governor.

One Governor instance is shared by every offensive module. It answers
three questions: is this URL in scope, may I spend a request now, and how
long until I may.

Scope is a file, not a flag — an autonomous run has no operator to
re-decide mid-flight. Unlisted means out of scope; `*` means everything
minus the exclude list.

Budget is three ceilings: total requests, per-host rate, wall clock. All
three are enforced here, so no module can overspend by forgetting.
"""
from __future__ import annotations

import fnmatch
import ipaddress
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.errors import ToolError
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["Scope", "Governor", "BudgetExhausted", "OutOfScope", "load_scope", "register"]


class OutOfScope(ToolError):
    pass


class BudgetExhausted(ToolError):
    pass


@dataclass
class Scope:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    allow_all: bool = False

    def allows(self, url: str) -> bool:
        host = (urllib.parse.urlsplit(url if "://" in url else "//" + url)
                .hostname or "").lower()
        if not host:
            return False
        for pattern in self.exclude:
            if self._match(host, pattern):
                return False
        if self.allow_all:
            return True
        return any(self._match(host, pattern) for pattern in self.include)

    @staticmethod
    def _match(host: str, pattern: str) -> bool:
        pattern = pattern.strip().lower()
        if not pattern:
            return False
        try:
            if "/" in pattern and not pattern.startswith("*"):
                return ipaddress.ip_address(host) in ipaddress.ip_network(
                    pattern, strict=False)
        except ValueError:
            pass
        if pattern.startswith("*."):
            return host == pattern[2:] or host.endswith(pattern[1:])
        if fnmatch.fnmatch(host, pattern):
            return True
        return host == pattern or host.endswith("." + pattern)

    def as_dict(self) -> dict[str, Any]:
        return {"include": self.include, "exclude": self.exclude,
                "allow_all": self.allow_all}


def load_scope(path: str | Path) -> Scope:
    """Parse a scope file.

        include: example.com, *.example.com, 10.0.0.0/24
        exclude: cdn.example.com
        allow_all: true
    """
    file = Path(path)
    if not file.is_file():
        raise ToolError(f"scope file not found: {path}")
    scope = Scope()
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("allow_all:"):
            scope.allow_all = line.split(":", 1)[1].strip().lower() in {
                "1", "true", "yes"}
            continue
        key, _, rest = line.partition(":")
        bucket = (scope.include if key.strip().lower() == "include"
                  else scope.exclude if key.strip().lower() == "exclude"
                  else None)
        if bucket is None:
            continue
        bucket.extend(p.strip() for p in rest.replace(" ", ",").split(",")
                      if p.strip())
    if not scope.include and not scope.allow_all:
        raise ToolError("scope file lists nothing to include")
    return scope


class Governor:
    """Shared request authority. Every module calls `spend()` first."""

    def __init__(self, scope: Scope, *, max_total: int = 50_000,
                 max_per_host: float = 5.0, max_seconds: float = 3_600.0,
                 jitter: float = 0.15) -> None:
        self.scope = scope
        self.max_total = max_total
        self.max_seconds = max_seconds
        self._jitter = jitter
        self._lock = threading.Lock()
        self._spent = 0
        self._started = time.monotonic()
        self._hosts: dict[str, float] = {}
        self._interval = 1.0 / max(0.1, max_per_host)
        self._stopped = threading.Event()

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self._spent)

    def _check_alive(self) -> None:
        if self._stopped.is_set():
            raise BudgetExhausted("governor stopped")
        if self._spent >= self.max_total:
            raise BudgetExhausted(f"total budget {self.max_total} spent")
        if time.monotonic() - self._started > self.max_seconds:
            raise BudgetExhausted(f"wall clock {self.max_seconds:.0f}s spent")

    def spend(self, url: str, *, cost: int = 1) -> None:
        """Claim a request slot or raise. Blocks for the host interval."""
        if not self.scope.allows(url):
            raise OutOfScope(f"{url} is not in scope")
        with self._lock:
            self._check_alive()
            self._spent += cost
            host = (urllib.parse.urlsplit(url if "://" in url else "//" + url)
                    .hostname or "")
            now = time.monotonic()
            wait = self._hosts.get(host, 0.0) - now
            self._hosts[host] = max(now, self._hosts.get(host, 0.0)) + self._interval
        if wait > 0:
            time.sleep(wait + (wait * self._jitter))

    def stop(self) -> None:
        self._stopped.set()

    def snapshot(self) -> dict[str, Any]:
        return {"spent": self._spent, "remaining": self.remaining,
                "elapsed": round(time.monotonic() - self._started, 1),
                "hosts_touched": len(self._hosts),
                "scope": self.scope.as_dict()}


def register(registry: Any) -> None:
    context = registry.context
    home = Path(getattr(context, "home", ".") or ".")

    @registry.register(
        "scope_load",
        description="Parse and validate a scope file (include/exclude/allow_all).",
        capability=Capability.DB_READ,
        parameters={"path": "str — scope file"},
    )
    def scope_load(path: str) -> dict[str, Any]:
        return load_scope(path).as_dict()

    @registry.register(
        "scope_check",
        description="Is this URL in scope?",
        capability=Capability.DB_READ,
        parameters={"path": "str — scope file", "url": "str"},
    )
    def scope_check(path: str, url: str) -> dict[str, Any]:
        scope = load_scope(path)
        return {"url": url, "in_scope": scope.allows(url)}
