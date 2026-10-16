"""Per-domain concurrency: polite, bounded parallelism for browsing.

Two caps, both enforced:

* **per-domain**: at most ``max_per_domain`` concurrent acquisitions for
  one host (default 2) — a burst against one site is throttled even when
  the global pool is idle.
* **global**: at most ``global_cap`` concurrent acquisitions overall
  (default 10) — the process never fans out without bound.

Hosts are normalized (lowercased, port stripped) so ``Example.COM:443``
and ``example.com`` share one bucket. Acquire with :meth:`slot` (a
context manager) or :meth:`try_acquire` (non-blocking / timed). A
timeout is a failure the caller sees — never a silent queue.
"""

from __future__ import annotations

import threading
import urllib.parse
from contextlib import contextmanager
from typing import Any, Iterator

from ..core.logging_setup import get_logger

__all__ = ["ConcurrencyError", "DomainConcurrency"]

_log = get_logger(__name__)


class ConcurrencyError(RuntimeError):
    """A slot could not be acquired within the timeout."""


def _normalize_domain(domain_or_url: str) -> str:
    """Host part of a URL or a bare domain, lowercased, port stripped."""
    raw = (domain_or_url or "").strip()
    if not raw:
        raise ConcurrencyError("concurrency slot needs a domain or URL")
    if "://" in raw:
        try:
            host = urllib.parse.urlparse(raw).hostname or ""
        except Exception as exc:
            raise ConcurrencyError(f"bad URL {raw!r}: {exc}") from exc
    else:
        host = raw.split("/")[0].split(":")[0]
    host = host.strip().lower()
    if not host:
        raise ConcurrencyError(f"could not determine a domain from {domain_or_url!r}")
    return host


class DomainConcurrency:
    """Per-domain semaphores + one global cap, all thread-safe."""

    def __init__(self, *, max_per_domain: int = 2, global_cap: int = 10) -> None:
        if int(max_per_domain or 0) < 1:
            raise ConcurrencyError(f"max_per_domain must be >= 1, got {max_per_domain!r}")
        if int(global_cap or 0) < 1:
            raise ConcurrencyError(f"global_cap must be >= 1, got {global_cap!r}")
        self.max_per_domain = int(max_per_domain)
        self.global_cap = int(global_cap)
        self._lock = threading.Lock()
        self._global = threading.BoundedSemaphore(self.global_cap)
        self._domain_semaphores: dict[str, threading.Semaphore] = {}
        self._in_use: dict[str, int] = {}
        self._global_in_use = 0

    # -- internals -------------------------------------------------------------
    def _semaphore_for(self, domain: str) -> threading.Semaphore:
        with self._lock:
            sem = self._domain_semaphores.get(domain)
            if sem is None:
                sem = threading.Semaphore(self.max_per_domain)
                self._domain_semaphores[domain] = sem
                self._in_use[domain] = 0
            return sem

    # -- acquire / release -------------------------------------------------------
    def try_acquire(self, domain_or_url: str, *, timeout: float = 0.0) -> bool:
        """Try to take a slot. Returns True on success, False on timeout
        (never blocks past ``timeout`` seconds). Raises ConcurrencyError
        on a bad domain."""
        domain = _normalize_domain(domain_or_url)
        sem = self._semaphore_for(domain)
        # Domain first, then global: a thread holding a global slot while
        # waiting on a domain slot would starve other domains.
        if not sem.acquire(timeout=max(0.0, float(timeout or 0.0))):
            return False
        if not self._global.acquire(timeout=max(0.0, float(timeout or 0.0))):
            sem.release()
            return False
        with self._lock:
            self._in_use[domain] = self._in_use.get(domain, 0) + 1
            self._global_in_use += 1
        return True

    def acquire(self, domain_or_url: str, *, timeout: float | None = None) -> str:
        """Block until a slot is free (``timeout=None`` = wait forever),
        or raise ConcurrencyError on timeout."""
        domain = _normalize_domain(domain_or_url)
        sem = self._semaphore_for(domain)
        wait = None if timeout is None else max(0.0, float(timeout))
        acquired = sem.acquire() if wait is None else sem.acquire(timeout=wait)
        if not acquired:
            raise ConcurrencyError(
                f"timed out waiting for a {domain} slot "
                f"(max {self.max_per_domain}/domain)")
        try:
            global_ok = (self._global.acquire() if wait is None
                         else self._global.acquire(timeout=wait))
            if not global_ok:
                raise ConcurrencyError(
                    f"timed out waiting for a global slot (cap {self.global_cap})")
        except BaseException:
            sem.release()
            raise
        with self._lock:
            self._in_use[domain] = self._in_use.get(domain, 0) + 1
            self._global_in_use += 1
        return domain

    def release(self, domain_or_url: str) -> None:
        """Release one slot. Releasing a domain with no held slot raises
        — a release/acquire imbalance is a bug, not a no-op."""
        domain = _normalize_domain(domain_or_url)
        with self._lock:
            held = self._in_use.get(domain, 0)
            if held <= 0:
                raise ConcurrencyError(
                    f"release without acquire for domain {domain!r}")
            self._in_use[domain] = held - 1
            self._global_in_use = max(0, self._global_in_use - 1)
        sem = self._semaphore_for(domain)
        sem.release()
        self._global.release()

    @contextmanager
    def slot(self, domain_or_url: str, *, timeout: float | None = None) -> Iterator[str]:
        """``with limiter.slot(url): ...`` — acquire, run, always release."""
        domain = self.acquire(domain_or_url, timeout=timeout)
        try:
            yield domain
        finally:
            self.release(domain)

    # -- observability --------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Current usage: global in-use/cap plus per-domain in-use/max.
        Domains with zero in-use are omitted (not fabricated)."""
        with self._lock:
            return {
                "global": {"in_use": self._global_in_use, "cap": self.global_cap},
                "per_domain": {d: {"in_use": n, "max": self.max_per_domain}
                               for d, n in sorted(self._in_use.items()) if n > 0},
                "max_per_domain": self.max_per_domain,
            }
