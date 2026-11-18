"""Rotation pool over the owner's proxy list.

``ProxyManager`` answers "which proxy is active"; this answers "which proxy
next". Candidates come from the same inventory (``ProxyManager.known()``),
are probed concurrently, scored by EWMA latency plus a failure penalty,
quarantined after repeated failure, and handed out stickily so a session
holds one egress.

Never raises on a dead proxy — a dead proxy is a data point.

This pool is NOT the proxy lab's rotation (``tools.proxylab``): that one
cycles the scraped free-proxy pool plus SSH tunnels and owns the
``core.http`` per-request resolver while enabled. This one scores the
owner's own proxies only, and it never touches the resolver slot or the
active proxy — ``ProxyManager`` stays the "which is active" authority.
The pick answers the question; the owner (or the brain) applies it with
``/proxy set``.

Threading: health rows are mutated only under ``self._lock``. Probes run
on the pool executor, but each probe fetches outside the lock and applies
its result under it — ``acquire``/``report``/``snapshot`` never race a
probe write. ``_health`` keys are fixed at construction and
``_sticky_url`` is only ever assigned from one of those keys (or None),
so the sticky lookup cannot KeyError; it still goes through ``.get()``
so a future edit to the key set fails soft instead of raising.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .proxy import ProxyManager, _mask_proxy_url, parse_proxy_url

_log = get_logger(__name__)
__all__ = ["RotationPool", "ProxyHealth", "register"]

_FAIL_PENALTY = 2_500.0     # ms-equivalent added per consecutive failure
_QUARANTINE_BASE = 60.0     # seconds, doubled per further failure
_QUARANTINE_CAP = 1_800.0


@dataclass
class ProxyHealth:
    url: str
    ok: bool = True
    fail_streak: int = 0
    latency_ms: float = 0.0
    uses: int = 0
    quarantined_until: float = 0.0

    @property
    def quarantined(self) -> bool:
        return self.quarantined_until > time.monotonic()

    @property
    def score(self) -> float:
        return self.latency_ms + self.fail_streak * _FAIL_PENALTY

    def as_dict(self) -> dict[str, Any]:
        return {
            "proxy": _mask_proxy_url(self.url), "ok": self.ok,
            "fail_streak": self.fail_streak,
            "latency_ms": round(self.latency_ms), "uses": self.uses,
            "quarantined_for": round(max(0.0, self.quarantined_until
                                         - time.monotonic())),
        }


class RotationPool:
    """Thread-safe scored pool with sticky hand-out."""

    def __init__(self, urls: list[str], *, probe_url: str,
                 timeout: float = 8.0, probe_workers: int = 12,
                 sticky: int = 10, max_fail_streak: int = 3,
                 rng: random.Random | None = None) -> None:
        self.probe_url = probe_url
        self.timeout = timeout
        self.probe_workers = max(1, min(probe_workers, 32))
        self.sticky = max(1, sticky)
        self.max_fail_streak = max(1, max_fail_streak)
        # Seeded RNG for tests; the module-level random otherwise.
        # ``random`` itself has .choice, so both shapes work here.
        self._rng = rng if rng is not None else random
        self._lock = threading.Lock()
        self._health: dict[str, ProxyHealth] = {}
        for raw in urls:
            try:
                parsed = parse_proxy_url(raw)
            except ToolError:
                continue
            self._health[parsed] = ProxyHealth(url=parsed)
        self._sticky_url: str | None = None
        self._sticky_left = 0

    def __len__(self) -> int:
        return len(self._health)

    # ── probing ────────────────────────────────────────────────────────────
    def _probe_one(self, url: str) -> ProxyHealth:
        # Fetch outside the lock; apply the result under it. The
        # pasted draft mutated the shared row on the probe thread,
        # racing acquire/report — the mutation is the only part that
        # needs the lock, never the network wait.
        client = HttpClient(timeout=self.timeout, proxy_url=url)
        started = time.monotonic()
        error = ""
        try:
            resp = client.get(self.probe_url, timeout=self.timeout)
            latency = (time.monotonic() - started) * 1000
            ok = bool(resp.ok)
        except Exception as exc:  # noqa: BLE001 - a dead proxy is a result
            latency = (time.monotonic() - started) * 1000
            ok = False
            error = str(exc)[:120]
        with self._lock:
            h = self._health[url]
            h.latency_ms = latency
            h.ok = ok
            h.fail_streak = 0 if ok else h.fail_streak + 1
        if error:
            _log.debug("proxy %s probe failed: %s",
                       _mask_proxy_url(url), error)
        return h

    def probe_all(self) -> list[ProxyHealth]:
        """Probe every proxy concurrently. Returns health, worst-last."""
        if not self._health:
            return []
        with ThreadPoolExecutor(max_workers=self.probe_workers) as pool:
            list(pool.map(self._probe_one, list(self._health)))
        with self._lock:
            return sorted(self._health.values(),
                          key=lambda h: (not h.ok, h.score))

    # ── hand-out ───────────────────────────────────────────────────────────
    def acquire(self, exclude: str | None = None) -> str | None:
        """Best healthy proxy, sticky for `sticky` uses. None = go direct."""
        with self._lock:
            sticky = (self._health.get(self._sticky_url)
                      if self._sticky_url else None)
            if (sticky is not None and self._sticky_left > 0
                    and not sticky.quarantined
                    and self._sticky_url != exclude):
                self._sticky_left -= 1
                sticky.uses += 1
                return self._sticky_url
            live = [h for h in self._health.values()
                    if not h.quarantined and h.url != exclude]
            if not live:
                self._sticky_url, self._sticky_left = None, 0
                return None
            # score-weighted: best wins most of the time, not always
            live.sort(key=lambda h: h.score)
            head = live[:max(1, min(3, len(live)))]
            pick = self._rng.choice(head) if len(head) > 1 else head[0]
            pick.uses += 1
            self._sticky_url, self._sticky_left = pick.url, self.sticky - 1
            return pick.url

    def report(self, url: str, *, ok: bool, latency_ms: float = 0.0) -> None:
        with self._lock:
            h = self._health.get(url)
            if h is None:
                return
            if latency_ms:
                h.latency_ms = (h.latency_ms * 0.7) + (latency_ms * 0.3)
            if ok:
                h.ok = True
                h.fail_streak = 0
                return
            h.ok = False
            h.fail_streak += 1
            if h.fail_streak >= self.max_fail_streak:
                back = min(_QUARANTINE_CAP,
                           _QUARANTINE_BASE * (2 ** (h.fail_streak
                                                     - self.max_fail_streak)))
                h.quarantined_until = time.monotonic() + back
                if self._sticky_url == url:
                    self._sticky_url, self._sticky_left = None, 0
                _log.info("quarantined %s for %ss", _mask_proxy_url(url), back)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = sorted(self._health.values(), key=lambda h: h.score)
            return {"size": len(rows),
                    "healthy": sum(1 for h in rows if h.ok and not h.quarantined),
                    "quarantined": sum(1 for h in rows if h.quarantined),
                    "rows": [h.as_dict() for h in rows]}


def pool_from_context(context: Any, **kw: Any) -> RotationPool:
    manager = ProxyManager(context)
    return RotationPool(manager.known(),
                        probe_url=manager.probe_url,
                        timeout=manager.probe_timeout, **kw)


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "proxy_rotate_probe",
        description="Probe the whole known proxy list concurrently and score "
                    "it (latency, failures, quarantine).",
        capability=Capability.NET_OUT,
    )
    def proxy_rotate_probe() -> dict[str, Any]:
        pool = pool_from_context(context)
        if not len(pool):
            return {"note": "no proxies known — add to NM_PROXY_KNOWN or /proxy set"}
        pool.probe_all()
        return pool.snapshot()

    @registry.register(
        "proxy_rotate_pick",
        description="Next proxy from the scored pool (sticky). Null means go direct.",
        # NET_OUT, not DB_READ: the pick probes the whole list live
        # before choosing, so it touches the network like the probe
        # tool does. The pasted draft labelled it DB_READ; a read
        # capability must not silently grant network I/O.
        capability=Capability.NET_OUT,
    )
    def proxy_rotate_pick() -> dict[str, Any]:
        pool = pool_from_context(context)
        pool.probe_all()
        chosen = pool.acquire()
        return {"proxy": _mask_proxy_url(chosen) if chosen else None,
                "pool": pool.snapshot()["healthy"]}
