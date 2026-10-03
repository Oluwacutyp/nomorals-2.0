"""Proxy / tunnel manager: route the bot's own outbound traffic.

The owner supplies their own proxies (residential, VPS, a WireGuard endpoint
fronted by an HTTP proxy — their infrastructure, their rules). This module
manages them:

* a **known list** (settings + environment) of candidate proxies
* **liveness tests** — each candidate answers a probe URL; the reply is the
  proxy's egress IP, so you see exactly what it does
* an **active proxy** that routes ALL ``HttpClient`` traffic process-wide
  (via ``core.http.set_default_proxy``) and persists across restarts

Schemes: ``http://``, ``https://``, ``socks5://`` (SOCKS needs PySocks
installed; without it the manager says so instead of failing mysteriously).
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.errors import ToolError
from ..core.http import (
    HttpClient,
    apply_socks_proxy,
    get_default_proxy,
    reset_socks_proxy,
    set_default_proxy,
)
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["ProxyManager", "parse_proxy_url", "register"]

_KV_KEY = "proxy.active"
_SCHEMES = ("http://", "https://", "socks5://", "socks5h://")


def parse_proxy_url(raw: str) -> str:
    """Normalize a proxy URL; raises ToolError for anything not routable."""
    url = (raw or "").strip()
    if not url:
        raise ToolError("empty proxy URL")
    if url.lower().startswith(_SCHEMES):
        return url
    # bare host:port — assume http
    return f"http://{url}"


def _socks_available() -> bool:
    try:
        import socks  # noqa: F401 - PySocks

        return True
    except ImportError:
        return False


class ProxyManager:
    """Owns the proxy list, tests candidates, applies the active one."""

    def __init__(self, context: Any) -> None:
        self.context = context
        settings = getattr(context, "settings", None)
        proxy = getattr(settings, "proxy", None) if settings else None
        self.probe_url = (getattr(proxy, "probe_url", "") or "https://api.ipify.org") if proxy else "https://api.ipify.org"
        self.probe_timeout = float(getattr(proxy, "probe_timeout", 10.0)) if proxy else 10.0
        self._known = getattr(proxy, "known", "") if proxy else ""
        self._active = getattr(proxy, "active", "") if proxy else ""
        db = getattr(context, "db", None)
        if db is not None:
            try:
                row = db.query_one("SELECT value FROM kv_store WHERE key = ?", (_KV_KEY,))
                if row:
                    self._active = str(json.loads(row["value"]).get("proxy") or "")
            except Exception:  # noqa: BLE001
                pass

    # ── inventory ────────────────────────────────────────────────────────────
    def known(self) -> list[str]:
        """All candidate proxies: settings list + active + environment."""
        seen: dict[str, None] = {}
        for url in [u.strip() for u in self._known.split(",") if u.strip()]:
            seen.setdefault(url, None)
        env = self._env_proxy()
        if env:
            seen.setdefault(env, None)
        if self._active:
            seen.setdefault(self._active, None)
        return list(seen)

    def _env_proxy(self) -> str:
        """The NM_PROXY_URL candidate: settings first, environment as the
        legacy fallback (settings are where the env var is documented)."""
        settings = getattr(self.context, "settings", None)
        proxy = getattr(settings, "proxy", None) if settings else None
        if proxy is not None and getattr(proxy, "url", ""):
            return str(proxy.url).strip()
        import os

        return (os.environ.get("NM_PROXY_URL") or "").strip()

    def status(self) -> dict[str, Any]:
        active = self._active or get_default_proxy()
        return {
            "active": active or "direct (no proxy)",
            "known": self.known(),
            "socks5_supported": _socks_available(),
            "raw_sockets_routed": active.lower().startswith(("socks5://", "socks5h://"))
            and _socks_available(),
            "probe_url": self.probe_url,
            "routes_all_clients": True,
        }

    # ── testing ─────────────────────────────────────────────────────────────
    def test(self, url: str = "", *, timeout: float | None = None) -> dict[str, Any]:
        """Probe one proxy (or all known). The reply is the egress IP — proof
        the traffic actually went through."""
        targets = [url] if url else self.known()
        if not targets:
            return {"results": [], "note": "no proxies to test — add them to NM_PROXY_KNOWN or /proxy set"}
        results = []
        for candidate in targets:
            results.append(self._test_one(candidate, timeout or self.probe_timeout))
        return {"results": results}

    def _test_one(self, url: str, timeout: float) -> dict[str, Any]:
        candidate = parse_proxy_url(url)
        if candidate.lower().startswith(("socks5", "socks5h")) and not _socks_available():
            return {"proxy": candidate, "ok": False,
                    "error": "SOCKS5 needs PySocks: pip install pysocks"}
        probe = HttpClient(timeout=timeout, proxy_url=candidate)
        started = time.monotonic()
        try:
            response = probe.get(self.probe_url, timeout=timeout)
            latency = round((time.monotonic() - started) * 1000)
            egress = (response.text or "").strip()[:64]
            return {
                "proxy": candidate, "ok": response.ok,
                "status": response.status, "latency_ms": latency,
                "egress_ip": egress,
            }
        except Exception as exc:  # noqa: BLE001 - a dead proxy is a result
            latency = round((time.monotonic() - started) * 1000)
            return {"proxy": candidate, "ok": False, "latency_ms": latency,
                    "error": str(exc)[:200]}

    # ── applying ─────────────────────────────────────────────────────────────
    def set_active(self, url: str, *, verify: bool = False) -> dict[str, Any]:
        """Route all outbound traffic through ``url`` ("" clears).

        Persists to kv_store so the choice survives restarts, and applies
        process-wide immediately via ``set_default_proxy``.
        """
        url = (url or "").strip()
        if url:
            url = parse_proxy_url(url)
            if verify:
                probe = self._test_one(url, self.probe_timeout)
                if not probe.get("ok"):
                    raise ToolError(f"proxy {url} failed its probe: {probe.get('error') or probe.get('status')}")
        self._active = url
        set_default_proxy(url)
        # SOCKS proxies additionally need the global PySocks patch so that
        # urllib and raw-socket code (OSINT, request crafting, …) route too.
        if url and url.lower().startswith(("socks5://", "socks5h://")):
            if apply_socks_proxy(url):
                _log.info("raw-socket traffic now routed through %s", url)
            else:
                _log.warning("SOCKS5 selected but PySocks is missing — only "
                             "HttpClient routes; pip install pysocks")
        else:
            reset_socks_proxy()  # idempotent: direct routing restored for raw sockets
        db = getattr(self.context, "db", None)
        if db is not None:
            try:
                with db.transaction():
                    db.execute(
                        "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (_KV_KEY, json.dumps({"proxy": url}), time.time()),
                    )
            except Exception:  # noqa: BLE001 - apply anyway; persistence is best-effort
                _log.warning("could not persist proxy choice", exc_info=True)
        _log.info("outbound proxy set to %s", url or "direct")
        return {"active": url or "direct", "persisted": db is not None,
                "note": "all new HttpClient instances now route through it" if url else "direct routing restored"}

    def clear(self) -> dict[str, Any]:
        return self.set_active("")


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "proxy_status",
        description="Current outbound routing: active proxy, known list, SOCKS5 support.",
        capability=Capability.DB_READ,
    )
    def proxy_status() -> dict[str, Any]:
        return ProxyManager(context).status()

    @registry.register(
        "proxy_list",
        description="List all known candidate proxies (settings + environment + active).",
        capability=Capability.DB_READ,
    )
    def proxy_list() -> dict[str, Any]:
        manager = ProxyManager(context)
        return {"proxies": manager.known()}

    @registry.register(
        "proxy_test",
        description=(
            "Probe a proxy (or all known): latency + egress IP — proof of where "
            "traffic actually goes. No args tests the whole known list."
        ),
        capability=Capability.NET_OUT,
        parameters={"proxy": "str (optional) — one proxy; omit to test all"},
    )
    def proxy_test(proxy: str = "") -> dict[str, Any]:
        return ProxyManager(context).test(proxy)

    @registry.register(
        "proxy_set",
        description=(
            "Route all outbound traffic through a proxy (http/https/socks5). "
            "Empty proxy clears the routing. Persists across restarts."
        ),
        capability=Capability.SYS_CONFIG,
        parameters={
            "proxy": "str — http://host:port | socks5://host:port | '' to clear",
            "verify": "bool (optional) — probe it before committing",
        },
    )
    def proxy_set(proxy: str = "", *, verify: str = "") -> dict[str, Any]:
        return ProxyManager(context).set_active(
            proxy, verify=(verify or "").lower() in {"1", "true", "yes", "on"}
        )

    @registry.register(
        "proxy_clear",
        description="Stop proxying: restore direct routing.",
        capability=Capability.SYS_CONFIG,
    )
    def proxy_clear() -> dict[str, Any]:
        return ProxyManager(context).clear()
