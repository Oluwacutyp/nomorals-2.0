"""Proxy pool connector — Devon's own proxy rotation pool.

A LOCAL management interface (AuthMethod.NONE): Devon keeps its own pool
of HTTP(S) proxies for rotation across research, fetching, and other
outbound work.

Storage: one vault credential per proxy under ``"connector:proxypool"``.
The vault username is the proxy's endpoint id (``protocol://host:port``);
the encrypted vault secret is a small JSON blob holding the proxy's
auth username/password (so password-less proxies still store a
decryptable secret); the endpoint, tags, and health history ride in
metadata. Proxy passwords never appear in plaintext config, logs, or
list output.

Health is never guessed: :meth:`health_check` sends a real HTTP request
through each proxy (stdlib urllib + ProxyHandler, auth via the proxy URL's
userinfo) and records latency, outcome, and timestamp. :meth:`rotate`
round-robins over proxies whose last recorded check was healthy.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..accounts.vault import Credential
from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["ProxyPoolConnector", "ProxyPoolError"]

_log = get_logger(__name__)

#: Health states recorded per proxy.
HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"


class ProxyPoolError(ConnectorError):
    """A proxy pool operation failed."""


@register_connector
class ProxyPoolConnector(Connector):
    """Devon's proxy pool: store, health-check, and rotate HTTP(S) proxies."""

    id = "proxypool"
    name = "Proxy Pool"
    description = (
        "Manage Devon's own pool of HTTP(S) proxies: store endpoints with "
        "vault-held credentials, health-check them with real requests, and "
        "rotate across the healthy ones. Local service — no external auth."
    )
    auth_methods = (AuthMethod.NONE,)
    PROVISIONABLE = ("proxy",)

    #: Proxy protocols this pool can store and actively health-check.
    PROTOCOLS = ("http", "https")

    #: Default target for health checks: small, stable, plain HTTP.
    DEFAULT_CHECK_URL = "http://example.com/"

    #: Password placeholder used in every non-secret view.
    MASK = "***"

    def __init__(self, vault: Any, http: Any = None) -> None:
        super().__init__(vault, http=http)
        self._rr = 0  # round-robin cursor (in-memory; resets on disconnect)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self, *, check: bool = True) -> ConnectResult:
        """Validate the pool is usable.

        There is no external auth to perform — "connected" means the pool
        holds proxies. With ``check=True`` (default) every proxy is
        live-probed first, so a successful connect means at least one
        proxy actually works right now. An empty pool is an honest
        failure, never a fake success.
        """
        proxies = self._load_all()
        if not proxies:
            return ConnectResult(
                ok=False,
                account="local proxy pool",
                message=(
                    "proxy pool is empty — nothing to connect to. Add your "
                    "first proxy with "
                    "add_proxy(host, port, username=..., password=...) "
                    "or provision('proxy', host=..., port=...)."
                ),
            )
        if check:
            summary = self.health_check()
            healthy = summary["healthy"]
            if not healthy:
                return ConnectResult(
                    ok=False,
                    account="local proxy pool",
                    message=(
                        f"pool holds {len(proxies)} proxies but none passed "
                        "a live health check — check endpoints and "
                        "credentials, then run health_check() again."
                    ),
                )
            return ConnectResult(
                ok=True,
                account="local proxy pool",
                message=(
                    f"proxy pool ready: {healthy}/{len(proxies)} proxies "
                    "healthy (live-checked)."
                ),
            )
        return ConnectResult(
            ok=True,
            account="local proxy pool",
            message=(
                f"proxy pool holds {len(proxies)} proxies "
                "(not live-checked; pass check=True to probe them)."
            ),
        )

    def disconnect(self) -> None:
        """Reset the rotation cursor. Idempotent.

        Stored proxies are NOT removed — they live in the encrypted
        vault until explicitly deleted with :meth:`remove_proxy`.
        """
        self._rr = 0
        _log.info("proxypool disconnected (rotation cursor reset)")

    def status(self) -> ConnectorStatus:
        proxies = self._load_all()
        if not proxies:
            return ConnectorStatus(
                connected=False,
                account="local proxy pool",
                detail=(
                    "proxy pool is empty — add one with "
                    "add_proxy(host, port)"
                ),
            )
        healthy = unhealthy = unknown = 0
        last_checked = 0.0
        for cred in proxies:
            health = (cred.metadata or {}).get("health") or {}
            state = health.get("status", UNKNOWN)
            if state == HEALTHY:
                healthy += 1
            elif state == UNHEALTHY:
                unhealthy += 1
            else:
                unknown += 1
            last_checked = max(last_checked, float(health.get("last_checked", 0.0) or 0.0))
        parts = [f"{len(proxies)} proxies", f"{healthy} healthy"]
        if unhealthy:
            parts.append(f"{unhealthy} unhealthy")
        if unknown:
            parts.append(f"{unknown} unchecked")
        detail = ", ".join(parts)
        if last_checked:
            detail += f"; last check {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(last_checked))}"
        else:
            detail += "; never checked"
        return ConnectorStatus(
            connected=healthy > 0,
            account="local proxy pool",
            last_checked=last_checked,
            detail=detail,
        )

    def test_connection(self) -> bool:
        """True when at least one proxy passes a live health check now."""
        try:
            return self.health_check()["healthy"] > 0
        except ConnectorError:
            return False

    # ── provisioning ───────────────────────────────────────────

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        """Provision a proxy into the pool (the pool's provisioning action)."""
        if kind == "proxy":
            return self.add_proxy(**kwargs)
        raise ConnectorError(
            f"proxypool cannot provision {kind!r} "
            f"(provisionable: {', '.join(self.PROVISIONABLE)})"
        )

    # ── pool management ────────────────────────────────────────

    def add_proxy(
        self,
        host: str,
        port: int | str,
        username: str | None = None,
        password: str | None = None,
        protocol: str = "http",
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Store a proxy endpoint. Re-adding an endpoint updates it in place.

        The proxy password is encrypted into the vault; list/status views
        never expose it. Returns the stored proxy with the password masked.
        """
        protocol = (protocol or "").strip().lower()
        if protocol not in self.PROTOCOLS:
            raise ProxyPoolError(
                f"unsupported proxy protocol {protocol!r}: "
                f"proxypool supports {', '.join(self.PROTOCOLS)}"
            )
        host = self._normalize_host(host)
        port_num = self._normalize_port(port)
        proxy_user = (username or "").strip() or ""
        if bool(proxy_user) != (password is not None and password != ""):
            raise ProxyPoolError(
                "proxy auth needs both username and password, or neither"
            )
        clean_tags = self._normalize_tags(tags)
        proxy_id = f"{protocol}://{host}:{port_num}"

        # Preserve health history when an endpoint is re-added/updated.
        health = self._blank_health()
        with contextlib.suppress(NotFound):
            existing = self.vault.get(self._service, proxy_id, mark_used=False)
            health = dict((existing.metadata or {}).get("health") or {}) or self._blank_health()

        metadata = {
            "proxy_id": proxy_id,
            "host": host,
            "port": port_num,
            "protocol": protocol,
            "proxy_username": proxy_user,
            "tags": clean_tags,
            "health": health,
        }
        cred = self.vault.store(
            service=self._service,
            username=proxy_id,
            # The vault cannot decrypt an empty secret, so the credential
            # is always a JSON blob — decryptable even with no auth.
            password=json.dumps(
                {"username": proxy_user, "password": password or ""}
            ),
            credential_type="proxy",
            tags=["connector", "proxypool", *clean_tags],
            metadata=metadata,
        )
        _log.info("proxypool: stored proxy %s", proxy_id)
        return self._public_view(cred, has_auth=bool(proxy_user))

    def list_proxies(self) -> list[dict[str, Any]]:
        """Every stored proxy with health status. Passwords are masked."""
        creds = sorted(self._load_all(), key=lambda c: c.username)
        return [self._public_view(c) for c in creds]

    def remove_proxy(self, proxy_id: str) -> None:
        """Delete a proxy from the pool. Idempotent — unknown ids are a no-op."""
        pid = self._normalize_id(proxy_id)
        with contextlib.suppress(NotFound):
            self.vault.delete(self._service, pid)
        _log.info("proxypool: removed proxy %s", pid)

    def get_proxy(self, proxy_id: str) -> dict[str, Any]:
        """Full connection details for one proxy, credential included.

        For use by other components (rotation, HttpClient wiring). The
        returned dict contains the real password and a ``url_with_auth``
        form — treat it as secret.
        """
        cred = self._require_proxy(proxy_id)
        return self._details(cred, include_secret=True)

    # ── health ─────────────────────────────────────────────────

    def health_check(
        self,
        proxy_id: str | None = None,
        *,
        url: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Probe proxies with a real HTTP request sent through each one.

        * ``proxy_id`` given: check one proxy. On failure the unhealthy
          state is recorded and a :class:`ProxyPoolError` is raised —
          failures are never silent marks.
        * ``proxy_id`` omitted: check every proxy concurrently and return
          a summary with per-proxy results. Individual failures are
          recorded on their proxies; the summary reports them.
        """
        url = url or self.DEFAULT_CHECK_URL
        if proxy_id is not None:
            return self._check_one(proxy_id, url=url, timeout=timeout)
        creds = self._load_all()
        if not creds:
            raise ProxyPoolError(
                "proxy pool is empty — add a proxy with "
                "add_proxy(host, port) before health-checking"
            )
        # Decrypt credentials on this thread first: vault reads happen
        # here, workers only run the (slow) network probes, and health
        # writes happen back here. Keeps every DB touch on one thread.
        targets: list[tuple[str, str]] = []
        for cred in creds:
            full = self.vault.get(self._service, cred.username, mark_used=False)
            targets.append((cred.username, self._proxy_url(full, with_auth=True)))
        results: dict[str, dict[str, Any]] = {}
        workers = max(1, min(8, len(targets)))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="proxypool-check"
        ) as pool:
            future_to_id = {
                pool.submit(self._probe_via_proxy, proxy_url, url, timeout): pid
                for pid, proxy_url in targets
            }
            for future in concurrent.futures.as_completed(future_to_id):
                pid = future_to_id[future]
                try:
                    status_code, latency_ms = future.result()
                    ok, error = True, ""
                except ProxyPoolError as exc:
                    status_code, latency_ms, ok, error = None, None, False, str(exc)
                updated = self._record_health(
                    pid,
                    ok=ok,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    error=error,
                )
                results[pid] = self._public_view(updated)
        healthy = sum(1 for r in results.values() if r["health"]["status"] == HEALTHY)
        summary = {
            "total": len(creds),
            "healthy": healthy,
            "unhealthy": sum(
                1 for r in results.values() if r["health"]["status"] == UNHEALTHY
            ),
            "results": results,
        }
        _log.info(
            "proxypool health check: %d/%d healthy", healthy, len(creds)
        )
        return summary

    def _check_one(
        self, proxy_id: str, *, url: str, timeout: float
    ) -> dict[str, Any]:
        result = self._probe_and_record(proxy_id, url, timeout)
        if result["health"]["status"] != HEALTHY:
            raise ProxyPoolError(
                f"proxy {result['id']} failed health check: "
                f"{result['health']['last_error']}"
            )
        return result

    def _probe_and_record(
        self, proxy_id: str, url: str, timeout: float
    ) -> dict[str, Any]:
        """Probe one proxy and persist its health. Returns the masked view."""
        cred = self._require_proxy(proxy_id)
        proxy_url = self._proxy_url(cred, with_auth=True)
        try:
            status_code, latency_ms = self._probe_via_proxy(
                proxy_url, url, timeout
            )
            error = ""
            ok = True
        except ProxyPoolError as exc:
            status_code, latency_ms, error, ok = None, None, str(exc), False
        updated = self._record_health(
            proxy_id,
            ok=ok,
            latency_ms=latency_ms,
            status_code=status_code,
            error=error,
        )
        return self._public_view(updated)

    def _probe_via_proxy(
        self, proxy_url: str, url: str, timeout: float
    ) -> tuple[int, float]:
        """Send one real HTTP request through the proxy.

        Returns (status code, latency_ms). Any completed HTTP response —
        whatever the status — proves the proxy forwarded the request, so
        it counts as healthy. Network/auth failures raise ProxyPoolError
        with a specific reason.
        """
        handler = urllib.request.ProxyHandler(
            {"http": proxy_url, "https": proxy_url}
        )
        opener = urllib.request.build_opener(handler)
        start = time.monotonic()
        try:
            with opener.open(url, timeout=timeout) as resp:
                status = int(resp.status)
                resp.read(65536)  # drain enough to prove the body flows
        except urllib.error.HTTPError as exc:
            # We got an HTTP response through the proxy — it forwarded.
            # 407 means the proxy itself rejected our credentials.
            if exc.code == 407:
                raise ProxyPoolError(
                    "proxy authentication failed (407): bad username/password"
                ) from exc
            return int(exc.code), (time.monotonic() - start) * 1000.0
        except urllib.error.URLError as exc:
            raise ProxyPoolError(
                f"proxy request failed: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProxyPoolError(
                f"proxy request timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise ProxyPoolError(f"proxy connection failed: {exc}") from exc
        return status, (time.monotonic() - start) * 1000.0

    def _record_health(
        self,
        proxy_id: str,
        *,
        ok: bool,
        latency_ms: float | None,
        status_code: int | None,
        error: str,
    ) -> Credential:
        cred = self.vault.get(self._service, proxy_id, mark_used=False)
        meta = dict(cred.metadata or {})
        health = dict(meta.get("health") or {}) or self._blank_health()
        now = time.time()
        health.update(
            {
                "status": HEALTHY if ok else UNHEALTHY,
                "latency_ms": latency_ms,
                "status_code": status_code,
                "last_checked": now,
                "last_error": error if not ok else "",
                "consecutive_failures": 0 if ok else int(health.get("consecutive_failures", 0) or 0) + 1,
                "checks": int(health.get("checks", 0) or 0) + 1,
            }
        )
        meta["health"] = health
        return self.vault.store(
            service=self._service,
            username=proxy_id,
            password=cred.password,
            credential_type=cred.credential_type,
            tags=list(cred.tags or []),
            metadata=meta,
        )

    # ── rotation ───────────────────────────────────────────────

    def rotate(self) -> dict[str, Any]:
        """Next healthy proxy, round-robin. Includes the credential.

        Skips proxies whose last recorded check was not healthy. Raises
        :class:`ProxyPoolError` when no healthy proxy exists — run
        :meth:`health_check` to refresh the pool first.
        """
        healthy = sorted(
            (
                c
                for c in self._load_all()
                if ((c.metadata or {}).get("health") or {}).get("status") == HEALTHY
            ),
            # Insertion order (auto-increment id), not alphabetical: the user
            # expects round-robin to follow the order they added proxies.
            key=lambda c: c.id,
        )
        if not healthy:
            total = len(self._load_all())
            raise ProxyPoolError(
                "no healthy proxies to rotate to"
                + (f" ({total} stored)" if total else " (pool is empty)")
                + " — run health_check() to refresh the pool"
            )
        pick = healthy[self._rr % len(healthy)]
        self._rr += 1
        cred = self.vault.get(self._service, pick.username, mark_used=False)
        _log.info("proxypool: rotated to %s", pick.username)
        return self._details(cred, include_secret=True)

    # ── internals ──────────────────────────────────────────────

    def _load_all(self) -> list[Credential]:
        return self.vault.list_all(service=self._service)

    def _require_proxy(self, proxy_id: str) -> Credential:
        pid = self._normalize_id(proxy_id)
        try:
            return self.vault.get(self._service, pid, mark_used=False)
        except NotFound as exc:
            raise ProxyPoolError(f"unknown proxy {proxy_id!r}") from exc

    @staticmethod
    def _blank_health() -> dict[str, Any]:
        return {
            "status": UNKNOWN,
            "latency_ms": None,
            "status_code": None,
            "last_checked": 0.0,
            "last_error": "",
            "consecutive_failures": 0,
            "checks": 0,
        }

    @staticmethod
    def _normalize_host(host: str) -> str:
        h = (host or "").strip().lower().rstrip(".")
        if not h or any(ch.isspace() or ord(ch) < 32 for ch in h):
            raise ProxyPoolError(
                f"invalid proxy host {host!r}: must be a hostname or IP"
            )
        if ":" in h and not (h.startswith("[") and h.endswith("]")):
            h = f"[{h}]"  # IPv6 literal
        return h

    @staticmethod
    def _normalize_port(port: int | str) -> int:
        try:
            num = int(str(port).strip())
        except (TypeError, ValueError) as exc:
            raise ProxyPoolError(
                f"invalid proxy port {port!r}: must be an integer 1-65535"
            ) from exc
        if not 1 <= num <= 65535:
            raise ProxyPoolError(
                f"invalid proxy port {port!r}: must be an integer 1-65535"
            )
        return num

    @staticmethod
    def _normalize_tags(
        tags: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if tags is None:
            return []
        if not isinstance(tags, (list, tuple)):
            raise ProxyPoolError("tags must be a list of strings")
        clean: list[str] = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ProxyPoolError("tags must be a list of strings")
            tag = tag.strip()
            if tag and tag not in clean:
                clean.append(tag)
        return clean

    @classmethod
    def _normalize_id(cls, proxy_id: str) -> str:
        """Normalize a free-form proxy id to the stored ``protocol://host:port`` form."""
        pid = (proxy_id or "").strip().lower()
        if not pid:
            raise ProxyPoolError("proxy id must not be empty")
        if "://" not in pid:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: expected "
                "'protocol://host:port'"
            )
        protocol, rest = pid.split("://", 1)
        if ":" not in rest:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: expected "
                "'protocol://host:port'"
            )
        host_part, _, port_part = rest.rpartition(":")
        host = cls._normalize_host(host_part)
        port = cls._normalize_port(port_part)
        if protocol not in cls.PROTOCOLS:
            raise ProxyPoolError(
                f"invalid proxy id {proxy_id!r}: unsupported protocol "
                f"{protocol!r}"
            )
        return f"{protocol}://{host}:{port}"

    def _proxy_url(self, cred: Credential, *, with_auth: bool) -> str:
        meta = cred.metadata or {}
        host = meta.get("host", "")
        port = meta.get("port", 0)
        protocol = meta.get("protocol", "http")
        auth = ""
        if with_auth:
            username, password = self._proxy_auth(cred)
            if username:
                user = urllib.parse.quote(username, safe="")
                pw = urllib.parse.quote(password, safe="")
                auth = f"{user}:{pw}@"
        return f"{protocol}://{auth}{host}:{port}"

    @staticmethod
    def _proxy_auth(cred: Credential) -> tuple[str, str]:
        """The proxy's (username, password) from the decrypted vault secret."""
        try:
            blob = json.loads(cred.password or "")
        except (ValueError, TypeError):
            blob = {}
        if isinstance(blob, dict):
            return str(blob.get("username") or ""), str(blob.get("password") or "")
        return "", ""

    def _public_view(
        self, cred: Credential, *, has_auth: bool | None = None
    ) -> dict[str, Any]:
        """Owner-safe view: health + endpoint, password masked, never leaked."""
        meta = cred.metadata or {}
        authed = (
            has_auth if has_auth is not None else bool(meta.get("proxy_username"))
        )
        return {
            "id": cred.username,
            "host": meta.get("host", ""),
            "port": meta.get("port", 0),
            "protocol": meta.get("protocol", "http"),
            "username": str(meta.get("proxy_username") or ""),
            "password": self.MASK if authed else "",
            "has_auth": authed,
            "tags": list(meta.get("tags") or []),
            "health": dict(meta.get("health") or {}) or self._blank_health(),
            "url": self._proxy_url(cred, with_auth=False),
            "added_at": cred.created_at,
            "updated_at": cred.updated_at,
        }

    def _details(self, cred: Credential, *, include_secret: bool) -> dict[str, Any]:
        """Connection details for other components.

        With ``include_secret=True`` the real password and a
        ``url_with_auth`` form are included — treat the result as secret.
        """
        view = self._public_view(cred)
        if include_secret:
            username, password = self._proxy_auth(cred)
            view["username"] = username
            view["password"] = password
            view["url_with_auth"] = self._proxy_url(cred, with_auth=True)
            view["proxy_url"] = view["url_with_auth"]  # alias: hand to HttpClient etc.
        return view
