"""HTTP mesh transport: a device mesh across the network.

:class:`LocalTransport` shares one database; :class:`HttpTransport` talks
to a remote hub (``nomorals.hub``) exposing the same operations as JSON
over HTTP. Same abstract interface — node code built on
:class:`~nomorals.mesh.transport.Transport` needs no changes to join a
hub instead of a shared volume.

Auth is a shared bearer secret (``NM_HUB_TOKEN``) presented as
``Authorization: Bearer <token>``, the same posture as the main API
server. The wire errors map back to the exceptions the local transport
raises: unknown node → :class:`NodeUnknown`, bad request →
:exc:`ValueError`, auth rejected → :class:`AuthError`, everything
unreachable → :class:`TransportError` / :class:`HubUnreachable`.

Resilience (the AWS Builders' Library + Resilience4j gold):

- **Full-jitter backoff** on retries: ``uniform(0, min(cap, base*2^attempt))``
  so nodes that lost the hub together don't retry in lockstep.
- **Retry the retryable**: network errors, timeouts, HTTP 5xx and 429
  (honoring ``Retry-After``). Never 4xx — those are answers, not blips.
- **Circuit breaker** (CLOSED → OPEN → HALF_OPEN): after a run of failures
  the transport fails fast without touching the network, giving the hub
  room to recover; a trial probe closes the circuit again.

Stdlib only (``http.client``) — no mandatory dependencies, Termux-safe.
Connections are kept alive and reused per host.
"""

from __future__ import annotations

import http.client
import json
import random
import threading
import time
import urllib.parse
from typing import Any

from ..core.logging_setup import get_logger
from .errors import (
    AuthError,
    CircuitOpen,
    HubUnreachable,
    MeshError,
    NodeUnknown,
    TransportError,
)
from .node import MeshNode
from .tasks import MeshTask
from .transport import Transport

__all__ = [
    "HttpTransport",
    "CircuitBreaker",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
]

_log = get_logger(__name__)

DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_CAP = 30.0
USER_AGENT = "nomorals-mesh/1.0"


def _full_jitter(attempt: int, base: float = DEFAULT_BACKOFF_BASE,
                 cap: float = DEFAULT_BACKOFF_CAP) -> float:
    """AWS 'full jitter': uniform(0, min(cap, base * 2**attempt))."""
    return random.uniform(0.0, min(cap, base * (2.0 ** max(0, attempt))))


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse a Retry-After header (delta-seconds form)."""
    try:
        raw = headers.get("Retry-After") if headers else None
    except Exception:  # noqa: BLE001 - best effort
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


class CircuitBreaker:
    """Resilience4j-style circuit breaker, stdlib edition.

    CLOSED: calls pass through, consecutive failures counted.
    OPEN: calls fail fast with :class:`CircuitOpen` — no network touched.
    HALF_OPEN: after ``reset_timeout`` one trial call is allowed through;
    success closes the circuit, failure re-opens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout = float(reset_timeout)
        if self.reset_timeout <= 0:
            raise ValueError("reset_timeout must be positive")
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (
            self._state == self.OPEN
            and time.monotonic() - self._opened_at >= self.reset_timeout
        ):
            self._state = self.HALF_OPEN

    def before_call(self) -> None:
        with self._lock:
            self._maybe_half_open()
            if self._state == self.OPEN:
                raise CircuitOpen(
                    "hub circuit is open: failing fast without network call",
                    detail={"reset_in": max(
                        0.0,
                        self.reset_timeout - (time.monotonic() - self._opened_at),
                    )},
                )

    def after_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state != self.CLOSED:
                _log.info("hub circuit closed (trial probe succeeded)")
            self._state = self.CLOSED

    def after_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == self.HALF_OPEN:
                self._open_locked("trial probe failed")
            elif self._failures >= self.failure_threshold:
                self._open_locked(
                    f"{self._failures} consecutive failures")

    def _open_locked(self, reason: str) -> None:
        if self._state != self.OPEN:
            _log.warning("hub circuit opened: %s", reason)
        self._state = self.OPEN
        self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Force the circuit closed (operator override)."""
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0

    def describe(self) -> str:
        glyph = {self.CLOSED: "●", self.OPEN: "○",
                 self.HALF_OPEN: "◐"}[self.state]
        return (f"{glyph} circuit {self.state} "
                f"(failures: {self._failures}/{self.failure_threshold})")


class _KeepAlivePool:
    """Minimal persistent-connection pool over stdlib http.client.

    One connection per (scheme, host, port); a broken connection is
    dropped and rebuilt on next use. Thread-safe.
    """

    def __init__(self) -> None:
        self._conns: dict[tuple[str, str, int], http.client.HTTPConnection] = {}
        self._lock = threading.Lock()

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, Any, bytes]:
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme or "http"
        host = parts.hostname or "localhost"
        port = parts.port or (443 if scheme == "https" else 80)
        key = (scheme, host, port)
        path = urllib.parse.urlunsplit(("", "", parts.path or "/",
                                        parts.query, ""))

        with self._lock:
            conn = self._conns.get(key)
            if conn is None:
                conn = self._new_connection(scheme, host, port, timeout)
                self._conns[key] = conn

        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            status, resp_headers = resp.status, resp.headers
        except Exception:
            # Drop the (probably half-dead) connection; the caller retries.
            with self._lock:
                self._conns.pop(key, None)
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            raise
        return status, resp_headers, raw

    @staticmethod
    def _new_connection(scheme: str, host: str, port: int,
                        timeout: float) -> http.client.HTTPConnection:
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, {}
        for conn in conns.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best effort
                pass


def _auth_headers(token: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class HttpTransport(Transport):
    """Mesh transport over HTTP to a remote hub.

    ``base_url`` is the hub root, e.g. ``http://192.168.1.5:8861``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        circuit_failure_threshold: int = 5,
        circuit_reset_timeout: float = 30.0,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.circuit = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            reset_timeout=circuit_reset_timeout,
        )
        self._pool = _KeepAlivePool()

    # ── wire plumbing ────────────────────────────────────────────────────

    @property
    def circuit_state(self) -> str:
        """closed | open | half_open — for status displays."""
        return self.circuit.state

    def reset_circuit(self) -> None:
        """Operator override: force the circuit closed."""
        self.circuit.reset()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
        body = None
        if payload is not None:
            body = json.dumps(payload, default=str).encode("utf-8")
        headers = _auth_headers(self.token)
        if body is not None:
            headers["Content-Length"] = str(len(body))
        call_timeout = self.timeout if timeout is None else timeout

        # Fail fast when the circuit is open — no network touched.
        self.circuit.before_call()

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                status, resp_headers, raw = self._pool.request(
                    method, url, body, headers, call_timeout)
            except (TimeoutError, OSError) as exc:
                last_exc = exc
                self.circuit.after_failure()
                _log.warning(
                    "hub %s %s failed (attempt %d/%d): %s",
                    method, path, attempt + 1, self.retries, exc)
                self._sleep_before_retry(attempt, None)
                continue
            except Exception as exc:  # noqa: BLE001 - pool-level surprise
                last_exc = exc
                self.circuit.after_failure()
                self._sleep_before_retry(attempt, None)
                continue

            if status in (429,) or 500 <= status <= 599:
                # Retryable HTTP: back off (honor Retry-After), count the
                # failure against the circuit, try again.
                self.circuit.after_failure()
                detail = self._error_detail(raw)
                _log.warning(
                    "hub %s %s -> %s (attempt %d/%d): %s",
                    method, path, status, attempt + 1, self.retries, detail)
                last_exc = TransportError(f"hub error {status}: {detail}")
                self._sleep_before_retry(
                    attempt, _retry_after_seconds(resp_headers))
                continue

            if 400 <= status <= 499:
                # Answers, not blips: translate once, never retry, and don't
                # trip the circuit for client-side mistakes.
                raise self._translate_http_error(status, raw)

            self.circuit.after_success()
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise TransportError(
                    f"hub returned non-JSON at {path}") from exc

        # Every attempt already counted its failure above; the logical call
        # just ran out of attempts.
        raise HubUnreachable(
            f"hub unreachable at {self.base_url} after {self.retries} "
            f"attempt(s): {last_exc}") from last_exc

    def _sleep_before_retry(self, attempt: int,
                            retry_after: float | None) -> None:
        if attempt + 1 >= self.retries:
            return
        delay = _full_jitter(attempt, self.backoff_base, self.backoff_cap)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_cap))
        time.sleep(delay)

    @staticmethod
    def _error_detail(raw: bytes) -> str:
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - best-effort error detail
            return ""
        return str(body.get("detail") or body.get("error") or "")

    @staticmethod
    def _translate_http_error(status: int, raw: bytes) -> Exception:
        """Map a hub HTTP status to the exception the local path raises."""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001 - best-effort error detail
            body = {}
        detail = str(body.get("detail") or body.get("error") or "")
        error_code = str(body.get("code") or "")
        if status == 401:
            return AuthError(detail or "hub rejected credentials")
        if status == 404 and error_code == "node_unknown":
            return NodeUnknown(detail or "node not registered")
        if status == 400:
            return ValueError(detail or "bad request")
        return TransportError(f"hub error {status}: {detail}")

    # ── Transport interface ──────────────────────────────────────────────

    def register(
        self, name: str, platform: str = "",
        capabilities: list[str] | None = None,
        node_id: str | None = None,
    ) -> MeshNode:
        data = self._request("POST", "/mesh/register", payload={
            "name": name,
            "platform": platform,
            "capabilities": capabilities or [],
            "node_id": node_id,
        })
        return MeshNode.from_dict(data.get("node") or {})

    def heartbeat(self, node_id: str,
                  info: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"node_id": node_id}
        if info:
            payload["info"] = info
        self._request("POST", "/mesh/heartbeat", payload=payload)

    def deregister(self, node_id: str) -> None:
        # The hub API has no deregister endpoint; say so plainly instead of
        # pretending the node left.
        raise MeshError(
            f"hub does not expose node deregistration for {node_id!r}")

    def active_nodes(self) -> list[MeshNode]:
        data = self._request("GET", "/mesh/nodes")
        return [MeshNode.from_dict(n) for n in (data.get("nodes") or [])]

    def ping(self) -> float:
        """Liveness probe: GET /health, returns round-trip seconds."""
        start = time.perf_counter()
        self._request("GET", "/health")
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
        data = self._request("POST", "/mesh/dispatch", payload={
            "task_type": task_type,
            "payload": payload or {},
            "origin_node": origin_node,
            "target_node": target_node,
            "priority": priority,
        })
        job_id = str(data.get("job_id") or "")
        if not job_id:
            raise TransportError("hub dispatch returned no job_id")
        return job_id

    def poll(self, node_id: str, *, batch: int = 5,
             wait: float = 0.0) -> list[MeshTask]:
        """Short-poll the hub, or long-poll client-side when ``wait`` > 0:
        keep asking until tasks arrive or the deadline passes. Works
        against the existing hub — no server change needed."""
        if wait <= 0:
            return self._poll_once(node_id, batch=batch)
        deadline = time.monotonic() + wait
        while True:
            tasks = self._poll_once(node_id, batch=batch)
            if tasks:
                return tasks
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            # Small pause between rounds; the hub answers instantly so the
            # wait loop is what creates the long-poll effect.
            time.sleep(min(1.0, remaining))

    def _poll_once(self, node_id: str, *, batch: int) -> list[MeshTask]:
        data = self._request("POST", "/mesh/poll", payload={
            "node_id": node_id,
            "batch": batch,
        })
        return [MeshTask.from_dict(t) for t in (data.get("tasks") or [])]

    def complete(self, job_id: str, result: Any = None) -> None:
        self._request("POST", "/mesh/complete", payload={
            "job_id": job_id,
            "result": result,
        })

    def fail(self, job_id: str, error: str = "", *, retry: bool = True) -> None:
        self._request("POST", "/mesh/fail", payload={
            "job_id": job_id,
            "error": error,
            "retry": retry,
        })

    def cancel(self, job_id: str) -> bool:
        raise MeshError(
            "hub does not expose task cancellation; "
            "cancel against the hub's local transport instead")

    def result(self, job_id: str) -> Any:
        raise MeshError(
            "hub does not expose task results; "
            "fetch results against the hub's local transport instead")

    def stats(self) -> dict[str, Any]:
        raise MeshError(
            "hub does not expose queue stats; "
            "introspect against the hub's local transport instead")

    def close(self) -> None:
        """Drop pooled keep-alive connections."""
        self._pool.close()
