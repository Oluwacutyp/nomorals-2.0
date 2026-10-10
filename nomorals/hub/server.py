"""Hub HTTP server: mesh + sync over one stdlib listener.

Thread-per-connection (``ThreadingHTTPServer``), same pattern as the
stream server. Each request is handled against the hub's own
:class:`LocalTransport` / :class:`LocalPeer`, so remote nodes get exactly
the semantics the local path has — the transport interface's documented
contract ("the reference implementation every remote transport must
match") holds by construction, not by reimplementation.

Auth posture mirrors the main API server: a configured token is required
on every route except ``/``, ``/health`` and ``/ready``
(``Authorization: Bearer <token>``); with no token the server refuses to
bind a non-loopback address, failing fast at startup instead of exposing
an unauthenticated mesh to a LAN.

Mined design notes (see ``HUB_SWEEP_MINING.md``):

* Long-polling (CouchDB ``_changes?feed=longpoll``, Matrix ``/sync``):
  ``POST /mesh/poll`` accepts ``wait`` and ``GET /sync/pull`` accepts
  ``timeout`` — the server holds the request until work arrives or the
  deadline passes, woken early by ``SyncStore`` notifications.
* Health (gRPC health protocol): ``/health`` always answers 200 with
  per-component ``SERVING``/``NOT_SERVING`` status in the body;
  ``/ready`` is the Kubernetes-style readiness probe (200/503).
* Observability (RQ registries, Syncthing ``strelaysrv /status``,
  Prometheus exposition): ``/mesh/stats``, ``/mesh/jobs``,
  ``/mesh/dead``, ``/metrics``.
* Hardening (Syncthing relay rate limits): per-IP token-bucket rate
  limiting with ``429`` + ``Retry-After``; optional TLS via stdlib
  ``ssl`` (no new dependencies); per-connection socket timeout so a hung
  client cannot park a handler thread forever.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..mesh.errors import MeshError, NodeUnknown, TaskNotFound
from ..mesh.transport import LocalTransport
from ..storage.db import Database
from ..sync.engine import LocalPeer
from ..sync.errors import SyncError
from ..sync.store import SyncRecord, SyncStore
from ..version import __version__

__all__ = [
    "HubServer", "serve",
    "MAX_BODY_BYTES", "PULL_MAX_LIMIT",
    "MAX_WAIT_SECONDS", "DEFAULT_RATE_LIMIT",
]

_log = get_logger(__name__)

#: Largest request body the hub will read — a backstop against a runaway
#: or hostile client. Push batches are already size-bounded client-side
#: (512KB); this is an order of magnitude above any legitimate call.
MAX_BODY_BYTES = 8 * 1024 * 1024

#: Upper bound on one /sync/pull page — protects the hub from a client
#: asking for the whole store in one response.
PULL_MAX_LIMIT = 5000

#: Upper bound on any long-poll wait (mesh poll ``wait``, sync pull
#: ``timeout``, mesh wait ``timeout``). Keeps one slow client from
#: holding a handler thread — and a shutdown — indefinitely.
MAX_WAIT_SECONDS = 60.0

#: Default per-IP request budget: 600 requests per 60s window. Generous
#: for long-polling workers (one request per wait cycle), tight enough to
#: blunt a runaway loop. ``0`` disables. ``NM_HUB_RATE_LIMIT`` overrides.
DEFAULT_RATE_LIMIT = 600
_RATE_WINDOW_SECONDS = 60.0

_DEFAULT_PULL_LIMIT = 500

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

#: Routes that skip auth (probes + service index).
_AUTH_EXEMPT = {"/", "/health", "/ready"}

#: Routes that skip rate limiting (probes must never 429).
_RATE_EXEMPT = {"/", "/health", "/ready"}

#: Human-readable route index served at ``GET /``.
ROUTE_INDEX = (
    "GET  /",
    "GET  /health",
    "GET  /ready",
    "GET  /metrics",
    "POST /mesh/register",
    "POST /mesh/heartbeat",
    "POST /mesh/deregister",
    "GET  /mesh/nodes",
    "POST /mesh/dispatch",
    "POST /mesh/dispatch-many",
    "POST /mesh/poll",
    "POST /mesh/complete",
    "POST /mesh/fail",
    "POST /mesh/cancel",
    "POST /mesh/progress",
    "GET  /mesh/job",
    "GET  /mesh/jobs",
    "GET  /mesh/dead",
    "POST /mesh/retry-dead",
    "POST /mesh/reclaim",
    "POST /mesh/reap-expired",
    "GET  /mesh/stats",
    "GET  /mesh/wait",
    "POST /sync/push",
    "GET  /sync/pull",
)


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK or host.startswith("127.")


class _Raw:
    """A non-JSON response body (e.g. Prometheus text)."""

    def __init__(self, content_type: str, body: bytes) -> None:
        self.content_type = content_type
        self.body = body


class _HttpError(Exception):
    """A route-level failure with an HTTP status and machine code."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


class _RateLimited(Exception):
    """Internal signal: the per-IP bucket is empty."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


class _RateLimiter:
    """Per-key token bucket (GCRA spirit, stdlib only).

    ``capacity`` requests refill over ``window`` seconds. Thread-safe;
    the bucket map is pruned when it grows past 10k keys so a scan of
    spoofed IPs cannot grow memory without bound.
    """

    def __init__(self, capacity: int, window: float) -> None:
        self.capacity = float(capacity)
        self.window = float(window)
        self._refill = self.capacity / self.window if self.window > 0 else 0.0
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self._refill)
            if tokens >= 1.0:
                self._buckets[key] = [tokens - 1.0, now]
                return True, 0.0
            retry_after = ((1.0 - tokens) / self._refill
                           if self._refill > 0 else self.window)
            self._buckets[key] = [tokens, now]
            if len(self._buckets) > 10_000:
                self._buckets = {
                    k: v for k, v in self._buckets.items()
                    if now - v[1] < self.window * 2
                }
            return False, max(0.0, retry_after)


def _q(query: dict[str, list[str]], name: str, default: str = "") -> str:
    vals = query.get(name)
    return vals[0] if vals else default


def _qint(query: dict[str, list[str]], name: str, default: int,
          lo: int | None = None, hi: int | None = None) -> int:
    raw = _q(query, name, "")
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise _HttpError(400, "bad_request",
                         f"bad {name}: {raw!r}") from None
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _qfloat(query: dict[str, list[str]], name: str, default: float) -> float:
    raw = _q(query, name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise _HttpError(400, "bad_request",
                         f"bad {name}: {raw!r}") from None


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the hub (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


class HubServer:
    """Serve mesh + sync endpoints for remote devices.

    ``db`` is the hub's own database (mesh tables + sync records live in
    it). ``device_id`` names the hub's sync store — the hub is a sync peer
    like any device, so records it relays fan out with its own seq space.

    Optional hardening (all stdlib, all optional):

    * ``certfile``/``keyfile`` — serve HTTPS (TLS 1.2+). May also come
      from ``NM_HUB_CERT``/``NM_HUB_KEY``.
    * ``rate_limit`` — per-IP requests per 60s window (``0`` disables).
      ``NM_HUB_RATE_LIMIT`` overrides the default.
    * ``cors_origins`` — tuple of allowed ``Origin`` values (``"*"`` for
      any); enables CORS headers + ``OPTIONS`` preflight. Off by default.
      ``NM_HUB_CORS_ORIGINS`` (comma-separated) overrides.
    * ``request_timeout`` — per-connection socket timeout, seconds.
    * ``max_wait`` — cap for long-poll ``wait``/``timeout`` params.

    Plain-argument config wins; when an argument is left at its default
    the matching ``NM_HUB_*`` env var (``TOKEN``, ``HOST``, ``PORT``,
    ``CERT``, ``KEY``, ``RATE_LIMIT``, ``CORS_ORIGINS``) fills it in.
    """

    def __init__(
        self,
        db: Database,
        *,
        host: str = "127.0.0.1",
        port: int = 8861,
        token: str = "",
        device_id: str = "hub",
        certfile: str | None = None,
        keyfile: str | None = None,
        rate_limit: int | None = None,
        rate_window: float = _RATE_WINDOW_SECONDS,
        cors_origins: tuple[str, ...] | None = None,
        request_timeout: float = 30.0,
        max_wait: float = MAX_WAIT_SECONDS,
    ) -> None:
        if db is None:
            raise ValueError("db is required")
        if host == "127.0.0.1":
            host = os.environ.get("NM_HUB_HOST") or host
        if port == 8861:
            port = int(os.environ.get("NM_HUB_PORT") or port)
        token = token or os.environ.get("NM_HUB_TOKEN") or ""
        if certfile is None:
            certfile = os.environ.get("NM_HUB_CERT") or None
        if keyfile is None:
            keyfile = os.environ.get("NM_HUB_KEY") or None
        if (certfile is None) != (keyfile is None):
            raise ValueError(
                "certfile and keyfile must be given together "
                "(NM_HUB_CERT / NM_HUB_KEY)")
        if rate_limit is None:
            rate_limit = int(os.environ.get("NM_HUB_RATE_LIMIT")
                             or DEFAULT_RATE_LIMIT)
        if cors_origins is None:
            raw = os.environ.get("NM_HUB_CORS_ORIGINS") or ""
            cors_origins = tuple(
                o.strip() for o in raw.split(",") if o.strip())
        if not token and not _is_loopback(host):
            raise ValueError(
                f"refusing to bind hub to non-loopback {host!r} without a "
                "token — set NM_HUB_TOKEN (or hub.token) first")
        self.db = db
        self.host = host
        self.port = port
        self.token = token
        self.certfile = certfile
        self.keyfile = keyfile
        self.request_timeout = float(request_timeout)
        self.max_wait = max(0.0, float(max_wait))
        self.cors_origins = tuple(cors_origins)
        self._limiter = (_RateLimiter(rate_limit, rate_window)
                         if rate_limit and rate_limit > 0 else None)
        self.transport = LocalTransport(db)
        self.sync_peer = LocalPeer(SyncStore(db, device_id=device_id))
        self._started_at = time.time()
        self._metrics_lock = threading.Lock()
        self._req_total: dict[tuple[str, str], int] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def tls(self) -> bool:
        return self.certfile is not None

    @property
    def url(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    # ── auth / limits ────────────────────────────────────────────────────

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.token:
            return True  # loopback-only binding enforced at construction
        presented = handler.headers.get("Authorization", "")
        if not presented.startswith("Bearer "):
            return False
        return hmac.compare_digest(presented[7:].strip(), self.token)

    def _cors_allowed(self, origin: str) -> bool:
        return any(o == "*" or o == origin for o in self.cors_origins)

    def _preflight_headers(self) -> dict[str, str]:
        allow = "*" if "*" in self.cors_origins else ", ".join(
            self.cors_origins)
        return {
            "Access-Control-Allow-Origin": allow,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400",
        }

    # ── request handling ─────────────────────────────────────────────────

    def _read_json(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        try:
            length = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            raise _HttpError(413, "body_too_large",
                             f"request body {length} bytes exceeds "
                             f"{MAX_BODY_BYTES} byte limit")
        raw = handler.rfile.read(max(length, 0)) if length else b""
        if not raw:
            return {}
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise _HttpError(400, "bad_json", f"invalid JSON body: {exc}")
        if not isinstance(doc, dict):
            raise _HttpError(400, "bad_json", "JSON body must be an object")
        return doc

    def _dispatch(self, handler: BaseHTTPRequestHandler,
                  method: str, path: str,
                  query: dict[str, list[str]]
                  ) -> tuple[int, str, bytes, dict[str, str]]:
        """Run one request. Returns (status, content_type, body, headers)."""
        remote = handler.client_address[0] if handler.client_address else "?"
        if method == "OPTIONS":
            if not self.cors_origins:
                raise _HttpError(404, "not_found",
                                 f"no route {method} {path}")
            return 204, "text/plain; charset=utf-8", b"", \
                self._preflight_headers()
        if path not in _RATE_EXEMPT and self._limiter is not None:
            allowed, retry_after = self._limiter.check(remote)
            if not allowed:
                _log.warning("hub rate-limited %s on %s %s", remote,
                             method, path)
                raise _RateLimited(retry_after)
        if path not in _AUTH_EXEMPT and not self._authorized(handler):
            _log.warning("hub unauthorized %s %s from %s", method, path,
                         remote)
            _emit("hub.auth_failed", {"path": path, "remote": remote})
            raise _HttpError(401, "unauthorized",
                             "missing or invalid bearer token")
        body = self._read_json(handler) if method == "POST" else {}
        result = self._route(handler, method, path, query, body)
        if isinstance(result, _Raw):
            return 200, result.content_type, result.body, {}
        raw = json.dumps(result, default=str).encode("utf-8")
        return 200, "application/json; charset=utf-8", raw, {}

    def _send(self, handler: BaseHTTPRequestHandler, status: int,
              content_type: str, raw: bytes,
              extra: dict[str, str]) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(raw)))
        handler.send_header("Cache-Control", "no-store")
        for key, value in extra.items():
            handler.send_header(key, value)
        origin = handler.headers.get("Origin")
        if self.cors_origins and origin and self._cors_allowed(origin):
            handler.send_header("Access-Control-Allow-Origin", origin)
            handler.send_header("Vary", "Origin")
        handler.end_headers()
        if raw:
            handler.wfile.write(raw)

    def _record(self, path: str, status: int) -> None:
        bucket = f"{status // 100}xx"
        with self._metrics_lock:
            key = (path, bucket)
            self._req_total[key] = self._req_total.get(key, 0) + 1

    # ── routes ───────────────────────────────────────────────────────────

    def _route(self, handler: BaseHTTPRequestHandler,
               method: str, path: str,
               query: dict[str, list[str]], body: dict[str, Any]) -> Any:
        # ── service / ops ──
        if method == "GET" and path == "/":
            return {"ok": True, "service": "nomorals-hub",
                    "version": __version__, "url": self.url,
                    "tls": self.tls, "routes": list(ROUTE_INDEX)}
        if method == "GET" and path == "/health":
            return self._health()
        if method == "GET" and path == "/ready":
            return self._ready()
        if method == "GET" and path == "/metrics":
            return self._metrics_text()

        # ── mesh: nodes ──
        if method == "POST" and path == "/mesh/register":
            labels = body.get("labels")
            node = self.transport.nodes.register(
                str(body.get("name") or ""),
                platform=str(body.get("platform") or ""),
                capabilities=[str(c) for c in (body.get("capabilities")
                                               or [])],
                node_id=body.get("node_id"),
                labels=({str(k): str(v) for k, v in labels.items()}
                        if isinstance(labels, dict) else None),
            )
            return {"ok": True, "node": node.to_dict()}
        if method == "POST" and path == "/mesh/heartbeat":
            node_id = str(body.get("node_id") or "")
            info = body.get("info")
            self.transport.heartbeat(
                node_id,
                info=dict(info) if isinstance(info, dict) else None,
            )
            return {"ok": True}
        if method == "POST" and path == "/mesh/deregister":
            node_id = str(body.get("node_id") or "")
            if not node_id:
                raise _HttpError(400, "bad_request",
                                 "node_id is required")
            self.transport.deregister(node_id)
            return {"ok": True}
        if method == "GET" and path == "/mesh/nodes":
            if _q(query, "all") in ("1", "true", "yes"):
                nodes = self.transport.nodes.list_all()
            else:
                nodes = self.transport.active_nodes()
            return {"ok": True, "nodes": [
                n.to_dict() for n in nodes]}

        # ── mesh: tasks ──
        if method == "POST" and path == "/mesh/dispatch":
            return {"ok": True, "job_id": self._dispatch_one(body)}
        if method == "POST" and path == "/mesh/dispatch-many":
            return {"ok": True, "job_ids": self._dispatch_many(body)}
        if method == "POST" and path == "/mesh/poll":
            wait = min(max(float(body.get("wait") or 0.0), 0.0),
                       self.max_wait)
            tasks = self.transport.poll(
                str(body.get("node_id") or ""),
                batch=int(body.get("batch") or 5),
                wait=wait,
            )
            return {"ok": True, "waited": wait > 0,
                    "tasks": [t.to_dict() for t in tasks]}
        if method == "POST" and path == "/mesh/complete":
            self.transport.complete(str(body.get("job_id") or ""),
                                    result=body.get("result"))
            return {"ok": True}
        if method == "POST" and path == "/mesh/fail":
            self.transport.fail(str(body.get("job_id") or ""),
                                error=str(body.get("error") or ""),
                                retry=bool(body.get("retry", True)))
            return {"ok": True}
        if method == "POST" and path == "/mesh/cancel":
            job_id = str(body.get("job_id") or "")
            cancelled = self.transport.cancel(job_id)
            return {"ok": True, "cancelled": bool(cancelled)}
        if method == "POST" and path == "/mesh/progress":
            job_id = str(body.get("job_id") or "")
            node_id = str(body.get("node_id") or "")
            detail = body.get("detail") or {}
            if not isinstance(detail, dict):
                raise _HttpError(400, "bad_request",
                                 "progress detail must be an object")
            alive = self.transport.tasks.heartbeat(job_id, node_id,
                                                   detail=dict(detail))
            return {"ok": True, "lease_alive": bool(alive)}
        if method == "GET" and path == "/mesh/job":
            return {"ok": True, "job": self._job_detail(
                _q(query, "job_id"))}
        if method == "GET" and path == "/mesh/jobs":
            node_id = _q(query, "node") or None
            limit = _qint(query, "limit", 50, lo=1, hi=500)
            jobs = self.transport.tasks.list_live(node_id=node_id,
                                                  limit=limit)
            return {"ok": True, "jobs": [t.to_dict() for t in jobs]}
        if method == "GET" and path == "/mesh/dead":
            limit = _qint(query, "limit", 100, lo=1, hi=500)
            dead = self.transport.tasks.dead(limit=limit)
            return {"ok": True, "jobs": [t.to_dict() for t in dead]}
        if method == "POST" and path == "/mesh/retry-dead":
            job_id = str(body.get("job_id") or "")
            delay = float(body.get("delay") or 0.0)
            replayed = self.transport.tasks.retry_dead(job_id, delay=delay)
            return {"ok": True, "replayed": bool(replayed)}
        if method == "POST" and path == "/mesh/reclaim":
            n = self.transport.tasks.reclaim()
            return {"ok": True, "reclaimed": int(n)}
        if method == "POST" and path == "/mesh/reap-expired":
            n = self.transport.tasks.reap_expired()
            return {"ok": True, "reaped": int(n)}
        if method == "GET" and path == "/mesh/stats":
            return {"ok": True, "stats": self.transport.stats()}
        if method == "GET" and path == "/mesh/wait":
            return self._wait_for_job(query)

        # ── sync ──
        if method == "POST" and path == "/sync/push":
            return self._sync_push(body)
        if method == "GET" and path == "/sync/pull":
            return self._sync_pull(query)

        raise _HttpError(404, "not_found", f"no route {method} {path}")

    # ── mesh helpers ─────────────────────────────────────────────────────

    def _dispatch_opts(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            raise _HttpError(400, "bad_request",
                             "dispatch payload must be an object")
        caps = [str(c) for c in (body.get("target_capabilities") or [])]
        labels = body.get("target_labels")
        expire_after = body.get("expire_after")
        return {
            "task_type": str(body.get("task_type") or ""),
            "payload": payload,
            "origin_node": str(body.get("origin_node") or ""),
            "target_node": body.get("target_node"),
            "priority": int(body.get("priority") or 0),
            "dedupe_key": (str(body["dedupe_key"])
                           if body.get("dedupe_key") else None),
            "delay": float(body.get("delay") or 0.0),
            "max_attempts": int(body.get("max_attempts") or 5),
            "expire_after": (float(expire_after)
                             if expire_after not in (None, "") else None),
            "target_capabilities": caps or None,
            "target_labels": ({str(k): str(v) for k, v in labels.items()}
                              if isinstance(labels, dict) and labels
                              else None),
        }

    def _dispatch_one(self, body: dict[str, Any]) -> str:
        opts = self._dispatch_opts(body)
        return self.transport.tasks.dispatch(
            opts.pop("task_type"), opts.pop("payload"), **opts)

    def _dispatch_many(self, body: dict[str, Any]) -> list[str]:
        payloads = body.get("payloads") or []
        if (not isinstance(payloads, list) or not payloads
                or not all(isinstance(p, dict) for p in payloads)):
            raise _HttpError(400, "bad_request",
                             "payloads must be a non-empty list of objects")
        opts = self._dispatch_opts({**body, "payload": {}})
        opts.pop("dedupe_key", None)  # one key can't dedupe a fan-out
        opts.pop("payload", None)  # placeholder only; payloads carry the data
        task_type = opts.pop("task_type")
        return self.transport.tasks.dispatch_many(task_type, payloads,
                                                  **opts)

    def _job_detail(self, job_id: str) -> dict[str, Any]:
        if not job_id:
            raise _HttpError(400, "bad_request", "job_id is required")
        tasks = self.transport.tasks
        job = tasks.queue.get(job_id)
        if job is None:
            raise _HttpError(404, "job_not_found",
                             f"no task with job id {job_id!r}")
        detail = type(tasks)._to_task(job).to_dict()
        detail["progress"] = tasks.progress(job_id)
        try:
            detail["result"] = tasks.result(job_id)
        except TaskNotFound:
            detail["result"] = None
        return detail

    def _wait_for_job(self, query: dict[str, list[str]]) -> dict[str, Any]:
        job_id = _q(query, "job_id")
        if not job_id:
            raise _HttpError(400, "bad_request", "job_id is required")
        timeout = min(max(_qfloat(query, "timeout", 30.0), 0.0),
                      self.max_wait)
        try:
            result = self.transport.tasks.wait_for_result(
                job_id, timeout=timeout)
        except TaskNotFound:
            raise _HttpError(404, "job_not_found",
                             f"no task with job id {job_id!r}") from None
        except TimeoutError:
            return {"ok": True, "ready": False}
        except MeshError as exc:
            # Terminal but unsuccessful: died or cancelled.
            return {"ok": True, "ready": True, "terminal": True,
                    "error": str(exc)}
        return {"ok": True, "ready": True, "terminal": True,
                "result": result}

    # ── sync helpers ─────────────────────────────────────────────────────

    def _sync_push(self, body: dict[str, Any]) -> dict[str, Any]:
        records = body.get("records") or []
        if not isinstance(records, list):
            raise _HttpError(400, "bad_request",
                             "sync records must be a list")
        # CouchDB _bulk_docs style: per-record results, never one
        # all-or-nothing verdict for a batch with a single bad row.
        valid: list[SyncRecord] = []
        results: list[dict[str, Any]] = []
        for index, doc in enumerate(records):
            if not isinstance(doc, dict):
                results.append({"index": index, "id": None, "ok": False,
                                "error": "record must be an object"})
                continue
            try:
                rec = SyncRecord.from_dict(doc)
            except Exception as exc:  # noqa: BLE001 - per-record report
                results.append({"index": index, "id": doc.get("key"),
                                "ok": False,
                                "error": f"bad_record: {exc}"})
                continue
            valid.append(rec)
            results.append({"index": index, "id": rec.key, "ok": True})
        applied = self.sync_peer.push_records(valid)
        return {"ok": True, "applied": int(applied), "results": results}

    def _sync_pull(self, query: dict[str, list[str]]) -> dict[str, Any]:
        limit = self._pull_limit(query)
        store = self.sync_peer.store
        timeout = min(max(_qfloat(query, "timeout", 0.0), 0.0),
                      self.max_wait)
        if "since_seq" in query:
            seq = _qint(query, "since_seq", 0)
            recs = self._pull_wait_seq(store, seq, limit, timeout)
            last = recs[-1].seq if recs else seq
            return {"ok": True,
                    "records": [r.to_dict() for r in recs],
                    "last_seq": store.max_seq(),
                    "pending": store.count_since_seq(last)}
        if "since_ts" in query:
            since = _qfloat(query, "since_ts", 0.0)
            all_recs = store.list_changed_since(since)
            page = all_recs[:limit]
            return {"ok": True,
                    "records": [r.to_dict() for r in page],
                    "last_seq": store.max_seq(),
                    "pending": max(0, len(all_recs) - len(page))}
        raise _HttpError(400, "bad_request",
                         "sync pull needs since_seq or since_ts")

    def _pull_wait_seq(self, store: SyncStore, seq: int, limit: int,
                       timeout: float) -> list[SyncRecord]:
        """Seq-cursor pull, long-polling when ``timeout`` > 0.

        Matrix ``/sync`` / CouchDB ``longpoll`` semantics: hold the
        request open until a record lands or the deadline passes. The
        store's subscriber list wakes sleepers the moment a push is
        applied — no blind re-polling.
        """
        if timeout <= 0:
            return store.list_since_seq(seq, limit=limit)
        arrived = threading.Event()

        def _wake(rec: SyncRecord) -> None:
            arrived.set()

        # Subscribe BEFORE the first read: a notify that lands between
        # read and wait sets the event, so we never sleep through data.
        store.subscribe(_wake)
        try:
            recs = store.list_since_seq(seq, limit=limit)
            if not recs:
                arrived.wait(timeout)
                recs = store.list_since_seq(seq, limit=limit)
        finally:
            store.unsubscribe(_wake)
        return recs

    @staticmethod
    def _pull_limit(query: dict[str, list[str]]) -> int:
        return _qint(query, "limit", _DEFAULT_PULL_LIMIT,
                     lo=1, hi=PULL_MAX_LIMIT)

    # ── ops helpers ──────────────────────────────────────────────────────

    def _component_status(self) -> dict[str, dict[str, Any]]:
        """gRPC-health-style per-component checks (SERVING/NOT_SERVING)."""
        comps: dict[str, dict[str, Any]] = {}
        try:
            latency_ms = self.transport.ping() * 1000.0
            comps["mesh"] = {"status": "SERVING",
                             "latency_ms": round(latency_ms, 2)}
        except Exception as exc:  # noqa: BLE001 - health must not raise
            comps["mesh"] = {"status": "NOT_SERVING", "error": str(exc)}
        try:
            alive = bool(self.sync_peer.ping())
            comps["sync"] = {"status": "SERVING" if alive else "NOT_SERVING"}
        except Exception as exc:  # noqa: BLE001 - health must not raise
            comps["sync"] = {"status": "NOT_SERVING", "error": str(exc)}
        return comps

    def _health(self) -> dict[str, Any]:
        components = self._component_status()
        ok = all(c["status"] == "SERVING" for c in components.values())
        try:
            totals = self.transport.stats().get("totals", {})
        except Exception:  # noqa: BLE001 - health must not raise
            totals = {}
        try:
            nodes_active = len(self.transport.active_nodes())
        except Exception:  # noqa: BLE001 - health must not raise
            nodes_active = -1
        try:
            head_seq = self.sync_peer.store.max_seq()
        except Exception:  # noqa: BLE001 - health must not raise
            head_seq = -1
        return {
            "ok": ok,
            "version": __version__,
            "ts": time.time(),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "tls": self.tls,
            "components": components,
            "nodes_active": nodes_active,
            "mesh_tasks": totals,
            "sync_head_seq": head_seq,
        }

    def _ready(self) -> dict[str, Any]:
        components = self._component_status()
        failing = [name for name, comp in components.items()
                   if comp["status"] != "SERVING"]
        if failing:
            raise _HttpError(503, "not_ready",
                             "failing components: " + ", ".join(failing))
        return {"ready": True, "components": components}

    def _metrics_text(self) -> _Raw:
        """Prometheus text exposition (``text/plain; version=0.0.4``)."""
        lines = [
            "# HELP nomorals_hub_requests_total HTTP requests served.",
            "# TYPE nomorals_hub_requests_total counter",
        ]
        with self._metrics_lock:
            items = sorted(self._req_total.items())
        for (route, status), count in items:
            lines.append(
                f'nomorals_hub_requests_total{{route="{route}",'
                f'status="{status}"}} {count}')
        lines += [
            "# HELP nomorals_hub_uptime_seconds Hub uptime.",
            "# TYPE nomorals_hub_uptime_seconds gauge",
            f"nomorals_hub_uptime_seconds {self.uptime_seconds:.1f}",
            "# HELP nomorals_hub_nodes_active Freshly-heartbeating nodes.",
            "# TYPE nomorals_hub_nodes_active gauge",
        ]
        try:
            lines.append(
                f"nomorals_hub_nodes_active "
                f"{len(self.transport.active_nodes())}")
        except Exception:  # noqa: BLE001 - metrics must not raise
            lines.append("nomorals_hub_nodes_active -1")
        lines += [
            "# HELP nomorals_hub_mesh_tasks Mesh tasks by status.",
            "# TYPE nomorals_hub_mesh_tasks gauge",
        ]
        try:
            totals = self.transport.stats().get("totals", {})
            for status, count in sorted(totals.items()):
                lines.append(
                    f'nomorals_hub_mesh_tasks{{status="{status}"}} {count}')
        except Exception:  # noqa: BLE001 - metrics must not raise
            pass
        lines += [
            "# HELP nomorals_hub_sync_head_seq Sync store head sequence.",
            "# TYPE nomorals_hub_sync_head_seq gauge",
        ]
        try:
            lines.append(
                f"nomorals_hub_sync_head_seq {self.sync_peer.store.max_seq()}")
        except Exception:  # noqa: BLE001 - metrics must not raise
            pass
        text = "\n".join(lines) + "\n"
        return _Raw("text/plain; version=0.0.4; charset=utf-8",
                    text.encode("utf-8"))

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, background: bool = True) -> None:
        if self._server is not None:
            raise RuntimeError("hub server already started")
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = f"NoMoralsHub/{__version__}"

            def log_message(self, fmt: str, *args: Any) -> None:
                _log.debug("hub %s - %s", self.address_string(), fmt % args)

            def _handle(self, method: str) -> None:
                started = time.monotonic()
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                status, content_type, raw, extra = (
                    500, "application/json; charset=utf-8", b"{}", {})
                try:
                    (status, content_type, raw,
                     extra) = server._dispatch(self, method, path, query)
                except _HttpError as exc:
                    status = exc.status
                    raw = json.dumps({
                        "ok": False, "code": exc.code, "error": exc.code,
                        "detail": exc.detail}).encode("utf-8")
                except _RateLimited as exc:
                    status = 429
                    extra = {"Retry-After": str(max(
                        1, int(math.ceil(exc.retry_after))))}
                    raw = json.dumps({
                        "ok": False, "code": "rate_limited",
                        "error": "rate_limited",
                        "detail": "too many requests; slow down"}).encode(
                            "utf-8")
                except NodeUnknown as exc:
                    status = 404
                    raw = json.dumps({
                        "ok": False, "code": "node_unknown",
                        "error": "node_unknown",
                        "detail": str(exc)}).encode("utf-8")
                except TaskNotFound as exc:
                    status = 404
                    raw = json.dumps({
                        "ok": False, "code": "job_not_found",
                        "error": "job_not_found",
                        "detail": str(exc)}).encode("utf-8")
                except ValueError as exc:
                    status = 400
                    raw = json.dumps({
                        "ok": False, "code": "bad_request",
                        "error": "bad_request",
                        "detail": str(exc)}).encode("utf-8")
                except (MeshError, SyncError) as exc:
                    _log.exception("hub route %s %s failed", method, path)
                    status = 500
                    raw = json.dumps({
                        "ok": False, "code": "hub_error",
                        "error": "hub_error",
                        "detail": str(exc)}).encode("utf-8")
                except Exception as exc:  # noqa: BLE001 - hub must stay up
                    _log.exception("hub route %s %s failed", method, path)
                    status = 500
                    raw = json.dumps({
                        "ok": False, "code": "internal",
                        "error": "internal",
                        "detail": str(exc)}).encode("utf-8")
                server._send(self, status, content_type, raw, extra)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                server._record(path, status)
                _log.debug("hub %s %s -> %d (%.1f ms)", method, path,
                           status, elapsed_ms)

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._handle("OPTIONS")

        Handler.timeout = self.request_timeout
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        # Handler threads must never block interpreter shutdown; a
        # bounded long-poll (max_wait) is the worst case a stop() waits out.
        self._server.daemon_threads = True
        if self.certfile:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            if hasattr(ssl, "TLSVersion"):
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(self.certfile, self.keyfile)
            self._server.socket = context.wrap_socket(
                self._server.socket, server_side=True)
        self.port = self._server.server_address[1]  # port=0 → real port
        _emit("hub.started", {"host": self.host, "port": self.port,
                              "url": self.url, "tls": self.tls})
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="hub-server",
                daemon=True,
            )
            self._thread.start()
            _log.info("hub server on %s:%d (background)", self.host, self.port)
        else:
            _log.info("hub server on %s:%d (foreground)", self.host, self.port)
            self._server.serve_forever()

    def stop(self) -> None:
        was_running = self._server is not None or self._thread is not None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if was_running:
            _emit("hub.stopped", {"host": self.host, "port": self.port,
                                  "url": self.url})


def serve(
    db: Database,
    *,
    host: str = "127.0.0.1",
    port: int = 8861,
    token: str = "",
    device_id: str = "hub",
    certfile: str | None = None,
    keyfile: str | None = None,
    rate_limit: int | None = None,
    rate_window: float = 60.0,
    cors_origins: tuple[str, ...] | None = None,
    request_timeout: float = 30.0,
    max_wait: float = MAX_WAIT_SECONDS,
    background: bool = True,
) -> HubServer:
    """Start the hub server. Returns the server instance.

    Extra keyword arguments (``certfile``, ``keyfile``, ``rate_limit``,
    ``rate_window``, ``cors_origins``, ``request_timeout``, ``max_wait``)
    are passed to :class:`HubServer`; unset values fall back to
    ``NM_HUB_*`` env vars.
    """
    server = HubServer(db, host=host, port=port, token=token,
                       device_id=device_id, certfile=certfile,
                       keyfile=keyfile, rate_limit=rate_limit,
                       rate_window=rate_window,
                       cors_origins=cors_origins,
                       request_timeout=request_timeout, max_wait=max_wait)
    server.start(background=background)
    return server
