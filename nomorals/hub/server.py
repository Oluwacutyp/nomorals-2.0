"""Hub HTTP server: mesh + sync over one stdlib listener.

Thread-per-connection (``ThreadingHTTPServer``), same pattern as the
stream server. Each request is handled against the hub's own
:class:`LocalTransport` / :class:`LocalPeer`, so remote nodes get exactly
the semantics the local path has — the transport interface's documented
contract ("the reference implementation every remote transport must
match") holds by construction, not by reimplementation.

Auth posture mirrors the main API server: a configured token is required
on every route except ``/health`` (``Authorization: Bearer <token>``);
with no token the server refuses to bind a non-loopback address, failing
fast at startup instead of exposing an unauthenticated mesh to a LAN.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..mesh.errors import MeshError, NodeUnknown
from ..mesh.transport import LocalTransport
from ..storage.db import Database
from ..sync.engine import LocalPeer
from ..sync.errors import SyncError
from ..sync.store import SyncRecord, SyncStore
from ..version import __version__

__all__ = ["HubServer", "serve", "MAX_BODY_BYTES", "PULL_MAX_LIMIT"]

_log = get_logger(__name__)

#: Largest request body the hub will read — a backstop against a runaway
#: or hostile client. Push batches are already size-bounded client-side
#: (512KB); this is an order of magnitude above any legitimate call.
MAX_BODY_BYTES = 8 * 1024 * 1024

#: Upper bound on one /sync/pull page — protects the hub from a client
#: asking for the whole store in one response.
PULL_MAX_LIMIT = 5000

_DEFAULT_PULL_LIMIT = 500

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK or host.startswith("127.")


def _send_json(handler: Any, status: int, payload: dict) -> None:
    raw = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


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
    """

    def __init__(
        self,
        db: Database,
        *,
        host: str = "127.0.0.1",
        port: int = 8861,
        token: str = "",
        device_id: str = "hub",
    ) -> None:
        if db is None:
            raise ValueError("db is required")
        token = token or ""
        if not token and not _is_loopback(host):
            raise ValueError(
                f"refusing to bind hub to non-loopback {host!r} without a "
                "token — set NM_HUB_TOKEN (or hub.token) first")
        self.db = db
        self.host = host
        self.port = port
        self.token = token
        self.transport = LocalTransport(db)
        self.sync_peer = LocalPeer(SyncStore(db, device_id=device_id))
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── auth ─────────────────────────────────────────────────────────────

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.token:
            return True  # loopback-only binding enforced at construction
        presented = handler.headers.get("Authorization", "")
        if not presented.startswith("Bearer "):
            return False
        return hmac.compare_digest(presented[7:].strip(), self.token)

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

    def _route(self, handler: BaseHTTPRequestHandler,
               method: str, path: str,
               query: dict[str, list[str]], body: dict[str, Any]) -> dict:
        if method == "GET" and path == "/health":
            return {"ok": True, "version": __version__,
                    "ts": time.time()}
        if method == "GET" and path == "/mesh/nodes":
            return {"ok": True, "nodes": [
                n.to_dict() for n in self.transport.active_nodes()]}
        if method == "POST" and path == "/mesh/register":
            node = self.transport.register(
                str(body.get("name") or ""),
                platform=str(body.get("platform") or ""),
                capabilities=[str(c) for c in (body.get("capabilities") or [])],
                node_id=body.get("node_id"),
            )
            return {"ok": True, "node": node.to_dict()}
        if method == "POST" and path == "/mesh/heartbeat":
            node_id = str(body.get("node_id") or "")
            self.transport.heartbeat(node_id)
            return {"ok": True}
        if method == "POST" and path == "/mesh/dispatch":
            payload = body.get("payload") or {}
            if not isinstance(payload, dict):
                raise _HttpError(400, "bad_request",
                                 "dispatch payload must be an object")
            job_id = self.transport.dispatch(
                str(body.get("task_type") or ""),
                payload,
                origin_node=str(body.get("origin_node") or ""),
                target_node=body.get("target_node"),
                priority=int(body.get("priority") or 0),
            )
            return {"ok": True, "job_id": job_id}
        if method == "POST" and path == "/mesh/poll":
            tasks = self.transport.poll(
                str(body.get("node_id") or ""),
                batch=int(body.get("batch") or 5),
            )
            return {"ok": True,
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
        if method == "POST" and path == "/sync/push":
            records = body.get("records") or []
            if not isinstance(records, list):
                raise _HttpError(400, "bad_request",
                                 "sync records must be a list")
            docs = [SyncRecord.from_dict(d) for d in records]
            applied = self.sync_peer.push_records(docs)
            return {"ok": True, "applied": applied}
        if method == "GET" and path == "/sync/pull":
            limit = self._pull_limit(query)
            if "since_seq" in query:
                try:
                    seq = int(query["since_seq"][0])
                except (ValueError, IndexError) as exc:
                    raise _HttpError(400, "bad_request",
                                     f"bad since_seq: {exc}") from exc
                recs = self.sync_peer.store.list_since_seq(seq)[:limit]
            elif "since_ts" in query:
                try:
                    since = float(query["since_ts"][0])
                except (ValueError, IndexError) as exc:
                    raise _HttpError(400, "bad_request",
                                     f"bad since_ts: {exc}") from exc
                recs = self.sync_peer.store.list_changed_since(since)[:limit]
            else:
                raise _HttpError(400, "bad_request",
                                 "sync pull needs since_seq or since_ts")
            return {"ok": True, "records": [r.to_dict() for r in recs]}
        raise _HttpError(404, "not_found", f"no route {method} {path}")

    @staticmethod
    def _pull_limit(query: dict[str, list[str]]) -> int:
        if "limit" not in query:
            return _DEFAULT_PULL_LIMIT
        try:
            limit = int(query["limit"][0])
        except (ValueError, IndexError) as exc:
            raise _HttpError(400, "bad_request",
                             f"bad limit: {exc}") from exc
        return max(1, min(limit, PULL_MAX_LIMIT))

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
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path != "/health" and not server._authorized(self):
                    _send_json(self, 401, {
                        "ok": False, "code": "unauthorized",
                        "error": "unauthorized",
                        "detail": "missing or invalid bearer token"})
                    return
                try:
                    body = server._read_json(self) if method == "POST" else {}
                    payload = server._route(self, method, parsed.path,
                                            query, body)
                except _HttpError as exc:
                    _send_json(self, exc.status, {
                        "ok": False, "code": exc.code, "error": exc.code,
                        "detail": exc.detail})
                    return
                except NodeUnknown as exc:
                    _send_json(self, 404, {
                        "ok": False, "code": "node_unknown",
                        "error": "node_unknown", "detail": str(exc)})
                    return
                except ValueError as exc:
                    _send_json(self, 400, {
                        "ok": False, "code": "bad_request",
                        "error": "bad_request", "detail": str(exc)})
                    return
                except (MeshError, SyncError) as exc:
                    _log.exception("hub route %s %s failed", method,
                                   parsed.path)
                    _send_json(self, 500, {
                        "ok": False, "code": "hub_error",
                        "error": "hub_error", "detail": str(exc)})
                    return
                except Exception as exc:  # noqa: BLE001 - hub must stay up
                    _log.exception("hub route %s %s failed", method,
                                   parsed.path)
                    _send_json(self, 500, {
                        "ok": False, "code": "internal",
                        "error": "internal", "detail": str(exc)})
                    return
                _send_json(self, 200, payload)

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]  # port=0 → real port
        _emit("hub.started", {"host": self.host, "port": self.port,
                              "url": self.url})
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

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class _HttpError(Exception):
    """A route-level failure with an HTTP status and machine code."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def serve(
    db: Database,
    *,
    host: str = "127.0.0.1",
    port: int = 8861,
    token: str = "",
    device_id: str = "hub",
    background: bool = True,
) -> HubServer:
    """Start the hub server. Returns the server instance."""
    server = HubServer(db, host=host, port=port, token=token,
                       device_id=device_id)
    server.start(background=background)
    return server
