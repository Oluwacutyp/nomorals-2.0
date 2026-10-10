"""SSE streaming server: live Timeline events over HTTP.

Stdlib only (``http.server.ThreadingHTTPServer``). Each subscriber holds
one thread; a shared :class:`~nomorals.stream.hub.EventHub` polls the
Timeline once and fans new events out to per-subscriber queues (the hub
pattern from the Mercure protocol) instead of every connection polling
the database itself.

``Last-Event-ID`` resume (with ``?since=`` / ``?lastEventId=`` as
fallbacks), a leading ``retry:`` hint, client-invisible ``: ping``
heartbeats, CORS, a send timeout for hung clients, and a graceful
``event: shutdown`` on server stop are all built in.

Fail fast: bad query params → 400, too many subscribers → 503. The stream
never silently drops the cursor — resume tokens are echoed back so
clients can resume.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..version import __version__
from ._poll import PAGE_LIMIT, POLL_INTERVAL as _POLL_INTERVAL
from .errors import StreamClosed, SubscriberLimitExceeded
from .hub import EventHub
from .sse import ServerSentEvent, format_sse

__all__ = [
    "DEFAULT_RETRY_MS",
    "HEARTBEAT_INTERVAL",
    "PAGE_LIMIT",
    "POLL_INTERVAL",
    "StreamServer",
    "emit_sse",
    "serve",
]

_log = get_logger(__name__)

HEARTBEAT_INTERVAL = 15.0  # seconds of idle before a `: ping` comment
POLL_INTERVAL = _POLL_INTERVAL  # seconds between hub Timeline polls

#: Reconnect delay advertised to EventSource clients at stream start.
DEFAULT_RETRY_MS = 3000

#: Give up on a client whose socket won't take bytes within this long.
DEFAULT_SEND_TIMEOUT = 10.0


def _send_json(handler: Any, status: int, payload: dict,
               extra_headers: dict[str, str] | None = None) -> None:
    """Write a small JSON response through a ``BaseHTTPRequestHandler``."""
    raw = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    for key, value in (extra_headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(raw)


def _cors_headers(cors: str | None) -> dict[str, str]:
    if not cors:
        return {}
    return {"Access-Control-Allow-Origin": cors, "Vary": "Origin"}


def _parse_since(query: dict[str, list[str]]) -> float:
    try:
        return float(query["since"][0]) if "since" in query else 0.0
    except (ValueError, IndexError) as exc:
        raise ValueError(f"bad since: {exc}") from exc


def _parse_last_event_id(
    handler: Any,
    query: dict[str, list[str]],
) -> int | None:
    """Resume token: ``Last-Event-ID`` header first (the SSE standard),
    ``?lastEventId=`` query param as fallback (non-EventSource clients)."""
    raw: str | None = None
    headers = getattr(handler, "headers", None)
    if headers is not None:
        raw = headers.get("Last-Event-ID")
    if raw is None and "lastEventId" in query:
        try:
            raw = query["lastEventId"][0]
        except IndexError:
            raw = None
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return None


def _write_frame(handler: Any, frame: bytes) -> None:
    handler.wfile.write(frame)
    handler.wfile.flush()


def emit_sse(
    handler: Any,
    timeline_factory: Callable[[], Any],
    query: dict[str, list[str]],
    *,
    hub: EventHub | None = None,
    retry_ms: int = DEFAULT_RETRY_MS,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
    cors: str | None = "*",
    send_timeout: float = DEFAULT_SEND_TIMEOUT,
) -> None:
    """Push Timeline events to ``handler`` as Server-Sent Events.

    ``handler`` is any ``BaseHTTPRequestHandler`` — this is the shared SSE
    loop used both by :class:`StreamServer` (its own port, shared hub) and
    by the main API server's ``GET /stream`` route (an ephemeral hub per
    connection, no separate port).

    ``query`` maps names to value lists (as ``parse_qs`` returns):
    ``since`` is an optional epoch cursor, ``topic`` an optional
    comma-separated topic-glob filter, ``lastEventId`` an optional hub
    sequence number. The ``Last-Event-ID`` request header takes precedence
    over the query fallbacks. Bad ``since`` → 400 JSON, fail fast; a full
    hub → 503 JSON.

    The call returns when the subscriber disconnects or the hub stops; the
    timeline instance is created fresh per poll by ``timeline_factory``
    and closed after each poll.
    """
    try:
        since = _parse_since(query)
    except ValueError as exc:
        _send_json(handler, 400, {"ok": False, "error": str(exc)})
        return
    topic = query["topic"][0] if "topic" in query else None
    last_event_id = _parse_last_event_id(handler, query)

    own_hub = hub is None
    if own_hub:
        hub = EventHub(timeline_factory)
        hub.start()
    assert hub is not None
    try:
        sub = hub.subscribe(
            topic, since=since, last_event_id=last_event_id)
    except SubscriberLimitExceeded as exc:
        _send_json(handler, 503, {"ok": False, "error": str(exc)},
                   extra_headers={"Retry-After": "5"})
        if own_hub:
            hub.stop()
        return
    except StreamClosed:
        _send_json(handler, 503, {"ok": False, "error": "stream is stopped"})
        if own_hub:
            hub.stop()
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    # no-store (never cache a live stream) + no-transform (proxies must
    # not recompress/re-chunk it — gzipping re-batches events).
    handler.send_header("Cache-Control", "no-store, no-transform")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")  # nginx: don't buffer
    for key, value in _cors_headers(cors).items():
        handler.send_header(key, value)
    handler.end_headers()

    # A hung client must not wedge this thread: bound every write.
    # (The previous timeout is restored afterwards — on the shared API
    # server path this socket may be reused for later requests.)
    connection = getattr(handler, "connection", None)
    prev_timeout: float | None = None
    if connection is not None:
        try:
            prev_timeout = connection.gettimeout()
            connection.settimeout(send_timeout)
        except OSError:  # pragma: no cover - exotic sockets
            _log.debug("stream: could not set send timeout", exc_info=True)
            connection = None

    resume_from = last_event_id if last_event_id is not None else since
    _log.info("stream subscriber connected (topic=%s resume_from=%r)",
              topic, resume_from)
    _emit("stream.serving", {
        "subscription_id": sub.id,
        "topic": topic,
        "resume_from": resume_from,
    })
    try:
        # Tell EventSource clients how fast to reconnect, then say hello.
        _write_frame(handler, format_sse(retry=retry_ms))
        _write_frame(handler, ServerSentEvent(
            event="ready",
            data=json.dumps({
                "server": f"NoMoralsStream/{__version__}",
                "hub_epoch": hub.epoch,
                "topics": sub.topics,
                "retry_ms": retry_ms,
                "resume_from": resume_from,
            }),
        ).encode())
        reported_dropped = 0
        while True:
            try:
                seq, event = sub.get(timeout=heartbeat_interval)
            except queue.Empty:
                # Client-invisible heartbeat: keeps proxies/LBs from
                # killing idle connections (their read timeouts only
                # reset on application data).
                _write_frame(handler, format_sse(comment="ping"))
                continue
            except StreamClosed:
                # Graceful drain: tell the client to reconnect cleanly
                # instead of seeing a bare connection drop.
                try:
                    _write_frame(handler, ServerSentEvent(
                        event="shutdown",
                        data=json.dumps({"reason": "server stopping"}),
                        retry=retry_ms,
                    ).encode())
                except OSError:
                    pass
                break
            frame = ServerSentEvent(
                event="timeline",
                id=str(seq),
                data=json.dumps(event, default=str),
            ).encode()
            _write_frame(handler, frame)
            if sub.dropped > reported_dropped:
                # Backpressure kicked in: tell the client it missed events
                # and how to re-sync, instead of staying silently gappy.
                reported_dropped = sub.dropped
                _write_frame(handler, ServerSentEvent(
                    event="stream-warning",
                    data=json.dumps({
                        "dropped": reported_dropped,
                        "hint": "slow consumer: reconnect with "
                                "?since=<epoch> to re-sync",
                    }),
                ).encode())
    except (ConnectionResetError, BrokenPipeError, socket.timeout):
        _log.info("stream subscriber disconnected")
    except Exception:  # noqa: BLE001 - log and close
        _log.exception("stream handler error")
    finally:
        if connection is not None:
            try:
                connection.settimeout(prev_timeout)
            except OSError:  # pragma: no cover - socket already gone
                pass
        sub.close()
        if own_hub:
            hub.stop()
        _emit("stream.served_done", {"subscription_id": sub.id})


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the stream server (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


class _TunedHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with saner socket/threads defaults for SSE."""

    daemon_threads = True  # don't let stray subscribers block shutdown
    allow_reuse_address = True

    def get_request(self):  # noqa: N802 - socketserver naming
        request, client_address = super().get_request()
        try:
            # Small frames, latency matters more than throughput.
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Let the kernel notice dead peers even between heartbeats.
            request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:  # pragma: no cover - exotic sockets
            _log.debug("stream: socket tuning failed", exc_info=True)
        return request, client_address


class StreamServer:
    """SSE server streaming Timeline events.

    ``timeline_factory`` is a zero-arg callable returning a Timeline-like
    with ``query(since=..., topic=..., limit=..., until=...)``. A factory
    (not an instance) keeps the server testable and avoids sharing
    connections across threads. All connections share one
    :class:`EventHub`, so the Timeline is polled once no matter how many
    dashboards are attached.
    """

    def __init__(
        self,
        timeline_factory: Callable[[], Any],
        *,
        host: str = "127.0.0.1",
        port: int = 8899,
        poll_interval: float = POLL_INTERVAL,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        retry_ms: int = DEFAULT_RETRY_MS,
        max_subscribers: int = 1024,
        queue_size: int = 512,
        ring_size: int = 10_000,
        cors: str | None = "*",
        send_timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> None:
        if timeline_factory is None:
            raise ValueError("timeline_factory is required")
        self.timeline_factory = timeline_factory
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.retry_ms = retry_ms
        self.cors = cors
        self.send_timeout = send_timeout
        self.hub = EventHub(
            timeline_factory,
            poll_interval=poll_interval,
            max_subscribers=max_subscribers,
            default_queue_size=queue_size,
            ring_size=ring_size,
        )
        self._server: _TunedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None

    def start(self, background: bool = True) -> None:
        if self._server is not None:
            raise RuntimeError("server already started")
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = f"NoMoralsStream/{__version__}"

            def log_message(self, fmt: str, *args: Any) -> None:
                _log.debug("stream %s - %s", self.address_string(), fmt % args)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._json(200, server.health())
                elif parsed.path == "/stream":
                    self._sse(parse_qs(parsed.query))
                else:
                    self._json(404, {"ok": False, "error": "not found"})

            def do_OPTIONS(self) -> None:  # noqa: N802
                # CORS preflight for dashboard clients on other origins.
                self.send_response(204)
                for key, value in _cors_headers(server.cors).items():
                    self.send_header(key, value)
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Last-Event-ID, Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _json(self, status: int, payload: dict) -> None:
                _send_json(self, status, payload,
                           extra_headers=_cors_headers(server.cors))

            def _sse(self, query: dict[str, list[str]]) -> None:
                emit_sse(
                    self, server.timeline_factory, query, hub=server.hub,
                    retry_ms=server.retry_ms,
                    heartbeat_interval=server.heartbeat_interval,
                    cors=server.cors, send_timeout=server.send_timeout)

        self.hub.start()
        try:
            self._server = _TunedHTTPServer((self.host, self.port), Handler)
        except Exception:
            self.hub.stop()
            raise
        # Resolve the real port when port=0 was requested.
        self.port = self._server.server_address[1]
        self._started_at = time.time()
        _emit("stream.started", {
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "background": background,
        })
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="stream-server",
                daemon=True,
            )
            self._thread.start()
            _log.info("stream server on %s:%d (background)", self.host, self.port)
        else:
            _log.info("stream server on %s:%d (foreground)", self.host, self.port)
            self._server.serve_forever()

    def stop(self) -> None:
        was_running = self._server is not None or self._thread is not None
        if self._server is not None:
            # Stop accepting, finish in-flight requests, then close.
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Closing the hub wakes every subscriber loop with a farewell
        # `event: shutdown` so EventSource clients reconnect cleanly.
        self.hub.stop()
        if was_running:
            _emit("stream.stopped", {
                "host": self.host,
                "port": self.port,
                "url": self.url,
            })

    def health(self) -> dict[str, Any]:
        """Enriched health payload: liveness plus stream vitals."""
        hub_stats = self.hub.stats()
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": round(hub_stats["uptime_s"], 3),
            "subscribers": hub_stats["subscribers"],
            "max_subscribers": hub_stats["max_subscribers"],
            "events_emitted": hub_stats["events_emitted"],
            "events_dropped": hub_stats["events_dropped"],
            "ring_size": hub_stats["ring_size"],
        }

    def stats(self) -> dict[str, Any]:
        """Full server stats (hub stats plus bind info)."""
        stats = self.hub.stats()
        stats.update({
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "running": self._server is not None,
        })
        return stats

    @property
    def subscriber_count(self) -> int:
        return self.hub.subscriber_count

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> StreamServer:
        self.start(background=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve(
    timeline_factory: Callable[[], Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8899,
    background: bool = True,
    **kwargs: Any,
) -> StreamServer:
    """Start the stream server. Returns the server instance.

    Extra keyword arguments are passed to :class:`StreamServer`
    (``poll_interval``, ``heartbeat_interval``, ``retry_ms``,
    ``max_subscribers``, ``queue_size``, ``ring_size``, ``cors``,
    ``send_timeout``).
    """
    server = StreamServer(timeline_factory, host=host, port=port, **kwargs)
    server.start(background=background)
    return server
