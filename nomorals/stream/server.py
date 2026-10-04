"""SSE streaming server: live Timeline events over HTTP.

Stdlib only (``http.server.ThreadingHTTPServer``). Each subscriber holds
one thread; the server polls the Timeline and pushes new events as
Server-Sent Events. Heartbeat comments keep idle connections alive
through proxies.

Fail fast: bad query params → 400. The stream never silently drops the
cursor — ``since`` is echoed back so clients can resume.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..version import __version__

__all__ = ["StreamServer", "emit_sse", "serve"]

_log = get_logger(__name__)


def _send_json(handler: Any, status: int, payload: dict) -> None:
    """Write a small JSON response through a ``BaseHTTPRequestHandler``."""
    raw = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def _ts_of(ev: dict[str, Any]) -> float:
    return float(ev.get("ts", 0) or 0)


def _id_of(ev: dict[str, Any]) -> str:
    return str(ev.get("event_id") or "")


def _emit_page(
    handler: Any,
    events: list[dict[str, Any]],
    cursor: float,
    seen_at_cursor: set[str],
) -> tuple[float, set[str]]:
    """Emit one newest-first page oldest-first. Returns (cursor, seen).

    ``Timeline.query(since=)`` is inclusive (``ts >= cursor``) while the
    cursor only advances on strictly newer timestamps — without identity
    tracking, an event whose ``ts`` equals the cursor is re-emitted on
    every poll. ``seen_at_cursor`` holds the event_ids already emitted at
    this cursor, so a connection is exactly-once per event while resume
    across reconnects stays at-least-once (the boundary event is
    delivered again, once, to the new connection).
    """
    # query() is newest-first; emit oldest-first.
    for ev in reversed(events):
        ts = _ts_of(ev)
        eid = _id_of(ev)
        if ts > cursor:
            cursor = ts
            seen_at_cursor = {eid} if eid else set()
        elif ts == cursor and eid and eid not in seen_at_cursor:
            seen_at_cursor.add(eid)
        else:
            continue  # duplicate of an already-emitted event — skip
        data = json.dumps(ev, default=str)
        chunk = (
            f"event: timeline\n"
            f"id: {cursor}\n"
            f"data: {data}\n\n"
        ).encode("utf-8")
        handler.wfile.write(chunk)
        handler.wfile.flush()
    return cursor, seen_at_cursor


def _query_page(
    timeline_factory: Callable[[], Any],
    topic: str | None,
    since: float,
    until: float | None,
) -> list[dict[str, Any]]:
    """One newest-first page; the timeline instance is never shared."""
    timeline = timeline_factory()
    try:
        kwargs: dict[str, Any] = {"since": since, "topic": topic,
                                  "limit": PAGE_LIMIT}
        if until is not None:
            kwargs["until"] = until
        return timeline.query(**kwargs)
    finally:
        close = getattr(timeline, "close", None)
        if callable(close):
            close()


def _drain_backlog(
    handler: Any,
    timeline_factory: Callable[[], Any],
    topic: str | None,
    cursor: float,
    seen_at_cursor: set[str],
    *,
    _floor: float | None = None,
    _ceiling: float | None = None,
    _depth: int = 0,
) -> tuple[float, set[str]]:
    """Emit every event with ``_floor <= ts`` (``ts <= _ceiling`` when set).

    A full page may hide older events behind it (the query is
    newest-first): those are drained first via a narrowed ``until``
    bound, so a burst bigger than one page cannot silently drop its
    tail. The inclusive-boundary overlap between the narrowed slice and
    its parent page is harmless — the event_id dedup in
    :func:`_emit_page` skips re-emission.

    Recursion strictly narrows ``_ceiling`` each level, so the only
    non-shrinking shape is a full page of identical timestamps; that —
    and any backlog deeper than ``_MAX_DRAIN_DEPTH`` pages — is emitted
    once and logged loudly instead of looping forever or dying
    silently.
    """
    floor = cursor if _floor is None else _floor
    events = _query_page(timeline_factory, topic, floor, _ceiling)
    if len(events) < PAGE_LIMIT:
        return _emit_page(handler, events, cursor, seen_at_cursor)
    oldest = min(_ts_of(e) for e in events)
    if _depth >= _MAX_DRAIN_DEPTH or (
            _ceiling is not None and oldest >= _ceiling):
        _log.warning(
            "stream: backlog of >%d events at/above ts %r truncated for "
            "this poll (depth=%d)", PAGE_LIMIT, oldest, _depth)
        return _emit_page(handler, events, cursor, seen_at_cursor)
    # Drain the older slice first, then this page.
    cursor, seen_at_cursor = _drain_backlog(
        handler, timeline_factory, topic, cursor, seen_at_cursor,
        _floor=floor, _ceiling=oldest, _depth=_depth + 1)
    return _emit_page(handler, events, cursor, seen_at_cursor)


def emit_sse(
    handler: Any,
    timeline_factory: Callable[[], Any],
    query: dict[str, list[str]],
) -> None:
    """Push Timeline events to ``handler`` as Server-Sent Events.

    ``handler`` is any ``BaseHTTPRequestHandler`` — this is the shared SSE
    loop used both by :class:`StreamServer` (its own port) and by the main
    API server's ``GET /stream`` route (no separate port).

    ``query`` maps names to value lists (as ``parse_qs`` returns); ``since``
    is an optional epoch cursor, ``topic`` an optional topic filter. Bad
    ``since`` → 400 JSON, fail fast. The call returns when the subscriber
    disconnects; the timeline instance is created fresh per poll by
    ``timeline_factory`` and closed after each poll.
    """
    try:
        since = float(query["since"][0]) if "since" in query else 0.0
    except (ValueError, IndexError) as exc:
        _send_json(handler, 400, {"ok": False, "error": f"bad since: {exc}"})
        return
    topic = query["topic"][0] if "topic" in query else None

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

    cursor = since
    seen_at_cursor: set[str] = set()
    last_beat = time.time()
    _log.info("stream subscriber connected (topic=%s)", topic)
    try:
        while True:
            cursor, seen_at_cursor = _drain_backlog(
                handler, timeline_factory, topic, cursor, seen_at_cursor)
            now = time.time()
            if now - last_beat >= HEARTBEAT_INTERVAL:
                handler.wfile.write(b":ping\n\n")
                handler.wfile.flush()
                last_beat = now
            time.sleep(POLL_INTERVAL)
    except (ConnectionResetError, BrokenPipeError):
        _log.info("stream subscriber disconnected")
    except Exception:  # noqa: BLE001 - log and close
        _log.exception("stream handler error")


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the stream server (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

HEARTBEAT_INTERVAL = 15.0  # seconds between :ping comments
POLL_INTERVAL = 1.0        # seconds between Timeline polls

#: Events per backlog-drain page. A poll that finds a full page drains the
#: older slice behind it (see _drain_backlog) instead of silently dropping
#: the tail the way a single fixed page would.
PAGE_LIMIT = 1000

#: Backlog-drain recursion cap: each level consumes at least one full
#: page, so this bounds the drain at ~50k events per poll — beyond that
#: the subscriber gets the newest slice and a loud warning, never a
#: RecursionError or a silent drop.
_MAX_DRAIN_DEPTH = 50


class StreamServer:
    """SSE server streaming Timeline events.

    ``timeline_factory`` is a zero-arg callable returning a Timeline-like
    with ``query(since=..., topic=..., limit=...)``. A factory (not an
    instance) keeps the server testable and avoids sharing connections
    across threads.
    """

    def __init__(
        self,
        timeline_factory: Callable[[], Any],
        *,
        host: str = "127.0.0.1",
        port: int = 8899,
    ) -> None:
        if timeline_factory is None:
            raise ValueError("timeline_factory is required")
        self.timeline_factory = timeline_factory
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

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
                    self._json(200, {"ok": True, "version": __version__})
                elif parsed.path == "/stream":
                    self._sse(parse_qs(parsed.query))
                else:
                    self._json(404, {"ok": False, "error": "not found"})

            def _json(self, status: int, payload: dict) -> None:
                _send_json(self, status, payload)

            def _sse(self, query: dict[str, list[str]]) -> None:
                emit_sse(self, server.timeline_factory, query)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        # Resolve the real port when port=0 was requested.
        self.port = self._server.server_address[1]
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
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if was_running:
            _emit("stream.stopped", {
                "host": self.host,
                "port": self.port,
                "url": self.url,
            })

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def serve(
    timeline_factory: Callable[[], Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8899,
    background: bool = True,
) -> StreamServer:
    """Start the stream server. Returns the server instance."""
    server = StreamServer(timeline_factory, host=host, port=port)
    server.start(background=background)
    return server
