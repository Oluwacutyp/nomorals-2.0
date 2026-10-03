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

from ..core.logging_setup import get_logger
from ..version import __version__

__all__ = ["StreamServer", "serve"]

_log = get_logger(__name__)

HEARTBEAT_INTERVAL = 15.0  # seconds between :ping comments
POLL_INTERVAL = 1.0        # seconds between Timeline polls


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
                raw = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)

            def _sse(self, query: dict[str, list[str]]) -> None:
                try:
                    since = float(query["since"][0]) if "since" in query else 0.0
                except (ValueError, IndexError) as exc:
                    self._json(400, {"ok": False, "error": f"bad since: {exc}"})
                    return
                topic = query["topic"][0] if "topic" in query else None

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                cursor = since
                last_beat = time.time()
                _log.info("stream subscriber connected (topic=%s)", topic)
                try:
                    while True:
                        timeline = server.timeline_factory()
                        try:
                            events = timeline.query(
                                since=cursor, topic=topic, limit=100
                            )
                        finally:
                            close = getattr(timeline, "close", None)
                            if callable(close):
                                close()
                        # query() is newest-first; emit oldest-first.
                        for ev in reversed(events):
                            ts = float(ev.get("ts", 0) or 0)
                            if ts > cursor:
                                cursor = ts
                            data = json.dumps(ev, default=str)
                            chunk = (
                                f"event: timeline\n"
                                f"id: {cursor}\n"
                                f"data: {data}\n\n"
                            ).encode("utf-8")
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        now = time.time()
                        if now - last_beat >= HEARTBEAT_INTERVAL:
                            self.wfile.write(b":ping\n\n")
                            self.wfile.flush()
                            last_beat = now
                        time.sleep(POLL_INTERVAL)
                except (ConnectionResetError, BrokenPipeError):
                    _log.info("stream subscriber disconnected")
                except Exception:  # noqa: BLE001 - log and close
                    _log.exception("stream handler error")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        # Resolve the real port when port=0 was requested.
        self.port = self._server.server_address[1]
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
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

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
