"""Streaming API: Server-Sent Events for live agent state.

Layer L5. A stdlib-only SSE server (no new dependencies) that streams
Timeline events to dashboards and future mobile clients. This is the
backend the web dashboard / Android app will consume — this package is
the API, not the clients.

Endpoints:
  GET /health          → {"ok": true, "version": ...}
  GET /stream          → text/event-stream of Timeline events
  GET /stream?topic=mission.*&since=<epoch> → filtered stream
"""

from __future__ import annotations

from .errors import StreamError
from .server import StreamServer, serve

__all__ = ["StreamError", "StreamServer", "serve"]
