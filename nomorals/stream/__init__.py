"""Streaming API: Server-Sent Events for live agent state.

Layer L5. A stdlib-only SSE server (no new dependencies) that streams
Timeline events to dashboards and future mobile clients. This is the
backend the web dashboard / Android app will consume — this package is
the API, not the clients.

One :class:`EventHub` polls the Timeline once and fans events out to
all subscribers (the Mercure hub pattern); each event carries a
hub-monotonic sequence number as its SSE ``id:`` so ``Last-Event-ID``
reconnects replay from a retained ring buffer.

Endpoints:
  GET /health          → {"ok": true, "version": ..., "subscribers": ...}
  GET /stream          → text/event-stream of Timeline events
  GET /stream?topic=mission.*&since=<epoch> → filtered stream
  GET /stream (Last-Event-ID: <seq>)        → resume after <seq>
"""

from __future__ import annotations

from .errors import StreamClosed, StreamError, SubscriberLimitExceeded
from .hub import EventHub, Subscription
from .server import StreamServer, emit_sse, serve
from .sse import ServerSentEvent, format_sse

__all__ = [
    "EventHub",
    "ServerSentEvent",
    "StreamClosed",
    "StreamError",
    "StreamServer",
    "SubscriberLimitExceeded",
    "Subscription",
    "emit_sse",
    "format_sse",
    "serve",
]
