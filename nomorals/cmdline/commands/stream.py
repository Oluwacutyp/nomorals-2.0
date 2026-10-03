"""``nm stream`` — SSE live event stream: serve the timeline over HTTP.

The :mod:`nomorals.stream` package is a stdlib-only Server-Sent Events
server that streams Timeline events (``GET /stream``,
``GET /stream?topic=mission.*&since=<epoch>``) plus a ``GET /health``
probe. This command wires it into the CLI:

* ``nm stream`` (default) — start the server in the foreground on
  ``--host``/``--port``, streaming the timeline from this profile's
  database. Ctrl-C stops it cleanly.
* ``nm stream status`` — probe ``/health`` on ``--host``/``--port``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

DEFAULT_STREAM_PORT = 8899


def _timeline_factory(context: Any):
    """Zero-arg factory producing a fresh Timeline per request.

    Returns None (and prints a fail-fast error) when the context carries
    no database path.
    """
    from ...os.timeline import Timeline

    db = getattr(context, "db", None)
    db_path = getattr(db, "path", None)
    if not db_path:
        print(
            "stream serve: context has no database path "
            "(need context.db.path) — cannot stream the timeline",
            file=sys.stderr,
        )
        return None

    def factory():
        return Timeline(db_path)

    return factory


def _cmd_stream_serve(args: argparse.Namespace, context: Any) -> int:
    """Start the SSE server in the foreground (blocks until Ctrl-C)."""
    from ...stream import StreamServer

    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or DEFAULT_STREAM_PORT
    factory = _timeline_factory(context)
    if factory is None:
        return 2

    server = StreamServer(factory, host=host, port=port)
    print(f"SSE stream: {server.url}/stream (topic/since query params supported)")
    print(f"health:     {server.url}/health")
    print("Ctrl-C to stop.")
    try:
        server.start(background=False)
    except KeyboardInterrupt:  # noqa: E103, E106 - deliberate top-level shutdown hook
        print("\nstream stopped", file=sys.stderr)
    finally:
        try:
            server.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown after Ctrl-C
            pass
    return 0


def _cmd_stream_status(args: argparse.Namespace, context: Any) -> int:
    """Probe the SSE server's /health endpoint."""
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or DEFAULT_STREAM_PORT
    url = f"http://{host}:{port}/health"
    as_json = bool(getattr(args, "json", False))
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
    except Exception as exc:  # noqa: BLE001 - unreachable server is the expected case
        print(f"stream: no server at {url} ({exc})", file=sys.stderr)
        return 1
    ok = bool(payload.get("ok"))
    if as_json:
        print(
            json.dumps(
                {
                    "host": host,
                    "port": port,
                    "running": ok,
                    "health": payload,
                },
                indent=2,
            )
        )
    else:
        state = "up" if ok else "unhealthy"
        version = payload.get("version", "?")
        print(f"stream server at {url}: {state} (version {version})")
    return 0 if ok else 1


def _cmd_stream(args: argparse.Namespace, context: Any) -> int:
    """Route ``nm stream [serve|status] [--host H] [--port P]``."""
    action = getattr(args, "action", None) or "serve"
    if action == "serve":
        return _cmd_stream_serve(args, context)
    if action == "status":
        return _cmd_stream_status(args, context)
    print(f"unknown stream action: {action}", file=sys.stderr)
    return 2
