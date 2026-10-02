"""$PROJECT_NAME -- stdlib-only REST API (no third-party dependencies).

A small JSON REST service with a thread-safe in-memory store:

    GET    /                API index (project name + endpoint list)
    GET    /api/health        liveness probe
    GET    /api/items         list all items
    POST   /api/items         create an item (JSON {"name": "..."})
    GET    /api/items/<id>    fetch one item
    PATCH  /api/items/<id>    partial update ({"name": ..., "done": ...})
    DELETE /api/items/<id>    delete an item

Errors are JSON envelopes: {"error": "..."}. Unknown paths 404,
wrong methods 405 with an Allow header.

Run:
    python run.py [--host 127.0.0.1] [--port 8000]
    PORT=9000 python run.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROJECT = "$PROJECT_NAME"
VERSION = "0.1.0"

_ITEM_RE = re.compile(r"^/api/items/(\d+)$")


class Store:
    """Thread-safe in-memory item store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[int, dict] = {}
        self._next_id = 1

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(i) for i in sorted(self._items.values(),
                                            key=lambda x: x["id"])]

    def create(self, name: str) -> dict:
        with self._lock:
            item = {"id": self._next_id, "name": name, "done": False}
            self._items[self._next_id] = item
            self._next_id += 1
            return dict(item)

    def get(self, item_id: int) -> dict | None:
        with self._lock:
            item = self._items.get(item_id)
            return dict(item) if item is not None else None

    def update(self, item_id: int, fields: dict) -> dict | None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return None
            if "name" in fields and str(fields["name"]).strip():
                item["name"] = str(fields["name"])
            if "done" in fields:
                item["done"] = bool(fields["done"])
            return dict(item)

    def delete(self, item_id: int) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None


STORE = Store()


class Handler(BaseHTTPRequestHandler):
    server_version = f"{PROJECT}/{VERSION}"

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object,
              extra: dict[str, str] | None = None) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json", extra)

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    def _read_json(self) -> tuple[dict | None, str | None]:
        """Return (payload, error).  error is None on success."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(max(0, length))
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, "invalid JSON body"
        if not isinstance(payload, dict):
            return None, "JSON body must be an object"
        return payload, None

    # -- routing ----------------------------------------------------------
    def _route(self) -> None:
        path = urlparse(self.path).path
        method = self.command

        if path == "/":
            if method != "GET":
                self._json(405, {"error": "method not allowed"},
                           {"Allow": "GET"})
                return
            self._json(200, {
                "project": PROJECT,
                "version": VERSION,
                "endpoints": [
                    "GET /api/health",
                    "GET /api/items",
                    "POST /api/items",
                    "GET /api/items/<id>",
                    "PATCH /api/items/<id>",
                    "DELETE /api/items/<id>",
                ],
            })
            return

        if path == "/api/health":
            if method != "GET":
                self._json(405, {"error": "method not allowed"},
                           {"Allow": "GET"})
                return
            self._json(200, {"status": "ok", "project": PROJECT,
                             "version": VERSION, "ts": time.time()})
            return

        if path == "/api/items":
            if method == "GET":
                self._json(200, {"items": STORE.list()})
            elif method == "POST":
                payload, err = self._read_json()
                if err:
                    self._error(400, err)
                    return
                name = str(payload.get("name", "")).strip()
                if not name:
                    self._error(400, "field 'name' is required and must be "
                                    "non-empty")
                    return
                self._json(201, STORE.create(name))
            else:
                self._json(405, {"error": "method not allowed"},
                           {"Allow": "GET, POST"})
            return

        match = _ITEM_RE.match(path)
        if match:
            item_id = int(match.group(1))
            if method == "GET":
                item = STORE.get(item_id)
                if item is None:
                    self._error(404, f"no item {item_id}")
                else:
                    self._json(200, item)
            elif method == "PATCH":
                payload, err = self._read_json()
                if err:
                    self._error(400, err)
                    return
                item = STORE.update(item_id, payload)
                if item is None:
                    self._error(404, f"no item {item_id}")
                else:
                    self._json(200, item)
            elif method == "DELETE":
                if STORE.delete(item_id):
                    self._json(200, {"deleted": item_id})
                else:
                    self._error(404, f"no item {item_id}")
            else:
                self._json(405, {"error": "method not allowed"},
                           {"Allow": "GET, PATCH, DELETE"})
            return

        self._error(404, f"not found: {path}")

    def do_GET(self) -> None:  # noqa: N802
        self._route()

    def do_POST(self) -> None:  # noqa: N802
        self._route()

    def do_PATCH(self) -> None:  # noqa: N802
        self._route()

    def do_DELETE(self) -> None:  # noqa: N802
        self._route()

    def do_PUT(self) -> None:  # noqa: N802
        self._route()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        pass  # keep test/console output clean


def make_server(host: str = "127.0.0.1",
                port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="$PROJECT_NAME",
        description=f"{PROJECT} -- stdlib-only REST API.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host",
                        default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"{PROJECT} REST API listening on "
          f"http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # noqa: E103, E106 - deliberate shutdown hook
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
