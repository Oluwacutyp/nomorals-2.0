"""$PROJECT_NAME -- stdlib-only HTTP app (no third-party dependencies).

Routes:
    GET  /            HTML landing page
    GET  /api/health  JSON health check
    POST /api/echo    echo the JSON body back with a server timestamp

Run:
    python run.py [--port 8000]
    PORT=9000 python run.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROJECT = "$PROJECT_NAME"
VERSION = "0.1.0"


def _html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{PROJECT}</title></head>
<body style="font-family: sans-serif; max-width: 640px; margin: 3rem auto;">
  <h1>{PROJECT}</h1>
  <p>A tiny stdlib-only web app. It works.</p>
  <ul>
    <li><a href="/api/health">/api/health</a> -- JSON health check</li>
    <li><code>POST /api/echo</code> -- JSON echo endpoint</li>
  </ul>
  <p><button id="b">ping /api/health</button> <span id="out"></span></p>
  <script>
    document.getElementById('b').onclick = async () => {{
      const r = await fetch('/api/health');
      document.getElementById('out').textContent = await r.text();
    }};
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = f"{PROJECT}/{VERSION}"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, _html().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"status": "ok", "project": PROJECT,
                             "version": VERSION, "ts": time.time()})
        else:
            self._json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/echo":
            self._json(404, {"error": "not found", "path": path})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(max(0, length))
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "invalid JSON"})
            return
        self._json(200, {"echo": payload, "project": PROJECT, "ts": time.time()})

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        pass  # keep test/console output clean


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="$PROJECT_NAME", description=f"{PROJECT} web server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"{PROJECT} listening on http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # noqa: E103, E106 - deliberate shutdown hook
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
