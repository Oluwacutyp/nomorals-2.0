"""HTTP API — stdlib only, so the service runs anywhere Python does.

Deliberately not FastAPI. A dependency-free server means the whole system starts
on a phone with nothing installed, and the API surface here is small enough that
a hand-rolled router is clearer than a framework would be.

Threaded, because agent runs are long and a single-threaded server would block
health checks behind them. Bearer auth when a token is configured; on a LAN that
token is the only thing between an open port and your memory store.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..core.errors import NoMoralsError, classify
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet
from ..version import __version__

__all__ = ["APIServer", "serve"]

_log = get_logger(__name__)

_ROUTE = re.compile(r"^/[a-z0-9_/-]+$")


class APIServer:
    """Tiny router plus a threaded HTTP server."""

    def __init__(self, context: Any, *, token: str = "") -> None:
        self.context = context
        self.token = token
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        self._register_defaults()

    def route(self, method: str, path: str):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[(method.upper(), path)] = fn
            return fn

        return decorator

    def dispatch(self, method: str, path: str, body: dict[str, Any], query: dict[str, str]) -> tuple[int, Any]:
        handler = self._routes.get((method.upper(), path))
        if handler is None:
            return 404, {"error": f"no route {method} {path}"}
        try:
            return 200, handler(body, query)
        except NoMoralsError as exc:
            outcome = classify(exc)
            return (429 if outcome.retryable else 400), {"error": outcome.message, "kind": type(exc).__name__}
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to a client
            _log.exception("api error on %s %s", method, path)
            return 500, {"error": type(exc).__name__}

    # ── routes ───────────────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        context = self.context

        @self.route("GET", "/health")
        def health(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            db = getattr(context, "db", None)
            schema = 0
            if db is not None:
                # The migration table is the source of truth; PRAGMA user_version
                # is never written, so reading it reported 0 forever.
                from ..storage.schema import MigrationRunner

                schema = MigrationRunner(db).current_version()
            return {"version": __version__, "ok": True, "schema_version": schema}

        @self.route("GET", "/models")
        def models(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..llm.registry import ModelRegistry

            registry = ModelRegistry(context.db)
            return {"stats": registry.stats(),
                    "models": [r.__dict__ for r in registry.list(limit=100)]}

        @self.route("GET", "/tools")
        def tools(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return {"tools": context.tools.register_builtins().schemas()}

        @self.route("POST", "/tools/call")
        def call_tool(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            name = str(body.get("name") or "")
            arguments = body.get("arguments") or {}
            if not isinstance(arguments, dict):
                return {"ok": False, "error": "arguments must be an object"}
            outcome = context.tools.call(
                name,
                actor=str(body.get("actor") or "api"),
                capabilities=CapabilitySet.all(),
                confirmation=str(body.get("confirmation") or "") or None,
                **arguments,
            )
            if outcome.ok:
                return {"ok": True, "result": outcome.value}
            return {"ok": False, "error": outcome.error.message if outcome.error else "failed"}

        @self.route("POST", "/chat")
        def chat(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..llm.base import Message, SamplingParams

            raw = body.get("messages") or [{"role": "user", "content": str(body.get("prompt") or "")}]
            messages = [
                Message(role=str(m.get("role") or "user"), content=str(m.get("content") or ""))
                for m in raw
                if isinstance(m, dict)
            ]
            params = SamplingParams(
                temperature=float(body.get("temperature", 0.7)),
                max_tokens=int(body.get("max_tokens", 1024)),
            )
            response = context.router.chat(messages, params)
            return response.to_dict()

        @self.route("POST", "/memory/remember")
        def remember(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            record_id = context.memory.remember(
                str(body.get("content") or ""),
                kind=str(body.get("kind") or "episode"),
                source=str(body.get("source") or "api"),
                importance=float(body.get("importance", 0.5)),
            )
            return {"id": record_id}

        @self.route("POST", "/memory/recall")
        def recall(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            result = context.memory.recall(
                str(body.get("query") or ""),
                limit=int(body.get("limit", 8)),
                kind=str(body.get("kind") or "") or None,
            )
            return result.to_dict()

        @self.route("GET", "/memory/stats")
        def memory_stats(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return context.memory.stats_snapshot()

        @self.route("POST", "/agents/run")
        def run_agent(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..agents.orchestrator import MasterOrchestrator

            goal = str(body.get("goal") or "")
            if not goal:
                return {"ok": False, "error": "goal is required"}
            orchestrator = MasterOrchestrator(context, max_steps=int(body.get("max_steps", 8)))
            result = orchestrator.run(goal, reflect=bool(body.get("reflect", False)))
            return result.to_dict()

        @self.route("POST", "/backup")
        def backup(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..storage.backup import BackupManager

            manager = BackupManager(context.db, context.settings.backup_dir)
            info = manager.create(label=str(body.get("label") or "api"))
            manager.rotate()
            return info.to_dict()

        @self.route("GET", "/events")
        def events(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            bus = getattr(context, "bus", None)
            return {"events": bus.snapshot() if bus is not None else []}


def _make_handler(server: APIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"NoMoralsCore/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # route through our logger
            _log.debug("api %s - %s", self.address_string(), fmt % args)

        def _authorized(self) -> bool:
            if not server.token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {server.token}"

        def _respond(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            if not _ROUTE.match(parsed.path):
                self._respond(400, {"error": "malformed path"})
                return
            if not self._authorized():
                self._respond(401, {"error": "missing or invalid bearer token"})
                return
            body: dict[str, Any] = {}
            if method != "GET":
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    try:
                        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    except json.JSONDecodeError:
                        self._respond(400, {"error": "body is not valid JSON"})
                        return
                    if not isinstance(body, dict):
                        self._respond(400, {"error": "body must be a JSON object"})
                        return
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, payload = server.dispatch(method, parsed.path, body, query)
            self._respond(status, payload)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

    return Handler


def serve(
    context: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str = "",
    background: bool = False,
) -> int:
    """Start the API. Returns 0 on clean shutdown."""
    api = APIServer(context, token=token or context.settings.api.token)
    httpd = ThreadingHTTPServer((host, port), _make_handler(api))
    httpd.daemon_threads = True
    _log.info("api listening on http://%s:%s", host, port)
    if not token and not context.settings.api.token:
        _log.warning("api has no auth token configured; do not expose this port")
    if background:
        thread = threading.Thread(target=httpd.serve_forever, name="api", daemon=True)
        thread.start()
        return 0
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E103, E106 - deliberate shutdown hook
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
